"""Tests for llm_ocr module — data classes, compute_word_confidence, build_recognition_prompt.

All tests are self-contained with no external API calls.
"""

from __future__ import annotations

import math
from dataclasses import fields as dataclass_fields
from unittest.mock import MagicMock, patch

import pytest

from forma.llm_ocr import (
    LLMVisionResponse,
    TokenUsage,
    WordConfidence,
    build_recognition_prompt,
    compute_word_confidence,
    extract_text_via_llm,
    validate_llm_recognition,
)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestWordConfidence:
    """Tests for WordConfidence dataclass."""

    def test_fields(self):
        field_names = {f.name for f in dataclass_fields(WordConfidence)}
        assert field_names == {"word", "confidence", "token_count"}

    def test_construction(self):
        wc = WordConfidence(word="세포막", confidence=0.95, token_count=2)
        assert wc.word == "세포막"
        assert wc.confidence == 0.95
        assert wc.token_count == 2


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_fields(self):
        field_names = {f.name for f in dataclass_fields(TokenUsage)}
        assert field_names == {"input_tokens", "output_tokens"}

    def test_construction(self):
        tu = TokenUsage(input_tokens=100, output_tokens=20)
        assert tu.input_tokens == 100
        assert tu.output_tokens == 20


class TestLLMVisionResponse:
    """Tests for LLMVisionResponse dataclass."""

    def test_fields(self):
        field_names = {f.name for f in dataclass_fields(LLMVisionResponse)}
        assert field_names == {
            "text",
            "word_confidences",
            "confidence_mean",
            "confidence_min",
            "usage",
            "finish_reason",
            "logprobs_raw",
            "safety_ratings",
        }

    def test_construction_full(self):
        wc = WordConfidence(word="hello", confidence=0.9, token_count=1)
        resp = LLMVisionResponse(
            text="hello",
            word_confidences=[wc],
            confidence_mean=0.9,
            confidence_min=0.9,
            usage=TokenUsage(input_tokens=50, output_tokens=5),
            finish_reason="STOP",
            logprobs_raw=[{"token": "hello", "log_probability": -0.1}],
            safety_ratings=[{"category": "SAFE"}],
        )
        assert resp.text == "hello"
        assert len(resp.word_confidences) == 1
        assert resp.confidence_mean == 0.9
        assert resp.usage.input_tokens == 50

    def test_construction_minimal(self):
        """All optional fields can be None."""
        resp = LLMVisionResponse(
            text="",
            word_confidences=None,
            confidence_mean=None,
            confidence_min=None,
            usage=TokenUsage(input_tokens=10, output_tokens=0),
            finish_reason="STOP",
            logprobs_raw=None,
            safety_ratings=None,
        )
        assert resp.text == ""
        assert resp.word_confidences is None
        assert resp.confidence_mean is None


# ---------------------------------------------------------------------------
# compute_word_confidence() tests
# ---------------------------------------------------------------------------


class TestComputeWordConfidence:
    """Tests for compute_word_confidence() — token-to-word mapping."""

    def _make_logprobs_result(self, tokens: list[dict]) -> MagicMock:
        """Build a mock logprobs_result with chosen_candidates.

        Each token dict should have 'token' (str) and 'log_probability' (float).
        """
        chosen = []
        for t in tokens:
            candidate = MagicMock()
            candidate.token = t["token"]
            candidate.log_probability = t["log_probability"]
            chosen.append(candidate)
        result = MagicMock()
        result.chosen_candidates = chosen
        return result

    def test_single_token_word(self):
        """A word that maps to exactly one token."""
        logprobs = self._make_logprobs_result([
            {"token": "hello", "log_probability": math.log(0.9)},
        ])
        result = compute_word_confidence(logprobs, "hello")
        assert len(result) == 1
        assert result[0].word == "hello"
        assert result[0].token_count == 1
        assert abs(result[0].confidence - 0.9) < 1e-6

    def test_multi_token_word_geometric_mean(self):
        """A word split into multiple tokens uses geometric mean."""
        # "세포막은" split into 세포 + 막 + 은
        # geometric mean of exp(log(0.8)), exp(log(0.9)), exp(log(0.7))
        # = (0.8 * 0.9 * 0.7)^(1/3)
        p1, p2, p3 = 0.8, 0.9, 0.7
        expected = (p1 * p2 * p3) ** (1.0 / 3.0)
        logprobs = self._make_logprobs_result([
            {"token": "세포", "log_probability": math.log(p1)},
            {"token": "막", "log_probability": math.log(p2)},
            {"token": "은", "log_probability": math.log(p3)},
        ])
        result = compute_word_confidence(logprobs, "세포막은")
        assert len(result) == 1
        assert result[0].word == "세포막은"
        assert result[0].token_count == 3
        assert abs(result[0].confidence - expected) < 1e-6

    def test_multiple_words(self):
        """Multiple space-separated words each get their own WordConfidence."""
        logprobs = self._make_logprobs_result([
            {"token": "hello", "log_probability": math.log(0.9)},
            {"token": " ", "log_probability": math.log(0.99)},
            {"token": "world", "log_probability": math.log(0.8)},
        ])
        result = compute_word_confidence(logprobs, "hello world")
        assert len(result) == 2
        assert result[0].word == "hello"
        assert result[1].word == "world"

    def test_empty_text_returns_empty(self):
        """Empty text returns empty list."""
        logprobs = self._make_logprobs_result([])
        result = compute_word_confidence(logprobs, "")
        assert result == []

    def test_none_logprobs_returns_empty(self):
        """None logprobs_result returns empty list."""
        result = compute_word_confidence(None, "some text")
        assert result == []

    def test_confidence_clamped_0_to_1(self):
        """Confidence values are clamped to [0.0, 1.0]."""
        # log_probability = 0.0 means exp(0) = 1.0 (max)
        logprobs = self._make_logprobs_result([
            {"token": "hi", "log_probability": 0.0},
        ])
        result = compute_word_confidence(logprobs, "hi")
        assert result[0].confidence <= 1.0
        assert result[0].confidence >= 0.0


# ---------------------------------------------------------------------------
# build_recognition_prompt() tests
# ---------------------------------------------------------------------------


class TestBuildRecognitionPrompt:
    """Tests for build_recognition_prompt() — structured Korean prompt."""

    def test_no_context(self):
        """Prompt generated without context still contains core instruction."""
        prompt = build_recognition_prompt(context=None)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should contain a Korean instruction about text recognition
        assert "텍스트" in prompt or "인식" in prompt or "읽" in prompt

    def test_full_context(self):
        """All context fields are included in the prompt."""
        ctx = {
            "subject": "생물학",
            "question": "세포막의 선택적 투과성을 설명하시오.",
            "answer_keywords": "인지질, 이중층, 단백질",
        }
        prompt = build_recognition_prompt(context=ctx)
        assert "생물학" in prompt
        assert "세포막" in prompt
        assert "인지질" in prompt

    def test_partial_context(self):
        """Only provided context fields appear in the prompt."""
        ctx = {"subject": "화학"}
        prompt = build_recognition_prompt(context=ctx)
        assert "화학" in prompt
        # question/keywords not provided — should not crash
        assert isinstance(prompt, str)

    def test_empty_context(self):
        """Empty dict treated like no context."""
        prompt = build_recognition_prompt(context={})
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# extract_text_via_llm() tests
# ---------------------------------------------------------------------------


class TestExtractTextViaLlm:
    """Tests for extract_text_via_llm() — LLM Vision batch processing."""

    def _make_full_response(
        self,
        text="세포막은",
        finish_reason="STOP",
        input_tokens=100,
        output_tokens=10,
        logprobs_result=None,
        safety_ratings=None,
    ):
        """Build a mock LLMFullResponse."""
        from forma.llm_provider import LLMFullResponse

        return LLMFullResponse(
            text=text,
            logprobs_result=logprobs_result,
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
            finish_reason=finish_reason,
            safety_ratings=safety_ratings,
        )

    def test_single_image(self, tmp_path):
        """Single image returns dict with one entry."""
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        mock_provider = MagicMock()
        mock_provider.generate_with_image_full.return_value = self._make_full_response()
        mock_provider.model_name = "gemini-2.5-flash"

        with patch("forma.llm_provider.create_provider", return_value=mock_provider):
            result = extract_text_via_llm(
                image_paths=[str(img)],
                provider="gemini",
                api_key="fake-key",
            )

        assert len(result) == 1
        key = str(img)
        assert key in result
        resp = result[key]
        assert isinstance(resp, LLMVisionResponse)
        assert resp.text == "세포막은"
        assert resp.finish_reason == "STOP"
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 10

    def test_multiple_images(self, tmp_path):
        """Multiple images each get their own response."""
        imgs = []
        for i in range(3):
            img = tmp_path / f"test_{i}.jpg"
            img.write_bytes(b"\xff\xd8\xff\xe0fake")
            imgs.append(str(img))

        mock_provider = MagicMock()
        mock_provider.generate_with_image_full.side_effect = [
            self._make_full_response(text=f"text_{i}") for i in range(3)
        ]
        mock_provider.model_name = "gemini-2.5-flash"

        with patch("forma.llm_provider.create_provider", return_value=mock_provider):
            result = extract_text_via_llm(
                image_paths=imgs,
                provider="gemini",
                api_key="fake-key",
            )

        assert len(result) == 3
        for i, path in enumerate(imgs):
            assert result[path].text == f"text_{i}"

    def test_with_logprobs(self, tmp_path):
        """When logprobs available, word_confidences are computed."""
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        # Build mock logprobs
        logprobs = MagicMock()
        candidate = MagicMock()
        candidate.token = "hello"
        candidate.log_probability = math.log(0.9)
        logprobs.chosen_candidates = [candidate]

        mock_provider = MagicMock()
        mock_provider.generate_with_image_full.return_value = self._make_full_response(
            text="hello", logprobs_result=logprobs,
        )
        mock_provider.model_name = "gemini-2.5-flash"

        with patch("forma.llm_provider.create_provider", return_value=mock_provider):
            result = extract_text_via_llm(
                image_paths=[str(img)],
                provider="gemini",
                api_key="fake-key",
            )

        resp = result[str(img)]
        assert resp.word_confidences is not None
        assert len(resp.word_confidences) == 1
        assert resp.confidence_mean is not None
        assert abs(resp.confidence_mean - 0.9) < 1e-6

    def test_failed_image_returns_empty_text(self, tmp_path):
        """Failed image recognition returns empty text with warning."""
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        mock_provider = MagicMock()
        mock_provider.generate_with_image_full.side_effect = Exception("API error")
        mock_provider.model_name = "gemini-2.5-flash"

        with patch("forma.llm_provider.create_provider", return_value=mock_provider):
            result = extract_text_via_llm(
                image_paths=[str(img)],
                provider="gemini",
                api_key="fake-key",
            )

        assert len(result) == 1
        resp = result[str(img)]
        assert resp.text == ""
        assert resp.word_confidences is None
        assert resp.confidence_mean is None

    def test_rate_limit_delay(self, tmp_path):
        """Rate limit delay is applied between images."""
        imgs = []
        for i in range(2):
            img = tmp_path / f"test_{i}.jpg"
            img.write_bytes(b"\xff\xd8\xff\xe0fake")
            imgs.append(str(img))

        mock_provider = MagicMock()
        mock_provider.generate_with_image_full.return_value = self._make_full_response()
        mock_provider.model_name = "gemini-2.5-flash"

        with (
            patch("forma.llm_provider.create_provider", return_value=mock_provider),
            patch("forma.llm_ocr.time.sleep") as mock_sleep,
        ):
            extract_text_via_llm(
                image_paths=imgs,
                provider="gemini",
                api_key="fake-key",
                rate_limit_delay=2.0,
            )

        # Sleep called between images (not before first)
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(2.0)


# ---------------------------------------------------------------------------
# validate_llm_recognition() tests
# ---------------------------------------------------------------------------


class TestValidateLlmRecognition:
    """Tests for validate_llm_recognition() — response validation."""

    def test_valid_response(self):
        """Normal response passes validation."""
        result = validate_llm_recognition(
            text="세포막은 선택적 투과성을 가진다",
            finish_reason="STOP",
            confidence_mean=0.85,
        )
        assert result["valid"] is True
        assert result["warnings"] == []

    def test_empty_text_invalid(self):
        """Empty text is invalid."""
        result = validate_llm_recognition(
            text="",
            finish_reason="STOP",
            confidence_mean=None,
        )
        assert result["valid"] is False
        assert any("빈" in w or "empty" in w.lower() for w in result["warnings"])

    def test_max_tokens_finish_reason_invalid(self):
        """finish_reason != STOP triggers invalid."""
        result = validate_llm_recognition(
            text="some text",
            finish_reason="MAX_TOKENS",
            confidence_mean=0.8,
        )
        assert result["valid"] is False
        assert any("MAX_TOKENS" in w or "finish" in w.lower() for w in result["warnings"])

    def test_long_text_hallucination_warning(self):
        """Text > 200 chars triggers hallucination warning."""
        long_text = "가" * 201
        result = validate_llm_recognition(
            text=long_text,
            finish_reason="STOP",
            confidence_mean=0.9,
        )
        # Still valid but with warning
        assert any("환각" in w or "hallucin" in w.lower() or "200" in w for w in result["warnings"])

    def test_low_confidence_warning(self):
        """confidence_mean < 0.3 triggers manual review warning."""
        result = validate_llm_recognition(
            text="blurry text",
            finish_reason="STOP",
            confidence_mean=0.2,
        )
        assert any("0.3" in w or "수동" in w or "review" in w.lower() for w in result["warnings"])

    def test_none_confidence_no_warning(self):
        """None confidence (Anthropic) does not trigger warning."""
        result = validate_llm_recognition(
            text="some text",
            finish_reason="STOP",
            confidence_mean=None,
        )
        assert result["valid"] is True
        # Should NOT have confidence warning
        assert not any("0.3" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# extract_text_via_llm() fallback chain tests
# ---------------------------------------------------------------------------


class TestExtractTextFallbackChain:
    """Tests for retry/fallback behavior in extract_text_via_llm."""

    def _make_full_response(self, text="ok", finish_reason="STOP", **kw):
        from forma.llm_provider import LLMFullResponse
        return LLMFullResponse(
            text=text,
            logprobs_result=kw.get("logprobs_result"),
            usage=kw.get("usage", {"input_tokens": 10, "output_tokens": 5}),
            finish_reason=finish_reason,
            safety_ratings=None,
        )

    def test_retry_on_empty_text(self, tmp_path):
        """Empty text response triggers retry with temperature=0.1."""
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        mock_provider = MagicMock()
        # 1st call returns empty, 2nd returns valid
        mock_provider.generate_with_image_full.side_effect = [
            self._make_full_response(text=""),
            self._make_full_response(text="retried text"),
        ]
        mock_provider.model_name = "gemini-2.5-flash"

        with patch("forma.llm_provider.create_provider", return_value=mock_provider):
            result = extract_text_via_llm(
                image_paths=[str(img)],
                provider="gemini",
                api_key="fake-key",
            )

        resp = result[str(img)]
        assert resp.text == "retried text"
        assert mock_provider.generate_with_image_full.call_count == 2

    def test_retry_on_max_tokens(self, tmp_path):
        """MAX_TOKENS finish_reason triggers retry."""
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        mock_provider = MagicMock()
        mock_provider.generate_with_image_full.side_effect = [
            self._make_full_response(text="partial", finish_reason="MAX_TOKENS"),
            self._make_full_response(text="complete text"),
        ]
        mock_provider.model_name = "gemini-2.5-flash"

        with patch("forma.llm_provider.create_provider", return_value=mock_provider):
            result = extract_text_via_llm(
                image_paths=[str(img)],
                provider="gemini",
                api_key="fake-key",
            )

        resp = result[str(img)]
        assert resp.text == "complete text"

    def test_both_attempts_fail_returns_empty(self, tmp_path):
        """Both 1st and 2nd attempts fail → empty text + warnings in finish_reason."""
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        mock_provider = MagicMock()
        mock_provider.generate_with_image_full.side_effect = [
            self._make_full_response(text=""),
            self._make_full_response(text=""),
        ]
        mock_provider.model_name = "gemini-2.5-flash"

        with patch("forma.llm_provider.create_provider", return_value=mock_provider):
            result = extract_text_via_llm(
                image_paths=[str(img)],
                provider="gemini",
                api_key="fake-key",
            )

        resp = result[str(img)]
        assert resp.text == ""
        assert mock_provider.generate_with_image_full.call_count == 2
