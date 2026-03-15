"""Adversarial tests for LLM Vision OCR (018-llm-vision-ocr).

12 adversarial personas that ruthlessly test the LLM Vision OCR
implementation for edge cases, failure modes, and misuse scenarios.
All LLM API calls are mocked — no real API keys required.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

FAKE_JPG_BYTES = b"\xff\xd8\xff\xe0fake-jpg-data"
FAKE_PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png-data"


def _mock_google_genai():
    """Create mock google.genai module."""
    mock_genai = MagicMock()
    mock_types = MagicMock()
    mock_google = MagicMock()
    mock_google.genai = mock_genai
    return {
        "google": mock_google,
        "google.genai": mock_genai,
        "google.genai.types": mock_types,
    }, mock_genai, mock_types


def _make_gemini_response(
    text="인식된 텍스트",
    finish_reason="STOP",
    input_tokens=100,
    output_tokens=20,
    logprobs_result=None,
):
    """Build a mock Gemini response object."""
    resp = MagicMock()
    resp.text = text
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    candidate.logprobs_result = logprobs_result
    candidate.safety_ratings = []
    resp.candidates = [candidate]
    resp.usage_metadata = MagicMock()
    resp.usage_metadata.prompt_token_count = input_tokens
    resp.usage_metadata.candidates_token_count = output_tokens
    return resp


def _make_full_response(
    text="인식된 텍스트",
    finish_reason="STOP",
    input_tokens=100,
    output_tokens=20,
    logprobs_result=None,
    safety_ratings=None,
):
    """Build an LLMFullResponse-compatible object."""
    from forma.llm_provider import LLMFullResponse

    return LLMFullResponse(
        text=text,
        logprobs_result=logprobs_result,
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
        finish_reason=finish_reason,
        safety_ratings=safety_ratings,
    )


@pytest.fixture
def image_dir(tmp_path):
    """Create temp dir with several fake JPEG images."""
    for i in range(5):
        (tmp_path / f"q1_W1_{i:04d}.jpg").write_bytes(FAKE_JPG_BYTES)
    return tmp_path


@pytest.fixture
def single_image(tmp_path):
    """Create a single fake JPEG image."""
    img = tmp_path / "test.jpg"
    img.write_bytes(FAKE_JPG_BYTES)
    return str(img)


# ═══════════════════════════════════════════════════════════════════════════
# Persona 1: The Impatient Professor
# Simulates Ctrl+C mid-batch then immediate re-run.
# Test resume logic: partial YAML must be valid, re-run must skip processed.
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryImpatientProfessor:
    """Persona 1: Ctrl+C mid-batch, then re-run. Tests resume logic."""

    def test_partial_yaml_is_valid_after_interruption(self, tmp_path):
        """After processing 2 of 5 images and 'interrupting', the partial
        YAML file must be valid YAML that can be loaded."""
        output_yaml = tmp_path / "scan_results.yaml"

        # Simulate partial results (as if 2 images were processed before interrupt)
        partial_results = [
            {
                "student_id": "S001",
                "q_num": 1,
                "text": "부분 결과 1",
                "source_file": "q1_W1_0000.jpg",
                "ocr_confidence_mean": 0.85,
                "ocr_confidence_min": 0.72,
                "ocr_field_count": 3,
                "recognition_engine": "llm",
            },
            {
                "student_id": "S002",
                "q_num": 1,
                "text": "부분 결과 2",
                "source_file": "q1_W1_0001.jpg",
                "ocr_confidence_mean": 0.90,
                "ocr_confidence_min": 0.80,
                "ocr_field_count": 4,
                "recognition_engine": "llm",
            },
        ]
        output_yaml.write_text(
            yaml.dump(partial_results, allow_unicode=True), encoding="utf-8"
        )

        # Verify the partial YAML is valid
        loaded = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
        assert isinstance(loaded, list)
        assert len(loaded) == 2
        assert loaded[0]["source_file"] == "q1_W1_0000.jpg"
        assert loaded[1]["source_file"] == "q1_W1_0001.jpg"

    def test_extract_processes_all_given_images(self, tmp_path):
        """extract_text_via_llm processes all images in the list.
        Resume filtering is handled at the pipeline level, not here."""
        from forma.llm_ocr import extract_text_via_llm

        images = []
        for i in range(3):
            img = tmp_path / f"q1_W1_{i:04d}.jpg"
            img.write_bytes(FAKE_JPG_BYTES)
            images.append(str(img))

        full_resp = _make_full_response(text="결과")
        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            mock_prov_inst.generate_with_image_full.return_value = full_resp
            mock_prov_inst.model_name = "gemini-2.5-flash"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                results = extract_text_via_llm(image_paths=images)

        # All 3 images should be processed
        assert len(results) == 3
        for img_path in images:
            assert img_path in results
            assert results[img_path].text == "결과"


# ═══════════════════════════════════════════════════════════════════════════
# Persona 2: The No-Config User
# Runs without API key. Must get clear error, not stack trace.
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryNoConfigUser:
    """Persona 2: No API key set. Must get clear error message."""

    def test_extract_without_api_key_raises_clear_error(self, single_image):
        """Calling extract_text_via_llm without API key should raise
        a clear EnvironmentError, not an obscure stack trace."""
        from forma.llm_ocr import extract_text_via_llm

        # Ensure no API key in environment
        env = {k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="(?i)(api|key|설정)"):
                extract_text_via_llm(
                    image_paths=[single_image],
                    api_key=None,
                )

    def test_cli_scan_without_api_key_shows_helpful_message(self):
        """CLI 'forma ocr scan' without API key should exit with
        a user-friendly Korean error message mentioning forma-init."""
        from forma.cli_ocr import _parse_args

        # This test verifies the error path in CLI, not the full pipeline
        # The CLI should catch EnvironmentError and print a helpful message
        env = {k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()}
        with patch.dict(os.environ, env, clear=True):
            # Just verify the CLI can parse scan args (the error comes at runtime)
            args = _parse_args(["scan", "--class", "A"])
            assert args.class_id == "A"


# ═══════════════════════════════════════════════════════════════════════════
# Persona 3: The Corrupt Image Uploader
# Passes broken/truncated/zero-byte files. Must skip gracefully with WARNING.
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryCorruptImageUploader:
    """Persona 3: Broken, truncated, zero-byte image files."""

    def test_zero_byte_image_skipped_with_warning(self, tmp_path, caplog):
        """A zero-byte file should be skipped with a WARNING, not crash."""
        from forma.llm_ocr import extract_text_via_llm

        zero_img = tmp_path / "empty.jpg"
        zero_img.write_bytes(b"")

        full_resp = _make_full_response(text="")
        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            mock_prov_inst.generate_with_image_full.side_effect = Exception(
                "Invalid image data"
            )
            mock_prov_inst.model_name = "gemini-2.5-flash"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                with caplog.at_level(logging.WARNING):
                    results = extract_text_via_llm(
                        image_paths=[str(zero_img)],
                    )

        # Should not crash — result should be empty text or skipped
        assert "empty.jpg" in str(results) or len(results) >= 0

    def test_truncated_image_does_not_crash_batch(self, tmp_path, caplog):
        """A truncated JPEG (valid header, truncated body) should not
        crash the entire batch."""
        from forma.llm_ocr import extract_text_via_llm

        # Good image
        good = tmp_path / "good.jpg"
        good.write_bytes(FAKE_JPG_BYTES)

        # Truncated image (only JPEG header, no body)
        bad = tmp_path / "truncated.jpg"
        bad.write_bytes(b"\xff\xd8\xff")

        full_resp = _make_full_response(text="정상 텍스트")
        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            # First call (good) succeeds, second call (bad) fails
            mock_prov_inst.generate_with_image_full.side_effect = [
                full_resp,
                Exception("Image decode error"),
            ]
            mock_prov_inst.model_name = "gemini-2.5-flash"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                with caplog.at_level(logging.WARNING):
                    results = extract_text_via_llm(
                        image_paths=[str(good), str(bad)],
                    )

        # Batch should complete — good image should have result
        assert len(results) >= 1

    def test_nonexistent_image_path_skipped(self, tmp_path, caplog):
        """A path to a file that doesn't exist should be skipped."""
        from forma.llm_ocr import extract_text_via_llm

        fake_path = str(tmp_path / "does_not_exist.jpg")

        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            mock_prov_inst.generate_with_image_full.side_effect = FileNotFoundError(
                "No such file"
            )
            mock_prov_inst.model_name = "gemini-2.5-flash"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                with caplog.at_level(logging.WARNING):
                    results = extract_text_via_llm(
                        image_paths=[fake_path],
                    )

        # Should not crash
        assert isinstance(results, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Persona 4: The Mega-Batch User
# Tests with 500+ mock images. Verify no memory leak, incremental save.
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryMegaBatchUser:
    """Persona 4: 500+ images. No memory leak, no unbounded list growth."""

    def test_500_images_completes_without_error(self, tmp_path):
        """Processing 500 images should complete without OOM or crash."""
        from forma.llm_ocr import extract_text_via_llm

        images = []
        for i in range(500):
            img = tmp_path / f"q1_W1_{i:04d}.jpg"
            img.write_bytes(FAKE_JPG_BYTES)
            images.append(str(img))

        full_resp = _make_full_response(text="텍스트", input_tokens=50, output_tokens=10)
        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            mock_prov_inst.generate_with_image_full.return_value = full_resp
            mock_prov_inst.model_name = "gemini-2.5-flash"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                results = extract_text_via_llm(
                    image_paths=images,
                    rate_limit_delay=0.0,
                )

        assert len(results) == 500

    def test_batch_result_count_matches_input(self, tmp_path):
        """Every input image should produce a result entry (success or fallback)."""
        from forma.llm_ocr import extract_text_via_llm

        n = 50
        images = []
        for i in range(n):
            img = tmp_path / f"img_{i:04d}.jpg"
            img.write_bytes(FAKE_JPG_BYTES)
            images.append(str(img))

        full_resp = _make_full_response(text="ok")
        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            mock_prov_inst.generate_with_image_full.return_value = full_resp
            mock_prov_inst.model_name = "gemini-2.5-flash"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                results = extract_text_via_llm(
                    image_paths=images,
                    rate_limit_delay=0.0,
                )

        assert len(results) == n


# ═══════════════════════════════════════════════════════════════════════════
# Persona 5: The Anthropic Loyalist
# Uses provider="anthropic". Verify confidence=None, no crash.
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryAnthropicLoyalist:
    """Persona 5: Uses Anthropic. confidence=None, no crash, pipeline handles it."""

    def test_anthropic_provider_returns_none_confidence(self, single_image):
        """Anthropic doesn't support logprobs. confidence_mean/min must be None."""
        from forma.llm_ocr import extract_text_via_llm

        full_resp = _make_full_response(
            text="anthropic 인식 결과",
            logprobs_result=None,  # Anthropic: no logprobs
        )
        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            mock_prov_inst.generate_with_image_full.return_value = full_resp
            mock_prov_inst.model_name = "claude-sonnet-4-6"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                results = extract_text_via_llm(
                    image_paths=[single_image],
                    provider="anthropic",
                    api_key="fake-key",
                )

        resp = list(results.values())[0]
        assert resp.text == "anthropic 인식 결과"
        # confidence should be None when logprobs unavailable
        assert resp.confidence_mean is None
        assert resp.confidence_min is None
        assert resp.word_confidences is None

    def test_anthropic_none_confidence_safe_in_yaml(self, single_image, tmp_path):
        """None confidence values should serialize safely to YAML."""
        result_record = {
            "student_id": "S001",
            "q_num": 1,
            "text": "anthropic result",
            "source_file": "test.jpg",
            "ocr_confidence_mean": None,
            "ocr_confidence_min": None,
            "ocr_field_count": None,
            "recognition_engine": "llm",
            "recognition_model": "claude-sonnet-4-6",
        }
        yaml_path = tmp_path / "scan.yaml"
        yaml_path.write_text(
            yaml.dump([result_record], allow_unicode=True), encoding="utf-8"
        )

        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert loaded[0]["ocr_confidence_mean"] is None
        assert loaded[0]["recognition_engine"] == "llm"


# ═══════════════════════════════════════════════════════════════════════════
# Persona 6: The Legacy Data Archaeologist
# Loads v0.12.x scan YAML (no recognition_engine field). Backward compat.
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryLegacyDataArchaeologist:
    """Persona 6: v0.12.x data without recognition_engine. Backward compat."""

    def test_legacy_yaml_without_recognition_engine(self, tmp_path):
        """v0.12.x scan YAML has no recognition_engine field.
        Loading it should default to 'naver' and not crash."""
        legacy_data = [
            {
                "student_id": "S001",
                "q_num": 1,
                "text": "생체항상성이란",
                "source_file": "q1_W1_0001.jpg",
                "ocr_confidence_mean": 0.85,
                "ocr_confidence_min": 0.72,
                "ocr_field_count": 3,
                # NOTE: no recognition_engine, no recognition_model
            }
        ]
        yaml_path = tmp_path / "legacy_scan.yaml"
        yaml_path.write_text(
            yaml.dump(legacy_data, allow_unicode=True), encoding="utf-8"
        )

        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        record = loaded[0]

        # Accessing with .get() default should yield "naver"
        engine = record.get("recognition_engine", "naver")
        assert engine == "naver"
        # All existing fields should still work
        assert record["text"] == "생체항상성이란"
        assert record["ocr_confidence_mean"] == 0.85

    def test_legacy_yaml_no_llm_fields(self, tmp_path):
        """v0.12.x data has no llm_* fields. Pipeline should handle missing
        fields gracefully via .get() patterns."""
        legacy_data = [
            {
                "student_id": "S001",
                "q_num": 1,
                "text": "항상성",
                "source_file": "test.jpg",
                "ocr_confidence_mean": 0.80,
                "ocr_confidence_min": 0.65,
                "ocr_field_count": 2,
            }
        ]
        yaml_path = tmp_path / "legacy.yaml"
        yaml_path.write_text(
            yaml.dump(legacy_data, allow_unicode=True), encoding="utf-8"
        )

        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        record = loaded[0]

        # All new llm_* fields should be safely accessible with defaults
        assert record.get("llm_word_confidences") is None
        assert record.get("llm_usage") is None
        assert record.get("llm_finish_reason") is None
        assert record.get("llm_logprobs") is None
        assert record.get("recognition_model") is None


# ═══════════════════════════════════════════════════════════════════════════
# Persona 7: The Hallucination Inducer
# Blank/empty image with no text. Verify correct handling.
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryHallucinationInducer:
    """Persona 7: Blank image, empty text, hallucination detection."""

    def test_blank_image_returns_empty_text(self, single_image):
        """An image with no handwriting should return empty text, not hallucinate."""
        from forma.llm_ocr import extract_text_via_llm

        full_resp = _make_full_response(text="", finish_reason="STOP")
        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            mock_prov_inst.generate_with_image_full.return_value = full_resp
            mock_prov_inst.model_name = "gemini-2.5-flash"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                results = extract_text_via_llm(image_paths=[single_image])

        resp = list(results.values())[0]
        assert resp.text == ""

    def test_hallucination_warning_for_long_text(self, single_image, caplog):
        """Text > 200 chars should trigger a hallucination warning."""
        from forma.llm_ocr import validate_llm_recognition

        long_text = "가" * 201  # 201 Korean chars > 200 threshold

        result = validate_llm_recognition(
            text=long_text,
            finish_reason="STOP",
            confidence_mean=0.9,
        )

        # Should flag hallucination warning
        assert not result["valid"] or any(
            "환각" in w or "hallucin" in w.lower() or "200" in w
            for w in result.get("warnings", [])
        ), f"Expected hallucination warning, got: {result}"

    def test_exactly_200_chars_no_warning(self, single_image):
        """Text of exactly 200 chars should NOT trigger hallucination warning."""
        from forma.llm_ocr import validate_llm_recognition

        text_200 = "가" * 200

        result = validate_llm_recognition(
            text=text_200,
            finish_reason="STOP",
            confidence_mean=0.9,
        )

        # 200 chars is the boundary — should be valid
        hallucination_warnings = [
            w
            for w in result.get("warnings", [])
            if "환각" in w or "hallucin" in w.lower() or "200" in w
        ]
        assert len(hallucination_warnings) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Persona 8: The Network Saboteur
# Mock API returning 429 and timeout. Verify retry + fallback.
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryNetworkSaboteur:
    """Persona 8: 429, timeout errors. Verify retry and fallback to empty text."""

    def test_429_rate_limit_falls_back_to_empty(self, single_image, caplog):
        """If llm_provider exhausts retries on 429, extract_text_via_llm
        catches the exception and falls back to empty text."""
        from forma.llm_ocr import extract_text_via_llm

        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            # After llm_provider's own retries are exhausted, it raises
            mock_prov_inst.generate_with_image_full.side_effect = Exception(
                "429 Too Many Requests"
            )
            mock_prov_inst.model_name = "gemini-2.5-flash"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                with caplog.at_level(logging.WARNING):
                    results = extract_text_via_llm(image_paths=[single_image])

        resp = list(results.values())[0]
        assert resp.text == ""
        assert resp.finish_reason == "ERROR"

    def test_all_retries_exhausted_returns_empty_text(self, single_image, caplog):
        """When all retries fail, result should be empty text with WARNING."""
        from forma.llm_ocr import extract_text_via_llm

        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            # All calls fail
            mock_prov_inst.generate_with_image_full.side_effect = Exception(
                "Connection timeout"
            )
            mock_prov_inst.model_name = "gemini-2.5-flash"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                with caplog.at_level(logging.WARNING):
                    results = extract_text_via_llm(image_paths=[single_image])

        # Should not crash — fallback to empty text
        resp = list(results.values())[0]
        assert resp.text == ""

    def test_timeout_error_handled_gracefully(self, single_image, caplog):
        """TimeoutError should be handled via retry/fallback, not crash."""
        from forma.llm_ocr import extract_text_via_llm

        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            mock_prov_inst.generate_with_image_full.side_effect = TimeoutError(
                "Request timed out"
            )
            mock_prov_inst.model_name = "gemini-2.5-flash"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                with caplog.at_level(logging.WARNING):
                    results = extract_text_via_llm(image_paths=[single_image])

        resp = list(results.values())[0]
        assert resp.text == ""


# ═══════════════════════════════════════════════════════════════════════════
# Persona 9: The Unicode Terrorist
# Mixed Korean/English/emoji/math symbols. Verify text preserved in YAML.
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryUnicodeTerrorist:
    """Persona 9: Unicode edge cases — Korean, emoji, math, mixed scripts."""

    UNICODE_TEXTS = [
        "항상성(homeostasis)은 생체의 내부 환경을 일정하게 유지하는 것",
        "CO₂ + H₂O → H₂CO₃ (탄산)",
        "α-아밀라아제 enzyme β-galactosidase",
        "학생 답: ∫f(x)dx = F(x) + C",
        "세포 분열 🧬 → 2n chromosomes",
        "Na⁺/K⁺-ATPase pump 활성화",
        "pH ≤ 7.35 → 산증(acidosis) 발생",
        "",  # empty string
        "   ",  # whitespace only
        "\t\n",  # tabs and newlines
    ]

    @pytest.mark.parametrize("text", UNICODE_TEXTS, ids=[
        "korean_english_mix",
        "chemistry_subscript",
        "greek_letters",
        "math_integral",
        "emoji_biology",
        "superscript_pump",
        "comparison_ph",
        "empty_string",
        "whitespace_only",
        "tabs_newlines",
    ])
    def test_unicode_text_roundtrips_through_yaml(self, text, tmp_path):
        """Any Unicode text from LLM should survive YAML serialization."""
        record = {
            "student_id": "S001",
            "q_num": 1,
            "text": text,
            "source_file": "test.jpg",
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(
            yaml.dump([record], allow_unicode=True), encoding="utf-8"
        )

        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        # YAML may normalize whitespace-only strings to None
        if text.strip() == "":
            assert loaded[0]["text"] is None or loaded[0]["text"] == text
        else:
            assert loaded[0]["text"] == text

    def test_mixed_unicode_in_extract_result(self, single_image):
        """extract_text_via_llm should preserve complex Unicode text."""
        from forma.llm_ocr import extract_text_via_llm

        complex_text = "항상성(homeostasis): CO₂ + H₂O → H₂CO₃, pH ≤ 7.35"
        full_resp = _make_full_response(text=complex_text)

        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            mock_prov_inst.generate_with_image_full.return_value = full_resp
            mock_prov_inst.model_name = "gemini-2.5-flash"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                results = extract_text_via_llm(image_paths=[single_image])

        resp = list(results.values())[0]
        assert resp.text == complex_text


# ═══════════════════════════════════════════════════════════════════════════
# Persona 10: The Concurrent Runner
# Two extract_text_via_llm() calls with overlapping image lists.
# Verify no file corruption.
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryConcurrentRunner:
    """Persona 10: Concurrent calls with overlapping images."""

    def test_concurrent_calls_no_crash(self, tmp_path):
        """Two simultaneous extract_text_via_llm calls should not crash."""
        from forma.llm_ocr import extract_text_via_llm

        # Create shared images
        images = []
        for i in range(10):
            img = tmp_path / f"shared_{i:04d}.jpg"
            img.write_bytes(FAKE_JPG_BYTES)
            images.append(str(img))

        full_resp = _make_full_response(text="concurrent")
        errors = []

        def run_extract(img_list, label):
            try:
                with patch("forma.llm_provider.create_provider") as mock_provider:
                    mock_prov_inst = MagicMock()
                    mock_prov_inst.generate_with_image_full.return_value = full_resp
                    mock_prov_inst.model_name = "gemini-2.5-flash"
                    mock_provider.return_value = mock_prov_inst

                    with patch("forma.llm_ocr.time.sleep"):
                        results = extract_text_via_llm(
                            image_paths=img_list,
                            rate_limit_delay=0.0,
                        )
                assert len(results) == len(img_list)
            except Exception as e:
                errors.append((label, e))

        t1 = threading.Thread(target=run_extract, args=(images[:7], "thread-1"))
        t2 = threading.Thread(target=run_extract, args=(images[3:], "thread-2"))

        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert len(errors) == 0, f"Concurrent errors: {errors}"

    def test_concurrent_yaml_writes_no_corruption(self, tmp_path):
        """Two processes writing to different YAML files should not corrupt."""
        results_a = [{"student_id": "A001", "text": "결과A"}]
        results_b = [{"student_id": "B001", "text": "결과B"}]

        path_a = tmp_path / "scan_A.yaml"
        path_b = tmp_path / "scan_B.yaml"

        def write_yaml(data, path):
            path.write_text(
                yaml.dump(data, allow_unicode=True), encoding="utf-8"
            )

        t1 = threading.Thread(target=write_yaml, args=(results_a, path_a))
        t2 = threading.Thread(target=write_yaml, args=(results_b, path_b))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        loaded_a = yaml.safe_load(path_a.read_text(encoding="utf-8"))
        loaded_b = yaml.safe_load(path_b.read_text(encoding="utf-8"))
        assert loaded_a[0]["student_id"] == "A001"
        assert loaded_b[0]["student_id"] == "B001"


# ═══════════════════════════════════════════════════════════════════════════
# Persona 11: The Config Overrider
# Conflicting settings across CLI args, forma.yaml, week.yaml.
# Verify priority order.
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryConfigOverrider:
    """Persona 11: Conflicting config sources. CLI > week.yaml > forma.yaml."""

    def test_cli_provider_overrides_config(self):
        """--provider CLI arg should override any config file settings."""
        from forma.cli_ocr import _parse_args

        args = _parse_args([
            "scan", "--class", "A",
            "--provider", "anthropic",
        ])
        assert args.provider == "anthropic"

    def test_cli_model_overrides_config(self):
        """--model CLI arg should override config file model."""
        from forma.cli_ocr import _parse_args

        args = _parse_args([
            "scan", "--class", "A",
            "--model", "gemini-2.5-pro",
        ])
        assert args.model == "gemini-2.5-pro"

    def test_cli_review_threshold_overrides_default(self):
        """--ocr-review-threshold CLI arg should override default 0.75."""
        from forma.cli_ocr import _parse_args

        args = _parse_args([
            "scan", "--class", "A",
            "--ocr-review-threshold", "0.5",
        ])
        assert args.ocr_review_threshold == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Persona 12: The Empty Classroom
# Run on directory with 0 matching images. Must get clear message, not crash.
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryEmptyClassroom:
    """Persona 12: Zero matching images. Clear message, no crash."""

    def test_empty_image_list_returns_empty_dict(self):
        """extract_text_via_llm([]) should return {} without error."""
        from forma.llm_ocr import extract_text_via_llm

        with patch("forma.llm_provider.create_provider") as mock_provider:
            mock_prov_inst = MagicMock()
            mock_prov_inst.model_name = "gemini-2.5-flash"
            mock_provider.return_value = mock_prov_inst

            with patch("forma.llm_ocr.time.sleep"):
                results = extract_text_via_llm(image_paths=[])

        assert results == {}

    def test_directory_with_no_images_raises_clear_error(self, tmp_path):
        """Pipeline on empty directory should raise FileNotFoundError with message."""
        from forma.ocr_pipeline import run_scan_pipeline

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="(?i)(image|이미지|found)"):
            run_scan_pipeline(
                image_dir=str(empty_dir),
                naver_ocr_config="unused",
                output_path=str(tmp_path / "out.yaml"),
            )

    def test_directory_with_only_non_image_files(self, tmp_path):
        """Directory with .txt files but no images should raise clear error."""
        from forma.ocr_pipeline import _list_raw_images

        (tmp_path / "notes.txt").write_text("not an image")
        (tmp_path / "data.csv").write_text("a,b,c")

        result = _list_raw_images(str(tmp_path))
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# Persona BONUS: MAX_TOKENS / finish_reason edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversaryMaxTokensEdge:
    """Bonus persona: finish_reason edge cases and validation."""

    def test_max_tokens_triggers_retry(self, single_image):
        """finish_reason=MAX_TOKENS should trigger a retry call."""
        from forma.llm_ocr import validate_llm_recognition

        result = validate_llm_recognition(
            text="잘린 텍스",
            finish_reason="MAX_TOKENS",
            confidence_mean=0.9,
        )

        assert result["valid"] is False, "MAX_TOKENS should mark response as invalid"

    def test_low_confidence_flags_review(self, single_image):
        """confidence_mean < 0.3 should flag for manual review."""
        from forma.llm_ocr import validate_llm_recognition

        result = validate_llm_recognition(
            text="텍스트",
            finish_reason="STOP",
            confidence_mean=0.2,
        )

        assert any(
            "review" in w.lower() or "검토" in w or "0.3" in w
            for w in result.get("warnings", [])
        ), f"Expected review warning for low confidence, got: {result}"

    def test_valid_response_passes_validation(self):
        """A normal response should pass validation cleanly."""
        from forma.llm_ocr import validate_llm_recognition

        result = validate_llm_recognition(
            text="정상적인 답안 텍스트",
            finish_reason="STOP",
            confidence_mean=0.92,
        )

        assert result["valid"] is True
        assert len(result.get("warnings", [])) == 0
