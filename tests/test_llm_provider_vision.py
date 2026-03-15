"""Tests for LLM provider vision (generate_with_image) support.

All API calls are mocked; no real API keys required.
"""

import os
import sys
from dataclasses import fields as dataclass_fields
from unittest.mock import MagicMock, patch, mock_open

import pytest

from forma.llm_provider import (
    AnthropicProvider,
    GeminiProvider,
    LLMFullResponse,
    LLMProvider,
    MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_JPG_BYTES = b"\xff\xd8\xff\xe0fake-jpg-data"
FAKE_PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png-data"


def _mock_google_genai():
    """Create mock google.genai module and install it in sys.modules."""
    mock_genai = MagicMock()
    mock_types = MagicMock()
    mock_google = MagicMock()
    mock_google.genai = mock_genai
    return {
        "google": mock_google,
        "google.genai": mock_genai,
        "google.genai.types": mock_types,
    }, mock_genai, mock_types


# ---------------------------------------------------------------------------
# Abstract base class contract
# ---------------------------------------------------------------------------


class TestLLMProviderVisionContract:
    """Verify generate_with_image exists on the abstract base."""

    def test_generate_with_image_method_exists(self):
        """LLMProvider should have generate_with_image method."""
        assert hasattr(LLMProvider, "generate_with_image")

    def test_abstract_impl_method_exists(self):
        """LLMProvider should have _generate_with_image_impl abstract method."""
        assert hasattr(LLMProvider, "_generate_with_image_impl")


# ---------------------------------------------------------------------------
# GeminiProvider vision tests
# ---------------------------------------------------------------------------


class TestGeminiProviderVision:
    """Tests for GeminiProvider.generate_with_image()."""

    def test_generate_with_image_jpg(self, tmp_path):
        """generate_with_image() sends image bytes to Gemini API (JPG)."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        modules, mock_genai, mock_types = _mock_google_genai()
        mock_response = MagicMock()
        mock_response.text = "I see a cat"
        mock_genai.Client.return_value.models.generate_content.return_value = (
            mock_response
        )

        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            result = provider.generate_with_image(
                prompt="What is in this image?",
                image_path=str(img_file),
            )

        assert result == "I see a cat"
        call_args = mock_genai.Client.return_value.models.generate_content.call_args
        assert call_args is not None
        # contents should be a list containing text and image part
        contents = call_args.kwargs.get("contents") or call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("contents")
        assert contents is not None

    def test_generate_with_image_png(self, tmp_path):
        """generate_with_image() works with PNG images."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(FAKE_PNG_BYTES)

        modules, mock_genai, mock_types = _mock_google_genai()
        mock_response = MagicMock()
        mock_response.text = "PNG content"
        mock_genai.Client.return_value.models.generate_content.return_value = (
            mock_response
        )

        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            result = provider.generate_with_image(
                prompt="Describe", image_path=str(img_file),
            )

        assert result == "PNG content"

    def test_generate_with_image_system_instruction(self, tmp_path):
        """system_instruction is passed through to Gemini config."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        modules, mock_genai, mock_types = _mock_google_genai()
        mock_response = MagicMock()
        mock_response.text = "response"
        mock_genai.Client.return_value.models.generate_content.return_value = (
            mock_response
        )

        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            provider.generate_with_image(
                prompt="Read text",
                image_path=str(img_file),
                system_instruction="You are an OCR assistant.",
            )

        # Verify generate_content was called with config containing system_instruction
        call_args = mock_genai.Client.return_value.models.generate_content.call_args
        assert call_args is not None

    def test_generate_with_image_unsupported_format(self, tmp_path):
        """Unsupported image format raises ValueError."""
        img_file = tmp_path / "test.bmp"
        img_file.write_bytes(b"fake-bmp")

        modules, _, _ = _mock_google_genai()
        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            with pytest.raises(ValueError, match="Unsupported image format"):
                provider.generate_with_image(
                    prompt="Read", image_path=str(img_file),
                )

    def test_generate_with_image_file_not_found(self):
        """Missing image file raises FileNotFoundError."""
        modules, _, _ = _mock_google_genai()
        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            with pytest.raises(FileNotFoundError):
                provider.generate_with_image(
                    prompt="Read", image_path="/nonexistent/path.jpg",
                )

    def test_generate_with_image_retries_on_429(self, tmp_path):
        """generate_with_image() retries on transient errors."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        modules, mock_genai, _ = _mock_google_genai()
        mock_response = MagicMock()
        mock_response.text = "ok"
        api_mock = mock_genai.Client.return_value.models
        api_mock.generate_content.side_effect = [
            Exception("429 Too Many Requests"),
            mock_response,
        ]

        with patch.dict(sys.modules, modules):
            with patch("forma.llm_provider.time.sleep"):
                provider = GeminiProvider(api_key="k")
                result = provider.generate_with_image(
                    prompt="Read", image_path=str(img_file),
                )

        assert result == "ok"
        assert api_mock.generate_content.call_count == 2


# ---------------------------------------------------------------------------
# AnthropicProvider vision tests
# ---------------------------------------------------------------------------


class TestAnthropicProviderVision:
    """Tests for AnthropicProvider.generate_with_image()."""

    def test_generate_with_image_jpg(self, tmp_path):
        """generate_with_image() sends base64 image to Anthropic API."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        mock_anthropic = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="Anthropic sees text")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="k")
            result = provider.generate_with_image(
                prompt="What text is in this image?",
                image_path=str(img_file),
            )

        assert result == "Anthropic sees text"
        # Verify API was called with image content block
        call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        # Should have image block and text block
        assert any(block["type"] == "image" for block in content)
        assert any(block["type"] == "text" for block in content)

    def test_generate_with_image_png(self, tmp_path):
        """generate_with_image() works with PNG for Anthropic."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(FAKE_PNG_BYTES)

        mock_anthropic = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="PNG text")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="k")
            result = provider.generate_with_image(
                prompt="Read", image_path=str(img_file),
            )

        assert result == "PNG text"
        call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]
        image_block = [
            b for b in messages[0]["content"] if b["type"] == "image"
        ][0]
        assert image_block["source"]["media_type"] == "image/png"

    def test_generate_with_image_system_instruction(self, tmp_path):
        """system_instruction is passed as 'system' parameter."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        mock_anthropic = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="ok")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="k")
            provider.generate_with_image(
                prompt="Read text",
                image_path=str(img_file),
                system_instruction="You are an OCR assistant.",
            )

        call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are an OCR assistant."

    def test_generate_with_image_unsupported_format(self, tmp_path):
        """Unsupported image format raises ValueError."""
        img_file = tmp_path / "test.gif"
        img_file.write_bytes(b"fake-gif")

        mock_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="k")
            with pytest.raises(ValueError, match="Unsupported image format"):
                provider.generate_with_image(
                    prompt="Read", image_path=str(img_file),
                )

    def test_generate_with_image_retries_on_429(self, tmp_path):
        """generate_with_image() retries on transient errors."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        mock_anthropic = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="ok")]
        api_mock = mock_anthropic.Anthropic.return_value.messages
        api_mock.create.side_effect = [
            Exception("429 Too Many Requests"),
            mock_msg,
        ]

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch("forma.llm_provider.time.sleep"):
                provider = AnthropicProvider(api_key="k")
                result = provider.generate_with_image(
                    prompt="Read", image_path=str(img_file),
                )

        assert result == "ok"
        assert api_mock.create.call_count == 2

    def test_generate_with_image_base64_encoding(self, tmp_path):
        """Image data is correctly base64-encoded."""
        import base64

        img_file = tmp_path / "test.jpeg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        mock_anthropic = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="ok")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="k")
            provider.generate_with_image(
                prompt="Read", image_path=str(img_file),
            )

        call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args.kwargs
        image_block = [
            b for b in call_kwargs["messages"][0]["content"]
            if b["type"] == "image"
        ][0]
        # Verify data is valid base64
        decoded = base64.b64decode(image_block["source"]["data"])
        assert decoded == FAKE_JPG_BYTES
        assert image_block["source"]["media_type"] == "image/jpeg"


# ---------------------------------------------------------------------------
# LLMFullResponse dataclass tests
# ---------------------------------------------------------------------------


class TestLLMFullResponse:
    """Tests for the LLMFullResponse dataclass."""

    def test_dataclass_has_required_fields(self):
        """LLMFullResponse must have text, logprobs_result, usage, finish_reason, safety_ratings."""
        field_names = {f.name for f in dataclass_fields(LLMFullResponse)}
        assert field_names == {
            "text",
            "logprobs_result",
            "usage",
            "finish_reason",
            "safety_ratings",
        }

    def test_construction(self):
        """LLMFullResponse can be constructed with all fields."""
        resp = LLMFullResponse(
            text="hello",
            logprobs_result=None,
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="STOP",
            safety_ratings=None,
        )
        assert resp.text == "hello"
        assert resp.logprobs_result is None
        assert resp.usage == {"input_tokens": 10, "output_tokens": 5}
        assert resp.finish_reason == "STOP"
        assert resp.safety_ratings is None

    def test_with_logprobs(self):
        """LLMFullResponse stores logprobs_result when provided."""
        fake_logprobs = [{"token": "hi", "log_probability": -0.1}]
        resp = LLMFullResponse(
            text="hi",
            logprobs_result=fake_logprobs,
            usage={"input_tokens": 5, "output_tokens": 1},
            finish_reason="STOP",
            safety_ratings=[{"category": "SAFE"}],
        )
        assert resp.logprobs_result == fake_logprobs
        assert resp.safety_ratings == [{"category": "SAFE"}]


# ---------------------------------------------------------------------------
# generate_with_image_full() — abstract contract
# ---------------------------------------------------------------------------


class TestGenerateWithImageFullContract:
    """Verify generate_with_image_full exists on the abstract base."""

    def test_method_exists_on_base(self):
        """LLMProvider should have generate_with_image_full method."""
        assert hasattr(LLMProvider, "generate_with_image_full")

    def test_existing_generate_with_image_unchanged(self):
        """generate_with_image() still exists (backward compat)."""
        assert hasattr(LLMProvider, "generate_with_image")


# ---------------------------------------------------------------------------
# GeminiProvider.generate_with_image_full() tests
# ---------------------------------------------------------------------------


class TestGeminiGenerateWithImageFull:
    """Tests for GeminiProvider.generate_with_image_full()."""

    def _make_gemini_response(
        self,
        text="recognized text",
        finish_reason="STOP",
        input_tokens=100,
        output_tokens=20,
        logprobs_result=None,
        safety_ratings=None,
    ):
        """Build a mock Gemini response object."""
        resp = MagicMock()
        resp.text = text

        # candidates[0]
        candidate = MagicMock()
        candidate.finish_reason = finish_reason
        candidate.logprobs_result = logprobs_result
        candidate.safety_ratings = safety_ratings or []
        resp.candidates = [candidate]

        # usage_metadata
        resp.usage_metadata = MagicMock()
        resp.usage_metadata.prompt_token_count = input_tokens
        resp.usage_metadata.candidates_token_count = output_tokens

        return resp

    def test_returns_llm_full_response(self, tmp_path):
        """generate_with_image_full() returns LLMFullResponse."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        modules, mock_genai, mock_types = _mock_google_genai()
        mock_response = self._make_gemini_response()
        mock_genai.Client.return_value.models.generate_content.return_value = (
            mock_response
        )

        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            result = provider.generate_with_image_full(
                prompt="Read text",
                image_path=str(img_file),
            )

        assert isinstance(result, LLMFullResponse)
        assert result.text == "recognized text"
        assert result.finish_reason == "STOP"
        assert result.usage == {"input_tokens": 100, "output_tokens": 20}

    def test_logprobs_forwarded_when_true(self, tmp_path):
        """response_logprobs=True is passed to GenerateContentConfig."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        modules, mock_genai, mock_types = _mock_google_genai()
        # Track kwargs passed to GenerateContentConfig via the genai.types attr
        config_kwargs_log = []
        orig_config_cls = mock_genai.types.GenerateContentConfig
        def capturing_config(**kwargs):
            config_kwargs_log.append(kwargs)
            return orig_config_cls(**kwargs)
        mock_genai.types.GenerateContentConfig = capturing_config

        fake_logprobs = MagicMock()
        mock_response = self._make_gemini_response(logprobs_result=fake_logprobs)
        mock_genai.Client.return_value.models.generate_content.return_value = (
            mock_response
        )

        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            result = provider.generate_with_image_full(
                prompt="Read text",
                image_path=str(img_file),
                response_logprobs=True,
            )

        # Verify config was called with response_logprobs=True
        assert len(config_kwargs_log) >= 1
        last_config = config_kwargs_log[-1]
        assert last_config.get("response_logprobs") is True
        # logprobs_result should be forwarded
        assert result.logprobs_result is fake_logprobs

    def test_logprobs_not_forwarded_when_false(self, tmp_path):
        """response_logprobs=False (default) does not set logprobs in config."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        modules, mock_genai, mock_types = _mock_google_genai()
        config_kwargs_log = []
        orig_config_cls = mock_genai.types.GenerateContentConfig
        def capturing_config(**kwargs):
            config_kwargs_log.append(kwargs)
            return orig_config_cls(**kwargs)
        mock_genai.types.GenerateContentConfig = capturing_config

        mock_response = self._make_gemini_response()
        mock_genai.Client.return_value.models.generate_content.return_value = (
            mock_response
        )

        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            provider.generate_with_image_full(
                prompt="Read text",
                image_path=str(img_file),
                response_logprobs=False,
            )

        assert len(config_kwargs_log) >= 1
        last_config = config_kwargs_log[-1]
        assert "response_logprobs" not in last_config

    def test_safety_ratings_extracted(self, tmp_path):
        """Safety ratings from candidate are included in response."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        modules, mock_genai, mock_types = _mock_google_genai()
        fake_ratings = [MagicMock()]
        mock_response = self._make_gemini_response(safety_ratings=fake_ratings)
        mock_genai.Client.return_value.models.generate_content.return_value = (
            mock_response
        )

        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            result = provider.generate_with_image_full(
                prompt="Read text",
                image_path=str(img_file),
            )

        assert result.safety_ratings is not None

    def test_retries_on_429(self, tmp_path):
        """generate_with_image_full() retries on transient errors."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        modules, mock_genai, _ = _mock_google_genai()
        mock_response = self._make_gemini_response()
        api_mock = mock_genai.Client.return_value.models
        api_mock.generate_content.side_effect = [
            Exception("429 Too Many Requests"),
            mock_response,
        ]

        with patch.dict(sys.modules, modules):
            with patch("forma.llm_provider.time.sleep"):
                provider = GeminiProvider(api_key="k")
                result = provider.generate_with_image_full(
                    prompt="Read", image_path=str(img_file),
                )

        assert isinstance(result, LLMFullResponse)
        assert result.text == "recognized text"
        assert api_mock.generate_content.call_count == 2

    def test_existing_generate_with_image_still_returns_str(self, tmp_path):
        """Backward compat: generate_with_image() still returns str."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        modules, mock_genai, _ = _mock_google_genai()
        mock_response = MagicMock()
        mock_response.text = "just a string"
        mock_genai.Client.return_value.models.generate_content.return_value = (
            mock_response
        )

        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            result = provider.generate_with_image(
                prompt="Read text", image_path=str(img_file),
            )

        assert isinstance(result, str)
        assert result == "just a string"


# ---------------------------------------------------------------------------
# AnthropicProvider.generate_with_image_full() tests
# ---------------------------------------------------------------------------


class TestAnthropicGenerateWithImageFull:
    """Tests for AnthropicProvider.generate_with_image_full()."""

    def _make_anthropic_response(
        self,
        text="anthropic text",
        stop_reason="end_turn",
        input_tokens=50,
        output_tokens=10,
    ):
        """Build a mock Anthropic response object."""
        resp = MagicMock()
        resp.content = [MagicMock(text=text)]
        resp.stop_reason = stop_reason
        resp.usage = MagicMock()
        resp.usage.input_tokens = input_tokens
        resp.usage.output_tokens = output_tokens
        return resp

    def test_returns_llm_full_response(self, tmp_path):
        """generate_with_image_full() returns LLMFullResponse."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        mock_anthropic = MagicMock()
        mock_msg = self._make_anthropic_response()
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="k")
            result = provider.generate_with_image_full(
                prompt="Read text",
                image_path=str(img_file),
            )

        assert isinstance(result, LLMFullResponse)
        assert result.text == "anthropic text"
        assert result.finish_reason == "end_turn"
        assert result.usage == {"input_tokens": 50, "output_tokens": 10}

    def test_logprobs_always_none(self, tmp_path):
        """Anthropic does not support logprobs — always None."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        mock_anthropic = MagicMock()
        mock_msg = self._make_anthropic_response()
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="k")
            result = provider.generate_with_image_full(
                prompt="Read text",
                image_path=str(img_file),
                response_logprobs=True,  # should be ignored
            )

        assert result.logprobs_result is None

    def test_safety_ratings_none(self, tmp_path):
        """Anthropic has no safety_ratings — always None."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        mock_anthropic = MagicMock()
        mock_msg = self._make_anthropic_response()
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="k")
            result = provider.generate_with_image_full(
                prompt="Read", image_path=str(img_file),
            )

        assert result.safety_ratings is None

    def test_retries_on_429(self, tmp_path):
        """generate_with_image_full() retries on transient errors."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        mock_anthropic = MagicMock()
        mock_msg = self._make_anthropic_response()
        api_mock = mock_anthropic.Anthropic.return_value.messages
        api_mock.create.side_effect = [
            Exception("429 Too Many Requests"),
            mock_msg,
        ]

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch("forma.llm_provider.time.sleep"):
                provider = AnthropicProvider(api_key="k")
                result = provider.generate_with_image_full(
                    prompt="Read", image_path=str(img_file),
                )

        assert isinstance(result, LLMFullResponse)
        assert api_mock.create.call_count == 2

    def test_existing_generate_with_image_still_returns_str(self, tmp_path):
        """Backward compat: generate_with_image() still returns str."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPG_BYTES)

        mock_anthropic = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="just a string")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="k")
            result = provider.generate_with_image(
                prompt="Read text", image_path=str(img_file),
            )

        assert isinstance(result, str)
        assert result == "just a string"
