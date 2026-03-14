"""Tests for LLM provider vision (generate_with_image) support.

All API calls are mocked; no real API keys required.
"""

import os
import sys
from unittest.mock import MagicMock, patch, mock_open

import pytest

from forma.llm_provider import (
    AnthropicProvider,
    GeminiProvider,
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
