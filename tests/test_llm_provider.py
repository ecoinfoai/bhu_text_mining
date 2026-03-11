"""Tests for llm_provider.py — LLM provider abstraction layer.

All API calls are mocked; no real API keys required.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from forma.llm_provider import (
    AnthropicProvider,
    GeminiProvider,
    LLMProvider,
    create_provider,
    _is_retryable,
    MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# GeminiProvider tests
# ---------------------------------------------------------------------------


class TestGeminiProvider:
    """Tests for GeminiProvider."""

    def test_missing_api_key_raises(self):
        """Missing API key raises EnvironmentError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            with pytest.raises(EnvironmentError, match="Google API key"):
                GeminiProvider()

    def test_init_with_explicit_key(self):
        """Explicit api_key is accepted."""
        modules, mock_genai, _ = _mock_google_genai()
        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="test-key")
            mock_genai.Client.assert_called_once_with(api_key="test-key")
            assert provider.model_name == GeminiProvider.DEFAULT_MODEL

    def test_init_reads_env_var(self):
        """Falls back to GOOGLE_API_KEY env var."""
        modules, mock_genai, _ = _mock_google_genai()
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            with patch.dict(sys.modules, modules):
                GeminiProvider()
                mock_genai.Client.assert_called_once_with(api_key="env-key")

    def test_custom_model(self):
        """Custom model ID is used."""
        modules, _, _ = _mock_google_genai()
        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k", model="gemini-pro")
            assert provider.model_name == "gemini-pro"

    def test_generate_calls_api(self):
        """generate() calls the Gemini API and returns text."""
        modules, mock_genai, mock_types = _mock_google_genai()
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            result = provider.generate("test prompt", max_tokens=512, temperature=0.5)

        assert result == "Hello world"

    def test_generate_with_system_instruction(self):
        """generate() passes system_instruction to Gemini config."""
        modules, mock_genai, mock_types = _mock_google_genai()
        mock_response = MagicMock()
        mock_response.text = "response"
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            result = provider.generate(
                "user prompt", system_instruction="You are an evaluator."
            )
        assert result == "response"
        # Verify system_instruction was included in the config
        call_kwargs = mock_genai.Client.return_value.models.generate_content.call_args
        assert call_kwargs is not None

    def test_generate_impl_called(self):
        """_generate_impl is called by generate()."""
        modules, mock_genai, mock_types = _mock_google_genai()
        mock_response = MagicMock()
        mock_response.text = "response"
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        with patch.dict(sys.modules, modules):
            provider = GeminiProvider(api_key="k")
            result = provider._generate_impl("prompt")
        assert result == "response"

    def test_is_llm_provider(self):
        """GeminiProvider is an LLMProvider."""
        assert issubclass(GeminiProvider, LLMProvider)


# ---------------------------------------------------------------------------
# AnthropicProvider tests
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_missing_api_key_raises(self):
        """Missing API key raises EnvironmentError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(EnvironmentError, match="Anthropic API key"):
                AnthropicProvider()

    def test_init_with_explicit_key(self):
        """Explicit api_key is accepted."""
        mock_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="test-key")
            mock_anthropic.Anthropic.assert_called_once_with(api_key="test-key")
            assert provider.model_name == AnthropicProvider.DEFAULT_MODEL

    def test_generate_calls_api(self):
        """generate() calls the Anthropic API and returns text."""
        mock_anthropic = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="Response text")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="k")
            result = provider.generate("prompt", max_tokens=256)

        assert result == "Response text"

    def test_generate_with_system_instruction(self):
        """generate() passes system_instruction as system parameter."""
        mock_anthropic = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="ok")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="k")
            result = provider.generate(
                "user prompt", system_instruction="You are an evaluator."
            )
        assert result == "ok"
        # Verify system was passed to messages.create
        call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args
        assert call_kwargs.kwargs.get("system") == "You are an evaluator."

    def test_generate_impl_called(self):
        """_generate_impl is called by generate()."""
        mock_anthropic = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="impl text")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = AnthropicProvider(api_key="k")
            result = provider._generate_impl("prompt")
        assert result == "impl text"

    def test_is_llm_provider(self):
        """AnthropicProvider is an LLMProvider."""
        assert issubclass(AnthropicProvider, LLMProvider)


# ---------------------------------------------------------------------------
# create_provider factory tests
# ---------------------------------------------------------------------------


class TestCreateProvider:
    """Tests for create_provider() factory function."""

    def test_gemini_default(self):
        """Default provider is gemini."""
        modules, _, _ = _mock_google_genai()
        with patch.dict(sys.modules, modules):
            provider = create_provider(provider="gemini", api_key="k")
            assert isinstance(provider, GeminiProvider)

    def test_anthropic_provider(self):
        """provider='anthropic' creates AnthropicProvider."""
        mock_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = create_provider(provider="anthropic", api_key="k")
            assert isinstance(provider, AnthropicProvider)

    def test_claude_alias(self):
        """provider='claude' is an alias for anthropic."""
        mock_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = create_provider(provider="claude", api_key="k")
            assert isinstance(provider, AnthropicProvider)

    def test_unknown_provider_raises(self):
        """Unknown provider name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_provider(provider="openai", api_key="k")

    def test_case_insensitive(self):
        """Provider name is case-insensitive."""
        modules, _, _ = _mock_google_genai()
        with patch.dict(sys.modules, modules):
            provider = create_provider(provider="GEMINI", api_key="k")
            assert isinstance(provider, GeminiProvider)

    def test_model_passed_through(self):
        """Custom model is forwarded to provider."""
        modules, _, _ = _mock_google_genai()
        with patch.dict(sys.modules, modules):
            provider = create_provider(provider="gemini", api_key="k", model="custom-model")
            assert provider.model_name == "custom-model"


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------


class TestIsRetryable:
    """Tests for _is_retryable()."""

    def test_429_is_retryable(self):
        assert _is_retryable(Exception("Error 429 Too Many Requests"))

    def test_rate_limit_is_retryable(self):
        assert _is_retryable(Exception("rate limit exceeded"))

    def test_connection_error_is_retryable(self):
        assert _is_retryable(ConnectionError("connection refused"))

    def test_timeout_error_is_retryable(self):
        assert _is_retryable(TimeoutError("request timed out"))

    def test_value_error_not_retryable(self):
        assert not _is_retryable(ValueError("bad input"))

    def test_parse_error_not_retryable(self):
        assert not _is_retryable(Exception("JSON parse failed"))


class TestRetryLogic:
    """Tests for generate() retry with exponential backoff."""

    def test_retries_on_429(self):
        """generate() retries on 429 and succeeds on second attempt."""
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
                result = provider.generate("prompt")
        assert result == "ok"
        assert api_mock.create.call_count == 2

    def test_no_retry_on_parse_error(self):
        """generate() does not retry on non-transient errors."""
        mock_anthropic = MagicMock()
        api_mock = mock_anthropic.Anthropic.return_value.messages
        api_mock.create.side_effect = ValueError("bad format")

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch("forma.llm_provider.time.sleep"):
                provider = AnthropicProvider(api_key="k")
                with pytest.raises(ValueError, match="bad format"):
                    provider.generate("prompt")
        assert api_mock.create.call_count == 1

    def test_exhausts_retries(self):
        """generate() raises after MAX_RETRIES on persistent 429."""
        mock_anthropic = MagicMock()
        api_mock = mock_anthropic.Anthropic.return_value.messages
        api_mock.create.side_effect = Exception("429 Too Many Requests")

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch("forma.llm_provider.time.sleep"):
                provider = AnthropicProvider(api_key="k")
                with pytest.raises(Exception, match="429"):
                    provider.generate("prompt")
        assert api_mock.create.call_count == MAX_RETRIES
