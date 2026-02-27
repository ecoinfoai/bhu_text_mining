"""Tests for embedding_cache.py singleton encoder.

RED phase: validates caching behaviour and encode_texts output shape.
Heavy sentence-transformer loading is mocked.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestGetEncoder:
    """Tests for get_encoder() LRU-cached singleton."""

    def test_get_encoder_returns_same_instance(self):
        """Test that two calls with same model return the same object."""
        mock_model = MagicMock()
        with patch(
            "src.embedding_cache.SentenceTransformer", return_value=mock_model
        ) as mock_cls:
            from src.embedding_cache import get_encoder

            get_encoder.cache_clear()
            enc1 = get_encoder("test-model")
            enc2 = get_encoder("test-model")
            assert enc1 is enc2
            assert mock_cls.call_count == 1

    def test_get_encoder_different_models_different_instances(self):
        """Test that different model names return different instances."""
        mock_a = MagicMock(name="model_a")
        mock_b = MagicMock(name="model_b")

        def side_effect(name):
            return mock_a if "a" in name else mock_b

        with patch(
            "src.embedding_cache.SentenceTransformer",
            side_effect=side_effect,
        ):
            from src.embedding_cache import get_encoder

            get_encoder.cache_clear()
            enc_a = get_encoder("model-a")
            enc_b = get_encoder("model-b")
            assert enc_a is not enc_b

    def test_get_encoder_uses_default_model(self):
        """Test default model name is ko-sroberta-multitask."""
        from src.embedding_cache import DEFAULT_MODEL

        assert "ko-sroberta" in DEFAULT_MODEL


class TestEncodeTexts:
    """Tests for encode_texts() wrapper function."""

    def test_encode_texts_returns_numpy_array(self):
        """Test that encode_texts returns a numpy ndarray."""
        dummy_embeddings = np.random.rand(3, 768).astype(np.float32)
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = dummy_embeddings

        with patch("src.embedding_cache.get_encoder", return_value=mock_encoder):
            from src.embedding_cache import encode_texts

            result = encode_texts(["문장1", "문장2", "문장3"])
            assert isinstance(result, np.ndarray)
            assert result.shape == (3, 768)

    def test_encode_texts_calls_encoder(self):
        """Test that encode_texts calls encoder.encode with correct texts."""
        dummy = np.zeros((2, 768), dtype=np.float32)
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = dummy

        with patch("src.embedding_cache.get_encoder", return_value=mock_encoder):
            from src.embedding_cache import encode_texts

            texts = ["세포막", "인지질 이중층"]
            encode_texts(texts)
            mock_encoder.encode.assert_called_once()
            call_args = mock_encoder.encode.call_args[0][0]
            assert call_args == texts

    def test_encode_texts_empty_list_raises(self):
        """Test that encoding an empty list raises ValueError."""
        with patch("src.embedding_cache.get_encoder", return_value=MagicMock()):
            from src.embedding_cache import encode_texts

            with pytest.raises(ValueError, match="empty"):
                encode_texts([])
