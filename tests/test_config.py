"""Tests for config.py — unified configuration management."""

import json
import os
from unittest.mock import patch

import pytest

from forma.config import (
    AGENIX_CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    LEGACY_CONFIG_PATHS,
    get_llm_config,
    get_naver_ocr_config,
    load_config,
)


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for load_config()."""

    def test_explicit_path(self, tmp_path):
        """Explicit config_path is used directly."""
        cfg = {"llm": {"provider": "gemini"}}
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        result = load_config(str(cfg_file))
        assert result == cfg

    def test_default_path_found(self, tmp_path):
        """Default path is found when no explicit path given."""
        cfg = {"naver_ocr": {"secret_key": "s", "api_url": "u"}}
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        with patch("forma.config.DEFAULT_CONFIG_PATH", str(cfg_file)):
            result = load_config()
        assert result == cfg

    def test_legacy_fallback(self, tmp_path):
        """Falls back to legacy path when default is missing."""
        cfg = {"secret_key": "s", "api_url": "u"}
        cfg_file = tmp_path / "legacy.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        with patch("forma.config.DEFAULT_CONFIG_PATH", "/nonexistent/path.json"):
            with patch("forma.config.LEGACY_CONFIG_PATHS", [str(cfg_file)]):
                result = load_config()
        assert result == cfg

    def test_agenix_path_found(self, tmp_path):
        """Finds config at agenix path."""
        cfg = {"llm": {"provider": "gemini"}}
        cfg_file = tmp_path / "agenix.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        with patch("forma.config.AGENIX_CONFIG_PATH", str(cfg_file)):
            with patch("forma.config.DEFAULT_CONFIG_PATH", "/nonexistent/a.json"):
                with patch("forma.config.LEGACY_CONFIG_PATHS", ["/nonexistent/b.json"]):
                    result = load_config()
        assert result == cfg

    def test_no_config_raises(self):
        """Raises FileNotFoundError when no config file found."""
        with patch("forma.config.AGENIX_CONFIG_PATH", "/nonexistent/z.json"):
            with patch("forma.config.DEFAULT_CONFIG_PATH", "/nonexistent/a.json"):
                with patch("forma.config.LEGACY_CONFIG_PATHS", ["/nonexistent/b.json"]):
                    with pytest.raises(FileNotFoundError, match="No config file"):
                        load_config()


# ---------------------------------------------------------------------------
# get_naver_ocr_config tests
# ---------------------------------------------------------------------------


class TestGetNaverOcrConfig:
    """Tests for get_naver_ocr_config()."""

    def test_nested_format(self):
        """Extracts from new nested naver_ocr format."""
        cfg = {"naver_ocr": {"secret_key": "sk", "api_url": "url"}}
        secret, url = get_naver_ocr_config(cfg)
        assert secret == "sk"
        assert url == "url"

    def test_flat_format(self):
        """Extracts from legacy flat format."""
        cfg = {"secret_key": "sk2", "api_url": "url2"}
        secret, url = get_naver_ocr_config(cfg)
        assert secret == "sk2"
        assert url == "url2"

    def test_missing_keys_raises(self):
        """Missing keys raise KeyError."""
        with pytest.raises(KeyError):
            get_naver_ocr_config({})


# ---------------------------------------------------------------------------
# get_llm_config tests
# ---------------------------------------------------------------------------


class TestGetLlmConfig:
    """Tests for get_llm_config()."""

    def test_full_config(self):
        """Extracts all LLM settings."""
        cfg = {
            "llm": {
                "provider": "anthropic",
                "api_key": "ak",
                "model": "claude-sonnet-4-6",
            }
        }
        result = get_llm_config(cfg)
        assert result["provider"] == "anthropic"
        assert result["api_key"] == "ak"
        assert result["model"] == "claude-sonnet-4-6"

    def test_defaults_when_missing(self):
        """Returns defaults when llm section is absent."""
        result = get_llm_config({})
        assert result["provider"] == "gemini"
        assert result["api_key"] is None
        assert result["model"] is None

    def test_partial_config(self):
        """Partial llm config fills in None for missing keys."""
        cfg = {"llm": {"provider": "gemini"}}
        result = get_llm_config(cfg)
        assert result["provider"] == "gemini"
        assert result["api_key"] is None
