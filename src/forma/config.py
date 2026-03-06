"""Unified configuration management for formative-analysis.

Searches ``~/.config/forma/config.json`` first, then falls
back to legacy paths.
"""

from __future__ import annotations

import json
import os
from typing import Optional


AGENIX_CONFIG_PATH = "/run/agenix/forma-config"
DEFAULT_CONFIG_PATH = "~/.config/forma/config.json"
LEGACY_CONFIG_PATHS = [
    "~/.config/bhu_text_mining/config.json",
    "~/.config/naver_ocr/naver_ocr_config.json",
]


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from JSON file.

    Resolution order:
    1. Explicit ``config_path`` argument.
    2. ``/run/agenix/forma-config`` (NixOS agenix).
    3. ``~/.config/forma/config.json``
    4. ``~/.config/bhu_text_mining/config.json`` (legacy).
    5. ``~/.config/naver_ocr/naver_ocr_config.json`` (legacy).

    Args:
        config_path: Explicit path to config file (optional).

    Returns:
        Parsed configuration dict.

    Raises:
        FileNotFoundError: If no config file is found.
    """
    candidates = []
    if config_path:
        candidates.append(config_path)
    else:
        candidates.append(AGENIX_CONFIG_PATH)
        candidates.append(DEFAULT_CONFIG_PATH)
        candidates.extend(LEGACY_CONFIG_PATHS)

    for path in candidates:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded):
            with open(expanded, "r", encoding="utf-8") as f:
                return json.load(f)

    searched = [os.path.expanduser(p) for p in candidates]
    raise FileNotFoundError(
        f"No config file found. Searched: {searched}"
    )


def get_naver_ocr_config(config: dict) -> tuple[str, str]:
    """Extract Naver OCR credentials from config dict.

    Supports both new nested format (``config["naver_ocr"]``) and
    legacy flat format (``config["secret_key"]``).

    Args:
        config: Parsed config dict.

    Returns:
        Tuple of (secret_key, api_url).

    Raises:
        KeyError: If required keys are missing.
    """
    if "naver_ocr" in config:
        ocr = config["naver_ocr"]
        return ocr["secret_key"], ocr["api_url"]
    return config["secret_key"], config["api_url"]


def get_llm_config(config: dict) -> dict:
    """Extract LLM settings from config dict.

    Returns:
        Dict with keys: provider, api_key, model (all optional,
        may be None if not set).
    """
    llm = config.get("llm", {})
    return {
        "provider": llm.get("provider", "gemini"),
        "api_key": llm.get("api_key"),
        "model": llm.get("model"),
    }
