"""Unified configuration management for formative-analysis.

Searches ``~/.config/formative-analysis/forma.json`` first, then falls
back to legacy paths.
"""

from __future__ import annotations

import json
import os
from typing import Optional


AGENIX_CONFIG_PATH = "/run/agenix/forma-config"
DEFAULT_CONFIG_PATH = "~/.config/formative-analysis/forma.json"
LEGACY_CONFIG_PATHS = [
    "~/.config/forma/config.json",
    "~/.config/bhu_text_mining/config.json",
    "~/.config/naver_ocr/naver_ocr_config.json",
]


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from JSON file.

    Resolution order:
    1. Explicit ``config_path`` argument.
    2. ``/run/agenix/forma-config`` (NixOS agenix).
    3. ``~/.config/formative-analysis/forma.json``
    4. ``~/.config/forma/config.json`` (legacy).
    5. ``~/.config/bhu_text_mining/config.json`` (legacy).
    6. ``~/.config/naver_ocr/naver_ocr_config.json`` (legacy).

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


JSON_FIELD_MAP = {
    "server": "smtp_server",
    "port": "smtp_port",
    "sender_email": "sender_email",
    "sender_name": "sender_name",
    "use_tls": "use_tls",
    "send_interval_sec": "send_interval_sec",
}


def get_smtp_config(config: dict):
    """Extract SMTP settings from forma.json config dict.

    Maps JSON-style field names (``server``, ``port``, etc.) to
    ``SmtpConfig`` fields via ``_build_smtp_config()``.

    Args:
        config: Parsed forma.json config dict.

    Returns:
        ``SmtpConfig`` instance.

    Raises:
        KeyError: If ``smtp`` section is missing or not a dict.
        ValueError: If required SMTP fields are invalid.
    """
    from forma.delivery_send import _build_smtp_config

    smtp_data = config.get("smtp")
    if not isinstance(smtp_data, dict):
        raise KeyError("forma.json에 'smtp' 섹션이 없습니다")

    return _build_smtp_config(smtp_data, field_map=JSON_FIELD_MAP)


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
