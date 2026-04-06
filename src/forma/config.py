"""Unified configuration management for formative-analysis.

Searches ``~/.config/formative-analysis/config.json`` first, then falls
back to the deprecated ``forma.json`` path.
"""

from __future__ import annotations

import json
import logging
import os
import warnings

logger = logging.getLogger(__name__)

# Known top-level sections in config.json
_EXPECTED_SECTIONS = {"naver_ocr", "smtp", "llm"}

AGENIX_CONFIG_PATH = "/run/agenix/forma-config"
DEFAULT_CONFIG_PATH = "~/.config/formative-analysis/config.json"
DEPRECATED_CONFIG_PATH = "~/.config/formative-analysis/forma.json"


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from JSON file.

    Resolution order:
    1. Explicit ``config_path`` argument.
    2. ``/run/agenix/forma-config`` (NixOS agenix).
    3. ``~/.config/formative-analysis/config.json``
    4. ``~/.config/formative-analysis/forma.json`` (deprecated).

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

    for path in candidates:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded):
            with open(expanded, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"config.json must be a JSON object ({{...}}), got {type(data).__name__}")
            for key in data:
                if key not in _EXPECTED_SECTIONS:
                    logger.warning("Unknown key in config.json: '%s'", key)
            return data

    # Deprecated fallback: forma.json
    if not config_path:
        dep_expanded = os.path.expanduser(DEPRECATED_CONFIG_PATH)
        if os.path.isfile(dep_expanded):
            warnings.warn(
                "forma.json is deprecated and will be removed in a future version. "
                "Rename it to config.json: "
                f"{dep_expanded}",
                DeprecationWarning,
                stacklevel=2,
            )
            with open(dep_expanded, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"config.json must be a JSON object ({{...}}), got {type(data).__name__}")
            for key in data:
                if key not in _EXPECTED_SECTIONS:
                    logger.warning("Unknown key in config.json: '%s'", key)
            return data

    searched = [os.path.expanduser(p) for p in candidates]
    raise FileNotFoundError(f"No config file found. Searched: {searched}")


def get_naver_ocr_config(config: dict) -> tuple[str, str]:
    """Extract Naver OCR credentials from config dict.

    Requires nested format (``config["naver_ocr"]``).

    Args:
        config: Parsed config dict.

    Returns:
        Tuple of (secret_key, api_url).

    Raises:
        KeyError: If required keys are missing.
    """
    if "naver_ocr" in config:
        ocr = config["naver_ocr"]
        if not isinstance(ocr, dict):
            raise KeyError("'naver_ocr' section must be a dict")
        return ocr["secret_key"], ocr["api_url"]
    raise KeyError("'naver_ocr' section not found in config")


JSON_FIELD_MAP = {
    "server": "smtp_server",
    "port": "smtp_port",
    "sender_email": "sender_email",
    "sender_name": "sender_name",
    "use_tls": "use_tls",
    "send_interval_sec": "send_interval_sec",
}


def get_smtp_config(config: dict):
    """Extract SMTP settings from config.json config dict.

    Maps JSON-style field names (``server``, ``port``, etc.) to
    ``SmtpConfig`` fields via ``_build_smtp_config()``.

    Args:
        config: Parsed config.json config dict.

    Returns:
        ``SmtpConfig`` instance.

    Raises:
        KeyError: If ``smtp`` section is missing or not a dict.
        ValueError: If required SMTP fields are invalid.
    """
    from forma.delivery_send import _build_smtp_config

    smtp_data = config.get("smtp")
    if not isinstance(smtp_data, dict):
        raise KeyError("'smtp' section not found in config.json")

    return _build_smtp_config(smtp_data, field_map=JSON_FIELD_MAP)


def get_smtp_password(config: dict) -> str | None:
    """Extract SMTP password from config dict.

    Returns the ``smtp.password`` field as a string, or ``None`` if absent.
    Does not raise if the ``smtp`` section is missing.

    Args:
        config: Parsed config.json config dict.

    Returns:
        SMTP password string, or ``None`` if not set.
    """
    smtp = config.get("smtp")
    if not isinstance(smtp, dict):
        return None
    value = smtp.get("password")
    if value is None:
        return None
    return str(value)


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


_DEFAULT_QUALITY_WEIGHTS: dict[str, float] = {
    "embedding": 0.25,
    "term_coverage": 0.25,
    "density": 0.15,
    "llm": 0.35,
}


def get_quality_weights(config: dict) -> dict[str, float]:
    """Extract quality ensemble weights from config dict.

    Reads ``domain_analysis.quality_weights`` from the project
    configuration. Missing keys are filled with defaults.

    Default weights::

        embedding: 0.25
        term_coverage: 0.25
        density: 0.15
        llm: 0.35

    Args:
        config: Parsed project configuration dict (forma.yaml style,
            may contain a ``domain_analysis`` section).

    Returns:
        Dict mapping signal name to weight (floats summing to ~1.0).
    """
    domain = config.get("domain_analysis", {})
    if not isinstance(domain, dict):
        return dict(_DEFAULT_QUALITY_WEIGHTS)

    overrides = domain.get("quality_weights", {})
    if not isinstance(overrides, dict):
        return dict(_DEFAULT_QUALITY_WEIGHTS)

    result = dict(_DEFAULT_QUALITY_WEIGHTS)
    for key in _DEFAULT_QUALITY_WEIGHTS:
        if key in overrides:
            try:
                result[key] = float(overrides[key])
            except (TypeError, ValueError):
                logger.warning(
                    "quality_weights.%s has invalid value, using default: %s",
                    key,
                    overrides[key],
                )
    return result
