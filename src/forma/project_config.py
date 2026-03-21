"""Unified project-level configuration for formative-analysis.

Provides ``ProjectConfiguration`` dataclass and utility functions for
discovering, loading, validating, and merging project configuration from
``forma.yaml`` files.

Resolution order (three-layer merge):
    1. CLI flags (highest priority)
    2. Project configuration file (``forma.yaml``)
    3. Existing system-level configuration chain (``config.py``)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Known top-level sections in forma.yaml
_KNOWN_SECTIONS = {
    "project", "classes", "paths", "ocr", "evaluation", "reports",
    "prediction", "current_week", "domain_analysis",
}

# Known keys within each section
_KNOWN_KEYS: dict[str, set[str]] = {
    "project": {"course_name", "year", "semester", "grade"},
    "classes": {"identifiers", "join_pattern", "eval_pattern"},
    "paths": {
        "exam_config", "join_dir", "output_dir", "longitudinal_store",
        "font_path",
    },
    "ocr": {"naver_config", "credentials", "spreadsheet_url", "num_questions", "ocr_model"},
    "evaluation": {
        "provider", "model", "skip_feedback", "skip_graph",
        "skip_statistical", "n_calls",
    },
    "reports": {"dpi", "skip_llm", "aggregate"},
    "prediction": {"model_path"},
    "domain_analysis": {"extract_model", "coverage_model", "feedback_model", "quality_weights", "llm_calls"},
}

# Mapping from forma.yaml nested key to flat field name
_SECTION_KEY_MAP: dict[str, dict[str, str]] = {
    "project": {
        "course_name": "course_name",
        "year": "year",
        "semester": "semester",
        "grade": "grade",
    },
    "classes": {
        "identifiers": "class_identifiers",
        "join_pattern": "join_pattern",
        "eval_pattern": "eval_pattern",
    },
    "paths": {
        "exam_config": "exam_config",
        "join_dir": "join_dir",
        "output_dir": "output_dir",
        "longitudinal_store": "longitudinal_store",
        "font_path": "font_path",
    },
    "ocr": {
        "naver_config": "naver_config",
        "credentials": "credentials",
        "spreadsheet_url": "spreadsheet_url",
        "num_questions": "num_questions",
        "ocr_model": "ocr_model",
    },
    "evaluation": {
        "provider": "provider",
        "model": "model",
        "skip_feedback": "skip_feedback",
        "skip_graph": "skip_graph",
        "skip_statistical": "skip_statistical",
        "n_calls": "n_calls",
    },
    "reports": {
        "dpi": "dpi",
        "skip_llm": "skip_llm",
        "aggregate": "aggregate",
    },
    "prediction": {
        "model_path": "model_path",
    },
}


@dataclass
class ProjectConfiguration:
    """Unified project-level configuration loaded from ``forma.yaml``.

    All fields have sensible defaults; a professor only needs to specify
    values that differ from the defaults.

    Attributes:
        course_name: Course name (e.g. "Human Anatomy and Physiology").
        year: Academic year (>= 2020).
        semester: Semester number (1 or 2).
        grade: Student grade year (>= 1).
        class_identifiers: List of class section identifiers (e.g. ["A", "B"]).
        join_pattern: File pattern for joined data, must contain ``{class}``.
        eval_pattern: Directory pattern for evaluation results.
        exam_config: Path to exam configuration YAML file.
        join_dir: Path to joined data directory.
        output_dir: Path to output directory.
        longitudinal_store: Path to longitudinal store YAML file.
        font_path: Path to Korean font file, or None for auto-detect.
        naver_config: Path to Naver OCR configuration.  **Deprecated** —
            use ``--provider gemini`` (LLM Vision OCR) instead.
        credentials: Credentials reference (resolved from env var).
        spreadsheet_url: Google Sheets URL.
        num_questions: Number of questions per exam (>= 1).
        ocr_model: LLM model ID for OCR (e.g. "gemini-2.0-flash"), or None for provider default.
        provider: LLM provider ("gemini" or "anthropic").
        model: LLM model name, or None for provider default.
        skip_feedback: Skip feedback generation.
        skip_graph: Skip graph comparison.
        skip_statistical: Skip statistical analysis.
        n_calls: Number of LLM calls per item (>= 1).
        dpi: Chart image resolution (72-600).
        skip_llm: Skip all LLM analysis.
        aggregate: Generate aggregate report.
        current_week: Current week number (>= 1).
        model_path: Path to pre-trained risk prediction model, or None.
    """

    course_name: str = ""
    year: int = 0
    semester: int = 0
    grade: int = 0
    class_identifiers: list[str] = field(default_factory=list)
    join_pattern: str = ""
    eval_pattern: str = ""
    exam_config: str = ""
    join_dir: str = ""
    output_dir: str = ""
    longitudinal_store: str = ""
    font_path: str | None = None
    naver_config: str = ""
    credentials: str = ""
    spreadsheet_url: str = ""
    num_questions: int = 5
    provider: str = "gemini"
    model: str | None = None
    skip_feedback: bool = False
    skip_graph: bool = False
    skip_statistical: bool = False
    n_calls: int = 3
    dpi: int = 150
    skip_llm: bool = False
    aggregate: bool = True
    current_week: int = 1
    ocr_model: str | None = None
    model_path: str | None = None


def find_project_config(start_dir: Path | None = None) -> Path | None:
    """Search from start_dir upward for ``forma.yaml``.

    Walks the directory tree from ``start_dir`` (default: CWD) upward,
    stopping at the first ``forma.yaml`` found or at a ``.git`` sentinel
    directory (project root) or filesystem root.

    Args:
        start_dir: Directory to start searching from. Defaults to CWD.

    Returns:
        Path to the ``forma.yaml`` file, or None if not found.
    """
    current = Path(start_dir or Path.cwd()).resolve()
    while True:
        candidate = current / "forma.yaml"
        if candidate.is_file():
            return candidate
        if (current / ".git").exists():
            break  # Reached project root
        parent = current.parent
        if parent == current:
            break  # Reached filesystem root
        current = parent
    return None


def load_project_config(path: Path) -> dict:
    """Load project configuration from a YAML file.

    Args:
        path: Path to the ``forma.yaml`` file.

    Returns:
        Parsed configuration dict. Returns empty dict for empty files.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is not None and not isinstance(data, dict):
        logger.warning(
            "Configuration file %s contains non-dict data (type: %s); "
            "treating as empty configuration",
            path,
            type(data).__name__,
        )
        return {}
    return data if isinstance(data, dict) else {}


def validate_project_config(config_dict: dict) -> None:
    """Validate a project configuration dict.

    Checks for unknown keys (warning), type mismatches (error), and
    value constraint violations (error). All errors are collected and
    reported together via a single ``ValueError``.

    Args:
        config_dict: Parsed configuration dict from ``load_project_config()``.

    Raises:
        ValueError: If any type or value constraint violations are found.
            The message contains all errors joined by newlines.
    """
    errors: list[str] = []

    # Check top-level keys
    for key in config_dict:
        if key not in _KNOWN_SECTIONS:
            logger.warning("Unknown configuration key: '%s'", key)

    # Check nested keys within known sections
    for section, known_keys in _KNOWN_KEYS.items():
        section_data = config_dict.get(section)
        if not isinstance(section_data, dict):
            continue
        for key in section_data:
            if key not in known_keys:
                logger.warning(
                    "Unknown key '%s' in section '%s'", key, section,
                )

    # --- Type and value checks ---

    # project section
    project = config_dict.get("project", {})
    if isinstance(project, dict):
        _check_int(project, "year", errors)
        _check_int(project, "semester", errors)
        _check_int(project, "grade", errors)
        _check_str(project, "course_name", errors)

        if "year" in project and isinstance(project["year"], int) and not isinstance(project["year"], bool):
            if project["year"] < 2020:
                errors.append("project.year must be >= 2020")

        if "semester" in project and isinstance(project["semester"], int) and not isinstance(project["semester"], bool):
            if project["semester"] not in (1, 2):
                errors.append("project.semester must be 1 or 2")

        if "grade" in project and isinstance(project["grade"], int) and not isinstance(project["grade"], bool):
            if project["grade"] < 1:
                errors.append("project.grade must be >= 1")

    # classes section
    classes = config_dict.get("classes", {})
    if isinstance(classes, dict):
        if "identifiers" in classes:
            if not isinstance(classes["identifiers"], list):
                errors.append("classes.identifiers must be a list")

        if "join_pattern" in classes:
            val = classes["join_pattern"]
            if not isinstance(val, str):
                errors.append("classes.join_pattern must be a string")
            elif val and "{class}" not in val:
                errors.append(
                    "classes.join_pattern must contain '{class}' placeholder",
                )

        if "eval_pattern" in classes:
            val = classes["eval_pattern"]
            if not isinstance(val, str):
                errors.append("classes.eval_pattern must be a string")
            elif val and "{class}" not in val:
                errors.append(
                    "classes.eval_pattern must contain '{class}' placeholder",
                )

    # ocr section
    ocr = config_dict.get("ocr", {})
    if isinstance(ocr, dict):
        _check_str_or_none(ocr, "ocr_model", errors)
        _check_int(ocr, "num_questions", errors)
        nq = ocr.get("num_questions")
        if nq is not None and isinstance(nq, int) and not isinstance(nq, bool):
            if ocr["num_questions"] < 1:
                errors.append("ocr.num_questions must be >= 1")

    # evaluation section
    evaluation = config_dict.get("evaluation", {})
    if isinstance(evaluation, dict):
        if "provider" in evaluation:
            if not isinstance(evaluation["provider"], str):
                errors.append("evaluation.provider must be a string")
            elif evaluation["provider"] not in ("gemini", "anthropic"):
                errors.append(
                    "evaluation.provider must be 'gemini' or 'anthropic'",
                )
        _check_int(evaluation, "n_calls", errors)
        nc = evaluation.get("n_calls")
        if nc is not None and isinstance(nc, int) and not isinstance(nc, bool):
            if evaluation["n_calls"] < 1:
                errors.append("evaluation.n_calls must be >= 1")
        _check_bool(evaluation, "skip_feedback", errors)
        _check_bool(evaluation, "skip_graph", errors)
        _check_bool(evaluation, "skip_statistical", errors)

    # reports section
    reports = config_dict.get("reports", {})
    if isinstance(reports, dict):
        _check_int(reports, "dpi", errors)
        if "dpi" in reports and isinstance(reports["dpi"], int) and not isinstance(reports["dpi"], bool):
            if not (72 <= reports["dpi"] <= 600):
                errors.append("reports.dpi must be between 72 and 600")
        _check_bool(reports, "skip_llm", errors)
        _check_bool(reports, "aggregate", errors)

    # top-level current_week
    if "current_week" in config_dict:
        if isinstance(config_dict["current_week"], bool):
            errors.append("current_week must be an integer, got bool")
        elif not isinstance(config_dict["current_week"], int):
            errors.append("current_week must be an integer")
        elif config_dict["current_week"] < 1:
            errors.append("current_week must be >= 1")

    if errors:
        raise ValueError(
            "Configuration validation errors:\n" + "\n".join(f"  - {e}" for e in errors),
        )


def merge_configs(
    cli_namespace: argparse.Namespace,
    project_config: dict,
    system_config: dict,
    *,
    explicit_keys: set[str] | None = None,
) -> dict:
    """Merge three configuration layers into a single resolved dict.

    Precedence: CLI flags (highest) > project config > system config > defaults.

    Only values explicitly set at each layer override lower layers.
    CLI values are considered "explicitly set" if their key appears in
    ``explicit_keys``.

    Args:
        cli_namespace: Parsed argparse Namespace with CLI flag values.
        project_config: Parsed project config dict (may be nested per forma.yaml schema).
        system_config: Parsed system config dict (flat, from ``config.py``).
        explicit_keys: Set of CLI arg names that were explicitly set by the user
            (not just argparse defaults). If None, all non-None CLI values are
            treated as explicit.

    Returns:
        Flat dict with resolved configuration values.
    """
    if explicit_keys is None:
        explicit_keys = set()

    # Step 1: Flatten project config (nested sections → flat keys)
    flat_project: dict[str, Any] = {}
    for section, key_map in _SECTION_KEY_MAP.items():
        section_data = project_config.get(section, {})
        if not isinstance(section_data, dict):
            continue
        for yaml_key, flat_key in key_map.items():
            if yaml_key in section_data:
                flat_project[flat_key] = section_data[yaml_key]

    # Handle top-level keys (e.g., current_week)
    if "current_week" in project_config:
        flat_project["current_week"] = project_config["current_week"]

    # Step 2: Build resolved dict using three-layer precedence
    result: dict[str, Any] = {}
    cli_dict = vars(cli_namespace)

    for key, cli_value in cli_dict.items():
        if key in explicit_keys:
            # CLI was explicitly set — highest priority
            result[key] = cli_value
        elif key in flat_project:
            # Project config has this value
            result[key] = flat_project[key]
        elif key in system_config:
            # System config has this value
            result[key] = system_config[key]
        else:
            # Fall through to argparse default
            result[key] = cli_value

    # Also add project config keys that CLI doesn't know about
    for key, value in flat_project.items():
        if key not in result:
            result[key] = value

    # Also add system config keys that neither CLI nor project config has
    for key, value in system_config.items():
        if key not in result:
            result[key] = value

    return result


def apply_project_config(
    args: argparse.Namespace,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Load project config and merge into CLI args namespace.

    Performs the full three-layer merge: CLI flags > forma.yaml > defaults.
    Skips config loading if ``args.no_config`` is True.

    The ``explicit_keys`` set is determined by scanning ``argv`` for
    flags that start with ``--`` and mapping them to namespace attribute
    names (e.g., ``--font-path`` -> ``font_path``).

    Args:
        args: Parsed argparse Namespace.
        argv: Raw CLI arguments (for explicit_keys detection).
            If None, no CLI flags are considered explicit.

    Returns:
        The same Namespace with values potentially overridden by project config.
    """
    if getattr(args, "no_config", False):
        return args

    # Determine which CLI flags were explicitly provided
    explicit_keys: set[str] = set()
    if argv:
        for token in argv:
            if token.startswith("--"):
                key = token.lstrip("-").split("=")[0].replace("-", "_")
                explicit_keys.add(key)

    # Discover and load project config
    config_path = find_project_config()
    if config_path is None:
        return args

    try:
        config_dict = load_project_config(config_path)
    except Exception as exc:
        logger.warning("Failed to load project config %s: %s", config_path, exc)
        return args

    try:
        validate_project_config(config_dict)
    except ValueError as exc:
        logger.warning("Project config validation failed: %s", exc)
        return args

    logger.info("Loaded project config: %s", config_path)

    # Merge: CLI explicit > project config > argparse default
    merged = merge_configs(args, config_dict, {}, explicit_keys=explicit_keys)

    # Apply merged values back to namespace
    for key, value in merged.items():
        if hasattr(args, key):
            setattr(args, key, value)

    return args


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _check_int(section: dict, key: str, errors: list[str]) -> None:
    """Check that a key, if present, is an int (not bool)."""
    if key in section:
        # Check bool FIRST — bool is a subclass of int in Python
        if isinstance(section[key], bool):
            errors.append(f"{key} must be an integer, got bool")
        elif not isinstance(section[key], int):
            errors.append(f"{key} must be an integer")


def _check_str(section: dict, key: str, errors: list[str]) -> None:
    """Check that a key, if present, is a str."""
    if key in section and not isinstance(section[key], str):
        errors.append(f"{key} must be a string")


def _check_str_or_none(section: dict, key: str, errors: list[str]) -> None:
    """Check that a key, if present, is a str or None."""
    if key in section and section[key] is not None and not isinstance(section[key], str):
        errors.append(f"{key} must be a string or null")


def _check_bool(section: dict, key: str, errors: list[str]) -> None:
    """Check that a key, if present, is a bool."""
    if key in section and not isinstance(section[key], bool):
        errors.append(f"{key} must be a boolean")
