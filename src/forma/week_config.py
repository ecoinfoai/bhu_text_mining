"""Weekly assessment configuration for formative-analysis.

Provides ``WeekConfiguration`` dataclass and utility functions for
discovering, loading, validating, and saving ``week.yaml`` files.
Supports ``{class}`` pattern resolution for class-specific paths.

Resolution order (five-level merge):
    1. CLI flags (highest priority)
    2. ``week.yaml`` (per-week settings)
    3. Legacy ``--eval-config`` file (backward compat)
    4. ``forma.yaml`` (semester-level defaults)
    5. Hardcoded defaults (lowest priority)
"""

from __future__ import annotations

import copy
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Path-valued fields in WeekConfiguration that support {class} patterns
_CLASS_PATTERN_FIELDS = (
    "ocr_image_dir_pattern",
    "ocr_ocr_output_pattern",
    "ocr_join_output_pattern",
    "eval_responses_pattern",
    "eval_output_pattern",
)


@dataclass
class WeekConfiguration:
    """Weekly assessment settings loaded from ``week.yaml``.

    Attributes:
        week: Week number (>= 1).
        select_source: Path to FormativeTest YAML.
        select_questions: Question sn numbers to extract.
        select_num_papers: Number of exam papers to print.
        select_form_url: Google Forms pre-filled URL template.
        select_exam_output: PDF output path (triggers PDF generation).
        ocr_num_questions: Number of answer areas per sheet.
        ocr_image_dir_pattern: Image directory pattern with {class}.
        ocr_ocr_output_pattern: OCR results YAML output pattern.
        ocr_join_output_pattern: Joined results YAML output pattern.
        ocr_join_forms_csv: CSV fallback for Google Forms data.
        ocr_student_id_column: Column name for student ID in CSV/Sheets.
        ocr_crop_coords: Crop coordinates per question area.
        eval_config: Path to exam config YAML.
        eval_questions_used: Question sn numbers used in evaluation.
        eval_responses_pattern: Joined responses YAML pattern.
        eval_output_pattern: Evaluation output directory pattern.
        eval_skip_feedback: Skip feedback generation.
        eval_skip_graph: Skip graph comparison.
        eval_generate_reports: Generate student PDF reports.
    """

    week: int = 0
    # select section
    select_source: str = ""
    select_questions: list[int] = field(default_factory=list)
    select_num_papers: int = 0
    select_form_url: str = ""
    select_exam_output: str = ""
    # ocr section
    ocr_num_questions: int = 0
    ocr_image_dir_pattern: str = ""
    ocr_ocr_output_pattern: str = ""
    ocr_join_output_pattern: str = ""
    ocr_join_forms_csv: str = ""
    ocr_student_id_column: str = ""
    ocr_crop_coords: list[list[int]] = field(default_factory=list)
    # eval section
    eval_config: str = ""
    eval_questions_used: list[int] = field(default_factory=list)
    eval_responses_pattern: str = ""
    eval_output_pattern: str = ""
    eval_skip_feedback: bool = False
    eval_skip_graph: bool = False
    eval_generate_reports: bool = False


def find_week_config(start_dir: Path | None = None) -> Path | None:
    """Search from start_dir upward for ``week.yaml``.

    Walks the directory tree from ``start_dir`` (default: CWD) upward,
    stopping at the first ``week.yaml`` found.  Stops searching when
    reaching a directory that contains ``forma.yaml`` (project root)
    or the filesystem root.

    Args:
        start_dir: Directory to start searching from. Defaults to CWD.

    Returns:
        Path to the ``week.yaml`` file, or None if not found.
    """
    current = Path(start_dir or Path.cwd()).resolve()
    while True:
        candidate = current / "week.yaml"
        if candidate.is_file():
            return candidate
        # Stop at project root (forma.yaml or .git sentinel)
        if (current / "forma.yaml").is_file() or (current / ".git").exists():
            break
        parent = current.parent
        if parent == current:
            break  # filesystem root
        current = parent
    return None


def load_week_config(path: Path) -> WeekConfiguration:
    """Load and validate a ``week.yaml`` file.

    Args:
        path: Path to the ``week.yaml`` file.

    Returns:
        Populated WeekConfiguration instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
        ValueError: If required fields are missing or invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"week.yaml not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"week.yaml must contain a YAML mapping, got {type(data).__name__}")

    # Validate core field
    if "week" not in data:
        raise ValueError("week.yaml: missing required field 'week'")
    week_val = data["week"]
    if not isinstance(week_val, int) or isinstance(week_val, bool) or week_val < 1:
        raise ValueError(f"week.yaml: 'week' must be an integer >= 1, got {week_val!r}")

    # Build dataclass from nested dict
    config = WeekConfiguration(week=week_val)

    select = data.get("select", {})
    if isinstance(select, dict):
        config.select_source = select.get("source", "")
        config.select_questions = select.get("questions", [])
        config.select_num_papers = select.get("num_papers", 0)
        config.select_form_url = select.get("form_url", "")
        config.select_exam_output = select.get("exam_output", "")

    ocr = data.get("ocr", {})
    if isinstance(ocr, dict):
        config.ocr_num_questions = ocr.get("num_questions", 0)
        config.ocr_image_dir_pattern = ocr.get("image_dir_pattern", "")
        config.ocr_ocr_output_pattern = ocr.get("ocr_output_pattern", "")
        config.ocr_join_output_pattern = ocr.get("join_output_pattern", "")
        config.ocr_join_forms_csv = ocr.get("join_forms_csv", "")
        config.ocr_student_id_column = ocr.get("student_id_column", "")
        config.ocr_crop_coords = ocr.get("crop_coords", [])

    eval_sec = data.get("eval", {})
    if isinstance(eval_sec, dict):
        config.eval_config = eval_sec.get("config", "")
        config.eval_questions_used = eval_sec.get("questions_used", [])
        config.eval_responses_pattern = eval_sec.get("responses_pattern", "")
        config.eval_output_pattern = eval_sec.get("output_pattern", "")
        config.eval_skip_feedback = eval_sec.get("skip_feedback", False)
        config.eval_skip_graph = eval_sec.get("skip_graph", False)
        config.eval_generate_reports = eval_sec.get("generate_reports", False)

    return config


def validate_week_config(
    config_dict: dict,
    required_section: str | None = None,
) -> None:
    """Validate a week configuration dict.

    Checks the ``week`` field and, if ``required_section`` is specified,
    ensures that section's required fields are present and valid.

    Args:
        config_dict: Parsed configuration dict.
        required_section: Section to validate ("select", "ocr", or "eval").

    Raises:
        ValueError: If validation fails.
    """
    errors: list[str] = []

    # week field
    week_val = config_dict.get("week")
    if week_val is None:
        errors.append("missing required field 'week'")
    elif not isinstance(week_val, int) or isinstance(week_val, bool) or week_val < 1:
        errors.append(f"'week' must be an integer >= 1, got {week_val!r}")

    if required_section == "select":
        select = config_dict.get("select", {})
        if not isinstance(select, dict):
            errors.append("'select' section must be a mapping")
        else:
            if not select.get("source"):
                errors.append("select.source is required")
            qs = select.get("questions")
            if not qs or not isinstance(qs, list):
                errors.append("select.questions is required (non-empty list)")
            np = select.get("num_papers", 0)
            if not isinstance(np, int) or isinstance(np, bool) or np < 1:
                errors.append("select.num_papers must be >= 1")

    elif required_section == "ocr":
        ocr = config_dict.get("ocr", {})
        if not isinstance(ocr, dict):
            errors.append("'ocr' section must be a mapping")
        else:
            nq = ocr.get("num_questions")
            if nq is None or (isinstance(nq, int) and not isinstance(nq, bool) and nq < 1):
                if nq is None:
                    errors.append("ocr.num_questions is required")
                else:
                    errors.append("ocr.num_questions must be >= 1")
            elif not isinstance(nq, int) or isinstance(nq, bool):
                errors.append("ocr.num_questions must be an integer")

            if not ocr.get("image_dir_pattern"):
                errors.append("ocr.image_dir_pattern is required")

            # Validate crop_coords format if present
            crop = ocr.get("crop_coords")
            if crop is not None:
                _validate_crop_coords(crop, errors)

    elif required_section == "eval":
        eval_sec = config_dict.get("eval", {})
        if not isinstance(eval_sec, dict):
            errors.append("'eval' section must be a mapping")
        else:
            if not eval_sec.get("config"):
                errors.append("eval.config is required")
            qu = eval_sec.get("questions_used")
            if not qu or not isinstance(qu, list):
                errors.append("eval.questions_used is required (non-empty list)")
            if not eval_sec.get("responses_pattern"):
                errors.append("eval.responses_pattern is required")

    if errors:
        raise ValueError(
            "week.yaml validation errors:\n" + "\n".join(f"  - {e}" for e in errors),
        )


def _validate_crop_coords(crop: Any, errors: list[str]) -> None:
    """Validate crop_coords format.

    Args:
        crop: Value to validate.
        errors: Error list to append to.
    """
    if not isinstance(crop, list):
        errors.append("ocr.crop_coords must be a list")
        return
    for i, entry in enumerate(crop):
        if not isinstance(entry, list) or len(entry) != 4:
            errors.append(
                f"ocr.crop_coords[{i}] must be a list of exactly 4 integers",
            )
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in entry):
            errors.append(f"ocr.crop_coords[{i}] must contain integers only")
            continue
        x1, y1, x2, y2 = entry
        if x1 >= x2:
            errors.append(f"ocr.crop_coords[{i}]: x1 ({x1}) must be < x2 ({x2})")
        if y1 >= y2:
            errors.append(f"ocr.crop_coords[{i}]: y1 ({y1}) must be < y2 ({y2})")


def resolve_class_patterns(
    config: WeekConfiguration,
    class_id: str | None,
) -> WeekConfiguration:
    """Replace ``{class}`` in path-valued fields with ``class_id``.

    Args:
        config: WeekConfiguration with potential {class} patterns.
        class_id: Class identifier to substitute (e.g. "A").

    Returns:
        New WeekConfiguration with resolved patterns.

    Raises:
        ValueError: If any field contains ``{class}`` but class_id is None.
    """
    resolved = copy.copy(config)
    for field_name in _CLASS_PATTERN_FIELDS:
        value = getattr(resolved, field_name, "")
        if not value or "{class}" not in value:
            continue
        if class_id is None:
            raise ValueError(
                f"Field '{field_name}' contains '{{class}}' pattern but "
                f"--class flag was not provided. Use --class to specify the "
                f"class identifier.",
            )
        setattr(resolved, field_name, value.replace("{class}", class_id))
    return resolved


def resolve_paths_relative_to(path_value: str, base_dir: Path) -> str:
    """Resolve a relative path against a base directory.

    Absolute paths are returned unchanged. Relative paths are resolved
    relative to ``base_dir`` (typically the week.yaml directory).

    Args:
        path_value: Path string to resolve.
        base_dir: Base directory for relative path resolution.

    Returns:
        Resolved path as a string.
    """
    p = Path(path_value)
    if p.is_absolute():
        return path_value
    return str(base_dir / path_value)


def save_crop_coords(
    week_yaml_path: Path,
    coords: list[list[int]],
) -> None:
    """Write crop coordinates back to ``week.yaml``.

    Loads the existing file, updates ``ocr.crop_coords``, and writes
    back. Uses atomic write (temp file + rename) to preserve the
    original file on failure.

    Args:
        week_yaml_path: Path to the week.yaml file.
        coords: Crop coordinates to save.

    Raises:
        OSError: If the file cannot be written.
    """
    week_yaml_path = Path(week_yaml_path)
    with open(week_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "ocr" not in data:
        data["ocr"] = {}
    data["ocr"]["crop_coords"] = coords

    # Atomic write: write to temp file, then rename
    parent_dir = week_yaml_path.parent
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=parent_dir, suffix=".yaml.tmp",
        )
        with open(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
        os.replace(tmp_path, str(week_yaml_path))
        logger.info("Crop coordinates saved to %s", week_yaml_path)
    except OSError:
        # Clean up temp file if it exists
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise


def merge_week_configs(
    cli: dict[str, Any],
    week_config: dict[str, Any],
    eval_config: dict[str, Any],
    project_config: dict[str, Any],
    defaults: dict[str, Any],
    *,
    explicit_cli_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Merge five configuration layers into a single resolved dict.

    Precedence: CLI (highest) > week.yaml > eval-config > forma.yaml > defaults.

    Only values explicitly set at each layer override lower layers.
    CLI values are considered "explicitly set" if their key appears in
    ``explicit_cli_keys``.

    Args:
        cli: CLI flag values.
        week_config: Flat dict from week.yaml.
        eval_config: Flat dict from legacy --eval-config file.
        project_config: Flat dict from forma.yaml.
        defaults: Hardcoded default values.
        explicit_cli_keys: Set of CLI keys explicitly provided by user.

    Returns:
        Flat dict with resolved configuration values.
    """
    if explicit_cli_keys is None:
        explicit_cli_keys = set()

    result: dict[str, Any] = dict(defaults)

    # Layer 4: forma.yaml overrides defaults
    for key, value in project_config.items():
        result[key] = value

    # Layer 3: legacy eval-config overrides forma.yaml
    for key, value in eval_config.items():
        result[key] = value

    # Layer 2: week.yaml overrides eval-config
    for key, value in week_config.items():
        result[key] = value

    # Layer 1: CLI explicit flags override everything
    for key in explicit_cli_keys:
        if key in cli:
            result[key] = cli[key]

    return result


def warn_if_class_unknown(
    class_id: str,
    identifiers: list[str],
) -> None:
    """Log a warning if class_id is not in the known identifiers list.

    This is a warning, not an error, to allow ad-hoc class names.

    Args:
        class_id: Class identifier provided via --class flag.
        identifiers: List of known class identifiers from forma.yaml.
    """
    if class_id not in identifiers:
        logger.warning(
            "Class '%s' is not in classes.identifiers %s. "
            "Proceeding with ad-hoc class name.",
            class_id,
            identifiers,
        )
