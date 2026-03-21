"""forma-select CLI — question selection and exam PDF generation.

Reads ``week.yaml`` select section, extracts questions from a
FormativeTest YAML file by ``sn`` number, writes ``questions.yaml``
with provenance metadata, and optionally generates an exam PDF.

Usage:
    forma-select
    forma-select --week-config path/to/week.yaml
    forma-select --no-config
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from forma.io_utils import atomic_write_yaml

from forma.exam_generator import ExamPDFGenerator
from forma.week_config import (
    find_week_config,
    load_week_config,
    resolve_paths_relative_to,
    validate_week_config,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build argparse parser for forma-select.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="forma-select",
        description="Formative assessment question selection and exam PDF generation",
    )
    parser.add_argument(
        "--week-config",
        help="week.yaml file path (auto-discovered if not specified)",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Disable week.yaml auto-discovery",
    )
    return parser


def _extract_questions(
    source_path: Path | str,
    sn_list: list[int],
) -> list[dict[str, Any]]:
    """Extract questions from a FormativeTest YAML by sn number.

    Args:
        source_path: Path to the FormativeTest YAML file.
        sn_list: List of question sn numbers to extract.

    Returns:
        List of question dicts in sn order.

    Raises:
        FileNotFoundError: If source file does not exist.
        ValueError: If any sn is not found in the source.
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    with open(source_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict):
        questions_list = data.get("questions", [])
    elif isinstance(data, list):
        questions_list = data
    else:
        raise ValueError(f"FormativeTest YAML format error: {source_path}")

    # Build sn → question lookup
    sn_map: dict[int, dict] = {}
    for q in questions_list:
        if isinstance(q, dict) and "sn" in q:
            sn_map[q["sn"]] = q

    # Extract in requested order
    result: list[dict] = []
    for sn in sn_list:
        if sn not in sn_map:
            raise ValueError(
                f"Question sn={sn} not found in source file: {source_path}",
            )
        result.append(sn_map[sn])

    return result


def _write_questions_yaml(
    questions: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_path: Path | str,
) -> None:
    """Write questions.yaml with provenance metadata.

    Args:
        questions: Extracted question dicts.
        metadata: Provenance metadata (source, selected_sn, week, etc).
        output_path: Path to write questions.yaml.
    """
    output = dict(metadata)
    output["questions"] = questions

    output_path = Path(output_path)
    atomic_write_yaml(output, output_path)

    logger.info("questions.yaml generated: %s", output_path)


def main(argv: list[str] | None = None) -> int:
    """Entry point for forma-select.

    Args:
        argv: CLI arguments. None for sys.argv.

    Returns:
        Exit code (0=success, 1=config error, 2=source not found,
        3=invalid sn, 4=PDF failure).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Discover week.yaml
    if args.week_config:
        week_yaml_path = Path(args.week_config)
    elif not args.no_config:
        week_yaml_path = find_week_config()
    else:
        week_yaml_path = None

    if week_yaml_path is None or not week_yaml_path.is_file():
        logger.error("week.yaml not found.")
        return 1

    # Load and validate
    try:
        config = load_week_config(week_yaml_path)
    except Exception as exc:
        logger.error("Failed to load week.yaml: %s", exc)
        return 1

    # Validate select section
    try:
        with open(week_yaml_path, encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}
        validate_week_config(raw_dict, required_section="select")
    except ValueError as exc:
        logger.error("week.yaml select section validation failed: %s", exc)
        return 1

    week_dir = week_yaml_path.parent

    # Resolve source path relative to week.yaml directory
    source_path = Path(resolve_paths_relative_to(config.select_source, week_dir))

    # Extract questions
    try:
        questions = _extract_questions(source_path, config.select_questions)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except ValueError as exc:
        logger.error("%s", exc)
        return 3

    # Load source metadata for PDF generation
    with open(source_path, encoding="utf-8") as f:
        source_data = yaml.safe_load(f) or {}
    source_metadata = source_data.get("metadata", {}) if isinstance(source_data, dict) else {}

    # Write questions.yaml
    metadata = {
        "source": str(source_path),
        "selected_sn": config.select_questions,
        "week": config.week,
        "num_papers": config.select_num_papers,
        "form_url": config.select_form_url,
    }
    questions_output = week_dir / "questions.yaml"
    _write_questions_yaml(questions, metadata, questions_output)

    # Optional: Generate exam PDF
    exam_output = config.select_exam_output
    if exam_output:
        exam_output_path = resolve_paths_relative_to(exam_output, week_dir)
        try:
            pdf_questions = [
                {"topic": q.get("topic", ""), "text": q.get("question", q.get("text", "")), "limit": q.get("limit", "")}
                for q in questions
            ]
            gen = ExamPDFGenerator()
            gen.create_exam_papers(
                questions=pdf_questions,
                num_papers=config.select_num_papers,
                output_path=exam_output_path,
                form_url_template=config.select_form_url or None,
                week_num=config.week,
                course_name=source_metadata.get("course_name", ""),
                year=source_metadata.get("year", 0),
                grade=source_metadata.get("grade", 0),
                semester=source_metadata.get("semester", 0),
            )
            logger.info("Exam PDF generated: %s", exam_output_path)
        except Exception as exc:
            logger.error("PDF generation failed: %s", exc)
            return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
