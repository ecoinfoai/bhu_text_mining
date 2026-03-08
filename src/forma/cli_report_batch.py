"""Batch multi-class report generator CLI."""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import yaml

from forma.report_data_loader import load_all_student_data
from forma.professor_report_data import build_professor_report_data, merge_professor_report_data
from forma.professor_report import ProfessorPDFReportGenerator
from forma.student_report import StudentPDFReportGenerator

logger = logging.getLogger(__name__)


def _load_exam_config(config_path: str) -> dict:
    """Load exam config YAML and return as dict.

    Args:
        config_path: Path to the exam config YAML file.

    Returns:
        Parsed YAML content as a dict, or empty dict on failure.
    """
    try:
        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("Failed to load exam config %s: %s", config_path, exc)
        return {}


def create_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for forma-report-batch."""
    parser = argparse.ArgumentParser(
        description="Generate PDF reports for multiple class sections."
    )
    parser.add_argument("--config", required=True, help="Exam YAML config path")
    parser.add_argument(
        "--join-dir", required=True, dest="join_dir",
        help="Directory with final YAML files",
    )
    parser.add_argument(
        "--join-pattern", required=True, dest="join_pattern",
        help="Pattern with {class} placeholder",
    )
    parser.add_argument(
        "--eval-pattern", required=True, dest="eval_pattern",
        help="Eval dir pattern with {class} placeholder",
    )
    parser.add_argument(
        "--output-dir", required=True, dest="output_dir",
        help="Root output directory",
    )
    parser.add_argument(
        "--classes", required=True, nargs="+",
        help="Class identifiers",
    )
    parser.add_argument(
        "--aggregate", action="store_true", default=False,
        help="Generate merged multi-class professor report",
    )
    parser.add_argument(
        "--no-individual", action="store_true", default=False, dest="no_individual",
        help="Skip student PDFs",
    )
    parser.add_argument(
        "--skip-llm", action="store_true", default=False, dest="skip_llm",
        help="Skip LLM analysis",
    )
    parser.add_argument(
        "--font-path", default=None, dest="font_path",
        help="Path to Korean font",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI")
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Verbose logging",
    )
    return parser


def main(argv=None):
    """Entry point for forma-report-batch CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # ADV-004: Validate {class} placeholder in patterns
    if "{class}" not in args.join_pattern:
        parser.error("--join-pattern must contain {class}")
    if "{class}" not in args.eval_pattern:
        parser.error("--eval-pattern must contain {class}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # AUD-001: Load exam config to extract exam_title, subject, week_num
    exam_config = _load_exam_config(args.config)
    metadata = exam_config.get("metadata", {}) if isinstance(exam_config, dict) else {}
    exam_title = metadata.get("chapter_name", "")
    subject = metadata.get("course_name", "")
    week_num = metadata.get("week_num", 1)

    per_class_reports = []

    for class_id in args.classes:
        # Resolve paths from patterns
        final_filename = args.join_pattern.replace("{class}", class_id)
        final_path = Path(args.join_dir) / final_filename
        eval_dirname = args.eval_pattern.replace("{class}", class_id)
        eval_dir = Path(args.join_dir) / eval_dirname

        # Skip with warning if missing (FR-007)
        if not final_path.exists():
            warnings.warn(
                f"Final YAML not found for class {class_id}: {final_path}"
            )
            logger.warning(
                "Skipping class %s: file not found: %s", class_id, final_path
            )
            continue

        class_output_dir = output_dir / class_id
        class_output_dir.mkdir(parents=True, exist_ok=True)

        # AUD-005: Wrap each class's processing block in try/except
        try:
            # Load data
            students, distributions = load_all_student_data(
                final_path=str(final_path),
                config_path=str(args.config),
                eval_dir=str(eval_dir),
            )

            # Generate student PDFs
            if not args.no_individual:
                student_gen = StudentPDFReportGenerator(
                    font_path=args.font_path,
                    dpi=args.dpi,
                )
                for student_data in students:
                    student_gen.generate_pdf(student_data, distributions, str(class_output_dir))

            # Build professor report data using exam config values (AUD-001)
            report_data = build_professor_report_data(
                students=students,
                distributions=distributions,
                class_name=class_id,
                week_num=week_num,
                subject=subject,
                exam_title=exam_title,
            )

            # Generate professor PDF
            prof_gen = ProfessorPDFReportGenerator(
                font_path=args.font_path,
                dpi=args.dpi,
            )
            prof_gen.generate_pdf(report_data, str(class_output_dir))

            per_class_reports.append(report_data)

        except Exception as exc:
            logger.error(
                "Failed to process class %s: %s", class_id, exc, exc_info=True
            )
            continue

    # ADV-003: Warn when --aggregate is True but not enough classes to merge
    if args.aggregate and len(per_class_reports) <= 1:
        logger.warning(
            "--aggregate requested but only %d class(es) were processed; "
            "aggregate report will not be generated.",
            len(per_class_reports),
        )

    # Aggregate report
    if args.aggregate and len(per_class_reports) > 1:
        merged = merge_professor_report_data(per_class_reports)
        agg_gen = ProfessorPDFReportGenerator(
            font_path=args.font_path,
            dpi=args.dpi,
        )
        agg_gen.generate_pdf(merged, str(output_dir))


if __name__ == "__main__":
    main()
