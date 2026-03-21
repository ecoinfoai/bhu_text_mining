"""CLI entry points for student longitudinal report generation.

Provides ``main()`` for single-student reports and ``batch_main()``
for bulk generation across all students in an ID CSV file.

Usage::

    forma-report-student --store STORE_YAML --student STUDENT_ID --id-csv ID_CSV --output OUTPUT_PDF
        [--weeks 1 2 3] [--font-path PATH] [--dpi INT] [--no-llm] [--no-config] [--verbose]

Exit codes:
    0 -- success
    1 -- data error (store not found, student not found)
    2 -- output error
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

__all__ = [
    "_build_batch_parser",
    "_build_summary_parser",
    "batch_main",
    "main",
    "summary_main",
]


def _create_llm_provider():
    """Try to create an LLM provider from config or environment.

    Checks forma.json, then forma.yaml, then environment variables.

    Returns:
        LLMProvider instance or None if no API key is available.
    """
    try:
        from forma.llm_provider import create_provider
    except ImportError:
        logger.debug("Cannot import llm_provider module.")
        return None

    # Try forma.json first
    api_key = None
    provider_name = "gemini"
    try:
        import json

        for config_name in ("forma.json", "../forma.json"):
            if os.path.isfile(config_name):
                with open(config_name) as f:
                    config = json.load(f)
                api_key = config.get("google_api_key") or config.get("api_key")
                if config.get("anthropic_api_key"):
                    api_key = config["anthropic_api_key"]
                    provider_name = "anthropic"
                break
    except Exception:
        pass

    # Fall back to environment variables
    if not api_key:
        if os.environ.get("GOOGLE_API_KEY"):
            api_key = os.environ["GOOGLE_API_KEY"]
            provider_name = "gemini"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            api_key = os.environ["ANTHROPIC_API_KEY"]
            provider_name = "anthropic"

    if not api_key:
        return None

    try:
        return create_provider(provider=provider_name, api_key=api_key)
    except Exception as exc:
        logger.warning("Failed to create LLM provider: %s", exc)
        return None


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for forma-report-student."""
    parser = argparse.ArgumentParser(
        prog="forma-report-student",
        description="Student individual longitudinal analysis PDF report generator",
    )
    # Required args
    parser.add_argument(
        "--store", required=True,
        help="Longitudinal store YAML file path",
    )
    parser.add_argument(
        "--student", required=True,
        help="Student ID",
    )
    parser.add_argument(
        "--id-csv", required=True, dest="id_csv",
        help="Student ID-name-class mapping CSV file path",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output PDF file path",
    )
    # Optional args
    parser.add_argument(
        "--weeks", type=int, nargs="+", default=None,
        help="Week list to include (all weeks if omitted)",
    )
    parser.add_argument(
        "--font-path", default=None, dest="font_path",
        help="Korean font file path (auto-detected if omitted)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Chart DPI (default: 150)",
    )
    parser.add_argument(
        "--no-llm", action="store_true", default=False, dest="no_llm",
        help="Generate report with charts only, no LLM calls",
    )
    parser.add_argument(
        "--no-config", action="store_true", default=False, dest="no_config",
        help="Skip forma.yaml config file",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable verbose logging",
    )
    return parser


def main(argv=None) -> int | None:
    """Entry point for forma-report-student CLI.

    Args:
        argv: Command-line arguments (for testing). None uses sys.argv.

    Returns:
        None on success; calls sys.exit() on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Apply project config (three-layer merge) unless --no-config
    if not args.no_config:
        try:
            from forma.project_config import apply_project_config
            raw_argv = argv if argv is not None else sys.argv[1:]
            apply_project_config(args, argv=raw_argv)
        except Exception as exc:
            logger.debug("Failed to apply project config (continuing): %s", exc)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Validate store file
    if not os.path.isfile(args.store):
        logger.error("Longitudinal store file not found: %s", args.store)
        sys.exit(1)

    # Validate optional font file
    if args.font_path and not os.path.isfile(args.font_path):
        logger.error("Font file not found: %s", args.font_path)
        sys.exit(1)

    # Load longitudinal store
    from forma.longitudinal_store import LongitudinalStore

    store = LongitudinalStore(args.store)
    store.load()

    # Determine available weeks
    if args.weeks:
        weeks = sorted(args.weeks)
    else:
        all_records = store.get_all_records()
        weeks = sorted({r.week for r in all_records})

    if not weeks:
        logger.error("No data in longitudinal store.")
        sys.exit(1)

    # Check student exists in store
    history = store.get_student_history(args.student)
    if not history:
        logger.error("Student '%s' not found in longitudinal store.", args.student)
        sys.exit(1)

    # Parse ID CSV for student name and class
    from forma.student_longitudinal_data import (
        build_cohort_distribution,
        build_student_data,
        evaluate_warnings,
        parse_id_csv,
    )

    id_map = parse_id_csv(args.id_csv)
    student_name = None
    class_name = None
    if args.student in id_map:
        student_name, class_name = id_map[args.student]

    # Build cohort distribution
    cohort = build_cohort_distribution(store, weeks)

    # Build student data
    student_data = build_student_data(
        store, args.student, weeks, cohort,
        student_name=student_name,
        class_name=class_name,
    )

    # Evaluate warnings
    warnings, alert_level = evaluate_warnings(student_data, cohort)

    # LLM interpretation
    llm_texts = None
    if not args.no_llm:
        try:
            from forma.student_longitudinal_data import anonymize
            from forma.student_longitudinal_llm import (
                generate_interpretation,
            )

            anon_summary = anonymize(student_data, warnings)

            # Try to create LLM provider from config or environment
            provider = _create_llm_provider()
            if provider is not None:
                llm_texts = generate_interpretation(anon_summary, provider)
                logger.info("LLM interpretation generated")
            else:
                logger.info("Cannot configure LLM provider, skipping LLM interpretation.")
        except Exception as exc:
            logger.warning("LLM interpretation generation failed (continuing): %s", exc)

    # Generate PDF
    from forma.student_longitudinal_report import StudentLongitudinalPDFReportGenerator

    try:
        gen = StudentLongitudinalPDFReportGenerator(
            font_path=args.font_path, dpi=args.dpi,
        )
        result = gen.generate_pdf(
            student_data, cohort, warnings, alert_level, args.output,
            llm_texts=llm_texts,
        )
        logger.info("Student report generated: %s", result)
    except FileNotFoundError as exc:
        logger.error("PDF generation failed: %s", exc)
        sys.exit(2)

    return None


def _build_batch_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for forma-report-student-batch."""
    parser = argparse.ArgumentParser(
        prog="forma-report-student-batch",
        description="Student individual longitudinal PDF report batch generator",
    )
    parser.add_argument(
        "--store", required=True,
        help="Longitudinal store YAML file path",
    )
    parser.add_argument(
        "--id-csv", required=True, dest="id_csv",
        help="Student ID-name-class mapping CSV file path",
    )
    parser.add_argument(
        "--output-dir", required=True, dest="output_dir",
        help="Output PDF directory path",
    )
    parser.add_argument(
        "--weeks", type=int, nargs="+", default=None,
        help="Week list to include (all weeks if omitted)",
    )
    parser.add_argument(
        "--font-path", default=None, dest="font_path",
        help="Korean font file path (auto-detected if omitted)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Chart DPI (default: 150)",
    )
    parser.add_argument(
        "--no-llm", action="store_true", default=False, dest="no_llm",
        help="Generate report with charts only, no LLM calls",
    )
    parser.add_argument(
        "--no-config", action="store_true", default=False, dest="no_config",
        help="Skip forma.yaml config file",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable verbose logging",
    )
    return parser


def batch_main(argv=None) -> int | None:
    """Generate student longitudinal reports in batch.

    Args:
        argv: Command-line arguments (for testing). None uses sys.argv.

    Returns:
        None on success; calls sys.exit() on error.
    """
    parser = _build_batch_parser()
    args = parser.parse_args(argv)

    # Apply project config unless --no-config
    if not args.no_config:
        try:
            from forma.project_config import apply_project_config
            raw_argv = argv if argv is not None else sys.argv[1:]
            apply_project_config(args, argv=raw_argv)
        except Exception as exc:
            logger.debug("Failed to apply project config (continuing): %s", exc)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Validate store file
    if not os.path.isfile(args.store):
        logger.error("Longitudinal store file not found: %s", args.store)
        sys.exit(1)

    # Validate optional font file
    if args.font_path and not os.path.isfile(args.font_path):
        logger.error("Font file not found: %s", args.font_path)
        sys.exit(1)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Load longitudinal store
    from forma.longitudinal_store import LongitudinalStore

    store = LongitudinalStore(args.store)
    store.load()

    # Determine available weeks
    if args.weeks:
        weeks = sorted(args.weeks)
    else:
        all_records = store.get_all_records()
        weeks = sorted({r.week for r in all_records})

    if not weeks:
        logger.error("No data in longitudinal store.")
        sys.exit(1)

    # Parse ID CSV for student name and class
    from forma.student_longitudinal_data import (
        build_cohort_distribution,
        build_student_data,
        evaluate_warnings,
        parse_id_csv,
    )

    id_map = parse_id_csv(args.id_csv)

    # Build cohort distribution ONCE for all students
    cohort = build_cohort_distribution(store, weeks)

    # Get all student IDs from store
    all_records = store.get_all_records()
    student_ids = sorted({r.student_id for r in all_records})

    if not student_ids:
        logger.error("No student data in longitudinal store.")
        sys.exit(1)

    # Prepare PDF generator
    from forma.student_longitudinal_report import StudentLongitudinalPDFReportGenerator

    try:
        gen = StudentLongitudinalPDFReportGenerator(
            font_path=args.font_path, dpi=args.dpi,
        )
    except FileNotFoundError as exc:
        logger.error("PDF generator initialization failed: %s", exc)
        sys.exit(2)

    # Prepare LLM provider (once for all students)
    llm_provider = None
    if not args.no_llm:
        try:
            llm_provider = _create_llm_provider()
            if llm_provider is None:
                logger.info("Cannot configure LLM provider, skipping LLM interpretation.")
        except Exception as exc:
            logger.warning("Failed to create LLM provider (continuing): %s", exc)

    total = len(student_ids)
    success = 0

    for idx, student_id in enumerate(student_ids, start=1):
        try:
            # Lookup name and class
            student_name = None
            class_name = None
            if student_id in id_map:
                student_name, class_name = id_map[student_id]

            # Build student data
            student_data = build_student_data(
                store, student_id, weeks, cohort,
                student_name=student_name,
                class_name=class_name,
            )

            # Evaluate warnings
            warnings, alert_level = evaluate_warnings(student_data, cohort)

            # LLM interpretation
            llm_texts = None
            if llm_provider is not None:
                try:
                    from forma.student_longitudinal_data import anonymize
                    from forma.student_longitudinal_llm import generate_interpretation

                    anon_summary = anonymize(student_data, warnings)
                    llm_texts = generate_interpretation(anon_summary, llm_provider)
                except Exception as exc:
                    logger.warning(
                        "Student %s LLM interpretation failed (continuing): %s", student_id, exc,
                    )

            # Generate PDF
            output_path = os.path.join(args.output_dir, f"{student_id}.pdf")
            gen.generate_pdf(
                student_data, cohort, warnings, alert_level, output_path,
                llm_texts=llm_texts,
            )
            success += 1
            print(f"\r[report] {idx}/{total} ({student_id}) ...", end="", flush=True)
        except Exception as exc:
            logger.error("Student %s report generation failed: %s", student_id, exc)
            print(f"\r[report] {idx}/{total} ({student_id}) FAIL", end="", flush=True)

    print()  # newline after progress
    print(f"[report] Complete: {success}/{total} reports generated")
    return None


def _build_summary_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for forma-report-student-summary."""
    parser = argparse.ArgumentParser(
        prog="forma-report-student-summary",
        description="All-student longitudinal summary PDF report generator (table only)",
    )
    parser.add_argument(
        "--store", required=True,
        help="Longitudinal store YAML file path",
    )
    parser.add_argument(
        "--id-csv", required=True, dest="id_csv",
        help="Student ID-name-class mapping CSV file path",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output PDF file path",
    )
    parser.add_argument(
        "--weeks", type=int, nargs="+", default=None,
        help="Week list to include (all weeks if omitted)",
    )
    parser.add_argument(
        "--course-name", default="", dest="course_name",
        help="Course name (displayed on cover)",
    )
    parser.add_argument(
        "--font-path", default=None, dest="font_path",
        help="Korean font file path (auto-detected if omitted)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI (default: 150)",
    )
    parser.add_argument(
        "--no-config", action="store_true", default=False, dest="no_config",
        help="Skip forma.yaml config file",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable verbose logging",
    )
    return parser


def summary_main(argv=None) -> int | None:
    """Entry point for forma-report-student-summary CLI.

    Generates a single PDF with a tabular overview of all students'
    longitudinal scores, trends, percentiles, and warning levels.

    Args:
        argv: Command-line arguments (for testing). None uses sys.argv.

    Returns:
        None on success; calls sys.exit() on error.
    """
    parser = _build_summary_parser()
    args = parser.parse_args(argv)

    # Apply project config unless --no-config
    if not args.no_config:
        try:
            from forma.project_config import apply_project_config
            raw_argv = argv if argv is not None else sys.argv[1:]
            apply_project_config(args, argv=raw_argv)
        except Exception as exc:
            logger.debug("Failed to apply project config (continuing): %s", exc)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Validate store file
    if not os.path.isfile(args.store):
        logger.error("Longitudinal store file not found: %s", args.store)
        sys.exit(1)

    # Validate optional font file
    if args.font_path and not os.path.isfile(args.font_path):
        logger.error("Font file not found: %s", args.font_path)
        sys.exit(1)

    # Load longitudinal store
    from forma.longitudinal_store import LongitudinalStore

    store = LongitudinalStore(args.store)
    store.load()

    # Determine available weeks
    if args.weeks:
        weeks = sorted(args.weeks)
    else:
        all_records = store.get_all_records()
        weeks = sorted({r.week for r in all_records})

    if not weeks:
        logger.error("No data in longitudinal store.")
        sys.exit(1)

    # Parse ID CSV
    from forma.student_longitudinal_data import (
        build_cohort_distribution,
        parse_id_csv,
    )

    id_map = parse_id_csv(args.id_csv)

    # Build cohort distribution
    cohort = build_cohort_distribution(store, weeks)

    # Build summary rows
    from forma.student_longitudinal_summary import (
        CohortSummaryPDFReportGenerator,
        build_summary_rows,
    )

    rows = build_summary_rows(store, weeks, cohort, id_map)

    # Generate PDF
    try:
        gen = CohortSummaryPDFReportGenerator(
            font_path=args.font_path, dpi=args.dpi,
        )
        result = gen.generate_pdf(
            rows, weeks, args.output, course_name=args.course_name,
        )
        logger.info("Summary report generated: %s", result)
    except FileNotFoundError as exc:
        logger.error("PDF generation failed: %s", exc)
        sys.exit(2)

    return None
