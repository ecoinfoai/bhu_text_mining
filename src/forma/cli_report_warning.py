"""CLI entry point for forma-report-warning -- generate early warning PDF report.

Usage::

    forma-report-warning --final FINAL_YAML --config EXAM_YAML --eval-dir EVAL_DIR --output OUTPUT_PDF
        [--longitudinal-store STORE] [--week INT] [--model MODEL_PKL]
        [--font-path PATH] [--dpi INT] [--verbose] [--no-config]

Exit codes:
    0 -- success
    1 -- data error
    2 -- output error
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for forma-report-warning."""
    parser = argparse.ArgumentParser(
        prog="forma-report-warning",
        description="Early warning PDF report generator",
    )
    # Required args
    parser.add_argument(
        "--final",
        required=True,
        help="Final result YAML file path",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Exam config YAML file path",
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        dest="eval_dir",
        help="Evaluation results directory path",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PDF file path",
    )
    # Optional args
    parser.add_argument(
        "--longitudinal-store",
        default=None,
        dest="longitudinal_store",
        help="Longitudinal store YAML file path (for prediction model)",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Current week number",
    )
    parser.add_argument(
        "--model",
        default=None,
        dest="model_path",
        help="Pre-trained prediction model file path (.pkl)",
    )
    parser.add_argument(
        "--font-path",
        default=None,
        dest="font_path",
        help="Korean font file path (auto-detected if omitted)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Chart DPI (default: 150)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        default=False,
        dest="no_config",
        help="Skip forma.yaml config file",
    )
    return parser


def main(argv=None) -> int | None:
    """Entry point for forma-report-warning CLI.

    Args:
        argv: Command-line arguments (for testing). None uses sys.argv.

    Returns:
        None on success; calls sys.exit() on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Apply project config (three-layer merge)
    from forma.project_config import apply_project_config

    raw_argv = argv if argv is not None else sys.argv[1:]
    apply_project_config(args, argv=raw_argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Validate required input files
    if not os.path.isfile(args.final):
        logger.error("Final result file not found: %s", args.final)
        sys.exit(1)
    if not os.path.isfile(args.config):
        logger.error("Exam config file not found: %s", args.config)
        sys.exit(1)
    if not os.path.isdir(args.eval_dir):
        logger.error("Evaluation results directory not found: %s", args.eval_dir)
        sys.exit(1)

    # Validate optional file args
    if args.model_path and not os.path.isfile(args.model_path):
        logger.error("Model file not found: %s", args.model_path)
        sys.exit(1)
    if args.font_path and not os.path.isfile(args.font_path):
        logger.error("Font file not found: %s", args.font_path)
        sys.exit(1)

    # Load student data
    from forma.report_data_loader import load_all_student_data
    from forma.professor_report_data import build_professor_report_data

    students, distributions = load_all_student_data(args.final, args.config, args.eval_dir)

    # Build professor report data to get at-risk identification
    report_data = build_professor_report_data(
        students,
        distributions,
        class_name="warning",
        week_num=args.week or 1,
        subject="",
        exam_title="",
    )

    # Extract at-risk students from report data
    at_risk_students: dict[str, dict] = {}
    for row in report_data.student_rows:
        if row.is_at_risk:
            at_risk_students[row.student_id] = {
                "is_at_risk": True,
                "reasons": getattr(row, "at_risk_reasons", ["rule_based"]),
            }

    # Extract concept scores per student
    concept_scores: dict[str, dict[str, float]] = {}
    for student in students:
        student_concepts: dict[str, float] = {}
        for q in student.questions:
            if hasattr(q, "concept_scores") and q.concept_scores:
                student_concepts.update(q.concept_scores)
        if student_concepts:
            concept_scores[student.student_id] = student_concepts

    # Extract score trajectories from longitudinal store if available
    score_trajectories: dict[str, list[float]] = {}
    absence_ratios: dict[str, float] = {}

    # Run model predictions if available
    risk_predictions = []
    if args.model_path:
        try:
            from forma.risk_predictor import load_model, FeatureExtractor, RiskPredictor

            trained_model = load_model(args.model_path)

            if args.longitudinal_store and os.path.isfile(args.longitudinal_store):
                from forma.longitudinal_store import LongitudinalStore

                store = LongitudinalStore(args.longitudinal_store)
                store.load()

                weeks = list(range(1, (args.week or 1) + 1))
                extractor = FeatureExtractor()
                feature_matrix, feature_names, student_ids = extractor.extract(
                    store,
                    weeks,
                )

                predictor = RiskPredictor()
                risk_predictions = predictor.predict(
                    trained_model,
                    feature_matrix,
                    student_ids,
                )
                logger.info("Model prediction complete: %d students", len(risk_predictions))
        except Exception as exc:
            logger.warning("Model prediction failed (continuing): %s", exc)

    # Build warning cards
    from forma.warning_report_data import build_warning_data

    warning_cards = build_warning_data(
        at_risk_students=at_risk_students,
        risk_predictions=risk_predictions,
        concept_scores=concept_scores,
        score_trajectories=score_trajectories,
        absence_ratios=absence_ratios,
    )

    logger.info("Warning cards generated: %d", len(warning_cards))

    # Generate PDF
    from forma.warning_report import WarningPDFReportGenerator

    try:
        gen = WarningPDFReportGenerator(font_path=args.font_path, dpi=args.dpi)
        gen.generate_pdf(
            warning_cards,
            args.output,
            class_name=report_data.class_name or "",
        )
    except FileNotFoundError as exc:
        logger.error("PDF generation failed: %s", exc)
        sys.exit(2)

    logger.info("Early warning report generated: %s", args.output)
    return None
