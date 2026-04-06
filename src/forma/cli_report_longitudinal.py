"""CLI entry point for longitudinal summary PDF report generator."""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for forma-report-longitudinal."""
    parser = argparse.ArgumentParser(
        prog="forma-report-longitudinal",
        description="Longitudinal analysis period summary PDF report generator",
    )
    parser.add_argument(
        "--store",
        required=True,
        help="Longitudinal store YAML file path",
    )
    parser.add_argument(
        "--class-name",
        required=True,
        dest="class_name",
        help="Class name (displayed on report cover)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PDF path",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="+",
        default=None,
        help="Week numbers to include (e.g. --weeks 1 2 3 4). All weeks if omitted",
    )
    parser.add_argument(
        "--exam-file",
        default=None,
        dest="exam_file",
        help="Exam file path (for concept mastery analysis reference)",
    )
    parser.add_argument(
        "--font-path",
        default=None,
        dest="font_path",
        help="Korean font path (auto-detected if omitted)",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        default=False,
        dest="no_config",
        help="Skip forma.yaml config file",
    )
    parser.add_argument(
        "--model",
        default=None,
        dest="model_path",
        help="Drop risk prediction model file path (.pkl)",
    )
    parser.add_argument(
        "--intervention-log",
        default=None,
        dest="intervention_log",
        help="Intervention log YAML path (enables before/after charts)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Filter by class_id (e.g. --classes A B C D)",
    )
    parser.add_argument(
        "--heatmap-layout",
        default=None,
        dest="heatmap_layout",
        help="Heatmap subplot layout as rows:cols (e.g. 1:4, 2:2)",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.45,
        dest="risk_threshold",
        help="Persistent risk cutoff (default 0.45)",
    )
    parser.add_argument(
        "--mastery-top-n",
        type=int,
        default=None,
        dest="mastery_top_n",
        help=("Show only top N concepts in mastery chart (ranked by absolute change)"),
    )
    return parser


def parse_heatmap_layout(value: str) -> tuple[int, int]:
    """Parse heatmap layout string 'rows:cols'.

    Args:
        value: String in 'rows:cols' format (e.g. '1:4').

    Returns:
        Tuple of (rows, cols).

    Raises:
        ValueError: If format is invalid or values are non-positive.
    """
    if ":" not in value:
        raise ValueError(f"Invalid layout '{value}': use rows:cols format")
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid layout '{value}': exactly one ':' required")
    try:
        rows, cols = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid layout '{value}': non-integer values")
    if rows < 1 or cols < 1:
        raise ValueError(f"Invalid layout '{value}': values must be positive")
    return (rows, cols)


def main() -> int | None:
    """Entry point for forma-report-longitudinal CLI."""
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)

    # Apply project config (three-layer merge)
    from forma.project_config import apply_project_config

    apply_project_config(args, argv=sys.argv[1:])

    # Validate store file exists
    if not os.path.isfile(args.store):
        logger.error("Longitudinal store file not found: %s", args.store)
        sys.exit(1)

    # Validate exam file if specified
    if args.exam_file and not os.path.isfile(args.exam_file):
        logger.error("Exam file not found: %s", args.exam_file)
        sys.exit(1)

    # Validate font path if specified
    if args.font_path and not os.path.isfile(args.font_path):
        logger.error("Font file not found: %s", args.font_path)
        sys.exit(1)

    # Load store
    from forma.longitudinal_store import LongitudinalStore

    store = LongitudinalStore(args.store)
    store.load()

    # Determine weeks
    weeks = args.weeks
    if weeks is None:
        # Auto-detect all weeks from store
        all_records = store.get_all_records()
        weeks = sorted({r.week for r in all_records})
        if not weeks:
            logger.error("Store contains no records.")
            sys.exit(1)
        logger.info("Auto-detected weeks: %s", weeks)

    # Build summary data
    from forma.longitudinal_report_data import build_longitudinal_summary

    summary = build_longitudinal_summary(
        store,
        weeks,
        args.class_name,
        class_ids=args.classes,
    )

    # v0.9.0: Risk prediction from pre-trained model (FR-014, FR-015)
    if args.model_path:
        if not os.path.isfile(args.model_path):
            logger.error("Model file not found: %s", args.model_path)
            sys.exit(1)
        try:
            from forma.risk_predictor import (
                FeatureExtractor,
                RiskPredictor,
                load_model,
            )

            trained_model = load_model(args.model_path)
            predictor = RiskPredictor()
            extractor = FeatureExtractor()
            matrix, feat_names, student_ids = extractor.extract(store, weeks)
            if matrix.shape[0] > 0:
                if feat_names == trained_model.feature_names:
                    preds = predictor.predict(
                        trained_model,
                        matrix,
                        student_ids,
                    )
                else:
                    logger.warning("Model feature mismatch — using cold start prediction")
                    preds = predictor.predict_cold_start(
                        matrix,
                        student_ids,
                        feat_names,
                    )
                summary.risk_predictions = preds
                logger.info("Drop risk prediction complete: %d students", len(preds))
        except Exception as exc:
            logger.warning("Risk prediction failed (continuing): %s", exc)

    # v0.10.0: Intervention effect analysis (FR-008, FR-011)
    intervention_effects = None
    if args.intervention_log:
        if not os.path.isfile(args.intervention_log):
            logger.error("Intervention log file not found: %s", args.intervention_log)
            sys.exit(1)
        try:
            from forma.intervention_effect import compute_intervention_effects
            from forma.intervention_store import InterventionLog

            ilog = InterventionLog(args.intervention_log)
            ilog.load()
            intervention_effects = compute_intervention_effects(ilog, store)
            logger.info(
                "Intervention effect analysis complete: %d records (%d valid)",
                len(intervention_effects),
                sum(1 for e in intervention_effects if e.sufficient_data),
            )
        except Exception as exc:
            logger.warning("Intervention effect analysis failed (continuing): %s", exc)

    # Per-class heatmap data (US5)
    class_data = None
    class_ids = None
    heatmap_layout = None
    if args.classes:
        class_ids = args.classes
        matrix = store.get_class_weekly_matrix(
            "ensemble_score",
        )
        all_records = store.get_all_records()
        # Map student → class_id
        sid_class: dict[str, str] = {}
        for rec in all_records:
            if rec.class_id and rec.student_id:
                sid_class[rec.student_id] = rec.class_id
        # Build {class_id: {student_id: {week: score}}}
        class_data = {}
        for cid in class_ids:
            class_data[cid] = {
                sid: {w: ws[w] for w in weeks if w in ws} for sid, ws in matrix.items() if sid_class.get(sid) == cid
            }
        if args.heatmap_layout:
            heatmap_layout = parse_heatmap_layout(
                args.heatmap_layout,
            )
        else:
            heatmap_layout = (1, len(class_ids))

    # Generate PDF
    from forma.longitudinal_report import LongitudinalPDFReportGenerator

    gen = LongitudinalPDFReportGenerator(font_path=args.font_path)
    output_path = gen.generate_pdf(
        summary,
        args.output,
        intervention_effects=intervention_effects,
        class_data=class_data,
        class_ids=class_ids,
        heatmap_layout=heatmap_layout,
        mastery_top_n=args.mastery_top_n,
    )

    logger.info("Longitudinal analysis report generated: %s", output_path)
    return None
