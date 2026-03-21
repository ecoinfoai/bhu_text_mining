"""CLI entry point for student individual PDF report generation.

Usage::

    forma-report --final <YAML> --config <YAML> --eval-dir <DIR> --output-dir <DIR>
                 [--student <ID>] [--font-path <PATH>] [--dpi <INT>] [--verbose]

Exit codes:
    0 — success
    1 — input error (missing file/arg)
    2 — data error (student not found)
    3 — rendering error (font missing, etc.)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

from forma.report_data_loader import load_all_student_data
from forma.student_report import StudentPDFReportGenerator

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for forma-report."""
    parser = argparse.ArgumentParser(
        prog="forma-report",
        description="Generate individual student PDF reports",
    )
    parser.add_argument(
        "--final",
        required=True,
        help="Student response YAML file path (e.g. anp_1A_final.yaml)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Exam config YAML file path (e.g. Ch01_FormativeTest.yaml)",
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Evaluation results directory path (e.g. eval_1A/)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="PDF output directory path",
    )
    parser.add_argument(
        "--student",
        default=None,
        help="Generate for specific student ID only (e.g. S015)",
    )
    parser.add_argument(
        "--font-path",
        default=None,
        help="Korean font file path (auto-detected if not specified)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Chart image resolution (default: 150, range: 72-600)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        dest="no_config",
        help="Skip forma.yaml config file",
    )
    parser.add_argument(
        "--longitudinal-store",
        default=None,
        dest="longitudinal_store",
        help="Longitudinal store YAML path (enables change indicators)",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Current week number (baseline for change comparison)",
    )
    parser.add_argument(
        "--grade-model",
        default=None,
        dest="grade_model_path",
        help="Grade prediction model file path (.pkl, for learning trend display)",
    )
    parser.add_argument(
        "--concept-deps",
        action="store_true",
        default=False,
        dest="concept_deps",
        help="Enable concept dependency learning paths (requires exam YAML definitions)",
    )
    parser.add_argument(
        "--intervention-log",
        default=None,
        dest="intervention_log",
        help="Intervention log YAML path (ignored in student reports, FR-013)",
    )
    return parser


def main() -> None:
    """Main entry point for forma-report CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    # Apply project config (three-layer merge)
    from forma.project_config import apply_project_config
    apply_project_config(args, argv=sys.argv[1:])

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )

    # Validate DPI range
    if not 72 <= args.dpi <= 600:
        print("Error: --dpi must be between 72 and 600", file=sys.stderr)
        sys.exit(1)

    # Validate input files exist
    if not os.path.exists(args.final):
        print(
            f"Error: File not found: {args.final}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(args.config):
        print(
            f"Error: File not found: {args.config}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isdir(args.eval_dir):
        print(
            f"Error: Directory not found: {args.eval_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.font_path and not os.path.exists(args.font_path):
        print(
            f"Error: Font file not found: {args.font_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate longitudinal args
    if args.longitudinal_store and not os.path.exists(args.longitudinal_store):
        print(
            f"Error: Longitudinal store not found: {args.longitudinal_store}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.week is not None and args.longitudinal_store is None:
        print(
            "Error: --week requires --longitudinal-store",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load data
    print(f"Loading evaluation data from {args.eval_dir} ...")
    try:
        students, distributions = load_all_student_data(
            args.final,
            args.config,
            args.eval_dir,
        )
    except yaml.YAMLError as exc:
        print(f"Error: Failed to parse YAML file: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: Failed to load data: {exc}", file=sys.stderr)
        sys.exit(1)

    # Count questions
    q_count = len(students[0].questions) if students else 0
    print(f"  Found {len(students)} students, {q_count} questions each.")

    # Filter by --student if specified
    if args.student:
        filtered = [s for s in students if s.student_id == args.student]
        if not filtered:
            print(
                f"Error: No data found for student {args.student}.",
                file=sys.stderr,
            )
            sys.exit(2)
        students = filtered

    # Load longitudinal store if provided
    long_store = None
    if args.longitudinal_store:
        from forma.longitudinal_store import LongitudinalStore

        long_store = LongitudinalStore(args.longitudinal_store)
        long_store.load()
        logger.info("Loaded longitudinal store: %s", args.longitudinal_store)

    # Parse concept dependencies from exam YAML (FR-015, FR-023)
    concept_dag = None
    if args.concept_deps:
        try:
            from forma.concept_dependency import (
                build_and_validate_dag,
                parse_concept_dependencies,
            )

            with open(args.config, encoding="utf-8") as fh:
                exam_yaml = yaml.safe_load(fh) or {}

            deps = parse_concept_dependencies(exam_yaml)
            if deps:
                concept_dag = build_and_validate_dag(deps)
                logger.info("Concept dependency DAG built: %d nodes", len(concept_dag.nodes))
            else:
                logger.info("No concept_dependencies in exam YAML, skipping learning path")
        except Exception as exc:
            logger.warning("Concept dependency parsing failed (continuing): %s", exc)

    # Load grade model and predict softened tiers (FR-031)
    grade_trend_map: dict[str, str] = {}
    if getattr(args, "grade_model_path", None) and long_store is not None:
        try:
            from forma.grade_predictor import (
                GradeFeatureExtractor, GradePredictor, load_grade_model,
            )

            trained = load_grade_model(args.grade_model_path)
            g_extractor = GradeFeatureExtractor()
            g_weeks = sorted({r.week for r in long_store.get_all_records()})
            g_matrix, g_feat, g_sids = g_extractor.extract(long_store, g_weeks)
            predictor = GradePredictor()
            if g_matrix.shape[0] > 0:
                if g_feat == trained.feature_names:
                    preds = predictor.predict(trained, g_matrix, g_sids)
                else:
                    preds = predictor.predict_cold_start(g_matrix, g_sids, g_feat)
                # Map predicted ordinal to softened tier
                for pred in preds:
                    if pred.predicted_ordinal >= 3:
                        grade_trend_map[pred.student_id] = "상위권"
                    elif pred.predicted_ordinal == 2:
                        grade_trend_map[pred.student_id] = "중위권"
                    else:
                        grade_trend_map[pred.student_id] = "하위권"
                logger.info("Grade trend prediction complete: %d students", len(grade_trend_map))
        except Exception as exc:
            logger.warning("Grade trend prediction failed (continuing): %s", exc)

    # Create report generator
    try:
        generator = StudentPDFReportGenerator(
            font_path=args.font_path,
            dpi=args.dpi,
        )
    except FileNotFoundError as exc:
        print(
            f"Error: Please install NanumGothic font. ({exc})",
            file=sys.stderr,
        )
        sys.exit(3)

    # Generate PDFs
    print("Generating student reports...")
    total = len(students)
    for idx, student in enumerate(students, 1):
        display_name = student.real_name or student.student_id
        filename = os.path.basename(
            generator._make_output_filename(student, args.output_dir),
        )
        print(
            f"  [{idx:>{len(str(total))}}/{total}] "
            f"{student.student_id} ({display_name}) "
            f"→ {filename} ...",
            end="",
        )
        try:
            # Compute longitudinal data if store is available
            weekly_deltas = None
            trajectory_chart = None
            if long_store is not None and args.week is not None:
                from forma.report_data_loader import compute_weekly_delta

                weekly_deltas = {}
                # Overall ensemble score delta
                overall_scores = [
                    q.ensemble_score for q in student.questions
                ]
                if overall_scores:
                    overall_mean = sum(overall_scores) / len(overall_scores)
                    weekly_deltas["overall"] = compute_weekly_delta(
                        student.student_id,
                        args.week,
                        overall_mean,
                        long_store,
                        "ensemble_score",
                    )
                # Per-question deltas
                for q in student.questions:
                    weekly_deltas[q.question_sn] = compute_weekly_delta(
                        student.student_id,
                        args.week,
                        q.ensemble_score,
                        long_store,
                        "ensemble_score",
                    )
                # Trajectory chart
                trajectory = long_store.get_student_trajectory(
                    student.student_id, "ensemble_score",
                )
                if trajectory:
                    weekly_scores = dict(trajectory)
                    # Add current week if not already present
                    if args.week not in weekly_scores and overall_scores:
                        weekly_scores[args.week] = overall_mean
                    trajectory_chart = generator.chart_builder.build_trajectory_bar_chart(
                        weekly_scores, args.week,
                    )

            # Compute learning path if concept DAG available (FR-020, FR-023)
            learning_path = None
            learning_path_chart = None
            if concept_dag is not None:
                from forma.learning_path import generate_learning_path
                from forma.learning_path_charts import build_learning_path_chart

                # Build per-concept scores from student's question concepts
                student_scores: dict[str, float] = {}
                for q in student.questions:
                    for c in q.concepts:
                        # Use max similarity across questions for same concept
                        if c.concept not in student_scores or c.similarity > student_scores[c.concept]:
                            student_scores[c.concept] = c.similarity

                learning_path = generate_learning_path(
                    student.student_id, student_scores, concept_dag,
                )
                if learning_path.ordered_path:
                    try:
                        learning_path_chart = build_learning_path_chart(
                            learning_path, concept_dag,
                        )
                    except Exception as chart_exc:
                        logger.warning(
                            "Learning path chart generation failed (continuing): %s", chart_exc,
                        )

            grade_trend = grade_trend_map.get(student.student_id)
            generator.generate_pdf(
                student, distributions, args.output_dir,
                weekly_deltas=weekly_deltas,
                trajectory_chart=trajectory_chart,
                grade_trend=grade_trend,
                learning_path=learning_path,
                learning_path_chart=learning_path_chart,
            )
            print(" done")
        except Exception as exc:
            print(f" ERROR: {exc}")
            logger.exception("Failed to generate PDF for %s", student.student_id)

    print(f"{total} reports generated in {args.output_dir}/")


if __name__ == "__main__":
    main()
