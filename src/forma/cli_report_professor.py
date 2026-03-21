"""CLI entry point for professor class summary PDF report generator."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from forma.professor_report import ProfessorPDFReportGenerator
from forma.professor_report_data import build_professor_report_data
from forma.professor_report_llm import generate_professor_analysis
from forma.report_data_loader import load_all_student_data

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for forma-report-professor."""
    parser = argparse.ArgumentParser(
        prog="forma-report-professor",
        description="Professor class summary PDF report generator",
    )
    # Required args
    parser.add_argument("--final", required=True, help="Final result YAML file path (anp_*_final.yaml)")
    parser.add_argument("--config", required=True, help="Exam config YAML file path (Ch*_FormativeTest.yaml)")
    parser.add_argument("--eval-dir", required=True, dest="eval_dir", help="Evaluation results directory path")
    parser.add_argument("--output-dir", required=True, dest="output_dir", help="PDF output directory path")
    # Optional args
    parser.add_argument("--forma-config", default=None, dest="forma_config", help="Forma config file path")
    parser.add_argument("--class-name", default=None, dest="class_name", help="Class name (auto-extracted from filename)")
    parser.add_argument("--skip-llm", action="store_true", dest="skip_llm", default=False, help="Skip AI analysis")
    parser.add_argument("--font-path", default=None, dest="font_path", help="Korean font file path")
    parser.add_argument("--dpi", type=int, default=150, help="Chart DPI (default: 150)")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")
    parser.add_argument("--no-config", action="store_true", default=False, dest="no_config",
                        help="Skip forma.yaml config file")
    parser.add_argument("--model", default=None, dest="model_path",
                        help="Drop risk prediction model file path (.pkl)")
    parser.add_argument("--transcript-dir", default=None, dest="transcript_dir",
                        help="Lecture transcript text file directory path")
    parser.add_argument("--longitudinal-store", default=None, dest="longitudinal_store",
                        help="Longitudinal store YAML path (risk group change display)")
    parser.add_argument("--week", type=int, default=None,
                        help="Current week number")
    parser.add_argument("--grade-model", default=None, dest="grade_model_path",
                        help="Grade prediction model file path (.pkl, from forma-train-grade)")
    parser.add_argument("--intervention-log", default=None, dest="intervention_log",
                        help="Intervention log YAML path (enables intervention effect analysis)")
    return parser


def main() -> int | None:
    """Entry point for forma-report-professor CLI."""
    args = _build_parser().parse_args()

    # Apply project config (three-layer merge)
    from forma.project_config import apply_project_config
    apply_project_config(args, argv=sys.argv[1:])

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Validate required input files/dirs exist
    if not os.path.isfile(args.final):
        logger.error("Final result file not found: %s", args.final)
        sys.exit(1)
    if not os.path.isfile(args.config):
        logger.error("Exam config file not found: %s", args.config)
        sys.exit(1)
    if not os.path.isdir(args.eval_dir):
        logger.error("Evaluation results directory not found: %s", args.eval_dir)
        sys.exit(1)

    # Validate longitudinal args
    if args.longitudinal_store and not os.path.isfile(args.longitudinal_store):
        logger.error("Longitudinal store file not found: %s", args.longitudinal_store)
        sys.exit(1)
    if args.week is not None and args.longitudinal_store is None:
        logger.error("--week requires --longitudinal-store")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load student data
    import yaml
    try:
        students, distributions = load_all_student_data(args.final, args.config, args.eval_dir)
    except yaml.YAMLError:
        logger.error("Error: Cannot read YAML file: %s", args.config)
        sys.exit(2)

    # Validate minimum student count
    if len(students) < 3:
        logger.error("Too few students (%d). At least 3 are required.", len(students))
        sys.exit(2)

    # Load exam config metadata for report
    with open(args.config, encoding="utf-8") as _f:
        _exam_cfg = yaml.safe_load(_f) or {}
    _meta = _exam_cfg.get("metadata", {}) if isinstance(_exam_cfg, dict) else {}

    # Build professor report data
    report_data = build_professor_report_data(
        students,
        distributions,
        class_name=args.class_name or "Unknown",
        week_num=_meta.get("week_num", args.week or 0),
        subject=_meta.get("course_name", ""),
        exam_title=_meta.get("chapter_name", "Formative Assessment"),
    )

    # Load longitudinal store ONCE if available (shared by risk movement,
    # risk prediction, grade prediction, and intervention effect analysis)
    long_store = None
    if args.longitudinal_store:
        try:
            from forma.longitudinal_store import LongitudinalStore
            long_store = LongitudinalStore(args.longitudinal_store)
            long_store.load()
        except Exception as exc:
            logger.warning("Failed to load longitudinal store (continuing): %s", exc)

    # v0.10.0: Parse concept dependencies from exam YAML → class deficit map (FR-021)
    concept_dag = None
    deficit_map = None
    deficit_map_chart = None
    try:
        import yaml

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

            # Build per-student concept scores from question concepts
            from forma.learning_path import build_class_deficit_map

            all_students_scores: dict[str, dict[str, float]] = {}
            for student in students:
                scores: dict[str, float] = {}
                for q in student.questions:
                    for c in q.concepts:
                        if c.concept not in scores or c.similarity > scores[c.concept]:
                            scores[c.concept] = c.similarity
                all_students_scores[student.student_id] = scores

            deficit_map = build_class_deficit_map(all_students_scores, concept_dag)
            logger.info("Class concept deficit map built: %d concepts", len(deficit_map.concept_counts))

            # Generate deficit map chart
            try:
                from forma.learning_path_charts import build_deficit_map_chart

                deficit_map_chart = build_deficit_map_chart(deficit_map)
            except Exception as chart_exc:
                logger.warning("Deficit map chart generation failed (continuing): %s", chart_exc)
    except Exception as exc:
        logger.warning("Concept dependency processing failed (continuing): %s", exc)

    # v0.7.3 T013a: Compute class knowledge aggregates from graph comparison data
    try:
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate
        from forma.evaluation_types import GraphComparisonResult

        for qstat in report_data.question_stats:
            qsn = qstat.question_sn
            # Collect comparison results and master edges for this question
            comparison_results = []
            master_edges_set: set[tuple[str, str, str]] = set()

            for student in students:
                for q in student.questions:
                    if q.question_sn != qsn:
                        continue
                    if not q.graph_master_edges:
                        continue

                    for me in q.graph_master_edges:
                        master_edges_set.add((me.subject, me.relation, me.object))

                    comparison_results.append(GraphComparisonResult(
                        student_id=student.student_id,
                        question_sn=qsn,
                        precision=0.0, recall=0.0, f1=q.graph_comparison_f1,
                        matched_edges=q.graph_matched_edges,
                        missing_edges=q.graph_missing_edges,
                        extra_edges=q.graph_extra_edges,
                        wrong_direction_edges=q.graph_wrong_direction_edges,
                    ))

            if comparison_results and master_edges_set:
                from forma.evaluation_types import TripletEdge
                master_edges_list = [
                    TripletEdge(subject=s, relation=r, object=o)
                    for s, r, o in sorted(master_edges_set)
                ]
                agg = build_class_knowledge_aggregate(
                    master_edges_list, comparison_results, qsn,
                )
                report_data.class_knowledge_aggregates.append(agg)
                qstat.class_knowledge_aggregate = agg
    except Exception as exc:
        logger.warning("Class knowledge aggregate graph computation failed (continuing): %s", exc)

    # v0.7.3 T017a: Compute misconception clusters per question
    try:
        from forma.misconception_clustering import cluster_misconceptions

        for qstat in report_data.question_stats:
            classified = getattr(qstat, "classified_misconceptions", [])
            if classified:
                clusters = cluster_misconceptions(classified)
                qstat.misconception_clusters = clusters
                logger.info(
                    "Question %d misconception clustering: %d inputs -> %d clusters",
                    qstat.question_sn, len(classified), len(clusters),
                )
    except Exception as exc:
        logger.warning("Misconception clustering failed (continuing): %s", exc)

    # T042: transcript loading + emphasis/gap computation (FR-019a)
    if args.transcript_dir and os.path.isdir(args.transcript_dir):
        from forma.emphasis_map import compute_emphasis_map
        from forma.lecture_gap_analysis import compute_lecture_gap

        # Load all .txt files from transcript_dir and concatenate
        transcript_lines: list[str] = []
        for fname in sorted(os.listdir(args.transcript_dir)):
            if fname.endswith(".txt"):
                fpath = os.path.join(args.transcript_dir, fname)
                try:
                    with open(fpath, encoding="utf-8") as fh:
                        transcript_lines.extend(fh.read().splitlines())
                except OSError as exc:
                    logger.warning("Failed to read transcript file: %s — %s", fpath, exc)

        if transcript_lines:
            # Gather master concepts from all question concept mastery rates
            master_concepts: set[str] = set()
            for qstat in report_data.question_stats:
                master_concepts.update(qstat.concept_mastery_rates.keys())

            if master_concepts:
                concept_list = sorted(master_concepts)
                sentences = [ln for ln in transcript_lines if ln.strip()]
                try:
                    emphasis_map = compute_emphasis_map(sentences, concept_list)
                    report_data.emphasis_map = emphasis_map
                    logger.info(
                        "Emphasis map generated: %d concepts, %d sentences",
                        emphasis_map.n_concepts, emphasis_map.n_sentences,
                    )

                    # Lecture concepts = concepts with emphasis score > 0
                    lecture_concepts: set[str] = {
                        c for c, score in emphasis_map.concept_scores.items()
                        if score > 0.0
                    }

                    # Student missing rates from concept_mastery_rates
                    student_missing_rates: dict[str, float] = {}
                    for qstat in report_data.question_stats:
                        for concept, mastery_rate in qstat.concept_mastery_rates.items():
                            # missing_rate = 1 - mastery_rate
                            current = student_missing_rates.get(concept, 0.0)
                            student_missing_rates[concept] = max(current, 1.0 - mastery_rate)

                    gap_report = compute_lecture_gap(
                        master_concepts,
                        lecture_concepts,
                        student_missing_rates=student_missing_rates,
                    )
                    report_data.lecture_gap_report = gap_report
                    logger.info(
                        "Lecture gap analysis complete: coverage %.1f%%, missing %d",
                        gap_report.coverage_ratio * 100,
                        len(gap_report.missed_concepts),
                    )
                except Exception as exc:
                    logger.warning("Emphasis/gap analysis failed (continuing): %s", exc)
        else:
            logger.warning("No .txt files found in transcript directory: %s", args.transcript_dir)
    elif args.transcript_dir:
        logger.warning("Transcript directory not found: %s", args.transcript_dir)

    # Conditional LLM analysis
    if not args.skip_llm:
        provider = None
        try:
            import anthropic  # noqa: PLC0415
            provider = anthropic.Anthropic()
        except Exception as exc:
            logger.warning("LLM client creation failed: %s", exc)
        try:
            generate_professor_analysis(provider, report_data)
        except Exception as exc:
            logger.warning("LLM analysis skipped: %s", exc)

        # v0.7.3 T021a: Generate LLM correction points for misconception clusters
        if provider is not None:
            try:
                from forma.professor_report_llm import generate_cluster_correction

                for qstat in report_data.question_stats:
                    for cluster in qstat.misconception_clusters:
                        if not cluster.correction_point:
                            correction = generate_cluster_correction(
                                cluster, cluster.centroid_edge, provider,
                            )
                            cluster.correction_point = correction
            except Exception as exc:
                logger.warning("Misconception cluster correction point generation failed (continuing): %s", exc)

    # Compute risk movement from longitudinal store
    risk_movement = None
    if long_store is not None and args.week is not None:
        try:
            from forma.professor_report_data import compute_risk_movement

            # Current at-risk students from report_data
            current_risk = {
                r.student_id for r in report_data.student_rows if r.is_at_risk
            }

            # Previous week at-risk: find the most recent week before args.week
            previous_risk: set[str] = set()
            all_weeks = sorted({
                d["week"] for d in long_store._records.values()
            })
            prev_weeks = [w for w in all_weeks if w < args.week]
            if prev_weeks:
                prev_week = prev_weeks[-1]
                prev_snapshot = long_store.get_class_snapshot(prev_week)
                # Rebuild at-risk logic for previous week: tier_level 0 = at-risk
                for rec in prev_snapshot:
                    if rec.tier_label in ("Beginning", "Developing"):
                        previous_risk.add(rec.student_id)

            risk_movement = compute_risk_movement(current_risk, previous_risk)
            report_data.risk_movement = risk_movement
            logger.info(
                "Risk movement: new %d, exited %d, persistent %d",
                len(risk_movement.newly_at_risk),
                len(risk_movement.exited_risk),
                len(risk_movement.persistent_risk),
            )
        except Exception as exc:
            logger.warning("Risk movement computation failed (continuing): %s", exc)

    # v0.9.0: Risk prediction from pre-trained model (FR-014, FR-015)
    if args.model_path:
        if not os.path.isfile(args.model_path):
            logger.error("Model file not found: %s", args.model_path)
            sys.exit(1)
        try:
            from forma.risk_predictor import (
                FeatureExtractor, RiskPredictor, load_model,
            )

            trained_model = load_model(args.model_path)
            predictor = RiskPredictor()

            if long_store is not None:
                weeks_list = sorted({
                    r.week for r in long_store.get_all_records()
                })
                extractor = FeatureExtractor()
                matrix, feat_names, student_ids = extractor.extract(
                    long_store, weeks_list,
                )
                if matrix.shape[0] > 0:
                    if feat_names == trained_model.feature_names:
                        preds = predictor.predict(
                            trained_model, matrix, student_ids,
                        )
                    else:
                        logger.warning(
                            "Model feature mismatch — using cold start prediction",
                        )
                        preds = predictor.predict_cold_start(
                            matrix, student_ids, feat_names,
                        )
                    report_data.risk_predictions = preds
                    logger.info("Drop risk prediction complete: %d students", len(preds))
            else:
                logger.info("No longitudinal store — skipping risk prediction")
        except Exception as exc:
            logger.warning("Risk prediction failed (continuing): %s", exc)

    # v0.10.0: Grade prediction from pre-trained grade model (FR-029, FR-030)
    grade_predictions = None
    if args.grade_model_path:
        if not os.path.isfile(args.grade_model_path):
            logger.error("Grade prediction model file not found: %s", args.grade_model_path)
            sys.exit(1)
        try:
            from forma.grade_predictor import (
                GradeFeatureExtractor, GradePredictor, load_grade_model,
            )

            trained_grade_model = load_grade_model(args.grade_model_path)
            grade_predictor = GradePredictor()

            if long_store is not None:
                g_weeks = sorted({r.week for r in long_store.get_all_records()})
                g_extractor = GradeFeatureExtractor()
                g_matrix, g_feat_names, g_student_ids = g_extractor.extract(
                    long_store, g_weeks,
                )
                if g_matrix.shape[0] > 0:
                    if g_feat_names == trained_grade_model.feature_names:
                        grade_predictions = grade_predictor.predict(
                            trained_grade_model, g_matrix, g_student_ids,
                        )
                    else:
                        logger.warning(
                            "Grade model feature mismatch — using cold start prediction",
                        )
                        grade_predictions = grade_predictor.predict_cold_start(
                            g_matrix, g_student_ids, g_feat_names,
                        )
                    report_data.grade_predictions = grade_predictions
                    logger.info("Grade prediction complete: %d students", len(grade_predictions))
            else:
                logger.info("No longitudinal store — skipping grade prediction")
        except Exception as exc:
            logger.warning("Grade prediction failed (continuing): %s", exc)

    # v0.10.0: Intervention effect analysis (FR-008, FR-010, FR-013)
    intervention_effects = None
    intervention_type_summaries = None
    if args.intervention_log:
        if not os.path.isfile(args.intervention_log):
            logger.error("Intervention log file not found: %s", args.intervention_log)
            sys.exit(1)
        try:
            from forma.intervention_effect import (
                compute_intervention_effects,
                compute_type_summary,
            )
            from forma.intervention_store import InterventionLog

            ilog = InterventionLog(args.intervention_log)
            ilog.load()

            if long_store is not None:
                intervention_effects = compute_intervention_effects(ilog, long_store)
                intervention_type_summaries = compute_type_summary(intervention_effects)
                logger.info(
                    "Intervention effect analysis complete: %d records (%d valid)",
                    len(intervention_effects),
                    sum(1 for e in intervention_effects if e.sufficient_data),
                )
            else:
                logger.info("No longitudinal store — skipping intervention effect analysis")
        except Exception as exc:
            logger.warning("Intervention effect analysis failed (continuing): %s", exc)

    # Validate font path if provided
    if args.font_path is not None and not os.path.isfile(args.font_path):
        logger.error("Font file not found: %s", args.font_path)
        sys.exit(3)

    # Generate PDF report
    try:
        ProfessorPDFReportGenerator(font_path=args.font_path, dpi=args.dpi).generate_pdf(
            report_data, args.output_dir, risk_movement=risk_movement,
            grade_predictions=grade_predictions,
            deficit_map=deficit_map,
            deficit_map_chart=deficit_map_chart,
            intervention_effects=intervention_effects,
            intervention_type_summaries=intervention_type_summaries,
        )
    except FileNotFoundError as exc:
        logger.error("File not found during PDF generation: %s", exc)
        sys.exit(3)

    logger.info("Professor report PDF generated: %s", args.output_dir)
    return None
