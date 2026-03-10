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

_LOG = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for forma-report-professor."""
    parser = argparse.ArgumentParser(
        prog="forma-report-professor",
        description="교수 학급 요약 PDF 리포트 생성기",
    )
    # Required args
    parser.add_argument("--final", required=True, help="최종 결과 YAML 파일 경로 (anp_*_final.yaml)")
    parser.add_argument("--config", required=True, help="시험 설정 YAML 파일 경로 (Ch*_FormativeTest.yaml)")
    parser.add_argument("--eval-dir", required=True, dest="eval_dir", help="평가 결과 디렉토리 경로")
    parser.add_argument("--output-dir", required=True, dest="output_dir", help="PDF 출력 디렉토리 경로")
    # Optional args
    parser.add_argument("--forma-config", default=None, dest="forma_config", help="forma 설정 파일 경로")
    parser.add_argument("--class-name", default=None, dest="class_name", help="학급명 (파일명에서 자동 추출)")
    parser.add_argument("--skip-llm", action="store_true", dest="skip_llm", default=False, help="AI 분석 생략")
    parser.add_argument("--font-path", default=None, dest="font_path", help="한글 폰트 파일 경로")
    parser.add_argument("--dpi", type=int, default=150, help="차트 DPI (기본값: 150)")
    parser.add_argument("--verbose", action="store_true", default=False, help="상세 로그 출력")
    parser.add_argument("--no-config", action="store_true", default=False, dest="no_config",
                        help="forma.yaml 설정 파일 무시")
    parser.add_argument("--model", default=None, dest="model_path",
                        help="드롭 리스크 예측 모델 파일 경로 (.pkl)")
    parser.add_argument("--transcript-dir", default=None, dest="transcript_dir",
                        help="강의 녹취록 텍스트 파일 디렉토리 경로")
    parser.add_argument("--longitudinal-store", default=None, dest="longitudinal_store",
                        help="종단 저장소 YAML 경로 (위험군 변동 표시)")
    parser.add_argument("--week", type=int, default=None,
                        help="현재 주차 번호")
    parser.add_argument("--grade-model", default=None, dest="grade_model_path",
                        help="성적 예측 모델 파일 경로 (.pkl, forma-train-grade 출력)")
    parser.add_argument("--intervention-log", default=None, dest="intervention_log",
                        help="개입 활동 로그 YAML 경로 (개입 효과 분석 활성화)")
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
        _LOG.error("최종 결과 파일이 존재하지 않습니다: %s", args.final)
        sys.exit(1)
    if not os.path.isfile(args.config):
        _LOG.error("시험 설정 파일이 존재하지 않습니다: %s", args.config)
        sys.exit(1)
    if not os.path.isdir(args.eval_dir):
        _LOG.error("평가 결과 디렉토리가 존재하지 않습니다: %s", args.eval_dir)
        sys.exit(1)

    # Validate longitudinal args
    if args.longitudinal_store and not os.path.isfile(args.longitudinal_store):
        _LOG.error("종단 저장소 파일이 존재하지 않습니다: %s", args.longitudinal_store)
        sys.exit(1)
    if args.week is not None and args.longitudinal_store is None:
        _LOG.error("--week 옵션은 --longitudinal-store와 함께 사용해야 합니다.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load student data
    students, distributions = load_all_student_data(args.final, args.config, args.eval_dir)

    # Validate minimum student count
    if len(students) < 3:
        _LOG.error("학생 수가 너무 적습니다 (%d명). 최소 3명 이상 필요합니다.", len(students))
        sys.exit(2)

    # Build professor report data
    report_data = build_professor_report_data(
        students,
        distributions,
        class_name=args.class_name or "Unknown",
        week_num=1,
        subject="과목",
        exam_title="형성평가",
    )

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
            _LOG.info("개념 의존성 DAG 구축 완료: %d개 노드", len(concept_dag.nodes))

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
            _LOG.info("학급 개념 결손 맵 구축 완료: %d개 개념", len(deficit_map.concept_counts))

            # Generate deficit map chart
            try:
                from forma.learning_path_charts import build_deficit_map_chart

                deficit_map_chart = build_deficit_map_chart(deficit_map)
            except Exception as chart_exc:
                _LOG.warning("결손 맵 차트 생성 실패 (계속 진행): %s", chart_exc)
    except Exception as exc:
        _LOG.warning("개념 의존성 처리 실패 (계속 진행): %s", exc)

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
        _LOG.warning("학급 집합 그래프 계산 실패 (계속 진행): %s", exc)

    # v0.7.3 T017a: Compute misconception clusters per question
    try:
        from forma.misconception_clustering import cluster_misconceptions

        for qstat in report_data.question_stats:
            classified = getattr(qstat, "classified_misconceptions", [])
            if classified:
                clusters = cluster_misconceptions(classified)
                qstat.misconception_clusters = clusters
                _LOG.info(
                    "문항 %d 오개념 클러스터링: %d개 입력 -> %d개 클러스터",
                    qstat.question_sn, len(classified), len(clusters),
                )
    except Exception as exc:
        _LOG.warning("오개념 클러스터링 실패 (계속 진행): %s", exc)

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
                    _LOG.warning("트랜스크립트 파일 읽기 실패: %s — %s", fpath, exc)

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
                    _LOG.info(
                        "강조도 맵 생성 완료: %d개 개념, %d개 문장",
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
                    _LOG.info(
                        "강의 갭 분석 완료: 커버리지 %.1f%%, 누락 %d개",
                        gap_report.coverage_ratio * 100,
                        len(gap_report.missed_concepts),
                    )
                except Exception as exc:
                    _LOG.warning("강조도/갭 분석 실패 (계속 진행): %s", exc)
        else:
            _LOG.warning("트랜스크립트 디렉토리에 .txt 파일이 없습니다: %s", args.transcript_dir)
    elif args.transcript_dir:
        _LOG.warning("트랜스크립트 디렉토리가 존재하지 않습니다: %s", args.transcript_dir)

    # Conditional LLM analysis
    if not args.skip_llm:
        provider = None
        try:
            import anthropic  # noqa: PLC0415
            provider = anthropic.Anthropic()
        except Exception as exc:
            _LOG.warning("LLM client creation failed: %s", exc)
        try:
            generate_professor_analysis(provider, report_data)
        except Exception as exc:
            _LOG.warning("LLM analysis skipped: %s", exc)

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
                _LOG.warning("오개념 클러스터 교정 포인트 생성 실패 (계속 진행): %s", exc)

    # Compute risk movement from longitudinal store
    risk_movement = None
    if args.longitudinal_store and args.week is not None:
        try:
            from forma.longitudinal_store import LongitudinalStore
            from forma.professor_report_data import compute_risk_movement

            long_store = LongitudinalStore(args.longitudinal_store)
            long_store.load()

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
            _LOG.info(
                "위험군 변동: 신규 %d, 탈출 %d, 지속 %d",
                len(risk_movement.newly_at_risk),
                len(risk_movement.exited_risk),
                len(risk_movement.persistent_risk),
            )
        except Exception as exc:
            _LOG.warning("위험군 변동 계산 실패 (계속 진행): %s", exc)

    # v0.9.0: Risk prediction from pre-trained model (FR-014, FR-015)
    if args.model_path:
        if not os.path.isfile(args.model_path):
            _LOG.error("모델 파일이 존재하지 않습니다: %s", args.model_path)
            sys.exit(1)
        try:
            from forma.risk_predictor import (
                FeatureExtractor, RiskPredictor, load_model,
            )

            trained_model = load_model(args.model_path)
            predictor = RiskPredictor()

            if args.longitudinal_store:
                from forma.longitudinal_store import LongitudinalStore as LS
                ls = LS(args.longitudinal_store)
                ls.load()
                weeks_list = sorted({
                    r.week for r in ls.get_all_records()
                })
                extractor = FeatureExtractor()
                matrix, feat_names, student_ids = extractor.extract(
                    ls, weeks_list,
                )
                if matrix.shape[0] > 0:
                    if feat_names == trained_model.feature_names:
                        preds = predictor.predict(
                            trained_model, matrix, student_ids,
                        )
                    else:
                        _LOG.warning(
                            "모델 피처 불일치 — cold start 예측 사용",
                        )
                        preds = predictor.predict_cold_start(
                            matrix, student_ids, feat_names,
                        )
                    report_data.risk_predictions = preds
                    _LOG.info("드롭 리스크 예측 완료: %d명", len(preds))
            else:
                _LOG.info("종단 저장소 없음 — 리스크 예측 건너뜀")
        except Exception as exc:
            _LOG.warning("리스크 예측 실패 (계속 진행): %s", exc)

    # v0.10.0: Grade prediction from pre-trained grade model (FR-029, FR-030)
    grade_predictions = None
    if args.grade_model_path:
        if not os.path.isfile(args.grade_model_path):
            _LOG.error("성적 예측 모델 파일이 존재하지 않습니다: %s", args.grade_model_path)
            sys.exit(1)
        try:
            from forma.grade_predictor import (
                GradeFeatureExtractor, GradePredictor, load_grade_model,
            )

            trained_grade_model = load_grade_model(args.grade_model_path)
            grade_predictor = GradePredictor()

            if args.longitudinal_store:
                from forma.longitudinal_store import LongitudinalStore as GLS
                gls = GLS(args.longitudinal_store)
                gls.load()
                g_weeks = sorted({r.week for r in gls.get_all_records()})
                g_extractor = GradeFeatureExtractor()
                g_matrix, g_feat_names, g_student_ids = g_extractor.extract(
                    gls, g_weeks,
                )
                if g_matrix.shape[0] > 0:
                    if g_feat_names == trained_grade_model.feature_names:
                        grade_predictions = grade_predictor.predict(
                            trained_grade_model, g_matrix, g_student_ids,
                        )
                    else:
                        _LOG.warning(
                            "성적 모델 피처 불일치 — cold start 예측 사용",
                        )
                        grade_predictions = grade_predictor.predict_cold_start(
                            g_matrix, g_student_ids, g_feat_names,
                        )
                    report_data.grade_predictions = grade_predictions
                    _LOG.info("성적 예측 완료: %d명", len(grade_predictions))
            else:
                _LOG.info("종단 저장소 없음 — 성적 예측 건너뜀")
        except Exception as exc:
            _LOG.warning("성적 예측 실패 (계속 진행): %s", exc)

    # v0.10.0: Intervention effect analysis (FR-008, FR-010, FR-013)
    intervention_effects = None
    intervention_type_summaries = None
    if args.intervention_log:
        if not os.path.isfile(args.intervention_log):
            _LOG.error("개입 로그 파일이 존재하지 않습니다: %s", args.intervention_log)
            sys.exit(1)
        try:
            from forma.intervention_effect import (
                compute_intervention_effects,
                compute_type_summary,
            )
            from forma.intervention_store import InterventionLog

            ilog = InterventionLog(args.intervention_log)
            ilog.load()

            if args.longitudinal_store:
                from forma.longitudinal_store import LongitudinalStore as ILS
                ils = ILS(args.longitudinal_store)
                ils.load()
                intervention_effects = compute_intervention_effects(ilog, ils)
                intervention_type_summaries = compute_type_summary(intervention_effects)
                _LOG.info(
                    "개입 효과 분석 완료: %d건 (유효 %d건)",
                    len(intervention_effects),
                    sum(1 for e in intervention_effects if e.sufficient_data),
                )
            else:
                _LOG.info("종단 저장소 없음 — 개입 효과 분석 건너뜀")
        except Exception as exc:
            _LOG.warning("개입 효과 분석 실패 (계속 진행): %s", exc)

    # Validate font path if provided
    if args.font_path is not None and not os.path.isfile(args.font_path):
        _LOG.error("폰트 파일이 존재하지 않습니다: %s", args.font_path)
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
        _LOG.error("PDF 생성 중 파일을 찾을 수 없습니다: %s", exc)
        sys.exit(3)

    _LOG.info("교수 리포트 PDF 생성이 완료되었습니다: %s", args.output_dir)
    return None
