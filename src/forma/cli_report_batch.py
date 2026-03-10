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
    parser.add_argument(
        "--no-config", action="store_true", default=False, dest="no_config",
        help="Skip forma.yaml loading",
    )
    parser.add_argument(
        "--transcript-pattern", default=None, dest="transcript_pattern",
        help="Transcript file pattern with {class} placeholder",
    )
    return parser


def main(argv=None):
    """Entry point for forma-report-batch CLI."""
    import sys as _sys

    parser = create_parser()
    args = parser.parse_args(argv)

    # Apply project config (three-layer merge)
    from forma.project_config import apply_project_config
    raw_argv = argv if argv is not None else _sys.argv[1:]
    apply_project_config(args, argv=raw_argv)

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

            # v0.7.3 T013a: Compute class knowledge aggregates
            try:
                from forma.class_knowledge_aggregate import build_class_knowledge_aggregate
                from forma.evaluation_types import GraphComparisonResult, TripletEdge

                for qstat in report_data.question_stats:
                    qsn = qstat.question_sn
                    comparison_results = []
                    master_edges_set: set[tuple[str, str, str]] = set()

                    for student in students:
                        for q in student.questions:
                            if q.question_sn != qsn or not q.graph_master_edges:
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
                logger.warning("분반 %s 학급 집합 그래프 계산 실패: %s", class_id, exc)

            # v0.7.3 T017a: Compute misconception clusters per question
            try:
                from forma.misconception_clustering import cluster_misconceptions

                for qstat in report_data.question_stats:
                    classified = getattr(qstat, "classified_misconceptions", [])
                    if classified:
                        clusters = cluster_misconceptions(classified)
                        qstat.misconception_clusters = clusters
                        logger.info(
                            "분반 %s 문항 %d 오개념 클러스터링: %d개 입력 -> %d개 클러스터",
                            class_id, qstat.question_sn, len(classified), len(clusters),
                        )
            except Exception as exc:
                logger.warning("분반 %s 오개념 클러스터링 실패: %s", class_id, exc)

            # T044: per-class transcript processing (FR-019b, FR-020)
            if args.transcript_pattern:
                from forma.emphasis_map import compute_emphasis_map
                from forma.lecture_gap_analysis import compute_lecture_gap

                transcript_dir = args.transcript_pattern.replace("{class}", class_id)
                if Path(transcript_dir).is_dir():
                    transcript_lines: list[str] = []
                    for fname in sorted(Path(transcript_dir).iterdir()):
                        if fname.suffix == ".txt":
                            try:
                                transcript_lines.extend(fname.read_text(encoding="utf-8").splitlines())
                            except OSError as exc:
                                logger.warning("트랜스크립트 파일 읽기 실패: %s — %s", fname, exc)

                    if transcript_lines:
                        master_concepts: set[str] = set()
                        for qstat in report_data.question_stats:
                            master_concepts.update(qstat.concept_mastery_rates.keys())

                        if master_concepts:
                            concept_list = sorted(master_concepts)
                            sentences = [ln for ln in transcript_lines if ln.strip()]
                            try:
                                emphasis_map = compute_emphasis_map(sentences, concept_list)
                                report_data.emphasis_map = emphasis_map

                                lecture_concepts: set[str] = {
                                    c for c, score in emphasis_map.concept_scores.items()
                                    if score > 0.0
                                }
                                student_missing_rates: dict[str, float] = {}
                                for qstat in report_data.question_stats:
                                    for concept, mastery_rate in qstat.concept_mastery_rates.items():
                                        current = student_missing_rates.get(concept, 0.0)
                                        student_missing_rates[concept] = max(current, 1.0 - mastery_rate)

                                gap_report = compute_lecture_gap(
                                    master_concepts,
                                    lecture_concepts,
                                    student_missing_rates=student_missing_rates,
                                )
                                report_data.lecture_gap_report = gap_report
                                logger.info(
                                    "분반 %s 강의 갭 분석: 커버리지 %.1f%%",
                                    class_id, gap_report.coverage_ratio * 100,
                                )
                            except Exception as exc:
                                logger.warning("분반 %s 강조도/갭 분석 실패: %s", class_id, exc)
                else:
                    logger.warning(
                        "분반 %s 트랜스크립트 디렉토리 없음: %s", class_id, transcript_dir
                    )

            # v0.7.3 T021a: Generate LLM correction points for misconception clusters
            if not args.skip_llm:
                try:
                    from forma.professor_report_llm import generate_cluster_correction

                    provider = None
                    try:
                        import anthropic  # noqa: PLC0415
                        provider = anthropic.Anthropic()
                    except Exception:
                        pass

                    if provider is not None:
                        for qstat in report_data.question_stats:
                            for cluster in qstat.misconception_clusters:
                                if not cluster.correction_point:
                                    correction = generate_cluster_correction(
                                        cluster, cluster.centroid_edge, provider,
                                    )
                                    cluster.correction_point = correction
                except Exception as exc:
                    logger.warning(
                        "분반 %s 오개념 클러스터 교정 포인트 생성 실패: %s", class_id, exc,
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

        # T060 [US4]: Compute cross-section comparison for aggregate report
        try:
            from forma.section_comparison import (
                CrossSectionReport,
                compute_concept_mastery_by_section,
                compute_pairwise_comparisons,
                compute_section_stats,
            )

            section_scores: dict[str, list[float]] = {}
            section_at_risk: dict[str, set[str]] = {}

            for report_data in per_class_reports:
                sec = report_data.class_name
                scores = [r.overall_ensemble_mean for r in report_data.student_rows]
                at_risk = {
                    r.student_id for r in report_data.student_rows if r.is_at_risk
                }
                section_scores[sec] = scores
                section_at_risk[sec] = at_risk

            stats_list = [
                compute_section_stats(sec, section_scores[sec], section_at_risk[sec])
                for sec in sorted(section_scores.keys())
            ]
            pairwise = compute_pairwise_comparisons(section_scores)

            # Concept mastery: section -> concept -> list of per-student values
            section_concept_data: dict[str, dict[str, list[float]]] = {}
            for report_data in per_class_reports:
                sec = report_data.class_name
                concept_values: dict[str, list[float]] = {}
                for qstat in report_data.question_stats:
                    for concept, rate in qstat.concept_mastery_rates.items():
                        concept_values.setdefault(concept, []).append(rate)
                section_concept_data[sec] = concept_values

            concept_mastery = compute_concept_mastery_by_section(section_concept_data)

            merged.cross_section_report = CrossSectionReport(
                section_stats=stats_list,
                pairwise_comparisons=pairwise,
                concept_mastery_by_section=concept_mastery,
                weekly_interaction=None,
            )
            logger.info(
                "분반 간 비교 분석 완료: %d개 분반, %d개 쌍대 비교",
                len(stats_list), len(pairwise),
            )
        except Exception as exc:
            logger.warning("분반 간 비교 분석 실패: %s", exc)

        agg_gen = ProfessorPDFReportGenerator(
            font_path=args.font_path,
            dpi=args.dpi,
        )
        agg_gen.generate_pdf(merged, str(output_dir))


if __name__ == "__main__":
    main()
