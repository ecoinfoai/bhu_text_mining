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
    parser.add_argument("--transcript-dir", default=None, dest="transcript_dir",
                        help="강의 녹취록 텍스트 파일 디렉토리 경로")
    return parser


def main() -> int | None:
    """Entry point for forma-report-professor CLI."""
    args = _build_parser().parse_args()

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

    # Validate font path if provided
    if args.font_path is not None and not os.path.isfile(args.font_path):
        _LOG.error("폰트 파일이 존재하지 않습니다: %s", args.font_path)
        sys.exit(3)

    # Generate PDF report
    try:
        ProfessorPDFReportGenerator(font_path=args.font_path, dpi=args.dpi).generate_pdf(
            report_data, args.output_dir
        )
    except FileNotFoundError as exc:
        _LOG.error("PDF 생성 중 파일을 찾을 수 없습니다: %s", exc)
        sys.exit(3)

    _LOG.info("교수 리포트 PDF 생성이 완료되었습니다: %s", args.output_dir)
    return None
