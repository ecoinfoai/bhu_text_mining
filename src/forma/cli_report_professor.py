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
