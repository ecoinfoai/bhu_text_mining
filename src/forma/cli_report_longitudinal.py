"""CLI entry point for longitudinal summary PDF report generator."""

from __future__ import annotations

import argparse
import logging
import os
import sys

_LOG = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for forma-report-longitudinal."""
    parser = argparse.ArgumentParser(
        prog="forma-report-longitudinal",
        description="종단 분석 기간 요약 PDF 리포트 생성기",
    )
    parser.add_argument(
        "--store", required=True,
        help="종단 저장소 YAML 파일 경로",
    )
    parser.add_argument(
        "--class-name", required=True, dest="class_name",
        help="분반명 (보고서 표지에 표시)",
    )
    parser.add_argument(
        "--output", required=True,
        help="출력 PDF 경로",
    )
    parser.add_argument(
        "--weeks", type=int, nargs="+", default=None,
        help="포함할 주차 번호 (예: --weeks 1 2 3 4). 생략 시 전체 주차",
    )
    parser.add_argument(
        "--exam-file", default=None, dest="exam_file",
        help="시험 파일 경로 (개념 마스터리 분석 참조용)",
    )
    parser.add_argument(
        "--font-path", default=None, dest="font_path",
        help="한국어 폰트 경로 (생략 시 자동 감지)",
    )
    return parser


def main() -> int | None:
    """Entry point for forma-report-longitudinal CLI."""
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)

    # Validate store file exists
    if not os.path.isfile(args.store):
        _LOG.error("종단 저장소 파일이 존재하지 않습니다: %s", args.store)
        sys.exit(1)

    # Validate exam file if specified
    if args.exam_file and not os.path.isfile(args.exam_file):
        _LOG.error("시험 파일이 존재하지 않습니다: %s", args.exam_file)
        sys.exit(1)

    # Validate font path if specified
    if args.font_path and not os.path.isfile(args.font_path):
        _LOG.error("폰트 파일이 존재하지 않습니다: %s", args.font_path)
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
            _LOG.error("저장소에 레코드가 없습니다.")
            sys.exit(1)
        _LOG.info("자동 감지된 주차: %s", weeks)

    # Build summary data
    from forma.longitudinal_report_data import build_longitudinal_summary
    summary = build_longitudinal_summary(store, weeks, args.class_name)

    # Generate PDF
    from forma.longitudinal_report import LongitudinalPDFReportGenerator
    gen = LongitudinalPDFReportGenerator(font_path=args.font_path)
    output_path = gen.generate(summary, args.output)

    _LOG.info("종단 분석 보고서 생성 완료: %s", output_path)
    return None
