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

from forma.report_data_loader import load_all_student_data
from forma.student_report import StudentPDFReportGenerator

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for forma-report."""
    parser = argparse.ArgumentParser(
        prog="forma-report",
        description="학생 개인별 PDF 리포트 생성",
    )
    parser.add_argument(
        "--final",
        required=True,
        help="학생 답변 YAML 파일 경로 (예: anp_1A_final.yaml)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="시험 설정 YAML 파일 경로 (예: Ch01_서론_FormativeTest.yaml)",
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="평가 결과 디렉토리 경로 (예: eval_1A/)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="PDF 출력 디렉토리 경로",
    )
    parser.add_argument(
        "--student",
        default=None,
        help="특정 학생 ID만 생성 (예: S015)",
    )
    parser.add_argument(
        "--font-path",
        default=None,
        help="한국어 폰트 파일 경로 (미지정 시 자동 탐색)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="차트 이미지 해상도 (기본: 150, 범위: 72-600)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )
    parser.add_argument(
        "--longitudinal-store",
        default=None,
        dest="longitudinal_store",
        help="종단 저장소 YAML 경로 (변화 표시 활성화)",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="현재 주차 번호 (변화 비교 기준)",
    )
    return parser


def main() -> None:
    """Main entry point for forma-report CLI."""
    parser = _build_parser()
    args = parser.parse_args()

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
    except Exception as exc:
        print(f"Error: 데이터 로딩 실패: {exc}", file=sys.stderr)
        sys.exit(1)

    # Count questions
    q_count = len(students[0].questions) if students else 0
    print(f"  Found {len(students)} students, {q_count} questions each.")

    # Filter by --student if specified
    if args.student:
        filtered = [s for s in students if s.student_id == args.student]
        if not filtered:
            print(
                f"Error: 학생 {args.student}의 데이터를 찾을 수 없습니다.",
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

    # Create report generator
    try:
        generator = StudentPDFReportGenerator(
            font_path=args.font_path,
            dpi=args.dpi,
        )
    except FileNotFoundError as exc:
        print(
            f"Error: NanumGothic 폰트를 설치하세요. ({exc})",
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

            generator.generate_pdf(
                student, distributions, args.output_dir,
                weekly_deltas=weekly_deltas,
                trajectory_chart=trajectory_chart,
            )
            print(" done")
        except Exception as exc:
            print(f" ERROR: {exc}")
            logger.exception("Failed to generate PDF for %s", student.student_id)

    print(f"{total} reports generated in {args.output_dir}/")


if __name__ == "__main__":
    main()
