"""bhu-exam CLI — formative exam paper generator.

Usage:
    bhu-exam --questions questions.yaml --num-papers 200 --output exam.pdf
    bhu-exam --questions-json '[{"topic":"T","text":"Q","limit":"50"}]' \\
             --num-papers 30 --output exam.pdf
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml

from src.exam_generator import ExamPDFGenerator


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="bhu-exam",
        description="형성평가 시험지 PDF 생성기",
    )

    # Question source (mutually exclusive)
    q_group = parser.add_mutually_exclusive_group(required=True)
    q_group.add_argument(
        "--questions",
        help="YAML 파일 경로 (문제 목록)",
    )
    q_group.add_argument(
        "--questions-json",
        help="인라인 JSON 문자열 (문제 목록)",
    )

    # Required arguments
    parser.add_argument(
        "--num-papers", type=int, required=True,
        help="시험지 매수",
    )
    parser.add_argument(
        "--output", required=True,
        help="PDF 출력 경로",
    )

    # Optional arguments with defaults
    parser.add_argument("--year", type=int, default=2025, help="학년도")
    parser.add_argument("--grade", type=int, default=1, help="학년")
    parser.add_argument("--semester", type=int, default=2, help="학기")
    parser.add_argument("--course", default="감염미생물학", help="과목명")
    parser.add_argument("--week", type=int, default=3, help="주차")
    parser.add_argument("--form-url", default=None, help="Google Forms URL 템플릿")
    parser.add_argument("--student-ids", nargs="+", default=None, help="학생 ID 목록")
    parser.add_argument("--font-path", default=None, help="폰트 경로")

    return parser.parse_args(argv)


def _load_questions(source: str) -> list[dict]:
    """Load questions from a YAML/YML file path or a JSON string."""
    if source.endswith((".yaml", ".yml")):
        with open(source, encoding="utf-8") as f:
            return yaml.safe_load(f)
    # Treat as JSON string
    return json.loads(source)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for bhu-exam."""
    args = _parse_args(argv)

    # Determine question source
    if args.questions is not None:
        questions = _load_questions(args.questions)
    else:
        questions = _load_questions(args.questions_json)

    generator = ExamPDFGenerator(font_path=args.font_path)
    generator.create_exam_papers(
        questions=questions,
        num_papers=args.num_papers,
        output_path=args.output,
        year=args.year,
        grade=args.grade,
        semester=args.semester,
        course_name=args.course,
        week_num=args.week,
        form_url_template=args.form_url,
        student_ids=args.student_ids,
    )


if __name__ == "__main__":
    main()
