"""bhu-exam CLI — formative exam paper generator.

Usage:
    bhu-exam --config exam.yaml --output exam.pdf
    bhu-exam --config exam.yaml --output exam.pdf --num-papers 50
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
        "--config",
        help="통합 YAML 파일 경로 (메타데이터 + 문제)",
    )
    q_group.add_argument(
        "--questions",
        help="YAML 파일 경로 (문제 목록)",
    )
    q_group.add_argument(
        "--questions-json",
        help="인라인 JSON 문자열 (문제 목록)",
    )

    # num-papers: optional (can come from config YAML)
    parser.add_argument(
        "--num-papers", type=int, default=None,
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


def _load_config(path: str) -> dict:
    """Load a YAML config file.

    - If the top-level is a list (legacy format), wrap it as ``{"questions": data}``.
    - If the top-level is a dict (unified format), validate ``questions`` key exists
      and return as-is.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return {"questions": data}
    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError("YAML must be a list of questions or a dict with a 'questions' key")
    return data


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

    # Mapping: YAML key -> (argparse dest, type-cast)
    _META_MAP: dict[str, tuple[str, type]] = {
        "year": ("year", int),
        "grade": ("grade", int),
        "semester": ("semester", int),
        "course": ("course", str),
        "week": ("week", int),
        "num-papers": ("num_papers", int),
        "form-url": ("form_url", str),
    }

    # Load config / questions
    if args.config is not None:
        cfg = _load_config(args.config)
        questions = cfg["questions"]
        # Merge YAML metadata as defaults; CLI args override
        for yaml_key, (attr, cast) in _META_MAP.items():
            if yaml_key in cfg and getattr(args, attr) is None:
                setattr(args, attr, cast(cfg[yaml_key]))
    elif args.questions is not None:
        questions = _load_questions(args.questions)
    else:
        questions = _load_questions(args.questions_json)

    if args.num_papers is None:
        raise SystemExit("error: --num-papers is required (via CLI or config YAML)")

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
