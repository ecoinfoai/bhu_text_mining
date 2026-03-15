"""CLI entry point for forma init — generate forma.yaml configuration template.

Usage::

    forma init [--output PATH] [--force]

Exit codes:
    0 — success
    1 — file exists (no --force)
    2 — write error
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Template with Korean comments and sensible defaults
_TEMPLATE = """\
# forma.yaml — 형성평가 분석 프로젝트 설정 파일 (학기 단위)
# 이 파일은 forma init 으로 생성되었습니다.
# 주차별 설정은 week.yaml에 작성합니다.

project:
  course_name: "{course_name}"    # 과목명
  year: {year}                     # 학년도 (예: 2026)
  semester: {semester}             # 학기 (1 또는 2)
  grade: {grade}                   # 학년 (1 이상)

classes:
  identifiers: {identifiers}      # 분반 목록 (예: [A, B, C, D])

paths:
  longitudinal_store: ""           # 종단 저장소 YAML 경로
  font_path: null                  # 한국어 폰트 경로 (null = 자동 탐색)

ocr:
  naver_config: ""                 # (deprecated) Naver OCR 설정 — LLM Vision 사용 권장
  ocr_model: null                  # OCR 모델 ID (null = gemini-2.5-flash)
  credentials: ""                  # 인증 정보 (환경변수에서 로드)
  spreadsheet_url: ""              # Google Sheets URL

evaluation:
  provider: "gemini"               # LLM 제공자 ("gemini" 또는 "anthropic")
  model: null                      # LLM 모델명 (null = 기본값)
  n_calls: 3                       # LLM 호출 횟수 (1 이상)

reports:
  dpi: 150                         # 차트 해상도 (72-600)

prediction:
  model_path: null                 # 드롭 리스크 예측 모델 경로
"""


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for forma init."""
    parser = argparse.ArgumentParser(
        prog="forma-init",
        description="forma.yaml 설정 파일 템플릿 생성",
    )
    parser.add_argument(
        "--output",
        default="forma.yaml",
        help="출력 파일 경로 (기본값: ./forma.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 파일 덮어쓰기",
    )
    return parser


def main() -> None:
    """Entry point for forma init CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    output_path = Path(args.output)

    # Overwrite protection
    if output_path.exists() and not args.force:
        print(
            f"Error: '{output_path}' already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Interactive prompts
    course_name = input("과목명 (course_name, 예: 인체구조와기능): ").strip()
    year_str = input("학년도 (year, 예: 2026): ").strip()
    semester_str = input("학기 (semester, 1 또는 2): ").strip()
    identifiers_str = input("분반 목록 (쉼표 구분, 예: A,B,C,D): ").strip()

    # Parse inputs with defaults
    year = int(year_str) if year_str.isdigit() else 0
    semester = int(semester_str) if semester_str in ("1", "2") else 0
    grade = 1

    if identifiers_str:
        identifiers = [s.strip() for s in identifiers_str.split(",") if s.strip()]
        identifiers_yaml = "[" + ", ".join(identifiers) + "]"
    else:
        identifiers_yaml = "[]"

    # Generate content
    content = _TEMPLATE.format(
        course_name=course_name,
        year=year,
        semester=semester,
        grade=grade,
        identifiers=identifiers_yaml,
    )

    # Write file
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        print(f"Error: Failed to write '{output_path}': {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Configuration template generated: {output_path}")


if __name__ == "__main__":
    main()
