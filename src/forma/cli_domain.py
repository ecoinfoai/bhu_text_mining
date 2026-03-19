"""CLI entry points for forma domain subcommands.

Provides ``extract_main()`` for textbook concept extraction,
``coverage_main()`` for lecture coverage analysis, and
``report_main()`` for PDF report generation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

__all__ = [
    "extract_main",
    "coverage_main",
    "report_main",
]

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# T015: extract_main
# ----------------------------------------------------------------


def _build_extract_parser() -> argparse.ArgumentParser:
    """Build argument parser for 'forma domain extract' subcommand.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="forma domain extract",
        description="교과서 텍스트에서 도메인 개념 추출",
    )
    parser.add_argument(
        "--textbook",
        type=str,
        action="append",
        default=None,
        help="교과서 챕터 텍스트 파일 경로 (반복 지정 가능, --summary와 택일)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 개념 YAML 파일 경로",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="최소 빈도 (기본값: 2, 영한 병기 용어는 항상 포함)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="개념 캐시 사용 안 함",
    )
    parser.add_argument(
        "--summary",
        type=str,
        action="append",
        default=None,
        help="챕터 요약 Markdown 파일 경로 (반복 지정 가능, 선택적 구조 가이드)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM 모델 ID 오버라이드 (기본: forma.yaml domain_analysis.extract_model)",
    )
    return parser


def extract_main(argv: list[str] | None = None) -> None:
    """Extract domain concepts from textbook text files.

    Uses LLM-based extraction (v2) when --model or --summary is
    provided. Falls back to v1 (KoNLPy word-level) otherwise.

    Args:
        argv: Command-line arguments. Uses sys.argv if None.
    """
    from forma.domain_concept_extractor import (
        extract_multi_chapter,
        extract_multi_chapter_llm,
        save_concepts_yaml,
    )

    parser = _build_extract_parser()
    args = parser.parse_args(argv)

    # Must have at least one of --textbook or --summary
    if not args.textbook and not args.summary:
        print(
            "오류: --textbook 또는 --summary 중 하나 이상 지정해야 합니다.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # When only --summary given, use summary files as textbook input for LLM
    if not args.textbook and args.summary:
        input_paths = args.summary
        summary_paths = None  # summary IS the input, no separate guide
        use_llm = True
    else:
        input_paths = args.textbook
        summary_paths = args.summary
        use_llm = args.model is not None or args.summary is not None

    # Validate input files exist
    for path in input_paths:
        if not Path(path).exists():
            print(
                f"오류: 파일을 찾을 수 없습니다: {path}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    # Validate summary files if provided separately
    if summary_paths:
        for summary_path in summary_paths:
            if not Path(summary_path).exists():
                logger.warning("요약 파일을 찾을 수 없습니다: %s", summary_path)

    no_cache = args.no_cache

    if use_llm:
        concepts_by_chapter = extract_multi_chapter_llm(
            textbook_paths=input_paths,
            summary_paths=summary_paths,
            model=args.model,
            no_cache=no_cache,
        )
    else:
        # v1 fallback: KoNLPy word-level extraction
        use_cache = not no_cache
        concepts_by_chapter = extract_multi_chapter(
            textbook_paths=input_paths,
            min_freq=args.min_freq,
            use_cache=use_cache,
        )

    # Save to YAML
    save_concepts_yaml(concepts_by_chapter, args.output)

    total_concepts = sum(len(cs) for cs in concepts_by_chapter.values())
    logger.info(
        "개념 추출 완료: %d개 챕터, 총 %d개 개념 → %s",
        len(concepts_by_chapter),
        total_concepts,
        args.output,
    )


# ----------------------------------------------------------------
# T028: coverage_main
# ----------------------------------------------------------------


def _build_coverage_parser() -> argparse.ArgumentParser:
    """Build argument parser for 'forma domain coverage' subcommand.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="forma domain coverage",
        description="교과서 개념 대비 강의 커버리지 분석",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        required=True,
        help="개념 목록 YAML 파일 (extract 출력)",
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        required=True,
        action="append",
        help="강의 녹취 파일 경로 (반복 지정 가능)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 커버리지 YAML 파일 경로",
    )
    parser.add_argument(
        "--week-config",
        type=str,
        default=None,
        help="주차 설정 YAML (teaching scope 포함)",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        help='CLI scope 오버라이드 (예: "2장:확산,능동수송;3장:")',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="유사도 임계값 (기본값: 0.65)",
    )
    parser.add_argument(
        "--eval-store",
        type=str,
        default=None,
        help="종단 데이터 YAML (형성평가 연결 분석용)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM 모델 ID 오버라이드 (기본: flash)",
    )
    parser.add_argument(
        "--no-pedagogy",
        action="store_true",
        help="교수법 분석 생략",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="네트워크 그래프 생성 생략",
    )
    return parser


def coverage_main(argv: list[str] | None = None) -> None:
    """Analyze lecture coverage against textbook concepts.

    Args:
        argv: Command-line arguments. Uses sys.argv if None.
    """
    import yaml

    from forma.domain_concept_extractor import load_concepts_yaml
    from forma.domain_coverage_analyzer import (
        TeachingScope,
        build_coverage_result,
        classify_concepts,
        compute_concept_emphasis,
        detect_extra_concepts,
        parse_scope_string,
        parse_teaching_scope,
        save_coverage_yaml,
    )

    parser = _build_coverage_parser()
    args = parser.parse_args(argv)

    # Validate inputs
    if not Path(args.concepts).exists():
        print(
            f"오류: 개념 파일을 찾을 수 없습니다: {args.concepts}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    for transcript_path in args.transcripts:
        if not Path(transcript_path).exists():
            print(
                f"오류: 녹취 파일을 찾을 수 없습니다: {transcript_path}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    # Load concepts
    concepts_by_chapter = load_concepts_yaml(args.concepts)
    all_concepts = []
    for chapter_concepts in concepts_by_chapter.values():
        all_concepts.extend(chapter_concepts)

    if not all_concepts:
        print("오류: 개념 목록이 비어 있습니다.", file=sys.stderr)
        raise SystemExit(1)

    # Build teaching scope
    week = 0
    if args.week_config:
        if not Path(args.week_config).exists():
            print(
                f"오류: 주차 설정 파일을 찾을 수 없습니다: {args.week_config}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        with open(args.week_config, encoding="utf-8") as f:
            week_data = yaml.safe_load(f)
        scope = parse_teaching_scope(week_data)
        week = week_data.get("week", 0)
    else:
        # Default: all chapters in scope, no restrictions
        all_chapters = sorted(concepts_by_chapter.keys())
        scope = TeachingScope(chapters=all_chapters, scope_rules={})

    # CLI scope override
    if args.scope:
        scope_rules = parse_scope_string(args.scope)
        scope.scope_rules.update(scope_rules)
        # Add any chapters from scope override
        for ch in scope_rules:
            if ch not in scope.chapters:
                scope.chapters.append(ch)

    # Compute emphasis
    emphasis_list = compute_concept_emphasis(
        transcript_paths=args.transcripts,
        concepts=all_concepts,
        threshold=args.threshold,
    )

    # Classify concepts
    classified = classify_concepts(all_concepts, emphasis_list, scope)

    # Detect extra concepts
    extras = detect_extra_concepts(
        transcript_paths=args.transcripts,
        concepts=all_concepts,
    )

    # Build result
    result = build_coverage_result(
        classified=classified,
        extras=extras,
        week=week,
        chapters=scope.chapters,
    )

    # Save
    save_coverage_yaml(result, args.output)

    logger.info(
        "커버리지 분석 완료: %d개 개념, 커버리지 %.1f%% → %s",
        result.total_textbook_concepts,
        result.effective_coverage_rate * 100,
        args.output,
    )


# ----------------------------------------------------------------
# T038: report_main
# ----------------------------------------------------------------


def _build_report_parser() -> argparse.ArgumentParser:
    """Build argument parser for 'forma domain report' subcommand.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="forma domain report",
        description="도메인 전달 분석 결과 PDF 보고서 생성",
    )
    parser.add_argument(
        "--coverage",
        type=str,
        required=True,
        help="전달 분석 결과 YAML (coverage/delivery 출력)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 PDF 파일 경로",
    )
    parser.add_argument(
        "--course-name",
        type=str,
        default="",
        help="교과목명 (보고서 헤더에 표시)",
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default=None,
        help="한국어 폰트 경로",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="차트 해상도 (기본값: 150)",
    )
    return parser


def report_main(argv: list[str] | None = None) -> None:
    """Generate PDF report from delivery analysis.

    Supports both v1 (CoverageResult) and v2 (DeliveryResult) YAML.

    Args:
        argv: Command-line arguments. Uses sys.argv if None.
    """
    import yaml

    from forma.domain_coverage_report import DomainDeliveryPDFReportGenerator

    parser = _build_report_parser()
    args = parser.parse_args(argv)

    # Validate input
    if not Path(args.coverage).exists():
        print(
            f"오류: 커버리지 파일을 찾을 수 없습니다: {args.coverage}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Detect YAML version and load accordingly
    with open(args.coverage, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    is_v2 = isinstance(raw, dict) and raw.get("version") == "v2"

    if is_v2:
        from forma.domain_coverage_analyzer import load_delivery_yaml
        result = load_delivery_yaml(args.coverage)
    else:
        from forma.domain_coverage_analyzer import load_coverage_yaml
        result = load_coverage_yaml(args.coverage)

    # Generate PDF
    generator = DomainDeliveryPDFReportGenerator(
        font_path=args.font_path,
        dpi=args.dpi,
    )

    output_path = generator.generate_pdf(
        result=result,
        output_path=args.output,
        course_name=args.course_name,
    )

    logger.info("PDF 보고서 생성 완료: %s", output_path)
