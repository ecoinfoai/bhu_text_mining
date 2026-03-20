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
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument(
        "--chunk",
        dest="force_chunk",
        action="store_true",
        default=None,
        help="청크 분할 강제 사용 (작은 파일도 청크 분할)",
    )
    chunk_group.add_argument(
        "--no-chunk",
        dest="force_chunk",
        action="store_false",
        help="청크 분할 사용 안 함 (큰 파일도 단일 호출)",
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
        extract_concepts_llm,
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
    force_chunk = args.force_chunk  # None=auto, True=force, False=disable

    if use_llm:
        # If force_chunk is specified, use per-file extraction with chunk control
        if force_chunk is not None:
            concepts_by_chapter = {}
            for i, path_str in enumerate(input_paths):
                chapter_name = Path(path_str).stem
                sp = None
                if summary_paths and i < len(summary_paths):
                    sp = summary_paths[i]
                concepts_by_chapter[chapter_name] = extract_concepts_llm(
                    textbook_path=path_str,
                    summary_path=sp,
                    model=args.model,
                    chapter_name=chapter_name,
                    no_cache=no_cache,
                    force_chunk=force_chunk,
                )
        else:
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
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="LLM 호출 생략 (임베딩/용어/밀도 신호만 사용)",
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
        _infer_section_from_filename,
        analyze_delivery_llm,
        build_delivery_result_v2,
        parse_scope_string,
        parse_teaching_scope,
        save_delivery_yaml,
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

    concept_names = [
        getattr(c, "concept", None) or getattr(c, "name_ko", "")
        for c in all_concepts
    ]

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
        all_chapters = sorted(concepts_by_chapter.keys())
        scope = TeachingScope(chapters=all_chapters, scope_rules={})

    # CLI scope override
    if args.scope:
        scope_rules = parse_scope_string(args.scope)
        scope.scope_rules.update(scope_rules)
        for ch in scope_rules:
            if ch not in scope.chapters:
                scope.chapters.append(ch)

    # T029: Load quality_weights from config
    quality_weights = None
    try:
        from forma.config import get_quality_weights, load_config as _load_cfg
        _cfg = _load_cfg()
        quality_weights = get_quality_weights(_cfg)
    except (FileNotFoundError, ImportError):
        pass

    no_llm = args.no_llm

    # LLM delivery analysis per transcript
    all_deliveries = []
    for transcript_path in args.transcripts:
        section_id = _infer_section_from_filename(Path(transcript_path).name)
        try:
            deliveries = analyze_delivery_llm(
                concepts=concept_names,
                transcript_path=transcript_path,
                section_id=section_id,
                model=args.model,
                no_llm=no_llm,
                quality_weights=quality_weights,
            )
            all_deliveries.extend(deliveries)
            logger.info(
                "전달 분석 완료: %s (분반 %s, %d개 개념)",
                transcript_path, section_id, len(deliveries),
            )
        except Exception:
            logger.warning(
                "LLM 전달 분석 실패: %s", transcript_path, exc_info=True,
            )

    if not all_deliveries:
        print("오류: 모든 녹취록에서 전달 분석에 실패했습니다.", file=sys.stderr)
        raise SystemExit(1)

    # Build v2 result
    result = build_delivery_result_v2(
        deliveries=all_deliveries,
        scope=scope,
        concepts=concept_names,
        week=week,
        chapters=scope.chapters,
    )

    # T047: Compute pairwise section comparisons
    try:
        from forma.domain_coverage_analyzer import (
            compute_delivery_pairwise_comparisons,
        )
        comparisons = compute_delivery_pairwise_comparisons(all_deliveries)
        if comparisons:
            result._section_comparisons = comparisons
            logger.info("분반 간 비교: %d개 쌍 계산 완료", len(comparisons))
    except Exception:
        logger.warning("분반 간 통계 비교 실패", exc_info=True)

    # Save
    save_delivery_yaml(result, args.output)

    logger.info(
        "전달 분석 완료: %d개 개념, 전달률 %.1f%% → %s",
        len(concept_names),
        result.effective_delivery_rate * 100,
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
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="챕터 요약 Markdown 파일 경로 (계층 분석용, 선택적)",
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

    # Parse hierarchy from summary if provided
    hierarchy = None
    if args.summary:
        summary_path = Path(args.summary)
        if summary_path.exists():
            from forma.domain_concept_extractor import parse_summary_hierarchy
            hierarchy = parse_summary_hierarchy(str(summary_path))
            logger.info("계층 구조 로드: %s", args.summary)
        else:
            logger.warning("요약 파일을 찾을 수 없습니다: %s", args.summary)

    # Generate PDF
    generator = DomainDeliveryPDFReportGenerator(
        font_path=args.font_path,
        dpi=args.dpi,
    )

    output_path = generator.generate_pdf(
        result=result,
        output_path=args.output,
        course_name=args.course_name,
        hierarchy=hierarchy,
    )

    logger.info("PDF 보고서 생성 완료: %s", output_path)
