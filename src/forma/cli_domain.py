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
        description="Extract domain concepts from textbook text",
    )
    parser.add_argument(
        "--textbook",
        type=str,
        action="append",
        default=None,
        help="Textbook chapter text file path (repeatable, mutually exclusive with --summary)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output concepts YAML file path",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum frequency (default: 2, bilingual terms always included)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable concept cache",
    )
    parser.add_argument(
        "--summary",
        type=str,
        action="append",
        default=None,
        help="Chapter summary Markdown file path (repeatable, optional structure guide)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model ID override (default: forma.yaml domain_analysis.extract_model)",
    )
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument(
        "--chunk",
        dest="force_chunk",
        action="store_true",
        default=None,
        help="Force chunk splitting (even for small files)",
    )
    chunk_group.add_argument(
        "--no-chunk",
        dest="force_chunk",
        action="store_false",
        help="Disable chunk splitting (single call even for large files)",
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
            "Error: At least one of --textbook or --summary must be specified.",
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
                f"Error: File not found: {path}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    # Validate summary files if provided separately
    if summary_paths:
        for summary_path in summary_paths:
            if not Path(summary_path).exists():
                logger.warning("Summary file not found: %s", summary_path)

    no_cache = args.no_cache
    force_chunk = args.force_chunk  # None=auto, True=force, False=disable

    n_inputs = len(input_paths)
    if use_llm:
        # If force_chunk is specified, use per-file extraction with chunk control
        if force_chunk is not None:
            concepts_by_chapter = {}
            for i, path_str in enumerate(input_paths):
                chapter_name = Path(path_str).stem
                print(
                    f"[{i + 1}/{n_inputs}] Extracting concepts: {chapter_name}...",
                    file=sys.stderr, flush=True,
                )
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
            print(
                f"Extracting concepts: {n_inputs} chapters...",
                file=sys.stderr, flush=True,
            )
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
        "Concept extraction complete: %d chapters, %d total concepts → %s",
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
        description="Analyze lecture coverage against textbook concepts",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        required=True,
        help="Concepts YAML file (extract output)",
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        required=True,
        action="append",
        help="Lecture transcript file path (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output coverage YAML file path",
    )
    parser.add_argument(
        "--week-config",
        type=str,
        default=None,
        help="Week config YAML (with teaching scope)",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        help='CLI scope override (e.g. "2장:확산,능동수송;3장:")',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Similarity threshold (default: 0.65)",
    )
    parser.add_argument(
        "--eval-store",
        type=str,
        default=None,
        help="Longitudinal data YAML (for formative assessment linking)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model ID override (default: flash)",
    )
    parser.add_argument(
        "--no-pedagogy",
        action="store_true",
        help="Skip pedagogy analysis",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip network graph generation",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM calls (use embedding/term/density signals only)",
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
            f"Error: Concepts file not found: {args.concepts}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    for transcript_path in args.transcripts:
        if not Path(transcript_path).exists():
            print(
                f"Error: Transcript file not found: {transcript_path}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    # Load concepts
    concepts_by_chapter = load_concepts_yaml(args.concepts)
    all_concepts = []
    for chapter_concepts in concepts_by_chapter.values():
        all_concepts.extend(chapter_concepts)

    if not all_concepts:
        print("Error: Concept list is empty.", file=sys.stderr)
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
                f"Error: Week config file not found: {args.week_config}",
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
    n_transcripts = len(args.transcripts)
    for t_idx, transcript_path in enumerate(args.transcripts):
        section_id = _infer_section_from_filename(Path(transcript_path).name)
        print(
            f"[{t_idx + 1}/{n_transcripts}] Analyzing delivery: "
            f"{Path(transcript_path).name} (section {section_id})...",
            file=sys.stderr, flush=True,
        )
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
            print(
                f"  ✓ {len(deliveries)} concepts analyzed",
                file=sys.stderr, flush=True,
            )
        except Exception:
            print(
                "  ✗ Analysis failed",
                file=sys.stderr, flush=True,
            )
            logger.warning(
                "LLM delivery analysis failed: %s", transcript_path, exc_info=True,
            )

    if not all_deliveries:
        print("Error: Delivery analysis failed for all transcripts.", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"Aggregating results: {len(all_deliveries)} delivery analyses...",
        file=sys.stderr, flush=True,
    )

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
            logger.info("Section pairwise comparison: %d pairs computed", len(comparisons))
    except Exception:
        logger.warning("Section pairwise statistical comparison failed", exc_info=True)

    # Save
    save_delivery_yaml(result, args.output)

    print(
        f"Complete: delivery rate {result.effective_delivery_rate * 100:.1f}% → {args.output}",
        file=sys.stderr, flush=True,
    )

    logger.info(
        "Delivery analysis complete: %d concepts, delivery rate %.1f%% -> %s",
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
        description="Generate domain delivery analysis PDF report",
    )
    parser.add_argument(
        "--coverage",
        type=str,
        required=True,
        help="Delivery analysis result YAML (coverage/delivery output)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PDF file path",
    )
    parser.add_argument(
        "--course-name",
        type=str,
        default="",
        help="Course name (displayed in report header)",
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default=None,
        help="Korean font path",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Chart resolution (default: 150)",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=None,
        dest="concepts_file",
        help="Concepts YAML file (for network graph, optional)",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Chapter summary Markdown file path (for hierarchy analysis, optional)",
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
            f"Error: Coverage file not found: {args.coverage}",
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
            logger.info("Hierarchy loaded: %s", args.summary)
        else:
            logger.warning("Summary file not found: %s", args.summary)

    # Build concept network from delivery data (if v2 with concepts)
    concept_network = None
    deliveries_by_section = None
    if is_v2 and hasattr(result, "deliveries") and result.deliveries:
        try:
            from forma.concept_network import build_concept_network
            from forma.domain_concept_extractor import DomainConcept

            # Extract unique concepts from deliveries
            seen = set()
            concepts_for_net = []
            for d in result.deliveries:
                if d.concept not in seen:
                    seen.add(d.concept)
                    concepts_for_net.append(DomainConcept(
                        concept=d.concept,
                        description="",
                        key_terms=getattr(d, "_key_terms", []),
                    ))

            # Try loading concepts file for richer key_terms
            if args.concepts_file:
                from forma.domain_concept_extractor import load_concepts_yaml
                cby = load_concepts_yaml(args.concepts_file)
                concept_map = {}
                for cs in cby.values():
                    for c in cs:
                        concept_map[c.concept] = c
                concepts_for_net = [
                    concept_map.get(c.concept, c) for c in concepts_for_net
                ]

            concept_network = build_concept_network(concepts_for_net)

            # Group deliveries by section
            deliveries_by_section = {}
            for d in result.deliveries:
                if d.section_id not in deliveries_by_section:
                    deliveries_by_section[d.section_id] = []
                deliveries_by_section[d.section_id].append(d)
        except Exception:
            logger.warning("Concept network construction failed", exc_info=True)

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
        concept_network=concept_network,
        deliveries_by_section=deliveries_by_section,
    )

    logger.info("PDF report generated: %s", output_path)
