"""CLI entry points for forma lecture subcommands.

Provides ``main_analyze()`` for single-transcript analysis,
``main_compare()`` for same-session cross-section comparison,
and ``main_class_compare()`` for all-session cross-class comparison.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from forma.lecture_preprocessor import preprocess_transcript
from forma.lecture_analyzer import (
    AnalysisResult,
    analyze_transcript,
    save_analysis_result,
    load_analysis_result,
)
from forma.lecture_comparison import (
    compare_sections,
    save_comparison_result,
)
from forma.lecture_merge import merge_analyses, save_merged_analysis
from forma.lecture_report import LectureReportGenerator

logger = logging.getLogger(__name__)


def main_analyze(argv: list[str] | None = None) -> None:
    """Analyze a single lecture transcript.

    Preprocesses the transcript, runs multi-stage analysis,
    generates a YAML cache and PDF report.

    Args:
        argv: Command-line arguments. Uses sys.argv if None.
    """
    parser = argparse.ArgumentParser(
        prog="forma lecture analyze",
        description="Analyze single lecture transcript",
    )
    parser.add_argument("--input", type=str, default=None, help="STT transcript file path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--class", dest="class_id", type=str, required=True, help="Class identifier")
    parser.add_argument("--week", type=int, default=None, help="Week number")
    parser.add_argument("--concepts", type=str, default=None, help="Exam concepts YAML path")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache")
    parser.add_argument("--top-n", type=int, default=50, help="Top keyword count")
    parser.add_argument("--no-triplets", action="store_true", help="Skip triplet extraction")
    parser.add_argument("--extra-stopwords", nargs="*", default=[], help="Additional stopwords")

    args = parser.parse_args(argv)

    # Validate input path
    input_path = args.input
    if input_path is None:
        print("Error: --input argument is required.", file=sys.stderr)
        raise SystemExit(1)

    # Path traversal check
    if "../" in input_path:
        print(f"Error: Path contains '../': {input_path}", file=sys.stderr)
        raise SystemExit(1)

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        raise SystemExit(1)

    # Check empty file
    if input_file.stat().st_size == 0:
        print(f"Error: File is empty: {input_path}", file=sys.stderr)
        raise SystemExit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Week number
    week = args.week if args.week is not None else 0

    # Load concepts if provided
    concepts: list[str] | None = None
    if args.concepts:
        if "../" in args.concepts:
            print(f"Error: Path contains '../': {args.concepts}", file=sys.stderr)
            raise SystemExit(1)
        try:
            with open(args.concepts, encoding="utf-8") as f:
                concepts_data = yaml.safe_load(f)
            if isinstance(concepts_data, dict) and "concepts" in concepts_data:
                concepts = concepts_data["concepts"]
            elif isinstance(concepts_data, list):
                concepts = concepts_data
            else:
                print("Error: Invalid concept YAML format.", file=sys.stderr)
                raise SystemExit(1)
        except yaml.YAMLError as e:
            print(f"Error: Concept YAML parse failed: {e}", file=sys.stderr)
            raise SystemExit(1)

    # Check cache
    cache_path = output_dir / f"analysis_{args.class_id}_w{week}.yaml"
    if not args.no_cache and cache_path.exists():
        try:
            result = load_analysis_result(cache_path)
            logger.info("Loaded analysis result from cache: %s", cache_path)
        except Exception:
            logger.warning("Cache load failed, re-analyzing", exc_info=True)
            result = None
    else:
        result = None

    if result is None:
        # Preprocess
        try:
            cleaned = preprocess_transcript(
                path=str(input_file),
                class_id=args.class_id,
                week=week,
                extra_stopwords=args.extra_stopwords if args.extra_stopwords else None,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            raise SystemExit(1)

        # Analyze
        result = analyze_transcript(
            cleaned=cleaned,
            concepts=concepts,
            top_n=args.top_n,
            no_triplets=args.no_triplets,
            provider=None,
        )

        # Save cache
        save_analysis_result(result, output_dir)

    # Generate PDF report
    try:
        gen = LectureReportGenerator()
        pdf_path = output_dir / f"lecture_report_{args.class_id}_w{week}.pdf"
        gen.generate_analysis_report(result, pdf_path)
        logger.info("PDF report generated: %s", pdf_path)
    except Exception:
        logger.warning("PDF report generation failed", exc_info=True)


def _load_concepts(concepts_path: str | None) -> list[str] | None:
    """Load concepts from a YAML file.

    Args:
        concepts_path: Path to concepts YAML file, or None to skip.

    Returns:
        List of concept strings, or None if not provided.
    """
    if not concepts_path:
        return None
    try:
        with open(concepts_path, encoding="utf-8") as f:
            concepts_data = yaml.safe_load(f)
        if isinstance(concepts_data, dict) and "concepts" in concepts_data:
            return concepts_data["concepts"]
        if isinstance(concepts_data, list):
            return concepts_data
        print("Error: Invalid concept YAML format.", file=sys.stderr)
        raise SystemExit(1)
    except yaml.YAMLError as e:
        print(f"Error: Concept YAML parse failed: {e}", file=sys.stderr)
        raise SystemExit(1)


def main_compare(argv: list[str] | None = None) -> None:
    """Compare lecture transcripts across sections for one week.

    Loads per-section analysis results and generates a cross-section
    comparison report with exclusive keywords, concept gaps, and
    emphasis variance.

    Args:
        argv: Command-line arguments. Uses sys.argv if None.
    """
    parser = argparse.ArgumentParser(
        prog="forma lecture compare",
        description="Compare sections for same session",
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Analysis results directory")
    parser.add_argument("--week", type=int, required=True, help="Week number")
    parser.add_argument("--classes", nargs="+", required=True, help="Class list to compare (at least 2)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--concepts", type=str, default=None, help="Exam concepts YAML path")
    parser.add_argument("--top-n", type=int, default=50, help="Top keyword count")
    args = parser.parse_args(argv)

    # Validate >= 2 classes
    if len(args.classes) < 2:
        print("At least 2 classes are required for comparison.", file=sys.stderr)
        raise SystemExit(2)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)

    # Validate analysis files exist
    for class_id in args.classes:
        path = input_dir / f"analysis_{class_id}_w{args.week}.yaml"
        if not path.exists():
            print(
                f"Error: Analysis result file not found: {path}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    # Load analyses
    analyses: dict[str, AnalysisResult] = {}
    for class_id in args.classes:
        path = input_dir / f"analysis_{class_id}_w{args.week}.yaml"
        analyses[class_id] = load_analysis_result(path)

    # Load concepts
    concepts = _load_concepts(args.concepts)

    # Compare
    comparison = compare_sections(
        analyses,
        concepts=concepts,
        top_n=args.top_n,
        comparison_type="session",
    )

    # Save comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_result(comparison, output_dir, prefix="comparison_session")

    # Generate PDF report
    try:
        gen = LectureReportGenerator()
        sections_label = "_".join(sorted(args.classes))
        pdf_path = output_dir / f"comparison_session_w{args.week}_{sections_label}.pdf"
        gen.generate_comparison_report(comparison, pdf_path)
        logger.info("Comparison PDF report generated: %s", pdf_path)
    except Exception:
        logger.warning("Comparison PDF report generation failed", exc_info=True)


def main_class_compare(argv: list[str] | None = None) -> None:
    """Compare lecture transcripts across classes for all sessions combined.

    Loads per-session analysis results for each class, merges them
    into class-level MergedAnalysis, then compares across classes.

    Args:
        argv: Command-line arguments. Uses sys.argv if None.
    """
    parser = argparse.ArgumentParser(
        prog="forma lecture class-compare",
        description="Compare sections across all sessions",
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Analysis results directory")
    parser.add_argument("--weeks", type=int, nargs="+", required=True, help="Week number list")
    parser.add_argument("--classes", nargs="+", required=True, help="Class list to compare (at least 2)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--concepts", type=str, default=None, help="Exam concepts YAML path")
    parser.add_argument("--top-n", type=int, default=50, help="Top keyword count")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache")
    args = parser.parse_args(argv)

    # Validate >= 2 classes
    if len(args.classes) < 2:
        print("At least 2 classes are required for comparison.", file=sys.stderr)
        raise SystemExit(2)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)

    # Validate all analysis files exist
    for class_id in args.classes:
        for week in args.weeks:
            path = input_dir / f"analysis_{class_id}_w{week}.yaml"
            if not path.exists():
                print(
                    f"Error: Analysis result file not found: {path}",
                    file=sys.stderr,
                )
                raise SystemExit(1)

    # Load concepts
    concepts = _load_concepts(args.concepts)

    # For each class: load per-week analyses -> merge
    merged_analyses: dict[str, AnalysisResult] = {}
    for class_id in args.classes:
        per_week: list[AnalysisResult] = []
        for week in args.weeks:
            path = input_dir / f"analysis_{class_id}_w{week}.yaml"
            per_week.append(load_analysis_result(path))

        merged = merge_analyses(per_week, class_id=class_id)
        save_merged_analysis(merged, output_dir)

        # Create a synthetic AnalysisResult from merged data for comparison
        synthetic = AnalysisResult(
            class_id=class_id,
            week=0,
            keyword_frequencies=merged.combined_keyword_frequencies,
            top_keywords=sorted(
                merged.combined_keyword_frequencies,
                key=merged.combined_keyword_frequencies.get,  # type: ignore[arg-type]
                reverse=True,
            )[:args.top_n],
            network_image_path=None,
            topics=None,
            topic_skipped_reason="merged analysis",
            concept_coverage=None,
            emphasis_scores=None,
            triplets=None,
            triplet_skipped_reason=None,
            sentence_count=0,
            analysis_timestamp="",
        )
        merged_analyses[class_id] = synthetic

    # Compare merged analyses
    comparison = compare_sections(
        merged_analyses,
        concepts=concepts,
        top_n=args.top_n,
        comparison_type="class",
    )

    # Save comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_result(comparison, output_dir, prefix="comparison_class")

    # Generate PDF report
    try:
        gen = LectureReportGenerator()
        sections_label = "_".join(sorted(args.classes))
        pdf_path = output_dir / f"comparison_class_{sections_label}.pdf"
        gen.generate_comparison_report(comparison, pdf_path)
        logger.info("Cross-class comparison PDF report generated: %s", pdf_path)
    except Exception:
        logger.warning("Cross-class comparison PDF report generation failed", exc_info=True)
