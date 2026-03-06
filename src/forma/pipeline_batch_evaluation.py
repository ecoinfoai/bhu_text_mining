"""Batch evaluation pipeline for multiple class sections.

Converts OCR join output files to evaluation input format, then runs
the 4-layer evaluation pipeline for each class section.

Usage:
    uv run python pipeline_batch_evaluation.py \\
        --config exams/Ch01_서론_FormativeTest.yaml \\
        --join-dir results/anp_w1/ \\
        --join-pattern "anp_1{class}_final.yaml" \\
        --output results/anp_w1_eval/ \\
        --classes A B C D \\
        --provider gemini \\
        --generate-reports
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from forma.response_converter import convert_join_file


def run_batch_evaluation(
    config_path: str,
    join_dir: str,
    join_pattern: str,
    output_dir: str,
    classes: list[str],
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    skip_llm: bool = False,
    skip_feedback: bool = False,
    skip_graph: bool = False,
    skip_statistical: bool = False,
    generate_reports: bool = False,
    lecture_transcript: Optional[str] = None,
    longitudinal_store: Optional[str] = None,
    questions_used: Optional[list[int]] = None,
) -> None:
    """Run evaluation pipeline for multiple class sections.

    For each class:
    1. Converts join output to evaluation input format.
    2. Runs the 4-layer evaluation pipeline.
    3. Optionally generates student PDF reports.

    Args:
        config_path: Path to exam YAML config file.
        join_dir: Directory containing join output files.
        join_pattern: Filename pattern with ``{class}`` placeholder.
        output_dir: Root output directory.
        classes: List of class identifiers (e.g., ["A", "B", "C", "D"]).
        provider: LLM provider name.
        api_key: LLM API key override.
        model: LLM model ID override.
        skip_llm: Deprecated, use skip_feedback.
        skip_feedback: If True, skip feedback generation.
        skip_graph: If True, skip triplet extraction/graph comparison.
        skip_statistical: If True, skip Layer 3.
        generate_reports: If True, generate student PDF reports.
        lecture_transcript: Path to lecture transcript file.
        longitudinal_store: Path to longitudinal data store.
    """
    import warnings

    from forma.pipeline_evaluation import run_evaluation_pipeline

    if skip_llm and not skip_feedback:
        warnings.warn(
            "--skip-llm is deprecated, use --skip-feedback instead",
            DeprecationWarning,
            stacklevel=2,
        )
        skip_feedback = True

    for cls in classes:
        print(f"\n{'='*60}")
        print(f"[batch] Processing class {cls}")
        print(f"{'='*60}")

        # Resolve join file path
        join_filename = join_pattern.replace("{class}", cls)
        join_path = os.path.join(join_dir, join_filename)

        if not os.path.isfile(join_path):
            print(f"[batch] WARNING: Join file not found: {join_path}, skipping.")
            continue

        # Create class output directory
        class_dir = os.path.join(output_dir, f"class_{cls}")
        os.makedirs(class_dir, exist_ok=True)

        # Convert join → responses format
        responses_path = os.path.join(class_dir, "responses.yaml")
        print(f"[batch] Converting: {join_path} → {responses_path}")
        convert_join_file(join_path, responses_path, questions_used)

        # Per-class longitudinal store
        cls_longitudinal = None
        if longitudinal_store:
            cls_longitudinal = os.path.join(
                os.path.dirname(longitudinal_store),
                f"class_{cls}_{os.path.basename(longitudinal_store)}",
            )

        # Run evaluation pipeline
        run_evaluation_pipeline(
            config_path=config_path,
            responses_path=responses_path,
            output_dir=class_dir,
            api_key=api_key,
            skip_feedback=skip_feedback,
            skip_graph=skip_graph,
            skip_statistical=skip_statistical,
            provider=provider,
            model=model,
            lecture_transcript=lecture_transcript,
            longitudinal_store=cls_longitudinal,
            generate_reports=generate_reports,
            questions_used=questions_used,
        )

        # Generate PDF reports if requested
        if generate_reports:
            _generate_class_reports(class_dir, config_path)

    print(f"\n[batch] All classes complete. Output: {output_dir}")


def _generate_class_reports(class_dir: str, config_path: str) -> None:
    """Generate student PDF reports for a single class."""
    from forma.evaluation_io import load_evaluation_yaml

    counseling_path = os.path.join(class_dir, "res_lvl4", "counseling_summary.yaml")
    if not os.path.isfile(counseling_path):
        print(f"[batch] No counseling summary found, skipping reports.")
        return

    try:
        from forma.report_generator import StudentReportGenerator

        config_data = load_evaluation_yaml(config_path)
        counseling_data = load_evaluation_yaml(counseling_path)
        reports_dir = os.path.join(class_dir, "reports")

        generator = StudentReportGenerator()
        paths = generator.generate_all_reports(
            counseling_data=counseling_data,
            config_data=config_data,
            output_dir=reports_dir,
        )
        print(f"[batch] Generated {len(paths)} PDF reports in {reports_dir}")
    except Exception as exc:
        print(f"[batch] Report generation failed: {exc}")


def main() -> None:
    """Parse arguments and run the batch evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Batch evaluation pipeline for multiple class sections"
    )
    parser.add_argument(
        "--config", required=True, help="Exam YAML config path"
    )
    parser.add_argument(
        "--join-dir", required=True, help="Directory with join output files"
    )
    parser.add_argument(
        "--join-pattern",
        required=True,
        help='Filename pattern with {class} placeholder (e.g., "anp_1{class}_final.yaml")',
    )
    parser.add_argument(
        "--output", required=True, help="Root output directory"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Class identifiers (e.g., A B C D)",
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        help="LLM provider: gemini (default) or anthropic",
    )
    parser.add_argument(
        "--api-key", default=None, help="LLM API key (overrides env var)"
    )
    parser.add_argument(
        "--model", default=None, help="LLM model ID override"
    )
    parser.add_argument(
        "--skip-feedback",
        action="store_true",
        help="Skip feedback generation",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Deprecated: use --skip-feedback instead",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip triplet extraction and graph comparison",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip Layer 3 statistical analysis",
    )
    parser.add_argument(
        "--lecture-transcript",
        default=None,
        help="Path to lecture transcript file",
    )
    parser.add_argument(
        "--longitudinal-store",
        default=None,
        help="Path to longitudinal data store",
    )
    parser.add_argument(
        "--generate-reports",
        action="store_true",
        help="Generate student PDF reports",
    )
    parser.add_argument(
        "--questions-used",
        nargs="+",
        type=int,
        default=None,
        help="출제 문항의 exam sn 번호를 q 순서대로 지정 (예: 1 3 → q1=sn1, q2=sn3)",
    )
    args = parser.parse_args()

    run_batch_evaluation(
        config_path=args.config,
        join_dir=args.join_dir,
        join_pattern=args.join_pattern,
        output_dir=args.output,
        classes=args.classes,
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        skip_llm=args.skip_llm,
        skip_feedback=args.skip_feedback,
        skip_graph=args.skip_graph,
        skip_statistical=args.skip_stats,
        generate_reports=args.generate_reports,
        lecture_transcript=args.lecture_transcript,
        longitudinal_store=args.longitudinal_store,
        questions_used=args.questions_used,
    )


if __name__ == "__main__":
    main()
