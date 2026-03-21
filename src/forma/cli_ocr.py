"""forma-ocr CLI — OCR pipeline for scanned exam answer sheets.

Usage:
    forma-ocr scan --config ocr_config.yaml
    forma-ocr scan --class A
    forma-ocr scan --class A --recrop

    forma-ocr join --ocr-results results.yaml \\
                   --output final.yaml \\
                   --spreadsheet-url "https://docs.google.com/spreadsheets/d/XXX" \\
                   [--forms-csv fallback.csv] \\
                   [--credentials credentials.json] \\
                   [--manual-mapping mapping.yaml] \\
                   [--student-id-column "sid"]

    forma-ocr join --class A

    forma-ocr compare --image scan.jpg --provider gemini
"""
from __future__ import annotations

import argparse
import logging
import sys

import yaml

from forma.ocr_pipeline import run_join_pipeline, run_scan_pipeline

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse forma-ocr CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="forma-ocr",
        description="OCR pipeline for scanned answer sheets",
    )
    parser.add_argument(
        "--no-config", action="store_true", default=False, dest="no_config",
        help="Skip forma.yaml config file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── scan subcommand ───────────────────────────
    scan_p = subparsers.add_parser(
        "scan",
        help="Image scan -> QR decode -> OCR -> YAML",
    )
    scan_source = scan_p.add_mutually_exclusive_group(required=False)
    scan_source.add_argument(
        "--config",
        help="OCR config YAML file path (legacy, deprecated)",
    )
    scan_source.add_argument(
        "--class", dest="class_id",
        help="Class identifier ({class} pattern in week.yaml)",
    )
    scan_p.add_argument(
        "--provider", dest="provider", default="gemini",
        help="LLM provider (gemini or anthropic, default: gemini)",
    )
    scan_p.add_argument(
        "--model", default=None,
        help="LLM model ID override",
    )
    scan_p.add_argument(
        "--subject", default=None,
        help="Subject name (LLM prompt context)",
    )
    scan_p.add_argument(
        "--question", default=None,
        help="Question text (LLM prompt context)",
    )
    scan_p.add_argument(
        "--answer-keywords", default=None, dest="answer_keywords",
        help="Key terms (comma-separated, LLM prompt context)",
    )
    scan_p.add_argument(
        "--num-questions", type=int, default=None,
        help="Number of questions (can use config YAML num-questions value)",
    )
    scan_p.add_argument(
        "--recrop", action="store_true", default=False,
        help="Ignore saved crop coordinates, re-select",
    )
    scan_p.add_argument(
        "--week-config", default=None, dest="week_config",
        help="week.yaml path (default: auto-discover from current directory)",
    )
    scan_p.add_argument(
        "--ocr-review-threshold", type=float, default=None,
        dest="ocr_review_threshold",
        help="OCR confidence review threshold (default: 0.75)",
    )

    # ── join subcommand ───────────────────────────
    join_p = subparsers.add_parser(
        "join",
        help="Join OCR results with Google Forms/Sheets",
    )
    join_p.add_argument(
        "--class", dest="class_id", default=None,
        help="Class identifier ({class} pattern in week.yaml)",
    )
    join_p.add_argument(
        "--ocr-results", required=False, default=None,
        help="OCR results YAML file path",
    )
    join_p.add_argument(
        "--output", required=False, default=None,
        help="Output YAML file path",
    )
    join_p.add_argument(
        "--spreadsheet-url", default=None,
        help="Google Sheets URL (preferred source)",
    )
    join_p.add_argument(
        "--forms-csv", default=None,
        help="Google Forms CSV file path (fallback)",
    )
    join_p.add_argument(
        "--credentials", default="credentials.json",
        help="OAuth2 credentials JSON path (default: credentials.json)",
    )
    join_p.add_argument(
        "--manual-mapping", default=None,
        help="Manual mapping YAML file path (for unmatched students)",
    )
    join_p.add_argument(
        "--student-id-column", default="student_id",
        help="Student ID column name (default: student_id)",
    )
    join_p.add_argument(
        "--week-config", default=None, dest="week_config",
        help="week.yaml path (default: auto-discover from current directory)",
    )
    join_p.add_argument(
        "--ocr-review-threshold", type=float, default=None,
        dest="ocr_review_threshold",
        help="OCR confidence review threshold (default: 0.75)",
    )

    # ── compare subcommand ─────────────────────────
    cmp_p = subparsers.add_parser(
        "compare",
        help="Naver OCR vs LLM Vision comparison (research)",
    )
    cmp_source = cmp_p.add_mutually_exclusive_group(required=True)
    cmp_source.add_argument(
        "--image",
        help="Image file path to compare (single image)",
    )
    cmp_source.add_argument(
        "--image-dir", dest="image_dir",
        help="Image directory to compare (batch mode)",
    )
    cmp_p.add_argument(
        "--provider", default="gemini",
        help="LLM provider (gemini or anthropic, default: gemini)",
    )
    cmp_p.add_argument(
        "--model", default=None,
        help="LLM model ID (default: provider default)",
    )
    cmp_p.add_argument(
        "--naver-config", default="", dest="naver_config",
        help="Naver OCR config JSON file path",
    )
    cmp_p.add_argument(
        "--prefix", default="q",
        help="Batch mode: image filename prefix (default: q)",
    )
    cmp_p.add_argument(
        "--subject", default=None,
        help="Subject name (LLM prompt context)",
    )
    cmp_p.add_argument(
        "--question", default=None,
        help="Question text (LLM prompt context)",
    )
    cmp_p.add_argument(
        "--answer-keywords", default=None, dest="answer_keywords",
        help="Key terms (LLM prompt context)",
    )
    cmp_p.add_argument(
        "--output", default=None,
        help="Comparison result YAML output path (required for batch mode)",
    )
    cmp_p.add_argument(
        "--no-resume", action="store_true", default=False, dest="no_resume",
        help="Batch mode: ignore previous results, start from scratch",
    )

    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "scan":
        has_source = (
            getattr(args, "config", None)
            or getattr(args, "class_id", None)
            or getattr(args, "provider", None)
        )
        if not has_source:
            scan_p.error(
                "At least one source must be specified: --config, --class, or --provider"
            )
    if (
        getattr(args, "command", None) == "join"
        and not getattr(args, "class_id", None)
        and getattr(args, "ocr_results", None) is not None
        and getattr(args, "output", None) is None
    ):
        join_p.error("--output is required when --ocr-results is specified without --class")
    return args


def _load_ocr_config(path: str) -> dict:
    """Load and validate an OCR config YAML file.

    Required keys: ``image-dir``, ``naver-ocr-config``, ``output``.
    Optional key: ``num-questions`` (int, default 2).

    Args:
        path: path to the YAML config file.

    Returns:
        Validated config dict.

    Raises:
        ValueError: if required keys are missing.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    required = {"image-dir", "naver-ocr-config", "output"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(
            f"OCR config YAML is missing required keys: {missing}. "
            "Required: image-dir, naver-ocr-config, output. "
            "[_load_ocr_config]"
        )
    return cfg


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for forma-ocr."""
    args = _parse_args(argv)

    # Apply project config (three-layer merge)
    from forma.project_config import apply_project_config
    raw_argv = argv if argv is not None else sys.argv[1:]
    apply_project_config(args, argv=raw_argv)

    # BUG-001 fix: reconstruct explicit_keys to distinguish CLI vs forma.yaml
    _explicit = {
        token.lstrip("-").split("=")[0].replace("-", "_")
        for token in (raw_argv or [])
        if token.startswith("--")
    }

    if args.command == "scan":
        if getattr(args, "config", None):
            # Legacy --config mode (takes precedence)
            cfg = _load_ocr_config(args.config)
            num_questions = (
                args.num_questions if "num_questions" in _explicit
                else int(cfg.get("num-questions", 2))
            )
            crop_coords = None
            raw_coords = cfg.get("crop-coords")
            if raw_coords is not None:
                crop_coords = [tuple(c) for c in raw_coords]

            ocr_review_threshold = args.ocr_review_threshold or 0.75
            run_scan_pipeline(
                image_dir=cfg["image-dir"],
                naver_ocr_config=cfg["naver-ocr-config"],
                output_path=cfg["output"],
                num_questions=num_questions,
                crop_coords=crop_coords,
                ocr_review_threshold=ocr_review_threshold,
            )
        elif getattr(args, "provider", None) and not getattr(args, "class_id", None):
            # LLM Vision mode (--provider without --class)
            # Minimal mode: scan current directory or week.yaml dir
            from forma.week_config import find_week_config, load_week_config
            from pathlib import Path

            image_dir = "."
            output_path = "scan_results.yaml"
            num_questions = args.num_questions or 2
            crop_coords = None
            ocr_review_threshold = args.ocr_review_threshold or 0.75

            # Try week.yaml for paths
            if args.week_config:
                week_yaml_path = Path(args.week_config)
                week_cfg = load_week_config(week_yaml_path)
                # Use defaults from week.yaml if available
                image_dir = str(week_yaml_path.parent)
                output_path = str(week_yaml_path.parent / "scan_results.yaml")

            # Resolve OCR model + API key from config
            ocr_model_from_config = None
            api_key_from_config = None
            if not args.no_config:
                try:
                    from forma.project_config import find_project_config, load_project_config
                    proj_path = find_project_config()
                    if proj_path:
                        proj = load_project_config(proj_path)
                        ocr_model_from_config = proj.get("ocr", {}).get("ocr_model")
                except Exception:
                    pass
                try:
                    from forma.config import get_llm_config, load_config
                    app_config = load_config()
                    llm_cfg = get_llm_config(app_config)
                    api_key_from_config = llm_cfg.get("api_key")
                except Exception:
                    pass
            resolved_model = (
                getattr(args, "model", None) if "model" in _explicit
                else ocr_model_from_config
            )

            # Build LLM context from CLI args
            llm_context = None
            subject = getattr(args, "subject", None)
            question = getattr(args, "question", None)
            answer_keywords = getattr(args, "answer_keywords", None)
            if subject or question or answer_keywords:
                llm_context = {}
                if subject:
                    llm_context["subject"] = subject
                if question:
                    llm_context["question"] = question
                if answer_keywords:
                    llm_context["answer_keywords"] = answer_keywords

            run_scan_pipeline(
                image_dir=image_dir,
                output_path=output_path,
                num_questions=num_questions,
                crop_coords=crop_coords,
                ocr_review_threshold=ocr_review_threshold,
                llm_provider=args.provider,
                llm_model=resolved_model,
                llm_api_key=api_key_from_config,
                llm_context=llm_context,
            )
        elif getattr(args, "class_id", None):
            # --class mode: load week.yaml and resolve patterns
            from forma.week_config import (
                find_week_config,
                load_week_config,
                resolve_class_patterns,
                save_crop_coords,
            )
            from pathlib import Path

            if args.week_config:
                week_yaml_path = Path(args.week_config)
            else:
                week_yaml_path = find_week_config()
            if week_yaml_path is None:
                print("Error: week.yaml not found.")
                sys.exit(1)
            week_cfg = load_week_config(week_yaml_path)
            resolved = resolve_class_patterns(week_cfg, args.class_id)
            base_dir = week_yaml_path.parent

            image_dir = str(base_dir / resolved.ocr_image_dir_pattern)
            output_path = str(base_dir / resolved.ocr_ocr_output_pattern)
            num_questions = (
                args.num_questions if "num_questions" in _explicit
                else resolved.ocr_num_questions if resolved.ocr_num_questions
                else args.num_questions
            )

            # Use saved crop coords unless --recrop
            crop_coords = None
            if not args.recrop and resolved.ocr_crop_coords:
                crop_coords = [tuple(c) for c in resolved.ocr_crop_coords]

            # Load OCR settings from forma.yaml + config.json
            naver_ocr_config = ""
            ocr_model_from_config = None
            api_key_from_config = None
            if not args.no_config:
                try:
                    from forma.project_config import find_project_config, load_project_config
                    proj_path = find_project_config()
                    if proj_path:
                        proj = load_project_config(proj_path)
                        ocr_section = proj.get("ocr", {})
                        naver_ocr_config = ocr_section.get("naver_config", "")
                        ocr_model_from_config = ocr_section.get("ocr_model")
                except Exception as exc:
                    logger.debug("Failed to load project config: %s", exc)
                try:
                    from forma.config import get_llm_config, load_config
                    app_config = load_config()
                    llm_cfg = get_llm_config(app_config)
                    api_key_from_config = llm_cfg.get("api_key")
                except Exception:
                    pass

            # Resolve OCR review threshold: CLI > week.yaml > default
            ocr_review_threshold = (
                args.ocr_review_threshold if "ocr_review_threshold" in _explicit
                else resolved.ocr_review_threshold
            )

            # Build LLM Vision kwargs when --provider is given
            llm_kwargs: dict = {}
            provider = getattr(args, "provider", None)
            if provider:
                llm_kwargs["llm_provider"] = provider
                # CLI --model > forma.yaml ocr.ocr_model > provider default
                llm_kwargs["llm_model"] = (
                    getattr(args, "model", None) if "model" in _explicit
                    else ocr_model_from_config
                )
                if api_key_from_config:
                    llm_kwargs["llm_api_key"] = api_key_from_config
                # Build context from CLI args
                subject = getattr(args, "subject", None)
                question = getattr(args, "question", None)
                answer_keywords = getattr(args, "answer_keywords", None)
                if subject or question or answer_keywords:
                    llm_context: dict = {}
                    if subject:
                        llm_context["subject"] = subject
                    if question:
                        llm_context["question"] = question
                    if answer_keywords:
                        llm_context["answer_keywords"] = answer_keywords
                    llm_kwargs["llm_context"] = llm_context

            results = run_scan_pipeline(
                image_dir=image_dir,
                naver_ocr_config=naver_ocr_config,
                output_path=output_path,
                num_questions=num_questions,
                crop_coords=crop_coords,
                ocr_review_threshold=ocr_review_threshold,
                **llm_kwargs,
            )

            # Auto-save crop coords back to week.yaml if newly selected
            if crop_coords is None and results:
                # Extract crop coords from the pipeline's interactive selection
                # The coords are saved by run_scan_pipeline internally;
                # also persist to week.yaml for reuse across classes
                try:
                    from forma.preprocess_imgs import _last_crop_coords
                    if _last_crop_coords:
                        save_crop_coords(week_yaml_path, _last_crop_coords)
                except (ImportError, AttributeError):
                    pass

    elif args.command == "join":
        if getattr(args, "class_id", None):
            # --class mode: load week.yaml and resolve patterns
            from forma.week_config import (
                find_week_config,
                load_week_config,
                resolve_class_patterns,
            )
            from pathlib import Path

            if args.week_config:
                week_yaml_path = Path(args.week_config)
            else:
                week_yaml_path = find_week_config()
            if week_yaml_path is None:
                print("Error: week.yaml not found.")
                sys.exit(1)
            week_cfg = load_week_config(week_yaml_path)
            resolved = resolve_class_patterns(week_cfg, args.class_id)
            base_dir = week_yaml_path.parent

            ocr_results_path = str(base_dir / resolved.ocr_ocr_output_pattern)
            output_path = str(base_dir / resolved.ocr_join_output_pattern)

            # Load spreadsheet_url and credentials from forma.yaml
            spreadsheet_url = args.spreadsheet_url
            credentials_path = args.credentials
            forms_csv = args.forms_csv or resolved.ocr_join_forms_csv
            if forms_csv and not Path(forms_csv).is_absolute():
                forms_csv = str(base_dir / forms_csv)
            student_id_column = (
                args.student_id_column if "student_id_column" in _explicit
                else resolved.ocr_student_id_column if resolved.ocr_student_id_column
                else args.student_id_column
            )
            if spreadsheet_url is None and not args.no_config:
                try:
                    from forma.project_config import find_project_config, load_project_config
                    proj_path = find_project_config()
                    if proj_path:
                        proj = load_project_config(proj_path)
                        spreadsheet_url = proj.get("ocr", {}).get("spreadsheet_url", "")
                        cred = proj.get("ocr", {}).get("credentials", "")
                        if cred:
                            credentials_path = cred
                except Exception as exc:
                    logger.debug("Failed to load project config: %s", exc)

            if not spreadsheet_url and not forms_csv:
                print(
                    "Error: At least one of spreadsheet_url (forma.yaml) or "
                    "--forms-csv is required."
                )
                sys.exit(1)

            # Resolve OCR review threshold: CLI > week.yaml > default
            join_threshold = (
                args.ocr_review_threshold if "ocr_review_threshold" in _explicit
                else resolved.ocr_review_threshold
            )

            run_join_pipeline(
                ocr_results_path=ocr_results_path,
                output_path=output_path,
                forms_csv_path=forms_csv,
                spreadsheet_url=spreadsheet_url or None,
                credentials_path=credentials_path,
                manual_mapping_path=args.manual_mapping,
                student_id_column=student_id_column,
                ocr_review_threshold=join_threshold,
            )
        else:
            # Legacy mode
            if args.spreadsheet_url is None and args.forms_csv is None:
                print(
                    "Error: At least one of --spreadsheet-url or "
                    "--forms-csv is required."
                )
                sys.exit(1)
            join_threshold = args.ocr_review_threshold or 0.75
            run_join_pipeline(
                ocr_results_path=args.ocr_results,
                output_path=args.output,
                forms_csv_path=args.forms_csv,
                spreadsheet_url=args.spreadsheet_url,
                credentials_path=args.credentials,
                manual_mapping_path=args.manual_mapping,
                student_id_column=args.student_id_column,
                ocr_review_threshold=join_threshold,
            )

    elif args.command == "compare":
        if getattr(args, "image_dir", None):
            # Batch mode
            if not args.output:
                print("Error: --output is required in batch mode.")
                sys.exit(1)
            main_compare_batch(
                image_dir=args.image_dir,
                output=args.output,
                provider=args.provider,
                model=args.model,
                prefix=getattr(args, "prefix", "q"),
                subject=getattr(args, "subject", None),
                question=getattr(args, "question", None),
                answer_keywords=getattr(args, "answer_keywords", None),
                no_resume=getattr(args, "no_resume", False),
                no_config=args.no_config,
            )
        else:
            main_compare(
                image=args.image,
                provider=args.provider,
                model=args.model,
                subject=getattr(args, "subject", None),
                question=getattr(args, "question", None),
                answer_keywords=getattr(args, "answer_keywords", None),
                output=getattr(args, "output", None),
                no_config=args.no_config,
            )


def main_compare(
    *,
    image: str,
    provider: str,
    model: str | None,
    subject: str | None,
    question: str | None,
    answer_keywords: str | None,
    output: str | None,
    no_config: bool,
) -> None:
    """Run OCR vs LLM Vision comparison for a single image."""
    import os

    from forma.llm_provider import create_provider
    from forma.naver_ocr import (
        extract_raw_ocr_data,
        load_naver_ocr_env,
        send_images_receive_ocr,
    )
    from forma.ocr_compare import compare_single_image

    if not os.path.isfile(image):
        print(f"Error: Image file not found: {image}")
        sys.exit(1)

    # Load Naver OCR config
    try:
        secret_key, api_url = load_naver_ocr_env()
    except Exception as exc:
        print(f"Error: Failed to load Naver OCR config: {exc}")
        sys.exit(1)

    # Run Naver OCR
    try:
        ocr_responses = send_images_receive_ocr(api_url, secret_key, [image])
    except Exception as exc:
        print(f"Error: Naver OCR API call failed: {exc}")
        sys.exit(1)

    # Extract raw data
    raw_data = extract_raw_ocr_data(ocr_responses)
    if not raw_data:
        print("Error: No OCR results.")
        sys.exit(1)

    # Get first image's data
    image_key = next(iter(raw_data))
    naver_raw = raw_data[image_key]

    # Create LLM provider
    try:
        llm = create_provider(provider=provider, model=model)
    except Exception as exc:
        print(f"Error: Failed to create LLM provider: {exc}")
        sys.exit(1)

    # Build context
    context: dict[str, str] | None = None
    if subject or question or answer_keywords:
        context = {}
        if subject:
            context["subject"] = subject
        if question:
            context["question"] = question
        if answer_keywords:
            context["answer_keywords"] = answer_keywords

    # Run comparison
    result = compare_single_image(
        image_path=image,
        naver_raw=naver_raw,
        llm_provider=llm,
        context=context,
    )

    # Print comparison table
    _print_comparison_table(result)

    # Save to YAML if requested
    if output:
        result_dict = {
            "image_path": result.image_path,
            "ocr_text": result.ocr_text,
            "llm_text": result.llm_text,
            "summary": result.summary,
            "field_comparisons": [
                {
                    "field_index": fc.field_index,
                    "ocr_text": fc.ocr_text,
                    "llm_text": fc.llm_text,
                    "ocr_confidence": fc.ocr_confidence,
                    "match": fc.match,
                }
                for fc in result.field_comparisons
            ],
        }
        with open(output, "w", encoding="utf-8") as f:
            yaml.dump(result_dict, f, allow_unicode=True, default_flow_style=False)
        print(f"\nResults saved: {output}")


def _print_comparison_table(result: object) -> None:
    """Print a comparison table with box-drawing characters."""
    print(f"\nImage: {result.image_path}")
    print(f"OCR text: {result.ocr_text}")
    print(f"LLM text: {result.llm_text}")
    print()

    idx_w = 4
    ocr_w = 16
    llm_w = 16
    conf_w = 8
    match_w = 6

    print(f"┌{'─' * idx_w}┬{'─' * ocr_w}┬{'─' * llm_w}┬{'─' * conf_w}┬{'─' * match_w}┐")
    print(f"│{'#':^{idx_w}}│{'OCR':^{ocr_w}}│{'LLM':^{llm_w}}│{'Conf':^{conf_w}}│{'Match':^{match_w}}│")
    print(f"├{'─' * idx_w}┼{'─' * ocr_w}┼{'─' * llm_w}┼{'─' * conf_w}┼{'─' * match_w}┤")

    for fc in result.field_comparisons:
        conf_str = f"{fc.ocr_confidence:.2f}" if fc.ocr_confidence is not None else "N/A"
        match_str = "O" if fc.match else "X"
        ocr_text = fc.ocr_text[:ocr_w - 2]
        llm_text = fc.llm_text[:llm_w - 2]
        print(
            f"│{fc.field_index:>{idx_w}}│"
            f"{ocr_text:<{ocr_w}}│"
            f"{llm_text:<{llm_w}}│"
            f"{conf_str:^{conf_w}}│"
            f"{match_str:^{match_w}}│"
        )

    print(f"└{'─' * idx_w}┴{'─' * ocr_w}┴{'─' * llm_w}┴{'─' * conf_w}┴{'─' * match_w}┘")

    s = result.summary
    print(f"\nTotal: {s['total']} fields, matched {s['match_count']}, mismatched {s['mismatch_count']}")


def main_compare_batch(
    *,
    image_dir: str,
    output: str,
    provider: str,
    model: str | None,
    prefix: str,
    subject: str | None,
    question: str | None,
    answer_keywords: str | None,
    no_resume: bool,
    no_config: bool,
) -> None:
    """Run batch OCR vs LLM Vision comparison for a directory."""
    import os

    from forma.ocr_compare import run_batch_comparison

    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        sys.exit(1)

    # Load naver_config from forma.yaml
    naver_config = ""
    if not no_config:
        try:
            from forma.project_config import (
                find_project_config,
                load_project_config,
            )
            proj_path = find_project_config()
            if proj_path:
                proj = load_project_config(proj_path)
                naver_config = proj.get("ocr", {}).get("naver_config", "")
        except Exception:
            pass

    # Build context
    context: dict[str, str] | None = None
    if subject or question or answer_keywords:
        context = {}
        if subject:
            context["subject"] = subject
        if question:
            context["question"] = question
        if answer_keywords:
            context["answer_keywords"] = answer_keywords

    run_batch_comparison(
        image_dir=image_dir,
        output_path=output,
        naver_config=naver_config,
        llm_provider_name=provider,
        llm_model=model,
        context=context,
        prefix=prefix,
        resume=not no_resume,
    )


if __name__ == "__main__":
    main()
