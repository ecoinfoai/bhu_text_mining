"""OCR vs LLM Vision comparison module.

Compares Naver CLOVA OCR results with LLM Vision recognition on the same
image to identify potential OCR errors, especially in low-confidence fields.

Usage (programmatic):
    result = run_comparison("image.jpg", naver_config="config.json")
    print(format_comparison_report(result))

Usage (CLI):
    forma ocr compare --image image.jpg --provider gemini
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FieldComparison:
    """Comparison result for a single OCR field."""

    field_index: int
    ocr_text: str
    llm_text: str
    ocr_confidence: float | None
    match: bool


@dataclass
class ComparisonResult:
    """Full comparison result for one image."""

    image_path: str
    ocr_text: str
    llm_text: str
    field_comparisons: list[FieldComparison]
    summary: dict[str, Any]


def build_comparison_prompt(
    ocr_fields: list[dict],
    context: dict[str, str] | None = None,
) -> str:
    """Build a structured prompt for LLM Vision to read text from an image.

    Args:
        ocr_fields: OCR field data (used for reference in prompt).
        context: Optional exam context (subject, question, answer_keywords).

    Returns:
        Prompt string to send with the image.
    """
    lines = [
        "이 이미지에서 손글씨로 작성된 모든 텍스트를 정확하게 읽어주세요.",
        "각 단어를 공백으로 구분하여 한 줄로 출력하세요.",
        "추가 설명 없이 읽은 텍스트만 출력하세요.",
    ]

    if context:
        lines.append("")
        lines.append("참고 정보:")
        if "subject" in context:
            lines.append(f"- 과목: {context['subject']}")
        if "question" in context:
            lines.append(f"- 문제: {context['question']}")
        if "answer_keywords" in context:
            lines.append(f"- 핵심 키워드: {context['answer_keywords']}")

    return "\n".join(lines)


def align_ocr_llm_tokens(
    ocr_tokens: list[str],
    llm_text: str,
) -> list[tuple[str, str, bool]]:
    """Align OCR tokens with LLM output tokens using simple word-level matching.

    Args:
        ocr_tokens: List of OCR infer_text values.
        llm_text: Full text from LLM Vision.

    Returns:
        List of (ocr_token, llm_token, match) tuples. Length == len(ocr_tokens).
    """
    if not ocr_tokens:
        return []

    llm_tokens = llm_text.split() if llm_text.strip() else []
    result: list[tuple[str, str, bool]] = []

    for i, ocr_tok in enumerate(ocr_tokens):
        if i < len(llm_tokens):
            llm_tok = llm_tokens[i]
            match = ocr_tok.strip() == llm_tok.strip()
            result.append((ocr_tok, llm_tok, match))
        else:
            result.append((ocr_tok, "", False))

    return result


def compare_single_image(
    image_path: str,
    naver_raw: dict,
    llm_provider: Any,
    context: dict[str, str] | None = None,
) -> ComparisonResult:
    """Compare Naver OCR and LLM Vision results for a single image.

    Args:
        image_path: Path to the image file.
        naver_raw: Raw OCR data from ``extract_raw_ocr_data()`` for this image.
        llm_provider: LLMProvider instance with ``generate_with_image()`` method.
        context: Optional exam context for prompt building.

    Returns:
        ComparisonResult with field-level comparisons and summary stats.
    """
    ocr_fields = naver_raw.get("fields", [])
    ocr_tokens = [f["infer_text"] for f in ocr_fields]
    ocr_text = " ".join(ocr_tokens)

    prompt = build_comparison_prompt(ocr_fields, context=context)
    llm_text = llm_provider.generate_with_image(
        prompt=prompt,
        image_path=image_path,
        max_tokens=4096,
    )
    llm_text = llm_text.strip()

    aligned = align_ocr_llm_tokens(ocr_tokens, llm_text)

    field_comparisons: list[FieldComparison] = []
    match_count = 0
    mismatch_count = 0

    for i, (ocr_tok, llm_tok, match) in enumerate(aligned):
        conf = ocr_fields[i].get("infer_confidence") if i < len(ocr_fields) else None
        field_comparisons.append(FieldComparison(
            field_index=i,
            ocr_text=ocr_tok,
            llm_text=llm_tok,
            ocr_confidence=conf,
            match=match,
        ))
        if match:
            match_count += 1
        else:
            mismatch_count += 1

    summary = {
        "total": len(field_comparisons),
        "match_count": match_count,
        "mismatch_count": mismatch_count,
    }

    return ComparisonResult(
        image_path=image_path,
        ocr_text=ocr_text,
        llm_text=llm_text,
        field_comparisons=field_comparisons,
        summary=summary,
    )


def run_comparison(
    image_path: str,
    naver_config: str = "",
    llm_provider_name: str = "gemini",
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    context: dict[str, str] | None = None,
) -> ComparisonResult:
    """End-to-end comparison: call both Naver OCR and Vision LLM.

    Args:
        image_path: Path to the image file.
        naver_config: Naver OCR config path (JSON).
        llm_provider_name: "gemini" or "anthropic".
        llm_model: LLM model override.
        llm_api_key: LLM API key (falls back to env var).
        context: Optional exam context for prompt building.

    Returns:
        ComparisonResult with field-level comparisons.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # --- Naver OCR ---
    from forma.naver_ocr import (
        extract_raw_ocr_data,
        load_naver_ocr_env,
        send_images_receive_ocr,
    )

    secret_key, api_url = load_naver_ocr_env(naver_config)
    logger.info("Naver OCR 호출: %s", image_path)
    ocr_responses = send_images_receive_ocr(api_url, secret_key, [image_path])
    raw_data = extract_raw_ocr_data(ocr_responses)

    image_name = os.path.basename(image_path)
    naver_raw = raw_data.get(image_name, {"fields": []})

    # --- Vision LLM ---
    from forma.llm_provider import create_provider

    logger.info("LLM Vision 호출: %s (%s)", llm_provider_name, llm_model or "default")
    provider = create_provider(
        provider=llm_provider_name, api_key=llm_api_key, model=llm_model,
    )

    return compare_single_image(
        image_path=image_path,
        naver_raw=naver_raw,
        llm_provider=provider,
        context=context,
    )


def format_comparison_report(result: ComparisonResult) -> str:
    """Format a ComparisonResult as a human-readable text report."""
    lines = [
        f"=== OCR 비교 결과: {os.path.basename(result.image_path)} ===",
        "",
        f"Naver OCR: {result.ocr_text}",
        f"LLM 인식:  {result.llm_text}",
        "",
        f"필드 수: {result.summary['total']}",
        f"일치:    {result.summary['match_count']}",
        f"불일치:  {result.summary['mismatch_count']}",
        "",
        "--- 필드별 비교 ---",
    ]

    for fc in result.field_comparisons:
        mark = "O" if fc.match else "X"
        conf_str = f"{fc.ocr_confidence:.2f}" if fc.ocr_confidence is not None else "N/A"
        lines.append(
            f"  [{mark}] {fc.field_index:3d}: "
            f"OCR={fc.ocr_text!r:20s}  LLM={fc.llm_text!r:20s}  "
            f"confidence={conf_str}"
        )

    return "\n".join(lines)


def run_batch_comparison(
    image_dir: str,
    output_path: str,
    naver_config: str = "",
    llm_provider_name: str = "gemini",
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    context: dict[str, str] | None = None,
    prefix: str = "q",
    resume: bool = True,
) -> list[dict]:
    """Batch-compare Naver OCR vs LLM Vision for all images in a directory.

    Processes images one at a time (rate-limit safe).  Saves results
    incrementally to ``output_path`` after each image, so interrupted
    runs can be resumed.

    Args:
        image_dir: Directory containing cropped answer images.
        output_path: YAML file path for comparison results.
        naver_config: Naver OCR config path.
        llm_provider_name: "gemini" or "anthropic".
        llm_model: LLM model override.
        llm_api_key: LLM API key.
        context: Optional exam context for prompt building.
        prefix: Image filename prefix to filter (default "q").
        resume: If True and output_path exists, skip already-processed images.

    Returns:
        List of comparison result dicts.
    """
    import yaml

    from forma.llm_provider import create_provider
    from forma.naver_ocr import (
        extract_raw_ocr_data,
        load_naver_ocr_env,
        send_images_receive_ocr,
    )

    # Collect target images
    image_files = sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.startswith(prefix) and f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not image_files:
        raise FileNotFoundError(
            f"No images with prefix {prefix!r} in {image_dir}"
        )

    # Load existing results for resume
    done: dict[str, dict] = {}
    if resume and os.path.isfile(output_path):
        with open(output_path, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or []
        for entry in existing:
            done[entry["image_name"]] = entry
        logger.info("기존 결과 %d건 로드 (재개 모드)", len(done))

    # Initialize providers
    secret_key, api_url = load_naver_ocr_env(naver_config)

    # Load LLM API key from config.json if not provided
    resolved_llm_key = llm_api_key
    if resolved_llm_key is None:
        try:
            from forma.config import load_config
            cfg = load_config()
            llm_cfg = cfg.get("llm", {})
            resolved_llm_key = llm_cfg.get("api_key")
            if not llm_provider_name or llm_provider_name == "gemini":
                llm_provider_name = llm_cfg.get("provider", llm_provider_name)
        except Exception:
            pass

    provider = create_provider(
        provider=llm_provider_name, api_key=resolved_llm_key, model=llm_model,
    )

    total = len(image_files)
    results = list(done.values())

    for idx, img_path in enumerate(image_files, 1):
        image_name = os.path.basename(img_path)

        if image_name in done:
            print(f"  [{idx:3d}/{total}] {image_name} — 건너뜀 (이미 처리)")
            continue

        print(f"  [{idx:3d}/{total}] {image_name} — 처리 중...", end="", flush=True)

        # 1) Naver OCR
        try:
            ocr_responses = send_images_receive_ocr(
                api_url, secret_key, [img_path],
            )
            raw_map = extract_raw_ocr_data(ocr_responses)
            naver_raw = raw_map.get(image_name, {"fields": []})
        except Exception as exc:
            logger.warning("Naver OCR 실패 %s: %s", image_name, exc)
            naver_raw = {"fields": []}

        # 2) LLM Vision
        try:
            llm_result = compare_single_image(
                image_path=img_path,
                naver_raw=naver_raw,
                llm_provider=provider,
                context=context,
            )
        except Exception as exc:
            logger.warning("LLM 비교 실패 %s: %s", image_name, exc)
            print(f" 실패: {exc}")
            continue

        # 3) Build result entry
        entry = {
            "image_name": image_name,
            "image_path": img_path,
            "naver_text": llm_result.ocr_text,
            "llm_text": llm_result.llm_text,
            "summary": llm_result.summary,
            "field_comparisons": [
                {
                    "field_index": fc.field_index,
                    "ocr_text": fc.ocr_text,
                    "llm_text": fc.llm_text,
                    "ocr_confidence": fc.ocr_confidence,
                    "match": fc.match,
                }
                for fc in llm_result.field_comparisons
            ],
            "naver_raw_fields": naver_raw.get("fields", []),
        }
        results.append(entry)
        done[image_name] = entry

        # 4) Incremental save
        _save_batch_results(results, output_path)

        match_rate = (
            llm_result.summary["match_count"] / llm_result.summary["total"]
            if llm_result.summary["total"] > 0
            else 0
        )
        print(
            f" 완료 (일치 {llm_result.summary['match_count']}"
            f"/{llm_result.summary['total']}"
            f" = {match_rate:.0%})"
        )

    # Final summary
    _print_batch_summary(results)
    return results


def _save_batch_results(results: list[dict], output_path: str) -> None:
    """Save batch results to YAML (atomic overwrite)."""
    import yaml

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            results, f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def _print_batch_summary(results: list[dict]) -> None:
    """Print aggregate statistics for batch comparison."""
    if not results:
        print("\n결과 없음")
        return

    total_fields = 0
    total_match = 0
    total_mismatch = 0

    for r in results:
        s = r.get("summary", {})
        total_fields += s.get("total", 0)
        total_match += s.get("match_count", 0)
        total_mismatch += s.get("mismatch_count", 0)

    overall_rate = total_match / total_fields if total_fields > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"배치 비교 완료: {len(results)}개 이미지")
    print(f"전체 필드: {total_fields}개")
    print(f"일치: {total_match}개 ({overall_rate:.1%})")
    print(f"불일치: {total_mismatch}개")
    print(f"{'=' * 50}")
