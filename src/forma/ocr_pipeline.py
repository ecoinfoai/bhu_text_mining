# src/ocr_pipeline.py
"""OCR pipeline orchestration for scanned exam answer sheets.

Workflow:
    1. Interactive crop selection per question area (or pre-supplied coords)
    2. Batch crop all images per question
    3. QR decode each cropped image → student_id + q_num
    4. Naver OCR each cropped image → text
    5. Join results → YAML output
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Optional

import yaml

from forma.io_utils import _atomic_write
from forma.naver_ocr import (
    extract_raw_ocr_data,
    extract_text_with_confidence,
    load_naver_ocr_env,
    prepare_image_files_list,
    send_images_receive_ocr,
)
from forma.preprocess_imgs import crop_and_save_images, show_image
from forma.qr_decode import decode_qr_from_image, parse_qr_content

logger = logging.getLogger(__name__)


def run_scan_pipeline(
    image_dir: str,
    naver_ocr_config: str = "",
    output_path: str = "",
    num_questions: int = 2,
    crop_coords: Optional[list[tuple[int, int, int, int]]] = None,
    ocr_review_threshold: float = 0.75,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_context: Optional[dict[str, str]] = None,
    llm_rate_limit_delay: float = 4.0,
) -> list[dict[str, Any]]:
    """Run the full scan pipeline on a directory of scanned sheets.

    Supports two recognition modes:
    - **LLM Vision mode** (``llm_provider`` set): uses LLM API for recognition
    - **Naver OCR mode** (legacy): uses Naver CLOVA OCR API

    Args:
        image_dir: directory containing raw scanned images.
        naver_ocr_config: path to Naver OCR config JSON (legacy mode).
        output_path: YAML file path for results.
        num_questions: number of answer areas per sheet.
        crop_coords: pre-defined crop coordinates list.
        ocr_review_threshold: confidence threshold for review (default 0.75).
        llm_provider: LLM provider name (``"gemini"`` or ``"anthropic"``).
            If set, uses LLM Vision instead of Naver OCR.
        llm_model: LLM model ID override.
        llm_api_key: LLM API key (falls back to env var).
        llm_context: Exam context for LLM prompt enrichment.
        llm_rate_limit_delay: Seconds between LLM API calls.

    Returns:
        List of result dicts.

    Raises:
        FileNotFoundError: if no images are found in *image_dir*.
    """
    use_llm = llm_provider is not None

    secret_key, api_url = ("", "") if use_llm else load_naver_ocr_env(naver_ocr_config)

    raw_images = _list_raw_images(image_dir)
    if not raw_images:
        raise FileNotFoundError(f"No images found in {image_dir!r}. [run_scan_pipeline]")
    sample_image = raw_images[0]

    # ── step 1: collect crop coordinates ─────────
    coords_list: list[tuple[int, int, int, int]]
    if crop_coords is not None:
        coords_list = list(crop_coords)
    else:
        coords_list = []
        for q_idx in range(1, num_questions + 1):
            print(f"\nSelect answer area for Q{q_idx} (click top-left then bottom-right):")
            coords_list.append(show_image(sample_image))

    # ── step 2: batch crop per question ──────────
    prefixes: list[str] = []
    for q_idx, coords in enumerate(coords_list, 1):
        prefix = f"q{q_idx}"
        crop_and_save_images(image_dir, coords, prefix)
        prefixes.append(prefix)

    # ── steps 3 & 4: QR decode + OCR ─────────────
    # Collect all cropped files upfront for progress tracking
    all_cropped: list[tuple[int, str]] = []
    for q_idx, prefix in enumerate(prefixes, 1):
        for img_path in sorted(prepare_image_files_list(image_dir, prefix + "_")):
            all_cropped.append((q_idx, img_path))

    total_files = len(all_cropped)

    # ── LLM batch recognition (before per-image loop) ──
    llm_results: dict[str, Any] = {}
    llm_model_name: str = ""
    if use_llm and total_files > 0:
        from forma.llm_ocr import extract_text_via_llm

        image_paths_only = [p for _, p in all_cropped]
        llm_results = extract_text_via_llm(
            image_paths=image_paths_only,
            provider=llm_provider,  # type: ignore[arg-type]
            model=llm_model,
            api_key=llm_api_key,
            context=llm_context,
            rate_limit_delay=llm_rate_limit_delay,
            review_threshold=ocr_review_threshold,
        )
        # Resolve model name for output
        from forma.llm_provider import create_provider as _create_prov

        try:
            _tmp = _create_prov(provider=llm_provider, api_key=llm_api_key, model=llm_model)  # type: ignore[arg-type]
            llm_model_name = _tmp.model_name
        except Exception:
            llm_model_name = llm_model or llm_provider or ""

    results: list[dict[str, Any]] = []
    qr_decoded = 0
    qr_failed = 0
    file_counter: dict[int, int] = {}

    print(f"\nQR decoding + OCR processing ({total_files} images)...")
    for processed, (q_idx, img_path) in enumerate(all_cropped, 1):
        file_counter[q_idx] = file_counter.get(q_idx, 0) + 1
        file_idx = file_counter[q_idx]
        source_file = os.path.basename(img_path)
        pct = processed * 100 // total_files
        print(
            f"\r  [{pct:3d}%] {processed}/{total_files}  {source_file}",
            end="",
            flush=True,
        )

        # QR decode
        raw_qr = decode_qr_from_image(img_path)
        if raw_qr is not None:
            try:
                parsed = parse_qr_content(raw_qr)
                student_id: str = parsed["student_id"]
                q_num: int = parsed.get("q_num") or q_idx
                qr_decoded += 1
            except ValueError:
                student_id = f"UNKNOWN_{file_idx:04d}"
                q_num = q_idx
                qr_failed += 1
        else:
            student_id = f"UNKNOWN_{file_idx:04d}"
            q_num = q_idx
            qr_failed += 1

        # Text recognition (LLM or Naver OCR)
        text = ""
        confidence_mean = None
        confidence_min = None
        field_count = None
        raw_fields: list[dict] | None = None
        llm_extra: dict[str, Any] = {}

        if use_llm:
            # LLM Vision mode — look up pre-computed result
            llm_resp = llm_results.get(img_path)
            if llm_resp is not None:
                text = llm_resp.text
                confidence_mean = llm_resp.confidence_mean
                confidence_min = llm_resp.confidence_min
                if llm_resp.word_confidences:
                    field_count = len(llm_resp.word_confidences)
                llm_extra["recognition_engine"] = "llm"
                llm_extra["recognition_model"] = llm_model_name
                if llm_resp.word_confidences is not None:
                    llm_extra["llm_word_confidences"] = [
                        {"word": wc.word, "confidence": wc.confidence, "token_count": wc.token_count}
                        for wc in llm_resp.word_confidences
                    ]
                if llm_resp.usage is not None:
                    llm_extra["llm_usage"] = {
                        "input_tokens": llm_resp.usage.input_tokens,
                        "output_tokens": llm_resp.usage.output_tokens,
                    }
                llm_extra["llm_finish_reason"] = llm_resp.finish_reason
        else:
            # Naver OCR mode (legacy)
            try:
                ocr_responses = send_images_receive_ocr(
                    api_url,
                    secret_key,
                    [img_path],
                )
                conf_map = extract_text_with_confidence(ocr_responses)
                if conf_map:
                    entry = next(iter(conf_map.values()))
                    text = entry["text"]
                    confidence_mean = entry["confidence_mean"]
                    confidence_min = entry["confidence_min"]
                    field_count = entry["field_count"]

                raw_map = extract_raw_ocr_data(ocr_responses)
                if raw_map:
                    raw_entry = next(iter(raw_map.values()))
                    raw_fields = raw_entry.get("fields")
            except Exception as exc:
                logger.warning("OCR failed for %s: %s", img_path, exc)

        result_entry: dict[str, Any] = {
            "student_id": student_id,
            "q_num": q_num,
            "text": text,
            "source_file": source_file,
            "ocr_confidence_mean": confidence_mean,
            "ocr_confidence_min": confidence_min,
            "ocr_field_count": field_count,
        }
        if raw_fields is not None:
            result_entry["ocr_raw_fields"] = raw_fields
        result_entry.update(llm_extra)
        results.append(result_entry)

    total = qr_decoded + qr_failed
    print(f"\n✓ {qr_decoded}/{total} QR decoded, ⚠ {qr_failed} failures")

    # Low-confidence summary
    low_conf = [
        r
        for r in results
        if r.get("ocr_confidence_mean") is not None and r["ocr_confidence_mean"] < ocr_review_threshold
    ]
    if low_conf:
        print(
            f"WARNING: {len(low_conf)} low-confidence entries detected "
            f"(confidence < {ocr_review_threshold}). "
            f"Run `forma ocr join` for details."
        )

    # Generate review_needed.yaml for LLM mode
    if use_llm and low_conf:
        review_entries = []
        for r in low_conf:
            reason_parts = []
            if r.get("ocr_confidence_mean") is not None:
                reason_parts.append(f"confidence {r['ocr_confidence_mean']:.2f} < {ocr_review_threshold}")
            review_entries.append(
                {
                    "image_name": r.get("source_file", ""),
                    "student_id": r.get("student_id", ""),
                    "q_num": r.get("q_num", 0),
                    "text": r.get("text", ""),
                    "ocr_confidence_mean": r.get("ocr_confidence_mean"),
                    "reason": "; ".join(reason_parts) if reason_parts else "low_confidence",
                }
            )
        review_path = os.path.join(
            os.path.dirname(os.path.abspath(output_path)),
            "review_needed.yaml",
        )
        _save_yaml(review_entries, review_path)

    _save_yaml(results, output_path)
    return results


def run_join_pipeline(
    ocr_results_path: str,
    output_path: str,
    forms_csv_path: str | None = None,
    spreadsheet_url: str | None = None,
    credentials_path: str = "credentials.json",
    manual_mapping_path: str | None = None,
    student_id_column: str = "student_id",
    ocr_review_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    """Join OCR results YAML with Google Forms data.

    Data source priority:
        1. *spreadsheet_url* → Google Sheets fetch
        2. On failure, *forms_csv_path* → CSV fallback
        3. Neither provided → ``ValueError``

    Args:
        ocr_results_path: path to OCR results YAML file.
        output_path: YAML file path for joined results.
        forms_csv_path: path to Google Forms CSV download (optional).
        spreadsheet_url: Google Sheets URL (optional).
        credentials_path: path to OAuth2 credentials JSON
            (used with *spreadsheet_url*).
        manual_mapping_path: YAML file mapping student IDs to
            form fields for students missing from Forms/CSV.
        student_id_column: column name for the student ID
            in the CSV / sheet (default: ``"student_id"``).
        ocr_review_threshold: confidence threshold for the
            low-confidence review table (default 0.75).

    Returns:
        List of joined result dicts.  Each entry includes all
        OCR result fields plus a ``"forms_data"`` sub-dict
        when a matching row is found.

    Raises:
        ValueError: if neither *spreadsheet_url* nor
            *forms_csv_path* is provided.
    """
    if spreadsheet_url is None and forms_csv_path is None:
        raise ValueError("At least one data source required: --spreadsheet-url or --forms-csv")

    with open(ocr_results_path, encoding="utf-8") as f:
        ocr_results: list[dict[str, Any]] = yaml.safe_load(f)

    # ── load forms data ──────────────────────────
    forms_data: dict[str, dict[str, str]] = {}

    if spreadsheet_url is not None:
        try:
            from forma.google_sheets import fetch_sheet_as_records

            records = fetch_sheet_as_records(
                spreadsheet_url,
                credentials_path=credentials_path,
            )
            for row in records:
                sid = str(row.get(student_id_column, "")).strip()
                if sid:
                    forms_data[sid] = {str(k): str(v) for k, v in row.items()}
        except Exception as exc:
            if forms_csv_path is not None:
                print(f"WARNING: Google Sheets access failed ({exc}), falling back to CSV: {forms_csv_path}")
            else:
                raise

    if not forms_data and forms_csv_path is not None:
        forms_data = _load_forms_csv(forms_csv_path, student_id_column)
        _source_label = "CSV"

    # ── manual mapping supplement ────────────────
    if manual_mapping_path is not None:
        with open(manual_mapping_path, encoding="utf-8") as f:
            manual: dict[str, dict[str, str]] = yaml.safe_load(f) or {}
        for sid, fields in manual.items():
            if sid not in forms_data:
                forms_data[sid] = fields

    # ── join ─────────────────────────────────────
    joined: list[dict[str, Any]] = []
    matched_sids: set[str] = set()
    for entry in ocr_results:
        sid = entry.get("student_id", "")
        forms_row = forms_data.get(sid, {})
        record: dict[str, Any] = dict(entry)
        if forms_row:
            extra = {k: v for k, v in forms_row.items() if k != student_id_column}
            record["forms_data"] = extra
            matched_sids.add(sid)
        joined.append(record)

    # ── match report ─────────────────────────────
    all_sids = {e.get("student_id", "") for e in ocr_results if not e.get("student_id", "").startswith("UNKNOWN_")}
    unmatched = sorted(all_sids - matched_sids)
    total = len(all_sids)
    matched = len(matched_sids)
    print(f"{matched}/{total} students matched")
    if unmatched:
        print(f"WARNING: {len(unmatched)} unmatched: {', '.join(unmatched)}")

    # OCR confidence review table
    _print_confidence_review_table(joined, threshold=ocr_review_threshold)

    _save_yaml(joined, output_path)
    return joined


def _load_forms_csv(
    csv_path: str,
    student_id_column: str,
) -> dict[str, dict[str, str]]:
    """Load a Google Forms CSV keyed by student ID."""
    forms_data: dict[str, dict[str, str]] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get(student_id_column, "").strip()
            if sid:
                forms_data[sid] = dict(row)
    return forms_data


def _print_confidence_review_table(
    joined: list[dict[str, Any]],
    threshold: float = 0.75,
) -> None:
    """Print a review table for low-confidence OCR entries.

    Args:
        joined: List of joined result dicts (with optional confidence fields).
        threshold: Confidence threshold below which entries are flagged.
    """
    # Check if any entry has confidence data
    has_confidence = any(entry.get("ocr_confidence_mean") is not None for entry in joined)
    if not has_confidence:
        return

    # Filter entries with confidence below threshold
    total_with_confidence = sum(1 for e in joined if e.get("ocr_confidence_mean") is not None)
    low_entries = [
        e for e in joined if e.get("ocr_confidence_mean") is not None and e["ocr_confidence_mean"] < threshold
    ]

    if not low_entries:
        print(f"OK: No OCR entries require review (all confidence >= {threshold})")
        return

    # Sort by confidence_mean ascending (INV-J01)
    low_entries.sort(key=lambda e: e["ocr_confidence_mean"])

    # Build table rows
    rows: list[tuple[str, str, str, str, str, str]] = []
    for e in low_entries:
        sid = e.get("student_id", "")
        forms = e.get("forms_data", {})
        name = forms.get("이름", "-")
        q_num = str(e.get("q_num", ""))
        conf_mean = f"{e['ocr_confidence_mean']:.2f}"
        conf_min = f"{e['ocr_confidence_min']:.2f}" if e.get("ocr_confidence_min") is not None else "-"
        text = e.get("text", "")
        preview = text[:30] + "..." if len(text) > 30 else text
        rows.append((sid, name, q_num, conf_mean, conf_min, preview))

    # Print table
    headers = ("Student ID", "Name", "Q#", "Conf", "Min Block", "Recognized text (first 30 chars)")
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def _row_str(vals: tuple) -> str:
        cells = " │ ".join(v.ljust(w) for v, w in zip(vals, col_widths))
        return f"│ {cells} │"

    sep = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
    bot = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"

    title = f"  OCR entries requiring review (confidence < {threshold})"
    title_row = f"│{title.ljust(sum(col_widths) + 3 * (len(col_widths) - 1) + 2)}│"
    title_top = "┌" + "─" * (sum(col_widths) + 3 * (len(col_widths) - 1) + 2) + "┐"

    print(title_top)
    print(title_row)
    print(sep)
    print(_row_str(headers))
    print(sep)
    for r in rows:
        print(_row_str(r))
    print(bot)

    pct = len(low_entries) * 100 / total_with_confidence if total_with_confidence else 0
    print(f"  {len(low_entries)} / {total_with_confidence} total ({pct:.1f}%)")


# ── internal helpers ─────────────────────────────


def _list_raw_images(image_dir: str) -> list[str]:
    """Return sorted list of non-cropped images in *image_dir*."""
    supported = (".jpg", ".jpeg", ".png")
    result = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(supported):
            continue
        # Skip already-cropped files (q1_*, q2_*, ...)
        if any(fname.startswith(f"q{i}_") for i in range(1, 30)):
            continue
        result.append(os.path.join(image_dir, fname))
    return result


def _save_yaml(
    data: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Save *data* as YAML to *output_path*, creating dirs as needed."""
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    # Dumper that quotes string values but not keys
    class _QuotedDumper(yaml.SafeDumper):
        pass

    def _quoted_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
        return dumper.represent_scalar(
            "tag:yaml.org,2002:str",
            data,
            style='"',
        )

    def _mapping(dumper: yaml.SafeDumper, data: dict) -> yaml.MappingNode:
        pairs = []
        for key, value in data.items():
            key_node = dumper.represent_str(key)  # plain key
            val_node = dumper.represent_data(value)
            pairs.append((key_node, val_node))
        return yaml.MappingNode("tag:yaml.org,2002:map", pairs)

    _QuotedDumper.add_representer(str, _quoted_str)
    _QuotedDumper.add_representer(dict, _mapping)

    def _write(f):
        yaml.dump(
            data,
            f,
            Dumper=_QuotedDumper,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    _atomic_write(_write, output_path)
    print(f"Results saved: {output_path}")
