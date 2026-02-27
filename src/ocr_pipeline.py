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
import os
from typing import Any, Optional

import yaml

from src.naver_ocr import (
    extract_text,
    load_naver_ocr_env,
    prepare_image_files_list,
    send_images_receive_ocr,
)
from src.qr_decode import decode_qr_from_image, parse_qr_content

# preprocess_imgs is imported lazily inside functions that need it
# because it calls matplotlib.use("Qt5Agg") at module level, which
# requires a display.  Lazy import keeps this module importable in
# headless / test environments.


def run_scan_pipeline(
    image_dir: str,
    naver_ocr_config: str,
    output_path: str,
    num_questions: int = 2,
    crop_coords: Optional[list[tuple[int, int, int, int]]] = None,
) -> list[dict[str, Any]]:
    """Run the full scan pipeline on a directory of scanned sheets.

    Args:
        image_dir: directory containing raw scanned images.
        naver_ocr_config: path to Naver OCR config JSON
            (``{"secret_key": ..., "api_url": ...}``).
        output_path: YAML file path for results.
        num_questions: number of answer areas per sheet.
        crop_coords: pre-defined crop coordinates list
            (one tuple per question).  Pass ``None`` for
            interactive click-to-crop mode.

    Returns:
        List of result dicts, each with keys:
        ``student_id``, ``q_num``, ``text``, ``source_file``.

    Raises:
        FileNotFoundError: if no images are found in *image_dir*.
    """
    secret_key, api_url = load_naver_ocr_env(naver_ocr_config)

    raw_images = _list_raw_images(image_dir)
    if not raw_images:
        raise FileNotFoundError(
            f"No images found in {image_dir!r}. "
            "[run_scan_pipeline]"
        )
    sample_image = raw_images[0]

    # ── step 1: collect crop coordinates ─────────
    coords_list: list[tuple[int, int, int, int]]
    if crop_coords is not None:
        coords_list = list(crop_coords)
    else:
        from src.preprocess_imgs import show_image  # lazy: needs display
        coords_list = []
        for q_idx in range(1, num_questions + 1):
            print(
                f"\nQ{q_idx} 답안 영역을 선택하세요 "
                f"(좌상단 → 우하단 클릭):"
            )
            coords_list.append(show_image(sample_image))

    # ── step 2: batch crop per question ──────────
    from src.preprocess_imgs import crop_and_save_images  # lazy
    prefixes: list[str] = []
    for q_idx, coords in enumerate(coords_list, 1):
        prefix = f"q{q_idx}"
        crop_and_save_images(image_dir, coords, prefix)
        prefixes.append(prefix)

    # ── steps 3 & 4: QR decode + OCR ─────────────
    results: list[dict[str, Any]] = []
    qr_decoded = 0
    qr_failed = 0

    for q_idx, prefix in enumerate(prefixes, 1):
        cropped_files = sorted(
            prepare_image_files_list(image_dir, prefix + "_")
        )

        for file_idx, img_path in enumerate(cropped_files):
            source_file = os.path.basename(img_path)

            # QR decode
            raw_qr = decode_qr_from_image(img_path)
            if raw_qr is not None:
                try:
                    parsed = parse_qr_content(raw_qr)
                    student_id: str = parsed["student_id"]
                    q_num: int = parsed.get("q_num") or q_idx
                    qr_decoded += 1
                except ValueError:
                    student_id = f"UNKNOWN_{file_idx + 1:04d}"
                    q_num = q_idx
                    qr_failed += 1
            else:
                student_id = f"UNKNOWN_{file_idx + 1:04d}"
                q_num = q_idx
                qr_failed += 1

            # Naver OCR
            text = ""
            try:
                ocr_responses = send_images_receive_ocr(
                    api_url, secret_key, [img_path],
                )
                text_map = extract_text(ocr_responses)
                text = next(iter(text_map.values()), "")
            except Exception:
                pass

            results.append({
                "student_id": student_id,
                "q_num": q_num,
                "text": text,
                "source_file": source_file,
            })

    total = qr_decoded + qr_failed
    print(
        f"\n✓ {qr_decoded}/{total} QR decoded, "
        f"⚠ {qr_failed} failures"
    )

    _save_yaml(results, output_path)
    return results


def run_join_pipeline(
    ocr_results_path: str,
    forms_csv_path: str,
    output_path: str,
    student_id_column: str = "student_id",
) -> list[dict[str, Any]]:
    """Join OCR results YAML with Google Forms CSV data.

    Args:
        ocr_results_path: path to OCR results YAML file.
        forms_csv_path: path to Google Forms CSV download.
        output_path: YAML file path for joined results.
        student_id_column: column name for the student ID
            in the CSV file (default: ``"student_id"``).

    Returns:
        List of joined result dicts.  Each entry includes all
        OCR result fields plus a ``"forms_data"`` sub-dict
        when a matching CSV row is found.
    """
    with open(ocr_results_path, encoding="utf-8") as f:
        ocr_results: list[dict[str, Any]] = yaml.safe_load(f)

    # Load CSV keyed by student_id
    forms_data: dict[str, dict[str, str]] = {}
    with open(
        forms_csv_path, encoding="utf-8-sig", newline="",
    ) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get(student_id_column, "").strip()
            if sid:
                forms_data[sid] = dict(row)

    # Join
    joined: list[dict[str, Any]] = []
    for entry in ocr_results:
        sid = entry.get("student_id", "")
        forms_row = forms_data.get(sid, {})
        record: dict[str, Any] = dict(entry)
        if forms_row:
            extra = {
                k: v
                for k, v in forms_row.items()
                if k != student_id_column
            }
            record["forms_data"] = extra
        joined.append(record)

    _save_yaml(joined, output_path)
    return joined


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
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    print(f"✓ 결과 저장: {output_path}")
