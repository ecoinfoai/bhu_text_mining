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

from forma.naver_ocr import (
    extract_text,
    load_naver_ocr_env,
    prepare_image_files_list,
    send_images_receive_ocr,
)
from forma.preprocess_imgs import crop_and_save_images, show_image
from forma.qr_decode import decode_qr_from_image, parse_qr_content


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
        coords_list = []
        for q_idx in range(1, num_questions + 1):
            print(
                f"\nQ{q_idx} 답안 영역을 선택하세요 "
                f"(좌상단 → 우하단 클릭):"
            )
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
        for img_path in sorted(
            prepare_image_files_list(image_dir, prefix + "_")
        ):
            all_cropped.append((q_idx, img_path))

    total_files = len(all_cropped)
    results: list[dict[str, Any]] = []
    qr_decoded = 0
    qr_failed = 0
    file_counter: dict[int, int] = {}

    print(f"\nQR 디코딩 + OCR 처리 중 ({total_files}개 이미지)...")
    for processed, (q_idx, img_path) in enumerate(all_cropped, 1):
        file_counter[q_idx] = file_counter.get(q_idx, 0) + 1
        file_idx = file_counter[q_idx]
        source_file = os.path.basename(img_path)
        pct = processed * 100 // total_files
        print(
            f"\r  [{pct:3d}%] {processed}/{total_files}  {source_file}",
            end="", flush=True,
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
    output_path: str,
    forms_csv_path: str | None = None,
    spreadsheet_url: str | None = None,
    credentials_path: str = "credentials.json",
    manual_mapping_path: str | None = None,
    student_id_column: str = "student_id",
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

    Returns:
        List of joined result dicts.  Each entry includes all
        OCR result fields plus a ``"forms_data"`` sub-dict
        when a matching row is found.

    Raises:
        ValueError: if neither *spreadsheet_url* nor
            *forms_csv_path* is provided.
    """
    if spreadsheet_url is None and forms_csv_path is None:
        raise ValueError(
            "At least one data source required: "
            "--spreadsheet-url or --forms-csv"
        )

    with open(ocr_results_path, encoding="utf-8") as f:
        ocr_results: list[dict[str, Any]] = yaml.safe_load(f)

    # ── load forms data ──────────────────────────
    forms_data: dict[str, dict[str, str]] = {}
    source_label = ""

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
            source_label = "Google Sheets"
        except Exception as exc:
            if forms_csv_path is not None:
                print(
                    f"⚠ Google Sheets 접근 실패 ({exc}), "
                    f"CSV 폴백: {forms_csv_path}"
                )
            else:
                raise

    if not forms_data and forms_csv_path is not None:
        forms_data = _load_forms_csv(forms_csv_path, student_id_column)
        source_label = "CSV"

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
            extra = {
                k: v
                for k, v in forms_row.items()
                if k != student_id_column
            }
            record["forms_data"] = extra
            matched_sids.add(sid)
        joined.append(record)

    # ── match report ─────────────────────────────
    all_sids = {
        e.get("student_id", "")
        for e in ocr_results
        if not e.get("student_id", "").startswith("UNKNOWN_")
    }
    unmatched = sorted(all_sids - matched_sids)
    total = len(all_sids)
    matched = len(matched_sids)
    print(f"✓ {matched}/{total} 학생 매칭 완료")
    if unmatched:
        print(f"⚠ {len(unmatched)}명 미매칭: {', '.join(unmatched)}")

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
            "tag:yaml.org,2002:str", data, style='"',
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

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f,
            Dumper=_QuotedDumper,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    print(f"✓ 결과 저장: {output_path}")
