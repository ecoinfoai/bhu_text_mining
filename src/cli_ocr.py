# src/cli_ocr.py
"""bhu-ocr CLI — OCR pipeline for scanned exam answer sheets.

Usage:
    bhu-ocr scan --config ocr_config.yaml
    bhu-ocr scan --config ocr_config.yaml --num-questions 3

    bhu-ocr join --ocr-results results.yaml \\
                 --output final.yaml \\
                 --spreadsheet-url "https://docs.google.com/spreadsheets/d/XXX" \\
                 [--forms-csv fallback.csv] \\
                 [--credentials credentials.json] \\
                 [--manual-mapping mapping.yaml] \\
                 [--student-id-column "sid"]
"""
from __future__ import annotations

import argparse
import sys

import yaml

from src.ocr_pipeline import run_join_pipeline, run_scan_pipeline


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse bhu-ocr CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="bhu-ocr",
        description="스캔된 답안지 OCR 파이프라인",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── scan subcommand ───────────────────────────
    scan_p = subparsers.add_parser(
        "scan",
        help="이미지 스캔 → QR 디코딩 → OCR → YAML",
    )
    scan_p.add_argument(
        "--config", required=True,
        help="OCR 설정 YAML 파일 경로",
    )
    scan_p.add_argument(
        "--num-questions", type=int, default=None,
        help="문제 수 (config YAML의 num-questions 값 사용 가능)",
    )

    # ── join subcommand ───────────────────────────
    join_p = subparsers.add_parser(
        "join",
        help="OCR 결과 + Google Forms/Sheets 조인",
    )
    join_p.add_argument(
        "--ocr-results", required=True,
        help="OCR 결과 YAML 파일 경로",
    )
    join_p.add_argument(
        "--output", required=True,
        help="출력 YAML 파일 경로",
    )
    join_p.add_argument(
        "--spreadsheet-url", default=None,
        help="Google Sheets URL (우선 사용)",
    )
    join_p.add_argument(
        "--forms-csv", default=None,
        help="Google Forms CSV 파일 경로 (폴백)",
    )
    join_p.add_argument(
        "--credentials", default="credentials.json",
        help="OAuth2 자격증명 JSON 경로 (기본값: credentials.json)",
    )
    join_p.add_argument(
        "--manual-mapping", default=None,
        help="수동 매핑 YAML 파일 경로 (미매칭 학생 보완)",
    )
    join_p.add_argument(
        "--student-id-column", default="student_id",
        help="학생 ID 컬럼명 (기본값: student_id)",
    )

    return parser.parse_args(argv)


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
        cfg = yaml.safe_load(f)
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
    """CLI entrypoint for bhu-ocr."""
    args = _parse_args(argv)

    if args.command == "scan":
        cfg = _load_ocr_config(args.config)
        num_questions = (
            args.num_questions
            if args.num_questions is not None
            else int(cfg.get("num-questions", 2))
        )
        crop_coords = None
        raw_coords = cfg.get("crop-coords")
        if raw_coords is not None:
            crop_coords = [tuple(c) for c in raw_coords]

        run_scan_pipeline(
            image_dir=cfg["image-dir"],
            naver_ocr_config=cfg["naver-ocr-config"],
            output_path=cfg["output"],
            num_questions=num_questions,
            crop_coords=crop_coords,
        )

    elif args.command == "join":
        if args.spreadsheet_url is None and args.forms_csv is None:
            print(
                "오류: --spreadsheet-url 또는 --forms-csv 중 "
                "하나 이상 필요합니다."
            )
            sys.exit(1)
        run_join_pipeline(
            ocr_results_path=args.ocr_results,
            output_path=args.output,
            forms_csv_path=args.forms_csv,
            spreadsheet_url=args.spreadsheet_url,
            credentials_path=args.credentials,
            manual_mapping_path=args.manual_mapping,
            student_id_column=args.student_id_column,
        )


if __name__ == "__main__":
    main()
