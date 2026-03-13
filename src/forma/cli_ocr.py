# src/cli_ocr.py
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
        description="스캔된 답안지 OCR 파이프라인",
    )
    parser.add_argument(
        "--no-config", action="store_true", default=False, dest="no_config",
        help="forma.yaml 설정 파일 무시",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── scan subcommand ───────────────────────────
    scan_p = subparsers.add_parser(
        "scan",
        help="이미지 스캔 → QR 디코딩 → OCR → YAML",
    )
    scan_source = scan_p.add_mutually_exclusive_group(required=True)
    scan_source.add_argument(
        "--config",
        help="OCR 설정 YAML 파일 경로 (레거시)",
    )
    scan_source.add_argument(
        "--class", dest="class_id",
        help="분반 식별자 (week.yaml의 {class} 패턴 치환)",
    )
    scan_p.add_argument(
        "--num-questions", type=int, default=None,
        help="문제 수 (config YAML의 num-questions 값 사용 가능)",
    )
    scan_p.add_argument(
        "--recrop", action="store_true", default=False,
        help="저장된 crop 좌표 무시, 재선택",
    )
    scan_p.add_argument(
        "--week-config", default=None, dest="week_config",
        help="week.yaml 경로 (기본: 현재 디렉토리에서 자동 탐색)",
    )

    # ── join subcommand ───────────────────────────
    join_p = subparsers.add_parser(
        "join",
        help="OCR 결과 + Google Forms/Sheets 조인",
    )
    join_p.add_argument(
        "--class", dest="class_id", default=None,
        help="분반 식별자 (week.yaml의 {class} 패턴 치환)",
    )
    join_p.add_argument(
        "--ocr-results", required=False, default=None,
        help="OCR 결과 YAML 파일 경로",
    )
    join_p.add_argument(
        "--output", required=False, default=None,
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
    join_p.add_argument(
        "--week-config", default=None, dest="week_config",
        help="week.yaml 경로 (기본: 현재 디렉토리에서 자동 탐색)",
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
    """CLI entrypoint for forma-ocr."""
    args = _parse_args(argv)

    # Apply project config (three-layer merge)
    from forma.project_config import apply_project_config
    raw_argv = argv if argv is not None else sys.argv[1:]
    apply_project_config(args, argv=raw_argv)

    if args.command == "scan":
        if getattr(args, "class_id", None):
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
                print("오류: week.yaml을 찾을 수 없습니다.")
                sys.exit(1)
            week_cfg = load_week_config(week_yaml_path)
            resolved = resolve_class_patterns(week_cfg, args.class_id)
            base_dir = week_yaml_path.parent

            image_dir = str(base_dir / resolved.ocr_image_dir_pattern)
            output_path = str(base_dir / resolved.ocr_ocr_output_pattern)
            num_questions = (
                args.num_questions
                if args.num_questions is not None
                else resolved.ocr_num_questions
            )

            # Use saved crop coords unless --recrop
            crop_coords = None
            if not args.recrop and resolved.ocr_crop_coords:
                crop_coords = [tuple(c) for c in resolved.ocr_crop_coords]

            # Load naver_ocr_config from forma.yaml
            naver_ocr_config = ""
            if not args.no_config:
                try:
                    from forma.project_config import find_project_config, load_project_config
                    proj_path = find_project_config()
                    if proj_path:
                        proj = load_project_config(proj_path)
                        naver_ocr_config = proj.get("ocr", {}).get("naver_config", "")
                except Exception as exc:
                    logger.debug("프로젝트 설정 로드 실패: %s", exc)

            results = run_scan_pipeline(
                image_dir=image_dir,
                naver_ocr_config=naver_ocr_config,
                output_path=output_path,
                num_questions=num_questions,
                crop_coords=crop_coords,
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
        else:
            # Legacy --config mode
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
                print("오류: week.yaml을 찾을 수 없습니다.")
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
                args.student_id_column
                if args.student_id_column != "student_id"
                else resolved.ocr_student_id_column or args.student_id_column
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
                    logger.debug("프로젝트 설정 로드 실패: %s", exc)

            if not spreadsheet_url and not forms_csv:
                print(
                    "오류: spreadsheet_url(forma.yaml) 또는 "
                    "--forms-csv 중 하나 이상 필요합니다."
                )
                sys.exit(1)

            run_join_pipeline(
                ocr_results_path=ocr_results_path,
                output_path=output_path,
                forms_csv_path=forms_csv,
                spreadsheet_url=spreadsheet_url or None,
                credentials_path=credentials_path,
                manual_mapping_path=args.manual_mapping,
                student_id_column=student_id_column,
            )
        else:
            # Legacy mode
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
