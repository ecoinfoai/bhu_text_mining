"""CLI entry point for forma-deliver -- report email delivery automation.

Usage::

    forma-deliver prepare --manifest MANIFEST --roster ROSTER --output-dir DIR
        [--force] [--no-config] [--verbose]

    forma-deliver send --staged DIR --template TEMPLATE --smtp-config CONFIG
        [--dry-run] [--retry-failed] [--force] [--notify-sender]
        [--password-from-stdin] [--no-config] [--verbose]

Subcommands:
    prepare     Collect student report files and create zip archives.
    send        Send emails with zip attachments via SMTP.

Exit codes:
    0 -- success
    1 -- input/data error
    2 -- file path not found
    3 -- partial failure (some emails failed)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with prepare/send subcommands.

    Returns:
        Configured ``ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        prog="forma-deliver",
        description="보고서 이메일 발송 자동화",
    )
    parser.add_argument(
        "--no-config", action="store_true", default=False, dest="no_config",
        help="forma.yaml 설정 파일 무시",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="상세 로그 출력",
    )

    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    # --- prepare subcommand ---
    prepare_parser = subparsers.add_parser(
        "prepare", help="학생별 보고서 수집 및 zip 생성",
    )
    prepare_parser.add_argument(
        "--manifest", required=True,
        help="발송 매니페스트 YAML 파일 경로",
    )
    prepare_parser.add_argument(
        "--roster", required=True,
        help="학생 명부 YAML 파일 경로",
    )
    prepare_parser.add_argument(
        "--output-dir", required=True, dest="output_dir",
        help="staging 폴더 출력 경로",
    )
    prepare_parser.add_argument(
        "--force", action="store_true", default=False,
        help="기존 staging 폴더 덮어쓰기",
    )

    # --- send subcommand ---
    send_parser = subparsers.add_parser(
        "send", help="이메일 발송",
    )
    send_parser.add_argument(
        "--staged", required=True,
        help="prepare에서 생성한 staging 폴더 경로",
    )
    send_parser.add_argument(
        "--template", required=True,
        help="이메일 템플릿 YAML 파일 경로",
    )
    send_parser.add_argument(
        "--smtp-config", required=False, default=None, dest="smtp_config",
        help="SMTP 설정 YAML 파일 경로 (미지정 시 forma.json smtp 섹션 사용)",
    )
    send_parser.add_argument(
        "--dry-run", action="store_true", default=False, dest="dry_run",
        help="미리보기만 (실제 발송 없음)",
    )
    send_parser.add_argument(
        "--retry-failed", action="store_true", default=False, dest="retry_failed",
        help="이전 실패 건만 재발송",
    )
    send_parser.add_argument(
        "--force", action="store_true", default=False,
        help="이미 발송 완료된 기록 무시하고 전체 재발송",
    )
    send_parser.add_argument(
        "--notify-sender", action="store_true", default=False, dest="notify_sender",
        help="교수자에게 결과 요약 이메일 발송",
    )
    send_parser.add_argument(
        "--password-from-stdin", action="store_true", default=False,
        dest="password_from_stdin",
        help="SMTP 비밀번호를 stdin에서 읽기",
    )

    return parser


def _cmd_prepare(args: argparse.Namespace) -> None:
    """Handle the ``prepare`` subcommand.

    Args:
        args: Parsed arguments from argparse.
    """
    from forma.delivery_prepare import prepare_delivery

    # Validate file existence (exit 2 for missing paths)
    if not os.path.exists(args.manifest):
        print(
            f"Error: 매니페스트 파일을 찾을 수 없습니다: {args.manifest}",
            file=sys.stderr,
        )
        sys.exit(2)

    if not os.path.exists(args.roster):
        print(
            f"Error: 명부 파일을 찾을 수 없습니다: {args.roster}",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        summary = prepare_delivery(
            manifest_path=args.manifest,
            roster_path=args.roster,
            output_dir=args.output_dir,
            force=args.force,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except (ValueError, FileExistsError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Console summary output
    print(
        f"준비 완료: 전체 {summary.total_students}명 "
        f"(ready={summary.ready}, warning={summary.warnings}, "
        f"error={summary.errors})"
    )

    if summary.errors > 0:
        error_students = [d for d in summary.details if d.status == "error"]
        for d in error_students:
            print(f"  [ERROR] {d.student_id} ({d.name}): {d.message}")

    if summary.warnings > 0:
        warning_students = [d for d in summary.details if d.status == "warning"]
        for d in warning_students:
            print(f"  [WARNING] {d.student_id} ({d.name}): {d.message}")


def _cmd_send(args: argparse.Namespace) -> None:
    """Handle the ``send`` subcommand.

    Args:
        args: Parsed arguments from argparse.
    """
    from forma.delivery_send import send_emails

    # Validate file existence (exit 2 for missing paths)
    if not os.path.exists(args.staged):
        print(f"Error: staging 폴더를 찾을 수 없습니다: {args.staged}", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.template):
        print(f"Error: 템플릿 파일을 찾을 수 없습니다: {args.template}", file=sys.stderr)
        sys.exit(2)

    # Resolve SMTP configuration: --smtp-config path or forma.json fallback
    smtp_config_obj = None
    smtp_config_path = getattr(args, "smtp_config", None) or ""

    if args.smtp_config:
        # Explicit --smtp-config path (deprecated)
        import warnings

        warnings.warn(
            "--smtp-config는 향후 버전에서 제거됩니다. "
            "forma.json의 smtp 섹션으로 마이그레이션하세요.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not os.path.exists(args.smtp_config):
            print(
                f"Error: SMTP 설정 파일을 찾을 수 없습니다: {args.smtp_config}",
                file=sys.stderr,
            )
            sys.exit(2)
    else:
        # Fallback to forma.json smtp section
        try:
            from forma.config import get_smtp_config, load_config

            config = load_config()
            smtp_config_obj = get_smtp_config(config)
        except (FileNotFoundError, KeyError, ValueError):
            print(
                "Error: SMTP 설정을 찾을 수 없습니다. "
                "--smtp-config 또는 forma.json smtp 섹션을 설정하세요.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Flag interaction: --retry-failed + --force is invalid
    if getattr(args, "retry_failed", False) and getattr(args, "force", False):
        print("Error: --retry-failed와 --force는 함께 사용할 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    # Read password from stdin if requested
    password = None
    if getattr(args, "password_from_stdin", False):
        password = sys.stdin.readline().rstrip("\n")

    try:
        log = send_emails(
            staging_dir=args.staged,
            template_path=args.template,
            smtp_config_path=smtp_config_path,
            force=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
            retry_failed=getattr(args, "retry_failed", False),
            password=password,
            smtp_config=smtp_config_obj,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except (ValueError, FileExistsError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Console summary output (FR-018)
    prefix = "[DRY-RUN] " if log.dry_run else ""
    print(
        f"{prefix}전체 {log.total}건 중 {log.success}건 성공, "
        f"{log.failed}건 실패"
    )

    # Notify sender summary email (FR-019)
    if getattr(args, "notify_sender", False) and not log.dry_run:
        from forma.delivery_send import load_smtp_config, send_summary_email

        notify_password = password
        if notify_password is None:
            notify_password = os.environ.get("FORMA_SMTP_PASSWORD", "")
        if notify_password:
            try:
                if smtp_config_obj is not None:
                    smtp_cfg = smtp_config_obj
                else:
                    smtp_cfg = load_smtp_config(args.smtp_config)
                send_summary_email(log, smtp_cfg, password=notify_password)
            except Exception as e:
                print(f"Warning: 요약 이메일 발송 실패: {e}", file=sys.stderr)
        else:
            print("Warning: --notify-sender 사용 시 비밀번호가 필요합니다.", file=sys.stderr)

    # Exit code 3 for partial failure
    if log.failed > 0 and not log.dry_run:
        sys.exit(3)


def main(argv=None) -> None:
    """Entry point for forma-deliver CLI.

    Args:
        argv: Command-line arguments (for testing). None uses sys.argv.
    """
    parser = _build_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # Re-raise argparse exits (missing required args, etc.)
        # Map argparse's default exit code 2 to our exit code 1 for no subcommand
        if e.code == 2 and argv is not None and not any(
            s in argv for s in ("prepare", "send")
        ):
            sys.exit(1)
        raise

    # Apply project config (three-layer merge)
    if not getattr(args, "no_config", False):
        from forma.project_config import apply_project_config
        raw_argv = argv if argv is not None else sys.argv[1:]
        apply_project_config(args, argv=raw_argv)

    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.subcommand == "prepare":
        _cmd_prepare(args)
    elif args.subcommand == "send":
        _cmd_send(args)


if __name__ == "__main__":
    main()
