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
        description="Report email delivery automation",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        default=False,
        dest="no_config",
        help="Skip forma.yaml config file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    # --- prepare subcommand ---
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Collect student reports and create zip archives",
    )
    prepare_parser.add_argument(
        "--manifest",
        required=True,
        help="Delivery manifest YAML file path",
    )
    prepare_parser.add_argument(
        "--roster",
        required=True,
        help="Student roster YAML file path",
    )
    prepare_parser.add_argument(
        "--output-dir",
        required=True,
        dest="output_dir",
        help="Staging folder output path",
    )
    prepare_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing staging folder",
    )
    prepare_parser.add_argument(
        "--no-config",
        action="store_true",
        default=argparse.SUPPRESS,
        dest="no_config",
        help="Skip forma.yaml config file",
    )
    prepare_parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable verbose logging",
    )

    # --- send subcommand ---
    send_parser = subparsers.add_parser(
        "send",
        help="Send emails",
    )
    send_parser.add_argument(
        "--staged",
        required=True,
        help="Staging folder path from prepare step",
    )
    send_parser.add_argument(
        "--template",
        required=True,
        help="Email template YAML file path",
    )
    send_parser.add_argument(
        "--smtp-config",
        required=False,
        default=None,
        dest="smtp_config",
        help="SMTP config YAML file path (uses config.json smtp section if not specified)",
    )
    send_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Preview only (no actual delivery)",
    )
    send_parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=False,
        dest="retry_failed",
        help="Resend previously failed items only",
    )
    send_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Ignore previously sent records and resend all",
    )
    send_parser.add_argument(
        "--notify-sender",
        action="store_true",
        default=False,
        dest="notify_sender",
        help="Send result summary email to instructor",
    )
    send_parser.add_argument(
        "--password-from-stdin",
        action="store_true",
        default=False,
        dest="password_from_stdin",
        help="Read SMTP password from stdin",
    )
    send_parser.add_argument(
        "--no-config",
        action="store_true",
        default=argparse.SUPPRESS,
        dest="no_config",
        help="Skip forma.yaml config file",
    )
    send_parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable verbose logging",
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
            f"Error: Manifest file not found: {args.manifest}",
            file=sys.stderr,
        )
        sys.exit(2)

    if not os.path.exists(args.roster):
        print(
            f"Error: Roster file not found: {args.roster}",
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
        f"Preparation complete: {summary.total_students} students total "
        f"(ready={summary.ready}, warning={summary.warnings}, "
        f"error={summary.errors})"
    )

    verbose = getattr(args, "verbose", False)
    if summary.errors > 0:
        error_students = [d for d in summary.details if d.status == "error"]
        for d in error_students:
            label = f"{d.student_id} ({d.name})" if verbose else d.student_id
            print(f"  [ERROR] {label}: {d.message}")

    if summary.warnings > 0:
        warning_students = [d for d in summary.details if d.status == "warning"]
        for d in warning_students:
            label = f"{d.student_id} ({d.name})" if verbose else d.student_id
            print(f"  [WARNING] {label}: {d.message}")


def _cmd_send(args: argparse.Namespace) -> None:
    """Handle the ``send`` subcommand.

    Args:
        args: Parsed arguments from argparse.
    """
    from forma.delivery_send import send_emails

    # Validate file existence (exit 2 for missing paths)
    if not os.path.exists(args.staged):
        print(f"Error: Staging folder not found: {args.staged}", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.template):
        print(f"Error: Template file not found: {args.template}", file=sys.stderr)
        sys.exit(2)

    # Resolve SMTP configuration: --smtp-config path or config.json fallback
    smtp_config_obj = None
    smtp_config_path = getattr(args, "smtp_config", None) or ""
    _config_password: str | None = None

    if args.smtp_config:
        # Explicit --smtp-config path (deprecated)
        import warnings

        warnings.warn(
            "--smtp-config is deprecated and will be removed in a future version. "
            "Migrate to the smtp section in config.json.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not os.path.exists(args.smtp_config):
            print(
                f"Error: SMTP config file not found: {args.smtp_config}",
                file=sys.stderr,
            )
            sys.exit(2)
    else:
        # Fallback to config.json smtp section
        try:
            from forma.config import get_smtp_config, get_smtp_password, load_config

            config = load_config()
            smtp_config_obj = get_smtp_config(config)
            _config_password: str | None = get_smtp_password(config)
        except (FileNotFoundError, KeyError, ValueError):
            print(
                "Error: SMTP config not found. Configure --smtp-config or config.json smtp section.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Flag interaction: --retry-failed + --force is invalid
    if getattr(args, "retry_failed", False) and getattr(args, "force", False):
        print("Error: --retry-failed and --force cannot be used together.", file=sys.stderr)
        sys.exit(1)

    # Resolve password: stdin > config.json smtp.password > env var (in send_emails)
    password: str | None = None
    if getattr(args, "password_from_stdin", False):
        password = sys.stdin.readline().rstrip("\n")
    elif _config_password is not None:
        password = _config_password

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
    print(f"{prefix}Total: {log.success}/{log.total} succeeded, {log.failed} failed")

    # Notify sender summary email (FR-019)
    if getattr(args, "notify_sender", False) and not log.dry_run:
        from forma.delivery_send import load_smtp_config, send_summary_email

        notify_password = password
        if notify_password is None:
            notify_password = _config_password or os.environ.get("FORMA_SMTP_PASSWORD", "")
        if notify_password:
            try:
                if smtp_config_obj is not None:
                    smtp_cfg = smtp_config_obj
                else:
                    smtp_cfg = load_smtp_config(args.smtp_config)
                send_summary_email(log, smtp_cfg, password=notify_password)
            except Exception as e:
                print(f"Warning: Summary email delivery failed: {e}", file=sys.stderr)
        else:
            print("Warning: --notify-sender requires a password.", file=sys.stderr)

    # Exit code 3 for partial failure
    if log.failed > 0 and not log.dry_run:
        sys.exit(3)


def _normalize_argv(argv: list[str]) -> list[str]:
    """Move --no-config and --verbose before subcommand to after it.

    Argparse subcommands don't inherit parent flags reliably, so we
    normalize the argv so these flags always appear after the subcommand.

    Args:
        argv: Raw argument list.

    Returns:
        Normalized argument list.
    """
    hoisted = []
    remaining = []
    subcommand_seen = False
    for arg in argv:
        if arg in ("prepare", "send"):
            subcommand_seen = True
            remaining.append(arg)
            # Insert hoisted flags right after the subcommand
            remaining.extend(hoisted)
            hoisted.clear()
        elif not subcommand_seen and arg in ("--no-config", "--verbose"):
            hoisted.append(arg)
        else:
            remaining.append(arg)
    # If no subcommand was found, put hoisted flags back
    if hoisted:
        remaining.extend(hoisted)
    return remaining


def main(argv=None) -> None:
    """Entry point for forma-deliver CLI.

    Args:
        argv: Command-line arguments (for testing). None uses sys.argv.
    """
    parser = _build_parser()

    raw = argv if argv is not None else sys.argv[1:]
    normalized = _normalize_argv(list(raw))

    try:
        args = parser.parse_args(normalized)
    except SystemExit as e:
        # Re-raise argparse exits (missing required args, etc.)
        # Map argparse's default exit code 2 to our exit code 1 for no subcommand
        if e.code == 2 and argv is not None and not any(s in argv for s in ("prepare", "send")):
            sys.exit(1)
        raise

    # Apply project config (three-layer merge)
    if not getattr(args, "no_config", False):
        from forma.project_config import apply_project_config

        raw_argv = argv if argv is not None else sys.argv[1:]
        apply_project_config(args, argv=raw_argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.subcommand == "prepare":
        _cmd_prepare(args)
    elif args.subcommand == "send":
        _cmd_send(args)


if __name__ == "__main__":
    main()
