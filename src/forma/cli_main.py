"""Unified ``forma`` CLI entry point.

Consolidates 14 separate ``forma-*`` commands into a single ``forma``
command with nested subparsers.

Usage::

    forma report student --help
    forma train risk --help
    forma --version
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

logger = logging.getLogger(__name__)

_VERSION = "0.12.0"

# ---------------------------------------------------------------------------
# Error messages (FR-048)
# ---------------------------------------------------------------------------

_ERROR_MESSAGES: dict[str, str] = {
    "file_not_found": "File not found: {path}",
    "yaml_parse_error": "YAML parse error: {path}",
    "unknown_command": "Unknown command '{cmd}'. Run 'forma --help' to see available commands.",
    "smtp_auth_failure": "SMTP authentication failed: check username or password.",
}


def _korean_error(msg_type: str, **kwargs: str) -> str:
    """Return a formatted error message for the given type.

    Args:
        msg_type: Key in ``_ERROR_MESSAGES``.
        **kwargs: Format parameters for the message template.

    Returns:
        Formatted error message string.
    """
    template = _ERROR_MESSAGES.get(msg_type, msg_type)
    return template.format(**kwargs)


# ---------------------------------------------------------------------------
# Progress logging (FR-049)
# ---------------------------------------------------------------------------


def log_progress(current: int, total: int, task_name: str) -> None:
    """Emit a progress log line in ``[N/total]`` format.

    Args:
        current: Current item number.
        total: Total item count.
        task_name: Description of the task being performed.
    """
    logger.info("[%d/%d] %s", current, total, task_name)


# ---------------------------------------------------------------------------
# Legacy deprecation wrapper factory (FR-019)
# ---------------------------------------------------------------------------


def _make_legacy_wrapper(legacy_name: str, new_name: str, target_main: object) -> object:
    """Create a wrapper that emits DeprecationWarning then calls *target_main*.

    Args:
        legacy_name: The old command name (e.g. ``'forma-report'``).
        new_name: The new command name (e.g. ``'forma report student'``).
        target_main: The original ``main()`` function to delegate to.

    Returns:
        A wrapper function suitable for use as a console_scripts entry point.
    """

    def _wrapper() -> None:
        warnings.warn(
            f"'{legacy_name}' is deprecated and will be removed in a future version. Use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        target_main()

    return _wrapper


# ---------------------------------------------------------------------------
# Command dispatch table
# ---------------------------------------------------------------------------

# Maps (group, subcommand) -> (module_path, function_name)
# None subcommand means the group IS the command (e.g., "init")
_COMMANDS: dict[tuple[str, str | None], tuple[str, str]] = {
    ("exam", None): ("forma.cli", "main"),
    ("ocr", None): ("forma.cli_ocr", "main"),
    ("eval", None): ("forma.pipeline_evaluation", "main"),
    ("eval", "batch"): ("forma.pipeline_batch_evaluation", "main"),
    ("report", "student"): ("forma.cli_report", "main"),
    ("report", "professor"): ("forma.cli_report_professor", "main"),
    ("report", "longitudinal"): ("forma.cli_report_longitudinal", "main"),
    ("report", "warning"): ("forma.cli_report_warning", "main"),
    ("report", "batch"): ("forma.cli_report_batch", "main"),
    ("train", "risk"): ("forma.cli_train", "main"),
    ("train", "grade"): ("forma.cli_train_grade", "main"),
    ("intervention", None): ("forma.cli_intervention", "main"),
    ("deliver", None): ("forma.cli_deliver", "main"),
    ("init", None): ("forma.cli_init", "main"),
    ("select", None): ("forma.cli_select", "main"),
    ("lecture", "analyze"): (
        "forma.cli_lecture", "main_analyze",
    ),
    ("lecture", "compare"): (
        "forma.cli_lecture", "main_compare",
    ),
    ("lecture", "class-compare"): (
        "forma.cli_lecture", "main_class_compare",
    ),
    ("backfill", "longitudinal"): (
        "forma.cli_backfill_longitudinal", "main",
    ),
    ("domain", "extract"): (
        "forma.cli_domain", "extract_main",
    ),
    ("domain", "coverage"): (
        "forma.cli_domain", "coverage_main",
    ),
    ("domain", "report"): (
        "forma.cli_domain", "report_main",
    ),
}

# Groups that have nested subcommands
_NESTED_GROUPS = {"report", "train", "eval", "lecture", "backfill", "domain"}


def _import_delegate(module_path: str, func_name: str) -> object:
    """Lazily import and return the delegate function."""
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


class _FormaParser(argparse.ArgumentParser):
    """Custom parser that emits a user-friendly error for unknown subcommands."""

    def error(self, message: str) -> None:  # noqa: D102
        if "invalid choice" in message:
            # Extract the bad command name from argparse error message
            # Format: "argument command: invalid choice: 'foo' (...)"
            import re

            m = re.search(r"invalid choice: '([^']*)'", message)
            cmd = m.group(1) if m else "?"
            print(
                _korean_error("unknown_command", cmd=cmd),
                file=sys.stderr,
            )
            raise SystemExit(2)
        super().error(message)


def _build_parser() -> _FormaParser:
    """Build the top-level argument parser with nested subparsers."""
    parser = _FormaParser(
        prog="forma",
        description="Formative Assessment Analysis Tool (형성평가 분석 도구)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"forma {_VERSION}",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose output"
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        default=False,
        help="Skip forma.yaml loading",
    )
    parser.add_argument("--font-path", default=None, help="Korean font file path")
    parser.add_argument("--dpi", type=int, default=150, help="Chart resolution DPI")

    subparsers = parser.add_subparsers(dest="command", title="commands")

    # --- Simple commands (no nested subcommands) ---
    subparsers.add_parser("exam", help="Generate exam papers")
    subparsers.add_parser("ocr", help="OCR scan processing")
    subparsers.add_parser("intervention", help="Manage intervention activities")
    subparsers.add_parser("deliver", help="Email report delivery")
    subparsers.add_parser("init", help="Initialize project configuration")
    subparsers.add_parser("select", help="Select student answers")

    # --- eval (optional 'batch' subcommand — parsed manually to avoid
    #     argparse consuming --class values as subcommand names) ---
    subparsers.add_parser("eval", help="Run evaluation pipeline")

    # --- report (nested subcommands) ---
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_sub = report_parser.add_subparsers(dest="report_sub")
    report_sub.add_parser("student", help="Individual student report")
    report_sub.add_parser("professor", help="Professor class summary report")
    report_sub.add_parser("longitudinal", help="Longitudinal analysis report")
    report_sub.add_parser("warning", help="Early warning report")
    report_sub.add_parser("batch", help="Batch report generation")

    # --- train (nested subcommands) ---
    train_parser = subparsers.add_parser("train", help="Train models")
    train_sub = train_parser.add_subparsers(dest="train_sub")
    train_sub.add_parser("risk", help="Train drop risk prediction model")
    train_sub.add_parser("grade", help="Train grade prediction model")

    # --- lecture (nested subcommands) ---
    lecture_parser = subparsers.add_parser(
        "lecture", help="Lecture transcript analysis",
    )
    lecture_sub = lecture_parser.add_subparsers(
        dest="lecture_sub",
    )
    lecture_sub.add_parser("analyze", help="Analyze single transcript")
    lecture_sub.add_parser(
        "compare", help="Compare sections for same session",
    )
    lecture_sub.add_parser(
        "class-compare", help="Compare sections across all sessions",
    )

    # --- backfill (nested subcommands) ---
    backfill_parser = subparsers.add_parser("backfill", help="Backfill existing results")
    backfill_sub = backfill_parser.add_subparsers(dest="backfill_sub")
    backfill_sub.add_parser("longitudinal", help="Backfill longitudinal store")

    # --- domain (nested subcommands) ---
    domain_parser = subparsers.add_parser(
        "domain", help="Textbook-lecture domain coverage analysis",
    )
    domain_sub = domain_parser.add_subparsers(dest="domain_sub")
    domain_sub.add_parser("extract", help="Extract textbook concepts")
    domain_sub.add_parser("coverage", help="Analyze lecture coverage")
    domain_sub.add_parser("report", help="Generate coverage PDF report")

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Unified CLI entry point for all forma commands."""
    parser = _build_parser()
    args, remaining = parser.parse_known_args(argv)

    # No command given: print help and exit
    if not args.command:
        parser.print_help()
        raise SystemExit(0)

    # Resolve the delegate key
    command = args.command

    if command in _NESTED_GROUPS:
        sub_attr = f"{command}_sub"
        sub_cmd = getattr(args, sub_attr, None)
        registered_subs = {k[1] for k in _COMMANDS if k[0] == command and k[1] is not None}
        # For commands without argparse subparsers (e.g., eval), check remaining
        if sub_cmd is None and remaining:
            candidate = remaining[0]
            if candidate in registered_subs:
                sub_cmd = candidate
                remaining = remaining[1:]
        if sub_cmd and sub_cmd in registered_subs:
            key = (command, sub_cmd)
        else:
            key = (command, None)
            if sub_cmd:
                remaining = [sub_cmd] + remaining
    else:
        key = (command, None)

    if key not in _COMMANDS:
        # Unknown subcommand
        print(
            _korean_error("unknown_command", cmd=command),
            file=sys.stderr,
        )
        raise SystemExit(2)

    module_path, func_name = _COMMANDS[key]
    delegate = _import_delegate(module_path, func_name)

    # Build the argv to pass to the delegate.
    # Delegate modules parse their own sys.argv or accept argv parameter.
    # We set sys.argv so that modules using parser.parse_args() (no argv param)
    # also work correctly.
    delegate_argv = remaining

    old_argv = sys.argv
    try:
        sys.argv = [module_path] + delegate_argv
        delegate()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
