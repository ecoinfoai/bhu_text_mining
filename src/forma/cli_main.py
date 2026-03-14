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
# Korean error messages (FR-048)
# ---------------------------------------------------------------------------

_ERROR_MESSAGES: dict[str, str] = {
    "file_not_found": "파일을 찾을 수 없습니다: {path}",
    "yaml_parse_error": "YAML 파싱 오류: {path}",
    "unknown_command": "알 수 없는 명령 '{cmd}'. 'forma --help'로 사용 가능한 명령을 확인하세요.",
    "smtp_auth_failure": "SMTP 인증 실패: 사용자명 또는 비밀번호를 확인하세요.",
}


def _korean_error(msg_type: str, **kwargs: str) -> str:
    """Return a Korean error message for the given type.

    Args:
        msg_type: Key in ``_ERROR_MESSAGES``.
        **kwargs: Format parameters for the message template.

    Returns:
        Formatted Korean error message string.
    """
    template = _ERROR_MESSAGES.get(msg_type, msg_type)
    return template.format(**kwargs)


# ---------------------------------------------------------------------------
# Progress logging (FR-049)
# ---------------------------------------------------------------------------


def log_progress(current: int, total: int, task_name: str) -> None:
    """Emit a progress log line in ``[N/총수]`` format.

    Args:
        current: Current item number.
        total: Total item count.
        task_name: Description of the task being performed.
    """
    logger.info("[%d/%d] %s", current, total, task_name)


# ---------------------------------------------------------------------------
# Legacy deprecation wrapper factory (FR-019)
# ---------------------------------------------------------------------------


def _make_legacy_wrapper(legacy_name: str, new_name: str, target_main):
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
            f"'{legacy_name}'는 향후 버전에서 제거됩니다. '{new_name}'를 사용하세요.",
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
}

# Groups that have nested subcommands
_NESTED_GROUPS = {"report", "train", "eval", "lecture"}


def _import_delegate(module_path: str, func_name: str):
    """Lazily import and return the delegate function."""
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


class _FormaParser(argparse.ArgumentParser):
    """Custom parser that emits Korean error for unknown subcommands."""

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
        "--verbose", action="store_true", default=False, help="상세 출력 활성화"
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        default=False,
        help="forma.yaml 로딩 건너뛰기",
    )
    parser.add_argument("--font-path", default=None, help="한글 폰트 경로")
    parser.add_argument("--dpi", type=int, default=150, help="차트 해상도")

    subparsers = parser.add_subparsers(dest="command", title="명령")

    # --- Simple commands (no nested subcommands) ---
    subparsers.add_parser("exam", help="시험지 생성")
    subparsers.add_parser("ocr", help="OCR 스캔 처리")
    subparsers.add_parser("intervention", help="중재 활동 관리")
    subparsers.add_parser("deliver", help="리포트 이메일 발송")
    subparsers.add_parser("init", help="프로젝트 설정 초기화")
    subparsers.add_parser("select", help="학생 답안 선별")

    # --- eval (optional 'batch' subcommand — parsed manually to avoid
    #     argparse consuming --class values as subcommand names) ---
    subparsers.add_parser("eval", help="평가 파이프라인 실행")

    # --- report (nested subcommands) ---
    report_parser = subparsers.add_parser("report", help="리포트 생성")
    report_sub = report_parser.add_subparsers(dest="report_sub")
    report_sub.add_parser("student", help="학생 개별 리포트")
    report_sub.add_parser("professor", help="교수용 리포트")
    report_sub.add_parser("longitudinal", help="종단 분석 리포트")
    report_sub.add_parser("warning", help="조기 경보 리포트")
    report_sub.add_parser("batch", help="배치 리포트 생성")

    # --- train (nested subcommands) ---
    train_parser = subparsers.add_parser("train", help="모델 학습")
    train_sub = train_parser.add_subparsers(dest="train_sub")
    train_sub.add_parser("risk", help="드롭 리스크 예측 모델 학습")
    train_sub.add_parser("grade", help="성적 예측 모델 학습")

    # --- lecture (nested subcommands) ---
    lecture_parser = subparsers.add_parser(
        "lecture", help="강의 녹취록 분석",
    )
    lecture_sub = lecture_parser.add_subparsers(
        dest="lecture_sub",
    )
    lecture_sub.add_parser("analyze", help="단일 녹취록 분석")
    lecture_sub.add_parser(
        "compare", help="동일 세션 반 간 비교",
    )
    lecture_sub.add_parser(
        "class-compare", help="전체 세션 반 간 비교",
    )

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
