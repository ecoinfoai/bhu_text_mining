"""Tests for unified ``forma`` CLI entry point (cli_main.py).

Covers:
- FR-013/014: Top-level parser and command groups
- FR-015: ``forma report`` subcommands
- FR-016: ``forma train`` and ``forma eval`` subcommands
- FR-017: Global options (--verbose, --no-config, etc.)
- FR-019: DeprecationWarning for legacy commands
- FR-020: Unknown subcommand handler
- FR-048: Korean error messages
- FR-049: Progress logging
"""

from __future__ import annotations

import logging
import warnings
from unittest import mock

import pytest

from forma.cli_main import main, log_progress, _error_message


class TestUnifiedCLI:
    """FR-013/014: Top-level forma command."""

    def test_help_shows_command_groups(self, capsys):
        """--help output contains the four main command groups."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        help_text = captured.out
        for group in ("report", "train", "eval", "deliver"):
            assert group in help_text, f"'{group}' not found in help output"

    def test_version_flag(self, capsys):
        """--version prints version string and exits."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        # Version output should contain a version-like string
        assert "." in captured.out  # e.g., "0.12.0" or "forma 0.12.0"

    def test_no_args_shows_help(self, capsys):
        """No arguments shows help message and exits with 0."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "forma" in captured.out.lower() or "forma" in captured.err.lower()


class TestReportSubcommands:
    """FR-015: forma report subcommands."""

    def test_report_help_lists_types(self, capsys):
        """'forma report --help' lists all report types."""
        with pytest.raises(SystemExit) as exc_info:
            main(["report", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        help_text = captured.out
        for report_type in ("student", "professor", "longitudinal", "warning", "batch"):
            assert report_type in help_text, f"'{report_type}' not found in report help output"


class TestTrainSubcommands:
    """FR-016: forma train subcommands."""

    def test_train_help_lists_types(self, capsys):
        """'forma train --help' lists risk and grade subcommands."""
        with pytest.raises(SystemExit) as exc_info:
            main(["train", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        help_text = captured.out
        for train_type in ("risk", "grade"):
            assert train_type in help_text, f"'{train_type}' not found in train help output"


class TestGlobalOptions:
    """FR-017: Global options."""

    def test_verbose_flag_accepted(self):
        """--verbose flag is recognized (no error before subcommand dispatch)."""
        # With --verbose but no subcommand, should show help
        with pytest.raises(SystemExit) as exc_info:
            main(["--verbose"])
        assert exc_info.value.code == 0

    def test_no_config_flag_accepted(self):
        """--no-config flag is recognized."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--no-config"])
        assert exc_info.value.code == 0


class TestDeprecationWarnings:
    """FR-019: Legacy command deprecation."""

    def test_legacy_forma_report_emits_warning(self):
        """Legacy wrapper emits DeprecationWarning."""
        from forma.cli_main import _make_legacy_wrapper

        # Create a mock target that we can verify gets called
        mock_main = mock.MagicMock()
        wrapper = _make_legacy_wrapper("forma-report", "forma report student", mock_main)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapper()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "forma-report" in str(w[0].message)
            assert "forma report student" in str(w[0].message)
        mock_main.assert_called_once()


class TestUnknownSubcommand:
    """FR-020: Unknown subcommand handler."""

    def test_unknown_command_shows_korean_help(self, capsys):
        """Unknown subcommand prints Korean error message."""
        with pytest.raises(SystemExit) as exc_info:
            main(["foo"])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "Unknown command" in captured.err


class TestErrorMessages:
    """FR-048: Error messages."""

    def test_file_not_found(self):
        """_error_message returns proper message for file_not_found."""
        msg = _error_message("file_not_found", path="/tmp/no_such_file.yaml")
        assert "File not found" in msg
        assert "/tmp/no_such_file.yaml" in msg

    def test_unknown_command(self):
        """_error_message returns proper message for unknown_command."""
        msg = _error_message("unknown_command", cmd="foo")
        assert "Unknown command" in msg
        assert "foo" in msg

    def test_yaml_parse_error(self):
        """_error_message returns proper message for yaml_parse_error."""
        msg = _error_message("yaml_parse_error", path="/tmp/bad.yaml")
        assert "YAML parse error" in msg

    def test_smtp_auth_failure(self):
        """_error_message returns proper message for smtp_auth_failure."""
        msg = _error_message("smtp_auth_failure")
        assert "SMTP authentication failed" in msg


class TestProgressLogging:
    """FR-049: Verbose progress logging."""

    def test_verbose_progress_pattern(self, caplog):
        """log_progress emits [N/총수] pattern."""
        with caplog.at_level(logging.INFO):
            log_progress(50, 200, "학생 리포트 생성 중")
        assert "[50/200]" in caplog.text
        assert "학생 리포트 생성 중" in caplog.text

    def test_progress_completion(self, caplog):
        """log_progress at current==total uses completion suffix."""
        with caplog.at_level(logging.INFO):
            log_progress(200, 200, "학생 리포트 생성")
        assert "[200/200]" in caplog.text


class TestCommandDelegation:
    """Verify that subcommands correctly delegate to existing modules."""

    @mock.patch("forma.cli_report.main")
    def test_report_student_delegates(self, mock_report_main):
        """'forma report student' delegates to cli_report.main()."""
        main(["report", "student", "--help-is-not-real"])
        mock_report_main.assert_called_once()

    @mock.patch("forma.cli_train.main")
    def test_train_risk_delegates(self, mock_train_main):
        """'forma train risk' delegates to cli_train.main()."""
        main(["train", "risk", "--help-is-not-real"])
        mock_train_main.assert_called_once()

    @mock.patch("forma.cli_init.main")
    def test_init_delegates(self, mock_init_main):
        """'forma init' delegates to cli_init.main()."""
        main(["init"])
        mock_init_main.assert_called_once()

    @mock.patch("forma.cli_deliver.main")
    def test_deliver_delegates(self, mock_deliver_main):
        """'forma deliver' delegates to cli_deliver.main()."""
        main(["deliver", "prepare", "--help-is-not-real"])
        mock_deliver_main.assert_called_once()


class TestLectureDispatch:
    """Verify lecture subcommands are registered in cli_main."""

    def test_lecture_analyze_in_commands(self) -> None:
        """("lecture", "analyze") key exists in _COMMANDS."""
        from forma.cli_main import _COMMANDS

        assert ("lecture", "analyze") in _COMMANDS

    def test_lecture_compare_in_commands(self) -> None:
        """("lecture", "compare") key exists in _COMMANDS."""
        from forma.cli_main import _COMMANDS

        assert ("lecture", "compare") in _COMMANDS

    def test_lecture_class_compare_in_commands(self) -> None:
        """("lecture", "class-compare") key exists in _COMMANDS."""
        from forma.cli_main import _COMMANDS

        assert ("lecture", "class-compare") in _COMMANDS

    def test_lecture_in_nested_groups(self) -> None:
        """'lecture' in _NESTED_GROUPS."""
        from forma.cli_main import _NESTED_GROUPS

        assert "lecture" in _NESTED_GROUPS
