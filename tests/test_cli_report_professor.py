"""Tests for cli_report_professor.py — forma-report-professor CLI entry point.

T044: Tests for _build_parser() — argument definitions and defaults.
T045: Tests for main() — success path and various error exit codes.

These tests are written in the RED phase: cli_report_professor.py does not
exist yet, so all tests will fail with ImportError until the module is
implemented.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from forma.cli_report_professor import _build_parser, main


# ---------------------------------------------------------------------------
# Sample mock data helpers
# ---------------------------------------------------------------------------


def _make_mock_professor_report_data():
    """Build a minimal mock ProfessorReportData-like object."""
    mock = MagicMock()
    mock.class_name = "1A"
    mock.week_num = 1
    mock.subject = "생리학"
    mock.exam_title = "Ch01 서론 형성평가"
    mock.n_students = 5
    mock.n_questions = 2
    return mock


def _make_mock_students(count: int = 5):
    """Build a list of mock StudentReportData objects."""
    students = []
    for i in range(count):
        s = MagicMock()
        s.student_id = f"S{i + 1:03d}"
        s.real_name = f"학생{i + 1}"
        s.student_number = f"202600{i + 1:04d}"
        s.class_name = "1A"
        s.week_num = 1
        s.questions = [MagicMock(), MagicMock()]
        students.append(s)
    return students


# ---------------------------------------------------------------------------
# T044: _build_parser() tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    """T044: Tests for _build_parser() — argument definitions and defaults."""

    def test_build_parser_returns_argument_parser(self):
        """_build_parser() returns an argparse.ArgumentParser."""
        import argparse
        parser = _build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_build_parser_has_required_final_arg(self):
        """Parser includes --final as a required argument."""
        parser = _build_parser()
        # Parsing without --final should raise SystemExit (missing required arg)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--config", "config.yaml",
                "--eval-dir", "eval/",
                "--output-dir", "out/",
            ])

    def test_build_parser_has_required_config_arg(self):
        """Parser includes --config as a required argument."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--final", "final.yaml",
                "--eval-dir", "eval/",
                "--output-dir", "out/",
            ])

    def test_build_parser_has_required_eval_dir_arg(self):
        """Parser includes --eval-dir as a required argument."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--final", "final.yaml",
                "--config", "config.yaml",
                "--output-dir", "out/",
            ])

    def test_build_parser_has_required_output_dir_arg(self):
        """Parser includes --output-dir as a required argument."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--final", "final.yaml",
                "--config", "config.yaml",
                "--eval-dir", "eval/",
            ])

    def test_build_parser_accepts_all_required_args(self):
        """Parser succeeds with all required args provided."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
        ])
        assert args.final == "final.yaml"
        assert args.config == "config.yaml"
        assert args.eval_dir == "eval/"
        assert args.output_dir == "out/"

    def test_build_parser_dpi_default_is_150(self):
        """--dpi defaults to 150 when not provided."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
        ])
        assert args.dpi == 150

    def test_build_parser_dpi_accepts_int(self):
        """--dpi accepts an integer value."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
            "--dpi", "300",
        ])
        assert args.dpi == 300
        assert isinstance(args.dpi, int)

    def test_build_parser_verbose_default_is_false(self):
        """--verbose defaults to False (store_true pattern)."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
        ])
        assert args.verbose is False

    def test_build_parser_verbose_flag_sets_true(self):
        """--verbose flag sets verbose to True."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
            "--verbose",
        ])
        assert args.verbose is True

    def test_build_parser_skip_llm_default_is_false(self):
        """--skip-llm defaults to False."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
        ])
        assert args.skip_llm is False

    def test_build_parser_skip_llm_flag_sets_true(self):
        """--skip-llm flag sets skip_llm to True."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
            "--skip-llm",
        ])
        assert args.skip_llm is True

    def test_build_parser_optional_forma_config(self):
        """--forma-config is optional and defaults to None."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
        ])
        assert args.forma_config is None

    def test_build_parser_optional_class_name(self):
        """--class-name is optional and defaults to None."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
        ])
        assert args.class_name is None

    def test_build_parser_class_name_accepts_string(self):
        """--class-name accepts a string value."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
            "--class-name", "1A분반",
        ])
        assert args.class_name == "1A분반"

    def test_build_parser_optional_font_path(self):
        """--font-path is optional and defaults to None."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
        ])
        assert args.font_path is None

    def test_build_parser_help_text_is_non_empty(self):
        """Parser has a non-empty description/help text."""
        parser = _build_parser()
        assert parser.description is not None
        assert len(parser.description.strip()) > 0

    def test_build_parser_transcript_dir_optional(self):
        """--transcript-dir is optional and defaults to None (FR-019a)."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
        ])
        assert args.transcript_dir is None

    def test_build_parser_transcript_dir_accepts_string(self):
        """--transcript-dir accepts a path string (FR-019a)."""
        parser = _build_parser()
        args = parser.parse_args([
            "--final", "final.yaml",
            "--config", "config.yaml",
            "--eval-dir", "eval/",
            "--output-dir", "out/",
            "--transcript-dir", "/some/transcript/dir",
        ])
        assert args.transcript_dir == "/some/transcript/dir"


# ---------------------------------------------------------------------------
# T045: main() tests
# ---------------------------------------------------------------------------


def _base_argv(cli_env: dict) -> list[str]:
    """Build a complete argv list from cli_env fixture."""
    return [
        "--final", cli_env["final"],
        "--config", cli_env["config"],
        "--eval-dir", cli_env["eval_dir"],
        "--output-dir", cli_env["output_dir"],
        "--skip-llm",
    ]


@pytest.fixture()
def cli_env(tmp_path):
    """Create minimal filesystem structure required by the professor CLI.

    Returns a dict with paths to the fake --final, --config, --eval-dir,
    and --output-dir arguments.
    """
    # --final: a YAML file that must exist
    final_file = tmp_path / "anp_1A_final.yaml"
    final_file.write_text("[]", encoding="utf-8")

    # --config: a YAML file that must exist
    config_file = tmp_path / "Ch01_exam_config.yaml"
    config_file.write_text("questions: []", encoding="utf-8")

    # --eval-dir: directory that must exist
    eval_dir = tmp_path / "eval_1A"
    eval_dir.mkdir()

    # --output-dir: output directory for PDF
    output_dir = tmp_path / "reports"
    output_dir.mkdir()

    return {
        "final": str(final_file),
        "config": str(config_file),
        "eval_dir": str(eval_dir),
        "output_dir": str(output_dir),
    }


class TestMain:
    """T045: Tests for main() — success path and error exit codes."""

    # -------------------------------------------------------------------------
    # Success path
    # -------------------------------------------------------------------------

    def test_main_success_path_exits_0(self, cli_env, monkeypatch):
        """Success path: all mocked correctly → exits with code 0 or returns normally."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            *_base_argv(cli_env),
        ])

        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()
        mock_generator = MagicMock()
        mock_generator.generate_pdf.return_value = str(
            cli_env["output_dir"] + "/professor_report.pdf"
        )

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=mock_generator,
        ):
            try:
                result = main()
                # If main() returns an int, it should be 0
                assert result in (None, 0)
            except SystemExit as exc:
                assert exc.code == 0

    def test_main_success_calls_generate_pdf(self, cli_env, monkeypatch):
        """Success path: ProfessorPDFReportGenerator.generate_pdf is called once."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            *_base_argv(cli_env),
        ])

        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()
        mock_generator = MagicMock()
        mock_generator.generate_pdf.return_value = str(
            cli_env["output_dir"] + "/professor_report.pdf"
        )

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=mock_generator,
        ):
            try:
                main()
            except SystemExit:
                pass

        mock_generator.generate_pdf.assert_called_once()

    def test_main_success_calls_build_professor_report_data(self, cli_env, monkeypatch):
        """Success path: build_professor_report_data is called with loaded students."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            *_base_argv(cli_env),
        ])

        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()
        mock_generator = MagicMock()
        mock_generator.generate_pdf.return_value = str(
            cli_env["output_dir"] + "/professor_report.pdf"
        )

        mock_build = MagicMock(return_value=mock_report_data)

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            mock_build,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=mock_generator,
        ):
            try:
                main()
            except SystemExit:
                pass

        mock_build.assert_called_once()

    # -------------------------------------------------------------------------
    # Missing/non-existent input files → exit code 1
    # -------------------------------------------------------------------------

    def test_main_missing_final_file_exits_1(self, cli_env, monkeypatch):
        """--final pointing to nonexistent file → sys.exit(1)."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            "--final", "/nonexistent/path/final.yaml",
            "--config", cli_env["config"],
            "--eval-dir", cli_env["eval_dir"],
            "--output-dir", cli_env["output_dir"],
            "--skip-llm",
        ])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_main_missing_config_file_exits_1(self, cli_env, monkeypatch):
        """--config pointing to nonexistent file → sys.exit(1)."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            "--final", cli_env["final"],
            "--config", "/nonexistent/path/config.yaml",
            "--eval-dir", cli_env["eval_dir"],
            "--output-dir", cli_env["output_dir"],
            "--skip-llm",
        ])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_main_missing_eval_dir_exits_1(self, cli_env, monkeypatch):
        """--eval-dir pointing to nonexistent directory → sys.exit(1)."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            "--final", cli_env["final"],
            "--config", cli_env["config"],
            "--eval-dir", "/nonexistent/eval_dir",
            "--output-dir", cli_env["output_dir"],
            "--skip-llm",
        ])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_main_requires_final_arg_missing_exits(self, monkeypatch, tmp_path):
        """Missing --final argument entirely → argparse sys.exit (code 1 or 2)."""
        config = tmp_path / "config.yaml"
        config.write_text("questions: []")
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            "--config", str(config),
            "--eval-dir", str(eval_dir),
            "--output-dir", str(out_dir),
            "--skip-llm",
        ])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code in (1, 2)

    def test_main_requires_config_arg_missing_exits(self, monkeypatch, tmp_path):
        """Missing --config argument entirely → argparse sys.exit (code 1 or 2)."""
        final_file = tmp_path / "final.yaml"
        final_file.write_text("[]")
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            "--final", str(final_file),
            "--eval-dir", str(eval_dir),
            "--output-dir", str(out_dir),
            "--skip-llm",
        ])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code in (1, 2)

    def test_main_requires_eval_dir_arg_missing_exits(self, monkeypatch, tmp_path):
        """Missing --eval-dir argument entirely → argparse sys.exit (code 1 or 2)."""
        final_file = tmp_path / "final.yaml"
        final_file.write_text("[]")
        config = tmp_path / "config.yaml"
        config.write_text("questions: []")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            "--final", str(final_file),
            "--config", str(config),
            "--output-dir", str(out_dir),
            "--skip-llm",
        ])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code in (1, 2)

    def test_main_requires_output_dir_arg_missing_exits(self, monkeypatch, tmp_path):
        """Missing --output-dir argument entirely → argparse sys.exit (code 1 or 2)."""
        final_file = tmp_path / "final.yaml"
        final_file.write_text("[]")
        config = tmp_path / "config.yaml"
        config.write_text("questions: []")
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            "--final", str(final_file),
            "--config", str(config),
            "--eval-dir", str(eval_dir),
            "--skip-llm",
        ])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code in (1, 2)

    # -------------------------------------------------------------------------
    # Insufficient students (<3) → exit code 2
    # -------------------------------------------------------------------------

    def test_main_insufficient_students_exits_2(self, cli_env, monkeypatch):
        """When load_all_student_data returns <3 students → sys.exit(2)."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            *_base_argv(cli_env),
        ])

        # Only 2 students — below the required minimum of 3
        mock_students = _make_mock_students(2)
        mock_distributions = MagicMock()

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 2

    def test_main_zero_students_exits_2(self, cli_env, monkeypatch):
        """When load_all_student_data returns 0 students → sys.exit(2)."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            *_base_argv(cli_env),
        ])

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=([], MagicMock()),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 2

    def test_main_exactly_3_students_does_not_exit_2(self, cli_env, monkeypatch):
        """When load_all_student_data returns exactly 3 students, no exit(2) raised."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            *_base_argv(cli_env),
        ])

        mock_students = _make_mock_students(3)
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()
        mock_generator = MagicMock()
        mock_generator.generate_pdf.return_value = str(
            cli_env["output_dir"] + "/professor_report.pdf"
        )

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=mock_generator,
        ):
            try:
                main()
            except SystemExit as exc:
                # Should not exit with code 2
                assert exc.code != 2

    # -------------------------------------------------------------------------
    # Font not found → exit code 3
    # -------------------------------------------------------------------------

    def test_main_font_not_found_exits_3(self, cli_env, monkeypatch):
        """--font-path pointing to nonexistent file → sys.exit(3)."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            "--final", cli_env["final"],
            "--config", cli_env["config"],
            "--eval-dir", cli_env["eval_dir"],
            "--output-dir", cli_env["output_dir"],
            "--skip-llm",
            "--font-path", "/nonexistent/NanumGothic.ttf",
        ])

        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 3

    def test_main_generator_file_not_found_exits_3(self, cli_env, monkeypatch):
        """ProfessorPDFReportGenerator raising FileNotFoundError → sys.exit(3)."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            *_base_argv(cli_env),
        ])

        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()

        def _raise_file_not_found(*args, **kwargs):
            raise FileNotFoundError("Korean font not found: /path/NanumGothic.ttf")

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            side_effect=_raise_file_not_found,
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 3

    # -------------------------------------------------------------------------
    # --skip-llm behavior
    # -------------------------------------------------------------------------

    def test_main_skip_llm_does_not_call_generate_professor_analysis(
        self, cli_env, monkeypatch
    ):
        """--skip-llm: generate_professor_analysis is NOT called."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            *_base_argv(cli_env),  # already includes --skip-llm
        ])

        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()
        mock_generator = MagicMock()
        mock_generator.generate_pdf.return_value = str(
            cli_env["output_dir"] + "/professor_report.pdf"
        )

        mock_llm_fn = MagicMock()

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=mock_generator,
        ), patch(
            "forma.cli_report_professor.generate_professor_analysis",
            mock_llm_fn,
        ):
            try:
                main()
            except SystemExit:
                pass

        mock_llm_fn.assert_not_called()

    def test_main_without_skip_llm_calls_generate_professor_analysis(
        self, cli_env, monkeypatch
    ):
        """Without --skip-llm: generate_professor_analysis IS called."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            "--final", cli_env["final"],
            "--config", cli_env["config"],
            "--eval-dir", cli_env["eval_dir"],
            "--output-dir", cli_env["output_dir"],
            # No --skip-llm
        ])

        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()
        mock_generator = MagicMock()
        mock_generator.generate_pdf.return_value = str(
            cli_env["output_dir"] + "/professor_report.pdf"
        )

        mock_llm_fn = MagicMock()

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=mock_generator,
        ), patch(
            "forma.cli_report_professor.generate_professor_analysis",
            mock_llm_fn,
        ):
            try:
                main()
            except SystemExit:
                pass

        mock_llm_fn.assert_called_once()

    # -------------------------------------------------------------------------
    # --verbose sets logging to DEBUG
    # -------------------------------------------------------------------------

    def test_main_verbose_sets_logging_debug(self, cli_env, monkeypatch):
        """--verbose flag configures the root logger to DEBUG level."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            *_base_argv(cli_env),
            "--verbose",
        ])

        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()
        mock_generator = MagicMock()
        mock_generator.generate_pdf.return_value = str(
            cli_env["output_dir"] + "/professor_report.pdf"
        )

        mock_basic_config = MagicMock()

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=mock_generator,
        ), patch(
            "logging.basicConfig",
            mock_basic_config,
        ):
            try:
                main()
            except SystemExit:
                pass

        # Verify that logging was configured with DEBUG level
        calls = mock_basic_config.call_args_list
        assert any(
            call.kwargs.get("level") == logging.DEBUG
            or (call.args and call.args[0] == logging.DEBUG)
            for call in calls
        ), f"Expected DEBUG level in logging.basicConfig calls: {calls}"

    def test_main_without_verbose_uses_info_level(self, cli_env, monkeypatch):
        """Without --verbose, logging is configured at INFO level."""
        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            *_base_argv(cli_env),
            # No --verbose
        ])

        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()
        mock_generator = MagicMock()
        mock_generator.generate_pdf.return_value = str(
            cli_env["output_dir"] + "/professor_report.pdf"
        )

        mock_basic_config = MagicMock()

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=mock_generator,
        ), patch(
            "logging.basicConfig",
            mock_basic_config,
        ):
            try:
                main()
            except SystemExit:
                pass

        # Verify that logging was configured without DEBUG level
        calls = mock_basic_config.call_args_list
        # Should have been called at least once, with INFO level (not DEBUG)
        if calls:
            last_call = calls[-1]
            level_used = last_call.kwargs.get("level") or (
                last_call.args[0] if last_call.args else None
            )
            assert level_used != logging.DEBUG, (
                f"Expected INFO (not DEBUG) level in logging.basicConfig: {calls}"
            )


# ---------------------------------------------------------------------------
# T045-transcript: --transcript-dir integration tests (FR-019a, SC-005)
# ---------------------------------------------------------------------------


class TestTranscriptDir:
    """Tests for --transcript-dir argument integration (FR-019a)."""

    def _run_main_with_mocks(self, monkeypatch, argv, mock_report_data=None):
        """Helper: run main() with standard mocks, return report_data."""
        if mock_report_data is None:
            mock_report_data = _make_mock_professor_report_data()
            mock_report_data.question_stats = []
        monkeypatch.setattr("sys.argv", ["forma-report-professor", *argv])
        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_generator = MagicMock()
        captured = {}

        def capture_build(**kwargs):
            return mock_report_data

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            side_effect=lambda *a, **kw: mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=mock_generator,
        ):
            try:
                main()
            except SystemExit:
                pass

        captured["report_data"] = mock_report_data
        return captured

    def test_no_transcript_dir_does_not_call_compute_emphasis_map(
        self, tmp_path, monkeypatch
    ):
        """SC-005: Without --transcript-dir, compute_emphasis_map is not called (backward compat)."""
        final_file = tmp_path / "final.yaml"
        final_file.write_text("[]", encoding="utf-8")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("questions: []", encoding="utf-8")
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        mock_report_data = _make_mock_professor_report_data()
        mock_report_data.question_stats = []

        argv = [
            "--final", str(final_file),
            "--config", str(config_file),
            "--eval-dir", str(eval_dir),
            "--output-dir", str(out_dir),
            "--skip-llm",
        ]

        monkeypatch.setattr("sys.argv", ["forma-report-professor", *argv])
        mock_students = _make_mock_students(5)

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, MagicMock()),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=MagicMock(),
        ), patch(
            "forma.emphasis_map.compute_emphasis_map",
        ) as mock_cem:
            try:
                main()
            except SystemExit:
                pass

        # compute_emphasis_map must NOT be called without --transcript-dir
        mock_cem.assert_not_called()

    def test_transcript_dir_nonexistent_logs_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        """--transcript-dir pointing to non-existent dir logs a warning and continues."""
        final_file = tmp_path / "final.yaml"
        final_file.write_text("[]", encoding="utf-8")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("questions: []", encoding="utf-8")
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        nonexistent = str(tmp_path / "no_such_dir")
        argv = [
            "--final", str(final_file),
            "--config", str(config_file),
            "--eval-dir", str(eval_dir),
            "--output-dir", str(out_dir),
            "--skip-llm",
            "--transcript-dir", nonexistent,
        ]

        mock_report_data = _make_mock_professor_report_data()
        mock_report_data.question_stats = []

        with caplog.at_level(logging.WARNING, logger="forma.cli_report_professor"):
            self._run_main_with_mocks(monkeypatch, argv, mock_report_data)

        # Should have logged a warning about non-existent dir
        assert any(
            "transcript directory not found" in record.message.lower()
            for record in caplog.records
        ), f"Expected transcript warning, got: {[r.message for r in caplog.records]}"

    def test_transcript_dir_with_txt_files_calls_compute_emphasis_map(
        self, tmp_path, monkeypatch
    ):
        """--transcript-dir with .txt files → compute_emphasis_map is called (FR-019a)."""
        final_file = tmp_path / "final.yaml"
        final_file.write_text("[]", encoding="utf-8")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("questions: []", encoding="utf-8")
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()
        (transcript_dir / "lecture1.txt").write_text(
            "심장은 혈액을 순환시키는 기관이다.\n펌프 기능이 핵심이다.",
            encoding="utf-8",
        )

        # Set up question_stats with concept_mastery_rates
        mock_qstat = MagicMock()
        mock_qstat.concept_mastery_rates = {"심장": 0.8, "혈액순환": 0.6}
        mock_report_data = _make_mock_professor_report_data()
        mock_report_data.question_stats = [mock_qstat]

        argv = [
            "--final", str(final_file),
            "--config", str(config_file),
            "--eval-dir", str(eval_dir),
            "--output-dir", str(out_dir),
            "--skip-llm",
            "--transcript-dir", str(transcript_dir),
        ]

        mock_emphasis_map = MagicMock()
        mock_emphasis_map.n_concepts = 2
        mock_emphasis_map.n_sentences = 2
        mock_emphasis_map.concept_scores = {"심장": 0.9, "혈액순환": 0.5}
        mock_gap_report = MagicMock()

        monkeypatch.setattr("sys.argv", ["forma-report-professor", *argv])
        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_generator = MagicMock()

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=mock_generator,
        ), patch(
            "forma.emphasis_map.compute_emphasis_map",
            return_value=mock_emphasis_map,
        ) as mock_cem, patch(
            "forma.lecture_gap_analysis.compute_lecture_gap",
            return_value=mock_gap_report,
        ) as mock_clg:
            try:
                main()
            except SystemExit:
                pass

        # compute_emphasis_map should have been called
        mock_cem.assert_called_once()
        # compute_lecture_gap should have been called
        mock_clg.assert_called_once()


# ---------------------------------------------------------------------------
# T048: US4 — concept_dependencies wiring in professor report CLI (FR-021)
# ---------------------------------------------------------------------------


class TestConceptDepsWiring:
    """T048 [US4] concept_dependencies wiring in forma-report-professor CLI."""

    def test_deficit_map_passed_to_generate_pdf_when_deps_present(
        self, cli_env, monkeypatch, tmp_path,
    ):
        """When exam YAML has concept_dependencies, deficit_map is passed to generate_pdf (FR-021)."""
        import yaml

        config_file = tmp_path / "exam_with_deps.yaml"
        config_data = {
            "questions": [],
            "concept_dependencies": [
                {"prerequisite": "세포막", "dependent": "삼투압"},
            ],
        }
        config_file.write_text(yaml.dump(config_data, allow_unicode=True), encoding="utf-8")

        # Build students with concept details
        mock_students = []
        for i in range(5):
            s = MagicMock()
            s.student_id = f"S{i + 1:03d}"
            s.real_name = f"학생{i + 1}"
            s.student_number = f"202600{i + 1:04d}"
            s.class_name = "1A"
            s.week_num = 1
            q = MagicMock()
            q.question_sn = 1
            concept1 = MagicMock()
            concept1.concept = "세포막"
            concept1.similarity = 0.8
            concept2 = MagicMock()
            concept2.concept = "삼투압"
            concept2.similarity = 0.3 if i < 3 else 0.7
            q.concepts = [concept1, concept2]
            s.questions = [q]
            mock_students.append(s)

        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()
        mock_report_data.question_stats = []
        mock_generator = MagicMock()
        mock_generator.generate_pdf.return_value = str(
            cli_env["output_dir"] + "/professor_report.pdf"
        )

        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            "--final", cli_env["final"],
            "--config", str(config_file),
            "--eval-dir", cli_env["eval_dir"],
            "--output-dir", cli_env["output_dir"],
            "--skip-llm",
        ])

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=mock_generator,
        ):
            try:
                main()
            except SystemExit:
                pass

        # generate_pdf should have been called with deficit_map kwarg
        mock_generator.generate_pdf.assert_called_once()
        call_kwargs = mock_generator.generate_pdf.call_args.kwargs
        assert "deficit_map" in call_kwargs
        assert call_kwargs["deficit_map"] is not None
        # Check deficit_map has correct structure
        dm = call_kwargs["deficit_map"]
        assert dm.total_students == 5
        assert "세포막" in dm.concept_counts
        assert "삼투압" in dm.concept_counts

    def test_no_deps_in_yaml_deficit_map_none(
        self, cli_env, monkeypatch,
    ):
        """When exam YAML has no concept_dependencies, deficit_map is None (FR-023)."""
        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()
        mock_report_data.question_stats = []
        mock_generator = MagicMock()
        mock_generator.generate_pdf.return_value = str(
            cli_env["output_dir"] + "/professor_report.pdf"
        )

        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            *_base_argv(cli_env),
        ])

        with patch(
            "forma.cli_report_professor.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_professor.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_professor.ProfessorPDFReportGenerator",
            return_value=mock_generator,
        ):
            try:
                main()
            except SystemExit:
                pass

        mock_generator.generate_pdf.assert_called_once()
        call_kwargs = mock_generator.generate_pdf.call_args.kwargs
        assert call_kwargs.get("deficit_map") is None
