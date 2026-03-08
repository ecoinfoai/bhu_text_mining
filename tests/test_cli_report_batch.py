"""Tests for cli_report_batch.py — forma-report-batch CLI entry point.

T011: Tests for create_parser() and main() for the batch multi-class PDF
report generator.

These tests are written in the RED phase: cli_report_batch.py does not
exist yet, so all tests will fail with ImportError until the module is
implemented.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, call, patch

import pytest

from forma.cli_report_batch import create_parser, main


# ---------------------------------------------------------------------------
# Sample mock data helpers
# ---------------------------------------------------------------------------


def _make_mock_professor_report_data(class_name: str = "A"):
    """Build a minimal mock ProfessorReportData-like object."""
    mock = MagicMock()
    mock.class_name = class_name
    mock.week_num = 1
    mock.subject = "생리학"
    mock.exam_title = "Ch01 서론 형성평가"
    mock.n_students = 5
    mock.n_questions = 2
    return mock


def _make_mock_students(count: int = 5, class_name: str = "A"):
    """Build a list of mock StudentReportData objects."""
    students = []
    for i in range(count):
        s = MagicMock()
        s.student_id = f"S{i + 1:03d}"
        s.real_name = f"학생{i + 1}"
        s.student_number = f"202600{i + 1:04d}"
        s.class_name = class_name
        s.week_num = 1
        s.questions = [MagicMock(), MagicMock()]
        students.append(s)
    return students


def _full_required_args(tmp_path) -> list[str]:
    """Return a complete argv list with all required arguments."""
    config = tmp_path / "exam_config.yaml"
    config.write_text("questions: []", encoding="utf-8")
    join_dir = tmp_path / "results"
    join_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    return [
        "--config", str(config),
        "--join-dir", str(join_dir),
        "--join-pattern", "anp_1{class}_final.yaml",
        "--eval-pattern", "eval_1{class}",
        "--output-dir", str(output_dir),
        "--classes", "A", "B",
    ]


# ---------------------------------------------------------------------------
# TestBatchCLIParser: argparse argument definitions and defaults
# ---------------------------------------------------------------------------


class TestBatchCLIParser:
    """Tests for create_parser() — argument definitions and defaults."""

    def test_parser_required_args(self, tmp_path):
        """Parsing all required args succeeds and values are stored correctly."""
        config = tmp_path / "exam.yaml"
        config.write_text("questions: []", encoding="utf-8")
        join_dir = tmp_path / "joins"
        join_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        parser = create_parser()
        args = parser.parse_args([
            "--config", str(config),
            "--join-dir", str(join_dir),
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", str(output_dir),
            "--classes", "A", "B",
        ])

        assert args.config == str(config)
        assert args.join_dir == str(join_dir)
        assert args.join_pattern == "{class}.yaml"
        assert args.eval_pattern == "eval_{class}"
        assert args.output_dir == str(output_dir)
        assert args.classes == ["A", "B"]

    def test_parser_missing_config(self, tmp_path):
        """Missing --config raises SystemExit (argparse required-arg error)."""
        join_dir = tmp_path / "joins"
        join_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--join-dir", str(join_dir),
                "--join-pattern", "{class}.yaml",
                "--eval-pattern", "eval_{class}",
                "--output-dir", str(output_dir),
                "--classes", "A",
            ])

    def test_parser_missing_classes(self, tmp_path):
        """Missing --classes raises SystemExit (argparse required-arg error)."""
        config = tmp_path / "exam.yaml"
        config.write_text("questions: []", encoding="utf-8")
        join_dir = tmp_path / "joins"
        join_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--config", str(config),
                "--join-dir", str(join_dir),
                "--join-pattern", "{class}.yaml",
                "--eval-pattern", "eval_{class}",
                "--output-dir", str(output_dir),
                # --classes omitted
            ])

    def test_parser_aggregate_flag(self, tmp_path):
        """--aggregate flag sets args.aggregate=True."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A",
            "--aggregate",
        ])
        assert args.aggregate is True

    def test_parser_aggregate_default_false(self, tmp_path):
        """--aggregate defaults to False when not provided."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A",
        ])
        assert args.aggregate is False

    def test_parser_no_individual_flag(self, tmp_path):
        """--no-individual flag sets args.no_individual=True."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A",
            "--no-individual",
        ])
        assert args.no_individual is True

    def test_parser_no_individual_default_false(self):
        """--no-individual defaults to False when not provided."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A",
        ])
        assert args.no_individual is False

    def test_parser_default_dpi(self):
        """--dpi defaults to 150 when not provided."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A",
        ])
        assert args.dpi == 150

    def test_parser_dpi_accepts_int(self):
        """--dpi accepts an integer value."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A",
            "--dpi", "300",
        ])
        assert args.dpi == 300
        assert isinstance(args.dpi, int)

    def test_parser_skip_llm_flag(self):
        """--skip-llm flag sets args.skip_llm=True."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A",
            "--skip-llm",
        ])
        assert args.skip_llm is True

    def test_parser_skip_llm_default_false(self):
        """--skip-llm defaults to False when not provided."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A",
        ])
        assert args.skip_llm is False

    def test_parser_verbose_flag(self):
        """--verbose flag sets args.verbose=True."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A",
            "--verbose",
        ])
        assert args.verbose is True

    def test_parser_verbose_default_false(self):
        """--verbose defaults to False when not provided."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A",
        ])
        assert args.verbose is False

    def test_parser_font_path_optional(self):
        """--font-path is optional and defaults to None."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A",
        ])
        assert args.font_path is None

    def test_parser_font_path_accepts_string(self):
        """--font-path accepts a string value."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A",
            "--font-path", "/usr/share/fonts/NanumGothic.ttf",
        ])
        assert args.font_path == "/usr/share/fonts/NanumGothic.ttf"

    def test_parser_classes_accepts_multiple_values(self):
        """--classes nargs='+' accepts multiple class identifiers."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "x.yaml",
            "--join-dir", "d",
            "--join-pattern", "{class}.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out",
            "--classes", "A", "B", "C", "D",
        ])
        assert args.classes == ["A", "B", "C", "D"]

    def test_parser_missing_join_dir(self):
        """Missing --join-dir raises SystemExit."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--config", "x.yaml",
                # --join-dir omitted
                "--join-pattern", "{class}.yaml",
                "--eval-pattern", "eval_{class}",
                "--output-dir", "out",
                "--classes", "A",
            ])

    def test_parser_missing_join_pattern(self):
        """Missing --join-pattern raises SystemExit."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--config", "x.yaml",
                "--join-dir", "d",
                # --join-pattern omitted
                "--eval-pattern", "eval_{class}",
                "--output-dir", "out",
                "--classes", "A",
            ])

    def test_parser_missing_eval_pattern(self):
        """Missing --eval-pattern raises SystemExit."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--config", "x.yaml",
                "--join-dir", "d",
                "--join-pattern", "{class}.yaml",
                # --eval-pattern omitted
                "--output-dir", "out",
                "--classes", "A",
            ])

    def test_parser_missing_output_dir(self):
        """Missing --output-dir raises SystemExit."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--config", "x.yaml",
                "--join-dir", "d",
                "--join-pattern", "{class}.yaml",
                "--eval-pattern", "eval_{class}",
                # --output-dir omitted
                "--classes", "A",
            ])

    def test_parser_returns_argument_parser(self):
        """create_parser() returns an argparse.ArgumentParser."""
        import argparse
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_transcript_pattern_optional(self):
        """--transcript-pattern is optional and defaults to None (FR-019b)."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "config.yaml",
            "--join-dir", "results/",
            "--join-pattern", "anp_{class}_final.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out/",
            "--classes", "A",
        ])
        assert args.transcript_pattern is None

    def test_parser_transcript_pattern_accepts_string(self):
        """--transcript-pattern accepts a path string with {class} placeholder (FR-019b)."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "config.yaml",
            "--join-dir", "results/",
            "--join-pattern", "anp_{class}_final.yaml",
            "--eval-pattern", "eval_{class}",
            "--output-dir", "out/",
            "--classes", "A",
            "--transcript-pattern", "/transcripts/{class}",
        ])
        assert args.transcript_pattern == "/transcripts/{class}"


# ---------------------------------------------------------------------------
# TestBatchCLIRun: integration tests with mocking
# ---------------------------------------------------------------------------


class TestBatchCLIRun:
    """Integration tests for main() with mocked dependencies."""

    def test_single_class_generates_per_class_dir(self, tmp_path, monkeypatch):
        """main() with --classes A creates output_dir/A/ subdirectory."""
        config = tmp_path / "exam.yaml"
        config.write_text("questions: []", encoding="utf-8")
        join_dir = tmp_path / "results"
        join_dir.mkdir()
        # Create the final YAML file that the pattern will resolve to
        final_file = join_dir / "anp_1A_final.yaml"
        final_file.write_text("[]", encoding="utf-8")
        eval_dir = tmp_path / "eval_1A"
        eval_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        monkeypatch.setattr("sys.argv", [
            "forma-report-batch",
            "--config", str(config),
            "--join-dir", str(join_dir),
            "--join-pattern", "anp_1{class}_final.yaml",
            "--eval-pattern", "eval_1{class}",
            "--output-dir", str(output_dir),
            "--classes", "A",
            "--skip-llm",
        ])

        mock_students = _make_mock_students(5, "A")
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data("A")
        mock_prof_gen = MagicMock()
        mock_student_gen = MagicMock()

        with patch(
            "forma.cli_report_batch.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_batch.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_batch.ProfessorPDFReportGenerator",
            return_value=mock_prof_gen,
        ), patch(
            "forma.cli_report_batch.StudentPDFReportGenerator",
            return_value=mock_student_gen,
        ):
            try:
                main()
            except SystemExit as exc:
                assert exc.code in (None, 0), f"Unexpected exit code: {exc.code}"

        # The per-class subdirectory must be created
        class_out = output_dir / "A"
        assert class_out.exists(), f"Expected per-class directory {class_out} to exist"

    def test_missing_section_warns_not_aborts(self, tmp_path, monkeypatch, caplog):
        """When final YAML is missing for one class, a warning is logged but processing continues."""
        config = tmp_path / "exam.yaml"
        config.write_text("questions: []", encoding="utf-8")
        join_dir = tmp_path / "results"
        join_dir.mkdir()
        # Only class B file exists; class A is missing
        final_b = join_dir / "anp_1B_final.yaml"
        final_b.write_text("[]", encoding="utf-8")
        eval_dir_b = tmp_path / "eval_1B"
        eval_dir_b.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        monkeypatch.setattr("sys.argv", [
            "forma-report-batch",
            "--config", str(config),
            "--join-dir", str(join_dir),
            "--join-pattern", "anp_1{class}_final.yaml",
            "--eval-pattern", "eval_1{class}",
            "--output-dir", str(output_dir),
            "--classes", "A", "B",
            "--skip-llm",
        ])

        mock_students = _make_mock_students(5, "B")
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data("B")
        mock_prof_gen = MagicMock()
        mock_student_gen = MagicMock()

        def _load_side_effect(final_path, config_path, eval_dir):
            # Raise for the missing class A file
            if "anp_1A_final.yaml" in str(final_path):
                raise FileNotFoundError(f"File not found: {final_path}")
            return mock_students, mock_distributions

        with patch(
            "forma.cli_report_batch.load_all_student_data",
            side_effect=_load_side_effect,
        ), patch(
            "forma.cli_report_batch.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_batch.ProfessorPDFReportGenerator",
            return_value=mock_prof_gen,
        ), patch(
            "forma.cli_report_batch.StudentPDFReportGenerator",
            return_value=mock_student_gen,
        ), caplog.at_level(logging.WARNING):
            try:
                main()
            except SystemExit as exc:
                assert exc.code in (None, 0), f"Unexpected exit code: {exc.code}"

        # The CLI must not abort entirely; class B should still be processed
        mock_prof_gen.generate_pdf.assert_called()

    def test_aggregate_flag_calls_merge(self, tmp_path, monkeypatch):
        """With --aggregate, merge_professor_report_data is called and aggregate PDF generated."""
        config = tmp_path / "exam.yaml"
        config.write_text("questions: []", encoding="utf-8")
        join_dir = tmp_path / "results"
        join_dir.mkdir()
        for cls in ("A", "B"):
            (join_dir / f"anp_1{cls}_final.yaml").write_text("[]", encoding="utf-8")
            (tmp_path / f"eval_1{cls}").mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        monkeypatch.setattr("sys.argv", [
            "forma-report-batch",
            "--config", str(config),
            "--join-dir", str(join_dir),
            "--join-pattern", "anp_1{class}_final.yaml",
            "--eval-pattern", "eval_1{class}",
            "--output-dir", str(output_dir),
            "--classes", "A", "B",
            "--aggregate",
            "--skip-llm",
        ])

        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_report_data_a = _make_mock_professor_report_data("A")
        mock_report_data_b = _make_mock_professor_report_data("B")
        mock_merged_data = _make_mock_professor_report_data("merged")
        mock_prof_gen = MagicMock()
        mock_student_gen = MagicMock()

        report_datas = [mock_report_data_a, mock_report_data_b]
        call_index = [0]

        def _build_side_effect(*args, **kwargs):
            rd = report_datas[call_index[0] % len(report_datas)]
            call_index[0] += 1
            return rd

        mock_merge = MagicMock(return_value=mock_merged_data)

        with patch(
            "forma.cli_report_batch.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_batch.build_professor_report_data",
            side_effect=_build_side_effect,
        ), patch(
            "forma.cli_report_batch.ProfessorPDFReportGenerator",
            return_value=mock_prof_gen,
        ), patch(
            "forma.cli_report_batch.StudentPDFReportGenerator",
            return_value=mock_student_gen,
        ), patch(
            "forma.cli_report_batch.merge_professor_report_data",
            mock_merge,
        ):
            try:
                main()
            except SystemExit as exc:
                assert exc.code in (None, 0), f"Unexpected exit code: {exc.code}"

        # merge_professor_report_data must have been called
        mock_merge.assert_called_once()
        # And the aggregate PDF must have been generated
        assert mock_prof_gen.generate_pdf.call_count >= 3  # 1 per class + 1 aggregate

    def test_no_individual_skips_student_pdfs(self, tmp_path, monkeypatch):
        """With --no-individual, StudentPDFReportGenerator is never called."""
        config = tmp_path / "exam.yaml"
        config.write_text("questions: []", encoding="utf-8")
        join_dir = tmp_path / "results"
        join_dir.mkdir()
        final_file = join_dir / "anp_1A_final.yaml"
        final_file.write_text("[]", encoding="utf-8")
        eval_dir = tmp_path / "eval_1A"
        eval_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        monkeypatch.setattr("sys.argv", [
            "forma-report-batch",
            "--config", str(config),
            "--join-dir", str(join_dir),
            "--join-pattern", "anp_1{class}_final.yaml",
            "--eval-pattern", "eval_1{class}",
            "--output-dir", str(output_dir),
            "--classes", "A",
            "--no-individual",
            "--skip-llm",
        ])

        mock_students = _make_mock_students(5, "A")
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data("A")
        mock_prof_gen = MagicMock()
        mock_student_gen_cls = MagicMock()

        with patch(
            "forma.cli_report_batch.load_all_student_data",
            return_value=(mock_students, mock_distributions),
        ), patch(
            "forma.cli_report_batch.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_batch.ProfessorPDFReportGenerator",
            return_value=mock_prof_gen,
        ), patch(
            "forma.cli_report_batch.StudentPDFReportGenerator",
            mock_student_gen_cls,
        ):
            try:
                main()
            except SystemExit as exc:
                assert exc.code in (None, 0), f"Unexpected exit code: {exc.code}"

        # StudentPDFReportGenerator must NOT have been instantiated or called
        mock_student_gen_cls.assert_not_called()

    def test_multiple_classes_processed(self, tmp_path, monkeypatch):
        """With --classes A B C, load_all_student_data is called once per class."""
        config = tmp_path / "exam.yaml"
        config.write_text("questions: []", encoding="utf-8")
        join_dir = tmp_path / "results"
        join_dir.mkdir()
        for cls in ("A", "B", "C"):
            (join_dir / f"anp_1{cls}_final.yaml").write_text("[]", encoding="utf-8")
            (tmp_path / f"eval_1{cls}").mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        monkeypatch.setattr("sys.argv", [
            "forma-report-batch",
            "--config", str(config),
            "--join-dir", str(join_dir),
            "--join-pattern", "anp_1{class}_final.yaml",
            "--eval-pattern", "eval_1{class}",
            "--output-dir", str(output_dir),
            "--classes", "A", "B", "C",
            "--skip-llm",
            "--no-individual",
        ])

        mock_students = _make_mock_students(5)
        mock_distributions = MagicMock()
        mock_report_data = _make_mock_professor_report_data()
        mock_prof_gen = MagicMock()

        mock_load = MagicMock(return_value=(mock_students, mock_distributions))

        with patch(
            "forma.cli_report_batch.load_all_student_data",
            mock_load,
        ), patch(
            "forma.cli_report_batch.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_batch.ProfessorPDFReportGenerator",
            return_value=mock_prof_gen,
        ):
            try:
                main()
            except SystemExit as exc:
                assert exc.code in (None, 0), f"Unexpected exit code: {exc.code}"

        # Called once per class (3 classes)
        assert mock_load.call_count == 3


# ---------------------------------------------------------------------------
# T046-transcript: --transcript-pattern integration tests (FR-019b, SC-005)
# ---------------------------------------------------------------------------


class TestTranscriptPattern:
    """Tests for --transcript-pattern argument integration (FR-019b)."""

    def _setup_single_class(self, tmp_path, class_id="A"):
        """Create minimal filesystem for a single-class batch run."""
        config = tmp_path / "exam.yaml"
        config.write_text("questions: []", encoding="utf-8")
        join_dir = tmp_path / "results"
        join_dir.mkdir()
        final_file = join_dir / f"anp_1{class_id}_final.yaml"
        final_file.write_text("[]", encoding="utf-8")
        eval_dir = join_dir / f"eval_1{class_id}"
        eval_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        return config, join_dir, output_dir

    def test_no_transcript_pattern_no_emphasis_processing(
        self, tmp_path, monkeypatch
    ):
        """SC-005: Without --transcript-pattern, no emphasis/gap processing occurs."""
        config, join_dir, output_dir = self._setup_single_class(tmp_path)
        mock_report_data = _make_mock_professor_report_data()
        mock_report_data.question_stats = []

        monkeypatch.setattr("sys.argv", [
            "forma-report-batch",
            "--config", str(config),
            "--join-dir", str(join_dir),
            "--join-pattern", "anp_1{class}_final.yaml",
            "--eval-pattern", "eval_1{class}",
            "--output-dir", str(output_dir),
            "--classes", "A",
            "--no-individual",
        ])

        with patch(
            "forma.cli_report_batch.load_all_student_data",
            return_value=(_make_mock_students(5), MagicMock()),
        ), patch(
            "forma.cli_report_batch.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_batch.ProfessorPDFReportGenerator",
            return_value=MagicMock(),
        ), patch(
            "forma.emphasis_map.compute_emphasis_map",
        ) as mock_cem:
            try:
                main()
            except SystemExit:
                pass

        # compute_emphasis_map must NOT be called without --transcript-pattern
        mock_cem.assert_not_called()

    def test_transcript_pattern_nonexistent_dir_logs_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        """--transcript-pattern resolving to non-existent dir logs warning and continues."""
        config, join_dir, output_dir = self._setup_single_class(tmp_path)
        mock_report_data = _make_mock_professor_report_data()
        mock_report_data.question_stats = []

        monkeypatch.setattr("sys.argv", [
            "forma-report-batch",
            "--config", str(config),
            "--join-dir", str(join_dir),
            "--join-pattern", "anp_1{class}_final.yaml",
            "--eval-pattern", "eval_1{class}",
            "--output-dir", str(output_dir),
            "--classes", "A",
            "--no-individual",
            "--transcript-pattern", str(tmp_path / "no_such_{class}"),
        ])

        with caplog.at_level(logging.WARNING, logger="forma.cli_report_batch"), patch(
            "forma.cli_report_batch.load_all_student_data",
            return_value=(_make_mock_students(5), MagicMock()),
        ), patch(
            "forma.cli_report_batch.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_batch.ProfessorPDFReportGenerator",
            return_value=MagicMock(),
        ):
            try:
                main()
            except SystemExit:
                pass

        assert any(
            "트랜스크립트" in record.message
            for record in caplog.records
        ), f"Expected transcript warning, got: {[r.message for r in caplog.records]}"

    def test_transcript_pattern_class_substitution(
        self, tmp_path, monkeypatch
    ):
        """--transcript-pattern with {class} → correct dir per class (FR-019b)."""
        config, join_dir, output_dir = self._setup_single_class(tmp_path, "A")
        # Create transcript directory for class A
        transcript_dir_A = tmp_path / "transcripts_A"
        transcript_dir_A.mkdir()
        (transcript_dir_A / "lecture.txt").write_text(
            "심장은 혈액을 순환시키는 기관이다.", encoding="utf-8"
        )

        mock_qstat = MagicMock()
        mock_qstat.concept_mastery_rates = {"심장": 0.8}
        mock_report_data = _make_mock_professor_report_data()
        mock_report_data.question_stats = [mock_qstat]

        pattern = str(tmp_path / "transcripts_{class}")
        monkeypatch.setattr("sys.argv", [
            "forma-report-batch",
            "--config", str(config),
            "--join-dir", str(join_dir),
            "--join-pattern", "anp_1{class}_final.yaml",
            "--eval-pattern", "eval_1{class}",
            "--output-dir", str(output_dir),
            "--classes", "A",
            "--no-individual",
            "--transcript-pattern", pattern,
        ])

        mock_emphasis = MagicMock()
        mock_emphasis.concept_scores = {"심장": 0.9}
        mock_gap = MagicMock()

        with patch(
            "forma.cli_report_batch.load_all_student_data",
            return_value=(_make_mock_students(5), MagicMock()),
        ), patch(
            "forma.cli_report_batch.build_professor_report_data",
            return_value=mock_report_data,
        ), patch(
            "forma.cli_report_batch.ProfessorPDFReportGenerator",
            return_value=MagicMock(),
        ), patch(
            "forma.emphasis_map.compute_emphasis_map",
            return_value=mock_emphasis,
        ) as mock_cem, patch(
            "forma.lecture_gap_analysis.compute_lecture_gap",
            return_value=mock_gap,
        ) as mock_clg:
            try:
                main()
            except SystemExit:
                pass

        # compute_emphasis_map should have been called for class A
        mock_cem.assert_called_once()
        mock_clg.assert_called_once()
