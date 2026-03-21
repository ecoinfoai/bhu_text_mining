"""Tests for cli_report.py — forma-report CLI entry point.

T013: US1 batch mode CLI tests (argparse, file validation, batch generation).
T027: US2 --student filter tests (single student, student not found).
T011: v0.9.0 US1 backward compatibility + --no-config tests.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from forma.report_data_loader import (
    ClassDistributions,
    ConceptDetail,
    QuestionReportData,
    StudentReportData,
)


# ---------------------------------------------------------------------------
# Sample mock data
# ---------------------------------------------------------------------------

MOCK_STUDENTS = [
    StudentReportData(
        student_id="S015",
        real_name="이유정",
        student_number="2026194126",
        class_name="A반",
        course_name="인체구조와기능",
        week_num=1,
        questions=[QuestionReportData(question_sn=1, ensemble_score=0.5)],
    ),
    StudentReportData(
        student_id="S039",
        real_name="박수영",
        student_number="2026194063",
        class_name="B반",
        course_name="인체구조와기능",
        week_num=1,
        questions=[QuestionReportData(question_sn=1, ensemble_score=0.8)],
    ),
]
MOCK_DISTS = ClassDistributions()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cli_env(tmp_path):
    """Create minimal filesystem structure required by the CLI.

    Returns a dict with paths to the fake --final, --config, --eval-dir,
    and --output-dir arguments.
    """
    # --final: a YAML file that must exist
    final_file = tmp_path / "anp_final.yaml"
    final_file.write_text("[]", encoding="utf-8")

    # --config: a YAML file that must exist
    config_file = tmp_path / "exam_config.yaml"
    config_file.write_text("questions: []", encoding="utf-8")

    # --eval-dir: directory with res_lvl4/ensemble_results.yaml
    eval_dir = tmp_path / "eval"
    res_lvl4 = eval_dir / "res_lvl4"
    res_lvl4.mkdir(parents=True)
    (res_lvl4 / "ensemble_results.yaml").write_text(
        "students: []", encoding="utf-8",
    )

    # --output-dir
    output_dir = tmp_path / "reports"
    output_dir.mkdir()

    return {
        "final": str(final_file),
        "config": str(config_file),
        "eval_dir": str(eval_dir),
        "output_dir": str(output_dir),
    }


def _base_argv(cli_env: dict) -> list[str]:
    """Build a complete argv list from cli_env fixture."""
    return [
        "--final", cli_env["final"],
        "--config", cli_env["config"],
        "--eval-dir", cli_env["eval_dir"],
        "--output-dir", cli_env["output_dir"],
    ]


# ---------------------------------------------------------------------------
# T013: US1 — CLI batch mode tests
# ---------------------------------------------------------------------------


class TestT013BatchMode:
    """T013 [US1] Batch mode CLI tests for required args and batch generation."""

    # -- Missing required arguments ------------------------------------------

    def test_main_requires_final_arg(self, cli_env, monkeypatch):
        """Missing --final exits with code 1 (argparse SystemExit)."""
        monkeypatch.setattr("sys.argv", [
            "forma-report",
            "--config", cli_env["config"],
            "--eval-dir", cli_env["eval_dir"],
            "--output-dir", cli_env["output_dir"],
        ])
        from forma.cli_report import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        # argparse exits with code 2 for missing required args;
        # the CLI may also map this to code 1 — accept either.
        assert exc_info.value.code in (1, 2)

    def test_main_requires_config_arg(self, cli_env, monkeypatch):
        """Missing --config exits with code 1."""
        monkeypatch.setattr("sys.argv", [
            "forma-report",
            "--final", cli_env["final"],
            "--eval-dir", cli_env["eval_dir"],
            "--output-dir", cli_env["output_dir"],
        ])
        from forma.cli_report import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code in (1, 2)

    def test_main_requires_eval_dir_arg(self, cli_env, monkeypatch):
        """Missing --eval-dir exits with code 1."""
        monkeypatch.setattr("sys.argv", [
            "forma-report",
            "--final", cli_env["final"],
            "--config", cli_env["config"],
            "--output-dir", cli_env["output_dir"],
        ])
        from forma.cli_report import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code in (1, 2)

    def test_main_requires_output_dir_arg(self, cli_env, monkeypatch):
        """Missing --output-dir exits with code 1."""
        monkeypatch.setattr("sys.argv", [
            "forma-report",
            "--final", cli_env["final"],
            "--config", cli_env["config"],
            "--eval-dir", cli_env["eval_dir"],
        ])
        from forma.cli_report import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code in (1, 2)

    # -- File / directory not found ------------------------------------------

    def test_main_file_not_found_exits_1(self, cli_env, monkeypatch):
        """--final pointing to nonexistent file exits with code 1."""
        monkeypatch.setattr("sys.argv", [
            "forma-report",
            "--final", "/nonexistent/path/to/anp_final.yaml",
            "--config", cli_env["config"],
            "--eval-dir", cli_env["eval_dir"],
            "--output-dir", cli_env["output_dir"],
        ])
        from forma.cli_report import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_main_eval_dir_not_found_exits_1(self, cli_env, monkeypatch):
        """--eval-dir pointing to nonexistent directory exits with code 1."""
        monkeypatch.setattr("sys.argv", [
            "forma-report",
            "--final", cli_env["final"],
            "--config", cli_env["config"],
            "--eval-dir", "/nonexistent/eval_dir",
            "--output-dir", cli_env["output_dir"],
        ])
        from forma.cli_report import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    # -- Batch generation ----------------------------------------------------

    def test_main_batch_generates_all_pdfs(self, cli_env, monkeypatch):
        """With 2 students, generate_pdf is called twice. Exit code 0."""
        monkeypatch.setattr("sys.argv", [
            "forma-report", *_base_argv(cli_env),
        ])

        mock_generator_instance = MagicMock()
        mock_generator_cls = MagicMock(return_value=mock_generator_instance)

        with patch(
            "forma.cli_report.load_all_student_data",
            return_value=(MOCK_STUDENTS, MOCK_DISTS),
        ), patch(
            "forma.cli_report.StudentPDFReportGenerator",
            mock_generator_cls,
        ):
            from forma.cli_report import main

            main()

        # generate_pdf called once per student
        assert mock_generator_instance.generate_pdf.call_count == 2

        # Verify each student was passed
        call_args_list = mock_generator_instance.generate_pdf.call_args_list
        student_ids_called = [
            call.args[0].student_id
            if call.args
            else call.kwargs.get("student_data", call.kwargs.get("student")).student_id
            for call in call_args_list
        ]
        assert "S015" in student_ids_called
        assert "S039" in student_ids_called

    def test_main_progress_output(self, cli_env, monkeypatch, capsys):
        """Verify stdout contains student count message."""
        monkeypatch.setattr("sys.argv", [
            "forma-report", *_base_argv(cli_env),
        ])

        mock_generator_instance = MagicMock()
        mock_generator_cls = MagicMock(return_value=mock_generator_instance)

        with patch(
            "forma.cli_report.load_all_student_data",
            return_value=(MOCK_STUDENTS, MOCK_DISTS),
        ), patch(
            "forma.cli_report.StudentPDFReportGenerator",
            mock_generator_cls,
        ):
            from forma.cli_report import main

            main()

        captured = capsys.readouterr()
        # Should mention student count (2 students)
        assert "2" in captured.out


# ---------------------------------------------------------------------------
# T027: US2 — --student filter tests
# ---------------------------------------------------------------------------


class TestT027StudentFilter:
    """T027 [US2] --student filter tests for single student and not found."""

    def test_student_filter_single(self, cli_env, monkeypatch):
        """--student S015 with 2 students data: generate_pdf called once for S015."""
        monkeypatch.setattr("sys.argv", [
            "forma-report",
            *_base_argv(cli_env),
            "--student", "S015",
        ])

        mock_generator_instance = MagicMock()
        mock_generator_cls = MagicMock(return_value=mock_generator_instance)

        with patch(
            "forma.cli_report.load_all_student_data",
            return_value=(MOCK_STUDENTS, MOCK_DISTS),
        ), patch(
            "forma.cli_report.StudentPDFReportGenerator",
            mock_generator_cls,
        ):
            from forma.cli_report import main

            main()

        # generate_pdf called exactly once
        assert mock_generator_instance.generate_pdf.call_count == 1

        # Verify the call was for S015
        call = mock_generator_instance.generate_pdf.call_args
        student_arg = (
            call.args[0] if call.args else
            call.kwargs.get("student_data", call.kwargs.get("student"))
        )
        assert student_arg.student_id == "S015"

    def test_student_not_found_exits_2(self, cli_env, monkeypatch, capsys):
        """--student S999 when not in data: exit code 2, error message in Korean."""
        monkeypatch.setattr("sys.argv", [
            "forma-report",
            *_base_argv(cli_env),
            "--student", "S999",
        ])

        mock_generator_instance = MagicMock()
        mock_generator_cls = MagicMock(return_value=mock_generator_instance)

        with patch(
            "forma.cli_report.load_all_student_data",
            return_value=(MOCK_STUDENTS, MOCK_DISTS),
        ), patch(
            "forma.cli_report.StudentPDFReportGenerator",
            mock_generator_cls,
        ):
            from forma.cli_report import main

            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 2

        # Verify error message mentions the student ID
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "S999" in combined
        assert "No data found" in combined


# ---------------------------------------------------------------------------
# T011: v0.9.0 US1 — Backward compatibility & --no-config tests
# ---------------------------------------------------------------------------


class TestT011BackwardCompat:
    """T011 [US1] Verify forma-report works identically with/without forma.yaml,
    and that --no-config ignores config file."""

    def test_no_config_flag_accepted(self, cli_env, monkeypatch):
        """--no-config flag is accepted without error."""
        monkeypatch.setattr("sys.argv", [
            "forma-report",
            *_base_argv(cli_env),
            "--no-config",
        ])

        mock_generator_instance = MagicMock()
        mock_generator_cls = MagicMock(return_value=mock_generator_instance)

        with patch(
            "forma.cli_report.load_all_student_data",
            return_value=(MOCK_STUDENTS, MOCK_DISTS),
        ), patch(
            "forma.cli_report.StudentPDFReportGenerator",
            mock_generator_cls,
        ):
            from forma.cli_report import main
            main()

        assert mock_generator_instance.generate_pdf.call_count == 2

    def test_works_without_config_file(self, cli_env, monkeypatch):
        """Works identically when no forma.yaml exists (backward compat)."""
        monkeypatch.setattr("sys.argv", [
            "forma-report", *_base_argv(cli_env),
        ])

        mock_generator_instance = MagicMock()
        mock_generator_cls = MagicMock(return_value=mock_generator_instance)

        with patch(
            "forma.cli_report.load_all_student_data",
            return_value=(MOCK_STUDENTS, MOCK_DISTS),
        ), patch(
            "forma.cli_report.StudentPDFReportGenerator",
            mock_generator_cls,
        ):
            from forma.cli_report import main
            main()

        assert mock_generator_instance.generate_pdf.call_count == 2

    def test_dpi_flag_still_works(self, cli_env, monkeypatch):
        """Explicit --dpi flag is respected."""
        monkeypatch.setattr("sys.argv", [
            "forma-report",
            *_base_argv(cli_env),
            "--dpi", "300",
        ])

        mock_generator_instance = MagicMock()
        mock_generator_cls = MagicMock(return_value=mock_generator_instance)

        with patch(
            "forma.cli_report.load_all_student_data",
            return_value=(MOCK_STUDENTS, MOCK_DISTS),
        ), patch(
            "forma.cli_report.StudentPDFReportGenerator",
            mock_generator_cls,
        ):
            from forma.cli_report import main
            main()

        # Verify DPI was passed to generator constructor
        call_kwargs = mock_generator_cls.call_args
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("dpi") == 300
        else:
            # Positional: font_path, dpi
            assert 300 in call_kwargs.args


# ---------------------------------------------------------------------------
# T047: US4 — --concept-deps flag wiring tests (FR-020, FR-023)
# ---------------------------------------------------------------------------


# Students with concept details for concept-deps tests
MOCK_STUDENTS_WITH_CONCEPTS = [
    StudentReportData(
        student_id="S015",
        real_name="이유정",
        student_number="2026194126",
        class_name="A반",
        course_name="인체구조와기능",
        week_num=1,
        questions=[QuestionReportData(
            question_sn=1,
            ensemble_score=0.5,
            concepts=[
                ConceptDetail(concept="세포막 구조", is_present=True, similarity=0.8, threshold=0.5),
                ConceptDetail(concept="물질 이동", is_present=False, similarity=0.2, threshold=0.5),
            ],
        )],
    ),
    StudentReportData(
        student_id="S039",
        real_name="박수영",
        student_number="2026194063",
        class_name="B반",
        course_name="인체구조와기능",
        week_num=1,
        questions=[QuestionReportData(
            question_sn=1,
            ensemble_score=0.8,
            concepts=[
                ConceptDetail(concept="세포막 구조", is_present=True, similarity=0.9, threshold=0.5),
                ConceptDetail(concept="물질 이동", is_present=True, similarity=0.7, threshold=0.5),
            ],
        )],
    ),
]


class TestT047ConceptDeps:
    """T047 [US4] --concept-deps flag wiring in forma-report CLI."""

    def test_concept_deps_flag_accepted(self, cli_env, monkeypatch):
        """--concept-deps flag is accepted by the parser (FR-023)."""
        from forma.cli_report import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--final", cli_env["final"],
            "--config", cli_env["config"],
            "--eval-dir", cli_env["eval_dir"],
            "--output-dir", cli_env["output_dir"],
            "--concept-deps",
        ])
        assert args.concept_deps is True

    def test_concept_deps_default_false(self, cli_env):
        """--concept-deps defaults to False (FR-023)."""
        from forma.cli_report import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--final", cli_env["final"],
            "--config", cli_env["config"],
            "--eval-dir", cli_env["eval_dir"],
            "--output-dir", cli_env["output_dir"],
        ])
        assert args.concept_deps is False

    def test_concept_deps_passes_learning_path_to_generate_pdf(
        self, cli_env, monkeypatch, tmp_path,
    ):
        """With --concept-deps and valid YAML, generate_pdf receives learning_path kwarg."""
        import yaml

        # Write exam config with concept_dependencies
        config_file = tmp_path / "exam_config_deps.yaml"
        config_data = {
            "questions": [],
            "concept_dependencies": [
                {"prerequisite": "세포막 구조", "dependent": "물질 이동"},
            ],
        }
        config_file.write_text(yaml.dump(config_data, allow_unicode=True), encoding="utf-8")

        monkeypatch.setattr("sys.argv", [
            "forma-report",
            "--final", cli_env["final"],
            "--config", str(config_file),
            "--eval-dir", cli_env["eval_dir"],
            "--output-dir", cli_env["output_dir"],
            "--concept-deps",
        ])

        mock_generator_instance = MagicMock()
        mock_generator_cls = MagicMock(return_value=mock_generator_instance)

        with patch(
            "forma.cli_report.load_all_student_data",
            return_value=(MOCK_STUDENTS_WITH_CONCEPTS, MOCK_DISTS),
        ), patch(
            "forma.cli_report.StudentPDFReportGenerator",
            mock_generator_cls,
        ):
            from forma.cli_report import main
            main()

        # Verify generate_pdf was called with learning_path kwarg
        assert mock_generator_instance.generate_pdf.call_count == 2
        for call in mock_generator_instance.generate_pdf.call_args_list:
            assert "learning_path" in call.kwargs
            assert call.kwargs["learning_path"] is not None

    def test_without_concept_deps_no_learning_path(self, cli_env, monkeypatch):
        """Without --concept-deps, learning_path is None in generate_pdf call."""
        monkeypatch.setattr("sys.argv", [
            "forma-report", *_base_argv(cli_env),
        ])

        mock_generator_instance = MagicMock()
        mock_generator_cls = MagicMock(return_value=mock_generator_instance)

        with patch(
            "forma.cli_report.load_all_student_data",
            return_value=(MOCK_STUDENTS, MOCK_DISTS),
        ), patch(
            "forma.cli_report.StudentPDFReportGenerator",
            mock_generator_cls,
        ):
            from forma.cli_report import main
            main()

        # learning_path should be None when --concept-deps not provided
        for call in mock_generator_instance.generate_pdf.call_args_list:
            assert call.kwargs.get("learning_path") is None

    def test_concept_deps_no_dependencies_in_yaml_omits_silently(
        self, cli_env, monkeypatch, tmp_path,
    ):
        """--concept-deps with YAML lacking concept_dependencies → learning_path is None (FR-023)."""
        import yaml

        config_file = tmp_path / "exam_no_deps.yaml"
        config_file.write_text(yaml.dump({"questions": []}, allow_unicode=True), encoding="utf-8")

        monkeypatch.setattr("sys.argv", [
            "forma-report",
            "--final", cli_env["final"],
            "--config", str(config_file),
            "--eval-dir", cli_env["eval_dir"],
            "--output-dir", cli_env["output_dir"],
            "--concept-deps",
        ])

        mock_generator_instance = MagicMock()
        mock_generator_cls = MagicMock(return_value=mock_generator_instance)

        with patch(
            "forma.cli_report.load_all_student_data",
            return_value=(MOCK_STUDENTS, MOCK_DISTS),
        ), patch(
            "forma.cli_report.StudentPDFReportGenerator",
            mock_generator_cls,
        ):
            from forma.cli_report import main
            main()

        for call in mock_generator_instance.generate_pdf.call_args_list:
            assert call.kwargs.get("learning_path") is None
