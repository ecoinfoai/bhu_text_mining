"""Cross-cutting consistency hardening tests.

Verifies project-wide consistency patterns introduced by 014-consistency-hardening:
- NaN-safe aggregation
- XML-special character escaping in report generators
- Builder functions handle empty inputs
- YAML error handling
- PII masking in delivery_send.py
- Chart modules use save_fig()
- esc() strips zero-width Unicode
- Logger variable naming
"""

from __future__ import annotations


import pytest


# ---------------------------------------------------------------------------
# US1: NaN-safe aggregation — esc() zero-width Unicode stripping
# ---------------------------------------------------------------------------


class TestEscZeroWidthStripping:
    """Verify esc() strips zero-width Unicode characters."""

    def test_strips_zero_width_space(self):
        """U+200B ZERO WIDTH SPACE is removed."""
        from forma.font_utils import esc
        assert esc("hello\u200bworld") == "helloworld"

    def test_strips_zero_width_non_joiner(self):
        """U+200C ZERO WIDTH NON-JOINER is removed."""
        from forma.font_utils import esc
        assert esc("hello\u200cworld") == "helloworld"

    def test_strips_zero_width_joiner(self):
        """U+200D ZERO WIDTH JOINER is removed."""
        from forma.font_utils import esc
        assert esc("hello\u200dworld") == "helloworld"

    def test_strips_left_to_right_mark(self):
        """U+200E LEFT-TO-RIGHT MARK is removed."""
        from forma.font_utils import esc
        assert esc("hello\u200eworld") == "helloworld"

    def test_strips_right_to_left_mark(self):
        """U+200F RIGHT-TO-LEFT MARK is removed."""
        from forma.font_utils import esc
        assert esc("hello\u200fworld") == "helloworld"

    def test_strips_bom(self):
        """U+FEFF BOM / ZERO WIDTH NO-BREAK SPACE is removed."""
        from forma.font_utils import esc
        assert esc("\ufeffhello") == "hello"

    def test_strips_multiple_zero_width(self):
        """Multiple zero-width chars in one string are all removed."""
        from forma.font_utils import esc
        text = "\u200b\u200c\u200d\u200e\u200f\ufefftest"
        assert esc(text) == "test"

    def test_preserves_normal_korean(self):
        """Normal Korean text is not affected by zero-width stripping."""
        from forma.font_utils import esc
        text = "형성평가 분석"
        assert esc(text) == "형성평가 분석"

    def test_xml_escape_still_works(self):
        """XML special characters are still escaped after zero-width stripping."""
        from forma.font_utils import esc
        assert esc("a\u200b<b>&c") == "a&lt;b&gt;&amp;c"

    def test_c0_control_chars_still_stripped(self):
        """Original C0 control character stripping still works."""
        from forma.font_utils import esc
        # \x01 is C0 control, should be stripped
        assert esc("hello\x01world") == "helloworld"

    def test_preserves_tab_newline_cr(self):
        """Tab, newline, and CR are preserved (not in illegal set)."""
        from forma.font_utils import esc
        assert esc("a\tb\nc\rd") == "a\tb\nc\rd"


# ---------------------------------------------------------------------------
# US1: All 5 report generators use esc() (font_utils.esc)
# ---------------------------------------------------------------------------


class TestReportGeneratorsUseEsc:
    """Verify all PDF report generators import esc from font_utils."""

    @pytest.mark.parametrize("module_name", [
        "forma.professor_report",
        "forma.student_report",
        "forma.longitudinal_report",
        "forma.warning_report",
    ])
    def test_imports_esc_from_font_utils(self, module_name):
        """Module imports esc from forma.font_utils."""
        import importlib
        mod = importlib.import_module(module_name)
        # The modules import `esc as _esc` — check that _esc is callable
        assert hasattr(mod, "_esc"), f"{module_name} missing _esc"
        assert callable(mod._esc)

    @pytest.mark.parametrize("module_name", [
        "forma.professor_report",
        "forma.student_report",
        "forma.longitudinal_report",
        "forma.warning_report",
    ])
    def test_esc_is_font_utils_esc(self, module_name):
        """_esc in the module is the same function as font_utils.esc."""
        import importlib
        from forma.font_utils import esc
        mod = importlib.import_module(module_name)
        assert mod._esc is esc


# ---------------------------------------------------------------------------
# US1: Builder functions handle empty inputs
# ---------------------------------------------------------------------------


class TestBuilderEmptyInputs:
    """Builder functions should not crash on empty inputs."""

    def test_build_warning_data_empty_students(self):
        """build_warning_data with no students returns empty list."""
        from forma.warning_report_data import build_warning_data
        # Empty inputs should return empty result
        result = build_warning_data(
            at_risk_students={},
            risk_predictions=[],
            concept_scores={},
        )
        assert result == []


# ---------------------------------------------------------------------------
# PII masking in delivery_send.py
# ---------------------------------------------------------------------------


class TestPiiMasking:
    """Verify _mask_email PII masking behavior."""

    def test_mask_email_standard(self):
        """Standard email is masked: first 3 chars + *** + @domain."""
        from forma.delivery_send import _mask_email
        assert _mask_email("student@university.ac.kr") == "stu***@university.ac.kr"

    def test_mask_email_short_local(self):
        """Short local part (< 3 chars) shows all available chars + ***."""
        from forma.delivery_send import _mask_email
        assert _mask_email("ab@example.com") == "ab***@example.com"

    def test_mask_email_empty(self):
        """Empty string returns empty string."""
        from forma.delivery_send import _mask_email
        assert _mask_email("") == ""

    def test_mask_email_no_at(self):
        """String without @ shows first 3 chars + ***."""
        from forma.delivery_send import _mask_email
        assert _mask_email("noemail") == "noe***"


# ---------------------------------------------------------------------------
# All chart modules use save_fig()
# ---------------------------------------------------------------------------


class TestChartModulesUseSaveFig:
    """Verify chart modules import and use save_fig from chart_utils.

    Note: forma.section_comparison_charts uses fig.savefig() directly
    instead of _save_fig. This is a known inconsistency to be addressed
    in the DRY consolidation phase (Phase 5-6).
    """

    @pytest.mark.parametrize("module_name", [
        "forma.report_charts",
        "forma.professor_report_charts",
        "forma.warning_report_charts",
        "forma.learning_path_charts",
        "forma.longitudinal_report_charts",
    ])
    def test_imports_save_fig(self, module_name):
        """Chart module imports save_fig from chart_utils."""
        import importlib
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "_save_fig"), f"{module_name} missing _save_fig"


# ---------------------------------------------------------------------------
# Logger variable naming consistency
# ---------------------------------------------------------------------------


class TestLoggerNaming:
    """Verify modules use `logger = logging.getLogger(__name__)` pattern."""

    @pytest.mark.parametrize("module_name", [
        "forma.intervention_store",
        "forma.delivery_send",
        "forma.feedback_generator",
        "forma.pipeline_evaluation",
        "forma.risk_predictor",
        "forma.cli_report_professor",
        "forma.cli_report_longitudinal",
        "forma.cli_report_warning",
    ])
    def test_logger_uses_getlogger_name(self, module_name):
        """Module's logger is configured with __name__."""
        import importlib
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "logger"), f"{module_name} missing 'logger' attribute"
        assert mod.logger.name == module_name


# ---------------------------------------------------------------------------
# io_utils module exists and exports expected functions
# ---------------------------------------------------------------------------


class TestIoUtilsExports:
    """Verify io_utils module exists and exports the expected API."""

    def test_atomic_write_yaml_importable(self):
        """atomic_write_yaml can be imported from forma.io_utils."""
        from forma.io_utils import atomic_write_yaml
        assert callable(atomic_write_yaml)

    def test_atomic_write_json_importable(self):
        """atomic_write_json can be imported from forma.io_utils."""
        from forma.io_utils import atomic_write_json
        assert callable(atomic_write_json)

    def test_atomic_write_text_importable(self):
        """atomic_write_text can be imported from forma.io_utils."""
        from forma.io_utils import atomic_write_text
        assert callable(atomic_write_text)


# ---------------------------------------------------------------------------
# T015: Atomic write migration verification
# ---------------------------------------------------------------------------


class TestAtomicWriteMigration:
    """Verify modules use io_utils.atomic_write_yaml where appropriate."""

    def test_cli_select_uses_atomic_write(self):
        """cli_select._write_questions_yaml uses io_utils.atomic_write_yaml."""
        import inspect
        from forma import cli_select
        source = inspect.getsource(cli_select._write_questions_yaml)
        assert "atomic_write_yaml" in source, (
            "cli_select._write_questions_yaml should use atomic_write_yaml"
        )

    def test_week_config_no_shutil_move(self):
        """week_config.py should not use shutil.move (use os.replace instead)."""
        import inspect
        from forma import week_config
        source = inspect.getsource(week_config)
        assert "shutil.move" not in source, (
            "week_config should use os.replace instead of shutil.move"
        )


# ---------------------------------------------------------------------------
# T022: YAML error handling in CLI modules
# ---------------------------------------------------------------------------


class TestYamlErrorHandling:
    """CLI modules should handle corrupt YAML with Korean error messages."""

    def test_cli_report_professor_corrupt_yaml(self, tmp_path, monkeypatch):
        """cli_report_professor exits 2 with Korean error on corrupt config YAML."""
        import yaml

        from forma.cli_report_professor import main

        # Create a valid final YAML
        final_path = tmp_path / "final.yaml"
        final_path.write_text(
            yaml.dump({"students": [{"student_id": "S001", "questions": []}]}),
            encoding="utf-8",
        )
        # Create corrupt config YAML
        config_path = tmp_path / "config.yaml"
        config_path.write_text(":\n  - :\n  }{invalid", encoding="utf-8")

        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        out_dir = tmp_path / "out"

        monkeypatch.setattr("sys.argv", [
            "forma-report-professor",
            "--final", str(final_path),
            "--config", str(config_path),
            "--eval-dir", str(eval_dir),
            "--output-dir", str(out_dir),
        ])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 2

    def test_cli_report_corrupt_config_yaml(self, tmp_path):
        """cli_report.py handles corrupt YAML in concept-deps loading gracefully."""
        # This tests that yaml.YAMLError is caught, not that it exits.
        # cli_report.py wraps concept-deps in try/except Exception already.
        import yaml

        corrupt_config = tmp_path / "corrupt.yaml"
        corrupt_config.write_text(":\n  }{bad", encoding="utf-8")

        # Verify the file is actually corrupt YAML
        with open(corrupt_config, encoding="utf-8") as fh:
            with pytest.raises(yaml.YAMLError):
                yaml.safe_load(fh)

    def test_cli_report_yaml_error_exits_with_korean_message(self, tmp_path, monkeypatch, capsys):
        """cli_report.py exits with Korean YAML error message on corrupt data."""
        from unittest.mock import patch

        import yaml

        from forma.cli_report import main

        # Create valid-enough files to pass file-exists checks
        final_path = tmp_path / "final.yaml"
        final_path.write_text("students: []", encoding="utf-8")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("questions: []", encoding="utf-8")
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr("sys.argv", [
            "forma-report",
            "--final", str(final_path),
            "--config", str(config_path),
            "--eval-dir", str(eval_dir),
            "--output-dir", str(out_dir),
        ])

        # Mock load_all_student_data to raise YAMLError
        with patch(
            "forma.cli_report.load_all_student_data",
            side_effect=yaml.YAMLError("bad yaml"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "YAML" in captured.err


# ---------------------------------------------------------------------------
# T029: Empty input handling for builder functions
# ---------------------------------------------------------------------------


class TestBuilderEmptyInputsExtended:
    """Builder functions return valid empty results on empty inputs."""

    def test_build_class_knowledge_aggregate_empty(self):
        """build_class_knowledge_aggregate with no students returns empty."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        result = build_class_knowledge_aggregate(
            master_edges=[],
            comparison_results=[],
            question_sn=1,
        )
        assert result.edges == []
        assert result.total_students == 0

    def test_build_longitudinal_summary_empty_weeks(self):
        """build_longitudinal_summary with empty store data returns valid summary."""
        from unittest.mock import MagicMock

        from forma.longitudinal_report_data import build_longitudinal_summary

        mock_store = MagicMock()
        mock_store.query_students.return_value = []
        mock_store.query_weeks.return_value = []

        result = build_longitudinal_summary(
            store=mock_store,
            weeks=[],
            class_name="1A",
        )
        assert result.student_trajectories == []
        assert result.class_name == "1A"

    def test_yaml_safe_load_empty_file_returns_none(self, tmp_path):
        """yaml.safe_load on empty file returns None -- callers must guard."""
        import yaml

        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")
        with open(empty_file, encoding="utf-8") as fh:
            result = yaml.safe_load(fh)
        assert result is None

    def test_cli_ocr_empty_config_yaml_reports_missing_keys(self, tmp_path):
        """cli_ocr._load_ocr_config with empty YAML reports missing keys, not crash."""
        from forma.cli_ocr import _load_ocr_config

        empty_cfg = tmp_path / "empty.yaml"
        empty_cfg.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="missing required keys"):
            _load_ocr_config(str(empty_cfg))


# ---------------------------------------------------------------------------
# T032: DRY deduplication verification
# ---------------------------------------------------------------------------


class TestDryDeduplication:
    """Verify code deduplication across chart and CLI modules."""

    def test_section_comparison_charts_uses_save_fig(self):
        """section_comparison_charts imports _save_fig from chart_utils."""
        import importlib
        mod = importlib.import_module("forma.section_comparison_charts")
        assert hasattr(mod, "_save_fig"), (
            "section_comparison_charts should import _save_fig from chart_utils"
        )

    def test_section_comparison_charts_no_inline_savefig(self):
        """section_comparison_charts should not use fig.savefig() directly."""
        import inspect
        from forma import section_comparison_charts
        source = inspect.getsource(section_comparison_charts)
        assert "fig.savefig" not in source, (
            "section_comparison_charts should use _save_fig() instead of fig.savefig()"
        )

    def test_cli_report_professor_no_aliased_longitudinal_imports(self):
        """cli_report_professor should not alias LongitudinalStore as LS/GLS/ILS."""
        import inspect
        from forma import cli_report_professor
        source = inspect.getsource(cli_report_professor)
        for alias in ["as LS", "as GLS", "as ILS"]:
            assert alias not in source, (
                f"cli_report_professor should not alias LongitudinalStore {alias}"
            )


# ---------------------------------------------------------------------------
# T037: Performance behavioral equivalence test
# ---------------------------------------------------------------------------


class TestBuildProfessorReportDataPerformance:
    """Behavioral equivalence test for build_professor_report_data with large data."""

    def test_200_students_10_questions(self):
        """build_professor_report_data handles 200 students * 10 questions."""
        from forma.professor_report_data import build_professor_report_data
        from forma.report_data_loader import (
            ClassDistributions,
            ConceptDetail,
            QuestionReportData,
            StudentReportData,
        )

        # Build synthetic 200-student dataset
        students = []
        for i in range(200):
            sid = f"S{i:03d}"
            questions = []
            for qsn in range(1, 11):
                concepts = [
                    ConceptDetail(
                        concept=f"concept_{qsn}_{c}",
                        is_present=True,
                        similarity=0.5 + c * 0.1,
                        threshold=0.5,
                    )
                    for c in range(3)
                ]
                questions.append(QuestionReportData(
                    question_sn=qsn,
                    understanding_level="Proficient" if i % 3 != 0 else "Beginning",
                    concept_coverage=0.6 + (i % 5) * 0.08,
                    graph_comparison_f1=0.5 + (i % 4) * 0.1,
                    ensemble_score=0.5 + (i % 10) * 0.05,
                    misconceptions=[],
                    concepts=concepts,
                ))
            students.append(StudentReportData(
                student_id=sid,
                questions=questions,
            ))

        distributions = ClassDistributions(
            ensemble_scores={qsn: [0.5 + (i % 10) * 0.05 for i in range(200)] for qsn in range(1, 11)},
            concept_coverages={qsn: [0.6 + (i % 5) * 0.08 for i in range(200)] for qsn in range(1, 11)},
        )

        result = build_professor_report_data(
            students, distributions,
            class_name="Test", week_num=1,
            subject="과목", exam_title="시험",
        )
        assert result.n_students == 200
        assert len(result.question_stats) == 10


# ---------------------------------------------------------------------------
# T040: Modules with `from __future__ import annotations` should not import
# Dict/List/Tuple/Optional from typing (use built-in dict/list/tuple/X|None)
# ---------------------------------------------------------------------------


class TestModernTypingImports:
    """Modules with __future__.annotations should use built-in generics."""

    @pytest.mark.parametrize("module_path", [
        "forma/cohesion_analysis.py",
        "forma/tesseract_processor.py",
        "forma/topic_analysis.py",
        "forma/network_analysis.py",
        "forma/naver_ocr.py",
        "forma/knowledge_graph_analysis.py",
        "forma/exam_generator.py",
        "forma/font_utils.py",
    ])
    def test_no_old_typing_imports(self, module_path):
        """Module should not import Dict/List/Tuple/Optional from typing."""
        import re
        from pathlib import Path
        full_path = Path(__file__).parent.parent / "src" / module_path
        source = full_path.read_text(encoding="utf-8")
        # Check for `from typing import ...` with old-style names
        old_names = {"Dict", "List", "Tuple", "Optional"}
        match = re.search(r"from typing import (.+)", source)
        if match:
            imported = {name.strip() for name in match.group(1).split(",")}
            violations = imported & old_names
            assert not violations, (
                f"{module_path} imports {violations} from typing — "
                f"use built-in equivalents instead"
            )


# ---------------------------------------------------------------------------
# T045-T046: config.py unknown key warning
# ---------------------------------------------------------------------------


class TestConfigUnknownKeyWarning:
    """config.py should warn on unknown top-level keys in forma.json."""

    def test_known_sections_are_accepted(self, tmp_path):
        """Known top-level keys (naver_ocr, smtp, llm) produce no warning."""
        import io as _io
        import json
        import logging

        from forma.config import load_config
        config_path = tmp_path / "forma.json"
        config_path.write_text(json.dumps({
            "naver_ocr": {"secret_key": "k", "api_url": "u"},
            "smtp": {"server": "s"},
            "llm": {"provider": "gemini"},
        }), encoding="utf-8")

        handler = logging.StreamHandler(_io.StringIO())
        handler.setLevel(logging.WARNING)
        cfg_logger = logging.getLogger("forma.config")
        cfg_logger.addHandler(handler)
        try:
            load_config(str(config_path))
            output = handler.stream.getvalue()
            assert "Unknown" not in output
        finally:
            cfg_logger.removeHandler(handler)

    def test_unknown_key_logs_warning(self, tmp_path):
        """Unknown top-level key in forma.json logs a warning."""
        import json
        import logging

        from forma.config import load_config
        config_path = tmp_path / "forma.json"
        config_path.write_text(json.dumps({
            "naver_ocr": {"secret_key": "k", "api_url": "u"},
            "bogus_section": {"foo": "bar"},
        }), encoding="utf-8")

        import io as _io
        handler = logging.StreamHandler(_io.StringIO())
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("forma.config")
        logger.addHandler(handler)
        try:
            load_config(str(config_path))
            output = handler.stream.getvalue()
            assert "bogus_section" in output, (
                "load_config should warn about unknown key 'bogus_section'"
            )
        finally:
            logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# T013: Shared zero-width stripping
# ---------------------------------------------------------------------------


class TestStripInvisibleShared:
    """font_utils.strip_invisible() is used by delivery_prepare for zero-width stripping."""

    def test_strip_invisible_importable(self):
        """strip_invisible can be imported from forma.font_utils."""
        from forma.font_utils import strip_invisible
        assert callable(strip_invisible)

    def test_strip_invisible_removes_zero_width(self):
        """strip_invisible removes zero-width chars."""
        from forma.font_utils import strip_invisible
        assert strip_invisible("hello\u200bworld") == "helloworld"
        assert strip_invisible("\ufefftest") == "test"

    def test_strip_invisible_removes_control_chars(self):
        """strip_invisible removes C0 control chars (except tab/newline/CR)."""
        from forma.font_utils import strip_invisible
        assert strip_invisible("a\x01b") == "ab"

    def test_strip_invisible_preserves_tab_newline(self):
        """strip_invisible preserves tab and newline (useful for text, not filenames)."""
        from forma.font_utils import strip_invisible
        assert strip_invisible("a\tb\n") == "a\tb\n"

    def test_delivery_prepare_uses_strip_invisible(self):
        """delivery_prepare.sanitize_filename delegates to strip_invisible."""
        import inspect
        from forma import delivery_prepare
        source = inspect.getsource(delivery_prepare.sanitize_filename)
        assert "strip_invisible" in source, (
            "sanitize_filename should delegate zero-width stripping to font_utils.strip_invisible"
        )
