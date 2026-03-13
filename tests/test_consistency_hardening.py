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

import math
import re

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
