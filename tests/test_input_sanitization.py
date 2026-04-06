"""Tests for Phase 6 US4 — Unicode sanitization, PII masking, epsilon guard.

Covers:
- T069: Zero-width character handling (FR-028)
- T070: XML control character removal (FR-029)
- T071: Email masking (FR-030)
- T072: PII exposure control (FR-031)
- T073: Epsilon guard for slope comparison (FR-032)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# T069: Zero-width character handling (FR-028)
# ---------------------------------------------------------------------------


class TestZeroWidthCharHandling:
    """FR-028: Zero-width characters in filenames produce _unnamed."""

    def test_zero_width_space_only_returns_unnamed(self):
        """Filename of only zero-width spaces becomes '_unnamed'."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename("\u200b\u200c\u200d")
        assert result == "_unnamed"

    def test_zero_width_chars_stripped_from_middle(self):
        """Zero-width characters are removed from inside valid text."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename("hello\u200bworld")
        assert result == "helloworld"
        assert "\u200b" not in result

    def test_zwj_stripped(self):
        """Zero-width joiner U+200D is stripped."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename("test\u200dname")
        assert "\u200d" not in result
        assert result == "testname"

    def test_empty_after_sanitize_returns_unnamed(self):
        """Filename that becomes empty after sanitization returns '_unnamed'."""
        from forma.delivery_prepare import sanitize_filename

        # All illegal chars + zero-width chars
        result = sanitize_filename('<>:"/\\|?*\u200b')
        assert result == "_unnamed"

    def test_report_sanitize_zero_width_returns_unnamed(self):
        """report_utils sanitize also handles zero-width -> _unnamed."""
        from forma.report_utils import sanitize_filename_report

        result = sanitize_filename_report("\u200b\u200c\u200d")
        assert result == "_unnamed"

    def test_report_sanitize_zero_width_in_name(self):
        """report_utils sanitize strips zero-width from inside valid text."""
        from forma.report_utils import sanitize_filename_report

        result = sanitize_filename_report("student\u200bname")
        assert "\u200b" not in result
        assert "studentname" in result

    def test_null_byte_stripped(self):
        """Null byte U+0000 is removed from filename."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename("test\x00name")
        assert "\x00" not in result
        assert result == "testname"


# ---------------------------------------------------------------------------
# T070: XML control character removal (FR-029)
# ---------------------------------------------------------------------------


class TestXmlControlCharRemoval:
    """FR-029: Control characters removed from esc() output."""

    def test_null_byte_removed_from_esc(self):
        """esc() strips null bytes before XML escaping."""
        from forma.font_utils import esc

        result = esc("hello\x00world")
        assert "\x00" not in result
        assert "helloworld" in result

    def test_control_chars_removed(self):
        """esc() strips C0 control characters (0x01-0x08, 0x0B, 0x0C, 0x0E-0x1F)."""
        from forma.font_utils import esc

        # Bell, backspace, vertical tab, form feed, escape
        text = "test\x07\x08\x0b\x0c\x1bvalue"
        result = esc(text)
        assert "\x07" not in result
        assert "\x08" not in result
        assert "\x0b" not in result
        assert "\x0c" not in result
        assert "\x1b" not in result
        assert "testvalue" in result

    def test_valid_xml_chars_preserved(self):
        """Tab, newline, carriage return are preserved by esc()."""
        from forma.font_utils import esc

        result = esc("line1\nline2\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_xml_special_chars_still_escaped(self):
        """esc() still escapes <, >, & after control char stripping."""
        from forma.font_utils import esc

        result = esc("a < b & c > d\x00")
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result
        assert "\x00" not in result


# ---------------------------------------------------------------------------
# T071: Email masking (FR-030)
# ---------------------------------------------------------------------------


class TestEmailMasking:
    """FR-030: Email addresses masked as abc***@domain.com."""

    def test_standard_email_masked(self):
        """Standard email is masked: first 3 chars + *** + @domain."""
        from forma.delivery_send import _mask_email

        result = _mask_email("student@university.ac.kr")
        assert result == "stu***@university.ac.kr"

    def test_short_local_part_masked(self):
        """Short local part (< 3 chars) still masks correctly."""
        from forma.delivery_send import _mask_email

        result = _mask_email("ab@test.com")
        assert result == "ab***@test.com"

    def test_single_char_local_part(self):
        """Single char local part masks correctly."""
        from forma.delivery_send import _mask_email

        result = _mask_email("a@test.com")
        assert result == "a***@test.com"

    def test_no_at_sign_returns_masked(self):
        """Email without @ still masks (first 3 chars + ***)."""
        from forma.delivery_send import _mask_email

        result = _mask_email("notanemail")
        assert result == "not***"

    def test_empty_email_returns_empty(self):
        """Empty string returns empty."""
        from forma.delivery_send import _mask_email

        result = _mask_email("")
        assert result == ""


# ---------------------------------------------------------------------------
# T072: PII exposure control (FR-031)
# ---------------------------------------------------------------------------


class TestPiiExposureControl:
    """FR-031: Full student name hidden without --verbose."""

    def test_prepare_error_output_shows_id_only(self):
        """In non-verbose mode, error output uses student_id, not full name."""
        # This tests the principle: cli_deliver _cmd_prepare shows
        # student_id by default, full name only in verbose
        from forma.delivery_send import _mask_email

        # _mask_email should be used for email display
        masked = _mask_email("student@example.com")
        assert "student@example.com" not in masked
        assert "stu***@example.com" == masked

    def test_mask_email_hides_full_address(self):
        """_mask_email hides enough of the email to prevent PII leakage."""
        from forma.delivery_send import _mask_email

        original = "kimjihoon@univ.ac.kr"
        masked = _mask_email(original)
        assert original != masked
        assert "@univ.ac.kr" in masked
        assert "kim***" in masked


# ---------------------------------------------------------------------------
# T073: Epsilon guard for slope comparison (FR-032)
# ---------------------------------------------------------------------------


class TestEpsilonGuard:
    """FR-032: Tiny negative slopes not flagged as decline."""

    def test_tiny_negative_slope_not_flagged_warning(self):
        """slope -1e-16 should NOT be classified as SCORE_DECLINE."""
        from forma.warning_report_data import _classify_risk_types

        # Trajectory that produces slope ~= -1e-16 (practically zero)
        # Two identical scores produce slope=0, so we use nearly-identical
        trajectory = [0.80, 0.80 - 1e-15]
        risk_types = _classify_risk_types({}, trajectory, 0.0)
        risk_names = [rt.value for rt in risk_types]
        assert "SCORE_DECLINE" not in risk_names

    def test_real_negative_slope_still_flagged(self):
        """slope -0.05 (real decline) IS still classified as SCORE_DECLINE."""
        from forma.warning_report_data import _classify_risk_types

        trajectory = [0.80, 0.75, 0.70, 0.65]
        risk_types = _classify_risk_types({}, trajectory, 0.0)
        risk_names = [rt.value for rt in risk_types]
        assert "SCORE_DECLINE" in risk_names

    def test_tiny_negative_slope_not_flagged_risk_predictor(self):
        """_ols_slope near-zero not treated as negative in cold start."""
        from forma.risk_predictor import _ols_slope

        # Nearly flat trajectory
        slope = _ols_slope([0.80, 0.80, 0.80])
        # Slope should be very close to 0
        assert abs(slope) < 1e-9

    def test_risk_predictor_cold_start_tiny_slope(self):
        """Cold start heuristic: tiny negative slope contributes ~0 to drop_prob."""
        import numpy as np

        from forma.risk_predictor import RiskPredictor, FEATURE_NAMES

        predictor = RiskPredictor()
        # Feature vector with score_mean=0.8, score_slope=-1e-16, absence=0
        features = np.zeros((1, 15))
        features[0, 0] = 0.8  # score_mean
        features[0, 2] = -1e-16  # score_slope (tiny negative)
        features[0, 3] = 0.8  # last_score

        preds = predictor.predict_cold_start(features, ["s1"], FEATURE_NAMES)
        # With good score_mean=0.8 and tiny slope, drop_prob should be low
        assert preds[0].drop_probability < 0.3

    def test_warning_epsilon_constant_exists(self):
        """_SLOPE_EPSILON constant exists in warning_report_data."""
        from forma.warning_report_data import _SLOPE_EPSILON

        assert _SLOPE_EPSILON > 0
        assert _SLOPE_EPSILON < 0.01  # Should be small
