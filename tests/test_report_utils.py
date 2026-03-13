"""Tests for shared report utilities (report_utils.py).

Covers:
- minimal_png_bytes() returns valid PNG header + 1x1 pixel
- sanitize_filename_report() cleans filenames
"""

from __future__ import annotations

from forma.report_utils import minimal_png_bytes, sanitize_filename_report


class TestMinimalPngBytes:
    """Tests for minimal_png_bytes()."""

    def test_returns_bytes_with_png_signature(self) -> None:
        """Result starts with the 8-byte PNG file signature."""
        data = minimal_png_bytes()
        assert isinstance(data, bytes)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_reasonable_size(self) -> None:
        """A 1x1 RGB PNG should be between 50 and 200 bytes."""
        data = minimal_png_bytes()
        assert 50 <= len(data) <= 200

    def test_idempotent(self) -> None:
        """Multiple calls return identical bytes."""
        assert minimal_png_bytes() == minimal_png_bytes()


class TestSanitizeFilenameReport:
    """Tests for sanitize_filename_report()."""

    def test_removes_unsafe_characters(self) -> None:
        """Characters like <, >, :, etc. are replaced with underscores."""
        result = sanitize_filename_report('file<name>:test/"foo"')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result

    def test_strips_leading_trailing_dots_and_underscores(self) -> None:
        """Leading/trailing dots and underscores are stripped."""
        result = sanitize_filename_report("._test_.")
        assert not result.startswith(".")
        assert not result.startswith("_")
        assert not result.endswith(".")
        assert not result.endswith("_")

    def test_preserves_korean_characters(self) -> None:
        """Korean text is preserved in the filename."""
        result = sanitize_filename_report("시험_결과")
        assert "시험" in result
        assert "결과" in result

    def test_plain_name_unchanged(self) -> None:
        """A name with no unsafe characters is returned unchanged."""
        assert sanitize_filename_report("report_2024") == "report_2024"
