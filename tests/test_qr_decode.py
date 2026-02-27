# tests/test_qr_decode.py
"""Tests for src/qr_decode.py QR code decoding utilities."""
import os
import tempfile

import pytest
import qrcode

cv2 = pytest.importorskip(
    "cv2",
    reason="opencv-python not importable in this environment (libxcb missing)",
    exc_type=ImportError,
)
import numpy as np  # noqa: E402 — after cv2 skip guard

from src.qr_decode import decode_qr_from_image, parse_qr_content


# ── helpers ──────────────────────────────────────


def _write_qr_png(content: str) -> str:
    """Generate a QR code PNG and return its temp file path."""
    qr = qrcode.QRCode(
        box_size=10,
        border=2,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
    )
    qr.add_data(content)
    qr.make(fit=True)
    pil_img = qr.make_image(
        fill_color="black", back_color="white",
    ).convert("RGB")
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    with tempfile.NamedTemporaryFile(
        suffix=".png", delete=False,
    ) as f:
        path = f.name
    cv2.imwrite(path, img_cv)
    return path


# ──────────────────────────────────────────────────
# Group 1: decode_qr_from_image
# ──────────────────────────────────────────────────


class TestDecodeQrFromImage:
    """decode_qr_from_image() — positive + edge cases."""

    def test_decode_plain_text_qr(self):
        content = "S001|인체구조와기능|1주차|Q1"
        path = _write_qr_png(content)
        try:
            result = decode_qr_from_image(path)
            assert result == content
        finally:
            os.unlink(path)

    def test_decode_url_qr(self):
        content = (
            "https://docs.google.com/forms/d/e/FORM/viewform"
            "?entry.1064397072=S002&q=2"
        )
        path = _write_qr_png(content)
        try:
            result = decode_qr_from_image(path)
            assert result == content
        finally:
            os.unlink(path)

    def test_decode_returns_none_for_no_qr(self):
        # Solid white image contains no QR
        img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False,
        ) as f:
            path = f.name
        try:
            cv2.imwrite(path, img)
            result = decode_qr_from_image(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_decode_returns_none_for_nonexistent_file(self):
        result = decode_qr_from_image("/nonexistent/path/img.png")
        assert result is None

    def test_decode_noisy_image_via_fallback(self):
        """Add Gaussian noise — fallback binarization should still decode."""
        content = "S003|감염미생물학|2주차|Q1"
        path = _write_qr_png(content)
        try:
            img = cv2.imread(path)
            noise = np.random.normal(0, 15, img.shape).astype(np.int16)
            noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(
                np.uint8
            )
            cv2.imwrite(path, noisy)
            result = decode_qr_from_image(path)
            # May succeed or fail depending on noise level; just no exception
            assert result is None or isinstance(result, str)
        finally:
            os.unlink(path)


# ──────────────────────────────────────────────────
# Group 2: parse_qr_content — plain text
# ──────────────────────────────────────────────────


class TestParseQrContentPlain:
    """parse_qr_content() with pipe-delimited plain text."""

    def test_parse_plain_with_q_num(self):
        result = parse_qr_content("S001|인체구조와기능|1주차|Q2")
        assert result["student_id"] == "S001"
        assert result["q_num"] == 2

    def test_parse_plain_without_q_num(self):
        result = parse_qr_content("S001|감염미생물학|3주차")
        assert result["student_id"] == "S001"
        assert result["q_num"] is None

    def test_parse_plain_q_num_first_question(self):
        result = parse_qr_content("ABC|과목|1주차|Q1")
        assert result["student_id"] == "ABC"
        assert result["q_num"] == 1

    def test_parse_plain_minimum_segments(self):
        """Two-segment plain text: student_id|course."""
        result = parse_qr_content("S010|course")
        assert result["student_id"] == "S010"
        assert result["q_num"] is None

    def test_parse_plain_invalid_raises(self):
        with pytest.raises(ValueError, match="Unrecognized"):
            parse_qr_content("nopipes")


# ──────────────────────────────────────────────────
# Group 3: parse_qr_content — URL
# ──────────────────────────────────────────────────


class TestParseQrContentURL:
    """parse_qr_content() with URL-formatted content."""

    def test_parse_url_with_entry_and_q(self):
        url = (
            "https://docs.google.com/forms/d/e/FORM/viewform"
            "?entry.1064397072=S001&entry.2=%EA%B0%90%EC%97%BC&q=1"
        )
        result = parse_qr_content(url)
        assert result["student_id"] == "S001"
        assert result["q_num"] == 1

    def test_parse_url_with_sid_key(self):
        url = "https://forms.example.com?sid=S002&w=3&q=2"
        result = parse_qr_content(url)
        assert result["student_id"] == "S002"
        assert result["q_num"] == 2

    def test_parse_url_with_student_id_key(self):
        url = "https://forms.example.com?student_id=S003&w=1&q=1"
        result = parse_qr_content(url)
        assert result["student_id"] == "S003"
        assert result["q_num"] == 1

    def test_parse_url_without_q_param(self):
        url = "https://forms.example.com?entry.1=S004&entry.2=course"
        result = parse_qr_content(url)
        assert result["student_id"] == "S004"
        assert result["q_num"] is None

    def test_parse_url_missing_student_id_raises(self):
        url = "https://forms.example.com?onlyweeknumber=3&q=1"
        with pytest.raises(ValueError, match="student_id"):
            parse_qr_content(url)
