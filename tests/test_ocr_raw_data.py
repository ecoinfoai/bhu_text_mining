"""Tests for naver_ocr.extract_raw_ocr_data()."""

from __future__ import annotations

import pytest

from forma.naver_ocr import extract_raw_ocr_data


# ---------------------------------------------------------------------------
# Mock OCR response fixtures
# ---------------------------------------------------------------------------

def _make_field(
    infer_text: str = "hello",
    infer_confidence: float = 0.95,
    bounding_poly: list[dict] | None = None,
    field_type: str = "",
    line_break: bool = False,
) -> dict:
    """Build a single OCR field dict matching Naver API structure."""
    field: dict = {
        "inferText": infer_text,
        "inferConfidence": infer_confidence,
    }
    if bounding_poly is not None:
        field["boundingPoly"] = {"vertices": bounding_poly}
    if field_type:
        field["type"] = field_type
    if line_break:
        field["lineBreak"] = True
    return field


def _make_response(
    image_name: str = "test.jpg",
    fields: list[dict] | None = None,
    infer_result: str = "SUCCESS",
) -> dict:
    """Build a mock Naver OCR API response for one image."""
    return {
        "images": [
            {
                "name": image_name,
                "inferResult": infer_result,
                "fields": fields if fields is not None else [],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractRawOcrData:
    """Tests for extract_raw_ocr_data()."""

    def test_basic_single_field(self):
        """Single field with all data preserved."""
        fields = [
            _make_field(
                infer_text="세포",
                infer_confidence=0.98,
                bounding_poly=[
                    {"x": 0, "y": 0}, {"x": 100, "y": 0},
                    {"x": 100, "y": 50}, {"x": 0, "y": 50},
                ],
                field_type="NORMAL",
                line_break=True,
            )
        ]
        resp = [_make_response(image_name="q1.jpg", fields=fields)]
        result = extract_raw_ocr_data(resp)

        assert "q1.jpg" in result
        data = result["q1.jpg"]
        assert data["infer_result"] == "SUCCESS"
        assert data["field_count"] == 1
        assert data["confidence_mean"] == pytest.approx(0.98)
        assert data["confidence_min"] == pytest.approx(0.98)
        assert len(data["fields"]) == 1

        f = data["fields"][0]
        assert f["infer_text"] == "세포"
        assert f["infer_confidence"] == pytest.approx(0.98)
        assert f["bounding_poly"] == [
            {"x": 0, "y": 0}, {"x": 100, "y": 0},
            {"x": 100, "y": 50}, {"x": 0, "y": 50},
        ]
        assert f["type"] == "NORMAL"
        assert f["line_break"] is True

    def test_multiple_fields_stats(self):
        """Confidence stats computed from multiple fields."""
        fields = [
            _make_field(infer_text="a", infer_confidence=0.80),
            _make_field(infer_text="b", infer_confidence=0.90),
            _make_field(infer_text="c", infer_confidence=1.0),
        ]
        resp = [_make_response(fields=fields)]
        result = extract_raw_ocr_data(resp)

        data = result["test.jpg"]
        assert data["field_count"] == 3
        assert data["confidence_mean"] == pytest.approx(0.9, abs=0.001)
        assert data["confidence_min"] == pytest.approx(0.80)

    def test_empty_fields(self):
        """No fields → None stats, empty fields list."""
        resp = [_make_response(fields=[])]
        result = extract_raw_ocr_data(resp)

        data = result["test.jpg"]
        assert data["field_count"] == 0
        assert data["confidence_mean"] is None
        assert data["confidence_min"] is None
        assert data["fields"] == []

    def test_missing_bounding_poly(self):
        """Fields without boundingPoly handled gracefully."""
        fields = [_make_field(infer_text="x", infer_confidence=0.75)]
        resp = [_make_response(fields=fields)]
        result = extract_raw_ocr_data(resp)

        f = result["test.jpg"]["fields"][0]
        assert f["bounding_poly"] is None

    def test_missing_optional_keys(self):
        """Fields missing type/lineBreak get defaults."""
        fields = [{"inferText": "word", "inferConfidence": 0.85}]
        resp = [_make_response(fields=fields)]
        result = extract_raw_ocr_data(resp)

        f = result["test.jpg"]["fields"][0]
        assert f["type"] == ""
        assert f["line_break"] is False

    def test_multiple_images(self):
        """Multiple responses (images) each get their own entry."""
        resp = [
            _make_response(
                image_name="q1.jpg",
                fields=[_make_field(infer_text="a", infer_confidence=0.9)],
            ),
            _make_response(
                image_name="q2.jpg",
                fields=[_make_field(infer_text="b", infer_confidence=0.8)],
            ),
        ]
        result = extract_raw_ocr_data(resp)

        assert len(result) == 2
        assert "q1.jpg" in result
        assert "q2.jpg" in result

    def test_error_infer_result(self):
        """ERROR infer_result still returns data structure."""
        resp = [_make_response(infer_result="ERROR", fields=[])]
        result = extract_raw_ocr_data(resp)

        data = result["test.jpg"]
        assert data["infer_result"] == "ERROR"
        assert data["field_count"] == 0

    def test_empty_responses(self):
        """Empty response list → empty result dict."""
        result = extract_raw_ocr_data([])
        assert result == {}

    def test_response_missing_images_key(self):
        """Response without 'images' key → skip gracefully."""
        result = extract_raw_ocr_data([{}])
        assert result == {}
