"""Tests for ocr_compare.py — OCR vs LLM Vision comparison module."""

from __future__ import annotations

from unittest.mock import MagicMock


from forma.ocr_compare import (
    ComparisonResult,
    FieldComparison,
    align_ocr_llm_tokens,
    build_comparison_prompt,
    compare_single_image,
    format_comparison_report,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_OCR_FIELDS = [
    {"infer_text": "세포", "infer_confidence": 0.98, "bounding_poly": None, "type": "", "line_break": False},
    {"infer_text": "분열", "infer_confidence": 0.95, "bounding_poly": None, "type": "", "line_break": False},
    {"infer_text": "과정", "infer_confidence": 0.70, "bounding_poly": None, "type": "", "line_break": True},
]


# ---------------------------------------------------------------------------
# FieldComparison / ComparisonResult dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Verify dataclass structure."""

    def test_field_comparison_fields(self):
        fc = FieldComparison(
            field_index=0,
            ocr_text="세포",
            llm_text="세포",
            ocr_confidence=0.98,
            match=True,
        )
        assert fc.field_index == 0
        assert fc.match is True

    def test_comparison_result_fields(self):
        cr = ComparisonResult(
            image_path="/tmp/test.jpg",
            ocr_text="세포 분열 과정",
            llm_text="세포 분열 과정",
            field_comparisons=[],
            summary={"match_count": 0, "mismatch_count": 0, "total": 0},
        )
        assert cr.image_path == "/tmp/test.jpg"
        assert cr.ocr_text == "세포 분열 과정"


# ---------------------------------------------------------------------------
# build_comparison_prompt tests
# ---------------------------------------------------------------------------


class TestBuildComparisonPrompt:
    """Tests for build_comparison_prompt()."""

    def test_basic_prompt(self):
        """Prompt includes instruction to read text from image."""
        prompt = build_comparison_prompt(SAMPLE_OCR_FIELDS)
        assert "이미지" in prompt or "image" in prompt.lower()

    def test_prompt_with_context(self):
        """Context (subject/question) is included in prompt."""
        context = {"subject": "생물학", "question": "세포 분열을 설명하시오"}
        prompt = build_comparison_prompt(SAMPLE_OCR_FIELDS, context=context)
        assert "생물학" in prompt
        assert "세포 분열" in prompt

    def test_prompt_without_context(self):
        """Prompt works without context."""
        prompt = build_comparison_prompt(SAMPLE_OCR_FIELDS, context=None)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# align_ocr_llm_tokens tests
# ---------------------------------------------------------------------------


class TestAlignOcrLlmTokens:
    """Tests for align_ocr_llm_tokens()."""

    def test_exact_match(self):
        """All OCR tokens appear in LLM text."""
        ocr_tokens = ["세포", "분열", "과정"]
        llm_text = "세포 분열 과정"
        result = align_ocr_llm_tokens(ocr_tokens, llm_text)
        assert len(result) == 3
        for ocr_tok, llm_tok, match in result:
            assert match is True

    def test_partial_mismatch(self):
        """Some OCR tokens differ from LLM."""
        ocr_tokens = ["세포", "분렬", "과정"]  # 분렬 is OCR error
        llm_text = "세포 분열 과정"
        result = align_ocr_llm_tokens(ocr_tokens, llm_text)
        assert result[0][2] is True  # 세포 matches
        assert result[1][2] is False  # 분렬 ≠ 분열
        assert result[2][2] is True  # 과정 matches

    def test_llm_has_extra_tokens(self):
        """LLM may produce more tokens than OCR fields."""
        ocr_tokens = ["세포", "분열"]
        llm_text = "세포 분열 과정 설명"
        result = align_ocr_llm_tokens(ocr_tokens, llm_text)
        assert len(result) == 2  # Only OCR token count matters

    def test_empty_ocr_tokens(self):
        """Empty OCR tokens → empty result."""
        result = align_ocr_llm_tokens([], "some text")
        assert result == []

    def test_empty_llm_text(self):
        """Empty LLM text → all mismatches."""
        result = align_ocr_llm_tokens(["세포", "분열"], "")
        assert len(result) == 2
        for _, _, match in result:
            assert match is False


# ---------------------------------------------------------------------------
# compare_single_image tests
# ---------------------------------------------------------------------------


class TestCompareSingleImage:
    """Tests for compare_single_image()."""

    def test_basic_comparison(self, tmp_path):
        """End-to-end comparison with mocked LLM."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0fake-jpg")

        naver_raw = {
            "infer_result": "SUCCESS",
            "fields": SAMPLE_OCR_FIELDS,
            "field_count": 3,
            "confidence_mean": 0.877,
            "confidence_min": 0.70,
        }

        mock_provider = MagicMock()
        mock_provider.generate_with_image.return_value = "세포 분열 과정"

        result = compare_single_image(
            image_path=str(img_file),
            naver_raw=naver_raw,
            llm_provider=mock_provider,
        )

        assert isinstance(result, ComparisonResult)
        assert result.image_path == str(img_file)
        assert result.ocr_text == "세포 분열 과정"
        assert result.llm_text == "세포 분열 과정"
        assert len(result.field_comparisons) == 3
        mock_provider.generate_with_image.assert_called_once()

    def test_comparison_with_mismatches(self, tmp_path):
        """Mismatches are correctly identified."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0fake")

        naver_raw = {
            "infer_result": "SUCCESS",
            "fields": [
                {
                    "infer_text": "세포",
                    "infer_confidence": 0.98,
                    "bounding_poly": None,
                    "type": "",
                    "line_break": False,
                },
                {
                    "infer_text": "분렬",
                    "infer_confidence": 0.40,
                    "bounding_poly": None,
                    "type": "",
                    "line_break": False,
                },
            ],
            "field_count": 2,
            "confidence_mean": 0.69,
            "confidence_min": 0.40,
        }

        mock_provider = MagicMock()
        mock_provider.generate_with_image.return_value = "세포 분열"

        result = compare_single_image(
            image_path=str(img_file),
            naver_raw=naver_raw,
            llm_provider=mock_provider,
        )

        assert result.summary["mismatch_count"] >= 1
        # The low-confidence field should be a mismatch
        mismatches = [fc for fc in result.field_comparisons if not fc.match]
        assert len(mismatches) >= 1

    def test_comparison_with_context(self, tmp_path):
        """Context is passed to prompt builder."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0fake")

        naver_raw = {
            "infer_result": "SUCCESS",
            "fields": SAMPLE_OCR_FIELDS,
            "field_count": 3,
            "confidence_mean": 0.877,
            "confidence_min": 0.70,
        }

        mock_provider = MagicMock()
        mock_provider.generate_with_image.return_value = "세포 분열 과정"

        context = {"subject": "생물학"}
        compare_single_image(
            image_path=str(img_file),
            naver_raw=naver_raw,
            llm_provider=mock_provider,
            context=context,
        )

        # Verify prompt included context
        call_args = mock_provider.generate_with_image.call_args
        prompt = call_args.kwargs.get("prompt") or call_args.args[0]
        assert "생물학" in prompt

    def test_summary_counts(self, tmp_path):
        """Summary has correct match/mismatch/total counts."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0fake")

        naver_raw = {
            "infer_result": "SUCCESS",
            "fields": [
                {"infer_text": "a", "infer_confidence": 0.9, "bounding_poly": None, "type": "", "line_break": False},
                {"infer_text": "b", "infer_confidence": 0.9, "bounding_poly": None, "type": "", "line_break": False},
            ],
            "field_count": 2,
            "confidence_mean": 0.9,
            "confidence_min": 0.9,
        }

        mock_provider = MagicMock()
        mock_provider.generate_with_image.return_value = "a b"

        result = compare_single_image(
            image_path=str(img_file),
            naver_raw=naver_raw,
            llm_provider=mock_provider,
        )

        assert result.summary["total"] == 2
        assert result.summary["match_count"] + result.summary["mismatch_count"] == 2


# ---------------------------------------------------------------------------
# format_comparison_report tests
# ---------------------------------------------------------------------------


class TestFormatComparisonReport:
    """Tests for format_comparison_report()."""

    def test_basic_format(self):
        result = ComparisonResult(
            image_path="/tmp/test.jpg",
            ocr_text="세포 분열",
            llm_text="세포 분열",
            field_comparisons=[
                FieldComparison(0, "세포", "세포", 0.98, True),
                FieldComparison(1, "분열", "분열", 0.95, True),
            ],
            summary={"total": 2, "match_count": 2, "mismatch_count": 0},
        )
        report = format_comparison_report(result)
        assert "Comparison Result" in report
        assert "test.jpg" in report
        assert "[O]" in report

    def test_mismatch_format(self):
        result = ComparisonResult(
            image_path="/tmp/test.jpg",
            ocr_text="분렬",
            llm_text="분열",
            field_comparisons=[
                FieldComparison(0, "분렬", "분열", 0.40, False),
            ],
            summary={"total": 1, "match_count": 0, "mismatch_count": 1},
        )
        report = format_comparison_report(result)
        assert "[X]" in report
        assert "0.40" in report


# ---------------------------------------------------------------------------
# CLI parse tests
# ---------------------------------------------------------------------------


class TestCliCompareParser:
    """Test that 'compare' subcommand parses correctly."""

    def test_parse_compare_basic(self):
        from forma.cli_ocr import _parse_args

        args = _parse_args(["compare", "--image", "test.jpg"])
        assert args.command == "compare"
        assert args.image == "test.jpg"
        assert args.provider == "gemini"
        assert args.model is None

    def test_parse_compare_with_provider(self):
        from forma.cli_ocr import _parse_args

        args = _parse_args(
            [
                "compare",
                "--image",
                "test.jpg",
                "--provider",
                "anthropic",
                "--model",
                "claude-sonnet-4-6",
            ]
        )
        assert args.provider == "anthropic"
        assert args.model == "claude-sonnet-4-6"

    def test_parse_compare_with_context(self):
        from forma.cli_ocr import _parse_args

        args = _parse_args(
            [
                "compare",
                "--image",
                "test.jpg",
                "--subject",
                "생물학",
                "--question",
                "세포 분열",
                "--answer-keywords",
                "유사분열",
            ]
        )
        assert args.subject == "생물학"
        assert args.question == "세포 분열"
        assert args.answer_keywords == "유사분열"
