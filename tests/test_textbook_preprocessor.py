"""Tests for textbook text preprocessing (T003, T004).

T003: Tests for clean_textbook_text() — page number removal, wide
      letter-spacing header removal, HUMAN ANATOMY marker removal,
      body text and caption preservation, empty input handling.

T004: Tests for extract_bilingual_terms() — Korean(English) pattern
      extraction, complex parenthetical handling, multiple terms,
      plain Korean text (no matches).
"""

from __future__ import annotations


from forma.textbook_preprocessor import clean_textbook_text, extract_bilingual_terms


# ----------------------------------------------------------------
# T003: clean_textbook_text
# ----------------------------------------------------------------


class TestCleanTextbookText:
    """Tests for clean_textbook_text()."""

    def test_removes_chapter_page_numbers(self) -> None:
        """Page number lines like '제 3 장 피부   43' are removed."""
        raw = "제 3 장 피부   43\n피부는 신체의 전체 표면을 덮고 있다."
        result = clean_textbook_text(raw)
        assert "제 3 장 피부   43" not in result
        assert "피부는 신체의 전체 표면을 덮고 있다." in result

    def test_removes_chapter_page_numbers_variant(self) -> None:
        """Page number lines like '제 1 장 서론    3' are removed."""
        raw = "제 1 장 서론    3\n인체의 구조와 기능을 학습한다."
        result = clean_textbook_text(raw)
        assert "제 1 장 서론    3" not in result
        assert "인체의 구조와 기능을 학습한다." in result

    def test_removes_wide_letter_spacing_headers(self) -> None:
        """Wide letter-spacing headers like 'C H A P T E R  03' are removed."""
        raw = "C H A P T E R  03\n피부(skin)는 신체의 전체 표면을 덮고 있는 기관이다."
        result = clean_textbook_text(raw)
        assert "C H A P T E R" not in result
        assert "피부(skin)는 신체의 전체 표면을 덮고 있는 기관이다." in result

    def test_removes_wide_letter_spacing_headers_variant(self) -> None:
        """Wide letter-spacing headers like 'C H A P T E R  01' are removed."""
        raw = "C H A P T E R  01\n서론 내용이 여기에 있습니다."
        result = clean_textbook_text(raw)
        assert "C H A P T E R" not in result

    def test_removes_human_anatomy_markers(self) -> None:
        """'H U M A N  A N A T O M Y  &  P H Y S I O L O G Y' markers removed."""
        raw = (
            "H U M A N  A N A T O M Y  &  P H Y S I O L O G Y\n"
            "표피는 피부의 가장 바깥층이다."
        )
        result = clean_textbook_text(raw)
        assert "H U M A N" not in result
        assert "표피는 피부의 가장 바깥층이다." in result

    def test_preserves_body_text(self) -> None:
        """Normal body text is preserved."""
        raw = "피부(skin)는 신체의 전체 표면을 덮고 있는 가장 큰 기관이다."
        result = clean_textbook_text(raw)
        assert "피부(skin)는 신체의 전체 표면을 덮고 있는 가장 큰 기관이다." in result

    def test_preserves_figure_table_captions(self) -> None:
        """Figure/table captions like '(그림 3-1, 3-2)' are preserved."""
        raw = "피부의 구조를 살펴보면 (그림 3-1, 3-2) 표피와 진피로 구성된다."
        result = clean_textbook_text(raw)
        assert "(그림 3-1, 3-2)" in result

    def test_removes_standalone_page_numbers(self) -> None:
        """Standalone page number lines (just a number) are removed."""
        raw = "표피는 피부의 바깥층이다.\n43\n진피는 표피 아래에 위치한다."
        result = clean_textbook_text(raw)
        assert "\n43\n" not in result
        assert "표피는 피부의 바깥층이다." in result
        assert "진피는 표피 아래에 위치한다." in result

    def test_collapses_multiple_blank_lines(self) -> None:
        """Multiple consecutive blank lines are collapsed to single blank line."""
        raw = "첫 번째 문단.\n\n\n\n두 번째 문단."
        result = clean_textbook_text(raw)
        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in result
        assert "첫 번째 문단." in result
        assert "두 번째 문단." in result

    def test_empty_input(self) -> None:
        """Empty string returns empty string."""
        assert clean_textbook_text("") == ""

    def test_whitespace_only_input(self) -> None:
        """Whitespace-only input returns empty or whitespace-collapsed result."""
        result = clean_textbook_text("   \n\n   ")
        assert result.strip() == ""


# ----------------------------------------------------------------
# T004: extract_bilingual_terms
# ----------------------------------------------------------------


class TestExtractBilingualTerms:
    """Tests for extract_bilingual_terms()."""

    def test_simple_bilingual_term(self) -> None:
        """'표피(epidermis)' extracts ('표피', 'epidermis')."""
        text = "표피(epidermis)는 피부의 가장 바깥층이다."
        result = extract_bilingual_terms(text)
        assert ("표피", "epidermis") in result

    def test_another_bilingual_term(self) -> None:
        """'진피(dermis)' extracts ('진피', 'dermis')."""
        text = "진피(dermis)는 표피 아래에 위치한다."
        result = extract_bilingual_terms(text)
        assert ("진피", "dermis") in result

    def test_complex_parenthetical(self) -> None:
        """Complex parenthetical with multiple terms is handled."""
        text = "피부밑조직(피하조직, subcutaneous tissue)은 진피 아래에 있다."
        result = extract_bilingual_terms(text)
        # Should extract the English portion
        found_english = [en for ko, en in result if "subcutaneous" in en]
        assert len(found_english) > 0

    def test_multiple_terms_in_sentence(self) -> None:
        """Multiple bilingual terms in one sentence are all extracted."""
        text = "피부는 표피(epidermis)와 진피(dermis)로 구성된다."
        result = extract_bilingual_terms(text)
        english_terms = {en for _, en in result}
        assert "epidermis" in english_terms
        assert "dermis" in english_terms

    def test_no_matches_in_plain_korean(self) -> None:
        """Plain Korean text without bilingual annotations returns empty list."""
        text = "피부는 인체의 가장 큰 기관이다."
        result = extract_bilingual_terms(text)
        assert result == []

    def test_korean_only_parenthetical_excluded(self) -> None:
        """Korean-only parenthetical like '피부밑조직(피하조직)' is excluded."""
        text = "피부밑조직(피하조직)은 지방층이다."
        result = extract_bilingual_terms(text)
        # Should not match since parenthetical has no English alphabetic chars
        assert len(result) == 0


# ----------------------------------------------------------------
# T004: Enhanced body-only filtering (v2)
# ----------------------------------------------------------------


class TestCleanTextbookTextV2:
    """Tests for enhanced clean_textbook_text() — figure/table/quiz removal."""

    def test_removes_standalone_figure_titles(self) -> None:
        """Standalone figure title lines like '그림 3-1 발열의 병인론' are removed."""
        raw = "본문 내용입니다.\n그림 3-1 발열의 병인론\n다음 문단 시작."
        result = clean_textbook_text(raw)
        assert "그림 3-1 발열의 병인론" not in result
        assert "본문 내용입니다." in result
        assert "다음 문단 시작." in result

    def test_removes_standalone_figure_titles_endash(self) -> None:
        """Figure titles with en-dash like '그림 3–1 피부 구조' are removed."""
        raw = "본문.\n그림 3–1 피부 구조\n다음."
        result = clean_textbook_text(raw)
        assert "그림 3–1 피부 구조" not in result

    def test_removes_standalone_table_titles(self) -> None:
        """Standalone table title lines like '표 3-1 피부의 구조' are removed."""
        raw = "본문 내용.\n표 3-1 피부의 구조\n다음 문단."
        result = clean_textbook_text(raw)
        assert "표 3-1 피부의 구조" not in result
        assert "본문 내용." in result
        assert "다음 문단." in result

    def test_removes_standalone_table_titles_endash(self) -> None:
        """Table titles with en-dash like '표 2–3 근육 유형' are removed."""
        raw = "본문.\n표 2–3 근육 유형\n다음."
        result = clean_textbook_text(raw)
        assert "표 2–3 근육 유형" not in result

    def test_preserves_inline_figure_refs(self) -> None:
        """Inline figure references like '(그림 3-1)' in body text are preserved."""
        raw = "피부의 구조를 살펴보면 (그림 3-1) 표피와 진피로 구성된다."
        result = clean_textbook_text(raw)
        assert "(그림 3-1)" in result
        assert "표피와 진피로 구성된다." in result

    def test_preserves_inline_table_refs(self) -> None:
        """Inline table references like '(표 3-1)' in body text are preserved."""
        raw = "세포의 종류는 (표 3-1) 참고하라."
        result = clean_textbook_text(raw)
        assert "(표 3-1)" in result

    def test_removes_quiz_section_문제(self) -> None:
        """Lines starting with quiz markers like '문제' remove the section."""
        raw = "본문 내용입니다.\n문제\n1. 표피는 무엇인가?\n2. 진피의 역할은?"
        result = clean_textbook_text(raw)
        assert "본문 내용입니다." in result
        assert "문제" not in result
        assert "표피는 무엇인가?" not in result

    def test_removes_quiz_section_퀴즈(self) -> None:
        """Lines starting with '퀴즈' remove the section."""
        raw = "본문.\n퀴즈\n1. 답하시오."
        result = clean_textbook_text(raw)
        assert "본문." in result
        assert "퀴즈" not in result

    def test_removes_quiz_section_복습문제(self) -> None:
        """Lines starting with '복습문제' remove the section."""
        raw = "본문.\n복습문제\n1. 설명하시오."
        result = clean_textbook_text(raw)
        assert "본문." in result
        assert "복습문제" not in result

    def test_removes_quiz_section_연습문제(self) -> None:
        """Lines starting with '연습문제' remove the section."""
        raw = "본문.\n연습문제\n1. 서술하시오."
        result = clean_textbook_text(raw)
        assert "본문." in result
        assert "연습문제" not in result


# ----------------------------------------------------------------
# T005: prepare_textbook_for_llm
# ----------------------------------------------------------------


class TestPrepareTextbookForLLM:
    """Tests for prepare_textbook_for_llm()."""

    def test_returns_cleaned_body_without_summary(self) -> None:
        """Without summary, returns (cleaned_body, None)."""
        from forma.textbook_preprocessor import prepare_textbook_for_llm

        raw = "C H A P T E R  03\n본문 내용입니다.\n그림 3-1 발열도\n더 많은 본문."
        body, guide = prepare_textbook_for_llm(raw)
        assert "본문 내용입니다." in body
        assert "그림 3-1 발열도" not in body
        assert guide is None

    def test_returns_cleaned_body_with_summary(self) -> None:
        """With summary path, returns (cleaned_body, summary_text)."""
        import tempfile
        from pathlib import Path

        from forma.textbook_preprocessor import prepare_textbook_for_llm

        raw = "본문 내용입니다.\n표 3-1 피부 구조\n진피에 대한 설명."
        summary_text = "# 3장 피부\n## 1. 표피\n- 4층 구조"

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "Ch03_Summary.md"
            summary_path.write_text(summary_text, encoding="utf-8")

            body, guide = prepare_textbook_for_llm(raw, summary_path=str(summary_path))
            assert "본문 내용입니다." in body
            assert "표 3-1 피부 구조" not in body
            assert guide is not None
            assert "3장 피부" in guide

    def test_nonexistent_summary_returns_none(self) -> None:
        """Non-existent summary path returns None for guide."""
        from forma.textbook_preprocessor import prepare_textbook_for_llm

        body, guide = prepare_textbook_for_llm("본문 내용.", summary_path="/nonexistent/file.md")
        assert "본문 내용." in body
        assert guide is None
