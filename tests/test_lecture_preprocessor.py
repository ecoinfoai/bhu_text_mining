"""Tests for lecture_preprocessor module.

Covers:
- Path validation (traversal rejection)
- File loading with encoding detection (UTF-8 / EUC-KR fallback)
- Korean filler word removal
- Repeated character normalization
- Mixed Korean-English token splitting
- Stopword building (3-layer) and filtering
- CleanedTranscript dataclass
- Full 8-step preprocessing pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest


# -------------------------------------------------------------------
# TestValidatePath
# -------------------------------------------------------------------

class TestValidatePath:
    """Test validate_path() path traversal rejection."""

    def test_validate_path_rejects_traversal(self) -> None:
        """Paths containing ``../`` raise ValueError."""
        from forma.lecture_preprocessor import validate_path

        with pytest.raises(ValueError, match="경로"):
            validate_path("data/../etc/passwd")

    def test_validate_path_accepts_normal(self) -> None:
        """Normal paths pass without error."""
        from forma.lecture_preprocessor import validate_path

        validate_path("/home/user/data/transcript.txt")
        validate_path("data/week01/transcript.txt")

    def test_validate_path_rejects_backslash_traversal(self) -> None:
        """Backslash traversal (Windows-style) is rejected."""
        from forma.lecture_preprocessor import validate_path

        with pytest.raises(ValueError, match="경로"):
            validate_path("data\\..\\etc\\passwd")

    def test_validate_path_rejects_null_byte(self) -> None:
        """Paths with null bytes are rejected."""
        from forma.lecture_preprocessor import validate_path

        with pytest.raises(ValueError, match="null"):
            validate_path("/tmp/test\x00.txt")

    def test_validate_path_allows_triple_dot(self) -> None:
        """Triple-dot '.../foo' is not traversal and should pass."""
        from forma.lecture_preprocessor import validate_path

        validate_path(".../foo")  # Should not raise

    def test_validate_path_rejects_trailing_dotdot(self) -> None:
        """Trailing '..' at end of path is traversal."""
        from forma.lecture_preprocessor import validate_path

        with pytest.raises(ValueError, match="경로"):
            validate_path("data/..")

    def test_validate_path_rejects_leading_dotdot(self) -> None:
        """Leading '..' at start of path is traversal."""
        from forma.lecture_preprocessor import validate_path

        with pytest.raises(ValueError, match="경로"):
            validate_path("../data/file.txt")


# -------------------------------------------------------------------
# TestLoadAndDecodeBOM
# -------------------------------------------------------------------

class TestLoadAndDecodeBOM:
    """Test load_and_decode() UTF-8 BOM handling."""

    def test_bom_stripped_from_content(
        self, tmp_path: Path,
    ) -> None:
        """UTF-8 BOM is stripped automatically."""
        from forma.lecture_preprocessor import load_and_decode

        p = tmp_path / "bom.txt"
        with open(p, "wb") as f:
            f.write(b"\xef\xbb\xbf")  # UTF-8 BOM
            f.write("세포 분열".encode("utf-8"))
        text, enc = load_and_decode(str(p))
        assert not text.startswith("\ufeff")
        assert text.startswith("세포")
        assert enc == "utf-8"

    def test_no_bom_still_works(
        self, tmp_path: Path,
    ) -> None:
        """Non-BOM UTF-8 files still work with utf-8-sig."""
        from forma.lecture_preprocessor import load_and_decode

        p = tmp_path / "nobom.txt"
        p.write_text("세포 분열 과정", encoding="utf-8")
        text, enc = load_and_decode(str(p))
        assert text == "세포 분열 과정"
        assert enc == "utf-8"


# -------------------------------------------------------------------
# TestSplitMixedTokensPreserve
# -------------------------------------------------------------------

class TestSplitMixedTokensPreserve:
    """Test split_mixed_tokens() preserves biology abbreviations."""

    def test_preserves_t_cell(self) -> None:
        """T세포 is not split into 'T 세포'."""
        from forma.lecture_preprocessor import split_mixed_tokens

        result = split_mixed_tokens("T세포가 면역에 중요합니다")
        assert "T세포" in result

    def test_preserves_b_cell(self) -> None:
        """B세포 is not split into 'B 세포'."""
        from forma.lecture_preprocessor import split_mixed_tokens

        result = split_mixed_tokens("B세포는 항체를 생산합니다")
        assert "B세포" in result

    def test_still_splits_non_abbreviations(self) -> None:
        """Non-abbreviation mixed tokens are still split."""
        from forma.lecture_preprocessor import split_mixed_tokens

        result = split_mixed_tokens("ATP합성효소")
        assert result == "ATP 합성효소"


# -------------------------------------------------------------------
# TestRemoveFillers
# -------------------------------------------------------------------

class TestRemoveFillers:
    """Test remove_fillers() Korean filler word removal."""

    def test_remove_fillers_korean(self) -> None:
        """Removes single-char fillers: 어, 음, 그."""
        from forma.lecture_preprocessor import remove_fillers

        text = "어 오늘은 음 세포에 대해 그 설명하겠습니다"
        result = remove_fillers(text)
        assert "세포" in result
        assert "설명하겠습니다" in result
        # Fillers should be removed (not present as standalone)
        words = result.split()
        assert "어" not in words
        assert "음" not in words
        assert "그" not in words

    def test_remove_fillers_repeated(self) -> None:
        """Removes repeated fillers: 어어, 음음, 그그."""
        from forma.lecture_preprocessor import remove_fillers

        text = "어어 이 부분은 음음 중요합니다 그그"
        result = remove_fillers(text)
        words = result.split()
        assert "어어" not in words
        assert "음음" not in words
        assert "그그" not in words
        assert "중요합니다" in result

    def test_remove_fillers_preserves_content(self) -> None:
        """Does not remove non-filler words."""
        from forma.lecture_preprocessor import remove_fillers

        text = "어떤 세포는 에너지를 생산합니다"
        result = remove_fillers(text)
        assert "어떤" in result
        assert "세포는" in result
        assert "에너지를" in result


# -------------------------------------------------------------------
# TestNormalizeRepeatedChars
# -------------------------------------------------------------------

class TestNormalizeRepeatedChars:
    """Test normalize_repeated_chars() compression."""

    def test_normalize_repeated_chars(self) -> None:
        """3+ consecutive identical chars compressed to 2."""
        from forma.lecture_preprocessor import normalize_repeated_chars

        assert normalize_repeated_chars("아아아아") == "아아"
        assert normalize_repeated_chars("ㅋㅋㅋㅋ") == "ㅋㅋ"
        assert normalize_repeated_chars("ㅎㅎㅎ") == "ㅎㅎ"

    def test_normalize_preserves_two(self) -> None:
        """Exactly 2 consecutive chars remain unchanged."""
        from forma.lecture_preprocessor import normalize_repeated_chars

        assert normalize_repeated_chars("아아") == "아아"
        assert normalize_repeated_chars("ㅋㅋ") == "ㅋㅋ"


# -------------------------------------------------------------------
# TestSplitMixedTokens
# -------------------------------------------------------------------

class TestSplitMixedTokens:
    """Test split_mixed_tokens() language boundary splitting."""

    def test_split_mixed_tokens(self) -> None:
        """English-to-Korean boundary inserts space."""
        from forma.lecture_preprocessor import split_mixed_tokens

        result = split_mixed_tokens("ATP합성효소")
        assert result == "ATP 합성효소"

    def test_split_mixed_tokens_korean_first(self) -> None:
        """Korean-to-English boundary inserts space."""
        from forma.lecture_preprocessor import split_mixed_tokens

        result = split_mixed_tokens("세포DNA")
        assert result == "세포 DNA"


# -------------------------------------------------------------------
# TestBuildStopwords
# -------------------------------------------------------------------

class TestBuildStopwords:
    """Test build_stopwords() 3-layer stopword construction."""

    def test_build_stopwords_includes_korean_grammar(self) -> None:
        """Common Korean particles present."""
        from forma.lecture_preprocessor import build_stopwords

        sw = build_stopwords()
        for particle in ("은", "는", "이", "가"):
            assert particle in sw, (
                f"Korean particle '{particle}' missing"
            )

    def test_build_stopwords_includes_english(self) -> None:
        """Standard English function words present."""
        from forma.lecture_preprocessor import build_stopwords

        sw = build_stopwords()
        for word in ("the", "a", "of", "is"):
            assert word in sw, (
                f"English stopword '{word}' missing"
            )

    def test_build_stopwords_includes_lecture_discourse(
        self,
    ) -> None:
        """Lecture discourse markers present."""
        from forma.lecture_preprocessor import build_stopwords

        sw = build_stopwords()
        for word in ("okay", "right", "basically"):
            assert word in sw, (
                f"Lecture discourse word '{word}' missing"
            )

    def test_build_stopwords_merges_extras(self) -> None:
        """Extra stopwords merged into result."""
        from forma.lecture_preprocessor import build_stopwords

        sw = build_stopwords(
            extra_stopwords=["커스텀단어", "another"],
        )
        assert "커스텀단어" in sw
        assert "another" in sw


# -------------------------------------------------------------------
# TestFilterStopwords
# -------------------------------------------------------------------

class TestFilterStopwords:
    """Test filter_stopwords() with abbreviation preservation."""

    def test_filter_stopwords_removes_stopwords(self) -> None:
        """Stopwords removed from word list."""
        from forma.lecture_preprocessor import filter_stopwords

        words = ["세포", "은", "에너지", "를", "생산"]
        stopwords = frozenset({"은", "를"})
        abbreviations = frozenset()
        result = filter_stopwords(
            words, stopwords, abbreviations,
        )
        assert result == ["세포", "에너지", "생산"]

    def test_filter_stopwords_preserves_abbreviations(
        self,
    ) -> None:
        """ATP, DNA, RNA kept even if in stopwords."""
        from forma.lecture_preprocessor import (
            filter_stopwords,
            BIOLOGY_ABBREVIATIONS,
        )

        words = ["ATP", "세포", "DNA", "RNA"]
        stopwords = frozenset({"ATP", "DNA"})
        result = filter_stopwords(
            words, stopwords, BIOLOGY_ABBREVIATIONS,
        )
        assert "ATP" in result
        assert "DNA" in result

    def test_filter_stopwords_preserves_extra_abbreviations(
        self,
    ) -> None:
        """User-added abbreviations preserved."""
        from forma.lecture_preprocessor import filter_stopwords

        words = ["PCR", "세포", "the"]
        stopwords = frozenset({"PCR", "the"})
        abbreviations = frozenset({"PCR"})
        result = filter_stopwords(
            words, stopwords, abbreviations,
        )
        assert "PCR" in result
        assert "the" not in result


# -------------------------------------------------------------------
# TestPreprocessTranscript
# -------------------------------------------------------------------

class TestPreprocessTranscript:
    """Test preprocess_transcript() full 8-step pipeline."""

    def test_preprocess_transcript_empty_file(
        self, tmp_path: Path,
    ) -> None:
        """Empty file raises ValueError with Korean message."""
        from forma.lecture_preprocessor import preprocess_transcript

        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="비어 있습니다"):
            preprocess_transcript(str(p), "A", 1)

    def test_preprocess_transcript_empty_after_cleaning(
        self, tmp_path: Path,
    ) -> None:
        """File with only fillers raises ValueError."""
        from forma.lecture_preprocessor import preprocess_transcript

        p = tmp_path / "fillers.txt"
        p.write_text(
            "어 음 그 저 뭐 아 예 네 응",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            preprocess_transcript(str(p), "A", 1)

    def test_preprocess_transcript_exceeds_length(
        self, tmp_path: Path,
    ) -> None:
        """File exceeding MAX_TRANSCRIPT_LENGTH raises ValueError."""
        from forma.lecture_preprocessor import (
            preprocess_transcript,
            MAX_TRANSCRIPT_LENGTH,
        )

        p = tmp_path / "long.txt"
        p.write_text(
            "세포 " * (MAX_TRANSCRIPT_LENGTH + 1),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            preprocess_transcript(str(p), "A", 1)

    def test_preprocess_transcript_returns_cleaned_transcript(
        self, tmp_path: Path,
    ) -> None:
        """Valid file returns CleanedTranscript with all fields."""
        from forma.lecture_preprocessor import (
            preprocess_transcript,
            CleanedTranscript,
        )

        content = (
            "오늘은 어 세포의 ATP 합성에 대해 설명하겠습니다"
        )
        p = tmp_path / "valid.txt"
        p.write_text(content, encoding="utf-8")

        result = preprocess_transcript(str(p), "A", 3)
        assert isinstance(result, CleanedTranscript)
        assert result.class_id == "A"
        assert result.week == 3
        assert result.source_path == str(p)
        assert result.encoding_used == "utf-8"
        assert result.char_count_raw == len(content)
        assert result.char_count_cleaned > 0
        assert result.raw_text == content
        assert len(result.cleaned_text) > 0


# -------------------------------------------------------------------
# TestLoadAndDecode
# -------------------------------------------------------------------

class TestLoadAndDecode:
    """Test load_and_decode() encoding detection."""

    def test_load_and_decode_utf8(
        self, tmp_path: Path,
    ) -> None:
        """UTF-8 file decoded correctly."""
        from forma.lecture_preprocessor import load_and_decode

        p = tmp_path / "utf8.txt"
        p.write_text("세포 분열 과정", encoding="utf-8")
        text, encoding = load_and_decode(str(p))
        assert text == "세포 분열 과정"
        assert encoding == "utf-8"

    def test_load_and_decode_euckr_fallback(
        self, tmp_path: Path, caplog,
    ) -> None:
        """EUC-KR file decoded with fallback and warning logged."""
        from forma.lecture_preprocessor import load_and_decode

        p = tmp_path / "euckr.txt"
        with open(p, "w", encoding="euc-kr") as f:
            f.write("세포 분열 과정")

        with caplog.at_level(logging.WARNING):
            text, encoding = load_and_decode(str(p))
        assert "세포 분열 과정" in text
        assert encoding == "euc-kr"


# -------------------------------------------------------------------
# TestBiologyAbbreviations
# -------------------------------------------------------------------

class TestBiologyAbbreviations:
    """Test BIOLOGY_ABBREVIATIONS constant."""

    def test_biology_abbreviations_frozenset(self) -> None:
        """Contains expected biology abbreviations."""
        from forma.lecture_preprocessor import (
            BIOLOGY_ABBREVIATIONS,
        )

        assert isinstance(BIOLOGY_ABBREVIATIONS, frozenset)
        for abbr in ("ATP", "DNA", "RNA", "pH", "mRNA"):
            assert abbr in BIOLOGY_ABBREVIATIONS, (
                f"'{abbr}' missing from BIOLOGY_ABBREVIATIONS"
            )
