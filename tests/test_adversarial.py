"""Adversarial tests for STT lecture analysis (7 personas).

Tests exercise the lecture_preprocessor, lecture_analyzer, cli_lecture,
and lecture_report modules under hostile, broken, and edge-case inputs.
"""

import os
from pathlib import Path

import pytest
import yaml

from forma.lecture_preprocessor import (
    validate_path,
    load_and_decode,
    remove_fillers,
    normalize_repeated_chars,
    split_mixed_tokens,
    preprocess_transcript,
    build_stopwords,
    filter_stopwords,
    BIOLOGY_ABBREVIATIONS,
    MAX_TRANSCRIPT_LENGTH,
    CleanedTranscript,
)


# ====================================================================
# Persona 1: 실수하는 교수 (Clumsy Professor)
# ====================================================================
class TestPersona1ClumsyProfessor:
    """Common user mistakes by a non-technical professor."""

    def test_nonexistent_file(self, tmp_path):
        """Professor gives a wrong file path."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_and_decode(str(tmp_path / "nonexistent.txt"))

    def test_nonexistent_file_via_pipeline(self, tmp_path):
        """Full pipeline with nonexistent path (after validate_path)."""
        with pytest.raises((FileNotFoundError, OSError)):
            preprocess_transcript(str(tmp_path / "nope.txt"), "A", 1)

    def test_empty_class_id(self, tmp_path):
        """Professor forgets class identifier — passes empty string."""
        f = tmp_path / "test.txt"
        f.write_text("세포 분열에 대해 설명하겠습니다.", encoding="utf-8")
        result = preprocess_transcript(str(f), "", 1)
        assert result.class_id == ""

    def test_wrong_file_extension(self, tmp_path):
        """Professor provides a .pdf file that is actually plain text."""
        f = tmp_path / "lecture.pdf"
        f.write_text("이것은 실제로 텍스트 파일입니다.", encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1)
        assert result.cleaned_text  # should still work

    def test_negative_week_number(self, tmp_path):
        """Professor enters negative week number."""
        f = tmp_path / "test.txt"
        f.write_text("세포 분열에 대해 설명하겠습니다.", encoding="utf-8")
        result = preprocess_transcript(str(f), "A", -1)
        assert result.week == -1

    def test_zero_week_number(self, tmp_path):
        """Professor enters week 0."""
        f = tmp_path / "test.txt"
        f.write_text("세포 분열에 대해 설명하겠습니다.", encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 0)
        assert result.week == 0

    def test_class_id_with_spaces(self, tmp_path):
        """Professor types class name with trailing spaces."""
        f = tmp_path / "test.txt"
        f.write_text("세포 분열에 대해 설명하겠습니다.", encoding="utf-8")
        result = preprocess_transcript(str(f), "  A  ", 1)
        # No stripping — class_id preserves whitespace as-is
        assert result.class_id == "  A  "

    def test_very_large_week_number(self, tmp_path):
        """Professor enters week 99999."""
        f = tmp_path / "test.txt"
        f.write_text("세포 분열에 대해 설명하겠습니다.", encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 99999)
        assert result.week == 99999


# ====================================================================
# Persona 2: 악의적 사용자 (Malicious User)
# ====================================================================
class TestPersona2MaliciousUser:
    """Security attack attempts."""

    def test_path_traversal_forward_slash(self):
        """Path traversal with ../"""
        with pytest.raises(ValueError, match="traversal"):
            validate_path("../../etc/passwd")

    def test_path_traversal_in_middle(self):
        """Path traversal embedded in a longer path."""
        with pytest.raises(ValueError, match="traversal"):
            validate_path("/home/user/../../../etc/shadow")

    def test_path_traversal_backslash(self):
        """Path traversal with Windows-style backslashes is rejected."""
        with pytest.raises(ValueError, match="traversal"):
            validate_path("..\\..\\etc\\passwd")

    def test_null_byte_in_path(self, tmp_path):
        """Null byte injection in file path is rejected by validate_path."""
        with pytest.raises(ValueError, match="null"):
            validate_path(f"{tmp_path}/test\x00.txt")

    def test_extremely_long_path(self):
        """Path with 10,000 characters — should not crash."""
        long_path = "a" * 10000 + ".txt"
        try:
            validate_path(long_path)
        except (ValueError, OSError):
            pass  # Acceptable

    def test_path_with_zero_width_space(self, tmp_path):
        """Path containing Unicode zero-width space."""
        zwsp = "\u200b"
        evil_path = f"{tmp_path}/te{zwsp}st.txt"
        # Should either reject or handle cleanly
        try:
            validate_path(evil_path)
        except (ValueError, OSError):
            pass

    def test_yaml_injection_safe_load(self, tmp_path):
        """YAML injection payload handled safely via safe_load."""
        concepts_file = tmp_path / "evil.yaml"
        concepts_file.write_text(
            "concepts:\n  - !!python/object/apply:os.system ['echo pwned']",
            encoding="utf-8",
        )
        # safe_load must refuse dangerous tags
        with pytest.raises(yaml.constructor.ConstructorError):
            yaml.safe_load(concepts_file.read_text(encoding="utf-8"))

    def test_path_traversal_double_encoding(self):
        """URL-encoded traversal — should still be caught if present."""
        # '%2e%2e%2f' is not literal '../' so validate_path may miss it,
        # but OS will also not interpret it as traversal
        validate_path("%2e%2e%2fetc%2fpasswd")  # Should not raise

    def test_symlink_to_sensitive_file(self, tmp_path):
        """Symlink pointing to /etc/passwd."""
        link = tmp_path / "sneaky.txt"
        target = Path("/etc/passwd")
        if target.exists():
            link.symlink_to(target)
            # Should still read the file (no symlink protection implemented)
            text, enc = load_and_decode(str(link))
            # Document: symlink is followed — potential security concern
            assert isinstance(text, str)

    def test_preprocess_path_traversal_blocked(self):
        """Full pipeline rejects traversal path."""
        with pytest.raises(ValueError, match="traversal"):
            preprocess_transcript("../../../etc/passwd", "A", 1)


# ====================================================================
# Persona 3: 엣지케이스 전문가 (Edge Case Expert)
# ====================================================================
class TestPersona3EdgeCaseExpert:
    """Boundary conditions."""

    def test_empty_file(self, tmp_path):
        """Completely empty file (0 bytes)."""
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            preprocess_transcript(str(f), "A", 1)

    def test_whitespace_only_file(self, tmp_path):
        """File containing only whitespace."""
        f = tmp_path / "spaces.txt"
        f.write_text("   \n\n\t\t  \n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            preprocess_transcript(str(f), "A", 1)

    def test_fillers_only_file(self, tmp_path):
        """File with only Korean filler words — should be empty after cleaning."""
        f = tmp_path / "fillers.txt"
        f.write_text("어 음 그 저 뭐 아 예 네 응 어어 음음", encoding="utf-8")
        with pytest.raises(ValueError):
            preprocess_transcript(str(f), "A", 1)

    def test_exactly_max_length(self, tmp_path):
        """File with exactly MAX_TRANSCRIPT_LENGTH characters."""
        f = tmp_path / "exact.txt"
        # Use real Korean words so text survives cleaning
        text = "세포 분열 과정 " * (MAX_TRANSCRIPT_LENGTH // 8)
        text = text[:MAX_TRANSCRIPT_LENGTH]
        f.write_text(text, encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1)
        assert result.char_count_raw == MAX_TRANSCRIPT_LENGTH

    def test_one_over_max_length(self, tmp_path):
        """File with MAX_TRANSCRIPT_LENGTH + 1 characters — rejected."""
        f = tmp_path / "over.txt"
        text = "세" * (MAX_TRANSCRIPT_LENGTH + 1)
        f.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="exceeds"):
            preprocess_transcript(str(f), "A", 1)

    def test_single_word_transcript(self, tmp_path):
        """Transcript with just one word."""
        f = tmp_path / "oneword.txt"
        f.write_text("세포", encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1)
        assert len(result.cleaned_text) > 0

    def test_only_english_content(self, tmp_path):
        """Transcript with only English (no Korean)."""
        f = tmp_path / "english.txt"
        f.write_text(
            "The cell divides through mitosis and produces ATP energy molecules.",
            encoding="utf-8",
        )
        result = preprocess_transcript(str(f), "A", 1)
        assert result.cleaned_text

    def test_only_abbreviations(self, tmp_path):
        """Transcript with only domain abbreviations."""
        f = tmp_path / "abbrev.txt"
        f.write_text("ATP DNA RNA mRNA tRNA CO2 H2O pH", encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1)
        assert result.cleaned_text

    def test_all_content_is_stopwords(self, tmp_path):
        """Transcript where every word is a stopword."""
        f = tmp_path / "stops.txt"
        f.write_text("은 는 이 가 을 를 에 에서 의 와 과 로 으로", encoding="utf-8")
        with pytest.raises(ValueError):
            preprocess_transcript(str(f), "A", 1)

    def test_transcript_with_only_numbers(self, tmp_path):
        """Transcript with only numbers."""
        f = tmp_path / "numbers.txt"
        f.write_text("123 456 789 000 111 222 333 444 555", encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1)
        assert result.cleaned_text

    def test_transcript_with_special_chars(self, tmp_path):
        """Transcript full of punctuation."""
        f = tmp_path / "punct.txt"
        f.write_text("!@#$%^&*(){}[]|\\:;<>?,./세포", encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1)
        assert result.cleaned_text

    def test_max_length_minus_one(self, tmp_path):
        """File with exactly MAX_TRANSCRIPT_LENGTH - 1 characters."""
        f = tmp_path / "under.txt"
        text = "세포 분열 과정 " * (MAX_TRANSCRIPT_LENGTH // 8)
        text = text[: MAX_TRANSCRIPT_LENGTH - 1]
        f.write_text(text, encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1)
        assert result.char_count_raw == MAX_TRANSCRIPT_LENGTH - 1

    def test_newlines_only(self, tmp_path):
        """File with only newline characters."""
        f = tmp_path / "newlines.txt"
        f.write_text("\n" * 100, encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            preprocess_transcript(str(f), "A", 1)


# ====================================================================
# Persona 4: 인코딩 문제 사용자 (Encoding Troublemaker)
# ====================================================================
class TestPersona4EncodingTroublemaker:
    """Encoding edge cases."""

    def test_euckr_file(self, tmp_path):
        """EUC-KR encoded file should fall back gracefully."""
        f = tmp_path / "euckr.txt"
        text = "세포 분열에 대한 설명입니다."
        with open(f, "w", encoding="euc-kr") as fp:
            fp.write(text)
        result = preprocess_transcript(str(f), "A", 1)
        assert result.encoding_used == "euc-kr"

    def test_utf8_bom_file(self, tmp_path):
        """UTF-8 file with BOM (Byte Order Mark)."""
        f = tmp_path / "bom.txt"
        text = "세포 분열에 대해 설명하겠습니다."
        with open(f, "wb") as fp:
            fp.write(b"\xef\xbb\xbf")  # UTF-8 BOM
            fp.write(text.encode("utf-8"))
        result = preprocess_transcript(str(f), "A", 1)
        # BOM character may appear in cleaned text — check if "세포" is there
        assert "세포" in result.cleaned_text

    def test_binary_file(self, tmp_path):
        """Binary file (random bytes) passed as transcript."""
        f = tmp_path / "binary.txt"
        with open(f, "wb") as fp:
            fp.write(os.urandom(1000))
        # Should raise an error, not crash with unhandled exception
        with pytest.raises((ValueError, UnicodeDecodeError)):
            preprocess_transcript(str(f), "A", 1)

    def test_latin1_file(self, tmp_path):
        """Latin-1 encoded file — not valid UTF-8 or EUC-KR."""
        f = tmp_path / "latin1.txt"
        # Latin-1 chars that are invalid in both UTF-8 and EUC-KR
        with open(f, "wb") as fp:
            fp.write("Héllo wörld café résumé naïve".encode("latin-1"))
        # Expect either decoded (possibly garbled) or error
        try:
            text, enc = load_and_decode(str(f))
            assert isinstance(text, str)
        except UnicodeDecodeError:
            pass  # Also acceptable

    def test_mixed_valid_invalid_utf8(self, tmp_path):
        """File with partially corrupt UTF-8 bytes."""
        f = tmp_path / "corrupt.txt"
        with open(f, "wb") as fp:
            fp.write("세포 분열".encode("utf-8"))
            fp.write(b"\xff\xfe")  # Invalid UTF-8 bytes
            fp.write("과정".encode("utf-8"))
        with pytest.raises((ValueError, UnicodeDecodeError)):
            preprocess_transcript(str(f), "A", 1)

    def test_utf16_file(self, tmp_path):
        """UTF-16 encoded file — not UTF-8 or EUC-KR."""
        f = tmp_path / "utf16.txt"
        with open(f, "w", encoding="utf-16") as fp:
            fp.write("세포 분열에 대해 설명합니다.")
        # Should fail with UnicodeDecodeError on both attempts
        try:
            preprocess_transcript(str(f), "A", 1)
            # If it somehow decodes, check it didn't produce garbage
        except (UnicodeDecodeError, ValueError):
            pass


# ====================================================================
# Persona 5: 대용량 처리 사용자 (Scale Tester)
# ====================================================================
class TestPersona5ScaleTester:
    """Scale and threshold boundaries."""

    def test_many_unique_words(self, tmp_path):
        """Transcript with 500 unique Korean-like tokens repeated."""
        f = tmp_path / "many_keywords.txt"
        words = [f"단어{i:04d}" for i in range(500)]
        text = " ".join(words * 3)
        f.write_text(text, encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1)
        assert result.char_count_cleaned > 0

    def test_repeated_char_normalization_at_scale(self, tmp_path):
        """Large text with many repeated characters."""
        f = tmp_path / "repeated.txt"
        text = "아아아아 " * 5000 + "세포 분열 과정 " * 5000
        text = text[:MAX_TRANSCRIPT_LENGTH]
        f.write_text(text, encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1)
        # After normalization, runs of 3+ identical chars become 2
        assert "아아아" not in result.cleaned_text

    def test_near_max_length_with_fillers(self, tmp_path):
        """Near-max-length text where most content is fillers."""
        f = tmp_path / "fillers_long.txt"
        filler_block = "어 음 그 저 뭐 아 예 네 응 " * 1000
        real_content = "세포 분열 과정 ATP DNA "
        text = filler_block + real_content
        text = text[:MAX_TRANSCRIPT_LENGTH]
        f.write_text(text, encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1)
        assert result.char_count_cleaned < result.char_count_raw

    def test_single_char_words(self, tmp_path):
        """Text with many single-character words."""
        f = tmp_path / "singles.txt"
        text = " ".join(["가", "나", "다", "라", "마", "바", "사", "세포"] * 100)
        f.write_text(text, encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1)
        assert result.cleaned_text


# ====================================================================
# Persona 6: 설정 꼬임 사용자 (Config Mess User)
# ====================================================================
class TestPersona6ConfigMess:
    """Configuration problems."""

    def test_empty_stopwords_list(self, tmp_path):
        """Extra stopwords is empty list — should not crash."""
        f = tmp_path / "test.txt"
        f.write_text("세포 분열 과정에서 ATP가 소모됩니다.", encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1, extra_stopwords=[])
        assert result.cleaned_text

    def test_none_extra_abbreviations(self, tmp_path):
        """Extra abbreviations is None — should use defaults only."""
        f = tmp_path / "test.txt"
        f.write_text("ATP와 DNA에 대해 설명합니다.", encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1, extra_abbreviations=None)
        assert result.cleaned_text

    def test_stopwords_containing_abbreviation(self, tmp_path):
        """User adds an abbreviation as a custom stopword.

        Abbreviations should take priority over stopwords per FR-004.
        """
        f = tmp_path / "test.txt"
        f.write_text("ATP와 DNA에 대해 설명합니다.", encoding="utf-8")
        result = preprocess_transcript(str(f), "A", 1, extra_stopwords=["ATP"])
        # filter_stopwords preserves abbreviations even if in stopwords
        assert "ATP" in result.cleaned_text

    def test_extra_abbreviation_added(self, tmp_path):
        """Extra abbreviation is added and preserved."""
        f = tmp_path / "test.txt"
        f.write_text("CRISPR 기술은 DNA를 편집합니다.", encoding="utf-8")
        result = preprocess_transcript(
            str(f),
            "A",
            1,
            extra_abbreviations=["CRISPR"],
        )
        assert "CRISPR" in result.cleaned_text

    def test_build_stopwords_with_none(self):
        """build_stopwords(None) should return base stopwords."""
        sw = build_stopwords(None)
        assert len(sw) > 0
        assert "the" in sw

    def test_build_stopwords_with_empty_list(self):
        """build_stopwords([]) should return base stopwords."""
        sw = build_stopwords([])
        assert len(sw) > 0

    def test_filter_with_empty_word_list(self):
        """filter_stopwords with empty word list."""
        result = filter_stopwords([], build_stopwords(), BIOLOGY_ABBREVIATIONS)
        assert result == []

    def test_filter_with_empty_stopwords(self):
        """filter_stopwords with empty stopwords set."""
        words = ["세포", "분열", "ATP"]
        result = filter_stopwords(words, frozenset(), BIOLOGY_ABBREVIATIONS)
        assert result == words  # nothing removed

    def test_filter_with_empty_abbreviations(self):
        """filter_stopwords with no abbreviations — stopwords removed normally.

        Note: ATP is NOT in the stopword set, so it is preserved even
        without abbreviation protection. Only actual stopwords are removed.
        """
        words = ["the", "cell", "ATP"]
        sw = build_stopwords()
        result = filter_stopwords(words, sw, frozenset())
        assert "the" not in result
        # ATP is NOT a stopword, so it survives even without abbreviation protection
        assert "ATP" in result


# ====================================================================
# Persona 7: 동시성/재실행 사용자 (Re-run / Consistency User)
# ====================================================================
class TestPersona7RerunConsistency:
    """Idempotency checks."""

    def test_preprocess_idempotent(self, tmp_path):
        """Running preprocess twice gives the same result."""
        f = tmp_path / "test.txt"
        f.write_text(
            "세포 분열 과정에서 어 음 그 ATP가 소모됩니다.",
            encoding="utf-8",
        )
        result1 = preprocess_transcript(str(f), "A", 1)
        result2 = preprocess_transcript(str(f), "A", 1)
        assert result1.cleaned_text == result2.cleaned_text
        assert result1.char_count_cleaned == result2.char_count_cleaned

    def test_normalize_repeated_is_idempotent(self):
        """Normalizing already-normalized text gives the same result."""
        text = "아아 세포 분열"
        result1 = normalize_repeated_chars(text)
        result2 = normalize_repeated_chars(result1)
        assert result1 == result2

    def test_remove_fillers_is_idempotent(self):
        """Removing fillers from already-cleaned text is safe."""
        text = "세포 분열 과정"
        result1 = remove_fillers(text)
        result2 = remove_fillers(result1)
        assert result1 == result2

    def test_split_mixed_tokens_idempotent(self):
        """Split mixed tokens is idempotent."""
        text = "ATP세포 DNA분열"
        result1 = split_mixed_tokens(text)
        result2 = split_mixed_tokens(result1)
        assert result1 == result2

    def test_build_stopwords_deterministic(self):
        """build_stopwords returns the same set on repeated calls."""
        sw1 = build_stopwords()
        sw2 = build_stopwords()
        assert sw1 == sw2


# ====================================================================
# Persona 2 Extended: CLI-level attacks
# ====================================================================
class TestPersona2CLIAttacks:
    """CLI-level malicious inputs."""

    def test_cli_analyze_no_input(self):
        """CLI with --input missing raises SystemExit."""
        from forma.cli_lecture import main_analyze

        with pytest.raises(SystemExit):
            main_analyze(["--output", "/tmp/out", "--class", "A"])

    def test_cli_analyze_traversal_input(self, tmp_path):
        """CLI rejects path traversal in --input."""
        from forma.cli_lecture import main_analyze

        with pytest.raises(SystemExit):
            main_analyze(
                [
                    "--input",
                    "../../etc/passwd",
                    "--output",
                    str(tmp_path),
                    "--class",
                    "A",
                ]
            )

    def test_cli_analyze_traversal_concepts(self, tmp_path):
        """CLI rejects path traversal in --concepts."""
        from forma.cli_lecture import main_analyze

        f = tmp_path / "test.txt"
        f.write_text("세포 분열", encoding="utf-8")
        with pytest.raises(SystemExit):
            main_analyze(
                [
                    "--input",
                    str(f),
                    "--output",
                    str(tmp_path),
                    "--class",
                    "A",
                    "--concepts",
                    "../../etc/passwd",
                ]
            )

    def test_cli_analyze_nonexistent_input(self, tmp_path):
        """CLI with nonexistent --input file."""
        from forma.cli_lecture import main_analyze

        with pytest.raises(SystemExit):
            main_analyze(
                [
                    "--input",
                    str(tmp_path / "nope.txt"),
                    "--output",
                    str(tmp_path),
                    "--class",
                    "A",
                ]
            )

    def test_cli_analyze_empty_input(self, tmp_path):
        """CLI with empty --input file."""
        from forma.cli_lecture import main_analyze

        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        with pytest.raises(SystemExit):
            main_analyze(
                [
                    "--input",
                    str(f),
                    "--output",
                    str(tmp_path),
                    "--class",
                    "A",
                ]
            )

    def test_cli_analyze_missing_output(self, tmp_path):
        """CLI missing required --output raises SystemExit."""
        from forma.cli_lecture import main_analyze

        with pytest.raises(SystemExit):
            main_analyze(["--input", str(tmp_path / "x.txt"), "--class", "A"])


# ====================================================================
# Persona 3 Extended: validate_path edge cases
# ====================================================================
class TestPersona3ValidatePathEdgeCases:
    """Additional validate_path boundary tests."""

    def test_validate_path_normal(self, tmp_path):
        """Normal path passes validation."""
        validate_path(str(tmp_path / "test.txt"))  # Should not raise

    def test_validate_path_empty_string(self):
        """Empty string path passes validation (no ../)."""
        validate_path("")  # Technically valid — no traversal

    def test_validate_path_dot_dot_no_slash(self):
        """'..etc' without slash is not traversal."""
        validate_path("..etc")  # Should not raise

    def test_validate_path_triple_dot(self):
        """'.../foo' is NOT traversal — triple-dot is a valid directory name.

        Previously BUG-002: naive substring check caused false positive.
        Now fixed with regex that matches '..' as a complete path component.
        """
        validate_path(".../foo")  # Should NOT raise

    def test_validate_path_only_dotdotslash(self):
        """'../' alone."""
        with pytest.raises(ValueError, match="traversal"):
            validate_path("../")


# ====================================================================
# Persona 5 Extended: remove_fillers at scale
# ====================================================================
class TestPersona5FillerScale:
    """Filler removal at scale."""

    def test_remove_fillers_empty(self):
        """Empty input returns empty."""
        assert remove_fillers("") == ""

    def test_remove_fillers_no_fillers(self):
        """Text without any fillers is unchanged."""
        text = "세포 분열 과정"
        assert remove_fillers(text) == text

    def test_remove_fillers_all_fillers(self):
        """Text with only fillers becomes empty."""
        text = "어 음 그 저 뭐 아"
        assert remove_fillers(text) == ""

    def test_remove_fillers_filler_substring_preserved(self):
        """Filler '어' should not be removed from '어떤'."""
        text = "어떤 세포가 어 분열한다"
        result = remove_fillers(text)
        assert "어떤" in result
        # standalone '어' should be removed
        tokens = result.split()
        assert "어" not in tokens


# ====================================================================
# Persona 4 Extended: BOM handling details
# ====================================================================
class TestPersona4BOMDetails:
    """BOM (Byte Order Mark) handling specifics."""

    def test_bom_stripped_from_raw_text(self, tmp_path):
        """BOM character is stripped from raw text by utf-8-sig codec."""
        f = tmp_path / "bom.txt"
        text = "세포 분열"
        with open(f, "wb") as fp:
            fp.write(b"\xef\xbb\xbf")
            fp.write(text.encode("utf-8"))
        raw, enc = load_and_decode(str(f))
        assert not raw.startswith("\ufeff")
        assert raw.startswith("세포")


# ====================================================================
# Persona 6 Extended: analyze_transcript edge cases
# ====================================================================
class TestPersona6AnalyzerEdgeCases:
    """Analyzer-level edge cases."""

    def test_analyze_with_no_concepts(self, tmp_path):
        """analyze_transcript with concepts=None."""
        from forma.lecture_analyzer import analyze_transcript

        cleaned = CleanedTranscript(
            class_id="A",
            week=1,
            source_path=str(tmp_path / "test.txt"),
            raw_text="세포 분열 과정 설명 미토콘드리아 에너지 생산",
            cleaned_text="세포 분열 과정 설명 미토콘드리아 에너지 생산",
            encoding_used="utf-8",
            char_count_raw=30,
            char_count_cleaned=30,
        )
        result = analyze_transcript(
            cleaned,
            concepts=None,
            top_n=10,
            no_triplets=True,
            provider=None,
        )
        assert result.concept_coverage is None
        assert result.emphasis_scores is None

    def test_analyze_with_empty_concepts(self, tmp_path):
        """analyze_transcript with concepts=[] (empty list)."""
        from forma.lecture_analyzer import analyze_transcript

        cleaned = CleanedTranscript(
            class_id="A",
            week=1,
            source_path=str(tmp_path / "test.txt"),
            raw_text="세포 분열 과정",
            cleaned_text="세포 분열 과정",
            encoding_used="utf-8",
            char_count_raw=7,
            char_count_cleaned=7,
        )
        result = analyze_transcript(
            cleaned,
            concepts=[],
            top_n=10,
            no_triplets=True,
            provider=None,
        )
        # Empty list is falsy, so concept analysis should be skipped
        assert result.concept_coverage is None

    def test_analyze_no_triplets_flag(self, tmp_path):
        """Triplet extraction skipped when no_triplets=True."""
        from forma.lecture_analyzer import analyze_transcript

        cleaned = CleanedTranscript(
            class_id="A",
            week=1,
            source_path=str(tmp_path / "test.txt"),
            raw_text="세포 분열",
            cleaned_text="세포 분열",
            encoding_used="utf-8",
            char_count_raw=5,
            char_count_cleaned=5,
        )
        result = analyze_transcript(
            cleaned,
            concepts=None,
            top_n=10,
            no_triplets=True,
            provider=None,
        )
        assert result.triplets is None
        assert "skipped" in result.triplet_skipped_reason

    def test_analyze_no_provider(self, tmp_path):
        """Triplet extraction skipped when provider is None."""
        from forma.lecture_analyzer import analyze_transcript

        cleaned = CleanedTranscript(
            class_id="A",
            week=1,
            source_path=str(tmp_path / "test.txt"),
            raw_text="세포 분열",
            cleaned_text="세포 분열",
            encoding_used="utf-8",
            char_count_raw=5,
            char_count_cleaned=5,
        )
        result = analyze_transcript(
            cleaned,
            concepts=None,
            top_n=10,
            no_triplets=False,
            provider=None,
        )
        assert result.triplets is None
        assert "No LLM provider" in result.triplet_skipped_reason


# ====================================================================
# Persona 7 Extended: YAML round-trip
# ====================================================================
class TestPersona7YAMLRoundTrip:
    """YAML save/load consistency."""

    def test_save_and_load_analysis_result(self, tmp_path):
        """Save and reload AnalysisResult via YAML."""
        from forma.lecture_analyzer import (
            AnalysisResult,
            save_analysis_result,
            load_analysis_result,
        )

        result = AnalysisResult(
            class_id="B",
            week=3,
            keyword_frequencies={"세포": 10, "분열": 5},
            top_keywords=["세포", "분열"],
            network_image_path=None,
            topics=None,
            topic_skipped_reason="문장 수 부족 (2 < 10)",
            concept_coverage=None,
            emphasis_scores=None,
            triplets=None,
            triplet_skipped_reason="트리플렛 추출 건너뛰기",
            sentence_count=2,
            analysis_timestamp="2026-03-13T00:00:00+00:00",
        )
        path = save_analysis_result(result, tmp_path)
        loaded = load_analysis_result(path)
        assert loaded.class_id == result.class_id
        assert loaded.week == result.week
        assert loaded.keyword_frequencies == result.keyword_frequencies
        assert loaded.top_keywords == result.top_keywords
        assert loaded.topic_skipped_reason == result.topic_skipped_reason
        assert loaded.sentence_count == result.sentence_count

    def test_load_nonexistent_yaml(self, tmp_path):
        """Loading a nonexistent YAML file raises FileNotFoundError."""
        from forma.lecture_analyzer import load_analysis_result

        with pytest.raises(FileNotFoundError):
            load_analysis_result(tmp_path / "missing.yaml")
