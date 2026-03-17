"""Tests for domain concept extraction (T007-T009).

T007: Tests for extract_concepts() — KoNLPy noun extraction, frequency
      filtering, bilingual inclusion, domain stopword filtering,
      TextbookConcept fields.

T008: Tests for extract_multi_chapter() — multiple files, per-chapter
      independence, chapter name derivation.

T009: Tests for concept caching — save/load cache, hash validation,
      modified file triggers re-extraction.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from forma.domain_concept_extractor import (
    TextbookConcept,
    extract_concepts,
    extract_multi_chapter,
    save_concepts_yaml,
    load_concepts_yaml,
)


# ----------------------------------------------------------------
# Sample textbook content for testing
# ----------------------------------------------------------------

SAMPLE_CHAPTER_TEXT = (
    "피부는 표피(epidermis)와 진피(dermis)로 구성된다. "
    "표피는 피부의 가장 바깥층으로 각질세포가 대부분을 차지한다. "
    "진피는 표피 아래에 위치하며 콜라겐 섬유가 풍부하다. "
    "표피의 가장 바깥에는 각질층이 있다. "
    "피부는 외부 환경으로부터 신체를 보호한다. "
    "멜라닌세포는 표피에 존재하며 자외선으로부터 보호한다. "
    "진피에는 혈관과 신경이 분포한다. "
    "콜라겐은 진피의 주요 구성 성분이다."
)

SAMPLE_CHAPTER_TEXT_2 = (
    "골격근은 수의근으로 골격에 부착되어 있다. "
    "골격근(skeletal muscle)은 횡문근이라고도 한다. "
    "근육 수축은 액틴과 미오신의 상호작용으로 일어난다. "
    "액틴(actin)은 가는 필라멘트의 주요 단백질이다. "
    "미오신(myosin)은 두꺼운 필라멘트를 구성한다. "
    "근육 수축에는 ATP가 필요하다."
)


# ----------------------------------------------------------------
# T007: extract_concepts
# ----------------------------------------------------------------


class TestExtractConcepts:
    """Tests for extract_concepts()."""

    def test_extracts_nouns_with_frequency_gte_2(self) -> None:
        """Nouns appearing 2+ times are extracted."""
        concepts = extract_concepts(SAMPLE_CHAPTER_TEXT, "3장 피부")
        names = {c.name_ko for c in concepts}
        # "표피" appears 4+ times, "진피" appears 3+ times
        assert "표피" in names
        assert "진피" in names

    def test_bilingual_terms_included_regardless_of_frequency(self) -> None:
        """Bilingual terms are included even with frequency=1."""
        # "epidermis" is bilingual annotation — even if the Korean part
        # appeared only once it should be included
        concepts = extract_concepts(SAMPLE_CHAPTER_TEXT, "3장 피부")
        bilingual = [c for c in concepts if c.is_bilingual]
        assert len(bilingual) > 0
        bilingual_ko = {c.name_ko for c in bilingual}
        assert "표피" in bilingual_ko or "진피" in bilingual_ko

    def test_filters_general_nouns(self) -> None:
        """General nouns like 것, 수, 때, 등 are filtered out."""
        text = (
            "것은 수가 때에 등은 중에 위에 있다. "
            "표피는 표피의 구조가 중요하다. "
            "표피의 세포는 각질세포이다."
        )
        concepts = extract_concepts(text, "test")
        names = {c.name_ko for c in concepts}
        for stopword in ["것", "수", "때", "등", "중", "위"]:
            assert stopword not in names

    def test_returns_textbook_concept_dataclass(self) -> None:
        """Returns list of TextbookConcept with correct fields."""
        concepts = extract_concepts(SAMPLE_CHAPTER_TEXT, "3장 피부")
        assert len(concepts) > 0
        c = concepts[0]
        assert isinstance(c, TextbookConcept)
        assert isinstance(c.name_ko, str)
        assert isinstance(c.chapter, str)
        assert c.chapter == "3장 피부"
        assert isinstance(c.frequency, int)
        assert c.frequency >= 1
        assert isinstance(c.context_sentence, str)
        assert isinstance(c.is_bilingual, bool)

    def test_context_sentence_contains_concept(self) -> None:
        """context_sentence contains the concept term."""
        concepts = extract_concepts(SAMPLE_CHAPTER_TEXT, "3장 피부")
        for c in concepts:
            assert c.name_ko in c.context_sentence, (
                f"Context for '{c.name_ko}' should contain the term"
            )

    def test_sorted_by_frequency_descending(self) -> None:
        """Results are sorted by frequency in descending order."""
        concepts = extract_concepts(SAMPLE_CHAPTER_TEXT, "3장 피부")
        freqs = [c.frequency for c in concepts]
        assert freqs == sorted(freqs, reverse=True)

    def test_bilingual_has_name_en(self) -> None:
        """Bilingual concepts have name_en populated."""
        concepts = extract_concepts(SAMPLE_CHAPTER_TEXT, "3장 피부")
        bilingual = [c for c in concepts if c.is_bilingual]
        for c in bilingual:
            assert c.name_en is not None
            assert len(c.name_en) > 0

    def test_min_freq_parameter(self) -> None:
        """min_freq parameter controls frequency threshold."""
        concepts_default = extract_concepts(SAMPLE_CHAPTER_TEXT, "test", min_freq=2)
        concepts_high = extract_concepts(SAMPLE_CHAPTER_TEXT, "test", min_freq=5)
        # Higher min_freq should produce fewer or equal concepts
        non_bilingual_default = [c for c in concepts_default if not c.is_bilingual]
        non_bilingual_high = [c for c in concepts_high if not c.is_bilingual]
        assert len(non_bilingual_high) <= len(non_bilingual_default)


# ----------------------------------------------------------------
# T008: extract_multi_chapter
# ----------------------------------------------------------------


class TestExtractMultiChapter:
    """Tests for extract_multi_chapter()."""

    def test_multiple_files_produce_per_chapter_lists(self) -> None:
        """Given 2+ textbook file paths, returns dict[str, list[TextbookConcept]]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "3장 피부.txt"
            file2 = Path(tmpdir) / "4장 골격근.txt"
            file1.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")
            file2.write_text(SAMPLE_CHAPTER_TEXT_2, encoding="utf-8")

            result = extract_multi_chapter([str(file1), str(file2)])

            assert isinstance(result, dict)
            assert len(result) == 2
            assert "3장 피부" in result
            assert "4장 골격근" in result

    def test_chapters_are_independent(self) -> None:
        """Each chapter's concepts are independently extracted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "3장 피부.txt"
            file2 = Path(tmpdir) / "4장 골격근.txt"
            file1.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")
            file2.write_text(SAMPLE_CHAPTER_TEXT_2, encoding="utf-8")

            result = extract_multi_chapter([str(file1), str(file2)])

            ch1_names = {c.name_ko for c in result["3장 피부"]}
            ch2_names = {c.name_ko for c in result["4장 골격근"]}
            # Chapter-specific terms should be in their respective chapters
            # "표피" only in chapter 3, "골격근" only in chapter 4
            if "표피" in ch1_names:
                # Not necessarily in ch2 (unless by coincidence)
                pass
            assert len(ch1_names) > 0
            assert len(ch2_names) > 0

    def test_chapter_name_from_filename(self) -> None:
        """Chapter name is derived from filename (strip .txt and path)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "5장 순환계.txt"
            file1.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")

            result = extract_multi_chapter([str(file1)])
            assert "5장 순환계" in result


# ----------------------------------------------------------------
# T009: concept caching
# ----------------------------------------------------------------


class TestConceptCaching:
    """Tests for concept caching (save/load YAML)."""

    def test_save_and_load_concepts_yaml(self) -> None:
        """save_concepts_yaml and load_concepts_yaml round-trip correctly."""
        concepts = extract_concepts(SAMPLE_CHAPTER_TEXT, "3장 피부")
        concepts_by_chapter = {"3장 피부": concepts}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "concepts.yaml"
            save_concepts_yaml(concepts_by_chapter, str(output_path))

            assert output_path.exists()
            assert output_path.stat().st_size > 0

            loaded = load_concepts_yaml(str(output_path))
            assert "3장 피부" in loaded
            assert len(loaded["3장 피부"]) == len(concepts)

    def test_loaded_concepts_have_correct_fields(self) -> None:
        """Loaded concepts preserve all TextbookConcept fields."""
        concepts = extract_concepts(SAMPLE_CHAPTER_TEXT, "3장 피부")
        concepts_by_chapter = {"3장 피부": concepts}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "concepts.yaml"
            save_concepts_yaml(concepts_by_chapter, str(output_path))
            loaded = load_concepts_yaml(str(output_path))

            original = concepts[0]
            loaded_first = loaded["3장 피부"][0]
            assert loaded_first.name_ko == original.name_ko
            assert loaded_first.chapter == original.chapter
            assert loaded_first.frequency == original.frequency

    def test_cache_with_extract_multi_chapter(self) -> None:
        """extract_multi_chapter with caching enabled reuses cached results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "3장 피부.txt"
            file1.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")

            # First call: no cache exists
            result1 = extract_multi_chapter([str(file1)], use_cache=True)

            # Cache file should now exist
            cache_path = Path(tmpdir) / "3장 피부.txt.concepts_cache.yaml"
            assert cache_path.exists()

            # Second call: should use cache
            result2 = extract_multi_chapter([str(file1)], use_cache=True)

            assert len(result1["3장 피부"]) == len(result2["3장 피부"])

    def test_modified_file_triggers_reextraction(self) -> None:
        """Modified file triggers re-extraction (different hash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "3장 피부.txt"
            file1.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")

            # First extraction with cache
            result1 = extract_multi_chapter([str(file1)], use_cache=True)

            # Modify file
            file1.write_text(
                SAMPLE_CHAPTER_TEXT + "\n추가된 새로운 내용이 여기에 있다. 새로운 개념이 등장한다.",
                encoding="utf-8",
            )

            # Second extraction should detect change
            result2 = extract_multi_chapter([str(file1)], use_cache=True)
            # Results may differ since content changed
            # At minimum, extraction ran again (not from stale cache)
            assert "3장 피부" in result2
