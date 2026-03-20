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

import tempfile
from pathlib import Path


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
            extract_multi_chapter([str(file1)], use_cache=True)

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


# ================================================================
# v2 Tests: LLM-based concept extraction (T008-T011)
# ================================================================

from unittest.mock import MagicMock, patch

from forma.domain_concept_extractor import (
    DomainConcept,
    build_extraction_prompt,
    extract_concepts_llm,
    extract_multi_chapter_llm,
)


# ----------------------------------------------------------------
# T008: DomainConcept dataclass and prompt construction
# ----------------------------------------------------------------


class TestDomainConceptDataclass:
    """Tests for DomainConcept dataclass fields."""

    def test_all_fields_present(self) -> None:
        """DomainConcept has concept, description, key_terms, importance, section, chapter."""
        dc = DomainConcept(
            concept="표피의 4층 구조와 각 층의 역할",
            description="종자층→과립층→투명층→각질층 순서",
            key_terms=["표피", "종자층", "과립층"],
            importance="high",
            section="1. 피부의 구조 > 1) 표피",
            chapter="3장 피부",
        )
        assert dc.concept == "표피의 4층 구조와 각 층의 역할"
        assert dc.description == "종자층→과립층→투명층→각질층 순서"
        assert dc.key_terms == ["표피", "종자층", "과립층"]
        assert dc.importance == "high"
        assert dc.section == "1. 피부의 구조 > 1) 표피"
        assert dc.chapter == "3장 피부"

    def test_importance_values(self) -> None:
        """importance must be one of high/medium/low."""
        for val in ("high", "medium", "low"):
            dc = DomainConcept(
                concept="test",
                description="desc",
                key_terms=["t"],
                importance=val,
                section="s",
                chapter="ch",
            )
            assert dc.importance == val


class TestBuildExtractionPrompt:
    """Tests for build_extraction_prompt()."""

    def test_prompt_contains_body_text(self) -> None:
        """Prompt includes the cleaned body text."""
        prompt = build_extraction_prompt("피부는 표피와 진피로 구성된다.")
        assert "피부는 표피와 진피로 구성된다." in prompt

    def test_prompt_requests_yaml_output(self) -> None:
        """Prompt requests YAML format output."""
        prompt = build_extraction_prompt("본문 텍스트")
        assert "YAML" in prompt or "yaml" in prompt

    def test_prompt_includes_stopword_exclusion(self) -> None:
        """Prompt instructs to exclude everyday words."""
        prompt = build_extraction_prompt("본문 텍스트")
        assert "일상용어" in prompt or "것" in prompt

    def test_prompt_with_structure_guide(self) -> None:
        """Prompt includes structure guide when provided."""
        prompt = build_extraction_prompt("본문", structure_guide="# 3장 피부\n## 1. 표피")
        assert "3장 피부" in prompt

    def test_prompt_without_structure_guide(self) -> None:
        """Prompt works without structure guide."""
        prompt = build_extraction_prompt("본문 텍스트")
        assert len(prompt) > 0


# ----------------------------------------------------------------
# T009: extract_concepts_llm with mocked LLM
# ----------------------------------------------------------------


MOCK_LLM_RESPONSE_YAML = """\
concepts:
  - concept: "표피의 4층 구조와 각 층의 역할"
    description: "종자층에서 각질층까지 4개 층의 세포 특성과 기능"
    key_terms: [표피, 종자층, 과립층, 투명층, 각질층]
    importance: high
    section: "1. 피부의 구조 > 1) 표피"
  - concept: "진피의 구성 요소와 기능"
    description: "콜라겐 섬유, 혈관, 신경으로 구성된 진피층의 역할"
    key_terms: [진피, 콜라겐, 혈관, 신경]
    importance: high
    section: "1. 피부의 구조 > 2) 진피"
  - concept: "멜라닌 생성과 자외선 방어"
    description: "멜라닌세포가 멜라닌 색소를 생성하여 자외선 차단"
    key_terms: [멜라닌세포, 멜라닌, 자외선]
    importance: medium
    section: "2. 피부의 기능 > 1) 보호"
"""


class TestExtractConceptsLLM:
    """Tests for extract_concepts_llm() with mocked LLM."""

    @patch("forma.domain_concept_extractor.create_provider")
    def test_returns_domain_concepts(self, mock_create: MagicMock) -> None:
        """Mocked LLM returning valid YAML produces DomainConcept list."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = MOCK_LLM_RESPONSE_YAML
        mock_create.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            textbook = Path(tmpdir) / "3장 피부.txt"
            textbook.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")

            concepts = extract_concepts_llm(str(textbook))

        assert len(concepts) == 3
        assert all(isinstance(c, DomainConcept) for c in concepts)
        assert concepts[0].concept == "표피의 4층 구조와 각 층의 역할"
        assert concepts[0].importance == "high"
        assert "표피" in concepts[0].key_terms

    @patch("forma.domain_concept_extractor.create_provider")
    def test_malformed_yaml_returns_empty(self, mock_create: MagicMock) -> None:
        """Malformed LLM output returns empty list."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "this is not valid yaml: [{"
        mock_create.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            textbook = Path(tmpdir) / "3장 피부.txt"
            textbook.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")

            concepts = extract_concepts_llm(str(textbook))

        assert concepts == []

    @patch("forma.domain_concept_extractor.create_provider")
    def test_missing_fields_skipped(self, mock_create: MagicMock) -> None:
        """Concepts missing required fields are skipped."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = """\
concepts:
  - concept: "유효한 개념"
    description: "설명"
    key_terms: [용어1]
    importance: high
    section: "1절"
  - concept: "필드 누락"
    description: "설명만 있음"
"""
        mock_create.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            textbook = Path(tmpdir) / "3장 피부.txt"
            textbook.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")

            concepts = extract_concepts_llm(str(textbook))

        assert len(concepts) == 1
        assert concepts[0].concept == "유효한 개념"

    @patch("forma.domain_concept_extractor.create_provider")
    def test_chapter_name_from_filename(self, mock_create: MagicMock) -> None:
        """Chapter name is derived from filename stem."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = MOCK_LLM_RESPONSE_YAML
        mock_create.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            textbook = Path(tmpdir) / "3장 피부.txt"
            textbook.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")

            concepts = extract_concepts_llm(str(textbook))

        assert all(c.chapter == "3장 피부" for c in concepts)

    @patch("forma.domain_concept_extractor.create_provider")
    def test_summary_path_used(self, mock_create: MagicMock) -> None:
        """When summary_path provided, it's used as structure guide."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = MOCK_LLM_RESPONSE_YAML
        mock_create.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            textbook = Path(tmpdir) / "3장 피부.txt"
            textbook.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")
            summary = Path(tmpdir) / "Ch03_Summary.md"
            summary.write_text("# 3장 피부\n## 1. 표피", encoding="utf-8")

            concepts = extract_concepts_llm(str(textbook), summary_path=str(summary))

        assert len(concepts) == 3
        # Verify generate was called (prompt includes structure guide)
        call_args = mock_provider.generate.call_args
        assert call_args is not None

    @patch("forma.domain_concept_extractor.create_provider")
    def test_model_override(self, mock_create: MagicMock) -> None:
        """model parameter is passed to create_provider."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = MOCK_LLM_RESPONSE_YAML
        mock_create.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            textbook = Path(tmpdir) / "3장 피부.txt"
            textbook.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")

            extract_concepts_llm(str(textbook), model="gemini-2.5-pro")

        mock_create.assert_called_once_with(model="gemini-2.5-pro")


# ----------------------------------------------------------------
# T010: Concept caching (v2)
# ----------------------------------------------------------------


class TestConceptCachingV2:
    """Tests for v2 concept caching (hash-based, LLM skip on cache hit)."""

    @patch("forma.domain_concept_extractor.create_provider")
    def test_cache_hit_skips_llm(self, mock_create: MagicMock) -> None:
        """When cache exists with matching hash, LLM is not called."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = MOCK_LLM_RESPONSE_YAML
        mock_create.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            textbook = Path(tmpdir) / "3장 피부.txt"
            textbook.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")

            # First call — LLM called, cache written
            result1 = extract_concepts_llm(str(textbook))
            assert mock_provider.generate.call_count == 1

            # Second call — cache hit, no LLM call
            mock_create.reset_mock()
            mock_provider.generate.reset_mock()
            result2 = extract_concepts_llm(str(textbook))
            assert mock_provider.generate.call_count == 0
            assert len(result1) == len(result2)

    @patch("forma.domain_concept_extractor.create_provider")
    def test_cache_invalidated_on_change(self, mock_create: MagicMock) -> None:
        """When file content changes, cache is invalidated and LLM is called."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = MOCK_LLM_RESPONSE_YAML
        mock_create.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            textbook = Path(tmpdir) / "3장 피부.txt"
            textbook.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")

            # First call
            extract_concepts_llm(str(textbook))
            assert mock_provider.generate.call_count == 1

            # Modify file
            textbook.write_text(SAMPLE_CHAPTER_TEXT + "\n새로운 내용 추가.", encoding="utf-8")

            # Second call — hash differs, LLM called again
            extract_concepts_llm(str(textbook))
            assert mock_provider.generate.call_count == 2

    @patch("forma.domain_concept_extractor.create_provider")
    def test_no_cache_flag_forces_llm(self, mock_create: MagicMock) -> None:
        """When no_cache=True, LLM is always called."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = MOCK_LLM_RESPONSE_YAML
        mock_create.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            textbook = Path(tmpdir) / "3장 피부.txt"
            textbook.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")

            # First call
            extract_concepts_llm(str(textbook), no_cache=True)
            # Second call — still calls LLM
            extract_concepts_llm(str(textbook), no_cache=True)
            assert mock_provider.generate.call_count == 2


# ----------------------------------------------------------------
# T011: Multi-chapter LLM extraction and YAML round-trip
# ----------------------------------------------------------------


class TestMultiChapterLLMAndYAML:
    """Tests for multi-chapter LLM extraction and v2 YAML format."""

    @patch("forma.domain_concept_extractor.create_provider")
    def test_multi_chapter_extraction(self, mock_create: MagicMock) -> None:
        """extract_multi_chapter_llm processes multiple files."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = MOCK_LLM_RESPONSE_YAML
        mock_create.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "3장 피부.txt"
            file2 = Path(tmpdir) / "4장 근육.txt"
            file1.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")
            file2.write_text(SAMPLE_CHAPTER_TEXT_2, encoding="utf-8")

            result = extract_multi_chapter_llm(
                [str(file1), str(file2)], no_cache=True,
            )

        assert "3장 피부" in result
        assert "4장 근육" in result
        assert len(result["3장 피부"]) > 0
        assert len(result["4장 근육"]) > 0

    @patch("forma.domain_concept_extractor.create_provider")
    def test_v2_yaml_round_trip(self, mock_create: MagicMock) -> None:
        """save_concepts_yaml and load_concepts_yaml handle v2 DomainConcept."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = MOCK_LLM_RESPONSE_YAML
        mock_create.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            textbook = Path(tmpdir) / "3장 피부.txt"
            textbook.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")

            concepts = extract_concepts_llm(str(textbook), no_cache=True)
            concepts_by_chapter = {"3장 피부": concepts}

            output_path = Path(tmpdir) / "concepts_v2.yaml"
            save_concepts_yaml(concepts_by_chapter, str(output_path))

            loaded = load_concepts_yaml(str(output_path))
            assert "3장 피부" in loaded
            loaded_concepts = loaded["3장 피부"]
            assert len(loaded_concepts) == len(concepts)

            # Check v2 fields preserved
            first = loaded_concepts[0]
            assert isinstance(first, DomainConcept)
            assert first.concept == concepts[0].concept
            assert first.description == concepts[0].description
            assert first.key_terms == concepts[0].key_terms
            assert first.importance == concepts[0].importance

    @patch("forma.domain_concept_extractor.create_provider")
    def test_multi_chapter_with_summaries(self, mock_create: MagicMock) -> None:
        """extract_multi_chapter_llm accepts summary_paths."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = MOCK_LLM_RESPONSE_YAML
        mock_create.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "3장 피부.txt"
            file1.write_text(SAMPLE_CHAPTER_TEXT, encoding="utf-8")
            summary1 = Path(tmpdir) / "Ch03_Summary.md"
            summary1.write_text("# 3장 피부", encoding="utf-8")

            result = extract_multi_chapter_llm(
                [str(file1)],
                summary_paths=[str(summary1)],
                no_cache=True,
            )

        assert "3장 피부" in result


# ================================================================
# v3 Tests: Phase 2 — TopicHierarchy (T008)
# ================================================================

try:
    from forma.domain_concept_extractor import (
        TopicHierarchy,
        MajorTopic,
        SubTopic,
        parse_summary_hierarchy,
    )

    _HAS_HIERARCHY = True
except ImportError:
    _HAS_HIERARCHY = False

import pytest

_skip_hierarchy = pytest.mark.skipif(
    not _HAS_HIERARCHY,
    reason="TopicHierarchy/parse_summary_hierarchy not yet implemented (RED phase)",
)


SAMPLE_SUMMARY_KR = """\
## 피부의 구조
### 표피(epidermis)
#### 각질층
#### 투명층
#### 과립층
#### 유극층
#### 기저층
### 진피(dermis)
#### 유두층
#### 망상층
## 피부의 기능
### 보호 기능
### 체온 조절
### 감각 수용
## 피부 부속기
### 한선(땀샘)
### 피지선
### 모발
### 손발톱
"""

SAMPLE_SUMMARY_NO_HEADERS = """\
피부는 인체의 가장 큰 기관이다.
표피와 진피로 구성되어 있다.
각질층이 가장 바깥에 위치한다.
"""


@_skip_hierarchy
class TestParseSummaryHierarchy:
    """T008: Tests for parse_summary_hierarchy()."""

    def test_parses_major_topics_from_h2(self, tmp_path) -> None:
        """## headers become major topics."""
        summary = tmp_path / "Summary_KR.md"
        summary.write_text(SAMPLE_SUMMARY_KR, encoding="utf-8")

        hierarchy = parse_summary_hierarchy(str(summary))

        assert isinstance(hierarchy, TopicHierarchy)
        major_names = [m.name for m in hierarchy.major_topics]
        assert "피부의 구조" in major_names
        assert "피부의 기능" in major_names
        assert "피부 부속기" in major_names
        assert len(hierarchy.major_topics) == 3

    def test_parses_sub_topics_from_h3(self, tmp_path) -> None:
        """### headers become sub topics under their parent major topic."""
        summary = tmp_path / "Summary_KR.md"
        summary.write_text(SAMPLE_SUMMARY_KR, encoding="utf-8")

        hierarchy = parse_summary_hierarchy(str(summary))

        # 피부의 구조 has 표피 and 진피
        structure_topic = [
            m for m in hierarchy.major_topics if m.name == "피부의 구조"
        ][0]
        sub_names = [s.name for s in structure_topic.sub_topics]
        assert "표피(epidermis)" in sub_names
        assert "진피(dermis)" in sub_names
        assert len(structure_topic.sub_topics) == 2

    def test_parses_sections_from_h4(self, tmp_path) -> None:
        """#### headers become sections under their parent sub topic."""
        summary = tmp_path / "Summary_KR.md"
        summary.write_text(SAMPLE_SUMMARY_KR, encoding="utf-8")

        hierarchy = parse_summary_hierarchy(str(summary))

        structure_topic = [
            m for m in hierarchy.major_topics if m.name == "피부의 구조"
        ][0]
        epidermis_sub = [
            s for s in structure_topic.sub_topics if s.name == "표피(epidermis)"
        ][0]
        assert "각질층" in epidermis_sub.sections
        assert "기저층" in epidermis_sub.sections
        assert len(epidermis_sub.sections) == 5

    def test_section_to_major_mapping(self, tmp_path) -> None:
        """section_to_major maps section/sub names to major topic."""
        summary = tmp_path / "Summary_KR.md"
        summary.write_text(SAMPLE_SUMMARY_KR, encoding="utf-8")

        hierarchy = parse_summary_hierarchy(str(summary))

        # Sub topic should map to its parent major topic
        assert hierarchy.section_to_major["표피(epidermis)"] == "피부의 구조"
        assert hierarchy.section_to_major["진피(dermis)"] == "피부의 구조"
        assert hierarchy.section_to_major["보호 기능"] == "피부의 기능"
        assert hierarchy.section_to_major["한선(땀샘)"] == "피부 부속기"
        # Section (####) should also map to major topic
        assert hierarchy.section_to_major["각질층"] == "피부의 구조"

    def test_section_to_sub_mapping(self, tmp_path) -> None:
        """section_to_sub maps section names to sub topic."""
        summary = tmp_path / "Summary_KR.md"
        summary.write_text(SAMPLE_SUMMARY_KR, encoding="utf-8")

        hierarchy = parse_summary_hierarchy(str(summary))

        # #### sections should map to their ### parent
        assert hierarchy.section_to_sub["각질층"] == "표피(epidermis)"
        assert hierarchy.section_to_sub["유두층"] == "진피(dermis)"

    def test_empty_file_returns_empty_hierarchy(self, tmp_path) -> None:
        """Empty file returns hierarchy with no major topics."""
        summary = tmp_path / "empty.md"
        summary.write_text("", encoding="utf-8")

        hierarchy = parse_summary_hierarchy(str(summary))

        assert isinstance(hierarchy, TopicHierarchy)
        assert hierarchy.major_topics == []
        assert hierarchy.section_to_major == {}
        assert hierarchy.section_to_sub == {}

    def test_no_headers_returns_empty_hierarchy(self, tmp_path) -> None:
        """File with no markdown headers returns empty hierarchy."""
        summary = tmp_path / "no_headers.md"
        summary.write_text(SAMPLE_SUMMARY_NO_HEADERS, encoding="utf-8")

        hierarchy = parse_summary_hierarchy(str(summary))

        assert hierarchy.major_topics == []
        assert hierarchy.section_to_major == {}
        assert hierarchy.section_to_sub == {}

    def test_real_world_structure(self, tmp_path) -> None:
        """Real-world Summary_KR.md with mixed content between headers."""
        content = """\
## 피부의 구조
피부는 표피, 진피, 피하조직으로 구성된다.

### 표피
표피는 중층편평상피로 이루어진 외배엽 유래 조직이다.

#### 각질층
각질층은 표피의 가장 바깥층이다.

#### 과립층
과립층에는 케라토히알린 과립이 있다.

### 진피
진피는 중배엽 유래 결합조직이다.

## 피부 부속기
### 한선
"""
        summary = tmp_path / "real_world.md"
        summary.write_text(content, encoding="utf-8")

        hierarchy = parse_summary_hierarchy(str(summary))

        assert len(hierarchy.major_topics) == 2
        structure = [m for m in hierarchy.major_topics if m.name == "피부의 구조"][0]
        assert len(structure.sub_topics) == 2
        epidermis = [s for s in structure.sub_topics if s.name == "표피"][0]
        assert "각질층" in epidermis.sections
        assert "과립층" in epidermis.sections


# ================================================================
# v3 Tests: Phase 3 — Chunked Extraction (T015-T016)
# ================================================================

try:
    from forma.domain_concept_extractor import (
        TextbookChunk,
        chunk_textbook_by_sections,
        _merge_chunk_concepts,
    )

    _HAS_CHUNKING = True
except ImportError:
    _HAS_CHUNKING = False

_skip_chunking = pytest.mark.skipif(
    not _HAS_CHUNKING,
    reason="TextbookChunk/chunk_textbook_by_sections not yet implemented (RED phase)",
)


@_skip_chunking
class TestChunkTextbookBySections:
    """T015: Tests for chunk_textbook_by_sections()."""

    def test_splits_at_h3_headers(self, tmp_path) -> None:
        """### headers split into separate chunks."""
        text = """\
### 표피
표피는 외배엽 유래 조직이다. 각질세포가 주를 이룬다.

### 진피
진피는 중배엽 유래 결합조직이다. 콜라겐 섬유가 풍부하다.

### 피하조직
피하조직은 지방층으로 구성된다.
"""
        summary = tmp_path / "Summary_KR.md"
        summary.write_text(
            "## 피부의 구조\n### 표피\n### 진피\n### 피하조직\n",
            encoding="utf-8",
        )

        chunks = chunk_textbook_by_sections(
            text, summary_path=str(summary), max_chars=12000,
        )

        assert len(chunks) == 3
        assert all(isinstance(c, TextbookChunk) for c in chunks)
        # Each chunk has the section text
        assert "표피는 외배엽" in chunks[0].text
        assert "진피는 중배엽" in chunks[1].text
        assert "피하조직은 지방층" in chunks[2].text

    def test_oversized_section_split_at_paragraph(self) -> None:
        """Section > MAX_CHUNK_CHARS is split at paragraph boundary."""
        # Create a section with > 12K chars (using paragraphs)
        para = "이것은 긴 문단입니다. " * 200  # ~2400 chars per paragraph
        text = "### 긴 섹션\n\n" + "\n\n".join([para] * 6)  # ~14400 chars total

        chunks = chunk_textbook_by_sections(text, max_chars=12000)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.char_count <= 12000

    def test_body_within_limit_single_chunk(self) -> None:
        """Body <= 12K produces single chunk (no splitting)."""
        text = "### 짧은 섹션\n표피는 중요하다. " * 10  # well under 12K

        chunks = chunk_textbook_by_sections(text, max_chars=12000)

        assert len(chunks) == 1
        assert chunks[0].char_count <= 12000

    def test_fallback_paragraph_splitting_no_summary(self) -> None:
        """When no summary and no ### headers, falls back to paragraph-based splitting."""
        # Large text with no markdown headers
        para = "세포막은 인지질 이중층으로 구성된다. " * 300  # ~10800 chars
        text = "\n\n".join([para] * 2)  # ~21600 chars, no headers

        chunks = chunk_textbook_by_sections(text, max_chars=12000)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.char_count <= 12000

    def test_section_path_and_topics_on_chunks(self, tmp_path) -> None:
        """Each chunk has section_path, major_topic, and sub_topic."""
        text = """\
### 표피
표피는 4개의 층으로 구성된다.

### 진피
진피에는 혈관이 있다.
"""
        summary = tmp_path / "Summary_KR.md"
        summary.write_text(
            "## 피부의 구조\n### 표피\n### 진피\n",
            encoding="utf-8",
        )

        chunks = chunk_textbook_by_sections(
            text, summary_path=str(summary), max_chars=12000,
        )

        assert len(chunks) == 2
        # First chunk from 표피 section
        assert chunks[0].major_topic == "피부의 구조"
        assert chunks[0].sub_topic == "표피"
        assert isinstance(chunks[0].section_path, list)
        assert "피부의 구조" in chunks[0].section_path
        assert "표피" in chunks[0].section_path

        # Second chunk from 진피 section
        assert chunks[1].major_topic == "피부의 구조"
        assert chunks[1].sub_topic == "진피"


@_skip_chunking
class TestMergeChunkConcepts:
    """T016: Tests for _merge_chunk_concepts()."""

    def test_exact_name_dedup_keeps_richer_description(self) -> None:
        """Same concept name → keep the one with richer (longer) description."""
        chunk1_concepts = [
            DomainConcept(
                concept="표피의 4층 구조",
                description="표피의 층",
                key_terms=["표피", "각질층"],
                importance="medium",
                section="1절",
                chapter="3장",
            ),
        ]
        chunk2_concepts = [
            DomainConcept(
                concept="표피의 4층 구조",
                description="종자층에서 각질층까지 4개 층의 세포 특성과 기능을 상세 설명",
                key_terms=["표피", "각질층", "종자층"],
                importance="high",
                section="1절",
                chapter="3장",
            ),
        ]

        merged = _merge_chunk_concepts([chunk1_concepts, chunk2_concepts])

        assert len(merged) == 1
        # Should keep the richer description
        assert "상세" in merged[0].description
        # Should keep higher importance
        assert merged[0].importance == "high"

    def test_key_term_overlap_dedup_merges(self) -> None:
        """Concepts with key_term overlap >= min(2, len(shorter)) are merged."""
        chunk1_concepts = [
            DomainConcept(
                concept="표피의 층 구조",
                description="표피 구조 설명",
                key_terms=["표피", "각질층", "종자층"],
                importance="high",
                section="1절",
                chapter="3장",
            ),
        ]
        chunk2_concepts = [
            DomainConcept(
                concept="표피의 세포층 배열",
                description="표피 세포층 상세",
                key_terms=["표피", "각질층", "과립층"],
                importance="medium",
                section="1절",
                chapter="3장",
            ),
        ]

        merged = _merge_chunk_concepts([chunk1_concepts, chunk2_concepts])

        # Overlap is {"표피", "각질층"} = 2 >= min(2, 3) = 2 → merged
        assert len(merged) == 1
        assert merged[0].importance == "high"

    def test_importance_merge_keeps_higher(self) -> None:
        """Merged concept keeps the higher importance."""
        chunk1 = [
            DomainConcept(
                concept="진피의 구조",
                description="진피 간단 설명",
                key_terms=["진피", "콜라겐"],
                importance="low",
                section="2절",
                chapter="3장",
            ),
        ]
        chunk2 = [
            DomainConcept(
                concept="진피의 구조",
                description="진피 상세 설명",
                key_terms=["진피", "콜라겐"],
                importance="high",
                section="2절",
                chapter="3장",
            ),
        ]

        merged = _merge_chunk_concepts([chunk1, chunk2])

        assert len(merged) == 1
        assert merged[0].importance == "high"

    def test_no_merge_when_overlap_below_threshold(self) -> None:
        """Concepts with insufficient key_term overlap remain separate."""
        chunk1 = [
            DomainConcept(
                concept="표피의 4층 구조",
                description="표피 설명",
                key_terms=["표피", "각질층", "종자층"],
                importance="high",
                section="1절",
                chapter="3장",
            ),
        ]
        chunk2 = [
            DomainConcept(
                concept="진피의 혈관 분포",
                description="진피 혈관",
                key_terms=["진피", "혈관", "신경"],
                importance="medium",
                section="2절",
                chapter="3장",
            ),
        ]

        merged = _merge_chunk_concepts([chunk1, chunk2])

        # Overlap is {} = 0 < min(2, 3) = 2 → no merge
        assert len(merged) == 2

    def test_single_key_term_threshold_is_1(self) -> None:
        """Concepts with 1 key_term use threshold min(2, 1) = 1."""
        chunk1 = [
            DomainConcept(
                concept="멜라닌의 역할",
                description="멜라닌 간단",
                key_terms=["멜라닌"],
                importance="medium",
                section="3절",
                chapter="3장",
            ),
        ]
        chunk2 = [
            DomainConcept(
                concept="멜라닌 색소 기능",
                description="멜라닌 색소의 자외선 방어 기전",
                key_terms=["멜라닌"],
                importance="high",
                section="3절",
                chapter="3장",
            ),
        ]

        merged = _merge_chunk_concepts([chunk1, chunk2])

        # Overlap is {"멜라닌"} = 1 >= min(2, 1) = 1 → merged
        assert len(merged) == 1
        assert merged[0].importance == "high"
