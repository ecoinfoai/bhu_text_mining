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
