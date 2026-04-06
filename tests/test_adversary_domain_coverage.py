"""Adversarial tests for domain coverage feature.

12 hostile personas attack textbook_preprocessor, domain_concept_extractor,
domain_coverage_analyzer, domain_coverage_charts, domain_coverage_report,
and cli_domain modules.
"""

from __future__ import annotations

import io
import os
import stat
import tempfile
from pathlib import Path

import pytest
import yaml

from forma.domain_concept_extractor import (
    TextbookConcept,
    load_concepts_yaml,
    save_concepts_yaml,
)
from forma.domain_coverage_analyzer import (
    ClassifiedConcept,
    ConceptEmphasis,
    ConceptState,
    CoverageResult,
    TeachingScope,
    build_coverage_result,
    classify_concepts,
    load_coverage_yaml,
    parse_scope_string,
    parse_teaching_scope,
    save_coverage_yaml,
)
from forma.textbook_preprocessor import clean_textbook_text, extract_bilingual_terms


# ============================================================
# Helper factories
# ============================================================


def _make_concept(
    name_ko: str = "세포",
    name_en: str | None = "cell",
    chapter: str = "1장",
    frequency: int = 5,
    context: str = "세포는 생명의 기본 단위이다",
    is_bilingual: bool = True,
) -> TextbookConcept:
    return TextbookConcept(
        name_ko=name_ko,
        name_en=name_en,
        chapter=chapter,
        frequency=frequency,
        context_sentence=context,
        is_bilingual=is_bilingual,
    )


def _make_emphasis(
    concept_name: str = "세포",
    chapter: str = "1장",
    section_scores: dict[str, float] | None = None,
    mean_score: float = 0.3,
    std_score: float = 0.1,
) -> ConceptEmphasis:
    if section_scores is None:
        section_scores = {"A": 0.4, "B": 0.2}
    return ConceptEmphasis(
        concept_name=concept_name,
        chapter=chapter,
        section_scores=section_scores,
        mean_score=mean_score,
        std_score=std_score,
    )


def _make_classified(
    name_ko: str = "세포",
    state: ConceptState = ConceptState.COVERED,
    in_scope: bool = True,
    mean_score: float = 0.3,
    std_score: float = 0.1,
    frequency: int = 5,
    chapter: str = "1장",
) -> ClassifiedConcept:
    concept = _make_concept(name_ko=name_ko, frequency=frequency, chapter=chapter)
    emphasis = _make_emphasis(
        concept_name=name_ko,
        chapter=chapter,
        mean_score=mean_score,
        std_score=std_score,
    )
    return ClassifiedConcept(
        concept=concept,
        state=state,
        emphasis=emphasis,
        in_scope=in_scope,
    )


def _make_coverage_result(**kwargs) -> CoverageResult:
    defaults = dict(
        week=1,
        chapters=["1장"],
        total_textbook_concepts=3,
        in_scope_count=2,
        skipped_count=1,
        covered_count=1,
        gap_count=1,
        extra_count=0,
        effective_coverage_rate=0.5,
        per_section_coverage={"A": 0.6, "B": 0.4},
        classified_concepts=[
            _make_classified("세포", ConceptState.COVERED),
            _make_classified("조직", ConceptState.GAP, mean_score=0.01),
            _make_classified("기관", ConceptState.SKIPPED, in_scope=False),
        ],
        extra_concepts=[],
        emphasis_bias_correlation=0.45,
        section_variance_top10=[("세포", 0.1), ("조직", 0.05)],
    )
    defaults.update(kwargs)
    return CoverageResult(**defaults)


# ============================================================
# Persona 1: The Empty Input Terrorist
# ============================================================


class TestEmptyInputTerrorist:
    """Attack with empty textbook file, transcript, concepts YAML, scope."""

    def test_clean_empty_text(self):
        """Empty string should return empty string, not crash."""
        assert clean_textbook_text("") == ""

    def test_clean_whitespace_only(self):
        """Whitespace-only input should not crash."""
        result = clean_textbook_text("   \n\n\n  ")
        assert isinstance(result, str)

    def test_extract_bilingual_empty(self):
        """No bilingual terms from empty text."""
        assert extract_bilingual_terms("") == []

    def test_classify_empty_concepts(self):
        """Classifying zero concepts should return empty list."""
        scope = TeachingScope(chapters=["1장"])
        result = classify_concepts([], [], scope)
        assert result == []

    def test_build_coverage_empty_classified(self):
        """build_coverage_result with empty lists should not crash."""
        result = build_coverage_result([], [])
        assert result.total_textbook_concepts == 0
        assert result.effective_coverage_rate == 0.0
        assert result.per_section_coverage == {}

    def test_load_concepts_yaml_empty_chapters(self, tmp_path):
        """YAML with empty chapters dict should load cleanly."""
        p = tmp_path / "concepts.yaml"
        p.write_text(yaml.dump({"chapters": {}}), encoding="utf-8")
        result = load_concepts_yaml(str(p))
        assert result == {}

    def test_parse_teaching_scope_empty_dict(self):
        """Empty dict → empty scope."""
        scope = parse_teaching_scope({})
        assert scope.chapters == []
        assert scope.scope_rules == {}

    def test_parse_scope_string_empty(self):
        """Empty scope string → empty dict."""
        assert parse_scope_string("") == {}
        assert parse_scope_string("   ") == {}


# ============================================================
# Persona 2: The Unicode Saboteur
# ============================================================


class TestUnicodeSaboteur:
    """Zero-width chars, RTL override, emoji in concept names."""

    def test_zero_width_chars_in_text(self):
        """Zero-width characters should not crash cleaning."""
        text = "세포\u200b는 \u200c생명의 \ufeff기본 단위이다"
        result = clean_textbook_text(text)
        assert isinstance(result, str)

    def test_rtl_override_in_concept_name(self):
        """RTL override char in concept name should not crash classification."""
        concept = _make_concept(name_ko="\u202eolleh세포")
        scope = TeachingScope(chapters=["1장"])
        emphasis = [_make_emphasis(concept_name=concept.name_ko)]
        result = classify_concepts([concept], emphasis, scope)
        assert len(result) == 1

    def test_emoji_in_chapter_name(self):
        """Emoji in chapter name should survive YAML round-trip."""
        concepts = {"3장 피부 \U0001f9ec": [_make_concept(chapter="3장 피부 \U0001f9ec")]}
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            save_concepts_yaml(concepts, f.name)
            loaded = load_concepts_yaml(f.name)
        os.unlink(f.name)
        assert "3장 피부 \U0001f9ec" in loaded

    def test_bilingual_with_unicode_parens(self):
        """Full-width parentheses should not match bilingual pattern."""
        # Full-width parens U+FF08 and U+FF09
        text = "세포\uff08cell\uff09"
        result = extract_bilingual_terms(text)
        # The regex only catches ASCII parens, so no match expected
        assert result == []

    def test_combining_diacritics(self):
        """Combining marks should not crash anything."""
        text = "se\u0301llo(cell)"
        result = extract_bilingual_terms(text)
        # Should handle gracefully even if term is weird
        assert isinstance(result, list)


# ============================================================
# Persona 3: The Giant File Attacker
# ============================================================


class TestGiantFileAttacker:
    """10MB textbook, 1000 concepts."""

    def test_clean_large_text(self):
        """10MB text should clean without OOM."""
        text = "세포는 생명의 기본 단위이다. " * 500_000  # ~15MB
        result = clean_textbook_text(text)
        assert len(result) > 0

    def test_classify_1000_concepts(self):
        """Classify 1000 concepts quickly."""
        concepts = [_make_concept(name_ko=f"개념{i}", chapter="1장") for i in range(1000)]
        emphasis = [_make_emphasis(concept_name=f"개념{i}", mean_score=0.1 * (i % 10)) for i in range(1000)]
        scope = TeachingScope(chapters=["1장"])
        result = classify_concepts(concepts, emphasis, scope)
        assert len(result) == 1000

    def test_build_coverage_1000_concepts(self):
        """build_coverage_result with 1000 classified concepts."""
        classified = [_make_classified(name_ko=f"개념{i}", frequency=i + 1) for i in range(1000)]
        result = build_coverage_result(classified, [])
        assert result.total_textbook_concepts == 1000

    def test_coverage_yaml_roundtrip_large(self, tmp_path):
        """YAML save/load with 500 concepts."""
        classified = [_make_classified(name_ko=f"개념{i}", frequency=i + 1) for i in range(500)]
        cr = build_coverage_result(classified, [])
        out = str(tmp_path / "big.yaml")
        save_coverage_yaml(cr, out)
        loaded = load_coverage_yaml(out)
        assert loaded.total_textbook_concepts == 500


# ============================================================
# Persona 4: The Regex Breaker
# ============================================================


class TestRegexBreaker:
    """Text that looks like page numbers but isn't; parenthetical that isn't bilingual."""

    def test_chapter_line_with_long_title(self):
        """'제 3 장 피부는 중요하다' has multi-word title — should NOT be stripped
        because regex expects single-word title + page number."""
        text = "제 3 장 피부는 중요하다\n본문 시작"
        result = clean_textbook_text(text)
        # The regex pattern is: ^제\s*\d+\s*장\s+\S+\s+\d+\s*$
        # "피부는 중요하다" has spaces, so \S+ only matches first word
        # and then \d+$ won't match "중요하다". So line should SURVIVE.
        assert "피부는 중요하다" in result

    def test_numeric_parenthetical_not_bilingual(self):
        """'세포(3개)' should not be extracted as bilingual term."""
        text = "세포(3개)를 관찰한다"
        result = extract_bilingual_terms(text)
        # No English alpha in parenthetical
        assert result == []

    def test_mixed_parenthetical(self):
        """'세포(3개, cell)' — has English, should extract."""
        text = "세포(3개, cell)"
        result = extract_bilingual_terms(text)
        assert len(result) == 1
        assert result[0][1] == "cell"

    def test_standalone_numbers_preserved_above_999(self):
        """Page number regex only matches 1-3 digits. '1234' should survive."""
        text = "1234\n본문"
        result = clean_textbook_text(text)
        assert "1234" in result

    def test_real_page_number_stripped(self):
        """'43' alone on a line should be stripped."""
        text = "본문 위\n43\n본문 아래"
        result = clean_textbook_text(text)
        assert "\n43\n" not in result


# ============================================================
# Persona 5: The Division-by-Zero Hunter
# ============================================================


class TestDivisionByZeroHunter:
    """Zero in-scope concepts, zero sections, identical emphasis scores."""

    def test_zero_in_scope(self):
        """All concepts skipped → effective_coverage_rate = 0.0, no ZeroDivisionError."""
        concepts = [_make_concept(name_ko="세포", chapter="1장")]
        scope = TeachingScope(chapters=["2장"])  # chapter doesn't match
        emphasis = [_make_emphasis(concept_name="세포")]
        classified = classify_concepts(concepts, emphasis, scope)
        result = build_coverage_result(classified, [])
        assert result.effective_coverage_rate == 0.0
        assert result.in_scope_count == 0

    def test_zero_sections_per_section_coverage(self):
        """No emphasis data → per_section_coverage empty, no division error."""
        concept = _make_concept()
        classified = [
            ClassifiedConcept(
                concept=concept,
                state=ConceptState.GAP,
                emphasis=None,
                in_scope=True,
            )
        ]
        result = build_coverage_result(classified, [])
        assert result.per_section_coverage == {}

    def test_identical_emphasis_scores_std(self):
        """All same scores → std=0, should not crash."""
        emphasis = _make_emphasis(
            section_scores={"A": 0.5, "B": 0.5, "C": 0.5},
            mean_score=0.5,
            std_score=0.0,
        )
        concept = _make_concept()
        classified = [
            ClassifiedConcept(
                concept=concept,
                state=ConceptState.COVERED,
                emphasis=emphasis,
                in_scope=True,
            )
        ]
        result = build_coverage_result(classified, [])
        assert result.section_variance_top10 == []  # std=0 is filtered out

    def test_single_section_no_stdev_crash(self):
        """One section → std_score should be 0, statistics.stdev needs n>=2."""
        emphasis = _make_emphasis(
            section_scores={"A": 0.5},
            mean_score=0.5,
            std_score=0.0,
        )
        assert emphasis.std_score == 0.0


# ============================================================
# Persona 6: The Scope Confusion Agent
# ============================================================


class TestScopeConfusionAgent:
    """Scope keyword matching nothing, nonexistent chapter, empty include_only."""

    def test_scope_keyword_matches_nothing(self):
        """include_only with non-matching keywords → all concepts out of scope."""
        concept = _make_concept(name_ko="세포", chapter="1장")
        scope = TeachingScope(
            chapters=["1장"],
            scope_rules={"1장": ["존재하지않는키워드"]},
        )
        assert scope.is_in_scope(concept) is False

    def test_scope_on_nonexistent_chapter(self):
        """Scope references chapter not in concept list → SKIPPED."""
        concept = _make_concept(chapter="1장")
        scope = TeachingScope(chapters=["99장"])
        classified = classify_concepts(
            [concept],
            [_make_emphasis()],
            scope,
        )
        assert classified[0].state == ConceptState.SKIPPED

    def test_empty_include_only_means_full_chapter(self):
        """Empty include_only list → full chapter in scope."""
        concept = _make_concept(chapter="1장")
        scope = TeachingScope(
            chapters=["1장"],
            scope_rules={"1장": []},
        )
        assert scope.is_in_scope(concept) is True

    def test_parse_scope_string_chapter_only(self):
        """Scope string with just chapter name, no colon → full chapter."""
        result = parse_scope_string("1장")
        assert result == {"1장": []}

    def test_parse_teaching_scope_invalid_types(self):
        """Non-dict textbook, non-list chapters → graceful defaults."""
        scope = parse_teaching_scope({"textbook": "not_a_dict"})
        assert scope.chapters == []

        scope = parse_teaching_scope({"textbook": {"chapters": "not_a_list"}})
        assert scope.chapters == []

    def test_scope_rules_invalid_include_only_type(self):
        """Non-list include_only → not added to scope_rules."""
        scope = parse_teaching_scope(
            {
                "textbook": {
                    "chapters": ["1장"],
                    "scope": {"1장": {"include_only": "not_a_list"}},
                }
            }
        )
        assert "1장" not in scope.scope_rules


# ============================================================
# Persona 7: The Cache Poisoner
# ============================================================


class TestCachePoisoner:
    """Corrupt cache file, cache with wrong hash, read-only cache directory."""

    def test_corrupt_cache_file(self, tmp_path):
        """Corrupt YAML cache should return None (fall through to re-extract)."""
        from forma.domain_concept_extractor import _cache_path_for, _load_cache

        # Create a textbook file
        textbook = tmp_path / "chapter.txt"
        textbook.write_text("세포는 생명체의 기본 단위이다.", encoding="utf-8")

        # Create corrupt cache
        cache = Path(_cache_path_for(str(textbook)))
        cache.write_text("{{{{invalid yaml!@#$", encoding="utf-8")

        result = _load_cache(str(textbook))
        assert result is None  # Should gracefully return None

    def test_cache_wrong_hash(self, tmp_path):
        """Cache with mismatched hash should be rejected."""
        from forma.domain_concept_extractor import _cache_path_for, _load_cache

        textbook = tmp_path / "chapter.txt"
        textbook.write_text("원래 내용", encoding="utf-8")

        cache = Path(_cache_path_for(str(textbook)))
        cache.write_text(
            yaml.dump(
                {
                    "hash": "deadbeef_wrong_hash",
                    "concepts": [],
                }
            ),
            encoding="utf-8",
        )

        result = _load_cache(str(textbook))
        assert result is None

    def test_cache_missing_keys(self, tmp_path):
        """Cache YAML missing required keys should return None."""
        from forma.domain_concept_extractor import _cache_path_for, _load_cache

        textbook = tmp_path / "chapter.txt"
        textbook.write_text("내용", encoding="utf-8")

        cache = Path(_cache_path_for(str(textbook)))
        cache.write_text(yaml.dump({"random_key": "value"}), encoding="utf-8")

        result = _load_cache(str(textbook))
        assert result is None

    def test_save_cache_readonly_dir(self, tmp_path):
        """Saving cache to read-only directory should handle gracefully."""
        from forma.domain_concept_extractor import _save_cache

        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        textbook = readonly_dir / "chapter.txt"
        textbook.write_text("내용", encoding="utf-8")

        # Make directory read-only
        readonly_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            # _save_cache doesn't catch exceptions; let's verify it raises
            with pytest.raises(PermissionError):
                _save_cache(str(textbook), "내용", [])
        finally:
            readonly_dir.chmod(stat.S_IRWXU)


# ============================================================
# Persona 8: The Encoding Attacker
# ============================================================


class TestEncodingAttacker:
    """EUC-KR encoded textbook, BOM markers, mixed encoding."""

    def test_bom_in_text(self):
        """UTF-8 BOM marker should not break cleaning."""
        text = "\ufeff세포는 기본 단위이다."
        result = clean_textbook_text(text)
        # BOM char is preserved as-is by clean_textbook_text but doesn't crash
        assert "세포" in result

    def test_bilingual_extraction_with_bom(self):
        """BOM should not interfere with bilingual extraction."""
        text = "\ufeff표피(epidermis)는 중요하다"
        result = extract_bilingual_terms(text)
        # \ufeff is a word character, so it may join 표피 but the match should work
        assert len(result) >= 1
        assert any(en == "epidermis" for _, en in result)

    def test_concepts_yaml_utf8_roundtrip(self, tmp_path):
        """Korean concept names survive YAML round-trip."""
        concepts = {"3장 피부": [_make_concept(name_ko="표피", name_en="epidermis", chapter="3장 피부")]}
        out = str(tmp_path / "concepts.yaml")
        save_concepts_yaml(concepts, out)
        loaded = load_concepts_yaml(out)
        assert "3장 피부" in loaded
        assert loaded["3장 피부"][0].name_ko == "표피"

    def test_coverage_yaml_utf8_roundtrip(self, tmp_path):
        """Korean in coverage result survives YAML round-trip."""
        cr = _make_coverage_result()
        out = str(tmp_path / "coverage.yaml")
        save_coverage_yaml(cr, out)
        loaded = load_coverage_yaml(out)
        assert loaded.week == 1


# ============================================================
# Persona 9: The Section Name Guesser
# ============================================================


class TestSectionNameGuesser:
    """Transcript filename without section pattern, ambiguous section names."""

    def test_infer_section_standard_pattern(self):
        """'1A_2주차_1차시.txt' → 'A'."""
        from forma.domain_coverage_analyzer import _infer_section_from_filename

        assert _infer_section_from_filename("1A_2주차_1차시.txt") == "A"

    def test_infer_section_prefix_pattern(self):
        """'A_w2_s1.txt' → 'A'."""
        from forma.domain_coverage_analyzer import _infer_section_from_filename

        assert _infer_section_from_filename("A_w2_s1.txt") == "A"

    def test_infer_section_section_prefix(self):
        """'sectionB_week3.txt' → 'B'."""
        from forma.domain_coverage_analyzer import _infer_section_from_filename

        assert _infer_section_from_filename("sectionB_week3.txt") == "B"

    def test_infer_section_fallback(self):
        """'random_transcript.txt' → 'random_transcript' (stem)."""
        from forma.domain_coverage_analyzer import _infer_section_from_filename

        assert _infer_section_from_filename("random_transcript.txt") == "random_transcript"

    def test_infer_section_no_extension(self):
        """Filename without extension → uses stem."""
        from forma.domain_coverage_analyzer import _infer_section_from_filename

        assert _infer_section_from_filename("transcript") == "transcript"

    def test_infer_section_lowercase(self):
        """'1a_2주차.txt' → 'A' (uppercased)."""
        from forma.domain_coverage_analyzer import _infer_section_from_filename

        assert _infer_section_from_filename("1a_2주차.txt") == "A"


# ============================================================
# Persona 10: The YAML Injection Attacker
# ============================================================


class TestYAMLInjectionAttacker:
    """Malicious YAML in concepts file, extremely long concept names."""

    def test_load_concepts_yaml_not_dict(self, tmp_path):
        """YAML containing a list instead of dict → ValueError."""
        p = tmp_path / "concepts.yaml"
        p.write_text(yaml.dump(["a", "b", "c"]), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid"):
            load_concepts_yaml(str(p))

    def test_load_concepts_yaml_missing_chapters_key(self, tmp_path):
        """Dict without 'chapters' key → ValueError."""
        p = tmp_path / "concepts.yaml"
        p.write_text(yaml.dump({"data": "something"}), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid"):
            load_concepts_yaml(str(p))

    def test_load_coverage_yaml_not_dict(self, tmp_path):
        """Coverage YAML that's a list → ValueError."""
        p = tmp_path / "coverage.yaml"
        p.write_text(yaml.dump([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid"):
            load_coverage_yaml(str(p))

    def test_extremely_long_concept_name(self, tmp_path):
        """500-char concept name should not crash YAML I/O."""
        long_name = "가" * 500
        concepts = {"1장": [_make_concept(name_ko=long_name, chapter="1장")]}
        out = str(tmp_path / "long.yaml")
        save_concepts_yaml(concepts, out)
        loaded = load_concepts_yaml(out)
        assert loaded["1장"][0].name_ko == long_name

    def test_yaml_safe_load_prevents_arbitrary_objects(self, tmp_path):
        """yaml.safe_load should prevent arbitrary Python objects."""
        p = tmp_path / "evil.yaml"
        p.write_text("!!python/object/apply:os.system ['echo pwned']", encoding="utf-8")
        # safe_load should raise an error, not execute
        with pytest.raises(Exception):
            load_concepts_yaml(str(p))

    def test_concept_name_with_yaml_special_chars(self, tmp_path):
        """Concept names with colons, brackets, etc. survive YAML."""
        name = "세포: {대장균} [E.coli]"
        concepts = {"1장": [_make_concept(name_ko=name, chapter="1장")]}
        out = str(tmp_path / "special.yaml")
        save_concepts_yaml(concepts, out)
        loaded = load_concepts_yaml(out)
        assert loaded["1장"][0].name_ko == name


# ============================================================
# Persona 11: The Memory Hog
# ============================================================


class TestMemoryHog:
    """Charts with many concepts; large classification tables."""

    def test_coverage_bar_chart_many_sections(self):
        """Coverage bar chart with 20 sections should not crash."""
        from forma.domain_coverage_charts import build_coverage_bar_chart

        cr = _make_coverage_result(
            per_section_coverage={f"S{i}": 0.5 + 0.02 * i for i in range(20)},
        )
        buf = build_coverage_bar_chart(cr)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_emphasis_bias_scatter_500_points(self):
        """Scatter chart with 500 in-scope concepts."""
        from forma.domain_coverage_charts import build_emphasis_bias_scatter

        classified = []
        for i in range(500):
            classified.append(
                _make_classified(
                    name_ko=f"개념{i}",
                    state=ConceptState.COVERED if i % 2 == 0 else ConceptState.GAP,
                    mean_score=0.1 * (i % 10),
                    frequency=i + 1,
                )
            )
        cr = _make_coverage_result(classified_concepts=classified)
        buf = build_emphasis_bias_scatter(cr)
        assert isinstance(buf, io.BytesIO)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_heatmap_max_concepts_cap(self):
        """Heatmap should cap at max_concepts even with 200 input concepts."""
        from forma.domain_coverage_charts import build_section_variance_heatmap

        classified = []
        for i in range(200):
            classified.append(
                _make_classified(
                    name_ko=f"개념{i}",
                    std_score=0.01 * (i + 1),
                )
            )
        cr = _make_coverage_result(classified_concepts=classified)
        buf = build_section_variance_heatmap(cr, max_concepts=20)
        assert isinstance(buf, io.BytesIO)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_heatmap_empty_data(self):
        """Heatmap with no emphasis data should produce a placeholder chart."""
        from forma.domain_coverage_charts import build_section_variance_heatmap

        concept = _make_concept()
        cc = ClassifiedConcept(
            concept=concept,
            state=ConceptState.GAP,
            emphasis=None,
            in_scope=True,
        )
        cr = _make_coverage_result(classified_concepts=[cc])
        buf = build_section_variance_heatmap(cr)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"
        import matplotlib.pyplot as plt

        plt.close("all")


# ============================================================
# Persona 12: The Regression Sniffer
# ============================================================


class TestRegressionSniffer:
    """Check all imports resolve and key functions exist."""

    def test_all_textbook_preprocessor_exports(self):
        """textbook_preprocessor __all__ entries are importable."""
        from forma import textbook_preprocessor

        for name in textbook_preprocessor.__all__:
            assert hasattr(textbook_preprocessor, name)

    def test_all_concept_extractor_exports(self):
        """domain_concept_extractor __all__ entries are importable."""
        from forma import domain_concept_extractor

        for name in domain_concept_extractor.__all__:
            assert hasattr(domain_concept_extractor, name)

    def test_all_coverage_analyzer_exports(self):
        """domain_coverage_analyzer __all__ entries are importable."""
        from forma import domain_coverage_analyzer

        for name in domain_coverage_analyzer.__all__:
            assert hasattr(domain_coverage_analyzer, name)

    def test_all_coverage_charts_exports(self):
        """domain_coverage_charts __all__ entries are importable."""
        from forma import domain_coverage_charts

        for name in domain_coverage_charts.__all__:
            assert hasattr(domain_coverage_charts, name)

    def test_all_coverage_report_exports(self):
        """domain_coverage_report __all__ entries are importable."""
        from forma import domain_coverage_report

        for name in domain_coverage_report.__all__:
            assert hasattr(domain_coverage_report, name)

    def test_all_cli_domain_exports(self):
        """cli_domain __all__ entries are importable."""
        from forma import cli_domain

        for name in cli_domain.__all__:
            assert hasattr(cli_domain, name)

    def test_coverage_yaml_round_trip(self, tmp_path):
        """CoverageResult survives save → load."""
        cr = _make_coverage_result()
        out = str(tmp_path / "rt.yaml")
        save_coverage_yaml(cr, out)
        loaded = load_coverage_yaml(out)
        assert loaded.week == cr.week
        assert loaded.covered_count == cr.covered_count
        assert loaded.gap_count == cr.gap_count
        assert loaded.effective_coverage_rate == pytest.approx(
            cr.effective_coverage_rate,
            abs=0.001,
        )

    def test_concepts_yaml_round_trip(self, tmp_path):
        """concepts YAML round-trip preserves data."""
        concepts = {
            "1장": [_make_concept(name_ko="세포"), _make_concept(name_ko="조직", name_en=None)],
            "2장": [_make_concept(name_ko="기관", chapter="2장")],
        }
        out = str(tmp_path / "concepts_rt.yaml")
        save_concepts_yaml(concepts, out)
        loaded = load_concepts_yaml(out)
        assert set(loaded.keys()) == {"1장", "2장"}
        assert loaded["1장"][0].name_ko == "세포"
        assert loaded["2장"][0].name_ko == "기관"

    def test_classify_state_transitions(self):
        """Verify correct state assignment for in-scope above/below threshold."""
        concept_above = _make_concept(name_ko="위", chapter="1장")
        concept_below = _make_concept(name_ko="아래", chapter="1장")
        concept_out = _make_concept(name_ko="밖", chapter="2장")

        emphasis = [
            _make_emphasis(concept_name="위", mean_score=0.10),
            _make_emphasis(concept_name="아래", mean_score=0.01),
            _make_emphasis(concept_name="밖", mean_score=0.50),
        ]

        scope = TeachingScope(chapters=["1장"])
        result = classify_concepts(
            [concept_above, concept_below, concept_out],
            emphasis,
            scope,
        )

        assert result[0].state == ConceptState.COVERED
        assert result[1].state == ConceptState.GAP
        assert result[2].state == ConceptState.SKIPPED
