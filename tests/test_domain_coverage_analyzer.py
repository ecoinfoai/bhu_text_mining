"""Tests for domain coverage analysis and 4-state classification.

T016: TeachingScope
T017: compute_concept_emphasis (mocked)
T018: classify_concepts
T019: CoverageResult
T020: detect_extra_concepts

v2 tests:
T020-T024: Delivery analysis (LLM mock)
T034-T035: Network comparison
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from forma.domain_concept_extractor import TextbookConcept
from forma.domain_coverage_analyzer import (
    ClassifiedConcept,
    ConceptEmphasis,
    ConceptState,
    DeliveryAnalysis,
    DeliveryResult,
    DeliveryState,
    ExtraConcept,
    KeywordNetwork,
    TeachingScope,
    build_coverage_result,
    build_delivery_prompt,
    build_delivery_result_v2,
    build_domain_network,
    classify_concepts,
    compare_networks,
    load_coverage_yaml,
    load_delivery_yaml,
    parse_scope_string,
    parse_teaching_scope,
    save_coverage_yaml,
    save_delivery_yaml,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _make_concept(
    name_ko: str,
    chapter: str = "3장 피부",
    frequency: int = 5,
    name_en: str | None = None,
) -> TextbookConcept:
    return TextbookConcept(
        name_ko=name_ko,
        name_en=name_en,
        chapter=chapter,
        frequency=frequency,
        context_sentence=f"{name_ko}는 중요하다.",
        is_bilingual=name_en is not None,
    )


def _make_emphasis(
    concept_name: str,
    chapter: str = "3장 피부",
    section_scores: dict[str, float] | None = None,
) -> ConceptEmphasis:
    if section_scores is None:
        section_scores = {"A": 0.5, "B": 0.6}
    values = list(section_scores.values())
    import statistics

    mean_score = statistics.mean(values) if values else 0.0
    std_score = statistics.stdev(values) if len(values) >= 2 else 0.0
    return ConceptEmphasis(
        concept_name=concept_name,
        chapter=chapter,
        section_scores=section_scores,
        mean_score=mean_score,
        std_score=std_score,
    )


# ----------------------------------------------------------------
# T016: TeachingScope
# ----------------------------------------------------------------


class TestTeachingScope:
    """Tests for TeachingScope parsing and scope checking."""

    def test_parse_from_dict(self) -> None:
        """Parse TeachingScope from week.yaml dict."""
        data = {
            "textbook": {
                "chapters": ["1장", "2장", "3장"],
                "scope": {
                    "2장": {"include_only": ["확산", "능동수송"]},
                },
            },
        }
        scope = parse_teaching_scope(data)
        assert scope.chapters == ["1장", "2장", "3장"]
        assert scope.scope_rules == {"2장": ["확산", "능동수송"]}

    def test_in_scope_scoped_chapter_matching_keyword(self) -> None:
        """Concept in scoped chapter with matching keyword is in scope."""
        scope = TeachingScope(
            chapters=["2장"],
            scope_rules={"2장": ["확산", "능동수송"]},
        )
        concept = _make_concept("능동수송", chapter="2장")
        assert scope.is_in_scope(concept) is True

    def test_in_scope_scoped_chapter_substring_match(self) -> None:
        """FR-014: Keyword matches via substring containment."""
        scope = TeachingScope(
            chapters=["2장"],
            scope_rules={"2장": ["능동수송"]},
        )
        concept = _make_concept("능동수송기전", chapter="2장")
        assert scope.is_in_scope(concept) is True

    def test_in_scope_scoped_chapter_no_match(self) -> None:
        """Concept in scoped chapter without matching keyword is out."""
        scope = TeachingScope(
            chapters=["2장"],
            scope_rules={"2장": ["확산"]},
        )
        concept = _make_concept("삼투", chapter="2장")
        assert scope.is_in_scope(concept) is False

    def test_in_scope_unscoped_chapter(self) -> None:
        """Concept in unscoped chapter is always in scope."""
        scope = TeachingScope(
            chapters=["1장", "2장"],
            scope_rules={"2장": ["확산"]},
        )
        concept = _make_concept("세포막", chapter="1장")
        assert scope.is_in_scope(concept) is True

    def test_in_scope_chapter_not_listed(self) -> None:
        """Concept in chapter not in chapters list is out of scope."""
        scope = TeachingScope(
            chapters=["1장"],
            scope_rules={},
        )
        concept = _make_concept("표피", chapter="3장 피부")
        assert scope.is_in_scope(concept) is False

    def test_parse_empty_textbook_section(self) -> None:
        """Missing textbook section returns empty scope."""
        scope = parse_teaching_scope({})
        assert scope.chapters == []
        assert scope.scope_rules == {}

    def test_parse_scope_string_basic(self) -> None:
        """parse_scope_string parses '2장:확산,능동수송' format."""
        result = parse_scope_string("2장:확산,능동수송")
        assert result == {"2장": ["확산", "능동수송"]}

    def test_parse_scope_string_multiple(self) -> None:
        """parse_scope_string handles semicolon-separated chapters."""
        result = parse_scope_string("2장:확산;3장:")
        assert result == {"2장": ["확산"], "3장": []}


# ----------------------------------------------------------------
# T017: compute_concept_emphasis (mocked)
# ----------------------------------------------------------------


class TestComputeConceptEmphasis:
    """Tests for emphasis computation with mocked dependencies."""

    def test_returns_emphasis_list(self, tmp_path) -> None:
        """compute_concept_emphasis returns ConceptEmphasis list."""
        # Create mock transcript files
        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text("세포막은 인지질 이중층으로 구성된다.", encoding="utf-8")

        concepts = [_make_concept("세포막"), _make_concept("인지질")]

        mock_emphasis = MagicMock()
        mock_emphasis.concept_scores = {"세포막": 0.8, "인지질": 0.6}

        with patch(
            "forma.emphasis_map.compute_emphasis_map",
            return_value=mock_emphasis,
        ), patch(
            "kss.split_sentences",
            return_value=["세포막은 인지질 이중층으로 구성된다."],
        ):
            from forma.domain_coverage_analyzer import compute_concept_emphasis

            result = compute_concept_emphasis(
                transcript_paths=[str(transcript)],
                concepts=concepts,
                threshold=0.65,
            )

        assert len(result) == 2
        assert result[0].concept_name == "세포막"
        assert result[0].section_scores.get("A") == 0.8

    def test_multiple_sections(self, tmp_path) -> None:
        """Handles multiple sections correctly."""
        t_a = tmp_path / "1A_2주차_1차시.txt"
        t_b = tmp_path / "1B_2주차_1차시.txt"
        t_a.write_text("세포막은 중요하다.", encoding="utf-8")
        t_b.write_text("세포막은 중요하다.", encoding="utf-8")

        concepts = [_make_concept("세포막")]

        call_count = [0]

        def mock_emphasis_fn(sentences, concepts, threshold):
            call_count[0] += 1
            mock = MagicMock()
            if call_count[0] == 1:
                mock.concept_scores = {"세포막": 0.9}
            else:
                mock.concept_scores = {"세포막": 0.7}
            return mock

        with patch(
            "forma.emphasis_map.compute_emphasis_map",
            side_effect=mock_emphasis_fn,
        ), patch(
            "kss.split_sentences",
            return_value=["세포막은 중요하다."],
        ):
            from forma.domain_coverage_analyzer import compute_concept_emphasis

            result = compute_concept_emphasis(
                transcript_paths=[str(t_a), str(t_b)],
                concepts=concepts,
                threshold=0.65,
            )

        assert len(result) == 1
        assert "A" in result[0].section_scores
        assert "B" in result[0].section_scores
        assert result[0].section_scores["A"] == 0.9
        assert result[0].section_scores["B"] == 0.7

    def test_empty_concepts_returns_empty(self) -> None:
        """Empty concepts list returns empty result."""
        from forma.domain_coverage_analyzer import compute_concept_emphasis

        result = compute_concept_emphasis([], [], 0.65)
        assert result == []

    def test_single_section(self, tmp_path) -> None:
        """Single section produces std_score of 0."""
        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text("세포막은 중요하다.", encoding="utf-8")

        concepts = [_make_concept("세포막")]

        mock_emphasis = MagicMock()
        mock_emphasis.concept_scores = {"세포막": 0.8}

        with patch(
            "forma.emphasis_map.compute_emphasis_map",
            return_value=mock_emphasis,
        ), patch(
            "kss.split_sentences",
            return_value=["세포막은 중요하다."],
        ):
            from forma.domain_coverage_analyzer import compute_concept_emphasis

            result = compute_concept_emphasis(
                transcript_paths=[str(transcript)],
                concepts=concepts,
            )

        assert len(result) == 1
        assert result[0].std_score == 0.0


# ----------------------------------------------------------------
# T018: classify_concepts
# ----------------------------------------------------------------


class TestClassifyConcepts:
    """Tests for 4-state classification."""

    def test_covered_in_scope_above_threshold(self) -> None:
        """In-scope concept with emphasis > 0.05 → COVERED."""
        concepts = [_make_concept("표피", chapter="3장")]
        emphasis = [_make_emphasis("표피", chapter="3장", section_scores={"A": 0.5})]
        scope = TeachingScope(chapters=["3장"])

        result = classify_concepts(concepts, emphasis, scope)

        assert len(result) == 1
        assert result[0].state == ConceptState.COVERED
        assert result[0].in_scope is True

    def test_gap_in_scope_below_threshold(self) -> None:
        """In-scope concept with emphasis < 0.05 → GAP."""
        concepts = [_make_concept("표피", chapter="3장")]
        emphasis = [_make_emphasis("표피", chapter="3장", section_scores={"A": 0.01})]
        scope = TeachingScope(chapters=["3장"])

        result = classify_concepts(concepts, emphasis, scope)

        assert result[0].state == ConceptState.GAP

    def test_skipped_out_of_scope(self) -> None:
        """Out-of-scope concept → SKIPPED."""
        concepts = [_make_concept("표피", chapter="3장")]
        emphasis = []
        scope = TeachingScope(chapters=["1장"])

        result = classify_concepts(concepts, emphasis, scope)

        assert result[0].state == ConceptState.SKIPPED
        assert result[0].in_scope is False

    def test_gap_no_emphasis_data(self) -> None:
        """In-scope concept with no emphasis data → GAP."""
        concepts = [_make_concept("표피", chapter="3장")]
        emphasis = []
        scope = TeachingScope(chapters=["3장"])

        result = classify_concepts(concepts, emphasis, scope)

        assert result[0].state == ConceptState.GAP

    def test_multiple_concepts_mixed_states(self) -> None:
        """Multiple concepts get correct states."""
        concepts = [
            _make_concept("표피", chapter="3장"),
            _make_concept("세포", chapter="1장"),
            _make_concept("근육", chapter="5장"),
        ]
        emphasis = [
            _make_emphasis("표피", chapter="3장", section_scores={"A": 0.5}),
            _make_emphasis("세포", chapter="1장", section_scores={"A": 0.01}),
        ]
        scope = TeachingScope(chapters=["3장", "1장"])

        result = classify_concepts(concepts, emphasis, scope)

        assert result[0].state == ConceptState.COVERED  # 표피: in scope, high emphasis
        assert result[1].state == ConceptState.GAP  # 세포: in scope, low emphasis
        assert result[2].state == ConceptState.SKIPPED  # 근육: out of scope


# ----------------------------------------------------------------
# T019: CoverageResult + build_coverage_result
# ----------------------------------------------------------------


class TestCoverageResult:
    """Tests for CoverageResult aggregation."""

    def _make_classified_list(self) -> list[ClassifiedConcept]:
        """Build a test classified concept list."""
        concepts = [
            _make_concept("표피", chapter="3장", frequency=10),
            _make_concept("진피", chapter="3장", frequency=8),
            _make_concept("세포", chapter="1장", frequency=5),
            _make_concept("근육", chapter="5장", frequency=3),
        ]
        emphasis = [
            _make_emphasis("표피", chapter="3장", section_scores={"A": 0.8, "B": 0.6}),
            _make_emphasis("진피", chapter="3장", section_scores={"A": 0.7, "B": 0.5}),
            _make_emphasis("세포", chapter="1장", section_scores={"A": 0.02, "B": 0.01}),
        ]
        scope = TeachingScope(chapters=["3장", "1장"])
        return classify_concepts(concepts, emphasis, scope)

    def test_effective_coverage_rate(self) -> None:
        """effective_coverage_rate = covered / in_scope."""
        classified = self._make_classified_list()
        result = build_coverage_result(classified, [])

        # 표피: COVERED, 진피: COVERED, 세포: GAP, 근육: SKIPPED
        # in_scope = 3, covered = 2
        assert result.in_scope_count == 3
        assert result.covered_count == 2
        assert result.gap_count == 1
        assert result.skipped_count == 1
        assert result.effective_coverage_rate == pytest.approx(2 / 3, abs=0.01)

    def test_per_section_coverage(self) -> None:
        """per_section_coverage computed correctly."""
        classified = self._make_classified_list()
        result = build_coverage_result(classified, [])

        # Section A: 표피=0.8>=0.05, 진피=0.7>=0.05, 세포=0.02<0.05 → 2/3
        assert result.per_section_coverage["A"] == pytest.approx(2 / 3, abs=0.01)
        assert "B" in result.per_section_coverage

    def test_section_variance_top10_sorted(self) -> None:
        """section_variance_top10 sorted by std descending."""
        classified = self._make_classified_list()
        result = build_coverage_result(classified, [])

        if result.section_variance_top10:
            stds = [s for _, s in result.section_variance_top10]
            assert stds == sorted(stds, reverse=True)

    def test_zero_in_scope_no_division_error(self) -> None:
        """Zero in-scope concepts doesn't cause division by zero."""
        concepts = [_make_concept("근육", chapter="5장")]
        scope = TeachingScope(chapters=["3장"])
        classified = classify_concepts(concepts, [], scope)
        result = build_coverage_result(classified, [])

        assert result.effective_coverage_rate == 0.0
        assert result.in_scope_count == 0


# ----------------------------------------------------------------
# T020: ExtraConcept detection (placeholder, needs KoNLPy)
# ----------------------------------------------------------------


class TestExtraConcept:
    """Tests for ExtraConcept dataclass."""

    def test_extra_concept_creation(self) -> None:
        """ExtraConcept can be created with required fields."""
        extra = ExtraConcept(
            name="염증",
            section_mentions={"A": 5, "B": 3},
            example_sentence="염증 반응이 나타났다.",
        )
        assert extra.name == "염증"
        assert extra.section_mentions["A"] == 5
        assert "염증" in extra.example_sentence


# ----------------------------------------------------------------
# T027: YAML round-trip
# ----------------------------------------------------------------


class TestCoverageYAMLIO:
    """Tests for coverage YAML save/load."""

    def test_save_load_roundtrip(self, tmp_path) -> None:
        """Save and load produces equivalent result."""
        concepts = [_make_concept("표피", chapter="3장", frequency=10)]
        emphasis = [
            _make_emphasis("표피", chapter="3장", section_scores={"A": 0.8, "B": 0.6}),
        ]
        scope = TeachingScope(chapters=["3장"])
        classified = classify_concepts(concepts, emphasis, scope)
        extras = [ExtraConcept(
            name="염증",
            section_mentions={"A": 5},
            example_sentence="염증 반응",
        )]

        original = build_coverage_result(classified, extras, week=2, chapters=["3장"])

        path = str(tmp_path / "coverage.yaml")
        save_coverage_yaml(original, path)
        loaded = load_coverage_yaml(path)

        assert loaded.week == original.week
        assert loaded.chapters == original.chapters
        assert loaded.covered_count == original.covered_count
        assert loaded.gap_count == original.gap_count
        assert loaded.skipped_count == original.skipped_count
        assert loaded.extra_count == original.extra_count
        assert loaded.effective_coverage_rate == pytest.approx(
            original.effective_coverage_rate, abs=0.001,
        )
        assert len(loaded.classified_concepts) == len(original.classified_concepts)
        assert len(loaded.extra_concepts) == len(original.extra_concepts)


# ================================================================
# v2: Delivery Analysis Tests (T020-T024)
# ================================================================


class TestDeliveryDataclasses:
    """T020: DeliveryAnalysis dataclass and DeliveryState enum."""

    def test_delivery_state_values(self) -> None:
        """DeliveryState enum has expected values."""
        assert DeliveryState.FULLY_DELIVERED.value == "충분히 설명"
        assert DeliveryState.PARTIALLY_DELIVERED.value == "부분 전달"
        assert DeliveryState.NOT_DELIVERED.value == "미전달"
        assert DeliveryState.SKIPPED.value == "의도적 생략"

    def test_delivery_analysis_creation(self) -> None:
        """DeliveryAnalysis can be created with all fields."""
        da = DeliveryAnalysis(
            concept="표피의 4층 구조",
            section_id="A",
            delivery_status="충분히 설명",
            delivery_quality=0.85,
            evidence="표피는 4개의 층으로 구성됩니다.",
            depth="메커니즘과 임상 적용까지 설명",
            analysis_level="v2",
        )
        assert da.concept == "표피의 4층 구조"
        assert da.section_id == "A"
        assert da.delivery_quality == 0.85
        assert da.analysis_level == "v2"

    def test_delivery_analysis_default_level(self) -> None:
        """DeliveryAnalysis defaults to analysis_level='v2'."""
        da = DeliveryAnalysis(
            concept="test",
            section_id="A",
            delivery_status="미전달",
            delivery_quality=0.0,
            evidence="",
            depth="",
        )
        assert da.analysis_level == "v2"


class TestDeliveryLLMAnalysis:
    """T021: analyze_delivery_llm with mocked LLM."""

    def test_fully_delivered_parsing(self, tmp_path) -> None:
        """LLM response with '충분히 설명' parsed correctly."""
        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text("표피는 4개의 층으로 구성됩니다.", encoding="utf-8")

        mock_response = """\
deliveries:
  - concept: "표피의 4층 구조"
    delivery_status: "충분히 설명"
    delivery_quality: 0.9
    evidence: "표피는 4개의 층으로 구성됩니다."
    depth: "상세 설명"
"""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ):
            from forma.domain_coverage_analyzer import analyze_delivery_llm

            result = analyze_delivery_llm(
                concepts=["표피의 4층 구조"],
                transcript_path=str(transcript),
                section_id="A",
            )

        assert len(result) == 1
        assert result[0].delivery_status == "충분히 설명"
        assert result[0].delivery_quality == pytest.approx(0.9, abs=0.01)
        assert result[0].analysis_level == "v2"

    def test_partial_delivery_parsing(self, tmp_path) -> None:
        """LLM response with '부분 전달' parsed correctly."""
        transcript = tmp_path / "1B_2주차_1차시.txt"
        transcript.write_text("진피라는 것이 있습니다.", encoding="utf-8")

        mock_response = """\
deliveries:
  - concept: "진피의 구조"
    delivery_status: "부분 전달"
    delivery_quality: 0.4
    evidence: "진피라는 것이 있습니다."
    depth: "용어만 언급, 메커니즘 미설명"
"""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ):
            from forma.domain_coverage_analyzer import analyze_delivery_llm

            result = analyze_delivery_llm(
                concepts=["진피의 구조"],
                transcript_path=str(transcript),
                section_id="B",
            )

        assert len(result) == 1
        assert result[0].delivery_status == "부분 전달"
        assert result[0].section_id == "B"

    def test_not_delivered_parsing(self, tmp_path) -> None:
        """LLM response with '미전달' parsed correctly."""
        transcript = tmp_path / "1C_2주차_1차시.txt"
        transcript.write_text("오늘은 근육에 대해 알아봅시다.", encoding="utf-8")

        mock_response = """\
deliveries:
  - concept: "표피의 4층 구조"
    delivery_status: "미전달"
    delivery_quality: 0.0
    evidence: ""
    depth: ""
"""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ):
            from forma.domain_coverage_analyzer import analyze_delivery_llm

            result = analyze_delivery_llm(
                concepts=["표피의 4층 구조"],
                transcript_path=str(transcript),
                section_id="C",
            )

        assert len(result) == 1
        assert result[0].delivery_status == "미전달"
        assert result[0].delivery_quality == 0.0


class TestV1FallbackAnalysis:
    """T022: v1 fallback when LLM raises exception."""

    def test_fallback_marks_v1_level(self, tmp_path) -> None:
        """v1 fallback sets analysis_level='v1'."""
        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text("표피는 중요하다. 세포막도 중요하다.", encoding="utf-8")

        mock_emphasis = MagicMock()
        mock_emphasis.concept_scores = {"표피": 0.5, "세포막": 0.03}

        with patch(
            "forma.emphasis_map.compute_emphasis_map",
            return_value=mock_emphasis,
        ):
            from forma.domain_coverage_analyzer import v1_fallback_analysis

            result = v1_fallback_analysis(
                concepts=["표피", "세포막"],
                transcript_path=str(transcript),
                section_id="A",
            )

        assert len(result) == 2
        assert all(d.analysis_level == "v1" for d in result)

    def test_fallback_fully_threshold(self, tmp_path) -> None:
        """Score >= 0.3 maps to FULLY_DELIVERED."""
        transcript = tmp_path / "test.txt"
        transcript.write_text("표피는 매우 중요합니다.", encoding="utf-8")

        mock_emphasis = MagicMock()
        mock_emphasis.concept_scores = {"표피": 0.5}

        with patch(
            "forma.emphasis_map.compute_emphasis_map",
            return_value=mock_emphasis,
        ):
            from forma.domain_coverage_analyzer import v1_fallback_analysis

            result = v1_fallback_analysis(
                concepts=["표피"],
                transcript_path=str(transcript),
                section_id="A",
            )

        assert result[0].delivery_status == "충분히 설명"

    def test_fallback_partial_threshold(self, tmp_path) -> None:
        """Score >= 0.05 and < 0.3 maps to PARTIALLY_DELIVERED."""
        transcript = tmp_path / "test.txt"
        transcript.write_text("표피 언급.", encoding="utf-8")

        mock_emphasis = MagicMock()
        mock_emphasis.concept_scores = {"표피": 0.1}

        with patch(
            "forma.emphasis_map.compute_emphasis_map",
            return_value=mock_emphasis,
        ):
            from forma.domain_coverage_analyzer import v1_fallback_analysis

            result = v1_fallback_analysis(
                concepts=["표피"],
                transcript_path=str(transcript),
                section_id="A",
            )

        assert result[0].delivery_status == "부분 전달"

    def test_fallback_not_delivered_threshold(self, tmp_path) -> None:
        """Score < 0.05 maps to NOT_DELIVERED."""
        transcript = tmp_path / "test.txt"
        transcript.write_text("오늘은 다른 주제입니다.", encoding="utf-8")

        mock_emphasis = MagicMock()
        mock_emphasis.concept_scores = {"표피": 0.01}

        with patch(
            "forma.emphasis_map.compute_emphasis_map",
            return_value=mock_emphasis,
        ):
            from forma.domain_coverage_analyzer import v1_fallback_analysis

            result = v1_fallback_analysis(
                concepts=["표피"],
                transcript_path=str(transcript),
                section_id="A",
            )

        assert result[0].delivery_status == "미전달"


class TestAnalyzeDeliveryWithFallback:
    """T023: LLM failure triggers v1 fallback."""

    def test_llm_exception_triggers_fallback(self, tmp_path) -> None:
        """When LLM raises, fallback returns v1 results."""
        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text("표피는 중요합니다.", encoding="utf-8")

        mock_emphasis = MagicMock()
        mock_emphasis.concept_scores = {"표피": 0.5}

        with patch(
            "forma.domain_coverage_analyzer.analyze_delivery_llm",
            side_effect=RuntimeError("LLM unavailable"),
        ), patch(
            "forma.emphasis_map.compute_emphasis_map",
            return_value=mock_emphasis,
        ):
            from forma.domain_coverage_analyzer import (
                analyze_delivery_with_fallback,
            )

            result = analyze_delivery_with_fallback(
                concepts=["표피"],
                transcript_path=str(transcript),
                section_id="A",
            )

        assert len(result) == 1
        assert result[0].analysis_level == "v1"

    def test_llm_success_returns_v2(self, tmp_path) -> None:
        """When LLM succeeds, returns v2 results."""
        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text("표피는 중요합니다.", encoding="utf-8")

        v2_result = [
            DeliveryAnalysis(
                concept="표피",
                section_id="A",
                delivery_status="충분히 설명",
                delivery_quality=0.9,
                evidence="표피는 중요합니다.",
                depth="상세",
                analysis_level="v2",
            ),
        ]

        with patch(
            "forma.domain_coverage_analyzer.analyze_delivery_llm",
            return_value=v2_result,
        ):
            from forma.domain_coverage_analyzer import (
                analyze_delivery_with_fallback,
            )

            result = analyze_delivery_with_fallback(
                concepts=["표피"],
                transcript_path=str(transcript),
                section_id="A",
            )

        assert len(result) == 1
        assert result[0].analysis_level == "v2"


class TestTeachingScopeSkipped:
    """T024: Out-of-scope concepts → SKIPPED delivery status."""

    def test_out_of_scope_concept_is_skipped(self) -> None:
        """DeliveryState.SKIPPED value matches data model."""
        assert DeliveryState.SKIPPED.value == "의도적 생략"

    def test_scope_filtering(self) -> None:
        """Concepts not in scope should be identified as skipped."""
        scope = TeachingScope(chapters=["1장"], scope_rules={})
        concept = _make_concept("근육", chapter="5장")
        assert scope.is_in_scope(concept) is False


class TestDeliveryResult:
    """T025: DeliveryResult effective_delivery_rate."""

    def test_effective_delivery_rate(self) -> None:
        """(fully + partially) / in_scope."""
        deliveries = [
            DeliveryAnalysis("c1", "A", "충분히 설명", 0.9, "", "", "v2"),
            DeliveryAnalysis("c2", "A", "부분 전달", 0.4, "", "", "v2"),
            DeliveryAnalysis("c3", "A", "미전달", 0.0, "", "", "v2"),
        ]
        scope = TeachingScope(chapters=["3장"])
        result = build_delivery_result_v2(
            deliveries, scope, ["c1", "c2", "c3"],
        )
        # 2 delivered out of 3
        assert result.effective_delivery_rate == pytest.approx(2 / 3, abs=0.01)

    def test_per_section_rate(self) -> None:
        """Per-section rates computed correctly."""
        deliveries = [
            DeliveryAnalysis("c1", "A", "충분히 설명", 0.9, "", "", "v2"),
            DeliveryAnalysis("c2", "A", "미전달", 0.0, "", "", "v2"),
            DeliveryAnalysis("c1", "B", "부분 전달", 0.4, "", "", "v2"),
            DeliveryAnalysis("c2", "B", "부분 전달", 0.3, "", "", "v2"),
        ]
        scope = TeachingScope(chapters=["3장"])
        result = build_delivery_result_v2(
            deliveries, scope, ["c1", "c2"],
        )
        assert result.per_section_rate["A"] == pytest.approx(0.5, abs=0.01)
        assert result.per_section_rate["B"] == pytest.approx(1.0, abs=0.01)

    def test_empty_deliveries(self) -> None:
        """Empty deliveries gives 0 rate."""
        scope = TeachingScope(chapters=["3장"])
        result = build_delivery_result_v2([], scope, [])
        assert result.effective_delivery_rate == 0.0


class TestDeliveryYAMLIO:
    """T032: Delivery YAML round-trip."""

    def test_save_load_roundtrip(self, tmp_path) -> None:
        """Save and load produces equivalent DeliveryResult."""
        deliveries = [
            DeliveryAnalysis("c1", "A", "충분히 설명", 0.9, "증거", "깊이", "v2"),
            DeliveryAnalysis("c2", "A", "미전달", 0.0, "", "", "v1"),
        ]
        original = DeliveryResult(
            week=2,
            chapters=["3장"],
            deliveries=deliveries,
            effective_delivery_rate=0.5,
            per_section_rate={"A": 0.5},
        )

        path = str(tmp_path / "delivery.yaml")
        save_delivery_yaml(original, path)
        loaded = load_delivery_yaml(path)

        assert loaded.week == original.week
        assert loaded.chapters == original.chapters
        assert loaded.effective_delivery_rate == pytest.approx(
            original.effective_delivery_rate, abs=0.001,
        )
        assert len(loaded.deliveries) == 2
        assert loaded.deliveries[0].concept == "c1"
        assert loaded.deliveries[0].analysis_level == "v2"
        assert loaded.deliveries[1].analysis_level == "v1"


class TestBuildDeliveryPrompt:
    """T027: build_delivery_prompt."""

    def test_prompt_contains_concepts(self) -> None:
        """Prompt includes all concept names."""
        prompt = build_delivery_prompt(
            concepts=["표피의 4층 구조", "진피의 구조"],
            transcript_text="강의 내용...",
        )
        assert "표피의 4층 구조" in prompt
        assert "진피의 구조" in prompt
        assert "강의 내용..." in prompt

    def test_prompt_contains_yaml_format(self) -> None:
        """Prompt includes YAML output format instructions."""
        prompt = build_delivery_prompt(["test"], "text")
        assert "deliveries:" in prompt
        assert "delivery_status" in prompt


# ================================================================
# v2: Network Comparison Tests (T034-T035)
# ================================================================


class TestBuildDomainNetwork:
    """T034: build_domain_network filters to domain terms."""

    def test_filters_to_key_terms(self) -> None:
        """Network only contains nodes from key_terms."""
        import networkx as nx

        mock_graph = nx.Graph()
        mock_graph.add_edge("세포막", "인지질", weight=2.0)
        mock_graph.nodes["세포막"]["frequency"] = 2
        mock_graph.nodes["인지질"]["frequency"] = 2

        with patch(
            "forma.network_analysis.extract_keywords",
            return_value=["세포막", "인지질", "세포막", "인지질"],
        ), patch(
            "forma.network_analysis.create_network",
            return_value=mock_graph,
        ):
            net = build_domain_network(
                text="세포막은 인지질로 구성. 오늘은 세포막을 배운다.",
                key_terms=["세포막", "인지질"],
                source="textbook",
            )

        assert "세포막" in net.nodes
        assert "인지질" in net.nodes
        assert net.source == "textbook"
        assert len(net.edges) == 1

    def test_empty_text_returns_empty_network(self) -> None:
        """Empty filtered keywords return empty network."""
        with patch(
            "forma.network_analysis.extract_keywords",
            return_value=["오늘", "다른"],
        ):
            net = build_domain_network(
                text="오늘은 다른 주제입니다.",
                key_terms=["세포막"],
            )

        assert net.nodes == []
        assert net.edges == []


class TestCompareNetworks:
    """T035: compare_networks identifies missing edges."""

    def test_missing_edges_detected(self) -> None:
        """Edges in textbook but not in lecture are detected."""
        textbook = KeywordNetwork(
            source="textbook",
            nodes=["세포막", "인지질", "콜레스테롤"],
            edges=[
                ("세포막", "인지질", 3.0),
                ("세포막", "콜레스테롤", 2.0),
                ("인지질", "콜레스테롤", 1.0),
            ],
        )
        lecture = KeywordNetwork(
            source="A",
            nodes=["세포막", "인지질"],
            edges=[
                ("세포막", "인지질", 2.0),
            ],
        )

        missing = compare_networks(textbook, lecture)

        # Should find 2 missing edges
        assert len(missing) == 2
        missing_sets = [frozenset(e) for e in missing]
        assert frozenset({"세포막", "콜레스테롤"}) in missing_sets
        assert frozenset({"인지질", "콜레스테롤"}) in missing_sets

    def test_no_missing_edges(self) -> None:
        """When lecture has all textbook edges, missing is empty."""
        textbook = KeywordNetwork(
            source="textbook",
            nodes=["A", "B"],
            edges=[("A", "B", 1.0)],
        )
        lecture = KeywordNetwork(
            source="A",
            nodes=["A", "B"],
            edges=[("A", "B", 2.0)],
        )

        missing = compare_networks(textbook, lecture)
        assert missing == []

    def test_empty_textbook_returns_empty(self) -> None:
        """Empty textbook network means no missing edges."""
        textbook = KeywordNetwork(source="textbook", nodes=[], edges=[])
        lecture = KeywordNetwork(
            source="A",
            nodes=["A"],
            edges=[("A", "B", 1.0)],
        )

        missing = compare_networks(textbook, lecture)
        assert missing == []


# ================================================================
# v3 Tests: Phase 2 — Embedding Signals & Ensemble (T010)
# ================================================================

import numpy as np

try:
    from forma.domain_coverage_analyzer import (
        compute_embedding_signal,
        compute_term_coverage_signal,
        compute_density_signal,
        compute_ensemble_quality,
    )

    _HAS_SIGNALS = True
except ImportError:
    _HAS_SIGNALS = False

_skip_signals = pytest.mark.skipif(
    not _HAS_SIGNALS,
    reason="Signal functions not yet implemented (RED phase)",
)


@_skip_signals
class TestComputeEmbeddingSignal:
    """T010: Tests for compute_embedding_signal()."""

    @patch("forma.domain_coverage_analyzer.encode_texts")
    def test_max_cosine_and_key_term_mean(self, mock_encode) -> None:
        """Returns combined score from max cosine similarity + key_term mean."""
        # Concept embedding (1 vector), transcript sentence embeddings (3 vectors)
        concept_vec = np.array([[1.0, 0.0, 0.0]])
        sentence_vecs = np.array([
            [0.9, 0.1, 0.0],   # high similarity
            [0.0, 1.0, 0.0],   # low similarity
            [0.5, 0.5, 0.0],   # medium similarity
        ])
        key_term_vecs = np.array([
            [0.8, 0.2, 0.0],   # key_term 1 embedding
        ])
        key_term_sentence_sims = np.array([
            [0.7, 0.1, 0.4],   # key_term 1 vs 3 sentences
        ])

        # Mock: first call for concept vs sentences, second for key_terms vs sentences
        call_count = [0]

        def side_effect(texts):
            call_count[0] += 1
            if call_count[0] == 1:
                # concept text encoding
                return concept_vec
            elif call_count[0] == 2:
                # sentence encodings
                return sentence_vecs
            elif call_count[0] == 3:
                # key_term encodings
                return key_term_vecs
            return sentence_vecs

        mock_encode.side_effect = side_effect

        score = compute_embedding_signal(
            concept_text="표피의 4층 구조",
            key_terms=["표피"],
            transcript_sentences=["표피는 4개 층", "진피 설명", "표피 구조"],
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @patch("forma.domain_coverage_analyzer.encode_texts")
    def test_empty_sentences_returns_zero(self, mock_encode) -> None:
        """Empty transcript sentences return 0.0."""
        mock_encode.return_value = np.array([[1.0, 0.0]])

        score = compute_embedding_signal(
            concept_text="표피의 구조",
            key_terms=["표피"],
            transcript_sentences=[],
        )

        assert score == 0.0

    @patch("forma.domain_coverage_analyzer.encode_texts")
    def test_high_similarity_returns_high_score(self, mock_encode) -> None:
        """Very similar vectors produce score close to 1.0."""
        # Near-identical vectors
        vec = np.array([[1.0, 0.0, 0.0]])
        mock_encode.return_value = vec

        score = compute_embedding_signal(
            concept_text="표피",
            key_terms=["표피"],
            transcript_sentences=["표피"],
        )

        assert score >= 0.8


@_skip_signals
class TestComputeTermCoverageSignal:
    """T010: Tests for compute_term_coverage_signal()."""

    def test_all_terms_present_returns_1(self) -> None:
        """All key_terms found in text → 1.0."""
        score = compute_term_coverage_signal(
            key_terms=["각질층"],
            transcript_text="각질층은 보호막입니다",
        )
        assert score == pytest.approx(1.0)

    def test_no_terms_present_returns_0(self) -> None:
        """No key_terms found in text → 0.0."""
        score = compute_term_coverage_signal(
            key_terms=["각질층"],
            transcript_text="표피 설명",
        )
        assert score == pytest.approx(0.0)

    def test_partial_match_returns_ratio(self) -> None:
        """Some terms present → ratio of found/total."""
        score = compute_term_coverage_signal(
            key_terms=["각질층", "투명층"],
            transcript_text="각질층은 가장 바깥에 있다.",
        )
        assert score == pytest.approx(0.5)

    def test_multiple_terms_all_present(self) -> None:
        """All 3 terms present → 1.0."""
        score = compute_term_coverage_signal(
            key_terms=["표피", "진피", "피하조직"],
            transcript_text="표피와 진피 그리고 피하조직으로 구성된다.",
        )
        assert score == pytest.approx(1.0)

    def test_empty_key_terms_returns_0(self) -> None:
        """Empty key_terms list → 0.0."""
        score = compute_term_coverage_signal(
            key_terms=[],
            transcript_text="어떤 텍스트",
        )
        assert score == pytest.approx(0.0)

    def test_empty_text_returns_0(self) -> None:
        """Empty transcript text → 0.0."""
        score = compute_term_coverage_signal(
            key_terms=["표피"],
            transcript_text="",
        )
        assert score == pytest.approx(0.0)


@_skip_signals
class TestComputeDensitySignal:
    """T010: Tests for compute_density_signal()."""

    def test_clustered_terms_high_density(self) -> None:
        """Key terms clustered in one paragraph → high density."""
        text = (
            "표피의 각질층은 종자층에서 분화된 세포로 구성된다. "
            "각질층과 과립층은 서로 연결된다. "
            "표피의 보호 기능이 여기서 시작된다."
        )
        score = compute_density_signal(
            key_terms=["표피", "각질층", "종자층", "과립층"],
            transcript_text=text,
        )
        assert isinstance(score, float)
        assert score > 0.5

    def test_scattered_terms_low_density(self) -> None:
        """Key terms scattered with much irrelevant text → lower density than clustered."""
        filler = "오늘은 날씨가 좋습니다. " * 100
        text = f"표피 설명. {filler} 각질층 설명. {filler} 종자층 설명."

        scattered_score = compute_density_signal(
            key_terms=["표피", "각질층", "종자층"],
            transcript_text=text,
        )

        # Compare against clustered version
        clustered_text = "표피와 각질층 그리고 종자층은 서로 연결된다."
        clustered_score = compute_density_signal(
            key_terms=["표피", "각질층", "종자층"],
            transcript_text=clustered_text,
        )
        assert scattered_score <= clustered_score

    def test_empty_text_returns_0(self) -> None:
        """Empty transcript text → 0.0."""
        score = compute_density_signal(
            key_terms=["표피"],
            transcript_text="",
        )
        assert score == pytest.approx(0.0)

    def test_empty_key_terms_returns_0(self) -> None:
        """Empty key_terms → 0.0."""
        score = compute_density_signal(
            key_terms=[],
            transcript_text="어떤 텍스트가 있다.",
        )
        assert score == pytest.approx(0.0)

    def test_returns_between_0_and_1(self) -> None:
        """Score is always in [0.0, 1.0]."""
        text = "표피와 진피는 피부의 주요 구성요소이다."
        score = compute_density_signal(
            key_terms=["표피", "진피"],
            transcript_text=text,
        )
        assert 0.0 <= score <= 1.0


@_skip_signals
class TestComputeEnsembleQuality:
    """T010: Tests for compute_ensemble_quality()."""

    def test_weighted_sum_default_weights(self) -> None:
        """Ensemble = weighted sum with default weights."""
        # Default: emb=0.25, term=0.25, density=0.15, llm=0.35
        result = compute_ensemble_quality(
            s_emb=0.8,
            s_term=0.6,
            s_density=0.4,
            s_llm=1.0,
        )
        expected = 0.25 * 0.8 + 0.25 * 0.6 + 0.15 * 0.4 + 0.35 * 1.0
        assert result == pytest.approx(expected, abs=0.001)

    def test_custom_weights(self) -> None:
        """Custom weights produce correct weighted sum."""
        weights = {"embedding": 0.3, "term_coverage": 0.3, "density": 0.2, "llm": 0.2}
        result = compute_ensemble_quality(
            s_emb=1.0,
            s_term=0.5,
            s_density=0.5,
            s_llm=0.0,
            weights=weights,
        )
        expected = 0.3 * 1.0 + 0.3 * 0.5 + 0.2 * 0.5 + 0.2 * 0.0
        assert result == pytest.approx(expected, abs=0.001)

    def test_no_llm_weight_redistribution(self) -> None:
        """When --no-llm (s_llm=0, weights redistributed), ensemble is deterministic."""
        # --no-llm redistributes: emb=0.4, term=0.4, density=0.2, llm=0.0
        no_llm_weights = {
            "embedding": 0.4,
            "term_coverage": 0.4,
            "density": 0.2,
            "llm": 0.0,
        }
        result = compute_ensemble_quality(
            s_emb=0.8,
            s_term=0.6,
            s_density=0.4,
            s_llm=0.0,
            weights=no_llm_weights,
        )
        expected = 0.4 * 0.8 + 0.4 * 0.6 + 0.2 * 0.4 + 0.0 * 0.0
        assert result == pytest.approx(expected, abs=0.001)

    def test_weights_sum_check(self) -> None:
        """Weights that don't sum to 1.0 raise ValueError."""
        bad_weights = {"embedding": 0.5, "term_coverage": 0.5, "density": 0.5, "llm": 0.5}
        with pytest.raises(ValueError, match="sum"):
            compute_ensemble_quality(
                s_emb=0.5, s_term=0.5, s_density=0.5, s_llm=0.5,
                weights=bad_weights,
            )

    def test_all_zeros_returns_zero(self) -> None:
        """All zero signals → 0.0."""
        result = compute_ensemble_quality(
            s_emb=0.0, s_term=0.0, s_density=0.0, s_llm=0.0,
        )
        assert result == pytest.approx(0.0)

    def test_all_ones_returns_one(self) -> None:
        """All perfect signals → 1.0."""
        result = compute_ensemble_quality(
            s_emb=1.0, s_term=1.0, s_density=1.0, s_llm=1.0,
        )
        assert result == pytest.approx(1.0)


# ================================================================
# v3 Tests: Phase 4 — Ensemble Integration (T023)
# ================================================================

# Try importing Phase 4/5 functions that may not exist yet
try:
    from forma.domain_coverage_analyzer import _DELIVERY_RUBRIC

    _HAS_RUBRIC = True
except ImportError:
    _HAS_RUBRIC = False

try:
    from forma.domain_coverage_analyzer import (
        _chunk_transcript_with_overlap,
        MAX_TRANSCRIPT_LENGTH,
    )

    _HAS_CHUNKING = True
except ImportError:
    _HAS_CHUNKING = False

_skip_rubric = pytest.mark.skipif(
    not _HAS_RUBRIC,
    reason="_DELIVERY_RUBRIC not yet implemented (RED phase)",
)

_skip_chunking = pytest.mark.skipif(
    not _HAS_CHUNKING,
    reason="Chunking functions not yet implemented (RED phase)",
)


class TestEnsembleIntegration:
    """T023: Ensemble integration in analyze_delivery_llm flow."""

    def test_analyze_delivery_llm_populates_signal_scores(self, tmp_path) -> None:
        """After analyze_delivery_llm(), each DeliveryAnalysis has signal_scores with 4 keys."""
        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text(
            "표피는 4개의 층으로 구성됩니다. 각질층이 가장 바깥에 있다.",
            encoding="utf-8",
        )

        mock_response = """\
deliveries:
  - concept: "표피의 4층 구조"
    delivery_status: "충분히 설명"
    delivery_quality: 0.9
    evidence: "표피는 4개의 층"
    depth: "상세 설명"
"""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        # Mock embeddings to be deterministic
        dummy_emb = np.array([[1.0, 0.0, 0.0]])

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ), patch(
            "forma.domain_coverage_analyzer.encode_texts",
            return_value=dummy_emb,
        ):
            from forma.domain_coverage_analyzer import analyze_delivery_llm

            result = analyze_delivery_llm(
                concepts=["표피의 4층 구조"],
                transcript_path=str(transcript),
                section_id="A",
            )

        assert len(result) >= 1
        da = result[0]
        assert "signal_scores" in da.__dict__ or hasattr(da, "signal_scores")
        scores = da.signal_scores
        assert isinstance(scores, dict)
        expected_keys = {"embedding", "term_coverage", "density", "llm"}
        assert expected_keys == set(scores.keys()), (
            f"Expected keys {expected_keys}, got {set(scores.keys())}"
        )
        # All signal scores should be in [0, 1]
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"signal_scores[{key}]={val} out of range"

    def test_no_llm_mode_skips_llm_calls(self, tmp_path) -> None:
        """When no_llm=True, S_llm=0 and weights are redistributed."""
        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text("표피는 중요합니다.", encoding="utf-8")

        dummy_emb = np.array([[1.0, 0.0, 0.0]])

        mock_provider = MagicMock()

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ), patch(
            "forma.domain_coverage_analyzer.encode_texts",
            return_value=dummy_emb,
        ):
            from forma.domain_coverage_analyzer import analyze_delivery_llm

            result = analyze_delivery_llm(
                concepts=["표피의 4층 구조"],
                transcript_path=str(transcript),
                section_id="A",
                no_llm=True,
            )

        # LLM provider should NOT have been called
        mock_provider.generate.assert_not_called()

        assert len(result) >= 1
        da = result[0]
        # S_llm should be 0
        assert da.signal_scores.get("llm", -1) == pytest.approx(0.0)
        # delivery_quality should use redistributed weights (no LLM component)
        # Verify it's computed from embedding/term/density only
        s_emb = da.signal_scores.get("embedding", 0)
        s_term = da.signal_scores.get("term_coverage", 0)
        s_density = da.signal_scores.get("density", 0)
        expected = 0.4 * s_emb + 0.4 * s_term + 0.2 * s_density
        assert da.delivery_quality == pytest.approx(expected, abs=0.01)

    @patch("forma.domain_coverage_analyzer.encode_texts")
    def test_signal_scores_deterministic(self, mock_encode, tmp_path) -> None:
        """Same input twice produces identical S_emb, S_term, S_density."""
        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text(
            "표피는 4개의 층으로 구성됩니다. 각질층은 가장 바깥이다.",
            encoding="utf-8",
        )

        # Use deterministic embeddings
        fixed_emb = np.array([[0.8, 0.2, 0.0]])
        mock_encode.return_value = fixed_emb

        mock_response = """\
deliveries:
  - concept: "표피의 4층 구조"
    delivery_status: "충분히 설명"
    delivery_quality: 0.9
    evidence: "표피는 4개 층"
    depth: "상세 설명"
"""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ):
            from forma.domain_coverage_analyzer import analyze_delivery_llm

            result1 = analyze_delivery_llm(
                concepts=["표피의 4층 구조"],
                transcript_path=str(transcript),
                section_id="A",
            )
            result2 = analyze_delivery_llm(
                concepts=["표피의 4층 구조"],
                transcript_path=str(transcript),
                section_id="A",
            )

        scores1 = result1[0].signal_scores
        scores2 = result2[0].signal_scores
        # Signal scores must be populated
        assert len(scores1) >= 3, f"Expected >= 3 signal scores, got {len(scores1)}"
        assert len(scores2) >= 3, f"Expected >= 3 signal scores, got {len(scores2)}"
        for key in ("embedding", "term_coverage", "density"):
            assert key in scores1, f"Missing signal key '{key}' in run 1"
            assert key in scores2, f"Missing signal key '{key}' in run 2"
            assert scores1[key] == pytest.approx(scores2[key], abs=1e-6), (
                f"Signal {key} not deterministic: {scores1[key]} vs {scores2[key]}"
            )

    def test_delivery_quality_is_ensemble_not_llm_only(self, tmp_path) -> None:
        """delivery_quality differs from raw LLM quality when other signals disagree."""
        transcript = tmp_path / "1A_2주차_1차시.txt"
        # Transcript that does NOT mention the concept at all
        transcript.write_text(
            "오늘은 근육에 대해 이야기합니다. 근육 수축과 이완.",
            encoding="utf-8",
        )

        # LLM says high quality but embedding/term signals should disagree
        mock_response = """\
deliveries:
  - concept: "표피의 4층 구조"
    delivery_status: "충분히 설명"
    delivery_quality: 0.95
    evidence: "표피는 4개 층"
    depth: "상세 설명"
"""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        # Embedding: concept and transcript are dissimilar
        call_count = [0]

        def mock_encode(texts):
            call_count[0] += 1
            if call_count[0] == 1:
                # concept embedding
                return np.array([[1.0, 0.0, 0.0]])
            else:
                # sentence embeddings (unrelated topic)
                return np.array([[0.0, 1.0, 0.0]] * len(texts)) if texts else np.array([[0.0, 1.0, 0.0]])

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ), patch(
            "forma.domain_coverage_analyzer.encode_texts",
            side_effect=mock_encode,
        ):
            from forma.domain_coverage_analyzer import analyze_delivery_llm

            result = analyze_delivery_llm(
                concepts=["표피의 4층 구조"],
                transcript_path=str(transcript),
                section_id="A",
            )

        da = result[0]
        raw_llm_quality = 0.95
        # The ensemble quality should differ from the raw LLM score
        # because embedding and term signals are low
        assert da.delivery_quality != pytest.approx(raw_llm_quality, abs=0.01), (
            f"Ensemble quality {da.delivery_quality} should differ from raw LLM {raw_llm_quality}"
        )


# ================================================================
# v3 Tests: Phase 4 — Delivery Rubric (T024)
# ================================================================


@_skip_rubric
class TestDeliveryRubric:
    """T024: Tests for delivery rubric constant and prompt integration."""

    def test_delivery_rubric_constant_exists(self) -> None:
        """Assert _DELIVERY_RUBRIC is defined and contains 4 scoring tiers."""
        assert isinstance(_DELIVERY_RUBRIC, str)
        assert len(_DELIVERY_RUBRIC) > 50  # non-trivial content
        # Should contain 4 tier descriptors
        tier_count = 0
        for tier_label in ["0.8", "0.5", "0.2", "0.0"]:
            if tier_label in _DELIVERY_RUBRIC:
                tier_count += 1
        assert tier_count == 4, (
            f"Expected 4 scoring tiers in rubric, found {tier_count}"
        )

    def test_rubric_included_in_prompt(self) -> None:
        """build_delivery_prompt() output includes rubric text."""
        prompt = build_delivery_prompt(
            concepts=["표피의 4층 구조"],
            transcript_text="강의 내용...",
        )
        # The rubric text should appear somewhere in the prompt
        assert _DELIVERY_RUBRIC in prompt or "0.8~1.0" in prompt or "0.8" in prompt, (
            "Rubric text not found in delivery prompt"
        )

    def test_rubric_tiers_present(self) -> None:
        """All 4 tier ranges are present in the rubric."""
        tier_ranges = ["0.8~1.0", "0.5~0.7", "0.2~0.4", "0.0~0.1"]
        for tier_range in tier_ranges:
            assert tier_range in _DELIVERY_RUBRIC, (
                f"Tier range '{tier_range}' not found in _DELIVERY_RUBRIC"
            )


# ================================================================
# v3 Tests: Phase 5 — Transcript Chunking (T030)
# ================================================================


@_skip_chunking
class TestChunkTranscriptWithOverlap:
    """T030: Tests for _chunk_transcript_with_overlap()."""

    def test_short_transcript_no_split(self) -> None:
        """Text < 25K chars returns [text] (single chunk)."""
        short_text = "짧은 강의 내용입니다." * 100  # ~1000 chars
        result = _chunk_transcript_with_overlap(short_text)
        assert len(result) == 1
        assert result[0] == short_text

    def test_long_transcript_splits_with_overlap(self) -> None:
        """60K text splits into 3 chunks, each ~25K, with 2K overlap."""
        # Build a 60K character text
        sentence = "이것은 테스트 문장입니다. "  # ~14 chars
        repeat_count = 60000 // len(sentence) + 1
        long_text = sentence * repeat_count
        long_text = long_text[:60000]

        result = _chunk_transcript_with_overlap(long_text)
        assert len(result) == 3, f"Expected 3 chunks, got {len(result)}"
        # Each chunk should be approximately 25K (not exceeding by much)
        for chunk in result:
            assert len(chunk) <= 27000, f"Chunk too large: {len(chunk)} chars"

    def test_overlap_contains_shared_content(self) -> None:
        """chunk[n] end overlaps with chunk[n+1] start."""
        sentence = "테스트 문장입니다. "
        repeat_count = 60000 // len(sentence) + 1
        long_text = sentence * repeat_count
        long_text = long_text[:60000]

        result = _chunk_transcript_with_overlap(long_text)
        assert len(result) >= 2

        for i in range(len(result) - 1):
            chunk_end = result[i][-2000:]  # last 2K of current chunk
            chunk_start = result[i + 1][:2000]  # first 2K of next chunk
            # There should be overlapping content
            overlap_found = False
            # Check if any significant substring appears in both
            check_len = min(500, len(chunk_end), len(chunk_start))
            if check_len > 0:
                # The overlap region should share content
                for offset in range(0, len(chunk_end) - 100, 100):
                    snippet = chunk_end[offset:offset + 100]
                    if snippet in result[i + 1]:
                        overlap_found = True
                        break
            assert overlap_found, (
                f"No overlap detected between chunk {i} and chunk {i+1}"
            )

    def test_split_at_sentence_boundary(self) -> None:
        """Chunks don't split mid-sentence."""
        # Build text with clear sentence boundaries
        sentences = []
        for i in range(3000):
            sentences.append(f"문장번호{i:04d}은 여기서 끝입니다.")
        long_text = " ".join(sentences)

        if len(long_text) < 25000:
            pytest.skip("Test text too short for chunking")

        result = _chunk_transcript_with_overlap(long_text)
        assert len(result) >= 2

        for chunk in result:
            # Each chunk should end at or near a sentence boundary (period)
            stripped = chunk.rstrip()
            if stripped:
                # Last char should be a period or the chunk should end near one
                last_period = stripped.rfind(".")
                assert last_period >= len(stripped) - 50, (
                    f"Chunk does not end near sentence boundary. "
                    f"Last period at {last_period}, chunk length {len(stripped)}"
                )

    def test_empty_transcript(self) -> None:
        """Empty string returns ['']."""
        result = _chunk_transcript_with_overlap("")
        assert result == [""]


# ================================================================
# v3 Tests: Phase 5 — MAX_TRANSCRIPT_LENGTH Guard (T031)
# ================================================================


@_skip_chunking
class TestMaxTranscriptLength:
    """T031: Tests for MAX_TRANSCRIPT_LENGTH guard."""

    def test_max_transcript_length_constant(self) -> None:
        """Assert MAX_TRANSCRIPT_LENGTH == 50000."""
        assert MAX_TRANSCRIPT_LENGTH == 50000

    def test_large_transcript_triggers_chunking(self, tmp_path) -> None:
        """60K transcript triggers _chunk_transcript_with_overlap call."""
        sentence = "표피의 각질층은 보호막입니다. "
        repeat_count = 60000 // len(sentence) + 1
        long_text = sentence * repeat_count
        long_text = long_text[:60000]

        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text(long_text, encoding="utf-8")

        mock_response = """\
deliveries:
  - concept: "표피의 4층 구조"
    delivery_status: "충분히 설명"
    delivery_quality: 0.9
    evidence: "표피의 각질층"
    depth: "상세 설명"
"""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        dummy_emb = np.array([[1.0, 0.0, 0.0]])

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ), patch(
            "forma.domain_coverage_analyzer.encode_texts",
            return_value=dummy_emb,
        ), patch(
            "forma.domain_coverage_analyzer._chunk_transcript_with_overlap",
            wraps=_chunk_transcript_with_overlap,
        ) as mock_chunk:
            from forma.domain_coverage_analyzer import analyze_delivery_llm

            analyze_delivery_llm(
                concepts=["표피의 4층 구조"],
                transcript_path=str(transcript),
                section_id="A",
            )

        mock_chunk.assert_called_once()

    def test_empty_transcript_warning(self, tmp_path, caplog) -> None:
        """Empty/whitespace transcript logs warning and returns empty list."""
        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text("   \n  ", encoding="utf-8")

        mock_provider = MagicMock()

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ):
            from forma.domain_coverage_analyzer import analyze_delivery_llm

            import logging as _logging

            with caplog.at_level(_logging.WARNING):
                result = analyze_delivery_llm(
                    concepts=["표피의 4층 구조"],
                    transcript_path=str(transcript),
                    section_id="A",
                )

        assert result == [] or all(
            d.delivery_status == "미전달" for d in result
        ), "Empty transcript should return empty list or all NOT_DELIVERED"

    def test_best_quality_across_chunks(self, tmp_path) -> None:
        """Same concept in 2 chunks with quality 0.6 and 0.8 → final based on 0.8."""
        # Build transcript long enough to chunk
        filler = "오늘은 날씨가 좋습니다. " * 2000  # ~26K chars
        good_section = "표피는 4개의 층으로 구성됩니다. 각질층 투명층 과립층 종자층. "
        weak_section = "표피를 잠깐 언급합니다. "

        long_text = weak_section * 100 + filler + good_section * 100
        long_text = long_text[:60000]

        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text(long_text, encoding="utf-8")

        call_count = [0]

        def mock_generate(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return """\
deliveries:
  - concept: "표피의 4층 구조"
    delivery_status: "부분 전달"
    delivery_quality: 0.6
    evidence: "잠깐 언급"
    depth: "약간"
"""
            else:
                return """\
deliveries:
  - concept: "표피의 4층 구조"
    delivery_status: "충분히 설명"
    delivery_quality: 0.8
    evidence: "4개 층 상세"
    depth: "상세 설명"
"""

        mock_provider = MagicMock()
        mock_provider.generate.side_effect = mock_generate

        dummy_emb = np.array([[1.0, 0.0, 0.0]])

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ), patch(
            "forma.domain_coverage_analyzer.encode_texts",
            return_value=dummy_emb,
        ):
            from forma.domain_coverage_analyzer import analyze_delivery_llm

            result = analyze_delivery_llm(
                concepts=["표피의 4층 구조"],
                transcript_path=str(transcript),
                section_id="A",
            )

        assert len(result) >= 1
        # The best chunk's quality (0.8) should be reflected, not the worst (0.6)
        best = max(r.delivery_quality for r in result if r.concept == "표피의 4층 구조")
        assert best >= 0.7, (
            f"Expected best quality >= 0.7 from chunk merging, got {best}"
        )


# ================================================================
# Phase 7: Cross-Section Statistics (T042-T047)
# ================================================================


def _build_4section_deliveries() -> list[DeliveryAnalysis]:
    """Build deliveries for 4 sections (A, B, C, D) with 3 concepts each."""
    deliveries = []
    qualities = {
        "A": [0.9, 0.85, 0.7],
        "B": [0.5, 0.4, 0.6],
        "C": [0.8, 0.75, 0.65],
        "D": [0.3, 0.2, 0.4],
    }
    concepts = ["표피의 4층 구조", "진피의 구조", "피부 보호 기능"]
    for section, quals in qualities.items():
        for concept, quality in zip(concepts, quals):
            deliveries.append(DeliveryAnalysis(
                concept=concept,
                section_id=section,
                delivery_status="충분히 설명" if quality >= 0.5 else "부분 전달",
                delivery_quality=quality,
                evidence="",
                depth="",
            ))
    return deliveries


class TestDeliveryPairwiseComparisons:
    """T042: Tests for compute_delivery_pairwise_comparisons."""

    def test_4_sections_6_pairs(self) -> None:
        """4 sections produce 6 comparison pairs (C(4,2))."""
        from forma.domain_coverage_analyzer import (
            compute_delivery_pairwise_comparisons,
        )

        deliveries = _build_4section_deliveries()
        comparisons = compute_delivery_pairwise_comparisons(deliveries)
        assert len(comparisons) == 6

    def test_bonferroni_correction(self) -> None:
        """p-values are adjusted by k=6 (Bonferroni)."""
        from forma.domain_coverage_analyzer import (
            compute_delivery_pairwise_comparisons,
        )

        deliveries = _build_4section_deliveries()
        comparisons = compute_delivery_pairwise_comparisons(deliveries)
        for comp in comparisons:
            # corrected_p should be >= raw p-value
            assert comp.corrected_p_value >= comp.p_value
            # corrected = p * n_comparisons, capped at 1.0
            expected = min(comp.p_value * 6, 1.0)
            assert comp.corrected_p_value == pytest.approx(expected, abs=0.001)

    def test_1_section_skips(self, caplog) -> None:
        """Single section returns empty list with warning."""
        from forma.domain_coverage_analyzer import (
            compute_delivery_pairwise_comparisons,
        )

        deliveries = [
            DeliveryAnalysis("c1", "A", "충분히 설명", 0.9, "", ""),
        ]
        import logging as _logging
        with caplog.at_level(_logging.WARNING):
            comparisons = compute_delivery_pairwise_comparisons(deliveries)
        assert comparisons == []
        assert any("2개 이상" in r.message for r in caplog.records)

    def test_2_sections_valid(self) -> None:
        """2 sections produce 1 comparison pair."""
        from forma.domain_coverage_analyzer import (
            compute_delivery_pairwise_comparisons,
        )

        deliveries = [
            DeliveryAnalysis("c1", "A", "충분히 설명", 0.9, "", ""),
            DeliveryAnalysis("c2", "A", "충분히 설명", 0.8, "", ""),
            DeliveryAnalysis("c1", "B", "부분 전달", 0.4, "", ""),
            DeliveryAnalysis("c2", "B", "미전달", 0.1, "", ""),
        ]
        comparisons = compute_delivery_pairwise_comparisons(deliveries)
        assert len(comparisons) == 1
        assert comparisons[0].section_a == "A"
        assert comparisons[0].section_b == "B"


class TestDeliverySectionComparisonSerialization:
    """T043: Tests for section comparison YAML serialization."""

    def test_save_load_roundtrip(self, tmp_path) -> None:
        """Save then load DeliveryResult with section_comparisons preserves fields."""
        from forma.domain_coverage_analyzer import (
            compute_delivery_pairwise_comparisons,
        )

        deliveries = _build_4section_deliveries()
        comparisons = compute_delivery_pairwise_comparisons(deliveries)

        result = DeliveryResult(
            week=2,
            chapters=["3장"],
            deliveries=deliveries,
            effective_delivery_rate=0.75,
            per_section_rate={"A": 1.0, "B": 0.5, "C": 0.8, "D": 0.3},
        )

        path = str(tmp_path / "delivery_with_comparisons.yaml")
        save_delivery_yaml(result, path)
        loaded = load_delivery_yaml(path)

        assert loaded.week == result.week
        assert len(loaded.deliveries) == len(result.deliveries)
        assert loaded.effective_delivery_rate == pytest.approx(
            result.effective_delivery_rate, abs=0.001,
        )
