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
