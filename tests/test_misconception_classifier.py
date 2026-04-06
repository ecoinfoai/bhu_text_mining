"""Tests for misconception_classifier.py — 4-pattern misconception classifier.

RED phase: these tests are written *before* the module exists and will fail
until ``src/forma/misconception_classifier.py`` is implemented.

Covers US1 (FR-001 ~ FR-006, SC-001):
  - MisconceptionPattern enum: 4 values
  - ClassifiedMisconception dataclass: 6 fields
  - classify_misconceptions(): pattern detection from GraphComparisonResult + HubGapEntry
  - aggregate_class_misconceptions(): class-level aggregation
"""

from __future__ import annotations

import pytest

from forma.evaluation_types import (
    GraphComparisonResult,
    HubGapEntry,
    TripletEdge,
)
from forma.misconception_classifier import (
    DEFAULT_INCLUSION_KEYWORDS,
    ClassifiedMisconception,
    MisconceptionPattern,
    aggregate_class_misconceptions,
    classify_misconceptions,
)


# ---------------------------------------------------------------------------
# FR-001: MisconceptionPattern enum has exactly 4 values
# ---------------------------------------------------------------------------


class TestMisconceptionPatternEnum:
    """FR-001: MisconceptionPattern enum has 4 values."""

    def test_enum_has_four_values(self):
        assert len(MisconceptionPattern) == 4

    def test_enum_values(self):
        expected = {
            "INCLUSION_ERROR",
            "CAUSAL_REVERSAL",
            "RELATION_CONFUSION",
            "CONCEPT_ABSENCE",
        }
        assert {p.value for p in MisconceptionPattern} == expected

    def test_enum_access_by_name(self):
        assert MisconceptionPattern.INCLUSION_ERROR.value == "INCLUSION_ERROR"
        assert MisconceptionPattern.CAUSAL_REVERSAL.value == "CAUSAL_REVERSAL"
        assert MisconceptionPattern.RELATION_CONFUSION.value == "RELATION_CONFUSION"
        assert MisconceptionPattern.CONCEPT_ABSENCE.value == "CONCEPT_ABSENCE"


# ---------------------------------------------------------------------------
# FR-002: ClassifiedMisconception dataclass
# ---------------------------------------------------------------------------


class TestClassifiedMisconception:
    """FR-002: ClassifiedMisconception has correct fields."""

    def test_fields_exist(self):
        cm = ClassifiedMisconception(
            pattern=MisconceptionPattern.CAUSAL_REVERSAL,
            master_edge=TripletEdge("A", "causes", "B"),
            student_edge=TripletEdge("B", "causes", "A"),
            concept=None,
            confidence=0.85,
            description="Reversed causal direction",
        )
        assert cm.pattern == MisconceptionPattern.CAUSAL_REVERSAL
        assert cm.master_edge == TripletEdge("A", "causes", "B")
        assert cm.student_edge == TripletEdge("B", "causes", "A")
        assert cm.concept is None
        assert cm.confidence == 0.85
        assert cm.description == "Reversed causal direction"

    def test_concept_absence_fields(self):
        cm = ClassifiedMisconception(
            pattern=MisconceptionPattern.CONCEPT_ABSENCE,
            master_edge=None,
            student_edge=None,
            concept="항상성",
            confidence=0.75,
            description="Missing concept: 항상성",
        )
        assert cm.pattern == MisconceptionPattern.CONCEPT_ABSENCE
        assert cm.concept == "항상성"
        assert cm.master_edge is None

    def test_confidence_range(self):
        cm = ClassifiedMisconception(
            pattern=MisconceptionPattern.CAUSAL_REVERSAL,
            master_edge=None,
            student_edge=None,
            concept=None,
            confidence=0.0,
            description="test",
        )
        assert 0.0 <= cm.confidence <= 1.0

        cm2 = ClassifiedMisconception(
            pattern=MisconceptionPattern.CAUSAL_REVERSAL,
            master_edge=None,
            student_edge=None,
            concept=None,
            confidence=1.0,
            description="test",
        )
        assert 0.0 <= cm2.confidence <= 1.0


# ---------------------------------------------------------------------------
# FR-003: CAUSAL_REVERSAL classification
# ---------------------------------------------------------------------------


class TestCausalReversal:
    """FR-003: wrong_direction_edges without inclusion keywords → CAUSAL_REVERSAL."""

    def test_basic_causal_reversal(self):
        """Wrong direction edge with non-inclusion relation → CAUSAL_REVERSAL."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[],
            extra_edges=[],
            wrong_direction_edges=[TripletEdge("수용체", "활성화", "신호물질")],
        )
        results = classify_misconceptions(graph_result, [])
        causal = [r for r in results if r.pattern == MisconceptionPattern.CAUSAL_REVERSAL]
        assert len(causal) == 1
        assert causal[0].confidence == pytest.approx(0.85)
        assert causal[0].student_edge == TripletEdge("수용체", "활성화", "신호물질")

    def test_multiple_causal_reversals(self):
        """Multiple wrong direction edges → multiple CAUSAL_REVERSAL."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[],
            extra_edges=[],
            wrong_direction_edges=[
                TripletEdge("A", "causes", "B"),
                TripletEdge("C", "leads_to", "D"),
            ],
        )
        results = classify_misconceptions(graph_result, [])
        causal = [r for r in results if r.pattern == MisconceptionPattern.CAUSAL_REVERSAL]
        assert len(causal) == 2


# ---------------------------------------------------------------------------
# FR-004: INCLUSION_ERROR classification (priority over CAUSAL_REVERSAL)
# ---------------------------------------------------------------------------


class TestInclusionError:
    """FR-004: wrong_direction_edges with inclusion keywords → INCLUSION_ERROR."""

    def test_inclusion_keyword_포함(self):
        """Wrong direction edge with '포함' in relation → INCLUSION_ERROR."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[],
            extra_edges=[],
            wrong_direction_edges=[TripletEdge("세포", "포함", "세포막")],
        )
        results = classify_misconceptions(graph_result, [])
        inclusion = [r for r in results if r.pattern == MisconceptionPattern.INCLUSION_ERROR]
        assert len(inclusion) == 1
        assert inclusion[0].confidence == pytest.approx(0.9)

    def test_inclusion_keyword_is_a(self):
        """Wrong direction edge with 'is-a' in relation → INCLUSION_ERROR."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[],
            extra_edges=[],
            wrong_direction_edges=[TripletEdge("포유류", "is-a", "동물")],
        )
        results = classify_misconceptions(graph_result, [])
        inclusion = [r for r in results if r.pattern == MisconceptionPattern.INCLUSION_ERROR]
        assert len(inclusion) == 1

    def test_inclusion_keyword_속함(self):
        """Wrong direction edge with '속함' in relation → INCLUSION_ERROR."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[],
            extra_edges=[],
            wrong_direction_edges=[TripletEdge("A", "속함", "B")],
        )
        results = classify_misconceptions(graph_result, [])
        inclusion = [r for r in results if r.pattern == MisconceptionPattern.INCLUSION_ERROR]
        assert len(inclusion) == 1

    def test_inclusion_priority_over_causal(self):
        """FR-005: INCLUSION_ERROR takes priority over CAUSAL_REVERSAL for same edge."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[],
            extra_edges=[],
            wrong_direction_edges=[TripletEdge("세포", "포함", "세포막")],
        )
        results = classify_misconceptions(graph_result, [])
        # Should be classified as INCLUSION_ERROR, NOT CAUSAL_REVERSAL
        causal = [r for r in results if r.pattern == MisconceptionPattern.CAUSAL_REVERSAL]
        inclusion = [r for r in results if r.pattern == MisconceptionPattern.INCLUSION_ERROR]
        assert len(inclusion) == 1
        assert len(causal) == 0

    def test_all_default_keywords_recognized(self):
        """All DEFAULT_INCLUSION_KEYWORDS should trigger INCLUSION_ERROR."""
        for kw in DEFAULT_INCLUSION_KEYWORDS:
            graph_result = GraphComparisonResult(
                student_id="s001",
                question_sn=1,
                precision=0.5,
                recall=0.5,
                f1=0.5,
                matched_edges=[],
                missing_edges=[],
                extra_edges=[],
                wrong_direction_edges=[TripletEdge("A", kw, "B")],
            )
            results = classify_misconceptions(graph_result, [])
            inclusion = [r for r in results if r.pattern == MisconceptionPattern.INCLUSION_ERROR]
            assert len(inclusion) == 1, f"Keyword '{kw}' not recognized as inclusion"


# ---------------------------------------------------------------------------
# FR-005: RELATION_CONFUSION classification
# ---------------------------------------------------------------------------


class TestRelationConfusion:
    """FR-005: extra_edges with different relation from matching missing_edges."""

    def test_basic_relation_confusion(self):
        """Extra edge same (S,O) as missing but different relation → RELATION_CONFUSION."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[TripletEdge("효소", "촉진", "반응")],
            extra_edges=[TripletEdge("효소", "억제", "반응")],
            wrong_direction_edges=[],
        )
        results = classify_misconceptions(graph_result, [])
        confused = [r for r in results if r.pattern == MisconceptionPattern.RELATION_CONFUSION]
        assert len(confused) == 1
        assert confused[0].confidence == pytest.approx(0.7)
        assert confused[0].master_edge == TripletEdge("효소", "촉진", "반응")
        assert confused[0].student_edge == TripletEdge("효소", "억제", "반응")

    def test_no_confusion_when_same_relation(self):
        """Extra edge with same relation as missing → NOT relation confusion."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[TripletEdge("효소", "촉진", "반응")],
            extra_edges=[TripletEdge("효소", "촉진", "반응")],
            wrong_direction_edges=[],
        )
        results = classify_misconceptions(graph_result, [])
        confused = [r for r in results if r.pattern == MisconceptionPattern.RELATION_CONFUSION]
        assert len(confused) == 0

    def test_no_confusion_different_subject_object(self):
        """Extra edge with different (S,O) → NOT relation confusion."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[TripletEdge("효소", "촉진", "반응")],
            extra_edges=[TripletEdge("호르몬", "억제", "분비")],
            wrong_direction_edges=[],
        )
        results = classify_misconceptions(graph_result, [])
        confused = [r for r in results if r.pattern == MisconceptionPattern.RELATION_CONFUSION]
        assert len(confused) == 0


# ---------------------------------------------------------------------------
# FR-006: CONCEPT_ABSENCE classification
# ---------------------------------------------------------------------------


class TestConceptAbsence:
    """FR-006: HubGapEntry with student_present=False → CONCEPT_ABSENCE."""

    def test_basic_concept_absence(self):
        """HubGapEntry.student_present=False → CONCEPT_ABSENCE, conf=degree_centrality."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[],
            extra_edges=[],
        )
        hub_gaps = [
            HubGapEntry(concept="항상성", degree_centrality=0.75, student_present=False),
        ]
        results = classify_misconceptions(graph_result, hub_gaps)
        absent = [r for r in results if r.pattern == MisconceptionPattern.CONCEPT_ABSENCE]
        assert len(absent) == 1
        assert absent[0].concept == "항상성"
        assert absent[0].confidence == pytest.approx(0.75)

    def test_present_concept_not_absent(self):
        """HubGapEntry.student_present=True → NOT classified as CONCEPT_ABSENCE."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[],
            extra_edges=[],
        )
        hub_gaps = [
            HubGapEntry(concept="항상성", degree_centrality=0.75, student_present=True),
        ]
        results = classify_misconceptions(graph_result, hub_gaps)
        absent = [r for r in results if r.pattern == MisconceptionPattern.CONCEPT_ABSENCE]
        assert len(absent) == 0

    def test_multiple_absent_concepts(self):
        """Multiple absent hub concepts → multiple CONCEPT_ABSENCE."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[],
            extra_edges=[],
        )
        hub_gaps = [
            HubGapEntry(concept="항상성", degree_centrality=0.75, student_present=False),
            HubGapEntry(concept="음성되먹임", degree_centrality=0.60, student_present=False),
            HubGapEntry(concept="체온조절", degree_centrality=0.40, student_present=True),
        ]
        results = classify_misconceptions(graph_result, hub_gaps)
        absent = [r for r in results if r.pattern == MisconceptionPattern.CONCEPT_ABSENCE]
        assert len(absent) == 2


# ---------------------------------------------------------------------------
# SC-001: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """SC-001: Edge cases and multi-pattern scenarios."""

    def test_empty_input(self):
        """Empty GraphComparisonResult and no hub gaps → empty list."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            matched_edges=[TripletEdge("A", "r", "B")],
            missing_edges=[],
            extra_edges=[],
            wrong_direction_edges=[],
        )
        results = classify_misconceptions(graph_result, [])
        assert results == []

    def test_multiple_patterns_simultaneously(self):
        """Multiple pattern types in same graph → each classified independently."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.3,
            recall=0.3,
            f1=0.3,
            matched_edges=[],
            missing_edges=[TripletEdge("효소", "촉진", "반응")],
            extra_edges=[TripletEdge("효소", "억제", "반응")],
            wrong_direction_edges=[TripletEdge("수용체", "활성화", "신호")],
        )
        hub_gaps = [
            HubGapEntry(concept="항상성", degree_centrality=0.75, student_present=False),
        ]
        results = classify_misconceptions(graph_result, hub_gaps)
        patterns = {r.pattern for r in results}
        assert MisconceptionPattern.CAUSAL_REVERSAL in patterns
        assert MisconceptionPattern.RELATION_CONFUSION in patterns
        assert MisconceptionPattern.CONCEPT_ABSENCE in patterns

    def test_custom_inclusion_keywords(self):
        """Custom inclusion_keywords override defaults."""
        graph_result = GraphComparisonResult(
            student_id="s001",
            question_sn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            matched_edges=[],
            missing_edges=[],
            extra_edges=[],
            wrong_direction_edges=[TripletEdge("A", "custom_relation", "B")],
        )
        # Default keywords → should be CAUSAL_REVERSAL
        results = classify_misconceptions(graph_result, [])
        causal = [r for r in results if r.pattern == MisconceptionPattern.CAUSAL_REVERSAL]
        assert len(causal) == 1

        # Custom keyword including "custom_relation" → should be INCLUSION_ERROR
        results2 = classify_misconceptions(graph_result, [], inclusion_keywords=["custom_relation"])
        inclusion = [r for r in results2 if r.pattern == MisconceptionPattern.INCLUSION_ERROR]
        assert len(inclusion) == 1
        causal2 = [r for r in results2 if r.pattern == MisconceptionPattern.CAUSAL_REVERSAL]
        assert len(causal2) == 0


# ---------------------------------------------------------------------------
# aggregate_class_misconceptions
# ---------------------------------------------------------------------------


class TestAggregateClassMisconceptions:
    """Tests for aggregate_class_misconceptions()."""

    def test_basic_aggregation(self):
        """Aggregate misconceptions across students → (pattern, description, count)."""
        student_misc = {
            "s001": [
                ClassifiedMisconception(
                    pattern=MisconceptionPattern.CAUSAL_REVERSAL,
                    master_edge=None,
                    student_edge=TripletEdge("A", "causes", "B"),
                    concept=None,
                    confidence=0.85,
                    description="Reversed: A→B",
                ),
            ],
            "s002": [
                ClassifiedMisconception(
                    pattern=MisconceptionPattern.CAUSAL_REVERSAL,
                    master_edge=None,
                    student_edge=TripletEdge("A", "causes", "B"),
                    concept=None,
                    confidence=0.85,
                    description="Reversed: A→B",
                ),
                ClassifiedMisconception(
                    pattern=MisconceptionPattern.CONCEPT_ABSENCE,
                    master_edge=None,
                    student_edge=None,
                    concept="항상성",
                    confidence=0.75,
                    description="Missing concept: 항상성",
                ),
            ],
        }
        result = aggregate_class_misconceptions(student_misc)
        # Should return list of (pattern, description, count), sorted by count desc
        assert len(result) >= 2
        # Most frequent first
        assert result[0][2] >= result[1][2]

    def test_empty_aggregation(self):
        """Empty input → empty list."""
        result = aggregate_class_misconceptions({})
        assert result == []

    def test_single_student(self):
        """Single student with one misconception."""
        student_misc = {
            "s001": [
                ClassifiedMisconception(
                    pattern=MisconceptionPattern.INCLUSION_ERROR,
                    master_edge=None,
                    student_edge=TripletEdge("세포", "포함", "세포막"),
                    concept=None,
                    confidence=0.9,
                    description="Inclusion error: 세포→세포막",
                ),
            ],
        }
        result = aggregate_class_misconceptions(student_misc)
        assert len(result) == 1
        assert result[0][0] == MisconceptionPattern.INCLUSION_ERROR
        assert result[0][2] == 1
