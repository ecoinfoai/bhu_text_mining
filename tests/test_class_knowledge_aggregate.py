"""Tests for class_knowledge_aggregate.py — dataclasses and build function.

RED phase: these tests are written *before* the build function exists and
will fail until build_class_knowledge_aggregate() is implemented.

Covers:
  T005 — AggregateEdge, ClassKnowledgeAggregate dataclass field presence/types.
  T005 — build_class_knowledge_aggregate() with synthetic data:
         30-student, all-missing, single-student, empty-student, zero-division.
  FR-001, FR-002, FR-003, FR-004, SC-001, SC-002.
"""

from __future__ import annotations

import pytest

from forma.evaluation_types import TripletEdge, GraphComparisonResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_master_edges() -> list[TripletEdge]:
    """Return a small set of master edges for testing."""
    return [
        TripletEdge(subject="심근경색", relation="원인", object="허혈"),
        TripletEdge(subject="허혈", relation="유발", object="조직손상"),
    ]


def _make_comparison_result(
    student_id: str,
    question_sn: int,
    matched: list[TripletEdge] | None = None,
    wrong_direction: list[TripletEdge] | None = None,
    missing: list[TripletEdge] | None = None,
) -> GraphComparisonResult:
    """Build a synthetic GraphComparisonResult."""
    return GraphComparisonResult(
        student_id=student_id,
        question_sn=question_sn,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        matched_edges=matched or [],
        missing_edges=missing or [],
        extra_edges=[],
        wrong_direction_edges=wrong_direction or [],
    )


# ===========================================================================
# T005: AggregateEdge dataclass tests
# ===========================================================================


class TestAggregateEdgeDataclass:
    """T005: AggregateEdge dataclass field presence and type validation."""

    def test_fields_present(self):
        """AggregateEdge has all required fields."""
        from forma.class_knowledge_aggregate import AggregateEdge

        edge = AggregateEdge(
            subject="A",
            relation="R",
            obj="B",
            correct_count=10,
            error_count=5,
            missing_count=5,
            total_students=20,
            correct_ratio=0.5,
        )
        assert edge.subject == "A"
        assert edge.relation == "R"
        assert edge.obj == "B"
        assert edge.correct_count == 10
        assert edge.error_count == 5
        assert edge.missing_count == 5
        assert edge.total_students == 20
        assert edge.correct_ratio == pytest.approx(0.5)

    def test_field_types(self):
        """AggregateEdge fields have correct types."""
        from forma.class_knowledge_aggregate import AggregateEdge

        edge = AggregateEdge(
            subject="X",
            relation="Y",
            obj="Z",
            correct_count=1,
            error_count=2,
            missing_count=3,
            total_students=6,
            correct_ratio=1 / 6,
        )
        assert isinstance(edge.subject, str)
        assert isinstance(edge.relation, str)
        assert isinstance(edge.obj, str)
        assert isinstance(edge.correct_count, int)
        assert isinstance(edge.error_count, int)
        assert isinstance(edge.missing_count, int)
        assert isinstance(edge.total_students, int)
        assert isinstance(edge.correct_ratio, float)


# ===========================================================================
# T005: ClassKnowledgeAggregate dataclass tests
# ===========================================================================


class TestClassKnowledgeAggregateDataclass:
    """T005: ClassKnowledgeAggregate dataclass field presence."""

    def test_fields_present(self):
        """ClassKnowledgeAggregate has question_sn, edges, total_students."""
        from forma.class_knowledge_aggregate import ClassKnowledgeAggregate

        agg = ClassKnowledgeAggregate(
            question_sn=1,
            edges=[],
            total_students=30,
        )
        assert agg.question_sn == 1
        assert agg.edges == []
        assert agg.total_students == 30

    def test_edges_default_empty(self):
        """ClassKnowledgeAggregate.edges defaults to empty list."""
        from forma.class_knowledge_aggregate import ClassKnowledgeAggregate

        agg = ClassKnowledgeAggregate(question_sn=1)
        assert agg.edges == []


# ===========================================================================
# T005: build_class_knowledge_aggregate() tests — main scenarios
# ===========================================================================


class TestBuildClassKnowledgeAggregate30Students:
    """T005: 30-student synthetic data test (FR-001, FR-002, SC-001, SC-002).

    Setup: 2 master edges, 30 students.
    Edge 1 ("심근경색→허혈"): 24 correct, 5 error (wrong direction), 1 missing.
    Edge 2 ("허혈→조직손상"): 30 correct, 0 error, 0 missing.
    """

    @pytest.fixture
    def master_edges(self) -> list[TripletEdge]:
        return _make_master_edges()

    @pytest.fixture
    def comparison_results(self, master_edges: list[TripletEdge]) -> list[GraphComparisonResult]:
        results = []
        edge1, edge2 = master_edges

        for i in range(30):
            if i < 24:
                # Correct for edge1
                matched = [edge1, edge2]
                wrong = []
            elif i < 29:
                # Wrong direction for edge1, correct for edge2
                # wrong_direction_edges stores the STUDENT's reversed edge
                matched = [edge2]
                wrong = [TripletEdge(subject=edge1.object, relation=edge1.relation, object=edge1.subject)]
            else:
                # Missing edge1, correct for edge2
                matched = [edge2]
                wrong = []

            results.append(
                _make_comparison_result(
                    student_id=f"S{i:03d}",
                    question_sn=1,
                    matched=matched,
                    wrong_direction=wrong,
                )
            )
        return results

    def test_total_students(self, master_edges, comparison_results):
        """ClassKnowledgeAggregate.total_students == 30."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        agg = build_class_knowledge_aggregate(master_edges, comparison_results, question_sn=1)
        assert agg.total_students == 30

    def test_edge_count(self, master_edges, comparison_results):
        """Two master edges produce two AggregateEdge entries."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        agg = build_class_knowledge_aggregate(master_edges, comparison_results, question_sn=1)
        assert len(agg.edges) == 2

    def test_edge1_correct_count(self, master_edges, comparison_results):
        """Edge1 correct_count == 24."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        agg = build_class_knowledge_aggregate(master_edges, comparison_results, question_sn=1)
        edge1 = next(e for e in agg.edges if e.subject == "심근경색")
        assert edge1.correct_count == 24

    def test_edge1_error_count(self, master_edges, comparison_results):
        """Edge1 error_count == 5."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        agg = build_class_knowledge_aggregate(master_edges, comparison_results, question_sn=1)
        edge1 = next(e for e in agg.edges if e.subject == "심근경색")
        assert edge1.error_count == 5

    def test_edge1_missing_count(self, master_edges, comparison_results):
        """Edge1 missing_count == 1."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        agg = build_class_knowledge_aggregate(master_edges, comparison_results, question_sn=1)
        edge1 = next(e for e in agg.edges if e.subject == "심근경색")
        assert edge1.missing_count == 1

    def test_edge1_correct_ratio(self, master_edges, comparison_results):
        """Edge1 correct_ratio == 24/30 == 0.8 (SC-002)."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        agg = build_class_knowledge_aggregate(master_edges, comparison_results, question_sn=1)
        edge1 = next(e for e in agg.edges if e.subject == "심근경색")
        assert edge1.correct_ratio == pytest.approx(0.8)

    def test_edge2_all_correct(self, master_edges, comparison_results):
        """Edge2 correct_count == 30, error_count == 0, missing_count == 0."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        agg = build_class_knowledge_aggregate(master_edges, comparison_results, question_sn=1)
        edge2 = next(e for e in agg.edges if e.subject == "허혈")
        assert edge2.correct_count == 30
        assert edge2.error_count == 0
        assert edge2.missing_count == 0
        assert edge2.correct_ratio == pytest.approx(1.0)

    def test_invariant_counts_sum_to_total(self, master_edges, comparison_results):
        """SC-001: correct_count + error_count + missing_count == total_students for every edge."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        agg = build_class_knowledge_aggregate(master_edges, comparison_results, question_sn=1)
        for edge in agg.edges:
            assert edge.correct_count + edge.error_count + edge.missing_count == edge.total_students

    def test_question_sn_propagated(self, master_edges, comparison_results):
        """question_sn is correctly set on the aggregate."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        agg = build_class_knowledge_aggregate(master_edges, comparison_results, question_sn=1)
        assert agg.question_sn == 1


class TestBuildClassKnowledgeAggregateAllMissing:
    """T005: All students missing a specific edge (FR-004 edge case)."""

    def test_all_missing_edge(self):
        """When all students miss an edge: missing_count == total_students, correct_ratio == 0.0."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        master = [TripletEdge(subject="A", relation="R", object="B")]
        # 10 students, none have the edge matched or wrong
        results = [_make_comparison_result(f"S{i}", 1, matched=[], wrong_direction=[]) for i in range(10)]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)

        assert len(agg.edges) == 1
        edge = agg.edges[0]
        assert edge.missing_count == 10
        assert edge.correct_count == 0
        assert edge.error_count == 0
        assert edge.correct_ratio == pytest.approx(0.0)
        assert edge.total_students == 10


class TestBuildClassKnowledgeAggregateSingleStudent:
    """T005: Single-student case."""

    def test_single_student_correct(self):
        """Single student with correct edge: counts are 1/0/0."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        master = [TripletEdge(subject="A", relation="R", object="B")]
        results = [
            _make_comparison_result(
                "S001",
                1,
                matched=[TripletEdge(subject="A", relation="R", object="B")],
            )
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)

        assert agg.total_students == 1
        edge = agg.edges[0]
        assert edge.correct_count == 1
        assert edge.error_count == 0
        assert edge.missing_count == 0
        assert edge.correct_ratio == pytest.approx(1.0)

    def test_single_student_error(self):
        """Single student with wrong direction: counts are 0/1/0."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        master = [TripletEdge(subject="A", relation="R", object="B")]
        results = [
            _make_comparison_result(
                "S001",
                1,
                # wrong_direction_edges stores the student's reversed edge (B→A, not A→B)
                wrong_direction=[TripletEdge(subject="B", relation="R", object="A")],
            )
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)

        edge = agg.edges[0]
        assert edge.correct_count == 0
        assert edge.error_count == 1
        assert edge.missing_count == 0

    def test_single_student_missing(self):
        """Single student missing the edge: counts are 0/0/1."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        master = [TripletEdge(subject="A", relation="R", object="B")]
        results = [_make_comparison_result("S001", 1)]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)

        edge = agg.edges[0]
        assert edge.correct_count == 0
        assert edge.error_count == 0
        assert edge.missing_count == 1


class TestBuildClassKnowledgeAggregateEmptyStudents:
    """T005: Empty student list (FR-004)."""

    def test_empty_students(self):
        """Empty comparison_results: total_students == 0, correct_ratio == 0.0."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        master = [TripletEdge(subject="A", relation="R", object="B")]
        agg = build_class_knowledge_aggregate(master, [], question_sn=1)

        assert agg.total_students == 0
        assert len(agg.edges) == 1
        edge = agg.edges[0]
        assert edge.correct_count == 0
        assert edge.error_count == 0
        assert edge.missing_count == 0
        assert edge.correct_ratio == pytest.approx(0.0)

    def test_empty_students_empty_master(self):
        """Both empty master edges and empty students: edges list is empty."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        agg = build_class_knowledge_aggregate([], [], question_sn=1)
        assert agg.total_students == 0
        assert agg.edges == []


class TestBuildClassKnowledgeAggregateZeroDivision:
    """T005: Division guard when total_students == 0 (SC-002)."""

    def test_zero_division_guard(self):
        """correct_ratio == 0.0 when total_students == 0 (no ZeroDivisionError)."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        master = [
            TripletEdge(subject="X", relation="Y", object="Z"),
        ]
        agg = build_class_knowledge_aggregate(master, [], question_sn=1)
        edge = agg.edges[0]
        assert edge.correct_ratio == pytest.approx(0.0)
        assert edge.total_students == 0
