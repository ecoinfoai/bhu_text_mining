"""Adversary attack tests for Phase 3 (US1): class knowledge aggregate.

6 personas attack build_class_knowledge_aggregate() and dataclasses:
1. Edge Case Hunter: boundary conditions
2. Memory Saboteur: resource exhaustion
3. Type System Antagonist: silent type errors
4. Concurrency Destroyer: thread safety
5. PDF Killer: downstream rendering safety
6. Data Integrity Enforcer: mathematical invariants

Each test is designed to expose bugs IF they exist.
"""

from __future__ import annotations

import math
import random
import threading
from typing import Any

import pytest

from forma.class_knowledge_aggregate import (
    AggregateEdge,
    ClassKnowledgeAggregate,
    build_class_knowledge_aggregate,
)
from forma.evaluation_types import GraphComparisonResult, TripletEdge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_comparison_result(
    student_id: str,
    question_sn: int,
    matched: list[TripletEdge] | None = None,
    wrong_direction: list[TripletEdge] | None = None,
    missing: list[TripletEdge] | None = None,
    extra: list[TripletEdge] | None = None,
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
        extra_edges=extra or [],
        wrong_direction_edges=wrong_direction or [],
    )


# ===========================================================================
# PERSONA 1: THE EDGE CASE HUNTER
# ===========================================================================


class TestEdgeCaseHunter:
    """Persona 1: Boundary conditions that crash production."""

    def test_total_students_zero_no_division_error(self):
        """total_students=0 must NOT cause ZeroDivisionError."""
        master = [TripletEdge("A", "R", "B")]
        agg = build_class_knowledge_aggregate(master, [], question_sn=1)
        assert agg.total_students == 0
        for edge in agg.edges:
            assert edge.correct_ratio == 0.0
            assert not math.isnan(edge.correct_ratio)
            assert not math.isinf(edge.correct_ratio)

    def test_all_students_correct_every_edge(self):
        """All students correct every edge -> correct_ratio == 1.0 for all."""
        master = [
            TripletEdge("A", "R", "B"),
            TripletEdge("C", "S", "D"),
        ]
        results = [
            _make_comparison_result(
                f"S{i}", 1,
                matched=[
                    TripletEdge("A", "R", "B"),
                    TripletEdge("C", "S", "D"),
                ],
            )
            for i in range(50)
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        for edge in agg.edges:
            assert edge.correct_ratio == pytest.approx(1.0)
            assert edge.error_count == 0
            assert edge.missing_count == 0

    def test_all_students_missing_every_edge(self):
        """All students missing every edge -> correct_ratio == 0.0, missing_count == total."""
        master = [
            TripletEdge("A", "R", "B"),
            TripletEdge("C", "S", "D"),
        ]
        results = [
            _make_comparison_result(f"S{i}", 1)
            for i in range(200)
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        for edge in agg.edges:
            assert edge.correct_ratio == pytest.approx(0.0)
            assert edge.missing_count == 200
            assert edge.correct_count == 0
            assert edge.error_count == 0

    def test_single_master_edge_200_students_all_error(self):
        """Single master edge with 200 students all in error direction."""
        master_edge = TripletEdge("X", "causes", "Y")
        # wrong_direction_edges stores the student's reversed edge (Y→X, not X→Y)
        reversed_edge = TripletEdge("Y", "causes", "X")
        results = [
            _make_comparison_result(
                f"S{i}", 1,
                wrong_direction=[reversed_edge],
            )
            for i in range(200)
        ]
        agg = build_class_knowledge_aggregate([master_edge], results, question_sn=1)
        assert len(agg.edges) == 1
        edge = agg.edges[0]
        assert edge.error_count == 200
        assert edge.correct_count == 0
        assert edge.missing_count == 0
        assert edge.correct_ratio == pytest.approx(0.0)

    def test_empty_master_edges_non_empty_students(self):
        """Empty master_edges with non-empty students -> empty edges list."""
        results = [
            _make_comparison_result(f"S{i}", 1,
                                    matched=[TripletEdge("A", "R", "B")])
            for i in range(5)
        ]
        agg = build_class_knowledge_aggregate([], results, question_sn=1)
        assert agg.edges == []
        assert agg.total_students == 5

    def test_student_has_edge_in_both_matched_and_wrong(self):
        """Edge case: student has same edge in BOTH matched AND wrong_direction.

        Spec says matched takes priority (correct). This tests the continue
        in the implementation.
        """
        master = [TripletEdge("A", "R", "B")]
        # Student has the edge in both matched and wrong_direction
        results = [
            _make_comparison_result(
                "S001", 1,
                matched=[TripletEdge("A", "R", "B")],
                wrong_direction=[TripletEdge("A", "R", "B")],
            ),
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        edge = agg.edges[0]
        # Should count as correct due to matched check first
        assert edge.correct_count == 1
        assert edge.error_count == 0
        assert edge.missing_count == 0

    def test_many_master_edges_few_students(self):
        """100 master edges, 1 student -> 100 AggregateEdges, each count 0 or 1."""
        master = [TripletEdge(f"S{i}", "R", f"O{i}") for i in range(100)]
        results = [
            _make_comparison_result(
                "S001", 1,
                matched=[master[0]],  # only first edge matched
            ),
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        assert len(agg.edges) == 100
        for edge in agg.edges:
            assert edge.correct_count + edge.error_count + edge.missing_count == 1


# ===========================================================================
# PERSONA 2: THE MEMORY SABOTEUR
# ===========================================================================


class TestMemorySaboteur:
    """Persona 2: Memory leaks and resource exhaustion."""

    def test_1000_students_10_edges_no_crash(self):
        """1000 students with 10 master edges completes without error."""
        master = [TripletEdge(f"S{i}", "R", f"O{i}") for i in range(10)]
        # Properly reverse wrong_direction edges (student writes O->R->S)
        reversed_5_8 = [
            TripletEdge(me.object, me.relation, me.subject)
            for me in master[5:8]
        ]
        results = [
            _make_comparison_result(
                f"S{i}", 1,
                matched=master[:5],
                wrong_direction=reversed_5_8,
            )
            for i in range(1000)
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        assert agg.total_students == 1000
        assert len(agg.edges) == 10
        # Verify the counts are correct with properly reversed edges
        for edge in agg.edges:
            idx = int(edge.subject[1:])
            if idx < 5:
                assert edge.correct_count == 1000
            elif idx < 8:
                assert edge.error_count == 1000
            else:
                assert edge.missing_count == 1000

    def test_aggregate_returns_new_list_each_call(self):
        """Each call returns a new edges list (no shared mutable state)."""
        master = [TripletEdge("A", "R", "B")]
        results1 = [_make_comparison_result("S001", 1, matched=[TripletEdge("A", "R", "B")])]
        results2 = [_make_comparison_result("S002", 1)]

        agg1 = build_class_knowledge_aggregate(master, results1, question_sn=1)
        agg2 = build_class_knowledge_aggregate(master, results2, question_sn=1)

        assert agg1.edges is not agg2.edges
        assert agg1.edges[0].correct_count == 1
        assert agg2.edges[0].correct_count == 0

    def test_repeated_calls_no_state_leak(self):
        """100 repeated calls do not accumulate state."""
        master = [TripletEdge("A", "R", "B")]
        for _ in range(100):
            results = [_make_comparison_result("S001", 1, matched=[TripletEdge("A", "R", "B")])]
            agg = build_class_knowledge_aggregate(master, results, question_sn=1)
            assert agg.edges[0].correct_count == 1
            assert agg.total_students == 1


# ===========================================================================
# PERSONA 3: THE TYPE SYSTEM ANTAGONIST
# ===========================================================================


class TestTypeSystemAntagonist:
    """Persona 3: Silent type errors that corrupt data."""

    def test_correct_ratio_is_float_not_int(self):
        """correct_ratio must be float, never int (even when ratio is exactly 1)."""
        master = [TripletEdge("A", "R", "B")]
        results = [
            _make_comparison_result("S001", 1, matched=[TripletEdge("A", "R", "B")])
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        assert isinstance(agg.edges[0].correct_ratio, float)

    def test_correct_ratio_never_nan(self):
        """correct_ratio must never be NaN."""
        master = [TripletEdge("A", "R", "B")]
        # Case 1: empty students
        agg = build_class_knowledge_aggregate(master, [], question_sn=1)
        for edge in agg.edges:
            assert not math.isnan(edge.correct_ratio)

    def test_correct_ratio_never_inf(self):
        """correct_ratio must never be Infinity."""
        master = [TripletEdge("A", "R", "B")]
        agg = build_class_knowledge_aggregate(master, [], question_sn=1)
        for edge in agg.edges:
            assert not math.isinf(edge.correct_ratio)

    def test_counts_are_int_not_float(self):
        """correct_count, error_count, missing_count must be int."""
        master = [TripletEdge("A", "R", "B")]
        results = [
            _make_comparison_result("S001", 1, matched=[TripletEdge("A", "R", "B")])
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        edge = agg.edges[0]
        assert isinstance(edge.correct_count, int)
        assert isinstance(edge.error_count, int)
        assert isinstance(edge.missing_count, int)
        assert isinstance(edge.total_students, int)

    def test_correct_ratio_exact_precision(self):
        """correct_ratio == correct_count / total_students with no floating point drift beyond 1e-9."""
        master = [TripletEdge("A", "R", "B")]
        # 7 out of 11 students correct -> ratio should be 7/11
        results = []
        for i in range(11):
            if i < 7:
                results.append(_make_comparison_result(
                    f"S{i}", 1, matched=[TripletEdge("A", "R", "B")]
                ))
            else:
                results.append(_make_comparison_result(f"S{i}", 1))

        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        edge = agg.edges[0]
        expected = 7 / 11
        assert abs(edge.correct_ratio - expected) < 1e-9

    def test_total_students_matches_len_comparison_results(self):
        """ClassKnowledgeAggregate.total_students == len(comparison_results)."""
        master = [TripletEdge("A", "R", "B")]
        results = [_make_comparison_result(f"S{i}", 1) for i in range(37)]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        assert agg.total_students == 37


# ===========================================================================
# PERSONA 4: THE CONCURRENCY DESTROYER
# ===========================================================================


class TestConcurrencyDestroyer:
    """Persona 4: Race conditions and shared state corruption."""

    def test_concurrent_builds_independent_results(self):
        """Two concurrent calls produce independent results (no shared state)."""
        master1 = [TripletEdge("A", "R1", "B")]
        master2 = [TripletEdge("C", "R2", "D")]

        results1 = [
            _make_comparison_result("S001", 1, matched=[TripletEdge("A", "R1", "B")])
            for _ in range(10)
        ]
        results2 = [
            _make_comparison_result("S001", 2)
            for _ in range(5)
        ]

        outputs: dict[str, ClassKnowledgeAggregate] = {}
        errors: list[Exception] = []

        def run1():
            try:
                outputs["agg1"] = build_class_knowledge_aggregate(master1, results1, question_sn=1)
            except Exception as e:
                errors.append(e)

        def run2():
            try:
                outputs["agg2"] = build_class_knowledge_aggregate(master2, results2, question_sn=2)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=run1)
        t2 = threading.Thread(target=run2)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Concurrent build errors: {errors}"
        assert outputs["agg1"].total_students == 10
        assert outputs["agg2"].total_students == 5
        assert outputs["agg1"].edges[0].correct_count == 10
        assert outputs["agg2"].edges[0].correct_count == 0

    def test_concurrent_builds_deterministic(self):
        """Same inputs produce same outputs when called concurrently."""
        master = [TripletEdge("A", "R", "B")]
        results = [
            _make_comparison_result(
                f"S{i}", 1,
                matched=[TripletEdge("A", "R", "B")] if i % 2 == 0 else [],
                wrong_direction=[TripletEdge("A", "R", "B")] if i % 2 == 1 else [],
            )
            for i in range(20)
        ]

        outputs: list[ClassKnowledgeAggregate] = [None] * 5  # type: ignore
        errors: list[Exception] = []

        def run(idx: int):
            try:
                outputs[idx] = build_class_knowledge_aggregate(master, results, question_sn=1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        for i in range(1, 5):
            assert outputs[i].edges[0].correct_count == outputs[0].edges[0].correct_count
            assert outputs[i].edges[0].error_count == outputs[0].edges[0].error_count


# ===========================================================================
# PERSONA 5: THE PDF KILLER
# ===========================================================================


class TestPDFKiller:
    """Persona 5: Data that will crash downstream PDF generation."""

    def test_concept_name_with_xml_special_chars(self):
        """Edge with < > & " ' in subject/obj - must not crash build_class_knowledge_aggregate."""
        master = [
            TripletEdge('<script>alert("xss")</script>', "관계", "A & B"),
        ]
        results = [
            _make_comparison_result(
                "S001", 1,
                matched=[TripletEdge('<script>alert("xss")</script>', "관계", "A & B")],
            ),
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        assert len(agg.edges) == 1
        edge = agg.edges[0]
        assert '<script>' in edge.subject
        assert '&' in edge.obj

    def test_concept_name_200_chars(self):
        """Very long concept name (200+ chars) in edge."""
        long_name = "가" * 200
        master = [TripletEdge(long_name, "R", "B")]
        results = [_make_comparison_result("S001", 1, matched=[TripletEdge(long_name, "R", "B")])]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        assert agg.edges[0].subject == long_name

    def test_aggregate_500_edges(self):
        """500 master edges should not crash or overflow."""
        master = [TripletEdge(f"S{i}", "R", f"O{i}") for i in range(500)]
        results = [
            _make_comparison_result(
                f"S{j}", 1,
                matched=[master[0], master[1]],
            )
            for j in range(10)
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        assert len(agg.edges) == 500

    def test_correct_ratio_exactly_0_3_boundary(self):
        """correct_ratio of exactly 0.3 - boundary for weak-edge filter."""
        master = [TripletEdge("A", "R", "B")]
        # 3 out of 10 correct -> exactly 0.3
        results = []
        for i in range(10):
            if i < 3:
                results.append(_make_comparison_result(
                    f"S{i}", 1, matched=[TripletEdge("A", "R", "B")]
                ))
            else:
                results.append(_make_comparison_result(f"S{i}", 1))
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        assert agg.edges[0].correct_ratio == pytest.approx(0.3)

    def test_all_edges_below_min_ratio_to_show(self):
        """All edges have correct_ratio < 0.05 (below min_ratio_to_show threshold)."""
        master = [TripletEdge(f"S{i}", "R", f"O{i}") for i in range(5)]
        # Only 1 student correct out of 100 -> ratio 0.01
        results = []
        for j in range(100):
            if j == 0:
                results.append(_make_comparison_result(
                    f"S{j}", 1, matched=[master[0]]
                ))
            else:
                results.append(_make_comparison_result(f"S{j}", 1))
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        # Edge 0 has ratio 0.01, edges 1-4 have ratio 0.0
        for edge in agg.edges:
            assert edge.correct_ratio < 0.05


# ===========================================================================
# PERSONA 6: THE DATA INTEGRITY ENFORCER
# ===========================================================================


class TestDataIntegrityEnforcer:
    """Persona 6: Mathematical invariant violations."""

    def test_invariant_counts_sum_to_total_random_1000(self):
        """correct + error + missing == total for ALL edges across 1000 random scenarios.

        Team-lead requirement: Persona 6 must verify this invariant with 1000
        parametrized random cases. Uses properly reversed wrong_direction edges.
        """
        random.seed(42)
        for trial in range(1000):
            n_edges = random.randint(1, 10)
            n_students = random.randint(0, 50)
            master = [TripletEdge(f"S{i}", "R", f"O{i}") for i in range(n_edges)]

            results = []
            for j in range(n_students):
                # Randomly assign edges to matched or wrong_direction
                matched = []
                wrong = []
                for me in master:
                    r = random.random()
                    if r < 0.5:
                        matched.append(me)
                    elif r < 0.7:
                        # Store the REVERSED edge (student's wrong direction)
                        wrong.append(TripletEdge(me.object, me.relation, me.subject))
                    # else: missing
                results.append(_make_comparison_result(
                    f"S{j}", 1, matched=matched, wrong_direction=wrong
                ))

            agg = build_class_knowledge_aggregate(master, results, question_sn=1)
            for edge in agg.edges:
                total = edge.correct_count + edge.error_count + edge.missing_count
                assert total == edge.total_students, (
                    f"Trial {trial}: {edge.subject} -> {edge.obj}: "
                    f"{edge.correct_count} + {edge.error_count} + {edge.missing_count} "
                    f"= {total} != {edge.total_students}"
                )

    def test_correct_ratio_exact_for_all_random_1000(self):
        """correct_ratio == correct_count / total_students for all edges across 1000 random scenarios."""
        random.seed(123)
        for trial in range(1000):
            n_edges = random.randint(1, 5)
            n_students = random.randint(1, 30)
            master = [TripletEdge(f"S{i}", "R", f"O{i}") for i in range(n_edges)]

            results = []
            for j in range(n_students):
                matched = [me for me in master if random.random() < 0.4]
                # Properly reverse for wrong_direction
                wrong = [
                    TripletEdge(me.object, me.relation, me.subject)
                    for me in master
                    if me not in matched and random.random() < 0.3
                ]
                results.append(_make_comparison_result(
                    f"S{j}", 1, matched=matched, wrong_direction=wrong
                ))

            agg = build_class_knowledge_aggregate(master, results, question_sn=1)
            for edge in agg.edges:
                expected = edge.correct_count / edge.total_students if edge.total_students > 0 else 0.0
                assert abs(edge.correct_ratio - expected) < 1e-9, (
                    f"Trial {trial}: ratio mismatch: "
                    f"{edge.correct_ratio} vs {expected}"
                )

    def test_correct_ratio_bounded_0_to_1(self):
        """correct_ratio is always in [0.0, 1.0]."""
        master = [TripletEdge("A", "R", "B")]
        for n in [0, 1, 5, 50]:
            results = [_make_comparison_result(f"S{i}", 1) for i in range(n)]
            agg = build_class_knowledge_aggregate(master, results, question_sn=1)
            for edge in agg.edges:
                assert 0.0 <= edge.correct_ratio <= 1.0

    def test_total_students_consistent_across_all_edges(self):
        """Every AggregateEdge.total_students == ClassKnowledgeAggregate.total_students."""
        master = [TripletEdge(f"S{i}", "R", f"O{i}") for i in range(5)]
        results = [_make_comparison_result(f"S{j}", 1) for j in range(17)]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        for edge in agg.edges:
            assert edge.total_students == agg.total_students

    def test_counts_are_non_negative(self):
        """All counts must be >= 0."""
        master = [TripletEdge("A", "R", "B")]
        for n in [0, 1, 10]:
            results = [_make_comparison_result(f"S{i}", 1) for i in range(n)]
            agg = build_class_knowledge_aggregate(master, results, question_sn=1)
            for edge in agg.edges:
                assert edge.correct_count >= 0
                assert edge.error_count >= 0
                assert edge.missing_count >= 0

    def test_wrong_direction_edge_matching_logic(self):
        """Wrong direction edge is counted correctly.

        The wrong_direction_edges in GraphComparisonResult contain the
        STUDENT's reversed edge (B->R->A), NOT the master edge (A->R->B).
        This reflects graph_comparator.py line 225:
            wrong_direction.append(student_edges[s_idx])

        When master is A->R->B and student writes B->R->A, the
        wrong_direction_edges contains TripletEdge("B","R","A").
        """
        master = [TripletEdge("A", "R", "B")]
        # Wrong direction: stores student's reversed edge (B→A), not master (A→B)
        results = [
            _make_comparison_result(
                "S001", 1,
                wrong_direction=[TripletEdge("B", "R", "A")],
            ),
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        assert agg.edges[0].error_count == 1
        assert agg.edges[0].correct_count == 0
        assert agg.edges[0].missing_count == 0

    def test_edge_obj_not_object_field_name(self):
        """AggregateEdge uses 'obj' field name (not 'object'), matching master TripletEdge.object."""
        master = [TripletEdge("Subject1", "Relation1", "Object1")]
        results = [
            _make_comparison_result(
                "S001", 1,
                matched=[TripletEdge("Subject1", "Relation1", "Object1")],
            ),
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        edge = agg.edges[0]
        # AggregateEdge stores it as 'obj', mapped from TripletEdge.object
        assert edge.obj == "Object1"
        assert edge.subject == "Subject1"
        assert edge.relation == "Relation1"

    def test_wrong_direction_cross_contamination(self):
        """ATTACK: Two master edges where one's reversal coincidentally matches another.

        Master: A->R1->B and B->R2->C
        If student reverses A->R1->B (producing B->R1->A), the wrong_direction
        matching uses (e.subject == master.object and e.object == master.subject).
        For master B->R2->C, the reversed edge B->R1->A does NOT match because
        e.object='A' != master.subject='B'.
        This verifies no cross-contamination between master edges.
        """
        master = [
            TripletEdge("A", "R1", "B"),
            TripletEdge("B", "R2", "C"),
        ]
        # Student reverses A->R1->B producing B->R1->A in wrong_direction
        results = [
            _make_comparison_result(
                "S001", 1,
                wrong_direction=[TripletEdge("B", "R1", "A")],
            ),
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        edge_ab = next(e for e in agg.edges if e.subject == "A")
        edge_bc = next(e for e in agg.edges if e.subject == "B")

        # A->R1->B should count as error (B->R1->A matches reversal)
        assert edge_ab.error_count == 1
        assert edge_ab.missing_count == 0
        # B->R2->C should count as missing (no cross-contamination)
        assert edge_bc.error_count == 0
        assert edge_bc.missing_count == 1

    def test_wrong_direction_relation_not_checked(self):
        """ATTACK: wrong_direction matching does NOT check relation field.

        Implementation line 110-111: checks only e.subject == master.object
        and e.object == master.subject, ignoring relation.

        This tests that a wrong-direction edge with a DIFFERENT relation
        still gets counted as error (by design), not silently dropped.
        """
        master = [TripletEdge("A", "causes", "B")]
        # Student reverses it but writes a different relation
        results = [
            _make_comparison_result(
                "S001", 1,
                wrong_direction=[TripletEdge("B", "caused-by", "A")],
            ),
        ]
        agg = build_class_knowledge_aggregate(master, results, question_sn=1)
        edge = agg.edges[0]
        # The matching logic only checks subject/object reversal, not relation
        assert edge.error_count == 1
        assert edge.correct_count == 0
        assert edge.missing_count == 0

    def test_random_1000_invariant_with_mixed_relation_names(self):
        """1000 random trials with diverse relation names to stress invariant."""
        random.seed(999)
        relations = ["원인", "유발", "포함", "구성", "is-a", "part-of", "causes"]
        for trial in range(1000):
            n_edges = random.randint(1, 8)
            n_students = random.randint(0, 30)
            master = [
                TripletEdge(f"S{i}", relations[i % len(relations)], f"O{i}")
                for i in range(n_edges)
            ]

            results = []
            for j in range(n_students):
                matched = []
                wrong = []
                for me in master:
                    r = random.random()
                    if r < 0.4:
                        matched.append(me)
                    elif r < 0.65:
                        wrong.append(TripletEdge(me.object, me.relation, me.subject))
                results.append(_make_comparison_result(
                    f"S{j}", 1, matched=matched, wrong_direction=wrong
                ))

            agg = build_class_knowledge_aggregate(master, results, question_sn=1)
            for edge in agg.edges:
                total = edge.correct_count + edge.error_count + edge.missing_count
                assert total == edge.total_students, (
                    f"Trial {trial}: {edge.subject}->{edge.obj}: "
                    f"{edge.correct_count}+{edge.error_count}+{edge.missing_count}"
                    f"={total} != {edge.total_students}"
                )
                assert 0.0 <= edge.correct_ratio <= 1.0
                if edge.total_students > 0:
                    expected = edge.correct_count / edge.total_students
                    assert abs(edge.correct_ratio - expected) < 1e-9
