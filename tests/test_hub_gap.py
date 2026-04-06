"""Tests for compute_hub_gap() and compute_class_hub_gap() in forma.hub_gap.

T021: TestComputeHubGap
T022: TestComputeClassHubGap

These tests are written in the RED phase and are expected to FAIL until
src/forma/hub_gap.py is created.
"""

from __future__ import annotations

import pytest

from forma.evaluation_types import TripletEdge
from forma.hub_gap import compute_hub_gap, compute_class_hub_gap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_edge(subject: str, relation: str, obj: str) -> TripletEdge:
    return TripletEdge(subject=subject, relation=relation, object=obj)


# ---------------------------------------------------------------------------
# T021: TestComputeHubGap
# ---------------------------------------------------------------------------


class TestComputeHubGap:
    """T021 — unit tests for compute_hub_gap()."""

    def test_basic_hub_gap(self):
        """Star graph: A is hub. Student knows A→B and A→C.

        Expects:
        - Up to 5 results returned (top_k=5)
        - A has highest degree centrality → result[0].concept == "A"
        - A appears in student edges → result[0].student_present == True
        - B also appears in results
        """
        master_edges = [
            make_edge("A", "rel", "B"),
            make_edge("A", "rel", "C"),
            make_edge("A", "rel", "D"),
            make_edge("A", "rel", "E"),
            make_edge("A", "rel", "F"),
        ]
        student_edges = [
            make_edge("A", "rel", "B"),
            make_edge("A", "rel", "C"),
        ]

        result = compute_hub_gap(master_edges, student_edges, top_k=5)

        assert len(result) <= 5
        assert len(result) >= 1
        assert result[0].concept == "A"
        assert result[0].student_present is True
        assert any(e.concept == "B" for e in result)

    def test_empty_student_all_absent(self):
        """When student has no edges, all HubGapEntry.student_present == False."""
        master_edges = [
            make_edge("X", "rel", "Y"),
            make_edge("X", "rel", "Z"),
            make_edge("X", "rel", "W"),
        ]
        student_edges: list[TripletEdge] = []

        result = compute_hub_gap(master_edges, student_edges, top_k=3)

        assert len(result) > 0
        for entry in result:
            assert entry.student_present is False

    def test_empty_master_returns_empty(self):
        """When master has no edges, result is an empty list."""
        master_edges: list[TripletEdge] = []
        student_edges = [
            make_edge("A", "rel", "B"),
        ]

        result = compute_hub_gap(master_edges, student_edges, top_k=5)

        assert result == []

    def test_top_k_limit(self):
        """With 10 unique nodes in master and top_k=3, result has exactly 3 entries."""
        # Build a graph with 10 unique nodes: hub M points to 9 others,
        # plus cross-links to ensure 10 unique nodes
        master_edges = [
            make_edge("M", "rel", "N1"),
            make_edge("M", "rel", "N2"),
            make_edge("M", "rel", "N3"),
            make_edge("M", "rel", "N4"),
            make_edge("M", "rel", "N5"),
            make_edge("M", "rel", "N6"),
            make_edge("M", "rel", "N7"),
            make_edge("M", "rel", "N8"),
            make_edge("M", "rel", "N9"),
        ]
        student_edges: list[TripletEdge] = []

        result = compute_hub_gap(master_edges, student_edges, top_k=3)

        assert len(result) == 3

    def test_directed_centrality(self):
        """Many edges pointing TO node X give X high degree centrality.

        X should appear among the top results due to high centrality.
        """
        # Many nodes point to X → X has high in-degree (and thus high degree centrality)
        master_edges = [
            make_edge("A", "rel", "X"),
            make_edge("B", "rel", "X"),
            make_edge("C", "rel", "X"),
            make_edge("D", "rel", "X"),
            make_edge("E", "rel", "X"),
            # X points to only one node
            make_edge("X", "rel", "Z"),
        ]
        student_edges: list[TripletEdge] = []

        result = compute_hub_gap(master_edges, student_edges, top_k=3)

        concept_names = [e.concept for e in result]
        assert "X" in concept_names, f"Expected X in top results, got: {concept_names}"

    def test_result_sorted_by_degree_centrality_descending(self):
        """Result entries are sorted by degree_centrality in descending order."""
        master_edges = [
            make_edge("HUB", "rel", "B"),
            make_edge("HUB", "rel", "C"),
            make_edge("HUB", "rel", "D"),
            make_edge("B", "rel", "E"),
        ]
        student_edges: list[TripletEdge] = []

        result = compute_hub_gap(master_edges, student_edges, top_k=4)

        centralities = [e.degree_centrality for e in result]
        assert centralities == sorted(centralities, reverse=True)

    def test_student_present_detection_uses_subject_and_object(self):
        """student_present is True when concept appears as subject OR object."""
        master_edges = [
            make_edge("HUB", "rel", "A"),
            make_edge("HUB", "rel", "B"),
            make_edge("HUB", "rel", "C"),
        ]
        # Student edge has HUB as an object (not subject)
        student_edges = [
            make_edge("X", "rel", "HUB"),
        ]

        result = compute_hub_gap(master_edges, student_edges, top_k=3)

        hub_entry = next((e for e in result if e.concept == "HUB"), None)
        assert hub_entry is not None
        assert hub_entry.student_present is True

    def test_hub_gap_entry_has_zero_class_inclusion_rate(self):
        """compute_hub_gap() returns HubGapEntry with class_inclusion_rate == 0.0."""
        master_edges = [
            make_edge("A", "rel", "B"),
            make_edge("A", "rel", "C"),
        ]
        student_edges = [make_edge("A", "rel", "B")]

        result = compute_hub_gap(master_edges, student_edges, top_k=2)

        for entry in result:
            assert entry.class_inclusion_rate == 0.0

    def test_self_loops_filtered(self):
        """Self-loop A→A is excluded; degree_centrality stays <= 1.0 for all entries."""
        master_edges = [
            make_edge("A", "rel", "A"),  # self-loop — must be filtered
            make_edge("A", "rel", "B"),
            make_edge("A", "rel", "C"),
        ]
        student_edges: list[TripletEdge] = []

        result = compute_hub_gap(master_edges, student_edges, top_k=5)

        assert len(result) >= 1
        for entry in result:
            assert entry.degree_centrality <= 1.0, (
                f"degree_centrality {entry.degree_centrality} > 1.0 for concept {entry.concept!r}"
            )


# ---------------------------------------------------------------------------
# T022: TestComputeClassHubGap
# ---------------------------------------------------------------------------


class TestComputeClassHubGap:
    """T022 — unit tests for compute_class_hub_gap()."""

    def test_class_inclusion_rate(self):
        """s1 and s2 include A; s3 does not → class_inclusion_rate for A ≈ 2/3."""
        master_edges = [
            make_edge("A", "rel", "B"),
            make_edge("A", "rel", "C"),
            make_edge("A", "rel", "D"),
        ]
        all_student_edges = {
            "s1": [make_edge("A", "rel", "B")],
            "s2": [make_edge("A", "rel", "C")],
            "s3": [make_edge("B", "rel", "C")],  # does NOT mention A
        }

        result = compute_class_hub_gap(master_edges, all_student_edges, top_k=3)

        hub_entry = next((e for e in result if e.concept == "A"), None)
        assert hub_entry is not None, "A should be in results as the hub"
        assert abs(hub_entry.class_inclusion_rate - 2 / 3) < 1e-6

    def test_all_students_include_concept(self):
        """When all students mention the hub concept, class_inclusion_rate == 1.0."""
        master_edges = [
            make_edge("HUB", "rel", "B"),
            make_edge("HUB", "rel", "C"),
            make_edge("HUB", "rel", "D"),
        ]
        all_student_edges = {
            "s1": [make_edge("HUB", "rel", "B")],
            "s2": [make_edge("X", "rel", "HUB")],  # HUB as object
            "s3": [make_edge("HUB", "rel", "C")],
        }

        result = compute_class_hub_gap(master_edges, all_student_edges, top_k=3)

        hub_entry = next((e for e in result if e.concept == "HUB"), None)
        assert hub_entry is not None
        assert hub_entry.class_inclusion_rate == pytest.approx(1.0)

    def test_no_students_include_concept(self):
        """When no student mentions the hub concept, class_inclusion_rate == 0.0."""
        master_edges = [
            make_edge("HUB", "rel", "B"),
            make_edge("HUB", "rel", "C"),
            make_edge("HUB", "rel", "D"),
        ]
        all_student_edges = {
            "s1": [make_edge("B", "rel", "C")],
            "s2": [make_edge("C", "rel", "D")],
            "s3": [make_edge("X", "rel", "Y")],
        }

        result = compute_class_hub_gap(master_edges, all_student_edges, top_k=3)

        hub_entry = next((e for e in result if e.concept == "HUB"), None)
        assert hub_entry is not None
        assert hub_entry.class_inclusion_rate == pytest.approx(0.0)

    def test_class_hub_gap_result_sorted_descending(self):
        """Results are sorted by degree_centrality descending."""
        master_edges = [
            make_edge("BIG_HUB", "rel", "A"),
            make_edge("BIG_HUB", "rel", "B"),
            make_edge("BIG_HUB", "rel", "C"),
            make_edge("BIG_HUB", "rel", "D"),
            make_edge("SMALL_HUB", "rel", "E"),
        ]
        all_student_edges = {
            "s1": [make_edge("BIG_HUB", "rel", "A")],
        }

        result = compute_class_hub_gap(master_edges, all_student_edges, top_k=5)

        centralities = [e.degree_centrality for e in result]
        assert centralities == sorted(centralities, reverse=True)

    def test_empty_master_returns_empty(self):
        """Empty master edges → empty result."""
        master_edges: list[TripletEdge] = []
        all_student_edges = {
            "s1": [make_edge("A", "rel", "B")],
        }

        result = compute_class_hub_gap(master_edges, all_student_edges, top_k=5)

        assert result == []

    def test_empty_class_returns_nan_or_zero_inclusion_rate(self):
        """Empty all_student_edges dict → result entries have class_inclusion_rate == 0.0."""
        master_edges = [
            make_edge("A", "rel", "B"),
            make_edge("A", "rel", "C"),
        ]
        all_student_edges: dict[str, list[TripletEdge]] = {}

        result = compute_class_hub_gap(master_edges, all_student_edges, top_k=2)

        # Either returns empty or returns entries with 0.0 inclusion rate
        for entry in result:
            assert entry.class_inclusion_rate == pytest.approx(0.0)

    def test_top_k_limit_class(self):
        """top_k=2 returns at most 2 entries even with many master nodes."""
        master_edges = [make_edge("H", "rel", f"N{i}") for i in range(10)]
        all_student_edges = {"s1": []}

        result = compute_class_hub_gap(master_edges, all_student_edges, top_k=2)

        assert len(result) == 2

    def test_class_inclusion_partial(self):
        """Partial inclusion: 3 of 5 students include concept → rate == 0.6."""
        master_edges = [
            make_edge("CORE", "rel", "A"),
            make_edge("CORE", "rel", "B"),
            make_edge("CORE", "rel", "C"),
        ]
        all_student_edges = {
            "s1": [make_edge("CORE", "rel", "A")],
            "s2": [make_edge("CORE", "rel", "B")],
            "s3": [make_edge("CORE", "rel", "C")],
            "s4": [make_edge("X", "rel", "Y")],
            "s5": [make_edge("P", "rel", "Q")],
        }

        result = compute_class_hub_gap(master_edges, all_student_edges, top_k=3)

        core_entry = next((e for e in result if e.concept == "CORE"), None)
        assert core_entry is not None
        assert core_entry.class_inclusion_rate == pytest.approx(0.6)
