"""Tests for graph_comparator.py — directed triplet graph comparison.

Embedding calls are mocked to avoid model download.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from forma.evaluation_types import GraphComparisonResult, TripletEdge
from forma.graph_comparator import GraphComparator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _e(s: str, r: str, o: str) -> TripletEdge:
    return TripletEdge(subject=s, relation=r, object=o)


# ---------------------------------------------------------------------------
# GraphComparator tests
# ---------------------------------------------------------------------------


class TestGraphComparator:
    """Tests for GraphComparator."""

    def test_empty_graphs(self):
        """Both empty → perfect F1."""
        gc = GraphComparator()
        result = gc.compare("s001", 1, [], [])
        assert result.f1 == pytest.approx(1.0)
        assert result.matched_edges == []

    def test_exact_match(self):
        """Identical edges → F1 = 1.0."""
        master = [_e("A", "causes", "B"), _e("B", "leads_to", "C")]
        student = [_e("A", "causes", "B"), _e("B", "leads_to", "C")]

        gc = GraphComparator()
        with patch("forma.graph_comparator.encode_texts") as mock_enc:
            embs = np.eye(6)  # 2 master + 2 student + 2 reversed
            # Make student[0] match master[0], student[1] match master[1]
            embs[2] = embs[0]  # student[0] == master[0]
            embs[3] = embs[1]  # student[1] == master[1]
            mock_enc.return_value = embs

            result = gc.compare("s001", 1, master, student)

        assert result.f1 == pytest.approx(1.0)
        assert len(result.missing_edges) == 0

    def test_missing_edges(self):
        """Student missing some edges → recall < 1."""
        master = [_e("A", "r", "B"), _e("C", "r", "D")]
        student = [_e("A", "r", "B")]

        gc = GraphComparator()
        # Use exact fallback
        with patch("forma.graph_comparator.encode_texts", side_effect=Exception):
            result = gc.compare("s001", 1, master, student)

        assert len(result.matched_edges) == 1
        assert len(result.missing_edges) == 1
        assert result.recall == pytest.approx(0.5)

    def test_extra_edges(self):
        """Student has extra edges → precision < 1."""
        master = [_e("A", "r", "B")]
        student = [_e("A", "r", "B"), _e("X", "r", "Y")]

        gc = GraphComparator()
        with patch("forma.graph_comparator.encode_texts", side_effect=Exception):
            result = gc.compare("s001", 1, master, student)

        assert len(result.extra_edges) == 1
        assert result.precision == pytest.approx(0.5)

    def test_wrong_direction_detected(self):
        """Reversed edge detected as wrong direction."""
        master = [_e("A", "r", "B")]
        student = [_e("B", "r", "A")]  # reversed

        gc = GraphComparator()
        with patch("forma.graph_comparator.encode_texts", side_effect=Exception):
            result = gc.compare("s001", 1, master, student)

        assert len(result.wrong_direction_edges) == 1
        assert len(result.matched_edges) == 0

    def test_node_aliases(self):
        """Node aliases resolve to canonical names."""
        gc = GraphComparator(node_aliases={"항상성": ["homeostasis", "호메오스타시스"]})
        master = [_e("항상성", "유지", "체온")]
        student = [_e("homeostasis", "유지", "체온")]

        with patch("forma.graph_comparator.encode_texts", side_effect=Exception):
            result = gc.compare("s001", 1, master, student)

        assert len(result.matched_edges) == 1

    def test_lecture_excluded_edges(self):
        """Edges with uncovered concepts excluded from P/R/F1."""
        master = [
            _e("A", "r", "B"),
            _e("X", "r", "Y"),  # X, Y not in lecture
        ]
        student = [_e("A", "r", "B")]

        gc = GraphComparator()
        with patch("forma.graph_comparator.encode_texts", side_effect=Exception):
            result = gc.compare(
                "s001",
                1,
                master,
                student,
                lecture_covered_concepts=["A", "B"],
            )

        assert len(result.lecture_excluded_edges) == 1
        assert result.recall == pytest.approx(1.0)  # Only 1 effective edge

    def test_student_id_preserved(self):
        """student_id and question_sn preserved in result."""
        gc = GraphComparator()
        result = gc.compare("s042", 3, [], [])
        assert result.student_id == "s042"
        assert result.question_sn == 3

    def test_returns_graph_comparison_result(self):
        """Return type is GraphComparisonResult."""
        gc = GraphComparator()
        result = gc.compare("s001", 1, [], [])
        assert isinstance(result, GraphComparisonResult)

    def test_no_student_edges(self):
        """No student edges → recall = 0, precision = 0."""
        master = [_e("A", "r", "B")]
        gc = GraphComparator()
        with patch("forma.graph_comparator.encode_texts", side_effect=Exception):
            result = gc.compare("s001", 1, master, [])
        assert result.recall == pytest.approx(0.0)
        assert len(result.missing_edges) == 1

    def test_no_master_edges(self):
        """No master edges → all student edges are extra."""
        student = [_e("A", "r", "B")]
        gc = GraphComparator()
        with patch("forma.graph_comparator.encode_texts", side_effect=Exception):
            result = gc.compare("s001", 1, [], student)
        assert len(result.extra_edges) == 1
