"""Tests for new knowledge_graph_analysis.py metric functions.

RED phase: validates align_graph_nodes, compute_node_recall,
compute_edge_jaccard, compute_centrality_deviation, compute_normalized_ged.
"""

import networkx as nx
import numpy as np
import pytest


def _make_triangle() -> nx.Graph:
    """Return a small triangle graph: A-B-C-A."""
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    return G


def _make_edge_ab() -> nx.Graph:
    """Return a graph with only A-B edge."""
    G = nx.Graph()
    G.add_edge("A", "B")
    return G


# ---------------------------------------------------------------------------
# align_graph_nodes
# ---------------------------------------------------------------------------


class TestAlignGraphNodes:
    """Tests for align_graph_nodes()."""

    def test_exact_match_returns_same_node(self):
        """Exact string match maps node to itself."""
        from src.knowledge_graph_analysis import align_graph_nodes

        G_ref = nx.Graph()
        G_ref.add_nodes_from(["세포막", "삼투"])
        G_stu = nx.Graph()
        G_stu.add_nodes_from(["세포막"])
        mapping = align_graph_nodes(G_ref, G_stu)
        assert mapping["세포막"] == "세포막"

    def test_no_match_below_threshold(self):
        """Non-matching nodes return None."""
        from src.knowledge_graph_analysis import align_graph_nodes

        G_ref = nx.Graph()
        G_ref.add_node("A")
        G_stu = nx.Graph()
        G_stu.add_node("Z")
        mapping = align_graph_nodes(G_ref, G_stu)
        assert mapping["Z"] is None

    def test_empty_student_graph_returns_empty(self):
        """Empty student graph returns empty dict."""
        from src.knowledge_graph_analysis import align_graph_nodes

        G_ref = _make_triangle()
        G_stu = nx.Graph()
        mapping = align_graph_nodes(G_ref, G_stu)
        assert mapping == {}

    def test_empty_ref_graph_maps_all_none(self):
        """Empty reference graph maps all student nodes to None."""
        from src.knowledge_graph_analysis import align_graph_nodes

        G_ref = nx.Graph()
        G_stu = nx.Graph()
        G_stu.add_nodes_from(["X", "Y"])
        mapping = align_graph_nodes(G_ref, G_stu)
        assert all(v is None for v in mapping.values())


# ---------------------------------------------------------------------------
# compute_node_recall
# ---------------------------------------------------------------------------


class TestComputeNodeRecall:
    """Tests for compute_node_recall()."""

    def test_perfect_recall(self):
        """Student graph with all reference nodes → recall=1.0."""
        from src.knowledge_graph_analysis import compute_node_recall

        G_ref = nx.Graph()
        G_ref.add_nodes_from(["A", "B", "C"])
        G_stu = nx.Graph()
        G_stu.add_nodes_from(["A", "B", "C", "D"])
        assert compute_node_recall(G_ref, G_stu) == pytest.approx(1.0)

    def test_zero_recall_no_overlap(self):
        """No shared nodes → recall=0.0."""
        from src.knowledge_graph_analysis import compute_node_recall

        G_ref = nx.Graph()
        G_ref.add_nodes_from(["A", "B"])
        G_stu = nx.Graph()
        G_stu.add_nodes_from(["X", "Y"])
        assert compute_node_recall(G_ref, G_stu) == pytest.approx(0.0)

    def test_partial_recall(self):
        """Half reference nodes present → recall=0.5."""
        from src.knowledge_graph_analysis import compute_node_recall

        G_ref = nx.Graph()
        G_ref.add_nodes_from(["A", "B"])
        G_stu = nx.Graph()
        G_stu.add_node("A")
        assert compute_node_recall(G_ref, G_stu) == pytest.approx(0.5)

    def test_empty_ref_returns_zero(self):
        """Empty reference graph → recall=0.0."""
        from src.knowledge_graph_analysis import compute_node_recall

        G_ref = nx.Graph()
        G_stu = nx.Graph()
        G_stu.add_node("X")
        assert compute_node_recall(G_ref, G_stu) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_edge_jaccard
# ---------------------------------------------------------------------------


class TestComputeEdgeJaccard:
    """Tests for compute_edge_jaccard()."""

    def test_identical_graphs_jaccard_one(self):
        """Identical edge sets → Jaccard=1.0."""
        from src.knowledge_graph_analysis import compute_edge_jaccard

        G = _make_triangle()
        assert compute_edge_jaccard(G, G.copy()) == pytest.approx(1.0)

    def test_no_shared_edges_jaccard_zero(self):
        """Disjoint edge sets → Jaccard=0.0."""
        from src.knowledge_graph_analysis import compute_edge_jaccard

        G_ref = nx.Graph()
        G_ref.add_edge("A", "B")
        G_stu = nx.Graph()
        G_stu.add_edge("X", "Y")
        assert compute_edge_jaccard(G_ref, G_stu) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partial overlap → Jaccard between 0 and 1."""
        from src.knowledge_graph_analysis import compute_edge_jaccard

        G_ref = _make_triangle()  # A-B, B-C, C-A
        G_stu = _make_edge_ab()   # A-B only
        jac = compute_edge_jaccard(G_ref, G_stu)
        assert 0.0 < jac < 1.0

    def test_both_edgeless_returns_zero(self):
        """Both graphs have no edges → Jaccard=0.0."""
        from src.knowledge_graph_analysis import compute_edge_jaccard

        G1, G2 = nx.Graph(), nx.Graph()
        G1.add_node("A")
        G2.add_node("A")
        assert compute_edge_jaccard(G1, G2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_centrality_deviation
# ---------------------------------------------------------------------------


class TestComputeCentralityDeviation:
    """Tests for compute_centrality_deviation()."""

    def test_identical_graphs_deviation_zero(self):
        """Identical graphs → deviation=0.0."""
        from src.knowledge_graph_analysis import compute_centrality_deviation

        G = _make_triangle()
        assert compute_centrality_deviation(G, G.copy()) == pytest.approx(0.0)

    def test_edgeless_student_graph_returns_one(self):
        """Student with no edges → CD=1.0 (defined)."""
        from src.knowledge_graph_analysis import compute_centrality_deviation

        G_ref = _make_triangle()
        G_stu = nx.Graph()
        G_stu.add_nodes_from(["A", "B", "C"])
        cd = compute_centrality_deviation(G_ref, G_stu)
        assert cd == pytest.approx(1.0)

    def test_empty_ref_returns_zero(self):
        """Empty reference → deviation=0.0."""
        from src.knowledge_graph_analysis import compute_centrality_deviation

        G_ref = nx.Graph()
        G_stu = _make_triangle()
        assert compute_centrality_deviation(G_ref, G_stu) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_normalized_ged
# ---------------------------------------------------------------------------


class TestComputeNormalizedGed:
    """Tests for compute_normalized_ged()."""

    def test_identical_small_graphs_ged_zero(self):
        """Identical small graphs → normalised GED=0.0."""
        from src.knowledge_graph_analysis import compute_normalized_ged

        G = _make_edge_ab()
        result = compute_normalized_ged(G, G.copy(), timeout=10)
        assert result == pytest.approx(0.0) or result is None

    def test_empty_graphs_return_zero(self):
        """Both empty graphs → GED=0.0."""
        from src.knowledge_graph_analysis import compute_normalized_ged

        G1, G2 = nx.Graph(), nx.Graph()
        result = compute_normalized_ged(G1, G2, timeout=5)
        assert result == pytest.approx(0.0)

    def test_result_in_zero_one_range(self):
        """GED is normalised to [0, 1]."""
        from src.knowledge_graph_analysis import compute_normalized_ged

        G_ref = _make_edge_ab()
        G_stu = nx.Graph()
        G_stu.add_edge("A", "C")
        result = compute_normalized_ged(G_ref, G_stu, timeout=10)
        if result is not None:
            assert 0.0 <= result <= 1.0

    def test_returns_none_or_float(self):
        """Return type is float or None."""
        from src.knowledge_graph_analysis import compute_normalized_ged

        G = _make_edge_ab()
        result = compute_normalized_ged(G, G.copy(), timeout=5)
        assert result is None or isinstance(result, float)
