"""Knowledge graph comparison and metrics (student vs reference graph overlay)."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def display_comparison_results(
    comparison_results: dict[str, dict[str, list]]
) -> pd.DataFrame:
    """
    Create a summary DataFrame of knowledge graph comparisons.

    Takes a dictionary of comparison results between reference and student knowledge
    graphs and transforms it into a DataFrame. For each student, combines node
    differences as comma-separated strings and counts edge differences. Useful for
    identifying key structural differences in students' knowledge representation.

    Args:
        comparison_results (dict[str, dict[str, list]]): Dictionary containing
            comparison results for each student, with missing/extra nodes and edges.

    Returns:
        pd.DataFrame: DataFrame with columns for student name, missing/extra nodes
            (as comma-separated strings), and missing/extra edge counts.

    Examples:
        >>> results = {
        ...     'student1': {
        ...         'missing_nodes': ['concept1', 'concept2'],
        ...         'extra_nodes': ['concept3'],
        ...         'missing_edges': [('concept1', 'concept2')],
        ...         'extra_edges': []
        ...     }
        ... }
        >>> display_comparison_results(results)
           Student         Missing Nodes Extra Nodes  Missing Edges  Extra Edges
        0  student1  concept1, concept2    concept3             1            0
    """
    all_data = []
    for student, results in comparison_results.items():
        all_data.append(
            {
                "Student": student,
                "Missing Nodes": ", ".join(results["missing_nodes"]),
                "Extra Nodes": ", ".join(results["extra_nodes"]),
                "Missing Edges": len(results["missing_edges"]),
                "Extra Edges": len(results["extra_edges"]),
            }
        )
    results_df = pd.DataFrame(all_data)

    return results_df



# ---------------------------------------------------------------------------
# New metrics for multi-layer evaluation framework (Phase 3 additions)
# Existing functions above are unchanged.
# ---------------------------------------------------------------------------


def align_graph_nodes(
    G_ref: nx.Graph,
    G_student: nx.Graph,
    embeddings: dict[str, np.ndarray] | None = None,
    threshold: float = 0.75,
) -> dict[str, str | None]:
    """Align student graph nodes to reference graph nodes via Hungarian algorithm.

    If ``embeddings`` are provided, similarity is cosine similarity between
    node embedding vectors.  Otherwise falls back to exact string matching.
    One-to-many mappings are possible (multiple student nodes → same ref node);
    only matches above ``threshold`` are kept.

    Args:
        G_ref: Reference (professor) knowledge graph.
        G_student: Student knowledge graph.
        embeddings: Optional dict mapping node label → embedding vector.
            If None, exact string matching is used.
        threshold: Minimum cosine similarity to accept a match (default 0.75).

    Returns:
        Dict mapping each student node → matched reference node (or None).

    Examples:
        >>> mapping = align_graph_nodes(G_ref, G_student)
        >>> mapping["세포막"]  # exact match
        '세포막'
    """
    ref_nodes = list(G_ref.nodes())
    stu_nodes = list(G_student.nodes())

    if not ref_nodes or not stu_nodes:
        return {n: None for n in stu_nodes}

    if embeddings is not None:
        ref_embs = np.array(
            [embeddings.get(n, np.zeros(1)) for n in ref_nodes]
        )
        stu_embs = np.array(
            [embeddings.get(n, np.zeros(1)) for n in stu_nodes]
        )
        sim_matrix = cosine_similarity(stu_embs, ref_embs)
    else:
        sim_matrix = np.zeros((len(stu_nodes), len(ref_nodes)))
        for i, sn in enumerate(stu_nodes):
            for j, rn in enumerate(ref_nodes):
                sim_matrix[i, j] = 1.0 if sn == rn else 0.0

    # Hungarian algorithm on negative similarity (minimisation)
    cost = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[str, str | None] = {n: None for n in stu_nodes}
    for r, c in zip(row_ind, col_ind):
        if sim_matrix[r, c] >= threshold:
            mapping[stu_nodes[r]] = ref_nodes[c]
    return mapping


def compute_node_recall(
    G_ref: nx.Graph,
    G_student: nx.Graph,
    aligned_nodes: dict[str, str | None] | None = None,
) -> float:
    """Compute the fraction of reference nodes recalled by the student graph.

    Node recall = |matched_ref_nodes| / |ref_nodes|.  If ``aligned_nodes``
    is None, exact string matching is used (equivalent to align_graph_nodes
    with no embeddings).

    Args:
        G_ref: Reference knowledge graph.
        G_student: Student knowledge graph.
        aligned_nodes: Optional pre-computed alignment from align_graph_nodes.

    Returns:
        Recall value in [0, 1].  Returns 0.0 if G_ref has no nodes.

    Examples:
        >>> compute_node_recall(G_ref, G_student)
        0.75
    """
    ref_nodes = set(G_ref.nodes())
    if not ref_nodes:
        return 0.0

    if aligned_nodes is not None:
        matched_refs = {v for v in aligned_nodes.values() if v is not None}
    else:
        matched_refs = ref_nodes & set(G_student.nodes())

    return len(matched_refs) / len(ref_nodes)


def compute_edge_jaccard(
    G_ref: nx.Graph,
    G_student: nx.Graph,
) -> float:
    """Compute Jaccard similarity of edge sets between two graphs.

    Jaccard = |E_ref ∩ E_stu| / |E_ref ∪ E_stu|.

    Args:
        G_ref: Reference knowledge graph.
        G_student: Student knowledge graph.

    Returns:
        Jaccard coefficient in [0, 1].  Returns 0.0 if both are edge-less.

    Examples:
        >>> compute_edge_jaccard(G_ref, G_student)
        0.5
    """
    ref_edges = {frozenset(e) for e in G_ref.edges()}
    stu_edges = {frozenset(e) for e in G_student.edges()}
    union = ref_edges | stu_edges
    if not union:
        return 0.0
    intersection = ref_edges & stu_edges
    return len(intersection) / len(union)


def compute_centrality_deviation(
    G_ref: nx.Graph,
    G_student: nx.Graph,
) -> float:
    """Compute mean absolute deviation in degree centrality between graphs.

    Only nodes present in G_ref are considered.  Missing student nodes are
    assigned centrality 0.  If |E_s| = 0, returns 1.0 (maximum deviation).

    Args:
        G_ref: Reference knowledge graph.
        G_student: Student knowledge graph.

    Returns:
        Mean absolute deviation in [0, 1].  Lower is better.

    Examples:
        >>> compute_centrality_deviation(G_ref, G_student)
        0.25
    """
    ref_nodes = list(G_ref.nodes())
    if not ref_nodes:
        return 0.0

    if G_student.number_of_edges() == 0:
        return 1.0

    ref_centrality = nx.degree_centrality(G_ref)
    stu_centrality = nx.degree_centrality(G_student)

    deviations = [
        abs(ref_centrality[n] - stu_centrality.get(n, 0.0))
        for n in ref_nodes
    ]
    return float(np.mean(deviations))


def compute_normalized_ged(
    G_ref: nx.Graph,
    G_student: nx.Graph,
    timeout: int = 30,
) -> float | None:
    """Compute normalised Graph Edit Distance with timeout and fallback.

    Uses NetworkX ``optimize_graph_edit_distance`` (approximate) with a
    ``timeout``-second limit.  Falls back to exact ``graph_edit_distance``
    for small graphs (≤8 nodes each).

    Normalisation: GED / (|V_ref| + |V_stu| + |E_ref| + |E_stu|).

    Args:
        G_ref: Reference knowledge graph.
        G_student: Student knowledge graph.
        timeout: Maximum seconds before returning None (default 30).

    Returns:
        Normalised GED in [0, 1], or None if computation timed out.

    Examples:
        >>> compute_normalized_ged(G_ref, G_student, timeout=5)
        0.3
    """
    import signal

    n_ref = G_ref.number_of_nodes()
    n_stu = G_student.number_of_nodes()
    e_ref = G_ref.number_of_edges()
    e_stu = G_student.number_of_edges()
    denom = n_ref + n_stu + e_ref + e_stu
    if denom == 0:
        return 0.0

    def _handler(_signum: int, _frame: object) -> None:
        raise TimeoutError("GED timeout")

    ged_value: float | None = None
    try:
        if n_ref <= 8 and n_stu <= 8:
            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(timeout)
            try:
                ged_value = float(nx.graph_edit_distance(G_ref, G_student))
            finally:
                signal.alarm(0)
        else:
            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(timeout)
            try:
                for v in nx.optimize_graph_edit_distance(G_ref, G_student):
                    ged_value = float(v)
            finally:
                signal.alarm(0)
    except TimeoutError:
        return None

    if ged_value is None:
        return None
    return min(ged_value / denom, 1.0)
