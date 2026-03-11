"""Hub centrality gap analysis for concept network comparison."""
from __future__ import annotations

import logging

import networkx as nx

from forma.evaluation_types import HubGapEntry, TripletEdge

logger = logging.getLogger(__name__)


def compute_hub_gap(
    master_edges: list[TripletEdge],
    student_edges: list[TripletEdge],
    top_k: int = 10,
) -> list[HubGapEntry]:
    """Compute top-k hub concepts from master graph and check student coverage.

    Returns HubGapEntry list sorted by degree_centrality descending.

    Args:
        master_edges: Edges of the master (reference) knowledge graph.
        student_edges: Edges of the student's knowledge graph.
        top_k: Number of top hub concepts to return.

    Returns:
        List of HubGapEntry sorted by degree_centrality descending.
    """
    if not master_edges:
        return []

    # Build directed graph from master edges (skip self-loops to avoid centrality > 1.0)
    G = nx.DiGraph()
    for e in master_edges:
        if e.subject != e.object:
            G.add_edge(e.subject, e.object)

    # Compute degree centrality (normalized, includes in + out degree)
    centrality = nx.degree_centrality(G)

    # Sort by centrality descending, take top_k
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Build student concept set (subject or object in any student edge)
    student_concepts: set[str] = set()
    for e in student_edges:
        student_concepts.add(e.subject)
        student_concepts.add(e.object)

    # Build HubGapEntry list
    entries = []
    for concept, cent in top_nodes:
        entries.append(HubGapEntry(
            concept=concept,
            degree_centrality=cent,
            student_present=(concept in student_concepts),
            class_inclusion_rate=0.0,
        ))

    return entries


def compute_class_hub_gap(
    master_edges: list[TripletEdge],
    all_student_edges: dict[str, list[TripletEdge]],
    top_k: int = 10,
) -> list[HubGapEntry]:
    """Compute hub gap with class-level inclusion rates.

    Returns HubGapEntry list sorted by degree_centrality descending.

    Args:
        master_edges: Edges of the master (reference) knowledge graph.
        all_student_edges: Dict mapping student_id to their list of edges.
        top_k: Number of top hub concepts to return.

    Returns:
        List of HubGapEntry sorted by degree_centrality descending,
        with class_inclusion_rate set to the fraction of students that
        mentioned each concept.
    """
    if not master_edges:
        return []

    n_students = len(all_student_edges)

    # Build directed graph from master edges (skip self-loops to avoid centrality > 1.0)
    G = nx.DiGraph()
    for e in master_edges:
        if e.subject != e.object:
            G.add_edge(e.subject, e.object)

    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # For each top node, compute inclusion rate across all students
    entries = []
    for concept, cent in top_nodes:
        if n_students == 0:
            rate = 0.0
        else:
            count = sum(
                1 for edges in all_student_edges.values()
                if any(concept in (e.subject, e.object) for e in edges)
            )
            rate = min(1.0, count / n_students)
        entries.append(HubGapEntry(
            concept=concept,
            degree_centrality=cent,
            student_present=False,
            class_inclusion_rate=rate,
        ))

    return entries
