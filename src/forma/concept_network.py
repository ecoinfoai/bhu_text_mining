"""Concept network graph construction and delivery overlay.

Builds a network of domain concepts connected by shared key terms
and semantic similarity, then overlays delivery quality data for
per-section visualization.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from forma.domain_concept_extractor import DomainConcept
from forma.domain_coverage_analyzer import DeliveryAnalysis
from forma.embedding_cache import encode_texts

logger = logging.getLogger(__name__)

__all__ = [
    "ConceptNode",
    "ConceptEdge",
    "ConceptNetwork",
    "build_concept_network",
    "overlay_delivery",
]


@dataclass
class ConceptNode:
    """A node in the concept network.

    Attributes:
        concept: Concept name (matches DomainConcept.concept).
        chapter: Source chapter identifier.
        importance: "high" / "medium" / "low".
        major_topic: Major topic grouping.
        delivery_quality: Delivery quality score 0.0-1.0.
        delivery_status: Delivery status label.
        key_terms: Domain-specific key terms.
    """

    concept: str
    chapter: str
    importance: str
    major_topic: str
    delivery_quality: float = 0.0
    delivery_status: str = ""
    key_terms: list[str] = field(default_factory=list)


@dataclass
class ConceptEdge:
    """An edge in the concept network.

    Attributes:
        source: Source concept name.
        target: Target concept name.
        relationship: Edge type ("shared_terms" or "semantic").
        weight: Edge weight (0.0-1.0).
    """

    source: str
    target: str
    relationship: str  # "shared_terms" | "semantic"
    weight: float


@dataclass
class ConceptNetwork:
    """A graph of domain concepts with typed edges.

    Attributes:
        nodes: List of concept nodes.
        edges: List of concept edges.
    """

    nodes: list[ConceptNode]
    edges: list[ConceptEdge]


def build_concept_network(
    concepts: list[DomainConcept],
    similarity_threshold: float = 0.6,
    min_shared_terms: int = 2,
) -> ConceptNetwork:
    """Build a concept network from domain concepts.

    Creates edges based on shared key terms and semantic similarity.
    When there are more than 30 concepts, filters to high + medium
    importance only.

    Args:
        concepts: List of DomainConcept from textbook extraction.
        similarity_threshold: Minimum cosine similarity for semantic
            edges (default 0.6).
        min_shared_terms: Minimum shared key_terms for term-based
            edges (default 2).

    Returns:
        ConceptNetwork with nodes and edges.
    """
    # Filter by importance if > 30 concepts
    if len(concepts) > 30:
        concepts = [
            c for c in concepts if c.importance in ("high", "medium")
        ]
        logger.info(
            "Filtered to %d high/medium concepts (>30 total)", len(concepts),
        )

    # Build nodes
    nodes = [
        ConceptNode(
            concept=c.concept,
            chapter=c.chapter,
            importance=c.importance,
            major_topic=c.major_topic,
            key_terms=list(c.key_terms),
        )
        for c in concepts
    ]

    edges: list[ConceptEdge] = []

    # Shared terms edges
    for i, j in combinations(range(len(concepts)), 2):
        c1, c2 = concepts[i], concepts[j]
        terms1 = set(c1.key_terms)
        terms2 = set(c2.key_terms)
        shared = terms1 & terms2
        if len(shared) >= min_shared_terms:
            weight = len(shared) / min(len(terms1), len(terms2))
            edges.append(ConceptEdge(
                source=c1.concept,
                target=c2.concept,
                relationship="shared_terms",
                weight=weight,
            ))

    # Semantic edges via embeddings
    if len(concepts) >= 2:
        try:
            texts = [c.concept for c in concepts]
            embeddings = encode_texts(texts)
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized = embeddings / norms
            sim_matrix = normalized @ normalized.T

            # Check existing shared_terms edges to avoid duplicates
            existing_pairs: set[tuple[str, str]] = set()
            for e in edges:
                existing_pairs.add((e.source, e.target))
                existing_pairs.add((e.target, e.source))

            for i, j in combinations(range(len(concepts)), 2):
                pair = (concepts[i].concept, concepts[j].concept)
                if pair in existing_pairs:
                    continue
                cosine = float(sim_matrix[i, j])
                if cosine > similarity_threshold:
                    edges.append(ConceptEdge(
                        source=concepts[i].concept,
                        target=concepts[j].concept,
                        relationship="semantic",
                        weight=cosine,
                    ))
        except Exception:
            logger.warning("Semantic edge computation failed", exc_info=True)

    return ConceptNetwork(nodes=nodes, edges=edges)


def overlay_delivery(
    network: ConceptNetwork,
    deliveries: list[DeliveryAnalysis],
    section_id: str,
) -> ConceptNetwork:
    """Overlay delivery data onto a concept network.

    For each node, finds the matching DeliveryAnalysis by concept name
    and section_id. Sets delivery_quality and delivery_status. If no
    match is found, defaults to quality=0.0 and status="미전달".

    Args:
        network: Base concept network.
        deliveries: List of DeliveryAnalysis to overlay.
        section_id: Section to filter deliveries by.

    Returns:
        New ConceptNetwork with delivery data set on nodes.
    """
    # Build lookup: concept -> DeliveryAnalysis for this section
    delivery_map: dict[str, DeliveryAnalysis] = {}
    for d in deliveries:
        if d.section_id == section_id:
            delivery_map[d.concept] = d

    # Deep copy to avoid mutating original
    new_nodes = []
    for node in network.nodes:
        new_node = copy.copy(node)
        new_node.key_terms = list(node.key_terms)
        if node.concept in delivery_map:
            d = delivery_map[node.concept]
            new_node.delivery_quality = d.delivery_quality
            new_node.delivery_status = d.delivery_status
        else:
            new_node.delivery_quality = 0.0
            new_node.delivery_status = "미전달"
        new_nodes.append(new_node)

    return ConceptNetwork(nodes=new_nodes, edges=list(network.edges))
