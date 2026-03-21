"""Tests for concept network graph construction and delivery overlay.

T048: TestBuildConceptNetwork — shared terms, semantic edges, filtering
T049: TestOverlayDelivery — delivery quality overlay on network nodes
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from forma.domain_concept_extractor import DomainConcept
from forma.domain_coverage_analyzer import DeliveryAnalysis


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _make_domain_concept(
    concept: str,
    key_terms: list[str] | None = None,
    importance: str = "high",
    chapter: str = "3장",
    major_topic: str = "피부 구조",
) -> DomainConcept:
    return DomainConcept(
        concept=concept,
        description=f"{concept} 설명",
        key_terms=key_terms or [],
        importance=importance,
        chapter=chapter,
        major_topic=major_topic,
    )


def _make_delivery(
    concept: str,
    section_id: str = "A",
    delivery_quality: float = 0.8,
    delivery_status: str = "충분히 설명",
) -> DeliveryAnalysis:
    return DeliveryAnalysis(
        concept=concept,
        section_id=section_id,
        delivery_status=delivery_status,
        delivery_quality=delivery_quality,
        evidence="증거",
        depth="상세",
    )


# ----------------------------------------------------------------
# T048: TestBuildConceptNetwork
# ----------------------------------------------------------------


class TestBuildConceptNetwork:
    """Tests for build_concept_network()."""

    def test_shared_terms_edge(self) -> None:
        """Two concepts sharing 2+ key_terms produce a shared_terms edge."""
        from forma.concept_network import build_concept_network

        concepts = [
            _make_domain_concept("표피 구조", key_terms=["표피", "각질층", "기저층"]),
            _make_domain_concept("표피 기능", key_terms=["표피", "각질층", "보호"]),
        ]

        with patch("forma.concept_network.encode_texts") as mock_enc:
            # Return orthogonal vectors so no semantic edge
            mock_enc.return_value = np.array([
                [1.0, 0.0],
                [0.0, 1.0],
            ])
            network = build_concept_network(concepts, min_shared_terms=2)

        # Should have exactly 1 shared_terms edge
        shared_edges = [e for e in network.edges if e.relationship == "shared_terms"]
        assert len(shared_edges) == 1
        edge = shared_edges[0]
        assert {edge.source, edge.target} == {"표피 구조", "표피 기능"}
        # weight = |intersection| / min(len1, len2) = 2/3
        assert abs(edge.weight - 2 / 3) < 0.01

    def test_no_edge_below_threshold(self) -> None:
        """Concepts sharing only 1 key_term produce no shared_terms edge."""
        from forma.concept_network import build_concept_network

        concepts = [
            _make_domain_concept("표피 구조", key_terms=["표피", "각질층"]),
            _make_domain_concept("진피 기능", key_terms=["표피", "콜라겐"]),
        ]

        with patch("forma.concept_network.encode_texts") as mock_enc:
            mock_enc.return_value = np.array([
                [1.0, 0.0],
                [0.0, 1.0],
            ])
            network = build_concept_network(concepts, min_shared_terms=2)

        shared_edges = [e for e in network.edges if e.relationship == "shared_terms"]
        assert len(shared_edges) == 0

    def test_semantic_edge(self) -> None:
        """High cosine similarity between concept texts creates semantic edge."""
        from forma.concept_network import build_concept_network

        concepts = [
            _make_domain_concept("세포막 구조", key_terms=["세포막"]),
            _make_domain_concept("세포벽 기능", key_terms=["세포벽"]),
        ]

        with patch("forma.concept_network.encode_texts") as mock_enc:
            # Cosine similarity = 0.9 (above 0.6 threshold)
            v1 = np.array([1.0, 0.1])
            v2 = np.array([0.95, 0.15])
            mock_enc.return_value = np.vstack([v1, v2])
            network = build_concept_network(concepts, similarity_threshold=0.6)

        semantic_edges = [e for e in network.edges if e.relationship == "semantic"]
        assert len(semantic_edges) == 1

    def test_importance_filtering(self) -> None:
        """When > 30 nodes, only high + medium importance are kept."""
        from forma.concept_network import build_concept_network

        concepts = []
        # 20 high, 10 medium, 5 low = 35 total > 30
        for i in range(20):
            concepts.append(
                _make_domain_concept(f"high_{i}", importance="high", key_terms=[f"t{i}"])
            )
        for i in range(10):
            concepts.append(
                _make_domain_concept(f"med_{i}", importance="medium", key_terms=[f"m{i}"])
            )
        for i in range(5):
            concepts.append(
                _make_domain_concept(f"low_{i}", importance="low", key_terms=[f"l{i}"])
            )

        with patch("forma.concept_network.encode_texts") as mock_enc:
            n = 30  # high + medium
            mock_enc.return_value = np.eye(n)
            network = build_concept_network(concepts)

        # low importance nodes should be filtered out
        node_names = {n.concept for n in network.nodes}
        low_names = {f"low_{i}" for i in range(5)}
        assert not (node_names & low_names)
        assert len(network.nodes) == 30


# ----------------------------------------------------------------
# T049: TestOverlayDelivery
# ----------------------------------------------------------------


class TestOverlayDelivery:
    """Tests for overlay_delivery()."""

    def test_overlay_sets_quality(self) -> None:
        """Overlay sets delivery_quality and delivery_status on matching nodes."""
        from forma.concept_network import (
            ConceptNetwork,
            ConceptNode,
            overlay_delivery,
        )

        nodes = [
            ConceptNode(
                concept="표피 구조",
                chapter="3장",
                importance="high",
                major_topic="피부",
                key_terms=["표피"],
            ),
        ]
        network = ConceptNetwork(nodes=nodes, edges=[])
        deliveries = [_make_delivery("표피 구조", section_id="A", delivery_quality=0.85)]

        result = overlay_delivery(network, deliveries, section_id="A")
        assert result.nodes[0].delivery_quality == 0.85
        assert result.nodes[0].delivery_status == "충분히 설명"

    def test_missing_delivery_fallback(self) -> None:
        """Concept not in deliveries gets quality=0.0, status='미전달'."""
        from forma.concept_network import (
            ConceptNetwork,
            ConceptNode,
            overlay_delivery,
        )

        nodes = [
            ConceptNode(
                concept="없는 개념",
                chapter="3장",
                importance="high",
                major_topic="피부",
                key_terms=[],
            ),
        ]
        network = ConceptNetwork(nodes=nodes, edges=[])
        deliveries = [_make_delivery("다른 개념", section_id="A")]

        result = overlay_delivery(network, deliveries, section_id="A")
        assert result.nodes[0].delivery_quality == 0.0
        assert result.nodes[0].delivery_status == "미전달"
