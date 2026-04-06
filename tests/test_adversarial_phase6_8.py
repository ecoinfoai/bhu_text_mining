"""Adversarial tests for Phase 6-8 implementation.

7+ attack personas testing edge cases and failure modes:
1. Empty hierarchy (no summary)
2. Single section statistics
3. Zero-edge network
4. Colorblind accessibility (cividis)
5. 30+ node congestion filtering
6. Isolated nodes (no edges)
7. Mismatched section_to_major mapping
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np

from forma.domain_concept_extractor import (
    DomainConcept,
    MajorTopic,
    SubTopic,
    TopicHierarchy,
)
from forma.domain_coverage_analyzer import (
    DeliveryAnalysis,
    DeliveryResult,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _dc(
    concept: str,
    key_terms: list[str] | None = None,
    importance: str = "high",
    chapter: str = "3장",
    major_topic: str = "피부",
) -> DomainConcept:
    return DomainConcept(
        concept=concept,
        description=f"{concept} 설명",
        key_terms=key_terms or [],
        importance=importance,
        chapter=chapter,
        major_topic=major_topic,
    )


def _da(
    concept: str,
    section_id: str = "A",
    quality: float = 0.8,
    status: str = "충분히 설명",
) -> DeliveryAnalysis:
    return DeliveryAnalysis(
        concept=concept,
        section_id=section_id,
        delivery_status=status,
        delivery_quality=quality,
        evidence="증거",
        depth="상세",
    )


def _is_valid_png(buf: io.BytesIO) -> bool:
    buf.seek(0)
    header = buf.read(8)
    return header[:4] == b"\x89PNG"


def _sample_result() -> DeliveryResult:
    return DeliveryResult(
        week=2,
        chapters=["3장"],
        deliveries=[
            _da("표피 구조", "A", 0.9),
            _da("표피 구조", "B", 0.5, "부분 전달"),
            _da("진피 기능", "A", 0.85),
            _da("진피 기능", "B", 0.0, "미전달"),
        ],
        effective_delivery_rate=0.75,
        per_section_rate={"A": 1.0, "B": 0.5},
    )


# ================================================================
# Attack 1: Empty hierarchy — report --summary not provided
# ================================================================


class TestAttack1EmptyHierarchy:
    """Report generation with hierarchy=None should skip hierarchical section."""

    def test_generate_pdf_no_hierarchy_no_crash(self, tmp_path: Path) -> None:
        """PDF generation without hierarchy must not crash."""
        from forma.domain_coverage_report import DomainDeliveryPDFReportGenerator

        result = _sample_result()
        output = str(tmp_path / "report.pdf")
        gen = DomainDeliveryPDFReportGenerator()
        path = gen.generate_pdf(result, output, hierarchy=None)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_stacked_chart_with_empty_hierarchy(self) -> None:
        """Stacked chart with empty major_topics should return placeholder."""
        from forma.domain_coverage_charts import build_topic_delivery_stacked_chart

        result = _sample_result()
        hierarchy = TopicHierarchy(
            major_topics=[],
            section_to_major={},
            section_to_sub={},
        )
        buf = build_topic_delivery_stacked_chart(result, hierarchy)
        assert _is_valid_png(buf)


# ================================================================
# Attack 2: Single section — statistics comparison should skip
# ================================================================


class TestAttack2SingleSection:
    """1 section only — pairwise comparisons must return empty list."""

    def test_1_section_returns_empty(self) -> None:
        from forma.domain_coverage_analyzer import (
            compute_delivery_pairwise_comparisons,
        )

        deliveries = [
            _da("표피", "A", 0.9),
            _da("진피", "A", 0.5),
        ]
        result = compute_delivery_pairwise_comparisons(deliveries)
        assert result == []

    def test_0_sections_returns_empty(self) -> None:
        from forma.domain_coverage_analyzer import (
            compute_delivery_pairwise_comparisons,
        )

        result = compute_delivery_pairwise_comparisons([])
        assert result == []

    def test_all_skipped_returns_empty(self) -> None:
        """All deliveries are 의도적 생략 — effectively 0 sections."""
        from forma.domain_coverage_analyzer import (
            compute_delivery_pairwise_comparisons,
        )

        deliveries = [
            DeliveryAnalysis(
                concept="표피",
                section_id="A",
                delivery_status="의도적 생략",
                delivery_quality=0.0,
                evidence="",
                depth="",
            ),
            DeliveryAnalysis(
                concept="진피",
                section_id="B",
                delivery_status="의도적 생략",
                delivery_quality=0.0,
                evidence="",
                depth="",
            ),
        ]
        result = compute_delivery_pairwise_comparisons(deliveries)
        assert result == []


# ================================================================
# Attack 3: Zero-edge network (no shared terms, low similarity)
# ================================================================


class TestAttack3ZeroEdgeNetwork:
    """All concepts have unique key_terms, low semantic similarity."""

    def test_zero_edges_network(self) -> None:
        from forma.concept_network import build_concept_network

        concepts = [
            _dc("세포막 구조", key_terms=["세포막", "인지질"]),
            _dc("소화 효소", key_terms=["아밀라아제", "펩신"]),
            _dc("호흡 기관", key_terms=["폐포", "기관지"]),
        ]

        with patch("forma.concept_network.encode_texts") as mock_enc:
            # Orthogonal vectors — no semantic edge
            mock_enc.return_value = np.eye(3)
            network = build_concept_network(concepts, min_shared_terms=2)

        assert len(network.edges) == 0
        assert len(network.nodes) == 3

    def test_zero_edge_chart_renders(self) -> None:
        """Chart must render even with 0 edges."""
        from forma.concept_network import ConceptNetwork, ConceptNode
        from forma.domain_coverage_charts import build_concept_network_chart

        nodes = [
            ConceptNode(
                concept="세포막",
                chapter="1장",
                importance="high",
                major_topic="세포",
                delivery_quality=0.5,
            ),
            ConceptNode(
                concept="소화",
                chapter="2장",
                importance="medium",
                major_topic="소화기",
                delivery_quality=0.3,
            ),
        ]
        network = ConceptNetwork(nodes=nodes, edges=[])
        buf = build_concept_network_chart(network)
        assert _is_valid_png(buf)


# ================================================================
# Attack 4: Colorblind accessibility (FR-024: cividis required)
# ================================================================


class TestAttack4ColorblindAccessibility:
    """Verify all quality-based coloring uses cividis, not RdYlGn."""

    def test_grouped_heatmap_uses_cividis(self) -> None:
        """build_grouped_quality_heatmap must use cividis colormap."""
        import forma.domain_coverage_charts as mod
        import inspect

        source = inspect.getsource(mod.build_grouped_quality_heatmap)
        assert "cividis" in source
        assert 'cmap="RdYlGn"' not in source

    def test_concept_network_chart_uses_cividis(self) -> None:
        """build_concept_network_chart must use cividis."""
        import forma.domain_coverage_charts as mod
        import inspect

        source = inspect.getsource(mod.build_concept_network_chart)
        assert "cividis" in source

    def test_concept_network_comparison_uses_cividis(self) -> None:
        """build_concept_network_comparison must use cividis."""
        import forma.domain_coverage_charts as mod
        import inspect

        source = inspect.getsource(mod.build_concept_network_comparison)
        assert "cividis" in source


# ================================================================
# Attack 5: 30+ node congestion — importance filtering
# ================================================================


class TestAttack5ThirtyPlusNodes:
    """35 concepts should filter to high+medium only."""

    def test_filtering_reduces_nodes(self) -> None:
        from forma.concept_network import build_concept_network

        concepts = []
        for i in range(15):
            concepts.append(_dc(f"high_{i}", importance="high", key_terms=[f"h{i}"]))
        for i in range(10):
            concepts.append(_dc(f"med_{i}", importance="medium", key_terms=[f"m{i}"]))
        for i in range(10):
            concepts.append(_dc(f"low_{i}", importance="low", key_terms=[f"l{i}"]))

        assert len(concepts) == 35

        with patch("forma.concept_network.encode_texts") as mock_enc:
            mock_enc.return_value = np.eye(25)  # 15 high + 10 medium
            network = build_concept_network(concepts)

        assert len(network.nodes) == 25
        for node in network.nodes:
            assert node.importance in ("high", "medium")

    def test_exactly_30_no_filter(self) -> None:
        """Exactly 30 concepts should NOT trigger filtering."""
        from forma.concept_network import build_concept_network

        concepts = []
        for i in range(20):
            concepts.append(_dc(f"high_{i}", importance="high", key_terms=[f"h{i}"]))
        for i in range(5):
            concepts.append(_dc(f"med_{i}", importance="medium", key_terms=[f"m{i}"]))
        for i in range(5):
            concepts.append(_dc(f"low_{i}", importance="low", key_terms=[f"l{i}"]))

        assert len(concepts) == 30

        with patch("forma.concept_network.encode_texts") as mock_enc:
            mock_enc.return_value = np.eye(30)
            network = build_concept_network(concepts)

        # Should keep all 30 (not > 30)
        assert len(network.nodes) == 30
        low_nodes = [n for n in network.nodes if n.importance == "low"]
        assert len(low_nodes) == 5

    def test_31_triggers_filter(self) -> None:
        """31 concepts should trigger filtering."""
        from forma.concept_network import build_concept_network

        concepts = []
        for i in range(20):
            concepts.append(_dc(f"high_{i}", importance="high", key_terms=[f"h{i}"]))
        for i in range(5):
            concepts.append(_dc(f"med_{i}", importance="medium", key_terms=[f"m{i}"]))
        for i in range(6):
            concepts.append(_dc(f"low_{i}", importance="low", key_terms=[f"l{i}"]))

        assert len(concepts) == 31

        with patch("forma.concept_network.encode_texts") as mock_enc:
            mock_enc.return_value = np.eye(25)
            network = build_concept_network(concepts)

        assert len(network.nodes) == 25
        low_nodes = [n for n in network.nodes if n.importance == "low"]
        assert len(low_nodes) == 0


# ================================================================
# Attack 6: Isolated nodes (0 edges, layout must handle)
# ================================================================


class TestAttack6IsolatedNodes:
    """Network with isolated nodes (no edges) should render cleanly."""

    def test_all_isolated_nodes_chart(self) -> None:
        from forma.concept_network import ConceptNetwork, ConceptNode
        from forma.domain_coverage_charts import build_concept_network_chart

        nodes = [
            ConceptNode(
                concept=f"개념_{i}",
                chapter="1장",
                importance="high",
                major_topic="기타",
                delivery_quality=i * 0.2,
                key_terms=[],
            )
            for i in range(5)
        ]
        network = ConceptNetwork(nodes=nodes, edges=[])
        buf = build_concept_network_chart(network)
        assert _is_valid_png(buf)

    def test_isolated_nodes_comparison(self) -> None:
        from forma.concept_network import ConceptNetwork, ConceptNode
        from forma.domain_coverage_charts import build_concept_network_comparison

        nodes = [
            ConceptNode(
                concept="독립개념A",
                chapter="1장",
                importance="high",
                major_topic="기타",
                key_terms=[],
            ),
            ConceptNode(
                concept="독립개념B",
                chapter="1장",
                importance="medium",
                major_topic="기타",
                key_terms=[],
            ),
        ]
        network = ConceptNetwork(nodes=nodes, edges=[])
        deliveries_by_section = {
            "A": [_da("독립개념A", "A", 0.9), _da("독립개념B", "A", 0.1, "미전달")],
            "B": [_da("독립개념A", "B", 0.3, "부분 전달"), _da("독립개념B", "B", 0.7)],
        }
        buf = build_concept_network_comparison(network, deliveries_by_section)
        assert _is_valid_png(buf)

    def test_empty_network_chart(self) -> None:
        """Completely empty network (0 nodes) should show placeholder text."""
        from forma.concept_network import ConceptNetwork
        from forma.domain_coverage_charts import build_concept_network_chart

        network = ConceptNetwork(nodes=[], edges=[])
        buf = build_concept_network_chart(network)
        assert _is_valid_png(buf)


# ================================================================
# Attack 7: Mismatched section_to_major mapping
# ================================================================


class TestAttack7MismatchedMapping:
    """section_to_major keys don't match any concept names in deliveries."""

    def test_all_concepts_fall_to_other(self) -> None:
        """When no mapping matches, concepts go to '기타' bucket."""
        from forma.domain_coverage_charts import build_topic_delivery_stacked_chart

        result = DeliveryResult(
            week=2,
            chapters=["3장"],
            deliveries=[
                _da("완전히_다른_개념명", "A", 0.9),
                _da("매칭불가_개념", "B", 0.5, "부분 전달"),
            ],
            effective_delivery_rate=0.7,
            per_section_rate={"A": 0.9, "B": 0.5},
        )
        hierarchy = TopicHierarchy(
            major_topics=[
                MajorTopic(
                    name="전혀 관련없는 주제",
                    sub_topics=[SubTopic(name="없는소주제", sections=[])],
                ),
            ],
            section_to_major={"없는키": "전혀 관련없는 주제"},
            section_to_sub={},
        )
        buf = build_topic_delivery_stacked_chart(result, hierarchy)
        assert _is_valid_png(buf)

    def test_grouped_heatmap_mismatched(self) -> None:
        """Grouped heatmap with no matching concepts still renders."""
        from forma.domain_coverage_charts import build_grouped_quality_heatmap

        result = DeliveryResult(
            week=2,
            chapters=["3장"],
            deliveries=[
                _da("X개념", "A", 0.8),
                _da("Y개념", "B", 0.3, "부분 전달"),
            ],
            effective_delivery_rate=0.55,
            per_section_rate={"A": 0.8, "B": 0.3},
        )
        hierarchy = TopicHierarchy(
            major_topics=[
                MajorTopic(
                    name="Z주제",
                    sub_topics=[SubTopic(name="Z소주제", sections=[])],
                ),
            ],
            section_to_major={"없는키워드": "Z주제"},
            section_to_sub={},
        )
        buf = build_grouped_quality_heatmap(result, hierarchy)
        assert _is_valid_png(buf)

    def test_hierarchical_coverage_chart_mismatched(self) -> None:
        """Grouped bar chart with no matching topics falls back to '기타'."""
        from forma.domain_coverage_charts import build_hierarchical_coverage_chart

        result = DeliveryResult(
            week=2,
            chapters=["3장"],
            deliveries=[
                _da("매칭불가A", "A", 0.7),
                _da("매칭불가B", "A", 0.4, "부분 전달"),
            ],
            effective_delivery_rate=0.55,
            per_section_rate={"A": 0.55},
        )
        hierarchy = TopicHierarchy(
            major_topics=[
                MajorTopic(
                    name="무관한 주제",
                    sub_topics=[SubTopic(name="무관한 소주제", sections=[])],
                ),
            ],
            section_to_major={"없는키": "무관한 주제"},
            section_to_sub={},
        )
        buf = build_hierarchical_coverage_chart(result, hierarchy)
        assert _is_valid_png(buf)


# ================================================================
# Bonus Attack 8: Overlay with duplicate concept names
# ================================================================


class TestAttack8DuplicateConcepts:
    """Multiple deliveries for same concept+section — last wins."""

    def test_overlay_last_delivery_wins(self) -> None:
        from forma.concept_network import ConceptNetwork, ConceptNode, overlay_delivery

        nodes = [
            ConceptNode(
                concept="표피",
                chapter="3장",
                importance="high",
                major_topic="피부",
                key_terms=[],
            ),
        ]
        network = ConceptNetwork(nodes=nodes, edges=[])
        deliveries = [
            _da("표피", "A", 0.3, "부분 전달"),
            _da("표피", "A", 0.9, "충분히 설명"),  # later entry
        ]
        result = overlay_delivery(network, deliveries, "A")
        # Last matching delivery should win
        assert result.nodes[0].delivery_quality == 0.9
        assert result.nodes[0].delivery_status == "충분히 설명"
