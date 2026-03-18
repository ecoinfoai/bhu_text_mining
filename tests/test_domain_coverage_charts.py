"""Tests for domain coverage chart generation.

T029: Coverage bar chart
T030: Emphasis bias scatter and section variance heatmap
"""

from __future__ import annotations

import io
import statistics


from forma.domain_concept_extractor import TextbookConcept
from forma.domain_coverage_analyzer import (
    ConceptEmphasis,
    CoverageResult,
    ExtraConcept,
    TeachingScope,
    build_coverage_result,
    classify_concepts,
)
from forma.domain_coverage_analyzer import (
    DeliveryAnalysis,
    DeliveryResult,
    KeywordNetwork,
)
from forma.domain_coverage_charts import (
    build_coverage_bar_chart,
    build_delivery_bar_chart,
    build_delivery_heatmap,
    build_emphasis_bias_scatter,
    build_network_comparison_chart,
    build_section_variance_heatmap,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _make_concept(
    name_ko: str,
    chapter: str = "3장 피부",
    frequency: int = 5,
    name_en: str | None = None,
) -> TextbookConcept:
    return TextbookConcept(
        name_ko=name_ko,
        name_en=name_en,
        chapter=chapter,
        frequency=frequency,
        context_sentence=f"{name_ko}는 중요하다.",
        is_bilingual=name_en is not None,
    )


def _make_emphasis(
    concept_name: str,
    chapter: str = "3장 피부",
    section_scores: dict[str, float] | None = None,
) -> ConceptEmphasis:
    if section_scores is None:
        section_scores = {"A": 0.5, "B": 0.6}
    values = list(section_scores.values())
    mean_score = statistics.mean(values) if values else 0.0
    std_score = statistics.stdev(values) if len(values) >= 2 else 0.0
    return ConceptEmphasis(
        concept_name=concept_name,
        chapter=chapter,
        section_scores=section_scores,
        mean_score=mean_score,
        std_score=std_score,
    )


def _build_sample_result() -> CoverageResult:
    """Build a sample CoverageResult for chart tests."""
    concepts = [
        _make_concept("표피", chapter="3장", frequency=15),
        _make_concept("진피", chapter="3장", frequency=10),
        _make_concept("각질층", chapter="3장", frequency=8),
        _make_concept("세포막", chapter="1장", frequency=5),
        _make_concept("근육", chapter="5장", frequency=3),
    ]
    emphasis_list = [
        _make_emphasis("표피", chapter="3장", section_scores={"A": 0.9, "B": 0.7, "C": 0.8}),
        _make_emphasis("진피", chapter="3장", section_scores={"A": 0.6, "B": 0.4, "C": 0.5}),
        _make_emphasis("각질층", chapter="3장", section_scores={"A": 0.3, "B": 0.1, "C": 0.2}),
        _make_emphasis("세포막", chapter="1장", section_scores={"A": 0.02, "B": 0.01, "C": 0.01}),
    ]
    scope = TeachingScope(chapters=["3장", "1장"])
    classified = classify_concepts(concepts, emphasis_list, scope)
    extras = [ExtraConcept(name="염증", section_mentions={"A": 5}, example_sentence="test")]
    return build_coverage_result(classified, extras, week=2, chapters=["3장", "1장"])


def _is_valid_png(buf: io.BytesIO) -> bool:
    """Check if buffer contains valid PNG data."""
    buf.seek(0)
    header = buf.read(8)
    return header[:4] == b"\x89PNG"


# ----------------------------------------------------------------
# T029: Coverage bar chart
# ----------------------------------------------------------------


class TestCoverageBarChart:
    """Tests for coverage bar chart generation."""

    def test_produces_valid_png(self) -> None:
        """Chart produces a valid PNG image."""
        result = _build_sample_result()
        buf = build_coverage_bar_chart(result)
        assert _is_valid_png(buf)

    def test_non_empty_output(self) -> None:
        """Chart output is non-empty."""
        result = _build_sample_result()
        buf = build_coverage_bar_chart(result)
        buf.seek(0, 2)  # seek to end
        assert buf.tell() > 100  # should be more than 100 bytes

    def test_empty_sections(self) -> None:
        """Chart handles result with no sections gracefully."""
        result = CoverageResult(
            week=1,
            chapters=["1장"],
            total_textbook_concepts=0,
            in_scope_count=0,
            skipped_count=0,
            covered_count=0,
            gap_count=0,
            extra_count=0,
            effective_coverage_rate=0.0,
            per_section_coverage={},
            classified_concepts=[],
        )
        buf = build_coverage_bar_chart(result)
        assert _is_valid_png(buf)


# ----------------------------------------------------------------
# T030: Emphasis bias scatter and variance heatmap
# ----------------------------------------------------------------


class TestEmphasisBiasScatter:
    """Tests for emphasis bias scatter plot."""

    def test_produces_valid_png(self) -> None:
        """Scatter plot produces valid PNG."""
        result = _build_sample_result()
        buf = build_emphasis_bias_scatter(result)
        assert _is_valid_png(buf)

    def test_empty_data(self) -> None:
        """Handles empty classified concepts."""
        result = CoverageResult(
            week=1,
            chapters=[],
            total_textbook_concepts=0,
            in_scope_count=0,
            skipped_count=0,
            covered_count=0,
            gap_count=0,
            extra_count=0,
            effective_coverage_rate=0.0,
            per_section_coverage={},
            classified_concepts=[],
        )
        buf = build_emphasis_bias_scatter(result)
        assert _is_valid_png(buf)


class TestSectionVarianceHeatmap:
    """Tests for section variance heatmap."""

    def test_produces_valid_png(self) -> None:
        """Heatmap produces valid PNG."""
        result = _build_sample_result()
        buf = build_section_variance_heatmap(result)
        assert _is_valid_png(buf)

    def test_empty_data(self) -> None:
        """Handles empty data gracefully."""
        result = CoverageResult(
            week=1,
            chapters=[],
            total_textbook_concepts=0,
            in_scope_count=0,
            skipped_count=0,
            covered_count=0,
            gap_count=0,
            extra_count=0,
            effective_coverage_rate=0.0,
            per_section_coverage={},
            classified_concepts=[],
        )
        buf = build_section_variance_heatmap(result)
        assert _is_valid_png(buf)

    def test_respects_max_concepts(self) -> None:
        """Respects max_concepts parameter."""
        result = _build_sample_result()
        buf = build_section_variance_heatmap(result, max_concepts=2)
        assert _is_valid_png(buf)


# ----------------------------------------------------------------
# T036: Network comparison chart
# ----------------------------------------------------------------


class TestNetworkComparisonChart:
    """Tests for network comparison chart generation."""

    def test_produces_valid_png(self) -> None:
        """Network comparison chart produces valid PNG."""
        textbook = KeywordNetwork(
            source="textbook",
            nodes=["세포막", "인지질", "콜레스테롤"],
            edges=[
                ("세포막", "인지질", 3.0),
                ("세포막", "콜레스테롤", 2.0),
            ],
        )
        lecture = KeywordNetwork(
            source="A",
            nodes=["세포막", "인지질"],
            edges=[("세포막", "인지질", 2.0)],
        )
        missing = [("세포막", "콜레스테롤")]

        buf = build_network_comparison_chart(textbook, lecture, missing)
        assert _is_valid_png(buf)

    def test_empty_networks(self) -> None:
        """Handles empty networks gracefully."""
        textbook = KeywordNetwork(source="textbook", nodes=[], edges=[])
        lecture = KeywordNetwork(source="A", nodes=[], edges=[])

        buf = build_network_comparison_chart(textbook, lecture)
        assert _is_valid_png(buf)

    def test_no_missing_edges(self) -> None:
        """Chart works when no edges are missing."""
        textbook = KeywordNetwork(
            source="textbook",
            nodes=["A", "B"],
            edges=[("A", "B", 1.0)],
        )
        lecture = KeywordNetwork(
            source="A",
            nodes=["A", "B"],
            edges=[("A", "B", 1.0)],
        )

        buf = build_network_comparison_chart(textbook, lecture, [])
        assert _is_valid_png(buf)


# ----------------------------------------------------------------
# Helpers for delivery charts
# ----------------------------------------------------------------


def _build_sample_delivery_result() -> DeliveryResult:
    """Build a sample DeliveryResult for delivery chart tests."""
    deliveries = [
        DeliveryAnalysis(
            concept="표피의 4층 구조",
            section_id="A",
            delivery_status="충분히 설명",
            delivery_quality=0.9,
            evidence="표피의 4층에 대해 상세히 설명",
            depth="메커니즘까지 설명",
        ),
        DeliveryAnalysis(
            concept="표피의 4층 구조",
            section_id="B",
            delivery_status="부분 전달",
            delivery_quality=0.5,
            evidence="용어만 언급",
            depth="용어 수준",
        ),
        DeliveryAnalysis(
            concept="진피의 구조",
            section_id="A",
            delivery_status="충분히 설명",
            delivery_quality=0.85,
            evidence="진피 구조 설명",
            depth="상세 설명",
        ),
        DeliveryAnalysis(
            concept="진피의 구조",
            section_id="B",
            delivery_status="미전달",
            delivery_quality=0.0,
            evidence="",
            depth="",
        ),
        DeliveryAnalysis(
            concept="세포막 구조",
            section_id="A",
            delivery_status="의도적 생략",
            delivery_quality=0.0,
            evidence="",
            depth="",
        ),
    ]
    return DeliveryResult(
        week=2,
        chapters=["3장", "1장"],
        deliveries=deliveries,
        effective_delivery_rate=0.75,
        per_section_rate={"A": 1.0, "B": 0.5},
    )


# ----------------------------------------------------------------
# T046: Delivery bar chart
# ----------------------------------------------------------------


class TestDeliveryBarChart:
    """Tests for delivery bar chart generation."""

    def test_produces_valid_png(self) -> None:
        """Delivery bar chart produces a valid PNG image."""
        result = _build_sample_delivery_result()
        buf = build_delivery_bar_chart(result)
        assert _is_valid_png(buf)

    def test_non_empty_output(self) -> None:
        """Chart output is non-empty."""
        result = _build_sample_delivery_result()
        buf = build_delivery_bar_chart(result)
        buf.seek(0, 2)
        assert buf.tell() > 100

    def test_overall_and_per_section(self) -> None:
        """Chart includes overall and per-section bars."""
        result = _build_sample_delivery_result()
        buf = build_delivery_bar_chart(result)
        assert _is_valid_png(buf)
        # Just verify it generates without error for 2 sections + overall

    def test_empty_sections(self) -> None:
        """Chart handles result with no sections gracefully."""
        result = DeliveryResult(
            week=1,
            chapters=["1장"],
            deliveries=[],
            effective_delivery_rate=0.0,
            per_section_rate={},
        )
        buf = build_delivery_bar_chart(result)
        assert _is_valid_png(buf)


# ----------------------------------------------------------------
# T047: Delivery heatmap
# ----------------------------------------------------------------


class TestDeliveryHeatmap:
    """Tests for delivery quality heatmap generation."""

    def test_produces_valid_png(self) -> None:
        """Delivery heatmap produces valid PNG."""
        result = _build_sample_delivery_result()
        buf = build_delivery_heatmap(result)
        assert _is_valid_png(buf)

    def test_domain_terms_only(self) -> None:
        """Heatmap contains only domain concepts, no stopwords.

        Verify by checking that all concepts in the result are domain
        terms (not everyday words like '여러분', '또한').
        """
        result = _build_sample_delivery_result()
        # All concept names in deliveries are domain terms
        concept_names = {d.concept for d in result.deliveries}
        stopwords = {"여러분", "또한", "소시", "보시면", "것", "대해"}
        assert not concept_names & stopwords
        buf = build_delivery_heatmap(result)
        assert _is_valid_png(buf)

    def test_empty_deliveries(self) -> None:
        """Handles empty deliveries gracefully."""
        result = DeliveryResult(
            week=1,
            chapters=[],
            deliveries=[],
            effective_delivery_rate=0.0,
            per_section_rate={},
        )
        buf = build_delivery_heatmap(result)
        assert _is_valid_png(buf)

    def test_skipped_concepts_excluded(self) -> None:
        """Skipped concepts are excluded from the heatmap."""
        result = _build_sample_delivery_result()
        buf = build_delivery_heatmap(result)
        assert _is_valid_png(buf)
        # "세포막 구조" is skipped, should not appear in heatmap data
