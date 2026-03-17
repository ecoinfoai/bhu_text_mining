"""Tests for domain coverage chart generation.

T029: Coverage bar chart
T030: Emphasis bias scatter and section variance heatmap
"""

from __future__ import annotations

import io
import statistics

import pytest

from forma.domain_concept_extractor import TextbookConcept
from forma.domain_coverage_analyzer import (
    ClassifiedConcept,
    ConceptEmphasis,
    ConceptState,
    CoverageResult,
    ExtraConcept,
    TeachingScope,
    build_coverage_result,
    classify_concepts,
)
from forma.domain_coverage_charts import (
    build_coverage_bar_chart,
    build_emphasis_bias_scatter,
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
