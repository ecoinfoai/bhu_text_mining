"""Tests for section_comparison_charts.py — Cross-Section Comparison Charts (US4).

T050: Tests for box plot, concept mastery heatmap, weekly interaction line chart.
Covers: 2-section and 4-section cases, empty/None weekly interaction data.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.font_manager import FontProperties as _RealFontProperties

PNG_HEADER = b"\x89PNG"


# ---------------------------------------------------------------------------
# Font mocking — use default FontProperties (no fname) so matplotlib works
# ---------------------------------------------------------------------------


def _mock_font_properties_factory(fname=None):
    """Return a real FontProperties without loading a .ttf file."""
    return _RealFontProperties()


@pytest.fixture(autouse=True)
def _patch_chart_fonts():
    """Auto-mock font discovery and FontProperties for all chart tests."""
    p1 = patch(
        "forma.section_comparison_charts.find_korean_font",
        return_value="/fake/NanumGothic.ttf",
    )
    p2 = patch(
        "forma.section_comparison_charts.FontProperties",
        side_effect=_mock_font_properties_factory,
    )
    with p1, p2:
        yield


# ---------------------------------------------------------------------------
# build_section_box_plot()
# ---------------------------------------------------------------------------


class TestBuildSectionBoxPlot:
    """Tests for build_section_box_plot chart generation."""

    def test_two_sections(self):
        from forma.section_comparison_charts import build_section_box_plot

        section_scores = {
            "A": [0.5, 0.6, 0.7, 0.8, 0.9],
            "B": [0.4, 0.5, 0.55, 0.6, 0.7],
        }
        buf = build_section_box_plot(section_scores)
        assert isinstance(buf, BytesIO)
        buf.seek(0)
        header = buf.read(4)
        assert header[:4] == PNG_HEADER

    def test_four_sections(self):
        from forma.section_comparison_charts import build_section_box_plot

        rng = np.random.default_rng(42)
        section_scores = {
            "A": list(rng.normal(0.7, 0.1, 30)),
            "B": list(rng.normal(0.6, 0.12, 25)),
            "C": list(rng.normal(0.65, 0.11, 20)),
            "D": list(rng.normal(0.55, 0.13, 35)),
        }
        buf = build_section_box_plot(section_scores)
        assert isinstance(buf, BytesIO)

    def test_single_section(self):
        from forma.section_comparison_charts import build_section_box_plot

        section_scores = {"A": [0.5, 0.6, 0.7]}
        buf = build_section_box_plot(section_scores)
        assert isinstance(buf, BytesIO)


# ---------------------------------------------------------------------------
# build_concept_mastery_heatmap()
# ---------------------------------------------------------------------------


class TestBuildConceptMasteryHeatmap:
    """Tests for build_concept_mastery_heatmap chart generation."""

    def test_basic_heatmap(self):
        from forma.section_comparison_charts import build_concept_mastery_heatmap

        concept_mastery = {
            "A": {"cell_membrane": 0.8, "mitochondria": 0.6, "nucleus": 0.9},
            "B": {"cell_membrane": 0.7, "mitochondria": 0.5, "nucleus": 0.75},
        }
        buf = build_concept_mastery_heatmap(concept_mastery)
        assert isinstance(buf, BytesIO)
        buf.seek(0)
        header = buf.read(4)
        assert header[:4] == PNG_HEADER

    def test_four_sections(self):
        from forma.section_comparison_charts import build_concept_mastery_heatmap

        concept_mastery = {
            "A": {"c1": 0.8, "c2": 0.6},
            "B": {"c1": 0.7, "c2": 0.5},
            "C": {"c1": 0.9, "c2": 0.4},
            "D": {"c1": 0.6, "c2": 0.7},
        }
        buf = build_concept_mastery_heatmap(concept_mastery)
        assert isinstance(buf, BytesIO)

    def test_empty_returns_none(self):
        from forma.section_comparison_charts import build_concept_mastery_heatmap

        result = build_concept_mastery_heatmap({})
        assert result is None

    def test_many_concepts_truncated(self):
        """Heatmap with many concepts should still render (may truncate)."""
        from forma.section_comparison_charts import build_concept_mastery_heatmap

        rng = np.random.default_rng(42)
        concepts = {f"concept_{i}": float(rng.random()) for i in range(30)}
        concept_mastery = {
            "A": concepts,
            "B": {k: float(rng.random()) for k in concepts},
        }
        buf = build_concept_mastery_heatmap(concept_mastery)
        assert isinstance(buf, BytesIO)


# ---------------------------------------------------------------------------
# build_weekly_interaction_chart()
# ---------------------------------------------------------------------------


class TestBuildWeeklyInteractionChart:
    """Tests for build_weekly_interaction_chart."""

    def test_basic_chart(self):
        from forma.section_comparison_charts import build_weekly_interaction_chart

        weekly_data = {
            "A": {1: 0.65, 2: 0.7, 3: 0.75},
            "B": {1: 0.55, 2: 0.6, 3: 0.65},
        }
        buf = build_weekly_interaction_chart(weekly_data)
        assert isinstance(buf, BytesIO)
        buf.seek(0)
        header = buf.read(4)
        assert header[:4] == PNG_HEADER

    def test_four_sections(self):
        from forma.section_comparison_charts import build_weekly_interaction_chart

        weekly_data = {
            "A": {1: 0.7, 2: 0.72, 3: 0.75},
            "B": {1: 0.6, 2: 0.65, 3: 0.68},
            "C": {1: 0.55, 2: 0.6, 3: 0.62},
            "D": {1: 0.5, 2: 0.55, 3: 0.58},
        }
        buf = build_weekly_interaction_chart(weekly_data)
        assert isinstance(buf, BytesIO)

    def test_none_returns_none(self):
        from forma.section_comparison_charts import build_weekly_interaction_chart

        result = build_weekly_interaction_chart(None)
        assert result is None

    def test_empty_returns_none(self):
        from forma.section_comparison_charts import build_weekly_interaction_chart

        result = build_weekly_interaction_chart({})
        assert result is None
