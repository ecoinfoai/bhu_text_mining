"""Tests for warning_report_charts.py — dashboard charts for early warning report.

T039 [US3]: Risk type distribution bar chart, deficit concepts horizontal bar chart.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from matplotlib.font_manager import FontProperties as _RealFontProperties

PNG_HEADER = b"\x89PNG"


def _mock_font_properties_factory(fname=None):
    """Return a real FontProperties without loading a .ttf file."""
    return _RealFontProperties()


@pytest.fixture()
def _mock_font():
    """Mock Korean font discovery so tests don't need real font files."""
    with (
        patch("forma.warning_report_charts.find_korean_font", return_value="/fake/font.ttf"),
        patch(
            "forma.warning_report_charts.FontProperties",
            side_effect=_mock_font_properties_factory,
        ),
    ):
        yield


class TestRiskTypeDistributionChart:
    """Tests for build_risk_type_distribution_chart()."""

    @pytest.mark.usefixtures("_mock_font")
    def test_returns_bytesio_png(self):
        """Chart returns BytesIO with PNG header."""
        from forma.warning_report_charts import build_risk_type_distribution_chart
        from forma.warning_report_data import RiskType

        counts = {RiskType.SCORE_DECLINE: 5, RiskType.PERSISTENT_LOW: 3}
        buf = build_risk_type_distribution_chart(counts)
        assert isinstance(buf, io.BytesIO)
        assert buf.read(4) == PNG_HEADER

    @pytest.mark.usefixtures("_mock_font")
    def test_all_risk_types(self):
        """Chart handles all 4 risk types."""
        from forma.warning_report_charts import build_risk_type_distribution_chart
        from forma.warning_report_data import RiskType

        counts = {rt: 2 for rt in RiskType}
        buf = build_risk_type_distribution_chart(counts)
        assert buf.read(4) == PNG_HEADER

    @pytest.mark.usefixtures("_mock_font")
    def test_empty_counts(self):
        """Chart handles empty counts without error."""
        from forma.warning_report_charts import build_risk_type_distribution_chart

        buf = build_risk_type_distribution_chart({})
        assert isinstance(buf, io.BytesIO)

    @pytest.mark.usefixtures("_mock_font")
    def test_single_risk_type(self):
        """Chart handles single risk type."""
        from forma.warning_report_charts import build_risk_type_distribution_chart
        from forma.warning_report_data import RiskType

        counts = {RiskType.CONCEPT_DEFICIT: 10}
        buf = build_risk_type_distribution_chart(counts)
        assert buf.read(4) == PNG_HEADER


class TestDeficitConceptsChart:
    """Tests for build_deficit_concepts_chart()."""

    @pytest.mark.usefixtures("_mock_font")
    def test_returns_bytesio_png(self):
        """Chart returns BytesIO with PNG header."""
        from forma.warning_report_charts import build_deficit_concepts_chart

        concepts = {"세포": 5, "조직": 3, "기관": 7}
        buf = build_deficit_concepts_chart(concepts)
        assert isinstance(buf, io.BytesIO)
        assert buf.read(4) == PNG_HEADER

    @pytest.mark.usefixtures("_mock_font")
    def test_top_10_limit(self):
        """Chart limits to top 10 concepts."""
        from forma.warning_report_charts import build_deficit_concepts_chart

        concepts = {f"concept_{i}": 15 - i for i in range(15)}
        buf = build_deficit_concepts_chart(concepts)
        assert isinstance(buf, io.BytesIO)
        assert buf.read(4) == PNG_HEADER

    @pytest.mark.usefixtures("_mock_font")
    def test_empty_concepts(self):
        """Chart handles empty concept dict without error."""
        from forma.warning_report_charts import build_deficit_concepts_chart

        buf = build_deficit_concepts_chart({})
        assert isinstance(buf, io.BytesIO)

    @pytest.mark.usefixtures("_mock_font")
    def test_korean_concept_names(self):
        """Chart handles Korean concept names."""
        from forma.warning_report_charts import build_deficit_concepts_chart

        concepts = {"세포막": 4, "핵산": 2, "효소": 6, "단백질": 3}
        buf = build_deficit_concepts_chart(concepts)
        assert buf.read(4) == PNG_HEADER
