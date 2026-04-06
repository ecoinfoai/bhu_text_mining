"""Tests for report_charts.py — chart generation for student PDF reports.

RED phase: these tests are written *before* the module exists and will fail
until ``src/forma/report_charts.py`` is implemented.

Covers task item T011 (US1 chart tests).

Charts are rendered using the matplotlib Agg backend so no display is needed.
FontProperties is mocked to avoid requiring an actual Korean font file in CI.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# PNG header constant
# ---------------------------------------------------------------------------

PNG_HEADER = b"\x89PNG"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_font(tmp_path):
    """Create a fake font file for testing."""
    font_file = tmp_path / "FakeFont.ttf"
    font_file.write_bytes(b"\x00" * 64)
    return str(font_file)


@pytest.fixture()
def chart_gen(mock_font):
    """Create a ReportChartGenerator with mocked FontProperties.

    FontProperties is replaced with a default instance so matplotlib can
    render text without needing a real Korean font file.
    """
    from matplotlib.font_manager import FontProperties

    with patch(
        "forma.report_charts.FontProperties",
        lambda fname: FontProperties(),
    ):
        from forma.report_charts import ReportChartGenerator

        return ReportChartGenerator(font_path=mock_font)


# ---------------------------------------------------------------------------
# TestReportChartGenerator
# ---------------------------------------------------------------------------


class TestReportChartGenerator:
    """T011: Tests for ReportChartGenerator chart methods."""

    # -- score_boxplot -------------------------------------------------------

    def test_score_boxplot_returns_bytesio(self, chart_gen):
        """score_boxplot returns a BytesIO containing valid PNG data."""
        result = chart_gen.score_boxplot(
            scores=[0.2, 0.4, 0.6, 0.8],
            student_score=0.5,
        )
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_score_boxplot_plt_close_called(self, mock_font):
        """plt.close is called after score_boxplot generates the chart."""
        from matplotlib.font_manager import FontProperties

        with patch(
            "forma.report_charts.FontProperties",
            lambda fname: FontProperties(),
        ):
            from forma.report_charts import ReportChartGenerator

            gen = ReportChartGenerator(font_path=mock_font)

        with patch("forma.report_charts.plt") as mock_plt, patch("forma.chart_utils.plt") as mock_chart_plt:
            # plt.subplots must return a (fig, ax) tuple for the code to work
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            # Configure savefig to write PNG bytes so the BytesIO is valid
            def fake_savefig(buf, **kwargs):
                buf.write(PNG_HEADER + b"\x00" * 10)

            mock_fig.savefig.side_effect = fake_savefig

            gen.score_boxplot(
                scores=[0.2, 0.4, 0.6, 0.8],
                student_score=0.5,
            )
            mock_chart_plt.close.assert_called_once_with(mock_fig)

    def test_score_boxplot_zero_variance(self, chart_gen):
        """score_boxplot handles zero-variance scores without crashing."""
        result = chart_gen.score_boxplot(
            scores=[0.5, 0.5, 0.5, 0.5],
            student_score=0.5,
        )
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_empty_scores_boxplot(self, chart_gen):
        """score_boxplot works with a single score value."""
        result = chart_gen.score_boxplot(
            scores=[0.5],
            student_score=0.5,
        )
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    # -- component_comparison ------------------------------------------------

    def test_component_comparison_returns_bytesio(self, chart_gen):
        """component_comparison returns a BytesIO containing valid PNG data."""
        distributions = {
            "concept_coverage": [0.2, 0.5, 0.8],
            "llm_rubric": [0.3, 0.6, 0.9],
        }
        student_scores = {
            "concept_coverage": 0.4,
            "llm_rubric": 0.5,
        }
        result = chart_gen.component_comparison(
            distributions=distributions,
            student_scores=student_scores,
            question_sn=1,
        )
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    # -- concept_coverage_bar ------------------------------------------------

    def test_concept_coverage_bar_returns_bytesio(self, chart_gen):
        """concept_coverage_bar returns a BytesIO containing valid PNG data."""
        concepts = [
            SimpleNamespace(
                concept="\ud56d\uc0c1\uc131",
                similarity=0.47,
                threshold=0.39,
                is_present=True,
            ),
            SimpleNamespace(
                concept="\uc74c\uc131\ub418\uba39\uc784",
                similarity=0.20,
                threshold=0.35,
                is_present=False,
            ),
        ]
        result = chart_gen.concept_coverage_bar(concepts=concepts)
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_concept_coverage_bar_empty_concepts(self, chart_gen):
        """concept_coverage_bar handles an empty concept list without crashing."""
        result = chart_gen.concept_coverage_bar(concepts=[])
        # Should either return a valid BytesIO or at minimum not raise
        assert isinstance(result, io.BytesIO)

    # -- understanding_badge -------------------------------------------------

    def test_understanding_badge_returns_bytesio(self, chart_gen):
        """understanding_badge returns a BytesIO containing valid PNG data."""
        result = chart_gen.understanding_badge(level="Advanced", score=0.85)
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    @pytest.mark.parametrize(
        "level",
        ["Advanced", "Proficient", "Developing", "Beginning"],
    )
    def test_understanding_badge_all_levels(self, chart_gen, level):
        """understanding_badge produces valid PNG for every defined level."""
        result = chart_gen.understanding_badge(level=level, score=0.5)
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    # -- radar_chart ---------------------------------------------------------

    def test_radar_chart_returns_bytesio(self, chart_gen):
        """T034: radar_chart returns a BytesIO containing valid PNG data."""
        result = chart_gen.radar_chart(
            student_axes=[0.5, 0.7, 0.3],
            class_avg_axes=[0.6, 0.6, 0.6],
            labels=["개념 커버리지", "LLM 루브릭", "Rasch 능력치"],
        )
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_radar_chart_three_plus_axes(self, chart_gen):
        """T034: radar_chart supports 3+ axes."""
        result = chart_gen.radar_chart(
            student_axes=[0.4, 0.6, 0.8, 0.5],
            class_avg_axes=[0.5, 0.5, 0.5, 0.5],
            labels=["A", "B", "C", "D"],
        )
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert data[:4] == PNG_HEADER

    def test_radar_chart_all_zero_values(self, chart_gen):
        """T034: radar_chart handles all-zero values without crash."""
        result = chart_gen.radar_chart(
            student_axes=[0.0, 0.0, 0.0],
            class_avg_axes=[0.0, 0.0, 0.0],
            labels=["X", "Y", "Z"],
        )
        assert isinstance(result, io.BytesIO)

    def test_radar_chart_too_few_axes(self, chart_gen):
        """T034: radar_chart handles fewer than 3 axes gracefully."""
        result = chart_gen.radar_chart(
            student_axes=[0.5, 0.3],
            class_avg_axes=[0.6, 0.4],
            labels=["A", "B"],
        )
        # Should still return BytesIO (with fallback message)
        assert isinstance(result, io.BytesIO)


# ---------------------------------------------------------------------------
# Phase 4: US2 — T017: Trajectory bar chart tests
# ---------------------------------------------------------------------------


class TestTrajectoryBarChart:
    """T017: build_trajectory_bar_chart — PNG output, current week highlighted."""

    def test_returns_png_bytesio(self, chart_gen):
        """Should return a BytesIO with valid PNG data."""
        result = chart_gen.build_trajectory_bar_chart(
            weekly_scores={1: 0.50, 2: 0.65, 3: 0.80},
            current_week=3,
        )
        assert isinstance(result, io.BytesIO)
        result.seek(0)
        assert result.read(4) == PNG_HEADER

    def test_single_week(self, chart_gen):
        """Single-week data should produce a valid chart with one bar."""
        result = chart_gen.build_trajectory_bar_chart(
            weekly_scores={1: 0.60},
            current_week=1,
        )
        assert isinstance(result, io.BytesIO)
        result.seek(0)
        assert result.read(4) == PNG_HEADER

    def test_non_contiguous_weeks(self, chart_gen):
        """Non-contiguous weeks (1, 3, 5) should render correctly."""
        result = chart_gen.build_trajectory_bar_chart(
            weekly_scores={1: 0.40, 3: 0.60, 5: 0.80},
            current_week=5,
        )
        assert isinstance(result, io.BytesIO)
        result.seek(0)
        assert result.read(4) == PNG_HEADER

    def test_current_week_not_in_data(self, chart_gen):
        """If current_week is not in the data, should still render without error."""
        result = chart_gen.build_trajectory_bar_chart(
            weekly_scores={1: 0.50, 2: 0.65},
            current_week=3,
        )
        assert isinstance(result, io.BytesIO)
