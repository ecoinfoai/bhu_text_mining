"""Tests for longitudinal_report_charts.py — chart generation for US3 longitudinal report.

RED phase: tests written BEFORE implementation (TDD).

Covers:
  T030 — build_trajectory_line_chart(): risk=red, normal=gray, class_avg=blue, single-week
  T031 — build_class_week_heatmap(): sorted by final week, missing weeks, 100+ cap
  T032 — build_concept_mastery_bar_chart(): sorted by delta desc, empty concept list
"""

from __future__ import annotations

import io
from unittest.mock import patch

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
    p1 = patch("forma.longitudinal_report_charts.find_korean_font",
               return_value="/fake/NanumGothic.ttf")
    p2 = patch("forma.longitudinal_report_charts.FontProperties",
               side_effect=_mock_font_properties_factory)
    p1.start()
    p2.start()
    yield
    p2.stop()
    p1.stop()


# ---------------------------------------------------------------------------
# Helpers — build mock LongitudinalSummaryData
# ---------------------------------------------------------------------------


def _make_trajectory(student_id, weekly_scores, overall_trend=0.0,
                     is_persistent_risk=False, risk_weeks=None):
    """Build a StudentTrajectory for testing."""
    from forma.longitudinal_report_data import StudentTrajectory
    return StudentTrajectory(
        student_id=student_id,
        weekly_scores=weekly_scores,
        overall_trend=overall_trend,
        is_persistent_risk=is_persistent_risk,
        risk_weeks=risk_weeks or [],
    )


def _make_concept_change(concept, start, end):
    """Build a ConceptMasteryChange for testing."""
    from forma.longitudinal_report_data import ConceptMasteryChange
    return ConceptMasteryChange(
        concept=concept,
        week_start_ratio=start,
        week_end_ratio=end,
        delta=end - start,
    )


def _make_summary(
    trajectories=None,
    class_weekly_averages=None,
    concept_changes=None,
    period_weeks=None,
    persistent_risk=None,
    total_students=None,
):
    """Build a LongitudinalSummaryData for testing."""
    from forma.longitudinal_report_data import LongitudinalSummaryData

    if trajectories is None:
        trajectories = [
            _make_trajectory("S001", {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6}),
            _make_trajectory("S002", {1: 0.7, 2: 0.72, 3: 0.75, 4: 0.80}),
            _make_trajectory("S003", {1: 0.2, 2: 0.25, 3: 0.3, 4: 0.35},
                             is_persistent_risk=True, risk_weeks=[1, 2, 3, 4]),
        ]
    if class_weekly_averages is None:
        class_weekly_averages = {1: 0.4, 2: 0.46, 3: 0.52, 4: 0.58}
    if concept_changes is None:
        concept_changes = [
            _make_concept_change("항상성", 0.6, 0.8),
            _make_concept_change("삼투", 0.4, 0.5),
        ]
    if period_weeks is None:
        period_weeks = [1, 2, 3, 4]
    if persistent_risk is None:
        persistent_risk = ["S003"]

    return LongitudinalSummaryData(
        class_name="1A",
        period_weeks=period_weeks,
        student_trajectories=trajectories,
        class_weekly_averages=class_weekly_averages,
        persistent_risk_students=persistent_risk,
        concept_mastery_changes=concept_changes,
        total_students=total_students or len(trajectories),
    )


# ---------------------------------------------------------------------------
# T030: build_trajectory_line_chart() tests
# ---------------------------------------------------------------------------


class TestTrajectoryLineChart:
    """Test build_trajectory_line_chart()."""

    def test_returns_png(self):
        """Chart returns valid PNG bytes in BytesIO."""
        from forma.longitudinal_report_charts import build_trajectory_line_chart
        summary = _make_summary()
        result = build_trajectory_line_chart(summary)
        assert isinstance(result, io.BytesIO)
        data = result.read()
        assert data[:4] == PNG_HEADER
        assert len(data) > 100

    def test_single_week(self):
        """Single week data should not crash — produces valid PNG."""
        from forma.longitudinal_report_charts import build_trajectory_line_chart

        summary = _make_summary(
            trajectories=[
                _make_trajectory("S001", {1: 0.5}),
                _make_trajectory("S002", {1: 0.7}),
            ],
            class_weekly_averages={1: 0.6},
            period_weeks=[1],
        )
        result = build_trajectory_line_chart(summary)
        assert isinstance(result, io.BytesIO)
        assert result.read()[:4] == PNG_HEADER

    def test_empty_trajectories(self):
        """No student trajectories — should not crash."""
        from forma.longitudinal_report_charts import build_trajectory_line_chart

        summary = _make_summary(
            trajectories=[],
            class_weekly_averages={},
            period_weeks=[1, 2],
            total_students=0,
        )
        result = build_trajectory_line_chart(summary)
        assert isinstance(result, io.BytesIO)
        assert result.read()[:4] == PNG_HEADER

    def test_risk_and_normal_students(self):
        """Both risk and normal students — chart should render both."""
        from forma.longitudinal_report_charts import build_trajectory_line_chart

        summary = _make_summary(
            trajectories=[
                _make_trajectory("S001", {1: 0.5, 2: 0.6}, is_persistent_risk=False),
                _make_trajectory("S003", {1: 0.2, 2: 0.25}, is_persistent_risk=True,
                                 risk_weeks=[1, 2]),
            ],
            class_weekly_averages={1: 0.35, 2: 0.425},
            period_weeks=[1, 2],
        )
        result = build_trajectory_line_chart(summary)
        data = result.read()
        assert len(data) > 100

    def test_non_contiguous_weeks(self):
        """Non-contiguous weeks [1, 3, 5] should render correctly on X-axis."""
        from forma.longitudinal_report_charts import build_trajectory_line_chart

        summary = _make_summary(
            trajectories=[
                _make_trajectory("S001", {1: 0.3, 3: 0.5, 5: 0.7}),
            ],
            class_weekly_averages={1: 0.3, 3: 0.5, 5: 0.7},
            period_weeks=[1, 3, 5],
        )
        result = build_trajectory_line_chart(summary)
        assert result.read()[:4] == PNG_HEADER


# ---------------------------------------------------------------------------
# T031: build_class_week_heatmap() tests
# ---------------------------------------------------------------------------


class TestClassWeekHeatmap:
    """Test build_class_week_heatmap()."""

    def test_returns_png(self):
        """Heatmap returns valid PNG bytes."""
        from forma.longitudinal_report_charts import build_class_week_heatmap
        summary = _make_summary()
        result = build_class_week_heatmap(summary)
        assert isinstance(result, io.BytesIO)
        assert result.read()[:4] == PNG_HEADER

    def test_single_week(self):
        """Single week heatmap — single column, no crash."""
        from forma.longitudinal_report_charts import build_class_week_heatmap

        summary = _make_summary(
            trajectories=[
                _make_trajectory("S001", {1: 0.5}),
                _make_trajectory("S002", {1: 0.7}),
            ],
            class_weekly_averages={1: 0.6},
            period_weeks=[1],
        )
        result = build_class_week_heatmap(summary)
        assert result.read()[:4] == PNG_HEADER

    def test_missing_weeks_handled(self):
        """Student missing some weeks — those cells should be handled (NaN/gray)."""
        from forma.longitudinal_report_charts import build_class_week_heatmap

        summary = _make_summary(
            trajectories=[
                _make_trajectory("S001", {1: 0.5, 2: 0.6, 3: 0.7, 4: 0.8}),
                _make_trajectory("S002", {1: 0.3, 4: 0.5}),
            ],
            class_weekly_averages={1: 0.4, 2: 0.6, 3: 0.7, 4: 0.65},
            period_weeks=[1, 2, 3, 4],
        )
        result = build_class_week_heatmap(summary)
        assert result.read()[:4] == PNG_HEADER

    def test_100_plus_students_capped(self):
        """100+ students -> show top 25 + bottom 25 with gap indicator."""
        from forma.longitudinal_report_charts import build_class_week_heatmap

        trajectories = []
        for i in range(120):
            score = 0.01 + (i / 120) * 0.99
            trajectories.append(
                _make_trajectory(f"S{i:03d}", {1: score, 2: score + 0.01})
            )

        summary = _make_summary(
            trajectories=trajectories,
            class_weekly_averages={1: 0.5, 2: 0.51},
            period_weeks=[1, 2],
            total_students=120,
        )
        result = build_class_week_heatmap(summary)
        data = result.read()
        assert data[:4] == PNG_HEADER
        assert len(data) > 100

    def test_empty_trajectories(self):
        """No students — should produce placeholder chart."""
        from forma.longitudinal_report_charts import build_class_week_heatmap

        summary = _make_summary(
            trajectories=[],
            class_weekly_averages={},
            period_weeks=[1, 2],
            total_students=0,
        )
        result = build_class_week_heatmap(summary)
        assert result.read()[:4] == PNG_HEADER

    def test_all_identical_scores(self):
        """All students have identical scores — heatmap color scale handles it."""
        from forma.longitudinal_report_charts import build_class_week_heatmap

        summary = _make_summary(
            trajectories=[
                _make_trajectory("S001", {1: 0.5, 2: 0.5}),
                _make_trajectory("S002", {1: 0.5, 2: 0.5}),
                _make_trajectory("S003", {1: 0.5, 2: 0.5}),
            ],
            class_weekly_averages={1: 0.5, 2: 0.5},
            period_weeks=[1, 2],
        )
        result = build_class_week_heatmap(summary)
        assert result.read()[:4] == PNG_HEADER


# ---------------------------------------------------------------------------
# T032: build_concept_mastery_bar_chart() tests
# ---------------------------------------------------------------------------


class TestConceptMasteryBarChart:
    """Test build_concept_mastery_bar_chart()."""

    def test_returns_png(self):
        """Bar chart returns valid PNG bytes."""
        from forma.longitudinal_report_charts import build_concept_mastery_bar_chart
        summary = _make_summary()
        result = build_concept_mastery_bar_chart(summary)
        assert isinstance(result, io.BytesIO)
        assert result.read()[:4] == PNG_HEADER

    def test_empty_concept_list(self):
        """Empty concept list — should produce placeholder chart, not crash."""
        from forma.longitudinal_report_charts import build_concept_mastery_bar_chart

        summary = _make_summary(concept_changes=[])
        result = build_concept_mastery_bar_chart(summary)
        assert isinstance(result, io.BytesIO)
        assert result.read()[:4] == PNG_HEADER

    def test_single_concept(self):
        """Single concept — should render a single bar."""
        from forma.longitudinal_report_charts import build_concept_mastery_bar_chart

        summary = _make_summary(
            concept_changes=[_make_concept_change("항상성", 0.4, 0.8)],
        )
        result = build_concept_mastery_bar_chart(summary)
        data = result.read()
        assert data[:4] == PNG_HEADER
        assert len(data) > 100

    def test_mixed_positive_negative_deltas(self):
        """Mix of positive and negative deltas — renders both."""
        from forma.longitudinal_report_charts import build_concept_mastery_bar_chart

        summary = _make_summary(
            concept_changes=[
                _make_concept_change("항상성", 0.4, 0.8),
                _make_concept_change("삼투", 0.7, 0.5),
                _make_concept_change("확산", 0.5, 0.5),
            ],
        )
        result = build_concept_mastery_bar_chart(summary)
        data = result.read()
        assert data[:4] == PNG_HEADER
        assert len(data) > 100

    def test_many_concepts(self):
        """10+ concepts — chart should handle vertical sizing."""
        from forma.longitudinal_report_charts import build_concept_mastery_bar_chart

        changes = [
            _make_concept_change(f"concept_{i}", 0.3 + i * 0.02, 0.5 + i * 0.03)
            for i in range(15)
        ]
        summary = _make_summary(concept_changes=changes)
        result = build_concept_mastery_bar_chart(summary)
        assert result.read()[:4] == PNG_HEADER


# ---------------------------------------------------------------------------
# T042: Edge case tests — all-identical scores, 100+ truncation verification
# ---------------------------------------------------------------------------


class TestHeatmapEdgeCases:
    """T042 edge case tests for heatmap chart."""

    def test_all_identical_scores_no_division_error(self):
        """All students score 0.5 every week — no division by zero in color normalization."""
        from forma.longitudinal_report_charts import build_class_week_heatmap

        trajectories = [
            _make_trajectory(f"S{i:03d}", {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5})
            for i in range(10)
        ]
        summary = _make_summary(
            trajectories=trajectories,
            class_weekly_averages={1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
            period_weeks=[1, 2, 3, 4],
            total_students=10,
        )
        result = build_class_week_heatmap(summary)
        data = result.read()
        assert data[:4] == PNG_HEADER
        assert len(data) > 100  # meaningful chart, not empty

    def test_120_students_truncation_top25_bottom25(self):
        """120 students — top 25 + bottom 25 shown, middle omitted."""
        from forma.longitudinal_report_charts import build_class_week_heatmap

        trajectories = []
        for i in range(120):
            # Spread scores from 0.01 to ~1.0 for clear ordering
            final_score = 0.01 + (i / 119) * 0.98
            trajectories.append(
                _make_trajectory(
                    f"S{i:03d}",
                    {1: final_score * 0.8, 2: final_score * 0.9, 3: final_score},
                )
            )

        summary = _make_summary(
            trajectories=trajectories,
            class_weekly_averages={1: 0.4, 2: 0.45, 3: 0.5},
            period_weeks=[1, 2, 3],
            total_students=120,
        )
        result = build_class_week_heatmap(summary)
        data = result.read()
        assert data[:4] == PNG_HEADER
        assert len(data) > 500  # substantial chart with 50 student rows

    def test_exactly_100_students_no_truncation(self):
        """Exactly 100 students — no truncation needed (boundary case)."""
        from forma.longitudinal_report_charts import build_class_week_heatmap

        trajectories = [
            _make_trajectory(f"S{i:03d}", {1: i / 100, 2: (i + 1) / 100})
            for i in range(100)
        ]
        summary = _make_summary(
            trajectories=trajectories,
            class_weekly_averages={1: 0.5, 2: 0.51},
            period_weeks=[1, 2],
            total_students=100,
        )
        result = build_class_week_heatmap(summary)
        data = result.read()
        assert data[:4] == PNG_HEADER

    def test_101_students_triggers_truncation(self):
        """101 students — triggers truncation (boundary case)."""
        from forma.longitudinal_report_charts import build_class_week_heatmap

        trajectories = [
            _make_trajectory(f"S{i:03d}", {1: i / 101, 2: (i + 1) / 101})
            for i in range(101)
        ]
        summary = _make_summary(
            trajectories=trajectories,
            class_weekly_averages={1: 0.5, 2: 0.51},
            period_weeks=[1, 2],
            total_students=101,
        )
        result = build_class_week_heatmap(summary)
        data = result.read()
        assert data[:4] == PNG_HEADER
        assert len(data) > 100
