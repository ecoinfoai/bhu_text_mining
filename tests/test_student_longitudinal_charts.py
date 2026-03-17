"""Tests for student_longitudinal_charts.py — per-student chart generation.

TDD RED phase: tests written before implementation.
T012-T015: coverage trend, component breakdown, cohort position, warning table.
"""

from __future__ import annotations

import io
import struct

from forma.student_longitudinal_data import (
    AlertLevel,
    CohortDistribution,
    CohortWeekStats,
    StudentLongitudinalData,
    WarningSignal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_student_data(
    weeks: list[int] | None = None,
    student_id: str = "s001",
    student_name: str = "홍길동",
    class_name: str = "A",
) -> StudentLongitudinalData:
    """Build minimal StudentLongitudinalData for chart tests."""
    if weeks is None:
        weeks = [1, 2, 3]
    scores_by_week = {}
    for w in weeks:
        scores_by_week[w] = {
            1: {
                "concept_coverage": 0.5 + 0.05 * w,
                "llm_rubric": 0.4 + 0.05 * w,
                "ensemble_score": 0.45 + 0.05 * w,
                "rasch_ability": -0.2 + 0.1 * w,
            },
            2: {
                "concept_coverage": 0.4 + 0.05 * w,
                "llm_rubric": 0.35 + 0.05 * w,
                "ensemble_score": 0.40 + 0.05 * w,
                "rasch_ability": -0.3 + 0.1 * w,
            },
        }
    percentiles = {w: 30.0 + 5.0 * w for w in weeks}
    return StudentLongitudinalData(
        student_id=student_id,
        student_name=student_name,
        class_name=class_name,
        weeks=weeks,
        scores_by_week=scores_by_week,
        trend_slope=0.05,
        trend_direction="상승",
        percentiles_by_week=percentiles,
    )


def _make_cohort(weeks: list[int] | None = None) -> CohortDistribution:
    """Build a CohortDistribution with plausible data for 10 students."""
    if weeks is None:
        weeks = [1, 2, 3]
    cohort = CohortDistribution()
    for w in weeks:
        # 10 student scores
        scores = [0.3 + 0.05 * i + 0.02 * w for i in range(10)]
        cohort.weekly_scores[w] = scores
        cohort.weekly_q_scores[w] = {
            1: [0.4 + 0.05 * i + 0.02 * w for i in range(10)],
            2: [0.3 + 0.05 * i + 0.02 * w for i in range(10)],
        }
        import numpy as np

        arr = np.array(scores)
        cohort.weekly_stats[w] = CohortWeekStats(
            week=w,
            median=float(np.median(arr)),
            q1=float(np.percentile(arr, 25)),
            q3=float(np.percentile(arr, 75)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr, ddof=0)),
            n=len(scores),
        )
    return cohort


def _is_valid_png(buf: io.BytesIO) -> bool:
    """Check if buffer starts with PNG magic bytes."""
    buf.seek(0)
    header = buf.read(8)
    return header == b"\x89PNG\r\n\x1a\n"


def _png_dimensions(buf: io.BytesIO) -> tuple[int, int]:
    """Read width and height from a PNG IHDR chunk."""
    buf.seek(16)
    width, height = struct.unpack(">II", buf.read(8))
    return width, height


# ---------------------------------------------------------------------------
# T012: build_coverage_trend_chart
# ---------------------------------------------------------------------------


class TestBuildCoverageTrendChart:
    """build_coverage_trend_chart returns a valid PNG BytesIO."""

    def test_returns_valid_png_q1(self):
        from forma.student_longitudinal_charts import build_coverage_trend_chart

        student = _make_student_data()
        cohort = _make_cohort()
        buf = build_coverage_trend_chart(student, cohort, qsn=1, font_path=None, dpi=72)
        assert isinstance(buf, io.BytesIO)
        assert _is_valid_png(buf)

    def test_returns_valid_png_q2(self):
        from forma.student_longitudinal_charts import build_coverage_trend_chart

        student = _make_student_data()
        cohort = _make_cohort()
        buf = build_coverage_trend_chart(student, cohort, qsn=2, font_path=None, dpi=72)
        assert isinstance(buf, io.BytesIO)
        assert _is_valid_png(buf)

    def test_png_has_nonzero_dimensions(self):
        from forma.student_longitudinal_charts import build_coverage_trend_chart

        student = _make_student_data()
        cohort = _make_cohort()
        buf = build_coverage_trend_chart(student, cohort, qsn=1, font_path=None, dpi=72)
        w, h = _png_dimensions(buf)
        assert w > 0
        assert h > 0

    def test_single_week_no_crash(self):
        from forma.student_longitudinal_charts import build_coverage_trend_chart

        student = _make_student_data(weeks=[1])
        cohort = _make_cohort(weeks=[1])
        buf = build_coverage_trend_chart(student, cohort, qsn=1, font_path=None, dpi=72)
        assert _is_valid_png(buf)

    def test_missing_q2_data_no_crash(self):
        """If student has no Q2 data for a week, chart still works."""
        from forma.student_longitudinal_charts import build_coverage_trend_chart

        student = _make_student_data(weeks=[1, 2])
        # Remove Q2 from week 2
        del student.scores_by_week[2][2]
        cohort = _make_cohort(weeks=[1, 2])
        buf = build_coverage_trend_chart(student, cohort, qsn=2, font_path=None, dpi=72)
        assert _is_valid_png(buf)

    def test_empty_weeks_no_crash(self):
        from forma.student_longitudinal_charts import build_coverage_trend_chart

        student = _make_student_data(weeks=[])
        cohort = _make_cohort(weeks=[])
        buf = build_coverage_trend_chart(student, cohort, qsn=1, font_path=None, dpi=72)
        assert _is_valid_png(buf)


# ---------------------------------------------------------------------------
# T013: build_component_breakdown_chart
# ---------------------------------------------------------------------------


class TestBuildComponentBreakdownChart:
    """build_component_breakdown_chart returns a valid PNG BytesIO."""

    def test_returns_valid_png(self):
        from forma.student_longitudinal_charts import build_component_breakdown_chart

        student = _make_student_data()
        buf = build_component_breakdown_chart(student, font_path=None, dpi=72)
        assert isinstance(buf, io.BytesIO)
        assert _is_valid_png(buf)

    def test_png_has_nonzero_dimensions(self):
        from forma.student_longitudinal_charts import build_component_breakdown_chart

        student = _make_student_data()
        buf = build_component_breakdown_chart(student, font_path=None, dpi=72)
        w, h = _png_dimensions(buf)
        assert w > 0
        assert h > 0

    def test_single_week_no_crash(self):
        from forma.student_longitudinal_charts import build_component_breakdown_chart

        student = _make_student_data(weeks=[1])
        buf = build_component_breakdown_chart(student, font_path=None, dpi=72)
        assert _is_valid_png(buf)

    def test_empty_weeks_no_crash(self):
        from forma.student_longitudinal_charts import build_component_breakdown_chart

        student = _make_student_data(weeks=[])
        buf = build_component_breakdown_chart(student, font_path=None, dpi=72)
        assert _is_valid_png(buf)


# ---------------------------------------------------------------------------
# T014: build_cohort_position_chart
# ---------------------------------------------------------------------------


class TestBuildCohortPositionChart:
    """build_cohort_position_chart returns a valid PNG BytesIO."""

    def test_returns_valid_png(self):
        from forma.student_longitudinal_charts import build_cohort_position_chart

        student = _make_student_data()
        cohort = _make_cohort()
        buf = build_cohort_position_chart(student, cohort, font_path=None, dpi=72)
        assert isinstance(buf, io.BytesIO)
        assert _is_valid_png(buf)

    def test_png_has_nonzero_dimensions(self):
        from forma.student_longitudinal_charts import build_cohort_position_chart

        student = _make_student_data()
        cohort = _make_cohort()
        buf = build_cohort_position_chart(student, cohort, font_path=None, dpi=72)
        w, h = _png_dimensions(buf)
        assert w > 0
        assert h > 0

    def test_single_week_no_crash(self):
        from forma.student_longitudinal_charts import build_cohort_position_chart

        student = _make_student_data(weeks=[1])
        cohort = _make_cohort(weeks=[1])
        buf = build_cohort_position_chart(student, cohort, font_path=None, dpi=72)
        assert _is_valid_png(buf)

    def test_empty_weeks_no_crash(self):
        from forma.student_longitudinal_charts import build_cohort_position_chart

        student = _make_student_data(weeks=[])
        cohort = _make_cohort(weeks=[])
        buf = build_cohort_position_chart(student, cohort, font_path=None, dpi=72)
        assert _is_valid_png(buf)


# ---------------------------------------------------------------------------
# T015: build_warning_table
# ---------------------------------------------------------------------------


class TestBuildWarningTable:
    """build_warning_table returns a valid PNG BytesIO."""

    def test_returns_valid_png_warning(self):
        from forma.student_longitudinal_charts import build_warning_table

        warnings = [
            WarningSignal(
                name="위험 구간 진입",
                triggered=True,
                severity="critical",
                detail="ensemble_score 0.40 < 0.45",
            ),
            WarningSignal(
                name="하위 백분위",
                triggered=False,
                severity="critical",
                detail="정상 범위",
            ),
            WarningSignal(
                name="저조한 개념 커버리지",
                triggered=True,
                severity="non-critical",
                detail="concept_coverage 0.25 < 0.30",
            ),
        ]
        buf = build_warning_table(warnings, AlertLevel.WARNING, font_path=None, dpi=72)
        assert isinstance(buf, io.BytesIO)
        assert _is_valid_png(buf)

    def test_returns_valid_png_normal(self):
        from forma.student_longitudinal_charts import build_warning_table

        warnings = [
            WarningSignal(
                name="위험 구간 진입",
                triggered=False,
                severity="critical",
                detail="정상 범위",
            ),
        ]
        buf = build_warning_table(warnings, AlertLevel.NORMAL, font_path=None, dpi=72)
        assert _is_valid_png(buf)

    def test_returns_valid_png_caution(self):
        from forma.student_longitudinal_charts import build_warning_table

        warnings = [
            WarningSignal(
                name="저조한 개념 커버리지",
                triggered=True,
                severity="non-critical",
                detail="concept_coverage 0.25 < 0.30",
            ),
        ]
        buf = build_warning_table(warnings, AlertLevel.CAUTION, font_path=None, dpi=72)
        assert _is_valid_png(buf)

    def test_empty_warnings_no_crash(self):
        from forma.student_longitudinal_charts import build_warning_table

        buf = build_warning_table([], AlertLevel.NORMAL, font_path=None, dpi=72)
        assert _is_valid_png(buf)
