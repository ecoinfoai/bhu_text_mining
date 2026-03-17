"""Student longitudinal data extraction and early warning evaluation.

Provides dataclasses and functions for building per-student longitudinal
trajectories, cohort distributions, warning signals, and anonymized
summaries from the LongitudinalStore.
"""

from __future__ import annotations

import csv
import enum
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from forma.longitudinal_store import LongitudinalStore

logger = logging.getLogger(__name__)

__all__ = [
    "AlertLevel",
    "AnonymizedStudentSummary",
    "CohortDistribution",
    "CohortWeekStats",
    "StudentLongitudinalData",
    "WarningSignal",
    "anonymize",
    "build_cohort_distribution",
    "build_student_data",
    "evaluate_warnings",
    "parse_id_csv",
]

# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

_RISK_ZONE_THRESHOLD = 0.45
_LOW_PERCENTILE_THRESHOLD = 20.0
_LOW_COVERAGE_THRESHOLD = 0.30
_SLOPE_RISING_THRESHOLD = 0.05
_SLOPE_FALLING_THRESHOLD = -0.05
_CONSECUTIVE_DECLINE_MIN_WEEKS = 3

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortWeekStats:
    """Per-week summary statistics for a single metric.

    Args:
        week: Week number.
        median: 50th percentile.
        q1: 25th percentile.
        q3: 75th percentile.
        min: Minimum value.
        max: Maximum value.
        mean: Arithmetic mean.
        std: Standard deviation.
        n: Number of students.
    """

    week: int
    median: float
    q1: float
    q3: float
    min: float
    max: float
    mean: float
    std: float
    n: int


@dataclass
class CohortDistribution:
    """Pre-computed cohort-wide statistics for box plots and percentile computation.

    Args:
        weekly_scores: {week: [all students' avg ensemble_scores]}.
        weekly_q_scores: {week: {qsn: [all students' concept_coverage]}}.
        weekly_stats: Pre-computed quartiles per week.
    """

    weekly_scores: dict[int, list[float]] = field(default_factory=dict)
    weekly_q_scores: dict[int, dict[int, list[float]]] = field(default_factory=dict)
    weekly_stats: dict[int, CohortWeekStats] = field(default_factory=dict)


@dataclass
class StudentLongitudinalData:
    """Per-student aggregated data for report generation.

    Args:
        student_id: Student identifier (학번).
        student_name: Student name from ID CSV, None if not found.
        class_name: Section (분반) from ID CSV, None if not found.
        weeks: Sorted list of weeks with data.
        scores_by_week: {week: {qsn: {metric: value}}}.
        trend_slope: OLS slope of ensemble_score across weeks (None if < 2 weeks).
        trend_direction: "상승" / "정체" / "하강" / "데이터 부족".
        percentiles_by_week: {week: percentile_rank} (0-100).
    """

    student_id: str
    student_name: Optional[str] = None
    class_name: Optional[str] = None
    weeks: list[int] = field(default_factory=list)
    scores_by_week: dict[int, dict[int, dict[str, float]]] = field(default_factory=dict)
    trend_slope: Optional[float] = None
    trend_direction: str = "데이터 부족"
    percentiles_by_week: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class WarningSignal:
    """Individual early warning indicator.

    Args:
        name: Signal name (e.g., "위험 구간 진입").
        triggered: Whether the signal is active.
        severity: "critical" or "non-critical".
        detail: Human-readable explanation.
        requires_min_weeks: Minimum weeks required (0 = always applicable).
    """

    name: str
    triggered: bool
    severity: str
    detail: str
    requires_min_weeks: int = 0


class AlertLevel(enum.Enum):
    """Aggregated warning level derived from individual warning signals."""

    NORMAL = "정상"
    CAUTION = "주의"
    WARNING = "경고"


@dataclass
class AnonymizedStudentSummary:
    """Data packet for LLM interpretation (no PII).

    Args:
        weekly_coverage_q1: {week: concept_coverage} for Q1.
        weekly_coverage_q2: {week: concept_coverage} for Q2.
        weekly_ensemble: {week: ensemble_score average}.
        percentiles: {week: percentile_rank}.
        trend_slope: OLS slope.
        trend_direction: 상승/정체/하강/데이터 부족.
        alert_level: 정상/주의/경고.
        triggered_signals: Signal names that fired.
        component_breakdown: {week: {metric: value}} averaged across questions.
    """

    weekly_coverage_q1: dict[int, float] = field(default_factory=dict)
    weekly_coverage_q2: dict[int, float] = field(default_factory=dict)
    weekly_ensemble: dict[int, float] = field(default_factory=dict)
    percentiles: dict[int, float] = field(default_factory=dict)
    trend_slope: Optional[float] = None
    trend_direction: str = "데이터 부족"
    alert_level: str = "정상"
    triggered_signals: list[str] = field(default_factory=list)
    component_breakdown: dict[int, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# T007: build_cohort_distribution
# ---------------------------------------------------------------------------


def build_cohort_distribution(
    store: LongitudinalStore,
    weeks: list[int],
) -> CohortDistribution:
    """Compute cohort-wide statistics from all students' scores per week.

    Args:
        store: Loaded LongitudinalStore instance.
        weeks: List of week numbers to include.

    Returns:
        CohortDistribution with per-week stats, score lists, and per-question
        concept_coverage distributions.
    """
    all_records = store.get_all_records()

    # Group records: {week: {student_id: {qsn: scores_dict}}}
    week_student_q: dict[int, dict[str, dict[int, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for rec in all_records:
        if rec.week in weeks:
            week_student_q[rec.week][rec.student_id][rec.question_sn] = dict(rec.scores)

    cohort = CohortDistribution()

    for week in sorted(weeks):
        student_q_map = week_student_q.get(week, {})
        if not student_q_map:
            continue

        # Per-student average ensemble_score across questions
        student_avg_scores: list[float] = []
        # Per-question concept_coverage
        q_coverage: dict[int, list[float]] = defaultdict(list)

        for student_id, q_map in student_q_map.items():
            # Average ensemble_score across all questions for this student-week
            ensemble_vals = [
                scores.get("ensemble_score", 0.0)
                for scores in q_map.values()
                if "ensemble_score" in scores
            ]
            if ensemble_vals:
                student_avg_scores.append(sum(ensemble_vals) / len(ensemble_vals))

            # Per-question concept_coverage
            for qsn, scores in q_map.items():
                if "concept_coverage" in scores:
                    q_coverage[qsn].append(scores["concept_coverage"])

        cohort.weekly_scores[week] = student_avg_scores
        cohort.weekly_q_scores[week] = dict(q_coverage)

        if student_avg_scores:
            arr = np.array(student_avg_scores)
            cohort.weekly_stats[week] = CohortWeekStats(
                week=week,
                median=float(np.median(arr)),
                q1=float(np.percentile(arr, 25)),
                q3=float(np.percentile(arr, 75)),
                min=float(np.min(arr)),
                max=float(np.max(arr)),
                mean=float(np.mean(arr)),
                std=float(np.std(arr, ddof=0)),
                n=len(student_avg_scores),
            )

    return cohort


# ---------------------------------------------------------------------------
# T008: build_student_data
# ---------------------------------------------------------------------------


def _compute_percentile(value: float, scores: list[float]) -> float:
    """Compute percentile rank of value within scores list.

    Uses 'weak' method: percentage of scores strictly less than value,
    then adds half of those equal to value (mean rank approach).

    Args:
        value: The value to rank.
        scores: List of all cohort scores.

    Returns:
        Percentile rank (0-100).
    """
    if not scores:
        return 0.0
    n = len(scores)
    below = sum(1 for s in scores if s < value)
    equal = sum(1 for s in scores if abs(s - value) < 1e-9)
    return 100.0 * (below + 0.5 * equal) / n


def build_student_data(
    store: LongitudinalStore,
    student_id: str,
    weeks: list[int],
    cohort: CohortDistribution,
    student_name: Optional[str] = None,
    class_name: Optional[str] = None,
) -> StudentLongitudinalData:
    """Build per-student longitudinal data from the store.

    Args:
        store: Loaded LongitudinalStore instance.
        student_id: Target student identifier (학번).
        weeks: List of week numbers to include.
        cohort: Pre-computed CohortDistribution for percentile calculation.
        student_name: Optional student name from ID CSV.
        class_name: Optional section name from ID CSV.

    Returns:
        StudentLongitudinalData with scores, trend, and percentiles.
    """
    history = store.get_student_history(student_id)

    # Filter to requested weeks and build scores_by_week
    scores_by_week: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
    available_weeks: set[int] = set()

    for rec in history:
        if rec.week in weeks:
            scores_by_week[rec.week][rec.question_sn] = dict(rec.scores)
            available_weeks.add(rec.week)

    sorted_weeks = sorted(available_weeks)

    # Compute per-week average ensemble_score for trend
    weekly_avg_ensemble: dict[int, float] = {}
    for week in sorted_weeks:
        q_map = scores_by_week[week]
        ensemble_vals = [
            scores.get("ensemble_score", 0.0)
            for scores in q_map.values()
            if "ensemble_score" in scores
        ]
        if ensemble_vals:
            weekly_avg_ensemble[week] = sum(ensemble_vals) / len(ensemble_vals)

    # OLS trend slope
    trend_slope: Optional[float] = None
    trend_direction = "데이터 부족"
    if len(weekly_avg_ensemble) >= 2:
        x = np.array(list(weekly_avg_ensemble.keys()), dtype=float)
        y = np.array(list(weekly_avg_ensemble.values()), dtype=float)
        coeffs = np.polyfit(x, y, 1)
        trend_slope = float(coeffs[0])
        if trend_slope > _SLOPE_RISING_THRESHOLD:
            trend_direction = "상승"
        elif trend_slope < _SLOPE_FALLING_THRESHOLD:
            trend_direction = "하강"
        else:
            trend_direction = "정체"

    # Percentile computation
    percentiles_by_week: dict[int, float] = {}
    for week in sorted_weeks:
        if week in weekly_avg_ensemble and week in cohort.weekly_scores:
            percentiles_by_week[week] = _compute_percentile(
                weekly_avg_ensemble[week],
                cohort.weekly_scores[week],
            )

    return StudentLongitudinalData(
        student_id=student_id,
        student_name=student_name,
        class_name=class_name,
        weeks=sorted_weeks,
        scores_by_week=dict(scores_by_week),
        trend_slope=trend_slope,
        trend_direction=trend_direction,
        percentiles_by_week=percentiles_by_week,
    )


# ---------------------------------------------------------------------------
# T009: evaluate_warnings
# ---------------------------------------------------------------------------


def evaluate_warnings(
    student_data: StudentLongitudinalData,
    cohort: CohortDistribution,
) -> tuple[list[WarningSignal], AlertLevel]:
    """Evaluate early warning signals for a student.

    Signals:
        - 위험 구간 진입: latest ensemble_score < 0.45 (critical, 0 weeks required)
        - 하위 백분위: latest percentile < 20 (critical, 0 weeks required)
        - 저조한 개념 커버리지: latest concept_coverage < 0.30 (non-critical, 0 weeks required)
        - 연속 하강: 3+ consecutive weeks of declining ensemble_score (non-critical, 3 weeks required)
        - 음수 추세 기울기: OLS slope < -0.05 (non-critical, 3 weeks required)

    Args:
        student_data: Pre-built StudentLongitudinalData.
        cohort: Pre-computed CohortDistribution.

    Returns:
        Tuple of (list of WarningSignal, overall AlertLevel).
    """
    signals: list[WarningSignal] = []
    n_weeks = len(student_data.weeks)

    # --- Compute latest week metrics ---
    latest_week = student_data.weeks[-1] if student_data.weeks else None

    # Latest average ensemble_score
    latest_ensemble: Optional[float] = None
    if latest_week and latest_week in student_data.scores_by_week:
        q_map = student_data.scores_by_week[latest_week]
        vals = [s.get("ensemble_score", 0.0) for s in q_map.values() if "ensemble_score" in s]
        if vals:
            latest_ensemble = sum(vals) / len(vals)

    # Latest average concept_coverage
    latest_coverage: Optional[float] = None
    if latest_week and latest_week in student_data.scores_by_week:
        q_map = student_data.scores_by_week[latest_week]
        vals = [s.get("concept_coverage", 0.0) for s in q_map.values() if "concept_coverage" in s]
        if vals:
            latest_coverage = sum(vals) / len(vals)

    # Latest percentile
    latest_percentile: Optional[float] = None
    if latest_week and latest_week in student_data.percentiles_by_week:
        latest_percentile = student_data.percentiles_by_week[latest_week]

    # --- Signal 1: 위험 구간 진입 (critical) ---
    risk_triggered = latest_ensemble is not None and latest_ensemble < _RISK_ZONE_THRESHOLD
    signals.append(WarningSignal(
        name="위험 구간 진입",
        triggered=risk_triggered,
        severity="critical",
        detail=f"ensemble_score {latest_ensemble:.2f} < {_RISK_ZONE_THRESHOLD}" if risk_triggered else "정상 범위",
        requires_min_weeks=0,
    ))

    # --- Signal 2: 하위 백분위 (critical) ---
    low_pct_triggered = latest_percentile is not None and latest_percentile < _LOW_PERCENTILE_THRESHOLD
    signals.append(WarningSignal(
        name="하위 백분위",
        triggered=low_pct_triggered,
        severity="critical",
        detail=f"백분위 {latest_percentile:.1f} < {_LOW_PERCENTILE_THRESHOLD}" if low_pct_triggered else "정상 범위",
        requires_min_weeks=0,
    ))

    # --- Signal 3: 저조한 개념 커버리지 (non-critical) ---
    low_cov_triggered = latest_coverage is not None and latest_coverage < _LOW_COVERAGE_THRESHOLD
    signals.append(WarningSignal(
        name="저조한 개념 커버리지",
        triggered=low_cov_triggered,
        severity="non-critical",
        detail=f"concept_coverage {latest_coverage:.2f} < {_LOW_COVERAGE_THRESHOLD}" if low_cov_triggered else "정상 범위",
        requires_min_weeks=0,
    ))

    # --- Signal 4: 연속 하강 (non-critical, requires 3+ weeks) ---
    if n_weeks >= _CONSECUTIVE_DECLINE_MIN_WEEKS:
        # Compute per-week average ensemble
        weekly_avgs: list[tuple[int, float]] = []
        for week in student_data.weeks:
            q_map = student_data.scores_by_week.get(week, {})
            vals = [s.get("ensemble_score", 0.0) for s in q_map.values() if "ensemble_score" in s]
            if vals:
                weekly_avgs.append((week, sum(vals) / len(vals)))

        # Check for consecutive decline
        consecutive = 0
        max_consecutive = 0
        for i in range(1, len(weekly_avgs)):
            if weekly_avgs[i][1] < weekly_avgs[i - 1][1]:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        decline_triggered = max_consecutive >= _CONSECUTIVE_DECLINE_MIN_WEEKS - 1  # 3 weeks = 2 declines
        signals.append(WarningSignal(
            name="연속 하강",
            triggered=decline_triggered,
            severity="non-critical",
            detail=f"{max_consecutive + 1}주 연속 하강" if decline_triggered else "연속 하강 없음",
            requires_min_weeks=_CONSECUTIVE_DECLINE_MIN_WEEKS,
        ))
    else:
        signals.append(WarningSignal(
            name="연속 하강",
            triggered=False,
            severity="non-critical",
            detail="데이터 부족 (최소 3주 필요)",
            requires_min_weeks=_CONSECUTIVE_DECLINE_MIN_WEEKS,
        ))

    # --- Signal 5: 음수 추세 기울기 (non-critical, requires 3+ weeks) ---
    if n_weeks >= _CONSECUTIVE_DECLINE_MIN_WEEKS:
        neg_slope_triggered = (
            student_data.trend_slope is not None
            and student_data.trend_slope < _SLOPE_FALLING_THRESHOLD
        )
        signals.append(WarningSignal(
            name="음수 추세 기울기",
            triggered=neg_slope_triggered,
            severity="non-critical",
            detail=f"기울기 {student_data.trend_slope:.4f}" if student_data.trend_slope is not None else "기울기 없음",
            requires_min_weeks=_CONSECUTIVE_DECLINE_MIN_WEEKS,
        ))
    else:
        signals.append(WarningSignal(
            name="음수 추세 기울기",
            triggered=False,
            severity="non-critical",
            detail="데이터 부족 (최소 3주 필요)",
            requires_min_weeks=_CONSECUTIVE_DECLINE_MIN_WEEKS,
        ))

    # --- Determine AlertLevel ---
    triggered_signals = [s for s in signals if s.triggered]
    has_critical = any(s.severity == "critical" for s in triggered_signals)
    has_non_critical = any(s.severity == "non-critical" for s in triggered_signals)

    if has_critical:
        level = AlertLevel.WARNING
    elif has_non_critical:
        level = AlertLevel.CAUTION
    else:
        level = AlertLevel.NORMAL

    return signals, level


# ---------------------------------------------------------------------------
# T010: parse_id_csv
# ---------------------------------------------------------------------------

# Google Forms CSV column names
_COL_STUDENT_ID = "학번을 입력하세요."
_COL_STUDENT_NAME = "이름을 입력하세요."
_COL_CLASS_NAME = "분반을 선택하세요."


def parse_id_csv(csv_path: str) -> dict[str, tuple[str, str]]:
    """Parse Google Forms ID CSV and return student ID mapping.

    Args:
        csv_path: Path to the CSV file with columns:
            타임스탬프, 익명ID, 분반을 선택하세요., 학번을 입력하세요., 이름을 입력하세요.

    Returns:
        Dict mapping 학번 → (이름, 분반). Empty dict if file is missing
        or columns are not found.
    """
    if not os.path.exists(csv_path):
        logger.warning("ID CSV not found: %s", csv_path)
        return {}

    result: dict[str, tuple[str, str]] = {}
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                logger.warning("Empty CSV file: %s", csv_path)
                return {}

            # Check required columns exist
            required = {_COL_STUDENT_ID, _COL_STUDENT_NAME, _COL_CLASS_NAME}
            if not required.issubset(set(reader.fieldnames)):
                logger.warning(
                    "CSV missing required columns. Expected %s, got %s",
                    required,
                    set(reader.fieldnames),
                )
                return {}

            for row in reader:
                student_id = row.get(_COL_STUDENT_ID, "").strip()
                student_name = row.get(_COL_STUDENT_NAME, "").strip()
                class_name = row.get(_COL_CLASS_NAME, "").strip()
                if student_id:
                    result[student_id] = (student_name, class_name)
    except (OSError, csv.Error) as exc:
        logger.warning("Failed to parse ID CSV %s: %s", csv_path, exc)
        return {}

    return result


# ---------------------------------------------------------------------------
# T011: anonymize
# ---------------------------------------------------------------------------


def anonymize(
    student_data: StudentLongitudinalData,
    signals: list[WarningSignal],
) -> AnonymizedStudentSummary:
    """Create an anonymized summary from student data for LLM interpretation.

    The resulting summary contains zero PII — no student_id, student_name,
    or class_name.

    Args:
        student_data: Per-student longitudinal data.
        signals: List of evaluated WarningSignal instances.

    Returns:
        AnonymizedStudentSummary with numerical data only.
    """
    weekly_coverage_q1: dict[int, float] = {}
    weekly_coverage_q2: dict[int, float] = {}
    weekly_ensemble: dict[int, float] = {}
    component_breakdown: dict[int, dict[str, float]] = {}

    for week in student_data.weeks:
        q_map = student_data.scores_by_week.get(week, {})

        # Q1 (question_sn=1) concept_coverage
        if 1 in q_map and "concept_coverage" in q_map[1]:
            weekly_coverage_q1[week] = q_map[1]["concept_coverage"]

        # Q2 (question_sn=2) concept_coverage
        if 2 in q_map and "concept_coverage" in q_map[2]:
            weekly_coverage_q2[week] = q_map[2]["concept_coverage"]

        # Average ensemble_score across questions
        ensemble_vals = [
            s.get("ensemble_score", 0.0)
            for s in q_map.values()
            if "ensemble_score" in s
        ]
        if ensemble_vals:
            weekly_ensemble[week] = sum(ensemble_vals) / len(ensemble_vals)

        # Component breakdown: average each metric across questions
        metrics = ("concept_coverage", "llm_rubric", "ensemble_score", "rasch_ability")
        breakdown: dict[str, float] = {}
        for metric in metrics:
            vals = [s[metric] for s in q_map.values() if metric in s]
            if vals:
                breakdown[metric] = sum(vals) / len(vals)
        if breakdown:
            component_breakdown[week] = breakdown

    # Determine alert level from signals
    triggered = [s for s in signals if s.triggered]
    has_critical = any(s.severity == "critical" for s in triggered)
    has_non_critical = any(s.severity == "non-critical" for s in triggered)
    if has_critical:
        alert_str = AlertLevel.WARNING.value
    elif has_non_critical:
        alert_str = AlertLevel.CAUTION.value
    else:
        alert_str = AlertLevel.NORMAL.value

    return AnonymizedStudentSummary(
        weekly_coverage_q1=weekly_coverage_q1,
        weekly_coverage_q2=weekly_coverage_q2,
        weekly_ensemble=weekly_ensemble,
        percentiles=dict(student_data.percentiles_by_week),
        trend_slope=student_data.trend_slope,
        trend_direction=student_data.trend_direction,
        alert_level=alert_str,
        triggered_signals=[s.name for s in triggered],
        component_breakdown=component_breakdown,
    )
