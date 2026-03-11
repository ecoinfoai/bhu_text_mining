"""Longitudinal summary report data model and builder.

Provides dataclasses and build function for generating a longitudinal
period summary PDF report from the LongitudinalStore.
No LLM API calls.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from forma.longitudinal_store import LongitudinalStore


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StudentTrajectory:
    """One student's full-period trajectory data.

    Args:
        student_id: Student identifier.
        weekly_scores: {week: ensemble_score} for selected period.
        overall_trend: OLS linear regression slope (numpy.polyfit deg=1).
        is_persistent_risk: True if at_risk every week in period.
        risk_weeks: List of weeks where student was at_risk.
    """

    student_id: str
    weekly_scores: dict[int, float]
    overall_trend: float
    is_persistent_risk: bool
    risk_weeks: list[int]


@dataclass
class ConceptMasteryChange:
    """Per-concept class mastery ratio change across period.

    Args:
        concept: Concept name.
        week_start_ratio: Class average ratio at first week (0.0-1.0).
        week_end_ratio: Class average ratio at last week (0.0-1.0).
        delta: week_end_ratio - week_start_ratio.
    """

    concept: str
    week_start_ratio: float
    week_end_ratio: float
    delta: float


@dataclass
class LongitudinalSummaryData:
    """Container for all data needed to generate the longitudinal summary PDF.

    Args:
        class_name: Class/section identifier.
        period_weeks: Sorted list of week numbers in the period.
        student_trajectories: All student trajectories.
        class_weekly_averages: {week: class_mean_score}.
        persistent_risk_students: Student IDs at_risk every week.
        concept_mastery_changes: Per-concept changes, sorted by delta desc.
        total_students: Total unique student count.
    """

    class_name: str
    period_weeks: list[int]
    student_trajectories: list[StudentTrajectory]
    class_weekly_averages: dict[int, float]
    persistent_risk_students: list[str]
    concept_mastery_changes: list[ConceptMasteryChange]
    total_students: int
    risk_predictions: list | None = None


# ---------------------------------------------------------------------------
# At-risk threshold (simple criterion: ensemble_score < 0.45)
# ---------------------------------------------------------------------------

_RISK_THRESHOLD = 0.45


def _is_score_at_risk(score: float) -> bool:
    """Return True if score is below at-risk threshold."""
    return score < _RISK_THRESHOLD


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_longitudinal_summary(
    store: LongitudinalStore,
    weeks: list[int],
    class_name: str,
) -> LongitudinalSummaryData:
    """Build LongitudinalSummaryData from store for specified weeks.

    Args:
        store: LongitudinalStore with loaded records.
        weeks: List of week numbers to include.
        class_name: Class/section identifier.

    Returns:
        LongitudinalSummaryData ready for PDF generation.
    """
    sorted_weeks = sorted(weeks)

    # Get the weekly matrix: {student_id: {week: avg_ensemble_score}}
    matrix = store.get_class_weekly_matrix("ensemble_score")

    # Collect all student IDs that have data in the requested weeks
    student_ids = set()
    for sid, week_scores in matrix.items():
        if any(w in week_scores for w in sorted_weeks):
            student_ids.add(sid)

    # Build per-student trajectories
    trajectories: list[StudentTrajectory] = []
    for sid in sorted(student_ids):
        week_scores = matrix.get(sid, {})
        # Filter to requested weeks only
        filtered = {w: week_scores[w] for w in sorted_weeks if w in week_scores}
        if not filtered:
            continue

        # OLS trend via numpy.polyfit deg=1
        if len(filtered) >= 2:
            ws = list(filtered.keys())
            scores = [filtered[w] for w in ws]
            coeffs = np.polyfit(ws, scores, 1)
            trend = float(coeffs[0])
        else:
            trend = 0.0

        # Determine risk weeks
        risk_weeks = [w for w, s in filtered.items() if _is_score_at_risk(s)]

        # Persistent risk: at_risk every week in the requested period
        # (only for weeks where student has data)
        weeks_with_data = [w for w in sorted_weeks if w in filtered]
        is_persistent_risk = (
            len(weeks_with_data) > 0
            and all(_is_score_at_risk(filtered[w]) for w in weeks_with_data)
        )

        trajectories.append(StudentTrajectory(
            student_id=sid,
            weekly_scores=filtered,
            overall_trend=trend,
            is_persistent_risk=is_persistent_risk,
            risk_weeks=risk_weeks,
        ))

    # Class weekly averages
    class_weekly_averages: dict[int, float] = {}
    for w in sorted_weeks:
        week_scores = [
            matrix[sid][w]
            for sid in student_ids
            if sid in matrix and w in matrix[sid]
        ]
        if week_scores:
            class_weekly_averages[w] = float(np.mean(week_scores))

    # Persistent risk students
    persistent_risk_students = [
        t.student_id for t in trajectories if t.is_persistent_risk
    ]

    # Concept mastery changes
    concept_mastery_changes = _compute_concept_mastery_changes(store, sorted_weeks)

    return LongitudinalSummaryData(
        class_name=class_name,
        period_weeks=sorted_weeks,
        student_trajectories=trajectories,
        class_weekly_averages=class_weekly_averages,
        persistent_risk_students=persistent_risk_students,
        concept_mastery_changes=concept_mastery_changes,
        total_students=len(student_ids),
    )


def _compute_concept_mastery_changes(
    store: LongitudinalStore,
    sorted_weeks: list[int],
) -> list[ConceptMasteryChange]:
    """Compute per-concept mastery ratio change from first to last week.

    Uses concept_scores from LongitudinalRecord to compute class-average
    per-concept ratio at first and last week.

    Returns:
        List of ConceptMasteryChange sorted by delta descending.
    """
    if len(sorted_weeks) < 1:
        return []

    first_week = sorted_weeks[0]
    last_week = sorted_weeks[-1]

    # Gather concept_scores for first and last week
    first_records = store.get_class_snapshot(first_week)
    last_records = store.get_class_snapshot(last_week)

    first_concept_scores = _aggregate_concept_scores(first_records)
    last_concept_scores = _aggregate_concept_scores(last_records)

    # Compute changes for all concepts present in both weeks
    all_concepts = set(first_concept_scores.keys()) | set(last_concept_scores.keys())

    changes: list[ConceptMasteryChange] = []
    for concept in all_concepts:
        start = first_concept_scores.get(concept, 0.0)
        end = last_concept_scores.get(concept, 0.0)
        changes.append(ConceptMasteryChange(
            concept=concept,
            week_start_ratio=start,
            week_end_ratio=end,
            delta=end - start,
        ))

    # Sort by delta descending
    changes.sort(key=lambda c: c.delta, reverse=True)
    return changes


def _aggregate_concept_scores(
    records: list,
) -> dict[str, float]:
    """Aggregate concept_scores from records into per-concept class average.

    Args:
        records: List of LongitudinalRecord for one week.

    Returns:
        {concept_name: class_average_ratio}
    """
    concept_values: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if rec.concept_scores:
            for concept, ratio in rec.concept_scores.items():
                concept_values[concept].append(ratio)

    return {
        concept: float(np.mean(values))
        for concept, values in concept_values.items()
        if values
    }
