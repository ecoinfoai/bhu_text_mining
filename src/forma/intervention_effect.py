"""Compute intervention effect analysis from pre/post ensemble_score.

Provides InterventionEffect computation (pre/post mean comparison) and
InterventionTypeSummary aggregation by intervention type.

Dataclasses:
    InterventionEffect: Per-student, per-intervention effect measurement.
    InterventionTypeSummary: Per-type aggregate statistics.

Functions:
    compute_intervention_effects: Compute effects from log + store.
    compute_type_summary: Aggregate effects by intervention type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InterventionEffect:
    """Per-student, per-intervention effect measurement.

    Attributes:
        student_id: Student identifier.
        intervention_id: Intervention record ID.
        intervention_type: Type of intervention (e.g., "면담").
        intervention_week: Week the intervention occurred.
        pre_mean: Mean ensemble_score in the N weeks before intervention.
        post_mean: Mean ensemble_score in the N weeks after intervention.
        score_change: post_mean - pre_mean (None if insufficient data).
        sufficient_data: Whether enough pre/post data exists.
    """

    student_id: str
    intervention_id: int
    intervention_type: str
    intervention_week: int
    pre_mean: Optional[float]
    post_mean: Optional[float]
    score_change: Optional[float]
    sufficient_data: bool


@dataclass
class InterventionTypeSummary:
    """Per-type aggregate statistics.

    Attributes:
        intervention_type: Type of intervention.
        n_total: Total interventions of this type.
        n_sufficient: Count with sufficient data.
        n_positive: Count with positive score_change (> 0).
        n_negative: Count with negative score_change (< 0).
        mean_change: Mean score_change across sufficient-data effects.
    """

    intervention_type: str
    n_total: int
    n_sufficient: int
    n_positive: int
    n_negative: int
    mean_change: float


def compute_intervention_effects(
    log,
    store,
    window: int = 2,
) -> list[InterventionEffect]:
    """Compute pre/post intervention effects from log and longitudinal store.

    For each intervention record, computes the mean ensemble_score in the
    `window` weeks before the intervention and the `window` weeks after.
    The intervention week itself is excluded from both pre and post.

    Args:
        log: InterventionLog instance (must be loaded).
        store: LongitudinalStore instance with get_student_trajectory().
        window: Number of weeks before/after to average (default: 2).

    Returns:
        List of InterventionEffect, one per intervention record.
    """
    records = log.get_records()
    effects: list[InterventionEffect] = []

    for rec in records:
        trajectory = store.get_student_trajectory(
            rec.student_id, "ensemble_score",
        )
        # trajectory is list of (week, score) tuples
        week_scores = {w: s for w, s in trajectory}

        # Compute pre weeks: (intervention_week - window) .. (intervention_week - 1)
        pre_weeks = [
            rec.week - i for i in range(1, window + 1)
        ]
        pre_scores = [week_scores[w] for w in pre_weeks if w in week_scores]

        # Compute post weeks: (intervention_week + 1) .. (intervention_week + window)
        post_weeks = [
            rec.week + i for i in range(1, window + 1)
        ]
        post_scores = [week_scores[w] for w in post_weeks if w in week_scores]

        if len(pre_scores) >= window and len(post_scores) >= window:
            pre_mean = sum(pre_scores) / len(pre_scores)
            post_mean = sum(post_scores) / len(post_scores)
            score_change = post_mean - pre_mean
            effects.append(InterventionEffect(
                student_id=rec.student_id,
                intervention_id=rec.id,
                intervention_type=rec.intervention_type,
                intervention_week=rec.week,
                pre_mean=pre_mean,
                post_mean=post_mean,
                score_change=score_change,
                sufficient_data=True,
            ))
        else:
            effects.append(InterventionEffect(
                student_id=rec.student_id,
                intervention_id=rec.id,
                intervention_type=rec.intervention_type,
                intervention_week=rec.week,
                pre_mean=None,
                post_mean=None,
                score_change=None,
                sufficient_data=False,
            ))

    return effects


def compute_type_summary(
    effects: list[InterventionEffect],
) -> list[InterventionTypeSummary]:
    """Aggregate intervention effects by type.

    Args:
        effects: List of InterventionEffect instances.

    Returns:
        List of InterventionTypeSummary, one per unique intervention_type.
    """
    if not effects:
        return []

    # Group by type
    by_type: dict[str, list[InterventionEffect]] = {}
    for e in effects:
        by_type.setdefault(e.intervention_type, []).append(e)

    summaries: list[InterventionTypeSummary] = []
    for itype, group in by_type.items():
        sufficient = [e for e in group if e.sufficient_data]
        changes = [e.score_change for e in sufficient if e.score_change is not None]
        n_positive = sum(1 for c in changes if c > 0)
        n_negative = sum(1 for c in changes if c < 0)
        mean_change = sum(changes) / len(changes) if changes else 0.0

        summaries.append(InterventionTypeSummary(
            intervention_type=itype,
            n_total=len(group),
            n_sufficient=len(sufficient),
            n_positive=n_positive,
            n_negative=n_negative,
            mean_change=mean_change,
        ))

    return summaries
