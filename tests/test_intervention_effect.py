"""Tests for intervention_effect.py — compute intervention effects.

T019-T020 [US2]: InterventionEffect computation, type-level summaries.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_store(trajectories: dict[str, list[tuple[int, float]]]):
    """Create a mock LongitudinalStore with get_student_trajectory."""
    store = MagicMock()

    def _get_trajectory(student_id: str, metric: str):
        return trajectories.get(student_id, [])

    store.get_student_trajectory.side_effect = _get_trajectory
    return store


def _make_intervention_log(records: list[dict], tmp_path: Path) -> "InterventionLog":  # noqa: F821
    """Create a real InterventionLog with pre-loaded records."""
    from forma.intervention_store import InterventionLog

    store_path = str(tmp_path / "intervention_log.yaml")
    data = {
        "_meta": {"next_id": len(records) + 1},
        "records": records,
    }
    Path(store_path).write_text(
        yaml.dump(data, allow_unicode=True),
        encoding="utf-8",
    )
    log = InterventionLog(store_path)
    log.load()
    return log


# ---------------------------------------------------------------------------
# T019: InterventionEffect + compute_intervention_effects
# ---------------------------------------------------------------------------


class TestInterventionEffectDataclass:
    """Tests for InterventionEffect dataclass."""

    def test_fields(self):
        """InterventionEffect has required fields."""
        from forma.intervention_effect import InterventionEffect

        e = InterventionEffect(
            student_id="S001",
            intervention_id=1,
            intervention_type="면담",
            intervention_week=3,
            pre_mean=0.40,
            post_mean=0.55,
            score_change=0.15,
            sufficient_data=True,
        )
        assert e.student_id == "S001"
        assert e.intervention_id == 1
        assert e.intervention_type == "면담"
        assert e.intervention_week == 3
        assert e.pre_mean == 0.40
        assert e.post_mean == 0.55
        assert e.score_change == 0.15
        assert e.sufficient_data is True

    def test_insufficient_data(self):
        """InterventionEffect with insufficient data has None means."""
        from forma.intervention_effect import InterventionEffect

        e = InterventionEffect(
            student_id="S001",
            intervention_id=1,
            intervention_type="면담",
            intervention_week=5,
            pre_mean=None,
            post_mean=None,
            score_change=None,
            sufficient_data=False,
        )
        assert e.sufficient_data is False
        assert e.pre_mean is None
        assert e.score_change is None


class TestComputeInterventionEffects:
    """Tests for compute_intervention_effects()."""

    def test_basic_pre_post(self, tmp_path):
        """Computes pre/post means correctly with N=2 window (FR-008)."""
        from forma.intervention_effect import compute_intervention_effects

        # S001: intervention at week 3
        # Trajectory: w1=0.30, w2=0.40, w3=0.45, w4=0.60, w5=0.65
        # pre (w1,w2): mean=0.35, post (w4,w5): mean=0.625
        store = _make_mock_store(
            {
                "S001": [(1, 0.30), (2, 0.40), (3, 0.45), (4, 0.60), (5, 0.65)],
            }
        )
        log = _make_intervention_log(
            [
                {
                    "id": 1,
                    "student_id": "S001",
                    "week": 3,
                    "intervention_type": "면담",
                    "description": "상담",
                    "recorded_at": "2026-01-01T00:00:00+00:00",
                    "outcome": None,
                }
            ],
            tmp_path,
        )

        effects = compute_intervention_effects(log, store, window=2)
        assert len(effects) == 1
        e = effects[0]
        assert e.student_id == "S001"
        assert e.sufficient_data is True
        assert abs(e.pre_mean - 0.35) < 0.001
        assert abs(e.post_mean - 0.625) < 0.001
        assert abs(e.score_change - 0.275) < 0.001

    def test_insufficient_post_data(self, tmp_path):
        """Insufficient post data when < N weeks after intervention (FR-012)."""
        from forma.intervention_effect import compute_intervention_effects

        # Intervention at week 4, only 1 week of post data
        store = _make_mock_store(
            {
                "S001": [(1, 0.30), (2, 0.40), (3, 0.50), (4, 0.55), (5, 0.60)],
            }
        )
        log = _make_intervention_log(
            [
                {
                    "id": 1,
                    "student_id": "S001",
                    "week": 4,
                    "intervention_type": "보충학습",
                    "description": "",
                    "recorded_at": "2026-01-01T00:00:00+00:00",
                    "outcome": None,
                }
            ],
            tmp_path,
        )

        effects = compute_intervention_effects(log, store, window=2)
        assert len(effects) == 1
        e = effects[0]
        assert e.sufficient_data is False

    def test_insufficient_pre_data(self, tmp_path):
        """Insufficient pre data when < N weeks before intervention."""
        from forma.intervention_effect import compute_intervention_effects

        # Intervention at week 1 — no pre data
        store = _make_mock_store(
            {
                "S001": [(1, 0.30), (2, 0.40), (3, 0.50)],
            }
        )
        log = _make_intervention_log(
            [
                {
                    "id": 1,
                    "student_id": "S001",
                    "week": 1,
                    "intervention_type": "면담",
                    "description": "",
                    "recorded_at": "2026-01-01T00:00:00+00:00",
                    "outcome": None,
                }
            ],
            tmp_path,
        )

        effects = compute_intervention_effects(log, store, window=2)
        assert len(effects) == 1
        assert effects[0].sufficient_data is False

    def test_multiple_interventions(self, tmp_path):
        """Multiple interventions for different students."""
        from forma.intervention_effect import compute_intervention_effects

        store = _make_mock_store(
            {
                "S001": [(1, 0.30), (2, 0.40), (3, 0.50), (4, 0.60), (5, 0.70)],
                "S002": [(1, 0.20), (2, 0.25), (3, 0.30), (4, 0.45), (5, 0.50)],
            }
        )
        log = _make_intervention_log(
            [
                {
                    "id": 1,
                    "student_id": "S001",
                    "week": 3,
                    "intervention_type": "면담",
                    "description": "",
                    "recorded_at": "2026-01-01T00:00:00+00:00",
                    "outcome": None,
                },
                {
                    "id": 2,
                    "student_id": "S002",
                    "week": 3,
                    "intervention_type": "보충학습",
                    "description": "",
                    "recorded_at": "2026-01-01T00:00:00+00:00",
                    "outcome": None,
                },
            ],
            tmp_path,
        )

        effects = compute_intervention_effects(log, store, window=2)
        assert len(effects) == 2

    def test_empty_log(self, tmp_path):
        """Empty log returns empty list."""
        from forma.intervention_effect import compute_intervention_effects

        store = _make_mock_store({})
        log = _make_intervention_log([], tmp_path)
        effects = compute_intervention_effects(log, store)
        assert effects == []

    def test_no_trajectory_data(self, tmp_path):
        """Student with no trajectory data returns insufficient."""
        from forma.intervention_effect import compute_intervention_effects

        store = _make_mock_store({"S001": []})
        log = _make_intervention_log(
            [
                {
                    "id": 1,
                    "student_id": "S001",
                    "week": 3,
                    "intervention_type": "면담",
                    "description": "",
                    "recorded_at": "2026-01-01T00:00:00+00:00",
                    "outcome": None,
                }
            ],
            tmp_path,
        )

        effects = compute_intervention_effects(log, store, window=2)
        assert len(effects) == 1
        assert effects[0].sufficient_data is False

    def test_custom_window(self, tmp_path):
        """Custom window size changes pre/post calculation."""
        from forma.intervention_effect import compute_intervention_effects

        # With window=1: pre=w2=0.40, post=w4=0.60
        store = _make_mock_store(
            {
                "S001": [(1, 0.30), (2, 0.40), (3, 0.50), (4, 0.60), (5, 0.70)],
            }
        )
        log = _make_intervention_log(
            [
                {
                    "id": 1,
                    "student_id": "S001",
                    "week": 3,
                    "intervention_type": "면담",
                    "description": "",
                    "recorded_at": "2026-01-01T00:00:00+00:00",
                    "outcome": None,
                }
            ],
            tmp_path,
        )

        effects = compute_intervention_effects(log, store, window=1)
        assert len(effects) == 1
        e = effects[0]
        assert e.sufficient_data is True
        assert abs(e.pre_mean - 0.40) < 0.001
        assert abs(e.post_mean - 0.60) < 0.001


# ---------------------------------------------------------------------------
# T020: InterventionTypeSummary + compute_type_summary
# ---------------------------------------------------------------------------


class TestInterventionTypeSummary:
    """Tests for InterventionTypeSummary and compute_type_summary()."""

    def test_dataclass_fields(self):
        """InterventionTypeSummary has expected fields."""
        from forma.intervention_effect import InterventionTypeSummary

        s = InterventionTypeSummary(
            intervention_type="면담",
            n_total=10,
            n_sufficient=8,
            n_positive=6,
            n_negative=2,
            mean_change=0.12,
        )
        assert s.intervention_type == "면담"
        assert s.n_total == 10
        assert s.n_sufficient == 8
        assert s.n_positive == 6
        assert s.n_negative == 2
        assert abs(s.mean_change - 0.12) < 0.001

    def test_compute_type_summary_basic(self):
        """compute_type_summary groups effects by type."""
        from forma.intervention_effect import (
            InterventionEffect,
            compute_type_summary,
        )

        effects = [
            InterventionEffect("S001", 1, "면담", 3, 0.30, 0.50, 0.20, True),
            InterventionEffect("S002", 2, "면담", 3, 0.40, 0.45, 0.05, True),
            InterventionEffect("S003", 3, "보충학습", 3, 0.50, 0.40, -0.10, True),
        ]
        summaries = compute_type_summary(effects)
        assert len(summaries) == 2

        # Find 면담 summary
        mentoring = [s for s in summaries if s.intervention_type == "면담"]
        assert len(mentoring) == 1
        m = mentoring[0]
        assert m.n_total == 2
        assert m.n_sufficient == 2
        assert m.n_positive == 2
        assert m.n_negative == 0
        assert abs(m.mean_change - 0.125) < 0.001

    def test_compute_type_summary_with_insufficient(self):
        """Insufficient data effects count in n_total but not n_sufficient."""
        from forma.intervention_effect import (
            InterventionEffect,
            compute_type_summary,
        )

        effects = [
            InterventionEffect("S001", 1, "면담", 3, 0.30, 0.50, 0.20, True),
            InterventionEffect("S002", 2, "면담", 5, None, None, None, False),
        ]
        summaries = compute_type_summary(effects)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.n_total == 2
        assert s.n_sufficient == 1
        assert s.n_positive == 1
        assert s.mean_change == 0.20

    def test_compute_type_summary_empty(self):
        """Empty effects list returns empty summaries."""
        from forma.intervention_effect import compute_type_summary

        assert compute_type_summary([]) == []

    def test_compute_type_summary_negative_changes(self):
        """Negative changes are counted as n_negative."""
        from forma.intervention_effect import (
            InterventionEffect,
            compute_type_summary,
        )

        effects = [
            InterventionEffect("S001", 1, "과제부여", 3, 0.60, 0.45, -0.15, True),
            InterventionEffect("S002", 2, "과제부여", 3, 0.50, 0.50, 0.00, True),
        ]
        summaries = compute_type_summary(effects)
        s = summaries[0]
        assert s.n_negative == 1
        assert s.n_positive == 0  # 0.00 is not positive
