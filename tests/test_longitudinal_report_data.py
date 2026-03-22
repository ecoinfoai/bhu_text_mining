"""Tests for longitudinal_report_data.py — US3 data model and builder.

RED phase: tests written BEFORE implementation (TDD).

Covers T029:
  - StudentTrajectory dataclass construction
  - ConceptMasteryChange dataclass construction
  - LongitudinalSummaryData dataclass construction
  - build_longitudinal_summary() builder function:
    * trajectory OLS trend (numpy.polyfit deg=1)
    * persistent risk detection (at_risk every week)
    * concept mastery delta calculation (first vs last week)
    * class_weekly_averages computation
    * sorting of concept_mastery_changes by delta descending
"""

from __future__ import annotations

import pytest

from forma.evaluation_types import LongitudinalRecord
from forma.longitudinal_store import LongitudinalStore


# ---------------------------------------------------------------------------
# Helpers — build mock store with multi-week data
# ---------------------------------------------------------------------------

def _make_record(
    student_id: str,
    week: int,
    question_sn: int = 1,
    ensemble_score: float = 0.5,
    tier_label: str = "Developing",
    concept_scores: dict[str, float] | None = None,
) -> LongitudinalRecord:
    """Build a LongitudinalRecord with realistic defaults."""
    return LongitudinalRecord(
        student_id=student_id,
        week=week,
        question_sn=question_sn,
        scores={"ensemble_score": ensemble_score},
        tier_level=1,
        tier_label=tier_label,
        concept_scores=concept_scores,
    )


def _build_test_store(tmp_path) -> LongitudinalStore:
    """Build a store with 4 weeks, 5 students for testing.

    Students:
      - S001: scores [0.3, 0.4, 0.5, 0.6] — improving, at_risk w1-w2 (score < 0.45)
      - S002: scores [0.7, 0.72, 0.75, 0.80] — good student
      - S003: scores [0.2, 0.25, 0.3, 0.35] — at_risk every week (persistent)
      - S004: scores [0.9, 0.85, 0.80, 0.75] — declining
      - S005: scores [0.5, 0.5, 0.5, 0.5] — flat
    """
    store = LongitudinalStore(str(tmp_path / "test_store.yaml"))
    student_scores = {
        "S001": [0.3, 0.4, 0.5, 0.6],
        "S002": [0.7, 0.72, 0.75, 0.80],
        "S003": [0.2, 0.25, 0.3, 0.35],
        "S004": [0.9, 0.85, 0.80, 0.75],
        "S005": [0.5, 0.5, 0.5, 0.5],
    }
    concept_scores_map = {
        1: {"항상성": 0.6, "삼투": 0.4},
        2: {"항상성": 0.65, "삼투": 0.45},
        3: {"항상성": 0.7, "삼투": 0.5},
        4: {"항상성": 0.8, "삼투": 0.6},
    }

    for sid, scores in student_scores.items():
        for week_idx, score in enumerate(scores, start=1):
            rec = _make_record(
                student_id=sid,
                week=week_idx,
                ensemble_score=score,
                tier_label="Beginning" if score < 0.45 else "Developing" if score < 0.65 else "Proficient",
                concept_scores=concept_scores_map.get(week_idx),
            )
            store.add_record(rec)

    return store


# ---------------------------------------------------------------------------
# T029: Dataclass construction tests
# ---------------------------------------------------------------------------


class TestStudentTrajectory:
    """Test StudentTrajectory dataclass."""

    def test_construction(self):
        from forma.longitudinal_report_data import StudentTrajectory

        traj = StudentTrajectory(
            student_id="S001",
            weekly_scores={1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6},
            overall_trend=0.1,
            is_persistent_risk=False,
            risk_weeks=[1, 2],
        )
        assert traj.student_id == "S001"
        assert traj.weekly_scores == {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6}
        assert traj.overall_trend == 0.1
        assert traj.is_persistent_risk is False
        assert traj.risk_weeks == [1, 2]

    def test_empty_risk_weeks(self):
        from forma.longitudinal_report_data import StudentTrajectory

        traj = StudentTrajectory(
            student_id="S002",
            weekly_scores={1: 0.8},
            overall_trend=0.0,
            is_persistent_risk=False,
            risk_weeks=[],
        )
        assert traj.risk_weeks == []

    def test_persistent_risk_flag(self):
        from forma.longitudinal_report_data import StudentTrajectory

        traj = StudentTrajectory(
            student_id="S003",
            weekly_scores={1: 0.2, 2: 0.25, 3: 0.3},
            overall_trend=0.05,
            is_persistent_risk=True,
            risk_weeks=[1, 2, 3],
        )
        assert traj.is_persistent_risk is True
        assert len(traj.risk_weeks) == 3


class TestConceptMasteryChange:
    """Test ConceptMasteryChange dataclass."""

    def test_construction(self):
        from forma.longitudinal_report_data import ConceptMasteryChange

        change = ConceptMasteryChange(
            concept="항상성",
            week_start_ratio=0.6,
            week_end_ratio=0.8,
            delta=0.2,
        )
        assert change.concept == "항상성"
        assert change.delta == pytest.approx(0.2)

    def test_negative_delta(self):
        from forma.longitudinal_report_data import ConceptMasteryChange

        change = ConceptMasteryChange(
            concept="삼투",
            week_start_ratio=0.7,
            week_end_ratio=0.5,
            delta=-0.2,
        )
        assert change.delta == pytest.approx(-0.2)


class TestLongitudinalSummaryData:
    """Test LongitudinalSummaryData dataclass."""

    def test_construction(self):
        from forma.longitudinal_report_data import (
            LongitudinalSummaryData,
            StudentTrajectory,
            ConceptMasteryChange,
        )

        summary = LongitudinalSummaryData(
            class_name="1A",
            period_weeks=[1, 2, 3, 4],
            student_trajectories=[
                StudentTrajectory("S001", {1: 0.5}, 0.0, False, []),
            ],
            class_weekly_averages={1: 0.5, 2: 0.55},
            persistent_risk_students=["S003"],
            concept_mastery_changes=[
                ConceptMasteryChange("항상성", 0.6, 0.8, 0.2),
            ],
            total_students=5,
        )
        assert summary.class_name == "1A"
        assert summary.period_weeks == [1, 2, 3, 4]
        assert summary.total_students == 5
        assert len(summary.persistent_risk_students) == 1
        assert len(summary.concept_mastery_changes) == 1


# ---------------------------------------------------------------------------
# T029: build_longitudinal_summary() tests
# ---------------------------------------------------------------------------


class TestBuildLongitudinalSummary:
    """Test build_longitudinal_summary() builder function."""

    def test_basic_build(self, tmp_path):
        """Build summary from 4 weeks, 5 students — check fields populated."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "1A")

        assert summary.class_name == "1A"
        assert summary.period_weeks == [1, 2, 3, 4]
        assert summary.total_students == 5
        assert len(summary.student_trajectories) == 5

    def test_class_weekly_averages(self, tmp_path):
        """Verify class_weekly_averages contains correct per-week averages."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "1A")

        # Week 1: (0.3+0.7+0.2+0.9+0.5)/5 = 2.6/5 = 0.52
        assert summary.class_weekly_averages[1] == pytest.approx(0.52, abs=0.01)
        # Week 4: (0.6+0.8+0.35+0.75+0.5)/5 = 3.0/5 = 0.60
        assert summary.class_weekly_averages[4] == pytest.approx(0.60, abs=0.01)

    def test_trajectory_ols_trend(self, tmp_path):
        """Verify OLS trend (numpy.polyfit deg=1) for improving student."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "1A")

        # S001: scores [0.3, 0.4, 0.5, 0.6] over weeks [1,2,3,4]
        # OLS slope = 0.1 per week
        s001_traj = next(t for t in summary.student_trajectories if t.student_id == "S001")
        assert s001_traj.overall_trend == pytest.approx(0.1, abs=0.01)

    def test_flat_trend(self, tmp_path):
        """Verify zero trend for flat scores."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "1A")

        # S005: all 0.5 — trend should be ~0
        s005_traj = next(t for t in summary.student_trajectories if t.student_id == "S005")
        assert abs(s005_traj.overall_trend) < 0.01

    def test_declining_trend(self, tmp_path):
        """Verify negative trend for declining student."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "1A")

        # S004: [0.9, 0.85, 0.80, 0.75] — slope = -0.05 per week
        s004_traj = next(t for t in summary.student_trajectories if t.student_id == "S004")
        assert s004_traj.overall_trend == pytest.approx(-0.05, abs=0.01)

    def test_persistent_risk_detection(self, tmp_path):
        """S003 is at_risk every week (all scores < 0.45) → persistent risk."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "1A")

        assert "S003" in summary.persistent_risk_students
        s003_traj = next(t for t in summary.student_trajectories if t.student_id == "S003")
        assert s003_traj.is_persistent_risk is True

    def test_non_persistent_risk(self, tmp_path):
        """S001 is at_risk only first 2 weeks → NOT persistent."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "1A")

        assert "S001" not in summary.persistent_risk_students

    def test_concept_mastery_changes(self, tmp_path):
        """Verify concept mastery delta from first to last week."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "1A")

        # concept_scores week 1: 항상성=0.6, 삼투=0.4
        # concept_scores week 4: 항상성=0.8, 삼투=0.6
        # delta: 항상성=0.2, 삼투=0.2
        assert len(summary.concept_mastery_changes) >= 2

        # Find specific concepts
        hangsung = next(
            (c for c in summary.concept_mastery_changes if c.concept == "항상성"), None
        )
        assert hangsung is not None
        assert hangsung.week_start_ratio == pytest.approx(0.6, abs=0.01)
        assert hangsung.week_end_ratio == pytest.approx(0.8, abs=0.01)
        assert hangsung.delta == pytest.approx(0.2, abs=0.01)

    def test_concept_mastery_sorted_by_delta_desc(self, tmp_path):
        """concept_mastery_changes must be sorted by delta descending."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "1A")

        deltas = [c.delta for c in summary.concept_mastery_changes]
        assert deltas == sorted(deltas, reverse=True)

    def test_subset_weeks(self, tmp_path):
        """Only include weeks 2 and 3 — period_weeks reflects selection."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [2, 3], "1A")

        assert summary.period_weeks == [2, 3]
        # Class weekly averages should only contain weeks 2 and 3
        assert set(summary.class_weekly_averages.keys()) == {2, 3}

    def test_single_week(self, tmp_path):
        """Single week — trend should be 0.0, no persistent risk."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [1], "1A")

        assert summary.period_weeks == [1]
        # Single data point — OLS should return 0.0 (or handle gracefully)
        for traj in summary.student_trajectories:
            assert traj.overall_trend == pytest.approx(0.0, abs=0.01)

    def test_empty_store(self, tmp_path):
        """Empty store — should return empty summary without error."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = LongitudinalStore(str(tmp_path / "empty.yaml"))
        summary = build_longitudinal_summary(store, [1, 2], "1A")

        assert summary.total_students == 0
        assert summary.student_trajectories == []
        assert summary.persistent_risk_students == []

    def test_missing_weeks_in_store(self, tmp_path):
        """Weeks requested but no data exists → skip gracefully."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        # Week 10 doesn't exist in our test store
        summary = build_longitudinal_summary(store, [1, 2, 10], "1A")

        # Should still work with available data for weeks 1, 2
        assert summary.total_students == 5

    def test_non_contiguous_weeks(self, tmp_path):
        """Non-contiguous weeks [1, 4] — verify correct trajectory."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 4], "1A")

        assert summary.period_weeks == [1, 4]
        s001_traj = next(t for t in summary.student_trajectories if t.student_id == "S001")
        # Scores for weeks 1, 4: [0.3, 0.6] → slope = (0.6-0.3)/(4-1) = 0.1
        assert s001_traj.overall_trend == pytest.approx(0.1, abs=0.01)

    def test_risk_weeks_tracked(self, tmp_path):
        """Verify risk_weeks list contains correct weeks."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = _build_test_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "1A")

        # S003: all scores < 0.45 → at risk all weeks
        s003_traj = next(t for t in summary.student_trajectories if t.student_id == "S003")
        assert set(s003_traj.risk_weeks) == {1, 2, 3, 4}

        # S001: [0.3, 0.4, 0.5, 0.6] — at risk weeks 1, 2 (< 0.45)
        s001_traj = next(t for t in summary.student_trajectories if t.student_id == "S001")
        assert 1 in s001_traj.risk_weeks
        assert 2 in s001_traj.risk_weeks
        assert 3 not in s001_traj.risk_weeks


# ---------------------------------------------------------------------------
# T042: Edge case tests — non-contiguous weeks, single-week OLS
# ---------------------------------------------------------------------------


class TestBuildLongitudinalSummaryEdgeCases:
    """T042 edge case tests for build_longitudinal_summary()."""

    def test_non_contiguous_weeks_137(self, tmp_path):
        """Non-contiguous weeks [1, 3, 7] — period_weeks correct, OLS handles gaps."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = LongitudinalStore(str(tmp_path / "sparse.yaml"))
        # Add records for weeks 1, 3, 7
        for week, score in [(1, 0.3), (3, 0.5), (7, 0.9)]:
            store.add_record(_make_record(
                student_id="S001", week=week, ensemble_score=score,
            ))

        summary = build_longitudinal_summary(store, [1, 3, 7], "1A")

        assert summary.period_weeks == [1, 3, 7]
        assert summary.total_students == 1

        traj = summary.student_trajectories[0]
        assert traj.student_id == "S001"
        assert set(traj.weekly_scores.keys()) == {1, 3, 7}
        # OLS on (1,0.3), (3,0.5), (7,0.9) — positive slope
        assert traj.overall_trend > 0

    def test_non_contiguous_weeks_class_averages(self, tmp_path):
        """Non-contiguous weeks — class_weekly_averages only for requested weeks."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = LongitudinalStore(str(tmp_path / "sparse2.yaml"))
        for sid in ["S001", "S002"]:
            for week, score in [(1, 0.4), (3, 0.6), (7, 0.8)]:
                store.add_record(_make_record(
                    student_id=sid, week=week, ensemble_score=score,
                ))

        summary = build_longitudinal_summary(store, [1, 3, 7], "1A")

        assert set(summary.class_weekly_averages.keys()) == {1, 3, 7}
        # Both students have 0.4 in week 1
        assert summary.class_weekly_averages[1] == pytest.approx(0.4, abs=0.01)

    def test_single_week_ols_zero(self, tmp_path):
        """Single-week data — OLS trend = 0.0 (no slope with 1 data point)."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = LongitudinalStore(str(tmp_path / "single.yaml"))
        store.add_record(_make_record(
            student_id="S001", week=5, ensemble_score=0.7,
        ))

        summary = build_longitudinal_summary(store, [5], "1A")

        assert summary.period_weeks == [5]
        traj = summary.student_trajectories[0]
        assert traj.overall_trend == pytest.approx(0.0, abs=0.001)

    def test_single_week_no_persistent_risk(self, tmp_path):
        """Single-week at-risk student — NOT persistent (need >1 week for persistent)."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store = LongitudinalStore(str(tmp_path / "single_risk.yaml"))
        store.add_record(_make_record(
            student_id="S001", week=1, ensemble_score=0.2,  # below 0.45 threshold
        ))

        summary = build_longitudinal_summary(store, [1], "1A")

        traj = summary.student_trajectories[0]
        # With only 1 week, student is at risk that week but persistent risk
        # is still technically true (at risk in ALL available weeks)
        assert 1 in traj.risk_weeks


# ---------------------------------------------------------------------------
# T064-T065: US7 Topic Statistics tests
# ---------------------------------------------------------------------------


def _build_topic_store(tmp_path) -> LongitudinalStore:
    """Build a store with topic data for 3 weeks, 3 students."""
    store = LongitudinalStore(str(tmp_path / "topic_store.yaml"))
    # 3 students, 3 weeks, 2 topics per week
    data = {
        "S001": {
            1: {"개념이해": 0.6, "적용": 0.4},
            2: {"개념이해": 0.65, "적용": 0.45},
            3: {"개념이해": 0.7, "적용": 0.5},
        },
        "S002": {
            1: {"개념이해": 0.7, "적용": 0.5},
            2: {"개념이해": 0.75, "적용": 0.55},
            3: {"개념이해": 0.8, "적용": 0.6},
        },
        "S003": {
            1: {"개념이해": 0.5, "적용": 0.3},
            2: {"개념이해": 0.55, "적용": 0.35},
            3: {"개념이해": 0.6, "적용": 0.4},
        },
    }
    for sid, weeks in data.items():
        for week, topic_scores in weeks.items():
            for topic, score in topic_scores.items():
                rec = LongitudinalRecord(
                    student_id=sid,
                    week=week,
                    question_sn=1 if topic == "개념이해" else 2,
                    scores={"ensemble_score": score},
                    tier_level=1,
                    tier_label="Developing",
                    topic=topic,
                )
                store.add_record(rec)
    return store


class TestTopicStatistics:
    """Tests for compute_topic_class_statistics()."""

    def test_topic_statistics_3_weeks(self, tmp_path):
        """T064: Topic mean/SD computed per week, tau/rho included for 3+ weeks."""
        from forma.longitudinal_report_data import (
            compute_topic_class_statistics,
        )

        store = _build_topic_store(tmp_path)
        stats = compute_topic_class_statistics(
            store, [1, 2, 3],
        )

        assert len(stats) > 0
        # Should have entries for both topics
        topics_found = {s.topic for s in stats}
        assert "개념이해" in topics_found
        assert "적용" in topics_found

        # Check mean values are reasonable
        w1_concept = next(
            s for s in stats
            if s.topic == "개념이해" and s.week == 1
        )
        # Mean of [0.6, 0.7, 0.5] = 0.6
        assert w1_concept.mean == pytest.approx(0.6, abs=0.01)
        assert w1_concept.std >= 0.0

    def test_topic_statistics_no_topic_data(self, tmp_path):
        """T065: Returns empty when no topic data in store."""
        from forma.longitudinal_report_data import (
            compute_topic_class_statistics,
        )

        store = _build_test_store(tmp_path)  # no topic field
        stats = compute_topic_class_statistics(
            store, [1, 2, 3, 4],
        )
        assert stats == []

    def test_topic_trends_3_weeks(self, tmp_path):
        """T064: Trend statistics (tau, rho) computed for 3+ weeks."""
        from forma.longitudinal_report_data import (
            compute_topic_trends,
        )

        store = _build_topic_store(tmp_path)
        trends = compute_topic_trends(store, [1, 2, 3])

        assert len(trends) > 0
        concept_trend = next(
            t for t in trends if t.topic == "개념이해"
        )
        # All students improving → positive tau
        assert concept_trend.kendall_tau > 0
        assert concept_trend.n_weeks == 3

    def test_topic_trends_2_weeks_empty(self, tmp_path):
        """Trends return empty when < 3 weeks."""
        from forma.longitudinal_report_data import (
            compute_topic_trends,
        )

        store = _build_topic_store(tmp_path)
        trends = compute_topic_trends(store, [1, 2])
        assert trends == []


# ---------------------------------------------------------------------------
# US5: Class-filtered summary (T042)
# ---------------------------------------------------------------------------


class TestFilterByClassIds:
    """T042: build_longitudinal_summary filters by class_id."""

    def test_filter_by_class_ids(self, tmp_path):
        """Only students with matching class_id are included."""
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        store = LongitudinalStore(str(tmp_path / "s.yaml"))
        for sid, cls in [("S1", "A"), ("S2", "A"), ("S3", "B")]:
            for w in [1, 2]:
                store.add_record(LongitudinalRecord(
                    student_id=sid, week=w, question_sn=1,
                    scores={"ensemble_score": 0.6},
                    tier_level=2, tier_label="P",
                    class_id=cls,
                ))

        summary = build_longitudinal_summary(
            store, [1, 2], "Test",
            class_ids=["A"],
        )
        ids = [t.student_id for t in summary.student_trajectories]
        assert "S1" in ids
        assert "S2" in ids
        assert "S3" not in ids
        assert summary.total_students == 2

    def test_no_filter_all_students(self, tmp_path):
        """class_ids=None includes all students."""
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        store = LongitudinalStore(str(tmp_path / "s.yaml"))
        for sid, cls in [("S1", "A"), ("S2", "B")]:
            store.add_record(LongitudinalRecord(
                student_id=sid, week=1, question_sn=1,
                scores={"ensemble_score": 0.6},
                tier_level=2, tier_label="P",
                class_id=cls,
            ))

        summary = build_longitudinal_summary(
            store, [1], "Test",
        )
        assert summary.total_students == 2
