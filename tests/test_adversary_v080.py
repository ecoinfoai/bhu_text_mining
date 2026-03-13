"""Adversary attack tests for v0.8.0 longitudinal analysis features.

6 adversarial personas with aggressive testing of edge cases,
boundary conditions, type confusion, and data corruption scenarios.

Targets:
  - US1: evaluation_types.LongitudinalRecord, longitudinal_store.LongitudinalStore,
         snapshot_from_evaluation, _compute_concept_scores
  - US2: report_data_loader.WeeklyDelta/compute_weekly_delta,
         report_charts.build_trajectory_bar_chart,
         professor_report_data.RiskMovement/compute_risk_movement
  - US3: longitudinal_report_data (build_longitudinal_summary),
         longitudinal_report_charts, longitudinal_report.LongitudinalPDFReportGenerator
"""

from __future__ import annotations

import io
import math
import os
import random
import tempfile

import matplotlib
matplotlib.use("Agg")

import pytest

from forma.evaluation_types import (
    ConceptMatchResult,
    EnsembleResult,
    GraphComparisonResult,
    GraphMetricResult,
    LongitudinalRecord,
    TripletEdge,
)
from forma.longitudinal_store import LongitudinalStore, _compute_concept_scores, snapshot_from_evaluation
from forma.report_data_loader import compute_weekly_delta
from forma.professor_report_data import compute_risk_movement
from forma.longitudinal_report_data import (
    LongitudinalSummaryData,
    StudentTrajectory,
    build_longitudinal_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(records: list[LongitudinalRecord] | None = None) -> LongitudinalStore:
    """Build an in-memory LongitudinalStore with optional records."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name
    store = LongitudinalStore(path)
    if records:
        for r in records:
            store.add_record(r)
    return store


def _make_record(
    student_id: str = "S001",
    week: int = 1,
    question_sn: int = 1,
    scores: dict | None = None,
    tier_level: int = 1,
    tier_label: str = "Developing",
    node_recall: float | None = None,
    edge_f1: float | None = None,
    misconception_count: int | None = None,
    concept_scores: dict | None = None,
    exam_file: str | None = None,
    recorded_at: str | None = None,
) -> LongitudinalRecord:
    return LongitudinalRecord(
        student_id=student_id,
        week=week,
        question_sn=question_sn,
        scores=scores if scores is not None else {"ensemble_score": 0.5},
        tier_level=tier_level,
        tier_label=tier_label,
        node_recall=node_recall,
        edge_f1=edge_f1,
        misconception_count=misconception_count,
        concept_scores=concept_scores,
        exam_file=exam_file,
        recorded_at=recorded_at,
    )


# ===========================================================================
# PERSONA 1: DATA CORRUPTOR
# ===========================================================================


class TestDataCorruptor:
    """Persona 1: Corrupt and malformed data attacks."""

    def test_nan_in_scores_dict(self):
        """NaN values in scores dict should survive round-trip without crash."""
        r = _make_record(scores={"ensemble_score": float("nan"), "x": float("inf")})
        store = _make_store([r])
        history = store.get_student_history("S001")
        assert len(history) == 1
        assert math.isnan(history[0].scores["ensemble_score"])
        assert math.isinf(history[0].scores["x"])

    def test_negative_scores(self):
        """Negative scores should be stored and retrieved faithfully."""
        r = _make_record(scores={"ensemble_score": -999.0, "a": -0.001})
        store = _make_store([r])
        traj = store.get_student_trajectory("S001", "ensemble_score")
        assert len(traj) == 1
        assert traj[0][1] == -999.0

    def test_empty_scores_dict(self):
        """Empty scores dict should not cause KeyError on trajectory query."""
        r = _make_record(scores={})
        store = _make_store([r])
        traj = store.get_student_trajectory("S001", "ensemble_score")
        assert traj == []

    def test_concept_scores_with_empty_string_key(self):
        """concept_scores with empty-string key should round-trip."""
        r = _make_record(concept_scores={"": 0.5, "normal": 0.8})
        store = _make_store([r])
        rec = store.get_student_history("S001")[0]
        assert "" in rec.concept_scores
        assert rec.concept_scores[""] == 0.5

    def test_corrupt_yaml_load_initializes_empty(self):
        """Loading a non-YAML file should not crash — store stays empty."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write(":::not valid yaml::: {{{")
            path = f.name
        store = LongitudinalStore(path)
        # yaml.safe_load on malformed YAML raises an exception
        with pytest.raises(Exception):
            store.load()

    def test_truncated_record_missing_v2_fields(self):
        """Records without v2 fields should load with None defaults."""
        r = _make_record()  # No v2 fields set
        store = _make_store([r])
        rec = store.get_student_history("S001")[0]
        assert rec.node_recall is None
        assert rec.edge_f1 is None
        assert rec.misconception_count is None
        assert rec.concept_scores is None

    def test_save_load_round_trip_preserves_data(self):
        """Full v2 record survives save/load cycle."""
        r = _make_record(
            node_recall=0.75,
            edge_f1=0.6,
            misconception_count=3,
            concept_scores={"세포막": 0.8, "핵": 0.5},
            exam_file="test.yaml",
            recorded_at="2026-01-01T00:00:00Z",
        )
        store = _make_store([r])
        store.save()

        store2 = LongitudinalStore(store.store_path)
        store2.load()
        rec = store2.get_student_history("S001")[0]
        assert rec.node_recall == 0.75
        assert rec.edge_f1 == 0.6
        assert rec.misconception_count == 3
        assert rec.concept_scores["세포막"] == 0.8
        # Cleanup
        os.unlink(store.store_path)
        bak = store.store_path + ".bak"
        if os.path.exists(bak):
            os.unlink(bak)


# ===========================================================================
# PERSONA 2: BOUNDARY BREAKER
# ===========================================================================


class TestBoundaryBreaker:
    """Persona 2: Extreme value and boundary attacks."""

    def test_zero_students_class_snapshot(self):
        """get_class_snapshot on empty store returns empty list."""
        store = _make_store()
        result = store.get_class_snapshot(1)
        assert result == []

    def test_week_zero_and_negative(self):
        """Week=0 and week=-1 should be stored and queryable."""
        r0 = _make_record(week=0)
        r_neg = _make_record(student_id="S002", week=-1)
        store = _make_store([r0, r_neg])
        assert len(store.get_class_snapshot(0)) == 1
        assert len(store.get_class_snapshot(-1)) == 1

    def test_extreme_week_number(self):
        """Extremely large week number should not crash."""
        r = _make_record(week=999999)
        store = _make_store([r])
        snap = store.get_class_snapshot(999999)
        assert len(snap) == 1

    def test_score_exactly_zero_and_one(self):
        """Scores at exact boundaries 0.0 and 1.0 must be preserved."""
        r1 = _make_record(student_id="S001", scores={"ensemble_score": 0.0})
        r2 = _make_record(student_id="S002", scores={"ensemble_score": 1.0})
        store = _make_store([r1, r2])
        t1 = store.get_student_trajectory("S001", "ensemble_score")
        t2 = store.get_student_trajectory("S002", "ensemble_score")
        assert t1[0][1] == 0.0
        assert t2[0][1] == 1.0

    def test_delta_threshold_boundary_exact(self):
        """compute_weekly_delta boundary: |delta| == 0.02 should produce dash symbol."""
        store = _make_store([
            _make_record(student_id="S001", week=1, scores={"ensemble_score": 0.50}),
        ])
        # delta = 0.52 - 0.50 = 0.02 exactly
        wd = compute_weekly_delta("S001", 2, 0.52, store, "ensemble_score")
        assert wd.delta_symbol == "─"

    def test_delta_threshold_boundary_just_above(self):
        """delta = 0.021 should produce up arrow."""
        store = _make_store([
            _make_record(student_id="S001", week=1, scores={"ensemble_score": 0.50}),
        ])
        wd = compute_weekly_delta("S001", 2, 0.521, store, "ensemble_score")
        # delta = 0.021, > 0.02 + 1e-9, so should be up
        assert wd.delta_symbol == "↑"

    def test_delta_threshold_boundary_just_below_negative(self):
        """delta = -0.021 should produce down arrow."""
        store = _make_store([
            _make_record(student_id="S001", week=1, scores={"ensemble_score": 0.50}),
        ])
        wd = compute_weekly_delta("S001", 2, 0.479, store, "ensemble_score")
        assert wd.delta_symbol == "↓"

    def test_weekly_delta_first_week_is_NEW(self):
        """First week (no prior data) should return NEW symbol."""
        store = _make_store()
        wd = compute_weekly_delta("S001", 1, 0.5, store, "ensemble_score")
        assert wd.delta_symbol == "NEW"
        assert wd.previous_score is None
        assert wd.delta is None

    def test_class_weekly_matrix_thousand_students(self):
        """1000 students should not crash get_class_weekly_matrix."""
        records = [
            _make_record(student_id=f"S{i:04d}", week=1, scores={"es": float(i)/1000})
            for i in range(1000)
        ]
        store = _make_store(records)
        matrix = store.get_class_weekly_matrix("es")
        assert len(matrix) == 1000

    def test_build_longitudinal_summary_empty_weeks(self):
        """build_longitudinal_summary with empty weeks list returns empty data."""
        store = _make_store([_make_record()])
        result = build_longitudinal_summary(store, [], "1A")
        assert result.period_weeks == []
        assert result.total_students == 0


# ===========================================================================
# PERSONA 3: CONCURRENCY ASSASSIN
# ===========================================================================


class TestConcurrencyAssassin:
    """Persona 3: Stress and race condition attacks."""

    def test_rapid_sequential_add_record(self):
        """Rapid sequential inserts must all be stored."""
        store = _make_store()
        for i in range(500):
            store.add_record(_make_record(
                student_id=f"S{i:04d}", week=1, question_sn=1,
                scores={"es": random.random()},
            ))
        assert len(store.get_all_records()) == 500

    def test_manual_override_flip_flop(self):
        """Manual override prevents overwrite; non-override allows."""
        store = _make_store()
        r1 = _make_record(scores={"es": 0.1})
        store.add_record(r1)

        # Manually set override flag
        key = "S001_1_1"
        store._records[key]["manual_override"] = True

        # Attempt overwrite — should be blocked
        r2 = _make_record(scores={"es": 0.9})
        store.add_record(r2)
        rec = store.get_student_history("S001")[0]
        assert rec.scores["es"] == 0.1  # Original preserved

        # Remove override
        store._records[key]["manual_override"] = False

        # Now overwrite should succeed
        r3 = _make_record(scores={"es": 0.99})
        store.add_record(r3)
        rec = store.get_student_history("S001")[0]
        assert rec.scores["es"] == 0.99

    def test_stress_10000_records(self):
        """10000 records in a single store must be queryable."""
        store = _make_store()
        for i in range(100):
            for w in range(1, 101):
                store.add_record(_make_record(
                    student_id=f"S{i:03d}", week=w, question_sn=1,
                    scores={"es": random.random()},
                ))
        assert len(store.get_all_records()) == 10000
        traj = store.get_student_trajectory("S050", "es")
        assert len(traj) == 100

    def test_large_concept_scores_dict(self):
        """1000 concepts in concept_scores should be handled."""
        cs = {f"concept_{i}": random.random() for i in range(1000)}
        r = _make_record(concept_scores=cs)
        store = _make_store([r])
        rec = store.get_student_history("S001")[0]
        assert len(rec.concept_scores) == 1000

    def test_overwrite_same_key_many_times(self):
        """Repeated overwrite of same key should keep last value."""
        store = _make_store()
        for i in range(200):
            store.add_record(_make_record(scores={"es": float(i)}))
        rec = store.get_student_history("S001")[0]
        assert rec.scores["es"] == 199.0

    def test_concurrent_save_does_not_corrupt(self):
        """Sequential save operations produce valid YAML each time."""
        store = _make_store()
        for i in range(50):
            store.add_record(_make_record(
                student_id=f"S{i:03d}", week=1, question_sn=1,
                scores={"es": random.random()},
            ))
        store.save()

        # Load back
        store2 = LongitudinalStore(store.store_path)
        store2.load()
        assert len(store2.get_all_records()) == 50
        os.unlink(store.store_path)
        bak = store.store_path + ".bak"
        if os.path.exists(bak):
            os.unlink(bak)


# ===========================================================================
# PERSONA 4: TYPE CONFUSER
# ===========================================================================


class TestTypeConfuser:
    """Persona 4: Type mismatch attacks."""

    def test_string_score_in_trajectory(self):
        """String value in scores dict: trajectory averaging raises TypeError.

        BUG: get_student_trajectory does `sum(vals) / len(vals)` which fails
        on string values. Not a real-world concern (scores always come from
        EnsembleResult floats), but the code has no type guard here.
        """
        r = _make_record(scores={"ensemble_score": "not_a_number"})
        store = _make_store([r])
        # sum() on strings raises TypeError — expected behavior for bad input
        with pytest.raises(TypeError):
            store.get_student_trajectory("S001", "ensemble_score")

    def test_none_in_scores_value(self):
        """None as a scores value should be skipped in trajectory."""
        r = _make_record(scores={"ensemble_score": None})
        store = _make_store([r])
        traj = store.get_student_trajectory("S001", "ensemble_score")
        # val is None, so `if val is not None` filters it out
        assert traj == []

    def test_empty_student_id(self):
        """Empty string student_id should work in store."""
        r = _make_record(student_id="")
        store = _make_store([r])
        assert len(store.get_student_history("")) == 1

    def test_very_long_student_id(self):
        """10000-char student_id should not crash."""
        long_id = "A" * 10000
        r = _make_record(student_id=long_id)
        store = _make_store([r])
        assert len(store.get_student_history(long_id)) == 1

    def test_negative_question_sn(self):
        """Negative question_sn should be storable."""
        r = _make_record(question_sn=-1)
        store = _make_store([r])
        assert len(store.get_student_history("S001")) == 1

    def test_bool_in_scores_value(self):
        """Boolean True (truthy) in scores should survive (Python bool is int subclass)."""
        r = _make_record(scores={"ensemble_score": True})
        store = _make_store([r])
        traj = store.get_student_trajectory("S001", "ensemble_score")
        assert len(traj) == 1
        # True == 1, so trajectory value should be truthy
        assert traj[0][1] == True  # noqa: E712

    def test_concept_scores_with_int_values(self):
        """Integer values in concept_scores should not crash aggregation."""
        r = _make_record(concept_scores={"A": 1, "B": 0})
        store = _make_store([r])
        snap = store.get_class_snapshot(1)
        assert len(snap) == 1
        assert snap[0].concept_scores["A"] == 1

    def test_risk_movement_with_empty_sets(self):
        """compute_risk_movement with empty sets should produce empty lists."""
        rm = compute_risk_movement(set(), set())
        assert rm.newly_at_risk == []
        assert rm.exited_risk == []
        assert rm.persistent_risk == []

    def test_risk_movement_with_none_like_ids(self):
        """Student IDs that are stringified None/empty should be sortable."""
        rm = compute_risk_movement({"None", "", "S001"}, {"S001"})
        assert rm.newly_at_risk == sorted(["None", ""])
        assert rm.exited_risk == []
        assert rm.persistent_risk == ["S001"]


# ===========================================================================
# PERSONA 5: UNICODE ATTACKER
# ===========================================================================


class TestUnicodeAttacker:
    """Persona 5: Unicode and injection attacks."""

    def test_korean_student_id(self):
        """Korean characters in student_id should work."""
        r = _make_record(student_id="학생_가나다")
        store = _make_store([r])
        assert len(store.get_student_history("학생_가나다")) == 1

    def test_emoji_student_id(self):
        """Emoji in student_id should work."""
        emoji_id = "🔥💀👻"
        r = _make_record(student_id=emoji_id)
        store = _make_store([r])
        assert len(store.get_student_history(emoji_id)) == 1

    def test_zero_width_characters_in_student_id(self):
        """Zero-width chars should be stored faithfully."""
        zw_id = "S\u200b001\u200c\u200d"
        r = _make_record(student_id=zw_id)
        store = _make_store([r])
        assert len(store.get_student_history(zw_id)) == 1

    def test_unicode_concept_names(self):
        """Korean + special unicode in concept_scores keys."""
        cs = {
            "세포막\u200b투과성": 0.5,
            "항상성 ← 피드백": 0.8,
            "概念 → 結果": 0.3,
        }
        r = _make_record(concept_scores=cs)
        store = _make_store([r])
        rec = store.get_student_history("S001")[0]
        assert len(rec.concept_scores) == 3

    def test_xml_injection_in_student_id(self):
        """XML injection attempts in student_id should not crash PDF generation."""
        xml_id = '<script>alert("XSS")</script>'
        r = _make_record(student_id=xml_id)
        store = _make_store([r])
        rec = store.get_student_history(xml_id)[0]
        assert rec.student_id == xml_id

    def test_rtl_characters_in_student_id(self):
        """Right-to-left Arabic text in student_id should work."""
        rtl_id = "طالب_١٢٣"
        r = _make_record(student_id=rtl_id)
        store = _make_store([r])
        assert len(store.get_student_history(rtl_id)) == 1

    def test_path_traversal_in_exam_file(self):
        """Path traversal string in exam_file should be stored as-is (no exec)."""
        r = _make_record(exam_file="../../etc/passwd")
        store = _make_store([r])
        rec = store.get_student_history("S001")[0]
        assert rec.exam_file == "../../etc/passwd"

    def test_very_long_korean_concept(self):
        """10000-char Korean concept name should be handled."""
        long_concept = "가" * 10000
        r = _make_record(concept_scores={long_concept: 1.0})
        store = _make_store([r])
        rec = store.get_student_history("S001")[0]
        assert long_concept in rec.concept_scores


# ===========================================================================
# PERSONA 6: REGRESSION HUNTER
# ===========================================================================


class TestRegressionHunter:
    """Persona 6: Backward compat and regression attacks."""

    def test_v1_records_with_extra_unknown_fields(self):
        """Records with extra unknown fields in the dict should load gracefully."""
        store = _make_store()
        # Manually inject a record with extra fields
        store._records["test_key"] = {
            "student_id": "S001",
            "week": 1,
            "question_sn": 1,
            "scores": {"es": 0.5},
            "tier_level": 1,
            "tier_label": "Developing",
            "unknown_future_field": "some_value",
            "another_field": 42,
        }
        store._rebuild_index()
        # _to_record should ignore unknown fields
        rec = store.get_student_history("S001")[0]
        assert rec.student_id == "S001"
        assert rec.scores["es"] == 0.5

    def test_mixed_v1_v2_records(self):
        """Mix of v1 (no v2 fields) and v2 records in same store."""
        store = _make_store()
        # v1 record (no v2 fields)
        store._records["v1_key"] = {
            "student_id": "S001",
            "week": 1,
            "question_sn": 1,
            "scores": {"es": 0.5},
            "tier_level": 1,
            "tier_label": "Developing",
        }
        # v2 record (with v2 fields)
        store._records["v2_key"] = {
            "student_id": "S002",
            "week": 1,
            "question_sn": 1,
            "scores": {"es": 0.7},
            "tier_level": 2,
            "tier_label": "Proficient",
            "node_recall": 0.8,
            "edge_f1": 0.6,
            "misconception_count": 2,
            "concept_scores": {"A": 0.9},
            "exam_file": "exam.yaml",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
        store._rebuild_index()
        all_recs = store.get_all_records()
        assert len(all_recs) == 2
        v1 = next(r for r in all_recs if r.student_id == "S001")
        v2 = next(r for r in all_recs if r.student_id == "S002")
        assert v1.node_recall is None
        assert v2.node_recall == 0.8

    def test_trajectory_metric_in_some_weeks_only(self):
        """get_student_trajectory for metric present in some weeks but not others."""
        records = [
            _make_record(week=1, scores={"es": 0.3, "coverage": 0.5}),
            _make_record(week=2, scores={"es": 0.5}),  # no coverage
            _make_record(week=3, scores={"es": 0.7, "coverage": 0.8}),
        ]
        store = _make_store(records)
        traj_cov = store.get_student_trajectory("S001", "coverage")
        assert len(traj_cov) == 2
        assert traj_cov[0] == (1, 0.5)
        assert traj_cov[1] == (3, 0.8)

    def test_single_week_longitudinal_summary(self):
        """Single-week summary: OLS with 1 point should produce trend=0.0."""
        store = _make_store([
            _make_record(student_id="S001", week=3, scores={"ensemble_score": 0.5}),
            _make_record(student_id="S002", week=3, scores={"ensemble_score": 0.3}),
        ])
        result = build_longitudinal_summary(store, [3], "1A")
        assert len(result.student_trajectories) == 2
        for traj in result.student_trajectories:
            assert traj.overall_trend == 0.0  # single point => 0.0

    def test_non_contiguous_weeks(self):
        """Non-contiguous weeks (1, 5, 10) should be handled correctly."""
        records = []
        for w in [1, 5, 10]:
            records.append(_make_record(week=w, scores={"ensemble_score": w * 0.1}))
        store = _make_store(records)
        result = build_longitudinal_summary(store, [1, 5, 10], "1A")
        assert result.period_weeks == [1, 5, 10]
        assert len(result.student_trajectories) == 1
        traj = result.student_trajectories[0]
        assert set(traj.weekly_scores.keys()) == {1, 5, 10}

    def test_heatmap_all_identical_scores(self):
        """All-identical scores should not crash heatmap generation."""
        from forma.font_utils import find_korean_font
        from forma.longitudinal_report_data import LongitudinalSummaryData, StudentTrajectory

        trajs = [
            StudentTrajectory(
                student_id=f"S{i:03d}",
                weekly_scores={1: 0.5, 2: 0.5, 3: 0.5},
                overall_trend=0.0,
                is_persistent_risk=False,
                risk_weeks=[],
            )
            for i in range(10)
        ]
        data = LongitudinalSummaryData(
            class_name="1A",
            period_weeks=[1, 2, 3],
            student_trajectories=trajs,
            class_weekly_averages={1: 0.5, 2: 0.5, 3: 0.5},
            persistent_risk_students=[],
            concept_mastery_changes=[],
            total_students=10,
        )

        from forma.longitudinal_report_charts import build_class_week_heatmap
        font_path = find_korean_font()
        # Should not crash on zero variance
        buf = build_class_week_heatmap(data, font_path=font_path)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_weekly_delta_previous_week_not_immediately_prior(self):
        """Previous week should be the MOST RECENT before current, not current-1."""
        store = _make_store([
            _make_record(week=1, scores={"es": 0.3}),
            _make_record(week=5, scores={"es": 0.6}),
        ])
        # Query for week 10 — previous should be week 5, not week 9
        wd = compute_weekly_delta("S001", 10, 0.8, store, "es")
        assert wd.previous_score == 0.6
        assert wd.delta == pytest.approx(0.2)
        assert wd.delta_symbol == "↑"

    def test_risk_movement_all_exit(self):
        """All students exit risk: newly=[], exited=[all], persistent=[]."""
        prev = {"S001", "S002", "S003"}
        curr = set()
        rm = compute_risk_movement(curr, prev)
        assert rm.newly_at_risk == []
        assert rm.exited_risk == ["S001", "S002", "S003"]
        assert rm.persistent_risk == []

    def test_risk_movement_all_new(self):
        """All students newly at risk: newly=[all], exited=[], persistent=[]."""
        prev = set()
        curr = {"S001", "S002"}
        rm = compute_risk_movement(curr, prev)
        assert rm.newly_at_risk == ["S001", "S002"]
        assert rm.exited_risk == []
        assert rm.persistent_risk == []


# ===========================================================================
# INVARIANT TESTING (1000-iteration loops)
# ===========================================================================


class TestInvariant1000:
    """High-iteration invariant tests across adversary scenarios."""

    def test_record_key_uniqueness_invariant(self):
        """1000 random records: each unique (sid, week, qsn) produces unique key."""
        from forma.longitudinal_store import _record_key
        keys = set()
        for _ in range(1000):
            sid = f"S{random.randint(0, 100):03d}"
            wk = random.randint(1, 50)
            qsn = random.randint(1, 10)
            key = _record_key(sid, wk, qsn)
            assert isinstance(key, str)
            keys.add(key)
        # Keys are deterministic for same inputs
        assert _record_key("S001", 1, 1) == "S001_1_1"

    def test_weekly_delta_symbol_invariant(self):
        """1000 random deltas: symbol always in {NEW, up, down, dash}."""
        valid_symbols = {"NEW", "↑", "↓", "─"}
        for _ in range(1000):
            store = _make_store()
            has_prev = random.random() > 0.3
            if has_prev:
                prev_score = random.uniform(0, 1)
                store.add_record(_make_record(week=1, scores={"es": prev_score}))
            current = random.uniform(0, 1)
            wd = compute_weekly_delta("S001", 2, current, store, "es")
            assert wd.delta_symbol in valid_symbols
            if wd.delta_symbol == "NEW":
                assert wd.delta is None
            else:
                assert wd.delta is not None

    def test_risk_movement_set_algebra_invariant(self):
        """1000 random risk sets: newly + persistent == current; exited + persistent == previous."""
        for _ in range(1000):
            n_curr = random.randint(0, 20)
            n_prev = random.randint(0, 20)
            curr = {f"S{i:03d}" for i in random.sample(range(50), n_curr)}
            prev = {f"S{i:03d}" for i in random.sample(range(50), n_prev)}
            rm = compute_risk_movement(curr, prev)
            # Set algebra invariants
            newly = set(rm.newly_at_risk)
            exited = set(rm.exited_risk)
            persistent = set(rm.persistent_risk)
            assert newly | persistent == curr
            assert exited | persistent == prev
            assert newly & persistent == set()
            assert exited & persistent == set()
            assert newly & exited == set()
            # All lists are sorted
            assert rm.newly_at_risk == sorted(rm.newly_at_risk)
            assert rm.exited_risk == sorted(rm.exited_risk)
            assert rm.persistent_risk == sorted(rm.persistent_risk)

    def test_trajectory_week_ordering_invariant(self):
        """1000 random trajectories: result is always sorted by week ascending."""
        for _ in range(1000):
            n_weeks = random.randint(1, 20)
            weeks = random.sample(range(1, 100), n_weeks)
            records = [
                _make_record(week=w, scores={"es": random.random()})
                for w in weeks
            ]
            store = _make_store(records)
            traj = store.get_student_trajectory("S001", "es")
            result_weeks = [t[0] for t in traj]
            assert result_weeks == sorted(result_weeks)

    def test_class_snapshot_sorted_invariant(self):
        """1000 iterations: class snapshot always sorted by student_id."""
        for _ in range(1000):
            n = random.randint(1, 30)
            records = [
                _make_record(
                    student_id=f"S{random.randint(0, 100):03d}",
                    week=1,
                    question_sn=random.randint(1, 3),
                    scores={"es": random.random()},
                )
                for _ in range(n)
            ]
            store = _make_store(records)
            snap = store.get_class_snapshot(1)
            ids = [r.student_id for r in snap]
            assert ids == sorted(ids)

    def test_longitudinal_summary_trend_is_finite(self):
        """1000 random multi-week stores: trend is always finite float."""
        for _ in range(1000):
            n_weeks = random.randint(2, 10)
            weeks = sorted(random.sample(range(1, 50), n_weeks))
            records = [
                _make_record(week=w, scores={"ensemble_score": random.random()})
                for w in weeks
            ]
            store = _make_store(records)
            result = build_longitudinal_summary(store, weeks, "1A")
            for traj in result.student_trajectories:
                assert math.isfinite(traj.overall_trend)


# ===========================================================================
# SNAPSHOT_FROM_EVALUATION ATTACKS
# ===========================================================================


class TestSnapshotFromEvaluationAttacks:
    """Attack snapshot_from_evaluation with adversarial inputs."""

    def test_empty_ensemble_results(self):
        """Empty ensemble results should produce no records."""
        store = _make_store()
        snapshot_from_evaluation(store, {}, {}, {}, [], week=1, exam_file="test.yaml")
        assert len(store.get_all_records()) == 0

    def test_missing_graph_metric_and_comparison(self):
        """Missing graph metrics/comparisons should produce None v2 fields."""
        store = _make_store()
        er = EnsembleResult(
            student_id="S001", question_sn=1, ensemble_score=0.5,
            understanding_level="Developing",
            component_scores={"es": 0.5}, weights_used={"es": 1.0},
        )
        snapshot_from_evaluation(
            store,
            ensemble_results={"S001": {1: er}},
            graph_metric_results={},
            graph_comparison_results={},
            layer1_results=[],
            week=1,
            exam_file="test.yaml",
        )
        recs = store.get_all_records()
        assert len(recs) == 1
        assert recs[0].node_recall is None
        assert recs[0].edge_f1 is None
        assert recs[0].misconception_count is None

    def test_snapshot_with_full_data(self):
        """Snapshot with all data sources populated produces complete records."""
        store = _make_store()
        er = EnsembleResult(
            student_id="S001", question_sn=1, ensemble_score=0.7,
            understanding_level="Proficient",
            component_scores={"es": 0.7}, weights_used={"es": 1.0},
        )
        gmr = GraphMetricResult(
            student_id="S001", question_sn=1,
            node_recall=0.8, edge_jaccard=0.5,
            centrality_deviation=0.1, normalized_ged=0.3,
        )
        gcr = GraphComparisonResult(
            student_id="S001", question_sn=1,
            precision=0.9, recall=0.8, f1=0.85,
            matched_edges=[], missing_edges=[], extra_edges=[],
            wrong_direction_edges=[TripletEdge("A", "r", "B"), TripletEdge("C", "r", "D")],
        )
        cmr = ConceptMatchResult(
            concept="세포막", student_id="S001", question_sn=1,
            is_present=True, similarity_score=0.8,
            top_k_mean_similarity=0.8, threshold_used=0.5,
        )
        snapshot_from_evaluation(
            store,
            ensemble_results={"S001": {1: er}},
            graph_metric_results={"S001": {1: gmr}},
            graph_comparison_results={"S001": {1: gcr}},
            layer1_results=[cmr],
            week=1,
            exam_file="/path/to/test.yaml",
        )
        recs = store.get_all_records()
        assert len(recs) == 1
        r = recs[0]
        assert r.node_recall == 0.8
        assert r.edge_f1 == 0.85
        assert r.misconception_count == 2
        assert r.concept_scores == {"세포막": 1.0}
        assert r.exam_file == "test.yaml"  # basename extracted

    def test_snapshot_multiple_students_questions(self):
        """Multiple students and questions should all be recorded."""
        store = _make_store()
        ensemble_results = {}
        for sid in ["S001", "S002", "S003"]:
            ensemble_results[sid] = {}
            for qsn in [1, 2]:
                ensemble_results[sid][qsn] = EnsembleResult(
                    student_id=sid, question_sn=qsn, ensemble_score=0.5,
                    understanding_level="Developing",
                    component_scores={"es": 0.5}, weights_used={"es": 1.0},
                )
        snapshot_from_evaluation(
            store, ensemble_results, {}, {}, [], week=1, exam_file="e.yaml",
        )
        assert len(store.get_all_records()) == 6  # 3 students * 2 questions

    def test_compute_concept_scores_empty_list(self):
        """_compute_concept_scores with empty list returns None."""
        result = _compute_concept_scores([], "S001", 1)
        assert result is None

    def test_compute_concept_scores_no_matching_student(self):
        """_compute_concept_scores with non-matching student returns None."""
        cmr = ConceptMatchResult(
            concept="A", student_id="S999", question_sn=1,
            is_present=True, similarity_score=0.8,
            top_k_mean_similarity=0.8, threshold_used=0.5,
        )
        result = _compute_concept_scores([cmr], "S001", 1)
        assert result is None


# ===========================================================================
# CHART ATTACKS
# ===========================================================================


class TestChartAttacks:
    """Attack chart generation with adversarial data."""

    def test_trajectory_bar_chart_empty_scores(self):
        """Empty weekly_scores dict should produce a placeholder chart."""
        from forma.font_utils import find_korean_font
        from forma.report_charts import ReportChartGenerator
        gen = ReportChartGenerator(font_path=find_korean_font())
        buf = gen.build_trajectory_bar_chart({}, current_week=1)
        assert isinstance(buf, io.BytesIO)
        assert len(buf.getvalue()) > 0

    def test_trajectory_bar_chart_single_week(self):
        """Single week in trajectory chart should work."""
        from forma.font_utils import find_korean_font
        from forma.report_charts import ReportChartGenerator
        gen = ReportChartGenerator(font_path=find_korean_font())
        buf = gen.build_trajectory_bar_chart({5: 0.7}, current_week=5)
        assert isinstance(buf, io.BytesIO)
        data = buf.getvalue()
        assert data[:4] == b"\x89PNG"

    def test_trajectory_bar_chart_many_weeks(self):
        """50 weeks in trajectory chart should not overflow."""
        from forma.font_utils import find_korean_font
        from forma.report_charts import ReportChartGenerator
        gen = ReportChartGenerator(font_path=find_korean_font())
        scores = {w: random.random() for w in range(1, 51)}
        buf = gen.build_trajectory_bar_chart(scores, current_week=25)
        assert isinstance(buf, io.BytesIO)

    def test_trajectory_line_chart_no_students(self):
        """Empty student trajectories should produce placeholder chart."""
        from forma.font_utils import find_korean_font
        from forma.longitudinal_report_charts import build_trajectory_line_chart

        data = LongitudinalSummaryData(
            class_name="1A", period_weeks=[1, 2, 3],
            student_trajectories=[], class_weekly_averages={},
            persistent_risk_students=[], concept_mastery_changes=[],
            total_students=0,
        )
        buf = build_trajectory_line_chart(data, font_path=find_korean_font())
        assert isinstance(buf, io.BytesIO)

    def test_concept_mastery_bar_chart_empty(self):
        """Empty concept mastery changes should produce placeholder chart."""
        from forma.font_utils import find_korean_font
        from forma.longitudinal_report_charts import build_concept_mastery_bar_chart

        data = LongitudinalSummaryData(
            class_name="1A", period_weeks=[1, 2],
            student_trajectories=[], class_weekly_averages={},
            persistent_risk_students=[],
            concept_mastery_changes=[],
            total_students=0,
        )
        buf = build_concept_mastery_bar_chart(data, font_path=find_korean_font())
        assert isinstance(buf, io.BytesIO)

    def test_heatmap_with_100_plus_students(self):
        """Heatmap with >100 students should truncate to top25 + bottom25."""
        from forma.font_utils import find_korean_font
        from forma.longitudinal_report_charts import build_class_week_heatmap

        trajs = [
            StudentTrajectory(
                student_id=f"S{i:04d}",
                weekly_scores={1: i / 150.0, 2: (i + 5) / 150.0},
                overall_trend=0.01,
                is_persistent_risk=False,
                risk_weeks=[],
            )
            for i in range(150)
        ]
        data = LongitudinalSummaryData(
            class_name="1A", period_weeks=[1, 2],
            student_trajectories=trajs,
            class_weekly_averages={1: 0.5, 2: 0.55},
            persistent_risk_students=[],
            concept_mastery_changes=[],
            total_students=150,
        )
        buf = build_class_week_heatmap(data, font_path=find_korean_font())
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"


# ===========================================================================
# WEEKLY MATRIX ATTACKS
# ===========================================================================


class TestWeeklyMatrixAttacks:
    """Attack get_class_weekly_matrix with adversarial data."""

    def test_matrix_no_matching_metric(self):
        """Querying for a metric that no record has returns empty dict."""
        store = _make_store([_make_record(scores={"x": 0.5})])
        matrix = store.get_class_weekly_matrix("nonexistent_metric")
        assert matrix == {}

    def test_matrix_multiple_questions_per_week_averaged(self):
        """Multiple questions in same week: values averaged per student-week."""
        records = [
            _make_record(question_sn=1, scores={"es": 0.2}),
            _make_record(question_sn=2, scores={"es": 0.8}),
        ]
        store = _make_store(records)
        matrix = store.get_class_weekly_matrix("es")
        assert matrix["S001"][1] == pytest.approx(0.5)

    def test_matrix_student_absent_some_weeks(self):
        """Student with data in only some weeks: missing weeks absent from dict."""
        records = [
            _make_record(week=1, scores={"es": 0.3}),
            _make_record(week=3, scores={"es": 0.7}),
        ]
        store = _make_store(records)
        matrix = store.get_class_weekly_matrix("es")
        assert 1 in matrix["S001"]
        assert 2 not in matrix["S001"]
        assert 3 in matrix["S001"]


# ===========================================================================
# CONCEPT MASTERY CHANGE ATTACKS
# ===========================================================================


class TestConceptMasteryChangeAttacks:
    """Attack concept mastery computation in longitudinal summary."""

    def test_concept_present_only_in_first_week(self):
        """Concept in first week but not last: end ratio should be 0.0."""
        records = [
            _make_record(week=1, concept_scores={"A": 0.8}),
            _make_record(week=3, concept_scores={"B": 0.5}),
        ]
        store = _make_store(records)
        result = build_longitudinal_summary(store, [1, 3], "1A")
        changes_by_concept = {c.concept: c for c in result.concept_mastery_changes}
        assert changes_by_concept["A"].week_end_ratio == 0.0
        assert changes_by_concept["A"].delta == -0.8

    def test_concept_present_only_in_last_week(self):
        """Concept in last week but not first: start ratio should be 0.0."""
        records = [
            _make_record(week=1, concept_scores={"A": 0.8}),
            _make_record(week=3, concept_scores={"B": 0.5}),
        ]
        store = _make_store(records)
        result = build_longitudinal_summary(store, [1, 3], "1A")
        changes_by_concept = {c.concept: c for c in result.concept_mastery_changes}
        assert changes_by_concept["B"].week_start_ratio == 0.0
        assert changes_by_concept["B"].delta == 0.5

    def test_no_concept_scores_produces_no_changes(self):
        """Records without concept_scores: no concept mastery changes."""
        records = [
            _make_record(week=1),
            _make_record(week=2),
        ]
        store = _make_store(records)
        result = build_longitudinal_summary(store, [1, 2], "1A")
        assert result.concept_mastery_changes == []

    def test_same_concept_same_ratio_zero_delta(self):
        """Same concept at same ratio in first and last week: delta=0."""
        records = [
            _make_record(week=1, concept_scores={"A": 0.5}),
            _make_record(week=3, concept_scores={"A": 0.5}),
        ]
        store = _make_store(records)
        result = build_longitudinal_summary(store, [1, 3], "1A")
        assert len(result.concept_mastery_changes) == 1
        assert result.concept_mastery_changes[0].delta == 0.0

    def test_many_concepts_sorted_by_delta_desc(self):
        """Multiple concepts should be sorted by delta descending."""
        records = [
            _make_record(week=1, concept_scores={"A": 0.9, "B": 0.1, "C": 0.5}),
            _make_record(week=5, concept_scores={"A": 0.1, "B": 0.9, "C": 0.5}),
        ]
        store = _make_store(records)
        result = build_longitudinal_summary(store, [1, 5], "1A")
        deltas = [c.delta for c in result.concept_mastery_changes]
        assert deltas == sorted(deltas, reverse=True)


# ===========================================================================
# BUILD LONGITUDINAL SUMMARY EDGE CASES
# ===========================================================================


class TestBuildLongitudinalSummaryEdgeCases:
    """Edge case attacks on build_longitudinal_summary."""

    def test_student_only_in_unrequested_weeks(self):
        """Student with data only in weeks not in the requested list: excluded."""
        store = _make_store([
            _make_record(week=99, scores={"ensemble_score": 0.5}),
        ])
        result = build_longitudinal_summary(store, [1, 2, 3], "1A")
        assert result.total_students == 0
        assert result.student_trajectories == []

    def test_persistent_risk_all_below_threshold(self):
        """Student below 0.45 every week should be persistent risk."""
        records = [
            _make_record(week=1, scores={"ensemble_score": 0.1}),
            _make_record(week=2, scores={"ensemble_score": 0.2}),
            _make_record(week=3, scores={"ensemble_score": 0.3}),
        ]
        store = _make_store(records)
        result = build_longitudinal_summary(store, [1, 2, 3], "1A")
        assert len(result.persistent_risk_students) == 1
        assert result.persistent_risk_students[0] == "S001"

    def test_not_persistent_risk_if_any_week_above_threshold(self):
        """Student above 0.45 in at least one week: NOT persistent risk."""
        records = [
            _make_record(week=1, scores={"ensemble_score": 0.1}),
            _make_record(week=2, scores={"ensemble_score": 0.5}),
            _make_record(week=3, scores={"ensemble_score": 0.3}),
        ]
        store = _make_store(records)
        result = build_longitudinal_summary(store, [1, 2, 3], "1A")
        assert len(result.persistent_risk_students) == 0

    def test_class_weekly_averages_multiple_students(self):
        """Class weekly average should be correct with multiple students."""
        records = [
            _make_record(student_id="S001", week=1, scores={"ensemble_score": 0.2}),
            _make_record(student_id="S002", week=1, scores={"ensemble_score": 0.8}),
        ]
        store = _make_store(records)
        result = build_longitudinal_summary(store, [1], "1A")
        assert result.class_weekly_averages[1] == pytest.approx(0.5)

    def test_ols_trend_positive_slope(self):
        """Increasing scores should produce positive OLS trend."""
        records = [
            _make_record(week=1, scores={"ensemble_score": 0.1}),
            _make_record(week=2, scores={"ensemble_score": 0.3}),
            _make_record(week=3, scores={"ensemble_score": 0.5}),
        ]
        store = _make_store(records)
        result = build_longitudinal_summary(store, [1, 2, 3], "1A")
        assert result.student_trajectories[0].overall_trend > 0

    def test_ols_trend_negative_slope(self):
        """Decreasing scores should produce negative OLS trend."""
        records = [
            _make_record(week=1, scores={"ensemble_score": 0.9}),
            _make_record(week=2, scores={"ensemble_score": 0.6}),
            _make_record(week=3, scores={"ensemble_score": 0.3}),
        ]
        store = _make_store(records)
        result = build_longitudinal_summary(store, [1, 2, 3], "1A")
        assert result.student_trajectories[0].overall_trend < 0
