"""Tests for longitudinal_store.py YAML-based student progress tracking."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from forma.evaluation_types import LongitudinalRecord
from forma.longitudinal_store import LongitudinalStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    student_id: str = "s001",
    week: int = 1,
    question_sn: int = 1,
    scores: dict | None = None,
    tier_level: int = 2,
    tier_label: str = "기전+용어",
) -> LongitudinalRecord:
    return LongitudinalRecord(
        student_id=student_id,
        week=week,
        question_sn=question_sn,
        scores=scores or {"concept_coverage": 0.8, "graph_f1": 0.6},
        tier_level=tier_level,
        tier_label=tier_label,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Save and reload preserves records."""

    def test_add_save_load_roundtrip(self, tmp_path):
        """Adding a record, saving, and reloading should preserve all fields."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        rec = _make_record()
        store.add_record(rec)
        store.save()

        store2 = LongitudinalStore(path)
        store2.load()
        loaded = store2.get_all_records()
        assert len(loaded) == 1
        assert loaded[0].student_id == "s001"
        assert loaded[0].week == 1
        assert loaded[0].question_sn == 1
        assert loaded[0].scores == {"concept_coverage": 0.8, "graph_f1": 0.6}
        assert loaded[0].tier_level == 2
        assert loaded[0].tier_label == "기전+용어"


class TestUpsert:
    """Upsert behavior on same key."""

    def test_upsert_updates_record(self, tmp_path):
        """Adding a record with the same key should overwrite the old one."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(_make_record(tier_level=1, tier_label="용어"))
        store.add_record(_make_record(tier_level=3, tier_label="완벽"))
        records = store.get_all_records()
        assert len(records) == 1
        assert records[0].tier_level == 3
        assert records[0].tier_label == "완벽"


class TestManualOverride:
    """Manual override records are preserved during upsert."""

    def test_manual_override_preserves_record(self, tmp_path):
        """A record with manual_override=True should not be overwritten."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        rec = _make_record(tier_level=2, tier_label="기전+용어")
        store.add_record(rec)
        # Simulate manual override by setting the flag directly
        key = "s001_1_1"
        store._records[key]["manual_override"] = True

        # Attempt upsert — should be rejected
        store.add_record(_make_record(tier_level=3, tier_label="완벽"))
        records = store.get_all_records()
        assert len(records) == 1
        assert records[0].tier_level == 2

    def test_manual_override_survives_roundtrip(self, tmp_path):
        """Manual override flag should persist through save/load."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(_make_record())
        store._records["s001_1_1"]["manual_override"] = True
        store.save()

        store2 = LongitudinalStore(path)
        store2.load()
        # Upsert should still be blocked
        store2.add_record(_make_record(tier_level=3, tier_label="완벽"))
        records = store2.get_all_records()
        assert records[0].tier_level == 2


class TestGetStudentHistory:
    """Filtering by student_id."""

    def test_returns_only_matching_student(self, tmp_path):
        """get_student_history should return only records for the given student."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(_make_record(student_id="s001", week=1))
        store.add_record(_make_record(student_id="s001", week=2))
        store.add_record(_make_record(student_id="s002", week=1))

        history = store.get_student_history("s001")
        assert len(history) == 2
        assert all(r.student_id == "s001" for r in history)

        history2 = store.get_student_history("s002")
        assert len(history2) == 1


class TestAtomicWrite:
    """Atomic save creates .bak backup."""

    def test_backup_created_on_save(self, tmp_path):
        """Second save should create a .bak file from the first save."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(_make_record(tier_level=1))
        store.save()

        store.add_record(_make_record(student_id="s002"))
        store.save()

        bak_path = path + ".bak"
        assert (tmp_path / "store.yaml.bak").exists()


class TestFileLocking:
    """File locking via fcntl."""

    def test_save_uses_exclusive_lock(self, tmp_path):
        """save() should acquire an exclusive lock during write."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(_make_record())

        with patch("forma.longitudinal_store.fcntl") as mock_fcntl:
            mock_fcntl.LOCK_EX = 2
            mock_fcntl.LOCK_UN = 8
            store.save()
            lock_calls = mock_fcntl.flock.call_args_list
            assert any(c.args[1] == 2 for c in lock_calls), "LOCK_EX not called"
            assert any(c.args[1] == 8 for c in lock_calls), "LOCK_UN not called"

    def test_load_uses_shared_lock(self, tmp_path):
        """load() should acquire a shared lock during read."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(_make_record())
        store.save()

        with patch("forma.longitudinal_store.fcntl") as mock_fcntl:
            mock_fcntl.LOCK_SH = 1
            mock_fcntl.LOCK_UN = 8
            store2 = LongitudinalStore(path)
            store2.load()
            lock_calls = mock_fcntl.flock.call_args_list
            assert any(c.args[1] == 1 for c in lock_calls), "LOCK_SH not called"


class TestGetAllRecords:
    """get_all_records returns all stored records."""

    def test_returns_all(self, tmp_path):
        """get_all_records should return every record in the store."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(_make_record(student_id="s001", week=1))
        store.add_record(_make_record(student_id="s002", week=1))
        store.add_record(_make_record(student_id="s003", week=2))
        assert len(store.get_all_records()) == 3


class TestLoadNonexistent:
    """Loading from a missing file initializes empty store."""

    def test_load_missing_file(self, tmp_path):
        """load() on nonexistent file should result in empty records."""
        path = str(tmp_path / "does_not_exist.yaml")
        store = LongitudinalStore(path)
        store.load()
        assert store.get_all_records() == []


# ---------------------------------------------------------------------------
# Phase 2: v2 field tests (T002, T003)
# ---------------------------------------------------------------------------


def _make_v2_record(
    student_id: str = "s001",
    week: int = 1,
    question_sn: int = 1,
    scores: dict | None = None,
    tier_level: int = 2,
    tier_label: str = "Proficient",
    node_recall: float | None = 0.80,
    edge_f1: float | None = 0.75,
    misconception_count: int | None = 2,
    concept_scores: dict[str, float] | None = None,
    exam_file: str | None = "Ch01_test.yaml",
    recorded_at: str | None = "2026-03-01T14:30:00+09:00",
) -> LongitudinalRecord:
    """Helper to construct a v2 LongitudinalRecord with all optional fields."""
    return LongitudinalRecord(
        student_id=student_id,
        week=week,
        question_sn=question_sn,
        scores=scores or {"ensemble_score": 0.72, "concept_coverage": 0.65},
        tier_level=tier_level,
        tier_label=tier_label,
        node_recall=node_recall,
        edge_f1=edge_f1,
        misconception_count=misconception_count,
        concept_scores=concept_scores or {"심근경색": 0.85, "허혈": 0.60},
        exam_file=exam_file,
        recorded_at=recorded_at,
    )


class TestV2Fields:
    """T002: LongitudinalRecord v2 field construction and defaults."""

    def test_v2_fields_default_to_none(self):
        """All v2 optional fields should default to None when not specified."""
        rec = _make_record()
        assert rec.node_recall is None
        assert rec.edge_f1 is None
        assert rec.misconception_count is None
        assert rec.concept_scores is None
        assert rec.exam_file is None
        assert rec.recorded_at is None

    def test_v2_fields_accept_values(self):
        """v2 fields should accept explicit values."""
        rec = _make_v2_record()
        assert rec.node_recall == 0.80
        assert rec.edge_f1 == 0.75
        assert rec.misconception_count == 2
        assert rec.concept_scores == {"심근경색": 0.85, "허혈": 0.60}
        assert rec.exam_file == "Ch01_test.yaml"
        assert rec.recorded_at == "2026-03-01T14:30:00+09:00"

    def test_v2_partial_fields(self):
        """Some v2 fields set, others remain None."""
        rec = LongitudinalRecord(
            student_id="s001",
            week=1,
            question_sn=1,
            scores={"ensemble_score": 0.5},
            tier_level=1,
            tier_label="Developing",
            node_recall=0.5,
            edge_f1=None,
            misconception_count=3,
        )
        assert rec.node_recall == 0.5
        assert rec.edge_f1 is None
        assert rec.misconception_count == 3
        assert rec.concept_scores is None
        assert rec.exam_file is None
        assert rec.recorded_at is None

    def test_v2_roundtrip_preserves_all_fields(self, tmp_path):
        """Save and reload a v2 record — all v2 fields should survive."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        rec = _make_v2_record()
        store.add_record(rec)
        store.save()

        store2 = LongitudinalStore(path)
        store2.load()
        loaded = store2.get_all_records()
        assert len(loaded) == 1
        r = loaded[0]
        assert r.node_recall == 0.80
        assert r.edge_f1 == 0.75
        assert r.misconception_count == 2
        assert r.concept_scores == {"심근경색": 0.85, "허혈": 0.60}
        assert r.exam_file == "Ch01_test.yaml"
        assert r.recorded_at == "2026-03-01T14:30:00+09:00"

    def test_v2_none_fields_roundtrip(self, tmp_path):
        """v2 record with all None optional fields should roundtrip correctly."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        rec = _make_record()  # v1-style, all v2 fields None
        store.add_record(rec)
        store.save()

        store2 = LongitudinalStore(path)
        store2.load()
        loaded = store2.get_all_records()
        assert len(loaded) == 1
        r = loaded[0]
        assert r.node_recall is None
        assert r.edge_f1 is None
        assert r.misconception_count is None
        assert r.concept_scores is None
        assert r.exam_file is None
        assert r.recorded_at is None


class TestV1BackwardCompat:
    """T003: v1 records (no v2 fields) load without error, v2 fields are None."""

    def test_v1_record_loads_v2_fields_as_none(self, tmp_path):
        """A v1 store file (no v2 keys) should load, with v2 fields as None."""
        path = str(tmp_path / "store.yaml")
        # Write a v1-format store file directly (no v2 keys)
        import yaml
        v1_data = {
            "records": {
                "s001_1_1": {
                    "student_id": "s001",
                    "week": 1,
                    "question_sn": 1,
                    "scores": {"ensemble_score": 0.72},
                    "tier_level": 2,
                    "tier_label": "Proficient",
                    "manual_override": False,
                }
            }
        }
        with open(path, "w") as f:
            yaml.dump(v1_data, f)

        store = LongitudinalStore(path)
        store.load()
        records = store.get_all_records()
        assert len(records) == 1
        r = records[0]
        # v1 fields preserved
        assert r.student_id == "s001"
        assert r.week == 1
        assert r.scores == {"ensemble_score": 0.72}
        assert r.tier_level == 2
        assert r.tier_label == "Proficient"
        # v2 fields default to None
        assert r.node_recall is None
        assert r.edge_f1 is None
        assert r.misconception_count is None
        assert r.concept_scores is None
        assert r.exam_file is None
        assert r.recorded_at is None

    def test_v1_v2_mixed_store(self, tmp_path):
        """A store with both v1 and v2 records should load all correctly."""
        path = str(tmp_path / "store.yaml")
        import yaml
        mixed_data = {
            "records": {
                "s001_1_1": {
                    "student_id": "s001",
                    "week": 1,
                    "question_sn": 1,
                    "scores": {"ensemble_score": 0.50},
                    "tier_level": 1,
                    "tier_label": "Developing",
                    "manual_override": False,
                },
                "s002_1_1": {
                    "student_id": "s002",
                    "week": 1,
                    "question_sn": 1,
                    "scores": {"ensemble_score": 0.80},
                    "tier_level": 3,
                    "tier_label": "Advanced",
                    "manual_override": False,
                    "node_recall": 0.90,
                    "edge_f1": 0.85,
                    "misconception_count": 1,
                    "concept_scores": {"세포막": 1.0},
                    "exam_file": "test.yaml",
                    "recorded_at": "2026-03-01T00:00:00Z",
                },
            }
        }
        with open(path, "w") as f:
            yaml.dump(mixed_data, f)

        store = LongitudinalStore(path)
        store.load()
        records = store.get_all_records()
        assert len(records) == 2

        # Find each record
        by_id = {r.student_id: r for r in records}
        r1 = by_id["s001"]
        r2 = by_id["s002"]

        # v1 record: all v2 fields None
        assert r1.node_recall is None
        assert r1.edge_f1 is None
        assert r1.misconception_count is None
        assert r1.concept_scores is None
        assert r1.exam_file is None
        assert r1.recorded_at is None

        # v2 record: all v2 fields present
        assert r2.node_recall == 0.90
        assert r2.edge_f1 == 0.85
        assert r2.misconception_count == 1
        assert r2.concept_scores == {"세포막": 1.0}
        assert r2.exam_file == "test.yaml"
        assert r2.recorded_at == "2026-03-01T00:00:00Z"

    def test_v1_record_upsert_with_v2_record(self, tmp_path):
        """Upserting a v2 record over a v1 record should update to v2."""
        path = str(tmp_path / "store.yaml")
        import yaml
        v1_data = {
            "records": {
                "s001_1_1": {
                    "student_id": "s001",
                    "week": 1,
                    "question_sn": 1,
                    "scores": {"ensemble_score": 0.50},
                    "tier_level": 1,
                    "tier_label": "Developing",
                    "manual_override": False,
                }
            }
        }
        with open(path, "w") as f:
            yaml.dump(v1_data, f)

        store = LongitudinalStore(path)
        store.load()

        # Upsert with v2 record
        v2_rec = _make_v2_record(
            student_id="s001", week=1, question_sn=1,
            node_recall=0.70, edge_f1=0.65, misconception_count=3,
        )
        store.add_record(v2_rec)
        store.save()

        store2 = LongitudinalStore(path)
        store2.load()
        records = store2.get_all_records()
        assert len(records) == 1
        r = records[0]
        assert r.node_recall == 0.70
        assert r.edge_f1 == 0.65
        assert r.misconception_count == 3


# ---------------------------------------------------------------------------
# Phase 3: US1 tests (T006-T010)
# ---------------------------------------------------------------------------


def _populate_multi_week_store(store: LongitudinalStore) -> None:
    """Add 3-week, 2-student, 1-question data to the store for query tests."""
    data = [
        ("s001", 1, 1, {"ensemble_score": 0.50}, 1, "Developing"),
        ("s001", 2, 1, {"ensemble_score": 0.65}, 2, "Proficient"),
        ("s001", 3, 1, {"ensemble_score": 0.80}, 3, "Advanced"),
        ("s002", 1, 1, {"ensemble_score": 0.40}, 1, "Beginning"),
        ("s002", 2, 1, {"ensemble_score": 0.55}, 2, "Developing"),
        ("s002", 3, 1, {"ensemble_score": 0.60}, 2, "Proficient"),
    ]
    for sid, wk, qsn, scores, tier, label in data:
        store.add_record(LongitudinalRecord(
            student_id=sid, week=wk, question_sn=qsn,
            scores=scores, tier_level=tier, tier_label=label,
        ))


class TestGetClassSnapshot:
    """T006: get_class_snapshot(week) — returns all records for a given week."""

    def test_returns_all_students_for_week(self, tmp_path):
        """Should return all student records for the specified week."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        _populate_multi_week_store(store)

        snapshot = store.get_class_snapshot(week=2)
        assert len(snapshot) == 2
        ids = [r.student_id for r in snapshot]
        assert "s001" in ids
        assert "s002" in ids

    def test_sorted_by_student_id(self, tmp_path):
        """Results should be sorted by student_id."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        _populate_multi_week_store(store)

        snapshot = store.get_class_snapshot(week=1)
        assert [r.student_id for r in snapshot] == ["s001", "s002"]

    def test_empty_week_returns_empty(self, tmp_path):
        """A week with no data should return empty list."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        _populate_multi_week_store(store)

        snapshot = store.get_class_snapshot(week=99)
        assert snapshot == []

    def test_multiple_questions_per_student(self, tmp_path):
        """Should return all question records for each student in the week."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(_make_record(student_id="s001", week=1, question_sn=1))
        store.add_record(_make_record(student_id="s001", week=1, question_sn=2))
        store.add_record(_make_record(student_id="s001", week=2, question_sn=1))

        snapshot = store.get_class_snapshot(week=1)
        assert len(snapshot) == 2
        qsns = sorted(r.question_sn for r in snapshot)
        assert qsns == [1, 2]


class TestGetStudentTrajectory:
    """T007: get_student_trajectory(student_id, metric) — returns [(week, value)]."""

    def test_returns_sorted_by_week(self, tmp_path):
        """Should return (week, value) pairs sorted by week."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        _populate_multi_week_store(store)

        trajectory = store.get_student_trajectory("s001", "ensemble_score")
        assert trajectory == [(1, 0.50), (2, 0.65), (3, 0.80)]

    def test_missing_metric_returns_empty(self, tmp_path):
        """A metric not in scores should return empty list."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        _populate_multi_week_store(store)

        trajectory = store.get_student_trajectory("s001", "nonexistent_metric")
        assert trajectory == []

    def test_nonexistent_student_returns_empty(self, tmp_path):
        """A student not in the store should return empty list."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        _populate_multi_week_store(store)

        trajectory = store.get_student_trajectory("s999", "ensemble_score")
        assert trajectory == []

    def test_non_contiguous_weeks(self, tmp_path):
        """Non-contiguous weeks (1, 3, 5) should still sort correctly."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(LongitudinalRecord(
            student_id="s001", week=5, question_sn=1,
            scores={"score": 0.9}, tier_level=3, tier_label="A",
        ))
        store.add_record(LongitudinalRecord(
            student_id="s001", week=1, question_sn=1,
            scores={"score": 0.3}, tier_level=1, tier_label="B",
        ))
        store.add_record(LongitudinalRecord(
            student_id="s001", week=3, question_sn=1,
            scores={"score": 0.6}, tier_level=2, tier_label="P",
        ))

        trajectory = store.get_student_trajectory("s001", "score")
        assert trajectory == [(1, 0.3), (3, 0.6), (5, 0.9)]

    def test_multiple_questions_uses_first_per_week(self, tmp_path):
        """When a student has multiple questions in a week, use average score per week."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(LongitudinalRecord(
            student_id="s001", week=1, question_sn=1,
            scores={"score": 0.4}, tier_level=1, tier_label="D",
        ))
        store.add_record(LongitudinalRecord(
            student_id="s001", week=1, question_sn=2,
            scores={"score": 0.6}, tier_level=2, tier_label="P",
        ))

        trajectory = store.get_student_trajectory("s001", "score")
        # Should average the two question scores: (0.4 + 0.6) / 2 = 0.5
        assert len(trajectory) == 1
        assert trajectory[0][0] == 1
        assert abs(trajectory[0][1] - 0.5) < 1e-9


class TestGetClassWeeklyMatrix:
    """T008: get_class_weekly_matrix(metric) — returns {student_id: {week: value}}."""

    def test_returns_correct_matrix(self, tmp_path):
        """Should return complete matrix with all students and weeks."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        _populate_multi_week_store(store)

        matrix = store.get_class_weekly_matrix("ensemble_score")
        assert set(matrix.keys()) == {"s001", "s002"}
        assert matrix["s001"] == {1: 0.50, 2: 0.65, 3: 0.80}
        assert matrix["s002"] == {1: 0.40, 2: 0.55, 3: 0.60}

    def test_missing_metric_returns_empty_inner(self, tmp_path):
        """A metric not in scores should produce empty inner dicts."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        _populate_multi_week_store(store)

        matrix = store.get_class_weekly_matrix("nonexistent")
        # All students present, but no weeks have data
        assert matrix == {}

    def test_partial_weeks(self, tmp_path):
        """Students with different week coverage should still work."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(LongitudinalRecord(
            student_id="s001", week=1, question_sn=1,
            scores={"score": 0.5}, tier_level=1, tier_label="D",
        ))
        store.add_record(LongitudinalRecord(
            student_id="s001", week=2, question_sn=1,
            scores={"score": 0.7}, tier_level=2, tier_label="P",
        ))
        store.add_record(LongitudinalRecord(
            student_id="s002", week=2, question_sn=1,
            scores={"score": 0.6}, tier_level=2, tier_label="P",
        ))

        matrix = store.get_class_weekly_matrix("score")
        assert matrix["s001"] == {1: 0.5, 2: 0.7}
        assert matrix["s002"] == {2: 0.6}

    def test_empty_store_returns_empty(self, tmp_path):
        """An empty store should return empty dict."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)

        matrix = store.get_class_weekly_matrix("score")
        assert matrix == {}


class TestSnapshotFromEvaluation:
    """T009: snapshot_from_evaluation() — upserts records with v2 fields."""

    def test_basic_snapshot(self, tmp_path):
        """snapshot_from_evaluation should create records with v2 fields."""
        from forma.longitudinal_store import snapshot_from_evaluation
        from forma.evaluation_types import (
            EnsembleResult, GraphComparisonResult, GraphMetricResult,
        )

        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)

        ensemble_results = {
            "s001": {
                1: EnsembleResult(
                    student_id="s001", question_sn=1,
                    ensemble_score=0.72, understanding_level="Proficient",
                    component_scores={"concept_coverage": 0.65, "graph_f1": 0.78},
                    weights_used={"concept_coverage": 0.4, "graph_f1": 0.6},
                )
            }
        }
        graph_metric_results = {
            "s001": {
                1: GraphMetricResult(
                    student_id="s001", question_sn=1,
                    node_recall=0.80, edge_jaccard=0.5,
                    centrality_deviation=0.3, normalized_ged=0.4,
                )
            }
        }
        graph_comparison_results = {
            "s001": {
                1: GraphComparisonResult(
                    student_id="s001", question_sn=1,
                    precision=0.8, recall=0.7, f1=0.75,
                    matched_edges=[], missing_edges=[],
                    extra_edges=[], wrong_direction_edges=[
                        # 2 wrong-direction edges
                        type("E", (), {"subject": "A", "relation": "r", "object": "B"})(),
                        type("E", (), {"subject": "C", "relation": "r", "object": "D"})(),
                    ],
                )
            }
        }

        snapshot_from_evaluation(
            store=store,
            ensemble_results=ensemble_results,
            graph_metric_results=graph_metric_results,
            graph_comparison_results=graph_comparison_results,
            layer1_results=[],
            week=3,
            exam_file="Ch01_test.yaml",
        )

        records = store.get_all_records()
        assert len(records) == 1
        r = records[0]
        assert r.student_id == "s001"
        assert r.week == 3
        assert r.question_sn == 1
        assert r.scores == {"concept_coverage": 0.65, "graph_f1": 0.78}
        assert r.tier_label == "Proficient"
        assert r.node_recall == 0.80
        assert r.edge_f1 == 0.75
        assert r.misconception_count == 2
        assert r.exam_file == "Ch01_test.yaml"
        assert r.recorded_at is not None  # ISO timestamp

    def test_respects_manual_override(self, tmp_path):
        """snapshot_from_evaluation should not overwrite manual_override records."""
        from forma.longitudinal_store import snapshot_from_evaluation
        from forma.evaluation_types import EnsembleResult

        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)

        # Pre-populate with manual override
        rec = _make_record(student_id="s001", week=3, question_sn=1, tier_level=2)
        store.add_record(rec)
        key = "s001_3_1"
        store._records[key]["manual_override"] = True

        ensemble_results = {
            "s001": {
                1: EnsembleResult(
                    student_id="s001", question_sn=1,
                    ensemble_score=0.90, understanding_level="Advanced",
                    component_scores={"concept_coverage": 0.95},
                    weights_used={"concept_coverage": 1.0},
                )
            }
        }

        snapshot_from_evaluation(
            store=store,
            ensemble_results=ensemble_results,
            graph_metric_results={},
            graph_comparison_results={},
            layer1_results=[],
            week=3,
            exam_file="test.yaml",
        )

        records = store.get_all_records()
        assert len(records) == 1
        assert records[0].tier_level == 2  # Preserved, not overwritten

    def test_multiple_students_and_questions(self, tmp_path):
        """snapshot_from_evaluation with multiple students and questions."""
        from forma.longitudinal_store import snapshot_from_evaluation
        from forma.evaluation_types import EnsembleResult

        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)

        ensemble_results = {
            "s001": {
                1: EnsembleResult(
                    student_id="s001", question_sn=1,
                    ensemble_score=0.70, understanding_level="Proficient",
                    component_scores={"score": 0.70},
                    weights_used={"score": 1.0},
                ),
                2: EnsembleResult(
                    student_id="s001", question_sn=2,
                    ensemble_score=0.60, understanding_level="Developing",
                    component_scores={"score": 0.60},
                    weights_used={"score": 1.0},
                ),
            },
            "s002": {
                1: EnsembleResult(
                    student_id="s002", question_sn=1,
                    ensemble_score=0.80, understanding_level="Advanced",
                    component_scores={"score": 0.80},
                    weights_used={"score": 1.0},
                ),
            },
        }

        snapshot_from_evaluation(
            store=store,
            ensemble_results=ensemble_results,
            graph_metric_results={},
            graph_comparison_results={},
            layer1_results=[],
            week=1,
            exam_file="test.yaml",
        )

        records = store.get_all_records()
        assert len(records) == 3

    def test_no_graph_results(self, tmp_path):
        """When graph results are empty, node_recall/edge_f1/misconception_count are None."""
        from forma.longitudinal_store import snapshot_from_evaluation
        from forma.evaluation_types import EnsembleResult

        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)

        ensemble_results = {
            "s001": {
                1: EnsembleResult(
                    student_id="s001", question_sn=1,
                    ensemble_score=0.70, understanding_level="Proficient",
                    component_scores={"score": 0.70},
                    weights_used={"score": 1.0},
                ),
            },
        }

        snapshot_from_evaluation(
            store=store,
            ensemble_results=ensemble_results,
            graph_metric_results={},
            graph_comparison_results={},
            layer1_results=[],
            week=1,
            exam_file="test.yaml",
        )

        records = store.get_all_records()
        r = records[0]
        assert r.node_recall is None
        assert r.edge_f1 is None
        assert r.misconception_count is None


class TestConceptScoresExtraction:
    """T010: concept_scores computed from ConceptMatchResult list."""

    def test_concept_scores_from_layer1(self, tmp_path):
        """concept_scores should aggregate per-concept is_present ratio."""
        from forma.longitudinal_store import snapshot_from_evaluation
        from forma.evaluation_types import (
            EnsembleResult, ConceptMatchResult,
        )

        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)

        ensemble_results = {
            "s001": {
                1: EnsembleResult(
                    student_id="s001", question_sn=1,
                    ensemble_score=0.70, understanding_level="Proficient",
                    component_scores={"score": 0.70},
                    weights_used={"score": 1.0},
                ),
            },
        }

        layer1_results = [
            ConceptMatchResult(
                concept="세포막", student_id="s001", question_sn=1,
                is_present=True, similarity_score=0.8,
                top_k_mean_similarity=0.8, threshold_used=0.5,
            ),
            ConceptMatchResult(
                concept="세포질", student_id="s001", question_sn=1,
                is_present=False, similarity_score=0.3,
                top_k_mean_similarity=0.3, threshold_used=0.5,
            ),
            ConceptMatchResult(
                concept="핵", student_id="s001", question_sn=1,
                is_present=True, similarity_score=0.9,
                top_k_mean_similarity=0.9, threshold_used=0.5,
            ),
        ]

        snapshot_from_evaluation(
            store=store,
            ensemble_results=ensemble_results,
            graph_metric_results={},
            graph_comparison_results={},
            layer1_results=layer1_results,
            week=1,
            exam_file="test.yaml",
        )

        records = store.get_all_records()
        r = records[0]
        assert r.concept_scores is not None
        assert r.concept_scores["세포막"] == 1.0  # 1/1 present
        assert r.concept_scores["세포질"] == 0.0  # 0/1 present
        assert r.concept_scores["핵"] == 1.0  # 1/1 present

    def test_concept_scores_empty_layer1(self, tmp_path):
        """Empty layer1_results should produce None concept_scores."""
        from forma.longitudinal_store import snapshot_from_evaluation
        from forma.evaluation_types import EnsembleResult

        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)

        ensemble_results = {
            "s001": {
                1: EnsembleResult(
                    student_id="s001", question_sn=1,
                    ensemble_score=0.70, understanding_level="Proficient",
                    component_scores={"score": 0.70},
                    weights_used={"score": 1.0},
                ),
            },
        }

        snapshot_from_evaluation(
            store=store,
            ensemble_results=ensemble_results,
            graph_metric_results={},
            graph_comparison_results={},
            layer1_results=[],
            week=1,
            exam_file="test.yaml",
        )

        records = store.get_all_records()
        assert records[0].concept_scores is None

    def test_concept_scores_multiple_judgments_per_concept(self, tmp_path):
        """Multiple ConceptMatchResult entries for same concept should average."""
        from forma.longitudinal_store import snapshot_from_evaluation
        from forma.evaluation_types import (
            EnsembleResult, ConceptMatchResult,
        )

        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)

        ensemble_results = {
            "s001": {
                1: EnsembleResult(
                    student_id="s001", question_sn=1,
                    ensemble_score=0.70, understanding_level="Proficient",
                    component_scores={"score": 0.70},
                    weights_used={"score": 1.0},
                ),
            },
        }

        # Same concept, 3 judgments: 2 present, 1 absent → ratio = 2/3
        layer1_results = [
            ConceptMatchResult(
                concept="세포막", student_id="s001", question_sn=1,
                is_present=True, similarity_score=0.8,
                top_k_mean_similarity=0.8, threshold_used=0.5,
            ),
            ConceptMatchResult(
                concept="세포막", student_id="s001", question_sn=1,
                is_present=True, similarity_score=0.7,
                top_k_mean_similarity=0.7, threshold_used=0.5,
            ),
            ConceptMatchResult(
                concept="세포막", student_id="s001", question_sn=1,
                is_present=False, similarity_score=0.3,
                top_k_mean_similarity=0.3, threshold_used=0.5,
            ),
        ]

        snapshot_from_evaluation(
            store=store,
            ensemble_results=ensemble_results,
            graph_metric_results={},
            graph_comparison_results={},
            layer1_results=layer1_results,
            week=1,
            exam_file="test.yaml",
        )

        records = store.get_all_records()
        r = records[0]
        assert r.concept_scores is not None
        assert abs(r.concept_scores["세포막"] - 2 / 3) < 1e-9

    def test_concept_scores_filters_by_student_and_question(self, tmp_path):
        """concept_scores should only use ConceptMatchResult for matching student+question."""
        from forma.longitudinal_store import snapshot_from_evaluation
        from forma.evaluation_types import (
            EnsembleResult, ConceptMatchResult,
        )

        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)

        ensemble_results = {
            "s001": {
                1: EnsembleResult(
                    student_id="s001", question_sn=1,
                    ensemble_score=0.70, understanding_level="Proficient",
                    component_scores={"score": 0.70},
                    weights_used={"score": 1.0},
                ),
            },
        }

        layer1_results = [
            # s001 q1 — should be included
            ConceptMatchResult(
                concept="A", student_id="s001", question_sn=1,
                is_present=True, similarity_score=0.8,
                top_k_mean_similarity=0.8, threshold_used=0.5,
            ),
            # s001 q2 — should NOT be included (different question)
            ConceptMatchResult(
                concept="B", student_id="s001", question_sn=2,
                is_present=True, similarity_score=0.8,
                top_k_mean_similarity=0.8, threshold_used=0.5,
            ),
            # s002 q1 — should NOT be included (different student)
            ConceptMatchResult(
                concept="C", student_id="s002", question_sn=1,
                is_present=True, similarity_score=0.8,
                top_k_mean_similarity=0.8, threshold_used=0.5,
            ),
        ]

        snapshot_from_evaluation(
            store=store,
            ensemble_results=ensemble_results,
            graph_metric_results={},
            graph_comparison_results={},
            layer1_results=layer1_results,
            week=1,
            exam_file="test.yaml",
        )

        records = store.get_all_records()
        r = records[0]
        assert r.concept_scores is not None
        assert set(r.concept_scores.keys()) == {"A"}
        assert r.concept_scores["A"] == 1.0


# ---------------------------------------------------------------------------
# T042: Edge case tests — 0-student week, non-contiguous weeks
# ---------------------------------------------------------------------------


class TestEdgeCaseZeroStudentWeek:
    """get_class_snapshot for a week with no data returns empty list."""

    def test_snapshot_nonexistent_week(self, tmp_path):
        """get_class_snapshot(week=99) on store with no week 99 → empty list."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(_make_record(student_id="s001", week=1))
        store.add_record(_make_record(student_id="s002", week=2))

        result = store.get_class_snapshot(99)
        assert result == []

    def test_snapshot_empty_store(self, tmp_path):
        """get_class_snapshot on empty store → empty list."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)

        result = store.get_class_snapshot(1)
        assert result == []


class TestEdgeCaseNonContiguousWeeks:
    """Store with non-contiguous weeks (1, 3, 7)."""

    def _build_sparse_store(self, tmp_path):
        """Build store with weeks 1, 3, 7 for student s001."""
        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(_make_record(
            student_id="s001", week=1,
            scores={"ensemble_score": 0.4},
        ))
        store.add_record(_make_record(
            student_id="s001", week=3,
            scores={"ensemble_score": 0.6},
        ))
        store.add_record(_make_record(
            student_id="s001", week=7,
            scores={"ensemble_score": 0.9},
        ))
        return store

    def test_trajectory_sparse_weeks(self, tmp_path):
        """get_student_trajectory with non-contiguous weeks returns correct sparse data."""
        store = self._build_sparse_store(tmp_path)
        traj = store.get_student_trajectory("s001", "ensemble_score")

        assert len(traj) == 3
        assert traj[0] == (1, 0.4)
        assert traj[1] == (3, 0.6)
        assert traj[2] == (7, 0.9)

    def test_trajectory_sorted_by_week(self, tmp_path):
        """Trajectory is sorted by week even with gaps."""
        store = self._build_sparse_store(tmp_path)
        traj = store.get_student_trajectory("s001", "ensemble_score")

        weeks = [w for w, _ in traj]
        assert weeks == sorted(weeks)

    def test_matrix_sparse_weeks(self, tmp_path):
        """get_class_weekly_matrix with non-contiguous weeks → correct sparse matrix."""
        store = self._build_sparse_store(tmp_path)
        # Add another student with different sparse weeks
        store.add_record(_make_record(
            student_id="s002", week=1,
            scores={"ensemble_score": 0.5},
        ))
        store.add_record(_make_record(
            student_id="s002", week=7,
            scores={"ensemble_score": 0.8},
        ))

        matrix = store.get_class_weekly_matrix("ensemble_score")

        # s001 has weeks 1, 3, 7
        assert set(matrix["s001"].keys()) == {1, 3, 7}
        assert matrix["s001"][1] == 0.4

        # s002 has weeks 1, 7 only (no week 3)
        assert set(matrix["s002"].keys()) == {1, 7}
        assert 3 not in matrix["s002"]

    def test_matrix_missing_student_weeks(self, tmp_path):
        """Students with different week coverage produce correct sparse matrix."""
        store = self._build_sparse_store(tmp_path)
        matrix = store.get_class_weekly_matrix("ensemble_score")

        # Only s001 in the store
        assert "s001" in matrix
        assert matrix["s001"][7] == 0.9
