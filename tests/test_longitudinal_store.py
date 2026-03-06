"""Tests for longitudinal_store.py YAML-based student progress tracking."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.evaluation_types import LongitudinalRecord
from src.longitudinal_store import LongitudinalStore


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

        with patch("src.longitudinal_store.fcntl") as mock_fcntl:
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

        with patch("src.longitudinal_store.fcntl") as mock_fcntl:
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
