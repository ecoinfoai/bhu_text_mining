"""Tests for intervention_store module — TDD RED phase.

Tests InterventionRecord dataclass, InterventionLog YAML persistence,
atomic write, auto-increment IDs, filtering, and outcome updates.
"""
from __future__ import annotations

import os
import time

import pytest
import yaml

from forma.intervention_store import (
    INTERVENTION_TYPES,
    InterventionLog,
    InterventionRecord,
)


# ---------------------------------------------------------------------------
# InterventionRecord dataclass tests
# ---------------------------------------------------------------------------


class TestInterventionRecord:
    """Tests for InterventionRecord dataclass."""

    def test_required_fields(self):
        """Record has required fields: student_id, week, intervention_type."""
        rec = InterventionRecord(
            id=1, student_id="s001", week=2, intervention_type="면담",
        )
        assert rec.student_id == "s001"
        assert rec.week == 2
        assert rec.intervention_type == "면담"
        assert rec.id == 1

    def test_default_description_empty(self):
        """Description defaults to empty string."""
        rec = InterventionRecord(
            id=1, student_id="s001", week=2, intervention_type="면담",
        )
        assert rec.description == ""

    def test_optional_fields_default_none(self):
        """Optional fields default to None."""
        rec = InterventionRecord(
            id=1, student_id="s001", week=2, intervention_type="면담",
        )
        assert rec.recorded_by is None
        assert rec.follow_up_week is None
        assert rec.outcome is None

    def test_recorded_at_auto_iso8601(self):
        """recorded_at is auto-set to ISO 8601 string."""
        rec = InterventionRecord(
            id=1, student_id="s001", week=2, intervention_type="면담",
        )
        assert rec.recorded_at is not None
        assert "T" in rec.recorded_at  # ISO 8601 format

    def test_all_fields_populated(self):
        """All fields can be set explicitly."""
        rec = InterventionRecord(
            id=42,
            student_id="s999",
            week=5,
            intervention_type="보충학습",
            description="보충 자료 제공",
            recorded_by="prof_kim",
            recorded_at="2026-03-10T12:00:00+00:00",
            follow_up_week=6,
            outcome="개선됨",
        )
        assert rec.id == 42
        assert rec.description == "보충 자료 제공"
        assert rec.recorded_by == "prof_kim"
        assert rec.follow_up_week == 6
        assert rec.outcome == "개선됨"


class TestInterventionTypes:
    """Tests for predefined intervention types."""

    def test_five_predefined_types(self):
        """5 predefined intervention types exist."""
        assert len(INTERVENTION_TYPES) == 5

    def test_expected_types(self):
        """Types match specification."""
        expected = {"면담", "보충학습", "과제부여", "멘토링", "기타"}
        assert set(INTERVENTION_TYPES) == expected


# ---------------------------------------------------------------------------
# InterventionLog persistence tests
# ---------------------------------------------------------------------------


class TestInterventionLog:
    """Tests for InterventionLog YAML persistence."""

    def test_create_empty_log(self, tmp_path):
        """New log initializes with no records."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        assert log.get_records() == []

    def test_add_record_returns_id(self, tmp_path):
        """add_record returns auto-assigned ID starting from 1."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        rec_id = log.add_record("s001", 2, "면담", description="상담")
        assert rec_id == 1

    def test_add_multiple_records_auto_increment(self, tmp_path):
        """IDs auto-increment for each new record."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        id1 = log.add_record("s001", 2, "면담")
        id2 = log.add_record("s002", 3, "보충학습")
        id3 = log.add_record("s001", 4, "기타")
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    def test_save_and_load_roundtrip(self, tmp_path):
        """Records survive save/load roundtrip."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        log.add_record("s001", 2, "면담", description="상담")
        log.add_record("s002", 3, "보충학습")
        log.save()

        log2 = InterventionLog(path)
        log2.load()
        records = log2.get_records()
        assert len(records) == 2
        assert records[0].student_id == "s001"
        assert records[0].intervention_type == "면담"
        assert records[1].student_id == "s002"

    def test_save_creates_yaml_file(self, tmp_path):
        """save() creates a YAML file on disk."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        log.add_record("s001", 2, "면담")
        log.save()
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "_meta" in data
        assert "records" in data

    def test_meta_next_id_persisted(self, tmp_path):
        """_meta.next_id is persisted and used on reload."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        log.add_record("s001", 2, "면담")
        log.add_record("s002", 3, "보충학습")
        log.save()

        log2 = InterventionLog(path)
        log2.load()
        new_id = log2.add_record("s003", 4, "기타")
        assert new_id == 3  # continues from where it left off

    def test_atomic_write_backup(self, tmp_path):
        """save() creates .bak backup of previous file."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        log.add_record("s001", 2, "면담")
        log.save()

        # Second save should create backup
        log.add_record("s002", 3, "보충학습")
        log.save()
        assert os.path.exists(path + ".bak")


# ---------------------------------------------------------------------------
# Filtering tests
# ---------------------------------------------------------------------------


class TestInterventionLogFiltering:
    """Tests for get_records filtering."""

    def _populate(self, tmp_path):
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        log.add_record("s001", 2, "면담", description="상담")
        log.add_record("s001", 3, "보충학습", description="보충")
        log.add_record("s002", 2, "과제부여", description="과제")
        log.add_record("s003", 4, "멘토링", description="멘토")
        return log

    def test_filter_by_student_id(self, tmp_path):
        """Filter records by student_id."""
        log = self._populate(tmp_path)
        records = log.get_records(student_id="s001")
        assert len(records) == 2
        assert all(r.student_id == "s001" for r in records)

    def test_filter_by_week(self, tmp_path):
        """Filter records by week."""
        log = self._populate(tmp_path)
        records = log.get_records(week=2)
        assert len(records) == 2
        assert all(r.week == 2 for r in records)

    def test_filter_by_student_and_week(self, tmp_path):
        """Filter by both student_id and week."""
        log = self._populate(tmp_path)
        records = log.get_records(student_id="s001", week=2)
        assert len(records) == 1
        assert records[0].intervention_type == "면담"

    def test_filter_no_match(self, tmp_path):
        """Filter returns empty list when no match."""
        log = self._populate(tmp_path)
        records = log.get_records(student_id="s999")
        assert records == []

    def test_get_all_records(self, tmp_path):
        """get_records with no filter returns all records."""
        log = self._populate(tmp_path)
        records = log.get_records()
        assert len(records) == 4


# ---------------------------------------------------------------------------
# Outcome update tests
# ---------------------------------------------------------------------------


class TestInterventionLogOutcome:
    """Tests for update_outcome."""

    def test_update_outcome_success(self, tmp_path):
        """update_outcome returns True and updates the record."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        rec_id = log.add_record("s001", 2, "면담")
        result = log.update_outcome(rec_id, "개선됨")
        assert result is True
        records = log.get_records(student_id="s001")
        assert records[0].outcome == "개선됨"

    def test_update_outcome_nonexistent_id(self, tmp_path):
        """update_outcome returns False for nonexistent ID."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        result = log.update_outcome(999, "개선됨")
        assert result is False

    def test_update_outcome_persists(self, tmp_path):
        """Updated outcome survives save/load."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        rec_id = log.add_record("s001", 2, "면담")
        log.update_outcome(rec_id, "변화없음")
        log.save()

        log2 = InterventionLog(path)
        log2.load()
        records = log2.get_records(student_id="s001")
        assert records[0].outcome == "변화없음"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestInterventionLogEdgeCases:
    """Edge case tests."""

    def test_invalid_intervention_type_raises(self, tmp_path):
        """Adding an invalid intervention type raises ValueError."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        with pytest.raises(ValueError, match="intervention_type"):
            log.add_record("s001", 2, "invalid_type")

    def test_add_record_with_optional_fields(self, tmp_path):
        """add_record accepts recorded_by and follow_up_week."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        rec_id = log.add_record(
            "s001", 2, "면담",
            description="상담",
            recorded_by="prof_kim",
            follow_up_week=4,
        )
        records = log.get_records(student_id="s001")
        assert records[0].recorded_by == "prof_kim"
        assert records[0].follow_up_week == 4

    def test_empty_file_load(self, tmp_path):
        """Loading from a file with no data works gracefully."""
        path = str(tmp_path / "log.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        log = InterventionLog(path)
        log.load()
        assert log.get_records() == []

    def test_concurrent_id_safety(self, tmp_path):
        """IDs never collide even after save/reload cycles."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        log.add_record("s001", 1, "면담")
        log.save()

        log2 = InterventionLog(path)
        log2.load()
        id2 = log2.add_record("s002", 2, "기타")
        assert id2 == 2

    def test_korean_description_roundtrip(self, tmp_path):
        """Korean text in description survives save/load."""
        path = str(tmp_path / "log.yaml")
        log = InterventionLog(path)
        log.load()
        log.add_record("s001", 2, "면담", description="학습 동기 부여 상담 진행")
        log.save()

        log2 = InterventionLog(path)
        log2.load()
        records = log2.get_records()
        assert records[0].description == "학습 동기 부여 상담 진행"
