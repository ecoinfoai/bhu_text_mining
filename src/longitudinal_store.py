"""YAML-based longitudinal data store for tracking student progress across weeks."""

from __future__ import annotations

import fcntl
import os
import tempfile

import yaml

from src.evaluation_types import LongitudinalRecord


def _record_key(student_id: str, week: int, question_sn: int) -> str:
    return f"{student_id}_{week}_{question_sn}"


class LongitudinalStore:
    """Persistent YAML store for longitudinal student evaluation records."""

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        self._records: dict[str, dict] = {}

    def _to_dict(self, record: LongitudinalRecord, manual_override: bool = False) -> dict:
        return {
            "student_id": record.student_id,
            "week": record.week,
            "question_sn": record.question_sn,
            "scores": dict(record.scores),
            "tier_level": record.tier_level,
            "tier_label": record.tier_label,
            "manual_override": manual_override,
        }

    def _to_record(self, data: dict) -> LongitudinalRecord:
        return LongitudinalRecord(
            student_id=data["student_id"],
            week=data["week"],
            question_sn=data["question_sn"],
            scores=data["scores"],
            tier_level=data["tier_level"],
            tier_label=data["tier_label"],
        )

    def add_record(self, record: LongitudinalRecord) -> None:
        """Upsert record by (student_id, week, question_sn).

        If the existing record has manual_override=True, it is preserved.
        """
        key = _record_key(record.student_id, record.week, record.question_sn)
        existing = self._records.get(key)
        if existing and existing.get("manual_override", False):
            return
        self._records[key] = self._to_dict(record)

    def get_student_history(self, student_id: str) -> list[LongitudinalRecord]:
        """Return all records for a given student."""
        return [
            self._to_record(d)
            for d in self._records.values()
            if d["student_id"] == student_id
        ]

    def get_all_records(self) -> list[LongitudinalRecord]:
        """Return all records in the store."""
        return [self._to_record(d) for d in self._records.values()]

    def save(self) -> None:
        """Atomic write with .bak backup and file locking."""
        data = {"records": self._records}
        dir_name = os.path.dirname(self.store_path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                fcntl.flock(f, fcntl.LOCK_UN)
            if os.path.exists(self.store_path):
                os.replace(self.store_path, self.store_path + ".bak")
            os.replace(tmp_path, self.store_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def load(self) -> None:
        """Load records from YAML file. Initializes empty if file doesn't exist."""
        if not os.path.exists(self.store_path):
            self._records = {}
            return
        with open(self.store_path) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = yaml.safe_load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        self._records = data.get("records", {}) if data else {}
