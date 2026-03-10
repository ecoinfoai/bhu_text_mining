"""YAML-based intervention activity log for tracking student interventions.

Provides InterventionRecord dataclass and InterventionLog class for
persistent storage of intervention activities with atomic writes,
auto-increment IDs, and filtered queries.

Dataclasses:
    InterventionRecord: Single intervention activity record.

Classes:
    InterventionLog: YAML-based persistent intervention log.

Constants:
    INTERVENTION_TYPES: 5 predefined intervention type strings.
"""

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

INTERVENTION_TYPES = ["면담", "보충학습", "과제부여", "멘토링", "기타"]


@dataclass
class InterventionRecord:
    """Single intervention activity record.

    Attributes:
        id: Auto-assigned unique identifier.
        student_id: Student identifier.
        week: Week number when the intervention occurred.
        intervention_type: One of INTERVENTION_TYPES.
        description: Free-text description of the intervention.
        recorded_by: Name of the person who recorded (optional).
        recorded_at: ISO 8601 timestamp of creation (auto-set).
        follow_up_week: Week number for follow-up (optional).
        outcome: Outcome description (optional, set later).
    """

    id: int
    student_id: str
    week: int
    intervention_type: str
    description: str = ""
    recorded_by: Optional[str] = None
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    follow_up_week: Optional[int] = None
    outcome: Optional[str] = None


class InterventionLog:
    """YAML-based persistent intervention log.

    Uses atomic write with tempfile + fcntl.flock + os.replace,
    same pattern as LongitudinalStore.

    File format:
        _meta:
            next_id: N
        records:
            - {id: 1, student_id: ..., ...}
            - ...

    Args:
        store_path: Path to the YAML log file.
    """

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        self._next_id: int = 1
        self._records: list[dict] = []

    def load(self) -> None:
        """Load records from YAML file. Initializes empty if file doesn't exist."""
        if not os.path.exists(self.store_path):
            self._records = []
            self._next_id = 1
            return
        with open(self.store_path, encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = yaml.safe_load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        if not data:
            self._records = []
            self._next_id = 1
            return
        meta = data.get("_meta", {})
        self._next_id = meta.get("next_id", 1)
        self._records = data.get("records", [])

    def save(self) -> None:
        """Atomic write with .bak backup and file locking."""
        data = {
            "_meta": {"next_id": self._next_id},
            "records": self._records,
        }
        dir_name = os.path.dirname(self.store_path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                yaml.dump(
                    data, f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
                fcntl.flock(f, fcntl.LOCK_UN)
            if os.path.exists(self.store_path):
                os.replace(self.store_path, self.store_path + ".bak")
            os.replace(tmp_path, self.store_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def add_record(
        self,
        student_id: str,
        week: int,
        intervention_type: str,
        *,
        description: str = "",
        recorded_by: Optional[str] = None,
        follow_up_week: Optional[int] = None,
    ) -> int:
        """Add a new intervention record and return its auto-assigned ID.

        Args:
            student_id: Student identifier.
            week: Week number.
            intervention_type: Must be one of INTERVENTION_TYPES.
            description: Free-text description.
            recorded_by: Name of the recorder.
            follow_up_week: Week number for follow-up.

        Returns:
            Auto-assigned record ID.

        Raises:
            ValueError: If intervention_type is not in INTERVENTION_TYPES.
        """
        if intervention_type not in INTERVENTION_TYPES:
            raise ValueError(
                f"Invalid intervention_type '{intervention_type}'. "
                f"Must be one of {INTERVENTION_TYPES}"
            )
        record_id = self._next_id
        self._next_id += 1
        record_dict = {
            "id": record_id,
            "student_id": student_id,
            "week": week,
            "intervention_type": intervention_type,
            "description": description,
            "recorded_by": recorded_by,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "follow_up_week": follow_up_week,
            "outcome": None,
        }
        self._records.append(record_dict)
        return record_id

    def get_records(
        self,
        *,
        student_id: Optional[str] = None,
        week: Optional[int] = None,
    ) -> list[InterventionRecord]:
        """Get filtered intervention records.

        Args:
            student_id: Filter by student_id (optional).
            week: Filter by week (optional).

        Returns:
            List of matching InterventionRecord objects.
        """
        results = []
        for d in self._records:
            if student_id is not None and d.get("student_id") != student_id:
                continue
            if week is not None and d.get("week") != week:
                continue
            results.append(self._to_record(d))
        return results

    def update_outcome(self, record_id: int, outcome: str) -> bool:
        """Update the outcome field of an existing record.

        Args:
            record_id: ID of the record to update.
            outcome: New outcome description.

        Returns:
            True if record was found and updated, False otherwise.
        """
        for d in self._records:
            if d.get("id") == record_id:
                d["outcome"] = outcome
                return True
        return False

    def _to_record(self, d: dict) -> InterventionRecord:
        """Convert a dict to an InterventionRecord."""
        return InterventionRecord(
            id=d["id"],
            student_id=d["student_id"],
            week=d["week"],
            intervention_type=d["intervention_type"],
            description=d.get("description", ""),
            recorded_by=d.get("recorded_by"),
            recorded_at=d.get("recorded_at", ""),
            follow_up_week=d.get("follow_up_week"),
            outcome=d.get("outcome"),
        )
