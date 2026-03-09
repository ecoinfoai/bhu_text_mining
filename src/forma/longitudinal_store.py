"""YAML-based longitudinal data store for tracking student progress across weeks."""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import yaml

from forma.evaluation_io import FormaDumper
from forma.evaluation_types import LongitudinalRecord


def _record_key(student_id: str, week: int, question_sn: int) -> str:
    return f"{student_id}_{week}_{question_sn}"


class LongitudinalStore:
    """Persistent YAML store for longitudinal student evaluation records."""

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        self._records: dict[str, dict] = {}

    def _to_dict(self, record: LongitudinalRecord, manual_override: bool = False) -> dict:
        d = {
            "student_id": record.student_id,
            "week": record.week,
            "question_sn": record.question_sn,
            "scores": dict(record.scores),
            "tier_level": record.tier_level,
            "tier_label": record.tier_label,
            "manual_override": manual_override,
        }
        # v2 fields — only include if not None to keep v1 files clean
        if record.node_recall is not None:
            d["node_recall"] = record.node_recall
        if record.edge_f1 is not None:
            d["edge_f1"] = record.edge_f1
        if record.misconception_count is not None:
            d["misconception_count"] = record.misconception_count
        if record.concept_scores is not None:
            d["concept_scores"] = dict(record.concept_scores)
        if record.exam_file is not None:
            d["exam_file"] = record.exam_file
        if record.recorded_at is not None:
            d["recorded_at"] = record.recorded_at
        return d

    def _to_record(self, data: dict) -> LongitudinalRecord:
        return LongitudinalRecord(
            student_id=data["student_id"],
            week=data["week"],
            question_sn=data["question_sn"],
            scores=data["scores"],
            tier_level=data["tier_level"],
            tier_label=data["tier_label"],
            node_recall=data.get("node_recall", None),
            edge_f1=data.get("edge_f1", None),
            misconception_count=data.get("misconception_count", None),
            concept_scores=data.get("concept_scores", None),
            exam_file=data.get("exam_file", None),
            recorded_at=data.get("recorded_at", None),
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
                yaml.dump(data, f, Dumper=FormaDumper, default_flow_style=False, allow_unicode=True, sort_keys=False)
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

    def get_class_snapshot(self, week: int) -> list[LongitudinalRecord]:
        """Return all records for a given week, sorted by student_id."""
        records = [
            self._to_record(d)
            for d in self._records.values()
            if d["week"] == week
        ]
        records.sort(key=lambda r: r.student_id)
        return records

    def get_student_trajectory(
        self, student_id: str, metric: str
    ) -> list[tuple[int, float]]:
        """Return [(week, value)] for a student's metric, sorted by week.

        When a student has multiple questions in a single week, the metric
        values are averaged across questions for that week.
        """
        week_values: dict[int, list[float]] = defaultdict(list)
        for d in self._records.values():
            if d["student_id"] != student_id:
                continue
            val = d["scores"].get(metric)
            if val is not None:
                week_values[d["week"]].append(val)
        result = [
            (wk, sum(vals) / len(vals))
            for wk, vals in sorted(week_values.items())
        ]
        return result

    def get_class_weekly_matrix(
        self, metric: str
    ) -> dict[str, dict[int, float]]:
        """Return {student_id: {week: value}} for a given metric.

        When a student has multiple questions in a single week, the metric
        values are averaged across questions for that week.
        Only students with at least one matching metric value are included.
        """
        matrix: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for d in self._records.values():
            val = d["scores"].get(metric)
            if val is not None:
                matrix[d["student_id"]][d["week"]].append(val)
        return {
            sid: {wk: sum(vs) / len(vs) for wk, vs in sorted(weeks.items())}
            for sid, weeks in matrix.items()
        }


def _compute_concept_scores(
    layer1_results: list,
    student_id: str,
    question_sn: int,
) -> Optional[dict[str, float]]:
    """Compute per-concept is_present ratio from ConceptMatchResult list."""
    counts: dict[str, list[bool]] = defaultdict(list)
    for cmr in layer1_results:
        if cmr.student_id == student_id and cmr.question_sn == question_sn:
            counts[cmr.concept].append(cmr.is_present)
    if not counts:
        return None
    return {
        concept: sum(1 for v in vals if v) / len(vals)
        for concept, vals in counts.items()
    }


def snapshot_from_evaluation(
    store: LongitudinalStore,
    ensemble_results: dict,
    graph_metric_results: dict,
    graph_comparison_results: dict,
    layer1_results: list,
    week: int,
    exam_file: str,
) -> None:
    """Upsert evaluation results into the store with v2 fields.

    Args:
        store: Target LongitudinalStore instance.
        ensemble_results: {student_id: {qsn: EnsembleResult}}.
        graph_metric_results: {student_id: {qsn: GraphMetricResult}}.
        graph_comparison_results: {student_id: {qsn: GraphComparisonResult}}.
        layer1_results: List of ConceptMatchResult.
        week: Current week number.
        exam_file: Exam file basename.
    """
    recorded_at = datetime.now(timezone.utc).isoformat()

    for student_id, q_results in ensemble_results.items():
        for qsn, er in q_results.items():
            # Graph metric results (node_recall)
            gmr = (graph_metric_results.get(student_id) or {}).get(qsn)
            node_recall = gmr.node_recall if gmr else None

            # Graph comparison results (edge_f1, misconception_count)
            gcr = (graph_comparison_results.get(student_id) or {}).get(qsn)
            edge_f1 = gcr.f1 if gcr else None
            misconception_count = (
                len(gcr.wrong_direction_edges) if gcr else None
            )

            # Concept scores from layer1
            concept_scores = _compute_concept_scores(
                layer1_results, student_id, qsn
            )

            record = LongitudinalRecord(
                student_id=student_id,
                week=week,
                question_sn=qsn,
                scores=dict(er.component_scores),
                tier_level=0,
                tier_label=er.understanding_level,
                node_recall=node_recall,
                edge_f1=edge_f1,
                misconception_count=misconception_count,
                concept_scores=concept_scores,
                exam_file=os.path.basename(exam_file),
                recorded_at=recorded_at,
            )
            store.add_record(record)
