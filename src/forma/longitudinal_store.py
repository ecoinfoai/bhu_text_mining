"""YAML-based longitudinal data store for tracking student progress across weeks."""

from __future__ import annotations

import fcntl
import math
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional  # used by _compute_concept_scores return type

import yaml

import re

from forma.evaluation_io import FormaDumper
from forma.evaluation_types import LongitudinalRecord


def _infer_class_id(path: str) -> str | None:
    """Extract class_id from directory or filename pattern.

    Matches patterns like ``eval_A``, ``final_BC.yaml``,
    ``eval_ABC/``. Returns None if no pattern matches.

    Args:
        path: Directory or file path string.

    Returns:
        Extracted class identifier or None.
    """
    m = re.search(r"[_]([A-Z]{1,3})(?:[_./\\]|$)", path)
    return m.group(1) if m else None


def _record_key(
    student_id: str,
    week: int,
    question_sn: int,
    class_id: str | None = None,
    semester: str | None = None,
) -> str:
    """Build a unique composite key for a longitudinal record.

    Args:
        student_id: Student identifier.
        week: Week number.
        question_sn: Question serial number.
        class_id: Optional section identifier for class isolation.
        semester: Optional semester label for semester isolation.

    Returns:
        Composite key string.
    """
    key = f"{student_id}_{week}_{question_sn}"
    if class_id is not None:
        key = f"{key}_{class_id}"
    if semester is not None:
        key = f"{key}_{semester}"
    return key


class LongitudinalStore:
    """Persistent YAML store for longitudinal student evaluation records."""

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        self._lock_path = str(self.store_path) + ".lock"
        self._records: dict[str, dict] = {}
        self._by_student: dict[str, list[str]] = {}
        self._by_week: dict[int, list[str]] = {}

    def _rebuild_index(self) -> None:
        """Rebuild _by_student and _by_week indexes from _records."""
        by_student: dict[str, list[str]] = {}
        by_week: dict[int, list[str]] = {}
        for key, d in self._records.items():
            sid = d["student_id"]
            wk = d["week"]
            by_student.setdefault(sid, []).append(key)
            by_week.setdefault(wk, []).append(key)
        self._by_student = by_student
        self._by_week = by_week

    def _to_dict(self, record: LongitudinalRecord, manual_override: bool = False) -> dict:
        """Convert a LongitudinalRecord to a plain dict for YAML storage."""
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
        if record.ocr_confidence_mean is not None:
            d["ocr_confidence_mean"] = record.ocr_confidence_mean
        if record.ocr_confidence_min is not None:
            d["ocr_confidence_min"] = record.ocr_confidence_min
        if record.topic is not None:
            d["topic"] = record.topic
        if record.class_id is not None:
            d["class_id"] = record.class_id
        if record.semester is not None:
            d["semester"] = record.semester
        return d

    def _to_record(self, data: dict) -> LongitudinalRecord:
        """Convert a stored dict back to a LongitudinalRecord."""
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
            ocr_confidence_mean=data.get("ocr_confidence_mean", None),
            ocr_confidence_min=data.get("ocr_confidence_min", None),
            topic=data.get("topic", None),
            class_id=data.get("class_id", None),
            semester=data.get("semester", None),
        )

    @staticmethod
    def _validate_scores(scores: dict[str, float]) -> None:
        """Validate score values, rejecting NaN, Inf, and out-of-range values.

        Args:
            scores: Dict of metric name to score value.

        Raises:
            ValueError: If any score is NaN, infinite, or negative.
        """
        for metric, value in scores.items():
            if not isinstance(value, (int, float)):
                continue
            if math.isnan(value):
                raise ValueError(f"Score cannot be NaN (metric='{metric}')")
            if math.isinf(value):
                raise ValueError(f"Score cannot be infinite (metric='{metric}')")
            if value < 0.0:
                raise ValueError(f"Score cannot be negative (metric='{metric}', value={value})")

    def add_record(self, record: LongitudinalRecord) -> None:
        """Upsert record by (student_id, week, question_sn[, class_id][, semester]).

        If the existing record has manual_override=True, it is preserved.
        When class_id or semester are set on the record, they become part of
        the composite key, preventing cross-class and cross-semester overwrites.

        Raises:
            ValueError: If any score value is NaN, infinite, or negative.
        """
        self._validate_scores(record.scores)
        key = _record_key(
            record.student_id,
            record.week,
            record.question_sn,
            class_id=record.class_id,
            semester=record.semester,
        )
        existing = self._records.get(key)
        if existing and existing.get("manual_override", False):
            return
        is_new = key not in self._records
        self._records[key] = self._to_dict(record)
        if is_new:
            self._by_student.setdefault(record.student_id, []).append(key)
            self._by_week.setdefault(record.week, []).append(key)

    def get_student_history(self, student_id: str) -> list[LongitudinalRecord]:
        """Return all records for a given student."""
        keys = self._by_student.get(student_id, [])
        return [self._to_record(self._records[k]) for k in keys if k in self._records]

    def get_all_records(self) -> list[LongitudinalRecord]:
        """Return all records in the store."""
        return [self._to_record(d) for d in self._records.values()]

    def save(self) -> None:
        """Atomic write with .bak backup and file locking."""
        data = {"records": self._records}
        dir_name = os.path.dirname(self.store_path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        lock_file = open(self._lock_path, "a")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            with os.fdopen(fd, "w") as f:
                yaml.dump(data, f, Dumper=FormaDumper, default_flow_style=False, allow_unicode=True, sort_keys=False)
            os.replace(tmp_path, self.store_path)
            try:
                shutil.copy2(str(self.store_path), str(self.store_path) + ".bak")
            except OSError:
                pass
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

    def load(self) -> None:
        """Load records from YAML file. Initializes empty if file doesn't exist."""
        if not os.path.exists(self.store_path):
            self._records = {}
            self._rebuild_index()
            return
        lock_file = open(self._lock_path, "a")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
            with open(self.store_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._records = data.get("records", {}) if data else {}
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
        self._rebuild_index()

    def get_class_snapshot(self, week: int) -> list[LongitudinalRecord]:
        """Return all records for a given week, sorted by student_id."""
        keys = self._by_week.get(week, [])
        records = [self._to_record(self._records[k]) for k in keys if k in self._records]
        records.sort(key=lambda r: r.student_id)
        return records

    def get_student_trajectory(self, student_id: str, metric: str) -> list[tuple[int, float]]:
        """Return [(week, value)] for a student's metric, sorted by week.

        When a student has multiple questions in a single week, the metric
        values are averaged across questions for that week.

        Looks up *metric* first in the ``scores`` sub-dict, then as a
        top-level record field (e.g. ``ocr_confidence_mean``).
        """
        week_values: dict[int, list[float]] = defaultdict(list)
        for key in self._by_student.get(student_id, []):
            d = self._records.get(key)
            if d is None:
                continue
            # Try scores sub-dict first, then top-level field
            val = d["scores"].get(metric)
            if val is None:
                val = d.get(metric)
            if val is not None:
                week_values[d["week"]].append(val)
        result = [(wk, sum(vals) / len(vals)) for wk, vals in sorted(week_values.items())]
        return result

    def get_class_weekly_matrix(self, metric: str) -> dict[str, dict[int, float]]:
        """Return {student_id: {week: value}} for a given metric.

        When a student has multiple questions in a single week, the metric
        values are averaged across questions for that week.
        Only students with at least one matching metric value are included.
        """
        matrix: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for d in self._records.values():
            val = d["scores"].get(metric)
            # Fallback: if ensemble_score not stored, compute from components
            if val is None and metric == "ensemble_score":
                component_vals = [
                    v for k, v in d["scores"].items() if isinstance(v, (int, float)) and k != "ensemble_score"
                ]
                if component_vals:
                    val = sum(component_vals) / len(component_vals)
            if val is not None:
                matrix[d["student_id"]][d["week"]].append(val)
        return {sid: {wk: sum(vs) / len(vs) for wk, vs in sorted(weeks.items())} for sid, weeks in matrix.items()}

    def get_topic_weekly_matrix(self, metric: str) -> dict[str, dict[str, dict[int, float]]]:
        """Return {student_id: {topic: {week: avg_score}}}.

        Groups records by topic and averages same-topic questions
        per week. Records without a topic are excluded.

        Args:
            metric: Score metric name (e.g. "ensemble_score").

        Returns:
            Nested dict: student -> topic -> week -> avg score.
        """
        # {sid: {topic: {week: [values]}}}
        raw: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for d in self._records.values():
            topic = d.get("topic")
            if topic is None:
                continue
            val = d["scores"].get(metric)
            if val is None:
                continue
            sid = d["student_id"]
            wk = d["week"]
            raw[sid][topic][wk].append(val)

        return {
            sid: {topic: {wk: sum(vs) / len(vs) for wk, vs in sorted(weeks.items())} for topic, weeks in topics.items()}
            for sid, topics in raw.items()
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
    return {concept: sum(1 for v in vals if v) / len(vals) for concept, vals in counts.items()}


def snapshot_from_evaluation(
    store: LongitudinalStore,
    ensemble_results: dict,
    graph_metric_results: dict,
    graph_comparison_results: dict,
    layer1_results: list,
    week: int,
    exam_file: str,
    ocr_confidence: dict | None = None,
    id_map: dict[str, str] | None = None,
    topics: dict[int, str] | None = None,
    class_id: str | None = None,
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
        ocr_confidence: {student_id: {qsn: {"mean": float, "min": float}}}.
        id_map: Optional mapping of anonymous ID to real student ID (학번).
            When provided, student_id is replaced with the real ID for
            longitudinal tracking across weeks.
        topics: Optional mapping of question_sn to topic string.
        class_id: Optional section identifier (e.g. "A").
    """
    recorded_at = datetime.now(timezone.utc).isoformat()

    for student_id, q_results in ensemble_results.items():
        real_id = id_map.get(student_id, student_id) if id_map else student_id
        for qsn, er in q_results.items():
            # Graph metric results (node_recall)
            gmr = (graph_metric_results.get(student_id) or {}).get(qsn)
            node_recall = gmr.node_recall if gmr else None

            # Graph comparison results (edge_f1, misconception_count)
            gcr = (graph_comparison_results.get(student_id) or {}).get(qsn)
            edge_f1 = gcr.f1 if gcr else None
            misconception_count = len(gcr.wrong_direction_edges) if gcr else None

            # Concept scores from layer1
            concept_scores = _compute_concept_scores(layer1_results, student_id, qsn)

            # OCR confidence from scan/join pipeline
            ocr_conf = (ocr_confidence.get(student_id) or {}).get(qsn) if ocr_confidence else None
            ocr_mean = ocr_conf["mean"] if ocr_conf else None
            ocr_min = ocr_conf["min"] if ocr_conf else None

            # Include ensemble_score in scores dict so longitudinal
            # queries like get_class_weekly_matrix("ensemble_score") work.
            scores_with_ensemble = dict(er.component_scores)
            if hasattr(er, "ensemble_score") and er.ensemble_score is not None:
                scores_with_ensemble["ensemble_score"] = er.ensemble_score

            topic = topics.get(qsn) if topics else None

            record = LongitudinalRecord(
                student_id=real_id,
                week=week,
                question_sn=qsn,
                scores=scores_with_ensemble,
                tier_level=0,
                tier_label=er.understanding_level,
                node_recall=node_recall,
                edge_f1=edge_f1,
                misconception_count=misconception_count,
                concept_scores=concept_scores,
                exam_file=os.path.basename(exam_file),
                recorded_at=recorded_at,
                ocr_confidence_mean=ocr_mean,
                ocr_confidence_min=ocr_min,
                topic=topic,
                class_id=class_id,
            )
            store.add_record(record)
