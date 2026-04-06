"""Tests for OCR confidence in longitudinal data (016-ocr-confidence Phase 5).

Covers:
  - LongitudinalRecord.ocr_confidence_mean/min optional fields
  - Backward compat: legacy YAML without confidence fields
  - _to_dict / _to_record round-trip with confidence
  - get_student_trajectory("ocr_confidence_mean") returns weekly trend
  - OCR confidence trend chart generation
"""

from __future__ import annotations

import io
import os

import pytest

from forma.evaluation_types import LongitudinalRecord


PNG_HEADER = b"\x89PNG"


# ──────────────────────────────────────────────────────────────
# Group 1: LongitudinalRecord confidence fields
# ──────────────────────────────────────────────────────────────


class TestLongitudinalRecordConfidence:
    """LongitudinalRecord has optional ocr_confidence_mean/min fields."""

    def test_default_is_none(self):
        """ocr_confidence_mean/min default to None."""
        rec = LongitudinalRecord(
            student_id="S001",
            week=1,
            question_sn=1,
            scores={"ensemble": 0.7},
            tier_level=2,
            tier_label="Proficient",
        )
        assert rec.ocr_confidence_mean is None
        assert rec.ocr_confidence_min is None

    def test_accepts_float_values(self):
        """ocr_confidence_mean/min accept float values."""
        rec = LongitudinalRecord(
            student_id="S001",
            week=1,
            question_sn=1,
            scores={"ensemble": 0.7},
            tier_level=2,
            tier_label="Proficient",
            ocr_confidence_mean=0.87,
            ocr_confidence_min=0.62,
        )
        assert rec.ocr_confidence_mean == 0.87
        assert rec.ocr_confidence_min == 0.62

    def test_backward_compat_existing_code(self):
        """Existing code creating LongitudinalRecord without confidence still works."""
        rec = LongitudinalRecord(
            student_id="S001",
            week=1,
            question_sn=1,
            scores={"ensemble": 0.7},
            tier_level=2,
            tier_label="Proficient",
            node_recall=0.8,
            edge_f1=0.6,
        )
        assert rec.node_recall == 0.8
        assert rec.ocr_confidence_mean is None


# ──────────────────────────────────────────────────────────────
# Group 2: LongitudinalStore _to_dict / _to_record with confidence
# ──────────────────────────────────────────────────────────────


class TestLongitudinalStoreConfidenceRoundTrip:
    """_to_dict / _to_record correctly serialize confidence fields."""

    def test_to_dict_includes_confidence_when_set(self, tmp_path):
        """_to_dict includes ocr_confidence_mean/min when non-None."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        rec = LongitudinalRecord(
            student_id="S001",
            week=1,
            question_sn=1,
            scores={"ensemble": 0.7},
            tier_level=2,
            tier_label="Proficient",
            ocr_confidence_mean=0.85,
            ocr_confidence_min=0.60,
        )
        d = store._to_dict(rec)
        assert d["ocr_confidence_mean"] == 0.85
        assert d["ocr_confidence_min"] == 0.60

    def test_to_dict_omits_confidence_when_none(self, tmp_path):
        """_to_dict omits confidence fields when None (keeps YAML clean)."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        rec = LongitudinalRecord(
            student_id="S001",
            week=1,
            question_sn=1,
            scores={"ensemble": 0.7},
            tier_level=2,
            tier_label="Proficient",
        )
        d = store._to_dict(rec)
        assert "ocr_confidence_mean" not in d
        assert "ocr_confidence_min" not in d

    def test_to_record_loads_confidence(self, tmp_path):
        """_to_record correctly loads confidence from dict."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        d = {
            "student_id": "S001",
            "week": 1,
            "question_sn": 1,
            "scores": {"ensemble": 0.7},
            "tier_level": 2,
            "tier_label": "Proficient",
            "ocr_confidence_mean": 0.85,
            "ocr_confidence_min": 0.60,
        }
        rec = store._to_record(d)
        assert rec.ocr_confidence_mean == 0.85
        assert rec.ocr_confidence_min == 0.60

    def test_to_record_legacy_yaml_no_confidence(self, tmp_path):
        """_to_record from legacy YAML (no confidence fields) → None."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        d = {
            "student_id": "S001",
            "week": 1,
            "question_sn": 1,
            "scores": {"ensemble": 0.7},
            "tier_level": 2,
            "tier_label": "Proficient",
        }
        rec = store._to_record(d)
        assert rec.ocr_confidence_mean is None
        assert rec.ocr_confidence_min is None


# ──────────────────────────────────────────────────────────────
# Group 3: get_student_trajectory for confidence metric
# ──────────────────────────────────────────────────────────────


class TestGetStudentTrajectoryConfidence:
    """get_student_trajectory supports ocr_confidence_mean."""

    def test_trajectory_returns_weekly_confidence(self, tmp_path):
        """Trajectory returns [(week, mean_confidence)] for ocr_confidence_mean."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))

        for week, conf in [(1, 0.85), (2, 0.78), (3, 0.90)]:
            rec = LongitudinalRecord(
                student_id="S001",
                week=week,
                question_sn=1,
                scores={"ensemble": 0.7},
                tier_level=2,
                tier_label="Proficient",
                ocr_confidence_mean=conf,
            )
            store.add_record(rec)

        traj = store.get_student_trajectory("S001", "ocr_confidence_mean")
        assert traj == [(1, 0.85), (2, 0.78), (3, 0.90)]

    def test_trajectory_skips_weeks_without_confidence(self, tmp_path):
        """Weeks where ocr_confidence_mean is None are excluded."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))

        # Week 1: has confidence, Week 2: no confidence
        store.add_record(
            LongitudinalRecord(
                student_id="S001",
                week=1,
                question_sn=1,
                scores={"ensemble": 0.7},
                tier_level=2,
                tier_label="Proficient",
                ocr_confidence_mean=0.85,
            )
        )
        store.add_record(
            LongitudinalRecord(
                student_id="S001",
                week=2,
                question_sn=1,
                scores={"ensemble": 0.7},
                tier_level=2,
                tier_label="Proficient",
            )
        )

        traj = store.get_student_trajectory("S001", "ocr_confidence_mean")
        assert traj == [(1, 0.85)]

    def test_trajectory_averages_across_questions(self, tmp_path):
        """Multiple questions in same week are averaged."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))

        store.add_record(
            LongitudinalRecord(
                student_id="S001",
                week=1,
                question_sn=1,
                scores={"ensemble": 0.7},
                tier_level=2,
                tier_label="Proficient",
                ocr_confidence_mean=0.80,
            )
        )
        store.add_record(
            LongitudinalRecord(
                student_id="S001",
                week=1,
                question_sn=2,
                scores={"ensemble": 0.7},
                tier_level=2,
                tier_label="Proficient",
                ocr_confidence_mean=0.90,
            )
        )

        traj = store.get_student_trajectory("S001", "ocr_confidence_mean")
        assert len(traj) == 1
        assert traj[0][0] == 1
        assert traj[0][1] == pytest.approx(0.85)


# ──────────────────────────────────────────────────────────────
# Group 4: OCR confidence trend chart
# ──────────────────────────────────────────────────────────────


class TestOcrConfidenceTrendChart:
    """Tests for build_ocr_confidence_trend_chart()."""

    def test_chart_returns_png(self):
        """Trend chart returns valid PNG BytesIO."""
        from forma.longitudinal_report_charts import build_ocr_confidence_trend_chart

        trajectories = {
            "S001": [(1, 0.85), (2, 0.78), (3, 0.90)],
            "S002": [(1, 0.60), (2, 0.55), (3, 0.50)],
        }
        buf = build_ocr_confidence_trend_chart(trajectories)
        assert isinstance(buf, io.BytesIO)
        buf.seek(0)
        assert buf.read(4) == PNG_HEADER

    def test_chart_empty_trajectories(self):
        """Empty trajectories dict returns placeholder PNG."""
        from forma.longitudinal_report_charts import build_ocr_confidence_trend_chart

        buf = build_ocr_confidence_trend_chart({})
        assert isinstance(buf, io.BytesIO)
        buf.seek(0)
        assert buf.read(4) == PNG_HEADER

    def test_chart_highlights_consecutive_low(self):
        """3 consecutive weeks below threshold marked in red (no crash)."""
        from forma.longitudinal_report_charts import build_ocr_confidence_trend_chart

        trajectories = {
            "S001": [(1, 0.60), (2, 0.55), (3, 0.50)],  # 3 weeks < 0.75
        }
        buf = build_ocr_confidence_trend_chart(
            trajectories,
            threshold=0.75,
        )
        assert isinstance(buf, io.BytesIO)
        buf.seek(0)
        assert buf.read(4) == PNG_HEADER


# ──────────────────────────────────────────────────────────────
# Group 5: snapshot_from_evaluation with OCR confidence
# ──────────────────────────────────────────────────────────────


class TestLongitudinalReportOcrSection:
    """Longitudinal report PDF includes OCR confidence trend section."""

    @staticmethod
    def _make_summary_data():
        from forma.longitudinal_report_data import (
            LongitudinalSummaryData,
            StudentTrajectory,
        )

        return LongitudinalSummaryData(
            class_name="1A",
            period_weeks=[1, 2, 3],
            student_trajectories=[
                StudentTrajectory("S001", {1: 0.7, 2: 0.75, 3: 0.8}, 0.05, False, []),
            ],
            class_weekly_averages={1: 0.7, 2: 0.75, 3: 0.8},
            persistent_risk_students=[],
            concept_mastery_changes=[],
            total_students=1,
        )

    def test_report_includes_ocr_section_when_data_present(self, tmp_path):
        """generate_pdf with ocr_confidence_trajectories produces PDF with OCR section."""
        from forma.longitudinal_report import LongitudinalPDFReportGenerator

        data = self._make_summary_data()
        ocr_trajectories = {
            "S001": [(1, 0.85), (2, 0.78), (3, 0.90)],
        }

        gen = LongitudinalPDFReportGenerator()
        out = str(tmp_path / "report.pdf")
        result = gen.generate_pdf(
            data,
            out,
            ocr_confidence_trajectories=ocr_trajectories,
        )
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_report_omits_ocr_section_when_no_data(self, tmp_path):
        """generate_pdf without ocr_confidence_trajectories still works."""
        from forma.longitudinal_report import LongitudinalPDFReportGenerator

        data = self._make_summary_data()
        gen = LongitudinalPDFReportGenerator()
        out = str(tmp_path / "report.pdf")
        result = gen.generate_pdf(data, out)
        assert os.path.exists(result)

    def test_report_omits_ocr_section_when_empty_trajectories(self, tmp_path):
        """Empty trajectories dict → no OCR section, no crash."""
        from forma.longitudinal_report import LongitudinalPDFReportGenerator

        data = self._make_summary_data()
        gen = LongitudinalPDFReportGenerator()
        out = str(tmp_path / "report.pdf")
        result = gen.generate_pdf(data, out, ocr_confidence_trajectories={})
        assert os.path.exists(result)


class TestSnapshotFromEvaluationOcrConfidence:
    """snapshot_from_evaluation passes OCR confidence to LongitudinalRecord."""

    def _make_ensemble_result(self):
        """Create minimal EnsembleResult-like object."""
        from types import SimpleNamespace

        return SimpleNamespace(
            component_scores={"ensemble": 0.75},
            understanding_level="Proficient",
        )

    def test_snapshot_passes_confidence_to_record(self, tmp_path):
        """snapshot_from_evaluation creates records with ocr_confidence when provided."""
        from forma.longitudinal_store import LongitudinalStore, snapshot_from_evaluation

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        ensemble_results = {"S001": {1: self._make_ensemble_result()}}

        ocr_confidence = {
            "S001": {1: {"mean": 0.85, "min": 0.62}},
        }

        snapshot_from_evaluation(
            store=store,
            ensemble_results=ensemble_results,
            graph_metric_results={},
            graph_comparison_results={},
            layer1_results=[],
            week=1,
            exam_file="exam.yaml",
            ocr_confidence=ocr_confidence,
        )

        records = store.get_all_records()
        assert len(records) == 1
        assert records[0].ocr_confidence_mean == 0.85
        assert records[0].ocr_confidence_min == 0.62

    def test_snapshot_without_confidence_backward_compat(self, tmp_path):
        """snapshot_from_evaluation without ocr_confidence still works (None)."""
        from forma.longitudinal_store import LongitudinalStore, snapshot_from_evaluation

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        ensemble_results = {"S001": {1: self._make_ensemble_result()}}

        snapshot_from_evaluation(
            store=store,
            ensemble_results=ensemble_results,
            graph_metric_results={},
            graph_comparison_results={},
            layer1_results=[],
            week=1,
            exam_file="exam.yaml",
        )

        records = store.get_all_records()
        assert len(records) == 1
        assert records[0].ocr_confidence_mean is None
        assert records[0].ocr_confidence_min is None

    def test_snapshot_missing_student_in_confidence(self, tmp_path):
        """Student not in ocr_confidence dict gets None confidence."""
        from forma.longitudinal_store import LongitudinalStore, snapshot_from_evaluation

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        ensemble_results = {"S001": {1: self._make_ensemble_result()}}

        ocr_confidence = {
            "S999": {1: {"mean": 0.85, "min": 0.62}},
        }

        snapshot_from_evaluation(
            store=store,
            ensemble_results=ensemble_results,
            graph_metric_results={},
            graph_comparison_results={},
            layer1_results=[],
            week=1,
            exam_file="exam.yaml",
            ocr_confidence=ocr_confidence,
        )

        records = store.get_all_records()
        assert records[0].ocr_confidence_mean is None
        assert records[0].ocr_confidence_min is None
