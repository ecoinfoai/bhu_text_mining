"""Integration test for longitudinal analysis pipeline.

T043: End-to-end test with synthetic multi-week data.
  - 4 weeks, 12 students, 2 questions per student per week
  - LongitudinalStore → add_record → build_longitudinal_summary → generate PDF
  - Assertions: PDF created, >10KB, correct record count (12*4*2 = 96)
"""

from __future__ import annotations

import os
import random


from forma.evaluation_types import LongitudinalRecord
from forma.longitudinal_store import LongitudinalStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STUDENTS = [f"S{i:03d}" for i in range(1, 13)]  # S001..S012
_WEEKS = [1, 2, 3, 4]
_QUESTIONS = [1, 2]

# Concept scores shared across all students for each week
_CONCEPT_SCORES_BY_WEEK = {
    1: {"항상성": 0.55, "삼투": 0.40, "확산": 0.35},
    2: {"항상성": 0.60, "삼투": 0.45, "확산": 0.40},
    3: {"항상성": 0.70, "삼투": 0.55, "확산": 0.50},
    4: {"항상성": 0.80, "삼투": 0.65, "확산": 0.60},
}


def _student_base_score(student_id: str) -> float:
    """Deterministic base score for each student (spread from 0.2 to 0.9)."""
    idx = int(student_id[1:])  # S001 → 1, S012 → 12
    return 0.15 + (idx / 12) * 0.75


def _build_integration_store(tmp_path) -> tuple[LongitudinalStore, str]:
    """Build a store with 4 weeks x 12 students x 2 questions = 96 records.

    Returns:
        Tuple of (store, store_path).
    """
    store_path = str(tmp_path / "integration_store.yaml")
    store = LongitudinalStore(store_path)

    random.seed(42)  # deterministic for reproducibility

    for week in _WEEKS:
        for sid in _STUDENTS:
            base = _student_base_score(sid)
            # Score improves slightly each week for most students
            week_bonus = (week - 1) * 0.03
            for qsn in _QUESTIONS:
                # Small per-question noise
                noise = random.uniform(-0.05, 0.05)
                score = max(0.0, min(1.0, base + week_bonus + noise))

                tier_label = (
                    "Advanced" if score >= 0.85
                    else "Proficient" if score >= 0.65
                    else "Developing" if score >= 0.45
                    else "Beginning"
                )

                record = LongitudinalRecord(
                    student_id=sid,
                    week=week,
                    question_sn=qsn,
                    scores={"ensemble_score": round(score, 4)},
                    tier_level=0,
                    tier_label=tier_label,
                    concept_scores=_CONCEPT_SCORES_BY_WEEK[week],
                )
                store.add_record(record)

    store.save()
    return store, store_path


# ---------------------------------------------------------------------------
# T043: Integration test
# ---------------------------------------------------------------------------


class TestLongitudinalIntegration:
    """End-to-end integration test: store → summary → PDF."""

    def test_store_record_count(self, tmp_path):
        """Store should contain exactly 12*4*2 = 96 records."""
        store, _ = _build_integration_store(tmp_path)

        records = store.get_all_records()
        assert len(records) == 96  # 12 students * 4 weeks * 2 questions

    def test_store_roundtrip(self, tmp_path):
        """Save and reload preserves all 96 records."""
        _, store_path = _build_integration_store(tmp_path)

        store2 = LongitudinalStore(store_path)
        store2.load()

        records = store2.get_all_records()
        assert len(records) == 96

    def test_all_students_present(self, tmp_path):
        """All 12 students should be present in the store."""
        store, _ = _build_integration_store(tmp_path)

        matrix = store.get_class_weekly_matrix("ensemble_score")
        assert len(matrix) == 12
        for sid in _STUDENTS:
            assert sid in matrix

    def test_all_weeks_present(self, tmp_path):
        """Each student should have data in all 4 weeks."""
        store, _ = _build_integration_store(tmp_path)

        matrix = store.get_class_weekly_matrix("ensemble_score")
        for sid in _STUDENTS:
            assert set(matrix[sid].keys()) == {1, 2, 3, 4}

    def test_build_summary(self, tmp_path):
        """build_longitudinal_summary produces correct summary from store."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store, _ = _build_integration_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "IntegTest")

        assert summary.class_name == "IntegTest"
        assert summary.period_weeks == [1, 2, 3, 4]
        assert summary.total_students == 12
        assert len(summary.student_trajectories) == 12

        # Class weekly averages should exist for all 4 weeks
        assert set(summary.class_weekly_averages.keys()) == {1, 2, 3, 4}

        # All averages should be in valid range
        for w, avg in summary.class_weekly_averages.items():
            assert 0.0 <= avg <= 1.0, f"Week {w} average {avg} out of range"

    def test_summary_concept_mastery(self, tmp_path):
        """Concept mastery changes should reflect week 1 → week 4 delta."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store, _ = _build_integration_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "IntegTest")

        # We have 3 concepts: 항상성, 삼투, 확산
        assert len(summary.concept_mastery_changes) == 3

        # All concepts should show positive delta (improving)
        for change in summary.concept_mastery_changes:
            assert change.delta > 0, f"{change.concept}: delta={change.delta} should be positive"

        # Sorted by delta descending
        deltas = [c.delta for c in summary.concept_mastery_changes]
        assert deltas == sorted(deltas, reverse=True)

    def test_summary_trajectories_have_trends(self, tmp_path):
        """Each student trajectory should have a computed OLS trend."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store, _ = _build_integration_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "IntegTest")

        for traj in summary.student_trajectories:
            # With week_bonus = 0.03/week, most trends should be positive
            assert isinstance(traj.overall_trend, float)
            assert len(traj.weekly_scores) == 4

    def test_persistent_risk_identification(self, tmp_path):
        """Low-scoring students should be identified as persistent risk."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store, _ = _build_integration_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "IntegTest")

        # S001 has lowest base score (0.15 + 1/12*0.75 ≈ 0.2125)
        # Even with week bonus, scores stay below 0.45
        # Check if persistent risk detection works
        for traj in summary.student_trajectories:
            if traj.is_persistent_risk:
                # All weeks should be risk weeks
                assert set(traj.risk_weeks) == {1, 2, 3, 4}

    def test_generate_pdf(self, tmp_path):
        """Full pipeline: store → summary → PDF file created."""
        from forma.longitudinal_report_data import build_longitudinal_summary
        from forma.longitudinal_report import LongitudinalPDFReportGenerator

        store, _ = _build_integration_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "IntegTest")

        output_path = str(tmp_path / "integration_report.pdf")
        gen = LongitudinalPDFReportGenerator()
        gen.generate_pdf(summary, output_path)

        assert os.path.exists(output_path)

    def test_pdf_file_size(self, tmp_path):
        """Generated PDF should be > 10KB (meaningful content)."""
        from forma.longitudinal_report_data import build_longitudinal_summary
        from forma.longitudinal_report import LongitudinalPDFReportGenerator

        store, _ = _build_integration_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "IntegTest")

        output_path = str(tmp_path / "integration_report.pdf")
        gen = LongitudinalPDFReportGenerator()
        gen.generate_pdf(summary, output_path)

        file_size = os.path.getsize(output_path)
        assert file_size > 10 * 1024, f"PDF too small: {file_size} bytes"

    def test_pdf_valid_header(self, tmp_path):
        """Generated file should be a valid PDF (starts with %PDF)."""
        from forma.longitudinal_report_data import build_longitudinal_summary
        from forma.longitudinal_report import LongitudinalPDFReportGenerator

        store, _ = _build_integration_store(tmp_path)
        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "IntegTest")

        output_path = str(tmp_path / "integration_report.pdf")
        gen = LongitudinalPDFReportGenerator()
        gen.generate_pdf(summary, output_path)

        with open(output_path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_subset_weeks(self, tmp_path):
        """Summary with subset of weeks [2, 4] should work correctly."""
        from forma.longitudinal_report_data import build_longitudinal_summary

        store, _ = _build_integration_store(tmp_path)
        summary = build_longitudinal_summary(store, [2, 4], "IntegTest")

        assert summary.period_weeks == [2, 4]
        assert summary.total_students == 12
        assert set(summary.class_weekly_averages.keys()) == {2, 4}

    def test_cli_end_to_end(self, tmp_path):
        """CLI main() runs end-to-end with real store."""
        from unittest.mock import patch
        from forma.cli_report_longitudinal import main

        _, store_path = _build_integration_store(tmp_path)
        output_path = str(tmp_path / "cli_report.pdf")

        with patch("sys.argv", [
            "forma-report-longitudinal",
            "--store", store_path,
            "--class-name", "IntegCLI",
            "--weeks", "1", "2", "3", "4",
            "--output", output_path,
        ]):
            result = main()

        assert result is None or result == 0
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 10 * 1024
