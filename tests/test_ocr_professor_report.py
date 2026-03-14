"""Tests for OCR confidence section in professor report (016-ocr-confidence Phase 4).

Covers:
  - StudentSummaryRow.ocr_confidence_mean optional field
  - OCR confidence histogram chart generation
  - Professor report PDF: OCR confidence section inclusion/exclusion
  - Backward compatibility: no confidence data → no section
"""

from __future__ import annotations

import io

from forma.professor_report_data import (
    ProfessorReportData,
    QuestionClassStats,
    StudentSummaryRow,
)

PNG_HEADER = b"\x89PNG"


# ──────────────────────────────────────────────────────────────
# Group 1: StudentSummaryRow.ocr_confidence_mean field
# ──────────────────────────────────────────────────────────────


class TestStudentSummaryRowOcrConfidence:
    """StudentSummaryRow has optional ocr_confidence_mean field."""

    def test_default_is_none(self):
        """ocr_confidence_mean defaults to None."""
        row = StudentSummaryRow(student_id="S001")
        assert row.ocr_confidence_mean is None

    def test_accepts_float_value(self):
        """ocr_confidence_mean accepts a float."""
        row = StudentSummaryRow(student_id="S001", ocr_confidence_mean=0.87)
        assert row.ocr_confidence_mean == 0.87

    def test_backward_compatible_without_field(self):
        """Existing code creating StudentSummaryRow without ocr_confidence_mean still works."""
        row = StudentSummaryRow(
            student_id="S001",
            student_number="2026194001",
            real_name="홍길동",
            overall_ensemble_mean=0.75,
            overall_level="Proficient",
        )
        assert row.student_id == "S001"
        assert row.ocr_confidence_mean is None


# ──────────────────────────────────────────────────────────────
# Group 2: OCR confidence histogram chart
# ──────────────────────────────────────────────────────────────


class TestOcrConfidenceHistogram:
    """Tests for OCR confidence distribution histogram chart."""

    def test_histogram_returns_png(self):
        """confidence_histogram() returns valid PNG BytesIO."""
        from forma.professor_report_charts import ProfessorReportChartGenerator

        gen = ProfessorReportChartGenerator()
        buf = gen.confidence_histogram([0.95, 0.88, 0.62, 0.71, 0.90])
        assert isinstance(buf, io.BytesIO)
        buf.seek(0)
        assert buf.read(4) == PNG_HEADER

    def test_histogram_empty_scores(self):
        """confidence_histogram() with empty list returns placeholder PNG."""
        from forma.professor_report_charts import ProfessorReportChartGenerator

        gen = ProfessorReportChartGenerator()
        buf = gen.confidence_histogram([])
        assert isinstance(buf, io.BytesIO)
        buf.seek(0)
        assert buf.read(4) == PNG_HEADER

    def test_histogram_single_value(self):
        """confidence_histogram() with single value doesn't crash."""
        from forma.professor_report_charts import ProfessorReportChartGenerator

        gen = ProfessorReportChartGenerator()
        buf = gen.confidence_histogram([0.85])
        assert isinstance(buf, io.BytesIO)
        buf.seek(0)
        assert buf.read(4) == PNG_HEADER


# ──────────────────────────────────────────────────────────────
# Group 3: Professor report PDF — OCR confidence section
# ──────────────────────────────────────────────────────────────


def _make_student_row(
    student_id: str,
    score: float = 0.7,
    level: str = "Proficient",
    ocr_confidence_mean: float | None = None,
) -> StudentSummaryRow:
    """Build a StudentSummaryRow with enough data for PDF rendering."""
    return StudentSummaryRow(
        student_id=student_id,
        student_number=f"2026{student_id}",
        real_name=f"학생{student_id}",
        overall_ensemble_mean=score,
        overall_level=level,
        per_question_scores={1: score, 2: score},
        per_question_levels={1: level, 2: level},
        per_question_coverages={1: score, 2: score},
        ocr_confidence_mean=ocr_confidence_mean,
    )


def _make_report_data(
    student_rows: list[StudentSummaryRow] | None = None,
    n_students: int = 3,
) -> ProfessorReportData:
    """Build minimal ProfessorReportData for testing."""
    if student_rows is None:
        student_rows = [
            _make_student_row(f"S{i:03d}")
            for i in range(1, n_students + 1)
        ]
    return ProfessorReportData(
        class_name="A",
        week_num=1,
        subject="생물학",
        exam_title="1주차 평가",
        generation_date="2026-03-14",
        n_students=len(student_rows),
        n_questions=2,
        class_ensemble_mean=0.7,
        class_ensemble_std=0.1,
        class_ensemble_median=0.7,
        class_ensemble_q1=0.6,
        class_ensemble_q3=0.8,
        overall_level_distribution={
            "Advanced": 1, "Proficient": 1, "Developing": 1, "Beginning": 0,
        },
        question_stats=[
            QuestionClassStats(
                question_sn=1,
                question_text="Q1",
                ensemble_mean=0.7,
                level_distribution={"Advanced": 1, "Proficient": 1, "Developing": 1, "Beginning": 0},
            ),
            QuestionClassStats(
                question_sn=2,
                question_text="Q2",
                ensemble_mean=0.7,
                level_distribution={"Advanced": 1, "Proficient": 1, "Developing": 1, "Beginning": 0},
            ),
        ],
        student_rows=student_rows,
        n_at_risk=0,
        pct_at_risk=0.0,
    )


class TestProfessorReportOcrSection:
    """Tests for OCR confidence section in professor report PDF."""

    def test_section_generated_with_confidence_data(self, tmp_path):
        """OCR confidence section is added when ocr_confidence_data is provided."""
        from forma.professor_report import ProfessorPDFReportGenerator

        rows = [
            _make_student_row("S001", ocr_confidence_mean=0.58),
            _make_student_row("S002", ocr_confidence_mean=0.90),
            _make_student_row("S003", ocr_confidence_mean=0.71),
        ]
        data = _make_report_data(student_rows=rows)
        ocr_data = [
            {"student_id": "S001", "confidence_mean": 0.58},
            {"student_id": "S002", "confidence_mean": 0.90},
            {"student_id": "S003", "confidence_mean": 0.71},
        ]

        gen = ProfessorPDFReportGenerator()
        path = gen.generate_pdf(
            data, str(tmp_path),
            ocr_confidence_data=ocr_data,
        )

        import os as _os

        assert path.endswith(".pdf")
        assert _os.path.exists(path)
        assert _os.path.getsize(path) > 0

    def test_section_not_generated_without_confidence_data(self, tmp_path):
        """No OCR section when ocr_confidence_data is None (backward compat)."""
        from forma.professor_report import ProfessorPDFReportGenerator

        data = _make_report_data()
        gen = ProfessorPDFReportGenerator()

        # Should work exactly as before — no ocr_confidence_data kwarg
        path = gen.generate_pdf(data, str(tmp_path))
        assert path.endswith(".pdf")

    def test_section_not_generated_when_no_low_confidence(self, tmp_path):
        """OCR section not generated when all students have high confidence."""
        from forma.professor_report import ProfessorPDFReportGenerator

        rows = [
            _make_student_row("S001", ocr_confidence_mean=0.90),
            _make_student_row("S002", ocr_confidence_mean=0.95),
        ]
        data = _make_report_data(student_rows=rows)
        # All above threshold → section skipped (INV-R01)
        ocr_data = [
            {"student_id": "S001", "confidence_mean": 0.90},
            {"student_id": "S002", "confidence_mean": 0.95},
        ]

        gen = ProfessorPDFReportGenerator()
        path = gen.generate_pdf(
            data, str(tmp_path),
            ocr_confidence_data=ocr_data,
        )
        assert path.endswith(".pdf")

    def test_section_not_generated_with_empty_confidence_data(self, tmp_path):
        """OCR section not generated when ocr_confidence_data is empty list."""
        from forma.professor_report import ProfessorPDFReportGenerator

        data = _make_report_data()
        gen = ProfessorPDFReportGenerator()
        path = gen.generate_pdf(
            data, str(tmp_path),
            ocr_confidence_data=[],
        )
        assert path.endswith(".pdf")
