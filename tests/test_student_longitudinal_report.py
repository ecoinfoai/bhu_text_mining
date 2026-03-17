"""Tests for student_longitudinal_report.py — PDF report generation.

TDD RED phase: tests written before implementation.
T016: StudentLongitudinalPDFReportGenerator.
"""

from __future__ import annotations

import os

from forma.student_longitudinal_data import (
    AlertLevel,
    CohortDistribution,
    CohortWeekStats,
    StudentLongitudinalData,
    WarningSignal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_student_data(
    weeks: list[int] | None = None,
    student_id: str = "s001",
    student_name: str = "홍길동",
    class_name: str = "A",
    include_q2: bool = True,
) -> StudentLongitudinalData:
    """Build minimal StudentLongitudinalData for report tests."""
    if weeks is None:
        weeks = [1, 2]
    scores_by_week = {}
    for w in weeks:
        q_map = {
            1: {
                "concept_coverage": 0.5 + 0.05 * w,
                "llm_rubric": 0.4 + 0.05 * w,
                "ensemble_score": 0.45 + 0.05 * w,
                "rasch_ability": -0.2 + 0.1 * w,
            },
        }
        if include_q2:
            q_map[2] = {
                "concept_coverage": 0.4 + 0.05 * w,
                "llm_rubric": 0.35 + 0.05 * w,
                "ensemble_score": 0.40 + 0.05 * w,
                "rasch_ability": -0.3 + 0.1 * w,
            }
        scores_by_week[w] = q_map
    percentiles = {w: 30.0 + 5.0 * w for w in weeks}
    return StudentLongitudinalData(
        student_id=student_id,
        student_name=student_name,
        class_name=class_name,
        weeks=weeks,
        scores_by_week=scores_by_week,
        trend_slope=0.05 if len(weeks) >= 2 else None,
        trend_direction="상승" if len(weeks) >= 2 else "데이터 부족",
        percentiles_by_week=percentiles,
    )


def _make_cohort(weeks: list[int] | None = None) -> CohortDistribution:
    if weeks is None:
        weeks = [1, 2]
    cohort = CohortDistribution()
    import numpy as np

    for w in weeks:
        scores = [0.3 + 0.05 * i + 0.02 * w for i in range(10)]
        cohort.weekly_scores[w] = scores
        cohort.weekly_q_scores[w] = {
            1: [0.4 + 0.05 * i + 0.02 * w for i in range(10)],
            2: [0.3 + 0.05 * i + 0.02 * w for i in range(10)],
        }
        arr = np.array(scores)
        cohort.weekly_stats[w] = CohortWeekStats(
            week=w,
            median=float(np.median(arr)),
            q1=float(np.percentile(arr, 25)),
            q3=float(np.percentile(arr, 75)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr, ddof=0)),
            n=len(scores),
        )
    return cohort


def _make_warnings_and_level() -> tuple[list[WarningSignal], AlertLevel]:
    warnings = [
        WarningSignal(
            name="위험 구간 진입",
            triggered=False,
            severity="critical",
            detail="정상 범위",
        ),
        WarningSignal(
            name="하위 백분위",
            triggered=False,
            severity="critical",
            detail="정상 범위",
        ),
        WarningSignal(
            name="저조한 개념 커버리지",
            triggered=False,
            severity="non-critical",
            detail="정상 범위",
        ),
    ]
    return warnings, AlertLevel.NORMAL


# ---------------------------------------------------------------------------
# T016: StudentLongitudinalPDFReportGenerator
# ---------------------------------------------------------------------------


class TestStudentLongitudinalPDFReportGenerator:
    """StudentLongitudinalPDFReportGenerator produces valid PDF files."""

    def test_generate_pdf_creates_file(self, tmp_path):
        from forma.student_longitudinal_report import (
            StudentLongitudinalPDFReportGenerator,
        )

        student = _make_student_data(weeks=[1, 2])
        cohort = _make_cohort(weeks=[1, 2])
        warnings, level = _make_warnings_and_level()
        output = str(tmp_path / "student_report.pdf")

        gen = StudentLongitudinalPDFReportGenerator(dpi=72)
        result = gen.generate_pdf(student, cohort, warnings, level, output)

        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_generate_pdf_returns_absolute_path(self, tmp_path):
        from forma.student_longitudinal_report import (
            StudentLongitudinalPDFReportGenerator,
        )

        student = _make_student_data(weeks=[1, 2])
        cohort = _make_cohort(weeks=[1, 2])
        warnings, level = _make_warnings_and_level()
        output = str(tmp_path / "student_report.pdf")

        gen = StudentLongitudinalPDFReportGenerator(dpi=72)
        result = gen.generate_pdf(student, cohort, warnings, level, output)
        assert os.path.isabs(result)

    def test_single_week_data_no_crash(self, tmp_path):
        from forma.student_longitudinal_report import (
            StudentLongitudinalPDFReportGenerator,
        )

        student = _make_student_data(weeks=[1])
        cohort = _make_cohort(weeks=[1])
        warnings, level = _make_warnings_and_level()
        output = str(tmp_path / "single_week.pdf")

        gen = StudentLongitudinalPDFReportGenerator(dpi=72)
        result = gen.generate_pdf(student, cohort, warnings, level, output)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_missing_q2_data_no_crash(self, tmp_path):
        from forma.student_longitudinal_report import (
            StudentLongitudinalPDFReportGenerator,
        )

        student = _make_student_data(weeks=[1, 2], include_q2=False)
        cohort = _make_cohort(weeks=[1, 2])
        warnings, level = _make_warnings_and_level()
        output = str(tmp_path / "no_q2.pdf")

        gen = StudentLongitudinalPDFReportGenerator(dpi=72)
        result = gen.generate_pdf(student, cohort, warnings, level, output)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_pdf_starts_with_pdf_header(self, tmp_path):
        from forma.student_longitudinal_report import (
            StudentLongitudinalPDFReportGenerator,
        )

        student = _make_student_data(weeks=[1, 2])
        cohort = _make_cohort(weeks=[1, 2])
        warnings, level = _make_warnings_and_level()
        output = str(tmp_path / "header_check.pdf")

        gen = StudentLongitudinalPDFReportGenerator(dpi=72)
        gen.generate_pdf(student, cohort, warnings, level, output)

        with open(output, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_with_llm_texts(self, tmp_path):
        from forma.student_longitudinal_report import (
            StudentLongitudinalPDFReportGenerator,
        )

        student = _make_student_data(weeks=[1, 2])
        cohort = _make_cohort(weeks=[1, 2])
        warnings, level = _make_warnings_and_level()
        output = str(tmp_path / "with_llm.pdf")

        llm_texts = {
            "coverage": "이 학생의 개념 커버리지는 상승 추세입니다.",
            "component": "항목별 점수가 균형을 이루고 있습니다.",
            "position": "전체 수강생 중 중상위권에 위치합니다.",
            "warning": "현재 경고 수준은 정상입니다.",
        }

        gen = StudentLongitudinalPDFReportGenerator(dpi=72)
        result = gen.generate_pdf(
            student, cohort, warnings, level, output, llm_texts=llm_texts,
        )
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_warning_level_shown_in_pdf(self, tmp_path):
        """Warning section with triggered signals produces valid PDF."""
        from forma.student_longitudinal_report import (
            StudentLongitudinalPDFReportGenerator,
        )

        student = _make_student_data(weeks=[1, 2])
        cohort = _make_cohort(weeks=[1, 2])
        warnings = [
            WarningSignal(
                name="위험 구간 진입",
                triggered=True,
                severity="critical",
                detail="ensemble_score 0.40 < 0.45",
            ),
        ]
        level = AlertLevel.WARNING
        output = str(tmp_path / "warning.pdf")

        gen = StudentLongitudinalPDFReportGenerator(dpi=72)
        result = gen.generate_pdf(student, cohort, warnings, level, output)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    # -----------------------------------------------------------------------
    # T027: PDF with LLM texts vs without
    # -----------------------------------------------------------------------

    def test_pdf_with_llm_texts_larger_than_without(self, tmp_path):
        """PDF with LLM interpretation texts should be larger than without."""
        from forma.student_longitudinal_report import (
            StudentLongitudinalPDFReportGenerator,
        )

        student = _make_student_data(weeks=[1, 2])
        cohort = _make_cohort(weeks=[1, 2])
        warnings, level = _make_warnings_and_level()

        gen = StudentLongitudinalPDFReportGenerator(dpi=72)

        # Without LLM
        out_no_llm = str(tmp_path / "no_llm.pdf")
        gen.generate_pdf(student, cohort, warnings, level, out_no_llm)
        size_no_llm = os.path.getsize(out_no_llm)

        # With LLM
        llm_texts = {
            "coverage": "이 학생의 개념 커버리지는 상승 추세를 보이고 있으며 "
                        "Q1과 Q2 모두에서 꾸준한 향상이 관찰됩니다.",
            "component": "항목별 점수가 전반적으로 균형을 이루고 있으며 "
                         "특히 개념 커버리지와 LLM 루브릭 간 격차가 줄어들고 있습니다.",
            "position": "전체 수강생 중 중상위권에 위치하며 백분위가 꾸준히 상승하고 있습니다.",
            "warning": "현재 경고 수준은 정상이며 모든 지표가 안정적입니다.",
        }
        out_with_llm = str(tmp_path / "with_llm.pdf")
        gen.generate_pdf(
            student, cohort, warnings, level, out_with_llm, llm_texts=llm_texts,
        )
        size_with_llm = os.path.getsize(out_with_llm)

        assert size_with_llm > size_no_llm

    def test_pdf_with_llm_texts_none_still_works(self, tmp_path):
        """PDF generation with llm_texts=None should not crash."""
        from forma.student_longitudinal_report import (
            StudentLongitudinalPDFReportGenerator,
        )

        student = _make_student_data(weeks=[1, 2])
        cohort = _make_cohort(weeks=[1, 2])
        warnings, level = _make_warnings_and_level()

        gen = StudentLongitudinalPDFReportGenerator(dpi=72)
        out = str(tmp_path / "no_llm_explicit.pdf")
        result = gen.generate_pdf(
            student, cohort, warnings, level, out, llm_texts=None,
        )
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0
