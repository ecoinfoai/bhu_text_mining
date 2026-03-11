"""Tests for professor_report_data.py — dataclasses and build function.

RED phase: these tests are written *before* the module exists and will fail
until ``src/forma/professor_report_data.py`` is implemented.

Covers:
  T002 — QuestionClassStats, StudentSummaryRow, ProfessorReportData dataclass
          instantiation and defaults.
  T003 — build_professor_report_data() class statistics, level distribution,
          student row construction, sorting, and z-score computation.
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# Sample data constants
# ---------------------------------------------------------------------------

CANONICAL_LEVELS = ("Advanced", "Proficient", "Developing", "Beginning")

# Scores chosen so that maths are easy to check:
#   Student A: Q1=0.90 (Advanced), Q2=0.70 (Proficient)  → mean=0.80
#   Student B: Q1=0.60 (Developing), Q2=0.50 (Developing) → mean=0.55
#   Student C: Q1=0.30 (Beginning), Q2=0.20 (Beginning)  → mean=0.25
#
# Class overall_ensemble means: [0.80, 0.55, 0.25]
# class_mean = (0.80+0.55+0.25)/3 = 0.5333…
# class_std  = population std of [0.80, 0.55, 0.25]
# numpy uses ddof=0 by default → matches np.nanmean / np.nanstd

_STUDENT_A_Q1_SCORE = 0.90
_STUDENT_A_Q2_SCORE = 0.70
_STUDENT_A_MEAN = (_STUDENT_A_Q1_SCORE + _STUDENT_A_Q2_SCORE) / 2  # 0.80

_STUDENT_B_Q1_SCORE = 0.60
_STUDENT_B_Q2_SCORE = 0.50
_STUDENT_B_MEAN = (_STUDENT_B_Q1_SCORE + _STUDENT_B_Q2_SCORE) / 2  # 0.55

_STUDENT_C_Q1_SCORE = 0.30
_STUDENT_C_Q2_SCORE = 0.20
_STUDENT_C_MEAN = (_STUDENT_C_Q1_SCORE + _STUDENT_C_Q2_SCORE) / 2  # 0.25

_CLASS_MEANS = [_STUDENT_A_MEAN, _STUDENT_B_MEAN, _STUDENT_C_MEAN]
_CLASS_MEAN = sum(_CLASS_MEANS) / len(_CLASS_MEANS)  # ≈ 0.5333


def _population_std(values: list[float]) -> float:
    """Population standard deviation (ddof=0), matching np.nanstd default."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)


_CLASS_STD = _population_std(_CLASS_MEANS)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_question_report(
    question_sn: int,
    ensemble_score: float,
    understanding_level: str,
    concept_coverage: float = 0.5,
    llm_median_score: float = 2.0,
    rasch_theta: float = 0.0,
    misconceptions: list[str] | None = None,
    question_text: str = "",
    topic: str = "",
) -> object:
    """Build a QuestionReportData without importing the module (import deferred)."""
    from forma.report_data_loader import QuestionReportData

    return QuestionReportData(
        question_sn=question_sn,
        question_text=question_text or f"Question {question_sn}",
        ensemble_score=ensemble_score,
        understanding_level=understanding_level,
        concept_coverage=concept_coverage,
        llm_median_score=llm_median_score,
        rasch_theta=rasch_theta,
        misconceptions=misconceptions or [],
    )


def _make_student(
    student_id: str,
    real_name: str,
    student_number: str,
    q1_score: float,
    q1_level: str,
    q2_score: float,
    q2_level: str,
    q1_coverage: float = 0.5,
    q2_coverage: float = 0.5,
    q1_misconceptions: list[str] | None = None,
    q2_misconceptions: list[str] | None = None,
) -> object:
    """Build a minimal StudentReportData with 2 questions."""
    from forma.report_data_loader import StudentReportData, QuestionReportData

    return StudentReportData(
        student_id=student_id,
        real_name=real_name,
        student_number=student_number,
        class_name="1A",
        course_name="Biology",
        chapter_name="Chapter 1",
        week_num=1,
        questions=[
            QuestionReportData(
                question_sn=1,
                question_text="Question 1 text",
                ensemble_score=q1_score,
                understanding_level=q1_level,
                concept_coverage=q1_coverage,
                llm_median_score=2.0,
                rasch_theta=0.0,
                misconceptions=q1_misconceptions or [],
            ),
            QuestionReportData(
                question_sn=2,
                question_text="Question 2 text",
                ensemble_score=q2_score,
                understanding_level=q2_level,
                concept_coverage=q2_coverage,
                llm_median_score=2.0,
                rasch_theta=0.0,
                misconceptions=q2_misconceptions or [],
            ),
        ],
    )


def _make_three_students() -> list:
    """Return 3 StudentReportData objects for standard tests."""
    return [
        _make_student(
            "SA", "Alice", "2026001",
            _STUDENT_A_Q1_SCORE, "Advanced",
            _STUDENT_A_Q2_SCORE, "Proficient",
        ),
        _make_student(
            "SB", "Bob", "2026002",
            _STUDENT_B_Q1_SCORE, "Developing",
            _STUDENT_B_Q2_SCORE, "Developing",
        ),
        _make_student(
            "SC", "Carol", "2026003",
            _STUDENT_C_Q1_SCORE, "Beginning",
            _STUDENT_C_Q2_SCORE, "Beginning",
        ),
    ]


def _make_distributions(students: list) -> object:
    """Build a ClassDistributions from the given students list."""
    from forma.report_data_loader import compute_class_distributions

    return compute_class_distributions(students)


# ===========================================================================
# T002: Dataclass instantiation tests
# ===========================================================================


class TestQuestionClassStatsInstantiation:
    """T002: QuestionClassStats dataclass instantiation."""

    def test_instantiation_all_fields(self):
        """QuestionClassStats stores all fields correctly."""
        from forma.professor_report_data import QuestionClassStats

        qcs = QuestionClassStats(
            question_sn=1,
            question_text="항상성의 정의를 서술하시오.",
            topic="항상성",
            ensemble_mean=0.65,
            ensemble_std=0.12,
            ensemble_median=0.68,
            concept_coverage_mean=0.55,
            llm_score_mean=0.70,
            rasch_theta_mean=-0.5,
            level_distribution={
                "Advanced": 5,
                "Proficient": 10,
                "Developing": 8,
                "Beginning": 7,
            },
            concept_mastery_rates={"항상성": 0.80, "음성되먹임": 0.45},
            misconception_frequencies=[("삼투와 확산 혼동", 5), ("기전 미이해", 3)],
        )
        assert qcs.question_sn == 1
        assert qcs.question_text == "항상성의 정의를 서술하시오."
        assert qcs.topic == "항상성"
        assert qcs.ensemble_mean == pytest.approx(0.65)
        assert qcs.ensemble_std == pytest.approx(0.12)
        assert qcs.ensemble_median == pytest.approx(0.68)
        assert qcs.concept_coverage_mean == pytest.approx(0.55)
        assert qcs.llm_score_mean == pytest.approx(0.70)
        assert qcs.rasch_theta_mean == pytest.approx(-0.5)
        assert qcs.level_distribution["Advanced"] == 5
        assert qcs.level_distribution["Beginning"] == 7
        assert qcs.concept_mastery_rates["항상성"] == pytest.approx(0.80)
        assert qcs.misconception_frequencies[0] == ("삼투와 확산 혼동", 5)

    def test_topic_may_be_empty(self):
        """QuestionClassStats accepts empty topic string."""
        from forma.professor_report_data import QuestionClassStats

        qcs = QuestionClassStats(
            question_sn=2,
            question_text="Q2",
            topic="",
            ensemble_mean=0.5,
            ensemble_std=0.1,
            ensemble_median=0.5,
            concept_coverage_mean=0.5,
            llm_score_mean=0.5,
            rasch_theta_mean=0.0,
            level_distribution={k: 0 for k in CANONICAL_LEVELS},
            concept_mastery_rates={},
            misconception_frequencies=[],
        )
        assert qcs.topic == ""

    def test_level_distribution_has_four_canonical_keys(self):
        """QuestionClassStats level_distribution can hold exactly 4 canonical keys."""
        from forma.professor_report_data import QuestionClassStats

        dist = {k: 0 for k in CANONICAL_LEVELS}
        qcs = QuestionClassStats(
            question_sn=1,
            question_text="Q",
            topic="",
            ensemble_mean=0.5,
            ensemble_std=0.0,
            ensemble_median=0.5,
            concept_coverage_mean=0.5,
            llm_score_mean=0.5,
            rasch_theta_mean=0.0,
            level_distribution=dist,
            concept_mastery_rates={},
            misconception_frequencies=[],
        )
        assert set(qcs.level_distribution.keys()) == set(CANONICAL_LEVELS)

    def test_misconception_frequencies_sorted_desc(self):
        """misconception_frequencies is stored in descending order by count."""
        from forma.professor_report_data import QuestionClassStats

        freqs = [("A", 10), ("B", 5), ("C", 2)]
        qcs = QuestionClassStats(
            question_sn=1,
            question_text="Q",
            topic="",
            ensemble_mean=0.5,
            ensemble_std=0.0,
            ensemble_median=0.5,
            concept_coverage_mean=0.5,
            llm_score_mean=0.5,
            rasch_theta_mean=0.0,
            level_distribution={k: 0 for k in CANONICAL_LEVELS},
            concept_mastery_rates={},
            misconception_frequencies=freqs,
        )
        counts = [c for _, c in qcs.misconception_frequencies]
        assert counts == sorted(counts, reverse=True)


class TestStudentSummaryRowInstantiation:
    """T002: StudentSummaryRow dataclass instantiation."""

    def test_instantiation_all_fields(self):
        """StudentSummaryRow stores all fields correctly."""
        from forma.professor_report_data import StudentSummaryRow

        row = StudentSummaryRow(
            student_id="SA",
            student_number="2026001",
            real_name="Alice",
            overall_ensemble_mean=0.75,
            overall_level="Proficient",
            per_question_scores={1: 0.80, 2: 0.70},
            per_question_levels={1: "Advanced", 2: "Proficient"},
            per_question_coverages={1: 0.85, 2: 0.65},
            is_at_risk=False,
            at_risk_reasons=[],
            z_score=0.80,
        )
        assert row.student_id == "SA"
        assert row.student_number == "2026001"
        assert row.real_name == "Alice"
        assert row.overall_ensemble_mean == pytest.approx(0.75)
        assert row.overall_level == "Proficient"
        assert row.per_question_scores[1] == pytest.approx(0.80)
        assert row.per_question_levels[2] == "Proficient"
        assert row.per_question_coverages[1] == pytest.approx(0.85)
        assert row.is_at_risk is False
        assert row.at_risk_reasons == []
        assert row.z_score == pytest.approx(0.80)

    def test_at_risk_with_reasons(self):
        """StudentSummaryRow stores is_at_risk=True with reason strings."""
        from forma.professor_report_data import StudentSummaryRow

        row = StudentSummaryRow(
            student_id="SC",
            student_number="2026003",
            real_name="Carol",
            overall_ensemble_mean=0.25,
            overall_level="Beginning",
            per_question_scores={1: 0.30, 2: 0.20},
            per_question_levels={1: "Beginning", 2: "Beginning"},
            per_question_coverages={1: 0.1, 2: 0.1},
            is_at_risk=True,
            at_risk_reasons=["종합점수 0.45 미만", "z-score < -1.0"],
            z_score=-1.5,
        )
        assert row.is_at_risk is True
        assert len(row.at_risk_reasons) >= 1
        assert row.z_score == pytest.approx(-1.5)

    def test_overall_level_must_be_canonical(self):
        """StudentSummaryRow overall_level accepts any of the 4 canonical levels."""
        from forma.professor_report_data import StudentSummaryRow

        for level in CANONICAL_LEVELS:
            row = StudentSummaryRow(
                student_id="S",
                student_number="N",
                real_name="R",
                overall_ensemble_mean=0.5,
                overall_level=level,
                per_question_scores={},
                per_question_levels={},
                per_question_coverages={},
                is_at_risk=False,
                at_risk_reasons=[],
                z_score=0.0,
            )
            assert row.overall_level == level


class TestProfessorReportDataInstantiation:
    """T002: ProfessorReportData dataclass instantiation and defaults."""

    def _make_minimal_report(self, **overrides):
        """Create a minimal valid ProfessorReportData."""
        from forma.professor_report_data import ProfessorReportData

        defaults = dict(
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Week 1 Formative Test",
            generation_date="2026-03-08",
            n_students=3,
            n_questions=2,
            class_ensemble_mean=0.53,
            class_ensemble_std=0.23,
            class_ensemble_median=0.55,
            class_ensemble_q1=0.25,
            class_ensemble_q3=0.80,
            overall_level_distribution={k: 0 for k in CANONICAL_LEVELS},
            question_stats=[],
            student_rows=[],
            n_at_risk=0,
            pct_at_risk=0.0,
        )
        defaults.update(overrides)
        return ProfessorReportData(**defaults)

    def test_instantiation_required_fields(self):
        """ProfessorReportData stores all required fields correctly."""
        report = self._make_minimal_report()
        assert report.class_name == "1A"
        assert report.week_num == 1
        assert report.subject == "Biology"
        assert report.exam_title == "Week 1 Formative Test"
        assert report.generation_date == "2026-03-08"
        assert report.n_students == 3
        assert report.n_questions == 2
        assert report.class_ensemble_mean == pytest.approx(0.53)
        assert report.class_ensemble_std == pytest.approx(0.23)
        assert report.class_ensemble_median == pytest.approx(0.55)
        assert report.class_ensemble_q1 == pytest.approx(0.25)
        assert report.class_ensemble_q3 == pytest.approx(0.80)
        assert report.n_at_risk == 0
        assert report.pct_at_risk == pytest.approx(0.0)

    def test_default_overall_assessment_empty(self):
        """ProfessorReportData.overall_assessment defaults to empty string."""
        report = self._make_minimal_report()
        assert report.overall_assessment == ""

    def test_default_teaching_suggestions_empty(self):
        """ProfessorReportData.teaching_suggestions defaults to empty string."""
        report = self._make_minimal_report()
        assert report.teaching_suggestions == ""

    def test_default_llm_model_used_empty(self):
        """ProfessorReportData.llm_model_used defaults to empty string."""
        report = self._make_minimal_report()
        assert report.llm_model_used == ""

    def test_default_llm_generation_failed_false(self):
        """ProfessorReportData.llm_generation_failed defaults to False."""
        report = self._make_minimal_report()
        assert report.llm_generation_failed is False

    def test_default_llm_error_message_empty(self):
        """ProfessorReportData.llm_error_message defaults to empty string."""
        report = self._make_minimal_report()
        assert report.llm_error_message == ""

    def test_overall_level_distribution_has_four_canonical_keys(self):
        """overall_level_distribution must contain exactly 4 canonical level keys."""
        report = self._make_minimal_report(
            overall_level_distribution={
                "Advanced": 1,
                "Proficient": 1,
                "Developing": 1,
                "Beginning": 0,
            }
        )
        assert set(report.overall_level_distribution.keys()) == set(CANONICAL_LEVELS)

    def test_llm_fields_can_be_set(self):
        """ProfessorReportData accepts non-default LLM field values."""
        report = self._make_minimal_report(
            overall_assessment="Overall the class performed adequately.",
            teaching_suggestions="Focus on concept coverage next week.",
            llm_model_used="gemini-2.5-flash",
            llm_generation_failed=True,
            llm_error_message="API timeout",
        )
        assert report.overall_assessment == "Overall the class performed adequately."
        assert report.teaching_suggestions == "Focus on concept coverage next week."
        assert report.llm_model_used == "gemini-2.5-flash"
        assert report.llm_generation_failed is True
        assert report.llm_error_message == "API timeout"


# ===========================================================================
# T006: New fields — hub_gap_entries, section, is_multi_class, section_names
# ===========================================================================


class TestQuestionClassStatsHubGapEntries:
    """T006: QuestionClassStats.hub_gap_entries default field."""

    def test_hub_gap_entries_default_empty_list(self):
        """QuestionClassStats.hub_gap_entries defaults to empty list."""
        from forma.professor_report_data import QuestionClassStats

        qcs = QuestionClassStats(question_sn=1)
        assert qcs.hub_gap_entries == []

    def test_hub_gap_entries_independent_across_instances(self):
        """QuestionClassStats.hub_gap_entries list is independent per instance."""
        from forma.professor_report_data import QuestionClassStats

        qcs1 = QuestionClassStats(question_sn=1)
        qcs2 = QuestionClassStats(question_sn=2)
        qcs1.hub_gap_entries.append("entry")
        assert qcs2.hub_gap_entries == []


class TestStudentSummaryRowSection:
    """T006: StudentSummaryRow.section default field."""

    def test_section_default_empty_string(self):
        """StudentSummaryRow.section defaults to empty string."""
        from forma.professor_report_data import StudentSummaryRow

        row = StudentSummaryRow(student_id="SA")
        assert row.section == ""

    def test_section_can_be_set(self):
        """StudentSummaryRow.section can be set to a section name."""
        from forma.professor_report_data import StudentSummaryRow

        row = StudentSummaryRow(student_id="SA", section="1A")
        assert row.section == "1A"


class TestProfessorReportDataMultiClassFields:
    """T006: ProfessorReportData.is_multi_class and section_names defaults."""

    def _make_minimal_report(self, **overrides):
        from forma.professor_report_data import ProfessorReportData

        defaults = dict(
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Week 1 Formative Test",
            generation_date="2026-03-08",
            n_students=3,
            n_questions=2,
            class_ensemble_mean=0.53,
            class_ensemble_std=0.23,
            class_ensemble_median=0.55,
            class_ensemble_q1=0.25,
            class_ensemble_q3=0.80,
            overall_level_distribution={k: 0 for k in CANONICAL_LEVELS},
            question_stats=[],
            student_rows=[],
            n_at_risk=0,
            pct_at_risk=0.0,
        )
        defaults.update(overrides)
        return ProfessorReportData(**defaults)

    def test_is_multi_class_default_false(self):
        """ProfessorReportData.is_multi_class defaults to False."""
        report = self._make_minimal_report()
        assert report.is_multi_class is False

    def test_section_names_default_empty_list(self):
        """ProfessorReportData.section_names defaults to empty list."""
        report = self._make_minimal_report()
        assert report.section_names == []

    def test_is_multi_class_can_be_set(self):
        """ProfessorReportData.is_multi_class can be set to True."""
        report = self._make_minimal_report(is_multi_class=True)
        assert report.is_multi_class is True

    def test_section_names_can_be_set(self):
        """ProfessorReportData.section_names can hold section name strings."""
        report = self._make_minimal_report(section_names=["1A", "1B", "1C"])
        assert report.section_names == ["1A", "1B", "1C"]

    def test_section_names_independent_across_instances(self):
        """ProfessorReportData.section_names list is independent per instance."""
        report1 = self._make_minimal_report()
        report2 = self._make_minimal_report()
        report1.section_names.append("1A")
        assert report2.section_names == []


# ===========================================================================
# T003: build_professor_report_data() tests
# ===========================================================================


class TestBuildProfessorReportDataClassStats:
    """T003: Class-level statistics computation."""

    def test_class_ensemble_mean(self):
        """class_ensemble_mean is the mean of per-student overall means."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        assert report.class_ensemble_mean == pytest.approx(_CLASS_MEAN, abs=1e-6)

    def test_class_ensemble_std(self):
        """class_ensemble_std is the population standard deviation of student means."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        assert report.class_ensemble_std == pytest.approx(_CLASS_STD, abs=1e-6)

    def test_class_ensemble_median(self):
        """class_ensemble_median is the median of per-student overall means."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        # sorted means = [0.25, 0.55, 0.80] → median = 0.55
        assert report.class_ensemble_median == pytest.approx(_STUDENT_B_MEAN, abs=1e-6)

    def test_class_ensemble_q1_and_q3(self):
        """class_ensemble_q1 and q3 are the 25th and 75th percentiles."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        # Both quartiles must be in [0, 1]
        assert 0.0 <= report.class_ensemble_q1 <= 1.0
        assert 0.0 <= report.class_ensemble_q3 <= 1.0
        # Q1 <= median <= Q3
        assert report.class_ensemble_q1 <= report.class_ensemble_median
        assert report.class_ensemble_median <= report.class_ensemble_q3

    def test_n_students_matches_input(self):
        """n_students equals len(students)."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        assert report.n_students == 3

    def test_n_questions_matches_unique_question_sns(self):
        """n_questions equals the number of unique question_sn values."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        assert report.n_questions == 2


class TestBuildProfessorReportDataLevelDistribution:
    """T003: Overall level distribution computation."""

    def test_overall_level_distribution_sums_to_n_students(self):
        """Sum of overall_level_distribution values equals n_students."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        total = sum(report.overall_level_distribution.values())
        assert total == report.n_students

    def test_overall_level_distribution_has_four_keys(self):
        """overall_level_distribution always has exactly 4 canonical level keys."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        assert set(report.overall_level_distribution.keys()) == set(CANONICAL_LEVELS)

    def test_overall_level_distribution_counts_correct(self):
        """Level counts reflect each student's overall_level assignment."""
        from forma.professor_report_data import build_professor_report_data

        # Student A: both questions Developing → overall=Developing
        # Student B: both questions Developing → overall=Developing
        # Student C: both questions Beginning  → overall=Beginning
        students = [
            _make_student("SA", "Alice", "001", 0.60, "Developing", 0.50, "Developing"),
            _make_student("SB", "Bob",   "002", 0.60, "Developing", 0.50, "Developing"),
            _make_student("SC", "Carol", "003", 0.30, "Beginning",  0.20, "Beginning"),
        ]
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        assert report.overall_level_distribution["Developing"] == 2
        assert report.overall_level_distribution["Beginning"] == 1
        assert report.overall_level_distribution["Proficient"] == 0
        assert report.overall_level_distribution["Advanced"] == 0


class TestBuildProfessorReportDataStudentRows:
    """T003: Student row construction."""

    def test_all_students_present_in_rows(self):
        """All input students appear in student_rows — none dropped."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        row_ids = {r.student_id for r in report.student_rows}
        assert row_ids == {"SA", "SB", "SC"}

    def test_student_rows_sorted_descending_by_overall_mean(self):
        """student_rows are sorted by overall_ensemble_mean descending."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        means = [r.overall_ensemble_mean for r in report.student_rows]
        assert means == sorted(means, reverse=True)

    def test_per_question_scores_populated(self):
        """per_question_scores has entries for each question_sn."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        for row in report.student_rows:
            assert 1 in row.per_question_scores
            assert 2 in row.per_question_scores

    def test_per_question_levels_populated(self):
        """per_question_levels has entries for each question_sn."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        for row in report.student_rows:
            assert 1 in row.per_question_levels
            assert 2 in row.per_question_levels
            for level in row.per_question_levels.values():
                assert level in CANONICAL_LEVELS

    def test_per_question_coverages_populated(self):
        """per_question_coverages has entries for each question_sn."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        for row in report.student_rows:
            assert 1 in row.per_question_coverages
            assert 2 in row.per_question_coverages

    def test_overall_level_is_mode_of_per_question_levels(self):
        """overall_level is the mode of per_question_levels; tie-breaks to lower."""
        from forma.professor_report_data import build_professor_report_data

        # Student C: Q1=Beginning, Q2=Beginning → mode=Beginning
        # Student A: Q1=Advanced,  Q2=Proficient → tie → lower=Proficient
        # Student B: Q1=Developing, Q2=Developing → mode=Developing
        students = [
            _make_student("SA", "Alice", "001", 0.90, "Advanced",   0.70, "Proficient"),
            _make_student("SB", "Bob",   "002", 0.60, "Developing", 0.50, "Developing"),
            _make_student("SC", "Carol", "003", 0.30, "Beginning",  0.20, "Beginning"),
        ]
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        rows_by_id = {r.student_id: r for r in report.student_rows}

        # SB: both Developing → mode=Developing
        assert rows_by_id["SB"].overall_level == "Developing"
        # SC: both Beginning → mode=Beginning
        assert rows_by_id["SC"].overall_level == "Beginning"
        # SA: Advanced vs Proficient tie → tie-break to lower → Proficient
        assert rows_by_id["SA"].overall_level == "Proficient"


class TestBuildProfessorReportDataZScore:
    """T003: z-score computation."""

    def test_z_score_computed_correctly(self):
        """z_score = (overall_ensemble_mean - class_mean) / class_std."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        rows_by_id = {r.student_id: r for r in report.student_rows}

        expected_z_a = (_STUDENT_A_MEAN - _CLASS_MEAN) / _CLASS_STD
        expected_z_b = (_STUDENT_B_MEAN - _CLASS_MEAN) / _CLASS_STD
        expected_z_c = (_STUDENT_C_MEAN - _CLASS_MEAN) / _CLASS_STD

        assert rows_by_id["SA"].z_score == pytest.approx(expected_z_a, abs=1e-6)
        assert rows_by_id["SB"].z_score == pytest.approx(expected_z_b, abs=1e-6)
        assert rows_by_id["SC"].z_score == pytest.approx(expected_z_c, abs=1e-6)

    def test_zero_variance_z_scores_are_zero(self):
        """When all students have the same score, z_scores are all 0.0, no crash."""
        from forma.professor_report_data import build_professor_report_data

        # All students score 0.65 on both questions → std=0
        students = [
            _make_student("SA", "Alice", "001", 0.65, "Proficient", 0.65, "Proficient"),
            _make_student("SB", "Bob",   "002", 0.65, "Proficient", 0.65, "Proficient"),
            _make_student("SC", "Carol", "003", 0.65, "Proficient", 0.65, "Proficient"),
        ]
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        for row in report.student_rows:
            assert row.z_score == pytest.approx(0.0)


class TestBuildProfessorReportDataAtRisk:
    """T003: At-risk identification."""

    def test_at_risk_low_score(self):
        """Student with overall_ensemble_mean < 0.45 is flagged as at-risk."""
        from forma.professor_report_data import build_professor_report_data

        students = [
            _make_student("SA", "Alice", "001", 0.80, "Advanced",  0.80, "Advanced"),
            _make_student("SB", "Bob",   "002", 0.80, "Advanced",  0.80, "Advanced"),
            # SC mean = 0.25 < 0.45 → at-risk
            _make_student("SC", "Carol", "003", 0.30, "Beginning", 0.20, "Beginning"),
        ]
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        rows_by_id = {r.student_id: r for r in report.student_rows}
        assert rows_by_id["SC"].is_at_risk is True
        assert len(rows_by_id["SC"].at_risk_reasons) >= 1

    def test_at_risk_reasons_non_empty_when_flagged(self):
        """at_risk_reasons is non-empty when is_at_risk is True."""
        from forma.professor_report_data import build_professor_report_data

        students = [
            _make_student("SA", "Alice", "001", 0.80, "Advanced",  0.80, "Advanced"),
            _make_student("SB", "Bob",   "002", 0.80, "Advanced",  0.80, "Advanced"),
            _make_student("SC", "Carol", "003", 0.20, "Beginning", 0.10, "Beginning",
                          q1_coverage=0.0, q2_coverage=0.0),
        ]
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        _rows_by_id = {r.student_id: r for r in report.student_rows}
        at_risk_rows = [r for r in report.student_rows if r.is_at_risk]
        for row in at_risk_rows:
            assert len(row.at_risk_reasons) > 0, (
                f"at_risk_reasons empty for at-risk student {row.student_id}"
            )

    def test_not_at_risk_safe_student(self):
        """Student with high score is not flagged as at-risk."""
        from forma.professor_report_data import build_professor_report_data

        students = [
            _make_student("SA", "Alice", "001", 0.90, "Advanced",  0.85, "Advanced"),
            _make_student("SB", "Bob",   "002", 0.50, "Developing", 0.45, "Developing"),
            _make_student("SC", "Carol", "003", 0.48, "Developing", 0.46, "Developing"),
        ]
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        rows_by_id = {r.student_id: r for r in report.student_rows}
        assert rows_by_id["SA"].is_at_risk is False
        assert rows_by_id["SA"].at_risk_reasons == []

    def test_n_at_risk_and_pct_at_risk_computed(self):
        """n_at_risk and pct_at_risk are consistent with flagged student count."""
        from forma.professor_report_data import build_professor_report_data

        students = [
            _make_student("SA", "Alice", "001", 0.90, "Advanced",  0.85, "Advanced"),
            _make_student("SB", "Bob",   "002", 0.90, "Advanced",  0.85, "Advanced"),
            # SC will be at-risk: score < 0.45
            _make_student("SC", "Carol", "003", 0.20, "Beginning", 0.10, "Beginning"),
        ]
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        flagged_count = sum(1 for r in report.student_rows if r.is_at_risk)
        assert report.n_at_risk == flagged_count
        expected_pct = (flagged_count / report.n_students) * 100.0
        assert report.pct_at_risk == pytest.approx(expected_pct, abs=1e-6)

    def test_at_risk_all_beginning_levels(self):
        """Student with all per_question_levels=Beginning is flagged at-risk."""
        from forma.professor_report_data import build_professor_report_data

        students = [
            # High-scoring classmates to avoid z-score being the only trigger
            _make_student("SA", "Alice", "001", 0.90, "Advanced",  0.85, "Advanced"),
            _make_student("SB", "Bob",   "002", 0.90, "Advanced",  0.85, "Advanced"),
            # SC just above score threshold but all Beginning levels
            _make_student("SC", "Carol", "003", 0.46, "Beginning", 0.46, "Beginning"),
        ]
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        rows_by_id = {r.student_id: r for r in report.student_rows}
        # SC has all Beginning per_question_levels → at-risk criterion 3
        assert rows_by_id["SC"].is_at_risk is True

    def test_at_risk_zero_coverage_on_any_question(self):
        """Student with any per_question_coverages == 0.0 is flagged at-risk."""
        from forma.professor_report_data import build_professor_report_data

        students = [
            _make_student("SA", "Alice", "001", 0.90, "Advanced",  0.85, "Advanced"),
            _make_student("SB", "Bob",   "002", 0.90, "Advanced",  0.85, "Advanced"),
            # SC has zero coverage on Q1
            _make_student("SC", "Carol", "003", 0.65, "Proficient", 0.65, "Proficient",
                          q1_coverage=0.0, q2_coverage=0.80),
        ]
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        rows_by_id = {r.student_id: r for r in report.student_rows}
        assert rows_by_id["SC"].is_at_risk is True

    def test_at_risk_misconceptions_threshold(self):
        """Student with >= 3 total misconceptions and not Advanced is flagged."""
        from forma.professor_report_data import build_professor_report_data

        students = [
            _make_student("SA", "Alice", "001", 0.90, "Advanced",  0.85, "Advanced"),
            _make_student("SB", "Bob",   "002", 0.90, "Advanced",  0.85, "Advanced"),
            # SC has 3 misconceptions across questions and level=Developing
            _make_student(
                "SC", "Carol", "003",
                0.65, "Proficient", 0.65, "Proficient",
                q1_misconceptions=["M1", "M2"],
                q2_misconceptions=["M3"],
            ),
        ]
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        rows_by_id = {r.student_id: r for r in report.student_rows}
        # SC: 3 misconceptions, not Advanced (Proficient) → at-risk criterion 5
        assert rows_by_id["SC"].is_at_risk is True

    def test_at_risk_low_average_coverage(self):
        """Student with mean per_question_coverages < 0.30 is flagged at-risk."""
        from forma.professor_report_data import build_professor_report_data

        students = [
            _make_student("SA", "Alice", "001", 0.90, "Advanced",  0.85, "Advanced"),
            _make_student("SB", "Bob",   "002", 0.90, "Advanced",  0.85, "Advanced"),
            # SC mean coverage = (0.10 + 0.20) / 2 = 0.15 < 0.30
            _make_student("SC", "Carol", "003", 0.65, "Proficient", 0.65, "Proficient",
                          q1_coverage=0.10, q2_coverage=0.20),
        ]
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        rows_by_id = {r.student_id: r for r in report.student_rows}
        assert rows_by_id["SC"].is_at_risk is True


class TestBuildProfessorReportDataMetadata:
    """T003: Metadata and generation_date propagation."""

    def test_class_name_propagated(self):
        """class_name from arguments appears in report."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="2B",
            week_num=3,
            subject="Chemistry",
            exam_title="Mid-term",
        )
        assert report.class_name == "2B"
        assert report.week_num == 3
        assert report.subject == "Chemistry"
        assert report.exam_title == "Mid-term"

    def test_generation_date_is_set(self):
        """generation_date is a non-empty ISO 8601 date string."""
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = _make_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )
        assert isinstance(report.generation_date, str)
        assert len(report.generation_date) >= 10  # At minimum "YYYY-MM-DD"


# ===========================================================================
# T019: Conditional formatting helpers
# ===========================================================================


class TestConditionalFormatting:
    """T019: compute_conditional_indicator() business-rule tests."""

    def test_score_above_upper_threshold_returns_plus(self):
        """score > mean + 0.5*std → '+'."""
        from forma.professor_report_data import compute_conditional_indicator

        # mean=0.5, std=0.2 → upper=0.6; score=0.7 exceeds upper
        result = compute_conditional_indicator(score=0.7, mean=0.5, std=0.2)
        assert result == "+"

    def test_score_below_lower_threshold_returns_minus(self):
        """score < mean - 0.5*std → '-'."""
        from forma.professor_report_data import compute_conditional_indicator

        # mean=0.5, std=0.2 → lower=0.4; score=0.3 is below lower
        result = compute_conditional_indicator(score=0.3, mean=0.5, std=0.2)
        assert result == "-"

    def test_score_between_thresholds_returns_empty(self):
        """score strictly between lower and upper thresholds → ''."""
        from forma.professor_report_data import compute_conditional_indicator

        # mean=0.5, std=0.2 → lower=0.4, upper=0.6; score=0.5 is in between
        result = compute_conditional_indicator(score=0.5, mean=0.5, std=0.2)
        assert result == ""

    def test_std_zero_always_returns_empty_regardless_of_score(self):
        """std == 0 → always '' regardless of score value."""
        from forma.professor_report_data import compute_conditional_indicator

        # Even a score far from the mean should return "" when std==0
        assert compute_conditional_indicator(score=1.0, mean=0.5, std=0.0) == ""
        assert compute_conditional_indicator(score=0.0, mean=0.5, std=0.0) == ""
        assert compute_conditional_indicator(score=0.5, mean=0.5, std=0.0) == ""

    def test_score_exactly_at_upper_boundary_inclusive_returns_plus(self):
        """score == mean + 0.5*std (boundary) → '+' (>= is inclusive)."""
        from forma.professor_report_data import compute_conditional_indicator

        # mean=0.5, std=0.2 → upper=0.6 exactly
        result = compute_conditional_indicator(score=0.6, mean=0.5, std=0.2)
        assert result == "+"

    def test_score_exactly_at_lower_boundary_inclusive_returns_minus(self):
        """score == mean - 0.5*std (boundary) → '-' (<= is inclusive)."""
        from forma.professor_report_data import compute_conditional_indicator

        # mean=0.5, std=0.2 → lower=0.4 exactly
        result = compute_conditional_indicator(score=0.4, mean=0.5, std=0.2)
        assert result == "-"


class TestGetConditionalBgColor:
    """T019: get_conditional_bg_color() business-rule tests."""

    def test_above_upper_threshold_returns_green(self):
        """score >= mean + 0.5*std → HexColor('#E8F5E9') (green)."""
        from forma.professor_report_data import get_conditional_bg_color
        from reportlab.lib.colors import HexColor

        color = get_conditional_bg_color(score=0.7, mean=0.5, std=0.2)
        assert color == HexColor("#E8F5E9")

    def test_below_lower_threshold_returns_red(self):
        """score <= mean - 0.5*std → HexColor('#FFEBEE') (red)."""
        from forma.professor_report_data import get_conditional_bg_color
        from reportlab.lib.colors import HexColor

        color = get_conditional_bg_color(score=0.3, mean=0.5, std=0.2)
        assert color == HexColor("#FFEBEE")

    def test_between_thresholds_returns_white(self):
        """score between thresholds → white (HexColor('#FFFFFF') or equivalent)."""
        from forma.professor_report_data import get_conditional_bg_color
        from reportlab.lib.colors import HexColor, white

        color = get_conditional_bg_color(score=0.5, mean=0.5, std=0.2)
        # Accept either HexColor("#FFFFFF") or reportlab's white constant
        assert color == HexColor("#FFFFFF") or color == white

    def test_std_zero_returns_white(self):
        """std == 0 → white (neutral), regardless of score."""
        from forma.professor_report_data import get_conditional_bg_color
        from reportlab.lib.colors import HexColor, white

        color = get_conditional_bg_color(score=0.9, mean=0.5, std=0.0)
        assert color == HexColor("#FFFFFF") or color == white


# ===========================================================================
# Augmented tests: additional boundary and robustness scenarios
# ===========================================================================


class TestComputeConditionalIndicatorAugmented:
    """Augmented: NaN and additional robustness for compute_conditional_indicator."""

    def test_nan_score_does_not_crash(self):
        """compute_conditional_indicator with NaN score must not raise an exception."""
        from forma.professor_report_data import compute_conditional_indicator

        # NaN >= any threshold is False, NaN <= any threshold is False → ''
        result = compute_conditional_indicator(score=float("nan"), mean=0.5, std=0.2)
        # Should return some string without crashing; NaN comparisons yield False → ''
        assert isinstance(result, str)

    def test_nan_mean_does_not_crash(self):
        """compute_conditional_indicator with NaN mean must not raise an exception."""
        from forma.professor_report_data import compute_conditional_indicator

        result = compute_conditional_indicator(score=0.7, mean=float("nan"), std=0.2)
        assert isinstance(result, str)

    def test_score_just_below_upper_boundary_returns_empty(self):
        """score just below mean + 0.5*std → '' (not '+'): boundary is >=, not >."""
        from forma.professor_report_data import compute_conditional_indicator

        import sys
        # mean=0.5, std=0.2 → upper=0.6; score = 0.6 - epsilon → ''
        eps = sys.float_info.epsilon * 1000  # small positive
        result = compute_conditional_indicator(score=0.6 - eps, mean=0.5, std=0.2)
        assert result == ""

    def test_score_just_above_lower_boundary_returns_empty(self):
        """score just above mean - 0.5*std → '' (not '-'): boundary is <=, not <."""
        from forma.professor_report_data import compute_conditional_indicator

        import sys
        # mean=0.5, std=0.2 → lower=0.4; score = 0.4 + epsilon → ''
        eps = sys.float_info.epsilon * 1000
        result = compute_conditional_indicator(score=0.4 + eps, mean=0.5, std=0.2)
        assert result == ""


# ===========================================================================
# T023: identify_at_risk() — new signature with misconception_counts dict
# ===========================================================================


def _make_at_risk_row(**overrides):
    """Build a StudentSummaryRow with safe (not-at-risk) defaults.

    The dataclass uses dict[int, str] for per_question_levels and
    dict[int, float] for per_question_coverages.  Defaults are chosen so that
    none of the 6 at-risk criteria fire unless explicitly overridden.
    """
    from forma.professor_report_data import StudentSummaryRow

    defaults = {
        "student_id": "S001",
        "student_number": "2026001",
        "real_name": "테스트",
        "overall_ensemble_mean": 0.60,   # safe: not < 0.45
        "overall_level": "Proficient",   # safe: not all-Beginning, not Advanced (for misc criterion)
        "per_question_scores": {1: 0.60, 2: 0.60},
        "per_question_levels": {1: "Proficient", 2: "Proficient"},   # safe: not all Beginning
        "per_question_coverages": {1: 0.80, 2: 0.80},               # safe: no zeros, avg=0.80
        "is_at_risk": False,
        "at_risk_reasons": [],
        "z_score": 0.0,                  # safe: not < -1.0
    }
    defaults.update(overrides)
    return StudentSummaryRow(**defaults)


class TestIdentifyAtRisk:
    """T023: identify_at_risk() with new misconception_counts: dict[int, int] signature.

    All tests in this class SHOULD FAIL (RED phase) because the new
    ``identify_at_risk(row, class_mean, class_std, misconception_counts)``
    signature does not yet exist in professor_report_data.py.
    """

    # ------------------------------------------------------------------
    # Criterion 1: overall_ensemble_mean < 0.45
    # ------------------------------------------------------------------

    def test_score_below_threshold_is_at_risk(self):
        """score=0.40 < 0.45 → (True, reasons containing '전체 점수')."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(overall_ensemble_mean=0.40, z_score=0.0)
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={},
        )
        assert is_risk is True
        assert any("전체 점수" in r for r in reasons), (
            f"Expected '전체 점수' in reasons, got: {reasons}"
        )

    def test_score_above_threshold_not_at_risk(self):
        """score=0.50 with all other criteria safe → (False, [])."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(overall_ensemble_mean=0.50, z_score=0.0)
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={},
        )
        assert is_risk is False
        assert reasons == []

    # ------------------------------------------------------------------
    # Criterion 2: z-score < -1.0
    # ------------------------------------------------------------------

    def test_z_score_below_minus_one_is_at_risk(self):
        """score=0.50, z=-1.5 → (True, reasons containing 'Z-점수')."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(overall_ensemble_mean=0.50, z_score=-1.5)
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={},
        )
        assert is_risk is True
        assert any("Z-점수" in r for r in reasons), (
            f"Expected 'Z-점수' in reasons, got: {reasons}"
        )

    def test_z_score_above_minus_one_not_at_risk(self):
        """score=0.50, z=-0.5 with all other criteria safe → (False, [])."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(overall_ensemble_mean=0.50, z_score=-0.5)
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={},
        )
        assert is_risk is False
        assert reasons == []

    # ------------------------------------------------------------------
    # Criterion 3: ALL per_question_levels are "Beginning"
    # ------------------------------------------------------------------

    def test_all_beginning_is_at_risk(self):
        """All per_question_levels='Beginning' → (True, has '모든 문항' reason)."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.50,
            z_score=0.0,
            per_question_levels={1: "Beginning", 2: "Beginning"},
        )
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.55,
            class_std=0.05,
            misconception_counts={},
        )
        assert is_risk is True
        assert any("모든 문항" in r for r in reasons), (
            f"Expected '모든 문항' in reasons, got: {reasons}"
        )

    def test_not_all_beginning_not_at_risk_for_criterion_3(self):
        """Mixed levels (Beginning + Proficient) → criterion 3 does NOT fire."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.50,
            z_score=0.0,
            per_question_levels={1: "Beginning", 2: "Proficient"},
        )
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.55,
            class_std=0.05,
            misconception_counts={},
        )
        # Criterion 3 must not appear; other criteria are also safe
        assert not any("모든 문항" in r for r in reasons), (
            f"Criterion 3 should not fire for mixed levels, got: {reasons}"
        )

    # ------------------------------------------------------------------
    # Criterion 4: ANY per_question_coverage == 0.0
    # ------------------------------------------------------------------

    def test_any_zero_coverage_is_at_risk(self):
        """One coverage=0.0 → (True, reason contains '커버리지 0%')."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.60,
            z_score=0.0,
            per_question_coverages={1: 0.0, 2: 0.80},
        )
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={},
        )
        assert is_risk is True
        assert any("커버리지 0%" in r for r in reasons), (
            f"Expected '커버리지 0%' in reasons, got: {reasons}"
        )

    def test_no_zero_coverage_not_at_risk_for_criterion_4(self):
        """All coverages > 0 → criterion 4 does NOT fire."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.60,
            z_score=0.0,
            per_question_coverages={1: 0.50, 2: 0.80},
        )
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={},
        )
        assert not any("커버리지 0%" in r for r in reasons), (
            f"Criterion 4 should not fire when all coverages > 0, got: {reasons}"
        )

    # ------------------------------------------------------------------
    # Criterion 5: total misconceptions >= 3 AND overall_level != "Advanced"
    # ------------------------------------------------------------------

    def test_misconceptions_3_non_advanced_is_at_risk(self):
        """total misconceptions >= 3, level != 'Advanced' → (True, reason present)."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.60,
            z_score=0.0,
            overall_level="Proficient",
        )
        # misconception_counts: {1: 2, 2: 1} → total=3, non-Advanced
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={1: 2, 2: 1},
        )
        assert is_risk is True
        assert any("오개념" in r for r in reasons), (
            f"Expected '오개념' in reasons, got: {reasons}"
        )

    def test_misconceptions_3_advanced_not_at_risk_for_criterion_5(self):
        """total >= 3 but level='Advanced' → criterion 5 does NOT fire."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.60,
            z_score=0.0,
            overall_level="Advanced",
        )
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={1: 2, 2: 1},
        )
        assert not any("오개념" in r for r in reasons), (
            f"Criterion 5 must not fire for Advanced level, got: {reasons}"
        )

    def test_misconceptions_below_3_not_at_risk_for_criterion_5(self):
        """total=2, non-Advanced → criterion 5 does NOT fire (below threshold)."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.60,
            z_score=0.0,
            overall_level="Proficient",
        )
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={1: 1, 2: 1},   # total=2 < 3
        )
        assert not any("오개념" in r for r in reasons), (
            f"Criterion 5 must not fire when total < 3, got: {reasons}"
        )

    # ------------------------------------------------------------------
    # Criterion 6: average of per_question_coverage < 0.30
    # ------------------------------------------------------------------

    def test_avg_coverage_below_30pct_is_at_risk(self):
        """avg coverage=0.25 < 0.30 → (True, reason contains '평균 커버리지')."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.60,
            z_score=0.0,
            per_question_coverages={1: 0.20, 2: 0.30},  # avg=0.25
        )
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={},
        )
        assert is_risk is True
        assert any("평균 커버리지" in r for r in reasons), (
            f"Expected '평균 커버리지' in reasons, got: {reasons}"
        )

    def test_avg_coverage_above_30pct_not_at_risk_for_criterion_6(self):
        """avg coverage=0.50 >= 0.30 → criterion 6 does NOT fire."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.60,
            z_score=0.0,
            per_question_coverages={1: 0.50, 2: 0.50},  # avg=0.50
        )
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={},
        )
        assert not any("평균 커버리지" in r for r in reasons), (
            f"Criterion 6 must not fire when avg coverage >= 0.30, got: {reasons}"
        )

    # ------------------------------------------------------------------
    # OR logic and edge cases
    # ------------------------------------------------------------------

    def test_or_logic_multiple_criteria(self):
        """score<0.45 AND z<-1.0 → (True, reasons list has BOTH reason strings)."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.40,  # criterion 1
            z_score=-1.5,               # criterion 2
        )
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={},
        )
        assert is_risk is True
        assert any("전체 점수" in r for r in reasons), (
            f"Expected criterion-1 reason '전체 점수', got: {reasons}"
        )
        assert any("Z-점수" in r for r in reasons), (
            f"Expected criterion-2 reason 'Z-점수', got: {reasons}"
        )

    def test_std_zero_skips_z_score_criterion(self):
        """class_std=0 → z criterion is skipped; score=0.60, all safe → (False, [])."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.60,
            z_score=0.0,
        )
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.0,     # zero std → z criterion skipped
            misconception_counts={},
        )
        assert is_risk is False
        assert reasons == []

    def test_empty_per_question_levels_does_not_trigger_criterion_3(self):
        """per_question_levels={} → criterion 3 must NOT fire (empty guard)."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.60,
            z_score=0.0,
            per_question_levels={},
            per_question_coverages={},
            per_question_scores={},
        )
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={},
        )
        # all() on an empty iterable returns True, so the implementation
        # MUST guard against empty per_question_levels
        assert not any("모든 문항" in r for r in reasons), (
            f"Criterion 3 must NOT fire for empty levels dict, got: {reasons}"
        )

    def test_empty_per_question_coverage_does_not_trigger_criterion_4_or_6(self):
        """per_question_coverages={} → neither criterion 4 nor criterion 6 fires."""
        from forma.professor_report_data import identify_at_risk

        row = _make_at_risk_row(
            overall_ensemble_mean=0.60,
            z_score=0.0,
            per_question_levels={},
            per_question_coverages={},
            per_question_scores={},
        )
        is_risk, reasons = identify_at_risk(
            row,
            class_mean=0.60,
            class_std=0.10,
            misconception_counts={},
        )
        assert not any("커버리지 0%" in r for r in reasons), (
            f"Criterion 4 must not fire for empty coverages, got: {reasons}"
        )
        assert not any("평균 커버리지" in r for r in reasons), (
            f"Criterion 6 must not fire for empty coverages, got: {reasons}"
        )


# ===========================================================================
# T049: Edge case tests for build_professor_report_data
# ===========================================================================


def _make_student_with_nan(
    student_id: str,
    real_name: str,
    student_number: str,
) -> object:
    """Build a StudentReportData where all question scores are NaN."""
    from forma.report_data_loader import StudentReportData, QuestionReportData

    return StudentReportData(
        student_id=student_id,
        real_name=real_name,
        student_number=student_number,
        class_name="1A",
        course_name="Biology",
        chapter_name="Chapter 1",
        week_num=1,
        questions=[
            QuestionReportData(
                question_sn=1,
                question_text="Q1",
                ensemble_score=float("nan"),
                understanding_level="Beginning",
                concept_coverage=float("nan"),
                llm_median_score=float("nan"),
                rasch_theta=0.0,
                misconceptions=[],
            ),
        ],
    )


class TestBuildProfessorReportDataEdgeCases:
    """T049: Edge case tests for build_professor_report_data."""

    def test_large_dataset_200_students(self):
        """build_professor_report_data handles 200 students without error."""
        from forma.report_data_loader import compute_class_distributions
        from forma.professor_report_data import build_professor_report_data

        students = []
        for i in range(200):
            score = round((i % 10) * 0.1, 1)  # 0.0, 0.1, ..., 0.9 repeated
            level = (
                "Advanced" if score >= 0.85
                else "Proficient" if score >= 0.65
                else "Developing" if score >= 0.45
                else "Beginning"
            )
            students.append(
                _make_student(
                    f"S{i:03d}", f"Student{i}", f"202600{i:03d}",
                    score, level, score, level,
                )
            )

        dists = compute_class_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )

        assert report.n_students == 200
        assert len(report.student_rows) == 200
        assert 0.0 <= report.class_ensemble_mean <= 1.0
        assert report.class_ensemble_std >= 0.0

    def test_exactly_3_students_minimum_threshold(self):
        """build_professor_report_data handles exactly 3 students (minimum threshold)."""
        from forma.report_data_loader import compute_class_distributions
        from forma.professor_report_data import build_professor_report_data

        students = _make_three_students()
        dists = compute_class_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )

        assert report.n_students == 3
        assert len(report.student_rows) == 3
        # Ensure stats are computed without error
        assert report.class_ensemble_mean > 0
        assert isinstance(report.class_ensemble_std, float)

    def test_nan_scores_do_not_crash(self):
        """build_professor_report_data handles NaN values in scores safely."""
        import math
        from forma.report_data_loader import StudentReportData, QuestionReportData, compute_class_distributions
        from forma.professor_report_data import build_professor_report_data

        nan = float("nan")
        students = []
        for i in range(3):
            students.append(StudentReportData(
                student_id=f"S{i}",
                real_name=f"Student{i}",
                student_number=f"2026{i:03d}",
                class_name="1A",
                course_name="Biology",
                chapter_name="Chapter 1",
                week_num=1,
                questions=[
                    QuestionReportData(
                        question_sn=1,
                        question_text="Q1",
                        ensemble_score=nan if i == 0 else 0.6,
                        understanding_level="Developing",
                        concept_coverage=nan if i == 0 else 0.5,
                        llm_median_score=2.0,
                        rasch_theta=0.0,
                        misconceptions=[],
                    ),
                ],
            ))

        dists = compute_class_distributions(students)
        # Must not raise
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )

        assert report.n_students == 3
        # class_ensemble_mean should be computed via nanmean — no NaN in result
        assert not math.isnan(report.class_ensemble_mean)

    def test_empty_concepts_and_misconceptions(self):
        """build_professor_report_data handles students with empty concept/misconception lists."""
        from forma.report_data_loader import StudentReportData, QuestionReportData, compute_class_distributions
        from forma.professor_report_data import build_professor_report_data

        students = []
        for i in range(3):
            students.append(StudentReportData(
                student_id=f"S{i}",
                real_name=f"Student{i}",
                student_number=f"2026{i:03d}",
                class_name="1A",
                course_name="Biology",
                chapter_name="Chapter 1",
                week_num=1,
                questions=[
                    QuestionReportData(
                        question_sn=1,
                        question_text="Q1",
                        ensemble_score=0.6 + i * 0.1,
                        understanding_level="Developing",
                        concept_coverage=0.5,
                        llm_median_score=2.0,
                        rasch_theta=0.0,
                        misconceptions=[],  # empty
                    ),
                ],
            ))

        dists = compute_class_distributions(students)
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )

        assert report.n_students == 3
        # concept_mastery_rates should be empty dict (no concepts)
        for qs in report.question_stats:
            assert isinstance(qs.concept_mastery_rates, dict)
            assert isinstance(qs.misconception_frequencies, list)

    def test_students_with_different_question_sets(self):
        """build_professor_report_data handles students with mismatched question coverage."""
        from forma.report_data_loader import StudentReportData, QuestionReportData, compute_class_distributions
        from forma.professor_report_data import build_professor_report_data

        # Student A answered Q1 and Q2; Student B answered only Q1

        students = [
            _make_student("SA", "Alice", "001", 0.7, "Proficient", 0.8, "Advanced"),
            # Student with only Q1 (missing Q2)
        ]

        # Build a student who only answered Q1
        students.append(StudentReportData(
            student_id="SB",
            real_name="Bob",
            student_number="002",
            class_name="1A",
            course_name="Biology",
            chapter_name="Chapter 1",
            week_num=1,
            questions=[
                QuestionReportData(
                    question_sn=1,
                    question_text="Q1",
                    ensemble_score=0.5,
                    understanding_level="Developing",
                    concept_coverage=0.4,
                    llm_median_score=2.0,
                    rasch_theta=0.0,
                    misconceptions=[],
                ),
            ],
        ))

        dists = compute_class_distributions(students)
        # Must not crash even with unequal question sets
        report = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
        )

        assert report.n_students == 2
        # Q2 only exists in SA's data, Q1 in both
        assert report.n_questions >= 1


# ===========================================================================
# T010: merge_professor_report_data() — multi-class merge tests
# ===========================================================================


def _make_report_for_section(
    class_name: str,
    student_specs: list[tuple[str, str, str]],
) -> object:
    """Build a ProfessorReportData for a given section with the given students.

    Args:
        class_name: Section identifier, e.g. "A" or "B".
        student_specs: List of (student_id, real_name, student_number).

    Returns:
        A ProfessorReportData instance for this section.
    """
    from forma.professor_report_data import build_professor_report_data
    from forma.report_data_loader import compute_class_distributions

    students = []
    for sid, name, num in student_specs:
        students.append(
            _make_student(
                sid, name, num,
                0.70, "Proficient", 0.60, "Proficient",
            )
        )
    dists = compute_class_distributions(students)
    return build_professor_report_data(
        students=students,
        distributions=dists,
        class_name=class_name,
        week_num=1,
        subject="Biology",
        exam_title="Test",
    )


class TestMergeProfessorReportData:
    """T010: merge_professor_report_data() multi-class merge tests.

    All tests in this class SHOULD FAIL (RED phase) because
    merge_professor_report_data does not yet exist in professor_report_data.py.
    """

    def test_merge_two_sections_student_count(self):
        """Merge 2 reports with 2+3 students → 5 total student_rows."""
        from forma.professor_report_data import merge_professor_report_data

        report_a = _make_report_for_section("A", [
            ("A1", "Alice", "001"),
            ("A2", "Aaron", "002"),
        ])
        report_b = _make_report_for_section("B", [
            ("B1", "Bob", "003"),
            ("B2", "Barbara", "004"),
            ("B3", "Brian", "005"),
        ])

        merged = merge_professor_report_data([report_a, report_b])

        assert len(merged.student_rows) == 5

    def test_merge_two_sections_class_name(self):
        """Merged class_name == 'A+B' and section_names == ['A', 'B']."""
        from forma.professor_report_data import merge_professor_report_data

        report_a = _make_report_for_section("A", [
            ("A1", "Alice", "001"),
        ])
        report_b = _make_report_for_section("B", [
            ("B1", "Bob", "002"),
        ])

        merged = merge_professor_report_data([report_a, report_b])

        assert merged.class_name == "A+B"
        assert merged.section_names == ["A", "B"]

    def test_merge_student_section_tagging(self):
        """Each student_row.section is set to the source class_name."""
        from forma.professor_report_data import merge_professor_report_data

        report_a = _make_report_for_section("A", [
            ("A1", "Alice", "001"),
            ("A2", "Aaron", "002"),
        ])
        report_b = _make_report_for_section("B", [
            ("B1", "Bob", "003"),
        ])

        merged = merge_professor_report_data([report_a, report_b])

        rows_by_id = {r.student_id: r for r in merged.student_rows}
        assert rows_by_id["A1"].section == "A"
        assert rows_by_id["A2"].section == "A"
        assert rows_by_id["B1"].section == "B"

    def test_merge_is_multi_class(self):
        """Merged report has is_multi_class=True."""
        from forma.professor_report_data import merge_professor_report_data

        report_a = _make_report_for_section("A", [
            ("A1", "Alice", "001"),
        ])
        report_b = _make_report_for_section("B", [
            ("B1", "Bob", "002"),
        ])

        merged = merge_professor_report_data([report_a, report_b])

        assert merged.is_multi_class is True

    def test_merge_question_stats_recalculated(self):
        """Merged QuestionClassStats has n_students reflecting total combined count.

        Specifically, for each question_sn, the sum of level_distribution values
        equals the total number of student_rows in the merged report.
        """
        from forma.professor_report_data import merge_professor_report_data

        report_a = _make_report_for_section("A", [
            ("A1", "Alice", "001"),
            ("A2", "Aaron", "002"),
        ])
        report_b = _make_report_for_section("B", [
            ("B1", "Bob", "003"),
            ("B2", "Barbara", "004"),
            ("B3", "Brian", "005"),
        ])

        merged = merge_professor_report_data([report_a, report_b])

        # Each question's level_distribution should sum to the total student count
        total_students = len(merged.student_rows)
        assert total_students == 5

        for qstat in merged.question_stats:
            level_sum = sum(qstat.level_distribution.values())
            assert level_sum == total_students, (
                f"Question {qstat.question_sn} level_distribution sums to "
                f"{level_sum}, expected {total_students}"
            )

    def test_merge_at_risk_re_identified(self):
        """at_risk_students list is present (re-computed) after merge.

        Specifically, n_at_risk and pct_at_risk are set on the merged report
        and are consistent with the merged student_rows.
        """
        from forma.professor_report_data import merge_professor_report_data

        report_a = _make_report_for_section("A", [
            ("A1", "Alice", "001"),
            ("A2", "Aaron", "002"),
        ])
        report_b = _make_report_for_section("B", [
            ("B1", "Bob", "003"),
        ])

        merged = merge_professor_report_data([report_a, report_b])

        # n_at_risk and pct_at_risk must be present and consistent
        flagged = sum(1 for r in merged.student_rows if r.is_at_risk)
        assert merged.n_at_risk == flagged
        expected_pct = (flagged / len(merged.student_rows)) * 100.0
        assert merged.pct_at_risk == pytest.approx(expected_pct, abs=1e-6)

    def test_merge_mismatched_question_sets(self):
        """ADV-006: section A has Q1+Q2, section B has only Q1.

        Merged question_stats should have Q1 with n_students from both,
        Q2 with n_students from A only (no crash, correct counts).
        """
        from forma.professor_report_data import build_professor_report_data, merge_professor_report_data
        from forma.report_data_loader import StudentReportData, QuestionReportData, compute_class_distributions

        # Section A: students with Q1 + Q2
        student_a1 = _make_student("A1", "Alice", "001", 0.70, "Proficient", 0.60, "Proficient")
        students_a = [student_a1]
        dists_a = compute_class_distributions(students_a)
        report_a = build_professor_report_data(
            students=students_a, distributions=dists_a,
            class_name="A", week_num=1, subject="Bio", exam_title="Test",
        )

        # Section B: student with only Q1
        student_b1 = StudentReportData(
            student_id="B1", real_name="Bob", student_number="003",
            class_name="B", course_name="Bio", chapter_name="Ch1", week_num=1,
            questions=[
                QuestionReportData(
                    question_sn=1, question_text="Q1", ensemble_score=0.5,
                    understanding_level="Developing", concept_coverage=0.4,
                    llm_median_score=2.0, rasch_theta=0.0, misconceptions=[],
                ),
            ],
        )
        students_b = [student_b1]
        dists_b = compute_class_distributions(students_b)
        report_b = build_professor_report_data(
            students=students_b, distributions=dists_b,
            class_name="B", week_num=1, subject="Bio", exam_title="Test",
        )

        # Must not crash
        merged = merge_professor_report_data([report_a, report_b])

        # Q1 should have level_distribution summing to 2 (both students)
        q1_stats = next((qs for qs in merged.question_stats if qs.question_sn == 1), None)
        assert q1_stats is not None
        assert sum(q1_stats.level_distribution.values()) == 2

        # Q2 should have level_distribution summing to 1 (only student A1)
        q2_stats = next((qs for qs in merged.question_stats if qs.question_sn == 2), None)
        assert q2_stats is not None
        assert sum(q2_stats.level_distribution.values()) == 1

    def test_merge_with_empty_section(self):
        """ADV-007: merge([report_with_students, report_with_no_students]).

        Should not crash; merged.n_students == len(report_with_students.student_rows).
        """
        from forma.professor_report_data import build_professor_report_data, merge_professor_report_data
        from forma.report_data_loader import compute_class_distributions

        # Section A with students
        report_a = _make_report_for_section("A", [
            ("A1", "Alice", "001"),
            ("A2", "Aaron", "002"),
        ])

        # Section B with no students — build an empty report manually
        students_b = []
        dists_b = compute_class_distributions(students_b)
        report_b = build_professor_report_data(
            students=students_b, distributions=dists_b,
            class_name="B", week_num=1, subject="Bio", exam_title="Test",
        )

        # Must not crash
        merged = merge_professor_report_data([report_a, report_b])

        # All students come from section A
        assert merged.n_students == len(report_a.student_rows)
        assert len(merged.student_rows) == len(report_a.student_rows)


# ===========================================================================
# T003 (v0.7.3): New fields — class_knowledge_aggregate, misconception_clusters,
#                 class_knowledge_aggregates
# ===========================================================================


class TestQuestionClassStatsNewFieldsV073:
    """T003: QuestionClassStats new fields for v0.7.3."""

    def test_class_knowledge_aggregate_default_none(self):
        """QuestionClassStats.class_knowledge_aggregate defaults to None."""
        from forma.professor_report_data import QuestionClassStats

        qcs = QuestionClassStats(question_sn=1)
        assert qcs.class_knowledge_aggregate is None

    def test_class_knowledge_aggregate_accepts_value(self):
        """QuestionClassStats.class_knowledge_aggregate can hold a ClassKnowledgeAggregate."""
        from forma.professor_report_data import QuestionClassStats
        from forma.class_knowledge_aggregate import ClassKnowledgeAggregate

        agg = ClassKnowledgeAggregate(question_sn=1, edges=[], total_students=0)
        qcs = QuestionClassStats(question_sn=1, class_knowledge_aggregate=agg)
        assert qcs.class_knowledge_aggregate is not None
        assert qcs.class_knowledge_aggregate.question_sn == 1

    def test_misconception_clusters_default_empty_list(self):
        """QuestionClassStats.misconception_clusters defaults to empty list."""
        from forma.professor_report_data import QuestionClassStats

        qcs = QuestionClassStats(question_sn=1)
        assert qcs.misconception_clusters == []

    def test_misconception_clusters_independent_across_instances(self):
        """QuestionClassStats.misconception_clusters list is independent per instance."""
        from forma.professor_report_data import QuestionClassStats

        qcs1 = QuestionClassStats(question_sn=1)
        qcs2 = QuestionClassStats(question_sn=2)
        qcs1.misconception_clusters.append("cluster")
        assert qcs2.misconception_clusters == []


class TestProfessorReportDataNewFieldsV073:
    """T003: ProfessorReportData new fields for v0.7.3."""

    def _make_minimal_report(self, **overrides):
        from forma.professor_report_data import ProfessorReportData

        defaults = dict(
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Week 1 Formative Test",
            generation_date="2026-03-08",
            n_students=3,
            n_questions=2,
            class_ensemble_mean=0.53,
            class_ensemble_std=0.23,
            class_ensemble_median=0.55,
            class_ensemble_q1=0.25,
            class_ensemble_q3=0.80,
            overall_level_distribution={k: 0 for k in CANONICAL_LEVELS},
            question_stats=[],
            student_rows=[],
            n_at_risk=0,
            pct_at_risk=0.0,
        )
        defaults.update(overrides)
        return ProfessorReportData(**defaults)

    def test_class_knowledge_aggregates_default_empty_list(self):
        """ProfessorReportData.class_knowledge_aggregates defaults to empty list."""
        report = self._make_minimal_report()
        assert report.class_knowledge_aggregates == []

    def test_class_knowledge_aggregates_independent_across_instances(self):
        """ProfessorReportData.class_knowledge_aggregates list is independent per instance."""
        report1 = self._make_minimal_report()
        report2 = self._make_minimal_report()
        report1.class_knowledge_aggregates.append("agg")
        assert report2.class_knowledge_aggregates == []

    def test_class_knowledge_aggregates_can_be_set(self):
        """ProfessorReportData.class_knowledge_aggregates accepts a list."""
        from forma.class_knowledge_aggregate import ClassKnowledgeAggregate

        agg = ClassKnowledgeAggregate(question_sn=1, edges=[], total_students=10)
        report = self._make_minimal_report(class_knowledge_aggregates=[agg])
        assert len(report.class_knowledge_aggregates) == 1
        assert report.class_knowledge_aggregates[0].total_students == 10

    def test_existing_tests_still_pass_with_new_defaults(self):
        """Backward compatibility: creating ProfessorReportData without new fields works."""
        report = self._make_minimal_report()
        # All existing required fields still accessible
        assert report.class_name == "1A"
        assert report.n_students == 3
        # New fields have safe defaults
        assert report.class_knowledge_aggregates == []


# ---------------------------------------------------------------------------
# Phase 4: US2 — T020: RiskMovement tests
# ---------------------------------------------------------------------------


class TestRiskMovement:
    """T020: RiskMovement dataclass and compute_risk_movement()."""

    def test_basic_risk_movement(self):
        """compute_risk_movement identifies newly at risk, exited, and persistent."""
        from forma.professor_report_data import RiskMovement, compute_risk_movement

        current_risk = {"s001", "s002", "s003"}
        previous_risk = {"s002", "s004"}

        movement = compute_risk_movement(current_risk, previous_risk)
        assert isinstance(movement, RiskMovement)
        assert sorted(movement.newly_at_risk) == ["s001", "s003"]
        assert movement.exited_risk == ["s004"]
        assert movement.persistent_risk == ["s002"]

    def test_no_previous_week(self):
        """When previous_risk is empty, all current are newly at risk."""
        from forma.professor_report_data import compute_risk_movement

        movement = compute_risk_movement({"s001", "s002"}, set())
        assert sorted(movement.newly_at_risk) == ["s001", "s002"]
        assert movement.exited_risk == []
        assert movement.persistent_risk == []

    def test_no_current_risk(self):
        """When no students are currently at risk, all previous have exited."""
        from forma.professor_report_data import compute_risk_movement

        movement = compute_risk_movement(set(), {"s001", "s002"})
        assert movement.newly_at_risk == []
        assert sorted(movement.exited_risk) == ["s001", "s002"]
        assert movement.persistent_risk == []

    def test_both_empty(self):
        """When both sets are empty, all lists should be empty."""
        from forma.professor_report_data import compute_risk_movement

        movement = compute_risk_movement(set(), set())
        assert movement.newly_at_risk == []
        assert movement.exited_risk == []
        assert movement.persistent_risk == []

    def test_identical_sets(self):
        """When current == previous, all are persistent, none new or exited."""
        from forma.professor_report_data import compute_risk_movement

        risk_set = {"s001", "s002"}
        movement = compute_risk_movement(risk_set, risk_set)
        assert movement.newly_at_risk == []
        assert movement.exited_risk == []
        assert sorted(movement.persistent_risk) == ["s001", "s002"]

    def test_risk_movement_sorted(self):
        """All lists in RiskMovement should be sorted by student_id."""
        from forma.professor_report_data import compute_risk_movement

        movement = compute_risk_movement(
            {"s003", "s001", "s005"}, {"s005", "s002", "s004"}
        )
        assert movement.newly_at_risk == sorted(movement.newly_at_risk)
        assert movement.exited_risk == sorted(movement.exited_risk)
        assert movement.persistent_risk == sorted(movement.persistent_risk)

    def test_risk_movement_dataclass_fields(self):
        """RiskMovement should have the expected fields."""
        from forma.professor_report_data import RiskMovement

        rm = RiskMovement(
            newly_at_risk=["s001"],
            exited_risk=["s002"],
            persistent_risk=["s003"],
        )
        assert rm.newly_at_risk == ["s001"]
        assert rm.exited_risk == ["s002"]
        assert rm.persistent_risk == ["s003"]
