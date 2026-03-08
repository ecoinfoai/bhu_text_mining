"""Professor class summary report data model and builder.

Provides dataclasses and build functions for generating a professor-facing
class summary PDF report from student evaluation results.
"""

from __future__ import annotations

import datetime
import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from reportlab.lib.colors import Color

from forma.report_data_loader import ClassDistributions, StudentReportData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Level ordering for tie-break (lower index = lower level)
# ---------------------------------------------------------------------------

_LEVEL_ORDER: dict[str, int] = {
    "Beginning": 0,
    "Developing": 1,
    "Proficient": 2,
    "Advanced": 3,
}

_CANONICAL_LEVELS = ("Advanced", "Proficient", "Developing", "Beginning")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QuestionClassStats:
    """Aggregated class-level statistics for a single exam question.

    Args:
        question_sn: Question serial number (1-based).
        question_text: Question text from exam config.
        topic: Topic/section from exam config.
        ensemble_mean: Mean ensemble score for this question.
        ensemble_std: Standard deviation of ensemble scores.
        ensemble_median: Median ensemble score.
        concept_coverage_mean: Mean concept coverage across class.
        llm_score_mean: Mean LLM rubric score (normalized).
        rasch_theta_mean: Mean Rasch ability estimate.
        level_distribution: Count of students at each understanding level.
        concept_mastery_rates: Per-concept fraction of students demonstrating mastery.
        misconception_frequencies: (misconception_text, frequency) pairs, sorted desc.
    """

    question_sn: int
    question_text: str = ""
    topic: str = ""
    ensemble_mean: float = 0.0
    ensemble_std: float = 0.0
    ensemble_median: float = 0.0
    concept_coverage_mean: float = 0.0
    llm_score_mean: float = 0.0
    rasch_theta_mean: float = 0.0
    level_distribution: dict[str, int] = field(default_factory=dict)
    concept_mastery_rates: dict[str, float] = field(default_factory=dict)
    misconception_frequencies: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class StudentSummaryRow:
    """Summary row for a single student in the class comparison table.

    Args:
        student_id: Anonymous student identifier (e.g., "S015").
        student_number: Student number (학번).
        real_name: Student's real name.
        overall_ensemble_mean: Mean of per-question ensemble scores.
        overall_level: Mode of per-question levels; tie-break to lower level.
        per_question_scores: Ensemble score per question {question_sn: score}.
        per_question_levels: Understanding level per question {question_sn: level}.
        per_question_coverages: Concept coverage per question {question_sn: coverage}.
        is_at_risk: Whether student meets any at-risk criterion.
        at_risk_reasons: Human-readable reasons for at-risk flag.
        z_score: Standardized score relative to class mean.
    """

    student_id: str
    student_number: str = ""
    real_name: str = ""
    overall_ensemble_mean: float = 0.0
    overall_level: str = "Beginning"
    per_question_scores: dict[int, float] = field(default_factory=dict)
    per_question_levels: dict[int, str] = field(default_factory=dict)
    per_question_coverages: dict[int, float] = field(default_factory=dict)
    is_at_risk: bool = False
    at_risk_reasons: list[str] = field(default_factory=list)
    z_score: float = 0.0


@dataclass
class ProfessorReportData:
    """Root entity representing the complete data for one professor class report.

    Args:
        class_name: Class/section identifier (e.g., "1A").
        week_num: Week number of the assessment.
        subject: Course subject name.
        exam_title: Exam configuration title.
        generation_date: ISO 8601 date string for report generation.
        n_students: Total number of students in dataset.
        n_questions: Total number of questions in exam.
        class_ensemble_mean: Class mean of per-student overall ensemble scores.
        class_ensemble_std: Class standard deviation of per-student overall scores.
        class_ensemble_median: Class median of per-student overall scores.
        class_ensemble_q1: First quartile (25th percentile).
        class_ensemble_q3: Third quartile (75th percentile).
        overall_level_distribution: Count per canonical level across all students.
        question_stats: Per-question aggregated statistics.
        student_rows: Student comparison rows, sorted by overall_ensemble_mean desc.
        n_at_risk: Number of at-risk students.
        pct_at_risk: Percentage of at-risk students.
        overall_assessment: LLM-generated or fallback overall assessment text.
        teaching_suggestions: LLM-generated or fallback teaching suggestions.
        llm_model_used: Model identifier string.
        llm_generation_failed: True if any LLM call failed.
        llm_error_message: Error details if LLM failed.
    """

    class_name: str
    week_num: int
    subject: str
    exam_title: str
    generation_date: str
    n_students: int
    n_questions: int
    class_ensemble_mean: float
    class_ensemble_std: float
    class_ensemble_median: float
    class_ensemble_q1: float
    class_ensemble_q3: float
    overall_level_distribution: dict[str, int]
    question_stats: list[QuestionClassStats]
    student_rows: list[StudentSummaryRow]
    n_at_risk: int
    pct_at_risk: float
    overall_assessment: str = ""
    teaching_suggestions: str = ""
    llm_model_used: str = ""
    llm_generation_failed: bool = False
    llm_error_message: str = ""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_overall_level(levels: list[str]) -> str:
    """Compute mode of understanding levels with conservative tie-break.

    Tie-break: when multiple levels share the highest count, the lower
    level (smaller index in _LEVEL_ORDER) wins.

    Args:
        levels: List of understanding level strings.

    Returns:
        The modal level, or "Beginning" if levels is empty.
    """
    if not levels:
        return "Beginning"

    counts = Counter(levels)
    max_count = max(counts.values())
    # Collect all levels with max count, sort by level order ascending (lower wins)
    tied = [lvl for lvl, cnt in counts.items() if cnt == max_count]
    tied.sort(key=lambda lvl: _LEVEL_ORDER.get(lvl, 0))
    return tied[0]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def identify_at_risk(
    row: StudentSummaryRow,
    class_mean: float,
    class_std: float,
    misconception_counts: dict[int, int],
) -> tuple[bool, list[str]]:
    """Determine whether a student is at risk based on 6 criteria (OR logic).

    Args:
        row: Student summary row with scores and level data.
        class_mean: Class mean of overall ensemble scores.
        class_std: Class standard deviation of overall ensemble scores.
        misconception_counts: Dict mapping question_sn (int) to misconception
            count (int) for this student.

    Returns:
        Tuple of (is_at_risk: bool, reasons: list[str]).
    """
    reasons: list[str] = []

    # Criterion 1: overall_ensemble_mean < 0.45
    if row.overall_ensemble_mean < 0.45:
        reasons.append(
            f"전체 점수 {row.overall_ensemble_mean:.2f} (기준: 0.45 미만)"
        )

    # Criterion 2: z_score < -1.0 (skip if class_std == 0)
    if class_std != 0 and row.z_score < -1.0:
        reasons.append(f"Z-점수 {row.z_score:.2f} (기준: -1.0 미만)")

    # Criterion 3: all per_question_levels are "Beginning" (not empty)
    if (
        len(row.per_question_levels) > 0
        and all(v == "Beginning" for v in row.per_question_levels.values())
    ):
        reasons.append("모든 문항 '하' 수준")

    # Criterion 4: any per_question_coverage == 0.0 (skip if empty)
    if row.per_question_coverages:
        for sn, cov in row.per_question_coverages.items():
            if cov == 0.0:
                reasons.append(f"문항 {sn}번 커버리지 0%")

    # Criterion 5: total misconceptions >= 3 AND overall_level != "Advanced"
    total_misconceptions = sum(misconception_counts.values())
    if total_misconceptions >= 3 and row.overall_level != "Advanced":
        reasons.append(
            f"오개념 {total_misconceptions}개 (기준: 3개 이상, 비Advanced)"
        )

    # Criterion 6: mean coverage < 0.30 (skip if empty)
    if len(row.per_question_coverages) > 0:
        avg = sum(row.per_question_coverages.values()) / len(
            row.per_question_coverages
        )
        if avg < 0.30:
            reasons.append(f"평균 커버리지 {avg:.1%} (기준: 30% 미만)")

    return (len(reasons) > 0, reasons)


def compute_conditional_indicator(score: float, mean: float, std: float) -> str:
    """Return '+' if score >= mean+0.5*std, '-' if score <= mean-0.5*std, else ''.

    Args:
        score: The score to compare.
        mean: Class mean.
        std: Class standard deviation.

    Returns:
        '+' if score >= mean + 0.5*std,
        '-' if score <= mean - 0.5*std,
        '' otherwise or if std == 0.
    """
    if std == 0:
        return ""
    if score >= mean + 0.5 * std:
        return "+"
    if score <= mean - 0.5 * std:
        return "-"
    return ""


def get_conditional_bg_color(score: float, mean: float, std: float) -> "Color":
    """Return HexColor for conditional formatting based on ±0.5 SD threshold.

    Args:
        score: The score to compare.
        mean: Class mean.
        std: Class standard deviation.

    Returns:
        HexColor('#E8F5E9') (green) if score >= mean + 0.5*std,
        HexColor('#FFEBEE') (red)   if score <= mean - 0.5*std,
        white otherwise or if std == 0.
    """
    from reportlab.lib.colors import HexColor, white
    indicator = compute_conditional_indicator(score, mean, std)
    if indicator == "+":
        return HexColor("#E8F5E9")
    if indicator == "-":
        return HexColor("#FFEBEE")
    return white


def build_professor_report_data(
    students: list[StudentReportData],
    distributions: ClassDistributions,
    class_name: str,
    week_num: int,
    subject: str,
    exam_title: str,
) -> ProfessorReportData:
    """Build a ProfessorReportData from student data and class distributions.

    Args:
        students: List of StudentReportData with fully populated questions.
        distributions: ClassDistributions computed from the same students.
        class_name: Class/section identifier (e.g., "1A").
        week_num: Week number of the assessment.
        subject: Course subject name.
        exam_title: Exam configuration title.

    Returns:
        ProfessorReportData ready for PDF generation and LLM analysis.
    """
    # ------------------------------------------------------------------
    # Step 1: Collect per-student overall ensemble means
    # ------------------------------------------------------------------
    student_overall_means: list[float] = []
    for student in students:
        if student.questions:
            q_scores = [q.ensemble_score for q in student.questions]
            student_mean = float(np.nanmean(q_scores))
        else:
            student_mean = 0.0
        student_overall_means.append(student_mean)

    # ------------------------------------------------------------------
    # Step 2: Compute class statistics
    # ------------------------------------------------------------------
    arr = np.array(student_overall_means, dtype=float)

    if len(arr) == 0:
        class_mean = 0.0
        class_std = 0.0
        class_median = 0.0
        class_q1 = 0.0
        class_q3 = 0.0
    else:
        class_mean = float(np.nanmean(arr))
        class_std = float(np.nanstd(arr))  # ddof=0 (population std)
        class_median = float(np.nanmedian(arr))
        class_q1 = float(np.nanpercentile(arr, 25))
        class_q3 = float(np.nanpercentile(arr, 75))

    # ------------------------------------------------------------------
    # Step 3: Compute z-scores
    # ------------------------------------------------------------------
    if class_std == 0.0:
        z_scores = [0.0] * len(students)
    else:
        z_scores = [
            (mean - class_mean) / class_std for mean in student_overall_means
        ]

    # ------------------------------------------------------------------
    # Step 4: Collect unique question_sn set
    # ------------------------------------------------------------------
    all_question_sns: set[int] = set()
    for student in students:
        for q in student.questions:
            all_question_sns.add(q.question_sn)
    sorted_question_sns = sorted(all_question_sns)

    # ------------------------------------------------------------------
    # Step 5: Build QuestionClassStats for each question_sn
    # ------------------------------------------------------------------
    question_stats: list[QuestionClassStats] = []

    for qsn in sorted_question_sns:
        # Score stats from distributions
        ens_scores = distributions.ensemble_scores.get(qsn, [])
        cov_scores = distributions.concept_coverages.get(qsn, [])
        llm_scores = distributions.llm_scores.get(qsn, [])
        rasch_thetas = distributions.rasch_thetas.get(qsn, [])

        if ens_scores:
            ens_arr = np.array(ens_scores, dtype=float)
            q_ensemble_mean = float(np.nanmean(ens_arr))
            q_ensemble_std = float(np.nanstd(ens_arr))
            q_ensemble_median = float(np.nanmedian(ens_arr))
        else:
            q_ensemble_mean = 0.0
            q_ensemble_std = 0.0
            q_ensemble_median = 0.0

        q_concept_coverage_mean = (
            float(np.nanmean(np.array(cov_scores, dtype=float))) if cov_scores else 0.0
        )
        q_llm_score_mean = (
            float(np.nanmean(np.array(llm_scores, dtype=float))) if llm_scores else 0.0
        )
        q_rasch_theta_mean = (
            float(np.nanmean(np.array(rasch_thetas, dtype=float)))
            if rasch_thetas
            else 0.0
        )

        # Level distribution for this question
        level_dist: dict[str, int] = {lvl: 0 for lvl in _CANONICAL_LEVELS}
        # Concept mastery rates: concept_name -> [is_present booleans]
        concept_present_map: dict[str, list[bool]] = {}
        # Misconception counts
        misconception_counter: Counter[str] = Counter()
        # question_text and topic from first student that has them
        q_text = ""
        q_topic = ""

        for student in students:
            for q in student.questions:
                if q.question_sn != qsn:
                    continue

                # Level distribution
                lvl = q.understanding_level
                if lvl in level_dist:
                    level_dist[lvl] += 1

                # Concept mastery rates
                for concept_detail in q.concepts:
                    concept_present_map.setdefault(
                        concept_detail.concept, []
                    ).append(concept_detail.is_present)

                # Misconceptions
                for m in q.misconceptions:
                    misconception_counter[m] += 1

                # question_text and topic: use first non-empty
                if not q_text and q.question_text:
                    q_text = q.question_text
                if not q_topic and hasattr(q, "topic") and q.topic:
                    q_topic = q.topic

        # Build concept_mastery_rates
        concept_mastery_rates: dict[str, float] = {
            concept: sum(1 for v in presence_list if v) / len(presence_list)
            for concept, presence_list in concept_present_map.items()
            if presence_list
        }

        # Build misconception_frequencies sorted desc by count
        misconception_frequencies: list[tuple[str, int]] = sorted(
            misconception_counter.items(), key=lambda x: x[1], reverse=True
        )

        question_stats.append(
            QuestionClassStats(
                question_sn=qsn,
                question_text=q_text,
                topic=q_topic,
                ensemble_mean=q_ensemble_mean,
                ensemble_std=q_ensemble_std,
                ensemble_median=q_ensemble_median,
                concept_coverage_mean=q_concept_coverage_mean,
                llm_score_mean=q_llm_score_mean,
                rasch_theta_mean=q_rasch_theta_mean,
                level_distribution=level_dist,
                concept_mastery_rates=concept_mastery_rates,
                misconception_frequencies=misconception_frequencies,
            )
        )

    # ------------------------------------------------------------------
    # Step 6: Build StudentSummaryRows
    # ------------------------------------------------------------------
    student_rows: list[StudentSummaryRow] = []

    for idx, student in enumerate(students):
        per_question_scores: dict[int, float] = {}
        per_question_levels: dict[int, str] = {}
        per_question_coverages: dict[int, float] = {}
        misconception_counts: dict[int, int] = {}

        for q in student.questions:
            per_question_scores[q.question_sn] = q.ensemble_score
            per_question_levels[q.question_sn] = q.understanding_level
            per_question_coverages[q.question_sn] = q.concept_coverage
            misconception_counts[q.question_sn] = len(q.misconceptions)

        # overall_ensemble_mean
        if per_question_scores:
            overall_mean = float(np.nanmean(list(per_question_scores.values())))
        else:
            overall_mean = 0.0

        # overall_level: mode of per_question_levels with tie-break to lower
        overall_level = _compute_overall_level(list(per_question_levels.values()))

        z_score = z_scores[idx]

        # Build partial row for at-risk identification
        row = StudentSummaryRow(
            student_id=student.student_id,
            student_number=student.student_number,
            real_name=student.real_name,
            overall_ensemble_mean=overall_mean,
            overall_level=overall_level,
            per_question_scores=per_question_scores,
            per_question_levels=per_question_levels,
            per_question_coverages=per_question_coverages,
            is_at_risk=False,
            at_risk_reasons=[],
            z_score=z_score,
        )

        is_at_risk, at_risk_reasons = identify_at_risk(
            row,
            class_mean=class_mean,
            class_std=class_std,
            misconception_counts=misconception_counts,
        )
        row.is_at_risk = is_at_risk
        row.at_risk_reasons = at_risk_reasons

        student_rows.append(row)

    # ------------------------------------------------------------------
    # Step 7: Sort student_rows by overall_ensemble_mean descending
    # ------------------------------------------------------------------
    student_rows.sort(key=lambda r: r.overall_ensemble_mean, reverse=True)

    # ------------------------------------------------------------------
    # Step 8: Compute overall_level_distribution
    # ------------------------------------------------------------------
    overall_level_distribution: dict[str, int] = {lvl: 0 for lvl in _CANONICAL_LEVELS}
    for row in student_rows:
        if row.overall_level in overall_level_distribution:
            overall_level_distribution[row.overall_level] += 1

    # ------------------------------------------------------------------
    # Step 9: Compute n_at_risk and pct_at_risk
    # ------------------------------------------------------------------
    n_at_risk = sum(1 for r in student_rows if r.is_at_risk)
    n_students = len(students)
    pct_at_risk = (n_at_risk / n_students * 100.0) if n_students > 0 else 0.0

    # ------------------------------------------------------------------
    # Step 10: Return ProfessorReportData
    # ------------------------------------------------------------------
    return ProfessorReportData(
        class_name=class_name,
        week_num=week_num,
        subject=subject,
        exam_title=exam_title,
        generation_date=datetime.date.today().isoformat(),
        n_students=n_students,
        n_questions=len(sorted_question_sns),
        class_ensemble_mean=class_mean,
        class_ensemble_std=class_std,
        class_ensemble_median=class_median,
        class_ensemble_q1=class_q1,
        class_ensemble_q3=class_q3,
        overall_level_distribution=overall_level_distribution,
        question_stats=question_stats,
        student_rows=student_rows,
        n_at_risk=n_at_risk,
        pct_at_risk=pct_at_risk,
    )
