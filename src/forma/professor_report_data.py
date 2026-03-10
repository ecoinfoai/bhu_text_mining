"""Professor class summary report data model and builder.

Provides dataclasses and build functions for generating a professor-facing
class summary PDF report from student evaluation results.
"""

from __future__ import annotations

import copy
import datetime
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from reportlab.lib.colors import Color

from forma.report_data_loader import ClassDistributions, StudentReportData
from forma.emphasis_map import InstructionalEmphasisMap
from forma.lecture_gap_analysis import LectureGapReport

if TYPE_CHECKING:
    from forma.class_knowledge_aggregate import ClassKnowledgeAggregate
    from forma.misconception_clustering import MisconceptionCluster
    from forma.section_comparison import CrossSectionReport

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
    hub_gap_entries: list = field(default_factory=list)
    classified_misconceptions: list = field(default_factory=list)
    class_knowledge_aggregate: ClassKnowledgeAggregate | None = None
    misconception_clusters: list[MisconceptionCluster] = field(default_factory=list)


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
    section: str = ""
    misconception_counts: dict = field(default_factory=dict)


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
    is_multi_class: bool = False
    section_names: list[str] = field(default_factory=list)
    lecture_gap_report: LectureGapReport | None = None
    emphasis_map: InstructionalEmphasisMap | None = None
    class_emphasis_maps: dict[str, InstructionalEmphasisMap] | None = None
    class_knowledge_aggregates: list[ClassKnowledgeAggregate] = field(default_factory=list)
    risk_movement: "RiskMovement | None" = None
    cross_section_report: "CrossSectionReport | None" = None
    risk_predictions: list | None = None
    grade_predictions: list | None = None


@dataclass
class RiskMovement:
    """Risk group movement between weeks.

    Args:
        newly_at_risk: Students newly at risk this week (sorted).
        exited_risk: Students who exited risk this week (sorted).
        persistent_risk: Students at risk in both weeks (sorted).
    """

    newly_at_risk: list[str] = field(default_factory=list)
    exited_risk: list[str] = field(default_factory=list)
    persistent_risk: list[str] = field(default_factory=list)


def compute_risk_movement(
    current_risk: set[str],
    previous_risk: set[str],
) -> RiskMovement:
    """Compute risk movement between current and previous week.

    Args:
        current_risk: Set of student IDs currently at risk.
        previous_risk: Set of student IDs at risk in the previous week.

    Returns:
        RiskMovement with sorted lists.
    """
    return RiskMovement(
        newly_at_risk=sorted(current_risk - previous_risk),
        exited_risk=sorted(previous_risk - current_risk),
        persistent_risk=sorted(current_risk & previous_risk),
    )


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


def merge_professor_report_data(
    reports: list["ProfessorReportData"],
) -> "ProfessorReportData":
    """Merge multiple per-class ProfessorReportData into one combined report.

    Args:
        reports: List of per-class ProfessorReportData instances to merge.

    Returns:
        A single ProfessorReportData with all students combined, statistics
        recalculated, and is_multi_class=True.
    """
    if not reports:
        raise ValueError("reports must be non-empty")

    # ------------------------------------------------------------------
    # Step 1: Deep-copy each student row, tag with source class_name (section),
    #         and collect all student rows (ADV-001: avoid mutating originals)
    # ------------------------------------------------------------------
    all_student_rows: list[StudentSummaryRow] = []
    for report in reports:
        for row in report.student_rows:
            new_row = copy.deepcopy(row)
            new_row.section = report.class_name
            all_student_rows.append(new_row)

    # ------------------------------------------------------------------
    # Step 2: Compute combined class statistics
    # ------------------------------------------------------------------
    student_overall_means = [row.overall_ensemble_mean for row in all_student_rows]
    arr = np.array(student_overall_means, dtype=float)

    if len(arr) == 0:
        class_mean = 0.0
        class_std = 0.0
        class_median = 0.0
        class_q1 = 0.0
        class_q3 = 0.0
    else:
        class_mean = float(np.nanmean(arr))
        class_std = float(np.nanstd(arr))
        class_median = float(np.nanmedian(arr))
        class_q1 = float(np.nanpercentile(arr, 25))
        class_q3 = float(np.nanpercentile(arr, 75))

    # ------------------------------------------------------------------
    # Step 3: Recompute z-scores for all students from combined distribution
    # ------------------------------------------------------------------
    if class_std == 0.0:
        for row in all_student_rows:
            row.z_score = 0.0
    else:
        for row in all_student_rows:
            row.z_score = (row.overall_ensemble_mean - class_mean) / class_std

    # ------------------------------------------------------------------
    # Step 4: Re-identify at-risk using combined class stats
    #         Use stored misconception_counts from per-class build (ADV-002)
    # ------------------------------------------------------------------
    for row in all_student_rows:
        is_at_risk, at_risk_reasons = identify_at_risk(
            row,
            class_mean=class_mean,
            class_std=class_std,
            misconception_counts=row.misconception_counts,
        )
        row.is_at_risk = is_at_risk
        row.at_risk_reasons = at_risk_reasons

    # ------------------------------------------------------------------
    # Step 5: Sort merged student rows by overall_ensemble_mean desc
    # ------------------------------------------------------------------
    all_student_rows.sort(key=lambda r: r.overall_ensemble_mean, reverse=True)

    # ------------------------------------------------------------------
    # Step 6: Collect all unique question_sn values across all reports
    # ------------------------------------------------------------------
    all_question_sns: set[int] = set()
    for row in all_student_rows:
        all_question_sns.update(row.per_question_scores.keys())
    sorted_question_sns = sorted(all_question_sns)

    # ------------------------------------------------------------------
    # Step 7: Build merged QuestionClassStats for each question_sn
    # ------------------------------------------------------------------
    # Gather existing QuestionClassStats from all reports by question_sn
    question_stats_by_sn: dict[int, list] = {}
    for report in reports:
        for qstat in report.question_stats:
            question_stats_by_sn.setdefault(qstat.question_sn, []).append(qstat)

    merged_question_stats: list[QuestionClassStats] = []

    for qsn in sorted_question_sns:
        # Gather per-student data for this question
        ens_scores: list[float] = []
        cov_scores: list[float] = []
        level_dist: dict[str, int] = {lvl: 0 for lvl in _CANONICAL_LEVELS}

        for row in all_student_rows:
            if qsn in row.per_question_scores:
                ens_scores.append(row.per_question_scores[qsn])
            if qsn in row.per_question_coverages:
                cov_scores.append(row.per_question_coverages[qsn])
            if qsn in row.per_question_levels:
                lvl = row.per_question_levels[qsn]
                if lvl in level_dist:
                    level_dist[lvl] += 1

        # Compute score stats
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

        # Take question metadata, concept mastery rates, hub_gap_entries,
        # llm_score_mean, rasch_theta_mean from per-report question stats
        q_text = ""
        q_topic = ""
        q_llm_score_mean = 0.0
        q_rasch_theta_mean = 0.0
        concept_mastery_map: dict[str, list[float]] = {}
        misconception_counter: Counter[str] = Counter()
        hub_gap_by_concept: dict[str, list[float]] = {}

        existing_qstats = question_stats_by_sn.get(qsn, [])
        for qstat in existing_qstats:
            if not q_text and qstat.question_text:
                q_text = qstat.question_text
            if not q_topic and qstat.topic:
                q_topic = qstat.topic
            # Weighted average llm/rasch by student count in that section
            q_llm_score_mean += qstat.llm_score_mean
            q_rasch_theta_mean += qstat.rasch_theta_mean
            # Merge concept mastery rates
            for concept, rate in qstat.concept_mastery_rates.items():
                concept_mastery_map.setdefault(concept, []).append(rate)
            # Merge misconception frequencies
            for text, count in qstat.misconception_frequencies:
                misconception_counter[text] += count
            # Merge hub_gap_entries by concept
            for entry in qstat.hub_gap_entries:
                concept_key = getattr(entry, "concept", str(entry))
                rate = getattr(entry, "class_inclusion_rate", 0.0)
                hub_gap_by_concept.setdefault(concept_key, []).append(rate)

        n_qstats = len(existing_qstats) if existing_qstats else 1
        q_llm_score_mean /= n_qstats
        q_rasch_theta_mean /= n_qstats

        # Average concept mastery rates across sections
        concept_mastery_rates: dict[str, float] = {
            concept: sum(rates) / len(rates)
            for concept, rates in concept_mastery_map.items()
            if rates
        }

        # Sort misconception frequencies desc
        misconception_frequencies: list[tuple[str, int]] = sorted(
            misconception_counter.items(), key=lambda x: x[1], reverse=True
        )

        # Merge hub_gap_entries: deduplicate by concept, average inclusion rate
        merged_hub_gap: list = []
        for entry_concept, rates in hub_gap_by_concept.items():
            # Reconstruct a simple entry from the first qstat that has this concept
            for qstat in existing_qstats:
                for entry in qstat.hub_gap_entries:
                    if getattr(entry, "concept", str(entry)) == entry_concept:
                        # Clone entry and set averaged rate if possible
                        try:
                            new_entry = copy.copy(entry)
                            new_entry.class_inclusion_rate = sum(rates) / len(rates)
                            merged_hub_gap.append(new_entry)
                        except Exception as exc:
                            logger.warning("hub_gap merge error: %s", exc)
                            merged_hub_gap.append(entry)
                        break
                else:
                    continue
                break

        merged_question_stats.append(
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
                hub_gap_entries=merged_hub_gap,
            )
        )

    # ------------------------------------------------------------------
    # Step 8: Compute overall_level_distribution
    # ------------------------------------------------------------------
    overall_level_distribution: dict[str, int] = {lvl: 0 for lvl in _CANONICAL_LEVELS}
    for row in all_student_rows:
        if row.overall_level in overall_level_distribution:
            overall_level_distribution[row.overall_level] += 1

    # ------------------------------------------------------------------
    # Step 9: Compute n_at_risk and pct_at_risk
    # ------------------------------------------------------------------
    n_at_risk = sum(1 for r in all_student_rows if r.is_at_risk)
    n_students = len(all_student_rows)
    pct_at_risk = (n_at_risk / n_students * 100.0) if n_students > 0 else 0.0

    # ------------------------------------------------------------------
    # Step 10: Build section_names and combined class_name
    # ------------------------------------------------------------------
    section_names = [r.class_name for r in reports]
    combined_class_name = "+".join(section_names)

    # ------------------------------------------------------------------
    # Step 11: Take metadata from first report
    # ------------------------------------------------------------------
    first = reports[0]

    return ProfessorReportData(
        class_name=combined_class_name,
        week_num=first.week_num,
        subject=first.subject,
        exam_title=first.exam_title,
        generation_date=datetime.date.today().isoformat(),
        n_students=n_students,
        n_questions=len(sorted_question_sns),
        class_ensemble_mean=class_mean,
        class_ensemble_std=class_std,
        class_ensemble_median=class_median,
        class_ensemble_q1=class_q1,
        class_ensemble_q3=class_q3,
        overall_level_distribution=overall_level_distribution,
        question_stats=merged_question_stats,
        student_rows=all_student_rows,
        n_at_risk=n_at_risk,
        pct_at_risk=pct_at_risk,
        is_multi_class=len(reports) > 1,
        section_names=section_names,
    )


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
            misconception_counts=misconception_counts,
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
