"""Cross-section comparison statistics for aggregate professor reports.

Computes per-section descriptive statistics, pairwise statistical significance
tests (Welch's t-test for N>=30, Mann-Whitney U for N<30), Cohen's d effect
sizes, and Bonferroni multiple comparison correction.

Typical usage::

    stats_a = compute_section_stats("A", scores_a, at_risk_a)
    comparisons = compute_pairwise_comparisons({"A": scores_a, "B": scores_b})
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SectionStats:
    """Per-section descriptive statistics.

    Args:
        section_name: Section identifier (e.g., "A", "1B").
        n_students: Number of students in the section.
        mean: Mean ensemble score (0.0-1.0).
        median: Median ensemble score (0.0-1.0).
        std: Standard deviation of ensemble scores.
        n_at_risk: Number of at-risk students in the section.
        pct_at_risk: Fraction of at-risk students (0.0-1.0).
    """

    section_name: str
    n_students: int
    mean: float
    median: float
    std: float
    n_at_risk: int
    pct_at_risk: float


@dataclass
class SectionComparison:
    """Pairwise statistical comparison between two sections.

    Args:
        section_a: First section identifier.
        section_b: Second section identifier.
        n_a: Sample size of section A.
        n_b: Sample size of section B.
        mean_a: Mean score of section A.
        mean_b: Mean score of section B.
        std_a: Standard deviation of section A.
        std_b: Standard deviation of section B.
        test_name: Statistical test used ("welch_t" or "mann_whitney_u").
        test_statistic: Test statistic value.
        p_value: Raw p-value from the test.
        p_value_corrected: Bonferroni-corrected p-value; None if only 2 sections.
        cohens_d: Cohen's d effect size.
        effect_size_label: "negligible", "small", "medium", or "large".
        is_significant: Whether the comparison is significant (p < 0.05).
    """

    section_a: str
    section_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    test_name: str
    test_statistic: float
    p_value: float
    p_value_corrected: float | None
    cohens_d: float
    effect_size_label: str
    is_significant: bool


@dataclass
class CrossSectionReport:
    """Aggregate comparison data for all sections.

    Args:
        section_stats: Per-section descriptive statistics (>= 2 sections).
        pairwise_comparisons: All C(n,2) pairwise comparison results.
        concept_mastery_by_section: section -> concept -> mean mastery.
        weekly_interaction: section -> week -> mean_score; None if no
            longitudinal data is available.
    """

    section_stats: list[SectionStats]
    pairwise_comparisons: list[SectionComparison]
    concept_mastery_by_section: dict[str, dict[str, float]]
    weekly_interaction: dict[str, dict[int, float]] | None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation. Returns 0.0 if pooled_std is zero.

    Args:
        group1: Scores for the first group.
        group2: Scores for the second group.

    Returns:
        Cohen's d value (positive when group1 > group2).
    """
    a = np.array(group1, dtype=float)
    b = np.array(group2, dtype=float)
    n1, n2 = len(a), len(b)
    var1 = float(np.var(a, ddof=1)) if n1 > 1 else 0.0
    var2 = float(np.var(b, ddof=1)) if n2 > 1 else 0.0
    denom = n1 + n2 - 2
    if denom <= 0:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / denom)
    if pooled_std == 0.0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _effect_size_label(d: float) -> str:
    """Classify absolute Cohen's d into an effect size label.

    Args:
        d: Cohen's d value (sign is ignored).

    Returns:
        One of "negligible", "small", "medium", "large".
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_section_stats(
    section_name: str,
    ensemble_scores: list[float],
    at_risk_ids: set[str],
) -> SectionStats:
    """Compute descriptive statistics for a single section.

    Args:
        section_name: Section identifier.
        ensemble_scores: List of per-student ensemble scores.
        at_risk_ids: Set of student IDs flagged as at-risk.

    Returns:
        SectionStats with count, mean, median, std, and at-risk info.
    """
    arr = np.array(ensemble_scores, dtype=float)
    n = len(arr)
    n_at_risk = len(at_risk_ids)
    with np.errstate(all="ignore"):
        mean_val = float(np.nanmean(arr)) if n > 0 else 0.0
        median_val = float(np.nanmedian(arr)) if n > 0 else 0.0
        std_val = float(np.nanstd(arr, ddof=0)) if n > 0 else 0.0
    # Guard against all-NaN arrays
    if np.isnan(mean_val):
        mean_val = 0.0
    if np.isnan(median_val):
        median_val = 0.0
    if np.isnan(std_val):
        std_val = 0.0
    return SectionStats(
        section_name=section_name,
        n_students=n,
        mean=mean_val,
        median=median_val,
        std=std_val,
        n_at_risk=n_at_risk,
        pct_at_risk=n_at_risk / n if n > 0 else 0.0,
    )


def compute_pairwise_comparisons(
    section_scores: dict[str, list[float]],
) -> list[SectionComparison]:
    """Compute pairwise statistical comparisons between all section pairs.

    Automatically selects the appropriate test:
    - Welch's t-test when both sections have N >= 30
    - Mann-Whitney U test when either section has N < 30

    Applies Bonferroni correction when 3+ sections are compared.

    Args:
        section_scores: Mapping of section_name -> list of ensemble scores.

    Returns:
        List of SectionComparison for each C(n,2) pair, or empty list if
        fewer than 2 sections.
    """
    sections = sorted(section_scores.keys())
    n_sections = len(sections)

    if n_sections < 2:
        return []

    pairs = list(itertools.combinations(sections, 2))
    n_pairs = len(pairs)
    apply_bonferroni = n_sections >= 3

    results: list[SectionComparison] = []

    for sec_a, sec_b in pairs:
        scores_a = section_scores[sec_a]
        scores_b = section_scores[sec_b]
        n_a = len(scores_a)
        n_b = len(scores_b)
        arr_a = np.array(scores_a, dtype=float)
        arr_b = np.array(scores_b, dtype=float)

        mean_a = float(np.mean(arr_a))
        mean_b = float(np.mean(arr_b))
        std_a = float(np.std(arr_a, ddof=0))
        std_b = float(np.std(arr_b, ddof=0))

        # Select test based on minimum sample size
        if min(n_a, n_b) >= 30:
            test_result = stats.ttest_ind(arr_a, arr_b, equal_var=False)
            test_name = "welch_t"
            test_statistic = float(test_result.statistic)
            p_value = float(test_result.pvalue)
        else:
            test_result = stats.mannwhitneyu(
                arr_a,
                arr_b,
                alternative="two-sided",
            )
            test_name = "mann_whitney_u"
            test_statistic = float(test_result.statistic)
            p_value = float(test_result.pvalue)

        # Bonferroni correction
        p_corrected: float | None = None
        if apply_bonferroni:
            p_corrected = min(p_value * n_pairs, 1.0)

        # Effect size
        d = _cohens_d(scores_a, scores_b)
        label = _effect_size_label(d)

        # Significance determination
        p_for_sig = p_corrected if p_corrected is not None else p_value
        is_significant = p_for_sig < 0.05

        results.append(
            SectionComparison(
                section_a=sec_a,
                section_b=sec_b,
                n_a=n_a,
                n_b=n_b,
                mean_a=mean_a,
                mean_b=mean_b,
                std_a=std_a,
                std_b=std_b,
                test_name=test_name,
                test_statistic=test_statistic,
                p_value=p_value,
                p_value_corrected=p_corrected,
                cohens_d=d,
                effect_size_label=label,
                is_significant=is_significant,
            )
        )

    return results


def compute_concept_mastery_by_section(
    section_data: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float]]:
    """Compute mean concept mastery per section.

    Args:
        section_data: section_name -> concept_name -> list of per-student
            mastery values.

    Returns:
        section_name -> concept_name -> mean mastery value.
    """
    result: dict[str, dict[str, float]] = {}
    for section, concepts in section_data.items():
        result[section] = {}
        for concept, values in concepts.items():
            if values:
                result[section][concept] = float(np.mean(values))
            else:
                result[section][concept] = 0.0
    return result


def compute_weekly_interaction(
    section_scores: dict[str, dict[int, list[float]]] | None,
) -> dict[str, dict[int, float]] | None:
    """Compute per-section per-week mean scores for interaction analysis.

    Args:
        section_scores: section_name -> week -> list of per-student scores.
            None if no longitudinal data is available.

    Returns:
        section_name -> week -> mean_score, or None if input is None or empty.
    """
    if not section_scores:
        return None

    result: dict[str, dict[int, float]] = {}
    for section, weeks in section_scores.items():
        result[section] = {}
        for week, scores in weeks.items():
            if scores:
                result[section][week] = float(np.mean(scores))
            else:
                result[section][week] = 0.0
    return result
