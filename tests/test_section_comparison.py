"""Tests for section_comparison.py — Cross-Section Comparison (US4).

T049: Tests for SectionStats, SectionComparison, CrossSectionReport dataclasses
and compute_section_stats(), compute_pairwise_comparisons(),
compute_concept_mastery_by_section().

Covers:
- Descriptive stats computation
- Welch's t-test for N>=30
- Mann-Whitney U for N<30
- Cohen's d calculation (including zero-variance -> 0.0)
- effect_size_label thresholds (negligible/small/medium/large)
- Bonferroni correction for 3+ sections
- 2 sections no correction
- Single section returns None/empty
- Zero-variance section edge case
- Unequal sample sizes
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# SectionStats dataclass
# ---------------------------------------------------------------------------


class TestSectionStats:
    """Tests for SectionStats dataclass."""

    def test_creation_basic(self):
        from forma.section_comparison import SectionStats

        ss = SectionStats(
            section_name="A",
            n_students=30,
            mean=0.72,
            median=0.75,
            std=0.12,
            n_at_risk=5,
            pct_at_risk=5 / 30,
        )
        assert ss.section_name == "A"
        assert ss.n_students == 30
        assert ss.mean == pytest.approx(0.72)
        assert ss.median == pytest.approx(0.75)
        assert ss.std == pytest.approx(0.12)
        assert ss.n_at_risk == 5
        assert ss.pct_at_risk == pytest.approx(5 / 30)

    def test_zero_at_risk(self):
        from forma.section_comparison import SectionStats

        ss = SectionStats(
            section_name="B",
            n_students=20,
            mean=0.85,
            median=0.88,
            std=0.05,
            n_at_risk=0,
            pct_at_risk=0.0,
        )
        assert ss.n_at_risk == 0
        assert ss.pct_at_risk == 0.0


# ---------------------------------------------------------------------------
# SectionComparison dataclass
# ---------------------------------------------------------------------------


class TestSectionComparison:
    """Tests for SectionComparison dataclass."""

    def test_creation_welch(self):
        from forma.section_comparison import SectionComparison

        sc = SectionComparison(
            section_a="A",
            section_b="B",
            n_a=35,
            n_b=40,
            mean_a=0.72,
            mean_b=0.68,
            std_a=0.12,
            std_b=0.15,
            test_name="welch_t",
            test_statistic=2.1,
            p_value=0.038,
            p_value_corrected=None,
            cohens_d=0.29,
            effect_size_label="small",
            is_significant=True,
        )
        assert sc.test_name == "welch_t"
        assert sc.p_value_corrected is None
        assert sc.is_significant is True

    def test_creation_mann_whitney(self):
        from forma.section_comparison import SectionComparison

        sc = SectionComparison(
            section_a="C",
            section_b="D",
            n_a=15,
            n_b=20,
            mean_a=0.6,
            mean_b=0.65,
            std_a=0.1,
            std_b=0.11,
            test_name="mann_whitney_u",
            test_statistic=120.0,
            p_value=0.25,
            p_value_corrected=0.75,
            cohens_d=-0.47,
            effect_size_label="small",
            is_significant=False,
        )
        assert sc.test_name == "mann_whitney_u"
        assert sc.p_value_corrected == pytest.approx(0.75)
        assert sc.is_significant is False


# ---------------------------------------------------------------------------
# CrossSectionReport dataclass
# ---------------------------------------------------------------------------


class TestCrossSectionReport:
    """Tests for CrossSectionReport dataclass."""

    def test_creation(self):
        from forma.section_comparison import (
            CrossSectionReport,
            SectionComparison,
            SectionStats,
        )

        report = CrossSectionReport(
            section_stats=[
                SectionStats("A", 30, 0.7, 0.72, 0.1, 3, 0.1),
                SectionStats("B", 25, 0.65, 0.66, 0.12, 5, 0.2),
            ],
            pairwise_comparisons=[
                SectionComparison(
                    "A", "B", 30, 25, 0.7, 0.65, 0.1, 0.12,
                    "welch_t", 1.8, 0.07, None, 0.45, "small", False,
                ),
            ],
            concept_mastery_by_section={
                "A": {"cell_membrane": 0.8, "mitochondria": 0.6},
                "B": {"cell_membrane": 0.7, "mitochondria": 0.5},
            },
            weekly_interaction=None,
        )
        assert len(report.section_stats) == 2
        assert len(report.pairwise_comparisons) == 1
        assert report.weekly_interaction is None


# ---------------------------------------------------------------------------
# compute_section_stats()
# ---------------------------------------------------------------------------


class TestComputeSectionStats:
    """Tests for compute_section_stats function."""

    def test_basic_stats(self):
        from forma.section_comparison import compute_section_stats

        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        at_risk_ids = {"S001"}
        result = compute_section_stats("A", scores, at_risk_ids)

        assert result.section_name == "A"
        assert result.n_students == 5
        assert result.mean == pytest.approx(0.7)
        assert result.median == pytest.approx(0.7)
        assert result.std == pytest.approx(np.std(scores, ddof=0), abs=1e-6)
        assert result.n_at_risk == 1
        assert result.pct_at_risk == pytest.approx(0.2)

    def test_single_student(self):
        from forma.section_comparison import compute_section_stats

        result = compute_section_stats("X", [0.75], set())
        assert result.n_students == 1
        assert result.mean == pytest.approx(0.75)
        assert result.median == pytest.approx(0.75)
        assert result.std == pytest.approx(0.0)
        assert result.n_at_risk == 0

    def test_all_at_risk(self):
        from forma.section_comparison import compute_section_stats

        at_risk = {"S1", "S2", "S3"}
        result = compute_section_stats("Z", [0.3, 0.2, 0.4], at_risk)
        assert result.n_at_risk == 3
        assert result.pct_at_risk == pytest.approx(1.0)

    def test_no_at_risk(self):
        from forma.section_comparison import compute_section_stats

        result = compute_section_stats("OK", [0.8, 0.9], set())
        assert result.n_at_risk == 0
        assert result.pct_at_risk == pytest.approx(0.0)

    def test_zero_variance(self):
        """All students with identical scores."""
        from forma.section_comparison import compute_section_stats

        result = compute_section_stats("Same", [0.5, 0.5, 0.5], set())
        assert result.std == pytest.approx(0.0)
        assert result.mean == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_pairwise_comparisons()
# ---------------------------------------------------------------------------


class TestComputePairwiseComparisons:
    """Tests for compute_pairwise_comparisons function."""

    def test_two_sections_large_n_welch(self):
        """Both N>=30 -> Welch's t-test, no Bonferroni correction."""
        from forma.section_comparison import compute_pairwise_comparisons

        rng = np.random.default_rng(42)
        scores = {
            "A": list(rng.normal(0.7, 0.1, 35)),
            "B": list(rng.normal(0.6, 0.12, 40)),
        }
        results = compute_pairwise_comparisons(scores)

        assert len(results) == 1
        comp = results[0]
        assert comp.section_a == "A"
        assert comp.section_b == "B"
        assert comp.test_name == "welch_t"
        assert comp.n_a == 35
        assert comp.n_b == 40
        assert comp.p_value_corrected is None  # Only 2 sections
        assert 0.0 <= comp.p_value <= 1.0
        assert isinstance(comp.cohens_d, float)
        assert comp.effect_size_label in ("negligible", "small", "medium", "large")

    def test_two_sections_small_n_mann_whitney(self):
        """Either N<30 -> Mann-Whitney U test."""
        from forma.section_comparison import compute_pairwise_comparisons

        rng = np.random.default_rng(42)
        scores = {
            "X": list(rng.normal(0.7, 0.1, 15)),
            "Y": list(rng.normal(0.5, 0.15, 20)),
        }
        results = compute_pairwise_comparisons(scores)

        assert len(results) == 1
        comp = results[0]
        assert comp.test_name == "mann_whitney_u"

    def test_three_sections_bonferroni(self):
        """3 sections = C(3,2)=3 pairs, Bonferroni correction applied."""
        from forma.section_comparison import compute_pairwise_comparisons

        rng = np.random.default_rng(42)
        scores = {
            "A": list(rng.normal(0.7, 0.1, 35)),
            "B": list(rng.normal(0.6, 0.12, 40)),
            "C": list(rng.normal(0.65, 0.11, 30)),
        }
        results = compute_pairwise_comparisons(scores)

        assert len(results) == 3  # C(3,2) = 3
        for comp in results:
            assert comp.p_value_corrected is not None
            assert comp.p_value_corrected >= comp.p_value
            assert comp.p_value_corrected <= 1.0

    def test_four_sections_bonferroni(self):
        """4 sections = C(4,2)=6 pairs, Bonferroni correction."""
        from forma.section_comparison import compute_pairwise_comparisons

        rng = np.random.default_rng(42)
        scores = {
            "A": list(rng.normal(0.7, 0.1, 35)),
            "B": list(rng.normal(0.6, 0.12, 40)),
            "C": list(rng.normal(0.65, 0.11, 30)),
            "D": list(rng.normal(0.55, 0.13, 32)),
        }
        results = compute_pairwise_comparisons(scores)

        assert len(results) == 6  # C(4,2) = 6
        for comp in results:
            assert comp.p_value_corrected is not None

    def test_mixed_sample_sizes(self):
        """One section N>=30, other N<30 -> Mann-Whitney U."""
        from forma.section_comparison import compute_pairwise_comparisons

        rng = np.random.default_rng(42)
        scores = {
            "Big": list(rng.normal(0.7, 0.1, 50)),
            "Small": list(rng.normal(0.6, 0.12, 10)),
        }
        results = compute_pairwise_comparisons(scores)

        assert len(results) == 1
        assert results[0].test_name == "mann_whitney_u"

    def test_n_exactly_30_uses_welch(self):
        """Both sections N=30 -> Welch's t-test (boundary case)."""
        from forma.section_comparison import compute_pairwise_comparisons

        rng = np.random.default_rng(42)
        scores = {
            "A": list(rng.normal(0.7, 0.1, 30)),
            "B": list(rng.normal(0.6, 0.1, 30)),
        }
        results = compute_pairwise_comparisons(scores)

        assert results[0].test_name == "welch_t"

    def test_n_29_uses_mann_whitney(self):
        """One section N=29 -> Mann-Whitney U (boundary case)."""
        from forma.section_comparison import compute_pairwise_comparisons

        rng = np.random.default_rng(42)
        scores = {
            "A": list(rng.normal(0.7, 0.1, 29)),
            "B": list(rng.normal(0.6, 0.1, 30)),
        }
        results = compute_pairwise_comparisons(scores)

        assert results[0].test_name == "mann_whitney_u"

    def test_zero_variance_cohens_d(self):
        """Zero variance in both groups -> Cohen's d = 0.0."""
        from forma.section_comparison import compute_pairwise_comparisons

        scores = {
            "Same1": [0.5] * 30,
            "Same2": [0.5] * 30,
        }
        results = compute_pairwise_comparisons(scores)

        assert results[0].cohens_d == pytest.approx(0.0)
        assert results[0].effect_size_label == "negligible"

    def test_zero_variance_one_group(self):
        """Zero variance in one group -> Cohen's d = 0.0."""
        from forma.section_comparison import compute_pairwise_comparisons

        rng = np.random.default_rng(42)
        scores = {
            "ZeroVar": [0.5] * 30,
            "Normal": list(rng.normal(0.7, 0.1, 30)),
        }
        results = compute_pairwise_comparisons(scores)

        # pooled std will be non-zero because one group has variance
        assert isinstance(results[0].cohens_d, float)
        assert not math.isnan(results[0].cohens_d)

    def test_effect_size_labels(self):
        """Verify effect size label thresholds."""
        from forma.section_comparison import _effect_size_label

        assert _effect_size_label(0.0) == "negligible"
        assert _effect_size_label(0.1) == "negligible"
        assert _effect_size_label(0.19) == "negligible"
        assert _effect_size_label(0.2) == "small"
        assert _effect_size_label(0.49) == "small"
        assert _effect_size_label(0.5) == "medium"
        assert _effect_size_label(0.79) == "medium"
        assert _effect_size_label(0.8) == "large"
        assert _effect_size_label(1.5) == "large"

    def test_significance_determination(self):
        """Significant when p_value (or corrected) < 0.05."""
        from forma.section_comparison import compute_pairwise_comparisons

        rng = np.random.default_rng(42)
        # Large effect, large N -> should be significant
        scores = {
            "High": list(rng.normal(0.9, 0.05, 50)),
            "Low": list(rng.normal(0.3, 0.05, 50)),
        }
        results = compute_pairwise_comparisons(scores)

        comp = results[0]
        assert comp.p_value < 0.05
        assert comp.is_significant is True

    def test_cohens_d_direction(self):
        """Cohen's d is positive when A > B."""
        from forma.section_comparison import compute_pairwise_comparisons

        rng = np.random.default_rng(42)
        scores = {
            "A": list(rng.normal(0.8, 0.05, 40)),
            "B": list(rng.normal(0.5, 0.05, 40)),
        }
        results = compute_pairwise_comparisons(scores)

        assert results[0].cohens_d > 0

    def test_empty_sections_dict(self):
        """Empty dict returns empty list."""
        from forma.section_comparison import compute_pairwise_comparisons

        results = compute_pairwise_comparisons({})
        assert results == []

    def test_single_section_returns_empty(self):
        """Single section returns empty list (no pairs)."""
        from forma.section_comparison import compute_pairwise_comparisons

        results = compute_pairwise_comparisons({"A": [0.5, 0.6, 0.7]})
        assert results == []

    def test_probability_bounds(self):
        """p_value and p_value_corrected are in [0, 1]."""
        from forma.section_comparison import compute_pairwise_comparisons

        rng = np.random.default_rng(42)
        scores = {
            "A": list(rng.normal(0.7, 0.1, 35)),
            "B": list(rng.normal(0.6, 0.1, 35)),
            "C": list(rng.normal(0.5, 0.1, 35)),
        }
        results = compute_pairwise_comparisons(scores)

        for comp in results:
            assert 0.0 <= comp.p_value <= 1.0
            if comp.p_value_corrected is not None:
                assert 0.0 <= comp.p_value_corrected <= 1.0

    def test_unequal_sample_sizes(self):
        """Works with very different N between sections."""
        from forma.section_comparison import compute_pairwise_comparisons

        rng = np.random.default_rng(42)
        scores = {
            "Large": list(rng.normal(0.7, 0.1, 100)),
            "Tiny": list(rng.normal(0.6, 0.12, 5)),
        }
        results = compute_pairwise_comparisons(scores)

        assert len(results) == 1
        comp = results[0]
        assert comp.n_a == 100
        assert comp.n_b == 5
        assert comp.test_name == "mann_whitney_u"  # 5 < 30


# ---------------------------------------------------------------------------
# compute_concept_mastery_by_section()
# ---------------------------------------------------------------------------


class TestComputeConceptMasteryBySection:
    """Tests for compute_concept_mastery_by_section function."""

    def test_basic(self):
        from forma.section_comparison import compute_concept_mastery_by_section

        # section_data: dict[str, dict[str, list[float]]]
        # section -> concept -> list of per-student mastery values
        section_data = {
            "A": {"cell_membrane": [0.8, 0.9, 0.7], "mitochondria": [0.5, 0.6]},
            "B": {"cell_membrane": [0.6, 0.7], "mitochondria": [0.4, 0.3, 0.5]},
        }
        result = compute_concept_mastery_by_section(section_data)

        assert result["A"]["cell_membrane"] == pytest.approx(0.8)
        assert result["A"]["mitochondria"] == pytest.approx(0.55)
        assert result["B"]["cell_membrane"] == pytest.approx(0.65)
        assert result["B"]["mitochondria"] == pytest.approx(0.4)

    def test_empty_section_data(self):
        from forma.section_comparison import compute_concept_mastery_by_section

        result = compute_concept_mastery_by_section({})
        assert result == {}

    def test_single_section_single_concept(self):
        from forma.section_comparison import compute_concept_mastery_by_section

        section_data = {
            "A": {"concept_x": [1.0]},
        }
        result = compute_concept_mastery_by_section(section_data)
        assert result["A"]["concept_x"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_weekly_interaction()
# ---------------------------------------------------------------------------


class TestComputeWeeklyInteraction:
    """Tests for compute_weekly_interaction function."""

    def test_basic(self):
        from forma.section_comparison import compute_weekly_interaction

        # section_scores: dict[str, dict[int, list[float]]]
        # section -> week -> list of per-student scores
        section_scores = {
            "A": {1: [0.6, 0.7], 2: [0.7, 0.8], 3: [0.8, 0.9]},
            "B": {1: [0.5, 0.6], 2: [0.55, 0.65], 3: [0.6, 0.7]},
        }
        result = compute_weekly_interaction(section_scores)

        assert result is not None
        assert result["A"][1] == pytest.approx(0.65)
        assert result["A"][2] == pytest.approx(0.75)
        assert result["B"][3] == pytest.approx(0.65)

    def test_empty_returns_none(self):
        from forma.section_comparison import compute_weekly_interaction

        result = compute_weekly_interaction({})
        assert result is None

    def test_none_input_returns_none(self):
        from forma.section_comparison import compute_weekly_interaction

        result = compute_weekly_interaction(None)
        assert result is None


# ---------------------------------------------------------------------------
# cohens_d helper
# ---------------------------------------------------------------------------


class TestCohensD:
    """Tests for the cohens_d computation helper."""

    def test_identical_groups(self):
        from forma.section_comparison import _cohens_d

        assert _cohens_d([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == pytest.approx(0.0)

    def test_known_value(self):
        """Manual calculation check."""
        from forma.section_comparison import _cohens_d

        g1 = [0.8, 0.9, 0.7]
        g2 = [0.5, 0.6, 0.4]
        # mean diff = 0.3
        # var1 = np.var([0.8,0.9,0.7], ddof=1) = 0.01
        # var2 = np.var([0.5,0.6,0.4], ddof=1) = 0.01
        # pooled_std = sqrt((2*0.01 + 2*0.01) / 4) = sqrt(0.01) = 0.1
        # d = 0.3 / 0.1 = 3.0
        d = _cohens_d(g1, g2)
        assert d == pytest.approx(3.0, abs=0.01)

    def test_negative_d(self):
        """Cohen's d is negative when group1 < group2."""
        from forma.section_comparison import _cohens_d

        d = _cohens_d([0.3, 0.4], [0.8, 0.9])
        assert d < 0

    def test_zero_variance_both(self):
        from forma.section_comparison import _cohens_d

        assert _cohens_d([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)

    def test_zero_pooled_std_different_means(self):
        """Zero variance in both groups but different means -> d = 0.0."""
        from forma.section_comparison import _cohens_d

        # All 0.3 vs All 0.7 -> zero variance each, pooled_std = 0 -> d = 0.0
        d = _cohens_d([0.3, 0.3], [0.7, 0.7])
        assert d == pytest.approx(0.0)
