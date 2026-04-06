"""Tests for lecture_gap_analysis.py — lecture gap analysis and cross-class comparison.

RED phase: these tests are written *before* the module exists and will fail
until ``src/forma/lecture_gap_analysis.py`` is implemented.

Covers US3 (FR-013 ~ FR-018, SC-004):
  - LectureGapReport dataclass
  - compute_lecture_gap(): master vs lecture concept gap analysis
  - compute_cross_class_emphasis_variance(): cross-class emphasis variance
"""

from __future__ import annotations

import pytest

from forma.lecture_gap_analysis import (
    LectureGapReport,
    compute_cross_class_emphasis_variance,
    compute_lecture_gap,
)


# ---------------------------------------------------------------------------
# FR-013: LectureGapReport dataclass
# ---------------------------------------------------------------------------


class TestLectureGapReport:
    """FR-013: LectureGapReport has correct fields."""

    def test_fields_exist(self):
        report = LectureGapReport(
            master_concepts={"A", "B", "C"},
            covered_concepts={"A", "B"},
            missed_concepts={"C"},
            extra_concepts={"D"},
            coverage_ratio=2 / 3,
            high_miss_overlap=[],
        )
        assert report.master_concepts == {"A", "B", "C"}
        assert report.covered_concepts == {"A", "B"}
        assert report.missed_concepts == {"C"}
        assert report.extra_concepts == {"D"}
        assert report.coverage_ratio == pytest.approx(2 / 3)
        assert report.high_miss_overlap == []

    def test_all_covered(self):
        report = LectureGapReport(
            master_concepts={"A", "B"},
            covered_concepts={"A", "B"},
            missed_concepts=set(),
            extra_concepts=set(),
            coverage_ratio=1.0,
            high_miss_overlap=[],
        )
        assert report.coverage_ratio == 1.0
        assert len(report.missed_concepts) == 0


# ---------------------------------------------------------------------------
# FR-014: compute_lecture_gap() coverage ratio
# ---------------------------------------------------------------------------


class TestComputeLectureGap:
    """FR-014: compute_lecture_gap computes coverage correctly."""

    def test_exact_coverage_ratio(self):
        """24 master / 18 covered → 75% exact."""
        master = {f"c{i}" for i in range(24)}
        lecture = {f"c{i}" for i in range(18)}  # 18 overlap with master

        result = compute_lecture_gap(master, lecture)
        assert result.coverage_ratio == pytest.approx(18 / 24)

    def test_missed_concepts(self):
        """Missed concepts = master - lecture."""
        master = {"A", "B", "C", "D"}
        lecture = {"A", "B"}

        result = compute_lecture_gap(master, lecture)
        assert result.missed_concepts == {"C", "D"}

    def test_extra_concepts(self):
        """Extra concepts = lecture - master."""
        master = {"A", "B"}
        lecture = {"A", "B", "C", "D"}

        result = compute_lecture_gap(master, lecture)
        assert result.extra_concepts == {"C", "D"}

    def test_covered_concepts(self):
        """Covered = master ∩ lecture."""
        master = {"A", "B", "C"}
        lecture = {"B", "C", "D"}

        result = compute_lecture_gap(master, lecture)
        assert result.covered_concepts == {"B", "C"}

    def test_empty_master(self):
        """Empty master → coverage 0.0, all lecture concepts are extra."""
        result = compute_lecture_gap(set(), {"A", "B"})
        assert result.coverage_ratio == 0.0
        assert result.extra_concepts == {"A", "B"}
        assert result.missed_concepts == set()

    def test_empty_lecture(self):
        """Empty lecture → coverage 0.0, all master concepts missed."""
        result = compute_lecture_gap({"A", "B"}, set())
        assert result.coverage_ratio == 0.0
        assert result.missed_concepts == {"A", "B"}

    def test_both_empty(self):
        """Both empty → coverage 0.0."""
        result = compute_lecture_gap(set(), set())
        assert result.coverage_ratio == 0.0


# ---------------------------------------------------------------------------
# FR-015: high_miss_overlap
# ---------------------------------------------------------------------------


class TestHighMissOverlap:
    """FR-015: student missing rates >= 0.50 → high_miss_overlap."""

    def test_high_miss_overlap_detected(self):
        """Concepts missed in lecture AND with student_missing_rate >= 0.50."""
        master = {"A", "B", "C", "D"}
        lecture = {"A", "B"}  # C, D missed

        student_missing_rates = {"C": 0.60, "D": 0.40, "A": 0.10}
        result = compute_lecture_gap(
            master,
            lecture,
            student_missing_rates=student_missing_rates,
        )
        # C missed in lecture and student_missing_rate >= 0.50 → overlap
        assert "C" in result.high_miss_overlap
        # D missed in lecture but student_missing_rate < 0.50 → no overlap
        assert "D" not in result.high_miss_overlap

    def test_no_overlap_when_all_covered(self):
        """No overlap when all master concepts are covered."""
        master = {"A", "B"}
        lecture = {"A", "B"}

        student_missing_rates = {"A": 0.80, "B": 0.90}
        result = compute_lecture_gap(
            master,
            lecture,
            student_missing_rates=student_missing_rates,
        )
        assert result.high_miss_overlap == []

    def test_no_student_rates_no_overlap(self):
        """Without student_missing_rates → empty high_miss_overlap."""
        master = {"A", "B", "C"}
        lecture = {"A"}

        result = compute_lecture_gap(master, lecture)
        assert result.high_miss_overlap == []


# ---------------------------------------------------------------------------
# FR-016: compute_cross_class_emphasis_variance()
# ---------------------------------------------------------------------------


class TestCrossClassEmphasisVariance:
    """FR-016/FR-021: cross-class emphasis variance sorted by stdev desc."""

    def test_basic_variance(self):
        """Stdev computed for top 5 concepts, sorted by stdev descending."""
        from forma.emphasis_map import InstructionalEmphasisMap
        import numpy as np

        class_maps = {
            "1A": InstructionalEmphasisMap(
                concept_scores={"A": 0.9, "B": 0.5, "C": 0.3},
                threshold_used=0.65,
                n_sentences=10,
                n_concepts=3,
            ),
            "1B": InstructionalEmphasisMap(
                concept_scores={"A": 0.7, "B": 0.8, "C": 0.2},
                threshold_used=0.65,
                n_sentences=10,
                n_concepts=3,
            ),
        }
        result = compute_cross_class_emphasis_variance(class_maps, top_n=5)
        # Result is list of (concept, stdev, per_class_scores)
        assert isinstance(result, list)
        assert len(result) == 3
        # Each entry is (str, float, dict)
        concept, stdev, per_class = result[0]
        assert isinstance(concept, str)
        assert isinstance(stdev, float)
        assert isinstance(per_class, dict)
        # Verify stdev of B: stdev([0.5, 0.8]) = 0.15
        b_entry = [r for r in result if r[0] == "B"][0]
        assert b_entry[1] == pytest.approx(float(np.std([0.5, 0.8])))
        # Verify per_class scores for B
        assert b_entry[2] == {"1A": 0.5, "1B": 0.8}
        # Sorted by stdev descending: first entry has highest stdev
        stdevs = [r[1] for r in result]
        assert stdevs == sorted(stdevs, reverse=True)

    def test_empty_input(self):
        """Empty class_maps → empty list."""
        result = compute_cross_class_emphasis_variance({})
        assert result == []

    def test_single_class(self):
        """Single class → empty list (< 2 classes)."""
        from forma.emphasis_map import InstructionalEmphasisMap

        class_maps = {
            "1A": InstructionalEmphasisMap(
                concept_scores={"A": 0.9, "B": 0.5},
                threshold_used=0.65,
                n_sentences=10,
                n_concepts=2,
            ),
        }
        result = compute_cross_class_emphasis_variance(class_maps)
        assert result == []

    def test_top_n_limits_output(self):
        """top_n=2 limits output to 2 concepts."""
        from forma.emphasis_map import InstructionalEmphasisMap

        class_maps = {
            "1A": InstructionalEmphasisMap(
                concept_scores={"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1},
                threshold_used=0.65,
                n_sentences=10,
                n_concepts=5,
            ),
            "1B": InstructionalEmphasisMap(
                concept_scores={"A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2, "E": 0.0},
                threshold_used=0.65,
                n_sentences=10,
                n_concepts=5,
            ),
        }
        result = compute_cross_class_emphasis_variance(class_maps, top_n=2)
        assert len(result) == 2
        # Sorted by stdev descending
        assert result[0][1] >= result[1][1]


# ---------------------------------------------------------------------------
# SC-004: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """SC-004: Edge cases for lecture gap analysis."""

    def test_perfect_overlap(self):
        """Identical master and lecture → 100% coverage, no gaps."""
        concepts = {"A", "B", "C"}
        result = compute_lecture_gap(concepts, concepts)
        assert result.coverage_ratio == 1.0
        assert result.missed_concepts == set()
        assert result.extra_concepts == set()

    def test_no_overlap(self):
        """Completely disjoint sets → 0% coverage."""
        result = compute_lecture_gap({"A", "B"}, {"C", "D"})
        assert result.coverage_ratio == 0.0
        assert result.missed_concepts == {"A", "B"}
        assert result.extra_concepts == {"C", "D"}
