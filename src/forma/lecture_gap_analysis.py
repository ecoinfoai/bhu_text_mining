"""Lecture gap analysis and cross-class emphasis comparison.

Compares master curriculum concepts against lecture coverage to identify
gaps, and computes cross-class emphasis variance for multi-section analysis.
No LLM imports are used in this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from forma.emphasis_map import InstructionalEmphasisMap

logger = logging.getLogger(__name__)


@dataclass
class LectureGapReport:
    """Report of gaps between master curriculum and lecture coverage.

    Args:
        master_concepts: Set of concepts in the master curriculum.
        covered_concepts: Set of master concepts covered in lecture.
        missed_concepts: Set of master concepts not covered in lecture.
        extra_concepts: Set of lecture concepts not in master curriculum.
        coverage_ratio: Fraction of master concepts covered (0.0 to 1.0).
        high_miss_overlap: Concepts missed in lecture AND with high
            student missing rate (>= 0.50).
    """

    master_concepts: set[str]
    covered_concepts: set[str]
    missed_concepts: set[str]
    extra_concepts: set[str]
    coverage_ratio: float
    high_miss_overlap: list[str] = field(default_factory=list)


def compute_lecture_gap(
    master_concepts: set[str],
    lecture_concepts: set[str],
    student_missing_rates: dict[str, float] | None = None,
    miss_threshold: float = 0.50,
) -> LectureGapReport:
    """Compute gap between master curriculum and lecture coverage.

    Args:
        master_concepts: Set of concepts in the master curriculum.
        lecture_concepts: Set of concepts mentioned in lecture transcript.
        student_missing_rates: Optional mapping of concept to student
            missing rate (fraction of students who missed the concept).
        miss_threshold: Threshold for high_miss_overlap (default 0.50).

    Returns:
        LectureGapReport with coverage statistics and gap details.
    """
    covered = master_concepts & lecture_concepts
    missed = master_concepts - lecture_concepts
    extra = lecture_concepts - master_concepts

    if len(master_concepts) > 0:
        coverage_ratio = len(covered) / len(master_concepts)
    else:
        coverage_ratio = 0.0

    high_miss_overlap: list[str] = []
    if student_missing_rates and missed:
        high_miss_overlap = sorted(
            concept for concept in missed
            if student_missing_rates.get(concept, 0.0) >= miss_threshold
        )

    return LectureGapReport(
        master_concepts=master_concepts,
        covered_concepts=covered,
        missed_concepts=missed,
        extra_concepts=extra,
        coverage_ratio=coverage_ratio,
        high_miss_overlap=high_miss_overlap,
    )


def compute_cross_class_emphasis_variance(
    class_emphasis_maps: dict[str, InstructionalEmphasisMap],
    top_n: int = 5,
) -> list[tuple[str, float, dict[str, float]]]:
    """Compute emphasis variance for top concepts across classes.

    For each concept that appears in any class's emphasis map, computes
    the standard deviation of emphasis scores across classes. Returns
    the top_n concepts sorted by stdev descending (FR-021).

    Requires at least 2 classes for meaningful variance; returns empty
    list for fewer than 2.

    Args:
        class_emphasis_maps: Mapping of class_name to InstructionalEmphasisMap.
        top_n: Number of top concepts to include in result.

    Returns:
        List of (concept, stdev, per_class_scores) tuples sorted by
        stdev descending. per_class_scores maps class_name to score.
    """
    if len(class_emphasis_maps) < 2:
        return []

    # Collect all unique concepts
    all_concepts: set[str] = set()
    for em in class_emphasis_maps.values():
        all_concepts.update(em.concept_scores.keys())

    # Compute per-concept stdev and per-class scores
    results: list[tuple[str, float, dict[str, float]]] = []
    for concept in all_concepts:
        per_class = {
            cls: em.concept_scores.get(concept, 0.0)
            for cls, em in class_emphasis_maps.items()
        }
        stdev = float(np.std(list(per_class.values())))
        results.append((concept, stdev, per_class))

    # Sort by stdev descending, take top_n
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
