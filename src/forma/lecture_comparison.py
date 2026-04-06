"""Cross-section lecture comparison analysis.

Compares multiple section analyses to identify exclusive keywords,
concept gaps, and emphasis variance across sections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import statistics

import yaml

from forma.io_utils import atomic_write_yaml
from forma.lecture_analyzer import AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class EmphasisVarianceEntry:
    """Single entry in emphasis variance ranking.

    Args:
        concept: Concept name.
        variance: Standard deviation of emphasis scores across sections.
        per_section_scores: Mapping of section ID to emphasis score.
    """

    concept: str
    variance: float
    per_section_scores: dict[str, float]


@dataclass
class ComparisonResult:
    """Cross-section comparison output.

    Args:
        comparison_type: Type of comparison ("session" or "class").
        sections_compared: Sorted list of section IDs compared.
        exclusive_keywords: Per-section list of keywords exclusive to that section's top-N.
        concept_gaps: Per-section list of missed concepts, or None if no concepts provided.
        emphasis_variance: Emphasis variance entries sorted by stdev descending.
        comparison_timestamp: ISO 8601 timestamp of comparison.
    """

    comparison_type: str
    sections_compared: list[str]
    exclusive_keywords: dict[str, list[str]]
    concept_gaps: dict[str, list[str]] | None
    emphasis_variance: list[EmphasisVarianceEntry]
    comparison_timestamp: str = ""


def compare_sections(
    analyses: dict[str, AnalysisResult],
    concepts: list[str] | None = None,
    top_n: int = 50,
    comparison_type: str = "session",
) -> ComparisonResult:
    """Compare multiple section analyses.

    Validates at least 2 sections (FR-016).
    Computes exclusive top-N keywords (FR-017).
    Computes concept gaps when concepts provided (FR-018).
    Computes emphasis variance (FR-019).

    Args:
        analyses: Mapping of section ID to AnalysisResult.
        concepts: Optional list of master concepts for gap analysis.
        top_n: Number of top keywords to consider for exclusivity.
        comparison_type: Type label ("session" or "class").

    Returns:
        ComparisonResult with all comparison data.

    Raises:
        ValueError: If fewer than 2 sections provided.
    """
    if len(analyses) < 2:
        raise ValueError(f"At least 2 sections required for comparison. Provided: {len(analyses)}")

    # 1. Exclusive keywords (FR-017)
    section_top_n: dict[str, set[str]] = {}
    for section_id, result in analyses.items():
        sorted_kw = sorted(
            result.keyword_frequencies.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        section_top_n[section_id] = {kw for kw, _ in sorted_kw[:top_n]}

    exclusive_keywords: dict[str, list[str]] = {}
    for section_id, top_set in section_top_n.items():
        other_top_sets: set[str] = set()
        for other_id, other_set in section_top_n.items():
            if other_id != section_id:
                other_top_sets |= other_set
        exclusive = sorted(top_set - other_top_sets)
        exclusive_keywords[section_id] = exclusive

    # 2. Concept gaps (FR-018)
    concept_gaps: dict[str, list[str]] | None = None
    if concepts:
        concept_gaps = {}
        for section_id, result in analyses.items():
            if result.concept_coverage:
                covered = set(result.concept_coverage.covered_concepts)
                missed = [c for c in concepts if c not in covered]
                concept_gaps[section_id] = missed
            else:
                concept_gaps[section_id] = list(concepts)

    # 3. Emphasis variance (FR-019)
    emphasis_variance: list[EmphasisVarianceEntry] = []
    all_concepts_set: set[str] = set()
    section_emphasis: dict[str, dict[str, float]] = {}
    for section_id, result in analyses.items():
        if result.emphasis_scores:
            section_emphasis[section_id] = result.emphasis_scores
            all_concepts_set.update(result.emphasis_scores.keys())

    if len(section_emphasis) >= 2:
        variance_entries: list[EmphasisVarianceEntry] = []
        for concept in all_concepts_set:
            scores = {sid: se.get(concept, 0.0) for sid, se in section_emphasis.items()}
            stdev = statistics.pstdev(list(scores.values()))
            variance_entries.append(
                EmphasisVarianceEntry(
                    concept=concept,
                    variance=stdev,
                    per_section_scores=scores,
                )
            )
        variance_entries.sort(key=lambda e: e.variance, reverse=True)
        emphasis_variance = variance_entries[:top_n]

    return ComparisonResult(
        comparison_type=comparison_type,
        sections_compared=sorted(analyses.keys()),
        exclusive_keywords=exclusive_keywords,
        concept_gaps=concept_gaps,
        emphasis_variance=emphasis_variance,
        comparison_timestamp=datetime.now(timezone.utc).isoformat(),
    )


def save_comparison_result(
    result: ComparisonResult,
    output_dir: Path,
    prefix: str = "comparison",
) -> Path:
    """Serialize ComparisonResult to a YAML file.

    Args:
        result: The comparison result to save.
        output_dir: Directory to write the YAML file.
        prefix: Filename prefix.

    Returns:
        Path to the saved YAML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sections_str = "_".join(result.sections_compared)
    filename = f"{prefix}_{sections_str}.yaml"
    output_path = output_dir / filename

    data: dict[str, Any] = {
        "comparison_type": result.comparison_type,
        "sections_compared": result.sections_compared,
        "exclusive_keywords": result.exclusive_keywords,
        "concept_gaps": result.concept_gaps,
        "emphasis_variance": [
            {
                "concept": e.concept,
                "variance": e.variance,
                "per_section_scores": e.per_section_scores,
            }
            for e in result.emphasis_variance
        ],
        "comparison_timestamp": result.comparison_timestamp,
    }

    atomic_write_yaml(data, output_path)

    return output_path


def load_comparison_result(path: Path) -> ComparisonResult:
    """Deserialize ComparisonResult from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        ComparisonResult reconstructed from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Comparison result file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    emphasis_variance = [
        EmphasisVarianceEntry(
            concept=e["concept"],
            variance=e["variance"],
            per_section_scores=e["per_section_scores"],
        )
        for e in data.get("emphasis_variance", [])
    ]

    return ComparisonResult(
        comparison_type=data["comparison_type"],
        sections_compared=data["sections_compared"],
        exclusive_keywords=data["exclusive_keywords"],
        concept_gaps=data.get("concept_gaps"),
        emphasis_variance=emphasis_variance,
        comparison_timestamp=data.get("comparison_timestamp", ""),
    )
