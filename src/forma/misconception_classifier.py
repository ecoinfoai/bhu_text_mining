"""Misconception pattern classifier for student knowledge graph errors.

Classifies structural errors found in graph comparison results into
four pedagogically meaningful misconception patterns: INCLUSION_ERROR,
CAUSAL_REVERSAL, RELATION_CONFUSION, and CONCEPT_ABSENCE.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from forma.evaluation_types import GraphComparisonResult, HubGapEntry, TripletEdge

logger = logging.getLogger(__name__)

DEFAULT_INCLUSION_KEYWORDS: list[str] = [
    "포함", "속함", "구성", "is-a", "part-of", "분류", "하위",
]


class MisconceptionPattern(Enum):
    """Four pedagogically meaningful misconception pattern types.

    Attributes:
        INCLUSION_ERROR: Reversed inclusion/hierarchy relationship.
        CAUSAL_REVERSAL: Reversed causal direction.
        RELATION_CONFUSION: Wrong relation label between correct concepts.
        CONCEPT_ABSENCE: Key concept entirely missing from student response.
    """

    INCLUSION_ERROR = "INCLUSION_ERROR"
    CAUSAL_REVERSAL = "CAUSAL_REVERSAL"
    RELATION_CONFUSION = "RELATION_CONFUSION"
    CONCEPT_ABSENCE = "CONCEPT_ABSENCE"


@dataclass
class ClassifiedMisconception:
    """A single classified misconception instance.

    Args:
        pattern: The misconception pattern type.
        master_edge: The master graph edge (None for CONCEPT_ABSENCE).
        student_edge: The student graph edge (None for CONCEPT_ABSENCE).
        concept: The concept name (used for CONCEPT_ABSENCE).
        confidence: Confidence score in [0.0, 1.0].
        description: Human-readable description of the misconception.
    """

    pattern: MisconceptionPattern
    master_edge: TripletEdge | None
    student_edge: TripletEdge | None
    concept: str | None
    confidence: float
    description: str


def _has_inclusion_keyword(
    relation: str,
    keywords: list[str],
) -> bool:
    """Check if relation string contains any inclusion keyword.

    Args:
        relation: The relation label from a TripletEdge.
        keywords: List of inclusion keywords to check against.

    Returns:
        True if any keyword is found in the relation string.
    """
    relation_lower = relation.lower()
    return any(kw.lower() in relation_lower for kw in keywords)


def classify_misconceptions(
    graph_result: GraphComparisonResult,
    hub_gap_entries: list[HubGapEntry],
    inclusion_keywords: list[str] | None = None,
) -> list[ClassifiedMisconception]:
    """Classify misconceptions from graph comparison and hub gap data.

    Processes wrong_direction_edges (→ INCLUSION_ERROR or CAUSAL_REVERSAL),
    extra_edges vs missing_edges (→ RELATION_CONFUSION), and hub gaps with
    student_present=False (→ CONCEPT_ABSENCE).

    INCLUSION_ERROR takes priority over CAUSAL_REVERSAL when inclusion
    keywords are detected in the relation label.

    Args:
        graph_result: Graph comparison result with edge-level details.
        hub_gap_entries: Hub gap entries for concept absence detection.
        inclusion_keywords: Custom inclusion keywords; defaults to
            DEFAULT_INCLUSION_KEYWORDS if None.

    Returns:
        List of ClassifiedMisconception instances, one per detected error.
    """
    if inclusion_keywords is None:
        inclusion_keywords = DEFAULT_INCLUSION_KEYWORDS

    results: list[ClassifiedMisconception] = []

    # --- INCLUSION_ERROR / CAUSAL_REVERSAL from wrong_direction_edges ---
    for edge in graph_result.wrong_direction_edges:
        if _has_inclusion_keyword(edge.relation, inclusion_keywords):
            results.append(ClassifiedMisconception(
                pattern=MisconceptionPattern.INCLUSION_ERROR,
                master_edge=TripletEdge(edge.object, edge.relation, edge.subject),
                student_edge=edge,
                concept=None,
                confidence=0.9,
                description=(
                    f"포함 관계 역전: {edge.subject}→{edge.relation}→{edge.object}"
                ),
            ))
        else:
            results.append(ClassifiedMisconception(
                pattern=MisconceptionPattern.CAUSAL_REVERSAL,
                master_edge=TripletEdge(edge.object, edge.relation, edge.subject),
                student_edge=edge,
                concept=None,
                confidence=0.85,
                description=(
                    f"인과 방향 역전: {edge.subject}→{edge.relation}→{edge.object}"
                ),
            ))

    # --- RELATION_CONFUSION from extra_edges matching missing_edges on (S, O) ---
    missing_by_so: dict[tuple[str, str], TripletEdge] = {}
    for edge in graph_result.missing_edges:
        missing_by_so[(edge.subject, edge.object)] = edge

    for extra in graph_result.extra_edges:
        key = (extra.subject, extra.object)
        if key in missing_by_so:
            master_edge = missing_by_so[key]
            if master_edge.relation != extra.relation:
                results.append(ClassifiedMisconception(
                    pattern=MisconceptionPattern.RELATION_CONFUSION,
                    master_edge=master_edge,
                    student_edge=extra,
                    concept=None,
                    confidence=0.7,
                    description=(
                        f"관계 혼동: {extra.subject}→{extra.relation}→{extra.object} "
                        f"(정답: {master_edge.relation})"
                    ),
                ))

    # --- CONCEPT_ABSENCE from hub_gap_entries ---
    for entry in hub_gap_entries:
        if not entry.student_present:
            results.append(ClassifiedMisconception(
                pattern=MisconceptionPattern.CONCEPT_ABSENCE,
                master_edge=None,
                student_edge=None,
                concept=entry.concept,
                confidence=entry.degree_centrality,
                description=f"핵심 개념 부재: {entry.concept}",
            ))

    return results


def aggregate_class_misconceptions(
    student_misconceptions: dict[str, list[ClassifiedMisconception]],
) -> list[tuple[MisconceptionPattern, str, int]]:
    """Aggregate misconceptions across all students in a class.

    Groups by (pattern, description) and counts occurrences, returning
    results sorted by count descending.

    Args:
        student_misconceptions: Mapping of student_id to their classified
            misconceptions.

    Returns:
        List of (pattern, description, count) tuples sorted by count desc.
    """
    counter: Counter[tuple[MisconceptionPattern, str]] = Counter()
    for misconceptions in student_misconceptions.values():
        for m in misconceptions:
            counter[(m.pattern, m.description)] += 1

    return sorted(
        [(pattern, desc, count) for (pattern, desc), count in counter.items()],
        key=lambda x: x[2],
        reverse=True,
    )
