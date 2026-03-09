"""Class-level knowledge aggregate computation.

Aggregates individual student GraphComparisonResult data into a class-level
knowledge graph summary, counting correct, error, and missing edges per
master edge across all students.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

__all__ = [
    "AggregateEdge",
    "ClassKnowledgeAggregate",
    "build_class_knowledge_aggregate",
]

logger = logging.getLogger(__name__)


@dataclass
class AggregateEdge:
    """A single edge in the class-level knowledge aggregate graph.

    Represents per-master-edge student response statistics across the entire
    class: how many students answered correctly, with errors, or missed this
    edge entirely.

    Args:
        subject: Subject node of the master edge.
        relation: Relation label of the master edge.
        obj: Object node of the master edge.
        correct_count: Students with this edge in matched_edges.
        error_count: Students with this edge in wrong_direction_edges.
        missing_count: Students missing this edge entirely.
        total_students: Total students evaluated (invariant:
            correct_count + error_count + missing_count == total_students).
        correct_ratio: correct_count / total_students, or 0.0 if
            total_students == 0.
    """

    subject: str
    relation: str
    obj: str
    correct_count: int
    error_count: int
    missing_count: int
    total_students: int
    correct_ratio: float


@dataclass
class ClassKnowledgeAggregate:
    """Per-question class-level knowledge aggregate.

    Contains all master edges with their aggregated student response
    statistics for a single exam question.

    Args:
        question_sn: Question serial number (1-based).
        edges: List of AggregateEdge instances, one per master edge.
        total_students: Total number of students evaluated.
    """

    question_sn: int
    edges: list[AggregateEdge] = field(default_factory=list)
    total_students: int = 0


def build_class_knowledge_aggregate(
    master_edges: list,
    comparison_results: list,
    question_sn: int,
) -> ClassKnowledgeAggregate:
    """Build a class-level knowledge aggregate from student comparison results.

    For each master edge, counts how many students answered correctly
    (edge in matched_edges), with error (edge in wrong_direction_edges),
    or missed the edge entirely.

    Args:
        master_edges: Master graph edges (TripletEdge instances) for this
            question.
        comparison_results: Per-student GraphComparisonResult instances.
        question_sn: Question serial number (1-based).

    Returns:
        ClassKnowledgeAggregate with one AggregateEdge per master edge.
    """
    total_students = len(comparison_results)
    aggregate_edges: list[AggregateEdge] = []

    for master_edge in master_edges:
        key = (master_edge.subject, master_edge.relation, master_edge.object)
        correct_count = 0
        error_count = 0

        for result in comparison_results:
            matched = any(
                (e.subject, e.relation, e.object) == key
                for e in result.matched_edges
            )
            if matched:
                correct_count += 1
                continue

            # wrong_direction_edges stores the student's reversed edge (B, rel, A)
            # when master is (A, rel, B) — match by reversed subject/object
            wrong = any(
                e.subject == master_edge.object and e.object == master_edge.subject
                for e in result.wrong_direction_edges
            )
            if wrong:
                error_count += 1

        missing_count = total_students - correct_count - error_count
        correct_ratio = correct_count / total_students if total_students > 0 else 0.0

        aggregate_edges.append(AggregateEdge(
            subject=master_edge.subject,
            relation=master_edge.relation,
            obj=master_edge.object,
            correct_count=correct_count,
            error_count=error_count,
            missing_count=missing_count,
            total_students=total_students,
            correct_ratio=correct_ratio,
        ))

    return ClassKnowledgeAggregate(
        question_sn=question_sn,
        edges=aggregate_edges,
        total_students=total_students,
    )
