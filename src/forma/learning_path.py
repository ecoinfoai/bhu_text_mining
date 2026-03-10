"""Learning path generation from concept dependency DAG and student scores.

Generates per-student ordered learning paths by identifying deficit concepts
(below threshold) and their unmastered prerequisites, then producing a
topologically sorted study order. Also provides class-wide deficit map.

Dataclasses:
    LearningPath: Per-student ordered concept study path.
    ClassDeficitMap: Class-wide deficit counts per concept.

Functions:
    generate_learning_path: Generate learning path for a single student.
    build_class_deficit_map: Build class-wide deficit map.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

from forma.concept_dependency import ConceptDependencyDAG

logger = logging.getLogger(__name__)

_MAX_PATH_LENGTH = 20


@dataclass
class LearningPath:
    """Per-student ordered concept study path.

    Attributes:
        student_id: Student identifier.
        deficit_concepts: Concepts where score < threshold.
        ordered_path: Topologically sorted study order (prerequisites first).
        capped: True if original path exceeded _MAX_PATH_LENGTH.
    """

    student_id: str
    deficit_concepts: list[str] = field(default_factory=list)
    ordered_path: list[str] = field(default_factory=list)
    capped: bool = False


@dataclass
class ClassDeficitMap:
    """Class-wide deficit counts per concept for DAG visualization.

    Attributes:
        concept_counts: {concept_name: number_of_students_with_deficit}.
        total_students: Total students in class.
        dag: Reference to the concept dependency DAG.
    """

    concept_counts: dict[str, int] = field(default_factory=dict)
    total_students: int = 0
    dag: ConceptDependencyDAG | None = None


def generate_learning_path(
    student_id: str,
    student_scores: dict[str, float],
    dag: ConceptDependencyDAG,
    threshold: float = 0.4,
) -> LearningPath:
    """Generate a learning path for a single student.

    Identifies deficit concepts (score < threshold), includes unmastered
    prerequisites from the DAG, and returns a topologically sorted path.

    Args:
        student_id: Student identifier.
        student_scores: {concept_name: score} for the student.
        dag: Validated concept dependency DAG.
        threshold: Score threshold below which a concept is deficit.

    Returns:
        LearningPath with ordered study path.
    """
    if not dag.nodes:
        return LearningPath(student_id=student_id)

    # Identify deficit concepts in the DAG
    deficit = set()
    for concept in dag.nodes:
        score = student_scores.get(concept)
        if score is None or score < threshold:
            deficit.add(concept)

    if not deficit:
        return LearningPath(student_id=student_id)

    # Expand to include unmastered prerequisites (transitive closure)
    to_include = set(deficit)
    queue = list(deficit)
    while queue:
        concept = queue.pop()
        for prereq in dag.predecessors(concept):
            if prereq not in to_include:
                prereq_score = student_scores.get(prereq)
                if prereq_score is None or prereq_score < threshold:
                    to_include.add(prereq)
                    queue.append(prereq)

    # Topological sort within the subgraph of included concepts
    subgraph = dag.graph.subgraph(to_include)
    try:
        topo_order = list(nx.topological_sort(subgraph))
    except nx.NetworkXUnfeasible:
        # Should not happen with validated DAG, but be safe
        topo_order = sorted(to_include)

    # Cap at _MAX_PATH_LENGTH
    capped = len(topo_order) > _MAX_PATH_LENGTH
    if capped:
        topo_order = topo_order[:_MAX_PATH_LENGTH]

    return LearningPath(
        student_id=student_id,
        deficit_concepts=sorted(deficit),
        ordered_path=topo_order,
        capped=capped,
    )


def build_class_deficit_map(
    all_students_scores: dict[str, dict[str, float]],
    dag: ConceptDependencyDAG,
    threshold: float = 0.4,
) -> ClassDeficitMap:
    """Build class-wide deficit map from all students' scores.

    Args:
        all_students_scores: {student_id: {concept: score}}.
        dag: Validated concept dependency DAG.
        threshold: Score threshold below which a concept is deficit.

    Returns:
        ClassDeficitMap with per-concept deficit counts.
    """
    concept_counts: dict[str, int] = {concept: 0 for concept in dag.nodes}

    for _student_id, scores in all_students_scores.items():
        for concept in dag.nodes:
            score = scores.get(concept)
            if score is None or score < threshold:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1

    return ClassDeficitMap(
        concept_counts=concept_counts,
        total_students=len(all_students_scores),
        dag=dag,
    )
