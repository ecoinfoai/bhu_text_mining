"""Concept dependency DAG: parse, validate, and query prerequisite relationships.

Parses concept_dependencies from exam YAML, builds a validated directed
acyclic graph (DAG) using NetworkX, and provides query APIs for
predecessors/successors.

Dataclasses:
    ConceptDependency: Single prerequisite -> dependent relationship.

Classes:
    ConceptDependencyDAG: Validated DAG wrapping nx.DiGraph.

Functions:
    parse_concept_dependencies: Parse exam YAML dict to dependency list.
    build_and_validate_dag: Build and validate DAG from dependency list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConceptDependency:
    """A single prerequisite -> dependent relationship.

    Attributes:
        prerequisite: Prerequisite concept name.
        dependent: Dependent concept name.
    """

    prerequisite: str
    dependent: str


class ConceptDependencyDAG:
    """Validated directed acyclic graph of concept dependencies.

    Attributes:
        graph: Underlying NetworkX DiGraph.
        nodes: All concept names in the DAG.
        edges: All dependency edges as ConceptDependency list.
    """

    def __init__(self, graph: nx.DiGraph, edges: list[ConceptDependency]) -> None:
        """Initialize with a validated DiGraph and edge list.

        Args:
            graph: NetworkX directed graph (must be acyclic).
            edges: List of ConceptDependency edges.
        """
        self._graph = graph
        self._edges = edges

    @property
    def graph(self) -> nx.DiGraph:
        """Return the underlying NetworkX DiGraph."""
        return self._graph

    @property
    def nodes(self) -> set[str]:
        """Return all concept names in the DAG."""
        return set(self._graph.nodes)

    @property
    def edges(self) -> list[ConceptDependency]:
        """Return all dependency edges."""
        return list(self._edges)

    def predecessors(self, node: str) -> list[str]:
        """Return direct prerequisites of a concept.

        Args:
            node: Concept name.

        Returns:
            List of prerequisite concept names.
        """
        return list(self._graph.predecessors(node))

    def successors(self, node: str) -> list[str]:
        """Return direct dependents of a concept.

        Args:
            node: Concept name.

        Returns:
            List of dependent concept names.
        """
        return list(self._graph.successors(node))


def parse_concept_dependencies(
    exam_yaml_dict: dict,
) -> list[ConceptDependency] | None:
    """Parse concept_dependencies from an exam YAML dictionary.

    Malformed entries (missing prerequisite/dependent key) are skipped
    with a warning log. Non-iterable values raise TypeError.

    Args:
        exam_yaml_dict: Parsed exam YAML as a dict.

    Returns:
        List of ConceptDependency if concept_dependencies key is present,
        None otherwise.
    """
    raw = exam_yaml_dict.get("concept_dependencies")
    if raw is None:
        return None
    result: list[ConceptDependency] = []
    for i, entry in enumerate(raw):
        try:
            result.append(
                ConceptDependency(
                    prerequisite=entry["prerequisite"],
                    dependent=entry["dependent"],
                )
            )
        except (KeyError, TypeError) as exc:
            logger.warning(
                "Skipping malformed concept_dependency entry at index %d: %s",
                i,
                exc,
            )
    return result


def build_and_validate_dag(
    dependencies: list[ConceptDependency],
    knowledge_graph_concepts: set[str] | None = None,
) -> ConceptDependencyDAG:
    """Build and validate a concept dependency DAG.

    Args:
        dependencies: List of ConceptDependency edges.
        knowledge_graph_concepts: Optional set of concept names from the
            knowledge graph. If provided, concepts in the DAG that are
            not in this set will trigger a warning.

    Returns:
        Validated ConceptDependencyDAG.

    Raises:
        ValueError: If the dependency graph contains a cycle.
    """
    g = nx.DiGraph()

    # Deduplicate edges using a set of (prerequisite, dependent) tuples
    seen: set[tuple[str, str]] = set()
    unique_deps: list[ConceptDependency] = []
    for dep in dependencies:
        key = (dep.prerequisite, dep.dependent)
        if key not in seen:
            seen.add(key)
            unique_deps.append(dep)
            g.add_edge(dep.prerequisite, dep.dependent)

    # Cycle detection
    if not nx.is_directed_acyclic_graph(g):
        try:
            cycle = nx.find_cycle(g, orientation="original")
            cycle_path = " -> ".join(u for u, _v, _d in cycle)
        except nx.NetworkXNoCycle:
            cycle_path = "unknown"
        raise ValueError(f"Concept dependency graph contains a cycle: {cycle_path}")

    # Knowledge graph concept warning
    if knowledge_graph_concepts is not None:
        dag_concepts = set(g.nodes)
        missing = dag_concepts - knowledge_graph_concepts
        for concept in sorted(missing):
            logger.warning(
                "Concept '%s' in dependency DAG is not found in knowledge_graph",
                concept,
            )

    return ConceptDependencyDAG(graph=g, edges=unique_deps)
