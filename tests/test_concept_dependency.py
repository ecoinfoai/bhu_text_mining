"""Tests for concept_dependency module — ConceptDependency, ConceptDependencyDAG.

Phase 2 (T005): Covers FR-015, FR-016, FR-017 and edge cases #4, #10.
Phase 5 (T032, T033): Extended with malformed YAML, FR-023 silent omission,
    multi-length cycle, orphan concept, topological order verification.
"""

from __future__ import annotations

import logging

import pytest


# ---------------------------------------------------------------------------
# ConceptDependency dataclass
# ---------------------------------------------------------------------------


class TestConceptDependency:
    """Tests for the ConceptDependency dataclass."""

    def test_create_concept_dependency(self):
        from forma.concept_dependency import ConceptDependency

        dep = ConceptDependency(prerequisite="세포막 구조", dependent="물질 이동")
        assert dep.prerequisite == "세포막 구조"
        assert dep.dependent == "물질 이동"

    def test_concept_dependency_equality(self):
        from forma.concept_dependency import ConceptDependency

        a = ConceptDependency(prerequisite="A", dependent="B")
        b = ConceptDependency(prerequisite="A", dependent="B")
        assert a == b

    def test_concept_dependency_different(self):
        from forma.concept_dependency import ConceptDependency

        a = ConceptDependency(prerequisite="A", dependent="B")
        b = ConceptDependency(prerequisite="B", dependent="A")
        assert a != b


# ---------------------------------------------------------------------------
# parse_concept_dependencies
# ---------------------------------------------------------------------------


class TestParseConceptDependencies:
    """Tests for parse_concept_dependencies(exam_yaml_dict)."""

    def test_parse_returns_list_when_present(self, sample_exam_yaml_with_deps):
        from forma.concept_dependency import ConceptDependency, parse_concept_dependencies

        result = parse_concept_dependencies(sample_exam_yaml_with_deps)
        assert result is not None
        assert len(result) == 4
        assert all(isinstance(d, ConceptDependency) for d in result)

    def test_parse_returns_none_when_absent(self):
        from forma.concept_dependency import parse_concept_dependencies

        exam = {"exam_name": "NoDepTest", "questions": []}
        result = parse_concept_dependencies(exam)
        assert result is None

    def test_parse_returns_none_for_empty_dict(self):
        from forma.concept_dependency import parse_concept_dependencies

        result = parse_concept_dependencies({})
        assert result is None

    def test_parse_preserves_order(self, sample_concept_dependencies):
        from forma.concept_dependency import parse_concept_dependencies

        exam = {"concept_dependencies": sample_concept_dependencies}
        result = parse_concept_dependencies(exam)
        assert result is not None
        assert result[0].prerequisite == "세포막 구조"
        assert result[0].dependent == "물질 이동"
        assert result[-1].prerequisite == "확산"
        assert result[-1].dependent == "삼투압"

    def test_parse_empty_list_returns_empty(self):
        from forma.concept_dependency import parse_concept_dependencies

        exam = {"concept_dependencies": []}
        result = parse_concept_dependencies(exam)
        assert result is not None
        assert result == []

    def test_parse_with_korean_concepts(self):
        from forma.concept_dependency import parse_concept_dependencies

        exam = {
            "concept_dependencies": [
                {"prerequisite": "항상성", "dependent": "음성되먹임"},
            ],
        }
        result = parse_concept_dependencies(exam)
        assert result is not None
        assert result[0].prerequisite == "항상성"
        assert result[0].dependent == "음성되먹임"


# ---------------------------------------------------------------------------
# build_and_validate_dag — valid DAGs
# ---------------------------------------------------------------------------


class TestBuildAndValidateDag:
    """Tests for build_and_validate_dag(dependencies, knowledge_graph_concepts)."""

    def test_build_valid_dag(self, sample_concept_dependencies):
        from forma.concept_dependency import (
            ConceptDependency,
            build_and_validate_dag,
        )

        deps = [
            ConceptDependency(prerequisite=d["prerequisite"], dependent=d["dependent"])
            for d in sample_concept_dependencies
        ]
        dag = build_and_validate_dag(deps)
        assert dag is not None
        assert len(dag.nodes) == 5
        assert len(dag.edges) == 4

    def test_dag_graph_property_is_digraph(self, sample_concept_dependencies):
        import networkx as nx

        from forma.concept_dependency import (
            ConceptDependency,
            build_and_validate_dag,
        )

        deps = [
            ConceptDependency(prerequisite=d["prerequisite"], dependent=d["dependent"])
            for d in sample_concept_dependencies
        ]
        dag = build_and_validate_dag(deps)
        assert isinstance(dag.graph, nx.DiGraph)

    def test_dag_predecessors(self):
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite="A", dependent="B"),
            ConceptDependency(prerequisite="A", dependent="C"),
        ]
        dag = build_and_validate_dag(deps)
        assert set(dag.predecessors("B")) == {"A"}
        assert set(dag.predecessors("C")) == {"A"}
        assert set(dag.predecessors("A")) == set()

    def test_dag_successors(self):
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite="A", dependent="B"),
            ConceptDependency(prerequisite="A", dependent="C"),
        ]
        dag = build_and_validate_dag(deps)
        assert set(dag.successors("A")) == {"B", "C"}
        assert set(dag.successors("B")) == set()

    def test_empty_dependencies_returns_empty_dag(self):
        from forma.concept_dependency import build_and_validate_dag

        dag = build_and_validate_dag([])
        assert len(dag.nodes) == 0
        assert len(dag.edges) == 0


# ---------------------------------------------------------------------------
# build_and_validate_dag — cycle detection (FR-016)
# ---------------------------------------------------------------------------


class TestCycleDetection:
    """Tests for cycle detection in build_and_validate_dag."""

    def test_cycle_raises_valueerror(self, cyclic_concept_dependencies):
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite=d["prerequisite"], dependent=d["dependent"])
            for d in cyclic_concept_dependencies
        ]
        with pytest.raises(ValueError, match="cycle"):
            build_and_validate_dag(deps)

    def test_cycle_error_includes_path(self, cyclic_concept_dependencies):
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite=d["prerequisite"], dependent=d["dependent"])
            for d in cyclic_concept_dependencies
        ]
        with pytest.raises(ValueError) as exc_info:
            build_and_validate_dag(deps)
        msg = str(exc_info.value)
        # cycle path should mention at least two of A, B, C
        cycle_nodes = {"A", "B", "C"}
        mentioned = sum(1 for n in cycle_nodes if n in msg)
        assert mentioned >= 2

    def test_self_loop_detected_as_cycle(self):
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [ConceptDependency(prerequisite="A", dependent="A")]
        with pytest.raises(ValueError, match="cycle"):
            build_and_validate_dag(deps)


# ---------------------------------------------------------------------------
# build_and_validate_dag — knowledge_graph concept warning (FR-017)
# ---------------------------------------------------------------------------


class TestKnowledgeGraphWarning:
    """Tests for missing concept warnings when knowledge_graph_concepts provided."""

    def test_no_warning_when_all_concepts_in_kg(
        self, sample_concept_dependencies, sample_knowledge_graph_concepts, caplog
    ):
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite=d["prerequisite"], dependent=d["dependent"])
            for d in sample_concept_dependencies
        ]
        with caplog.at_level(logging.WARNING, logger="forma.concept_dependency"):
            build_and_validate_dag(deps, knowledge_graph_concepts=sample_knowledge_graph_concepts)
        assert len(caplog.records) == 0

    def test_warning_for_missing_concept(self, caplog):
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [ConceptDependency(prerequisite="알려진개념", dependent="미지의개념")]
        kg_concepts = {"알려진개념"}
        with caplog.at_level(logging.WARNING, logger="forma.concept_dependency"):
            dag = build_and_validate_dag(deps, knowledge_graph_concepts=kg_concepts)
        # Should still build the DAG successfully
        assert dag is not None
        assert len(dag.nodes) == 2
        # But should have logged a warning about "미지의개념"
        assert any("미지의개념" in r.message for r in caplog.records)

    def test_no_warning_when_kg_concepts_none(self, caplog):
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [ConceptDependency(prerequisite="A", dependent="B")]
        with caplog.at_level(logging.WARNING, logger="forma.concept_dependency"):
            build_and_validate_dag(deps, knowledge_graph_concepts=None)
        assert len(caplog.records) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests: isolated nodes, duplicates, large graphs."""

    def test_isolated_nodes_handled(self):
        """Isolated nodes (no edges) should not appear in DAG."""
        from forma.concept_dependency import build_and_validate_dag

        dag = build_and_validate_dag([])
        # Empty graph has no isolated nodes
        assert len(dag.nodes) == 0

    def test_duplicate_edges_deduplicated(self):
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite="A", dependent="B"),
            ConceptDependency(prerequisite="A", dependent="B"),  # duplicate
            ConceptDependency(prerequisite="B", dependent="C"),
        ]
        dag = build_and_validate_dag(deps)
        assert len(dag.nodes) == 3
        assert len(dag.edges) == 2  # deduplicated

    def test_large_dag_100_nodes(self):
        """100+ nodes should complete without error (Edge Case #10)."""
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        # Linear chain: C0 → C1 → C2 → ... → C119
        deps = [
            ConceptDependency(prerequisite=f"C{i}", dependent=f"C{i+1}")
            for i in range(120)
        ]
        dag = build_and_validate_dag(deps)
        assert len(dag.nodes) == 121
        assert len(dag.edges) == 120

    def test_diamond_dag_valid(self):
        """Diamond shape (A→B, A→C, B→D, C→D) is a valid DAG."""
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite="A", dependent="B"),
            ConceptDependency(prerequisite="A", dependent="C"),
            ConceptDependency(prerequisite="B", dependent="D"),
            ConceptDependency(prerequisite="C", dependent="D"),
        ]
        dag = build_and_validate_dag(deps)
        assert len(dag.nodes) == 4
        assert len(dag.edges) == 4

    def test_single_edge_dag(self):
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [ConceptDependency(prerequisite="X", dependent="Y")]
        dag = build_and_validate_dag(deps)
        assert len(dag.nodes) == 2
        assert len(dag.edges) == 1
        assert set(dag.predecessors("Y")) == {"X"}
        assert set(dag.successors("X")) == {"Y"}


# ---------------------------------------------------------------------------
# US3 Phase 5: T032 — Malformed YAML parsing (FR-015, FR-023)
# ---------------------------------------------------------------------------


class TestParseConceptDependenciesMalformed:
    """Tests for malformed concept_dependencies in exam YAML — T032."""

    def test_malformed_entry_missing_prerequisite_skipped(self, caplog):
        """Entry missing 'prerequisite' key is skipped with warning."""
        from forma.concept_dependency import parse_concept_dependencies

        exam = {
            "concept_dependencies": [
                {"dependent": "B"},  # missing prerequisite
                {"prerequisite": "X", "dependent": "Y"},  # valid
            ],
        }
        with caplog.at_level(logging.WARNING, logger="forma.concept_dependency"):
            result = parse_concept_dependencies(exam)
        assert result is not None
        assert len(result) == 1
        assert result[0].prerequisite == "X"
        assert any("index 0" in r.message for r in caplog.records)

    def test_malformed_entry_missing_dependent_skipped(self, caplog):
        """Entry missing 'dependent' key is skipped with warning."""
        from forma.concept_dependency import parse_concept_dependencies

        exam = {
            "concept_dependencies": [
                {"prerequisite": "A"},  # missing dependent
            ],
        }
        with caplog.at_level(logging.WARNING, logger="forma.concept_dependency"):
            result = parse_concept_dependencies(exam)
        assert result is not None
        assert len(result) == 0
        assert len(caplog.records) >= 1

    def test_concept_dependencies_not_a_list_skipped(self, caplog):
        """concept_dependencies value is a string — entries skipped with warnings."""
        from forma.concept_dependency import parse_concept_dependencies

        exam = {"concept_dependencies": "not_a_list"}
        with caplog.at_level(logging.WARNING, logger="forma.concept_dependency"):
            result = parse_concept_dependencies(exam)
        assert result is not None
        assert len(result) == 0
        assert len(caplog.records) > 0

    def test_concept_dependencies_none_value(self):
        """concept_dependencies explicitly set to None → returns None."""
        from forma.concept_dependency import parse_concept_dependencies

        exam = {"concept_dependencies": None}
        result = parse_concept_dependencies(exam)
        assert result is None

    def test_fr023_silent_omission_when_no_deps(self):
        """FR-023: When no concept_dependencies, parse returns None (silent omit)."""
        from forma.concept_dependency import parse_concept_dependencies

        exam = {
            "exam_name": "Test",
            "questions": [{"question_sn": 1}],
        }
        result = parse_concept_dependencies(exam)
        assert result is None


# ---------------------------------------------------------------------------
# US3 Phase 5: T033 — Additional edge cases
# ---------------------------------------------------------------------------


class TestAdditionalEdgeCases:
    """Additional edge case tests for US3 — T033."""

    def test_two_node_cycle(self):
        """Two-node cycle: A -> B -> A."""
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite="A", dependent="B"),
            ConceptDependency(prerequisite="B", dependent="A"),
        ]
        with pytest.raises(ValueError, match="cycle"):
            build_and_validate_dag(deps)

    def test_long_cycle_detected(self):
        """5-node cycle: A -> B -> C -> D -> E -> A."""
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite="A", dependent="B"),
            ConceptDependency(prerequisite="B", dependent="C"),
            ConceptDependency(prerequisite="C", dependent="D"),
            ConceptDependency(prerequisite="D", dependent="E"),
            ConceptDependency(prerequisite="E", dependent="A"),
        ]
        with pytest.raises(ValueError, match="cycle"):
            build_and_validate_dag(deps)

    def test_mixed_valid_and_orphan_warning(self, caplog):
        """DAG with some concepts in KG, some not — warns only for missing."""
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite="세포", dependent="조직"),
            ConceptDependency(prerequisite="조직", dependent="기관"),
        ]
        kg = {"세포", "조직"}  # "기관" is missing
        with caplog.at_level(logging.WARNING, logger="forma.concept_dependency"):
            dag = build_and_validate_dag(deps, knowledge_graph_concepts=kg)
        assert len(dag.nodes) == 3
        warning_msgs = [r.message for r in caplog.records]
        assert any("기관" in m for m in warning_msgs)
        assert not any("세포" in m for m in warning_msgs)
        assert not any("조직" in m for m in warning_msgs)

    def test_multiple_missing_concepts_all_warned(self, caplog):
        """Multiple missing concepts each get their own warning."""
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite="X", dependent="Y"),
            ConceptDependency(prerequisite="Y", dependent="Z"),
        ]
        kg: set[str] = set()  # none match
        with caplog.at_level(logging.WARNING, logger="forma.concept_dependency"):
            build_and_validate_dag(deps, knowledge_graph_concepts=kg)
        warned_concepts = {r.message for r in caplog.records}
        assert len(caplog.records) == 3  # X, Y, Z

    def test_topological_order_valid(self):
        """Verify DAG allows topological sort (nx integration)."""
        import networkx as nx

        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite="A", dependent="B"),
            ConceptDependency(prerequisite="B", dependent="C"),
            ConceptDependency(prerequisite="A", dependent="C"),
        ]
        dag = build_and_validate_dag(deps)
        topo = list(nx.topological_sort(dag.graph))
        assert topo.index("A") < topo.index("B")
        assert topo.index("B") < topo.index("C")

    def test_wide_dag_many_roots(self):
        """DAG with many roots converging to single sink."""
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite=f"R{i}", dependent="SINK")
            for i in range(50)
        ]
        dag = build_and_validate_dag(deps)
        assert len(dag.nodes) == 51
        assert len(dag.edges) == 50
        assert set(dag.predecessors("SINK")) == {f"R{i}" for i in range(50)}

    def test_parse_then_build_integration(self, sample_exam_yaml_with_deps):
        """Integration: parse exam YAML then build DAG."""
        from forma.concept_dependency import (
            build_and_validate_dag,
            parse_concept_dependencies,
        )

        deps = parse_concept_dependencies(sample_exam_yaml_with_deps)
        assert deps is not None
        dag = build_and_validate_dag(deps)
        assert len(dag.nodes) == 5
        assert len(dag.edges) == 4

    def test_disconnected_components_valid(self):
        """Two separate sub-DAGs with no cross-edges are a valid DAG."""
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite="A", dependent="B"),
            ConceptDependency(prerequisite="X", dependent="Y"),
        ]
        dag = build_and_validate_dag(deps)
        assert len(dag.nodes) == 4
        assert len(dag.edges) == 2
        assert set(dag.successors("A")) == {"B"}
        assert set(dag.successors("X")) == {"Y"}
        # No cross-edges
        assert set(dag.predecessors("B")) == {"A"}
        assert set(dag.predecessors("Y")) == {"X"}

    def test_fan_in_all_edges_to_single_node(self):
        """All edges point to the same dependent node (fan-in) — valid DAG."""
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [
            ConceptDependency(prerequisite="P1", dependent="CORE"),
            ConceptDependency(prerequisite="P2", dependent="CORE"),
            ConceptDependency(prerequisite="P3", dependent="CORE"),
        ]
        dag = build_and_validate_dag(deps)
        assert len(dag.nodes) == 4
        assert set(dag.predecessors("CORE")) == {"P1", "P2", "P3"}
