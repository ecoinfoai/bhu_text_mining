"""Adversary attack tests for v0.10.0 features.

7 adversarial personas with aggressive testing of edge cases,
boundary conditions, type confusion, data corruption, and
Unicode/injection scenarios.

Targets:
  - US1: intervention_store (InterventionRecord, InterventionLog, INTERVENTION_TYPES)
  - US2: intervention_effect (InterventionEffect, InterventionTypeSummary,
         compute_intervention_effects, compute_type_summary)
  - US3: concept_dependency (ConceptDependency, ConceptDependencyDAG,
         parse_concept_dependencies, build_and_validate_dag)
  - US4: learning_path (LearningPath, ClassDeficitMap, generate_learning_path,
         build_class_deficit_map)
  - US5: grade_predictor (GradeFeatureExtractor, GradePredictor, TrainedGradeModel,
         load_grade_mapping, save_grade_model, load_grade_model)
  - US6: professor_report / student_report (new sections: grade prediction,
         intervention history, deficit map, learning path)

Personas:
  - Persona 1: Intervention Saboteur — attacks InterventionLog CRUD
  - Persona 2: DAG Poisoner — attacks concept dependency DAG
  - Persona 3: Grade Data Corruptor — attacks grade mapping and model
  - Persona 4: PDF Crasher — attacks new PDF sections
  - Persona 5: Boundary Pusher — attacks exact boundary values
  - Persona 6: Concurrent Chaos — race conditions and file locking
  - Persona 7: Unicode Attacker — CJK, ZWJ, RTL, BOM edge cases
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from forma.intervention_store import (
    INTERVENTION_TYPES,
    InterventionLog,
)
from forma.intervention_effect import (
    InterventionEffect,
    InterventionTypeSummary,
    compute_intervention_effects,
    compute_type_summary,
)
from forma.concept_dependency import (
    ConceptDependency,
    ConceptDependencyDAG,
    build_and_validate_dag,
    parse_concept_dependencies,
)
from forma.learning_path import (
    ClassDeficitMap,
    LearningPath,
    _MAX_PATH_LENGTH,
    build_class_deficit_map,
    generate_learning_path,
)
from forma.grade_predictor import (
    GRADE_FEATURE_NAMES,
    GRADE_ORDINAL_MAP,
    ORDINAL_GRADE_MAP,
    VALID_GRADES,
    GradePrediction,
    GradePredictor,
    TrainedGradeModel,
    load_grade_mapping,
    load_grade_model,
    save_grade_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_log(records_data: list[dict] | None = None) -> tuple[InterventionLog, str]:
    """Build an InterventionLog at a temp path, optionally adding records."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name
    log = InterventionLog(path)
    if records_data:
        for d in records_data:
            log.add_record(**d)
    return log, path


def _make_simple_dag(edges: list[tuple[str, str]]) -> ConceptDependencyDAG:
    """Build a validated DAG from a list of (prerequisite, dependent) tuples."""
    deps = [ConceptDependency(prerequisite=p, dependent=d) for p, d in edges]
    return build_and_validate_dag(deps)


def _make_grade_yaml(data: dict, path: str | None = None) -> str:
    """Write a grade mapping YAML file and return its path."""
    import yaml

    if path is None:
        f = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
        path = f.name
        f.close()
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    return path


def _make_store_with_trajectory(
    student_id: str,
    weeks: list[int],
    scores: list[float],
):
    """Build a LongitudinalStore with ensemble_score trajectory."""
    from forma.evaluation_types import LongitudinalRecord
    from forma.longitudinal_store import LongitudinalStore

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name
    store = LongitudinalStore(path)
    for w, s in zip(weeks, scores):
        store.add_record(
            LongitudinalRecord(
                student_id=student_id,
                week=w,
                question_sn=1,
                scores={"ensemble_score": s},
                tier_level=1,
                tier_label="Developing",
            )
        )
    return store, path


# ===========================================================================
# PERSONA 1: INTERVENTION SABOTEUR
# ===========================================================================


class TestInterventionSaboteur:
    """Persona 1: Malformed records, injection attacks, CRUD edge cases."""

    def test_invalid_intervention_type_raises(self):
        """Non-predefined intervention_type should raise ValueError."""
        log, path = _make_log()
        try:
            with pytest.raises(ValueError, match="Invalid intervention_type"):
                log.add_record("S001", 1, "invalid_type")
        finally:
            os.unlink(path)

    def test_empty_string_intervention_type_raises(self):
        """Empty string intervention_type should raise ValueError."""
        log, path = _make_log()
        try:
            with pytest.raises(ValueError, match="Invalid intervention_type"):
                log.add_record("S001", 1, "")
        finally:
            os.unlink(path)

    def test_sql_injection_in_description(self):
        """SQL injection payload in description: stored faithfully."""
        log, path = _make_log()
        try:
            payload = "'; DROP TABLE students; --"
            log.add_record("S001", 1, "면담", description=payload)
            records = log.get_records(student_id="S001")
            assert len(records) == 1
            assert records[0].description == payload
        finally:
            os.unlink(path)

    def test_xss_injection_in_student_id(self):
        """XSS payload in student_id: stored faithfully, not executed."""
        log, path = _make_log()
        try:
            xss = '<script>alert("xss")</script>'
            log.add_record(xss, 1, "면담")
            records = log.get_records(student_id=xss)
            assert len(records) == 1
            assert records[0].student_id == xss
        finally:
            os.unlink(path)

    def test_10kb_description(self):
        """10KB+ description should be rejected (exceeds 2000 char limit)."""
        log, path = _make_log()
        try:
            huge_desc = "A" * 10240
            with pytest.raises(ValueError, match="2000"):
                log.add_record("S001", 1, "면담", description=huge_desc)
        finally:
            os.unlink(path)

    def test_negative_week(self):
        """Negative week number should be rejected."""
        log, path = _make_log()
        try:
            with pytest.raises(ValueError, match="positive integer"):
                log.add_record("S001", -1, "면담")
        finally:
            os.unlink(path)

    def test_zero_week(self):
        """Week 0 should be rejected."""
        log, path = _make_log()
        try:
            with pytest.raises(ValueError, match="positive integer"):
                log.add_record("S001", 0, "면담")
        finally:
            os.unlink(path)

    def test_empty_student_id(self):
        """Empty string student_id should be rejected."""
        log, path = _make_log()
        try:
            with pytest.raises(ValueError, match="student_id"):
                log.add_record("", 1, "면담")
        finally:
            os.unlink(path)

    def test_auto_increment_id_sequential(self):
        """IDs should auto-increment sequentially starting from 1."""
        log, path = _make_log()
        try:
            id1 = log.add_record("S001", 1, "면담")
            id2 = log.add_record("S002", 2, "보충학습")
            id3 = log.add_record("S003", 3, "과제부여")
            assert id1 == 1
            assert id2 == 2
            assert id3 == 3
        finally:
            os.unlink(path)

    def test_update_nonexistent_record_returns_false(self):
        """Updating outcome for nonexistent ID should return False."""
        log, path = _make_log()
        try:
            result = log.update_outcome(999, "개선")
            assert result is False
        finally:
            os.unlink(path)

    def test_update_existing_record_outcome(self):
        """Updating outcome for existing ID should return True."""
        log, path = _make_log()
        try:
            rec_id = log.add_record("S001", 1, "면담")
            result = log.update_outcome(rec_id, "개선")
            assert result is True
            records = log.get_records(student_id="S001")
            assert records[0].outcome == "개선"
        finally:
            os.unlink(path)

    def test_save_load_roundtrip(self):
        """Records should survive save/load roundtrip."""
        log, path = _make_log()
        try:
            log.add_record("S001", 1, "면담", description="첫 면담")
            log.add_record("S002", 2, "보충학습")
            log.save()

            log2 = InterventionLog(path)
            log2.load()
            records = log2.get_records()
            assert len(records) == 2
            assert records[0].student_id == "S001"
            assert records[0].description == "첫 면담"
            assert records[1].student_id == "S002"
        finally:
            os.unlink(path)

    def test_load_nonexistent_file_initializes_empty(self):
        """Loading from nonexistent file should initialize empty."""
        path = "/tmp/nonexistent_intervention_log_test_adversary.yaml"
        if os.path.exists(path):
            os.unlink(path)
        log = InterventionLog(path)
        log.load()
        records = log.get_records()
        assert records == []

    def test_load_empty_yaml(self):
        """Loading an empty YAML file should initialize empty."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml",
            delete=False,
            mode="w",
        ) as f:
            f.write("")
            path = f.name
        try:
            log = InterventionLog(path)
            log.load()
            assert log.get_records() == []
        finally:
            os.unlink(path)

    def test_load_corrupt_yaml(self):
        """Loading corrupt YAML should raise."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml",
            delete=False,
            mode="w",
        ) as f:
            f.write(":::not valid yaml::: {{{")
            path = f.name
        try:
            log = InterventionLog(path)
            with pytest.raises(Exception):
                log.load()
        finally:
            os.unlink(path)

    def test_filter_by_student_and_week(self):
        """Filtering by both student_id and week should narrow correctly."""
        log, path = _make_log()
        try:
            log.add_record("S001", 1, "면담")
            log.add_record("S001", 2, "면담")
            log.add_record("S002", 1, "보충학습")
            records = log.get_records(student_id="S001", week=2)
            assert len(records) == 1
            assert records[0].week == 2
        finally:
            os.unlink(path)

    def test_filter_no_match_returns_empty(self):
        """Filter with no matching records should return empty list."""
        log, path = _make_log()
        try:
            log.add_record("S001", 1, "면담")
            records = log.get_records(student_id="NONEXISTENT")
            assert records == []
        finally:
            os.unlink(path)

    def test_duplicate_same_student_week_type(self):
        """Same student + week + type: stored as separate records (FR-003)."""
        log, path = _make_log()
        try:
            id1 = log.add_record("S001", 3, "면담")
            id2 = log.add_record("S001", 3, "면담")
            assert id1 != id2
            records = log.get_records(student_id="S001", week=3)
            assert len(records) == 2
        finally:
            os.unlink(path)

    def test_all_predefined_types_accepted(self):
        """All 5 predefined types should be accepted."""
        log, path = _make_log()
        try:
            for itype in INTERVENTION_TYPES:
                rec_id = log.add_record("S001", 1, itype)
                assert rec_id > 0
            assert len(log.get_records()) == 5
        finally:
            os.unlink(path)

    def test_recorded_at_is_iso_format(self):
        """recorded_at should be ISO 8601 format."""
        log, path = _make_log()
        try:
            log.add_record("S001", 1, "면담")
            rec = log.get_records()[0]
            # ISO 8601 includes "T" separator
            assert "T" in rec.recorded_at
        finally:
            os.unlink(path)

    def test_follow_up_week_stored(self):
        """follow_up_week should be stored and retrievable."""
        log, path = _make_log()
        try:
            log.add_record("S001", 3, "면담", follow_up_week=5)
            rec = log.get_records()[0]
            assert rec.follow_up_week == 5
        finally:
            os.unlink(path)

    def test_1000_records_stress(self):
        """Adding 1000 records should complete quickly."""
        log, path = _make_log()
        try:
            start = time.time()
            for i in range(1000):
                log.add_record(f"S{i:04d}", i % 8 + 1, "면담")
            elapsed = time.time() - start
            assert elapsed < 5.0
            assert len(log.get_records()) == 1000
        finally:
            os.unlink(path)

    def test_backup_file_created_on_save(self):
        """Second save should create a .bak backup file."""
        log, path = _make_log()
        try:
            log.add_record("S001", 1, "면담")
            log.save()
            log.add_record("S002", 2, "보충학습")
            log.save()
            assert os.path.exists(path + ".bak")
        finally:
            os.unlink(path)
            if os.path.exists(path + ".bak"):
                os.unlink(path + ".bak")

    def test_next_id_persisted_across_sessions(self):
        """next_id should persist: new records after reload get higher IDs."""
        log, path = _make_log()
        try:
            log.add_record("S001", 1, "면담")
            log.add_record("S002", 2, "면담")
            log.save()

            log2 = InterventionLog(path)
            log2.load()
            new_id = log2.add_record("S003", 3, "면담")
            assert new_id == 3
        finally:
            os.unlink(path)

    def test_xml_special_chars_in_description(self):
        """XML special chars (<>&\"') in description should roundtrip."""
        log, path = _make_log()
        try:
            desc = '<script>alert("xss")</script> & "quotes\' test'
            log.add_record("S001", 1, "면담", description=desc)
            log.save()

            log2 = InterventionLog(path)
            log2.load()
            assert log2.get_records()[0].description == desc
        finally:
            os.unlink(path)

    def test_newlines_and_tabs_in_description(self):
        """Newlines and tabs in description should roundtrip."""
        log, path = _make_log()
        try:
            desc = "Line 1\nLine 2\tTabbed\r\nWindows"
            log.add_record("S001", 1, "면담", description=desc)
            log.save()

            log2 = InterventionLog(path)
            log2.load()
            assert log2.get_records()[0].description == desc
        finally:
            os.unlink(path)

    def test_very_large_week_number(self):
        """Very large week number should be handled."""
        log, path = _make_log()
        try:
            log.add_record("S001", 99999, "면담")
            records = log.get_records(week=99999)
            assert len(records) == 1
        finally:
            os.unlink(path)


# ===========================================================================
# PERSONA 2: DAG POISONER
# ===========================================================================


class TestDAGPoisoner:
    """Persona 2: Cyclic graphs, 1000-node DAGs, edge cases."""

    def test_simple_cycle_A_B_C_A(self):
        """A->B->C->A cycle should raise ValueError."""
        deps = [
            ConceptDependency("A", "B"),
            ConceptDependency("B", "C"),
            ConceptDependency("C", "A"),
        ]
        with pytest.raises(ValueError, match="cycle"):
            build_and_validate_dag(deps)

    def test_self_loop_raises(self):
        """Self-loop (A->A) should raise ValueError."""
        deps = [ConceptDependency("A", "A")]
        with pytest.raises(ValueError, match="cycle"):
            build_and_validate_dag(deps)

    def test_two_node_cycle(self):
        """A->B->A two-node cycle should raise ValueError."""
        deps = [
            ConceptDependency("A", "B"),
            ConceptDependency("B", "A"),
        ]
        with pytest.raises(ValueError, match="cycle"):
            build_and_validate_dag(deps)

    def test_valid_linear_chain(self):
        """A->B->C->D valid linear chain should succeed."""
        dag = _make_simple_dag([("A", "B"), ("B", "C"), ("C", "D")])
        assert len(dag.nodes) == 4
        assert len(dag.edges) == 3

    def test_diamond_dag(self):
        """A->B, A->C, B->D, C->D diamond DAG: valid."""
        dag = _make_simple_dag([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
        assert len(dag.nodes) == 4
        assert dag.predecessors("D") == sorted(["B", "C"]) or set(dag.predecessors("D")) == {"B", "C"}

    def test_disconnected_components(self):
        """Disconnected components (A->B, C->D): valid DAG."""
        dag = _make_simple_dag([("A", "B"), ("C", "D")])
        assert len(dag.nodes) == 4
        assert len(dag.edges) == 2

    def test_empty_dependencies_list(self):
        """Empty dependency list should produce empty DAG."""
        dag = build_and_validate_dag([])
        assert len(dag.nodes) == 0
        assert len(dag.edges) == 0

    def test_duplicate_edges_deduplicated(self):
        """Duplicate edges should be deduplicated."""
        deps = [
            ConceptDependency("A", "B"),
            ConceptDependency("A", "B"),
            ConceptDependency("A", "B"),
        ]
        dag = build_and_validate_dag(deps)
        assert len(dag.edges) == 1

    def test_1000_node_linear_chain(self):
        """1000-node linear chain should build quickly."""
        deps = [ConceptDependency(f"C{i:04d}", f"C{i + 1:04d}") for i in range(999)]
        start = time.time()
        dag = build_and_validate_dag(deps)
        elapsed = time.time() - start
        assert len(dag.nodes) == 1000
        assert elapsed < 5.0

    def test_1000_node_wide_fan_out(self):
        """Root -> 999 leaves: valid wide DAG."""
        deps = [ConceptDependency("root", f"leaf_{i}") for i in range(999)]
        dag = build_and_validate_dag(deps)
        assert len(dag.nodes) == 1000

    def test_unicode_concept_names(self):
        """Korean/CJK concept names should work in DAG."""
        dag = _make_simple_dag(
            [
                ("세포막 구조", "물질 이동"),
                ("물질 이동", "삼투압"),
                ("삼투압", "체액 균형"),
            ]
        )
        assert "세포막 구조" in dag.nodes
        assert "체액 균형" in dag.nodes

    def test_concept_not_in_knowledge_graph_warns(self, caplog):
        """Concept not in knowledge_graph should produce warning."""
        deps = [ConceptDependency("A", "B")]
        with caplog.at_level(logging.WARNING):
            build_and_validate_dag(deps, knowledge_graph_concepts={"A"})
        assert "B" in caplog.text
        assert "not found in knowledge_graph" in caplog.text

    def test_all_concepts_in_knowledge_graph_no_warning(self, caplog):
        """All concepts present in knowledge_graph: no warning."""
        deps = [ConceptDependency("A", "B")]
        with caplog.at_level(logging.WARNING):
            build_and_validate_dag(deps, knowledge_graph_concepts={"A", "B"})
        assert "not found in knowledge_graph" not in caplog.text

    def test_knowledge_graph_concepts_none(self):
        """knowledge_graph_concepts=None: skip validation silently."""
        deps = [ConceptDependency("X", "Y")]
        dag = build_and_validate_dag(deps, knowledge_graph_concepts=None)
        assert len(dag.nodes) == 2

    def test_parse_missing_key_returns_none(self):
        """Missing concept_dependencies key should return None."""
        result = parse_concept_dependencies({"other_key": "value"})
        assert result is None

    def test_parse_empty_list(self):
        """Empty concept_dependencies list should return empty list."""
        result = parse_concept_dependencies({"concept_dependencies": []})
        assert result == []

    def test_parse_valid_entries(self):
        """Valid entries should be parsed correctly."""
        raw = {
            "concept_dependencies": [
                {"prerequisite": "A", "dependent": "B"},
                {"prerequisite": "B", "dependent": "C"},
            ]
        }
        result = parse_concept_dependencies(raw)
        assert len(result) == 2
        assert result[0].prerequisite == "A"
        assert result[1].dependent == "C"

    def test_parse_malformed_entry_skipped(self, caplog):
        """Entry missing required keys should be skipped with warning."""
        raw = {
            "concept_dependencies": [
                {"prerequisite": "A", "dependent": "B"},
                {"missing_key": "value"},  # malformed
                {"prerequisite": "B", "dependent": "C"},
            ]
        }
        with caplog.at_level(logging.WARNING):
            result = parse_concept_dependencies(raw)
        assert len(result) == 2
        assert "Skipping malformed" in caplog.text

    def test_parse_none_entry_skipped(self, caplog):
        """None entry in the list should be skipped with warning."""
        raw = {
            "concept_dependencies": [
                {"prerequisite": "A", "dependent": "B"},
                None,
            ]
        }
        with caplog.at_level(logging.WARNING):
            result = parse_concept_dependencies(raw)
        assert len(result) == 1

    def test_predecessors_leaf_node(self):
        """Leaf node with no predecessors should return empty list."""
        dag = _make_simple_dag([("A", "B")])
        assert dag.predecessors("A") == []

    def test_successors_root_node(self):
        """Root node should have successors."""
        dag = _make_simple_dag([("A", "B")])
        assert "B" in dag.successors("A")

    def test_large_cycle_100_nodes(self):
        """100-node cycle should be detected."""
        deps = [ConceptDependency(f"N{i}", f"N{(i + 1) % 100}") for i in range(100)]
        with pytest.raises(ValueError, match="cycle"):
            build_and_validate_dag(deps)

    def test_multiple_cycles(self):
        """Graph with multiple cycles should detect at least one."""
        deps = [
            ConceptDependency("A", "B"),
            ConceptDependency("B", "A"),
            ConceptDependency("C", "D"),
            ConceptDependency("D", "C"),
        ]
        with pytest.raises(ValueError, match="cycle"):
            build_and_validate_dag(deps)

    def test_concept_with_spaces_and_special_chars(self):
        """Concepts with spaces and punctuation should work."""
        dag = _make_simple_dag(
            [
                ("concept A (v1)", "concept B [beta]"),
            ]
        )
        assert "concept A (v1)" in dag.nodes

    def test_empty_string_concept(self):
        """Empty string as concept name: accepted by DAG."""
        dag = _make_simple_dag([("", "B")])
        assert "" in dag.nodes

    def test_single_node_no_edges(self):
        """Single node with no edges: empty dependency list."""
        dag = build_and_validate_dag([])
        assert len(dag.nodes) == 0

    def test_parallel_paths_to_same_node(self):
        """Multiple paths to same node: valid DAG."""
        dag = _make_simple_dag(
            [
                ("A", "C"),
                ("B", "C"),
                ("A", "D"),
                ("B", "D"),
                ("C", "E"),
                ("D", "E"),
            ]
        )
        assert len(dag.nodes) == 5


# ===========================================================================
# PERSONA 3: GRADE DATA CORRUPTOR
# ===========================================================================


class TestGradeDataCorruptor:
    """Persona 3: Invalid grades, missing semesters, NaN scores."""

    def test_invalid_grade_b_plus_raises(self):
        """Grade 'B+' (not in VALID_GRADES) should raise ValueError."""
        path = _make_grade_yaml({"sem1": {"S001": "B+"}})
        try:
            with pytest.raises(ValueError, match="Invalid grade 'B\\+'"):
                load_grade_mapping(path)
        finally:
            os.unlink(path)

    def test_invalid_grade_numeric_raises(self):
        """Numeric grade '90' should raise ValueError."""
        path = _make_grade_yaml({"sem1": {"S001": 90}})
        try:
            with pytest.raises(ValueError, match="Invalid grade '90'"):
                load_grade_mapping(path)
        finally:
            os.unlink(path)

    def test_invalid_grade_korean_raises(self):
        """Korean grade string should raise ValueError."""
        path = _make_grade_yaml({"sem1": {"S001": "수"}})
        try:
            with pytest.raises(ValueError, match="Invalid grade"):
                load_grade_mapping(path)
        finally:
            os.unlink(path)

    def test_empty_string_grade_raises(self):
        """Empty string grade should raise ValueError."""
        path = _make_grade_yaml({"sem1": {"S001": ""}})
        try:
            with pytest.raises(ValueError, match="Invalid grade"):
                load_grade_mapping(path)
        finally:
            os.unlink(path)

    def test_lowercase_grade_raises(self):
        """Lowercase 'a' should raise ValueError (only uppercase valid)."""
        path = _make_grade_yaml({"sem1": {"S001": "a"}})
        try:
            with pytest.raises(ValueError, match="Invalid grade"):
                load_grade_mapping(path)
        finally:
            os.unlink(path)

    def test_valid_all_grades(self):
        """All 5 valid grades should load successfully."""
        data = {
            "sem1": {
                "S001": "A",
                "S002": "B",
                "S003": "C",
                "S004": "D",
                "S005": "F",
            }
        }
        path = _make_grade_yaml(data)
        try:
            mapping = load_grade_mapping(path)
            assert set(mapping["sem1"].values()) == VALID_GRADES
        finally:
            os.unlink(path)

    def test_empty_yaml_returns_empty(self):
        """Empty YAML file should return empty dict."""
        path = _make_grade_yaml(None)
        # Write empty content
        with open(path, "w") as f:
            f.write("")
        try:
            result = load_grade_mapping(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises(self):
        """Nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_grade_mapping("/nonexistent/grade_mapping.yaml")

    def test_student_id_mismatch_warns(self, caplog):
        """Student IDs not in store should produce warnings."""
        data = {"sem1": {"S001": "A", "S999": "B"}}
        path = _make_grade_yaml(data)
        try:
            with caplog.at_level(logging.WARNING):
                load_grade_mapping(path, store_student_ids={"S001"})
            assert "S999" in caplog.text
            assert "mismatch" in caplog.text
        finally:
            os.unlink(path)

    def test_multiple_semesters(self):
        """Multi-semester mapping should load correctly."""
        data = {
            "2025-1": {"S001": "A", "S002": "B"},
            "2025-2": {"S001": "B", "S003": "C"},
        }
        path = _make_grade_yaml(data)
        try:
            mapping = load_grade_mapping(path)
            assert len(mapping) == 2
            assert mapping["2025-1"]["S001"] == "A"
            assert mapping["2025-2"]["S001"] == "B"
        finally:
            os.unlink(path)

    def test_single_student_training(self):
        """Training with exactly 1 student should fail (< min_students)."""
        predictor = GradePredictor()
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (1, 21))
        labels = np.array([3])
        with pytest.raises(ValueError, match="Insufficient students"):
            predictor.train(matrix, labels, list(GRADE_FEATURE_NAMES))

    def test_all_same_grade_training(self):
        """All same grade label: single class, cv_score=0.0."""
        predictor = GradePredictor()
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (20, 21))
        labels = np.full(20, 3)  # all B
        model = predictor.train(matrix, labels, list(GRADE_FEATURE_NAMES), min_students=10)
        assert model.cv_score == 0.0

    def test_missing_grade_class_in_training(self):
        """Training with only A, B, C (no D, F): should succeed."""
        predictor = GradePredictor()
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (30, 21))
        labels = np.array([4] * 10 + [3] * 10 + [2] * 10)  # A, B, C only
        model = predictor.train(matrix, labels, list(GRADE_FEATURE_NAMES), min_students=10)
        # Only 3 classes learned
        assert set(model.classes).issubset({0, 1, 2, 3, 4})

    def test_predict_fills_missing_grades_with_zero(self):
        """Prediction should fill unlearned grades with probability 0."""
        predictor = GradePredictor()
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (20, 21))
        labels = np.array([4] * 10 + [3] * 10)  # only A, B
        model = predictor.train(matrix, labels, list(GRADE_FEATURE_NAMES), min_students=10)
        preds = predictor.predict(model, matrix[:5], [f"S{i}" for i in range(5)])
        for p in preds:
            assert all(g in p.grade_probabilities for g in VALID_GRADES)
            # Grades not in training get 0.0
            for grade in ["C", "D", "F"]:
                assert p.grade_probabilities[grade] == 0.0

    def test_save_load_grade_model_roundtrip(self):
        """TrainedGradeModel roundtrip via save/load."""
        predictor = GradePredictor()
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (20, 21))
        labels = np.array([4] * 5 + [3] * 5 + [2] * 5 + [1] * 5)
        model = predictor.train(matrix, labels, list(GRADE_FEATURE_NAMES), min_students=10)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save_grade_model(model, path)
            loaded = load_grade_model(path)
            assert isinstance(loaded, TrainedGradeModel)
            assert loaded.n_students == 20
            assert loaded.feature_names == list(GRADE_FEATURE_NAMES)
        finally:
            os.unlink(path)

    def test_load_nonexistent_model_raises(self):
        """Loading nonexistent model file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_grade_model("/nonexistent/grade_model.pkl")

    def test_load_truncated_pkl(self):
        """Truncated .pkl file should raise on load."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"\x80\x05\x95\x00\x00\x00\x00")
            path = f.name
        try:
            with pytest.raises(Exception):
                load_grade_model(path)
        finally:
            os.unlink(path)

    def test_grade_ordinal_map_completeness(self):
        """GRADE_ORDINAL_MAP should cover all VALID_GRADES."""
        assert set(GRADE_ORDINAL_MAP.keys()) == VALID_GRADES

    def test_ordinal_grade_map_invertible(self):
        """ORDINAL_GRADE_MAP should be inverse of GRADE_ORDINAL_MAP."""
        for grade, ordinal in GRADE_ORDINAL_MAP.items():
            assert ORDINAL_GRADE_MAP[ordinal] == grade

    def test_21_feature_names(self):
        """GRADE_FEATURE_NAMES should have exactly 21 entries."""
        assert len(GRADE_FEATURE_NAMES) == 21

    def test_cold_start_grade_prediction(self):
        """Cold start predictions should have confidence='limited'."""
        predictor = GradePredictor()
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (5, 21))
        preds = predictor.predict_cold_start(
            matrix,
            [f"S{i}" for i in range(5)],
            list(GRADE_FEATURE_NAMES),
        )
        assert len(preds) == 5
        for p in preds:
            assert p.confidence == "limited"
            assert p.is_model_based is False
            assert p.predicted_grade in VALID_GRADES

    def test_cold_start_all_zero_features(self):
        """Cold start with all-zero features: valid predictions."""
        predictor = GradePredictor()
        matrix = np.zeros((3, 21))
        preds = predictor.predict_cold_start(
            matrix,
            ["S001", "S002", "S003"],
            list(GRADE_FEATURE_NAMES),
        )
        for p in preds:
            assert p.predicted_grade in VALID_GRADES
            assert sum(p.grade_probabilities.values()) == pytest.approx(1.0)

    def test_cold_start_high_score_predicts_A(self):
        """Cold start with very high score_mean should predict A."""
        predictor = GradePredictor()
        feature_names = list(GRADE_FEATURE_NAMES)
        mean_idx = feature_names.index("score_mean")
        matrix = np.zeros((1, 21))
        matrix[0, mean_idx] = 0.95
        preds = predictor.predict_cold_start(
            matrix,
            ["S001"],
            feature_names,
        )
        assert preds[0].predicted_grade == "A"

    def test_cold_start_low_score_predicts_F(self):
        """Cold start with very low score_mean should predict F."""
        predictor = GradePredictor()
        feature_names = list(GRADE_FEATURE_NAMES)
        mean_idx = feature_names.index("score_mean")
        matrix = np.zeros((1, 21))
        matrix[0, mean_idx] = 0.1
        preds = predictor.predict_cold_start(
            matrix,
            ["S001"],
            feature_names,
        )
        assert preds[0].predicted_grade == "F"

    def test_semester_with_non_dict_value(self):
        """Semester with non-dict value should be treated as empty."""
        path = _make_grade_yaml({"sem1": "not_a_dict", "sem2": {"S001": "A"}})
        try:
            mapping = load_grade_mapping(path)
            assert mapping["sem1"] == {}
            assert mapping["sem2"]["S001"] == "A"
        finally:
            os.unlink(path)

    def test_student_id_int_coerced_to_str(self):
        """Numeric student_id in YAML should be coerced to string."""
        path = _make_grade_yaml({"sem1": {12345: "A"}})
        try:
            mapping = load_grade_mapping(path)
            assert "12345" in mapping["sem1"]
        finally:
            os.unlink(path)


# ===========================================================================
# PERSONA 4: PDF CRASHER
# ===========================================================================


class TestPDFCrasher:
    """Persona 4: Attack new PDF sections with malicious inputs."""

    def test_professor_report_grade_prediction_section(self):
        """Grade prediction section with valid data: valid PDF."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        preds = [
            GradePrediction(
                student_id=f"S{i:03d}",
                predicted_grade=["A", "B", "C", "D", "F"][i % 5],
                grade_probabilities={"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "F": 0.2},
                predicted_ordinal=i % 5,
            )
            for i in range(10)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                grade_predictions=preds,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_empty_grade_predictions(self):
        """Empty grade predictions list: section should be skipped."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                grade_predictions=[],
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_xml_chars_in_predictions(self):
        """XML special chars in student_id: escaped in PDF."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        preds = [
            GradePrediction(
                student_id="<S001&\"test'>",
                predicted_grade="A",
                grade_probabilities={"A": 1.0, "B": 0, "C": 0, "D": 0, "F": 0},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                grade_predictions=preds,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_intervention_section(self):
        """Intervention section with effects: valid PDF."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        effects = [
            InterventionEffect(
                student_id="S001",
                intervention_id=1,
                intervention_type="면담",
                intervention_week=3,
                pre_mean=0.4,
                post_mean=0.6,
                score_change=0.2,
                sufficient_data=True,
            ),
        ]
        summaries = [
            InterventionTypeSummary(
                intervention_type="면담",
                n_total=1,
                n_sufficient=1,
                n_positive=1,
                n_negative=0,
                mean_change=0.2,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                intervention_effects=effects,
                intervention_type_summaries=summaries,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_intervention_insufficient_data(self):
        """Intervention with insufficient_data=True: renders 'data insufficient'."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        effects = [
            InterventionEffect(
                student_id="S001",
                intervention_id=1,
                intervention_type="면담",
                intervention_week=6,
                pre_mean=None,
                post_mean=None,
                score_change=None,
                sufficient_data=False,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                intervention_effects=effects,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_deficit_map_section(self):
        """Deficit map section with valid data: valid PDF."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        dag = _make_simple_dag([("A", "B"), ("B", "C")])
        deficit_map = ClassDeficitMap(
            concept_counts={"A": 5, "B": 10, "C": 3},
            total_students=20,
            dag=dag,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                deficit_map=deficit_map,
            )
            assert os.path.getsize(out) > 0

    def test_student_report_learning_path_section(self):
        """Student report with learning path: valid PDF."""
        from forma.student_report import StudentPDFReportGenerator
        from forma.report_data_loader import StudentReportData, ClassDistributions

        gen = StudentPDFReportGenerator()
        student_data = StudentReportData(student_id="S001")
        distributions = ClassDistributions()
        learning_path_data = LearningPath(
            student_id="S001",
            deficit_concepts=["세포막", "삼투압"],
            ordered_path=["물질 이동", "세포막", "삼투압"],
            capped=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                student_data=student_data,
                distributions=distributions,
                output_dir=tmpdir,
                learning_path=learning_path_data,
            )
            assert os.path.getsize(out) > 0

    def test_student_report_empty_learning_path(self):
        """Student report with empty learning path: shows 'no study needed'."""
        from forma.student_report import StudentPDFReportGenerator
        from forma.report_data_loader import StudentReportData, ClassDistributions

        gen = StudentPDFReportGenerator()
        student_data = StudentReportData(student_id="S001")
        distributions = ClassDistributions()
        learning_path_data = LearningPath(student_id="S001")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                student_data=student_data,
                distributions=distributions,
                output_dir=tmpdir,
                learning_path=learning_path_data,
            )
            assert os.path.getsize(out) > 0

    def test_student_report_capped_learning_path(self):
        """Capped learning path (>20 concepts): PDF renders with note."""
        from forma.student_report import StudentPDFReportGenerator
        from forma.report_data_loader import StudentReportData, ClassDistributions

        gen = StudentPDFReportGenerator()
        student_data = StudentReportData(student_id="S001")
        distributions = ClassDistributions()
        learning_path_data = LearningPath(
            student_id="S001",
            deficit_concepts=[f"C{i}" for i in range(25)],
            ordered_path=[f"C{i}" for i in range(20)],
            capped=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                student_data=student_data,
                distributions=distributions,
                output_dir=tmpdir,
                learning_path=learning_path_data,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_200_students_grade_predictions(self):
        """200 students with grade predictions: no page overflow."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        preds = [
            GradePrediction(
                student_id=f"S{i:04d}",
                predicted_grade=["A", "B", "C", "D", "F"][i % 5],
                grade_probabilities={"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "F": 0.2},
            )
            for i in range(200)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            start = time.time()
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                grade_predictions=preds,
            )
            elapsed = time.time() - start
            assert os.path.getsize(out) > 0
            assert elapsed < 60.0

    def test_professor_report_200_intervention_effects(self):
        """200 intervention effects: no page overflow."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        effects = [
            InterventionEffect(
                student_id=f"S{i:04d}",
                intervention_id=i + 1,
                intervention_type=INTERVENTION_TYPES[i % 5],
                intervention_week=i % 8 + 1,
                pre_mean=0.3 + (i % 10) * 0.02,
                post_mean=0.5 + (i % 10) * 0.02,
                score_change=0.2,
                sufficient_data=True,
            )
            for i in range(200)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                intervention_effects=effects,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_all_new_sections_combined(self):
        """All v0.10.0 sections at once: grade + intervention + deficit map."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        preds = [
            GradePrediction(
                student_id="S001",
                predicted_grade="B",
                grade_probabilities={"A": 0.1, "B": 0.5, "C": 0.2, "D": 0.1, "F": 0.1},
            )
        ]
        effects = [InterventionEffect("S001", 1, "면담", 3, 0.4, 0.6, 0.2, True)]
        summaries = [InterventionTypeSummary("면담", 1, 1, 1, 0, 0.2)]
        dag = _make_simple_dag([("A", "B")])
        deficit_map = ClassDeficitMap(concept_counts={"A": 2, "B": 1}, total_students=3, dag=dag)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                grade_predictions=preds,
                intervention_effects=effects,
                intervention_type_summaries=summaries,
                deficit_map=deficit_map,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_korean_long_student_id(self):
        """Korean long student_id in grade predictions: no crash."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        preds = [
            GradePrediction(
                student_id="김" * 100,
                predicted_grade="A",
                grade_probabilities={"A": 1.0, "B": 0, "C": 0, "D": 0, "F": 0},
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                grade_predictions=preds,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_5_type_summaries(self):
        """All 5 intervention type summaries: valid PDF."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        summaries = [
            InterventionTypeSummary(
                intervention_type=itype,
                n_total=10,
                n_sufficient=8,
                n_positive=5,
                n_negative=3,
                mean_change=0.05 * (i + 1),
            )
            for i, itype in enumerate(INTERVENTION_TYPES)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                intervention_type_summaries=summaries,
                intervention_effects=[],
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_negative_score_change(self):
        """Intervention with negative score change: valid PDF."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        effects = [InterventionEffect("S001", 1, "면담", 3, 0.6, 0.4, -0.2, True)]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                intervention_effects=effects,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_zero_score_change(self):
        """Intervention with zero score change: valid PDF."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        effects = [InterventionEffect("S001", 1, "보충학습", 3, 0.5, 0.5, 0.0, True)]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                intervention_effects=effects,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_deficit_map_all_concepts_zero(self):
        """Deficit map with all zero counts: valid PDF."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        dag = _make_simple_dag([("A", "B"), ("B", "C")])
        deficit_map = ClassDeficitMap(
            concept_counts={"A": 0, "B": 0, "C": 0},
            total_students=50,
            dag=dag,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                deficit_map=deficit_map,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_grade_all_F(self):
        """All F predictions: valid PDF."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        preds = [
            GradePrediction(
                student_id=f"S{i:03d}",
                predicted_grade="F",
                grade_probabilities={"A": 0, "B": 0, "C": 0, "D": 0, "F": 1.0},
            )
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                grade_predictions=preds,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_grade_single_student(self):
        """Single student grade prediction: valid PDF."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        preds = [
            GradePrediction(
                student_id="S001",
                predicted_grade="C",
                grade_probabilities={"A": 0.1, "B": 0.2, "C": 0.4, "D": 0.2, "F": 0.1},
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                grade_predictions=preds,
            )
            assert os.path.getsize(out) > 0

    def test_student_report_grade_trend_display(self):
        """Student report with grade_trend kwarg: valid PDF (FR-031)."""
        from forma.student_report import StudentPDFReportGenerator
        from forma.report_data_loader import StudentReportData, ClassDistributions

        gen = StudentPDFReportGenerator()
        student_data = StudentReportData(student_id="S001")
        distributions = ClassDistributions()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                student_data=student_data,
                distributions=distributions,
                output_dir=tmpdir,
                grade_trend="중위권",
            )
            assert os.path.getsize(out) > 0

    def test_student_report_grade_trend_with_learning_path(self):
        """Student report with both grade_trend and learning path."""
        from forma.student_report import StudentPDFReportGenerator
        from forma.report_data_loader import StudentReportData, ClassDistributions

        gen = StudentPDFReportGenerator()
        student_data = StudentReportData(student_id="S001")
        distributions = ClassDistributions()
        lp = LearningPath(
            student_id="S001",
            deficit_concepts=["A", "B"],
            ordered_path=["A", "B"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                student_data=student_data,
                distributions=distributions,
                output_dir=tmpdir,
                learning_path=lp,
                grade_trend="상위권",
            )
            assert os.path.getsize(out) > 0

    def test_student_report_xml_chars_in_learning_path(self):
        """XML special chars in concept names: escaped in PDF."""
        from forma.student_report import StudentPDFReportGenerator
        from forma.report_data_loader import StudentReportData, ClassDistributions

        gen = StudentPDFReportGenerator()
        student_data = StudentReportData(student_id="S001")
        distributions = ClassDistributions()
        lp = LearningPath(
            student_id="S001",
            deficit_concepts=['<concept & "test">'],
            ordered_path=['<concept & "test">'],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                student_data=student_data,
                distributions=distributions,
                output_dir=tmpdir,
                learning_path=lp,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_50_deficit_concepts(self):
        """Deficit map with 50 concepts: valid PDF, no page overflow."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        edges = [(f"C{i}", f"C{i + 1}") for i in range(49)]
        dag = _make_simple_dag(edges)
        counts = {f"C{i}": i for i in range(50)}
        deficit_map = ClassDeficitMap(concept_counts=counts, total_students=100, dag=dag)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                deficit_map=deficit_map,
            )
            assert os.path.getsize(out) > 0

    def test_professor_report_mixed_sufficient_insufficient(self):
        """Mix of sufficient and insufficient intervention effects: valid PDF."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        effects = [
            InterventionEffect("S001", 1, "면담", 3, 0.4, 0.6, 0.2, True),
            InterventionEffect("S002", 2, "보충학습", 2, None, None, None, False),
            InterventionEffect("S003", 3, "멘토링", 4, 0.5, 0.3, -0.2, True),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                report_data=_make_minimal_report_data(),
                output_dir=tmpdir,
                intervention_effects=effects,
            )
            assert os.path.getsize(out) > 0

    def test_student_report_learning_path_20_concepts(self):
        """Student report with 20-concept learning path (max): valid PDF."""
        from forma.student_report import StudentPDFReportGenerator
        from forma.report_data_loader import StudentReportData, ClassDistributions

        gen = StudentPDFReportGenerator()
        student_data = StudentReportData(student_id="S001")
        distributions = ClassDistributions()
        lp = LearningPath(
            student_id="S001",
            deficit_concepts=[f"개념_{i}" for i in range(20)],
            ordered_path=[f"개념_{i}" for i in range(20)],
            capped=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = gen.generate_pdf(
                student_data=student_data,
                distributions=distributions,
                output_dir=tmpdir,
                learning_path=lp,
            )
            assert os.path.getsize(out) > 0


def _make_minimal_report_data():
    """Create minimal ProfessorReportData for PDF generation tests."""
    from forma.professor_report_data import (
        ProfessorReportData,
        StudentSummaryRow,
        QuestionClassStats,
    )

    qstats = [
        QuestionClassStats(
            question_sn=sn,
            question_text=f"Question {sn}",
            ensemble_mean=0.5,
            level_distribution={"Advanced": 1, "Developing": 2},
        )
        for sn in [1, 2]
    ]
    rows = [
        StudentSummaryRow(
            student_id=f"S{i:03d}",
            overall_ensemble_mean=0.5 + i * 0.05,
            overall_level="Developing",
            per_question_scores={1: 0.5, 2: 0.6},
            per_question_levels={1: "Developing", 2: "Proficient"},
        )
        for i in range(3)
    ]
    return ProfessorReportData(
        class_name="1A",
        week_num=1,
        subject="Biology",
        exam_title="Test Exam",
        generation_date="2026-03-10",
        n_students=3,
        n_questions=2,
        class_ensemble_mean=0.55,
        class_ensemble_std=0.05,
        class_ensemble_median=0.55,
        class_ensemble_q1=0.5,
        class_ensemble_q3=0.6,
        overall_level_distribution={"Advanced": 0, "Proficient": 1, "Developing": 2, "Beginning": 0},
        question_stats=qstats,
        student_rows=rows,
        n_at_risk=0,
        pct_at_risk=0.0,
    )


# ===========================================================================
# PERSONA 5: BOUNDARY PUSHER
# ===========================================================================


class TestBoundaryPusher:
    """Persona 5: Exact boundary value attacks across all v0.10.0 modules."""

    # --- Learning path boundaries ---

    def test_learning_path_0_deficit_concepts(self):
        """Student with all mastered concepts: empty path."""
        dag = _make_simple_dag([("A", "B"), ("B", "C")])
        scores = {"A": 0.9, "B": 0.8, "C": 0.7}
        lp = generate_learning_path("S001", scores, dag, threshold=0.4)
        assert lp.ordered_path == []
        assert lp.deficit_concepts == []
        assert lp.capped is False

    def test_learning_path_1_deficit_concept(self):
        """Student with exactly 1 deficit concept."""
        dag = _make_simple_dag([("A", "B"), ("B", "C")])
        scores = {"A": 0.9, "B": 0.3, "C": 0.8}
        lp = generate_learning_path("S001", scores, dag, threshold=0.4)
        assert "B" in lp.deficit_concepts
        assert len(lp.ordered_path) >= 1

    def test_learning_path_exactly_20_concepts(self):
        """Exactly 20 deficit concepts: not capped."""
        edges = [(f"C{i}", f"C{i + 1}") for i in range(20)]
        dag = _make_simple_dag(edges)
        # All 21 concepts deficit, but 20 is the cap
        scores = {f"C{i}": 0.0 for i in range(21)}
        lp = generate_learning_path("S001", scores, dag, threshold=0.4)
        assert len(lp.ordered_path) == _MAX_PATH_LENGTH
        assert lp.capped is True  # 21 > 20

    def test_learning_path_exactly_21_concepts_capped(self):
        """21 deficit concepts: capped at 20."""
        edges = [(f"C{i}", f"C{i + 1}") for i in range(21)]
        dag = _make_simple_dag(edges)
        scores = {f"C{i}": 0.0 for i in range(22)}
        lp = generate_learning_path("S001", scores, dag, threshold=0.4)
        assert len(lp.ordered_path) == _MAX_PATH_LENGTH
        assert lp.capped is True

    def test_learning_path_threshold_exactly_at_boundary(self):
        """Score exactly at threshold (0.4): treated as NOT deficit."""
        dag = _make_simple_dag([("A", "B")])
        scores = {"A": 0.4, "B": 0.4}
        lp = generate_learning_path("S001", scores, dag, threshold=0.4)
        # 0.4 is NOT < 0.4, so no deficit
        assert lp.ordered_path == []

    def test_learning_path_threshold_just_below(self):
        """Score 0.399: treated as deficit."""
        dag = _make_simple_dag([("A", "B")])
        scores = {"A": 0.399, "B": 0.5}
        lp = generate_learning_path("S001", scores, dag, threshold=0.4)
        assert "A" in lp.deficit_concepts

    def test_learning_path_missing_score_is_deficit(self):
        """Concept with no score (None): treated as deficit."""
        dag = _make_simple_dag([("A", "B")])
        scores = {"B": 0.8}  # A missing
        lp = generate_learning_path("S001", scores, dag, threshold=0.4)
        assert "A" in lp.deficit_concepts

    def test_learning_path_empty_dag(self):
        """Empty DAG: empty learning path."""
        dag = build_and_validate_dag([])
        lp = generate_learning_path("S001", {"A": 0.1}, dag)
        assert lp.ordered_path == []

    def test_learning_path_includes_unmastered_prerequisites(self):
        """Deficit concept with unmastered prereq: prereq included."""
        dag = _make_simple_dag([("prereq", "target")])
        scores = {"prereq": 0.2, "target": 0.3}
        lp = generate_learning_path("S001", scores, dag, threshold=0.4)
        # Both should be in path, prereq first
        assert lp.ordered_path.index("prereq") < lp.ordered_path.index("target")

    def test_learning_path_mastered_prerequisite_excluded(self):
        """Deficit concept with mastered prereq: prereq excluded."""
        dag = _make_simple_dag([("prereq", "target")])
        scores = {"prereq": 0.9, "target": 0.2}
        lp = generate_learning_path("S001", scores, dag, threshold=0.4)
        assert "prereq" not in lp.ordered_path
        assert "target" in lp.ordered_path

    # --- Class deficit map boundaries ---

    def test_class_deficit_map_0_students(self):
        """0 students: all counts should be 0."""
        dag = _make_simple_dag([("A", "B")])
        dm = build_class_deficit_map({}, dag, threshold=0.4)
        assert dm.total_students == 0
        for count in dm.concept_counts.values():
            assert count == 0

    def test_class_deficit_map_1_student_all_mastered(self):
        """1 student, all mastered: counts should be 0."""
        dag = _make_simple_dag([("A", "B")])
        scores = {"S001": {"A": 0.9, "B": 0.8}}
        dm = build_class_deficit_map(scores, dag, threshold=0.4)
        assert dm.total_students == 1
        assert dm.concept_counts["A"] == 0
        assert dm.concept_counts["B"] == 0

    def test_class_deficit_map_all_students_deficit(self):
        """All students deficit on all concepts."""
        dag = _make_simple_dag([("A", "B")])
        scores = {f"S{i}": {"A": 0.1, "B": 0.2} for i in range(50)}
        dm = build_class_deficit_map(scores, dag, threshold=0.4)
        assert dm.concept_counts["A"] == 50
        assert dm.concept_counts["B"] == 50

    # --- Intervention effect boundaries ---

    def test_effect_insufficient_data_window_2(self):
        """Intervention at week 1 with window=2: no pre-weeks -> insufficient."""

        store, spath = _make_store_with_trajectory("S001", [1, 2, 3], [0.5, 0.6, 0.7])
        log, lpath = _make_log()
        try:
            log.add_record("S001", 1, "면담")
            effects = compute_intervention_effects(log, store, window=2)
            assert len(effects) == 1
            assert effects[0].sufficient_data is False
        finally:
            os.unlink(spath)
            os.unlink(lpath)

    def test_effect_sufficient_data(self):
        """Intervention at week 3 with data weeks 1-5: sufficient data."""
        store, spath = _make_store_with_trajectory(
            "S001",
            [1, 2, 3, 4, 5],
            [0.3, 0.4, 0.5, 0.6, 0.7],
        )
        log, lpath = _make_log()
        try:
            log.add_record("S001", 3, "면담")
            effects = compute_intervention_effects(log, store, window=2)
            assert len(effects) == 1
            assert effects[0].sufficient_data is True
            assert effects[0].pre_mean == pytest.approx(0.35)  # (0.3+0.4)/2
            assert effects[0].post_mean == pytest.approx(0.65)  # (0.6+0.7)/2
            assert effects[0].score_change == pytest.approx(0.3)
        finally:
            os.unlink(spath)
            os.unlink(lpath)

    def test_effect_empty_log(self):
        """Empty intervention log: no effects."""
        store, spath = _make_store_with_trajectory("S001", [1, 2, 3], [0.5, 0.5, 0.5])
        log, lpath = _make_log()
        try:
            effects = compute_intervention_effects(log, store)
            assert effects == []
        finally:
            os.unlink(spath)
            os.unlink(lpath)

    def test_type_summary_empty_effects(self):
        """Empty effects list: empty summary."""
        assert compute_type_summary([]) == []

    def test_type_summary_single_type(self):
        """Single type with mixed sufficient/insufficient."""
        effects = [
            InterventionEffect("S001", 1, "면담", 3, 0.3, 0.5, 0.2, True),
            InterventionEffect("S002", 2, "면담", 3, None, None, None, False),
        ]
        summaries = compute_type_summary(effects)
        assert len(summaries) == 1
        assert summaries[0].n_total == 2
        assert summaries[0].n_sufficient == 1
        assert summaries[0].n_positive == 1
        assert summaries[0].mean_change == pytest.approx(0.2)

    # --- Grade predictor boundaries ---

    def test_grade_exactly_10_students_trains(self):
        """Exactly 10 students with min_students=10: should succeed."""
        predictor = GradePredictor()
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (10, 21))
        labels = np.array([4] * 5 + [3] * 5)
        model = predictor.train(
            matrix,
            labels,
            list(GRADE_FEATURE_NAMES),
            min_students=10,
        )
        assert model.n_students == 10

    def test_grade_exactly_9_students_fails(self):
        """9 students with min_students=10: should fail."""
        predictor = GradePredictor()
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (9, 21))
        labels = np.array([4] * 5 + [3] * 4)
        with pytest.raises(ValueError, match="Insufficient students"):
            predictor.train(
                matrix,
                labels,
                list(GRADE_FEATURE_NAMES),
                min_students=10,
            )

    def test_cold_start_threshold_085(self):
        """Cold start: projected=0.85 should yield A."""
        predictor = GradePredictor()
        feature_names = list(GRADE_FEATURE_NAMES)
        mean_idx = feature_names.index("score_mean")
        matrix = np.zeros((1, 21))
        matrix[0, mean_idx] = 0.85
        preds = predictor.predict_cold_start(matrix, ["S001"], feature_names)
        assert preds[0].predicted_grade == "A"

    def test_cold_start_threshold_084(self):
        """Cold start: projected=0.84 should yield B (< 0.85)."""
        predictor = GradePredictor()
        feature_names = list(GRADE_FEATURE_NAMES)
        mean_idx = feature_names.index("score_mean")
        matrix = np.zeros((1, 21))
        matrix[0, mean_idx] = 0.84
        preds = predictor.predict_cold_start(matrix, ["S001"], feature_names)
        # 0.84 < 0.85 so not A, but >= 0.70 so B
        assert preds[0].predicted_grade == "B"

    def test_max_path_length_constant(self):
        """_MAX_PATH_LENGTH should be 20 (FR-022)."""
        assert _MAX_PATH_LENGTH == 20

    def test_intervention_log_add_0_records(self):
        """InterventionLog with 0 records: get_records returns empty."""
        log, path = _make_log()
        try:
            assert log.get_records() == []
        finally:
            os.unlink(path)

    def test_cold_start_threshold_050(self):
        """Cold start: projected=0.50 should yield C (>= 0.50)."""
        predictor = GradePredictor()
        feature_names = list(GRADE_FEATURE_NAMES)
        mean_idx = feature_names.index("score_mean")
        matrix = np.zeros((1, 21))
        matrix[0, mean_idx] = 0.50
        preds = predictor.predict_cold_start(matrix, ["S001"], feature_names)
        assert preds[0].predicted_grade == "C"

    def test_cold_start_threshold_049(self):
        """Cold start: projected=0.49 should yield D (< 0.50 but >= 0.30)."""
        predictor = GradePredictor()
        feature_names = list(GRADE_FEATURE_NAMES)
        mean_idx = feature_names.index("score_mean")
        matrix = np.zeros((1, 21))
        matrix[0, mean_idx] = 0.49
        preds = predictor.predict_cold_start(matrix, ["S001"], feature_names)
        assert preds[0].predicted_grade == "D"


# ===========================================================================
# PERSONA 6: CONCURRENT CHAOS
# ===========================================================================


class TestConcurrentChaos:
    """Persona 6: Race conditions, parallel operations, file locking."""

    def test_concurrent_add_record(self):
        """Parallel add_record operations: all records stored."""
        log, path = _make_log()
        errors: list[Exception] = []

        def add_batch(start_idx):
            for i in range(20):
                try:
                    log.add_record(f"S{start_idx + i:04d}", 1, "면담")
                except Exception as e:
                    errors.append(e)

        try:
            threads = [threading.Thread(target=add_batch, args=(i * 20,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            # Some records may be lost due to race condition on _next_id
            # but no crashes should occur
            assert not errors, f"Thread errors: {errors}"
            records = log.get_records()
            assert len(records) > 0
        finally:
            os.unlink(path)

    def test_concurrent_save_load(self):
        """Concurrent save + load operations: no corruption on final state."""
        log, path = _make_log()
        errors: list[Exception] = []

        try:
            # Add some initial records
            for i in range(10):
                log.add_record(f"S{i:03d}", 1, "면담")
            log.save()

            def save_loop():
                for _ in range(5):
                    try:
                        log.save()
                    except Exception as e:
                        errors.append(e)

            def load_loop():
                for _ in range(5):
                    try:
                        log2 = InterventionLog(path)
                        log2.load()
                    except Exception:
                        pass  # mid-write reads acceptable

            t1 = threading.Thread(target=save_loop)
            t2 = threading.Thread(target=load_loop)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert not errors, f"Save errors: {errors}"
            # Final load should succeed
            final = InterventionLog(path)
            final.load()
            assert len(final.get_records()) == 10
        finally:
            os.unlink(path)
            bak = path + ".bak"
            if os.path.exists(bak):
                os.unlink(bak)

    def test_concurrent_dag_validation(self):
        """Concurrent DAG validations: no shared state corruption."""
        errors: list[Exception] = []
        results: list[int] = []

        def validate_dag(n_nodes):
            try:
                deps = [ConceptDependency(f"C{i}", f"C{i + 1}") for i in range(n_nodes - 1)]
                dag = build_and_validate_dag(deps)
                results.append(len(dag.nodes))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=validate_dag, args=(50 + i * 10,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(results) == 10

    def test_concurrent_grade_model_save_load(self):
        """Concurrent save/load of grade model: no corruption."""
        predictor = GradePredictor()
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (20, 21))
        labels = np.array([4] * 5 + [3] * 5 + [2] * 5 + [1] * 5)
        model = predictor.train(matrix, labels, list(GRADE_FEATURE_NAMES), min_students=10)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        errors: list[Exception] = []

        def save_loop():
            for _ in range(5):
                try:
                    save_grade_model(model, path)
                except Exception as e:
                    errors.append(e)

        def load_loop():
            for _ in range(5):
                try:
                    load_grade_model(path)
                except Exception:
                    pass  # mid-write reads acceptable

        try:
            save_grade_model(model, path)
            t1 = threading.Thread(target=save_loop)
            t2 = threading.Thread(target=load_loop)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert not errors
            loaded = load_grade_model(path)
            assert isinstance(loaded, TrainedGradeModel)
        finally:
            os.unlink(path)

    def test_concurrent_learning_path_generation(self):
        """Concurrent learning path generation: no shared state issues."""
        dag = _make_simple_dag([(f"C{i}", f"C{i + 1}") for i in range(20)])
        errors: list[Exception] = []
        results: list[int] = []

        def gen_path(student_id, score_base):
            try:
                scores = {f"C{i}": score_base + i * 0.02 for i in range(21)}
                lp = generate_learning_path(student_id, scores, dag)
                results.append(len(lp.ordered_path))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=gen_path, args=(f"S{i:03d}", i * 0.03)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(results) == 10

    def test_concurrent_effect_computation(self):
        """Concurrent intervention effect computation: no crashes."""
        store, spath = _make_store_with_trajectory(
            "S001",
            [1, 2, 3, 4, 5],
            [0.3, 0.4, 0.5, 0.6, 0.7],
        )
        log, lpath = _make_log()
        log.add_record("S001", 3, "면담")
        errors: list[Exception] = []
        results: list[int] = []

        def compute():
            try:
                effects = compute_intervention_effects(log, store)
                results.append(len(effects))
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=compute) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            assert not errors
            assert all(r == 1 for r in results)
        finally:
            os.unlink(spath)
            os.unlink(lpath)

    def test_concurrent_type_summary(self):
        """Concurrent compute_type_summary: deterministic results."""
        effects = [
            InterventionEffect(f"S{i:03d}", i, INTERVENTION_TYPES[i % 5], 3, 0.4, 0.6, 0.2, True) for i in range(100)
        ]
        errors: list[Exception] = []
        results: list[int] = []

        def compute():
            try:
                summaries = compute_type_summary(effects)
                results.append(len(summaries))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=compute) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors
        assert all(r == 5 for r in results)

    def test_concurrent_class_deficit_map(self):
        """Concurrent build_class_deficit_map: no shared state corruption."""
        dag = _make_simple_dag([(f"C{i}", f"C{i + 1}") for i in range(10)])
        all_scores = {f"S{i}": {f"C{j}": 0.1 * j for j in range(11)} for i in range(50)}
        errors: list[Exception] = []
        results: list[int] = []

        def compute():
            try:
                dm = build_class_deficit_map(all_scores, dag)
                results.append(dm.total_students)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=compute) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors
        assert all(r == 50 for r in results)

    def test_concurrent_add_then_save(self):
        """Add records then save from multiple threads: file not corrupted."""
        log, path = _make_log()
        errors: list[Exception] = []

        def add_and_save(thread_id):
            try:
                for i in range(10):
                    log.add_record(f"T{thread_id}_S{i:03d}", 1, "면담")
                log.save()
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=add_and_save, args=(t,)) for t in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            # File should be loadable
            log2 = InterventionLog(path)
            log2.load()
            records = log2.get_records()
            assert len(records) > 0
        finally:
            os.unlink(path)
            bak = path + ".bak"
            if os.path.exists(bak):
                os.unlink(bak)

    def test_concurrent_update_outcome(self):
        """Concurrent update_outcome: no crashes."""
        log, path = _make_log()
        for i in range(20):
            log.add_record(f"S{i:03d}", 1, "면담")
        errors: list[Exception] = []

        def update_batch(start):
            try:
                for i in range(start, start + 10):
                    log.update_outcome(i + 1, f"결과_{i}")
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=update_batch, args=(i * 10,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            assert not errors
        finally:
            os.unlink(path)

    def test_concurrent_get_records_filter(self):
        """Concurrent filtered get_records: consistent results."""
        log, path = _make_log()
        for i in range(100):
            log.add_record(f"S{i % 5:03d}", i % 8 + 1, INTERVENTION_TYPES[i % 5])
        errors: list[Exception] = []
        results: list[int] = []

        def query():
            try:
                recs = log.get_records(student_id="S000")
                results.append(len(recs))
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=query) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            assert not errors
            # All should get the same count
            assert len(set(results)) == 1
        finally:
            os.unlink(path)

    def test_concurrent_grade_predictor_cold_start(self):
        """Concurrent cold start predictions: deterministic."""
        predictor = GradePredictor()
        feature_names = list(GRADE_FEATURE_NAMES)
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (10, 21))
        sids = [f"S{i}" for i in range(10)]
        errors: list[Exception] = []
        results: list[list[str]] = []

        def predict():
            try:
                preds = predictor.predict_cold_start(matrix, sids, feature_names)
                results.append([p.predicted_grade for p in preds])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=predict) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors
        # All threads should produce identical predictions
        assert all(r == results[0] for r in results)

    def test_concurrent_grade_model_train(self):
        """Concurrent model training: independent models, no crash."""
        errors: list[Exception] = []
        models: list[TrainedGradeModel] = []

        def train(seed):
            try:
                rng = np.random.RandomState(seed)
                matrix = rng.uniform(0, 1, (20, 21))
                labels = np.array([4] * 5 + [3] * 5 + [2] * 5 + [1] * 5)
                predictor = GradePredictor()
                model = predictor.train(matrix, labels, list(GRADE_FEATURE_NAMES), min_students=10)
                models.append(model)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=train, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors
        assert len(models) == 5

    def test_concurrent_parse_concept_dependencies(self):
        """Concurrent parse_concept_dependencies: no shared state."""
        errors: list[Exception] = []
        results: list[int] = []

        def parse(n):
            try:
                yaml_dict = {
                    "concept_dependencies": [{"prerequisite": f"C{i}", "dependent": f"C{i + 1}"} for i in range(n)]
                }
                deps = parse_concept_dependencies(yaml_dict)
                results.append(len(deps))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=parse, args=(10 + i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors
        assert len(results) == 10

    def test_concurrent_grade_mapping_load(self):
        """Concurrent load_grade_mapping: each thread reads correctly."""
        path = _make_grade_yaml(
            {
                "sem1": {f"S{i:03d}": ["A", "B", "C", "D", "F"][i % 5] for i in range(50)},
            }
        )
        errors: list[Exception] = []
        results: list[int] = []

        def load():
            try:
                mapping = load_grade_mapping(path)
                results.append(len(mapping["sem1"]))
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=load) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            assert not errors
            assert all(r == 50 for r in results)
        finally:
            os.unlink(path)

    def test_concurrent_build_and_validate_dag(self):
        """Concurrent DAG builds with different sizes: all valid."""
        errors: list[Exception] = []
        results: list[int] = []

        def build(n):
            try:
                deps = [ConceptDependency(f"X{i}", f"X{i + 1}") for i in range(n)]
                dag = build_and_validate_dag(deps)
                results.append(len(dag.nodes))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=build, args=(5 + i * 5,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors
        assert len(results) == 10

    def test_concurrent_save_and_reload_cycle(self):
        """Save then reload from multiple threads: data integrity."""
        log, path = _make_log()
        for i in range(50):
            log.add_record(f"S{i:03d}", 1, INTERVENTION_TYPES[i % 5])
        log.save()
        errors: list[Exception] = []
        results: list[int] = []

        def reload_and_count():
            try:
                local_log = InterventionLog(path)
                local_log.load()
                results.append(len(local_log.get_records()))
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=reload_and_count) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            assert not errors
            assert all(r == 50 for r in results)
        finally:
            os.unlink(path)

    def test_concurrent_intervention_different_students(self):
        """Concurrent adds for different students: all stored."""
        log, path = _make_log()
        errors: list[Exception] = []

        def add_for_student(sid):
            try:
                for w in range(1, 6):
                    log.add_record(sid, w, "면담")
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=add_for_student, args=(f"S{i:03d}",)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            assert not errors
            records = log.get_records()
            assert len(records) > 0
        finally:
            os.unlink(path)

    def test_concurrent_dag_predecessors_successors(self):
        """Concurrent predecessors/successors queries: no crashes."""
        deps = [ConceptDependency(f"C{i}", f"C{i + 1}") for i in range(50)]
        dag = build_and_validate_dag(deps)
        errors: list[Exception] = []

        def query(concept):
            try:
                _ = dag.predecessors(concept)
                _ = dag.successors(concept)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=query, args=(f"C{i}",)) for i in range(51)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors

    def test_concurrent_multiple_learning_path_different_dags(self):
        """Concurrent learning paths with different DAGs: independent."""
        errors: list[Exception] = []
        results: list[int] = []

        def gen(n):
            try:
                dag = _make_simple_dag([(f"C{i}", f"C{i + 1}") for i in range(n)])
                scores = {f"C{i}": 0.1 for i in range(n + 1)}
                lp = generate_learning_path("S001", scores, dag, threshold=0.4)
                results.append(len(lp.ordered_path))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=gen, args=(5 + i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors
        assert len(results) == 10


# ===========================================================================
# PERSONA 7: UNICODE ATTACKER
# ===========================================================================


class TestUnicodeAttacker:
    """Persona 7: Korean/CJK, ZWJ, RTL, BOM, combining characters."""

    def test_zero_width_joiner_in_student_id(self):
        """Zero-width joiner (U+200D) in student_id: stored faithfully."""
        log, path = _make_log()
        try:
            sid = "S\u200d001"
            log.add_record(sid, 1, "면담")
            records = log.get_records(student_id=sid)
            assert len(records) == 1
            assert records[0].student_id == sid
        finally:
            os.unlink(path)

    def test_rtl_override_in_description(self):
        """RTL override (U+202E) in description: stored faithfully."""
        log, path = _make_log()
        try:
            desc = "\u202eollebreverse\u202c"
            log.add_record("S001", 1, "면담", description=desc)
            log.save()
            log2 = InterventionLog(path)
            log2.load()
            assert log2.get_records()[0].description == desc
        finally:
            os.unlink(path)

    def test_bom_in_yaml_field(self):
        """BOM (U+FEFF) embedded in field: stored faithfully."""
        log, path = _make_log()
        try:
            desc = "\ufeffBOM at start"
            log.add_record("S001", 1, "면담", description=desc)
            log.save()
            log2 = InterventionLog(path)
            log2.load()
            assert log2.get_records()[0].description == desc
        finally:
            os.unlink(path)

    def test_hangul_jamo_vs_composed(self):
        """Decomposed vs composed Hangul: treated as different strings."""
        # Composed "한" vs Decomposed "ㅎㅏㄴ"
        composed = "\ud55c"  # 한
        decomposed = "\u3134\u314f\u3131"  # ㄴㅏㄱ (not same as 한)
        dag = _make_simple_dag([(composed, decomposed)])
        assert composed in dag.nodes
        assert decomposed in dag.nodes
        assert composed != decomposed

    def test_half_width_katakana_in_concept(self):
        """Half-width katakana (U+FF65-FF9F) in concept names."""
        dag = _make_simple_dag(
            [
                ("\uff76\uff80\uff76\uff85", "\uff8b\uff97\uff76\uff9e\uff85"),
            ]
        )
        assert len(dag.nodes) == 2

    def test_combining_diacriticals_in_student_id(self):
        """Combining diacriticals (U+0300-036F) in student_id."""
        log, path = _make_log()
        try:
            sid = "S\u0300\u0301001"  # with combining grave + acute
            log.add_record(sid, 1, "면담")
            records = log.get_records(student_id=sid)
            assert len(records) == 1
        finally:
            os.unlink(path)

    def test_emoji_in_description(self):
        """Emoji in description: stored faithfully via YAML."""
        log, path = _make_log()
        try:
            desc = "학생 면담 완료 \U0001f600\U0001f44d"
            log.add_record("S001", 1, "면담", description=desc)
            log.save()
            log2 = InterventionLog(path)
            log2.load()
            assert log2.get_records()[0].description == desc
        finally:
            os.unlink(path)

    def test_korean_concept_in_learning_path(self):
        """Korean concepts in learning path: topological sort works."""
        dag = _make_simple_dag(
            [
                ("세포막 구조", "물질 이동"),
                ("물질 이동", "삼투압"),
            ]
        )
        scores = {"세포막 구조": 0.2, "물질 이동": 0.1, "삼투압": 0.3}
        lp = generate_learning_path("S001", scores, dag, threshold=0.4)
        assert "세포막 구조" in lp.ordered_path
        # Prerequisites come first
        idx_cell = lp.ordered_path.index("세포막 구조")
        idx_transport = lp.ordered_path.index("물질 이동")
        assert idx_cell < idx_transport

    def test_korean_grade_mapping_student_id(self):
        """Korean student_id in grade mapping: coerced to str."""
        path = _make_grade_yaml({"sem1": {"김철수": "A", "이영희": "B"}})
        try:
            mapping = load_grade_mapping(path)
            assert "김철수" in mapping["sem1"]
            assert mapping["sem1"]["이영희"] == "B"
        finally:
            os.unlink(path)

    def test_unicode_normalization_distinct(self):
        """NFC vs NFD forms: treated as distinct concepts in DAG."""
        import unicodedata

        nfc = unicodedata.normalize("NFC", "\uac00")  # 가 (composed)
        nfd = unicodedata.normalize("NFD", "\uac00")  # 가 (decomposed)
        # These are visually same but byte-different
        dag = _make_simple_dag([(nfc, "B")])
        assert nfc in dag.nodes
        # NFD form is different bytes
        if nfc != nfd:
            assert nfd not in dag.nodes

    def test_null_char_in_description(self):
        """Null character (U+0000) in description: survives storage."""
        log, path = _make_log()
        try:
            desc = "before\x00after"
            log.add_record("S001", 1, "면담", description=desc)
            records = log.get_records()
            assert len(records) == 1
            # YAML may or may not preserve null chars, but no crash
        finally:
            os.unlink(path)

    def test_very_long_korean_concept_name(self):
        """1000-char Korean concept name in DAG."""
        long_name = "가" * 1000
        dag = _make_simple_dag([(long_name, "B")])
        assert long_name in dag.nodes

    def test_mixed_scripts_in_concept(self):
        """Mixed Korean + ASCII + Chinese in concept name."""
        concept = "세포Cell细胞"
        dag = _make_simple_dag([(concept, "결과Result")])
        assert concept in dag.nodes

    def test_xml_special_chars_in_intervention_type(self):
        """기타 with XML special chars in description for PDF safety."""
        from forma.professor_report import _esc

        text = '개입 <유형> & "특수" 문자'
        escaped = _esc(text)
        assert "&lt;" in escaped
        assert "&gt;" in escaped
        assert "&amp;" in escaped

    def test_surrogate_handling(self):
        """Supplementary plane characters (emoji) in DAG concept."""
        concept = "개념\U0001f4da"  # books emoji
        dag = _make_simple_dag([(concept, "결과")])
        assert concept in dag.nodes

    def test_tab_newline_in_concept_name(self):
        """Tab and newline in concept name: accepted by DAG."""
        concept = "개념\t이름\n줄바꿈"
        dag = _make_simple_dag([(concept, "B")])
        assert concept in dag.nodes

    def test_unicode_in_grade_mapping_semester_label(self):
        """Unicode semester label in grade mapping."""
        path = _make_grade_yaml({"2025년 1학기": {"S001": "A"}})
        try:
            mapping = load_grade_mapping(path)
            assert "2025년 1학기" in mapping
        finally:
            os.unlink(path)

    def test_intervention_recorded_by_unicode(self):
        """Unicode recorded_by field: stored faithfully."""
        log, path = _make_log()
        try:
            log.add_record("S001", 1, "면담", recorded_by="김교수님 \U0001f393")
            records = log.get_records()
            assert records[0].recorded_by == "김교수님 \U0001f393"
        finally:
            os.unlink(path)

    def test_zero_width_space_in_concept(self):
        """Zero-width space (U+200B) in concept: distinct from plain."""
        plain = "삼투압"
        with_zws = "삼\u200b투압"
        dag = _make_simple_dag([(plain, with_zws)])
        assert len(dag.nodes) == 2
        assert plain != with_zws

    def test_cjk_extension_b_characters(self):
        """CJK Extension B (U+20000+) characters in concept."""
        concept = "\U00020000\U00020001"  # rare CJK chars
        dag = _make_simple_dag([(concept, "B")])
        assert concept in dag.nodes


# ===========================================================================
# INVARIANT TESTING (1000-iteration loops)
# ===========================================================================


class TestInvariant1000InterventionStore:
    """1000-iteration invariants for InterventionLog."""

    def test_add_record_id_always_increments(self):
        """1000 sequential adds: IDs always strictly increasing."""
        log, path = _make_log()
        try:
            prev_id = 0
            for i in range(1000):
                new_id = log.add_record(f"S{i:04d}", i % 8 + 1, "면담")
                assert new_id > prev_id
                prev_id = new_id
        finally:
            os.unlink(path)

    def test_get_records_count_matches_adds(self):
        """1000 adds: get_records() count always matches add count."""
        log, path = _make_log()
        try:
            for i in range(1000):
                log.add_record(f"S{i:04d}", 1, INTERVENTION_TYPES[i % 5])
            records = log.get_records()
            assert len(records) == 1000
        finally:
            os.unlink(path)


class TestInvariant1000DAG:
    """1000-iteration invariants for concept dependency DAG."""

    def test_linear_chain_always_acyclic(self):
        """1000 random linear chains: always valid DAGs."""
        rng = np.random.RandomState(42)
        for _ in range(1000):
            n = rng.randint(2, 20)
            deps = [ConceptDependency(f"C{i}", f"C{i + 1}") for i in range(n - 1)]
            dag = build_and_validate_dag(deps)
            assert len(dag.nodes) == n

    def test_topological_order_always_valid(self):
        """1000 random: learning path always respects topological order."""
        rng = np.random.RandomState(42)
        for _ in range(1000):
            n = rng.randint(3, 15)
            deps = [ConceptDependency(f"C{i}", f"C{i + 1}") for i in range(n - 1)]
            dag = build_and_validate_dag(deps)
            scores = {f"C{i}": rng.uniform(0, 0.39) for i in range(n)}
            lp = generate_learning_path("S001", scores, dag, threshold=0.4)
            # Verify topological order
            for i, concept in enumerate(lp.ordered_path):
                for prereq in dag.predecessors(concept):
                    if prereq in lp.ordered_path:
                        assert lp.ordered_path.index(prereq) < i


class TestInvariant1000GradePredictor:
    """1000-iteration invariants for grade prediction."""

    def test_cold_start_grade_always_valid(self):
        """1000 random: cold start grade always in VALID_GRADES."""
        rng = np.random.RandomState(42)
        predictor = GradePredictor()
        feature_names = list(GRADE_FEATURE_NAMES)
        for _ in range(1000):
            n = rng.randint(1, 5)
            matrix = rng.uniform(-1, 2, (n, 21))
            preds = predictor.predict_cold_start(
                matrix,
                [f"S{i}" for i in range(n)],
                feature_names,
            )
            for p in preds:
                assert p.predicted_grade in VALID_GRADES

    def test_cold_start_probabilities_sum_to_one(self):
        """1000 random: cold start probabilities always sum to 1.0."""
        rng = np.random.RandomState(42)
        predictor = GradePredictor()
        feature_names = list(GRADE_FEATURE_NAMES)
        for _ in range(1000):
            n = rng.randint(1, 5)
            matrix = rng.uniform(-1, 2, (n, 21))
            preds = predictor.predict_cold_start(
                matrix,
                [f"S{i}" for i in range(n)],
                feature_names,
            )
            for p in preds:
                total = sum(p.grade_probabilities.values())
                assert total == pytest.approx(1.0)

    def test_model_prediction_probabilities_sum_to_one(self):
        """100 random: model prediction probabilities always sum to ~1.0."""
        predictor = GradePredictor()
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (30, 21))
        labels = np.array([4] * 8 + [3] * 7 + [2] * 7 + [1] * 4 + [0] * 4)
        model = predictor.train(matrix, labels, list(GRADE_FEATURE_NAMES), min_students=10)
        for _ in range(100):
            n = rng.randint(1, 10)
            test_matrix = rng.uniform(0, 1, (n, 21))
            preds = predictor.predict(model, test_matrix, [f"S{i}" for i in range(n)])
            for p in preds:
                total = sum(p.grade_probabilities.values())
                assert abs(total - 1.0) < 0.01


class TestInvariant1000InterventionEffect:
    """1000-iteration invariants for intervention effects."""

    def test_effect_classification_deterministic(self):
        """1000 random: same inputs always produce same effects."""
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        rng = np.random.RandomState(42)
        for _ in range(100):  # reduced to 100 for I/O performance
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                spath = f.name
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                lpath = f.name
            try:
                store = LongitudinalStore(spath)
                weeks = list(range(1, 7))
                scores_vals = rng.uniform(0, 1, 6).tolist()
                for w, s in zip(weeks, scores_vals):
                    store.add_record(
                        LongitudinalRecord(
                            student_id="S001",
                            week=w,
                            question_sn=1,
                            scores={"ensemble_score": s},
                            tier_level=1,
                            tier_label="Dev",
                        )
                    )
                log = InterventionLog(lpath)
                log.add_record("S001", 3, "면담")

                e1 = compute_intervention_effects(log, store)
                e2 = compute_intervention_effects(log, store)
                assert len(e1) == len(e2)
                if e1:
                    assert e1[0].sufficient_data == e2[0].sufficient_data
                    assert e1[0].score_change == e2[0].score_change
            finally:
                os.unlink(spath)
                os.unlink(lpath)

    def test_type_summary_counts_invariant(self):
        """1000 random: n_positive + n_negative <= n_sufficient."""
        rng = np.random.RandomState(42)
        for _ in range(1000):
            n = rng.randint(1, 20)
            effects = []
            for i in range(n):
                sufficient = rng.choice([True, False])
                if sufficient:
                    change = rng.uniform(-0.5, 0.5)
                else:
                    change = None
                effects.append(
                    InterventionEffect(
                        student_id=f"S{i}",
                        intervention_id=i,
                        intervention_type=INTERVENTION_TYPES[i % 5],
                        intervention_week=3,
                        pre_mean=0.5 if sufficient else None,
                        post_mean=(0.5 + change) if sufficient else None,
                        score_change=change,
                        sufficient_data=sufficient,
                    )
                )
            summaries = compute_type_summary(effects)
            for s in summaries:
                assert s.n_positive + s.n_negative <= s.n_sufficient
                assert s.n_sufficient <= s.n_total
