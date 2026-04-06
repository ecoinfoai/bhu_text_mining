"""Tests for learning_path module — LearningPath, generate_learning_path, ClassDeficitMap.

Phase 6 (T037, T038): RED phase tests for US4.
Covers FR-018, FR-019, FR-020, FR-021, FR-022.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# T037: generate_learning_path tests (FR-018, FR-019, FR-022)
# ---------------------------------------------------------------------------


class TestGenerateLearningPath:
    """Tests for generate_learning_path()."""

    def _make_dag(self, dep_dicts):
        """Helper: build a ConceptDependencyDAG from list of dicts."""
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [ConceptDependency(prerequisite=d["prerequisite"], dependent=d["dependent"]) for d in dep_dicts]
        return build_and_validate_dag(deps)

    def test_basic_learning_path(self):
        """FR-018: deficit concepts produce ordered learning path."""
        from forma.learning_path import generate_learning_path

        # DAG: A → B → C → D
        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
                {"prerequisite": "B", "dependent": "C"},
                {"prerequisite": "C", "dependent": "D"},
            ]
        )
        # Student has deficit in C and D (scores below threshold)
        student_scores = {"A": 0.8, "B": 0.7, "C": 0.2, "D": 0.1}
        path = generate_learning_path("s001", student_scores, dag, threshold=0.4)
        assert path.student_id == "s001"
        assert "C" in path.ordered_path
        assert "D" in path.ordered_path
        # Topological order: C before D
        assert path.ordered_path.index("C") < path.ordered_path.index("D")

    def test_prerequisite_inclusion(self):
        """FR-019: unmastered prerequisites are included in path."""
        from forma.learning_path import generate_learning_path

        # DAG: A → B → C
        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
                {"prerequisite": "B", "dependent": "C"},
            ]
        )
        # Student only has deficit in C, but B is also unmastered
        student_scores = {"A": 0.8, "B": 0.3, "C": 0.2}
        path = generate_learning_path("s001", student_scores, dag, threshold=0.4)
        # Both B and C should be in path
        assert "B" in path.ordered_path
        assert "C" in path.ordered_path
        # B before C (prerequisite order)
        assert path.ordered_path.index("B") < path.ordered_path.index("C")

    def test_all_mastered_returns_empty_path(self):
        """All concepts above threshold → empty path."""
        from forma.learning_path import generate_learning_path

        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
            ]
        )
        student_scores = {"A": 0.9, "B": 0.8}
        path = generate_learning_path("s001", student_scores, dag, threshold=0.4)
        assert path.ordered_path == []
        assert path.deficit_concepts == []

    def test_cap_at_20_concepts(self):
        """FR-022: path capped at 20 concepts."""
        from forma.learning_path import generate_learning_path

        # Long chain: C0 → C1 → ... → C29 (30 nodes)
        dag = self._make_dag([{"prerequisite": f"C{i}", "dependent": f"C{i + 1}"} for i in range(29)])
        # All concepts are deficit
        student_scores = {f"C{i}": 0.1 for i in range(30)}
        path = generate_learning_path("s001", student_scores, dag, threshold=0.4)
        assert len(path.ordered_path) <= 20
        assert path.capped is True

    def test_not_capped_under_20(self):
        """Path under 20 concepts is not capped."""
        from forma.learning_path import generate_learning_path

        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
                {"prerequisite": "B", "dependent": "C"},
            ]
        )
        student_scores = {"A": 0.1, "B": 0.2, "C": 0.3}
        path = generate_learning_path("s001", student_scores, dag, threshold=0.4)
        assert path.capped is False

    def test_default_threshold(self):
        """Default threshold is 0.4."""
        from forma.learning_path import generate_learning_path

        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
            ]
        )
        # B is at 0.39, just below default 0.4
        student_scores = {"A": 0.8, "B": 0.39}
        path = generate_learning_path("s001", student_scores, dag)
        assert "B" in path.ordered_path

    def test_score_at_threshold_not_deficit(self):
        """Score exactly at threshold is NOT a deficit."""
        from forma.learning_path import generate_learning_path

        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
            ]
        )
        student_scores = {"A": 0.8, "B": 0.4}
        path = generate_learning_path("s001", student_scores, dag, threshold=0.4)
        assert "B" not in path.ordered_path

    def test_concept_not_in_scores_treated_as_deficit(self):
        """Concept in DAG but not in student_scores → treated as deficit."""
        from forma.learning_path import generate_learning_path

        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
            ]
        )
        # B has no score entry
        student_scores = {"A": 0.8}
        path = generate_learning_path("s001", student_scores, dag, threshold=0.4)
        assert "B" in path.ordered_path

    def test_diamond_dag_path(self):
        """Diamond DAG (A→B, A→C, B→D, C→D): all deficit → topological order."""
        from forma.learning_path import generate_learning_path

        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
                {"prerequisite": "A", "dependent": "C"},
                {"prerequisite": "B", "dependent": "D"},
                {"prerequisite": "C", "dependent": "D"},
            ]
        )
        student_scores = {"A": 0.1, "B": 0.2, "C": 0.2, "D": 0.1}
        path = generate_learning_path("s001", student_scores, dag, threshold=0.4)
        assert "A" in path.ordered_path
        assert "D" in path.ordered_path
        # A must come before B, C, D
        assert path.ordered_path.index("A") < path.ordered_path.index("D")

    def test_empty_dag_returns_empty_path(self):
        """Empty DAG (no deps) → empty path."""
        from forma.concept_dependency import build_and_validate_dag
        from forma.learning_path import generate_learning_path

        dag = build_and_validate_dag([])
        student_scores = {"X": 0.1}
        path = generate_learning_path("s001", student_scores, dag, threshold=0.4)
        assert path.ordered_path == []

    def test_deficit_concepts_list(self):
        """deficit_concepts contains all concepts below threshold."""
        from forma.learning_path import generate_learning_path

        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
                {"prerequisite": "B", "dependent": "C"},
            ]
        )
        student_scores = {"A": 0.8, "B": 0.2, "C": 0.1}
        path = generate_learning_path("s001", student_scores, dag, threshold=0.4)
        assert set(path.deficit_concepts) == {"B", "C"}


# ---------------------------------------------------------------------------
# T037: LearningPath dataclass tests
# ---------------------------------------------------------------------------


class TestLearningPathDataclass:
    """Tests for LearningPath dataclass."""

    def test_learning_path_creation(self):
        from forma.learning_path import LearningPath

        lp = LearningPath(
            student_id="s001",
            deficit_concepts=["B", "C"],
            ordered_path=["B", "C"],
            capped=False,
        )
        assert lp.student_id == "s001"
        assert lp.deficit_concepts == ["B", "C"]
        assert lp.ordered_path == ["B", "C"]
        assert lp.capped is False


# ---------------------------------------------------------------------------
# T038: build_class_deficit_map tests (FR-021)
# ---------------------------------------------------------------------------


class TestBuildClassDeficitMap:
    """Tests for build_class_deficit_map()."""

    def _make_dag(self, dep_dicts):
        from forma.concept_dependency import ConceptDependency, build_and_validate_dag

        deps = [ConceptDependency(prerequisite=d["prerequisite"], dependent=d["dependent"]) for d in dep_dicts]
        return build_and_validate_dag(deps)

    def test_basic_deficit_map(self):
        """Counts students with deficit per concept."""
        from forma.learning_path import build_class_deficit_map

        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
                {"prerequisite": "B", "dependent": "C"},
            ]
        )
        all_scores = {
            "s001": {"A": 0.8, "B": 0.2, "C": 0.1},
            "s002": {"A": 0.9, "B": 0.5, "C": 0.3},
            "s003": {"A": 0.1, "B": 0.1, "C": 0.1},
        }
        deficit_map = build_class_deficit_map(all_scores, dag, threshold=0.4)
        assert deficit_map.total_students == 3
        # A: s003 deficit (1)
        assert deficit_map.concept_counts["A"] == 1
        # B: s001, s003 deficit (2)
        assert deficit_map.concept_counts["B"] == 2
        # C: s001, s002, s003 deficit (3)
        assert deficit_map.concept_counts["C"] == 3

    def test_deficit_map_all_mastered(self):
        """All students mastered → all counts 0."""
        from forma.learning_path import build_class_deficit_map

        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
            ]
        )
        all_scores = {
            "s001": {"A": 0.9, "B": 0.8},
        }
        deficit_map = build_class_deficit_map(all_scores, dag, threshold=0.4)
        assert deficit_map.concept_counts.get("A", 0) == 0
        assert deficit_map.concept_counts.get("B", 0) == 0

    def test_deficit_map_dag_reference(self):
        """ClassDeficitMap stores reference to the DAG."""
        from forma.learning_path import build_class_deficit_map

        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
            ]
        )
        all_scores = {"s001": {"A": 0.1, "B": 0.1}}
        deficit_map = build_class_deficit_map(all_scores, dag, threshold=0.4)
        assert deficit_map.dag is dag

    def test_deficit_map_empty_scores(self):
        """Empty student scores → total_students=0, all counts 0."""
        from forma.learning_path import build_class_deficit_map

        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
            ]
        )
        deficit_map = build_class_deficit_map({}, dag, threshold=0.4)
        assert deficit_map.total_students == 0

    def test_deficit_map_missing_concept_in_scores(self):
        """Concept in DAG but not in student scores → counted as deficit."""
        from forma.learning_path import build_class_deficit_map

        dag = self._make_dag(
            [
                {"prerequisite": "A", "dependent": "B"},
            ]
        )
        all_scores = {"s001": {"A": 0.8}}  # B missing
        deficit_map = build_class_deficit_map(all_scores, dag, threshold=0.4)
        assert deficit_map.concept_counts["B"] == 1


# ---------------------------------------------------------------------------
# T038: ClassDeficitMap dataclass tests
# ---------------------------------------------------------------------------


class TestClassDeficitMapDataclass:
    """Tests for ClassDeficitMap dataclass."""

    def test_class_deficit_map_creation(self):
        from forma.concept_dependency import build_and_validate_dag
        from forma.learning_path import ClassDeficitMap

        dag = build_and_validate_dag([])
        dm = ClassDeficitMap(
            concept_counts={"A": 5, "B": 3},
            total_students=10,
            dag=dag,
        )
        assert dm.concept_counts == {"A": 5, "B": 3}
        assert dm.total_students == 10
