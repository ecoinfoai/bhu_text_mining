"""Tests for learning_path_charts module — DAG visualization + deficit map charts.

Phase 6 (T039): RED phase tests for US4 charts.
Covers FR-020, FR-021.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from matplotlib.font_manager import FontProperties as _RealFontProperties

PNG_HEADER = b"\x89PNG"


def _mock_font_properties_factory(fname=None):
    """Return a real FontProperties without loading a .ttf file."""
    return _RealFontProperties()


@pytest.fixture()
def _mock_font():
    """Mock Korean font discovery so tests don't need real font files."""
    with (
        patch("forma.learning_path_charts.find_korean_font", return_value="/fake/font.ttf"),
        patch(
            "forma.learning_path_charts.FontProperties",
            side_effect=_mock_font_properties_factory,
        ),
    ):
        yield


def _make_dag(dep_dicts):
    """Helper: build a ConceptDependencyDAG from list of dicts."""
    from forma.concept_dependency import ConceptDependency, build_and_validate_dag

    deps = [
        ConceptDependency(prerequisite=d["prerequisite"], dependent=d["dependent"])
        for d in dep_dicts
    ]
    return build_and_validate_dag(deps)


# ---------------------------------------------------------------------------
# T039: build_learning_path_chart tests (FR-020)
# ---------------------------------------------------------------------------


class TestBuildLearningPathChart:
    """Tests for build_learning_path_chart() — student DAG visualization."""

    @pytest.mark.usefixtures("_mock_font")
    def test_returns_bytesio_png(self):
        """Chart returns BytesIO with PNG header."""
        from forma.learning_path import LearningPath
        from forma.learning_path_charts import build_learning_path_chart

        dag = _make_dag([
            {"prerequisite": "A", "dependent": "B"},
            {"prerequisite": "B", "dependent": "C"},
        ])
        lp = LearningPath(
            student_id="s001",
            deficit_concepts=["B", "C"],
            ordered_path=["B", "C"],
            capped=False,
        )
        buf = build_learning_path_chart(lp, dag)
        assert isinstance(buf, io.BytesIO)
        assert buf.read(4) == PNG_HEADER

    @pytest.mark.usefixtures("_mock_font")
    def test_empty_path_returns_png(self):
        """Chart is generated even for empty path (all mastered)."""
        from forma.learning_path import LearningPath
        from forma.learning_path_charts import build_learning_path_chart

        dag = _make_dag([
            {"prerequisite": "A", "dependent": "B"},
        ])
        lp = LearningPath(
            student_id="s001",
            deficit_concepts=[],
            ordered_path=[],
            capped=False,
        )
        buf = build_learning_path_chart(lp, dag)
        assert isinstance(buf, io.BytesIO)
        assert buf.read(4) == PNG_HEADER

    @pytest.mark.usefixtures("_mock_font")
    def test_korean_concepts_in_chart(self):
        """Chart handles Korean concept names without error."""
        from forma.learning_path import LearningPath
        from forma.learning_path_charts import build_learning_path_chart

        dag = _make_dag([
            {"prerequisite": "세포막 구조", "dependent": "물질 이동"},
        ])
        lp = LearningPath(
            student_id="s001",
            deficit_concepts=["물질 이동"],
            ordered_path=["물질 이동"],
            capped=False,
        )
        buf = build_learning_path_chart(lp, dag)
        assert isinstance(buf, io.BytesIO)
        assert buf.read(4) == PNG_HEADER


# ---------------------------------------------------------------------------
# T039: build_deficit_map_chart tests (FR-021)
# ---------------------------------------------------------------------------


class TestBuildDeficitMapChart:
    """Tests for build_deficit_map_chart() — class-wide deficit visualization."""

    @pytest.mark.usefixtures("_mock_font")
    def test_returns_bytesio_png(self):
        """Chart returns BytesIO with PNG header."""
        from forma.learning_path import ClassDeficitMap
        from forma.learning_path_charts import build_deficit_map_chart

        dag = _make_dag([
            {"prerequisite": "A", "dependent": "B"},
            {"prerequisite": "B", "dependent": "C"},
        ])
        deficit_map = ClassDeficitMap(
            concept_counts={"A": 2, "B": 5, "C": 8},
            total_students=10,
            dag=dag,
        )
        buf = build_deficit_map_chart(deficit_map)
        assert isinstance(buf, io.BytesIO)
        assert buf.read(4) == PNG_HEADER

    @pytest.mark.usefixtures("_mock_font")
    def test_empty_counts_returns_png(self):
        """Chart handles all-zero counts."""
        from forma.learning_path import ClassDeficitMap
        from forma.learning_path_charts import build_deficit_map_chart

        dag = _make_dag([
            {"prerequisite": "A", "dependent": "B"},
        ])
        deficit_map = ClassDeficitMap(
            concept_counts={"A": 0, "B": 0},
            total_students=10,
            dag=dag,
        )
        buf = build_deficit_map_chart(deficit_map)
        assert isinstance(buf, io.BytesIO)
        assert buf.read(4) == PNG_HEADER

    @pytest.mark.usefixtures("_mock_font")
    def test_large_dag_chart(self):
        """Chart handles 20+ nodes without crashing."""
        from forma.learning_path import ClassDeficitMap
        from forma.learning_path_charts import build_deficit_map_chart

        dag = _make_dag([
            {"prerequisite": f"C{i}", "dependent": f"C{i+1}"}
            for i in range(25)
        ])
        counts = {f"C{i}": i for i in range(26)}
        deficit_map = ClassDeficitMap(
            concept_counts=counts,
            total_students=30,
            dag=dag,
        )
        buf = build_deficit_map_chart(deficit_map)
        assert isinstance(buf, io.BytesIO)
        assert buf.read(4) == PNG_HEADER

    @pytest.mark.usefixtures("_mock_font")
    def test_single_node_dag_chart(self):
        """Chart handles DAG with 2 nodes (minimal)."""
        from forma.learning_path import ClassDeficitMap
        from forma.learning_path_charts import build_deficit_map_chart

        dag = _make_dag([
            {"prerequisite": "X", "dependent": "Y"},
        ])
        deficit_map = ClassDeficitMap(
            concept_counts={"X": 1, "Y": 3},
            total_students=5,
            dag=dag,
        )
        buf = build_deficit_map_chart(deficit_map)
        assert isinstance(buf, io.BytesIO)
        assert buf.read(4) == PNG_HEADER
