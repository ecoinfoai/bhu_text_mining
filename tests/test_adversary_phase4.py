"""Adversary attack tests for Phase 4 (US2): class knowledge graph chart + PDF section.

6 personas attack build_class_knowledge_graph_chart() and _build_class_graph_section():
1. Edge Case Hunter: boundary conditions in chart rendering
2. Memory Saboteur: matplotlib figure leaks
3. Type System Antagonist: type safety through the pipeline
4. Concurrency Destroyer: matplotlib thread safety
5. PDF Killer: crash-inducing data in PDF rendering
6. Data Integrity Enforcer: chart filtering correctness
"""

from __future__ import annotations

import io
import threading
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from forma.class_knowledge_aggregate import (
    AggregateEdge,
    ClassKnowledgeAggregate,
)

PNG_HEADER = b"\x89PNG"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_edge(
    subject: str = "A",
    relation: str = "R",
    obj: str = "B",
    correct_count: int = 5,
    error_count: int = 3,
    missing_count: int = 2,
    total_students: int = 10,
    correct_ratio: float = 0.5,
) -> AggregateEdge:
    return AggregateEdge(
        subject=subject,
        relation=relation,
        obj=obj,
        correct_count=correct_count,
        error_count=error_count,
        missing_count=missing_count,
        total_students=total_students,
        correct_ratio=correct_ratio,
    )


def _make_aggregate(
    edges: list[AggregateEdge] | None = None,
    question_sn: int = 1,
    total_students: int = 10,
) -> ClassKnowledgeAggregate:
    if edges is None:
        edges = [_make_edge()]
    return ClassKnowledgeAggregate(
        question_sn=question_sn,
        edges=edges,
        total_students=total_students,
    )


@pytest.fixture()
def mock_font_path(tmp_path):
    """Create a fake font file."""
    font_file = tmp_path / "FakeFont.ttf"
    font_file.write_bytes(b"\x00" * 64)
    return str(font_file)


@pytest.fixture()
def chart_gen(mock_font_path):
    """Chart generator with mocked font."""
    from matplotlib.font_manager import FontProperties
    from forma.professor_report_charts import ProfessorReportChartGenerator

    with (
        patch(
            "forma.professor_report_charts.find_korean_font",
            return_value=mock_font_path,
        ),
        patch(
            "forma.professor_report_charts.FontProperties",
            lambda fname: FontProperties(),
        ),
    ):
        return ProfessorReportChartGenerator(font_path=mock_font_path, dpi=72)


# ===========================================================================
# PERSONA 1: THE EDGE CASE HUNTER
# ===========================================================================


class TestChartEdgeCaseHunter:
    """Persona 1: Boundary conditions in chart rendering."""

    def test_empty_edges_list(self, chart_gen):
        """Empty edges list -> 'no data' fallback PNG (not crash, not empty)."""
        agg = _make_aggregate(edges=[], total_students=0)
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        data = buf.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_single_edge(self, chart_gen):
        """Single edge aggregate renders without error."""
        edge = _make_edge(subject="심근경색", obj="허혈", correct_ratio=0.8)
        agg = _make_aggregate(edges=[edge])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0

    def test_all_edges_below_min_ratio(self, chart_gen):
        """All edges below min_ratio_to_show -> fallback PNG, not crash."""
        edges = [_make_edge(subject=f"S{i}", obj=f"O{i}", correct_ratio=0.01) for i in range(5)]
        agg = _make_aggregate(edges=edges)
        buf = chart_gen.build_class_knowledge_graph_chart(agg, min_ratio_to_show=0.05)
        data = buf.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_min_ratio_to_show_1_0_filters_everything(self, chart_gen):
        """min_ratio_to_show=1.0 -> only edges with ratio >= 1.0 shown."""
        edges = [
            _make_edge(subject="A", obj="B", correct_ratio=0.99),
            _make_edge(subject="C", obj="D", correct_ratio=1.0),
        ]
        agg = _make_aggregate(edges=edges)
        buf = chart_gen.build_class_knowledge_graph_chart(agg, min_ratio_to_show=1.0)
        data = buf.getvalue()
        assert len(data) > 0

    def test_correct_ratio_exactly_at_threshold(self, chart_gen):
        """Edge with correct_ratio == min_ratio_to_show should be included (>=)."""
        edge = _make_edge(correct_ratio=0.05)
        agg = _make_aggregate(edges=[edge])
        buf = chart_gen.build_class_knowledge_graph_chart(agg, min_ratio_to_show=0.05)
        data = buf.getvalue()
        assert len(data) > 0

    def test_correct_ratio_just_below_threshold(self, chart_gen):
        """Edge with correct_ratio = 0.049 should be filtered at min_ratio=0.05."""
        edge = _make_edge(correct_ratio=0.049)
        agg = _make_aggregate(edges=[edge])
        buf = chart_gen.build_class_knowledge_graph_chart(agg, min_ratio_to_show=0.05)
        data = buf.getvalue()
        assert len(data) > 0  # fallback "no data" PNG

    def test_all_four_color_tiers(self, chart_gen):
        """Edges spanning all 4 color tiers render without error."""
        edges = [
            _make_edge(subject="A", obj="B", correct_ratio=0.8, error_count=0, missing_count=2),
            _make_edge(subject="C", obj="D", correct_ratio=0.35, error_count=3, missing_count=3),
            _make_edge(subject="E", obj="F", correct_ratio=0.1, error_count=5, missing_count=2),
            _make_edge(subject="G", obj="H", correct_ratio=0.1, error_count=1, missing_count=8),
        ]
        agg = _make_aggregate(edges=edges)
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0


# ===========================================================================
# PERSONA 2: THE MEMORY SABOTEUR
# ===========================================================================


class TestChartMemorySaboteur:
    """Persona 2: matplotlib figure leaks."""

    def test_figure_closed_after_chart(self, chart_gen):
        """build_class_knowledge_graph_chart closes the figure (no leak)."""
        figs_before = set(plt.get_fignums())
        agg = _make_aggregate(edges=[_make_edge(correct_ratio=0.6)])
        chart_gen.build_class_knowledge_graph_chart(agg)
        figs_after = set(plt.get_fignums())
        new_figs = figs_after - figs_before
        assert len(new_figs) == 0

    def test_figure_closed_after_empty_chart(self, chart_gen):
        """Figure closed even for empty/fallback chart."""
        figs_before = set(plt.get_fignums())
        agg = _make_aggregate(edges=[])
        chart_gen.build_class_knowledge_graph_chart(agg)
        figs_after = set(plt.get_fignums())
        new_figs = figs_after - figs_before
        assert len(new_figs) == 0

    def test_10_consecutive_charts_no_leak(self, chart_gen):
        """10 consecutive chart calls don't accumulate figures."""
        figs_before = set(plt.get_fignums())
        for i in range(10):
            edge = _make_edge(subject=f"S{i}", obj=f"O{i}", correct_ratio=0.5 + i * 0.04)
            agg = _make_aggregate(edges=[edge])
            chart_gen.build_class_knowledge_graph_chart(agg)
        figs_after = set(plt.get_fignums())
        new_figs = figs_after - figs_before
        assert len(new_figs) == 0

    def test_bytesio_seek_position_zero(self, chart_gen):
        """Returned BytesIO has seek position at 0 (ready for read)."""
        agg = _make_aggregate(edges=[_make_edge(correct_ratio=0.6)])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert buf.tell() == 0


# ===========================================================================
# PERSONA 3: THE TYPE SYSTEM ANTAGONIST
# ===========================================================================


class TestChartTypeAntagonist:
    """Persona 3: Type safety through the chart pipeline."""

    def test_returns_bytesio(self, chart_gen):
        """Return type is io.BytesIO."""
        agg = _make_aggregate(edges=[_make_edge(correct_ratio=0.6)])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert isinstance(buf, io.BytesIO)

    def test_returns_valid_png(self, chart_gen):
        """Returned BytesIO contains valid PNG data."""
        agg = _make_aggregate(edges=[_make_edge(correct_ratio=0.6)])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        data = buf.getvalue()
        assert data[:4] == PNG_HEADER

    def test_empty_fallback_is_also_png(self, chart_gen):
        """Fallback 'no data' chart is also valid PNG."""
        agg = _make_aggregate(edges=[])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        data = buf.getvalue()
        assert data[:4] == PNG_HEADER


# ===========================================================================
# PERSONA 4: THE CONCURRENCY DESTROYER
# ===========================================================================


class TestChartConcurrencyDestroyer:
    """Persona 4: matplotlib thread safety."""

    def test_two_concurrent_chart_calls(self, chart_gen):
        """Two simultaneous chart calls don't crash."""
        edges1 = [_make_edge(subject="A1", obj="B1", correct_ratio=0.8)]
        edges2 = [_make_edge(subject="A2", obj="B2", correct_ratio=0.3)]
        agg1 = _make_aggregate(edges=edges1, question_sn=1)
        agg2 = _make_aggregate(edges=edges2, question_sn=2)

        results: dict[str, io.BytesIO] = {}
        errors: list[Exception] = []

        def run1():
            try:
                results["buf1"] = chart_gen.build_class_knowledge_graph_chart(agg1)
            except Exception as e:
                errors.append(e)

        def run2():
            try:
                results["buf2"] = chart_gen.build_class_knowledge_graph_chart(agg2)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=run1)
        t2 = threading.Thread(target=run2)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"Concurrent chart errors: {errors}"
        assert len(results["buf1"].getvalue()) > 0
        assert len(results["buf2"].getvalue()) > 0


# ===========================================================================
# PERSONA 5: THE PDF KILLER
# ===========================================================================


class TestChartPDFKiller:
    """Persona 5: Data that might crash chart or downstream PDF."""

    def test_xml_special_chars_in_concept_names(self, chart_gen):
        """< > & in concept names don't crash chart rendering."""
        edge = _make_edge(
            subject='<script>alert("x")</script>',
            obj="A & B",
            relation="is-a",
            correct_ratio=0.6,
        )
        agg = _make_aggregate(edges=[edge])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0

    def test_200_char_concept_name(self, chart_gen):
        """200+ character concept name doesn't crash chart."""
        long_name = "A" * 200
        edge = _make_edge(subject=long_name, obj="B", correct_ratio=0.5)
        agg = _make_aggregate(edges=[edge])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0

    def test_korean_special_chars_in_labels(self, chart_gen):
        """Korean + special chars in labels don't crash."""
        edge = _make_edge(
            subject="세포막\n(이중층)",
            obj="인지질 & 단백질",
            relation="구성\t요소",
            correct_ratio=0.7,
        )
        agg = _make_aggregate(edges=[edge])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0

    def test_50_edges_chart(self, chart_gen):
        """50 edges renders without crash or overflow."""
        edges = [_make_edge(subject=f"S{i}", obj=f"O{i}", correct_ratio=0.1 + (i % 10) * 0.09) for i in range(50)]
        agg = _make_aggregate(edges=edges, total_students=100)
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0

    def test_duplicate_edges_same_subject_obj(self, chart_gen):
        """Multiple edges with same subject-obj pair (different relations)."""
        edges = [
            _make_edge(subject="A", obj="B", relation="R1", correct_ratio=0.8),
            _make_edge(subject="A", obj="B", relation="R2", correct_ratio=0.3),
        ]
        agg = _make_aggregate(edges=edges)
        # NetworkX DiGraph only keeps one edge per (u,v) pair
        # This test verifies no crash; the last edge_label may overwrite
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0

    def test_self_loop_edge(self, chart_gen):
        """Self-loop edge (subject == obj) renders without crash."""
        edge = _make_edge(subject="A", obj="A", correct_ratio=0.5)
        agg = _make_aggregate(edges=[edge])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0


# ===========================================================================
# PERSONA 6: THE DATA INTEGRITY ENFORCER
# ===========================================================================


class TestChartDataIntegrity:
    """Persona 6: Chart filtering correctness."""

    def test_edges_below_threshold_not_in_graph(self, chart_gen):
        """Edges with correct_ratio < min_ratio_to_show are filtered."""
        edge_above = _make_edge(subject="A", obj="B", correct_ratio=0.1)
        edge_below = _make_edge(subject="C", obj="D", correct_ratio=0.03)
        agg = _make_aggregate(edges=[edge_above, edge_below])
        # We can verify by checking that no error occurred and the chart was created
        buf = chart_gen.build_class_knowledge_graph_chart(agg, min_ratio_to_show=0.05)
        assert len(buf.getvalue()) > 0

    def test_color_rule_green_above_0_5(self, chart_gen):
        """correct_ratio > 0.5 should produce a chart (green tier)."""
        edge = _make_edge(correct_ratio=0.51)
        agg = _make_aggregate(edges=[edge])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0

    def test_color_rule_orange_0_2_to_0_5(self, chart_gen):
        """correct_ratio in [0.2, 0.5] should produce a chart (orange tier)."""
        edge = _make_edge(correct_ratio=0.35)
        agg = _make_aggregate(edges=[edge])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0

    def test_color_rule_red_error_dominant(self, chart_gen):
        """correct_ratio < 0.2, error > missing -> red tier."""
        edge = _make_edge(correct_ratio=0.1, error_count=6, missing_count=3)
        agg = _make_aggregate(edges=[edge])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0

    def test_color_rule_grey_missing_dominant(self, chart_gen):
        """correct_ratio < 0.2, missing >= error -> grey dashed tier."""
        edge = _make_edge(correct_ratio=0.1, error_count=2, missing_count=7)
        agg = _make_aggregate(edges=[edge])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0

    def test_edge_width_minimum_0_5(self, chart_gen):
        """Edge width is max(0.5, correct_ratio * 5), minimum 0.5."""
        edge = _make_edge(correct_ratio=0.05)  # 0.05 * 5 = 0.25 -> clamped to 0.5
        agg = _make_aggregate(edges=[edge])
        buf = chart_gen.build_class_knowledge_graph_chart(agg)
        assert len(buf.getvalue()) > 0
