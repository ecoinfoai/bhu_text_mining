"""Tests for graph_visualizer module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forma.evaluation_types import TripletEdge


@pytest.fixture()
def _mock_font(monkeypatch):
    """Mock find_korean_font and FontProperties to avoid font dependencies."""
    from matplotlib.font_manager import FontProperties

    monkeypatch.setattr(
        "forma.graph_visualizer.find_korean_font",
        lambda: "/fake/NanumGothic.ttf",
    )
    # Use a real FontProperties with the default font so matplotlib can render
    monkeypatch.setattr(
        "forma.graph_visualizer.FontProperties",
        lambda fname: FontProperties(),
    )


@pytest.fixture()
def visualizer(_mock_font):
    from forma.graph_visualizer import GraphVisualizer

    return GraphVisualizer()


@pytest.fixture()
def sample_edges():
    return {
        "master": [
            TripletEdge("수용체", "감지", "자극"),
            TripletEdge("자극", "초과", "한계점"),
        ],
        "student": [
            TripletEdge("수용체", "감지", "자극"),
            TripletEdge("자극", "미달", "한계점"),
        ],
        "matched": [TripletEdge("수용체", "감지", "자극")],
        "missing": [TripletEdge("자극", "초과", "한계점")],
        "extra": [TripletEdge("자극", "미달", "한계점")],
        "wrong_direction": [],
    }


class TestGraphVisualizer:
    """Tests for GraphVisualizer."""

    def test_creates_png_file(self, visualizer, sample_edges, tmp_path):
        """Should create a PNG file at the specified output path."""
        out = str(tmp_path / "graph.png")
        result = visualizer.visualize_comparison(
            master_edges=sample_edges["master"],
            student_edges=sample_edges["student"],
            matched_edges=sample_edges["matched"],
            missing_edges=sample_edges["missing"],
            extra_edges=sample_edges["extra"],
            wrong_direction_edges=sample_edges["wrong_direction"],
            output_path=out,
        )
        assert Path(result).exists()
        assert result.endswith(".png")

    def test_returns_absolute_path(self, visualizer, sample_edges, tmp_path):
        """Should return an absolute path."""
        out = str(tmp_path / "graph.png")
        result = visualizer.visualize_comparison(
            master_edges=sample_edges["master"],
            student_edges=sample_edges["student"],
            matched_edges=sample_edges["matched"],
            missing_edges=sample_edges["missing"],
            extra_edges=sample_edges["extra"],
            wrong_direction_edges=sample_edges["wrong_direction"],
            output_path=out,
        )
        assert Path(result).is_absolute()

    def test_creates_parent_directories(self, visualizer, sample_edges, tmp_path):
        """Should create parent directories if they don't exist."""
        out = str(tmp_path / "nested" / "dir" / "graph.png")
        result = visualizer.visualize_comparison(
            master_edges=sample_edges["master"],
            student_edges=sample_edges["student"],
            matched_edges=sample_edges["matched"],
            missing_edges=sample_edges["missing"],
            extra_edges=sample_edges["extra"],
            wrong_direction_edges=sample_edges["wrong_direction"],
            output_path=out,
        )
        assert Path(result).exists()

    def test_handles_empty_edge_lists(self, visualizer, tmp_path):
        """Should handle all empty edge lists gracefully."""
        out = str(tmp_path / "empty.png")
        result = visualizer.visualize_comparison(
            master_edges=[],
            student_edges=[],
            matched_edges=[],
            missing_edges=[],
            extra_edges=[],
            wrong_direction_edges=[],
            output_path=out,
        )
        assert Path(result).exists()

    def test_works_with_korean_text(self, visualizer, tmp_path):
        """Should handle Korean text in nodes and relations."""
        edges = [TripletEdge("세포막", "조절", "물질이동")]
        out = str(tmp_path / "korean.png")
        result = visualizer.visualize_comparison(
            master_edges=edges,
            student_edges=edges,
            matched_edges=edges,
            missing_edges=[],
            extra_edges=[],
            wrong_direction_edges=[],
            output_path=out,
        )
        assert Path(result).exists()

    def test_wrong_direction_edges_rendered(self, visualizer, tmp_path):
        """Should render wrong direction edges."""
        wrong = [TripletEdge("A", "causes", "B")]
        out = str(tmp_path / "wrong_dir.png")
        result = visualizer.visualize_comparison(
            master_edges=[],
            student_edges=[],
            matched_edges=[],
            missing_edges=[],
            extra_edges=[],
            wrong_direction_edges=wrong,
            output_path=out,
        )
        assert Path(result).exists()

    def test_custom_font_path(self, _mock_font):
        """Should accept a custom font path."""
        from forma.graph_visualizer import GraphVisualizer

        viz = GraphVisualizer(font_path="/custom/font.ttf")
        assert viz._font_path == "/custom/font.ttf"

    def test_title_applied(self, visualizer, sample_edges, tmp_path):
        """Should accept a title parameter without error."""
        out = str(tmp_path / "titled.png")
        result = visualizer.visualize_comparison(
            master_edges=sample_edges["master"],
            student_edges=sample_edges["student"],
            matched_edges=sample_edges["matched"],
            missing_edges=sample_edges["missing"],
            extra_edges=sample_edges["extra"],
            wrong_direction_edges=sample_edges["wrong_direction"],
            output_path=out,
            title="비교 그래프",
        )
        assert Path(result).exists()


class TestVisualizeComparisonToBytesIO:
    """Tests for GraphVisualizer.visualize_comparison_to_bytesio()."""

    def test_returns_bytesio_and_omitted_count(self, visualizer, sample_edges):
        """Should return (BytesIO, int) with valid PNG bytes and omitted_count == 0."""
        import io

        result = visualizer.visualize_comparison_to_bytesio(
            matched=sample_edges["matched"],
            missing=sample_edges["missing"],
            extra=sample_edges["extra"],
            wrong_direction=sample_edges["wrong_direction"],
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        buf, omitted_count = result
        assert isinstance(buf, io.BytesIO)
        assert isinstance(omitted_count, int)
        buf.seek(0)
        assert buf.read(4) == b"\x89PNG"
        assert omitted_count == 0

    def test_edge_capping_returns_correct_omitted_count(self, visualizer):
        """Should return omitted_count == 5 when total edges exceed max_edges=30."""
        # 35 total edges: 10 matched, 10 missing, 10 wrong_direction, 5 extra
        matched = [TripletEdge(f"A{i}", "rel", f"B{i}") for i in range(10)]
        missing = [TripletEdge(f"C{i}", "rel", f"D{i}") for i in range(10)]
        wrong_dir = [TripletEdge(f"E{i}", "rel", f"F{i}") for i in range(10)]
        extra = [TripletEdge(f"G{i}", "rel", f"H{i}") for i in range(5)]

        result = visualizer.visualize_comparison_to_bytesio(
            matched=matched,
            missing=missing,
            extra=extra,
            wrong_direction=wrong_dir,
            max_edges=30,
        )

        buf, omitted_count = result
        assert omitted_count == 5
        buf.seek(0)
        assert buf.read(4) == b"\x89PNG"

    def test_empty_edges_returns_valid_bytesio(self, visualizer):
        """Should return (BytesIO, 0) with valid PNG bytes when all edge lists are empty."""
        import io

        result = visualizer.visualize_comparison_to_bytesio(
            matched=[],
            missing=[],
            extra=[],
            wrong_direction=[],
        )

        assert isinstance(result, tuple)
        buf, omitted_count = result
        assert isinstance(buf, io.BytesIO)
        assert omitted_count == 0
        buf.seek(0)
        assert buf.read(4) == b"\x89PNG"

    def test_capping_priority_drops_extra_first(self, visualizer):
        """Should drop extra edges first: matched=20, missing=5, wrong_dir=5, extra=10, max=30."""
        # total = 40, max = 30, so omitted_count should be 10 (all extra dropped)
        matched = [TripletEdge(f"A{i}", "rel", f"B{i}") for i in range(20)]
        missing = [TripletEdge(f"C{i}", "rel", f"D{i}") for i in range(5)]
        wrong_dir = [TripletEdge(f"E{i}", "rel", f"F{i}") for i in range(5)]
        extra = [TripletEdge(f"G{i}", "rel", f"H{i}") for i in range(10)]

        result = visualizer.visualize_comparison_to_bytesio(
            matched=matched,
            missing=missing,
            extra=extra,
            wrong_direction=wrong_dir,
            max_edges=30,
        )

        buf, omitted_count = result
        assert omitted_count == 10
        buf.seek(0)
        assert buf.read(4) == b"\x89PNG"
