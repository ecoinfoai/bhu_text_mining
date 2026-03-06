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
