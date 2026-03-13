"""Tests for forma.lecture_merge."""

from __future__ import annotations

from pathlib import Path

import pytest

from forma.lecture_analyzer import AnalysisResult
from forma.lecture_merge import (
    MergedAnalysis,
    merge_analyses,
    save_merged_analysis,
    load_merged_analysis,
)


def _make_analysis(
    class_id: str = "A",
    week: int = 1,
    keyword_frequencies: dict[str, int] | None = None,
) -> AnalysisResult:
    """Create a minimal AnalysisResult for testing."""
    if keyword_frequencies is None:
        keyword_frequencies = {"세포": 20, "ATP": 10}
    return AnalysisResult(
        class_id=class_id,
        week=week,
        keyword_frequencies=keyword_frequencies,
        top_keywords=sorted(keyword_frequencies, key=keyword_frequencies.get, reverse=True),
        network_image_path=None,
        topics=None,
        topic_skipped_reason="테스트",
        concept_coverage=None,
        emphasis_scores=None,
        triplets=None,
        triplet_skipped_reason=None,
        sentence_count=5,
        analysis_timestamp="2026-01-01T00:00:00",
    )


class TestMergedAnalysis:
    """Test MergedAnalysis dataclass."""

    def test_creation(self) -> None:
        """MergedAnalysis dataclass creates with required fields."""
        ma = MergedAnalysis(
            class_id="A",
            weeks=[1, 2],
            combined_keyword_frequencies={"세포": 42, "ATP": 25},
            per_session_keyword_frequencies={
                1: {"세포": 20, "ATP": 10},
                2: {"세포": 22, "ATP": 15},
            },
            session_boundary_markers=[
                "--- Session 1 (Week 1) ---",
                "--- Session 2 (Week 2) ---",
            ],
            merged_text="session 1 text. --- Session 2 (Week 2) --- session 2 text.",
        )
        assert ma.class_id == "A"
        assert ma.weeks == [1, 2]
        assert ma.combined_keyword_frequencies["세포"] == 42
        assert len(ma.per_session_keyword_frequencies) == 2
        assert len(ma.session_boundary_markers) == 2


class TestMergeAnalyses:
    """Test merge_analyses() function."""

    def test_merge_two_sessions(self) -> None:
        """Two AnalysisResults merge keyword frequencies by summing."""
        a1 = _make_analysis(week=1, keyword_frequencies={"세포": 20, "ATP": 10})
        a2 = _make_analysis(week=2, keyword_frequencies={"세포": 22, "DNA": 15})
        result = merge_analyses([a1, a2], class_id="A")

        assert result.combined_keyword_frequencies["세포"] == 42
        assert result.combined_keyword_frequencies["ATP"] == 10
        assert result.combined_keyword_frequencies["DNA"] == 15

    def test_merge_preserves_per_session(self) -> None:
        """Per-session keyword frequencies preserved (FR-022)."""
        a1 = _make_analysis(week=1, keyword_frequencies={"세포": 20, "ATP": 10})
        a2 = _make_analysis(week=2, keyword_frequencies={"세포": 22, "DNA": 15})
        result = merge_analyses([a1, a2], class_id="A")

        assert result.per_session_keyword_frequencies[1] == {"세포": 20, "ATP": 10}
        assert result.per_session_keyword_frequencies[2] == {"세포": 22, "DNA": 15}

    def test_merge_session_boundary_markers(self) -> None:
        """Session boundary markers present in merged output."""
        a1 = _make_analysis(week=1)
        a2 = _make_analysis(week=2)
        result = merge_analyses([a1, a2], class_id="A")

        assert len(result.session_boundary_markers) == 2
        assert "Week 1" in result.session_boundary_markers[0]
        assert "Week 2" in result.session_boundary_markers[1]

    def test_merge_session_order(self) -> None:
        """Sessions sorted by week number regardless of input order."""
        a3 = _make_analysis(week=3, keyword_frequencies={"DNA": 5})
        a1 = _make_analysis(week=1, keyword_frequencies={"세포": 10})
        result = merge_analyses([a3, a1], class_id="A")

        assert result.weeks == [1, 3]
        # Boundary markers should be in week order
        assert "Week 1" in result.session_boundary_markers[0]
        assert "Week 3" in result.session_boundary_markers[1]

    def test_merge_single_session(self) -> None:
        """Single-session input returns valid MergedAnalysis."""
        a1 = _make_analysis(week=1, keyword_frequencies={"세포": 20})
        result = merge_analyses([a1], class_id="A")

        assert result.weeks == [1]
        assert result.combined_keyword_frequencies == {"세포": 20}
        assert len(result.session_boundary_markers) == 1

    def test_merge_empty_raises(self) -> None:
        """Empty analyses list raises ValueError."""
        with pytest.raises(ValueError, match="병합할 분석 결과가 없습니다"):
            merge_analyses([], class_id="A")


class TestMergedCaching:
    """Test YAML serialization round-trip."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        """MergedAnalysis survives YAML serialization."""
        ma = MergedAnalysis(
            class_id="A",
            weeks=[1, 2],
            combined_keyword_frequencies={"세포": 42, "ATP": 25},
            per_session_keyword_frequencies={
                1: {"세포": 20, "ATP": 10},
                2: {"세포": 22, "ATP": 15},
            },
            session_boundary_markers=[
                "--- Session 1 (Week 1) ---",
                "--- Session 2 (Week 2) ---",
            ],
            merged_text="session 1 text.\n--- Session 2 (Week 2) ---\nsession 2 text.",
        )
        path = save_merged_analysis(ma, tmp_path)
        assert path.exists()

        loaded = load_merged_analysis(path)
        assert loaded.class_id == ma.class_id
        assert loaded.weeks == ma.weeks
        assert loaded.combined_keyword_frequencies == ma.combined_keyword_frequencies
        assert loaded.per_session_keyword_frequencies == ma.per_session_keyword_frequencies
        assert loaded.session_boundary_markers == ma.session_boundary_markers
        assert loaded.merged_text == ma.merged_text

    def test_load_missing_file(self) -> None:
        """Loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="병합 분석 결과 파일"):
            load_merged_analysis(Path("/nonexistent/path.yaml"))

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """save_merged_analysis creates output directory if missing."""
        ma = MergedAnalysis(
            class_id="B",
            weeks=[1],
            combined_keyword_frequencies={"DNA": 5},
            per_session_keyword_frequencies={1: {"DNA": 5}},
            session_boundary_markers=["--- Session 1 (Week 1) ---"],
            merged_text="--- Session 1 (Week 1) ---",
        )
        nested = tmp_path / "deep" / "nested"
        path = save_merged_analysis(ma, nested)
        assert path.exists()
        assert nested.is_dir()
