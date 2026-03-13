"""Tests for forma.lecture_comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from forma.lecture_analyzer import AnalysisResult, ConceptCoverage


def _make_result(
    class_id: str = "A",
    week: int = 1,
    keyword_frequencies: dict[str, int] | None = None,
    top_keywords: list[str] | None = None,
    emphasis_scores: dict[str, float] | None = None,
    concept_coverage: ConceptCoverage | None = None,
) -> AnalysisResult:
    """Create a minimal AnalysisResult for testing."""
    kw_freq = keyword_frequencies or {}
    top_kw = top_keywords or list(kw_freq.keys())
    return AnalysisResult(
        class_id=class_id,
        week=week,
        keyword_frequencies=kw_freq,
        top_keywords=top_kw,
        network_image_path=None,
        topics=None,
        topic_skipped_reason="test",
        concept_coverage=concept_coverage,
        emphasis_scores=emphasis_scores,
        triplets=None,
        triplet_skipped_reason=None,
        sentence_count=10,
        analysis_timestamp="2026-01-01T00:00:00",
    )


class TestEmphasisVarianceEntry:
    def test_creation(self) -> None:
        """EmphasisVarianceEntry stores concept, variance, per_section_scores."""
        from forma.lecture_comparison import EmphasisVarianceEntry

        entry = EmphasisVarianceEntry(
            concept="세포",
            variance=0.25,
            per_section_scores={"A": 0.8, "B": 0.3},
        )
        assert entry.concept == "세포"
        assert entry.variance == 0.25
        assert entry.per_section_scores == {"A": 0.8, "B": 0.3}


class TestComparisonResult:
    def test_creation(self) -> None:
        """ComparisonResult stores all comparison data."""
        from forma.lecture_comparison import ComparisonResult, EmphasisVarianceEntry

        result = ComparisonResult(
            comparison_type="session",
            sections_compared=["A", "B"],
            exclusive_keywords={"A": ["ATP"], "B": ["DNA"]},
            concept_gaps={"A": ["세포"], "B": []},
            emphasis_variance=[
                EmphasisVarianceEntry(
                    concept="세포", variance=0.5, per_section_scores={"A": 1.0, "B": 0.0},
                ),
            ],
            comparison_timestamp="2026-01-01T00:00:00",
        )
        assert result.comparison_type == "session"
        assert result.sections_compared == ["A", "B"]
        assert result.exclusive_keywords["A"] == ["ATP"]
        assert result.concept_gaps is not None
        assert len(result.emphasis_variance) == 1


class TestCompareSections:
    def test_compare_basic_two_sections(self) -> None:
        """Two sections with different top keywords produce exclusive keywords."""
        from forma.lecture_comparison import compare_sections

        result_a = _make_result(
            class_id="A",
            keyword_frequencies={"세포": 10, "ATP": 8, "단백질": 6, "공통": 5},
            top_keywords=["세포", "ATP", "단백질", "공통"],
        )
        result_b = _make_result(
            class_id="B",
            keyword_frequencies={"세포": 9, "DNA": 7, "RNA": 5, "공통": 4},
            top_keywords=["세포", "DNA", "RNA", "공통"],
        )

        comparison = compare_sections({"A": result_a, "B": result_b}, top_n=4)

        # ATP, 단백질 exclusive to A; DNA, RNA exclusive to B
        assert "ATP" in comparison.exclusive_keywords["A"]
        assert "단백질" in comparison.exclusive_keywords["A"]
        assert "DNA" in comparison.exclusive_keywords["B"]
        assert "RNA" in comparison.exclusive_keywords["B"]
        # 세포, 공통 are shared — not exclusive
        assert "세포" not in comparison.exclusive_keywords["A"]
        assert "세포" not in comparison.exclusive_keywords["B"]
        assert "공통" not in comparison.exclusive_keywords["A"]

    def test_compare_exclusive_top_n(self) -> None:
        """Exclusive means in section's top-N but absent from ALL other sections' top-N (FR-017)."""
        from forma.lecture_comparison import compare_sections

        # Section A has "미토콘드리아" in top-3
        # Section B has "미토콘드리아" at low frequency but not in top-3
        result_a = _make_result(
            class_id="A",
            keyword_frequencies={"세포": 10, "미토콘드리아": 8, "ATP": 6},
            top_keywords=["세포", "미토콘드리아", "ATP"],
        )
        result_b = _make_result(
            class_id="B",
            keyword_frequencies={"세포": 9, "DNA": 7, "RNA": 5, "미토콘드리아": 1},
            top_keywords=["세포", "DNA", "RNA"],  # 미토콘드리아 not in top-3
        )

        comparison = compare_sections({"A": result_a, "B": result_b}, top_n=3)

        # 미토콘드리아 is in A's top-3 but NOT in B's top-3 → exclusive to A
        assert "미토콘드리아" in comparison.exclusive_keywords["A"]

    def test_compare_concept_gaps_with_concepts(self) -> None:
        """When concepts provided, identify which sections miss which concepts (FR-018)."""
        from forma.lecture_comparison import compare_sections

        coverage_a = ConceptCoverage(
            total_concepts=3,
            covered_concepts=["세포", "ATP"],
            missed_concepts=["DNA"],
            coverage_ratio=2 / 3,
        )
        coverage_b = ConceptCoverage(
            total_concepts=3,
            covered_concepts=["세포", "DNA"],
            missed_concepts=["ATP"],
            coverage_ratio=2 / 3,
        )

        result_a = _make_result(class_id="A", concept_coverage=coverage_a)
        result_b = _make_result(class_id="B", concept_coverage=coverage_b)

        comparison = compare_sections(
            {"A": result_a, "B": result_b},
            concepts=["세포", "ATP", "DNA"],
        )

        assert comparison.concept_gaps is not None
        assert "DNA" in comparison.concept_gaps["A"]
        assert "ATP" in comparison.concept_gaps["B"]
        assert "세포" not in comparison.concept_gaps["A"]
        assert "세포" not in comparison.concept_gaps["B"]

    def test_compare_concept_gaps_without_concepts(self) -> None:
        """Without concepts, concept_gaps is None."""
        from forma.lecture_comparison import compare_sections

        result_a = _make_result(class_id="A")
        result_b = _make_result(class_id="B")

        comparison = compare_sections({"A": result_a, "B": result_b})
        assert comparison.concept_gaps is None

    def test_compare_emphasis_variance(self) -> None:
        """Concepts ranked by emphasis stdev descending (FR-019)."""
        from forma.lecture_comparison import compare_sections

        result_a = _make_result(
            class_id="A",
            emphasis_scores={"세포": 1.0, "ATP": 0.5, "DNA": 0.2},
        )
        result_b = _make_result(
            class_id="B",
            emphasis_scores={"세포": 0.0, "ATP": 0.5, "DNA": 0.2},
        )

        comparison = compare_sections({"A": result_a, "B": result_b})

        assert len(comparison.emphasis_variance) > 0
        # 세포 has the largest variance (1.0 vs 0.0 → stdev=0.5)
        assert comparison.emphasis_variance[0].concept == "세포"
        # ATP has stdev 0 (0.5 vs 0.5) → should be ranked lower
        atp_entries = [e for e in comparison.emphasis_variance if e.concept == "ATP"]
        assert len(atp_entries) == 1
        assert atp_entries[0].variance == pytest.approx(0.0)
        # Ensure descending order
        for i in range(len(comparison.emphasis_variance) - 1):
            assert comparison.emphasis_variance[i].variance >= comparison.emphasis_variance[i + 1].variance

    def test_compare_fewer_than_two_raises(self) -> None:
        """Fewer than 2 sections raises ValueError (FR-016)."""
        from forma.lecture_comparison import compare_sections

        result_a = _make_result(class_id="A")
        with pytest.raises(ValueError, match="최소 2개 반"):
            compare_sections({"A": result_a})

    def test_compare_four_sections(self) -> None:
        """Four sections produce correct exclusive keywords for each."""
        from forma.lecture_comparison import compare_sections

        analyses = {}
        for cid, unique_kw in [("A", "ATP"), ("B", "DNA"), ("C", "RNA"), ("D", "GFR")]:
            analyses[cid] = _make_result(
                class_id=cid,
                keyword_frequencies={"세포": 10, unique_kw: 5},
                top_keywords=["세포", unique_kw],
            )

        comparison = compare_sections(analyses, top_n=10)

        assert len(comparison.sections_compared) == 4
        for cid, unique_kw in [("A", "ATP"), ("B", "DNA"), ("C", "RNA"), ("D", "GFR")]:
            assert unique_kw in comparison.exclusive_keywords[cid]
        # 세포 is shared across all — never exclusive
        for cid in "ABCD":
            assert "세포" not in comparison.exclusive_keywords[cid]


class TestComparisonCaching:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        """ComparisonResult survives YAML serialization."""
        from forma.lecture_comparison import (
            ComparisonResult,
            EmphasisVarianceEntry,
            save_comparison_result,
            load_comparison_result,
        )

        original = ComparisonResult(
            comparison_type="session",
            sections_compared=["A", "B"],
            exclusive_keywords={"A": ["ATP", "단백질"], "B": ["DNA"]},
            concept_gaps={"A": ["DNA"], "B": ["ATP"]},
            emphasis_variance=[
                EmphasisVarianceEntry(
                    concept="세포",
                    variance=0.5,
                    per_section_scores={"A": 1.0, "B": 0.0},
                ),
            ],
            comparison_timestamp="2026-01-01T00:00:00+00:00",
        )

        saved_path = save_comparison_result(original, tmp_path, prefix="test")
        assert saved_path.exists()

        loaded = load_comparison_result(saved_path)
        assert loaded.comparison_type == original.comparison_type
        assert loaded.sections_compared == original.sections_compared
        assert loaded.exclusive_keywords == original.exclusive_keywords
        assert loaded.concept_gaps == original.concept_gaps
        assert len(loaded.emphasis_variance) == 1
        assert loaded.emphasis_variance[0].concept == "세포"
        assert loaded.emphasis_variance[0].variance == pytest.approx(0.5)
        assert loaded.emphasis_variance[0].per_section_scores == {"A": 1.0, "B": 0.0}
        assert loaded.comparison_timestamp == original.comparison_timestamp
