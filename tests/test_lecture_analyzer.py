"""Tests for forma.lecture_analyzer."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from forma.lecture_preprocessor import CleanedTranscript


def _make_cleaned(text: str = "세포막 단백질 인지질 이중층 세포막 구조",
                  class_id: str = "A", week: int = 1) -> CleanedTranscript:
    """Helper to create a CleanedTranscript for testing."""
    return CleanedTranscript(
        class_id=class_id,
        week=week,
        source_path="/tmp/test.txt",
        raw_text=text,
        cleaned_text=text,
        encoding_used="utf-8",
        char_count_raw=len(text),
        char_count_cleaned=len(text),
    )


class TestAnalysisResult:
    """Test AnalysisResult dataclass creation and fields."""

    def test_dataclass_creation_minimal(self) -> None:
        from forma.lecture_analyzer import AnalysisResult

        result = AnalysisResult(
            class_id="A",
            week=1,
            keyword_frequencies={"세포막": 3, "단백질": 2},
            top_keywords=["세포막", "단백질"],
            network_image_path=None,
            topics=None,
            topic_skipped_reason="문장 수 부족",
            concept_coverage=None,
            emphasis_scores=None,
            triplets=None,
            triplet_skipped_reason=None,
            sentence_count=5,
            analysis_timestamp="2026-01-01T00:00:00",
        )
        assert result.class_id == "A"
        assert result.week == 1
        assert result.keyword_frequencies == {"세포막": 3, "단백질": 2}
        assert result.top_keywords == ["세포막", "단백질"]
        assert result.network_image_path is None
        assert result.topics is None
        assert result.topic_skipped_reason == "문장 수 부족"
        assert result.concept_coverage is None
        assert result.emphasis_scores is None
        assert result.sentence_count == 5

    def test_dataclass_with_all_fields(self) -> None:
        from forma.lecture_analyzer import (
            AnalysisResult, TopicSummary, ConceptCoverage,
        )

        topic = TopicSummary(
            topic_id=0,
            keywords=["세포막", "단백질"],
            sentence_count=5,
            representative_sentence="세포막은 인지질 이중층으로 구성된다.",
        )
        coverage = ConceptCoverage(
            total_concepts=3,
            covered_concepts=["세포막", "단백질"],
            missed_concepts=["미토콘드리아"],
            coverage_ratio=2 / 3,
        )
        result = AnalysisResult(
            class_id="B",
            week=2,
            keyword_frequencies={"세포막": 5},
            top_keywords=["세포막"],
            network_image_path=Path("/tmp/net.png"),
            topics=[topic],
            topic_skipped_reason=None,
            concept_coverage=coverage,
            emphasis_scores={"세포막": 0.8, "단백질": 0.5},
            triplets=[{"subject": "세포막", "relation": "구성", "object": "인지질"}],
            triplet_skipped_reason=None,
            sentence_count=15,
            analysis_timestamp="2026-01-01T00:00:00",
        )
        assert result.topics is not None
        assert len(result.topics) == 1
        assert result.concept_coverage is not None
        assert result.concept_coverage.coverage_ratio == pytest.approx(2 / 3)
        assert result.triplets is not None


class TestTopicSummary:
    """Test TopicSummary dataclass."""

    def test_creation(self) -> None:
        from forma.lecture_analyzer import TopicSummary

        ts = TopicSummary(
            topic_id=1,
            keywords=["세포", "핵"],
            sentence_count=3,
            representative_sentence="세포에는 핵이 있다.",
        )
        assert ts.topic_id == 1
        assert ts.keywords == ["세포", "핵"]
        assert ts.sentence_count == 3


class TestConceptCoverage:
    """Test ConceptCoverage dataclass."""

    def test_creation(self) -> None:
        from forma.lecture_analyzer import ConceptCoverage

        cc = ConceptCoverage(
            total_concepts=5,
            covered_concepts=["A", "B"],
            missed_concepts=["C", "D", "E"],
            coverage_ratio=0.4,
        )
        assert cc.total_concepts == 5
        assert cc.coverage_ratio == 0.4


class TestAnalyzeTranscript:
    """Test analyze_transcript() function."""

    @patch("forma.lecture_analyzer.extract_keywords")
    @patch("forma.lecture_analyzer.create_network")
    @patch("forma.lecture_analyzer.kss")
    def test_analyze_basic(
        self, mock_kss, mock_create_net, mock_extract_kw,
    ) -> None:
        """Basic analysis returns AnalysisResult with keywords."""
        from forma.lecture_analyzer import analyze_transcript, AnalysisResult

        mock_extract_kw.return_value = ["세포막", "단백질", "세포막", "인지질"]
        mock_create_net.return_value = MagicMock()
        mock_kss.split_sentences.return_value = ["세포막은 중요하다.", "단백질도 중요하다."]

        cleaned = _make_cleaned()
        result = analyze_transcript(cleaned, concepts=None, top_n=10,
                                    no_triplets=True, provider=None)

        assert isinstance(result, AnalysisResult)
        assert result.class_id == "A"
        assert result.week == 1
        assert "세포막" in result.keyword_frequencies
        assert result.keyword_frequencies["세포막"] == 2
        assert result.top_keywords[0] == "세포막"
        assert result.sentence_count == 2

    @patch("forma.lecture_analyzer.extract_keywords")
    @patch("forma.lecture_analyzer.create_network")
    @patch("forma.lecture_analyzer.kss")
    def test_analyze_topics_skipped_few_sentences(
        self, mock_kss, mock_create_net, mock_extract_kw,
    ) -> None:
        """Less than 10 sentences -> topic_skipped_reason set."""
        from forma.lecture_analyzer import analyze_transcript

        mock_extract_kw.return_value = ["세포막", "단백질"]
        mock_create_net.return_value = MagicMock()
        mock_kss.split_sentences.return_value = ["문장1.", "문장2.", "문장3."]

        cleaned = _make_cleaned()
        result = analyze_transcript(cleaned, concepts=None, top_n=10,
                                    no_triplets=True, provider=None)

        assert result.topics is None
        assert result.topic_skipped_reason is not None
        assert "10" in result.topic_skipped_reason or "부족" in result.topic_skipped_reason

    @patch("forma.lecture_analyzer.compute_lecture_gap")
    @patch("forma.lecture_analyzer.compute_emphasis_map")
    @patch("forma.lecture_analyzer.extract_keywords")
    @patch("forma.lecture_analyzer.create_network")
    @patch("forma.lecture_analyzer.kss")
    def test_analyze_with_concepts(
        self, mock_kss, mock_create_net, mock_extract_kw,
        mock_emphasis, mock_gap,
    ) -> None:
        """concepts provided -> concept_coverage populated."""
        from forma.lecture_analyzer import analyze_transcript
        from forma.emphasis_map import InstructionalEmphasisMap
        from forma.lecture_gap_analysis import LectureGapReport

        mock_extract_kw.return_value = ["세포막", "단백질"]
        mock_create_net.return_value = MagicMock()
        mock_kss.split_sentences.return_value = ["문장1."]

        mock_emphasis.return_value = InstructionalEmphasisMap(
            concept_scores={"세포막": 0.9, "미토콘드리아": 0.1},
            threshold_used=0.65,
            n_sentences=1,
            n_concepts=2,
        )
        mock_gap.return_value = LectureGapReport(
            master_concepts={"세포막", "미토콘드리아"},
            covered_concepts={"세포막"},
            missed_concepts={"미토콘드리아"},
            extra_concepts=set(),
            coverage_ratio=0.5,
        )

        cleaned = _make_cleaned()
        result = analyze_transcript(
            cleaned, concepts=["세포막", "미토콘드리아"],
            top_n=10, no_triplets=True, provider=None,
        )

        assert result.concept_coverage is not None
        assert result.concept_coverage.total_concepts == 2
        assert "세포막" in result.concept_coverage.covered_concepts
        assert "미토콘드리아" in result.concept_coverage.missed_concepts

    @patch("forma.lecture_analyzer.compute_emphasis_map")
    @patch("forma.lecture_analyzer.extract_keywords")
    @patch("forma.lecture_analyzer.create_network")
    @patch("forma.lecture_analyzer.kss")
    def test_analyze_emphasis_scores(
        self, mock_kss, mock_create_net, mock_extract_kw, mock_emphasis,
    ) -> None:
        """concepts provided -> emphasis_scores populated."""
        from forma.lecture_analyzer import analyze_transcript
        from forma.emphasis_map import InstructionalEmphasisMap

        mock_extract_kw.return_value = ["세포막"]
        mock_create_net.return_value = MagicMock()
        mock_kss.split_sentences.return_value = ["문장1."]

        mock_emphasis.return_value = InstructionalEmphasisMap(
            concept_scores={"세포막": 0.8},
            threshold_used=0.65,
            n_sentences=1,
            n_concepts=1,
        )

        cleaned = _make_cleaned()
        # Also patch compute_lecture_gap since concepts triggers it
        with patch("forma.lecture_analyzer.compute_lecture_gap") as mock_gap:
            from forma.lecture_gap_analysis import LectureGapReport
            mock_gap.return_value = LectureGapReport(
                master_concepts={"세포막"},
                covered_concepts={"세포막"},
                missed_concepts=set(),
                extra_concepts=set(),
                coverage_ratio=1.0,
            )
            result = analyze_transcript(
                cleaned, concepts=["세포막"],
                top_n=10, no_triplets=True, provider=None,
            )

        assert result.emphasis_scores is not None
        assert result.emphasis_scores["세포막"] == pytest.approx(0.8)

    @patch("forma.lecture_analyzer.extract_keywords")
    @patch("forma.lecture_analyzer.create_network")
    @patch("forma.lecture_analyzer.kss")
    def test_analyze_triplets_skipped_no_provider(
        self, mock_kss, mock_create_net, mock_extract_kw,
    ) -> None:
        """No LLM provider -> triplet_skipped_reason set."""
        from forma.lecture_analyzer import analyze_transcript

        mock_extract_kw.return_value = ["세포막"]
        mock_create_net.return_value = MagicMock()
        mock_kss.split_sentences.return_value = ["문장."]

        cleaned = _make_cleaned()
        result = analyze_transcript(cleaned, concepts=None, top_n=10,
                                    no_triplets=False, provider=None)

        assert result.triplets is None
        assert result.triplet_skipped_reason is not None

    @patch("forma.lecture_analyzer.extract_keywords")
    @patch("forma.lecture_analyzer.create_network")
    @patch("forma.lecture_analyzer.kss")
    def test_analyze_triplets_skipped_flag(
        self, mock_kss, mock_create_net, mock_extract_kw,
    ) -> None:
        """no_triplets=True -> triplet_skipped_reason set."""
        from forma.lecture_analyzer import analyze_transcript

        mock_extract_kw.return_value = ["세포막"]
        mock_create_net.return_value = MagicMock()
        mock_kss.split_sentences.return_value = ["문장."]

        cleaned = _make_cleaned()
        result = analyze_transcript(cleaned, concepts=None, top_n=10,
                                    no_triplets=True, provider=MagicMock())

        assert result.triplets is None
        assert result.triplet_skipped_reason is not None

    @patch("forma.lecture_analyzer.extract_keywords")
    @patch("forma.lecture_analyzer.create_network")
    @patch("forma.lecture_analyzer.kss")
    def test_analyze_stage_failure_independent(
        self, mock_kss, mock_create_net, mock_extract_kw,
    ) -> None:
        """One stage throws, others still run (FR-027)."""
        from forma.lecture_analyzer import analyze_transcript

        mock_extract_kw.return_value = ["세포막", "단백질"]
        mock_create_net.side_effect = RuntimeError("network failure")
        mock_kss.split_sentences.return_value = ["문장."]

        cleaned = _make_cleaned()
        result = analyze_transcript(cleaned, concepts=None, top_n=10,
                                    no_triplets=True, provider=None)

        # keyword extraction succeeded even though network failed
        assert result.keyword_frequencies is not None
        assert len(result.keyword_frequencies) > 0
        # network image should be None due to failure
        assert result.network_image_path is None


class TestCaching:
    """Test save/load round-trip and cache hit behavior."""

    @patch("forma.lecture_analyzer.extract_keywords")
    @patch("forma.lecture_analyzer.create_network")
    @patch("forma.lecture_analyzer.kss")
    def test_save_and_load_analysis_result(
        self, mock_kss, mock_create_net, mock_extract_kw, tmp_path: Path,
    ) -> None:
        """Round-trip YAML serialization preserves data."""
        from forma.lecture_analyzer import (
            AnalysisResult, save_analysis_result, load_analysis_result,
        )

        result = AnalysisResult(
            class_id="A",
            week=1,
            keyword_frequencies={"세포막": 3, "단백질": 2},
            top_keywords=["세포막", "단백질"],
            network_image_path=None,
            topics=None,
            topic_skipped_reason="문장 수 부족 (3 < 10)",
            concept_coverage=None,
            emphasis_scores={"세포막": 0.8},
            triplets=None,
            triplet_skipped_reason="LLM 미제공",
            sentence_count=3,
            analysis_timestamp="2026-01-01T00:00:00",
        )

        saved_path = save_analysis_result(result, tmp_path)
        assert saved_path.exists()
        assert saved_path.suffix == ".yaml"

        loaded = load_analysis_result(saved_path)
        assert loaded.class_id == result.class_id
        assert loaded.week == result.week
        assert loaded.keyword_frequencies == result.keyword_frequencies
        assert loaded.top_keywords == result.top_keywords
        assert loaded.topic_skipped_reason == result.topic_skipped_reason
        assert loaded.emphasis_scores == result.emphasis_scores
        assert loaded.sentence_count == result.sentence_count

    def test_load_analysis_result_not_found(self, tmp_path: Path) -> None:
        """Loading from non-existent path raises FileNotFoundError."""
        from forma.lecture_analyzer import load_analysis_result

        with pytest.raises(FileNotFoundError):
            load_analysis_result(tmp_path / "nonexistent.yaml")
