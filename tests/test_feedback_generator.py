"""Tests for feedback_generator.py — coaching feedback generation.

All LLM calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.evaluation_types import (
    FeedbackResult,
    GraphComparisonResult,
    TripletEdge,
)
from src.feedback_generator import (
    EMPTY_RESPONSE_FEEDBACK,
    FeedbackGenerator,
    _format_edges,
    _truncate_at_sentence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_comparison(
    f1: float = 0.7,
    n_matched: int = 3,
    n_missing: int = 1,
    n_wrong: int = 0,
) -> GraphComparisonResult:
    matched = [TripletEdge(f"S{i}", "r", f"O{i}") for i in range(n_matched)]
    missing = [TripletEdge(f"MS{i}", "r", f"MO{i}") for i in range(n_missing)]
    wrong = [TripletEdge(f"WS{i}", "r", f"WO{i}") for i in range(n_wrong)]
    return GraphComparisonResult(
        student_id="s001",
        question_sn=1,
        precision=f1,
        recall=f1,
        f1=f1,
        matched_edges=matched,
        missing_edges=missing,
        extra_edges=[],
        wrong_direction_edges=wrong,
    )


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestFormatEdges:
    """Tests for _format_edges()."""

    def test_empty_list(self):
        assert _format_edges([]) == "(없음)"

    def test_formats_edges(self):
        edges = [TripletEdge("A", "causes", "B")]
        result = _format_edges(edges)
        assert "A" in result
        assert "causes" in result
        assert "B" in result


class TestTruncateAtSentence:
    """Tests for _truncate_at_sentence()."""

    def test_short_text_unchanged(self):
        assert _truncate_at_sentence("짧은 문장.", 100) == "짧은 문장."

    def test_truncates_at_period(self):
        text = "첫 문장. 두 번째 문장. 세 번째 긴 문장입니다."
        result = _truncate_at_sentence(text, 20)
        assert result.endswith(".")
        assert len(result) <= 20

    def test_no_period_truncates_at_max(self):
        text = "마침표 없는 긴 텍스트" * 10
        result = _truncate_at_sentence(text, 30)
        assert len(result) <= 30


# ---------------------------------------------------------------------------
# FeedbackGenerator tests
# ---------------------------------------------------------------------------


class TestFeedbackGenerator:
    """Tests for FeedbackGenerator."""

    @pytest.fixture()
    def mock_provider(self):
        prov = MagicMock()
        prov.generate.return_value = "좋은 답변입니다. 개선점은 다음과 같습니다."
        return prov

    def test_empty_response(self, mock_provider):
        """Empty student response returns template feedback."""
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            student_id="s001",
            question_sn=1,
            question="Q?",
            student_response="",
            concept_coverage=0.0,
            graph_comparison=None,
            tier_level=0,
            tier_label="미달",
        )
        assert result.feedback_text == EMPTY_RESPONSE_FEEDBACK
        assert result.tier_level == 0
        assert mock_provider.generate.call_count == 0

    def test_generates_feedback(self, mock_provider):
        """Normal response generates LLM feedback."""
        gen = FeedbackGenerator(mock_provider)
        gc = _make_graph_comparison()
        result = gen.generate(
            student_id="s001",
            question_sn=1,
            question="항상성이란?",
            student_response="체온을 일정하게 유지하는 것",
            concept_coverage=0.6,
            graph_comparison=gc,
            tier_level=1,
            tier_label="기전 이해",
        )
        assert isinstance(result, FeedbackResult)
        assert result.feedback_text == "좋은 답변입니다. 개선점은 다음과 같습니다."
        assert result.student_id == "s001"
        assert mock_provider.generate.call_count == 1

    def test_feedback_within_char_limit(self, mock_provider):
        """Feedback is truncated to max_chars."""
        mock_provider.generate.return_value = "A" * 3000
        gen = FeedbackGenerator(mock_provider, max_chars=100)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해"
        )
        assert result.char_count <= 100

    def test_data_sources_with_graph(self, mock_provider):
        """Data sources include graph_f1 when graph_comparison present."""
        gen = FeedbackGenerator(mock_provider)
        gc = _make_graph_comparison()
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, gc, 2, "기전+용어"
        )
        assert "graph_f1" in result.data_sources_used

    def test_data_sources_without_graph(self, mock_provider):
        """Data sources exclude graph_f1 when no graph_comparison."""
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해"
        )
        assert "graph_f1" not in result.data_sources_used

    def test_provider_failure(self, mock_provider):
        """Provider exception returns error message feedback."""
        mock_provider.generate.side_effect = Exception("API down")
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해"
        )
        assert "실패" in result.feedback_text

    def test_tier_level_preserved(self, mock_provider):
        """Tier level and label preserved in result."""
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 3, "전문적 구조화"
        )
        assert result.tier_level == 3
        assert result.tier_label == "전문적 구조화"
