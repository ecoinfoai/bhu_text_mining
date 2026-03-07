"""Tests for triplet_extractor.py — LLM-based triplet extraction.

All LLM calls are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from forma.evaluation_types import TripletEdge, TripletExtractionResult
from forma.triplet_extractor import TripletExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_JSON_RESPONSE = json.dumps([
    {"subject": "수용체", "relation": "감지하다", "object": "한계점"},
    {"subject": "통합센터", "relation": "명령하다", "object": "효과기"},
])

SAMPLE_JSON_WITH_MARKDOWN = f"```json\n{SAMPLE_JSON_RESPONSE}\n```"


def _make_provider(responses: list[str]) -> MagicMock:
    """Create mock LLM provider returning given responses."""
    prov = MagicMock()
    prov.generate.side_effect = responses
    return prov


# ---------------------------------------------------------------------------
# TripletExtractor tests
# ---------------------------------------------------------------------------


class TestTripletExtractor:
    """Tests for TripletExtractor."""

    def test_empty_response_returns_empty(self):
        """Empty student response returns empty triplets."""
        prov = _make_provider([])
        ext = TripletExtractor(prov)
        result = ext.extract("s001", 1, "질문?", "", ["A", "B"])
        assert result.triplets == []
        assert result.student_id == "s001"

    @patch("forma.triplet_extractor.encode_texts")
    def test_extraction_with_consensus(self, mock_encode):
        """3-call consensus returns agreed triplets."""
        import numpy as np

        # All 3 calls return the same triplets
        responses = [SAMPLE_JSON_WITH_MARKDOWN] * 3
        prov = _make_provider(responses)

        # Mock embeddings: identical triplets → high similarity
        mock_encode.return_value = np.array([
            [1.0, 0.0], [0.0, 1.0],  # call 0
            [1.0, 0.0], [0.0, 1.0],  # call 1
            [1.0, 0.0], [0.0, 1.0],  # call 2
        ])

        ext = TripletExtractor(prov)
        result = ext.extract("s001", 1, "질문?", "학생 답변", ["수용체", "통합센터"])

        assert isinstance(result, TripletExtractionResult)
        assert len(result.call_results) == 3
        assert result.consensus_method == "majority_2of3"

    def test_parse_failure_returns_empty(self):
        """LLM returning non-JSON → empty triplets for that call."""
        responses = ["invalid json", SAMPLE_JSON_RESPONSE, SAMPLE_JSON_RESPONSE]
        prov = _make_provider(responses)

        ext = TripletExtractor(prov)
        # Use exact consensus fallback to avoid embedding dependency
        with patch("forma.triplet_extractor.encode_texts", side_effect=Exception("no model")):
            result = ext.extract("s001", 1, "Q?", "answer", ["A"])

        assert isinstance(result, TripletExtractionResult)
        # First call failed → only 2 calls contributed
        assert len(result.call_results[0]) == 0

    def test_parse_json_in_markdown(self):
        """Parses JSON wrapped in ```json``` blocks."""
        prov = _make_provider([SAMPLE_JSON_WITH_MARKDOWN] * 3)
        ext = TripletExtractor(prov)
        triplets = ext._parse_triplets(SAMPLE_JSON_WITH_MARKDOWN)
        assert len(triplets) == 2
        assert triplets[0].subject == "수용체"

    def test_parse_plain_json(self):
        """Parses plain JSON array."""
        prov = _make_provider([])
        ext = TripletExtractor(prov)
        triplets = ext._parse_triplets(SAMPLE_JSON_RESPONSE)
        assert len(triplets) == 2

    def test_parse_invalid_json(self):
        """Invalid JSON returns empty list."""
        prov = _make_provider([])
        ext = TripletExtractor(prov)
        triplets = ext._parse_triplets("not json at all")
        assert triplets == []

    def test_parse_missing_keys(self):
        """Items missing required keys are skipped."""
        prov = _make_provider([])
        ext = TripletExtractor(prov)
        data = json.dumps([
            {"subject": "A", "relation": "r"},  # missing object
            {"subject": "B", "relation": "r", "object": "C"},  # valid
        ])
        triplets = ext._parse_triplets(data)
        assert len(triplets) == 1
        assert triplets[0].subject == "B"

    def test_exact_consensus_fallback(self):
        """Exact consensus works when embeddings unavailable."""
        prov = _make_provider([])
        ext = TripletExtractor(prov, n_calls=3)

        t1 = TripletEdge("A", "r", "B")
        t2 = TripletEdge("C", "r", "D")
        call_results = [
            [t1, t2],
            [t1],
            [t1, t2],
        ]
        consensus = ext._exact_consensus(call_results)
        assert len(consensus) == 2  # Both appear in ≥2 calls

    def test_exact_consensus_minority_excluded(self):
        """Triplets appearing in only 1 of 3 calls are excluded."""
        prov = _make_provider([])
        ext = TripletExtractor(prov, n_calls=3)

        t1 = TripletEdge("A", "r", "B")
        t2 = TripletEdge("C", "r", "D")
        call_results = [
            [t1],
            [t2],
            [t1],
        ]
        consensus = ext._exact_consensus(call_results)
        assert len(consensus) == 1
        assert consensus[0].subject == "A"

    def test_n_calls_2_consensus_requires_2_votes(self):
        """With n_calls=2, min_votes must be 2 (not 1). Bug fix test."""
        prov = _make_provider([])
        ext = TripletExtractor(prov, n_calls=2)

        t1 = TripletEdge("A", "r", "B")
        t2 = TripletEdge("C", "r", "D")
        # t1 appears in only 1 call, t2 appears in only 1 call
        call_results = [
            [t1],
            [t2],
        ]
        consensus = ext._exact_consensus(call_results)
        # With min_votes=2, neither should pass (only 1 vote each)
        assert len(consensus) == 0

    def test_n_calls_2_consensus_passes_with_agreement(self):
        """With n_calls=2, triplets in both calls pass consensus."""
        prov = _make_provider([])
        ext = TripletExtractor(prov, n_calls=2)

        t1 = TripletEdge("A", "r", "B")
        call_results = [
            [t1],
            [t1],
        ]
        consensus = ext._exact_consensus(call_results)
        assert len(consensus) == 1

    def test_n_calls_zero_raises(self):
        """n_calls=0 raises ValueError."""
        prov = _make_provider([])
        with pytest.raises(ValueError, match="n_calls"):
            TripletExtractor(prov, n_calls=0)

    def test_n_calls_negative_raises(self):
        """n_calls=-1 raises ValueError."""
        prov = _make_provider([])
        with pytest.raises(ValueError, match="n_calls"):
            TripletExtractor(prov, n_calls=-1)

    def test_provider_exception_returns_empty_call(self):
        """LLM provider exception → empty list for that call."""
        prov = MagicMock()
        prov.generate.side_effect = Exception("API error")

        ext = TripletExtractor(prov)
        with patch("forma.triplet_extractor.encode_texts", side_effect=Exception("no model")):
            result = ext.extract("s001", 1, "Q?", "answer", ["A"])

        assert all(len(cr) == 0 for cr in result.call_results)
