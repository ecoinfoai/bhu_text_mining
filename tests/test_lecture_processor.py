"""Tests for lecture_processor.py — lecture transcript processing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from forma.lecture_processor import (
    extract_lecture_covered_concepts,
    extract_lecture_tone_sample,
    extract_triplets_from_lecture,
    load_transcript,
    segment_text,
    MAX_TRANSCRIPT_LENGTH,
)


# ---------------------------------------------------------------------------
# load_transcript
# ---------------------------------------------------------------------------


class TestLoadTranscript:
    def test_loads_valid_file(self, tmp_path):
        f = tmp_path / "lecture.txt"
        f.write_text("세포막은 인지질 이중층으로 구성됩니다.", encoding="utf-8")
        result = load_transcript(str(f))
        assert "세포막" in result

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_transcript("/nonexistent/path/lecture.txt")

    def test_raises_value_error_for_long_text(self, tmp_path):
        f = tmp_path / "long.txt"
        f.write_text("가" * (MAX_TRANSCRIPT_LENGTH + 1), encoding="utf-8")
        with pytest.raises(ValueError, match="exceeds maximum"):
            load_transcript(str(f))

    def test_accepts_exactly_max_length(self, tmp_path):
        f = tmp_path / "exact.txt"
        f.write_text("가" * MAX_TRANSCRIPT_LENGTH, encoding="utf-8")
        result = load_transcript(str(f))
        assert len(result) == MAX_TRANSCRIPT_LENGTH


# ---------------------------------------------------------------------------
# segment_text
# ---------------------------------------------------------------------------


class TestSegmentText:
    def test_splits_at_sentence_boundaries(self):
        text = "첫 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다."
        segments = segment_text(text, max_chars=30)
        assert len(segments) >= 2
        for seg in segments:
            assert len(seg) <= 30

    def test_respects_max_chars(self):
        text = "짧은 문장. " * 50
        segments = segment_text(text, max_chars=40)
        for seg in segments:
            assert len(seg) <= 40

    def test_handles_text_without_periods(self):
        text = "마침표가 없는 텍스트"
        segments = segment_text(text, max_chars=100)
        assert segments == [text]

    def test_empty_text(self):
        assert segment_text("") == []

    def test_single_sentence_within_limit(self):
        text = "하나의 문장입니다."
        segments = segment_text(text, max_chars=100)
        assert segments == [text]

    def test_splits_on_exclamation_and_question(self):
        text = "정말요! 그렇습니까? 네 맞습니다."
        segments = segment_text(text, max_chars=15)
        assert len(segments) >= 2


# ---------------------------------------------------------------------------
# extract_lecture_covered_concepts
# ---------------------------------------------------------------------------


class TestExtractLectureCoveredConcepts:
    @patch("forma.lecture_processor.encode_texts")
    def test_finds_matching_concepts(self, mock_encode):
        lecture_vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        concept_vecs = np.array(
            [
                [0.9, 0.1, 0.0],  # high similarity
                [0.0, 0.0, 1.0],  # low similarity
            ],
            dtype=np.float32,
        )
        mock_encode.side_effect = [lecture_vec, concept_vecs]

        result = extract_lecture_covered_concepts(
            "강의 텍스트",
            ["세포막", "광합성"],
            threshold=0.75,
        )
        assert "세포막" in result
        assert "광합성" not in result

    @patch("forma.lecture_processor.encode_texts")
    def test_empty_master_concepts(self, mock_encode):
        result = extract_lecture_covered_concepts("텍스트", [], threshold=0.75)
        assert result == []
        mock_encode.assert_not_called()

    @patch("forma.lecture_processor.encode_texts")
    def test_threshold_boundary(self, mock_encode):
        vec = np.array([[1.0, 0.0]], dtype=np.float32)
        mock_encode.side_effect = [vec, vec]

        result = extract_lecture_covered_concepts(
            "텍스트", ["개념"], threshold=1.0
        )
        assert result == ["개념"]


# ---------------------------------------------------------------------------
# extract_lecture_tone_sample
# ---------------------------------------------------------------------------


class TestExtractLectureToneSample:
    def test_short_text_returned_as_is(self):
        text = "짧은 텍스트입니다."
        assert extract_lecture_tone_sample(text, max_chars=500) == text

    def test_truncates_at_sentence_boundary(self):
        text = "첫 번째 문장입니다. 두 번째 문장입니다. " + "추가 텍스트 " * 100
        result = extract_lecture_tone_sample(text, max_chars=30)
        assert len(result) <= 30
        assert result.endswith(".")

    def test_truncates_without_boundary(self):
        text = "마침표없음" * 200
        result = extract_lecture_tone_sample(text, max_chars=50)
        assert len(result) <= 50


# ---------------------------------------------------------------------------
# extract_triplets_from_lecture
# ---------------------------------------------------------------------------


class TestExtractTripletsFromLecture:
    def test_parses_json_response(self):
        mock_provider = MagicMock()
        mock_provider.generate.return_value = """```json
[
  {"subject": "세포막", "relation": "구성하다", "object": "인지질"},
  {"subject": "미토콘드리아", "relation": "생산하다", "object": "ATP"}
]
```"""
        triplets = extract_triplets_from_lecture("강의 내용", mock_provider)
        assert len(triplets) == 2
        assert triplets[0].subject == "세포막"
        assert triplets[0].relation == "구성하다"
        assert triplets[0].object == "인지질"
        assert triplets[1].subject == "미토콘드리아"

    def test_parses_raw_json_without_fences(self):
        mock_provider = MagicMock()
        mock_provider.generate.return_value = (
            '[{"subject": "A", "relation": "B", "object": "C"}]'
        )
        triplets = extract_triplets_from_lecture("텍스트", mock_provider)
        assert len(triplets) == 1
        assert triplets[0].subject == "A"

    def test_calls_provider_with_correct_params(self):
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "[]"
        extract_triplets_from_lecture("내용", mock_provider)
        mock_provider.generate.assert_called_once()
        call_kwargs = mock_provider.generate.call_args
        assert call_kwargs[1]["max_tokens"] == 1024
        assert call_kwargs[1]["temperature"] == 0.0

    def test_empty_response(self):
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "[]"
        triplets = extract_triplets_from_lecture("텍스트", mock_provider)
        assert triplets == []
