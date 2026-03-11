"""End-to-end tests for the v2 graph-based evaluation pipeline.

All LLM and embedding calls are mocked. Tests verify the full flow
from config → extraction → comparison → feedback → ensemble.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from forma.evaluation_types import (
    TripletEdge,
    TripletExtractionResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


V2_CONFIG = {
    "course_name": "인체구조와기능",
    "questions": [
        {
            "sn": 1,
            "topic": "항상성",
            "question_type": "essay",
            "question": "항상성의 기전을 설명하시오.",
            "model_answer": "수용체가 한계점 일탈을 감지하면 통합센터가 효과기에 명령한다.",
            "keywords": ["항상성", "수용체", "통합센터", "효과기"],
            "rubric": {"high": "완벽", "mid": "부분", "low": "미달"},
            "support": {"advanced": "", "beginning": "기초 복습"},
            "knowledge_graph": {
                "edges": [
                    {"subject": "수용체", "relation": "감지", "object": "한계점 일탈"},
                    {"subject": "통합센터", "relation": "명령", "object": "효과기"},
                ],
                "similarity_threshold": 0.80,
                "node_aliases": {
                    "항상성": ["homeostasis"],
                },
            },
            "rubric_tiers": {
                "level_3": {"label": "전문적 구조화", "min_graph_f1": 0.85, "requires_terminology": True},
                "level_2": {"label": "기전+용어", "min_graph_f1": 0.6, "requires_terminology": True},
                "level_1": {"label": "기전 이해", "min_graph_f1": 0.3, "requires_terminology": False},
                "level_0": {"label": "미달", "min_graph_f1": 0.0, "requires_terminology": False},
            },
        },
    ],
}

V1_CONFIG = {
    "course_name": "인체구조와기능",
    "questions": [
        {
            "sn": 1,
            "question_type": "essay",
            "question": "세포막의 기능은?",
            "model_answer": "선택적 투과성으로 물질 이동 조절.",
            "keywords": ["세포막", "선택적 투과성"],
            "rubric": {"high": "완벽", "mid": "부분", "low": "미달"},
        },
    ],
}

RESPONSES = {
    "responses": {
        "s001": {1: "수용체가 한계점 일탈을 감지하고 통합센터가 효과기에 명령한다."},
        "s002": {1: "항상성은 체온 유지입니다."},
        "s003": {1: ""},
    },
}


@pytest.fixture()
def v2_config_file(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(V2_CONFIG, allow_unicode=True), encoding="utf-8")
    return str(p)


@pytest.fixture()
def v1_config_file(tmp_path):
    p = tmp_path / "config_v1.yaml"
    p.write_text(yaml.dump(V1_CONFIG, allow_unicode=True), encoding="utf-8")
    return str(p)


@pytest.fixture()
def responses_file(tmp_path):
    p = tmp_path / "responses.yaml"
    p.write_text(yaml.dump(RESPONSES, allow_unicode=True), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_triplet_json(triplets: list[dict]) -> str:
    return f"```json\n{json.dumps(triplets, ensure_ascii=False)}\n```"


GOOD_TRIPLETS = _mock_triplet_json([
    {"subject": "수용체", "relation": "감지", "object": "한계점 일탈"},
    {"subject": "통합센터", "relation": "명령", "object": "효과기"},
])

PARTIAL_TRIPLETS = _mock_triplet_json([
    {"subject": "항상성", "relation": "유지", "object": "체온"},
])


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------


class TestE2EGraphPipeline:
    """End-to-end tests for the v2 graph-based pipeline."""

    @patch("forma.graph_comparator.encode_texts")
    @patch("forma.triplet_extractor.encode_texts")
    @patch("forma.concept_checker.encode_texts")
    def test_v2_full_flow(
        self,
        mock_concept_enc,
        mock_triplet_enc,
        mock_graph_enc,
        v2_config_file,
        responses_file,
        tmp_path,
    ):
        """Full v2 flow with mocked LLM and embeddings."""
        output_dir = str(tmp_path / "output")

        # Mock concept checker embeddings
        mock_concept_enc.return_value = np.random.rand(4, 10).astype(np.float32)

        # Mock triplet extraction embeddings
        mock_triplet_enc.return_value = np.eye(6, dtype=np.float32)

        # Mock graph comparator embeddings (force exact fallback)
        mock_graph_enc.side_effect = Exception("use exact matching")

        # Mock LLM provider
        with patch("forma.triplet_extractor.TripletExtractor._single_extraction") as mock_extract:
            mock_extract.return_value = [
                TripletEdge("수용체", "감지", "한계점 일탈"),
                TripletEdge("통합센터", "명령", "효과기"),
            ]
            with patch("forma.feedback_generator.FeedbackGenerator.generate") as mock_feedback:
                from forma.evaluation_types import FeedbackResult
                mock_feedback.return_value = FeedbackResult(
                    student_id="test",
                    question_sn=1,
                    feedback_text="좋은 답변입니다.",
                    char_count=8,
                    data_sources_used=["graph_f1"],
                    tier_level=2,
                    tier_label="기전+용어",
                )

                from forma.pipeline_evaluation import run_evaluation_pipeline

                run_evaluation_pipeline(
                    config_path=v2_config_file,
                    responses_path=responses_file,
                    output_dir=output_dir,
                    skip_feedback=False,
                    skip_statistical=True,
                    api_key="fake-key",
                    provider="gemini",
                )

        # Verify output files exist
        assert os.path.isfile(os.path.join(output_dir, "res_lvl1", "concept_results.yaml"))
        assert os.path.isfile(os.path.join(output_dir, "res_lvl4", "ensemble_results.yaml"))
        assert os.path.isfile(os.path.join(output_dir, "res_lvl4", "counseling_summary.yaml"))

    @patch("forma.concept_checker.encode_texts")
    def test_v1_compat(
        self,
        mock_enc,
        v1_config_file,
        responses_file,
        tmp_path,
    ):
        """v1 config without knowledge_graph works as before."""
        output_dir = str(tmp_path / "output_v1")
        mock_enc.return_value = np.random.rand(2, 10).astype(np.float32)

        from forma.pipeline_evaluation import run_evaluation_pipeline

        run_evaluation_pipeline(
            config_path=v1_config_file,
            responses_path=responses_file,
            output_dir=output_dir,
            skip_feedback=True,
            skip_statistical=True,
        )

        # Should produce concept results
        assert os.path.isfile(os.path.join(output_dir, "res_lvl1", "concept_results.yaml"))
        # No graph comparison file (v1 mode)
        assert not os.path.isfile(
            os.path.join(output_dir, "res_lvl1", "graph_comparison_results.yaml")
        )

    @patch("forma.concept_checker.encode_texts")
    def test_empty_response_produces_result(
        self,
        mock_enc,
        v2_config_file,
        tmp_path,
    ):
        """Empty student response gets a score (not silently skipped)."""
        responses = {"responses": {"s_empty": {1: ""}}}
        resp_file = tmp_path / "empty_resp.yaml"
        resp_file.write_text(yaml.dump(responses, allow_unicode=True), encoding="utf-8")

        output_dir = str(tmp_path / "output_empty")
        mock_enc.return_value = np.random.rand(2, 10).astype(np.float32)

        from forma.pipeline_evaluation import run_evaluation_pipeline

        run_evaluation_pipeline(
            config_path=v2_config_file,
            responses_path=str(resp_file),
            output_dir=output_dir,
            skip_feedback=True,
            skip_graph=True,
            skip_statistical=True,
        )

        # Should have ensemble results for the student
        ensemble_path = os.path.join(output_dir, "res_lvl4", "ensemble_results.yaml")
        assert os.path.isfile(ensemble_path)


class TestConfigValidation:
    """Tests for config validation in pipeline."""

    def test_invalid_config_raises(self, tmp_path):
        """Invalid config raises ValueError."""
        bad_config = {
            "questions": [
                {"sn": 1, "question_type": "invalid_type"},
            ],
        }
        cfg_file = tmp_path / "bad_config.yaml"
        cfg_file.write_text(yaml.dump(bad_config, allow_unicode=True), encoding="utf-8")

        responses = {"responses": {"s001": {1: "answer"}}}
        resp_file = tmp_path / "resp.yaml"
        resp_file.write_text(yaml.dump(responses, allow_unicode=True), encoding="utf-8")

        from forma.pipeline_evaluation import run_evaluation_pipeline

        with pytest.raises(ValueError, match="validation failed"):
            run_evaluation_pipeline(
                config_path=str(cfg_file),
                responses_path=str(resp_file),
                output_dir=str(tmp_path / "out"),
                skip_feedback=True,
                skip_statistical=True,
            )


class TestTripletExtractorE2E:
    """Integration tests for triplet extractor."""

    def test_triplet_extraction_with_mock_llm(self):
        """TripletExtractor extracts and achieves consensus with mock LLM."""
        from forma.triplet_extractor import TripletExtractor

        triplet_json = json.dumps([
            {"subject": "A", "relation": "causes", "object": "B"},
        ])
        mock_prov = MagicMock()
        mock_prov.generate.return_value = f"```json\n{triplet_json}\n```"

        extractor = TripletExtractor(mock_prov)

        with patch("forma.triplet_extractor.encode_texts", side_effect=Exception):
            result = extractor.extract("s001", 1, "Q?", "A causes B", ["A", "B"])

        assert isinstance(result, TripletExtractionResult)
        assert mock_prov.generate.call_count == 3


class TestGraphComparatorE2E:
    """Integration tests for graph comparator."""

    def test_exact_match_f1_one(self):
        """Identical edges yield F1 = 1.0 with exact matching."""
        from forma.graph_comparator import GraphComparator

        master = [TripletEdge("A", "r", "B"), TripletEdge("C", "r", "D")]
        student = [TripletEdge("A", "r", "B"), TripletEdge("C", "r", "D")]

        gc = GraphComparator()
        with patch("forma.graph_comparator.encode_texts", side_effect=Exception):
            result = gc.compare("s001", 1, master, student)

        assert result.f1 == pytest.approx(1.0)
        assert len(result.missing_edges) == 0


class TestFeedbackGeneratorE2E:
    """Integration tests for feedback generator."""

    def test_generates_feedback_text(self):
        """FeedbackGenerator produces feedback text from mock LLM."""
        from forma.feedback_generator import FeedbackGenerator

        mock_prov = MagicMock()
        mock_prov.generate.return_value = "개념 이해도가 좋습니다. 추가 학습이 필요합니다."

        gen = FeedbackGenerator(mock_prov)
        result = gen.generate(
            student_id="s001",
            question_sn=1,
            question="Q?",
            student_response="답변",
            concept_coverage=0.8,
            graph_comparison=None,
            tier_level=2,
            tier_label="기전+용어",
        )

        assert result.char_count > 0
        assert result.tier_level == 2
