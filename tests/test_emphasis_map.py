"""Tests for emphasis_map.py — instructional emphasis mapping.

RED phase: these tests are written *before* the module exists and will fail
until ``src/forma/emphasis_map.py`` is implemented.

Covers US2 (FR-008 ~ FR-012, SC-002 ~ SC-003):
  - InstructionalEmphasisMap dataclass
  - compute_emphasis_map(): embedding-based concept emphasis scoring
  - compute_weighted_concept_coverage(): emphasis-weighted coverage
  - EnsembleScorer integration with emphasis_map parameter
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from forma.emphasis_map import (
    InstructionalEmphasisMap,
    compute_emphasis_map,
    compute_weighted_concept_coverage,
)


# ---------------------------------------------------------------------------
# FR-008: InstructionalEmphasisMap dataclass
# ---------------------------------------------------------------------------


class TestInstructionalEmphasisMap:
    """FR-008: InstructionalEmphasisMap has correct fields."""

    def test_fields_exist(self):
        em = InstructionalEmphasisMap(
            concept_scores={"항상성": 0.8, "음성되먹임": 0.3},
            threshold_used=0.5,
            n_sentences=20,
            n_concepts=2,
        )
        assert em.concept_scores == {"항상성": 0.8, "음성되먹임": 0.3}
        assert em.threshold_used == 0.5
        assert em.n_sentences == 20
        assert em.n_concepts == 2

    def test_empty_concept_scores(self):
        em = InstructionalEmphasisMap(
            concept_scores={},
            threshold_used=0.5,
            n_sentences=0,
            n_concepts=0,
        )
        assert em.concept_scores == {}
        assert em.n_concepts == 0


# ---------------------------------------------------------------------------
# FR-009: compute_emphasis_map()
# ---------------------------------------------------------------------------


class TestComputeEmphasisMap:
    """FR-009: compute_emphasis_map scores concepts by transcript emphasis."""

    def _mock_encode_texts(self, texts, model_name=None):
        """Mock encode_texts that returns deterministic embeddings."""
        # Use simple hash-based embeddings for deterministic results
        embeddings = []
        for text in texts:
            # Create a simple embedding based on the text
            np.random.seed(hash(text) % 2**31)
            embeddings.append(np.random.randn(384).astype(np.float32))
        return np.array(embeddings)

    @patch("forma.emphasis_map.encode_texts")
    def test_high_mention_concept_scores_higher(self, mock_encode):
        """Concept mentioned in many sentences scores higher than rare one."""
        # Create embeddings where concept A is similar to many sentences
        dim = 384

        # Concept embeddings
        concept_a_emb = np.ones(dim, dtype=np.float32)  # "항상성"
        concept_b_emb = -np.ones(dim, dtype=np.float32)  # "음성되먹임"

        # Sentence embeddings: 10 similar to concept A, 1 similar to concept B
        sentence_embs = []
        for _ in range(10):
            sentence_embs.append(concept_a_emb + np.random.randn(dim).astype(np.float32) * 0.1)
        sentence_embs.append(concept_b_emb + np.random.randn(dim).astype(np.float32) * 0.1)

        all_texts_embs = np.array(sentence_embs + [concept_a_emb, concept_b_emb])

        def side_effect(texts, model_name=None):
            return all_texts_embs[:len(texts)]

        mock_encode.side_effect = side_effect

        sentences = [f"항상성 문장 {i}" for i in range(10)] + ["음성되먹임 문장"]
        concepts = ["항상성", "음성되먹임"]

        result = compute_emphasis_map(sentences, concepts)
        assert isinstance(result, InstructionalEmphasisMap)
        assert result.concept_scores["항상성"] > result.concept_scores["음성되먹임"]

    @patch("forma.emphasis_map.encode_texts")
    def test_scores_in_zero_one_range(self, mock_encode):
        """All concept scores must be in [0.0, 1.0]."""
        dim = 384
        mock_encode.return_value = np.random.randn(5, dim).astype(np.float32)

        sentences = ["문장1", "문장2", "문장3"]
        concepts = ["개념1", "개념2"]

        result = compute_emphasis_map(sentences, concepts)
        for score in result.concept_scores.values():
            assert 0.0 <= score <= 1.0

    @patch("forma.emphasis_map.encode_texts")
    def test_empty_transcript(self, mock_encode):
        """Empty transcript → empty concept_scores."""
        mock_encode.return_value = np.array([]).reshape(0, 384)
        result = compute_emphasis_map([], ["항상성", "음성되먹임"])
        assert result.concept_scores == {}
        assert result.n_sentences == 0

    @patch("forma.emphasis_map.encode_texts")
    def test_empty_concepts(self, mock_encode):
        """Empty concepts → empty concept_scores."""
        mock_encode.return_value = np.random.randn(3, 384).astype(np.float32)
        result = compute_emphasis_map(["문장1", "문장2", "문장3"], [])
        assert result.concept_scores == {}
        assert result.n_concepts == 0

    @patch("forma.emphasis_map.encode_texts")
    def test_n_sentences_and_n_concepts_set(self, mock_encode):
        """n_sentences and n_concepts reflect input sizes."""
        dim = 384
        n_sent = 5
        n_conc = 3
        mock_encode.return_value = np.random.randn(n_sent + n_conc, dim).astype(np.float32)

        sentences = [f"문장{i}" for i in range(n_sent)]
        concepts = [f"개념{i}" for i in range(n_conc)]

        result = compute_emphasis_map(sentences, concepts)
        assert result.n_sentences == n_sent
        assert result.n_concepts == n_conc

    @patch("forma.emphasis_map.encode_texts")
    def test_threshold_used_is_recorded(self, mock_encode):
        """threshold_used reflects the threshold parameter."""
        dim = 384
        mock_encode.return_value = np.random.randn(5, dim).astype(np.float32)

        result = compute_emphasis_map(
            ["문장1", "문장2", "문장3"],
            ["개념1", "개념2"],
            threshold=0.42,
        )
        assert result.threshold_used == 0.42


# ---------------------------------------------------------------------------
# FR-010: compute_weighted_concept_coverage()
# ---------------------------------------------------------------------------


class TestWeightedConceptCoverage:
    """FR-010: compute_weighted_concept_coverage computes emphasis-weighted coverage."""

    def test_weighted_coverage_formula(self):
        """Weighted coverage = Σ(e_i × m_i) / Σ(e_i)."""
        emphasis_map = InstructionalEmphasisMap(
            concept_scores={"A": 0.8, "B": 0.2},
            threshold_used=0.5,
            n_sentences=10,
            n_concepts=2,
        )
        # Student has A (weight 0.8) but not B (weight 0.2)
        mastery = {"A": 1.0, "B": 0.0}
        result = compute_weighted_concept_coverage(emphasis_map, mastery)
        expected = (0.8 * 1.0 + 0.2 * 0.0) / (0.8 + 0.2)
        assert result == pytest.approx(expected)

    def test_uniform_emphasis_equals_simple_mean(self):
        """Uniform emphasis → weighted coverage == simple mean."""
        emphasis_map = InstructionalEmphasisMap(
            concept_scores={"A": 0.5, "B": 0.5, "C": 0.5},
            threshold_used=0.5,
            n_sentences=10,
            n_concepts=3,
        )
        mastery = {"A": 1.0, "B": 0.0, "C": 1.0}
        result = compute_weighted_concept_coverage(emphasis_map, mastery)
        simple_mean = (1.0 + 0.0 + 1.0) / 3
        assert result == pytest.approx(simple_mean)

    def test_empty_emphasis_map(self):
        """Empty emphasis map → 0.0 coverage."""
        emphasis_map = InstructionalEmphasisMap(
            concept_scores={},
            threshold_used=0.5,
            n_sentences=0,
            n_concepts=0,
        )
        result = compute_weighted_concept_coverage(emphasis_map, {"A": 1.0})
        assert result == 0.0

    def test_missing_concept_in_mastery_treated_as_zero(self):
        """Concept in emphasis but not in mastery → treated as 0 mastery."""
        emphasis_map = InstructionalEmphasisMap(
            concept_scores={"A": 0.8, "B": 0.5},
            threshold_used=0.5,
            n_sentences=10,
            n_concepts=2,
        )
        mastery = {"A": 1.0}  # B missing
        result = compute_weighted_concept_coverage(emphasis_map, mastery)
        expected = (0.8 * 1.0 + 0.5 * 0.0) / (0.8 + 0.5)
        assert result == pytest.approx(expected)

    def test_all_zero_emphasis(self):
        """All zero emphasis weights → 0.0 coverage."""
        emphasis_map = InstructionalEmphasisMap(
            concept_scores={"A": 0.0, "B": 0.0},
            threshold_used=0.5,
            n_sentences=10,
            n_concepts=2,
        )
        result = compute_weighted_concept_coverage(emphasis_map, {"A": 1.0, "B": 1.0})
        assert result == 0.0


# ---------------------------------------------------------------------------
# FR-011/012: EnsembleScorer integration with emphasis_map
# ---------------------------------------------------------------------------


class TestEnsembleScorerEmphasisIntegration:
    """FR-011/012: EnsembleScorer.compute_score() accepts emphasis_map param."""

    def test_emphasis_map_none_backward_compat(self):
        """emphasis_map=None produces same result as before."""
        from forma.ensemble_scorer import EnsembleScorer
        from forma.evaluation_types import ConceptMatchResult

        scorer = EnsembleScorer()
        concepts = [
            ConceptMatchResult(
                concept="A", student_id="s001", question_sn=1,
                is_present=True, similarity_score=0.8,
                top_k_mean_similarity=0.8, threshold_used=0.5,
            ),
            ConceptMatchResult(
                concept="B", student_id="s001", question_sn=1,
                is_present=False, similarity_score=0.3,
                top_k_mean_similarity=0.3, threshold_used=0.5,
            ),
        ]
        result_without = scorer.compute_score(
            concept_results=concepts,
            llm_result=None,
            statistical_result=None,
            graph_result=None,
            bertscore_f1=None,
            student_id="s001",
            question_sn=1,
            emphasis_map=None,
        )
        result_default = scorer.compute_score(
            concept_results=concepts,
            llm_result=None,
            statistical_result=None,
            graph_result=None,
            bertscore_f1=None,
            student_id="s001",
            question_sn=1,
        )
        assert result_without.ensemble_score == pytest.approx(result_default.ensemble_score)

    def test_emphasis_map_modifies_concept_coverage(self):
        """emphasis_map changes concept_coverage component score."""
        from forma.ensemble_scorer import EnsembleScorer
        from forma.evaluation_types import ConceptMatchResult

        scorer = EnsembleScorer()
        concepts = [
            ConceptMatchResult(
                concept="A", student_id="s001", question_sn=1,
                is_present=True, similarity_score=0.8,
                top_k_mean_similarity=0.8, threshold_used=0.5,
            ),
            ConceptMatchResult(
                concept="B", student_id="s001", question_sn=1,
                is_present=False, similarity_score=0.3,
                top_k_mean_similarity=0.3, threshold_used=0.5,
            ),
        ]
        emphasis = InstructionalEmphasisMap(
            concept_scores={"A": 0.9, "B": 0.1},
            threshold_used=0.5,
            n_sentences=10,
            n_concepts=2,
        )
        result_with = scorer.compute_score(
            concept_results=concepts,
            llm_result=None,
            statistical_result=None,
            graph_result=None,
            bertscore_f1=None,
            student_id="s001",
            question_sn=1,
            emphasis_map=emphasis,
        )
        result_without = scorer.compute_score(
            concept_results=concepts,
            llm_result=None,
            statistical_result=None,
            graph_result=None,
            bertscore_f1=None,
            student_id="s001",
            question_sn=1,
        )
        # With emphasis, A (present, weight 0.9) dominates
        # coverage = (0.9*1 + 0.1*0) / (0.9+0.1) = 0.9
        # Without emphasis: 1/2 = 0.5
        assert result_with.component_scores["concept_coverage"] > result_without.component_scores["concept_coverage"]


# ---------------------------------------------------------------------------
# SC-002: Single concept
# ---------------------------------------------------------------------------


class TestSingleConcept:
    """SC-002: Single concept edge case."""

    @patch("forma.emphasis_map.encode_texts")
    def test_single_concept_single_sentence(self, mock_encode):
        """Single concept, single sentence → valid result."""
        dim = 384
        mock_encode.return_value = np.random.randn(2, dim).astype(np.float32)

        result = compute_emphasis_map(["문장"], ["개념"])
        assert len(result.concept_scores) == 1
        assert result.n_sentences == 1
        assert result.n_concepts == 1
        score = list(result.concept_scores.values())[0]
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# SC-003: Large number of concepts
# ---------------------------------------------------------------------------


class TestLargeConceptSet:
    """SC-003: Many concepts."""

    @patch("forma.emphasis_map.encode_texts")
    def test_many_concepts(self, mock_encode):
        """100 concepts → all scored in [0.0, 1.0]."""
        dim = 384
        n_sent = 50
        n_conc = 100
        mock_encode.return_value = np.random.randn(n_sent + n_conc, dim).astype(np.float32)

        sentences = [f"문장{i}" for i in range(n_sent)]
        concepts = [f"개념{i}" for i in range(n_conc)]

        result = compute_emphasis_map(sentences, concepts)
        assert len(result.concept_scores) == n_conc
        for score in result.concept_scores.values():
            assert 0.0 <= score <= 1.0

    @patch("forma.emphasis_map.encode_texts")
    def test_concepts_with_identical_embeddings(self, mock_encode):
        """All identical embeddings → all concept scores equal.

        With threshold-counting, identical embeddings yield cosine sim 1.0
        for every (concept, sentence) pair, so all hit counts equal n_sentences
        and max-normalization yields 1.0 for all concepts.
        """
        dim = 384
        uniform = np.ones((5, dim), dtype=np.float32)
        mock_encode.return_value = uniform

        result = compute_emphasis_map(["s1", "s2", "s3"], ["c1", "c2"])
        scores = list(result.concept_scores.values())
        # All scores must be equal
        assert scores[0] == pytest.approx(scores[1])
        # With identical embeddings all above threshold → all 1.0
        assert scores[0] == pytest.approx(1.0)

    @patch("forma.emphasis_map.encode_texts")
    def test_custom_model_name_passed(self, mock_encode):
        """model_name parameter is passed through to encode_texts."""
        dim = 384
        mock_encode.return_value = np.random.randn(4, dim).astype(np.float32)

        compute_emphasis_map(
            ["s1", "s2"],
            ["c1", "c2"],
            model_name="custom-model",
        )
        mock_encode.assert_called_once()
        call_kwargs = mock_encode.call_args
        assert call_kwargs[1].get("model_name") == "custom-model" or \
            (len(call_kwargs[0]) > 1 and call_kwargs[0][1] == "custom-model")
