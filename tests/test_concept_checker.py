"""Tests for concept_checker.py Layer 1 concept-presence detection.

RED phase: tests Korean particle selection, template building, top-k mean
similarity, adaptive threshold, and full check_concept_presence() logic.
All heavy embedding calls are mocked.
"""

import math

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.concept_checker import (
    _select_particle,
    build_concept_template,
    split_student_response,
    compute_top_k_mean_similarity,
    adaptive_threshold,
    check_concept_presence,
    check_all_concepts,
)
from src.evaluation_types import ConceptMatchResult


class TestSelectParticle:
    """Tests for _select_particle() Korean particle selection."""

    def test_consonant_final_selects_with_final(self):
        """'세포막' ends with consonant ㄱ → particle '은'."""
        result = _select_particle("세포막", "은", "는")
        assert result == "은"

    def test_vowel_final_selects_without_final(self):
        """'세포' ends with vowel → particle '는'."""
        result = _select_particle("세포", "은", "는")
        assert result == "는"

    def test_empty_word_returns_without_final(self):
        """Empty word → particle without final consonant."""
        result = _select_particle("", "은", "는")
        assert result == "는"

    def test_non_korean_char_returns_without_final(self):
        """ASCII word → particle without final consonant (fallback)."""
        result = _select_particle("DNA", "은", "는")
        assert result == "는"

    def test_eul_reul_consonant(self):
        """'인지질' ends with consonant ㄹ → particle '을'."""
        result = _select_particle("인지질", "을", "를")
        assert result == "을"

    def test_eul_reul_vowel(self):
        """'세포' ends with vowel → particle '를'."""
        result = _select_particle("세포", "을", "를")
        assert result == "를"


class TestBuildConceptTemplate:
    """Tests for build_concept_template()."""

    def test_consonant_final_uses_eun(self):
        """'세포막' → '세포막은 중요한 개념이다'."""
        result = build_concept_template("세포막")
        assert result == "세포막은 중요한 개념이다"

    def test_vowel_final_uses_neun(self):
        """'세포' → '세포는 중요한 개념이다'."""
        result = build_concept_template("세포")
        assert result == "세포는 중요한 개념이다"

    def test_template_contains_concept(self):
        """Template must embed the concept term."""
        concept = "확산"
        template = build_concept_template(concept)
        assert concept in template


class TestSplitStudentResponse:
    """Tests for split_student_response()."""

    def test_split_multiple_sentences(self):
        """Korean text with two sentences splits into two."""
        text = "세포막은 인지질 이중층으로 구성됩니다. 선택적 투과성을 가집니다."
        with patch("src.concept_checker.kss") as mock_kss:
            mock_kss.split_sentences.return_value = [
                "세포막은 인지질 이중층으로 구성됩니다.",
                "선택적 투과성을 가집니다.",
            ]
            result = split_student_response(text)
        assert len(result) == 2

    def test_kss_failure_fallback_to_full_text(self):
        """If KSS raises, returns the whole text as single sentence."""
        text = "세포막은 인지질 이중층입니다."
        with patch("src.concept_checker.kss") as mock_kss:
            mock_kss.split_sentences.side_effect = RuntimeError("kss error")
            result = split_student_response(text)
        assert result == [text]

    def test_kss_empty_result_fallback(self):
        """If KSS returns [], falls back to full text."""
        text = "짧은 답변"
        with patch("src.concept_checker.kss") as mock_kss:
            mock_kss.split_sentences.return_value = []
            result = split_student_response(text)
        assert result == [text]


class TestComputeTopKMeanSimilarity:
    """Tests for compute_top_k_mean_similarity()."""

    def test_top_k_mean_single_sentence(self):
        """With one sentence, result equals its cosine sim to concept."""
        sent_emb = np.array([[1.0, 0.0]])
        concept_emb = np.array([1.0, 0.0])
        result = compute_top_k_mean_similarity(sent_emb, concept_emb, k=2)
        assert result == pytest.approx(1.0)

    def test_top_k_mean_selects_highest(self):
        """Top-2 mean should be higher than full mean when sims vary."""
        sent_emb = np.array([
            [1.0, 0.0],   # sim=1.0 to [1,0]
            [0.0, 1.0],   # sim=0.0 to [1,0]
            [0.7, 0.7],   # sim≈0.707
        ])
        concept_emb = np.array([1.0, 0.0])
        result = compute_top_k_mean_similarity(sent_emb, concept_emb, k=2)
        expected = (1.0 + 0.7071) / 2
        assert result == pytest.approx(expected, abs=0.01)

    def test_top_k_clamped_to_n_sentences(self):
        """k > n_sentences is clamped to n_sentences."""
        sent_emb = np.array([[1.0, 0.0]])
        concept_emb = np.array([1.0, 0.0])
        result = compute_top_k_mean_similarity(sent_emb, concept_emb, k=5)
        assert result == pytest.approx(1.0)


class TestAdaptiveThreshold:
    """Tests for adaptive_threshold()."""

    def test_threshold_increases_with_more_concepts(self):
        """More concepts → higher threshold (false positive penalty)."""
        t5 = adaptive_threshold(0.45, 5)
        t10 = adaptive_threshold(0.45, 10)
        assert t10 > t5

    def test_threshold_formula(self):
        """Verify τ + α·log(M) formula numerically."""
        base, alpha, m = 0.45, 0.02, 5
        expected = base + alpha * math.log(m)
        result = adaptive_threshold(base, m, alpha)
        assert result == pytest.approx(expected)

    def test_zero_n_concepts_raises(self):
        """n_concepts=0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_concepts"):
            adaptive_threshold(0.45, 0)

    def test_negative_n_concepts_raises(self):
        """Negative n_concepts must raise ValueError."""
        with pytest.raises(ValueError, match="n_concepts"):
            adaptive_threshold(0.45, -3)


class TestCheckConceptPresence:
    """Tests for check_concept_presence()."""

    def _patch_encode(self, n_sentences: int, high_sim: bool = True):
        """Return a patch context that produces plausible embeddings."""
        dim = 4
        if high_sim:
            # All sentences aligned with concept → high similarity
            sent_embs = np.ones((n_sentences, dim), dtype=np.float32)
            sent_embs /= np.linalg.norm(sent_embs, axis=1, keepdims=True)
            concept_emb = np.ones(dim, dtype=np.float32)
            concept_emb /= np.linalg.norm(concept_emb)
        else:
            # Sentences orthogonal to concept → low similarity
            sent_embs = np.zeros((n_sentences, dim), dtype=np.float32)
            sent_embs[:, 0] = 1.0
            concept_emb = np.zeros(dim, dtype=np.float32)
            concept_emb[1] = 1.0

        all_embs = np.vstack([sent_embs, concept_emb.reshape(1, -1)])
        return all_embs

    def test_present_when_high_similarity(self):
        """Concept detected when sentences align with concept embedding."""
        all_embs = self._patch_encode(2, high_sim=True)
        with patch("src.concept_checker.encode_texts", return_value=all_embs):
            with patch(
                "src.concept_checker.split_student_response",
                return_value=["s1", "s2"],
            ):
                result = check_concept_presence(
                    "세포막은 인지질 이중층입니다.",
                    "세포막",
                    n_concepts=5,
                )
        assert isinstance(result, ConceptMatchResult)
        assert result.is_present is True

    def test_absent_when_low_similarity(self):
        """Concept not detected when sentences orthogonal to concept."""
        all_embs = self._patch_encode(2, high_sim=False)
        with patch("src.concept_checker.encode_texts", return_value=all_embs):
            with patch(
                "src.concept_checker.split_student_response",
                return_value=["s1", "s2"],
            ):
                result = check_concept_presence(
                    "물이 흐릅니다.",
                    "세포막",
                    n_concepts=5,
                )
        assert result.is_present is False

    def test_empty_student_text_raises(self):
        """Empty student_text raises ValueError."""
        with pytest.raises(ValueError, match="student_text"):
            check_concept_presence("", "세포막", n_concepts=5)

    def test_whitespace_only_text_raises(self):
        """Whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="student_text"):
            check_concept_presence("   ", "세포막", n_concepts=5)

    def test_empty_concept_raises(self):
        """Empty concept string raises ValueError."""
        with pytest.raises(ValueError, match="concept"):
            check_concept_presence("답변 텍스트", "", n_concepts=5)

    def test_result_fields_populated(self):
        """Check that returned ConceptMatchResult has non-default fields."""
        all_embs = self._patch_encode(1, high_sim=True)
        with patch("src.concept_checker.encode_texts", return_value=all_embs):
            with patch(
                "src.concept_checker.split_student_response",
                return_value=["s1"],
            ):
                result = check_concept_presence(
                    "세포막은 중요합니다.",
                    "세포막",
                    n_concepts=3,
                    base_threshold=0.45,
                )
        assert result.concept == "세포막"
        assert result.threshold_used > 0
        assert result.method == "top_k_mean"


class TestCheckAllConcepts:
    """Tests for check_all_concepts() batch check."""

    def test_returns_one_result_per_concept(self):
        """check_all_concepts returns exactly one result per concept."""
        concepts = ["세포막", "삼투", "확산"]
        dim = 4
        # 3 sentences + 3 concept templates = 6 embeddings per call
        # Mock encode_texts to return identity-like embeddings
        call_count = [0]

        def mock_encode(texts, model_name=None):
            n = len(texts)
            embs = np.eye(max(n, 1), 4, dtype=np.float32)[:n]
            return embs

        with patch("src.concept_checker.encode_texts", side_effect=mock_encode):
            with patch(
                "src.concept_checker.split_student_response",
                return_value=["s1", "s2", "s3"],
            ):
                results = check_all_concepts(
                    student_text="세포막은 삼투에 의해 물이 이동합니다.",
                    student_id="s001",
                    question_sn=1,
                    concepts=concepts,
                )
        assert len(results) == 3

    def test_student_id_and_question_sn_populated(self):
        """check_all_concepts sets student_id and question_sn on results."""

        def mock_encode(texts, model_name=None):
            n = len(texts)
            return np.ones((n, 4), dtype=np.float32)

        with patch("src.concept_checker.encode_texts", side_effect=mock_encode):
            with patch(
                "src.concept_checker.split_student_response",
                return_value=["s1"],
            ):
                results = check_all_concepts(
                    student_text="세포막",
                    student_id="s042",
                    question_sn=3,
                    concepts=["세포막"],
                )
        assert results[0].student_id == "s042"
        assert results[0].question_sn == 3
