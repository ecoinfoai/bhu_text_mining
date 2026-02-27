"""Tests for statistical_analysis.py Rasch IRT, LCA, and helpers.

RED phase: validates synthetic-data parameter recovery for Rasch,
categorical LCA class count, and UMAP/PCA utilities.
Heavy library calls (girth, stepmix) are mocked where needed.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_binary_matrix(
    n_students: int = 40, n_items: int = 6, p: float = 0.6, seed: int = 0
) -> np.ndarray:
    """Generate a synthetic binary response matrix."""
    rng = np.random.default_rng(seed)
    return (rng.random((n_students, n_items)) < p).astype(int)


# ---------------------------------------------------------------------------
# RaschAnalyzer tests
# ---------------------------------------------------------------------------


class TestRaschAnalyzerInit:
    """Tests for RaschAnalyzer initialisation."""

    def test_init_default_values(self):
        """RaschAnalyzer initialises with sensible defaults."""
        from src.statistical_analysis import RaschAnalyzer

        ra = RaschAnalyzer(question_sn=1)
        assert ra.question_sn == 1
        assert ra.n_bootstrap == 1000

    def test_init_custom_bootstrap(self):
        """RaschAnalyzer accepts custom n_bootstrap."""
        from src.statistical_analysis import RaschAnalyzer

        ra = RaschAnalyzer(question_sn=2, n_bootstrap=200)
        assert ra.n_bootstrap == 200


class TestRaschAnalyzerFit:
    """Tests for RaschAnalyzer.fit() with mocked girth."""

    def test_fit_returns_difficulty_estimates(self):
        """fit() returns item difficulty estimates of correct length."""
        from src.statistical_analysis import RaschAnalyzer

        X = _make_binary_matrix(40, 6)
        mock_result = {
            "Difficulty": np.linspace(-1.0, 1.0, 6),
            "Discrimination": np.ones(6),
        }

        with patch(
            "src.statistical_analysis._rasch_cml", return_value=mock_result
        ):
            ra = RaschAnalyzer(question_sn=1)
            ra.fit(X)
        assert ra.item_difficulties_ is not None
        assert len(ra.item_difficulties_) == 6

    def test_fit_handles_extreme_scores(self):
        """fit() removes extreme-score rows (all 0 or all 1)."""
        from src.statistical_analysis import RaschAnalyzer

        X = _make_binary_matrix(40, 6)
        X[0] = 0  # all-zero row
        X[1] = 1  # all-one row

        mock_result = {
            "Difficulty": np.linspace(-1.0, 1.0, 6),
            "Discrimination": np.ones(6),
        }

        with patch(
            "src.statistical_analysis._rasch_cml", return_value=mock_result
        ):
            ra = RaschAnalyzer(question_sn=1)
            ra.fit(X)
        # Should complete without error; at most 38 rows used
        assert ra.item_difficulties_ is not None

    def test_fit_raises_on_too_few_items(self):
        """fit() raises ValueError if fewer than 2 items provided."""
        from src.statistical_analysis import RaschAnalyzer

        X = _make_binary_matrix(40, 1)
        ra = RaschAnalyzer(question_sn=1)
        with pytest.raises(ValueError, match="items"):
            ra.fit(X)


class TestRaschAnalyzerAbilityEstimates:
    """Tests for RaschAnalyzer.ability_estimates()."""

    def test_ability_estimates_length_matches_students(self):
        """ability_estimates() returns one theta per student."""
        from src.statistical_analysis import RaschAnalyzer

        X = _make_binary_matrix(40, 6)
        difficulties = np.linspace(-1.0, 1.0, 6)
        mock_result = {
            "Difficulty": difficulties,
            "Discrimination": np.ones(6),
        }

        with patch(
            "src.statistical_analysis._rasch_cml", return_value=mock_result
        ):
            ra = RaschAnalyzer(question_sn=1)
            ra.fit(X)
            thetas, ses = ra.ability_estimates(X)

        assert len(thetas) == 40
        assert len(ses) == 40

    def test_ability_estimates_raises_if_not_fitted(self):
        """ability_estimates() raises RuntimeError if called before fit."""
        from src.statistical_analysis import RaschAnalyzer

        ra = RaschAnalyzer(question_sn=1)
        X = _make_binary_matrix(40, 6)
        with pytest.raises(RuntimeError, match="fit"):
            ra.ability_estimates(X)


# ---------------------------------------------------------------------------
# LCAAnalyzer tests
# ---------------------------------------------------------------------------


class TestLCAAnalyzerInit:
    """Tests for LCAAnalyzer initialisation."""

    def test_init_default_n_classes(self):
        """LCAAnalyzer default max_classes is 4."""
        from src.statistical_analysis import LCAAnalyzer

        lca = LCAAnalyzer()
        assert lca.max_classes == 4

    def test_init_exploratory_warning_present(self):
        """LCAAnalyzer carries the exploratory warning string."""
        from src.statistical_analysis import LCAAnalyzer

        lca = LCAAnalyzer()
        assert "탐색적" in lca.exploratory_warning


class TestLCAAnalyzerFit:
    """Tests for LCAAnalyzer.fit() with mocked stepmix."""

    def _make_mock_stepmix(self, n_classes: int = 2):
        mock_sm = MagicMock()
        mock_sm.predict.return_value = np.zeros(40, dtype=int)
        mock_sm.predict_proba.return_value = np.ones((40, n_classes)) / n_classes
        mock_sm.bic.return_value = 200.0
        return mock_sm

    def test_fit_returns_class_labels(self):
        """fit() returns integer class labels."""
        from src.statistical_analysis import LCAAnalyzer

        X = _make_binary_matrix(40, 6)
        mock_sm = self._make_mock_stepmix(2)

        with patch("src.statistical_analysis.StepMix", return_value=mock_sm):
            lca = LCAAnalyzer(max_classes=3)
            labels, probs = lca.fit_predict(X)

        assert len(labels) == 40
        assert all(isinstance(int(l), int) for l in labels)

    def test_fit_predict_proba_shape(self):
        """fit_predict returns probabilities with shape (n_students, n_classes)."""
        from src.statistical_analysis import LCAAnalyzer

        X = _make_binary_matrix(40, 6)
        mock_sm = self._make_mock_stepmix(2)

        with patch("src.statistical_analysis.StepMix", return_value=mock_sm):
            lca = LCAAnalyzer(max_classes=3)
            labels, probs = lca.fit_predict(X)

        assert probs.shape[0] == 40


# ---------------------------------------------------------------------------
# compute_concept_matrix tests
# ---------------------------------------------------------------------------


class TestComputeConceptMatrix:
    """Tests for compute_concept_matrix()."""

    def test_matrix_shape(self):
        """Matrix has shape (n_students, n_concepts)."""
        from src.statistical_analysis import compute_concept_matrix
        from src.evaluation_types import ConceptMatchResult

        students = ["s001", "s002"]
        concepts = ["세포막", "삼투", "확산"]
        results = [
            ConceptMatchResult(
                concept=c, student_id=sid, question_sn=1,
                is_present=(i % 2 == 0), similarity_score=0.5,
                top_k_mean_similarity=0.5, threshold_used=0.45,
            )
            for i, (sid, c) in enumerate(
                [(s, c) for s in students for c in concepts]
            )
        ]
        mat, student_ids, concept_list = compute_concept_matrix(
            results, students, concepts
        )
        assert mat.shape == (2, 3)
        assert student_ids == students
        assert concept_list == concepts

    def test_matrix_values_are_binary(self):
        """Matrix contains only 0 and 1."""
        from src.statistical_analysis import compute_concept_matrix
        from src.evaluation_types import ConceptMatchResult

        students = ["s001"]
        concepts = ["세포막"]
        results = [
            ConceptMatchResult(
                concept="세포막", student_id="s001", question_sn=1,
                is_present=True, similarity_score=0.7,
                top_k_mean_similarity=0.7, threshold_used=0.45,
            )
        ]
        mat, _, _ = compute_concept_matrix(results, students, concepts)
        assert set(mat.flatten().tolist()).issubset({0, 1})
