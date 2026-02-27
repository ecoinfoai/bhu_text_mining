"""Tests for ensemble_scorer.py Layer 4.

RED phase: validates weight validation, scoring, classification,
and ensemble computation from mocked layer results.
"""

import pytest

from src.evaluation_types import (
    AggregatedLLMResult,
    ConceptMatchResult,
    EnsembleResult,
    GraphMetricResult,
    StatisticalResult,
)
from src.ensemble_scorer import (
    DEFAULT_WEIGHTS,
    UNDERSTANDING_THRESHOLDS,
    EnsembleScorer,
    classify_understanding_level,
    normalize_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_concept_results(
    student_id: str, question_sn: int, n: int, n_present: int
) -> list[ConceptMatchResult]:
    """Build n ConceptMatchResults with n_present being True."""
    results = []
    for i in range(n):
        results.append(
            ConceptMatchResult(
                concept=f"concept_{i}",
                student_id=student_id,
                question_sn=question_sn,
                is_present=(i < n_present),
                similarity_score=0.7 if i < n_present else 0.2,
                top_k_mean_similarity=0.7 if i < n_present else 0.2,
                threshold_used=0.45,
            )
        )
    return results


def _make_llm_result(
    student_id: str, question_sn: int, score: float
) -> AggregatedLLMResult:
    return AggregatedLLMResult(
        student_id=student_id,
        question_sn=question_sn,
        median_rubric_score=score,
        rubric_label="mid",
        reasoning="test",
        misconceptions=[],
        uncertain=False,
        icc_value=0.85,
        individual_calls=[],
    )


def _make_stat_result(student_id: str, question_sn: int) -> StatisticalResult:
    return StatisticalResult(
        student_id=student_id,
        question_sn=question_sn,
        rasch_theta=0.5,
        rasch_theta_se=0.4,
    )


def _make_graph_result(
    student_id: str, question_sn: int, recall: float
) -> GraphMetricResult:
    return GraphMetricResult(
        student_id=student_id,
        question_sn=question_sn,
        node_recall=recall,
        edge_jaccard=0.5,
        centrality_deviation=0.2,
        normalized_ged=0.3,
    )


# ---------------------------------------------------------------------------
# DEFAULT_WEIGHTS tests
# ---------------------------------------------------------------------------


class TestDefaultWeights:
    """Tests for DEFAULT_WEIGHTS constant."""

    def test_default_weights_sum_to_one(self):
        """DEFAULT_WEIGHTS must sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_default_weights_has_all_keys(self):
        """DEFAULT_WEIGHTS must contain all 5 metric keys."""
        required = {
            "concept_coverage",
            "llm_rubric",
            "rasch_ability",
            "kg_node_recall",
            "bertscore",
        }
        assert required.issubset(DEFAULT_WEIGHTS.keys())

    def test_default_weights_are_positive(self):
        """All default weights must be positive."""
        for k, v in DEFAULT_WEIGHTS.items():
            assert v > 0, f"Weight for {k!r} is not positive: {v}"


# ---------------------------------------------------------------------------
# classify_understanding_level tests
# ---------------------------------------------------------------------------


class TestClassifyUnderstandingLevel:
    """Tests for classify_understanding_level()."""

    def test_advanced_threshold(self):
        """Score ≥ 0.85 → Advanced."""
        assert classify_understanding_level(0.90) == "Advanced"
        assert classify_understanding_level(0.85) == "Advanced"

    def test_proficient_threshold(self):
        """0.65 ≤ score < 0.85 → Proficient."""
        assert classify_understanding_level(0.70) == "Proficient"
        assert classify_understanding_level(0.65) == "Proficient"

    def test_developing_threshold(self):
        """0.45 ≤ score < 0.65 → Developing."""
        assert classify_understanding_level(0.55) == "Developing"
        assert classify_understanding_level(0.45) == "Developing"

    def test_beginning_threshold(self):
        """score < 0.45 → Beginning."""
        assert classify_understanding_level(0.30) == "Beginning"
        assert classify_understanding_level(0.0) == "Beginning"

    def test_boundary_below_advanced(self):
        """0.849 → Proficient (just below 0.85)."""
        assert classify_understanding_level(0.849) == "Proficient"


# ---------------------------------------------------------------------------
# normalize_score tests
# ---------------------------------------------------------------------------


class TestNormalizeScore:
    """Tests for normalize_score()."""

    def test_normalize_already_in_range(self):
        """Score already in [0,1] passes through."""
        result = normalize_score(0.7)
        assert 0.0 <= result <= 1.0

    def test_normalize_returns_float(self):
        """Return type is float."""
        assert isinstance(normalize_score(0.5), float)

    def test_normalize_zero_score(self):
        """Score 0.0 maps to valid output."""
        result = normalize_score(0.0)
        assert 0.0 <= result <= 1.0

    def test_normalize_one_score(self):
        """Score 1.0 maps to valid output."""
        result = normalize_score(1.0)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# EnsembleScorer initialisation tests
# ---------------------------------------------------------------------------


class TestEnsembleScorerInit:
    """Tests for EnsembleScorer.__init__()."""

    def test_init_default_weights_accepted(self):
        """Default weights sum to 1.0 and are accepted."""
        scorer = EnsembleScorer()
        assert sum(scorer.weights.values()) == pytest.approx(1.0)

    def test_init_custom_weights_accepted(self):
        """Custom weights summing to 1.0 are accepted."""
        w = {
            "concept_coverage": 0.5,
            "llm_rubric": 0.2,
            "rasch_ability": 0.1,
            "kg_node_recall": 0.1,
            "bertscore": 0.1,
        }
        scorer = EnsembleScorer(weights=w)
        assert scorer.weights["concept_coverage"] == pytest.approx(0.5)

    def test_init_bad_weights_raise(self):
        """Weights not summing to 1.0 raise ValueError."""
        w = {"concept_coverage": 0.5, "other": 0.6}
        with pytest.raises(ValueError, match="sum"):
            EnsembleScorer(weights=w)


# ---------------------------------------------------------------------------
# EnsembleScorer.compute_score tests
# ---------------------------------------------------------------------------


class TestEnsembleScorerComputeScore:
    """Tests for EnsembleScorer.compute_score()."""

    @pytest.fixture()
    def scorer(self):
        return EnsembleScorer()

    def test_returns_ensemble_result(self, scorer):
        """compute_score returns an EnsembleResult."""
        concept_results = _make_concept_results("s001", 1, 5, 4)
        llm = _make_llm_result("s001", 1, 2.5)
        stat = _make_stat_result("s001", 1)
        graph = _make_graph_result("s001", 1, 0.75)

        result = scorer.compute_score(
            concept_results=concept_results,
            llm_result=llm,
            statistical_result=stat,
            graph_result=graph,
            bertscore_f1=0.72,
            student_id="s001",
            question_sn=1,
        )
        assert isinstance(result, EnsembleResult)

    def test_ensemble_score_in_zero_one(self, scorer):
        """Ensemble score is in [0, 1]."""
        concept_results = _make_concept_results("s001", 1, 5, 3)
        result = scorer.compute_score(
            concept_results=concept_results,
            llm_result=None,
            statistical_result=None,
            graph_result=None,
            bertscore_f1=None,
            student_id="s001",
            question_sn=1,
        )
        assert 0.0 <= result.ensemble_score <= 1.0

    def test_understanding_level_populated(self, scorer):
        """understanding_level is one of the 4 valid levels."""
        concept_results = _make_concept_results("s001", 1, 5, 5)
        result = scorer.compute_score(
            concept_results=concept_results,
            llm_result=None,
            statistical_result=None,
            graph_result=None,
            bertscore_f1=None,
            student_id="s001",
            question_sn=1,
        )
        valid_levels = {"Advanced", "Proficient", "Developing", "Beginning"}
        assert result.understanding_level in valid_levels

    def test_student_id_preserved(self, scorer):
        """student_id is preserved in EnsembleResult."""
        concept_results = _make_concept_results("s042", 2, 3, 2)
        result = scorer.compute_score(
            concept_results=concept_results,
            llm_result=None,
            statistical_result=None,
            graph_result=None,
            bertscore_f1=None,
            student_id="s042",
            question_sn=2,
        )
        assert result.student_id == "s042"
        assert result.question_sn == 2

    def test_high_concept_coverage_raises_score(self, scorer):
        """Perfect concept coverage → higher score than zero coverage."""
        cr_high = _make_concept_results("s001", 1, 5, 5)
        cr_low = _make_concept_results("s002", 1, 5, 0)
        r_high = scorer.compute_score(
            cr_high, None, None, None, None, "s001", 1
        )
        r_low = scorer.compute_score(
            cr_low, None, None, None, None, "s002", 1
        )
        assert r_high.ensemble_score > r_low.ensemble_score

    def test_component_scores_populated(self, scorer):
        """component_scores contains at least concept_coverage key."""
        concept_results = _make_concept_results("s001", 1, 5, 3)
        result = scorer.compute_score(
            concept_results=concept_results,
            llm_result=None,
            statistical_result=None,
            graph_result=None,
            bertscore_f1=None,
            student_id="s001",
            question_sn=1,
        )
        assert "concept_coverage" in result.component_scores

    def test_all_layers_produce_valid_result(self, scorer):
        """All four layers provided → valid EnsembleResult."""
        concept_results = _make_concept_results("s001", 1, 6, 5)
        llm = _make_llm_result("s001", 1, 3.0)
        stat = _make_stat_result("s001", 1)
        graph = _make_graph_result("s001", 1, 0.80)
        result = scorer.compute_score(
            concept_results=concept_results,
            llm_result=llm,
            statistical_result=stat,
            graph_result=graph,
            bertscore_f1=0.80,
            student_id="s001",
            question_sn=1,
        )
        assert isinstance(result, EnsembleResult)
        assert result.ensemble_score >= 0.0
