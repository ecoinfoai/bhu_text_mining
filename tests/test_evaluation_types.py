"""Tests for evaluation_types.py shared dataclasses.

RED phase: ensures dataclasses exist with correct fields and types.
"""

import pytest
from src.evaluation_types import (
    ConceptMatchResult,
    LLMJudgeResult,
    AggregatedLLMResult,
    StatisticalResult,
    GraphMetricResult,
    EnsembleResult,
)


class TestConceptMatchResult:
    """Tests for ConceptMatchResult dataclass."""

    def test_concept_match_result_creation_success(self):
        """Test successful ConceptMatchResult instantiation."""
        result = ConceptMatchResult(
            concept="세포막",
            student_id="s001",
            question_sn=1,
            is_present=True,
            similarity_score=0.72,
            top_k_mean_similarity=0.72,
            threshold_used=0.45,
        )
        assert result.concept == "세포막"
        assert result.student_id == "s001"
        assert result.question_sn == 1
        assert result.is_present is True
        assert result.similarity_score == pytest.approx(0.72)
        assert result.threshold_used == pytest.approx(0.45)
        assert result.method == "top_k_mean"

    def test_concept_match_result_default_method(self):
        """Test default method is 'top_k_mean'."""
        result = ConceptMatchResult(
            concept="삼투",
            student_id="s002",
            question_sn=2,
            is_present=False,
            similarity_score=0.30,
            top_k_mean_similarity=0.30,
            threshold_used=0.47,
        )
        assert result.method == "top_k_mean"

    def test_concept_match_result_fields_immutable_via_dataclass(self):
        """Test that all expected fields are present."""
        result = ConceptMatchResult(
            concept="확산",
            student_id="s003",
            question_sn=3,
            is_present=True,
            similarity_score=0.65,
            top_k_mean_similarity=0.65,
            threshold_used=0.46,
            method="custom",
        )
        assert result.method == "custom"


class TestLLMJudgeResult:
    """Tests for LLMJudgeResult dataclass."""

    def test_llm_judge_result_creation_success(self):
        """Test successful LLMJudgeResult instantiation."""
        result = LLMJudgeResult(
            student_id="s001",
            question_sn=1,
            rubric_score=2,
            rubric_label="mid",
            reasoning="세포막의 기본 기능을 서술했으나 선택적 투과성 언급 없음.",
            misconceptions=["삼투를 확산과 혼동"],
            uncertain=False,
            call_index=1,
        )
        assert result.rubric_score == 2
        assert result.rubric_label == "mid"
        assert result.call_index == 1
        assert result.uncertain is False
        assert len(result.misconceptions) == 1

    def test_llm_judge_result_uncertain_flag(self):
        """Test uncertain flag can be set to True."""
        result = LLMJudgeResult(
            student_id="s002",
            question_sn=2,
            rubric_score=1,
            rubric_label="low",
            reasoning="판단 불가.",
            misconceptions=[],
            uncertain=True,
            call_index=2,
        )
        assert result.uncertain is True
        assert result.misconceptions == []


class TestAggregatedLLMResult:
    """Tests for AggregatedLLMResult dataclass."""

    def _make_call(self, score: int, idx: int) -> LLMJudgeResult:
        return LLMJudgeResult(
            student_id="s001",
            question_sn=1,
            rubric_score=score,
            rubric_label="mid" if score == 2 else "high",
            reasoning="test",
            misconceptions=[],
            uncertain=False,
            call_index=idx,
        )

    def test_aggregated_llm_result_creation_success(self):
        """Test AggregatedLLMResult stores 3 individual calls."""
        calls = [self._make_call(s, i) for s, i in [(2, 1), (2, 2), (3, 3)]]
        agg = AggregatedLLMResult(
            student_id="s001",
            question_sn=1,
            median_rubric_score=2.0,
            rubric_label="mid",
            reasoning="세포막 기능 부분 이해.",
            misconceptions=["확산/삼투 혼동"],
            uncertain=False,
            icc_value=0.85,
            individual_calls=calls,
        )
        assert agg.median_rubric_score == pytest.approx(2.0)
        assert len(agg.individual_calls) == 3
        assert agg.icc_value == pytest.approx(0.85)

    def test_aggregated_llm_result_none_icc(self):
        """Test AggregatedLLMResult with icc_value=None."""
        agg = AggregatedLLMResult(
            student_id="s001",
            question_sn=1,
            median_rubric_score=3.0,
            rubric_label="high",
            reasoning="완벽한 이해.",
            misconceptions=[],
            uncertain=False,
            icc_value=None,
            individual_calls=[],
        )
        assert agg.icc_value is None


class TestStatisticalResult:
    """Tests for StatisticalResult dataclass."""

    def test_statistical_result_all_none_defaults(self):
        """Test StatisticalResult default None fields."""
        result = StatisticalResult(student_id="s001", question_sn=1)
        assert result.rasch_theta is None
        assert result.rasch_theta_se is None
        assert result.lca_class is None
        assert result.lca_class_probability is None

    def test_statistical_result_with_values(self):
        """Test StatisticalResult with Rasch and LCA values."""
        result = StatisticalResult(
            student_id="s001",
            question_sn=1,
            rasch_theta=0.45,
            rasch_theta_se=0.41,
            rasch_item_difficulty=-0.2,
            lca_class=1,
            lca_class_probability=0.78,
        )
        assert result.rasch_theta == pytest.approx(0.45)
        assert result.lca_class == 1

    def test_statistical_result_exploratory_warning_present(self):
        """Test that exploratory warning string is non-empty by default."""
        result = StatisticalResult(student_id="s002", question_sn=2)
        assert len(result.lca_exploratory_warning) > 0
        assert "탐색적" in result.lca_exploratory_warning


class TestGraphMetricResult:
    """Tests for GraphMetricResult dataclass."""

    def test_graph_metric_result_creation_success(self):
        """Test GraphMetricResult with all metric values."""
        result = GraphMetricResult(
            student_id="s001",
            question_sn=1,
            node_recall=0.75,
            edge_jaccard=0.50,
            centrality_deviation=0.30,
            normalized_ged=0.40,
        )
        assert result.node_recall == pytest.approx(0.75)
        assert result.edge_jaccard == pytest.approx(0.50)
        assert result.centrality_deviation == pytest.approx(0.30)

    def test_graph_metric_result_none_ged(self):
        """Test GraphMetricResult with None normalized_ged (timeout)."""
        result = GraphMetricResult(
            student_id="s001",
            question_sn=1,
            node_recall=0.80,
            edge_jaccard=0.60,
            centrality_deviation=0.20,
            normalized_ged=None,
        )
        assert result.normalized_ged is None


class TestEnsembleResult:
    """Tests for EnsembleResult dataclass."""

    def test_ensemble_result_creation_success(self):
        """Test EnsembleResult with all fields."""
        weights = {
            "concept_coverage": 0.35,
            "llm_rubric": 0.30,
            "rasch_ability": 0.15,
            "kg_node_recall": 0.10,
            "bertscore": 0.10,
        }
        scores = {
            "concept_coverage": 0.80,
            "llm_rubric": 0.67,
            "rasch_ability": 0.60,
            "kg_node_recall": 0.75,
            "bertscore": 0.72,
        }
        result = EnsembleResult(
            student_id="s001",
            question_sn=1,
            ensemble_score=0.73,
            understanding_level="Proficient",
            component_scores=scores,
            weights_used=weights,
        )
        assert result.ensemble_score == pytest.approx(0.73)
        assert result.understanding_level == "Proficient"
        assert "concept_coverage" in result.component_scores
        assert "concept_coverage" in result.weights_used
