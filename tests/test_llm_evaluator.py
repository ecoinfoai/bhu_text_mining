"""Tests for llm_evaluator.py LLM-as-a-Judge with 3-call protocol.

RED phase: validates API key handling, YAML response parsing, 3-call
aggregation, and error paths.  All Anthropic API calls are mocked.
"""

import os
import statistics
from unittest.mock import MagicMock, patch, call

import pytest

from src.evaluation_types import LLMJudgeResult, AggregatedLLMResult
from src.llm_evaluator import LLMEvaluator, compute_icc_2_1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_anthropic():
    """Patch anthropic.Anthropic so no real API calls are made."""
    with patch("src.llm_evaluator.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_cls, mock_client


def _make_message(text: str) -> MagicMock:
    """Build a fake Anthropic message response."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


VALID_YAML_RESPONSE = """\
```yaml
rubric_score: 2
rubric_label: mid
reasoning: 학생이 기본 개념을 이해하고 있으나 세부 사항이 부족합니다.
misconceptions:
  - 삼투와 확산을 혼동
uncertain: false
```"""

HIGH_SCORE_RESPONSE = """\
```yaml
rubric_score: 3
rubric_label: high
reasoning: 세포막 구조와 기능을 완벽하게 설명하였습니다.
misconceptions: []
uncertain: false
```"""


# ---------------------------------------------------------------------------
# LLMEvaluator initialisation tests
# ---------------------------------------------------------------------------


class TestLLMEvaluatorInit:
    """Tests for LLMEvaluator.__init__()."""

    def test_init_with_explicit_api_key(self, mock_anthropic):
        """Explicit api_key is accepted without env var."""
        mock_cls, _ = mock_anthropic
        evaluator = LLMEvaluator(api_key="test-key-abc")
        mock_cls.assert_called_once_with(api_key="test-key-abc")
        assert evaluator.n_calls == 3

    def test_init_reads_env_var(self, mock_anthropic):
        """Missing api_key falls back to ANTHROPIC_API_KEY env var."""
        mock_cls, _ = mock_anthropic
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key-xyz"}):
            evaluator = LLMEvaluator()
        mock_cls.assert_called_once_with(api_key="env-key-xyz")

    def test_init_missing_key_raises(self, mock_anthropic):
        """Missing API key raises EnvironmentError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(EnvironmentError, match="API key"):
                LLMEvaluator()

    def test_init_default_n_calls(self, mock_anthropic):
        """Default number of calls is 3."""
        evaluator = LLMEvaluator(api_key="k")
        assert evaluator.n_calls == 3

    def test_init_temperature_default(self, mock_anthropic):
        """Default temperature is 0.0."""
        evaluator = LLMEvaluator(api_key="k")
        assert evaluator.temperature == 0.0


# ---------------------------------------------------------------------------
# YAML parsing tests
# ---------------------------------------------------------------------------


class TestParseYamlResponse:
    """Tests for LLMEvaluator._parse_yaml_response()."""

    @pytest.fixture()
    def evaluator(self, mock_anthropic):
        return LLMEvaluator(api_key="k")

    def test_parses_yaml_block(self, evaluator):
        """Parses YAML fenced block correctly."""
        parsed = evaluator._parse_yaml_response(VALID_YAML_RESPONSE)
        assert parsed["rubric_score"] == 2
        assert parsed["rubric_label"] == "mid"
        assert isinstance(parsed["misconceptions"], list)

    def test_parses_plain_yaml(self, evaluator):
        """Parses plain (unfenced) YAML string."""
        plain = "rubric_score: 1\nrubric_label: low\nreasoning: test\n"
        plain += "misconceptions: []\nuncertain: false\n"
        parsed = evaluator._parse_yaml_response(plain)
        assert parsed["rubric_score"] == 1

    def test_non_dict_response_raises(self, evaluator):
        """Non-dict YAML raises ValueError."""
        with pytest.raises(ValueError, match="parse"):
            evaluator._parse_yaml_response("- item1\n- item2")


# ---------------------------------------------------------------------------
# evaluate_response tests
# ---------------------------------------------------------------------------


class TestEvaluateResponse:
    """Tests for LLMEvaluator.evaluate_response() 3-call protocol."""

    @pytest.fixture()
    def evaluator(self, mock_anthropic):
        _, mock_client = mock_anthropic
        mock_client.messages.create.return_value = _make_message(
            VALID_YAML_RESPONSE
        )
        return LLMEvaluator(api_key="k")

    def _call_evaluate(self, evaluator: LLMEvaluator) -> AggregatedLLMResult:
        return evaluator.evaluate_response(
            student_id="s001",
            question_sn=1,
            question="세포막의 기능은?",
            student_response="물질 이동을 조절합니다.",
            model_answer="선택적 투과성을 통해 물질 이동을 조절합니다.",
            rubric_high="구조+기능 완벽 기술",
            rubric_mid="기능만 기술",
            rubric_low="개념 혼동",
            concepts=["세포막", "선택적 투과성"],
        )

    def test_returns_aggregated_result(self, evaluator):
        """evaluate_response returns AggregatedLLMResult."""
        result = self._call_evaluate(evaluator)
        assert isinstance(result, AggregatedLLMResult)

    def test_makes_three_api_calls(self, mock_anthropic, evaluator):
        """3-call protocol makes exactly 3 API calls."""
        _, mock_client = mock_anthropic
        mock_client.messages.create.return_value = _make_message(
            VALID_YAML_RESPONSE
        )
        self._call_evaluate(evaluator)
        assert mock_client.messages.create.call_count == 3

    def test_individual_calls_stored(self, evaluator):
        """AggregatedLLMResult stores all 3 individual calls."""
        result = self._call_evaluate(evaluator)
        assert len(result.individual_calls) == 3

    def test_call_indices_are_1_2_3(self, evaluator):
        """Individual call indices are 1, 2, 3."""
        result = self._call_evaluate(evaluator)
        indices = [c.call_index for c in result.individual_calls]
        assert sorted(indices) == [1, 2, 3]

    def test_median_score_computed(self, evaluator):
        """Median score is numeric."""
        result = self._call_evaluate(evaluator)
        assert isinstance(result.median_rubric_score, (int, float))

    def test_median_aggregation_with_mixed_scores(self, mock_anthropic):
        """Median of [2, 2, 3] == 2.0."""
        _, mock_client = mock_anthropic
        responses = [
            _make_message(VALID_YAML_RESPONSE),   # score=2
            _make_message(VALID_YAML_RESPONSE),   # score=2
            _make_message(HIGH_SCORE_RESPONSE),   # score=3
        ]
        mock_client.messages.create.side_effect = responses
        evaluator = LLMEvaluator(api_key="k")
        result = self._call_evaluate(evaluator)
        assert result.median_rubric_score == pytest.approx(2.0)

    def test_misconceptions_union_across_calls(self, mock_anthropic):
        """Misconceptions from all calls are merged (union, no duplicates)."""
        resp1 = """\
```yaml
rubric_score: 2
rubric_label: mid
reasoning: test
misconceptions:
  - 삼투 혼동
uncertain: false
```"""
        resp2 = """\
```yaml
rubric_score: 2
rubric_label: mid
reasoning: test
misconceptions:
  - 확산 오개념
uncertain: false
```"""
        _, mock_client = mock_anthropic
        mock_client.messages.create.side_effect = [
            _make_message(resp1),
            _make_message(resp2),
            _make_message(VALID_YAML_RESPONSE),
        ]
        evaluator = LLMEvaluator(api_key="k")
        result = self._call_evaluate(evaluator)
        assert "삼투 혼동" in result.misconceptions
        assert "확산 오개념" in result.misconceptions

    def test_uncertain_true_if_any_call_uncertain(self, mock_anthropic):
        """uncertain=True if any of the 3 calls flagged uncertainty."""
        uncertain_resp = """\
```yaml
rubric_score: 2
rubric_label: mid
reasoning: test
misconceptions: []
uncertain: true
```"""
        _, mock_client = mock_anthropic
        mock_client.messages.create.side_effect = [
            _make_message(VALID_YAML_RESPONSE),
            _make_message(uncertain_resp),
            _make_message(VALID_YAML_RESPONSE),
        ]
        evaluator = LLMEvaluator(api_key="k")
        result = self._call_evaluate(evaluator)
        assert result.uncertain is True

    def test_student_id_and_question_sn_preserved(self, evaluator):
        """student_id and question_sn are preserved in aggregated result."""
        result = self._call_evaluate(evaluator)
        assert result.student_id == "s001"
        assert result.question_sn == 1


# ---------------------------------------------------------------------------
# compute_icc_2_1 tests
# ---------------------------------------------------------------------------


class TestComputeICC21:
    """Tests for compute_icc_2_1() inter-rater reliability."""

    def test_perfect_agreement_icc_equals_one(self):
        """Identical ratings → ICC(2,1) = 1.0."""
        import numpy as np
        ratings = np.array([[2, 2, 2], [3, 3, 3], [1, 1, 1]], dtype=float)
        icc = compute_icc_2_1(ratings)
        assert icc == pytest.approx(1.0, abs=1e-6)

    def test_random_ratings_icc_in_range(self):
        """Random ratings → ICC in [-1, 1]."""
        import numpy as np
        rng = np.random.default_rng(42)
        ratings = rng.integers(1, 4, size=(10, 3)).astype(float)
        icc = compute_icc_2_1(ratings)
        assert -1.0 <= icc <= 1.0

    def test_icc_returns_float(self):
        """Return type is Python float."""
        import numpy as np
        ratings = np.array([[1, 2, 3], [2, 2, 2]], dtype=float)
        assert isinstance(compute_icc_2_1(ratings), float)
