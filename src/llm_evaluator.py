"""Layer 2: LLM-as-a-Judge rubric evaluation with 3-call reliability protocol.

Uses the Anthropic Claude API with temperature=0.0 (note: not guaranteed
deterministic across API calls) and median-aggregates three independent
calls to improve reliability.  ICC(2,1) < 0.7 triggers automatic weight
down-weighting in the ensemble layer.
"""

from __future__ import annotations

import os
import re
import statistics
from typing import Optional

import anthropic
import numpy as np
import yaml

from src.evaluation_types import AggregatedLLMResult, LLMJudgeResult
from src.prompt_templates import render_rubric_prompt


# ---------------------------------------------------------------------------
# Standalone statistical helpers
# ---------------------------------------------------------------------------


def compute_icc_2_1(ratings: np.ndarray) -> float:
    """Compute ICC(2,1) — two-way random, single-measures reliability.

    Used to assess consistency across the 3 LLM calls for each student-
    question pair.  If ICC < 0.7 the Layer-2 weight is down-weighted in
    the ensemble.

    Args:
        ratings: Float array of shape (n_subjects, n_raters).

    Returns:
        ICC(2,1) value in [-1, 1].

    Examples:
        >>> import numpy as np
        >>> compute_icc_2_1(np.array([[2,2,2],[3,3,3]], dtype=float))
        1.0
    """
    n, k = ratings.shape
    grand_mean = ratings.mean()

    ss_rows = k * float(np.sum((ratings.mean(axis=1) - grand_mean) ** 2))
    ss_cols = n * float(np.sum((ratings.mean(axis=0) - grand_mean) ** 2))
    ss_total = float(np.sum((ratings - grand_mean) ** 2))
    ss_err = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_err = ss_err / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0

    denominator = ms_rows + (k - 1) * ms_err
    if denominator <= 0:
        return 0.0
    return float((ms_rows - ms_err) / denominator)


# ---------------------------------------------------------------------------
# LLMEvaluator class
# ---------------------------------------------------------------------------


class LLMEvaluator:
    """Rubric evaluator backed by Claude API with 3-call reliability protocol.

    Implements the 3-call median aggregation required by the plan.
    Per-student ICC(2,1) can be computed externally using
    ``compute_icc_2_1()``.

    Args:
        api_key: Anthropic API key.  If None, reads ``ANTHROPIC_API_KEY``
            environment variable.
        model: Claude model ID (default claude-sonnet-4-6).
        n_calls: Number of independent API calls per evaluation (default 3).
        temperature: Sampling temperature (default 0.0).

    Raises:
        EnvironmentError: If no API key is available.

    Examples:
        >>> import os
        >>> os.environ["ANTHROPIC_API_KEY"] = "sk-..."
        >>> evaluator = LLMEvaluator()
        >>> evaluator.n_calls
        3
    """

    DEFAULT_MODEL: str = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        n_calls: int = 3,
        temperature: float = 0.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "Anthropic API key required in LLMEvaluator.__init__(). "
                "Set the ANTHROPIC_API_KEY environment variable or pass "
                "api_key= explicitly."
            )
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.n_calls = n_calls
        self.temperature = temperature

    def _parse_yaml_response(self, response_text: str) -> dict:
        """Parse a YAML block from an LLM response string.

        Tries to extract a ```yaml … ``` fenced block first; falls back to
        parsing the entire response as YAML.

        Args:
            response_text: Raw text returned by the LLM.

        Returns:
            Parsed dict.

        Raises:
            ValueError: If the response does not parse to a dict.
        """
        pattern = r"```yaml\s*(.*?)\s*```"
        match = re.search(pattern, response_text, re.DOTALL)
        yaml_str = match.group(1) if match else response_text

        parsed = yaml.safe_load(yaml_str)
        if not isinstance(parsed, dict):
            raise ValueError(
                f"LLM response did not parse to a dict in "
                f"_parse_yaml_response(). "
                f"Got: {response_text[:200]}"
            )
        return parsed

    def _single_call(
        self,
        prompt: str,
        student_id: str,
        question_sn: int,
        call_index: int,
    ) -> LLMJudgeResult:
        """Execute one API call and return a parsed LLMJudgeResult.

        Args:
            prompt: Rendered evaluation prompt.
            student_id: For result stamping.
            question_sn: For result stamping.
            call_index: 1-based index (1, 2, or 3).

        Returns:
            LLMJudgeResult parsed from the API response.
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        content = message.content[0].text
        parsed = self._parse_yaml_response(content)

        misconceptions = parsed.get("misconceptions") or []
        if isinstance(misconceptions, str):
            misconceptions = [misconceptions] if misconceptions else []

        return LLMJudgeResult(
            student_id=student_id,
            question_sn=question_sn,
            rubric_score=int(parsed.get("rubric_score", 1)),
            rubric_label=str(parsed.get("rubric_label", "low")),
            reasoning=str(parsed.get("reasoning", "")),
            misconceptions=list(misconceptions),
            uncertain=bool(parsed.get("uncertain", False)),
            call_index=call_index,
        )

    def evaluate_response(
        self,
        student_id: str,
        question_sn: int,
        question: str,
        student_response: str,
        model_answer: str,
        rubric_high: str,
        rubric_mid: str,
        rubric_low: str,
        concepts: list[str],
    ) -> AggregatedLLMResult:
        """Evaluate one student response with the 3-call reliability protocol.

        Makes ``self.n_calls`` independent API calls, computes the median
        rubric score, and unions misconceptions across calls.

        Args:
            student_id: Student identifier.
            question_sn: Question serial number.
            question: Exam question text.
            student_response: Student's raw answer.
            model_answer: Professor's model answer.
            rubric_high: High-performance rubric description.
            rubric_mid: Mid-performance rubric description.
            rubric_low: Low-performance rubric description.
            concepts: Key concept list for the question.

        Returns:
            AggregatedLLMResult with median score and merged metadata.
        """
        prompt = render_rubric_prompt(
            question=question,
            student_response=student_response,
            model_answer=model_answer,
            rubric_high=rubric_high,
            rubric_mid=rubric_mid,
            rubric_low=rubric_low,
            concepts=concepts,
        )

        calls: list[LLMJudgeResult] = []
        for i in range(1, self.n_calls + 1):
            result = self._single_call(
                prompt=prompt,
                student_id=student_id,
                question_sn=question_sn,
                call_index=i,
            )
            calls.append(result)

        scores = [c.rubric_score for c in calls]
        median_score = float(statistics.median(scores))

        median_call = min(calls, key=lambda c: abs(c.rubric_score - median_score))

        seen: set[str] = set()
        all_misconceptions: list[str] = []
        for c in calls:
            for m in c.misconceptions:
                if m and m not in seen:
                    seen.add(m)
                    all_misconceptions.append(m)

        uncertain = any(c.uncertain for c in calls)

        return AggregatedLLMResult(
            student_id=student_id,
            question_sn=question_sn,
            median_rubric_score=median_score,
            rubric_label=median_call.rubric_label,
            reasoning=median_call.reasoning,
            misconceptions=all_misconceptions,
            uncertain=uncertain,
            icc_value=None,
            individual_calls=calls,
        )
