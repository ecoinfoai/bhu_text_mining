"""Layer 2: LLM-as-a-Judge rubric evaluation with 3-call reliability protocol.

Uses an LLM provider (Gemini by default, Claude optional) with
temperature=0.0 (note: not guaranteed deterministic across API calls)
and median-aggregates three independent calls to improve reliability.
ICC(2,1) < 0.7 triggers automatic weight down-weighting in the ensemble
layer.
"""

from __future__ import annotations

import re
import statistics
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import yaml

from forma.evaluation_types import AggregatedLLMResult, FailedCall, LLMJudgeResult
from forma.llm_provider import create_provider
from forma.prompt_templates import (
    RUBRIC_SYSTEM_INSTRUCTION,
    render_rubric_prompt,
)


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
    """Rubric evaluator backed by LLM with 3-call reliability protocol.

    Implements the 3-call median aggregation required by the plan.
    Per-student ICC(2,1) can be computed externally using
    ``compute_icc_2_1()``.

    Args:
        api_key: API key for the LLM provider.  If None, reads the
            appropriate environment variable for the selected provider.
        model: Model ID override (uses provider default if None).
        n_calls: Number of independent API calls per evaluation (default 3).
        temperature: Sampling temperature (default 0.0).
        provider: LLM provider name — ``"gemini"`` (default) or
            ``"anthropic"``.

    Raises:
        EnvironmentError: If no API key is available.

    Examples:
        >>> import os
        >>> os.environ["GOOGLE_API_KEY"] = "test-key"
        >>> evaluator = LLMEvaluator()
        >>> evaluator.n_calls
        3
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        n_calls: int = 3,
        temperature: float = 0.0,
        provider: str = "gemini",
    ) -> None:
        if n_calls < 1:
            raise ValueError(
                f"n_calls must be >= 1, got {n_calls}. "
                "Use n_calls=1 for single-call mode."
            )
        self.provider = create_provider(
            provider=provider, api_key=api_key, model=model,
        )
        self.model = self.provider.model_name
        self.n_calls = n_calls
        self.temperature = temperature
        self.failed_calls: list[FailedCall] = []

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove markdown code fences from LLM response text.

        Args:
            text: Raw LLM response that may contain ```yaml ... ``` fencing.

        Returns:
            Text with code fences stripped.
        """
        # Try fenced ```yaml ... ``` first
        pattern = r"```(?:yaml)?\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Strip leading/trailing ``` lines if regex missed
        yaml_str = text.strip()
        if yaml_str.startswith("```"):
            first_nl = yaml_str.find("\n")
            if first_nl != -1:
                yaml_str = yaml_str[first_nl + 1:]
        if yaml_str.endswith("```"):
            yaml_str = yaml_str[:-3]
        return yaml_str.strip()

    @staticmethod
    def _sanitize_yaml_str(yaml_str: str) -> str:
        """Sanitize YAML string to handle problematic quoting.

        LLM responses often contain Korean text with embedded single quotes
        (e.g. ``'생체항상성'을 '생체합산성'으로 오인함.``) which break
        YAML list parsing. This wraps bare values in double quotes.

        Args:
            yaml_str: YAML string that may have quoting issues.

        Returns:
            Sanitized YAML string.
        """
        lines = yaml_str.split("\n")
        sanitized = []
        for line in lines:
            stripped = line.lstrip()
            # Fix list items with problematic quotes: "- 'text'..." patterns
            if stripped.startswith("- '") and not stripped.startswith("- ''"):
                indent = line[: len(line) - len(stripped)]
                value = stripped[2:]  # after "- "
                # Wrap the value in double quotes, escaping existing ones
                value = value.replace('"', '\\"')
                line = f'{indent}- "{value}"'
            # Fix bare key values that start with quotes
            elif ": '" in stripped and not stripped.endswith("'"):
                colon_idx = line.index(": '")
                key_part = line[: colon_idx + 2]  # "key: "
                value = line[colon_idx + 2:]
                value = value.replace('"', '\\"')
                line = f'{key_part}"{value}"'
            sanitized.append(line)
        return "\n".join(sanitized)

    @staticmethod
    def _fallback_parse(yaml_str: str) -> dict:
        """Best-effort key-value extraction when YAML parsing fails.

        Extracts top-level ``key: value`` pairs and ``key:`` followed by
        ``- item`` lists using simple regex, so the pipeline can continue
        even with malformed YAML.

        Args:
            yaml_str: YAML-like string that failed ``yaml.safe_load``.

        Returns:
            Dict with extracted key-value pairs.

        Raises:
            ValueError: If no key-value pairs could be extracted.
        """
        result: dict = {}
        current_key: str | None = None
        current_list: list[str] | None = None

        for line in yaml_str.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # List item under a key
            if stripped.startswith("- ") and current_key is not None:
                if current_list is None:
                    current_list = []
                item = stripped[2:].strip().strip("'\"")
                current_list.append(item)
                continue

            # Flush any pending list
            if current_key is not None and current_list is not None:
                result[current_key] = current_list
                current_list = None
                current_key = None

            # Key: value pair
            kv_match = re.match(r"^(\w[\w_]*):\s*(.*)", stripped)
            if kv_match:
                key = kv_match.group(1)
                val = kv_match.group(2).strip()
                if val:
                    # Strip quotes and try numeric conversion
                    val = val.strip("'\"")
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            if val.lower() in ("true", "false"):
                                val = val.lower() == "true"
                    result[key] = val
                    current_key = None
                else:
                    # Value on next lines (list or block)
                    current_key = key

        # Flush trailing list
        if current_key is not None and current_list is not None:
            result[current_key] = current_list

        if not result:
            raise ValueError("Fallback parser extracted no key-value pairs")
        return result

    def _parse_yaml_response(self, response_text: str) -> dict:
        """Parse a YAML block from an LLM response string.

        Tries multiple strategies in order:
        1. Strip code fences and ``yaml.safe_load``
        2. Sanitize problematic quoting and retry ``yaml.safe_load``
        3. Fallback regex-based key-value extraction

        Args:
            response_text: Raw text returned by the LLM.

        Returns:
            Parsed dict.

        Raises:
            ValueError: If all parsing strategies fail.
        """
        yaml_str = self._strip_code_fence(response_text)

        # Strategy 1: direct YAML parse
        try:
            parsed = yaml.safe_load(yaml_str)
            if isinstance(parsed, dict):
                return parsed
        except yaml.YAMLError:
            pass

        # Strategy 2: sanitize quoting and retry
        try:
            sanitized = self._sanitize_yaml_str(yaml_str)
            parsed = yaml.safe_load(sanitized)
            if isinstance(parsed, dict):
                return parsed
        except yaml.YAMLError:
            pass

        # Strategy 3: fallback regex extraction
        try:
            return self._fallback_parse(yaml_str)
        except ValueError:
            pass

        raise ValueError(
            f"All YAML parsing strategies failed in "
            f"_parse_yaml_response(). "
            f"Raw response: {response_text[:300]}"
        )

    def _single_call(
        self,
        prompt: str,
        student_id: str,
        question_sn: int,
        call_index: int,
        system_instruction: Optional[str] = None,
    ) -> LLMJudgeResult:
        """Execute one API call and return a parsed LLMJudgeResult.

        Args:
            prompt: Rendered evaluation prompt (user message).
            student_id: For result stamping.
            question_sn: For result stamping.
            call_index: 1-based index (1, 2, or 3).
            system_instruction: Optional system-level instruction.

        Returns:
            LLMJudgeResult parsed from the API response.
        """
        try:
            content = self.provider.generate(
                prompt=prompt,
                max_tokens=2048,
                temperature=self.temperature,
                system_instruction=system_instruction,
            )
        except Exception as exc:
            warnings.warn(
                f"LLM API call failed (call {call_index}, "
                f"{student_id} q{question_sn}): {exc}. "
                f"Using fallback low-confidence result.",
                stacklevel=2,
            )
            self.failed_calls.append(FailedCall(
                student_id=student_id,
                question_sn=question_sn,
                call_index=call_index,
                error_type=type(exc).__name__,
                error_message=str(exc)[:200],
                prompt=prompt,
            ))
            return LLMJudgeResult(
                student_id=student_id,
                question_sn=question_sn,
                rubric_score=1,
                rubric_label="low",
                reasoning=f"[API error] {type(exc).__name__}: {str(exc)[:150]}",
                misconceptions=[],
                uncertain=True,
                call_index=call_index,
            )

        try:
            parsed = self._parse_yaml_response(content)
        except ValueError as exc:
            warnings.warn(
                f"LLM response parsing failed (call {call_index}, "
                f"{student_id} q{question_sn}): {exc}. "
                f"Using fallback low-confidence result.",
                stacklevel=2,
            )
            return LLMJudgeResult(
                student_id=student_id,
                question_sn=question_sn,
                rubric_score=1,
                rubric_label="low",
                reasoning=f"[parse error] {content[:200]}",
                misconceptions=[],
                uncertain=True,
                call_index=call_index,
            )

        misconceptions = parsed.get("misconceptions") or []
        if isinstance(misconceptions, str):
            misconceptions = [misconceptions] if misconceptions else []
        misconceptions = [
            str(m) if not isinstance(m, str) else m
            for m in misconceptions
        ]

        return LLMJudgeResult(
            student_id=student_id,
            question_sn=question_sn,
            rubric_score=int(parsed.get("rubric_score", 1)),
            rubric_label=str(parsed.get("rubric_label", "low")),
            reasoning=str(parsed.get("reasoning", "")),
            misconceptions=misconceptions,
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
        if self.n_calls == 1:
            calls.append(self._single_call(
                prompt=prompt,
                student_id=student_id,
                question_sn=question_sn,
                call_index=1,
                system_instruction=RUBRIC_SYSTEM_INSTRUCTION,
            ))
        else:
            with ThreadPoolExecutor(max_workers=self.n_calls) as executor:
                futures = {
                    executor.submit(
                        self._single_call,
                        prompt=prompt,
                        student_id=student_id,
                        question_sn=question_sn,
                        call_index=i,
                        system_instruction=RUBRIC_SYSTEM_INSTRUCTION,
                    ): i
                    for i in range(1, self.n_calls + 1)
                }
                for future in as_completed(futures):
                    calls.append(future.result())

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
            icc_value=None,  # computed per-question in pipeline
            individual_calls=calls,
        )

    def retry_failed_calls(
        self,
        results_map: dict[tuple[str, int], AggregatedLLMResult],
    ) -> int:
        """Retry previously failed LLM calls and replace fallback results.

        After the pipeline completes, call this to retry any calls that
        failed due to transient errors. Successfully retried calls replace
        the fallback (uncertain) results in-place.

        Args:
            results_map: Mapping of (student_id, question_sn) to the
                AggregatedLLMResult that should be patched.

        Returns:
            Number of calls successfully retried.
        """
        if not self.failed_calls:
            return 0

        remaining: list[FailedCall] = []
        retried = 0

        for fc in self.failed_calls:
            try:
                content = self.provider.generate(
                    prompt=fc.prompt,
                    max_tokens=2048,
                    temperature=self.temperature,
                    system_instruction=RUBRIC_SYSTEM_INSTRUCTION,
                )
                parsed = self._parse_yaml_response(content)
                misconceptions = parsed.get("misconceptions") or []
                if isinstance(misconceptions, str):
                    misconceptions = [misconceptions] if misconceptions else []
                misconceptions = [
                    str(m) if not isinstance(m, str) else m
                    for m in misconceptions
                ]

                new_result = LLMJudgeResult(
                    student_id=fc.student_id,
                    question_sn=fc.question_sn,
                    rubric_score=int(parsed.get("rubric_score", 1)),
                    rubric_label=str(parsed.get("rubric_label", "low")),
                    reasoning=str(parsed.get("reasoning", "")),
                    misconceptions=misconceptions,
                    uncertain=bool(parsed.get("uncertain", False)),
                    call_index=fc.call_index,
                )

                # Replace in aggregated result
                key = (fc.student_id, fc.question_sn)
                agg = results_map.get(key)
                if agg is not None:
                    for idx, call in enumerate(agg.individual_calls):
                        if call.call_index == fc.call_index:
                            agg.individual_calls[idx] = new_result
                            break
                    # Recompute median
                    scores = [c.rubric_score for c in agg.individual_calls]
                    agg.median_rubric_score = float(statistics.median(scores))
                    median_call = min(
                        agg.individual_calls,
                        key=lambda c: abs(c.rubric_score - agg.median_rubric_score),
                    )
                    agg.rubric_label = median_call.rubric_label
                    agg.reasoning = median_call.reasoning
                    agg.uncertain = any(c.uncertain for c in agg.individual_calls)

                retried += 1
            except Exception:
                remaining.append(fc)

        self.failed_calls = remaining
        return retried
