"""Tests for student_longitudinal_llm.py — LLM interpretation generation.

TDD RED phase: tests written before implementation.
T025: build_llm_prompt
T026: generate_interpretation
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

from forma.student_longitudinal_data import AnonymizedStudentSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary() -> AnonymizedStudentSummary:
    """Build a realistic anonymized summary with 3 weeks of data."""
    return AnonymizedStudentSummary(
        weekly_coverage_q1={1: 0.55, 2: 0.60, 3: 0.65},
        weekly_coverage_q2={1: 0.40, 2: 0.50, 3: 0.55},
        weekly_ensemble={1: 0.50, 2: 0.55, 3: 0.60},
        percentiles={1: 35.0, 2: 45.0, 3: 55.0},
        trend_slope=0.05,
        trend_direction="상승",
        alert_level="정상",
        triggered_signals=[],
        component_breakdown={
            1: {
                "concept_coverage": 0.475,
                "llm_rubric": 0.40,
                "ensemble_score": 0.50,
                "rasch_ability": -0.1,
            },
            2: {
                "concept_coverage": 0.55,
                "llm_rubric": 0.45,
                "ensemble_score": 0.55,
                "rasch_ability": 0.0,
            },
            3: {
                "concept_coverage": 0.60,
                "llm_rubric": 0.50,
                "ensemble_score": 0.60,
                "rasch_ability": 0.1,
            },
        },
    )


def _make_warning_summary() -> AnonymizedStudentSummary:
    """Build summary with active warning signals."""
    return AnonymizedStudentSummary(
        weekly_coverage_q1={1: 0.30, 2: 0.25},
        weekly_coverage_q2={1: 0.20, 2: 0.15},
        weekly_ensemble={1: 0.35, 2: 0.30},
        percentiles={1: 15.0, 2: 10.0},
        trend_slope=-0.10,
        trend_direction="하강",
        alert_level="경고",
        triggered_signals=["위험 구간 진입", "하위 백분위"],
        component_breakdown={
            1: {"concept_coverage": 0.25, "llm_rubric": 0.20, "ensemble_score": 0.35},
            2: {"concept_coverage": 0.20, "llm_rubric": 0.15, "ensemble_score": 0.30},
        },
    )


# ---------------------------------------------------------------------------
# T025: build_llm_prompt
# ---------------------------------------------------------------------------


class TestBuildLLMPrompt:
    """build_llm_prompt generates a Korean prompt from anonymized data."""

    def test_prompt_is_korean(self):
        from forma.student_longitudinal_llm import build_llm_prompt

        summary = _make_summary()
        prompt = build_llm_prompt(summary)

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        # Should contain Korean characters
        assert any("\uac00" <= ch <= "\ud7a3" for ch in prompt)

    def test_prompt_contains_no_student_id(self):
        from forma.student_longitudinal_llm import build_llm_prompt

        summary = _make_summary()
        prompt = build_llm_prompt(summary)

        # No 10-digit student ID patterns
        assert not re.search(r"\b\d{10}\b", prompt)
        # No common Korean name patterns (3-character names)
        # The summary has no PII, so the prompt should have none
        assert "s001" not in prompt.lower()
        assert "홍길동" not in prompt

    def test_prompt_contains_numerical_data(self):
        from forma.student_longitudinal_llm import build_llm_prompt

        summary = _make_summary()
        prompt = build_llm_prompt(summary)

        # Should contain coverage percentages or decimal values
        assert "0.55" in prompt or "55" in prompt
        # Should contain percentile data
        assert "35" in prompt or "45" in prompt or "55" in prompt
        # Should contain trend info
        assert "상승" in prompt

    def test_prompt_contains_section_markers(self):
        from forma.student_longitudinal_llm import build_llm_prompt

        summary = _make_summary()
        prompt = build_llm_prompt(summary)

        # Should request 4 sections
        assert "커버리지" in prompt
        assert "항목" in prompt or "분석" in prompt

    def test_prompt_with_warning_summary(self):
        from forma.student_longitudinal_llm import build_llm_prompt

        summary = _make_warning_summary()
        prompt = build_llm_prompt(summary)

        assert "경고" in prompt
        assert "위험 구간 진입" in prompt or "하위 백분위" in prompt
        assert "하강" in prompt


# ---------------------------------------------------------------------------
# T026: generate_interpretation
# ---------------------------------------------------------------------------


_MOCK_RESPONSE = """[커버리지 분석]
이 학생의 개념 커버리지는 1주차 55%에서 3주차 65%로 꾸준히 상승하고 있습니다. Q2의 커버리지도 동반 상승하여 전반적인 학습 이해도가 개선되고 있습니다.

[항목별 분석]
개념 커버리지와 LLM 루브릭 점수가 균형적으로 상승하고 있어 단순 암기가 아닌 실질적 이해도 향상을 보이고 있습니다. Rasch 능력치도 음수에서 양수로 전환되었습니다.

[상대 위치 분석]
백분위가 35에서 55로 상승하여 중상위권으로 이동하고 있습니다. 현재 추세가 유지되면 상위 30% 진입이 가능할 것으로 예상됩니다.

[조기 경고 분석]
현재 모든 경고 지표가 정상 범위 내에 있으며, 상승 추세를 감안할 때 당분간 위험 요소는 낮습니다."""


class TestGenerateInterpretation:
    """generate_interpretation calls LLM and parses response."""

    def test_returns_dict_with_correct_keys(self):
        from forma.student_longitudinal_llm import generate_interpretation

        summary = _make_summary()
        mock_provider = MagicMock()
        mock_provider.generate.return_value = _MOCK_RESPONSE

        result = generate_interpretation(summary, mock_provider)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"coverage", "component", "position", "warning"}

    def test_values_are_korean_strings(self):
        from forma.student_longitudinal_llm import generate_interpretation

        summary = _make_summary()
        mock_provider = MagicMock()
        mock_provider.generate.return_value = _MOCK_RESPONSE

        result = generate_interpretation(summary, mock_provider)

        for key, value in result.items():
            assert isinstance(value, str), f"{key} should be str"
            assert len(value) > 0, f"{key} should not be empty"
            # Each value should contain Korean
            assert any("\uac00" <= ch <= "\ud7a3" for ch in value), f"{key} should contain Korean characters"

    def test_provider_called_with_system_instruction(self):
        from forma.student_longitudinal_llm import generate_interpretation

        summary = _make_summary()
        mock_provider = MagicMock()
        mock_provider.generate.return_value = _MOCK_RESPONSE

        generate_interpretation(summary, mock_provider)

        mock_provider.generate.assert_called_once()
        call_kwargs = mock_provider.generate.call_args
        # system_instruction should be provided
        assert "system_instruction" in call_kwargs.kwargs or (
            len(call_kwargs.args) >= 4 and call_kwargs.args[3] is not None
        )

    def test_graceful_failure_on_exception(self):
        from forma.student_longitudinal_llm import generate_interpretation

        summary = _make_summary()
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = RuntimeError("API error")

        result = generate_interpretation(summary, mock_provider)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"coverage", "component", "position", "warning"}
        for value in result.values():
            assert value is None

    def test_graceful_failure_on_connection_error(self):
        from forma.student_longitudinal_llm import generate_interpretation

        summary = _make_summary()
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = ConnectionError("Network down")

        result = generate_interpretation(summary, mock_provider)

        assert all(v is None for v in result.values())

    def test_partial_response_handling(self):
        """If LLM response has only some sections, missing ones are None."""
        from forma.student_longitudinal_llm import generate_interpretation

        partial_response = """[커버리지 분석]
커버리지가 상승 추세입니다.

[조기 경고 분석]
경고 수준은 정상입니다."""

        summary = _make_summary()
        mock_provider = MagicMock()
        mock_provider.generate.return_value = partial_response

        result = generate_interpretation(summary, mock_provider)

        assert result["coverage"] is not None
        assert result["warning"] is not None
        # Missing sections should be None
        assert result["component"] is None
        assert result["position"] is None
