"""Tests for prompt_templates.py.

RED phase: validates template rendering, placeholder substitution,
and YAML block presence in rendered output.
"""

import pytest

from src.prompt_templates import (
    render_rubric_prompt,
    render_concept_reasoning_prompt,
    RUBRIC_EVALUATION_TEMPLATE,
    CONCEPT_REASONING_TEMPLATE,
)


class TestRenderRubricPrompt:
    """Tests for render_rubric_prompt()."""

    @pytest.fixture()
    def base_kwargs(self):
        return dict(
            question="세포막의 기능을 설명하시오.",
            student_response="세포막은 물질 이동을 조절합니다.",
            model_answer="세포막은 인지질 이중층으로 구성되어 선택적 투과성을 가집니다.",
            rubric_high="세포막 구조와 기능 모두 정확히 기술",
            rubric_mid="기능만 기술, 구조 누락",
            rubric_low="개념 혼동 또는 미기술",
            concepts=["인지질 이중층", "선택적 투과성", "세포막"],
        )

    def test_render_returns_string(self, base_kwargs):
        """render_rubric_prompt returns a non-empty string."""
        result = render_rubric_prompt(**base_kwargs)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_render_contains_question(self, base_kwargs):
        """Rendered prompt contains the question text."""
        result = render_rubric_prompt(**base_kwargs)
        assert base_kwargs["question"] in result

    def test_render_contains_student_response(self, base_kwargs):
        """Rendered prompt contains the student response."""
        result = render_rubric_prompt(**base_kwargs)
        assert base_kwargs["student_response"] in result

    def test_render_contains_model_answer(self, base_kwargs):
        """Rendered prompt contains the model answer."""
        result = render_rubric_prompt(**base_kwargs)
        assert base_kwargs["model_answer"] in result

    def test_render_contains_all_concepts(self, base_kwargs):
        """All concepts appear in the rendered prompt."""
        result = render_rubric_prompt(**base_kwargs)
        for concept in base_kwargs["concepts"]:
            assert concept in result

    def test_render_contains_rubric_levels(self, base_kwargs):
        """Rubric high/mid/low texts appear in rendered prompt."""
        result = render_rubric_prompt(**base_kwargs)
        assert base_kwargs["rubric_high"] in result
        assert base_kwargs["rubric_mid"] in result
        assert base_kwargs["rubric_low"] in result

    def test_render_contains_yaml_format_instruction(self, base_kwargs):
        """Rendered prompt instructs YAML response format."""
        result = render_rubric_prompt(**base_kwargs)
        assert "yaml" in result.lower()
        assert "rubric_score" in result

    def test_render_empty_concepts_list(self, base_kwargs):
        """Empty concepts list is handled gracefully."""
        base_kwargs["concepts"] = []
        result = render_rubric_prompt(**base_kwargs)
        assert isinstance(result, str)

    def test_render_korean_chars_preserved(self, base_kwargs):
        """Korean characters are not escaped or corrupted."""
        result = render_rubric_prompt(**base_kwargs)
        assert "세포막" in result
        assert "인지질" in result


class TestRenderConceptReasoningPrompt:
    """Tests for render_concept_reasoning_prompt()."""

    def test_render_contains_concept(self):
        """Concept name appears in the reasoning prompt."""
        result = render_concept_reasoning_prompt(
            concept="삼투", student_response="물은 농도 차이로 이동합니다."
        )
        assert "삼투" in result

    def test_render_contains_student_response(self):
        """Student response appears in the reasoning prompt."""
        resp = "세포막을 통한 물의 이동입니다."
        result = render_concept_reasoning_prompt(
            concept="삼투", student_response=resp
        )
        assert resp in result

    def test_render_contains_yaml_instruction(self):
        """Reasoning prompt instructs YAML format."""
        result = render_concept_reasoning_prompt(
            concept="확산", student_response="분자 운동."
        )
        assert "yaml" in result.lower()
        assert "concept_understood" in result
