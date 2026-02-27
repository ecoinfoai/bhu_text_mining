"""Prompt templates for LLM-as-a-Judge evaluation.

All templates are Korean-language and produce YAML-parseable outputs.
The rubric evaluation template implements the 3-score scale (1=low,
2=mid, 3=high) defined in the plan.
"""

from __future__ import annotations

from string import Template


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

RUBRIC_EVALUATION_TEMPLATE = Template(
    """\
다음은 학생의 서술형 답변입니다. 주어진 루브릭에 따라 평가해주세요.

## 질문
$question

## 학생 답변
$student_response

## 모범 답안
$model_answer

## 루브릭
- 상 (high, 3점): $rubric_high
- 중 (mid, 2점): $rubric_mid
- 하 (low, 1점): $rubric_low

## 핵심 개념 목록
$concepts

## 평가 지침
- 학생이 각 핵심 개념을 올바르게 사용했는지 확인하세요.
- 오개념(misconception)이나 사실적 오류가 있으면 명시하세요.
- 확신하지 못하는 경우 uncertain: true로 표시하세요.

다음 YAML 형식으로 정확하게 응답하세요 (다른 텍스트 없이):

```yaml
rubric_score: [1, 2, 또는 3]
rubric_label: [high, mid, 또는 low]
reasoning: [평가 근거를 2-3문장으로 설명]
misconceptions:
  - [오개념 1 (없으면 이 줄 삭제)]
uncertain: [true 또는 false]
```"""
)

CONCEPT_REASONING_TEMPLATE = Template(
    """\
다음 학생 답변에서 개념 '$concept'에 대한 이해를 평가해주세요.

학생 답변: $student_response

다음 YAML 형식으로 응답하세요:

```yaml
concept_understood: [true 또는 false]
evidence: [답변에서 개념 이해를 보여주는 구체적 부분]
explanation: [평가 이유 1-2문장]
```"""
)


# ---------------------------------------------------------------------------
# Rendering functions
# ---------------------------------------------------------------------------


def render_rubric_prompt(
    question: str,
    student_response: str,
    model_answer: str,
    rubric_high: str,
    rubric_mid: str,
    rubric_low: str,
    concepts: list[str],
) -> str:
    """Render the rubric evaluation prompt for a student response.

    Args:
        question: Exam question text.
        student_response: Student's raw answer.
        model_answer: Professor's model answer.
        rubric_high: Description of high-performance criteria.
        rubric_mid: Description of mid-performance criteria.
        rubric_low: Description of low-performance criteria.
        concepts: List of key concept terms for the question.

    Returns:
        Formatted prompt string ready for Claude API.

    Examples:
        >>> prompt = render_rubric_prompt(
        ...     question="세포막의 기능은?",
        ...     student_response="물질 이동 조절.",
        ...     model_answer="선택적 투과성을 통해 물질 이동 조절.",
        ...     rubric_high="구조+기능 완벽 기술",
        ...     rubric_mid="기능만 기술",
        ...     rubric_low="개념 혼동",
        ...     concepts=["세포막", "선택적 투과성"],
        ... )
        >>> "rubric_score" in prompt
        True
    """
    concepts_str = (
        "\n".join(f"- {c}" for c in concepts)
        if concepts
        else "- (개념 목록 없음)"
    )
    return RUBRIC_EVALUATION_TEMPLATE.substitute(
        question=question,
        student_response=student_response,
        model_answer=model_answer,
        rubric_high=rubric_high,
        rubric_mid=rubric_mid,
        rubric_low=rubric_low,
        concepts=concepts_str,
    )


def render_concept_reasoning_prompt(
    concept: str,
    student_response: str,
) -> str:
    """Render a concept-level reasoning prompt.

    Used when deeper per-concept LLM reasoning is needed beyond the
    binary Layer-1 similarity check.

    Args:
        concept: The concept term to probe.
        student_response: Student's raw answer text.

    Returns:
        Formatted prompt string.

    Examples:
        >>> p = render_concept_reasoning_prompt("삼투", "물이 이동합니다.")
        >>> "concept_understood" in p
        True
    """
    return CONCEPT_REASONING_TEMPLATE.substitute(
        concept=concept,
        student_response=student_response,
    )
