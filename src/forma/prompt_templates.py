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
- 이 답변은 자필 작성 후 OCR 처리된 텍스트입니다. 글씨체 특성이나 OCR 인식 오류로 인한 오탈자가 포함될 수 있습니다.
- 맥락상 의미를 파악할 수 있는 오탈자는 감점하지 마세요 (예: '생체합산성' → '생체항상성', '세포먹' → '세포막').
- 학생이 각 핵심 개념을 올바르게 사용했는지 확인하세요.
- 오개념(misconception)이나 사실적 오류가 있으면 명시하세요. 단, OCR 오류와 오개념을 구별하세요.
- 확신하지 못하는 경우 uncertain: true로 표시하세요.

다음 YAML 형식으로 정확하게 응답하세요 (다른 텍스트 없이):

rubric_score: [1, 2, 또는 3]
rubric_label: [high, mid, 또는 low]
reasoning: "[평가 근거를 2-3문장으로 완결된 문장으로 설명. 반드시 큰따옴표로 감싸세요]"
misconceptions:
  - "[오개념 1을 큰따옴표로 감싸세요 (없으면 이 줄 삭제)]"
uncertain: [true 또는 false]

주의사항:
- reasoning과 misconceptions 값은 반드시 큰따옴표(")로 감싸세요.
- 값 안에서 큰따옴표가 필요하면 작은따옴표(')를 사용하세요.
- 코드 블록(```)으로 감싸지 마세요. 위 형식 그대로 출력하세요.
- 문장을 중간에 끊지 마세요. 반드시 마침표로 끝나는 완결된 문장을 쓰세요."""
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


TRIPLET_EXTRACTION_TEMPLATE = Template("""\
다음 학생의 서술형 답변에서 지식 관계를 (주어, 관계, 목적어) 트리플릿으로 추출하세요.

## 질문
$question

## 마스터 개념 노드 (참고용)
$master_nodes

## 학생 답변
<student_response>
$student_response
</student_response>

## 출력 형식
다음 JSON 형식으로 정확하게 응답하세요 (다른 텍스트 없이):

```json
[
  {"subject": "주어", "relation": "관계동사", "object": "목적어"},
  ...
]
```

## 지침
- 이 답변은 자필 작성 후 OCR 처리된 텍스트입니다. 오탈자가 포함될 수 있으므로, 맥락상 의미를 파악하여 마스터 노드와 매칭하세요.
- 관계(relation)는 동사형으로 표현 (예: "감지하다", "명령하다", "구성하다")
- 학생 답변에 명시적으로 나타난 관계만 추출 (추론하지 마시오)
- 마스터 개념 노드에 해당하는 용어가 있으면 해당 표현을 사용 (OCR 오탈자를 올바른 용어로 교정하여 기록)
- 답변이 비어있거나 관계를 추출할 수 없으면 빈 배열 []을 반환
""")


FEEDBACK_GENERATION_TEMPLATE = Template("""\
다음은 학생의 서술형 답변에 대한 정량적 평가 결과입니다.
이 데이터를 바탕으로 학생에게 건설적인 학습 코칭 피드백을 작성하세요.

## 질문
$question

## 학생 답변
<student_response>
$student_response
</student_response>

## 정량 평가 결과
- 개념 커버리지: $concept_coverage
- 그래프 F1 점수: $graph_f1
- 이해도 수준: $tier_label (Level $tier_level)
- 매칭된 에지 수: $matched_count
- 누락된 에지 수: $missing_count
- 방향 오류 에지 수: $wrong_direction_count

## 누락된 관계 (학생이 언급하지 않은 핵심 관계)
$missing_edges_text

## 방향 오류 관계 (학생이 방향을 잘못 이해한 관계)
$wrong_direction_text

$lecture_tone_section

## 피드백 작성 지침
1. 이 답변은 자필 작성 후 OCR 처리된 텍스트입니다. 오탈자가 포함될 수 있으므로 맥락을 고려하여 피드백하세요.
2. 루브릭 평가결과를 해설하세요
3. 임베딩/그래프 분석 결과를 구체적으로 설명하세요
4. 학습 제안을 제시하세요
5. 제공된 데이터 이상의 정보를 추가하지 마시오
6. $length_guidance

## 출력 형식 규칙 (반드시 준수)
- 마크다운 문법(#, **, *, ---, ```)을 사용하지 마세요. 순수 텍스트만 작성하세요.
- 따옴표는 작은따옴표(')만 사용하세요. 큰따옴표("), 이중 작은따옴표(''), 꺾쇠(「」) 등 다른 따옴표를 쓰지 마세요.
- 글머리 기호는 숫자(1. 2. 3.)만 사용하세요. 불릿(-, *, •)을 쓰지 마세요.
- 반드시 다음 3개 단락 구조로 작성하세요:
  [평가 요약] 이해도 수준, 개념 커버리지, 점수를 2~3문장으로 요약
  [분석 결과] 잘한 점과 부족한 점을 구체적으로 설명 (3~5문장)
  [학습 제안] 보완할 내용과 학습 방법을 구체적으로 제안 (2~3문장)
- 각 단락은 [평가 요약], [분석 결과], [학습 제안]으로 시작하세요.
- 문장을 중간에 끊지 마세요. 반드시 완결된 문장으로 작성하세요.
- 글자 수: $length_guidance_chars자 이상 작성하세요.
""")


def render_triplet_extraction_prompt(
    question: str,
    student_response: str,
    master_nodes: list[str],
) -> str:
    """Render the triplet extraction prompt for a student response.

    Wraps student text in XML tags for prompt injection defense.

    Args:
        question: Exam question text.
        student_response: Student's raw answer (wrapped in XML tags).
        master_nodes: List of master concept node names for guidance.

    Returns:
        Formatted prompt string.
    """
    nodes_str = (
        "\n".join(f"- {n}" for n in master_nodes)
        if master_nodes
        else "- (노드 목록 없음)"
    )
    return TRIPLET_EXTRACTION_TEMPLATE.substitute(
        question=question,
        student_response=student_response,
        master_nodes=nodes_str,
    )


def render_feedback_prompt(
    question: str,
    student_response: str,
    concept_coverage: float,
    graph_f1: float,
    tier_level: int,
    tier_label: str,
    matched_count: int,
    missing_count: int,
    wrong_direction_count: int,
    missing_edges_text: str,
    wrong_direction_text: str,
    lecture_tone: str = "",
    length_guidance: str = "Level 0은 ~2000자, Level 3은 ~500자로 작성",
) -> str:
    """Render the feedback generation prompt.

    Args:
        question: Exam question text.
        student_response: Student's raw answer.
        concept_coverage: Concept coverage ratio.
        graph_f1: Graph comparison F1 score.
        tier_level: Rubric tier level (0-3).
        tier_label: Rubric tier label.
        matched_count: Number of matched edges.
        missing_count: Number of missing edges.
        wrong_direction_count: Number of wrong direction edges.
        missing_edges_text: Formatted text of missing edges.
        wrong_direction_text: Formatted text of wrong direction edges.
        lecture_tone: Optional lecture tone sample.
        length_guidance: Feedback length guidance string.

    Returns:
        Formatted prompt string.
    """
    lecture_section = ""
    if lecture_tone:
        lecture_section = f"\n## 강의 톤 참조\n{lecture_tone}\n"

    # Compute minimum character count for the prompt
    tier_char_targets = {3: 500, 2: 1000, 1: 1500, 0: 2000}
    length_chars = tier_char_targets.get(tier_level, 2000)

    return FEEDBACK_GENERATION_TEMPLATE.substitute(
        question=question,
        student_response=student_response,
        concept_coverage=f"{concept_coverage:.0%}",
        graph_f1=f"{graph_f1:.2f}",
        tier_level=tier_level,
        tier_label=tier_label,
        matched_count=matched_count,
        missing_count=missing_count,
        wrong_direction_count=wrong_direction_count,
        missing_edges_text=missing_edges_text or "(없음)",
        wrong_direction_text=wrong_direction_text or "(없음)",
        lecture_tone_section=lecture_section,
        length_guidance=length_guidance,
        length_guidance_chars=length_chars,
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


# ---------------------------------------------------------------------------
# Lecture triplet extraction
# ---------------------------------------------------------------------------

LECTURE_TRIPLET_EXTRACTION_TEMPLATE = Template("""\
다음 강의 내용에서 핵심 지식을 (주어, 관계, 목적어) 트리플릿으로 추출하세요.

## 강의 내용
$lecture_text

## 출력 형식
다음 JSON 형식으로 응답하세요 (다른 텍스트 없이):

```json
[
  {"subject": "주어", "relation": "관계동사", "object": "목적어"},
  ...
]
```

## 지침
- 관계는 동사형으로 표현 (예: "구성하다", "조절하다", "포함하다")
- 핵심 개념 간의 관계만 추출
- 최대 15개 트리플릿
""")


def render_lecture_triplet_prompt(lecture_text: str) -> str:
    """Render the lecture triplet extraction prompt."""
    return LECTURE_TRIPLET_EXTRACTION_TEMPLATE.substitute(lecture_text=lecture_text)
