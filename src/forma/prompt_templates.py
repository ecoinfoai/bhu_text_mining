"""Prompt templates for LLM-as-a-Judge evaluation.

All templates are Korean-language and produce YAML-parseable outputs.
The rubric evaluation template implements the 3-score scale (1=low,
2=mid, 3=high) defined in the plan.
"""

from __future__ import annotations

from string import Template


# ---------------------------------------------------------------------------
# System instructions (separated from user prompts for better LLM behavior)
# ---------------------------------------------------------------------------

RUBRIC_SYSTEM_INSTRUCTION = (
    "당신은 대학교 형성평가 답안을 루브릭 기준에 따라 정확하게 평가하는 전문 평가자입니다. "
    "이 답변은 자필 작성 후 OCR 처리된 텍스트이므로 오탈자를 맥락상 이해하여 평가하세요. "
    "지시된 YAML 형식으로만 응답하고, 다른 텍스트를 추가하지 마세요. "
    "reasoning은 반드시 2-3개의 완결된 산문 문장으로 작성하세요. "
    "글머리 기호(-, *, •)나 번호 매기기(1. 2.)를 절대 사용하지 말고, 연속된 문장으로만 쓰세요."
)

FEEDBACK_SYSTEM_INSTRUCTION = (
    "당신은 대학교 형성평가 결과를 바탕으로 학생의 성장을 돕는 "
    "따뜻하고 격려적인 학습 코칭 피드백 작성자입니다. "
    "학생의 강점을 먼저 인정하고, 부족한 부분은 발전 가능성으로 표현하세요. "
    "반드시 [현재 상태], [원인], [학생에게 권하는 사항] 세 단락으로 구성하세요. "
    "각 단락은 반드시 대괄호로 감싼 제목(예: [현재 상태])으로 시작해야 합니다. "
    "각 단락은 2~3문장으로 간결하게 작성하세요. "
    "마크다운 문법(#, **, *, ```)을 사용하지 마세요. "
    "문장을 중간에 끊지 말고 반드시 마침표로 끝나는 완결된 문장을 쓰세요. "
    "다음 부정적 표현은 절대 사용하지 마세요: "
    "놓쳤습니다, 부족합니다, 언급하지 않았습니다, 잘못 이해하고 있습니다, "
    "오류가 있습니다, 틀렸습니다. "
    "대신 격려적 대안을 사용하세요: "
    "'추가로 학습하면 좋겠습니다', '더 보완하면 좋겠습니다', "
    "'추가로 다뤄보면 좋겠습니다', '조금 다르게 이해하고 있는 부분이 있습니다', "
    "'확인이 필요한 부분이 있습니다', '다시 확인해보면 좋겠습니다'. "
    "총 300~600자 이내로 작성하세요."
)


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
reasoning: "[평가 근거를 2-3문장의 연속된 산문으로 설명. 불릿이나 번호 없이 하나의 단락으로. 반드시 큰따옴표로 감싸세요]"
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
다음은 학생의 서술형 답변에 대한 평가 결과입니다.
이 데이터를 바탕으로 학생의 성장을 돕는 격려적인 코칭 피드백을 작성하세요.

## 질문
$question

## 학생 답변
<student_response>
$student_response
</student_response>

## 평가 결과
- 개념 커버리지: $concept_coverage
- 이해도 수준: $tier_label (Level $tier_level)
- 올바르게 파악한 관계 수: $matched_count
- 추가 학습이 필요한 관계 수: $missing_count
- 방향 재확인이 필요한 관계 수: $wrong_direction_count

## 추가 학습이 필요한 관계
$missing_edges_text

## 방향 재확인이 필요한 관계
$wrong_direction_text

$lecture_tone_section

## 피드백 작성 지침
1. 이 답변은 자필 작성 후 OCR 처리된 텍스트입니다. 오탈자가 포함될 수 있으므로 맥락을 고려하여 피드백하세요.
2. 학생이 잘 이해한 부분을 먼저 인정하세요.
3. 부족한 부분은 발전 가능성으로 표현하세요.
4. 구체적인 학습 방향을 제시하세요.
5. 제공된 데이터 이상의 정보를 추가하지 마시오.
6. 기술 용어(임베딩, 코사인 유사도, F1, 그래프 메트릭, 루브릭 점수 등)를 사용하지 마세요. 학생이 이해할 수 있는 자연스러운 표현만 사용하세요.
7. $length_guidance

## 출력 형식 규칙 (위반 시 무효 처리됨 — 반드시 준수)
- 마크다운 문법(#, **, *, ---, ```)을 절대 사용하지 마세요. 순수 텍스트만 작성하세요.
- 따옴표는 작은따옴표(')만 사용하세요.
- 문장을 중간에 끊지 마세요. 모든 문장은 반드시 마침표(.)로 끝나야 합니다.
- 총 $length_guidance_chars자 이내로 작성하세요.

## 출력 구조 (정확히 아래 형식으로 작성하세요)
아래 3개 단락을 순서대로 작성하세요. 각 단락의 첫 글자는 반드시 대괄호([)로 시작합니다:

[현재 상태] (학생의 현재 이해도 수준과 강점을 2~3문장으로 설명)

[원인] (부족한 부분의 원인을 격려적 톤으로 2~3문장으로 설명)

[학생에게 권하는 사항] (구체적인 학습 방향과 방법을 2~3문장으로 제안)
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
    length_guidance: str = "Level 0은 ~600자, Level 3은 ~300자로 작성",
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
    tier_char_targets = {3: 300, 2: 400, 1: 500, 0: 600}
    length_chars = tier_char_targets.get(tier_level, 600)

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


# ---------------------------------------------------------------------------
# Professor Report — LLM Analysis Templates
# ---------------------------------------------------------------------------
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from forma.professor_report_data import ProfessorReportData

PROFESSOR_ANALYSIS_SYSTEM_INSTRUCTION = """\
당신은 대학 강의 분석 전문가입니다.
교수자를 위한 수업 개선 분석 보고서를 작성합니다.
학생 개인 정보(이름, 학번 등)는 절대 포함하지 마세요.
마크다운 헤더(#, ##, ###)를 사용하지 마세요.
분석은 한국어로 작성하세요.
"""

PROFESSOR_OVERALL_ASSESSMENT_TEMPLATE = Template("""\
다음 분반 수업 데이터를 분석하여 전반적인 수업 평가를 작성하세요.

<class_data>
$class_stats_xml
</class_data>

위 데이터를 바탕으로:
1. 학생들의 전반적인 학업 성취 수준을 평가하세요
2. 두드러진 강점과 개선이 필요한 영역을 파악하세요
3. 수업 전반에 대한 종합 의견을 제시하세요

분석 결과를 명확하고 실용적인 한국어로 작성하세요.
""")

PROFESSOR_TEACHING_SUGGESTIONS_TEMPLATE = Template("""\
다음 분반 수업 데이터를 분석하여 교수법 개선 제안을 작성하세요.

<class_data>
$class_stats_xml
</class_data>

위 데이터를 바탕으로:
1. 학습 효과를 높이기 위한 구체적인 교수법을 제안하세요
2. 위험 학생 지원 방안을 제시하세요
3. 다음 수업을 위한 실질적인 개선 방향을 제안하세요

제안은 구체적이고 실행 가능한 한국어로 작성하세요.
""")


def _build_class_stats_xml(report_data: "ProfessorReportData") -> str:
    """Build XML summary of class statistics for LLM prompt.

    Includes only aggregate statistics — no individual student PII.
    """
    lines = [
        f"<class_name>{report_data.class_name}</class_name>",
        f"<subject>{report_data.subject}</subject>",
        f"<n_students>{report_data.n_students}</n_students>",
        f"<n_at_risk>{report_data.n_at_risk}</n_at_risk>",
        f"<class_ensemble_mean>{report_data.class_ensemble_mean:.3f}</class_ensemble_mean>",
        f"<class_ensemble_std>{report_data.class_ensemble_std:.3f}</class_ensemble_std>",
    ]

    # Level distribution
    if report_data.overall_level_distribution:
        lines.append("<level_distribution>")
        for level, count in report_data.overall_level_distribution.items():
            lines.append(f"  <level name='{level}'>{count}</level>")
        lines.append("</level_distribution>")

    # Per-question stats
    if report_data.question_stats:
        lines.append("<questions>")
        for qs in report_data.question_stats:
            lines.append(f"  <question sn='{qs.question_sn}'>")
            lines.append(f"    <ensemble_mean>{qs.ensemble_mean:.3f}</ensemble_mean>")
            lines.append(f"    <ensemble_std>{qs.ensemble_std:.3f}</ensemble_std>")
            if qs.misconception_frequencies:
                lines.append("    <top_misconceptions>")
                for text, freq in sorted(qs.misconception_frequencies, key=lambda x: x[1], reverse=True)[:3]:
                    lines.append(f"      <misconception freq='{freq}'>{text}</misconception>")
                lines.append("    </top_misconceptions>")
            lines.append("  </question>")
        lines.append("</questions>")

    return "\n".join(lines)


def render_professor_overall_assessment_prompt(report_data: "ProfessorReportData") -> str:
    """Render the overall assessment prompt for professor report LLM analysis.

    Uses class-level aggregate statistics only. No student PII.

    Returns:
        Formatted prompt string for LLM overall assessment.
    """
    class_stats_xml = _build_class_stats_xml(report_data)
    return PROFESSOR_OVERALL_ASSESSMENT_TEMPLATE.substitute(
        class_stats_xml=class_stats_xml,
    )


def render_professor_teaching_suggestions_prompt(report_data: "ProfessorReportData") -> str:
    """Render the teaching suggestions prompt for professor report LLM analysis.

    Uses class-level aggregate statistics only. No student PII.

    Returns:
        Formatted prompt string for LLM teaching suggestions.
    """
    class_stats_xml = _build_class_stats_xml(report_data)
    return PROFESSOR_TEACHING_SUGGESTIONS_TEMPLATE.substitute(
        class_stats_xml=class_stats_xml,
    )
