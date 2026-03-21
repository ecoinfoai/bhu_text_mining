"""Coaching feedback generator using LLM with quantitative data input.

The LLM role is feedback writer (not scorer). All scoring is done
deterministically by the pipeline; this module only generates
human-readable coaching text.
"""

from __future__ import annotations

import logging
from typing import Optional

from forma.evaluation_types import (
    FeedbackResult,
    GraphComparisonResult,
    TripletEdge,
)
from forma.llm_provider import LLMProvider
from forma.prompt_templates import FEEDBACK_SYSTEM_INSTRUCTION, render_feedback_prompt

logger = logging.getLogger(__name__)

MAX_FEEDBACK_CHARS: int = 600

# Tier-based length targets
TIER_LENGTH_TARGETS: dict[int, int] = {
    3: 300,
    2: 400,
    1: 500,
    0: 600,
}

# Token budget for feedback generation.
# Korean ≈ 2.5-3 tokens/char; 600 chars target ≈ 1500-1800 tokens.
MAX_FEEDBACK_TOKENS: int = 2000

_REQUIRED_SECTIONS = ["[현재 상태]", "[원인]", "[학생에게 권하는 사항]"]

NEGATIVE_EXPRESSIONS: dict[str, str] = {
    "놓쳤습니다": "추가로 학습하면 좋겠습니다",
    "부족합니다": "더 보완하면 좋겠습니다",
    "언급하지 않았습니다": "추가로 다뤄보면 좋겠습니다",
    "잘못 이해하고 있습니다": "조금 다르게 이해하고 있는 부분이 있습니다",
    "오류가 있습니다": "확인이 필요한 부분이 있습니다",
    "틀렸습니다": "다시 확인해보면 좋겠습니다",
}

EMPTY_RESPONSE_FEEDBACK: str = (
    "답변이 제출되지 않았습니다. "
    "담당 교수님과 학습내용에 대해 꼭 상의하세요."
)

FALLBACK_TEMPLATES: dict[int, str] = {
    0: (
        "[현재 상태] 학습의 첫 단계에 있으며, 핵심 개념에 대한 이해를 넓혀가는 중입니다. "
        "기초적인 개념 파악을 위한 노력이 보입니다. "
        "앞으로의 학습을 통해 충분히 발전할 수 있습니다.\n"
        "[원인] 아직 핵심 개념들 사이의 관계를 충분히 파악하지 못한 부분이 있습니다. "
        "개념 간 연결 고리를 더 탐구하면 이해가 깊어질 것입니다. "
        "기본 용어의 정의부터 차근차근 정리해 나가면 좋겠습니다.\n"
        "[학생에게 권하는 사항] 교재의 핵심 개념 정리 부분을 다시 읽어보면서 "
        "각 개념이 어떻게 연결되는지 정리해 보시기 바랍니다. "
        "개념 간 관계를 그림으로 그려보면 이해에 도움이 됩니다."
    ),
    1: (
        "[현재 상태] 기본적인 개념에 대한 이해가 형성되고 있으며, 좋은 출발점을 보여주고 있습니다. "
        "일부 핵심 관계를 올바르게 파악하고 있습니다. "
        "추가적인 학습을 통해 더 큰 발전이 가능합니다.\n"
        "[원인] 개념 간의 세부적인 연결 관계에서 더 보완할 부분이 있습니다. "
        "관계의 방향성이나 인과 관계를 더 명확히 하면 좋겠습니다. "
        "핵심 메커니즘에 대한 추가 학습이 도움이 될 것입니다.\n"
        "[학생에게 권하는 사항] 이해한 개념들을 바탕으로 관계의 방향성을 다시 정리해 보세요. "
        "핵심 개념 간의 인과 관계를 중심으로 복습하면 더 깊은 이해에 도달할 수 있습니다."
    ),
    2: (
        "[현재 상태] 주요 개념들에 대해 잘 이해하고 있으며, 대부분의 관계를 정확하게 파악하고 있습니다. "
        "학습 내용에 대한 탄탄한 기반을 보여주고 있습니다. "
        "조금만 더 보완하면 매우 우수한 수준에 도달할 수 있습니다.\n"
        "[원인] 일부 세부적인 개념 연결에서 더 깊이 있는 탐구가 도움이 될 것입니다. "
        "전체적인 이해는 갖추고 있으나 정밀도를 높이면 더 좋겠습니다.\n"
        "[학생에게 권하는 사항] 현재의 좋은 이해를 바탕으로 응용 문제나 실제 사례에 적용해 보세요. "
        "심화 내용을 탐구하면 더욱 완벽한 이해에 도달할 수 있습니다. "
        "학습한 내용을 자신의 말로 다시 설명하는 연습도 추천합니다."
    ),
    3: (
        "[현재 상태] 핵심 개념과 그 관계를 매우 잘 이해하고 있으며, 우수한 학습 성과를 보여주고 있습니다. "
        "개념 간의 연결을 정확하게 파악하고 있습니다. "
        "체계적이고 논리적인 답변을 작성했습니다.\n"
        "[원인] 높은 수준의 이해를 보여주고 있어 더 깊은 탐구가 가능합니다. "
        "현재 수준을 유지하면서 심화 학습을 진행하면 좋겠습니다.\n"
        "[학생에게 권하는 사항] 학습한 내용을 실제 상황이나 다른 주제와 연결해 보세요. "
        "심화 학습이나 관련 연구 자료를 탐구하면 전문적인 이해로 발전할 수 있습니다. "
        "다른 학생들에게 설명해 보는 것도 좋은 학습 방법입니다."
    ),
}


def _format_edges(edges: list[TripletEdge]) -> str:
    """Format edge list as readable text."""
    if not edges:
        return "(없음)"
    lines = []
    for e in edges:
        lines.append(f"- {e.subject} → ({e.relation}) → {e.object}")
    return "\n".join(lines)


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at the last complete sentence within max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Find last sentence-ending punctuation
    for punct in (".", "。", "!", "?"):
        last_idx = truncated.rfind(punct)
        if last_idx > 0:
            return truncated[: last_idx + 1]
    return truncated


def _soften_tone(text: str) -> str:
    """Replace banned negative expressions with encouraging alternatives."""
    for banned, replacement in NEGATIVE_EXPRESSIONS.items():
        text = text.replace(banned, replacement)
    return text


def _validate_and_repair(text: str) -> str:
    """Validate feedback structure and attempt minor repairs.

    Checks for required sections, truncates excess sentences (>3 per section),
    and ensures the text ends with a period. Raises ValueError if a required
    section is missing (triggering retry).

    Args:
        text: Raw feedback text to validate.

    Returns:
        Repaired text.

    Raises:
        ValueError: If a required section is missing (unrecoverable without retry).
    """
    for section in _REQUIRED_SECTIONS:
        if section not in text:
            raise ValueError(f"Missing required section: {section}")

    # Truncate excess sentences per section (max 3)
    parts = []
    section_starts = []
    for section in _REQUIRED_SECTIONS:
        idx = text.index(section)
        section_starts.append((idx, section))
    section_starts.sort(key=lambda x: x[0])

    for i, (start_idx, section_name) in enumerate(section_starts):
        content_start = start_idx + len(section_name)
        if i + 1 < len(section_starts):
            content_end = section_starts[i + 1][0]
        else:
            content_end = len(text)
        content = text[content_start:content_end].strip()
        sentences = [s.strip() for s in content.split(".") if s.strip()]
        if len(sentences) > 3:
            sentences = sentences[:3]
        section_text = ". ".join(sentences)
        if section_text and not section_text.endswith("."):
            section_text += "."
        parts.append(f"{section_name} {section_text}")

    result = "\n".join(parts)

    # Ensure text ends with period
    if result and not result.rstrip().endswith("."):
        result = result.rstrip() + "."

    return result


class FeedbackGenerator:
    """Generate coaching feedback from quantitative evaluation data.

    Args:
        provider: LLM provider instance.
        max_chars: Maximum feedback length in characters (default 600).
    """

    def __init__(
        self,
        provider: LLMProvider,
        max_chars: int = MAX_FEEDBACK_CHARS,
    ) -> None:
        self._provider = provider
        self._max_chars = max_chars

    def generate(
        self,
        student_id: str,
        question_sn: int,
        question: str,
        student_response: str,
        concept_coverage: float,
        graph_comparison: Optional[GraphComparisonResult],
        tier_level: int,
        tier_label: str,
        lecture_tone: str = "",
    ) -> FeedbackResult:
        """Generate coaching feedback for a student response.

        Args:
            student_id: Student identifier.
            question_sn: Question serial number.
            question: The exam question text.
            student_response: Student's essay answer.
            concept_coverage: Concept coverage ratio [0, 1].
            graph_comparison: Graph comparison result (or None).
            tier_level: Rubric tier level (0-3).
            tier_label: Rubric tier label string.
            lecture_tone: Optional lecture tone sample.

        Returns:
            FeedbackResult with coaching text.
        """
        # Handle empty responses
        if not student_response or not student_response.strip():
            return FeedbackResult(
                student_id=student_id,
                question_sn=question_sn,
                feedback_text=EMPTY_RESPONSE_FEEDBACK,
                char_count=len(EMPTY_RESPONSE_FEEDBACK),
                data_sources_used=["empty_response_template"],
                tier_level=0,
                tier_label="미달",
            )

        # Build data for prompt
        graph_f1 = graph_comparison.f1 if graph_comparison else 0.0
        matched_count = len(graph_comparison.matched_edges) if graph_comparison else 0
        missing_count = len(graph_comparison.missing_edges) if graph_comparison else 0
        wrong_count = (
            len(graph_comparison.wrong_direction_edges) if graph_comparison else 0
        )
        missing_text = (
            _format_edges(graph_comparison.missing_edges)
            if graph_comparison
            else "(없음)"
        )
        wrong_text = (
            _format_edges(graph_comparison.wrong_direction_edges)
            if graph_comparison
            else "(없음)"
        )

        # Length guidance by tier
        target_len = TIER_LENGTH_TARGETS.get(tier_level, self._max_chars)
        length_guidance = f"이 학생은 Level {tier_level}이므로 약 {target_len}자 이내로 작성"

        data_sources = ["concept_coverage", "tier_level"]
        if graph_comparison:
            data_sources.extend(["graph_f1", "edge_analysis"])

        prompt = render_feedback_prompt(
            question=question,
            student_response=student_response,
            concept_coverage=concept_coverage,
            graph_f1=graph_f1,
            tier_level=tier_level,
            tier_label=tier_label,
            matched_count=matched_count,
            missing_count=missing_count,
            wrong_direction_count=wrong_count,
            missing_edges_text=missing_text,
            wrong_direction_text=wrong_text,
            lecture_tone=lecture_tone,
            length_guidance=length_guidance,
        )

        feedback = None
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                raw_feedback = self._provider.generate(
                    prompt,
                    max_tokens=MAX_FEEDBACK_TOKENS,
                    temperature=0.3,
                    system_instruction=FEEDBACK_SYSTEM_INSTRUCTION,
                )
                processed = _soften_tone(raw_feedback.strip())
                processed = _truncate_at_sentence(processed, self._max_chars)
                feedback = _validate_and_repair(processed)
                break
            except ValueError as ve:
                logger.warning(
                    "Feedback for %s q%d attempt %d structural failure: %s",
                    student_id, question_sn, attempt + 1, ve,
                )
                continue
            except Exception as exc:
                logger.error("Feedback generation failed: %s", exc)
                feedback = f"피드백 생성에 실패했습니다: {exc}"
                break

        if feedback is None:
            logger.warning(
                "Feedback for %s q%d using fallback template after %d failed attempts",
                student_id, question_sn, max_attempts,
            )
            feedback = FALLBACK_TEMPLATES.get(tier_level, FALLBACK_TEMPLATES[0])

        return FeedbackResult(
            student_id=student_id,
            question_sn=question_sn,
            feedback_text=feedback,
            char_count=len(feedback),
            data_sources_used=data_sources,
            tier_level=tier_level,
            tier_label=tier_label,
        )
