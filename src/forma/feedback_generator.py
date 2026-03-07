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

# Tier-based length targets (guidance for LLM prompt, not enforced in code)
TIER_LENGTH_TARGETS: dict[int, int] = {
    3: 500,
    2: 1000,
    1: 1500,
    0: 2000,
}

# Token budget for feedback generation.
# Korean ≈ 2.5-3 tokens/char; 2000 chars target ≈ 5000-6000 tokens.
MAX_FEEDBACK_TOKENS: int = 6000

_REQUIRED_SECTIONS = ["[평가 요약]", "[분석 결과]", "[학습 제안]"]

EMPTY_RESPONSE_FEEDBACK: str = (
    "답변이 제출되지 않았습니다. "
    "해당 주제에 대한 이해를 표현하기 위해 "
    "핵심 개념과 그 관계를 중심으로 서술해 보시기 바랍니다."
)


def _format_edges(edges: list[TripletEdge]) -> str:
    """Format edge list as readable text."""
    if not edges:
        return "(없음)"
    lines = []
    for e in edges:
        lines.append(f"- {e.subject} → ({e.relation}) → {e.object}")
    return "\n".join(lines)


class FeedbackGenerator:
    """Generate coaching feedback from quantitative evaluation data.

    Args:
        provider: LLM provider instance.
    """

    def __init__(
        self,
        provider: LLMProvider,
    ) -> None:
        self._provider = provider

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
        target_len = TIER_LENGTH_TARGETS.get(tier_level, 2000)
        length_guidance = f"이 학생은 Level {tier_level}이므로 약 {target_len}자로 작성"

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

        try:
            raw_feedback = self._provider.generate(
                prompt,
                max_tokens=MAX_FEEDBACK_TOKENS,
                temperature=0.3,
                system_instruction=FEEDBACK_SYSTEM_INSTRUCTION,
            )
            feedback = raw_feedback.strip()
        except Exception as exc:
            logger.error("Feedback generation failed: %s", exc)
            feedback = f"피드백 생성에 실패했습니다: {exc}"

        # Validate format compliance
        for section in _REQUIRED_SECTIONS:
            if section not in feedback:
                logger.warning(
                    "Feedback for %s q%d missing section: %s",
                    student_id, question_sn, section,
                )
        if feedback and feedback.rstrip()[-1:] not in ".。!?":
            logger.warning(
                "Feedback for %s q%d appears truncated (no ending punctuation)",
                student_id, question_sn,
            )

        return FeedbackResult(
            student_id=student_id,
            question_sn=question_sn,
            feedback_text=feedback,
            char_count=len(feedback),
            data_sources_used=data_sources,
            tier_level=tier_level,
            tier_label=tier_label,
        )
