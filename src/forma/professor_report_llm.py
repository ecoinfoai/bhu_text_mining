"""LLM-based teaching analysis for professor reports.
This is the ONLY module allowed to import LLM clients (Constitution Principle VI).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from forma.professor_report_data import ProfessorReportData


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider used in professor report generation."""

    model_name: str

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str = "",
        max_tokens: int = 1500,
        temperature: float = 0.3,
    ) -> str: ...

logger = logging.getLogger(__name__)

# Named constants for LLM calls
_MAX_TOKENS = 1500
_TEMPERATURE = 0.3


def _render_overall_assessment_prompt(report_data: "ProfessorReportData") -> str:
    """Build overall assessment prompt with class statistics.
    Uses render_professor_overall_assessment_prompt from prompt_templates.
    """
    from forma.prompt_templates import render_professor_overall_assessment_prompt
    return render_professor_overall_assessment_prompt(report_data)


def _render_teaching_suggestions_prompt(report_data: "ProfessorReportData") -> str:
    """Build teaching suggestions prompt with class statistics and misconceptions."""
    from forma.prompt_templates import render_professor_teaching_suggestions_prompt
    return render_professor_teaching_suggestions_prompt(report_data)


def _build_fallback_overall(report_data: "ProfessorReportData") -> str:
    """Generate fallback overall assessment text using f-strings (no LLM)."""
    mean = report_data.class_ensemble_mean
    n = report_data.n_students
    n_at_risk = report_data.n_at_risk
    return (
        f"[자동 생성 요약] 전체 {n}명 학생의 종합 평균 점수는 {mean:.2f}점입니다. "
        f"위험 학생 {n_at_risk}명이 확인되었습니다. "
        f"AI 분석을 이용할 수 없어 기본 통계 요약으로 대체되었습니다."
    )


def _build_fallback_suggestions(report_data: "ProfessorReportData") -> str:
    """Generate fallback teaching suggestions text using f-strings (no LLM)."""
    n_at_risk = report_data.n_at_risk
    n = report_data.n_students
    pct = (n_at_risk / n * 100) if n > 0 else 0.0
    return (
        f"[자동 생성 제안] 위험 학생 {n_at_risk}명 ({pct:.1f}%)에 대한 개별 지도가 권장됩니다. "
        f"AI 분석을 이용할 수 없어 통계 기반 기본 제안으로 대체되었습니다."
    )


def generate_professor_analysis(provider: LLMProvider, report_data: "ProfessorReportData") -> None:
    """Generate LLM-based teaching analysis. Modifies report_data in-place.

    Args:
        provider: LLM provider with .generate(prompt, system_instruction, ...) -> str
                  and .model_name: str attribute
        report_data: ProfessorReportData to populate with analysis results
    """
    from forma.prompt_templates import PROFESSOR_ANALYSIS_SYSTEM_INSTRUCTION

    overall_failed = False
    suggestions_failed = False
    errors = []

    # Call 1: Overall assessment
    try:
        overall_prompt = _render_overall_assessment_prompt(report_data)
        raw = provider.generate(
            overall_prompt,
            system_instruction=PROFESSOR_ANALYSIS_SYSTEM_INSTRUCTION,
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
        if not raw or not raw.strip():
            raise ValueError("Empty response from LLM")
        # Strip markdown code fences if present
        overall_text = raw.strip()
        if overall_text.startswith("```"):
            lines = overall_text.split("\n")
            overall_text = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
        if not overall_text.strip():
            raise ValueError("Empty response after stripping markdown")
        report_data.overall_assessment = overall_text
    except Exception as e:
        overall_failed = True
        errors.append(f"overall_assessment: {e}")
        report_data.overall_assessment = _build_fallback_overall(report_data)
        logger.warning("Overall assessment LLM call failed: %s", e)

    # Call 2: Teaching suggestions (independent try/except)
    try:
        suggestions_prompt = _render_teaching_suggestions_prompt(report_data)
        raw = provider.generate(
            suggestions_prompt,
            system_instruction=PROFESSOR_ANALYSIS_SYSTEM_INSTRUCTION,
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
        if not raw or not raw.strip():
            raise ValueError("Empty response from LLM")
        suggestions_text = raw.strip()
        if suggestions_text.startswith("```"):
            lines = suggestions_text.split("\n")
            suggestions_text = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
        if not suggestions_text.strip():
            raise ValueError("Empty response after stripping markdown")
        report_data.teaching_suggestions = suggestions_text
    except Exception as e:
        suggestions_failed = True
        errors.append(f"teaching_suggestions: {e}")
        report_data.teaching_suggestions = _build_fallback_suggestions(report_data)
        logger.warning("Teaching suggestions LLM call failed: %s", e)

    # Set metadata
    report_data.llm_generation_failed = overall_failed or suggestions_failed
    report_data.llm_model_used = getattr(provider, "model_name", "unknown")
    report_data.llm_error_message = "; ".join(errors) if errors else None


def generate_cluster_correction(
    cluster: object,
    master_edge: object,
    provider: LLMProvider,
) -> str:
    """Generate a 1-2 sentence correction point for a misconception cluster.

    Called ONLY from CLI entry points (Constitution VI). Never from
    professor_report.py (PDF generation).

    Args:
        cluster: MisconceptionCluster with pattern, representative_error,
            and student_errors.
        master_edge: TripletEdge or None. None for CONCEPT_ABSENCE pattern.
        provider: LLM provider with .generate() method.

    Returns:
        Correction point as str. Empty string "" on failure or empty
        response. Never returns None (I2 fix).
    """
    try:
        pattern_name = (
            cluster.pattern.value
            if hasattr(cluster.pattern, "value")
            else str(cluster.pattern)
        )

        if master_edge is not None:
            edge_str = (
                f"{master_edge.subject} -> {master_edge.relation} -> {master_edge.object}"
            )
        else:
            edge_str = "없음"

        prompt = (
            f"다음 오개념 클러스터에 대한 교정 포인트를 1-2문장으로 작성해 주세요.\n\n"
            f"패턴: {pattern_name}\n"
            f"대표 오류: {cluster.representative_error}\n"
            f"마스터 엣지: {edge_str}\n"
            f"학생 수: {cluster.member_count}명\n\n"
            f"간결하고 명확한 교정 포인트를 한국어로 작성해 주세요."
        )

        raw = provider.generate(
            prompt,
            max_tokens=500,
            temperature=0.3,
        )

        if raw is None or not str(raw).strip():
            return ""

        return str(raw).strip()
    except Exception as exc:
        logger.warning("Cluster correction generation failed: %s", exc)
        return ""
