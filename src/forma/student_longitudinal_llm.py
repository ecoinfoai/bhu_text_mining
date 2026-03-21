"""LLM interpretation generator for student longitudinal reports.

Generates anonymized data summaries and Korean interpretation text
via LLM provider. No student PII is ever sent to the LLM.
"""

from __future__ import annotations

import logging

from forma.student_longitudinal_data import AnonymizedStudentSummary

logger = logging.getLogger(__name__)

__all__ = [
    "build_llm_prompt",
    "generate_interpretation",
]

_SYSTEM_INSTRUCTION = (
    "당신은 대학 형성평가 데이터를 분석하는 교육 전문가입니다. "
    "주어진 학생의 수치 데이터를 바탕으로 학습 상태를 한국어로 간결하게 해석해주세요. "
    "각 섹션별로 2-3문장으로 작성하세요."
)

_SECTION_MARKERS = {
    "coverage": "[커버리지 분석]",
    "component": "[항목별 분석]",
    "position": "[상대 위치 분석]",
    "warning": "[조기 경고 분석]",
}


def build_llm_prompt(summary: AnonymizedStudentSummary) -> str:
    """Format anonymized student data into a structured Korean prompt.

    Args:
        summary: Anonymized data packet with no PII.

    Returns:
        Korean prompt string containing all numerical data and requesting
        4 interpretation sections.
    """
    lines: list[str] = []
    lines.append("## 학생 형성평가 종단 데이터 분석 요청")
    lines.append("")

    # Weekly coverage per question
    lines.append("### 주차별 개념 커버리지")
    if summary.weekly_coverage_q1:
        q1_items = [f"{w}주차: {v:.2f}" for w, v in sorted(summary.weekly_coverage_q1.items())]
        lines.append(f"- Q1: {', '.join(q1_items)}")
    if summary.weekly_coverage_q2:
        q2_items = [f"{w}주차: {v:.2f}" for w, v in sorted(summary.weekly_coverage_q2.items())]
        lines.append(f"- Q2: {', '.join(q2_items)}")
    lines.append("")

    # Weekly ensemble scores
    lines.append("### 주차별 앙상블 점수")
    if summary.weekly_ensemble:
        ensemble_items = [f"{w}주차: {v:.2f}" for w, v in sorted(summary.weekly_ensemble.items())]
        lines.append(f"- {', '.join(ensemble_items)}")
    lines.append("")

    # Component breakdown
    if summary.component_breakdown:
        lines.append("### 주차별 항목별 점수")
        for week in sorted(summary.component_breakdown.keys()):
            breakdown = summary.component_breakdown[week]
            parts = [f"{k}: {v:.2f}" for k, v in sorted(breakdown.items())]
            lines.append(f"- {week}주차: {', '.join(parts)}")
        lines.append("")

    # Percentiles
    lines.append("### 주차별 백분위")
    if summary.percentiles:
        pct_items = [f"{w}주차: {v:.1f}%ile" for w, v in sorted(summary.percentiles.items())]
        lines.append(f"- {', '.join(pct_items)}")
    lines.append("")

    # Trend
    lines.append("### 추세 정보")
    lines.append(f"- 추세 방향: {summary.trend_direction}")
    if summary.trend_slope is not None:
        lines.append(f"- OLS 기울기: {summary.trend_slope:.4f}")
    lines.append("")

    # Alert level and signals
    lines.append("### 경고 상태")
    lines.append(f"- 경고 수준: {summary.alert_level}")
    if summary.triggered_signals:
        lines.append(f"- 발동된 경고 신호: {', '.join(summary.triggered_signals)}")
    else:
        lines.append("- 발동된 경고 신호: 없음")
    lines.append("")

    # Request output format
    lines.append("### 분석 요청")
    lines.append("위 데이터를 바탕으로 다음 4개 섹션으로 해석을 작성해주세요:")
    lines.append("[커버리지 분석] — 개념 커버리지 추세 해석")
    lines.append("[항목별 분석] — 항목별 점수 분해 해석")
    lines.append("[상대 위치 분석] — 백분위 및 상대 위치 해석")
    lines.append("[조기 경고 분석] — 경고 상태 및 향후 위험 요소 해석")

    return "\n".join(lines)


def _parse_sections(text: str) -> dict[str, str | None]:
    """Parse LLM response into 4 sections by marker.

    Args:
        text: Raw LLM response text.

    Returns:
        Dict with keys "coverage", "component", "position", "warning".
        Values are stripped text or None if the section marker was not found.
    """
    result: dict[str, str | None] = {
        "coverage": None,
        "component": None,
        "position": None,
        "warning": None,
    }

    # Find positions of each marker
    marker_positions: list[tuple[int, str]] = []
    for key, marker in _SECTION_MARKERS.items():
        pos = text.find(marker)
        if pos != -1:
            marker_positions.append((pos, key))

    if not marker_positions:
        return result

    # Sort by position
    marker_positions.sort(key=lambda x: x[0])

    for i, (pos, key) in enumerate(marker_positions):
        marker = _SECTION_MARKERS[key]
        start = pos + len(marker)
        if i + 1 < len(marker_positions):
            end = marker_positions[i + 1][0]
        else:
            end = len(text)
        section_text = text[start:end].strip()
        if section_text:
            result[key] = section_text

    return result


def generate_interpretation(
    summary: AnonymizedStudentSummary,
    provider,
) -> dict[str, str | None]:
    """Generate LLM interpretation from anonymized student data.

    Args:
        summary: Anonymized data packet (no PII).
        provider: LLM provider instance with generate() method.

    Returns:
        Dict with keys "coverage", "component", "position", "warning".
        Each value is a Korean interpretation string, or None if
        generation failed or the section was not found in the response.
    """
    empty_result: dict[str, str | None] = {
        "coverage": None,
        "component": None,
        "position": None,
        "warning": None,
    }

    try:
        prompt = build_llm_prompt(summary)
        response = provider.generate(
            prompt,
            max_tokens=1024,
            temperature=0.3,
            system_instruction=_SYSTEM_INSTRUCTION,
        )
        return _parse_sections(response)
    except Exception as exc:
        logger.warning("Failed to generate LLM interpretation: %s", exc)
        return empty_result
