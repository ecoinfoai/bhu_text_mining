"""Pedagogy analysis for instructor speech patterns.

Analyzes lecture transcripts to identify habitual expressions
(filler words to reduce) and effective pedagogy patterns
(analogies, examples, engagement), completely separate from
domain concept analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "HabitualExpression",
    "EffectivePattern",
    "PedagogyAnalysis",
    "build_pedagogy_prompt",
    "analyze_pedagogy_llm",
]


# ----------------------------------------------------------------
# Dataclasses (T043)
# ----------------------------------------------------------------


@dataclass
class HabitualExpression:
    """A frequently repeated filler expression.

    Attributes:
        expression: The repeated phrase (e.g. "여러분").
        frequency_per_minute: Occurrences per minute of lecture.
        total_count: Total occurrences.
        recommendation: "사용 자제 권장" or "정상 범위".
    """

    expression: str
    frequency_per_minute: float
    total_count: int
    recommendation: str


@dataclass
class EffectivePattern:
    """An effective pedagogy pattern found in the lecture.

    Attributes:
        pattern_type: Category ("비유/유추", "임상 사례",
            "학생 질문 유도", "일상 경험 연결").
        count: Number of occurrences.
        examples: Representative example sentences (max 3).
    """

    pattern_type: str
    count: int
    examples: list[str] = field(default_factory=list)


@dataclass
class PedagogyAnalysis:
    """Complete pedagogy analysis for a section.

    Attributes:
        section_id: Class section identifier.
        habitual_expressions: Top N overused filler expressions.
        effective_patterns: Identified effective pedagogy patterns.
        domain_ratio: Estimated ratio of domain explanation to total.
    """

    section_id: str
    habitual_expressions: list[HabitualExpression] = field(
        default_factory=list,
    )
    effective_patterns: list[EffectivePattern] = field(
        default_factory=list,
    )
    domain_ratio: float = 0.0


# ----------------------------------------------------------------
# Prompt construction (T044)
# ----------------------------------------------------------------

_PEDAGOGY_SYSTEM_INSTRUCTION = (
    "당신은 대학 강의 화법 분석 전문가입니다. 도메인 지식이 아닌 교수자의 화법 패턴만 분석해주세요."
)

_PEDAGOGY_PROMPT_TEMPLATE = """\
아래 강의 녹취에서 교수자의 화법 패턴을 분석해주세요.
도메인 전문 용어(해부학, 생리학 개념)는 분석에서 제외하고,
교수자의 습관적 표현과 효과적인 교수법 패턴만 분석합니다.

## 분석 항목

### 1. 습관적 표현 (TOP 5)
강의 중 반복되는 비전문 표현 (예: "여러분", "보시면", "그래서", "이거", "자")
각 표현에 대해:
- expression: 표현
- total_count: 총 횟수
- recommendation: "사용 자제 권장" 또는 "정상 범위"

### 2. 효과적 교수법 패턴
다음 유형의 패턴을 찾아주세요:
- "비유/유추": 개념을 비유나 유추로 설명한 부분
- "임상 사례": 임상 또는 실생활 사례를 든 부분
- "학생 질문 유도": 학생에게 질문하거나 참여를 유도한 부분
- "일상 경험 연결": 일상 경험과 연결하여 설명한 부분
각 패턴에 대해 횟수와 예시 문장(최대 3개)을 포함하세요.

### 3. 도메인 설명 비율
전체 발화 중 도메인 전문 설명이 차지하는 비율 (0.0 ~ 1.0)

## 강의 녹취
{transcript_text}

## 출력 형식 (YAML)
```yaml
habitual_expressions:
  - expression: "여러분"
    total_count: 45
    recommendation: "사용 자제 권장"
effective_patterns:
  - pattern_type: "비유/유추"
    count: 3
    examples:
      - "세포막을 지퍼에 비유하면..."
domain_ratio: 0.65
```
"""


def build_pedagogy_prompt(transcript_text: str) -> str:
    """Construct LLM prompt for pedagogy analysis.

    Args:
        transcript_text: Full lecture transcript text.

    Returns:
        Formatted prompt string.
    """
    return _PEDAGOGY_PROMPT_TEMPLATE.format(
        transcript_text=transcript_text,
    )


# ----------------------------------------------------------------
# LLM pedagogy analysis (T045)
# ----------------------------------------------------------------


def _parse_pedagogy_response(
    response_text: str,
    section_id: str,
) -> PedagogyAnalysis:
    """Parse LLM YAML response into PedagogyAnalysis.

    Args:
        response_text: Raw LLM text response.
        section_id: Section identifier.

    Returns:
        PedagogyAnalysis instance. Empty analysis on parse failure.
    """
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        logger.warning("Failed to parse pedagogy analysis LLM response YAML")
        return PedagogyAnalysis(section_id=section_id)

    if not isinstance(data, dict):
        return PedagogyAnalysis(section_id=section_id)

    # Parse habitual expressions
    habitual: list[HabitualExpression] = []
    for item in data.get("habitual_expressions", []):
        if not isinstance(item, dict):
            continue
        expr = item.get("expression", "")
        if not expr:
            continue
        habitual.append(
            HabitualExpression(
                expression=expr,
                frequency_per_minute=float(
                    item.get("frequency_per_minute", 0.0),
                ),
                total_count=int(item.get("total_count", 0)),
                recommendation=item.get("recommendation", "정상 범위"),
            )
        )

    # Parse effective patterns
    patterns: list[EffectivePattern] = []
    for item in data.get("effective_patterns", []):
        if not isinstance(item, dict):
            continue
        ptype = item.get("pattern_type", "")
        if not ptype:
            continue
        examples = item.get("examples", [])
        if not isinstance(examples, list):
            examples = [str(examples)]
        patterns.append(
            EffectivePattern(
                pattern_type=ptype,
                count=int(item.get("count", 0)),
                examples=examples[:3],
            )
        )

    domain_ratio = float(data.get("domain_ratio", 0.0))
    domain_ratio = max(0.0, min(1.0, domain_ratio))

    return PedagogyAnalysis(
        section_id=section_id,
        habitual_expressions=habitual[:5],
        effective_patterns=patterns,
        domain_ratio=domain_ratio,
    )


def analyze_pedagogy_llm(
    transcript_path: str,
    section_id: str,
    model: str | None = None,
) -> PedagogyAnalysis:
    """Analyze instructor pedagogy patterns using LLM.

    Args:
        transcript_path: Path to lecture transcript file.
        section_id: Section identifier (A, B, C, D).
        model: Optional LLM model ID override.

    Returns:
        PedagogyAnalysis for the section.
    """
    from forma.config import get_llm_config, load_config
    from forma.llm_provider import create_provider

    try:
        cfg = load_config()
        llm_cfg = get_llm_config(cfg)
    except FileNotFoundError:
        llm_cfg = {}

    path = Path(transcript_path)
    try:
        transcript_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        transcript_text = path.read_text(encoding="euc-kr")

    prompt = build_pedagogy_prompt(transcript_text)
    provider = create_provider(
        provider=llm_cfg.get("provider", "gemini"),
        api_key=llm_cfg.get("api_key"),
        model=model or llm_cfg.get("model"),
    )

    try:
        response = provider.generate(
            prompt=prompt,
            max_tokens=4096,
            temperature=0.0,
            system_instruction=_PEDAGOGY_SYSTEM_INSTRUCTION,
        )
    except Exception:
        logger.warning(
            "Section %s pedagogy analysis LLM call failed",
            section_id,
            exc_info=True,
        )
        return PedagogyAnalysis(section_id=section_id)

    return _parse_pedagogy_response(response, section_id)
