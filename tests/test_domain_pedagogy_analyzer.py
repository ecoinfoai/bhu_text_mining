"""Tests for pedagogy analysis module.

T040: Dataclass creation
T041: analyze_pedagogy_llm with mocked LLM
T042: Isolation check (no domain terms)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from forma.domain_pedagogy_analyzer import (
    EffectivePattern,
    HabitualExpression,
    PedagogyAnalysis,
    analyze_pedagogy_llm,
    build_pedagogy_prompt,
)


# ----------------------------------------------------------------
# T040: Dataclass tests
# ----------------------------------------------------------------


class TestPedagogyDataclasses:
    """T040: HabitualExpression, EffectivePattern, PedagogyAnalysis."""

    def test_habitual_expression_creation(self) -> None:
        """HabitualExpression can be created with all fields."""
        expr = HabitualExpression(
            expression="여러분",
            frequency_per_minute=2.5,
            total_count=45,
            recommendation="사용 자제 권장",
        )
        assert expr.expression == "여러분"
        assert expr.frequency_per_minute == 2.5
        assert expr.total_count == 45
        assert expr.recommendation == "사용 자제 권장"

    def test_effective_pattern_creation(self) -> None:
        """EffectivePattern can be created with all fields."""
        pattern = EffectivePattern(
            pattern_type="비유/유추",
            count=3,
            examples=["세포막을 지퍼에 비유하면..."],
        )
        assert pattern.pattern_type == "비유/유추"
        assert pattern.count == 3
        assert len(pattern.examples) == 1

    def test_effective_pattern_default_examples(self) -> None:
        """EffectivePattern defaults to empty examples list."""
        pattern = EffectivePattern(pattern_type="임상 사례", count=1)
        assert pattern.examples == []

    def test_pedagogy_analysis_creation(self) -> None:
        """PedagogyAnalysis can be created with section_id."""
        pa = PedagogyAnalysis(
            section_id="A",
            habitual_expressions=[
                HabitualExpression("여러분", 2.5, 45, "사용 자제 권장"),
            ],
            effective_patterns=[
                EffectivePattern("비유/유추", 3, ["예시 문장"]),
            ],
            domain_ratio=0.65,
        )
        assert pa.section_id == "A"
        assert len(pa.habitual_expressions) == 1
        assert len(pa.effective_patterns) == 1
        assert pa.domain_ratio == 0.65

    def test_pedagogy_analysis_defaults(self) -> None:
        """PedagogyAnalysis has sensible defaults."""
        pa = PedagogyAnalysis(section_id="B")
        assert pa.habitual_expressions == []
        assert pa.effective_patterns == []
        assert pa.domain_ratio == 0.0


# ----------------------------------------------------------------
# T041: analyze_pedagogy_llm with mocked LLM
# ----------------------------------------------------------------


class TestAnalyzePedagogyLLM:
    """T041: LLM-based pedagogy analysis with mocked responses."""

    def test_parses_habitual_expressions(self, tmp_path) -> None:
        """LLM response with habitual expressions parsed correctly."""
        transcript = tmp_path / "1A_2주차_1차시.txt"
        transcript.write_text(
            "여러분 오늘은 피부에 대해 알아봅시다. 여러분 보시면 표피가 있습니다.",
            encoding="utf-8",
        )

        mock_response = """\
habitual_expressions:
  - expression: "여러분"
    total_count: 30
    frequency_per_minute: 2.0
    recommendation: "사용 자제 권장"
  - expression: "보시면"
    total_count: 15
    frequency_per_minute: 1.0
    recommendation: "사용 자제 권장"
  - expression: "그래서"
    total_count: 10
    frequency_per_minute: 0.7
    recommendation: "정상 범위"
  - expression: "자"
    total_count: 8
    frequency_per_minute: 0.5
    recommendation: "정상 범위"
  - expression: "이거"
    total_count: 5
    frequency_per_minute: 0.3
    recommendation: "정상 범위"
effective_patterns:
  - pattern_type: "비유/유추"
    count: 3
    examples:
      - "세포막을 지퍼에 비유하면 이해가 쉽습니다."
  - pattern_type: "학생 질문 유도"
    count: 5
    examples:
      - "이 부분 왜 그런지 생각해보세요."
      - "누가 한번 설명해볼까요?"
domain_ratio: 0.7
"""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ):
            result = analyze_pedagogy_llm(
                transcript_path=str(transcript),
                section_id="A",
            )

        assert result.section_id == "A"
        assert len(result.habitual_expressions) == 5
        assert result.habitual_expressions[0].expression == "여러분"
        assert result.habitual_expressions[0].total_count == 30
        assert result.habitual_expressions[0].recommendation == "사용 자제 권장"

    def test_parses_effective_patterns(self, tmp_path) -> None:
        """Effective patterns parsed with examples."""
        transcript = tmp_path / "test.txt"
        transcript.write_text("강의 내용", encoding="utf-8")

        mock_response = """\
habitual_expressions: []
effective_patterns:
  - pattern_type: "비유/유추"
    count: 2
    examples:
      - "세포막을 담장으로 비유하면"
      - "삼투를 스펀지로 이해하면"
  - pattern_type: "임상 사례"
    count: 1
    examples:
      - "화상 환자의 피부 재생 과정"
domain_ratio: 0.6
"""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ):
            result = analyze_pedagogy_llm(
                transcript_path=str(transcript),
                section_id="B",
            )

        assert len(result.effective_patterns) == 2
        assert result.effective_patterns[0].pattern_type == "비유/유추"
        assert result.effective_patterns[0].count == 2
        assert len(result.effective_patterns[0].examples) == 2

    def test_domain_ratio_parsed(self, tmp_path) -> None:
        """domain_ratio parsed and clamped to [0, 1]."""
        transcript = tmp_path / "test.txt"
        transcript.write_text("내용", encoding="utf-8")

        mock_response = """\
habitual_expressions: []
effective_patterns: []
domain_ratio: 0.75
"""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ):
            result = analyze_pedagogy_llm(
                transcript_path=str(transcript),
                section_id="A",
            )

        assert result.domain_ratio == 0.75

    def test_llm_failure_returns_empty(self, tmp_path) -> None:
        """LLM failure returns empty PedagogyAnalysis."""
        transcript = tmp_path / "test.txt"
        transcript.write_text("내용", encoding="utf-8")

        mock_provider = MagicMock()
        mock_provider.generate.side_effect = RuntimeError("LLM error")

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ):
            result = analyze_pedagogy_llm(
                transcript_path=str(transcript),
                section_id="A",
            )

        assert result.section_id == "A"
        assert result.habitual_expressions == []
        assert result.effective_patterns == []

    def test_malformed_yaml_returns_empty(self, tmp_path) -> None:
        """Malformed YAML returns empty analysis."""
        transcript = tmp_path / "test.txt"
        transcript.write_text("내용", encoding="utf-8")

        mock_provider = MagicMock()
        mock_provider.generate.return_value = "not: valid: yaml: [["

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ):
            result = analyze_pedagogy_llm(
                transcript_path=str(transcript),
                section_id="A",
            )

        assert result.section_id == "A"

    def test_habitual_capped_at_5(self, tmp_path) -> None:
        """Habitual expressions are capped at top 5."""
        transcript = tmp_path / "test.txt"
        transcript.write_text("내용", encoding="utf-8")

        expressions = "\n".join(
            f'  - expression: "expr{i}"\n    total_count: {10 - i}\n'
            f'    recommendation: "정상 범위"'
            for i in range(8)
        )
        mock_response = f"habitual_expressions:\n{expressions}\n"
        mock_response += "effective_patterns: []\ndomain_ratio: 0.5\n"

        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        with patch(
            "forma.llm_provider.create_provider",
            return_value=mock_provider,
        ):
            result = analyze_pedagogy_llm(
                transcript_path=str(transcript),
                section_id="A",
            )

        assert len(result.habitual_expressions) <= 5


# ----------------------------------------------------------------
# T042: Isolation check — no domain concept terms
# ----------------------------------------------------------------


class TestPedagogyIsolation:
    """T042: Pedagogy results contain no domain concept terms."""

    def test_prompt_excludes_domain_instruction(self) -> None:
        """Prompt instructs to exclude domain terms."""
        prompt = build_pedagogy_prompt("강의 내용입니다.")
        assert "도메인 전문 용어" in prompt
        assert "제외" in prompt

    def test_prompt_focuses_on_speech_patterns(self) -> None:
        """Prompt focuses on speech patterns, not domain content."""
        prompt = build_pedagogy_prompt("text")
        assert "습관적 표현" in prompt
        assert "교수법 패턴" in prompt

    def test_system_instruction_isolates_from_domain(self) -> None:
        """System instruction specifies speech pattern analysis only."""
        from forma.domain_pedagogy_analyzer import _PEDAGOGY_SYSTEM_INSTRUCTION

        assert "화법 분석" in _PEDAGOGY_SYSTEM_INSTRUCTION
        assert "도메인" in _PEDAGOGY_SYSTEM_INSTRUCTION
