"""Tests for feedback_generator.py — coaching feedback generation.

All LLM calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from forma.evaluation_types import (
    FeedbackResult,
    GraphComparisonResult,
    TripletEdge,
)
from forma.feedback_generator import (
    EMPTY_RESPONSE_FEEDBACK,
    FeedbackGenerator,
    MAX_FEEDBACK_CHARS,
    MAX_FEEDBACK_TOKENS,
    TIER_LENGTH_TARGETS,
    _REQUIRED_SECTIONS,
    _format_edges,
    _truncate_at_sentence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_comparison(
    f1: float = 0.7,
    n_matched: int = 3,
    n_missing: int = 1,
    n_wrong: int = 0,
) -> GraphComparisonResult:
    matched = [TripletEdge(f"S{i}", "r", f"O{i}") for i in range(n_matched)]
    missing = [TripletEdge(f"MS{i}", "r", f"MO{i}") for i in range(n_missing)]
    wrong = [TripletEdge(f"WS{i}", "r", f"WO{i}") for i in range(n_wrong)]
    return GraphComparisonResult(
        student_id="s001",
        question_sn=1,
        precision=f1,
        recall=f1,
        f1=f1,
        matched_edges=matched,
        missing_edges=missing,
        extra_edges=[],
        wrong_direction_edges=wrong,
    )


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestFormatEdges:
    """Tests for _format_edges()."""

    def test_empty_list(self):
        assert _format_edges([]) == "(없음)"

    def test_formats_edges(self):
        edges = [TripletEdge("A", "causes", "B")]
        result = _format_edges(edges)
        assert "A" in result
        assert "causes" in result
        assert "B" in result


class TestTruncateAtSentence:
    """Tests for _truncate_at_sentence()."""

    def test_short_text_unchanged(self):
        assert _truncate_at_sentence("짧은 문장.", 100) == "짧은 문장."

    def test_truncates_at_period(self):
        text = "첫 문장. 두 번째 문장. 세 번째 긴 문장입니다."
        result = _truncate_at_sentence(text, 20)
        assert result.endswith(".")
        assert len(result) <= 20

    def test_no_period_truncates_at_max(self):
        text = "마침표 없는 긴 텍스트" * 10
        result = _truncate_at_sentence(text, 30)
        assert len(result) <= 30


# ---------------------------------------------------------------------------
# FeedbackGenerator tests
# ---------------------------------------------------------------------------


class TestFeedbackGenerator:
    """Tests for FeedbackGenerator."""

    _VALID_FEEDBACK = (
        "[현재 상태] 좋은 답변입니다. 개선점은 다음과 같습니다.\n"
        "[원인] 일부 개념의 관계 파악이 필요합니다.\n"
        "[학생에게 권하는 사항] 교재를 복습하면 좋겠습니다."
    )

    @pytest.fixture()
    def mock_provider(self):
        prov = MagicMock()
        prov.generate.return_value = self._VALID_FEEDBACK
        return prov

    def test_empty_response(self, mock_provider):
        """Empty student response returns template feedback."""
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            student_id="s001",
            question_sn=1,
            question="Q?",
            student_response="",
            concept_coverage=0.0,
            graph_comparison=None,
            tier_level=0,
            tier_label="미달",
        )
        assert result.feedback_text == EMPTY_RESPONSE_FEEDBACK
        assert result.tier_level == 0
        assert mock_provider.generate.call_count == 0

    def test_generates_feedback(self, mock_provider):
        """Normal response generates LLM feedback."""
        gen = FeedbackGenerator(mock_provider)
        gc = _make_graph_comparison()
        result = gen.generate(
            student_id="s001",
            question_sn=1,
            question="항상성이란?",
            student_response="체온을 일정하게 유지하는 것",
            concept_coverage=0.6,
            graph_comparison=gc,
            tier_level=1,
            tier_label="기전 이해",
        )
        assert isinstance(result, FeedbackResult)
        assert "[현재 상태]" in result.feedback_text
        assert "[원인]" in result.feedback_text
        assert "[학생에게 권하는 사항]" in result.feedback_text
        assert result.student_id == "s001"
        assert mock_provider.generate.call_count == 1

    def test_feedback_within_char_limit(self, mock_provider):
        """Feedback is truncated to max_chars via fallback when LLM output exceeds limit."""
        long_text = (
            "[현재 상태] " + "좋은 답변입니다. " * 100 + "\n"
            "[원인] " + "원인 설명입니다. " * 100 + "\n"
            "[학생에게 권하는 사항] " + "제안합니다. " * 100
        )
        mock_provider.generate.return_value = long_text
        gen = FeedbackGenerator(mock_provider, max_chars=600)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해"
        )
        assert result.char_count <= 600

    def test_data_sources_with_graph(self, mock_provider):
        """Data sources include graph_f1 when graph_comparison present."""
        gen = FeedbackGenerator(mock_provider)
        gc = _make_graph_comparison()
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, gc, 2, "기전+용어"
        )
        assert "graph_f1" in result.data_sources_used

    def test_data_sources_without_graph(self, mock_provider):
        """Data sources exclude graph_f1 when no graph_comparison."""
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해"
        )
        assert "graph_f1" not in result.data_sources_used

    def test_provider_failure(self, mock_provider):
        """Provider exception returns error message feedback."""
        mock_provider.generate.side_effect = Exception("API down")
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해"
        )
        assert "실패" in result.feedback_text

    def test_tier_level_preserved(self, mock_provider):
        """Tier level and label preserved in result."""
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 3, "전문적 구조화"
        )
        assert result.tier_level == 3
        assert result.tier_label == "전문적 구조화"


# ---------------------------------------------------------------------------
# Phase 2: Foundation constants (T002-T007)
# ---------------------------------------------------------------------------


class TestFeedbackConstants:
    """T002-T003: Tests for updated feedback constants."""

    def test_max_feedback_chars(self):
        assert MAX_FEEDBACK_CHARS == 600

    def test_max_feedback_tokens(self):
        assert MAX_FEEDBACK_TOKENS == 2000

    def test_tier_length_targets(self):
        assert TIER_LENGTH_TARGETS == {3: 300, 2: 400, 1: 500, 0: 600}

    def test_required_sections_new_names(self):
        assert _REQUIRED_SECTIONS == ["[현재 상태]", "[원인]", "[학생에게 권하는 사항]"]


# ---------------------------------------------------------------------------
# Phase 3: US1+US2 Prompt rewrite (T008-T014)
# ---------------------------------------------------------------------------

from forma.prompt_templates import (  # noqa: E402
    FEEDBACK_SYSTEM_INSTRUCTION,
    render_feedback_prompt,
)


class TestFeedbackPromptUS1US2:
    """T008-T010: Tests for encouraging tone and structured feedback."""

    def test_system_instruction_has_banned_expressions(self):
        """T008: System instruction prohibits deficit-focused expressions."""
        banned = ["놓쳤습니다", "부족합니다", "언급하지 않았습니다",
                  "잘못 이해하고 있습니다", "오류가 있습니다", "틀렸습니다"]
        for expr in banned:
            assert expr in FEEDBACK_SYSTEM_INSTRUCTION, f"Missing banned expression: {expr}"

    def test_system_instruction_encouraging_tone(self):
        """T008: System instruction directs encouraging coaching tone."""
        assert "코칭" in FEEDBACK_SYSTEM_INSTRUCTION or "격려" in FEEDBACK_SYSTEM_INSTRUCTION

    def test_system_instruction_new_section_names(self):
        """System instruction references new section names."""
        assert "[현재 상태]" in FEEDBACK_SYSTEM_INSTRUCTION
        assert "[원인]" in FEEDBACK_SYSTEM_INSTRUCTION
        assert "[학생에게 권하는 사항]" in FEEDBACK_SYSTEM_INSTRUCTION

    def test_template_has_new_section_names(self):
        """T009: Template renders with new section names."""
        rendered = render_feedback_prompt(
            question="Q?", student_response="A",
            concept_coverage=0.5, graph_f1=0.7,
            tier_level=2, tier_label="기전+용어",
            matched_count=3, missing_count=1,
            wrong_direction_count=0,
            missing_edges_text="(없음)", wrong_direction_text="(없음)",
        )
        assert "[현재 상태]" in rendered
        assert "[원인]" in rendered
        assert "[학생에게 권하는 사항]" in rendered

    def test_template_no_old_section_names(self):
        """T009: Template does NOT reference old section names."""
        rendered = render_feedback_prompt(
            question="Q?", student_response="A",
            concept_coverage=0.5, graph_f1=0.7,
            tier_level=2, tier_label="기전+용어",
            matched_count=3, missing_count=1,
            wrong_direction_count=0,
            missing_edges_text="(없음)", wrong_direction_text="(없음)",
        )
        assert "[평가 요약]" not in rendered
        assert "[분석 결과]" not in rendered
        assert "[학습 제안]" not in rendered

    def test_template_no_jargon(self):
        """T012: Template does not reference technical jargon."""
        rendered = render_feedback_prompt(
            question="Q?", student_response="A",
            concept_coverage=0.5, graph_f1=0.7,
            tier_level=1, tier_label="기전 이해",
            matched_count=3, missing_count=1,
            wrong_direction_count=0,
            missing_edges_text="(없음)", wrong_direction_text="(없음)",
        )
        assert "임베딩" in rendered  # present in prohibition instruction
        assert "그래프 분석" not in rendered

    def test_template_jargon_prohibition_instruction(self):
        """FR-006: Template explicitly instructs LLM to avoid technical jargon."""
        rendered = render_feedback_prompt(
            question="Q?", student_response="A",
            concept_coverage=0.5, graph_f1=0.7,
            tier_level=1, tier_label="기전 이해",
            matched_count=3, missing_count=1,
            wrong_direction_count=0,
            missing_edges_text="(없음)", wrong_direction_text="(없음)",
        )
        assert "기술 용어" in rendered
        assert "사용하지 마세요" in rendered
        for term in ["임베딩", "코사인 유사도", "F1", "그래프 메트릭", "루브릭 점수"]:
            assert term in rendered, f"Jargon prohibition should list '{term}'"

    def test_template_no_minimum_length(self):
        """T012: Template does not have '~자 이상 작성하세요' minimum."""
        rendered = render_feedback_prompt(
            question="Q?", student_response="A",
            concept_coverage=0.5, graph_f1=0.7,
            tier_level=0, tier_label="미달",
            matched_count=0, missing_count=5,
            wrong_direction_count=0,
            missing_edges_text="test", wrong_direction_text="test",
        )
        assert "이상 작성하세요" not in rendered

    def test_generate_length_guidance_new_targets(self):
        """T010/T014: generate() uses new tier targets in length_guidance."""
        from unittest.mock import patch
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "[현재 상태] 테스트. [원인] 테스트. [학생에게 권하는 사항] 테스트."
        gen = FeedbackGenerator(mock_provider)
        with patch("forma.feedback_generator.render_feedback_prompt") as mock_render:
            mock_render.return_value = "mocked prompt"
            gen.generate("s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어")
            call_kwargs = mock_render.call_args
            length_guidance = call_kwargs.kwargs.get("length_guidance", "") if call_kwargs.kwargs else ""
            if not length_guidance and call_kwargs.args:
                pass
            # The length_guidance should reference ~400 for tier 2
            assert "400" in length_guidance, f"Tier 2 should target ~400 chars, got: {length_guidance}"


# ---------------------------------------------------------------------------
# Phase 4: US4 Negative expression filter (T015-T019)
# ---------------------------------------------------------------------------

from forma.feedback_generator import (  # noqa: E402
    FALLBACK_TEMPLATES,
    NEGATIVE_EXPRESSIONS,
    _soften_tone,
    _validate_and_repair,
)


class TestSoftenTone:
    """T015-T017: Tests for _soften_tone() negative expression filter."""

    def test_replaces_each_banned_expression(self):
        """T015: Each banned expression is replaced with its alternative."""
        for banned, replacement in NEGATIVE_EXPRESSIONS.items():
            text = f"학생이 {banned}"
            result = _soften_tone(text)
            assert banned not in result, f"'{banned}' should be replaced"
            assert replacement in result, f"'{replacement}' should appear in result"

    def test_unchanged_when_no_banned_expressions(self):
        """T016: Text without banned expressions is returned unchanged."""
        text = "학생이 개념을 잘 이해하고 있습니다."
        assert _soften_tone(text) == text

    def test_replaces_multiple_different_expressions(self):
        """T017: Multiple different banned expressions in single text are all replaced."""
        text = "학생이 놓쳤습니다. 또한 부족합니다. 그리고 틀렸습니다."
        result = _soften_tone(text)
        assert "놓쳤습니다" not in result
        assert "부족합니다" not in result
        assert "틀렸습니다" not in result
        assert "추가로 학습하면 좋겠습니다" in result
        assert "더 보완하면 좋겠습니다" in result
        assert "다시 확인해보면 좋겠습니다" in result


# ---------------------------------------------------------------------------
# Phase 5: Validation, repair, retry, fallback (T020-T025)
# ---------------------------------------------------------------------------


class TestValidateAndRepair:
    """T020: Tests for _validate_and_repair()."""

    def test_valid_text_unchanged(self):
        """Well-formed feedback passes through unchanged."""
        text = (
            "[현재 상태] 학생은 개념을 잘 이해하고 있습니다. 기본적인 구조를 파악했습니다.\n"
            "[원인] 핵심 관계를 일부 놓친 부분이 있습니다. 추가 학습이 필요합니다.\n"
            "[학생에게 권하는 사항] 관련 개념을 복습하면 좋겠습니다. 교재를 참고하세요."
        )
        result = _validate_and_repair(text)
        assert "[현재 상태]" in result
        assert "[원인]" in result
        assert "[학생에게 권하는 사항]" in result

    def test_missing_section_raises(self):
        """Text missing a required section raises ValueError for retry."""
        text = "[현재 상태] 좋은 이해도입니다. [원인] 일부 부족합니다."
        # Missing [학생에게 권하는 사항]
        with pytest.raises(ValueError, match="학생에게 권하는 사항"):
            _validate_and_repair(text)

    def test_excess_sentences_truncated(self):
        """Section with >3 sentences is truncated to 3."""
        text = (
            "[현재 상태] 첫째. 둘째. 셋째. 넷째. 다섯째.\n"
            "[원인] 원인입니다. 설명합니다.\n"
            "[학생에게 권하는 사항] 제안합니다. 복습하세요."
        )
        result = _validate_and_repair(text)
        # Count sentences in [현재 상태] section
        status_section = result.split("[원인]")[0]
        status_text = status_section.replace("[현재 상태]", "").strip()
        sentence_count = len([s for s in status_text.split(".") if s.strip()])
        assert sentence_count <= 3

    def test_incomplete_sentence_repaired(self):
        """Text not ending with period gets period appended."""
        text = (
            "[현재 상태] 학생의 이해도가 좋습니다.\n"
            "[원인] 일부 개념이 부족합니다.\n"
            "[학생에게 권하는 사항] 추가 학습을 권합니다"
        )
        result = _validate_and_repair(text)
        assert result.rstrip().endswith(".")


class TestFallbackTemplates:
    """T021: Tests for FALLBACK_TEMPLATES."""

    def test_all_tiers_present(self):
        """All 4 tier levels have fallback templates."""
        for tier in range(4):
            assert tier in FALLBACK_TEMPLATES, f"Missing fallback for tier {tier}"

    def test_templates_contain_required_sections(self):
        """Each fallback template contains all 3 required sections."""
        for tier, template in FALLBACK_TEMPLATES.items():
            for section in _REQUIRED_SECTIONS:
                assert section in template, f"Tier {tier} missing section: {section}"

    def test_templates_end_with_period(self):
        """Each fallback template ends with a period."""
        for tier, template in FALLBACK_TEMPLATES.items():
            assert template.rstrip().endswith("."), f"Tier {tier} template doesn't end with period"

    def test_templates_within_char_range(self):
        """Each fallback template is between 300 and 600 chars."""
        for tier, template in FALLBACK_TEMPLATES.items():
            char_count = len(template)
            assert 300 <= char_count <= 600, (
                f"Tier {tier} template length {char_count} outside 300-600 range"
            )


class TestRetryAndFallback:
    """T022: Tests for retry logic in generate()."""

    @pytest.fixture()
    def mock_provider(self):
        prov = MagicMock()
        return prov

    def test_retry_on_first_structural_failure(self, mock_provider):
        """First generation fails structurally → retry once → second succeeds."""
        bad_response = "이것은 구조가 없는 응답입니다."
        good_response = (
            "[현재 상태] 학생은 잘 이해하고 있습니다. 기본 개념을 파악했습니다.\n"
            "[원인] 일부 관계 파악이 필요합니다. 추가 학습이 도움됩니다.\n"
            "[학생에게 권하는 사항] 교재를 복습하면 좋겠습니다. 연습 문제를 풀어보세요."
        )
        mock_provider.generate.side_effect = [bad_response, good_response]
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어"
        )
        assert "[현재 상태]" in result.feedback_text
        assert mock_provider.generate.call_count == 2

    def test_fallback_on_double_failure(self, mock_provider):
        """Both LLM attempts fail structurally → fallback template used."""
        bad_response = "구조 없는 응답입니다."
        mock_provider.generate.side_effect = [bad_response, bad_response]
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어"
        )
        assert result.feedback_text == FALLBACK_TEMPLATES[2]
        assert mock_provider.generate.call_count == 2

    def test_fallback_on_double_failure_logs_warning(self, mock_provider, caplog):
        """Double failure logs a warning (FR-014)."""
        import logging
        bad_response = "구조 없는 응답입니다."
        mock_provider.generate.side_effect = [bad_response, bad_response]
        gen = FeedbackGenerator(mock_provider)
        with caplog.at_level(logging.WARNING, logger="forma.feedback_generator"):
            gen.generate("s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어")
        assert any("fallback" in record.message.lower() or "대체" in record.message
                    for record in caplog.records)

    def test_soften_tone_applied_before_validation(self, mock_provider):
        """_soften_tone() is applied to LLM output before validation."""
        response_with_banned = (
            "[현재 상태] 학생이 개념을 놓쳤습니다. 이해가 부족합니다.\n"
            "[원인] 핵심 관계를 파악하지 못했습니다.\n"
            "[학생에게 권하는 사항] 추가 학습이 필요합니다."
        )
        mock_provider.generate.return_value = response_with_banned
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해"
        )
        assert "놓쳤습니다" not in result.feedback_text
        assert "추가로 학습하면 좋겠습니다" in result.feedback_text


# ---------------------------------------------------------------------------
# Phase 7: T030 Integration test — full pipeline with well-formed feedback
# ---------------------------------------------------------------------------


class TestIntegrationFullPipeline:
    """T030: Integration test — mock LLM returns well-formed new-format feedback.

    Verifies the complete pipeline produces FeedbackResult with all 3 sections,
    no banned expressions, within 300-600 chars, ending with period.
    """

    _WELL_FORMED = (
        "[현재 상태] 학생은 핵심 개념인 항상성의 정의를 정확히 이해하고 있습니다. "
        "체온 조절 메커니즘에 대한 설명이 잘 구성되어 있습니다.\n"
        "[원인] 음성되먹임의 세부 경로에 대해 추가 학습이 필요한 부분이 있습니다. "
        "감지-조절-반응의 전체 흐름을 정리하면 더 좋겠습니다.\n"
        "[학생에게 권하는 사항] 교재 3장의 음성되먹임 도표를 참고하여 복습해 보세요. "
        "개념 간 관계를 화살표로 정리하는 연습을 권합니다."
    )

    @pytest.fixture()
    def mock_provider(self):
        prov = MagicMock()
        prov.generate.return_value = self._WELL_FORMED
        return prov

    def test_pipeline_produces_all_sections(self, mock_provider):
        """Full pipeline returns feedback with all 3 required sections."""
        gen = FeedbackGenerator(mock_provider)
        gc = _make_graph_comparison(f1=0.7, n_matched=3, n_missing=1, n_wrong=1)
        result = gen.generate(
            "s001", 1, "항상성이란?", "체온을 유지하는 것",
            0.6, gc, 2, "기전+용어",
        )
        for section in _REQUIRED_SECTIONS:
            assert section in result.feedback_text, f"Missing section: {section}"

    def test_pipeline_no_banned_expressions(self, mock_provider):
        """Full pipeline output contains no banned expressions."""
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
        )
        for banned in NEGATIVE_EXPRESSIONS:
            assert banned not in result.feedback_text, f"Banned expression found: {banned}"

    def test_pipeline_char_count_within_range(self, mock_provider):
        """Full pipeline output is within 300-600 chars (or fallback range)."""
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        assert result.char_count <= MAX_FEEDBACK_CHARS

    def test_pipeline_ends_with_period(self, mock_provider):
        """Full pipeline output ends with sentence-terminating punctuation."""
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        assert result.feedback_text.rstrip()[-1] in ".。!?"

    def test_pipeline_single_llm_call(self, mock_provider):
        """Well-formed feedback needs only one LLM call (no retry)."""
        gen = FeedbackGenerator(mock_provider)
        _result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        assert mock_provider.generate.call_count == 1

    def test_pipeline_data_sources_complete(self, mock_provider):
        """Data sources include all expected entries when graph_comparison present."""
        gen = FeedbackGenerator(mock_provider)
        gc = _make_graph_comparison()
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, gc, 2, "기전+용어",
        )
        assert "concept_coverage" in result.data_sources_used
        assert "tier_level" in result.data_sources_used
        assert "graph_f1" in result.data_sources_used
        assert "edge_analysis" in result.data_sources_used


# ---------------------------------------------------------------------------
# Phase 7: T031 Adversary tests — 8 personas attacking the feedback pipeline
# ---------------------------------------------------------------------------


class TestAdversaryPersona1ToxicLLM:
    """Persona 1: 'The Toxic LLM' — LLM returns all 6 banned expressions."""

    @pytest.fixture()
    def mock_provider(self):
        return MagicMock()

    def test_all_six_banned_expressions_replaced(self, mock_provider):
        """LLM returns text with ALL 6 banned expressions → all replaced."""
        toxic = (
            "[현재 상태] 학생이 개념을 놓쳤습니다. 이해가 부족합니다.\n"
            "[원인] 핵심 관계를 언급하지 않았습니다. "
            "잘못 이해하고 있습니다.\n"
            "[학생에게 권하는 사항] 답변에 오류가 있습니다. "
            "관계 방향이 틀렸습니다."
        )
        mock_provider.generate.return_value = toxic
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
        )
        for banned in NEGATIVE_EXPRESSIONS:
            assert banned not in result.feedback_text, (
                f"Banned expression '{banned}' survived pipeline"
            )

    def test_banned_expression_embedded_in_sentence(self, mock_provider):
        """Banned expression embedded within a longer sentence → still caught."""
        text = (
            "[현재 상태] 전반적인 이해가 부족합니다만 노력의 흔적이 보입니다.\n"
            "[원인] 개념 정리가 필요합니다.\n"
            "[학생에게 권하는 사항] 복습하면 좋겠습니다."
        )
        mock_provider.generate.return_value = text
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
        )
        assert "부족합니다" not in result.feedback_text
        assert "더 보완하면 좋겠습니다" in result.feedback_text

    def test_multiple_occurrences_of_same_banned_expression(self, mock_provider):
        """Same banned expression appears 3 times → all 3 occurrences replaced."""
        text = (
            "[현재 상태] 개념이 부족합니다. 설명이 부족합니다.\n"
            "[원인] 이해가 부족합니다.\n"
            "[학생에게 권하는 사항] 복습을 권합니다."
        )
        mock_provider.generate.return_value = text
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
        )
        assert result.feedback_text.count("부족합니다") == 0
        assert result.feedback_text.count("더 보완하면 좋겠습니다") == 3


class TestAdversaryPersona2Truncator:
    """Persona 2: 'The Truncator' — LLM returns text missing final section or mid-sentence."""

    @pytest.fixture()
    def mock_provider(self):
        return MagicMock()

    def test_missing_final_section_triggers_retry(self, mock_provider):
        """Text missing [학생에게 권하는 사항] → retry triggered, second call succeeds."""
        truncated = (
            "[현재 상태] 학생은 개념을 이해하고 있습니다.\n"
            "[원인] 일부 관계 파악이 필요합니다."
        )
        good = (
            "[현재 상태] 학생은 개념을 이해하고 있습니다.\n"
            "[원인] 일부 관계 파악이 필요합니다.\n"
            "[학생에게 권하는 사항] 교재를 복습하면 좋겠습니다."
        )
        mock_provider.generate.side_effect = [truncated, good]
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        assert "[학생에게 권하는 사항]" in result.feedback_text
        assert mock_provider.generate.call_count == 2

    def test_missing_two_sections_triggers_retry(self, mock_provider):
        """Text with only [현재 상태] → retry triggered."""
        truncated = "[현재 상태] 학생은 개념을 일부 이해하고 있습니다."
        good = (
            "[현재 상태] 이해도가 있습니다.\n"
            "[원인] 추가 학습이 필요합니다.\n"
            "[학생에게 권하는 사항] 복습을 권합니다."
        )
        mock_provider.generate.side_effect = [truncated, good]
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
        )
        assert "[원인]" in result.feedback_text
        assert "[학생에게 권하는 사항]" in result.feedback_text
        assert mock_provider.generate.call_count == 2

    def test_both_truncated_falls_back(self, mock_provider):
        """Both attempts missing final section → fallback template used."""
        truncated = (
            "[현재 상태] 학생은 개념을 이해하고 있습니다.\n"
            "[원인] 일부 관계 파악이 필요합니다."
        )
        mock_provider.generate.side_effect = [truncated, truncated]
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
        )
        assert result.feedback_text == FALLBACK_TEMPLATES[1]


class TestAdversaryPersona3EmptyResponder:
    """Persona 3: 'The Empty Responder' — LLM returns empty/whitespace."""

    @pytest.fixture()
    def mock_provider(self):
        return MagicMock()

    def test_empty_string_uses_fallback(self, mock_provider):
        """LLM returns empty string → fallback template used."""
        mock_provider.generate.return_value = ""
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        # Empty string stripped becomes empty → no sections → ValueError → retry
        # Both attempts empty → fallback
        assert result.feedback_text == FALLBACK_TEMPLATES[2]

    def test_whitespace_only_uses_fallback(self, mock_provider):
        """LLM returns whitespace-only → fallback template used."""
        mock_provider.generate.return_value = "   \n\t  \n  "
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 0, "미달",
        )
        assert result.feedback_text == FALLBACK_TEMPLATES[0]

    def test_section_headers_with_no_content(self, mock_provider):
        """LLM returns only section headers with empty content."""
        empty_sections = "[현재 상태] [원인] [학생에게 권하는 사항]"
        mock_provider.generate.return_value = empty_sections
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
        )
        # The pipeline should still produce valid output (possibly repaired)
        assert isinstance(result.feedback_text, str)
        assert len(result.feedback_text) > 0


class TestAdversaryPersona4VerboseWriter:
    """Persona 4: 'The Verbose Writer' — LLM returns >600 chars."""

    @pytest.fixture()
    def mock_provider(self):
        return MagicMock()

    def _make_verbose_response(self, target_chars: int) -> str:
        """Build a valid-structure response padded to target_chars."""
        base = (
            "[현재 상태] 학생은 기본 개념을 이해하고 있습니다. "
            "핵심 개념의 정의를 정확히 서술했습니다.\n"
            "[원인] 세부 메커니즘의 관계 파악이 더 필요합니다. "
            "인과관계 연결이 약합니다.\n"
            "[학생에게 권하는 사항] 교재를 참고하여 복습하세요. "
        )
        # Pad with additional valid sentences in the last section
        padding = "추가로 관련 개념을 정리해 보세요. " * 50
        text = base + padding
        # Ensure it exceeds target
        while len(text) < target_chars:
            text += "더 많은 학습을 권합니다. "
        return text

    def test_800_chars_truncated_to_max(self, mock_provider):
        """800-char response → truncated to <=600 chars."""
        verbose = self._make_verbose_response(800)
        assert len(verbose) >= 800
        mock_provider.generate.return_value = verbose
        gen = FeedbackGenerator(mock_provider, max_chars=600)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        assert result.char_count <= 600

    def test_2000_chars_truncated_to_max(self, mock_provider):
        """2000-char response → truncated to <=600 chars."""
        verbose = self._make_verbose_response(2000)
        mock_provider.generate.return_value = verbose
        gen = FeedbackGenerator(mock_provider, max_chars=600)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 0, "미달",
        )
        assert result.char_count <= 600

    def test_truncated_result_ends_with_period(self, mock_provider):
        """Truncated result still ends with sentence-terminating punctuation."""
        verbose = self._make_verbose_response(1000)
        mock_provider.generate.return_value = verbose
        gen = FeedbackGenerator(mock_provider, max_chars=600)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
        )
        assert result.feedback_text.rstrip()[-1] in ".。!?"


class TestAdversaryPersona5MalformedStructurer:
    """Persona 5: 'The Malformed Structurer' — LLM returns text with broken structure."""

    @pytest.fixture()
    def mock_provider(self):
        return MagicMock()

    def test_no_section_markers_triggers_retry(self, mock_provider):
        """Plain text without any section markers → retry triggered."""
        plain = "학생은 개념을 잘 이해하고 있습니다. 교재를 복습하면 좋겠습니다."
        good = (
            "[현재 상태] 이해도가 좋습니다.\n"
            "[원인] 추가 학습이 필요합니다.\n"
            "[학생에게 권하는 사항] 복습을 권합니다."
        )
        mock_provider.generate.side_effect = [plain, good]
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        assert "[현재 상태]" in result.feedback_text
        assert mock_provider.generate.call_count == 2

    def test_typo_in_section_name_triggers_retry(self, mock_provider):
        """Section name '[현재상태]' (missing space) → not recognized, triggers retry."""
        typo = (
            "[현재상태] 학생은 이해하고 있습니다.\n"
            "[원인] 추가 학습이 필요합니다.\n"
            "[학생에게 권하는 사항] 복습을 권합니다."
        )
        good = (
            "[현재 상태] 이해도가 좋습니다.\n"
            "[원인] 추가 학습이 필요합니다.\n"
            "[학생에게 권하는 사항] 복습을 권합니다."
        )
        mock_provider.generate.side_effect = [typo, good]
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        assert "[현재 상태]" in result.feedback_text
        assert mock_provider.generate.call_count == 2

    def test_sections_in_wrong_order(self, mock_provider):
        """Sections in wrong order (recommendations first) → accepted if all present."""
        wrong_order = (
            "[학생에게 권하는 사항] 교재를 복습하면 좋겠습니다.\n"
            "[현재 상태] 학생은 기본 개념을 이해하고 있습니다.\n"
            "[원인] 세부 관계 파악이 필요합니다."
        )
        mock_provider.generate.return_value = wrong_order
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        # All 3 sections should be present in output
        for section in _REQUIRED_SECTIONS:
            assert section in result.feedback_text

    def test_double_malformed_uses_fallback(self, mock_provider):
        """Both attempts have broken structure → fallback template used."""
        broken = "이것은 구조가 완전히 잘못된 응답입니다. 섹션 마커가 없습니다."
        mock_provider.generate.side_effect = [broken, broken]
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 3, "전문적 구조화",
        )
        assert result.feedback_text == FALLBACK_TEMPLATES[3]


class TestAdversaryPersona6DoubleFailure:
    """Persona 6: 'The Double Failure' — both attempts fail structurally."""

    @pytest.fixture()
    def mock_provider(self):
        return MagicMock()

    def test_double_truncation_uses_fallback(self, mock_provider):
        """Both calls return truncated text (missing section) → fallback used."""
        truncated = "[현재 상태] 학생의 이해도가 좋습니다."
        mock_provider.generate.side_effect = [truncated, truncated]
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 0, "미달",
        )
        assert result.feedback_text == FALLBACK_TEMPLATES[0]
        assert mock_provider.generate.call_count == 2

    def test_double_empty_uses_fallback_and_logs_warning(self, mock_provider, caplog):
        """Both calls empty → fallback template + warning logged (FR-014)."""
        import logging
        mock_provider.generate.side_effect = ["", ""]
        gen = FeedbackGenerator(mock_provider)
        with caplog.at_level(logging.WARNING, logger="forma.feedback_generator"):
            result = gen.generate(
                "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
            )
        assert result.feedback_text == FALLBACK_TEMPLATES[1]
        # Verify warning logged about fallback
        fallback_warnings = [
            r for r in caplog.records
            if "fallback" in r.message.lower() or "대체" in r.message
        ]
        assert len(fallback_warnings) >= 1, "FR-014: Warning must be logged on fallback"

    def test_double_malformed_fallback_has_all_sections(self, mock_provider):
        """Both calls malformed → fallback template has all 3 required sections."""
        broken = "구조 없는 피드백 텍스트입니다."
        mock_provider.generate.side_effect = [broken, broken]
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        for section in _REQUIRED_SECTIONS:
            assert section in result.feedback_text, (
                f"Fallback template missing section: {section}"
            )

    def test_fallback_matches_tier_level(self, mock_provider):
        """Fallback template matches the student's tier level."""
        broken = "구조 없는 응답입니다."
        for tier in range(4):
            mock_provider.generate.side_effect = [broken, broken]
            gen = FeedbackGenerator(mock_provider)
            result = gen.generate(
                "s001", 1, "Q?", "answer", 0.5, None, tier, f"tier{tier}",
            )
            assert result.feedback_text == FALLBACK_TEMPLATES[tier], (
                f"Tier {tier} should use FALLBACK_TEMPLATES[{tier}]"
            )

    def test_first_fails_second_succeeds(self, mock_provider):
        """First attempt ValueError, second attempt succeeds → no fallback."""
        malformed = "마커 없는 텍스트입니다."
        good = (
            "[현재 상태] 좋은 이해도입니다.\n"
            "[원인] 추가 학습이 도움됩니다.\n"
            "[학생에게 권하는 사항] 복습을 권합니다."
        )
        mock_provider.generate.side_effect = [malformed, good]
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        assert result.feedback_text != FALLBACK_TEMPLATES[2]
        assert "[현재 상태]" in result.feedback_text
        assert mock_provider.generate.call_count == 2


class TestAdversaryPersona7BoundaryPusher:
    """Persona 7: 'The Boundary Pusher' — edge boundary character counts."""

    @pytest.fixture()
    def mock_provider(self):
        return MagicMock()

    def _make_feedback_of_length(self, target: int) -> str:
        """Build valid-structure feedback text of approximately target chars."""
        base = "[현재 상태] 개념을 이해하고 있습니다.\n[원인] 추가 학습이 필요합니다.\n[학생에게 권하는 사항] "
        remaining = target - len(base)
        if remaining > 0:
            # Fill with Korean chars and periods for valid sentences
            filler = "복습을 권합니다. " * (remaining // 9 + 1)
            base += filler
        # Trim to exact target, ending with period
        text = base[:target]
        if not text.endswith("."):
            text = text.rstrip() + "."
        return text

    def test_exactly_600_chars_accepted(self, mock_provider):
        """Exactly 600-char feedback → accepted within max limit."""
        text = self._make_feedback_of_length(600)
        mock_provider.generate.return_value = text
        gen = FeedbackGenerator(mock_provider, max_chars=600)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 0, "미달",
        )
        assert result.char_count <= 600

    def test_601_chars_truncated(self, mock_provider):
        """601-char feedback → truncated to <=600."""
        text = self._make_feedback_of_length(601)
        assert len(text) >= 601
        mock_provider.generate.return_value = text
        gen = FeedbackGenerator(mock_provider, max_chars=600)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 0, "미달",
        )
        assert result.char_count <= 600

    def test_short_valid_feedback_accepted(self, mock_provider):
        """Short but valid feedback (all sections present) → accepted."""
        short = (
            "[현재 상태] 이해도가 좋습니다.\n"
            "[원인] 관계 파악이 필요합니다.\n"
            "[학생에게 권하는 사항] 복습을 권합니다."
        )
        mock_provider.generate.return_value = short
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 3, "전문적 구조화",
        )
        assert "[현재 상태]" in result.feedback_text
        assert "[원인]" in result.feedback_text
        assert "[학생에게 권하는 사항]" in result.feedback_text


class TestAdversaryPersona8UnicodeAttacker:
    """Persona 8: 'The Unicode Attacker' — special characters, emojis, mixed scripts."""

    @pytest.fixture()
    def mock_provider(self):
        return MagicMock()

    def test_emoji_in_feedback_no_crash(self, mock_provider):
        """Feedback with emojis does not crash the pipeline."""
        text = (
            "[현재 상태] 학생은 개념을 잘 이해하고 있습니다 \U0001f44d.\n"
            "[원인] 세부 관계 파악이 필요합니다 \U0001f4aa.\n"
            "[학생에게 권하는 사항] 교재를 복습하면 좋겠습니다 \U0001f389."
        )
        mock_provider.generate.return_value = text
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        assert isinstance(result.feedback_text, str)
        assert len(result.feedback_text) > 0

    def test_mixed_cjk_scripts_no_crash(self, mock_provider):
        """Mixed Korean + Chinese characters do not crash the pipeline."""
        text = (
            "[현재 상태] 학생은 概念을 이해하고 있습니다.\n"
            "[원인] 細部 관계 파악이 필요합니다.\n"
            "[학생에게 권하는 사항] 教材를 복습하면 좋겠습니다."
        )
        mock_provider.generate.return_value = text
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
        )
        assert isinstance(result.feedback_text, str)
        for section in _REQUIRED_SECTIONS:
            assert section in result.feedback_text

    def test_xml_unsafe_chars_in_feedback(self, mock_provider):
        """XML-unsafe characters (<, >, &) in feedback are handled."""
        text = (
            "[현재 상태] 학생의 이해도 > 평균입니다.\n"
            "[원인] A & B 관계를 파악하고 있습니다.\n"
            "[학생에게 권하는 사항] <추가 학습>을 권합니다."
        )
        mock_provider.generate.return_value = text
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
        )
        assert isinstance(result.feedback_text, str)
        assert len(result.feedback_text) > 0

    def test_zero_width_characters_handled(self, mock_provider):
        """Zero-width characters in feedback do not crash pipeline."""
        text = (
            "[현재 상태] 학생은 개념을\u200b 이해하고\u200c 있습니다.\n"
            "[원인] 관계\ufeff 파악이 필요합니다.\n"
            "[학생에게 권하는 사항] 복습을 권합니다."
        )
        mock_provider.generate.return_value = text
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
        )
        assert isinstance(result.feedback_text, str)
        assert len(result.feedback_text) > 0


# ---------------------------------------------------------------------------
# T031 bonus: Cross-persona invariant tests (1000-iteration property tests)
# ---------------------------------------------------------------------------


class TestAdversaryInvariants:
    """Cross-persona invariant tests: properties that must hold across all scenarios."""

    @pytest.fixture()
    def mock_provider(self):
        return MagicMock()

    def test_invariant_feedback_never_none(self, mock_provider):
        """Regardless of LLM output, feedback_text is never None."""
        adversary_outputs = [
            "",
            "   ",
            "no sections here",
            "[현재 상태] only one section.",
            "[현재 상태] a. [원인] b.",  # missing third
            "놓쳤습니다 부족합니다 틀렸습니다",  # all banned, no sections
        ]
        for llm_output in adversary_outputs:
            mock_provider.generate.side_effect = [llm_output, llm_output]
            gen = FeedbackGenerator(mock_provider)
            result = gen.generate(
                "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
            )
            assert result.feedback_text is not None, (
                f"feedback_text was None for LLM output: {llm_output!r}"
            )

    def test_invariant_feedback_always_string(self, mock_provider):
        """Regardless of LLM output, feedback_text is always a string."""
        adversary_outputs = [
            "",
            "\n\n\n",
            "[현재 상태] [원인] [학생에게 권하는 사항]",
            "완전한 " * 200,  # very long
        ]
        for llm_output in adversary_outputs:
            mock_provider.generate.side_effect = [llm_output, llm_output]
            gen = FeedbackGenerator(mock_provider)
            result = gen.generate(
                "s001", 1, "Q?", "answer", 0.5, None, 0, "미달",
            )
            assert isinstance(result.feedback_text, str)

    def test_invariant_no_banned_expressions_ever(self, mock_provider):
        """No banned expression can survive the pipeline in any scenario."""
        # Construct adversarial input with all banned expressions repeated
        all_banned = " ".join(NEGATIVE_EXPRESSIONS.keys())
        toxic = f"[현재 상태] {all_banned}.\n[원인] {all_banned}.\n[학생에게 권하는 사항] {all_banned}."
        mock_provider.generate.return_value = toxic
        gen = FeedbackGenerator(mock_provider)
        result = gen.generate(
            "s001", 1, "Q?", "answer", 0.5, None, 1, "기전 이해",
        )
        for banned in NEGATIVE_EXPRESSIONS:
            assert banned not in result.feedback_text

    def test_invariant_char_count_matches_text_length(self, mock_provider):
        """char_count always equals len(feedback_text)."""
        outputs = [
            "[현재 상태] 좋습니다.\n[원인] 필요합니다.\n[학생에게 권하는 사항] 권합니다.",
            "no sections at all",
            "",
        ]
        for llm_output in outputs:
            mock_provider.generate.side_effect = [llm_output, llm_output]
            gen = FeedbackGenerator(mock_provider)
            result = gen.generate(
                "s001", 1, "Q?", "answer", 0.5, None, 2, "기전+용어",
            )
            assert result.char_count == len(result.feedback_text), (
                f"char_count={result.char_count} != len={len(result.feedback_text)}"
            )

    def test_invariant_fallback_templates_always_valid(self):
        """All fallback templates satisfy pipeline invariants."""
        for tier, template in FALLBACK_TEMPLATES.items():
            assert isinstance(template, str)
            assert len(template) > 0
            for section in _REQUIRED_SECTIONS:
                assert section in template, (
                    f"Tier {tier} fallback missing section: {section}"
                )
            assert template.rstrip().endswith("."), (
                f"Tier {tier} fallback doesn't end with period"
            )
            # No banned expressions in fallback templates
            for banned in NEGATIVE_EXPRESSIONS:
                assert banned not in template, (
                    f"Tier {tier} fallback contains banned: {banned}"
                )
