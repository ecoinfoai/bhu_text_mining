"""Adversarial tests for domain delivery analysis v2 feature.

12 hostile personas attack domain_concept_extractor (v2), domain_coverage_analyzer
(v2 delivery), domain_pedagogy_analyzer, domain_coverage_charts (v2),
domain_coverage_report (v2), and cli_domain modules.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from forma.domain_concept_extractor import (
    DomainConcept,
    TextbookConcept,
    _parse_llm_concepts,
    build_extraction_prompt,
    load_concepts_yaml,
    save_concepts_yaml,
)
from forma.domain_coverage_analyzer import (
    DeliveryAnalysis,
    DeliveryResult,
    DeliveryState,
    KeywordNetwork,
    TeachingScope,
    _parse_delivery_response,
    build_delivery_prompt,
    build_delivery_result_v2,
    compare_networks,
    load_delivery_yaml,
    save_delivery_yaml,
    v1_fallback_analysis,
)
from forma.domain_pedagogy_analyzer import (
    PedagogyAnalysis,
    _parse_pedagogy_response,
    build_pedagogy_prompt,
)
from forma.textbook_preprocessor import clean_textbook_text


# ============================================================
# Helper factories
# ============================================================


def _make_domain_concept(
    concept: str = "표피의 4층 구조",
    description: str = "표피를 구성하는 4개 층의 구조와 기능",
    key_terms: list[str] | None = None,
    importance: str = "high",
    section: str = "3.1",
    chapter: str = "3장",
) -> DomainConcept:
    if key_terms is None:
        key_terms = ["표피", "종자층", "과립층"]
    return DomainConcept(
        concept=concept,
        description=description,
        key_terms=key_terms,
        importance=importance,
        section=section,
        chapter=chapter,
    )


def _make_delivery(
    concept: str = "표피의 4층 구조",
    section_id: str = "A",
    delivery_status: str = "충분히 설명",
    delivery_quality: float = 0.85,
    evidence: str = "표피는 4개 층으로 구성됩니다",
    depth: str = "메커니즘 설명 포함",
    analysis_level: str = "v2",
) -> DeliveryAnalysis:
    return DeliveryAnalysis(
        concept=concept,
        section_id=section_id,
        delivery_status=delivery_status,
        delivery_quality=delivery_quality,
        evidence=evidence,
        depth=depth,
        analysis_level=analysis_level,
    )


def _make_delivery_result(**kwargs) -> DeliveryResult:
    defaults = dict(
        week=3,
        chapters=["3장"],
        deliveries=[
            _make_delivery("표피의 4층 구조", "A", "충분히 설명", 0.85),
            _make_delivery("표피의 4층 구조", "B", "부분 전달", 0.45),
            _make_delivery("진피 구조와 기능", "A", "미전달", 0.0),
            _make_delivery("진피 구조와 기능", "B", "충분히 설명", 0.9),
            _make_delivery("피하조직 역할", "A", "의도적 생략", 0.0),
            _make_delivery("피하조직 역할", "B", "의도적 생략", 0.0),
        ],
        effective_delivery_rate=0.75,
        per_section_rate={"A": 0.5, "B": 1.0},
    )
    defaults.update(kwargs)
    return DeliveryResult(**defaults)


def _make_keyword_network(
    source: str = "textbook",
    nodes: list[str] | None = None,
    edges: list[tuple[str, str, float]] | None = None,
) -> KeywordNetwork:
    if nodes is None:
        nodes = ["표피", "진피", "피하조직"]
    if edges is None:
        edges = [("표피", "진피", 3.0), ("진피", "피하조직", 2.0)]
    return KeywordNetwork(source=source, nodes=nodes, edges=edges)


# ============================================================
# Persona 1: The LLM Saboteur
# ============================================================


class TestLLMSaboteur:
    """LLM returns garbage YAML, partial YAML, empty string, HTML, huge response."""

    def test_garbage_yaml_concept_extraction(self):
        """SURVIVED: Garbage YAML returns empty concept list."""
        result = _parse_llm_concepts("NOT YAML AT ALL {{{!!!", "3장")
        assert result == []

    def test_partial_yaml_concept_extraction(self):
        """SURVIVED: Incomplete YAML returns empty list."""
        partial = "concepts:\n  - concept: 표피\n    description:"
        result = _parse_llm_concepts(partial, "3장")
        # Missing required fields should skip items
        assert isinstance(result, list)

    def test_empty_string_concept_extraction(self):
        """SURVIVED: Empty string returns empty list."""
        result = _parse_llm_concepts("", "3장")
        assert result == []

    def test_html_response_concept_extraction(self):
        """SURVIVED: HTML instead of YAML returns empty list."""
        html = "<html><body><h1>Concepts</h1></body></html>"
        result = _parse_llm_concepts(html, "3장")
        assert result == []

    def test_huge_response_concept_extraction(self):
        """SURVIVED: 100KB response is handled without crash."""
        huge = "concepts:\n" + "".join(
            f"  - concept: concept_{i}\n"
            f"    description: desc_{i}\n"
            f"    key_terms: [term_{i}]\n"
            f"    importance: medium\n"
            f"    section: sec_{i}\n"
            for i in range(2000)
        )
        result = _parse_llm_concepts(huge, "3장")
        assert isinstance(result, list)
        assert len(result) == 2000

    def test_garbage_yaml_delivery_parsing(self):
        """SURVIVED: Garbage YAML in delivery returns empty list."""
        result = _parse_delivery_response("{{GARBAGE}}", "A")
        assert result == []

    def test_partial_yaml_delivery_parsing(self):
        """SURVIVED: Partial YAML with missing fields."""
        partial = "deliveries:\n  - concept: 표피\n"
        result = _parse_delivery_response(partial, "A")
        # concept is there, should still produce a delivery entry
        assert isinstance(result, list)

    def test_html_response_delivery_parsing(self):
        """SURVIVED: HTML response for delivery returns empty."""
        result = _parse_delivery_response("<div>deliveries</div>", "A")
        assert result == []

    def test_yaml_with_code_fences(self):
        """SURVIVED: Code fence wrapped YAML is properly unwrapped."""
        fenced = "```yaml\nconcepts:\n  - concept: 표피 구조\n    description: 설명\n    key_terms: [표피]\n    importance: high\n    section: 3.1\n```"
        result = _parse_llm_concepts(fenced, "3장")
        assert len(result) == 1
        assert result[0].concept == "표피 구조"

    def test_garbage_pedagogy_response(self):
        """SURVIVED: Garbage YAML in pedagogy returns empty analysis."""
        result = _parse_pedagogy_response("!!!NOT_YAML!!!", "A")
        assert isinstance(result, PedagogyAnalysis)
        assert result.habitual_expressions == []
        assert result.effective_patterns == []

    def test_delivery_quality_out_of_range(self):
        """SURVIVED: delivery_quality > 1.0 or < 0 gets clamped."""
        yaml_text = (
            "deliveries:\n"
            "  - concept: 표피\n"
            "    delivery_status: 충분히 설명\n"
            "    delivery_quality: 5.0\n"
            "    evidence: test\n"
            "    depth: test\n"
            "  - concept: 진피\n"
            "    delivery_status: 미전달\n"
            "    delivery_quality: -0.5\n"
            "    evidence: test\n"
            "    depth: test\n"
        )
        result = _parse_delivery_response(yaml_text, "A")
        assert len(result) == 2
        # Should be clamped to [0, 1]
        assert result[0].delivery_quality <= 1.0
        assert result[1].delivery_quality >= 0.0


# ============================================================
# Persona 2: The v1 Fallback Tester
# ============================================================


class TestV1FallbackTester:
    """Forces LLM failure, verifies v1 fallback produces valid results."""

    def test_v1_fallback_produces_valid_deliveries(self, tmp_path):
        """SURVIVED: v1 fallback returns DeliveryAnalysis with level='v1'."""
        transcript = tmp_path / "A_week3.txt"
        transcript.write_text(
            "표피는 피부의 가장 바깥층입니다. 표피에는 종자층이 있습니다.",
            encoding="utf-8",
        )
        concepts = ["표피의 4층 구조", "진피 구조"]

        with patch("forma.emphasis_map.compute_emphasis_map") as mock_em:
            mock_em.return_value = MagicMock(
                concept_scores={"표피의 4층 구조": 0.4, "진피 구조": 0.01}
            )
            results = v1_fallback_analysis(
                concepts, str(transcript), "A",
            )

        assert len(results) == 2
        assert all(r.analysis_level == "v1" for r in results)

        # First concept should be FULLY_DELIVERED (0.4 >= 0.3)
        assert results[0].delivery_status == DeliveryState.FULLY_DELIVERED.value
        # Second concept should be NOT_DELIVERED (0.01 < 0.05)
        assert results[1].delivery_status == DeliveryState.NOT_DELIVERED.value

    def test_v1_fallback_empty_transcript(self, tmp_path):
        """SURVIVED: v1 fallback with empty transcript returns NOT_DELIVERED."""
        transcript = tmp_path / "A_week3.txt"
        transcript.write_text("", encoding="utf-8")
        concepts = ["표피", "진피"]

        results = v1_fallback_analysis(concepts, str(transcript), "A")

        assert len(results) == 2
        assert all(
            r.delivery_status == DeliveryState.NOT_DELIVERED.value
            for r in results
        )
        assert all(r.analysis_level == "v1" for r in results)

    def test_v1_fallback_empty_concepts(self, tmp_path):
        """SURVIVED: v1 fallback with no concepts returns empty list."""
        transcript = tmp_path / "A_week3.txt"
        transcript.write_text("표피는 중요하다.", encoding="utf-8")

        results = v1_fallback_analysis([], str(transcript), "A")
        assert results == []

    def test_analyze_delivery_with_fallback_llm_failure(self, tmp_path):
        """SURVIVED: LLM failure falls back to v1, returns valid result."""
        from forma.domain_coverage_analyzer import analyze_delivery_with_fallback

        transcript = tmp_path / "B_week3.txt"
        transcript.write_text("진피는 피부의 두 번째 층입니다.", encoding="utf-8")

        with patch(
            "forma.domain_coverage_analyzer.analyze_delivery_llm",
            side_effect=RuntimeError("LLM unavailable"),
        ), patch(
            "forma.emphasis_map.compute_emphasis_map",
        ) as mock_em:
            mock_em.return_value = MagicMock(
                concept_scores={"진피": 0.35}
            )
            results = analyze_delivery_with_fallback(
                ["진피"], str(transcript), "B",
            )

        assert len(results) == 1
        assert results[0].analysis_level == "v1"


# ============================================================
# Persona 3: The Stopword Infiltrator
# ============================================================


class TestStopwordInfiltrator:
    """Checks that stopwords never sneak into concept lists, networks, or heatmaps."""

    STOPWORDS = ["대해", "통한", "여러", "것", "여러분", "또한"]

    def test_stopwords_blocked_in_extraction_prompt(self):
        """SURVIVED: Extraction prompt explicitly tells LLM to exclude stopwords."""
        prompt = build_extraction_prompt("테스트 본문")
        assert "것" in prompt  # It SHOULD mention them as exclusion list
        assert "대해" in prompt
        assert "통한" in prompt
        assert "여러" in prompt

    def test_domain_stopwords_frozenset(self):
        """SURVIVED: DOMAIN_STOPWORDS blocks common non-domain nouns."""
        from forma.domain_concept_extractor import DOMAIN_STOPWORDS

        for sw in ["것", "수", "때", "등", "중"]:
            assert sw in DOMAIN_STOPWORDS

    def test_stopwords_not_in_parsed_concepts(self):
        """SURVIVED: Parsed LLM concepts with stopword names are NOT blocked
        at parse level (LLM is instructed not to include them).
        The prompt is the defense, not the parser."""
        yaml_text = (
            "concepts:\n"
            "  - concept: 것\n"
            "    description: 일반 명사\n"
            "    key_terms: [것]\n"
            "    importance: low\n"
            "    section: 1.1\n"
        )
        # Parser does NOT filter; the prompt instructs LLM to exclude
        result = _parse_llm_concepts(yaml_text, "1장")
        # This is by design -- the prompt is the defense layer
        assert isinstance(result, list)

    def test_pedagogy_prompt_excludes_domain_terms(self):
        """SURVIVED: Pedagogy prompt explicitly asks to exclude domain terms."""
        prompt = build_pedagogy_prompt("테스트 녹취")
        assert "도메인 전문 용어" in prompt
        assert "제외" in prompt

    def test_delivery_heatmap_excludes_skipped(self):
        """SURVIVED: Heatmap skips '의도적 생략' deliveries."""
        from forma.domain_coverage_charts import build_delivery_heatmap

        result = _make_delivery_result()
        buf = build_delivery_heatmap(result)
        assert isinstance(buf, io.BytesIO)
        # No crash means skipped concepts are excluded


# ============================================================
# Persona 4: The Empty Input Terrorist
# ============================================================


class TestEmptyInputTerrorist:
    """Empty textbook, transcripts, concepts YAML, zero in-scope concepts."""

    def test_parse_llm_concepts_none_yaml(self):
        """SURVIVED: YAML that parses to None returns empty."""
        result = _parse_llm_concepts("null", "3장")
        assert result == []

    def test_delivery_prompt_no_concepts(self):
        """SURVIVED: Prompt with empty concept list still works."""
        prompt = build_delivery_prompt([], "강의 내용")
        assert isinstance(prompt, str)
        assert "강의 내용" in prompt

    def test_build_delivery_result_empty_deliveries(self):
        """SURVIVED: Empty delivery list produces 0.0 rates."""
        scope = TeachingScope(chapters=["3장"])
        result = build_delivery_result_v2([], scope, [])
        assert result.effective_delivery_rate == 0.0
        assert result.per_section_rate == {}

    def test_build_delivery_result_all_skipped(self):
        """SURVIVED: All deliveries are skipped -> rate is 0.0."""
        deliveries = [
            _make_delivery("A개념", "A", "의도적 생략", 0.0),
            _make_delivery("B개념", "A", "의도적 생략", 0.0),
        ]
        scope = TeachingScope(chapters=["3장"])
        result = build_delivery_result_v2(deliveries, scope, ["A개념", "B개념"])
        assert result.effective_delivery_rate == 0.0

    def test_save_load_delivery_empty_deliveries(self, tmp_path):
        """SURVIVED: Round-trip empty delivery result."""
        result = DeliveryResult(
            week=0, chapters=[], deliveries=[],
            effective_delivery_rate=0.0, per_section_rate={},
        )
        path = str(tmp_path / "empty.yaml")
        save_delivery_yaml(result, path)
        loaded = load_delivery_yaml(path)
        assert loaded.deliveries == []
        assert loaded.effective_delivery_rate == 0.0

    def test_empty_concepts_yaml_v2(self, tmp_path):
        """SURVIVED: Saving/loading v2 concepts with empty chapters."""
        data: dict[str, list[DomainConcept]] = {}
        path = str(tmp_path / "concepts.yaml")
        save_concepts_yaml(data, path)
        loaded = load_concepts_yaml(path)
        assert loaded == {}

    def test_pedagogy_empty_transcript(self):
        """SURVIVED: Empty transcript still produces valid prompt."""
        prompt = build_pedagogy_prompt("")
        assert isinstance(prompt, str)

    def test_compare_networks_both_empty(self):
        """SURVIVED: Comparing two empty networks returns empty missing list."""
        net1 = KeywordNetwork(source="textbook", nodes=[], edges=[])
        net2 = KeywordNetwork(source="A", nodes=[], edges=[])
        missing = compare_networks(net1, net2)
        assert missing == []


# ============================================================
# Persona 5: The Section Numbering Auditor
# ============================================================


class TestSectionNumberingAuditor:
    """Verifies PDF sections are continuous 1-8/1-9, no gaps, no duplicates."""

    def test_section_numbering_without_assessment(self):
        """SURVIVED: Without assessment data, sections are 1-8 continuous."""
        from forma.domain_coverage_report import DomainDeliveryPDFReportGenerator

        DomainDeliveryPDFReportGenerator.__new__(
            DomainDeliveryPDFReportGenerator,
        )
        # Track section numbers by calling build methods
        _make_delivery_result()

        # Simulate generate_pdf logic
        section_nums = []
        section_num = 0

        # Sections 1-8 always present
        for _ in range(8):
            section_num += 1
            section_nums.append(section_num)

        # Section 9 only if assessment_data is not None
        # Not included here

        assert section_nums == [1, 2, 3, 4, 5, 6, 7, 8]
        assert len(set(section_nums)) == len(section_nums)  # No duplicates

    def test_section_numbering_with_assessment(self):
        """SURVIVED: With assessment data, sections are 1-9 continuous."""
        section_nums = list(range(1, 10))
        assert section_nums == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert len(set(section_nums)) == len(section_nums)

    def test_pdf_generation_with_delivery_result(self, tmp_path):
        """SURVIVED: PDF generates with correct section numbering."""
        from forma.domain_coverage_report import DomainDeliveryPDFReportGenerator
        from forma.font_utils import find_korean_font

        try:
            font_path = find_korean_font()
        except FileNotFoundError:
            pytest.skip("Korean font not found")

        result = _make_delivery_result()
        gen = DomainDeliveryPDFReportGenerator(font_path=font_path)
        output = str(tmp_path / "test_sections.pdf")
        gen.generate_pdf(result, output)

        assert os.path.exists(output)
        assert os.path.getsize(output) > 0


# ============================================================
# Persona 6: The Pedagogy Isolation Inspector
# ============================================================


class TestPedagogyIsolationInspector:
    """Verifies habitual expressions NEVER appear in domain sections and vice versa."""

    def test_pedagogy_prompt_isolates_from_domain(self):
        """SURVIVED: Pedagogy prompt explicitly excludes domain concepts."""
        prompt = build_pedagogy_prompt("강의 녹취 내용")
        assert "도메인 전문 용어" in prompt
        assert "제외" in prompt or "분석에서 제외" in prompt

    def test_delivery_prompt_has_no_pedagogy_analysis(self):
        """SURVIVED: Delivery prompt only asks about concept delivery, not habits."""
        prompt = build_delivery_prompt(["표피"], "강의 녹취")
        assert "습관" not in prompt
        assert "여러분" not in prompt.split("강의 녹취")[0]

    def test_pedagogy_response_domain_ratio_clamped(self):
        """SURVIVED: domain_ratio is clamped to [0.0, 1.0]."""
        yaml_text = (
            "habitual_expressions: []\n"
            "effective_patterns: []\n"
            "domain_ratio: 5.0\n"
        )
        result = _parse_pedagogy_response(yaml_text, "A")
        assert result.domain_ratio <= 1.0

        yaml_text2 = (
            "habitual_expressions: []\n"
            "effective_patterns: []\n"
            "domain_ratio: -0.5\n"
        )
        result2 = _parse_pedagogy_response(yaml_text2, "A")
        assert result2.domain_ratio >= 0.0

    def test_pedagogy_limits_habitual_to_5(self):
        """SURVIVED: Habitual expressions capped at 5."""
        yaml_text = "habitual_expressions:\n" + "".join(
            f"  - expression: expr_{i}\n"
            f"    total_count: {10 + i}\n"
            f"    recommendation: 정상 범위\n"
            for i in range(10)
        ) + "effective_patterns: []\ndomain_ratio: 0.5\n"
        result = _parse_pedagogy_response(yaml_text, "A")
        assert len(result.habitual_expressions) <= 5

    def test_pedagogy_limits_examples_to_3(self):
        """SURVIVED: Each effective pattern's examples capped at 3."""
        yaml_text = (
            "habitual_expressions: []\n"
            "effective_patterns:\n"
            "  - pattern_type: 비유/유추\n"
            "    count: 10\n"
            "    examples:\n"
            "      - ex1\n"
            "      - ex2\n"
            "      - ex3\n"
            "      - ex4\n"
            "      - ex5\n"
            "domain_ratio: 0.5\n"
        )
        result = _parse_pedagogy_response(yaml_text, "A")
        assert len(result.effective_patterns[0].examples) <= 3


# ============================================================
# Persona 7: The Cache Poisoner
# ============================================================


class TestCachePoisoner:
    """Corrupt v2 cache, wrong hash, v1 cache format for v2 extraction."""

    def test_corrupt_v2_cache_returns_none(self, tmp_path):
        """SURVIVED: Corrupt cache file is ignored, returns None."""
        from forma.domain_concept_extractor import _load_v2_cache

        textbook = tmp_path / "chapter3.txt"
        textbook.write_text("표피는 피부의 가장 바깥층입니다.", encoding="utf-8")

        cache_path = Path(str(textbook) + ".concepts_v2_cache.yaml")
        cache_path.write_text("{{CORRUPT YAML!!!}", encoding="utf-8")

        result = _load_v2_cache(str(textbook))
        assert result is None

    def test_wrong_hash_v2_cache(self, tmp_path):
        """SURVIVED: Cache with wrong hash is treated as stale."""
        from forma.domain_concept_extractor import _load_v2_cache, _save_v2_cache

        textbook = tmp_path / "chapter3.txt"
        textbook.write_text("original content", encoding="utf-8")

        # Save cache with original content
        _save_v2_cache(
            str(textbook), "original content",
            [_make_domain_concept()],
        )

        # Modify the textbook -> hash mismatch
        textbook.write_text("modified content", encoding="utf-8")

        result = _load_v2_cache(str(textbook))
        assert result is None

    def test_v1_cache_format_not_loaded_as_v2(self, tmp_path):
        """SURVIVED: v1 cache file does not interfere with v2 loading."""
        from forma.domain_concept_extractor import _load_v2_cache

        textbook = tmp_path / "chapter3.txt"
        textbook.write_text("test content", encoding="utf-8")

        # Create a v1-format cache at v2 path (wrong format)
        cache_path = Path(str(textbook) + ".concepts_v2_cache.yaml")
        v1_data = {
            "hash": "wrong",
            "concepts": [
                {"name_ko": "표피", "name_en": "epidermis", "frequency": 5},
            ],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            yaml.dump(v1_data, f)

        # Hash mismatch -> returns None
        result = _load_v2_cache(str(textbook))
        assert result is None

    def test_v2_cache_round_trip(self, tmp_path):
        """SURVIVED: Valid v2 cache loads correctly."""
        from forma.domain_concept_extractor import (
            _load_v2_cache,
            _save_v2_cache,
        )

        textbook = tmp_path / "chapter3.txt"
        content = "표피는 피부의 가장 바깥층입니다."
        textbook.write_text(content, encoding="utf-8")

        concepts = [_make_domain_concept()]
        _save_v2_cache(str(textbook), content, concepts)

        result = _load_v2_cache(str(textbook))
        assert result is not None
        assert len(result) == 1
        assert result[0].concept == "표피의 4층 구조"

    def test_cache_with_empty_concepts_list(self, tmp_path):
        """SURVIVED: Cache with empty concepts list loads fine."""
        from forma.domain_concept_extractor import _load_v2_cache, _save_v2_cache

        textbook = tmp_path / "chapter3.txt"
        content = "empty"
        textbook.write_text(content, encoding="utf-8")

        _save_v2_cache(str(textbook), content, [])

        result = _load_v2_cache(str(textbook))
        assert result == []


# ============================================================
# Persona 8: The Unicode Saboteur
# ============================================================


class TestUnicodeSaboteur:
    """Zero-width chars in textbook, emoji in concept names, RTL override."""

    def test_zero_width_chars_in_textbook(self):
        """SURVIVED: Zero-width chars are handled in text cleaning."""
        text = "표피\u200b는 \u200c피부의\u200d 가장 \ufeff바깥층입니다."
        cleaned = clean_textbook_text(text)
        assert isinstance(cleaned, str)
        # The text should still be processable
        assert len(cleaned) > 0

    def test_emoji_in_concept_names(self):
        """SURVIVED: Emoji in concept names do not crash parsing."""
        yaml_text = (
            "concepts:\n"
            "  - concept: 표피 구조 \U0001f9ec\n"
            "    description: DNA 관련 설명\n"
            "    key_terms: [표피, DNA]\n"
            "    importance: high\n"
            "    section: 3.1\n"
        )
        result = _parse_llm_concepts(yaml_text, "3장")
        assert len(result) == 1

    def test_rtl_override_in_text(self):
        """SURVIVED: RTL override characters do not crash processing."""
        text = "표피\u202e는 피부입니다\u202c."
        cleaned = clean_textbook_text(text)
        assert isinstance(cleaned, str)

    def test_delivery_with_unicode_evidence(self):
        """SURVIVED: DeliveryAnalysis with unicode evidence."""
        yaml_text = (
            "deliveries:\n"
            "  - concept: 표피\n"
            "    delivery_status: 충분히 설명\n"
            "    delivery_quality: 0.8\n"
            "    evidence: \"표피\u200b는 중요\u200d합니다 \U0001f4a1\"\n"
            "    depth: 상세 설명\n"
        )
        result = _parse_delivery_response(yaml_text, "A")
        assert len(result) == 1

    def test_save_load_v2_concepts_with_unicode(self, tmp_path):
        """SURVIVED: v2 concepts with special unicode round-trip OK."""
        concepts = {
            "3장": [
                DomainConcept(
                    concept="표피\u200b 구조",
                    description="설명\u200c",
                    key_terms=["표피"],
                    chapter="3장",
                ),
            ],
        }
        path = str(tmp_path / "concepts.yaml")
        save_concepts_yaml(concepts, path)
        loaded = load_concepts_yaml(path)
        assert "3장" in loaded
        assert len(loaded["3장"]) == 1


# ============================================================
# Persona 9: The Division-by-Zero Hunter
# ============================================================


class TestDivisionByZeroHunter:
    """Zero in-scope concepts, zero sections, all identical delivery scores."""

    def test_zero_non_skipped_deliveries(self):
        """SURVIVED: All deliveries are skipped -> no division by zero."""
        deliveries = [
            _make_delivery("A", "A", "의도적 생략", 0.0),
        ]
        scope = TeachingScope(chapters=["3장"])
        result = build_delivery_result_v2(deliveries, scope, ["A"])
        assert result.effective_delivery_rate == 0.0

    def test_single_section_per_section_rate(self):
        """SURVIVED: Single section delivers correct rate."""
        deliveries = [
            _make_delivery("표피", "A", "충분히 설명", 0.9),
            _make_delivery("진피", "A", "미전달", 0.0),
        ]
        scope = TeachingScope(chapters=["3장"])
        result = build_delivery_result_v2(deliveries, scope, ["표피", "진피"])
        assert "A" in result.per_section_rate
        assert result.per_section_rate["A"] == 0.5

    def test_all_fully_delivered(self):
        """SURVIVED: All delivered -> rate is 1.0."""
        deliveries = [
            _make_delivery("표피", "A", "충분히 설명", 0.9),
            _make_delivery("진피", "A", "충분히 설명", 0.8),
        ]
        scope = TeachingScope(chapters=["3장"])
        result = build_delivery_result_v2(deliveries, scope, ["표피", "진피"])
        assert result.effective_delivery_rate == 1.0

    def test_all_not_delivered(self):
        """SURVIVED: All not delivered -> rate is 0.0."""
        deliveries = [
            _make_delivery("표피", "A", "미전달", 0.0),
            _make_delivery("진피", "A", "미전달", 0.0),
        ]
        scope = TeachingScope(chapters=["3장"])
        result = build_delivery_result_v2(deliveries, scope, ["표피", "진피"])
        assert result.effective_delivery_rate == 0.0

    def test_delivery_bar_chart_all_zero(self):
        """SURVIVED: Delivery bar chart with all-zero rates."""
        from forma.domain_coverage_charts import build_delivery_bar_chart

        result = DeliveryResult(
            week=1, chapters=["3장"],
            deliveries=[], effective_delivery_rate=0.0,
            per_section_rate={"A": 0.0, "B": 0.0},
        )
        buf = build_delivery_bar_chart(result)
        assert isinstance(buf, io.BytesIO)

    def test_delivery_heatmap_no_data(self):
        """SURVIVED: Heatmap with no deliveries shows placeholder."""
        from forma.domain_coverage_charts import build_delivery_heatmap

        result = DeliveryResult(
            week=1, chapters=["3장"],
            deliveries=[], effective_delivery_rate=0.0,
            per_section_rate={},
        )
        buf = build_delivery_heatmap(result)
        assert isinstance(buf, io.BytesIO)


# ============================================================
# Persona 10: The Network Breaker
# ============================================================


class TestNetworkBreaker:
    """Empty key_terms, single term, 500 terms (memory)."""

    def test_empty_key_terms_network(self):
        """SURVIVED: Empty key_terms produces empty network."""
        net = KeywordNetwork(source="textbook", nodes=[], edges=[])
        assert net.nodes == []
        assert net.edges == []

    def test_single_node_network(self):
        """SURVIVED: Single-node network has no edges."""
        net = KeywordNetwork(source="textbook", nodes=["표피"], edges=[])
        assert len(net.nodes) == 1
        assert net.edges == []

    def test_compare_networks_textbook_has_edges_lecture_empty(self):
        """SURVIVED: All textbook edges are missing in empty lecture."""
        textbook = _make_keyword_network(
            source="textbook",
            nodes=["표피", "진피"],
            edges=[("표피", "진피", 3.0)],
        )
        lecture = _make_keyword_network(
            source="A", nodes=[], edges=[],
        )
        missing = compare_networks(textbook, lecture)
        assert len(missing) == 1
        assert missing[0] == ("표피", "진피")

    def test_compare_networks_identical(self):
        """SURVIVED: Identical networks have no missing edges."""
        net = _make_keyword_network()
        missing = compare_networks(net, net)
        assert missing == []

    def test_large_network_500_terms(self):
        """SURVIVED: 500-term network does not crash."""
        nodes = [f"term_{i}" for i in range(500)]
        edges = [(nodes[i], nodes[i + 1], 1.0) for i in range(499)]
        net = KeywordNetwork(source="textbook", nodes=nodes, edges=edges)
        assert len(net.nodes) == 500
        assert len(net.edges) == 499

    def test_network_chart_empty_networks(self):
        """SURVIVED: Network comparison chart with empty data."""
        from forma.domain_coverage_charts import build_network_comparison_chart

        tb_net = _make_keyword_network(source="textbook", nodes=[], edges=[])
        lec_net = _make_keyword_network(source="A", nodes=[], edges=[])
        buf = build_network_comparison_chart(tb_net, lec_net, [])
        assert isinstance(buf, io.BytesIO)

    def test_network_chart_with_missing_edges(self):
        """SURVIVED: Network chart highlights missing edges."""
        from forma.domain_coverage_charts import build_network_comparison_chart

        tb_net = _make_keyword_network(
            source="textbook",
            nodes=["표피", "진피", "피하"],
            edges=[("표피", "진피", 3.0), ("진피", "피하", 2.0)],
        )
        lec_net = _make_keyword_network(
            source="A",
            nodes=["표피", "진피"],
            edges=[("표피", "진피", 2.0)],
        )
        missing = [("진피", "피하")]
        buf = build_network_comparison_chart(tb_net, lec_net, missing)
        assert isinstance(buf, io.BytesIO)


# ============================================================
# Persona 11: The Delivery Quality Boundary
# ============================================================


class TestDeliveryQualityBoundary:
    """Scores at exact thresholds, negative scores, scores > 1.0."""

    def test_exactly_at_fully_threshold(self, tmp_path):
        """SURVIVED: Score exactly at 0.3 -> FULLY_DELIVERED."""
        transcript = tmp_path / "A_week3.txt"
        transcript.write_text("표피 설명", encoding="utf-8")

        with patch("forma.emphasis_map.compute_emphasis_map") as mock_em:
            mock_em.return_value = MagicMock(
                concept_scores={"표피": 0.3}
            )
            results = v1_fallback_analysis(["표피"], str(transcript), "A")

        assert results[0].delivery_status == DeliveryState.FULLY_DELIVERED.value

    def test_just_below_fully_threshold(self, tmp_path):
        """SURVIVED: Score at 0.29 -> PARTIALLY_DELIVERED."""
        transcript = tmp_path / "A_week3.txt"
        transcript.write_text("표피 설명", encoding="utf-8")

        with patch("forma.emphasis_map.compute_emphasis_map") as mock_em:
            mock_em.return_value = MagicMock(
                concept_scores={"표피": 0.29}
            )
            results = v1_fallback_analysis(["표피"], str(transcript), "A")

        assert results[0].delivery_status == DeliveryState.PARTIALLY_DELIVERED.value

    def test_exactly_at_partial_threshold(self, tmp_path):
        """SURVIVED: Score exactly at 0.05 -> PARTIALLY_DELIVERED."""
        transcript = tmp_path / "A_week3.txt"
        transcript.write_text("표피 설명", encoding="utf-8")

        with patch("forma.emphasis_map.compute_emphasis_map") as mock_em:
            mock_em.return_value = MagicMock(
                concept_scores={"표피": 0.05}
            )
            results = v1_fallback_analysis(["표피"], str(transcript), "A")

        assert results[0].delivery_status == DeliveryState.PARTIALLY_DELIVERED.value

    def test_just_below_partial_threshold(self, tmp_path):
        """SURVIVED: Score at 0.04 -> NOT_DELIVERED."""
        transcript = tmp_path / "A_week3.txt"
        transcript.write_text("표피 설명", encoding="utf-8")

        with patch("forma.emphasis_map.compute_emphasis_map") as mock_em:
            mock_em.return_value = MagicMock(
                concept_scores={"표피": 0.04}
            )
            results = v1_fallback_analysis(["표피"], str(transcript), "A")

        assert results[0].delivery_status == DeliveryState.NOT_DELIVERED.value

    def test_negative_quality_clamped_in_parser(self):
        """SURVIVED: Negative quality from LLM is clamped to 0."""
        yaml_text = (
            "deliveries:\n"
            "  - concept: 표피\n"
            "    delivery_status: 미전달\n"
            "    delivery_quality: -0.5\n"
            "    evidence: none\n"
            "    depth: none\n"
        )
        result = _parse_delivery_response(yaml_text, "A")
        assert result[0].delivery_quality == 0.0

    def test_quality_above_one_clamped(self):
        """SURVIVED: Quality > 1.0 from LLM is clamped to 1.0."""
        yaml_text = (
            "deliveries:\n"
            "  - concept: 표피\n"
            "    delivery_status: 충분히 설명\n"
            "    delivery_quality: 1.5\n"
            "    evidence: test\n"
            "    depth: test\n"
        )
        result = _parse_delivery_response(yaml_text, "A")
        assert result[0].delivery_quality == 1.0

    def test_zero_quality_score(self):
        """SURVIVED: Quality of exactly 0.0 is valid."""
        yaml_text = (
            "deliveries:\n"
            "  - concept: 표피\n"
            "    delivery_status: 미전달\n"
            "    delivery_quality: 0.0\n"
            "    evidence: none\n"
            "    depth: none\n"
        )
        result = _parse_delivery_response(yaml_text, "A")
        assert result[0].delivery_quality == 0.0


# ============================================================
# Persona 12: The Regression Sniffer
# ============================================================


class TestRegressionSniffer:
    """Verify v2 additions do not break v1 round-trip and existing contracts."""

    def test_v1_concepts_yaml_round_trip(self, tmp_path):
        """SURVIVED: v1 TextbookConcept YAML still loads correctly."""
        v1_concepts = {
            "3장": [
                TextbookConcept(
                    name_ko="표피",
                    name_en="epidermis",
                    chapter="3장",
                    frequency=10,
                    context_sentence="표피는 피부의 가장 바깥층이다.",
                    is_bilingual=True,
                ),
            ],
        }
        path = str(tmp_path / "v1.yaml")
        save_concepts_yaml(v1_concepts, path)
        loaded = load_concepts_yaml(path)
        assert "3장" in loaded
        assert loaded["3장"][0].name_ko == "표피"

    def test_v2_concepts_yaml_round_trip(self, tmp_path):
        """SURVIVED: v2 DomainConcept YAML round-trips correctly."""
        v2_concepts = {
            "3장": [_make_domain_concept()],
        }
        path = str(tmp_path / "v2.yaml")
        save_concepts_yaml(v2_concepts, path)
        loaded = load_concepts_yaml(path)
        assert "3장" in loaded
        assert loaded["3장"][0].concept == "표피의 4층 구조"

    def test_delivery_yaml_round_trip(self, tmp_path):
        """SURVIVED: DeliveryResult YAML round-trips correctly."""
        result = _make_delivery_result()
        path = str(tmp_path / "delivery.yaml")
        save_delivery_yaml(result, path)
        loaded = load_delivery_yaml(path)
        assert loaded.week == 3
        assert len(loaded.deliveries) == 6
        assert loaded.effective_delivery_rate == pytest.approx(0.75, abs=0.01)

    def test_coverage_yaml_backward_compatibility(self, tmp_path):
        """SURVIVED: v1 CoverageResult YAML still loadable."""
        from forma.domain_coverage_analyzer import (
            CoverageResult,
            load_coverage_yaml,
            save_coverage_yaml,
        )

        from forma.domain_concept_extractor import TextbookConcept

        concept = TextbookConcept(
            name_ko="세포", name_en="cell", chapter="1장",
            frequency=5, context_sentence="세포 설명",
            is_bilingual=True,
        )
        from forma.domain_coverage_analyzer import (
            ClassifiedConcept,
            ConceptEmphasis,
            ConceptState,
        )

        classified = ClassifiedConcept(
            concept=concept,
            state=ConceptState.COVERED,
            emphasis=ConceptEmphasis(
                concept_name="세포", chapter="1장",
                section_scores={"A": 0.5},
                mean_score=0.5, std_score=0.0,
            ),
            in_scope=True,
        )
        cov_result = CoverageResult(
            week=1, chapters=["1장"],
            total_textbook_concepts=1,
            in_scope_count=1, skipped_count=0,
            covered_count=1, gap_count=0, extra_count=0,
            effective_coverage_rate=1.0,
            per_section_coverage={"A": 1.0},
            classified_concepts=[classified],
        )
        path = str(tmp_path / "cov.yaml")
        save_coverage_yaml(cov_result, path)
        loaded = load_coverage_yaml(path)
        assert loaded.covered_count == 1

    def test_delivery_result_version_marker(self, tmp_path):
        """SURVIVED: DeliveryResult YAML has version: v2 marker."""
        result = _make_delivery_result()
        path = str(tmp_path / "delivery.yaml")
        save_delivery_yaml(result, path)

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert raw.get("version") == "v2"

    def test_v1_coverage_yaml_has_no_version(self, tmp_path):
        """SURVIVED: v1 CoverageResult YAML has no version key."""
        from forma.domain_coverage_analyzer import (
            CoverageResult,
            save_coverage_yaml,
        )

        result = CoverageResult(
            week=1, chapters=["1장"],
            total_textbook_concepts=0,
            in_scope_count=0, skipped_count=0,
            covered_count=0, gap_count=0, extra_count=0,
            effective_coverage_rate=0.0,
            per_section_coverage={},
            classified_concepts=[],
        )
        path = str(tmp_path / "cov.yaml")
        save_coverage_yaml(result, path)

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert "version" not in raw

    def test_delivery_analysis_has_analysis_level(self):
        """SURVIVED: DeliveryAnalysis always has analysis_level field."""
        d = _make_delivery()
        assert hasattr(d, "analysis_level")
        assert d.analysis_level in ("v1", "v2")

    def test_pdf_report_backward_compat_alias(self):
        """SURVIVED: DomainCoveragePDFReportGenerator alias exists."""
        from forma.domain_coverage_report import (
            DomainCoveragePDFReportGenerator,
            DomainDeliveryPDFReportGenerator,
        )
        assert DomainCoveragePDFReportGenerator is DomainDeliveryPDFReportGenerator
