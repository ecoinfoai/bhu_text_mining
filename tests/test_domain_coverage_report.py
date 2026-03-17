"""Tests for domain coverage PDF report generation.

T031: PDF generation tests
"""

from __future__ import annotations

import statistics
from pathlib import Path


from forma.domain_concept_extractor import TextbookConcept
from forma.domain_coverage_analyzer import (
    ConceptEmphasis,
    CoverageResult,
    ExtraConcept,
    TeachingScope,
    build_coverage_result,
    classify_concepts,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _make_concept(
    name_ko: str,
    chapter: str = "3장 피부",
    frequency: int = 5,
    name_en: str | None = None,
) -> TextbookConcept:
    return TextbookConcept(
        name_ko=name_ko,
        name_en=name_en,
        chapter=chapter,
        frequency=frequency,
        context_sentence=f"{name_ko}는 중요하다.",
        is_bilingual=name_en is not None,
    )


def _make_emphasis(
    concept_name: str,
    chapter: str = "3장 피부",
    section_scores: dict[str, float] | None = None,
) -> ConceptEmphasis:
    if section_scores is None:
        section_scores = {"A": 0.5, "B": 0.6}
    values = list(section_scores.values())
    mean_score = statistics.mean(values) if values else 0.0
    std_score = statistics.stdev(values) if len(values) >= 2 else 0.0
    return ConceptEmphasis(
        concept_name=concept_name,
        chapter=chapter,
        section_scores=section_scores,
        mean_score=mean_score,
        std_score=std_score,
    )


def _build_sample_result() -> CoverageResult:
    """Build a sample CoverageResult with typical data."""
    concepts = [
        _make_concept("표피", chapter="3장", frequency=15, name_en="epidermis"),
        _make_concept("진피", chapter="3장", frequency=10, name_en="dermis"),
        _make_concept("각질층", chapter="3장", frequency=8),
        _make_concept("세포막", chapter="1장", frequency=5),
        _make_concept("근육", chapter="5장", frequency=3),
    ]
    emphasis_list = [
        _make_emphasis("표피", chapter="3장", section_scores={"A": 0.9, "B": 0.7, "C": 0.8}),
        _make_emphasis("진피", chapter="3장", section_scores={"A": 0.6, "B": 0.4, "C": 0.5}),
        _make_emphasis("각질층", chapter="3장", section_scores={"A": 0.3, "B": 0.1, "C": 0.2}),
        _make_emphasis("세포막", chapter="1장", section_scores={"A": 0.02, "B": 0.01, "C": 0.01}),
    ]
    scope = TeachingScope(chapters=["3장", "1장"])
    classified = classify_concepts(concepts, emphasis_list, scope)
    extras = [ExtraConcept(
        name="염증",
        section_mentions={"A": 5, "B": 3},
        example_sentence="염증 반응이 피부에서 나타났다.",
    )]
    return build_coverage_result(classified, extras, week=2, chapters=["3장", "1장"])


# ----------------------------------------------------------------
# T031: PDF generation
# ----------------------------------------------------------------


class TestDomainCoveragePDFReport:
    """Tests for DomainCoveragePDFReportGenerator."""

    def test_pdf_exists_and_nonzero(self, tmp_path) -> None:
        """Generated PDF exists and is non-zero."""
        from forma.domain_coverage_report import DomainCoveragePDFReportGenerator

        result = _build_sample_result()
        output = str(tmp_path / "report.pdf")

        gen = DomainCoveragePDFReportGenerator()
        path = gen.generate_pdf(result, output, course_name="인체구조와기능")

        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_pdf_with_typical_data(self, tmp_path) -> None:
        """PDF with typical data generates without error."""
        from forma.domain_coverage_report import DomainCoveragePDFReportGenerator

        result = _build_sample_result()
        output = str(tmp_path / "report.pdf")

        gen = DomainCoveragePDFReportGenerator()
        path = gen.generate_pdf(result, output)

        assert Path(path).exists()
        # Check file starts with PDF magic bytes
        with open(path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_pdf_with_empty_concepts(self, tmp_path) -> None:
        """PDF with empty concepts handles gracefully."""
        from forma.domain_coverage_report import DomainCoveragePDFReportGenerator

        result = CoverageResult(
            week=1,
            chapters=["1장"],
            total_textbook_concepts=0,
            in_scope_count=0,
            skipped_count=0,
            covered_count=0,
            gap_count=0,
            extra_count=0,
            effective_coverage_rate=0.0,
            per_section_coverage={},
            classified_concepts=[],
        )
        output = str(tmp_path / "empty_report.pdf")

        gen = DomainCoveragePDFReportGenerator()
        path = gen.generate_pdf(result, output)

        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_pdf_no_course_name(self, tmp_path) -> None:
        """PDF without course name generates correctly."""
        from forma.domain_coverage_report import DomainCoveragePDFReportGenerator

        result = _build_sample_result()
        output = str(tmp_path / "report.pdf")

        gen = DomainCoveragePDFReportGenerator()
        path = gen.generate_pdf(result, output)

        assert Path(path).exists()
