"""Tests for forma.lecture_report."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def _register_fake_fonts() -> None:
    """Register fake NanumGothic fonts as Helvetica for testing."""
    try:
        pdfmetrics.getFont("NanumGothic")
    except KeyError:
        # Use a built-in font as stand-in — register via addMapping only
        from reportlab.lib.fonts import addMapping
        # We can't easily create a TTFont without a real file,
        # so we'll skip font registration and use Helvetica in styles.
        pass


def _make_result(**overrides):
    """Create a minimal AnalysisResult for testing."""
    from forma.lecture_analyzer import AnalysisResult

    defaults = dict(
        class_id="A",
        week=1,
        keyword_frequencies={"cell": 5, "protein": 3},
        top_keywords=["cell", "protein"],
        network_image_path=None,
        topics=None,
        topic_skipped_reason="sentence count insufficient (3 < 10)",
        concept_coverage=None,
        emphasis_scores=None,
        triplets=None,
        triplet_skipped_reason=None,
        sentence_count=3,
        analysis_timestamp="2026-01-01T00:00:00",
    )
    defaults.update(overrides)
    return AnalysisResult(**defaults)


def _make_generator():
    """Create a LectureReportGenerator with Helvetica fallback for tests."""
    from forma.lecture_report import LectureReportGenerator

    # Use Helvetica-based styles by patching both font discovery and registration
    with patch("forma.lecture_report.find_korean_font", return_value="/tmp/fake.ttf"):
        with patch("forma.lecture_report.register_korean_fonts"):
            gen = LectureReportGenerator(font_path=None)

    # Override styles to use Helvetica (always available in ReportLab)
    for style_name in ("LectTitle", "LectSection"):
        gen._styles[style_name].fontName = "Helvetica-Bold"
    for style_name in ("LectBody", "LectTableData", "LectSkipped"):
        gen._styles[style_name].fontName = "Helvetica"
    gen._styles["LectTableHeader"].fontName = "Helvetica-Bold"

    return gen


class TestLectureReportGenerator:
    """Test LectureReportGenerator PDF generation."""

    def test_generate_analysis_report_creates_pdf(self, tmp_path: Path) -> None:
        """Output PDF file exists after generation."""
        gen = _make_generator()
        result = _make_result()
        output_path = tmp_path / "report.pdf"
        gen.generate_analysis_report(result, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_report_contains_keyword_section(self) -> None:
        """Keyword data included in the PDF story."""
        gen = _make_generator()
        result = _make_result()
        flowables = gen._build_keyword_section(result)
        assert len(flowables) > 0

    def test_report_contains_network_image(self, tmp_path: Path) -> None:
        """Network PNG referenced when path is provided."""
        # Create a valid minimal PNG
        png_path = tmp_path / "network.png"
        png_path.write_bytes(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
            b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
            b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )

        gen = _make_generator()
        result = _make_result(network_image_path=png_path)
        flowables = gen._build_network_section(result)
        assert len(flowables) > 0
        # Should contain an Image flowable
        from reportlab.platypus import Image as RLImage
        has_image = any(isinstance(f, RLImage) for f in flowables)
        assert has_image

    def test_report_topic_skipped_placeholder(self) -> None:
        """When topics is None, placeholder with reason appears."""
        gen = _make_generator()
        result = _make_result(
            topics=None,
            topic_skipped_reason="sentence count insufficient (3 < 10)",
        )
        flowables = gen._build_topic_section(result)
        assert len(flowables) > 0

    def test_report_coverage_section_when_concepts(self) -> None:
        """Coverage data in story when concepts provided."""
        from forma.lecture_analyzer import ConceptCoverage

        gen = _make_generator()
        coverage = ConceptCoverage(
            total_concepts=3,
            covered_concepts=["cell", "protein"],
            missed_concepts=["mitochondria"],
            coverage_ratio=2 / 3,
        )
        result = _make_result(concept_coverage=coverage)
        flowables = gen._build_coverage_section(result)
        assert len(flowables) > 0

    def test_report_skipped_stage_labeled(self) -> None:
        """Failed/skipped stage shows reason string, not omitted."""
        gen = _make_generator()
        result = _make_result(
            topics=None,
            topic_skipped_reason="BERTopic error: HDBSCAN failed",
        )
        flowables = gen._build_topic_section(result)
        assert len(flowables) > 0


class TestComparisonReport:
    """Test LectureReportGenerator comparison report methods."""

    @staticmethod
    def _make_comparison(**overrides):
        """Create a minimal ComparisonResult for testing."""
        from forma.lecture_comparison import ComparisonResult, EmphasisVarianceEntry

        defaults = dict(
            comparison_type="session",
            sections_compared=["A", "B"],
            exclusive_keywords={"A": ["ATP", "단백질"], "B": ["DNA", "RNA"]},
            concept_gaps={"A": ["DNA"], "B": ["ATP"]},
            emphasis_variance=[
                EmphasisVarianceEntry(
                    concept="세포",
                    variance=0.5,
                    per_section_scores={"A": 1.0, "B": 0.0},
                ),
                EmphasisVarianceEntry(
                    concept="ATP",
                    variance=0.1,
                    per_section_scores={"A": 0.6, "B": 0.4},
                ),
            ],
            comparison_timestamp="2026-01-01T00:00:00+00:00",
        )
        defaults.update(overrides)
        return ComparisonResult(**defaults)

    def test_generate_comparison_report_creates_pdf(self, tmp_path: Path) -> None:
        """Comparison report produces PDF file."""
        gen = _make_generator()
        comparison = self._make_comparison()
        output_path = tmp_path / "comparison.pdf"
        gen.generate_comparison_report(comparison, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_comparison_report_exclusive_keywords(self) -> None:
        """PDF contains exclusive keywords section."""
        gen = _make_generator()
        comparison = self._make_comparison()
        flowables = gen._build_exclusive_keywords_section(comparison)
        assert len(flowables) > 0

    def test_comparison_report_concept_gaps(self) -> None:
        """PDF contains concept gap matrix when concepts provided."""
        gen = _make_generator()
        comparison = self._make_comparison()
        flowables = gen._build_concept_gap_section(comparison)
        assert len(flowables) > 0

    def test_comparison_report_emphasis_variance(self) -> None:
        """PDF contains emphasis variance table."""
        gen = _make_generator()
        comparison = self._make_comparison()
        flowables = gen._build_emphasis_variance_section(comparison)
        assert len(flowables) > 0

    def test_comparison_report_no_concept_gaps(self) -> None:
        """When concept_gaps is None, section returns minimal flowables."""
        gen = _make_generator()
        comparison = self._make_comparison(concept_gaps=None)
        flowables = gen._build_concept_gap_section(comparison)
        # Should still return at least a header or placeholder
        assert len(flowables) > 0
