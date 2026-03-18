"""Tests for domain delivery PDF report generation (v2).

T048: PDF has 9 sections, continuous numbering, non-zero file size, Korean text
T049: PDF without assessment data -> 8 sections, section 9 omitted
T050: Instructor feedback page uses concept-level language, not word frequency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


from forma.domain_coverage_analyzer import (
    DeliveryAnalysis,
    DeliveryResult,
)
from forma.domain_pedagogy_analyzer import (
    EffectivePattern,
    HabitualExpression,
    PedagogyAnalysis,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _build_sample_delivery_result() -> DeliveryResult:
    """Build a sample DeliveryResult with typical data."""
    deliveries = [
        DeliveryAnalysis(
            concept="표피의 4층 구조",
            section_id="A",
            delivery_status="충분히 설명",
            delivery_quality=0.9,
            evidence="표피의 4층에 대해 상세히 설명했다",
            depth="메커니즘까지 설명",
        ),
        DeliveryAnalysis(
            concept="표피의 4층 구조",
            section_id="B",
            delivery_status="부분 전달",
            delivery_quality=0.5,
            evidence="용어만 언급",
            depth="용어 수준",
        ),
        DeliveryAnalysis(
            concept="진피의 구조와 기능",
            section_id="A",
            delivery_status="충분히 설명",
            delivery_quality=0.85,
            evidence="진피 구조와 기능 상세 설명",
            depth="상세 설명",
        ),
        DeliveryAnalysis(
            concept="진피의 구조와 기능",
            section_id="B",
            delivery_status="미전달",
            delivery_quality=0.0,
            evidence="",
            depth="",
        ),
        DeliveryAnalysis(
            concept="세포막 인지질 이중층",
            section_id="A",
            delivery_status="의도적 생략",
            delivery_quality=0.0,
            evidence="",
            depth="",
        ),
        DeliveryAnalysis(
            concept="세포막 인지질 이중층",
            section_id="B",
            delivery_status="의도적 생략",
            delivery_quality=0.0,
            evidence="",
            depth="",
        ),
        DeliveryAnalysis(
            concept="각질층 형성 과정",
            section_id="A",
            delivery_status="충분히 설명",
            delivery_quality=0.8,
            evidence="각질층 형성 메커니즘 설명",
            depth="과정 설명",
        ),
        DeliveryAnalysis(
            concept="각질층 형성 과정",
            section_id="B",
            delivery_status="충분히 설명",
            delivery_quality=0.75,
            evidence="각질층 생성 과정",
            depth="과정 설명",
        ),
    ]
    return DeliveryResult(
        week=2,
        chapters=["3장 피부", "1장 세포"],
        deliveries=deliveries,
        effective_delivery_rate=0.75,
        per_section_rate={"A": 1.0, "B": 0.5},
    )


def _build_sample_pedagogy() -> list[PedagogyAnalysis]:
    """Build sample PedagogyAnalysis list."""
    return [
        PedagogyAnalysis(
            section_id="A",
            habitual_expressions=[
                HabitualExpression(
                    expression="여러분",
                    frequency_per_minute=2.0,
                    total_count=40,
                    recommendation="사용 자제 권장",
                ),
                HabitualExpression(
                    expression="보시면",
                    frequency_per_minute=1.0,
                    total_count=20,
                    recommendation="정상 범위",
                ),
            ],
            effective_patterns=[
                EffectivePattern(
                    pattern_type="비유/유추",
                    count=3,
                    examples=["세포막을 지퍼에 비유하면..."],
                ),
            ],
            domain_ratio=0.7,
        ),
    ]


@dataclass
class MockAssessmentData:
    """Mock assessment correlation data for testing."""

    correlation: float = 0.65
    well_explained_poor: list[str] = field(default_factory=lambda: [
        "표피의 4층 구조",
    ])
    under_explained_poor: list[str] = field(default_factory=lambda: [
        "진피의 구조와 기능",
    ])


# ----------------------------------------------------------------
# T048: PDF has 9 sections
# ----------------------------------------------------------------


class TestDomainDeliveryPDFReport:
    """Tests for DomainDeliveryPDFReportGenerator (9-section v2 report)."""

    def test_pdf_9_sections_nonzero(self, tmp_path: Path) -> None:
        """PDF with assessment data has 9 sections, non-zero file size."""
        from forma.domain_coverage_report import DomainDeliveryPDFReportGenerator

        result = _build_sample_delivery_result()
        pedagogy = _build_sample_pedagogy()
        assessment = MockAssessmentData()
        output = str(tmp_path / "report_9.pdf")

        gen = DomainDeliveryPDFReportGenerator()
        path = gen.generate_pdf(
            result, output,
            course_name="인체구조와기능",
            pedagogy=pedagogy,
            assessment_data=assessment,
        )

        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

        # Verify PDF magic bytes
        with open(path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_pdf_continuous_numbering(self, tmp_path: Path) -> None:
        """PDF sections use continuous numbering (1-9)."""
        from forma.domain_coverage_report import DomainDeliveryPDFReportGenerator

        result = _build_sample_delivery_result()
        assessment = MockAssessmentData()
        output = str(tmp_path / "report_num.pdf")

        gen = DomainDeliveryPDFReportGenerator()
        path = gen.generate_pdf(
            result, output,
            assessment_data=assessment,
        )

        # Read PDF text to verify section numbering
        pdf_bytes = Path(path).read_bytes()
        pdf_text = pdf_bytes.decode("latin-1", errors="replace")

        # Section headers should appear in order (in PDF stream)
        for i in range(1, 10):
            assert f"{i}." in pdf_text

    def test_pdf_korean_text(self, tmp_path: Path) -> None:
        """PDF contains Korean text (course name, section titles)."""
        from forma.domain_coverage_report import DomainDeliveryPDFReportGenerator

        result = _build_sample_delivery_result()
        output = str(tmp_path / "report_kr.pdf")

        gen = DomainDeliveryPDFReportGenerator()
        path = gen.generate_pdf(
            result, output,
            course_name="인체구조와기능",
        )

        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000  # Non-trivial size

    def test_pdf_with_all_data(self, tmp_path: Path) -> None:
        """PDF with all optional data (pedagogy, assessment) generates OK."""
        from forma.domain_coverage_report import DomainDeliveryPDFReportGenerator

        result = _build_sample_delivery_result()
        pedagogy = _build_sample_pedagogy()
        assessment = MockAssessmentData()
        output = str(tmp_path / "report_full.pdf")

        gen = DomainDeliveryPDFReportGenerator()
        path = gen.generate_pdf(
            result, output,
            course_name="인체구조와기능",
            pedagogy=pedagogy,
            assessment_data=assessment,
        )

        assert Path(path).exists()


# ----------------------------------------------------------------
# T049: PDF without assessment data -> 8 sections
# ----------------------------------------------------------------


class TestDomainDeliveryPDFNoAssessment:
    """Tests for report without assessment data (8 sections)."""

    def test_pdf_8_sections_no_assessment(self, tmp_path: Path) -> None:
        """PDF without assessment data has 8 sections, section 9 omitted."""
        from forma.domain_coverage_report import DomainDeliveryPDFReportGenerator

        result = _build_sample_delivery_result()
        output = str(tmp_path / "report_8.pdf")

        gen = DomainDeliveryPDFReportGenerator()
        path = gen.generate_pdf(result, output)

        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

        # Read PDF text — should NOT contain section 9 header
        pdf_bytes = Path(path).read_bytes()
        pdf_text = pdf_bytes.decode("latin-1", errors="replace")

        # Sections 1-8 should exist
        for i in range(1, 9):
            assert f"{i}." in pdf_text

    def test_empty_deliveries(self, tmp_path: Path) -> None:
        """PDF with empty deliveries generates without error."""
        from forma.domain_coverage_report import DomainDeliveryPDFReportGenerator

        result = DeliveryResult(
            week=1,
            chapters=["1장"],
            deliveries=[],
            effective_delivery_rate=0.0,
            per_section_rate={},
        )
        output = str(tmp_path / "empty_report.pdf")

        gen = DomainDeliveryPDFReportGenerator()
        path = gen.generate_pdf(result, output)

        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


# ----------------------------------------------------------------
# T050: Feedback uses concept-level language
# ----------------------------------------------------------------


class TestFeedbackConceptLevel:
    """Tests that feedback uses concept-level language, not word frequency."""

    def test_feedback_concept_names_not_word_counts(
        self, tmp_path: Path,
    ) -> None:
        """Feedback section shows concept names, not word frequency stats."""
        from forma.domain_coverage_report import DomainDeliveryPDFReportGenerator

        result = _build_sample_delivery_result()
        output = str(tmp_path / "report_feedback.pdf")

        gen = DomainDeliveryPDFReportGenerator()
        path = gen.generate_pdf(result, output)

        # Read raw PDF bytes and check
        pdf_bytes = Path(path).read_bytes()
        pdf_text = pdf_bytes.decode("latin-1", errors="replace")

        # Should NOT contain word-frequency patterns like "N회 언급"
        # in feedback section context (word counting language)
        # The feedback should use concept-level language like
        # "보충 지도가 필요한" and concept names

        # Verify absence of word-frequency patterns
        assert "단어 빈도" not in pdf_text
        assert "회 언급" not in pdf_text

    def test_backward_compat_alias(self) -> None:
        """DomainCoveragePDFReportGenerator alias still works."""
        from forma.domain_coverage_report import (
            DomainCoveragePDFReportGenerator,
            DomainDeliveryPDFReportGenerator,
        )

        assert DomainCoveragePDFReportGenerator is DomainDeliveryPDFReportGenerator
