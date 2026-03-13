"""Tests for warning_report.py — WarningPDFReportGenerator.

T040 [US3]: PDF generation with warning cards, cover page, dashboard, per-student cards.
"""

from __future__ import annotations

import os


from forma.warning_report_data import RiskType, WarningCard


def _make_warning_card(
    student_id: str = "S001",
    risk_types: list[RiskType] | None = None,
    detection_methods: list[str] | None = None,
    deficit_concepts: list[str] | None = None,
    interventions: list[str] | None = None,
    drop_probability: float | None = 0.7,
    risk_severity: float = 0.7,
) -> WarningCard:
    """Build a WarningCard for testing."""
    return WarningCard(
        student_id=student_id,
        risk_types=risk_types or [RiskType.SCORE_DECLINE],
        detection_methods=detection_methods or ["rule_based"],
        deficit_concepts=deficit_concepts or ["세포", "조직"],
        misconception_patterns=["CAUSAL_REVERSAL"],
        interventions=interventions or ["최근 성적 하락 추세에 대한 개별 면담 권장"],
        drop_probability=drop_probability,
        risk_severity=risk_severity,
    )


class TestWarningPDFReportGenerator:
    """Tests for WarningPDFReportGenerator."""

    def test_generate_creates_pdf(self, tmp_path):
        """generate() creates a PDF file."""
        from forma.warning_report import WarningPDFReportGenerator

        cards = [_make_warning_card()]
        gen = WarningPDFReportGenerator()
        output = str(tmp_path / "warning.pdf")
        gen.generate_pdf(cards, output, class_name="1A")
        assert os.path.isfile(output)
        assert os.path.getsize(output) > 0

    def test_generate_with_multiple_cards(self, tmp_path):
        """generate() handles multiple warning cards."""
        from forma.warning_report import WarningPDFReportGenerator

        cards = [
            _make_warning_card("S001", risk_severity=0.9),
            _make_warning_card("S002", risk_severity=0.7),
            _make_warning_card("S003", risk_severity=0.5),
        ]
        gen = WarningPDFReportGenerator()
        output = str(tmp_path / "warning.pdf")
        gen.generate_pdf(cards, output, class_name="1A")
        assert os.path.isfile(output)

    def test_generate_empty_cards(self, tmp_path):
        """generate() with empty cards produces a summary-only PDF."""
        from forma.warning_report import WarningPDFReportGenerator

        gen = WarningPDFReportGenerator()
        output = str(tmp_path / "warning.pdf")
        gen.generate_pdf([], output, class_name="1A")
        assert os.path.isfile(output)

    def test_generate_all_risk_types(self, tmp_path):
        """generate() handles cards with all risk types."""
        from forma.warning_report import WarningPDFReportGenerator

        cards = [_make_warning_card(
            risk_types=list(RiskType),
            interventions=[
                "최근 성적 하락 추세에 대한 개별 면담 권장",
                "기초 개념 보충 학습 프로그램 안내",
            ],
        )]
        gen = WarningPDFReportGenerator()
        output = str(tmp_path / "warning.pdf")
        gen.generate_pdf(cards, output, class_name="1A")
        assert os.path.isfile(output)

    def test_generate_no_drop_probability(self, tmp_path):
        """generate() handles cards with no model prediction."""
        from forma.warning_report import WarningPDFReportGenerator

        cards = [_make_warning_card(drop_probability=None)]
        gen = WarningPDFReportGenerator()
        output = str(tmp_path / "warning.pdf")
        gen.generate_pdf(cards, output, class_name="1A")
        assert os.path.isfile(output)

    def test_pdf_is_valid(self, tmp_path):
        """Generated file starts with PDF header."""
        from forma.warning_report import WarningPDFReportGenerator

        cards = [_make_warning_card()]
        gen = WarningPDFReportGenerator()
        output = str(tmp_path / "warning.pdf")
        gen.generate_pdf(cards, output, class_name="1A")
        with open(output, "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_korean_text_safety(self, tmp_path):
        """Korean text in deficit concepts and interventions renders safely."""
        from forma.warning_report import WarningPDFReportGenerator

        cards = [_make_warning_card(
            deficit_concepts=["세포막 투과성", "삼투압 조절 기전"],
            interventions=["결손 개념 목록 기반 맞춤 보충 자료 제공"],
        )]
        gen = WarningPDFReportGenerator()
        output = str(tmp_path / "warning.pdf")
        gen.generate_pdf(cards, output, class_name="1A반")
        assert os.path.isfile(output)

    def test_custom_font_and_dpi(self, tmp_path):
        """Custom font_path and dpi are accepted."""
        from forma.warning_report import WarningPDFReportGenerator

        gen = WarningPDFReportGenerator(dpi=100)
        cards = [_make_warning_card()]
        output = str(tmp_path / "warning.pdf")
        gen.generate_pdf(cards, output, class_name="1A")
        assert os.path.isfile(output)
