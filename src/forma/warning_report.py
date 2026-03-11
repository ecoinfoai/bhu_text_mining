"""Early warning PDF report generator using ReportLab Platypus.

Builds A4 PDF reports containing:
    - Cover page with title, confidentiality notice, and summary stats
    - Class dashboard with risk type distribution and deficit concept charts
    - Per-student warning cards (risk types, deficit concepts, interventions)
    - No-warning summary page when no at-risk students

No LLM API calls are made during PDF generation (Constitution VI).
"""

from __future__ import annotations

import io
import logging
import os
from collections import Counter

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from forma.font_utils import esc as _esc, find_korean_font, register_korean_fonts
from forma.warning_report_data import RiskType, WarningCard

logger = logging.getLogger(__name__)


# Risk type display colors
_RISK_TYPE_COLORS = {
    RiskType.SCORE_DECLINE: "#E53935",
    RiskType.PERSISTENT_LOW: "#FB8C00",
    RiskType.CONCEPT_DEFICIT: "#7B1FA2",
    RiskType.PARTICIPATION_DECLINE: "#1E88E5",
}


class WarningPDFReportGenerator:
    """Generate early warning PDF reports using ReportLab Platypus.

    Args:
        font_path: Path to Korean .ttf font. Auto-detected if None.
        dpi: Resolution for chart images (default 150).
    """

    def __init__(self, font_path: str | None = None, dpi: int = 150) -> None:
        if font_path is None:
            font_path = find_korean_font()
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Korean font not found: {font_path}")

        self._font_path = font_path
        self._dpi = dpi

        register_korean_fonts(font_path)

        # Define paragraph styles
        self._styles = getSampleStyleSheet()
        self._styles.add(ParagraphStyle(
            "WarnTitle",
            parent=self._styles["Title"],
            fontName="NanumGothicBold",
            fontSize=18,
            spaceAfter=12,
        ))
        self._styles.add(ParagraphStyle(
            "WarnSection",
            parent=self._styles["Heading2"],
            fontName="NanumGothicBold",
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
        ))
        self._styles.add(ParagraphStyle(
            "WarnBody",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=10,
            leading=14,
            spaceAfter=4,
        ))
        self._styles.add(ParagraphStyle(
            "WarnTableHeader",
            parent=self._styles["Normal"],
            fontName="NanumGothicBold",
            fontSize=8,
            textColor=HexColor("#FFFFFF"),
            alignment=1,
        ))
        self._styles.add(ParagraphStyle(
            "WarnTableData",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=8,
            alignment=1,
        ))
        self._styles.add(ParagraphStyle(
            "WarnNotice",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=9,
            textColor=HexColor("#C62828"),
            leading=13,
            spaceAfter=8,
        ))

    def generate(
        self,
        warning_cards: list[WarningCard],
        output_path: str,
        *,
        class_name: str = "",
    ) -> str:
        """Generate the early warning PDF report.

        Args:
            warning_cards: List of WarningCard, sorted by risk_severity desc.
            output_path: Full path for the output PDF file.
            class_name: Class name for display on cover page.

        Returns:
            Absolute path to generated PDF file.
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        story: list = []

        # Cover page
        story.extend(self._build_cover_page(warning_cards, class_name))
        story.append(PageBreak())

        if warning_cards:
            # Dashboard page
            story.extend(self._build_dashboard(warning_cards))
            story.append(PageBreak())

            # Per-student warning cards
            for i, card in enumerate(warning_cards):
                story.extend(self._build_student_card(card))
                if i < len(warning_cards) - 1:
                    story.append(PageBreak())
        else:
            # No-warning summary
            story.extend(self._build_no_warning_page())

        # Build PDF
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            topMargin=15 * mm,
            bottomMargin=15 * mm,
        )
        doc.build(story)
        logger.info("Early warning report generated: %s", output_path)
        return os.path.abspath(output_path)

    def _build_cover_page(
        self,
        warning_cards: list[WarningCard],
        class_name: str,
    ) -> list:
        """Build cover page with title, confidentiality notice, and summary."""
        story: list = []
        story.append(Spacer(1, 40 * mm))
        story.append(Paragraph(
            _esc("조기 경고 보고서"),
            self._styles["WarnTitle"],
        ))
        if class_name:
            story.append(Paragraph(
                _esc(f"분반: {class_name}"),
                self._styles["WarnBody"],
            ))
        story.append(Spacer(1, 10 * mm))

        # Confidentiality notice
        story.append(Paragraph(
            _esc("본 보고서는 교수자 전용이며, 학생 개인정보 보호를 위해 "
                 "외부 유출을 금합니다."),
            self._styles["WarnNotice"],
        ))
        # FR-022: General guidance disclaimer
        story.append(Paragraph(
            _esc("본 보고서의 모든 개입 권고는 일반적 지침이며, "
                 "교수자의 전문적 판단을 대체하지 않습니다."),
            self._styles["WarnNotice"],
        ))
        story.append(Spacer(1, 10 * mm))

        # Summary stats
        n_total = len(warning_cards)
        story.append(Paragraph(
            _esc(f"위험군 학생 수: {n_total}명"),
            self._styles["WarnBody"],
        ))

        if warning_cards:
            # Count by risk type
            risk_counts: Counter = Counter()
            for card in warning_cards:
                for rt in card.risk_types:
                    risk_counts[rt] += 1

            for rt in RiskType:
                count = risk_counts.get(rt, 0)
                if count > 0:
                    story.append(Paragraph(
                        _esc(f"  - {rt.label}: {count}명"),
                        self._styles["WarnBody"],
                    ))

            # Detection method summary
            rule_count = sum(1 for c in warning_cards if "rule_based" in c.detection_methods)
            model_count = sum(1 for c in warning_cards if "model_predicted" in c.detection_methods)
            story.append(Spacer(1, 5 * mm))
            story.append(Paragraph(
                _esc(f"규칙 기반 감지: {rule_count}명 / 모델 예측 감지: {model_count}명"),
                self._styles["WarnBody"],
            ))

        return story

    def _build_dashboard(self, warning_cards: list[WarningCard]) -> list:
        """Build dashboard page with charts and risk type summary table."""
        story: list = []
        story.append(Paragraph(
            _esc("위험 유형 대시보드"),
            self._styles["WarnSection"],
        ))

        # Risk type distribution chart
        try:
            from forma.warning_report_charts import build_risk_type_distribution_chart

            risk_counts: Counter = Counter()
            for card in warning_cards:
                for rt in card.risk_types:
                    risk_counts[rt] += 1

            chart_buf = build_risk_type_distribution_chart(
                dict(risk_counts),
                font_path=self._font_path,
                dpi=self._dpi,
            )
            story.append(Image(chart_buf, width=160 * mm, height=80 * mm))
            story.append(Spacer(1, 5 * mm))
        except Exception as exc:
            logger.warning("Risk type chart generation failed: %s", exc)

        # Deficit concepts chart
        try:
            from forma.warning_report_charts import build_deficit_concepts_chart

            concept_counts: Counter = Counter()
            for card in warning_cards:
                for concept in card.deficit_concepts:
                    concept_counts[concept] += 1

            if concept_counts:
                chart_buf = build_deficit_concepts_chart(
                    dict(concept_counts),
                    font_path=self._font_path,
                    dpi=self._dpi,
                )
                story.append(Image(chart_buf, width=160 * mm, height=80 * mm))
                story.append(Spacer(1, 5 * mm))
        except Exception as exc:
            logger.warning("Deficit concepts chart generation failed: %s", exc)

        # Summary table
        story.append(Paragraph(
            _esc("위험 유형 요약표"),
            self._styles["WarnSection"],
        ))

        header = [
            Paragraph(_esc("위험 유형"), self._styles["WarnTableHeader"]),
            Paragraph(_esc("학생 수"), self._styles["WarnTableHeader"]),
            Paragraph(_esc("대표 중재 방안"), self._styles["WarnTableHeader"]),
        ]

        from forma.warning_report_data import INTERVENTION_MAP

        table_data = [header]
        risk_counts_for_table: Counter = Counter()
        for card in warning_cards:
            for rt in card.risk_types:
                risk_counts_for_table[rt] += 1

        for rt in RiskType:
            count = risk_counts_for_table.get(rt, 0)
            interventions = INTERVENTION_MAP.get(rt.value, [])
            first_intervention = interventions[0] if interventions else "-"
            table_data.append([
                Paragraph(_esc(rt.label), self._styles["WarnTableData"]),
                Paragraph(_esc(str(count)), self._styles["WarnTableData"]),
                Paragraph(_esc(first_intervention), self._styles["WarnTableData"]),
            ])

        table = Table(table_data, colWidths=[40 * mm, 25 * mm, 105 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#37474F")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(table)

        return story

    def _build_student_card(self, card: WarningCard) -> list:
        """Build a per-student warning card section."""
        story: list = []
        story.append(Paragraph(
            _esc(f"학생 경고 카드: {card.student_id}"),
            self._styles["WarnSection"],
        ))

        # Risk types
        risk_labels = ", ".join(rt.label for rt in card.risk_types)
        story.append(Paragraph(
            _esc(f"위험 유형: {risk_labels}"),
            self._styles["WarnBody"],
        ))

        # Detection methods
        methods = ", ".join(card.detection_methods)
        story.append(Paragraph(
            _esc(f"감지 방법: {methods}"),
            self._styles["WarnBody"],
        ))

        # Drop probability
        if card.drop_probability is not None:
            story.append(Paragraph(
                _esc(f"이탈 확률: {card.drop_probability:.1%}"),
                self._styles["WarnBody"],
            ))

        # Risk severity
        story.append(Paragraph(
            _esc(f"위험도: {card.risk_severity:.2f}"),
            self._styles["WarnBody"],
        ))
        story.append(Spacer(1, 3 * mm))

        # Deficit concepts
        if card.deficit_concepts:
            story.append(Paragraph(
                _esc("결손 개념:"),
                self._styles["WarnBody"],
            ))
            for concept in card.deficit_concepts:
                story.append(Paragraph(
                    _esc(f"  - {concept}"),
                    self._styles["WarnBody"],
                ))
            story.append(Spacer(1, 3 * mm))

        # Misconception patterns
        if card.misconception_patterns:
            story.append(Paragraph(
                _esc("오개념 패턴:"),
                self._styles["WarnBody"],
            ))
            for pattern in card.misconception_patterns:
                story.append(Paragraph(
                    _esc(f"  - {pattern}"),
                    self._styles["WarnBody"],
                ))
            story.append(Spacer(1, 3 * mm))

        # Interventions
        if card.interventions:
            story.append(Paragraph(
                _esc("권장 중재 방안:"),
                self._styles["WarnBody"],
            ))
            for idx, intervention in enumerate(card.interventions, 1):
                story.append(Paragraph(
                    _esc(f"  {idx}. {intervention}"),
                    self._styles["WarnBody"],
                ))

        return story

    def _build_no_warning_page(self) -> list:
        """Build a summary page for when no students are at risk."""
        story: list = []
        story.append(Spacer(1, 40 * mm))
        story.append(Paragraph(
            _esc("위험군 학생이 없습니다"),
            self._styles["WarnSection"],
        ))
        story.append(Spacer(1, 10 * mm))
        story.append(Paragraph(
            _esc("현재 분석 결과, 조기 경고 대상 학생이 감지되지 않았습니다. "
                 "모든 학생이 정상적인 학업 진행 상태입니다."),
            self._styles["WarnBody"],
        ))
        return story
