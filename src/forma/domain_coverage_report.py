"""PDF report generation for domain coverage analysis.

Generates a multi-page PDF report containing coverage summary,
4-state classification table, gap/skipped/extra lists, emphasis
bias scatter plot, section variance heatmap, and instructor
feedback summary.
"""

from __future__ import annotations

import io
import logging
import os

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from forma.font_utils import esc as _esc, find_korean_font, register_korean_fonts
from forma.report_utils import minimal_png_bytes

logger = logging.getLogger(__name__)

__all__ = ["DomainCoveragePDFReportGenerator"]

_FALLBACK_PNG: bytes = minimal_png_bytes()

_STATE_COLORS = {
    "COVERED": "#4CAF50",
    "GAP": "#F44336",
    "SKIPPED": "#9E9E9E",
    "EXTRA": "#2196F3",
}


class DomainCoveragePDFReportGenerator:
    """Generate domain coverage PDF reports using ReportLab Platypus.

    Args:
        font_path: Path to Korean .ttf font. Auto-detected if None.
        dpi: Resolution for chart images (default 150).
    """

    def __init__(
        self,
        font_path: str | None = None,
        dpi: int = 150,
    ) -> None:
        if font_path is None:
            font_path = find_korean_font()
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Korean font not found: {font_path}")

        self._font_path = font_path
        self._dpi = dpi

        register_korean_fonts(font_path)

        self._styles = getSampleStyleSheet()
        self._styles.add(ParagraphStyle(
            "DcTitle",
            parent=self._styles["Title"],
            fontName="NanumGothicBold",
            fontSize=20,
            spaceAfter=12,
        ))
        self._styles.add(ParagraphStyle(
            "DcSection",
            parent=self._styles["Heading2"],
            fontName="NanumGothicBold",
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
        ))
        self._styles.add(ParagraphStyle(
            "DcBody",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=10,
            leading=14,
            spaceAfter=4,
        ))
        self._styles.add(ParagraphStyle(
            "DcSmall",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=8,
            leading=11,
            spaceAfter=2,
            textColor=HexColor("#666666"),
        ))
        self._styles.add(ParagraphStyle(
            "DcTableCell",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=8,
            leading=10,
        ))

    def generate_pdf(
        self,
        result: object,
        output_path: str,
        course_name: str = "",
    ) -> str:
        """Generate the domain coverage report PDF.

        Args:
            result: CoverageResult from domain_coverage_analyzer.
            output_path: Output path for the PDF file.
            course_name: Optional course name for header.

        Returns:
            Absolute path to generated PDF file.
        """
        story: list = []
        story.extend(self._build_cover_page(result, course_name))
        story.extend(self._build_summary_section(result))
        story.extend(self._build_classification_table(result))
        story.extend(self._build_gap_section(result))
        story.extend(self._build_skipped_section(result))
        story.extend(self._build_extra_section(result))
        story.extend(self._build_bias_scatter_section(result))
        story.extend(self._build_variance_heatmap_section(result))
        story.extend(self._build_feedback_section(result))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )
        doc.build(story)
        return os.path.abspath(output_path)

    def _build_cover_page(self, result: object, course_name: str) -> list:
        """Build cover page."""
        story = []
        story.append(Spacer(1, 40 * mm))
        story.append(Paragraph(
            _esc("교과서-강의 도메인 커버리지 분석 보고서"),
            self._styles["DcTitle"],
        ))
        story.append(Spacer(1, 10 * mm))

        if course_name:
            story.append(Paragraph(
                f"교과목: {_esc(course_name)}",
                self._styles["DcBody"],
            ))

        if result.week:
            story.append(Paragraph(
                f"주차: {result.week}주차",
                self._styles["DcBody"],
            ))

        if result.chapters:
            story.append(Paragraph(
                f"분석 대상: {_esc(', '.join(result.chapters))}",
                self._styles["DcBody"],
            ))

        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph(
            f"실효 커버리지율: {result.effective_coverage_rate:.1%}",
            self._styles["DcBody"],
        ))

        story.append(PageBreak())
        return story

    def _build_summary_section(self, result: object) -> list:
        """Build summary statistics section."""
        story = []
        story.append(Paragraph("1. 커버리지 요약", self._styles["DcSection"]))
        story.append(Spacer(1, 3 * mm))

        summary_lines = [
            f"전체 교과서 개념: {result.total_textbook_concepts}개",
            f"수업 범위 내: {result.in_scope_count}개",
            f"다룸 (COVERED): {result.covered_count}개",
            f"누락 위험 (GAP): {result.gap_count}개",
            f"의도적 생략 (SKIPPED): {result.skipped_count}개",
            f"추가 설명 (EXTRA): {result.extra_count}개",
            f"실효 커버리지율: {result.effective_coverage_rate:.1%}",
        ]
        for line in summary_lines:
            story.append(Paragraph(_esc(line), self._styles["DcBody"]))

        # Coverage bar chart
        try:
            from forma.domain_coverage_charts import build_coverage_bar_chart
            chart_buf = build_coverage_bar_chart(
                result, font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Spacer(1, 5 * mm))
            story.append(Image(chart_buf, width=160 * mm, height=100 * mm))
        except Exception as exc:
            logger.warning("커버리지 바 차트 생성 실패: %s", exc)
            story.append(Image(
                io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm,
            ))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_classification_table(self, result: object) -> list:
        """Build 4-state classification table."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            "2. 4-상태 분류표", self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        # Table header
        headers = ["개념", "챕터", "상태", "평균 강조도"]

        # Collect sections for per-section columns
        sections: set[str] = set()
        for cc in result.classified_concepts:
            if cc.emphasis is not None:
                sections.update(cc.emphasis.section_scores.keys())
        sections_sorted = sorted(sections)
        headers.extend(f"{s}반" for s in sections_sorted)

        cell_style = self._styles["DcTableCell"]
        header_row = [Paragraph(f"<b>{_esc(h)}</b>", cell_style) for h in headers]

        data = [header_row]
        for cc in result.classified_concepts:
            state_color = _STATE_COLORS.get(cc.state.name, "#000000")
            state_label = f'<font color="{state_color}"><b>{_esc(cc.state.value)}</b></font>'

            mean_emph = ""
            if cc.emphasis is not None:
                mean_emph = f"{cc.emphasis.mean_score:.3f}"

            row = [
                Paragraph(_esc(cc.concept.name_ko), cell_style),
                Paragraph(_esc(cc.concept.chapter), cell_style),
                Paragraph(state_label, cell_style),
                Paragraph(mean_emph, cell_style),
            ]

            for section in sections_sorted:
                if cc.emphasis is not None:
                    score = cc.emphasis.section_scores.get(section, 0.0)
                    row.append(Paragraph(f"{score:.3f}", cell_style))
                else:
                    row.append(Paragraph("—", cell_style))

            data.append(row)

        if len(data) > 1:
            col_widths = [50 * mm, 35 * mm, 25 * mm, 22 * mm]
            remaining = (A4[0] - 30 * mm) - sum(col_widths)
            if sections_sorted:
                per_section = remaining / len(sections_sorted)
                col_widths.extend([per_section] * len(sections_sorted))

            table = Table(data, colWidths=col_widths, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E0E0E0")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(table)
        else:
            story.append(Paragraph(
                "분류된 개념 없음", self._styles["DcBody"],
            ))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_gap_section(self, result: object) -> list:
        """Build gap concepts list (red highlight)."""
        from forma.domain_coverage_analyzer import ConceptState

        story = []
        gaps = [
            cc for cc in result.classified_concepts
            if cc.state == ConceptState.GAP
        ]

        if not gaps:
            return story

        story.append(Paragraph(
            "3. 누락 위험 개념", self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        for cc in gaps:
            text = f'<font color="#F44336"><b>{_esc(cc.concept.name_ko)}</b></font>'
            text += f" ({_esc(cc.concept.chapter)})"
            if cc.concept.name_en:
                text += f" — {_esc(cc.concept.name_en)}"
            story.append(Paragraph(text, self._styles["DcBody"]))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_skipped_section(self, result: object) -> list:
        """Build skipped concepts list (gray)."""
        from forma.domain_coverage_analyzer import ConceptState

        story = []
        skipped = [
            cc for cc in result.classified_concepts
            if cc.state == ConceptState.SKIPPED
        ]

        if not skipped:
            return story

        story.append(Paragraph(
            "4. 의도적 생략 개념", self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        for cc in skipped:
            text = f'<font color="#9E9E9E">{_esc(cc.concept.name_ko)}</font>'
            text += f" ({_esc(cc.concept.chapter)})"
            story.append(Paragraph(text, self._styles["DcBody"]))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_extra_section(self, result: object) -> list:
        """Build extra concepts list (blue)."""
        story = []

        if not result.extra_concepts:
            return story

        story.append(Paragraph(
            "5. 추가 설명 개념", self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        for extra in result.extra_concepts:
            total = sum(extra.section_mentions.values())
            text = f'<font color="#2196F3"><b>{_esc(extra.name)}</b></font>'
            text += f" (총 {total}회 언급)"
            story.append(Paragraph(text, self._styles["DcBody"]))
            if extra.example_sentence:
                story.append(Paragraph(
                    f"예: \"{_esc(extra.example_sentence[:100])}\"",
                    self._styles["DcSmall"],
                ))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_bias_scatter_section(self, result: object) -> list:
        """Build emphasis bias scatter plot section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            "6. 강조 편향 분석", self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        try:
            from forma.domain_coverage_charts import build_emphasis_bias_scatter
            chart_buf = build_emphasis_bias_scatter(
                result, font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Image(chart_buf, width=160 * mm, height=110 * mm))
        except Exception as exc:
            logger.warning("강조 편향 차트 생성 실패: %s", exc)
            story.append(Image(
                io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm,
            ))

        if result.emphasis_bias_correlation is not None:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(
                f"Spearman ρ = {result.emphasis_bias_correlation:.3f}",
                self._styles["DcBody"],
            ))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_variance_heatmap_section(self, result: object) -> list:
        """Build section variance heatmap section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            "7. 분반 간 강조도 편차", self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        try:
            from forma.domain_coverage_charts import build_section_variance_heatmap
            chart_buf = build_section_variance_heatmap(
                result, font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Image(chart_buf, width=160 * mm, height=120 * mm))
        except Exception as exc:
            logger.warning("편차 히트맵 생성 실패: %s", exc)
            story.append(Image(
                io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm,
            ))

        if result.section_variance_top10:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(
                "편차 상위 10개 개념:",
                self._styles["DcBody"],
            ))
            for name, std in result.section_variance_top10:
                story.append(Paragraph(
                    f"  - {_esc(name)}: σ = {std:.3f}",
                    self._styles["DcSmall"],
                ))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_feedback_section(self, result: object) -> list:
        """Build instructor feedback summary page."""
        from forma.domain_coverage_analyzer import ConceptState

        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            "8. 교수자 피드백 요약", self._styles["DcSection"],
        ))
        story.append(Spacer(1, 5 * mm))

        # Gap concepts requiring supplementary instruction
        gaps = [
            cc for cc in result.classified_concepts
            if cc.state == ConceptState.GAP
        ]
        if gaps:
            story.append(Paragraph(
                "<b>보충 지도가 필요한 개념:</b>",
                self._styles["DcBody"],
            ))
            for cc in gaps[:10]:
                text = f"  - {_esc(cc.concept.name_ko)} ({_esc(cc.concept.chapter)})"
                story.append(Paragraph(text, self._styles["DcBody"]))
            story.append(Spacer(1, 3 * mm))
        else:
            story.append(Paragraph(
                "누락 위험 개념이 없습니다. 모든 수업 범위 내 개념이 다루어졌습니다.",
                self._styles["DcBody"],
            ))
            story.append(Spacer(1, 3 * mm))

        # Section uniformity issues
        if result.section_variance_top10:
            story.append(Paragraph(
                "<b>분반 균일성 이슈:</b>",
                self._styles["DcBody"],
            ))
            story.append(Paragraph(
                "아래 개념은 분반 간 강조도 편차가 크므로 전달 방식 재검토를 권장합니다:",
                self._styles["DcSmall"],
            ))
            for name, std in result.section_variance_top10[:5]:
                story.append(Paragraph(
                    f"  - {_esc(name)} (σ = {std:.3f})",
                    self._styles["DcBody"],
                ))
            story.append(Spacer(1, 3 * mm))

        # Overall assessment
        rate = result.effective_coverage_rate
        if rate >= 0.9:
            assessment = "우수: 수업 범위 내 개념의 90% 이상을 강의에서 다루고 있습니다."
        elif rate >= 0.7:
            assessment = "양호: 수업 범위 내 개념의 70% 이상을 다루고 있으나, 일부 누락 개념에 대한 보완이 필요합니다."
        elif rate >= 0.5:
            assessment = "주의: 수업 범위 내 개념의 절반 이상이 다루어지지 않았습니다. 수업 내용 재구성을 권장합니다."
        else:
            assessment = "경고: 수업 범위 내 개념의 절반 미만이 다루어졌습니다. 강의 내용과 교과서 범위의 불일치를 점검하시기 바랍니다."

        story.append(Paragraph(
            f"<b>종합 평가:</b> {_esc(assessment)}",
            self._styles["DcBody"],
        ))

        story.append(Spacer(1, 5 * mm))
        return story
