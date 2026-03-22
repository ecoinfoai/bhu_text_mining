"""Student longitudinal PDF report generator using ReportLab Platypus.

Builds per-student A4 PDF reports containing concept coverage trends,
score component breakdowns, cohort-relative positioning, and early
warning sections with optional LLM interpretation text.
"""

from __future__ import annotations

import io
import logging
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from forma.font_utils import esc as _esc, find_korean_font, register_korean_fonts
from forma.report_utils import minimal_png_bytes
from forma.student_longitudinal_data import (
    AlertLevel,
    CohortDistribution,
    StudentLongitudinalData,
    WarningSignal,
    compute_topic_trends,
)

logger = logging.getLogger(__name__)

__all__ = ["StudentLongitudinalPDFReportGenerator"]

_FALLBACK_PNG: bytes = minimal_png_bytes()

_ALERT_COLORS = {
    AlertLevel.NORMAL: "#2E7D32",
    AlertLevel.CAUTION: "#F57F17",
    AlertLevel.WARNING: "#C62828",
}

SCORE_INTERPRETATION_LEGEND = (
    "점수 해석 기준: 성취도 점수(ensemble_score)는 0.0~1.0 범위이며, "
    "≥0.70은 '우수'(개념을 정확히 이해하고 적용), "
    "0.45~0.70은 '보통'(기본 개념은 이해하나 일부 부족), "
    "<0.45는 '위험'(핵심 개념 미달, 보충 학습 필요)으로 해석합니다."
)


class StudentLongitudinalPDFReportGenerator:
    """Generate per-student longitudinal PDF reports using ReportLab Platypus.

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

        self._styles = getSampleStyleSheet()
        self._styles.add(ParagraphStyle(
            "StuTitle",
            parent=self._styles["Title"],
            fontName="NanumGothicBold",
            fontSize=18,
            spaceAfter=12,
        ))
        self._styles.add(ParagraphStyle(
            "StuSection",
            parent=self._styles["Heading2"],
            fontName="NanumGothicBold",
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
        ))
        self._styles.add(ParagraphStyle(
            "StuBody",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=10,
            leading=14,
            spaceAfter=4,
        ))
        self._styles.add(ParagraphStyle(
            "StuSmall",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=8,
            leading=11,
            spaceAfter=2,
            textColor=HexColor("#666666"),
        ))

    def generate_pdf(
        self,
        student_data: StudentLongitudinalData,
        cohort: CohortDistribution,
        warnings: list[WarningSignal],
        alert_level: AlertLevel,
        output_path: str,
        llm_texts: dict[str, str] | None = None,
    ) -> str:
        """Generate the student longitudinal report PDF.

        Args:
            student_data: Per-student longitudinal data.
            cohort: Cohort distribution for box plots.
            warnings: Evaluated warning signals.
            alert_level: Overall alert level.
            output_path: Output path for the PDF file.
            llm_texts: Optional dict with keys "coverage", "component",
                "position", "warning" mapping to LLM interpretation text.

        Returns:
            Absolute path to generated PDF file.
        """
        if llm_texts is None:
            llm_texts = {}

        story: list = []
        story.extend(self._build_cover_page(student_data, alert_level))
        story.extend(self._build_coverage_section(
            student_data, cohort, llm_text=llm_texts.get("coverage")))
        story.extend(self._build_component_section(
            student_data, llm_text=llm_texts.get("component")))
        story.extend(self._build_position_section(
            student_data, cohort, llm_text=llm_texts.get("position")))
        story.extend(self._build_warning_section(
            warnings, alert_level, llm_text=llm_texts.get("warning")))
        story.extend(self._build_topic_trend_section(student_data))
        story.extend(self._build_score_legend())

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

    def _build_cover_page(
        self,
        student_data: StudentLongitudinalData,
        alert_level: AlertLevel,
    ) -> list:
        """Build cover page with student info and alert badge."""
        story = []
        story.append(Spacer(1, 40 * mm))
        story.append(Paragraph(
            _esc("학생 개인 종단 분석 보고서"),
            self._styles["StuTitle"],
        ))
        story.append(Spacer(1, 10 * mm))

        name_str = student_data.student_name or "(이름 미확인)"
        class_str = student_data.class_name or "(분반 미확인)"
        weeks_str = ", ".join(str(w) for w in student_data.weeks) if student_data.weeks else "—"

        info_lines = [
            f"학번: {_esc(student_data.student_id)}",
            f"이름: {_esc(name_str)}",
            f"분반: {_esc(class_str)}",
            f"분석 기간: {_esc(weeks_str)} 주차",
            f"추세: {_esc(student_data.trend_direction)}",
        ]
        for line in info_lines:
            story.append(Paragraph(line, self._styles["StuBody"]))
            story.append(Spacer(1, 2 * mm))

        color = _ALERT_COLORS.get(alert_level, "#666666")
        alert_text = f'<font color="{color}"><b>경고 수준: {_esc(alert_level.value)}</b></font>'
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph(alert_text, self._styles["StuBody"]))

        story.append(PageBreak())
        return story

    def _build_coverage_section(
        self,
        student_data: StudentLongitudinalData,
        cohort: CohortDistribution,
        llm_text: str | None = None,
    ) -> list:
        """Build concept coverage trend section with Q1 and Q2 charts."""
        story = []
        story.append(Paragraph("1. 주차별 개념 커버리지 추세", self._styles["StuSection"]))
        story.append(Spacer(1, 3 * mm))

        from forma.student_longitudinal_charts import build_coverage_trend_chart

        for qsn in [1, 2]:
            try:
                chart_buf = build_coverage_trend_chart(
                    student_data, cohort, qsn,
                    font_path=self._font_path, dpi=self._dpi,
                )
                story.append(Image(chart_buf, width=160 * mm, height=100 * mm))
            except Exception as exc:
                logger.warning("Failed to generate Q%d coverage chart: %s", qsn, exc)
                story.append(Image(io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm))
            story.append(Spacer(1, 3 * mm))

        story.append(Paragraph(
            "파란 선 = 학생, 박스 = 전체 수강생 분포",
            self._styles["StuSmall"],
        ))

        if llm_text:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(_esc(llm_text), self._styles["StuBody"]))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_component_section(
        self,
        student_data: StudentLongitudinalData,
        llm_text: str | None = None,
    ) -> list:
        """Build score component breakdown section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("2. 항목별 점수 분해", self._styles["StuSection"]))
        story.append(Spacer(1, 3 * mm))

        from forma.student_longitudinal_charts import build_component_breakdown_chart

        try:
            chart_buf = build_component_breakdown_chart(
                student_data, font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Image(chart_buf, width=160 * mm, height=100 * mm))
        except Exception as exc:
            logger.warning("Failed to generate component breakdown chart: %s", exc)
            story.append(Image(io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm))

        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            "파랑 = 개념 커버리지, 주황 = LLM 루브릭, 초록 = 앙상블 점수, 보라 = Rasch 능력치 (보조축)",
            self._styles["StuSmall"],
        ))

        if llm_text:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(_esc(llm_text), self._styles["StuBody"]))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_position_section(
        self,
        student_data: StudentLongitudinalData,
        cohort: CohortDistribution,
        llm_text: str | None = None,
    ) -> list:
        """Build cohort-relative position section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("3. 전체 수강생 내 상대 위치", self._styles["StuSection"]))
        story.append(Spacer(1, 3 * mm))

        from forma.student_longitudinal_charts import build_cohort_position_chart

        try:
            chart_buf = build_cohort_position_chart(
                student_data, cohort,
                font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Image(chart_buf, width=160 * mm, height=100 * mm))
        except Exception as exc:
            logger.warning("Failed to generate cohort position chart: %s", exc)
            story.append(Image(io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm))

        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            "박스 = 전체 수강생 분포, 빨간 별 = 학생 (백분위 라벨 표시)",
            self._styles["StuSmall"],
        ))

        if llm_text:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(_esc(llm_text), self._styles["StuBody"]))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_warning_section(
        self,
        warnings: list[WarningSignal],
        alert_level: AlertLevel,
        llm_text: str | None = None,
    ) -> list:
        """Build early warning status section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("4. 조기 경고 상태", self._styles["StuSection"]))
        story.append(Spacer(1, 3 * mm))

        from forma.student_longitudinal_charts import build_warning_table

        try:
            chart_buf = build_warning_table(
                warnings, alert_level,
                font_path=self._font_path, dpi=self._dpi,
            )
            n_signals = max(len(warnings), 1)
            chart_height = max(60, (n_signals * 12 + 30))
            story.append(Image(
                chart_buf, width=160 * mm,
                height=min(chart_height, 200) * mm,
            ))
        except Exception as exc:
            logger.warning("Failed to generate warning table: %s", exc)
            story.append(Image(io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm))

        story.append(Spacer(1, 3 * mm))

        if llm_text:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(_esc(llm_text), self._styles["StuBody"]))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_score_legend(self) -> list:
        """Build score interpretation legend (1 paragraph)."""
        story: list = []
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            _esc(SCORE_INTERPRETATION_LEGEND),
            self._styles["StuSmall"],
        ))
        story.append(Spacer(1, 3 * mm))
        return story

    def _build_topic_trend_section(
        self,
        student_data: StudentLongitudinalData,
    ) -> list:
        """Build topic-based trend section with table.

        Renders a per-topic trend table with Kendall tau
        and Spearman rho when topic_scores is available.

        Args:
            student_data: Student data with topic_scores.

        Returns:
            List of ReportLab flowables.
        """
        story: list = []
        if not student_data.topic_scores:
            return story

        story.append(PageBreak())
        story.append(Paragraph(
            "5. Topic별 성취도 추세",
            self._styles["StuSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        # Score legend
        story.extend(self._build_score_legend())

        # Compute trends
        trends = compute_topic_trends(
            student_data.topic_scores
        )

        # Build table
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors

        weeks = sorted(
            set(
                w
                for ws in student_data.topic_scores.values()
                for w in ws
            )
        )
        header = ["Topic", "주차 수"]
        for w in weeks:
            header.append(f"{w}주")
        header.extend(["τ", "ρ", "해석"])

        table_data = [header]
        for trend in trends:
            ts = student_data.topic_scores.get(
                trend.topic, {}
            )
            row = [trend.topic, str(trend.n_weeks)]
            for w in weeks:
                v = ts.get(w)
                row.append(
                    f"{v:.2f}" if v is not None else "—"
                )
            if trend.kendall_tau is not None:
                row.append(f"{trend.kendall_tau:+.2f}")
                row.append(f"{trend.spearman_rho:+.2f}")
                if trend.kendall_p < 0.05:
                    row.append(
                        "상승↑"
                        if trend.kendall_tau > 0
                        else "하강↓"
                    )
                else:
                    row.append("변동 없음")
            else:
                row.extend(["—", "—", "3주 이상 필요"])
            table_data.append(row)

        n_cols = len(header)
        col_w = 160 * mm / n_cols
        t = Table(table_data, colWidths=[col_w] * n_cols)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "NanumGothicBold"),
            ("FONTNAME", (0, 1), (-1, -1), "NanumGothic"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ]))
        story.append(t)
        story.append(Spacer(1, 3 * mm))

        for trend in trends:
            if trend.interpretation:
                story.append(Paragraph(
                    _esc(trend.interpretation),
                    self._styles["StuSmall"],
                ))

        return story
