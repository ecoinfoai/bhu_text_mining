"""Longitudinal summary PDF report generator using ReportLab Platypus.

Builds A4 PDF reports from pre-computed LongitudinalSummaryData.
Each PDF contains a cover page, class trend section, trajectory section,
heatmap section, risk analysis section, and concept mastery section.
No LLM API calls are made during PDF generation.
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
    Spacer, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from forma.font_utils import esc as _esc, find_korean_font, register_korean_fonts
from forma.longitudinal_report_data import LongitudinalSummaryData
from forma.report_utils import minimal_png_bytes

logger = logging.getLogger(__name__)


_FALLBACK_PNG: bytes = minimal_png_bytes()


class LongitudinalPDFReportGenerator:
    """Generate longitudinal summary PDF reports using ReportLab Platypus.

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
            "LongTitle",
            parent=self._styles["Title"],
            fontName="NanumGothicBold",
            fontSize=18,
            spaceAfter=12,
        ))
        self._styles.add(ParagraphStyle(
            "LongSection",
            parent=self._styles["Heading2"],
            fontName="NanumGothicBold",
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
        ))
        self._styles.add(ParagraphStyle(
            "LongBody",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=10,
            leading=14,
            spaceAfter=4,
        ))
        self._styles.add(ParagraphStyle(
            "LongTableHeader",
            parent=self._styles["Normal"],
            fontName="NanumGothicBold",
            fontSize=8,
            textColor=HexColor("#FFFFFF"),
            alignment=1,
        ))
        self._styles.add(ParagraphStyle(
            "LongTableData",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=8,
            alignment=1,
        ))

    def generate_pdf(
        self,
        summary_data: LongitudinalSummaryData,
        output_path: str,
        intervention_effects: list | None = None,
        ocr_confidence_trajectories: dict | None = None,
    ) -> str:
        """Generate the longitudinal summary report PDF.

        Args:
            summary_data: Complete summary data.
            output_path: Output path for the PDF file.
            intervention_effects: Optional list of InterventionEffect (v0.10.0 US2, FR-011).
            ocr_confidence_trajectories: {student_id: [(week, mean_confidence), ...]}
                for OCR confidence trend chart (v0.12.5 US3).

        Returns:
            Absolute path to generated PDF file.
        """
        story: list = []
        story.extend(self._build_cover_page(summary_data))
        story.extend(self._build_class_trend_section(summary_data))
        story.extend(self._build_trajectory_section(summary_data))
        story.extend(self._build_heatmap_section(summary_data))
        story.extend(self._build_risk_analysis_section(summary_data))
        story.extend(self._build_concept_mastery_section(summary_data))

        # Risk trend section (v0.9.0 US2)
        if summary_data.risk_predictions:
            story.extend(self._build_risk_trend_section(summary_data))

        # Intervention effect chart section (v0.10.0 US2, FR-011)
        if intervention_effects:
            story.extend(self._build_intervention_chart_section(intervention_effects))

        # OCR confidence trend section (v0.12.5 US3)
        if ocr_confidence_trajectories:
            story.extend(self._build_ocr_confidence_trend_section(ocr_confidence_trajectories))

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

    def _build_cover_page(self, data: LongitudinalSummaryData) -> list:
        """Build cover page with report title and summary info."""
        story = []
        story.append(Spacer(1, 60 * mm))
        story.append(Paragraph(
            _esc("종단 분석 요약 보고서"),
            self._styles["LongTitle"],
        ))
        story.append(Spacer(1, 10 * mm))

        weeks_str = ", ".join(str(w) for w in data.period_weeks)
        info_lines = [
            f"학급: {_esc(data.class_name)}",
            f"분석 기간: {_esc(weeks_str)} 주차",
            f"전체 학생 수: {data.total_students}명",
            f"지속 위험군: {len(data.persistent_risk_students)}명",
        ]
        for line in info_lines:
            story.append(Paragraph(line, self._styles["LongBody"]))
            story.append(Spacer(1, 2 * mm))

        story.append(PageBreak())
        return story

    def _build_class_trend_section(self, data: LongitudinalSummaryData) -> list:
        """Build class achievement trend section with weekly averages table."""
        story = []
        story.append(Paragraph("1. 학급 성취도 추이", self._styles["LongSection"]))
        story.append(Spacer(1, 3 * mm))

        if not data.class_weekly_averages:
            story.append(Paragraph("데이터가 없습니다.", self._styles["LongBody"]))
            return story

        # Weekly averages table
        header = [Paragraph("주차", self._styles["LongTableHeader"]),
                  Paragraph("학급 평균", self._styles["LongTableHeader"])]
        rows = [header]
        for week in data.period_weeks:
            avg = data.class_weekly_averages.get(week)
            avg_str = f"{avg:.3f}" if avg is not None else "—"
            rows.append([
                Paragraph(f"W{week}", self._styles["LongTableData"]),
                Paragraph(avg_str, self._styles["LongTableData"]),
            ])

        table = Table(rows, colWidths=[40 * mm, 60 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F5F5F5")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 5 * mm))
        return story

    def _build_trajectory_section(self, data: LongitudinalSummaryData) -> list:
        """Build student trajectory line chart section."""
        story = []
        story.append(Paragraph("2. 학생별 점수 궤적", self._styles["LongSection"]))
        story.append(Spacer(1, 3 * mm))

        from forma.longitudinal_report_charts import build_trajectory_line_chart
        try:
            chart_buf = build_trajectory_line_chart(
                data, font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Image(chart_buf, width=160 * mm, height=100 * mm))
        except Exception as exc:
            logger.warning("Failed to generate trajectory chart: %s", exc)
            story.append(Image(io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm))

        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            "빨강 = 지속 위험군, 회색 = 일반 학생, 파랑 = 학급 평균",
            self._styles["LongBody"],
        ))
        story.append(Spacer(1, 5 * mm))
        return story

    def _build_heatmap_section(self, data: LongitudinalSummaryData) -> list:
        """Build student x week heatmap section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("3. 학생×주차 히트맵", self._styles["LongSection"]))
        story.append(Spacer(1, 3 * mm))

        from forma.longitudinal_report_charts import build_class_week_heatmap
        try:
            chart_buf = build_class_week_heatmap(
                data, font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Image(chart_buf, width=160 * mm, height=120 * mm))
        except Exception as exc:
            logger.warning("Failed to generate heatmap: %s", exc)
            story.append(Image(io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm))

        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            "최종 주차 점수 기준 정렬. 빨강(낮음) → 초록(높음). 회색 = 결측.",
            self._styles["LongBody"],
        ))
        story.append(Spacer(1, 5 * mm))
        return story

    def _build_risk_analysis_section(self, data: LongitudinalSummaryData) -> list:
        """Build persistent risk student analysis section."""
        story = []
        story.append(Paragraph("4. 지속 위험군 분석", self._styles["LongSection"]))
        story.append(Spacer(1, 3 * mm))

        n_persistent = len(data.persistent_risk_students)
        story.append(Paragraph(
            f"전 기간 지속 위험군: <b>{n_persistent}명</b>",
            self._styles["LongBody"],
        ))
        story.append(Spacer(1, 2 * mm))

        if data.persistent_risk_students:
            # Risk student table
            header = [
                Paragraph("학생", self._styles["LongTableHeader"]),
                Paragraph("최종 주차 점수", self._styles["LongTableHeader"]),
                Paragraph("추세(기울기)", self._styles["LongTableHeader"]),
            ]
            rows = [header]

            for sid in data.persistent_risk_students:
                traj = next(
                    (t for t in data.student_trajectories if t.student_id == sid),
                    None,
                )
                if traj is None:
                    continue

                # Get final week score
                final_score = "—"
                if data.period_weeks and traj.weekly_scores:
                    for w in reversed(data.period_weeks):
                        if w in traj.weekly_scores:
                            final_score = f"{traj.weekly_scores[w]:.3f}"
                            break

                trend_str = f"{traj.overall_trend:+.4f}"

                rows.append([
                    Paragraph(_esc(sid), self._styles["LongTableData"]),
                    Paragraph(final_score, self._styles["LongTableData"]),
                    Paragraph(trend_str, self._styles["LongTableData"]),
                ])

            table = Table(rows, colWidths=[50 * mm, 50 * mm, 50 * mm])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#C62828")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(table)
        else:
            story.append(Paragraph(
                "지속 위험군 학생이 없습니다.",
                self._styles["LongBody"],
            ))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_concept_mastery_section(self, data: LongitudinalSummaryData) -> list:
        """Build concept mastery change section with chart and table."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("5. 개념별 마스터리 변화", self._styles["LongSection"]))
        story.append(Spacer(1, 3 * mm))

        if not data.concept_mastery_changes:
            story.append(Paragraph(
                "개념 마스터리 데이터가 없습니다.",
                self._styles["LongBody"],
            ))
            return story

        # Bar chart
        from forma.longitudinal_report_charts import build_concept_mastery_bar_chart
        try:
            chart_buf = build_concept_mastery_bar_chart(
                data, font_path=self._font_path, dpi=self._dpi,
            )
            n = len(data.concept_mastery_changes)
            chart_height = max(60, n * 15)
            story.append(Image(chart_buf, width=160 * mm,
                               height=min(chart_height, 200) * mm))
        except Exception as exc:
            logger.warning("Failed to generate concept mastery chart: %s", exc)
            story.append(Image(io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm))

        story.append(Spacer(1, 5 * mm))

        # Detail table
        header = [
            Paragraph("개념", self._styles["LongTableHeader"]),
            Paragraph("첫 주차", self._styles["LongTableHeader"]),
            Paragraph("마지막 주차", self._styles["LongTableHeader"]),
            Paragraph("변화(Δ)", self._styles["LongTableHeader"]),
        ]
        rows = [header]
        for c in data.concept_mastery_changes:
            delta_str = f"{c.delta:+.3f}"
            rows.append([
                Paragraph(_esc(c.concept), self._styles["LongTableData"]),
                Paragraph(f"{c.week_start_ratio:.3f}", self._styles["LongTableData"]),
                Paragraph(f"{c.week_end_ratio:.3f}", self._styles["LongTableData"]),
                Paragraph(delta_str, self._styles["LongTableData"]),
            ])

        table = Table(rows, colWidths=[50 * mm, 35 * mm, 35 * mm, 35 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F5F5F5")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 5 * mm))
        return story

    def _build_risk_trend_section(self, data: LongitudinalSummaryData) -> list:
        """Build risk prediction trend section with chart.

        Args:
            data: Summary data with risk_predictions field.

        Returns:
            List of ReportLab flowables.
        """
        story = []
        story.append(PageBreak())
        story.append(Paragraph("6. 드롭 리스크 예측", self._styles["LongSection"]))
        story.append(Spacer(1, 3 * mm))

        if not data.risk_predictions:
            story.append(Paragraph(
                "리스크 예측 데이터가 없습니다.",
                self._styles["LongBody"],
            ))
            return story

        from forma.longitudinal_report_charts import build_risk_trend_chart
        try:
            chart_buf = build_risk_trend_chart(
                data.risk_predictions,
                font_path=self._font_path,
                dpi=self._dpi,
            )
            story.append(Image(chart_buf, width=160 * mm, height=80 * mm))
        except Exception as exc:
            logger.warning("Failed to generate risk trend chart: %s", exc)
            story.append(Image(io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_intervention_chart_section(self, effects: list) -> list:
        """Build intervention effect pre/post chart section (FR-011).

        Args:
            effects: List of InterventionEffect objects.

        Returns:
            List of ReportLab flowables.
        """
        story = []
        story.append(PageBreak())
        story.append(Paragraph("개입 전후 점수 변화", self._styles["LongSection"]))
        story.append(Spacer(1, 3 * mm))

        from forma.longitudinal_report_charts import build_intervention_effect_chart
        try:
            n_sufficient = sum(1 for e in effects if e.sufficient_data)
            chart_height = max(60, n_sufficient * 12)
            chart_buf = build_intervention_effect_chart(
                effects,
                font_path=self._font_path,
                dpi=self._dpi,
            )
            story.append(Image(
                chart_buf, width=160 * mm,
                height=min(chart_height, 200) * mm,
            ))
        except Exception as exc:
            logger.warning("Failed to generate intervention effect chart: %s", exc)
            story.append(Image(io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm))

        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            "파랑 = 개입 전 평균, 초록 = 개입 후 평균. "
            "데이터 부족 건은 제외.",
            self._styles["LongBody"],
        ))
        story.append(Spacer(1, 5 * mm))
        return story

    def _build_ocr_confidence_trend_section(
        self, trajectories: dict[str, list[tuple[int, float]]],
    ) -> list:
        """Build OCR confidence trend chart section.

        Args:
            trajectories: {student_id: [(week, mean_confidence), ...]}.

        Returns:
            List of ReportLab flowables.
        """
        story = []
        story.append(PageBreak())
        story.append(Paragraph("텍스트 인식 신뢰도 추이", self._styles["LongSection"]))
        story.append(Spacer(1, 3 * mm))

        from forma.longitudinal_report_charts import build_ocr_confidence_trend_chart
        try:
            chart_buf = build_ocr_confidence_trend_chart(
                trajectories,
                font_path=self._font_path,
                dpi=self._dpi,
            )
            story.append(Image(
                chart_buf, width=160 * mm, height=100 * mm,
            ))
        except Exception as exc:
            logger.warning("Failed to generate OCR confidence trend chart: %s", exc)
            story.append(Image(io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm))

        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            "3주 이상 연속 기준값(0.75) 미만 학생은 빨간색으로 표시. "
            "인식률 데이터가 없는 주차는 건너뜁니다.",
            self._styles["LongBody"],
        ))
        story.append(Spacer(1, 5 * mm))
        return story
