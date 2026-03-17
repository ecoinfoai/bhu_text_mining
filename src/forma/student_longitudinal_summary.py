"""Cohort summary table PDF report for all students.

Generates a single PDF with tabular overview of all students'
longitudinal scores, trends, percentiles, and warning levels.
No charts — tables only for quick cohort-level review.
No LLM API calls.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from forma.font_utils import esc as _esc, find_korean_font, register_korean_fonts
from forma.longitudinal_store import LongitudinalStore
from forma.student_longitudinal_data import (
    AlertLevel,
    CohortDistribution,
    build_student_data,
    evaluate_warnings,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CohortSummaryPDFReportGenerator",
    "StudentSummaryRow",
    "build_summary_rows",
]

# ---------------------------------------------------------------------------
# Alert level sort priority (lower = earlier in list)
# ---------------------------------------------------------------------------

_ALERT_SORT_ORDER = {
    AlertLevel.WARNING: 0,
    AlertLevel.CAUTION: 1,
    AlertLevel.NORMAL: 2,
}

_ALERT_LABEL = {
    AlertLevel.WARNING: "경고",
    AlertLevel.CAUTION: "주의",
    AlertLevel.NORMAL: "정상",
}

_ALERT_TEXT_COLOR = {
    AlertLevel.WARNING: "#C62828",
    AlertLevel.CAUTION: "#F57F17",
    AlertLevel.NORMAL: "#2E7D32",
}

_ALERT_ROW_BG = {
    AlertLevel.WARNING: "#FFEBEE",
    AlertLevel.CAUTION: "#FFF8E1",
    AlertLevel.NORMAL: None,  # alternating white/light gray
}

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class StudentSummaryRow:
    """One row of the cohort summary table.

    Args:
        student_id: Student identifier (학번).
        student_name: Student name from ID CSV, or "" if not found.
        class_name: Section (분반) from ID CSV, or "" if not found.
        weekly_ensemble: {week: avg ensemble_score}.
        weekly_coverage_q1: {week: concept_coverage for Q1}.
        weekly_coverage_q2: {week: concept_coverage for Q2}.
        trend_direction: 상승/정체/하강/데이터 부족.
        trend_slope: OLS slope, or None.
        latest_percentile: Most recent week's percentile (0-100).
        alert_level: AlertLevel enum value.
        triggered_signals: Signal names that fired.
    """

    student_id: str
    student_name: str = ""
    class_name: str = ""
    weekly_ensemble: dict[int, float] = field(default_factory=dict)
    weekly_coverage_q1: dict[int, float] = field(default_factory=dict)
    weekly_coverage_q2: dict[int, float] = field(default_factory=dict)
    trend_direction: str = "데이터 부족"
    trend_slope: Optional[float] = None
    latest_percentile: float = 0.0
    alert_level: AlertLevel = AlertLevel.NORMAL
    triggered_signals: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# build_summary_rows
# ---------------------------------------------------------------------------


def build_summary_rows(
    store: LongitudinalStore,
    weeks: list[int],
    cohort: CohortDistribution,
    id_map: dict[str, tuple[str, str]],
) -> list[StudentSummaryRow]:
    """Build summary rows for all students in the longitudinal store.

    Args:
        store: Loaded LongitudinalStore instance.
        weeks: List of week numbers to include.
        cohort: Pre-computed CohortDistribution.
        id_map: {student_id: (name, class_name)} from parse_id_csv.

    Returns:
        List of StudentSummaryRow, sorted by alert level (WARNING first,
        then CAUTION, then NORMAL), then by student_id within each level.
    """
    all_records = store.get_all_records()
    student_ids = sorted({r.student_id for r in all_records})

    rows: list[StudentSummaryRow] = []
    for sid in student_ids:
        name, cls = id_map.get(sid, ("", ""))

        student_data = build_student_data(
            store, sid, weeks, cohort,
            student_name=name or None,
            class_name=cls or None,
        )

        signals, alert_level = evaluate_warnings(student_data, cohort)
        triggered = [s.name for s in signals if s.triggered]

        # Weekly ensemble averages
        weekly_ensemble: dict[int, float] = {}
        weekly_cov_q1: dict[int, float] = {}
        weekly_cov_q2: dict[int, float] = {}

        for week in student_data.weeks:
            q_map = student_data.scores_by_week.get(week, {})

            # Average ensemble_score across questions
            ensemble_vals = [
                s.get("ensemble_score", 0.0)
                for s in q_map.values()
                if "ensemble_score" in s
            ]
            if ensemble_vals:
                weekly_ensemble[week] = sum(ensemble_vals) / len(ensemble_vals)

            # Q1 concept_coverage
            if 1 in q_map and "concept_coverage" in q_map[1]:
                weekly_cov_q1[week] = q_map[1]["concept_coverage"]

            # Q2 concept_coverage
            if 2 in q_map and "concept_coverage" in q_map[2]:
                weekly_cov_q2[week] = q_map[2]["concept_coverage"]

        # Latest percentile
        latest_pct = 0.0
        if student_data.weeks and student_data.percentiles_by_week:
            latest_week = student_data.weeks[-1]
            latest_pct = student_data.percentiles_by_week.get(latest_week, 0.0)

        rows.append(StudentSummaryRow(
            student_id=sid,
            student_name=name,
            class_name=cls,
            weekly_ensemble=weekly_ensemble,
            weekly_coverage_q1=weekly_cov_q1,
            weekly_coverage_q2=weekly_cov_q2,
            trend_direction=student_data.trend_direction,
            trend_slope=student_data.trend_slope,
            latest_percentile=latest_pct,
            alert_level=alert_level,
            triggered_signals=triggered,
        ))

    # Sort: WARNING first, then CAUTION, then NORMAL; within same level, by student_id
    rows.sort(key=lambda r: (_ALERT_SORT_ORDER.get(r.alert_level, 99), r.student_id))
    return rows


# ---------------------------------------------------------------------------
# PDF Report Generator
# ---------------------------------------------------------------------------


_TREND_ARROWS = {
    "상승": "\u2191상승",
    "정체": "\u2192정체",
    "하강": "\u2193하강",
    "데이터 부족": "-",
}


class CohortSummaryPDFReportGenerator:
    """Generate cohort summary table PDF using ReportLab Platypus.

    Produces a landscape A4 PDF with a single large table listing all
    students and their longitudinal metrics. No charts.

    Args:
        font_path: Path to Korean .ttf font. Auto-detected if None.
        dpi: Unused (kept for API consistency). Default 150.
    """

    def __init__(self, font_path: Optional[str] = None, dpi: int = 150) -> None:
        if font_path is None:
            font_path = find_korean_font()
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Korean font not found: {font_path}")

        self._font_path = font_path
        self._dpi = dpi

        register_korean_fonts(font_path)

        self._styles = getSampleStyleSheet()
        self._styles.add(ParagraphStyle(
            "SumTitle",
            parent=self._styles["Title"],
            fontName="NanumGothicBold",
            fontSize=18,
            spaceAfter=12,
        ))
        self._styles.add(ParagraphStyle(
            "SumSection",
            parent=self._styles["Heading2"],
            fontName="NanumGothicBold",
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
        ))
        self._styles.add(ParagraphStyle(
            "SumBody",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=10,
            leading=14,
            spaceAfter=4,
        ))
        self._styles.add(ParagraphStyle(
            "SumTableHeader",
            parent=self._styles["Normal"],
            fontName="NanumGothicBold",
            fontSize=8,
            textColor=HexColor("#FFFFFF"),
            alignment=1,  # CENTER
        ))
        self._styles.add(ParagraphStyle(
            "SumTableData",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=8,
            alignment=1,  # CENTER
        ))
        self._styles.add(ParagraphStyle(
            "SumTableDataSmall",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=7,
            alignment=1,
        ))

    def generate_pdf(
        self,
        rows: list[StudentSummaryRow],
        weeks: list[int],
        output_path: str,
        course_name: str = "",
    ) -> str:
        """Generate the cohort summary PDF.

        Args:
            rows: List of StudentSummaryRow (pre-sorted).
            weeks: Ordered list of week numbers.
            output_path: Output path for the PDF file.
            course_name: Optional course name for the cover page.

        Returns:
            Absolute path to generated PDF file.
        """
        story: list = []
        story.extend(self._build_cover_page(rows, weeks, course_name))

        if not rows:
            story.append(Paragraph(
                _esc("데이터 없음 — 종단 저장소에 학생 데이터가 없습니다."),
                self._styles["SumBody"],
            ))
        else:
            story.extend(self._build_summary_stats(rows))
            story.append(Spacer(1, 5 * mm))
            story.extend(self._build_main_table(rows, weeks))

        doc = SimpleDocTemplate(
            output_path,
            pagesize=landscape(A4),
            leftMargin=10 * mm,
            rightMargin=10 * mm,
            topMargin=15 * mm,
            bottomMargin=15 * mm,
        )
        doc.build(story)
        return os.path.abspath(output_path)

    def _build_cover_page(
        self,
        rows: list[StudentSummaryRow],
        weeks: list[int],
        course_name: str,
    ) -> list:
        """Build cover page with title and summary counts."""
        story = []
        story.append(Spacer(1, 30 * mm))
        story.append(Paragraph(
            _esc("전체 수강생 종단 분석 요약"),
            self._styles["SumTitle"],
        ))
        story.append(Spacer(1, 10 * mm))

        today_str = date.today().strftime("%Y-%m-%d")
        weeks_str = ", ".join(str(w) for w in weeks) if weeks else "-"

        info_lines = [
            f"생성일: {today_str}",
        ]
        if course_name:
            info_lines.append(f"과목: {_esc(course_name)}")
        info_lines.append(f"분석 주차: {_esc(weeks_str)}")
        info_lines.append(f"전체 학생 수: {len(rows)}명")

        for line in info_lines:
            story.append(Paragraph(line, self._styles["SumBody"]))
            story.append(Spacer(1, 2 * mm))

        story.append(PageBreak())
        return story

    def _build_summary_stats(self, rows: list[StudentSummaryRow]) -> list:
        """Build summary statistics paragraph."""
        story = []
        story.append(Paragraph("경고 수준 요약", self._styles["SumSection"]))

        n_warning = sum(1 for r in rows if r.alert_level == AlertLevel.WARNING)
        n_caution = sum(1 for r in rows if r.alert_level == AlertLevel.CAUTION)
        n_normal = sum(1 for r in rows if r.alert_level == AlertLevel.NORMAL)

        stats_text = (
            f'<font color="#C62828"><b>경고: {n_warning}명</b></font> | '
            f'<font color="#F57F17"><b>주의: {n_caution}명</b></font> | '
            f'<font color="#2E7D32"><b>정상: {n_normal}명</b></font> | '
            f"전체: {len(rows)}명"
        )
        story.append(Paragraph(stats_text, self._styles["SumBody"]))
        return story

    def _build_main_table(
        self,
        rows: list[StudentSummaryRow],
        weeks: list[int],
    ) -> list:
        """Build the main student summary table."""
        story = []
        story.append(Paragraph("학생별 종단 분석", self._styles["SumSection"]))
        story.append(Spacer(1, 3 * mm))

        # Build header row
        header_cells = [
            Paragraph("#", self._styles["SumTableHeader"]),
            Paragraph("학번", self._styles["SumTableHeader"]),
            Paragraph("이름", self._styles["SumTableHeader"]),
            Paragraph("분반", self._styles["SumTableHeader"]),
        ]
        for w in weeks:
            header_cells.append(
                Paragraph(f"W{w}", self._styles["SumTableHeader"]),
            )
        header_cells.append(Paragraph("추세", self._styles["SumTableHeader"]))
        header_cells.append(Paragraph("백분위", self._styles["SumTableHeader"]))
        header_cells.append(Paragraph("경고", self._styles["SumTableHeader"]))

        table_data = [header_cells]

        # Build data rows
        for idx, row in enumerate(rows, start=1):
            cells = [
                Paragraph(str(idx), self._styles["SumTableData"]),
                Paragraph(_esc(row.student_id), self._styles["SumTableDataSmall"]),
                Paragraph(_esc(row.student_name or "-"), self._styles["SumTableData"]),
                Paragraph(_esc(row.class_name or "-"), self._styles["SumTableData"]),
            ]
            # Weekly ensemble scores
            for w in weeks:
                if w in row.weekly_ensemble:
                    score_pct = int(round(row.weekly_ensemble[w] * 100))
                    cells.append(
                        Paragraph(f"{score_pct}%", self._styles["SumTableData"]),
                    )
                else:
                    cells.append(
                        Paragraph("-", self._styles["SumTableData"]),
                    )

            # Trend
            trend_text = _TREND_ARROWS.get(row.trend_direction, row.trend_direction)
            cells.append(Paragraph(_esc(trend_text), self._styles["SumTableData"]))

            # Percentile
            cells.append(
                Paragraph(f"{int(round(row.latest_percentile))}", self._styles["SumTableData"]),
            )

            # Alert level with color
            alert_label = _ALERT_LABEL.get(row.alert_level, "")
            alert_color = _ALERT_TEXT_COLOR.get(row.alert_level, "#000000")
            cells.append(
                Paragraph(
                    f'<font color="{alert_color}"><b>{_esc(alert_label)}</b></font>',
                    self._styles["SumTableData"],
                ),
            )

            table_data.append(cells)

        # Calculate column widths
        n_weeks = len(weeks)
        # Fixed columns: #(8mm), 학번(22mm), 이름(18mm), 분반(12mm), 추세(18mm), 백분위(14mm), 경고(14mm)
        fixed_width = 8 + 22 + 18 + 12 + 18 + 14 + 14  # = 106mm
        available = 277 - fixed_width  # landscape A4 ~ 277mm usable (297 - 20 margins)
        week_col_width = max(12, available / max(n_weeks, 1))

        col_widths = [
            8 * mm, 22 * mm, 18 * mm, 12 * mm,
        ]
        col_widths.extend([week_col_width * mm] * n_weeks)
        col_widths.extend([18 * mm, 14 * mm, 14 * mm])

        table = Table(table_data, colWidths=col_widths, repeatRows=1)

        # Table styling
        style_commands = [
            # Header row
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            # Grid
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            # Font sizes are handled by ParagraphStyle
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]

        # Row background colors based on alert level
        for idx, row in enumerate(rows, start=1):
            bg_color = _ALERT_ROW_BG.get(row.alert_level)
            if bg_color:
                style_commands.append(
                    ("BACKGROUND", (0, idx), (-1, idx), HexColor(bg_color)),
                )
            else:
                # Alternating white/light gray for NORMAL
                alt_color = "#FFFFFF" if idx % 2 == 1 else "#F5F5F5"
                style_commands.append(
                    ("BACKGROUND", (0, idx), (-1, idx), HexColor(alt_color)),
                )

        table.setStyle(TableStyle(style_commands))
        story.append(table)
        story.append(Spacer(1, 5 * mm))

        return story
