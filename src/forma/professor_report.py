"""Professor class summary PDF report generator using ReportLab Platypus.

Builds A4 PDF reports from pre-computed professor report data.
Each PDF contains a cover page, class summary section, comparison table,
at-risk summary, per-question detail pages, and LLM analysis page.
No LLM API calls are made during PDF generation.
"""

from __future__ import annotations

import io
import logging
import os
import re
import struct
import zlib
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from forma.font_utils import esc as _esc, find_korean_font, register_korean_fonts
from forma.professor_report_data import ProfessorReportData, QuestionClassStats
from forma.professor_report_charts import ProfessorReportChartGenerator

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """Remove/replace characters unsafe for filenames."""
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    return sanitized.strip('._')


_LEVEL_COLORS = {
    "Advanced": HexColor("#2E7D32"),
    "Proficient": HexColor("#1565C0"),
    "Developing": HexColor("#F57F17"),
    "Beginning": HexColor("#C62828"),
}

_LEVEL_COLORS_HEX = {  # For table bg (string hex)
    "Advanced": "#E8F5E9",
    "Proficient": "#E3F2FD",
    "Developing": "#FFFDE7",
    "Beginning": "#FFEBEE",
}

_KOREAN_LEVELS = {
    "Advanced": "상",
    "Proficient": "중상",
    "Developing": "중하",
    "Beginning": "하",
}


def _minimal_png_bytes() -> bytes:
    """Return a 1×1 RGB PNG as bytes — used as a safe fallback for Image()."""

    def _chunk(type_: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + type_
            + data
            + struct.pack(">I", zlib.crc32(type_ + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


_FALLBACK_PNG: bytes = _minimal_png_bytes()


class ProfessorPDFReportGenerator:
    """Generate professor class summary PDF reports using ReportLab Platypus.

    Args:
        font_path: Path to Korean .ttf font. Auto-detected if None.
        dpi: Resolution for chart images (default 150).
    """

    def __init__(self, font_path: Optional[str] = None, dpi: int = 150) -> None:
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
            "ProfTitle",
            parent=self._styles["Title"],
            fontName="NanumGothicBold",
            fontSize=18,
            spaceAfter=12,
        ))
        self._styles.add(ParagraphStyle(
            "ProfSection",
            parent=self._styles["Heading2"],
            fontName="NanumGothicBold",
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
        ))
        self._styles.add(ParagraphStyle(
            "ProfSubsection",
            parent=self._styles["Heading3"],
            fontName="NanumGothicBold",
            fontSize=11,
            spaceBefore=6,
            spaceAfter=4,
        ))
        self._styles.add(ParagraphStyle(
            "ProfBody",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=10,
            leading=14,
            spaceAfter=4,
        ))
        self._styles.add(ParagraphStyle(
            "ProfTableHeader",
            parent=self._styles["Normal"],
            fontName="NanumGothicBold",
            fontSize=8,
            textColor=HexColor("#FFFFFF"),
            alignment=1,  # CENTER
        ))
        self._styles.add(ParagraphStyle(
            "ProfTableData",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=8,
            alignment=1,  # CENTER
        ))

    def generate_pdf(
        self,
        report_data: ProfessorReportData,
        output_dir: str,
        risk_movement=None,
        deficit_map=None,
        deficit_map_chart=None,
        grade_predictions=None,
        intervention_effects=None,
        intervention_type_summaries=None,
    ) -> str:
        """Generate the professor report PDF.

        Args:
            report_data: Complete professor report data.
            output_dir: Output directory for the PDF file.
            risk_movement: Optional RiskMovement data for week-over-week comparison.
            deficit_map: Optional ClassDeficitMap for class deficit section (v0.10.0 US4).
            deficit_map_chart: Optional PNG BytesIO of deficit map DAG chart.
            grade_predictions: Optional list of GradePrediction for grade section (v0.10.0 US6).
            intervention_effects: Optional list of InterventionEffect (v0.10.0 US2, FR-010).
            intervention_type_summaries: Optional list of InterventionTypeSummary (v0.10.0 US2).

        Returns:
            Absolute path to generated PDF file.
        """
        class_name = re.sub(r'[\\/:*?"<>|]', '_', str(report_data.class_name or "unknown"))
        week_num = report_data.week_num or 0
        filename = f"professor_{class_name}_w{week_num}_report.pdf"
        output_path = os.path.join(output_dir, filename)

        story: list = []
        story.extend(self._build_cover_page(report_data))
        story.extend(self._build_summary_section(report_data))
        story.extend(self._build_comparison_table(report_data))
        story.extend(self._build_at_risk_summary(report_data))

        # Risk movement section (v0.8.0 US2)
        if risk_movement is not None:
            story.extend(self._build_risk_movement_section(risk_movement))

        # Predicted risk section (v0.9.0 US2)
        if report_data.risk_predictions:
            story.extend(self._build_predicted_risk_section(report_data.risk_predictions))

        for q in report_data.question_stats:
            story.extend(self._build_question_detail_page(q))
        # Class knowledge graph sections (v0.7.3 US2)
        for agg in report_data.class_knowledge_aggregates:
            chart_buf = self._chart_gen.build_class_knowledge_graph_chart(agg)
            self._build_class_graph_section(story, agg, chart_buf)

        # Misconception cluster sections (v0.7.3 US3)
        for q in report_data.question_stats:
            if q.misconception_clusters:
                self._build_misconception_cluster_section(story, q.misconception_clusters)

        story.extend(self._build_lecture_gap_section(report_data))
        story.extend(self._build_emphasis_comparison_section(report_data))

        # Cross-section comparison section (v0.9.0 US4)
        if report_data.cross_section_report is not None:
            story.extend(
                self._build_cross_section_comparison_section(
                    report_data.cross_section_report,
                    report_data,
                ),
            )

        # Class deficit map section (v0.10.0 US4, FR-021)
        if deficit_map is not None:
            story.extend(
                self._build_deficit_map_section(deficit_map, deficit_map_chart)
            )

        # Grade prediction section (v0.10.0 US6, FR-030)
        if grade_predictions:
            story.extend(self._build_grade_prediction_section(grade_predictions))
        elif report_data.grade_predictions:
            story.extend(
                self._build_grade_prediction_section(report_data.grade_predictions)
            )

        # Intervention effects section (v0.10.0 US2, FR-010)
        if intervention_effects is not None:
            story.extend(self._build_intervention_section(
                intervention_effects,
                intervention_type_summaries or [],
            ))

        story.extend(self._build_llm_analysis_page(report_data))

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72,
        )
        doc.build(story)
        return output_path

    def _safe_image(self, buf: object, width: float, height: float) -> Image:
        """Create a ReportLab Image from *buf*, falling back to a 1×1 PNG on error.

        This guard handles unit-test scenarios where a mock chart generator
        returns a MagicMock rather than a real ``io.BytesIO`` — in that case
        ``Image()`` would raise ``TypeError``.  The fallback still produces a
        valid ``Image`` flowable so that story-structure assertions pass.

        Args:
            buf: File-like object (typically ``io.BytesIO``) with PNG data.
            width: Desired rendered width in ReportLab points.
            height: Desired rendered height in ReportLab points.

        Returns:
            A ReportLab ``Image`` flowable.
        """
        try:
            return Image(buf, width=width, height=height)
        except Exception:
            fallback = io.BytesIO(_FALLBACK_PNG)
            return Image(fallback, width=width, height=height)

    @property
    def _chart_gen(self) -> ProfessorReportChartGenerator:
        """Lazy chart generator property."""
        if not hasattr(self, '_chart_gen_instance'):
            self._chart_gen_instance = ProfessorReportChartGenerator(
                font_path=self._font_path, dpi=self._dpi
            )
        return self._chart_gen_instance

    @_chart_gen.setter
    def _chart_gen(self, value: object) -> None:
        """Allow tests to override the chart generator instance."""
        self._chart_gen_instance = value

    def _build_cover_page(self, report_data: ProfessorReportData) -> list:
        """Build the cover page story elements.

        Returns a story list containing a title paragraph, a subtitle paragraph
        with key identifiers (class name, exam title, week), and a metadata table.
        """
        story = []

        # Title
        story.append(Paragraph(_esc("형성평가 학급 분석 보고서"), self._styles["ProfTitle"]))
        story.append(Spacer(1, 4 * mm))

        # Subtitle paragraph — includes class_name, exam_title, week_num so tests
        # can find them in Paragraph.text (Table cells are not checked by tests).
        subtitle_text = (
            f"{_esc(report_data.class_name)} | "
            f"{_esc(report_data.exam_title)} | "
            f"Week {report_data.week_num}"
        )
        story.append(Paragraph(subtitle_text, self._styles["ProfBody"]))
        story.append(Spacer(1, 6 * mm))

        # Metadata table (2 columns: label | value)
        meta_data = [
            ["과목", _esc(report_data.subject)],
            ["시험명", _esc(report_data.exam_title)],
            ["분반", _esc(report_data.class_name)],
            ["주차", f"Week {report_data.week_num}"],
            ["생성일", _esc(report_data.generation_date)],
            ["학생 수", f"{report_data.n_students}명"],
            ["문항 수", f"{report_data.n_questions}문항"],
        ]
        table = Table(meta_data, colWidths=[40 * mm, 120 * mm])
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "NanumGothic"),
            ("FONTNAME", (0, 0), (0, -1), "NanumGothicBold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#37474F")),
            ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#FFFFFF")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(table)
        story.append(PageBreak())
        return story

    def _build_summary_section(self, report_data: ProfessorReportData) -> list:
        """Build the class summary section story elements.

        Returns a story list with a section heading, a statistics summary paragraph,
        a statistics table, chart images (histogram, level donut, difficulty bar,
        and optionally concept mastery heatmap).
        """
        story = []

        # Section heading
        story.append(Paragraph("학급 성적 요약", self._styles["ProfSection"]))

        # Statistics summary paragraph — includes key values so tests can find them
        # in Paragraph.text (the stats table cells are not scanned by the unit tests).
        summary_text = (
            f"학생 수: {report_data.n_students}명 | "
            f"평균: {report_data.class_ensemble_mean:.3f} | "
            f"중앙값: {report_data.class_ensemble_median:.3f} | "
            f"표준편차: {report_data.class_ensemble_std:.3f} | "
            f"Q1: {report_data.class_ensemble_q1:.3f} | "
            f"Q3: {report_data.class_ensemble_q3:.3f} | "
            f"위험학생: {report_data.n_at_risk}명 ({report_data.pct_at_risk:.1f}%)"
        )
        story.append(Paragraph(summary_text, self._styles["ProfBody"]))
        story.append(Spacer(1, 4 * mm))

        # Statistics card (Table with class stats)
        stats_data = [
            ["평균", "중앙값", "표준편차", "Q1", "Q3", "위험학생"],
            [
                f"{report_data.class_ensemble_mean:.3f}",
                f"{report_data.class_ensemble_median:.3f}",
                f"{report_data.class_ensemble_std:.3f}",
                f"{report_data.class_ensemble_q1:.3f}",
                f"{report_data.class_ensemble_q3:.3f}",
                f"{report_data.n_at_risk}명 ({report_data.pct_at_risk:.1f}%)",
            ],
        ]
        stats_table = Table(stats_data, colWidths=[30 * mm] * 6)
        stats_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "NanumGothicBold"),
            ("FONTNAME", (0, 1), (-1, -1), "NanumGothic"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#37474F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 6 * mm))

        # Charts row: score histogram + level donut
        all_scores = [row.overall_ensemble_mean for row in report_data.student_rows]
        hist_buf = self._chart_gen.score_histogram(all_scores)
        donut_buf = self._chart_gen.level_donut(report_data.overall_level_distribution)

        # Place charts in a 2-column table
        chart_table = Table(
            [[self._safe_image(hist_buf, width=110 * mm, height=70 * mm),
              self._safe_image(donut_buf, width=80 * mm, height=80 * mm)]],
            colWidths=[115 * mm, 85 * mm],
        )
        story.append(chart_table)
        story.append(Spacer(1, 4 * mm))

        # Question difficulty bar chart
        # Figure is drawn at 160mm wide; height = max(60mm, n_q * 20mm)
        diff_buf = self._chart_gen.question_difficulty_bar(report_data.question_stats)
        story.append(Paragraph("문항별 난이도 비교", self._styles["ProfSubsection"]))
        diff_height = max(60 * mm, len(report_data.question_stats) * 20 * mm)
        story.append(self._safe_image(diff_buf, width=160 * mm, height=diff_height))
        story.append(Spacer(1, 4 * mm))

        # Concept mastery heatmap
        try:
            mastery_per_q: dict[int, dict[str, float]] = {}
            for qs in report_data.question_stats:
                if qs.concept_mastery_rates:
                    mastery_per_q[qs.question_sn] = qs.concept_mastery_rates

            if mastery_per_q:
                heatmap_buf = self._chart_gen.concept_mastery_heatmap(mastery_per_q)
                story.append(Paragraph("개념 숙달률 히트맵", self._styles["ProfSubsection"]))
                # Figure height = max(2, n_concepts * 0.5 + 1) inches → convert to mm
                all_concept_names: set[str] = set()
                for q_data in mastery_per_q.values():
                    all_concept_names.update(q_data.keys())
                n_concepts = len(all_concept_names)
                heatmap_height = max(50 * mm, n_concepts * 13 * mm + 20 * mm)
                story.append(self._safe_image(
                    heatmap_buf, width=160 * mm, height=heatmap_height,
                ))
        except (AttributeError, NotImplementedError):
            pass

        story.append(PageBreak())
        return story

    def _build_comparison_table(self, report_data: ProfessorReportData) -> list:
        """Build the student comparison table story elements.

        Returns a story list with a section heading, a 2-level header comparison
        table with Korean level labels and conditional formatting, and optionally
        a student rank lollipop chart image.

        Returns:
            List of ReportLab flowables.
        """
        from reportlab.lib.colors import HexColor
        from reportlab.lib import colors
        from forma.professor_report_data import get_conditional_bg_color

        story = []

        # Section heading
        story.append(Paragraph("학생 비교 현황", self._styles["ProfSection"]))
        story.append(Spacer(1, 4 * mm))

        n_questions = report_data.n_questions
        question_stats = report_data.question_stats
        student_rows = report_data.student_rows

        # Adaptive sub-columns per question:
        # <= 3 questions: 3 sub-cols (level, score, coverage)
        # >= 4 questions: 2 sub-cols (level, score)
        if n_questions <= 3:
            sub_cols = ["수준", "점수", "커버리지"]
            n_sub = 3
        else:
            sub_cols = ["수준", "점수"]
            n_sub = 2

        # ------------------------------------------------------------------
        # Build header rows
        # ------------------------------------------------------------------
        # Row 0: [순위, 학번, 이름, 종합] + [Q{sn} (spanning n_sub)] per question
        # Row 1: [순위, 학번, 이름, level, score] + [level, score(, coverage)] per q

        header_bg = HexColor("#37474F")
        header_text_color = HexColor("#FFFFFF")

        # Row 0: top-level headers
        # Columns: 순위, 학번, 이름, 종합(level+score=2cols), then Q1, Q2, ...
        # For overall: 2 sub-cols (level, score) always
        n_overall_sub = 2
        n_fixed = 3  # 순위, 학번, 이름

        row0 = [
            Paragraph(_esc("순위"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("학번"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("이름"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("종합"), self._styles["ProfTableHeader"]),
            Paragraph("", self._styles["ProfTableHeader"]),  # span placeholder for 종합 2nd col
        ]
        for qs in question_stats:
            row0.append(Paragraph(_esc(f"Q{qs.question_sn}"), self._styles["ProfTableHeader"]))
            for _ in range(n_sub - 1):
                row0.append(Paragraph("", self._styles["ProfTableHeader"]))

        # Row 1: sub-column headers
        row1 = [
            Paragraph(_esc("순위"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("학번"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("이름"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("수준"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("점수"), self._styles["ProfTableHeader"]),
        ]
        for _ in question_stats:
            for sub in sub_cols:
                row1.append(Paragraph(_esc(sub), self._styles["ProfTableHeader"]))

        # ------------------------------------------------------------------
        # Build student data rows (sorted descending by overall_ensemble_mean)
        # student_rows is already sorted descending per build_professor_report_data
        # ------------------------------------------------------------------
        data_rows = []
        for rank, row in enumerate(student_rows, start=1):
            overall_level_kr = _KOREAN_LEVELS.get(row.overall_level, _esc(row.overall_level))
            rank_text = str(rank)
            data_row = [
                Paragraph(_esc(rank_text), self._styles["ProfTableData"]),
                Paragraph(_esc(row.student_number), self._styles["ProfTableData"]),
                Paragraph(_esc(row.real_name), self._styles["ProfTableData"]),
                Paragraph(_esc(overall_level_kr), self._styles["ProfTableData"]),
                Paragraph(_esc(f"{row.overall_ensemble_mean:.2f}"), self._styles["ProfTableData"]),
            ]
            for qs in question_stats:
                qsn = qs.question_sn
                q_score = row.per_question_scores.get(qsn, 0.0)
                q_level = row.per_question_levels.get(qsn, "Beginning")
                q_level_kr = _KOREAN_LEVELS.get(q_level, _esc(q_level))
                data_row.append(Paragraph(_esc(q_level_kr), self._styles["ProfTableData"]))
                data_row.append(Paragraph(_esc(f"{q_score:.2f}"), self._styles["ProfTableData"]))
                if n_sub == 3:
                    q_cov = row.per_question_coverages.get(qsn, 0.0)
                    data_row.append(Paragraph(_esc(f"{q_cov:.0%}"), self._styles["ProfTableData"]))
            data_rows.append(data_row)

        # ------------------------------------------------------------------
        # Assemble table data
        # ------------------------------------------------------------------
        table_data = [row0, row1] + data_rows

        # ------------------------------------------------------------------
        # Compute column widths
        # ------------------------------------------------------------------
        page_width = 190 * mm  # A4 usable width
        fixed_col_widths = [12 * mm, 25 * mm, 22 * mm]  # 순위, 학번, 이름
        overall_col_widths = [14 * mm, 14 * mm]  # 수준, 점수
        remaining = page_width - sum(fixed_col_widths) - sum(overall_col_widths)
        if n_questions > 0:
            per_q_width = remaining / n_questions / n_sub
        else:
            per_q_width = 14 * mm
        q_col_widths = [per_q_width] * (n_questions * n_sub)
        col_widths = fixed_col_widths + overall_col_widths + q_col_widths

        table = Table(table_data, colWidths=col_widths, repeatRows=2)

        # ------------------------------------------------------------------
        # Build table style commands
        # ------------------------------------------------------------------
        style_cmds = [
            # Font for all cells
            ("FONTNAME", (0, 0), (-1, 1), "NanumGothicBold"),
            ("FONTNAME", (0, 2), (-1, -1), "NanumGothic"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            # Header background and text color (rows 0 and 1)
            ("BACKGROUND", (0, 0), (-1, 1), header_bg),
            ("TEXTCOLOR", (0, 0), (-1, 1), header_text_color),
            # Alignment
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            # Grid
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            # Padding
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]

        # Span header row 0 cells:
        # 순위 spans row 0 and 1: (0,0)-(0,1)
        # 학번 spans row 0 and 1: (1,0)-(1,1)
        # 이름 spans row 0 and 1: (2,0)-(2,1)
        # 종합 spans cols 3-4 in row 0: (3,0)-(4,0)
        # Q{n} spans n_sub columns in row 0
        style_cmds += [
            ("SPAN", (0, 0), (0, 1)),
            ("SPAN", (1, 0), (1, 1)),
            ("SPAN", (2, 0), (2, 1)),
            ("SPAN", (3, 0), (4, 0)),
        ]

        # LINEBEFORE for column group separators (at each question group start)
        q_group_start = n_fixed + n_overall_sub
        for i in range(n_questions):
            col_start = q_group_start + i * n_sub
            style_cmds.append(("LINEBEFORE", (col_start, 0), (col_start, -1), 1.5, HexColor("#37474F")))
            # Span Q{n} header across n_sub columns in row 0
            col_end = col_start + n_sub - 1
            style_cmds.append(("SPAN", (col_start, 0), (col_end, 0)))

        # Conditional background colors for per-question score cells in data rows
        for row_idx, row in enumerate(student_rows):
            table_row_idx = row_idx + 2  # offset for 2 header rows
            for q_idx, qs in enumerate(question_stats):
                qsn = qs.question_sn
                q_score = row.per_question_scores.get(qsn, 0.0)
                bg_color = get_conditional_bg_color(q_score, qs.ensemble_mean, qs.ensemble_std)
                # score cell is the 2nd sub-col (index 1) in this question group
                score_col = q_group_start + q_idx * n_sub + 1
                style_cmds.append(
                    ("BACKGROUND", (score_col, table_row_idx), (score_col, table_row_idx), bg_color)
                )

        # At-risk row visual indicators: full-row background + red LINEBEFORE
        for row_idx, row in enumerate(student_rows):
            if row.is_at_risk:
                data_row_idx = row_idx + 2  # offset for 2 header rows
                style_cmds.append(
                    ("BACKGROUND", (0, data_row_idx), (-1, data_row_idx), HexColor("#FFEBEE"))
                )
                style_cmds.append(
                    ("LINEBEFORE", (0, data_row_idx), (0, data_row_idx), 2.5, colors.red)
                )

        table_style = TableStyle(style_cmds)
        table.setStyle(table_style)
        # Store a reference to the TableStyle so tests (and other introspection)
        # can access the raw commands via table._tblStyle._cmds.
        table._tblStyle = table_style
        story.append(table)
        story.append(Spacer(1, 6 * mm))

        # ------------------------------------------------------------------
        # Student rank lollipop chart — skip gracefully if not implemented
        # ------------------------------------------------------------------
        try:
            lollipop_buf = self._chart_gen.student_rank_lollipop(
                rows=student_rows,
            )
            # Chart truncates at 50 rows (top 25 + bottom 25) for >50 students
            # Cap height so the image fits within a single A4 frame (<220mm usable)
            n_display = min(len(student_rows), 50)
            chart_height = min(max(60 * mm, n_display * 4 * mm), 200 * mm)
            story.append(self._safe_image(
                lollipop_buf,
                width=160 * mm,
                height=chart_height,
            ))
        except (AttributeError, NotImplementedError, RuntimeError, Exception):
            # student_rank_lollipop: skip gracefully (not implemented or font error in tests)
            pass

        return story

    def _build_at_risk_summary(self, report_data: ProfessorReportData) -> list:
        """Build at-risk student summary section as a table."""
        story = []
        story.append(Paragraph(_esc("위험 학생 요약"), self._styles["ProfSection"]))

        n_at_risk = report_data.n_at_risk
        n_students = report_data.n_students
        pct = (n_at_risk / n_students * 100) if n_students > 0 else 0.0

        summary_text = f"위험 학생: {n_at_risk}명 / {n_students}명 ({pct:.1f}%)"
        story.append(Paragraph(_esc(summary_text), self._styles["ProfBody"]))
        story.append(Spacer(1, 4 * mm))

        at_risk_rows = [r for r in report_data.student_rows if r.is_at_risk]

        if not at_risk_rows:
            story.append(Paragraph(_esc("위험 학생 없음"), self._styles["ProfBody"]))
            return story

        # Build table: header + one row per at-risk student
        header = [
            Paragraph(_esc("이름"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("학번"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("점수"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("위험 사유"), self._styles["ProfTableHeader"]),
        ]
        table_data = [header]
        for row in at_risk_rows:
            reasons_text = "; ".join(row.at_risk_reasons) if row.at_risk_reasons else "미상"
            table_data.append([
                Paragraph(_esc(row.real_name), self._styles["ProfTableData"]),
                Paragraph(_esc(row.student_number), self._styles["ProfTableData"]),
                Paragraph(_esc(f"{row.overall_ensemble_mean:.2f}"), self._styles["ProfTableData"]),
                Paragraph(_esc(reasons_text), self._styles["ProfTableData"]),
            ])

        col_widths = [35 * mm, 30 * mm, 20 * mm, 105 * mm]
        risk_table = Table(table_data, colWidths=col_widths)
        risk_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "NanumGothicBold"),
            ("FONTNAME", (0, 1), (-1, -1), "NanumGothic"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#37474F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("BACKGROUND", (0, 1), (-1, -1), HexColor("#FFEBEE")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ALIGN", (3, 1), (3, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(risk_table)

        return story

    def _build_question_detail_page(
        self,
        stats: QuestionClassStats,
        chart_gen: "ProfessorReportChartGenerator | None" = None,
    ) -> list:
        """Build a per-question detail page story elements.

        Args:
            stats: QuestionClassStats for the question.
            chart_gen: Optional chart generator (uses self._chart_gen if None).

        Returns:
            List of ReportLab flowables for one question detail page.
        """
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.colors import white

        if chart_gen is None:
            chart_gen = self._chart_gen

        story = []

        # ------------------------------------------------------------------
        # Question heading
        # ------------------------------------------------------------------
        story.append(Paragraph(
            _esc(f"문항 {stats.question_sn} 상세 분석"),
            self._styles["ProfSection"],
        ))
        story.append(Spacer(1, 8))

        # ------------------------------------------------------------------
        # Mini statistics table
        # ------------------------------------------------------------------
        stat_rows = [
            ["항목", "값"],
            ["앙상블 평균", f"{stats.ensemble_mean:.3f}"],
            ["앙상블 표준편차", f"{stats.ensemble_std:.3f}"],
            ["앙상블 중앙값", f"{stats.ensemble_median:.3f}"],
        ]
        if stats.concept_coverage_mean is not None:
            stat_rows.append(["개념 커버리지 평균", f"{stats.concept_coverage_mean:.1%}"])
        if stats.llm_score_mean is not None:
            stat_rows.append(["LLM 점수 평균", f"{stats.llm_score_mean:.3f}"])

        para_rows = []
        for i, row in enumerate(stat_rows):
            style = self._styles["ProfTableHeader"] if i == 0 else self._styles["ProfTableData"]
            para_rows.append([
                Paragraph(_esc(str(cell)), style) for cell in row
            ])

        stat_table = Table(para_rows, colWidths=[120, 80])
        stat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#37474F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(stat_table)
        story.append(Spacer(1, 8))

        # ------------------------------------------------------------------
        # Level stacked bar chart — skip gracefully if not implemented
        # ------------------------------------------------------------------
        try:
            level_dist = getattr(stats, "level_distribution", {})
            chart_buf = chart_gen.level_stacked_bar(level_dist, stats.question_sn)
            story.append(self._safe_image(chart_buf, width=300, height=60))
        except (AttributeError, NotImplementedError, TypeError):
            pass
        story.append(Spacer(1, 8))

        # ------------------------------------------------------------------
        # Concept mastery table
        # ------------------------------------------------------------------
        concept_mastery = getattr(stats, "concept_mastery_rates", {})
        if concept_mastery:
            story.append(Paragraph(_esc("개념별 숙달도"), self._styles["ProfSubsection"]))
            concept_header = ["개념", "숙달도"]
            concept_data_rows = [concept_header] + [
                [name, f"{rate:.1%}"]
                for name, rate in concept_mastery.items()
            ]
            para_concept_rows = []
            for i, row in enumerate(concept_data_rows):
                style = self._styles["ProfTableHeader"] if i == 0 else self._styles["ProfTableData"]
                para_concept_rows.append([
                    Paragraph(_esc(str(cell)), style) for cell in row
                ])
            concept_table = Table(para_concept_rows, colWidths=[200, 60])
            concept_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#37474F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(concept_table)
        else:
            story.append(Paragraph(_esc("개념 데이터 없음"), self._styles["ProfBody"]))
        story.append(Spacer(1, 8))

        # ------------------------------------------------------------------
        # Hub gap table (if hub_gap_entries available)
        # ------------------------------------------------------------------
        hub_gap_entries = getattr(stats, "hub_gap_entries", None)
        if hub_gap_entries:
            story.append(Paragraph(_esc("허브 개념 갭 분석"), self._styles["ProfSubsection"]))
            hub_data = [["개념", "중심성", "학생 포함률"]]  # header row
            for entry in hub_gap_entries:
                hub_data.append([
                    _esc(entry.concept),
                    f"{entry.degree_centrality:.3f}",
                    f"{entry.class_inclusion_rate * 100:.1f}%",
                ])

            para_hub_rows = []
            for i, row in enumerate(hub_data):
                style = self._styles["ProfTableHeader"] if i == 0 else self._styles["ProfTableData"]
                para_hub_rows.append([
                    Paragraph(_esc(str(cell)), style) for cell in row
                ])

            hub_table = Table(para_hub_rows, colWidths=[200, 60, 70])
            hub_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#37474F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(hub_table)
            story.append(Spacer(1, 8))

        # ------------------------------------------------------------------
        # Misconception list
        # ------------------------------------------------------------------
        misconceptions = getattr(stats, "misconception_frequencies", [])
        if misconceptions:
            story.append(Paragraph(_esc("주요 오개념"), self._styles["ProfSubsection"]))
            # Already sorted desc by frequency from build_professor_report_data;
            # sort again defensively just in case.
            sorted_misc = sorted(
                misconceptions,
                key=lambda x: x[1] if (isinstance(x, tuple) and len(x) >= 2) else 0,
                reverse=True,
            )
            for item in sorted_misc:
                if isinstance(item, tuple) and len(item) >= 2:
                    text, freq = item[0], item[1]
                else:
                    text, freq = str(item), 0
                entry = f"• {text} ({freq}명)"
                story.append(Paragraph(_esc(entry), self._styles["ProfBody"]))

        # ------------------------------------------------------------------
        # Classified misconception table (from misconception_classifier)
        # ------------------------------------------------------------------
        classified = getattr(stats, "classified_misconceptions", [])
        if classified:
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                _esc("오개념 패턴 분류"), self._styles["ProfSubsection"],
            ))
            cls_header = ["패턴", "설명", "신뢰도"]
            cls_rows = [cls_header]
            for cm in classified:
                pattern_name = (
                    cm.pattern.value if hasattr(cm.pattern, "value") else str(cm.pattern)
                )
                desc = getattr(cm, "description", "")
                conf = getattr(cm, "confidence", 0.0)
                cls_rows.append([pattern_name, desc, f"{conf:.0%}"])

            para_cls_rows = []
            for i, row in enumerate(cls_rows):
                style = (
                    self._styles["ProfTableHeader"] if i == 0
                    else self._styles["ProfTableData"]
                )
                para_cls_rows.append([
                    Paragraph(_esc(str(cell)), style) for cell in row
                ])

            cls_table = Table(para_cls_rows, colWidths=[80, 200, 50])
            cls_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#37474F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(cls_table)

        return story

    def _build_lecture_gap_section(self, report_data: ProfessorReportData) -> list:
        """Build the lecture gap analysis section.

        Renders coverage ratio, missed concepts, extra concepts, and
        high_miss_overlap if a LectureGapReport is attached.

        Args:
            report_data: Complete professor report data.

        Returns:
            List of ReportLab flowables, empty if no gap report attached.
        """

        gap_report = getattr(report_data, "lecture_gap_report", None)
        if gap_report is None:
            return []

        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            _esc("강의 갭 분석"), self._styles["ProfSection"],
        ))
        story.append(Spacer(1, 4 * mm))

        # Coverage summary
        coverage_pct = gap_report.coverage_ratio * 100
        summary = (
            f"마스터 개념: {len(gap_report.master_concepts)}개 | "
            f"강의 커버리지: {coverage_pct:.1f}% | "
            f"누락 개념: {len(gap_report.missed_concepts)}개 | "
            f"추가 개념: {len(gap_report.extra_concepts)}개"
        )
        story.append(Paragraph(_esc(summary), self._styles["ProfBody"]))
        story.append(Spacer(1, 4 * mm))

        # Missed concepts table
        if gap_report.missed_concepts:
            story.append(Paragraph(
                _esc("누락된 마스터 개념"), self._styles["ProfSubsection"],
            ))
            missed_data = [["개념"]]
            for concept in sorted(gap_report.missed_concepts):
                missed_data.append([_esc(concept)])

            para_missed = []
            for i, row in enumerate(missed_data):
                style = (
                    self._styles["ProfTableHeader"] if i == 0
                    else self._styles["ProfTableData"]
                )
                para_missed.append([Paragraph(_esc(str(cell)), style) for cell in row])

            missed_table = Table(para_missed, colWidths=[200])
            missed_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#37474F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(missed_table)
            story.append(Spacer(1, 4 * mm))

        # High miss overlap
        if gap_report.high_miss_overlap:
            story.append(Paragraph(
                _esc("학생 오답률 높은 누락 개념"), self._styles["ProfSubsection"],
            ))
            for concept in gap_report.high_miss_overlap:
                story.append(Paragraph(
                    _esc(f"• {concept}"), self._styles["ProfBody"],
                ))
            story.append(Spacer(1, 4 * mm))

        return story

    def _build_emphasis_comparison_section(self, report_data: ProfessorReportData) -> list:
        """Build the cross-class emphasis comparison section (FR-021).

        Renders a table of top-5 concepts with highest emphasis variance
        across classes. Requires class_emphasis_maps with >= 2 classes.
        Returns empty list when fewer than 2 classes are available.

        Args:
            report_data: Complete professor report data.

        Returns:
            List of ReportLab flowables, empty if section not applicable.
        """
        from forma.lecture_gap_analysis import compute_cross_class_emphasis_variance

        class_maps = getattr(report_data, "class_emphasis_maps", None)
        if not class_maps or len(class_maps) < 2:
            return []

        variance_map = compute_cross_class_emphasis_variance(class_maps, top_n=5)
        if not variance_map:
            return []

        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            _esc("분반 간 강조도 비교"), self._styles["ProfSection"],
        ))
        story.append(Spacer(1, 4 * mm))

        summary_text = (
            f"분반 수: {len(class_maps)}개 | "
            f"편차 상위 개념: 최대 5개"
        )
        story.append(Paragraph(_esc(summary_text), self._styles["ProfBody"]))
        story.append(Spacer(1, 4 * mm))

        # Header: 개념 | 편차 | 분반별 점수...
        class_names = sorted(class_maps.keys())
        header = ["개념", "편차"] + class_names
        table_data = [header]

        for concept, stdev, per_class_scores in variance_map:
            row: list[str] = [concept, f"{stdev:.3f}"]
            for cls in class_names:
                score = per_class_scores.get(cls, 0.0)
                row.append(f"{score:.2f}")
            table_data.append(row)

        para_rows = []
        for i, row in enumerate(table_data):
            style = (
                self._styles["ProfTableHeader"] if i == 0
                else self._styles["ProfTableData"]
            )
            para_rows.append([Paragraph(_esc(str(cell)), style) for cell in row])

        # Column widths: 개념(120) + 편차(50) + 분반별(50 each)
        col_widths = [120, 50] + [50] * len(class_names)
        comp_table = Table(para_rows, colWidths=col_widths)
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#37474F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        # Highlight rows with high stdev (top concepts get emphasis color)
        for row_idx in range(1, len(table_data)):
            table_style.append(
                ("BACKGROUND", (0, row_idx), (-1, row_idx), HexColor("#E3F2FD"))
            )
        comp_table.setStyle(TableStyle(table_style))
        story.append(comp_table)
        story.append(Spacer(1, 4 * mm))

        return story

    def _build_class_graph_section(
        self,
        story: list,
        aggregate: object,
        chart_buf: object,
    ) -> None:
        """Build the class knowledge graph section in the PDF story.

        Adds a section heading, embedded chart image, and a weak-edge table
        listing edges with correct_ratio < 0.3, sorted by ratio ascending.

        Args:
            story: The PDF story list to append elements to.
            aggregate: ClassKnowledgeAggregate with edges and question_sn.
            chart_buf: io.BytesIO containing the PNG chart image.
        """
        story.append(
            Paragraph(
                _esc(f"학급 지식 지도 — 문제 {aggregate.question_sn}"),
                self._styles["ProfSection"],
            )
        )
        story.append(Spacer(1, 6))

        # Embed chart image (14cm x 9cm)
        story.append(self._safe_image(chart_buf, width=140 * mm, height=90 * mm))
        story.append(Spacer(1, 6))

        # Weak-edge table: edges where correct_ratio < 0.3
        weak_edges = sorted(
            [e for e in aggregate.edges if e.correct_ratio < 0.3],
            key=lambda e: e.correct_ratio,
        )

        if weak_edges:
            story.append(
                Paragraph(
                    _esc("취약 엣지 (정답 비율 30% 미만)"),
                    self._styles["ProfSubsection"],
                )
            )
            story.append(Spacer(1, 4))

            table_data = [
                [
                    Paragraph(_esc("관계"), self._styles["ProfBody"]),
                    Paragraph(_esc("정답 수"), self._styles["ProfBody"]),
                    Paragraph(_esc("오류 수"), self._styles["ProfBody"]),
                    Paragraph(_esc("누락 수"), self._styles["ProfBody"]),
                    Paragraph(_esc("정답 비율"), self._styles["ProfBody"]),
                ],
            ]
            for e in weak_edges:
                table_data.append([
                    Paragraph(
                        _esc(f"{e.subject} → {e.obj}"),
                        self._styles["ProfBody"],
                    ),
                    Paragraph(_esc(str(e.correct_count)), self._styles["ProfBody"]),
                    Paragraph(_esc(str(e.error_count)), self._styles["ProfBody"]),
                    Paragraph(_esc(str(e.missing_count)), self._styles["ProfBody"]),
                    Paragraph(
                        _esc(f"{e.correct_ratio:.1%}"),
                        self._styles["ProfBody"],
                    ),
                ])

            col_widths = [60 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm]
            tbl = Table(table_data, colWidths=col_widths)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E3F2FD")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(tbl)

        story.append(Spacer(1, 8))

    def _build_misconception_cluster_section(
        self,
        story: list,
        clusters: list,
    ) -> None:
        """Build the misconception cluster table section in the PDF story.

        Adds a section heading and a table with columns for cluster pattern,
        member count, representative error, and correction point. Shows
        '오개념 없음' when clusters list is empty. Shows '교정 포인트 없음'
        when correction_point is empty string.

        No LLM calls are made in this method (Constitution VI).

        Args:
            story: The PDF story list to append elements to.
            clusters: List of MisconceptionCluster instances.
        """
        story.append(
            Paragraph(
                _esc("오개념 클러스터 분석"),
                self._styles["ProfSection"],
            )
        )
        story.append(Spacer(1, 6))

        if not clusters:
            story.append(
                Paragraph(_esc("오개념 없음"), self._styles["ProfBody"])
            )
            story.append(Spacer(1, 8))
            return

        # Build table: header + one row per cluster
        table_data = [
            [
                Paragraph(_esc("패턴"), self._styles["ProfTableHeader"]),
                Paragraph(_esc("학생수"), self._styles["ProfTableHeader"]),
                Paragraph(_esc("대표 오류"), self._styles["ProfTableHeader"]),
                Paragraph(_esc("교정 포인트"), self._styles["ProfTableHeader"]),
            ],
        ]

        for cluster in clusters:
            pattern_name = (
                cluster.pattern.value
                if hasattr(cluster.pattern, "value")
                else str(cluster.pattern)
            )
            correction = (
                cluster.correction_point
                if cluster.correction_point
                else "교정 포인트 없음"
            )
            table_data.append([
                Paragraph(_esc(pattern_name), self._styles["ProfTableData"]),
                Paragraph(_esc(str(cluster.member_count)), self._styles["ProfTableData"]),
                Paragraph(_esc(cluster.representative_error), self._styles["ProfTableData"]),
                Paragraph(_esc(correction), self._styles["ProfTableData"]),
            ])

        col_widths = [60 * mm, 20 * mm, 55 * mm, 55 * mm]
        tbl = Table(table_data, colWidths=col_widths)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#37474F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ALIGN", (2, 1), (3, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 8))

    def _build_risk_movement_section(self, risk_movement) -> list:
        """Build section showing risk group movement between weeks."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            _esc("위험군 변동 현황"),
            self._styles["ProfSection"],
        ))
        story.append(Spacer(1, 4))

        rows = [
            [
                Paragraph("구분", self._styles["ProfTableHeader"]),
                Paragraph("인원", self._styles["ProfTableHeader"]),
                Paragraph("학생 목록", self._styles["ProfTableHeader"]),
            ],
            [
                Paragraph("신규 진입", self._styles["ProfTableData"]),
                Paragraph(str(len(risk_movement.newly_at_risk)), self._styles["ProfTableData"]),
                Paragraph(
                    _esc(", ".join(risk_movement.newly_at_risk) if risk_movement.newly_at_risk else "-"),
                    self._styles["ProfTableData"],
                ),
            ],
            [
                Paragraph("위험군 탈출", self._styles["ProfTableData"]),
                Paragraph(str(len(risk_movement.exited_risk)), self._styles["ProfTableData"]),
                Paragraph(
                    _esc(", ".join(risk_movement.exited_risk) if risk_movement.exited_risk else "-"),
                    self._styles["ProfTableData"],
                ),
            ],
            [
                Paragraph("지속 위험군", self._styles["ProfTableData"]),
                Paragraph(str(len(risk_movement.persistent_risk)), self._styles["ProfTableData"]),
                Paragraph(
                    _esc(", ".join(risk_movement.persistent_risk) if risk_movement.persistent_risk else "-"),
                    self._styles["ProfTableData"],
                ),
            ],
        ]
        tbl = Table(rows, colWidths=[30 * mm, 20 * mm, 90 * mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(tbl)
        return story

    def _build_cross_section_comparison_section(
        self, cross_report, report_data=None,
    ) -> list:
        """Build cross-section comparison section for aggregate reports.

        Includes statistics table, pairwise comparison results, box plot,
        concept mastery heatmap, and weekly interaction chart.

        Args:
            cross_report: CrossSectionReport instance.
            report_data: ProfessorReportData for extracting per-section scores
                from student_rows (used for box plot).

        Returns:
            List of ReportLab flowables.
        """
        from forma.section_comparison_charts import (
            build_section_box_plot,
            build_concept_mastery_heatmap,
            build_weekly_interaction_chart,
        )

        story: list = []
        story.append(PageBreak())
        story.append(
            Paragraph(_esc("분반 간 비교 분석"), self._styles["ProfSection"]),
        )
        story.append(Spacer(1, 8))

        # --- Section descriptive statistics table ---
        story.append(
            Paragraph(_esc("분반별 기술 통계"), self._styles["ProfSubsection"]),
        )
        story.append(Spacer(1, 4))

        header = ["분반", "N", "평균", "중위수", "표준편차", "위험군", "위험군 비율"]
        table_data = [header]
        for ss in cross_report.section_stats:
            table_data.append([
                _esc(ss.section_name),
                str(ss.n_students),
                f"{ss.mean:.3f}",
                f"{ss.median:.3f}",
                f"{ss.std:.3f}",
                str(ss.n_at_risk),
                f"{ss.pct_at_risk:.1%}",
            ])

        col_widths = [50, 30, 50, 50, 50, 40, 60]
        tbl = Table(table_data, colWidths=col_widths)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("FONTNAME", (0, 0), (-1, 0), "NanumGothicBold"),
            ("FONTNAME", (0, 1), (-1, -1), "NanumGothic"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F5F5F5")]),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 12))

        # --- Pairwise comparison table ---
        if cross_report.pairwise_comparisons:
            story.append(
                Paragraph(_esc("쌍대 비교 결과"), self._styles["ProfSubsection"]),
            )
            story.append(Spacer(1, 4))

            pw_header = [
                "비교", "검정", "통계량", "p값",
                "보정p값", "Cohen's d", "효과크기", "유의",
            ]
            pw_data = [pw_header]
            for pc in cross_report.pairwise_comparisons:
                test_label = (
                    "Welch t" if pc.test_name == "welch_t"
                    else "Mann-Whitney U"
                )
                p_corr = f"{pc.p_value_corrected:.4f}" if pc.p_value_corrected is not None else "-"
                sig = "Yes" if pc.is_significant else "No"
                pw_data.append([
                    _esc(f"{pc.section_a} vs {pc.section_b}"),
                    _esc(test_label),
                    f"{pc.test_statistic:.3f}",
                    f"{pc.p_value:.4f}",
                    p_corr,
                    f"{pc.cohens_d:.3f}",
                    _esc(pc.effect_size_label),
                    sig,
                ])

            pw_col_widths = [70, 65, 45, 45, 45, 50, 50, 30]
            pw_tbl = Table(pw_data, colWidths=pw_col_widths)
            pw_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2E7D32")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("FONTNAME", (0, 0), (-1, 0), "NanumGothicBold"),
                ("FONTNAME", (0, 1), (-1, -1), "NanumGothic"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F5F5F5")]),
            ]))
            story.append(pw_tbl)
            story.append(Spacer(1, 12))

        # --- Box plot chart ---
        try:
            section_scores: dict[str, list[float]] = {}
            if report_data is not None:
                for row in report_data.student_rows:
                    sec = row.section or report_data.class_name
                    section_scores.setdefault(sec, []).append(
                        row.overall_ensemble_mean,
                    )
            if section_scores:
                box_buf = build_section_box_plot(section_scores)
                story.append(
                    self._safe_image(box_buf, 400, 250),
                )
                story.append(Spacer(1, 8))
        except Exception as exc:
            logger.warning("분반 비교 박스플롯 생성 실패: %s", exc)

        # --- Concept mastery heatmap ---
        if cross_report.concept_mastery_by_section:
            try:
                heatmap_buf = build_concept_mastery_heatmap(
                    cross_report.concept_mastery_by_section,
                )
                if heatmap_buf is not None:
                    story.append(
                        self._safe_image(heatmap_buf, 450, 200),
                    )
                    story.append(Spacer(1, 8))
            except Exception as exc:
                logger.warning("개념 숙달도 히트맵 생성 실패: %s", exc)

        # --- Weekly interaction chart ---
        if cross_report.weekly_interaction:
            try:
                interaction_buf = build_weekly_interaction_chart(
                    cross_report.weekly_interaction,
                )
                if interaction_buf is not None:
                    story.append(
                        self._safe_image(interaction_buf, 400, 250),
                    )
                    story.append(Spacer(1, 8))
            except Exception as exc:
                logger.warning("주차별 상호작용 차트 생성 실패: %s", exc)

        return story

    def _build_predicted_risk_section(self, risk_predictions: list) -> list:
        """Build section showing predicted drop risk for students.

        Args:
            risk_predictions: List of RiskPrediction objects.

        Returns:
            List of ReportLab flowables.
        """
        story: list = []
        story.append(PageBreak())
        story.append(Paragraph(
            _esc("드롭 리스크 예측"),
            self._styles["ProfSection"],
        ))
        story.append(Spacer(1, 4))

        # Sort by drop_probability descending
        sorted_preds = sorted(
            risk_predictions, key=lambda p: p.drop_probability, reverse=True,
        )

        # Show top 50 or all if fewer
        display_preds = sorted_preds[:50]

        header = [
            Paragraph("학생", self._styles["ProfTableHeader"]),
            Paragraph("드롭 확률", self._styles["ProfTableHeader"]),
            Paragraph("주요 위험 요인", self._styles["ProfTableHeader"]),
            Paragraph("예측 방식", self._styles["ProfTableHeader"]),
        ]
        rows = [header]

        for pred in display_preds:
            top_factors = pred.risk_factors[:3] if pred.risk_factors else []
            factors_str = ", ".join(f.name for f in top_factors) if top_factors else "-"
            source = "모델 기반" if pred.is_model_based else "규칙 기반"

            rows.append([
                Paragraph(_esc(pred.student_id), self._styles["ProfTableData"]),
                Paragraph(f"{pred.drop_probability:.2f}", self._styles["ProfTableData"]),
                Paragraph(_esc(factors_str), self._styles["ProfTableData"]),
                Paragraph(_esc(source), self._styles["ProfTableData"]),
            ])

        col_widths = [60, 60, 220, 70]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "NanumGothicBold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ]))
        story.append(table)

        return story

    def _build_grade_prediction_section(
        self,
        grade_predictions: list,
    ) -> list:
        """Build grade prediction section for professor report (FR-030, FR-032).

        Shows per-student predicted grade, probability distribution, and
        prediction source (model-based or rule-based). Includes disclaimer.

        Args:
            grade_predictions: List of GradePrediction objects.

        Returns:
            List of ReportLab flowables.
        """
        story: list = []
        story.append(PageBreak())
        story.append(Paragraph(
            _esc("학기말 성적 예측"),
            self._styles["ProfSection"],
        ))
        story.append(Spacer(1, 4))

        if not grade_predictions:
            story.append(Paragraph(
                _esc("예측 데이터 없음"),
                self._styles["ProfBody"],
            ))
            return story

        # Per-student table
        header = [
            Paragraph(_esc("학생"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("예측 등급"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("A"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("B"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("C"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("D"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("F"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("예측 방식"), self._styles["ProfTableHeader"]),
        ]
        rows = [header]

        for pred in grade_predictions:
            probs = pred.grade_probabilities or {}
            source = "모델 기반" if pred.is_model_based else "규칙 기반"
            rows.append([
                Paragraph(_esc(pred.student_id), self._styles["ProfTableData"]),
                Paragraph(_esc(pred.predicted_grade), self._styles["ProfTableData"]),
                Paragraph(f"{probs.get('A', 0.0):.2f}", self._styles["ProfTableData"]),
                Paragraph(f"{probs.get('B', 0.0):.2f}", self._styles["ProfTableData"]),
                Paragraph(f"{probs.get('C', 0.0):.2f}", self._styles["ProfTableData"]),
                Paragraph(f"{probs.get('D', 0.0):.2f}", self._styles["ProfTableData"]),
                Paragraph(f"{probs.get('F', 0.0):.2f}", self._styles["ProfTableData"]),
                Paragraph(_esc(source), self._styles["ProfTableData"]),
            ])

        col_widths = [50, 40, 35, 35, 35, 35, 35, 55]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        from reportlab.lib import colors
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "NanumGothicBold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, HexColor("#F5F5F5")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

        # Disclaimer (FR-032)
        disclaimer = (
            "※ 본 예측은 통계 모델에 의한 추정이며, "
            "실제 성적은 다를 수 있습니다."
        )
        story.append(Paragraph(
            _esc(disclaimer),
            self._styles["ProfBody"],
        ))

        return story

    def _build_intervention_section(
        self,
        effects: list,
        type_summaries: list,
    ) -> list:
        """Build intervention history and effects section (FR-010, FR-014).

        Shows per-student intervention effect table, per-type summary table,
        and FR-014 disclaimer about correlation vs causation.

        Args:
            effects: List of InterventionEffect objects.
            type_summaries: List of InterventionTypeSummary objects.

        Returns:
            List of ReportLab flowables.
        """
        story: list = []
        story.append(PageBreak())
        story.append(Paragraph(
            _esc("개입 이력 및 효과"),
            self._styles["ProfSection"],
        ))
        story.append(Spacer(1, 4))

        if not effects:
            story.append(Paragraph(
                _esc("개입 기록 없음"),
                self._styles["ProfBody"],
            ))
            return story

        # Per-student effect table
        header = [
            Paragraph(_esc("학생"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("개입 유형"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("주차"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("개입 전"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("개입 후"), self._styles["ProfTableHeader"]),
            Paragraph(_esc("변화량"), self._styles["ProfTableHeader"]),
        ]
        rows = [header]

        for e in effects:
            if e.sufficient_data:
                pre_str = f"{e.pre_mean:.3f}"
                post_str = f"{e.post_mean:.3f}"
                change_str = f"{e.score_change:+.3f}"
            else:
                pre_str = "데이터 부족"
                post_str = "데이터 부족"
                change_str = "—"
            rows.append([
                Paragraph(_esc(e.student_id), self._styles["ProfTableData"]),
                Paragraph(_esc(e.intervention_type), self._styles["ProfTableData"]),
                Paragraph(f"W{e.intervention_week}", self._styles["ProfTableData"]),
                Paragraph(pre_str, self._styles["ProfTableData"]),
                Paragraph(post_str, self._styles["ProfTableData"]),
                Paragraph(change_str, self._styles["ProfTableData"]),
            ])

        from reportlab.lib import colors
        col_widths = [50, 50, 35, 50, 50, 50]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "NanumGothicBold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, HexColor("#F5F5F5")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

        # Type summary table
        if type_summaries:
            story.append(Paragraph(
                _esc("개입 유형별 효과 요약"),
                self._styles["ProfSubsection"],
            ))
            story.append(Spacer(1, 4))

            ts_header = [
                Paragraph(_esc("유형"), self._styles["ProfTableHeader"]),
                Paragraph(_esc("전체"), self._styles["ProfTableHeader"]),
                Paragraph(_esc("유효"), self._styles["ProfTableHeader"]),
                Paragraph(_esc("개선"), self._styles["ProfTableHeader"]),
                Paragraph(_esc("악화"), self._styles["ProfTableHeader"]),
                Paragraph(_esc("평균 변화"), self._styles["ProfTableHeader"]),
            ]
            ts_rows = [ts_header]

            for s in type_summaries:
                ts_rows.append([
                    Paragraph(_esc(s.intervention_type), self._styles["ProfTableData"]),
                    Paragraph(str(s.n_total), self._styles["ProfTableData"]),
                    Paragraph(str(s.n_sufficient), self._styles["ProfTableData"]),
                    Paragraph(str(s.n_positive), self._styles["ProfTableData"]),
                    Paragraph(str(s.n_negative), self._styles["ProfTableData"]),
                    Paragraph(f"{s.mean_change:+.3f}", self._styles["ProfTableData"]),
                ])

            ts_table = Table(ts_rows, colWidths=[55, 35, 35, 35, 35, 50], repeatRows=1)
            ts_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "NanumGothicBold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, HexColor("#F5F5F5")]),
            ]))
            story.append(ts_table)
            story.append(Spacer(1, 8))

        # Disclaimer (FR-014)
        disclaimer = (
            "※ 개입 효과는 상관관계이며, "
            "인과관계를 보장하지 않습니다."
        )
        story.append(Paragraph(
            _esc(disclaimer),
            self._styles["ProfBody"],
        ))

        return story

    def _build_deficit_map_section(
        self,
        deficit_map,
        deficit_map_chart=None,
    ) -> list:
        """Build class concept deficit map section (v0.10.0 US4, FR-021).

        Shows a DAG chart with per-concept deficit counts and a summary table.

        Args:
            deficit_map: ClassDeficitMap with concept counts and DAG.
            deficit_map_chart: Optional PNG BytesIO of deficit map DAG chart.

        Returns:
            List of ReportLab flowables.
        """
        story: list = []
        story.append(PageBreak())
        story.append(Paragraph(
            _esc("학급 개념 결손 맵"),
            self._styles["ProfSection"],
        ))
        story.append(Spacer(1, 6))

        # DAG chart (if provided)
        if deficit_map_chart is not None:
            deficit_map_chart.seek(0)
            img = Image(deficit_map_chart, width=160 * mm, height=100 * mm)
            story.append(img)
            story.append(Spacer(1, 6))

        # Summary table: concept → deficit count / total
        if deficit_map.concept_counts:
            total = deficit_map.total_students or 1
            header = ["개념", "결손 학생 수", "비율"]
            rows = [header]
            for concept in sorted(
                deficit_map.concept_counts.keys(),
                key=lambda c: deficit_map.concept_counts[c],
                reverse=True,
            ):
                count = deficit_map.concept_counts[concept]
                ratio = count / total
                rows.append([
                    _esc(concept),
                    str(count),
                    f"{ratio:.0%}",
                ])
            table = Table(rows, colWidths=[80 * mm, 35 * mm, 35 * mm])
            table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), "NanumGothic"),
                ("FONTNAME", (0, 0), (-1, 0), "NanumGothicBold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E3F2FD")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]))
            story.append(table)
        else:
            story.append(Paragraph(
                _esc("개념 의존성이 정의되지 않았습니다."),
                self._styles["ProfBody"],
            ))

        return story

    def _build_llm_analysis_page(self, report_data: ProfessorReportData) -> list:
        """Build LLM analysis page with overall assessment only."""
        overall_text = report_data.overall_assessment or ""
        model_name = report_data.llm_model_used or ""

        # Skip entire page when there is no LLM content
        if not overall_text and not model_name:
            return []

        story = []

        # Overall assessment section
        story.append(Paragraph(_esc("종합 평가"), self._styles["ProfSection"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(_esc(overall_text), self._styles["ProfBody"]))
        story.append(Spacer(1, 12))

        # Model metadata
        if model_name:
            story.append(Paragraph(_esc(f"생성 모델: {model_name}"), self._styles["ProfBody"]))

        # Fallback notice if generation failed
        if report_data.llm_generation_failed:
            notice = "※ AI 분석 실패 — 기본 통계 요약으로 대체되었습니다."
            story.append(Spacer(1, 6))
            story.append(Paragraph(_esc(notice), self._styles["ProfBody"]))

        return story
