"""PDF report generator for lecture transcript analysis results.

Generates A4 PDF reports from AnalysisResult data using ReportLab Platypus.
Each PDF contains sections for keywords, network visualization, topics,
concept coverage, and emphasis scores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    Image, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from forma.font_utils import esc, find_korean_font, register_korean_fonts
from forma.lecture_analyzer import AnalysisResult
from forma.lecture_comparison import ComparisonResult

logger = logging.getLogger(__name__)


class LectureReportGenerator:
    """Generate lecture analysis PDF reports using ReportLab Platypus.

    Args:
        font_path: Path to Korean .ttf font. Auto-detected if None.
    """

    def __init__(self, font_path: str | None = None) -> None:
        """Initialize the report generator and register fonts.

        Args:
            font_path: Path to Korean .ttf font file. If None,
                auto-detects using find_korean_font().
        """
        if font_path is None:
            font_path = find_korean_font()

        self._font_path = font_path
        register_korean_fonts(font_path)

        self._styles = getSampleStyleSheet()
        self._styles.add(ParagraphStyle(
            "LectTitle",
            parent=self._styles["Title"],
            fontName="NanumGothicBold",
            fontSize=18,
            spaceAfter=12,
        ))
        self._styles.add(ParagraphStyle(
            "LectSection",
            parent=self._styles["Heading2"],
            fontName="NanumGothicBold",
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
        ))
        self._styles.add(ParagraphStyle(
            "LectBody",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=10,
            leading=14,
            spaceAfter=4,
        ))
        self._styles.add(ParagraphStyle(
            "LectTableHeader",
            parent=self._styles["Normal"],
            fontName="NanumGothicBold",
            fontSize=9,
            textColor=HexColor("#FFFFFF"),
            alignment=1,
        ))
        self._styles.add(ParagraphStyle(
            "LectTableData",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=9,
            alignment=1,
        ))
        self._styles.add(ParagraphStyle(
            "LectSkipped",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=10,
            leading=14,
            spaceAfter=4,
            textColor=HexColor("#888888"),
        ))

    def generate_analysis_report(
        self, result: AnalysisResult, output_path: Path,
    ) -> None:
        """Build a PDF report from an AnalysisResult.

        Args:
            result: The analysis result to render.
            output_path: Path to write the PDF file.
        """
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )
        story: list[Any] = []
        story.extend(self._build_title_section(result))
        story.extend(self._build_keyword_section(result))
        story.extend(self._build_network_section(result))
        story.extend(self._build_topic_section(result))
        if result.concept_coverage:
            story.extend(self._build_coverage_section(result))
        if result.emphasis_scores:
            story.extend(self._build_emphasis_section(result))
        doc.build(story)

    def _build_title_section(self, result: AnalysisResult) -> list:
        """Build the title section with class/week info."""
        title = f"강의 분석 보고서 - {esc(result.class_id)}반 {result.week}주차"
        flowables: list[Any] = [
            Paragraph(title, self._styles["LectTitle"]),
            Spacer(1, 6 * mm),
            Paragraph(
                f"분석 시각: {esc(result.analysis_timestamp)}",
                self._styles["LectBody"],
            ),
            Paragraph(
                f"문장 수: {result.sentence_count}",
                self._styles["LectBody"],
            ),
            Spacer(1, 8 * mm),
        ]
        return flowables

    def _build_keyword_section(self, result: AnalysisResult) -> list:
        """Build keyword frequency table section."""
        flowables: list[Any] = [
            Paragraph("주요 키워드", self._styles["LectSection"]),
            Spacer(1, 4 * mm),
        ]

        if not result.keyword_frequencies:
            flowables.append(Paragraph(
                "키워드 추출 결과 없음", self._styles["LectSkipped"],
            ))
            return flowables

        # Build table with top keywords
        header = [
            Paragraph("순위", self._styles["LectTableHeader"]),
            Paragraph("키워드", self._styles["LectTableHeader"]),
            Paragraph("빈도", self._styles["LectTableHeader"]),
        ]
        rows = [header]
        for i, kw in enumerate(result.top_keywords[:20], 1):
            freq = result.keyword_frequencies.get(kw, 0)
            rows.append([
                Paragraph(str(i), self._styles["LectTableData"]),
                Paragraph(esc(kw), self._styles["LectTableData"]),
                Paragraph(str(freq), self._styles["LectTableData"]),
            ])

        table = Table(rows, colWidths=[30 * mm, 80 * mm, 30 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#F2F2F2")]),
        ]))
        flowables.append(table)
        flowables.append(Spacer(1, 8 * mm))
        return flowables

    def _build_network_section(self, result: AnalysisResult) -> list:
        """Build network image section."""
        flowables: list[Any] = [
            Paragraph("키워드 네트워크", self._styles["LectSection"]),
            Spacer(1, 4 * mm),
        ]

        if result.network_image_path and Path(result.network_image_path).exists():
            img = Image(
                str(result.network_image_path),
                width=160 * mm,
                height=120 * mm,
            )
            flowables.append(img)
        else:
            flowables.append(Paragraph(
                "네트워크 이미지 없음", self._styles["LectSkipped"],
            ))

        flowables.append(Spacer(1, 8 * mm))
        return flowables

    def _build_topic_section(self, result: AnalysisResult) -> list:
        """Build topic modeling section or skipped placeholder."""
        flowables: list[Any] = [
            Paragraph("토픽 분석", self._styles["LectSection"]),
            Spacer(1, 4 * mm),
        ]

        if result.topics is None:
            reason = result.topic_skipped_reason or "알 수 없는 이유"
            flowables.append(Paragraph(
                f"[건너뜀] {esc(reason)}", self._styles["LectSkipped"],
            ))
            flowables.append(Spacer(1, 8 * mm))
            return flowables

        # Topic table
        header = [
            Paragraph("토픽", self._styles["LectTableHeader"]),
            Paragraph("키워드", self._styles["LectTableHeader"]),
            Paragraph("문장 수", self._styles["LectTableHeader"]),
        ]
        rows = [header]
        for topic in result.topics:
            rows.append([
                Paragraph(str(topic.topic_id), self._styles["LectTableData"]),
                Paragraph(
                    esc(", ".join(topic.keywords[:5])),
                    self._styles["LectTableData"],
                ),
                Paragraph(str(topic.sentence_count), self._styles["LectTableData"]),
            ])

        table = Table(rows, colWidths=[25 * mm, 100 * mm, 30 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#F2F2F2")]),
        ]))
        flowables.append(table)
        flowables.append(Spacer(1, 8 * mm))
        return flowables

    def _build_coverage_section(self, result: AnalysisResult) -> list:
        """Build concept coverage section."""
        flowables: list[Any] = [
            Paragraph("개념 커버리지", self._styles["LectSection"]),
            Spacer(1, 4 * mm),
        ]

        cc = result.concept_coverage
        if cc is None:
            return flowables

        ratio_pct = f"{cc.coverage_ratio * 100:.1f}%"
        flowables.append(Paragraph(
            f"커버리지: {ratio_pct} ({len(cc.covered_concepts)}/{cc.total_concepts})",
            self._styles["LectBody"],
        ))

        if cc.covered_concepts:
            flowables.append(Paragraph(
                f"포함 개념: {esc(', '.join(cc.covered_concepts))}",
                self._styles["LectBody"],
            ))
        if cc.missed_concepts:
            flowables.append(Paragraph(
                f"누락 개념: {esc(', '.join(cc.missed_concepts))}",
                self._styles["LectBody"],
            ))

        flowables.append(Spacer(1, 8 * mm))
        return flowables

    def _build_emphasis_section(self, result: AnalysisResult) -> list:
        """Build emphasis scores section."""
        flowables: list[Any] = [
            Paragraph("강조도 분석", self._styles["LectSection"]),
            Spacer(1, 4 * mm),
        ]

        if not result.emphasis_scores:
            return flowables

        header = [
            Paragraph("개념", self._styles["LectTableHeader"]),
            Paragraph("강조도", self._styles["LectTableHeader"]),
        ]
        rows = [header]
        sorted_scores = sorted(
            result.emphasis_scores.items(), key=lambda x: x[1], reverse=True,
        )
        for concept, score in sorted_scores:
            rows.append([
                Paragraph(esc(concept), self._styles["LectTableData"]),
                Paragraph(f"{score:.3f}", self._styles["LectTableData"]),
            ])

        table = Table(rows, colWidths=[80 * mm, 40 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#F2F2F2")]),
        ]))
        flowables.append(table)
        flowables.append(Spacer(1, 8 * mm))
        return flowables

    # ------------------------------------------------------------------
    # Comparison report methods
    # ------------------------------------------------------------------

    def generate_comparison_report(
        self, comparison: ComparisonResult, output_path: Path,
    ) -> None:
        """Build a PDF comparison report from a ComparisonResult.

        Args:
            comparison: The comparison result to render.
            output_path: Path to write the PDF file.
        """
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )
        story: list[Any] = []

        # Title
        sections_label = ", ".join(comparison.sections_compared)
        title = f"반 간 비교 보고서 — {esc(sections_label)}"
        story.append(Paragraph(title, self._styles["LectTitle"]))
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph(
            f"비교 유형: {esc(comparison.comparison_type)}",
            self._styles["LectBody"],
        ))
        story.append(Paragraph(
            f"비교 시각: {esc(comparison.comparison_timestamp)}",
            self._styles["LectBody"],
        ))
        story.append(Spacer(1, 8 * mm))

        # Sections
        story.extend(self._build_exclusive_keywords_section(comparison))
        story.extend(self._build_concept_gap_section(comparison))
        story.extend(self._build_emphasis_variance_section(comparison))

        doc.build(story)

    def _build_exclusive_keywords_section(
        self, comparison: ComparisonResult,
    ) -> list:
        """Build table showing exclusive keywords per section.

        Args:
            comparison: ComparisonResult with exclusive_keywords data.

        Returns:
            List of ReportLab flowables.
        """
        flowables: list[Any] = [
            Paragraph("반별 고유 키워드", self._styles["LectSection"]),
            Spacer(1, 4 * mm),
        ]

        header = [
            Paragraph("반", self._styles["LectTableHeader"]),
            Paragraph("고유 키워드", self._styles["LectTableHeader"]),
        ]
        rows = [header]
        for section_id in sorted(comparison.exclusive_keywords.keys()):
            keywords = comparison.exclusive_keywords[section_id]
            kw_text = ", ".join(keywords) if keywords else "(없음)"
            rows.append([
                Paragraph(esc(section_id), self._styles["LectTableData"]),
                Paragraph(esc(kw_text), self._styles["LectTableData"]),
            ])

        table = Table(rows, colWidths=[30 * mm, 130 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, HexColor("#F2F2F2")]),
        ]))
        flowables.append(table)
        flowables.append(Spacer(1, 8 * mm))
        return flowables

    def _build_concept_gap_section(
        self, comparison: ComparisonResult,
    ) -> list:
        """Build matrix of covered/missed concepts per section.

        Args:
            comparison: ComparisonResult with concept_gaps data.

        Returns:
            List of ReportLab flowables.
        """
        flowables: list[Any] = [
            Paragraph("개념 누락 현황", self._styles["LectSection"]),
            Spacer(1, 4 * mm),
        ]

        if comparison.concept_gaps is None:
            flowables.append(Paragraph(
                "개념 목록 미제공 — 개념 누락 분석 생략",
                self._styles["LectSkipped"],
            ))
            flowables.append(Spacer(1, 8 * mm))
            return flowables

        header = [
            Paragraph("반", self._styles["LectTableHeader"]),
            Paragraph("누락 개념", self._styles["LectTableHeader"]),
        ]
        rows = [header]
        for section_id in sorted(comparison.concept_gaps.keys()):
            missed = comparison.concept_gaps[section_id]
            missed_text = ", ".join(missed) if missed else "(전체 포함)"
            rows.append([
                Paragraph(esc(section_id), self._styles["LectTableData"]),
                Paragraph(esc(missed_text), self._styles["LectTableData"]),
            ])

        table = Table(rows, colWidths=[30 * mm, 130 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, HexColor("#F2F2F2")]),
        ]))
        flowables.append(table)
        flowables.append(Spacer(1, 8 * mm))
        return flowables

    def _build_emphasis_variance_section(
        self, comparison: ComparisonResult,
    ) -> list:
        """Build table of top-N emphasis variance concepts with per-section scores.

        Args:
            comparison: ComparisonResult with emphasis_variance data.

        Returns:
            List of ReportLab flowables.
        """
        flowables: list[Any] = [
            Paragraph("강조도 편차 분석", self._styles["LectSection"]),
            Spacer(1, 4 * mm),
        ]

        if not comparison.emphasis_variance:
            flowables.append(Paragraph(
                "강조도 데이터 없음", self._styles["LectSkipped"],
            ))
            flowables.append(Spacer(1, 8 * mm))
            return flowables

        # Build header: 개념, 편차, section1, section2, ...
        sections = sorted(comparison.sections_compared)
        header_cells = [
            Paragraph("개념", self._styles["LectTableHeader"]),
            Paragraph("편차(σ)", self._styles["LectTableHeader"]),
        ]
        for sid in sections:
            header_cells.append(
                Paragraph(esc(sid), self._styles["LectTableHeader"]),
            )

        rows = [header_cells]
        for entry in comparison.emphasis_variance[:20]:
            row = [
                Paragraph(esc(entry.concept), self._styles["LectTableData"]),
                Paragraph(f"{entry.variance:.3f}", self._styles["LectTableData"]),
            ]
            for sid in sections:
                score = entry.per_section_scores.get(sid, 0.0)
                row.append(
                    Paragraph(f"{score:.3f}", self._styles["LectTableData"]),
                )
            rows.append(row)

        col_widths = [60 * mm, 30 * mm] + [30 * mm] * len(sections)
        # Adjust if too wide
        total = sum(col_widths)
        max_width = 170 * mm
        if total > max_width:
            scale = float(max_width) / float(total)
            col_widths = [w * scale for w in col_widths]

        table = Table(rows, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, HexColor("#F2F2F2")]),
        ]))
        flowables.append(table)
        flowables.append(Spacer(1, 8 * mm))
        return flowables
