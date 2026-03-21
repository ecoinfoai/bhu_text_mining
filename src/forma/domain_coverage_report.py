"""PDF report generation for domain delivery analysis (v2).

Generates a multi-page PDF report with up to 9 sections:
1. Concept delivery summary
2. Per-concept delivery detail table
3. Undelivered/partial concepts with action items
4. Intentionally skipped concepts
5. Keyword network comparison
6. Cross-section delivery comparison
7. Pedagogy analysis (separate from domain)
8. Instructor feedback summary (concept-level, NOT word frequency)
9. Formative assessment correlation (only if data available)

Section numbering is continuous with no gaps.
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

__all__ = ["DomainDeliveryPDFReportGenerator"]

_FALLBACK_PNG: bytes = minimal_png_bytes()

_STATUS_COLORS = {
    "충분히 설명": "#4CAF50",
    "부분 전달": "#FF9800",
    "미전달": "#F44336",
    "의도적 생략": "#9E9E9E",
}


class DomainDeliveryPDFReportGenerator:
    """Generate domain delivery PDF reports using ReportLab Platypus.

    Produces a 9-section (or 8-section if no assessment data) report
    with continuous section numbering and concept-level feedback.

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
        pedagogy: list | None = None,
        textbook_net: object | None = None,
        section_nets: list | None = None,
        missing_edges_map: dict | None = None,
        assessment_data: object | None = None,
        hierarchy: object | None = None,
        concept_network: object | None = None,
        deliveries_by_section: dict | None = None,
    ) -> str:
        """Generate the domain delivery report PDF.

        Args:
            result: DeliveryResult from domain_coverage_analyzer.
            output_path: Output path for the PDF file.
            course_name: Optional course name for header.
            pedagogy: Optional list of PedagogyAnalysis per section.
            textbook_net: Optional KeywordNetwork for textbook.
            section_nets: Optional list of KeywordNetwork per section.
            missing_edges_map: Optional {section: [(u,v),...]} missing edges.
            assessment_data: Optional assessment correlation data.
            hierarchy: Optional TopicHierarchy for hierarchical charts.
            concept_network: Optional ConceptNetwork for network graph.
            deliveries_by_section: Optional {section: [DeliveryAnalysis]}
                for per-section network comparison.

        Returns:
            Absolute path to generated PDF file.
        """
        if pedagogy is None:
            pedagogy = []
        if section_nets is None:
            section_nets = []
        if missing_edges_map is None:
            missing_edges_map = {}

        story: list = []
        section_num = 0

        # Cover page
        story.extend(self._build_cover_page(result, course_name))

        # Section 1: Delivery summary
        section_num += 1
        story.extend(self._build_delivery_summary(result, section_num))

        # Section 2: Concept detail table
        section_num += 1
        story.extend(self._build_concept_detail_table(result, section_num))

        # Section 3: Undelivered / partially delivered
        section_num += 1
        story.extend(self._build_undelivered_section(result, section_num))

        # Section 4: Intentionally skipped
        section_num += 1
        story.extend(self._build_skipped_section(result, section_num))

        # Section 5: Network comparison
        section_num += 1
        story.extend(self._build_network_section(
            textbook_net, section_nets, missing_edges_map, section_num,
        ))

        # Hierarchical overview (only when hierarchy is provided)
        if hierarchy is not None:
            section_num += 1
            story.extend(self._build_hierarchical_overview_section(
                result, hierarchy, section_num,
            ))

        # Concept network graph (only when concept_network provided)
        if concept_network is not None:
            section_num += 1
            story.extend(self._build_concept_network_section(
                concept_network, result, section_num,
                deliveries_by_section=deliveries_by_section,
            ))

        # Section N: Section comparison heatmap
        section_num += 1
        story.extend(self._build_section_comparison(result, section_num))

        # Section 7: Pedagogy analysis
        section_num += 1
        story.extend(self._build_pedagogy_section(pedagogy, section_num))

        # Section 8: Instructor feedback summary
        section_num += 1
        story.extend(self._build_feedback_summary(result, section_num))

        # Section 9: Assessment correlation (only if data exists)
        if assessment_data is not None:
            section_num += 1
            story.extend(self._build_assessment_section(
                assessment_data, section_num,
            ))

        os.makedirs(
            os.path.dirname(os.path.abspath(output_path)) or ".",
            exist_ok=True,
        )

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

    # ----------------------------------------------------------------
    # Cover page
    # ----------------------------------------------------------------

    def _build_cover_page(self, result: object, course_name: str) -> list:
        """Build cover page."""
        story = []
        story.append(Spacer(1, 40 * mm))
        story.append(Paragraph(
            _esc("교과서-강의 도메인 전달 분석 보고서"),
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
            f"실효 전달율: {result.effective_delivery_rate:.1%}",
            self._styles["DcBody"],
        ))

        story.append(PageBreak())
        return story

    # ----------------------------------------------------------------
    # Section 1: Delivery summary
    # ----------------------------------------------------------------

    def _build_delivery_summary(
        self, result: object, section_num: int,
    ) -> list:
        """Build concept delivery summary with bar chart."""
        story = []
        story.append(Paragraph(
            f"{section_num}. 개념 전달 요약",
            self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        # Count delivery states
        fully = 0
        partial = 0
        not_delivered = 0
        skipped = 0
        for d in result.deliveries:
            if d.delivery_status == "충분히 설명":
                fully += 1
            elif d.delivery_status == "부분 전달":
                partial += 1
            elif d.delivery_status == "미전달":
                not_delivered += 1
            elif d.delivery_status == "의도적 생략":
                skipped += 1

        # Get unique concept count
        concepts_set = {d.concept for d in result.deliveries}
        total_concepts = len(concepts_set)

        # Get unique section count
        sections_set = {d.section_id for d in result.deliveries}
        n_sections = len(sections_set)

        summary_lines = [
            f"분석 개념 수: {total_concepts}개",
            f"분석 분반 수: {n_sections}개",
            f"충분히 설명: {fully}건",
            f"부분 전달: {partial}건",
            f"미전달: {not_delivered}건",
            f"의도적 생략: {skipped}건",
            f"실효 전달율: {result.effective_delivery_rate:.1%}",
        ]
        for line in summary_lines:
            story.append(Paragraph(_esc(line), self._styles["DcBody"]))

        # Delivery bar chart
        try:
            from forma.domain_coverage_charts import build_delivery_bar_chart
            chart_buf = build_delivery_bar_chart(
                result, font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Spacer(1, 5 * mm))
            story.append(Image(chart_buf, width=160 * mm, height=100 * mm))
        except Exception as exc:
            logger.warning("전달율 바 차트 생성 실패: %s", exc)
            story.append(Image(
                io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm,
            ))

        story.append(Spacer(1, 5 * mm))
        return story

    # ----------------------------------------------------------------
    # Section 2: Concept detail table
    # ----------------------------------------------------------------

    def _build_concept_detail_table(
        self, result: object, section_num: int,
    ) -> list:
        """Build per-concept delivery detail table with colored rows."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            f"{section_num}. 개념별 전달 상세",
            self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        # Collect sections
        sections: set[str] = set()
        for d in result.deliveries:
            sections.add(d.section_id)
        sections_sorted = sorted(sections)

        # Build concept-keyed data: {concept: {section: delivery}}
        concept_map: dict[str, dict[str, object]] = {}
        for d in result.deliveries:
            if d.concept not in concept_map:
                concept_map[d.concept] = {}
            concept_map[d.concept][d.section_id] = d

        cell_style = self._styles["DcTableCell"]

        headers = ["개념", "상태", "품질"]
        headers.extend(f"{s}반" for s in sections_sorted)
        header_row = [
            Paragraph(f"<b>{_esc(h)}</b>", cell_style) for h in headers
        ]

        data = [header_row]
        row_colors = []

        for concept_name in sorted(concept_map.keys()):
            sec_data = concept_map[concept_name]
            # Pick representative status (worst across sections)
            statuses = [d.delivery_status for d in sec_data.values()]
            if "미전달" in statuses:
                rep_status = "미전달"
            elif "부분 전달" in statuses:
                rep_status = "부분 전달"
            elif "의도적 생략" in statuses:
                rep_status = "의도적 생략"
            else:
                rep_status = "충분히 설명"

            color = _STATUS_COLORS.get(rep_status, "#000000")
            status_label = (
                f'<font color="{color}"><b>{_esc(rep_status)}</b></font>'
            )

            # Average quality
            qualities = [d.delivery_quality for d in sec_data.values()]
            avg_q = sum(qualities) / len(qualities) if qualities else 0.0

            row = [
                Paragraph(_esc(concept_name), cell_style),
                Paragraph(status_label, cell_style),
                Paragraph(f"{avg_q:.2f}", cell_style),
            ]

            for section in sections_sorted:
                if section in sec_data:
                    d = sec_data[section]
                    q = d.delivery_quality
                    row.append(Paragraph(f"{q:.2f}", cell_style))
                else:
                    row.append(Paragraph("—", cell_style))

            data.append(row)
            row_colors.append(color)

        if len(data) > 1:
            col_widths = [50 * mm, 25 * mm, 18 * mm]
            remaining = (A4[0] - 30 * mm) - sum(col_widths)
            if sections_sorted:
                per_section = remaining / len(sections_sorted)
                col_widths.extend([per_section] * len(sections_sorted))

            table = Table(data, colWidths=col_widths, repeatRows=1)
            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E0E0E0")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
            table.setStyle(TableStyle(style_cmds))
            story.append(table)
        else:
            story.append(Paragraph(
                "전달 분석 데이터 없음", self._styles["DcBody"],
            ))

        story.append(Spacer(1, 5 * mm))
        return story

    # ----------------------------------------------------------------
    # Section 3: Undelivered / partially delivered
    # ----------------------------------------------------------------

    def _build_undelivered_section(
        self, result: object, section_num: int,
    ) -> list:
        """Build undelivered + partially delivered concepts with actions."""
        story = []
        story.append(Paragraph(
            f"{section_num}. 미전달/부분전달 개념",
            self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        # Collect undelivered and partial per concept
        concept_issues: dict[str, list[object]] = {}
        for d in result.deliveries:
            if d.delivery_status in ("미전달", "부분 전달"):
                if d.concept not in concept_issues:
                    concept_issues[d.concept] = []
                concept_issues[d.concept].append(d)

        if not concept_issues:
            story.append(Paragraph(
                "미전달 또는 부분전달 개념이 없습니다.",
                self._styles["DcBody"],
            ))
            story.append(Spacer(1, 5 * mm))
            return story

        for concept_name, issues in sorted(concept_issues.items()):
            statuses = [d.delivery_status for d in issues]
            has_not = "미전달" in statuses

            if has_not:
                color = "#F44336"
                label = "미전달"
            else:
                color = "#FF9800"
                label = "부분 전달"

            text = (
                f'<font color="{color}"><b>{_esc(concept_name)}</b></font>'
                f" [{_esc(label)}]"
            )
            story.append(Paragraph(text, self._styles["DcBody"]))

            # Show evidence/depth from deliveries
            for d in issues:
                detail = f"  {d.section_id}반: "
                if d.depth:
                    detail += _esc(d.depth)
                elif d.evidence:
                    detail += _esc(d.evidence[:100])
                else:
                    detail += "상세 정보 없음"
                story.append(Paragraph(detail, self._styles["DcSmall"]))

            # Action item
            if has_not:
                action = "보강 권장: 해당 개념의 메커니즘과 핵심 내용을 수업에 포함"
            else:
                action = "보강 권장: 용어 수준을 넘어 메커니즘과 과정 설명 보완"
            story.append(Paragraph(
                f'  <font color="#1565C0">{_esc(action)}</font>',
                self._styles["DcSmall"],
            ))
            story.append(Spacer(1, 2 * mm))

        story.append(Spacer(1, 5 * mm))
        return story

    # ----------------------------------------------------------------
    # Section 4: Intentionally skipped
    # ----------------------------------------------------------------

    def _build_skipped_section(
        self, result: object, section_num: int,
    ) -> list:
        """Build intentionally skipped concepts list (gray, reference)."""
        story = []
        story.append(Paragraph(
            f"{section_num}. 의도적 생략 개념",
            self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        skipped_concepts: set[str] = set()
        for d in result.deliveries:
            if d.delivery_status == "의도적 생략":
                skipped_concepts.add(d.concept)

        if not skipped_concepts:
            story.append(Paragraph(
                "의도적 생략 개념이 없습니다.",
                self._styles["DcBody"],
            ))
        else:
            story.append(Paragraph(
                "아래 개념은 해당 주차 수업 범위에 포함되지 않아 생략되었습니다 (참고용).",
                self._styles["DcSmall"],
            ))
            for name in sorted(skipped_concepts):
                text = f'<font color="#9E9E9E">{_esc(name)}</font>'
                story.append(Paragraph(text, self._styles["DcBody"]))

        story.append(Spacer(1, 5 * mm))
        return story

    # ----------------------------------------------------------------
    # Section 5: Network comparison
    # ----------------------------------------------------------------

    def _build_network_section(
        self,
        textbook_net: object | None,
        section_nets: list,
        missing_edges_map: dict,
        section_num: int,
    ) -> list:
        """Build network comparison charts (textbook + sections)."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            f"{section_num}. 핵심 용어 네트워크 비교",
            self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        if textbook_net is None or not section_nets:
            story.append(Paragraph(
                "네트워크 비교 데이터가 제공되지 않았습니다.",
                self._styles["DcBody"],
            ))
            story.append(Spacer(1, 5 * mm))
            return story

        try:
            from forma.domain_coverage_charts import (
                build_network_comparison_chart,
            )
            for lecture_net in section_nets:
                missing = missing_edges_map.get(lecture_net.source, [])
                chart_buf = build_network_comparison_chart(
                    textbook_net, lecture_net, missing,
                    font_path=self._font_path, dpi=self._dpi,
                )
                story.append(Image(
                    chart_buf, width=170 * mm, height=75 * mm,
                ))
                # Text summary: missing edges for this section
                if missing:
                    edges_str = ", ".join(
                        f"{_esc(u)}-{_esc(v)}" for u, v in missing[:5]
                    )
                    suffix = (
                        f" 외 {len(missing) - 5}개"
                        if len(missing) > 5 else ""
                    )
                    story.append(Paragraph(
                        f"  {_esc(lecture_net.source)}반 누락 연결: "
                        f"{edges_str}{suffix}",
                        self._styles["DcSmall"],
                    ))
                else:
                    story.append(Paragraph(
                        f"  {_esc(lecture_net.source)}반: "
                        "교과서 네트워크 연결 모두 포함",
                        self._styles["DcSmall"],
                    ))
                story.append(Spacer(1, 3 * mm))
        except Exception as exc:
            logger.warning("네트워크 비교 차트 생성 실패: %s", exc)
            story.append(Image(
                io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm,
            ))

        story.append(Spacer(1, 5 * mm))
        return story

    # ----------------------------------------------------------------
    # Section 6: Section comparison (delivery heatmap)
    # ----------------------------------------------------------------

    def _build_section_comparison(
        self, result: object, section_num: int,
    ) -> list:
        """Build cross-section delivery comparison with heatmap."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            f"{section_num}. 분반 간 전달 비교",
            self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        try:
            from forma.domain_coverage_charts import build_delivery_heatmap
            chart_buf = build_delivery_heatmap(
                result, font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Image(chart_buf, width=160 * mm, height=120 * mm))
        except Exception as exc:
            logger.warning("전달 히트맵 생성 실패: %s", exc)
            story.append(Image(
                io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm,
            ))

        # Text summary: highest/lowest quality concepts
        non_skipped = [
            d for d in result.deliveries
            if d.delivery_status != "의도적 생략"
        ]
        if non_skipped:
            concept_avg: dict[str, list[float]] = {}
            for d in non_skipped:
                if d.concept not in concept_avg:
                    concept_avg[d.concept] = []
                concept_avg[d.concept].append(d.delivery_quality)

            avg_scores = {
                c: sum(vs) / len(vs) for c, vs in concept_avg.items()
            }
            if avg_scores:
                best = max(avg_scores, key=avg_scores.get)
                worst = min(avg_scores, key=avg_scores.get)
                story.append(Spacer(1, 2 * mm))
                story.append(Paragraph(
                    f"  최고 전달: {_esc(best)} "
                    f"(평균 {avg_scores[best]:.2f}), "
                    f"최저 전달: {_esc(worst)} "
                    f"(평균 {avg_scores[worst]:.2f})",
                    self._styles["DcSmall"],
                ))

        # Section comparison table (T046)
        section_comparisons = getattr(result, "_section_comparisons", None)
        if section_comparisons:
            story.append(Spacer(1, 5 * mm))
            story.extend(
                self._build_section_comparison_table(
                    section_comparisons, section_num,
                ),
            )

        # Per-section rate differences
        if result.per_section_rate:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(
                "<b>분반별 전달율:</b>",
                self._styles["DcBody"],
            ))
            for section, rate in sorted(result.per_section_rate.items()):
                story.append(Paragraph(
                    f"  {_esc(section)}반: {rate:.1%}",
                    self._styles["DcBody"],
                ))

        story.append(Spacer(1, 5 * mm))
        return story

    # ----------------------------------------------------------------
    # Section comparison table (T046)
    # ----------------------------------------------------------------

    def _build_section_comparison_table(
        self,
        comparisons: list,
        section_num: int,
    ) -> list:
        """Build statistical comparison table between sections.

        Columns: 분반 A, 분반 B, 평균 A, 평균 B, 검정, p-value, 보정 p, 유의

        Args:
            comparisons: List of DeliverySectionComparison.
            section_num: Current section number.

        Returns:
            List of ReportLab flowables.
        """
        story: list = []
        story.append(Paragraph(
            "<b>분반 간 통계 비교:</b>",
            self._styles["DcBody"],
        ))
        story.append(Spacer(1, 2 * mm))

        cell_style = self._styles["DcTableCell"]
        headers = [
            "분반 A", "분반 B", "평균 A", "평균 B",
            "검정", "p-value", "보정 p", "유의",
        ]
        header_row = [
            Paragraph(f"<b>{_esc(h)}</b>", cell_style) for h in headers
        ]

        data = [header_row]
        for comp in comparisons:
            sig_text = "유의" if comp.significant else "-"
            sig_color = "#F44336" if comp.significant else "#666666"
            row = [
                Paragraph(f"{_esc(comp.section_a)}반", cell_style),
                Paragraph(f"{_esc(comp.section_b)}반", cell_style),
                Paragraph(f"{comp.mean_a:.3f}", cell_style),
                Paragraph(f"{comp.mean_b:.3f}", cell_style),
                Paragraph(_esc(comp.test_name), cell_style),
                Paragraph(f"{comp.p_value:.4f}", cell_style),
                Paragraph(f"{comp.corrected_p_value:.4f}", cell_style),
                Paragraph(
                    f'<font color="{sig_color}">{_esc(sig_text)}</font>',
                    cell_style,
                ),
            ]
            data.append(row)

        if len(data) > 1:
            col_widths = [20 * mm, 20 * mm, 20 * mm, 20 * mm,
                          25 * mm, 22 * mm, 22 * mm, 18 * mm]
            table = Table(data, colWidths=col_widths, repeatRows=1)
            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E0E0E0")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
            table.setStyle(TableStyle(style_cmds))
            story.append(table)

        return story

    # ----------------------------------------------------------------
    # Section 7: Pedagogy analysis (separate from domain)
    # ----------------------------------------------------------------

    def _build_pedagogy_section(
        self, pedagogy: list, section_num: int,
    ) -> list:
        """Build pedagogy analysis section (habitual + effective patterns)."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            f"{section_num}. 교수법 참고",
            self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        if not pedagogy:
            story.append(Paragraph(
                "교수법 분석 데이터가 제공되지 않았습니다.",
                self._styles["DcBody"],
            ))
            story.append(Spacer(1, 5 * mm))
            return story

        for pa in pedagogy:
            story.append(Paragraph(
                f"<b>{_esc(pa.section_id)}반</b>",
                self._styles["DcBody"],
            ))
            story.append(Spacer(1, 2 * mm))

            # Habitual expressions (TOP 5)
            if pa.habitual_expressions:
                story.append(Paragraph(
                    "<b>습관적 표현 (TOP 5):</b>",
                    self._styles["DcBody"],
                ))
                for he in pa.habitual_expressions[:5]:
                    rec_color = (
                        "#F44336" if he.recommendation == "사용 자제 권장"
                        else "#4CAF50"
                    )
                    text = (
                        f'  "{_esc(he.expression)}" — '
                        f"총 {he.total_count}회"
                        f' <font color="{rec_color}">'
                        f"({_esc(he.recommendation)})</font>"
                    )
                    story.append(Paragraph(text, self._styles["DcSmall"]))

            # Effective patterns
            if pa.effective_patterns:
                story.append(Spacer(1, 2 * mm))
                story.append(Paragraph(
                    "<b>효과적 교수법 패턴:</b>",
                    self._styles["DcBody"],
                ))
                for ep in pa.effective_patterns:
                    text = (
                        f'  {_esc(ep.pattern_type)}: '
                        f"{ep.count}회"
                    )
                    story.append(Paragraph(text, self._styles["DcSmall"]))
                    for ex in ep.examples[:3]:
                        story.append(Paragraph(
                            f'    예: "{_esc(ex[:80])}"',
                            self._styles["DcSmall"],
                        ))

            story.append(Spacer(1, 3 * mm))

        story.append(Spacer(1, 5 * mm))
        return story

    # ----------------------------------------------------------------
    # Section 8: Instructor feedback summary (concept-level)
    # ----------------------------------------------------------------

    def _build_feedback_summary(
        self, result: object, section_num: int,
    ) -> list:
        """Build concept-level instructor feedback (NOT word frequency)."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            f"{section_num}. 교수자 피드백 요약",
            self._styles["DcSection"],
        ))
        story.append(Spacer(1, 5 * mm))

        # Undelivered concepts needing supplementary instruction
        not_delivered: dict[str, list[str]] = {}
        partial: dict[str, list[str]] = {}

        for d in result.deliveries:
            if d.delivery_status == "미전달":
                if d.concept not in not_delivered:
                    not_delivered[d.concept] = []
                not_delivered[d.concept].append(d.section_id)
            elif d.delivery_status == "부분 전달":
                if d.concept not in partial:
                    partial[d.concept] = []
                partial[d.concept].append(d.section_id)

        if not_delivered:
            story.append(Paragraph(
                "<b>보충 지도가 필요한 개념:</b>",
                self._styles["DcBody"],
            ))
            for concept_name, secs in sorted(not_delivered.items()):
                secs_str = ", ".join(f"{s}반" for s in sorted(secs))
                text = (
                    f'  - <font color="#F44336"><b>'
                    f"{_esc(concept_name)}</b></font>"
                    f" ({_esc(secs_str)})"
                )
                story.append(Paragraph(text, self._styles["DcBody"]))
            story.append(Spacer(1, 3 * mm))

        if partial:
            story.append(Paragraph(
                "<b>설명 심화가 필요한 개념:</b>",
                self._styles["DcBody"],
            ))
            for concept_name, secs in sorted(partial.items()):
                secs_str = ", ".join(f"{s}반" for s in sorted(secs))
                text = (
                    f'  - <font color="#FF9800"><b>'
                    f"{_esc(concept_name)}</b></font>"
                    f" ({_esc(secs_str)})"
                )
                story.append(Paragraph(text, self._styles["DcBody"]))
            story.append(Spacer(1, 3 * mm))

        if not not_delivered and not partial:
            story.append(Paragraph(
                "모든 수업 범위 내 개념이 충분히 전달되었습니다.",
                self._styles["DcBody"],
            ))
            story.append(Spacer(1, 3 * mm))

        # Section uniformity feedback
        section_rates = result.per_section_rate
        if len(section_rates) >= 2:
            rates = list(section_rates.values())
            max_r = max(rates)
            min_r = min(rates)
            gap = max_r - min_r
            if gap > 0.15:
                story.append(Paragraph(
                    "<b>분반 균일성 이슈:</b>",
                    self._styles["DcBody"],
                ))
                best = max(section_rates, key=section_rates.get)
                worst = min(section_rates, key=section_rates.get)
                story.append(Paragraph(
                    f"  {_esc(best)}반({max_r:.1%})과 "
                    f"{_esc(worst)}반({min_r:.1%}) 간 "
                    f"전달율 차이가 {gap:.1%}p입니다. "
                    "전달 방식 재검토를 권장합니다.",
                    self._styles["DcSmall"],
                ))
                story.append(Spacer(1, 3 * mm))

        # Overall assessment
        rate = result.effective_delivery_rate
        if rate >= 0.9:
            assessment = (
                "우수: 수업 범위 내 개념의 90% 이상이 "
                "효과적으로 전달되고 있습니다."
            )
        elif rate >= 0.7:
            assessment = (
                "양호: 수업 범위 내 개념의 70% 이상이 전달되고 있으나, "
                "일부 개념에 대한 설명 보완이 필요합니다."
            )
        elif rate >= 0.5:
            assessment = (
                "주의: 수업 범위 내 개념의 절반 이상이 충분히 전달되지 "
                "않았습니다. 수업 내용 재구성을 권장합니다."
            )
        else:
            assessment = (
                "경고: 수업 범위 내 개념의 절반 미만이 전달되었습니다. "
                "강의 내용과 교과서 범위의 불일치를 점검하시기 바랍니다."
            )

        story.append(Paragraph(
            f"<b>종합 평가:</b> {_esc(assessment)}",
            self._styles["DcBody"],
        ))

        story.append(Spacer(1, 5 * mm))
        return story

    # ----------------------------------------------------------------
    # Hierarchical overview (T040)
    # ----------------------------------------------------------------

    def _build_hierarchical_overview_section(
        self,
        result: object,
        hierarchy: object,
        section_num: int,
    ) -> list:
        """Build hierarchical overview section with 3 charts.

        Includes topic delivery stacked bar, grouped coverage bar,
        and grouped quality heatmap.

        Args:
            result: DeliveryResult.
            hierarchy: TopicHierarchy from summary file.
            section_num: Current section number.

        Returns:
            List of ReportLab flowables.
        """
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            f"{section_num}. 대주제별 계층 분석",
            self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        try:
            from forma.domain_coverage_charts import (
                build_grouped_quality_heatmap,
                build_hierarchical_coverage_chart,
                build_topic_delivery_stacked_chart,
            )

            # Chart 1: Topic delivery stacked bar
            chart1 = build_topic_delivery_stacked_chart(
                result, hierarchy,
                font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Image(chart1, width=160 * mm, height=80 * mm))
            story.append(Spacer(1, 5 * mm))

            # Chart 2: Hierarchical coverage grouped bar
            chart2 = build_hierarchical_coverage_chart(
                result, hierarchy,
                font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Image(chart2, width=160 * mm, height=80 * mm))
            story.append(Spacer(1, 5 * mm))

            # Chart 3: Grouped quality heatmap
            chart3 = build_grouped_quality_heatmap(
                result, hierarchy,
                font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Image(chart3, width=160 * mm, height=100 * mm))
        except Exception as exc:
            logger.warning("계층 분석 차트 생성 실패: %s", exc)
            story.append(Image(
                io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm,
            ))

        story.append(Spacer(1, 5 * mm))
        return story

    # ----------------------------------------------------------------
    # Section 9: Assessment correlation (optional)
    # ----------------------------------------------------------------

    def _build_assessment_section(
        self, assessment_data: object, section_num: int,
    ) -> list:
        """Build formative assessment correlation section.

        Only included when assessment_data is not None.
        """
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            f"{section_num}. 형성평가 연결 분석",
            self._styles["DcSection"],
        ))
        story.append(Spacer(1, 5 * mm))

        # Correlation coefficient
        corr = getattr(assessment_data, "correlation", None)
        if corr is not None:
            story.append(Paragraph(
                f"전달 품질-습득률 상관계수: Spearman ρ = {corr:.3f}",
                self._styles["DcBody"],
            ))
            story.append(Spacer(1, 3 * mm))

        # Well-explained but poorly understood
        well_poor = getattr(assessment_data, "well_explained_poor", [])
        if well_poor:
            story.append(Paragraph(
                "<b>잘 설명했으나 습득률이 낮은 개념:</b>",
                self._styles["DcBody"],
            ))
            for item in well_poor:
                name = item if isinstance(item, str) else getattr(
                    item, "concept", str(item),
                )
                story.append(Paragraph(
                    f"  - {_esc(name)}",
                    self._styles["DcBody"],
                ))
            story.append(Spacer(1, 3 * mm))

        # Under-explained and poorly understood
        under_poor = getattr(assessment_data, "under_explained_poor", [])
        if under_poor:
            story.append(Paragraph(
                "<b>설명 부족하고 습득률도 낮은 개념:</b>",
                self._styles["DcBody"],
            ))
            for item in under_poor:
                name = item if isinstance(item, str) else getattr(
                    item, "concept", str(item),
                )
                story.append(Paragraph(
                    f"  - {_esc(name)}",
                    self._styles["DcBody"],
                ))
            story.append(Spacer(1, 3 * mm))

        if not well_poor and not under_poor and corr is None:
            story.append(Paragraph(
                "형성평가 연결 분석 상세 데이터가 제공되지 않았습니다.",
                self._styles["DcBody"],
            ))

        story.append(Spacer(1, 5 * mm))
        return story


    # ----------------------------------------------------------------
    # Concept network graph section (T056)
    # ----------------------------------------------------------------

    def _build_concept_network_section(
        self,
        network: object,
        result: object,
        section_num: int,
        deliveries_by_section: dict | None = None,
    ) -> list:
        """Build concept network graph section with charts and summary.

        Args:
            network: ConceptNetwork with nodes and edges.
            result: DeliveryResult for delivery overlay.
            section_num: Current section number.
            deliveries_by_section: Optional per-section delivery data
                for comparison chart.

        Returns:
            List of ReportLab flowables.
        """
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            f"{section_num}. 개념 네트워크 그래프",
            self._styles["DcSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        # Single network chart
        try:
            from forma.domain_coverage_charts import build_concept_network_chart
            chart_buf = build_concept_network_chart(
                network, font_path=self._font_path, dpi=self._dpi,
            )
            story.append(Image(chart_buf, width=160 * mm, height=130 * mm))
            story.append(Spacer(1, 3 * mm))
        except Exception as exc:
            logger.warning("개념 네트워크 차트 생성 실패: %s", exc)
            story.append(Image(
                io.BytesIO(_FALLBACK_PNG), width=10 * mm, height=10 * mm,
            ))

        # Comparison chart (if multi-section data available)
        if deliveries_by_section and len(deliveries_by_section) >= 2:
            try:
                from forma.domain_coverage_charts import (
                    build_concept_network_comparison,
                )
                comp_buf = build_concept_network_comparison(
                    network, deliveries_by_section,
                    font_path=self._font_path, dpi=self._dpi,
                )
                story.append(PageBreak())
                story.append(Image(
                    comp_buf, width=170 * mm, height=130 * mm,
                ))
                story.append(Spacer(1, 3 * mm))
            except Exception as exc:
                logger.warning("네트워크 비교 차트 생성 실패: %s", exc)

        # Text summary: top 3 edges by weight
        if network.edges:
            sorted_edges = sorted(
                network.edges, key=lambda e: e.weight, reverse=True,
            )[:3]
            connections = ", ".join(
                f"{e.source}-{e.target}" for e in sorted_edges
            )
            story.append(Paragraph(
                f"주요 연결: {_esc(connections)}",
                self._styles["DcBody"],
            ))

        story.append(Spacer(1, 5 * mm))
        return story


# Backward compatibility alias
DomainCoveragePDFReportGenerator = DomainDeliveryPDFReportGenerator
