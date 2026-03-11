"""Student individual PDF report generator using ReportLab Platypus.

Builds A4 PDF reports from pre-computed YAML evaluation data.
Each PDF contains header, summary charts, per-question sections
with answer comparison, charts, and feedback.  No LLM API calls.
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from forma.font_utils import esc as _esc, find_korean_font, register_korean_fonts
from forma.graph_visualizer import GraphVisualizer
from forma.report_charts import ReportChartGenerator
from forma.report_data_loader import (
    ClassDistributions,
    QuestionReportData,
    StudentReportData,
)

logger = logging.getLogger(__name__)

# Understanding level → colour mapping
_LEVEL_COLORS = {
    "Advanced": HexColor("#2E7D32"),
    "Proficient": HexColor("#1565C0"),
    "Developing": HexColor("#F57F17"),
    "Beginning": HexColor("#C62828"),
}


class StudentPDFReportGenerator:
    """Generate per-student PDF reports with charts and feedback.

    Args:
        font_path: Path to Korean .ttf font.  Auto-detected if None.
        dpi: Resolution for chart images (default 150).
    """

    def __init__(
        self,
        font_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        if font_path is None:
            font_path = find_korean_font()
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font not found: {font_path}")

        self._font_path = font_path
        self._dpi = dpi

        register_korean_fonts(font_path)

        self._styles = getSampleStyleSheet()
        self._styles.add(ParagraphStyle(
            "KoreanTitle",
            parent=self._styles["Title"],
            fontName="NanumGothicBold",
            fontSize=18,
        ))
        self._styles.add(ParagraphStyle(
            "KoreanHeading",
            parent=self._styles["Heading2"],
            fontName="NanumGothicBold",
            fontSize=14,
        ))
        self._styles.add(ParagraphStyle(
            "KoreanSubheading",
            parent=self._styles["Heading3"],
            fontName="NanumGothicBold",
            fontSize=12,
        ))
        self._styles.add(ParagraphStyle(
            "KoreanBody",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=11,
            leading=16,
        ))
        self._styles.add(ParagraphStyle(
            "KoreanSmall",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=9,
            leading=13,
        ))
        self._styles.add(ParagraphStyle(
            "KoreanAnswerText",
            parent=self._styles["Normal"],
            fontName="NanumGothic",
            fontSize=9,
            leading=13,
            wordWrap="CJK",
        ))

        self._chart_gen = ReportChartGenerator(
            font_path=font_path,
            dpi=dpi,
        )
        self._graph_visualizer = None  # lazy init

    def generate_pdf(
        self,
        student_data: StudentReportData,
        distributions: ClassDistributions,
        output_dir: str,
        weekly_deltas: Optional[dict] = None,
        trajectory_chart: Optional["io.BytesIO"] = None,
        learning_path: Optional["LearningPath"] = None,
        learning_path_chart: Optional["io.BytesIO"] = None,
        grade_trend: Optional[str] = None,
    ) -> str:
        """Generate a single student's PDF report.

        Args:
            student_data: StudentReportData for this student.
            distributions: Class-level distributions for comparison.
            output_dir: Directory for output PDF files.
            weekly_deltas: Optional dict of delta info {qsn_or_"overall": WeeklyDelta}.
            trajectory_chart: Optional PNG BytesIO of weekly trajectory bar chart.
            learning_path: Optional LearningPath for this student (v0.10.0 US4).
            learning_path_chart: Optional PNG BytesIO of learning path DAG chart.
            grade_trend: Optional softened grade tier ("상위권"/"중위권"/"하위권")
                for grade trend display (v0.10.0 US6, FR-031).

        Returns:
            Absolute path to the generated PDF.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = self._make_output_filename(student_data, output_dir)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )

        story: list = []

        # Header
        story.extend(self._build_header_section(student_data))

        # Summary (with optional delta display)
        story.extend(
            self._build_summary_section(student_data, distributions),
        )

        # Delta info section (if available)
        if weekly_deltas:
            story.extend(self._build_delta_section(weekly_deltas))

        # Trajectory chart (if available)
        if trajectory_chart is not None:
            story.extend(self._build_trajectory_section(trajectory_chart))

        # Learning path section (v0.10.0 US4, FR-020, FR-023)
        if learning_path is not None:
            story.extend(
                self._build_learning_path_section(learning_path, learning_path_chart)
            )

        # Grade trend section (v0.10.0 US6, FR-031)
        if grade_trend is not None:
            story.extend(self._build_grade_trend_section(grade_trend))

        # Per-question sections
        for q in student_data.questions:
            story.extend(
                self._build_question_section(q, distributions),
            )

        doc.build(story)
        return os.path.abspath(output_path)

    def _build_delta_section(self, weekly_deltas: dict) -> list:
        """Build section showing score changes from previous week."""
        story = []
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            _esc("전주 대비 변화"),
            self._styles["KoreanHeading"],
        ))

        rows = [["항목", "현재 점수", "이전 점수", "변화"]]
        overall = weekly_deltas.get("overall")
        if overall:
            prev = f"{overall.previous_score:.2f}" if overall.previous_score is not None else "-"
            delta_str = f"{overall.delta:+.2f}" if overall.delta is not None else "-"
            rows.append(["종합", f"{overall.current_score:.2f}", prev,
                         f"{overall.delta_symbol} {delta_str}"])

        for key, wd in weekly_deltas.items():
            if key == "overall":
                continue
            prev = f"{wd.previous_score:.2f}" if wd.previous_score is not None else "-"
            delta_str = f"{wd.delta:+.2f}" if wd.delta is not None else "-"
            rows.append([f"문제 {key}", f"{wd.current_score:.2f}", prev,
                         f"{wd.delta_symbol} {delta_str}"])

        if len(rows) > 1:
            tbl = Table(rows, colWidths=[40 * mm, 30 * mm, 30 * mm, 40 * mm])
            tbl.setStyle(TableStyle([
                ("FONT", (0, 0), (-1, -1), "NanumGothic", 8),
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E3F2FD")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]))
            story.append(tbl)

        return story

    def _build_trajectory_section(self, trajectory_chart: "io.BytesIO") -> list:
        """Build section showing weekly score trajectory chart."""
        import io as _io
        story = []
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            _esc("주차별 점수 추이"),
            self._styles["KoreanHeading"],
        ))
        trajectory_chart.seek(0)
        img = Image(trajectory_chart, width=120 * mm, height=60 * mm)
        story.append(img)
        return story

    def _build_learning_path_section(
        self,
        learning_path: "LearningPath",
        learning_path_chart: Optional["io.BytesIO"] = None,
    ) -> list:
        """Build recommended learning path section (v0.10.0 US4, FR-020).

        Shows ordered concept list and optional DAG visualization.
        Displays "추가 학습 불필요" when path is empty.

        Args:
            learning_path: LearningPath for this student.
            learning_path_chart: Optional PNG BytesIO of DAG chart.

        Returns:
            List of ReportLab flowables.
        """
        story: list = []
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph(
            _esc("추천 학습 경로"),
            self._styles["KoreanHeading"],
        ))
        story.append(Spacer(1, 3 * mm))

        if not learning_path.ordered_path:
            story.append(Paragraph(
                _esc("추가 학습 불필요 — 모든 개념이 숙달되었습니다."),
                self._styles["KoreanBody"],
            ))
            return story

        # Ordered concept table
        header = ["순서", "개념"]
        rows = [header]
        for i, concept in enumerate(learning_path.ordered_path, 1):
            rows.append([str(i), _esc(concept)])

        tbl = Table(rows, colWidths=[20 * mm, 130 * mm])
        tbl.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "NanumGothic", 9),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E3F2FD")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ]))
        story.append(tbl)

        if learning_path.capped:
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(
                _esc("※ 결손 개념이 많아 상위 20개만 표시합니다."),
                self._styles["KoreanBody"],
            ))

        # DAG chart (if provided)
        if learning_path_chart is not None:
            story.append(Spacer(1, 3 * mm))
            learning_path_chart.seek(0)
            img = Image(learning_path_chart, width=140 * mm, height=80 * mm)
            story.append(img)

        return story

    def _build_grade_trend_section(self, grade_trend: str) -> list:
        """Build softened grade trend section (FR-031).

        Shows only tier-level language (상위권/중위권/하위권) without
        explicit grade predictions (A/B/C/D/F). No specific grades
        are revealed to students.

        Args:
            grade_trend: Softened tier string ("상위권", "중위권", or "하위권").

        Returns:
            List of ReportLab flowables.
        """
        story: list = []
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph(
            _esc("학습 추이 예측"),
            self._styles["KoreanHeading"],
        ))
        story.append(Spacer(1, 3 * mm))

        trend_text = (
            f"현재 학습 추이가 유지되면 "
            f"{_esc(grade_trend)}으로 예상됩니다."
        )
        story.append(Paragraph(trend_text, self._styles["KoreanBody"]))

        return story

    def _build_header_section(
        self,
        student_data: StudentReportData,
    ) -> list:
        """Build the report header with student info."""
        story = []

        story.append(Paragraph(
            _esc("학생 개인별 평가 리포트"),
            self._styles["KoreanTitle"],
        ))

        # Course and chapter info
        info_parts = []
        if student_data.course_name:
            info_parts.append(f"과목: {_esc(student_data.course_name)}")
        if student_data.chapter_name:
            info_parts.append(f"챕터: {_esc(student_data.chapter_name)}")
        if student_data.week_num:
            info_parts.append(f"주차: {student_data.week_num}주차")
        if info_parts:
            story.append(Paragraph(
                " | ".join(info_parts),
                self._styles["KoreanBody"],
            ))

        story.append(Spacer(1, 5 * mm))

        # Student info table
        info_data = [
            [
                Paragraph("학번", self._styles["KoreanSmall"]),
                Paragraph(
                    _esc(student_data.student_number),
                    self._styles["KoreanBody"],
                ),
                Paragraph("이름", self._styles["KoreanSmall"]),
                Paragraph(
                    _esc(student_data.real_name),
                    self._styles["KoreanBody"],
                ),
                Paragraph("분반", self._styles["KoreanSmall"]),
                Paragraph(
                    _esc(student_data.class_name),
                    self._styles["KoreanBody"],
                ),
            ],
        ]
        info_table = Table(
            info_data,
            colWidths=[20 * mm, 40 * mm, 15 * mm, 35 * mm, 15 * mm, 35 * mm],
        )
        info_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("BACKGROUND", (0, 0), (0, 0), HexColor("#F0F0F0")),
            ("BACKGROUND", (2, 0), (2, 0), HexColor("#F0F0F0")),
            ("BACKGROUND", (4, 0), (4, 0), HexColor("#F0F0F0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 8 * mm))

        return story

    def _build_summary_section(
        self,
        student_data: StudentReportData,
        distributions: ClassDistributions,
    ) -> list:
        """Build the summary section with overall charts and table."""
        story = []

        story.append(Paragraph(
            _esc("종합 평가 요약"),
            self._styles["KoreanHeading"],
        ))
        story.append(Spacer(1, 3 * mm))

        # Overall box-whisker chart
        if distributions.overall_ensemble:
            avg_score = sum(
                q.ensemble_score for q in student_data.questions
            ) / max(len(student_data.questions), 1)
            buf = self._chart_gen.score_boxplot(
                distributions.overall_ensemble,
                avg_score,
                title="종합 점수 분포",
            )
            try:
                story.append(Image(buf, width=160 * mm, height=50 * mm))
                story.append(Spacer(1, 5 * mm))
            except Exception as exc:
                logger.warning("Summary boxplot image rendering failed: %s", exc)
                story.append(Spacer(1, 5 * mm))

        # Radar chart — student vs class average multi-dimensional profile
        if student_data.questions:
            n_q = len(student_data.questions)
            mean_cc = sum(q.concept_coverage for q in student_data.questions) / n_q
            mean_llm = sum(q.llm_median_score for q in student_data.questions) / n_q
            mean_rasch = sum(q.rasch_theta for q in student_data.questions) / n_q

            norm_llm = max(0.0, min(1.0, (mean_llm - 1) / 2))
            norm_rasch = max(0.0, min(1.0, (mean_rasch + 5) / 10))
            student_axes = [mean_cc, norm_llm, norm_rasch]

            all_cc = [v for vals in distributions.concept_coverages.values() for v in vals]
            all_llm = [v for vals in distributions.llm_scores.values() for v in vals]
            all_rasch = [v for vals in distributions.rasch_thetas.values() for v in vals]

            avg_cc = sum(all_cc) / max(len(all_cc), 1)
            avg_llm_raw = sum(all_llm) / max(len(all_llm), 1)
            avg_rasch_raw = sum(all_rasch) / max(len(all_rasch), 1)

            avg_llm_norm = max(0.0, min(1.0, (avg_llm_raw - 1) / 2))
            avg_rasch_norm = max(0.0, min(1.0, (avg_rasch_raw + 5) / 10))
            class_avg_axes = [avg_cc, avg_llm_norm, avg_rasch_norm]

            labels = ["개념 커버리지", "LLM 루브릭", "Rasch 능력치"]

            radar_buf = self._chart_gen.radar_chart(
                student_axes, class_avg_axes, labels,
            )
            try:
                story.append(Image(radar_buf, width=90 * mm, height=90 * mm))
                story.append(Spacer(1, 5 * mm))
            except Exception as exc:
                logger.warning("Summary radar chart rendering failed: %s", exc)
                story.append(Spacer(1, 5 * mm))

        # Summary table
        if student_data.questions:
            header = [
                Paragraph("문항", self._styles["KoreanSmall"]),
                Paragraph("종합점수", self._styles["KoreanSmall"]),
                Paragraph("이해도 수준", self._styles["KoreanSmall"]),
                Paragraph("개념 커버리지", self._styles["KoreanSmall"]),
            ]
            rows = [header]
            for q in student_data.questions:
                level_color = _LEVEL_COLORS.get(
                    q.understanding_level, HexColor("#000000"),
                )
                rows.append([
                    Paragraph(
                        f"Q{q.question_sn}",
                        self._styles["KoreanBody"],
                    ),
                    Paragraph(
                        f"{q.ensemble_score:.2f}",
                        self._styles["KoreanBody"],
                    ),
                    Paragraph(
                        f'<font color="{level_color.hexval()}">'
                        f"{_esc(q.understanding_level)}</font>",
                        self._styles["KoreanBody"],
                    ),
                    Paragraph(
                        f"{q.concept_coverage:.0%}",
                        self._styles["KoreanBody"],
                    ),
                ])

            summary_table = Table(
                rows,
                colWidths=[30 * mm, 35 * mm, 45 * mm, 50 * mm],
            )
            summary_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E8E8E8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(summary_table)

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_question_section(
        self,
        question_data: QuestionReportData,
        distributions: ClassDistributions,
    ) -> list:
        """Build a per-question section with charts and feedback."""
        story = []
        qsn = question_data.question_sn

        story.append(PageBreak())

        # Question heading
        story.append(Paragraph(
            _esc(f"문항 {qsn}"),
            self._styles["KoreanHeading"],
        ))
        story.append(Spacer(1, 3 * mm))

        # Understanding badge
        badge_buf = self._chart_gen.understanding_badge(
            question_data.understanding_level,
            question_data.ensemble_score,
        )
        try:
            story.append(Image(badge_buf, width=60 * mm, height=15 * mm))
            story.append(Spacer(1, 5 * mm))
        except Exception:
            logger.debug("Skipping understanding badge image (could not load)")
            story.append(Spacer(1, 5 * mm))

        # Side-by-side answer comparison table
        story.append(Paragraph(
            _esc("답안 비교"),
            self._styles["KoreanSubheading"],
        ))
        answer_data = [
            [
                Paragraph("모범답안", self._styles["KoreanSmall"]),
                Paragraph("학생답안", self._styles["KoreanSmall"]),
            ],
            [
                Paragraph(
                    _esc(question_data.model_answer),
                    self._styles["KoreanAnswerText"],
                ),
                Paragraph(
                    _esc(question_data.student_answer),
                    self._styles["KoreanAnswerText"],
                ),
            ],
        ]
        answer_table = Table(answer_data, colWidths=[85 * mm, 85 * mm])
        answer_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E8E8E8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(answer_table)
        story.append(Spacer(1, 5 * mm))

        # Concept coverage bar chart
        if question_data.concepts:
            concept_buf = self._chart_gen.concept_coverage_bar(
                question_data.concepts,
            )
            concept_height = max(30, len(question_data.concepts) * 20)
            try:
                story.append(Image(
                    concept_buf,
                    width=160 * mm,
                    height=concept_height * mm / 25.4 * 25.4,
                ))
                story.append(Spacer(1, 5 * mm))
            except Exception:
                logger.debug("Skipping concept coverage image (could not load)")
                story.append(Spacer(1, 5 * mm))

        # Component comparison box-whisker
        comp_dists = distributions.component_scores.get(qsn, {})
        if comp_dists:
            comp_student = question_data.component_scores or {}
            comp_buf = self._chart_gen.component_comparison(
                comp_dists,
                comp_student,
                qsn,
            )
            try:
                story.append(Image(comp_buf, width=160 * mm, height=80 * mm))
                story.append(Spacer(1, 5 * mm))
            except Exception:
                logger.debug("Skipping component comparison image (could not load)")
                story.append(Spacer(1, 5 * mm))

        # Graph diagram (if graph comparison data available)
        if question_data.graph_matched_edges or question_data.graph_missing_edges:
            if self._graph_visualizer is None:
                self._graph_visualizer = GraphVisualizer(
                    font_path=self._font_path,
                )
            buf, omitted = self._graph_visualizer.visualize_comparison_to_bytesio(
                matched=question_data.graph_matched_edges,
                missing=question_data.graph_missing_edges,
                extra=question_data.graph_extra_edges,
                wrong_direction=question_data.graph_wrong_direction_edges,
                title=f"Q{question_data.question_sn} 개념 네트워크 비교",
                max_edges=30,
            )
            try:
                buf.seek(0)
                story.append(Image(buf, width=160 * mm, height=100 * mm))
                if omitted > 0:
                    story.append(Paragraph(
                        f"추가 {omitted}개 엣지 생략",
                        self._styles["KoreanSmall"],
                    ))
            except Exception as exc:
                logger.warning("Graph diagram rendering failed: %s", exc)

        # Hub gap table (if hub_gap_entries available)
        hub_gap_entries = getattr(question_data, "hub_gap_entries", None)
        if hub_gap_entries:
            hub_data = [["개념", "중심성", "포함"]]  # header row
            for entry in hub_gap_entries:
                presence = "O" if entry.student_present else "X"
                hub_data.append([
                    entry.concept,
                    f"{entry.degree_centrality:.3f}",
                    presence,
                ])

            hub_table = Table(hub_data, colWidths=[80 * mm, 40 * mm, 30 * mm])
            hub_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#404040")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("FONTNAME", (0, 0), (-1, -1), "NanumGothic"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F5F5F5")]),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(hub_table)
            story.append(Spacer(1, 5 * mm))

        # Feedback text
        story.append(Paragraph(
            _esc("피드백"),
            self._styles["KoreanSubheading"],
        ))

        sections = parse_feedback_sections(question_data.feedback_text)
        for section_name, section_text in sections.items():
            story.append(Paragraph(
                f"<b>[{_esc(section_name)}]</b>",
                self._styles["KoreanBody"],
            ))
            story.append(Paragraph(
                _esc(section_text),
                self._styles["KoreanBody"],
            ))
            story.append(Spacer(1, 2 * mm))

        # Misconceptions
        if question_data.misconceptions:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(
                _esc("감지된 오개념"),
                self._styles["KoreanSubheading"],
            ))
            for m in question_data.misconceptions:
                story.append(Paragraph(
                    f"• {_esc(m)}",
                    self._styles["KoreanBody"],
                ))

        return story

    @staticmethod
    def _make_output_filename(
        student_data: StudentReportData,
        output_dir: str,
    ) -> str:
        """Generate output filename in {분반코드}_{주차}_{학번}_{이름}.pdf format.

        Args:
            student_data: Student data with class/name/number info.
            output_dir: Output directory path.

        Returns:
            Full file path.
        """
        # Extract class code: "A반" → "1A", "B반" → "1B"
        class_name = student_data.class_name or "X"
        class_letter = class_name[0] if class_name else "X"
        class_code = f"1{class_letter}"

        week = f"w{student_data.week_num}"
        student_num = student_data.student_number or student_data.student_id
        name = student_data.real_name or student_data.student_id

        safe_name = _sanitize_filename(name)
        safe_num = _sanitize_filename(student_num)

        filename = f"{class_code}_{week}_{safe_num}_{safe_name}.pdf"
        return os.path.join(output_dir, filename)


def _sanitize_filename(name: str) -> str:
    """Remove OS-unsafe characters from a filename component.

    Args:
        name: Raw filename component.

    Returns:
        Sanitized string safe for all OS file systems.
    """
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def parse_feedback_sections(text: str) -> dict[str, str]:
    """Parse feedback text into named sections.

    Supports both old format ([평가 요약], [분석 결과], [학습 제안]) and
    new format ([현재 상태], [원인], [학생에게 권하는 사항]) section names
    for backward compatibility.

    Args:
        text: Raw feedback text possibly containing section markers.

    Returns:
        Dict mapping section name to content.  If no markers found,
        returns {"전체 피드백": text}.
    """
    if not text or text == "(피드백 데이터 없음)":
        return {"전체 피드백": text or ""}

    section_names = ["평가 요약", "분석 결과", "학습 제안", "현재 상태", "원인", "학생에게 권하는 사항"]
    pattern = r"\[(" + "|".join(re.escape(s) for s in section_names) + r")\]"
    parts = re.split(pattern, text)

    result: dict[str, str] = {}
    if len(parts) < 2:
        # No section markers found — return full text
        return {"전체 피드백": text.strip()}

    # parts[0] is text before first marker (usually empty)
    i = 1
    while i < len(parts) - 1:
        section_name = parts[i].strip()
        section_content = parts[i + 1].strip()
        result[section_name] = section_content
        i += 2

    return result if result else {"전체 피드백": text.strip()}
