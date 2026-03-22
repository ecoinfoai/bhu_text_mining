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

METHODS_SECTION_TEXT: str = (
    "본 보고서의 성취도 점수(ensemble score)는 학생의 서술형 응답을 "
    "3개 독립 평가 레이어로 분석한 뒤 가중 앙상블한 종합 점수입니다. "
    "각 레이어는 서로 다른 관점에서 응답 품질을 평가하며, "
    "단일 평가 방법의 편향을 보완합니다.\n\n"
    "■ Layer 1: 개념 키워드 매칭 (Concept Coverage)\n\n"
    "학생 응답에 출제 의도에 해당하는 핵심 개념이 포함되어 "
    "있는지를 임베딩 유사도로 판정합니다.\n"
    "- 임베딩 모델: ko-sroberta-multitask (한국어 문장 임베딩)\n"
    "- 유사도 측정: 코사인 유사도, "
    "top-k 평균 (k = min(2, 문장 수))\n"
    "- 적응형 임계값: τ(M) = 0.35 + 0.02 · log(M) "
    "(M = 해당 문항의 핵심 개념 수. 개념이 많을수록 "
    "임계값을 높여 거짓 양성을 억제)\n"
    "- 산출값: 정확히 매칭된 개념 수 / "
    "전체 핵심 개념 수 (0.0 ~ 1.0)\n\n"
    "■ Layer 2: LLM 루브릭 채점 (Rubric Scoring)\n\n"
    "사전 정의된 3단계 루브릭(high / mid / low)을 기준으로 "
    "대규모 언어 모델이 응답을 채점합니다.\n"
    "- 신뢰성 확보: 동일 응답에 대해 3회 독립 호출 실시\n"
    "- 최종 점수: 3회 점수의 중위값(median) 채택 "
    "(이상치 1개가 결과를 왜곡하지 않도록 중위값 사용)\n"
    "- 산출값: 루브릭 점수를 0.0 ~ 1.0으로 정규화\n\n"
    "■ Layer 3: Rasch IRT 문항 난이도 보정 "
    "(Item Response Theory)\n\n"
    "문항 자체의 난이도를 고려하여 학생 능력 추정치를 "
    "보정합니다.\n"
    "- 모델: Rasch 모형 (1-parameter IRT)\n"
    "- 보정 효과: 어려운 문항에서 중간 점수를 받은 학생은 "
    "쉬운 문항에서 같은 점수를 받은 학생보다 높게 평가\n"
    "- 산출값: theta 추정치를 0.0 ~ 1.0으로 정규화\n\n"
    "■ 앙상블 (Ensemble)\n\n"
    "위 3개 레이어의 점수를 가중 평균하여 최종 성취도 "
    "점수를 산출합니다. 가중치는 각 레이어의 신뢰도 지표에 "
    "따라 동적으로 조정됩니다.\n\n"
    "최종 점수 해석:\n"
    "  0.70 이상 — 우수 "
    "(핵심 개념을 정확히 이해하고 서술)\n"
    "  0.45 ~ 0.70 — 보통 "
    "(부분적 이해, 보완 필요)\n"
    "  0.45 미만 — 위험 "
    "(핵심 개념 누락 또는 오개념 포함)"
)

RISK_INTERPRETATION_GUIDE: str = (
    "■ 선정 기준\n\n"
    "분석 기간 내 모든 주차에서 성취도 점수가 0.45 미만인 "
    "학생을 \"지속 위험군\"으로 분류합니다. 단일 주차에서만 "
    "0.45 미만인 학생은 일시적 부진으로 간주하여 이 목록에 "
    "포함되지 않습니다.\n\n"
    "■ 표 컬럼 해석\n\n"
    "학생 — 학생 식별자 (익명 ID)\n"
    "최종 점수 — 분석 기간 중 가장 마지막 주차의 "
    "성취도 점수\n"
    "추세(기울기) — OLS(최소자승법) 선형 회귀의 기울기 "
    "계수. 분석 기간의 주차 번호(x)와 성취도 점수(y)에 "
    "대해 1차 다항식 적합(numpy.polyfit)으로 산출\n\n"
    "■ 추세 기울기 해석\n\n"
    "양수(+) → 점수가 주차가 지남에 따라 상승하는 추세\n"
    "음수(-) → 점수가 주차가 지남에 따라 하락하는 추세\n"
    "0 근처 → 정체 (개선도 악화도 없음)\n\n"
    "■ 중재 우선순위 판단\n\n"
    "기울기 > +0.05 (개선 추세 학생):\n"
    "  성취도가 점차 향상되고 있습니다. 적극적 "
    "중재(면담, 보충학습)로 위험 구간 이탈을 "
    "앞당길 수 있는 우선 지원 대상입니다.\n\n"
    "기울기 < -0.05 (하락 추세 학생):\n"
    "  성취도가 지속적으로 하락하고 있어 즉각적인 "
    "개입이 필요합니다. 면담을 통해 학습 장애 "
    "요인을 파악하는 것을 권장합니다.\n\n"
    "기울기 ≈ 0 (정체 학생):\n"
    "  뚜렷한 변화 없이 낮은 성취도가 유지되고 "
    "있습니다. 기존 학습 방법의 전환이 필요하며, "
    "멘토링이나 학습 전략 변경을 고려하십시오."
)

MASTERY_INTERPRETATION_GUIDE: str = (
    "■ 표/차트 해석\n\n"
    "첫 주차/마지막 주차 비율: 해당 개념을 정확히 언급한 "
    "학생 비율 (0.0 ~ 1.0)\n"
    "변화(Δ) > 0: 학급 전체적으로 해당 개념 이해도 향상\n"
    "변화(Δ) < 0: 해당 개념 이해도 하락. "
    "보충 설명이 필요할 수 있음"
)


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
        mastery_top_n: int | None = None,
        class_data: dict | None = None,
        class_ids: list[str] | None = None,
        heatmap_layout: tuple[int, int] | None = None,
    ) -> str:
        """Generate the longitudinal summary report PDF.

        Args:
            summary_data: Complete summary data.
            output_path: Output path for the PDF file.
            intervention_effects: Optional list of InterventionEffect (v0.10.0 US2, FR-011).
            ocr_confidence_trajectories: {student_id: [(week, mean_confidence), ...]}
                for OCR confidence trend chart (v0.12.5 US3).
            mastery_top_n: If set, limit mastery chart to top N concepts by |delta|.
            class_data: {class_id: {student_id: {week: score}}} for per-class heatmaps.
            class_ids: Ordered class identifiers for subplot titles.
            heatmap_layout: (rows, cols) subplot grid dimensions.

        Returns:
            Absolute path to generated PDF file.
        """
        story: list = []
        story.extend(self._build_cover_page(summary_data))
        story.extend(self._build_methods_section())
        story.extend(self._build_class_trend_section(summary_data))
        story.extend(self._build_trajectory_section(summary_data))
        story.extend(self._build_heatmap_section(
            summary_data,
            class_data=class_data,
            class_ids=class_ids,
            heatmap_layout=heatmap_layout,
        ))
        story.extend(self._build_risk_analysis_section(summary_data))
        story.extend(self._build_concept_mastery_section(
            summary_data, mastery_top_n=mastery_top_n,
        ))

        # Topic statistics section (v0.13.1 US7)
        if (
            summary_data.topic_statistics
            or summary_data.topic_trends
        ):
            story.extend(
                self._build_topic_statistics_section(
                    summary_data.topic_statistics or [],
                    summary_data.topic_trends or [],
                )
            )

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

    def _build_methods_section(self) -> list:
        """Build static Methods section describing score computation.

        Returns:
            List of ReportLab flowables for the Methods section.
        """
        story: list = []
        story.append(Paragraph(
            _esc("방법: 성취도 점수 산출"),
            self._styles["LongSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        for paragraph in METHODS_SECTION_TEXT.split("\n\n"):
            cleaned = paragraph.strip()
            if not cleaned:
                continue
            story.append(Paragraph(
                _esc(cleaned).replace("\n", "<br/>"),
                self._styles["LongBody"],
            ))
            story.append(Spacer(1, 2 * mm))

        story.append(PageBreak())
        return story

    def _build_class_trend_section(self, data: LongitudinalSummaryData) -> list:
        """Build class achievement trend section with weekly averages table."""
        story = []
        story.append(Paragraph("1. 학급 성취도 추이", self._styles["LongSection"]))
        story.append(Spacer(1, 3 * mm))

        # Interpretation guide
        _guide = (
            "학급 평균: 해당 주차에 응시한 전체 학생의 "
            "성취도 점수(ensemble score) 평균입니다. "
            "0.70 이상 = 우수, 0.45~0.70 = 보통, "
            "0.45 미만 = 위험 구간입니다."
        )
        story.append(Paragraph(
            _esc(_guide), self._styles["LongBody"],
        ))
        story.append(Spacer(1, 2 * mm))

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

        # Interpretation guide
        _guide = (
            "각 선은 한 학생의 주차별 성취도 변화를 나타냅니다. "
            "빨간 실선 = 지속 위험군(매 주차 0.45 미만), "
            "회색 선 = 일반 학생, 파란 굵은 선 = 학급 평균. "
            "선의 기울기가 양수이면 개선 추세, "
            "음수이면 하락 추세를 의미합니다."
        )
        story.append(Paragraph(
            _esc(_guide), self._styles["LongBody"],
        ))
        story.append(Spacer(1, 2 * mm))

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

    def _build_heatmap_section(
        self,
        data: LongitudinalSummaryData,
        class_data: dict | None = None,
        class_ids: list[str] | None = None,
        heatmap_layout: tuple[int, int] | None = None,
    ) -> list:
        """Build student x week heatmap section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            "3. 학생×주차 히트맵",
            self._styles["LongSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        try:
            if (
                class_data
                and class_ids
                and heatmap_layout
            ):
                from forma.longitudinal_report_charts import (
                    build_class_heatmap_subplots,
                )
                chart_buf = build_class_heatmap_subplots(
                    class_data, class_ids, heatmap_layout,
                    font_path=self._font_path,
                    dpi=self._dpi,
                )
                rows, cols = heatmap_layout
                h = rows * 60 * mm
                story.append(Image(
                    chart_buf,
                    width=160 * mm,
                    height=h,
                ))
            else:
                from forma.longitudinal_report_charts import (
                    build_class_week_heatmap,
                )
                chart_buf = build_class_week_heatmap(
                    data,
                    font_path=self._font_path,
                    dpi=self._dpi,
                )
                story.append(Image(
                    chart_buf,
                    width=160 * mm,
                    height=120 * mm,
                ))
        except Exception as exc:
            logger.warning(
                "Failed to generate heatmap: %s", exc,
            )
            story.append(Image(
                io.BytesIO(_FALLBACK_PNG),
                width=10 * mm, height=10 * mm,
            ))

        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            "최종 주차 점수 기준 정렬. "
            "빨강(낮음) → 초록(높음). 회색 = 결측.",
            self._styles["LongBody"],
        ))
        story.append(Spacer(1, 5 * mm))
        return story

    def _build_risk_analysis_section(self, data: LongitudinalSummaryData) -> list:
        """Build persistent risk student analysis section with interpretation guide."""
        story = []
        story.append(Paragraph(
            "4. 지속 위험군 분석",
            self._styles["LongSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        # Interpretation guide (always shown)
        story.append(Paragraph(
            _esc("지속 위험군 분석 — 해석 가이드"),
            self._styles["LongSection"],
        ))
        story.append(Spacer(1, 2 * mm))
        for paragraph in RISK_INTERPRETATION_GUIDE.split("\n\n"):
            cleaned = paragraph.strip()
            if not cleaned:
                continue
            story.append(Paragraph(
                _esc(cleaned).replace("\n", "<br/>"),
                self._styles["LongBody"],
            ))
            story.append(Spacer(1, 2 * mm))
        story.append(Spacer(1, 3 * mm))

        n_persistent = len(data.persistent_risk_students)
        story.append(Paragraph(
            f"전 기간 지속 위험군: <b>{n_persistent}명</b>",
            self._styles["LongBody"],
        ))
        story.append(Spacer(1, 2 * mm))

        if data.persistent_risk_students:
            story.extend(self._build_risk_tables_by_trend(data))
        else:
            story.append(Paragraph(
                "지속 위험군 학생이 없습니다.",
                self._styles["LongBody"],
            ))

        story.append(Spacer(1, 5 * mm))
        return story

    def _build_risk_tables_by_trend(
        self,
        data: LongitudinalSummaryData,
    ) -> list:
        """Build risk tables split by trend category.

        Args:
            data: Summary data with persistent risk students.

        Returns:
            List of ReportLab flowables with 3 tables.
        """
        improving = []
        declining = []
        stagnant = []

        for sid in data.persistent_risk_students:
            traj = next(
                (t for t in data.student_trajectories
                 if t.student_id == sid),
                None,
            )
            if traj is None:
                continue
            final_score = "—"
            if data.period_weeks and traj.weekly_scores:
                for w in reversed(data.period_weeks):
                    if w in traj.weekly_scores:
                        final_score = (
                            f"{traj.weekly_scores[w]:.3f}"
                        )
                        break
            row = (sid, final_score, traj.overall_trend)
            if traj.overall_trend > 0.05:
                improving.append(row)
            elif traj.overall_trend < -0.05:
                declining.append(row)
            else:
                stagnant.append(row)

        groups = [
            ("하락 추세 학생", "#C62828", declining),
            ("정체 학생", "#E65100", stagnant),
            ("개선 추세 학생", "#2E7D32", improving),
        ]
        story: list = []
        for title, color, rows in groups:
            story.append(Paragraph(
                f"{_esc(title)} ({len(rows)}명)",
                self._styles["LongSection"],
            ))
            story.append(Spacer(1, 2 * mm))
            if not rows:
                story.append(Paragraph(
                    "해당 학생 없음",
                    self._styles["LongBody"],
                ))
                story.append(Spacer(1, 3 * mm))
                continue
            story.append(
                self._build_risk_table(rows, color)
            )
            story.append(Spacer(1, 4 * mm))
        return story

    def _build_risk_table(
        self,
        rows: list[tuple[str, str, float]],
        header_color: str,
    ) -> Table:
        """Build a single risk group table.

        Args:
            rows: List of (student_id, final_score, trend).
            header_color: Hex color for header background.

        Returns:
            ReportLab Table flowable.
        """
        header = [
            Paragraph("학생", self._styles["LongTableHeader"]),
            Paragraph("최종 점수",
                       self._styles["LongTableHeader"]),
            Paragraph("추세(기울기)",
                       self._styles["LongTableHeader"]),
        ]
        table_rows = [header]
        for sid, score, trend in rows:
            table_rows.append([
                Paragraph(_esc(sid),
                          self._styles["LongTableData"]),
                Paragraph(score,
                          self._styles["LongTableData"]),
                Paragraph(f"{trend:+.4f}",
                          self._styles["LongTableData"]),
            ])
        tbl = Table(
            table_rows,
            colWidths=[50 * mm, 45 * mm, 45 * mm],
        )
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             HexColor(header_color)),
            ("TEXTCOLOR", (0, 0), (-1, 0),
             HexColor("#FFFFFF")),
            ("GRID", (0, 0), (-1, -1), 0.5,
             HexColor("#CCCCCC")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return tbl

    def _build_concept_mastery_section(
        self,
        data: LongitudinalSummaryData,
        mastery_top_n: int | None = None,
    ) -> list:
        """Build concept mastery change section with chart and table.

        Auto-switches to heatmap when 5+ weeks. Applies top_n
        filtering when mastery_top_n is set.

        Args:
            data: Summary data with concept mastery changes.
            mastery_top_n: Limit chart to top N concepts by |delta|.

        Returns:
            List of ReportLab flowables.
        """
        story: list = []
        story.append(PageBreak())
        story.append(Paragraph(
            "5. 개념별 마스터리 변화",
            self._styles["LongSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        # Interpretation guide
        for para in MASTERY_INTERPRETATION_GUIDE.split("\n\n"):
            cleaned = para.strip()
            if cleaned:
                story.append(Paragraph(
                    _esc(cleaned).replace("\n", "<br/>"),
                    self._styles["LongBody"],
                ))
                story.append(Spacer(1, 2 * mm))
        story.append(Spacer(1, 3 * mm))

        if not data.concept_mastery_changes:
            story.append(Paragraph(
                "개념 마스터리 데이터가 없습니다.",
                self._styles["LongBody"],
            ))
            return story

        use_heatmap = len(data.period_weeks) >= 5

        if use_heatmap:
            self._build_mastery_heatmap(
                story, data, mastery_top_n,
            )
        else:
            self._build_mastery_bar_chart(story, data)

        story.append(Spacer(1, 5 * mm))

        # Detail table (apply top_n filtering)
        changes = data.concept_mastery_changes
        if mastery_top_n and mastery_top_n < len(changes):
            changes = sorted(
                changes, key=lambda c: abs(c.delta),
                reverse=True,
            )[:mastery_top_n]
            changes = sorted(
                changes, key=lambda c: c.delta,
                reverse=True,
            )

        header = [
            Paragraph(
                "개념", self._styles["LongTableHeader"],
            ),
            Paragraph(
                "첫 주차", self._styles["LongTableHeader"],
            ),
            Paragraph(
                "마지막 주차",
                self._styles["LongTableHeader"],
            ),
            Paragraph(
                "변화(Δ)",
                self._styles["LongTableHeader"],
            ),
        ]
        rows = [header]
        for c in changes:
            delta_str = f"{c.delta:+.3f}"
            rows.append([
                Paragraph(
                    _esc(c.concept),
                    self._styles["LongTableData"],
                ),
                Paragraph(
                    f"{c.week_start_ratio:.3f}",
                    self._styles["LongTableData"],
                ),
                Paragraph(
                    f"{c.week_end_ratio:.3f}",
                    self._styles["LongTableData"],
                ),
                Paragraph(
                    delta_str,
                    self._styles["LongTableData"],
                ),
            ])

        table = Table(
            rows,
            colWidths=[50 * mm, 35 * mm, 35 * mm, 35 * mm],
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0),
             HexColor("#FFFFFF")),
            ("GRID", (0, 0), (-1, -1), 0.5,
             HexColor("#CCCCCC")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [HexColor("#FFFFFF"),
              HexColor("#F5F5F5")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 5 * mm))
        return story

    def _build_mastery_bar_chart(
        self,
        story: list,
        data: LongitudinalSummaryData,
    ) -> None:
        """Append bar chart for < 5 weeks mastery data."""
        from forma.longitudinal_report_charts import (
            build_concept_mastery_bar_chart,
        )
        try:
            chart_buf = build_concept_mastery_bar_chart(
                data,
                font_path=self._font_path,
                dpi=self._dpi,
            )
            n = len(data.concept_mastery_changes)
            chart_height = max(60, n * 15)
            story.append(Image(
                chart_buf, width=160 * mm,
                height=min(chart_height, 200) * mm,
            ))
        except Exception as exc:
            logger.warning(
                "Failed to generate mastery bar chart: %s",
                exc,
            )
            story.append(Image(
                io.BytesIO(_FALLBACK_PNG),
                width=10 * mm, height=10 * mm,
            ))

    def _build_mastery_heatmap(
        self,
        story: list,
        data: LongitudinalSummaryData,
        mastery_top_n: int | None,
    ) -> None:
        """Append concept x week heatmap for 5+ weeks."""
        from forma.longitudinal_report_charts import (
            build_concept_mastery_heatmap,
        )
        try:
            # Build mastery_data from concept_mastery_changes
            # For heatmap we need per-week data, but changes
            # only have first/last. Use what we have.
            mastery_data: dict[str, dict[int, float]] = {}
            for c in data.concept_mastery_changes:
                mastery_data[c.concept] = {}
                if data.period_weeks:
                    first_w = data.period_weeks[0]
                    last_w = data.period_weeks[-1]
                    mastery_data[c.concept][first_w] = (
                        c.week_start_ratio
                    )
                    mastery_data[c.concept][last_w] = (
                        c.week_end_ratio
                    )

            chart_buf = build_concept_mastery_heatmap(
                mastery_data, data.period_weeks,
                top_n=mastery_top_n,
                font_path=self._font_path,
                dpi=self._dpi,
            )
            n = len(data.concept_mastery_changes)
            chart_height = max(60, n * 12 + 30)
            story.append(Image(
                chart_buf, width=160 * mm,
                height=min(chart_height, 200) * mm,
            ))
        except Exception as exc:
            logger.warning(
                "Failed to generate mastery heatmap: %s",
                exc,
            )
            story.append(Image(
                io.BytesIO(_FALLBACK_PNG),
                width=10 * mm, height=10 * mm,
            ))

    def _build_topic_statistics_section(
        self,
        stats: list,
        trends: list,
    ) -> list:
        """Build topic-level statistics section with table and trends.

        Args:
            stats: List of TopicWeekStats objects.
            trends: List of TopicTrendResult objects.

        Returns:
            List of ReportLab flowables.
        """
        story: list = []
        story.append(PageBreak())
        story.append(Paragraph(
            "6. 주제별 종단 통계",
            self._styles["LongSection"],
        ))
        story.append(Spacer(1, 3 * mm))

        if not stats:
            story.append(Paragraph(
                "주제(topic) 데이터가 없습니다.",
                self._styles["LongBody"],
            ))
            return story

        # Build topic×week table
        # Group stats by topic
        topics: dict[str, dict[int, tuple]] = {}
        all_weeks: set[int] = set()
        for s in stats:
            if s.topic not in topics:
                topics[s.topic] = {}
            topics[s.topic][s.week] = (s.mean, s.std)
            all_weeks.add(s.week)

        sorted_weeks = sorted(all_weeks)
        sorted_topics = sorted(topics.keys())

        # Header row
        header = [Paragraph(
            "주차", self._styles["LongTableHeader"],
        )]
        for topic in sorted_topics:
            header.append(Paragraph(
                _esc(f"{topic}(평균)"),
                self._styles["LongTableHeader"],
            ))
            header.append(Paragraph(
                _esc(f"{topic}(SD)"),
                self._styles["LongTableHeader"],
            ))

        rows = [header]
        for week in sorted_weeks:
            row = [Paragraph(
                f"W{week}", self._styles["LongTableData"],
            )]
            for topic in sorted_topics:
                if week in topics[topic]:
                    m, sd = topics[topic][week]
                    row.append(Paragraph(
                        f"{m:.3f}",
                        self._styles["LongTableData"],
                    ))
                    row.append(Paragraph(
                        f"{sd:.3f}",
                        self._styles["LongTableData"],
                    ))
                else:
                    row.append(Paragraph(
                        "—", self._styles["LongTableData"],
                    ))
                    row.append(Paragraph(
                        "—", self._styles["LongTableData"],
                    ))
            rows.append(row)

        n_cols = 1 + len(sorted_topics) * 2
        col_w = 155.0 / n_cols
        col_widths = [col_w * mm] * n_cols
        table = Table(rows, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0),
             HexColor("#FFFFFF")),
            ("GRID", (0, 0), (-1, -1), 0.5,
             HexColor("#CCCCCC")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [HexColor("#FFFFFF"),
              HexColor("#F5F5F5")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 5 * mm))

        # Trend summary
        if trends:
            story.append(Paragraph(
                "주제별 추세 분석",
                self._styles["LongSection"],
            ))
            story.append(Spacer(1, 2 * mm))
            for t in trends:
                tau_sig = (
                    "유의" if t.kendall_p < 0.05
                    else "비유의"
                )
                line = (
                    f"{_esc(t.topic)} 추세: "
                    f"tau = {t.kendall_tau:+.2f} "
                    f"(p = {t.kendall_p:.3f}, {tau_sig}), "
                    f"rho = {t.spearman_rho:+.2f} "
                    f"(p = {t.spearman_p:.3f})"
                )
                story.append(Paragraph(
                    line, self._styles["LongBody"],
                ))
                story.append(Spacer(1, 1 * mm))

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
