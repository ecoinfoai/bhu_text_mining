"""Student counseling PDF report generator.

Generates per-student A4 PDF reports summarising their evaluation
results for professor-student counseling sessions.  Uses ReportLab
with NanumGothic font for Korean text rendering.
"""

from __future__ import annotations

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from forma.font_utils import esc as _esc, find_korean_font
from forma.io_utils import safe_filename


# Understanding level → colour mapping
_LEVEL_COLORS = {
    "Advanced": HexColor("#2E7D32"),  # green
    "Proficient": HexColor("#1565C0"),  # blue
    "Developing": HexColor("#F57F17"),  # amber
    "Beginning": HexColor("#C62828"),  # red
}


class StudentReportGenerator:
    """Generate per-student counseling PDF reports.

    Args:
        font_path: Path to Korean .ttf font.  Auto-detected if None.
    """

    def __init__(self, font_path: str | None = None) -> None:
        if font_path is None:
            font_path = find_korean_font()
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font not found: {font_path}")

        pdfmetrics.registerFont(TTFont("NanumGothic", font_path))
        bold_path = font_path.replace(".ttf", "Bold.ttf")
        if os.path.exists(bold_path):
            pdfmetrics.registerFont(TTFont("NanumGothicBold", bold_path))
        else:
            pdfmetrics.registerFont(TTFont("NanumGothicBold", font_path))

        self._styles = getSampleStyleSheet()
        self._styles.add(
            ParagraphStyle(
                "KoreanTitle",
                parent=self._styles["Title"],
                fontName="NanumGothicBold",
                fontSize=18,
            )
        )
        self._styles.add(
            ParagraphStyle(
                "KoreanHeading",
                parent=self._styles["Heading2"],
                fontName="NanumGothicBold",
                fontSize=14,
            )
        )
        self._styles.add(
            ParagraphStyle(
                "KoreanBody",
                parent=self._styles["Normal"],
                fontName="NanumGothic",
                fontSize=11,
                leading=16,
            )
        )
        self._styles.add(
            ParagraphStyle(
                "KoreanSmall",
                parent=self._styles["Normal"],
                fontName="NanumGothic",
                fontSize=9,
                leading=13,
            )
        )

    def generate_pdf(
        self,
        student_id: str,
        counseling_data: dict,
        config_data: dict,
        output_path: str,
    ) -> str:
        """Generate a single student's counseling PDF report.

        Args:
            student_id: Student identifier.
            counseling_data: Counseling summary dict (from pipeline output).
            config_data: Exam config dict.
            output_path: Path for the output PDF file.

        Returns:
            Absolute path to the generated PDF.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )

        story = []

        # Header
        course_name = config_data.get("course_name", "")
        story.append(
            Paragraph(
                f"학습 평가 리포트 — {_esc(student_id)}",
                self._styles["KoreanTitle"],
            )
        )
        if course_name:
            story.append(
                Paragraph(
                    f"과목: {_esc(course_name)}",
                    self._styles["KoreanBody"],
                )
            )
        story.append(Spacer(1, 10 * mm))

        # Find student data in counseling summary
        student_entry = self._find_student(counseling_data, student_id)
        if not student_entry:
            story.append(
                Paragraph(
                    f"학생 {_esc(student_id)}의 평가 데이터가 없습니다.",
                    self._styles["KoreanBody"],
                )
            )
            doc.build(story)
            return os.path.abspath(output_path)

        # Per-question sections
        questions = student_entry.get("questions", [])
        for q in questions:
            qsn = q.get("question_sn", "?")
            level = q.get("understanding_level", "N/A")
            coverage = q.get("concept_coverage", 0.0)
            guidance = q.get("support_guidance", "")
            misconceptions = q.get("misconceptions", [])

            # Question heading
            story.append(
                Paragraph(
                    f"문항 {_esc(str(qsn))}",
                    self._styles["KoreanHeading"],
                )
            )

            # Understanding level with colour
            color = _LEVEL_COLORS.get(level, HexColor("#000000"))
            story.append(
                Paragraph(
                    f'이해도 수준: <font color="{color.hexval()}">{_esc(str(level))}</font>',
                    self._styles["KoreanBody"],
                )
            )

            # Concept coverage
            story.append(
                Paragraph(
                    f"개념 커버리지: {coverage:.0%}",
                    self._styles["KoreanBody"],
                )
            )

            # Misconceptions
            if misconceptions:
                story.append(
                    Paragraph(
                        "발견된 오개념:",
                        self._styles["KoreanBody"],
                    )
                )
                for m in misconceptions:
                    story.append(
                        Paragraph(
                            f"  • {_esc(str(m))}",
                            self._styles["KoreanSmall"],
                        )
                    )

            # Support guidance
            if guidance:
                story.append(Spacer(1, 3 * mm))
                story.append(
                    Paragraph(
                        f"학습 지도 방안: {_esc(str(guidance))}",
                        self._styles["KoreanBody"],
                    )
                )

            story.append(Spacer(1, 8 * mm))

        # Overall summary
        story.append(
            Paragraph(
                "종합 소견",
                self._styles["KoreanHeading"],
            )
        )
        levels = [q.get("understanding_level", "N/A") for q in questions]
        coverages = [q.get("concept_coverage", 0.0) for q in questions]
        avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0

        story.append(
            Paragraph(
                f"전체 문항 수: {len(questions)}",
                self._styles["KoreanBody"],
            )
        )
        story.append(
            Paragraph(
                f"평균 개념 커버리지: {avg_coverage:.0%}",
                self._styles["KoreanBody"],
            )
        )

        # Level distribution
        level_counts: dict[str, int] = {}
        for lv in levels:
            level_counts[lv] = level_counts.get(lv, 0) + 1
        level_str = ", ".join(f"{_esc(str(k))}: {v}문항" for k, v in level_counts.items())
        story.append(
            Paragraph(
                f"이해도 분포: {level_str}",
                self._styles["KoreanBody"],
            )
        )

        doc.build(story)
        return os.path.abspath(output_path)

    def generate_all_reports(
        self,
        counseling_data: dict,
        config_data: dict,
        output_dir: str,
    ) -> list[str]:
        """Generate PDF reports for all students in counseling data.

        Args:
            counseling_data: Full counseling summary dict.
            config_data: Exam config dict.
            output_dir: Directory for output PDFs.

        Returns:
            List of generated PDF file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths: list[str] = []

        for student in counseling_data.get("students", []):
            sid = student.get("student_id", "unknown")
            output_path = os.path.join(output_dir, f"{safe_filename(sid)}_report.pdf")
            path = self.generate_pdf(
                student_id=sid,
                counseling_data=counseling_data,
                config_data=config_data,
                output_path=output_path,
            )
            paths.append(path)

        return paths

    @staticmethod
    def _find_student(counseling_data: dict, student_id: str) -> dict | None:
        """Find a student's entry in counseling data."""
        for student in counseling_data.get("students", []):
            if student.get("student_id") == student_id:
                return student
        return None
