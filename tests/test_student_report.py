"""Tests for student_report.py — student individual PDF report generation.

RED phase: these tests are written *before* the module exists and will fail
until ``src/forma/student_report.py`` is implemented.

Covers task item T012 (US1 PDF generator tests).

Font discovery and PDF registration are mocked to avoid OS / font dependency
in CI.  No actual PDF files are generated in unit tests.
"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from forma.report_data_loader import (
    ClassDistributions,
    ConceptDetail,
    QuestionReportData,
    StudentReportData,
)


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------


def _make_student(
    *,
    student_id: str = "S015",
    real_name: str = "이유정",
    student_number: str = "2026194126",
    class_name: str = "A반",
    course_name: str = "인체구조와기능",
    chapter_name: str = "서론",
    week_num: int = 1,
    questions: list[QuestionReportData] | None = None,
) -> StudentReportData:
    """Build a minimal StudentReportData for testing."""
    if questions is None:
        questions = [_make_question()]
    return StudentReportData(
        student_id=student_id,
        real_name=real_name,
        student_number=student_number,
        class_name=class_name,
        course_name=course_name,
        chapter_name=chapter_name,
        week_num=week_num,
        questions=questions,
    )


def _make_question(
    *,
    question_sn: int = 1,
    question_text: str = "항상성의 정의와 예를 서술하시오.",
    model_answer: str = "항상성이란 체내 환경을 일정하게 유지하는 성질이다.",
    student_answer: str = "생체항상성은 체내 환경을 일정하게 유지하는 것이다.",
    concept_coverage: float = 0.5,
    concepts: list[ConceptDetail] | None = None,
    llm_median_score: float = 2.0,
    llm_label: str = "mid",
    ensemble_score: float = 0.26,
    understanding_level: str = "Beginning",
    component_scores: dict[str, float] | None = None,
    feedback_text: str = (
        "[평가 요약]\n항상성 개념 부분 이해.\n"
        "[분석 결과]\n세부 기전 설명 부족.\n"
        "[학습 제안]\n교과서 3장 복습 권장."
    ),
    misconceptions: list[str] | None = None,
) -> QuestionReportData:
    """Build a minimal QuestionReportData for testing."""
    if concepts is None:
        concepts = [
            ConceptDetail(
                concept="항상성",
                is_present=True,
                similarity=0.47,
                threshold=0.39,
            ),
            ConceptDetail(
                concept="음성되먹임",
                is_present=False,
                similarity=0.20,
                threshold=0.35,
            ),
        ]
    if component_scores is None:
        component_scores = {
            "concept_coverage": 0.17,
            "llm_rubric": 0.5,
            "rasch_ability": 0.0,
        }
    if misconceptions is None:
        misconceptions = ["삼투와 확산 혼동"]
    return QuestionReportData(
        question_sn=question_sn,
        question_text=question_text,
        model_answer=model_answer,
        student_answer=student_answer,
        concept_coverage=concept_coverage,
        concepts=concepts,
        llm_median_score=llm_median_score,
        llm_label=llm_label,
        ensemble_score=ensemble_score,
        understanding_level=understanding_level,
        component_scores=component_scores,
        feedback_text=feedback_text,
        misconceptions=misconceptions,
    )


def _make_distributions() -> ClassDistributions:
    """Build a minimal ClassDistributions for testing."""
    return ClassDistributions(
        ensemble_scores={1: [0.19, 0.26, 0.55, 0.75, 0.90]},
        concept_coverages={1: [0.0, 0.17, 0.50, 0.80, 1.0]},
        llm_scores={1: [1.0, 2.0, 2.0, 3.0, 3.0]},
        rasch_thetas={1: [-4.85, -2.0, 0.0, 1.5, 3.0]},
        component_scores={
            1: {
                "concept_coverage": [0.0, 0.17, 0.50, 0.80, 1.0],
                "llm_rubric": [0.33, 0.50, 0.67, 0.83, 1.0],
                "rasch_ability": [0.0, 0.15, 0.50, 0.65, 1.0],
            },
        },
        overall_ensemble=[0.19, 0.26, 0.55, 0.75, 0.90],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_font(tmp_path):
    """Create a fake font file for testing."""
    font_file = tmp_path / "FakeFont.ttf"
    font_file.write_bytes(b"\x00" * 64)
    return str(font_file)


@pytest.fixture()
def generator(mock_font):
    """Create a StudentPDFReportGenerator with mocked font registration."""
    with patch("forma.student_report.find_korean_font", return_value=mock_font):
        with patch("forma.student_report.pdfmetrics.registerFont"):
            with patch("forma.student_report.TTFont"):
                with patch("forma.student_report.ReportChartGenerator"):
                    from forma.student_report import StudentPDFReportGenerator

                    return StudentPDFReportGenerator(font_path=mock_font)


# ===========================================================================
# T012: StudentPDFReportGenerator tests
# ===========================================================================


class TestGeneratorInit:
    """T012-1: Font registration on init."""

    def test_generator_init_registers_font(self, mock_font):
        """StudentPDFReportGenerator registers NanumGothic font on init."""
        with patch(
            "forma.student_report.find_korean_font", return_value=mock_font,
        ):
            with patch(
                "forma.student_report.pdfmetrics.registerFont",
            ) as mock_register:
                with patch("forma.student_report.TTFont") as mock_ttfont:
                    from forma.student_report import StudentPDFReportGenerator

                    StudentPDFReportGenerator(font_path=mock_font)

                    # TTFont should have been called with font name and path
                    mock_ttfont.assert_called()
                    # registerFont should have been called at least once
                    assert mock_register.call_count >= 1


class TestGeneratePDF:
    """T012-2, T012-3: generate_pdf integration tests."""

    def test_generate_pdf_builds_story(self, generator, tmp_path):
        """generate_pdf calls doc.build() with a story list."""
        import io as _io

        student = _make_student()
        distributions = _make_distributions()
        output_dir = str(tmp_path / "reports")
        os.makedirs(output_dir, exist_ok=True)

        mock_chart = MagicMock()
        dummy = _io.BytesIO(b"\x89PNG\x00" * 10)
        mock_chart.score_boxplot.return_value = dummy
        mock_chart.component_comparison.return_value = dummy
        mock_chart.concept_coverage_bar.return_value = dummy
        mock_chart.understanding_badge.return_value = dummy
        generator._chart_gen = mock_chart

        with patch("forma.student_report.SimpleDocTemplate") as mock_doc_cls:
            with patch("forma.student_report.Image"):
                with patch("forma.student_report.Paragraph"):
                    with patch("forma.student_report.Spacer"):
                        with patch("forma.student_report.PageBreak"):
                            with patch("forma.student_report.Table"):
                                mock_doc = MagicMock()
                                mock_doc_cls.return_value = mock_doc

                                generator.generate_pdf(
                                    student_data=student,
                                    distributions=distributions,
                                    output_dir=output_dir,
                                )

                                # doc.build() must be called once
                                mock_doc.build.assert_called_once()
                                story_arg = mock_doc.build.call_args[0][0]
                                assert isinstance(story_arg, list)
                                assert len(story_arg) > 0

    def test_generate_pdf_creates_output_dir(self, generator, tmp_path):
        """generate_pdf creates the output directory if it does not exist."""
        import io as _io

        student = _make_student()
        distributions = _make_distributions()
        output_dir = str(tmp_path / "deep" / "nested" / "reports")

        mock_chart = MagicMock()
        dummy = _io.BytesIO(b"\x89PNG\x00" * 10)
        mock_chart.score_boxplot.return_value = dummy
        mock_chart.component_comparison.return_value = dummy
        mock_chart.concept_coverage_bar.return_value = dummy
        mock_chart.understanding_badge.return_value = dummy
        generator._chart_gen = mock_chart

        with patch("forma.student_report.SimpleDocTemplate") as mock_doc_cls:
            with patch("forma.student_report.Image"):
                with patch("forma.student_report.Paragraph"):
                    with patch("forma.student_report.Spacer"):
                        with patch("forma.student_report.PageBreak"):
                            with patch("forma.student_report.Table"):
                                mock_doc = MagicMock()
                                mock_doc_cls.return_value = mock_doc

                                generator.generate_pdf(
                                    student_data=student,
                                    distributions=distributions,
                                    output_dir=output_dir,
                                )

                                assert os.path.isdir(output_dir)


class TestMakeOutputFilename:
    """T012-4, T012-5: Output filename format."""

    def test_make_output_filename(self, generator):
        """Filename follows the {분반코드}_w{주차}_{학번}_{이름}.pdf pattern."""
        from forma.student_report import StudentPDFReportGenerator

        student = _make_student(
            real_name="이유정",
            student_number="2026194126",
            class_name="A반",
            week_num=1,
        )
        output_dir = "/tmp/reports"

        result = StudentPDFReportGenerator._make_output_filename(
            student, output_dir,
        )
        expected = os.path.join(output_dir, "1A_w1_2026194126_이유정.pdf")
        assert result == expected

    def test_make_output_filename_class_name_extraction_A(self, generator):
        """class_name='A반' produces '1A' prefix."""
        from forma.student_report import StudentPDFReportGenerator

        student = _make_student(class_name="A반", week_num=1)
        result = StudentPDFReportGenerator._make_output_filename(
            student, "/tmp",
        )
        filename = os.path.basename(result)
        assert filename.startswith("1A_w1_")

    def test_make_output_filename_class_name_extraction_B(self, generator):
        """class_name='B반' produces '1B' prefix."""
        from forma.student_report import StudentPDFReportGenerator

        student = _make_student(class_name="B반", week_num=1)
        result = StudentPDFReportGenerator._make_output_filename(
            student, "/tmp",
        )
        filename = os.path.basename(result)
        assert filename.startswith("1B_w1_")


class TestSanitizeFilename:
    """T012-6, T012-7: Filename sanitization."""

    def test_sanitize_filename(self):
        """Characters illegal in filenames are replaced with underscores."""
        from forma.student_report import _sanitize_filename

        result = _sanitize_filename('이름/with:special')
        assert result == "이름_with_special"

    def test_sanitize_filename_clean(self):
        """Clean Korean name passes through unchanged."""
        from forma.student_report import _sanitize_filename

        result = _sanitize_filename("이유정")
        assert result == "이유정"


class TestParseFeedbackSections:
    """T012-8, T012-9, T012-10: Feedback text parsing."""

    def test_parse_feedback_sections_all(self):
        """Full feedback text with all three sections is parsed correctly."""
        from forma.student_report import parse_feedback_sections

        text = (
            "[평가 요약]\n항상성 개념 부분 이해.\n"
            "[분석 결과]\n세부 기전 설명 부족.\n"
            "[학습 제안]\n교과서 3장 복습 권장."
        )
        result = parse_feedback_sections(text)

        assert "평가 요약" in result
        assert "분석 결과" in result
        assert "학습 제안" in result
        assert "항상성 개념 부분 이해" in result["평가 요약"]
        assert "세부 기전 설명 부족" in result["분석 결과"]
        assert "교과서 3장 복습 권장" in result["학습 제안"]

    def test_parse_feedback_sections_partial(self):
        """Feedback with only [평가 요약] returns dict with at least that key."""
        from forma.student_report import parse_feedback_sections

        text = "[평가 요약]\n항상성 개념 이해가 부족합니다."
        result = parse_feedback_sections(text)

        assert "평가 요약" in result
        assert "항상성 개념 이해가 부족합니다" in result["평가 요약"]

    def test_parse_feedback_sections_empty(self):
        """Empty text returns an empty dict or dict with empty values."""
        from forma.student_report import parse_feedback_sections

        result = parse_feedback_sections("")
        assert isinstance(result, dict)
        # Either returns empty dict or dict with empty values
        for value in result.values():
            assert value == "" or value is None


class TestBuildHeaderSection:
    """T012-11: Header section includes student info."""

    def test_build_header_includes_student_info(self, generator):
        """Header story elements contain student name and student number."""
        student = _make_student(
            real_name="이유정",
            student_number="2026194126",
            class_name="A반",
        )

        with patch("forma.student_report.Paragraph") as mock_para:
            with patch("forma.student_report.Table") as mock_table:
                with patch("forma.student_report.Spacer"):
                    story = generator._build_header_section(student)

        # story should be a list of flowables
        assert isinstance(story, list)
        assert len(story) > 0

        # Verify student info appears in the rendered content.
        # Check all Paragraph calls and Table calls for student data.
        all_call_args = []
        for call in mock_para.call_args_list:
            if call.args:
                all_call_args.append(str(call.args[0]))

        all_text = " ".join(all_call_args)

        # The header should reference the report title
        title_found = any(
            "학생 개인별 평가 리포트" in arg for arg in all_call_args
        )
        assert title_found, (
            "Header should contain the report title '학생 개인별 평가 리포트'"
        )


class TestBuildQuestionSection:
    """T012-12: Question section includes answer comparison table."""

    def test_build_question_section_includes_answer_table(self, generator):
        """Question section story contains a Table flowable for answers."""
        question = _make_question()
        distributions = _make_distributions()

        # Mock chart generator to return dummy BytesIO
        import io

        dummy_png = io.BytesIO(b"\x89PNG\x00" * 10)

        mock_chart = MagicMock()
        mock_chart.score_boxplot.return_value = io.BytesIO(
            b"\x89PNG\x00" * 10,
        )
        mock_chart.component_comparison.return_value = io.BytesIO(
            b"\x89PNG\x00" * 10,
        )
        mock_chart.concept_coverage_bar.return_value = io.BytesIO(
            b"\x89PNG\x00" * 10,
        )
        mock_chart.understanding_badge.return_value = io.BytesIO(
            b"\x89PNG\x00" * 10,
        )
        generator._chart_gen = mock_chart

        with patch("forma.student_report.Image"):
            with patch("forma.student_report.Paragraph"):
                with patch("forma.student_report.Spacer"):
                    with patch("forma.student_report.PageBreak"):
                        with patch("forma.student_report.Table") as mock_table:
                            story = generator._build_question_section(
                                question, distributions,
                            )

        assert isinstance(story, list)
        assert len(story) > 0
        # Table constructor should have been called (answer comparison)
        assert mock_table.called, (
            "Question section should create a Table for answer comparison"
        )


class TestXmlEscapeInAnswerText:
    """T012-13: XML escape for student answer text with special characters."""

    def test_xml_escape_in_answer_text(self, generator):
        """Student answer containing '<script>' should be XML-escaped."""
        from xml.sax.saxutils import escape

        dangerous_answer = '<script>alert("XSS")</script>'
        question = _make_question(student_answer=dangerous_answer)
        distributions = _make_distributions()

        import io

        mock_chart = MagicMock()
        mock_chart.score_boxplot.return_value = io.BytesIO(
            b"\x89PNG\x00" * 10,
        )
        mock_chart.component_comparison.return_value = io.BytesIO(
            b"\x89PNG\x00" * 10,
        )
        mock_chart.concept_coverage_bar.return_value = io.BytesIO(
            b"\x89PNG\x00" * 10,
        )
        mock_chart.understanding_badge.return_value = io.BytesIO(
            b"\x89PNG\x00" * 10,
        )

        with patch("forma.student_report.Image"):
            with patch("forma.student_report.Paragraph") as mock_para:
                with patch.object(
                    generator,
                    "_chart_gen",
                    mock_chart,
                    create=True,
                ):
                    story = generator._build_question_section(
                        question, distributions,
                    )

        # Collect all text passed to Paragraph calls
        all_para_texts = []
        for call in mock_para.call_args_list:
            if call.args:
                all_para_texts.append(str(call.args[0]))

        # The raw '<script>' must NOT appear unescaped in any Paragraph
        escaped_form = escape(dangerous_answer)
        raw_in_paragraphs = any(
            "<script>" in text for text in all_para_texts
        )
        escaped_in_paragraphs = any(
            "&lt;script&gt;" in text for text in all_para_texts
        )

        # Either the raw tag should be absent, or the escaped form should
        # appear.  ReportLab's Paragraph uses XML, so unescaped angle
        # brackets will cause parsing errors.
        assert not raw_in_paragraphs or escaped_in_paragraphs, (
            "Student answer text with '<script>' must be XML-escaped "
            "before being passed to Paragraph"
        )


# ===========================================================================
# T030 [US3]: parse_feedback_sections edge cases
# ===========================================================================


class TestParseFeedbackEdgeCases:
    """T030: Additional edge cases for parse_feedback_sections."""

    def test_missing_sections_fallback(self):
        """Text with only some sections returns dict with available ones."""
        from forma.student_report import parse_feedback_sections

        text = "[평가 요약]\n개념 이해 부족.\n[학습 제안]\n복습 권장."
        result = parse_feedback_sections(text)
        assert "평가 요약" in result
        assert "학습 제안" in result

    def test_long_feedback_text(self):
        """Feedback over 2000 characters is parsed without truncation."""
        from forma.student_report import parse_feedback_sections

        long_content = "가" * 700
        text = (
            f"[평가 요약]\n{long_content}\n"
            f"[분석 결과]\n{long_content}\n"
            f"[학습 제안]\n{long_content}"
        )
        assert len(text) > 2000
        result = parse_feedback_sections(text)
        assert len(result["평가 요약"]) == 700
        assert len(result["분석 결과"]) == 700
        assert len(result["학습 제안"]) == 700

    def test_no_section_markers(self):
        """Plain text without markers returns full text under generic key."""
        from forma.student_report import parse_feedback_sections

        text = "이 학생은 전반적으로 개념 이해도가 낮습니다."
        result = parse_feedback_sections(text)
        assert "전체 피드백" in result
        assert "이 학생은" in result["전체 피드백"]

    def test_placeholder_feedback(self):
        """Placeholder feedback text returns generic key."""
        from forma.student_report import parse_feedback_sections

        result = parse_feedback_sections("(피드백 데이터 없음)")
        assert isinstance(result, dict)


# ===========================================================================
# T031 [US3]: XML escaping edge cases
# ===========================================================================


class TestXmlEscapeEdgeCases:
    """T031: XML escaping for OCR artifacts in student answers."""

    def test_escape_ampersand(self):
        """Ampersand in text is escaped to &amp;."""
        from forma.student_report import _esc

        assert "&amp;" in _esc("A & B")

    def test_escape_angle_brackets(self):
        """Angle brackets are escaped to &lt; and &gt;."""
        from forma.student_report import _esc

        result = _esc("<tag>content</tag>")
        assert "&lt;" in result
        assert "&gt;" in result

    def test_broken_unicode_no_crash(self):
        """Broken unicode characters don't crash the escaper."""
        from forma.student_report import _esc

        # Unicode replacement character and other edge chars
        text = "답안 \ufffd 텍스트 \u0000"
        result = _esc(text)
        assert isinstance(result, str)
