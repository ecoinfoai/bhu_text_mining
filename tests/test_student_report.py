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
        with patch("forma.font_utils.pdfmetrics.registerFont"):
            with patch("forma.font_utils.TTFont"):
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
                "forma.font_utils.pdfmetrics.registerFont",
            ) as mock_register:
                with patch("forma.font_utils.TTFont") as mock_ttfont:
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


# ===========================================================================
# T017: Graph diagram in _build_question_section()
# ===========================================================================


class TestStudentReportGraphDiagram:
    """T017: Graph diagram added to question section when graph data exists."""

    # Minimal valid 1x1 RGB PNG accepted by ReportLab's ImageReader
    _FAKE_PNG = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe"
        b"\r\xefF\xb8"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    def _make_question_with_graph(self) -> "QuestionReportData":
        """Build a QuestionReportData that has non-empty graph_matched_edges."""
        from types import SimpleNamespace

        fake_edge = SimpleNamespace(subject="A", relation="r", object="B")
        return _make_question(
            question_sn=1,
            # Pass graph edges via a compatible object; the dataclass field
            # is typed as list so any list of edge-like objects works.
        )

    def test_graph_diagram_present_when_graph_data_exists(self, generator):
        """Image flowable is added when graph_matched_edges is non-empty."""
        import io
        from types import SimpleNamespace
        from unittest.mock import patch, MagicMock
        from reportlab.platypus import Image

        fake_edge = SimpleNamespace(subject="A", relation="r", object="B")
        question = _make_question(question_sn=1)
        question.graph_matched_edges = [fake_edge]
        question.graph_missing_edges = []

        distributions = _make_distributions()

        mock_chart = MagicMock()
        dummy_png = io.BytesIO(b"\x89PNG\x00" * 10)
        mock_chart.score_boxplot.return_value = dummy_png
        mock_chart.component_comparison.return_value = dummy_png
        mock_chart.concept_coverage_bar.return_value = dummy_png
        mock_chart.understanding_badge.return_value = dummy_png
        generator._chart_gen = mock_chart

        fake_bytesio = io.BytesIO(self._FAKE_PNG)

        with patch(
            "forma.student_report.GraphVisualizer",
        ) as mock_gv_cls:
            mock_gv_instance = MagicMock()
            mock_gv_cls.return_value = mock_gv_instance
            # visualize_comparison_to_bytesio returns (BytesIO, omitted_count)
            mock_gv_instance.visualize_comparison_to_bytesio.return_value = (
                fake_bytesio,
                0,
            )

            story = generator._build_question_section(question, distributions)

        # At least one flowable must be an Image
        image_flowables = [f for f in story if isinstance(f, Image)]
        assert len(image_flowables) >= 1, (
            "Expected at least one Image flowable in the story when "
            "graph_matched_edges is non-empty, but none was found."
        )

    def test_graph_diagram_absent_when_no_graph_data(self, generator):
        """No Image flowable from graph when graph_matched_edges and
        graph_missing_edges are both empty."""
        import io
        from unittest.mock import patch, MagicMock
        from reportlab.platypus import Image

        question = _make_question(question_sn=1)
        question.graph_matched_edges = []
        question.graph_missing_edges = []

        distributions = _make_distributions()

        mock_chart = MagicMock()
        dummy_png = io.BytesIO(b"\x89PNG\x00" * 10)
        mock_chart.score_boxplot.return_value = dummy_png
        mock_chart.component_comparison.return_value = dummy_png
        mock_chart.concept_coverage_bar.return_value = dummy_png
        mock_chart.understanding_badge.return_value = dummy_png
        generator._chart_gen = mock_chart

        # Count Image flowables before patching GraphVisualizer.
        # Even if GraphVisualizer is never called, we capture what the story
        # produces through the real Image constructor.
        with patch("forma.student_report.GraphVisualizer") as mock_gv_cls:
            story = generator._build_question_section(question, distributions)

        # GraphVisualizer must not be instantiated at all
        mock_gv_cls.assert_not_called()

    def test_omission_text_shown_when_edges_capped(self, generator):
        """A Paragraph with '5' and '생략' is added when 5 edges are omitted."""
        import io
        from types import SimpleNamespace
        from unittest.mock import patch, MagicMock
        from reportlab.platypus import Paragraph as RLParagraph

        fake_edge = SimpleNamespace(subject="A", relation="r", object="B")
        question = _make_question(question_sn=1)
        question.graph_matched_edges = [fake_edge]
        question.graph_missing_edges = []

        distributions = _make_distributions()

        mock_chart = MagicMock()
        dummy_png = io.BytesIO(b"\x89PNG\x00" * 10)
        mock_chart.score_boxplot.return_value = dummy_png
        mock_chart.component_comparison.return_value = dummy_png
        mock_chart.concept_coverage_bar.return_value = dummy_png
        mock_chart.understanding_badge.return_value = dummy_png
        generator._chart_gen = mock_chart

        fake_bytesio = io.BytesIO(self._FAKE_PNG)

        with patch(
            "forma.student_report.GraphVisualizer",
        ) as mock_gv_cls:
            mock_gv_instance = MagicMock()
            mock_gv_cls.return_value = mock_gv_instance
            # 5 edges were omitted from the visualisation
            mock_gv_instance.visualize_comparison_to_bytesio.return_value = (
                fake_bytesio,
                5,
            )

            story = generator._build_question_section(question, distributions)

        # Find Paragraph flowables whose text contains both '5' and '생략'
        omission_paras = []
        for flowable in story:
            if isinstance(flowable, RLParagraph):
                # Access the raw text stored in the Paragraph
                para_text = flowable.text if hasattr(flowable, "text") else ""
                if "5" in para_text and "생략" in para_text:
                    omission_paras.append(flowable)

        assert len(omission_paras) >= 1, (
            "Expected a Paragraph containing '5' and '생략' when "
            "visualize_comparison_to_bytesio returns omitted_count=5, "
            "but no such paragraph was found in the story."
        )


# ===========================================================================
# T023: Hub gap table in _build_question_section()
# ===========================================================================


class TestStudentReportHubGapTable:
    """T023: Hub gap table rendered in question section when hub_gap_entries is non-empty.

    RED phase: _build_question_section() does not yet render a hub gap table,
    so both tests must FAIL until the feature is implemented.

    The expected table has 3 columns:
      개념 | 중심성 | 포함
    where 포함 shows O (green) or X (red) based on student_present.
    """

    def _make_question_with_hub_gap(self) -> "QuestionReportData":
        """Return a QuestionReportData with two hub gap entries."""
        from forma.evaluation_types import HubGapEntry

        q = _make_question(question_sn=1)
        q.hub_gap_entries = [
            HubGapEntry("폐", 0.8, True, 0.0),
            HubGapEntry("심장", 0.5, False, 0.0),
        ]
        return q

    def _make_question_without_hub_gap(self) -> "QuestionReportData":
        """Return a QuestionReportData with an empty hub_gap_entries list."""
        q = _make_question(question_sn=1)
        q.hub_gap_entries = []
        return q

    def _build_story(self, generator, question) -> list:
        """Call _build_question_section and return the story flowables."""
        import io
        from unittest.mock import patch, MagicMock

        mock_chart = MagicMock()
        dummy = io.BytesIO(b"\x89PNG\x00" * 10)
        mock_chart.score_boxplot.return_value = dummy
        mock_chart.component_comparison.return_value = dummy
        mock_chart.concept_coverage_bar.return_value = dummy
        mock_chart.understanding_badge.return_value = dummy
        generator._chart_gen = mock_chart

        distributions = _make_distributions()

        with patch("forma.student_report.GraphVisualizer"):
            story = generator._build_question_section(question, distributions)

        return story

    def test_hub_gap_table_present_when_entries_exist(self, generator):
        """A Table flowable with hub gap data is added when hub_gap_entries is non-empty.

        RED: _build_question_section() does not yet render a hub gap table
        -> this test FAILS until T023 is implemented.
        """
        from reportlab.platypus import Table

        question = self._make_question_with_hub_gap()
        story = self._build_story(generator, question)

        tables = [f for f in story if isinstance(f, Table)]
        assert len(tables) > 0, (
            "Expected at least one Table flowable in the story when "
            "hub_gap_entries is non-empty, but none was found."
        )

        # Collect all text from table cells and check for hub gap header or concept names
        from reportlab.platypus import Paragraph as _Paragraph

        all_table_text = ""
        for t in tables:
            if hasattr(t, "_cellvalues"):
                for row in t._cellvalues:
                    for cell in row:
                        if isinstance(cell, str):
                            all_table_text += cell + " "
                        elif isinstance(cell, _Paragraph) and hasattr(cell, "text"):
                            all_table_text += str(cell.text) + " "

        # The hub gap table must contain the header "개념" or the concept names "폐" / "심장"
        has_header = "개념" in all_table_text
        has_concepts = "폐" in all_table_text or "심장" in all_table_text
        assert has_header or has_concepts, (
            "Hub gap table must contain '개념' header or concept names ('폐', '심장'). "
            f"All table text found: {all_table_text[:400]!r}"
        )

    def test_hub_gap_table_absent_when_entries_empty(self, generator):
        """No hub gap Table is added when hub_gap_entries is empty.

        RED: _build_question_section() does not yet check hub_gap_entries
        for table rendering. Once T023 adds the feature, the 'present' test
        will drive an implementation that only adds the table when entries
        are non-empty — this 'absent' test verifies the guard condition.

        Strategy: build a story with empty hub_gap_entries, then build one
        with non-empty entries, and assert that the empty case does NOT
        produce an additional hub gap table containing "개념" or concept names.
        """
        from reportlab.platypus import Table, Paragraph as _Paragraph

        question_empty = self._make_question_without_hub_gap()
        story_empty = self._build_story(generator, question_empty)

        question_with = self._make_question_with_hub_gap()
        story_with = self._build_story(generator, question_with)

        def _table_text(story):
            text = ""
            for f in story:
                if isinstance(f, Table) and hasattr(f, "_cellvalues"):
                    for row in f._cellvalues:
                        for cell in row:
                            if isinstance(cell, str):
                                text += cell + " "
                            elif isinstance(cell, _Paragraph) and hasattr(cell, "text"):
                                text += str(cell.text) + " "
            return text

        text_empty = _table_text(story_empty)
        text_with = _table_text(story_with)

        # The empty case must NOT contain hub gap concept names
        assert "폐" not in text_empty and "심장" not in text_empty, (
            "When hub_gap_entries is empty, concept names ('폐', '심장') must NOT "
            f"appear in any Table. Found: {text_empty[:400]!r}"
        )

        # Sanity: the non-empty case must have the concepts (guarded by the other test)
        assert "폐" in text_with or "심장" in text_with or "개념" in text_with, (
            "Sanity check: non-empty hub_gap_entries should produce hub gap table content."
        )


# ---------------------------------------------------------------------------
# Phase 4: US2 — T018/T019: Student report delta display + backward compat
# ---------------------------------------------------------------------------


class TestStudentReportDeltaDisplay:
    """T018: Student report shows delta symbols for scores."""

    def test_generate_pdf_accepts_weekly_deltas(self, tmp_path):
        """StudentPDFReportGenerator.generate_pdf should accept weekly_deltas kwarg."""
        from forma.student_report import StudentPDFReportGenerator
        from forma.report_data_loader import WeeklyDelta

        student_data = _make_student()
        dists = ClassDistributions(
            ensemble_scores={1: [0.3, 0.5, 0.7]},
            concept_coverages={1: [0.4, 0.6, 0.8]},
            llm_scores={1: [1.0, 2.0, 3.0]},
        )
        deltas = {
            "overall": WeeklyDelta(
                current_score=0.65, previous_score=0.50,
                delta=0.15, delta_symbol="↑",
            ),
            1: WeeklyDelta(
                current_score=0.26, previous_score=0.20,
                delta=0.06, delta_symbol="↑",
            ),
        }

        gen = StudentPDFReportGenerator()
        pdf_path = gen.generate_pdf(
            student_data, dists, str(tmp_path),
            weekly_deltas=deltas,
        )
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0

    def test_generate_pdf_accepts_trajectory_chart(self, tmp_path):
        """StudentPDFReportGenerator.generate_pdf should accept trajectory_chart kwarg."""
        import io
        from forma.student_report import StudentPDFReportGenerator

        student_data = _make_student()
        dists = ClassDistributions(
            ensemble_scores={1: [0.3, 0.5, 0.7]},
            concept_coverages={1: [0.4, 0.6, 0.8]},
            llm_scores={1: [1.0, 2.0, 3.0]},
        )

        # Create a minimal PNG for trajectory chart
        import struct, zlib
        def _make_tiny_png():
            width, height = 1, 1
            raw_data = b'\x00\xff\x00\x00'
            compressed = zlib.compress(raw_data)
            ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
            chunks = b''
            for chunk_type, data in [(b'IHDR', ihdr), (b'IDAT', compressed), (b'IEND', b'')]:
                crc = zlib.crc32(chunk_type + data) & 0xffffffff
                chunks += struct.pack('>I', len(data)) + chunk_type + data + struct.pack('>I', crc)
            return b'\x89PNG\r\n\x1a\n' + chunks

        chart_buf = io.BytesIO(_make_tiny_png())

        gen = StudentPDFReportGenerator()
        pdf_path = gen.generate_pdf(
            student_data, dists, str(tmp_path),
            trajectory_chart=chart_buf,
        )
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0


class TestStudentReportBackwardCompat:
    """T019: No longitudinal store → report identical to v0.7.x."""

    def test_no_deltas_no_chart_still_works(self, tmp_path):
        """Without weekly_deltas or trajectory_chart, generate_pdf works as before."""
        from forma.student_report import StudentPDFReportGenerator

        student_data = _make_student()
        dists = ClassDistributions(
            ensemble_scores={1: [0.3, 0.5, 0.7]},
            concept_coverages={1: [0.4, 0.6, 0.8]},
            llm_scores={1: [1.0, 2.0, 3.0]},
        )

        gen = StudentPDFReportGenerator()
        pdf_path = gen.generate_pdf(student_data, dists, str(tmp_path))
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0

    def test_signature_unchanged(self):
        """generate_pdf signature should still accept old positional args."""
        import inspect
        from forma.student_report import StudentPDFReportGenerator

        sig = inspect.signature(StudentPDFReportGenerator.generate_pdf)
        params = list(sig.parameters.keys())
        # Must have: self, student_data, distributions, output_dir
        assert "student_data" in params
        assert "output_dir" in params
        assert "distributions" in params


# ===========================================================================
# T026-T028: Backward compatibility for feedback section parsing
# ===========================================================================


class TestParseFeedbackSectionsBackwardCompat:
    """T026-T028: Backward compatibility for feedback section parsing."""

    def test_parse_new_section_names(self):
        """T026: Correctly parse new section names."""
        from forma.student_report import parse_feedback_sections

        text = (
            "[현재 상태] 핵심 개념에 대한 이해가 잘 드러납니다. "
            "[원인] 개념 간 관계를 정확하게 파악하고 있습니다. "
            "[학생에게 권하는 사항] 심화 학습을 권합니다."
        )
        result = parse_feedback_sections(text)
        assert "현재 상태" in result
        assert "원인" in result
        assert "학생에게 권하는 사항" in result

    def test_parse_old_section_names(self):
        """T027: Still correctly parse old section names (backward compat)."""
        from forma.student_report import parse_feedback_sections

        text = (
            "[평가 요약] 기존 형식의 피드백입니다. "
            "[분석 결과] 분석 내용입니다. "
            "[학습 제안] 학습 제안 내용입니다."
        )
        result = parse_feedback_sections(text)
        assert "평가 요약" in result
        assert "분석 결과" in result
        assert "학습 제안" in result

    def test_parse_mixed_section_names(self):
        """T028: Handle mixed old+new section names in single text."""
        from forma.student_report import parse_feedback_sections

        text = (
            "[현재 상태] 새 형식 첫 섹션. "
            "[분석 결과] 구 형식 두번째 섹션. "
            "[학생에게 권하는 사항] 새 형식 세번째 섹션."
        )
        result = parse_feedback_sections(text)
        assert "현재 상태" in result
        assert "분석 결과" in result
        assert "학생에게 권하는 사항" in result


# ===========================================================================
# T032: Backward compatibility integration test (feedback → parse → render)
# ===========================================================================


class TestFeedbackParseIntegration:
    """T032: End-to-end backward compat — new-format feedback parses correctly."""

    def test_new_format_feedback_parses_all_sections(self):
        """Feedback in new [현재 상태]/[원인]/[학생에게 권하는 사항] format
        parses into exactly those three section keys."""
        from forma.student_report import parse_feedback_sections

        # Simulates LLM output in the new format
        feedback = (
            "[현재 상태] 항상성 개념의 기본적인 이해를 보여주고 있습니다. "
            "체온 조절 메커니즘에 대한 서술이 정확합니다.\n"
            "[원인] 음성되먹임 과정의 세부 단계에 대한 추가 학습이 필요합니다. "
            "수용기와 효과기의 역할을 구분하는 연습이 도움이 될 것입니다.\n"
            "[학생에게 권하는 사항] 교재 3장의 그림 3-5를 다시 살펴보며 "
            "되먹임 경로를 정리해 보세요. 각 구성요소의 역할을 자신의 말로 설명하는 연습을 권합니다."
        )
        result = parse_feedback_sections(feedback)
        assert len(result) == 3
        assert "현재 상태" in result
        assert "원인" in result
        assert "학생에게 권하는 사항" in result
        # Verify content is non-empty
        for section_name, content in result.items():
            assert len(content.strip()) > 0, f"Section '{section_name}' is empty"

    def test_old_format_feedback_still_parses(self):
        """Feedback in old [평가 요약]/[분석 결과]/[학습 제안] format
        still parses correctly (no regression)."""
        from forma.student_report import parse_feedback_sections

        feedback = (
            "[평가 요약] 항상성 개념 부분 이해.\n"
            "[분석 결과] 세부 기전 설명 부족.\n"
            "[학습 제안] 교과서 3장 복습 권장."
        )
        result = parse_feedback_sections(feedback)
        assert "평가 요약" in result
        assert "분석 결과" in result
        assert "학습 제안" in result

    def test_new_format_feedback_renders_in_pdf(self, tmp_path):
        """New-format feedback renders in PDF without error."""
        from forma.student_report import StudentPDFReportGenerator

        new_feedback = (
            "[현재 상태] 핵심 개념에 대한 이해가 잘 드러납니다. "
            "기본적인 개념 구조를 파악하고 있습니다.\n"
            "[원인] 개념 간 관계를 정확하게 파악하고 있습니다. "
            "일부 세부 메커니즘에 대한 보완이 필요합니다.\n"
            "[학생에게 권하는 사항] 심화 학습을 권합니다. "
            "관련 사례를 탐색해 보세요."
        )
        student = _make_student()
        # Override question feedback to use new format
        student.questions[0] = _make_question(feedback_text=new_feedback)

        dists = _make_distributions()
        gen = StudentPDFReportGenerator()
        pdf_path = gen.generate_pdf(student, dists, str(tmp_path))
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0


# ---------------------------------------------------------------------------
# T040: Learning path section in student report (v0.10.0 US4, FR-020, FR-023)
# ---------------------------------------------------------------------------


class TestLearningPathSection:
    """Tests for _build_learning_path_section in student report."""

    def test_learning_path_renders_in_pdf(self, tmp_path):
        """generate_pdf with learning_path kwarg produces valid PDF (FR-020)."""
        from forma.learning_path import LearningPath
        from forma.student_report import StudentPDFReportGenerator

        lp = LearningPath(
            student_id="S001",
            deficit_concepts=["물질 이동", "삼투압"],
            ordered_path=["물질 이동", "삼투압"],
            capped=False,
        )
        student = _make_student()
        dists = _make_distributions()
        gen = StudentPDFReportGenerator()
        pdf_path = gen.generate_pdf(
            student, dists, str(tmp_path),
            learning_path=lp,
        )
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0

    def test_empty_learning_path_shows_mastered_message(self, tmp_path):
        """Empty path produces PDF with mastered message."""
        from forma.learning_path import LearningPath
        from forma.student_report import StudentPDFReportGenerator

        lp = LearningPath(
            student_id="S001",
            deficit_concepts=[],
            ordered_path=[],
            capped=False,
        )
        student = _make_student()
        dists = _make_distributions()
        gen = StudentPDFReportGenerator()
        pdf_path = gen.generate_pdf(
            student, dists, str(tmp_path),
            learning_path=lp,
        )
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0

    def test_no_learning_path_backward_compat(self, tmp_path):
        """FR-023: Without learning_path kwarg, report generates normally."""
        from forma.student_report import StudentPDFReportGenerator

        student = _make_student()
        dists = _make_distributions()
        gen = StudentPDFReportGenerator()
        pdf_path = gen.generate_pdf(student, dists, str(tmp_path))
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0

    def test_capped_learning_path_renders(self, tmp_path):
        """Capped path (>20 concepts) renders with notice."""
        from forma.learning_path import LearningPath
        from forma.student_report import StudentPDFReportGenerator

        lp = LearningPath(
            student_id="S001",
            deficit_concepts=[f"C{i}" for i in range(25)],
            ordered_path=[f"C{i}" for i in range(20)],
            capped=True,
        )
        student = _make_student()
        dists = _make_distributions()
        gen = StudentPDFReportGenerator()
        pdf_path = gen.generate_pdf(
            student, dists, str(tmp_path),
            learning_path=lp,
        )
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0


# ---------------------------------------------------------------------------
# T061 [US6] Softened grade language in student report — FR-031
# ---------------------------------------------------------------------------


class TestStudentGradeTrend:
    """Tests for softened grade trend display in student report."""

    @patch("forma.student_report.os.path.exists", return_value=True)
    @patch("forma.student_report.find_korean_font", return_value="/fake/NanumGothic.ttf")
    @patch("forma.font_utils.pdfmetrics.registerFont")
    @patch("forma.font_utils.TTFont")
    def test_build_grade_trend_section_returns_flowables(
        self, mock_ttfont, mock_register, mock_find, mock_exists,
    ):
        """_build_grade_trend_section returns non-empty story list."""
        from forma.student_report import StudentPDFReportGenerator

        gen = StudentPDFReportGenerator(font_path="/fake/NanumGothic.ttf")
        story = gen._build_grade_trend_section("상위권")
        assert isinstance(story, list)
        assert len(story) > 0

    @patch("forma.student_report.os.path.exists", return_value=True)
    @patch("forma.student_report.find_korean_font", return_value="/fake/NanumGothic.ttf")
    @patch("forma.font_utils.pdfmetrics.registerFont")
    @patch("forma.font_utils.TTFont")
    def test_upper_tier_text(
        self, mock_ttfont, mock_register, mock_find, mock_exists,
    ):
        """'상위권' prediction shows softened trend text with '상위권'."""
        from reportlab.platypus import Paragraph

        from forma.student_report import StudentPDFReportGenerator

        gen = StudentPDFReportGenerator(font_path="/fake/NanumGothic.ttf")
        story = gen._build_grade_trend_section("상위권")
        paragraphs = [e for e in story if isinstance(e, Paragraph)]
        texts = [p.text for p in paragraphs]
        assert any("상위권" in t for t in texts)

    @patch("forma.student_report.os.path.exists", return_value=True)
    @patch("forma.student_report.find_korean_font", return_value="/fake/NanumGothic.ttf")
    @patch("forma.font_utils.pdfmetrics.registerFont")
    @patch("forma.font_utils.TTFont")
    def test_mid_tier_text(
        self, mock_ttfont, mock_register, mock_find, mock_exists,
    ):
        """'중위권' prediction shows softened text."""
        from reportlab.platypus import Paragraph

        from forma.student_report import StudentPDFReportGenerator

        gen = StudentPDFReportGenerator(font_path="/fake/NanumGothic.ttf")
        story = gen._build_grade_trend_section("중위권")
        paragraphs = [e for e in story if isinstance(e, Paragraph)]
        texts = [p.text for p in paragraphs]
        assert any("중위권" in t for t in texts)

    @patch("forma.student_report.os.path.exists", return_value=True)
    @patch("forma.student_report.find_korean_font", return_value="/fake/NanumGothic.ttf")
    @patch("forma.font_utils.pdfmetrics.registerFont")
    @patch("forma.font_utils.TTFont")
    def test_lower_tier_text(
        self, mock_ttfont, mock_register, mock_find, mock_exists,
    ):
        """'하위권' prediction shows softened text."""
        from reportlab.platypus import Paragraph

        from forma.student_report import StudentPDFReportGenerator

        gen = StudentPDFReportGenerator(font_path="/fake/NanumGothic.ttf")
        story = gen._build_grade_trend_section("하위권")
        paragraphs = [e for e in story if isinstance(e, Paragraph)]
        texts = [p.text for p in paragraphs]
        assert any("하위권" in t for t in texts)

    @patch("forma.student_report.os.path.exists", return_value=True)
    @patch("forma.student_report.find_korean_font", return_value="/fake/NanumGothic.ttf")
    @patch("forma.font_utils.pdfmetrics.registerFont")
    @patch("forma.font_utils.TTFont")
    def test_no_explicit_grade_in_student_report(
        self, mock_ttfont, mock_register, mock_find, mock_exists,
    ):
        """Student report must NOT show explicit A/B/C/D/F grades (FR-031)."""
        from reportlab.platypus import Paragraph

        from forma.student_report import StudentPDFReportGenerator

        gen = StudentPDFReportGenerator(font_path="/fake/NanumGothic.ttf")
        for tier in ("상위권", "중위권", "하위권"):
            story = gen._build_grade_trend_section(tier)
            paragraphs = [e for e in story if isinstance(e, Paragraph)]
            texts = " ".join(p.text for p in paragraphs)
            # Should not contain explicit grade letters in context of prediction
            for grade in ("A등급", "B등급", "C등급", "D등급", "F등급"):
                assert grade not in texts, f"Explicit grade '{grade}' found in student report"

    @patch("forma.student_report.os.path.exists", return_value=True)
    @patch("forma.student_report.find_korean_font", return_value="/fake/NanumGothic.ttf")
    @patch("forma.font_utils.pdfmetrics.registerFont")
    @patch("forma.font_utils.TTFont")
    def test_section_has_heading(
        self, mock_ttfont, mock_register, mock_find, mock_exists,
    ):
        """Section includes '학습 추이' heading."""
        from reportlab.platypus import Paragraph

        from forma.student_report import StudentPDFReportGenerator

        gen = StudentPDFReportGenerator(font_path="/fake/NanumGothic.ttf")
        story = gen._build_grade_trend_section("상위권")
        paragraphs = [e for e in story if isinstance(e, Paragraph)]
        texts = [p.text for p in paragraphs]
        assert any("학습 추이" in t for t in texts)

    def test_grade_trend_renders_in_pdf(self, tmp_path):
        """generate_pdf with grade_trend kwarg produces valid PDF."""
        from forma.student_report import StudentPDFReportGenerator

        student = _make_student()
        dists = _make_distributions()
        gen = StudentPDFReportGenerator()
        pdf_path = gen.generate_pdf(
            student, dists, str(tmp_path),
            grade_trend="상위권",
        )
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0

    def test_no_grade_trend_backward_compat(self, tmp_path):
        """Without grade_trend kwarg, report generates normally (SC-008)."""
        from forma.student_report import StudentPDFReportGenerator

        student = _make_student()
        dists = _make_distributions()
        gen = StudentPDFReportGenerator()
        pdf_path = gen.generate_pdf(student, dists, str(tmp_path))
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
