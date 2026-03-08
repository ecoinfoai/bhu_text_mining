"""Tests for professor_report.py — professor class summary PDF report generation.

RED phase: these tests are written *before* the module exists and will fail
until ``src/forma/professor_report.py`` is implemented.

Covers task T012:
  - ProfessorPDFReportGenerator.__init__: font registration, style creation
  - _build_cover_page: returns story with Paragraph and Table flowables,
    includes class_name in content
  - _build_summary_section: returns story with Image flowables (charts),
    includes statistics values in Paragraphs

Font discovery and PDF registration are mocked to avoid OS / font dependency
in CI.  No actual PDF files are generated in unit tests.
"""

from __future__ import annotations

import io
from unittest.mock import patch, MagicMock

import pytest

from forma.professor_report_data import (
    ProfessorReportData,
    QuestionClassStats,
    StudentSummaryRow,
)


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

_CANONICAL_LEVELS = ("Advanced", "Proficient", "Developing", "Beginning")


def _make_question_class_stats(
    question_sn: int = 1,
    question_text: str = "항상성의 정의를 서술하시오.",
    topic: str = "항상성",
    ensemble_mean: float = 0.65,
    ensemble_std: float = 0.12,
    ensemble_median: float = 0.67,
) -> QuestionClassStats:
    """Build a minimal QuestionClassStats for testing."""
    return QuestionClassStats(
        question_sn=question_sn,
        question_text=question_text,
        topic=topic,
        ensemble_mean=ensemble_mean,
        ensemble_std=ensemble_std,
        ensemble_median=ensemble_median,
        concept_coverage_mean=0.55,
        llm_score_mean=0.70,
        rasch_theta_mean=-0.3,
        level_distribution={"Advanced": 1, "Proficient": 2, "Developing": 1, "Beginning": 1},
        concept_mastery_rates={"항상성": 0.80, "음성되먹임": 0.45},
        misconception_frequencies=[("삼투와 확산 혼동", 2)],
    )


def _make_student_summary_row(
    student_id: str = "S001",
    real_name: str = "김철수",
    student_number: str = "2026100001",
    overall_ensemble_mean: float = 0.65,
    overall_level: str = "Proficient",
    is_at_risk: bool = False,
) -> StudentSummaryRow:
    """Build a minimal StudentSummaryRow for testing."""
    return StudentSummaryRow(
        student_id=student_id,
        student_number=student_number,
        real_name=real_name,
        overall_ensemble_mean=overall_ensemble_mean,
        overall_level=overall_level,
        per_question_scores={1: 0.65, 2: 0.65},
        per_question_levels={1: overall_level, 2: overall_level},
        per_question_coverages={1: 0.60, 2: 0.55},
        is_at_risk=is_at_risk,
        at_risk_reasons=["종합점수 0.45 미만"] if is_at_risk else [],
        z_score=0.0,
    )


def _make_professor_report_data() -> ProfessorReportData:
    """Build a minimal ProfessorReportData for testing.

    Uses the fixture values specified in the task context:
      class_name="1A", week_num=1, subject="생리학",
      exam_title="Ch01 서론 형성평가"
      n_students=5, n_questions=2
      class_ensemble_mean=0.65, class_ensemble_std=0.12,
      class_ensemble_median=0.67, q1=0.55, q3=0.75
      overall_level_distribution={"Advanced":1,"Proficient":2,"Developing":1,"Beginning":1}
      n_at_risk=1, pct_at_risk=20.0
    """
    question_stats = [
        _make_question_class_stats(question_sn=1),
        _make_question_class_stats(
            question_sn=2,
            question_text="음성 되먹임의 예를 드시오.",
            topic="되먹임",
            ensemble_mean=0.60,
            ensemble_std=0.15,
            ensemble_median=0.62,
        ),
    ]

    student_rows = [
        _make_student_summary_row("S001", "김철수", "2026100001", 0.85, "Advanced"),
        _make_student_summary_row("S002", "이영희", "2026100002", 0.72, "Proficient"),
        _make_student_summary_row("S003", "박민준", "2026100003", 0.65, "Proficient"),
        _make_student_summary_row("S004", "최수진", "2026100004", 0.50, "Developing"),
        _make_student_summary_row("S005", "정하늘", "2026100005", 0.30, "Beginning",
                                  is_at_risk=True),
    ]

    return ProfessorReportData(
        class_name="1A",
        week_num=1,
        subject="생리학",
        exam_title="Ch01 서론 형성평가",
        generation_date="2026-03-08",
        n_students=5,
        n_questions=2,
        class_ensemble_mean=0.65,
        class_ensemble_std=0.12,
        class_ensemble_median=0.67,
        class_ensemble_q1=0.55,
        class_ensemble_q3=0.75,
        overall_level_distribution={"Advanced": 1, "Proficient": 2, "Developing": 1, "Beginning": 1},
        question_stats=question_stats,
        student_rows=student_rows,
        n_at_risk=1,
        pct_at_risk=20.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_font(tmp_path):
    """Create a fake font file for testing."""
    font_file = tmp_path / "NanumGothic.ttf"
    font_file.write_bytes(b"\x00" * 64)
    return str(font_file)


@pytest.fixture()
def report_data() -> ProfessorReportData:
    """Return a minimal ProfessorReportData for testing."""
    return _make_professor_report_data()


@pytest.fixture()
def generator(mock_font):
    """Create a ProfessorPDFReportGenerator with mocked font registration."""
    with patch("forma.professor_report.find_korean_font", return_value=mock_font):
        with patch("forma.professor_report.pdfmetrics.registerFont"):
            with patch("forma.professor_report.TTFont"):
                from forma.professor_report import ProfessorPDFReportGenerator

                return ProfessorPDFReportGenerator(font_path=mock_font)


# ===========================================================================
# TestGeneratorInit: __init__ tests
# ===========================================================================


class TestGeneratorInit:
    """Tests for ProfessorPDFReportGenerator.__init__."""

    def test_instantiation_succeeds_with_mocked_font(self, mock_font):
        """ProfessorPDFReportGenerator instantiates without error with a valid font path."""
        with patch("forma.professor_report.find_korean_font", return_value=mock_font):
            with patch("forma.professor_report.pdfmetrics.registerFont"):
                with patch("forma.professor_report.TTFont"):
                    from forma.professor_report import ProfessorPDFReportGenerator

                    gen = ProfessorPDFReportGenerator(font_path=mock_font)
                    assert gen is not None

    def test_raises_file_not_found_if_font_missing(self, tmp_path):
        """ProfessorPDFReportGenerator raises FileNotFoundError for missing font."""
        nonexistent_font = str(tmp_path / "does_not_exist.ttf")
        with pytest.raises(FileNotFoundError):
            from forma.professor_report import ProfessorPDFReportGenerator

            ProfessorPDFReportGenerator(font_path=nonexistent_font)

    def test_styles_dict_contains_prof_title(self, generator):
        """_styles dict contains 'ProfTitle' style."""
        assert "ProfTitle" in generator._styles

    def test_styles_dict_contains_prof_section(self, generator):
        """_styles dict contains 'ProfSection' style."""
        assert "ProfSection" in generator._styles

    def test_styles_dict_contains_prof_subsection(self, generator):
        """_styles dict contains 'ProfSubsection' style."""
        assert "ProfSubsection" in generator._styles

    def test_styles_dict_contains_prof_body(self, generator):
        """_styles dict contains 'ProfBody' style."""
        assert "ProfBody" in generator._styles

    def test_styles_dict_contains_prof_table_header(self, generator):
        """_styles dict contains 'ProfTableHeader' style."""
        assert "ProfTableHeader" in generator._styles

    def test_styles_dict_contains_prof_table_data(self, generator):
        """_styles dict contains 'ProfTableData' style."""
        assert "ProfTableData" in generator._styles

    def test_level_colors_dict_exists_with_four_keys(self, generator):
        """_LEVEL_COLORS dict exists and contains all 4 canonical understanding levels."""
        # The _LEVEL_COLORS attribute or module-level dict should be accessible
        # Check via the module-level dict or instance attribute
        from forma.professor_report import _LEVEL_COLORS

        assert set(_LEVEL_COLORS.keys()) == set(_CANONICAL_LEVELS)

    def test_registers_nanum_gothic_font(self, mock_font):
        """ProfessorPDFReportGenerator registers NanumGothic font on init."""
        with patch("forma.professor_report.find_korean_font", return_value=mock_font):
            with patch("forma.professor_report.pdfmetrics.registerFont") as mock_register:
                with patch("forma.professor_report.TTFont") as mock_ttfont:
                    from forma.professor_report import ProfessorPDFReportGenerator

                    ProfessorPDFReportGenerator(font_path=mock_font)

                    # TTFont should have been called with font name and path
                    mock_ttfont.assert_called()
                    # registerFont should have been called at least once
                    assert mock_register.call_count >= 1


# ===========================================================================
# TestBuildCoverPage: _build_cover_page tests
# ===========================================================================


class TestBuildCoverPage:
    """Tests for ProfessorPDFReportGenerator._build_cover_page."""

    def test_cover_page_returns_non_empty_list(self, generator, report_data):
        """_build_cover_page returns a non-empty list."""
        story = generator._build_cover_page(report_data)
        assert isinstance(story, list)
        assert len(story) > 0

    def test_cover_page_contains_paragraph_flowable(self, generator, report_data):
        """_build_cover_page includes at least one Paragraph flowable."""
        from reportlab.platypus import Paragraph

        story = generator._build_cover_page(report_data)
        paragraphs = [f for f in story if isinstance(f, Paragraph)]
        assert len(paragraphs) > 0, (
            "_build_cover_page must include at least one Paragraph flowable"
        )

    def test_cover_page_contains_table_flowable(self, generator, report_data):
        """_build_cover_page includes at least one Table flowable for metadata."""
        from reportlab.platypus import Table

        story = generator._build_cover_page(report_data)
        tables = [f for f in story if isinstance(f, Table)]
        assert len(tables) > 0, (
            "_build_cover_page must include a Table flowable for metadata"
        )

    def test_cover_page_class_name_in_story_content(self, generator, report_data):
        """class_name '1A' appears somewhere in the cover page story content."""
        from reportlab.platypus import Paragraph

        story = generator._build_cover_page(report_data)
        paragraphs = [f for f in story if isinstance(f, Paragraph)]

        # Collect text from Paragraph objects
        all_text = " ".join(str(p.text) for p in paragraphs if hasattr(p, "text"))

        assert "1A" in all_text, (
            "class_name '1A' must appear in a Paragraph in the cover page story"
        )

    def test_cover_page_exam_title_in_story_content(self, generator, report_data):
        """exam_title appears somewhere in the cover page story content."""
        from reportlab.platypus import Paragraph

        story = generator._build_cover_page(report_data)
        paragraphs = [f for f in story if isinstance(f, Paragraph)]
        all_text = " ".join(str(p.text) for p in paragraphs if hasattr(p, "text"))

        assert "Ch01 서론 형성평가" in all_text, (
            "exam_title must appear in the cover page story content"
        )

    def test_cover_page_week_num_in_story_content(self, generator, report_data):
        """week_num appears somewhere in the cover page story content."""
        from reportlab.platypus import Paragraph

        story = generator._build_cover_page(report_data)
        paragraphs = [f for f in story if isinstance(f, Paragraph)]
        all_text = " ".join(str(p.text) for p in paragraphs if hasattr(p, "text"))

        # Week number 1 should appear in the story
        assert "1" in all_text, (
            "week_num must appear in the cover page story content"
        )


# ===========================================================================
# TestBuildSummarySection: _build_summary_section tests
# ===========================================================================


class TestBuildSummarySection:
    """Tests for ProfessorPDFReportGenerator._build_summary_section."""

    def _make_mock_chart_gen(self) -> MagicMock:
        """Return a MagicMock chart generator that returns fake PNG BytesIO objects."""
        mock_chart_gen = MagicMock()
        dummy_png = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        # Any method called on the chart generator returns a fresh BytesIO
        def _return_dummy(*args, **kwargs):
            return io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_chart_gen.level_distribution_pie = MagicMock(side_effect=_return_dummy)
        mock_chart_gen.score_distribution_histogram = MagicMock(side_effect=_return_dummy)
        mock_chart_gen.at_risk_summary_bar = MagicMock(side_effect=_return_dummy)
        mock_chart_gen.question_comparison_bar = MagicMock(side_effect=_return_dummy)
        # Fallback for any other method
        mock_chart_gen.configure_mock(**{"return_value": dummy_png})

        return mock_chart_gen

    def test_summary_section_returns_non_empty_list(self, generator, report_data):
        """_build_summary_section returns a non-empty list."""
        mock_chart_gen = self._make_mock_chart_gen()
        generator._chart_gen = mock_chart_gen

        story = generator._build_summary_section(report_data)
        assert isinstance(story, list)
        assert len(story) > 0

    def test_summary_section_contains_paragraph_flowable(self, generator, report_data):
        """_build_summary_section includes at least one Paragraph flowable."""
        from reportlab.platypus import Paragraph

        mock_chart_gen = self._make_mock_chart_gen()
        generator._chart_gen = mock_chart_gen

        story = generator._build_summary_section(report_data)
        paragraphs = [f for f in story if isinstance(f, Paragraph)]
        assert len(paragraphs) > 0, (
            "_build_summary_section must include at least one Paragraph flowable"
        )

    def test_summary_section_contains_image_flowable(self, generator, report_data):
        """_build_summary_section includes at least one Image flowable from charts."""
        from reportlab.platypus import Image

        mock_chart_gen = self._make_mock_chart_gen()
        generator._chart_gen = mock_chart_gen

        story = generator._build_summary_section(report_data)
        images = [f for f in story if isinstance(f, Image)]
        assert len(images) > 0, (
            "_build_summary_section must embed at least one chart Image flowable"
        )

    def test_summary_section_mean_in_story_content(self, generator, report_data):
        """class_ensemble_mean (0.65) appears in Paragraph content of summary section."""
        from reportlab.platypus import Paragraph

        mock_chart_gen = self._make_mock_chart_gen()
        generator._chart_gen = mock_chart_gen

        story = generator._build_summary_section(report_data)
        paragraphs = [f for f in story if isinstance(f, Paragraph)]
        all_text = " ".join(str(p.text) for p in paragraphs if hasattr(p, "text"))

        # class_ensemble_mean=0.65; check that "0.65" appears in text
        assert "0.65" in all_text, (
            "class_ensemble_mean value '0.65' must appear in summary section content"
        )

    def test_summary_section_std_in_story_content(self, generator, report_data):
        """class_ensemble_std (0.12) appears in Paragraph content of summary section."""
        from reportlab.platypus import Paragraph

        mock_chart_gen = self._make_mock_chart_gen()
        generator._chart_gen = mock_chart_gen

        story = generator._build_summary_section(report_data)
        paragraphs = [f for f in story if isinstance(f, Paragraph)]
        all_text = " ".join(str(p.text) for p in paragraphs if hasattr(p, "text"))

        assert "0.12" in all_text, (
            "class_ensemble_std value '0.12' must appear in summary section content"
        )

    def test_summary_section_median_in_story_content(self, generator, report_data):
        """class_ensemble_median (0.67) appears in Paragraph content of summary section."""
        from reportlab.platypus import Paragraph

        mock_chart_gen = self._make_mock_chart_gen()
        generator._chart_gen = mock_chart_gen

        story = generator._build_summary_section(report_data)
        paragraphs = [f for f in story if isinstance(f, Paragraph)]
        all_text = " ".join(str(p.text) for p in paragraphs if hasattr(p, "text"))

        assert "0.67" in all_text, (
            "class_ensemble_median value '0.67' must appear in summary section content"
        )

    def test_summary_section_at_risk_info_in_story(self, generator, report_data):
        """n_at_risk or pct_at_risk appears in summary section content."""
        from reportlab.platypus import Paragraph

        mock_chart_gen = self._make_mock_chart_gen()
        generator._chart_gen = mock_chart_gen

        story = generator._build_summary_section(report_data)
        paragraphs = [f for f in story if isinstance(f, Paragraph)]
        all_text = " ".join(str(p.text) for p in paragraphs if hasattr(p, "text"))

        # n_at_risk=1 or pct_at_risk=20.0 should appear somewhere
        at_risk_mentioned = ("1" in all_text) or ("20" in all_text)
        assert at_risk_mentioned, (
            "At-risk information (n_at_risk or pct_at_risk) must appear in summary section"
        )

    def test_summary_section_n_students_in_story(self, generator, report_data):
        """n_students (5) is referenced in the summary section content."""
        from reportlab.platypus import Paragraph

        mock_chart_gen = self._make_mock_chart_gen()
        generator._chart_gen = mock_chart_gen

        story = generator._build_summary_section(report_data)
        paragraphs = [f for f in story if isinstance(f, Paragraph)]
        all_text = " ".join(str(p.text) for p in paragraphs if hasattr(p, "text"))

        assert "5" in all_text, (
            "n_students value '5' must appear somewhere in summary section content"
        )


# ===========================================================================
# TestBuildComparisonTable: _build_comparison_table tests
# ===========================================================================


class TestBuildComparisonTable:
    """Tests for ProfessorPDFReportGenerator._build_comparison_table."""

    def _make_report_data_for_comparison(self) -> ProfessorReportData:
        """Create minimal ProfessorReportData with 3 students, 2 questions.

        Students are pre-sorted descending by overall_ensemble_mean:
          - S001 김철수 0.90 (Advanced)
          - S002 이영희 0.65 (Proficient)
          - S003 박민준 0.40 (Beginning)
        """
        question_stats = [
            _make_question_class_stats(question_sn=1, ensemble_mean=0.75, ensemble_std=0.10),
            _make_question_class_stats(
                question_sn=2,
                question_text="음성 되먹임의 예를 드시오.",
                topic="되먹임",
                ensemble_mean=0.60,
                ensemble_std=0.15,
            ),
        ]

        student_rows = [
            StudentSummaryRow(
                student_id="S001",
                real_name="김철수",
                student_number="2026100001",
                overall_ensemble_mean=0.90,
                overall_level="Advanced",
                per_question_scores={1: 0.90, 2: 0.90},
                per_question_levels={1: "Advanced", 2: "Advanced"},
                per_question_coverages={1: 0.88, 2: 0.85},
                is_at_risk=False,
                at_risk_reasons=[],
                z_score=1.2,
            ),
            StudentSummaryRow(
                student_id="S002",
                real_name="이영희",
                student_number="2026100002",
                overall_ensemble_mean=0.65,
                overall_level="Proficient",
                per_question_scores={1: 0.65, 2: 0.65},
                per_question_levels={1: "Proficient", 2: "Proficient"},
                per_question_coverages={1: 0.60, 2: 0.55},
                is_at_risk=False,
                at_risk_reasons=[],
                z_score=0.0,
            ),
            StudentSummaryRow(
                student_id="S003",
                real_name="박민준",
                student_number="2026100003",
                overall_ensemble_mean=0.40,
                overall_level="Beginning",
                per_question_scores={1: 0.40, 2: 0.40},
                per_question_levels={1: "Beginning", 2: "Beginning"},
                per_question_coverages={1: 0.35, 2: 0.30},
                is_at_risk=True,
                at_risk_reasons=["종합점수 0.45 미만"],
                z_score=-1.2,
            ),
        ]

        return ProfessorReportData(
            class_name="2A",
            week_num=2,
            subject="생리학",
            exam_title="Ch02 항상성 형성평가",
            generation_date="2026-03-08",
            n_students=3,
            n_questions=2,
            class_ensemble_mean=0.65,
            class_ensemble_std=0.20,
            class_ensemble_median=0.65,
            class_ensemble_q1=0.525,
            class_ensemble_q3=0.775,
            overall_level_distribution={"Advanced": 1, "Proficient": 1, "Developing": 0, "Beginning": 1},
            question_stats=question_stats,
            student_rows=student_rows,
            n_at_risk=1,
            pct_at_risk=33.3,
        )

    def _make_generator(self, mock_font):
        """Create ProfessorPDFReportGenerator with mocked font registration."""
        with patch("forma.professor_report.find_korean_font", return_value=mock_font):
            with patch("forma.professor_report.pdfmetrics.registerFont"):
                with patch("forma.professor_report.TTFont"):
                    from forma.professor_report import ProfessorPDFReportGenerator

                    return ProfessorPDFReportGenerator(font_path=mock_font)

    def _extract_table_text(self, table) -> str:
        """Recursively extract all text from a Table's data cells."""
        from reportlab.platypus import Paragraph

        texts = []
        if hasattr(table, "_cellvalues"):
            for row in table._cellvalues:
                for cell in row:
                    if isinstance(cell, str):
                        texts.append(cell)
                    elif isinstance(cell, Paragraph) and hasattr(cell, "text"):
                        texts.append(str(cell.text))
                    elif isinstance(cell, list):
                        for item in cell:
                            if isinstance(item, Paragraph) and hasattr(item, "text"):
                                texts.append(str(item.text))
                            elif isinstance(item, str):
                                texts.append(item)
        return " ".join(texts)

    def test_returns_non_empty_list(self, mock_font):
        """_build_comparison_table returns a non-empty list."""
        gen = self._make_generator(mock_font)
        report_data = self._make_report_data_for_comparison()

        story = gen._build_comparison_table(report_data)

        assert isinstance(story, list)
        assert len(story) > 0, "_build_comparison_table must return a non-empty list"

    def test_contains_table_flowable(self, mock_font):
        """_build_comparison_table includes at least one Table flowable."""
        from reportlab.platypus import Table

        gen = self._make_generator(mock_font)
        report_data = self._make_report_data_for_comparison()

        story = gen._build_comparison_table(report_data)
        tables = [f for f in story if isinstance(f, Table)]

        assert len(tables) > 0, (
            "_build_comparison_table must include at least one Table flowable"
        )

    def test_student_count_in_rows(self, mock_font):
        """All 3 students have rows represented in the comparison table."""
        from reportlab.platypus import Table

        gen = self._make_generator(mock_font)
        report_data = self._make_report_data_for_comparison()

        story = gen._build_comparison_table(report_data)
        tables = [f for f in story if isinstance(f, Table)]
        assert len(tables) > 0

        # Combine all table text and check for student identifiers
        all_table_text = " ".join(self._extract_table_text(t) for t in tables)

        # All three student identifiers (names or IDs) should appear
        for identifier in ("김철수", "이영희", "박민준"):
            assert identifier in all_table_text, (
                f"Student identifier '{identifier}' must appear in the comparison table"
            )

    def test_sorted_descending(self, mock_font):
        """Rows are ordered with the highest-scoring student first (0.90 before 0.65 before 0.40)."""
        from reportlab.platypus import Table

        gen = self._make_generator(mock_font)
        report_data = self._make_report_data_for_comparison()

        story = gen._build_comparison_table(report_data)
        tables = [f for f in story if isinstance(f, Table)]
        assert len(tables) > 0

        # Collect row order from the first table that contains student rows
        main_table = tables[0]
        student_name_order = []
        if hasattr(main_table, "_cellvalues"):
            from reportlab.platypus import Paragraph
            for row in main_table._cellvalues:
                for cell in row:
                    cell_text = ""
                    if isinstance(cell, str):
                        cell_text = cell
                    elif isinstance(cell, Paragraph) and hasattr(cell, "text"):
                        cell_text = str(cell.text)
                    for name in ("김철수", "이영희", "박민준"):
                        if name in cell_text and name not in student_name_order:
                            student_name_order.append(name)

        assert student_name_order == ["김철수", "이영희", "박민준"], (
            "Students must appear in descending order of overall_ensemble_mean: "
            f"김철수(0.90), 이영희(0.65), 박민준(0.40). Got: {student_name_order}"
        )

    def test_korean_level_labels(self, mock_font):
        """Korean level abbreviations 상/중상/중하/하 appear in the table, not English."""
        from reportlab.platypus import Table

        gen = self._make_generator(mock_font)
        report_data = self._make_report_data_for_comparison()

        story = gen._build_comparison_table(report_data)
        tables = [f for f in story if isinstance(f, Table)]
        assert len(tables) > 0

        all_table_text = " ".join(self._extract_table_text(t) for t in tables)

        # At least one Korean level abbreviation must appear
        korean_levels = ("상", "중상", "중하", "하")
        found_korean = any(lvl in all_table_text for lvl in korean_levels)
        assert found_korean, (
            f"Korean level labels (상/중상/중하/하) must appear in comparison table. "
            f"Found text: {all_table_text[:200]}"
        )

        # English level names must NOT be the primary labels in cells
        # (they can appear elsewhere but Korean abbreviations must be present)
        # We check Advanced/Proficient/Developing/Beginning are NOT the sole labels
        # by confirming at least one Korean label is there.
        # This is already confirmed by the assertion above.

    def test_image_flowable_present_or_graceful(self, mock_font):
        """Image flowable (lollipop chart) is present OR the method completes without crash.

        The lollipop chart (student_rank_lollipop) may not be implemented yet.
        The method must either include an Image flowable or skip it gracefully.
        In both cases, no exception should be raised.
        """
        from reportlab.platypus import Image

        gen = self._make_generator(mock_font)
        report_data = self._make_report_data_for_comparison()

        # Must not raise any exception
        story = gen._build_comparison_table(report_data)

        # If an Image is present, validate it is a proper Image flowable
        images = [f for f in story if isinstance(f, Image)]
        # Either images exist (lollipop implemented) or story is still non-empty
        # (graceful skip). Both are acceptable.
        assert isinstance(story, list) and len(story) > 0, (
            "_build_comparison_table must return a non-empty story "
            "whether or not the lollipop chart is implemented"
        )

    # ------------------------------------------------------------------
    # Augmented tests: layout column count and robustness
    # ------------------------------------------------------------------

    def _make_report_data_n_questions(self, n_questions: int) -> ProfessorReportData:
        """Build ProfessorReportData with n_questions questions and 2 students."""
        question_stats = []
        for i in range(1, n_questions + 1):
            question_stats.append(
                _make_question_class_stats(
                    question_sn=i,
                    question_text=f"Question {i}",
                    topic=f"Topic {i}",
                    ensemble_mean=0.65,
                    ensemble_std=0.10,
                )
            )

        student_rows = [
            StudentSummaryRow(
                student_id="S001",
                real_name="김철수",
                student_number="2026100001",
                overall_ensemble_mean=0.80,
                overall_level="Advanced",
                per_question_scores={i: 0.80 for i in range(1, n_questions + 1)},
                per_question_levels={i: "Advanced" for i in range(1, n_questions + 1)},
                per_question_coverages={i: 0.75 for i in range(1, n_questions + 1)},
                is_at_risk=False,
                at_risk_reasons=[],
                z_score=0.5,
            ),
            StudentSummaryRow(
                student_id="S002",
                real_name="이영희",
                student_number="2026100002",
                overall_ensemble_mean=0.50,
                overall_level="Developing",
                per_question_scores={i: 0.50 for i in range(1, n_questions + 1)},
                per_question_levels={i: "Developing" for i in range(1, n_questions + 1)},
                per_question_coverages={i: 0.40 for i in range(1, n_questions + 1)},
                is_at_risk=False,
                at_risk_reasons=[],
                z_score=-0.5,
            ),
        ]

        return ProfessorReportData(
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
            generation_date="2026-03-08",
            n_students=2,
            n_questions=n_questions,
            class_ensemble_mean=0.65,
            class_ensemble_std=0.15,
            class_ensemble_median=0.65,
            class_ensemble_q1=0.50,
            class_ensemble_q3=0.80,
            overall_level_distribution={"Advanced": 1, "Proficient": 0, "Developing": 1, "Beginning": 0},
            question_stats=question_stats,
            student_rows=student_rows,
            n_at_risk=0,
            pct_at_risk=0.0,
        )

    def test_two_col_layout_for_four_or_more_questions(self, mock_font):
        """With >= 4 questions the table uses 2 sub-cols per question (수준, 점수 only).

        With 4 questions, n_sub=2, so total columns =
        3 (fixed) + 2 (overall) + 4*2 (questions) = 13.
        Verifying that '커버리지' header text does NOT appear (it only appears in 3-col mode).
        """
        from reportlab.platypus import Table

        gen = self._make_generator(mock_font)
        report_data = self._make_report_data_n_questions(4)

        story = gen._build_comparison_table(report_data)
        assert isinstance(story, list) and len(story) > 0

        tables = [f for f in story if isinstance(f, Table)]
        assert len(tables) > 0, "Must contain at least one Table flowable"

        # In 2-col mode, '커버리지' sub-header must NOT appear
        all_table_text = " ".join(self._extract_table_text(t) for t in tables)
        assert "커버리지" not in all_table_text, (
            "With 4+ questions (2-col layout), '커버리지' sub-column must NOT appear. "
            f"Got table text: {all_table_text[:300]}"
        )

    def test_three_col_layout_for_three_or_fewer_questions(self, mock_font):
        """With <= 3 questions the table uses 3 sub-cols per question (수준, 점수, 커버리지).

        With 2 questions, n_sub=3, so '커버리지' header text appears in the table.
        """
        from reportlab.platypus import Table

        gen = self._make_generator(mock_font)
        report_data = self._make_report_data_n_questions(2)

        story = gen._build_comparison_table(report_data)
        assert isinstance(story, list) and len(story) > 0

        tables = [f for f in story if isinstance(f, Table)]
        assert len(tables) > 0, "Must contain at least one Table flowable"

        all_table_text = " ".join(self._extract_table_text(t) for t in tables)
        assert "커버리지" in all_table_text, (
            "With <= 3 questions (3-col layout), '커버리지' sub-column must appear. "
            f"Got table text: {all_table_text[:300]}"
        )

    def test_three_col_layout_for_exactly_three_questions(self, mock_font):
        """With exactly 3 questions, layout is still 3-col (커버리지 present)."""
        from reportlab.platypus import Table

        gen = self._make_generator(mock_font)
        report_data = self._make_report_data_n_questions(3)

        story = gen._build_comparison_table(report_data)
        tables = [f for f in story if isinstance(f, Table)]
        assert len(tables) > 0

        all_table_text = " ".join(self._extract_table_text(t) for t in tables)
        assert "커버리지" in all_table_text, (
            "With exactly 3 questions (3-col layout), '커버리지' must appear"
        )

    def test_empty_student_rows_does_not_crash(self, mock_font):
        """_build_comparison_table with zero students must not raise an exception."""
        gen = self._make_generator(mock_font)

        # Report with no students
        report_data = ProfessorReportData(
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
            generation_date="2026-03-08",
            n_students=0,
            n_questions=2,
            class_ensemble_mean=0.0,
            class_ensemble_std=0.0,
            class_ensemble_median=0.0,
            class_ensemble_q1=0.0,
            class_ensemble_q3=0.0,
            overall_level_distribution={"Advanced": 0, "Proficient": 0, "Developing": 0, "Beginning": 0},
            question_stats=[
                _make_question_class_stats(question_sn=1),
                _make_question_class_stats(question_sn=2),
            ],
            student_rows=[],  # empty
            n_at_risk=0,
            pct_at_risk=0.0,
        )

        # Must not crash
        story = gen._build_comparison_table(report_data)
        assert isinstance(story, list) and len(story) > 0

    def test_single_student_does_not_crash(self, mock_font):
        """_build_comparison_table with a single student must work without error."""
        from reportlab.platypus import Table

        gen = self._make_generator(mock_font)

        report_data = ProfessorReportData(
            class_name="1A",
            week_num=1,
            subject="Biology",
            exam_title="Test",
            generation_date="2026-03-08",
            n_students=1,
            n_questions=2,
            class_ensemble_mean=0.70,
            class_ensemble_std=0.0,
            class_ensemble_median=0.70,
            class_ensemble_q1=0.70,
            class_ensemble_q3=0.70,
            overall_level_distribution={"Advanced": 1, "Proficient": 0, "Developing": 0, "Beginning": 0},
            question_stats=[
                _make_question_class_stats(question_sn=1, ensemble_mean=0.70, ensemble_std=0.0),
                _make_question_class_stats(question_sn=2, ensemble_mean=0.70, ensemble_std=0.0),
            ],
            student_rows=[
                StudentSummaryRow(
                    student_id="S001",
                    real_name="유일학생",
                    student_number="2026100001",
                    overall_ensemble_mean=0.70,
                    overall_level="Proficient",
                    per_question_scores={1: 0.70, 2: 0.70},
                    per_question_levels={1: "Proficient", 2: "Proficient"},
                    per_question_coverages={1: 0.65, 2: 0.65},
                    is_at_risk=False,
                    at_risk_reasons=[],
                    z_score=0.0,
                ),
            ],
            n_at_risk=0,
            pct_at_risk=0.0,
        )

        story = gen._build_comparison_table(report_data)
        assert isinstance(story, list) and len(story) > 0

        tables = [f for f in story if isinstance(f, Table)]
        assert len(tables) > 0

        all_table_text = " ".join(self._extract_table_text(t) for t in tables)
        assert "유일학생" in all_table_text, "Single student name must appear in the table"


# ===========================================================================
# TestAtRiskVisualIndicators: visual indicator tests for _build_comparison_table
# ===========================================================================


def _make_report_data_with_at_risk() -> "ProfessorReportData":
    """Create a ProfessorReportData with 2 students: 1 at-risk, 1 normal.

    - S001 김철수 0.80 Advanced  is_at_risk=False
    - S002 이위험 0.35 Beginning is_at_risk=True
    """
    from forma.professor_report_data import (
        ProfessorReportData,
        QuestionClassStats,
        StudentSummaryRow,
    )

    question_stats = [
        QuestionClassStats(
            question_sn=1,
            question_text="항상성의 정의를 서술하시오.",
            topic="항상성",
            ensemble_mean=0.65,
            ensemble_std=0.12,
            ensemble_median=0.67,
            concept_coverage_mean=0.55,
            llm_score_mean=0.70,
            rasch_theta_mean=-0.3,
            level_distribution={"Advanced": 1, "Proficient": 0, "Developing": 0, "Beginning": 1},
            concept_mastery_rates={},
            misconception_frequencies=[],
        ),
    ]

    student_rows = [
        StudentSummaryRow(
            student_id="S001",
            real_name="김철수",
            student_number="2026100001",
            overall_ensemble_mean=0.80,
            overall_level="Advanced",
            per_question_scores={1: 0.80},
            per_question_levels={1: "Advanced"},
            per_question_coverages={1: 0.75},
            is_at_risk=False,
            at_risk_reasons=[],
            z_score=1.0,
        ),
        StudentSummaryRow(
            student_id="S002",
            real_name="이위험",
            student_number="2026100002",
            overall_ensemble_mean=0.35,
            overall_level="Beginning",
            per_question_scores={1: 0.35},
            per_question_levels={1: "Beginning"},
            per_question_coverages={1: 0.30},
            is_at_risk=True,
            at_risk_reasons=["종합점수 0.45 미만"],
            z_score=-1.0,
        ),
    ]

    return ProfessorReportData(
        class_name="1A",
        week_num=1,
        subject="생리학",
        exam_title="Ch01 서론 형성평가",
        generation_date="2026-03-08",
        n_students=2,
        n_questions=1,
        class_ensemble_mean=0.575,
        class_ensemble_std=0.225,
        class_ensemble_median=0.575,
        class_ensemble_q1=0.4625,
        class_ensemble_q3=0.6875,
        overall_level_distribution={"Advanced": 1, "Proficient": 0, "Developing": 0, "Beginning": 1},
        question_stats=question_stats,
        student_rows=student_rows,
        n_at_risk=1,
        pct_at_risk=50.0,
    )


def _make_generator_for_at_risk(mock_font_path: str) -> "ProfessorPDFReportGenerator":
    """Helper: instantiate ProfessorPDFReportGenerator with mocked font registration."""
    with patch("forma.professor_report.find_korean_font", return_value=mock_font_path):
        with patch("forma.professor_report.pdfmetrics.registerFont"):
            with patch("forma.professor_report.TTFont"):
                from forma.professor_report import ProfessorPDFReportGenerator

                return ProfessorPDFReportGenerator(font_path=mock_font_path)


class TestAtRiskVisualIndicators:
    """Tests for at-risk visual indicators in _build_comparison_table.

    RED phase: these tests will FAIL until the visual changes are implemented.
    """

    def _get_main_table(self, story):
        """Return the first Table flowable from the story."""
        from reportlab.platypus import Table

        tables = [f for f in story if isinstance(f, Table)]
        assert len(tables) > 0, "Story must contain at least one Table flowable"
        return tables[0]

    def _get_table_style_commands(self, table):
        """Extract style commands from a ReportLab Table."""
        # TableStyle commands are stored in table._tblStyle._cmds
        if hasattr(table, "_tblStyle") and hasattr(table._tblStyle, "_cmds"):
            return table._tblStyle._cmds
        # Alternative internal attribute name used by some ReportLab versions
        if hasattr(table, "style") and hasattr(table.style, "_cmds"):
            return table.style._cmds
        return []

    def _extract_rank_cell_text(self, table, row_index: int) -> str:
        """Extract text from the rank cell (column 0) of the given data row.

        row_index is 0-based relative to student data rows (offset by 2 header rows).
        """
        from reportlab.platypus import Paragraph

        table_row_idx = row_index + 2  # 2 header rows
        if hasattr(table, "_cellvalues") and len(table._cellvalues) > table_row_idx:
            cell = table._cellvalues[table_row_idx][0]
            if isinstance(cell, Paragraph) and hasattr(cell, "text"):
                return str(cell.text)
            if isinstance(cell, str):
                return cell
        return ""

    def test_at_risk_row_no_exclamation_prefix(self, mock_font):
        """At-risk student rank cell must NOT have '!' prefix.

        The '!' prefix was removed per user feedback — visual distinction is
        provided by red row background and red LINEBEFORE instead.
        """
        gen = _make_generator_for_at_risk(mock_font)
        report_data = _make_report_data_with_at_risk()

        story = gen._build_comparison_table(report_data)
        table = self._get_main_table(story)

        # S002 이위험 is at-risk and is the 2nd student (rank 2, data row index 1)
        rank_text = self._extract_rank_cell_text(table, row_index=1)
        assert "!" not in rank_text, (
            f"At-risk student rank cell must NOT contain '!' prefix. "
            f"Got rank cell text: {rank_text!r}"
        )

    def test_non_at_risk_row_no_exclamation(self, mock_font):
        """Normal (non-at-risk) student row must NOT have '!' in rank cell text.

        GREEN expectation: passing once at-risk logic is in place, '!' is only
        for at-risk students.  We verify the normal student does not get it.
        This test should PASS even before implementation (rank cell is plain '1').
        """
        gen = _make_generator_for_at_risk(mock_font)
        report_data = _make_report_data_with_at_risk()

        story = gen._build_comparison_table(report_data)
        table = self._get_main_table(story)

        # S001 김철수 is NOT at-risk (rank 1, data row index 0)
        rank_text = self._extract_rank_cell_text(table, row_index=0)
        assert "!" not in rank_text, (
            f"Non-at-risk student rank cell must NOT contain '!'. "
            f"Got rank cell text: {rank_text!r}"
        )

    def test_at_risk_row_background_red(self, mock_font):
        """At-risk student row has BACKGROUND set to #FFEBEE (light red).

        RED: no red row background is applied yet -> will FAIL.
        """
        from reportlab.lib.colors import HexColor

        gen = _make_generator_for_at_risk(mock_font)
        report_data = _make_report_data_with_at_risk()

        story = gen._build_comparison_table(report_data)
        table = self._get_main_table(story)

        cmds = self._get_table_style_commands(table)

        # S002 이위험 is rank 2, table row index = 2 + 1 = 3
        at_risk_table_row = 3
        target_color = HexColor("#FFEBEE")

        found_red_bg = False
        for cmd in cmds:
            if len(cmd) < 4:
                continue
            name = cmd[0]
            if name != "BACKGROUND":
                continue
            # cmd format: (name, start_col_row, end_col_row, color)
            start = cmd[1]
            end = cmd[2]
            color = cmd[3]
            # Check if this BACKGROUND command covers the entire at-risk row
            # (start col == 0, end col == -1 or last col)
            row_start = start[1]
            row_end = end[1]
            col_start = start[0]
            col_end = end[0]
            if row_start <= at_risk_table_row <= row_end:
                # Must cover the full row (col 0 to -1 or col 0 to last)
                if col_start == 0 and (col_end == -1 or col_end > 0):
                    # Check color match: #FFEBEE
                    if hasattr(color, "hexval"):
                        color_hex = color.hexval().upper()
                    elif hasattr(color, "_rgb"):
                        r, g, b = (int(x * 255) for x in color._rgb)
                        color_hex = f"#{r:02X}{g:02X}{b:02X}"
                    else:
                        color_hex = str(color)
                    if "FFEBEE" in color_hex or color == target_color:
                        found_red_bg = True
                        break

        assert found_red_bg, (
            "At-risk student row must have BACKGROUND set to #FFEBEE (light red). "
            f"Style commands found: {[c[0] for c in cmds if c[0] == 'BACKGROUND']}"
        )

    def test_at_risk_row_has_red_linebefore(self, mock_font):
        """At-risk student row has a LINEBEFORE with red color and width >= 2.0 at column 0.

        RED: no red LINEBEFORE is applied to at-risk rows yet -> will FAIL.
        """
        from reportlab.lib.colors import HexColor, red as rl_red

        gen = _make_generator_for_at_risk(mock_font)
        report_data = _make_report_data_with_at_risk()

        story = gen._build_comparison_table(report_data)
        table = self._get_main_table(story)

        cmds = self._get_table_style_commands(table)

        # S002 이위험 is rank 2, table row index = 3
        at_risk_table_row = 3

        found_red_linebefore = False
        for cmd in cmds:
            if len(cmd) < 5:
                continue
            name = cmd[0]
            if name != "LINEBEFORE":
                continue
            # cmd format: (name, start, end, width, color)
            start = cmd[1]
            end = cmd[2]
            width = cmd[3]
            color = cmd[4]

            # Must be at column 0 (leftmost column)
            if start[0] != 0:
                continue

            # Must cover at-risk row
            row_start = start[1]
            row_end = end[1]
            if not (row_start <= at_risk_table_row <= row_end):
                continue

            # Must have width >= 2.0
            if width < 2.0:
                continue

            # Must be red-ish color
            red_color = HexColor("#FF0000")
            if hasattr(color, "hexval"):
                color_hex = color.hexval().upper()
                is_red = color_hex.startswith("#FF") or "FF0000" in color_hex or color == rl_red
            else:
                is_red = color == rl_red or color == red_color
            if is_red:
                found_red_linebefore = True
                break

        assert found_red_linebefore, (
            "At-risk student row must have LINEBEFORE at column 0 with red color "
            f"and width >= 2.0. "
            f"LINEBEFORE commands found: {[c for c in cmds if c[0] == 'LINEBEFORE']}"
        )


# ===========================================================================
# TestBuildAtRiskSummary: tests for _build_at_risk_summary
# ===========================================================================


class TestBuildAtRiskSummary:
    """Tests for ProfessorPDFReportGenerator._build_at_risk_summary.

    RED phase: _build_at_risk_summary raises NotImplementedError -> all tests FAIL.
    """

    def test_returns_flowables(self, mock_font):
        """_build_at_risk_summary returns a non-empty list of flowables.

        RED: raises NotImplementedError -> will FAIL.
        """
        gen = _make_generator_for_at_risk(mock_font)
        report_data = _make_report_data_with_at_risk()

        result = gen._build_at_risk_summary(report_data)

        assert isinstance(result, list), (
            "_build_at_risk_summary must return a list"
        )
        assert len(result) > 0, (
            "_build_at_risk_summary must return a non-empty list of flowables"
        )

    def test_contains_at_risk_count(self, mock_font):
        """Flowables returned by _build_at_risk_summary contain the at-risk count.

        report_data has n_at_risk=1 and n_students=2, so '1' must appear in text.
        RED: raises NotImplementedError -> will FAIL.
        """
        from reportlab.platypus import Paragraph

        gen = _make_generator_for_at_risk(mock_font)
        report_data = _make_report_data_with_at_risk()

        result = gen._build_at_risk_summary(report_data)

        assert isinstance(result, list) and len(result) > 0, (
            "_build_at_risk_summary must return a non-empty list"
        )

        # Collect all text from Paragraph flowables
        all_text = " ".join(
            str(f.text)
            for f in result
            if isinstance(f, Paragraph) and hasattr(f, "text")
        )

        # n_at_risk=1 — '1' must appear somewhere in the text
        assert str(report_data.n_at_risk) in all_text, (
            f"At-risk count ({report_data.n_at_risk}) must appear in _build_at_risk_summary "
            f"flowable text. Got: {all_text!r}"
        )


# ===========================================================================
# TestBuildLlmAnalysisPage: tests for _build_llm_analysis_page
# ===========================================================================


def _make_report_data_with_llm(failed: bool = False) -> "ProfessorReportData":
    """Build a ProfessorReportData with LLM analysis fields populated.

    Args:
        failed: If True, sets llm_generation_failed=True and populates
                llm_error_message.

    Returns:
        ProfessorReportData with overall_assessment, teaching_suggestions,
        llm_model_used, llm_generation_failed, and llm_error_message set.
    """
    rd = _make_professor_report_data()
    rd.overall_assessment = "종합 평가 내용입니다."
    rd.teaching_suggestions = "교수법 제안 내용입니다."
    rd.llm_model_used = "claude-test-model"
    rd.llm_generation_failed = failed
    rd.llm_error_message = "API 오류" if failed else ""
    return rd


class TestBuildLlmAnalysisPage:
    """Tests for ProfessorPDFReportGenerator._build_llm_analysis_page.

    RED phase: _build_llm_analysis_page raises NotImplementedError -> all tests FAIL.
    """

    def _make_generator(self, mock_font: str) -> "ProfessorPDFReportGenerator":
        """Create ProfessorPDFReportGenerator with mocked font registration."""
        with patch("forma.professor_report.find_korean_font", return_value=mock_font):
            with patch("forma.professor_report.pdfmetrics.registerFont"):
                with patch("forma.professor_report.TTFont"):
                    from forma.professor_report import ProfessorPDFReportGenerator

                    return ProfessorPDFReportGenerator(font_path=mock_font)

    def _collect_paragraph_text(self, flowables: list) -> str:
        """Collect all Paragraph .text values joined into a single string."""
        from reportlab.platypus import Paragraph

        return " ".join(
            str(f.text)
            for f in flowables
            if isinstance(f, Paragraph) and hasattr(f, "text")
        )

    def test_returns_flowables(self, mock_font):
        """_build_llm_analysis_page returns a non-empty list of flowables.

        RED: raises NotImplementedError -> will FAIL.
        """
        gen = self._make_generator(mock_font)
        report_data = _make_report_data_with_llm(failed=False)

        result = gen._build_llm_analysis_page(report_data)

        assert isinstance(result, list), (
            "_build_llm_analysis_page must return a list"
        )
        assert len(result) > 0, (
            "_build_llm_analysis_page must return a non-empty list of flowables"
        )

    def test_overall_assessment_content_present(self, mock_font):
        """A Paragraph flowable contains the overall_assessment text.

        report_data.overall_assessment = '종합 평가 내용입니다.'
        RED: raises NotImplementedError -> will FAIL.
        """
        gen = self._make_generator(mock_font)
        report_data = _make_report_data_with_llm(failed=False)

        result = gen._build_llm_analysis_page(report_data)

        all_text = self._collect_paragraph_text(result)
        assert report_data.overall_assessment in all_text, (
            f"overall_assessment text '{report_data.overall_assessment}' must appear "
            f"in a Paragraph flowable. Got: {all_text!r}"
        )

    def test_teaching_suggestions_not_shown(self, mock_font):
        """Teaching suggestions section is intentionally NOT shown in the PDF.

        Per user feedback, 교수법 제안 section was removed from the report.
        The teaching_suggestions text must NOT appear in the LLM analysis page.
        """
        gen = self._make_generator(mock_font)
        report_data = _make_report_data_with_llm(failed=False)

        result = gen._build_llm_analysis_page(report_data)

        all_text = self._collect_paragraph_text(result)
        assert report_data.teaching_suggestions not in all_text, (
            f"teaching_suggestions text must NOT appear (section was removed). "
            f"Got: {all_text!r}"
        )

    def test_model_metadata_displayed(self, mock_font):
        """The llm_model_used value appears in a flowable text.

        report_data.llm_model_used = 'claude-test-model'
        RED: raises NotImplementedError -> will FAIL.
        """
        gen = self._make_generator(mock_font)
        report_data = _make_report_data_with_llm(failed=False)

        result = gen._build_llm_analysis_page(report_data)

        all_text = self._collect_paragraph_text(result)
        assert report_data.llm_model_used in all_text, (
            f"llm_model_used value '{report_data.llm_model_used}' must appear "
            f"in a flowable text. Got: {all_text!r}"
        )

    def test_fallback_notice_when_failed(self, mock_font):
        """When llm_generation_failed=True, flowables contain a fallback indicator.

        A '실패', '대체', or 'fallback' indicator must appear in the text when
        LLM generation has failed.
        RED: raises NotImplementedError -> will FAIL.
        """
        gen = self._make_generator(mock_font)
        report_data = _make_report_data_with_llm(failed=True)

        result = gen._build_llm_analysis_page(report_data)

        all_text = self._collect_paragraph_text(result)
        fallback_indicators = ("실패", "대체", "fallback", "Fallback", "FALLBACK")
        found_indicator = any(indicator in all_text for indicator in fallback_indicators)
        assert found_indicator, (
            "When llm_generation_failed=True, flowables must contain a '실패', '대체', "
            f"or 'fallback' indicator text. Got: {all_text!r}"
        )

    def test_generation_failed_false_no_fallback_notice(self, mock_font):
        """When llm_generation_failed=False, no fallback notice appears in flowables.

        RED: raises NotImplementedError -> will FAIL.
        """
        gen = self._make_generator(mock_font)
        report_data = _make_report_data_with_llm(failed=False)

        result = gen._build_llm_analysis_page(report_data)

        all_text = self._collect_paragraph_text(result)
        fallback_indicators = ("실패", "대체", "fallback", "Fallback", "FALLBACK")
        found_indicator = any(indicator in all_text for indicator in fallback_indicators)
        assert not found_indicator, (
            "When llm_generation_failed=False, no fallback notice should appear. "
            f"Got unexpected indicator in: {all_text!r}"
        )


# ===========================================================================
# TestBuildQuestionDetailPage: tests for _build_question_detail_page
# ===========================================================================


def _make_question_stats(**overrides) -> QuestionClassStats:
    """Build a minimal QuestionClassStats fixture for question detail page tests.

    Provides sensible defaults for all fields; callers can override any field
    via keyword arguments.
    """
    defaults = {
        "question_sn": 1,
        "question_text": "항상성의 정의를 서술하시오.",
        "topic": "항상성",
        "ensemble_mean": 0.65,
        "ensemble_std": 0.12,
        "ensemble_median": 0.67,
        "concept_coverage_mean": 0.55,
        "llm_score_mean": 0.70,
        "rasch_theta_mean": -0.3,
        "level_distribution": {
            "Advanced": 2,
            "Proficient": 3,
            "Developing": 2,
            "Beginning": 1,
        },
        "concept_mastery_rates": {
            "항상성": 0.80,
            "음성되먹임": 0.45,
        },
        "misconception_frequencies": [
            ("삼투와 확산 혼동", 3),
            ("음성되먹임 방향 오류", 2),
        ],
    }
    defaults.update(overrides)
    return QuestionClassStats(**defaults)


def _make_mock_chart_gen_for_detail() -> MagicMock:
    """Return a MagicMock chart generator that returns a minimal PNG BytesIO."""

    def _return_png(*args, **kwargs):
        return io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    mock = MagicMock()
    mock.level_stacked_bar = MagicMock(side_effect=_return_png)
    mock.question_level_bar = MagicMock(side_effect=_return_png)
    mock.configure_mock(**{"return_value": io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)})
    return mock


def _make_generator_for_question_detail(mock_font_path: str) -> "ProfessorPDFReportGenerator":
    """Instantiate ProfessorPDFReportGenerator with mocked font registration."""
    with patch("forma.professor_report.find_korean_font", return_value=mock_font_path):
        with patch("forma.professor_report.pdfmetrics.registerFont"):
            with patch("forma.professor_report.TTFont"):
                from forma.professor_report import ProfessorPDFReportGenerator

                return ProfessorPDFReportGenerator(font_path=mock_font_path)


class TestBuildQuestionDetailPage:
    """Tests for ProfessorPDFReportGenerator._build_question_detail_page.

    RED phase: _build_question_detail_page raises NotImplementedError ->
    all tests in this class should FAIL until T035 is implemented.
    """

    def _collect_paragraph_text(self, flowables: list) -> str:
        """Collect all Paragraph .text values joined into a single string."""
        from reportlab.platypus import Paragraph

        return " ".join(
            str(f.text)
            for f in flowables
            if isinstance(f, Paragraph) and hasattr(f, "text")
        )

    def test_returns_flowables(self, mock_font):
        """_build_question_detail_page returns a non-empty list of flowables.

        RED: raises NotImplementedError -> will FAIL.
        """
        gen = _make_generator_for_question_detail(mock_font)
        stats = _make_question_stats()
        mock_chart_gen = _make_mock_chart_gen_for_detail()

        result = gen._build_question_detail_page(stats, mock_chart_gen)

        assert isinstance(result, list), (
            "_build_question_detail_page must return a list"
        )
        assert len(result) > 0, (
            "_build_question_detail_page must return a non-empty list of flowables"
        )

    def test_stats_table_present(self, mock_font):
        """A Table flowable appears in the result from _build_question_detail_page.

        The stats summary table (ensemble_mean, std, median, etc.) must be
        present as a Table flowable.

        RED: raises NotImplementedError -> will FAIL.
        """
        from reportlab.platypus import Table

        gen = _make_generator_for_question_detail(mock_font)
        stats = _make_question_stats()
        mock_chart_gen = _make_mock_chart_gen_for_detail()

        result = gen._build_question_detail_page(stats, mock_chart_gen)

        tables = [f for f in result if isinstance(f, Table)]
        assert len(tables) > 0, (
            "_build_question_detail_page must include at least one Table flowable "
            "for the question statistics summary"
        )

    def test_chart_embedded(self, mock_font):
        """An Image flowable is present (from stacked bar / level chart), or skip gracefully.

        The chart of level distribution must be embedded as an Image flowable,
        OR the method completes without crashing if the chart is not yet implemented.

        RED: raises NotImplementedError -> will FAIL.
        """
        from reportlab.platypus import Image

        gen = _make_generator_for_question_detail(mock_font)
        stats = _make_question_stats()
        mock_chart_gen = _make_mock_chart_gen_for_detail()

        # Must not raise any exception
        result = gen._build_question_detail_page(stats, mock_chart_gen)

        assert isinstance(result, list) and len(result) > 0, (
            "_build_question_detail_page must return a non-empty list "
            "whether or not the chart Image is embedded"
        )

        # If chart IS embedded, it must be a proper Image flowable
        images = [f for f in result if isinstance(f, Image)]
        # Either an Image is present, or the method gracefully skips it.
        # Both outcomes are acceptable; only the crash is forbidden.
        _ = images  # checked above that result is non-empty

    def test_concept_mastery_table(self, mock_font):
        """A Table with concept names appears in flowables when concepts are present.

        stats.concept_mastery_rates = {"항상성": 0.80, "음성되먹임": 0.45}
        At least one Table must contain concept names in its cells.

        RED: raises NotImplementedError -> will FAIL.
        """
        from reportlab.platypus import Table, Paragraph

        gen = _make_generator_for_question_detail(mock_font)
        stats = _make_question_stats(
            concept_mastery_rates={"항상성": 0.80, "음성되먹임": 0.45}
        )
        mock_chart_gen = _make_mock_chart_gen_for_detail()

        result = gen._build_question_detail_page(stats, mock_chart_gen)

        # Collect all text from all tables
        all_table_text = ""
        for f in result:
            if isinstance(f, Table) and hasattr(f, "_cellvalues"):
                for row in f._cellvalues:
                    for cell in row:
                        if isinstance(cell, str):
                            all_table_text += cell + " "
                        elif isinstance(cell, Paragraph) and hasattr(cell, "text"):
                            all_table_text += str(cell.text) + " "

        # At least one concept name must appear in a table cell
        concept_names = list(stats.concept_mastery_rates.keys())
        found_concept = any(name in all_table_text for name in concept_names)
        assert found_concept, (
            "When concept_mastery_rates is non-empty, a Table with concept names "
            f"must appear in the flowables. "
            f"Concepts expected: {concept_names}. "
            f"Table text found: {all_table_text[:300]!r}"
        )

    def test_empty_concepts_fallback(self, mock_font):
        """If stats.concept_mastery_rates is empty, no crash and fallback text present.

        When concept_mastery_rates == {}, the method must not crash and
        must include a Paragraph with text '개념 데이터 없음' (or similar fallback).

        RED: raises NotImplementedError -> will FAIL.
        """
        gen = _make_generator_for_question_detail(mock_font)
        stats = _make_question_stats(concept_mastery_rates={})
        mock_chart_gen = _make_mock_chart_gen_for_detail()

        # Must not crash
        result = gen._build_question_detail_page(stats, mock_chart_gen)

        assert isinstance(result, list) and len(result) > 0, (
            "_build_question_detail_page must return a non-empty list even with empty concepts"
        )

        all_text = self._collect_paragraph_text(result)
        assert "개념 데이터 없음" in all_text, (
            "When concept_mastery_rates is empty, a Paragraph with '개념 데이터 없음' "
            f"must appear in flowables. Got: {all_text!r}"
        )

    def test_misconception_list(self, mock_font):
        """If stats has misconceptions, they appear in flowables text.

        stats.misconception_frequencies = [("삼투와 확산 혼동", 3), ("음성되먹임 방향 오류", 2)]
        The misconception text must be present somewhere in the flowable texts.

        RED: raises NotImplementedError -> will FAIL.
        """
        gen = _make_generator_for_question_detail(mock_font)
        stats = _make_question_stats(
            misconception_frequencies=[
                ("삼투와 확산 혼동", 3),
                ("음성되먹임 방향 오류", 2),
            ]
        )
        mock_chart_gen = _make_mock_chart_gen_for_detail()

        result = gen._build_question_detail_page(stats, mock_chart_gen)

        assert isinstance(result, list) and len(result) > 0

        all_para_text = self._collect_paragraph_text(result)

        # Also collect text from Table cells
        from reportlab.platypus import Table, Paragraph as _Paragraph
        all_text = all_para_text
        for f in result:
            if isinstance(f, Table) and hasattr(f, "_cellvalues"):
                for row in f._cellvalues:
                    for cell in row:
                        if isinstance(cell, str):
                            all_text += " " + cell
                        elif isinstance(cell, _Paragraph) and hasattr(cell, "text"):
                            all_text += " " + str(cell.text)

        misconception_text = "삼투와 확산 혼동"
        assert misconception_text in all_text, (
            f"Misconception text '{misconception_text}' must appear in flowables. "
            f"Got: {all_text[:400]!r}"
        )

    def test_empty_misconceptions_no_crash(self, mock_font):
        """Empty misconceptions list causes no crash.

        stats.misconception_frequencies = []
        The method must return a non-empty list without raising any exception.

        RED: raises NotImplementedError -> will FAIL.
        """
        gen = _make_generator_for_question_detail(mock_font)
        stats = _make_question_stats(misconception_frequencies=[])
        mock_chart_gen = _make_mock_chart_gen_for_detail()

        # Must not raise any exception
        result = gen._build_question_detail_page(stats, mock_chart_gen)

        assert isinstance(result, list) and len(result) > 0, (
            "_build_question_detail_page must return a non-empty list "
            "even when misconception_frequencies is empty"
        )

    def test_question_number_in_output(self, mock_font):
        """question_sn value appears somewhere in the flowable text.

        stats.question_sn = 7  (non-default to make the assertion meaningful)
        The number '7' (or 'Q7') must appear in at least one Paragraph flowable.

        RED: raises NotImplementedError -> will FAIL.
        """
        gen = _make_generator_for_question_detail(mock_font)
        stats = _make_question_stats(question_sn=7)
        mock_chart_gen = _make_mock_chart_gen_for_detail()

        result = gen._build_question_detail_page(stats, mock_chart_gen)

        assert isinstance(result, list) and len(result) > 0

        all_text = self._collect_paragraph_text(result)

        # Also collect text from Table cells
        from reportlab.platypus import Table, Paragraph as _Paragraph
        for f in result:
            if isinstance(f, Table) and hasattr(f, "_cellvalues"):
                for row in f._cellvalues:
                    for cell in row:
                        if isinstance(cell, str):
                            all_text += " " + cell
                        elif isinstance(cell, _Paragraph) and hasattr(cell, "text"):
                            all_text += " " + str(cell.text)

        assert "7" in all_text, (
            f"question_sn value '7' must appear somewhere in the flowable text. "
            f"Got: {all_text[:400]!r}"
        )


# ===========================================================================
# T050: Edge case tests — special characters, long AI responses, Korean text
# ===========================================================================


def _make_generator_for_edge(tmp_font_path: str) -> "ProfessorPDFReportGenerator":
    """Create ProfessorPDFReportGenerator with mocked font registration."""
    with patch("forma.professor_report.find_korean_font", return_value=tmp_font_path):
        with patch("forma.professor_report.pdfmetrics.registerFont"):
            with patch("forma.professor_report.TTFont"):
                from forma.professor_report import ProfessorPDFReportGenerator
                return ProfessorPDFReportGenerator(font_path=tmp_font_path)


class TestEscFunction:
    """T050: Tests for the _esc() XML escape helper."""

    def test_esc_ampersand(self):
        """_esc escapes '&' to '&amp;'."""
        from forma.professor_report import _esc
        result = _esc("AT&T")
        assert "&amp;" in result
        assert "&T" not in result

    def test_esc_less_than(self):
        """_esc escapes '<' to '&lt;'."""
        from forma.professor_report import _esc
        result = _esc("<script>")
        assert "&lt;" in result
        assert "<script>" not in result

    def test_esc_greater_than(self):
        """_esc escapes '>' to '&gt;'."""
        from forma.professor_report import _esc
        result = _esc("1>0")
        assert "&gt;" in result

    def test_esc_double_quote(self):
        """_esc handles double quote (no change by xml.sax.saxutils.escape by default)."""
        from forma.professor_report import _esc
        result = _esc('say "hello"')
        # xml.sax.saxutils.escape does NOT escape " by default
        assert isinstance(result, str)
        assert "hello" in result

    def test_esc_single_quote(self):
        """_esc handles single quote (no change by xml.sax.saxutils.escape by default)."""
        from forma.professor_report import _esc
        result = _esc("it's fine")
        assert isinstance(result, str)
        assert "fine" in result

    def test_esc_mixed_special_chars(self):
        """_esc handles a string with mixed special XML characters."""
        from forma.professor_report import _esc
        raw = '<Alice & "Bob" <> Carol>'
        result = _esc(raw)
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result
        # Original <, &, > should be gone
        assert "<Alice" not in result


class TestSpecialCharactersInStudentNames:
    """T050: Special characters in student names do not break report generation."""

    def _make_report_with_special_name(self, name: str) -> ProfessorReportData:
        """Build a ProfessorReportData with a student whose name has special characters."""
        row = StudentSummaryRow(
            student_id="S001",
            real_name=name,
            student_number="2026100001",
            overall_ensemble_mean=0.65,
            overall_level="Proficient",
            per_question_scores={1: 0.65},
            per_question_levels={1: "Proficient"},
            per_question_coverages={1: 0.60},
            is_at_risk=False,
            at_risk_reasons=[],
            z_score=0.0,
        )
        return ProfessorReportData(
            class_name="1A",
            week_num=1,
            subject="생리학",
            exam_title="Ch01 형성평가",
            generation_date="2026-03-08",
            n_students=1,
            n_questions=1,
            class_ensemble_mean=0.65,
            class_ensemble_std=0.0,
            class_ensemble_median=0.65,
            class_ensemble_q1=0.65,
            class_ensemble_q3=0.65,
            overall_level_distribution={"Advanced": 0, "Proficient": 1, "Developing": 0, "Beginning": 0},
            question_stats=[_make_question_class_stats()],
            student_rows=[row],
            n_at_risk=0,
            pct_at_risk=0.0,
        )

    def test_name_with_ampersand_no_crash(self, mock_font):
        """Student name with '&' does not crash _build_comparison_table."""
        gen = _make_generator_for_edge(mock_font)
        report_data = self._make_report_with_special_name("Alice & Bob")
        # Must not raise
        story = gen._build_comparison_table(report_data)
        assert isinstance(story, list) and len(story) > 0

    def test_name_with_angle_brackets_no_crash(self, mock_font):
        """Student name with '<' and '>' does not crash _build_comparison_table."""
        gen = _make_generator_for_edge(mock_font)
        report_data = self._make_report_with_special_name("<Student>")
        story = gen._build_comparison_table(report_data)
        assert isinstance(story, list) and len(story) > 0

    def test_name_with_korean_text_no_crash(self, mock_font):
        """Student name with Korean characters does not crash _build_comparison_table."""
        gen = _make_generator_for_edge(mock_font)
        report_data = self._make_report_with_special_name("김철수")
        story = gen._build_comparison_table(report_data)
        assert isinstance(story, list) and len(story) > 0


class TestLongAIResponseHandling:
    """T050: 2000+ character AI response handling in _build_llm_analysis_page."""

    def _make_gen(self, mock_font: str) -> "ProfessorPDFReportGenerator":
        with patch("forma.professor_report.find_korean_font", return_value=mock_font):
            with patch("forma.professor_report.pdfmetrics.registerFont"):
                with patch("forma.professor_report.TTFont"):
                    from forma.professor_report import ProfessorPDFReportGenerator
                    return ProfessorPDFReportGenerator(font_path=mock_font)

    def test_2000_char_overall_assessment_no_crash(self, mock_font):
        """overall_assessment with 2000+ characters does not crash _build_llm_analysis_page."""
        gen = self._make_gen(mock_font)
        rd = _make_professor_report_data()
        # 2000 characters of Korean text
        rd.overall_assessment = "종합 평가 내용입니다. " * 200  # ~2200 chars
        rd.teaching_suggestions = "교수법 제안."
        rd.llm_model_used = "test-model"

        # Must not raise
        result = gen._build_llm_analysis_page(rd)
        assert isinstance(result, list) and len(result) > 0

    def test_2000_char_teaching_suggestions_no_crash(self, mock_font):
        """teaching_suggestions with 2000+ characters does not crash _build_llm_analysis_page."""
        gen = self._make_gen(mock_font)
        rd = _make_professor_report_data()
        rd.overall_assessment = "종합 평가."
        rd.teaching_suggestions = "교수법 제안 내용입니다. " * 200  # ~2400 chars
        rd.llm_model_used = "test-model"

        result = gen._build_llm_analysis_page(rd)
        assert isinstance(result, list) and len(result) > 0

    def test_markup_in_ai_response_escaped(self, mock_font):
        """HTML/XML markup in AI response is escaped, not interpreted as tags."""
        gen = self._make_gen(mock_font)
        rd = _make_professor_report_data()
        # Include XML/HTML special chars
        rd.overall_assessment = "Score < 0.5 means 'Below Average' & requires attention."
        rd.teaching_suggestions = "Focus on <homeostasis> & 'feedback' mechanisms."
        rd.llm_model_used = "test-model"

        # Must not raise a ReportLab XML parse error
        result = gen._build_llm_analysis_page(rd)
        assert isinstance(result, list) and len(result) > 0


# ===========================================================================
# T052b: Performance benchmark — 200 students < 60 seconds (SC-001)
# ===========================================================================


class TestPerformanceBenchmark:
    """T052b: generate_pdf for 200 synthetic students must complete in < 60s."""

    def _make_200_student_report(self) -> ProfessorReportData:
        """Build a ProfessorReportData with 200 synthetic students."""
        from forma.report_data_loader import StudentReportData, QuestionReportData, compute_class_distributions
        from forma.professor_report_data import build_professor_report_data

        students = []
        for i in range(200):
            score_q1 = round((i % 10) * 0.1, 1)
            score_q2 = round(((i + 3) % 10) * 0.1, 1)
            level_q1 = (
                "Advanced" if score_q1 >= 0.85
                else "Proficient" if score_q1 >= 0.65
                else "Developing" if score_q1 >= 0.45
                else "Beginning"
            )
            level_q2 = (
                "Advanced" if score_q2 >= 0.85
                else "Proficient" if score_q2 >= 0.65
                else "Developing" if score_q2 >= 0.45
                else "Beginning"
            )
            students.append(StudentReportData(
                student_id=f"S{i:03d}",
                real_name=f"학생{i:03d}",
                student_number=f"2026{i:03d}",
                class_name="1A",
                course_name="생리학",
                chapter_name="Chapter 1",
                week_num=1,
                questions=[
                    QuestionReportData(
                        question_sn=1,
                        question_text="항상성의 정의를 서술하시오.",
                        ensemble_score=score_q1,
                        understanding_level=level_q1,
                        concept_coverage=0.5,
                        llm_median_score=2.0,
                        rasch_theta=0.0,
                        misconceptions=["개념 오류"] if score_q1 < 0.45 else [],
                    ),
                    QuestionReportData(
                        question_sn=2,
                        question_text="음성 되먹임의 예를 드시오.",
                        ensemble_score=score_q2,
                        understanding_level=level_q2,
                        concept_coverage=0.5,
                        llm_median_score=2.0,
                        rasch_theta=0.0,
                        misconceptions=[],
                    ),
                ],
            ))

        dists = compute_class_distributions(students)
        report_data = build_professor_report_data(
            students=students,
            distributions=dists,
            class_name="1A",
            week_num=1,
            subject="생리학",
            exam_title="Ch01 서론 형성평가",
        )
        report_data.overall_assessment = "종합 평가: 학급 전체적으로 양호한 수준입니다."
        report_data.teaching_suggestions = "교수법 제안: 오개념 학생을 위한 추가 지도가 필요합니다."
        report_data.llm_model_used = "claude-test"
        return report_data

    def test_generate_pdf_200_students_under_60s(self, tmp_path):
        """Generating a PDF for 200 students completes in < 60 seconds (SC-001)."""
        import time
        from forma.professor_report import ProfessorPDFReportGenerator

        report_data = self._make_200_student_report()

        start = time.perf_counter()
        gen = ProfessorPDFReportGenerator()
        pdf_path = gen.generate_pdf(report_data, str(tmp_path))
        elapsed = time.perf_counter() - start

        assert elapsed < 60.0, (
            f"PDF generation for 200 students took {elapsed:.1f}s — expected < 60s (SC-001)"
        )
        assert pdf_path.endswith(".pdf")
        import os
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0


# ===========================================================================
# T024: Hub gap table in _build_question_detail_page()
# ===========================================================================


class TestProfessorReportHubGapTable:
    """T024: Hub gap table rendered in question detail page when hub_gap_entries is non-empty.

    RED phase: _build_question_detail_page() does not yet render a hub gap table,
    so both tests must FAIL until the feature is implemented.

    The expected table has 3 columns:
      개념 | 중심성 | 학생 포함률
    where 학생 포함률 is formatted as a percentage (e.g., "66.7%").
    """

    def _make_stats_with_hub_gap(self) -> "QuestionClassStats":
        """Return a QuestionClassStats with two hub gap entries."""
        from forma.evaluation_types import HubGapEntry

        stats = _make_question_stats(question_sn=1)
        stats.hub_gap_entries = [
            HubGapEntry("폐", 0.8, False, 0.667),
            HubGapEntry("심장", 0.5, False, 0.333),
        ]
        return stats

    def _make_stats_without_hub_gap(self) -> "QuestionClassStats":
        """Return a QuestionClassStats with an empty hub_gap_entries list."""
        stats = _make_question_stats(question_sn=1)
        stats.hub_gap_entries = []
        return stats

    def _extract_all_text(self, flowables: list) -> str:
        """Extract all text from Paragraph and Table flowables."""
        from reportlab.platypus import Table, Paragraph

        texts = []
        for f in flowables:
            if isinstance(f, Paragraph) and hasattr(f, "text"):
                texts.append(str(f.text))
            elif isinstance(f, Table) and hasattr(f, "_cellvalues"):
                for row in f._cellvalues:
                    for cell in row:
                        if isinstance(cell, str):
                            texts.append(cell)
                        elif isinstance(cell, Paragraph) and hasattr(cell, "text"):
                            texts.append(str(cell.text))
        return " ".join(texts)

    def test_hub_gap_table_present_when_entries_exist(self, mock_font):
        """A Table flowable with hub gap data is added when hub_gap_entries is non-empty.

        Expects:
        - At least one Table flowable in the result
        - The table contains "66.7%" or "33.3%" (class_inclusion_rate as percentage)

        RED: _build_question_detail_page() does not yet render a hub gap table
        -> this test FAILS until T024 is implemented.
        """
        from reportlab.platypus import Table

        gen = _make_generator_for_question_detail(mock_font)
        stats = self._make_stats_with_hub_gap()
        mock_chart_gen = _make_mock_chart_gen_for_detail()

        result = gen._build_question_detail_page(stats, mock_chart_gen)

        tables = [f for f in result if isinstance(f, Table)]
        assert len(tables) > 0, (
            "Expected at least one Table flowable in _build_question_detail_page result "
            "when hub_gap_entries is non-empty, but none was found."
        )

        all_text = self._extract_all_text(result)

        # class_inclusion_rate=0.667 → "66.7%",  0.333 → "33.3%"
        has_percentage = "66.7%" in all_text or "33.3%" in all_text
        assert has_percentage, (
            "Hub gap table must contain class_inclusion_rate formatted as percentage "
            "('66.7%' or '33.3%'). "
            f"All flowable text found: {all_text[:500]!r}"
        )

    def test_hub_gap_table_absent_when_entries_empty(self, mock_font):
        """No hub gap Table is added when hub_gap_entries is empty.

        Strategy: build results for both empty and non-empty hub_gap_entries,
        then confirm the empty case does NOT produce hub gap percentage text.

        RED: Once T024 adds the feature, the 'present' test drives an
        implementation that only adds the table when entries are non-empty
        — this 'absent' test verifies the guard condition.
        """
        gen = _make_generator_for_question_detail(mock_font)
        stats_empty = self._make_stats_without_hub_gap()
        stats_with = self._make_stats_with_hub_gap()
        mock_chart_gen = _make_mock_chart_gen_for_detail()

        result_empty = gen._build_question_detail_page(stats_empty, mock_chart_gen)
        result_with = gen._build_question_detail_page(stats_with, mock_chart_gen)

        text_empty = self._extract_all_text(result_empty)
        text_with = self._extract_all_text(result_with)

        # The empty case must NOT contain percentage strings from hub gap entries
        assert "66.7%" not in text_empty and "33.3%" not in text_empty, (
            "When hub_gap_entries is empty, hub gap percentage values must NOT "
            f"appear in any flowable. Found: {text_empty[:400]!r}"
        )

        # Sanity: the non-empty case must have the percentages (guarded by other test)
        assert "66.7%" in text_with or "33.3%" in text_with, (
            "Sanity check: non-empty hub_gap_entries should produce percentage text."
        )


# ===========================================================================
# TestClassifiedMisconceptionTable: classified misconception integration
# ===========================================================================


class TestClassifiedMisconceptionTable:
    """Tests for classified misconception table in question detail page."""

    @staticmethod
    def _extract_all_text(story_elements):
        """Extract all text from Paragraph and Table elements in the story."""
        texts = []
        for el in story_elements:
            if hasattr(el, "text"):
                texts.append(el.text)
            # Also extract text from Table cells
            if hasattr(el, "_cellvalues"):
                for row in el._cellvalues:
                    for cell in row:
                        if hasattr(cell, "text"):
                            texts.append(cell.text)
        return " ".join(texts)

    def test_classified_table_rendered_when_present(self, mock_font):
        """Classified misconception table appears when classified_misconceptions is set."""
        from forma.misconception_classifier import (
            ClassifiedMisconception,
            MisconceptionPattern,
        )
        from forma.evaluation_types import TripletEdge

        gen = _make_generator_for_question_detail(mock_font)
        stats = _make_question_class_stats()
        stats.classified_misconceptions = [
            ClassifiedMisconception(
                pattern=MisconceptionPattern.CAUSAL_REVERSAL,
                master_edge=TripletEdge("B", "causes", "A"),
                student_edge=TripletEdge("A", "causes", "B"),
                concept=None,
                confidence=0.85,
                description="인과 방향 역전: A→causes→B",
            ),
        ]
        mock_chart_gen = _make_mock_chart_gen_for_detail()
        result = gen._build_question_detail_page(stats, mock_chart_gen)
        text = self._extract_all_text(result)
        assert "오개념 패턴 분류" in text
        assert "CAUSAL_REVERSAL" in text

    def test_classified_table_absent_when_empty(self, mock_font):
        """Classified misconception table is absent when classified_misconceptions is empty."""
        gen = _make_generator_for_question_detail(mock_font)
        stats = _make_question_class_stats()
        stats.classified_misconceptions = []
        mock_chart_gen = _make_mock_chart_gen_for_detail()
        result = gen._build_question_detail_page(stats, mock_chart_gen)
        text = self._extract_all_text(result)
        assert "오개념 패턴 분류" not in text

    def test_classified_table_shows_all_patterns(self, mock_font):
        """Table shows multiple pattern types when present."""
        from forma.misconception_classifier import (
            ClassifiedMisconception,
            MisconceptionPattern,
        )

        gen = _make_generator_for_question_detail(mock_font)
        stats = _make_question_class_stats()
        stats.classified_misconceptions = [
            ClassifiedMisconception(
                pattern=MisconceptionPattern.CAUSAL_REVERSAL,
                master_edge=None, student_edge=None, concept=None,
                confidence=0.85, description="Reversed",
            ),
            ClassifiedMisconception(
                pattern=MisconceptionPattern.INCLUSION_ERROR,
                master_edge=None, student_edge=None, concept=None,
                confidence=0.9, description="Inclusion error",
            ),
            ClassifiedMisconception(
                pattern=MisconceptionPattern.CONCEPT_ABSENCE,
                master_edge=None, student_edge=None, concept="항상성",
                confidence=0.75, description="Missing: 항상성",
            ),
        ]
        mock_chart_gen = _make_mock_chart_gen_for_detail()
        result = gen._build_question_detail_page(stats, mock_chart_gen)
        text = self._extract_all_text(result)
        assert "CAUSAL_REVERSAL" in text
        assert "INCLUSION_ERROR" in text
        assert "CONCEPT_ABSENCE" in text

    def test_classified_confidence_displayed(self, mock_font):
        """Confidence percentage is shown in the classified table."""
        from forma.misconception_classifier import (
            ClassifiedMisconception,
            MisconceptionPattern,
        )

        gen = _make_generator_for_question_detail(mock_font)
        stats = _make_question_class_stats()
        stats.classified_misconceptions = [
            ClassifiedMisconception(
                pattern=MisconceptionPattern.CAUSAL_REVERSAL,
                master_edge=None, student_edge=None, concept=None,
                confidence=0.85, description="Test",
            ),
        ]
        mock_chart_gen = _make_mock_chart_gen_for_detail()
        result = gen._build_question_detail_page(stats, mock_chart_gen)
        text = self._extract_all_text(result)
        assert "85%" in text


# ===========================================================================
# TestLectureGapSection: lecture gap PDF section
# ===========================================================================


class TestLectureGapSection:
    """Tests for lecture gap section in professor PDF report."""

    @staticmethod
    def _extract_all_text(story_elements):
        """Extract all text from Paragraph and Table elements in the story."""
        texts = []
        for el in story_elements:
            if hasattr(el, "text"):
                texts.append(el.text)
            if hasattr(el, "_cellvalues"):
                for row in el._cellvalues:
                    for cell in row:
                        if hasattr(cell, "text"):
                            texts.append(cell.text)
        return " ".join(texts)

    def test_gap_section_rendered_when_present(self, mock_font, report_data):
        """Lecture gap section appears when lecture_gap_report is set."""
        from forma.lecture_gap_analysis import LectureGapReport

        gen = _make_generator_for_question_detail(mock_font)
        report_data.lecture_gap_report = LectureGapReport(
            master_concepts={"A", "B", "C"},
            covered_concepts={"A"},
            missed_concepts={"B", "C"},
            extra_concepts={"D"},
            coverage_ratio=1 / 3,
            high_miss_overlap=["B"],
        )
        result = gen._build_lecture_gap_section(report_data)
        text = self._extract_all_text(result)
        assert "강의 갭 분석" in text
        assert "33.3%" in text
        assert "누락된 마스터 개념" in text

    def test_gap_section_absent_when_none(self, mock_font, report_data):
        """Lecture gap section is absent when lecture_gap_report is None."""
        gen = _make_generator_for_question_detail(mock_font)
        report_data.lecture_gap_report = None
        result = gen._build_lecture_gap_section(report_data)
        assert result == []

    def test_high_miss_overlap_shown(self, mock_font, report_data):
        """High miss overlap concepts are listed."""
        from forma.lecture_gap_analysis import LectureGapReport

        gen = _make_generator_for_question_detail(mock_font)
        report_data.lecture_gap_report = LectureGapReport(
            master_concepts={"A", "B", "C"},
            covered_concepts={"A"},
            missed_concepts={"B", "C"},
            extra_concepts=set(),
            coverage_ratio=1 / 3,
            high_miss_overlap=["B", "C"],
        )
        result = gen._build_lecture_gap_section(report_data)
        text = self._extract_all_text(result)
        assert "학생 오답률 높은 누락 개념" in text
        assert "B" in text
        assert "C" in text

    def test_gap_section_no_missed_concepts(self, mock_font, report_data):
        """No missed concepts → no missed table."""
        from forma.lecture_gap_analysis import LectureGapReport

        gen = _make_generator_for_question_detail(mock_font)
        report_data.lecture_gap_report = LectureGapReport(
            master_concepts={"A", "B"},
            covered_concepts={"A", "B"},
            missed_concepts=set(),
            extra_concepts=set(),
            coverage_ratio=1.0,
            high_miss_overlap=[],
        )
        result = gen._build_lecture_gap_section(report_data)
        text = self._extract_all_text(result)
        assert "100.0%" in text
        assert "누락된 마스터 개념" not in text


# ---------------------------------------------------------------------------
# T038: _build_emphasis_comparison_section() — FR-021 PDF rendering
# ---------------------------------------------------------------------------


class TestEmphasisComparisonSection:
    """Tests for cross-class emphasis comparison section in professor PDF report."""

    @staticmethod
    def _extract_all_text(story_elements):
        """Extract all text from Paragraph and Table elements in the story."""
        texts = []
        for el in story_elements:
            if hasattr(el, "text"):
                texts.append(el.text)
            if hasattr(el, "_cellvalues"):
                for row in el._cellvalues:
                    for cell in row:
                        if hasattr(cell, "text"):
                            texts.append(cell.text)
        return " ".join(texts)

    def test_section_rendered_with_two_classes(self, mock_font, report_data):
        """Section renders when class_emphasis_maps has >= 2 classes."""
        from forma.emphasis_map import InstructionalEmphasisMap

        gen = _make_generator_for_question_detail(mock_font)
        report_data.class_emphasis_maps = {
            "1A": InstructionalEmphasisMap(
                concept_scores={"A": 0.9, "B": 0.5},
                threshold_used=0.65, n_sentences=10, n_concepts=2,
            ),
            "1B": InstructionalEmphasisMap(
                concept_scores={"A": 0.7, "B": 0.8},
                threshold_used=0.65, n_sentences=10, n_concepts=2,
            ),
        }
        result = gen._build_emphasis_comparison_section(report_data)
        assert len(result) > 0
        text = self._extract_all_text(result)
        assert "분반 간 강조도 비교" in text
        assert "분반 수: 2개" in text

    def test_section_empty_without_class_maps(self, mock_font, report_data):
        """Section returns empty list when class_emphasis_maps is None."""
        gen = _make_generator_for_question_detail(mock_font)
        report_data.class_emphasis_maps = None
        result = gen._build_emphasis_comparison_section(report_data)
        assert result == []

    def test_section_empty_with_single_class(self, mock_font, report_data):
        """Section returns empty list when < 2 classes."""
        from forma.emphasis_map import InstructionalEmphasisMap

        gen = _make_generator_for_question_detail(mock_font)
        report_data.class_emphasis_maps = {
            "1A": InstructionalEmphasisMap(
                concept_scores={"A": 0.9, "B": 0.5},
                threshold_used=0.65, n_sentences=10, n_concepts=2,
            ),
        }
        result = gen._build_emphasis_comparison_section(report_data)
        assert result == []

    def test_table_contains_concept_and_stdev(self, mock_font, report_data):
        """Table includes concept names and stdev values."""
        from forma.emphasis_map import InstructionalEmphasisMap
        import numpy as np

        gen = _make_generator_for_question_detail(mock_font)
        report_data.class_emphasis_maps = {
            "1A": InstructionalEmphasisMap(
                concept_scores={"항상성": 0.9, "세포": 0.3},
                threshold_used=0.65, n_sentences=10, n_concepts=2,
            ),
            "1B": InstructionalEmphasisMap(
                concept_scores={"항상성": 0.5, "세포": 0.3},
                threshold_used=0.65, n_sentences=10, n_concepts=2,
            ),
        }
        result = gen._build_emphasis_comparison_section(report_data)
        text = self._extract_all_text(result)
        # Concept name appears
        assert "항상성" in text
        # Stdev of [0.9, 0.5] = 0.200
        expected_stdev = f"{float(np.std([0.9, 0.5])):.3f}"
        assert expected_stdev in text

    def test_table_contains_per_class_scores(self, mock_font, report_data):
        """Table includes per-class score columns."""
        from forma.emphasis_map import InstructionalEmphasisMap

        gen = _make_generator_for_question_detail(mock_font)
        report_data.class_emphasis_maps = {
            "1A": InstructionalEmphasisMap(
                concept_scores={"A": 0.90},
                threshold_used=0.65, n_sentences=10, n_concepts=1,
            ),
            "1B": InstructionalEmphasisMap(
                concept_scores={"A": 0.70},
                threshold_used=0.65, n_sentences=10, n_concepts=1,
            ),
        }
        result = gen._build_emphasis_comparison_section(report_data)
        text = self._extract_all_text(result)
        # Per-class score columns: 1A and 1B headers
        assert "1A" in text
        assert "1B" in text
        # Per-class scores formatted as f"{score:.2f}"
        assert "0.90" in text
        assert "0.70" in text

    def test_section_escapes_special_characters(self, mock_font, report_data):
        """Special characters in concept names are XML-escaped."""
        from forma.emphasis_map import InstructionalEmphasisMap

        gen = _make_generator_for_question_detail(mock_font)
        report_data.class_emphasis_maps = {
            "1A": InstructionalEmphasisMap(
                concept_scores={"A<B>C&D": 0.9},
                threshold_used=0.65, n_sentences=10, n_concepts=1,
            ),
            "1B": InstructionalEmphasisMap(
                concept_scores={"A<B>C&D": 0.5},
                threshold_used=0.65, n_sentences=10, n_concepts=1,
            ),
        }
        # Should not raise XML parsing error
        result = gen._build_emphasis_comparison_section(report_data)
        assert len(result) > 0
