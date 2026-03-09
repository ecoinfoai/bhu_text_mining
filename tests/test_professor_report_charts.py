"""Tests for professor_report_charts.py — chart generation for professor PDF reports.

RED phase: these tests are written *before* the module exists and will fail
until ``src/forma/professor_report_charts.py`` is implemented.

Covers:
  T008 — ProfessorReportChartGenerator.__init__ and _save_fig()
  T009 — score_histogram() with normal, zero-variance, single-student, and empty cases
  T010 — level_donut() with normal distribution, single level, total=0, all four levels
  T011 — question_difficulty_bar() with multiple questions, single question, varying means

Charts are rendered using the matplotlib Agg backend so no display is needed.
FontProperties is mocked to avoid requiring an actual Korean font file in CI.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# RED phase imports — these modules do not yet exist
# ---------------------------------------------------------------------------

# Import from modules not yet created (RED phase):
from forma.professor_report_charts import ProfessorReportChartGenerator
from forma.professor_report_data import QuestionClassStats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PNG_HEADER = b"\x89PNG"

CANONICAL_LEVELS = ("Advanced", "Proficient", "Developing", "Beginning")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_question_class_stats(
    question_sn: int = 1,
    question_text: str = "",
    ensemble_mean: float = 0.6,
    level_distribution: dict[str, int] | None = None,
) -> QuestionClassStats:
    """Build a minimal QuestionClassStats for testing."""
    if level_distribution is None:
        level_distribution = {
            "Advanced": 5,
            "Proficient": 10,
            "Developing": 8,
            "Beginning": 7,
        }
    return QuestionClassStats(
        question_sn=question_sn,
        question_text=question_text or f"Question {question_sn} text",
        topic="",
        ensemble_mean=ensemble_mean,
        ensemble_std=0.1,
        ensemble_median=ensemble_mean,
        concept_coverage_mean=0.5,
        llm_score_mean=0.6,
        rasch_theta_mean=0.0,
        level_distribution=level_distribution,
        concept_mastery_rates={},
        misconception_frequencies=[],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_font_path(tmp_path):
    """Create a fake font file and return its path."""
    font_file = tmp_path / "FakeFont.ttf"
    font_file.write_bytes(b"\x00" * 64)
    return str(font_file)


@pytest.fixture()
def chart_gen(mock_font_path):
    """Create ProfessorReportChartGenerator with mocked FontProperties.

    Patches find_korean_font (to avoid FileNotFoundError in CI) and
    FontProperties (to avoid loading a real TTF).  The Agg backend
    renders charts correctly without a real font file.
    """
    from matplotlib.font_manager import FontProperties

    with (
        patch(
            "forma.professor_report_charts.find_korean_font",
            return_value=mock_font_path,
        ),
        patch(
            "forma.professor_report_charts.FontProperties",
            lambda fname: FontProperties(),
        ),
    ):
        return ProfessorReportChartGenerator(font_path=mock_font_path, dpi=72)


# ===========================================================================
# T008: __init__ and _save_fig
# ===========================================================================


class TestProfessorReportChartGeneratorInit:
    """T008: Instantiation and _save_fig() tests."""

    def test_instantiates_without_error_with_font_path(self, mock_font_path):
        """Generator instantiates without error when font_path is supplied."""
        from matplotlib.font_manager import FontProperties

        with (
            patch(
                "forma.professor_report_charts.find_korean_font",
                return_value=mock_font_path,
            ),
            patch(
                "forma.professor_report_charts.FontProperties",
                lambda fname: FontProperties(),
            ),
        ):
            gen = ProfessorReportChartGenerator(font_path=mock_font_path, dpi=72)
        assert gen is not None

    def test_instantiates_auto_detect_font(self, mock_font_path):
        """Generator calls find_korean_font when font_path is None."""
        from matplotlib.font_manager import FontProperties

        mock_find = MagicMock(return_value=mock_font_path)
        with (
            patch("forma.professor_report_charts.find_korean_font", mock_find),
            patch(
                "forma.professor_report_charts.FontProperties",
                lambda fname: FontProperties(),
            ),
        ):
            gen = ProfessorReportChartGenerator(font_path=None, dpi=72)
        mock_find.assert_called_once()
        assert gen is not None

    def test_provided_font_path_skips_auto_detect(self, mock_font_path):
        """When font_path is given, find_korean_font is NOT called."""
        from matplotlib.font_manager import FontProperties

        mock_find = MagicMock(return_value=mock_font_path)
        with (
            patch("forma.professor_report_charts.find_korean_font", mock_find),
            patch(
                "forma.professor_report_charts.FontProperties",
                lambda fname: FontProperties(),
            ),
        ):
            ProfessorReportChartGenerator(font_path=mock_font_path, dpi=72)
        mock_find.assert_not_called()

    def test_save_fig_returns_bytesio(self, chart_gen):
        """_save_fig returns an io.BytesIO object."""
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        result = chart_gen._save_fig(fig)
        assert isinstance(result, io.BytesIO)

    def test_save_fig_position_is_zero(self, chart_gen):
        """_save_fig returns BytesIO with seek position at 0."""
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        result = chart_gen._save_fig(fig)
        assert result.tell() == 0

    def test_save_fig_returns_valid_png(self, chart_gen):
        """_save_fig returns BytesIO containing valid PNG data (correct header)."""
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        result = chart_gen._save_fig(fig)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_save_fig_closes_figure(self, chart_gen):
        """_save_fig closes the figure after saving (no resource leak)."""
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        num_before = plt.get_fignums()
        chart_gen._save_fig(fig)
        num_after = plt.get_fignums()
        # The figure should be closed; its number should not appear anymore
        assert fig.number not in num_after


# ===========================================================================
# T009: score_histogram
# ===========================================================================


class TestScoreHistogram:
    """T009: score_histogram() edge and normal cases."""

    def test_normal_scores_returns_bytesio(self, chart_gen):
        """score_histogram returns io.BytesIO for a normal score list."""
        result = chart_gen.score_histogram([0.3, 0.5, 0.7, 0.8, 0.6])
        assert isinstance(result, io.BytesIO)

    def test_normal_scores_valid_png(self, chart_gen):
        """score_histogram returns valid PNG for normal scores."""
        result = chart_gen.score_histogram([0.3, 0.5, 0.7, 0.8, 0.6])
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_zero_variance_no_crash(self, chart_gen):
        """score_histogram handles zero-variance input without raising."""
        result = chart_gen.score_histogram([0.6, 0.6, 0.6])
        assert isinstance(result, io.BytesIO)

    def test_zero_variance_valid_png(self, chart_gen):
        """score_histogram returns valid PNG for zero-variance scores."""
        result = chart_gen.score_histogram([0.6, 0.6, 0.6])
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_single_student_no_crash(self, chart_gen):
        """score_histogram handles a single-element list without raising."""
        result = chart_gen.score_histogram([0.5])
        assert isinstance(result, io.BytesIO)

    def test_single_student_valid_png(self, chart_gen):
        """score_histogram returns valid PNG for a single student."""
        result = chart_gen.score_histogram([0.5])
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_empty_list_no_crash(self, chart_gen):
        """score_histogram handles an empty list without raising."""
        result = chart_gen.score_histogram([])
        assert isinstance(result, io.BytesIO)

    def test_empty_list_returns_non_empty_bytesio(self, chart_gen):
        """score_histogram returns non-empty BytesIO even for empty input."""
        result = chart_gen.score_histogram([])
        assert len(result.getvalue()) > 0

    def test_histogram_bytesio_seek_position_zero(self, chart_gen):
        """score_histogram returns BytesIO with seek position at 0."""
        result = chart_gen.score_histogram([0.4, 0.6, 0.8])
        assert result.tell() == 0


# ===========================================================================
# T010: level_donut
# ===========================================================================


class TestLevelDonut:
    """T010: level_donut() edge and normal cases."""

    def test_normal_distribution_returns_bytesio(self, chart_gen):
        """level_donut returns io.BytesIO for a normal level distribution."""
        dist = {"Advanced": 5, "Proficient": 10, "Developing": 8, "Beginning": 7}
        result = chart_gen.level_donut(dist)
        assert isinstance(result, io.BytesIO)

    def test_normal_distribution_valid_png(self, chart_gen):
        """level_donut returns valid PNG for a normal distribution."""
        dist = {"Advanced": 5, "Proficient": 10, "Developing": 8, "Beginning": 7}
        result = chart_gen.level_donut(dist)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_single_level_100_percent_no_crash(self, chart_gen):
        """level_donut handles a single non-zero level without raising."""
        dist = {"Advanced": 30, "Proficient": 0, "Developing": 0, "Beginning": 0}
        result = chart_gen.level_donut(dist)
        assert isinstance(result, io.BytesIO)

    def test_single_level_valid_png(self, chart_gen):
        """level_donut returns valid PNG for a single non-zero level."""
        dist = {"Advanced": 30, "Proficient": 0, "Developing": 0, "Beginning": 0}
        result = chart_gen.level_donut(dist)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_total_zero_no_zero_division_error(self, chart_gen):
        """level_donut does not raise ZeroDivisionError when all counts are 0."""
        dist = {"Advanced": 0, "Proficient": 0, "Developing": 0, "Beginning": 0}
        # Must not raise
        result = chart_gen.level_donut(dist)
        assert isinstance(result, io.BytesIO)

    def test_total_zero_returns_valid_bytesio(self, chart_gen):
        """level_donut returns non-empty BytesIO when total=0 (fallback chart)."""
        dist = {"Advanced": 0, "Proficient": 0, "Developing": 0, "Beginning": 0}
        result = chart_gen.level_donut(dist)
        assert len(result.getvalue()) > 0

    def test_all_four_levels_present_returns_non_empty(self, chart_gen):
        """level_donut with all four levels produces a non-empty BytesIO."""
        dist = {"Advanced": 3, "Proficient": 7, "Developing": 5, "Beginning": 2}
        result = chart_gen.level_donut(dist)
        data = result.getvalue()
        assert len(data) > 0

    def test_donut_seek_position_zero(self, chart_gen):
        """level_donut returns BytesIO with seek position at 0."""
        dist = {"Advanced": 5, "Proficient": 10, "Developing": 8, "Beginning": 7}
        result = chart_gen.level_donut(dist)
        assert result.tell() == 0


# ===========================================================================
# T011: question_difficulty_bar
# ===========================================================================


class TestQuestionDifficultyBar:
    """T011: question_difficulty_bar() edge and normal cases."""

    def test_multiple_questions_returns_bytesio(self, chart_gen):
        """question_difficulty_bar returns io.BytesIO for multiple questions."""
        stats = [
            _make_question_class_stats(question_sn=1, ensemble_mean=0.3),
            _make_question_class_stats(question_sn=2, ensemble_mean=0.6),
            _make_question_class_stats(question_sn=3, ensemble_mean=0.8),
        ]
        result = chart_gen.question_difficulty_bar(stats)
        assert isinstance(result, io.BytesIO)

    def test_multiple_questions_valid_png(self, chart_gen):
        """question_difficulty_bar returns valid PNG for multiple questions."""
        stats = [
            _make_question_class_stats(question_sn=1, ensemble_mean=0.3),
            _make_question_class_stats(question_sn=2, ensemble_mean=0.6),
            _make_question_class_stats(question_sn=3, ensemble_mean=0.8),
        ]
        result = chart_gen.question_difficulty_bar(stats)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_single_question_no_crash(self, chart_gen):
        """question_difficulty_bar handles a single question without raising."""
        stats = [_make_question_class_stats(question_sn=1, ensemble_mean=0.55)]
        result = chart_gen.question_difficulty_bar(stats)
        assert isinstance(result, io.BytesIO)

    def test_single_question_valid_png(self, chart_gen):
        """question_difficulty_bar returns valid PNG for a single question."""
        stats = [_make_question_class_stats(question_sn=1, ensemble_mean=0.55)]
        result = chart_gen.question_difficulty_bar(stats)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_varying_means_non_empty_bytesio(self, chart_gen):
        """question_difficulty_bar returns non-empty BytesIO for varying means."""
        stats = [
            _make_question_class_stats(question_sn=1, ensemble_mean=0.10),
            _make_question_class_stats(question_sn=2, ensemble_mean=0.50),
            _make_question_class_stats(question_sn=3, ensemble_mean=0.90),
        ]
        result = chart_gen.question_difficulty_bar(stats)
        assert len(result.getvalue()) > 0

    def test_empty_stats_no_crash(self, chart_gen):
        """question_difficulty_bar handles an empty stats list without raising."""
        result = chart_gen.question_difficulty_bar([])
        assert isinstance(result, io.BytesIO)

    def test_empty_stats_returns_valid_bytesio(self, chart_gen):
        """question_difficulty_bar returns non-empty BytesIO for empty input."""
        result = chart_gen.question_difficulty_bar([])
        assert len(result.getvalue()) > 0

    def test_difficulty_bar_seek_position_zero(self, chart_gen):
        """question_difficulty_bar returns BytesIO with seek position at 0."""
        stats = [
            _make_question_class_stats(question_sn=1, ensemble_mean=0.4),
            _make_question_class_stats(question_sn=2, ensemble_mean=0.7),
        ]
        result = chart_gen.question_difficulty_bar(stats)
        assert result.tell() == 0


# ===========================================================================
# T036: concept_mastery_heatmap
# ===========================================================================


class TestConceptMasteryHeatmap:
    """T036: concept_mastery_heatmap() edge and normal cases."""

    def test_returns_bytes_io(self, chart_gen):
        """concept_mastery_heatmap returns io.BytesIO with PNG header for 2 questions x 3 concepts."""
        mastery_data = {
            1: {"Concept A": 0.8, "Concept B": 0.5, "Concept C": 0.3},
            2: {"Concept A": 0.6, "Concept B": 0.9, "Concept C": 0.4},
        }
        result = chart_gen.concept_mastery_heatmap(mastery_data)
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_empty_concepts_no_crash(self, chart_gen):
        """concept_mastery_heatmap handles empty dict without raising."""
        result = chart_gen.concept_mastery_heatmap({})
        assert isinstance(result, io.BytesIO)

    def test_single_question(self, chart_gen):
        """concept_mastery_heatmap handles 1 question with multiple concepts without raising."""
        mastery_data = {
            1: {"Concept X": 0.7, "Concept Y": 0.4, "Concept Z": 1.0},
        }
        result = chart_gen.concept_mastery_heatmap(mastery_data)
        assert isinstance(result, io.BytesIO)

    def test_multiple_questions_multiple_concepts(self, chart_gen):
        """concept_mastery_heatmap handles 3 questions x 5 concepts without raising."""
        mastery_data = {
            1: {"C1": 0.9, "C2": 0.8, "C3": 0.5, "C4": 0.3, "C5": 0.2},
            2: {"C1": 0.6, "C2": 0.7, "C3": 0.4, "C4": 0.8, "C5": 0.9},
            3: {"C1": 0.1, "C2": 0.3, "C3": 0.9, "C4": 0.5, "C5": 0.6},
        }
        result = chart_gen.concept_mastery_heatmap(mastery_data)
        assert isinstance(result, io.BytesIO)

    def test_closed_after_render(self, chart_gen):
        """concept_mastery_heatmap closes the figure after rendering (no memory leak)."""
        import matplotlib.pyplot as plt

        mastery_data = {
            1: {"Concept A": 0.8, "Concept B": 0.5},
            2: {"Concept A": 0.6, "Concept B": 0.9},
        }
        figs_before = set(plt.get_fignums())
        result = chart_gen.concept_mastery_heatmap(mastery_data)
        figs_after = set(plt.get_fignums())
        # Any figure opened by this call must have been closed
        new_figs = figs_after - figs_before
        assert len(new_figs) == 0
        assert isinstance(result, io.BytesIO)


# ===========================================================================
# T037: student_rank_lollipop
# ===========================================================================


def _make_student_summary_row(
    real_name: str,
    overall_ensemble_mean: float,
    is_at_risk: bool = False,
) -> "StudentSummaryRow":
    """Build a minimal StudentSummaryRow for testing."""
    from forma.professor_report_data import StudentSummaryRow

    return StudentSummaryRow(
        student_id=real_name,
        real_name=real_name,
        overall_ensemble_mean=overall_ensemble_mean,
        is_at_risk=is_at_risk,
    )


class TestStudentRankLollipop:
    """T037: student_rank_lollipop() edge and normal cases."""

    def test_returns_bytes_io(self, chart_gen):
        """student_rank_lollipop returns io.BytesIO with PNG header for 5 rows."""
        rows = [
            _make_student_summary_row("Alice", 0.9),
            _make_student_summary_row("Bob", 0.75),
            _make_student_summary_row("Carol", 0.6),
            _make_student_summary_row("Dave", 0.45),
            _make_student_summary_row("Eve", 0.3),
        ]
        result = chart_gen.student_rank_lollipop(rows)
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_empty_rows_no_crash(self, chart_gen):
        """student_rank_lollipop handles empty rows list without raising."""
        result = chart_gen.student_rank_lollipop([])
        assert isinstance(result, io.BytesIO)

    def test_large_dataset_no_crash(self, chart_gen):
        """student_rank_lollipop handles 60 students (top/bottom 25 with gap) without raising."""
        rows = [
            _make_student_summary_row(f"Student_{i:02d}", round(i / 60, 2))
            for i in range(60)
        ]
        result = chart_gen.student_rank_lollipop(rows)
        assert isinstance(result, io.BytesIO)

    def test_at_risk_highlighting(self, chart_gen):
        """student_rank_lollipop with at-risk rows does not raise."""
        rows = [
            _make_student_summary_row("Alice", 0.9, is_at_risk=False),
            _make_student_summary_row("Bob", 0.4, is_at_risk=True),
            _make_student_summary_row("Carol", 0.35, is_at_risk=True),
            _make_student_summary_row("Dave", 0.8, is_at_risk=False),
            _make_student_summary_row("Eve", 0.2, is_at_risk=True),
        ]
        result = chart_gen.student_rank_lollipop(rows, highlight_at_risk=True)
        assert isinstance(result, io.BytesIO)

    def test_all_same_score(self, chart_gen):
        """student_rank_lollipop handles all-same score (zero variance) without raising."""
        rows = [
            _make_student_summary_row(f"Student_{i}", 0.5)
            for i in range(5)
        ]
        result = chart_gen.student_rank_lollipop(rows)
        assert isinstance(result, io.BytesIO)

    def test_highlight_false(self, chart_gen):
        """student_rank_lollipop with highlight_at_risk=False does not raise."""
        rows = [
            _make_student_summary_row("Alice", 0.9, is_at_risk=False),
            _make_student_summary_row("Bob", 0.4, is_at_risk=True),
            _make_student_summary_row("Carol", 0.6, is_at_risk=False),
        ]
        result = chart_gen.student_rank_lollipop(rows, highlight_at_risk=False)
        assert isinstance(result, io.BytesIO)


# ===========================================================================
# T038: question_level_stacked_bar
# ===========================================================================


class TestQuestionLevelStackedBar:
    """T038: question_level_stacked_bar() edge and normal cases."""

    def test_returns_bytes_io(self, chart_gen):
        """question_level_stacked_bar returns io.BytesIO with PNG header for all 4 levels."""
        level_dist = {"Advanced": 8, "Proficient": 12, "Developing": 6, "Beginning": 4}
        result = chart_gen.question_level_stacked_bar(level_dist, question_sn=1)
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_single_level_100pct(self, chart_gen):
        """question_level_stacked_bar handles a single non-zero level without raising."""
        level_dist = {"Advanced": 30, "Proficient": 0, "Developing": 0, "Beginning": 0}
        result = chart_gen.question_level_stacked_bar(level_dist, question_sn=2)
        assert isinstance(result, io.BytesIO)

    def test_zero_total_no_crash(self, chart_gen):
        """question_level_stacked_bar does not divide by zero when all levels are 0."""
        level_dist = {"Advanced": 0, "Proficient": 0, "Developing": 0, "Beginning": 0}
        result = chart_gen.question_level_stacked_bar(level_dist, question_sn=3)
        assert isinstance(result, io.BytesIO)

    def test_colors_from_level_colors(self, chart_gen):
        """question_level_stacked_bar uses _LEVEL_COLORS without AttributeError."""
        level_dist = {"Advanced": 5, "Proficient": 10, "Developing": 8, "Beginning": 7}
        # Should not raise AttributeError if _LEVEL_COLORS is accessed correctly
        result = chart_gen.question_level_stacked_bar(level_dist, question_sn=1)
        assert isinstance(result, io.BytesIO)

    def test_question_sn_shown(self, chart_gen):
        """question_level_stacked_bar handles different question_sn values without raising."""
        level_dist = {"Advanced": 3, "Proficient": 7, "Developing": 5, "Beginning": 2}
        for sn in [1, 5, 10, 99]:
            result = chart_gen.question_level_stacked_bar(level_dist, question_sn=sn)
            assert isinstance(result, io.BytesIO)


# ===========================================================================
# T051: Additional edge case tests for charts
# ===========================================================================


class TestScoreHistogramAllZero:
    """T051: score_histogram with all-zero scores."""

    def test_all_zero_scores_no_crash(self, chart_gen):
        """score_histogram handles all-zero scores without raising."""
        result = chart_gen.score_histogram([0.0, 0.0, 0.0, 0.0, 0.0])
        assert isinstance(result, io.BytesIO)

    def test_all_zero_scores_valid_png(self, chart_gen):
        """score_histogram returns valid PNG for all-zero scores."""
        result = chart_gen.score_histogram([0.0, 0.0, 0.0])
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_all_one_scores_no_crash(self, chart_gen):
        """score_histogram handles all-one (max) scores without raising."""
        result = chart_gen.score_histogram([1.0, 1.0, 1.0])
        assert isinstance(result, io.BytesIO)


class TestStudentRankLollipopEdgeCases:
    """T051: Edge cases for student_rank_lollipop."""

    def test_single_student_lollipop_no_crash(self, chart_gen):
        """student_rank_lollipop handles a single student without raising."""
        rows = [_make_student_summary_row("유일학생", 0.7)]
        result = chart_gen.student_rank_lollipop(rows)
        assert isinstance(result, io.BytesIO)

    def test_single_student_lollipop_valid_png(self, chart_gen):
        """student_rank_lollipop returns valid PNG for a single student."""
        rows = [_make_student_summary_row("유일학생", 0.7)]
        result = chart_gen.student_rank_lollipop(rows)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_single_at_risk_student_lollipop(self, chart_gen):
        """student_rank_lollipop handles a single at-risk student without raising."""
        rows = [_make_student_summary_row("위험학생", 0.2, is_at_risk=True)]
        result = chart_gen.student_rank_lollipop(rows, highlight_at_risk=True)
        assert isinstance(result, io.BytesIO)


class TestConceptMasteryHeatmapLongText:
    """T051: concept_mastery_heatmap with 100-char concept names."""

    def test_100_char_concept_name_no_crash(self, chart_gen):
        """concept_mastery_heatmap handles 100-character concept names without raising."""
        long_name = "개념이름이매우길어서" * 10  # ~90+ chars Korean
        mastery_data = {
            1: {long_name: 0.7, "짧은개념": 0.5},
            2: {long_name: 0.4, "짧은개념": 0.8},
        }
        result = chart_gen.concept_mastery_heatmap(mastery_data)
        assert isinstance(result, io.BytesIO)

    def test_100_char_concept_name_valid_png(self, chart_gen):
        """concept_mastery_heatmap returns valid PNG for 100-char concept names."""
        long_name = "A" * 100  # exactly 100 ASCII chars
        mastery_data = {1: {long_name: 0.6}}
        result = chart_gen.concept_mastery_heatmap(mastery_data)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER


# ===========================================================================
# T009 (v0.7.3): build_class_knowledge_graph_chart
# ===========================================================================


def _make_aggregate(
    edges: list[tuple[str, str, str, float, int, int]] | None = None,
    question_sn: int = 1,
    total_students: int = 30,
):
    """Build a synthetic ClassKnowledgeAggregate for testing.

    Each edge tuple: (subject, relation, obj, correct_ratio, error_count, missing_count).
    correct_count is derived as total_students - error_count - missing_count.
    """
    from forma.class_knowledge_aggregate import AggregateEdge, ClassKnowledgeAggregate

    if edges is None:
        edges = [
            ("심근경색", "원인", "허혈", 0.8, 5, 1),
            ("허혈", "유발", "조직손상", 1.0, 0, 0),
            ("조직손상", "촉발", "염증", 0.3, 10, 11),
        ]

    agg_edges = []
    for subj, rel, obj, ratio, err, miss in edges:
        correct = total_students - err - miss
        agg_edges.append(AggregateEdge(
            subject=subj,
            relation=rel,
            obj=obj,
            correct_count=correct,
            error_count=err,
            missing_count=miss,
            total_students=total_students,
            correct_ratio=ratio,
        ))
    return ClassKnowledgeAggregate(
        question_sn=question_sn,
        edges=agg_edges,
        total_students=total_students,
    )


class TestBuildClassKnowledgeGraphChart:
    """T009: build_class_knowledge_graph_chart() chart tests (SC-003, FR-005, FR-007)."""

    def test_returns_bytesio(self, chart_gen):
        """build_class_knowledge_graph_chart returns io.BytesIO (SC-003)."""
        agg = _make_aggregate()
        result = chart_gen.build_class_knowledge_graph_chart(agg)
        assert isinstance(result, io.BytesIO)

    def test_returns_nonempty_png(self, chart_gen):
        """build_class_knowledge_graph_chart returns non-empty PNG (FR-005)."""
        agg = _make_aggregate()
        result = chart_gen.build_class_knowledge_graph_chart(agg)
        data = result.getvalue()
        assert len(data) > 0
        assert data[:4] == PNG_HEADER

    def test_min_ratio_filtering(self, chart_gen):
        """Edges with correct_ratio < min_ratio_to_show are filtered (SC-004, FR-007).

        We set min_ratio_to_show=0.5. Only the edge with ratio=0.8 and ratio=1.0
        should appear in the graph. The edge with ratio=0.3 should be filtered out.
        The chart should still be produced without error.
        """
        agg = _make_aggregate()
        result = chart_gen.build_class_knowledge_graph_chart(agg, min_ratio_to_show=0.5)
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0

    def test_single_edge_aggregate(self, chart_gen):
        """Single-edge aggregate works without error."""
        agg = _make_aggregate(
            edges=[("A", "R", "B", 0.6, 5, 7)],
            total_students=30,
        )
        result = chart_gen.build_class_knowledge_graph_chart(agg)
        assert isinstance(result, io.BytesIO)
        assert len(result.getvalue()) > 0

    def test_all_edges_below_threshold(self, chart_gen):
        """All edges below min_ratio_to_show: returns non-empty BytesIO (FR-009)."""
        agg = _make_aggregate(
            edges=[("A", "R", "B", 0.02, 1, 28)],
            total_students=30,
        )
        result = chart_gen.build_class_knowledge_graph_chart(agg, min_ratio_to_show=0.05)
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0

    def test_empty_edges(self, chart_gen):
        """Empty aggregate (no edges): returns non-empty BytesIO (FR-009)."""
        agg = _make_aggregate(edges=[], total_students=0)
        result = chart_gen.build_class_knowledge_graph_chart(agg)
        assert isinstance(result, io.BytesIO)
        data = result.getvalue()
        assert len(data) > 0

    def test_seek_position_zero(self, chart_gen):
        """Returned BytesIO has seek position at 0."""
        agg = _make_aggregate()
        result = chart_gen.build_class_knowledge_graph_chart(agg)
        assert result.tell() == 0
