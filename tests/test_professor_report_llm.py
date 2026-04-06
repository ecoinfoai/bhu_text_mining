"""Tests for professor_report_llm.py — LLM analysis generation.

RED phase: tests written before implementation. All tests should FAIL
until src/forma/professor_report_llm.py is fully implemented.

Covers:
  T028 — generate_professor_analysis() success, partial failure, total failure,
          empty response, PII check, and model_used field.
  T029 — _build_fallback_overall() and _build_fallback_suggestions() helpers.
  T030 — _render_overall_assessment_prompt() and
          _render_teaching_suggestions_prompt() prompt rendering.
"""

from __future__ import annotations

from unittest.mock import MagicMock


from forma.professor_report_data import (
    ProfessorReportData,
    QuestionClassStats,
    StudentSummaryRow,
)


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


def _make_test_report_data() -> ProfessorReportData:
    """Create a minimal ProfessorReportData for LLM tests.

    Includes two students with distinct PII (names / student numbers)
    so that PII-leakage tests can check the generated text does not
    contain that information.
    """
    student_rows = [
        StudentSummaryRow(
            student_id="S001",
            student_number="20230001",
            real_name="김철수",
            overall_ensemble_mean=0.80,
            overall_level="Proficient",
            per_question_scores={1: 0.85, 2: 0.75},
            per_question_levels={1: "Proficient", 2: "Proficient"},
            per_question_coverages={1: 0.80, 2: 0.70},
            is_at_risk=False,
            at_risk_reasons=[],
            z_score=0.9,
        ),
        StudentSummaryRow(
            student_id="S002",
            student_number="20230002",
            real_name="이영희",
            overall_ensemble_mean=0.35,
            overall_level="Beginning",
            per_question_scores={1: 0.30, 2: 0.40},
            per_question_levels={1: "Beginning", 2: "Beginning"},
            per_question_coverages={1: 0.20, 2: 0.30},
            is_at_risk=True,
            at_risk_reasons=["전체 점수 0.35 (기준: 0.45 미만)"],
            z_score=-0.9,
        ),
    ]

    question_stats = [
        QuestionClassStats(
            question_sn=1,
            question_text="세포막의 기능은?",
            topic="세포생물학",
            ensemble_mean=0.575,
            ensemble_std=0.275,
            ensemble_median=0.575,
            concept_coverage_mean=0.50,
            llm_score_mean=0.6,
            rasch_theta_mean=0.1,
            level_distribution={"Advanced": 0, "Proficient": 1, "Developing": 0, "Beginning": 1},
            concept_mastery_rates={"세포막": 0.5},
            misconception_frequencies=[("세포막과 세포벽 혼동", 1)],
        ),
        QuestionClassStats(
            question_sn=2,
            question_text="삼투현상이란?",
            topic="삼투",
            ensemble_mean=0.575,
            ensemble_std=0.175,
            ensemble_median=0.575,
            concept_coverage_mean=0.50,
            llm_score_mean=0.6,
            rasch_theta_mean=0.0,
            level_distribution={"Advanced": 0, "Proficient": 1, "Developing": 0, "Beginning": 1},
            concept_mastery_rates={"삼투": 0.5},
            misconception_frequencies=[],
        ),
    ]

    return ProfessorReportData(
        class_name="1A",
        week_num=3,
        subject="생물학",
        exam_title="3주차 형성평가",
        generation_date="2026-03-08",
        n_students=2,
        n_questions=2,
        class_ensemble_mean=0.575,
        class_ensemble_std=0.225,
        class_ensemble_median=0.575,
        class_ensemble_q1=0.4625,
        class_ensemble_q3=0.6875,
        overall_level_distribution={"Advanced": 0, "Proficient": 1, "Developing": 0, "Beginning": 1},
        question_stats=question_stats,
        student_rows=student_rows,
        n_at_risk=1,
        pct_at_risk=50.0,
        overall_assessment="",
        teaching_suggestions="",
        llm_model_used="",
        llm_generation_failed=False,
        llm_error_message="",
    )


def _make_mock_provider(responses=None, side_effects=None):
    """Create a mock LLM provider.

    Args:
        responses: List of return values for sequential generate() calls.
        side_effects: List of side effects (exceptions or values) for generate().

    Returns:
        MagicMock provider with model_name attribute.
    """
    provider = MagicMock()
    if side_effects:
        provider.generate.side_effect = side_effects
    elif responses:
        provider.generate.side_effect = responses
    provider.model_name = "claude-test"
    return provider


# ---------------------------------------------------------------------------
# T028: Tests for generate_professor_analysis()
# ---------------------------------------------------------------------------


class TestGenerateProfessorAnalysis:
    """Tests for generate_professor_analysis(provider, report_data)."""

    def test_success_case(self):
        """Both LLM calls succeed — all fields populated correctly."""
        from forma.professor_report_llm import generate_professor_analysis

        overall_text = "학급 전체 평균은 0.575로 보통 수준입니다. 이해도가 고르게 분포되어 있습니다."
        suggestions_text = "세포막 개념 강화가 필요합니다. 추가 실험 활동을 권장합니다."
        provider = _make_mock_provider(responses=[overall_text, suggestions_text])
        report_data = _make_test_report_data()

        generate_professor_analysis(provider, report_data)

        assert report_data.overall_assessment == overall_text
        assert report_data.teaching_suggestions == suggestions_text
        assert report_data.llm_model_used == "claude-test"
        assert report_data.llm_generation_failed is False
        assert report_data.llm_error_message in (None, "")

    def test_partial_failure_overall_fails(self):
        """First call (overall assessment) raises Exception — fallback used, flag set."""
        from forma.professor_report_llm import generate_professor_analysis

        suggestions_text = "세포막 개념 강화가 필요합니다."
        provider = _make_mock_provider(side_effects=[RuntimeError("API timeout"), suggestions_text])
        report_data = _make_test_report_data()

        generate_professor_analysis(provider, report_data)

        assert report_data.overall_assessment != ""  # fallback text provided
        assert report_data.teaching_suggestions == suggestions_text
        assert report_data.llm_generation_failed is True

    def test_partial_failure_suggestions_fails(self):
        """Second call (teaching suggestions) raises Exception — fallback used, flag set."""
        from forma.professor_report_llm import generate_professor_analysis

        overall_text = "학급 전체 평균은 0.575입니다."
        provider = _make_mock_provider(side_effects=[overall_text, RuntimeError("connection reset")])
        report_data = _make_test_report_data()

        generate_professor_analysis(provider, report_data)

        assert report_data.overall_assessment == overall_text
        assert report_data.teaching_suggestions != ""  # fallback text provided
        assert report_data.llm_generation_failed is True

    def test_total_failure(self):
        """Both LLM calls raise Exception — fallback used for both, flag set."""
        from forma.professor_report_llm import generate_professor_analysis

        provider = _make_mock_provider(side_effects=[RuntimeError("no connection"), RuntimeError("no connection")])
        report_data = _make_test_report_data()

        generate_professor_analysis(provider, report_data)

        assert report_data.overall_assessment != ""  # fallback
        assert report_data.teaching_suggestions != ""  # fallback
        assert report_data.llm_generation_failed is True

    def test_empty_response_uses_fallback(self):
        """Empty string response treated as failure — fallback used for that field."""
        from forma.professor_report_llm import generate_professor_analysis

        suggestions_text = "세포막 개념 강화가 필요합니다."
        provider = _make_mock_provider(responses=["", suggestions_text])
        report_data = _make_test_report_data()

        generate_professor_analysis(provider, report_data)

        # Empty response for overall_assessment must not be stored as-is
        assert report_data.overall_assessment != ""
        # overall generation failed (empty = failure), so flag should be True
        assert report_data.llm_generation_failed is True

    def test_no_pii_in_stored_result(self):
        """Student names and IDs from student_rows must NOT appear in results."""
        from forma.professor_report_llm import generate_professor_analysis

        # Provider returns text that does NOT contain PII
        overall_text = "학급 전체 평균은 0.575입니다. 이해도 분포가 넓습니다."
        suggestions_text = "취약 영역 집중 보완이 필요합니다."
        provider = _make_mock_provider(responses=[overall_text, suggestions_text])
        report_data = _make_test_report_data()

        generate_professor_analysis(provider, report_data)

        # Collect all PII strings that must NOT appear
        pii_strings = []
        for row in report_data.student_rows:
            if row.real_name:
                pii_strings.append(row.real_name)
            if row.student_number:
                pii_strings.append(row.student_number)

        for pii in pii_strings:
            assert pii not in report_data.overall_assessment, f"PII '{pii}' found in overall_assessment"
            assert pii not in report_data.teaching_suggestions, f"PII '{pii}' found in teaching_suggestions"

    def test_model_used_set(self):
        """After successful call, llm_model_used is not empty."""
        from forma.professor_report_llm import generate_professor_analysis

        provider = _make_mock_provider(responses=["전체 분석 텍스트입니다.", "교수법 제안입니다."])
        report_data = _make_test_report_data()

        generate_professor_analysis(provider, report_data)

        assert report_data.llm_model_used != ""
        assert isinstance(report_data.llm_model_used, str)


# ---------------------------------------------------------------------------
# T029: Tests for fallback functions
# ---------------------------------------------------------------------------


class TestFallbackFunctions:
    """Tests for _build_fallback_overall() and _build_fallback_suggestions()."""

    def test_build_fallback_overall_returns_string(self):
        """_build_fallback_overall returns a non-empty string."""
        from forma.professor_report_llm import _build_fallback_overall

        report_data = _make_test_report_data()
        result = _build_fallback_overall(report_data)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_fallback_overall_contains_stats(self):
        """Fallback overall text mentions class mean and n_students."""
        from forma.professor_report_llm import _build_fallback_overall

        report_data = _make_test_report_data()
        result = _build_fallback_overall(report_data)

        # Should contain the mean value (0.575) in some form
        mean_str = f"{report_data.class_ensemble_mean:.3f}"
        mean_alt = f"{report_data.class_ensemble_mean:.2f}"
        assert mean_str in result or mean_alt in result or "0.575" in result or "0.58" in result, (
            f"Expected class mean in fallback text, got: {result}"
        )

        # Should mention student count
        assert str(report_data.n_students) in result, (
            f"Expected n_students={report_data.n_students} in fallback text, got: {result}"
        )

    def test_build_fallback_suggestions_returns_string(self):
        """_build_fallback_suggestions returns a non-empty string."""
        from forma.professor_report_llm import _build_fallback_suggestions

        report_data = _make_test_report_data()
        result = _build_fallback_suggestions(report_data)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_fallback_suggestions_contains_at_risk(self):
        """Fallback suggestions text mentions n_at_risk count."""
        from forma.professor_report_llm import _build_fallback_suggestions

        report_data = _make_test_report_data()
        result = _build_fallback_suggestions(report_data)

        assert str(report_data.n_at_risk) in result, (
            f"Expected n_at_risk={report_data.n_at_risk} in fallback text, got: {result}"
        )

    def test_no_format_errors(self):
        """Both fallback functions execute without raising with typical data."""
        from forma.professor_report_llm import (
            _build_fallback_overall,
            _build_fallback_suggestions,
        )

        report_data = _make_test_report_data()

        # Neither call should raise
        overall = _build_fallback_overall(report_data)
        suggestions = _build_fallback_suggestions(report_data)

        assert isinstance(overall, str)
        assert isinstance(suggestions, str)


# ---------------------------------------------------------------------------
# T030: Tests for prompt rendering functions
# ---------------------------------------------------------------------------


class TestPromptRendering:
    """Tests for _render_overall_assessment_prompt() and
    _render_teaching_suggestions_prompt()."""

    def test_render_overall_assessment_prompt_returns_string(self):
        """_render_overall_assessment_prompt returns a non-empty string."""
        from forma.professor_report_llm import _render_overall_assessment_prompt

        report_data = _make_test_report_data()
        result = _render_overall_assessment_prompt(report_data)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_overall_assessment_prompt_has_xml_tags(self):
        """Overall assessment prompt contains <class_data> XML tags."""
        from forma.professor_report_llm import _render_overall_assessment_prompt

        report_data = _make_test_report_data()
        result = _render_overall_assessment_prompt(report_data)

        assert "<class_data>" in result
        assert "</class_data>" in result

    def test_overall_assessment_prompt_has_stats(self):
        """Overall assessment prompt contains class mean and n_students info."""
        from forma.professor_report_llm import _render_overall_assessment_prompt

        report_data = _make_test_report_data()
        result = _render_overall_assessment_prompt(report_data)

        # n_students must appear
        assert str(report_data.n_students) in result

        # class mean value must appear in some rounded form
        mean_str = f"{report_data.class_ensemble_mean:.3f}"
        assert mean_str in result or str(report_data.class_ensemble_mean) in result, (
            f"Expected class mean in prompt, got: {result[:500]}"
        )

    def test_no_student_pii_in_overall_prompt(self):
        """Student real_name and student_number must NOT appear in overall prompt."""
        from forma.professor_report_llm import _render_overall_assessment_prompt

        report_data = _make_test_report_data()
        result = _render_overall_assessment_prompt(report_data)

        for row in report_data.student_rows:
            if row.real_name:
                assert row.real_name not in result, f"PII '{row.real_name}' found in overall assessment prompt"
            if row.student_number:
                assert row.student_number not in result, (
                    f"PII '{row.student_number}' found in overall assessment prompt"
                )

    def test_render_teaching_suggestions_prompt_returns_string(self):
        """_render_teaching_suggestions_prompt returns a non-empty string."""
        from forma.professor_report_llm import _render_teaching_suggestions_prompt

        report_data = _make_test_report_data()
        result = _render_teaching_suggestions_prompt(report_data)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_teaching_suggestions_prompt_has_xml_tags(self):
        """Teaching suggestions prompt contains XML tags."""
        from forma.professor_report_llm import _render_teaching_suggestions_prompt

        report_data = _make_test_report_data()
        result = _render_teaching_suggestions_prompt(report_data)

        assert "<class_data>" in result
        assert "</class_data>" in result

    def test_no_student_pii_in_suggestions_prompt(self):
        """Student real_name and student_number must NOT appear in suggestions prompt."""
        from forma.professor_report_llm import _render_teaching_suggestions_prompt

        report_data = _make_test_report_data()
        result = _render_teaching_suggestions_prompt(report_data)

        for row in report_data.student_rows:
            if row.real_name:
                assert row.real_name not in result, f"PII '{row.real_name}' found in teaching suggestions prompt"
            if row.student_number:
                assert row.student_number not in result, (
                    f"PII '{row.student_number}' found in teaching suggestions prompt"
                )


# ---------------------------------------------------------------------------
# T019: Tests for generate_cluster_correction() (US4)
# ---------------------------------------------------------------------------


def _make_test_cluster():
    """Build a minimal MisconceptionCluster for LLM correction tests."""
    from forma.misconception_clustering import MisconceptionCluster
    from forma.misconception_classifier import MisconceptionPattern
    from forma.evaluation_types import TripletEdge

    return MisconceptionCluster(
        cluster_id=0,
        pattern=MisconceptionPattern.CAUSAL_REVERSAL,
        representative_error="인과 방향 역전: A->causes->B",
        member_count=5,
        student_errors=[
            "인과 방향 역전: A->causes->B",
            "인과 방향 역전: C->causes->D",
            "인과 방향 역전: E->causes->F",
            "인과 방향 역전: G->causes->H",
            "인과 방향 역전: I->causes->J",
        ],
        correction_point="",
        centroid_edge=TripletEdge("B", "causes", "A"),
    )


def _make_test_cluster_no_edge():
    """Build a MisconceptionCluster with no master edge (CONCEPT_ABSENCE)."""
    from forma.misconception_clustering import MisconceptionCluster
    from forma.misconception_classifier import MisconceptionPattern

    return MisconceptionCluster(
        cluster_id=1,
        pattern=MisconceptionPattern.CONCEPT_ABSENCE,
        representative_error="핵심 개념 부재: 항상성",
        member_count=3,
        student_errors=["부재1", "부재2", "부재3"],
        correction_point="",
        centroid_edge=None,
    )


class TestGenerateClusterCorrection:
    """Tests for generate_cluster_correction(cluster, master_edge, provider).

    RED phase: function does not exist yet -> all tests FAIL until T020
    is implemented.
    """

    def test_success_returns_nonempty_str(self):
        """Mock LLM returning a string -> returns non-empty str (FR-015)."""
        from forma.professor_report_llm import generate_cluster_correction

        cluster = _make_test_cluster()
        provider = _make_mock_provider(responses=["A는 B를 유발하는 것이 아니라 B가 A를 유발합니다."])

        result = generate_cluster_correction(cluster, cluster.centroid_edge, provider)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_llm_exception_returns_empty_string(self):
        """Mock LLM raising exception -> returns '' (not None) without raising (FR-016, I2)."""
        from forma.professor_report_llm import generate_cluster_correction

        cluster = _make_test_cluster()
        provider = _make_mock_provider(side_effects=[RuntimeError("API timeout")])

        result = generate_cluster_correction(cluster, cluster.centroid_edge, provider)

        assert isinstance(result, str)
        assert result == ""

    def test_llm_returns_none_returns_empty_string(self):
        """Mock LLM returning None -> returns '' (not None) without raising (FR-016)."""
        from forma.professor_report_llm import generate_cluster_correction

        cluster = _make_test_cluster()
        provider = _make_mock_provider(responses=[None])

        result = generate_cluster_correction(cluster, cluster.centroid_edge, provider)

        assert isinstance(result, str)
        assert result == ""

    def test_llm_called_once(self):
        """LLM called exactly once per cluster (not per student) (FR-015)."""
        from forma.professor_report_llm import generate_cluster_correction

        cluster = _make_test_cluster()
        provider = _make_mock_provider(responses=["교정 포인트 텍스트"])

        generate_cluster_correction(cluster, cluster.centroid_edge, provider)

        assert provider.generate.call_count == 1

    def test_result_type_is_str_not_optional(self):
        """correction_point type is str not Optional[str] (I2 fix)."""
        from forma.professor_report_llm import generate_cluster_correction

        cluster = _make_test_cluster()
        provider = _make_mock_provider(responses=["교정 결과"])

        result = generate_cluster_correction(cluster, cluster.centroid_edge, provider)

        assert isinstance(result, str)
        assert result is not None

    def test_with_none_master_edge(self):
        """Works with centroid_edge=None (CONCEPT_ABSENCE pattern)."""
        from forma.professor_report_llm import generate_cluster_correction

        cluster = _make_test_cluster_no_edge()
        provider = _make_mock_provider(responses=["항상성 개념을 다시 설명하세요."])

        result = generate_cluster_correction(cluster, None, provider)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_response_returns_empty_string(self):
        """Empty string response from LLM returns ''."""
        from forma.professor_report_llm import generate_cluster_correction

        cluster = _make_test_cluster()
        provider = _make_mock_provider(responses=[""])

        result = generate_cluster_correction(cluster, cluster.centroid_edge, provider)

        assert isinstance(result, str)
        assert result == ""
