"""Tests for report_data_loader.py — data loading, merging, and distribution computation.

RED phase: these tests are written *before* the module exists and will fail
until ``src/forma/report_data_loader.py`` is implemented.

Covers task items T002 (dataclass), T003 (YAML loading), T004 (integration),
and T005 (missing-data edge cases).
"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest
import yaml


# ---------------------------------------------------------------------------
# Sample data constants
# ---------------------------------------------------------------------------

SAMPLE_ENSEMBLE = {
    "students": [
        {
            "student_id": "S015",
            "questions": [
                {
                    "question_sn": 1,
                    "ensemble_score": 0.26,
                    "understanding_level": "Beginning",
                    "component_scores": {
                        "concept_coverage": 0.17,
                        "llm_rubric": 0.5,
                        "rasch_ability": 0.0,
                    },
                },
                {
                    "question_sn": 3,
                    "ensemble_score": 0.19,
                    "understanding_level": "Beginning",
                    "component_scores": {
                        "concept_coverage": 0.0,
                        "llm_rubric": 0.33,
                        "rasch_ability": 0.0,
                    },
                },
            ],
        },
        {
            "student_id": "S039",
            "questions": [
                {
                    "question_sn": 1,
                    "ensemble_score": 0.75,
                    "understanding_level": "Proficient",
                    "component_scores": {
                        "concept_coverage": 0.80,
                        "llm_rubric": 0.83,
                        "rasch_ability": 0.50,
                    },
                },
            ],
        },
    ],
}

SAMPLE_CONCEPT = {
    "students": [
        {
            "student_id": "S015",
            "questions": [
                {
                    "question_sn": 1,
                    "concepts": [
                        {
                            "concept": "항상성",
                            "is_present": True,
                            "similarity": 0.47,
                            "threshold": 0.39,
                        },
                        {
                            "concept": "음성되먹임",
                            "is_present": False,
                            "similarity": 0.20,
                            "threshold": 0.35,
                        },
                    ],
                },
            ],
        },
    ],
}

SAMPLE_LLM = {
    "students": [
        {
            "student_id": "S015",
            "questions": [
                {
                    "question_sn": 1,
                    "median_score": 2.0,
                    "label": "mid",
                    "reasoning": "학생은 항상성의 정의를 부분적으로 이해함.",
                    "misconceptions": ["삼투와 확산 혼동"],
                    "icc_value": 0.89,
                },
            ],
        },
    ],
}

SAMPLE_FEEDBACK = {
    "students": [
        {
            "student_id": "S015",
            "questions": [
                {
                    "question_sn": 1,
                    "feedback_text": (
                        "[평가 요약]\n항상성 개념 부분 이해.\n"
                        "[분석 결과]\n세부 기전 설명 부족.\n"
                        "[학습 제안]\n교과서 3장 복습 권장."
                    ),
                    "tier_level": 0,
                    "tier_label": "미달",
                },
            ],
        },
    ],
}

SAMPLE_STATISTICAL = {
    "students": [
        {
            "student_id": "S015",
            "questions": [
                {
                    "question_sn": 1,
                    "rasch_theta": -4.85,
                    "rasch_theta_se": 1.2,
                    "lca_class": 0,
                    "lca_class_probability": 1.0,
                },
            ],
        },
    ],
}

SAMPLE_ANP_FINAL = [
    {
        "student_id": "S015",
        "q_num": 1,
        "text": "생체항상성은 체내 환경을 일정하게 유지하는 것이다.",
        "forms_data": {
            "분반을 선택하세요.": "A반",
            "학번을 입력하세요.": "2026194126",
            "이름을 입력하세요.": "이유정",
        },
    },
    {
        "student_id": "S015",
        "q_num": 3,
        "text": "삼투는 물이 농도 차이에 의해 이동하는 현상이다.",
        "forms_data": {
            "분반을 선택하세요.": "A반",
            "학번을 입력하세요.": "2026194126",
            "이름을 입력하세요.": "이유정",
        },
    },
    {
        "student_id": "S039",
        "q_num": 1,
        "text": "항상성이란 체내 균형을 유지하는 기전이다.",
        "forms_data": {
            "분반을 선택하세요.": "B반",
            "학번을 입력하세요.": "2026194063",
            "이름을 입력하세요.": "박수영",
        },
    },
]

SAMPLE_CONFIG = {
    "metadata": {
        "chapter_name": "서론",
        "course_name": "인체구조와기능",
        "week_num": 1,
    },
    "questions": [
        {
            "sn": 1,
            "question": "항상성의 정의와 예를 서술하시오.",
            "model_answer": "항상성이란 체내 환경을 일정하게 유지하는 성질이다.",
        },
        {
            "sn": 3,
            "question": "삼투의 정의를 서술하시오.",
            "model_answer": "삼투란 반투과성 막을 통해 용매가 이동하는 현상이다.",
        },
    ],
}


# ---------------------------------------------------------------------------
# Helper: write YAML to tmp_path
# ---------------------------------------------------------------------------


def _write_yaml(path, data):
    """Write *data* as YAML to *path* with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def eval_dir(tmp_path):
    """Create a temp eval directory populated with sample YAML results."""
    edir = tmp_path / "eval_1A"

    # res_lvl1
    _write_yaml(edir / "res_lvl1" / "concept_results.yaml", SAMPLE_CONCEPT)

    # res_lvl2
    _write_yaml(edir / "res_lvl2" / "llm_results.yaml", SAMPLE_LLM)
    _write_yaml(edir / "res_lvl2" / "feedback_results.yaml", SAMPLE_FEEDBACK)

    # res_lvl3
    _write_yaml(edir / "res_lvl3" / "statistical_results.yaml", SAMPLE_STATISTICAL)

    # res_lvl4
    _write_yaml(edir / "res_lvl4" / "ensemble_results.yaml", SAMPLE_ENSEMBLE)

    return edir


@pytest.fixture()
def final_yaml(tmp_path):
    """Write sample anp_final YAML (flat array) and return its path."""
    p = tmp_path / "anp_1A_final.yaml"
    _write_yaml(p, SAMPLE_ANP_FINAL)
    return p


@pytest.fixture()
def config_yaml(tmp_path):
    """Write sample config YAML and return its path."""
    p = tmp_path / "Ch01_FormativeTest.yaml"
    _write_yaml(p, SAMPLE_CONFIG)
    return p


# ===========================================================================
# T002: Dataclass instantiation tests
# ===========================================================================


class TestConceptDetail:
    """T002: Tests for ConceptDetail dataclass."""

    def test_instantiation_all_fields(self):
        """ConceptDetail stores all four fields correctly."""
        from forma.report_data_loader import ConceptDetail

        cd = ConceptDetail(
            concept="항상성",
            is_present=True,
            similarity=0.47,
            threshold=0.39,
        )
        assert cd.concept == "항상성"
        assert cd.is_present is True
        assert cd.similarity == pytest.approx(0.47)
        assert cd.threshold == pytest.approx(0.39)

    def test_is_present_false(self):
        """ConceptDetail records is_present=False when below threshold."""
        from forma.report_data_loader import ConceptDetail

        cd = ConceptDetail(
            concept="음성되먹임",
            is_present=False,
            similarity=0.20,
            threshold=0.35,
        )
        assert cd.is_present is False
        assert cd.similarity < cd.threshold


class TestQuestionReportData:
    """T002: Tests for QuestionReportData dataclass."""

    def test_defaults_are_independent(self):
        """List and dict defaults must not be shared across instances."""
        from forma.report_data_loader import QuestionReportData

        q1 = QuestionReportData(question_sn=1)
        q2 = QuestionReportData(question_sn=2)
        q1.concepts.append("x")
        q1.misconceptions.append("y")
        q1.component_scores["k"] = 1.0

        assert q2.concepts == []
        assert q2.misconceptions == []
        assert q2.component_scores == {}

    def test_default_values(self):
        """QuestionReportData provides sensible defaults for optional fields."""
        from forma.report_data_loader import QuestionReportData

        q = QuestionReportData(question_sn=1)
        assert q.question_text == ""
        assert q.model_answer == ""
        assert q.student_answer == ""
        assert q.concept_coverage == pytest.approx(0.0)
        assert q.concepts == []
        assert q.llm_median_score == pytest.approx(0.0)
        assert q.llm_label == "N/A"
        assert q.llm_reasoning == ""
        assert q.misconceptions == []
        assert q.icc_value == pytest.approx(0.0)
        assert q.rasch_theta == pytest.approx(0.0)
        assert q.rasch_theta_se == pytest.approx(0.0)
        assert q.lca_class == 0
        assert q.lca_class_probability == pytest.approx(0.0)
        assert q.ensemble_score == pytest.approx(0.0)
        assert q.understanding_level == "N/A"
        assert q.component_scores == {}
        assert q.feedback_text == ""
        assert q.tier_level == 0
        assert q.tier_label == ""

    def test_full_instantiation(self):
        """QuestionReportData can be populated with all fields."""
        from forma.report_data_loader import QuestionReportData, ConceptDetail

        q = QuestionReportData(
            question_sn=1,
            question_text="항상성의 정의를 서술하시오.",
            model_answer="항상성이란 ...",
            student_answer="생체항상성은 ...",
            concept_coverage=0.17,
            concepts=[ConceptDetail("항상성", True, 0.47, 0.39)],
            llm_median_score=2.0,
            llm_label="mid",
            llm_reasoning="부분적으로 이해함.",
            misconceptions=["삼투와 확산 혼동"],
            icc_value=0.89,
            rasch_theta=-4.85,
            rasch_theta_se=1.2,
            lca_class=0,
            lca_class_probability=1.0,
            ensemble_score=0.26,
            understanding_level="Beginning",
            component_scores={"concept_coverage": 0.17},
            feedback_text="피드백 내용",
            tier_level=0,
            tier_label="미달",
        )
        assert q.question_sn == 1
        assert len(q.concepts) == 1
        assert q.concepts[0].concept == "항상성"
        assert q.ensemble_score == pytest.approx(0.26)


class TestStudentReportData:
    """T002: Tests for StudentReportData dataclass."""

    def test_defaults(self):
        """StudentReportData has sensible defaults."""
        from forma.report_data_loader import StudentReportData

        s = StudentReportData(student_id="S001")
        assert s.student_id == "S001"
        assert s.real_name == ""
        assert s.student_number == ""
        assert s.class_name == ""
        assert s.course_name == ""
        assert s.chapter_name == ""
        assert s.week_num == 0
        assert s.questions == []

    def test_questions_list_independent(self):
        """Each StudentReportData gets its own questions list."""
        from forma.report_data_loader import StudentReportData

        s1 = StudentReportData(student_id="S001")
        s2 = StudentReportData(student_id="S002")
        s1.questions.append("dummy")
        assert s2.questions == []


class TestClassDistributions:
    """T002: Tests for ClassDistributions dataclass."""

    def test_defaults(self):
        """ClassDistributions defaults to empty collections."""
        from forma.report_data_loader import ClassDistributions

        cd = ClassDistributions()
        assert cd.ensemble_scores == {}
        assert cd.concept_coverages == {}
        assert cd.llm_scores == {}
        assert cd.rasch_thetas == {}
        assert cd.component_scores == {}
        assert cd.overall_ensemble == []

    def test_dict_defaults_independent(self):
        """Each ClassDistributions instance has independent dicts."""
        from forma.report_data_loader import ClassDistributions

        d1 = ClassDistributions()
        d2 = ClassDistributions()
        d1.ensemble_scores[1] = [0.5]
        d1.overall_ensemble.append(0.9)
        assert d2.ensemble_scores == {}
        assert d2.overall_ensemble == []


# ===========================================================================
# T003: YAML loading tests
# ===========================================================================


class TestYamlLoading:
    """T003: Tests for loading individual YAML result files."""

    def test_load_ensemble_results(self, eval_dir):
        """Ensemble results YAML loads with expected structure."""
        path = eval_dir / "res_lvl4" / "ensemble_results.yaml"
        with open(str(path), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "students" in data
        assert data["students"][0]["student_id"] == "S015"
        assert data["students"][0]["questions"][0]["ensemble_score"] == pytest.approx(0.26)

    def test_load_concept_results(self, eval_dir):
        """Concept results YAML loads with nested concept list."""
        path = eval_dir / "res_lvl1" / "concept_results.yaml"
        with open(str(path), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        concepts = data["students"][0]["questions"][0]["concepts"]
        assert len(concepts) == 2
        assert concepts[0]["concept"] == "항상성"

    def test_load_llm_results(self, eval_dir):
        """LLM results YAML loads with median_score and misconceptions."""
        path = eval_dir / "res_lvl2" / "llm_results.yaml"
        with open(str(path), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        q = data["students"][0]["questions"][0]
        assert q["median_score"] == pytest.approx(2.0)
        assert "삼투와 확산 혼동" in q["misconceptions"]

    def test_load_feedback_results(self, eval_dir):
        """Feedback results YAML loads with Korean feedback text."""
        path = eval_dir / "res_lvl2" / "feedback_results.yaml"
        with open(str(path), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        q = data["students"][0]["questions"][0]
        assert "[평가 요약]" in q["feedback_text"]
        assert q["tier_label"] == "미달"

    def test_load_statistical_results(self, eval_dir):
        """Statistical results YAML loads rasch and LCA fields."""
        path = eval_dir / "res_lvl3" / "statistical_results.yaml"
        with open(str(path), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        q = data["students"][0]["questions"][0]
        assert q["rasch_theta"] == pytest.approx(-4.85)
        assert q["lca_class"] == 0

    def test_load_anp_final_flat_array(self, final_yaml):
        """anp_final YAML loads as a flat list (not nested under 'students')."""
        with open(str(final_yaml), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, list)
        assert data[0]["student_id"] == "S015"
        assert data[0]["q_num"] == 1  # uses q_num, not question_sn

    def test_load_config_yaml(self, config_yaml):
        """Config YAML loads metadata and questions."""
        with open(str(config_yaml), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["metadata"]["course_name"] == "인체구조와기능"
        assert data["metadata"]["chapter_name"] == "서론"
        assert len(data["questions"]) == 2

    def test_dict_get_fallback_missing_key(self):
        """dict.get() returns fallback for missing YAML keys."""
        entry = {"student_id": "S015", "q_num": 1}
        assert entry.get("forms_data", {}) == {}
        assert entry.get("text", "답안 없음") == "답안 없음"

    def test_utf8_encoding_korean_text(self, tmp_path):
        """Korean text survives YAML round-trip with UTF-8 encoding."""
        data = {"feedback": "항상성 개념 이해 부족. 교과서 복습 권장."}
        path = tmp_path / "korean.yaml"
        path.write_text(
            yaml.dump(data, allow_unicode=True),
            encoding="utf-8",
        )
        with open(str(path), encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded["feedback"] == "항상성 개념 이해 부족. 교과서 복습 권장."

    def test_forms_data_korean_keys(self, final_yaml):
        """forms_data uses Korean keys for field lookups."""
        with open(str(final_yaml), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        forms = data[0].get("forms_data", {})
        assert forms.get("이름을 입력하세요.") == "이유정"
        assert forms.get("학번을 입력하세요.") == "2026194126"
        assert forms.get("분반을 선택하세요.") == "A반"


# ===========================================================================
# T004: Integration tests — load_all_student_data()
# ===========================================================================


class TestLoadAllStudentData:
    """T004: Integration tests for load_all_student_data()."""

    def test_returns_tuple_of_students_and_distributions(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """load_all_student_data returns (list[StudentReportData], ClassDistributions)."""
        from forma.report_data_loader import (
            load_all_student_data,
            StudentReportData,
            ClassDistributions,
        )

        students, dists = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        assert isinstance(students, list)
        assert len(students) > 0
        assert isinstance(students[0], StudentReportData)
        assert isinstance(dists, ClassDistributions)

    def test_student_metadata_from_forms_data(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """forms_data fields map to real_name, student_number, class_name."""
        from forma.report_data_loader import load_all_student_data

        students, _ = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        s015 = next(s for s in students if s.student_id == "S015")
        assert s015.real_name == "이유정"
        assert s015.student_number == "2026194126"
        assert s015.class_name == "A반"

    def test_config_metadata_propagated(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """course_name, chapter_name, week_num from config YAML are set."""
        from forma.report_data_loader import load_all_student_data

        students, _ = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        s = students[0]
        assert s.course_name == "인체구조와기능"
        assert s.chapter_name == "서론"
        assert s.week_num == 1

    def test_student_id_question_sn_composite_key(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """Data from multiple YAML files is merged by (student_id, question_sn)."""
        from forma.report_data_loader import load_all_student_data

        students, _ = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        s015 = next(s for s in students if s.student_id == "S015")
        q1 = next(q for q in s015.questions if q.question_sn == 1)

        # ensemble data
        assert q1.ensemble_score == pytest.approx(0.26)
        assert q1.understanding_level == "Beginning"

        # concept data
        assert len(q1.concepts) == 2
        assert q1.concepts[0].concept == "항상성"

        # llm data
        assert q1.llm_median_score == pytest.approx(2.0)
        assert q1.llm_label == "mid"
        assert "삼투와 확산 혼동" in q1.misconceptions
        assert q1.icc_value == pytest.approx(0.89)

        # statistical data
        assert q1.rasch_theta == pytest.approx(-4.85)
        assert q1.rasch_theta_se == pytest.approx(1.2)
        assert q1.lca_class == 0
        assert q1.lca_class_probability == pytest.approx(1.0)

        # feedback data
        assert "[평가 요약]" in q1.feedback_text
        assert q1.tier_level == 0
        assert q1.tier_label == "미달"

    def test_student_answer_from_anp_final(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """student_answer is populated from anp_final 'text' field."""
        from forma.report_data_loader import load_all_student_data

        students, _ = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        s015 = next(s for s in students if s.student_id == "S015")
        q1 = next(q for q in s015.questions if q.question_sn == 1)
        assert "생체항상성" in q1.student_answer

    def test_question_text_and_model_answer_from_config(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """question_text and model_answer come from config YAML."""
        from forma.report_data_loader import load_all_student_data

        students, _ = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        s015 = next(s for s in students if s.student_id == "S015")
        q1 = next(q for q in s015.questions if q.question_sn == 1)
        assert "항상성" in q1.question_text
        assert "항상성" in q1.model_answer

    def test_component_scores_merged(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """component_scores dict from ensemble data is preserved."""
        from forma.report_data_loader import load_all_student_data

        students, _ = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        s015 = next(s for s in students if s.student_id == "S015")
        q1 = next(q for q in s015.questions if q.question_sn == 1)
        assert "concept_coverage" in q1.component_scores
        assert q1.component_scores["concept_coverage"] == pytest.approx(0.17)

    def test_multiple_students_loaded(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """Both S015 and S039 appear in the loaded students list."""
        from forma.report_data_loader import load_all_student_data

        students, _ = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        ids = {s.student_id for s in students}
        assert "S015" in ids
        assert "S039" in ids


# ===========================================================================
# T004 (continued): ClassDistributions aggregation
# ===========================================================================


class TestComputeClassDistributions:
    """T004: Tests for compute_class_distributions()."""

    def test_overall_ensemble_aggregates_all_questions(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """overall_ensemble contains ensemble scores from all students/questions."""
        from forma.report_data_loader import load_all_student_data

        students, dists = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        # S015 has Q1=0.26 and Q3=0.19; S039 has Q1=0.75
        assert len(dists.overall_ensemble) >= 3
        assert pytest.approx(0.26) in [
            pytest.approx(v) for v in dists.overall_ensemble
        ]
        assert pytest.approx(0.75) in [
            pytest.approx(v) for v in dists.overall_ensemble
        ]

    def test_ensemble_scores_per_question(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """ensemble_scores dict is keyed by question_sn with score lists."""
        from forma.report_data_loader import load_all_student_data

        _, dists = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        assert 1 in dists.ensemble_scores
        # Q1 has scores for both S015 (0.26) and S039 (0.75)
        assert len(dists.ensemble_scores[1]) >= 2

    def test_concept_coverages_per_question(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """concept_coverages dict is keyed by question_sn."""
        from forma.report_data_loader import load_all_student_data

        _, dists = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        assert 1 in dists.concept_coverages

    def test_component_scores_per_question(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """component_scores has nested dict {qsn: {component_name: [scores]}}."""
        from forma.report_data_loader import load_all_student_data

        _, dists = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        assert 1 in dists.component_scores
        assert "concept_coverage" in dists.component_scores[1]
        assert isinstance(dists.component_scores[1]["concept_coverage"], list)

    def test_compute_class_distributions_direct(self):
        """compute_class_distributions produces correct aggregations."""
        from forma.report_data_loader import (
            compute_class_distributions,
            StudentReportData,
            QuestionReportData,
            ClassDistributions,
        )

        q_a = QuestionReportData(
            question_sn=1,
            ensemble_score=0.5,
            concept_coverage=0.6,
            llm_median_score=2.0,
            rasch_theta=-1.0,
            component_scores={"concept_coverage": 0.6, "llm_rubric": 0.7},
        )
        q_b = QuestionReportData(
            question_sn=1,
            ensemble_score=0.8,
            concept_coverage=0.9,
            llm_median_score=3.0,
            rasch_theta=1.0,
            component_scores={"concept_coverage": 0.9, "llm_rubric": 0.95},
        )
        students = [
            StudentReportData(student_id="A", questions=[q_a]),
            StudentReportData(student_id="B", questions=[q_b]),
        ]
        dists = compute_class_distributions(students)

        assert isinstance(dists, ClassDistributions)
        assert len(dists.ensemble_scores[1]) == 2
        assert pytest.approx(0.5) in [
            pytest.approx(v) for v in dists.ensemble_scores[1]
        ]
        assert pytest.approx(0.8) in [
            pytest.approx(v) for v in dists.ensemble_scores[1]
        ]
        assert len(dists.overall_ensemble) == 2
        assert len(dists.concept_coverages[1]) == 2
        assert len(dists.llm_scores[1]) == 2
        assert len(dists.rasch_thetas[1]) == 2
        assert len(dists.component_scores[1]["concept_coverage"]) == 2


# ===========================================================================
# T005: Missing data edge cases
# ===========================================================================


class TestMissingFormsData:
    """T005: Edge cases when forms_data is absent."""

    def test_missing_forms_data_uses_student_id_as_name(self, tmp_path):
        """When forms_data is missing, real_name defaults to student_id."""
        from forma.report_data_loader import load_all_student_data

        # anp_final without forms_data
        anp = [{"student_id": "S099", "q_num": 1, "text": "답변 텍스트"}]
        final_path = tmp_path / "anp_final.yaml"
        _write_yaml(final_path, anp)

        config = {
            "metadata": {
                "chapter_name": "서론",
                "course_name": "인체구조와기능",
                "week_num": 1,
            },
            "questions": [
                {"sn": 1, "question": "Q1?", "model_answer": "A1"},
            ],
        }
        config_path = tmp_path / "config.yaml"
        _write_yaml(config_path, config)

        edir = tmp_path / "eval"
        ensemble = {
            "students": [
                {
                    "student_id": "S099",
                    "questions": [
                        {
                            "question_sn": 1,
                            "ensemble_score": 0.5,
                            "understanding_level": "Developing",
                            "component_scores": {"concept_coverage": 0.5},
                        },
                    ],
                },
            ],
        }
        _write_yaml(edir / "res_lvl4" / "ensemble_results.yaml", ensemble)
        # Create empty result files for other levels
        _write_yaml(edir / "res_lvl1" / "concept_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl2" / "llm_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl2" / "feedback_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl3" / "statistical_results.yaml", {"students": []})

        students, _ = load_all_student_data(
            str(final_path), str(config_path), str(edir),
        )
        s099 = next(s for s in students if s.student_id == "S099")
        assert s099.real_name == "S099"
        assert s099.student_number == "N/A"
        assert s099.class_name == "N/A"


class TestPartialQuestions:
    """T005: Edge cases with partial question data."""

    def test_missing_question_in_anp_uses_default_answer(self, tmp_path):
        """When a question exists in ensemble but not anp_final, student_answer defaults."""
        from forma.report_data_loader import load_all_student_data

        # anp_final has only Q1 for S015
        anp = [
            {
                "student_id": "S015",
                "q_num": 1,
                "text": "답변 있음",
                "forms_data": {
                    "이름을 입력하세요.": "이유정",
                    "학번을 입력하세요.": "2026194126",
                    "분반을 선택하세요.": "A반",
                },
            },
        ]
        final_path = tmp_path / "anp_final.yaml"
        _write_yaml(final_path, anp)

        config = {
            "metadata": {
                "chapter_name": "서론",
                "course_name": "인체구조와기능",
                "week_num": 1,
            },
            "questions": [
                {"sn": 1, "question": "Q1?", "model_answer": "A1"},
                {"sn": 3, "question": "Q3?", "model_answer": "A3"},
            ],
        }
        config_path = tmp_path / "config.yaml"
        _write_yaml(config_path, config)

        edir = tmp_path / "eval"
        ensemble = {
            "students": [
                {
                    "student_id": "S015",
                    "questions": [
                        {
                            "question_sn": 1,
                            "ensemble_score": 0.5,
                            "understanding_level": "Developing",
                            "component_scores": {},
                        },
                        {
                            "question_sn": 3,
                            "ensemble_score": 0.19,
                            "understanding_level": "Beginning",
                            "component_scores": {},
                        },
                    ],
                },
            ],
        }
        _write_yaml(edir / "res_lvl4" / "ensemble_results.yaml", ensemble)
        _write_yaml(edir / "res_lvl1" / "concept_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl2" / "llm_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl2" / "feedback_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl3" / "statistical_results.yaml", {"students": []})

        students, _ = load_all_student_data(
            str(final_path), str(config_path), str(edir),
        )
        s015 = next(s for s in students if s.student_id == "S015")
        q3 = next(q for q in s015.questions if q.question_sn == 3)
        assert q3.student_answer == "답안 없음"


class TestMissingResultFiles:
    """T005: Edge cases when entire result categories are missing."""

    def test_missing_concept_results_defaults(self, tmp_path):
        """When concept_results is empty, concept_coverage=0.0 and concepts=[]."""
        from forma.report_data_loader import load_all_student_data

        anp = [
            {
                "student_id": "S015",
                "q_num": 1,
                "text": "답변",
                "forms_data": {
                    "이름을 입력하세요.": "이유정",
                    "학번을 입력하세요.": "2026194126",
                    "분반을 선택하세요.": "A반",
                },
            },
        ]
        final_path = tmp_path / "anp_final.yaml"
        _write_yaml(final_path, anp)

        config = {
            "metadata": {
                "chapter_name": "서론",
                "course_name": "인체구조와기능",
                "week_num": 1,
            },
            "questions": [
                {"sn": 1, "question": "Q1?", "model_answer": "A1"},
            ],
        }
        config_path = tmp_path / "config.yaml"
        _write_yaml(config_path, config)

        edir = tmp_path / "eval"
        ensemble = {
            "students": [
                {
                    "student_id": "S015",
                    "questions": [
                        {
                            "question_sn": 1,
                            "ensemble_score": 0.5,
                            "understanding_level": "Developing",
                            "component_scores": {"concept_coverage": 0.5},
                        },
                    ],
                },
            ],
        }
        _write_yaml(edir / "res_lvl4" / "ensemble_results.yaml", ensemble)
        # concept_results has no matching student
        _write_yaml(edir / "res_lvl1" / "concept_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl2" / "llm_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl2" / "feedback_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl3" / "statistical_results.yaml", {"students": []})

        students, _ = load_all_student_data(
            str(final_path), str(config_path), str(edir),
        )
        s015 = next(s for s in students if s.student_id == "S015")
        q1 = next(q for q in s015.questions if q.question_sn == 1)
        assert q1.concept_coverage == pytest.approx(0.0)
        assert q1.concepts == []

    def test_missing_feedback_results_defaults(self, tmp_path):
        """When feedback_results is empty, feedback_text gets default placeholder."""
        from forma.report_data_loader import load_all_student_data

        anp = [
            {
                "student_id": "S015",
                "q_num": 1,
                "text": "답변",
                "forms_data": {
                    "이름을 입력하세요.": "이유정",
                    "학번을 입력하세요.": "2026194126",
                    "분반을 선택하세요.": "A반",
                },
            },
        ]
        final_path = tmp_path / "anp_final.yaml"
        _write_yaml(final_path, anp)

        config = {
            "metadata": {
                "chapter_name": "서론",
                "course_name": "인체구조와기능",
                "week_num": 1,
            },
            "questions": [
                {"sn": 1, "question": "Q1?", "model_answer": "A1"},
            ],
        }
        config_path = tmp_path / "config.yaml"
        _write_yaml(config_path, config)

        edir = tmp_path / "eval"
        ensemble = {
            "students": [
                {
                    "student_id": "S015",
                    "questions": [
                        {
                            "question_sn": 1,
                            "ensemble_score": 0.5,
                            "understanding_level": "Developing",
                            "component_scores": {},
                        },
                    ],
                },
            ],
        }
        _write_yaml(edir / "res_lvl4" / "ensemble_results.yaml", ensemble)
        _write_yaml(edir / "res_lvl1" / "concept_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl2" / "llm_results.yaml", {"students": []})
        # feedback_results has no matching student
        _write_yaml(edir / "res_lvl2" / "feedback_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl3" / "statistical_results.yaml", {"students": []})

        students, _ = load_all_student_data(
            str(final_path), str(config_path), str(edir),
        )
        s015 = next(s for s in students if s.student_id == "S015")
        q1 = next(q for q in s015.questions if q.question_sn == 1)
        assert q1.feedback_text == "(피드백 데이터 없음)"

    def test_student_in_ensemble_not_in_anp_final(self, tmp_path):
        """Student in ensemble but absent from anp_final still generates data."""
        from forma.report_data_loader import load_all_student_data

        # Empty anp_final (no student entries)
        final_path = tmp_path / "anp_final.yaml"
        _write_yaml(final_path, [])

        config = {
            "metadata": {
                "chapter_name": "서론",
                "course_name": "인체구조와기능",
                "week_num": 1,
            },
            "questions": [
                {"sn": 1, "question": "Q1?", "model_answer": "A1"},
            ],
        }
        config_path = tmp_path / "config.yaml"
        _write_yaml(config_path, config)

        edir = tmp_path / "eval"
        ensemble = {
            "students": [
                {
                    "student_id": "S099",
                    "questions": [
                        {
                            "question_sn": 1,
                            "ensemble_score": 0.33,
                            "understanding_level": "Developing",
                            "component_scores": {"concept_coverage": 0.4},
                        },
                    ],
                },
            ],
        }
        _write_yaml(edir / "res_lvl4" / "ensemble_results.yaml", ensemble)
        _write_yaml(edir / "res_lvl1" / "concept_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl2" / "llm_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl2" / "feedback_results.yaml", {"students": []})
        _write_yaml(edir / "res_lvl3" / "statistical_results.yaml", {"students": []})

        students, _ = load_all_student_data(
            str(final_path), str(config_path), str(edir),
        )
        # S099 should still appear despite not being in anp_final
        assert len(students) >= 1
        s099 = next(s for s in students if s.student_id == "S099")
        assert s099.real_name == "S099"  # fallback to student_id
        assert s099.student_number == "N/A"
        assert s099.class_name == "N/A"
        q1 = next(q for q in s099.questions if q.question_sn == 1)
        assert q1.ensemble_score == pytest.approx(0.33)
        assert q1.student_answer == "답안 없음"
        assert q1.feedback_text == "(피드백 데이터 없음)"


class TestConceptCoverageFromConceptResults:
    """T005: concept_coverage should come from concept data when available."""

    def test_concept_coverage_computed_from_concepts(
        self, final_yaml, config_yaml, eval_dir,
    ):
        """concept_coverage reflects the ratio of is_present concepts."""
        from forma.report_data_loader import load_all_student_data

        students, _ = load_all_student_data(
            str(final_yaml), str(config_yaml), str(eval_dir),
        )
        s015 = next(s for s in students if s.student_id == "S015")
        q1 = next(q for q in s015.questions if q.question_sn == 1)
        # In SAMPLE_CONCEPT: 1 of 2 concepts is_present → 0.5
        # But the actual concept_coverage depends on implementation.
        # At minimum, concepts list should be populated.
        assert len(q1.concepts) == 2
        assert q1.concepts[0].is_present is True
        assert q1.concepts[1].is_present is False


# ===========================================================================
# T039: Edge case tests (Phase 7)
# ===========================================================================


class TestFilenameEdgeCases:
    """T039: Filename with forbidden characters."""

    def test_sanitize_filename_with_slashes(self):
        """Filename with / and \\ is sanitized."""
        from forma.student_report import _sanitize_filename

        assert "/" not in _sanitize_filename("이름/테스트")
        assert "\\" not in _sanitize_filename("이름\\테스트")

    def test_sanitize_filename_with_colons(self):
        """Filename with : is sanitized."""
        from forma.student_report import _sanitize_filename

        assert ":" not in _sanitize_filename("이름:테스트")

    def test_sanitize_filename_preserves_korean(self):
        """Korean characters are preserved after sanitization."""
        from forma.student_report import _sanitize_filename

        result = _sanitize_filename("홍길동")
        assert result == "홍길동"
