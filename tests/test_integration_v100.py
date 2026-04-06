"""Integration tests for v0.10.0 cross-feature interactions.

T063: End-to-end integration tests covering:
  - InterventionLog + LongitudinalStore -> compute_intervention_effects / compute_type_summary
  - ConceptDependency DAG parse/validate + LearningPath generation + class deficit map
  - GradePredictor cold start + load_grade_mapping + backward compat fallback
  - Backward compatibility: professor_report.generate_pdf and student_report.generate_pdf
    without any new v0.10.0 kwargs
  - CLI entry points: forma-intervention --help, forma-train-grade --help,
    cli_report.py --intervention-log ignored (FR-013)
  - Independent feature toggles: each feature works without the others
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from forma.evaluation_types import LongitudinalRecord
from forma.longitudinal_store import LongitudinalStore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_N_STUDENTS = 12
_STUDENTS = [f"S{i:03d}" for i in range(1, _N_STUDENTS + 1)]
_WEEKS = [1, 2, 3, 4, 5, 6]
_QUESTIONS = [1, 2]

_CONCEPTS_BY_WEEK = {
    1: {"A": 0.50, "B": 0.45, "C": 0.35, "D": 0.30},
    2: {"A": 0.55, "B": 0.48, "C": 0.38, "D": 0.33},
    3: {"A": 0.60, "B": 0.52, "C": 0.42, "D": 0.36},
    4: {"A": 0.65, "B": 0.56, "C": 0.46, "D": 0.40},
    5: {"A": 0.70, "B": 0.60, "C": 0.50, "D": 0.44},
    6: {"A": 0.75, "B": 0.64, "C": 0.55, "D": 0.48},
}


def _student_base_score(student_id: str) -> float:
    """Deterministic base score spread from ~0.15 to ~0.85."""
    idx = int(student_id[1:])
    return 0.10 + (idx / _N_STUDENTS) * 0.80


def _build_store(tmp_path) -> LongitudinalStore:
    """Build a longitudinal store with 12 students x 6 weeks x 2 questions."""
    store_path = str(tmp_path / "store.yaml")
    store = LongitudinalStore(store_path)
    random.seed(42)

    for week in _WEEKS:
        for sid in _STUDENTS:
            base = _student_base_score(sid)
            week_bonus = (week - 1) * 0.02
            for qsn in _QUESTIONS:
                noise = random.uniform(-0.04, 0.04)
                score = max(0.0, min(1.0, base + week_bonus + noise))

                tier_level = 3 if score >= 0.85 else 2 if score >= 0.65 else 1 if score >= 0.45 else 0
                tier_label = (
                    "Advanced"
                    if tier_level == 3
                    else "Proficient"
                    if tier_level == 2
                    else "Developing"
                    if tier_level == 1
                    else "Beginning"
                )

                record = LongitudinalRecord(
                    student_id=sid,
                    week=week,
                    question_sn=qsn,
                    scores={
                        "ensemble_score": round(score, 4),
                        "concept_coverage": round(max(0, score - 0.1), 4),
                    },
                    tier_level=tier_level,
                    tier_label=tier_label,
                    concept_scores=_CONCEPTS_BY_WEEK.get(week, {}),
                    edge_f1=round(max(0.0, score - 0.12), 4),
                    misconception_count=max(0, 3 - tier_level),
                )
                store.add_record(record)

    store.save()
    return store


def _build_intervention_log(tmp_path):
    """Build intervention log with records for several students."""
    from forma.intervention_store import InterventionLog

    log_path = str(tmp_path / "intervention_log.yaml")
    records = [
        {
            "id": 1,
            "student_id": "S001",
            "week": 2,
            "intervention_type": "면담",
            "description": "1:1 상담",
            "recorded_at": "2026-01-01T00:00:00+00:00",
            "outcome": None,
        },
        {
            "id": 2,
            "student_id": "S002",
            "week": 2,
            "intervention_type": "보충학습",
            "description": "추가 학습",
            "recorded_at": "2026-01-01T00:00:00+00:00",
            "outcome": None,
        },
        {
            "id": 3,
            "student_id": "S003",
            "week": 3,
            "intervention_type": "면담",
            "description": "상담",
            "recorded_at": "2026-01-01T00:00:00+00:00",
            "outcome": None,
        },
    ]
    data = {
        "_meta": {"next_id": len(records) + 1},
        "records": records,
    }
    Path(log_path).write_text(
        yaml.dump(data, allow_unicode=True),
        encoding="utf-8",
    )
    log = InterventionLog(log_path)
    log.load()
    return log


def _build_exam_yaml_with_deps(tmp_path):
    """Build exam YAML with concept_dependencies chain: A -> B -> C -> D."""
    exam = {
        "title": "통합 테스트 시험",
        "subject": "생물학",
        "questions": [
            {"id": 1, "text": "항상성이란?", "concepts": ["항상성", "삼투"]},
            {"id": 2, "text": "확산 원리?", "concepts": ["확산", "능동수송"]},
        ],
        "concept_dependencies": [
            {"prerequisite": "항상성", "dependent": "삼투"},
            {"prerequisite": "삼투", "dependent": "확산"},
            {"prerequisite": "확산", "dependent": "능동수송"},
        ],
    }
    path = tmp_path / "exam_with_deps.yaml"
    path.write_text(yaml.dump(exam, allow_unicode=True), encoding="utf-8")
    return str(path)


def _build_grade_mapping(tmp_path):
    """Build grade mapping YAML for grade prediction tests."""
    random.seed(99)
    mapping = {
        "2025-2": {sid: random.choice(["A", "B", "C", "D", "F"]) for sid in _STUDENTS},
    }
    path = tmp_path / "grade_mapping.yaml"
    path.write_text(yaml.dump(mapping, allow_unicode=True), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# 1. TestInterventionPipeline (~3 tests)
# ---------------------------------------------------------------------------


class TestInterventionPipeline:
    """InterventionLog + LongitudinalStore -> effects + type summaries."""

    def test_compute_effects_sufficient_data(self, tmp_path):
        """Records with enough pre/post weeks produce effects with sufficient_data=True.

        Creates log with 3 records at weeks 2, 2, 3. With window=2 and 6 weeks
        of store data, all three should have sufficient pre/post data.
        Verifies pre_mean, post_mean, score_change, sufficient_data fields.
        """
        from forma.intervention_effect import compute_intervention_effects
        from forma.intervention_store import InterventionLog

        store = _build_store(tmp_path)

        log_path = str(tmp_path / "intervention.yaml")
        log = InterventionLog(log_path)
        log.load()

        # S003 at week 3 -- pre weeks 1,2 and post weeks 4,5 available
        log.add_record("S003", 3, "면담", description="상담 진행")
        # S006 at week 3 -- pre weeks 1,2 and post weeks 4,5 available
        log.add_record("S006", 3, "보충학습", description="보충 과제")
        # S009 at week 4 -- pre weeks 2,3 and post weeks 5,6 available
        log.add_record("S009", 4, "멘토링", description="멘토 배정")
        log.save()

        # Reload to verify persistence roundtrip
        log2 = InterventionLog(log_path)
        log2.load()

        effects = compute_intervention_effects(log2, store, window=2)

        assert len(effects) == 3
        for eff in effects:
            assert eff.sufficient_data is True, (
                f"Student {eff.student_id} at week {eff.intervention_week} should have sufficient data with window=2"
            )
            assert eff.pre_mean is not None
            assert eff.post_mean is not None
            assert eff.score_change is not None
            assert isinstance(eff.score_change, float)
            # score_change = post_mean - pre_mean
            assert abs(eff.score_change - (eff.post_mean - eff.pre_mean)) < 1e-9

    def test_compute_effects_insufficient_data(self, tmp_path):
        """Intervention at week 1 with window=2 has no pre data -> sufficient_data=False."""
        from forma.intervention_effect import compute_intervention_effects
        from forma.intervention_store import InterventionLog

        store = _build_store(tmp_path)

        log_path = str(tmp_path / "intervention.yaml")
        log = InterventionLog(log_path)
        log.load()

        # Week 1 intervention: no pre weeks available for window=2
        log.add_record("S005", 1, "기타", description="조기 개입")
        log.save()

        effects = compute_intervention_effects(log, store, window=2)

        assert len(effects) == 1
        eff = effects[0]
        assert eff.sufficient_data is False
        assert eff.pre_mean is None
        assert eff.post_mean is None
        assert eff.score_change is None

    def test_compute_type_summary_aggregates(self, tmp_path):
        """compute_type_summary aggregates effects correctly by intervention type."""
        from forma.intervention_effect import (
            compute_intervention_effects,
            compute_type_summary,
        )

        store = _build_store(tmp_path)
        log = _build_intervention_log(tmp_path)

        effects = compute_intervention_effects(log, store, window=2)
        summaries = compute_type_summary(effects)

        assert len(summaries) >= 1

        summary_map = {s.intervention_type: s for s in summaries}
        assert "면담" in summary_map
        assert "보충학습" in summary_map

        # 면담: S001 week 2 + S003 week 3 = 2 records
        s_meeting = summary_map["면담"]
        assert s_meeting.n_total == 2

        # 보충학습: S002 week 2 = 1 record
        s_supp = summary_map["보충학습"]
        assert s_supp.n_total == 1

        # For all sufficient effects, n_positive + n_negative <= n_sufficient
        for s in summaries:
            assert s.n_positive + s.n_negative <= s.n_sufficient


# ---------------------------------------------------------------------------
# 2. TestLearningPathPipeline (~3 tests)
# ---------------------------------------------------------------------------


class TestLearningPathPipeline:
    """ConceptDependency DAG + generate_learning_path + class deficit map."""

    def test_dag_parse_build_validate(self, tmp_path):
        """Parse concept_dependencies from exam YAML, build valid DAG."""
        from forma.concept_dependency import (
            build_and_validate_dag,
            parse_concept_dependencies,
        )

        exam_path = _build_exam_yaml_with_deps(tmp_path)
        with open(exam_path, encoding="utf-8") as f:
            exam_yaml = yaml.safe_load(f)

        deps = parse_concept_dependencies(exam_yaml)
        assert deps is not None
        assert len(deps) == 3

        dag = build_and_validate_dag(deps)
        assert dag.nodes == {"항상성", "삼투", "확산", "능동수송"}
        assert len(dag.edges) == 3

        # Verify chain: 항상성 -> 삼투 -> 확산 -> 능동수송
        assert dag.predecessors("항상성") == []
        assert dag.successors("능동수송") == []
        assert dag.predecessors("능동수송") == ["확산"]
        assert dag.successors("항상성") == ["삼투"]

    def test_learning_path_topological_order(self, tmp_path):
        """Student with deficit in 삼투, 확산, 능동수송 gets path in topological order."""
        from forma.concept_dependency import (
            build_and_validate_dag,
            parse_concept_dependencies,
        )
        from forma.learning_path import generate_learning_path

        exam_path = _build_exam_yaml_with_deps(tmp_path)
        with open(exam_path, encoding="utf-8") as f:
            exam_yaml = yaml.safe_load(f)

        deps = parse_concept_dependencies(exam_yaml)
        dag = build_and_validate_dag(deps)

        # 항상성 mastered (0.8), 삼투/확산/능동수송 deficit (< 0.4)
        student_scores = {"항상성": 0.80, "삼투": 0.20, "확산": 0.15, "능동수송": 0.10}

        path = generate_learning_path("S001", student_scores, dag, threshold=0.4)

        assert path.student_id == "S001"
        # 항상성 is mastered so not included
        assert "항상성" not in path.ordered_path
        assert "삼투" in path.ordered_path
        assert "확산" in path.ordered_path
        assert "능동수송" in path.ordered_path

        # Verify topological order: 삼투 before 확산 before 능동수송
        idx_osm = path.ordered_path.index("삼투")
        idx_dif = path.ordered_path.index("확산")
        idx_act = path.ordered_path.index("능동수송")
        assert idx_osm < idx_dif < idx_act, (
            f"Topological order violated: 삼투@{idx_osm}, 확산@{idx_dif}, 능동수송@{idx_act}"
        )

        # deficit_concepts should include all three
        assert set(path.deficit_concepts) == {"삼투", "확산", "능동수송"}

    def test_class_deficit_map_counts(self, tmp_path):
        """build_class_deficit_map counts per-concept deficits across students."""
        from forma.concept_dependency import (
            build_and_validate_dag,
            parse_concept_dependencies,
        )
        from forma.learning_path import build_class_deficit_map

        exam_path = _build_exam_yaml_with_deps(tmp_path)
        with open(exam_path, encoding="utf-8") as f:
            exam_yaml = yaml.safe_load(f)

        deps = parse_concept_dependencies(exam_yaml)
        dag = build_and_validate_dag(deps)

        # 4 students with varying deficits
        all_scores = {
            "S001": {"항상성": 0.8, "삼투": 0.3, "확산": 0.2, "능동수송": 0.1},  # deficit: 삼투,확산,능동수송
            "S002": {"항상성": 0.9, "삼투": 0.7, "확산": 0.3, "능동수송": 0.5},  # deficit: 확산
            "S003": {"항상성": 0.5, "삼투": 0.5, "확산": 0.5, "능동수송": 0.5},  # no deficit
            "S004": {"항상성": 0.1, "삼투": 0.1, "확산": 0.1, "능동수송": 0.1},  # deficit: all
        }

        deficit_map = build_class_deficit_map(all_scores, dag, threshold=0.4)

        assert deficit_map.total_students == 4
        assert deficit_map.dag is dag

        # 항상성: deficit for S004 only -> 1
        assert deficit_map.concept_counts["항상성"] == 1
        # 삼투: deficit for S001, S004 -> 2
        assert deficit_map.concept_counts["삼투"] == 2
        # 확산: deficit for S001, S002, S004 -> 3
        assert deficit_map.concept_counts["확산"] == 3
        # 능동수송: deficit for S001, S004 -> 2
        assert deficit_map.concept_counts["능동수송"] == 2


# ---------------------------------------------------------------------------
# 3. TestGradePredictionPipeline (~3 tests)
# ---------------------------------------------------------------------------


class TestGradePredictionPipeline:
    """GradePredictor cold start, load_grade_mapping, backward compat fallback."""

    def test_predict_cold_start_known_scores(self):
        """predict_cold_start with known feature values produces valid A/B/C/D/F grades."""
        from forma.grade_predictor import (
            GRADE_FEATURE_NAMES,
            VALID_GRADES,
            GradePredictor,
        )

        predictor = GradePredictor()

        n_features = len(GRADE_FEATURE_NAMES)
        student_ids = ["S_high", "S_mid", "S_low", "S_vlow"]

        # Set score_mean (idx 0) and score_slope (idx 2), rest zeros
        matrix = np.zeros((4, n_features), dtype=float)
        # S_high: projected = 0.90 + 0.05*0.5 + 0 = 0.925 -> A
        matrix[0, 0] = 0.90
        matrix[0, 2] = 0.05
        # S_mid: projected = 0.65 + 0.02*0.5 + 0 = 0.66 -> B (>= 0.50)
        matrix[1, 0] = 0.65
        matrix[1, 2] = 0.02
        # S_low: projected = 0.40 + (-0.02)*0.5 + 0 = 0.39 -> D
        matrix[2, 0] = 0.40
        matrix[2, 2] = -0.02
        # S_vlow: projected = 0.15 + (-0.05)*0.5 + 0 = 0.125 -> F
        matrix[3, 0] = 0.15
        matrix[3, 2] = -0.05

        predictions = predictor.predict_cold_start(
            matrix,
            student_ids,
            GRADE_FEATURE_NAMES,
        )

        assert len(predictions) == 4
        for pred in predictions:
            assert pred.predicted_grade in VALID_GRADES
            assert pred.is_model_based is False
            assert pred.confidence == "limited"
            assert pred.predicted_ordinal in (0, 1, 2, 3, 4)
            # Cold start probabilities: 1.0 for predicted, 0.0 for others
            total_prob = sum(pred.grade_probabilities.values())
            assert abs(total_prob - 1.0) < 1e-9

        # S_high should predict A (projected >= 0.85)
        assert predictions[0].predicted_grade == "A"
        # S_vlow should predict F (projected < 0.30)
        assert predictions[3].predicted_grade == "F"

    def test_load_grade_mapping_valid_yaml(self, tmp_path):
        """load_grade_mapping loads and validates a well-formed YAML file."""
        from forma.grade_predictor import load_grade_mapping

        grade_data = {
            "2025-1": {
                "S001": "A",
                "S002": "B",
                "S003": "C",
                "S004": "D",
                "S005": "F",
            },
            "2025-2": {
                "S001": "B",
                "S002": "A",
                "S003": "B",
            },
        }

        grade_path = str(tmp_path / "grades.yaml")
        with open(grade_path, "w", encoding="utf-8") as f:
            yaml.dump(grade_data, f, allow_unicode=True)

        result = load_grade_mapping(grade_path)

        assert "2025-1" in result
        assert "2025-2" in result
        assert len(result["2025-1"]) == 5
        assert result["2025-1"]["S001"] == "A"
        assert result["2025-2"]["S002"] == "A"

    def test_grade_predictor_cold_start_fallback(self):
        """GradePredictor with insufficient data raises ValueError; cold start still works."""
        from forma.grade_predictor import (
            GRADE_FEATURE_NAMES,
            VALID_GRADES,
            GradePredictor,
        )

        predictor = GradePredictor()

        # Only 3 students -- below min_students=10 for training
        n_features = len(GRADE_FEATURE_NAMES)
        rng = np.random.default_rng(42)
        matrix = rng.random((3, n_features))
        student_ids = ["S001", "S002", "S003"]
        labels = np.array([4, 2, 0])  # A, C, F

        # Training should raise ValueError due to insufficient students
        with pytest.raises(ValueError, match="Insufficient students"):
            predictor.train(matrix, labels, GRADE_FEATURE_NAMES, min_students=10)

        # Cold start fallback should still work
        predictions = predictor.predict_cold_start(
            matrix,
            student_ids,
            GRADE_FEATURE_NAMES,
        )
        assert len(predictions) == 3
        for pred in predictions:
            assert pred.predicted_grade in VALID_GRADES
            assert pred.is_model_based is False
            assert pred.confidence == "limited"


# ---------------------------------------------------------------------------
# 4. TestBackwardCompatibility (~3 tests)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Verify generate_pdf() works WITHOUT any new v0.10.0 kwargs."""

    def _make_professor_report_data(self):
        """Minimal ProfessorReportData for testing."""
        from forma.professor_report_data import (
            ProfessorReportData,
            QuestionClassStats,
            StudentSummaryRow,
        )

        return ProfessorReportData(
            class_name="1A",
            week_num=3,
            subject="테스트",
            exam_title="형성평가",
            generation_date="2026-01-01",
            n_students=3,
            n_questions=1,
            class_ensemble_mean=0.5,
            class_ensemble_std=0.1,
            class_ensemble_median=0.5,
            class_ensemble_q1=0.4,
            class_ensemble_q3=0.6,
            overall_level_distribution={
                "Advanced": 0,
                "Proficient": 1,
                "Developing": 1,
                "Beginning": 1,
            },
            question_stats=[QuestionClassStats(question_sn=1)],
            student_rows=[StudentSummaryRow(student_id=f"S{i:03d}") for i in range(1, 4)],
            n_at_risk=1,
            pct_at_risk=33.3,
        )

    def test_professor_report_no_new_kwargs(self, tmp_path):
        """professor_report generate_pdf() works without intervention_effects, deficit_map, grade_predictions."""
        from forma.professor_report import ProfessorPDFReportGenerator

        report_data = self._make_professor_report_data()

        gen = ProfessorPDFReportGenerator()
        # Call with NO new v0.10.0 kwargs -- backward compat
        pdf_path = gen.generate_pdf(report_data, str(tmp_path))

        assert pdf_path.endswith(".pdf")
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0

    def test_professor_report_new_kwargs_none(self, tmp_path):
        """professor_report generate_pdf() accepts new kwargs explicitly set to None."""
        from forma.professor_report import ProfessorPDFReportGenerator

        report_data = self._make_professor_report_data()

        gen = ProfessorPDFReportGenerator()
        pdf_path = gen.generate_pdf(
            report_data,
            str(tmp_path),
            intervention_effects=None,
            intervention_type_summaries=None,
            deficit_map=None,
            grade_predictions=None,
        )

        assert pdf_path.endswith(".pdf")
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0

    def test_student_report_no_new_kwargs(self, tmp_path):
        """student_report generate_pdf() works WITHOUT learning_path and grade_trend."""
        from forma.report_data_loader import (
            ClassDistributions,
            StudentReportData,
        )
        from forma.student_report import StudentPDFReportGenerator

        student = StudentReportData(
            student_id="S001",
            course_name="생물학",
            chapter_name="세포",
            questions=[],
        )
        dists = ClassDistributions()

        gen = StudentPDFReportGenerator()
        # Call with NO new kwargs -- backward compat
        pdf_path = gen.generate_pdf(student, dists, str(tmp_path))

        assert pdf_path.endswith(".pdf")
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0


# ---------------------------------------------------------------------------
# 5. TestCLIEntryPoints (~3 tests)
# ---------------------------------------------------------------------------


class TestCLIEntryPoints:
    """CLI --help exits cleanly and FR-013 compliance."""

    def test_forma_intervention_help_exit_0(self):
        """forma-intervention --help exits with code 0."""
        from forma.cli_intervention import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_forma_train_grade_help_exit_0(self):
        """forma-train-grade --help exits with code 0."""
        from forma.cli_train_grade import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_cli_report_intervention_log_ignored_fr013(self):
        """FR-013: student report CLI accepts --intervention-log but ignores it.

        The flag exists for config merge compatibility but is not used to
        include intervention data in student reports.
        """
        from forma.cli_report import _build_parser

        parser = _build_parser()

        # Without --intervention-log: defaults to None
        args_no_flag = parser.parse_args(
            [
                "--final",
                "final.yaml",
                "--config",
                "config.yaml",
                "--eval-dir",
                "eval/",
                "--output-dir",
                "out/",
                "--no-config",
            ]
        )
        assert args_no_flag.intervention_log is None

        # With --intervention-log: accepted but value is captured (ignored at runtime)
        args_with_flag = parser.parse_args(
            [
                "--final",
                "final.yaml",
                "--config",
                "config.yaml",
                "--eval-dir",
                "eval/",
                "--output-dir",
                "out/",
                "--intervention-log",
                "log.yaml",
                "--no-config",
            ]
        )
        assert args_with_flag.intervention_log == "log.yaml"


# ---------------------------------------------------------------------------
# 6. TestIndependentFeatureToggles (~2 tests)
# ---------------------------------------------------------------------------


class TestIndependentFeatureToggles:
    """Each v0.10.0 feature works independently without the others."""

    def test_intervention_without_learning_path_or_grade(self, tmp_path):
        """Intervention pipeline works without any concept DAG or grade data."""
        from forma.intervention_effect import (
            compute_intervention_effects,
            compute_type_summary,
        )
        from forma.intervention_store import InterventionLog

        store = _build_store(tmp_path)

        log_path = str(tmp_path / "intervention.yaml")
        log = InterventionLog(log_path)
        log.load()
        log.add_record("S003", 3, "면담", description="독립 테스트")
        log.save()

        effects = compute_intervention_effects(log, store, window=2)
        assert len(effects) == 1
        assert effects[0].sufficient_data is True

        summaries = compute_type_summary(effects)
        assert len(summaries) == 1
        assert summaries[0].intervention_type == "면담"

        # No concept_dependency or grade_predictor imports needed

    def test_learning_path_without_intervention_or_grade(self):
        """Learning path generation works without intervention log or grade model."""
        from forma.concept_dependency import (
            build_and_validate_dag,
            parse_concept_dependencies,
        )
        from forma.learning_path import generate_learning_path

        exam_yaml = {
            "concept_dependencies": [
                {"prerequisite": "X", "dependent": "Y"},
                {"prerequisite": "Y", "dependent": "Z"},
            ],
        }

        deps = parse_concept_dependencies(exam_yaml)
        dag = build_and_validate_dag(deps)

        # Student with deficit in Y and Z
        scores = {"X": 0.9, "Y": 0.2, "Z": 0.1}
        path = generate_learning_path("SOLO", scores, dag, threshold=0.4)

        assert len(path.ordered_path) >= 2
        assert path.ordered_path.index("Y") < path.ordered_path.index("Z")
        # No InterventionLog or GradePredictor needed
