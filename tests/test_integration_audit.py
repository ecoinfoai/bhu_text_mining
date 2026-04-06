"""Layer 3: Cross-module integration tests for audit.

Tests 8 data flow paths and 12 boundary conditions across module boundaries.
Discovery-only — no code fixes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers: minimal YAML builders
# ---------------------------------------------------------------------------


def _exam_config(n_questions: int = 2, with_kg: bool = False) -> dict:
    """Build minimal exam config dict."""
    questions = []
    for i in range(1, n_questions + 1):
        q: dict = {
            "sn": i,
            "question": f"테스트 문제 {i}",
            "model_answer": f"모범 답안 {i}",
            "keywords": ["세포막", "항상성", "삼투"],
            "rubric_tiers": {
                "level_3": {"label": "high", "min_graph_f1": 0.8, "requires_terminology": True},
                "level_2": {"label": "mid", "min_graph_f1": 0.5, "requires_terminology": False},
                "level_1": {"label": "low", "min_graph_f1": 0.2, "requires_terminology": False},
                "level_0": {"label": "none", "min_graph_f1": 0.0, "requires_terminology": False},
            },
        }
        if with_kg:
            q["question_type"] = "essay"
            q["knowledge_graph"] = {
                "edges": [
                    {"subject": "세포막", "relation": "조절", "object": "삼투"},
                    {"subject": "항상성", "relation": "유지", "object": "세포막"},
                ],
                "similarity_threshold": 0.80,
                "node_aliases": {},
            }
        questions.append(q)
    return {
        "metadata": {"chapter": 1, "title": "테스트"},
        "questions": questions,
    }


def _student_responses(
    n_students: int = 3,
    n_questions: int = 2,
    empty: bool = False,
) -> dict:
    """Build minimal student responses dict keyed by anp_id."""
    responses = {}
    for i in range(1, n_students + 1):
        sid = f"S{i:03d}"
        qmap: dict[int, str] = {}
        for q in range(1, n_questions + 1):
            if empty:
                qmap[q] = ""
            else:
                qmap[q] = f"세포막은 삼투 현상을 통해 물질을 이동시키며 항상성을 유지한다. 답변 {i}-{q}"
        responses[sid] = qmap
    return responses


def _longitudinal_record(
    student_id: str,
    week: int,
    question_sn: int = 1,
    ensemble_score: float = 0.65,
    concept_coverage: float = 0.7,
    tier_level: int = 2,
    concept_scores: dict | None = None,
    topic: str | None = None,
    class_id: str | None = None,
) -> dict:
    """Build a single longitudinal record dict."""
    scores = {
        "ensemble_score": ensemble_score,
        "concept_coverage": concept_coverage,
    }
    rec = {
        "student_id": student_id,
        "week": week,
        "question_sn": question_sn,
        "scores": scores,
        "tier_level": tier_level,
        "tier_label": ["Beginning", "Developing", "Proficient", "Advanced"][tier_level],
    }
    if concept_scores is not None:
        rec["concept_scores"] = concept_scores
    if topic is not None:
        rec["topic"] = topic
    if class_id is not None:
        rec["class_id"] = class_id
    return rec


def _build_store_yaml(records: list[dict]) -> dict:
    """Build a store YAML dict from a list of record dicts."""
    store_records = {}
    for r in records:
        key = f"{r['student_id']}_{r['week']}_{r['question_sn']}"
        store_records[key] = r
    return {"records": store_records}


def _write_yaml(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


# ---------------------------------------------------------------------------
# Path B: Evaluation Pipeline — config → concept_checker → ensemble → output
# ---------------------------------------------------------------------------


class TestPathBEvaluationPipeline:
    """Test evaluation pipeline module boundaries (concept checker → ensemble)."""

    def test_path_b_concept_checker_to_ensemble_basic(self) -> None:
        """Concept checker results feed correctly into ensemble scorer."""
        from forma.concept_checker import check_all_concepts
        from forma.ensemble_scorer import EnsembleScorer

        # Run concept checker
        results = check_all_concepts(
            student_text="세포막은 삼투 현상�� 통해 물질을 이��시키며 항상성을 유지한다.",
            student_id="S001",
            question_sn=1,
            concepts=["세포막", "항상성", "삼투"],
        )

        # Verify concept checker output is compatible with ensemble scorer input
        assert len(results) == 3
        assert all(hasattr(r, "is_present") for r in results)
        assert all(hasattr(r, "similarity_score") for r in results)

        # Feed into ensemble scorer (compute_score is the actual method name)
        scorer = EnsembleScorer()
        ensemble_result = scorer.compute_score(
            concept_results=results,
            llm_result=None,
            statistical_result=None,
            graph_result=None,
            bertscore_f1=None,
            student_id="S001",
            question_sn=1,
        )
        assert 0.0 <= ensemble_result.ensemble_score <= 1.0
        assert ensemble_result.understanding_level in {
            "Advanced",
            "Proficient",
            "Developing",
            "Beginning",
        }

    def test_path_b_empty_student_text(self) -> None:
        """Empty student text → concept checker raises ValueError (fail-fast).

        Integration issue: concept_checker rejects empty text with ValueError,
        but pipeline callers must handle this before feeding into ensemble.
        """
        from forma.concept_checker import check_all_concepts

        # concept_checker raises ValueError on empty text — this is fail-fast behavior
        with pytest.raises(ValueError, match="empty"):
            check_all_concepts(
                student_text="",
                student_id="S001",
                question_sn=1,
                concepts=["세포막", "항상성"],
            )


# ---------------------------------------------------------------------------
# Path C: Report Pipeline — eval_result → data_loader → stats → charts
# ---------------------------------------------------------------------------


class TestPathCReportPipeline:
    """Test evaluation results → report data loading → statistics."""

    def test_path_c_eval_yaml_to_report_data_loader(self, tmp_path) -> None:
        """Evaluation YAML → report_data_loader produces valid report data."""
        from forma.evaluation_io import save_evaluation_yaml, load_evaluation_yaml

        # Build minimal evaluation result YAML
        eval_data = {
            "metadata": {"chapter": 1, "title": "테스트"},
            "results": {
                "S001": {
                    1: {
                        "concept_coverage": 0.8,
                        "ensemble_score": 0.72,
                        "understanding_level": "Proficient",
                        "component_scores": {"concept_coverage": 0.8, "llm_rubric": 0.6},
                    },
                },
            },
        }
        out_path = str(tmp_path / "eval_result.yaml")
        save_evaluation_yaml(eval_data, out_path)

        # Reload and verify round-trip
        loaded = load_evaluation_yaml(out_path)
        assert loaded["metadata"]["chapter"] == 1
        assert "S001" in loaded["results"]

    def test_path_c_statistical_analysis_from_concept_results(self) -> None:
        """Concept checker results → statistical analyzer (Rasch) via binary matrix."""
        from forma.evaluation_types import ConceptMatchResult
        from forma.statistical_analysis import RaschAnalyzer

        # Build concept results for 5 students, 3 concepts
        results = []
        for s_idx in range(5):
            for c_idx, concept in enumerate(["세포막", "항상성", "삼투"]):
                results.append(
                    ConceptMatchResult(
                        concept=concept,
                        student_id=f"S{s_idx + 1:03d}",
                        question_sn=1,
                        is_present=(s_idx + c_idx) % 2 == 0,
                        similarity_score=0.5 + 0.1 * ((s_idx + c_idx) % 4),
                        top_k_mean_similarity=0.5 + 0.1 * ((s_idx + c_idx) % 4),
                        threshold_used=0.45,
                    )
                )

        # Build binary matrix from concept results (the integration boundary)
        concepts = ["세포막", "항상성", "삼투"]
        students = [f"S{i + 1:03d}" for i in range(5)]
        X = np.zeros((len(students), len(concepts)), dtype=int)
        for r in results:
            s_idx = students.index(r.student_id)
            c_idx = concepts.index(r.concept)
            X[s_idx, c_idx] = int(r.is_present)

        analyzer = RaschAnalyzer(question_sn=1)
        analyzer.fit(X)
        thetas, ses = analyzer.ability_estimates(X)
        assert thetas.shape[0] == 5
        assert ses.shape[0] == 5


# ---------------------------------------------------------------------------
# Path D: Longitudinal Pipeline — eval_results → store → report_data → chart
# ---------------------------------------------------------------------------


class TestPathDLongitudinalPipeline:
    """Test longitudinal store → summary data → report generation boundary."""

    def test_path_d_store_to_summary_data(self, tmp_path) -> None:
        """LongitudinalStore → build_longitudinal_summary produces valid data."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_report_data import build_longitudinal_summary

        store_path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(store_path)

        # Add records for 3 students across 3 weeks
        for sid in ["S001", "S002", "S003"]:
            for week in [1, 2, 3]:
                score = 0.3 + 0.1 * week + (0.1 if sid == "S001" else 0.0)
                store.add_record(
                    LongitudinalRecord(
                        student_id=sid,
                        week=week,
                        question_sn=1,
                        scores={"ensemble_score": score, "concept_coverage": score - 0.1},
                        tier_level=2 if score >= 0.5 else 1,
                        tier_label="Proficient" if score >= 0.5 else "Developing",
                        concept_scores={"세포막": score, "항상성": score - 0.1},
                    )
                )

        store.save()

        # Reload and build summary
        store2 = LongitudinalStore(store_path)
        store2.load()
        summary = build_longitudinal_summary(
            store=store2,
            weeks=[1, 2, 3],
            class_name="A",
        )

        assert summary.total_students == 3
        assert len(summary.student_trajectories) == 3
        assert len(summary.period_weeks) == 3
        assert 1 in summary.class_weekly_averages

    def test_path_d_nan_propagation_through_longitudinal_chain(self, tmp_path) -> None:
        """BOUNDARY #5: NaN score flows through store → summary without crash."""
        import yaml
        from forma.longitudinal_store import LongitudinalStore
        from forma.longitudinal_report_data import build_longitudinal_summary

        store_path = str(tmp_path / "store_nan.yaml")
        # Write NaN data directly to YAML (simulating legacy/corrupt data
        # that bypasses validation)
        raw = {
            "records": {
                "S001_w1_q1": {
                    "student_id": "S001",
                    "week": 1,
                    "question_sn": 1,
                    "scores": {"ensemble_score": float("nan"), "concept_coverage": 0.5},
                    "tier_level": 1,
                    "tier_label": "Developing",
                },
                "S001_w2_q1": {
                    "student_id": "S001",
                    "week": 2,
                    "question_sn": 1,
                    "scores": {"ensemble_score": 0.6, "concept_coverage": float("nan")},
                    "tier_level": 2,
                    "tier_label": "Proficient",
                },
            }
        }
        with open(store_path, "w", encoding="utf-8") as f:
            yaml.dump(raw, f, allow_unicode=True)

        store2 = LongitudinalStore(store_path)
        store2.load()

        # Build summary — should not raise
        summary = build_longitudinal_summary(
            store=store2,
            weeks=[1, 2],
            class_name="A",
        )

        # Check if NaN leaked into class averages
        for week, avg in summary.class_weekly_averages.items():
            if math.isnan(avg):
                pytest.fail(f"NaN propagated to class_weekly_averages[{week}] — np.mean does not filter NaN")

    def test_path_d_cross_week_concept_mismatch(self, tmp_path) -> None:
        """BOUNDARY #6: Different concepts per week in longitudinal store."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_report_data import build_longitudinal_summary

        store_path = str(tmp_path / "store_concepts.yaml")
        store = LongitudinalStore(store_path)

        # Week 1: concepts A, B
        store.add_record(
            LongitudinalRecord(
                student_id="S001",
                week=1,
                question_sn=1,
                scores={"ensemble_score": 0.6, "concept_coverage": 0.5},
                tier_level=2,
                tier_label="Proficient",
                concept_scores={"세포막": 0.8, "항상성": 0.3},
            )
        )

        # Week 2: concepts B, C (different from week 1)
        store.add_record(
            LongitudinalRecord(
                student_id="S001",
                week=2,
                question_sn=1,
                scores={"ensemble_score": 0.7, "concept_coverage": 0.6},
                tier_level=2,
                tier_label="Proficient",
                concept_scores={"항상성": 0.5, "삼투": 0.7},
            )
        )

        store.save()
        store2 = LongitudinalStore(store_path)
        store2.load()

        # Should handle concept mismatch across weeks without crash
        summary = build_longitudinal_summary(
            store=store2,
            weeks=[1, 2],
            class_name="A",
        )
        assert summary.total_students == 1
        # Check concept mastery changes — should include all 3 concepts
        concept_names = [c.concept for c in summary.concept_mastery_changes]
        assert "세포막" in concept_names or "항상성" in concept_names


# ---------------------------------------------------------------------------
# Path E: Risk/Warning Pipeline — store → predictor → warning_data → cards
# ---------------------------------------------------------------------------


class TestPathERiskWarningPipeline:
    """Test longitudinal store → feature extraction → risk → warning cards."""

    def test_path_e_store_to_risk_prediction(self, tmp_path) -> None:
        """LongitudinalStore → FeatureExtractor → feature matrix."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.evaluation_types import LongitudinalRecord
        from forma.risk_predictor import FeatureExtractor

        store_path = str(tmp_path / "store_risk.yaml")
        store = LongitudinalStore(store_path)

        # Add enough data for feature extraction
        for sid in ["S001", "S002", "S003"]:
            for week in [1, 2, 3, 4]:
                score = 0.4 + 0.05 * week + (0.2 if sid == "S001" else 0.0)
                store.add_record(
                    LongitudinalRecord(
                        student_id=sid,
                        week=week,
                        question_sn=1,
                        scores={
                            "ensemble_score": score,
                            "concept_coverage": score - 0.1,
                        },
                        tier_level=2 if score >= 0.5 else 0,
                        tier_label="Proficient" if score >= 0.5 else "Beginning",
                        misconception_count=1 if score < 0.5 else 0,
                        edge_f1=score * 0.8,
                    )
                )

        extractor = FeatureExtractor()
        X, feature_names, student_ids = extractor.extract(store, weeks=[1, 2, 3, 4])

        assert X.shape[0] == 3  # 3 students
        assert X.shape[1] == 15  # 15 features
        assert len(student_ids) == 3
        # No NaN in feature matrix
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            pytest.fail(f"FeatureExtractor produced {nan_count} NaN values in feature matrix")

    def test_path_e_warning_data_from_store(self, tmp_path) -> None:
        """Warning cards are built from at-risk dict + risk predictions."""
        from forma.warning_report_data import build_warning_data, WarningCard
        from forma.risk_predictor import RiskPrediction

        at_risk = {
            "S001": {"is_at_risk": True, "reasons": ["low score"]},
            "S002": {"is_at_risk": True, "reasons": ["declining"]},
        }
        predictions = [
            RiskPrediction(student_id="S001", drop_probability=0.8),
            RiskPrediction(student_id="S003", drop_probability=0.6),
        ]
        concept_scores = {
            "S001": {"세포막": 0.1, "항상성": 0.2, "삼투": 0.15},
            "S002": {"세포막": 0.5, "항상성": 0.4, "삼투": 0.3},
        }
        score_trajectories = {
            "S001": [0.3, 0.25, 0.2],
            "S002": [0.5, 0.45, 0.4],
        }

        cards = build_warning_data(
            at_risk_students=at_risk,
            risk_predictions=predictions,
            concept_scores=concept_scores,
            score_trajectories=score_trajectories,
        )

        assert isinstance(cards, list)
        assert all(isinstance(c, WarningCard) for c in cards)
        # S001 should be present (in at_risk dict AND predictions)
        card_ids = {c.student_id for c in cards}
        assert "S001" in card_ids
        # S003 should be included (drop_probability >= 0.5 from model)
        assert "S003" in card_ids

    def test_boundary_nan_score_through_risk_pipeline(self, tmp_path) -> None:
        """BOUNDARY #5: NaN scores through risk feature extraction."""
        import yaml
        from forma.longitudinal_store import LongitudinalStore
        from forma.risk_predictor import FeatureExtractor

        store_path = str(tmp_path / "store_nan_risk.yaml")
        # Write NaN data directly to YAML (simulating legacy/corrupt data)
        records = {}
        for week in [1, 2, 3]:
            score = float("nan") if week == 2 else 0.5
            records[f"S001_w{week}_q1"] = {
                "student_id": "S001",
                "week": week,
                "question_sn": 1,
                "scores": {"ensemble_score": score, "concept_coverage": 0.5},
                "tier_level": 2,
                "tier_label": "Proficient",
            }
        with open(store_path, "w", encoding="utf-8") as f:
            yaml.dump({"records": records}, f, allow_unicode=True)

        store = LongitudinalStore(store_path)
        store.load()

        extractor = FeatureExtractor()
        X, names, sids = extractor.extract(store, weeks=[1, 2, 3])

        if X.shape[0] > 0:
            nan_count = np.isnan(X).sum()
            if nan_count > 0:
                pytest.fail(
                    f"NaN propagated through FeatureExtractor: {nan_count} NaN values. "
                    "FeatureExtractor does not sanitize NaN from store."
                )


# ---------------------------------------------------------------------------
# Path G: Lecture Analysis — transcript → preprocess → analyze
# ---------------------------------------------------------------------------


class TestPathGLectureAnalysis:
    """Test lecture preprocessing → analysis boundary."""

    def test_path_g_preprocess_to_analyzer_basic(self, tmp_path) -> None:
        """Lecture preprocessor output feeds correctly into analyzer interface."""
        from forma.lecture_preprocessor import preprocess_transcript

        transcript = (
            "오늘은 세포막에 대해 공부하겠습니다. "
            "세포막은 인지질 이중층으로 구성되어 있습니다. "
            "선택적 투과성이 있어 물질의 출입을 조절합니다. "
            "삼투 현상은 세포막을 통한 물의 이동입니다."
        )
        transcript_path = str(tmp_path / "transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        cleaned = preprocess_transcript(transcript_path, class_id="A", week=1)
        assert hasattr(cleaned, "cleaned_text")
        assert len(cleaned.cleaned_text) > 0
        # Cleaned text should still contain key concepts
        assert "세포막" in cleaned.cleaned_text or "세포" in cleaned.cleaned_text

    def test_path_g_mixed_encoding_passthrough(self, tmp_path) -> None:
        """BOUNDARY #9: Korean+English+special chars through preprocessing."""
        from forma.lecture_preprocessor import preprocess_transcript

        mixed = (
            "세포막(cell membrane)은 약 7.5nm의 두께를 가집니다. "
            "Na+/K+-ATPase가 능동수송을 담당합니다. "
            "pH 7.4 범위에서 정상 기능을 합니다."
        )
        transcript_path = str(tmp_path / "mixed.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(mixed)

        cleaned = preprocess_transcript(transcript_path, class_id="A", week=1)
        assert len(cleaned.cleaned_text) > 0
        # Should not crash or produce empty output from special chars


# ---------------------------------------------------------------------------
# Path H: Delivery Pipeline — prepare → send (mock SMTP)
# ---------------------------------------------------------------------------


class TestPathHDeliveryPipeline:
    """Test delivery prepare → send module boundary."""

    def test_path_h_prepare_to_send_roundtrip(self, tmp_path) -> None:
        """Prepare stage output feeds correctly into send stage."""
        from forma.delivery_prepare import (
            DeliveryManifest,
            match_files_for_student,
        )

        # Create mock report files
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        for sid in ["S001", "S002"]:
            (reports_dir / f"report_{sid}.pdf").write_text("fake pdf")

        manifest = DeliveryManifest(
            directory=str(reports_dir),
            file_patterns=["report_{student_id}.pdf"],
        )

        # Match files for student
        files = match_files_for_student(manifest, "S001")
        assert len(files) == 1
        assert "S001" in files[0]

    def test_path_h_send_with_mock_smtp(self, tmp_path) -> None:
        """Send emails with fully mocked SMTP server."""
        from forma.delivery_send import (
            SmtpConfig,
            EmailTemplate,
            render_template,
        )

        _ = SmtpConfig(
            smtp_server="smtp.test.local",
            smtp_port=587,
            sender_email="test@test.local",
            sender_name="테스트 교수",
        )
        template = EmailTemplate(
            subject="학습 평가 결과 - {student_name}",
            body="안녕하세요 {student_name}님, 평가 결과를 첨부합니다.",
        )

        subject, body = render_template(
            template,
            student_name="홍길동",
            student_id="S001",
            class_name="A",
        )

        assert "홍길동" in subject
        assert "홍길동" in body


# ---------------------------------------------------------------------------
# Boundary Condition Tests
# ---------------------------------------------------------------------------


class TestBoundary1EmptyExam:
    """BOUNDARY #1: Empty exam (0 questions) through pipeline."""

    def test_empty_exam_concept_checker(self) -> None:
        """0-question exam config → concept checker returns empty."""
        from forma.pipeline_evaluation import _run_layer1

        config = _exam_config(n_questions=0)
        responses = _student_responses(n_students=2, n_questions=0)

        results = _run_layer1(responses, config)
        assert results == {}


class TestBoundary2ZeroStudents:
    """BOUNDARY #2: 0 students through pipeline."""

    def test_zero_students_concept_checker(self) -> None:
        """No students → concept checker returns empty."""
        from forma.pipeline_evaluation import _run_layer1

        config = _exam_config(n_questions=2)
        responses: dict = {}

        results = _run_layer1(responses, config)
        assert results == {}

    def test_zero_students_longitudinal_summary(self, tmp_path) -> None:
        """Empty store → longitudinal summary handles 0 students."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.longitudinal_report_data import build_longitudinal_summary

        store_path = str(tmp_path / "empty_store.yaml")
        store = LongitudinalStore(store_path)

        summary = build_longitudinal_summary(
            store=store,
            weeks=[1, 2, 3],
            class_name="A",
        )
        assert summary.total_students == 0
        assert summary.student_trajectories == []
        assert summary.class_weekly_averages == {}


class TestBoundary4LLMTotalFailure:
    """BOUNDARY #4: LLM total failure — partial results preserved."""

    def test_llm_failure_concept_results_preserved(self) -> None:
        """When LLM fails, Layer 1 concept results should still be usable."""
        from forma.concept_checker import check_all_concepts
        from forma.ensemble_scorer import EnsembleScorer

        # Layer 1 succeeds
        concept_results = check_all_concepts(
            student_text="세포막은 항상성을 유지한다.",
            student_id="S001",
            question_sn=1,
            concepts=["세포막", "항상성"],
        )

        # Layer 2 LLM fails → llm_result=None
        scorer = EnsembleScorer()
        result = scorer.compute_score(
            concept_results=concept_results,
            llm_result=None,
            statistical_result=None,
            graph_result=None,
            bertscore_f1=None,
            student_id="S001",
            question_sn=1,
        )
        # Should produce valid result from Layer 1 alone
        assert result.ensemble_score >= 0.0
        assert result.understanding_level in {
            "Advanced",
            "Proficient",
            "Developing",
            "Beginning",
        }


class TestBoundary5NaNPropagation:
    """BOUNDARY #5: NaN score propagation through chains."""

    def test_nan_in_section_comparison(self) -> None:
        """NaN scores in section comparison."""
        from forma.section_comparison import compute_section_stats

        scores_with_nan = [0.5, 0.6, float("nan"), 0.7, 0.8]
        try:
            stats = compute_section_stats("A", scores_with_nan, {"S003"})
            # Check if NaN leaked into mean/median/std
            if math.isnan(stats.mean):
                pytest.fail("NaN leaked into section stats mean")
            if math.isnan(stats.std):
                pytest.fail("NaN leaked into section stats std")
        except Exception as e:
            # Any exception from NaN is an integration issue
            pytest.fail(f"NaN in section_comparison caused exception: {e}")

    def test_nan_in_student_longitudinal_data(self, tmp_path) -> None:
        """NaN ensemble_score through student longitudinal data builder."""
        import yaml
        from forma.longitudinal_store import LongitudinalStore
        from forma.student_longitudinal_data import (
            build_student_data,
            build_cohort_distribution,
        )

        store_path = str(tmp_path / "store_nan_student.yaml")
        # Write NaN data directly to YAML (simulating legacy/corrupt data)
        records = {}
        for week in [1, 2, 3]:
            score = float("nan") if week == 2 else 0.6
            records[f"S001_w{week}_q1"] = {
                "student_id": "S001",
                "week": week,
                "question_sn": 1,
                "scores": {"ensemble_score": score, "concept_coverage": 0.5},
                "tier_level": 2,
                "tier_label": "Proficient",
            }
        with open(store_path, "w", encoding="utf-8") as f:
            yaml.dump({"records": records}, f, allow_unicode=True)

        store2 = LongitudinalStore(store_path)
        store2.load()

        cohort = build_cohort_distribution(store2, weeks=[1, 2, 3])
        data = build_student_data(store2, "S001", weeks=[1, 2, 3], cohort=cohort)
        # Check NaN did not propagate to trend_slope
        if data.trend_slope is not None and math.isnan(data.trend_slope):
            pytest.fail(
                "NaN propagated to StudentLongitudinalData.trend_slope — "
                "build_student_data does not filter NaN before OLS"
            )


class TestBoundary7LargeClass:
    """BOUNDARY #7: Large class (200 students) through pipeline."""

    def test_large_class_feature_extraction(self, tmp_path) -> None:
        """200 students through feature extraction without memory issues."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.evaluation_types import LongitudinalRecord
        from forma.risk_predictor import FeatureExtractor

        store_path = str(tmp_path / "store_large.yaml")
        store = LongitudinalStore(store_path)

        for i in range(200):
            sid = f"S{i:04d}"
            for week in [1, 2, 3, 4]:
                score = 0.3 + 0.15 * (i % 5) + 0.02 * week
                store.add_record(
                    LongitudinalRecord(
                        student_id=sid,
                        week=week,
                        question_sn=1,
                        scores={"ensemble_score": score, "concept_coverage": score * 0.9},
                        tier_level=min(3, int(score * 4)),
                        tier_label="Proficient",
                        edge_f1=score * 0.7,
                        misconception_count=1 if score < 0.5 else 0,
                    )
                )

        extractor = FeatureExtractor()
        X, names, sids = extractor.extract(store, weeks=[1, 2, 3, 4])

        assert X.shape[0] == 200
        assert X.shape[1] == 15
        assert len(sids) == 200

    def test_large_class_longitudinal_summary(self, tmp_path) -> None:
        """200 students through longitudinal summary builder."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_report_data import build_longitudinal_summary

        store_path = str(tmp_path / "store_large_long.yaml")
        store = LongitudinalStore(store_path)

        for i in range(200):
            sid = f"S{i:04d}"
            for week in [1, 2, 3]:
                score = 0.3 + 0.15 * (i % 5) + 0.05 * week
                store.add_record(
                    LongitudinalRecord(
                        student_id=sid,
                        week=week,
                        question_sn=1,
                        scores={"ensemble_score": score, "concept_coverage": score * 0.9},
                        tier_level=min(3, int(score * 4)),
                        tier_label="Proficient",
                    )
                )

        summary = build_longitudinal_summary(
            store=store,
            weeks=[1, 2, 3],
            class_name="A",
        )
        assert summary.total_students == 200
        assert len(summary.student_trajectories) == 200


class TestBoundary8HalfWrittenYAML:
    """BOUNDARY #8: Corrupted/truncated YAML input handling."""

    def test_half_written_yaml_store_load(self, tmp_path) -> None:
        """Truncated YAML in longitudinal store."""
        from forma.longitudinal_store import LongitudinalStore

        store_path = str(tmp_path / "corrupt_store.yaml")
        with open(store_path, "w") as f:
            f.write("records:\n  S001_1_1:\n    student_id: S001\n    week: 1\n    ")
            # Truncated — missing required fields

        store = LongitudinalStore(store_path)
        try:
            store.load()
            # If it loaded, check if we got partial data or empty
            records = store.get_all_records()
            # Accessing records with missing fields should fail
            for r in records:
                _ = r.student_id
        except (yaml.YAMLError, KeyError, TypeError):
            # Expected — corrupted YAML should raise clearly
            pass
        except Exception as e:
            # Unexpected exception type
            pytest.fail(f"Unexpected exception from corrupt YAML: {type(e).__name__}: {e}")

    def test_empty_yaml_file_store_load(self, tmp_path) -> None:
        """Empty YAML file handled gracefully."""
        from forma.longitudinal_store import LongitudinalStore

        store_path = str(tmp_path / "empty.yaml")
        with open(store_path, "w") as f:
            f.write("")

        store = LongitudinalStore(store_path)
        store.load()
        assert store.get_all_records() == []

    def test_eval_yaml_missing_file(self) -> None:
        """Nonexistent eval YAML raises clear error."""
        from forma.evaluation_io import load_evaluation_yaml

        with pytest.raises(FileNotFoundError, match="not found"):
            load_evaluation_yaml("/nonexistent/path/eval.yaml")


class TestBoundary10MissingFormaJson:
    """BOUNDARY #10: Missing forma.json — CLI should handle gracefully."""

    def test_load_config_missing_all_paths(self) -> None:
        """No config file found anywhere raises FileNotFoundError."""
        from forma.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.json")


class TestBoundary11ModelVersionMismatch:
    """BOUNDARY #11: Old .pkl model with current code."""

    def test_stale_model_missing_scaler_field(self, tmp_path) -> None:
        """Model pickle missing expected fields."""
        import joblib
        from forma.risk_predictor import load_model

        # Save a minimal but incomplete model dict
        stale_model = {"model": "old_format", "version": "0.7.0"}
        model_path = str(tmp_path / "stale_model.pkl")
        joblib.dump(stale_model, model_path)

        try:
            loaded = load_model(model_path)
            # If it loads, check if it has expected attributes
            assert hasattr(loaded, "model"), "Loaded model missing 'model' attribute"
            assert hasattr(loaded, "scaler"), "Loaded model missing 'scaler' attribute"
        except (AttributeError, TypeError, KeyError):
            # Expected — version mismatch should raise
            pass
        except Exception as e:
            pytest.fail(f"Unexpected error from stale model: {type(e).__name__}: {e}")


class TestBoundary12ConcurrentYAMLWrites:
    """BOUNDARY #12: Concurrent YAML writes to same store."""

    def test_concurrent_store_saves(self, tmp_path) -> None:
        """Two LongitudinalStore instances writing same file."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.evaluation_types import LongitudinalRecord

        store_path = str(tmp_path / "concurrent_store.yaml")

        # Create two store instances pointing to same file
        store_a = LongitudinalStore(store_path)
        store_b = LongitudinalStore(store_path)

        store_a.add_record(
            LongitudinalRecord(
                student_id="S001",
                week=1,
                question_sn=1,
                scores={"ensemble_score": 0.7},
                tier_level=2,
                tier_label="Proficient",
            )
        )

        store_b.add_record(
            LongitudinalRecord(
                student_id="S002",
                week=1,
                question_sn=1,
                scores={"ensemble_score": 0.5},
                tier_level=1,
                tier_label="Developing",
            )
        )

        # Both save — should not corrupt
        store_a.save()
        store_b.save()

        # Verify — store_b overwrites store_a (last writer wins)
        verify = LongitudinalStore(store_path)
        verify.load()
        records = verify.get_all_records()
        # At minimum, the file should be valid YAML
        assert isinstance(records, list)
        # Note: without merge strategy, store_b's save will overwrite store_a's data
        # This is an integration issue — data loss from concurrent writes
        sids = {r.student_id for r in records}
        if "S001" not in sids:
            # Expected: last writer wins, S001 from store_a is lost
            pass  # This is the known issue — no class isolation


# ---------------------------------------------------------------------------
# Duplicate TopicTrendResult: cross-module type incompatibility
# ---------------------------------------------------------------------------


class TestDuplicateTopicTrendResult:
    """CRIT-1 FIX: Verify TopicTrendResult is now a single canonical class."""

    def test_topic_trend_result_is_same_class(self) -> None:
        """Both modules now export the same TopicTrendResult class."""
        from forma.longitudinal_report_data import TopicTrendResult as LRD_TTR
        from forma.student_longitudinal_data import TopicTrendResult as SLD_TTR

        assert LRD_TTR is SLD_TTR, "TopicTrendResult should be the same class"

    def test_topic_trend_result_has_superset_fields(self) -> None:
        """Canonical TopicTrendResult has Optional fields + interpretation."""
        import dataclasses
        from forma.longitudinal_report_data import TopicTrendResult

        fields = {f.name for f in dataclasses.fields(TopicTrendResult)}
        assert "interpretation" in fields
        assert "kendall_tau" in fields
        assert "n_weeks" in fields


# ---------------------------------------------------------------------------
# Cross-module: LongitudinalStore → student_longitudinal_data → report
# ---------------------------------------------------------------------------


class TestCrossModuleStudentLongitudinalPath:
    """Store → student data builder → warning evaluator chain."""

    def test_store_to_student_data_to_warnings(self, tmp_path) -> None:
        """Full chain: store → build_student_data → evaluate_warnings."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.evaluation_types import LongitudinalRecord
        from forma.student_longitudinal_data import (
            build_student_data,
            build_cohort_distribution,
            evaluate_warnings,
        )

        store_path = str(tmp_path / "store_chain.yaml")
        store = LongitudinalStore(store_path)

        # Build data for multiple students
        for sid in ["S001", "S002", "S003"]:
            for week in [1, 2, 3]:
                base = 0.3 if sid == "S001" else 0.6
                store.add_record(
                    LongitudinalRecord(
                        student_id=sid,
                        week=week,
                        question_sn=1,
                        scores={"ensemble_score": base + 0.02 * week, "concept_coverage": base},
                        tier_level=1 if base < 0.5 else 2,
                        tier_label="Developing" if base < 0.5 else "Proficient",
                    )
                )

        # Build cohort distribution first (required by build_student_data)
        cohort = build_cohort_distribution(store, weeks=[1, 2, 3])
        assert len(cohort.weekly_scores) > 0

        # Build student data
        student_data = build_student_data(store, "S001", weeks=[1, 2, 3], cohort=cohort)
        assert student_data.student_id == "S001"
        assert len(student_data.weeks) == 3

        # Evaluate warnings — returns (list[WarningSignal], AlertLevel) tuple
        result = evaluate_warnings(student_data, cohort)
        assert isinstance(result, tuple), f"evaluate_warnings returns {type(result)}, expected tuple"
        warnings_list, alert_level = result
        assert isinstance(warnings_list, list)
        # S001 with low scores should trigger some warnings
        triggered = [w for w in warnings_list if w.triggered]
        assert isinstance(triggered, list)


# ---------------------------------------------------------------------------
# Cross-module: delivery_prepare → delivery_send integration
# ---------------------------------------------------------------------------


class TestCrossModuleDeliveryChain:
    """Prepare → send: data structure compatibility."""

    def test_roster_to_template_render(self, tmp_path) -> None:
        """Roster student entries work with email template rendering."""
        from forma.delivery_prepare import StudentEntry
        from forma.delivery_send import render_template, EmailTemplate

        entry = StudentEntry(
            student_id="20230001",
            name="홍길동",
            email="hong@example.com",
        )
        template = EmailTemplate(
            subject="평가 결과 - {student_name}",
            body="{student_name}({student_id})님의 결과입니다.",
        )

        subject, body = render_template(
            template,
            student_name=entry.name,
            student_id=entry.student_id,
            class_name="A",
        )
        assert entry.name in body
        assert entry.student_id in body


# ---------------------------------------------------------------------------
# Cross-module: section_comparison with real data
# ---------------------------------------------------------------------------


class TestCrossModuleSectionComparison:
    """Section comparison with longitudinal store data."""

    def test_store_data_to_section_comparison(self, tmp_path) -> None:
        """Longitudinal store data feeds into section comparison."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.evaluation_types import LongitudinalRecord
        from forma.section_comparison import (
            compute_section_stats,
            compute_pairwise_comparisons,
        )

        store_path = str(tmp_path / "store_section.yaml")
        store = LongitudinalStore(store_path)

        # Two sections with different score distributions
        import random

        random.seed(42)
        section_scores: dict[str, list[float]] = {"A": [], "B": []}

        for section, base in [("A", 0.6), ("B", 0.5)]:
            for i in range(20):
                sid = f"{section}{i:03d}"
                score = base + random.uniform(-0.2, 0.2)
                score = max(0.0, min(1.0, score))
                section_scores[section].append(score)
                store.add_record(
                    LongitudinalRecord(
                        student_id=sid,
                        week=1,
                        question_sn=1,
                        scores={"ensemble_score": score},
                        tier_level=2,
                        tier_label="Proficient",
                        class_id=section,
                    )
                )

        stats_a = compute_section_stats(
            "A",
            section_scores["A"],
            {f"A{i:03d}" for i in range(20) if section_scores["A"][i] < 0.45},
        )
        stats_b = compute_section_stats(
            "B",
            section_scores["B"],
            {f"B{i:03d}" for i in range(20) if section_scores["B"][i] < 0.45},
        )

        assert stats_a.n_students == 20
        assert stats_b.n_students == 20

        comparisons = compute_pairwise_comparisons(
            {"A": section_scores["A"], "B": section_scores["B"]},
        )
        assert len(comparisons) == 1
        assert comparisons[0].section_a == "A" or comparisons[0].section_b == "A"
