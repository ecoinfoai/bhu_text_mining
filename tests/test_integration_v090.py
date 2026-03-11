"""Integration tests for v0.9.0 cross-feature interactions.

T062: End-to-end integration tests covering:
  - ProjectConfiguration load/validate/merge roundtrip
  - FeatureExtractor from LongitudinalStore (20 students x 4 weeks)
  - RiskPredictor train/save/load/predict roundtrip
  - WarningCard generation from predictions + rule-based union
  - Cross-section pairwise comparisons with significance testing
  - Full pipeline: config -> store -> train -> predict -> warning -> comparison
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import pytest
import yaml

from forma.evaluation_types import LongitudinalRecord
from forma.longitudinal_store import LongitudinalStore
from forma.project_config import (
    load_project_config,
    merge_configs,
    validate_project_config,
)
from forma.risk_predictor import (
    FEATURE_NAMES,
    FeatureExtractor,
    RiskPredictor,
    load_model,
    save_model,
)
from forma.section_comparison import (
    SectionComparison,
    SectionStats,
    compute_pairwise_comparisons,
    compute_section_stats,
)
from forma.warning_report_data import (
    RiskType,
    WarningCard,
    build_warning_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_STUDENTS = 20
_STUDENTS = [f"S{i:03d}" for i in range(1, _N_STUDENTS + 1)]
_WEEKS = [1, 2, 3, 4]
_QUESTIONS = [1, 2]

_CONCEPTS = {
    1: {"항상성": 0.55, "삼투": 0.40, "확산": 0.35, "능동수송": 0.30},
    2: {"항상성": 0.60, "삼투": 0.48, "확산": 0.42, "능동수송": 0.36},
    3: {"항상성": 0.70, "삼투": 0.55, "확산": 0.50, "능동수송": 0.45},
    4: {"항상성": 0.80, "삼투": 0.65, "확산": 0.60, "능동수송": 0.55},
}


def _student_base_score(student_id: str) -> float:
    """Deterministic base score spread from ~0.15 to ~0.9."""
    idx = int(student_id[1:])
    return 0.10 + (idx / _N_STUDENTS) * 0.80


def _build_store(tmp_path, *, include_v2_fields: bool = True) -> LongitudinalStore:
    """Build a store with 20 students x 4 weeks x 2 questions = 160 records."""
    store_path = str(tmp_path / "store.yaml")
    store = LongitudinalStore(store_path)
    random.seed(42)

    for week in _WEEKS:
        for sid in _STUDENTS:
            base = _student_base_score(sid)
            week_bonus = (week - 1) * 0.03
            for qsn in _QUESTIONS:
                noise = random.uniform(-0.05, 0.05)
                score = max(0.0, min(1.0, base + week_bonus + noise))

                tier_level = (
                    3 if score >= 0.85
                    else 2 if score >= 0.65
                    else 1 if score >= 0.45
                    else 0
                )
                tier_label = (
                    "Advanced" if tier_level == 3
                    else "Proficient" if tier_level == 2
                    else "Developing" if tier_level == 1
                    else "Beginning"
                )

                kwargs = dict(
                    student_id=sid,
                    week=week,
                    question_sn=qsn,
                    scores={
                        "ensemble_score": round(score, 4),
                        "concept_coverage": round(max(0, score - 0.1), 4),
                    },
                    tier_level=tier_level,
                    tier_label=tier_label,
                    concept_scores=_CONCEPTS[week],
                )

                if include_v2_fields:
                    kwargs["edge_f1"] = round(max(0.0, score - 0.15), 4)
                    kwargs["misconception_count"] = max(0, 3 - tier_level)

                record = LongitudinalRecord(**kwargs)
                store.add_record(record)

    store.save()
    return store


def _write_forma_yaml(tmp_path, content: dict) -> str:
    """Write a forma.yaml and return its path."""
    path = tmp_path / "forma.yaml"
    path.write_text(yaml.dump(content, allow_unicode=True), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Test 1: config load/validate roundtrip
# ---------------------------------------------------------------------------


class TestConfigLoadValidateRoundtrip:
    """Write forma.yaml to tmpdir, load, validate, check fields."""

    def test_config_load_validate_roundtrip(self, tmp_path):
        config_dict = {
            "project": {
                "course_name": "인체구조와기능",
                "year": 2026,
                "semester": 1,
                "grade": 1,
            },
            "classes": {
                "identifiers": ["A", "B", "C"],
                "join_pattern": "joined_{class}.yaml",
                "eval_pattern": "eval_{class}/",
            },
            "paths": {
                "exam_config": "exams/Ch01.yaml",
                "join_dir": "results/w1",
                "output_dir": "output/",
                "longitudinal_store": "store.yaml",
            },
            "evaluation": {
                "provider": "gemini",
                "n_calls": 3,
                "skip_feedback": False,
            },
            "reports": {
                "dpi": 200,
                "aggregate": True,
            },
            "prediction": {
                "model_path": "models/risk.pkl",
            },
            "current_week": 4,
        }

        yaml_path = _write_forma_yaml(tmp_path, config_dict)

        # Load
        loaded = load_project_config(yaml_path)
        assert loaded == config_dict

        # Validate (should not raise)
        validate_project_config(loaded)

        # Verify specific fields roundtripped
        assert loaded["project"]["course_name"] == "인체구조와기능"
        assert loaded["project"]["year"] == 2026
        assert loaded["classes"]["identifiers"] == ["A", "B", "C"]
        assert loaded["current_week"] == 4
        assert loaded["prediction"]["model_path"] == "models/risk.pkl"

    def test_invalid_config_raises(self, tmp_path):
        """Invalid values should raise ValueError."""
        bad_config = {
            "project": {"year": 1999, "semester": 5},
            "current_week": -1,
        }
        yaml_path = _write_forma_yaml(tmp_path, bad_config)
        loaded = load_project_config(yaml_path)
        with pytest.raises(ValueError, match="Configuration validation errors"):
            validate_project_config(loaded)


# ---------------------------------------------------------------------------
# Test 2: config merge with CLI overrides
# ---------------------------------------------------------------------------


class TestConfigMergeCLIOverrides:
    """Merge project config with CLI overrides, verify precedence."""

    def test_config_merge_cli_overrides(self, tmp_path):
        project_config = {
            "project": {"course_name": "해부학", "year": 2026},
            "evaluation": {"provider": "gemini", "n_calls": 5},
            "reports": {"dpi": 300},
            "current_week": 3,
        }

        # CLI namespace with defaults + one explicit override
        cli_ns = argparse.Namespace(
            course_name="",
            year=0,
            provider="gemini",
            n_calls=3,
            dpi=150,
            current_week=1,
            output_dir="cli_output/",
        )

        # Only n_calls and output_dir were explicitly set on CLI
        explicit_keys = {"n_calls", "output_dir"}

        merged = merge_configs(cli_ns, project_config, {}, explicit_keys=explicit_keys)

        # CLI explicit wins over project config
        assert merged["n_calls"] == 3  # CLI explicit (not project's 5)
        assert merged["output_dir"] == "cli_output/"  # CLI explicit

        # Project config wins over CLI defaults
        assert merged["course_name"] == "해부학"  # project > CLI default ""
        assert merged["year"] == 2026  # project > CLI default 0
        assert merged["dpi"] == 300  # project > CLI default 150
        assert merged["current_week"] == 3  # project > CLI default 1

    def test_system_config_fallback(self, tmp_path):
        """System config provides values when neither CLI nor project has them."""
        cli_ns = argparse.Namespace(provider="gemini", custom_key=None)
        project_config = {}
        system_config = {"extra_setting": "from_system"}

        merged = merge_configs(cli_ns, project_config, system_config)

        assert merged["extra_setting"] == "from_system"


# ---------------------------------------------------------------------------
# Test 3: feature extraction from store
# ---------------------------------------------------------------------------


class TestFeatureExtractionFromStore:
    """Create store with 20 students x 4 weeks, extract 15 features."""

    def test_feature_extraction_from_store(self, tmp_path):
        store = _build_store(tmp_path)
        extractor = FeatureExtractor()

        matrix, feature_names, student_ids = extractor.extract(
            store, _WEEKS, class_name=None,
        )

        # Shape: 20 students x 15 features
        assert matrix.shape == (_N_STUDENTS, 15)

        # Feature names match canonical order
        assert feature_names == list(FEATURE_NAMES)
        assert len(feature_names) == 15

        # All 20 students extracted
        assert len(student_ids) == _N_STUDENTS
        assert set(student_ids) == set(_STUDENTS)

        # No NaN values
        assert not np.any(np.isnan(matrix))

        # score_mean should be in [0, 1] for all students
        score_means = matrix[:, 0]
        assert np.all(score_means >= 0.0)
        assert np.all(score_means <= 1.0)

        # last_score should also be in [0, 1]
        last_scores = matrix[:, 3]
        assert np.all(last_scores >= 0.0)
        assert np.all(last_scores <= 1.0)

        # absence_count should be 0 (all students present all weeks)
        absence_counts = matrix[:, 9]
        assert np.all(absence_counts == 0.0)

    def test_feature_variance_across_students(self, tmp_path):
        """Students with different base scores should produce varied features."""
        store = _build_store(tmp_path)
        extractor = FeatureExtractor()
        matrix, _, student_ids = extractor.extract(store, _WEEKS)

        # score_mean should vary across students
        score_means = matrix[:, 0]
        assert np.std(score_means) > 0.1, "Students should have varied scores"

        # Higher-indexed students should generally have higher means
        low_idx = student_ids.index("S001")
        high_idx = student_ids.index("S020")
        assert matrix[high_idx, 0] > matrix[low_idx, 0]


# ---------------------------------------------------------------------------
# Test 4: train/predict roundtrip with save/load
# ---------------------------------------------------------------------------


class TestTrainPredictRoundtrip:
    """Train model, save, load, predict, verify RiskPrediction fields."""

    def test_train_predict_roundtrip(self, tmp_path):
        store = _build_store(tmp_path)
        extractor = FeatureExtractor()

        matrix, feature_names, student_ids = extractor.extract(store, _WEEKS)

        # Create binary labels: students with score_mean < 0.45 are "drop"
        labels = (matrix[:, 0] < 0.45).astype(int)

        # Ensure we have both classes
        assert np.sum(labels == 0) >= 1
        assert np.sum(labels == 1) >= 1

        # Train
        predictor = RiskPredictor()
        trained = predictor.train(
            matrix, labels, feature_names,
            min_students=10, n_weeks=4, target_threshold=0.45,
        )

        assert trained.n_students == _N_STUDENTS
        assert trained.n_weeks == 4
        assert trained.feature_names == list(FEATURE_NAMES)
        assert trained.cv_score >= 0.0  # some score computed

        # Save and reload
        model_path = tmp_path / "risk_model.pkl"
        save_model(trained, model_path)
        assert model_path.exists()

        loaded = load_model(model_path)
        assert loaded.n_students == trained.n_students
        assert loaded.feature_names == trained.feature_names

        # Predict with loaded model
        predictions = predictor.predict(loaded, matrix, student_ids)

        assert len(predictions) == _N_STUDENTS
        for pred in predictions:
            assert pred.student_id in student_ids
            assert 0.0 <= pred.drop_probability <= 1.0
            assert pred.is_model_based is True
            assert pred.confidence == "high"
            assert pred.predicted_tier in (0, 1, 2, 3)
            assert len(pred.risk_factors) == 15  # one per feature

            # Risk factors should be sorted by importance descending
            importances = [f.importance for f in pred.risk_factors]
            assert importances == sorted(importances, reverse=True)

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading from non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "does_not_exist.pkl")


# ---------------------------------------------------------------------------
# Test 5: warning data from predictions
# ---------------------------------------------------------------------------


class TestWarningDataFromPredictions:
    """Create RiskPredictions + student data -> build_warning_data -> verify."""

    def test_warning_data_from_predictions(self, tmp_path):
        store = _build_store(tmp_path)
        extractor = FeatureExtractor()
        predictor = RiskPredictor()

        matrix, feature_names, student_ids = extractor.extract(store, _WEEKS)
        labels = (matrix[:, 0] < 0.45).astype(int)
        trained = predictor.train(matrix, labels, feature_names, min_students=10)
        predictions = predictor.predict(trained, matrix, student_ids)

        # Build rule-based at-risk set (low-scoring students)
        at_risk_students = {}
        for sid in student_ids:
            base = _student_base_score(sid)
            is_at_risk = base < 0.40
            at_risk_students[sid] = {
                "is_at_risk": is_at_risk,
                "reasons": ["low_score"] if is_at_risk else [],
            }

        # Concept scores for deficit detection
        concept_scores = {}
        for sid in student_ids:
            base = _student_base_score(sid)
            concept_scores[sid] = {
                "항상성": min(1.0, base + 0.2),
                "삼투": base,
                "확산": max(0.0, base - 0.1),
                "능동수송": max(0.0, base - 0.15),
                "세포분열": max(0.0, base - 0.2),
            }

        # Score trajectories for trend analysis
        score_trajectories = {}
        for sid in student_ids:
            base = _student_base_score(sid)
            score_trajectories[sid] = [
                base, base + 0.03, base + 0.06, base + 0.09,
            ]

        absence_ratios = {sid: 0.0 for sid in student_ids}

        cards = build_warning_data(
            at_risk_students=at_risk_students,
            risk_predictions=predictions,
            concept_scores=concept_scores,
            score_trajectories=score_trajectories,
            absence_ratios=absence_ratios,
        )

        # At least some cards should be generated
        assert len(cards) >= 1

        # All cards should be WarningCard instances
        for card in cards:
            assert isinstance(card, WarningCard)
            assert card.student_id in student_ids
            assert len(card.risk_types) >= 1
            assert len(card.detection_methods) >= 1
            assert len(card.interventions) >= 1
            assert card.risk_severity >= 0.0

        # Cards should be sorted by severity descending
        severities = [c.risk_severity for c in cards]
        assert severities == sorted(severities, reverse=True)


# ---------------------------------------------------------------------------
# Test 6: warning data union inclusion
# ---------------------------------------------------------------------------


class TestWarningDataUnionInclusion:
    """Rule-based at-risk + model-predicted -> union set."""

    def test_warning_data_union_inclusion(self, tmp_path):
        from forma.risk_predictor import RiskPrediction

        # Student A: only rule-based at-risk (model says low probability)
        # Student B: only model-predicted (drop_prob >= 0.5, not rule-based)
        # Student C: both rule-based and model-predicted
        # Student D: neither (should not appear)

        at_risk_students = {
            "A": {"is_at_risk": True, "reasons": ["low_score"]},
            "B": {"is_at_risk": False, "reasons": []},
            "C": {"is_at_risk": True, "reasons": ["low_score"]},
            "D": {"is_at_risk": False, "reasons": []},
        }

        predictions = [
            RiskPrediction(student_id="A", drop_probability=0.2),
            RiskPrediction(student_id="B", drop_probability=0.7),
            RiskPrediction(student_id="C", drop_probability=0.8),
            RiskPrediction(student_id="D", drop_probability=0.1),
        ]

        concept_scores = {
            sid: {"c1": 0.2, "c2": 0.1, "c3": 0.15}
            for sid in ["A", "B", "C", "D"]
        }
        score_trajectories = {
            "A": [0.3, 0.28, 0.25, 0.22],
            "B": [0.4, 0.35, 0.30, 0.25],
            "C": [0.2, 0.18, 0.15, 0.12],
            "D": [0.8, 0.82, 0.85, 0.88],
        }

        cards = build_warning_data(
            at_risk_students=at_risk_students,
            risk_predictions=predictions,
            concept_scores=concept_scores,
            score_trajectories=score_trajectories,
        )

        card_ids = {c.student_id for c in cards}

        # Union: A (rule), B (model), C (both) should be included
        assert "A" in card_ids, "Rule-based at-risk student A must be included"
        assert "B" in card_ids, "Model-predicted student B must be included"
        assert "C" in card_ids, "Both-flagged student C must be included"
        assert "D" not in card_ids, "Neither-flagged student D must be excluded"

        # Check detection methods for each
        card_map = {c.student_id: c for c in cards}

        assert "rule_based" in card_map["A"].detection_methods
        assert "model_predicted" in card_map["B"].detection_methods
        assert "rule_based" in card_map["C"].detection_methods
        assert "model_predicted" in card_map["C"].detection_methods


# ---------------------------------------------------------------------------
# Test 7: cross-section two classes
# ---------------------------------------------------------------------------


class TestCrossSectionTwoClasses:
    """Two sections with different means -> stats + comparisons."""

    def test_cross_section_two_classes(self):
        rng = np.random.default_rng(42)

        # Section A: higher scores (mean ~0.75)
        scores_a = list(rng.normal(0.75, 0.10, size=40).clip(0, 1))
        # Section B: lower scores (mean ~0.45)
        scores_b = list(rng.normal(0.45, 0.10, size=40).clip(0, 1))

        stats_a = compute_section_stats("A", scores_a, at_risk_ids=set())
        stats_b = compute_section_stats("B", scores_b, at_risk_ids={"s1", "s2", "s3"})

        assert isinstance(stats_a, SectionStats)
        assert isinstance(stats_b, SectionStats)
        assert stats_a.n_students == 40
        assert stats_b.n_students == 40
        assert stats_a.mean > stats_b.mean
        assert stats_b.n_at_risk == 3
        assert stats_b.pct_at_risk == 3 / 40

        # Pairwise comparison
        comparisons = compute_pairwise_comparisons({
            "A": scores_a,
            "B": scores_b,
        })

        assert len(comparisons) == 1
        cmp = comparisons[0]
        assert isinstance(cmp, SectionComparison)
        assert cmp.section_a == "A"
        assert cmp.section_b == "B"
        assert cmp.n_a == 40
        assert cmp.n_b == 40

        # With N>=30, should use Welch's t-test
        assert cmp.test_name == "welch_t"

        # Large mean difference should be significant
        assert cmp.is_significant is True, (
            f"p={cmp.p_value}, expected significant difference"
        )

        # Effect size should be large (d > 0.8) given ~0.3 mean difference
        assert cmp.effect_size_label in ("medium", "large"), (
            f"d={cmp.cohens_d}, label={cmp.effect_size_label}"
        )

        # No Bonferroni correction for 2 sections
        assert cmp.p_value_corrected is None

    def test_three_sections_bonferroni(self):
        """Three sections trigger Bonferroni correction on p-values."""
        rng = np.random.default_rng(99)

        section_scores = {
            "A": list(rng.normal(0.80, 0.08, size=35).clip(0, 1)),
            "B": list(rng.normal(0.60, 0.10, size=35).clip(0, 1)),
            "C": list(rng.normal(0.40, 0.12, size=35).clip(0, 1)),
        }

        comparisons = compute_pairwise_comparisons(section_scores)

        # C(3,2) = 3 pairwise comparisons
        assert len(comparisons) == 3

        for cmp in comparisons:
            # All should have Bonferroni correction
            assert cmp.p_value_corrected is not None
            # Corrected p should be >= raw p
            assert cmp.p_value_corrected >= cmp.p_value

    def test_small_sections_mann_whitney(self):
        """Sections with N<30 use Mann-Whitney U test."""
        rng = np.random.default_rng(7)

        section_scores = {
            "X": list(rng.normal(0.70, 0.10, size=15).clip(0, 1)),
            "Y": list(rng.normal(0.40, 0.10, size=15).clip(0, 1)),
        }

        comparisons = compute_pairwise_comparisons(section_scores)

        assert len(comparisons) == 1
        assert comparisons[0].test_name == "mann_whitney_u"


# ---------------------------------------------------------------------------
# Test 8: full pipeline end-to-end
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Config -> store -> train -> predict -> warning -> comparison."""

    def test_full_pipeline(self, tmp_path):
        # ---- Step 1: Configuration ----
        config_dict = {
            "project": {
                "course_name": "인체구조와기능",
                "year": 2026,
                "semester": 1,
                "grade": 1,
            },
            "classes": {
                "identifiers": ["A", "B"],
                "join_pattern": "joined_{class}.yaml",
                "eval_pattern": "eval_{class}/",
            },
            "paths": {
                "output_dir": str(tmp_path / "output"),
                "longitudinal_store": str(tmp_path / "store.yaml"),
            },
            "prediction": {
                "model_path": str(tmp_path / "model.pkl"),
            },
            "current_week": 4,
        }
        yaml_path = _write_forma_yaml(tmp_path, config_dict)
        loaded_config = load_project_config(yaml_path)
        validate_project_config(loaded_config)

        # ---- Step 2: Build longitudinal store ----
        store = _build_store(tmp_path)

        # Verify record count
        records = store.get_all_records()
        assert len(records) == _N_STUDENTS * len(_WEEKS) * len(_QUESTIONS)

        # ---- Step 3: Feature extraction ----
        extractor = FeatureExtractor()
        matrix, feature_names, student_ids = extractor.extract(store, _WEEKS)
        assert matrix.shape == (_N_STUDENTS, 15)

        # ---- Step 4: Train and save model ----
        labels = (matrix[:, 0] < 0.45).astype(int)
        predictor = RiskPredictor()
        trained = predictor.train(
            matrix, labels, feature_names,
            min_students=10, n_weeks=4,
        )

        model_path = tmp_path / "model.pkl"
        save_model(trained, model_path)

        # ---- Step 5: Load model and predict ----
        loaded_model = load_model(model_path)
        predictions = predictor.predict(loaded_model, matrix, student_ids)
        assert len(predictions) == _N_STUDENTS

        # ---- Step 6: Build warning data ----
        at_risk_students = {}
        concept_scores = {}
        score_trajectories = {}
        absence_ratios = {}

        for sid in student_ids:
            base = _student_base_score(sid)
            at_risk_students[sid] = {
                "is_at_risk": base < 0.40,
                "reasons": ["low_score"] if base < 0.40 else [],
            }
            concept_scores[sid] = {
                "항상성": min(1.0, base + 0.2),
                "삼투": base,
                "확산": max(0.0, base - 0.1),
                "능동수송": max(0.0, base - 0.15),
                "세포분열": max(0.0, base - 0.2),
            }
            score_trajectories[sid] = [
                base, base + 0.03, base + 0.06, base + 0.09,
            ]
            absence_ratios[sid] = 0.0

        cards = build_warning_data(
            at_risk_students=at_risk_students,
            risk_predictions=predictions,
            concept_scores=concept_scores,
            score_trajectories=score_trajectories,
            absence_ratios=absence_ratios,
        )

        # At least the low-scoring students should produce cards
        assert len(cards) >= 1
        for card in cards:
            assert isinstance(card, WarningCard)
            assert len(card.risk_types) >= 1
            assert all(isinstance(rt, RiskType) for rt in card.risk_types)
            assert len(card.interventions) >= 1

        # ---- Step 7: Cross-section comparison ----
        # Split students into two sections: first 10 and last 10
        section_a_ids = student_ids[:10]
        section_b_ids = student_ids[10:]

        # Get scores from features (score_mean = column 0)
        scores_a = [float(matrix[student_ids.index(sid), 0]) for sid in section_a_ids]
        scores_b = [float(matrix[student_ids.index(sid), 0]) for sid in section_b_ids]

        stats_a = compute_section_stats("A", scores_a, at_risk_ids=set())
        stats_b = compute_section_stats("B", scores_b, at_risk_ids=set())

        assert stats_a.n_students == 10
        assert stats_b.n_students == 10
        # Section B (higher-index students) should have higher mean
        assert stats_b.mean > stats_a.mean

        comparisons = compute_pairwise_comparisons({
            "A": scores_a,
            "B": scores_b,
        })
        assert len(comparisons) == 1
        cmp = comparisons[0]
        assert isinstance(cmp, SectionComparison)
        # With N<30 per section, Mann-Whitney should be used
        assert cmp.test_name == "mann_whitney_u"

        # ---- Verify pipeline integrity ----
        # All IDs in warning cards should be valid student IDs
        for card in cards:
            assert card.student_id in student_ids

        # Highest severity card should be among low-scoring students
        if cards:
            worst = cards[0]
            assert _student_base_score(worst.student_id) < 0.5

    def test_full_pipeline_cold_start(self, tmp_path):
        """Pipeline works in cold-start mode (no pre-trained model)."""
        store = _build_store(tmp_path)
        extractor = FeatureExtractor()
        predictor = RiskPredictor()

        matrix, feature_names, student_ids = extractor.extract(store, _WEEKS)

        # Cold start: no trained model available
        predictions = predictor.predict_cold_start(
            matrix, student_ids, feature_names,
        )

        assert len(predictions) == _N_STUDENTS
        for pred in predictions:
            assert pred.is_model_based is False
            assert pred.confidence == "limited"
            assert 0.0 <= pred.drop_probability <= 1.0

        # Can still build warning data from cold-start predictions
        at_risk = {sid: {"is_at_risk": False, "reasons": []} for sid in student_ids}
        concept_scores = {sid: {} for sid in student_ids}

        cards = build_warning_data(
            at_risk_students=at_risk,
            risk_predictions=predictions,
            concept_scores=concept_scores,
        )

        # Cards from model-predicted students with drop_prob >= 0.5
        for card in cards:
            assert "model_predicted" in card.detection_methods

    def test_cross_feature_overlap(self, tmp_path):
        """Cross-check: warning card students overlap with high drop_probability predictions."""
        store = _build_store(tmp_path)
        extractor = FeatureExtractor()
        predictor = RiskPredictor()

        matrix, feature_names, student_ids = extractor.extract(store, _WEEKS)
        labels = (matrix[:, 0] < 0.45).astype(int)
        trained = predictor.train(matrix, labels, feature_names, min_students=10)
        predictions = predictor.predict(trained, matrix, student_ids)

        # Rule-based at-risk: students with low base score
        at_risk_students = {}
        concept_scores = {}
        score_trajectories = {}
        for sid in student_ids:
            base = _student_base_score(sid)
            at_risk_students[sid] = {
                "is_at_risk": base < 0.40,
                "reasons": ["low_score"] if base < 0.40 else [],
            }
            concept_scores[sid] = {
                "항상성": min(1.0, base + 0.2),
                "삼투": base,
                "확산": max(0.0, base - 0.1),
                "능동수송": max(0.0, base - 0.15),
                "세포분열": max(0.0, base - 0.2),
            }
            score_trajectories[sid] = [
                base, base + 0.03, base + 0.06, base + 0.09,
            ]

        cards = build_warning_data(
            at_risk_students=at_risk_students,
            risk_predictions=predictions,
            concept_scores=concept_scores,
            score_trajectories=score_trajectories,
            absence_ratios={sid: 0.0 for sid in student_ids},
        )

        card_ids = {c.student_id for c in cards}

        # High drop probability students (>= 0.5) must appear in warning cards
        high_prob_ids = {
            p.student_id for p in predictions if p.drop_probability >= 0.5
        }
        assert high_prob_ids.issubset(card_ids), (
            f"High-prob students missing from cards: {high_prob_ids - card_ids}"
        )

        # Rule-based at-risk students must also appear in warning cards
        rule_at_risk_ids = {
            sid for sid, info in at_risk_students.items()
            if info["is_at_risk"]
        }
        assert rule_at_risk_ids.issubset(card_ids), (
            f"Rule-based at-risk missing from cards: {rule_at_risk_ids - card_ids}"
        )

        # Cards should be exactly the union (no extras)
        expected_union = rule_at_risk_ids | high_prob_ids
        assert card_ids == expected_union, (
            f"Card IDs mismatch: extra={card_ids - expected_union}, "
            f"missing={expected_union - card_ids}"
        )

        # Every card with drop_probability should have model_predicted method
        for card in cards:
            if card.drop_probability is not None and card.drop_probability >= 0.5:
                assert "model_predicted" in card.detection_methods

        # Every card from rule-based detection should have rule_based method
        for card in cards:
            if card.student_id in rule_at_risk_ids:
                assert "rule_based" in card.detection_methods

        # Severity ordering: cards sorted descending by risk_severity
        severities = [c.risk_severity for c in cards]
        assert severities == sorted(severities, reverse=True)

        # Cross-section comparison with warning-flagged students
        # Section A = at-risk students, Section B = non-at-risk (from card set)
        at_risk_scores = [
            float(matrix[student_ids.index(sid), 0])
            for sid in card_ids if sid in student_ids
        ]
        safe_ids = set(student_ids) - card_ids
        safe_scores = [
            float(matrix[student_ids.index(sid), 0])
            for sid in safe_ids
        ]

        if at_risk_scores and safe_scores:
            stats_risk = compute_section_stats(
                "at_risk", at_risk_scores, card_ids,
            )
            stats_safe = compute_section_stats(
                "safe", safe_scores, set(),
            )
            # Safe students should have higher mean score
            assert stats_safe.mean > stats_risk.mean, (
                f"safe={stats_safe.mean:.3f} should be > at_risk={stats_risk.mean:.3f}"
            )

            comparisons = compute_pairwise_comparisons({
                "at_risk": at_risk_scores, "safe": safe_scores,
            })
            assert len(comparisons) == 1
            assert comparisons[0].cohens_d != 0.0  # groups should differ
