"""Tests for grade_predictor module — TDD RED + GREEN.

Tests GradeMapping loading, validation (A/B/C/D/F only), semester grouping,
student_id mismatch warnings, GradeFeatureExtractor, GradePredictor, and
model persistence.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from forma.grade_predictor import (
    GRADE_FEATURE_NAMES,
    GRADE_ORDINAL_MAP,
    VALID_GRADES,
    GradeFeatureExtractor,
    GradePredictor,
    TrainedGradeModel,
    load_grade_mapping,
    load_grade_model,
    save_grade_model,
)


# ---------------------------------------------------------------------------
# VALID_GRADES constant
# ---------------------------------------------------------------------------


class TestValidGrades:
    """Tests for VALID_GRADES constant."""

    def test_five_valid_grades(self):
        """Exactly 5 valid grades: A, B, C, D, F."""
        assert VALID_GRADES == {"A", "B", "C", "D", "F"}

    def test_no_plus_minus_grades(self):
        """Plus/minus grades are NOT valid."""
        assert "B+" not in VALID_GRADES
        assert "A-" not in VALID_GRADES
        assert "C+" not in VALID_GRADES


# ---------------------------------------------------------------------------
# load_grade_mapping tests
# ---------------------------------------------------------------------------


class TestLoadGradeMapping:
    """Tests for load_grade_mapping function."""

    def test_load_from_yaml_file(self, tmp_path, sample_grade_mapping):
        """Loads grade mapping from YAML file."""
        path = str(tmp_path / "grades.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(sample_grade_mapping, f, allow_unicode=True)

        result = load_grade_mapping(path)
        assert "2024-1학기" in result
        assert "2024-2학기" in result

    def test_semester_structure(self, tmp_path, sample_grade_mapping):
        """Loaded mapping has {semester: {student_id: grade}} structure."""
        path = str(tmp_path / "grades.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(sample_grade_mapping, f, allow_unicode=True)

        result = load_grade_mapping(path)
        sem1 = result["2024-1학기"]
        assert sem1["s001"] == "A"
        assert sem1["s002"] == "B"
        assert sem1["s003"] == "D"
        assert sem1["s004"] == "F"

    def test_file_not_found_raises(self):
        """FileNotFoundError when path doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_grade_mapping("/nonexistent/grades.yaml")

    def test_invalid_grade_raises(self, tmp_path):
        """Invalid grade (e.g., B+) raises ValueError."""
        data = {
            "2024-1학기": {
                "s001": "A",
                "s002": "B+",  # invalid
            },
        }
        path = str(tmp_path / "grades.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        with pytest.raises(ValueError, match="B\\+"):
            load_grade_mapping(path)

    def test_invalid_grade_numeric_raises(self, tmp_path):
        """Numeric grades are rejected."""
        data = {
            "2024-1학기": {
                "s001": 90,
            },
        }
        path = str(tmp_path / "grades.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        with pytest.raises(ValueError):
            load_grade_mapping(path)

    def test_multiple_semesters(self, tmp_path, sample_grade_mapping):
        """Multiple semesters are loaded correctly."""
        path = str(tmp_path / "grades.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(sample_grade_mapping, f, allow_unicode=True)

        result = load_grade_mapping(path)
        assert len(result) == 2

    def test_empty_semester_ok(self, tmp_path):
        """Empty semester dict is allowed."""
        data = {
            "2024-1학기": {},
        }
        path = str(tmp_path / "grades.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        result = load_grade_mapping(path)
        assert result["2024-1학기"] == {}


# ---------------------------------------------------------------------------
# Student ID mismatch warning tests
# ---------------------------------------------------------------------------


class TestStudentIdMismatchWarning:
    """Tests for student_id mismatch warnings."""

    def test_warn_on_student_mismatch(self, tmp_path, caplog):
        """Logs warning when grade student_ids don't match store student_ids."""
        data = {
            "2024-1학기": {
                "s001": "A",
                "s999": "F",  # not in store
            },
        }
        path = str(tmp_path / "grades.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        store_student_ids = {"s001", "s002"}
        with caplog.at_level(logging.WARNING):
            _result = load_grade_mapping(path, store_student_ids=store_student_ids)
        assert any("s999" in msg for msg in caplog.messages)

    def test_no_warn_when_all_match(self, tmp_path, caplog):
        """No warning when all student_ids match."""
        data = {
            "2024-1학기": {
                "s001": "A",
                "s002": "B",
            },
        }
        path = str(tmp_path / "grades.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        store_student_ids = {"s001", "s002", "s003"}
        with caplog.at_level(logging.WARNING):
            _result = load_grade_mapping(path, store_student_ids=store_student_ids)
        assert not any("mismatch" in msg.lower() for msg in caplog.messages)

    def test_no_warn_when_store_ids_none(self, tmp_path, caplog):
        """No warning when store_student_ids is None."""
        data = {
            "2024-1학기": {
                "s001": "A",
            },
        }
        path = str(tmp_path / "grades.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        with caplog.at_level(logging.WARNING):
            _result = load_grade_mapping(path, store_student_ids=None)
        assert len(caplog.records) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestGradePredictorEdgeCases:
    """Edge case tests."""

    def test_case_sensitive_grades(self, tmp_path):
        """Lowercase grades are rejected (must be uppercase)."""
        data = {
            "2024-1학기": {"s001": "a"},
        }
        path = str(tmp_path / "grades.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        with pytest.raises(ValueError):
            load_grade_mapping(path)

    def test_korean_semester_labels(self, tmp_path):
        """Korean semester labels are preserved correctly."""
        data = {
            "2024-1학기": {"s001": "A"},
            "2024-2학기": {"s002": "B"},
        }
        path = str(tmp_path / "grades.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        result = load_grade_mapping(path)
        assert "2024-1학기" in result
        assert "2024-2학기" in result

    def test_single_student_single_semester(self, tmp_path):
        """Minimal valid mapping: 1 semester, 1 student."""
        data = {"S1": {"s001": "F"}}
        path = str(tmp_path / "grades.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        result = load_grade_mapping(path)
        assert result["S1"]["s001"] == "F"


# ===========================================================================
# Phase 7: Grade prediction model (T050-T059)
# ===========================================================================


# ---------------------------------------------------------------------------
# T050: GRADE_ORDINAL_MAP and constants
# ---------------------------------------------------------------------------


class TestGradeOrdinalMap:
    """Tests for GRADE_ORDINAL_MAP constant."""

    def test_ordinal_encoding(self):
        """Grades map to ordinals: A=4, B=3, C=2, D=1, F=0."""
        assert GRADE_ORDINAL_MAP == {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}

    def test_all_valid_grades_have_mapping(self):
        """Every VALID_GRADES member has an ordinal mapping."""
        for g in VALID_GRADES:
            assert g in GRADE_ORDINAL_MAP


class TestGradeFeatureNames:
    """Tests for GRADE_FEATURE_NAMES."""

    def test_feature_count(self):
        """21 features: 15 base + 6 grade-specific."""
        assert len(GRADE_FEATURE_NAMES) == 21

    def test_base_features_included(self):
        """All 15 base features from risk_predictor are included."""
        from forma.risk_predictor import FEATURE_NAMES as BASE_FEATURES

        for name in BASE_FEATURES:
            assert name in GRADE_FEATURE_NAMES

    def test_grade_specific_features_present(self):
        """6 grade-specific features are present."""
        grade_specific = [
            "prior_grade_ordinal",
            "grade_trend",
            "best_prior_grade",
            "worst_prior_grade",
            "n_prior_semesters",
            "prior_grade_variance",
        ]
        for name in grade_specific:
            assert name in GRADE_FEATURE_NAMES


# ---------------------------------------------------------------------------
# T050: GradeFeatureExtractor
# ---------------------------------------------------------------------------


def _make_mock_store(n_students=10, n_weeks=4):
    """Create a mock LongitudinalStore with deterministic data."""
    from forma.evaluation_types import LongitudinalRecord

    store = MagicMock()
    records = []
    student_ids = [f"S{i+1:03d}" for i in range(n_students)]

    for idx, sid in enumerate(student_ids):
        base = 0.3 + (idx / n_students) * 0.5
        for w in range(1, n_weeks + 1):
            score = max(0.0, min(1.0, base + (w - 2) * 0.05))
            records.append(LongitudinalRecord(
                student_id=sid,
                week=w,
                question_sn=1,
                scores={"ensemble_score": score, "concept_coverage": score * 0.9},
                tier_level=2 if score >= 0.45 else 0,
                tier_label="Proficient" if score >= 0.45 else "Beginning",
            ))

    store.get_all_records.return_value = records

    # Build class_weekly_matrix mock
    matrix = {}
    for idx, sid in enumerate(student_ids):
        base = 0.3 + (idx / n_students) * 0.5
        week_scores = {}
        for w in range(1, n_weeks + 1):
            week_scores[w] = max(0.0, min(1.0, base + (w - 2) * 0.05))
        matrix[sid] = week_scores
    store.get_class_weekly_matrix.return_value = matrix
    return store, student_ids


class TestGradeFeatureExtractor:
    """Tests for GradeFeatureExtractor (T050 / FR-026)."""

    def test_extract_returns_correct_shape(self):
        """Feature matrix shape is (n_students, 21)."""
        store, student_ids = _make_mock_store(n_students=10)
        grade_history = {"S001": [4], "S002": [3, 2]}  # prior grades as ordinals

        extractor = GradeFeatureExtractor()
        matrix, feat_names, sids = extractor.extract(
            store, weeks=[1, 2, 3, 4], grade_history=grade_history,
        )
        assert matrix.shape == (10, 21)
        assert len(feat_names) == 21
        assert len(sids) == 10

    def test_extract_feature_names_match(self):
        """Returned feature names match GRADE_FEATURE_NAMES."""
        store, _ = _make_mock_store(n_students=5)
        extractor = GradeFeatureExtractor()
        _, feat_names, _ = extractor.extract(store, weeks=[1, 2, 3, 4])
        assert feat_names == list(GRADE_FEATURE_NAMES)

    def test_prior_grade_ordinal_feature(self):
        """prior_grade_ordinal uses most recent prior semester grade."""
        store, student_ids = _make_mock_store(n_students=3)
        # S001 had grade A (4) last semester, S002 had C (2)
        grade_history = {"S001": [4], "S002": [2]}

        extractor = GradeFeatureExtractor()
        matrix, feat_names, sids = extractor.extract(
            store, weeks=[1, 2, 3, 4], grade_history=grade_history,
        )
        prior_idx = feat_names.index("prior_grade_ordinal")
        s001_idx = sids.index("S001")
        s002_idx = sids.index("S002")
        assert matrix[s001_idx, prior_idx] == 4.0
        assert matrix[s002_idx, prior_idx] == 2.0

    def test_no_grade_history_uses_default(self):
        """Students with no grade history get default values (0)."""
        store, _ = _make_mock_store(n_students=3)
        extractor = GradeFeatureExtractor()
        matrix, feat_names, sids = extractor.extract(
            store, weeks=[1, 2, 3, 4], grade_history={},
        )
        prior_idx = feat_names.index("prior_grade_ordinal")
        # All should be 0 (no prior grades)
        assert all(matrix[i, prior_idx] == 0.0 for i in range(len(sids)))

    def test_grade_trend_with_multiple_semesters(self):
        """grade_trend is OLS slope across prior semesters."""
        store, _ = _make_mock_store(n_students=3)
        # S001: grades [2, 3, 4] = improving trend (positive slope)
        grade_history = {"S001": [2, 3, 4]}
        extractor = GradeFeatureExtractor()
        matrix, feat_names, sids = extractor.extract(
            store, weeks=[1, 2, 3, 4], grade_history=grade_history,
        )
        trend_idx = feat_names.index("grade_trend")
        s001_idx = sids.index("S001")
        assert matrix[s001_idx, trend_idx] > 0  # positive trend

    def test_n_prior_semesters_feature(self):
        """n_prior_semesters counts the number of prior semester grades."""
        store, _ = _make_mock_store(n_students=3)
        grade_history = {"S001": [4, 3, 2], "S002": [3]}
        extractor = GradeFeatureExtractor()
        matrix, feat_names, sids = extractor.extract(
            store, weeks=[1, 2, 3, 4], grade_history=grade_history,
        )
        n_idx = feat_names.index("n_prior_semesters")
        s001_idx = sids.index("S001")
        s002_idx = sids.index("S002")
        assert matrix[s001_idx, n_idx] == 3.0
        assert matrix[s002_idx, n_idx] == 1.0


# ---------------------------------------------------------------------------
# T051: GradePredictor.train()
# ---------------------------------------------------------------------------


class TestGradePredictorTrain:
    """Tests for GradePredictor.train() (T051 / FR-027, FR-033, FR-034)."""

    def _make_training_data(self, n=20):
        """Deterministic training data: features + ordinal labels."""
        rng = np.random.RandomState(42)
        X = rng.rand(n, 21)
        # Label: based on feature[0] score_mean
        labels = np.array([
            4 if X[i, 0] > 0.8 else
            3 if X[i, 0] > 0.6 else
            2 if X[i, 0] > 0.4 else
            1 if X[i, 0] > 0.2 else 0
            for i in range(n)
        ])
        return X, labels

    def test_train_returns_model(self):
        """train() returns a TrainedGradeModel."""
        X, labels = self._make_training_data()
        predictor = GradePredictor()
        model = predictor.train(
            X, labels, list(GRADE_FEATURE_NAMES),
            n_weeks=4,
        )
        assert isinstance(model, TrainedGradeModel)

    def test_trained_model_has_metadata(self):
        """TrainedGradeModel has training metadata."""
        X, labels = self._make_training_data()
        predictor = GradePredictor()
        model = predictor.train(X, labels, list(GRADE_FEATURE_NAMES), n_weeks=4)
        assert model.n_students == 20
        assert model.n_weeks == 4
        assert model.feature_names == list(GRADE_FEATURE_NAMES)
        assert model.training_date  # non-empty

    def test_train_insufficient_students_raises(self):
        """train() raises ValueError with too few students."""
        X = np.random.rand(3, 21)
        labels = np.array([4, 3, 2])
        predictor = GradePredictor()
        with pytest.raises(ValueError, match="Insufficient"):
            predictor.train(X, labels, list(GRADE_FEATURE_NAMES), min_students=10)

    def test_train_single_class_does_not_crash(self):
        """Single class in labels does not crash (augments)."""
        X = np.random.RandomState(42).rand(15, 21)
        labels = np.full(15, 3)  # all grade B
        predictor = GradePredictor()
        model = predictor.train(X, labels, list(GRADE_FEATURE_NAMES))
        assert model is not None

    def test_cv_score_is_float(self):
        """Cross-validation score is a float."""
        X, labels = self._make_training_data()
        predictor = GradePredictor()
        model = predictor.train(X, labels, list(GRADE_FEATURE_NAMES))
        assert isinstance(model.cv_score, float)


# ---------------------------------------------------------------------------
# T051: GradePredictor.predict()
# ---------------------------------------------------------------------------


class TestGradePredictorPredict:
    """Tests for GradePredictor.predict() (FR-027)."""

    def _train_model(self):
        """Train a model for prediction tests."""
        rng = np.random.RandomState(42)
        X = rng.rand(30, 21)
        labels = np.array([
            4 if X[i, 0] > 0.8 else
            3 if X[i, 0] > 0.6 else
            2 if X[i, 0] > 0.4 else
            1 if X[i, 0] > 0.2 else 0
            for i in range(30)
        ])
        predictor = GradePredictor()
        model = predictor.train(X, labels, list(GRADE_FEATURE_NAMES))
        return predictor, model

    def test_predict_returns_list(self):
        """predict() returns a list of GradePrediction."""
        predictor, model = self._train_model()
        X_test = np.random.RandomState(99).rand(5, 21)
        sids = [f"S{i:03d}" for i in range(5)]
        results = predictor.predict(model, X_test, sids)
        assert len(results) == 5

    def test_prediction_has_grade_and_probability(self):
        """Each prediction has predicted_grade and grade_probabilities."""
        predictor, model = self._train_model()
        X_test = np.random.RandomState(99).rand(3, 21)
        sids = ["S001", "S002", "S003"]
        results = predictor.predict(model, X_test, sids)
        for r in results:
            assert r.student_id in sids
            assert r.predicted_grade in VALID_GRADES
            assert isinstance(r.grade_probabilities, dict)
            # Probabilities should sum to ~1.0
            total = sum(r.grade_probabilities.values())
            assert abs(total - 1.0) < 0.01

    def test_prediction_confidence(self):
        """Predictions have confidence field."""
        predictor, model = self._train_model()
        X_test = np.random.RandomState(99).rand(3, 21)
        sids = ["S001", "S002", "S003"]
        results = predictor.predict(model, X_test, sids)
        for r in results:
            assert r.confidence in ("high", "limited")
            assert r.is_model_based is True


# ---------------------------------------------------------------------------
# T052: Cold-start prediction
# ---------------------------------------------------------------------------


class TestGradePredictorColdStart:
    """Tests for predict_cold_start() (T052 / FR-028)."""

    def test_cold_start_returns_predictions(self):
        """predict_cold_start returns predictions without a trained model."""
        predictor = GradePredictor()
        X = np.random.RandomState(42).rand(5, 21)
        sids = [f"S{i:03d}" for i in range(5)]
        results = predictor.predict_cold_start(X, sids, list(GRADE_FEATURE_NAMES))
        assert len(results) == 5

    def test_cold_start_has_limited_confidence(self):
        """Cold-start predictions have confidence='limited'."""
        predictor = GradePredictor()
        X = np.random.RandomState(42).rand(3, 21)
        sids = ["S001", "S002", "S003"]
        results = predictor.predict_cold_start(X, sids, list(GRADE_FEATURE_NAMES))
        for r in results:
            assert r.confidence == "limited"
            assert r.is_model_based is False

    def test_cold_start_predicted_grade_is_valid(self):
        """Cold-start predicted_grade is one of VALID_GRADES."""
        predictor = GradePredictor()
        X = np.random.RandomState(42).rand(5, 21)
        sids = [f"S{i:03d}" for i in range(5)]
        results = predictor.predict_cold_start(X, sids, list(GRADE_FEATURE_NAMES))
        for r in results:
            assert r.predicted_grade in VALID_GRADES

    def test_cold_start_deterministic(self):
        """Same input produces same output (deterministic)."""
        predictor = GradePredictor()
        X = np.random.RandomState(42).rand(5, 21)
        sids = [f"S{i:03d}" for i in range(5)]
        r1 = predictor.predict_cold_start(X, sids, list(GRADE_FEATURE_NAMES))
        r2 = predictor.predict_cold_start(X, sids, list(GRADE_FEATURE_NAMES))
        for a, b in zip(r1, r2):
            assert a.predicted_grade == b.predicted_grade


# ---------------------------------------------------------------------------
# T057: Model persistence
# ---------------------------------------------------------------------------


class TestGradeModelPersistence:
    """Tests for save_grade_model / load_grade_model (T057)."""

    def _train_model(self):
        """Train a model for persistence tests."""
        rng = np.random.RandomState(42)
        X = rng.rand(20, 21)
        labels = np.array([
            4 if X[i, 0] > 0.8 else 3 if X[i, 0] > 0.6 else
            2 if X[i, 0] > 0.4 else 1 if X[i, 0] > 0.2 else 0
            for i in range(20)
        ])
        predictor = GradePredictor()
        return predictor.train(X, labels, list(GRADE_FEATURE_NAMES))

    def test_save_creates_file(self, tmp_path):
        """save_grade_model creates a .pkl file."""
        model = self._train_model()
        path = tmp_path / "grade_model.pkl"
        save_grade_model(model, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_load_roundtrip(self, tmp_path):
        """Model survives save/load roundtrip."""
        model = self._train_model()
        path = tmp_path / "grade_model.pkl"
        save_grade_model(model, path)

        loaded = load_grade_model(path)
        assert isinstance(loaded, TrainedGradeModel)
        assert loaded.feature_names == model.feature_names
        assert loaded.n_students == model.n_students
        assert loaded.n_weeks == model.n_weeks

    def test_load_nonexistent_raises(self):
        """load_grade_model raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_grade_model("/nonexistent/model.pkl")

    def test_predictions_same_after_reload(self, tmp_path):
        """Predictions from reloaded model match original."""
        model = self._train_model()
        path = tmp_path / "grade_model.pkl"
        save_grade_model(model, path)

        loaded = load_grade_model(path)
        X_test = np.random.RandomState(99).rand(5, 21)
        sids = [f"S{i:03d}" for i in range(5)]

        predictor = GradePredictor()
        r1 = predictor.predict(model, X_test, sids)
        r2 = predictor.predict(loaded, X_test, sids)
        for a, b in zip(r1, r2):
            assert a.predicted_grade == b.predicted_grade
