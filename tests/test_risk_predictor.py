"""Tests for risk_predictor.py — FeatureExtractor, RiskPredictor, TrainedRiskModel.

T019: FeatureExtractor tests (15 features, sparse data, missing v2 fields)
T020: RiskPredictor tests (train, predict, cold start, bounds)
T021: TrainedRiskModel persistence tests (joblib roundtrip)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: build a mock LongitudinalStore
# ---------------------------------------------------------------------------

def _build_mock_store(records_data: list[dict]) -> MagicMock:
    """Build a mock LongitudinalStore from a list of record dicts.

    Each dict should have: student_id, week, question_sn, scores, tier_level,
    tier_label, and optional v2 fields (edge_f1, misconception_count, concept_scores).
    """
    from forma.evaluation_types import LongitudinalRecord

    records = []
    for d in records_data:
        records.append(LongitudinalRecord(
            student_id=d["student_id"],
            week=d["week"],
            question_sn=d.get("question_sn", 1),
            scores=d.get("scores", {"ensemble_score": 0.5}),
            tier_level=d.get("tier_level", 2),
            tier_label=d.get("tier_label", "Proficient"),
            edge_f1=d.get("edge_f1"),
            misconception_count=d.get("misconception_count"),
            concept_scores=d.get("concept_scores"),
        ))

    store = MagicMock()
    store.get_all_records.return_value = records

    # get_student_trajectory: returns [(week, value)] for a student metric
    def _trajectory(student_id, metric):
        from collections import defaultdict
        week_values: dict[int, list[float]] = defaultdict(list)
        for r in records:
            if r.student_id != student_id:
                continue
            val = r.scores.get(metric)
            if val is not None:
                week_values[r.week].append(val)
        return [
            (wk, sum(vs) / len(vs))
            for wk, vs in sorted(week_values.items())
        ]

    store.get_student_trajectory.side_effect = _trajectory

    # get_class_weekly_matrix: returns {student_id: {week: value}}
    def _matrix(metric):
        from collections import defaultdict
        matrix: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in records:
            val = r.scores.get(metric)
            if val is not None:
                matrix[r.student_id][r.week].append(val)
        return {
            sid: {wk: sum(vs) / len(vs) for wk, vs in sorted(weeks.items())}
            for sid, weeks in matrix.items()
        }

    store.get_class_weekly_matrix.side_effect = _matrix

    return store


def _make_3week_data(n_students: int = 15) -> list[dict]:
    """Generate 3-week data for n_students with varying scores."""
    data = []
    for i in range(n_students):
        sid = f"S{i+1:03d}"
        base_score = 0.3 + (i / n_students) * 0.5  # 0.3 to 0.8
        for week in [1, 2, 3]:
            score = base_score + (week - 2) * 0.05  # slight trend
            score = max(0.0, min(1.0, score))
            data.append({
                "student_id": sid,
                "week": week,
                "question_sn": 1,
                "scores": {
                    "ensemble_score": score,
                    "concept_coverage": score * 0.9,
                },
                "tier_level": 2 if score >= 0.45 else 0,
                "tier_label": "Proficient" if score >= 0.45 else "Beginning",
                "edge_f1": score * 0.8,
                "misconception_count": max(0, int((1 - score) * 5)),
            })
    return data


# ---------------------------------------------------------------------------
# T019: FeatureExtractor tests
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def test_extract_basic(self):
        """Extract features from 3-week store produces correct shape."""
        from forma.risk_predictor import FeatureExtractor

        data = _make_3week_data(n_students=10)
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, feature_names, student_ids = extractor.extract(
            store, weeks=[1, 2, 3],
        )

        assert isinstance(matrix, np.ndarray)
        assert matrix.shape[0] == 10  # 10 students
        assert matrix.shape[1] == len(feature_names)
        assert len(student_ids) == 10
        assert len(feature_names) == 15

    def test_feature_names(self):
        """Feature names match the 15 expected features."""
        from forma.risk_predictor import FeatureExtractor

        data = _make_3week_data(n_students=5)
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        _, feature_names, _ = extractor.extract(store, weeks=[1, 2, 3])

        expected = [
            "score_mean", "score_variance", "score_slope", "last_score",
            "coverage_mean", "coverage_slope",
            "tier_low_ratio",
            "misconception_mean", "misconception_slope",
            "absence_count", "absence_ratio",
            "z_score_mean", "z_score_slope",
            "edge_f1_mean", "edge_f1_slope",
        ]
        assert feature_names == expected

    def test_single_week_slope_zero(self):
        """With single week of data, all slopes should be 0.0."""
        from forma.risk_predictor import FeatureExtractor

        data = [
            {
                "student_id": "S001", "week": 1, "question_sn": 1,
                "scores": {"ensemble_score": 0.6, "concept_coverage": 0.5},
                "tier_level": 2, "tier_label": "Proficient",
                "edge_f1": 0.7, "misconception_count": 1,
            },
        ]
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, names, _ = extractor.extract(store, weeks=[1])

        # Slopes should be 0.0
        slope_indices = [i for i, n in enumerate(names) if "slope" in n]
        for idx in slope_indices:
            assert matrix[0, idx] == 0.0, f"Slope for {names[idx]} should be 0.0"

    def test_missing_v2_fields_fallback(self):
        """Missing v2 fields (edge_f1, misconception_count) fall back to 0.0."""
        from forma.risk_predictor import FeatureExtractor

        data = [
            {
                "student_id": "S001", "week": w, "question_sn": 1,
                "scores": {"ensemble_score": 0.5, "concept_coverage": 0.4},
                "tier_level": 2, "tier_label": "Proficient",
                # No edge_f1, misconception_count
            }
            for w in [1, 2, 3]
        ]
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, names, _ = extractor.extract(store, weeks=[1, 2, 3])

        edge_f1_idx = names.index("edge_f1_mean")
        assert matrix[0, edge_f1_idx] == 0.0

        misconception_idx = names.index("misconception_mean")
        assert matrix[0, misconception_idx] == 0.0

    def test_sparse_data_absence(self):
        """Student missing a week is counted as absence."""
        from forma.risk_predictor import FeatureExtractor

        data = [
            # S001 has weeks 1, 2, 3
            {"student_id": "S001", "week": 1, "scores": {"ensemble_score": 0.5, "concept_coverage": 0.4},
             "tier_level": 2, "tier_label": "Proficient"},
            {"student_id": "S001", "week": 2, "scores": {"ensemble_score": 0.6, "concept_coverage": 0.5},
             "tier_level": 2, "tier_label": "Proficient"},
            {"student_id": "S001", "week": 3, "scores": {"ensemble_score": 0.7, "concept_coverage": 0.6},
             "tier_level": 2, "tier_label": "Proficient"},
            # S002 only has week 1 (missing 2 and 3)
            {"student_id": "S002", "week": 1, "scores": {"ensemble_score": 0.4, "concept_coverage": 0.3},
             "tier_level": 0, "tier_label": "Beginning"},
        ]
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, names, student_ids = extractor.extract(store, weeks=[1, 2, 3])

        s002_idx = student_ids.index("S002")
        absence_count_idx = names.index("absence_count")
        absence_ratio_idx = names.index("absence_ratio")

        assert matrix[s002_idx, absence_count_idx] == 2  # missing 2 weeks
        assert abs(matrix[s002_idx, absence_ratio_idx] - 2 / 3) < 1e-9

    def test_z_score_computation(self):
        """Z-scores are computed relative to class mean per week."""
        from forma.risk_predictor import FeatureExtractor

        data = [
            {"student_id": "S001", "week": 1, "scores": {"ensemble_score": 0.8, "concept_coverage": 0.7},
             "tier_level": 3, "tier_label": "Advanced"},
            {"student_id": "S002", "week": 1, "scores": {"ensemble_score": 0.2, "concept_coverage": 0.1},
             "tier_level": 0, "tier_label": "Beginning"},
        ]
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, names, student_ids = extractor.extract(store, weeks=[1])

        z_idx = names.index("z_score_mean")
        s001_z = matrix[student_ids.index("S001"), z_idx]
        s002_z = matrix[student_ids.index("S002"), z_idx]

        # S001 should have positive z-score, S002 negative
        assert s001_z > 0
        assert s002_z < 0


# ---------------------------------------------------------------------------
# T020: RiskPredictor tests
# ---------------------------------------------------------------------------

class TestRiskPredictor:
    """Tests for RiskPredictor."""

    def test_train_returns_model(self):
        """Training with sufficient data returns a TrainedRiskModel."""
        from forma.risk_predictor import FeatureExtractor, RiskPredictor

        data = _make_3week_data(n_students=20)
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, feature_names, student_ids = extractor.extract(
            store, weeks=[1, 2, 3],
        )

        # Label: last score < 0.45 → drop
        labels = (matrix[:, feature_names.index("last_score")] < 0.45).astype(int)

        predictor = RiskPredictor()
        model = predictor.train(matrix, labels, feature_names)

        assert model is not None
        assert model.feature_names == feature_names
        assert model.n_students == 20
        assert model.n_weeks == 3
        assert 0.0 <= model.cv_score <= 1.0
        assert model.target_threshold == 0.45

    def test_predict_returns_predictions(self):
        """Prediction returns list of RiskPrediction with valid bounds."""
        from forma.risk_predictor import FeatureExtractor, RiskPredictor

        data = _make_3week_data(n_students=20)
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, feature_names, student_ids = extractor.extract(
            store, weeks=[1, 2, 3],
        )
        labels = (matrix[:, feature_names.index("last_score")] < 0.45).astype(int)

        predictor = RiskPredictor()
        model = predictor.train(matrix, labels, feature_names)
        predictions = predictor.predict(model, matrix, student_ids)

        assert len(predictions) == 20
        for pred in predictions:
            assert 0.0 <= pred.drop_probability <= 1.0
            assert pred.is_model_based is True
            assert pred.confidence == "high"
            assert len(pred.risk_factors) > 0

    def test_risk_factors_sorted_by_importance(self):
        """Risk factors are sorted by importance descending."""
        from forma.risk_predictor import FeatureExtractor, RiskPredictor

        data = _make_3week_data(n_students=20)
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, feature_names, student_ids = extractor.extract(
            store, weeks=[1, 2, 3],
        )
        labels = (matrix[:, feature_names.index("last_score")] < 0.45).astype(int)

        predictor = RiskPredictor()
        model = predictor.train(matrix, labels, feature_names)
        predictions = predictor.predict(model, matrix, student_ids)

        for pred in predictions:
            importances = [f.importance for f in pred.risk_factors]
            assert importances == sorted(importances, reverse=True)

    def test_cold_start_prediction(self):
        """Cold start returns predictions with limited confidence."""
        from forma.risk_predictor import FeatureExtractor, RiskPredictor

        data = _make_3week_data(n_students=10)
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, feature_names, student_ids = extractor.extract(
            store, weeks=[1, 2, 3],
        )

        predictor = RiskPredictor()
        predictions = predictor.predict_cold_start(
            matrix, student_ids, feature_names,
        )

        assert len(predictions) == 10
        for pred in predictions:
            assert 0.0 <= pred.drop_probability <= 1.0
            assert pred.is_model_based is False
            assert pred.confidence == "limited"

    def test_insufficient_training_data(self):
        """Training with too few students raises ValueError."""
        from forma.risk_predictor import RiskPredictor

        matrix = np.random.rand(5, 15)  # Only 5 students
        labels = np.array([0, 0, 0, 1, 1])
        feature_names = [f"feat_{i}" for i in range(15)]

        predictor = RiskPredictor()
        with pytest.raises(ValueError, match="student"):
            predictor.train(
                matrix, labels, feature_names, min_students=10,
            )

    def test_drop_probability_bounds(self):
        """All drop probabilities are within [0.0, 1.0]."""
        from forma.risk_predictor import FeatureExtractor, RiskPredictor

        data = _make_3week_data(n_students=30)
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, feature_names, student_ids = extractor.extract(
            store, weeks=[1, 2, 3],
        )
        labels = (matrix[:, feature_names.index("last_score")] < 0.45).astype(int)

        predictor = RiskPredictor()
        model = predictor.train(matrix, labels, feature_names)
        predictions = predictor.predict(model, matrix, student_ids)

        for pred in predictions:
            assert 0.0 <= pred.drop_probability <= 1.0


# ---------------------------------------------------------------------------
# T021: TrainedRiskModel persistence tests
# ---------------------------------------------------------------------------

class TestModelPersistence:
    """Tests for save_model/load_model (joblib roundtrip)."""

    def test_save_load_roundtrip(self, tmp_path):
        """Saving and loading a model produces same predictions."""
        from forma.risk_predictor import (
            FeatureExtractor,
            RiskPredictor,
            save_model,
            load_model,
        )

        data = _make_3week_data(n_students=20)
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, feature_names, student_ids = extractor.extract(
            store, weeks=[1, 2, 3],
        )
        labels = (matrix[:, feature_names.index("last_score")] < 0.45).astype(int)

        predictor = RiskPredictor()
        model = predictor.train(matrix, labels, feature_names)

        # Save
        model_path = tmp_path / "model.pkl"
        save_model(model, model_path)
        assert model_path.exists()

        # Load
        loaded = load_model(model_path)
        assert loaded.feature_names == model.feature_names
        assert loaded.cv_score == model.cv_score
        assert loaded.n_students == model.n_students

        # Same predictions
        pred_original = predictor.predict(model, matrix, student_ids)
        pred_loaded = predictor.predict(loaded, matrix, student_ids)

        for orig, load in zip(pred_original, pred_loaded):
            assert abs(orig.drop_probability - load.drop_probability) < 1e-9

    def test_feature_names_preserved(self, tmp_path):
        """Feature names are preserved after save/load."""
        from forma.risk_predictor import (
            FeatureExtractor,
            RiskPredictor,
            save_model,
            load_model,
        )

        data = _make_3week_data(n_students=15)
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, feature_names, student_ids = extractor.extract(
            store, weeks=[1, 2, 3],
        )
        labels = (matrix[:, feature_names.index("last_score")] < 0.45).astype(int)

        predictor = RiskPredictor()
        model = predictor.train(matrix, labels, feature_names)

        model_path = tmp_path / "model.pkl"
        save_model(model, model_path)
        loaded = load_model(model_path)

        assert loaded.feature_names == feature_names

    def test_cv_score_preserved(self, tmp_path):
        """CV score is preserved after save/load."""
        from forma.risk_predictor import (
            FeatureExtractor,
            RiskPredictor,
            save_model,
            load_model,
        )

        data = _make_3week_data(n_students=20)
        store = _build_mock_store(data)

        extractor = FeatureExtractor()
        matrix, feature_names, student_ids = extractor.extract(
            store, weeks=[1, 2, 3],
        )
        labels = (matrix[:, feature_names.index("last_score")] < 0.45).astype(int)

        predictor = RiskPredictor()
        model = predictor.train(matrix, labels, feature_names)

        model_path = tmp_path / "model.pkl"
        save_model(model, model_path)
        loaded = load_model(model_path)

        assert loaded.cv_score == model.cv_score


# ---------------------------------------------------------------------------
# T008: NaN-injected data tests for FeatureExtractor
# ---------------------------------------------------------------------------


class TestFeatureExtractorNanSafety:
    """Tests that FeatureExtractor handles NaN values without crashing."""

    def test_nan_in_ensemble_score(self):
        """NaN in ensemble_score does not crash feature extraction."""
        from forma.risk_predictor import FeatureExtractor

        data = [
            {"student_id": "S001", "week": 1, "question_sn": 1,
             "scores": {"ensemble_score": float("nan"), "concept_coverage": 0.5},
             "tier_level": 2, "tier_label": "Proficient"},
            {"student_id": "S001", "week": 2, "question_sn": 1,
             "scores": {"ensemble_score": 0.6, "concept_coverage": 0.5},
             "tier_level": 2, "tier_label": "Proficient"},
            {"student_id": "S001", "week": 3, "question_sn": 1,
             "scores": {"ensemble_score": 0.7, "concept_coverage": 0.6},
             "tier_level": 3, "tier_label": "Advanced"},
        ]
        store = _build_mock_store(data)
        extractor = FeatureExtractor()
        matrix, names, sids = extractor.extract(store, weeks=[1, 2, 3])

        assert matrix.shape == (1, 15)
        # score_mean must not be NaN
        score_mean_idx = names.index("score_mean")
        assert not np.isnan(matrix[0, score_mean_idx]), "score_mean should be NaN-safe"
        # Week 1 NaN -> _safe_nanmean([nan])=0.0; weeks 2,3 produce 0.6, 0.7
        # score_mean = mean([0.0, 0.6, 0.7]) ≈ 0.433
        assert abs(matrix[0, score_mean_idx] - 0.4333) < 0.05, "score_mean should include 0.0 for NaN weeks"

    def test_nan_in_concept_coverage(self):
        """NaN in concept_coverage does not produce NaN in coverage_mean."""
        from forma.risk_predictor import FeatureExtractor

        data = [
            {"student_id": "S001", "week": 1, "question_sn": 1,
             "scores": {"ensemble_score": 0.5, "concept_coverage": float("nan")},
             "tier_level": 2, "tier_label": "Proficient"},
            {"student_id": "S001", "week": 2, "question_sn": 1,
             "scores": {"ensemble_score": 0.6, "concept_coverage": 0.5},
             "tier_level": 2, "tier_label": "Proficient"},
        ]
        store = _build_mock_store(data)
        extractor = FeatureExtractor()
        matrix, names, sids = extractor.extract(store, weeks=[1, 2])

        cov_idx = names.index("coverage_mean")
        assert not np.isnan(matrix[0, cov_idx]), "coverage_mean should be NaN-safe"
        # Week 1 NaN filtered, week 2 coverage = 0.5 -> mean should be close to 0.25..0.5
        assert matrix[0, cov_idx] > 0.0, "coverage_mean should reflect valid data"

    def test_nan_in_edge_f1(self):
        """NaN in edge_f1 does not produce NaN in edge_f1_mean."""
        from forma.risk_predictor import FeatureExtractor

        data = [
            {"student_id": "S001", "week": 1, "question_sn": 1,
             "scores": {"ensemble_score": 0.5, "concept_coverage": 0.4},
             "tier_level": 2, "tier_label": "Proficient",
             "edge_f1": float("nan")},
            {"student_id": "S001", "week": 2, "question_sn": 1,
             "scores": {"ensemble_score": 0.6, "concept_coverage": 0.5},
             "tier_level": 2, "tier_label": "Proficient",
             "edge_f1": 0.7},
        ]
        store = _build_mock_store(data)
        extractor = FeatureExtractor()
        matrix, names, sids = extractor.extract(store, weeks=[1, 2])

        f1_idx = names.index("edge_f1_mean")
        assert not np.isnan(matrix[0, f1_idx]), "edge_f1_mean should be NaN-safe"
        # Week 1 edge_f1 is NaN (passed as float('nan'), goes through float(r.edge_f1))
        # Week 2 edge_f1 = 0.7 -> mean of [nan_mean, 0.7] should be non-NaN
        assert matrix[0, f1_idx] > 0.0, "edge_f1_mean should reflect valid data"

    def test_all_nan_scores_produces_zero(self):
        """When all scores are NaN, features should be 0.0, not NaN."""
        from forma.risk_predictor import FeatureExtractor

        data = [
            {"student_id": "S001", "week": w, "question_sn": 1,
             "scores": {"ensemble_score": float("nan"), "concept_coverage": float("nan")},
             "tier_level": 2, "tier_label": "Proficient",
             "edge_f1": float("nan"), "misconception_count": None}
            for w in [1, 2, 3]
        ]
        store = _build_mock_store(data)
        extractor = FeatureExtractor()
        matrix, names, sids = extractor.extract(store, weeks=[1, 2, 3])

        # No NaN values in the feature matrix
        assert not np.any(np.isnan(matrix)), "Feature matrix should have no NaN values"

    def test_nan_in_class_mean_for_zscore(self):
        """NaN in class-wide scores does not crash z-score computation."""
        from forma.risk_predictor import FeatureExtractor

        data = [
            {"student_id": "S001", "week": 1, "question_sn": 1,
             "scores": {"ensemble_score": float("nan"), "concept_coverage": 0.5},
             "tier_level": 2, "tier_label": "Proficient"},
            {"student_id": "S002", "week": 1, "question_sn": 1,
             "scores": {"ensemble_score": 0.6, "concept_coverage": 0.5},
             "tier_level": 2, "tier_label": "Proficient"},
        ]
        store = _build_mock_store(data)
        extractor = FeatureExtractor()
        matrix, names, sids = extractor.extract(store, weeks=[1])

        z_idx = names.index("z_score_mean")
        # Should not crash and should not be NaN
        for i in range(len(sids)):
            assert not np.isnan(matrix[i, z_idx]), f"z_score_mean for {sids[i]} should not be NaN"
