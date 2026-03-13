"""Drop risk prediction using logistic regression on longitudinal features.

Provides feature extraction from longitudinal data, model training via
logistic regression with StratifiedKFold CV, and prediction with risk
factor explanations.

Dataclasses:
    RiskFactor: Individual feature contributing to risk.
    RiskPrediction: Per-student risk prediction result.
    TrainedRiskModel: Persisted model wrapper with metadata.

Classes:
    FeatureExtractor: Extracts 15-feature vectors from LongitudinalStore.
    RiskPredictor: Trains and predicts with LogisticRegression.

Functions:
    save_model: Persist TrainedRiskModel via joblib.
    load_model: Load TrainedRiskModel from joblib file.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# 15 feature names in order
FEATURE_NAMES = [
    "score_mean", "score_variance", "score_slope", "last_score",
    "coverage_mean", "coverage_slope",
    "tier_low_ratio",
    "misconception_mean", "misconception_slope",
    "absence_count", "absence_ratio",
    "z_score_mean", "z_score_slope",
    "edge_f1_mean", "edge_f1_slope",
]


@dataclass
class RiskFactor:
    """Individual feature contributing to a student's risk prediction.

    Attributes:
        name: Feature name from FEATURE_NAMES.
        importance: Absolute coefficient value.
        value: Student's feature value.
        direction: "increasing_risk" or "decreasing_risk".
    """

    name: str
    importance: float
    value: float
    direction: str


@dataclass
class RiskPrediction:
    """Per-student risk prediction result.

    Attributes:
        student_id: Student identifier.
        drop_probability: Predicted probability of dropping (0.0-1.0).
        risk_factors: Contributing factors sorted by importance descending.
        predicted_tier: Predicted final tier (0-3).
        is_model_based: True if model-based, False if rule-based fallback.
        confidence: "high" for model-based, "limited" for cold start.
    """

    student_id: str
    drop_probability: float
    risk_factors: list[RiskFactor] = field(default_factory=list)
    predicted_tier: int = 2
    is_model_based: bool = True
    confidence: str = "high"


@dataclass
class TrainedRiskModel:
    """Persisted prediction model wrapper.

    Attributes:
        model: Fitted LogisticRegression instance.
        feature_names: Ordered feature names matching model columns.
        scaler: Fitted StandardScaler for feature normalization.
        training_date: ISO 8601 date of training.
        n_students: Number of students used in training.
        n_weeks: Number of weeks of data used.
        cv_score: Mean cross-validation accuracy score.
        target_threshold: Score threshold defining "drop" (default 0.45).
    """

    model: LogisticRegression
    feature_names: list[str]
    scaler: StandardScaler
    training_date: str
    n_students: int
    n_weeks: int
    cv_score: float
    target_threshold: float = 0.45


def _ols_slope(values: list[float]) -> float:
    """Compute OLS slope from a sequence of values.

    Args:
        values: Time-ordered values.

    Returns:
        OLS slope, or 0.0 if fewer than 2 values.
    """
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    coeffs = np.polyfit(x, values, deg=1)
    return float(coeffs[0])


class FeatureExtractor:
    """Extracts 15-feature vectors from a LongitudinalStore.

    Features per student (see FEATURE_NAMES):
    1-4: score_mean, score_variance, score_slope, last_score
    5-6: coverage_mean, coverage_slope
    7: tier_low_ratio (fraction of weeks at tier 0 or 1)
    8-9: misconception_mean, misconception_slope
    10-11: absence_count, absence_ratio
    12-13: z_score_mean, z_score_slope
    14-15: edge_f1_mean, edge_f1_slope
    """

    def extract(
        self,
        store,
        weeks: list[int],
        class_name: str | None = None,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Extract feature matrix from store.

        Args:
            store: LongitudinalStore instance.
            weeks: List of week numbers to include.
            class_name: Optional class filter (not used currently).

        Returns:
            Tuple of (feature_matrix, feature_names, student_ids).
            feature_matrix shape: (n_students, 15).
        """
        all_records = store.get_all_records()
        week_set = set(weeks)
        n_weeks = len(weeks)

        # Group records by student
        student_records: dict[str, list] = defaultdict(list)
        for r in all_records:
            if r.week in week_set:
                student_records[r.student_id].append(r)

        # Get class-wide weekly matrix for z-score computation
        class_matrix = store.get_class_weekly_matrix("ensemble_score")

        # Compute per-week class mean and std
        week_stats: dict[int, tuple[float, float]] = {}
        for w in weeks:
            week_scores = [
                scores[w] for scores in class_matrix.values() if w in scores
            ]
            if week_scores:
                mean = np.mean(week_scores)
                std = np.std(week_scores, ddof=0)
                week_stats[w] = (float(mean), float(std))

        student_ids = sorted(student_records.keys())
        n_students = len(student_ids)
        matrix = np.zeros((n_students, 15), dtype=float)

        for idx, sid in enumerate(student_ids):
            records = student_records[sid]
            matrix[idx] = self._extract_student_features(
                records, weeks, n_weeks, class_matrix.get(sid, {}),
                week_stats,
            )

        return matrix, list(FEATURE_NAMES), student_ids

    def _extract_student_features(
        self,
        records: list,
        weeks: list[int],
        n_weeks: int,
        student_weekly_scores: dict[int, float],
        week_stats: dict[int, tuple[float, float]],
    ) -> np.ndarray:
        """Extract 15 features for a single student.

        Args:
            records: Student's LongitudinalRecords for the requested weeks.
            weeks: All requested week numbers.
            n_weeks: Total number of weeks.
            student_weekly_scores: {week: ensemble_score} for this student.
            week_stats: {week: (class_mean, class_std)} for z-score.

        Returns:
            1-D array of 15 features.
        """
        features = np.zeros(15, dtype=float)

        # Collect per-week aggregates
        week_scores: dict[int, list[float]] = defaultdict(list)
        week_coverage: dict[int, list[float]] = defaultdict(list)
        week_tiers: dict[int, list[int]] = defaultdict(list)
        week_misconceptions: dict[int, list[float]] = defaultdict(list)
        week_edge_f1: dict[int, list[float]] = defaultdict(list)

        for r in records:
            score = r.scores.get("ensemble_score", 0.0)
            coverage = r.scores.get("concept_coverage", 0.0)
            week_scores[r.week].append(score)
            week_coverage[r.week].append(coverage)
            week_tiers[r.week].append(r.tier_level)
            week_misconceptions[r.week].append(
                float(r.misconception_count) if r.misconception_count is not None else 0.0
            )
            week_edge_f1[r.week].append(
                float(r.edge_f1) if r.edge_f1 is not None else 0.0
            )

        # Average per week
        scores_by_week = {w: np.mean(vs) for w, vs in sorted(week_scores.items())}
        coverage_by_week = {w: np.mean(vs) for w, vs in sorted(week_coverage.items())}
        _tier_by_week = {w: np.mean(vs) for w, vs in sorted(week_tiers.items())}
        misc_by_week = {w: np.mean(vs) for w, vs in sorted(week_misconceptions.items())}
        f1_by_week = {w: np.mean(vs) for w, vs in sorted(week_edge_f1.items())}

        score_values = [scores_by_week[w] for w in sorted(scores_by_week)]
        coverage_values = [coverage_by_week[w] for w in sorted(coverage_by_week)]
        misc_values = [misc_by_week[w] for w in sorted(misc_by_week)]
        f1_values = [f1_by_week[w] for w in sorted(f1_by_week)]

        # 1-4: Score features
        features[0] = np.mean(score_values) if score_values else 0.0  # score_mean
        features[1] = np.var(score_values) if len(score_values) > 1 else 0.0  # score_variance
        features[2] = _ols_slope(score_values)  # score_slope
        features[3] = score_values[-1] if score_values else 0.0  # last_score

        # 5-6: Coverage features
        features[4] = np.mean(coverage_values) if coverage_values else 0.0  # coverage_mean
        features[5] = _ols_slope(coverage_values)  # coverage_slope

        # 7: Tier low ratio
        all_tiers = [t for tiers in week_tiers.values() for t in tiers]
        features[6] = sum(1 for t in all_tiers if t <= 1) / len(all_tiers) if all_tiers else 0.0

        # 8-9: Misconception features
        features[7] = np.mean(misc_values) if misc_values else 0.0  # misconception_mean
        features[8] = _ols_slope(misc_values)  # misconception_slope

        # 10-11: Absence features
        present_weeks = set(week_scores.keys())
        absent = sum(1 for w in weeks if w not in present_weeks)
        features[9] = float(absent)  # absence_count
        features[10] = absent / n_weeks if n_weeks > 0 else 0.0  # absence_ratio

        # 12-13: Z-score features
        z_scores = []
        for w in sorted(scores_by_week.keys()):
            if w in week_stats:
                mean, std = week_stats[w]
                if std > 0:
                    z = (scores_by_week[w] - mean) / std
                else:
                    z = 0.0
                z_scores.append(z)
        features[11] = np.mean(z_scores) if z_scores else 0.0  # z_score_mean
        features[12] = _ols_slope(z_scores)  # z_score_slope

        # 14-15: Edge F1 features
        features[13] = np.mean(f1_values) if f1_values else 0.0  # edge_f1_mean
        features[14] = _ols_slope(f1_values)  # edge_f1_slope

        return features


class RiskPredictor:
    """Trains logistic regression models and predicts drop risk.

    Uses StandardScaler for feature normalization and StratifiedKFold
    cross-validation for training quality metrics.
    """

    def train(
        self,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str],
        *,
        min_students: int = 10,
        n_weeks: int = 3,
        target_threshold: float = 0.45,
    ) -> TrainedRiskModel:
        """Train a logistic regression model.

        Args:
            feature_matrix: (n_students, n_features) array.
            labels: Binary labels (1 = drop, 0 = no drop).
            feature_names: Feature names matching matrix columns.
            min_students: Minimum required students.
            n_weeks: Number of weeks of data.
            target_threshold: Score threshold for drop definition.

        Returns:
            TrainedRiskModel with fitted model, scaler, and metadata.

        Raises:
            ValueError: If insufficient students for training.
        """
        n_students = feature_matrix.shape[0]
        if n_students < min_students:
            raise ValueError(
                f"Insufficient students for training: {n_students} < {min_students}"
            )

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix)

        # Check for single-class labels before fitting
        n_unique = len(np.unique(labels))

        # Train logistic regression
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )
        if n_unique >= 2:
            model.fit(X_scaled, labels)
        else:
            # Single class: fit with a dummy second class to avoid sklearn error
            logger.warning(
                "Training data contains only one class. "
                "Synthetic augmentation used. Model predictions may be unreliable."
            )
            X_aug = np.vstack([X_scaled, X_scaled[0:1]])
            labels_aug = np.append(labels, 1 - labels[0])
            model.fit(X_aug, labels_aug)

        # Cross-validation score
        if n_unique < 2:
            cv_score = 0.0
        else:
            n_splits = min(5, min(np.sum(labels == 0), np.sum(labels == 1)))
            if n_splits < 2:
                cv_score = 0.0
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_scaled, labels, cv=cv, scoring="accuracy")
                cv_score = float(np.mean(scores))

        return TrainedRiskModel(
            model=model,
            feature_names=feature_names,
            scaler=scaler,
            training_date=datetime.now(timezone.utc).isoformat(),
            n_students=n_students,
            n_weeks=n_weeks,
            cv_score=cv_score,
            target_threshold=target_threshold,
        )

    def predict(
        self,
        trained_model: TrainedRiskModel,
        feature_matrix: np.ndarray,
        student_ids: list[str],
    ) -> list[RiskPrediction]:
        """Predict drop risk using a trained model.

        Args:
            trained_model: Fitted TrainedRiskModel.
            feature_matrix: (n_students, n_features) array.
            student_ids: Student identifiers matching matrix rows.

        Returns:
            List of RiskPrediction, one per student.
        """
        X_scaled = trained_model.scaler.transform(feature_matrix)
        probabilities = trained_model.model.predict_proba(X_scaled)

        # Column index for drop class (label=1)
        classes = trained_model.model.classes_
        drop_idx = list(classes).index(1) if 1 in classes else 0

        coefficients = trained_model.model.coef_[0]
        feature_names = trained_model.feature_names

        predictions = []
        for i, sid in enumerate(student_ids):
            drop_prob = float(probabilities[i, drop_idx])

            # Compute risk factors from coefficients
            factors = []
            for j, name in enumerate(feature_names):
                coef = coefficients[j]
                contribution = coef * X_scaled[i, j]
                factors.append(RiskFactor(
                    name=name,
                    importance=abs(float(coef)),
                    value=float(feature_matrix[i, j]),
                    direction="increasing_risk" if contribution > 0 else "decreasing_risk",
                ))

            # Sort by importance descending
            factors.sort(key=lambda f: f.importance, reverse=True)

            # Predicted tier based on probability
            if drop_prob >= 0.7:
                predicted_tier = 0
            elif drop_prob >= 0.5:
                predicted_tier = 1
            elif drop_prob >= 0.3:
                predicted_tier = 2
            else:
                predicted_tier = 3

            predictions.append(RiskPrediction(
                student_id=sid,
                drop_probability=drop_prob,
                risk_factors=factors,
                predicted_tier=predicted_tier,
                is_model_based=True,
                confidence="high",
            ))

        return predictions

    def predict_cold_start(
        self,
        feature_matrix: np.ndarray,
        student_ids: list[str],
        feature_names: list[str],
    ) -> list[RiskPrediction]:
        """Predict drop risk without a pre-trained model (cold start).

        Uses simple heuristics based on feature values. Results have
        limited confidence.

        Args:
            feature_matrix: (n_students, n_features) array.
            student_ids: Student identifiers.
            feature_names: Feature names.

        Returns:
            List of RiskPrediction with confidence="limited".
        """
        predictions = []
        score_mean_idx = feature_names.index("score_mean") if "score_mean" in feature_names else 0
        score_slope_idx = feature_names.index("score_slope") if "score_slope" in feature_names else 2
        absence_ratio_idx = feature_names.index("absence_ratio") if "absence_ratio" in feature_names else 10

        for i, sid in enumerate(student_ids):
            score_mean = feature_matrix[i, score_mean_idx]
            score_slope = feature_matrix[i, score_slope_idx]
            absence_ratio = feature_matrix[i, absence_ratio_idx]

            # Simple heuristic: weighted combination
            drop_prob = max(0.0, min(1.0,
                (1.0 - score_mean) * 0.5 +
                max(0, -score_slope) * 2.0 +
                absence_ratio * 0.3
            ))

            factors = []
            for j, name in enumerate(feature_names):
                factors.append(RiskFactor(
                    name=name,
                    importance=abs(float(feature_matrix[i, j])),
                    value=float(feature_matrix[i, j]),
                    direction="increasing_risk" if feature_matrix[i, j] > 0.5 else "decreasing_risk",
                ))
            factors.sort(key=lambda f: f.importance, reverse=True)

            predicted_tier = 0 if drop_prob >= 0.7 else (1 if drop_prob >= 0.5 else (2 if drop_prob >= 0.3 else 3))

            predictions.append(RiskPrediction(
                student_id=sid,
                drop_probability=drop_prob,
                risk_factors=factors,
                predicted_tier=predicted_tier,
                is_model_based=False,
                confidence="limited",
            ))

        return predictions


def save_model(model: TrainedRiskModel, path: Path | str) -> None:
    """Persist a TrainedRiskModel via joblib.

    Args:
        model: Trained model to save.
        path: Output file path (.pkl).
    """
    joblib.dump(model, str(path))
    logger.info("Model saved to %s", path)


def load_model(path: Path | str) -> TrainedRiskModel:
    """Load a TrainedRiskModel from a joblib file.

    Args:
        path: Path to the .pkl model file.

    Returns:
        Loaded TrainedRiskModel.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the loaded object is not a TrainedRiskModel.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    obj = joblib.load(str(path))
    if not isinstance(obj, TrainedRiskModel):
        raise TypeError(
            f"Expected TrainedRiskModel, got {type(obj).__name__}"
        )
    logger.info("Model loaded from %s", path)
    return obj
