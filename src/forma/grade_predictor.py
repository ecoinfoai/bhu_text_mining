"""Grade prediction model: feature extraction, training, and persistence.

Loads grade mapping YAML files, extracts 21 features (15 base + 6 grade-specific)
from longitudinal data, trains LogisticRegression for ordinal grade prediction,
and provides model persistence via joblib.

Constants:
    VALID_GRADES: Set of allowed grade strings.
    GRADE_ORDINAL_MAP: Ordinal encoding A=4, B=3, C=2, D=1, F=0.
    GRADE_FEATURE_NAMES: 21 feature names in order.

Dataclasses:
    GradePrediction: Per-student grade prediction result.
    TrainedGradeModel: Persisted grade model wrapper with metadata.

Classes:
    GradeFeatureExtractor: Extracts 21-feature vectors from LongitudinalStore.
    GradePredictor: Trains and predicts with LogisticRegression.

Functions:
    load_grade_mapping: Load and validate grade mapping from YAML file.
    save_grade_model: Persist TrainedGradeModel via joblib.
    load_grade_model: Load TrainedGradeModel from joblib file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional  # used in GradeFeatureExtractor.extract param

import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from forma.risk_predictor import FEATURE_NAMES as BASE_FEATURE_NAMES
from forma.risk_predictor import FeatureExtractor

logger = logging.getLogger(__name__)

VALID_GRADES = {"A", "B", "C", "D", "F"}

GRADE_ORDINAL_MAP = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}

ORDINAL_GRADE_MAP = {v: k for k, v in GRADE_ORDINAL_MAP.items()}

# 21 features: 15 base + 6 grade-specific
GRADE_FEATURE_NAMES = list(BASE_FEATURE_NAMES) + [
    "prior_grade_ordinal",
    "grade_trend",
    "best_prior_grade",
    "worst_prior_grade",
    "n_prior_semesters",
    "prior_grade_variance",
]


def load_grade_mapping(
    path: str,
    store_student_ids: Optional[set[str]] = None,
) -> dict[str, dict[str, str]]:
    """Load and validate grade mapping from a YAML file.

    Expected YAML format:
        semester_label:
            student_id: grade
            ...

    Args:
        path: Path to the grade mapping YAML file.
        store_student_ids: Optional set of known student IDs for
            mismatch validation.

    Returns:
        Dict with semester-grouped structure {semester_label: {student_id: grade}}.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any grade is not in VALID_GRADES.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise

    if not data or not isinstance(data, dict):
        return {}

    result: dict[str, dict[str, str]] = {}
    all_grade_student_ids: set[str] = set()

    for semester_label, student_grades in data.items():
        if not isinstance(student_grades, dict):
            result[semester_label] = {}
            continue

        validated: dict[str, str] = {}
        for student_id, grade in student_grades.items():
            student_id_str = str(student_id)
            grade_str = str(grade)
            if grade_str not in VALID_GRADES:
                raise ValueError(
                    f"Invalid grade '{grade_str}' for student '{student_id_str}' "
                    f"in semester '{semester_label}'. "
                    f"Must be one of {sorted(VALID_GRADES)}"
                )
            validated[student_id_str] = grade_str
            all_grade_student_ids.add(student_id_str)

        result[semester_label] = validated

    if store_student_ids is not None:
        mismatched = all_grade_student_ids - store_student_ids
        for sid in sorted(mismatched):
            logger.warning(
                "Student '%s' in grade mapping not found in longitudinal store (mismatch)",
                sid,
            )

    return result


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GradePrediction:
    """Per-student grade prediction result.

    Attributes:
        student_id: Student identifier.
        predicted_grade: Predicted letter grade (A/B/C/D/F).
        grade_probabilities: {grade: probability} for all 5 grades.
        predicted_ordinal: Predicted ordinal value (0-4).
        is_model_based: True if model-based, False if cold-start.
        confidence: "high" for model-based, "limited" for cold start.
    """

    student_id: str
    predicted_grade: str
    grade_probabilities: dict[str, float] = field(default_factory=dict)
    predicted_ordinal: int = 2
    is_model_based: bool = True
    confidence: str = "high"


@dataclass
class TrainedGradeModel:
    """Persisted grade prediction model wrapper.

    Attributes:
        model: Fitted LogisticRegression instance.
        feature_names: Ordered feature names matching model columns.
        scaler: Fitted StandardScaler for feature normalization.
        training_date: ISO 8601 date of training.
        n_students: Number of students used in training.
        n_weeks: Number of weeks of data used.
        cv_score: Mean cross-validation accuracy score.
        classes: Ordinal class labels the model was trained on.
    """

    model: LogisticRegression
    feature_names: list[str]
    scaler: StandardScaler
    training_date: str
    n_students: int
    n_weeks: int
    cv_score: float
    classes: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _ols_slope(values: list[float]) -> float:
    """Compute OLS slope from a sequence of values."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    coeffs = np.polyfit(x, values, deg=1)
    return float(coeffs[0])


class GradeFeatureExtractor:
    """Extracts 21-feature vectors from LongitudinalStore + grade history.

    Features 1-15: Same as risk_predictor.FeatureExtractor (base features).
    Features 16-21: Grade-specific features from prior semester grades.
        16: prior_grade_ordinal — most recent prior grade as ordinal
        17: grade_trend — OLS slope across prior semesters
        18: best_prior_grade — max ordinal among prior grades
        19: worst_prior_grade — min ordinal among prior grades
        20: n_prior_semesters — count of prior semester records
        21: prior_grade_variance — variance of prior grades
    """

    def __init__(self) -> None:
        self._base_extractor = FeatureExtractor()

    def extract(
        self,
        store,
        weeks: list[int],
        grade_history: Optional[dict[str, list[int]]] = None,
        class_name: str | None = None,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Extract 21-feature matrix from store and grade history.

        Args:
            store: LongitudinalStore instance.
            weeks: List of week numbers to include.
            grade_history: {student_id: [ordinal_grades]} in chronological order.
                Each ordinal grade is an int (0-4). If None, all grade
                features default to 0.
            class_name: Optional class filter.

        Returns:
            Tuple of (feature_matrix, feature_names, student_ids).
            feature_matrix shape: (n_students, 21).
        """
        if grade_history is None:
            grade_history = {}

        # Extract 15 base features
        base_matrix, _, student_ids = self._base_extractor.extract(
            store, weeks, class_name,
        )

        n_students = len(student_ids)
        # Build 6 grade-specific feature columns
        grade_features = np.zeros((n_students, 6), dtype=float)

        for idx, sid in enumerate(student_ids):
            grades = grade_history.get(sid, [])
            if grades:
                grade_features[idx, 0] = float(grades[-1])  # prior_grade_ordinal (most recent)
                grade_features[idx, 1] = _ols_slope([float(g) for g in grades])  # grade_trend
                grade_features[idx, 2] = float(max(grades))  # best_prior_grade
                grade_features[idx, 3] = float(min(grades))  # worst_prior_grade
                grade_features[idx, 4] = float(len(grades))  # n_prior_semesters
                grade_features[idx, 5] = float(np.var(grades)) if len(grades) > 1 else 0.0  # prior_grade_variance

        # Concatenate base + grade features
        full_matrix = np.hstack([base_matrix, grade_features])

        return full_matrix, list(GRADE_FEATURE_NAMES), student_ids


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class GradePredictor:
    """Trains logistic regression models and predicts semester grades.

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
        n_weeks: int = 4,
    ) -> TrainedGradeModel:
        """Train a grade prediction model.

        Args:
            feature_matrix: (n_students, n_features) array.
            labels: Ordinal grade labels (0-4).
            feature_names: Feature names matching matrix columns.
            min_students: Minimum required students.
            n_weeks: Number of weeks of data.

        Returns:
            TrainedGradeModel with fitted model, scaler, and metadata.

        Raises:
            ValueError: If insufficient students for training.
        """
        n_students = feature_matrix.shape[0]
        if n_students < min_students:
            raise ValueError(
                f"Insufficient students for training: {n_students} < {min_students}"
            )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix)

        n_unique = len(np.unique(labels))

        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )

        if n_unique >= 2:
            model.fit(X_scaled, labels)
        else:
            logger.warning(
                "Training data contains only one class. "
                "Synthetic augmentation used. Model predictions may be unreliable."
            )
            X_aug = np.vstack([X_scaled, X_scaled[0:1]])
            other_label = (labels[0] + 1) % 5
            labels_aug = np.append(labels, other_label)
            model.fit(X_aug, labels_aug)

        # Cross-validation
        if n_unique < 2:
            cv_score = 0.0
        else:
            min_class_count = min(
                int(np.sum(labels == c)) for c in np.unique(labels)
            )
            n_splits = min(5, min_class_count)
            if n_splits < 2:
                cv_score = 0.0
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_scaled, labels, cv=cv, scoring="accuracy")
                cv_score = float(np.mean(scores))

        return TrainedGradeModel(
            model=model,
            feature_names=feature_names,
            scaler=scaler,
            training_date=datetime.now(timezone.utc).isoformat(),
            n_students=n_students,
            n_weeks=n_weeks,
            cv_score=cv_score,
            classes=sorted(int(c) for c in model.classes_),
        )

    def predict(
        self,
        trained_model: TrainedGradeModel,
        feature_matrix: np.ndarray,
        student_ids: list[str],
    ) -> list[GradePrediction]:
        """Predict grades using a trained model.

        Args:
            trained_model: Fitted TrainedGradeModel.
            feature_matrix: (n_students, n_features) array.
            student_ids: Student identifiers matching matrix rows.

        Returns:
            List of GradePrediction, one per student.
        """
        X_scaled = trained_model.scaler.transform(feature_matrix)
        probabilities = trained_model.model.predict_proba(X_scaled)
        predicted_ordinals = trained_model.model.predict(X_scaled)
        classes = list(trained_model.model.classes_)

        predictions = []
        for i, sid in enumerate(student_ids):
            ordinal = int(predicted_ordinals[i])
            grade = ORDINAL_GRADE_MAP.get(ordinal, "C")

            # Build grade probability dict
            grade_probs: dict[str, float] = {}
            for j, cls in enumerate(classes):
                g = ORDINAL_GRADE_MAP.get(int(cls), "C")
                grade_probs[g] = float(probabilities[i, j])

            # Fill missing grades with 0
            for g in VALID_GRADES:
                if g not in grade_probs:
                    grade_probs[g] = 0.0

            predictions.append(GradePrediction(
                student_id=sid,
                predicted_grade=grade,
                grade_probabilities=grade_probs,
                predicted_ordinal=ordinal,
                is_model_based=True,
                confidence="high",
            ))

        return predictions

    def predict_cold_start(
        self,
        feature_matrix: np.ndarray,
        student_ids: list[str],
        feature_names: list[str],
    ) -> list[GradePrediction]:
        """Predict grades without a pre-trained model (cold start).

        Uses OLS trend on ensemble_score + percentile thresholds
        to assign grades.

        Args:
            feature_matrix: (n_students, n_features) array.
            student_ids: Student identifiers.
            feature_names: Feature names.

        Returns:
            List of GradePrediction with confidence="limited".
        """
        score_mean_idx = (
            feature_names.index("score_mean")
            if "score_mean" in feature_names
            else 0
        )
        score_slope_idx = (
            feature_names.index("score_slope")
            if "score_slope" in feature_names
            else 2
        )
        prior_idx = (
            feature_names.index("prior_grade_ordinal")
            if "prior_grade_ordinal" in feature_names
            else None
        )

        predictions = []
        for i, sid in enumerate(student_ids):
            score_mean = float(feature_matrix[i, score_mean_idx])
            score_slope = float(feature_matrix[i, score_slope_idx])
            prior_grade = float(feature_matrix[i, prior_idx]) if prior_idx is not None else 0.0

            # Projected score: mean + slope contribution + prior grade weight
            projected = score_mean + score_slope * 0.5 + prior_grade * 0.1

            # Percentile-based thresholds
            if projected >= 0.85:
                ordinal = 4  # A
            elif projected >= 0.70:
                ordinal = 3  # B
            elif projected >= 0.50:
                ordinal = 2  # C
            elif projected >= 0.30:
                ordinal = 1  # D
            else:
                ordinal = 0  # F

            grade = ORDINAL_GRADE_MAP[ordinal]

            predictions.append(GradePrediction(
                student_id=sid,
                predicted_grade=grade,
                grade_probabilities={g: (1.0 if g == grade else 0.0) for g in VALID_GRADES},
                predicted_ordinal=ordinal,
                is_model_based=False,
                confidence="limited",
            ))

        return predictions


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def save_grade_model(model: TrainedGradeModel, path: Path | str) -> None:
    """Persist a TrainedGradeModel via joblib.

    Args:
        model: Trained model to save.
        path: Output file path (.pkl).
    """
    joblib.dump(model, str(path))
    logger.info("Grade model saved to %s", path)


def load_grade_model(path: Path | str) -> TrainedGradeModel:
    """Load a TrainedGradeModel from a joblib file.

    Args:
        path: Path to the .pkl model file.

    Returns:
        Loaded TrainedGradeModel.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the loaded object is not a TrainedGradeModel.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Grade model file not found: {path}")
    obj = joblib.load(str(path))
    if not isinstance(obj, TrainedGradeModel):
        raise TypeError(
            f"Expected TrainedGradeModel, got {type(obj).__name__}"
        )
    logger.info("Grade model loaded from %s", path)
    return obj
