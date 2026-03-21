"""Early warning report data: risk classification, warning cards, and intervention mapping.

Provides RiskType enum, WarningCard dataclass, INTERVENTION_MAP, and
build_warning_data() for constructing per-student warning intervention briefs.

Risk type classification rules (FR-016b):
    SCORE_DECLINE:         Negative score slope (OLS)
    PERSISTENT_LOW:        All weekly scores below 0.45
    CONCEPT_DEFICIT:       >= 3 concepts with mastery < 0.3
    PARTICIPATION_DECLINE: absence_ratio > 0.3
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Threshold constants
_DEFICIT_MASTERY_THRESHOLD = 0.3
_PERSISTENT_LOW_THRESHOLD = 0.45
_CONCEPT_DEFICIT_MIN_COUNT = 3
_ABSENCE_HIGH_THRESHOLD = 0.3
_DROP_PROB_INCLUSION_THRESHOLD = 0.5
_SLOPE_EPSILON = 1e-9  # FR-032: guard against float-noise false positives


class RiskType(enum.Enum):
    """Risk type classification for at-risk students (FR-016b)."""

    SCORE_DECLINE = "SCORE_DECLINE"
    PERSISTENT_LOW = "PERSISTENT_LOW"
    CONCEPT_DEFICIT = "CONCEPT_DEFICIT"
    PARTICIPATION_DECLINE = "PARTICIPATION_DECLINE"

    @property
    def label(self) -> str:
        """Korean label for the risk type."""
        return _KOREAN_LABELS[self]


_KOREAN_LABELS: dict[RiskType, str] = {
    RiskType.SCORE_DECLINE: "점수 하락형",
    RiskType.PERSISTENT_LOW: "지속 저성취형",
    RiskType.CONCEPT_DEFICIT: "개념 결손형",
    RiskType.PARTICIPATION_DECLINE: "참여 저하형",
}


INTERVENTION_MAP: dict[str, list[str]] = {
    "SCORE_DECLINE": [
        "최근 성적 하락 추세에 대한 개별 면담 권장",
        "이전 주차 핵심 개념 복습 자료 제공",
        "학습 계획 재수립 지도",
    ],
    "PERSISTENT_LOW": [
        "기초 개념 보충 학습 프로그램 안내",
        "학습 튜터링 또는 스터디 그룹 매칭",
        "학습 동기 및 장애 요인 상담",
    ],
    "CONCEPT_DEFICIT": [
        "결손 개념 목록 기반 맞춤 보충 자료 제공",
        "오개념 교정을 위한 개별 피드백 강화",
        "관련 실습/사례 학습 추가 안내",
    ],
    "PARTICIPATION_DECLINE": [
        "출석 및 참여 패턴 확인 (건강, 개인 사정 등)",
        "학습 참여 독려 개별 연락",
        "학과 상담 또는 학생 지원 서비스 연계",
    ],
}


@dataclass
class WarningCard:
    """Per-student early warning intervention brief.

    Attributes:
        student_id: Student identifier.
        risk_types: Classified risk types (non-empty).
        detection_methods: "rule_based" and/or "model_predicted".
        deficit_concepts: Concepts with mastery < 0.3.
        misconception_patterns: From misconception classifier.
        interventions: Mapped from INTERVENTION_MAP.
        drop_probability: Model-predicted probability, or None.
        risk_severity: For ordering (drop_prob or rule-based score).
    """

    student_id: str
    risk_types: list[RiskType]
    detection_methods: list[str]
    deficit_concepts: list[str] = field(default_factory=list)
    misconception_patterns: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    drop_probability: float | None = None
    risk_severity: float = 0.0


def _classify_risk_types(
    concept_scores: dict[str, float],
    score_trajectory: list[float],
    absence_ratio: float,
) -> list[RiskType]:
    """Classify a student's risk types based on available data.

    Args:
        concept_scores: {concept: mastery_score} for this student.
        score_trajectory: Weekly scores in chronological order.
        absence_ratio: Fraction of missed sessions.

    Returns:
        List of applicable RiskType values.
    """
    risk_types: list[RiskType] = []

    # SCORE_DECLINE: negative OLS slope on score trajectory
    if len(score_trajectory) >= 2:
        x = np.arange(len(score_trajectory), dtype=float)
        coeffs = np.polyfit(x, score_trajectory, deg=1)
        if coeffs[0] < -_SLOPE_EPSILON:
            risk_types.append(RiskType.SCORE_DECLINE)

    # PERSISTENT_LOW: all weekly scores below threshold
    if score_trajectory and all(
        s < _PERSISTENT_LOW_THRESHOLD for s in score_trajectory
    ):
        risk_types.append(RiskType.PERSISTENT_LOW)

    # CONCEPT_DEFICIT: >= 3 concepts below mastery threshold
    deficit_count = sum(
        1 for score in concept_scores.values()
        if score < _DEFICIT_MASTERY_THRESHOLD
    )
    if deficit_count >= _CONCEPT_DEFICIT_MIN_COUNT:
        risk_types.append(RiskType.CONCEPT_DEFICIT)

    # PARTICIPATION_DECLINE: high absence ratio
    if absence_ratio > _ABSENCE_HIGH_THRESHOLD:
        risk_types.append(RiskType.PARTICIPATION_DECLINE)

    return risk_types


def _compute_rule_based_severity(
    concept_scores: dict[str, float],
    score_trajectory: list[float],
    absence_ratio: float,
) -> float:
    """Compute a heuristic severity score for rule-based detections.

    Combines multiple signals into a 0.0-1.0 severity score.

    Args:
        concept_scores: {concept: mastery_score}.
        score_trajectory: Weekly scores.
        absence_ratio: Fraction of missed sessions.

    Returns:
        Severity score in [0.0, 1.0].
    """
    severity = 0.0
    n_signals = 0

    # Score trajectory component
    if score_trajectory:
        mean_score = sum(score_trajectory) / len(score_trajectory)
        severity += (1.0 - mean_score)
        n_signals += 1

    # Concept deficit component
    if concept_scores:
        deficit_ratio = sum(
            1 for s in concept_scores.values() if s < _DEFICIT_MASTERY_THRESHOLD
        ) / len(concept_scores)
        severity += deficit_ratio
        n_signals += 1

    # Absence component
    severity += absence_ratio
    n_signals += 1

    return min(1.0, severity / n_signals) if n_signals > 0 else 0.5


def build_warning_data(
    at_risk_students: dict[str, dict],
    risk_predictions: list,
    concept_scores: dict[str, dict[str, float]],
    score_trajectories: dict[str, list[float]] | None = None,
    absence_ratios: dict[str, float] | None = None,
) -> list[WarningCard]:
    """Build warning cards for at-risk students.

    Uses union inclusion: students flagged by rule-based detection OR
    with drop_probability >= 0.5 from model predictions.

    Args:
        at_risk_students: {student_id: {is_at_risk, reasons}} from identify_at_risk.
        risk_predictions: List of RiskPrediction objects from risk_predictor.
        concept_scores: {student_id: {concept: mastery}} for deficit extraction.
        score_trajectories: {student_id: [weekly_scores]} for trend analysis.
        absence_ratios: {student_id: absence_ratio} for participation check.

    Returns:
        List of WarningCard sorted by risk_severity descending.
    """
    if score_trajectories is None:
        score_trajectories = {}
    if absence_ratios is None:
        absence_ratios = {}

    # Build prediction lookup
    pred_lookup: dict[str, object] = {}
    for pred in risk_predictions:
        pred_lookup[pred.student_id] = pred

    # Determine union set of at-risk student IDs
    rule_based_ids: set[str] = set()
    for sid, info in at_risk_students.items():
        if info.get("is_at_risk", False):
            rule_based_ids.add(sid)

    model_predicted_ids: set[str] = set()
    for pred in risk_predictions:
        if pred.drop_probability >= _DROP_PROB_INCLUSION_THRESHOLD:
            model_predicted_ids.add(pred.student_id)

    all_at_risk_ids = rule_based_ids | model_predicted_ids

    if not all_at_risk_ids:
        return []

    cards: list[WarningCard] = []
    for sid in all_at_risk_ids:
        # Detection methods
        detection_methods: list[str] = []
        if sid in rule_based_ids:
            detection_methods.append("rule_based")
        if sid in model_predicted_ids:
            detection_methods.append("model_predicted")

        # Student data
        student_concepts = concept_scores.get(sid, {})
        trajectory = score_trajectories.get(sid, [])
        absence = absence_ratios.get(sid, 0.0)

        # Classify risk types
        risk_types = _classify_risk_types(student_concepts, trajectory, absence)

        # If no specific risk type classified, add a default based on detection
        if not risk_types:
            if trajectory:
                risk_types.append(RiskType.SCORE_DECLINE)
            else:
                risk_types.append(RiskType.PERSISTENT_LOW)

        # Extract deficit concepts (mastery < 0.3)
        deficit_concepts = [
            concept for concept, score in student_concepts.items()
            if score < _DEFICIT_MASTERY_THRESHOLD
        ]

        # Map interventions from risk types
        interventions: list[str] = []
        seen: set[str] = set()
        for rt in risk_types:
            for intervention in INTERVENTION_MAP.get(rt.value, []):
                if intervention not in seen:
                    interventions.append(intervention)
                    seen.add(intervention)

        # Determine drop probability and severity
        pred = pred_lookup.get(sid)
        drop_probability = pred.drop_probability if pred else None

        # Severity: use model probability if available, else rule-based heuristic
        if drop_probability is not None:
            risk_severity = drop_probability
        else:
            risk_severity = _compute_rule_based_severity(
                student_concepts, trajectory, absence,
            )

        cards.append(WarningCard(
            student_id=sid,
            risk_types=risk_types,
            detection_methods=detection_methods,
            deficit_concepts=deficit_concepts,
            misconception_patterns=[],
            interventions=interventions,
            drop_probability=drop_probability,
            risk_severity=risk_severity,
        ))

    # Sort by risk_severity descending
    cards.sort(key=lambda c: c.risk_severity, reverse=True)

    return cards
