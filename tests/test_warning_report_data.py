"""Tests for warning_report_data.py — RiskType, WarningCard, build_warning_data().

T038 [US3]: Risk type classification, union inclusion, intervention mapping, ordering.
"""

from __future__ import annotations


class TestRiskType:
    """Tests for RiskType enum."""

    def test_four_risk_types(self):
        """RiskType has exactly 4 values."""
        from forma.warning_report_data import RiskType

        assert len(RiskType) == 4

    def test_risk_type_values(self):
        """RiskType values match spec."""
        from forma.warning_report_data import RiskType

        assert RiskType.SCORE_DECLINE.value == "SCORE_DECLINE"
        assert RiskType.PERSISTENT_LOW.value == "PERSISTENT_LOW"
        assert RiskType.CONCEPT_DEFICIT.value == "CONCEPT_DEFICIT"
        assert RiskType.PARTICIPATION_DECLINE.value == "PARTICIPATION_DECLINE"

    def test_korean_labels(self):
        """Each risk type has a Korean label."""
        from forma.warning_report_data import RiskType

        assert RiskType.SCORE_DECLINE.label == "점수 하락형"
        assert RiskType.PERSISTENT_LOW.label == "지속 저성취형"
        assert RiskType.CONCEPT_DEFICIT.label == "개념 결손형"
        assert RiskType.PARTICIPATION_DECLINE.label == "참여 저하형"


class TestInterventionMap:
    """Tests for INTERVENTION_MAP."""

    def test_all_types_have_interventions(self):
        """Every risk type has at least 2 intervention recommendations."""
        from forma.warning_report_data import INTERVENTION_MAP, RiskType

        for rt in RiskType:
            assert rt.value in INTERVENTION_MAP, f"Missing {rt.value}"
            assert len(INTERVENTION_MAP[rt.value]) >= 2


class TestWarningCard:
    """Tests for WarningCard dataclass."""

    def test_creation(self):
        """WarningCard can be created with required fields."""
        from forma.warning_report_data import RiskType, WarningCard

        card = WarningCard(
            student_id="S001",
            risk_types=[RiskType.SCORE_DECLINE],
            detection_methods=["rule_based"],
            deficit_concepts=["세포", "조직"],
            misconception_patterns=["CAUSAL_REVERSAL"],
            interventions=["최근 성적 하락 추세에 대한 개별 면담 권장"],
            drop_probability=0.7,
            risk_severity=0.7,
        )
        assert card.student_id == "S001"
        assert card.risk_severity == 0.7


class TestBuildWarningData:
    """Tests for build_warning_data()."""

    def test_empty_students_returns_empty(self):
        """No at-risk students returns empty list."""
        from forma.warning_report_data import build_warning_data

        result = build_warning_data(
            at_risk_students={},
            risk_predictions=[],
            concept_scores={},
        )
        assert result == []

    def test_rule_based_inclusion(self):
        """Students flagged by rule-based detection are included."""
        from forma.warning_report_data import build_warning_data

        result = build_warning_data(
            at_risk_students={
                "S001": {"is_at_risk": True, "reasons": ["low score"]},
            },
            risk_predictions=[],
            concept_scores={"S001": {"세포": 0.2, "조직": 0.1, "기관": 0.8}},
            score_trajectories={"S001": [0.5, 0.4, 0.3]},
            absence_ratios={"S001": 0.1},
        )
        assert len(result) == 1
        assert result[0].student_id == "S001"
        assert "rule_based" in result[0].detection_methods

    def test_model_predicted_inclusion(self):
        """Students with drop_probability >= 0.5 are included."""
        from forma.risk_predictor import RiskPrediction
        from forma.warning_report_data import build_warning_data

        pred = RiskPrediction(
            student_id="S002",
            drop_probability=0.6,
            is_model_based=True,
        )
        result = build_warning_data(
            at_risk_students={},
            risk_predictions=[pred],
            concept_scores={"S002": {"세포": 0.5}},
            score_trajectories={"S002": [0.5, 0.5, 0.5]},
        )
        assert len(result) == 1
        assert "model_predicted" in result[0].detection_methods

    def test_union_inclusion_both_methods(self):
        """Student flagged by both methods has both in detection_methods."""
        from forma.risk_predictor import RiskPrediction
        from forma.warning_report_data import build_warning_data

        pred = RiskPrediction(
            student_id="S001",
            drop_probability=0.7,
            is_model_based=True,
        )
        result = build_warning_data(
            at_risk_students={
                "S001": {"is_at_risk": True, "reasons": ["low score"]},
            },
            risk_predictions=[pred],
            concept_scores={"S001": {"세포": 0.2}},
            score_trajectories={"S001": [0.5, 0.4, 0.3]},
        )
        assert len(result) == 1
        assert "rule_based" in result[0].detection_methods
        assert "model_predicted" in result[0].detection_methods

    def test_deficit_concepts_below_threshold(self):
        """Concepts with mastery < 0.3 are listed as deficit concepts."""
        from forma.warning_report_data import build_warning_data

        result = build_warning_data(
            at_risk_students={
                "S001": {"is_at_risk": True, "reasons": ["low"]},
            },
            risk_predictions=[],
            concept_scores={"S001": {"세포": 0.1, "조직": 0.2, "기관": 0.8}},
            score_trajectories={"S001": [0.4]},
        )
        assert "세포" in result[0].deficit_concepts
        assert "조직" in result[0].deficit_concepts
        assert "기관" not in result[0].deficit_concepts

    def test_ordering_by_severity(self):
        """Warning cards are sorted by risk_severity descending."""
        from forma.risk_predictor import RiskPrediction
        from forma.warning_report_data import build_warning_data

        preds = [
            RiskPrediction(student_id="S001", drop_probability=0.5),
            RiskPrediction(student_id="S002", drop_probability=0.9),
            RiskPrediction(student_id="S003", drop_probability=0.7),
        ]
        result = build_warning_data(
            at_risk_students={},
            risk_predictions=preds,
            concept_scores={},
            score_trajectories={},
        )
        severities = [c.risk_severity for c in result]
        assert severities == sorted(severities, reverse=True)

    def test_score_decline_classification(self):
        """Negative score slope triggers SCORE_DECLINE risk type."""
        from forma.warning_report_data import RiskType, build_warning_data

        result = build_warning_data(
            at_risk_students={
                "S001": {"is_at_risk": True, "reasons": ["low"]},
            },
            risk_predictions=[],
            concept_scores={},
            score_trajectories={"S001": [0.6, 0.5, 0.3]},  # declining
        )
        assert any(RiskType.SCORE_DECLINE in c.risk_types for c in result)

    def test_persistent_low_classification(self):
        """All weeks below 0.45 triggers PERSISTENT_LOW risk type."""
        from forma.warning_report_data import RiskType, build_warning_data

        result = build_warning_data(
            at_risk_students={
                "S001": {"is_at_risk": True, "reasons": ["low"]},
            },
            risk_predictions=[],
            concept_scores={},
            score_trajectories={"S001": [0.3, 0.2, 0.4]},  # all below 0.45
        )
        assert any(RiskType.PERSISTENT_LOW in c.risk_types for c in result)

    def test_concept_deficit_classification(self):
        """>=3 concepts below 0.3 triggers CONCEPT_DEFICIT risk type."""
        from forma.warning_report_data import RiskType, build_warning_data

        result = build_warning_data(
            at_risk_students={
                "S001": {"is_at_risk": True, "reasons": ["low"]},
            },
            risk_predictions=[],
            concept_scores={
                "S001": {"A": 0.1, "B": 0.2, "C": 0.1, "D": 0.8},
            },
            score_trajectories={"S001": [0.4]},
        )
        assert any(RiskType.CONCEPT_DEFICIT in c.risk_types for c in result)

    def test_participation_decline_classification(self):
        """absence_ratio > 0.3 triggers PARTICIPATION_DECLINE risk type."""
        from forma.warning_report_data import RiskType, build_warning_data

        result = build_warning_data(
            at_risk_students={
                "S001": {"is_at_risk": True, "reasons": ["low"]},
            },
            risk_predictions=[],
            concept_scores={},
            score_trajectories={"S001": [0.4]},
            absence_ratios={"S001": 0.5},
        )
        assert any(RiskType.PARTICIPATION_DECLINE in c.risk_types for c in result)

    def test_interventions_mapped(self):
        """Warning cards have interventions from INTERVENTION_MAP."""
        from forma.warning_report_data import build_warning_data

        result = build_warning_data(
            at_risk_students={
                "S001": {"is_at_risk": True, "reasons": ["low"]},
            },
            risk_predictions=[],
            concept_scores={},
            score_trajectories={"S001": [0.3, 0.2, 0.1]},
        )
        assert len(result[0].interventions) > 0

    def test_below_threshold_not_included(self):
        """Students with probability < 0.5 and not rule-based are excluded."""
        from forma.risk_predictor import RiskPrediction
        from forma.warning_report_data import build_warning_data

        pred = RiskPrediction(student_id="S099", drop_probability=0.3)
        result = build_warning_data(
            at_risk_students={},
            risk_predictions=[pred],
            concept_scores={},
        )
        assert len(result) == 0
