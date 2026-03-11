"""Layer 4: Weighted ensemble scorer and understanding-level classifier.

Default weights are professor-defined (criterion-referenced), NOT
auto-learned.  Auto-weighting is an optional extension that requires
an external holistic grade for validation.

Understanding levels are criterion-referenced:
  Advanced  : E ≥ 0.85
  Proficient: 0.65 ≤ E < 0.85
  Developing: 0.45 ≤ E < 0.65
  Beginning : E < 0.45
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from forma.evaluation_types import (
    AggregatedLLMResult,
    ConceptMatchResult,
    EnsembleResult,
    GraphComparisonResult,
    GraphMetricResult,
    StatisticalResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# v1 weights (no knowledge_graph triplet comparison)
DEFAULT_WEIGHTS: dict[str, float] = {
    "concept_coverage": 0.35,
    "llm_rubric": 0.30,
    "rasch_ability": 0.15,
    "kg_node_recall": 0.10,
    "bertscore": 0.10,
}

WEIGHTS_V1: dict[str, float] = dict(DEFAULT_WEIGHTS)

# v2 weights (with knowledge_graph triplet comparison)
WEIGHTS_V2: dict[str, float] = {
    "concept_coverage": 0.25,
    "graph_f1": 0.30,
    "rasch_ability": 0.15,
    "rubric_level": 0.15,
    "bertscore": 0.10,
    "misconception_penalty": -0.05,
}

UNDERSTANDING_THRESHOLDS: dict[str, float] = {
    "Advanced": 0.85,
    "Proficient": 0.65,
    "Developing": 0.45,
    "Beginning": 0.0,
}


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def classify_understanding_level(
    ensemble_score: float,
    thresholds: dict[str, float] = UNDERSTANDING_THRESHOLDS,
) -> str:
    """Classify a student's understanding level by criterion-referenced cutoffs.

    Evaluates levels from highest to lowest; returns the first level whose
    threshold the score meets or exceeds.

    Args:
        ensemble_score: Ensemble score in [0, 1].
        thresholds: Dict mapping level name → minimum score (default
            UNDERSTANDING_THRESHOLDS).

    Returns:
        One of "Advanced", "Proficient", "Developing", "Beginning".

    Examples:
        >>> classify_understanding_level(0.90)
        'Advanced'
        >>> classify_understanding_level(0.30)
        'Beginning'
    """
    for level in ["Advanced", "Proficient", "Developing", "Beginning"]:
        if ensemble_score >= thresholds[level]:
            return level
    return "Beginning"


def normalize_score(
    raw: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> float:
    """Normalize a raw score to [0, 1] using linear clipping.

    Sigmoid transformation preserves ordinal ordering and is documented
    in the plan; this implementation uses linear clamp for simplicity.
    For scores already in [0, 1] this is a no-op.

    Args:
        raw: Raw score value.
        min_val: Minimum of the raw scale (default 0.0).
        max_val: Maximum of the raw scale (default 1.0).

    Returns:
        Normalized float in [0, 1].

    Examples:
        >>> normalize_score(0.7)
        0.7
        >>> normalize_score(3.0, 1.0, 3.0)
        1.0
    """
    if max_val <= min_val:
        return 0.0
    return float(np.clip((raw - min_val) / (max_val - min_val), 0.0, 1.0))


# ---------------------------------------------------------------------------
# EnsembleScorer class
# ---------------------------------------------------------------------------


class EnsembleScorer:
    """Compute a weighted ensemble score from all evaluation layers.

    By default uses professor-defined weights.  Components with None
    values are skipped and their weights are redistributed proportionally
    to the remaining components.

    Args:
        weights: Dict mapping metric key → weight.  Must sum to 1.0.
            Defaults to DEFAULT_WEIGHTS.
        thresholds: Understanding-level cutoffs.  Defaults to
            UNDERSTANDING_THRESHOLDS.

    Raises:
        ValueError: If weights do not sum to 1.0 (within 1e-6 tolerance).

    Examples:
        >>> scorer = EnsembleScorer()
        >>> sum(scorer.weights.values())
        1.0
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        thresholds: Optional[dict[str, float]] = None,
    ) -> None:
        self.weights: dict[str, float] = (
            weights if weights is not None else dict(DEFAULT_WEIGHTS)
        )
        self.thresholds: dict[str, float] = (
            thresholds if thresholds is not None else dict(UNDERSTANDING_THRESHOLDS)
        )
        self._validate_weights()

    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Ensemble weights must sum to 1.0, got {total:.6f} in "
                "EnsembleScorer._validate_weights(). "
                "Adjust the weight values so they sum to exactly 1.0."
            )

    def _concept_coverage_score(
        self, concept_results: list[ConceptMatchResult]
    ) -> float:
        """Compute fraction of concepts present."""
        if not concept_results:
            return 0.0
        return float(sum(r.is_present for r in concept_results) / len(concept_results))

    def _llm_rubric_score(self, llm_result: AggregatedLLMResult) -> float:
        """Normalize median rubric score (1–3) to [0, 1]."""
        return normalize_score(llm_result.median_rubric_score, 1.0, 3.0)

    def _rasch_ability_score(self, stat_result: StatisticalResult) -> Optional[float]:
        """Normalize Rasch theta to [0, 1] using [-4, +4] logit scale."""
        if stat_result.rasch_theta is None:
            return None
        return normalize_score(stat_result.rasch_theta, -4.0, 4.0)

    def _kg_node_recall_score(self, graph_result: GraphMetricResult) -> float:
        """Return node recall directly (already in [0, 1])."""
        return float(np.clip(graph_result.node_recall, 0.0, 1.0))

    def _graph_f1_score(self, graph_comparison: GraphComparisonResult) -> float:
        """Return graph comparison F1 (already in [0, 1])."""
        return float(np.clip(graph_comparison.f1, 0.0, 1.0))

    def _misconception_penalty_score(
        self, graph_comparison: GraphComparisonResult
    ) -> float:
        """Compute misconception penalty from wrong-direction edges.

        Returns a value in [0, 1] proportional to wrong-direction count.
        """
        total = (
            len(graph_comparison.matched_edges)
            + len(graph_comparison.missing_edges)
            + len(graph_comparison.extra_edges)
            + len(graph_comparison.wrong_direction_edges)
        )
        if total == 0:
            return 0.0
        return float(
            np.clip(len(graph_comparison.wrong_direction_edges) / total, 0.0, 1.0)
        )

    def _deterministic_rubric_score(
        self,
        graph_comparison: GraphComparisonResult,
        rubric_tiers: Optional[list] = None,
    ) -> float:
        """Compute deterministic rubric score from graph F1 and tier config.

        Without rubric_tiers, maps F1 linearly to [0, 1].
        With rubric_tiers, returns tier_level / 3.
        """
        if rubric_tiers:
            f1 = graph_comparison.f1
            best_level = 0
            for tier in rubric_tiers:
                if f1 >= tier.min_graph_f1:
                    best_level = max(best_level, tier.level)
            return float(best_level) / 3.0
        return float(np.clip(graph_comparison.f1, 0.0, 1.0))

    def compute_score(
        self,
        concept_results: list[ConceptMatchResult],
        llm_result: Optional[AggregatedLLMResult],
        statistical_result: Optional[StatisticalResult],
        graph_result: Optional[GraphMetricResult],
        bertscore_f1: Optional[float],
        student_id: str,
        question_sn: int,
        graph_comparison: Optional[GraphComparisonResult] = None,
        rubric_tiers: Optional[list] = None,
        emphasis_map: Optional["InstructionalEmphasisMap"] = None,  # noqa: F821
    ) -> EnsembleResult:
        """Compute the weighted ensemble score from all available layers.

        Auto-selects v1 or v2 weights based on ``graph_comparison`` presence.
        Missing layers (None) are dropped and their weights redistributed
        proportionally.  Concept coverage is always required.

        Args:
            concept_results: Layer-1 concept match results.
            llm_result: Layer-2 aggregated LLM result (or None to skip).
            statistical_result: Layer-3 Rasch/LCA result (or None to skip).
            graph_result: Layer-3 KG metric result (or None to skip).
            bertscore_f1: BERTScore F1 from cohesion analysis (or None).
            student_id: Student identifier for result stamping.
            question_sn: Question serial number for result stamping.
            graph_comparison: v2 directed triplet comparison (or None).
            rubric_tiers: Optional list of RubricTier for deterministic rubric.
            emphasis_map: Optional InstructionalEmphasisMap for weighted
                concept coverage. None preserves existing behavior.

        Returns:
            EnsembleResult with score, level, and component breakdown.
        """
        component_scores: dict[str, float] = {}
        active_weights: dict[str, float] = {}

        # Auto-select weight preset
        use_v2 = graph_comparison is not None
        weights = self.weights

        # --- concept_coverage (always computed) ---
        if emphasis_map is not None:
            from forma.emphasis_map import compute_weighted_concept_coverage
            mastery = {
                r.concept: 1.0 if r.is_present else 0.0
                for r in concept_results
            }
            cc = compute_weighted_concept_coverage(emphasis_map, mastery)
        else:
            cc = self._concept_coverage_score(concept_results)
        component_scores["concept_coverage"] = cc
        active_weights["concept_coverage"] = weights.get(
            "concept_coverage", 0.35 if not use_v2 else 0.25
        )

        if use_v2 and graph_comparison is not None:
            # --- v2 mode: graph-based scoring ---

            # graph_f1
            gf1 = self._graph_f1_score(graph_comparison)
            component_scores["graph_f1"] = gf1
            active_weights["graph_f1"] = weights.get("graph_f1", 0.30)

            # rubric_level (deterministic)
            rl = self._deterministic_rubric_score(graph_comparison, rubric_tiers)
            component_scores["rubric_level"] = rl
            active_weights["rubric_level"] = weights.get("rubric_level", 0.15)

            # misconception_penalty
            mp = self._misconception_penalty_score(graph_comparison)
            component_scores["misconception_penalty"] = mp
            active_weights["misconception_penalty"] = weights.get(
                "misconception_penalty", -0.05
            )
        else:
            # --- v1 mode: LLM-based scoring ---
            if llm_result is not None:
                lr = self._llm_rubric_score(llm_result)
                component_scores["llm_rubric"] = lr
                active_weights["llm_rubric"] = weights.get("llm_rubric", 0.30)

            # kg_node_recall (v1 only)
            if graph_result is not None:
                kg = self._kg_node_recall_score(graph_result)
                component_scores["kg_node_recall"] = kg
                active_weights["kg_node_recall"] = weights.get(
                    "kg_node_recall", 0.10
                )

        # --- rasch_ability (both modes) ---
        if statistical_result is not None:
            ra = self._rasch_ability_score(statistical_result)
            if ra is not None:
                component_scores["rasch_ability"] = ra
                active_weights["rasch_ability"] = weights.get(
                    "rasch_ability", 0.15
                )

        # --- bertscore (both modes) ---
        if bertscore_f1 is not None:
            bs = float(np.clip(bertscore_f1, 0.0, 1.0))
            component_scores["bertscore"] = bs
            active_weights["bertscore"] = weights.get("bertscore", 0.10)

        # --- weighted sum with proportional redistribution ---
        # Separate positive and negative weights
        pos_weights = {
            k: v for k, v in active_weights.items() if v >= 0
        }
        neg_weights = {
            k: v for k, v in active_weights.items() if v < 0
        }

        total_pos = sum(pos_weights.values())
        if total_pos <= 0:
            ensemble_score = 0.0
        else:
            ensemble_score = sum(
                component_scores[k] * pos_weights[k] / total_pos
                for k in pos_weights
            )
            # Apply negative weights (penalties) directly
            for k, w in neg_weights.items():
                ensemble_score += component_scores[k] * abs(w)  # penalty reduces score

        ensemble_score = float(np.clip(ensemble_score, 0.0, 1.0))
        level = classify_understanding_level(ensemble_score, self.thresholds)

        return EnsembleResult(
            student_id=student_id,
            question_sn=question_sn,
            ensemble_score=ensemble_score,
            understanding_level=level,
            component_scores=component_scores,
            weights_used=active_weights,
        )
