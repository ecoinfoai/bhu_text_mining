"""Shared dataclasses for the multi-layer concept evaluation framework.

All layers (concept checker, LLM judge, statistical analysis, ensemble)
exchange results via these typed dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConceptMatchResult:
    """Result of concept-presence detection for a single (student, concept) pair.

    Uses top-k mean cosine similarity (not max-pooling) to reduce false
    positives. The binary field uses ``x_k`` naming to distinguish it
    from Rasch item difficulty ``b_k``.

    Args:
        concept: Concept term that was checked.
        student_id: Student identifier.
        question_sn: Question serial number.
        is_present: Binary presence indicator (x_k).
        similarity_score: Raw top-k mean cosine similarity.
        top_k_mean_similarity: Alias for similarity_score (explicit name).
        threshold_used: Adaptive threshold that was applied.
        method: Matching algorithm name (default "top_k_mean").

    Examples:
        >>> r = ConceptMatchResult(
        ...     concept="세포막", student_id="s001", question_sn=1,
        ...     is_present=True, similarity_score=0.72,
        ...     top_k_mean_similarity=0.72, threshold_used=0.45,
        ... )
        >>> r.is_present
        True
    """

    concept: str
    student_id: str
    question_sn: int
    is_present: bool
    similarity_score: float
    top_k_mean_similarity: float
    threshold_used: float
    method: str = "top_k_mean"


@dataclass
class LLMJudgeResult:
    """Rubric evaluation result from a single LLM API call.

    Part of the 3-call reliability protocol. Three instances are
    aggregated into an AggregatedLLMResult via median scoring.

    Args:
        student_id: Student identifier.
        question_sn: Question serial number.
        rubric_score: Integer score (1=low, 2=mid, 3=high).
        rubric_label: Human-readable label ("high", "mid", "low").
        reasoning: LLM reasoning text.
        misconceptions: List of detected misconceptions/errors.
        uncertain: Whether LLM flagged low confidence.
        call_index: Which call this is (1, 2, or 3).

    Examples:
        >>> r = LLMJudgeResult(
        ...     student_id="s001", question_sn=1, rubric_score=2,
        ...     rubric_label="mid", reasoning="기본 이해 있음",
        ...     misconceptions=[], uncertain=False, call_index=1,
        ... )
        >>> r.rubric_score
        2
    """

    student_id: str
    question_sn: int
    rubric_score: int
    rubric_label: str
    reasoning: str
    misconceptions: list[str]
    uncertain: bool
    call_index: int


@dataclass
class AggregatedLLMResult:
    """Median-aggregated LLM result across n_calls evaluations.

    The ICC(2,1) field is computed separately after all student
    evaluations are complete.  If ICC < 0.7, Layer 2 weight is
    automatically down-weighted in the ensemble.

    Args:
        student_id: Student identifier.
        question_sn: Question serial number.
        median_rubric_score: Median score across n_calls.
        rubric_label: Label from the call closest to median.
        reasoning: Reasoning from the call closest to median.
        misconceptions: Union of all misconceptions across calls.
        uncertain: True if any call flagged uncertainty.
        icc_value: ICC(2,1) across calls; None until computed.
        individual_calls: The raw LLMJudgeResult objects.

    Examples:
        >>> agg = AggregatedLLMResult(
        ...     student_id="s001", question_sn=1,
        ...     median_rubric_score=2.0, rubric_label="mid",
        ...     reasoning="test", misconceptions=[],
        ...     uncertain=False, icc_value=0.85, individual_calls=[],
        ... )
        >>> agg.median_rubric_score
        2.0
    """

    student_id: str
    question_sn: int
    median_rubric_score: float
    rubric_label: str
    reasoning: str
    misconceptions: list[str]
    uncertain: bool
    icc_value: Optional[float]
    individual_calls: list[LLMJudgeResult]


@dataclass
class StatisticalResult:
    """Rasch IRT and LCA results for a single student-question pair.

    Rasch analysis is performed per-question (not pooled) to protect
    unidimensionality assumptions.  LCA results carry an exploratory
    warning because N < 60 in typical usage.

    Args:
        student_id: Student identifier.
        question_sn: Question serial number.
        rasch_theta: Estimated person ability (WLE).  None if not computed.
        rasch_theta_se: Standard error of theta estimate.
        rasch_item_difficulty: Item difficulty for this question.
        lca_class: Assigned latent class index (0-based).
        lca_class_probability: Posterior probability of assigned class.
        lca_exploratory_warning: Mandatory warning string for N < 60.

    Examples:
        >>> r = StatisticalResult(student_id="s001", question_sn=1)
        >>> r.rasch_theta is None
        True
    """

    student_id: str
    question_sn: int
    rasch_theta: Optional[float] = None
    rasch_theta_se: Optional[float] = None
    rasch_item_difficulty: Optional[float] = None
    lca_class: Optional[int] = None
    lca_class_probability: Optional[float] = None
    lca_exploratory_warning: str = (
        "이 분류는 탐색적이며, 표본 크기 제한으로 신뢰도가 낮습니다"
    )


@dataclass
class GraphMetricResult:
    """Knowledge-graph comparison metrics for one student-question pair.

    If GED computation times out (> 30 s), ``normalized_ged`` is None
    and an approximate beam-search value may be stored separately.

    Args:
        student_id: Student identifier.
        question_sn: Question serial number.
        node_recall: Fraction of reference nodes present in student graph.
        edge_jaccard: Jaccard similarity of edge sets.
        centrality_deviation: Mean absolute deviation in centrality scores.
        normalized_ged: Normalised graph edit distance; None on timeout.

    Examples:
        >>> r = GraphMetricResult(
        ...     student_id="s001", question_sn=1,
        ...     node_recall=0.75, edge_jaccard=0.50,
        ...     centrality_deviation=0.30, normalized_ged=0.40,
        ... )
        >>> r.node_recall
        0.75
    """

    student_id: str
    question_sn: int
    node_recall: float
    edge_jaccard: float
    centrality_deviation: float
    normalized_ged: Optional[float]


@dataclass
class EnsembleResult:
    """Final weighted ensemble score and understanding-level classification.

    Default weights are professor-defined (not auto-learned).
    Classification is criterion-referenced:
      Advanced ≥ 0.85, Proficient ≥ 0.65, Developing ≥ 0.45, else Beginning.

    Args:
        student_id: Student identifier.
        question_sn: Question serial number.
        ensemble_score: Weighted combination in [0, 1].
        understanding_level: Criterion-referenced level string.
        component_scores: Per-metric scores before weighting.
        weights_used: Weights applied to each component.

    Examples:
        >>> r = EnsembleResult(
        ...     student_id="s001", question_sn=1,
        ...     ensemble_score=0.73, understanding_level="Proficient",
        ...     component_scores={}, weights_used={},
        ... )
        >>> r.understanding_level
        'Proficient'
    """

    student_id: str
    question_sn: int
    ensemble_score: float
    understanding_level: str
    component_scores: dict[str, float]
    weights_used: dict[str, float]
