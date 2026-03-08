"""Instructional emphasis mapping via embedding-based concept scoring.

Computes how strongly each concept is emphasized in a lecture transcript
by measuring cosine similarity between sentence and concept embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from forma.embedding_cache import encode_texts

logger = logging.getLogger(__name__)


@dataclass
class InstructionalEmphasisMap:
    """Mapping of concept emphasis scores derived from lecture transcript.

    Args:
        concept_scores: Mapping of concept name to emphasis score in [0, 1].
        threshold_used: Similarity threshold used for scoring.
        n_sentences: Number of transcript sentences processed.
        n_concepts: Number of concepts scored.
    """

    concept_scores: dict[str, float]
    threshold_used: float
    n_sentences: int
    n_concepts: int


def compute_emphasis_map(
    sentences: list[str],
    concepts: list[str],
    threshold: float = 0.65,
    model_name: str | None = None,
) -> InstructionalEmphasisMap:
    """Compute instructional emphasis scores for concepts from transcript.

    Encodes all sentences and concepts into embeddings, then counts
    sentences with cosine similarity >= threshold for each concept.
    Scores are normalized to [0, 1] by dividing by the maximum count.

    Args:
        sentences: Lecture transcript sentences.
        concepts: Concept terms to score.
        threshold: Cosine similarity threshold for counting (default 0.65).
        model_name: Embedding model name (uses default if None).

    Returns:
        InstructionalEmphasisMap with per-concept emphasis scores.
    """
    if not sentences or not concepts:
        return InstructionalEmphasisMap(
            concept_scores={},
            threshold_used=threshold,
            n_sentences=len(sentences),
            n_concepts=len(concepts),
        )

    # Encode all texts in a single batch
    all_texts = sentences + concepts
    encode_kwargs = {"texts": all_texts}
    if model_name is not None:
        encode_kwargs["model_name"] = model_name
    all_embeddings = encode_texts(**encode_kwargs)

    n_sent = len(sentences)
    sentence_embs = all_embeddings[:n_sent]
    concept_embs = all_embeddings[n_sent:]

    # Normalize for cosine similarity
    sent_norms = np.linalg.norm(sentence_embs, axis=1, keepdims=True)
    sent_norms = np.where(sent_norms == 0, 1.0, sent_norms)
    sentence_embs_normed = sentence_embs / sent_norms

    conc_norms = np.linalg.norm(concept_embs, axis=1, keepdims=True)
    conc_norms = np.where(conc_norms == 0, 1.0, conc_norms)
    concept_embs_normed = concept_embs / conc_norms

    # Cosine similarity matrix: (n_concepts, n_sentences)
    sim_matrix = concept_embs_normed @ sentence_embs_normed.T

    # FR-009: count sentences with cosine similarity >= threshold per concept
    above_threshold = (sim_matrix >= threshold).astype(np.float64)
    hit_counts = above_threshold.sum(axis=1)  # shape: (n_concepts,)

    # Normalize by max count → [0, 1]
    max_count = float(hit_counts.max())
    if max_count > 0:
        normalized = hit_counts / max_count
    else:
        normalized = np.zeros_like(hit_counts)

    concept_scores = {
        concept: float(np.clip(score, 0.0, 1.0))
        for concept, score in zip(concepts, normalized)
    }

    return InstructionalEmphasisMap(
        concept_scores=concept_scores,
        threshold_used=threshold,
        n_sentences=n_sent,
        n_concepts=len(concepts),
    )


def compute_weighted_concept_coverage(
    emphasis_map: InstructionalEmphasisMap,
    mastery: dict[str, float],
) -> float:
    """Compute emphasis-weighted concept coverage.

    Formula: sum(e_i * m_i) / sum(e_i) where e_i is the emphasis weight
    and m_i is the mastery indicator (0 or 1) for concept i.

    Args:
        emphasis_map: Instructional emphasis map with concept weights.
        mastery: Mapping of concept name to mastery score (0.0 or 1.0).

    Returns:
        Weighted coverage in [0, 1], or 0.0 if no emphasis weights.
    """
    if not emphasis_map.concept_scores:
        return 0.0

    total_weight = sum(emphasis_map.concept_scores.values())
    if total_weight <= 0:
        return 0.0

    weighted_sum = sum(
        weight * mastery.get(concept, 0.0)
        for concept, weight in emphasis_map.concept_scores.items()
    )

    return weighted_sum / total_weight
