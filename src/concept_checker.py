"""Layer 1: Concept-presence detection via semantic similarity.

Uses top-k mean cosine similarity (not max-pooling) to reduce false
positives, and an adaptive threshold τ(M) = τ_base + α·log(M) that
penalises questions with many concepts.

Korean particles are selected correctly based on whether the concept
term ends with a consonant (받침 있음 → 은/이/을) or vowel (→ 는/가/를).
"""

from __future__ import annotations

import math
import unicodedata

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import kss

from src.embedding_cache import encode_texts, DEFAULT_MODEL
from src.evaluation_types import ConceptMatchResult


# ---------------------------------------------------------------------------
# Korean particle utilities
# ---------------------------------------------------------------------------


def _select_particle(
    word: str, with_final: str, without_final: str
) -> str:
    """Select Korean post-positional particle based on final consonant.

    Korean particles alternate between two forms depending on whether
    the preceding word ends with a consonant (받침) or vowel.

    Args:
        word: Korean word (or empty string / ASCII).
        with_final: Particle when word ends with consonant (e.g. "은").
        without_final: Particle when word ends with vowel (e.g. "는").

    Returns:
        Appropriate particle string.

    Examples:
        >>> _select_particle("세포막", "은", "는")
        '은'
        >>> _select_particle("세포", "은", "는")
        '는'
    """
    if not word:
        return without_final
    last_char = word[-1]
    code = ord(last_char)
    if 0xAC00 <= code <= 0xD7A3:
        jongseong = (code - 0xAC00) % 28
        return with_final if jongseong != 0 else without_final
    return without_final


def build_concept_template(concept: str) -> str:
    """Build a concept-in-context sentence with correct Korean particle.

    Creates "X은/는 중요한 개념이다" choosing the particle based on
    whether the last character of ``concept`` has a final consonant.

    Args:
        concept: Concept term (Korean or mixed Korean/ASCII).

    Returns:
        Context sentence string.

    Examples:
        >>> build_concept_template("세포막")
        '세포막은 중요한 개념이다'
        >>> build_concept_template("세포")
        '세포는 중요한 개념이다'
    """
    particle = _select_particle(concept, "은", "는")
    return f"{concept}{particle} 중요한 개념이다"


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


def split_student_response(text: str) -> list[str]:
    """Split student response into sentences using KSS, with fallback.

    If KSS raises or returns an empty list, the entire text is treated
    as a single sentence (fail-safe fallback).

    Args:
        text: Raw student response text.

    Returns:
        List of sentence strings (at least one element).

    Examples:
        >>> split_student_response("Hello. World.")
        ['Hello.', 'World.']
    """
    try:
        sentences = kss.split_sentences(text)
        return sentences if sentences else [text]
    except Exception:
        return [text]


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_top_k_mean_similarity(
    sentence_embeddings: np.ndarray,
    concept_embedding: np.ndarray,
    k: int,
) -> float:
    """Compute the top-k mean cosine similarity between sentences and concept.

    Selects the k highest sentence-concept similarities and averages them.
    k is clamped to the number of available sentences.

    Args:
        sentence_embeddings: Float array of shape (n_sentences, embed_dim).
        concept_embedding: Float array of shape (embed_dim,) or (1, embed_dim).
        k: Number of top similarities to average.

    Returns:
        Mean cosine similarity of the top-k sentences (float in [-1, 1]).

    Examples:
        >>> import numpy as np
        >>> s = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> c = np.array([1.0, 0.0])
        >>> compute_top_k_mean_similarity(s, c, k=1)
        1.0
    """
    concept_emb = concept_embedding.reshape(1, -1)
    sims = cosine_similarity(sentence_embeddings, concept_emb).flatten()
    k_actual = min(k, len(sims))
    top_k_sims = np.sort(sims)[-k_actual:]
    return float(np.mean(top_k_sims))


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------


def adaptive_threshold(
    base_threshold: float,
    n_concepts: int,
    alpha: float = 0.02,
) -> float:
    """Compute adaptive similarity threshold based on concept count.

    Formula: τ(M) = τ_base + α · log(M)

    Raising the threshold with more concepts reduces false positives when
    a question covers many closely related terms.

    Args:
        base_threshold: Base threshold τ_base (cold-start default 0.45).
        n_concepts: Number of concepts M in the question (must be > 0).
        alpha: Scaling coefficient (default 0.02; calibrate from pilot).

    Returns:
        Adjusted threshold value.

    Raises:
        ValueError: If n_concepts is not positive.

    Examples:
        >>> import math
        >>> adaptive_threshold(0.45, 5, alpha=0.02)  # doctest: +ELLIPSIS
        0.482...
    """
    if n_concepts <= 0:
        raise ValueError(
            f"n_concepts must be a positive integer, got {n_concepts} "
            "in adaptive_threshold(). "
            "Pass the total number of concepts for the question."
        )
    return base_threshold + alpha * math.log(n_concepts)


# ---------------------------------------------------------------------------
# Main concept-checking functions
# ---------------------------------------------------------------------------


def check_concept_presence(
    student_text: str,
    concept: str,
    n_concepts: int,
    base_threshold: float = 0.45,
    k: int = 2,
    alpha: float = 0.02,
    model_name: str = DEFAULT_MODEL,
) -> ConceptMatchResult:
    """Check whether a single concept is present in a student response.

    Pipeline:
    1. Split ``student_text`` into sentences via KSS (with fallback).
    2. Build concept-in-context template with correct Korean particle.
    3. Encode sentences + template with the cached encoder.
    4. Compute top-k mean cosine similarity.
    5. Compare against adaptive threshold τ(M).

    The returned ConceptMatchResult has ``student_id=""`` and
    ``question_sn=0``; callers should set these before storage.

    Args:
        student_text: Raw student answer text (non-empty).
        concept: Concept term to detect (non-empty).
        n_concepts: Total number of concepts in the question.
        base_threshold: Base cosine similarity threshold (default 0.45).
        k: Top-k sentences to average (default 2 = min(2, n_sentences)).
        alpha: Adaptive threshold scaling factor (default 0.02).
        model_name: Sentence encoder model name.

    Returns:
        ConceptMatchResult for this (student_text, concept) pair.

    Raises:
        ValueError: If ``student_text`` or ``concept`` is empty.

    Examples:
        >>> # (requires ko-sroberta-multitask; mock in tests)
        >>> result = check_concept_presence("세포막은 인지질 이중층", "세포막", 5)
        >>> isinstance(result, ConceptMatchResult)
        True
    """
    if not student_text or not student_text.strip():
        raise ValueError(
            "student_text is empty in check_concept_presence(). "
            "Provide a non-empty student response string."
        )
    if not concept or not concept.strip():
        raise ValueError(
            "concept is empty in check_concept_presence(). "
            "Provide a non-empty concept term."
        )

    sentences = split_student_response(student_text)
    concept_template = build_concept_template(concept)

    all_texts = sentences + [concept_template]
    all_embeddings = encode_texts(all_texts, model_name=model_name)

    sentence_embeddings = all_embeddings[:-1]
    concept_embedding = all_embeddings[-1]

    k_actual = min(k, len(sentences))
    top_k_sim = compute_top_k_mean_similarity(
        sentence_embeddings, concept_embedding, k=k_actual
    )

    threshold = adaptive_threshold(base_threshold, n_concepts, alpha)
    is_present = top_k_sim >= threshold

    return ConceptMatchResult(
        concept=concept,
        student_id="",
        question_sn=0,
        is_present=is_present,
        similarity_score=top_k_sim,
        top_k_mean_similarity=top_k_sim,
        threshold_used=threshold,
        method="top_k_mean",
    )


def check_all_concepts(
    student_text: str,
    student_id: str,
    question_sn: int,
    concepts: list[str],
    base_threshold: float = 0.45,
    k: int = 2,
    alpha: float = 0.02,
    model_name: str = DEFAULT_MODEL,
) -> list[ConceptMatchResult]:
    """Check presence of all concepts for one student response.

    Calls check_concept_presence for each concept and sets the
    ``student_id`` and ``question_sn`` fields on every result.

    Args:
        student_text: Student answer text.
        student_id: Student identifier to stamp on results.
        question_sn: Question serial number to stamp on results.
        concepts: List of concept terms to check.
        base_threshold: Base cosine similarity threshold.
        k: Top-k sentences for similarity averaging.
        alpha: Adaptive threshold scaling factor.
        model_name: Sentence encoder model.

    Returns:
        List of ConceptMatchResult, one per concept in order.

    Examples:
        >>> results = check_all_concepts(
        ...     "세포막은 인지질 이중층입니다.",
        ...     "s001", 1, ["세포막", "인지질"],
        ... )
        >>> len(results)
        2
    """
    results: list[ConceptMatchResult] = []
    n = len(concepts)
    for concept in concepts:
        r = check_concept_presence(
            student_text=student_text,
            concept=concept,
            n_concepts=n,
            base_threshold=base_threshold,
            k=k,
            alpha=alpha,
            model_name=model_name,
        )
        r.student_id = student_id
        r.question_sn = question_sn
        results.append(r)
    return results
