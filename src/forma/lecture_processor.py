"""Lecture transcript processor for extracting concepts and triplets.

Processes lecture transcripts to extract knowledge triplets and determine
which master concepts are covered, supporting the graph-based evaluation
pipeline.
"""

from __future__ import annotations

import json
import re

import numpy as np

from forma.embedding_cache import encode_texts
from forma.evaluation_types import TripletEdge
from forma.llm_provider import LLMProvider
from forma.prompt_templates import render_lecture_triplet_prompt

MAX_TRANSCRIPT_LENGTH: int = 10_000


def load_transcript(path: str) -> str:
    """Load a UTF-8 text file and validate length.

    Args:
        path: Path to the transcript file.

    Returns:
        Transcript text content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the transcript exceeds MAX_TRANSCRIPT_LENGTH chars.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if len(text) > MAX_TRANSCRIPT_LENGTH:
        raise ValueError(f"Transcript length {len(text)} exceeds maximum {MAX_TRANSCRIPT_LENGTH} characters.")
    return text


def segment_text(text: str, max_chars: int = 1000) -> list[str]:
    """Split text into segments at sentence boundaries.

    Splits on Korean/standard sentence-ending punctuation (., !, ?).
    Each segment is at most max_chars characters. Never splits mid-sentence.

    Args:
        text: Input text to segment.
        max_chars: Maximum characters per segment.

    Returns:
        List of text segments.
    """
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s*", text)
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return [text]

    segments: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                segments.append(current)
            current = sentence

    if current:
        segments.append(current)

    return segments


def extract_triplets_from_lecture(text: str, provider: LLMProvider) -> list[TripletEdge]:
    """Use LLM to extract triplets from lecture text.

    Args:
        text: Lecture text content.
        provider: LLM provider instance.

    Returns:
        List of TripletEdge objects extracted from the lecture.
    """
    prompt = render_lecture_triplet_prompt(text)
    response = provider.generate(prompt, max_tokens=1024, temperature=0.0)

    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        raw = response.strip()

    parsed = json.loads(raw)
    triplets: list[TripletEdge] = []
    for item in parsed:
        triplets.append(
            TripletEdge(
                subject=item["subject"],
                relation=item["relation"],
                object=item["object"],
            )
        )
    return triplets


def extract_lecture_covered_concepts(
    lecture_text: str,
    master_concepts: list[str],
    threshold: float = 0.75,
) -> list[str]:
    """Find which master concepts are covered in the lecture text.

    Uses embedding similarity between lecture text and each master concept
    to determine coverage.

    Args:
        lecture_text: The lecture transcript text.
        master_concepts: List of concept strings to check against.
        threshold: Minimum cosine similarity to consider a concept covered.

    Returns:
        List of master concepts that are covered in the lecture.
    """
    if not master_concepts:
        return []

    lecture_embeddings = encode_texts([lecture_text])
    concept_embeddings = encode_texts(master_concepts)

    lecture_vec = lecture_embeddings[0]
    lecture_norm = np.linalg.norm(lecture_vec)
    if lecture_norm == 0:
        return []

    covered: list[str] = []
    for i, concept in enumerate(master_concepts):
        concept_vec = concept_embeddings[i]
        concept_norm = np.linalg.norm(concept_vec)
        if concept_norm == 0:
            continue
        similarity = float(np.dot(lecture_vec, concept_vec) / (lecture_norm * concept_norm))
        if similarity >= threshold:
            covered.append(concept)

    return covered


def extract_lecture_tone_sample(text: str, max_chars: int = 500) -> str:
    """Extract first max_chars of text at sentence boundary for tone reference.

    Args:
        text: Input lecture text.
        max_chars: Maximum characters to extract.

    Returns:
        Truncated text ending at a sentence boundary.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_boundary = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )

    if last_boundary > 0:
        return truncated[: last_boundary + 1]

    return truncated
