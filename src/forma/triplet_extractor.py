"""LLM-based triplet extraction from student responses.

Uses a 3-call protocol with majority consensus to extract reliable
(subject, relation, object) triplets from essay answers.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from forma.embedding_cache import encode_texts
from forma.evaluation_types import TripletEdge, TripletExtractionResult
from forma.llm_provider import LLMProvider
from forma.prompt_templates import render_triplet_extraction_prompt

logger = logging.getLogger(__name__)

DEFAULT_CONSENSUS_THRESHOLD: float = 0.85
N_CALLS: int = 3


class TripletExtractor:
    """Extract knowledge triplets from student responses via LLM.

    Uses a 3-call protocol: each call independently extracts triplets,
    then majority consensus (2 of 3 agree) determines final set.

    Args:
        provider: LLM provider instance.
        consensus_threshold: Embedding similarity threshold for triplet
            matching across calls (default 0.85).
        n_calls: Number of independent extraction calls (default 3).
    """

    def __init__(
        self,
        provider: LLMProvider,
        consensus_threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
        n_calls: int = N_CALLS,
    ) -> None:
        self._provider = provider
        self._threshold = consensus_threshold
        self._n_calls = n_calls

    def extract(
        self,
        student_id: str,
        question_sn: int,
        question: str,
        student_response: str,
        master_nodes: list[str],
    ) -> TripletExtractionResult:
        """Extract triplets from a student response using multi-call consensus.

        Args:
            student_id: Student identifier.
            question_sn: Question serial number.
            question: The exam question text.
            student_response: Student's essay answer.
            master_nodes: List of master concept node names.

        Returns:
            TripletExtractionResult with consensus triplets.
        """
        if not student_response or not student_response.strip():
            return TripletExtractionResult(
                student_id=student_id,
                question_sn=question_sn,
                triplets=[],
                call_results=[[] for _ in range(self._n_calls)],
            )

        call_results: list[list[TripletEdge]] = []
        for _ in range(self._n_calls):
            triplets = self._single_extraction(
                question, student_response, master_nodes
            )
            call_results.append(triplets)

        consensus = self._compute_consensus(call_results)

        return TripletExtractionResult(
            student_id=student_id,
            question_sn=question_sn,
            triplets=consensus,
            call_results=call_results,
        )

    def _single_extraction(
        self,
        question: str,
        student_response: str,
        master_nodes: list[str],
    ) -> list[TripletEdge]:
        """Run a single LLM extraction call.

        Returns empty list on parse failure (graceful degradation).
        """
        prompt = render_triplet_extraction_prompt(
            question=question,
            student_response=student_response,
            master_nodes=master_nodes,
        )
        try:
            raw = self._provider.generate(prompt, max_tokens=1024, temperature=0.3)
            return self._parse_triplets(raw)
        except Exception as exc:
            logger.warning("Triplet extraction call failed: %s", exc)
            return []

    def _parse_triplets(self, raw_response: str) -> list[TripletEdge]:
        """Parse JSON triplet array from LLM response."""
        text = raw_response.strip()

        # Extract JSON from markdown code block if present
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse triplet JSON: %.100s", text)
            return []

        if not isinstance(data, list):
            return []

        triplets: list[TripletEdge] = []
        for item in data:
            if (
                isinstance(item, dict)
                and "subject" in item
                and "relation" in item
                and "object" in item
            ):
                triplets.append(
                    TripletEdge(
                        subject=str(item["subject"]),
                        relation=str(item["relation"]),
                        object=str(item["object"]),
                    )
                )
        return triplets

    def _compute_consensus(
        self, call_results: list[list[TripletEdge]]
    ) -> list[TripletEdge]:
        """Compute majority consensus across extraction calls.

        A triplet is included if it appears (by embedding similarity)
        in at least ceil(n_calls/2) of the calls.
        """
        if not any(call_results):
            return []

        # Flatten all triplets with call index
        all_triplets: list[tuple[int, TripletEdge]] = []
        for call_idx, triplets in enumerate(call_results):
            for t in triplets:
                all_triplets.append((call_idx, t))

        if not all_triplets:
            return []

        # Encode triplet strings for similarity comparison
        triplet_strs = [
            f"{t.subject} {t.relation} {t.object}" for _, t in all_triplets
        ]

        try:
            embeddings = encode_texts(triplet_strs)
        except Exception:
            # Fallback: exact string matching
            return self._exact_consensus(call_results)

        sim_matrix = cosine_similarity(embeddings)
        min_votes = (self._n_calls + 1) // 2  # ceil(n/2)

        used = set()
        consensus: list[TripletEdge] = []

        for i, (call_i, triplet_i) in enumerate(all_triplets):
            if i in used:
                continue

            # Find similar triplets from OTHER calls
            voting_calls = {call_i}
            for j, (call_j, _) in enumerate(all_triplets):
                if j == i or j in used or call_j == call_i:
                    continue
                if (
                    sim_matrix[i, j] >= self._threshold
                    and call_j not in voting_calls
                ):
                    voting_calls.add(call_j)
                    used.add(j)

            if len(voting_calls) >= min_votes:
                consensus.append(triplet_i)
                used.add(i)

        return consensus

    def _exact_consensus(
        self, call_results: list[list[TripletEdge]]
    ) -> list[TripletEdge]:
        """Fallback consensus using exact string matching."""
        min_votes = (self._n_calls + 1) // 2
        key_counts: dict[str, tuple[int, TripletEdge]] = {}

        for triplets in call_results:
            seen_keys: set[str] = set()
            for t in triplets:
                key = f"{t.subject}|{t.relation}|{t.object}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    if key in key_counts:
                        count, edge = key_counts[key]
                        key_counts[key] = (count + 1, edge)
                    else:
                        key_counts[key] = (1, t)

        return [
            edge for count, edge in key_counts.values() if count >= min_votes
        ]
