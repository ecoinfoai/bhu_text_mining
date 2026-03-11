"""Directed triplet graph comparison between student and master graphs.

Supports fuzzy edge matching via embedding similarity, node aliases,
and lecture coverage exclusion.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from forma.embedding_cache import encode_texts
from forma.evaluation_types import GraphComparisonResult, TripletEdge

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD: float = 0.80


class GraphComparator:
    """Compare student triplet graph against master knowledge graph.

    Supports fuzzy edge matching using embedding similarity, node
    alias resolution, and lecture coverage exclusion.

    Args:
        similarity_threshold: Minimum cosine similarity for fuzzy
            edge matching (default 0.80).
        node_aliases: Dict mapping canonical node name to list of
            alternative representations.
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        node_aliases: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self._threshold = similarity_threshold
        self._aliases = node_aliases or {}
        self._alias_map = self._build_alias_map()

    def _build_alias_map(self) -> dict[str, str]:
        """Build reverse mapping: alias → canonical name."""
        mapping: dict[str, str] = {}
        for canonical, aliases in self._aliases.items():
            mapping[canonical.lower()] = canonical
            for alias in aliases:
                mapping[alias.lower()] = canonical
        return mapping

    def _normalize_node(self, node: str) -> str:
        """Normalize a node name using alias mapping."""
        return self._alias_map.get(node.lower(), node)

    def _normalize_edge(self, edge: TripletEdge) -> TripletEdge:
        """Normalize edge nodes using alias mapping."""
        return TripletEdge(
            subject=self._normalize_node(edge.subject),
            relation=edge.relation,
            object=self._normalize_node(edge.object),
        )

    def compare(
        self,
        student_id: str,
        question_sn: int,
        master_edges: list[TripletEdge],
        student_edges: list[TripletEdge],
        lecture_covered_concepts: Optional[list[str]] = None,
    ) -> GraphComparisonResult:
        """Compare student edges against master edges.

        Args:
            student_id: Student identifier.
            question_sn: Question serial number.
            master_edges: Master knowledge graph edges.
            student_edges: Student-extracted edges.
            lecture_covered_concepts: Concepts covered in lecture.
                Master edges involving uncovered concepts are excluded
                from P/R/F1 computation.

        Returns:
            GraphComparisonResult with P/R/F1 and edge classifications.
        """
        # Normalize all edges
        master_norm = [self._normalize_edge(e) for e in master_edges]
        student_norm = [self._normalize_edge(e) for e in student_edges]

        # Exclude master edges not covered in lecture
        lecture_excluded: list[TripletEdge] = []
        effective_master: list[TripletEdge] = []
        if lecture_covered_concepts is not None:
            covered_lower = {c.lower() for c in lecture_covered_concepts}
            for e in master_norm:
                subj_covered = e.subject.lower() in covered_lower
                obj_covered = e.object.lower() in covered_lower
                if subj_covered or obj_covered:
                    effective_master.append(e)
                else:
                    lecture_excluded.append(e)
        else:
            effective_master = list(master_norm)

        if not effective_master and not student_norm:
            return GraphComparisonResult(
                student_id=student_id,
                question_sn=question_sn,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                matched_edges=[],
                missing_edges=[],
                extra_edges=[],
                lecture_excluded_edges=lecture_excluded,
            )

        # Compute edge matching
        matched, missing, extra, wrong_dir, fuzzy_count = self._match_edges(
            effective_master, student_norm
        )

        # Compute P/R/F1
        n_matched = len(matched)
        n_student = len(student_norm)
        n_master = len(effective_master)

        precision = n_matched / n_student if n_student > 0 else 0.0
        recall = n_matched / n_master if n_master > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return GraphComparisonResult(
            student_id=student_id,
            question_sn=question_sn,
            precision=precision,
            recall=recall,
            f1=f1,
            matched_edges=matched,
            missing_edges=missing,
            extra_edges=extra,
            wrong_direction_edges=wrong_dir,
            lecture_excluded_edges=lecture_excluded,
            fuzzy_matched=fuzzy_count,
        )

    def _match_edges(
        self,
        master_edges: list[TripletEdge],
        student_edges: list[TripletEdge],
    ) -> tuple[
        list[TripletEdge],
        list[TripletEdge],
        list[TripletEdge],
        list[TripletEdge],
        int,
    ]:
        """Match student edges to master edges.

        Returns:
            (matched, missing, extra, wrong_direction, fuzzy_count)
        """
        if not master_edges:
            return [], [], list(student_edges), [], 0
        if not student_edges:
            return [], list(master_edges), [], [], 0

        # Encode all edge strings for fuzzy matching
        master_strs = [
            f"{e.subject} {e.relation} {e.object}" for e in master_edges
        ]
        student_strs = [
            f"{e.subject} {e.relation} {e.object}" for e in student_edges
        ]
        # Also encode reversed student edges for direction detection
        student_reversed_strs = [
            f"{e.object} {e.relation} {e.subject}" for e in student_edges
        ]

        try:
            all_texts = master_strs + student_strs + student_reversed_strs
            all_embs = encode_texts(all_texts)
            n_m = len(master_strs)
            n_s = len(student_strs)
            master_embs = all_embs[:n_m]
            student_embs = all_embs[n_m : n_m + n_s]
            reversed_embs = all_embs[n_m + n_s :]
        except Exception:
            return self._exact_match_edges(master_edges, student_edges)

        # Compute similarity matrices
        sim_forward = cosine_similarity(student_embs, master_embs)
        sim_reversed = cosine_similarity(reversed_embs, master_embs)

        matched_master_idx: set[int] = set()
        matched_student_idx: set[int] = set()
        wrong_direction: list[TripletEdge] = []
        fuzzy_count = 0

        # First pass: find best matches
        for s_idx in range(len(student_edges)):
            best_m_idx = int(np.argmax(sim_forward[s_idx]))
            best_sim = sim_forward[s_idx, best_m_idx]

            if best_sim >= self._threshold and best_m_idx not in matched_master_idx:
                matched_master_idx.add(best_m_idx)
                matched_student_idx.add(s_idx)
                # Check if it was a fuzzy match (not exact)
                if best_sim < 0.99:
                    fuzzy_count += 1
                continue

            # Check for wrong direction
            best_rev_m_idx = int(np.argmax(sim_reversed[s_idx]))
            best_rev_sim = sim_reversed[s_idx, best_rev_m_idx]
            if (
                best_rev_sim >= self._threshold
                and best_rev_m_idx not in matched_master_idx
            ):
                wrong_direction.append(student_edges[s_idx])
                matched_student_idx.add(s_idx)

        matched = [
            student_edges[i]
            for i in sorted(matched_student_idx - {
                i for i, e in enumerate(student_edges)
                if e in wrong_direction
            })
        ]
        missing = [
            master_edges[i]
            for i in range(len(master_edges))
            if i not in matched_master_idx
        ]
        extra = [
            student_edges[i]
            for i in range(len(student_edges))
            if i not in matched_student_idx
        ]

        return matched, missing, extra, wrong_direction, fuzzy_count

    def _exact_match_edges(
        self,
        master_edges: list[TripletEdge],
        student_edges: list[TripletEdge],
    ) -> tuple[
        list[TripletEdge],
        list[TripletEdge],
        list[TripletEdge],
        list[TripletEdge],
        int,
    ]:
        """Fallback exact string matching for edges."""
        master_keys = {
            (e.subject, e.relation, e.object): e for e in master_edges
        }
        _master_reversed = {
            (e.object, e.relation, e.subject): e for e in master_edges
        }

        matched: list[TripletEdge] = []
        extra: list[TripletEdge] = []
        wrong_dir: list[TripletEdge] = []
        used_master: set[tuple[str, str, str]] = set()

        for se in student_edges:
            key = (se.subject, se.relation, se.object)
            rev_key = (se.object, se.relation, se.subject)
            if key in master_keys and key not in used_master:
                matched.append(se)
                used_master.add(key)
            elif rev_key in master_keys and rev_key not in used_master:
                wrong_dir.append(se)
            else:
                extra.append(se)

        missing = [
            e
            for e in master_edges
            if (e.subject, e.relation, e.object) not in used_master
        ]

        return matched, missing, extra, wrong_dir, 0
