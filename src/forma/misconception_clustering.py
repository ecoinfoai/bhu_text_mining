"""Misconception clustering via embedding-based KMeans.

Groups ClassifiedMisconception instances into clusters using text
embeddings and scikit-learn KMeans, producing MisconceptionCluster
summaries with representative errors and optional LLM correction points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forma.evaluation_types import TripletEdge

from collections import Counter

import numpy as np

from forma.embedding_cache import encode_texts
from forma.misconception_classifier import ClassifiedMisconception, MisconceptionPattern

__all__ = [
    "MisconceptionCluster",
    "cluster_misconceptions",
]

logger = logging.getLogger(__name__)


@dataclass
class MisconceptionCluster:
    """A cluster of similar misconceptions grouped by embedding similarity.

    Args:
        cluster_id: Unique cluster identifier within the question scope.
            -1 is reserved for the 'OTHER' group (merged small clusters).
        pattern: Majority MisconceptionPattern among cluster members.
        representative_error: Misconception description closest to cluster
            centroid (embedding space).
        member_count: Number of misconceptions in this cluster.
        student_errors: All misconception descriptions in this cluster.
        correction_point: LLM-generated correction text. Empty string ""
            means not yet generated or generation failed. Never None.
        centroid_edge: Master edge corresponding to representative_error.
            None for CONCEPT_ABSENCE pattern.
    """

    cluster_id: int
    pattern: MisconceptionPattern
    representative_error: str
    member_count: int
    student_errors: list[str] = field(default_factory=list)
    correction_point: str = ""
    centroid_edge: TripletEdge | None = None


def cluster_misconceptions(
    classified: list[ClassifiedMisconception],
    n_clusters: int = 5,
    min_cluster_size: int = 2,
) -> list[MisconceptionCluster]:
    """Cluster classified misconceptions using embedding-based KMeans.

    Groups ClassifiedMisconception instances into clusters by computing text
    embeddings and applying KMeans clustering. Small clusters (below
    min_cluster_size) are merged into an OTHER group (cluster_id=-1).

    Args:
        classified: List of ClassifiedMisconception instances to cluster.
            Empty list returns [].
        n_clusters: Maximum number of clusters. Actual count is
            min(n_clusters, len(classified)).
        min_cluster_size: Clusters with fewer members are merged into
            the OTHER group (cluster_id=-1).

    Returns:
        List of MisconceptionCluster sorted by member_count descending.
    """
    if not classified:
        return []

    # Build embedding texts
    texts: list[str] = []
    for m in classified:
        if m.master_edge is not None:
            edge_str = f"{m.master_edge.subject} {m.master_edge.relation} {m.master_edge.object}"
            texts.append(f"{edge_str} -> {m.description}")
        else:
            texts.append(m.description)

    # Encode texts to embeddings
    embeddings = encode_texts(texts)

    # Fit KMeans
    k = min(n_clusters, len(classified))
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # Group items by cluster label
    cluster_groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        cluster_groups.setdefault(int(label), []).append(idx)

    # Build clusters
    regular_clusters: list[MisconceptionCluster] = []
    other_items: list[int] = []

    for label, member_indices in sorted(cluster_groups.items()):
        if len(member_indices) < min_cluster_size:
            other_items.extend(member_indices)
            continue

        # Find representative: closest to centroid (argmin L2 distance)
        centroid = centroids[label]
        member_embeddings = embeddings[member_indices]
        distances = np.linalg.norm(member_embeddings - centroid, axis=1)
        rep_local_idx = int(np.argmin(distances))
        rep_idx = member_indices[rep_local_idx]

        # Majority pattern
        pattern_counts: Counter[MisconceptionPattern] = Counter()
        for idx in member_indices:
            pattern_counts[classified[idx].pattern] += 1
        majority_pattern = pattern_counts.most_common(1)[0][0]

        # Student errors
        student_errors = [classified[idx].description for idx in member_indices]

        # Centroid edge from representative
        centroid_edge = classified[rep_idx].master_edge

        regular_clusters.append(
            MisconceptionCluster(
                cluster_id=label,
                pattern=majority_pattern,
                representative_error=classified[rep_idx].description,
                member_count=len(member_indices),
                student_errors=student_errors,
                correction_point="",
                centroid_edge=centroid_edge,
            )
        )

    # Merge small clusters into OTHER (cluster_id=-1)
    if other_items:
        pattern_counts = Counter()
        for idx in other_items:
            pattern_counts[classified[idx].pattern] += 1
        majority_pattern = pattern_counts.most_common(1)[0][0]

        # Representative: first item in OTHER group
        rep_idx = other_items[0]
        student_errors = [classified[idx].description for idx in other_items]
        centroid_edge = classified[rep_idx].master_edge

        regular_clusters.append(
            MisconceptionCluster(
                cluster_id=-1,
                pattern=majority_pattern,
                representative_error=classified[rep_idx].description,
                member_count=len(other_items),
                student_errors=student_errors,
                correction_point="",
                centroid_edge=centroid_edge,
            )
        )

    # Sort by member_count descending
    regular_clusters.sort(key=lambda c: c.member_count, reverse=True)

    return regular_clusters
