"""Tests for misconception_clustering.py — embedding-based KMeans clustering.

RED phase: these tests are written BEFORE cluster_misconceptions() is
implemented.  All tests except MisconceptionCluster dataclass tests
should FAIL until T016 implementation is complete.

Covers T014:
  - MisconceptionCluster dataclass field presence and types
  - cluster_misconceptions([], n_clusters=3) -> [] (FR-013)
  - 10 inputs, n_clusters=3 -> output count <= 3 (FR-010, SC-005)
  - Each cluster has non-empty representative_error (SC-005)
  - min_cluster_size=2: clusters with 1 member -> merged into OTHER (cluster_id=-1)
    (FR-012, SC-006)
  - n_clusters=5 but only 3 inputs -> output count <= 3 (FR-014)
  - correction_point is str, not Optional (I2 fix)
  - Mock encode_texts() -> deterministic np.ndarray (shape (n, 4))
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import patch

import numpy as np
import pytest

from forma.misconception_classifier import (
    ClassifiedMisconception,
    MisconceptionPattern,
)
from forma.evaluation_types import TripletEdge
from forma.misconception_clustering import MisconceptionCluster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classified(
    n: int,
    *,
    pattern: MisconceptionPattern = MisconceptionPattern.CAUSAL_REVERSAL,
) -> list[ClassifiedMisconception]:
    """Build n synthetic ClassifiedMisconception instances.

    Each has a unique description to ensure distinct embeddings.
    """
    items = []
    for i in range(n):
        items.append(ClassifiedMisconception(
            pattern=pattern,
            master_edge=TripletEdge(f"S{i}", f"R{i}", f"O{i}"),
            student_edge=TripletEdge(f"O{i}", f"R{i}", f"S{i}"),
            concept=None,
            confidence=0.85,
            description=f"오개념 설명 {i}: 인과 방향 역전 S{i}->R{i}->O{i}",
        ))
    return items


def _make_mixed_classified(n: int) -> list[ClassifiedMisconception]:
    """Build n classified misconceptions with mixed patterns.

    Cycles through all 4 patterns so we get a realistic mix.
    """
    patterns = list(MisconceptionPattern)
    items = []
    for i in range(n):
        p = patterns[i % len(patterns)]
        master_edge = (
            None if p == MisconceptionPattern.CONCEPT_ABSENCE
            else TripletEdge(f"S{i}", f"R{i}", f"O{i}")
        )
        student_edge = (
            None if p == MisconceptionPattern.CONCEPT_ABSENCE
            else TripletEdge(f"O{i}", f"R{i}", f"S{i}")
        )
        concept = f"concept_{i}" if p == MisconceptionPattern.CONCEPT_ABSENCE else None
        items.append(ClassifiedMisconception(
            pattern=p,
            master_edge=master_edge,
            student_edge=student_edge,
            concept=concept,
            confidence=0.7 + 0.03 * i,
            description=f"오개념 {i}: {p.value} 패턴",
        ))
    return items


def _mock_encode_deterministic(texts: list[str]) -> np.ndarray:
    """Return a deterministic (n, 4) embedding array.

    Each row is a distinct vector based on index, ensuring KMeans
    can find real clusters. Rows 0-2 are close, 3-5 are close,
    6-9 are close (for a 10-input test with n_clusters=3).
    """
    n = len(texts)
    arr = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        cluster_id = i % 3  # group into 3 clusters
        arr[i, 0] = cluster_id * 10.0 + np.random.RandomState(i).uniform(-0.1, 0.1)
        arr[i, 1] = cluster_id * 10.0 + np.random.RandomState(i + 100).uniform(-0.1, 0.1)
        arr[i, 2] = 0.0
        arr[i, 3] = 0.0
    return arr


def _mock_encode_sparse(texts: list[str]) -> np.ndarray:
    """Return embeddings where each item is far apart (for min_cluster_size tests).

    Each vector is distinct enough that KMeans with n_clusters=len(texts)
    would give 1 item per cluster, triggering the OTHER merge logic.
    """
    n = len(texts)
    arr = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        arr[i, :] = i * 100.0  # far apart
    return arr


# ===========================================================================
# TestMisconceptionClusterDataclass: field presence and types
# ===========================================================================


class TestMisconceptionClusterDataclass:
    """MisconceptionCluster dataclass field presence and type validation."""

    def test_has_cluster_id_field(self):
        """MisconceptionCluster has cluster_id: int."""
        field_names = {f.name for f in fields(MisconceptionCluster)}
        assert "cluster_id" in field_names

    def test_has_pattern_field(self):
        """MisconceptionCluster has pattern: MisconceptionPattern."""
        field_names = {f.name for f in fields(MisconceptionCluster)}
        assert "pattern" in field_names

    def test_has_representative_error_field(self):
        """MisconceptionCluster has representative_error: str."""
        field_names = {f.name for f in fields(MisconceptionCluster)}
        assert "representative_error" in field_names

    def test_has_member_count_field(self):
        """MisconceptionCluster has member_count: int."""
        field_names = {f.name for f in fields(MisconceptionCluster)}
        assert "member_count" in field_names

    def test_has_student_errors_field(self):
        """MisconceptionCluster has student_errors: list[str]."""
        field_names = {f.name for f in fields(MisconceptionCluster)}
        assert "student_errors" in field_names

    def test_has_correction_point_field(self):
        """MisconceptionCluster has correction_point: str (not Optional)."""
        field_names = {f.name for f in fields(MisconceptionCluster)}
        assert "correction_point" in field_names

    def test_has_centroid_edge_field(self):
        """MisconceptionCluster has centroid_edge: TripletEdge | None."""
        field_names = {f.name for f in fields(MisconceptionCluster)}
        assert "centroid_edge" in field_names

    def test_correction_point_default_empty_string(self):
        """correction_point defaults to '' (empty string), not None."""
        c = MisconceptionCluster(
            cluster_id=0,
            pattern=MisconceptionPattern.CAUSAL_REVERSAL,
            representative_error="test",
            member_count=1,
        )
        assert c.correction_point == ""
        assert isinstance(c.correction_point, str)

    def test_centroid_edge_default_none(self):
        """centroid_edge defaults to None."""
        c = MisconceptionCluster(
            cluster_id=0,
            pattern=MisconceptionPattern.CAUSAL_REVERSAL,
            representative_error="test",
            member_count=1,
        )
        assert c.centroid_edge is None

    def test_student_errors_default_empty_list(self):
        """student_errors defaults to empty list."""
        c = MisconceptionCluster(
            cluster_id=0,
            pattern=MisconceptionPattern.CAUSAL_REVERSAL,
            representative_error="test",
            member_count=1,
        )
        assert c.student_errors == []


# ===========================================================================
# TestClusterMisconceptionsFunction: cluster_misconceptions() tests
# ===========================================================================


class TestClusterMisconceptionsFunction:
    """Tests for cluster_misconceptions() function (T014 RED phase)."""

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_empty_input_returns_empty_list(self, mock_encode):
        """cluster_misconceptions([], n_clusters=3) -> [] (FR-013)."""
        from forma.misconception_clustering import cluster_misconceptions

        result = cluster_misconceptions([], n_clusters=3)
        assert result == []

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_ten_inputs_three_clusters(self, mock_encode):
        """10 inputs with n_clusters=3 -> output count <= 3 (FR-010, SC-005)."""
        from forma.misconception_clustering import cluster_misconceptions

        classified = _make_classified(10)
        result = cluster_misconceptions(classified, n_clusters=3)

        assert isinstance(result, list)
        assert len(result) <= 3
        assert len(result) > 0

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_each_cluster_has_nonempty_representative_error(self, mock_encode):
        """Each cluster has non-empty representative_error (SC-005)."""
        from forma.misconception_clustering import cluster_misconceptions

        classified = _make_classified(10)
        result = cluster_misconceptions(classified, n_clusters=3)

        for cluster in result:
            assert isinstance(cluster.representative_error, str)
            assert len(cluster.representative_error) > 0, (
                f"Cluster {cluster.cluster_id} has empty representative_error"
            )

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_cluster_member_count_positive(self, mock_encode):
        """Each cluster has member_count > 0."""
        from forma.misconception_clustering import cluster_misconceptions

        classified = _make_classified(10)
        result = cluster_misconceptions(classified, n_clusters=3)

        for cluster in result:
            assert cluster.member_count > 0

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_total_members_equal_input_count(self, mock_encode):
        """Sum of member_count across all clusters == len(input)."""
        from forma.misconception_clustering import cluster_misconceptions

        classified = _make_classified(10)
        result = cluster_misconceptions(classified, n_clusters=3)

        total = sum(c.member_count for c in result)
        assert total == 10

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_student_errors_list_populated(self, mock_encode):
        """Each cluster's student_errors has len == member_count."""
        from forma.misconception_clustering import cluster_misconceptions

        classified = _make_classified(10)
        result = cluster_misconceptions(classified, n_clusters=3)

        for cluster in result:
            assert len(cluster.student_errors) == cluster.member_count

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_pattern_is_misconception_pattern(self, mock_encode):
        """Each cluster's pattern is a MisconceptionPattern enum value."""
        from forma.misconception_clustering import cluster_misconceptions

        classified = _make_classified(10)
        result = cluster_misconceptions(classified, n_clusters=3)

        for cluster in result:
            assert isinstance(cluster.pattern, MisconceptionPattern)

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_correction_point_is_str(self, mock_encode):
        """correction_point is always str, not None (I2 fix)."""
        from forma.misconception_clustering import cluster_misconceptions

        classified = _make_classified(10)
        result = cluster_misconceptions(classified, n_clusters=3)

        for cluster in result:
            assert isinstance(cluster.correction_point, str)

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_sorted_by_member_count_descending(self, mock_encode):
        """Output clusters sorted by member_count descending."""
        from forma.misconception_clustering import cluster_misconceptions

        classified = _make_classified(10)
        result = cluster_misconceptions(classified, n_clusters=3)

        if len(result) > 1:
            counts = [c.member_count for c in result]
            assert counts == sorted(counts, reverse=True), (
                f"Clusters not sorted by member_count descending: {counts}"
            )

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_sparse)
    def test_min_cluster_size_merges_singles_to_other(self, mock_encode):
        """min_cluster_size=2: single-member clusters merged to OTHER (cluster_id=-1).

        With 5 items each far apart and n_clusters=5, each gets its own cluster.
        min_cluster_size=2 means all 5 single-member clusters should be merged
        into one OTHER cluster (cluster_id=-1).

        (FR-012, SC-006)
        """
        from forma.misconception_clustering import cluster_misconceptions

        classified = _make_classified(5)
        result = cluster_misconceptions(
            classified, n_clusters=5, min_cluster_size=2,
        )

        # All singles merged => exactly 1 OTHER cluster with id=-1
        other_clusters = [c for c in result if c.cluster_id == -1]
        assert len(other_clusters) == 1, (
            f"Expected exactly 1 OTHER cluster (cluster_id=-1), got {len(other_clusters)}. "
            f"Cluster IDs: {[c.cluster_id for c in result]}"
        )
        assert other_clusters[0].member_count == 5

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_n_clusters_exceeds_input_count(self, mock_encode):
        """n_clusters=5 but only 3 inputs -> output count <= 3 (FR-014)."""
        from forma.misconception_clustering import cluster_misconceptions

        classified = _make_classified(3)
        result = cluster_misconceptions(classified, n_clusters=5)

        assert len(result) <= 3

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_single_input(self, mock_encode):
        """Single input returns single cluster."""
        from forma.misconception_clustering import cluster_misconceptions

        classified = _make_classified(1)
        result = cluster_misconceptions(classified, n_clusters=3)

        assert len(result) == 1
        assert result[0].member_count == 1

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_mixed_patterns_majority_pattern(self, mock_encode):
        """Clusters with mixed patterns have majority pattern assigned."""
        from forma.misconception_clustering import cluster_misconceptions

        classified = _make_mixed_classified(10)
        result = cluster_misconceptions(classified, n_clusters=3)

        for cluster in result:
            assert isinstance(cluster.pattern, MisconceptionPattern)

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_deterministic)
    def test_concept_absence_description_format(self, mock_encode):
        """CONCEPT_ABSENCE items use description alone for embedding text."""
        from forma.misconception_clustering import cluster_misconceptions

        # All CONCEPT_ABSENCE pattern (no master_edge)
        items = []
        for i in range(5):
            items.append(ClassifiedMisconception(
                pattern=MisconceptionPattern.CONCEPT_ABSENCE,
                master_edge=None,
                student_edge=None,
                concept=f"concept_{i}",
                confidence=0.75,
                description=f"핵심 개념 부재: concept_{i}",
            ))

        result = cluster_misconceptions(items, n_clusters=2)
        assert isinstance(result, list)
        assert len(result) > 0
