"""Adversary attack tests for Phase 5+6 (US3+US4): misconception clustering + LLM correction.

6 personas attack cluster_misconceptions() and generate_cluster_correction():
1. Edge Case Hunter: boundary conditions in clustering
2. Memory Saboteur: resource exhaustion in clustering
3. Type System Antagonist: type safety in cluster pipeline
4. Concurrency Destroyer: thread safety with KMeans
5. PDF Killer: crash-inducing cluster data in PDF
6. Data Integrity Enforcer: clustering invariants
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from forma.evaluation_types import TripletEdge
from forma.misconception_classifier import (
    ClassifiedMisconception,
    MisconceptionPattern,
)
from forma.misconception_clustering import (
    MisconceptionCluster,
    cluster_misconceptions,
)
from forma.professor_report_llm import generate_cluster_correction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classified(
    pattern: MisconceptionPattern = MisconceptionPattern.CAUSAL_REVERSAL,
    description: str = "인과 방향 역전: A->R->B",
    master_edge: TripletEdge | None = None,
    concept: str | None = None,
) -> ClassifiedMisconception:
    if master_edge is None and pattern != MisconceptionPattern.CONCEPT_ABSENCE:
        master_edge = TripletEdge("A", "R", "B")
    return ClassifiedMisconception(
        pattern=pattern,
        master_edge=master_edge,
        student_edge=TripletEdge("B", "R", "A") if master_edge else None,
        concept=concept,
        confidence=0.85,
        description=description,
    )


def _make_n_classified(n: int, seed: int = 42) -> list[ClassifiedMisconception]:
    """Generate n distinct ClassifiedMisconception instances."""
    patterns = list(MisconceptionPattern)
    items = []
    for i in range(n):
        p = patterns[i % len(patterns)]
        me = None if p == MisconceptionPattern.CONCEPT_ABSENCE else TripletEdge(f"S{i}", "R", f"O{i}")
        items.append(_make_classified(
            pattern=p,
            description=f"오류 설명 {i}: 패턴={p.value}",
            master_edge=me,
            concept=f"Concept{i}" if p == MisconceptionPattern.CONCEPT_ABSENCE else None,
        ))
    return items


def _mock_encode_texts(texts: list[str]) -> np.ndarray:
    """Deterministic mock embeddings: each text gets a distinct vector."""
    n = len(texts)
    rng = np.random.RandomState(42)
    return rng.randn(n, 4).astype(np.float32)


# ===========================================================================
# PERSONA 1: THE EDGE CASE HUNTER
# ===========================================================================


class TestClusterEdgeCaseHunter:
    """Persona 1: Boundary conditions in clustering."""

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_empty_input_returns_empty_list(self, mock_enc):
        """Empty classified list -> returns []."""
        result = cluster_misconceptions([], n_clusters=3)
        assert result == []
        mock_enc.assert_not_called()  # Should not call encode_texts

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_single_item_returns_one_cluster(self, mock_enc):
        """Single item -> 1 cluster (n_clusters clamped to 1)."""
        items = [_make_classified(description="single error")]
        result = cluster_misconceptions(items, n_clusters=5)
        # May be 1 regular cluster or 1 OTHER cluster depending on min_cluster_size
        assert len(result) >= 1
        total = sum(c.member_count for c in result)
        assert total == 1

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_n_clusters_greater_than_items(self, mock_enc):
        """n_clusters > len(classified) -> actual clusters <= len(classified)."""
        items = _make_n_classified(3)
        result = cluster_misconceptions(items, n_clusters=10)
        # KMeans k = min(10, 3) = 3
        total = sum(c.member_count for c in result)
        assert total == 3

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_n_clusters_1(self, mock_enc):
        """n_clusters=1 -> all items in one cluster."""
        items = _make_n_classified(5)
        result = cluster_misconceptions(items, n_clusters=1)
        total = sum(c.member_count for c in result)
        assert total == 5

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_min_cluster_size_larger_than_all_clusters(self, mock_enc):
        """min_cluster_size > all cluster sizes -> everything in OTHER."""
        items = _make_n_classified(4)
        result = cluster_misconceptions(items, n_clusters=4, min_cluster_size=100)
        # All should be in OTHER since each cluster has ~1 member
        total = sum(c.member_count for c in result)
        assert total == 4
        # There should be an OTHER cluster with cluster_id=-1
        other = [c for c in result if c.cluster_id == -1]
        assert len(other) >= 1

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_all_same_description(self, mock_enc):
        """All items have identical description -> clustering still works."""
        items = [
            _make_classified(description="동일한 오류")
            for _ in range(5)
        ]
        result = cluster_misconceptions(items, n_clusters=2)
        total = sum(c.member_count for c in result)
        assert total == 5

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_concept_absence_no_master_edge(self, mock_enc):
        """CONCEPT_ABSENCE items (master_edge=None) cluster without error."""
        items = [
            _make_classified(
                pattern=MisconceptionPattern.CONCEPT_ABSENCE,
                description=f"핵심 개념 부재: Concept{i}",
                master_edge=None,
                concept=f"Concept{i}",
            )
            for i in range(5)
        ]
        result = cluster_misconceptions(items, n_clusters=2)
        total = sum(c.member_count for c in result)
        assert total == 5


# ===========================================================================
# PERSONA 2: THE MEMORY SABOTEUR
# ===========================================================================


class TestClusterMemorySaboteur:
    """Persona 2: Resource exhaustion in clustering."""

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_100_items_completes(self, mock_enc):
        """100 items clusters without crash."""
        items = _make_n_classified(100)
        result = cluster_misconceptions(items, n_clusters=5)
        total = sum(c.member_count for c in result)
        assert total == 100

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_repeated_calls_independent(self, mock_enc):
        """Repeated calls return independent results."""
        items1 = _make_n_classified(5)
        items2 = _make_n_classified(10)
        r1 = cluster_misconceptions(items1, n_clusters=2)
        r2 = cluster_misconceptions(items2, n_clusters=3)
        assert sum(c.member_count for c in r1) == 5
        assert sum(c.member_count for c in r2) == 10


# ===========================================================================
# PERSONA 3: THE TYPE SYSTEM ANTAGONIST
# ===========================================================================


class TestClusterTypeAntagonist:
    """Persona 3: Type safety in the cluster pipeline."""

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_cluster_id_is_int(self, mock_enc):
        """cluster_id is always int (not str)."""
        items = _make_n_classified(5)
        result = cluster_misconceptions(items, n_clusters=2)
        for cluster in result:
            assert isinstance(cluster.cluster_id, int)

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_other_cluster_id_is_negative_one(self, mock_enc):
        """OTHER cluster has cluster_id == -1 (int, not str)."""
        items = _make_n_classified(6)
        result = cluster_misconceptions(items, n_clusters=6, min_cluster_size=100)
        others = [c for c in result if c.cluster_id == -1]
        assert len(others) >= 1
        for o in others:
            assert o.cluster_id == -1
            assert isinstance(o.cluster_id, int)

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_correction_point_is_str_not_none(self, mock_enc):
        """correction_point is always str, never None."""
        items = _make_n_classified(5)
        result = cluster_misconceptions(items, n_clusters=2)
        for cluster in result:
            assert isinstance(cluster.correction_point, str)
            # Default should be ""
            assert cluster.correction_point == ""

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_representative_error_is_str(self, mock_enc):
        """representative_error is always a non-empty str."""
        items = _make_n_classified(5)
        result = cluster_misconceptions(items, n_clusters=2)
        for cluster in result:
            assert isinstance(cluster.representative_error, str)
            assert len(cluster.representative_error) > 0

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_member_count_is_int(self, mock_enc):
        """member_count is int."""
        items = _make_n_classified(5)
        result = cluster_misconceptions(items, n_clusters=2)
        for cluster in result:
            assert isinstance(cluster.member_count, int)

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_pattern_is_misconception_pattern(self, mock_enc):
        """pattern is MisconceptionPattern enum."""
        items = _make_n_classified(5)
        result = cluster_misconceptions(items, n_clusters=2)
        for cluster in result:
            assert isinstance(cluster.pattern, MisconceptionPattern)

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_student_errors_contains_only_strings(self, mock_enc):
        """student_errors list contains only str items."""
        items = _make_n_classified(5)
        result = cluster_misconceptions(items, n_clusters=2)
        for cluster in result:
            for err in cluster.student_errors:
                assert isinstance(err, str)


# ===========================================================================
# PERSONA 4: THE CONCURRENCY DESTROYER
# ===========================================================================


class TestClusterConcurrencyDestroyer:
    """Persona 4: Thread safety with KMeans."""

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_kmeans_deterministic_with_random_state(self, mock_enc):
        """KMeans with random_state=42 produces deterministic results."""
        items = _make_n_classified(10)
        r1 = cluster_misconceptions(items, n_clusters=3)
        r2 = cluster_misconceptions(items, n_clusters=3)
        # Same item counts per cluster
        counts1 = sorted([c.member_count for c in r1])
        counts2 = sorted([c.member_count for c in r2])
        assert counts1 == counts2


# ===========================================================================
# PERSONA 5: THE PDF KILLER — LLM correction
# ===========================================================================


class TestLLMCorrectionPDFKiller:
    """Persona 5: LLM correction failure modes."""

    def test_generate_cluster_correction_success(self):
        """Mock LLM returns valid correction -> non-empty str returned."""
        cluster = MisconceptionCluster(
            cluster_id=0,
            pattern=MisconceptionPattern.CAUSAL_REVERSAL,
            representative_error="인과 방향 역전: A->R->B",
            member_count=5,
            student_errors=["error1", "error2"],
        )
        provider = MagicMock()
        provider.generate.return_value = "학생들이 인과 방향을 혼동합니다."
        provider.model_name = "test-model"

        result = generate_cluster_correction(
            cluster, TripletEdge("A", "R", "B"), provider
        )
        assert isinstance(result, str)
        assert len(result) > 0
        provider.generate.assert_called_once()

    def test_generate_cluster_correction_llm_exception_returns_empty_str(self):
        """LLM exception -> returns "" (not None), no exception propagated."""
        cluster = MisconceptionCluster(
            cluster_id=0,
            pattern=MisconceptionPattern.CAUSAL_REVERSAL,
            representative_error="error",
            member_count=1,
            student_errors=["error"],
        )
        provider = MagicMock()
        provider.generate.side_effect = RuntimeError("API down")
        provider.model_name = "test-model"

        result = generate_cluster_correction(cluster, None, provider)
        assert result == ""
        assert isinstance(result, str)

    def test_generate_cluster_correction_llm_returns_none(self):
        """LLM returns None -> returns "" (empty string)."""
        cluster = MisconceptionCluster(
            cluster_id=0,
            pattern=MisconceptionPattern.CONCEPT_ABSENCE,
            representative_error="핵심 개념 부재",
            member_count=2,
            student_errors=["err1", "err2"],
        )
        provider = MagicMock()
        provider.generate.return_value = None
        provider.model_name = "test-model"

        result = generate_cluster_correction(cluster, None, provider)
        assert result == ""
        assert isinstance(result, str)

    def test_generate_cluster_correction_llm_returns_empty(self):
        """LLM returns empty string -> returns ""."""
        cluster = MisconceptionCluster(
            cluster_id=0,
            pattern=MisconceptionPattern.RELATION_CONFUSION,
            representative_error="관계 혼동",
            member_count=3,
            student_errors=["e1", "e2", "e3"],
        )
        provider = MagicMock()
        provider.generate.return_value = "   "
        provider.model_name = "test-model"

        result = generate_cluster_correction(
            cluster, TripletEdge("A", "R", "B"), provider
        )
        assert result == ""

    def test_generate_cluster_correction_llm_returns_whitespace_only(self):
        """LLM returns whitespace -> returns ""."""
        cluster = MisconceptionCluster(
            cluster_id=0,
            pattern=MisconceptionPattern.INCLUSION_ERROR,
            representative_error="포함 관계 역전",
            member_count=2,
            student_errors=["e1", "e2"],
        )
        provider = MagicMock()
        provider.generate.return_value = "\n\t  \n"
        provider.model_name = "test-model"

        result = generate_cluster_correction(
            cluster, TripletEdge("X", "Y", "Z"), provider
        )
        assert result == ""

    def test_generate_cluster_correction_called_once_per_cluster(self):
        """LLM is called exactly once (not per student)."""
        cluster = MisconceptionCluster(
            cluster_id=0,
            pattern=MisconceptionPattern.CAUSAL_REVERSAL,
            representative_error="error",
            member_count=10,
            student_errors=[f"e{i}" for i in range(10)],
        )
        provider = MagicMock()
        provider.generate.return_value = "correction text"
        provider.model_name = "test-model"

        generate_cluster_correction(
            cluster, TripletEdge("A", "R", "B"), provider
        )
        assert provider.generate.call_count == 1

    def test_generate_cluster_correction_with_none_master_edge(self):
        """master_edge=None (CONCEPT_ABSENCE) doesn't crash."""
        cluster = MisconceptionCluster(
            cluster_id=0,
            pattern=MisconceptionPattern.CONCEPT_ABSENCE,
            representative_error="핵심 개념 부재: 세포막",
            member_count=3,
            student_errors=["e1", "e2", "e3"],
        )
        provider = MagicMock()
        provider.generate.return_value = "세포막의 중요성을 강조하세요."
        provider.model_name = "test-model"

        result = generate_cluster_correction(cluster, None, provider)
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# PERSONA 6: THE DATA INTEGRITY ENFORCER
# ===========================================================================


class TestClusterDataIntegrity:
    """Persona 6: Mathematical invariant violations."""

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_all_inputs_accounted_for(self, mock_enc):
        """Every input item appears in exactly one cluster's member_count."""
        items = _make_n_classified(10)
        result = cluster_misconceptions(items, n_clusters=3)
        total = sum(c.member_count for c in result)
        assert total == 10

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_member_count_equals_len_student_errors(self, mock_enc):
        """member_count == len(student_errors) for every cluster."""
        items = _make_n_classified(10)
        result = cluster_misconceptions(items, n_clusters=3)
        for cluster in result:
            assert cluster.member_count == len(cluster.student_errors), (
                f"Cluster {cluster.cluster_id}: member_count={cluster.member_count} "
                f"but len(student_errors)={len(cluster.student_errors)}"
            )

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_representative_error_in_student_errors(self, mock_enc):
        """representative_error is actually in the student_errors list."""
        items = _make_n_classified(10)
        result = cluster_misconceptions(items, n_clusters=3)
        for cluster in result:
            assert cluster.representative_error in cluster.student_errors, (
                f"Cluster {cluster.cluster_id}: representative_error "
                f"'{cluster.representative_error}' not in student_errors"
            )

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_sorted_by_member_count_descending(self, mock_enc):
        """Output clusters are sorted by member_count descending."""
        items = _make_n_classified(15)
        result = cluster_misconceptions(items, n_clusters=4)
        counts = [c.member_count for c in result]
        assert counts == sorted(counts, reverse=True)

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_no_duplicate_student_errors_across_clusters(self, mock_enc):
        """No student_error string appears in multiple clusters.

        Each input ClassifiedMisconception should be assigned to exactly one cluster.
        NOTE: If two inputs have the same description string, they are different
        items but have the same text. We verify by total count matching.
        """
        items = _make_n_classified(10)
        result = cluster_misconceptions(items, n_clusters=3)
        all_errors: list[str] = []
        for cluster in result:
            all_errors.extend(cluster.student_errors)
        # Total errors should equal total items
        assert len(all_errors) == 10

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_other_cluster_members_sum(self, mock_enc):
        """OTHER cluster member_count is sum of all small clusters merged into it."""
        items = _make_n_classified(8)
        # n_clusters=8, min_cluster_size=100 -> all go to OTHER
        result = cluster_misconceptions(items, n_clusters=8, min_cluster_size=100)
        total = sum(c.member_count for c in result)
        assert total == 8

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_correction_point_default_empty_string(self, mock_enc):
        """All clusters have correction_point == '' by default (not None)."""
        items = _make_n_classified(5)
        result = cluster_misconceptions(items, n_clusters=2)
        for cluster in result:
            assert cluster.correction_point == ""
            assert cluster.correction_point is not None

    @patch("forma.misconception_clustering.encode_texts", side_effect=_mock_encode_texts)
    def test_cluster_ids_unique_except_other(self, mock_enc):
        """Regular cluster IDs are unique. Only one OTHER cluster (id=-1)."""
        items = _make_n_classified(10)
        result = cluster_misconceptions(items, n_clusters=5, min_cluster_size=2)
        ids = [c.cluster_id for c in result]
        regular_ids = [i for i in ids if i != -1]
        assert len(regular_ids) == len(set(regular_ids))
        # At most one OTHER cluster
        assert ids.count(-1) <= 1
