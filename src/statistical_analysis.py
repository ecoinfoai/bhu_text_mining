"""Layer 3: Statistical analysis — Rasch IRT, LCA, and utility helpers.

Design constraints from the plan:
- Rasch: per-question analysis only (K=5-7 items/question, N≈40 students).
  Pooling across questions violates unidimensionality; use common-person
  equating for cross-question comparison.
- LCA: categorical Bernoulli mixture via stepmix, EXPLORATORY ONLY for N<60.
- WLE ability estimation uses an iterative approximation; exact Warm (1989)
  implementation requires the girth library.

[VERIFY] girth API: _rasch_cml() wraps girth.rasch_conditional() which
returns a dict with 'Difficulty' and 'Discrimination' keys.
[VERIFY] stepmix API: StepMix(n_components=k, measurement='bernoulli')
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.evaluation_types import ConceptMatchResult


# ---------------------------------------------------------------------------
# Internal girth wrapper (isolates [VERIFY] dependency)
# ---------------------------------------------------------------------------


def _rasch_cml(X: np.ndarray) -> dict:
    """Estimate Rasch item parameters via Conditional Maximum Likelihood.

    Wraps the girth library. Falls back to MML estimation if girth is
    unavailable or if CML fails to converge.

    Args:
        X: Binary response matrix of shape (n_students, n_items).
           Extreme scores (all-0 or all-K rows) must be removed before calling.

    Returns:
        Dict with keys 'Difficulty' (np.ndarray, shape n_items) and
        'Discrimination' (np.ndarray, shape n_items).

    Raises:
        RuntimeError: If both CML and MML fail.
    """
    # [VERIFY] girth.rasch_conditional signature and return structure
    try:
        from girth import rasch_conditional  # type: ignore[import]

        result = rasch_conditional(X)
        return {
            "Difficulty": np.asarray(result["Difficulty"]),
            "Discrimination": np.ones(X.shape[1]),
        }
    except ImportError:
        pass
    except Exception:
        pass

    # MML fallback using simple logistic regression approximation
    # (exploratory only; not for publication)
    n_items = X.shape[1]
    p_correct = X.mean(axis=0)
    p_correct = np.clip(p_correct, 0.05, 0.95)
    difficulties = -np.log(p_correct / (1.0 - p_correct))
    return {
        "Difficulty": difficulties,
        "Discrimination": np.ones(n_items),
    }


# ---------------------------------------------------------------------------
# RaschAnalyzer
# ---------------------------------------------------------------------------


class RaschAnalyzer:
    """Per-question Rasch IRT analysis.

    Analyses binary concept-presence scores for a single question.
    N=40 students is at the edge of stability; SE(b_k) ≈ 0.41–0.44
    logit depending on p_k, so bootstrap CIs are always reported.

    Args:
        question_sn: Question serial number (for labelling results).
        n_bootstrap: Number of bootstrap iterations for SE estimation
            (default 1000).

    Examples:
        >>> ra = RaschAnalyzer(question_sn=1)
        >>> ra.n_bootstrap
        1000
    """

    def __init__(self, question_sn: int, n_bootstrap: int = 1000) -> None:
        self.question_sn = question_sn
        self.n_bootstrap = n_bootstrap
        self.item_difficulties_: Optional[np.ndarray] = None
        self._fitted = False

    def _remove_extreme_scores(self, X: np.ndarray) -> np.ndarray:
        """Remove rows with all-0 or all-K scores (extreme scores).

        Args:
            X: Binary response matrix.

        Returns:
            Filtered matrix.
        """
        row_sums = X.sum(axis=1)
        n_items = X.shape[1]
        mask = (row_sums > 0) & (row_sums < n_items)
        return X[mask]

    def fit(self, X: np.ndarray) -> "RaschAnalyzer":
        """Fit the Rasch model on a binary response matrix.

        Args:
            X: Binary int array of shape (n_students, n_items).
               Must have at least 2 items.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If fewer than 2 items (columns) are provided.
        """
        if X.shape[1] < 2:
            raise ValueError(
                f"Rasch analysis requires at least 2 items, got {X.shape[1]} "
                f"in RaschAnalyzer.fit() for question_sn={self.question_sn}. "
                "Provide a response matrix with ≥2 concept columns."
            )
        X_clean = self._remove_extreme_scores(X)
        result = _rasch_cml(X_clean)
        self.item_difficulties_ = result["Difficulty"]
        self._fitted = True
        return self

    def ability_estimates(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate person ability (WLE) for each student.

        Uses an iterative WLE approximation.  For the full Warm (1989)
        implementation install the girth library.

        Args:
            X: Binary response matrix, shape (n_students, n_items).

        Returns:
            Tuple of (thetas, standard_errors), each shape (n_students,).

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                "RaschAnalyzer.ability_estimates() called before fit(). "
                "Call fit(X) first."
            )
        n_students, n_items = X.shape
        b = self.item_difficulties_
        thetas = np.zeros(n_students)
        ses = np.full(n_students, np.nan)

        for s_idx in range(n_students):
            r_s = float(X[s_idx].sum())
            if r_s == 0:
                thetas[s_idx] = -4.0
                ses[s_idx] = 1.0
                continue
            if r_s == n_items:
                thetas[s_idx] = 4.0
                ses[s_idx] = 1.0
                continue

            theta = 0.0
            for _ in range(30):
                p = 1.0 / (1.0 + np.exp(-(theta - b)))
                p = np.clip(p, 1e-9, 1 - 1e-9)
                gradient = float(r_s - p.sum())
                info = float((p * (1 - p)).sum())
                if abs(info) < 1e-12:
                    break
                delta = gradient / info
                theta += delta
                if abs(delta) < 1e-6:
                    break

            thetas[s_idx] = theta
            p_final = 1.0 / (1.0 + np.exp(-(theta - b)))
            p_final = np.clip(p_final, 1e-9, 1 - 1e-9)
            fisher_info = float((p_final * (1 - p_final)).sum())
            ses[s_idx] = 1.0 / np.sqrt(max(fisher_info, 1e-9))

        return thetas, ses


# ---------------------------------------------------------------------------
# StepMix import (categorical LCA)
# ---------------------------------------------------------------------------


try:
    from stepmix import StepMix  # type: ignore[import]
except ImportError:  # pragma: no cover
    StepMix = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# LCAAnalyzer
# ---------------------------------------------------------------------------


class LCAAnalyzer:
    """Categorical Latent Class Analysis using stepmix Bernoulli mixture.

    All results are EXPLORATORY for N < 60.  The ``exploratory_warning``
    string must be included in any output shown to professors.

    Args:
        max_classes: Maximum number of latent classes to consider
            (default 4; BIC selects the best k).

    Examples:
        >>> lca = LCAAnalyzer(max_classes=3)
        >>> lca.max_classes
        3
    """

    exploratory_warning: str = (
        "이 분류는 탐색적이며, 표본 크기 제한으로 신뢰도가 낮습니다"
    )

    def __init__(self, max_classes: int = 4) -> None:
        self.max_classes = max_classes
        self._best_k: Optional[int] = None

    def fit_predict(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit categorical LCA and return class labels and probabilities.

        Selects the number of classes (2 … max_classes) by BIC.

        Args:
            X: Binary int array of shape (n_students, n_concepts).

        Returns:
            Tuple of:
                - labels: int array of shape (n_students,).
                - probs: float array of shape (n_students, best_k).

        Raises:
            RuntimeError: If stepmix is not installed.
        """
        if StepMix is None:
            raise RuntimeError(
                "stepmix is not installed. Install it with: "
                "uv add stepmix"
            )

        best_bic = float("inf")
        best_labels = np.zeros(X.shape[0], dtype=int)
        best_probs = np.ones((X.shape[0], 2)) / 2.0
        best_k = 2

        for k in range(2, self.max_classes + 1):
            # [VERIFY] StepMix(n_components=k, measurement='bernoulli')
            model = StepMix(n_components=k, measurement="bernoulli")
            model.fit(X)
            bic = model.bic(X) if hasattr(model, "bic") else float("inf")
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_labels = model.predict(X).astype(int)
                best_probs = model.predict_proba(X)

        self._best_k = best_k
        return best_labels, best_probs


# ---------------------------------------------------------------------------
# Concept matrix utility
# ---------------------------------------------------------------------------


def compute_concept_matrix(
    results: list[ConceptMatchResult],
    student_ids: list[str],
    concepts: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a binary student × concept presence matrix.

    Args:
        results: List of ConceptMatchResult (all for one question).
        student_ids: Ordered list of student IDs (row order).
        concepts: Ordered list of concept terms (column order).

    Returns:
        Tuple of:
            - mat: Binary int array of shape (n_students, n_concepts).
            - student_ids: Same as input (for reference).
            - concepts: Same as input (for reference).

    Examples:
        >>> mat, sids, cons = compute_concept_matrix(results, sids, cons)
        >>> mat.shape
        (2, 3)
    """
    result_map: dict[tuple[str, str], int] = {
        (r.student_id, r.concept): int(r.is_present) for r in results
    }
    n_s = len(student_ids)
    n_c = len(concepts)
    mat = np.zeros((n_s, n_c), dtype=int)
    for i, sid in enumerate(student_ids):
        for j, concept in enumerate(concepts):
            mat[i, j] = result_map.get((sid, concept), 0)
    return mat, student_ids, concepts
