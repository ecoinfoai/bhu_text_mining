"""Chart generation for domain coverage reports.

Builds matplotlib charts (coverage bar, emphasis bias scatter,
section variance heatmap) as BytesIO PNG objects for PDF embedding.
"""

from __future__ import annotations

import io
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.font_manager import FontProperties

from forma.chart_utils import save_fig

logger = logging.getLogger(__name__)

__all__ = [
    "build_coverage_bar_chart",
    "build_emphasis_bias_scatter",
    "build_section_variance_heatmap",
]

# Section color palette
_SECTION_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0",
                   "#F44336", "#00BCD4", "#795548", "#607D8B"]


def _get_font_props(font_path: str | None) -> FontProperties | None:
    """Get FontProperties for Korean text rendering.

    Args:
        font_path: Path to Korean TTF font, or None.

    Returns:
        FontProperties instance or None.
    """
    if font_path:
        return FontProperties(fname=font_path)
    return None


# ----------------------------------------------------------------
# T033: Coverage bar chart
# ----------------------------------------------------------------


def build_coverage_bar_chart(
    result: object,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build coverage bar chart showing overall + per-section rates.

    Y-axis: [0, 1], X-axis: "전체" + individual sections.
    Title: "실효 커버리지율"

    Args:
        result: CoverageResult with effective_coverage_rate and
            per_section_coverage.
        font_path: Path to Korean font.
        dpi: Chart resolution.

    Returns:
        BytesIO PNG buffer.
    """
    fp = _get_font_props(font_path)

    labels = ["전체"]
    values = [result.effective_coverage_rate]
    colors = ["#333333"]

    for i, (section, rate) in enumerate(
        sorted(result.per_section_coverage.items())
    ):
        labels.append(f"{section}반")
        values.append(rate)
        colors.append(_SECTION_COLORS[i % len(_SECTION_COLORS)])

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontproperties=fp,
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("커버리지율", fontproperties=fp)
    ax.set_title("실효 커버리지율", fontproperties=fp, fontsize=14, fontweight="bold")

    # Set tick label fonts
    if fp:
        for label in ax.get_xticklabels():
            label.set_fontproperties(fp)
        for label in ax.get_yticklabels():
            label.set_fontproperties(fp)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return save_fig(fig, dpi=dpi)


# ----------------------------------------------------------------
# T034: Emphasis bias scatter
# ----------------------------------------------------------------


def build_emphasis_bias_scatter(
    result: object,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build scatter plot of textbook frequency vs lecture emphasis.

    X = textbook frequency, Y = mean lecture emphasis (in-scope only).
    Green = COVERED, Red = GAP. Spearman rho shown in corner.
    Title: "강조 편향 분석"

    Args:
        result: CoverageResult with classified_concepts.
        font_path: Path to Korean font.
        dpi: Chart resolution.

    Returns:
        BytesIO PNG buffer.
    """
    from forma.domain_coverage_analyzer import ConceptState

    fp = _get_font_props(font_path)

    covered_x, covered_y = [], []
    gap_x, gap_y = [], []

    for cc in result.classified_concepts:
        if not cc.in_scope or cc.emphasis is None:
            continue
        freq = cc.concept.frequency
        emph = cc.emphasis.mean_score
        if cc.state == ConceptState.COVERED:
            covered_x.append(freq)
            covered_y.append(emph)
        elif cc.state == ConceptState.GAP:
            gap_x.append(freq)
            gap_y.append(emph)

    fig, ax = plt.subplots(figsize=(8, 6))

    if covered_x:
        ax.scatter(
            covered_x, covered_y, c="#4CAF50", alpha=0.7,
            label="다룸 (COVERED)", s=50, edgecolors="white",
        )
    if gap_x:
        ax.scatter(
            gap_x, gap_y, c="#F44336", alpha=0.7,
            label="누락 위험 (GAP)", s=50, marker="x", linewidths=2,
        )

    ax.set_xlabel("교과서 빈도", fontproperties=fp)
    ax.set_ylabel("강의 강조도", fontproperties=fp)
    ax.set_title("강조 편향 분석", fontproperties=fp, fontsize=14, fontweight="bold")

    # Show Spearman rho
    if result.emphasis_bias_correlation is not None:
        rho_text = f"Spearman ρ = {result.emphasis_bias_correlation:.3f}"
        ax.text(
            0.95, 0.95, rho_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontproperties=fp,
            bbox={"boxstyle": "round", "facecolor": "#f0f0f0", "alpha": 0.8},
        )

    if fp:
        for label in ax.get_xticklabels():
            label.set_fontproperties(fp)
        for label in ax.get_yticklabels():
            label.set_fontproperties(fp)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(prop=fp) if fp else ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return save_fig(fig, dpi=dpi)


# ----------------------------------------------------------------
# T035: Section variance heatmap
# ----------------------------------------------------------------


def build_section_variance_heatmap(
    result: object,
    font_path: str | None = None,
    dpi: int = 150,
    max_concepts: int = 20,
) -> io.BytesIO:
    """Build heatmap of concept emphasis across sections.

    Rows = concepts (top N by variance), Columns = sections.
    Color map: RdYlGn. Title: "분반 간 강조도 편차"

    Args:
        result: CoverageResult with classified_concepts.
        font_path: Path to Korean font.
        dpi: Chart resolution.
        max_concepts: Maximum number of concepts to show.

    Returns:
        BytesIO PNG buffer.
    """
    fp = _get_font_props(font_path)

    # Collect concepts with emphasis data, sorted by std descending
    with_emphasis = [
        cc for cc in result.classified_concepts
        if cc.emphasis is not None and cc.emphasis.std_score > 0
    ]
    with_emphasis.sort(key=lambda cc: cc.emphasis.std_score, reverse=True)
    with_emphasis = with_emphasis[:max_concepts]

    if not with_emphasis:
        # Create empty chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5, "분반 간 편차 데이터 없음",
            ha="center", va="center", fontproperties=fp, fontsize=14,
        )
        ax.set_title("분반 간 강조도 편차", fontproperties=fp, fontsize=14)
        return save_fig(fig, dpi=dpi)

    # Collect all sections
    sections: set[str] = set()
    for cc in with_emphasis:
        sections.update(cc.emphasis.section_scores.keys())
    sections_sorted = sorted(sections)

    # Build matrix
    concept_names = [cc.concept.name_ko for cc in with_emphasis]
    matrix = np.zeros((len(with_emphasis), len(sections_sorted)))
    for i, cc in enumerate(with_emphasis):
        for j, section in enumerate(sections_sorted):
            matrix[i, j] = cc.emphasis.section_scores.get(section, 0.0)

    fig_height = max(4, len(concept_names) * 0.4 + 2)
    fig, ax = plt.subplots(figsize=(max(6, len(sections_sorted) * 1.5 + 2), fig_height))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(sections_sorted)))
    ax.set_xticklabels(
        [f"{s}반" for s in sections_sorted],
        fontproperties=fp,
    )
    ax.set_yticks(range(len(concept_names)))
    ax.set_yticklabels(concept_names, fontproperties=fp, fontsize=8)

    # Add text annotations
    for i in range(len(concept_names)):
        for j in range(len(sections_sorted)):
            val = matrix[i, j]
            color = "white" if val < 0.3 or val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title(
        "분반 간 강조도 편차",
        fontproperties=fp, fontsize=14, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="강조도", shrink=0.8)
    fig.tight_layout()

    return save_fig(fig, dpi=dpi)
