"""Dashboard chart generation for early warning PDF reports.

Produces two chart types as PNG io.BytesIO buffers:
    - Risk type distribution bar chart
    - Deficit concepts horizontal bar chart (top 10)

Uses matplotlib Agg backend -- no display server required.
"""

from __future__ import annotations

import io
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

from forma.chart_utils import save_fig as _save_fig  # noqa: E402
from forma.font_utils import find_korean_font  # noqa: E402

logger = logging.getLogger(__name__)

# Risk type colors
_RISK_COLORS = {
    "SCORE_DECLINE": "#E53935",
    "PERSISTENT_LOW": "#FB8C00",
    "CONCEPT_DEFICIT": "#7B1FA2",
    "PARTICIPATION_DECLINE": "#1E88E5",
}


def _get_font_prop(font_path: str | None = None) -> FontProperties:
    """Get FontProperties for Korean text rendering."""
    if font_path is None:
        font_path = find_korean_font()
    return FontProperties(fname=font_path)


def build_risk_type_distribution_chart(
    risk_type_counts: dict,
    *,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build a bar chart of risk type distribution.

    Args:
        risk_type_counts: {RiskType: count} mapping.
        font_path: Path to Korean font, or None for auto-detect.
        dpi: Chart resolution.

    Returns:
        PNG image as BytesIO.
    """
    font_prop = _get_font_prop(font_path)

    if not risk_type_counts:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                fontproperties=font_prop, fontsize=14)
        ax.set_axis_off()
        return _save_fig(fig, dpi=dpi)

    labels = []
    values = []
    colors = []
    for rt, count in risk_type_counts.items():
        labels.append(rt.label)
        values.append(count)
        colors.append(_RISK_COLORS.get(rt.value, "#757575"))

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontproperties=font_prop, fontsize=9)
    ax.set_ylabel("학생 수", fontproperties=font_prop, fontsize=10)
    ax.set_title("위험 유형별 분포", fontproperties=font_prop, fontsize=12)

    # Add count labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            str(val), ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    return _save_fig(fig, dpi=dpi)


def build_deficit_concepts_chart(
    concept_counts: dict[str, int],
    *,
    font_path: str | None = None,
    dpi: int = 150,
    max_concepts: int = 10,
) -> io.BytesIO:
    """Build a horizontal bar chart of top deficit concepts.

    Args:
        concept_counts: {concept_name: student_count} mapping.
        font_path: Path to Korean font, or None for auto-detect.
        dpi: Chart resolution.
        max_concepts: Maximum number of concepts to display.

    Returns:
        PNG image as BytesIO.
    """
    font_prop = _get_font_prop(font_path)

    if not concept_counts:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                fontproperties=font_prop, fontsize=14)
        ax.set_axis_off()
        return _save_fig(fig, dpi=dpi)

    # Sort by count descending, take top N
    sorted_items = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:max_concepts]

    # Reverse for horizontal bar (highest at top)
    labels = [item[0] for item in reversed(top_items)]
    values = [item[1] for item in reversed(top_items)]

    height = max(3, len(labels) * 0.4)
    fig, ax = plt.subplots(figsize=(6, height))
    ax.barh(range(len(labels)), values, color="#E53935", alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontproperties=font_prop, fontsize=9)
    ax.set_xlabel("학생 수", fontproperties=font_prop, fontsize=10)
    ax.set_title("결손 개념 빈도 (상위)", fontproperties=font_prop, fontsize=12)

    fig.tight_layout()
    return _save_fig(fig, dpi=dpi)
