"""Chart generation for cross-section comparison in aggregate professor reports.

Produces three chart types as PNG ``io.BytesIO`` buffers:
- Section score distribution box plot
- Concept mastery heatmap (sections x concepts)
- Weekly interaction line chart (week x mean score per section)

Uses matplotlib Agg backend -- no display server required. No LLM API calls.
"""

from __future__ import annotations

import io
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

from forma.chart_utils import save_fig as _save_fig  # noqa: E402
from forma.font_utils import find_korean_font  # noqa: E402

logger = logging.getLogger(__name__)

# Section color palette (up to 8 sections)
_SECTION_COLORS = [
    "#1565C0", "#2E7D32", "#F57F17", "#C62828",
    "#6A1B9A", "#00695C", "#E65100", "#37474F",
]


def _get_font_prop() -> FontProperties:
    """Get FontProperties for Korean text rendering."""
    font_path = find_korean_font()
    return FontProperties(fname=font_path)


def build_section_box_plot(
    section_scores: dict[str, list[float]],
    dpi: int = 150,
) -> io.BytesIO:
    """Generate a box plot comparing score distributions across sections.

    Args:
        section_scores: section_name -> list of ensemble scores.
        dpi: Image resolution.

    Returns:
        PNG image as BytesIO buffer.
    """
    font_prop = _get_font_prop()
    sections = sorted(section_scores.keys())
    data = [section_scores[s] for s in sections]

    fig, ax = plt.subplots(figsize=(max(4, len(sections) * 1.5), 5))
    bp = ax.boxplot(
        data,
        tick_labels=sections,
        patch_artist=True,
        widths=0.6,
    )

    for i, patch in enumerate(bp["boxes"]):
        color = _SECTION_COLORS[i % len(_SECTION_COLORS)]
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Ensemble Score", fontproperties=font_prop)
    ax.set_title("분반별 점수 분포", fontproperties=font_prop, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)

    fig.tight_layout()

    return _save_fig(fig, dpi=dpi)


def build_concept_mastery_heatmap(
    concept_mastery: dict[str, dict[str, float]],
    dpi: int = 150,
) -> io.BytesIO | None:
    """Generate a heatmap of concept mastery across sections.

    Args:
        concept_mastery: section_name -> concept_name -> mean mastery.
        dpi: Image resolution.

    Returns:
        PNG image as BytesIO buffer, or None if data is empty.
    """
    if not concept_mastery:
        return None

    font_prop = _get_font_prop()
    sections = sorted(concept_mastery.keys())

    # Collect all concepts across sections
    all_concepts: set[str] = set()
    for concepts in concept_mastery.values():
        all_concepts.update(concepts.keys())
    concepts_list = sorted(all_concepts)

    if not concepts_list:
        return None

    # Build matrix: rows=sections, cols=concepts
    matrix = np.zeros((len(sections), len(concepts_list)))
    for i, section in enumerate(sections):
        for j, concept in enumerate(concepts_list):
            matrix[i, j] = concept_mastery[section].get(concept, 0.0)

    # Truncate to top 20 concepts if too many
    max_concepts = 20
    if len(concepts_list) > max_concepts:
        # Keep concepts with highest variance across sections
        variances = np.var(matrix, axis=0)
        top_indices = np.argsort(variances)[-max_concepts:]
        top_indices.sort()
        concepts_list = [concepts_list[i] for i in top_indices]
        matrix = matrix[:, top_indices]

    fig_width = max(6, len(concepts_list) * 0.4 + 2)
    fig_height = max(3, len(sections) * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Mastery", shrink=0.8)

    ax.set_xticks(range(len(concepts_list)))
    ax.set_xticklabels(concepts_list, rotation=45, ha="right", fontproperties=font_prop, fontsize=8)
    ax.set_yticks(range(len(sections)))
    ax.set_yticklabels(sections, fontproperties=font_prop)
    ax.set_title("분반별 개념 숙달도", fontproperties=font_prop, fontsize=14)

    fig.tight_layout()

    return _save_fig(fig, dpi=dpi)


def build_weekly_interaction_chart(
    weekly_data: dict[str, dict[int, float]] | None,
    dpi: int = 150,
) -> io.BytesIO | None:
    """Generate a line chart showing section score trends across weeks.

    Args:
        weekly_data: section_name -> week -> mean_score. None if no
            longitudinal data available.
        dpi: Image resolution.

    Returns:
        PNG image as BytesIO buffer, or None if data is empty/None.
    """
    if not weekly_data:
        return None

    font_prop = _get_font_prop()
    sections = sorted(weekly_data.keys())

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, section in enumerate(sections):
        weeks_dict = weekly_data[section]
        weeks = sorted(weeks_dict.keys())
        scores = [weeks_dict[w] for w in weeks]
        color = _SECTION_COLORS[i % len(_SECTION_COLORS)]
        ax.plot(weeks, scores, marker="o", label=section, color=color, linewidth=2)

    ax.set_xlabel("주차 (Week)", fontproperties=font_prop)
    ax.set_ylabel("평균 점수", fontproperties=font_prop)
    ax.set_title("분반별 주차별 점수 변화", fontproperties=font_prop, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(prop=font_prop)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return _save_fig(fig, dpi=dpi)
