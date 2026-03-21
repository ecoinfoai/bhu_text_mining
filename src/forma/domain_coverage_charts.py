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
    "build_network_comparison_chart",
    "build_delivery_bar_chart",
    "build_delivery_heatmap",
    "build_topic_delivery_stacked_chart",
    "build_hierarchical_coverage_chart",
    "build_grouped_quality_heatmap",
    "build_concept_network_chart",
    "build_concept_network_comparison",
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


def _get_font_family(fp: FontProperties | None) -> str | None:
    """Extract font family name for networkx labels.

    Args:
        fp: FontProperties instance or None.

    Returns:
        Font family name string, or None if no font.
    """
    if fp is None:
        return None
    return fp.get_name()


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

    im = ax.imshow(matrix, cmap="cividis", aspect="auto", vmin=0, vmax=1)

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
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("강조도", fontproperties=fp)
    fig.tight_layout()

    return save_fig(fig, dpi=dpi)


# ----------------------------------------------------------------
# T039: Network comparison chart
# ----------------------------------------------------------------


def build_network_comparison_chart(
    textbook_net: object,
    lecture_net: object,
    missing_edges: list[tuple[str, str]] | None = None,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build side-by-side network comparison chart.

    Left panel: textbook network. Right panel: lecture network.
    Missing edges highlighted in red dashed lines.
    Title: "핵심 용어 네트워크 비교"

    Args:
        textbook_net: KeywordNetwork for textbook.
        lecture_net: KeywordNetwork for a lecture section.
        missing_edges: Edges in textbook but absent in lecture.
        font_path: Path to Korean font.
        dpi: Chart resolution.

    Returns:
        BytesIO PNG buffer.
    """
    import networkx as nx

    fp = _get_font_props(font_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def _draw_network(ax, net, title, highlight_missing=None):
        """Draw a KeywordNetwork on an axes."""
        G = nx.Graph()
        for node in net.nodes:
            G.add_node(node)
        for u, v, w in net.edges:
            G.add_edge(u, v, weight=w)

        if not G.nodes():
            ax.text(
                0.5, 0.5, "데이터 없음",
                ha="center", va="center", fontproperties=fp, fontsize=14,
            )
            ax.set_title(title, fontproperties=fp, fontsize=12)
            ax.axis("off")
            return

        pos = nx.spring_layout(G, seed=42)

        # Draw edges
        edge_colors = []
        edge_styles = []
        for u, v in G.edges():
            if highlight_missing and (u, v) in highlight_missing:
                edge_colors.append("#F44336")
                edge_styles.append("dashed")
            else:
                edge_colors.append("#999999")
                edge_styles.append("solid")

        # Draw normal edges
        normal_edges = [
            (u, v) for u, v in G.edges()
            if not highlight_missing or (u, v) not in highlight_missing
        ]
        if normal_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=normal_edges,
                edge_color="#999999", alpha=0.6, ax=ax,
            )

        # Draw missing edges in red dashed
        if highlight_missing:
            missing_in_graph = [
                (u, v) for u, v in G.edges()
                if (u, v) in highlight_missing
            ]
            if missing_in_graph:
                nx.draw_networkx_edges(
                    G, pos, edgelist=missing_in_graph,
                    edge_color="#F44336", style="dashed",
                    alpha=0.8, width=2, ax=ax,
                )

        nx.draw_networkx_nodes(
            G, pos, node_color="#4CAF50", node_size=300,
            alpha=0.8, ax=ax,
        )
        _ff = _get_font_family(fp)
        _label_kw = {"font_family": _ff} if _ff else {}
        nx.draw_networkx_labels(
            G, pos, font_size=8, ax=ax, **_label_kw,
        )

        ax.set_title(title, fontproperties=fp, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Build missing edge set for lecture highlighting
    missing_set = None
    if missing_edges:
        missing_set = set()
        for u, v in missing_edges:
            missing_set.add((u, v))
            missing_set.add((v, u))

    _draw_network(ax1, textbook_net, "교과서 네트워크")
    _draw_network(ax2, lecture_net, f"{lecture_net.source}반 네트워크", missing_set)

    fig.suptitle(
        "핵심 용어 네트워크 비교",
        fontproperties=fp, fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    return save_fig(fig, dpi=dpi)


# ----------------------------------------------------------------
# T051: Delivery bar chart
# ----------------------------------------------------------------


def build_delivery_bar_chart(
    result: object,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build delivery rate bar chart showing overall + per-section rates.

    Y-axis: [0, 1], X-axis: "전체" + individual sections.
    Title: "실효 전달율"

    Args:
        result: DeliveryResult with effective_delivery_rate and
            per_section_rate.
        font_path: Path to Korean font.
        dpi: Chart resolution.

    Returns:
        BytesIO PNG buffer.
    """
    fp = _get_font_props(font_path)

    labels = ["전체"]
    values = [result.effective_delivery_rate]
    colors = ["#333333"]

    for i, (section, rate) in enumerate(
        sorted(result.per_section_rate.items())
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
    ax.set_ylabel("전달율", fontproperties=fp)
    ax.set_title("실효 전달율", fontproperties=fp, fontsize=14, fontweight="bold")

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
# T052: Delivery heatmap
# ----------------------------------------------------------------


_DELIVERY_CMAP_COLORS = {
    "충분히 설명": "#4CAF50",
    "부분 전달": "#FF9800",
    "미전달": "#F44336",
    "의도적 생략": "#9E9E9E",
}


def build_delivery_heatmap(
    result: object,
    font_path: str | None = None,
    dpi: int = 150,
    max_concepts: int = 30,
) -> io.BytesIO:
    """Build delivery quality heatmap (concepts x sections).

    Rows = domain concepts (ONLY domain terms, no stopwords).
    Columns = sections. Color: delivery_quality (0-1), RdYlGn.
    Title: "분반별 개념 전달 품질"

    Args:
        result: DeliveryResult with deliveries list.
        font_path: Path to Korean font.
        dpi: Chart resolution.
        max_concepts: Maximum number of concepts to show.

    Returns:
        BytesIO PNG buffer.
    """
    fp = _get_font_props(font_path)

    # Group deliveries by concept and section
    concept_section: dict[str, dict[str, float]] = {}
    sections: set[str] = set()
    for d in result.deliveries:
        if d.delivery_status == "의도적 생략":
            continue
        if d.concept not in concept_section:
            concept_section[d.concept] = {}
        concept_section[d.concept][d.section_id] = d.delivery_quality
        sections.add(d.section_id)

    if not concept_section:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5, "전달 분석 데이터 없음",
            ha="center", va="center", fontproperties=fp, fontsize=14,
        )
        ax.set_title(
            "분반별 개념 전달 품질",
            fontproperties=fp, fontsize=14,
        )
        return save_fig(fig, dpi=dpi)

    sections_sorted = sorted(sections)
    concepts_sorted = sorted(concept_section.keys())[:max_concepts]

    # Build matrix
    matrix = np.zeros((len(concepts_sorted), len(sections_sorted)))
    for i, concept in enumerate(concepts_sorted):
        for j, section in enumerate(sections_sorted):
            matrix[i, j] = concept_section[concept].get(section, 0.0)

    fig_height = max(4, len(concepts_sorted) * 0.4 + 2)
    fig, ax = plt.subplots(
        figsize=(max(6, len(sections_sorted) * 1.5 + 2), fig_height),
    )

    im = ax.imshow(matrix, cmap="cividis", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(sections_sorted)))
    ax.set_xticklabels(
        [f"{s}반" for s in sections_sorted],
        fontproperties=fp,
    )
    ax.set_yticks(range(len(concepts_sorted)))
    ax.set_yticklabels(concepts_sorted, fontproperties=fp, fontsize=8)

    # Add text annotations
    for i in range(len(concepts_sorted)):
        for j in range(len(sections_sorted)):
            val = matrix[i, j]
            color = "white" if val < 0.3 or val > 0.7 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=7, color=color,
            )

    ax.set_title(
        "분반별 개념 전달 품질",
        fontproperties=fp, fontsize=14, fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("전달 품질", fontproperties=fp)
    fig.tight_layout()

    return save_fig(fig, dpi=dpi)


# ----------------------------------------------------------------
# T037: Topic delivery stacked chart
# ----------------------------------------------------------------


def _aggregate_topic_delivery(
    result: object,
    hierarchy: object,
) -> dict[str, dict[str, int]]:
    """Aggregate delivery counts per major topic.

    Args:
        result: DeliveryResult with deliveries list.
        hierarchy: TopicHierarchy with section_to_major mapping.

    Returns:
        Dict mapping major_topic -> {"충분히 설명": N, "부분 전달": N, "미전달": N}.
    """
    topic_counts: dict[str, dict[str, int]] = {}

    # Build sub_topic name -> major_topic name mapping
    concept_to_major: dict[str, str] = {}
    for mt in hierarchy.major_topics:
        for st in mt.sub_topics:
            concept_to_major[st.name] = mt.name

    for d in result.deliveries:
        if d.delivery_status == "의도적 생략":
            continue

        major = ""
        for key, mt_name in concept_to_major.items():
            if key in d.concept:
                major = mt_name
                break
        if not major:
            for key, mt_name in hierarchy.section_to_major.items():
                if key in d.concept:
                    major = mt_name
                    break
        if not major:
            major = "기타"

        if major not in topic_counts:
            topic_counts[major] = {"충분히 설명": 0, "부분 전달": 0, "미전달": 0}

        status = d.delivery_status
        if status in topic_counts[major]:
            topic_counts[major][status] += 1

    return topic_counts


def build_topic_delivery_stacked_chart(
    result: object,
    hierarchy: object,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build horizontal stacked bar chart per major topic.

    3 segments: 충분히 설명 (green), 부분 전달 (orange), 미전달 (red).

    Args:
        result: DeliveryResult with deliveries list.
        hierarchy: TopicHierarchy with major_topics.
        font_path: Path to Korean font.
        dpi: Chart resolution.

    Returns:
        BytesIO PNG buffer.
    """
    fp = _get_font_props(font_path)

    topic_counts = _aggregate_topic_delivery(result, hierarchy)

    if not topic_counts:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5, "계층 전달 데이터 없음",
            ha="center", va="center", fontproperties=fp, fontsize=14,
        )
        ax.set_title("대주제별 전달 현황", fontproperties=fp, fontsize=14)
        return save_fig(fig, dpi=dpi)

    topics = list(topic_counts.keys())
    fully = [topic_counts[t].get("충분히 설명", 0) for t in topics]
    partial = [topic_counts[t].get("부분 전달", 0) for t in topics]
    not_del = [topic_counts[t].get("미전달", 0) for t in topics]

    y_pos = np.arange(len(topics))
    fig_height = max(4, len(topics) * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(y_pos, fully, color="#4CAF50", label="충분히 설명", edgecolor="white")
    ax.barh(
        y_pos, partial, left=fully,
        color="#FF9800", label="부분 전달", edgecolor="white",
    )
    left_partial = [f + p for f, p in zip(fully, partial)]
    ax.barh(
        y_pos, not_del, left=left_partial,
        color="#F44336", label="미전달", edgecolor="white",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics, fontproperties=fp)
    ax.set_xlabel("개념-분반 수", fontproperties=fp)
    ax.set_title(
        "대주제별 전달 현황",
        fontproperties=fp, fontsize=14, fontweight="bold",
    )
    if fp:
        ax.legend(prop=fp)
    else:
        ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return save_fig(fig, dpi=dpi)


# ----------------------------------------------------------------
# T038: Hierarchical coverage chart (grouped bar)
# ----------------------------------------------------------------


def _compute_topic_section_qualities(
    result: object,
    hierarchy: object,
) -> tuple[dict[str, dict[str, list[float]]], list[str]]:
    """Compute per-major-topic, per-section quality lists.

    Args:
        result: DeliveryResult with deliveries list.
        hierarchy: TopicHierarchy.

    Returns:
        (topic_section_qualities, sections_sorted) tuple.
    """
    concept_to_major: dict[str, str] = {}
    for mt in hierarchy.major_topics:
        for st in mt.sub_topics:
            concept_to_major[st.name] = mt.name
    for key, mt_name in hierarchy.section_to_major.items():
        concept_to_major[key] = mt_name

    sections: set[str] = set()
    topic_section_qualities: dict[str, dict[str, list[float]]] = {}

    for d in result.deliveries:
        if d.delivery_status == "의도적 생략":
            continue
        sections.add(d.section_id)

        major = ""
        for key, mt_name in concept_to_major.items():
            if key in d.concept:
                major = mt_name
                break
        if not major:
            major = "기타"

        if major not in topic_section_qualities:
            topic_section_qualities[major] = {}
        if d.section_id not in topic_section_qualities[major]:
            topic_section_qualities[major][d.section_id] = []
        topic_section_qualities[major][d.section_id].append(d.delivery_quality)

    return topic_section_qualities, sorted(sections)


def build_hierarchical_coverage_chart(
    result: object,
    hierarchy: object,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build grouped horizontal bar chart: major topics as groups, one bar per section.

    Args:
        result: DeliveryResult with deliveries and per_section_rate.
        hierarchy: TopicHierarchy with major_topics.
        font_path: Path to Korean font.
        dpi: Chart resolution.

    Returns:
        BytesIO PNG buffer.
    """
    fp = _get_font_props(font_path)

    topic_section_qualities, sections_sorted = _compute_topic_section_qualities(
        result, hierarchy,
    )

    if not sections_sorted or not topic_section_qualities:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5, "계층 커버리지 데이터 없음",
            ha="center", va="center", fontproperties=fp, fontsize=14,
        )
        ax.set_title("대주제별 분반 전달율", fontproperties=fp, fontsize=14)
        return save_fig(fig, dpi=dpi)

    topics = [
        mt.name for mt in hierarchy.major_topics
        if mt.name in topic_section_qualities
    ]
    if not topics:
        topics = list(topic_section_qualities.keys())

    n_sections = len(sections_sorted)
    n_topics = len(topics)
    bar_height = 0.8 / n_sections

    fig_height = max(4, n_topics * 1.2 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    for i, section in enumerate(sections_sorted):
        color = _SECTION_COLORS[i % len(_SECTION_COLORS)]
        y_positions = []
        values = []
        for j, topic in enumerate(topics):
            y_positions.append(j + i * bar_height)
            quals = topic_section_qualities.get(topic, {}).get(section, [])
            values.append(sum(quals) / len(quals) if quals else 0.0)
        ax.barh(
            y_positions, values, height=bar_height * 0.9,
            color=color, label=f"{section}반", edgecolor="white",
        )

    ax.set_yticks(
        [j + (n_sections - 1) * bar_height / 2 for j in range(n_topics)],
    )
    ax.set_yticklabels(topics, fontproperties=fp)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("평균 전달 품질", fontproperties=fp)
    ax.set_title(
        "대주제별 분반 전달율",
        fontproperties=fp, fontsize=14, fontweight="bold",
    )
    if fp:
        ax.legend(prop=fp)
    else:
        ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return save_fig(fig, dpi=dpi)


# ----------------------------------------------------------------
# T039: Grouped quality heatmap
# ----------------------------------------------------------------


def build_grouped_quality_heatmap(
    result: object,
    hierarchy: object,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build heatmap with rows grouped by major topic.

    Rows = concepts grouped by major_topic (bold separator rows).
    Columns = sections (A, B, C, D).
    Cell color = cividis colormap (colorblind safe per FR-024).
    Numeric annotations in each cell.

    Args:
        result: DeliveryResult with deliveries list.
        hierarchy: TopicHierarchy with major_topics.
        font_path: Path to Korean font.
        dpi: Chart resolution.

    Returns:
        BytesIO PNG buffer.
    """
    fp = _get_font_props(font_path)

    # Group deliveries by concept and section
    concept_section: dict[str, dict[str, float]] = {}
    sections: set[str] = set()
    for d in result.deliveries:
        if d.delivery_status == "의도적 생략":
            continue
        if d.concept not in concept_section:
            concept_section[d.concept] = {}
        concept_section[d.concept][d.section_id] = d.delivery_quality
        sections.add(d.section_id)

    if not concept_section:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5, "전달 분석 데이터 없음",
            ha="center", va="center", fontproperties=fp, fontsize=14,
        )
        ax.set_title("계층 전달 품질 히트맵", fontproperties=fp, fontsize=14)
        return save_fig(fig, dpi=dpi)

    sections_sorted = sorted(sections)

    # Build concept-to-major mapping
    concept_to_major: dict[str, str] = {}
    for mt in hierarchy.major_topics:
        for st in mt.sub_topics:
            concept_to_major[st.name] = mt.name
    for key, mt_name in hierarchy.section_to_major.items():
        concept_to_major[key] = mt_name

    # Group concepts by major topic
    topic_concepts: dict[str, list[str]] = {}
    unmatched: list[str] = []
    for concept_name in sorted(concept_section.keys()):
        major = ""
        for key, mt_name in concept_to_major.items():
            if key in concept_name:
                major = mt_name
                break
        if major:
            if major not in topic_concepts:
                topic_concepts[major] = []
            topic_concepts[major].append(concept_name)
        else:
            unmatched.append(concept_name)

    # Build ordered concept list
    ordered_concepts: list[str] = []
    group_labels: list[str] = []
    group_boundaries: list[int] = []

    for mt in hierarchy.major_topics:
        if mt.name in topic_concepts:
            group_boundaries.append(len(ordered_concepts))
            for c in topic_concepts[mt.name]:
                ordered_concepts.append(c)
                group_labels.append(c)

    if unmatched:
        group_boundaries.append(len(ordered_concepts))
        for c in unmatched:
            ordered_concepts.append(c)
            group_labels.append(c)

    if not ordered_concepts:
        ordered_concepts = sorted(concept_section.keys())
        group_labels = list(ordered_concepts)

    # Build matrix
    matrix = np.zeros((len(ordered_concepts), len(sections_sorted)))
    for i, concept in enumerate(ordered_concepts):
        for j, section in enumerate(sections_sorted):
            matrix[i, j] = concept_section.get(concept, {}).get(section, 0.0)

    fig_height = max(4, len(ordered_concepts) * 0.4 + 2)
    fig, ax = plt.subplots(
        figsize=(max(6, len(sections_sorted) * 1.5 + 3), fig_height),
    )

    im = ax.imshow(matrix, cmap="cividis", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(sections_sorted)))
    ax.set_xticklabels(
        [f"{s}반" for s in sections_sorted],
        fontproperties=fp,
    )
    ax.set_yticks(range(len(group_labels)))
    ax.set_yticklabels(group_labels, fontproperties=fp, fontsize=8)

    # Bold group separator lines
    for boundary in group_boundaries[1:]:
        ax.axhline(y=boundary - 0.5, color="black", linewidth=1.5)

    # Numeric annotations
    for i in range(len(ordered_concepts)):
        for j in range(len(sections_sorted)):
            val = matrix[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=7, color=color,
            )

    ax.set_title(
        "계층 전달 품질 히트맵",
        fontproperties=fp, fontsize=14, fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("전달 품질", fontproperties=fp)
    fig.tight_layout()

    return save_fig(fig, dpi=dpi)


# ----------------------------------------------------------------
# T054: Concept network chart
# ----------------------------------------------------------------


_IMPORTANCE_SIZE = {"high": 400, "medium": 250, "low": 150}
_EDGE_TYPE_COLORS = {"shared_terms": "#2196F3", "semantic": "#9C27B0"}


def build_concept_network_chart(
    network: object,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build concept network graph colored by delivery quality.

    Node color: cividis colormap based on delivery_quality (0-1).
    Node size: importance (high=400, medium=250, low=150).
    Edge color: shared_terms=#2196F3, semantic=#9C27B0.
    Edge width: weight * 2.5.
    Labels: concept name truncated to 15 chars.

    Args:
        network: ConceptNetwork with nodes and edges.
        font_path: Path to Korean font.
        dpi: Chart resolution.

    Returns:
        BytesIO PNG buffer.
    """
    import networkx as nx

    fp = _get_font_props(font_path)

    G = nx.Graph()
    for node in network.nodes:
        G.add_node(
            node.concept,
            delivery_quality=node.delivery_quality,
            importance=node.importance,
        )
    for edge in network.edges:
        G.add_edge(
            edge.source, edge.target,
            relationship=edge.relationship,
            weight=edge.weight,
        )

    fig, ax = plt.subplots(figsize=(10, 8))

    if not G.nodes():
        ax.text(
            0.5, 0.5, "네트워크 데이터 없음",
            ha="center", va="center", fontproperties=fp, fontsize=14,
        )
        ax.axis("off")
        return save_fig(fig, dpi=dpi)

    pos = nx.kamada_kawai_layout(G)

    # Node colors (cividis) and sizes
    cmap = plt.cm.cividis
    node_colors = [
        cmap(G.nodes[n].get("delivery_quality", 0.0)) for n in G.nodes()
    ]
    node_sizes = [
        _IMPORTANCE_SIZE.get(G.nodes[n].get("importance", "low"), 150)
        for n in G.nodes()
    ]

    # Draw edges by type
    for rel_type, color in _EDGE_TYPE_COLORS.items():
        edge_list = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relationship") == rel_type
        ]
        if edge_list:
            widths = [G[u][v]["weight"] * 2.5 for u, v in edge_list]
            nx.draw_networkx_edges(
                G, pos, edgelist=edge_list,
                edge_color=color, width=widths, alpha=0.6, ax=ax,
            )

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        alpha=0.85, ax=ax, edgecolors="white", linewidths=0.5,
    )

    # Labels truncated to 15 chars
    labels = {n: n[:15] for n in G.nodes()}
    _ff = _get_font_family(fp)
    _label_kw = {"font_family": _ff} if _ff else {}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax, **_label_kw)

    ax.set_title(
        "개념 네트워크 그래프",
        fontproperties=fp, fontsize=14, fontweight="bold",
    )
    ax.axis("off")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("전달 품질", fontproperties=fp)
    fig.tight_layout()

    return save_fig(fig, dpi=dpi)


# ----------------------------------------------------------------
# T055: Concept network comparison chart
# ----------------------------------------------------------------


def build_concept_network_comparison(
    network: object,
    deliveries_by_section: dict[str, list],
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build concept network comparison across sections.

    Creates a subplot grid with the same topology but different node
    colors per section based on delivery quality overlay.

    Args:
        network: Base ConceptNetwork (topology).
        deliveries_by_section: {section_id: [DeliveryAnalysis, ...]}.
        font_path: Path to Korean font.
        dpi: Chart resolution.

    Returns:
        BytesIO PNG buffer.
    """
    import networkx as nx
    from forma.concept_network import overlay_delivery

    fp = _get_font_props(font_path)
    _ff = _get_font_family(fp)
    sections = sorted(deliveries_by_section.keys())
    n_sections = len(sections)

    if n_sections == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5, 0.5, "비교 데이터 없음",
            ha="center", va="center", fontproperties=fp, fontsize=14,
        )
        ax.axis("off")
        return save_fig(fig, dpi=dpi)

    # Grid layout: 2x2 for 4, 1xN for < 4
    if n_sections <= 2:
        nrows, ncols = 1, n_sections
    else:
        nrows, ncols = 2, 2

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows),
    )
    if n_sections == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else list(axes)

    # Build base graph for consistent layout
    G = nx.Graph()
    for node in network.nodes:
        G.add_node(node.concept, importance=node.importance)
    for edge in network.edges:
        G.add_edge(
            edge.source, edge.target,
            relationship=edge.relationship,
            weight=edge.weight,
        )

    pos = nx.kamada_kawai_layout(G) if G.nodes() else {}
    cmap = plt.cm.cividis

    for idx, section in enumerate(sections):
        ax = axes[idx]
        overlaid = overlay_delivery(
            network, deliveries_by_section[section], section,
        )

        if not G.nodes():
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
            ax.set_title(f"{section}반", fontproperties=fp, fontsize=12)
            ax.axis("off")
            continue

        # Node colors from overlaid delivery quality
        quality_map = {n.concept: n.delivery_quality for n in overlaid.nodes}
        node_colors = [cmap(quality_map.get(n, 0.0)) for n in G.nodes()]
        node_sizes = [
            _IMPORTANCE_SIZE.get(G.nodes[n].get("importance", "low"), 150)
            for n in G.nodes()
        ]

        # Draw edges
        for rel_type, color in _EDGE_TYPE_COLORS.items():
            edge_list = [
                (u, v) for u, v, d in G.edges(data=True)
                if d.get("relationship") == rel_type
            ]
            if edge_list:
                widths = [G[u][v]["weight"] * 2.5 for u, v in edge_list]
                nx.draw_networkx_edges(
                    G, pos, edgelist=edge_list,
                    edge_color=color, width=widths, alpha=0.6, ax=ax,
                )

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_sizes,
            alpha=0.85, ax=ax, edgecolors="white", linewidths=0.5,
        )

        labels = {n: n[:15] for n in G.nodes()}
        _label_kw = {"font_family": _ff} if _ff else {}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax, **_label_kw)

        ax.set_title(
            f"{section}반", fontproperties=fp, fontsize=12, fontweight="bold",
        )
        ax.axis("off")

    # Hide unused subplots
    for idx in range(n_sections, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        "분반별 개념 네트워크 비교",
        fontproperties=fp, fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    return save_fig(fig, dpi=dpi)
