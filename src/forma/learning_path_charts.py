"""Chart generation for learning path DAG and class deficit map.

Produces DAG visualization charts as PNG BytesIO buffers for embedding
in ReportLab PDFs. Uses matplotlib Agg backend and networkx graph drawing.

Functions:
    build_learning_path_chart: Student learning path DAG chart.
    build_deficit_map_chart: Class-wide deficit map DAG chart.
"""

from __future__ import annotations

import io
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

from forma.chart_utils import save_fig as _save_fig  # noqa: E402
from forma.concept_dependency import ConceptDependencyDAG  # noqa: E402
from forma.font_utils import find_korean_font  # noqa: E402
from forma.learning_path import ClassDeficitMap, LearningPath  # noqa: E402

logger = logging.getLogger(__name__)

_COLOR_MASTERED = "#C8E6C9"  # light green
_COLOR_DEFICIT = "#FFCDD2"  # light red
_COLOR_IN_PATH = "#EF5350"  # red
_COLOR_DEFAULT = "#E0E0E0"  # grey
_EDGE_COLOR = "#757575"


def build_learning_path_chart(
    learning_path: LearningPath,
    dag: ConceptDependencyDAG,
    dpi: int = 150,
) -> io.BytesIO:
    """Build a DAG visualization highlighting the student's learning path.

    Deficit concepts are highlighted in red, mastered in green,
    path order is shown via node numbering.

    Args:
        learning_path: Student's learning path with deficit/ordered concepts.
        dag: The concept dependency DAG.
        dpi: Chart resolution.

    Returns:
        PNG image as BytesIO buffer.
    """
    font_path = find_korean_font()
    font_prop = FontProperties(fname=font_path)

    g = dag.graph
    fig, ax = plt.subplots(1, 1, figsize=(8, max(4, len(g.nodes) * 0.4)))
    ax.set_title("추천 학습 경로", fontproperties=font_prop, fontsize=14)

    if len(g.nodes) == 0:
        ax.text(0.5, 0.5, "개념 의존성 없음", ha="center", va="center", fontproperties=font_prop, fontsize=12)
        ax.set_axis_off()
        buf = _save_fig(fig, dpi=dpi)
        return buf

    deficit_set = set(learning_path.deficit_concepts)
    path_set = set(learning_path.ordered_path)

    node_colors = []
    for node in g.nodes:
        if node in path_set:
            node_colors.append(_COLOR_IN_PATH)
        elif node in deficit_set:
            node_colors.append(_COLOR_DEFICIT)
        else:
            node_colors.append(_COLOR_MASTERED)

    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except (ImportError, Exception):
        pos = nx.spring_layout(g, seed=42)

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=800, edgecolors="#424242", linewidths=1.0)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=_EDGE_COLOR, arrows=True, arrowsize=15)

    # Labels with path order numbers for deficit concepts
    labels = {}
    for i, concept in enumerate(learning_path.ordered_path, 1):
        labels[concept] = f"{i}. {concept}"
    for node in g.nodes:
        if node not in labels:
            labels[node] = node

    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=8, font_family=font_prop.get_name())

    ax.set_axis_off()
    buf = _save_fig(fig, dpi=dpi)
    return buf


def build_deficit_map_chart(
    deficit_map: ClassDeficitMap,
    dpi: int = 150,
) -> io.BytesIO:
    """Build a class-wide deficit map DAG with per-concept deficit counts.

    Node color intensity proportional to deficit ratio.

    Args:
        deficit_map: ClassDeficitMap with concept counts and DAG.
        dpi: Chart resolution.

    Returns:
        PNG image as BytesIO buffer.
    """
    font_path = find_korean_font()
    font_prop = FontProperties(fname=font_path)

    dag = deficit_map.dag
    g = dag.graph
    total = max(deficit_map.total_students, 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, max(4, len(g.nodes) * 0.4)))
    ax.set_title("학급 개념 결손 맵", fontproperties=font_prop, fontsize=14)

    if len(g.nodes) == 0:
        ax.text(0.5, 0.5, "개념 의존성 없음", ha="center", va="center", fontproperties=font_prop, fontsize=12)
        ax.set_axis_off()
        buf = _save_fig(fig, dpi=dpi)
        return buf

    # Color by deficit ratio: higher ratio → more red
    node_colors = []
    for node in g.nodes:
        count = deficit_map.concept_counts.get(node, 0)
        ratio = count / total
        if ratio >= 0.5:
            node_colors.append("#EF5350")  # red
        elif ratio >= 0.3:
            node_colors.append("#FFA726")  # orange
        elif ratio > 0:
            node_colors.append("#FFF176")  # yellow
        else:
            node_colors.append(_COLOR_MASTERED)  # green

    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except (ImportError, Exception):
        pos = nx.spring_layout(g, seed=42)

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=800, edgecolors="#424242", linewidths=1.0)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=_EDGE_COLOR, arrows=True, arrowsize=15)

    # Labels with count
    labels = {}
    for node in g.nodes:
        count = deficit_map.concept_counts.get(node, 0)
        labels[node] = f"{node}\n({count}/{total})"

    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7, font_family=font_prop.get_name())

    ax.set_axis_off()
    buf = _save_fig(fig, dpi=dpi)
    return buf
