"""Directed graph visualizer for master vs student knowledge graph comparison."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

from src.evaluation_types import TripletEdge  # noqa: E402
from src.font_utils import find_korean_font  # noqa: E402


class GraphVisualizer:
    """Superimposes master and student knowledge graphs with color-coded edges."""

    def __init__(self, font_path: Optional[str] = None) -> None:
        self._font_path = font_path or find_korean_font()
        self._font_prop = FontProperties(fname=self._font_path)

    def visualize_comparison(
        self,
        master_edges: list[TripletEdge],
        student_edges: list[TripletEdge],
        matched_edges: list[TripletEdge],
        missing_edges: list[TripletEdge],
        extra_edges: list[TripletEdge],
        wrong_direction_edges: list[TripletEdge],
        output_path: str,
        title: str = "",
    ) -> str:
        """Render a comparison graph and save as PNG.

        Returns:
            Absolute path to the saved PNG file.
        """
        G = nx.DiGraph()

        edge_styles: list[tuple[str, str, dict]] = []

        for e in matched_edges:
            edge_styles.append(
                (e.subject, e.object, {"label": e.relation, "color": "green", "style": "solid"})
            )
        for e in wrong_direction_edges:
            edge_styles.append(
                (e.subject, e.object, {"label": e.relation, "color": "red", "style": "dashed"})
            )
        for e in missing_edges:
            edge_styles.append(
                (e.subject, e.object, {"label": e.relation, "color": "gray", "style": "dashed"})
            )
        for e in extra_edges:
            edge_styles.append(
                (e.subject, e.object, {"label": e.relation, "color": "orange", "style": "dashed"})
            )

        for src, dst, attrs in edge_styles:
            G.add_edge(src, dst, **attrs)

        fig, ax = plt.subplots(figsize=(12, 8))

        if len(G.nodes) == 0:
            ax.text(0.5, 0.5, "No edges to display", ha="center", va="center", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            pos = nx.spring_layout(G, seed=42)

            for src, dst, attrs in G.edges(data=True):
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(src, dst)],
                    edge_color=attrs.get("color", "black"),
                    style=attrs.get("style", "solid"),
                    ax=ax,
                    arrows=True,
                    arrowsize=15,
                )

            nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightblue", node_size=600)

            for node, (x, y) in pos.items():
                ax.text(
                    x,
                    y,
                    node,
                    fontproperties=self._font_prop,
                    ha="center",
                    va="center",
                    fontsize=9,
                )

            edge_labels = {(src, dst): attrs["label"] for src, dst, attrs in G.edges(data=True)}
            for (src, dst), label in edge_labels.items():
                x = (pos[src][0] + pos[dst][0]) / 2
                y = (pos[src][1] + pos[dst][1]) / 2
                ax.text(
                    x,
                    y,
                    label,
                    fontproperties=self._font_prop,
                    ha="center",
                    va="center",
                    fontsize=7,
                )

        if title:
            ax.set_title(title, fontproperties=self._font_prop, fontsize=14)

        ax.axis("off")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return str(out.resolve())
