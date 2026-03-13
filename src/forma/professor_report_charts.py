"""Chart generation for professor class summary PDF reports.

Produces 4 chart types as PNG io.BytesIO buffers for embedding in ReportLab PDFs.
Uses matplotlib Agg backend — no display server required. No LLM API calls.
"""
from __future__ import annotations

import io
import logging

import matplotlib

matplotlib.use("Agg")  # MUST be before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

from forma.chart_utils import save_fig as _save_fig  # noqa: E402
from forma.font_utils import find_korean_font  # noqa: E402
from forma.professor_report_data import QuestionClassStats  # noqa: E402
from forma.report_charts import _LEVEL_COLORS  # noqa: E402

# _LEVEL_COLORS = {"Advanced": "#2E7D32", "Proficient": "#1565C0",
#                  "Developing": "#F57F17", "Beginning": "#C62828"}

logger = logging.getLogger(__name__)

_LEVEL_ORDER = ["Advanced", "Proficient", "Developing", "Beginning"]
_LEVEL_KOREAN = {
    "Advanced": "상",
    "Proficient": "중상",
    "Developing": "중하",
    "Beginning": "하",
}


class ProfessorReportChartGenerator:
    """Generate matplotlib charts as PNG BytesIO for professor report PDF embedding.

    Args:
        font_path: Path to Korean .ttf font. Auto-detected if None.
        dpi: Resolution for chart images (default 150).
    """

    def __init__(self, font_path: str | None = None, dpi: int = 150) -> None:
        if font_path is None:
            font_path = find_korean_font()
        self._font_path = font_path
        self._font_prop = FontProperties(fname=font_path)
        self._dpi = dpi

    def score_histogram(self, scores: list[float], bins: int = 10) -> io.BytesIO:
        """Histogram of class ensemble scores with mean and median lines.

        Args:
            scores: List of class ensemble scores (may be empty).
            bins: Number of histogram bins (default 10).

        Returns:
            PNG image as BytesIO.
        """
        fig, ax = plt.subplots(figsize=(120 / 25.4, 80 / 25.4))

        if not scores:
            ax.text(
                0.5,
                0.5,
                "데이터 없음",
                ha="center",
                va="center",
                fontproperties=self._font_prop,
                fontsize=12,
            )
            ax.set_xlim(0, 1.0)
            return _save_fig(fig, dpi=self._dpi)

        scores_arr = np.array(scores, dtype=float)
        mean_val = float(np.mean(scores_arr))
        median_val = float(np.median(scores_arr))

        if len(scores_arr) == 1 or np.std(scores_arr) == 0:
            # Single-value bar: just show a bar at the unique value
            unique_val = scores_arr[0]
            ax.bar([unique_val], [len(scores_arr)], width=0.05, color="#1565C0", alpha=0.7)
        else:
            ax.hist(scores_arr, bins=bins, color="#1565C0", alpha=0.7, edgecolor="white")

        ax.axvline(mean_val, color="orange", linestyle="--", label="평균")
        ax.axvline(median_val, color="blue", linestyle=":", label="중앙값")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("점수", fontproperties=self._font_prop, fontsize=9)
        ax.set_ylabel("학생 수", fontproperties=self._font_prop, fontsize=9)
        ax.set_title("점수 분포", fontproperties=self._font_prop, fontsize=11)
        ax.legend(prop=self._font_prop, fontsize=8)

        fig.tight_layout()
        return _save_fig(fig, dpi=self._dpi)

    def level_donut(self, level_dist: dict[str, int]) -> io.BytesIO:
        """Proportional donut chart of understanding level distribution.

        Args:
            level_dist: {"Advanced": n, "Proficient": n, "Developing": n, "Beginning": n}

        Returns:
            PNG image as BytesIO.
        """
        fig, ax = plt.subplots(figsize=(100 / 25.4, 100 / 25.4))

        total = sum(level_dist.values())
        if total == 0:
            ax.text(
                0.5,
                0.5,
                "데이터 없음",
                ha="center",
                va="center",
                fontproperties=self._font_prop,
                fontsize=12,
            )
            ax.axis("off")
            return _save_fig(fig, dpi=self._dpi)

        levels = _LEVEL_ORDER
        counts = [level_dist.get(lvl, 0) for lvl in levels]
        colors = [_LEVEL_COLORS.get(lvl, "#888888") for lvl in levels]

        # Filter out zero-count levels for wedge display but keep proportional
        nonzero_levels = [(lvl, cnt, col) for lvl, cnt, col in zip(levels, counts, colors) if cnt > 0]
        nz_counts = [x[1] for x in nonzero_levels]
        nz_colors = [x[2] for x in nonzero_levels]
        nz_labels = []
        for lvl, cnt, col in nonzero_levels:
            pct = cnt / total * 100
            kr = _LEVEL_KOREAN.get(lvl, lvl)
            nz_labels.append(f"{kr}\n{cnt}명 ({pct:.0f}%)")

        wedges, texts = ax.pie(
            nz_counts,
            labels=nz_labels,
            colors=nz_colors,
            wedgeprops=dict(width=0.4),
            startangle=90,
            textprops={"fontproperties": self._font_prop, "fontsize": 8},
        )

        ax.set_title("이해 수준 분포", fontproperties=self._font_prop, fontsize=11)

        fig.tight_layout()
        return _save_fig(fig, dpi=self._dpi)

    def question_difficulty_bar(self, stats: list[QuestionClassStats]) -> io.BytesIO:
        """Horizontal bar chart of per-question ensemble mean scores.

        Args:
            stats: List of QuestionClassStats with ensemble_mean values.

        Returns:
            PNG image as BytesIO.
        """
        if not stats:
            fig, ax = plt.subplots(figsize=(100 / 25.4, 60 / 25.4))
            ax.text(
                0.5,
                0.5,
                "데이터 없음",
                ha="center",
                va="center",
                fontproperties=self._font_prop,
                fontsize=12,
            )
            ax.axis("off")
            return _save_fig(fig, dpi=self._dpi)

        n = len(stats)
        fig_w = 160 / 25.4  # 160mm — matches PDF embedding width, no horizontal stretch
        figsize = (fig_w, max(60 / 25.4, n * 20 / 25.4))
        fig, ax = plt.subplots(figsize=figsize)

        labels = [f"Q{stat.question_sn}" for stat in stats]
        means = [stat.ensemble_mean for stat in stats]

        bar_colors = []
        for mean in means:
            if mean >= 0.85:
                bar_colors.append(_LEVEL_COLORS["Advanced"])
            elif mean >= 0.65:
                bar_colors.append(_LEVEL_COLORS["Proficient"])
            elif mean >= 0.45:
                bar_colors.append(_LEVEL_COLORS["Developing"])
            else:
                bar_colors.append(_LEVEL_COLORS["Beginning"])

        positions = list(range(n))
        bars = ax.barh(positions, means, color=bar_colors, height=0.6)

        # Value labels at end of each bar
        for bar, mean in zip(bars, means):
            ax.text(
                mean + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{mean:.2f}",
                va="center",
                fontproperties=self._font_prop,
                fontsize=8,
            )

        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontproperties=self._font_prop, fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("앙상블 점수", fontproperties=self._font_prop, fontsize=9)
        ax.set_title("문항별 난이도", fontproperties=self._font_prop, fontsize=11)
        ax.invert_yaxis()

        fig.tight_layout()
        return _save_fig(fig, dpi=self._dpi)

    def concept_mastery_heatmap(self, mastery_data: dict[int, dict[str, float]]) -> io.BytesIO:
        """Heatmap of concept mastery rates across questions.

        Args:
            mastery_data: {question_sn: {concept_name: mastery_rate (0.0-1.0)}}

        Returns:
            BytesIO containing PNG image.
        """
        if not mastery_data:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "개념 데이터 없음", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            return _save_fig(fig, dpi=self._dpi)

        # Collect all concepts across all questions
        all_concepts = sorted({
            concept
            for q_data in mastery_data.values()
            for concept in q_data
        })
        question_sns = sorted(mastery_data.keys())

        if not all_concepts:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "개념 데이터 없음", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            return _save_fig(fig, dpi=self._dpi)

        # Build matrix
        matrix = np.array([
            [mastery_data[sn].get(c, 0.0) for c in all_concepts]
            for sn in question_sns
        ])

        fig_width = 160 / 25.4   # 160mm — matches PDF embedding width
        fig_height = max(2, len(all_concepts) * 0.5 + 1)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        im = ax.imshow(matrix.T, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, fraction=0.02)

        fp = self._font_prop
        ax.set_xticks(range(len(question_sns)))
        ax.set_xticklabels([f"Q{sn}" for sn in question_sns], fontproperties=fp)
        ax.set_yticks(range(len(all_concepts)))
        ax.set_yticklabels(all_concepts, fontproperties=fp)
        ax.set_title("개념 숙달도 히트맵", fontproperties=fp)

        plt.tight_layout()
        return _save_fig(fig, dpi=self._dpi)

    def student_rank_lollipop(self, rows: list, highlight_at_risk: bool = True) -> io.BytesIO:
        """Horizontal lollipop chart of student scores sorted by rank.

        For > 50 students: shows top 25 and bottom 25 with gap indicator.
        At-risk students highlighted in red.
        """
        if not rows:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(0.5, 0.5, "학생 데이터 없음", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            return _save_fig(fig, dpi=self._dpi)

        sorted_rows = sorted(rows, key=lambda r: r.overall_ensemble_mean, reverse=True)

        # Truncate for large datasets
        gap_marker = None
        if len(sorted_rows) > 50:
            top = sorted_rows[:25]
            bottom = sorted_rows[-25:]
            gap_marker = 25  # index where gap appears
            display_rows = top + bottom
        else:
            display_rows = sorted_rows

        scores = [r.overall_ensemble_mean for r in display_rows]
        names = [r.real_name for r in display_rows]
        at_risk_flags = [getattr(r, "is_at_risk", False) for r in display_rows]

        fig_w = 160 / 25.4   # 160mm — matches PDF embedding width
        fig_height = max(2, len(display_rows) * 0.18 + 0.5)  # ~4.6mm per row
        fig, ax = plt.subplots(figsize=(fig_w, fig_height))

        fp = self._font_prop
        y_positions = list(range(len(display_rows)))

        for i, (score, name, is_at_risk) in enumerate(zip(scores, names, at_risk_flags)):
            color = "red" if (highlight_at_risk and is_at_risk) else "steelblue"
            ax.hlines(i, 0, score, colors=color, linewidth=1.5, alpha=0.7)
            ax.plot(score, i, "o", color=color, markersize=6)

        # Gap indicator
        if gap_marker is not None:
            ax.axhline(y=gap_marker - 0.5, color="gray", linestyle="--", alpha=0.5)
            ax.text(0.5, gap_marker - 0.5, "···", ha="center", va="center",
                    fontproperties=fp, fontsize=12, color="gray")

        ax.set_yticks(y_positions)
        ax.set_yticklabels(names, fontproperties=fp, fontsize=8)
        ax.set_xlabel("종합 점수", fontproperties=fp)
        ax.set_title("학생 순위 차트", fontproperties=fp)
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()

        plt.tight_layout()
        return _save_fig(fig, dpi=self._dpi)

    def question_level_stacked_bar(self, level_dist: dict[str, int], question_sn: int) -> io.BytesIO:
        """Horizontal stacked bar of level distribution for one question.

        Args:
            level_dist: {"Advanced": N, "Proficient": N, "Developing": N, "Beginning": N}
            question_sn: Question serial number (for title)
        """
        LEVELS = ["Beginning", "Developing", "Proficient", "Advanced"]
        total = sum(level_dist.get(lv, 0) for lv in LEVELS)

        fig, ax = plt.subplots(figsize=(6, 1.5))
        fp = self._font_prop

        if total == 0:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                    transform=ax.transAxes, fontproperties=fp)
            ax.axis("off")
            ax.set_title(f"Q{question_sn} 수준 분포", fontproperties=fp)
            return _save_fig(fig, dpi=self._dpi)

        left = 0.0
        for level in LEVELS:
            count = level_dist.get(level, 0)
            pct = count / total
            color = _LEVEL_COLORS.get(level, "#CCCCCC")
            ax.barh(0, pct, left=left, color=color, height=0.5, label=level)
            if pct > 0.05:
                ax.text(left + pct / 2, 0, f"{pct:.0%}", ha="center", va="center",
                        fontsize=8, fontproperties=fp, color="white" if pct > 0.1 else "black")
            left += pct

        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("비율", fontproperties=fp)
        ax.set_title(f"Q{question_sn} 수준 분포", fontproperties=fp)
        ax.legend(loc="upper right", prop=fp, fontsize=7)

        plt.tight_layout()
        return _save_fig(fig, dpi=self._dpi)

    def build_class_knowledge_graph_chart(
        self,
        aggregate: object,
        min_ratio_to_show: float = 0.05,
    ) -> io.BytesIO:
        """Build a directed knowledge graph chart from class aggregate data.

        Filters edges below min_ratio_to_show, then renders a NetworkX
        directed graph with edge widths and colors based on correct_ratio.

        Color rules (FR-006):
            - correct_ratio > 0.5: green (#2E7D32)
            - 0.2 <= correct_ratio <= 0.5: orange (#F57F17)
            - error_count > missing_count: red (#C62828)
            - else: grey (#9E9E9E), dashed

        Args:
            aggregate: ClassKnowledgeAggregate instance.
            min_ratio_to_show: Minimum correct_ratio to include an edge
                in the chart (default 0.05). Edges below this are filtered.

        Returns:
            PNG image as io.BytesIO.
        """
        import networkx as nx

        display_edges = [
            e for e in aggregate.edges
            if e.correct_ratio >= min_ratio_to_show
        ]

        fig, ax = plt.subplots(figsize=(160 / 25.4, 100 / 25.4))
        fp = self._font_prop

        if not display_edges:
            ax.text(
                0.5, 0.5, "표시할 데이터 없음",
                ha="center", va="center",
                fontproperties=fp, fontsize=12,
            )
            ax.axis("off")
            return _save_fig(fig, dpi=self._dpi)

        G = nx.DiGraph()
        edge_colors = []
        edge_widths = []
        edge_styles = []

        for e in display_edges:
            G.add_edge(e.subject, e.obj, label=e.relation)

            if e.correct_ratio > 0.5:
                edge_colors.append("#2E7D32")
                edge_styles.append("solid")
            elif e.correct_ratio >= 0.2:
                edge_colors.append("#F57F17")
                edge_styles.append("solid")
            elif e.error_count > e.missing_count:
                edge_colors.append("#C62828")
                edge_styles.append("solid")
            else:
                edge_colors.append("#9E9E9E")
                edge_styles.append("dashed")

            edge_widths.append(max(0.5, e.correct_ratio * 5))

        pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color="#E3F2FD", node_size=800, edgecolors="#1565C0",
        )

        for i, (u, v) in enumerate(G.edges()):
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=[(u, v)],
                edge_color=[edge_colors[i]],
                width=edge_widths[i],
                style=edge_styles[i],
                arrows=True,
                arrowsize=15,
                connectionstyle="arc3,rad=0.1",
            )

        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_family=fp.get_name() if hasattr(fp, "get_name") else "sans-serif",
            font_size=9,
        )

        edge_labels = {
            (e.subject, e.obj): f"{e.relation}\n({e.correct_ratio:.0%})"
            for e in display_edges
        }
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=7,
            font_family=fp.get_name() if hasattr(fp, "get_name") else "sans-serif",
        )

        ax.set_title(
            f"학급 지식 지도 — 문제 {aggregate.question_sn}",
            fontproperties=fp, fontsize=11,
        )
        ax.axis("off")
        fig.tight_layout()
        return _save_fig(fig, dpi=self._dpi)
