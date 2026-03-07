"""Chart generation for student individual PDF reports.

Produces five chart types as PNG ``io.BytesIO`` buffers for embedding
in ReportLab PDFs.  Uses matplotlib Agg backend — no display server
required.  No LLM API calls.
"""

from __future__ import annotations

import io
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

from forma.font_utils import find_korean_font  # noqa: E402

logger = logging.getLogger(__name__)

# Understanding level → colour mapping (consistent with report_generator.py)
_LEVEL_COLORS = {
    "Advanced": "#2E7D32",
    "Proficient": "#1565C0",
    "Developing": "#F57F17",
    "Beginning": "#C62828",
}


class ReportChartGenerator:
    """Generate matplotlib charts as PNG BytesIO for PDF embedding.

    Args:
        font_path: Path to Korean .ttf font.  Auto-detected if None.
        dpi: Resolution for chart images (default 150).
    """

    def __init__(self, font_path: str | None = None, dpi: int = 150) -> None:
        if font_path is None:
            font_path = find_korean_font()
        self._font_path = font_path
        self._font_prop = FontProperties(fname=font_path)
        self._dpi = dpi

    def _save_fig(self, fig: plt.Figure) -> io.BytesIO:
        """Save figure to BytesIO as PNG and close it."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    def score_boxplot(
        self,
        scores: list[float],
        student_score: float,
        title: str = "",
    ) -> io.BytesIO:
        """Create a horizontal box-whisker chart with student marker.

        Args:
            scores: All student scores for the class.
            student_score: This student's score (red diamond).
            title: Optional chart title.

        Returns:
            PNG image as BytesIO.
        """
        fig, ax = plt.subplots(figsize=(160 / 25.4, 50 / 25.4))

        # Handle zero variance
        scores_arr = np.array(scores, dtype=float)
        if len(scores_arr) < 2 or np.std(scores_arr) == 0:
            whis = 0
        else:
            whis = 1.5

        bp = ax.boxplot(
            scores_arr,
            vert=False,
            widths=0.5,
            patch_artist=True,
            whis=whis,
        )

        # Style the box
        for patch in bp["boxes"]:
            patch.set_facecolor("#CCE5FF")
        for median in bp["medians"]:
            median.set_color("orange")
            median.set_linewidth(2)

        # Student marker (red diamond)
        ax.plot(
            student_score,
            1,
            marker="D",
            color="red",
            markersize=10,
            zorder=5,
        )

        # Axis range
        all_vals = list(scores) + [student_score]
        max_val = max(all_vals) if all_vals else 1.0
        ax.set_xlim(0, max(1.0, max_val * 1.1))

        if title:
            ax.set_title(title, fontproperties=self._font_prop, fontsize=11)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

        fig.tight_layout()
        return self._save_fig(fig)

    def component_comparison(
        self,
        distributions: dict[str, list[float]],
        student_scores: dict[str, float],
        question_sn: int,
    ) -> io.BytesIO:
        """Create grouped horizontal box-whisker for evaluation components.

        Args:
            distributions: {component_name: [all_scores]}.
            student_scores: {component_name: student_score}.
            question_sn: Question number for title.

        Returns:
            PNG image as BytesIO.
        """
        components = list(distributions.keys())
        if not components:
            # Empty chart
            fig, ax = plt.subplots(figsize=(160 / 25.4, 30 / 25.4))
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                    fontproperties=self._font_prop)
            return self._save_fig(fig)

        n = len(components)
        fig, ax = plt.subplots(
            figsize=(160 / 25.4, max(80 / 25.4, n * 25 / 25.4)),
        )

        data = [distributions[c] for c in components]
        positions = list(range(1, n + 1))

        bp = ax.boxplot(
            data,
            vert=False,
            positions=positions,
            widths=0.5,
            patch_artist=True,
        )

        for patch in bp["boxes"]:
            patch.set_facecolor("#CCE5FF")
        for median in bp["medians"]:
            median.set_color("orange")
            median.set_linewidth(2)

        # Student markers
        for i, comp in enumerate(components):
            if comp in student_scores:
                ax.plot(
                    student_scores[comp],
                    i + 1,
                    marker="D",
                    color="red",
                    markersize=8,
                    zorder=5,
                )

        # Korean labels
        labels = [_translate_component(c) for c in components]
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontproperties=self._font_prop, fontsize=9)
        ax.set_xlim(0, 1.1)

        title = f"문항 {question_sn} 평가 구성요소 비교"
        ax.set_title(title, fontproperties=self._font_prop, fontsize=11)

        fig.tight_layout()
        return self._save_fig(fig)

    def concept_coverage_bar(
        self,
        concepts: list,
    ) -> io.BytesIO:
        """Create horizontal bar chart for concept coverage.

        Args:
            concepts: List of ConceptDetail (or dicts with concept,
                similarity, threshold, is_present fields).

        Returns:
            PNG image as BytesIO.
        """
        if not concepts:
            fig, ax = plt.subplots(figsize=(160 / 25.4, 30 / 25.4))
            ax.text(0.5, 0.5, "개념 데이터 없음", ha="center", va="center",
                    fontproperties=self._font_prop)
            return self._save_fig(fig)

        n = len(concepts)
        height = max(30, n * 20) / 25.4  # mm to inches
        fig, ax = plt.subplots(figsize=(160 / 25.4, height))

        names = []
        similarities = []
        colors = []
        thresholds = []

        for c in concepts:
            if hasattr(c, "concept"):
                names.append(c.concept)
                similarities.append(c.similarity)
                colors.append("#2E7D32" if c.is_present else "#C62828")
                thresholds.append(c.threshold)
            else:
                names.append(c.get("concept", ""))
                similarities.append(c.get("similarity", 0.0))
                is_p = c.get("is_present", False)
                colors.append("#2E7D32" if is_p else "#C62828")
                thresholds.append(c.get("threshold", 0.0))

        positions = list(range(n))
        ax.barh(positions, similarities, color=colors, height=0.6)

        # Threshold dashed line (use average threshold)
        if thresholds:
            avg_threshold = sum(thresholds) / len(thresholds)
            ax.axvline(
                x=avg_threshold,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label=f"임계값 ({avg_threshold:.2f})",
            )

        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontproperties=self._font_prop, fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("유사도", fontproperties=self._font_prop, fontsize=9)
        ax.set_title(
            "개념별 커버리지",
            fontproperties=self._font_prop,
            fontsize=11,
        )
        ax.legend(prop=self._font_prop, fontsize=8)
        ax.invert_yaxis()

        fig.tight_layout()
        return self._save_fig(fig)

    def understanding_badge(
        self,
        level: str,
        score: float,
    ) -> io.BytesIO:
        """Create a colored badge showing understanding level.

        Args:
            level: Understanding level string.
            score: Ensemble score.

        Returns:
            PNG image as BytesIO.
        """
        color = _LEVEL_COLORS.get(level, "#666666")
        fig, ax = plt.subplots(figsize=(60 / 25.4, 15 / 25.4))

        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.9))
        ax.text(
            0.5,
            0.5,
            f"{level}  ({score:.2f})",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
            fontproperties=self._font_prop,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        fig.tight_layout(pad=0)
        return self._save_fig(fig)

    def radar_chart(
        self,
        student_axes: list[float],
        class_avg_axes: list[float],
        labels: list[str],
    ) -> io.BytesIO:
        """Create a radar chart comparing student vs class average.

        Args:
            student_axes: Student values for each axis (normalized 0–1).
            class_avg_axes: Class average values for each axis.
            labels: Korean labels for each axis.

        Returns:
            PNG image as BytesIO.
        """
        n = len(labels)
        if n < 3:
            fig, ax = plt.subplots(figsize=(120 / 25.4, 120 / 25.4))
            ax.text(0.5, 0.5, "축 부족", ha="center", va="center",
                    fontproperties=self._font_prop)
            return self._save_fig(fig)

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        # Close the polygon
        student_vals = list(student_axes) + [student_axes[0]]
        class_vals = list(class_avg_axes) + [class_avg_axes[0]]
        angles += [angles[0]]

        fig, ax = plt.subplots(
            figsize=(120 / 25.4, 120 / 25.4),
            subplot_kw={"polar": True},
        )

        # Student profile (blue solid)
        ax.plot(angles, student_vals, "o-", color="#1565C0",
                linewidth=2, label="학생")
        ax.fill(angles, student_vals, color="#1565C0", alpha=0.15)

        # Class average (gray dashed)
        ax.plot(angles, class_vals, "o--", color="#888888",
                linewidth=1.5, label="학급 평균")

        ax.set_thetagrids(
            np.degrees(angles[:-1]),
            labels,
            fontproperties=self._font_prop,
            fontsize=9,
        )
        ax.set_ylim(0, 1)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.3, 1.1),
            prop=self._font_prop,
            fontsize=8,
        )

        fig.tight_layout()
        return self._save_fig(fig)


def _translate_component(name: str) -> str:
    """Translate component key names to Korean labels."""
    translations = {
        "concept_coverage": "개념 커버리지",
        "llm_rubric": "LLM 루브릭",
        "rasch_ability": "Rasch 능력치",
    }
    return translations.get(name, name)
