"""Chart generators for student longitudinal PDF reports.

Produces matplotlib figures for concept coverage trends, score component
breakdowns, cohort-relative position box plots, and warning summary visuals.
Uses matplotlib Agg backend -- no display server required. No LLM API calls.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # MUST be before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

from forma.chart_utils import save_fig as _save_fig  # noqa: E402
from forma.font_utils import find_korean_font  # noqa: E402
from forma.student_longitudinal_data import (  # noqa: E402
    AlertLevel,
    CohortDistribution,
    StudentLongitudinalData,
    WarningSignal,
)

__all__ = [
    "build_coverage_trend_chart",
    "build_component_breakdown_chart",
    "build_cohort_position_chart",
    "build_warning_table",
]


def _get_font_prop(font_path: str | None = None) -> FontProperties:
    """Get FontProperties for Korean text rendering."""
    if font_path is None:
        font_path = find_korean_font()
    return FontProperties(fname=font_path)


# ---------------------------------------------------------------------------
# T018: Coverage trend chart
# ---------------------------------------------------------------------------


def build_coverage_trend_chart(
    student_data: StudentLongitudinalData,
    cohort: CohortDistribution,
    qsn: int,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build concept coverage trend chart for a single question.

    X axis = weeks, Y axis = concept_coverage [0,1].
    Student data as line plot with markers.
    Cohort data as box-and-whisker plot per week.

    Args:
        student_data: Per-student longitudinal data.
        cohort: Cohort distribution for box plots.
        qsn: Question serial number (1 or 2).
        font_path: Korean font path. Auto-detected if None.
        dpi: Chart DPI resolution.

    Returns:
        PNG image as io.BytesIO.
    """
    fp = _get_font_prop(font_path)
    fig, ax = plt.subplots(figsize=(160 / 25.4, 100 / 25.4))

    weeks = student_data.weeks
    if not weeks:
        q_label = "개념이해" if qsn == 1 else "적용"
        ax.text(0.5, 0.5, f"문제 {qsn} ({q_label}) — 데이터 없음",
                ha="center", va="center", fontproperties=fp, fontsize=12)
        ax.axis("off")
        return _save_fig(fig, dpi=dpi)

    # Collect student coverage values per week
    student_weeks = []
    student_vals = []
    for w in weeks:
        q_map = student_data.scores_by_week.get(w, {})
        if qsn in q_map and "concept_coverage" in q_map[qsn]:
            student_weeks.append(w)
            student_vals.append(q_map[qsn]["concept_coverage"])

    # Cohort box plot data
    box_positions = []
    box_data = []
    for w in weeks:
        q_scores = cohort.weekly_q_scores.get(w, {}).get(qsn, [])
        if q_scores:
            box_positions.append(w)
            box_data.append(q_scores)

    # Draw cohort box plots
    if box_data:
        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.4,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#E3F2FD")
            patch.set_edgecolor("#90CAF9")
        for median in bp["medians"]:
            median.set_color("#FF9800")
            median.set_linewidth(1.5)

    # Draw student line plot
    if student_weeks:
        ax.plot(
            student_weeks, student_vals,
            "o-", color="#1565C0", linewidth=2, markersize=6,
            label="학생", zorder=5,
        )

    q_label = "개념이해" if qsn == 1 else "적용"
    ax.set_xlabel("주차", fontproperties=fp, fontsize=9)
    ax.set_ylabel("개념 커버리지", fontproperties=fp, fontsize=9)
    ax.set_title(f"문제 {qsn} — 개념 커버리지 추세 ({q_label})",
                 fontproperties=fp, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(weeks)
    ax.set_xticklabels([str(w) for w in weeks], fontproperties=fp)
    if student_weeks or box_data:
        ax.legend(prop=fp, fontsize=8, loc="best")

    fig.tight_layout()
    return _save_fig(fig, dpi=dpi)


# ---------------------------------------------------------------------------
# T019: Component breakdown chart
# ---------------------------------------------------------------------------


def build_component_breakdown_chart(
    student_data: StudentLongitudinalData,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build score component grouped bar chart per week.

    Groups: concept_coverage, llm_rubric, ensemble_score on [0,1] scale.
    rasch_ability on secondary y-axis.

    Args:
        student_data: Per-student longitudinal data.
        font_path: Korean font path. Auto-detected if None.
        dpi: Chart DPI resolution.

    Returns:
        PNG image as io.BytesIO.
    """
    fp = _get_font_prop(font_path)
    fig, ax = plt.subplots(figsize=(160 / 25.4, 100 / 25.4))

    weeks = student_data.weeks
    if not weeks:
        ax.text(0.5, 0.5, "항목별 데이터 없음",
                ha="center", va="center", fontproperties=fp, fontsize=12)
        ax.axis("off")
        return _save_fig(fig, dpi=dpi)

    # Average across questions per week
    metrics = ["concept_coverage", "llm_rubric", "ensemble_score"]
    metric_labels = ["개념 커버리지", "LLM 루브릭", "앙상블 점수"]
    metric_colors = ["#2196F3", "#FF9800", "#4CAF50"]
    rasch_color = "#9C27B0"

    bar_width = 0.2
    x = np.arange(len(weeks))

    for idx, metric in enumerate(metrics):
        vals = []
        for w in weeks:
            q_map = student_data.scores_by_week.get(w, {})
            metric_vals = [s[metric] for s in q_map.values() if metric in s]
            vals.append(sum(metric_vals) / len(metric_vals) if metric_vals else 0.0)
        ax.bar(x + idx * bar_width, vals, bar_width,
               label=metric_labels[idx], color=metric_colors[idx], alpha=0.85)

    ax.set_xlabel("주차", fontproperties=fp, fontsize=9)
    ax.set_ylabel("점수 [0, 1]", fontproperties=fp, fontsize=9)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"W{w}" for w in weeks], fontproperties=fp)
    ax.set_ylim(0, 1.05)

    # rasch_ability on secondary axis
    ax2 = ax.twinx()
    rasch_vals = []
    for w in weeks:
        q_map = student_data.scores_by_week.get(w, {})
        rv = [s["rasch_ability"] for s in q_map.values() if "rasch_ability" in s]
        rasch_vals.append(sum(rv) / len(rv) if rv else 0.0)

    ax2.plot(x + bar_width, rasch_vals, "s--", color=rasch_color,
             linewidth=1.5, markersize=5, label="Rasch 능력치")
    ax2.set_ylabel("Rasch 능력치", fontproperties=fp, fontsize=9, color=rasch_color)
    ax2.tick_params(axis="y", labelcolor=rasch_color)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, prop=fp, fontsize=7, loc="best")

    ax.set_title("항목별 점수 분해", fontproperties=fp, fontsize=11)
    fig.tight_layout()
    return _save_fig(fig, dpi=dpi)


# ---------------------------------------------------------------------------
# T020: Cohort position chart
# ---------------------------------------------------------------------------


def build_cohort_position_chart(
    student_data: StudentLongitudinalData,
    cohort: CohortDistribution,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build cohort-relative position box plot chart.

    Box plot of all students' ensemble_score per week with
    the target student's score as a red star marker.

    Args:
        student_data: Per-student longitudinal data.
        cohort: Cohort distribution for box plot data.
        font_path: Korean font path. Auto-detected if None.
        dpi: Chart DPI resolution.

    Returns:
        PNG image as io.BytesIO.
    """
    fp = _get_font_prop(font_path)
    fig, ax = plt.subplots(figsize=(160 / 25.4, 100 / 25.4))

    weeks = student_data.weeks
    if not weeks:
        ax.text(0.5, 0.5, "상대 위치 데이터 없음",
                ha="center", va="center", fontproperties=fp, fontsize=12)
        ax.axis("off")
        return _save_fig(fig, dpi=dpi)

    # Cohort box plots
    box_positions = []
    box_data = []
    for w in weeks:
        scores = cohort.weekly_scores.get(w, [])
        if scores:
            box_positions.append(w)
            box_data.append(scores)

    if box_data:
        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.4,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#E8F5E9")
            patch.set_edgecolor("#66BB6A")
        for median in bp["medians"]:
            median.set_color("#FF9800")
            median.set_linewidth(1.5)

    # Student score overlay (red star)
    student_weeks = []
    student_scores = []
    for w in weeks:
        q_map = student_data.scores_by_week.get(w, {})
        vals = [s.get("ensemble_score", 0.0) for s in q_map.values() if "ensemble_score" in s]
        if vals:
            student_weeks.append(w)
            student_scores.append(sum(vals) / len(vals))

    if student_weeks:
        ax.plot(
            student_weeks, student_scores,
            "*", color="red", markersize=14, zorder=10,
            label="학생",
        )

        # Percentile labels
        for w, score in zip(student_weeks, student_scores):
            pct = student_data.percentiles_by_week.get(w)
            if pct is not None:
                ax.annotate(
                    f"{pct:.0f}th",
                    (w, score),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontproperties=fp,
                    fontsize=7,
                    color="red",
                )

    ax.set_xlabel("주차", fontproperties=fp, fontsize=9)
    ax.set_ylabel("앙상블 점수", fontproperties=fp, fontsize=9)
    ax.set_title("전체 수강생 내 상대 위치", fontproperties=fp, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(weeks)
    ax.set_xticklabels([str(w) for w in weeks], fontproperties=fp)
    if student_weeks or box_data:
        ax.legend(prop=fp, fontsize=8, loc="best")

    fig.tight_layout()
    return _save_fig(fig, dpi=dpi)


# ---------------------------------------------------------------------------
# T021: Warning table chart
# ---------------------------------------------------------------------------


_ALERT_COLORS = {
    AlertLevel.NORMAL: "#2E7D32",
    AlertLevel.CAUTION: "#F57F17",
    AlertLevel.WARNING: "#C62828",
}

_ALERT_LABELS = {
    AlertLevel.NORMAL: "정상",
    AlertLevel.CAUTION: "주의",
    AlertLevel.WARNING: "경고",
}


def build_warning_table(
    warnings: list[WarningSignal],
    alert_level: AlertLevel,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build warning status visual as a PNG image.

    Shows alert level badge at top, with list of triggered/inactive signals.

    Args:
        warnings: List of WarningSignal instances.
        alert_level: Overall AlertLevel.
        font_path: Korean font path. Auto-detected if None.
        dpi: Chart DPI resolution.

    Returns:
        PNG image as io.BytesIO.
    """
    fp = _get_font_prop(font_path)
    n_signals = max(len(warnings), 1)
    fig_height = max(60 / 25.4, (n_signals * 12 + 30) / 25.4)
    fig, ax = plt.subplots(figsize=(160 / 25.4, fig_height))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_signals + 3)

    color = _ALERT_COLORS.get(alert_level, "#666666")
    label = _ALERT_LABELS.get(alert_level, "알 수 없음")

    # Alert level badge
    badge_y = n_signals + 2
    circle = plt.Circle((1, badge_y), 0.4, color=color, zorder=5)
    ax.add_patch(circle)
    ax.text(2, badge_y, f"경고 수준: {label}",
            fontproperties=fp, fontsize=12, fontweight="bold",
            va="center", color=color)

    # Signal list
    for i, signal in enumerate(warnings):
        y = n_signals - i
        if signal.triggered:
            marker = "\u2717"  # X mark
            sig_color = "#C62828"
        else:
            marker = "\u2713"  # checkmark
            sig_color = "#2E7D32"

        ax.text(1, y, marker, fontsize=14, va="center", ha="center",
                color=sig_color, fontweight="bold")
        ax.text(2, y, signal.name,
                fontproperties=fp, fontsize=9, va="center", color="#333333")
        ax.text(6, y, signal.detail,
                fontproperties=fp, fontsize=8, va="center", color="#666666")

    fig.tight_layout()
    return _save_fig(fig, dpi=dpi)
