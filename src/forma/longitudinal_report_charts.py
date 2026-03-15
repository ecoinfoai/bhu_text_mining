"""Chart generation for longitudinal summary PDF reports.

Produces 3 chart types as PNG io.BytesIO buffers for embedding in ReportLab PDFs.
Uses matplotlib Agg backend — no display server required. No LLM API calls.
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

from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from forma.longitudinal_report_data import LongitudinalSummaryData


def _get_font_prop(font_path: str | None = None) -> FontProperties:
    """Get FontProperties for Korean text rendering."""
    if font_path is None:
        font_path = find_korean_font()
    return FontProperties(fname=font_path)


def build_trajectory_line_chart(
    summary_data: LongitudinalSummaryData,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build student trajectory line chart.

    Risk students: red solid lines.
    Normal students: gray lines with alpha=0.3.
    Class average: blue bold line.

    Args:
        summary_data: LongitudinalSummaryData with trajectories and averages.
        font_path: Path to Korean font. Auto-detected if None.
        dpi: Chart DPI resolution.

    Returns:
        PNG image as io.BytesIO.
    """
    fp = _get_font_prop(font_path)
    fig, ax = plt.subplots(figsize=(160 / 25.4, 100 / 25.4))

    if not summary_data.student_trajectories:
        ax.text(0.5, 0.5, "학생 데이터 없음",
                ha="center", va="center", fontproperties=fp, fontsize=12)
        ax.axis("off")
        return _save_fig(fig, dpi=dpi)

    weeks = sorted(summary_data.period_weeks)

    # Plot normal students first (background), then risk students on top
    for traj in summary_data.student_trajectories:
        if traj.is_persistent_risk:
            continue
        ws = sorted(traj.weekly_scores.keys())
        scores = [traj.weekly_scores[w] for w in ws]
        ax.plot(ws, scores, color="gray", alpha=0.3, linewidth=1)

    for traj in summary_data.student_trajectories:
        if not traj.is_persistent_risk:
            continue
        ws = sorted(traj.weekly_scores.keys())
        scores = [traj.weekly_scores[w] for w in ws]
        ax.plot(ws, scores, color="red", alpha=0.8, linewidth=1.5,
                label=traj.student_id)

    # Class average line (blue bold)
    if summary_data.class_weekly_averages:
        avg_weeks = sorted(summary_data.class_weekly_averages.keys())
        avg_scores = [summary_data.class_weekly_averages[w] for w in avg_weeks]
        ax.plot(avg_weeks, avg_scores, color="blue", linewidth=2.5,
                label="학급 평균", zorder=10)

    ax.set_xlabel("주차", fontproperties=fp, fontsize=9)
    ax.set_ylabel("앙상블 점수", fontproperties=fp, fontsize=9)
    ax.set_title("학생별 점수 궤적", fontproperties=fp, fontsize=11)
    ax.set_ylim(0, 1.05)

    if weeks:
        ax.set_xticks(weeks)
        ax.set_xticklabels([str(w) for w in weeks], fontproperties=fp)

    # Legend — show risk students + class average
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Deduplicate
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  prop=fp, fontsize=7, loc="best")

    fig.tight_layout()
    return _save_fig(fig, dpi=dpi)


def build_class_week_heatmap(
    summary_data: LongitudinalSummaryData,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build student x week heatmap chart.

    Students sorted by final week score descending.
    Color scale: red (low) to green (high).
    100+ students: top 25 + bottom 25, "... N명 생략" gap.

    Args:
        summary_data: LongitudinalSummaryData with trajectories.
        font_path: Path to Korean font. Auto-detected if None.
        dpi: Chart DPI resolution.

    Returns:
        PNG image as io.BytesIO.
    """
    fp = _get_font_prop(font_path)
    fig, ax = plt.subplots(figsize=(160 / 25.4, 120 / 25.4))

    if not summary_data.student_trajectories:
        ax.text(0.5, 0.5, "학생 데이터 없음",
                ha="center", va="center", fontproperties=fp, fontsize=12)
        ax.axis("off")
        return _save_fig(fig, dpi=dpi)

    weeks = sorted(summary_data.period_weeks)

    # Sort students by final week score descending
    def _final_score(traj):
        for w in reversed(weeks):
            if w in traj.weekly_scores:
                return traj.weekly_scores[w]
        return 0.0

    sorted_trajs = sorted(
        summary_data.student_trajectories,
        key=_final_score,
        reverse=True,
    )

    # Cap at 50 display rows for 100+ students
    gap_label = None
    if len(sorted_trajs) > 100:
        top = sorted_trajs[:25]
        bottom = sorted_trajs[-25:]
        n_omitted = len(sorted_trajs) - 50
        gap_label = f"... {n_omitted}명 생략"
        display_trajs = top + bottom
    else:
        display_trajs = sorted_trajs

    # Build matrix (students x weeks) with NaN for missing
    n_students = len(display_trajs)
    n_weeks = len(weeks)
    matrix = np.full((n_students, n_weeks), np.nan)
    labels = []

    for i, traj in enumerate(display_trajs):
        labels.append(traj.student_id)
        for j, w in enumerate(weeks):
            if w in traj.weekly_scores:
                matrix[i, j] = traj.weekly_scores[w]

    # Handle edge case: figure height for number of students
    fig_height = max(3, n_students * 0.25 + 1.5)
    fig.set_size_inches(160 / 25.4, fig_height)

    # Create masked array for NaN handling
    masked_matrix = np.ma.masked_invalid(matrix)

    # Set color map with gray for missing values
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="lightgray")

    im = ax.imshow(masked_matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.02, label="점수")

    # Insert gap indicator if needed
    if gap_label is not None:
        ax.axhline(y=24.5, color="white", linewidth=3)
        ax.text(n_weeks / 2 - 0.5, 24.5, gap_label,
                ha="center", va="center", fontproperties=fp, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.9))

    ax.set_xticks(range(n_weeks))
    ax.set_xticklabels([f"W{w}" for w in weeks], fontproperties=fp, fontsize=8)
    ax.set_yticks(range(n_students))
    ax.set_yticklabels(labels, fontproperties=fp, fontsize=7)
    ax.set_title("학생×주차 히트맵", fontproperties=fp, fontsize=11)

    fig.tight_layout()
    return _save_fig(fig, dpi=dpi)


def build_concept_mastery_bar_chart(
    summary_data: LongitudinalSummaryData,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build horizontal bar chart of concept mastery delta.

    Sorted by delta descending.
    Positive delta: green bars. Negative delta: red bars.

    Args:
        summary_data: LongitudinalSummaryData with concept_mastery_changes.
        font_path: Path to Korean font. Auto-detected if None.
        dpi: Chart DPI resolution.

    Returns:
        PNG image as io.BytesIO.
    """
    fp = _get_font_prop(font_path)
    changes = summary_data.concept_mastery_changes

    if not changes:
        fig, ax = plt.subplots(figsize=(160 / 25.4, 60 / 25.4))
        ax.text(0.5, 0.5, "개념 데이터 없음",
                ha="center", va="center", fontproperties=fp, fontsize=12)
        ax.axis("off")
        return _save_fig(fig, dpi=dpi)

    n = len(changes)
    fig_height = max(60 / 25.4, n * 15 / 25.4)
    fig, ax = plt.subplots(figsize=(160 / 25.4, fig_height))

    concepts = [c.concept for c in changes]
    deltas = [c.delta for c in changes]
    colors = ["#2E7D32" if d >= 0 else "#C62828" for d in deltas]

    positions = list(range(n))
    bars = ax.barh(positions, deltas, color=colors, height=0.6)

    # Value labels
    for bar, delta in zip(bars, deltas):
        x_pos = delta + 0.01 if delta >= 0 else delta - 0.01
        ha = "left" if delta >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{delta:+.2f}", va="center", ha=ha,
                fontproperties=fp, fontsize=8)

    ax.set_yticks(positions)
    ax.set_yticklabels(concepts, fontproperties=fp, fontsize=9)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("정답률 변화 (Δ)", fontproperties=fp, fontsize=9)
    ax.set_title("개념별 마스터리 변화", fontproperties=fp, fontsize=11)
    ax.invert_yaxis()

    fig.tight_layout()
    return _save_fig(fig, dpi=dpi)


def build_intervention_effect_chart(
    effects: list,
    *,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build pre/post bar chart for intervention effects.

    Shows paired bars (pre=blue, post=green) for each student with
    sufficient data. Insufficient data effects are excluded.

    Args:
        effects: List of InterventionEffect objects.
        font_path: Korean font path.
        dpi: Image resolution.

    Returns:
        PNG image as BytesIO buffer.
    """
    font_prop = _get_font_prop(font_path)

    # Filter to sufficient data only
    sufficient = [e for e in effects if e.sufficient_data]

    if not sufficient:
        fig, ax = plt.subplots(figsize=(160 / 25.4, 60 / 25.4))
        ax.text(0.5, 0.5, "개입 효과 데이터 없음",
                ha="center", va="center", fontproperties=font_prop, fontsize=12)
        ax.axis("off")
        return _save_fig(fig, dpi=dpi)

    n = len(sufficient)
    fig_height = max(60 / 25.4, n * 12 / 25.4)
    fig, ax = plt.subplots(figsize=(160 / 25.4, fig_height))

    labels = [f"{e.student_id} ({e.intervention_type})" for e in sufficient]
    pre_scores = [e.pre_mean for e in sufficient]
    post_scores = [e.post_mean for e in sufficient]

    y_pos = np.arange(n)
    bar_height = 0.35

    ax.barh(y_pos - bar_height / 2, pre_scores, bar_height,
            label="개입 전", color="#42A5F5", alpha=0.8)
    ax.barh(y_pos + bar_height / 2, post_scores, bar_height,
            label="개입 후", color="#66BB6A", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontproperties=font_prop, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("앙상블 점수", fontproperties=font_prop, fontsize=9)
    ax.set_title("개입 전후 점수 변화", fontproperties=font_prop, fontsize=11)
    ax.legend(prop=font_prop, fontsize=8, loc="best")
    ax.invert_yaxis()

    fig.tight_layout()
    return _save_fig(fig, dpi=dpi)


def build_risk_trend_chart(
    risk_predictions: list,
    *,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build risk probability horizontal bar chart for top at-risk students.

    Args:
        risk_predictions: List of RiskPrediction objects.
        font_path: Korean font path.
        dpi: Image resolution.

    Returns:
        PNG image as BytesIO buffer.
    """
    font_prop = _get_font_prop(font_path)
    fig, ax = plt.subplots(figsize=(8, 4))

    # Sort by probability descending, show top 10
    sorted_preds = sorted(
        risk_predictions, key=lambda p: p.drop_probability, reverse=True,
    )[:10]

    if not sorted_preds:
        ax.text(0.5, 0.5, "예측 데이터 없음", ha="center", va="center",
                fontproperties=font_prop, fontsize=14)
    else:
        student_ids = [p.student_id for p in sorted_preds]
        probs = [p.drop_probability for p in sorted_preds]
        bar_colors = ["#C62828" if p >= 0.5 else "#F57F17" for p in probs]

        y_pos = np.arange(len(student_ids))
        ax.barh(y_pos, probs, color=bar_colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(student_ids, fontsize=8)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.invert_yaxis()

    ax.set_xlabel("드롭 확률", fontproperties=font_prop)
    ax.set_title("드롭 리스크 예측 (상위 10명)", fontproperties=font_prop, fontsize=12)

    fig.tight_layout()
    return _save_fig(fig, dpi=dpi)


def build_ocr_confidence_trend_chart(
    trajectories: dict[str, list[tuple[int, float]]],
    *,
    threshold: float = 0.75,
    font_path: str | None = None,
    dpi: int = 150,
) -> io.BytesIO:
    """Build OCR confidence trend line chart across weeks.

    Students with 3+ consecutive weeks below *threshold* are drawn in
    red; others in gray.

    Args:
        trajectories: ``{student_id: [(week, confidence_mean), ...]}``
        threshold: Low-confidence threshold (default 0.75).
        font_path: Korean font path.
        dpi: Image resolution.

    Returns:
        PNG image as BytesIO buffer.
    """
    font_prop = _get_font_prop(font_path)
    fig, ax = plt.subplots(figsize=(8, 4))

    if not trajectories:
        ax.text(
            0.5, 0.5, "데이터 없음",
            ha="center", va="center",
            fontproperties=font_prop, fontsize=14,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return _save_fig(fig, dpi=dpi)

    for sid, traj in trajectories.items():
        if not traj:
            continue
        weeks = [t[0] for t in traj]
        vals = [t[1] for t in traj]

        # Check for 3+ consecutive weeks below threshold
        consecutive = 0
        max_consecutive = 0
        for v in vals:
            if v < threshold:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        if max_consecutive >= 3:
            ax.plot(weeks, vals, color="red", alpha=0.8, linewidth=1.5)
        else:
            ax.plot(weeks, vals, color="gray", alpha=0.3, linewidth=0.8)

    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("주차", fontproperties=font_prop)
    ax.set_ylabel("텍스트 인식 신뢰도", fontproperties=font_prop)
    ax.set_title("텍스트 인식 신뢰도 추이", fontproperties=font_prop, fontsize=12)

    fig.tight_layout()
    return _save_fig(fig, dpi=dpi)
