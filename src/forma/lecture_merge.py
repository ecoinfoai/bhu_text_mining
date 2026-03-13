"""Multi-session transcript merger for class-level analysis.

Merges multiple per-session AnalysisResult objects into a single
MergedAnalysis for cross-class comparison across all sessions combined.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from forma.lecture_analyzer import AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class MergedAnalysis:
    """Result of merging multiple session analyses for one class.

    Args:
        class_id: Class section identifier.
        weeks: Week numbers included in merge.
        combined_keyword_frequencies: Merged keyword frequencies (sum).
        per_session_keyword_frequencies: Week -> keyword frequencies.
        session_boundary_markers: Boundary marker strings.
        merged_text: Combined cleaned text with boundary markers.
    """

    class_id: str
    weeks: list[int]
    combined_keyword_frequencies: dict[str, int]
    per_session_keyword_frequencies: dict[int, dict[str, int]]
    session_boundary_markers: list[str]
    merged_text: str


def merge_analyses(
    analyses: list[AnalysisResult],
    class_id: str,
) -> MergedAnalysis:
    """Merge multiple session analyses into one class-level result.

    Sessions are merged in week order. Per-session keyword data
    is preserved alongside combined data (FR-022).

    Args:
        analyses: List of AnalysisResult objects for one class.
        class_id: Class section identifier.

    Returns:
        MergedAnalysis with combined and per-session data.

    Raises:
        ValueError: If analyses is empty.
    """
    if not analyses:
        raise ValueError(
            f"병합할 분석 결과가 없습니다: class_id={class_id}"
        )

    # Sort by week
    sorted_analyses = sorted(analyses, key=lambda a: a.week)

    weeks = [a.week for a in sorted_analyses]
    combined_freq: Counter[str] = Counter()
    per_session_freq: dict[int, dict[str, int]] = {}
    boundary_markers: list[str] = []
    text_parts: list[str] = []

    for idx, a in enumerate(sorted_analyses, 1):
        marker = f"--- Session {idx} (Week {a.week}) ---"
        boundary_markers.append(marker)
        text_parts.append(marker)
        combined_freq.update(a.keyword_frequencies)
        per_session_freq[a.week] = dict(a.keyword_frequencies)

    logger.info(
        "병합 완료: class_id=%s, weeks=%s, 키워드 수=%d",
        class_id, weeks, len(combined_freq),
    )

    return MergedAnalysis(
        class_id=class_id,
        weeks=weeks,
        combined_keyword_frequencies=dict(combined_freq),
        per_session_keyword_frequencies=per_session_freq,
        session_boundary_markers=boundary_markers,
        merged_text="\n".join(text_parts),
    )


def save_merged_analysis(
    result: MergedAnalysis,
    output_dir: Path,
) -> Path:
    """Save MergedAnalysis to YAML file.

    Args:
        result: MergedAnalysis to save.
        output_dir: Directory to write the file.

    Returns:
        Path to the saved YAML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.class_id}_merged_analysis.yaml"
    data = {
        "class_id": result.class_id,
        "weeks": result.weeks,
        "combined_keyword_frequencies": result.combined_keyword_frequencies,
        "per_session_keyword_frequencies": {
            str(k): v for k, v in result.per_session_keyword_frequencies.items()
        },
        "session_boundary_markers": result.session_boundary_markers,
        "merged_text": result.merged_text,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    logger.info("병합 분석 결과 저장: %s", path)
    return path


def load_merged_analysis(path: Path) -> MergedAnalysis:
    """Load MergedAnalysis from YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        MergedAnalysis instance.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"병합 분석 결과 파일을 찾을 수 없습니다: {path}"
        )
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return MergedAnalysis(
        class_id=data["class_id"],
        weeks=data["weeks"],
        combined_keyword_frequencies=data["combined_keyword_frequencies"],
        per_session_keyword_frequencies={
            int(k): v for k, v in data["per_session_keyword_frequencies"].items()
        },
        session_boundary_markers=data["session_boundary_markers"],
        merged_text=data.get("merged_text", ""),
    )
