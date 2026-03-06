"""Convert OCR join output to evaluation pipeline input format.

Join output is a flat list of dicts; evaluation input uses a nested
``{responses: {student_id: {question_sn: text}}}`` structure.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def convert_join_to_responses(
    join_data: list[dict],
    questions_used: list[int] | None = None,
) -> dict:
    """Convert flat join list to nested responses dict.

    Args:
        join_data: List of dicts with keys ``student_id``, ``q_num``,
            ``text``.
        questions_used: Exam ``sn`` numbers in q_num order.
            e.g. ``[1, 3]`` means q1 maps to sn1, q2 maps to sn3.
            When *None*, q_num is used as-is.

    Returns:
        Dict in evaluation input format::

            {"responses": {"S015": {1: "생체항상성은..."}}}
    """
    q_to_sn: dict[int, int] | None = None
    if questions_used:
        q_to_sn = {i + 1: sn for i, sn in enumerate(questions_used)}

    responses: dict[str, dict[int, str]] = {}
    for entry in join_data:
        sid = str(entry["student_id"])
        q_num = int(entry["q_num"])
        text = str(entry.get("text", ""))

        if q_to_sn is not None:
            sn = q_to_sn.get(q_num)
            if sn is None:
                continue
        else:
            sn = q_num

        responses.setdefault(sid, {})[sn] = text
    return {"responses": responses}


def convert_join_file(
    join_path: str,
    output_path: str,
    questions_used: list[int] | None = None,
) -> None:
    """Convert a join YAML file to evaluation input YAML file.

    Args:
        join_path: Path to join output YAML file (flat list format).
        output_path: Path for the converted evaluation input file.
        questions_used: Exam ``sn`` numbers in q_num order (see
            :func:`convert_join_to_responses`).
    """
    with open(join_path, "r", encoding="utf-8") as f:
        join_data = yaml.safe_load(f)

    if not isinstance(join_data, list):
        raise ValueError(
            f"Expected a list of dicts in join file, got {type(join_data).__name__}"
        )

    result = convert_join_to_responses(join_data, questions_used)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(result, f, allow_unicode=True, default_flow_style=False)


def filter_exam_config(
    config_data: dict,
    questions_used: list[int],
) -> dict:
    """Filter exam config to only include selected questions.

    Args:
        config_data: Full exam config dict.
        questions_used: List of ``sn`` numbers to keep.

    Returns:
        Shallow copy of *config_data* with ``questions`` filtered.

    Raises:
        KeyError: If any sn in *questions_used* is not found.
    """
    available = {q["sn"] for q in config_data.get("questions", [])}
    missing = set(questions_used) - available
    if missing:
        raise KeyError(
            f"questions_used contains sn {sorted(missing)} "
            f"not found in exam config. Available: {sorted(available)}"
        )

    filtered = dict(config_data)
    filtered["questions"] = [
        q for q in config_data["questions"]
        if q["sn"] in set(questions_used)
    ]
    return filtered
