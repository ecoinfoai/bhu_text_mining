"""Convert OCR join output to evaluation pipeline input format.

Join output is a flat list of dicts; evaluation input uses a nested
``{responses: {student_id: {question_sn: text}}}`` structure.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def convert_join_to_responses(join_data: list[dict]) -> dict:
    """Convert flat join list to nested responses dict.

    Args:
        join_data: List of dicts with keys ``student_id``, ``q_num``,
            ``text``.

    Returns:
        Dict in evaluation input format::

            {"responses": {"S015": {1: "생체항상성은..."}}}
    """
    responses: dict[str, dict[int, str]] = {}
    for entry in join_data:
        sid = str(entry["student_id"])
        q_num = int(entry["q_num"])
        text = str(entry.get("text", ""))
        responses.setdefault(sid, {})[q_num] = text
    return {"responses": responses}


def convert_join_file(join_path: str, output_path: str) -> None:
    """Convert a join YAML file to evaluation input YAML file.

    Args:
        join_path: Path to join output YAML file (flat list format).
        output_path: Path for the converted evaluation input file.
    """
    with open(join_path, "r", encoding="utf-8") as f:
        join_data = yaml.safe_load(f)

    if not isinstance(join_data, list):
        raise ValueError(
            f"Expected a list of dicts in join file, got {type(join_data).__name__}"
        )

    result = convert_join_to_responses(join_data)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(result, f, allow_unicode=True, default_flow_style=False)
