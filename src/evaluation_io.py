"""YAML I/O utilities for the multi-layer evaluation framework.

Wraps topic_analysis.load_yaml_data() with path validation and
provides save/extract helpers for evaluation data.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

from src.topic_analysis import load_yaml_data


def load_evaluation_yaml(yaml_path: str) -> dict[str, Any]:
    """Load an evaluation YAML file with path validation.

    Wraps ``topic_analysis.load_yaml_data`` and raises an informative
    error when the file does not exist (fail-fast).

    Args:
        yaml_path: Absolute or relative path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist on disk.

    Examples:
        >>> data = load_evaluation_yaml("exams/Ch01_서론_FormativeTest.yaml")
        >>> data["metadata"]["chapter"]
        1
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Evaluation YAML not found: '{yaml_path}'. "
            "Check that the path is correct and the file exists."
        )
    return load_yaml_data(yaml_path)


def save_evaluation_yaml(
    data: dict[str, Any], output_path: str
) -> None:
    """Save evaluation results to a YAML file.

    Creates parent directories automatically.  Uses ``allow_unicode=True``
    so Korean text is written as-is (not escaped).

    Args:
        data: Result data to serialise.
        output_path: Destination file path (created if absent).

    Returns:
        None

    Examples:
        >>> save_evaluation_yaml({"score": 0.85}, "/tmp/result.yaml")
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, allow_unicode=True, default_flow_style=False)


def extract_student_responses(
    evaluation_data: dict[str, Any],
) -> dict[str, dict[int, str]]:
    """Extract student responses keyed by student_id and question_sn.

    Expects the top-level dict to contain a ``"responses"`` key whose
    value maps student IDs → {question_sn (int) → response text}.

    Args:
        evaluation_data: Parsed YAML dict with a ``"responses"`` key.

    Returns:
        Mapping of student_id → {question_sn → response_text}.

    Raises:
        KeyError: If ``"responses"`` key is absent from evaluation_data.

    Examples:
        >>> data = {"responses": {"s001": {1: "세포막은 인지질 이중층."}}}
        >>> result = extract_student_responses(data)
        >>> result["s001"][1]
        '세포막은 인지질 이중층.'
    """
    if "responses" not in evaluation_data:
        raise KeyError(
            "Key 'responses' not found in evaluation data. "
            "The YAML file must contain a top-level 'responses' mapping."
        )
    raw: dict[str, Any] = evaluation_data["responses"]
    return {
        student_id: {int(qsn): text for qsn, text in q_map.items()}
        for student_id, q_map in raw.items()
    }
