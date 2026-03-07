"""YAML I/O utilities for the multi-layer evaluation framework.

Wraps topic_analysis.load_yaml_data() with path validation and
provides save/extract helpers for evaluation data.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import yaml

from forma.topic_analysis import load_yaml_data


# ---------------------------------------------------------------------------
# Custom YAML Dumper: quotes string values, rounds floats to 2dp
# ---------------------------------------------------------------------------


class FormaDumper(yaml.SafeDumper):
    """Custom YAML dumper: unquoted keys, quoted string values, 2dp floats."""

    # Instance flag toggled by represent_mapping to distinguish keys vs values.
    _is_key: bool = False

    def represent_mapping(
        self, tag: str, mapping: Any, flow_style: Any = None,
    ) -> yaml.MappingNode:
        """Override to set _is_key flag so str representer can skip quoting keys."""
        value: list = []
        if self.default_flow_style is not None:
            node = yaml.MappingNode(tag, value, flow_style=self.default_flow_style)
        else:
            node = yaml.MappingNode(tag, value, flow_style=flow_style)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node
        best_style = True
        if hasattr(mapping, "items"):
            mapping = list(mapping.items())
        for item_key, item_value in mapping:
            self._is_key = True
            node_key = self.represent_data(item_key)
            self._is_key = False
            node_value = self.represent_data(item_value)
            if not (isinstance(node_key, yaml.ScalarNode) and not node_key.style):
                best_style = False
            if not (isinstance(node_value, yaml.ScalarNode) and not node_value.style):
                best_style = False
            value.append((node_key, node_value))
        if self.default_flow_style is None:
            if len(value) != 0 and best_style:
                node.flow_style = True
        return node


def _represent_str(dumper: FormaDumper, data: str) -> yaml.ScalarNode:
    """Keys: plain (no quotes). Values: double-quoted (multiline → literal block)."""
    if dumper._is_key:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


def _represent_rounded_float(
    dumper: yaml.SafeDumper, data: float,
) -> yaml.ScalarNode:
    """Represent floats rounded to 2 decimal places."""
    rounded = round(data, 2)
    # Avoid -0.0
    if rounded == 0.0:
        rounded = 0.0
    return dumper.represent_float(rounded)


FormaDumper.add_representer(str, _represent_str)
FormaDumper.add_representer(float, _represent_rounded_float)


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
    """Save evaluation results to a YAML file using atomic write.

    Writes to a temporary file first, then atomically replaces the
    target via ``os.replace()``.  Creates a ``.bak`` backup of any
    existing file before overwriting.

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

    # Backup existing file
    if os.path.isfile(output_path):
        bak_path = output_path + ".bak"
        try:
            os.replace(output_path, bak_path)
        except OSError:
            pass

    # Atomic write: tempfile → os.replace
    fd, tmp_path = tempfile.mkstemp(
        dir=parent or ".", suffix=".tmp", prefix=".eval_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.dump(
                data, fh,
                Dumper=FormaDumper,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
        os.replace(tmp_path, output_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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
