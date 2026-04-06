"""Exam config YAML validation utilities.

Validates knowledge_graph edges, rubric_tiers, question_type values,
and provides warnings for potentially problematic configurations.
"""

from __future__ import annotations

import math
import warnings
from typing import Any


def validate_question_config(question: dict[str, Any]) -> list[str]:
    """Validate a single question's configuration.

    Args:
        question: Question config dict from exam YAML.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    sn = question.get("sn", "?")

    # question_type validation
    qtype = question.get("question_type", "essay")
    if qtype not in ("essay", "short_answer"):
        errors.append(f"Q{sn}: question_type must be 'essay' or 'short_answer', got {qtype!r}")

    # knowledge_graph validation
    kg = question.get("knowledge_graph")
    if kg is not None:
        edges = kg.get("edges", [])
        if not isinstance(edges, list):
            errors.append(f"Q{sn}: knowledge_graph.edges must be a list")
        else:
            for i, edge in enumerate(edges):
                if not isinstance(edge, dict):
                    errors.append(f"Q{sn}: edge[{i}] must be a dict with subject/relation/object")
                    continue
                for key in ("subject", "relation", "object"):
                    if key not in edge:
                        errors.append(f"Q{sn}: edge[{i}] missing required key '{key}'")

    # rubric_tiers validation
    tiers = question.get("rubric_tiers")
    if tiers is not None:
        for level_key, tier in tiers.items():
            if not isinstance(tier, dict):
                continue
            f1 = tier.get("min_graph_f1", 0.0)
            if f1 > 0.95:
                warnings.warn(
                    f"Q{sn}: {level_key}.min_graph_f1={f1} > 0.95 is very strict; students may never reach this tier.",
                    stacklevel=2,
                )

    return errors


def validate_edge_answer_ratio(question: dict[str, Any], answer_limit_chars: int = 200) -> list[str]:
    """Warn if edge count is too high relative to answer length limit.

    Args:
        question: Question config dict.
        answer_limit_chars: Expected maximum answer length in characters.

    Returns:
        List of warning messages.
    """
    warnings_list: list[str] = []
    sn = question.get("sn", "?")
    kg = question.get("knowledge_graph")
    if kg is None:
        return warnings_list

    edges = kg.get("edges", [])
    threshold = math.ceil(answer_limit_chars / 40)
    if len(edges) > threshold:
        warnings_list.append(
            f"Q{sn}: {len(edges)} edges > ceil({answer_limit_chars}/40)="
            f"{threshold}. Students may not cover all edges in "
            f"{answer_limit_chars} chars."
        )
    return warnings_list


def validate_exam_config(config_data: dict[str, Any]) -> list[str]:
    """Validate the full exam configuration.

    Args:
        config_data: Parsed exam YAML dict.

    Returns:
        List of all error messages across questions.
    """
    errors: list[str] = []
    questions = config_data.get("questions", [])
    for q in questions:
        errors.extend(validate_question_config(q))
        errors.extend(validate_edge_answer_ratio(q))
    return errors
