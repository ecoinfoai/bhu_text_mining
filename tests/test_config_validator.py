"""Tests for config_validator.py — exam config validation."""

from __future__ import annotations

import warnings

import pytest

from src.config_validator import (
    validate_exam_config,
    validate_edge_answer_ratio,
    validate_question_config,
)


class TestValidateQuestionConfig:
    """Tests for validate_question_config()."""

    def test_valid_essay_question(self):
        """Valid essay question with knowledge_graph passes."""
        q = {
            "sn": 1,
            "question_type": "essay",
            "knowledge_graph": {
                "edges": [
                    {"subject": "A", "relation": "causes", "object": "B"},
                ]
            },
        }
        assert validate_question_config(q) == []

    def test_valid_short_answer(self):
        """Valid short_answer question passes."""
        q = {"sn": 2, "question_type": "short_answer"}
        assert validate_question_config(q) == []

    def test_invalid_question_type(self):
        """Invalid question_type produces error."""
        q = {"sn": 1, "question_type": "multiple_choice"}
        errors = validate_question_config(q)
        assert any("question_type" in e for e in errors)

    def test_default_question_type_is_essay(self):
        """Missing question_type defaults to essay (no error)."""
        q = {"sn": 1}
        assert validate_question_config(q) == []

    def test_edge_missing_keys(self):
        """Edge missing required keys produces error."""
        q = {
            "sn": 1,
            "knowledge_graph": {
                "edges": [{"subject": "A", "relation": "r"}]
            },
        }
        errors = validate_question_config(q)
        assert any("object" in e for e in errors)

    def test_edge_not_dict(self):
        """Non-dict edge produces error."""
        q = {
            "sn": 1,
            "knowledge_graph": {"edges": ["not a dict"]},
        }
        errors = validate_question_config(q)
        assert len(errors) == 1

    def test_edges_not_list(self):
        """Non-list edges produces error."""
        q = {
            "sn": 1,
            "knowledge_graph": {"edges": "not a list"},
        }
        errors = validate_question_config(q)
        assert any("must be a list" in e for e in errors)

    def test_rubric_tier_high_f1_warns(self):
        """min_graph_f1 > 0.95 emits warning."""
        q = {
            "sn": 1,
            "rubric_tiers": {
                "level_3": {"min_graph_f1": 0.99, "requires_terminology": True},
            },
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_question_config(q)
            assert len(w) == 1
            assert "0.95" in str(w[0].message)


class TestValidateEdgeAnswerRatio:
    """Tests for validate_edge_answer_ratio()."""

    def test_no_knowledge_graph(self):
        """No knowledge_graph returns empty warnings."""
        assert validate_edge_answer_ratio({"sn": 1}) == []

    def test_acceptable_ratio(self):
        """Few edges relative to answer length passes."""
        q = {
            "sn": 1,
            "knowledge_graph": {
                "edges": [
                    {"subject": "A", "relation": "r", "object": "B"},
                ]
            },
        }
        assert validate_edge_answer_ratio(q, answer_limit_chars=200) == []

    def test_too_many_edges_warns(self):
        """Many edges relative to answer length warns."""
        edges = [
            {"subject": f"S{i}", "relation": "r", "object": f"O{i}"}
            for i in range(20)
        ]
        q = {"sn": 1, "knowledge_graph": {"edges": edges}}
        warns = validate_edge_answer_ratio(q, answer_limit_chars=200)
        assert len(warns) == 1


class TestValidateExamConfig:
    """Tests for validate_exam_config()."""

    def test_valid_config(self):
        """Valid full config produces no errors."""
        config = {
            "questions": [
                {
                    "sn": 1,
                    "question_type": "essay",
                    "knowledge_graph": {
                        "edges": [
                            {"subject": "A", "relation": "r", "object": "B"},
                        ]
                    },
                },
                {"sn": 2, "question_type": "short_answer"},
            ]
        }
        assert validate_exam_config(config) == []

    def test_empty_questions(self):
        """Empty questions list produces no errors."""
        assert validate_exam_config({"questions": []}) == []

    def test_multiple_errors_collected(self):
        """Errors from multiple questions are all collected."""
        config = {
            "questions": [
                {"sn": 1, "question_type": "invalid1"},
                {"sn": 2, "question_type": "invalid2"},
            ]
        }
        errors = validate_exam_config(config)
        assert len(errors) == 2
