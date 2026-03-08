"""Tests for pipeline_evaluation.py — serialization helpers.

T008: Tests for _serialize_graph_comparison() edge list serialization.
Tests are written in RED phase — they will fail until T009 implements
the matched_edges, missing_edges, extra_edges, wrong_direction_edges
edge-list fields in _serialize_graph_comparison().
"""

from __future__ import annotations

import pytest

from forma.evaluation_types import (
    GraphComparisonResult,
    TripletEdge,
)


def _make_edge(subject: str, relation: str, obj: str) -> TripletEdge:
    """Helper to create a TripletEdge."""
    return TripletEdge(subject=subject, relation=relation, object=obj)


def _make_gcr(
    student_id: str = "S001",
    question_sn: int = 1,
    matched: list[TripletEdge] | None = None,
    missing: list[TripletEdge] | None = None,
    extra: list[TripletEdge] | None = None,
    wrong_direction: list[TripletEdge] | None = None,
    precision: float = 0.75,
    recall: float = 0.60,
    f1: float = 0.67,
) -> GraphComparisonResult:
    """Build a minimal GraphComparisonResult for testing."""
    return GraphComparisonResult(
        student_id=student_id,
        question_sn=question_sn,
        precision=precision,
        recall=recall,
        f1=f1,
        matched_edges=matched or [],
        missing_edges=missing or [],
        extra_edges=extra or [],
        wrong_direction_edges=wrong_direction or [],
    )


def _call_serialize(results: dict) -> dict:
    """Import and call the module-level _serialize_graph_comparison()."""
    from forma.pipeline_evaluation import _serialize_graph_comparison

    return _serialize_graph_comparison(results)


class TestSerializeGraphComparisonEdgeLists:
    """T008: Tests that _serialize_graph_comparison includes edge lists."""

    def test_matched_edges_key_present(self):
        """Each question dict must contain 'matched_edges' key."""
        edge = _make_edge("수용체", "감지", "한계점 일탈")
        gcr = _make_gcr(matched=[edge])
        result = _call_serialize({"S001": {1: gcr}})
        q_dict = result["students"][0]["questions"][0]
        assert "matched_edges" in q_dict

    def test_missing_edges_key_present(self):
        """Each question dict must contain 'missing_edges' key."""
        edge = _make_edge("효과기", "반응", "열 생성")
        gcr = _make_gcr(missing=[edge])
        result = _call_serialize({"S001": {1: gcr}})
        q_dict = result["students"][0]["questions"][0]
        assert "missing_edges" in q_dict

    def test_extra_edges_key_present(self):
        """Each question dict must contain 'extra_edges' key."""
        edge = _make_edge("세포", "구성", "단백질")
        gcr = _make_gcr(extra=[edge])
        result = _call_serialize({"S001": {1: gcr}})
        q_dict = result["students"][0]["questions"][0]
        assert "extra_edges" in q_dict

    def test_wrong_direction_edges_key_present(self):
        """Each question dict must contain 'wrong_direction_edges' key."""
        edge = _make_edge("뇌", "제어", "심박수")
        gcr = _make_gcr(wrong_direction=[edge])
        result = _call_serialize({"S001": {1: gcr}})
        q_dict = result["students"][0]["questions"][0]
        assert "wrong_direction_edges" in q_dict

    def test_matched_edges_content_subject_relation_object(self):
        """matched_edges items must have 'subject', 'relation', 'object' keys."""
        edge = _make_edge("수용체", "감지", "한계점 일탈")
        gcr = _make_gcr(matched=[edge])
        result = _call_serialize({"S001": {1: gcr}})
        matched = result["students"][0]["questions"][0]["matched_edges"]
        assert len(matched) == 1
        assert matched[0]["subject"] == "수용체"
        assert matched[0]["relation"] == "감지"
        assert matched[0]["object"] == "한계점 일탈"

    def test_missing_edges_content_subject_relation_object(self):
        """missing_edges items must have 'subject', 'relation', 'object' keys."""
        edge = _make_edge("효과기", "반응", "열 생성")
        gcr = _make_gcr(missing=[edge])
        result = _call_serialize({"S001": {1: gcr}})
        missing = result["students"][0]["questions"][0]["missing_edges"]
        assert len(missing) == 1
        assert missing[0]["subject"] == "효과기"
        assert missing[0]["relation"] == "반응"
        assert missing[0]["object"] == "열 생성"

    def test_extra_edges_content_subject_relation_object(self):
        """extra_edges items must have 'subject', 'relation', 'object' keys."""
        edge = _make_edge("세포", "구성", "단백질")
        gcr = _make_gcr(extra=[edge])
        result = _call_serialize({"S001": {1: gcr}})
        extra = result["students"][0]["questions"][0]["extra_edges"]
        assert len(extra) == 1
        assert extra[0]["subject"] == "세포"
        assert extra[0]["relation"] == "구성"
        assert extra[0]["object"] == "단백질"

    def test_wrong_direction_edges_content_subject_relation_object(self):
        """wrong_direction_edges items must have 'subject', 'relation', 'object' keys."""
        edge = _make_edge("뇌", "제어", "심박수")
        gcr = _make_gcr(wrong_direction=[edge])
        result = _call_serialize({"S001": {1: gcr}})
        wrong = result["students"][0]["questions"][0]["wrong_direction_edges"]
        assert len(wrong) == 1
        assert wrong[0]["subject"] == "뇌"
        assert wrong[0]["relation"] == "제어"
        assert wrong[0]["object"] == "심박수"

    def test_empty_edge_lists_serialize_as_empty_lists(self):
        """When GCR has no edges, all edge list keys serialize as []."""
        gcr = _make_gcr()
        result = _call_serialize({"S001": {1: gcr}})
        q_dict = result["students"][0]["questions"][0]
        assert q_dict["matched_edges"] == []
        assert q_dict["missing_edges"] == []
        assert q_dict["extra_edges"] == []
        assert q_dict["wrong_direction_edges"] == []

    def test_multiple_edges_all_serialized(self):
        """Multiple edges in a list are all serialized."""
        edges = [
            _make_edge("A", "관련", "B"),
            _make_edge("C", "포함", "D"),
            _make_edge("E", "활성화", "F"),
        ]
        gcr = _make_gcr(matched=edges)
        result = _call_serialize({"S001": {1: gcr}})
        matched = result["students"][0]["questions"][0]["matched_edges"]
        assert len(matched) == 3
        assert matched[2]["subject"] == "E"

    def test_existing_count_fields_still_present(self):
        """Existing matched_count, missing_count etc. fields still present."""
        edge = _make_edge("X", "Y", "Z")
        gcr = _make_gcr(matched=[edge], missing=[edge])
        result = _call_serialize({"S001": {1: gcr}})
        q_dict = result["students"][0]["questions"][0]
        assert "matched_count" in q_dict
        assert "missing_count" in q_dict
        assert "precision" in q_dict
        assert "recall" in q_dict
        assert "f1" in q_dict

    def test_multiple_questions_all_have_edge_lists(self):
        """All questions across multiple students/questions include edge lists."""
        gcr1 = _make_gcr(student_id="S001", question_sn=1)
        gcr2 = _make_gcr(student_id="S001", question_sn=2)
        gcr3 = _make_gcr(student_id="S002", question_sn=1)
        results = {
            "S001": {1: gcr1, 2: gcr2},
            "S002": {1: gcr3},
        }
        output = _call_serialize(results)
        for student_entry in output["students"]:
            for q_dict in student_entry["questions"]:
                assert "matched_edges" in q_dict
                assert "missing_edges" in q_dict
                assert "extra_edges" in q_dict
                assert "wrong_direction_edges" in q_dict
