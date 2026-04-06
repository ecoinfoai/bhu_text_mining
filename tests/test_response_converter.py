"""Tests for response_converter.py — join → evaluation format conversion."""

import yaml

import pytest

from forma.response_converter import (
    convert_join_file,
    convert_join_to_responses,
    filter_exam_config,
)


# ---------------------------------------------------------------------------
# convert_join_to_responses tests
# ---------------------------------------------------------------------------


class TestConvertJoinToResponses:
    """Tests for convert_join_to_responses()."""

    def test_basic_conversion(self):
        """Flat list converts to nested responses dict."""
        join_data = [
            {"student_id": "S015", "q_num": 1, "text": "생체항상성은..."},
            {"student_id": "S015", "q_num": 2, "text": "세포막은..."},
            {"student_id": "S020", "q_num": 1, "text": "항상성이란..."},
        ]
        result = convert_join_to_responses(join_data)

        assert "responses" in result
        assert result["responses"]["S015"][1] == "생체항상성은..."
        assert result["responses"]["S015"][2] == "세포막은..."
        assert result["responses"]["S020"][1] == "항상성이란..."

    def test_empty_list(self):
        """Empty list produces empty responses."""
        result = convert_join_to_responses([])
        assert result == {"responses": {}}

    def test_missing_text_defaults_empty(self):
        """Missing text field defaults to empty string."""
        join_data = [{"student_id": "S001", "q_num": 1}]
        result = convert_join_to_responses(join_data)
        assert result["responses"]["S001"][1] == ""

    def test_student_id_coerced_to_str(self):
        """Numeric student_id is coerced to string."""
        join_data = [{"student_id": 42, "q_num": 1, "text": "answer"}]
        result = convert_join_to_responses(join_data)
        assert "42" in result["responses"]

    def test_q_num_coerced_to_int(self):
        """String q_num is coerced to int."""
        join_data = [{"student_id": "S001", "q_num": "2", "text": "answer"}]
        result = convert_join_to_responses(join_data)
        assert 2 in result["responses"]["S001"]

    def test_questions_used_mapping(self):
        """questions_used remaps q_num to exam sn."""
        join_data = [
            {"student_id": "S001", "q_num": 1, "text": "항상성은..."},
            {"student_id": "S001", "q_num": 2, "text": "양성되먹임은..."},
            {"student_id": "S002", "q_num": 1, "text": "항상성이란..."},
        ]
        result = convert_join_to_responses(join_data, questions_used=[1, 3])

        # q1→sn1, q2→sn3
        assert result["responses"]["S001"][1] == "항상성은..."
        assert result["responses"]["S001"][3] == "양성되먹임은..."
        assert 2 not in result["responses"]["S001"]
        assert result["responses"]["S002"][1] == "항상성이란..."

    def test_questions_used_skips_unmapped(self):
        """Entries with q_num not in mapping are skipped."""
        join_data = [
            {"student_id": "S001", "q_num": 1, "text": "a"},
            {"student_id": "S001", "q_num": 3, "text": "should be skipped"},
        ]
        result = convert_join_to_responses(join_data, questions_used=[1, 3])
        # q1→sn1 OK, q3 has no mapping (only q1,q2 mapped)
        assert 1 in result["responses"]["S001"]
        assert 3 not in result["responses"]["S001"]


# ---------------------------------------------------------------------------
# convert_join_file tests
# ---------------------------------------------------------------------------


class TestConvertJoinFile:
    """Tests for convert_join_file()."""

    def test_file_conversion(self, tmp_path):
        """Writes correctly formatted YAML file."""
        join_data = [
            {"student_id": "S001", "q_num": 1, "text": "답변1"},
            {"student_id": "S002", "q_num": 1, "text": "답변2"},
        ]
        join_file = tmp_path / "join.yaml"
        join_file.write_text(yaml.dump(join_data, allow_unicode=True), encoding="utf-8")

        out_file = tmp_path / "output" / "responses.yaml"
        convert_join_file(str(join_file), str(out_file))

        assert out_file.exists()
        with open(out_file, "r", encoding="utf-8") as f:
            result = yaml.safe_load(f)
        assert result["responses"]["S001"][1] == "답변1"
        assert result["responses"]["S002"][1] == "답변2"

    def test_creates_parent_dirs(self, tmp_path):
        """Output parent directories are created automatically."""
        join_data = [{"student_id": "S001", "q_num": 1, "text": "x"}]
        join_file = tmp_path / "join.yaml"
        join_file.write_text(yaml.dump(join_data, allow_unicode=True), encoding="utf-8")

        out_file = tmp_path / "deep" / "nested" / "dir" / "out.yaml"
        convert_join_file(str(join_file), str(out_file))
        assert out_file.exists()

    def test_file_with_questions_used(self, tmp_path):
        """File conversion applies questions_used mapping."""
        join_data = [
            {"student_id": "S001", "q_num": 1, "text": "ans1"},
            {"student_id": "S001", "q_num": 2, "text": "ans2"},
        ]
        join_file = tmp_path / "join.yaml"
        join_file.write_text(yaml.dump(join_data, allow_unicode=True), encoding="utf-8")

        out_file = tmp_path / "out.yaml"
        convert_join_file(str(join_file), str(out_file), questions_used=[1, 3])

        with open(out_file, "r", encoding="utf-8") as f:
            result = yaml.safe_load(f)
        assert result["responses"]["S001"][1] == "ans1"
        assert result["responses"]["S001"][3] == "ans2"

    def test_non_list_input_raises(self, tmp_path):
        """Non-list YAML input raises ValueError."""
        join_file = tmp_path / "bad.yaml"
        join_file.write_text("key: value", encoding="utf-8")

        with pytest.raises(ValueError, match="Expected a list"):
            convert_join_file(str(join_file), str(tmp_path / "out.yaml"))


# ---------------------------------------------------------------------------
# filter_exam_config tests
# ---------------------------------------------------------------------------


class TestFilterExamConfig:
    """Tests for filter_exam_config()."""

    def test_filters_to_selected_questions(self):
        """Selected questions are kept and renumbered to 1, 2, ..."""
        config = {
            "metadata": {"chapter": 1},
            "questions": [
                {"sn": 1, "question": "Q1"},
                {"sn": 2, "question": "Q2"},
                {"sn": 3, "question": "Q3"},
                {"sn": 4, "question": "Q4"},
                {"sn": 5, "question": "Q5"},
            ],
        }
        result = filter_exam_config(config, [1, 3])
        assert len(result["questions"]) == 2
        # Renumbered to 1, 2 (student-facing)
        assert result["questions"][0]["sn"] == 1
        assert result["questions"][0]["question"] == "Q1"
        assert result["questions"][1]["sn"] == 2
        assert result["questions"][1]["question"] == "Q3"
        assert result["metadata"] == {"chapter": 1}

    def test_missing_sn_raises(self):
        """KeyError when questions_used contains unknown sn."""
        config = {"questions": [{"sn": 1}, {"sn": 2}]}
        with pytest.raises(KeyError, match="sn \\[99\\]"):
            filter_exam_config(config, [1, 99])
