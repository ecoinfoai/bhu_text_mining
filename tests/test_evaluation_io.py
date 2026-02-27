"""Tests for evaluation_io.py YAML I/O utilities.

RED phase: validate file loading, saving, and student-response extraction.
"""

import os
import tempfile

import pytest
import yaml

from src.evaluation_io import (
    load_evaluation_yaml,
    save_evaluation_yaml,
    extract_student_responses,
)


@pytest.fixture()
def sample_yaml_file(tmp_path):
    """Write a minimal evaluation YAML and return its path."""
    data = {
        "metadata": {"chapter": 1, "course_name": "인체구조와기능"},
        "responses": {
            "s001": {1: "세포막은 인지질 이중층으로 구성됩니다."},
            "s002": {1: "세포막은 물질 이동을 조절합니다."},
        },
    }
    p = tmp_path / "eval.yaml"
    p.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
    return str(p)


class TestLoadEvaluationYaml:
    """Tests for load_evaluation_yaml()."""

    def test_load_returns_dict(self, sample_yaml_file):
        """Test that a valid YAML file loads as a dict."""
        data = load_evaluation_yaml(sample_yaml_file)
        assert isinstance(data, dict)
        assert "metadata" in data

    def test_load_preserves_content(self, sample_yaml_file):
        """Test that loaded content matches written content."""
        data = load_evaluation_yaml(sample_yaml_file)
        assert data["metadata"]["chapter"] == 1
        assert data["metadata"]["course_name"] == "인체구조와기능"

    def test_load_missing_file_raises_file_not_found(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        missing = str(tmp_path / "missing.yaml")
        with pytest.raises(FileNotFoundError, match="missing.yaml"):
            load_evaluation_yaml(missing)


class TestSaveEvaluationYaml:
    """Tests for save_evaluation_yaml()."""

    def test_save_creates_file(self, tmp_path):
        """Test that save creates the output file."""
        data = {"result": "ok", "score": 0.75}
        out = str(tmp_path / "out" / "result.yaml")
        save_evaluation_yaml(data, out)
        assert os.path.exists(out)

    def test_save_roundtrip(self, tmp_path):
        """Test save → load round-trip preserves data."""
        data = {"student": "s001", "score": 0.85, "level": "Advanced"}
        out = str(tmp_path / "result.yaml")
        save_evaluation_yaml(data, out)
        loaded = load_evaluation_yaml(out)
        assert loaded["student"] == "s001"
        assert loaded["score"] == pytest.approx(0.85)

    def test_save_unicode_preserved(self, tmp_path):
        """Test that Korean characters survive save/load."""
        data = {"level": "우수", "comment": "세포막 개념 완벽 이해"}
        out = str(tmp_path / "kr.yaml")
        save_evaluation_yaml(data, out)
        loaded = load_evaluation_yaml(out)
        assert loaded["level"] == "우수"
        assert "세포막" in loaded["comment"]

    def test_save_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created automatically."""
        out = str(tmp_path / "deep" / "nested" / "result.yaml")
        save_evaluation_yaml({"x": 1}, out)
        assert os.path.exists(out)


class TestExtractStudentResponses:
    """Tests for extract_student_responses()."""

    def test_extract_basic_structure(self):
        """Test extraction returns student_id → {question_sn → text}."""
        data = {
            "responses": {
                "s001": {1: "세포막은 인지질 이중층.", 2: "삼투는 농도 차이."},
                "s002": {1: "세포막은 물질 이동 조절."},
            }
        }
        result = extract_student_responses(data)
        assert "s001" in result
        assert 1 in result["s001"]
        assert result["s001"][1] == "세포막은 인지질 이중층."

    def test_extract_multiple_questions(self):
        """Test extraction handles multiple questions per student."""
        data = {
            "responses": {
                "s001": {1: "답변1", 2: "답변2", 3: "답변3"},
            }
        }
        result = extract_student_responses(data)
        assert len(result["s001"]) == 3

    def test_extract_missing_responses_key_raises(self):
        """Test that missing 'responses' key raises KeyError."""
        with pytest.raises(KeyError, match="responses"):
            extract_student_responses({"metadata": {}})

    def test_extract_empty_responses_returns_empty(self):
        """Test that empty responses dict returns empty result."""
        result = extract_student_responses({"responses": {}})
        assert result == {}
