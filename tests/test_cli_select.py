"""Tests for cli_select module — forma-select command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


def _write_formative_test(path: Path) -> Path:
    """Write a sample FormativeTest YAML with 5 questions."""
    data = {
        "metadata": {
            "chapter": "Ch01",
            "topic": "서론",
        },
        "questions": [
            {"sn": 1, "topic": "세포", "text": "세포의 기본 구조를 설명하시오.", "limit": "50자"},
            {"sn": 2, "topic": "조직", "text": "조직의 종류를 나열하시오.", "limit": "100자"},
            {"sn": 3, "topic": "기관", "text": "기관계의 역할을 서술하시오.", "limit": "80자"},
            {"sn": 4, "topic": "항상성", "text": "항상성 유지 기전을 설명하시오.", "limit": "120자"},
            {"sn": 5, "topic": "해부학", "text": "해부학적 자세를 설명하시오.", "limit": "60자"},
        ],
        "pdf_questions": [
            {"topic": "세포", "text": "세포의 기본 구조를 설명하시오.", "limit": "50자"},
            {"topic": "조직", "text": "조직의 종류를 나열하시오.", "limit": "100자"},
            {"topic": "기관", "text": "기관계의 역할을 서술하시오.", "limit": "80자"},
            {"topic": "항상성", "text": "항상성 유지 기전을 설명하시오.", "limit": "120자"},
            {"topic": "해부학", "text": "해부학적 자세를 설명하시오.", "limit": "60자"},
        ],
    }
    source_path = path / "FormativeTest.yaml"
    with open(source_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    return source_path


def _write_week_yaml(path: Path, source_name: str = "FormativeTest.yaml", **overrides) -> Path:
    """Write a sample week.yaml with select section."""
    data = {
        "week": 1,
        "select": {
            "source": source_name,
            "questions": [1, 3],
            "num_papers": 220,
            "form_url": "https://forms.google.com/d/e/xxx/viewform?usp=pp_url&entry.1={student_id}",
            "exam_output": "",
        },
    }
    select = data["select"]
    for k, v in overrides.items():
        if k in select:
            select[k] = v
        else:
            data[k] = v
    yaml_path = path / "week.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


# ---------------------------------------------------------------------------
# T024: Extract correct questions by sn
# ---------------------------------------------------------------------------


class TestExtractQuestions:
    """Test _extract_questions() filters correctly by sn."""

    def test_extracts_correct_questions(self, tmp_path: Path) -> None:
        """Extract questions with sn 1 and 3 from source."""
        from forma.cli_select import _extract_questions

        source_path = _write_formative_test(tmp_path)
        questions = _extract_questions(source_path, [1, 3])

        assert len(questions) == 2
        sns = [q["sn"] for q in questions]
        assert sns == [1, 3]
        assert questions[0]["topic"] == "세포"
        assert questions[1]["topic"] == "기관"


# ---------------------------------------------------------------------------
# T025: Write questions.yaml with metadata
# ---------------------------------------------------------------------------


class TestWriteQuestionsYaml:
    """Test _write_questions_yaml() output."""

    def test_writes_with_metadata(self, tmp_path: Path) -> None:
        """questions.yaml includes source, selected_sn, week, num_papers, form_url."""
        from forma.cli_select import _write_questions_yaml

        questions = [
            {"sn": 1, "topic": "세포", "text": "세포의 기본 구조를 설명하시오.", "limit": "50자"},
            {"sn": 3, "topic": "기관", "text": "기관계의 역할을 서술하시오.", "limit": "80자"},
        ]
        metadata = {
            "source": "FormativeTest.yaml",
            "selected_sn": [1, 3],
            "week": 1,
            "num_papers": 220,
            "form_url": "https://forms.google.com/xxx",
        }
        output = tmp_path / "questions.yaml"
        _write_questions_yaml(questions, metadata, output)

        with open(output, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["source"] == "FormativeTest.yaml"
        assert data["selected_sn"] == [1, 3]
        assert data["week"] == 1
        assert data["num_papers"] == 220
        assert data["form_url"] == "https://forms.google.com/xxx"
        assert len(data["questions"]) == 2


# ---------------------------------------------------------------------------
# T026: Generates exam PDF when exam_output specified
# ---------------------------------------------------------------------------


class TestGenerateExamPdf:
    """Test PDF generation via ExamPDFGenerator."""

    @patch("forma.cli_select.ExamPDFGenerator")
    def test_generates_pdf_when_exam_output_set(self, mock_gen_cls, tmp_path: Path) -> None:
        """PDF generated when select.exam_output is specified."""
        from forma.cli_select import main

        _write_formative_test(tmp_path)
        pdf_path = str(tmp_path / "exam.pdf")
        _write_week_yaml(tmp_path, exam_output=pdf_path)

        mock_gen = MagicMock()
        mock_gen_cls.return_value = mock_gen

        with patch("forma.cli_select.find_week_config", return_value=tmp_path / "week.yaml"):
            result = main(["--week-config", str(tmp_path / "week.yaml")])

        assert result == 0
        mock_gen.create_exam_papers.assert_called_once()


# ---------------------------------------------------------------------------
# T027: Exit code 3 on invalid sn reference
# ---------------------------------------------------------------------------


class TestInvalidSn:
    """Test error on invalid sn reference."""

    def test_exit_code_3_on_invalid_sn(self, tmp_path: Path) -> None:
        """Exit code 3 when sn not found in source."""
        from forma.cli_select import _extract_questions

        source = _write_formative_test(tmp_path)
        with pytest.raises(ValueError, match="99"):
            _extract_questions(source, [1, 99])


# ---------------------------------------------------------------------------
# T028: Exit code 1 when missing select section
# ---------------------------------------------------------------------------


class TestMissingSelectSection:
    """Test error when week.yaml has no select section."""

    def test_exit_code_1_missing_select(self, tmp_path: Path) -> None:
        """Exit code 1 when week.yaml missing select section."""
        from forma.cli_select import main

        yaml_path = tmp_path / "week.yaml"
        yaml_path.write_text("week: 1\n", encoding="utf-8")

        result = main(["--week-config", str(yaml_path)])
        assert result == 1


# ---------------------------------------------------------------------------
# T028a: Exit code 2 when source file not found
# ---------------------------------------------------------------------------


class TestSourceNotFound:
    """Test error when source FormativeTest file not found."""

    def test_exit_code_2_source_not_found(self, tmp_path: Path) -> None:
        """Exit code 2 when source file does not exist."""
        from forma.cli_select import main

        _write_week_yaml(tmp_path, source_name="nonexistent.yaml")

        result = main(["--week-config", str(tmp_path / "week.yaml")])
        assert result == 2


# ---------------------------------------------------------------------------
# T028b: Exit code 4 on PDF generation failure
# ---------------------------------------------------------------------------


class TestPdfGenerationFailure:
    """Test exit code 4 on PDF generation failure."""

    @patch("forma.cli_select.ExamPDFGenerator")
    def test_exit_code_4_on_pdf_failure(self, mock_gen_cls, tmp_path: Path) -> None:
        """Exit code 4 when ExamPDFGenerator raises."""
        from forma.cli_select import main

        _write_formative_test(tmp_path)
        pdf_path = str(tmp_path / "exam.pdf")
        _write_week_yaml(tmp_path, exam_output=pdf_path)

        mock_gen = MagicMock()
        mock_gen.create_exam_papers.side_effect = RuntimeError("PDF error")
        mock_gen_cls.return_value = mock_gen

        result = main(["--week-config", str(tmp_path / "week.yaml")])
        assert result == 4
