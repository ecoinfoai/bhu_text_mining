"""Tests for report_generator.py — student counseling PDF generation.

Font discovery is mocked to avoid OS dependency in CI.
"""

import os
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


SAMPLE_COUNSELING = {
    "students": [
        {
            "student_id": "S001",
            "questions": [
                {
                    "question_sn": 1,
                    "understanding_level": "Proficient",
                    "concept_coverage": 0.75,
                    "support_guidance": "세부 개념 보충 필요",
                    "misconceptions": ["삼투와 확산 혼동"],
                },
                {
                    "question_sn": 2,
                    "understanding_level": "Developing",
                    "concept_coverage": 0.40,
                    "support_guidance": "기초 개념 재학습 권장",
                    "misconceptions": [],
                },
            ],
        },
        {
            "student_id": "S002",
            "questions": [
                {
                    "question_sn": 1,
                    "understanding_level": "Advanced",
                    "concept_coverage": 0.95,
                    "support_guidance": "",
                    "misconceptions": [],
                },
            ],
        },
    ],
}

SAMPLE_CONFIG = {
    "course_name": "생물학개론",
    "questions": [
        {"sn": 1, "question": "세포막의 기능은?"},
        {"sn": 2, "question": "생체항상성이란?"},
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_font(tmp_path):
    """Create a fake font file for testing."""
    font_file = tmp_path / "FakeFont.ttf"
    # Write minimal TrueType header (enough for reportlab to not crash
    # during registration in tests that mock the build step)
    font_file.write_bytes(b"\x00" * 64)
    return str(font_file)


@pytest.fixture()
def generator(mock_font):
    """Create a StudentReportGenerator with mocked font registration."""
    with patch("src.report_generator.find_korean_font", return_value=mock_font):
        with patch("src.report_generator.pdfmetrics.registerFont"):
            with patch("src.report_generator.TTFont"):
                from src.report_generator import StudentReportGenerator
                return StudentReportGenerator(font_path=mock_font)


# ---------------------------------------------------------------------------
# StudentReportGenerator tests
# ---------------------------------------------------------------------------


class TestStudentReportGenerator:
    """Tests for StudentReportGenerator."""

    def test_find_student_exists(self, generator):
        """_find_student returns student entry when found."""
        result = generator._find_student(SAMPLE_COUNSELING, "S001")
        assert result is not None
        assert result["student_id"] == "S001"

    def test_find_student_missing(self, generator):
        """_find_student returns None when not found."""
        result = generator._find_student(SAMPLE_COUNSELING, "S999")
        assert result is None

    def test_generate_report_creates_pdf(self, generator, tmp_path):
        """generate_report creates a PDF file."""
        output = str(tmp_path / "S001_report.pdf")
        with patch("src.report_generator.SimpleDocTemplate") as mock_doc_cls:
            with patch("src.report_generator.Paragraph"):
                with patch("src.report_generator.Spacer"):
                    mock_doc = MagicMock()
                    mock_doc_cls.return_value = mock_doc
                    path = generator.generate_report(
                        student_id="S001",
                        counseling_data=SAMPLE_COUNSELING,
                        config_data=SAMPLE_CONFIG,
                        output_path=output,
                    )
                    mock_doc.build.assert_called_once()
                    assert path == os.path.abspath(output)

    def test_generate_report_missing_student(self, generator, tmp_path):
        """generate_report handles missing student gracefully."""
        output = str(tmp_path / "S999_report.pdf")
        with patch("src.report_generator.SimpleDocTemplate") as mock_doc_cls:
            with patch("src.report_generator.Paragraph"):
                with patch("src.report_generator.Spacer"):
                    mock_doc = MagicMock()
                    mock_doc_cls.return_value = mock_doc
                    generator.generate_report(
                        student_id="S999",
                        counseling_data=SAMPLE_COUNSELING,
                        config_data=SAMPLE_CONFIG,
                        output_path=output,
                    )
                    mock_doc.build.assert_called_once()

    def test_generate_all_reports(self, generator, tmp_path):
        """generate_all_reports generates one PDF per student."""
        reports_dir = str(tmp_path / "reports")
        with patch.object(generator, "generate_report") as mock_gen:
            mock_gen.return_value = "path"
            paths = generator.generate_all_reports(
                counseling_data=SAMPLE_COUNSELING,
                config_data=SAMPLE_CONFIG,
                output_dir=reports_dir,
            )
            assert len(paths) == 2
            assert mock_gen.call_count == 2

    def test_generate_all_reports_creates_dir(self, generator, tmp_path):
        """generate_all_reports creates output directory."""
        reports_dir = str(tmp_path / "deep" / "nested" / "reports")
        with patch.object(generator, "generate_report", return_value="p"):
            generator.generate_all_reports(
                counseling_data=SAMPLE_COUNSELING,
                config_data=SAMPLE_CONFIG,
                output_dir=reports_dir,
            )
            assert os.path.isdir(reports_dir)


class TestFontUtils:
    """Tests for font_utils.find_korean_font()."""

    def test_raises_when_no_font(self):
        """Raises FileNotFoundError when no font is found."""
        from src.font_utils import find_korean_font

        with patch("src.font_utils.os.path.exists", return_value=False):
            with patch("src.font_utils.glob.glob", return_value=[]):
                with pytest.raises(FileNotFoundError, match="Korean font"):
                    find_korean_font()

    def test_returns_first_found(self):
        """Returns the first font path found."""
        from src.font_utils import find_korean_font

        def fake_exists(path):
            return path.endswith("NanumGothic.ttf") and "truetype" in path

        with patch("src.font_utils.platform.system", return_value="Linux"):
            with patch("src.font_utils.os.path.exists", side_effect=fake_exists):
                with patch("src.font_utils.glob.glob", return_value=[]):
                    result = find_korean_font()
                    assert "NanumGothic.ttf" in result
