# tests/test_exam_generator.py
"""TDD RED tests for ExamPDFGenerator QR code + layout upgrade."""
import os
import tempfile

import pytest
from PIL import Image as PILImage

from src.exam_generator import ExamPDFGenerator


# ──────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────

@pytest.fixture
def generator() -> ExamPDFGenerator:
    font_path = _find_nanum_font()
    return ExamPDFGenerator(font_path=font_path)


def _find_nanum_font() -> str:
    """Locate NanumGothic.ttf across OS variants."""
    import glob

    candidates = [
        "/usr/share/fonts/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/Library/Fonts/NanumGothic.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
    ]
    # NixOS store paths
    candidates += glob.glob(
        "/nix/store/*/share/fonts/NanumGothic.ttf",
    )
    for p in candidates:
        if os.path.exists(p):
            return p
    pytest.skip("NanumGothic.ttf not found on this system")


@pytest.fixture
def sample_questions() -> list:
    return [
        {
            "topic": "개념이해",
            "text": "미생물의 정의를 서술하시오.",
            "limit": "100자 내외",
        },
        {
            "topic": "적용",
            "text": "감염 경로 3가지를 설명하시오.",
            "limit": "150자 내외",
        },
    ]


# ──────────────────────────────────────────────────
# Group 1: 입력 검증 (5개)
# ──────────────────────────────────────────────────

class TestInputValidation:
    """FR-VAL-001: input validation tests."""

    def test_create_exam_papers_empty_questions_raises(
        self, generator,
    ):
        with pytest.raises(ValueError, match="questions must not be empty"):
            generator.create_exam_papers(
                questions=[],
                num_papers=1,
                output_path="/tmp/test.pdf",
            )

    def test_create_exam_papers_invalid_num_papers_raises(
        self, generator, sample_questions,
    ):
        with pytest.raises(ValueError, match="num_papers must be a positive"):
            generator.create_exam_papers(
                questions=sample_questions,
                num_papers=-1,
                output_path="/tmp/test.pdf",
            )

    def test_create_exam_papers_question_missing_keys_raises(
        self, generator,
    ):
        bad_questions = [{"topic": "개념이해", "text": "문제"}]
        with pytest.raises(ValueError, match="missing keys"):
            generator.create_exam_papers(
                questions=bad_questions,
                num_papers=1,
                output_path="/tmp/test.pdf",
            )

    def test_create_exam_papers_student_ids_length_mismatch_raises(
        self, generator, sample_questions,
    ):
        with pytest.raises(
            ValueError, match="student_ids length",
        ):
            generator.create_exam_papers(
                questions=sample_questions,
                num_papers=3,
                output_path="/tmp/test.pdf",
                student_ids=["S001", "S002"],
            )

    def test_create_exam_papers_form_url_without_placeholder_raises(
        self, generator, sample_questions,
    ):
        with pytest.raises(
            ValueError, match="student_id.*placeholder",
        ):
            generator.create_exam_papers(
                questions=sample_questions,
                num_papers=1,
                output_path="/tmp/test.pdf",
                form_url_template="https://example.com/form",
            )

    def test_create_exam_papers_num_papers_zero_raises(
        self, generator, sample_questions,
    ):
        with pytest.raises(ValueError, match="positive"):
            generator.create_exam_papers(
                questions=sample_questions,
                num_papers=0,
                output_path="/tmp/test.pdf",
            )


# ──────────────────────────────────────────────────
# Group 2: 학생 ID 생성 (2개)
# ──────────────────────────────────────────────────

class TestStudentIdGeneration:
    """FR-QR-003: student ID generation tests."""

    def test_generate_student_ids_default(self):
        ids = ExamPDFGenerator._generate_student_ids(3)
        assert ids == ["S001", "S002", "S003"]

    def test_generate_student_ids_custom_passthrough(self):
        custom = ["ABC", "DEF"]
        ids = ExamPDFGenerator._generate_student_ids(
            2, student_ids=custom,
        )
        assert ids == custom


# ──────────────────────────────────────────────────
# Group 3: QR 코드 생성 (3개)
# ──────────────────────────────────────────────────

class TestQRCodeGeneration:
    """FR-QR-001, FR-QR-002: QR code generation tests."""

    def test_generate_qr_image_returns_pil_image(self):
        img = ExamPDFGenerator._generate_qr_image("test")
        assert isinstance(img, PILImage.Image)

    def test_generate_qr_image_with_korean_url_encoding(self):
        url = (
            "https://docs.google.com/forms/d/e/FORM/viewform"
            "?entry.1=S001"
            "&entry.2=%EA%B0%90%EC%97%BC%EB%AF%B8%EC%83%9D%EB%AC%BC%ED%95%99"
        )
        img = ExamPDFGenerator._generate_qr_image(url)
        assert isinstance(img, PILImage.Image)
        assert img.size[0] > 0 and img.size[1] > 0

    def test_generate_qr_image_fallback_plain_text(self):
        plain = "S001|감염미생물학|3주차"
        img = ExamPDFGenerator._generate_qr_image(plain)
        assert isinstance(img, PILImage.Image)


# ──────────────────────────────────────────────────
# Group 4: URL 포맷팅 (2개)
# ──────────────────────────────────────────────────

class TestURLFormatting:
    """FR-QR-004: URL formatting tests."""

    def test_format_qr_content_with_template(self):
        template = (
            "https://docs.google.com/forms/d/e/FORM/viewform"
            "?entry.1={student_id}"
            "&entry.2={course_name}"
            "&entry.3={week_num}"
        )
        content = ExamPDFGenerator._format_qr_content(
            student_id="S001",
            course_name="감염미생물학",
            week_num=3,
            form_url_template=template,
        )
        assert "S001" in content
        assert "%EA%B0%90%EC%97%BC" in content
        assert "3" in content
        assert content.startswith("https://")

    def test_format_qr_content_without_template_plain_text(self):
        content = ExamPDFGenerator._format_qr_content(
            student_id="S001",
            course_name="감염미생물학",
            week_num=3,
        )
        assert content == "S001|감염미생물학|3주차"


# ──────────────────────────────────────────────────
# Group 5: 페이지 렌더링 (3개)
# ──────────────────────────────────────────────────

class TestPageRendering:
    """FR-FONT-001/002/003, FR-LAYOUT-001/002: rendering tests."""

    def test_draw_page_produces_valid_pdf(
        self, generator, sample_questions,
    ):
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False,
        ) as f:
            path = f.name
        try:
            generator.create_exam_papers(
                questions=sample_questions,
                num_papers=1,
                output_path=path,
            )
            size = os.path.getsize(path)
            assert size > 0
            with open(path, "rb") as f:
                header = f.read(5)
            assert header == b"%PDF-"
        finally:
            os.unlink(path)

    def test_draw_page_with_qr_produces_larger_pdf(
        self, generator, sample_questions,
    ):
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False,
        ) as f:
            path_no_qr = f.name
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False,
        ) as f:
            path_qr = f.name

        template = (
            "https://forms.example.com/form"
            "?sid={student_id}"
            "&course={course_name}"
            "&week={week_num}"
        )
        try:
            generator.create_exam_papers(
                questions=sample_questions,
                num_papers=1,
                output_path=path_no_qr,
            )
            generator.create_exam_papers(
                questions=sample_questions,
                num_papers=1,
                output_path=path_qr,
                form_url_template=template,
                student_ids=["S001"],
            )
            size_no_qr = os.path.getsize(path_no_qr)
            size_qr = os.path.getsize(path_qr)
            assert size_qr > size_no_qr
        finally:
            os.unlink(path_no_qr)
            os.unlink(path_qr)

    def test_font_sizes_applied_correctly(self, generator):
        assert generator.FONT_SIZE_TITLE == 20
        assert generator.FONT_SIZE_BODY == 14
        assert generator.FONT_SIZE_FOOTER == 14

    def test_layout_constants(self, generator):
        """FR-LAYOUT-002: answer area and guideline constants."""
        assert generator.ANSWER_HEIGHT_MM == 63
        assert generator.ANSWER_NUM_LINES == 7
        assert generator.QR_SIZE_MM == 22

    def test_qr_import_error_when_missing(self):
        """Verify ImportError when qrcode is unavailable."""
        import src.exam_generator as mod

        original = mod._HAS_QRCODE
        try:
            mod._HAS_QRCODE = False
            with pytest.raises(
                ImportError, match="qrcode package",
            ):
                ExamPDFGenerator._generate_qr_image("test")
        finally:
            mod._HAS_QRCODE = original


# ──────────────────────────────────────────────────
# Group 6: 통합 / 하위호환 (2개)
# ──────────────────────────────────────────────────

class TestIntegration:
    """FR-COMPAT-001: backward compatibility tests."""

    def test_create_exam_papers_generates_correct_page_count(
        self, generator, sample_questions,
    ):
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False,
        ) as f:
            path = f.name
        try:
            generator.create_exam_papers(
                questions=sample_questions,
                num_papers=5,
                output_path=path,
                student_ids=["S001", "S002", "S003", "S004", "S005"],
                form_url_template=(
                    "https://example.com?sid={student_id}"
                    "&c={course_name}&w={week_num}"
                ),
            )
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_create_exam_papers_backward_compatible(
        self, generator, sample_questions,
    ):
        """Calling without new params must work identically."""
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False,
        ) as f:
            path = f.name
        try:
            generator.create_exam_papers(
                questions=sample_questions,
                num_papers=2,
                output_path=path,
            )
            size = os.path.getsize(path)
            assert size > 0
            with open(path, "rb") as f:
                header = f.read(5)
            assert header == b"%PDF-"
        finally:
            os.unlink(path)
