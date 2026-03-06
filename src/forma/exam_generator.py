# src/exam_generator.py
"""ExamPDFGenerator: formative exam PDF with optional QR codes.

Generates A4 exam papers with student ID QR codes for
privacy-preserving OCR workflows.
"""
import glob
import os
import platform
from typing import Dict, List, Optional
from urllib.parse import quote

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

try:
    import qrcode
    from PIL import Image as PILImage

    _HAS_QRCODE = True
except ImportError:  # pragma: no cover
    _HAS_QRCODE = False


class ExamPDFGenerator:
    """텍스트 기반 시험지 PDF 생성기 (QR 코드 지원)."""

    FONT_SIZE_TITLE: int = 20
    FONT_SIZE_BODY: int = 14
    FONT_SIZE_FOOTER: int = 14
    QR_SIZE_MM: int = 22
    ANSWER_HEIGHT_MM: int = 63
    ANSWER_NUM_LINES: int = 7
    _REQUIRED_KEYS = {"topic", "text", "limit"}

    def __init__(self, font_path: Optional[str] = None) -> None:
        if font_path is None:
            font_path = self._find_font()
        if not os.path.exists(font_path):
            raise FileNotFoundError(
                f"폰트를 찾을 수 없습니다: {font_path}\n"
                f"font_path 인자로 직접 경로를 지정하거나\n"
                f"나눔고딕 폰트를 설치하세요."
            )
        pdfmetrics.registerFont(
            TTFont("NanumGothic", font_path),
        )
        bold_path = font_path.replace(".ttf", "Bold.ttf")
        if os.path.exists(bold_path):
            pdfmetrics.registerFont(
                TTFont("NanumGothicBold", bold_path),
            )
        else:
            pdfmetrics.registerFont(
                TTFont("NanumGothicBold", font_path),
            )
        self.page_width, self.page_height = A4
        self.margin = 20 * mm

    # ── font discovery ───────────────────────────

    def _find_font(self) -> str:
        """OS별 폰트 자동 탐색."""
        system = platform.system()
        search_paths: List[str] = []
        if system == "Windows":
            search_paths = [
                "C:/Windows/Fonts/malgun.ttf",
                "C:/Windows/Fonts/NanumGothic.ttf",
            ]
        elif system == "Darwin":
            search_paths = [
                "/Library/Fonts/NanumGothic.ttf",
                "/System/Library/Fonts/AppleGothic.ttf",
            ]
        else:
            search_paths = [
                "/usr/share/fonts/nanum/NanumGothic.ttf",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            ]
            search_paths += glob.glob(
                "/nix/store/*/share/fonts/NanumGothic.ttf",
            )
        for path in search_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError("폰트를 찾을 수 없습니다.")

    # ── input validation ─────────────────────────

    @staticmethod
    def _validate_questions(
        questions: List[Dict[str, str]],
    ) -> None:
        """Validate questions list.

        Raises:
            ValueError: on empty list or missing keys.
        """
        if not questions:
            raise ValueError(
                "questions must not be empty. "
                "Provide at least one question dict "
                "with keys 'topic', 'text', 'limit'. "
                "[_validate_questions]"
            )
        required = ExamPDFGenerator._REQUIRED_KEYS
        for i, q in enumerate(questions):
            missing = required - set(q.keys())
            if missing:
                raise ValueError(
                    f"questions[{i}] is missing keys: "
                    f"{missing}. Each question must have "
                    "'topic', 'text', 'limit'. "
                    "[_validate_questions]"
                )

    # ── student ID generation ────────────────────

    @staticmethod
    def _generate_student_ids(
        num_papers: int,
        student_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate or validate student ID list.

        Args:
            num_papers: number of papers to generate.
            student_ids: custom IDs or None for auto.

        Returns:
            List of student ID strings.

        Raises:
            ValueError: when length mismatches num_papers.
        """
        if student_ids is not None:
            if len(student_ids) != num_papers:
                raise ValueError(
                    f"student_ids length ({len(student_ids)})"
                    f" must equal num_papers ({num_papers}). "
                    "[_generate_student_ids]"
                )
            return student_ids
        return [f"S{i:03d}" for i in range(1, num_papers + 1)]

    # ── QR content formatting ────────────────────

    @staticmethod
    def _format_qr_content(
        student_id: str,
        course_name: str,
        week_num: int,
        form_url_template: Optional[str] = None,
        q_num: Optional[int] = None,
    ) -> str:
        """Format QR code content string.

        Args:
            student_id: e.g. "S001".
            course_name: e.g. "감염미생물학".
            week_num: week number.
            form_url_template: Google Forms URL with
                {student_id}, {course_name}, {week_num}.
            q_num: question number for per-question QR.
                None omits the question field.

        Returns:
            URL string or plain-text fallback.
        """
        if form_url_template is None:
            base = f"{student_id}|{course_name}|{week_num}주차"
            if q_num is not None:
                return f"{base}|Q{q_num}"
            return base
        url = form_url_template.format(
            student_id=quote(student_id, safe=""),
            course_name=quote(course_name, safe=""),
            week_num=week_num,
        )
        if q_num is not None:
            url = f"{url}&q={q_num}"
        return url

    # ── QR image generation ──────────────────────

    @staticmethod
    def _generate_qr_image(
        content: str,
        size_mm: int = 22,
    ) -> "PILImage.Image":
        """Generate QR code as PIL Image.

        Args:
            content: data to encode.
            size_mm: target size in millimeters.

        Returns:
            PIL Image of the QR code.

        Raises:
            ImportError: if qrcode package missing.
        """
        if not _HAS_QRCODE:
            raise ImportError(
                "qrcode package is required. Install: "
                "pip install 'qrcode[pil]>=7.0'. "
                "[_generate_qr_image]"
            )
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=1,
        )
        qr.add_data(content)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        px = int(size_mm * 96 / 25.4)
        return img.resize((px, px))

    # ── page drawing sub-methods ─────────────────

    def _draw_header(
        self,
        c: canvas.Canvas,
        paper_num: int,
        year: int,
        grade: int,
        semester: int,
        course_name: str,
        week_num: int,
    ) -> float:
        """Draw serial number and title. Returns y_pos."""
        y = self.page_height - self.margin
        serial = f"{paper_num:04d}"
        c.setFont("NanumGothicBold", 36)
        c.drawRightString(
            self.page_width - self.margin,
            y - 8 * mm,
            serial,
        )
        c.setFont("NanumGothic", 9)
        c.setFillGray(0.5)
        c.drawRightString(
            self.page_width - self.margin,
            y - 14 * mm,
            "일련번호",
        )
        c.setFillGray(0)
        y -= 10 * mm
        c.setFont("NanumGothic", 10)
        line1 = (
            f"{year}학년도 {grade}학년 "
            f"{semester}학기 {course_name}"
        )
        c.drawCentredString(self.page_width / 2, y, line1)
        y -= 7 * mm
        c.setFont(
            "NanumGothicBold", self.FONT_SIZE_TITLE,
        )
        line2 = f"{week_num}주차 형성평가"
        c.drawCentredString(self.page_width / 2, y, line2)
        y -= 8 * mm
        c.setLineWidth(1)
        c.line(
            self.margin, y,
            self.page_width - self.margin, y,
        )
        return y

    def _draw_student_info(
        self,
        c: canvas.Canvas,
        y_pos: float,
    ) -> float:
        """Draw student info fields. Returns y_pos."""
        y = y_pos - 8 * mm
        body = self.FONT_SIZE_BODY
        c.setFont("NanumGothic", body)
        c.setLineWidth(0.5)
        # Line 1: 학년, 분반, 학번
        c.drawString(self.margin, y, "학년:")
        ul1 = self.margin + 15 * mm
        c.line(ul1, y - 1 * mm, ul1 + 20 * mm, y - 1 * mm)
        c.drawString(ul1 + 25 * mm, y, "분반:")
        ul2 = ul1 + 40 * mm
        c.line(ul2, y - 1 * mm, ul2 + 20 * mm, y - 1 * mm)
        c.drawString(ul2 + 25 * mm, y, "학번:")
        ul3 = ul2 + 40 * mm
        c.line(
            ul3, y - 1 * mm,
            self.page_width - self.margin, y - 1 * mm,
        )
        # Line 2: 이름
        y -= 8 * mm
        c.drawString(self.margin, y, "이름:")
        c.line(
            ul1, y - 1 * mm,
            ul1 + 40 * mm, y - 1 * mm,
        )
        y -= 6 * mm
        c.setLineWidth(0.5)
        c.line(
            self.margin, y,
            self.page_width - self.margin, y,
        )
        return y

    def _draw_question(
        self,
        c: canvas.Canvas,
        y_pos: float,
        q_num: int,
        question: Dict[str, str],
        qr_image: Optional["PILImage.Image"] = None,
    ) -> float:
        """Draw a single question with answer area.

        Returns y_pos after the question block.
        """
        y = y_pos - 10 * mm
        body = self.FONT_SIZE_BODY
        c.setFont("NanumGothicBold", body)
        c.setFillGray(0.3)
        c.drawString(
            self.margin, y, f"[{question['topic']}]",
        )
        c.setFillGray(0)
        y -= 6 * mm
        q_label = f"문제 {q_num}. "
        c.setFont("NanumGothicBold", body)
        num_w = c.stringWidth(q_label, "NanumGothicBold", body)
        c.drawString(self.margin, y, q_label)
        c.setFont("NanumGothic", body)
        full = f"{question['text']} ({question['limit']})"
        text_x = self.margin + num_w
        max_w = self.page_width - self.margin - text_x
        y = self._wrap_text(c, full, text_x, y, max_w, body)
        y -= 2 * mm
        ah = self.ANSWER_HEIGHT_MM * mm
        qr_mm = self.QR_SIZE_MM * mm
        if qr_image is not None:
            ans_w = (
                self.page_width - 2 * self.margin
                - qr_mm - 5 * mm
            )
        else:
            ans_w = self.page_width - 2 * self.margin
        c.setStrokeGray(0.7)
        c.setLineWidth(0.5)
        c.rect(self.margin, y - ah, ans_w, ah)
        self._draw_guidelines(c, y, ah, ans_w)
        if qr_image is not None:
            self._place_qr(c, y, ah, qr_image, ans_w)
        c.setStrokeGray(0)
        return y - ah - 5 * mm

    def _wrap_text(
        self,
        c: canvas.Canvas,
        text: str,
        x: float,
        y: float,
        max_w: float,
        font_size: int,
    ) -> float:
        """Wrap text with word-splitting. Returns y."""
        words = text.split()
        line = ""
        indent_x = self.margin + 5 * mm
        full_w = self.page_width - self.margin - indent_x
        first = True
        for word in words:
            test = line + word + " "
            w = max_w if first else full_w
            if c.stringWidth(test, "NanumGothic", font_size) <= w:
                line = test
            else:
                if line:
                    cx = x if first else indent_x
                    c.drawString(cx, y, line.strip())
                    y -= 5 * mm
                    first = False
                line = word + " "
        if line:
            cx = x if first else indent_x
            c.drawString(cx, y, line.strip())
            y -= 5 * mm
        return y

    def _draw_guidelines(
        self,
        c: canvas.Canvas,
        y: float,
        ah: float,
        ans_w: float,
    ) -> None:
        """Draw answer area guide lines."""
        c.setStrokeGray(0.85)
        c.setLineWidth(0.3)
        spacing = ah / (self.ANSWER_NUM_LINES + 1)
        for i in range(1, self.ANSWER_NUM_LINES + 1):
            ly = y - i * spacing
            c.line(
                self.margin + 2 * mm, ly,
                self.margin + ans_w - 2 * mm, ly,
            )

    def _place_qr(
        self,
        c: canvas.Canvas,
        y: float,
        ah: float,
        qr_image: "PILImage.Image",
        ans_w: float,
    ) -> None:
        """Place QR image to the right of answer area."""
        from reportlab.lib.utils import ImageReader

        qr_mm = self.QR_SIZE_MM * mm
        qr_x = self.margin + ans_w + 5 * mm
        qr_y = y - ah + (ah - qr_mm) / 2
        reader = ImageReader(qr_image)
        c.drawImage(
            reader, qr_x, qr_y, qr_mm, qr_mm,
        )

    def _draw_footer(self, c: canvas.Canvas) -> None:
        """Draw bottom instruction text."""
        c.setFillGray(0.5)
        c.setFont("NanumGothic", self.FONT_SIZE_FOOTER)
        c.drawCentredString(
            self.page_width / 2,
            self.margin - 5 * mm,
            "※ 답안은 검은색 볼펜으로 작성하세요.",
        )

    # ── page assembly ────────────────────────────

    def _draw_page(
        self,
        c: canvas.Canvas,
        questions: List[Dict[str, str]],
        paper_num: int,
        year: int,
        grade: int,
        semester: int,
        course_name: str,
        week_num: int,
        student_id: Optional[str] = None,
        form_url_template: Optional[str] = None,
    ) -> None:
        """Draw a single exam page with per-question QR codes."""
        y = self._draw_header(
            c, paper_num, year, grade,
            semester, course_name, week_num,
        )
        y = self._draw_student_info(c, y)
        for q_num, q in enumerate(questions, 1):
            qr_img = None
            if student_id and form_url_template:
                content = self._format_qr_content(
                    student_id, course_name, week_num,
                    form_url_template, q_num=q_num,
                )
                qr_img = self._generate_qr_image(
                    content, self.QR_SIZE_MM,
                )
            y = self._draw_question(c, y, q_num, q, qr_img)
        self._draw_footer(c)

    # ── input orchestration ────────────────────────

    @staticmethod
    def _validate_inputs(
        num_papers: int,
        form_url_template: Optional[str],
    ) -> None:
        """Validate scalar inputs at entry point.

        Raises:
            ValueError: on invalid num_papers or template.
        """
        if not isinstance(num_papers, int) or num_papers < 1:
            raise ValueError(
                "num_papers must be a positive integer, "
                f"got {num_papers!r}. [create_exam_papers]"
            )
        if (
            form_url_template is not None
            and "{student_id}" not in form_url_template
        ):
            raise ValueError(
                "form_url_template must contain "
                "'{student_id}' placeholder. "
                "[create_exam_papers]"
            )

    # ── main entry point ─────────────────────────

    def create_exam_papers(
        self,
        questions: List[Dict[str, str]],
        num_papers: int,
        output_path: str,
        year: int = 2025,
        grade: int = 1,
        semester: int = 2,
        course_name: str = "감염미생물학",
        week_num: int = 3,
        form_url_template: Optional[str] = None,
        student_ids: Optional[List[str]] = None,
    ) -> None:
        """Generate exam paper PDF.

        Args:
            questions: list of dicts with keys
                'topic', 'text', 'limit'.
            num_papers: number of copies.
            output_path: PDF file path.
            form_url_template: Google Forms URL with
                {student_id}, {course_name}, {week_num}
                placeholders. None disables QR.
            student_ids: custom student IDs or None
                for auto-generation (S001, S002, ...).

        Raises:
            ValueError: on invalid input.
        """
        self._validate_inputs(num_papers, form_url_template)
        self._validate_questions(questions)
        ids = self._generate_student_ids(
            num_papers, student_ids,
        )
        self._render_pdf(
            questions, ids, output_path,
            year, grade, semester,
            course_name, week_num, form_url_template,
        )

    def _render_pdf(
        self,
        questions: List[Dict[str, str]],
        ids: List[str],
        output_path: str,
        year: int,
        grade: int,
        semester: int,
        course_name: str,
        week_num: int,
        form_url_template: Optional[str],
    ) -> None:
        """Render all pages to a PDF file."""
        c = canvas.Canvas(output_path, pagesize=A4)
        for i, sid in enumerate(ids):
            self._draw_page(
                c, questions, i + 1,
                year, grade, semester,
                course_name, week_num,
                student_id=sid if form_url_template else None,
                form_url_template=form_url_template,
            )
            c.showPage()
        c.save()
        print(
            f"✓ 시험지 {len(ids)}장 생성 완료: "
            f"{output_path}"
        )
