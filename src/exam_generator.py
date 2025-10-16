# src/exam_generator.py
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import black, grey
import os
import platform
from typing import List, Dict

class ExamPDFGenerator:
    """텍스트 기반 시험지 PDF 생성기"""
    
    def __init__(self, font_path: str = None):
        # 폰트 경로 자동 탐색 또는 사용자 지정
        if font_path is None:
            font_path = self._find_font()
        
        if not os.path.exists(font_path):
            raise FileNotFoundError(
                f"폰트를 찾을 수 없습니다: {font_path}\n"
                f"font_path 인자로 직접 경로를 지정하거나\n"
                f"나눔고딕 폰트를 설치하세요."
            )
        
        # 폰트 등록
        pdfmetrics.registerFont(TTFont('NanumGothic', font_path))
        
        # Bold 폰트 (선택사항)
        bold_path = font_path.replace('.ttf', 'Bold.ttf')
        if os.path.exists(bold_path):
            pdfmetrics.registerFont(TTFont('NanumGothicBold', bold_path))
        else:
            pdfmetrics.registerFont(TTFont('NanumGothicBold', font_path))
        
        self.page_width, self.page_height = A4
        self.margin = 20 * mm
        self.line_height = 5 * mm
    
    def _find_font(self) -> str:
        """OS별 폰트 자동 탐색"""
        system = platform.system()
        
        search_paths = []
        if system == "Windows":
            search_paths = [
                "C:/Windows/Fonts/malgun.ttf",  # 맑은 고딕
                "C:/Windows/Fonts/NanumGothic.ttf",
            ]
        elif system == "Darwin":  # macOS
            search_paths = [
                "/Library/Fonts/NanumGothic.ttf",
                "/System/Library/Fonts/AppleGothic.ttf",
            ]
        else:  # Linux
            search_paths = [
                "/usr/share/fonts/nanum/NanumGothic.ttf",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("폰트를 찾을 수 없습니다.")
    
    def create_exam_papers(
        self,
        questions: List[Dict[str, str]],
        num_papers: int,
        output_path: str,
        year: int = 2025,
        grade: int = 1,
        semester: int = 2,
        course_name: str = "감염미생물학",
        week_num: int = 3
    ) -> None:
        """
        시험지 PDF 생성
        
        Args:
            questions: [
                {
                    'topic': '개념이해',
                    'text': '문항 내용',
                    'limit': '100자 내외'
                },
                ...
            ]
            num_papers: 생성할 시험지 매수
            output_path: 저장 경로
        """
        c = canvas.Canvas(output_path, pagesize=A4)
        
        for paper_num in range(1, num_papers + 1):
            self._draw_page(
                c, questions, paper_num, 
                year, grade, semester, course_name, week_num
            )
            c.showPage()
        
        c.save()
        print(f"✓ 시험지 {num_papers}장 생성 완료: {output_path}")
    
    def _draw_page(
        self, c, questions, paper_num, 
        year, grade, semester, course_name, week_num
    ):
        """단일 페이지 그리기 (텍스트만)"""
        
        y_pos = self.page_height - self.margin
        
        # ========== 일련번호 (우측 상단, 매우 크게) ==========
        serial_num = f"{paper_num:04d}"
        
        c.setFont('NanumGothicBold', 36)  # 크게!
        c.drawRightString(
            self.page_width - self.margin,
            y_pos - 8*mm,
            serial_num
        )
        
        # 일련번호 라벨
        c.setFont('NanumGothic', 9)
        c.setFillGray(0.5)
        c.drawRightString(
            self.page_width - self.margin,
            y_pos - 14*mm,
            "일련번호"
        )
        c.setFillGray(0)
        
        # ========== 제목 영역 ==========
        y_pos -= 10*mm
        
        # 과목 및 주차 정보
        c.setFont('NanumGothic', 10)
        title_line1 = f"{year}학년도 {grade}학년 {semester}학기 {course_name}"
        c.drawCentredString(self.page_width / 2, y_pos, title_line1)
        
        y_pos -= 7*mm
        c.setFont('NanumGothicBold', 16)
        title_line2 = f"{week_num}주차 형성평가"
        c.drawCentredString(self.page_width / 2, y_pos, title_line2)
        
        # ========== 구분선 ==========
        y_pos -= 8*mm
        c.setLineWidth(1)
        c.line(self.margin, y_pos, self.page_width - self.margin, y_pos)
        
        # ========== 학생 정보 입력란 (수정됨) ==========
        y_pos -= 10*mm
        
        c.setFont('NanumGothic', 11)
        
        # 첫 번째 줄: 학년, 분반
        # 학년
        c.drawString(self.margin, y_pos, "학년:")
        c.rect(self.margin + 15*mm, y_pos - 3*mm, 25*mm, 6*mm)
        
        # 분반
        c.drawString(self.margin + 50*mm, y_pos, "분반:")
        c.rect(self.margin + 65*mm, y_pos - 3*mm, 25*mm, 6*mm)
        
        y_pos -= 10*mm
        
        # 두 번째 줄: 학번
        c.drawString(self.margin, y_pos, "학번:")
        c.rect(self.margin + 15*mm, y_pos - 3*mm, 70*mm, 6*mm)
        
        y_pos -= 10*mm
        
        # 세 번째 줄: 이름
        c.drawString(self.margin, y_pos, "이름:")
        c.rect(self.margin + 15*mm, y_pos - 3*mm, 40*mm, 6*mm)
        
        # ========== 구분선 ==========
        y_pos -= 8*mm
        c.setLineWidth(0.5)
        c.line(self.margin, y_pos, self.page_width - self.margin, y_pos)
        
        # ========== 문항들 ==========
        y_pos -= 10*mm
        
        for q_num, question in enumerate(questions, 1):
            # 문항 헤더 (카테고리)
            c.setFont('NanumGothicBold', 10)
            c.setFillGray(0.3)
            header_text = f"[{question['topic']}]"
            c.drawString(self.margin, y_pos, header_text)
            c.setFillGray(0)
            
            y_pos -= 6*mm
            
            # 문항 번호 + 텍스트 (같은 줄에 작성)
            c.setFont('NanumGothicBold', 10)
            question_num_text = f"문제 {q_num}. "
            
            # 문제 번호의 너비 계산
            num_width = c.stringWidth(question_num_text, 'NanumGothicBold', 10)
            
            # 문제 번호 그리기
            c.drawString(self.margin, y_pos, question_num_text)
            
            # 문항 텍스트 (줄바꿈 처리)
            c.setFont('NanumGothic', 10)
            full_question = f"{question['text']} ({question['limit']})"
            
            # 첫 줄은 문제 번호 뒤에서 시작
            first_line_x = self.margin + num_width
            first_line_max_width = self.page_width - self.margin - first_line_x
            
            # 나머지 줄은 margin + 5mm에서 시작
            other_line_max_width = self.page_width - 2 * self.margin - 5*mm
            
            # 텍스트를 단어 단위로 분리
            words = full_question.split()
            lines = []
            current_line = ""
            is_first_line = True
            
            for word in words:
                test_line = current_line + word + " "
                
                # 첫 줄과 나머지 줄의 최대 너비가 다름
                max_width = first_line_max_width if is_first_line else other_line_max_width
                width = c.stringWidth(test_line, 'NanumGothic', 10)
                
                if width <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append((current_line.strip(), is_first_line))
                        is_first_line = False
                    current_line = word + " "
            
            if current_line:
                lines.append((current_line.strip(), is_first_line))
            
            # 첫 줄 그리기
            if lines:
                first_text, is_first = lines[0]
                c.drawString(first_line_x, y_pos, first_text)
                y_pos -= 5*mm
            
            # 나머지 줄 그리기
            for line_text, _ in lines[1:]:
                c.drawString(self.margin + 5*mm, y_pos, line_text)
                y_pos -= 5*mm
            
            # 답안 작성 공간 (높이 절반으로 줄임: 55mm → 27mm)
            y_pos -= 2*mm
            
            # 답안 영역 표시 (연한 회색 테두리)
            answer_height = 32*mm  # 절반 정도로 줄임
            c.setStrokeGray(0.7)
            c.setLineWidth(0.5)
            c.rect(
                self.margin, 
                y_pos - answer_height,
                self.page_width - 2 * self.margin, 
                answer_height
            )
            
            # 답안 작성 안내선 (가이드라인)
            c.setStrokeGray(0.85)
            c.setLineWidth(0.3)
            num_lines = int(answer_height / self.line_height)
            
            for i in range(1, num_lines):
                line_y = y_pos - i * self.line_height
                c.line(
                    self.margin + 2*mm, 
                    line_y,
                    self.page_width - self.margin - 2*mm, 
                    line_y
                )
            
            # 다음 문항으로
            y_pos -= answer_height + 5*mm
            
            # 리셋
            c.setStrokeGray(0)
        
        # ========== 하단 안내 ==========
        c.setFillGray(0.5)
        c.setFont('NanumGothic', 8)
        footer_text = f"※ 답안은 검은색 볼펜으로 작성하세요."
        c.drawCentredString(
            self.page_width / 2, 
            self.margin - 5*mm, 
            footer_text
        )
