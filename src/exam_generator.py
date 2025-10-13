from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from typing import List, Dict

class ExamPDFGenerator:
    """2025학년도 형성평가 시험지 생성기"""
    
    def __init__(self, font_path: str = "/usr/share/fonts/nanum/NanumGothic.ttf"):
        # register your Korean font
        pdfmetrics.registerFont(TTFont('NanumGothic', font_path))
        bold_path = font_path.replace('.ttf', 'Bold.ttf')
        if os.path.exists(bold_path):
            pdfmetrics.registerFont(TTFont('NanumGothicBold', bold_path))
        
        self.page_width, self.page_height = A4
        self.margin = 15 * mm
    
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
        시험지 PDF 일괄 생성
        
        Args:
            questions: [{'topic': '개념이해', 'text': '문항내용', 'limit': '100자 내외'}, ...]
            num_papers: 생성할 매수
            output_path: 저장 경로
            year: 연도 (기본값: 2025)
            grade: 학년 (기본값: 1)
            semester: 학기 (기본값: 2)
            course_name: 과목명
            week_num: 주차
        """
        c = canvas.Canvas(output_path, pagesize=A4)
        
        for paper_num in range(1, num_papers + 1):
            self._draw_single_page(
                c, questions, paper_num, year, grade, semester, course_name, week_num
            )
            c.showPage()
        
        c.save()
        print(f"✓ 시험지 {num_papers}장 생성 완료: {output_path}")
    
    def _draw_single_page(
        self, c, questions, paper_num, year, grade, semester, course_name, week_num
    ):
        """단일 시험지 페이지 그리기"""
        
        # ============ 테두리 ============
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(1.5)
        c.rect(self.margin, self.margin, 
               self.page_width - 2*self.margin, 
               self.page_height - 2*self.margin)
        
        # ============ 상단 헤더 ============
        y_pos = self.page_height - self.margin - 15*mm
        
        # 제목 박스
        header_height = 25*mm
        c.rect(self.margin, y_pos - header_height,
               self.page_width - 2*self.margin, header_height)
        
        # 연도/학년/학기/과목명
        c.setFont('NanumGothic', 11)
        title_line1 = f"{year}학년도 {grade}학년 {semester}학기 {course_name}"
        c.drawCentredString(self.page_width / 2, y_pos - 8*mm, title_line1)
        
        # "N주차 형성평가" (크게)
        c.setFont('NanumGothicBold', 18)
        title_line2 = f"{week_num}주차 형성평가"
        c.drawCentredString(self.page_width / 2, y_pos - 18*mm, title_line2)
        
        y_pos -= header_height + 5*mm
        
        # ============ 학번/반/학번/이름 입력란 ============
        info_height = 12*mm
        
        # 외곽 박스
        c.rect(self.margin, y_pos - info_height,
               self.page_width - 2*self.margin, info_height)
        
        # 4칸으로 분할
        section_width = (self.page_width - 2*self.margin) / 4
        
        for i in range(1, 4):
            x = self.margin + i * section_width
            c.line(x, y_pos - info_height, x, y_pos)
        
        # 라벨 및 입력란
        c.setFont('NanumGothic', 10)
        labels = ['분반', '반', '학번', '이름']
        
        for i, label in enumerate(labels):
            x_center = self.margin + (i + 0.5) * section_width
            
            # 라벨
            c.drawCentredString(x_center, y_pos - 6*mm, label)
            
            # 입력 박스 (학번/이름만)
            if i >= 2:  # 학번, 이름
                box_width = section_width * 0.8
                box_x = self.margin + i * section_width + section_width * 0.1
                c.rect(box_x, y_pos - info_height + 1*mm,
                       box_width, 5*mm)
        
        y_pos -= info_height + 5*mm
        
        # ============ 문항들 ============
        for q_num, question in enumerate(questions, 1):
            # 문항 헤더 박스
            question_header_height = 15*mm
            
            # 좌측: 문항 주제
            topic_width = 30*mm
            c.rect(self.margin, y_pos - question_header_height,
                   topic_width, question_header_height)
            
            c.setFont('NanumGothicBold', 10)
            c.drawCentredString(self.margin + topic_width/2, 
                               y_pos - question_header_height/2 + 2*mm,
                               question['topic'])
            
            # 우측: 문항 텍스트
            question_text_width = self.page_width - 2*self.margin - topic_width
            c.rect(self.margin + topic_width, y_pos - question_header_height,
                   question_text_width, question_header_height)
            
            # 문항 텍스트 (줄바꿈 처리)
            c.setFont('NanumGothic', 9)
            full_text = f"문제 {q_num}. {question['text']} ({question['limit']})"
            
            lines = self._wrap_text(c, full_text, question_text_width - 4*mm, 
                                   'NanumGothic', 9)
            
            text_y = y_pos - 5*mm
            for line in lines:
                c.drawString(self.margin + topic_width + 2*mm, text_y, line)
                text_y -= 4*mm
            
            y_pos -= question_header_height
            
            # 답안 작성 영역
            answer_height = 60*mm
            
            # 외곽 박스
            c.setFillColorRGB(0.95, 0.95, 0.95)  # 연한 회색 배경
            c.rect(self.margin, y_pos - answer_height,
                   self.page_width - 2*self.margin, answer_height, 
                   fill=1, stroke=1)
            
            # 답안 작성 안내선
            c.setStrokeColorRGB(0.85, 0.85, 0.85)
            c.setLineWidth(0.3)
            for i in range(1, int(answer_height / (5*mm))):
                line_y = y_pos - i * 5*mm
                c.line(self.margin + 2*mm, line_y,
                       self.page_width - self.margin - 2*mm, line_y)
            
            c.setStrokeColorRGB(0, 0, 0)
            c.setLineWidth(1)
            c.setFillColorRGB(1, 1, 1)
            
            y_pos -= answer_height + 3*mm
        
        # ============ 일련번호 (우측 상단, 크고 명확하게) ============
        serial_num = f"{paper_num:04d}"  # 4자리
        
        # 일련번호 박스 위치
        serial_x = self.page_width - self.margin - 45*mm
        serial_y = self.page_height - self.margin - 45*mm
        
        # 박스
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(2)
        c.rect(serial_x, serial_y, 40*mm, 15*mm)
        
        # 숫자 (매우 크고 굵게)
        c.setFont('NanumGothicBold', 28)
        c.drawCentredString(serial_x + 20*mm, serial_y + 4*mm, serial_num)
        
        # "일련번호" 라벨
        c.setFont('NanumGothic', 8)
        c.drawCentredString(serial_x + 20*mm, serial_y + 12*mm, "일련번호")
    
    def _wrap_text(self, c, text, max_width, font_name, font_size):
        """텍스트 자동 줄바꿈"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + word + " "
            if c.stringWidth(test_line, font_name, font_size) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        return lines
