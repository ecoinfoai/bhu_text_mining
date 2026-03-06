import pytesseract
from PIL import Image
import pandas as pd
import re
import os
import cv2
import numpy as np
from typing import Tuple

class TesseractProcessor:
    """Tesseract OCR 처리기 - 학번/이름/일련번호 인식"""
    
    def __init__(self):
        # 각 영역별 최적화된 설정
        self.serial_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
        self.id_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
        self.name_config = '--psm 7 --oem 3'
    
    def preprocess_for_ocr(self, image: Image.Image, enhance: bool = True) -> Image.Image:
        """OCR 정확도 향상을 위한 전처리"""
        img_array = np.array(image)
        
        # 그레이스케일 변환
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        if enhance:
            # 대비 향상 (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        
        # 이진화 (Otsu)
        _, binary = cv2.threshold(gray, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        
        # 모폴로지 연산 (선명하게)
        kernel = np.ones((2, 2), np.uint8)
        morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return Image.fromarray(morphed)
    
    def extract_serial_number(
        self, 
        image: Image.Image, 
        box: Tuple[int, int, int, int]
    ) -> str:
        """
        일련번호 추출 (4자리 숫자)
        
        Args:
            image: 원본 이미지
            box: 일련번호 영역 좌표 (left, upper, right, lower)
        
        Returns:
            "0001", "0002", ... 형식의 4자리 숫자
        """
        cropped = image.crop(box)
        processed = self.preprocess_for_ocr(cropped, enhance=True)
        
        # OCR 실행
        text = pytesseract.image_to_string(
            processed,
            lang='kor',
            config=self.serial_config
        ).strip()
        
        # 숫자만 추출
        digits = re.sub(r'\D', '', text)
        
        # 4자리로 정규화
        if len(digits) == 4:
            return digits
        elif len(digits) > 4:
            return digits[:4]
        elif len(digits) > 0:
            return digits.zfill(4)
        else:
            return "ERROR_SN"
    
    def extract_student_id(
        self, 
        image: Image.Image, 
        box: Tuple[int, int, int, int]
    ) -> str:
        """
        학번 추출 (10자리)
        
        Returns:
            "2025194072" 형식
        """
        cropped = image.crop(box)
        processed = self.preprocess_for_ocr(cropped)
        
        text = pytesseract.image_to_string(
            processed,
            lang='eng',
            config=self.id_config
        ).strip()
        
        # 숫자만 추출
        digits = re.sub(r'\D', '', text)
        
        # 10자리 검증
        if len(digits) == 10:
            return digits
        else:
            return f"ERROR_ID_{digits}"
    
    def extract_name(
        self, 
        image: Image.Image, 
        box: Tuple[int, int, int, int]
    ) -> str:
        """
        이름 추출 (한글 2-4자)
        
        Returns:
            "홍길동" 형식
        """
        cropped = image.crop(box)
        processed = self.preprocess_for_ocr(cropped)
        
        text = pytesseract.image_to_string(
            processed,
            lang='kor',
            config=self.name_config
        ).strip()
        
        # 공백 제거
        name = re.sub(r'\s+', '', text)
        
        # 한글만 추출
        korean_only = re.sub(r'[^가-힣]', '', name)
        
        if 2 <= len(korean_only) <= 4:
            return korean_only
        else:
            return f"ERROR_NAME_{text}"
    
    def process_single_image(
        self,
        image_path: str,
        serial_box: Tuple[int, int, int, int],
        id_box: Tuple[int, int, int, int],
        name_box: Tuple[int, int, int, int]
    ) -> dict:
        """단일 이미지 처리"""
        image = Image.open(image_path)
        
        return {
            'filename': os.path.basename(image_path),
            'serialNum': self.extract_serial_number(image, serial_box),
            'ID': self.extract_student_id(image, id_box),
            'names': self.extract_name(image, name_box)
        }
    
    def batch_process(
        self,
        image_dir: str,
        serial_box: Tuple[int, int, int, int],
        id_box: Tuple[int, int, int, int],
        name_box: Tuple[int, int, int, int],
        output_csv: str = None
    ) -> pd.DataFrame:
        """전체 이미지 일괄 처리"""
        results = []
        
        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and not f.startswith('answer_cropped')
        ])
        
        print(f"총 {len(image_files)}장 처리 시작...\n")
        
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(image_dir, filename)
            
            try:
                info = self.process_single_image(
                    image_path, serial_box, id_box, name_box
                )
                results.append(info)
                
                # 에러 체크
                errors = [k for k, v in info.items() 
                         if isinstance(v, str) and 'ERROR' in v]
                
                if errors:
                    print(f"⚠ [{i:3d}/{len(image_files)}] {filename}")
                    print(f"   일련번호: {info['serialNum']}, "
                          f"학번: {info['ID']}, 이름: {info['names']}")
                    print(f"   에러 필드: {errors}")
                else:
                    print(f"✓ [{i:3d}/{len(image_files)}] {filename} → "
                          f"{info['serialNum']} | {info['ID']} | {info['names']}")
                
            except Exception as e:
                print(f"✗ [{i:3d}/{len(image_files)}] {filename}: 예외 발생 - {e}")
                results.append({
                    'filename': filename,
                    'serialNum': 'ERROR_EXCEPTION',
                    'ID': 'ERROR_EXCEPTION',
                    'names': 'ERROR_EXCEPTION'
                })
        
        df = pd.DataFrame(results)
        
        # 통계
        total = len(df)
        errors = df.apply(lambda row: any('ERROR' in str(v) for v in row), axis=1).sum()
        success_rate = (total - errors) / total * 100 if total > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"처리 완료: {total}장 | 성공: {total-errors}장 | 에러: {errors}장")
        print(f"성공률: {success_rate:.1f}%")
        print(f"{'='*60}\n")
        
        if output_csv:
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"→ 매핑 테이블 저장: {output_csv}\n")
        
        return df
