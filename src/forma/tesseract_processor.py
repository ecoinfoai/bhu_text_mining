"""Tesseract OCR processor for student ID, name, and serial number extraction.

Uses pytesseract with preprocessing (CLAHE, Otsu binarization, denoising)
to extract structured fields from scanned exam answer sheets.
"""

from __future__ import annotations

import pytesseract
from PIL import Image
import pandas as pd
import re
import os
import cv2
import numpy as np

class TesseractProcessor:
    """Tesseract OCR processor for student ID, name, and serial number recognition."""

    def __init__(self) -> None:
        # Optimized settings per field type
        self.serial_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
        self.id_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
        self.name_config = '--psm 7 --oem 3'

    def preprocess_for_ocr(self, image: Image.Image, enhance: bool = True) -> Image.Image:
        """Preprocess image to improve OCR accuracy.

        Applies grayscale conversion, CLAHE contrast enhancement,
        Otsu binarization, denoising, and morphological closing.

        Args:
            image: Input PIL image.
            enhance: If True, apply CLAHE contrast enhancement.

        Returns:
            Preprocessed PIL image.
        """
        img_array = np.array(image)

        # Grayscale conversion
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        if enhance:
            # Contrast enhancement (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

        # Binarization (Otsu)
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Noise removal
        denoised = cv2.fastNlMeansDenoising(binary, h=10)

        # Morphological closing (sharpen)
        kernel = np.ones((2, 2), np.uint8)
        morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

        return Image.fromarray(morphed)

    def extract_serial_number(
        self,
        image: Image.Image,
        box: tuple[int, int, int, int],
    ) -> str:
        """Extract serial number (4-digit) from the specified image region.

        Args:
            image: Source PIL image.
            box: Crop region coordinates (left, upper, right, lower).

        Returns:
            Zero-padded 4-digit string (e.g. "0001"), or "ERROR_SN" on failure.
        """
        cropped = image.crop(box)
        processed = self.preprocess_for_ocr(cropped, enhance=True)

        text = pytesseract.image_to_string(
            processed,
            lang='kor',
            config=self.serial_config
        ).strip()

        # Extract digits only
        digits = re.sub(r'\D', '', text)

        # Normalize to 4 digits
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
        box: tuple[int, int, int, int],
    ) -> str:
        """Extract student ID (10-digit) from the specified image region.

        Args:
            image: Source PIL image.
            box: Crop region coordinates (left, upper, right, lower).

        Returns:
            10-digit student ID string, or "ERROR_ID_<digits>" on failure.
        """
        cropped = image.crop(box)
        processed = self.preprocess_for_ocr(cropped)

        text = pytesseract.image_to_string(
            processed,
            lang='eng',
            config=self.id_config
        ).strip()

        # Extract digits only
        digits = re.sub(r'\D', '', text)

        # Validate 10 digits
        if len(digits) == 10:
            return digits
        else:
            return f"ERROR_ID_{digits}"

    def extract_name(
        self,
        image: Image.Image,
        box: tuple[int, int, int, int],
    ) -> str:
        """Extract Korean name (2-4 characters) from the specified image region.

        Args:
            image: Source PIL image.
            box: Crop region coordinates (left, upper, right, lower).

        Returns:
            Korean name string, or "ERROR_NAME_<raw>" on failure.
        """
        cropped = image.crop(box)
        processed = self.preprocess_for_ocr(cropped)

        text = pytesseract.image_to_string(
            processed,
            lang='kor',
            config=self.name_config
        ).strip()

        # Remove whitespace
        name = re.sub(r'\s+', '', text)

        # Extract Korean characters only
        korean_only = re.sub(r'[^가-힣]', '', name)

        if 2 <= len(korean_only) <= 4:
            return korean_only
        else:
            return f"ERROR_NAME_{text}"

    def process_single_image(
        self,
        image_path: str,
        serial_box: tuple[int, int, int, int],
        id_box: tuple[int, int, int, int],
        name_box: tuple[int, int, int, int],
    ) -> dict[str, str]:
        """Process a single scanned image to extract serial number, ID, and name.

        Args:
            image_path: Path to the scanned image file.
            serial_box: Crop region for the serial number.
            id_box: Crop region for the student ID.
            name_box: Crop region for the student name.

        Returns:
            Dict with keys: filename, serialNum, ID, names.
        """
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
        serial_box: tuple[int, int, int, int],
        id_box: tuple[int, int, int, int],
        name_box: tuple[int, int, int, int],
        output_csv: str | None = None,
    ) -> pd.DataFrame:
        """Batch-process all images in a directory.

        Args:
            image_dir: Directory containing scanned images.
            serial_box: Crop region for the serial number.
            id_box: Crop region for the student ID.
            name_box: Crop region for the student name.
            output_csv: Optional CSV output path for the mapping table.

        Returns:
            DataFrame with columns: filename, serialNum, ID, names.
        """
        results: list[dict[str, str]] = []

        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and not f.startswith('answer_cropped')
        ])

        print(f"Processing {len(image_files)} images...\n")

        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(image_dir, filename)

            try:
                info = self.process_single_image(
                    image_path, serial_box, id_box, name_box
                )
                results.append(info)

                # Error check
                errors = [k for k, v in info.items()
                         if isinstance(v, str) and 'ERROR' in v]

                if errors:
                    print(f"WARNING [{i:3d}/{len(image_files)}] {filename}")
                    print(f"   serial: {info['serialNum']}, "
                          f"ID: {info['ID']}, name: {info['names']}")
                    print(f"   error fields: {errors}")
                else:
                    print(f"OK [{i:3d}/{len(image_files)}] {filename} -> "
                          f"{info['serialNum']} | {info['ID']} | {info['names']}")

            except Exception as e:
                print(f"FAIL [{i:3d}/{len(image_files)}] {filename}: exception - {e}")
                results.append({
                    'filename': filename,
                    'serialNum': 'ERROR_EXCEPTION',
                    'ID': 'ERROR_EXCEPTION',
                    'names': 'ERROR_EXCEPTION'
                })

        df = pd.DataFrame(results)

        # Statistics
        total = len(df)
        errors = df.apply(lambda row: any('ERROR' in str(v) for v in row), axis=1).sum()
        success_rate = (total - errors) / total * 100 if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"Complete: {total} images | success: {total-errors} | errors: {errors}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"{'='*60}\n")

        if output_csv:
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"Mapping table saved: {output_csv}\n")

        return df
