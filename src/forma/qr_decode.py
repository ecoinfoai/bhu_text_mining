# src/qr_decode.py
"""QR code decoding utilities for scanned exam sheets.

Uses pyzbar (zbar) as primary decoder for robust scanning,
with OpenCV QRCodeDetector as fallback.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

try:
    from pyzbar.pyzbar import decode as _pyzbar_decode
    _HAS_PYZBAR = True
except ImportError:
    _HAS_PYZBAR = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


def decode_qr_from_image(image_path: str) -> Optional[str]:
    """Decode QR code from an image file.

    Tries pyzbar first (more reliable for scanned images),
    then falls back to OpenCV QRCodeDetector with preprocessing.

    Args:
        image_path: path to the image file (.jpg/.png).

    Returns:
        Decoded QR content string, or None if not found.

    Raises:
        ImportError: if neither pyzbar nor opencv is available.
    """
    if not _HAS_PYZBAR and not _HAS_CV2:
        raise ImportError(
            "QR decoding requires pyzbar or opencv-python-headless. "
            "Install: pip install pyzbar"
        )

    from PIL import Image

    img = Image.open(image_path)

    if _HAS_PYZBAR:
        result = _decode_pyzbar_with_preprocess(img)
        if result:
            return result

    if _HAS_CV2:
        result = _decode_cv2_with_preprocess(image_path)
        if result:
            return result

    return None


def _decode_pyzbar_with_preprocess(img: "Image.Image") -> Optional[str]:  # noqa: F821
    """Try pyzbar with progressively stronger preprocessing."""
    from PIL import Image, ImageFilter

    # 1) Original
    results = _pyzbar_decode(img)
    if results:
        return results[0].data.decode("utf-8")

    gray = img.convert("L")

    # 2) Grayscale
    results = _pyzbar_decode(gray)
    if results:
        return results[0].data.decode("utf-8")

    # 3) Sharpen + contrast
    sharpened = gray.filter(ImageFilter.SHARPEN)
    results = _pyzbar_decode(sharpened)
    if results:
        return results[0].data.decode("utf-8")

    # 4) Binary threshold (scan noise removal)
    threshold = 128
    binary = gray.point(lambda p: 255 if p > threshold else 0)
    results = _pyzbar_decode(binary)
    if results:
        return results[0].data.decode("utf-8")

    # 5) Upscale 2x (helps with small QR codes)
    w, h = gray.size
    upscaled = gray.resize((w * 2, h * 2), Image.LANCZOS)
    results = _pyzbar_decode(upscaled)
    if results:
        return results[0].data.decode("utf-8")

    return None


def _decode_cv2_with_preprocess(image_path: str) -> Optional[str]:
    """Try OpenCV QRCodeDetector with preprocessing."""
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return None

    detector = cv2.QRCodeDetector()

    data, _, _ = detector.detectAndDecode(img_cv)
    if data:
        return data

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    data, _, _ = detector.detectAndDecode(binary)
    if data:
        return data

    return None


def parse_qr_content(raw: str) -> dict:
    """Parse QR content string into a structured dict.

    Handles two formats:

    - Plain: ``"S001|과목|1주차|Q2"``
      → ``{"student_id": "S001", "q_num": 2}``
    - URL: ``"...entry.1064397072=S001&q=2"``
      → ``{"student_id": "S001", "q_num": 2}``

    Args:
        raw: decoded QR content string.

    Returns:
        dict with keys:
            ``"student_id"`` (str) and ``"q_num"`` (int or None).

    Raises:
        ValueError: if the format is not recognized or student_id
            cannot be extracted.
    """
    # ── URL format ────────────────────────────────
    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https"):
        params = parse_qs(parsed.query)

        student_id: Optional[str] = None
        q_num: Optional[int] = None

        # Prefer explicit 'student_id' or 'sid' key
        for key in ("student_id", "sid"):
            if key in params:
                student_id = unquote(params[key][0])
                break

        # Fall back to first entry.* parameter
        if student_id is None:
            for key in sorted(params.keys()):
                if key.startswith("entry."):
                    student_id = unquote(params[key][0])
                    break

        if student_id is None:
            raise ValueError(
                f"Cannot extract student_id from URL: {raw!r}. "
                "Expected 'student_id=', 'sid=', or 'entry.*=' "
                "parameter. [parse_qr_content]"
            )

        # Extract q_num from 'q' parameter
        if "q" in params:
            try:
                q_num = int(params["q"][0])
            except (ValueError, IndexError):
                pass

        return {"student_id": student_id, "q_num": q_num}

    # ── Plain text format: "S001|course|주차" or "S001|course|주차|Q2" ──
    parts = raw.split("|")
    if len(parts) < 2:
        raise ValueError(
            f"Unrecognized QR content format: {raw!r}. "
            "Expected 'student_id|...' pipe-delimited or a URL. "
            "[parse_qr_content]"
        )

    student_id = parts[0]
    q_num = None

    # Look for Q-num tag in the last segment
    if len(parts) >= 4:
        match = re.match(r"Q(\d+)$", parts[-1].strip(), re.IGNORECASE)
        if match:
            q_num = int(match.group(1))

    return {"student_id": student_id, "q_num": q_num}
