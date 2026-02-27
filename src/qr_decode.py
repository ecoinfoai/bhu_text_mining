# src/qr_decode.py
"""QR code decoding utilities for scanned exam sheets.

Uses OpenCV QRCodeDetector (already available via opencv-python).
No additional dependencies required.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


def decode_qr_from_image(image_path: str) -> Optional[str]:
    """Decode QR code from an image file.

    Attempts decoding on the original image first, then falls
    back to grayscale + Otsu binarization for low-contrast scans.

    Args:
        image_path: path to the image file (.jpg/.png).

    Returns:
        Decoded QR content string, or None if not found.
    """
    if not _HAS_CV2:
        raise ImportError(
            "opencv-python is required. Install: "
            "pip install opencv-python-headless. "
            "[decode_qr_from_image]"
        )
    img = cv2.imread(image_path)
    if img is None:
        return None

    detector = cv2.QRCodeDetector()

    # First attempt: original image
    data, _, _ = detector.detectAndDecode(img)
    if data:
        return data

    # Fallback: grayscale + Otsu binarization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
