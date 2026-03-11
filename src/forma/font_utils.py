"""Shared font discovery and text utilities for PDF generation.

Extracted from ``exam_generator.py`` for reuse by report generators.
Provides font discovery, font registration, and XML-safe text escaping.
"""

from __future__ import annotations

import glob
import os
import platform
import xml.sax.saxutils
from typing import List

from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def find_korean_font() -> str:
    """Find a Korean font (NanumGothic or equivalent) on the system.

    Searches OS-specific paths including NixOS glob patterns.

    Returns:
        Absolute path to a Korean .ttf font file.

    Raises:
        FileNotFoundError: If no Korean font is found.
    """
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
    raise FileNotFoundError(
        "Korean font not found. Install NanumGothic or specify font_path."
    )


def esc(text: str) -> str:
    """Escape text for use in ReportLab XML/Paragraph markup."""
    return xml.sax.saxutils.escape(str(text))


def register_korean_fonts(font_path: str) -> None:
    """Register NanumGothic regular and bold fonts with ReportLab.

    Registers ``NanumGothic`` and ``NanumGothicBold`` font faces and
    sets up font family mappings so ReportLab's paragraph parser can
    resolve bold variants.

    If the bold variant file (``NanumGothicBold.ttf``) is not found,
    the regular font is used as a fallback for the bold face.

    Args:
        font_path: Path to the NanumGothic regular ``.ttf`` file.
    """
    pdfmetrics.registerFont(TTFont("NanumGothic", font_path))
    bold_path = font_path.replace(".ttf", "Bold.ttf")
    if os.path.exists(bold_path):
        pdfmetrics.registerFont(TTFont("NanumGothicBold", bold_path))
    else:
        pdfmetrics.registerFont(TTFont("NanumGothicBold", font_path))

    addMapping("NanumGothic", 0, 0, "NanumGothic")
    addMapping("NanumGothic", 1, 0, "NanumGothicBold")
    addMapping("NanumGothic", 0, 1, "NanumGothic")
    addMapping("NanumGothic", 1, 1, "NanumGothicBold")
