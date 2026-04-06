"""Shared font discovery and text utilities for PDF generation.

Extracted from ``exam_generator.py`` for reuse by report generators.
Provides font discovery, font registration, and XML-safe text escaping.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import xml.sax.saxutils
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# C0 control characters illegal in XML (keep \t=0x09, \n=0x0A, \r=0x0D)
# plus zero-width Unicode characters that can cause rendering issues in PDFs
_XML_ILLEGAL_CTRL = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    r"\u200b\u200c\u200d\u200e\u200f\ufeff]"
)


def find_korean_font() -> str:
    """Find a Korean font (NanumGothic or equivalent) on the system.

    Searches OS-specific paths including NixOS glob patterns.

    Returns:
        Absolute path to a Korean .ttf font file.

    Raises:
        FileNotFoundError: If no Korean font is found.
    """
    system = platform.system()
    search_paths: list[str] = []

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
    raise FileNotFoundError("Korean font not found. Install NanumGothic or specify font_path.")


def strip_invisible(text: str) -> str:
    """Strip invisible characters from text.

    Removes C0 control characters (except tab, newline, carriage return)
    and zero-width Unicode characters (U+200B..U+200F, U+FEFF).

    This is the shared stripping logic used by both :func:`esc` (for XML)
    and ``delivery_prepare.sanitize_filename`` (for filenames).
    """
    return _XML_ILLEGAL_CTRL.sub("", str(text))


def esc(text: str) -> str:
    """Escape text for use in ReportLab XML/Paragraph markup.

    Strips C0 control characters (except tab, newline, carriage return)
    and zero-width Unicode characters (U+200B..U+200F, U+FEFF) before
    applying XML entity escaping.
    """
    return xml.sax.saxutils.escape(strip_invisible(text))


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
