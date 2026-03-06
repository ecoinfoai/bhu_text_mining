"""Shared font discovery utilities for PDF generation.

Extracted from ``exam_generator.py`` for reuse by report_generator.
"""

from __future__ import annotations

import glob
import os
import platform
from typing import List


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
