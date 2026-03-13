"""Shared report utilities.

Provides small helpers reused across multiple PDF report generators:

- ``minimal_png_bytes()`` — 1×1 RGB PNG for placeholder images.
- ``sanitize_filename_report()`` — clean filenames for report output.
"""

from __future__ import annotations

import re
import struct
import zlib


def minimal_png_bytes() -> bytes:
    """Return a 1x1 RGB PNG as bytes -- used as a safe fallback for Image()."""

    def _chunk(type_: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + type_
            + data
            + struct.pack(">I", zlib.crc32(type_ + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


_ZERO_WIDTH_RE = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff]')


def sanitize_filename_report(name: str) -> str:
    """Sanitize *name* for safe use as a filename component.

    Removes/replaces characters unsafe for filenames across major OSes,
    strips zero-width Unicode characters, and strips leading/trailing dots
    and underscores.  Returns ``"_unnamed"`` if the result is empty.

    Args:
        name: Raw name string (may contain Korean, special chars).

    Returns:
        Cleaned filename-safe string.
    """
    sanitized = _ZERO_WIDTH_RE.sub('', name)
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', sanitized)
    sanitized = sanitized.strip('._')
    if not sanitized:
        return "_unnamed"
    return sanitized
