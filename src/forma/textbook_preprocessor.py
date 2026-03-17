"""Textbook text preprocessing for domain knowledge extraction.

Handles cleaning of PDF-extracted textbook text by removing page numbers,
chapter headers with wide letter-spacing, "HUMAN ANATOMY & PHYSIOLOGY"
markers, and other non-body elements while preserving figure/table captions
and body text.
"""

from __future__ import annotations

import re

__all__ = [
    "clean_textbook_text",
    "extract_bilingual_terms",
]


def clean_textbook_text(raw_text: str) -> str:
    """Clean PDF-extracted textbook text by removing non-body elements.

    Applies regex patterns in order to remove:
        1. Wide letter-spacing headers (e.g. ``C H A P T E R  03``)
        2. ``H U M A N  A N A T O M Y  &  P H Y S I O L O G Y`` markers
        3. Chapter page number lines (e.g. ``제 3 장 피부   43``)
        4. Standalone page numbers (just a number on a line)
        5. Collapses multiple blank lines to a single blank line

    Body text, figure/table captions, and Korean content are preserved.

    Args:
        raw_text: Raw text extracted from PDF.

    Returns:
        Cleaned text with non-body elements removed.
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Wide letter-spacing lines: 3+ uppercase letters separated by spaces
    #    (e.g. "C H A P T E R  03", "H U M A N  A N A T O M Y  &  P H Y S I O L O G Y")
    text = re.sub(r"^[A-Z]\s+[A-Z]\s+[A-Z][\sA-Z&0-9]*$", "", text, flags=re.MULTILINE)

    # 2. Chapter page number lines: "제 N 장 <title>   <page_number>"
    text = re.sub(r"^제\s*\d+\s*장\s+\S+\s+\d+\s*$", "", text, flags=re.MULTILINE)

    # 3. Standalone page numbers: just a number on a line
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # 4. Collapse multiple blank lines to single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def extract_bilingual_terms(text: str) -> list[tuple[str, str]]:
    """Extract Korean-English bilingual terms from textbook text.

    Finds patterns like ``표피(epidermis)`` where a Korean word is
    followed by an English term in parentheses. The English part
    must contain at least one alphabetic character.

    Also handles complex parenthetical forms like
    ``피부밑조직(피하조직, subcutaneous tissue)`` by extracting the
    English portion from within the parentheses.

    Args:
        text: Preprocessed textbook text.

    Returns:
        List of (korean_term, english_term) tuples.
    """
    results: list[tuple[str, str]] = []

    # Pattern: Korean word followed by parenthesized content containing English
    # The parenthetical may contain Korean synonyms and English terms
    pattern = re.compile(r"(\w+)\s*\(([^)]+)\)")

    for match in pattern.finditer(text):
        korean = match.group(1)
        paren_content = match.group(2)

        # Check if parenthetical contains English alphabetic characters
        if not re.search(r"[a-zA-Z]", paren_content):
            continue

        # Extract the English portion from the parenthetical content
        # Handle cases like "피하조직, subcutaneous tissue" -> "subcutaneous tissue"
        # Split by comma and find parts with English
        parts = [p.strip() for p in paren_content.split(",")]
        english_parts = [p for p in parts if re.search(r"[a-zA-Z]", p)]

        if english_parts:
            english = ", ".join(english_parts).strip()
            results.append((korean, english))

    return results
