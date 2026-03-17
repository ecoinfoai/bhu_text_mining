"""Domain concept extraction from preprocessed textbook text.

Extracts domain-specific terminology (Korean nouns, bilingual terms)
from textbook chapters, with frequency filtering, caching, and
structured YAML output.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from forma.textbook_preprocessor import clean_textbook_text, extract_bilingual_terms

logger = logging.getLogger(__name__)

__all__ = [
    "TextbookConcept",
    "extract_concepts",
    "extract_multi_chapter",
    "save_concepts_yaml",
    "load_concepts_yaml",
]

# ----------------------------------------------------------------
# Domain stopwords — general nouns that are not domain-specific
# ----------------------------------------------------------------

DOMAIN_STOPWORDS: frozenset[str] = frozenset({
    "것", "수", "때", "등", "중", "위", "및", "또는",
    "그림", "표", "참고", "경우", "이상", "이하", "정도",
    "부위", "부분", "기능", "구조", "역할", "특징", "과정",
    "결과", "종류", "방법", "이름", "형태", "상태",
})


# ----------------------------------------------------------------
# TextbookConcept dataclass (T011)
# ----------------------------------------------------------------


@dataclass
class TextbookConcept:
    """A domain term extracted from a textbook chapter.

    Attributes:
        name_ko: Korean term (e.g. "표피").
        name_en: English equivalent (e.g. "epidermis"), None if
            no bilingual annotation found.
        chapter: Source chapter identifier (e.g. "3장 피부").
        frequency: Occurrence count in the chapter.
        context_sentence: One representative sentence containing
            the term.
        is_bilingual: True if extracted from Korean(English) pattern.
    """

    name_ko: str
    name_en: str | None
    chapter: str
    frequency: int
    context_sentence: str
    is_bilingual: bool


# ----------------------------------------------------------------
# Sentence splitting helper
# ----------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using common Korean sentence delimiters.

    Args:
        text: Input text.

    Returns:
        List of non-empty sentences.
    """
    # Split on period, exclamation, question mark followed by space or end
    sentences = re.split(r"[.!?]\s*", text)
    return [s.strip() for s in sentences if s.strip()]


# ----------------------------------------------------------------
# T011: extract_concepts
# ----------------------------------------------------------------


def extract_concepts(
    cleaned_text: str,
    chapter_name: str,
    min_freq: int = 2,
) -> list[TextbookConcept]:
    """Extract domain concepts from cleaned textbook text.

    Steps:
        1. Extract bilingual terms via ``extract_bilingual_terms()``.
        2. Extract all nouns via KoNLPy Okt.
        3. Count frequencies.
        4. Filter: freq >= min_freq OR is_bilingual.
        5. Filter out domain stopwords.
        6. For each concept: find first sentence containing the term.
        7. Return sorted by frequency descending.

    Args:
        cleaned_text: Preprocessed textbook text.
        chapter_name: Chapter identifier (e.g. "3장 피부").
        min_freq: Minimum frequency threshold for non-bilingual terms.
            Defaults to 2.

    Returns:
        List of TextbookConcept sorted by frequency descending.
    """
    from konlpy.tag import Okt

    if not cleaned_text.strip():
        return []

    # Step 1: Extract bilingual terms
    bilingual_terms = extract_bilingual_terms(cleaned_text)
    bilingual_dict: dict[str, str] = {}
    for ko, en in bilingual_terms:
        if ko not in bilingual_dict:
            bilingual_dict[ko] = en

    # Step 2: Extract all nouns via KoNLPy Okt
    okt = Okt()
    nouns = okt.nouns(cleaned_text)

    # Step 3: Count frequencies
    noun_counts = Counter(nouns)

    # Step 4 & 5: Filter by frequency and stopwords
    sentences = _split_sentences(cleaned_text)

    concepts: list[TextbookConcept] = []
    seen: set[str] = set()

    # Process bilingual terms first (always included)
    for ko, en in bilingual_terms:
        if ko in seen:
            continue
        if ko in DOMAIN_STOPWORDS:
            continue
        if len(ko) <= 1:
            continue

        seen.add(ko)
        freq = noun_counts.get(ko, 1)

        # Find context sentence
        context = _find_context_sentence(ko, sentences)

        concepts.append(TextbookConcept(
            name_ko=ko,
            name_en=en,
            chapter=chapter_name,
            frequency=freq,
            context_sentence=context,
            is_bilingual=True,
        ))

    # Process non-bilingual nouns
    for noun, freq in noun_counts.items():
        if noun in seen:
            continue
        if noun in DOMAIN_STOPWORDS:
            continue
        if len(noun) <= 1:
            continue
        if freq < min_freq:
            continue

        seen.add(noun)

        context = _find_context_sentence(noun, sentences)

        concepts.append(TextbookConcept(
            name_ko=noun,
            name_en=bilingual_dict.get(noun),
            chapter=chapter_name,
            frequency=freq,
            context_sentence=context,
            is_bilingual=False,
        ))

    # Step 7: Sort by frequency descending
    concepts.sort(key=lambda c: c.frequency, reverse=True)

    return concepts


def _find_context_sentence(term: str, sentences: list[str]) -> str:
    """Find the first sentence containing the given term.

    Args:
        term: The term to search for.
        sentences: List of sentences to search.

    Returns:
        First sentence containing the term, or empty string if not found.
    """
    for sent in sentences:
        if term in sent:
            return sent
    return ""


# ----------------------------------------------------------------
# T012: extract_multi_chapter
# ----------------------------------------------------------------


def extract_multi_chapter(
    textbook_paths: list[str],
    min_freq: int = 2,
    use_cache: bool = False,
) -> dict[str, list[TextbookConcept]]:
    """Extract concepts from multiple textbook chapter files.

    For each path: reads the file (UTF-8), cleans the text, extracts
    concepts. Chapter name is derived from the filename (strip .txt
    extension and leading path).

    Args:
        textbook_paths: List of file paths to textbook chapter files.
        min_freq: Minimum frequency threshold for non-bilingual terms.
        use_cache: If True, uses file content hash caching.

    Returns:
        Dict mapping chapter name to list of TextbookConcept.
    """
    result: dict[str, list[TextbookConcept]] = {}

    for path_str in textbook_paths:
        path = Path(path_str)
        chapter_name = path.stem  # filename without extension

        if use_cache:
            cached = _load_cache(path_str)
            if cached is not None:
                logger.info("캐시에서 개념 로드: %s", path_str)
                result[chapter_name] = cached
                continue

        # Read file
        raw_text = path.read_text(encoding="utf-8")

        # Clean text
        cleaned = clean_textbook_text(raw_text)

        # Extract concepts
        concepts = extract_concepts(cleaned, chapter_name, min_freq=min_freq)

        # Save cache if enabled
        if use_cache:
            _save_cache(path_str, raw_text, concepts)

        result[chapter_name] = concepts

    return result


# ----------------------------------------------------------------
# T013: Concept caching
# ----------------------------------------------------------------


def _compute_file_hash(content: str) -> str:
    """Compute MD5 hash of file content.

    Args:
        content: File content string.

    Returns:
        Hex digest of MD5 hash.
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _cache_path_for(file_path: str) -> Path:
    """Get cache file path for a given textbook file.

    Args:
        file_path: Path to the textbook file.

    Returns:
        Path to the cache file.
    """
    return Path(file_path + ".concepts_cache.yaml")


def _save_cache(
    file_path: str,
    content: str,
    concepts: list[TextbookConcept],
) -> None:
    """Save concept extraction cache.

    Args:
        file_path: Path to the original textbook file.
        content: File content for hash computation.
        concepts: Extracted concepts to cache.
    """
    cache_path = _cache_path_for(file_path)
    cache_data = {
        "hash": _compute_file_hash(content),
        "concepts": [
            {
                "name_ko": c.name_ko,
                "name_en": c.name_en,
                "chapter": c.chapter,
                "frequency": c.frequency,
                "context": c.context_sentence,
                "is_bilingual": c.is_bilingual,
            }
            for c in concepts
        ],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        yaml.dump(cache_data, f, allow_unicode=True, default_flow_style=False)


def _load_cache(file_path: str) -> list[TextbookConcept] | None:
    """Load concept extraction cache if valid.

    Checks if cache exists and file hash matches. Returns None
    if cache is missing, corrupt, or stale.

    Args:
        file_path: Path to the original textbook file.

    Returns:
        List of cached TextbookConcept, or None if cache invalid.
    """
    cache_path = _cache_path_for(file_path)
    if not cache_path.exists():
        return None

    try:
        # Read current file content
        current_content = Path(file_path).read_text(encoding="utf-8")
        current_hash = _compute_file_hash(current_content)

        with open(cache_path, encoding="utf-8") as f:
            cache_data = yaml.safe_load(f)

        if cache_data is None or cache_data.get("hash") != current_hash:
            return None

        return [
            TextbookConcept(
                name_ko=c["name_ko"],
                name_en=c.get("name_en"),
                chapter=c["chapter"],
                frequency=c["frequency"],
                context_sentence=c.get("context", ""),
                is_bilingual=c.get("is_bilingual", False),
            )
            for c in cache_data.get("concepts", [])
        ]
    except Exception:
        logger.warning("캐시 로드 실패: %s", cache_path, exc_info=True)
        return None


# ----------------------------------------------------------------
# T014: save_concepts_yaml / load_concepts_yaml
# ----------------------------------------------------------------


def save_concepts_yaml(
    concepts_by_chapter: dict[str, list[TextbookConcept]],
    output_path: str,
) -> None:
    """Save extracted concepts to a YAML file.

    Output format follows the contract in cli-interface.md::

        chapters:
          "3장 피부":
            concepts:
              - name_ko: "표피"
                name_en: "epidermis"
                frequency: 15
                context: "피부는 표피(epidermis)와..."
                is_bilingual: true

    Args:
        concepts_by_chapter: Dict mapping chapter name to concept list.
        output_path: Path to write the YAML file.
    """
    data: dict[str, dict] = {"chapters": {}}

    for chapter_name, concepts in concepts_by_chapter.items():
        data["chapters"][chapter_name] = {
            "concepts": [
                {
                    "name_ko": c.name_ko,
                    "name_en": c.name_en,
                    "frequency": c.frequency,
                    "context": c.context_sentence,
                    "is_bilingual": c.is_bilingual,
                }
                for c in concepts
            ]
        }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def load_concepts_yaml(
    path: str,
) -> dict[str, list[TextbookConcept]]:
    """Load concepts from a YAML file produced by save_concepts_yaml.

    Args:
        path: Path to the concepts YAML file.

    Returns:
        Dict mapping chapter name to list of TextbookConcept.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If YAML structure is invalid.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "chapters" not in data:
        raise ValueError(f"올바르지 않은 개념 YAML 형식: {path}")

    result: dict[str, list[TextbookConcept]] = {}

    for chapter_name, chapter_data in data["chapters"].items():
        concepts: list[TextbookConcept] = []
        for c in chapter_data.get("concepts", []):
            concepts.append(TextbookConcept(
                name_ko=c["name_ko"],
                name_en=c.get("name_en"),
                chapter=chapter_name,
                frequency=c["frequency"],
                context_sentence=c.get("context", ""),
                is_bilingual=c.get("is_bilingual", False),
            ))
        result[chapter_name] = concepts

    return result
