"""STT lecture transcript preprocessing pipeline.

Handles Korean-English mixed content from speech-to-text outputs.
Provides filler removal, repeated-char normalization, language-boundary
splitting, stopword filtering, and a full 8-step pipeline that returns
a ``CleanedTranscript`` dataclass.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

KOREAN_FILLERS: frozenset[str] = frozenset({
    "어", "음", "그", "저", "뭐", "아", "예", "네", "응",
    "어어", "음음", "그그", "저저", "뭐뭐",
})

BIOLOGY_ABBREVIATIONS: frozenset[str] = frozenset({
    "ATP", "ADP", "AMP", "DNA", "RNA", "mRNA", "tRNA",
    "rRNA", "pH", "Ca", "Na", "K", "Cl", "Fe", "O2",
    "CO2", "H2O", "ECG", "EEG", "MRI", "CT", "IgG",
    "IgE", "NK", "T세포", "B세포", "ACh", "GABA", "CNS",
    "PNS", "ANS", "GFR", "ADH", "FSH", "LH",
})

MAX_TRANSCRIPT_LENGTH: int = 50_000

# Pre-built regex for Korean Unicode ranges:
#   Hangul Syllables: \uAC00-\uD7AF
#   Hangul Jamo: \u1100-\u11FF
#   Hangul Compat Jamo: \u3131-\u3163
_HANGUL = r"\uAC00-\uD7AF\u3131-\u3163\u1100-\u11FF"
_LATIN = r"A-Za-z0-9"

# ------------------------------------------------------------------
# Korean grammar stopwords (~70 items)
# ------------------------------------------------------------------
_KOREAN_GRAMMAR: frozenset[str] = frozenset({
    "은", "는", "이", "가", "을", "를", "에", "에서", "의",
    "와", "과", "로", "으로", "도", "만", "까지", "부터",
    "다", "라", "야", "하고", "하면", "하는", "하여",
    "한", "할", "함", "합", "했", "하다", "있다", "없다",
    "것", "수", "등", "및", "또", "더", "그리고", "그래서",
    "그러나", "그런데", "또한", "하지만", "그렇지만",
    "때문에", "위해", "통해", "대해", "따라", "관한",
    "된", "되는", "되어", "되었다", "되면", "되고",
    "인", "인데", "이런", "저런", "그런",
    "안", "못", "잘", "매우", "아주", "정말", "진짜",
    "좀", "약간", "거의", "이미", "아직", "바로", "다시",
})

# ------------------------------------------------------------------
# Lecture discourse markers
# ------------------------------------------------------------------
_LECTURE_DISCOURSE: frozenset[str] = frozenset({
    "여기서", "자", "그래서", "그러면", "그런데",
    "이렇게", "저렇게", "봅시다", "보겠습니다",
    "말씀드리겠습니다", "넘어가겠습니다",
    "살펴보겠습니다", "다음으로", "먼저",
    "마지막으로", "정리하면", "요약하면",
    "결론적으로", "따라서", "즉",
    "okay", "right", "well", "basically",
    "actually", "so", "like", "just",
    "um", "uh", "you", "know",
})

# ------------------------------------------------------------------
# English function words (~180 items, common subset)
# ------------------------------------------------------------------
_ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "to", "in", "is", "are",
    "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "and", "but", "or", "nor", "not", "no", "for",
    "with", "at", "by", "from", "on", "as", "into",
    "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "than",
    "too", "very", "just", "because", "about", "up",
    "down", "if", "while", "until", "that", "which",
    "who", "whom", "this", "these", "those", "what",
    "it", "its", "he", "she", "they", "them", "we",
    "us", "i", "me", "my", "your", "his", "her",
    "our", "their", "him", "also", "any", "many",
    "much", "still", "already", "even", "ever", "never",
    "now", "often", "since", "yet", "however",
    "therefore", "thus", "hence", "although", "though",
    "either", "neither", "whether", "whereas",
    "while", "unless", "per", "via", "among",
})


# ------------------------------------------------------------------
# Dataclass
# ------------------------------------------------------------------

@dataclass
class CleanedTranscript:
    """Result of the preprocessing pipeline.

    Attributes:
        class_id: Class identifier (e.g. "A").
        week: Week number.
        source_path: Original file path.
        raw_text: Original text before cleaning.
        cleaned_text: Text after all cleaning steps.
        encoding_used: Detected file encoding.
        char_count_raw: Character count of raw text.
        char_count_cleaned: Character count of cleaned text.
    """

    class_id: str
    week: int
    source_path: str
    raw_text: str
    cleaned_text: str
    encoding_used: str
    char_count_raw: int
    char_count_cleaned: int


# ------------------------------------------------------------------
# Public functions
# ------------------------------------------------------------------

def validate_path(path: str) -> None:
    """Validate file path for security.

    Rejects paths containing directory traversal sequences.

    Args:
        path: File path to validate.

    Raises:
        ValueError: If path contains traversal sequences or null bytes.
    """
    if '\x00' in path:
        raise ValueError(
            f"경로에 null 바이트가 포함되어 있습니다: {path!r}"
        )
    if re.search(r'(?:^|[/\\])\.\.(?:[/\\]|$)', path):
        raise ValueError(
            f"경로에 디렉토리 탐색 시퀀스가 포함되어 있습니다. "
            f"보안상 허용되지 않습니다: {path}"
        )


def load_and_decode(path: str) -> tuple[str, str]:
    """Load a file, trying UTF-8 first then EUC-KR fallback.

    Args:
        path: Path to the text file.

    Returns:
        Tuple of (decoded text, encoding name).

    Raises:
        FileNotFoundError: If file does not exist.
        UnicodeDecodeError: If neither encoding works.
    """
    try:
        with open(path, encoding="utf-8-sig") as f:
            return f.read(), "utf-8"
    except UnicodeDecodeError:
        logger.warning(
            "UTF-8 디코딩 실패, EUC-KR로 재시도: %s", path,
        )
        with open(path, encoding="euc-kr") as f:
            return f.read(), "euc-kr"


def remove_fillers(text: str) -> str:
    """Remove Korean filler words from text.

    Uses whitespace-based tokenization to avoid removing
    substrings of longer words (e.g. "어" in "어떤").

    Args:
        text: Input text.

    Returns:
        Text with filler words removed.
    """
    tokens = text.split()
    filtered = [t for t in tokens if t not in KOREAN_FILLERS]
    return " ".join(filtered)


def normalize_repeated_chars(text: str) -> str:
    """Compress 3+ consecutive identical characters to 2.

    Args:
        text: Input text.

    Returns:
        Text with repeated characters normalized.
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def _split_at_boundary(word: str) -> str:
    """Insert space at Korean-English language boundaries in a word.

    Args:
        word: Single word token.

    Returns:
        Word with spaces inserted at language boundaries.
    """
    # Latin -> Korean boundary
    word = re.sub(
        rf"([{_LATIN}])([{_HANGUL}])", r"\1 \2", word,
    )
    # Korean -> Latin boundary
    word = re.sub(
        rf"([{_HANGUL}])([{_LATIN}])", r"\1 \2", word,
    )
    return word


def split_mixed_tokens(
    text: str,
    preserve: frozenset[str] | None = None,
) -> str:
    """Insert space at Korean-English language boundaries.

    Detects transitions between Hangul and Latin character
    ranges and inserts a space at each boundary.  Tokens that
    match an entry in ``preserve`` are kept intact.

    Args:
        text: Input text with potentially mixed tokens.
        preserve: Set of tokens to skip splitting.
            Defaults to ``BIOLOGY_ABBREVIATIONS``.

    Returns:
        Text with spaces inserted at language boundaries.
    """
    if preserve is None:
        preserve = BIOLOGY_ABBREVIATIONS

    words = text.split()
    result: list[str] = []
    for word in words:
        if word in preserve or _starts_with_preserved(word, preserve):
            result.append(word)
        else:
            result.append(_split_at_boundary(word))
    return " ".join(result)


def _starts_with_preserved(
    word: str, preserve: frozenset[str],
) -> bool:
    """Check if word starts with a preserved abbreviation.

    Returns True if the word begins with a multi-script abbreviation
    (containing both Latin and Hangul) from the preserve set.

    Args:
        word: Token to check.
        preserve: Set of abbreviations to preserve.

    Returns:
        True if word starts with a preserved abbreviation.
    """
    for abbr in preserve:
        if len(abbr) < 2:
            continue
        # Only protect multi-script abbreviations (e.g. T세포, B세포)
        has_latin = any(c.isascii() and c.isalpha() for c in abbr)
        has_hangul = bool(re.search(rf"[{_HANGUL}]", abbr))
        if has_latin and has_hangul and word.startswith(abbr):
            return True
    return False


def build_stopwords(
    extra_stopwords: list[str] | None = None,
) -> frozenset[str]:
    """Build a 3-layer stopword set with optional extras.

    Layers:
        1. Korean grammar particles and function words
        2. Lecture discourse markers (Korean + English)
        3. English function words

    Args:
        extra_stopwords: Additional stopwords to include.

    Returns:
        Merged frozenset of all stopwords.
    """
    combined: set[str] = set()
    combined.update(_KOREAN_GRAMMAR)
    combined.update(_LECTURE_DISCOURSE)
    combined.update(_ENGLISH_STOPWORDS)
    if extra_stopwords:
        combined.update(extra_stopwords)
    return frozenset(combined)


def filter_stopwords(
    words: list[str],
    stopwords: frozenset[str],
    abbreviations: frozenset[str],
) -> list[str]:
    """Remove stopwords but preserve domain abbreviations.

    Args:
        words: List of word tokens.
        stopwords: Set of words to remove.
        abbreviations: Set of abbreviations to preserve
            even if they appear in stopwords.

    Returns:
        Filtered word list.
    """
    return [
        w for w in words
        if w in abbreviations or w not in stopwords
    ]


def preprocess_transcript(
    path: str,
    class_id: str,
    week: int,
    extra_stopwords: list[str] | None = None,
    extra_abbreviations: list[str] | None = None,
) -> CleanedTranscript:
    """Full 8-step transcript preprocessing pipeline.

    Steps:
        1. Validate path (reject ``../``)
        2. Load and decode (UTF-8, EUC-KR fallback)
        3. Length check (> MAX_TRANSCRIPT_LENGTH)
        4. Remove fillers
        5. Normalize repeated characters
        6. Split mixed tokens
        7. Build stopwords/abbreviations and filter
        8. Validate non-empty after cleaning

    Args:
        path: Path to transcript file.
        class_id: Class identifier (e.g. "A").
        week: Week number.
        extra_stopwords: Additional stopwords to include.
        extra_abbreviations: Additional abbreviations to
            preserve during filtering.

    Returns:
        CleanedTranscript with all fields populated.

    Raises:
        ValueError: If path has traversal, file is empty,
            exceeds length limit, or is empty after cleaning.
    """
    # Step 1: validate path
    validate_path(path)

    # Step 2: load and decode
    raw_text, encoding_used = load_and_decode(path)

    # Check empty file
    if not raw_text.strip():
        raise ValueError(
            f"파일이 비어 있습니다: {path}"
        )

    # Step 3: length check
    if len(raw_text) > MAX_TRANSCRIPT_LENGTH:
        raise ValueError(
            f"파일이 최대 길이({MAX_TRANSCRIPT_LENGTH}자)를 "
            f"초과합니다: {len(raw_text)}자 ({path})"
        )

    # Step 4: remove fillers
    text = remove_fillers(raw_text)

    # Step 5: normalize repeated chars
    text = normalize_repeated_chars(text)

    # Step 6: split mixed tokens
    text = split_mixed_tokens(text)

    # Step 7: build stopwords and filter
    abbrevs = set(BIOLOGY_ABBREVIATIONS)
    if extra_abbreviations:
        abbrevs.update(extra_abbreviations)
    frozen_abbrevs = frozenset(abbrevs)

    stopwords = build_stopwords(extra_stopwords)
    tokens = text.split()
    filtered = filter_stopwords(tokens, stopwords, frozen_abbrevs)
    cleaned_text = " ".join(filtered)

    # Step 8: validate non-empty after cleaning
    if not cleaned_text.strip():
        raise ValueError(
            f"전처리 후 텍스트가 비어 있습니다: {path}"
        )

    return CleanedTranscript(
        class_id=class_id,
        week=week,
        source_path=path,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        encoding_used=encoding_used,
        char_count_raw=len(raw_text),
        char_count_cleaned=len(cleaned_text),
    )
