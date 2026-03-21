"""Domain concept extraction from preprocessed textbook text.

Provides v1 (KoNLPy word-level) and v2 (LLM semantic concept)
extraction from textbook chapters, with frequency filtering, caching,
and structured YAML output.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from forma.textbook_preprocessor import clean_textbook_text, extract_bilingual_terms

logger = logging.getLogger(__name__)

__all__ = [
    "TextbookConcept",
    "DomainConcept",
    "TextbookChunk",
    "extract_concepts",
    "extract_multi_chapter",
    "extract_concepts_llm",
    "extract_concepts_llm_chunked",
    "extract_multi_chapter_llm",
    "build_extraction_prompt",
    "save_concepts_yaml",
    "load_concepts_yaml",
    "MajorTopic",
    "SubTopic",
    "TopicHierarchy",
    "parse_summary_hierarchy",
    "chunk_textbook_by_sections",
    "_merge_chunk_concepts",
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
    concepts_by_chapter: dict[str, list],
    output_path: str,
) -> None:
    """Save extracted concepts to a YAML file.

    Supports both v1 (TextbookConcept) and v2 (DomainConcept) formats.
    Auto-detects based on the type of the first concept in the first
    non-empty chapter.

    Args:
        concepts_by_chapter: Dict mapping chapter name to concept list.
        output_path: Path to write the YAML file.
    """
    data: dict[str, dict] = {"chapters": {}}

    # Detect format from first non-empty chapter
    is_v2 = False
    for concepts in concepts_by_chapter.values():
        if concepts:
            is_v2 = isinstance(concepts[0], DomainConcept)
            break

    for chapter_name, concepts in concepts_by_chapter.items():
        if is_v2:
            data["chapters"][chapter_name] = {
                "concepts": [
                    {
                        "concept": c.concept,
                        "description": c.description,
                        "key_terms": c.key_terms,
                        "importance": c.importance,
                        "section": c.section,
                        "major_topic": c.major_topic,
                        "sub_topic": c.sub_topic,
                    }
                    for c in concepts
                ]
            }
        else:
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
) -> dict[str, list]:
    """Load concepts from a YAML file produced by save_concepts_yaml.

    Auto-detects v1 (TextbookConcept) vs v2 (DomainConcept) format
    based on the keys present in the first concept.

    Args:
        path: Path to the concepts YAML file.

    Returns:
        Dict mapping chapter name to list of TextbookConcept or DomainConcept.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If YAML structure is invalid.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "chapters" not in data:
        raise ValueError(f"올바르지 않은 개념 YAML 형식: {path}")

    result: dict[str, list] = {}

    for chapter_name, chapter_data in data["chapters"].items():
        raw_concepts = chapter_data.get("concepts", [])

        # Detect v2 format: presence of "concept" key (vs "name_ko")
        is_v2 = raw_concepts and "concept" in raw_concepts[0]

        if is_v2:
            concepts: list = []
            for c in raw_concepts:
                concepts.append(DomainConcept(
                    concept=c["concept"],
                    description=c.get("description", ""),
                    key_terms=c.get("key_terms", []),
                    importance=c.get("importance", "medium"),
                    section=c.get("section", ""),
                    chapter=chapter_name,
                    major_topic=c.get("major_topic", ""),
                    sub_topic=c.get("sub_topic", ""),
                ))
            result[chapter_name] = concepts
        else:
            concepts_v1: list = []
            for c in raw_concepts:
                concepts_v1.append(TextbookConcept(
                    name_ko=c["name_ko"],
                    name_en=c.get("name_en"),
                    chapter=chapter_name,
                    frequency=c["frequency"],
                    context_sentence=c.get("context", ""),
                    is_bilingual=c.get("is_bilingual", False),
                ))
            result[chapter_name] = concepts_v1

    return result


# ================================================================
# v2: LLM-based semantic concept extraction
# ================================================================


@dataclass
class DomainConcept:
    """A semantic concept extracted from a textbook chapter by LLM.

    NOT a single word — a meaningful phrase describing a mechanism,
    structure, or process.

    Attributes:
        concept: Meaningful phrase (e.g. "표피의 4층 구조와 각 층의 역할").
        description: Explanation of what this concept covers.
        key_terms: Domain-specific terms (e.g. [표피, 종자층, 과립층]).
        importance: "high" / "medium" / "low".
        section: Source section in textbook.
        chapter: Chapter identifier (e.g. "3장 피부").
    """

    concept: str
    description: str
    key_terms: list[str] = field(default_factory=list)
    importance: str = "medium"
    section: str = ""
    chapter: str = ""
    major_topic: str = ""
    sub_topic: str = ""


# ----------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------

_SYSTEM_INSTRUCTION = "당신은 해부생리학 교과서 분석 전문가입니다."

_EXTRACTION_PROMPT_TEMPLATE = """\
아래는 교과서 본문 텍스트입니다. 이 텍스트에서 핵심 도메인 개념을 추출해주세요.

## 규칙
1. 각 개념은 의미 단위(구, 절 수준)로 추출하세요. 단일 단어가 아닌 메커니즘이나 구조를 설명하는 구입니다.
2. 일상용어(것, 수, 때, 대해, 통한, 여러, 또한), 접속어, 일반명사는 절대 포함하지 마세요.
3. 해부생리학 도메인 전문 개념만 추출하세요.
4. 각 개념에 대해 다음 필드를 포함하세요:
   - concept: 의미 단위 이름 (예: "표피의 4층 구조와 각 층의 역할")
   - description: 개념을 15자 이내로 간결하게 설명
   - key_terms: 핵심 도메인 용어 3~5개 (한국어)
   - importance: high / medium / low
   - section: 교과서 내 소속 절
5. 최대 30개 개념만 추출하세요. importance가 높은 순으로 선별하세요.
6. description은 반드시 짧게 작성하세요. 긴 문장은 금지합니다.

## 출력 형식 (YAML, 코드 펜스 없이 바로 출력)
concepts:
  - concept: "개념 이름"
    description: "15자 이내 간결 설명"
    key_terms: [용어1, 용어2, 용어3]
    importance: high
    section: "절 이름"

{structure_section}

## 교과서 본문
{body_text}
"""


def build_extraction_prompt(
    cleaned_body: str,
    structure_guide: str | None = None,
) -> str:
    """Construct the LLM prompt for concept extraction.

    Args:
        cleaned_body: Cleaned textbook body text.
        structure_guide: Optional chapter summary Markdown for structure.

    Returns:
        Formatted prompt string.
    """
    if structure_guide:
        structure_section = (
            "## 챕터 구조 가이드 (참고)\n"
            f"{structure_guide}\n"
        )
    else:
        structure_section = ""

    return _EXTRACTION_PROMPT_TEMPLATE.format(
        body_text=cleaned_body,
        structure_section=structure_section,
    )


# ----------------------------------------------------------------
# LLM extraction
# ----------------------------------------------------------------


def _recover_truncated_yaml(text: str) -> dict | None:
    """Try to recover a truncated YAML concept list.

    Strategy: find the last complete ``- concept:`` entry by scanning
    backwards for a line matching ``  - concept:`` and truncating there,
    then re-parse.

    Returns:
        Parsed dict with 'concepts' key, or None on failure.
    """
    lines = text.split("\n")
    # Walk backwards to find the last *complete* concept block start
    # A concept block is complete if there's another ``- concept:`` after it
    # (meaning the *previous* block finished). So find the second-to-last marker.
    markers = [i for i, ln in enumerate(lines) if ln.strip().startswith("- concept:")]
    if len(markers) < 2:
        return None

    # Keep everything up to (but not including) the last incomplete block
    truncated = "\n".join(lines[: markers[-1]])
    try:
        data = yaml.safe_load(truncated)
        if isinstance(data, dict) and "concepts" in data:
            logger.info(
                "잘린 응답 구제 성공: %d개 개념 파싱됨",
                len(data["concepts"]),
            )
            return data
    except yaml.YAMLError:
        pass
    return None


def _parse_llm_concepts(
    response_text: str,
    chapter_name: str,
) -> list[DomainConcept]:
    """Parse LLM YAML response into DomainConcept list.

    Gracefully handles malformed YAML and missing fields.

    Args:
        response_text: Raw LLM text response.
        chapter_name: Chapter identifier to set on each concept.

    Returns:
        List of valid DomainConcept objects. Empty list on parse failure.
    """
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        # Remove first line (```yaml or ```) and last line (```)
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        # Truncated response recovery: cut back to last complete concept entry
        logger.warning("YAML 파싱 실패, 잘린 응답 구제 시도: %s", exc)
        data = _recover_truncated_yaml(text)
        if data is None:
            logger.warning(
                "구제 실패\n--- 응답 원문 (처음 500자) ---\n%s", text[:500],
            )
            return []

    if not isinstance(data, dict) or "concepts" not in data:
        logger.warning("LLM 응답에 'concepts' 키가 없음")
        return []

    concepts: list[DomainConcept] = []
    for item in data["concepts"]:
        if not isinstance(item, dict):
            continue
        # Required fields
        concept_name = item.get("concept")
        description = item.get("description")
        key_terms = item.get("key_terms")
        if not concept_name or not description or not key_terms:
            logger.debug("필수 필드 누락, 건너뜀: %s", item)
            continue

        concepts.append(DomainConcept(
            concept=concept_name,
            description=description,
            key_terms=key_terms if isinstance(key_terms, list) else [key_terms],
            importance=item.get("importance", "medium"),
            section=item.get("section", ""),
            chapter=chapter_name,
        ))

    return concepts


def _v2_cache_path_for(file_path: str) -> Path:
    """Get v2 cache file path for a given textbook file."""
    return Path(file_path + ".concepts_v2_cache.yaml")


def _save_v2_cache(
    file_path: str,
    content: str,
    concepts: list[DomainConcept],
) -> None:
    """Save v2 concept extraction cache."""
    cache_path = _v2_cache_path_for(file_path)
    cache_data = {
        "hash": _compute_file_hash(content),
        "concepts": [
            {
                "concept": c.concept,
                "description": c.description,
                "key_terms": c.key_terms,
                "importance": c.importance,
                "section": c.section,
                "chapter": c.chapter,
            }
            for c in concepts
        ],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        yaml.dump(cache_data, f, allow_unicode=True, default_flow_style=False)


def _load_v2_cache(file_path: str) -> list[DomainConcept] | None:
    """Load v2 concept extraction cache if valid."""
    cache_path = _v2_cache_path_for(file_path)
    if not cache_path.exists():
        return None

    try:
        current_content = Path(file_path).read_text(encoding="utf-8")
        current_hash = _compute_file_hash(current_content)

        with open(cache_path, encoding="utf-8") as f:
            cache_data = yaml.safe_load(f)

        if cache_data is None or cache_data.get("hash") != current_hash:
            return None

        return [
            DomainConcept(
                concept=c["concept"],
                description=c.get("description", ""),
                key_terms=c.get("key_terms", []),
                importance=c.get("importance", "medium"),
                section=c.get("section", ""),
                chapter=c.get("chapter", ""),
            )
            for c in cache_data.get("concepts", [])
        ]
    except Exception:
        logger.warning("v2 캐시 로드 실패: %s", cache_path, exc_info=True)
        return None


def create_provider(model: str | None = None):
    """Create LLM provider for concept extraction.

    Reads API key from config.json / forma.json via ``load_config()``
    before falling back to environment variables.

    Args:
        model: Optional model ID override.

    Returns:
        LLMProvider instance.
    """
    from forma.config import get_llm_config, load_config
    from forma.llm_provider import create_provider as _create

    try:
        cfg = load_config()
        llm_cfg = get_llm_config(cfg)
    except FileNotFoundError:
        llm_cfg = {}

    return _create(
        provider=llm_cfg.get("provider", "gemini"),
        api_key=llm_cfg.get("api_key"),
        model=model or llm_cfg.get("model"),
    )


_CHUNK_EXTRACTION_PROMPT_TEMPLATE = """\
아래는 교과서 본문의 일부입니다 (섹션: {section_context}).
이 텍스트에서 핵심 도메인 개념을 추출해주세요.

## 규칙
1. 각 개념은 의미 단위(구, 절 수준)로 추출하세요. 단일 단어가 아닌 메커니즘이나 구조를 설명하는 구입니다.
2. 일상용어(것, 수, 때, 대해, 통한, 여러, 또한), 접속어, 일반명사는 절대 포함하지 마세요.
3. 해부생리학 도메인 전문 개념만 추출하세요.
4. 각 개념에 대해 다음 필드를 포함하세요:
   - concept: 의미 단위 이름 (예: "표피의 4층 구조와 각 층의 역할")
   - description: 개념을 15자 이내로 간결하게 설명
   - key_terms: 핵심 도메인 용어 3~5개 (한국어)
   - importance: high / medium / low
   - section: 교과서 내 소속 절
5. 최대 10개 개념만 추출하세요. importance가 높은 순으로 선별하세요.
6. description은 반드시 짧게 작성하세요. 긴 문장은 금지합니다.

## 출력 형식 (YAML, 코드 펜스 없이 바로 출력)
concepts:
  - concept: "개념 이름"
    description: "15자 이내 간결 설명"
    key_terms: [용어1, 용어2, 용어3]
    importance: high
    section: "절 이름"

## 교과서 본문
{body_text}
"""


def extract_concepts_llm_chunked(
    textbook_path: str,
    summary_path: str | None = None,
    model: str | None = None,
    chapter_name: str | None = None,
    no_cache: bool = False,
) -> list[DomainConcept]:
    """Extract concepts from a large textbook via chunked LLM calls.

    Splits the textbook text into chunks using
    ``chunk_textbook_by_sections()``, calls the LLM for each chunk
    (max 10 concepts per chunk, max_tokens=4096), sets major_topic
    and sub_topic from chunk metadata, then merges results with
    ``_merge_chunk_concepts()``.

    Args:
        textbook_path: Path to textbook chapter text file.
        summary_path: Optional chapter summary Markdown path.
        model: Optional LLM model ID override.
        chapter_name: Override chapter name (default: from filename stem).
        no_cache: If True, skip cache lookup and force LLM call.

    Returns:
        List of DomainConcept. Empty list on failure.
    """
    import time

    path = Path(textbook_path)
    if chapter_name is None:
        chapter_name = path.stem

    # Check cache first
    if not no_cache:
        cached = _load_v2_cache(textbook_path)
        if cached is not None:
            logger.info("v2 캐시에서 개념 로드 (chunked): %s", textbook_path)
            return cached

    raw_text = path.read_text(encoding="utf-8")

    # Get cleaned body for chunking
    from forma.textbook_preprocessor import prepare_textbook_for_llm
    cleaned_body, _structure_guide = prepare_textbook_for_llm(
        raw_text, summary_path=summary_path,
    )

    if not cleaned_body.strip():
        logger.warning("본문이 비어 있음: %s", textbook_path)
        return []

    # Chunk the text
    chunks = chunk_textbook_by_sections(
        cleaned_body, summary_path=summary_path,
    )
    logger.info("청크 분할: %d개 청크", len(chunks))

    provider = create_provider(model=model)
    chunk_results: list[list[DomainConcept]] = []

    for idx, chunk in enumerate(chunks):
        section_context = " > ".join(chunk.section_path) if chunk.section_path else f"청크 {idx + 1}"

        prompt = _CHUNK_EXTRACTION_PROMPT_TEMPLATE.format(
            section_context=section_context,
            body_text=chunk.text,
        )

        # Rate limiting between LLM calls
        if idx > 0:
            time.sleep(4.0)

        try:
            response = provider.generate(
                prompt=prompt,
                max_tokens=4096,
                temperature=0.0,
                system_instruction=_SYSTEM_INSTRUCTION,
            )
            concepts = _parse_llm_concepts(response, chapter_name)

            # Set major_topic and sub_topic from chunk metadata
            for c in concepts:
                c.major_topic = chunk.major_topic
                c.sub_topic = chunk.sub_topic

            chunk_results.append(concepts)
            logger.info(
                "청크 %d/%d 완료: %d개 개념 추출",
                idx + 1, len(chunks), len(concepts),
            )
        except Exception:
            logger.warning(
                "청크 %d LLM 호출 실패", idx + 1, exc_info=True,
            )
            chunk_results.append([])

    # Merge across chunks
    merged = _merge_chunk_concepts(chunk_results)

    # Save cache
    if not no_cache:
        _save_v2_cache(textbook_path, raw_text, merged)

    return merged


def extract_concepts_llm(
    textbook_path: str,
    summary_path: str | None = None,
    model: str | None = None,
    chapter_name: str | None = None,
    no_cache: bool = False,
    force_chunk: bool | None = None,
) -> list[DomainConcept]:
    """Extract domain concepts from textbook using LLM.

    Steps:
        1. Read and preprocess textbook text (body-only).
        2. Check v2 cache (skip LLM if hit, unless no_cache).
        3. If body > 12000 chars (or force_chunk=True), route to chunked.
        4. Build extraction prompt with optional structure guide.
        5. Call LLM and parse YAML response.
        6. Save cache.

    Args:
        textbook_path: Path to textbook chapter text file.
        summary_path: Optional chapter summary Markdown path.
        model: Optional LLM model ID override.
        chapter_name: Override chapter name (default: from filename stem).
        no_cache: If True, skip cache lookup and force LLM call.
        force_chunk: If True, force chunking. If False, force single call.
            If None (default), auto-detect based on text length > 12000.

    Returns:
        List of DomainConcept. Empty list on LLM/parse failure.
    """
    from forma.textbook_preprocessor import prepare_textbook_for_llm

    path = Path(textbook_path)
    if chapter_name is None:
        chapter_name = path.stem

    # Check cache first (unless no_cache)
    if not no_cache:
        cached = _load_v2_cache(textbook_path)
        if cached is not None:
            logger.info("v2 캐시에서 개념 로드: %s", textbook_path)
            return cached

    # Read and preprocess
    raw_text = path.read_text(encoding="utf-8")
    cleaned_body, structure_guide = prepare_textbook_for_llm(
        raw_text, summary_path=summary_path,
    )

    if not cleaned_body.strip():
        logger.warning("본문이 비어 있음: %s", textbook_path)
        return []

    # T021: Route to chunked if body > 12000 chars
    should_chunk = force_chunk if force_chunk is not None else (len(cleaned_body) > 12000)
    if should_chunk:
        logger.info("본문 %d자 → 청크 추출 모드", len(cleaned_body))
        return extract_concepts_llm_chunked(
            textbook_path=textbook_path,
            summary_path=summary_path,
            model=model,
            chapter_name=chapter_name,
            no_cache=no_cache,
        )

    # Build prompt and call LLM
    prompt = build_extraction_prompt(cleaned_body, structure_guide)
    provider = create_provider(model=model)

    try:
        response = provider.generate(
            prompt=prompt,
            max_tokens=16384,
            temperature=0.0,
            system_instruction=_SYSTEM_INSTRUCTION,
        )
    except Exception:
        logger.warning("LLM 호출 실패: %s", textbook_path, exc_info=True)
        return []

    # Parse response
    concepts = _parse_llm_concepts(response, chapter_name)

    # Save cache
    if not no_cache:
        _save_v2_cache(textbook_path, raw_text, concepts)

    return concepts


def extract_multi_chapter_llm(
    textbook_paths: list[str],
    summary_paths: list[str] | None = None,
    model: str | None = None,
    no_cache: bool = False,
) -> dict[str, list[DomainConcept]]:
    """Extract concepts from multiple textbook chapters using LLM.

    Args:
        textbook_paths: List of textbook chapter file paths.
        summary_paths: Optional list of summary Markdown paths
            (matched by index with textbook_paths).
        model: Optional LLM model ID override.
        no_cache: If True, skip cache and force LLM calls.

    Returns:
        Dict mapping chapter name to list of DomainConcept.
    """
    result: dict[str, list[DomainConcept]] = {}

    for i, path_str in enumerate(textbook_paths):
        chapter_name = Path(path_str).stem

        summary_path = None
        if summary_paths and i < len(summary_paths):
            summary_path = summary_paths[i]

        concepts = extract_concepts_llm(
            textbook_path=path_str,
            summary_path=summary_path,
            model=model,
            chapter_name=chapter_name,
            no_cache=no_cache,
        )

        result[chapter_name] = concepts

    return result


# ================================================================
# v3: Topic Hierarchy (T009)
# ================================================================


@dataclass
class SubTopic:
    """A sub-topic (### level) within a major topic.

    Attributes:
        name: Sub-topic name (### header text).
        sections: Section names (#### headers) under this sub-topic.
    """

    name: str
    sections: list[str] = field(default_factory=list)


@dataclass
class MajorTopic:
    """A major topic (## level) in the textbook hierarchy.

    Attributes:
        name: Major topic name (## header text).
        sub_topics: Sub-topics (### level) under this major topic.
    """

    name: str
    sub_topics: list[SubTopic] = field(default_factory=list)


@dataclass
class TopicHierarchy:
    """Parsed hierarchy from Summary_KR.md (## / ### / ####).

    Attributes:
        major_topics: Ordered list of major topics.
        section_to_major: Mapping from section name to major topic name.
        section_to_sub: Mapping from section name to sub-topic name.
    """

    major_topics: list[MajorTopic] = field(default_factory=list)
    section_to_major: dict[str, str] = field(default_factory=dict)
    section_to_sub: dict[str, str] = field(default_factory=dict)


def _fuzzy_section_match(section: str, candidates: list[str]) -> str | None:
    """Find the best fuzzy match for a section name among candidates.

    Uses substring containment in both directions: checks if the section
    name is contained in a candidate or vice versa.

    Args:
        section: Section name to match.
        candidates: List of candidate names.

    Returns:
        Best matching candidate, or None if no match found.
    """
    section_clean = section.strip()
    if not section_clean:
        return None

    # Exact match first
    for c in candidates:
        if c == section_clean:
            return c

    # Substring match (section in candidate or candidate in section)
    for c in candidates:
        if section_clean in c or c in section_clean:
            return c

    return None


def parse_summary_hierarchy(summary_path: str) -> TopicHierarchy:
    """Parse Summary_KR.md into a TopicHierarchy.

    Reads the Markdown file and extracts ## (major topic), ### (sub-topic),
    and #### (section) headers to build a hierarchical structure. Also
    populates section_to_major and section_to_sub mappings using fuzzy
    substring matching for section names.

    Args:
        summary_path: Path to the Summary_KR.md file.

    Returns:
        TopicHierarchy with parsed major topics, sub-topics, and sections.

    Raises:
        FileNotFoundError: If the summary file does not exist.
    """
    path = Path(summary_path)
    text = path.read_text(encoding="utf-8")

    major_topics: list[MajorTopic] = []
    section_to_major: dict[str, str] = {}
    section_to_sub: dict[str, str] = {}

    current_major: MajorTopic | None = None
    current_sub: SubTopic | None = None

    for line in text.split("\n"):
        stripped = line.strip()

        # #### section header (must check before ### and ##)
        if stripped.startswith("#### "):
            section_name = stripped[5:].strip()
            if section_name and current_sub is not None:
                current_sub.sections.append(section_name)
                if current_major is not None:
                    section_to_major[section_name] = current_major.name
                section_to_sub[section_name] = current_sub.name

        # ### sub-topic header (must check before ##)
        elif stripped.startswith("### "):
            sub_name = stripped[4:].strip()
            if sub_name:
                current_sub = SubTopic(name=sub_name)
                if current_major is not None:
                    current_major.sub_topics.append(current_sub)
                    section_to_major[sub_name] = current_major.name
                section_to_sub[sub_name] = sub_name

        # ## major topic header
        elif stripped.startswith("## "):
            major_name = stripped[3:].strip()
            if major_name:
                current_major = MajorTopic(name=major_name)
                major_topics.append(current_major)
                current_sub = None
                section_to_major[major_name] = major_name

    return TopicHierarchy(
        major_topics=major_topics,
        section_to_major=section_to_major,
        section_to_sub=section_to_sub,
    )


# ================================================================
# v3: Chunked Extraction (T017-T022)
# ================================================================


@dataclass
class TextbookChunk:
    """A chunk of textbook text split by section boundaries.

    Attributes:
        chapter: Chapter identifier.
        section_path: Hierarchy path (e.g. ["피부의 구조", "표피"]).
        major_topic: Major topic name from hierarchy.
        sub_topic: Sub-topic name from hierarchy.
        text: Chunk text content.
        char_count: Character count of text.
    """

    chapter: str
    section_path: list[str]
    major_topic: str
    sub_topic: str
    text: str
    char_count: int


def _split_at_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split text at paragraph boundaries (double newlines).

    Args:
        text: Text to split.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks, each <= max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para.strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If a single paragraph exceeds max_chars, include it anyway
            if len(para.strip()) > max_chars:
                chunks.append(para.strip())
                current = ""
            else:
                current = para.strip()

    if current:
        chunks.append(current)

    return chunks if chunks else [text]


def chunk_textbook_by_sections(
    text: str,
    summary_path: str | None = None,
    max_chars: int = 12000,
) -> list[TextbookChunk]:
    """Split textbook text into chunks by section boundaries.

    If summary_path is provided, parses headers with
    ``parse_summary_hierarchy()`` and splits text at ### boundaries.
    Each ### section becomes one chunk. If a chunk exceeds max_chars,
    it is further split at paragraph boundaries (double newline).

    If no summary_path is given, splits at paragraph boundaries
    with max_chars limit.

    If text <= max_chars, returns a single chunk.

    Args:
        text: Full textbook text.
        summary_path: Optional path to Summary_KR.md for hierarchy.
        max_chars: Maximum characters per chunk. Defaults to 12000.

    Returns:
        List of TextbookChunk instances.
    """
    if not text.strip():
        return []

    # Parse hierarchy if available
    hierarchy: TopicHierarchy | None = None
    if summary_path:
        hierarchy = parse_summary_hierarchy(summary_path)

    # Check for ### headers
    h3_pattern = re.compile(r"^###\s+(.+)$", re.MULTILINE)
    has_h3_headers = bool(h3_pattern.search(text))

    # If text fits in a single chunk AND no header-based splitting needed
    if len(text) <= max_chars and not has_h3_headers:
        major_topic = ""
        sub_topic = ""
        section_path_ret: list[str] = []

        if hierarchy and hierarchy.major_topics:
            major_topic = hierarchy.major_topics[0].name
            section_path_ret.append(major_topic)
            if hierarchy.major_topics[0].sub_topics:
                sub_topic = hierarchy.major_topics[0].sub_topics[0].name
                section_path_ret.append(sub_topic)

        return [TextbookChunk(
            chapter="",
            section_path=section_path_ret,
            major_topic=major_topic,
            sub_topic=sub_topic,
            text=text.strip(),
            char_count=len(text.strip()),
        )]

    # Split at ### headers
    h3_matches = list(h3_pattern.finditer(text))

    if h3_matches:
        sections: list[tuple[str, str]] = []  # (header_name, section_text)
        for idx, match in enumerate(h3_matches):
            header_name = match.group(1).strip()
            start = match.end()
            end = h3_matches[idx + 1].start() if idx + 1 < len(h3_matches) else len(text)
            section_text = text[start:end].strip()
            sections.append((header_name, section_text))

        chunks: list[TextbookChunk] = []
        for header_name, section_text in sections:
            # Determine major_topic and sub_topic from hierarchy
            major_topic = ""
            sub_topic = header_name
            section_path_list: list[str] = []

            if hierarchy:
                # Look up in hierarchy mappings
                matched_major = hierarchy.section_to_major.get(header_name, "")
                if matched_major:
                    major_topic = matched_major
                    section_path_list.append(major_topic)
                section_path_list.append(header_name)
            else:
                section_path_list = [header_name]

            # If section exceeds max_chars, split at paragraphs
            if len(section_text) > max_chars:
                sub_chunks = _split_at_paragraphs(section_text, max_chars)
                for sc in sub_chunks:
                    chunks.append(TextbookChunk(
                        chapter="",
                        section_path=list(section_path_list),
                        major_topic=major_topic,
                        sub_topic=sub_topic,
                        text=sc,
                        char_count=len(sc),
                    ))
            else:
                chunks.append(TextbookChunk(
                    chapter="",
                    section_path=list(section_path_list),
                    major_topic=major_topic,
                    sub_topic=sub_topic,
                    text=section_text,
                    char_count=len(section_text),
                ))

        return chunks

    # No ### headers: split at paragraph boundaries
    para_chunks = _split_at_paragraphs(text, max_chars)
    return [
        TextbookChunk(
            chapter="",
            section_path=[],
            major_topic="",
            sub_topic="",
            text=pc,
            char_count=len(pc),
        )
        for pc in para_chunks
    ]


_IMPORTANCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def _merge_chunk_concepts(
    chunk_results: list[list[DomainConcept]],
) -> list[DomainConcept]:
    """Merge concepts from multiple chunks with deduplication.

    Dedup rules:
    1. Exact concept name match: keep longer description, higher importance.
    2. Key_term overlap >= min(2, len(shorter_list)): merge (union key_terms,
       keep higher importance, longer description).

    Args:
        chunk_results: List of concept lists, one per chunk.

    Returns:
        Deduplicated list of DomainConcept.
    """
    # Flatten all concepts
    all_concepts: list[DomainConcept] = []
    for chunk in chunk_results:
        all_concepts.extend(chunk)

    if not all_concepts:
        return []

    # Rule 1: exact name dedup
    name_map: dict[str, DomainConcept] = {}
    for concept in all_concepts:
        if concept.concept in name_map:
            existing = name_map[concept.concept]
            # Keep longer description
            if len(concept.description) > len(existing.description):
                merged = DomainConcept(
                    concept=concept.concept,
                    description=concept.description,
                    key_terms=list(set(existing.key_terms) | set(concept.key_terms)),
                    importance=_higher_importance(existing.importance, concept.importance),
                    section=existing.section or concept.section,
                    chapter=existing.chapter or concept.chapter,
                    major_topic=existing.major_topic or concept.major_topic,
                    sub_topic=existing.sub_topic or concept.sub_topic,
                )
                name_map[concept.concept] = merged
            else:
                existing_merged = DomainConcept(
                    concept=existing.concept,
                    description=existing.description,
                    key_terms=list(set(existing.key_terms) | set(concept.key_terms)),
                    importance=_higher_importance(existing.importance, concept.importance),
                    section=existing.section or concept.section,
                    chapter=existing.chapter or concept.chapter,
                    major_topic=existing.major_topic or concept.major_topic,
                    sub_topic=existing.sub_topic or concept.sub_topic,
                )
                name_map[concept.concept] = existing_merged
        else:
            name_map[concept.concept] = concept

    # Rule 2: key_term overlap merge
    concepts_list = list(name_map.values())
    merged_indices: set[int] = set()
    result: list[DomainConcept] = []

    for i in range(len(concepts_list)):
        if i in merged_indices:
            continue
        current = concepts_list[i]
        for j in range(i + 1, len(concepts_list)):
            if j in merged_indices:
                continue
            other = concepts_list[j]
            overlap = set(current.key_terms) & set(other.key_terms)
            shorter_len = min(len(current.key_terms), len(other.key_terms))
            threshold = min(2, shorter_len) if shorter_len > 0 else 0

            if threshold > 0 and len(overlap) >= threshold:
                # Merge other into current
                merged_indices.add(j)
                current = DomainConcept(
                    concept=current.concept if len(current.description) >= len(other.description) else other.concept,
                    description=current.description if len(current.description) >= len(other.description) else other.description,
                    key_terms=list(set(current.key_terms) | set(other.key_terms)),
                    importance=_higher_importance(current.importance, other.importance),
                    section=current.section or other.section,
                    chapter=current.chapter or other.chapter,
                    major_topic=current.major_topic or other.major_topic,
                    sub_topic=current.sub_topic or other.sub_topic,
                )

        result.append(current)

    return result


def _higher_importance(a: str, b: str) -> str:
    """Return the higher importance level.

    Args:
        a: First importance level.
        b: Second importance level.

    Returns:
        The higher of the two importance levels.
    """
    return a if _IMPORTANCE_ORDER.get(a, 0) >= _IMPORTANCE_ORDER.get(b, 0) else b
