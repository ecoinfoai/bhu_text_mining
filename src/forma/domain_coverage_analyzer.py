"""Domain knowledge coverage analysis and 4-state classification.

Computes per-concept lecture emphasis, classifies concepts into
covered/gap/skipped/extra states based on teaching scope, and
aggregates coverage metrics.
"""

from __future__ import annotations

import enum
import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from forma.domain_concept_extractor import TextbookConcept

logger = logging.getLogger(__name__)

__all__ = [
    "TeachingScope",
    "parse_teaching_scope",
    "parse_scope_string",
    "ConceptEmphasis",
    "ConceptState",
    "ClassifiedConcept",
    "ExtraConcept",
    "CoverageResult",
    "compute_concept_emphasis",
    "classify_concepts",
    "detect_extra_concepts",
    "build_coverage_result",
    "save_coverage_yaml",
    "load_coverage_yaml",
]

GAP_THRESHOLD: float = 0.05


# ----------------------------------------------------------------
# TeachingScope (T022)
# ----------------------------------------------------------------


@dataclass
class TeachingScope:
    """Per-week teaching scope defining lesson boundaries.

    Attributes:
        chapters: Chapter identifiers included this week.
        scope_rules: Mapping of chapter to included keyword list.
            Empty list means full chapter is in scope.
    """

    chapters: list[str]
    scope_rules: dict[str, list[str]] = field(default_factory=dict)

    def is_in_scope(self, concept: TextbookConcept) -> bool:
        """Check whether a textbook concept is within teaching scope.

        Decision logic (FR-005, FR-014):
        1. If concept's chapter is not in chapters → False
        2. If chapter has no scope_rules entry → True (full chapter)
        3. If chapter has scope_rules → True only if any keyword
           is a substring of concept.name_ko

        Args:
            concept: TextbookConcept to check.

        Returns:
            True if concept is within this week's teaching scope.
        """
        if concept.chapter not in self.chapters:
            return False
        keywords = self.scope_rules.get(concept.chapter, [])
        if not keywords:
            return True
        return any(kw in concept.name_ko for kw in keywords)


def parse_teaching_scope(week_config_dict: dict) -> TeachingScope:
    """Parse TeachingScope from a week.yaml dict.

    Expects structure::

        textbook:
          chapters: ["1장", "2장"]
          scope:
            "2장":
              include_only: ["확산", "능동수송"]

    Args:
        week_config_dict: Parsed YAML dict from week.yaml.

    Returns:
        TeachingScope instance.
    """
    textbook = week_config_dict.get("textbook", {})
    if not isinstance(textbook, dict):
        return TeachingScope(chapters=[], scope_rules={})

    chapters = textbook.get("chapters", [])
    if not isinstance(chapters, list):
        chapters = []

    scope_rules: dict[str, list[str]] = {}
    scope_section = textbook.get("scope", {})
    if isinstance(scope_section, dict):
        for chapter_key, rules in scope_section.items():
            if isinstance(rules, dict):
                include_only = rules.get("include_only", [])
                if isinstance(include_only, list):
                    scope_rules[chapter_key] = include_only

    return TeachingScope(chapters=chapters, scope_rules=scope_rules)


def parse_scope_string(scope_str: str) -> dict[str, list[str]]:
    """Parse CLI --scope override string.

    Format: ``"2장:확산,능동수송;3장:"`` (semicolon-separated chapters,
    colon separates chapter from keywords, comma separates keywords).
    Empty keyword list means full chapter.

    Args:
        scope_str: Scope override string.

    Returns:
        Dict mapping chapter to keyword list.
    """
    result: dict[str, list[str]] = {}
    for part in scope_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            chapter, keywords_str = part.split(":", 1)
            chapter = chapter.strip()
            keywords = [
                kw.strip() for kw in keywords_str.split(",") if kw.strip()
            ]
            result[chapter] = keywords
        else:
            result[part.strip()] = []
    return result


# ----------------------------------------------------------------
# ConceptEmphasis (T023)
# ----------------------------------------------------------------


@dataclass
class ConceptEmphasis:
    """Per-concept, per-section lecture emphasis measurement.

    Attributes:
        concept_name: Korean concept name.
        chapter: Source chapter.
        section_scores: Mapping section_id → emphasis score.
        mean_score: Average across sections.
        std_score: Standard deviation across sections.
    """

    concept_name: str
    chapter: str
    section_scores: dict[str, float]
    mean_score: float
    std_score: float


def _infer_section_from_filename(filename: str) -> str:
    """Infer section ID from transcript filename.

    Patterns matched:
    - ``"1A_2주차_1차시.txt"`` → ``"A"``
    - ``"A_w2_s1.txt"`` → ``"A"``
    - ``"sectionB_week3.txt"`` → ``"B"``

    Falls back to the full stem if no pattern matches.

    Args:
        filename: Transcript filename (not full path).

    Returns:
        Section identifier string.
    """
    stem = Path(filename).stem

    # Pattern: "1A_2주차" → section "A"
    m = re.match(r"\d([A-Za-z])_", stem)
    if m:
        return m.group(1).upper()

    # Pattern: "A_" at start
    m = re.match(r"([A-Za-z])_", stem)
    if m:
        return m.group(1).upper()

    # Pattern: "section" prefix
    m = re.match(r"section\s*([A-Za-z])", stem, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return stem


def compute_concept_emphasis(
    transcript_paths: list[str],
    concepts: list[TextbookConcept],
    threshold: float = 0.65,
) -> list[ConceptEmphasis]:
    """Compute per-concept emphasis across lecture transcripts.

    For each transcript file:
    1. Determine section from filename
    2. Load and split into sentences
    3. Compute emphasis scores via embedding similarity

    Multiple sessions for the same section are merged (averaged).

    Args:
        transcript_paths: Paths to lecture transcript files.
        concepts: Textbook concepts to score.
        threshold: Similarity threshold for emphasis scoring.

    Returns:
        List of ConceptEmphasis, one per concept.
    """
    import kss

    from forma.emphasis_map import compute_emphasis_map

    if not concepts or not transcript_paths:
        return []

    concept_names = [c.name_ko for c in concepts]

    # Collect per-section, per-session scores
    # section → list of {concept_name: score}
    section_sessions: dict[str, list[dict[str, float]]] = {}

    for path_str in transcript_paths:
        path = Path(path_str)
        if not path.exists():
            logger.warning("녹취 파일 없음, 건너뜀: %s", path_str)
            continue

        section = _infer_section_from_filename(path.name)

        # Load and preprocess
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="euc-kr")

        # Split into sentences using kss
        try:
            sentences = kss.split_sentences(text)
        except Exception:
            # Fallback: split on sentence-ending punctuation
            sentences = re.split(r"[.!?]\s*", text)
            sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            continue

        # Compute emphasis
        emphasis = compute_emphasis_map(
            sentences=sentences,
            concepts=concept_names,
            threshold=threshold,
        )

        if section not in section_sessions:
            section_sessions[section] = []
        section_sessions[section].append(emphasis.concept_scores)

    # Merge sessions per section (average)
    section_merged: dict[str, dict[str, float]] = {}
    for section, sessions in section_sessions.items():
        merged: dict[str, float] = {}
        for name in concept_names:
            scores = [s.get(name, 0.0) for s in sessions]
            merged[name] = sum(scores) / len(scores) if scores else 0.0
        section_merged[section] = merged

    # Build ConceptEmphasis list
    result: list[ConceptEmphasis] = []
    for concept in concepts:
        section_scores: dict[str, float] = {}
        for section, scores in section_merged.items():
            section_scores[section] = scores.get(concept.name_ko, 0.0)

        values = list(section_scores.values())
        mean_score = statistics.mean(values) if values else 0.0
        std_score = statistics.stdev(values) if len(values) >= 2 else 0.0

        result.append(ConceptEmphasis(
            concept_name=concept.name_ko,
            chapter=concept.chapter,
            section_scores=section_scores,
            mean_score=mean_score,
            std_score=std_score,
        ))

    return result


# ----------------------------------------------------------------
# ConceptState + Classification (T024)
# ----------------------------------------------------------------


class ConceptState(enum.Enum):
    """4-state classification for a textbook concept."""

    COVERED = "다룸"
    GAP = "누락 위험"
    SKIPPED = "의도적 생략"
    EXTRA = "추가 설명"


@dataclass
class ClassifiedConcept:
    """A textbook concept with its state and emphasis data.

    Attributes:
        concept: The underlying TextbookConcept.
        state: 4-state classification.
        emphasis: Emphasis data, None for SKIPPED without measurement.
        in_scope: Whether concept is within teaching scope.
    """

    concept: TextbookConcept
    state: ConceptState
    emphasis: ConceptEmphasis | None
    in_scope: bool


def classify_concepts(
    concepts: list[TextbookConcept],
    emphasis_list: list[ConceptEmphasis],
    scope: TeachingScope,
    gap_threshold: float = GAP_THRESHOLD,
) -> list[ClassifiedConcept]:
    """Classify each textbook concept into a 4-state classification.

    Decision tree per data-model.md:
    1. chapter not in scope → SKIPPED
    2. concept not in scope (include_only) → SKIPPED
    3. in scope + mean emphasis >= threshold → COVERED
    4. in scope + mean emphasis < threshold → GAP

    Args:
        concepts: All textbook concepts.
        emphasis_list: Emphasis data for each concept.
        scope: Teaching scope configuration.
        gap_threshold: Below this emphasis score → GAP.

    Returns:
        List of ClassifiedConcept in same order as input concepts.
    """
    emphasis_map: dict[str, ConceptEmphasis] = {
        e.concept_name: e for e in emphasis_list
    }

    result: list[ClassifiedConcept] = []
    for concept in concepts:
        in_scope = scope.is_in_scope(concept)
        emphasis = emphasis_map.get(concept.name_ko)

        if not in_scope:
            state = ConceptState.SKIPPED
        elif emphasis is not None and emphasis.mean_score >= gap_threshold:
            state = ConceptState.COVERED
        else:
            state = ConceptState.GAP

        result.append(ClassifiedConcept(
            concept=concept,
            state=state,
            emphasis=emphasis,
            in_scope=in_scope,
        ))

    return result


# ----------------------------------------------------------------
# ExtraConcept detection (T025)
# ----------------------------------------------------------------


@dataclass
class ExtraConcept:
    """A concept found in lecture but not in textbook.

    Attributes:
        name: Term found in lecture.
        section_mentions: Mapping section → mention count.
        example_sentence: Representative lecture sentence.
    """

    name: str
    section_mentions: dict[str, int]
    example_sentence: str


def detect_extra_concepts(
    transcript_paths: list[str],
    concepts: list[TextbookConcept],
    min_mentions: int = 3,
) -> list[ExtraConcept]:
    """Detect concepts mentioned in lectures but not in textbook.

    Extracts nouns from transcripts using KoNLPy Okt, then filters
    out words matching any textbook concept name. Words appearing
    at least ``min_mentions`` times are returned.

    Args:
        transcript_paths: Paths to lecture transcript files.
        concepts: Known textbook concepts.
        min_mentions: Minimum total mention count.

    Returns:
        List of ExtraConcept sorted by total mentions descending.
    """
    from collections import Counter

    from konlpy.tag import Okt

    concept_names = {c.name_ko for c in concepts}

    # Stopwords for extra concept detection
    stopwords = frozenset({
        "것", "수", "때", "등", "중", "위", "및", "또는", "이", "그",
        "저", "여기", "거기", "이것", "그것", "아까", "지금", "오늘",
        "다음", "부분", "경우", "정도", "이상", "이하",
    })

    okt = Okt()
    section_nouns: dict[str, Counter] = {}  # section → word → count
    section_sentences: dict[str, dict[str, str]] = {}  # section → word → sentence

    for path_str in transcript_paths:
        path = Path(path_str)
        if not path.exists():
            continue

        section = _infer_section_from_filename(path.name)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="euc-kr")

        nouns = okt.nouns(text)
        if section not in section_nouns:
            section_nouns[section] = Counter()
            section_sentences[section] = {}
        section_nouns[section].update(nouns)

        # Find example sentences
        raw_sentences = re.split(r"[.!?]\s*", text)
        for noun in set(nouns):
            if noun not in section_sentences[section]:
                for sent in raw_sentences:
                    if noun in sent and len(sent.strip()) > 5:
                        section_sentences[section][noun] = sent.strip()
                        break

    # Aggregate across sections
    all_words: set[str] = set()
    for counter in section_nouns.values():
        all_words.update(counter.keys())

    extras: list[ExtraConcept] = []
    for word in all_words:
        if word in concept_names:
            continue
        if word in stopwords:
            continue
        if len(word) <= 1:
            continue

        mentions: dict[str, int] = {}
        total = 0
        example = ""
        for section, counter in section_nouns.items():
            count = counter.get(word, 0)
            if count > 0:
                mentions[section] = count
                total += count
                if not example:
                    example = section_sentences.get(section, {}).get(word, "")

        if total >= min_mentions:
            extras.append(ExtraConcept(
                name=word,
                section_mentions=mentions,
                example_sentence=example,
            ))

    extras.sort(key=lambda e: sum(e.section_mentions.values()), reverse=True)
    return extras


# ----------------------------------------------------------------
# CoverageResult (T026)
# ----------------------------------------------------------------


@dataclass
class CoverageResult:
    """Aggregated coverage analysis output.

    Attributes:
        week: Week number.
        chapters: Chapters analyzed.
        total_textbook_concepts: All concepts from specified chapters.
        in_scope_count: Concepts within teaching scope.
        skipped_count: Intentionally excluded concepts.
        covered_count: In-scope AND mentioned in lecture.
        gap_count: In-scope AND NOT mentioned.
        extra_count: Not in textbook, mentioned in lecture.
        effective_coverage_rate: covered / in_scope (0 if in_scope=0).
        per_section_coverage: {section: coverage_rate}.
        classified_concepts: Full concept list with states.
        extra_concepts: Concepts found in lecture but not textbook.
        emphasis_bias_correlation: Spearman rho (textbook freq vs emphasis).
        section_variance_top10: Top 10 concepts by cross-section std.
        assessment_correlation: Spearman rho (emphasis vs mastery), optional.
    """

    week: int
    chapters: list[str]
    total_textbook_concepts: int
    in_scope_count: int
    skipped_count: int
    covered_count: int
    gap_count: int
    extra_count: int
    effective_coverage_rate: float
    per_section_coverage: dict[str, float]
    classified_concepts: list[ClassifiedConcept]
    extra_concepts: list[ExtraConcept] = field(default_factory=list)
    emphasis_bias_correlation: float | None = None
    section_variance_top10: list[tuple[str, float]] = field(
        default_factory=list,
    )
    assessment_correlation: float | None = None


def build_coverage_result(
    classified: list[ClassifiedConcept],
    extras: list[ExtraConcept],
    week: int = 0,
    chapters: list[str] | None = None,
) -> CoverageResult:
    """Build CoverageResult from classified concepts and extras.

    Computes summary counts, effective coverage rate, per-section
    coverage, emphasis bias correlation, and section variance top 10.

    Args:
        classified: Classified concept list.
        extras: Extra concept list.
        week: Week number.
        chapters: Chapters analyzed.

    Returns:
        CoverageResult with all aggregated metrics.
    """
    if chapters is None:
        chapters = sorted({c.concept.chapter for c in classified})

    total = len(classified)
    in_scope = [c for c in classified if c.in_scope]
    in_scope_count = len(in_scope)
    covered = [c for c in classified if c.state == ConceptState.COVERED]
    covered_count = len(covered)
    gap = [c for c in classified if c.state == ConceptState.GAP]
    gap_count = len(gap)
    skipped = [c for c in classified if c.state == ConceptState.SKIPPED]
    skipped_count = len(skipped)
    extra_count = len(extras)

    effective_coverage_rate = (
        covered_count / in_scope_count if in_scope_count > 0 else 0.0
    )

    # Per-section coverage
    per_section_coverage = _compute_per_section_coverage(in_scope)

    # Emphasis bias correlation
    emphasis_bias = _compute_emphasis_bias(in_scope)

    # Section variance top 10
    variance_top10 = _compute_section_variance_top10(classified)

    return CoverageResult(
        week=week,
        chapters=chapters,
        total_textbook_concepts=total,
        in_scope_count=in_scope_count,
        skipped_count=skipped_count,
        covered_count=covered_count,
        gap_count=gap_count,
        extra_count=extra_count,
        effective_coverage_rate=effective_coverage_rate,
        per_section_coverage=per_section_coverage,
        classified_concepts=classified,
        extra_concepts=extras,
        emphasis_bias_correlation=emphasis_bias,
        section_variance_top10=variance_top10,
    )


def _compute_per_section_coverage(
    in_scope: list[ClassifiedConcept],
) -> dict[str, float]:
    """Compute coverage rate per section.

    For each section, coverage = concepts with section emphasis >= GAP_THRESHOLD
    divided by total in-scope concepts.

    Args:
        in_scope: In-scope classified concepts.

    Returns:
        Dict mapping section to coverage rate.
    """
    if not in_scope:
        return {}

    # Collect all sections
    sections: set[str] = set()
    for c in in_scope:
        if c.emphasis is not None:
            sections.update(c.emphasis.section_scores.keys())

    result: dict[str, float] = {}
    for section in sorted(sections):
        covered_in_section = 0
        for c in in_scope:
            if c.emphasis is not None:
                score = c.emphasis.section_scores.get(section, 0.0)
                if score >= GAP_THRESHOLD:
                    covered_in_section += 1
        result[section] = covered_in_section / len(in_scope)

    return result


def _compute_emphasis_bias(
    in_scope: list[ClassifiedConcept],
) -> float | None:
    """Compute Spearman correlation between textbook frequency and emphasis.

    Args:
        in_scope: In-scope classified concepts with emphasis data.

    Returns:
        Spearman rho, or None if insufficient data.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        logger.warning("scipy 미설치, 강조 편향 상관 분석 건너뜀")
        return None

    pairs = []
    for c in in_scope:
        if c.emphasis is not None:
            pairs.append((c.concept.frequency, c.emphasis.mean_score))

    if len(pairs) < 3:
        return None

    freqs, scores = zip(*pairs)
    rho, _ = spearmanr(freqs, scores)
    return float(rho)


def _compute_section_variance_top10(
    classified: list[ClassifiedConcept],
) -> list[tuple[str, float]]:
    """Compute top 10 concepts by cross-section emphasis variance.

    Args:
        classified: All classified concepts.

    Returns:
        List of (concept_name, std) sorted by std descending, max 10.
    """
    variance_data: list[tuple[str, float]] = []
    for c in classified:
        if c.emphasis is not None and c.emphasis.std_score > 0:
            variance_data.append(
                (c.concept.name_ko, c.emphasis.std_score)
            )

    variance_data.sort(key=lambda x: x[1], reverse=True)
    return variance_data[:10]


# ----------------------------------------------------------------
# YAML I/O (T027)
# ----------------------------------------------------------------


def save_coverage_yaml(
    result: CoverageResult,
    output_path: str,
) -> None:
    """Save CoverageResult to YAML following contract format.

    Args:
        result: Coverage analysis result.
        output_path: Output YAML path.
    """
    data: dict = {
        "week": result.week,
        "chapters": result.chapters,
        "summary": {
            "total_concepts": result.total_textbook_concepts,
            "in_scope": result.in_scope_count,
            "covered": result.covered_count,
            "gap": result.gap_count,
            "skipped": result.skipped_count,
            "extra": result.extra_count,
            "effective_coverage": round(result.effective_coverage_rate, 4),
            "per_section": {
                k: round(v, 4)
                for k, v in result.per_section_coverage.items()
            },
        },
        "concepts": [],
        "extra_concepts": [],
    }

    if result.emphasis_bias_correlation is not None:
        data["summary"]["emphasis_bias_rho"] = round(
            result.emphasis_bias_correlation, 4,
        )

    if result.section_variance_top10:
        data["summary"]["section_variance_top10"] = [
            {"name": name, "std": round(std, 4)}
            for name, std in result.section_variance_top10
        ]

    for cc in result.classified_concepts:
        entry: dict = {
            "name": cc.concept.name_ko,
            "chapter": cc.concept.chapter,
            "state": cc.state.name.lower(),
            "in_scope": cc.in_scope,
        }
        if cc.emphasis is not None:
            entry["emphasis"] = {
                k: round(v, 4)
                for k, v in cc.emphasis.section_scores.items()
            }
            entry["mean_emphasis"] = round(cc.emphasis.mean_score, 4)
            entry["std_emphasis"] = round(cc.emphasis.std_score, 4)
        if cc.concept.name_en:
            entry["name_en"] = cc.concept.name_en
        data["concepts"].append(entry)

    for extra in result.extra_concepts:
        data["extra_concepts"].append({
            "name": extra.name,
            "section_mentions": extra.section_mentions,
            "example": extra.example_sentence,
        })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def load_coverage_yaml(path: str) -> CoverageResult:
    """Load CoverageResult from a YAML file.

    Args:
        path: Path to coverage YAML.

    Returns:
        Reconstructed CoverageResult.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If YAML structure is invalid.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"올바르지 않은 커버리지 YAML 형식: {path}")

    summary = data.get("summary", {})

    # Reconstruct classified concepts
    classified: list[ClassifiedConcept] = []
    for entry in data.get("concepts", []):
        concept = TextbookConcept(
            name_ko=entry["name"],
            name_en=entry.get("name_en"),
            chapter=entry["chapter"],
            frequency=0,
            context_sentence="",
            is_bilingual=False,
        )

        state_str = entry.get("state", "covered").upper()
        try:
            state = ConceptState[state_str]
        except KeyError:
            state = ConceptState.COVERED

        emphasis = None
        if "emphasis" in entry:
            section_scores = entry["emphasis"]
            emphasis = ConceptEmphasis(
                concept_name=entry["name"],
                chapter=entry["chapter"],
                section_scores=section_scores,
                mean_score=entry.get("mean_emphasis", 0.0),
                std_score=entry.get("std_emphasis", 0.0),
            )

        classified.append(ClassifiedConcept(
            concept=concept,
            state=state,
            emphasis=emphasis,
            in_scope=entry.get("in_scope", True),
        ))

    # Reconstruct extra concepts
    extras: list[ExtraConcept] = []
    for entry in data.get("extra_concepts", []):
        extras.append(ExtraConcept(
            name=entry["name"],
            section_mentions=entry.get("section_mentions", {}),
            example_sentence=entry.get("example", ""),
        ))

    # Reconstruct variance top 10
    variance_top10: list[tuple[str, float]] = []
    for item in summary.get("section_variance_top10", []):
        variance_top10.append((item["name"], item["std"]))

    return CoverageResult(
        week=data.get("week", 0),
        chapters=data.get("chapters", []),
        total_textbook_concepts=summary.get("total_concepts", len(classified)),
        in_scope_count=summary.get("in_scope", 0),
        skipped_count=summary.get("skipped", 0),
        covered_count=summary.get("covered", 0),
        gap_count=summary.get("gap", 0),
        extra_count=summary.get("extra", 0),
        effective_coverage_rate=summary.get("effective_coverage", 0.0),
        per_section_coverage=summary.get("per_section", {}),
        classified_concepts=classified,
        extra_concepts=extras,
        emphasis_bias_correlation=summary.get("emphasis_bias_rho"),
        section_variance_top10=variance_top10,
    )
