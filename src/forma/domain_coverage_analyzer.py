"""Domain knowledge coverage analysis and 4-state classification.

Computes per-concept lecture emphasis, classifies concepts into
covered/gap/skipped/extra states based on teaching scope, and
aggregates coverage metrics.

v2 additions: LLM-based delivery analysis (DeliveryAnalysis,
DeliveryState), keyword network comparison, and delivery result
aggregation.
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
    # v2 delivery analysis
    "DeliveryState",
    "DeliveryAnalysis",
    "DeliveryResult",
    "build_delivery_prompt",
    "analyze_delivery_llm",
    "v1_fallback_analysis",
    "analyze_delivery_with_fallback",
    "build_delivery_result_v2",
    "save_delivery_yaml",
    "load_delivery_yaml",
    # v2 network comparison
    "KeywordNetwork",
    "build_domain_network",
    "compare_networks",
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


# ================================================================
# v2: LLM-based Delivery Analysis (Phase 4)
# ================================================================


# ----------------------------------------------------------------
# DeliveryState enum (T026)
# ----------------------------------------------------------------


class DeliveryState(enum.Enum):
    """Delivery status classification for a concept in a section."""

    FULLY_DELIVERED = "충분히 설명"
    PARTIALLY_DELIVERED = "부분 전달"
    NOT_DELIVERED = "미전달"
    SKIPPED = "의도적 생략"


# ----------------------------------------------------------------
# DeliveryAnalysis dataclass (T026)
# ----------------------------------------------------------------


@dataclass
class DeliveryAnalysis:
    """Per-concept, per-section delivery assessment.

    Attributes:
        concept: Concept name (matches DomainConcept.concept).
        section_id: Class section (A, B, C, D).
        delivery_status: Delivery status label.
        delivery_quality: Quality score 0.0 ~ 1.0.
        evidence: Transcript excerpt proving the assessment.
        depth: Summary of explanation depth.
        analysis_level: "v2" (LLM) or "v1" (embedding fallback).
    """

    concept: str
    section_id: str
    delivery_status: str
    delivery_quality: float
    evidence: str
    depth: str
    analysis_level: str = "v2"


# ----------------------------------------------------------------
# DeliveryResult dataclass (T031)
# ----------------------------------------------------------------


@dataclass
class DeliveryResult:
    """Aggregated v2 delivery analysis output.

    Attributes:
        week: Week number.
        chapters: Chapters analyzed.
        deliveries: Per-concept, per-section delivery assessments.
        effective_delivery_rate: (fully + partially) / in_scope.
        per_section_rate: {section: delivery_rate}.
    """

    week: int
    chapters: list[str]
    deliveries: list[DeliveryAnalysis]
    effective_delivery_rate: float
    per_section_rate: dict[str, float]


# ----------------------------------------------------------------
# Prompt construction (T027)
# ----------------------------------------------------------------

_DELIVERY_SYSTEM_INSTRUCTION = (
    "당신은 해부생리학 강의 분석 전문가입니다. "
    "교과서 개념이 강의에서 어떻게 전달되었는지 분석해주세요."
)

_DELIVERY_PROMPT_TEMPLATE = """\
아래 교과서 핵심 개념 목록과 강의 녹취를 비교하여, 각 개념이 강의에서 \
어떻게 전달되었는지 분석해주세요.

## 분석 기준
- "충분히 설명": 개념의 메커니즘, 구조, 과정이 상세히 설명됨
- "부분 전달": 용어만 언급되었으나 메커니즘이나 설명이 부족함
- "미전달": 강의에서 전혀 언급되지 않음

## 교과서 핵심 개념
{concepts_section}

## 강의 녹취
{transcript_text}

## 출력 형식 (YAML)
```yaml
deliveries:
  - concept: "개념 이름"
    delivery_status: "충분히 설명"
    delivery_quality: 0.85
    evidence: "녹취 내 근거 문장"
    depth: "메커니즘과 임상 적용까지 설명"
```
"""


def build_delivery_prompt(
    concepts: list[str],
    transcript_text: str,
) -> str:
    """Construct LLM prompt for delivery analysis.

    Args:
        concepts: List of concept names to analyze.
        transcript_text: Full lecture transcript text.

    Returns:
        Formatted prompt string.
    """
    concepts_section = "\n".join(f"- {c}" for c in concepts)
    return _DELIVERY_PROMPT_TEMPLATE.format(
        concepts_section=concepts_section,
        transcript_text=transcript_text,
    )


# ----------------------------------------------------------------
# LLM delivery analysis (T028)
# ----------------------------------------------------------------

_STATUS_MAP: dict[str, DeliveryState] = {
    "충분히 설명": DeliveryState.FULLY_DELIVERED,
    "부분 전달": DeliveryState.PARTIALLY_DELIVERED,
    "미전달": DeliveryState.NOT_DELIVERED,
}


def _parse_delivery_response(
    response_text: str,
    section_id: str,
) -> list[DeliveryAnalysis]:
    """Parse LLM YAML response into DeliveryAnalysis list.

    Args:
        response_text: Raw LLM text response.
        section_id: Section identifier.

    Returns:
        List of DeliveryAnalysis. Empty on parse failure.
    """
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        logger.warning("전달 분석 LLM 응답 YAML 파싱 실패")
        return []

    if not isinstance(data, dict) or "deliveries" not in data:
        logger.warning("LLM 응답에 'deliveries' 키가 없음")
        return []

    results: list[DeliveryAnalysis] = []
    for item in data["deliveries"]:
        if not isinstance(item, dict):
            continue
        concept_name = item.get("concept", "")
        if not concept_name:
            continue

        status = item.get("delivery_status", "미전달")
        quality = float(item.get("delivery_quality", 0.0))
        quality = max(0.0, min(1.0, quality))

        results.append(DeliveryAnalysis(
            concept=concept_name,
            section_id=section_id,
            delivery_status=status,
            delivery_quality=quality,
            evidence=item.get("evidence", ""),
            depth=item.get("depth", ""),
            analysis_level="v2",
        ))

    return results


def analyze_delivery_llm(
    concepts: list[str],
    transcript_path: str,
    section_id: str,
    model: str | None = None,
) -> list[DeliveryAnalysis]:
    """Analyze concept delivery using LLM.

    Args:
        concepts: List of concept names to analyze.
        transcript_path: Path to lecture transcript file.
        section_id: Section identifier (A, B, C, D).
        model: Optional LLM model ID override.

    Returns:
        List of DeliveryAnalysis for the section.

    Raises:
        Exception: If LLM call fails (caller should handle fallback).
    """
    from forma.config import get_llm_config, load_config
    from forma.llm_provider import create_provider

    try:
        cfg = load_config()
        llm_cfg = get_llm_config(cfg)
    except FileNotFoundError:
        llm_cfg = {}

    path = Path(transcript_path)
    try:
        transcript_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        transcript_text = path.read_text(encoding="euc-kr")

    prompt = build_delivery_prompt(concepts, transcript_text)
    provider = create_provider(
        provider=llm_cfg.get("provider", "gemini"),
        api_key=llm_cfg.get("api_key"),
        model=model or llm_cfg.get("model"),
    )

    response = provider.generate(
        prompt=prompt,
        max_tokens=4096,
        temperature=0.0,
        system_instruction=_DELIVERY_SYSTEM_INSTRUCTION,
    )

    return _parse_delivery_response(response, section_id)


# ----------------------------------------------------------------
# v1 fallback analysis (T029)
# ----------------------------------------------------------------

_V1_FULLY_THRESHOLD: float = 0.3
_V1_PARTIAL_THRESHOLD: float = 0.05


def v1_fallback_analysis(
    concepts: list[str],
    transcript_path: str,
    section_id: str,
    threshold: float = 0.65,
) -> list[DeliveryAnalysis]:
    """Fallback delivery analysis using v1 embedding-based emphasis.

    Uses emphasis_map scores with thresholds:
    - >= 0.3 -> FULLY_DELIVERED
    - >= 0.05 -> PARTIALLY_DELIVERED
    - < 0.05 -> NOT_DELIVERED

    Args:
        concepts: List of concept names.
        transcript_path: Path to transcript file.
        section_id: Section identifier.
        threshold: Embedding similarity threshold.

    Returns:
        List of DeliveryAnalysis marked as analysis_level="v1".
    """
    from forma.emphasis_map import compute_emphasis_map

    path = Path(transcript_path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="euc-kr")

    # Split into sentences
    sentences = re.split(r"[.!?]\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences or not concepts:
        return [
            DeliveryAnalysis(
                concept=c,
                section_id=section_id,
                delivery_status=DeliveryState.NOT_DELIVERED.value,
                delivery_quality=0.0,
                evidence="",
                depth="",
                analysis_level="v1",
            )
            for c in concepts
        ]

    emphasis = compute_emphasis_map(
        sentences=sentences,
        concepts=concepts,
        threshold=threshold,
    )

    results: list[DeliveryAnalysis] = []
    for concept_name in concepts:
        score = emphasis.concept_scores.get(concept_name, 0.0)

        if score >= _V1_FULLY_THRESHOLD:
            status = DeliveryState.FULLY_DELIVERED
        elif score >= _V1_PARTIAL_THRESHOLD:
            status = DeliveryState.PARTIALLY_DELIVERED
        else:
            status = DeliveryState.NOT_DELIVERED

        results.append(DeliveryAnalysis(
            concept=concept_name,
            section_id=section_id,
            delivery_status=status.value,
            delivery_quality=score,
            evidence="",
            depth="",
            analysis_level="v1",
        ))

    return results


# ----------------------------------------------------------------
# Delivery with fallback (T030)
# ----------------------------------------------------------------


def analyze_delivery_with_fallback(
    concepts: list[str],
    transcript_path: str,
    section_id: str,
    model: str | None = None,
    threshold: float = 0.65,
) -> list[DeliveryAnalysis]:
    """Analyze delivery via LLM, falling back to v1 on failure.

    Args:
        concepts: Concept names to analyze.
        transcript_path: Path to transcript.
        section_id: Section ID.
        model: LLM model override.
        threshold: Embedding similarity threshold for v1 fallback.

    Returns:
        List of DeliveryAnalysis (v2 or v1 level).
    """
    try:
        return analyze_delivery_llm(concepts, transcript_path, section_id, model)
    except Exception:
        logger.warning(
            "%s반 LLM 전달 분석 실패, v1 fallback 사용",
            section_id,
            exc_info=True,
        )
        return v1_fallback_analysis(
            concepts, transcript_path, section_id, threshold,
        )


# ----------------------------------------------------------------
# DeliveryResult builder (T031)
# ----------------------------------------------------------------


def build_delivery_result_v2(
    deliveries: list[DeliveryAnalysis],
    scope: TeachingScope,
    concepts: list[str],
    week: int = 0,
    chapters: list[str] | None = None,
) -> DeliveryResult:
    """Build aggregated DeliveryResult from delivery analyses.

    effective_delivery_rate = (fully + partially) / in_scope_count
    per_section_rate = per section version of the same.

    Args:
        deliveries: All delivery analyses across sections.
        scope: Teaching scope for in-scope filtering.
        concepts: All concept names.
        week: Week number.
        chapters: Chapters analyzed.

    Returns:
        DeliveryResult with aggregated metrics.
    """
    if chapters is None:
        chapters = scope.chapters

    # Group by section
    section_deliveries: dict[str, list[DeliveryAnalysis]] = {}
    for d in deliveries:
        if d.section_id not in section_deliveries:
            section_deliveries[d.section_id] = []
        section_deliveries[d.section_id].append(d)

    # Filter to non-skipped deliveries only
    non_skipped = [
        d for d in deliveries
        if d.delivery_status != DeliveryState.SKIPPED.value
    ]

    # Overall effective delivery rate
    if non_skipped:
        delivered = sum(
            1 for d in non_skipped
            if d.delivery_status in (
                DeliveryState.FULLY_DELIVERED.value,
                DeliveryState.PARTIALLY_DELIVERED.value,
            )
        )
        effective_rate = delivered / len(non_skipped)
    else:
        effective_rate = 0.0

    # Per-section rate
    per_section_rate: dict[str, float] = {}
    for section_id, sec_deliveries in sorted(section_deliveries.items()):
        sec_non_skipped = [
            d for d in sec_deliveries
            if d.delivery_status != DeliveryState.SKIPPED.value
        ]
        if sec_non_skipped:
            sec_delivered = sum(
                1 for d in sec_non_skipped
                if d.delivery_status in (
                    DeliveryState.FULLY_DELIVERED.value,
                    DeliveryState.PARTIALLY_DELIVERED.value,
                )
            )
            per_section_rate[section_id] = sec_delivered / len(sec_non_skipped)
        else:
            per_section_rate[section_id] = 0.0

    return DeliveryResult(
        week=week,
        chapters=chapters,
        deliveries=deliveries,
        effective_delivery_rate=effective_rate,
        per_section_rate=per_section_rate,
    )


# ----------------------------------------------------------------
# Delivery YAML I/O (T032)
# ----------------------------------------------------------------


def save_delivery_yaml(
    result: DeliveryResult,
    output_path: str,
) -> None:
    """Save DeliveryResult to YAML.

    Args:
        result: Delivery analysis result.
        output_path: Output file path.
    """
    data: dict = {
        "version": "v2",
        "week": result.week,
        "chapters": result.chapters,
        "summary": {
            "effective_delivery_rate": round(
                result.effective_delivery_rate, 4,
            ),
            "per_section_rate": {
                k: round(v, 4)
                for k, v in result.per_section_rate.items()
            },
        },
        "deliveries": [
            {
                "concept": d.concept,
                "section_id": d.section_id,
                "delivery_status": d.delivery_status,
                "delivery_quality": round(d.delivery_quality, 4),
                "evidence": d.evidence,
                "depth": d.depth,
                "analysis_level": d.analysis_level,
            }
            for d in result.deliveries
        ],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def load_delivery_yaml(path: str) -> DeliveryResult:
    """Load DeliveryResult from YAML.

    Args:
        path: Path to delivery YAML file.

    Returns:
        Reconstructed DeliveryResult.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If YAML structure is invalid.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"올바르지 않은 전달분석 YAML 형식: {path}")

    summary = data.get("summary", {})

    deliveries = [
        DeliveryAnalysis(
            concept=d["concept"],
            section_id=d["section_id"],
            delivery_status=d["delivery_status"],
            delivery_quality=d.get("delivery_quality", 0.0),
            evidence=d.get("evidence", ""),
            depth=d.get("depth", ""),
            analysis_level=d.get("analysis_level", "v2"),
        )
        for d in data.get("deliveries", [])
    ]

    return DeliveryResult(
        week=data.get("week", 0),
        chapters=data.get("chapters", []),
        deliveries=deliveries,
        effective_delivery_rate=summary.get("effective_delivery_rate", 0.0),
        per_section_rate=summary.get("per_section_rate", {}),
    )


# ================================================================
# v2: Keyword Network Comparison (Phase 5)
# ================================================================


@dataclass
class KeywordNetwork:
    """Co-occurrence graph of domain key terms.

    Attributes:
        source: "textbook" or section_id (A, B, C, D).
        nodes: Domain terms.
        edges: List of (term1, term2, weight) co-occurrence tuples.
        missing_vs_textbook: Edges in textbook but absent here.
    """

    source: str
    nodes: list[str]
    edges: list[tuple[str, str, float]]
    missing_vs_textbook: list[tuple[str, str]] = field(default_factory=list)


def build_domain_network(
    text: str,
    key_terms: list[str],
    source: str = "textbook",
    window_size: int = 2,
) -> KeywordNetwork:
    """Build keyword co-occurrence network filtered to domain terms.

    Uses network_analysis.extract_keywords (with empty stopwords)
    and network_analysis.create_network, then filters to only
    edges between key_terms.

    Args:
        text: Full text (textbook body or transcript).
        key_terms: Domain-specific terms to include.
        source: Source identifier.
        window_size: Co-occurrence window size.

    Returns:
        KeywordNetwork with filtered nodes and edges.
    """
    from forma.network_analysis import create_network, extract_keywords

    # Extract all keywords (using empty stopwords to keep everything)
    keywords = extract_keywords(text, stopwords=set())

    # Filter to only domain key_terms
    key_terms_set = set(key_terms)
    filtered_keywords = [kw for kw in keywords if kw in key_terms_set]

    if not filtered_keywords:
        return KeywordNetwork(
            source=source,
            nodes=[],
            edges=[],
        )

    # Build network
    graph = create_network(filtered_keywords, window_size=window_size)

    # Extract nodes and edges
    nodes = [n for n in graph.nodes() if n in key_terms_set]
    edges: list[tuple[str, str, float]] = []
    for u, v, data in graph.edges(data=True):
        if u in key_terms_set and v in key_terms_set:
            edges.append((u, v, float(data.get("weight", 1.0))))

    return KeywordNetwork(
        source=source,
        nodes=nodes,
        edges=edges,
    )


def compare_networks(
    textbook_net: KeywordNetwork,
    lecture_net: KeywordNetwork,
) -> list[tuple[str, str]]:
    """Compare textbook and lecture networks to find missing edges.

    Identifies edges present in the textbook network but absent
    in the lecture network.

    Args:
        textbook_net: Textbook keyword network.
        lecture_net: Lecture keyword network.

    Returns:
        List of (term1, term2) edges missing in the lecture.
    """
    lecture_edge_set: set[frozenset[str]] = set()
    for u, v, _w in lecture_net.edges:
        lecture_edge_set.add(frozenset({u, v}))

    missing: list[tuple[str, str]] = []
    for u, v, _w in textbook_net.edges:
        edge_key = frozenset({u, v})
        if edge_key not in lecture_edge_set:
            missing.append((u, v))

    return missing
