"""Single-file STT lecture transcript analysis orchestrator.

Coordinates keyword extraction, network analysis, topic modeling,
concept coverage, emphasis mapping, and triplet extraction into a
unified AnalysisResult. Each stage runs independently so that a
failure in one does not block the others (FR-027).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx
import yaml

from forma.lecture_preprocessor import CleanedTranscript, build_stopwords
from forma.network_analysis import extract_keywords, create_network
from forma.emphasis_map import compute_emphasis_map
from forma.lecture_gap_analysis import compute_lecture_gap

logger = logging.getLogger(__name__)

# Lazy imports for heavy deps
kss: Any = None


def _ensure_kss() -> Any:
    """Lazy-import kss to avoid import-time cost."""
    global kss
    if kss is None:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            import kss as _kss
        kss = _kss
    return kss


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------

@dataclass
class TopicSummary:
    """Summary of a single topic from BERTopic analysis.

    Args:
        topic_id: BERTopic topic index.
        keywords: Top keywords for this topic.
        sentence_count: Number of sentences assigned to this topic.
        representative_sentence: Most representative sentence.
    """

    topic_id: int
    keywords: list[str]
    sentence_count: int
    representative_sentence: str


@dataclass
class ConceptCoverage:
    """Coverage statistics for master concepts in a lecture.

    Args:
        total_concepts: Total number of master concepts.
        covered_concepts: Concepts found in the lecture.
        missed_concepts: Concepts not found in the lecture.
        coverage_ratio: Fraction of concepts covered.
    """

    total_concepts: int
    covered_concepts: list[str]
    missed_concepts: list[str]
    coverage_ratio: float


@dataclass
class AnalysisResult:
    """Complete result of a single lecture transcript analysis.

    Args:
        class_id: Class identifier (e.g. "A").
        week: Week number.
        keyword_frequencies: Mapping of keyword to frequency count.
        top_keywords: Top-N keywords sorted by frequency.
        network_image_path: Path to keyword co-occurrence network PNG.
        topics: BERTopic topic summaries, or None if skipped.
        topic_skipped_reason: Reason topics were skipped (FR-027).
        concept_coverage: Concept coverage statistics, or None.
        emphasis_scores: Per-concept emphasis scores, or None.
        triplets: Extracted knowledge triplets, or None.
        triplet_skipped_reason: Reason triplets were skipped.
        sentence_count: Number of sentences in transcript.
        analysis_timestamp: ISO 8601 timestamp of analysis.
    """

    class_id: str
    week: int
    keyword_frequencies: dict[str, int]
    top_keywords: list[str]
    network_image_path: Path | None
    topics: list[TopicSummary] | None
    topic_skipped_reason: str | None
    concept_coverage: ConceptCoverage | None
    emphasis_scores: dict[str, float] | None
    triplets: list[Any] | None
    triplet_skipped_reason: str | None
    sentence_count: int
    analysis_timestamp: str


# ------------------------------------------------------------------
# Network visualization (headless, saves to file)
# ------------------------------------------------------------------

def _save_network_image(
    G: nx.Graph,
    output_path: Path,
    title: str = "Keyword Network",
) -> None:
    """Save keyword co-occurrence network as PNG image.

    Replicates the visualization logic from network_analysis.visualize_network
    but saves to file instead of calling plt.show().

    Args:
        G: NetworkX graph with 'frequency' node attrs and 'weight' edge attrs.
        output_path: Path to save the PNG file.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42)

    frequencies = nx.get_node_attributes(G, "frequency")
    max_frequency = max(frequencies.values()) if frequencies else 1

    node_sizes = [
        1000 * (freq / max_frequency) for freq in frequencies.values()
    ]

    weights = list(nx.get_edge_attributes(G, "weight").values())
    max_weight = max(weights) if weights else 1
    edge_widths = [3 * (weight / max_weight) for weight in weights]

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.9, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, width=edge_widths, edge_color="gray", alpha=0.5, ax=ax,
    )

    for node, (x, y) in pos.items():
        freq = frequencies.get(node, 1)
        font_size = 10 + 15 * (freq / max_frequency)
        ax.text(
            x, y, s=node, fontsize=font_size,
            horizontalalignment="center", verticalalignment="center",
        )

    ax.set_title(title)
    ax.axis("off")
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------------

_MIN_SENTENCES_FOR_TOPICS = 10


def analyze_transcript(
    cleaned: CleanedTranscript,
    concepts: list[str] | None,
    top_n: int,
    no_triplets: bool,
    provider: Any | None,
) -> AnalysisResult:
    """Analyze a preprocessed lecture transcript.

    Runs keyword extraction, network analysis, topic modeling,
    concept coverage, emphasis mapping, and triplet extraction.
    Each stage is independent; a failure in one logs a warning
    and continues with the others (FR-027).

    Args:
        cleaned: Preprocessed transcript from preprocess_transcript().
        concepts: Optional list of master concepts for coverage analysis.
        top_n: Number of top keywords to include.
        no_triplets: If True, skip triplet extraction.
        provider: LLM provider for triplet extraction. None to skip.

    Returns:
        AnalysisResult with all available analysis data.
    """
    text = cleaned.cleaned_text
    stopwords = build_stopwords()

    # Stage 1: Keyword extraction
    keyword_frequencies: dict[str, int] = {}
    top_keywords: list[str] = []
    try:
        keywords = extract_keywords(text, stopwords)
        keyword_frequencies = dict(Counter(keywords))
        sorted_kw = sorted(
            keyword_frequencies.items(), key=lambda x: x[1], reverse=True,
        )
        top_keywords = [kw for kw, _ in sorted_kw[:top_n]]
    except Exception:
        logger.warning("Keyword extraction failed", exc_info=True)

    # Stage 2: Network analysis
    network_image_path: Path | None = None
    try:
        G = create_network(keywords if keyword_frequencies else [], window_size=2)
        if G.number_of_nodes() > 0:
            # Save to a temp location; caller can move if needed
            import tempfile
            net_path = Path(tempfile.mktemp(suffix=".png", prefix="keyword_net_"))
            _save_network_image(G, net_path)
            network_image_path = net_path
    except Exception:
        logger.warning("Network analysis failed", exc_info=True)

    # Stage 3: Sentence splitting
    sentences: list[str] = []
    try:
        _kss = _ensure_kss()
        sentences = _kss.split_sentences(text)
    except Exception:
        logger.warning("Sentence splitting failed", exc_info=True)

    sentence_count = len(sentences)

    # Stage 4: Topic modeling
    topics: list[TopicSummary] | None = None
    topic_skipped_reason: str | None = None
    if sentence_count < _MIN_SENTENCES_FOR_TOPICS:
        topic_skipped_reason = (
            f"Insufficient sentences ({sentence_count} < {_MIN_SENTENCES_FOR_TOPICS})"
        )
    else:
        try:
            from forma.topic_analysis import (
                configure_bertopic,
                analyze_topics_with_bertopic,
                generate_topic_keywords_table,
            )
            env_config = {
                "umap": {"n_neighbors": 5, "n_components": 2, "random_state": 42},
                "hdbscan": {"min_cluster_size": 3},
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            }
            topic_model = configure_bertopic(env_config)
            topic_ids, _probs = analyze_topics_with_bertopic(topic_model, sentences)
            kw_table = generate_topic_keywords_table(topic_model, sentences, topic_ids)

            topics = []
            topic_counter = Counter(topic_ids)
            for _, row in kw_table.iterrows():
                tid = int(row["Topic"])
                kw_list = [kw.strip() for kw in str(row["Keywords"]).split(",")]
                # Find representative sentence
                rep_sentence = ""
                for i, t in enumerate(topic_ids):
                    if t == tid:
                        rep_sentence = sentences[i]
                        break
                topics.append(TopicSummary(
                    topic_id=tid,
                    keywords=kw_list,
                    sentence_count=topic_counter.get(tid, 0),
                    representative_sentence=rep_sentence,
                ))
        except Exception:
            logger.warning("Topic modeling failed", exc_info=True)
            topic_skipped_reason = "Topic modeling error"

    # Stage 5: Concept coverage and emphasis
    concept_coverage: ConceptCoverage | None = None
    emphasis_scores: dict[str, float] | None = None
    if concepts:
        try:
            emphasis_map = compute_emphasis_map(sentences, concepts)
            emphasis_scores = emphasis_map.concept_scores
        except Exception:
            logger.warning("Emphasis analysis failed", exc_info=True)

        try:
            # Determine which concepts are covered using keyword overlap
            lecture_concept_set = set(top_keywords) if top_keywords else set()
            # Also use emphasis scores to infer coverage
            if emphasis_scores:
                for c, score in emphasis_scores.items():
                    if score > 0.3:
                        lecture_concept_set.add(c)

            gap_report = compute_lecture_gap(
                master_concepts=set(concepts),
                lecture_concepts=lecture_concept_set,
            )
            concept_coverage = ConceptCoverage(
                total_concepts=len(concepts),
                covered_concepts=sorted(gap_report.covered_concepts),
                missed_concepts=sorted(gap_report.missed_concepts),
                coverage_ratio=gap_report.coverage_ratio,
            )
        except Exception:
            logger.warning("Concept coverage analysis failed", exc_info=True)

    # Stage 6: Triplet extraction
    triplets: list[Any] | None = None
    triplet_skipped_reason: str | None = None
    if no_triplets:
        triplet_skipped_reason = "Triplet extraction skipped (--no-triplets)"
    elif provider is None:
        triplet_skipped_reason = "No LLM provider specified"
    else:
        try:
            from forma.lecture_processor import extract_triplets_from_lecture
            raw_triplets = extract_triplets_from_lecture(text, provider)
            triplets = [
                {"subject": t.subject, "relation": t.relation, "object": t.object}
                for t in raw_triplets
            ]
        except Exception:
            logger.warning("Triplet extraction failed", exc_info=True)
            triplet_skipped_reason = "Triplet extraction error"

    return AnalysisResult(
        class_id=cleaned.class_id,
        week=cleaned.week,
        keyword_frequencies=keyword_frequencies,
        top_keywords=top_keywords,
        network_image_path=network_image_path,
        topics=topics,
        topic_skipped_reason=topic_skipped_reason,
        concept_coverage=concept_coverage,
        emphasis_scores=emphasis_scores,
        triplets=triplets,
        triplet_skipped_reason=triplet_skipped_reason,
        sentence_count=sentence_count,
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ------------------------------------------------------------------
# YAML serialization
# ------------------------------------------------------------------

def save_analysis_result(result: AnalysisResult, output_dir: Path) -> Path:
    """Serialize AnalysisResult to a YAML file.

    Args:
        result: The analysis result to save.
        output_dir: Directory to write the YAML file.

    Returns:
        Path to the saved YAML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"analysis_{result.class_id}_w{result.week}.yaml"
    output_path = output_dir / filename

    data = _result_to_dict(result)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    return output_path


def load_analysis_result(path: Path) -> AnalysisResult:
    """Deserialize AnalysisResult from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        AnalysisResult reconstructed from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Analysis result file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return _dict_to_result(data)


def _result_to_dict(result: AnalysisResult) -> dict:
    """Convert AnalysisResult to a plain dict for YAML serialization."""
    data: dict[str, Any] = {
        "class_id": result.class_id,
        "week": result.week,
        "keyword_frequencies": result.keyword_frequencies,
        "top_keywords": result.top_keywords,
        "network_image_path": str(result.network_image_path) if result.network_image_path else None,
        "topic_skipped_reason": result.topic_skipped_reason,
        "emphasis_scores": result.emphasis_scores,
        "triplet_skipped_reason": result.triplet_skipped_reason,
        "sentence_count": result.sentence_count,
        "analysis_timestamp": result.analysis_timestamp,
    }

    if result.topics is not None:
        data["topics"] = [
            {
                "topic_id": t.topic_id,
                "keywords": t.keywords,
                "sentence_count": t.sentence_count,
                "representative_sentence": t.representative_sentence,
            }
            for t in result.topics
        ]
    else:
        data["topics"] = None

    if result.concept_coverage is not None:
        data["concept_coverage"] = {
            "total_concepts": result.concept_coverage.total_concepts,
            "covered_concepts": result.concept_coverage.covered_concepts,
            "missed_concepts": result.concept_coverage.missed_concepts,
            "coverage_ratio": result.concept_coverage.coverage_ratio,
        }
    else:
        data["concept_coverage"] = None

    if result.triplets is not None:
        data["triplets"] = result.triplets
    else:
        data["triplets"] = None

    return data


def _dict_to_result(data: dict) -> AnalysisResult:
    """Convert a plain dict from YAML to AnalysisResult."""
    topics = None
    if data.get("topics") is not None:
        topics = [
            TopicSummary(
                topic_id=t["topic_id"],
                keywords=t["keywords"],
                sentence_count=t["sentence_count"],
                representative_sentence=t["representative_sentence"],
            )
            for t in data["topics"]
        ]

    concept_coverage = None
    if data.get("concept_coverage") is not None:
        cc = data["concept_coverage"]
        concept_coverage = ConceptCoverage(
            total_concepts=cc["total_concepts"],
            covered_concepts=cc["covered_concepts"],
            missed_concepts=cc["missed_concepts"],
            coverage_ratio=cc["coverage_ratio"],
        )

    net_path = data.get("network_image_path")

    return AnalysisResult(
        class_id=data["class_id"],
        week=data["week"],
        keyword_frequencies=data.get("keyword_frequencies", {}),
        top_keywords=data.get("top_keywords", []),
        network_image_path=Path(net_path) if net_path else None,
        topics=topics,
        topic_skipped_reason=data.get("topic_skipped_reason"),
        concept_coverage=concept_coverage,
        emphasis_scores=data.get("emphasis_scores"),
        triplets=data.get("triplets"),
        triplet_skipped_reason=data.get("triplet_skipped_reason"),
        sentence_count=data.get("sentence_count", 0),
        analysis_timestamp=data.get("analysis_timestamp", ""),
    )
