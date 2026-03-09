"""Data loading and merging for student individual PDF reports.

Reads V1 pipeline YAML outputs (res_lvl1–res_lvl4), student response
YAML (anp_final), and exam configuration YAML to produce per-student
report data structures.  No LLM API calls — fully offline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import yaml

from forma.evaluation_types import TripletEdge

if TYPE_CHECKING:
    from forma.longitudinal_store import LongitudinalStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# forms_data Korean key constants (Google Forms question text → field)
# ---------------------------------------------------------------------------

_FORMS_KEY_NAME = "이름을 입력하세요."
_FORMS_KEY_STUDENT_NUMBER = "학번을 입력하세요."
_FORMS_KEY_CLASS = "분반을 선택하세요."


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConceptDetail:
    """Single concept similarity analysis result from L1 concept_results.

    Args:
        concept: Concept name (e.g. "항상성").
        is_present: Whether similarity >= threshold.
        similarity: Semantic similarity score (0.0–1.0).
        threshold: Adaptive threshold for this concept.
    """

    concept: str
    is_present: bool
    similarity: float
    threshold: float


@dataclass
class QuestionReportData:
    """Evaluation data for one student–question pair, merged from L1–L4.

    Args:
        question_sn: Question serial number (e.g. 1 or 3).
    """

    question_sn: int
    question_text: str = ""
    model_answer: str = ""
    student_answer: str = ""
    concept_coverage: float = 0.0
    concepts: list[ConceptDetail] = field(default_factory=list)
    llm_median_score: float = 0.0
    llm_label: str = "N/A"
    llm_reasoning: str = ""
    misconceptions: list[str] = field(default_factory=list)
    icc_value: float = 0.0
    rasch_theta: float = 0.0
    rasch_theta_se: float = 0.0
    lca_class: int = 0
    lca_class_probability: float = 0.0
    ensemble_score: float = 0.0
    understanding_level: str = "N/A"
    component_scores: dict[str, float] = field(default_factory=dict)
    feedback_text: str = ""
    tier_level: int = 0
    tier_label: str = ""
    graph_comparison_f1: float = 0.0
    graph_matched_edges: list = field(default_factory=list)
    graph_missing_edges: list = field(default_factory=list)
    graph_extra_edges: list = field(default_factory=list)
    graph_wrong_direction_edges: list = field(default_factory=list)
    graph_master_edges: list = field(default_factory=list)
    graph_student_edges: list = field(default_factory=list)
    hub_gap_entries: list = field(default_factory=list)


@dataclass
class StudentReportData:
    """Aggregated report data for a single student.

    Args:
        student_id: Anonymous ID (e.g. "S015").
    """

    student_id: str
    real_name: str = ""
    student_number: str = ""
    class_name: str = ""
    course_name: str = ""
    chapter_name: str = ""
    week_num: int = 0
    questions: list[QuestionReportData] = field(default_factory=list)


@dataclass
class ClassDistributions:
    """Class-level score distributions for comparative charts.

    All dicts are keyed by ``question_sn``.
    """

    ensemble_scores: dict[int, list[float]] = field(default_factory=dict)
    concept_coverages: dict[int, list[float]] = field(default_factory=dict)
    llm_scores: dict[int, list[float]] = field(default_factory=dict)
    rasch_thetas: dict[int, list[float]] = field(default_factory=dict)
    component_scores: dict[int, dict[str, list[float]]] = field(
        default_factory=dict,
    )
    overall_ensemble: list[float] = field(default_factory=list)


@dataclass
class WeeklyDelta:
    """Change information compared to the previous week.

    Args:
        current_score: Current week's score.
        previous_score: Previous week's score (None if first week).
        delta: Score difference (None if first week).
        delta_symbol: Direction symbol: "↑", "↓", "─", or "NEW".
    """

    current_score: float
    previous_score: Optional[float]
    delta: Optional[float]
    delta_symbol: str


def compute_weekly_delta(
    student_id: str,
    current_week: int,
    current_score: float,
    store: LongitudinalStore,
    metric: str,
) -> WeeklyDelta:
    """Compute the delta between current week and the previous available week.

    The "previous week" is the most recent week with data before current_week.
    Threshold: |delta| <= 0.02 → "─", otherwise "↑" or "↓".
    First week (no previous) → "NEW".
    """
    trajectory = store.get_student_trajectory(student_id, metric)
    # Find the most recent week strictly before current_week
    previous_score = None
    for wk, val in reversed(trajectory):
        if wk < current_week:
            previous_score = val
            break

    if previous_score is None:
        return WeeklyDelta(
            current_score=current_score,
            previous_score=None,
            delta=None,
            delta_symbol="NEW",
        )

    delta = current_score - previous_score
    if abs(delta) <= 0.02 + 1e-9:
        symbol = "─"
    elif delta > 0:
        symbol = "↑"
    else:
        symbol = "↓"

    return WeeklyDelta(
        current_score=current_score,
        previous_score=previous_score,
        delta=delta,
        delta_symbol=symbol,
    )


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict | list | None:
    """Load a YAML file with UTF-8 encoding, returning None on failure."""
    if not os.path.exists(path):
        logger.warning("YAML file not found: %s", path)
        return None
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_lookup(
    data: dict | None,
) -> dict[tuple[str, int], dict]:
    """Build a ``{(student_id, question_sn): question_dict}`` lookup.

    Works for any YAML with ``students[].questions[]`` structure.
    """
    lookup: dict[tuple[str, int], dict] = {}
    if data is None:
        return lookup
    for student in data.get("students", []):
        sid = student.get("student_id", "")
        for q in student.get("questions", []):
            qsn = q.get("question_sn", 0)
            lookup[(sid, qsn)] = q
    return lookup


def _build_anp_lookup(
    data: list | None,
) -> dict[tuple[str, int], dict]:
    """Build lookup from anp_final flat array (uses ``q_num``)."""
    lookup: dict[tuple[str, int], dict] = {}
    if data is None:
        return lookup
    for entry in data:
        sid = entry.get("student_id", "")
        qsn = entry.get("q_num", 0)
        lookup[(sid, qsn)] = entry
    return lookup


def _extract_forms_data(
    anp_entries: list[dict],
) -> dict:
    """Extract forms_data fields from the first entry for a student."""
    for entry in anp_entries:
        forms = entry.get("forms_data")
        if forms:
            return {
                "real_name": str(forms.get(_FORMS_KEY_NAME, "")),
                "student_number": str(forms.get(_FORMS_KEY_STUDENT_NUMBER, "")),
                "class_name": str(forms.get(_FORMS_KEY_CLASS, "")),
            }
    return {}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(
    config_path: str,
) -> tuple[dict, dict[int, dict]]:
    """Load exam config YAML.

    Returns:
        Tuple of (metadata_dict, {sn: question_dict}).
    """
    data = _load_yaml(config_path)
    if data is None:
        return {}, {}
    metadata = data.get("metadata", {})
    questions: dict[int, dict] = {}
    for q in data.get("questions", []):
        sn = q.get("sn", 0)
        questions[sn] = q
    return metadata, questions


# ---------------------------------------------------------------------------
# Main loading function
# ---------------------------------------------------------------------------


def load_all_student_data(
    final_path: str,
    config_path: str,
    eval_dir: str,
) -> tuple[list[StudentReportData], ClassDistributions]:
    """Load and merge all YAML sources into per-student report data.

    Uses ``ensemble_results.yaml`` as the authoritative student list.

    Args:
        final_path: Path to anp_final YAML (flat array with forms_data).
        config_path: Path to exam configuration YAML.
        eval_dir: Path to evaluation results directory containing
            ``res_lvl1/`` through ``res_lvl4/``.

    Returns:
        Tuple of (student_data_list, class_distributions).
    """
    # Load all YAML sources
    anp_data = _load_yaml(final_path)
    if not isinstance(anp_data, list):
        anp_data = []

    metadata, config_questions = _load_config(config_path)

    ensemble_data = _load_yaml(
        os.path.join(eval_dir, "res_lvl4", "ensemble_results.yaml"),
    )
    concept_data = _load_yaml(
        os.path.join(eval_dir, "res_lvl1", "concept_results.yaml"),
    )
    llm_data = _load_yaml(
        os.path.join(eval_dir, "res_lvl2", "llm_results.yaml"),
    )
    feedback_data = _load_yaml(
        os.path.join(eval_dir, "res_lvl2", "feedback_results.yaml"),
    )
    statistical_data = _load_yaml(
        os.path.join(eval_dir, "res_lvl3", "statistical_results.yaml"),
    )

    # Build lookup tables
    anp_lookup = _build_anp_lookup(anp_data)
    concept_lookup = _build_lookup(concept_data)
    llm_lookup = _build_lookup(llm_data)
    feedback_lookup = _build_lookup(feedback_data)
    stat_lookup = _build_lookup(statistical_data)

    # Group anp entries by student_id
    anp_by_student: dict[str, list[dict]] = {}
    for entry in anp_data:
        sid = entry.get("student_id", "")
        anp_by_student.setdefault(sid, []).append(entry)

    # Iterate over ensemble (authoritative list)
    students: list[StudentReportData] = []
    if ensemble_data is None:
        logger.error("ensemble_results.yaml is missing or invalid")
        return [], ClassDistributions()

    for ens_student in ensemble_data.get("students", []):
        sid = ens_student.get("student_id", "")

        # Extract forms_data
        forms = _extract_forms_data(anp_by_student.get(sid, []))

        student = StudentReportData(
            student_id=sid,
            real_name=forms.get("real_name", "") or sid,
            student_number=forms.get("student_number", "") or "N/A",
            class_name=forms.get("class_name", "") or "N/A",
            course_name=metadata.get("course_name", ""),
            chapter_name=metadata.get("chapter_name", ""),
            week_num=metadata.get("week_num", 0),
        )

        # Fallback: if no forms_data was found, use student_id
        if not forms:
            student.real_name = sid
            student.student_number = "N/A"
            student.class_name = "N/A"

        # Build question data
        for ens_q in ens_student.get("questions", []):
            qsn = ens_q.get("question_sn", 0)
            key = (sid, qsn)

            # Config data
            cfg_q = config_questions.get(qsn, {})

            # ANP data (student answer)
            anp_entry = anp_lookup.get(key, {})
            student_answer = anp_entry.get("text", "답안 없음")

            # Concept data
            concept_q = concept_lookup.get(key, {})
            raw_concepts = concept_q.get("concepts", [])
            concepts = [
                ConceptDetail(
                    concept=c.get("concept", ""),
                    is_present=c.get("is_present", False),
                    similarity=c.get("similarity", 0.0),
                    threshold=c.get("threshold", 0.0),
                )
                for c in raw_concepts
            ]
            if concepts:
                concept_coverage = sum(
                    1 for c in concepts if c.is_present
                ) / len(concepts)
            else:
                concept_coverage = 0.0

            # LLM data
            llm_q = llm_lookup.get(key, {})

            # Feedback data
            fb_q = feedback_lookup.get(key, {})
            feedback_text = fb_q.get("feedback_text", "")
            if not feedback_text:
                feedback_text = "(피드백 데이터 없음)"

            # Statistical data
            stat_q = stat_lookup.get(key, {})

            question = QuestionReportData(
                question_sn=qsn,
                question_text=cfg_q.get("question", ""),
                model_answer=cfg_q.get("model_answer", ""),
                student_answer=student_answer,
                concept_coverage=concept_coverage,
                concepts=concepts,
                llm_median_score=llm_q.get("median_score", 0.0),
                llm_label=llm_q.get("label", "N/A"),
                llm_reasoning=llm_q.get("reasoning", ""),
                misconceptions=llm_q.get("misconceptions", []),
                icc_value=llm_q.get("icc_value", 0.0),
                rasch_theta=stat_q.get("rasch_theta", 0.0),
                rasch_theta_se=stat_q.get("rasch_theta_se", 0.0),
                lca_class=stat_q.get("lca_class", 0),
                lca_class_probability=stat_q.get("lca_class_probability", 0.0),
                ensemble_score=ens_q.get("ensemble_score", 0.0),
                understanding_level=ens_q.get("understanding_level", "N/A"),
                component_scores=ens_q.get("component_scores", {}),
                feedback_text=feedback_text,
                tier_level=fb_q.get("tier_level", 0),
                tier_label=fb_q.get("tier_label", ""),
            )
            student.questions.append(question)

        # Load graph comparison results (optional — graceful skip if missing)
        graph_path = os.path.join(
            eval_dir, sid, "res_lvl1", "graph_comparison_results.yaml",
        )
        if os.path.exists(graph_path):
            try:
                with open(graph_path, encoding="utf-8") as gf:
                    graph_data = yaml.safe_load(gf)

                if isinstance(graph_data, dict) and sid in graph_data:
                    student_graph = graph_data[sid]
                    # Build a lookup from question_sn to question object
                    q_by_sn = {q.question_sn: q for q in student.questions}
                    for q_key, q_graph in student_graph.items():
                        # Map "question_1" → 1, "question_3" → 3, etc.
                        try:
                            q_sn = int(q_key.split("_")[-1])
                        except (ValueError, AttributeError):
                            continue
                        q_data = q_by_sn.get(q_sn)
                        if q_data is None:
                            continue

                        def _to_triplets(edges_list: list) -> list[TripletEdge]:
                            result = []
                            for e in edges_list:
                                if isinstance(e, dict):
                                    result.append(
                                        TripletEdge(
                                            subject=e.get("subject", ""),
                                            relation=e.get("relation", ""),
                                            object=e.get("object", ""),
                                        )
                                    )
                            return result

                        matched = _to_triplets(q_graph.get("matched_edges", []))
                        missing = _to_triplets(q_graph.get("missing_edges", []))
                        extra = _to_triplets(q_graph.get("extra_edges", []))
                        wrong_dir = _to_triplets(
                            q_graph.get("wrong_direction_edges", []),
                        )

                        q_data.graph_comparison_f1 = float(
                            q_graph.get("f1", 0.0),
                        )
                        q_data.graph_matched_edges = matched
                        q_data.graph_missing_edges = missing
                        q_data.graph_extra_edges = extra
                        q_data.graph_wrong_direction_edges = wrong_dir

                        # Reconstruct display edge lists
                        # master = matched + missing + wrong_direction
                        q_data.graph_master_edges = matched + missing + wrong_dir
                        # student = matched + extra + wrong_direction
                        q_data.graph_student_edges = matched + extra + wrong_dir
            except Exception:
                logger.warning(
                    "Failed to load graph comparison data for student %s", sid,
                )

        students.append(student)

    distributions = compute_class_distributions(students)
    return students, distributions


def compute_class_distributions(
    students: list[StudentReportData],
) -> ClassDistributions:
    """Compute class-level score distributions from student data.

    Args:
        students: List of StudentReportData with populated questions.

    Returns:
        ClassDistributions with per-question and overall aggregations.
    """
    dists = ClassDistributions()

    for student in students:
        for q in student.questions:
            qsn = q.question_sn

            # Ensemble scores
            dists.ensemble_scores.setdefault(qsn, []).append(q.ensemble_score)
            dists.overall_ensemble.append(q.ensemble_score)

            # Concept coverages
            dists.concept_coverages.setdefault(qsn, []).append(
                q.concept_coverage,
            )

            # LLM scores
            dists.llm_scores.setdefault(qsn, []).append(q.llm_median_score)

            # Rasch thetas
            dists.rasch_thetas.setdefault(qsn, []).append(q.rasch_theta)

            # Component scores
            dists.component_scores.setdefault(qsn, {})
            for comp_name, comp_val in q.component_scores.items():
                dists.component_scores[qsn].setdefault(
                    comp_name, [],
                ).append(comp_val)

    return dists
