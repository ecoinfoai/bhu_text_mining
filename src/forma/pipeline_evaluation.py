"""Multi-layer concept evaluation pipeline orchestration (v2).

Runs evaluation layers in sequence for every (student, question) pair.

v2 changes:
- Triplet extraction + graph comparison for essay questions with knowledge_graph
- LLM role changed from scorer to feedback generator
- Graph overlay visualization
- Longitudinal data accumulation
- Backward compatible with v1 configs (no knowledge_graph)

Usage:
    uv run python pipeline_evaluation.py \\
        --config exams/Ch01_서론_FormativeTest.yaml \\
        --responses data/responses.yaml \\
        --output results/ch01/

Optional flags:
    --skip-feedback   Skip feedback generation (saves API cost)
    --skip-llm        Deprecated alias for --skip-feedback
    --skip-graph      Skip triplet extraction / graph comparison
    --skip-stats      Skip Layer 3 statistical analysis
    --lecture-transcript PATH   Lecture transcript file
    --longitudinal-store PATH   Longitudinal data store path
    --generate-reports  Generate student PDF reports
    --api-key KEY     Override API key environment variable
    --provider NAME   LLM provider: gemini (default) or anthropic
    --model ID        LLM model override
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Optional

from forma.concept_checker import check_all_concepts
from forma.config_validator import validate_exam_config
from forma.ensemble_scorer import EnsembleScorer
from forma.evaluation_io import (
    extract_student_responses,
    load_evaluation_yaml,
    save_evaluation_yaml,
)
from forma.response_converter import (
    convert_join_to_responses,
    filter_exam_config,
)
from forma.evaluation_types import (
    AggregatedLLMResult,
    EnsembleResult,
    FeedbackResult,
    GraphComparisonResult,
    GraphMetricResult,
    RubricTier,
    StatisticalResult,
    TripletEdge,
    TripletExtractionResult,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_question_config(config_data: dict, question_sn: int) -> dict:
    """Extract a single question's config by serial number."""
    questions = config_data.get("questions", [])
    for q in questions:
        if q.get("sn") == question_sn:
            return q
    raise KeyError(
        f"Question sn={question_sn} not found in exam config. "
        f"Available: {[q.get('sn') for q in questions]}"
    )


def _is_v2_question(q: dict) -> bool:
    """Check if question uses v2 mode (essay + knowledge_graph)."""
    qtype = q.get("question_type", "essay")
    return qtype == "essay" and "knowledge_graph" in q


def _get_master_edges(q: dict) -> list[TripletEdge]:
    """Extract master edges from question config."""
    kg = q.get("knowledge_graph", {})
    edges = kg.get("edges", [])
    return [
        TripletEdge(
            subject=e["subject"],
            relation=e["relation"],
            object=e["object"],
        )
        for e in edges
    ]


def _get_master_nodes(q: dict) -> list[str]:
    """Extract unique node names from master edges."""
    edges = _get_master_edges(q)
    nodes: set[str] = set()
    for e in edges:
        nodes.add(e.subject)
        nodes.add(e.object)
    return sorted(nodes)


def _get_rubric_tiers(q: dict) -> list[RubricTier]:
    """Parse rubric_tiers from question config."""
    tiers_cfg = q.get("rubric_tiers", {})
    tiers: list[RubricTier] = []
    for key, cfg in tiers_cfg.items():
        level = int(key.replace("level_", ""))
        tiers.append(
            RubricTier(
                level=level,
                label=cfg.get("label", ""),
                min_graph_f1=cfg.get("min_graph_f1", 0.0),
                requires_terminology=cfg.get("requires_terminology", False),
            )
        )
    return sorted(tiers, key=lambda t: t.level)


# ---------------------------------------------------------------------------
# Layer runners
# ---------------------------------------------------------------------------


def _run_layer1(
    student_responses: dict[str, dict[int, str]],
    config_data: dict,
) -> dict[str, dict[int, list]]:
    """Run Layer 1 concept checking for all students and questions."""
    results: dict[str, dict[int, list]] = {}
    questions = config_data.get("questions", [])

    # Build work list for progress tracking
    work: list[tuple[dict, str]] = []
    for q in questions:
        if not q.get("keywords"):
            continue
        for student_id, responses in student_responses.items():
            if responses.get(q["sn"], ""):
                work.append((q, student_id))

    total = len(work)
    for idx, (q, student_id) in enumerate(work, 1):
        qsn = q["sn"]
        concepts = q["keywords"]
        text = student_responses[student_id][qsn]
        print(
            f"\r[pipeline] Concept check: {idx}/{total} "
            f"({student_id}, q{qsn}) …",
            end="", flush=True,
        )
        cr = check_all_concepts(
            student_text=text,
            student_id=student_id,
            question_sn=qsn,
            concepts=concepts,
        )
        results.setdefault(student_id, {})[qsn] = cr

    if total:
        print()  # newline after progress

    return results


def _run_triplet_extraction(
    student_responses: dict[str, dict[int, str]],
    config_data: dict,
    api_key: Optional[str],
    provider: str = "gemini",
    model: Optional[str] = None,
    n_calls: int = 3,
) -> dict[str, dict[int, TripletExtractionResult]]:
    """Run triplet extraction for v2 questions."""
    from forma.llm_provider import create_provider
    from forma.triplet_extractor import TripletExtractor

    llm_prov = create_provider(provider=provider, api_key=api_key, model=model)
    extractor = TripletExtractor(llm_prov, n_calls=n_calls)
    results: dict[str, dict[int, TripletExtractionResult]] = {}

    for q in config_data.get("questions", []):
        if not _is_v2_question(q):
            continue
        qsn = q["sn"]
        master_nodes = _get_master_nodes(q)

        for student_id, responses in student_responses.items():
            text = responses.get(qsn, "")
            ter = extractor.extract(
                student_id=student_id,
                question_sn=qsn,
                question=q.get("question", ""),
                student_response=text,
                master_nodes=master_nodes,
            )
            results.setdefault(student_id, {})[qsn] = ter

    return results


def _run_graph_comparison(
    triplet_results: dict[str, dict[int, TripletExtractionResult]],
    config_data: dict,
    lecture_covered_concepts: Optional[list[str]] = None,
) -> dict[str, dict[int, GraphComparisonResult]]:
    """Run graph comparison for v2 questions."""
    from forma.graph_comparator import GraphComparator

    results: dict[str, dict[int, GraphComparisonResult]] = {}

    for q in config_data.get("questions", []):
        if not _is_v2_question(q):
            continue
        qsn = q["sn"]
        master_edges = _get_master_edges(q)
        kg = q.get("knowledge_graph", {})
        threshold = kg.get("similarity_threshold", 0.80)
        aliases = kg.get("node_aliases", {})

        comparator = GraphComparator(
            similarity_threshold=threshold,
            node_aliases=aliases,
        )

        for student_id, q_results in triplet_results.items():
            ter = q_results.get(qsn)
            if ter is None:
                continue
            gcr = comparator.compare(
                student_id=student_id,
                question_sn=qsn,
                master_edges=master_edges,
                student_edges=ter.triplets,
                lecture_covered_concepts=lecture_covered_concepts,
            )
            results.setdefault(student_id, {})[qsn] = gcr

    return results


def _run_layer2_v1(
    student_responses: dict[str, dict[int, str]],
    config_data: dict,
    api_key: Optional[str],
    provider: str = "gemini",
    model: Optional[str] = None,
    n_calls: int = 3,
) -> tuple[dict[str, dict[int, AggregatedLLMResult]], "LLMEvaluator"]:
    """Run v1 Layer 2 LLM evaluation for non-v2 questions.

    Returns:
        Tuple of (results dict, evaluator instance). The evaluator is
        returned so the pipeline can call retry_failed_calls() later.
    """
    from forma.llm_evaluator import LLMEvaluator

    evaluator = LLMEvaluator(
        api_key=api_key, provider=provider, model=model, n_calls=n_calls,
    )
    results: dict[str, dict[int, AggregatedLLMResult]] = {}

    # Build work list for progress tracking
    work: list[tuple[dict, str]] = []
    for q in config_data.get("questions", []):
        if _is_v2_question(q):
            continue
        for student_id, responses in student_responses.items():
            if responses.get(q["sn"], ""):
                work.append((q, student_id))

    total = len(work)
    for idx, (q, student_id) in enumerate(work, 1):
        qsn = q["sn"]
        rubric = q.get("rubric", {})
        text = student_responses[student_id][qsn]
        print(
            f"\r[pipeline] LLM eval: {idx}/{total} "
            f"({student_id}, q{qsn}) …",
            end="", flush=True,
        )
        agg = evaluator.evaluate_response(
            student_id=student_id,
            question_sn=qsn,
            question=q.get("question", ""),
            student_response=text,
            model_answer=q.get("model_answer", ""),
            rubric_high=rubric.get("high", ""),
            rubric_mid=rubric.get("mid", ""),
            rubric_low=rubric.get("low", ""),
            concepts=q.get("keywords", []),
        )
        results.setdefault(student_id, {})[qsn] = agg

    if total:
        print()  # newline after progress

    return results, evaluator


def _run_feedback(
    student_responses: dict[str, dict[int, str]],
    config_data: dict,
    ensemble_results: dict[str, dict[int, EnsembleResult]],
    graph_results: Optional[dict[str, dict[int, GraphComparisonResult]]],
    api_key: Optional[str],
    provider: str = "gemini",
    model: Optional[str] = None,
    lecture_tone: str = "",
) -> dict[str, dict[int, FeedbackResult]]:
    """Run feedback generation for all students and questions."""
    from forma.feedback_generator import FeedbackGenerator
    from forma.llm_provider import create_provider

    llm_prov = create_provider(provider=provider, api_key=api_key, model=model)
    gen = FeedbackGenerator(llm_prov)
    results: dict[str, dict[int, FeedbackResult]] = {}

    # Build work list for progress tracking
    work: list[tuple[dict, str]] = []
    for q in config_data.get("questions", []):
        qsn = q["sn"]
        for student_id in student_responses:
            er = (ensemble_results.get(student_id) or {}).get(qsn)
            if er is not None:
                work.append((q, student_id))

    total = len(work)
    for idx, (q, student_id) in enumerate(work, 1):
        qsn = q["sn"]
        tiers = _get_rubric_tiers(q) if _is_v2_question(q) else []
        text = student_responses[student_id].get(qsn, "")
        er = ensemble_results[student_id][qsn]

        gc = None
        if graph_results:
            gc = (graph_results.get(student_id) or {}).get(qsn)

        # Determine tier
        tier_level = 0
        tier_label = "미달"
        if gc and tiers:
            for tier in sorted(tiers, key=lambda t: t.level, reverse=True):
                if gc.f1 >= tier.min_graph_f1:
                    tier_level = tier.level
                    tier_label = tier.label
                    break

        coverage = er.component_scores.get("concept_coverage", 0.0)

        print(
            f"\r[pipeline] Feedback: {idx}/{total} "
            f"({student_id}, q{qsn}) …",
            end="", flush=True,
        )
        fr = gen.generate(
            student_id=student_id,
            question_sn=qsn,
            question=q.get("question", ""),
            student_response=text,
            concept_coverage=coverage,
            graph_comparison=gc,
            tier_level=tier_level,
            tier_label=tier_label,
            lecture_tone=lecture_tone,
        )
        results.setdefault(student_id, {})[qsn] = fr

    if total:
        print()  # newline after progress

    return results


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


def _build_counseling_summary(
    ensemble_results: dict[str, dict[int, EnsembleResult]],
    config_data: dict,
    llm_results: Optional[dict[str, dict[int, AggregatedLLMResult]]],
    feedback_results: Optional[dict[str, dict[int, FeedbackResult]]] = None,
) -> dict:
    """Build professor-facing counseling summary."""
    summary: dict = {"students": []}
    questions = config_data.get("questions", [])

    for student_id, q_results in ensemble_results.items():
        student_entry: dict = {"student_id": student_id, "questions": []}
        for qsn, er in sorted(q_results.items()):
            q_cfg = next(
                (q for q in questions if q.get("sn") == qsn), {}
            )
            support = q_cfg.get("support", {})
            level_key = er.understanding_level.lower()

            misconceptions: list[str] = []
            if llm_results and student_id in llm_results:
                lr = llm_results[student_id].get(qsn)
                if lr:
                    misconceptions = lr.misconceptions

            q_entry: dict = {
                "question_sn": qsn,
                "understanding_level": er.understanding_level,
                "concept_coverage": er.component_scores.get(
                    "concept_coverage", 0.0
                ),
                "support_guidance": support.get(level_key, ""),
                "misconceptions": misconceptions,
            }

            # Add feedback if available
            if feedback_results:
                fr = (feedback_results.get(student_id) or {}).get(qsn)
                if fr:
                    q_entry["feedback"] = fr.feedback_text
                    q_entry["tier_level"] = fr.tier_level
                    q_entry["tier_label"] = fr.tier_label

            student_entry["questions"].append(q_entry)
        summary["students"].append(student_entry)

    return summary


def _build_technical_report(
    ensemble_results: dict[str, dict[int, EnsembleResult]],
    layer1_results: dict[str, dict[int, list]],
    llm_results: Optional[dict[str, dict[int, AggregatedLLMResult]]],
    stat_results: Optional[dict[str, dict[int, StatisticalResult]]],
    graph_results: Optional[dict[str, dict[int, GraphComparisonResult]]] = None,
) -> dict:
    """Build technical report with all parameters."""
    report: dict = {"students": []}

    for student_id, q_results in ensemble_results.items():
        student_entry: dict = {
            "student_id": student_id,
            "questions": [],
        }
        for qsn, er in sorted(q_results.items()):
            q_entry: dict = {
                "question_sn": qsn,
                "ensemble_score": round(er.ensemble_score, 4),
                "understanding_level": er.understanding_level,
                "component_scores": {
                    k: round(v, 4) for k, v in er.component_scores.items()
                },
                "weights_used": {
                    k: round(v, 4) for k, v in er.weights_used.items()
                },
            }

            # Layer 1 detail
            l1 = (layer1_results.get(student_id) or {}).get(qsn, [])
            q_entry["concept_details"] = [
                {
                    "concept": r.concept,
                    "is_present": r.is_present,
                    "similarity": round(r.similarity_score, 4),
                    "threshold": round(r.threshold_used, 4),
                }
                for r in l1
            ]

            # Layer 2 detail (v1 mode)
            if llm_results:
                lr = (llm_results.get(student_id) or {}).get(qsn)
                if lr:
                    q_entry["llm_evaluation"] = {
                        "median_score": lr.median_rubric_score,
                        "label": lr.rubric_label,
                        "reasoning": lr.reasoning,
                        "misconceptions": lr.misconceptions,
                        "uncertain": lr.uncertain,
                        "icc_value": lr.icc_value,
                    }

            # Layer 3 detail
            if stat_results:
                sr = (stat_results.get(student_id) or {}).get(qsn)
                if sr:
                    q_entry["statistical_analysis"] = {
                        "rasch_theta": sr.rasch_theta,
                        "rasch_theta_se": sr.rasch_theta_se,
                        "lca_class": sr.lca_class,
                        "lca_class_probability": sr.lca_class_probability,
                        "lca_exploratory_warning": sr.lca_exploratory_warning,
                    }

            # Graph comparison detail (v2 mode)
            if graph_results:
                gcr = (graph_results.get(student_id) or {}).get(qsn)
                if gcr:
                    q_entry["graph_comparison"] = {
                        "precision": round(gcr.precision, 4),
                        "recall": round(gcr.recall, 4),
                        "f1": round(gcr.f1, 4),
                        "matched_count": len(gcr.matched_edges),
                        "missing_count": len(gcr.missing_edges),
                        "extra_count": len(gcr.extra_edges),
                        "wrong_direction_count": len(gcr.wrong_direction_edges),
                        "fuzzy_matched": gcr.fuzzy_matched,
                    }

            student_entry["questions"].append(q_entry)
        report["students"].append(student_entry)

    return report


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _serialize_layer1(results: dict[str, dict[int, list]]) -> dict:
    """Serialize Layer 1 concept results for YAML output."""
    out: dict = {"students": []}
    for sid, q_results in results.items():
        entry = {"student_id": sid, "questions": []}
        for qsn, concepts in sorted(q_results.items()):
            entry["questions"].append({
                "question_sn": qsn,
                "concepts": [
                    {
                        "concept": r.concept,
                        "is_present": r.is_present,
                        "similarity": round(r.similarity_score, 4),
                        "threshold": round(r.threshold_used, 4),
                    }
                    for r in concepts
                ],
            })
        out["students"].append(entry)
    return out


def _serialize_graph_comparison(
    results: dict[str, dict[int, GraphComparisonResult]]
) -> dict:
    """Serialize graph comparison results for YAML output."""
    out: dict = {"students": []}
    for sid, q_results in results.items():
        entry = {"student_id": sid, "questions": []}
        for qsn, gcr in sorted(q_results.items()):
            entry["questions"].append({
                "question_sn": qsn,
                "precision": round(gcr.precision, 4),
                "recall": round(gcr.recall, 4),
                "f1": round(gcr.f1, 4),
                "matched_count": len(gcr.matched_edges),
                "missing_count": len(gcr.missing_edges),
                "extra_count": len(gcr.extra_edges),
                "wrong_direction_count": len(gcr.wrong_direction_edges),
                "fuzzy_matched": gcr.fuzzy_matched,
            })
        out["students"].append(entry)
    return out


def _serialize_feedback(
    results: dict[str, dict[int, FeedbackResult]]
) -> dict:
    """Serialize feedback results for YAML output."""
    out: dict = {"students": []}
    for sid, q_results in results.items():
        entry = {"student_id": sid, "questions": []}
        for qsn, fr in sorted(q_results.items()):
            entry["questions"].append({
                "question_sn": qsn,
                "feedback_text": fr.feedback_text,
                "char_count": fr.char_count,
                "tier_level": fr.tier_level,
                "tier_label": fr.tier_label,
                "data_sources_used": fr.data_sources_used,
            })
        out["students"].append(entry)
    return out


def _serialize_layer2(results: dict[str, dict[int, AggregatedLLMResult]]) -> dict:
    """Serialize Layer 2 LLM results for YAML output."""
    out: dict = {"students": []}
    for sid, q_results in results.items():
        entry = {"student_id": sid, "questions": []}
        for qsn, agg in sorted(q_results.items()):
            entry["questions"].append({
                "question_sn": qsn,
                "median_score": agg.median_rubric_score,
                "label": agg.rubric_label,
                "reasoning": agg.reasoning,
                "misconceptions": agg.misconceptions,
                "uncertain": agg.uncertain,
                "icc_value": agg.icc_value,
            })
        out["students"].append(entry)
    return out


def _serialize_layer3(results: dict[str, dict[int, StatisticalResult]]) -> dict:
    """Serialize Layer 3 statistical results for YAML output."""
    out: dict = {"students": []}
    for sid, q_results in results.items():
        entry = {"student_id": sid, "questions": []}
        for qsn, sr in sorted(q_results.items()):
            entry["questions"].append({
                "question_sn": qsn,
                "rasch_theta": sr.rasch_theta,
                "rasch_theta_se": sr.rasch_theta_se,
                "lca_class": sr.lca_class,
                "lca_class_probability": sr.lca_class_probability,
            })
        out["students"].append(entry)
    return out


def _serialize_ensemble(results: dict[str, dict[int, EnsembleResult]]) -> dict:
    """Serialize Layer 4 ensemble results for YAML output."""
    out: dict = {"students": []}
    for sid, q_results in results.items():
        entry = {"student_id": sid, "questions": []}
        for qsn, er in sorted(q_results.items()):
            entry["questions"].append({
                "question_sn": qsn,
                "ensemble_score": round(er.ensemble_score, 4),
                "understanding_level": er.understanding_level,
                "component_scores": {
                    k: round(v, 4) for k, v in er.component_scores.items()
                },
            })
        out["students"].append(entry)
    return out


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_evaluation_pipeline(
    config_path: str,
    responses_path: str,
    output_dir: str,
    api_key: Optional[str] = None,
    skip_llm: bool = False,
    skip_feedback: bool = False,
    skip_graph: bool = False,
    skip_statistical: bool = False,
    provider: str = "gemini",
    model: Optional[str] = None,
    lecture_transcript: Optional[str] = None,
    longitudinal_store: Optional[str] = None,
    generate_reports: bool = False,
    questions_used: Optional[list[int]] = None,
    n_calls: int = 3,
) -> None:
    """Run the full multi-layer concept evaluation pipeline.

    Supports both v1 (keyword + LLM rubric) and v2 (triplet graph +
    feedback) modes, auto-detected per question based on config.

    Args:
        config_path: Path to exam YAML config file.
        responses_path: Path to student responses YAML file.
        output_dir: Directory for output files.
        api_key: LLM API key override.
        skip_llm: Deprecated alias for skip_feedback.
        skip_feedback: Skip feedback generation.
        skip_graph: Skip triplet extraction and graph comparison.
        skip_statistical: Skip Layer 3 statistical analysis.
        provider: LLM provider name.
        model: LLM model ID override.
        lecture_transcript: Path to lecture transcript file.
        longitudinal_store: Path to longitudinal data store.
        generate_reports: Whether to generate PDF reports.
        questions_used: Exam ``sn`` numbers in q_num order.
            e.g. ``[1, 3]`` means OCR q1 maps to exam sn1, q2 to sn3.
            When provided, *responses_path* is treated as OCR join
            output (flat list) and the exam config is filtered to
            only the selected questions.
        n_calls: Number of independent LLM calls per evaluation
            (default 3). Use 2 for cost savings.
    """
    # Handle deprecated --skip-llm
    if skip_llm and not skip_feedback:
        warnings.warn(
            "--skip-llm is deprecated, use --skip-feedback instead",
            DeprecationWarning,
            stacklevel=2,
        )
        skip_feedback = True

    # --- Resolve API key from forma config if not provided ---
    if api_key is None:
        try:
            from forma.config import get_llm_config, load_config

            app_config = load_config()
            llm_cfg = get_llm_config(app_config)
            if llm_cfg.get("api_key"):
                api_key = llm_cfg["api_key"]
                if not provider or provider == "gemini":
                    provider = llm_cfg.get("provider", provider)
                print(f"[pipeline] API key loaded from forma config ({provider})")
        except FileNotFoundError:
            pass

    config_data = load_evaluation_yaml(config_path)

    if questions_used:
        # Filter exam config to selected questions
        config_data = filter_exam_config(config_data, questions_used)
        print(
            f"[pipeline] questions_used={questions_used} → "
            f"sn {[q['sn'] for q in config_data['questions']]}"
        )

        # Load OCR join output and remap q_num → sn
        import yaml

        with open(responses_path, "r", encoding="utf-8") as f:
            join_data = yaml.safe_load(f)
        if isinstance(join_data, list):
            responses_data = convert_join_to_responses(
                join_data, questions_used
            )
        else:
            responses_data = join_data
        student_responses = extract_student_responses(responses_data)
    else:
        responses_data = load_evaluation_yaml(responses_path)
        student_responses = extract_student_responses(responses_data)

    # Validate config
    errors = validate_exam_config(config_data)
    if errors:
        for e in errors:
            print(f"[pipeline] CONFIG ERROR: {e}")
        raise ValueError(f"Config validation failed with {len(errors)} errors")

    questions = config_data.get("questions", [])
    has_v2 = any(_is_v2_question(q) for q in questions)

    print(
        f"[pipeline] {len(student_responses)} students, "
        f"{len(questions)} questions "
        f"(v2: {sum(1 for q in questions if _is_v2_question(q))})"
    )

    # --- Load lecture transcript if provided ---
    lecture_tone = ""
    lecture_covered_concepts: Optional[list[str]] = None
    if lecture_transcript:
        from forma.lecture_processor import (
            extract_lecture_covered_concepts,
            extract_lecture_tone_sample,
            load_transcript,
        )

        print(f"[pipeline] Loading lecture transcript: {lecture_transcript}")
        lecture_text = load_transcript(lecture_transcript)
        lecture_tone = extract_lecture_tone_sample(lecture_text)

        # Collect all master nodes for coverage check
        all_master_nodes: list[str] = []
        for q in questions:
            if _is_v2_question(q):
                all_master_nodes.extend(_get_master_nodes(q))
        if all_master_nodes:
            lecture_covered_concepts = extract_lecture_covered_concepts(
                lecture_text, list(set(all_master_nodes))
            )
            print(
                f"[pipeline] Lecture covers {len(lecture_covered_concepts)} "
                f"of {len(set(all_master_nodes))} master concepts"
            )

    # === Phase 1: Layer 1 (concept checker + triplet extraction + graph comparison) ===
    # Pre-load embedding model and kss so their startup messages
    # don't interfere with progress bars.
    print("[pipeline] Phase 1: loading models …", end="", flush=True)
    from forma.embedding_cache import get_encoder, _suppress_noisy_output
    get_encoder()  # warm up — suppresses LOAD REPORT
    with _suppress_noisy_output():
        try:
            import kss as _kss  # noqa: F811
            _kss.split_sentences("_")  # triggers kss backend detection message
        except Exception:
            pass
    print(" done.")
    print("[pipeline] Phase 1: concept checking …")
    layer1_results = _run_layer1(student_responses, config_data)

    # Triplet extraction + graph comparison (v2 questions only)
    triplet_results: Optional[dict] = None
    graph_results: Optional[dict] = None
    if has_v2 and not skip_graph:
        print(f"[pipeline] Phase 1: triplet extraction ({provider}, {n_calls}-call) …")
        triplet_results = _run_triplet_extraction(
            student_responses, config_data, api_key,
            provider=provider, model=model, n_calls=n_calls,
        )
        print("[pipeline] Phase 1: graph comparison …")
        graph_results = _run_graph_comparison(
            triplet_results, config_data, lecture_covered_concepts
        )

    # === Phase 2: Statistical analysis (Rasch IRT) ===
    stat_results: Optional[dict] = None
    if not skip_statistical:
        from forma.statistical_analysis import (
            RaschAnalyzer, LCAAnalyzer, compute_concept_matrix,
        )
        import numpy as np

        print("[pipeline] Phase 2: statistical analysis …")
        stat_results = {}
        stat_questions = [q for q in questions if q.get("keywords")]
        for qi, q in enumerate(stat_questions, 1):
            qsn = q["sn"]
            concepts = q["keywords"]
            print(
                f"\r[pipeline]   Rasch IRT: q{qsn} ({qi}/{len(stat_questions)}) …",
                end="", flush=True,
            )
            student_ids = list(student_responses.keys())
            flat = [
                r
                for sid in student_ids
                for r in (layer1_results.get(sid) or {}).get(qsn, [])
            ]
            if not flat or not concepts:
                continue
            mat, sids, _ = compute_concept_matrix(flat, student_ids, concepts)
            if mat.shape[1] < 2:
                continue
            try:
                ra = RaschAnalyzer(question_sn=qsn)
                ra.fit(mat)
                thetas, ses = ra.ability_estimates(mat)

                # LCA (exploratory, graceful if stepmix unavailable)
                lca_labels = None
                lca_probs = None
                try:
                    max_k = max(2, min(4, mat.shape[0] // 10))
                    lca = LCAAnalyzer(max_classes=max_k)
                    lca_labels, lca_probs = lca.fit_predict(mat)
                except Exception:
                    pass

                for i, sid in enumerate(sids):
                    sr = StatisticalResult(
                        student_id=sid,
                        question_sn=qsn,
                        rasch_theta=float(thetas[i]),
                        rasch_theta_se=float(ses[i]),
                    )
                    if lca_labels is not None:
                        sr.lca_class = int(lca_labels[i])
                        sr.lca_class_probability = float(
                            lca_probs[i, lca_labels[i]]
                        )
                    stat_results.setdefault(sid, {})[qsn] = sr
            except Exception as exc:
                print(f"\n[pipeline]   Rasch failed for q{qsn}: {exc}")
        if stat_questions:
            print()

    # === Phase 3: LLM rubric evaluation (3-call per student×question) ===
    llm_results: Optional[dict] = None
    llm_evaluator = None
    has_v1 = any(not _is_v2_question(q) for q in questions)
    if has_v1 and not skip_feedback:
        print(f"[pipeline] Phase 3: LLM rubric evaluation ({provider}) …")
        llm_results, llm_evaluator = _run_layer2_v1(
            student_responses, config_data, api_key,
            provider=provider, model=model, n_calls=n_calls,
        )

    # --- Post-LLM: compute per-question ICC(2,1) across students ---
    if llm_results:
        from forma.llm_evaluator import compute_icc_2_1
        import numpy as _np

        # Group by question
        q_calls: dict[int, list[list[int]]] = {}
        q_aggs: dict[int, list[AggregatedLLMResult]] = {}
        for sid, q_dict in llm_results.items():
            for qsn, agg in q_dict.items():
                scores = [c.rubric_score for c in agg.individual_calls]
                q_calls.setdefault(qsn, []).append(scores)
                q_aggs.setdefault(qsn, []).append(agg)
        for qsn, all_scores in q_calls.items():
            try:
                ratings = _np.array(all_scores, dtype=float)
                if ratings.shape[0] >= 2 and ratings.shape[1] >= 2:
                    icc = compute_icc_2_1(ratings)
                else:
                    icc = 1.0
                for agg in q_aggs[qsn]:
                    agg.icc_value = icc
            except Exception:
                pass

    # === Phase 4: Ensemble scoring ===
    print("[pipeline] Phase 4: ensemble scoring …")
    scorer = EnsembleScorer()
    ensemble_results: dict[str, dict[int, EnsembleResult]] = {}

    ensemble_work = [
        (sid, qsn)
        for sid, q_responses in student_responses.items()
        for qsn in q_responses
    ]
    for idx, (student_id, qsn) in enumerate(ensemble_work, 1):
        print(
            f"\r[pipeline]   Ensemble: {idx}/{len(ensemble_work)} "
            f"({student_id}, q{qsn}) …",
            end="", flush=True,
        )
        q_cfg = _load_question_config(config_data, qsn)
        cr = (layer1_results.get(student_id) or {}).get(qsn, [])
        llm = (llm_results.get(student_id) or {}).get(qsn) if llm_results else None
        stat = (stat_results.get(student_id) or {}).get(qsn) if stat_results else None
        gc = (graph_results.get(student_id) or {}).get(qsn) if graph_results else None
        rubric_tiers = _get_rubric_tiers(q_cfg) if _is_v2_question(q_cfg) else None

        er = scorer.compute_score(
            concept_results=cr,
            llm_result=llm,
            statistical_result=stat,
            graph_result=None,
            bertscore_f1=None,
            student_id=student_id,
            question_sn=qsn,
            graph_comparison=gc,
            rubric_tiers=rubric_tiers,
        )
        ensemble_results.setdefault(student_id, {})[qsn] = er
    if ensemble_work:
        print()

    # === Phase 5: Feedback generation ===
    feedback_results: Optional[dict] = None
    if not skip_feedback:
        print(f"[pipeline] Phase 5: feedback generation ({provider}) …")
        feedback_results = _run_feedback(
            student_responses, config_data, ensemble_results,
            graph_results, api_key,
            provider=provider, model=model,
            lecture_tone=lecture_tone,
        )

    # === Phase 5b: Retry failed LLM calls ===
    if llm_evaluator and llm_evaluator.failed_calls:
        n_failed = len(llm_evaluator.failed_calls)
        print(f"[pipeline] Retrying {n_failed} failed LLM call(s) …")
        results_map: dict[tuple[str, int], AggregatedLLMResult] = {}
        if llm_results:
            for sid, q_dict in llm_results.items():
                for qsn, agg in q_dict.items():
                    results_map[(sid, qsn)] = agg
        retried = llm_evaluator.retry_failed_calls(results_map)
        still_failed = len(llm_evaluator.failed_calls)
        print(
            f"[pipeline] Retry complete: {retried} recovered, "
            f"{still_failed} still failed."
        )

    # === Phase 6: Output ===
    print("[pipeline] Phase 6: writing results …")

    # Layer 1
    l1_dir = os.path.join(output_dir, "res_lvl1")
    save_evaluation_yaml(
        _serialize_layer1(layer1_results),
        os.path.join(l1_dir, "concept_results.yaml"),
    )

    # Graph comparison (v2)
    if graph_results:
        save_evaluation_yaml(
            _serialize_graph_comparison(graph_results),
            os.path.join(l1_dir, "graph_comparison_results.yaml"),
        )

    # Layer 2 (feedback or v1 LLM)
    l2_dir = os.path.join(output_dir, "res_lvl2")
    if feedback_results:
        save_evaluation_yaml(
            _serialize_feedback(feedback_results),
            os.path.join(l2_dir, "feedback_results.yaml"),
        )
    if llm_results:
        save_evaluation_yaml(
            _serialize_layer2(llm_results),
            os.path.join(l2_dir, "llm_results.yaml"),
        )

    # Layer 3
    if stat_results:
        l3_dir = os.path.join(output_dir, "res_lvl3")
        save_evaluation_yaml(
            _serialize_layer3(stat_results),
            os.path.join(l3_dir, "statistical_results.yaml"),
        )

    # Layer 4
    l4_dir = os.path.join(output_dir, "res_lvl4")
    counseling = _build_counseling_summary(
        ensemble_results, config_data, llm_results, feedback_results
    )
    technical = _build_technical_report(
        ensemble_results, layer1_results, llm_results, stat_results,
        graph_results,
    )
    save_evaluation_yaml(
        _serialize_ensemble(ensemble_results),
        os.path.join(l4_dir, "ensemble_results.yaml"),
    )
    save_evaluation_yaml(counseling, os.path.join(l4_dir, "counseling_summary.yaml"))
    save_evaluation_yaml(technical, os.path.join(l4_dir, "technical_report.yaml"))

    # Graph visualizations (v2)
    if graph_results and triplet_results:
        _generate_graph_visualizations(
            config_data, triplet_results, graph_results, output_dir
        )

    # Longitudinal data
    if longitudinal_store:
        _save_longitudinal(
            longitudinal_store, ensemble_results, graph_results, config_data
        )

    # PDF reports
    if generate_reports:
        _generate_pdf_reports(output_dir, config_data, counseling, graph_results)

    print(f"[pipeline] Done. Results written to: {output_dir}")


def _generate_graph_visualizations(
    config_data: dict,
    triplet_results: dict,
    graph_results: dict,
    output_dir: str,
) -> None:
    """Generate graph overlay PNGs for v2 questions."""
    try:
        from forma.graph_visualizer import GraphVisualizer

        viz = GraphVisualizer()
        graphs_dir = os.path.join(output_dir, "res_lvl4", "graphs")

        for student_id, q_results in graph_results.items():
            for qsn, gcr in q_results.items():
                q_cfg = _load_question_config(config_data, qsn)
                master_edges = _get_master_edges(q_cfg)
                ter = (triplet_results.get(student_id) or {}).get(qsn)
                student_edges = ter.triplets if ter else []

                output_path = os.path.join(
                    graphs_dir, f"{student_id}_q{qsn}.png"
                )
                viz.visualize_comparison(
                    master_edges=master_edges,
                    student_edges=student_edges,
                    matched_edges=gcr.matched_edges,
                    missing_edges=gcr.missing_edges,
                    extra_edges=gcr.extra_edges,
                    wrong_direction_edges=gcr.wrong_direction_edges,
                    output_path=output_path,
                    title=f"{student_id} Q{qsn}",
                )
        print(f"[pipeline] Graph visualizations saved to {graphs_dir}")
    except Exception as exc:
        print(f"[pipeline] Graph visualization failed: {exc}")


def _save_longitudinal(
    store_path: str,
    ensemble_results: dict,
    graph_results: Optional[dict],
    config_data: dict,
) -> None:
    """Save results to longitudinal store."""
    try:
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(store_path)
        store.load()

        week = config_data.get("longitudinal", {}).get("week", 1)

        for student_id, q_results in ensemble_results.items():
            for qsn, er in q_results.items():
                scores = dict(er.component_scores)
                record = LongitudinalRecord(
                    student_id=student_id,
                    week=week,
                    question_sn=qsn,
                    scores=scores,
                    tier_level=0,
                    tier_label=er.understanding_level,
                )
                store.add_record(record)

        store.save()
        print(f"[pipeline] Longitudinal data saved to {store_path}")
    except Exception as exc:
        print(f"[pipeline] Longitudinal save failed: {exc}")


def _generate_pdf_reports(
    output_dir: str,
    config_data: dict,
    counseling: dict,
    graph_results: Optional[dict],
) -> None:
    """Generate student PDF reports."""
    try:
        from forma.report_generator import StudentReportGenerator

        generator = StudentReportGenerator()
        reports_dir = os.path.join(output_dir, "res_lvl4", "reports")
        paths = generator.generate_all_reports(
            counseling_data=counseling,
            config_data=config_data,
            output_dir=reports_dir,
        )
        print(f"[pipeline] Generated {len(paths)} PDF reports in {reports_dir}")
    except Exception as exc:
        print(f"[pipeline] Report generation failed: {exc}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_eval_config(path: str) -> dict:
    """Load an eval-config YAML and return a flat options dict.

    The YAML keys map directly to ``run_evaluation_pipeline()`` kwargs.
    Paths inside the YAML are resolved relative to the YAML file's
    directory so the user can write relative paths naturally.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve relative paths against the YAML file's directory
    base = os.path.dirname(os.path.abspath(path))
    for key in ("config", "responses", "output", "lecture_transcript",
                "longitudinal_store"):
        val = cfg.get(key)
        if val and not os.path.isabs(val):
            cfg[key] = os.path.join(base, val)

    return cfg


def main() -> None:
    """Parse arguments and run the evaluation pipeline.

    Supports two usage modes:

    1. ``forma-eval --eval-config eval_w1_A.yaml``
       — all options in a single YAML file.
    2. ``forma-eval --config ... --responses ... --output ...``
       — traditional CLI flags (CLI flags override eval-config values).
    """
    # Suppress noisy warnings from dependencies
    warnings.filterwarnings("ignore", message="overflow", category=RuntimeWarning)
    # HuggingFace Hub: suppress "unauthenticated requests" warning
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Suppress sentence-transformers / RobertaModel LOAD REPORT noise
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Multi-layer concept evaluation pipeline (v2)"
    )
    parser.add_argument(
        "--eval-config",
        default=None,
        help="평가 환경설정 YAML (모든 옵션을 파일 하나에 지정)",
    )
    parser.add_argument(
        "--config", default=None, help="Exam YAML config path"
    )
    parser.add_argument(
        "--responses", default=None, help="Student responses YAML path"
    )
    parser.add_argument(
        "--output", default=None, help="Output directory"
    )
    parser.add_argument(
        "--api-key", default=None, help="LLM API key (overrides env var)"
    )
    parser.add_argument(
        "--provider", default=None, help="LLM provider: gemini | anthropic"
    )
    parser.add_argument(
        "--model", default=None, help="LLM model ID override"
    )
    parser.add_argument(
        "--skip-feedback", action="store_true", default=None,
        help="Skip feedback generation",
    )
    parser.add_argument(
        "--skip-llm", action="store_true", default=None,
        help="Deprecated: use --skip-feedback instead",
    )
    parser.add_argument(
        "--skip-graph", action="store_true", default=None,
        help="Skip triplet extraction and graph comparison",
    )
    parser.add_argument(
        "--skip-stats", action="store_true", default=None,
        help="Skip Layer 3 statistical analysis",
    )
    parser.add_argument(
        "--lecture-transcript", default=None,
        help="Path to lecture transcript file",
    )
    parser.add_argument(
        "--longitudinal-store", default=None,
        help="Path to longitudinal data store",
    )
    parser.add_argument(
        "--generate-reports", action="store_true", default=None,
        help="Generate student PDF reports",
    )
    parser.add_argument(
        "--questions-used", nargs="+", type=int, default=None,
        help="출제 문항의 exam sn 번호를 q 순서대로 (예: 1 3)",
    )
    parser.add_argument(
        "--n-calls", type=int, default=None,
        help="LLM 호출 횟수 (기본 3, 비용 절감시 2)",
    )
    args = parser.parse_args()

    # --- Merge eval-config YAML with CLI flags (CLI wins) ---
    ecfg: dict = {}
    if args.eval_config:
        ecfg = _load_eval_config(args.eval_config)

    def _resolve(cli_val, yaml_key, default=None):
        """CLI flag > eval-config YAML > default."""
        if cli_val is not None:
            return cli_val
        return ecfg.get(yaml_key, default)

    config_path = _resolve(args.config, "config")
    responses_path = _resolve(args.responses, "responses")
    output_dir = _resolve(args.output, "output")

    if not config_path or not responses_path or not output_dir:
        parser.error(
            "--eval-config 또는 --config/--responses/--output 을 지정하세요."
        )

    run_evaluation_pipeline(
        config_path=config_path,
        responses_path=responses_path,
        output_dir=output_dir,
        api_key=_resolve(args.api_key, "api_key"),
        skip_llm=_resolve(args.skip_llm, "skip_llm", False),
        skip_feedback=_resolve(args.skip_feedback, "skip_feedback", False),
        skip_graph=_resolve(args.skip_graph, "skip_graph", False),
        skip_statistical=_resolve(args.skip_stats, "skip_stats", False),
        provider=_resolve(args.provider, "provider", "gemini"),
        model=_resolve(args.model, "model"),
        lecture_transcript=_resolve(args.lecture_transcript, "lecture_transcript"),
        longitudinal_store=_resolve(args.longitudinal_store, "longitudinal_store"),
        generate_reports=_resolve(args.generate_reports, "generate_reports", False),
        questions_used=_resolve(args.questions_used, "questions_used"),
        n_calls=_resolve(args.n_calls, "n_calls", 3),
    )


if __name__ == "__main__":
    main()
