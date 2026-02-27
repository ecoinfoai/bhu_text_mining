"""Multi-layer concept evaluation pipeline orchestration.

Runs all four evaluation layers in sequence for every (student, question)
pair in the input YAML and writes counseling + technical reports.

Usage:
    uv run python pipeline_evaluation.py \\
        --config exams/Ch01_서론_FormativeTest.yaml \\
        --responses data/responses.yaml \\
        --output results/ch01/

Optional flags:
    --skip-llm      Skip Layer 2 LLM evaluation (saves API cost)
    --skip-stats    Skip Layer 3 statistical analysis
    --api-key KEY   Override ANTHROPIC_API_KEY environment variable
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import yaml

from src.concept_checker import check_all_concepts
from src.ensemble_scorer import EnsembleScorer
from src.evaluation_io import (
    extract_student_responses,
    load_evaluation_yaml,
    save_evaluation_yaml,
)
from src.evaluation_types import (
    AggregatedLLMResult,
    EnsembleResult,
    GraphMetricResult,
    StatisticalResult,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_question_config(
    config_data: dict, question_sn: int
) -> dict:
    """Extract a single question's config by serial number.

    Args:
        config_data: Parsed exam YAML dict.
        question_sn: Question serial number (1-based).

    Returns:
        Question config dict with keys: question, model_answer, keywords,
        rubric, etc.

    Raises:
        KeyError: If the question_sn is not found.
    """
    questions = config_data.get("questions", [])
    for q in questions:
        if q.get("sn") == question_sn:
            return q
    raise KeyError(
        f"Question sn={question_sn} not found in exam config. "
        f"Available: {[q.get('sn') for q in questions]}"
    )


def _run_layer1(
    student_responses: dict[str, dict[int, str]],
    config_data: dict,
) -> dict[str, dict[int, list]]:
    """Run Layer 1 concept checking for all students and questions.

    Returns nested dict: student_id → question_sn → [ConceptMatchResult].
    """
    results: dict[str, dict[int, list]] = {}
    questions = config_data.get("questions", [])

    for q in questions:
        qsn: int = q["sn"]
        concepts: list[str] = q.get("keywords", [])
        if not concepts:
            continue

        for student_id, responses in student_responses.items():
            text = responses.get(qsn, "")
            if not text:
                continue
            cr = check_all_concepts(
                student_text=text,
                student_id=student_id,
                question_sn=qsn,
                concepts=concepts,
            )
            results.setdefault(student_id, {})[qsn] = cr

    return results


def _run_layer2(
    student_responses: dict[str, dict[int, str]],
    config_data: dict,
    api_key: Optional[str],
) -> dict[str, dict[int, AggregatedLLMResult]]:
    """Run Layer 2 LLM evaluation for all students and questions.

    Returns nested dict: student_id → question_sn → AggregatedLLMResult.
    """
    from src.llm_evaluator import LLMEvaluator

    evaluator = LLMEvaluator(api_key=api_key)
    results: dict[str, dict[int, AggregatedLLMResult]] = {}
    questions = config_data.get("questions", [])

    for q in questions:
        qsn: int = q["sn"]
        rubric = q.get("rubric", {})

        for student_id, responses in student_responses.items():
            text = responses.get(qsn, "")
            if not text:
                continue
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

    return results


def _build_counseling_summary(
    ensemble_results: dict[str, dict[int, EnsembleResult]],
    config_data: dict,
    llm_results: Optional[dict[str, dict[int, AggregatedLLMResult]]],
) -> dict:
    """Build professor-facing counseling summary (no technical stats).

    Args:
        ensemble_results: Computed ensemble scores.
        config_data: Exam configuration.
        llm_results: LLM evaluation results for misconceptions.

    Returns:
        Dict ready for YAML serialisation.
    """
    summary: dict = {"students": []}
    questions = config_data.get("questions", [])

    for student_id, q_results in ensemble_results.items():
        student_entry: dict = {"student_id": student_id, "questions": []}
        for qsn, er in sorted(q_results.items()):
            q_cfg = next(
                (q for q in questions if q.get("sn") == qsn), {}
            )
            rubric = q_cfg.get("rubric", {})
            support = q_cfg.get("support", {})
            level_key = er.understanding_level.lower()

            misconceptions: list[str] = []
            if llm_results and student_id in llm_results:
                lr = llm_results[student_id].get(qsn)
                if lr:
                    misconceptions = lr.misconceptions

            q_entry = {
                "question_sn": qsn,
                "understanding_level": er.understanding_level,
                "concept_coverage": er.component_scores.get(
                    "concept_coverage", 0.0
                ),
                "support_guidance": support.get(level_key, ""),
                "misconceptions": misconceptions,
            }
            student_entry["questions"].append(q_entry)
        summary["students"].append(student_entry)

    return summary


def _build_technical_report(
    ensemble_results: dict[str, dict[int, EnsembleResult]],
    layer1_results: dict[str, dict[int, list]],
    llm_results: Optional[dict[str, dict[int, AggregatedLLMResult]]],
    stat_results: Optional[dict[str, dict[int, StatisticalResult]]],
) -> dict:
    """Build technical report with all parameters, SE, and fit statistics.

    Args:
        ensemble_results: Computed ensemble scores.
        layer1_results: Layer 1 concept match results.
        llm_results: Layer 2 LLM evaluation results.
        stat_results: Layer 3 statistical results.

    Returns:
        Dict ready for YAML serialisation.
    """
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

            # Layer 2 detail
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

            student_entry["questions"].append(q_entry)
        report["students"].append(student_entry)

    return report


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_evaluation_pipeline(
    config_path: str,
    responses_path: str,
    output_dir: str,
    api_key: Optional[str] = None,
    skip_llm: bool = False,
    skip_statistical: bool = False,
) -> None:
    """Run the full multi-layer concept evaluation pipeline.

    Loads exam config and student responses, runs Layers 1–4 in sequence,
    and writes two YAML output files per run:
    - ``counseling_summary.yaml``: Plain-language professor guidance.
    - ``technical_report.yaml``: Full statistical output.

    Args:
        config_path: Path to exam YAML config file.
        responses_path: Path to student responses YAML file.
        output_dir: Directory for output files (created if absent).
        api_key: Anthropic API key (overrides env var).
        skip_llm: If True, skip Layer 2 LLM evaluation.
        skip_statistical: If True, skip Layer 3 statistical analysis.

    Raises:
        FileNotFoundError: If config or responses file is missing.
    """
    config_data = load_evaluation_yaml(config_path)
    responses_data = load_evaluation_yaml(responses_path)
    student_responses = extract_student_responses(responses_data)

    print(
        f"[pipeline] {len(student_responses)} students, "
        f"{len(config_data.get('questions', []))} questions"
    )

    # --- Layer 1: Concept checker ---
    print("[pipeline] Layer 1: concept checking …")
    layer1_results = _run_layer1(student_responses, config_data)

    # --- Layer 2: LLM evaluation ---
    llm_results: Optional[dict] = None
    if not skip_llm:
        print("[pipeline] Layer 2: LLM evaluation (3-call protocol) …")
        llm_results = _run_layer2(student_responses, config_data, api_key)

    # --- Layer 3: Statistical analysis (optional) ---
    stat_results: Optional[dict] = None
    if not skip_statistical:
        from src.statistical_analysis import (
            RaschAnalyzer,
            compute_concept_matrix,
        )
        import numpy as np

        stat_results = {}
        for q in config_data.get("questions", []):
            qsn = q["sn"]
            concepts = q.get("keywords", [])
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
                for i, sid in enumerate(sids):
                    stat_results.setdefault(sid, {})[qsn] = StatisticalResult(
                        student_id=sid,
                        question_sn=qsn,
                        rasch_theta=float(thetas[i]),
                        rasch_theta_se=float(ses[i]),
                    )
            except Exception as exc:
                print(f"[pipeline] Rasch failed for q{qsn}: {exc}")

    # --- Layer 4: Ensemble ---
    print("[pipeline] Layer 4: ensemble scoring …")
    scorer = EnsembleScorer()
    ensemble_results: dict[str, dict[int, EnsembleResult]] = {}

    for student_id, q_responses in student_responses.items():
        for qsn in q_responses:
            cr = (layer1_results.get(student_id) or {}).get(qsn, [])
            llm = (llm_results.get(student_id) or {}).get(qsn) if llm_results else None
            stat = (stat_results.get(student_id) or {}).get(qsn) if stat_results else None
            er = scorer.compute_score(
                concept_results=cr,
                llm_result=llm,
                statistical_result=stat,
                graph_result=None,
                bertscore_f1=None,
                student_id=student_id,
                question_sn=qsn,
            )
            ensemble_results.setdefault(student_id, {})[qsn] = er

    # --- Output ---
    counseling = _build_counseling_summary(
        ensemble_results, config_data, llm_results
    )
    technical = _build_technical_report(
        ensemble_results, layer1_results, llm_results, stat_results
    )

    save_evaluation_yaml(counseling, os.path.join(output_dir, "counseling_summary.yaml"))
    save_evaluation_yaml(technical, os.path.join(output_dir, "technical_report.yaml"))
    print(f"[pipeline] Done. Results written to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Multi-layer concept evaluation pipeline"
    )
    parser.add_argument(
        "--config", required=True, help="Exam YAML config path"
    )
    parser.add_argument(
        "--responses", required=True, help="Student responses YAML path"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory"
    )
    parser.add_argument(
        "--api-key", default=None, help="Anthropic API key (overrides env var)"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip Layer 2 LLM evaluation",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip Layer 3 statistical analysis",
    )
    args = parser.parse_args()

    run_evaluation_pipeline(
        config_path=args.config,
        responses_path=args.responses,
        output_dir=args.output,
        api_key=args.api_key,
        skip_llm=args.skip_llm,
        skip_statistical=args.skip_stats,
    )


if __name__ == "__main__":
    main()
