"""Backfill longitudinal store from existing evaluation results.

Usage:
    forma backfill longitudinal \
        --eval-dir eval_A --eval-dir eval_B --eval-dir eval_C \
        --store longitudinal.yaml \
        --week 1 \
        --exam-file Ch01_서론_FormativeTest.yaml \
        [--responses final_A.yaml --responses final_B.yaml ...]
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml


def _load_ensemble_results(eval_dir: str) -> dict[str, dict[int, object]]:
    """Load ensemble_results.yaml and return {student_id: {qsn: pseudo-EnsembleResult}}.

    Returns a dict of SimpleNamespace-like objects with .component_scores and
    .understanding_level, matching what snapshot_from_evaluation expects.
    """
    path = os.path.join(eval_dir, "res_lvl4", "ensemble_results.yaml")
    if not os.path.exists(path):
        print(f"[backfill] WARNING: {path} not found, skipping", file=sys.stderr)
        return {}

    with open(path) as f:
        data = yaml.safe_load(f)

    results: dict[str, dict[int, object]] = {}
    for student in data.get("students", []):
        sid = student["student_id"]
        results[sid] = {}
        for q in student.get("questions", []):
            qsn = int(q["question_sn"])

            class _ER:
                def __init__(self, component_scores: dict, understanding_level: str):
                    self.component_scores = component_scores
                    self.understanding_level = understanding_level

            results[sid][qsn] = _ER(
                component_scores=dict(q.get("component_scores", {})),
                understanding_level=q.get("understanding_level", ""),
            )
    return results


def _load_concept_results(eval_dir: str) -> list:
    """Load concept_results.yaml and return list of pseudo-ConceptMatchResult."""
    path = os.path.join(eval_dir, "res_lvl1", "concept_results.yaml")
    if not os.path.exists(path):
        return []

    with open(path) as f:
        data = yaml.safe_load(f)

    class _CMR:
        def __init__(self, student_id: str, question_sn: int, concept: str, is_present: bool):
            self.student_id = student_id
            self.question_sn = question_sn
            self.concept = concept
            self.is_present = is_present

    results = []
    for student in data.get("students", []):
        sid = student["student_id"]
        for q in student.get("questions", []):
            qsn = int(q["question_sn"])
            for c in q.get("concepts", []):
                results.append(_CMR(
                    student_id=sid,
                    question_sn=qsn,
                    concept=c["concept"],
                    is_present=bool(c.get("is_present", False)),
                ))
    return results


def _extract_ocr_confidence(
    responses: list[dict],
) -> dict[str, dict[int, dict[str, float | None]]]:
    """Extract OCR confidence from join output list."""
    result: dict[str, dict[int, dict[str, float | None]]] = {}
    for entry in responses:
        sid = entry.get("student_id")
        qn = entry.get("q_num")
        mean = entry.get("ocr_confidence_mean")
        _min = entry.get("ocr_confidence_min")
        if sid is not None and qn is not None and mean is not None:
            result.setdefault(sid, {})[qn] = {"mean": mean, "min": _min}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill longitudinal store from existing eval results",
    )
    parser.add_argument(
        "--eval-dir",
        action="append",
        required=True,
        help="Evaluation output directory (repeatable, e.g. --eval-dir eval_A --eval-dir eval_B)",
    )
    parser.add_argument(
        "--store",
        required=True,
        help="Path to longitudinal store YAML (created if not exists)",
    )
    parser.add_argument("--week", type=int, required=True, help="Week number")
    parser.add_argument(
        "--exam-file",
        default="",
        help="Exam config filename for metadata",
    )
    parser.add_argument(
        "--responses",
        action="append",
        default=None,
        help="Path to final_X.yaml for OCR confidence (repeatable, optional)",
    )
    args = parser.parse_args()

    from forma.longitudinal_store import LongitudinalStore, snapshot_from_evaluation

    store = LongitudinalStore(args.store)
    if os.path.exists(args.store):
        store.load()

    total_students = 0

    for eval_dir in args.eval_dir:
        if not os.path.isdir(eval_dir):
            print(f"[backfill] WARNING: {eval_dir} is not a directory, skipping", file=sys.stderr)
            continue

        ensemble_results = _load_ensemble_results(eval_dir)
        layer1_results = _load_concept_results(eval_dir)

        if not ensemble_results:
            print(f"[backfill] No ensemble results in {eval_dir}, skipping")
            continue

        n = len(ensemble_results)
        total_students += n
        print(f"[backfill] {eval_dir}: {n} students loaded")

        snapshot_from_evaluation(
            store=store,
            ensemble_results=ensemble_results,
            graph_metric_results={},
            graph_comparison_results={},
            layer1_results=layer1_results,
            week=args.week,
            exam_file=args.exam_file,
        )

    # OCR confidence — load responses and patch records
    if args.responses:
        all_responses: list[dict] = []
        for resp_path in args.responses:
            if not os.path.exists(resp_path):
                print(f"[backfill] WARNING: {resp_path} not found, skipping", file=sys.stderr)
                continue
            with open(resp_path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, list):
                all_responses.extend(data)

        if all_responses:
            ocr_conf = _extract_ocr_confidence(all_responses)
            patched = 0
            for key, record_dict in store._records.items():
                if record_dict["week"] != args.week:
                    continue
                sid = record_dict["student_id"]
                qsn = record_dict["question_sn"]
                conf = (ocr_conf.get(sid) or {}).get(qsn)
                if conf:
                    record_dict["ocr_confidence_mean"] = conf["mean"]
                    record_dict["ocr_confidence_min"] = conf["min"]
                    patched += 1
            if patched:
                print(f"[backfill] OCR confidence patched for {patched} records")

    store.save()
    print(f"[backfill] Done. {total_students} students saved to {args.store}")


if __name__ == "__main__":
    main()
