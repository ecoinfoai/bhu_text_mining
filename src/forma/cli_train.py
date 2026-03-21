"""CLI entry point for forma-train — train drop risk prediction model.

Usage::

    forma-train --store STORE_PATH --output MODEL_PATH
                [--threshold FLOAT] [--min-weeks INT] [--min-students INT] [--verbose]

Exit codes:
    0 — success
    1 — insufficient data or input error
    2 — file error
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for forma-train."""
    parser = argparse.ArgumentParser(
        prog="forma-train",
        description="Train drop risk prediction model",
    )
    parser.add_argument(
        "--store", required=True,
        help="Longitudinal store YAML file path",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output model file path (.pkl)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.45,
        help="Drop score threshold (default: 0.45)",
    )
    parser.add_argument(
        "--min-weeks", type=int, default=3, dest="min_weeks",
        help="Minimum number of weeks (default: 3)",
    )
    parser.add_argument(
        "--min-students", type=int, default=10, dest="min_students",
        help="Minimum number of students (default: 10)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main() -> None:
    """Entry point for forma-train CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Validate store file exists
    if not os.path.isfile(args.store):
        print(f"Error: Store file not found: {args.store}", file=sys.stderr)
        sys.exit(1)

    # Load store
    from forma.longitudinal_store import LongitudinalStore

    store = LongitudinalStore(args.store)
    store.load()

    all_records = store.get_all_records()
    if not all_records:
        print("Error: Store contains no records", file=sys.stderr)
        sys.exit(1)

    # Determine weeks
    all_weeks = sorted({r.week for r in all_records})
    n_weeks = len(all_weeks)

    if n_weeks < args.min_weeks:
        print(
            f"Error: Insufficient weeks: {n_weeks} < {args.min_weeks}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract features
    from forma.risk_predictor import FeatureExtractor, RiskPredictor, save_model

    extractor = FeatureExtractor()
    matrix, feature_names, student_ids = extractor.extract(store, all_weeks)

    n_students = len(student_ids)
    if n_students < args.min_students:
        print(
            f"Error: Insufficient students: {n_students} < {args.min_students}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Label: final week score < threshold → drop
    import numpy as np

    last_score_idx = feature_names.index("last_score")
    labels = (matrix[:, last_score_idx] < args.threshold).astype(int)

    n_drop = int(np.sum(labels))
    n_pass = n_students - n_drop
    print(f"Training data: {n_students} students, {n_weeks} weeks, {len(feature_names)} features")
    print(f"  Drop: {n_drop}, Pass: {n_pass}")

    # Train
    predictor = RiskPredictor()
    try:
        model = predictor.train(
            matrix, labels, feature_names,
            min_students=args.min_students,
            n_weeks=n_weeks,
            target_threshold=args.threshold,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  CV accuracy: {model.cv_score:.3f}")

    # Save model
    try:
        save_model(model, args.output)
    except OSError as exc:
        print(f"Error: Failed to save model: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Model saved: {args.output}")


if __name__ == "__main__":
    main()
