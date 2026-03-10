"""CLI entry point for forma-train-grade -- train semester grade prediction model.

Usage::

    forma-train-grade --store STORE_PATH --grades GRADES_PATH --output MODEL_PATH
        [--semester LABEL] [--min-students INT] [--verbose] [--no-config]

Exit codes:
    0 -- success
    1 -- data/input error
    2 -- file error
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for forma-train-grade."""
    parser = argparse.ArgumentParser(
        prog="forma-train-grade",
        description="학기말 성적 예측 모델 학습",
    )
    parser.add_argument(
        "--store", required=True,
        help="종단 저장소 YAML 파일 경로",
    )
    parser.add_argument(
        "--grades", required=True,
        help="성적 매핑 YAML 파일 경로",
    )
    parser.add_argument(
        "--output", required=True,
        help="출력 모델 파일 경로 (.pkl)",
    )
    parser.add_argument(
        "--semester", default=None,
        help="학습에 사용할 학기 라벨 (생략 시 마지막 학기)",
    )
    parser.add_argument(
        "--min-students", type=int, default=10, dest="min_students",
        help="최소 학생 수 (기본값: 10)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="상세 로그 출력",
    )
    parser.add_argument(
        "--no-config", action="store_true", default=False, dest="no_config",
        help="forma.yaml 설정 파일 무시",
    )
    return parser


def main(argv=None) -> None:
    """Entry point for forma-train-grade CLI.

    Args:
        argv: Command-line arguments (for testing). None uses sys.argv.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Apply project config
    if not args.no_config:
        from forma.project_config import apply_project_config
        raw_argv = argv if argv is not None else sys.argv[1:]
        apply_project_config(args, argv=raw_argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Validate input files
    if not os.path.isfile(args.store):
        print(f"Error: Store file not found: {args.store}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.grades):
        print(f"Error: Grades file not found: {args.grades}", file=sys.stderr)
        sys.exit(1)

    # Load store
    from forma.longitudinal_store import LongitudinalStore

    store = LongitudinalStore(args.store)
    store.load()

    all_records = store.get_all_records()
    if not all_records:
        print("Error: Store contains no records", file=sys.stderr)
        sys.exit(1)

    all_weeks = sorted({r.week for r in all_records})

    # Load grade mapping
    from forma.grade_predictor import (
        GRADE_FEATURE_NAMES,
        GRADE_ORDINAL_MAP,
        GradeFeatureExtractor,
        GradePredictor,
        load_grade_mapping,
        save_grade_model,
    )

    try:
        grade_mapping = load_grade_mapping(args.grades)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not grade_mapping:
        print("Error: Grade mapping is empty", file=sys.stderr)
        sys.exit(1)

    # Select semester
    if args.semester:
        if args.semester not in grade_mapping:
            print(
                f"Error: Semester '{args.semester}' not found in grades file",
                file=sys.stderr,
            )
            sys.exit(1)
        semester_grades = grade_mapping[args.semester]
    else:
        # Use the last semester
        last_semester = list(grade_mapping.keys())[-1]
        semester_grades = grade_mapping[last_semester]
        logger.info("Using last semester: %s", last_semester)

    # Build grade history (all semesters as ordinal lists per student)
    grade_history: dict[str, list[int]] = {}
    for sem_label, sg in grade_mapping.items():
        for sid, grade in sg.items():
            if sid not in grade_history:
                grade_history[sid] = []
            grade_history[sid].append(GRADE_ORDINAL_MAP[grade])

    # Extract features
    extractor = GradeFeatureExtractor()
    matrix, feature_names, student_ids = extractor.extract(
        store, all_weeks, grade_history=grade_history,
    )

    # Build labels: only students that have grades in the selected semester
    import numpy as np

    valid_indices = []
    labels_list = []
    for i, sid in enumerate(student_ids):
        if sid in semester_grades:
            valid_indices.append(i)
            labels_list.append(GRADE_ORDINAL_MAP[semester_grades[sid]])

    if len(valid_indices) < args.min_students:
        print(
            f"Error: Insufficient students with grades: "
            f"{len(valid_indices)} < {args.min_students}",
            file=sys.stderr,
        )
        sys.exit(1)

    X_train = matrix[valid_indices]
    labels = np.array(labels_list)

    n_students = len(valid_indices)
    n_unique = len(np.unique(labels))
    print(
        f"Training data: {n_students} students, {len(all_weeks)} weeks, "
        f"{len(feature_names)} features, {n_unique} grade classes"
    )

    # Train
    predictor = GradePredictor()
    try:
        model = predictor.train(
            X_train, labels, feature_names,
            min_students=args.min_students,
            n_weeks=len(all_weeks),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  CV accuracy: {model.cv_score:.3f}")

    # Save model
    try:
        save_grade_model(model, args.output)
    except OSError as exc:
        print(f"Error: Failed to save model: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Grade model saved: {args.output}")


if __name__ == "__main__":
    main()
