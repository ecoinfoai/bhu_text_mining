"""CLI entry point for forma-intervention -- manage intervention activity records.

Usage::

    forma-intervention [--no-config] [--verbose] add --store STORE --student ID --week N --type TYPE
        [--description TEXT] [--recorded-by NAME] [--follow-up-week N]

    forma-intervention [--no-config] [--verbose] list --store STORE
        [--student ID] [--week N]

    forma-intervention [--no-config] [--verbose] update --store STORE --id N --outcome OUTCOME

Subcommands:
    add     Add a new intervention record.
    list    List intervention records with optional filters.
    update  Update the outcome of an existing intervention record.

Exit codes:
    0 -- success
    1 -- input/data error
    2 -- file error
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Valid outcome values for the update subcommand (strict CLI validation)
VALID_OUTCOMES = ("개선", "유지", "악화")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with add/list/update subcommands."""
    parser = argparse.ArgumentParser(
        prog="forma-intervention",
        description="개입 활동 기록 관리",
    )
    parser.add_argument(
        "--no-config", action="store_true", default=False, dest="no_config",
        help="forma.yaml 설정 파일 무시",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="상세 로그 출력",
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # --- add subcommand ---
    add_parser = subparsers.add_parser("add", help="개입 기록 추가")
    add_parser.add_argument(
        "--store", required=True,
        help="개입 기록 YAML 파일 경로",
    )
    add_parser.add_argument(
        "--student", required=True,
        help="학생 ID",
    )
    add_parser.add_argument(
        "--week", type=int, required=True,
        help="주차 번호",
    )
    add_parser.add_argument(
        "--type", required=True,
        help="개입 유형 (면담/보충학습/과제부여/멘토링/기타)",
    )
    add_parser.add_argument(
        "--description", default="",
        help="개입 내용 설명",
    )
    add_parser.add_argument(
        "--recorded-by", default=None, dest="recorded_by",
        help="기록자 이름",
    )
    add_parser.add_argument(
        "--follow-up-week", type=int, default=None, dest="follow_up_week",
        help="후속 조치 주차",
    )

    # --- list subcommand ---
    list_parser = subparsers.add_parser("list", help="개입 기록 조회")
    list_parser.add_argument(
        "--store", required=True,
        help="개입 기록 YAML 파일 경로",
    )
    list_parser.add_argument(
        "--student", default=None,
        help="학생 ID 필터",
    )
    list_parser.add_argument(
        "--week", type=int, default=None,
        help="주차 필터",
    )

    # --- update subcommand ---
    update_parser = subparsers.add_parser("update", help="개입 결과 업데이트")
    update_parser.add_argument(
        "--store", required=True,
        help="개입 기록 YAML 파일 경로",
    )
    update_parser.add_argument(
        "--id", type=int, required=True,
        help="개입 기록 ID",
    )
    update_parser.add_argument(
        "--outcome", required=True,
        help="결과 (개선/유지/악화)",
    )

    return parser


def _cmd_add(args: argparse.Namespace) -> None:
    """Handle the `add` subcommand."""
    from forma.intervention_store import INTERVENTION_TYPES, InterventionLog

    if args.week < 1:
        print(
            f"Error: 주차 번호는 1 이상이어야 합니다 (입력값: {args.week})",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.type not in INTERVENTION_TYPES:
        print(
            f"Error: 유효하지 않은 개입 유형 '{args.type}'. "
            f"허용: {', '.join(INTERVENTION_TYPES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    log = InterventionLog(args.store)
    log.load()

    record_id = log.add_record(
        student_id=args.student,
        week=args.week,
        intervention_type=args.type,
        description=args.description,
        recorded_by=args.recorded_by,
        follow_up_week=args.follow_up_week,
    )
    log.save()

    print(f"개입 기록 추가 완료 (ID: {record_id})")


def _cmd_list(args: argparse.Namespace) -> None:
    """Handle the `list` subcommand."""
    from forma.intervention_store import InterventionLog

    if not os.path.exists(args.store):
        print(f"Error: 기록 파일을 찾을 수 없습니다: {args.store}", file=sys.stderr)
        sys.exit(1)

    log = InterventionLog(args.store)
    log.load()

    records = log.get_records(student_id=args.student, week=args.week)

    # Sort by week (chronological), then by ID
    records.sort(key=lambda r: (r.week, r.id))

    if not records:
        print("기록 없음 (0건)")
        return

    print(f"조회 결과: {len(records)}건")
    print(f"{'ID':>4}  {'학생':<8}  {'주차':>4}  {'유형':<8}  {'설명':<20}  {'결과':<6}")
    print("-" * 72)
    for r in records:
        outcome_str = r.outcome if r.outcome else "-"
        desc = r.description[:20] if r.description else ""
        print(
            f"{r.id:>4}  {r.student_id:<8}  {r.week:>4}  "
            f"{r.intervention_type:<8}  {desc:<20}  {outcome_str:<6}"
        )


def _cmd_update(args: argparse.Namespace) -> None:
    """Handle the `update` subcommand."""
    from forma.intervention_store import InterventionLog

    if not os.path.exists(args.store):
        print(f"Error: 기록 파일을 찾을 수 없습니다: {args.store}", file=sys.stderr)
        sys.exit(1)

    # Strict outcome validation at CLI level
    if args.outcome not in VALID_OUTCOMES:
        print(
            f"Error: 유효하지 않은 결과 '{args.outcome}'. "
            f"허용: {', '.join(VALID_OUTCOMES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    log = InterventionLog(args.store)
    log.load()

    success = log.update_outcome(args.id, args.outcome)
    if not success:
        print(f"Error: ID {args.id}에 해당하는 기록을 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    log.save()
    print(f"기록 ID {args.id}의 결과가 '{args.outcome}'(으)로 업데이트되었습니다.")


def main(argv=None) -> None:
    """Entry point for forma-intervention CLI.

    Args:
        argv: Command-line arguments (for testing). None uses sys.argv.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Apply project config (three-layer merge)
    if not getattr(args, "no_config", False):
        from forma.project_config import apply_project_config
        raw_argv = argv if argv is not None else sys.argv[1:]
        apply_project_config(args, argv=raw_argv)

    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    if args.subcommand == "add":
        _cmd_add(args)
    elif args.subcommand == "list":
        _cmd_list(args)
    elif args.subcommand == "update":
        _cmd_update(args)


if __name__ == "__main__":
    main()
