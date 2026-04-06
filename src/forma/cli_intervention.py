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
        description="Manage intervention activity records",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        default=False,
        dest="no_config",
        help="Skip forma.yaml config file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # --- add subcommand ---
    add_parser = subparsers.add_parser("add", help="Add intervention record")
    add_parser.add_argument(
        "--store",
        required=True,
        help="Intervention record YAML file path",
    )
    add_parser.add_argument(
        "--student",
        required=True,
        help="Student ID",
    )
    add_parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="Week number",
    )
    add_parser.add_argument(
        "--type",
        required=True,
        help="Intervention type",
    )
    add_parser.add_argument(
        "--description",
        default="",
        help="Intervention description",
    )
    add_parser.add_argument(
        "--recorded-by",
        default=None,
        dest="recorded_by",
        help="Recorded by",
    )
    add_parser.add_argument(
        "--follow-up-week",
        type=int,
        default=None,
        dest="follow_up_week",
        help="Follow-up week",
    )

    # --- list subcommand ---
    list_parser = subparsers.add_parser("list", help="List intervention records")
    list_parser.add_argument(
        "--store",
        required=True,
        help="Intervention record YAML file path",
    )
    list_parser.add_argument(
        "--student",
        default=None,
        help="Student ID filter",
    )
    list_parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Week filter",
    )

    # --- update subcommand ---
    update_parser = subparsers.add_parser("update", help="Update intervention outcome")
    update_parser.add_argument(
        "--store",
        required=True,
        help="Intervention record YAML file path",
    )
    update_parser.add_argument(
        "--id",
        type=int,
        required=True,
        help="Intervention record ID",
    )
    update_parser.add_argument(
        "--outcome",
        required=True,
        help="Outcome",
    )

    return parser


def _cmd_add(args: argparse.Namespace) -> None:
    """Handle the `add` subcommand."""
    from forma.intervention_store import INTERVENTION_TYPES, InterventionLog

    if args.week < 1:
        print(
            f"Error: Week number must be >= 1 (got: {args.week})",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.type not in INTERVENTION_TYPES:
        print(
            f"Error: Invalid intervention type '{args.type}'. Allowed: {', '.join(INTERVENTION_TYPES)}",
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

    print(f"Intervention record added (ID: {record_id})")


def _cmd_list(args: argparse.Namespace) -> None:
    """Handle the `list` subcommand."""
    from forma.intervention_store import InterventionLog

    if not os.path.exists(args.store):
        print(f"Error: Record file not found: {args.store}", file=sys.stderr)
        sys.exit(1)

    log = InterventionLog(args.store)
    log.load()

    records = log.get_records(student_id=args.student, week=args.week)

    # Sort by week (chronological), then by ID
    records.sort(key=lambda r: (r.week, r.id))

    if not records:
        print("No records found (0 records)")
        return

    print(f"Query result: {len(records)} records")
    print(f"{'ID':>4}  {'Student':<8}  {'Week':>4}  {'Type':<8}  {'Description':<20}  {'Outcome':<6}")
    print("-" * 72)
    for r in records:
        outcome_str = r.outcome if r.outcome else "-"
        desc = r.description[:20] if r.description else ""
        print(f"{r.id:>4}  {r.student_id:<8}  {r.week:>4}  {r.intervention_type:<8}  {desc:<20}  {outcome_str:<6}")


def _cmd_update(args: argparse.Namespace) -> None:
    """Handle the `update` subcommand."""
    from forma.intervention_store import InterventionLog

    if not os.path.exists(args.store):
        print(f"Error: Record file not found: {args.store}", file=sys.stderr)
        sys.exit(1)

    # Strict outcome validation at CLI level
    if args.outcome not in VALID_OUTCOMES:
        print(
            f"Error: Invalid outcome '{args.outcome}'. Allowed: {', '.join(VALID_OUTCOMES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    log = InterventionLog(args.store)
    log.load()

    success = log.update_outcome(args.id, args.outcome)
    if not success:
        print(f"Error: ID {args.id} not found.", file=sys.stderr)
        sys.exit(1)

    log.save()
    print(f"Record ID {args.id} outcome updated to '{args.outcome}'.")


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
