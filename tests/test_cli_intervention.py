"""Tests for cli_intervention.py — forma-intervention CLI entry point.

T011: Tests for `forma-intervention add` (argparse, type validation, success output with ID)
T012: Tests for `forma-intervention list` (filter student/week, chronological sort, empty result)
T013: Tests for `forma-intervention update` (--id, outcome validation 개선/유지/악화, nonexistent ID error)

Covers FR-004, FR-005, FR-006, FR-035.
"""

from __future__ import annotations

import os

import pytest
import yaml


# ---------------------------------------------------------------------------
# T011: Parser tests — subcommands and arguments
# ---------------------------------------------------------------------------


class TestCliInterventionParser:
    """Tests for argument parsing with subcommands."""

    def test_add_subcommand_required_args(self):
        """add subcommand requires --store, --student, --week, --type."""
        from forma.cli_intervention import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "add",
                "--store",
                "log.yaml",
                "--student",
                "s001",
                "--week",
                "2",
                "--type",
                "면담",
            ]
        )
        assert args.subcommand == "add"
        assert args.store == "log.yaml"
        assert args.student == "s001"
        assert args.week == 2
        assert args.type == "면담"

    def test_add_subcommand_optional_args(self):
        """add subcommand accepts --description, --recorded-by, --follow-up-week."""
        from forma.cli_intervention import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "add",
                "--store",
                "log.yaml",
                "--student",
                "s001",
                "--week",
                "2",
                "--type",
                "면담",
                "--description",
                "학습 상담",
                "--recorded-by",
                "prof_kim",
                "--follow-up-week",
                "4",
            ]
        )
        assert args.description == "학습 상담"
        assert args.recorded_by == "prof_kim"
        assert args.follow_up_week == 4

    def test_list_subcommand(self):
        """list subcommand requires --store, accepts --student, --week."""
        from forma.cli_intervention import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "list",
                "--store",
                "log.yaml",
            ]
        )
        assert args.subcommand == "list"
        assert args.store == "log.yaml"

    def test_list_subcommand_filters(self):
        """list subcommand accepts --student and --week filters."""
        from forma.cli_intervention import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "list",
                "--store",
                "log.yaml",
                "--student",
                "s001",
                "--week",
                "3",
            ]
        )
        assert args.student == "s001"
        assert args.week == 3

    def test_update_subcommand(self):
        """update subcommand requires --store, --id, --outcome."""
        from forma.cli_intervention import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "update",
                "--store",
                "log.yaml",
                "--id",
                "1",
                "--outcome",
                "개선",
            ]
        )
        assert args.subcommand == "update"
        assert args.id == 1
        assert args.outcome == "개선"

    def test_no_config_flag(self):
        """--no-config flag is accepted at top level."""
        from forma.cli_intervention import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--no-config",
                "add",
                "--store",
                "log.yaml",
                "--student",
                "s001",
                "--week",
                "2",
                "--type",
                "면담",
            ]
        )
        assert args.no_config is True

    def test_verbose_flag(self):
        """--verbose flag is accepted."""
        from forma.cli_intervention import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--verbose",
                "add",
                "--store",
                "log.yaml",
                "--student",
                "s001",
                "--week",
                "2",
                "--type",
                "면담",
            ]
        )
        assert args.verbose is True


# ---------------------------------------------------------------------------
# T011: `add` subcommand — success and validation
# ---------------------------------------------------------------------------


class TestCliInterventionAdd:
    """Tests for `forma-intervention add`."""

    def test_add_success_prints_id(self, tmp_path, capsys):
        """Successful add prints assigned record ID."""
        store_path = str(tmp_path / "intervention_log.yaml")
        from forma.cli_intervention import main

        main(
            [
                "--no-config",
                "add",
                "--store",
                store_path,
                "--student",
                "s001",
                "--week",
                "2",
                "--type",
                "면담",
                "--description",
                "학습 상담",
            ]
        )

        captured = capsys.readouterr()
        assert "1" in captured.out  # ID should be printed

    def test_add_creates_log_file(self, tmp_path):
        """add creates log YAML file if it doesn't exist."""
        store_path = str(tmp_path / "intervention_log.yaml")
        from forma.cli_intervention import main

        main(
            [
                "--no-config",
                "add",
                "--store",
                store_path,
                "--student",
                "s001",
                "--week",
                "2",
                "--type",
                "면담",
            ]
        )
        assert os.path.exists(store_path)

    def test_add_invalid_type_exits(self, tmp_path):
        """Invalid intervention type exits with error."""
        store_path = str(tmp_path / "intervention_log.yaml")
        from forma.cli_intervention import main

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--no-config",
                    "add",
                    "--store",
                    store_path,
                    "--student",
                    "s001",
                    "--week",
                    "2",
                    "--type",
                    "invalid_type",
                ]
            )
        assert exc_info.value.code != 0

    def test_add_all_five_types(self, tmp_path, capsys):
        """All 5 predefined types are accepted."""
        from forma.cli_intervention import main

        store_path = str(tmp_path / "intervention_log.yaml")
        types = ["면담", "보충학습", "과제부여", "멘토링", "기타"]
        for i, t in enumerate(types, 1):
            main(
                [
                    "--no-config",
                    "add",
                    "--store",
                    store_path,
                    "--student",
                    f"s{i:03d}",
                    "--week",
                    "2",
                    "--type",
                    t,
                ]
            )

        captured = capsys.readouterr()
        # All should succeed (5 IDs printed)
        for i in range(1, 6):
            assert str(i) in captured.out

    def test_add_multiple_records_auto_increment(self, tmp_path, capsys):
        """Multiple adds auto-increment IDs."""
        from forma.cli_intervention import main

        store_path = str(tmp_path / "intervention_log.yaml")
        main(
            [
                "--no-config",
                "add",
                "--store",
                store_path,
                "--student",
                "s001",
                "--week",
                "2",
                "--type",
                "면담",
            ]
        )
        main(
            [
                "--no-config",
                "add",
                "--store",
                store_path,
                "--student",
                "s002",
                "--week",
                "3",
                "--type",
                "보충학습",
            ]
        )

        # Verify file has 2 records
        with open(store_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert len(data["records"]) == 2

    def test_add_with_follow_up_week(self, tmp_path):
        """add with --follow-up-week stores the value."""
        from forma.cli_intervention import main

        store_path = str(tmp_path / "intervention_log.yaml")
        main(
            [
                "--no-config",
                "add",
                "--store",
                store_path,
                "--student",
                "s001",
                "--week",
                "2",
                "--type",
                "면담",
                "--follow-up-week",
                "4",
            ]
        )

        with open(store_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["records"][0]["follow_up_week"] == 4

    def test_add_negative_week_exits(self, tmp_path):
        """Negative week value exits with error."""
        from forma.cli_intervention import main

        store_path = str(tmp_path / "intervention_log.yaml")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--no-config",
                    "add",
                    "--store",
                    store_path,
                    "--student",
                    "s001",
                    "--week",
                    "-1",
                    "--type",
                    "면담",
                ]
            )
        assert exc_info.value.code != 0

    def test_add_zero_week_exits(self, tmp_path):
        """Week=0 exits with error (must be >= 1)."""
        from forma.cli_intervention import main

        store_path = str(tmp_path / "intervention_log.yaml")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--no-config",
                    "add",
                    "--store",
                    store_path,
                    "--student",
                    "s001",
                    "--week",
                    "0",
                    "--type",
                    "면담",
                ]
            )
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# T012: `list` subcommand
# ---------------------------------------------------------------------------


class TestCliInterventionList:
    """Tests for `forma-intervention list`."""

    def _add_records(self, store_path):
        """Add several records for testing."""
        from forma.cli_intervention import main

        main(
            [
                "--no-config",
                "add",
                "--store",
                store_path,
                "--student",
                "s001",
                "--week",
                "2",
                "--type",
                "면담",
                "--description",
                "상담",
            ]
        )
        main(
            [
                "--no-config",
                "add",
                "--store",
                store_path,
                "--student",
                "s001",
                "--week",
                "3",
                "--type",
                "보충학습",
                "--description",
                "보충",
            ]
        )
        main(
            [
                "--no-config",
                "add",
                "--store",
                store_path,
                "--student",
                "s002",
                "--week",
                "2",
                "--type",
                "과제부여",
                "--description",
                "과제",
            ]
        )

    def test_list_all(self, tmp_path, capsys):
        """list with no filters shows all records."""
        store_path = str(tmp_path / "intervention_log.yaml")
        self._add_records(store_path)

        from forma.cli_intervention import main

        main(["--no-config", "list", "--store", store_path])

        captured = capsys.readouterr()
        assert "s001" in captured.out
        assert "s002" in captured.out

    def test_list_filter_by_student(self, tmp_path, capsys):
        """list --student filters by student_id."""
        store_path = str(tmp_path / "intervention_log.yaml")
        self._add_records(store_path)

        from forma.cli_intervention import main

        main(["--no-config", "list", "--store", store_path, "--student", "s001"])

        captured = capsys.readouterr()
        assert "s001" in captured.out
        assert "s002" not in captured.out

    def test_list_filter_by_week(self, tmp_path, capsys):
        """list --week filters by week number."""
        store_path = str(tmp_path / "intervention_log.yaml")
        self._add_records(store_path)

        from forma.cli_intervention import main

        main(["--no-config", "list", "--store", store_path, "--week", "2"])

        captured = capsys.readouterr()
        # Week 2: s001 면담, s002 과제부여
        assert "면담" in captured.out
        assert "과제부여" in captured.out
        # Week 3 records should not appear
        assert "보충학습" not in captured.out

    def test_list_empty_result_message(self, tmp_path, capsys):
        """list with no matching records outputs '기록 없음' message."""
        store_path = str(tmp_path / "intervention_log.yaml")
        self._add_records(store_path)

        from forma.cli_intervention import main

        main(["--no-config", "list", "--store", store_path, "--student", "s999"])

        captured = capsys.readouterr()
        assert "No records found" in captured.out

    def test_list_nonexistent_store_exits(self, tmp_path):
        """list with nonexistent store file exits with error."""
        from forma.cli_intervention import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--no-config", "list", "--store", str(tmp_path / "missing.yaml")])
        assert exc_info.value.code != 0

    def test_list_combined_filters(self, tmp_path, capsys):
        """list with both --student and --week filters combined."""
        store_path = str(tmp_path / "intervention_log.yaml")
        self._add_records(store_path)

        from forma.cli_intervention import main

        main(["--no-config", "list", "--store", store_path, "--student", "s001", "--week", "2"])

        captured = capsys.readouterr()
        assert "s001" in captured.out
        assert "면담" in captured.out
        # s001 week 3 보충학습 should not appear
        assert "보충학습" not in captured.out
        # s002 should not appear
        assert "s002" not in captured.out

    def test_list_tabular_columns(self, tmp_path, capsys):
        """list output contains expected column headers."""
        store_path = str(tmp_path / "intervention_log.yaml")
        self._add_records(store_path)

        from forma.cli_intervention import main

        main(["--no-config", "list", "--store", store_path])

        captured = capsys.readouterr()
        # Should contain column headers
        assert "ID" in captured.out
        assert "Student" in captured.out
        assert "Week" in captured.out
        assert "Type" in captured.out
        assert "Description" in captured.out
        assert "Outcome" in captured.out

    def test_list_chronological_order(self, tmp_path, capsys):
        """list output is in chronological order (by week)."""
        store_path = str(tmp_path / "intervention_log.yaml")
        from forma.cli_intervention import main

        # Add in non-chronological order
        main(
            [
                "--no-config",
                "add",
                "--store",
                store_path,
                "--student",
                "s001",
                "--week",
                "4",
                "--type",
                "기타",
                "--description",
                "week4",
            ]
        )
        main(
            [
                "--no-config",
                "add",
                "--store",
                store_path,
                "--student",
                "s001",
                "--week",
                "1",
                "--type",
                "면담",
                "--description",
                "week1",
            ]
        )
        main(
            [
                "--no-config",
                "add",
                "--store",
                store_path,
                "--student",
                "s001",
                "--week",
                "3",
                "--type",
                "보충학습",
                "--description",
                "week3",
            ]
        )

        main(["--no-config", "list", "--store", store_path])

        captured = capsys.readouterr()
        # Extract data lines (skip header and separator)
        lines = [ln for ln in captured.out.strip().split("\n") if ln.strip() and "---" not in ln]
        # Data lines should be sorted by week: week1, week3, week4
        data_lines = [ln for ln in lines if "면담" in ln or "보충학습" in ln or "기타" in ln]
        assert len(data_lines) == 3
        # First line should be week 1, last should be week 4
        assert "면담" in data_lines[0]  # week 1
        assert "보충학습" in data_lines[1]  # week 3
        assert "기타" in data_lines[2]  # week 4


# ---------------------------------------------------------------------------
# T013: `update` subcommand — outcome validation
# ---------------------------------------------------------------------------


class TestCliInterventionUpdate:
    """Tests for `forma-intervention update`."""

    def _add_one_record(self, store_path):
        """Add a single record and return."""
        from forma.cli_intervention import main

        main(["--no-config", "add", "--store", store_path, "--student", "s001", "--week", "2", "--type", "면담"])

    def test_update_success(self, tmp_path, capsys):
        """update with valid outcome prints success."""
        store_path = str(tmp_path / "intervention_log.yaml")
        self._add_one_record(store_path)

        from forma.cli_intervention import main

        main(
            [
                "--no-config",
                "update",
                "--store",
                store_path,
                "--id",
                "1",
                "--outcome",
                "개선",
            ]
        )

        captured = capsys.readouterr()
        # Should indicate success
        assert captured.out.strip() != ""

    def test_update_outcome_persists(self, tmp_path):
        """Updated outcome persists in YAML file."""
        store_path = str(tmp_path / "intervention_log.yaml")
        self._add_one_record(store_path)

        from forma.cli_intervention import main

        main(
            [
                "--no-config",
                "update",
                "--store",
                store_path,
                "--id",
                "1",
                "--outcome",
                "유지",
            ]
        )

        with open(store_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["records"][0]["outcome"] == "유지"

    def test_update_all_valid_outcomes(self, tmp_path, capsys):
        """All three valid outcomes (개선/유지/악화) are accepted."""
        from forma.cli_intervention import main

        store_path = str(tmp_path / "intervention_log.yaml")
        # Add 3 records
        for i in range(3):
            main(
                ["--no-config", "add", "--store", store_path, "--student", f"s{i:03d}", "--week", "2", "--type", "면담"]
            )

        valid_outcomes = ["개선", "유지", "악화"]
        for i, outcome in enumerate(valid_outcomes, 1):
            main(
                [
                    "--no-config",
                    "update",
                    "--store",
                    store_path,
                    "--id",
                    str(i),
                    "--outcome",
                    outcome,
                ]
            )

        # Verify all outcomes persisted
        with open(store_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        outcomes = [r["outcome"] for r in data["records"]]
        assert outcomes == ["개선", "유지", "악화"]

    def test_update_invalid_outcome_exits(self, tmp_path):
        """Invalid outcome value exits with error."""
        store_path = str(tmp_path / "intervention_log.yaml")
        self._add_one_record(store_path)

        from forma.cli_intervention import main

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--no-config",
                    "update",
                    "--store",
                    store_path,
                    "--id",
                    "1",
                    "--outcome",
                    "개선됨",  # invalid — must be exactly 개선/유지/악화
                ]
            )
        assert exc_info.value.code != 0

    def test_update_nonexistent_id_exits(self, tmp_path):
        """Nonexistent record ID exits with error."""
        store_path = str(tmp_path / "intervention_log.yaml")
        self._add_one_record(store_path)

        from forma.cli_intervention import main

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--no-config",
                    "update",
                    "--store",
                    store_path,
                    "--id",
                    "999",
                    "--outcome",
                    "개선",
                ]
            )
        assert exc_info.value.code != 0

    def test_update_nonexistent_log_exits(self, tmp_path):
        """Nonexistent log file exits with error."""
        from forma.cli_intervention import main

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--no-config",
                    "update",
                    "--store",
                    str(tmp_path / "missing.yaml"),
                    "--id",
                    "1",
                    "--outcome",
                    "개선",
                ]
            )
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestCliInterventionEdgeCases:
    """Edge case tests."""

    def test_korean_description_roundtrip(self, tmp_path, capsys):
        """Korean text in --description survives add/list roundtrip."""
        store_path = str(tmp_path / "intervention_log.yaml")
        from forma.cli_intervention import main

        main(
            [
                "--no-config",
                "add",
                "--store",
                store_path,
                "--student",
                "s001",
                "--week",
                "2",
                "--type",
                "면담",
                "--description",
                "학습 동기 부여 및 진로 상담 진행",
            ]
        )

        main(["--no-config", "list", "--store", store_path])
        captured = capsys.readouterr()
        assert "학습 동기 부여 및 진로 상담 진행" in captured.out

    def test_no_subcommand_shows_help(self):
        """No subcommand exits with error (help displayed)."""
        from forma.cli_intervention import main

        with pytest.raises(SystemExit):
            main(["--no-config"])
