"""Tests for cli_report_longitudinal.py — CLI for US3 longitudinal report.

RED phase: tests written BEFORE implementation (TDD).

Covers T034:
  - CLI argument parsing (required: --store, --class-name, --output)
  - Optional arguments: --weeks, --exam-file, --font-path
  - Validation: missing required args, non-existent store file
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# T034: CLI argument parsing tests
# ---------------------------------------------------------------------------


class TestCLIArgumentParsing:
    """Test forma-report-longitudinal argument parsing."""

    def test_required_args(self):
        """Parser accepts required args: --store, --class-name, --output."""
        from forma.cli_report_longitudinal import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "/path/to/store.yaml",
            "--class-name", "1A",
            "--output", "/path/to/output.pdf",
        ])
        assert args.store == "/path/to/store.yaml"
        assert args.class_name == "1A"
        assert args.output == "/path/to/output.pdf"

    def test_optional_weeks(self):
        """--weeks accepts multiple integers."""
        from forma.cli_report_longitudinal import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "store.yaml",
            "--class-name", "1A",
            "--output", "out.pdf",
            "--weeks", "1", "2", "3", "4",
        ])
        assert args.weeks == [1, 2, 3, 4]

    def test_weeks_default_none(self):
        """--weeks omitted → defaults to None (all weeks)."""
        from forma.cli_report_longitudinal import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "store.yaml",
            "--class-name", "1A",
            "--output", "out.pdf",
        ])
        assert args.weeks is None

    def test_optional_exam_file(self):
        """--exam-file is optional."""
        from forma.cli_report_longitudinal import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "store.yaml",
            "--class-name", "1A",
            "--output", "out.pdf",
            "--exam-file", "exam.yaml",
        ])
        assert args.exam_file == "exam.yaml"

    def test_exam_file_default_none(self):
        """--exam-file omitted → defaults to None."""
        from forma.cli_report_longitudinal import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "store.yaml",
            "--class-name", "1A",
            "--output", "out.pdf",
        ])
        assert args.exam_file is None

    def test_optional_font_path(self):
        """--font-path is optional."""
        from forma.cli_report_longitudinal import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "store.yaml",
            "--class-name", "1A",
            "--output", "out.pdf",
            "--font-path", "/path/to/font.ttf",
        ])
        assert args.font_path == "/path/to/font.ttf"

    def test_font_path_default_none(self):
        """--font-path omitted → defaults to None (auto-detect)."""
        from forma.cli_report_longitudinal import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "store.yaml",
            "--class-name", "1A",
            "--output", "out.pdf",
        ])
        assert args.font_path is None

    def test_missing_required_store(self):
        """Missing --store → parser error."""
        from forma.cli_report_longitudinal import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--class-name", "1A",
                "--output", "out.pdf",
            ])

    def test_missing_required_class_name(self):
        """Missing --class-name → parser error."""
        from forma.cli_report_longitudinal import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--store", "store.yaml",
                "--output", "out.pdf",
            ])

    def test_missing_required_output(self):
        """Missing --output → parser error."""
        from forma.cli_report_longitudinal import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--store", "store.yaml",
                "--class-name", "1A",
            ])


class TestCLIMainFunction:
    """Test the main() function behavior."""

    def test_nonexistent_store_exits(self, tmp_path):
        """Non-existent store file should cause sys.exit(1)."""
        from forma.cli_report_longitudinal import main

        fake_store = str(tmp_path / "nonexistent.yaml")
        fake_output = str(tmp_path / "out.pdf")

        with patch("sys.argv", [
            "forma-report-longitudinal",
            "--store", fake_store,
            "--class-name", "1A",
            "--output", fake_output,
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_successful_run(self, tmp_path):
        """Successful run with valid store creates PDF."""
        from forma.cli_report_longitudinal import main
        from forma.longitudinal_store import LongitudinalStore
        from forma.evaluation_types import LongitudinalRecord

        # Create a store with test data
        store_path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(store_path)
        for week in range(1, 4):
            for sid in ["S001", "S002", "S003"]:
                store.add_record(LongitudinalRecord(
                    student_id=sid,
                    week=week,
                    question_sn=1,
                    scores={"ensemble_score": 0.5 + week * 0.05},
                    tier_level=1,
                    tier_label="Developing",
                    concept_scores={"항상성": 0.6 + week * 0.05},
                ))
        store.save()

        output_path = str(tmp_path / "report.pdf")
        with patch("sys.argv", [
            "forma-report-longitudinal",
            "--store", store_path,
            "--class-name", "1A",
            "--output", output_path,
        ]):
            result = main()

        assert result is None or result == 0
        assert os.path.exists(output_path)


# ---------------------------------------------------------------------------
# US5: --classes, --heatmap-layout, --risk-threshold, parse_heatmap_layout
# ---------------------------------------------------------------------------


class TestParseHeatmapLayout:
    """T041: parse_heatmap_layout validates rows:cols format."""

    def test_valid_1_4(self):
        """'1:4' → (1, 4)."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )
        assert parse_heatmap_layout("1:4") == (1, 4)

    def test_valid_2_2(self):
        """'2:2' → (2, 2)."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )
        assert parse_heatmap_layout("2:2") == (2, 2)

    def test_reject_x_separator(self):
        """'1x4' → ValueError."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )
        with pytest.raises(ValueError):
            parse_heatmap_layout("1x4")

    def test_reject_abc(self):
        """'abc' → ValueError."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )
        with pytest.raises(ValueError):
            parse_heatmap_layout("abc")

    def test_reject_zero(self):
        """'0:4' → ValueError (non-positive)."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )
        with pytest.raises(ValueError):
            parse_heatmap_layout("0:4")


class TestClassesArgParsing:
    """T041 extended: --classes and --heatmap-layout args."""

    def test_classes_arg(self):
        """--classes A B C D parsed correctly."""
        from forma.cli_report_longitudinal import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--store", "s.yaml",
            "--class-name", "1A",
            "--output", "o.pdf",
            "--classes", "A", "B", "C", "D",
        ])
        assert args.classes == ["A", "B", "C", "D"]

    def test_heatmap_layout_arg(self):
        """--heatmap-layout 1:4 parsed as string."""
        from forma.cli_report_longitudinal import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--store", "s.yaml",
            "--class-name", "1A",
            "--output", "o.pdf",
            "--heatmap-layout", "1:4",
        ])
        assert args.heatmap_layout == "1:4"

    def test_risk_threshold_arg(self):
        """--risk-threshold 0.50 parsed as float."""
        from forma.cli_report_longitudinal import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--store", "s.yaml",
            "--class-name", "1A",
            "--output", "o.pdf",
            "--risk-threshold", "0.50",
        ])
        assert args.risk_threshold == 0.50

    def test_defaults(self):
        """Default values for new optional args."""
        from forma.cli_report_longitudinal import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--store", "s.yaml",
            "--class-name", "1A",
            "--output", "o.pdf",
        ])
        assert args.classes is None
        assert args.heatmap_layout is None
        assert args.risk_threshold == 0.45
