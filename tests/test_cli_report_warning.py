"""Tests for cli_report_warning.py — forma-report-warning CLI.

T041 [US3]: Required args, optional args, --no-config, end-to-end with mock data.
"""

from __future__ import annotations


import pytest
import yaml


def _write_minimal_data(tmp_path):
    """Write minimal final YAML, config, and eval dir for CLI tests."""
    # Minimal exam config
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "metadata": {"chapter_name": "테스트", "course_name": "과목", "week_num": 1},
                "questions": [
                    {
                        "question_number": 1,
                        "question_text": "테스트 문제",
                        "master_concepts": [
                            {"subject": "A", "relation": "is_a", "object": "B"},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    # Minimal final YAML (list of student responses)
    final_path = tmp_path / "final.yaml"
    students = []
    for i in range(5):
        students.append(
            {
                "student_id": f"S{i:03d}",
                "questions": [
                    {
                        "question_number": 1,
                        "student_answer": "답변",
                    },
                ],
            }
        )
    final_path.write_text(yaml.dump(students), encoding="utf-8")

    # Eval dir
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    return str(final_path), str(config_path), str(eval_dir)


class TestCliReportWarningParser:
    """Tests for argument parsing."""

    def test_required_args(self):
        """Parser requires --final, --config, --eval-dir, --output."""
        from forma.cli_report_warning import _build_parser

        parser = _build_parser()
        # Should succeed with all required args
        args = parser.parse_args(
            [
                "--final",
                "f.yaml",
                "--config",
                "c.yaml",
                "--eval-dir",
                "eval/",
                "--output",
                "out.pdf",
            ]
        )
        assert args.final == "f.yaml"
        assert args.config == "c.yaml"
        assert args.eval_dir == "eval/"
        assert args.output == "out.pdf"

    def test_optional_model_flag(self):
        """--model flag is accepted."""
        from forma.cli_report_warning import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--final",
                "f.yaml",
                "--config",
                "c.yaml",
                "--eval-dir",
                "eval/",
                "--output",
                "out.pdf",
                "--model",
                "model.pkl",
            ]
        )
        assert args.model_path == "model.pkl"

    def test_optional_longitudinal_flags(self):
        """--longitudinal-store and --week flags are accepted."""
        from forma.cli_report_warning import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--final",
                "f.yaml",
                "--config",
                "c.yaml",
                "--eval-dir",
                "eval/",
                "--output",
                "out.pdf",
                "--longitudinal-store",
                "store.yaml",
                "--week",
                "3",
            ]
        )
        assert args.longitudinal_store == "store.yaml"
        assert args.week == 3

    def test_no_config_flag(self):
        """--no-config flag is accepted."""
        from forma.cli_report_warning import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--final",
                "f.yaml",
                "--config",
                "c.yaml",
                "--eval-dir",
                "eval/",
                "--output",
                "out.pdf",
                "--no-config",
            ]
        )
        assert args.no_config is True

    def test_dpi_flag(self):
        """--dpi flag is accepted."""
        from forma.cli_report_warning import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--final",
                "f.yaml",
                "--config",
                "c.yaml",
                "--eval-dir",
                "eval/",
                "--output",
                "out.pdf",
                "--dpi",
                "200",
            ]
        )
        assert args.dpi == 200

    def test_default_dpi(self):
        """Default DPI is 150."""
        from forma.cli_report_warning import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--final",
                "f.yaml",
                "--config",
                "c.yaml",
                "--eval-dir",
                "eval/",
                "--output",
                "out.pdf",
            ]
        )
        assert args.dpi == 150


class TestCliReportWarningValidation:
    """Tests for input file validation."""

    def test_missing_final_file(self, tmp_path):
        """Exits with error if --final file doesn't exist."""
        from forma.cli_report_warning import main

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--final",
                    str(tmp_path / "nonexistent.yaml"),
                    "--config",
                    str(tmp_path / "c.yaml"),
                    "--eval-dir",
                    str(tmp_path),
                    "--output",
                    str(tmp_path / "out.pdf"),
                    "--no-config",
                ]
            )
        assert exc_info.value.code != 0

    def test_missing_config_file(self, tmp_path):
        """Exits with error if --config file doesn't exist."""
        from forma.cli_report_warning import main

        final = tmp_path / "final.yaml"
        final.write_text("[]", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--final",
                    str(final),
                    "--config",
                    str(tmp_path / "nonexistent.yaml"),
                    "--eval-dir",
                    str(tmp_path),
                    "--output",
                    str(tmp_path / "out.pdf"),
                    "--no-config",
                ]
            )
        assert exc_info.value.code != 0
