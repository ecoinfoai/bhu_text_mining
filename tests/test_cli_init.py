"""Tests for cli_init.py — forma init CLI entry point.

T010 [US1]: Template generation, overwrite protection, --force, --output,
            generated YAML is valid and parseable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


class TestFormaInit:
    """Tests for forma init CLI."""

    def test_generates_yaml_template(self, tmp_path: Path, monkeypatch):
        """forma init creates a forma.yaml file with valid YAML content."""
        output_path = tmp_path / "forma.yaml"
        monkeypatch.setattr(
            "sys.argv",
            [
                "forma-init",
                "--output",
                str(output_path),
            ],
        )

        # Mock interactive prompts to return defaults
        with patch("builtins.input", return_value=""):
            from forma.cli_init import main

            main()

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert len(content) > 0

        # YAML should be parseable
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)

    def test_template_contains_expected_sections(self, tmp_path: Path, monkeypatch):
        """Generated template includes project, classes, paths, etc. sections."""
        output_path = tmp_path / "forma.yaml"
        monkeypatch.setattr(
            "sys.argv",
            [
                "forma-init",
                "--output",
                str(output_path),
            ],
        )

        with patch("builtins.input", return_value=""):
            from forma.cli_init import main

            main()

        content = output_path.read_text(encoding="utf-8")
        # Check for key sections
        assert "project:" in content
        assert "classes:" in content
        assert "paths:" in content

    def test_template_has_comments(self, tmp_path: Path, monkeypatch):
        """Generated template has comment lines (#) explaining settings."""
        output_path = tmp_path / "forma.yaml"
        monkeypatch.setattr(
            "sys.argv",
            [
                "forma-init",
                "--output",
                str(output_path),
            ],
        )

        with patch("builtins.input", return_value=""):
            from forma.cli_init import main

            main()

        content = output_path.read_text(encoding="utf-8")
        comment_lines = [ln for ln in content.splitlines() if ln.strip().startswith("#")]
        assert len(comment_lines) >= 3  # Should have several comments

    def test_overwrite_protection(self, tmp_path: Path, monkeypatch):
        """forma init refuses to overwrite existing file without --force."""
        output_path = tmp_path / "forma.yaml"
        output_path.write_text("existing: true\n", encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "forma-init",
                "--output",
                str(output_path),
            ],
        )

        with patch("builtins.input", return_value=""):
            from forma.cli_init import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        # Content should be unchanged
        assert output_path.read_text(encoding="utf-8") == "existing: true\n"

    def test_force_overwrites(self, tmp_path: Path, monkeypatch):
        """forma init --force overwrites existing file."""
        output_path = tmp_path / "forma.yaml"
        output_path.write_text("existing: true\n", encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "forma-init",
                "--output",
                str(output_path),
                "--force",
            ],
        )

        with patch("builtins.input", return_value=""):
            from forma.cli_init import main

            main()

        content = output_path.read_text(encoding="utf-8")
        assert content != "existing: true\n"
        # Should be valid YAML
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)

    def test_custom_output_path(self, tmp_path: Path, monkeypatch):
        """forma init --output custom_path.yaml writes to custom path."""
        custom = tmp_path / "my_config.yaml"
        monkeypatch.setattr(
            "sys.argv",
            [
                "forma-init",
                "--output",
                str(custom),
            ],
        )

        with patch("builtins.input", return_value=""):
            from forma.cli_init import main

            main()

        assert custom.exists()

    def test_default_output_is_forma_yaml(self, tmp_path: Path, monkeypatch):
        """Default output path is ./forma.yaml."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["forma-init"])

        with patch("builtins.input", return_value=""):
            from forma.cli_init import main

            main()

        assert (tmp_path / "forma.yaml").exists()

    def test_interactive_course_name(self, tmp_path: Path, monkeypatch):
        """Interactive prompt for course_name populates the template."""
        output_path = tmp_path / "forma.yaml"
        monkeypatch.setattr(
            "sys.argv",
            [
                "forma-init",
                "--output",
                str(output_path),
            ],
        )

        responses = iter(["인체구조와기능", "", "", ""])
        with patch("builtins.input", side_effect=lambda _="": next(responses, "")):
            from forma.cli_init import main

            main()

        parsed = yaml.safe_load(output_path.read_text(encoding="utf-8"))
        assert parsed["project"]["course_name"] == "인체구조와기능"

    def test_generated_yaml_is_loadable(self, tmp_path: Path, monkeypatch):
        """Generated config can be loaded by load_project_config."""
        output_path = tmp_path / "forma.yaml"
        monkeypatch.setattr(
            "sys.argv",
            [
                "forma-init",
                "--output",
                str(output_path),
            ],
        )

        # Provide valid interactive input
        responses = iter(["테스트과목", "2026", "1", "A,B"])
        with patch("builtins.input", side_effect=lambda _="": next(responses, "")):
            from forma.cli_init import main

            main()

        from forma.project_config import load_project_config, validate_project_config

        config = load_project_config(output_path)
        validate_project_config(config)  # Should not raise

    # ---------------------------------------------------------------
    # T061: Template does NOT contain week-specific fields
    # ---------------------------------------------------------------
    def test_template_excludes_week_specific_fields(self, tmp_path: Path, monkeypatch):
        """Slimmed template must NOT contain week-specific fields."""
        output_path = tmp_path / "forma.yaml"
        monkeypatch.setattr(
            "sys.argv",
            [
                "forma-init",
                "--output",
                str(output_path),
            ],
        )

        with patch("builtins.input", return_value=""):
            from forma.cli_init import main

            main()

        content = output_path.read_text(encoding="utf-8")
        # These fields should have been removed (moved to week.yaml)
        assert "num_questions" not in content
        assert "exam_config" not in content
        assert "join_dir" not in content
        assert "output_dir" not in content
        assert "current_week" not in content
        assert "join_pattern" not in content
        assert "eval_pattern" not in content
        assert "skip_feedback" not in content
        assert "skip_graph" not in content
        assert "skip_statistical" not in content
        assert "skip_llm" not in content
        assert "aggregate" not in content

    # ---------------------------------------------------------------
    # T062: Template DOES contain semester-level fields
    # ---------------------------------------------------------------
    def test_template_contains_semester_fields(self, tmp_path: Path, monkeypatch):
        """Slimmed template must contain semester-level fields."""
        output_path = tmp_path / "forma.yaml"
        monkeypatch.setattr(
            "sys.argv",
            [
                "forma-init",
                "--output",
                str(output_path),
            ],
        )

        with patch("builtins.input", return_value=""):
            from forma.cli_init import main

            main()

        content = output_path.read_text(encoding="utf-8")
        # These fields should be present
        assert "project:" in content
        assert "course_name" in content
        assert "identifiers" in content
        assert "naver_config" in content
        assert "credentials" in content
        assert "spreadsheet_url" in content
        assert "provider" in content
        assert "n_calls" in content
        assert "longitudinal_store" in content
        assert "font_path" in content
        assert "dpi" in content

    def test_generated_yaml_utf8(self, tmp_path: Path, monkeypatch):
        """Generated file uses UTF-8 encoding for Korean content."""
        output_path = tmp_path / "forma.yaml"
        monkeypatch.setattr(
            "sys.argv",
            [
                "forma-init",
                "--output",
                str(output_path),
            ],
        )

        responses = iter(["해부학", "", "", ""])
        with patch("builtins.input", side_effect=lambda _="": next(responses, "")):
            from forma.cli_init import main

            main()

        content = output_path.read_bytes()
        # Should be valid UTF-8
        decoded = content.decode("utf-8")
        assert "해부학" in decoded
