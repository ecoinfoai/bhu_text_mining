"""Tests for project_config.py — ProjectConfiguration, find/load/validate/merge.

Tests cover:
- ProjectConfiguration dataclass defaults and construction
- find_project_config(): CWD, parent, .git sentinel, no config
- load_project_config(): valid YAML, empty file, encoding
- validate_project_config(): unknown keys warn, type errors, value constraints,
  all-errors-collected, valid config passes
- merge_configs(): CLI > forma.yaml > system config, unset values fall through
- Edge cases: empty file, non-ASCII paths, missing sections
"""

from __future__ import annotations

import argparse
import logging
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# T004-A: ProjectConfiguration dataclass tests
# ---------------------------------------------------------------------------
class TestProjectConfiguration:
    """Tests for ProjectConfiguration dataclass."""

    def test_default_values(self):
        """All fields have correct defaults per data-model.md."""
        from forma.project_config import ProjectConfiguration

        cfg = ProjectConfiguration()
        assert cfg.course_name == ""
        assert cfg.year == 0
        assert cfg.semester == 0
        assert cfg.grade == 0
        assert cfg.class_identifiers == []
        assert cfg.join_pattern == ""
        assert cfg.eval_pattern == ""
        assert cfg.exam_config == ""
        assert cfg.join_dir == ""
        assert cfg.output_dir == ""
        assert cfg.longitudinal_store == ""
        assert cfg.font_path is None
        assert cfg.naver_config == ""
        assert cfg.credentials == ""
        assert cfg.spreadsheet_url == ""
        assert cfg.num_questions == 5
        assert cfg.provider == "gemini"
        assert cfg.model is None
        assert cfg.skip_feedback is False
        assert cfg.skip_graph is False
        assert cfg.skip_statistical is False
        assert cfg.n_calls == 3
        assert cfg.dpi == 150
        assert cfg.skip_llm is False
        assert cfg.aggregate is True
        assert cfg.current_week == 1
        assert cfg.model_path is None

    def test_custom_values(self):
        """Can construct with custom values."""
        from forma.project_config import ProjectConfiguration

        cfg = ProjectConfiguration(
            course_name="인체구조와기능",
            year=2026,
            semester=1,
            grade=2,
            class_identifiers=["A", "B"],
            dpi=300,
            provider="anthropic",
        )
        assert cfg.course_name == "인체구조와기능"
        assert cfg.year == 2026
        assert cfg.semester == 1
        assert cfg.grade == 2
        assert cfg.class_identifiers == ["A", "B"]
        assert cfg.dpi == 300
        assert cfg.provider == "anthropic"

    def test_class_identifiers_not_shared(self):
        """Default list should not be shared between instances."""
        from forma.project_config import ProjectConfiguration

        cfg1 = ProjectConfiguration()
        cfg2 = ProjectConfiguration()
        cfg1.class_identifiers.append("X")
        assert cfg2.class_identifiers == []


# ---------------------------------------------------------------------------
# T004-B: find_project_config() tests
# ---------------------------------------------------------------------------
class TestFindProjectConfig:
    """Tests for find_project_config()."""

    def test_finds_in_cwd(self, tmp_path: Path):
        """Finds forma.yaml in the current working directory."""
        from forma.project_config import find_project_config

        config_file = tmp_path / "forma.yaml"
        config_file.write_text("course_name: test\n", encoding="utf-8")

        result = find_project_config(start_dir=tmp_path)
        assert result is not None
        assert result.name == "forma.yaml"
        assert result.resolve() == config_file.resolve()

    def test_finds_in_parent(self, tmp_path: Path):
        """Finds forma.yaml in a parent directory."""
        from forma.project_config import find_project_config

        config_file = tmp_path / "forma.yaml"
        config_file.write_text("course_name: test\n", encoding="utf-8")
        child = tmp_path / "subdir"
        child.mkdir()

        result = find_project_config(start_dir=child)
        assert result is not None
        assert result.resolve() == config_file.resolve()

    def test_stops_at_git_sentinel(self, tmp_path: Path):
        """Stops searching at .git directory even if forma.yaml is higher."""
        from forma.project_config import find_project_config

        # Put forma.yaml in parent, .git in child
        parent = tmp_path
        (parent / "forma.yaml").write_text("course_name: above\n", encoding="utf-8")
        child = parent / "project"
        child.mkdir()
        (child / ".git").mkdir()  # .git sentinel
        grandchild = child / "subdir"
        grandchild.mkdir()

        result = find_project_config(start_dir=grandchild)
        assert result is None  # Should not find parent's config

    def test_finds_config_at_git_level(self, tmp_path: Path):
        """Finds forma.yaml at the same level as .git."""
        from forma.project_config import find_project_config

        (tmp_path / ".git").mkdir()
        config_file = tmp_path / "forma.yaml"
        config_file.write_text("course_name: project\n", encoding="utf-8")
        child = tmp_path / "subdir"
        child.mkdir()

        result = find_project_config(start_dir=child)
        assert result is not None
        assert result.resolve() == config_file.resolve()

    def test_returns_none_no_config(self, tmp_path: Path):
        """Returns None when no forma.yaml is found."""
        from forma.project_config import find_project_config

        (tmp_path / ".git").mkdir()
        result = find_project_config(start_dir=tmp_path)
        assert result is None

    def test_defaults_to_cwd(self, tmp_path: Path):
        """Uses CWD when start_dir is None."""
        from forma.project_config import find_project_config

        config_file = tmp_path / "forma.yaml"
        config_file.write_text("course_name: test\n", encoding="utf-8")

        with patch("forma.project_config.Path") as mock_path_cls:
            # Make Path.cwd() return tmp_path while keeping Path(x) working
            import pathlib

            original_path = pathlib.Path

            def side_effect(*args, **kwargs):
                if not args and not kwargs:
                    # Path() call — should not happen typically
                    return original_path()
                return original_path(*args, **kwargs)

            mock_path_cls.side_effect = side_effect
            mock_path_cls.cwd.return_value = original_path(tmp_path)

            result = find_project_config(start_dir=None)
            assert result is not None

    def test_non_ascii_path(self, tmp_path: Path):
        """Works with non-ASCII (Korean) directory names."""
        from forma.project_config import find_project_config

        korean_dir = tmp_path / "프로젝트"
        korean_dir.mkdir()
        config_file = korean_dir / "forma.yaml"
        config_file.write_text("course_name: 테스트\n", encoding="utf-8")

        result = find_project_config(start_dir=korean_dir)
        assert result is not None
        assert result.resolve() == config_file.resolve()


# ---------------------------------------------------------------------------
# T004-C: load_project_config() tests
# ---------------------------------------------------------------------------
class TestLoadProjectConfig:
    """Tests for load_project_config()."""

    def test_load_valid_yaml(self, tmp_path: Path):
        """Loads a valid YAML config and returns a dict."""
        from forma.project_config import load_project_config

        config_file = tmp_path / "forma.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            project:
              course_name: "인체구조와기능"
              year: 2026
              semester: 1
            classes:
              identifiers: [A, B, C]
            """),
            encoding="utf-8",
        )

        result = load_project_config(config_file)
        assert isinstance(result, dict)
        assert result["project"]["course_name"] == "인체구조와기능"
        assert result["project"]["year"] == 2026
        assert result["classes"]["identifiers"] == ["A", "B", "C"]

    def test_load_empty_file(self, tmp_path: Path):
        """Empty YAML file returns empty dict."""
        from forma.project_config import load_project_config

        config_file = tmp_path / "forma.yaml"
        config_file.write_text("", encoding="utf-8")

        result = load_project_config(config_file)
        assert result == {}

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Raises FileNotFoundError for nonexistent file."""
        from forma.project_config import load_project_config

        with pytest.raises(FileNotFoundError):
            load_project_config(tmp_path / "missing.yaml")

    def test_load_invalid_yaml_syntax(self, tmp_path: Path):
        """Raises yaml.YAMLError for syntactically invalid YAML."""
        from forma.project_config import load_project_config

        config_file = tmp_path / "forma.yaml"
        config_file.write_text("{{invalid: yaml::", encoding="utf-8")

        with pytest.raises(yaml.YAMLError):
            load_project_config(config_file)

    def test_load_utf8_encoding(self, tmp_path: Path):
        """Correctly reads UTF-8 Korean text."""
        from forma.project_config import load_project_config

        config_file = tmp_path / "forma.yaml"
        config_file.write_text(
            "project:\n  course_name: 간호학과\n",
            encoding="utf-8",
        )

        result = load_project_config(config_file)
        assert result["project"]["course_name"] == "간호학과"

    def test_load_flat_config(self, tmp_path: Path):
        """Loads flat (non-nested) YAML config."""
        from forma.project_config import load_project_config

        config_file = tmp_path / "forma.yaml"
        config_file.write_text("current_week: 5\n", encoding="utf-8")

        result = load_project_config(config_file)
        assert result["current_week"] == 5


# ---------------------------------------------------------------------------
# T004-D: validate_project_config() tests
# ---------------------------------------------------------------------------
class TestValidateProjectConfig:
    """Tests for validate_project_config()."""

    def test_valid_config_passes(self):
        """A fully valid config dict produces no errors."""
        from forma.project_config import validate_project_config

        config = {
            "project": {
                "course_name": "테스트",
                "year": 2026,
                "semester": 1,
                "grade": 2,
            },
            "classes": {
                "identifiers": ["A", "B"],
                "join_pattern": "join_{class}.yaml",
                "eval_pattern": "eval_{class}/",
            },
            "reports": {"dpi": 150, "skip_llm": False},
            "current_week": 3,
        }
        # Should not raise
        validate_project_config(config)

    def test_unknown_top_level_key_warns(self, caplog):
        """Unknown top-level keys produce a warning, not an error."""
        from forma.project_config import validate_project_config

        config = {"unknown_section": {"foo": "bar"}}
        with caplog.at_level(logging.WARNING):
            validate_project_config(config)
        assert any("unknown_section" in r.message for r in caplog.records)

    def test_unknown_nested_key_warns(self, caplog):
        """Unknown nested keys within known sections produce a warning."""
        from forma.project_config import validate_project_config

        config = {"project": {"unknown_key": "value"}}
        with caplog.at_level(logging.WARNING):
            validate_project_config(config)
        assert any("unknown_key" in r.message for r in caplog.records)

    def test_type_error_year_string(self):
        """String value for year raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"project": {"year": "not_a_number"}}
        with pytest.raises(ValueError, match="year"):
            validate_project_config(config)

    def test_type_error_dpi_string(self):
        """String value for dpi raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"reports": {"dpi": "abc"}}
        with pytest.raises(ValueError, match="dpi"):
            validate_project_config(config)

    def test_value_error_year_too_low(self):
        """Year below 2020 raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"project": {"year": 2019}}
        with pytest.raises(ValueError, match="year"):
            validate_project_config(config)

    def test_value_error_semester_invalid(self):
        """Semester not in {1, 2} raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"project": {"semester": 3}}
        with pytest.raises(ValueError, match="semester"):
            validate_project_config(config)

    def test_value_error_dpi_too_low(self):
        """DPI below 72 raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"reports": {"dpi": 50}}
        with pytest.raises(ValueError, match="dpi"):
            validate_project_config(config)

    def test_value_error_dpi_too_high(self):
        """DPI above 600 raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"reports": {"dpi": 1200}}
        with pytest.raises(ValueError, match="dpi"):
            validate_project_config(config)

    def test_value_error_num_questions_zero(self):
        """num_questions < 1 raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"ocr": {"num_questions": 0}}
        with pytest.raises(ValueError, match="num_questions"):
            validate_project_config(config)

    def test_value_error_negative_num_questions(self):
        """Negative num_questions raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"ocr": {"num_questions": -3}}
        with pytest.raises(ValueError, match="num_questions"):
            validate_project_config(config)

    def test_value_error_provider_invalid(self):
        """Provider not in {"gemini", "anthropic"} raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"evaluation": {"provider": "openai"}}
        with pytest.raises(ValueError, match="provider"):
            validate_project_config(config)

    def test_value_error_n_calls_zero(self):
        """n_calls < 1 raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"evaluation": {"n_calls": 0}}
        with pytest.raises(ValueError, match="n_calls"):
            validate_project_config(config)

    def test_value_error_join_pattern_missing_placeholder(self):
        """join_pattern without {class} raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"classes": {"join_pattern": "join_A.yaml"}}
        with pytest.raises(ValueError, match="join_pattern"):
            validate_project_config(config)

    def test_value_error_eval_pattern_missing_placeholder(self):
        """eval_pattern without {class} raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"classes": {"eval_pattern": "eval_A/"}}
        with pytest.raises(ValueError, match="eval_pattern"):
            validate_project_config(config)

    def test_join_pattern_empty_passes(self):
        """Empty join_pattern passes (no {class} check needed)."""
        from forma.project_config import validate_project_config

        config = {"classes": {"join_pattern": ""}}
        validate_project_config(config)  # No error

    def test_all_errors_collected(self):
        """Multiple errors are collected and reported together."""
        from forma.project_config import validate_project_config

        config = {
            "project": {"year": "abc", "semester": 5},
            "reports": {"dpi": -1},
        }
        with pytest.raises(ValueError) as exc_info:
            validate_project_config(config)
        msg = str(exc_info.value)
        assert "year" in msg
        assert "semester" in msg
        assert "dpi" in msg

    def test_empty_config_passes(self):
        """Empty config dict is valid (all keys optional)."""
        from forma.project_config import validate_project_config

        validate_project_config({})  # Should not raise

    def test_type_error_identifiers_not_list(self):
        """identifiers must be a list."""
        from forma.project_config import validate_project_config

        config = {"classes": {"identifiers": "A"}}
        with pytest.raises(ValueError, match="identifiers"):
            validate_project_config(config)

    def test_type_error_skip_llm_not_bool(self):
        """skip_llm must be bool."""
        from forma.project_config import validate_project_config

        config = {"reports": {"skip_llm": "yes"}}
        with pytest.raises(ValueError, match="skip_llm"):
            validate_project_config(config)

    def test_grade_too_low(self):
        """Grade < 1 raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"project": {"grade": 0}}
        with pytest.raises(ValueError, match="grade"):
            validate_project_config(config)

    def test_current_week_too_low(self):
        """current_week < 1 raises ValueError."""
        from forma.project_config import validate_project_config

        config = {"current_week": 0}
        with pytest.raises(ValueError, match="current_week"):
            validate_project_config(config)


# ---------------------------------------------------------------------------
# T004-E: merge_configs() tests
# ---------------------------------------------------------------------------
class TestMergeConfigs:
    """Tests for merge_configs()."""

    def test_cli_overrides_project_config(self):
        """CLI args override project config values."""
        from forma.project_config import merge_configs

        cli_ns = argparse.Namespace(dpi=300, verbose=True, font_path=None)
        project = {"reports": {"dpi": 150}}
        system = {}
        # explicit_keys tells which CLI args were explicitly set
        result = merge_configs(cli_ns, project, system, explicit_keys={"dpi"})
        assert result["dpi"] == 300

    def test_project_overrides_system_config(self):
        """Project config overrides system config values."""
        from forma.project_config import merge_configs

        cli_ns = argparse.Namespace(dpi=150)
        project = {"reports": {"dpi": 200}}
        system = {"dpi": 100}
        result = merge_configs(cli_ns, project, system, explicit_keys=set())
        assert result["dpi"] == 200

    def test_system_config_fallback(self):
        """System config provides fallback when CLI and project don't set value."""
        from forma.project_config import merge_configs

        cli_ns = argparse.Namespace(provider="gemini")
        project = {}
        system = {"provider": "anthropic"}
        result = merge_configs(cli_ns, project, system, explicit_keys=set())
        assert result["provider"] == "anthropic"

    def test_default_fallback(self):
        """Argparse default is used when no other layer sets value."""
        from forma.project_config import merge_configs

        cli_ns = argparse.Namespace(dpi=150)
        project = {}
        system = {}
        result = merge_configs(cli_ns, project, system, explicit_keys=set())
        assert result["dpi"] == 150

    def test_full_merge_precedence(self):
        """Full three-layer merge with correct precedence."""
        from forma.project_config import merge_configs

        cli_ns = argparse.Namespace(
            dpi=300,
            provider="gemini",
            skip_llm=False,
            font_path=None,
        )
        project = {
            "reports": {"dpi": 200, "skip_llm": True},
            "evaluation": {"provider": "anthropic"},
        }
        system = {"provider": "gemini", "dpi": 100}

        # CLI explicitly set dpi=300
        result = merge_configs(
            cli_ns, project, system, explicit_keys={"dpi"},
        )
        assert result["dpi"] == 300  # CLI wins
        assert result["provider"] == "anthropic"  # project wins over system
        assert result["skip_llm"] is True  # project wins over default

    def test_project_config_nested_sections(self):
        """Project config nested sections are flattened correctly."""
        from forma.project_config import merge_configs

        cli_ns = argparse.Namespace(
            course_name="",
            year=0,
            num_questions=5,
        )
        project = {
            "project": {"course_name": "간호학과", "year": 2026},
            "ocr": {"num_questions": 10},
        }
        system = {}
        result = merge_configs(cli_ns, project, system, explicit_keys=set())
        assert result["course_name"] == "간호학과"
        assert result["year"] == 2026
        assert result["num_questions"] == 10

    def test_empty_project_config(self):
        """Empty project config falls through to defaults."""
        from forma.project_config import merge_configs

        cli_ns = argparse.Namespace(dpi=150, provider="gemini")
        result = merge_configs(cli_ns, {}, {}, explicit_keys=set())
        assert result["dpi"] == 150
        assert result["provider"] == "gemini"

    def test_none_values_not_override(self):
        """None CLI values (not explicitly set) don't override project config."""
        from forma.project_config import merge_configs

        cli_ns = argparse.Namespace(font_path=None, model_path=None)
        project = {"paths": {"font_path": "/some/font.ttf"}}
        system = {}
        result = merge_configs(cli_ns, project, system, explicit_keys=set())
        assert result["font_path"] == "/some/font.ttf"

    def test_class_identifiers_from_project(self):
        """List field (class_identifiers) loaded from project config."""
        from forma.project_config import merge_configs

        cli_ns = argparse.Namespace(class_identifiers=None)
        project = {"classes": {"identifiers": ["A", "B", "C"]}}
        system = {}
        result = merge_configs(cli_ns, project, system, explicit_keys=set())
        assert result["class_identifiers"] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# apply_project_config() tests
# ---------------------------------------------------------------------------
class TestApplyProjectConfig:
    """Tests for apply_project_config() helper."""

    def test_no_config_flag_skips_loading(self, tmp_path):
        """When no_config=True, project config is not loaded."""
        from forma.project_config import apply_project_config

        args = argparse.Namespace(no_config=True, dpi=150)
        result = apply_project_config(args, argv=["--no-config"])
        assert result.dpi == 150

    def test_no_forma_yaml_returns_unchanged(self, tmp_path, monkeypatch):
        """When no forma.yaml found, args are returned unchanged."""
        from forma.project_config import apply_project_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git").mkdir()  # sentinel stops upward search
        args = argparse.Namespace(no_config=False, dpi=150, font_path=None)
        result = apply_project_config(args, argv=[])
        assert result.dpi == 150
        assert result.font_path is None

    def test_forma_yaml_overrides_defaults(self, tmp_path, monkeypatch):
        """forma.yaml values override argparse defaults."""
        from forma.project_config import apply_project_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git").mkdir()
        config = {"reports": {"dpi": 300}}
        (tmp_path / "forma.yaml").write_text(
            yaml.dump(config), encoding="utf-8",
        )
        args = argparse.Namespace(no_config=False, dpi=150, font_path=None)
        result = apply_project_config(args, argv=[])
        assert result.dpi == 300

    def test_explicit_cli_flag_overrides_config(self, tmp_path, monkeypatch):
        """Explicitly-set CLI flag overrides forma.yaml value."""
        from forma.project_config import apply_project_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git").mkdir()
        config = {"reports": {"dpi": 300}}
        (tmp_path / "forma.yaml").write_text(
            yaml.dump(config), encoding="utf-8",
        )
        args = argparse.Namespace(no_config=False, dpi=200, font_path=None)
        result = apply_project_config(args, argv=["--dpi", "200"])
        assert result.dpi == 200  # CLI explicit wins

    def test_invalid_config_warns_and_continues(self, tmp_path, monkeypatch, caplog):
        """Invalid config logs warning and returns args unchanged."""
        from forma.project_config import apply_project_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git").mkdir()
        config = {"project": {"year": "not_an_int"}}
        (tmp_path / "forma.yaml").write_text(
            yaml.dump(config), encoding="utf-8",
        )
        args = argparse.Namespace(no_config=False, dpi=150)
        with caplog.at_level(logging.WARNING):
            result = apply_project_config(args, argv=[])
        assert result.dpi == 150  # unchanged
        assert "validation failed" in caplog.text.lower()

    def test_explicit_keys_detection_from_argv(self, tmp_path, monkeypatch):
        """argv flags are correctly mapped to explicit_keys."""
        from forma.project_config import apply_project_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git").mkdir()
        config = {"reports": {"dpi": 300, "skip_llm": True}}
        (tmp_path / "forma.yaml").write_text(
            yaml.dump(config), encoding="utf-8",
        )
        args = argparse.Namespace(
            no_config=False, dpi=100, skip_llm=False, font_path=None,
        )
        # --dpi is explicit, --skip-llm is not in argv
        result = apply_project_config(args, argv=["--dpi", "100"])
        assert result.dpi == 100       # explicit CLI wins
        assert result.skip_llm is True  # config wins (not in argv)
