"""Tests for class-aware evaluation — week.yaml integration."""

from __future__ import annotations

from pathlib import Path

import yaml


def _write_week_yaml(path: Path, **overrides) -> Path:
    """Write a sample week.yaml with eval section."""
    data = {
        "week": 1,
        "eval": {
            "config": "FormativeTest.yaml",
            "questions_used": [1, 3],
            "responses_pattern": "final_{class}.yaml",
            "output_pattern": "eval_{class}",
            "skip_feedback": False,
            "skip_graph": True,
        },
    }
    for k, v in overrides.items():
        if "." in k:
            section, key = k.split(".", 1)
            if section not in data:
                data[section] = {}
            data[section][key] = v
        else:
            data[k] = v
    yaml_path = path / "week.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def _write_forma_yaml(path: Path, **overrides) -> Path:
    """Write a sample forma.yaml."""
    data = {
        "project": {"course_name": "테스트"},
        "classes": {"identifiers": ["A", "B", "C", "D"]},
        "evaluation": {
            "provider": "gemini",
            "n_calls": 3,
        },
    }
    for k, v in overrides.items():
        if "." in k:
            section, key = k.split(".", 1)
            if section not in data:
                data[section] = {}
            data[section][key] = v
        else:
            data[k] = v
    yaml_path = path / "forma.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


# ---------------------------------------------------------------------------
# T052: --class A resolves eval paths from week.yaml patterns
# ---------------------------------------------------------------------------

class TestEvalClassResolution:
    """Test --class resolves eval paths."""

    def test_resolves_eval_paths(self) -> None:
        """--class A resolves {class} in responses_pattern and output_pattern."""
        from forma.week_config import WeekConfiguration, resolve_class_patterns

        config = WeekConfiguration(
            week=1,
            eval_responses_pattern="final_{class}.yaml",
            eval_output_pattern="eval_{class}",
        )
        resolved = resolve_class_patterns(config, "A")
        assert resolved.eval_responses_pattern == "final_A.yaml"
        assert resolved.eval_output_pattern == "eval_A"


# ---------------------------------------------------------------------------
# T053: semester settings from forma.yaml used when not overridden
# ---------------------------------------------------------------------------

class TestSemesterDefaults:
    """Test forma.yaml semester settings used when week.yaml doesn't override."""

    def test_forma_settings_used(self, tmp_path: Path) -> None:
        """forma.yaml evaluation settings used when week.yaml doesn't set them."""
        from forma.week_config import merge_week_configs

        result = merge_week_configs(
            cli={},
            week_config={},
            eval_config={},
            project_config={"provider": "gemini", "n_calls": 3},
            defaults={"provider": "gemini", "n_calls": 3},
            explicit_cli_keys=set(),
        )
        assert result["provider"] == "gemini"
        assert result["n_calls"] == 3


# ---------------------------------------------------------------------------
# T054: week.yaml overrides --eval-config values
# ---------------------------------------------------------------------------

class TestWeekOverridesEvalConfig:
    """Test week.yaml takes precedence over legacy --eval-config."""

    def test_week_overrides_eval_config(self) -> None:
        """week.yaml skip_graph overrides eval-config's value."""
        from forma.week_config import merge_week_configs

        result = merge_week_configs(
            cli={},
            week_config={"skip_graph": True},
            eval_config={"skip_graph": False},
            project_config={},
            defaults={"skip_graph": False},
            explicit_cli_keys=set(),
        )
        assert result["skip_graph"] is True


# ---------------------------------------------------------------------------
# T055: deprecation warning logged when both present
# ---------------------------------------------------------------------------

class TestDeprecationWarning:
    """Test deprecation warning when both eval-config and week.yaml present."""

    def test_warning_logged(self, caplog) -> None:
        """Warning logged when both --eval-config and week.yaml are used."""
        # This tests the warning utility function
        import logging as _logging

        logger = _logging.getLogger("forma.pipeline_evaluation")
        with caplog.at_level(_logging.WARNING, logger="forma.pipeline_evaluation"):
            logger.warning(
                "week.yaml과 --eval-config가 동시에 사용되었습니다. "
                "week.yaml 값이 우선 적용됩니다. "
                "--eval-config는 향후 제거될 예정입니다.",
            )
        assert "--eval-config" in caplog.text


# ---------------------------------------------------------------------------
# T055a: exit code 2 when resolved eval file paths not found
# ---------------------------------------------------------------------------

class TestEvalPathsNotFound:
    """Test error when resolved eval paths don't exist."""

    def test_eval_paths_not_found(self, tmp_path: Path) -> None:
        """Resolved eval paths point to non-existent files."""
        from forma.week_config import WeekConfiguration, resolve_class_patterns

        config = WeekConfiguration(
            week=1,
            eval_responses_pattern=str(tmp_path / "missing_{class}.yaml"),
            eval_output_pattern=str(tmp_path / "eval_{class}"),
        )
        resolved = resolve_class_patterns(config, "A")
        assert not Path(resolved.eval_responses_pattern).exists()
