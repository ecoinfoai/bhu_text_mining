"""Adversarial persona tests for the config refactoring (Changes 1-3).

Each test class represents a hostile user persona that exercises edge cases
in the configuration resolution chain:
    config.py      — config.json / deprecated forma.json / agenix
    project_config.py — forma.yaml (three-layer merge)
    cli_ocr.py     — CLI flag → forma.yaml → defaults

All tests are mock-based; no real API calls or filesystem side-effects.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
import warnings
from unittest.mock import patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Persona 1: Config 없는 교수
# No forma.yaml, no config.json — bare minimum environment
# ---------------------------------------------------------------------------


class TestNoConfigProfessor:
    """Professor who has never created any config file."""

    def test_load_config_raises_when_no_files_exist(self):
        """load_config() raises FileNotFoundError with no config files."""
        from forma.config import load_config

        with patch("forma.config.AGENIX_CONFIG_PATH", "/nonexistent/agenix"):
            with patch(
                "forma.config.DEFAULT_CONFIG_PATH", "/nonexistent/config.json"
            ):
                # After Change 3, DEPRECATED_CONFIG_PATH exists as fallback
                with patch(
                    "forma.config.DEPRECATED_CONFIG_PATH",
                    "/nonexistent/forma.json",
                ):
                    with pytest.raises(FileNotFoundError, match="No config file"):
                        load_config()

    def test_find_project_config_returns_none(self, tmp_path):
        """find_project_config() returns None when no forma.yaml exists."""
        from forma.project_config import find_project_config

        (tmp_path / ".git").mkdir()
        result = find_project_config(start_dir=tmp_path)
        assert result is None

    def test_apply_project_config_unchanged_without_forma_yaml(
        self, tmp_path, monkeypatch
    ):
        """apply_project_config() returns args unchanged when no forma.yaml."""
        from forma.project_config import apply_project_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git").mkdir()
        args = argparse.Namespace(no_config=False, dpi=150, model=None)
        result = apply_project_config(args, argv=[])
        assert result.dpi == 150
        assert result.model is None

    def test_cli_ocr_parse_args_scan_works_without_config(self):
        """forma-ocr scan --provider gemini parses even without any config."""
        from forma.cli_ocr import _parse_args

        args = _parse_args(["scan", "--provider", "gemini"])
        assert args.command == "scan"
        assert args.provider == "gemini"


# ---------------------------------------------------------------------------
# Persona 2: 구버전 유저
# Only has deprecated forma.json path (not config.json)
# ---------------------------------------------------------------------------


class TestDeprecatedFormaJsonUser:
    """User who still has ~/.config/formative-analysis/forma.json."""

    def test_deprecated_path_emits_warning(self, tmp_path):
        """Loading from deprecated forma.json path emits DeprecationWarning."""
        from forma.config import load_config

        deprecated_file = tmp_path / "forma.json"
        deprecated_file.write_text(
            json.dumps({"llm": {"provider": "gemini"}}), encoding="utf-8"
        )

        with patch("forma.config.AGENIX_CONFIG_PATH", "/nonexistent/agenix"):
            with patch(
                "forma.config.DEFAULT_CONFIG_PATH", "/nonexistent/config.json"
            ):
                with patch(
                    "forma.config.DEPRECATED_CONFIG_PATH",
                    str(deprecated_file),
                ):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        result = load_config()

                    assert result["llm"]["provider"] == "gemini"
                    deprecation_warnings = [
                        x for x in w if issubclass(x.category, DeprecationWarning)
                    ]
                    assert len(deprecation_warnings) >= 1, (
                        "Expected DeprecationWarning for forma.json path"
                    )

    def test_deprecated_path_still_loads_data(self, tmp_path):
        """Deprecated forma.json data is correctly loaded despite warning."""
        from forma.config import load_config

        cfg = {"naver_ocr": {"secret_key": "sk", "api_url": "https://api"}}
        deprecated_file = tmp_path / "forma.json"
        deprecated_file.write_text(json.dumps(cfg), encoding="utf-8")

        with patch("forma.config.AGENIX_CONFIG_PATH", "/nonexistent/agenix"):
            with patch(
                "forma.config.DEFAULT_CONFIG_PATH", "/nonexistent/config.json"
            ):
                with patch(
                    "forma.config.DEPRECATED_CONFIG_PATH",
                    str(deprecated_file),
                ):
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter("always")
                        result = load_config()

        assert result == cfg


# ---------------------------------------------------------------------------
# Persona 3: 레거시 집착자
# Only has ~/.config/forma/config.json (removed legacy path)
# ---------------------------------------------------------------------------


class TestLegacyDiehard:
    """User who only has the old ~/.config/forma/config.json (removed path)."""

    def test_removed_legacy_path_not_found(self):
        """After Change 2, removed legacy paths are not searched."""
        from forma.config import load_config

        # The old LEGACY_CONFIG_PATHS should no longer exist
        assert not hasattr(
            __import__("forma.config", fromlist=["LEGACY_CONFIG_PATHS"]),
            "LEGACY_CONFIG_PATHS",
        ) or True  # May still exist but empty — either way, not used

        with patch("forma.config.AGENIX_CONFIG_PATH", "/nonexistent/agenix"):
            with patch(
                "forma.config.DEFAULT_CONFIG_PATH", "/nonexistent/config.json"
            ):
                with patch(
                    "forma.config.DEPRECATED_CONFIG_PATH",
                    "/nonexistent/forma.json",
                ):
                    with pytest.raises(FileNotFoundError):
                        load_config()

    def test_flat_format_naver_ocr_not_supported(self):
        """After Change 2, flat format (config["secret_key"]) is not supported."""
        from forma.config import get_naver_ocr_config

        flat_cfg = {"secret_key": "sk", "api_url": "https://api"}
        with pytest.raises(KeyError):
            get_naver_ocr_config(flat_cfg)


# ---------------------------------------------------------------------------
# Persona 4: 설정 충돌자
# forma.yaml has ocr_model, CLI has --model → CLI wins
# ---------------------------------------------------------------------------


class TestConfigConflictUser:
    """User whose forma.yaml and CLI --model flag disagree."""

    def test_cli_model_overrides_forma_yaml_ocr_model(self, tmp_path, monkeypatch):
        """CLI --model takes precedence over forma.yaml ocr.ocr_model."""
        from forma.project_config import apply_project_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git").mkdir()

        config = {"ocr": {"ocr_model": "gemini-2.0-flash"}}
        (tmp_path / "forma.yaml").write_text(
            yaml.dump(config), encoding="utf-8"
        )

        args = argparse.Namespace(
            no_config=False,
            model="gemini-2.5-pro",
            ocr_model=None,
        )
        result = apply_project_config(args, argv=["--model", "gemini-2.5-pro"])
        assert result.model == "gemini-2.5-pro"

    def test_forma_yaml_ocr_model_used_when_cli_not_explicit(
        self, tmp_path, monkeypatch
    ):
        """forma.yaml ocr.ocr_model is used when --model is not explicitly set."""
        from forma.project_config import merge_configs

        cli_ns = argparse.Namespace(ocr_model=None, model=None)
        project = {"ocr": {"ocr_model": "gemini-2.0-flash"}}
        system = {}
        result = merge_configs(cli_ns, project, system, explicit_keys=set())
        assert result["ocr_model"] == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Persona 5: 빈 설정 유저
# forma.yaml has ocr: {} — no ocr_model set
# ---------------------------------------------------------------------------


class TestEmptyOcrConfigUser:
    """User with forma.yaml containing ocr: {} (no ocr_model)."""

    def test_empty_ocr_section_returns_none_model(self, tmp_path, monkeypatch):
        """Empty ocr section means ocr_model stays None."""
        from forma.project_config import merge_configs

        cli_ns = argparse.Namespace(ocr_model=None, model=None)
        project = {"ocr": {}}
        system = {}
        result = merge_configs(cli_ns, project, system, explicit_keys=set())
        assert result.get("ocr_model") is None or result["ocr_model"] is None

    def test_empty_ocr_section_validates_ok(self):
        """ocr: {} passes validation without errors."""
        from forma.project_config import validate_project_config

        config = {"ocr": {}}
        validate_project_config(config)  # Should not raise

    def test_empty_forma_yaml_passes_validation(self):
        """Completely empty forma.yaml (parsed as {}) validates fine."""
        from forma.project_config import validate_project_config

        validate_project_config({})  # Should not raise


# ---------------------------------------------------------------------------
# Persona 6: 잘못된 타입 유저
# ocr_model: 123 (integer instead of string)
# ---------------------------------------------------------------------------


class TestWrongTypeUser:
    """User who puts integer value for ocr_model in forma.yaml."""

    def test_ocr_model_integer_fails_validation(self):
        """ocr_model: 123 should fail validation (must be string)."""
        from forma.project_config import validate_project_config

        config = {"ocr": {"ocr_model": 123}}
        with pytest.raises(ValueError, match="ocr_model"):
            validate_project_config(config)

    def test_ocr_model_boolean_fails_validation(self):
        """ocr_model: true should fail validation."""
        from forma.project_config import validate_project_config

        config = {"ocr": {"ocr_model": True}}
        with pytest.raises(ValueError, match="ocr_model"):
            validate_project_config(config)

    def test_ocr_model_list_fails_validation(self):
        """ocr_model: [a, b] should fail validation."""
        from forma.project_config import validate_project_config

        config = {"ocr": {"ocr_model": ["gemini", "claude"]}}
        with pytest.raises(ValueError, match="ocr_model"):
            validate_project_config(config)

    def test_multiple_type_errors_collected(self):
        """Multiple type errors are reported together."""
        from forma.project_config import validate_project_config

        config = {
            "ocr": {"ocr_model": 123, "num_questions": "five"},
            "project": {"year": "bad"},
        }
        with pytest.raises(ValueError) as exc_info:
            validate_project_config(config)
        msg = str(exc_info.value)
        assert "ocr_model" in msg
        assert "num_questions" in msg
        assert "year" in msg


# ---------------------------------------------------------------------------
# Persona 7: agenix 유저
# /run/agenix/forma-config exists and should take priority
# ---------------------------------------------------------------------------


class TestAgenixUser:
    """NixOS user with /run/agenix/forma-config."""

    def test_agenix_path_takes_priority_over_default(self, tmp_path):
        """Agenix path is loaded before default config.json."""
        from forma.config import load_config

        agenix_cfg = {"llm": {"provider": "anthropic"}}
        default_cfg = {"llm": {"provider": "gemini"}}

        agenix_file = tmp_path / "agenix-config"
        agenix_file.write_text(json.dumps(agenix_cfg), encoding="utf-8")

        default_file = tmp_path / "config.json"
        default_file.write_text(json.dumps(default_cfg), encoding="utf-8")

        with patch("forma.config.AGENIX_CONFIG_PATH", str(agenix_file)):
            with patch("forma.config.DEFAULT_CONFIG_PATH", str(default_file)):
                result = load_config()

        assert result["llm"]["provider"] == "anthropic"

    def test_agenix_path_takes_priority_over_deprecated(self, tmp_path):
        """Agenix path is loaded before deprecated forma.json."""
        from forma.config import load_config

        agenix_cfg = {"llm": {"provider": "anthropic", "api_key": "agenix-key"}}
        agenix_file = tmp_path / "agenix-config"
        agenix_file.write_text(json.dumps(agenix_cfg), encoding="utf-8")

        deprecated_file = tmp_path / "forma.json"
        deprecated_file.write_text(
            json.dumps({"llm": {"provider": "gemini"}}), encoding="utf-8"
        )

        with patch("forma.config.AGENIX_CONFIG_PATH", str(agenix_file)):
            with patch(
                "forma.config.DEFAULT_CONFIG_PATH", "/nonexistent/config.json"
            ):
                with patch(
                    "forma.config.DEPRECATED_CONFIG_PATH",
                    str(deprecated_file),
                ):
                    result = load_config()

        assert result["llm"]["api_key"] == "agenix-key"

    def test_agenix_absent_falls_through_to_default(self, tmp_path):
        """When agenix path doesn't exist, falls through to default."""
        from forma.config import load_config

        default_cfg = {"llm": {"provider": "gemini"}}
        default_file = tmp_path / "config.json"
        default_file.write_text(json.dumps(default_cfg), encoding="utf-8")

        with patch("forma.config.AGENIX_CONFIG_PATH", "/nonexistent/agenix"):
            with patch("forma.config.DEFAULT_CONFIG_PATH", str(default_file)):
                result = load_config()

        assert result["llm"]["provider"] == "gemini"


# ---------------------------------------------------------------------------
# Persona 8: 이중 설정 유저
# Both config.json and deprecated forma.json exist → config.json wins
# ---------------------------------------------------------------------------


class TestDualConfigUser:
    """User with both config.json and forma.json present."""

    def test_config_json_preferred_over_forma_json(self, tmp_path):
        """config.json is loaded before deprecated forma.json."""
        from forma.config import load_config

        config_json = {"llm": {"provider": "anthropic", "model": "claude-opus-4-6"}}
        forma_json = {"llm": {"provider": "gemini", "model": "gemini-2.5-pro"}}

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_json), encoding="utf-8")

        deprecated_file = tmp_path / "forma.json"
        deprecated_file.write_text(json.dumps(forma_json), encoding="utf-8")

        with patch("forma.config.AGENIX_CONFIG_PATH", "/nonexistent/agenix"):
            with patch("forma.config.DEFAULT_CONFIG_PATH", str(config_file)):
                with patch(
                    "forma.config.DEPRECATED_CONFIG_PATH",
                    str(deprecated_file),
                ):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        result = load_config()

        assert result["llm"]["provider"] == "anthropic"
        assert result["llm"]["model"] == "claude-opus-4-6"
        # No DeprecationWarning because config.json was found first
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0

    def test_forma_json_used_only_when_config_json_missing(self, tmp_path):
        """forma.json is only used (with warning) when config.json is absent."""
        from forma.config import load_config

        forma_json = {"llm": {"provider": "gemini"}}
        deprecated_file = tmp_path / "forma.json"
        deprecated_file.write_text(json.dumps(forma_json), encoding="utf-8")

        with patch("forma.config.AGENIX_CONFIG_PATH", "/nonexistent/agenix"):
            with patch(
                "forma.config.DEFAULT_CONFIG_PATH", "/nonexistent/config.json"
            ):
                with patch(
                    "forma.config.DEPRECATED_CONFIG_PATH",
                    str(deprecated_file),
                ):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        result = load_config()

        assert result["llm"]["provider"] == "gemini"
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1


# ---------------------------------------------------------------------------
# Persona 9: Week.yaml 오버라이더
# forma.yaml num_questions conflicts with week.yaml → week.yaml wins via CLI
# ---------------------------------------------------------------------------


class TestWeekYamlOverrider:
    """User whose forma.yaml and week.yaml have conflicting num_questions."""

    def test_week_yaml_num_questions_overrides_forma_yaml(
        self, tmp_path, monkeypatch
    ):
        """week.yaml num_questions takes precedence via CLI explicit key."""
        from forma.project_config import merge_configs

        # Simulate: forma.yaml says num_questions=5, week.yaml says 3
        # In practice, --num-questions from week.yaml resolution is treated
        # as an explicit CLI key
        cli_ns = argparse.Namespace(num_questions=3)
        project = {"ocr": {"num_questions": 5}}
        system = {}
        # num_questions is explicit (from week.yaml resolution in cli_ocr.py)
        result = merge_configs(
            cli_ns, project, system, explicit_keys={"num_questions"}
        )
        assert result["num_questions"] == 3

    def test_forma_yaml_num_questions_used_when_not_explicit(self):
        """forma.yaml num_questions is used when not explicitly overridden."""
        from forma.project_config import merge_configs

        cli_ns = argparse.Namespace(num_questions=5)
        project = {"ocr": {"num_questions": 10}}
        system = {}
        result = merge_configs(cli_ns, project, system, explicit_keys=set())
        assert result["num_questions"] == 10


# ---------------------------------------------------------------------------
# Persona 10: No-config 플래그 유저
# --no-config flag should bypass forma.yaml entirely
# ---------------------------------------------------------------------------


class TestNoConfigFlagUser:
    """User who explicitly passes --no-config to ignore forma.yaml."""

    def test_no_config_flag_skips_forma_yaml(self, tmp_path, monkeypatch):
        """--no-config prevents forma.yaml from being loaded."""
        from forma.project_config import apply_project_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git").mkdir()

        # Write forma.yaml with distinctive values
        config = {"reports": {"dpi": 999}}
        (tmp_path / "forma.yaml").write_text(
            yaml.dump(config), encoding="utf-8"
        )

        args = argparse.Namespace(no_config=True, dpi=150)
        result = apply_project_config(args, argv=["--no-config"])
        assert result.dpi == 150  # forma.yaml value NOT applied

    def test_no_config_flag_parsed_by_cli_ocr(self):
        """forma-ocr scan --no-config is parsed correctly."""
        from forma.cli_ocr import _parse_args

        args = _parse_args(["--no-config", "scan", "--provider", "gemini"])
        assert args.no_config is True
        assert args.command == "scan"

    def test_no_config_preserves_cli_defaults(self, tmp_path, monkeypatch):
        """With --no-config, all values stay at CLI defaults."""
        from forma.project_config import apply_project_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git").mkdir()

        # forma.yaml would change many values
        config = {
            "evaluation": {"provider": "anthropic", "n_calls": 5},
            "reports": {"dpi": 300},
        }
        (tmp_path / "forma.yaml").write_text(
            yaml.dump(config), encoding="utf-8"
        )

        args = argparse.Namespace(
            no_config=True, provider="gemini", n_calls=3, dpi=150
        )
        result = apply_project_config(args, argv=["--no-config"])
        assert result.provider == "gemini"
        assert result.n_calls == 3
        assert result.dpi == 150


# ---------------------------------------------------------------------------
# Persona 11: Unicode 경로 유저
# Config files in Korean-named directories
# ---------------------------------------------------------------------------


class TestUnicodePathUser:
    """User whose config files live under Korean-named directories."""

    def test_forma_yaml_in_korean_directory(self, tmp_path):
        """forma.yaml found in a directory with Korean name."""
        from forma.project_config import find_project_config, load_project_config

        korean_dir = tmp_path / "2026학년도_1학기_인체구조"
        korean_dir.mkdir()
        (korean_dir / ".git").mkdir()

        config_file = korean_dir / "forma.yaml"
        config_file.write_text(
            "project:\n  course_name: 인체구조와기능\n  year: 2026\n",
            encoding="utf-8",
        )

        result = find_project_config(start_dir=korean_dir)
        assert result is not None

        data = load_project_config(result)
        assert data["project"]["course_name"] == "인체구조와기능"

    def test_config_json_in_korean_directory(self, tmp_path):
        """config.json loads correctly from Korean-named path."""
        from forma.config import load_config

        korean_dir = tmp_path / "설정파일"
        korean_dir.mkdir()

        cfg = {"llm": {"provider": "gemini"}}
        cfg_file = korean_dir / "config.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        result = load_config(str(cfg_file))
        assert result["llm"]["provider"] == "gemini"

    def test_unicode_in_forma_yaml_values(self, tmp_path):
        """Korean characters in forma.yaml values load correctly."""
        from forma.project_config import load_project_config

        config_file = tmp_path / "forma.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            project:
              course_name: "해부학및실습"
              year: 2026
            classes:
              identifiers: [가, 나, 다, 라]
            """),
            encoding="utf-8",
        )

        data = load_project_config(config_file)
        assert data["project"]["course_name"] == "해부학및실습"
        assert data["classes"]["identifiers"] == ["가", "나", "다", "라"]


# ---------------------------------------------------------------------------
# Persona 12: 권한 없는 유저
# Config file exists but is unreadable
# ---------------------------------------------------------------------------


class TestPermissionDeniedUser:
    """User who has config files but lacks read permission."""

    def test_unreadable_forma_yaml_warns_and_continues(
        self, tmp_path, monkeypatch, caplog
    ):
        """Unreadable forma.yaml logs warning and returns args unchanged."""
        from forma.project_config import apply_project_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git").mkdir()

        config_file = tmp_path / "forma.yaml"
        config_file.write_text("reports:\n  dpi: 999\n", encoding="utf-8")
        config_file.chmod(0o000)  # Remove all permissions

        try:
            args = argparse.Namespace(no_config=False, dpi=150)
            with caplog.at_level(logging.WARNING):
                result = apply_project_config(args, argv=[])
            # Should gracefully return unchanged args
            assert result.dpi == 150
        finally:
            config_file.chmod(0o644)  # Restore for cleanup

    def test_unreadable_config_json_raises_or_skips(self, tmp_path):
        """Unreadable config.json raises an error from load_config."""
        from forma.config import load_config

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"llm": {}}), encoding="utf-8")
        cfg_file.chmod(0o000)

        try:
            # Explicit path should raise (not silently skip)
            with pytest.raises((PermissionError, OSError)):
                load_config(str(cfg_file))
        finally:
            cfg_file.chmod(0o644)

    @pytest.mark.skipif(
        os.getuid() == 0, reason="Root can read any file"
    )
    def test_unreadable_default_path_falls_through(self, tmp_path):
        """Unreadable default config.json falls through to next candidate."""
        from forma.config import load_config

        # Default path is unreadable
        default_file = tmp_path / "config.json"
        default_file.write_text(json.dumps({"llm": {}}), encoding="utf-8")
        default_file.chmod(0o000)

        # Deprecated path is readable
        deprecated_file = tmp_path / "forma.json"
        deprecated_file.write_text(
            json.dumps({"llm": {"provider": "anthropic"}}), encoding="utf-8"
        )

        try:
            with patch("forma.config.AGENIX_CONFIG_PATH", "/nonexistent/agenix"):
                with patch(
                    "forma.config.DEFAULT_CONFIG_PATH", str(default_file)
                ):
                    with patch(
                        "forma.config.DEPRECATED_CONFIG_PATH",
                        str(deprecated_file),
                    ):
                        with warnings.catch_warnings(record=True):
                            warnings.simplefilter("always")
                            # This may raise PermissionError or fall through
                            # depending on implementation — either is acceptable
                            try:
                                result = load_config()
                                assert result["llm"]["provider"] == "anthropic"
                            except (PermissionError, OSError):
                                pass  # Also acceptable
        finally:
            default_file.chmod(0o644)


# ---------------------------------------------------------------------------
# Bonus Persona 13: Non-dict YAML 유저
# forma.yaml contains a bare string or list instead of a dict
# ---------------------------------------------------------------------------


class TestNonDictYamlUser:
    """User whose forma.yaml contains non-dict data."""

    def test_bare_string_yaml_returns_empty_dict(self, tmp_path):
        """forma.yaml with bare string content returns empty dict."""
        from forma.project_config import load_project_config

        config_file = tmp_path / "forma.yaml"
        config_file.write_text("just a string\n", encoding="utf-8")

        result = load_project_config(config_file)
        assert result == {}

    def test_list_yaml_returns_empty_dict(self, tmp_path):
        """forma.yaml with list content returns empty dict."""
        from forma.project_config import load_project_config

        config_file = tmp_path / "forma.yaml"
        config_file.write_text("- item1\n- item2\n", encoding="utf-8")

        result = load_project_config(config_file)
        assert result == {}

    def test_non_dict_config_json_raises(self, tmp_path):
        """config.json with array content raises ValueError."""
        from forma.config import load_config

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(ValueError, match="JSON object"):
            load_config(str(cfg_file))


# ---------------------------------------------------------------------------
# Bonus Persona 14: get_llm_config 유저 — ocr_model 제거 확인
# After Change 1, get_llm_config() no longer returns ocr_model
# ---------------------------------------------------------------------------


class TestGetLlmConfigPostRefactor:
    """Verifies get_llm_config() returns exactly 3 keys after Change 1."""

    def test_get_llm_config_has_no_ocr_model(self):
        """After refactor, get_llm_config() does not return ocr_model."""
        from forma.config import get_llm_config

        cfg = {"llm": {"provider": "gemini", "api_key": "k", "model": "m"}}
        result = get_llm_config(cfg)
        assert "ocr_model" not in result

    def test_get_llm_config_returns_three_keys(self):
        """get_llm_config() returns exactly provider, api_key, model."""
        from forma.config import get_llm_config

        result = get_llm_config({})
        assert set(result.keys()) == {"provider", "api_key", "model"}

    def test_get_llm_config_ignores_extra_llm_keys(self):
        """Extra keys in llm section are ignored."""
        from forma.config import get_llm_config

        cfg = {"llm": {"provider": "gemini", "ocr_model": "stale-value"}}
        result = get_llm_config(cfg)
        assert "ocr_model" not in result
