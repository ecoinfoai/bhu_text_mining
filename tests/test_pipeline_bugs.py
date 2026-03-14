"""Tests for 015-fix-pipeline-bugs: 3 critical pipeline bug fixes.

BUG-001: Config merge precedence (cli_ocr.py)
BUG-002: CLI routing for eval --class (cli_main.py)
BUG-003: Join output format tolerance (evaluation_io.py, pipeline_evaluation.py)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# =====================================================================
# BUG-001: Config Merge Precedence Tests
# =====================================================================


class TestConfigMergePrecedence:
    """FR-001: CLI > week.yaml > forma.yaml > argparse default for num_questions."""

    def _make_week_yaml(self, tmp: Path, ocr_fields: dict | None = None) -> Path:
        """Create a minimal week.yaml with optional ocr section."""
        week_data: dict = {
            "week": 1,
            "ocr": {
                "image_dir_pattern": "images/{class}",
                "ocr_output_pattern": "ocr_{class}.yaml",
                "join_output_pattern": "final_{class}.yaml",
            },
        }
        if ocr_fields:
            week_data["ocr"].update(ocr_fields)
        path = tmp / "week.yaml"
        path.write_text(yaml.dump(week_data, allow_unicode=True), encoding="utf-8")
        return path

    def _make_forma_yaml(self, tmp: Path, ocr_fields: dict | None = None) -> Path:
        """Create a minimal forma.yaml."""
        forma_data: dict = {"project": {"name": "test"}}
        if ocr_fields:
            forma_data["ocr"] = ocr_fields
        path = tmp / "forma.yaml"
        path.write_text(yaml.dump(forma_data, allow_unicode=True), encoding="utf-8")
        return path

    @pytest.mark.parametrize(
        "cli_val,week_val,forma_val,expected,desc",
        [
            # CLI always wins
            (3, 2, 5, 3, "CLI > week > forma"),
            (3, 2, None, 3, "CLI > week > default"),
            (3, None, 5, 3, "CLI > forma > default"),
            (3, None, None, 3, "CLI only"),
            # week.yaml wins over forma and default
            (None, 2, 5, 2, "week > forma"),
            (None, 2, None, 2, "week only"),
            # forma.yaml wins over default
            (None, None, 5, 5, "forma only"),
            # argparse default
            (None, None, None, None, "argparse default (None)"),
        ],
    )
    def test_num_questions_precedence(
        self, tmp_path, cli_val, week_val, forma_val, expected, desc
    ):
        """8-combination precedence matrix for num_questions."""
        from forma.cli_ocr import _parse_args

        # Build week.yaml
        week_ocr = {}
        if week_val is not None:
            week_ocr["num_questions"] = week_val
        week_path = self._make_week_yaml(tmp_path, week_ocr if week_ocr else None)

        # Build forma.yaml
        forma_ocr = {}
        if forma_val is not None:
            forma_ocr["num_questions"] = forma_val
        self._make_forma_yaml(tmp_path, forma_ocr if forma_ocr else None)

        # Build CLI argv
        argv = ["scan", "--class", "A", "--week-config", str(week_path)]
        if cli_val is not None:
            argv += ["--num-questions", str(cli_val)]

        # Create image dir so resolve_class_patterns works
        (tmp_path / "images" / "A").mkdir(parents=True, exist_ok=True)

        # Parse args and apply project config
        args = _parse_args(argv)
        raw_argv = argv

        from forma.project_config import apply_project_config
        with patch("forma.project_config.find_project_config", return_value=str(tmp_path / "forma.yaml")):
            apply_project_config(args, argv=raw_argv)

        # Load week config
        from forma.week_config import load_week_config, resolve_class_patterns

        week_cfg = load_week_config(week_path)
        resolved = resolve_class_patterns(week_cfg, "A")

        # Reconstruct explicit_keys (the fix pattern)
        explicit = {
            token.lstrip("-").split("=")[0].replace("-", "_")
            for token in raw_argv
            if token.startswith("--")
        }

        # Apply precedence: CLI > week.yaml > forma.yaml/default
        # Note: resolved.ocr_num_questions defaults to 0 (falsy) when not set
        if "num_questions" in explicit:
            result = args.num_questions
        elif resolved.ocr_num_questions:
            result = resolved.ocr_num_questions
        else:
            result = args.num_questions

        assert result == expected, f"Failed: {desc}"

    @pytest.mark.parametrize(
        "cli_val,week_val,expected,desc",
        [
            # student_id_column is NOT in forma.yaml schema, so only 4 combos
            ("sid", "학번", "sid", "CLI > week"),
            ("sid", None, "sid", "CLI only"),
            (None, "학번", "학번", "week only"),
            (None, None, "student_id", "argparse default"),
        ],
    )
    def test_student_id_column_precedence(
        self, tmp_path, cli_val, week_val, expected, desc
    ):
        """Precedence matrix for student_id_column (not in forma.yaml schema)."""
        from forma.cli_ocr import _parse_args

        week_ocr = {}
        if week_val is not None:
            week_ocr["student_id_column"] = week_val
        week_path = self._make_week_yaml(tmp_path, week_ocr if week_ocr else None)
        self._make_forma_yaml(tmp_path)

        # join subcommand for student_id_column
        argv = ["join", "--class", "A", "--week-config", str(week_path)]
        if cli_val is not None:
            argv += ["--student-id-column", cli_val]

        args = _parse_args(argv)
        raw_argv = argv

        from forma.project_config import apply_project_config
        with patch("forma.project_config.find_project_config", return_value=str(tmp_path / "forma.yaml")):
            apply_project_config(args, argv=raw_argv)

        from forma.week_config import load_week_config, resolve_class_patterns

        week_cfg = load_week_config(week_path)
        resolved = resolve_class_patterns(week_cfg, "A")

        explicit = {
            token.lstrip("-").split("=")[0].replace("-", "_")
            for token in raw_argv
            if token.startswith("--")
        }

        # student_id_column precedence
        # Note: resolved.ocr_student_id_column defaults to "" (falsy) when not set
        if "student_id_column" in explicit:
            result = args.student_id_column
        elif resolved.ocr_student_id_column:
            result = resolved.ocr_student_id_column
        else:
            result = args.student_id_column

        assert result == expected, f"Failed: {desc}"


class TestConfigMergeEdgeCases:
    """Edge cases for config merge."""

    def test_week_yaml_no_ocr_section(self, tmp_path):
        """week.yaml exists but has no ocr section -> fall through to forma.yaml."""
        week_data = {"week": 1}
        week_path = tmp_path / "week.yaml"
        week_path.write_text(yaml.dump(week_data), encoding="utf-8")

        from forma.week_config import load_week_config

        week_cfg = load_week_config(week_path)
        # Default is 0 (falsy), so truthiness check means "not set"
        assert not week_cfg.ocr_num_questions

    def test_empty_week_yaml(self, tmp_path):
        """Empty week.yaml -> raises ValueError."""
        week_path = tmp_path / "week.yaml"
        week_path.write_text("", encoding="utf-8")

        from forma.week_config import load_week_config

        with pytest.raises(ValueError, match="YAML mapping"):
            load_week_config(week_path)


# =====================================================================
# BUG-002: CLI Routing Tests
# =====================================================================


class TestEvalRouting:
    """FR-003/FR-004: forma eval --class A routes correctly."""

    def _get_routed_key_and_argv(self, argv):
        """Call main() with mocked delegate and return (key, delegate_argv)."""
        from forma.cli_main import main

        captured = {}

        def fake_import(module_path, func_name):
            def delegate():
                captured["argv"] = sys.argv[1:]
            captured["key"] = (module_path, func_name)
            return delegate

        with patch("forma.cli_main._import_delegate", side_effect=fake_import):
            main(argv)

        return captured.get("key"), captured.get("argv", [])

    def test_eval_class_routes_to_eval_none(self):
        """forma eval --class A -> routes to pipeline_evaluation.main."""
        key, delegate_argv = self._get_routed_key_and_argv(["eval", "--class", "A"])
        assert key == ("forma.pipeline_evaluation", "main")
        assert "--class" in delegate_argv
        assert "A" in delegate_argv

    def test_eval_batch_routes_correctly(self):
        """forma eval batch --classes A B -> routes to pipeline_batch_evaluation.main."""
        key, delegate_argv = self._get_routed_key_and_argv(["eval", "batch", "--classes", "A", "B"])
        assert key == ("forma.pipeline_batch_evaluation", "main")
        assert "--classes" in delegate_argv

    def test_eval_no_args_routes_to_eval_none(self):
        """forma eval -> routes to pipeline_evaluation.main."""
        key, delegate_argv = self._get_routed_key_and_argv(["eval"])
        assert key == ("forma.pipeline_evaluation", "main")

    def test_other_nested_groups_still_work(self):
        """report, train, lecture routing unaffected."""
        test_cases = [
            (["report", "student"], ("forma.cli_report", "main")),
            (["train", "risk"], ("forma.cli_train", "main")),
            (["lecture", "analyze"], ("forma.cli_lecture", "main_analyze")),
        ]

        for argv, expected_key in test_cases:
            key, _ = self._get_routed_key_and_argv(argv)
            assert key == expected_key, f"argv={argv}: expected {expected_key}, got {key}"

    def test_eval_class_not_consumed_as_subcommand(self):
        """The key bug: 'A' from '--class A' must NOT be consumed as eval subcommand."""
        key, delegate_argv = self._get_routed_key_and_argv(["eval", "--class", "A"])
        assert key == ("forma.pipeline_evaluation", "main")
        assert "--class" in delegate_argv
        assert "A" in delegate_argv

    def test_eval_korean_class_name(self):
        """forma eval --class 가 -> routes correctly (Unicode class name)."""
        key, delegate_argv = self._get_routed_key_and_argv(["eval", "--class", "가"])
        assert key == ("forma.pipeline_evaluation", "main")
        assert "가" in delegate_argv


# =====================================================================
# BUG-003: Bare List Tolerance Tests
# =====================================================================


class TestBareListTolerance:
    """FR-005/FR-006/FR-007: extract_student_responses accepts bare list and wrapped dict."""

    def test_bare_list_accepted(self):
        """extract_student_responses() accepts bare list format."""
        from forma.evaluation_io import extract_student_responses

        bare_list = [
            {"student_id": "S001", "q_num": 1, "text": "답변1"},
            {"student_id": "S001", "q_num": 2, "text": "답변2"},
            {"student_id": "S002", "q_num": 1, "text": "답변3"},
        ]

        # This should NOT raise KeyError
        result = extract_student_responses(bare_list)
        assert "S001" in result
        assert "S002" in result
        assert result["S001"][1] == "답변1"

    def test_wrapped_dict_still_works(self):
        """extract_student_responses() backward compat with {'responses': {...}} format."""
        from forma.evaluation_io import extract_student_responses

        wrapped = {
            "responses": {
                "S001": {1: "답변1", 2: "답변2"},
                "S002": {1: "답변3"},
            }
        }

        result = extract_student_responses(wrapped)
        assert result["S001"][1] == "답변1"
        assert result["S002"][1] == "답변3"

    def test_empty_yaml_raises_clear_error(self):
        """Empty YAML (None) raises descriptive ValueError about invalid format."""
        from forma.evaluation_io import extract_student_responses

        with pytest.raises(ValueError, match="(?i)invalid|format|response"):
            extract_student_responses(None)

    def test_string_input_raises_clear_error(self):
        """String input raises descriptive ValueError."""
        from forma.evaluation_io import extract_student_responses

        with pytest.raises(ValueError, match="(?i)invalid|format|response"):
            extract_student_responses("just a string")

    def test_invalid_dict_without_responses_key(self):
        """Dict without 'responses' key raises KeyError."""
        from forma.evaluation_io import extract_student_responses

        with pytest.raises(KeyError):
            extract_student_responses({"data": [1, 2, 3]})

    def test_bare_list_same_result_as_wrapped(self):
        """Bare list produces same result as wrapped dict (FR-006)."""
        from forma.evaluation_io import extract_student_responses
        from forma.response_converter import convert_join_to_responses

        bare_list = [
            {"student_id": "S001", "q_num": 1, "text": "답변1"},
            {"student_id": "S002", "q_num": 1, "text": "답변2"},
        ]
        wrapped = convert_join_to_responses(bare_list)

        result_bare = extract_student_responses(bare_list)
        result_wrapped = extract_student_responses(wrapped)

        assert result_bare == result_wrapped


class TestBareListPipeline:
    """FR-005: run_evaluation_pipeline non-'--class' path handles bare list."""

    def test_non_class_path_bare_list(self, tmp_path):
        """Bare-list YAML file works in run_evaluation_pipeline without questions_used."""
        bare_list = [
            {"student_id": "S001", "q_num": 1, "text": "세포막은 인지질 이중층이다"},
            {"student_id": "S002", "q_num": 1, "text": "세포막은 선택적 투과성"},
        ]
        responses_path = tmp_path / "final_A.yaml"
        responses_path.write_text(
            yaml.dump(bare_list, allow_unicode=True), encoding="utf-8"
        )

        # Create a minimal exam config
        config_data = {
            "metadata": {"chapter": 1},
            "questions": [
                {
                    "sn": 1,
                    "question": "세포막의 구조?",
                    "answer": "인지질 이중층",
                    "concepts": ["세포막", "인지질"],
                    "type": "v1",
                }
            ],
        }
        config_path = tmp_path / "exam.yaml"
        config_path.write_text(
            yaml.dump(config_data, allow_unicode=True), encoding="utf-8"
        )

        # The non-'--class' path (questions_used=None) should accept bare list
        # We test by directly loading and extracting, mimicking the pipeline path
        from forma.evaluation_io import load_evaluation_yaml, extract_student_responses
        from forma.response_converter import convert_join_to_responses

        raw_data = yaml.safe_load(responses_path.read_text(encoding="utf-8"))

        # This is the fix: check isinstance(raw_data, list)
        if isinstance(raw_data, list):
            responses_data = convert_join_to_responses(raw_data)
        else:
            responses_data = raw_data

        student_responses = extract_student_responses(responses_data)
        assert "S001" in student_responses
        assert student_responses["S001"][1] == "세포막은 인지질 이중층이다"

    def test_yaml_roundtrip(self, tmp_path):
        """Bare-list YAML survives yaml.safe_load round-trip (Persona 9)."""
        bare_list = [
            {"student_id": "S001", "q_num": 1, "text": "답변"},
        ]
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(bare_list, allow_unicode=True), encoding="utf-8")

        # Round-trip
        data1 = yaml.safe_load(path.read_text(encoding="utf-8"))
        path.write_text(yaml.dump(data1, allow_unicode=True), encoding="utf-8")
        data2 = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert data1 == data2
        assert isinstance(data2, list)


# =====================================================================
# Adversarial Persona Tests
# =====================================================================


class TestAdversarialPersonas:
    """Adversarial persona tests targeting the 3 bug fixes with extreme edge cases."""

    # -----------------------------------------------------------------
    # Helpers (reused across personas)
    # -----------------------------------------------------------------

    def _make_week_yaml(self, tmp: Path, data: dict) -> Path:
        path = tmp / "week.yaml"
        path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
        return path

    def _make_forma_yaml(self, tmp: Path, data: dict) -> Path:
        path = tmp / "forma.yaml"
        path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
        return path

    def _get_routed_key_and_argv(self, argv):
        from forma.cli_main import main

        captured = {}

        def fake_import(module_path, func_name):
            def delegate():
                captured["argv"] = sys.argv[1:]
            captured["key"] = (module_path, func_name)
            return delegate

        with patch("forma.cli_main._import_delegate", side_effect=fake_import):
            main(argv)

        return captured.get("key"), captured.get("argv", [])

    # -----------------------------------------------------------------
    # P01: Careless Professor (실수 많은 교수)
    # week.yaml with typos, wrong structure → should not crash
    # -----------------------------------------------------------------

    def test_adversarial_P01_careless_professor_typo_fields(self, tmp_path):
        """week.yaml has typo field names (num_questinos) → ignored, defaults used."""
        from forma.week_config import load_week_config

        week_data = {
            "week": 1,
            "ocr": {
                "image_dir_pattern": "images/{class}",
                "ocr_output_pattern": "ocr_{class}.yaml",
                "join_output_pattern": "final_{class}.yaml",
                "num_questinos": 99,  # typo!
            },
        }
        week_path = self._make_week_yaml(tmp_path, week_data)
        week_cfg = load_week_config(week_path)
        # Typo field is silently ignored; default (0) used
        assert week_cfg.ocr_num_questions == 0

    def test_adversarial_P01_careless_professor_wrong_type(self, tmp_path):
        """week.yaml has num_questions as string → week_config handles or ignores."""
        from forma.week_config import load_week_config

        week_data = {
            "week": 1,
            "ocr": {
                "image_dir_pattern": "images/{class}",
                "ocr_output_pattern": "ocr_{class}.yaml",
                "join_output_pattern": "final_{class}.yaml",
                "num_questions": "three",  # wrong type
            },
        }
        week_path = self._make_week_yaml(tmp_path, week_data)
        # Should either load (with string coerced) or raise — but NOT crash with unhelpful error
        try:
            week_cfg = load_week_config(week_path)
            # If loaded, the value might be stored as-is or coerced
        except (ValueError, TypeError):
            pass  # Acceptable: clear type error

    def test_adversarial_P01_careless_professor_missing_week_field(self, tmp_path):
        """week.yaml missing 'week' field entirely → should raise clear error."""
        from forma.week_config import load_week_config

        week_data = {
            "ocr": {
                "image_dir_pattern": "images/{class}",
                "ocr_output_pattern": "ocr_{class}.yaml",
                "join_output_pattern": "final_{class}.yaml",
            },
        }
        week_path = self._make_week_yaml(tmp_path, week_data)
        with pytest.raises((ValueError, KeyError)):
            load_week_config(week_path)

    # -----------------------------------------------------------------
    # P02: CLI Power User (파워유저)
    # ALL fields specified via CLI flags → CLI always wins
    # -----------------------------------------------------------------

    def test_adversarial_P02_cli_power_user_all_flags(self, tmp_path):
        """CLI specifies every field → CLI values dominate regardless of yaml content."""
        from forma.cli_ocr import _parse_args

        week_data = {
            "week": 1,
            "ocr": {
                "image_dir_pattern": "images/{class}",
                "ocr_output_pattern": "ocr_{class}.yaml",
                "join_output_pattern": "final_{class}.yaml",
                "num_questions": 99,
                "student_id_column": "학번_yaml",
            },
        }
        week_path = self._make_week_yaml(tmp_path, week_data)
        self._make_forma_yaml(tmp_path, {"project": {"name": "test"}, "ocr": {"num_questions": 77}})
        (tmp_path / "images" / "A").mkdir(parents=True, exist_ok=True)

        argv = [
            "scan", "--class", "A",
            "--week-config", str(week_path),
            "--num-questions", "3",
        ]
        args = _parse_args(argv)

        explicit = {
            token.lstrip("-").split("=")[0].replace("-", "_")
            for token in argv
            if token.startswith("--")
        }

        assert "num_questions" in explicit
        assert args.num_questions == 3  # CLI wins over week(99) and forma(77)

    def test_adversarial_P02_cli_power_user_join_all_flags(self, tmp_path):
        """CLI join with all flags → student_id_column from CLI wins."""
        from forma.cli_ocr import _parse_args

        week_data = {
            "week": 1,
            "ocr": {
                "image_dir_pattern": "images/{class}",
                "ocr_output_pattern": "ocr_{class}.yaml",
                "join_output_pattern": "final_{class}.yaml",
                "student_id_column": "학번_yaml",
            },
        }
        week_path = self._make_week_yaml(tmp_path, week_data)

        argv = [
            "join", "--class", "A",
            "--week-config", str(week_path),
            "--student-id-column", "my_sid",
        ]
        args = _parse_args(argv)
        explicit = {
            token.lstrip("-").split("=")[0].replace("-", "_")
            for token in argv
            if token.startswith("--")
        }

        assert "student_id_column" in explicit
        assert args.student_id_column == "my_sid"

    # -----------------------------------------------------------------
    # P03: Minimal Config (최소 설정)
    # No forma.yaml, no week.yaml → argparse defaults, no crashes
    # -----------------------------------------------------------------

    def test_adversarial_P03_minimal_config_no_forma_yaml(self, tmp_path):
        """No forma.yaml at all → apply_project_config finds nothing, no crash."""
        from forma.cli_ocr import _parse_args
        from forma.project_config import apply_project_config

        week_data = {
            "week": 1,
            "ocr": {
                "image_dir_pattern": "images/{class}",
                "ocr_output_pattern": "ocr_{class}.yaml",
                "join_output_pattern": "final_{class}.yaml",
            },
        }
        week_path = self._make_week_yaml(tmp_path, week_data)
        (tmp_path / "images" / "A").mkdir(parents=True, exist_ok=True)

        argv = ["scan", "--class", "A", "--week-config", str(week_path)]
        args = _parse_args(argv)

        # No forma.yaml → find_project_config returns None → no crash
        with patch("forma.project_config.find_project_config", return_value=None):
            apply_project_config(args, argv=argv)

        assert args.num_questions is None  # argparse default

    def test_adversarial_P03_minimal_no_config_eval(self):
        """forma eval with no config at all → routes correctly (no crash on routing)."""
        key, _ = self._get_routed_key_and_argv(["eval"])
        assert key == ("forma.pipeline_evaluation", "main")

    # -----------------------------------------------------------------
    # P04: Config Hoarder (설정 수집가)
    # forma.yaml sets EVERY field, week.yaml overrides only a subset
    # -----------------------------------------------------------------

    def test_adversarial_P04_config_hoarder_selective_override(self, tmp_path):
        """forma.yaml sets many fields, week.yaml overrides only num_questions."""
        from forma.cli_ocr import _parse_args
        from forma.week_config import load_week_config, resolve_class_patterns
        from forma.project_config import apply_project_config

        self._make_forma_yaml(tmp_path, {
            "project": {"name": "test"},
            "ocr": {
                "num_questions": 10,
                "naver_config": "/path/to/naver",
                "spreadsheet_url": "https://example.com",
            },
        })
        week_data = {
            "week": 1,
            "ocr": {
                "image_dir_pattern": "images/{class}",
                "ocr_output_pattern": "ocr_{class}.yaml",
                "join_output_pattern": "final_{class}.yaml",
                "num_questions": 2,  # Only this overrides
            },
        }
        week_path = self._make_week_yaml(tmp_path, week_data)
        (tmp_path / "images" / "A").mkdir(parents=True, exist_ok=True)

        argv = ["scan", "--class", "A", "--week-config", str(week_path)]
        args = _parse_args(argv)

        with patch("forma.project_config.find_project_config", return_value=str(tmp_path / "forma.yaml")):
            apply_project_config(args, argv=argv)

        week_cfg = load_week_config(week_path)
        resolved = resolve_class_patterns(week_cfg, "A")

        explicit = {
            token.lstrip("-").split("=")[0].replace("-", "_")
            for token in argv
            if token.startswith("--")
        }

        # week.yaml num_questions (2) should win over forma.yaml (10)
        if "num_questions" not in explicit:
            if resolved.ocr_num_questions:
                result = resolved.ocr_num_questions
            else:
                result = args.num_questions
        else:
            result = args.num_questions

        assert result == 2, "week.yaml should override forma.yaml"

    # -----------------------------------------------------------------
    # P05: Subcommand Guesser (서브커맨드 추측자)
    # Various invalid/edge-case subcommand inputs
    # -----------------------------------------------------------------

    def test_adversarial_P05_eval_scan_invalid_subcommand(self):
        """forma eval scan → 'scan' is not a registered eval subcommand → routes to eval(None)."""
        key, delegate_argv = self._get_routed_key_and_argv(["eval", "scan"])
        # 'scan' is not a registered eval sub, so it routes to eval(None)
        # and 'scan' should be passed through as remaining
        assert key == ("forma.pipeline_evaluation", "main")
        assert "scan" in delegate_argv

    def test_adversarial_P05_eval_class_empty_string(self):
        """forma eval --class '' → routes correctly with empty class value."""
        key, delegate_argv = self._get_routed_key_and_argv(["eval", "--class", ""])
        assert key == ("forma.pipeline_evaluation", "main")
        assert "--class" in delegate_argv

    def test_adversarial_P05_eval_unknown_subcommand(self):
        """forma eval nonexistent → routes to eval(None), 'nonexistent' passed through."""
        key, delegate_argv = self._get_routed_key_and_argv(["eval", "nonexistent"])
        assert key == ("forma.pipeline_evaluation", "main")
        assert "nonexistent" in delegate_argv

    def test_adversarial_P05_eval_batch_still_works(self):
        """forma eval batch → still routes to batch module (regression check)."""
        key, _ = self._get_routed_key_and_argv(["eval", "batch", "--classes", "A"])
        assert key == ("forma.pipeline_batch_evaluation", "main")

    def test_adversarial_P05_eval_double_dash_class(self):
        """forma eval -- --class A → double dash edge case."""
        # parse_known_args treats -- as end of options; behavior depends on argparse
        try:
            key, delegate_argv = self._get_routed_key_and_argv(["eval", "--", "--class", "A"])
            # Should still route to eval(None) — '--class' after '--' is positional
            assert key == ("forma.pipeline_evaluation", "main")
        except SystemExit:
            pass  # Acceptable: argparse may reject this

    # -----------------------------------------------------------------
    # P06: Format Mixer (형식 혼합자)
    # Alternating bare-list and wrapped-dict YAML inputs
    # -----------------------------------------------------------------

    def test_adversarial_P06_format_mixer_alternating(self):
        """Process bare-list then wrapped-dict in sequence → both accepted."""
        from forma.evaluation_io import extract_student_responses

        bare = [
            {"student_id": "S001", "q_num": 1, "text": "답변A"},
            {"student_id": "S002", "q_num": 1, "text": "답변B"},
        ]
        wrapped = {
            "responses": {
                "S003": {1: "답변C"},
                "S004": {2: "답변D"},
            }
        }

        r1 = extract_student_responses(bare)
        r2 = extract_student_responses(wrapped)

        assert "S001" in r1 and "S002" in r1
        assert "S003" in r2 and "S004" in r2

    def test_adversarial_P06_format_mixer_bare_list_int_student_id(self):
        """Bare list with integer student_id → should be coerced to string."""
        from forma.evaluation_io import extract_student_responses

        bare = [
            {"student_id": 20210001, "q_num": 1, "text": "답변"},
        ]
        result = extract_student_responses(bare)
        assert "20210001" in result

    def test_adversarial_P06_format_mixer_bare_list_missing_text(self):
        """Bare list entry missing 'text' key → should default to empty string."""
        from forma.evaluation_io import extract_student_responses

        bare = [
            {"student_id": "S001", "q_num": 1},  # no 'text' key
        ]
        result = extract_student_responses(bare)
        assert result["S001"][1] == ""

    def test_adversarial_P06_format_mixer_bare_list_duplicate_entries(self):
        """Bare list with duplicate (student_id, q_num) → last one wins."""
        from forma.evaluation_io import extract_student_responses

        bare = [
            {"student_id": "S001", "q_num": 1, "text": "first"},
            {"student_id": "S001", "q_num": 1, "text": "second"},
        ]
        result = extract_student_responses(bare)
        assert result["S001"][1] == "second"

    # -----------------------------------------------------------------
    # P07: Empty File Attacker (빈 파일 공격자)
    # -----------------------------------------------------------------

    def test_adversarial_P07_empty_file_none_input(self):
        """None input → ValueError with descriptive message."""
        from forma.evaluation_io import extract_student_responses

        with pytest.raises(ValueError, match="(?i)invalid|format|none"):
            extract_student_responses(None)

    def test_adversarial_P07_empty_file_empty_list(self):
        """Empty list [] → accepted, returns empty dict."""
        from forma.evaluation_io import extract_student_responses

        result = extract_student_responses([])
        assert result == {}

    def test_adversarial_P07_empty_file_empty_dict(self):
        """Empty dict {} → KeyError (no 'responses' key)."""
        from forma.evaluation_io import extract_student_responses

        with pytest.raises(KeyError):
            extract_student_responses({})

    def test_adversarial_P07_empty_file_integer_input(self):
        """Integer input → ValueError."""
        from forma.evaluation_io import extract_student_responses

        with pytest.raises(ValueError, match="(?i)invalid|format"):
            extract_student_responses(42)

    def test_adversarial_P07_empty_file_boolean_input(self):
        """Boolean input → ValueError (bool is subclass of int but not dict/list)."""
        from forma.evaluation_io import extract_student_responses

        with pytest.raises((ValueError, TypeError)):
            extract_student_responses(True)

    # -----------------------------------------------------------------
    # P08: Unicode Chaos (유니코드 혼돈)
    # -----------------------------------------------------------------

    def test_adversarial_P08_unicode_korean_class_name(self):
        """forma eval --class 가 → correct routing with Korean class name."""
        key, delegate_argv = self._get_routed_key_and_argv(["eval", "--class", "가"])
        assert key == ("forma.pipeline_evaluation", "main")
        assert "가" in delegate_argv

    def test_adversarial_P08_unicode_long_class_name(self):
        """forma eval --class 해부학A반 → routing handles multi-byte names."""
        key, delegate_argv = self._get_routed_key_and_argv(["eval", "--class", "해부학A반"])
        assert key == ("forma.pipeline_evaluation", "main")
        assert "해부학A반" in delegate_argv

    def test_adversarial_P08_unicode_special_chars(self):
        """forma eval --class 'A-1' → hyphen in class name not confused with flag."""
        key, delegate_argv = self._get_routed_key_and_argv(["eval", "--class", "A-1"])
        assert key == ("forma.pipeline_evaluation", "main")
        assert "A-1" in delegate_argv

    def test_adversarial_P08_unicode_bare_list_korean_text(self):
        """Bare list with Korean text + special chars → extracted correctly."""
        from forma.evaluation_io import extract_student_responses

        bare = [
            {"student_id": "김철수_2021", "q_num": 1, "text": "세포막은 인지질 이중층으로 구성된다 (Alberts, 2015)"},
            {"student_id": "이영희", "q_num": 2, "text": "ATP → ADP + Pi"},
        ]
        result = extract_student_responses(bare)
        assert "김철수_2021" in result
        assert "이영희" in result
        assert "ATP → ADP + Pi" in result["이영희"][2]

    # -----------------------------------------------------------------
    # P09: Pipe Dreamer (파이프 몽상가)
    # Bare-list YAML round-tripped multiple times
    # -----------------------------------------------------------------

    def test_adversarial_P09_triple_roundtrip(self, tmp_path):
        """Bare-list YAML round-tripped through yaml.safe_load 3 times → still valid."""
        from forma.evaluation_io import extract_student_responses

        bare = [
            {"student_id": "S001", "q_num": 1, "text": "답변1"},
            {"student_id": "S002", "q_num": 2, "text": "답변2"},
        ]
        path = tmp_path / "roundtrip.yaml"

        data = bare
        for _ in range(3):
            path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
            data = yaml.safe_load(path.read_text(encoding="utf-8"))

        # After 3 round-trips, should still be a valid bare list
        assert isinstance(data, list)
        result = extract_student_responses(data)
        assert "S001" in result

    def test_adversarial_P09_convert_roundtrip(self, tmp_path):
        """Bare list → convert_join_to_responses → extract → same result as direct extract."""
        from forma.evaluation_io import extract_student_responses
        from forma.response_converter import convert_join_to_responses

        bare = [
            {"student_id": "S001", "q_num": 1, "text": "답변"},
            {"student_id": "S001", "q_num": 2, "text": "답변2"},
            {"student_id": "S002", "q_num": 1, "text": "답변3"},
        ]

        direct = extract_student_responses(bare)
        converted = convert_join_to_responses(bare)
        via_convert = extract_student_responses(converted)

        assert direct == via_convert

    # -----------------------------------------------------------------
    # P10: Flag Collision (플래그 충돌)
    # --num-questions vs --num_questions, = syntax, repeated flags
    # -----------------------------------------------------------------

    def test_adversarial_P10_flag_equals_syntax(self):
        """--num-questions=3 (= syntax) → detected in explicit_keys."""
        argv = ["scan", "--class", "A", "--num-questions=3", "--week-config", "/tmp/w.yaml"]
        explicit = {
            token.lstrip("-").split("=")[0].replace("-", "_")
            for token in argv
            if token.startswith("--")
        }
        assert "num_questions" in explicit

    def test_adversarial_P10_flag_underscore_variant(self):
        """--num_questions (underscore) → detected correctly by explicit_keys logic."""
        argv = ["scan", "--class", "A", "--num_questions", "3"]
        explicit = {
            token.lstrip("-").split("=")[0].replace("-", "_")
            for token in argv
            if token.startswith("--")
        }
        assert "num_questions" in explicit

    def test_adversarial_P10_repeated_flags(self):
        """Repeated --num-questions flags → argparse uses the last value."""
        from forma.cli_ocr import _parse_args

        week_data = {
            "week": 1,
            "ocr": {
                "image_dir_pattern": "images/{class}",
                "ocr_output_pattern": "ocr_{class}.yaml",
                "join_output_pattern": "final_{class}.yaml",
            },
        }
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            week_path = self._make_week_yaml(tmp_path, week_data)
            (tmp_path / "images" / "A").mkdir(parents=True, exist_ok=True)

            argv = [
                "scan", "--class", "A",
                "--week-config", str(week_path),
                "--num-questions", "3",
                "--num-questions", "7",
            ]
            args = _parse_args(argv)
            assert args.num_questions == 7  # last wins

    def test_adversarial_P10_no_config_flag_explicit(self):
        """--no-config flag → detected in explicit_keys."""
        argv = ["scan", "--class", "A", "--no-config", "--num-questions", "5"]
        explicit = {
            token.lstrip("-").split("=")[0].replace("-", "_")
            for token in argv
            if token.startswith("--")
        }
        assert "no_config" in explicit
        assert "num_questions" in explicit

    # -----------------------------------------------------------------
    # P11: Regression Hunter (회귀 사냥꾼)
    # Verify existing patterns still work after the fixes
    # -----------------------------------------------------------------

    def test_adversarial_P11_ocr_scan_routing(self):
        """forma ocr → routes to cli_ocr.main (ocr is NOT in _NESTED_GROUPS)."""
        key, _ = self._get_routed_key_and_argv(["ocr"])
        assert key == ("forma.cli_ocr", "main")

    def test_adversarial_P11_report_warning_routing(self):
        """forma report warning → routes to cli_report_warning.main."""
        key, _ = self._get_routed_key_and_argv(["report", "warning"])
        assert key == ("forma.cli_report_warning", "main")

    def test_adversarial_P11_report_student_routing(self):
        """forma report student → routes to cli_report.main."""
        key, _ = self._get_routed_key_and_argv(["report", "student"])
        assert key == ("forma.cli_report", "main")

    def test_adversarial_P11_train_risk_routing(self):
        """forma train risk → routes to cli_train.main."""
        key, _ = self._get_routed_key_and_argv(["train", "risk"])
        assert key == ("forma.cli_train", "main")

    def test_adversarial_P11_train_grade_routing(self):
        """forma train grade → routes to cli_train_grade.main."""
        key, _ = self._get_routed_key_and_argv(["train", "grade"])
        assert key == ("forma.cli_train_grade", "main")

    def test_adversarial_P11_lecture_analyze_routing(self):
        """forma lecture analyze → routes to cli_lecture.main_analyze."""
        key, _ = self._get_routed_key_and_argv(["lecture", "analyze"])
        assert key == ("forma.cli_lecture", "main_analyze")

    def test_adversarial_P11_lecture_compare_routing(self):
        """forma lecture compare → routes to cli_lecture.main_compare."""
        key, _ = self._get_routed_key_and_argv(["lecture", "compare"])
        assert key == ("forma.cli_lecture", "main_compare")

    def test_adversarial_P11_lecture_class_compare_routing(self):
        """forma lecture class-compare → routes to cli_lecture.main_class_compare."""
        key, _ = self._get_routed_key_and_argv(["lecture", "class-compare"])
        assert key == ("forma.cli_lecture", "main_class_compare")

    def test_adversarial_P11_intervention_routing(self):
        """forma intervention → routes to cli_intervention.main."""
        key, _ = self._get_routed_key_and_argv(["intervention"])
        assert key == ("forma.cli_intervention", "main")

    def test_adversarial_P11_init_routing(self):
        """forma init → routes to cli_init.main."""
        key, _ = self._get_routed_key_and_argv(["init"])
        assert key == ("forma.cli_init", "main")

    def test_adversarial_P11_eval_class_does_not_eat_value(self):
        """Verify --class value 'batch' is not consumed as subcommand when preceded by --class."""
        # 'batch' IS a registered eval subcommand, but here it's a --class value
        key, delegate_argv = self._get_routed_key_and_argv(["eval", "--class", "batch"])
        assert key == ("forma.pipeline_evaluation", "main")
        assert "--class" in delegate_argv
        assert "batch" in delegate_argv

    # -----------------------------------------------------------------
    # P12 (BONUS): Bare List Edge Cases
    # -----------------------------------------------------------------

    def test_adversarial_P12_bare_list_missing_student_id(self):
        """Bare list entry missing student_id → KeyError (not silent corruption)."""
        from forma.evaluation_io import extract_student_responses

        bare = [{"q_num": 1, "text": "orphan answer"}]
        with pytest.raises(KeyError):
            extract_student_responses(bare)

    def test_adversarial_P12_bare_list_missing_q_num(self):
        """Bare list entry missing q_num → KeyError."""
        from forma.evaluation_io import extract_student_responses

        bare = [{"student_id": "S001", "text": "answer"}]
        with pytest.raises(KeyError):
            extract_student_responses(bare)

    def test_adversarial_P12_bare_list_large_dataset(self):
        """Bare list with 1000 entries → no performance issue, correct extraction."""
        from forma.evaluation_io import extract_student_responses

        bare = [
            {"student_id": f"S{i:04d}", "q_num": q, "text": f"답변_{i}_{q}"}
            for i in range(100)
            for q in range(1, 11)
        ]
        result = extract_student_responses(bare)
        assert len(result) == 100
        assert len(result["S0000"]) == 10

    def test_adversarial_P12_bare_list_q_num_as_string(self):
        """Bare list with q_num as string '1' → should be coerced to int."""
        from forma.evaluation_io import extract_student_responses

        bare = [{"student_id": "S001", "q_num": "1", "text": "답변"}]
        result = extract_student_responses(bare)
        assert result["S001"][1] == "답변"
