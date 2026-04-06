# tests/test_cli_ocr.py
"""Tests for src/cli_ocr.py bhu-ocr CLI."""

from __future__ import annotations

import csv
from unittest.mock import patch

import pytest
import yaml

from forma.cli_ocr import _load_ocr_config, _parse_args, main


# ── fixtures ──────────────────────────────────────


@pytest.fixture
def ocr_config_yaml(tmp_path):
    """Write a minimal valid OCR config YAML."""
    import json as _json

    # Also create the referenced naver config
    naver_cfg = tmp_path / "naver.json"
    naver_cfg.write_text(_json.dumps({"secret_key": "k", "api_url": "https://fake/api"}))
    cfg = {
        "image-dir": str(tmp_path / "scans"),
        "naver-ocr-config": str(naver_cfg),
        "output": str(tmp_path / "results.yaml"),
        "num-questions": 2,
    }
    path = tmp_path / "ocr_config.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)


@pytest.fixture
def ocr_results_yaml(tmp_path):
    data = [
        {
            "student_id": "S001",
            "q_num": 1,
            "text": "sample",
            "source_file": "q1_W1_0001.jpg",
        }
    ]
    path = tmp_path / "ocr_results.yaml"
    path.write_text(yaml.dump(data, allow_unicode=True))
    return str(path)


@pytest.fixture
def forms_csv(tmp_path):
    path = tmp_path / "responses.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["student_id", "이름"])
        w.writeheader()
        w.writerow({"student_id": "S001", "이름": "홍길동"})
    return str(path)


# ──────────────────────────────────────────────────
# Group 1: _parse_args
# ──────────────────────────────────────────────────


class TestParseArgs:
    def test_scan_subcommand_parsed(self, ocr_config_yaml):
        args = _parse_args(["scan", "--config", ocr_config_yaml])
        assert args.command == "scan"
        assert args.config == ocr_config_yaml

    def test_scan_num_questions_default_none(self, ocr_config_yaml):
        args = _parse_args(["scan", "--config", ocr_config_yaml])
        assert args.num_questions is None

    def test_scan_num_questions_override(self, ocr_config_yaml):
        args = _parse_args(["scan", "--config", ocr_config_yaml, "--num-questions", "3"])
        assert args.num_questions == 3

    def test_join_subcommand_parsed_with_csv(self, tmp_path):
        args = _parse_args(
            [
                "join",
                "--ocr-results",
                "results.yaml",
                "--forms-csv",
                "responses.csv",
                "--output",
                "final.yaml",
            ]
        )
        assert args.command == "join"
        assert args.ocr_results == "results.yaml"
        assert args.forms_csv == "responses.csv"
        assert args.output == "final.yaml"
        assert args.spreadsheet_url is None

    def test_join_subcommand_parsed_with_sheets(self):
        url = "https://docs.google.com/spreadsheets/d/abc"
        args = _parse_args(
            [
                "join",
                "--ocr-results",
                "r.yaml",
                "--output",
                "o.yaml",
                "--spreadsheet-url",
                url,
            ]
        )
        assert args.spreadsheet_url == url
        assert args.forms_csv is None

    def test_join_default_student_id_column(self):
        args = _parse_args(
            [
                "join",
                "--ocr-results",
                "r.yaml",
                "--forms-csv",
                "f.csv",
                "--output",
                "o.yaml",
            ]
        )
        assert args.student_id_column == "student_id"

    def test_join_custom_student_id_column(self):
        args = _parse_args(
            [
                "join",
                "--ocr-results",
                "r.yaml",
                "--forms-csv",
                "f.csv",
                "--output",
                "o.yaml",
                "--student-id-column",
                "sid",
            ]
        )
        assert args.student_id_column == "sid"

    def test_join_credentials_default(self):
        args = _parse_args(
            [
                "join",
                "--ocr-results",
                "r.yaml",
                "--output",
                "o.yaml",
                "--forms-csv",
                "f.csv",
            ]
        )
        assert args.credentials == "credentials.json"

    def test_join_credentials_custom(self):
        args = _parse_args(
            [
                "join",
                "--ocr-results",
                "r.yaml",
                "--output",
                "o.yaml",
                "--forms-csv",
                "f.csv",
                "--credentials",
                "my_creds.json",
            ]
        )
        assert args.credentials == "my_creds.json"

    def test_join_manual_mapping_arg(self):
        args = _parse_args(
            [
                "join",
                "--ocr-results",
                "r.yaml",
                "--output",
                "o.yaml",
                "--forms-csv",
                "f.csv",
                "--manual-mapping",
                "mapping.yaml",
            ]
        )
        assert args.manual_mapping == "mapping.yaml"

    def test_missing_subcommand_raises(self):
        with pytest.raises(SystemExit):
            _parse_args([])

    def test_scan_no_args_defaults_to_gemini(self):
        """v0.13.0: scan without --config/--class uses default gemini provider."""
        args = _parse_args(["scan"])
        assert args.command == "scan"
        assert args.provider == "gemini"
        assert args.class_id is None

    def test_join_missing_output_raises(self):
        with pytest.raises(SystemExit):
            _parse_args(["join", "--ocr-results", "r.yaml"])


# ──────────────────────────────────────────────────
# Group 2: _load_ocr_config
# ──────────────────────────────────────────────────


class TestLoadOcrConfig:
    def test_loads_valid_config(self, ocr_config_yaml):
        cfg = _load_ocr_config(ocr_config_yaml)
        assert "image-dir" in cfg
        assert "naver-ocr-config" in cfg
        assert "output" in cfg

    def test_raises_on_missing_keys(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump({"image-dir": "/some/path"}))
        with pytest.raises(ValueError, match="missing required keys"):
            _load_ocr_config(str(path))

    def test_num_questions_optional(self, tmp_path):
        import json as _json

        naver_cfg = tmp_path / "n.json"
        naver_cfg.write_text(_json.dumps({"secret_key": "k", "api_url": "u"}))
        cfg = {
            "image-dir": "/tmp",
            "naver-ocr-config": str(naver_cfg),
            "output": "/tmp/out.yaml",
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg))
        loaded = _load_ocr_config(str(path))
        assert loaded.get("num-questions", 2) == 2


# ──────────────────────────────────────────────────
# Group 3: main() integration (scan)
# ──────────────────────────────────────────────────


class TestMainScan:
    def test_main_scan_calls_pipeline(self, ocr_config_yaml):
        with patch(
            "forma.cli_ocr.run_scan_pipeline",
            return_value=[],
        ) as mock_scan:
            main(["scan", "--config", ocr_config_yaml])

        mock_scan.assert_called_once()
        call_kwargs = mock_scan.call_args.kwargs
        assert "image_dir" in call_kwargs
        assert "naver_ocr_config" in call_kwargs
        assert "output_path" in call_kwargs

    def test_main_scan_num_questions_from_config(self, ocr_config_yaml):
        """num-questions from YAML (2) used when CLI not given."""
        with patch(
            "forma.cli_ocr.run_scan_pipeline",
            return_value=[],
        ) as mock_scan:
            main(["scan", "--config", ocr_config_yaml])
        assert mock_scan.call_args.kwargs["num_questions"] == 2

    def test_main_scan_cli_num_questions_overrides_config(self, ocr_config_yaml):
        with patch(
            "forma.cli_ocr.run_scan_pipeline",
            return_value=[],
        ) as mock_scan:
            main(
                [
                    "scan",
                    "--config",
                    ocr_config_yaml,
                    "--num-questions",
                    "4",
                ]
            )
        assert mock_scan.call_args.kwargs["num_questions"] == 4


# ──────────────────────────────────────────────────
# Group 4: main() integration (join)
# ──────────────────────────────────────────────────


class TestMainJoin:
    def test_main_join_calls_pipeline_with_csv(self, ocr_results_yaml, forms_csv, tmp_path):
        out = str(tmp_path / "final.yaml")
        with patch(
            "forma.cli_ocr.run_join_pipeline",
            return_value=[],
        ) as mock_join:
            main(
                [
                    "join",
                    "--ocr-results",
                    ocr_results_yaml,
                    "--forms-csv",
                    forms_csv,
                    "--output",
                    out,
                ]
            )
        mock_join.assert_called_once()
        call_kwargs = mock_join.call_args.kwargs
        assert call_kwargs["ocr_results_path"] == ocr_results_yaml
        assert call_kwargs["forms_csv_path"] == forms_csv
        assert call_kwargs["output_path"] == out
        assert call_kwargs["spreadsheet_url"] is None
        assert call_kwargs["student_id_column"] == "student_id"

    def test_main_join_with_spreadsheet_url(self, ocr_results_yaml, tmp_path):
        out = str(tmp_path / "final.yaml")
        url = "https://docs.google.com/spreadsheets/d/abc"
        with patch(
            "forma.cli_ocr.run_join_pipeline",
            return_value=[],
        ) as mock_join:
            main(
                [
                    "join",
                    "--ocr-results",
                    ocr_results_yaml,
                    "--output",
                    out,
                    "--spreadsheet-url",
                    url,
                ]
            )
        call_kwargs = mock_join.call_args.kwargs
        assert call_kwargs["spreadsheet_url"] == url
        assert call_kwargs["forms_csv_path"] is None

    def test_main_join_custom_student_id_column(self, ocr_results_yaml, forms_csv, tmp_path):
        out = str(tmp_path / "final.yaml")
        with patch(
            "forma.cli_ocr.run_join_pipeline",
            return_value=[],
        ) as mock_join:
            main(
                [
                    "join",
                    "--ocr-results",
                    ocr_results_yaml,
                    "--forms-csv",
                    forms_csv,
                    "--output",
                    out,
                    "--student-id-column",
                    "sid",
                ]
            )
        assert mock_join.call_args.kwargs["student_id_column"] == "sid"

    def test_main_join_no_source_exits(self, ocr_results_yaml, tmp_path):
        """Exit with error when neither --spreadsheet-url nor --forms-csv."""
        out = str(tmp_path / "final.yaml")
        with pytest.raises(SystemExit):
            main(
                [
                    "join",
                    "--ocr-results",
                    ocr_results_yaml,
                    "--output",
                    out,
                ]
            )

    def test_main_join_passes_manual_mapping(self, ocr_results_yaml, forms_csv, tmp_path):
        out = str(tmp_path / "final.yaml")
        mapping = str(tmp_path / "mapping.yaml")
        with patch(
            "forma.cli_ocr.run_join_pipeline",
            return_value=[],
        ) as mock_join:
            main(
                [
                    "join",
                    "--ocr-results",
                    ocr_results_yaml,
                    "--forms-csv",
                    forms_csv,
                    "--output",
                    out,
                    "--manual-mapping",
                    mapping,
                ]
            )
        assert mock_join.call_args.kwargs["manual_mapping_path"] == mapping


# ──────────────────────────────────────────────────
# Group 5: scan --provider / --model args (LLM Vision)
# ──────────────────────────────────────────────────


class TestScanLLMArgs:
    """Tests for scan subcommand --provider and --model args."""

    def test_parse_scan_with_provider(self):
        """--provider arg is parsed for scan subcommand."""
        args = _parse_args(["scan", "--provider", "gemini"])
        assert args.provider == "gemini"
        assert args.command == "scan"

    def test_parse_scan_with_model(self):
        """--model arg is parsed for scan subcommand."""
        args = _parse_args(["scan", "--provider", "gemini", "--model", "gemini-2.5-flash"])
        assert args.model == "gemini-2.5-flash"

    def test_parse_scan_provider_default(self):
        """--provider defaults to 'gemini'."""
        args = _parse_args(["scan", "--provider", "gemini"])
        assert args.provider == "gemini"

    def test_scan_provider_defaults_gemini(self):
        """v0.13.0: --provider defaults to 'gemini' when not specified."""
        args = _parse_args(["scan", "--class", "A"])
        assert args.provider == "gemini"

    def test_main_scan_llm_passes_provider_to_pipeline(self, tmp_path):
        """main() passes llm_provider to run_scan_pipeline."""
        (tmp_path / "img.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
        with patch(
            "forma.cli_ocr.run_scan_pipeline",
            return_value=[],
        ) as mock_scan:
            main(["--no-config", "scan", "--provider", "gemini"])

        call_kwargs = mock_scan.call_args.kwargs
        assert call_kwargs.get("llm_provider") == "gemini"

    def test_main_scan_llm_passes_model(self, tmp_path):
        """main() passes llm_model to run_scan_pipeline."""
        with patch(
            "forma.cli_ocr.run_scan_pipeline",
            return_value=[],
        ) as mock_scan:
            main(["--no-config", "scan", "--provider", "anthropic", "--model", "claude-sonnet-4-6"])

        call_kwargs = mock_scan.call_args.kwargs
        assert call_kwargs.get("llm_provider") == "anthropic"
        assert call_kwargs.get("llm_model") == "claude-sonnet-4-6"

    def test_parse_scan_context_args(self):
        """--subject, --question, --answer-keywords are parsed."""
        args = _parse_args(
            [
                "scan",
                "--provider",
                "gemini",
                "--subject",
                "생물학",
                "--question",
                "세포막을 설명하시오",
                "--answer-keywords",
                "인지질,이중층",
            ]
        )
        assert args.subject == "생물학"
        assert args.question == "세포막을 설명하시오"
        assert args.answer_keywords == "인지질,이중층"

    def test_parse_scan_class_with_provider(self):
        """--class and --provider can coexist (LLM Vision via week.yaml)."""
        args = _parse_args(
            [
                "scan",
                "--class",
                "A",
                "--provider",
                "anthropic",
            ]
        )
        assert args.class_id == "A"
        assert args.provider == "anthropic"

    def test_main_scan_class_with_provider_passes_llm(self, tmp_path):
        """main() passes llm_provider when both --class and --provider given."""
        from types import SimpleNamespace

        mock_resolved = SimpleNamespace(
            ocr_image_dir_pattern="images_A",
            ocr_ocr_output_pattern="scan_A.yaml",
            ocr_num_questions=2,
            ocr_crop_coords=None,
            ocr_review_threshold=0.75,
        )
        (tmp_path / "images_A").mkdir()
        with (
            patch(
                "forma.cli_ocr.run_scan_pipeline",
                return_value=[],
            ) as mock_scan,
            patch(
                "forma.week_config.find_week_config",
                return_value=tmp_path / "week.yaml",
            ),
            patch(
                "forma.week_config.load_week_config",
            ),
            patch(
                "forma.week_config.resolve_class_patterns",
                return_value=mock_resolved,
            ),
        ):
            main(["--no-config", "scan", "--class", "A", "--provider", "anthropic"])

        call_kwargs = mock_scan.call_args.kwargs
        assert call_kwargs.get("llm_provider") == "anthropic"

    def test_main_scan_llm_passes_context(self):
        """main() passes context dict to run_scan_pipeline."""
        with patch(
            "forma.cli_ocr.run_scan_pipeline",
            return_value=[],
        ) as mock_scan:
            main(
                [
                    "--no-config",
                    "scan",
                    "--provider",
                    "gemini",
                    "--subject",
                    "생물학",
                    "--question",
                    "세포막을 설명하시오",
                    "--answer-keywords",
                    "인지질,이중층",
                ]
            )

        call_kwargs = mock_scan.call_args.kwargs
        ctx = call_kwargs.get("llm_context")
        assert ctx is not None
        assert ctx["subject"] == "생물학"
        assert ctx["question"] == "세포막을 설명하시오"
        assert ctx["answer_keywords"] == "인지질,이중층"
