# tests/test_cli_ocr.py
"""Tests for src/cli_ocr.py bhu-ocr CLI."""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

# preprocess_imgs calls matplotlib.use("Qt5Agg") at module level.
# Mock it before cli_ocr → ocr_pipeline → preprocess_imgs import chain.
if "src.preprocess_imgs" not in sys.modules:
    sys.modules["src.preprocess_imgs"] = MagicMock()

from src.cli_ocr import _load_ocr_config, _parse_args, main  # noqa: E402


# ── fixtures ──────────────────────────────────────


@pytest.fixture
def ocr_config_yaml(tmp_path):
    """Write a minimal valid OCR config YAML."""
    import json as _json

    # Also create the referenced naver config
    naver_cfg = tmp_path / "naver.json"
    naver_cfg.write_text(
        _json.dumps({"secret_key": "k", "api_url": "https://fake/api"})
    )
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
        args = _parse_args(
            ["scan", "--config", ocr_config_yaml, "--num-questions", "3"]
        )
        assert args.num_questions == 3

    def test_join_subcommand_parsed(self, tmp_path):
        args = _parse_args(
            [
                "join",
                "--ocr-results", "results.yaml",
                "--forms-csv", "responses.csv",
                "--output", "final.yaml",
            ]
        )
        assert args.command == "join"
        assert args.ocr_results == "results.yaml"
        assert args.forms_csv == "responses.csv"
        assert args.output == "final.yaml"

    def test_join_default_student_id_column(self):
        args = _parse_args(
            [
                "join",
                "--ocr-results", "r.yaml",
                "--forms-csv", "f.csv",
                "--output", "o.yaml",
            ]
        )
        assert args.student_id_column == "student_id"

    def test_join_custom_student_id_column(self):
        args = _parse_args(
            [
                "join",
                "--ocr-results", "r.yaml",
                "--forms-csv", "f.csv",
                "--output", "o.yaml",
                "--student-id-column", "sid",
            ]
        )
        assert args.student_id_column == "sid"

    def test_missing_subcommand_raises(self):
        with pytest.raises(SystemExit):
            _parse_args([])

    def test_scan_missing_config_raises(self):
        with pytest.raises(SystemExit):
            _parse_args(["scan"])

    def test_join_missing_required_args_raises(self):
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
            "src.cli_ocr.run_scan_pipeline",
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
            "src.cli_ocr.run_scan_pipeline",
            return_value=[],
        ) as mock_scan:
            main(["scan", "--config", ocr_config_yaml])
        assert mock_scan.call_args.kwargs["num_questions"] == 2

    def test_main_scan_cli_num_questions_overrides_config(
        self, ocr_config_yaml
    ):
        with patch(
            "src.cli_ocr.run_scan_pipeline",
            return_value=[],
        ) as mock_scan:
            main(
                [
                    "scan",
                    "--config", ocr_config_yaml,
                    "--num-questions", "4",
                ]
            )
        assert mock_scan.call_args.kwargs["num_questions"] == 4


# ──────────────────────────────────────────────────
# Group 4: main() integration (join)
# ──────────────────────────────────────────────────


class TestMainJoin:
    def test_main_join_calls_pipeline(
        self, ocr_results_yaml, forms_csv, tmp_path
    ):
        out = str(tmp_path / "final.yaml")
        with patch(
            "src.cli_ocr.run_join_pipeline",
            return_value=[],
        ) as mock_join:
            main(
                [
                    "join",
                    "--ocr-results", ocr_results_yaml,
                    "--forms-csv", forms_csv,
                    "--output", out,
                ]
            )
        mock_join.assert_called_once()
        call_kwargs = mock_join.call_args.kwargs
        assert call_kwargs["ocr_results_path"] == ocr_results_yaml
        assert call_kwargs["forms_csv_path"] == forms_csv
        assert call_kwargs["output_path"] == out
        assert call_kwargs["student_id_column"] == "student_id"

    def test_main_join_custom_student_id_column(
        self, ocr_results_yaml, forms_csv, tmp_path
    ):
        out = str(tmp_path / "final.yaml")
        with patch(
            "src.cli_ocr.run_join_pipeline",
            return_value=[],
        ) as mock_join:
            main(
                [
                    "join",
                    "--ocr-results", ocr_results_yaml,
                    "--forms-csv", forms_csv,
                    "--output", out,
                    "--student-id-column", "sid",
                ]
            )
        assert (
            mock_join.call_args.kwargs["student_id_column"] == "sid"
        )
