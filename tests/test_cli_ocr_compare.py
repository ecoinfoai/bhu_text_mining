"""Tests for forma-ocr compare CLI subcommand."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import yaml

from forma.cli_ocr import _parse_args, main_compare


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------


class TestCompareParseArgs:
    """Tests for compare subparser in _parse_args()."""

    def test_minimal_args(self):
        """Minimal args: just --image."""
        args = _parse_args(["compare", "--image", "test.jpg"])
        assert args.command == "compare"
        assert args.image == "test.jpg"
        assert args.provider == "gemini"

    def test_all_args(self):
        """All optional args are parsed."""
        args = _parse_args([
            "compare",
            "--image", "test.jpg",
            "--provider", "anthropic",
            "--model", "claude-sonnet-4-6",
            "--subject", "생물학",
            "--question", "세포 분열을 설명하시오",
            "--answer-keywords", "세포막,핵분열",
            "--output", "result.yaml",
        ])
        assert args.provider == "anthropic"
        assert args.model == "claude-sonnet-4-6"
        assert args.subject == "생물학"
        assert args.question == "세포 분열을 설명하시오"
        assert args.answer_keywords == "세포막,핵분열"
        assert args.output == "result.yaml"

    def test_image_required(self):
        """--image is required."""
        with pytest.raises(SystemExit):
            _parse_args(["compare"])

    def test_default_provider(self):
        """Default provider is gemini."""
        args = _parse_args(["compare", "--image", "x.jpg"])
        assert args.provider == "gemini"

    def test_model_default_none(self):
        """Model defaults to None (use provider default)."""
        args = _parse_args(["compare", "--image", "x.jpg"])
        assert args.model is None


# ---------------------------------------------------------------------------
# main_compare tests
# ---------------------------------------------------------------------------


class TestMainCompare:
    """Tests for main_compare() function."""

    def test_basic_comparison_flow(self, tmp_path):
        """End-to-end compare flow with all external calls mocked."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0fake-jpg")

        # Mock Naver OCR response
        ocr_response = [{
            "images": [{
                "name": "test.jpg",
                "inferResult": "SUCCESS",
                "fields": [
                    {"inferText": "세포", "inferConfidence": 0.98},
                    {"inferText": "분열", "inferConfidence": 0.95},
                ],
            }],
        }]

        mock_provider = MagicMock()
        mock_provider.generate_with_image.return_value = "세포 분열"

        with (
            patch("forma.naver_ocr.send_images_receive_ocr", return_value=ocr_response),
            patch("forma.naver_ocr.load_naver_ocr_env", return_value=("key", "https://api")),
            patch("forma.llm_provider.create_provider", return_value=mock_provider),
        ):
            main_compare(
                image=str(img_file),
                provider="gemini",
                model=None,
                subject=None,
                question=None,
                answer_keywords=None,
                output=None,
                no_config=False,
            )

    def test_output_yaml_saved(self, tmp_path):
        """Results are saved to YAML when --output is specified."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0fake-jpg")
        out_file = tmp_path / "result.yaml"

        ocr_response = [{
            "images": [{
                "name": "test.jpg",
                "inferResult": "SUCCESS",
                "fields": [
                    {"inferText": "세포", "inferConfidence": 0.98},
                ],
            }],
        }]

        mock_provider = MagicMock()
        mock_provider.generate_with_image.return_value = "세포"

        with (
            patch("forma.naver_ocr.send_images_receive_ocr", return_value=ocr_response),
            patch("forma.naver_ocr.load_naver_ocr_env", return_value=("key", "https://api")),
            patch("forma.llm_provider.create_provider", return_value=mock_provider),
        ):
            main_compare(
                image=str(img_file),
                provider="gemini",
                model=None,
                subject=None,
                question=None,
                answer_keywords=None,
                output=str(out_file),
                no_config=False,
            )

        assert out_file.exists()
        saved = yaml.safe_load(out_file.read_text(encoding="utf-8"))
        assert "ocr_text" in saved
        assert "llm_text" in saved
        assert "summary" in saved

    def test_with_context_args(self, tmp_path):
        """Subject/question/keywords are passed as context."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0fake-jpg")

        ocr_response = [{
            "images": [{
                "name": "test.jpg",
                "inferResult": "SUCCESS",
                "fields": [
                    {"inferText": "세포", "inferConfidence": 0.98},
                ],
            }],
        }]

        mock_provider = MagicMock()
        mock_provider.generate_with_image.return_value = "세포"

        with (
            patch("forma.naver_ocr.send_images_receive_ocr", return_value=ocr_response),
            patch("forma.naver_ocr.load_naver_ocr_env", return_value=("key", "https://api")),
            patch("forma.llm_provider.create_provider", return_value=mock_provider),
        ):
            main_compare(
                image=str(img_file),
                provider="gemini",
                model=None,
                subject="생물학",
                question="세포 분열을 설명하시오",
                answer_keywords="세포막,핵분열",
                output=None,
                no_config=False,
            )

        # Verify generate_with_image was called with prompt containing context
        call_args = mock_provider.generate_with_image.call_args
        prompt = call_args.kwargs.get("prompt") or call_args.args[0]
        assert "생물학" in prompt

    def test_image_not_found(self, tmp_path):
        """Non-existent image prints error and exits."""
        with pytest.raises(SystemExit):
            main_compare(
                image="/nonexistent/path.jpg",
                provider="gemini",
                model=None,
                subject=None,
                question=None,
                answer_keywords=None,
                output=None,
                no_config=False,
            )
