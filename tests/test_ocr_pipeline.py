# tests/test_ocr_pipeline.py
"""Tests for src/ocr_pipeline.py."""
from __future__ import annotations

import csv
import os
from unittest.mock import patch

import pytest
import yaml

from forma.ocr_pipeline import (
    _list_raw_images,
    _save_yaml,
    run_join_pipeline,
    run_scan_pipeline,
)


# ── fixtures ──────────────────────────────────────


@pytest.fixture
def image_dir(tmp_path):
    """Create a temp dir with two dummy JPEG files."""
    for name in ("W1_0001.jpg", "W1_0002.jpg"):
        (tmp_path / name).write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
    return str(tmp_path)


@pytest.fixture
def ocr_config_file(tmp_path):
    """Write a minimal Naver OCR config JSON."""
    import json

    cfg = {"secret_key": "fake_key", "api_url": "https://fake.ocr/api"}
    path = tmp_path / "naver_ocr.json"
    path.write_text(json.dumps(cfg))
    return str(path)


@pytest.fixture
def ocr_results_yaml(tmp_path):
    """Write a sample OCR results YAML."""
    data = [
        {
            "student_id": "S001",
            "q_num": 1,
            "text": "생체항상성이란...",
            "source_file": "q1_W1_0001.jpg",
        },
        {
            "student_id": "S001",
            "q_num": 2,
            "text": "양성되먹임기전은...",
            "source_file": "q2_W1_0001.jpg",
        },
        {
            "student_id": "S002",
            "q_num": 1,
            "text": "항상성 유지는...",
            "source_file": "q1_W1_0002.jpg",
        },
    ]
    path = tmp_path / "ocr_results.yaml"
    path.write_text(
        yaml.dump(data, allow_unicode=True), encoding="utf-8"
    )
    return str(path)


@pytest.fixture
def forms_csv_file(tmp_path):
    """Write a sample Google Forms CSV."""
    path = tmp_path / "responses.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["student_id", "학번", "이름"],
        )
        writer.writeheader()
        writer.writerow(
            {"student_id": "S001", "학번": "2026194001", "이름": "홍길동"}
        )
        writer.writerow(
            {"student_id": "S002", "학번": "2026194002", "이름": "김영희"}
        )
    return str(path)


# ──────────────────────────────────────────────────
# Group 1: _list_raw_images helper
# ──────────────────────────────────────────────────


class TestListRawImages:
    def test_returns_only_non_cropped_images(self, tmp_path):
        for name in ("W1_0001.jpg", "q1_W1_0001.jpg", "q2_W1_0001.jpg"):
            (tmp_path / name).write_bytes(b"\x00")
        result = _list_raw_images(str(tmp_path))
        basenames = [os.path.basename(p) for p in result]
        assert "W1_0001.jpg" in basenames
        assert "q1_W1_0001.jpg" not in basenames
        assert "q2_W1_0001.jpg" not in basenames

    def test_returns_sorted_list(self, tmp_path):
        for name in ("b.jpg", "a.jpg", "c.jpg"):
            (tmp_path / name).write_bytes(b"\x00")
        result = _list_raw_images(str(tmp_path))
        basenames = [os.path.basename(p) for p in result]
        assert basenames == sorted(basenames)

    def test_skips_non_image_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("ignore")
        (tmp_path / "scan.jpg").write_bytes(b"\x00")
        result = _list_raw_images(str(tmp_path))
        basenames = [os.path.basename(p) for p in result]
        assert "notes.txt" not in basenames
        assert "scan.jpg" in basenames


# ──────────────────────────────────────────────────
# Group 2: _save_yaml helper
# ──────────────────────────────────────────────────


class TestSaveYaml:
    def test_saves_data_correctly(self, tmp_path):
        data = [{"student_id": "S001", "q_num": 1, "text": "hello"}]
        out = str(tmp_path / "out.yaml")
        _save_yaml(data, out)
        with open(out, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == data

    def test_creates_parent_directories(self, tmp_path):
        out = str(tmp_path / "sub" / "dir" / "out.yaml")
        _save_yaml([], out)
        assert os.path.exists(out)

    def test_unicode_preserved(self, tmp_path):
        data = [{"student_id": "S001", "text": "한국어 텍스트"}]
        out = str(tmp_path / "out.yaml")
        _save_yaml(data, out)
        with open(out, encoding="utf-8") as f:
            content = f.read()
        assert "한국어" in content


# ──────────────────────────────────────────────────
# Group 3: run_scan_pipeline (mocked I/O)
# ──────────────────────────────────────────────────


class TestRunScanPipeline:
    def test_raises_when_no_images(self, tmp_path, ocr_config_file):
        with pytest.raises(FileNotFoundError, match="No images"):
            run_scan_pipeline(
                image_dir=str(tmp_path),
                naver_ocr_config=ocr_config_file,
                output_path=str(tmp_path / "out.yaml"),
                num_questions=2,
                crop_coords=[(0, 0, 10, 10), (0, 10, 10, 20)],
            )

    def test_pipeline_with_mocked_ocr_and_qr(
        self, image_dir, ocr_config_file, tmp_path
    ):
        """Full pipeline with mocked crop, QR, OCR — verifies output."""
        out = str(tmp_path / "results.yaml")
        crop_coords = [(0, 0, 5, 5), (0, 5, 5, 10)]

        # crop_and_save_images creates files that prepare_image_files_list
        # will find.  Mock both I/O-heavy parts.
        with (
            patch(
                "forma.ocr_pipeline.crop_and_save_images",
            ) as mock_crop,
            patch(
                "forma.ocr_pipeline.prepare_image_files_list",
                return_value=[],
            ),
            patch(
                "forma.ocr_pipeline.send_images_receive_ocr",
                return_value=[],
            ),
        ):
            results = run_scan_pipeline(
                image_dir=image_dir,
                naver_ocr_config=ocr_config_file,
                output_path=out,
                num_questions=2,
                crop_coords=crop_coords,
            )

        # crop called once per question
        assert mock_crop.call_count == 2
        assert isinstance(results, list)
        # Output YAML written
        assert os.path.exists(out)

    def test_unknown_fallback_on_qr_failure(
        self, image_dir, ocr_config_file, tmp_path
    ):
        """QR decode failure → UNKNOWN student_id."""
        out = str(tmp_path / "results.yaml")
        fake_img = str(tmp_path / "fake.jpg")
        open(fake_img, "wb").close()

        with (
            patch("forma.ocr_pipeline.crop_and_save_images"),
            patch(
                "forma.ocr_pipeline.prepare_image_files_list",
                return_value=[fake_img],
            ),
            patch(
                "forma.ocr_pipeline.decode_qr_from_image",
                return_value=None,
            ),
            patch(
                "forma.ocr_pipeline.send_images_receive_ocr",
                return_value=[],
            ),
        ):
            results = run_scan_pipeline(
                image_dir=image_dir,
                naver_ocr_config=ocr_config_file,
                output_path=out,
                num_questions=1,
                crop_coords=[(0, 0, 5, 5)],
            )

        assert len(results) == 1
        assert results[0]["student_id"].startswith("UNKNOWN_")


# ──────────────────────────────────────────────────
# Group 3b: run_scan_pipeline with LLM Vision
# ──────────────────────────────────────────────────


class TestRunScanPipelineLLM:
    """Tests for run_scan_pipeline with llm_provider (LLM Vision mode)."""

    def test_llm_mode_output_fields(self, image_dir, tmp_path):
        """LLM mode produces all required output fields."""
        out = str(tmp_path / "results.yaml")
        fake_img = str(tmp_path / "fake.jpg")
        open(fake_img, "wb").close()

        from forma.llm_ocr import LLMVisionResponse, TokenUsage, WordConfidence

        mock_llm_result = {
            fake_img: LLMVisionResponse(
                text="세포막은 선택적 투과성을 가진다",
                word_confidences=[
                    WordConfidence(word="세포막은", confidence=0.92, token_count=3),
                    WordConfidence(word="선택적", confidence=0.88, token_count=2),
                    WordConfidence(word="투과성을", confidence=0.85, token_count=2),
                    WordConfidence(word="가진다", confidence=0.95, token_count=1),
                ],
                confidence_mean=0.90,
                confidence_min=0.85,
                usage=TokenUsage(input_tokens=200, output_tokens=15),
                finish_reason="STOP",
                logprobs_raw=[{"token": "세포", "log_probability": -0.08}],
                safety_ratings=None,
            ),
        }

        with (
            patch("forma.ocr_pipeline.crop_and_save_images"),
            patch(
                "forma.ocr_pipeline.prepare_image_files_list",
                return_value=[fake_img],
            ),
            patch(
                "forma.ocr_pipeline.decode_qr_from_image",
                return_value="S001_1",
            ),
            patch(
                "forma.ocr_pipeline.parse_qr_content",
                return_value={"student_id": "S001", "q_num": 1},
            ),
            patch(
                "forma.llm_ocr.extract_text_via_llm",
                return_value=mock_llm_result,
            ),
        ):
            results = run_scan_pipeline(
                image_dir=image_dir,
                output_path=out,
                num_questions=1,
                crop_coords=[(0, 0, 5, 5)],
                llm_provider="gemini",
            )

        assert len(results) == 1
        r = results[0]
        # Existing fields
        assert r["student_id"] == "S001"
        assert r["q_num"] == 1
        assert r["text"] == "세포막은 선택적 투과성을 가진다"
        assert r["ocr_confidence_mean"] == 0.90
        assert r["ocr_confidence_min"] == 0.85
        assert r["ocr_field_count"] == 4
        # New LLM fields
        assert r["recognition_engine"] == "llm"
        assert r["recognition_model"] is not None
        assert isinstance(r["llm_word_confidences"], list)
        assert isinstance(r["llm_usage"], dict)
        assert r["llm_usage"]["input_tokens"] == 200
        assert r["llm_finish_reason"] == "STOP"

    def test_llm_mode_no_naver_config_required(self, image_dir, tmp_path):
        """LLM mode does not require naver_ocr_config."""
        out = str(tmp_path / "results.yaml")

        with (
            patch("forma.ocr_pipeline.crop_and_save_images"),
            patch(
                "forma.ocr_pipeline.prepare_image_files_list",
                return_value=[],
            ),
        ):
            # Should NOT raise — naver_ocr_config not needed in LLM mode
            results = run_scan_pipeline(
                image_dir=image_dir,
                output_path=out,
                num_questions=1,
                crop_coords=[(0, 0, 5, 5)],
                llm_provider="gemini",
            )

        assert isinstance(results, list)


# ──────────────────────────────────────────────────
# Group 3c: review_needed.yaml generation
# ──────────────────────────────────────────────────


class TestReviewNeeded:
    """Tests for review_needed.yaml generation in LLM scan pipeline."""

    def test_low_confidence_creates_review_needed(self, image_dir, tmp_path):
        """Images below review_threshold produce review_needed.yaml."""
        out = str(tmp_path / "results.yaml")
        review_path = str(tmp_path / "review_needed.yaml")
        fake_img = str(tmp_path / "fake.jpg")
        open(fake_img, "wb").close()

        from forma.llm_ocr import LLMVisionResponse, TokenUsage, WordConfidence

        mock_llm_result = {
            fake_img: LLMVisionResponse(
                text="blurry",
                word_confidences=[WordConfidence("blurry", 0.5, 1)],
                confidence_mean=0.5,
                confidence_min=0.5,
                usage=TokenUsage(input_tokens=50, output_tokens=5),
                finish_reason="STOP",
                logprobs_raw=None,
                safety_ratings=None,
            ),
        }

        with (
            patch("forma.ocr_pipeline.crop_and_save_images"),
            patch("forma.ocr_pipeline.prepare_image_files_list", return_value=[fake_img]),
            patch("forma.ocr_pipeline.decode_qr_from_image", return_value="S001_1"),
            patch("forma.ocr_pipeline.parse_qr_content", return_value={"student_id": "S001", "q_num": 1}),
            patch("forma.llm_ocr.extract_text_via_llm", return_value=mock_llm_result),
        ):
            run_scan_pipeline(
                image_dir=image_dir,
                output_path=out,
                num_questions=1,
                crop_coords=[(0, 0, 5, 5)],
                llm_provider="gemini",
                ocr_review_threshold=0.75,
            )

        # review_needed.yaml should exist
        import os
        review_path = os.path.join(os.path.dirname(out), "review_needed.yaml")
        assert os.path.exists(review_path)
        with open(review_path, encoding="utf-8") as f:
            review_data = yaml.safe_load(f)
        assert isinstance(review_data, list)
        assert len(review_data) == 1
        assert review_data[0]["student_id"] == "S001"
        assert "confidence" in review_data[0]["reason"].lower() or "0.75" in review_data[0]["reason"]

    def test_high_confidence_no_review_needed(self, image_dir, tmp_path):
        """All images above threshold → no review_needed.yaml."""
        out = str(tmp_path / "results.yaml")
        fake_img = str(tmp_path / "fake.jpg")
        open(fake_img, "wb").close()

        from forma.llm_ocr import LLMVisionResponse, TokenUsage, WordConfidence

        mock_llm_result = {
            fake_img: LLMVisionResponse(
                text="clear text",
                word_confidences=[WordConfidence("clear", 0.95, 1)],
                confidence_mean=0.95,
                confidence_min=0.95,
                usage=TokenUsage(input_tokens=50, output_tokens=5),
                finish_reason="STOP",
                logprobs_raw=None,
                safety_ratings=None,
            ),
        }

        with (
            patch("forma.ocr_pipeline.crop_and_save_images"),
            patch("forma.ocr_pipeline.prepare_image_files_list", return_value=[fake_img]),
            patch("forma.ocr_pipeline.decode_qr_from_image", return_value="S001_1"),
            patch("forma.ocr_pipeline.parse_qr_content", return_value={"student_id": "S001", "q_num": 1}),
            patch("forma.llm_ocr.extract_text_via_llm", return_value=mock_llm_result),
        ):
            run_scan_pipeline(
                image_dir=image_dir,
                output_path=out,
                num_questions=1,
                crop_coords=[(0, 0, 5, 5)],
                llm_provider="gemini",
                ocr_review_threshold=0.75,
            )

        import os
        review_path = os.path.join(os.path.dirname(out), "review_needed.yaml")
        assert not os.path.exists(review_path)


# ──────────────────────────────────────────────────
# Group 4: run_join_pipeline
# ──────────────────────────────────────────────────


class TestRunJoinPipeline:
    def test_join_adds_forms_data(
        self, ocr_results_yaml, forms_csv_file, tmp_path
    ):
        out = str(tmp_path / "final.yaml")
        joined = run_join_pipeline(
            ocr_results_path=ocr_results_yaml,
            output_path=out,
            forms_csv_path=forms_csv_file,
        )
        s001_entries = [e for e in joined if e["student_id"] == "S001"]
        assert len(s001_entries) == 2
        for entry in s001_entries:
            assert "forms_data" in entry
            assert entry["forms_data"]["이름"] == "홍길동"

    def test_join_missing_student_keeps_entry_without_forms_data(
        self, ocr_results_yaml, forms_csv_file, tmp_path
    ):
        # Add entry with unknown student_id
        with open(ocr_results_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data.append(
            {
                "student_id": "S999",
                "q_num": 1,
                "text": "no match",
                "source_file": "q1_W1_0099.jpg",
            }
        )
        with open(ocr_results_yaml, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        out = str(tmp_path / "final.yaml")
        joined = run_join_pipeline(
            ocr_results_path=ocr_results_yaml,
            output_path=out,
            forms_csv_path=forms_csv_file,
        )
        s999 = next(e for e in joined if e["student_id"] == "S999")
        assert "forms_data" not in s999

    def test_join_output_yaml_written(
        self, ocr_results_yaml, forms_csv_file, tmp_path
    ):
        out = str(tmp_path / "final.yaml")
        run_join_pipeline(
            ocr_results_path=ocr_results_yaml,
            output_path=out,
            forms_csv_path=forms_csv_file,
        )
        assert os.path.exists(out)
        with open(out, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert isinstance(loaded, list)
        assert len(loaded) > 0

    def test_join_custom_student_id_column(self, tmp_path):
        """Custom CSV column name for student IDs."""
        ocr_yaml = tmp_path / "ocr.yaml"
        yaml.dump(
            [{"student_id": "S001", "q_num": 1, "text": "t"}],
            ocr_yaml.open("w"),
        )
        csv_path = tmp_path / "forms.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sid", "이름"])
            w.writeheader()
            w.writerow({"sid": "S001", "이름": "테스트"})

        out = str(tmp_path / "out.yaml")
        joined = run_join_pipeline(
            ocr_results_path=str(ocr_yaml),
            output_path=out,
            forms_csv_path=str(csv_path),
            student_id_column="sid",
        )
        assert joined[0]["forms_data"]["이름"] == "테스트"

    def test_join_raises_without_any_source(self, ocr_results_yaml, tmp_path):
        """ValueError when neither spreadsheet_url nor forms_csv given."""
        out = str(tmp_path / "final.yaml")
        with pytest.raises(ValueError, match="At least one data source"):
            run_join_pipeline(
                ocr_results_path=ocr_results_yaml,
                output_path=out,
            )

    def test_join_sheets_success_skips_csv(
        self, ocr_results_yaml, forms_csv_file, tmp_path
    ):
        """When Sheets fetch succeeds, CSV is not used."""
        out = str(tmp_path / "final.yaml")
        mock_records = [
            {"student_id": "S001", "학번": "2026194001", "이름": "시트홍"},
            {"student_id": "S002", "학번": "2026194002", "이름": "시트김"},
        ]
        with patch(
            "forma.google_sheets.fetch_sheet_as_records",
            return_value=mock_records,
        ):
            joined = run_join_pipeline(
                ocr_results_path=ocr_results_yaml,
                output_path=out,
                forms_csv_path=forms_csv_file,
                spreadsheet_url="https://docs.google.com/spreadsheets/d/abc",
            )
        s001 = [e for e in joined if e["student_id"] == "S001"]
        assert s001[0]["forms_data"]["이름"] == "시트홍"

    def test_join_sheets_failure_falls_back_to_csv(
        self, ocr_results_yaml, forms_csv_file, tmp_path
    ):
        """When Sheets fetch fails and CSV is available, use CSV."""
        out = str(tmp_path / "final.yaml")
        with patch(
            "forma.google_sheets.fetch_sheet_as_records",
            side_effect=RuntimeError("Network error"),
        ):
            joined = run_join_pipeline(
                ocr_results_path=ocr_results_yaml,
                output_path=out,
                forms_csv_path=forms_csv_file,
                spreadsheet_url="https://docs.google.com/spreadsheets/d/abc",
            )
        s001 = [e for e in joined if e["student_id"] == "S001"]
        assert s001[0]["forms_data"]["이름"] == "홍길동"

    def test_join_manual_mapping_supplements(
        self, ocr_results_yaml, forms_csv_file, tmp_path
    ):
        """Manual mapping adds missing students without overwriting."""
        # Add S003 to OCR results
        with open(ocr_results_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data.append(
            {
                "student_id": "S003",
                "q_num": 1,
                "text": "missing student",
                "source_file": "q1_W1_0003.jpg",
            }
        )
        with open(ocr_results_yaml, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        # Manual mapping for S003 (and S001 which should NOT overwrite)
        mapping_path = tmp_path / "mapping.yaml"
        mapping_path.write_text(
            yaml.dump(
                {
                    "S003": {"이름": "수동추가", "학번": "2026194003"},
                    "S001": {"이름": "덮어쓰면안됨"},
                },
                allow_unicode=True,
            )
        )

        out = str(tmp_path / "final.yaml")
        joined = run_join_pipeline(
            ocr_results_path=ocr_results_yaml,
            output_path=out,
            forms_csv_path=forms_csv_file,
            manual_mapping_path=str(mapping_path),
        )
        s003 = next(e for e in joined if e["student_id"] == "S003")
        assert s003["forms_data"]["이름"] == "수동추가"
        # S001 should still have CSV data, not manual mapping
        s001 = next(e for e in joined if e["student_id"] == "S001")
        assert s001["forms_data"]["이름"] == "홍길동"

    def test_join_match_report_output(
        self, ocr_results_yaml, forms_csv_file, tmp_path, capsys
    ):
        """Match report prints matched/unmatched counts."""
        # Add unmatched student
        with open(ocr_results_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data.append(
            {
                "student_id": "S099",
                "q_num": 1,
                "text": "unmatched",
                "source_file": "q1_W1_0099.jpg",
            }
        )
        with open(ocr_results_yaml, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        out = str(tmp_path / "final.yaml")
        run_join_pipeline(
            ocr_results_path=ocr_results_yaml,
            output_path=out,
            forms_csv_path=forms_csv_file,
        )
        captured = capsys.readouterr().out
        assert "students matched" in captured
        assert "unmatched" in captured
        assert "S099" in captured
