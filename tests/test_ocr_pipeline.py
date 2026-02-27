# tests/test_ocr_pipeline.py
"""Tests for src/ocr_pipeline.py."""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

# preprocess_imgs calls matplotlib.use("Qt5Agg") at module level which
# requires a display.  Mock the whole module before any import of
# ocr_pipeline so the lazy imports inside its functions also resolve
# against the mock.
if "src.preprocess_imgs" not in sys.modules:
    sys.modules["src.preprocess_imgs"] = MagicMock()

from src.ocr_pipeline import (  # noqa: E402 — after sys.modules patch
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
                "src.preprocess_imgs.crop_and_save_images",
            ) as mock_crop,
            patch(
                "src.ocr_pipeline.prepare_image_files_list",
                return_value=[],
            ),
            patch(
                "src.ocr_pipeline.send_images_receive_ocr",
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
            patch("src.preprocess_imgs.crop_and_save_images"),
            patch(
                "src.ocr_pipeline.prepare_image_files_list",
                return_value=[fake_img],
            ),
            patch(
                "src.ocr_pipeline.decode_qr_from_image",
                return_value=None,
            ),
            patch(
                "src.ocr_pipeline.send_images_receive_ocr",
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
# Group 4: run_join_pipeline
# ──────────────────────────────────────────────────


class TestRunJoinPipeline:
    def test_join_adds_forms_data(
        self, ocr_results_yaml, forms_csv_file, tmp_path
    ):
        out = str(tmp_path / "final.yaml")
        joined = run_join_pipeline(
            ocr_results_path=ocr_results_yaml,
            forms_csv_path=forms_csv_file,
            output_path=out,
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
            forms_csv_path=forms_csv_file,
            output_path=out,
        )
        s999 = next(e for e in joined if e["student_id"] == "S999")
        assert "forms_data" not in s999

    def test_join_output_yaml_written(
        self, ocr_results_yaml, forms_csv_file, tmp_path
    ):
        out = str(tmp_path / "final.yaml")
        run_join_pipeline(
            ocr_results_path=ocr_results_yaml,
            forms_csv_path=forms_csv_file,
            output_path=out,
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
            forms_csv_path=str(csv_path),
            output_path=out,
            student_id_column="sid",
        )
        assert joined[0]["forms_data"]["이름"] == "테스트"
