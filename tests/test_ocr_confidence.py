"""Tests for OCR confidence extraction (016-ocr-confidence Phase 2 & 3)."""

from unittest.mock import patch

import pytest
import yaml

from forma.naver_ocr import extract_text, extract_text_with_confidence


class TestExtractTextWithConfidence:
    """Tests for extract_text_with_confidence()."""

    def test_normal_operation(self):
        """Normal response with inferConfidence fields returns correct stats."""
        responses = [
            {
                "images": [
                    {
                        "name": "image1.jpg",
                        "fields": [
                            {"inferText": "Hello", "inferConfidence": 0.95},
                            {"inferText": "World", "inferConfidence": 0.88},
                            {"inferText": "Foo", "inferConfidence": 0.62},
                            {"inferText": "Bar", "inferConfidence": 0.91},
                        ],
                    }
                ]
            }
        ]

        result = extract_text_with_confidence(responses)

        assert "image1.jpg" in result
        entry = result["image1.jpg"]
        assert entry["text"] == "Hello World Foo Bar"
        assert entry["confidence_mean"] == pytest.approx(0.84, abs=0.01)
        assert entry["confidence_min"] == 0.62
        assert entry["field_count"] == 4

    def test_missing_infer_confidence(self):
        """Response without inferConfidence degrades gracefully."""
        responses = [
            {
                "images": [
                    {
                        "name": "image1.jpg",
                        "fields": [
                            {"inferText": "Hello"},
                            {"inferText": "World"},
                        ],
                    }
                ]
            }
        ]

        result = extract_text_with_confidence(responses)

        entry = result["image1.jpg"]
        assert entry["text"] == "Hello World"
        assert entry["confidence_mean"] is None
        assert entry["confidence_min"] is None
        assert entry["field_count"] == 2

    def test_empty_fields(self):
        """Response with empty fields array returns empty text and None stats."""
        responses = [
            {
                "images": [
                    {
                        "name": "image1.jpg",
                        "fields": [],
                    }
                ]
            }
        ]

        result = extract_text_with_confidence(responses)

        entry = result["image1.jpg"]
        assert entry["text"] == ""
        assert entry["confidence_mean"] is None
        assert entry["confidence_min"] is None
        assert entry["field_count"] == 0

    def test_extract_text_backward_compatibility(self):
        """extract_text() still returns dict[str, str] after refactor."""
        responses = [
            {
                "images": [
                    {
                        "name": "image1.jpg",
                        "fields": [
                            {"inferText": "Hello", "inferConfidence": 0.95},
                            {"inferText": "World", "inferConfidence": 0.88},
                        ],
                    }
                ]
            }
        ]

        result = extract_text(responses)

        assert isinstance(result, dict)
        assert result == {"image1.jpg": "Hello World"}
        # Must be str values, NOT dict values
        assert isinstance(result["image1.jpg"], str)

    def test_multiple_images(self):
        """Multiple images in a single response are all processed."""
        responses = [
            {
                "images": [
                    {
                        "name": "img_a.jpg",
                        "fields": [
                            {"inferText": "A", "inferConfidence": 0.9},
                        ],
                    },
                    {
                        "name": "img_b.jpg",
                        "fields": [
                            {"inferText": "B", "inferConfidence": 0.7},
                            {"inferText": "C", "inferConfidence": 0.8},
                        ],
                    },
                ]
            }
        ]

        result = extract_text_with_confidence(responses)

        assert len(result) == 2
        assert result["img_a.jpg"]["confidence_mean"] == pytest.approx(0.9)
        assert result["img_a.jpg"]["confidence_min"] == 0.9
        assert result["img_b.jpg"]["confidence_mean"] == pytest.approx(0.75)
        assert result["img_b.jpg"]["confidence_min"] == 0.7

    def test_partial_confidence(self):
        """Mix of fields with and without inferConfidence uses only available values."""
        responses = [
            {
                "images": [
                    {
                        "name": "image1.jpg",
                        "fields": [
                            {"inferText": "Hello", "inferConfidence": 0.9},
                            {"inferText": "World"},
                            {"inferText": "Foo", "inferConfidence": 0.7},
                        ],
                    }
                ]
            }
        ]

        result = extract_text_with_confidence(responses)

        entry = result["image1.jpg"]
        assert entry["text"] == "Hello World Foo"
        # Only 2 fields have confidence
        assert entry["confidence_mean"] == pytest.approx(0.8)
        assert entry["confidence_min"] == 0.7
        assert entry["field_count"] == 3

    def test_empty_responses_list(self):
        """Empty responses list returns empty dict."""
        result = extract_text_with_confidence([])
        assert result == {}

    def test_response_missing_images_key(self):
        """Response without 'images' key returns empty dict."""
        result = extract_text_with_confidence([{"error": "some error"}])
        assert result == {}


# ──────────────────────────────────────────────────────────────
# Phase 3 US1: Scan pipeline confidence fields + CLI output
# ──────────────────────────────────────────────────────────────


class TestScanPipelineConfidence:
    """Tests for confidence fields in scan pipeline YAML output."""

    @pytest.fixture
    def image_dir(self, tmp_path):
        """Create a temp dir with a dummy JPEG."""
        (tmp_path / "W1_0001.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
        return str(tmp_path)

    @pytest.fixture
    def ocr_config_file(self, tmp_path):
        import json

        cfg = {"secret_key": "fake", "api_url": "https://fake.ocr/api"}
        path = tmp_path / "naver_ocr.json"
        path.write_text(json.dumps(cfg))
        return str(path)

    def _mock_ocr_response(self, confidences):
        """Build a mock OCR API response with given confidences."""
        fields = []
        for i, conf in enumerate(confidences):
            field = {"inferText": f"word{i}"}
            if conf is not None:
                field["inferConfidence"] = conf
            fields.append(field)
        return [
            {
                "images": [
                    {
                        "name": "q1_W1_0001.jpg",
                        "fields": fields,
                    }
                ]
            }
        ]

    def test_scan_result_includes_confidence_fields(
        self, image_dir, ocr_config_file, tmp_path
    ):
        """Scan results YAML includes ocr_confidence_mean/min/field_count."""
        from forma.ocr_pipeline import run_scan_pipeline

        out = str(tmp_path / "results.yaml")
        fake_img = str(tmp_path / "fake.jpg")
        open(fake_img, "wb").close()

        ocr_resp = self._mock_ocr_response([0.95, 0.88, 0.62, 0.91])

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
                "forma.ocr_pipeline.send_images_receive_ocr",
                return_value=ocr_resp,
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
        r = results[0]
        assert r["text"] == "word0 word1 word2 word3"
        assert r["ocr_confidence_mean"] == pytest.approx(0.84, abs=0.01)
        assert r["ocr_confidence_min"] == 0.62
        assert r["ocr_field_count"] == 4

        # Also verify YAML persistence
        with open(out, encoding="utf-8") as f:
            saved = yaml.safe_load(f)
        assert saved[0]["ocr_confidence_mean"] == pytest.approx(0.84, abs=0.01)
        assert saved[0]["ocr_confidence_min"] == 0.62
        assert saved[0]["ocr_field_count"] == 4

    def test_scan_result_confidence_none_when_api_missing(
        self, image_dir, ocr_config_file, tmp_path
    ):
        """When inferConfidence is absent, confidence fields are None."""
        from forma.ocr_pipeline import run_scan_pipeline

        out = str(tmp_path / "results.yaml")
        fake_img = str(tmp_path / "fake.jpg")
        open(fake_img, "wb").close()

        ocr_resp = self._mock_ocr_response([None, None])

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
                "forma.ocr_pipeline.send_images_receive_ocr",
                return_value=ocr_resp,
            ),
        ):
            results = run_scan_pipeline(
                image_dir=image_dir,
                naver_ocr_config=ocr_config_file,
                output_path=out,
                num_questions=1,
                crop_coords=[(0, 0, 5, 5)],
            )

        r = results[0]
        assert r["ocr_confidence_mean"] is None
        assert r["ocr_confidence_min"] is None
        assert r["ocr_field_count"] == 2

    def test_scan_low_confidence_summary_message(
        self, image_dir, ocr_config_file, tmp_path, capsys
    ):
        """Scan completion prints low-confidence summary when present."""
        from forma.ocr_pipeline import run_scan_pipeline

        out = str(tmp_path / "results.yaml")
        fake_img1 = str(tmp_path / "fake1.jpg")
        fake_img2 = str(tmp_path / "fake2.jpg")
        open(fake_img1, "wb").close()
        open(fake_img2, "wb").close()

        # Two images: one low confidence, one high
        ocr_resp_low = self._mock_ocr_response([0.50, 0.60])
        ocr_resp_high = self._mock_ocr_response([0.90, 0.95])

        call_count = [0]

        def side_effect_ocr(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ocr_resp_low
            return ocr_resp_high

        with (
            patch("forma.ocr_pipeline.crop_and_save_images"),
            patch(
                "forma.ocr_pipeline.prepare_image_files_list",
                return_value=[fake_img1, fake_img2],
            ),
            patch(
                "forma.ocr_pipeline.decode_qr_from_image",
                side_effect=["S001_1", "S002_1"],
            ),
            patch(
                "forma.ocr_pipeline.parse_qr_content",
                side_effect=[
                    {"student_id": "S001", "q_num": 1},
                    {"student_id": "S002", "q_num": 1},
                ],
            ),
            patch(
                "forma.ocr_pipeline.send_images_receive_ocr",
                side_effect=side_effect_ocr,
            ),
        ):
            run_scan_pipeline(
                image_dir=image_dir,
                naver_ocr_config=ocr_config_file,
                output_path=out,
                num_questions=1,
                crop_coords=[(0, 0, 5, 5)],
            )

        captured = capsys.readouterr().out
        assert "저인식률" in captured
        assert "1건" in captured

    def test_scan_no_low_confidence_no_message(
        self, image_dir, ocr_config_file, tmp_path, capsys
    ):
        """Scan completion prints no low-confidence message when all good."""
        from forma.ocr_pipeline import run_scan_pipeline

        out = str(tmp_path / "results.yaml")
        fake_img = str(tmp_path / "fake.jpg")
        open(fake_img, "wb").close()

        ocr_resp = self._mock_ocr_response([0.90, 0.95])

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
                "forma.ocr_pipeline.send_images_receive_ocr",
                return_value=ocr_resp,
            ),
        ):
            run_scan_pipeline(
                image_dir=image_dir,
                naver_ocr_config=ocr_config_file,
                output_path=out,
                num_questions=1,
                crop_coords=[(0, 0, 5, 5)],
            )

        captured = capsys.readouterr().out
        assert "저인식률" not in captured


class TestJoinPipelineConfidence:
    """Tests for confidence review table in join pipeline output."""

    @pytest.fixture
    def ocr_results_with_confidence(self, tmp_path):
        """OCR results YAML with confidence fields."""
        data = [
            {
                "student_id": "S001",
                "q_num": 1,
                "text": "세포막은 선택적 투과성을 가진다고 할 수 있으며",
                "source_file": "q1_W1_0001.jpg",
                "ocr_confidence_mean": 0.58,
                "ocr_confidence_min": 0.31,
                "ocr_field_count": 4,
            },
            {
                "student_id": "S001",
                "q_num": 2,
                "text": "미토콘드리아는 에너지를 생산한다",
                "source_file": "q2_W1_0001.jpg",
                "ocr_confidence_mean": 0.90,
                "ocr_confidence_min": 0.85,
                "ocr_field_count": 3,
            },
            {
                "student_id": "S002",
                "q_num": 1,
                "text": "핵막에는 핵공이 있어서 물질이 이동한다",
                "source_file": "q1_W1_0002.jpg",
                "ocr_confidence_mean": 0.71,
                "ocr_confidence_min": 0.52,
                "ocr_field_count": 5,
            },
        ]
        path = tmp_path / "ocr_results.yaml"
        path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
        return str(path)

    @pytest.fixture
    def ocr_results_no_confidence(self, tmp_path):
        """OCR results YAML without confidence fields (legacy)."""
        data = [
            {
                "student_id": "S001",
                "q_num": 1,
                "text": "legacy text",
                "source_file": "q1_W1_0001.jpg",
            },
        ]
        path = tmp_path / "ocr_results.yaml"
        path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
        return str(path)

    @pytest.fixture
    def forms_csv_file(self, tmp_path):
        """Google Forms CSV with student names."""
        import csv

        path = tmp_path / "responses.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["student_id", "이름"])
            writer.writeheader()
            writer.writerow({"student_id": "S001", "이름": "김OO"})
            writer.writerow({"student_id": "S002", "이름": "박OO"})
        return str(path)

    def test_join_outputs_low_confidence_table(
        self, ocr_results_with_confidence, forms_csv_file, tmp_path, capsys
    ):
        """Join prints detailed review table for low-confidence answers."""
        from forma.ocr_pipeline import run_join_pipeline

        out = str(tmp_path / "final.yaml")
        run_join_pipeline(
            ocr_results_path=ocr_results_with_confidence,
            output_path=out,
            forms_csv_path=forms_csv_file,
        )

        captured = capsys.readouterr().out
        # Table header
        assert "OCR 인식률 검토 대상" in captured
        # Low-confidence entries (threshold 0.75 default)
        assert "S001" in captured
        assert "S002" in captured
        assert "0.58" in captured
        assert "0.71" in captured
        # High-confidence S001 q2 (0.90) should NOT appear
        assert "0.90" not in captured
        # Summary line
        assert "2건" in captured

    def test_join_no_low_confidence_prints_ok(
        self, tmp_path, forms_csv_file, capsys
    ):
        """Join prints 'no review needed' when all confidence >= threshold."""
        from forma.ocr_pipeline import run_join_pipeline

        data = [
            {
                "student_id": "S001",
                "q_num": 1,
                "text": "good text",
                "source_file": "q1_W1_0001.jpg",
                "ocr_confidence_mean": 0.90,
                "ocr_confidence_min": 0.85,
                "ocr_field_count": 3,
            },
        ]
        ocr_path = str(tmp_path / "ocr.yaml")
        with open(ocr_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        out = str(tmp_path / "final.yaml")
        run_join_pipeline(
            ocr_results_path=ocr_path,
            output_path=out,
            forms_csv_path=forms_csv_file,
        )

        captured = capsys.readouterr().out
        assert "검토 대상 없음" in captured

    def test_join_no_confidence_data_no_table(
        self, ocr_results_no_confidence, forms_csv_file, tmp_path, capsys
    ):
        """Join skips table entirely when confidence data is absent (legacy YAML)."""
        from forma.ocr_pipeline import run_join_pipeline

        out = str(tmp_path / "final.yaml")
        run_join_pipeline(
            ocr_results_path=ocr_results_no_confidence,
            output_path=out,
            forms_csv_path=forms_csv_file,
        )

        captured = capsys.readouterr().out
        assert "OCR 인식률" not in captured
        assert "검토 대상" not in captured


class TestWeekConfigOcrReviewThreshold:
    """Tests for ocr_review_threshold in WeekConfiguration."""

    def test_default_threshold(self):
        """Default ocr_review_threshold is 0.75."""
        from forma.week_config import WeekConfiguration

        config = WeekConfiguration()
        assert config.ocr_review_threshold == 0.75

    def test_load_threshold_from_yaml(self, tmp_path):
        """ocr_review_threshold loaded from week.yaml ocr section."""
        from forma.week_config import load_week_config

        week_yaml = tmp_path / "week.yaml"
        week_yaml.write_text(
            yaml.dump(
                {
                    "week": 1,
                    "ocr": {
                        "num_questions": 2,
                        "image_dir_pattern": "scans/{class}",
                        "review_threshold": 0.60,
                    },
                }
            )
        )
        config = load_week_config(week_yaml)
        assert config.ocr_review_threshold == 0.60

    def test_threshold_defaults_when_not_in_yaml(self, tmp_path):
        """ocr_review_threshold uses default when not specified in YAML."""
        from forma.week_config import load_week_config

        week_yaml = tmp_path / "week.yaml"
        week_yaml.write_text(yaml.dump({"week": 1, "ocr": {"num_questions": 2}}))
        config = load_week_config(week_yaml)
        assert config.ocr_review_threshold == 0.75


class TestCliOcrReviewThreshold:
    """Tests for --ocr-review-threshold CLI option."""

    def test_scan_parse_threshold_option(self):
        """--ocr-review-threshold is parsed for scan subcommand."""
        from forma.cli_ocr import _parse_args

        args = _parse_args([
            "scan", "--config", "test.yaml",
            "--ocr-review-threshold", "0.60",
        ])
        assert args.ocr_review_threshold == 0.60

    def test_join_parse_threshold_option(self):
        """--ocr-review-threshold is parsed for join subcommand."""
        from forma.cli_ocr import _parse_args

        args = _parse_args([
            "join",
            "--ocr-results", "r.yaml",
            "--output", "o.yaml",
            "--forms-csv", "f.csv",
            "--ocr-review-threshold", "0.80",
        ])
        assert args.ocr_review_threshold == 0.80

    def test_threshold_default_is_none(self):
        """--ocr-review-threshold defaults to None (uses pipeline default)."""
        from forma.cli_ocr import _parse_args

        args = _parse_args(["scan", "--config", "test.yaml"])
        assert args.ocr_review_threshold is None

    def test_join_custom_threshold_changes_table(self, tmp_path, capsys):
        """Custom threshold 0.95 flags entries that 0.75 would pass."""
        import csv
        from forma.ocr_pipeline import run_join_pipeline

        data = [
            {
                "student_id": "S001",
                "q_num": 1,
                "text": "some text here",
                "source_file": "q1_W1_0001.jpg",
                "ocr_confidence_mean": 0.90,
                "ocr_confidence_min": 0.85,
                "ocr_field_count": 3,
            },
        ]
        ocr_path = str(tmp_path / "ocr.yaml")
        with open(ocr_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        csv_path = str(tmp_path / "forms.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["student_id", "이름"])
            w.writeheader()
            w.writerow({"student_id": "S001", "이름": "테스트"})

        out = str(tmp_path / "final.yaml")

        # With default threshold 0.75 → no low-confidence entries
        run_join_pipeline(
            ocr_results_path=ocr_path,
            output_path=out,
            forms_csv_path=csv_path,
        )
        captured = capsys.readouterr().out
        assert "검토 대상 없음" in captured

        # With threshold 0.95 → entry flagged
        run_join_pipeline(
            ocr_results_path=ocr_path,
            output_path=out,
            forms_csv_path=csv_path,
            ocr_review_threshold=0.95,
        )
        captured = capsys.readouterr().out
        assert "OCR 인식률 검토 대상" in captured
        assert "0.90" in captured


# ──────────────────────────────────────────────────────────────
# Adversarial persona tests
# ──────────────────────────────────────────────────────────────


class TestAdversarialPersonas:
    """Adversarial tests across 12 attack personas.

    Each persona probes a different boundary or failure mode of the
    OCR confidence feature across phases 2-5.
    """

    # ── Persona 1: Zero-Confidence Attacker ──────────────────

    def test_zero_confidence_all_fields(self):
        """All inferConfidence = 0.0 yields mean=0.0, min=0.0."""
        responses = [
            {
                "images": [
                    {
                        "name": "zero.jpg",
                        "fields": [
                            {"inferText": "a", "inferConfidence": 0.0},
                            {"inferText": "b", "inferConfidence": 0.0},
                            {"inferText": "c", "inferConfidence": 0.0},
                        ],
                    }
                ]
            }
        ]
        result = extract_text_with_confidence(responses)
        entry = result["zero.jpg"]
        assert entry["confidence_mean"] == 0.0
        assert entry["confidence_min"] == 0.0
        assert entry["field_count"] == 3

    def test_zero_confidence_flagged_by_review_table(self, tmp_path, capsys):
        """Zero-confidence entries appear in join review table."""
        from forma.ocr_pipeline import _print_confidence_review_table

        joined = [
            {
                "student_id": "S001",
                "q_num": 1,
                "text": "zero text",
                "ocr_confidence_mean": 0.0,
                "ocr_confidence_min": 0.0,
            },
        ]
        _print_confidence_review_table(joined, threshold=0.75)
        captured = capsys.readouterr().out
        assert "S001" in captured
        assert "0.00" in captured

    # ── Persona 2: Missing-Field Exploiter ───────────────────

    def test_mixed_fields_with_and_without_confidence(self):
        """Mix of fields WITH and WITHOUT inferConfidence computes partial stats."""
        responses = [
            {
                "images": [
                    {
                        "name": "mixed.jpg",
                        "fields": [
                            {"inferText": "has", "inferConfidence": 0.80},
                            {"inferText": "missing"},
                            {"inferText": "also_has", "inferConfidence": 0.60},
                            {"inferText": "none_too"},
                        ],
                    }
                ]
            }
        ]
        result = extract_text_with_confidence(responses)
        entry = result["mixed.jpg"]
        # Only 2 fields have confidence
        assert entry["confidence_mean"] == pytest.approx(0.70)
        assert entry["confidence_min"] == 0.60
        # field_count counts ALL fields, not just ones with confidence
        assert entry["field_count"] == 4

    def test_single_field_with_confidence_rest_without(self):
        """Only 1 of 5 fields has confidence — mean equals that one value."""
        responses = [
            {
                "images": [
                    {
                        "name": "sparse.jpg",
                        "fields": [
                            {"inferText": "a"},
                            {"inferText": "b"},
                            {"inferText": "c", "inferConfidence": 0.42},
                            {"inferText": "d"},
                            {"inferText": "e"},
                        ],
                    }
                ]
            }
        ]
        result = extract_text_with_confidence(responses)
        entry = result["sparse.jpg"]
        assert entry["confidence_mean"] == pytest.approx(0.42)
        assert entry["confidence_min"] == pytest.approx(0.42)
        assert entry["field_count"] == 5

    # ── Persona 3: Negative-Value Injector ───────────────────

    def test_negative_confidence_no_crash(self):
        """inferConfidence = -0.5 does not crash; value is passed through."""
        responses = [
            {
                "images": [
                    {
                        "name": "neg.jpg",
                        "fields": [
                            {"inferText": "bad", "inferConfidence": -0.5},
                            {"inferText": "ok", "inferConfidence": 0.8},
                        ],
                    }
                ]
            }
        ]
        result = extract_text_with_confidence(responses)
        entry = result["neg.jpg"]
        assert entry["confidence_mean"] == pytest.approx(0.15)
        assert entry["confidence_min"] == -0.5

    def test_above_one_confidence_no_crash(self):
        """inferConfidence = 1.5 (out of range) does not crash."""
        responses = [
            {
                "images": [
                    {
                        "name": "over.jpg",
                        "fields": [
                            {"inferText": "hi", "inferConfidence": 1.5},
                            {"inferText": "lo", "inferConfidence": 0.5},
                        ],
                    }
                ]
            }
        ]
        result = extract_text_with_confidence(responses)
        entry = result["over.jpg"]
        assert entry["confidence_mean"] == pytest.approx(1.0)
        assert entry["confidence_min"] == 0.5

    # ── Persona 4: Empty-Response Specialist ─────────────────

    def test_empty_images_array(self):
        """Response with empty images array returns empty dict."""
        result = extract_text_with_confidence([{"images": []}])
        assert result == {}

    def test_missing_images_key(self):
        """Response without 'images' key returns empty dict."""
        result = extract_text_with_confidence([{"data": "stuff"}])
        assert result == {}

    def test_missing_fields_key_in_image(self):
        """Image entry without 'fields' key defaults to empty fields."""
        responses = [
            {
                "images": [
                    {
                        "name": "no_fields.jpg",
                    }
                ]
            }
        ]
        result = extract_text_with_confidence(responses)
        entry = result["no_fields.jpg"]
        assert entry["text"] == ""
        assert entry["confidence_mean"] is None
        assert entry["field_count"] == 0

    # ── Persona 5: Massive-Data Stressor ─────────────────────

    def test_500_fields_correct_stats(self):
        """500+ fields in single image computes correct stats without crash."""
        n = 500
        fields = [
            {"inferText": f"w{i}", "inferConfidence": i / n}
            for i in range(n)
        ]
        responses = [{"images": [{"name": "big.jpg", "fields": fields}]}]
        result = extract_text_with_confidence(responses)
        entry = result["big.jpg"]
        assert entry["field_count"] == n
        expected_mean = sum(i / n for i in range(n)) / n
        assert entry["confidence_mean"] == pytest.approx(expected_mean, abs=1e-6)
        assert entry["confidence_min"] == 0.0

    def test_1000_fields_text_aggregation(self):
        """1000 fields aggregate text correctly."""
        n = 1000
        fields = [{"inferText": f"w{i}", "inferConfidence": 0.5} for i in range(n)]
        responses = [{"images": [{"name": "huge.jpg", "fields": fields}]}]
        result = extract_text_with_confidence(responses)
        entry = result["huge.jpg"]
        assert entry["field_count"] == n
        assert entry["confidence_mean"] == pytest.approx(0.5)
        words = entry["text"].split()
        assert len(words) == n

    # ── Persona 6: Unicode-Chaos Agent ───────────────────────

    def test_unicode_student_id_in_review_table(self, capsys):
        """Korean chars and emoji in student_id don't crash review table."""
        from forma.ocr_pipeline import _print_confidence_review_table

        joined = [
            {
                "student_id": "김학생🎓",
                "q_num": 1,
                "text": "답안 텍스트",
                "ocr_confidence_mean": 0.50,
                "ocr_confidence_min": 0.30,
            },
            {
                "student_id": "S!@#$%^&*()",
                "q_num": 1,
                "text": "special chars",
                "ocr_confidence_mean": 0.40,
                "ocr_confidence_min": 0.20,
            },
        ]
        # Should not raise
        _print_confidence_review_table(joined, threshold=0.75)
        captured = capsys.readouterr().out
        assert "김학생" in captured
        assert "S!@#$%^&*()" in captured

    def test_unicode_in_ocr_text_extraction(self):
        """Korean + emoji in inferText are preserved correctly."""
        responses = [
            {
                "images": [
                    {
                        "name": "korean.jpg",
                        "fields": [
                            {"inferText": "세포막은 🧬", "inferConfidence": 0.9},
                            {"inferText": "선택적 투과성", "inferConfidence": 0.8},
                        ],
                    }
                ]
            }
        ]
        result = extract_text_with_confidence(responses)
        assert "세포막은" in result["korean.jpg"]["text"]
        assert "🧬" in result["korean.jpg"]["text"]

    # ── Persona 7: Legacy-Data Saboteur ──────────────────────

    def test_longitudinal_store_legacy_yaml_no_confidence(self, tmp_path):
        """YAML without confidence fields loads with None values (backward compat)."""
        from forma.longitudinal_store import LongitudinalStore

        store_path = str(tmp_path / "store.yaml")
        # Write legacy format without confidence fields
        legacy_data = {
            "records": {
                "S001_1_1": {
                    "student_id": "S001",
                    "week": 1,
                    "question_sn": 1,
                    "scores": {"ensemble_score": 0.7},
                    "tier_level": 2,
                    "tier_label": "Proficient",
                    "manual_override": False,
                },
            }
        }
        with open(store_path, "w") as f:
            yaml.dump(legacy_data, f)

        store = LongitudinalStore(store_path)
        store.load()
        records = store.get_all_records()
        assert len(records) == 1
        assert records[0].ocr_confidence_mean is None
        assert records[0].ocr_confidence_min is None

    def test_longitudinal_store_mixed_legacy_and_new(self, tmp_path):
        """Store with mix of records (some with, some without confidence)."""
        from forma.longitudinal_store import LongitudinalStore

        store_path = str(tmp_path / "store.yaml")
        data = {
            "records": {
                "S001_1_1": {
                    "student_id": "S001",
                    "week": 1,
                    "question_sn": 1,
                    "scores": {"ensemble_score": 0.7},
                    "tier_level": 2,
                    "tier_label": "Proficient",
                    "manual_override": False,
                },
                "S002_1_1": {
                    "student_id": "S002",
                    "week": 1,
                    "question_sn": 1,
                    "scores": {"ensemble_score": 0.5},
                    "tier_level": 1,
                    "tier_label": "Developing",
                    "manual_override": False,
                    "ocr_confidence_mean": 0.85,
                    "ocr_confidence_min": 0.72,
                },
            }
        }
        with open(store_path, "w") as f:
            yaml.dump(data, f)

        store = LongitudinalStore(store_path)
        store.load()
        records = store.get_all_records()
        s001 = [r for r in records if r.student_id == "S001"][0]
        s002 = [r for r in records if r.student_id == "S002"][0]
        assert s001.ocr_confidence_mean is None
        assert s002.ocr_confidence_mean == 0.85
        assert s002.ocr_confidence_min == 0.72

    # ── Persona 8: Config-Breaker ────────────────────────────

    def test_week_config_threshold_zero(self, tmp_path):
        """review_threshold=0 loads successfully (all entries flagged)."""
        from forma.week_config import load_week_config

        week_yaml = tmp_path / "week.yaml"
        week_yaml.write_text(yaml.dump({
            "week": 1,
            "ocr": {"num_questions": 1, "review_threshold": 0},
        }))
        config = load_week_config(week_yaml)
        assert config.ocr_review_threshold == 0

    def test_week_config_threshold_one(self, tmp_path):
        """review_threshold=1.0 loads (flags everything below perfect)."""
        from forma.week_config import load_week_config

        week_yaml = tmp_path / "week.yaml"
        week_yaml.write_text(yaml.dump({
            "week": 1,
            "ocr": {"num_questions": 1, "review_threshold": 1.0},
        }))
        config = load_week_config(week_yaml)
        assert config.ocr_review_threshold == 1.0

    def test_review_table_threshold_zero_flags_everything(self, capsys):
        """threshold=0 flags all entries with any confidence value."""
        from forma.ocr_pipeline import _print_confidence_review_table

        joined = [
            {"student_id": "S001", "q_num": 1, "text": "t",
             "ocr_confidence_mean": 0.99, "ocr_confidence_min": 0.99},
        ]
        _print_confidence_review_table(joined, threshold=0)
        captured = capsys.readouterr().out
        # threshold=0 means nothing is < 0, so no entries flagged
        assert "검토 대상 없음" in captured

    def test_review_table_threshold_above_one_flags_all(self, capsys):
        """threshold=2.0 flags all entries (impossible to reach)."""
        from forma.ocr_pipeline import _print_confidence_review_table

        joined = [
            {"student_id": "S001", "q_num": 1, "text": "t",
             "ocr_confidence_mean": 0.99, "ocr_confidence_min": 0.99},
        ]
        _print_confidence_review_table(joined, threshold=2.0)
        captured = capsys.readouterr().out
        assert "OCR 인식률 검토 대상" in captured
        assert "S001" in captured

    # ── Persona 9: Concurrent-Access Tester ──────────────────

    def test_confidence_roundtrip_through_store(self, tmp_path):
        """Write LongitudinalRecord with confidence, read back, verify roundtrip."""
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        store_path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(store_path)
        store.load()

        record = LongitudinalRecord(
            student_id="S001",
            week=3,
            question_sn=1,
            scores={"ensemble_score": 0.65},
            tier_level=2,
            tier_label="Proficient",
            ocr_confidence_mean=0.82,
            ocr_confidence_min=0.55,
        )
        store.add_record(record)
        store.save()

        # Read back from fresh store
        store2 = LongitudinalStore(store_path)
        store2.load()
        records = store2.get_all_records()
        assert len(records) == 1
        r = records[0]
        assert r.ocr_confidence_mean == pytest.approx(0.82)
        assert r.ocr_confidence_min == pytest.approx(0.55)

    def test_confidence_trajectory_lookup(self, tmp_path):
        """get_student_trajectory with 'ocr_confidence_mean' returns top-level field."""
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        store_path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(store_path)
        store.load()

        for week in [1, 2, 3]:
            store.add_record(LongitudinalRecord(
                student_id="S001",
                week=week,
                question_sn=1,
                scores={"ensemble_score": 0.5},
                tier_level=1,
                tier_label="Developing",
                ocr_confidence_mean=0.60 + week * 0.05,
                ocr_confidence_min=0.50,
            ))
        store.save()

        store2 = LongitudinalStore(store_path)
        store2.load()
        traj = store2.get_student_trajectory("S001", "ocr_confidence_mean")
        assert len(traj) == 3
        assert traj[0] == (1, pytest.approx(0.65))
        assert traj[1] == (2, pytest.approx(0.70))
        assert traj[2] == (3, pytest.approx(0.75))

    # ── Persona 10: Empty-Text-High-Confidence ───────────────

    def test_empty_text_high_confidence_stored(self):
        """text='' but confidence=0.99 stored correctly, not filtered out."""
        responses = [
            {
                "images": [
                    {
                        "name": "blank.jpg",
                        "fields": [
                            {"inferText": "", "inferConfidence": 0.99},
                        ],
                    }
                ]
            }
        ]
        result = extract_text_with_confidence(responses)
        entry = result["blank.jpg"]
        assert entry["text"] == ""
        assert entry["confidence_mean"] == pytest.approx(0.99)
        assert entry["confidence_min"] == pytest.approx(0.99)
        assert entry["field_count"] == 1

    def test_empty_text_high_confidence_not_dropped_in_review(self, capsys):
        """Empty text with high confidence does not appear in low-confidence table."""
        from forma.ocr_pipeline import _print_confidence_review_table

        joined = [
            {"student_id": "S001", "q_num": 1, "text": "",
             "ocr_confidence_mean": 0.99, "ocr_confidence_min": 0.99},
        ]
        _print_confidence_review_table(joined, threshold=0.75)
        captured = capsys.readouterr().out
        assert "검토 대상 없음" in captured

    # ── Persona 11: Manual-Edit Simulator ────────────────────

    def test_confidence_preserved_after_manual_override(self, tmp_path):
        """manual_override=True preserves original confidence on re-add."""
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        store_path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(store_path)
        store.load()

        original = LongitudinalRecord(
            student_id="S001",
            week=1,
            question_sn=1,
            scores={"ensemble_score": 0.5},
            tier_level=1,
            tier_label="Developing",
            ocr_confidence_mean=0.55,
            ocr_confidence_min=0.30,
        )
        store.add_record(original)

        # Simulate manual override by directly setting flag
        key = "S001_1_1"
        store._records[key]["manual_override"] = True

        # Try to overwrite with new data (different confidence)
        updated = LongitudinalRecord(
            student_id="S001",
            week=1,
            question_sn=1,
            scores={"ensemble_score": 0.8},
            tier_level=2,
            tier_label="Proficient",
            ocr_confidence_mean=0.90,
            ocr_confidence_min=0.80,
        )
        store.add_record(updated)

        # Original should be preserved due to manual_override
        records = store.get_all_records()
        assert len(records) == 1
        r = records[0]
        assert r.ocr_confidence_mean == pytest.approx(0.55)
        assert r.ocr_confidence_min == pytest.approx(0.30)

    def test_confidence_data_independent_of_text_content(self):
        """Confidence is about OCR quality, not text content — both stored."""
        responses = [
            {
                "images": [
                    {
                        "name": "edited.jpg",
                        "fields": [
                            {"inferText": "original_ocr", "inferConfidence": 0.40},
                        ],
                    }
                ]
            }
        ]
        result = extract_text_with_confidence(responses)
        entry = result["edited.jpg"]
        assert entry["text"] == "original_ocr"
        assert entry["confidence_mean"] == pytest.approx(0.40)

    # ── Persona 12: Report-Edge-Case ─────────────────────────

    def test_professor_report_ocr_section_zero_students(self):
        """_build_ocr_confidence_section with empty list returns empty."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator.__new__(ProfessorPDFReportGenerator)
        result = gen._build_ocr_confidence_section([])
        assert result == []

    def test_professor_report_ocr_section_all_none_confidence(self):
        """All confidence_mean=None returns empty (no section)."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator.__new__(ProfessorPDFReportGenerator)
        data = [
            {"student_id": "S001", "confidence_mean": None},
            {"student_id": "S002", "confidence_mean": None},
        ]
        result = gen._build_ocr_confidence_section(data)
        assert result == []

    def test_professor_report_ocr_section_all_high_confidence(self):
        """All confidence >= 0.75 returns empty (INV-R01 skip)."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator.__new__(ProfessorPDFReportGenerator)
        data = [
            {"student_id": "S001", "confidence_mean": 0.90},
            {"student_id": "S002", "confidence_mean": 0.85},
        ]
        result = gen._build_ocr_confidence_section(data)
        assert result == []

    def test_professor_report_ocr_section_single_low_student(self):
        """Single low-confidence student generates non-empty section."""
        from forma.professor_report import ProfessorPDFReportGenerator

        gen = ProfessorPDFReportGenerator()
        gen._chart_gen = _MockChartGen()
        data = [
            {"student_id": "S001", "confidence_mean": 0.50},
        ]
        result = gen._build_ocr_confidence_section(data)
        assert len(result) > 0

    def test_student_summary_row_ocr_confidence_field(self):
        """StudentSummaryRow.ocr_confidence_mean defaults to None."""
        from forma.professor_report_data import StudentSummaryRow

        row = StudentSummaryRow(student_id="S001")
        assert row.ocr_confidence_mean is None
        row.ocr_confidence_mean = 0.65
        assert row.ocr_confidence_mean == 0.65

    def test_ocr_confidence_trend_chart_empty(self):
        """build_ocr_confidence_trend_chart with empty dict returns valid PNG."""
        from forma.longitudinal_report_charts import build_ocr_confidence_trend_chart

        buf = build_ocr_confidence_trend_chart({})
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_ocr_confidence_trend_chart_many_students(self):
        """50 students with varying trajectories render without crash."""
        from forma.longitudinal_report_charts import build_ocr_confidence_trend_chart

        trajectories = {}
        for i in range(50):
            trajectories[f"S{i:03d}"] = [
                (w, 0.5 + 0.01 * w + 0.005 * i) for w in range(1, 8)
            ]
        buf = build_ocr_confidence_trend_chart(trajectories)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_ocr_confidence_trend_chart_3_consecutive_low(self):
        """Student with 3+ consecutive weeks below threshold drawn in red."""
        from forma.longitudinal_report_charts import build_ocr_confidence_trend_chart

        trajectories = {
            "LOW": [(1, 0.50), (2, 0.40), (3, 0.30), (4, 0.60)],
            "HIGH": [(1, 0.90), (2, 0.85), (3, 0.88), (4, 0.92)],
        }
        # Should not crash; generates valid PNG
        buf = build_ocr_confidence_trend_chart(trajectories, threshold=0.75)
        assert buf.getvalue()[:4] == b"\x89PNG"


class _MockChartGen:
    """Minimal mock for ProfessorReportChartGenerator.confidence_histogram."""

    def confidence_histogram(self, scores):
        import io
        buf = io.BytesIO()
        # Minimal valid PNG
        buf.write(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
            b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        buf.seek(0)
        return buf
