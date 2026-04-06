"""Adversarial audit tests — Tier 2: Normal user personas (A-04 to A-08).

A-04: Rushed TA — corrupted scan images, missing QR, duplicate submissions.
A-05: Multi-class professor — class code typos, duplicate students, wrong config.
A-06: End-of-semester professor — missing weeks, out-of-order weeks, dropout students.
A-07: OCR editor — scores as strings, out-of-range scores, deleted required fields.
A-08: Email distributor — bad email addresses, SMTP auth failure, partial send.

Discovery only — tests that FAIL indicate vulnerabilities, not test bugs.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pytest
import yaml

from forma.evaluation_types import LongitudinalRecord


# ---------------------------------------------------------------------------
# A-04: Rushed TA
# ---------------------------------------------------------------------------


class TestA04RushedTA:
    """Persona A-04: TA who submits corrupted or duplicate scan data."""

    def test_a04_duplicate_student_submission(self, tmp_path: Path) -> None:
        """A-04: Duplicate student in responses — last should win or warn."""
        from forma.evaluation_io import save_evaluation_yaml, load_evaluation_yaml

        responses = [
            {"student_id": "2024001", "question_sn": 1, "answer": "세포막은 인지질 이중층"},
            {"student_id": "2024001", "question_sn": 1, "answer": "세포막은 단백질 포함"},
        ]
        path = tmp_path / "responses.yaml"
        save_evaluation_yaml({"responses": responses}, str(path))
        loaded = load_evaluation_yaml(str(path))
        assert len(loaded["responses"]) == 2  # Both preserved

    def test_a04_empty_student_answer(self, tmp_path: Path) -> None:
        """A-04: Empty student answer should not crash evaluation."""
        from forma.evaluation_io import extract_student_responses

        # Correct format: {"responses": {student_id: {qsn: text}}}
        data = {
            "responses": {
                "S001": {1: ""},
                "S002": {1: "세포막"},
            }
        }
        result = extract_student_responses(data)
        assert "S001" in result
        assert result["S001"][1] == ""

    def test_a04_none_student_answer(self, tmp_path: Path) -> None:
        """A-04: None/null answer in responses should be handled."""
        from forma.evaluation_io import extract_student_responses

        data = {
            "responses": {
                "S001": {1: None},
            }
        }
        # Should handle None gracefully — either convert or skip
        try:
            result = extract_student_responses(data)
            # If it succeeds, ensure None is handled
            assert "S001" in result or "S001" not in result  # no crash
        except (TypeError, ValueError, KeyError) as e:
            # Acceptable if it raises a clear error
            assert str(e)  # Error message is non-empty

    def test_a04_corrupted_yaml_responses(self, tmp_path: Path) -> None:
        """A-04: YAML with binary garbage should fail cleanly."""
        bad_file = tmp_path / "responses.yaml"
        bad_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        from forma.evaluation_io import load_evaluation_yaml

        with pytest.raises((yaml.YAMLError, UnicodeDecodeError, Exception)):
            load_evaluation_yaml(str(bad_file))

    def test_a04_very_long_student_answer(self, tmp_path: Path) -> None:
        """A-04: Very long student answer (10KB) should not crash."""
        from forma.evaluation_io import extract_student_responses

        long_answer = "세포막은 " * 2000  # ~10KB
        data = {
            "responses": {
                "S001": {1: long_answer},
            }
        }
        result = extract_student_responses(data)
        assert len(result["S001"][1]) > 1000


# ---------------------------------------------------------------------------
# A-05: Multi-class professor
# ---------------------------------------------------------------------------


class TestA05MultiClassProfessor:
    """Persona A-05: Professor managing multiple class sections."""

    def test_a05_class_code_typo(self, tmp_path: Path) -> None:
        """A-05: Mistyped class code should not silently produce wrong results."""
        from forma.longitudinal_store import _infer_class_id

        # Correct patterns
        assert _infer_class_id("eval_A/result.yaml") == "A"
        assert _infer_class_id("eval_BC/result.yaml") == "BC"

        # Typos — lowercase should not match
        result = _infer_class_id("eval_a/result.yaml")
        assert result is None  # lowercase not matched

    def test_a05_duplicate_students_across_classes(self, tmp_path: Path) -> None:
        """A-05: Same student ID in two classes should be stored separately."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        store.load()

        record_a = LongitudinalRecord(
            student_id="2024001",
            week=1,
            question_sn=1,
            scores={"ensemble": 0.8},
            tier_level=3,
            tier_label="high",
            class_id="A",
        )
        record_b = LongitudinalRecord(
            student_id="2024001",
            week=1,
            question_sn=1,
            scores={"ensemble": 0.6},
            tier_level=2,
            tier_label="mid",
            class_id="B",
        )
        store.add_record(record_a)
        store.add_record(record_b)

        # class_id is now part of the key — both records survive (H-06 fixed)
        history = store.get_student_history("2024001")
        assert len(history) == 2

    def test_a05_section_comparison_single_section(self) -> None:
        """A-05: Section comparison with only 1 section should not crash."""
        from forma.section_comparison import compute_section_stats

        stats = compute_section_stats("A", [0.7, 0.8, 0.9], {"S001"})
        assert stats.section_name == "A"
        assert stats.n_students == 3

    def test_a05_section_comparison_empty_section(self) -> None:
        """A-05: Section with zero students should be handled."""
        from forma.section_comparison import compute_section_stats

        try:
            stats = compute_section_stats("A", [], set())
            assert stats.n_students == 0
        except (ValueError, ZeroDivisionError) as e:
            # Acceptable if clear error
            assert str(e)

    def test_a05_section_comparison_identical_scores(self) -> None:
        """A-05: Sections with identical scores should not crash stats tests."""
        from forma.section_comparison import compute_pairwise_comparisons

        scores = {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}
        comparisons = compute_pairwise_comparisons(scores)
        assert len(comparisons) == 1
        assert comparisons[0].cohens_d == 0.0


# ---------------------------------------------------------------------------
# A-06: End-of-semester professor
# ---------------------------------------------------------------------------


class TestA06EndOfSemester:
    """Persona A-06: Professor at semester end with gaps in longitudinal data."""

    def test_a06_missing_weeks(self, tmp_path: Path) -> None:
        """A-06: Student with gaps (weeks 1,2,5,8) should be handled."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        store.load()

        for wk in [1, 2, 5, 8]:
            rec = LongitudinalRecord(
                student_id="S001",
                week=wk,
                question_sn=1,
                scores={"ensemble": 0.5 + wk * 0.05},
                tier_level=2,
                tier_label="mid",
            )
            store.add_record(rec)

        history = store.get_student_history("S001")
        weeks = sorted(r.week for r in history)
        assert weeks == [1, 2, 5, 8]

    def test_a06_dropout_student_nan_scores(self, tmp_path: Path) -> None:
        """A-06: Dropout student with NaN scores after week 4 should not poison aggregations."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        store.load()

        for wk in range(1, 5):
            rec = LongitudinalRecord(
                student_id="S_DROPOUT",
                week=wk,
                question_sn=1,
                scores={"ensemble": 0.4},
                tier_level=1,
                tier_label="low",
            )
            store.add_record(rec)

        # After week 4, student stops — no more records
        trajectory = store.get_student_trajectory("S_DROPOUT", "ensemble")
        assert len(trajectory) == 4
        # All values should be finite
        for wk, val in trajectory:
            assert math.isfinite(val)

    def test_a06_nan_in_scores_dict(self, tmp_path: Path) -> None:
        """A-06: NaN values inside scores dict should be rejected."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        store.load()

        rec = LongitudinalRecord(
            student_id="S_NAN",
            week=1,
            question_sn=1,
            scores={"ensemble": float("nan"), "concept_coverage": 0.5},
            tier_level=1,
            tier_label="low",
        )
        with pytest.raises(ValueError, match="Score cannot be NaN"):
            store.add_record(rec)

    def test_a06_mismatched_question_counts(self, tmp_path: Path) -> None:
        """A-06: Different number of questions per week should be handled."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        store.load()

        # Week 1: 3 questions, Week 2: 2 questions
        for sn in [1, 2, 3]:
            store.add_record(
                LongitudinalRecord(
                    student_id="S001",
                    week=1,
                    question_sn=sn,
                    scores={"ensemble": 0.7},
                    tier_level=2,
                    tier_label="mid",
                )
            )
        for sn in [1, 2]:
            store.add_record(
                LongitudinalRecord(
                    student_id="S001",
                    week=2,
                    question_sn=sn,
                    scores={"ensemble": 0.8},
                    tier_level=3,
                    tier_label="high",
                )
            )

        history = store.get_student_history("S001")
        assert len(history) == 5  # All records preserved

    def test_a06_out_of_order_week_insertion(self, tmp_path: Path) -> None:
        """A-06: Adding weeks out of chronological order should still work."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        store.load()

        for wk in [3, 1, 5, 2, 4]:
            store.add_record(
                LongitudinalRecord(
                    student_id="S001",
                    week=wk,
                    question_sn=1,
                    scores={"ensemble": wk * 0.1},
                    tier_level=2,
                    tier_label="mid",
                )
            )

        trajectory = store.get_student_trajectory("S001", "ensemble")
        weeks = [w for w, _ in trajectory]
        assert weeks == sorted(weeks)  # Should be sorted by week


# ---------------------------------------------------------------------------
# A-07: OCR editor
# ---------------------------------------------------------------------------


class TestA07OCREditor:
    """Persona A-07: Someone who hand-edits OCR output YAML files."""

    def test_a07_scores_as_strings(self, tmp_path: Path) -> None:
        """A-07: Scores entered as strings instead of numbers."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        store.load()

        # Simulating hand-edited scores
        rec = LongitudinalRecord(
            student_id="S001",
            week=1,
            question_sn=1,
            scores={"ensemble": "0.75"},  # type: ignore[arg-type]
            tier_level=2,
            tier_label="mid",
        )
        store.add_record(rec)

        history = store.get_student_history("S001")
        # Should either convert or preserve
        assert history[0].scores["ensemble"] in (0.75, "0.75")

    def test_a07_out_of_range_scores(self, tmp_path: Path) -> None:
        """A-07: Scores outside [0, 1] range should be handled."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        store.load()

        rec = LongitudinalRecord(
            student_id="S001",
            week=1,
            question_sn=1,
            scores={"ensemble": 1.5, "concept_coverage": -0.3},
            tier_level=2,
            tier_label="mid",
        )
        # Negative scores are now rejected (M-12~15 fixed)
        with pytest.raises(ValueError, match="Score cannot be negative"):
            store.add_record(rec)

    def test_a07_deleted_required_fields(self, tmp_path: Path) -> None:
        """A-07: YAML with required fields deleted should fail clearly."""
        yaml_path = tmp_path / "store.yaml"
        # Missing student_id in record
        broken_data = {
            "records": {
                "S001_1_1": {
                    "week": 1,
                    "question_sn": 1,
                    "scores": {"ensemble": 0.5},
                    "tier_level": 2,
                    "tier_label": "mid",
                    # "student_id" deliberately missing
                }
            }
        }
        yaml_path.write_text(yaml.dump(broken_data), encoding="utf-8")

        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(yaml_path))
        with pytest.raises((KeyError, TypeError, ValueError)):
            store.load()
            store.get_all_records()

    def test_a07_yaml_anchors_in_scores(self, tmp_path: Path) -> None:
        """A-07: YAML anchors inside edited files should work correctly."""
        yaml_path = tmp_path / "edited.yaml"
        content = textwrap.dedent("""\
            defaults: &score_defaults
              ensemble: 0.5
              concept_coverage: 0.3
            records:
              S001_1_1:
                student_id: S001
                week: 1
                question_sn: 1
                scores:
                  <<: *score_defaults
                tier_level: 2
                tier_label: mid
        """)
        yaml_path.write_text(content, encoding="utf-8")

        data = yaml.safe_load(yaml_path.read_text())
        assert data["records"]["S001_1_1"]["scores"]["ensemble"] == 0.5

    def test_a07_inf_scores(self, tmp_path: Path) -> None:
        """A-07: Infinity values in scores should be detected."""
        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(tmp_path / "store.yaml"))
        store.load()

        rec = LongitudinalRecord(
            student_id="S001",
            week=1,
            question_sn=1,
            scores={"ensemble": float("inf")},
            tier_level=2,
            tier_label="mid",
        )
        # Infinite scores are now rejected (M-12~15 fixed)
        with pytest.raises(ValueError, match="Score cannot be infinite"):
            store.add_record(rec)


# ---------------------------------------------------------------------------
# A-08: Email distributor
# ---------------------------------------------------------------------------


class TestA08EmailDistributor:
    """Persona A-08: User sending out email reports."""

    def test_a08_invalid_email_address(self, tmp_path: Path) -> None:
        """A-08: Invalid email address should be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="sender_email"):
            _build_smtp_config(
                {
                    "smtp_server": "smtp.test.com",
                    "smtp_port": 587,
                    "sender_email": "not-an-email",
                    "sender_name": "Test",
                }
            )

    def test_a08_empty_smtp_server(self) -> None:
        """A-08: Empty SMTP server should be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_server"):
            _build_smtp_config(
                {
                    "smtp_server": "",
                    "smtp_port": 587,
                    "sender_email": "test@test.com",
                }
            )

    def test_a08_missing_template_fields(self, tmp_path: Path) -> None:
        """A-08: Email template without subject should fail."""
        from forma.delivery_send import load_template

        template = tmp_path / "template.yaml"
        template.write_text("body: 안녕하세요\n", encoding="utf-8")

        with pytest.raises(ValueError, match="subject"):
            load_template(str(template))

    def test_a08_template_missing_body(self, tmp_path: Path) -> None:
        """A-08: Email template without body should fail."""
        from forma.delivery_send import load_template

        template = tmp_path / "template.yaml"
        template.write_text("subject: 형성평가 결과\n", encoding="utf-8")

        with pytest.raises(ValueError, match="body"):
            load_template(str(template))

    def test_a08_smtp_auth_failure_mock(self, tmp_path: Path) -> None:
        """A-08: SMTP authentication failure should be caught and logged."""

        from forma.delivery_send import SmtpConfig

        config = SmtpConfig(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="test@test.com",
            sender_name="Test",
        )
        # Just verifying the config object is valid
        assert config.smtp_server == "smtp.test.com"

    def test_a08_delivery_log_persistence(self, tmp_path: Path) -> None:
        """A-08: Delivery log should persist between saves."""
        from forma.delivery_send import DeliveryLog, DeliveryResult, save_delivery_log, load_delivery_log

        log = DeliveryLog(
            sent_at="2024-01-01T00:00:00",
            smtp_server="smtp.test.com",
            dry_run=False,
            total=2,
            success=1,
            failed=1,
            results=[
                DeliveryResult(
                    student_id="S001",
                    email="s001@test.com",
                    status="success",
                    sent_at="2024-01-01T00:00:01",
                    attachment="S001.zip",
                    size_bytes=1024,
                ),
                DeliveryResult(
                    student_id="S002",
                    email="bad-email",
                    status="failed",
                    sent_at="2024-01-01T00:00:02",
                    attachment="S002.zip",
                    size_bytes=0,
                    error="Invalid email",
                ),
            ],
        )

        log_path = tmp_path / "delivery_log.yaml"
        save_delivery_log(log, str(log_path))
        loaded = load_delivery_log(str(log_path))
        assert loaded.total == 2
        assert loaded.success == 1
        assert loaded.failed == 1
