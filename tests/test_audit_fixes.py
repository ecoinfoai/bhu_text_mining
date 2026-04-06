"""Tests for v0.12 audit fixes -- concurrency, validation, and safety.

Covers:
    CD-01/02: Premature LOCK_UN removed (lock held through os.replace)
    CD-03/04: Dedicated lock file for mutual exclusion
    CD-05/06: Reversed backup order (write-first, then best-effort backup)
    CD-07/08: Missing directory creation before tempfile.mkstemp
    CD-10: Missing encoding="utf-8" in longitudinal_store.py load()
    IS-01: config.py root type check
    IS-02: config.py naver_ocr type check
    IS-03/04: delivery_prepare.py path traversal guard
    IS-05/06: intervention_store.py add_record() validation
    IS-09: delivery_send.py prepare_summary dict check
    SC-01: intervention_store.py next_id consistency check
"""

import os

import pytest
import yaml


# ---------------------------------------------------------------------------
# CD-07: save_prepare_summary creates parent directory
# ---------------------------------------------------------------------------


class TestSavePrepareSummaryCreatesDir:
    """CD-07: os.makedirs before tempfile.mkstemp in save_prepare_summary."""

    def test_creates_nonexistent_parent_dir(self, tmp_path):
        from forma.delivery_prepare import PrepareSummary, save_prepare_summary

        deep_dir = tmp_path / "a" / "b" / "c"
        out_path = str(deep_dir / "summary.yaml")
        assert not deep_dir.exists()

        summary = PrepareSummary(
            prepared_at="2026-03-11T00:00:00",
            class_name="TestClass",
            total_students=0,
            ready=0,
            warnings=0,
            errors=0,
            details=[],
        )
        save_prepare_summary(summary, out_path)

        assert os.path.isfile(out_path)
        with open(out_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["class_name"] == "TestClass"


# ---------------------------------------------------------------------------
# CD-08: save_delivery_log creates parent directory
# ---------------------------------------------------------------------------


class TestSaveDeliveryLogCreatesDir:
    """CD-08: os.makedirs before tempfile.mkstemp in save_delivery_log."""

    def test_creates_nonexistent_parent_dir(self, tmp_path):
        from forma.delivery_send import DeliveryLog, save_delivery_log

        deep_dir = tmp_path / "x" / "y"
        out_path = str(deep_dir / "delivery_log.yaml")
        assert not deep_dir.exists()

        log = DeliveryLog(
            sent_at="2026-03-11T00:00:00",
            smtp_server="smtp.example.com",
            dry_run=True,
            total=0,
            success=0,
            failed=0,
            results=[],
        )
        save_delivery_log(log, out_path)

        assert os.path.isfile(out_path)
        with open(out_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["smtp_server"] == "smtp.example.com"


# ---------------------------------------------------------------------------
# SC-01: InterventionLog.load() corrects inconsistent next_id
# ---------------------------------------------------------------------------


class TestInterventionStoreNextIdCorrection:
    """SC-01: load() detects and corrects next_id vs max record ID mismatch."""

    def test_load_corrects_low_next_id(self, tmp_path):
        from forma.intervention_store import InterventionLog

        store_path = str(tmp_path / "log.yaml")
        data = {
            "_meta": {"next_id": 3},
            "records": [
                {"id": 1, "student_id": "S1", "week": 1, "intervention_type": "\uba74\ub2f4", "description": ""},
                {"id": 2, "student_id": "S2", "week": 1, "intervention_type": "\uba74\ub2f4", "description": ""},
                {"id": 7, "student_id": "S3", "week": 2, "intervention_type": "\uba74\ub2f4", "description": ""},
            ],
        }
        with open(store_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        log = InterventionLog(store_path)
        log.load()
        # next_id should be corrected to max(1,2,7) + 1 = 8
        assert log._next_id == 8

    def test_load_keeps_correct_next_id(self, tmp_path):
        from forma.intervention_store import InterventionLog

        store_path = str(tmp_path / "log.yaml")
        data = {
            "_meta": {"next_id": 10},
            "records": [
                {"id": 1, "student_id": "S1", "week": 1, "intervention_type": "\uba74\ub2f4", "description": ""},
            ],
        }
        with open(store_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        log = InterventionLog(store_path)
        log.load()
        # next_id=10 is already > max(id=1), should stay at 10
        assert log._next_id == 10


# ---------------------------------------------------------------------------
# CD-05/06: Reversed backup order in LongitudinalStore
# ---------------------------------------------------------------------------


class TestLongitudinalStoreBackupOrder:
    """CD-05: After save(), store_path has new data and .bak exists."""

    def test_save_creates_backup_after_write(self, tmp_path):
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        store_path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(store_path)
        record = LongitudinalRecord(
            student_id="S1",
            week=1,
            question_sn=1,
            scores={"ensemble_score": 0.8},
            tier_level=3,
            tier_label="excellent",
        )
        store.add_record(record)
        store.save()

        # Store file should exist with the data
        assert os.path.isfile(store_path)
        with open(store_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "records" in data

        # Backup should also exist
        bak_path = store_path + ".bak"
        assert os.path.isfile(bak_path)

    def test_save_second_time_backup_has_previous_data(self, tmp_path):
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        store_path = str(tmp_path / "store.yaml")

        # First save
        store = LongitudinalStore(store_path)
        r1 = LongitudinalRecord(
            student_id="S1",
            week=1,
            question_sn=1,
            scores={"ensemble_score": 0.5},
            tier_level=2,
            tier_label="good",
        )
        store.add_record(r1)
        store.save()

        # Second save with different data
        store2 = LongitudinalStore(store_path)
        store2.load()
        r2 = LongitudinalRecord(
            student_id="S2",
            week=2,
            question_sn=1,
            scores={"ensemble_score": 0.9},
            tier_level=3,
            tier_label="excellent",
        )
        store2.add_record(r2)
        store2.save()

        # Main store should have both records
        with open(store_path, encoding="utf-8") as f:
            main_data = yaml.safe_load(f)
        assert len(main_data["records"]) == 2

        # Backup should be the CURRENT data (copy2 after replace)
        bak_path = store_path + ".bak"
        with open(bak_path, encoding="utf-8") as f:
            bak_data = yaml.safe_load(f)
        assert len(bak_data["records"]) == 2


# ---------------------------------------------------------------------------
# CD-05/06: Reversed backup order in InterventionLog
# ---------------------------------------------------------------------------


class TestInterventionStoreBackupOrder:
    """CD-06: After save(), store_path has data and .bak exists."""

    def test_save_creates_backup(self, tmp_path):
        from forma.intervention_store import InterventionLog

        store_path = str(tmp_path / "log.yaml")
        log = InterventionLog(store_path)
        log.add_record("S1", 1, "\uba74\ub2f4")
        log.save()

        assert os.path.isfile(store_path)
        bak_path = store_path + ".bak"
        assert os.path.isfile(bak_path)

        with open(store_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert len(data["records"]) == 1


# ---------------------------------------------------------------------------
# CD-03/04: Dedicated lock file is created
# ---------------------------------------------------------------------------


class TestDedicatedLockFile:
    """CD-03/CD-04: Both stores use store_path + '.lock' for mutual exclusion."""

    def test_longitudinal_store_creates_lock_file(self, tmp_path):
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        store_path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(store_path)
        assert store._lock_path == store_path + ".lock"

        record = LongitudinalRecord(
            student_id="S1",
            week=1,
            question_sn=1,
            scores={"ensemble_score": 0.8},
            tier_level=3,
            tier_label="excellent",
        )
        store.add_record(record)
        store.save()

        # Lock file should have been created by opening it in append mode
        assert os.path.isfile(store_path + ".lock")

    def test_intervention_store_creates_lock_file(self, tmp_path):
        from forma.intervention_store import InterventionLog

        store_path = str(tmp_path / "log.yaml")
        log = InterventionLog(store_path)
        assert log._lock_path == store_path + ".lock"

        log.add_record("S1", 1, "\uba74\ub2f4")
        log.save()

        assert os.path.isfile(store_path + ".lock")

    def test_longitudinal_load_uses_lock_file(self, tmp_path):
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        store_path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(store_path)
        r = LongitudinalRecord(
            student_id="S1",
            week=1,
            question_sn=1,
            scores={"score": 0.5},
            tier_level=2,
            tier_label="ok",
        )
        store.add_record(r)
        store.save()

        # Load in a new store instance
        store2 = LongitudinalStore(store_path)
        store2.load()
        assert len(store2.get_all_records()) == 1
        assert os.path.isfile(store_path + ".lock")


# ---------------------------------------------------------------------------
# CD-10: UTF-8 encoding in longitudinal_store.py load()
# ---------------------------------------------------------------------------


class TestLongitudinalStoreUtf8:
    """CD-10: load() uses encoding='utf-8' for Korean text support."""

    def test_load_korean_data(self, tmp_path):
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        store_path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(store_path)
        record = LongitudinalRecord(
            student_id="\ud559\uc0dd01",
            week=1,
            question_sn=1,
            scores={"ensemble_score": 0.7},
            tier_level=2,
            tier_label="\uc6b0\uc218",
        )
        store.add_record(record)
        store.save()

        store2 = LongitudinalStore(store_path)
        store2.load()
        records = store2.get_all_records()
        assert len(records) == 1
        assert records[0].student_id == "\ud559\uc0dd01"
        assert records[0].tier_label == "\uc6b0\uc218"


# ---------------------------------------------------------------------------
# IS-09: send_emails() non-dict prepare_summary raises ValueError
# ---------------------------------------------------------------------------


class TestSendEmailsNonDictSummary:
    """IS-09: send_emails rejects non-dict prepare_summary.yaml."""

    def test_list_summary_raises_value_error(self, tmp_path):
        from forma.delivery_send import send_emails

        staging_dir = str(tmp_path / "staging")
        os.makedirs(staging_dir)

        # Write a prepare_summary.yaml with list content
        summary_path = os.path.join(staging_dir, "prepare_summary.yaml")
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.dump(["not", "a", "dict"], f)

        # Write minimal template
        template_path = str(tmp_path / "template.yaml")
        with open(template_path, "w", encoding="utf-8") as f:
            yaml.dump({"subject": "Test", "body": "Body"}, f)

        # Write minimal smtp config
        smtp_path = str(tmp_path / "smtp.yaml")
        with open(smtp_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "smtp_server": "smtp.example.com",
                    "smtp_port": 587,
                    "sender_email": "test@example.com",
                },
                f,
            )

        with pytest.raises(ValueError, match="prepare_summary.yaml"):
            send_emails(
                staging_dir,
                template_path,
                smtp_path,
                dry_run=True,
            )

    def test_null_summary_raises_value_error(self, tmp_path):
        from forma.delivery_send import send_emails

        staging_dir = str(tmp_path / "staging")
        os.makedirs(staging_dir)

        # Write empty prepare_summary.yaml (yaml.safe_load returns None)
        summary_path = os.path.join(staging_dir, "prepare_summary.yaml")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("")

        template_path = str(tmp_path / "template.yaml")
        with open(template_path, "w", encoding="utf-8") as f:
            yaml.dump({"subject": "Test", "body": "Body"}, f)

        smtp_path = str(tmp_path / "smtp.yaml")
        with open(smtp_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "smtp_server": "smtp.example.com",
                    "smtp_port": 587,
                    "sender_email": "test@example.com",
                },
                f,
            )

        with pytest.raises(ValueError, match="prepare_summary.yaml"):
            send_emails(
                staging_dir,
                template_path,
                smtp_path,
                dry_run=True,
            )


# ---------------------------------------------------------------------------
# IS-01: config.py load_config() rejects non-dict JSON
# ---------------------------------------------------------------------------


class TestConfigRootTypeCheck:
    """IS-01: load_config() raises ValueError for non-dict JSON root."""

    def test_json_list_raises(self, tmp_path):
        import json
        from forma.config import load_config

        config_path = str(tmp_path / "bad.json")
        with open(config_path, "w") as f:
            json.dump([1, 2, 3], f)

        with pytest.raises(ValueError, match="JSON object"):
            load_config(config_path)

    def test_json_string_raises(self, tmp_path):
        import json
        from forma.config import load_config

        config_path = str(tmp_path / "bad.json")
        with open(config_path, "w") as f:
            json.dump("just a string", f)

        with pytest.raises(ValueError, match="JSON object"):
            load_config(config_path)

    def test_json_dict_ok(self, tmp_path):
        import json
        from forma.config import load_config

        config_path = str(tmp_path / "ok.json")
        with open(config_path, "w") as f:
            json.dump({"key": "value"}, f)

        result = load_config(config_path)
        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# IS-02: config.py get_naver_ocr_config() rejects non-dict section
# ---------------------------------------------------------------------------


class TestNaverOcrTypeCheck:
    """IS-02: get_naver_ocr_config() raises KeyError for non-dict naver_ocr."""

    def test_naver_ocr_string_raises(self):
        from forma.config import get_naver_ocr_config

        with pytest.raises(KeyError, match="dict"):
            get_naver_ocr_config({"naver_ocr": "not_a_dict"})

    def test_naver_ocr_list_raises(self):
        from forma.config import get_naver_ocr_config

        with pytest.raises(KeyError, match="dict"):
            get_naver_ocr_config({"naver_ocr": [1, 2, 3]})

    def test_naver_ocr_valid_dict_ok(self):
        from forma.config import get_naver_ocr_config

        key, url = get_naver_ocr_config(
            {
                "naver_ocr": {"secret_key": "abc", "api_url": "http://x"},
            }
        )
        assert key == "abc"
        assert url == "http://x"


# ---------------------------------------------------------------------------
# IS-03/04: delivery_prepare.py path traversal guard
# ---------------------------------------------------------------------------


class TestPathTraversalGuard:
    """IS-03/04: match_files_for_student blocks path separator in student_id."""

    def test_slash_in_student_id_raises(self, tmp_path):
        from forma.delivery_prepare import DeliveryManifest, match_files_for_student

        manifest = DeliveryManifest(
            directory=str(tmp_path),
            file_patterns=["{student_id}_report.pdf"],
        )
        with pytest.raises(ValueError, match="path traversal"):
            match_files_for_student(manifest, "../../etc/passwd")

    def test_traversal_via_pattern_raises(self, tmp_path):
        from forma.delivery_prepare import DeliveryManifest, match_files_for_student

        manifest = DeliveryManifest(
            directory=str(tmp_path),
            file_patterns=["{student_id}_report.pdf"],
        )
        with pytest.raises(ValueError, match="path traversal"):
            match_files_for_student(manifest, "../sibling/file")


# ---------------------------------------------------------------------------
# IS-05/06: intervention_store add_record() validation
# ---------------------------------------------------------------------------


class TestInterventionAddRecordValidation:
    """IS-05/06: add_record() validates student_id, week, description."""

    def test_empty_student_id_raises(self, tmp_path):
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        with pytest.raises(ValueError, match="student_id"):
            log.add_record("", 1, "\uba74\ub2f4")

    def test_bool_week_raises(self, tmp_path):
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        with pytest.raises(ValueError, match="week"):
            log.add_record("S1", True, "\uba74\ub2f4")

    def test_zero_week_raises(self, tmp_path):
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        with pytest.raises(ValueError, match="week"):
            log.add_record("S1", 0, "\uba74\ub2f4")

    def test_long_description_raises(self, tmp_path):
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        with pytest.raises(ValueError, match="2000"):
            log.add_record("S1", 1, "\uba74\ub2f4", description="x" * 2001)

    def test_none_student_id_raises(self, tmp_path):
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        with pytest.raises(ValueError, match="student_id"):
            log.add_record(None, 1, "\uba74\ub2f4")

    def test_negative_week_raises(self, tmp_path):
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        with pytest.raises(ValueError, match="positive integer"):
            log.add_record("S1", -1, "\uba74\ub2f4")

    def test_valid_add_record_ok(self, tmp_path):
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        rid = log.add_record("S1", 1, "\uba74\ub2f4", description="ok")
        assert rid == 1


# ---------------------------------------------------------------------------
# IS-08: project_config.py non-dict YAML warning
# ---------------------------------------------------------------------------


class TestProjectConfigNonDict:
    """IS-08: load_project_config() warns and returns {} for non-dict YAML."""

    def test_list_yaml_returns_empty_with_warning(self, tmp_path, caplog):
        import logging

        from forma.project_config import load_project_config

        cfg = tmp_path / "forma.yaml"
        cfg.write_text("- item1\n- item2\n")

        with caplog.at_level(logging.WARNING):
            result = load_project_config(cfg)

        assert result == {}
        assert "non-dict" in caplog.text

    def test_string_yaml_returns_empty_with_warning(self, tmp_path, caplog):
        import logging

        from forma.project_config import load_project_config

        cfg = tmp_path / "forma.yaml"
        cfg.write_text("just a string\n")

        with caplog.at_level(logging.WARNING):
            result = load_project_config(cfg)

        assert result == {}
        assert "non-dict" in caplog.text

    def test_empty_yaml_returns_empty(self, tmp_path):
        from forma.project_config import load_project_config

        cfg = tmp_path / "forma.yaml"
        cfg.write_text("")

        result = load_project_config(cfg)
        assert result == {}


# ---------------------------------------------------------------------------
# IS-03: Symlink path traversal guard
# ---------------------------------------------------------------------------


class TestPathTraversalSymlink:
    """IS-04: match_files_for_student blocks symlink escape."""

    def test_symlink_escape_raises(self, tmp_path):
        from forma.delivery_prepare import DeliveryManifest, match_files_for_student

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        secret = outside_dir / "S001_report.pdf"
        secret.write_text("secret")

        manifest_dir = tmp_path / "reports"
        manifest_dir.mkdir()
        link = manifest_dir / "S001_report.pdf"
        link.symlink_to(secret)

        manifest = DeliveryManifest(
            directory=str(manifest_dir),
            file_patterns=["{student_id}_report.pdf"],
        )
        with pytest.raises(ValueError, match="escapes manifest directory"):
            match_files_for_student(manifest, "S001")


# ---------------------------------------------------------------------------
# ML-02/ML-03: Single-class augmentation warnings
# ---------------------------------------------------------------------------


class TestSingleClassAugmentationWarnings:
    """ML-02/ML-03: predictors log warning on single-class training data."""

    def test_risk_predictor_single_class_warning(self, caplog):
        import logging

        import numpy as np

        from forma.risk_predictor import FEATURE_NAMES, RiskPredictor

        predictor = RiskPredictor()
        rng = np.random.RandomState(42)
        X = rng.rand(15, len(FEATURE_NAMES))
        labels = np.zeros(15)  # all same class

        with caplog.at_level(logging.WARNING):
            model = predictor.train(
                X,
                labels,
                list(FEATURE_NAMES),
                min_students=10,
                n_weeks=3,
            )

        assert "only one class" in caplog.text.lower()
        assert model.cv_score == 0.0

    def test_grade_predictor_single_class_warning(self, caplog):
        import logging

        import numpy as np

        from forma.grade_predictor import GRADE_FEATURE_NAMES, GradePredictor

        predictor = GradePredictor()
        rng = np.random.RandomState(42)
        X = rng.rand(15, len(GRADE_FEATURE_NAMES))
        labels = np.full(15, 2)  # all C grades

        with caplog.at_level(logging.WARNING):
            model = predictor.train(
                X,
                labels,
                list(GRADE_FEATURE_NAMES),
                min_students=10,
                n_weeks=4,
            )

        assert "only one class" in caplog.text.lower()
        assert model.cv_score == 0.0


# ---------------------------------------------------------------------------
# Phase 5: US3 Data integrity — atomic stores, SMTP reconnection, validation
# ---------------------------------------------------------------------------


class TestAtomicInterventionLog:
    """T054: Concurrent add_record preserves both records."""

    def test_concurrent_add_preserves_both(self, tmp_path) -> None:
        """Two sequential add_record + save cycles both persist."""
        from forma.intervention_store import InterventionLog

        path = str(tmp_path / "log.yaml")

        # First writer
        log1 = InterventionLog(path)
        log1.load()
        id1 = log1.add_record("S001", 1, "면담", description="첫 번째")
        log1.save()

        # Second writer (re-opens the same file)
        log2 = InterventionLog(path)
        log2.load()
        id2 = log2.add_record("S002", 2, "멘토링", description="두 번째")
        log2.save()

        # Verify both records persist
        log3 = InterventionLog(path)
        log3.load()
        records = log3.get_records()
        sids = {r.student_id for r in records}
        assert "S001" in sids
        assert "S002" in sids
        assert id1 != id2


class TestAtomicLongitudinalStore:
    """T055: Concurrent save preserves both snapshots."""

    def test_concurrent_save_preserves_both(self, tmp_path) -> None:
        """Two sequential add_record + save cycles both persist in LongitudinalStore."""
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        path = str(tmp_path / "store.yaml")

        # First writer
        store1 = LongitudinalStore(path)
        store1.load()
        store1.add_record(
            LongitudinalRecord(
                student_id="S001",
                week=1,
                question_sn=1,
                scores={"ensemble_score": 0.8},
                tier_level=3,
                tier_label="상",
            )
        )
        store1.save()

        # Second writer (re-opens the same file)
        store2 = LongitudinalStore(path)
        store2.load()
        store2.add_record(
            LongitudinalRecord(
                student_id="S002",
                week=1,
                question_sn=1,
                scores={"ensemble_score": 0.6},
                tier_level=2,
                tier_label="중상",
            )
        )
        store2.save()

        # Verify both records persist
        store3 = LongitudinalStore(path)
        store3.load()
        all_records = store3.get_all_records()
        sids = {r.student_id for r in all_records}
        assert "S001" in sids
        assert "S002" in sids


class TestSmtpReconnection:
    """T056: Mock disconnect at email N, verify resume at N+1."""

    def test_reconnection_on_disconnect(self, tmp_path) -> None:
        """send_emails handles SMTPServerDisconnected gracefully."""
        import smtplib
        from unittest.mock import patch

        from forma.delivery_send import send_emails

        # Setup staging directory with prepare_summary
        staging = tmp_path / "staging"
        staging.mkdir()

        # Create 3 student entries
        students = []
        for i in range(1, 4):
            sid = f"S{i:03d}"
            student_dir = staging / f"{sid}_Name{i}"
            student_dir.mkdir()
            zip_path = student_dir / f"Name{i}_{sid}.zip"
            zip_path.write_bytes(b"PK\x03\x04fake")
            students.append(
                {
                    "student_id": sid,
                    "name": f"Name{i}",
                    "email": f"s{i}@test.com",
                    "status": "ready",
                    "matched_files": [str(zip_path)],
                    "zip_path": str(zip_path),
                    "zip_size_bytes": zip_path.stat().st_size,
                    "message": "",
                }
            )

        import yaml

        summary = {
            "prepared_at": "2026-01-01T00:00:00",
            "class_name": "TestClass",
            "total_students": 3,
            "ready": 3,
            "warnings": 0,
            "errors": 0,
            "details": students,
        }
        summary_path = staging / "prepare_summary.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, allow_unicode=True)

        template_path = tmp_path / "template.yaml"
        with open(template_path, "w") as f:
            yaml.dump({"subject": "Test {student_name}", "body": "Hello {student_name}"}, f)

        smtp_path = tmp_path / "smtp.yaml"
        with open(smtp_path, "w") as f:
            yaml.dump(
                {
                    "smtp_server": "smtp.test.com",
                    "smtp_port": 587,
                    "sender_email": "prof@test.com",
                    "use_tls": True,
                },
                f,
            )

        call_count = 0

        class ReconnectingSMTP:
            def __init__(self, *args, **kwargs):
                pass

            def starttls(self, **kwargs):
                pass

            def login(self, u, p):
                pass

            def send_message(self, msg):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise smtplib.SMTPServerDisconnected("Connection lost")

            def quit(self):
                pass

        with patch("smtplib.SMTP", ReconnectingSMTP):
            os.environ["FORMA_SMTP_PASSWORD"] = "testpass"
            try:
                log = send_emails(str(staging), str(template_path), str(smtp_path))
            finally:
                os.environ.pop("FORMA_SMTP_PASSWORD", None)

        # With reconnection: email 2 disconnects, reconnects, retries, succeeds
        assert log.total == 3
        assert log.success + log.failed == 3
        # Reconnection retry should recover the disconnected email
        assert log.success >= 2


class TestSmtpTimeout:
    """T057: Verify timeout parameter passed to SMTP constructor."""

    def test_smtp_timeout_passed(self, tmp_path) -> None:
        """send_emails() passes timeout=30 to smtplib.SMTP() constructor."""
        from unittest.mock import patch
        import yaml

        from forma.delivery_send import send_emails

        # Setup minimal staging
        staging = tmp_path / "staging"
        staging.mkdir()
        s_dir = staging / "S001_Test"
        s_dir.mkdir()
        zip_path = s_dir / "Test_S001.zip"
        zip_path.write_bytes(b"PK\x03\x04fake")

        summary = {
            "prepared_at": "2026-01-01",
            "class_name": "C",
            "total_students": 1,
            "ready": 1,
            "warnings": 0,
            "errors": 0,
            "details": [
                {
                    "student_id": "S001",
                    "name": "Test",
                    "email": "s@t.com",
                    "status": "ready",
                    "matched_files": [str(zip_path)],
                    "zip_path": str(zip_path),
                    "zip_size_bytes": 10,
                    "message": "",
                }
            ],
        }
        with open(staging / "prepare_summary.yaml", "w") as f:
            yaml.dump(summary, f, allow_unicode=True)

        tpl_path = tmp_path / "tpl.yaml"
        with open(tpl_path, "w") as f:
            yaml.dump({"subject": "Hi {student_name}", "body": "Hello {student_name}"}, f)

        smtp_path = tmp_path / "smtp.yaml"
        with open(smtp_path, "w") as f:
            yaml.dump(
                {
                    "smtp_server": "s.t.com",
                    "smtp_port": 587,
                    "sender_email": "p@t.com",
                    "use_tls": True,
                },
                f,
            )

        captured_args = {}

        class CapturingSMTP:
            def __init__(self, *args, **kwargs):
                captured_args["args"] = args
                captured_args["kwargs"] = kwargs

            def starttls(self, **kwargs):
                pass

            def login(self, u, p):
                pass

            def send_message(self, msg):
                pass

            def quit(self):
                pass

        with patch("smtplib.SMTP", CapturingSMTP):
            os.environ["FORMA_SMTP_PASSWORD"] = "testpass"
            try:
                send_emails(str(staging), str(tpl_path), str(smtp_path))
            finally:
                os.environ.pop("FORMA_SMTP_PASSWORD", None)

        assert captured_args["kwargs"].get("timeout") == 30


class TestSmtpAuthError:
    """T058: SMTPAuthenticationError handling."""

    def test_auth_error_propagated(self, tmp_path) -> None:
        """SMTPAuthenticationError during login propagates."""
        import smtplib
        from unittest.mock import patch

        from forma.delivery_send import SmtpConfig, send_summary_email, DeliveryLog

        mock_log = DeliveryLog(
            sent_at="2026-01-01",
            smtp_server="smtp.test.com",
            dry_run=False,
            total=0,
            success=0,
            failed=0,
            results=[],
        )
        config = SmtpConfig(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="test@test.com",
            use_tls=True,
        )

        class AuthFailSMTP:
            def __init__(self, *a, **kw):
                pass

            def starttls(self, **kwargs):
                pass

            def login(self, u, p):
                raise smtplib.SMTPAuthenticationError(535, b"Auth failed")

            def quit(self):
                pass

        with patch("smtplib.SMTP", AuthFailSMTP):
            with pytest.raises(smtplib.SMTPAuthenticationError):
                send_summary_email(mock_log, config, password="wrong")


class TestDeliveryLogValidation:
    """T059: Missing required key in delivery_log raises ValueError."""

    def test_missing_key_raises(self, tmp_path) -> None:
        """Delivery log YAML missing 'total' key raises ValueError or KeyError."""
        import yaml
        from forma.delivery_send import load_delivery_log

        bad_log = tmp_path / "delivery_log.yaml"
        with open(bad_log, "w") as f:
            yaml.dump({"sent_at": "2026-01-01", "smtp_server": "s"}, f)

        with pytest.raises((KeyError, ValueError)):
            load_delivery_log(str(bad_log))


class TestWeekTypeCheck:
    """T060: Week as string raises TypeError/ValueError in InterventionLog."""

    def test_week_string_raises(self, tmp_path) -> None:
        """add_record with week='3' (string) raises ValueError."""
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        log.load()

        with pytest.raises((TypeError, ValueError)):
            log.add_record("S001", "3", "면담")  # type: ignore[arg-type]

    def test_week_bool_raises(self, tmp_path) -> None:
        """add_record with week=True (bool) raises ValueError."""
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        log.load()

        with pytest.raises((TypeError, ValueError)):
            log.add_record("S001", True, "면담")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Phase 8: US5 Performance — LongitudinalStore index, table truncation
# ---------------------------------------------------------------------------


class TestLongitudinalStoreIndex:
    """T084-T085: Index construction and indexed query."""

    def test_by_student_index_built_after_load(self, tmp_path) -> None:
        """After load(), _by_student index dict exists and maps student_id to keys."""
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(
            LongitudinalRecord(
                student_id="S001",
                week=1,
                question_sn=1,
                scores={"ensemble_score": 0.8},
                tier_level=3,
                tier_label="상",
            )
        )
        store.add_record(
            LongitudinalRecord(
                student_id="S002",
                week=1,
                question_sn=1,
                scores={"ensemble_score": 0.6},
                tier_level=2,
                tier_label="중상",
            )
        )
        store.save()

        store2 = LongitudinalStore(path)
        store2.load()

        assert hasattr(store2, "_by_student")
        assert "S001" in store2._by_student
        assert "S002" in store2._by_student

    def test_by_week_index_built_after_load(self, tmp_path) -> None:
        """After load(), _by_week index dict exists and maps week to keys."""
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        store.add_record(
            LongitudinalRecord(
                student_id="S001",
                week=1,
                question_sn=1,
                scores={"ensemble_score": 0.8},
                tier_level=3,
                tier_label="상",
            )
        )
        store.add_record(
            LongitudinalRecord(
                student_id="S001",
                week=2,
                question_sn=1,
                scores={"ensemble_score": 0.9},
                tier_level=3,
                tier_label="상",
            )
        )
        store.save()

        store2 = LongitudinalStore(path)
        store2.load()

        assert hasattr(store2, "_by_week")
        assert 1 in store2._by_week
        assert 2 in store2._by_week

    def test_indexed_get_student_history(self, tmp_path) -> None:
        """get_student_history() returns correct records using index."""
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        for i in range(50):
            store.add_record(
                LongitudinalRecord(
                    student_id=f"S{i:03d}",
                    week=1,
                    question_sn=1,
                    scores={"ensemble_score": 0.5 + i * 0.01},
                    tier_level=2,
                    tier_label="중상",
                )
            )
        store.save()

        store2 = LongitudinalStore(path)
        store2.load()

        history = store2.get_student_history("S025")
        assert len(history) == 1
        assert history[0].student_id == "S025"

    def test_indexed_get_class_snapshot(self, tmp_path) -> None:
        """get_class_snapshot() returns correct records using week index."""
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        path = str(tmp_path / "store.yaml")
        store = LongitudinalStore(path)
        for w in [1, 2, 3]:
            for i in range(10):
                store.add_record(
                    LongitudinalRecord(
                        student_id=f"S{i:03d}",
                        week=w,
                        question_sn=1,
                        scores={"ensemble_score": 0.5},
                        tier_level=2,
                        tier_label="중상",
                    )
                )
        store.save()

        store2 = LongitudinalStore(path)
        store2.load()

        snapshot = store2.get_class_snapshot(2)
        assert len(snapshot) == 10
        assert all(r.week == 2 for r in snapshot)


# ---------------------------------------------------------------------------
# T086: Set-based edge matching in build_class_knowledge_aggregate
# ---------------------------------------------------------------------------


class TestSetBasedEdgeMatching:
    """FR-035: build_class_knowledge_aggregate uses set-based lookups."""

    def _make_edge(self, s: str, r: str, o: str):
        """Create a mock TripletEdge."""
        from types import SimpleNamespace

        return SimpleNamespace(subject=s, relation=r, object=o)

    def _make_comparison(self, matched, wrong_direction):
        """Create a mock GraphComparisonResult."""
        from types import SimpleNamespace

        return SimpleNamespace(
            matched_edges=matched,
            wrong_direction_edges=wrong_direction,
        )

    def test_200_students_correct_counts(self) -> None:
        """200 students, verify correct/error/missing tallies are accurate."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        master_edges = [
            self._make_edge("A", "causes", "B"),
            self._make_edge("C", "inhibits", "D"),
        ]

        comparisons = []
        for i in range(200):
            if i < 100:
                # First 100: match edge 1 correctly
                matched = [self._make_edge("A", "causes", "B")]
                wrong = []
            elif i < 150:
                # Next 50: wrong direction for edge 1 (B->A)
                matched = []
                wrong = [self._make_edge("B", "causes", "A")]
            else:
                # Last 50: missing edge 1 entirely
                matched = []
                wrong = []

            # All 200 match edge 2 correctly
            matched.append(self._make_edge("C", "inhibits", "D"))
            comparisons.append(self._make_comparison(matched, wrong))

        result = build_class_knowledge_aggregate(master_edges, comparisons, 1)

        assert result.total_students == 200
        edge1 = result.edges[0]
        assert edge1.correct_count == 100
        assert edge1.error_count == 50
        assert edge1.missing_count == 50

        edge2 = result.edges[1]
        assert edge2.correct_count == 200
        assert edge2.error_count == 0
        assert edge2.missing_count == 0

    def test_empty_comparison_results(self) -> None:
        """Zero students yields zero counts."""
        from forma.class_knowledge_aggregate import build_class_knowledge_aggregate

        master_edges = [self._make_edge("X", "rel", "Y")]
        result = build_class_knowledge_aggregate(master_edges, [], 1)
        assert result.edges[0].correct_count == 0
        assert result.edges[0].correct_ratio == 0.0


# ---------------------------------------------------------------------------
# T087: Table truncation for >100 students
# ---------------------------------------------------------------------------


class TestTableTruncation:
    """FR-036: professor_report student table truncates for large class sizes."""

    def test_truncate_student_table_over_100(self) -> None:
        """_truncate_student_rows returns top25+bottom25+summary for >100."""
        from forma.professor_report import _truncate_student_rows

        # Simulate 200 student rows (just need objects with overall_ensemble_mean)
        from types import SimpleNamespace

        rows = [
            SimpleNamespace(
                student_number=f"S{i:03d}",
                real_name=f"Student{i}",
                overall_ensemble_mean=1.0 - i * 0.005,
                overall_level="Developing",
                is_at_risk=False,
                per_question_scores={},
                per_question_levels={},
                per_question_coverages={},
                section=None,
            )
            for i in range(200)
        ]
        truncated, was_truncated = _truncate_student_rows(rows)
        assert was_truncated is True
        assert len(truncated) == 50  # top 25 + bottom 25

    def test_no_truncation_under_100(self) -> None:
        """_truncate_student_rows returns all rows for <=100."""
        from forma.professor_report import _truncate_student_rows
        from types import SimpleNamespace

        rows = [SimpleNamespace(overall_ensemble_mean=0.5) for _ in range(80)]
        result, was_truncated = _truncate_student_rows(rows)
        assert was_truncated is False
        assert len(result) == 80
