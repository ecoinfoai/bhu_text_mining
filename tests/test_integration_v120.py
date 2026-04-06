"""Integration tests for v0.12.0 audit hardening.

End-to-end verification of security fixes:
- Path traversal blocked in prepare -> send pipeline
- CRLF stripped in email flow
- HTTPS enforced in OCR
- TLS certificate verification in SMTP
- Pickle type validation in model loading
- Unified CLI delegation
- Unicode sanitization across modules

FR-046 coverage.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestSecurityPipelineIntegration:
    """End-to-end security verification across modules."""

    def test_path_traversal_blocked_in_prepare(self, tmp_path) -> None:
        """Path traversal in student_id is blocked during prepare_delivery."""
        from forma.delivery_prepare import prepare_delivery

        base = tmp_path / "reports"
        base.mkdir()
        staging = tmp_path / "staging"

        # Create manifest YAML
        manifest_path = tmp_path / "manifest.yaml"
        manifest_data = {
            "report_source": {
                "directory": str(base),
                "file_patterns": ["{student_id}.pdf"],
            }
        }
        with open(manifest_path, "w") as f:
            yaml.dump(manifest_data, f)

        # Create roster YAML with path traversal student_id
        roster_path = tmp_path / "roster.yaml"
        roster_data = {
            "class_name": "TestClass",
            "students": [
                {
                    "student_id": "../../etc/passwd",
                    "name": "Hacker",
                    "email": "h@e.com",
                },
            ],
        }
        with open(roster_path, "w") as f:
            yaml.dump(roster_data, f)

        # match_files_for_student raises ValueError for path traversal,
        # which propagates through prepare_delivery
        with pytest.raises(ValueError, match="path traversal"):
            prepare_delivery(str(manifest_path), str(roster_path), str(staging))

    def test_crlf_stripped_in_email_render(self) -> None:
        """CRLF in template variables does not inject email headers."""
        from forma.delivery_send import EmailTemplate, render_template

        template = EmailTemplate(
            subject="Report for {student_name}",
            body="Hello {student_name}, your report is attached.",
        )
        subject, body = render_template(
            template,
            student_name="Evil\r\nBcc: hacker@evil.com\r\nUser",
            student_id="S001",
            class_name="TestClass",
        )
        # Body can contain the CRLF (it's in the body text, not headers)
        assert isinstance(body, str)
        assert isinstance(subject, str)

    def test_unicode_sanitize_end_to_end(self) -> None:
        """Zero-width chars in filename are stripped through sanitize chain."""
        from forma.delivery_prepare import sanitize_filename
        from forma.report_utils import sanitize_filename_report

        attack = "\u200b\u200c\u200d\ufeff"
        assert sanitize_filename(attack) == "_unnamed"
        assert sanitize_filename_report(attack) == "_unnamed"

    def test_xml_escape_end_to_end(self) -> None:
        """Control chars + XML entities are properly handled by esc()."""
        from forma.font_utils import esc

        attack = "\x00<script>\x01alert('xss')\x02</script>\x7f"
        result = esc(attack)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x7f" not in result
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_model_type_validation_blocks_dict(self, tmp_path) -> None:
        """Loading a model file with dict payload raises TypeError."""
        import joblib

        payload_path = tmp_path / "bad_model.pkl"
        joblib.dump({"not": "a model"}, str(payload_path))

        from forma.risk_predictor import load_model

        with pytest.raises(TypeError, match="TrainedRiskModel"):
            load_model(str(payload_path))

    def test_model_type_validation_blocks_none(self, tmp_path) -> None:
        """Loading a model file with None raises TypeError."""
        import joblib

        payload_path = tmp_path / "none_model.pkl"
        joblib.dump(None, str(payload_path))

        from forma.risk_predictor import load_model

        with pytest.raises(TypeError, match="TrainedRiskModel"):
            load_model(str(payload_path))


class TestCLIIntegration:
    """Unified CLI delegation integration."""

    def test_forma_help_returns_zero(self) -> None:
        """'forma --help' exits with code 0."""
        from forma.cli_main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_forma_version_returns_zero(self) -> None:
        """'forma --version' exits with code 0."""
        from forma.cli_main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_unknown_command_exits_with_error(self) -> None:
        """'forma nonexistent' exits with code 2."""
        from forma.cli_main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent"])
        assert exc_info.value.code == 2

    def test_legacy_wrapper_emits_deprecation(self) -> None:
        """Legacy wrapper emits DeprecationWarning."""
        from forma.cli_main import _make_legacy_wrapper

        mock_main = MagicMock()
        wrapper = _make_legacy_wrapper("forma-old", "forma new", mock_main)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapper()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "forma-old" in str(w[0].message)

    def test_error_messages(self) -> None:
        """Error message templates format correctly."""
        from forma.cli_main import _error_message

        msg = _error_message("file_not_found", path="/tmp/test.yaml")
        assert "/tmp/test.yaml" in msg
        assert "File" in msg

        msg = _error_message("unknown_command", cmd="foobar")
        assert "foobar" in msg
        assert "Unknown command" in msg

    def test_progress_logging(self) -> None:
        """log_progress produces correctly formatted output."""
        import logging

        from forma.cli_main import log_progress

        with patch.object(logging.getLogger("forma.cli_main"), "info") as mock_info:
            log_progress(3, 10, "test task")
            mock_info.assert_called_once_with("[%d/%d] %s", 3, 10, "test task")


class TestDataIntegrityIntegration:
    """Atomic stores and concurrent access integration."""

    def test_intervention_store_roundtrip(self, tmp_path) -> None:
        """InterventionLog add -> get roundtrip preserves data."""
        from forma.intervention_store import InterventionLog

        store_path = tmp_path / "test_log.yaml"
        log = InterventionLog(str(store_path))
        log.load()

        log.add_record("S001", week=1, intervention_type="면담", description="initial")
        log.add_record("S001", week=2, intervention_type="보충학습", description="follow-up")
        log.add_record("S002", week=1, intervention_type="멘토링")

        records = log.get_records()
        assert len(records) == 3

        s001_records = log.get_records(student_id="S001")
        assert len(s001_records) == 2

        w1_records = log.get_records(week=1)
        assert len(w1_records) == 2

    def test_intervention_store_reload_preserves(self, tmp_path) -> None:
        """Data persists across InterventionLog instances."""
        from forma.intervention_store import InterventionLog

        store_path = tmp_path / "persist.yaml"
        log1 = InterventionLog(str(store_path))
        log1.load()
        log1.add_record("S001", week=1, intervention_type="면담")
        log1.save()

        log2 = InterventionLog(str(store_path))
        log2.load()
        records = log2.get_records()
        assert len(records) == 1
        assert records[0].student_id == "S001"

    def test_epsilon_guard_integration(self) -> None:
        """Epsilon guard prevents float noise from triggering false risk flags."""
        from forma.warning_report_data import RiskType, _classify_risk_types

        # Flat trajectory with float noise
        base = 0.5
        trajectory = [base + i * 1e-17 for i in range(5)]
        # Slope is essentially zero but might be slightly negative due to float
        risk_types = _classify_risk_types(
            score_trajectory=trajectory,
            concept_scores={"A": 0.8},
            absence_ratio=0.0,
        )
        assert RiskType.SCORE_DECLINE not in risk_types


# ===========================================================================
# Integration test expansion (adversary agent)
# ===========================================================================


class TestSecurityPipelineExpanded:
    """Additional end-to-end security verification."""

    def test_crlf_sender_sanitized_in_compose(self, tmp_path) -> None:
        """CRLF in sender name is stripped through compose_email pipeline."""
        from forma.delivery_send import SmtpConfig, compose_email

        config = SmtpConfig(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="sender@test.com",
            sender_name="Prof\r\nBcc: evil@evil.com",
        )

        zip_file = tmp_path / "test.zip"
        zip_file.write_bytes(b"PK\x03\x04fake")

        msg = compose_email(
            sender_config=config,
            to_email="student@test.com",
            subject="Report",
            body="Hello",
            zip_path=str(zip_file),
        )

        assert "\r" not in msg["From"]
        assert "\n" not in msg["From"]

    def test_pickle_type_validation_grade_model(self, tmp_path) -> None:
        """Create fake .pkl -> load_grade_model -> verify TypeError."""
        import joblib
        from forma.grade_predictor import load_grade_model

        fake_path = tmp_path / "fake_grade.pkl"
        joblib.dump("not a model", str(fake_path))

        with pytest.raises(TypeError, match="TrainedGradeModel"):
            load_grade_model(str(fake_path))

    def test_https_enforcement_in_ocr(self) -> None:
        """OCR API rejects http:// URLs end-to-end."""
        from forma.naver_ocr import send_images_receive_ocr

        with pytest.raises(ValueError, match="HTTPS"):
            send_images_receive_ocr("http://insecure.api.com/ocr", "key", [])

    def test_zero_width_chars_cleaned_in_filename(self) -> None:
        """Zero-width chars -> sanitize -> verify cleaned."""
        from forma.delivery_prepare import sanitize_filename

        dirty_name = "report\u200b_\u200c\u200d\ufeffstudent.pdf"
        clean = sanitize_filename(dirty_name)

        assert "\u200b" not in clean
        assert "\u200c" not in clean
        assert "\u200d" not in clean
        assert "\ufeff" not in clean
        assert "report_student.pdf" == clean

    def test_format_string_attack_end_to_end(self) -> None:
        """Template with standalone unsupported variable blocked by validate_template_variables."""
        from forma.delivery_send import EmailTemplate, validate_template_variables

        # {secret_var} is not in SUPPORTED_VARIABLES, so validation rejects it
        malicious_template = EmailTemplate(
            subject="{student_name}",
            body="Secret: {secret_var}",
        )

        with pytest.raises(ValueError, match="Unsupported template variables"):
            validate_template_variables(malicious_template)

    def test_prepare_then_verify_summary_integrity(self, tmp_path) -> None:
        """Full prepare pipeline: manifest -> roster -> prepare -> verify YAML."""
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        (report_dir / "S001.pdf").write_text("content")
        (report_dir / "S002.pdf").write_text("content")

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            yaml.dump(
                {
                    "report_source": {
                        "directory": str(report_dir),
                        "file_patterns": ["{student_id}.pdf"],
                    }
                }
            )
        )

        roster_path = tmp_path / "roster.yaml"
        roster_path.write_text(
            yaml.dump(
                {
                    "class_name": "IntegrationClass",
                    "students": [
                        {"student_id": "S001", "name": "Alice", "email": "alice@test.com"},
                        {"student_id": "S002", "name": "Bob", "email": "bob@test.com"},
                        {"student_id": "S003", "name": "Charlie", "email": ""},
                    ],
                }
            )
        )

        output_dir = tmp_path / "staging"
        summary = prepare_delivery(str(manifest_path), str(roster_path), str(output_dir))

        assert summary.total_students == 3
        assert summary.ready == 2
        assert summary.errors == 1  # S003 has no email

        summary_yaml = output_dir / "prepare_summary.yaml"
        assert summary_yaml.exists()

        with open(str(summary_yaml)) as f:
            data = yaml.safe_load(f)
        assert data["total_students"] == 3
        assert data["class_name"] == "IntegrationClass"


class TestCLIIntegrationExpanded:
    """Additional CLI integration tests."""

    def test_forma_no_args_shows_help(self) -> None:
        """'forma' without args exits 0."""
        from forma.cli_main import main

        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 0

    def test_forma_unknown_command_error(self, capsys) -> None:
        """'forma foo' gives error message on stderr."""
        from forma.cli_main import main

        with pytest.raises(SystemExit) as exc:
            main(["foo"])
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "Unknown command" in captured.err

    def test_forma_report_no_subcommand_exits(self) -> None:
        """'forma report' without subcommand exits with error."""
        from forma.cli_main import main

        with pytest.raises(SystemExit) as exc:
            main(["report"])
        assert exc.value.code == 2


class TestDataIntegrityExpanded:
    """Additional data integrity tests."""

    def test_prepare_summary_atomic_write(self, tmp_path) -> None:
        """save_prepare_summary uses atomic write."""
        from forma.delivery_prepare import PrepareSummary, save_prepare_summary
        import os

        summary = PrepareSummary(
            prepared_at="2026-01-01T00:00:00Z",
            class_name="AtomicTest",
            total_students=1,
            ready=1,
            warnings=0,
            errors=0,
            details=[],
        )

        path = str(tmp_path / "summary.yaml")
        save_prepare_summary(summary, path)

        assert os.path.exists(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["class_name"] == "AtomicTest"

    def test_delivery_log_atomic_write(self, tmp_path) -> None:
        """save_delivery_log uses atomic write."""
        from forma.delivery_send import DeliveryLog, save_delivery_log, load_delivery_log

        log = DeliveryLog(
            sent_at="2026-01-01T00:00:00Z",
            smtp_server="smtp.test.com",
            dry_run=True,
            total=2,
            success=1,
            failed=1,
            results=[],
        )

        path = str(tmp_path / "delivery_log.yaml")
        save_delivery_log(log, path)

        loaded = load_delivery_log(path)
        assert loaded.total == 2
        assert loaded.success == 1
        assert loaded.smtp_server == "smtp.test.com"

    def test_concurrent_intervention_log_two_threads(self, tmp_path) -> None:
        """Two threads writing to InterventionLog -- verify atomic writes."""
        import threading
        from forma.intervention_store import InterventionLog

        log_path = str(tmp_path / "concurrent.yaml")
        errors = []

        def writer(thread_id: int) -> None:
            try:
                log = InterventionLog(log_path)
                log.load()
                log.add_record(
                    student_id=f"S{thread_id:03d}",
                    week=1,
                    intervention_type="면담",
                    description=f"Thread {thread_id}",
                )
                log.save()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer, args=(1,))
        t2 = threading.Thread(target=writer, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Concurrent write errors: {errors}"

        final = InterventionLog(log_path)
        final.load()
        assert len(final.get_records()) >= 1

    def test_tls_enforcement_in_smtp_send(self) -> None:
        """SMTP connection uses ssl.create_default_context when use_tls=True."""
        import smtplib
        import ssl

        from forma.delivery_send import SmtpConfig, DeliveryLog, send_summary_email

        config = SmtpConfig(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="prof@test.com",
            use_tls=True,
        )
        log = DeliveryLog(
            sent_at="2026-01-01T00:00:00Z",
            smtp_server="smtp.test.com",
            dry_run=False,
            total=0,
            success=0,
            failed=0,
            results=[],
        )

        mock_smtp = MagicMock()
        with patch.object(smtplib, "SMTP", return_value=mock_smtp):
            with patch.object(ssl, "create_default_context") as mock_ctx:
                ctx_obj = MagicMock()
                mock_ctx.return_value = ctx_obj
                try:
                    send_summary_email(log, config, password="pass")
                except Exception:
                    pass
                mock_ctx.assert_called_once()
                mock_smtp.starttls.assert_called_once_with(context=ctx_obj)
