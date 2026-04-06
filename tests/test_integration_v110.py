"""Integration tests for v0.11.0 email delivery pipeline.

T029: End-to-end integration tests covering:
  - prepare -> send full pipeline with mock SMTP
  - 5-student scenario: 3 ready, 1 warning, 1 error
  - prepare_summary.yaml correctness
  - delivery_log.yaml correctness
  - Dry-run -> real send transition (dry_run log does NOT block real send)
  - Retry-failed flow after partial failure
  - CLI end-to-end (prepare + send subcommands)
"""

from __future__ import annotations

import os
import textwrap

import pytest
import yaml


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STUDENTS = [
    {"student_id": "S001", "name": "홍길동", "email": "s001@u.kr"},
    {"student_id": "S002", "name": "김영희", "email": "s002@u.kr"},
    {"student_id": "S003", "name": "이철수", "email": "s003@u.kr"},
    {"student_id": "S004", "name": "박민수", "email": "s004@u.kr"},
    {"student_id": "S005", "name": "정유진", "email": "s005@u.kr"},
]


def _create_report_dir(tmp_path, students_with_files):
    """Create a report directory with PDF files for specified students.

    Args:
        tmp_path: Temporary directory path.
        students_with_files: Dict mapping student_id to list of file patterns
            to create (e.g., {"S001": ["report", "graph"]}).

    Returns:
        Path to the report directory.
    """
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    for sid, patterns in students_with_files.items():
        for pattern in patterns:
            filename = f"{sid}_{pattern}.pdf"
            (report_dir / filename).write_bytes(b"%PDF-1.4 fake content " + sid.encode())

    return str(report_dir)


def _create_manifest(tmp_path, report_dir):
    """Create a delivery manifest YAML.

    Returns:
        Path to the manifest file.
    """
    manifest = {
        "report_source": {
            "directory": report_dir,
            "file_patterns": [
                "{student_id}_report.pdf",
                "{student_id}_graph.pdf",
            ],
        }
    }
    path = tmp_path / "manifest.yaml"
    with open(str(path), "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, allow_unicode=True)
    return str(path)


def _create_roster(tmp_path, students=None):
    """Create a student roster YAML.

    Returns:
        Path to the roster file.
    """
    if students is None:
        students = _STUDENTS

    roster = {
        "class_name": "해부생리학 1A",
        "students": students,
    }
    path = tmp_path / "roster.yaml"
    with open(str(path), "w", encoding="utf-8") as f:
        yaml.dump(roster, f, allow_unicode=True)
    return str(path)


def _create_template(tmp_path):
    """Create an email template YAML.

    Returns:
        Path to the template file.
    """
    path = tmp_path / "template.yaml"
    path.write_text(
        textwrap.dedent("""\
            subject: "[해부생리학 1A] {student_name} 형성평가 결과"
            body: |
              {student_name}({student_id}) 학생에게,

              {class_name} 수업의 형성평가 결과를 첨부 파일로 보내드립니다.
              확인 후 궁금한 점이 있으면 연락 바랍니다.
        """),
        encoding="utf-8",
    )
    return str(path)


def _create_smtp_config(tmp_path):
    """Create an SMTP config YAML.

    Returns:
        Path to the SMTP config file.
    """
    path = tmp_path / "smtp.yaml"
    path.write_text(
        textwrap.dedent("""\
            smtp_server: "smtp.gmail.com"
            smtp_port: 587
            sender_email: "prof@univ.kr"
            sender_name: "담당교수"
            use_tls: true
            send_interval_sec: 0.0
        """),
        encoding="utf-8",
    )
    return str(path)


class MockSMTP:
    """Mock SMTP server for testing."""

    instances = []
    sent_messages = []

    def __init__(self, *args, **kwargs):
        MockSMTP.instances.append(self)

    def starttls(self, **kwargs):
        pass

    def login(self, user, password):
        pass

    def send_message(self, msg):
        MockSMTP.sent_messages.append(msg)

    def quit(self):
        pass

    @classmethod
    def reset(cls):
        cls.instances.clear()
        cls.sent_messages.clear()


# ---------------------------------------------------------------------------
# T029: E2E Integration — prepare then send
# ---------------------------------------------------------------------------


class TestE2EPrepareAndSend:
    """Full pipeline: prepare -> send with 5 students (3 ready, 1 warning, 1 error)."""

    def setup_method(self):
        MockSMTP.reset()

    def test_prepare_5_students_mixed_status(self, tmp_path):
        """Prepare with 5 students: 3 ready (both files), 1 warning (1 file), 1 error (no files)."""
        from forma.delivery_prepare import prepare_delivery

        # S001-S003: both report + graph files -> ready
        # S004: only report file -> warning (1 pattern missing)
        # S005: no files at all -> error
        students_with_files = {
            "S001": ["report", "graph"],
            "S002": ["report", "graph"],
            "S003": ["report", "graph"],
            "S004": ["report"],
            # S005: intentionally no files
        }

        report_dir = _create_report_dir(tmp_path, students_with_files)
        manifest_path = _create_manifest(tmp_path, report_dir)
        roster_path = _create_roster(tmp_path)
        output_dir = str(tmp_path / "staging")

        summary = prepare_delivery(manifest_path, roster_path, output_dir)

        assert summary.total_students == 5
        assert summary.ready == 3
        assert summary.warnings == 1
        assert summary.errors == 1

        # Verify per-student status
        by_id = {d.student_id: d for d in summary.details}
        assert by_id["S001"].status == "ready"
        assert by_id["S002"].status == "ready"
        assert by_id["S003"].status == "ready"
        assert by_id["S004"].status == "warning"
        assert by_id["S005"].status == "error"

        # Verify zips exist for ready and warning students
        for sid in ("S001", "S002", "S003", "S004"):
            assert by_id[sid].zip_path is not None
            assert os.path.exists(by_id[sid].zip_path)

        # Verify error student has no zip
        assert by_id["S005"].zip_path is None

        # Verify prepare_summary.yaml was written
        summary_file = os.path.join(output_dir, "prepare_summary.yaml")
        assert os.path.exists(summary_file)

    def test_full_pipeline_prepare_then_send(self, tmp_path, monkeypatch):
        """Full E2E: prepare -> send -> verify delivery_log."""
        from forma.delivery_prepare import prepare_delivery
        from forma.delivery_send import load_delivery_log, send_emails

        # 3 ready, 1 warning, 1 error
        students_with_files = {
            "S001": ["report", "graph"],
            "S002": ["report", "graph"],
            "S003": ["report", "graph"],
            "S004": ["report"],
        }

        report_dir = _create_report_dir(tmp_path, students_with_files)
        manifest_path = _create_manifest(tmp_path, report_dir)
        roster_path = _create_roster(tmp_path)
        staging_dir = str(tmp_path / "staging")
        template_path = _create_template(tmp_path)
        smtp_config_path = _create_smtp_config(tmp_path)

        # Step 1: Prepare
        summary = prepare_delivery(manifest_path, roster_path, staging_dir)
        assert summary.ready == 3
        assert summary.warnings == 1
        assert summary.errors == 1

        # Step 2: Send (mock SMTP)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_pw")
        monkeypatch.setattr("smtplib.SMTP", MockSMTP)

        log = send_emails(staging_dir, template_path, smtp_config_path)

        # 4 students should be sent (ready + warning), error skipped
        assert log.total == 4
        assert log.success == 4
        assert log.failed == 0
        assert log.dry_run is False

        # Verify SMTP was called 4 times
        assert len(MockSMTP.sent_messages) == 4

        # Verify delivery_log.yaml exists and is correct
        log_path = os.path.join(staging_dir, "delivery_log.yaml")
        loaded_log = load_delivery_log(log_path)
        assert loaded_log.total == 4
        assert loaded_log.success == 4

        # Verify each sent email has correct recipient
        recipients = {msg["To"] for msg in MockSMTP.sent_messages}
        assert recipients == {"s001@u.kr", "s002@u.kr", "s003@u.kr", "s004@u.kr"}

    def test_dry_run_then_real_send(self, tmp_path, monkeypatch):
        """Dry-run should NOT block subsequent real send."""
        from forma.delivery_prepare import prepare_delivery
        from forma.delivery_send import send_emails

        students_with_files = {
            "S001": ["report", "graph"],
            "S002": ["report", "graph"],
        }
        roster = [_STUDENTS[0], _STUDENTS[1]]

        report_dir = _create_report_dir(tmp_path, students_with_files)
        manifest_path = _create_manifest(tmp_path, report_dir)
        roster_path = _create_roster(tmp_path, students=roster)
        staging_dir = str(tmp_path / "staging")
        template_path = _create_template(tmp_path)
        smtp_config_path = _create_smtp_config(tmp_path)

        prepare_delivery(manifest_path, roster_path, staging_dir)

        # Step 1: Dry-run (no password needed)
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)
        dry_log = send_emails(
            staging_dir,
            template_path,
            smtp_config_path,
            dry_run=True,
        )
        assert dry_log.dry_run is True
        assert dry_log.success == 2

        # Step 2: Real send (should NOT be blocked by dry-run log)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_pw")
        monkeypatch.setattr("smtplib.SMTP", MockSMTP)

        real_log = send_emails(staging_dir, template_path, smtp_config_path)
        assert real_log.dry_run is False
        assert real_log.success == 2

    def test_partial_failure_then_retry(self, tmp_path, monkeypatch):
        """Partial failure -> retry-failed sends only to failed students."""
        from forma.delivery_prepare import prepare_delivery
        from forma.delivery_send import send_emails

        students_with_files = {
            "S001": ["report", "graph"],
            "S002": ["report", "graph"],
            "S003": ["report", "graph"],
        }
        roster = [_STUDENTS[0], _STUDENTS[1], _STUDENTS[2]]

        report_dir = _create_report_dir(tmp_path, students_with_files)
        manifest_path = _create_manifest(tmp_path, report_dir)
        roster_path = _create_roster(tmp_path, students=roster)
        staging_dir = str(tmp_path / "staging")
        template_path = _create_template(tmp_path)
        smtp_config_path = _create_smtp_config(tmp_path)

        prepare_delivery(manifest_path, roster_path, staging_dir)

        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_pw")

        # First send: S002 fails
        call_count = 0

        class FailSecondSMTP:
            def __init__(self, *a, **kw):
                pass

            def starttls(self, **kwargs):
                pass

            def login(self, u, p):
                pass

            def send_message(self, msg):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("SMTP error for second")

            def quit(self):
                pass

        monkeypatch.setattr("smtplib.SMTP", FailSecondSMTP)
        log1 = send_emails(staging_dir, template_path, smtp_config_path)

        assert log1.total == 3
        assert log1.success == 2
        assert log1.failed == 1

        # Retry: only the failed student
        monkeypatch.setattr("smtplib.SMTP", MockSMTP)
        log2 = send_emails(
            staging_dir,
            template_path,
            smtp_config_path,
            retry_failed=True,
        )

        assert log2.total == 1
        assert log2.success == 1
        assert log2.failed == 0

    def test_template_variable_substitution(self, tmp_path, monkeypatch):
        """Verify template variables are correctly substituted in sent emails."""
        from forma.delivery_prepare import prepare_delivery
        from forma.delivery_send import send_emails

        students_with_files = {"S001": ["report", "graph"]}
        roster = [_STUDENTS[0]]

        report_dir = _create_report_dir(tmp_path, students_with_files)
        manifest_path = _create_manifest(tmp_path, report_dir)
        roster_path = _create_roster(tmp_path, students=roster)
        staging_dir = str(tmp_path / "staging")
        template_path = _create_template(tmp_path)
        smtp_config_path = _create_smtp_config(tmp_path)

        prepare_delivery(manifest_path, roster_path, staging_dir)

        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_pw")
        monkeypatch.setattr("smtplib.SMTP", MockSMTP)

        send_emails(staging_dir, template_path, smtp_config_path)

        assert len(MockSMTP.sent_messages) == 1
        msg = MockSMTP.sent_messages[0]

        # Subject should contain student name
        assert "홍길동" in msg["Subject"]
        assert msg["To"] == "s001@u.kr"

        # From should contain sender name
        assert "담당교수" in msg["From"]

    def test_resend_prevention_and_force(self, tmp_path, monkeypatch):
        """After successful send, re-send is blocked; --force overrides."""
        from forma.delivery_prepare import prepare_delivery
        from forma.delivery_send import send_emails

        students_with_files = {"S001": ["report", "graph"]}
        roster = [_STUDENTS[0]]

        report_dir = _create_report_dir(tmp_path, students_with_files)
        manifest_path = _create_manifest(tmp_path, report_dir)
        roster_path = _create_roster(tmp_path, students=roster)
        staging_dir = str(tmp_path / "staging")
        template_path = _create_template(tmp_path)
        smtp_config_path = _create_smtp_config(tmp_path)

        prepare_delivery(manifest_path, roster_path, staging_dir)

        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_pw")
        monkeypatch.setattr("smtplib.SMTP", MockSMTP)

        # First send
        send_emails(staging_dir, template_path, smtp_config_path)

        # Re-send without force → should raise
        with pytest.raises((ValueError, FileExistsError)):
            send_emails(staging_dir, template_path, smtp_config_path)

        # Re-send with force → should succeed
        MockSMTP.reset()
        log = send_emails(staging_dir, template_path, smtp_config_path, force=True)
        assert log.success == 1


# ---------------------------------------------------------------------------
# T029: CLI Integration — prepare + send subcommands
# ---------------------------------------------------------------------------


class TestCliE2E:
    """CLI end-to-end tests: forma-deliver prepare + send."""

    def setup_method(self):
        MockSMTP.reset()

    def test_cli_prepare_then_send(self, tmp_path, monkeypatch, capsys):
        """CLI: prepare -> send -> verify exit code and output."""
        from forma.cli_deliver import main

        students_with_files = {
            "S001": ["report", "graph"],
            "S002": ["report", "graph"],
        }
        roster = [_STUDENTS[0], _STUDENTS[1]]

        report_dir = _create_report_dir(tmp_path, students_with_files)
        manifest_path = _create_manifest(tmp_path, report_dir)
        roster_path = _create_roster(tmp_path, students=roster)
        staging_dir = str(tmp_path / "staging")
        template_path = _create_template(tmp_path)
        smtp_config_path = _create_smtp_config(tmp_path)

        # CLI: prepare
        main(
            [
                "--no-config",
                "prepare",
                "--manifest",
                manifest_path,
                "--roster",
                roster_path,
                "--output-dir",
                staging_dir,
            ]
        )
        captured = capsys.readouterr()
        assert "2 students" in captured.out
        assert "ready=2" in captured.out

        # CLI: send (mock SMTP)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_pw")
        monkeypatch.setattr("smtplib.SMTP", MockSMTP)

        main(
            [
                "--no-config",
                "send",
                "--staged",
                staging_dir,
                "--template",
                template_path,
                "--smtp-config",
                smtp_config_path,
            ]
        )
        captured = capsys.readouterr()
        assert "2/2 succeeded" in captured.out

    def test_cli_dry_run(self, tmp_path, monkeypatch, capsys):
        """CLI: --dry-run produces [DRY-RUN] prefix in output."""
        from forma.cli_deliver import main

        students_with_files = {"S001": ["report", "graph"]}
        roster = [_STUDENTS[0]]

        report_dir = _create_report_dir(tmp_path, students_with_files)
        manifest_path = _create_manifest(tmp_path, report_dir)
        roster_path = _create_roster(tmp_path, students=roster)
        staging_dir = str(tmp_path / "staging")
        template_path = _create_template(tmp_path)
        smtp_config_path = _create_smtp_config(tmp_path)

        # Prepare
        main(
            [
                "--no-config",
                "prepare",
                "--manifest",
                manifest_path,
                "--roster",
                roster_path,
                "--output-dir",
                staging_dir,
            ]
        )

        # Send with dry-run (no password needed)
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)
        main(
            [
                "--no-config",
                "send",
                "--staged",
                staging_dir,
                "--template",
                template_path,
                "--smtp-config",
                smtp_config_path,
                "--dry-run",
            ]
        )
        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out


# ---------------------------------------------------------------------------
# T029: Completion summary integration
# ---------------------------------------------------------------------------


class TestSummaryIntegration:
    """Integration tests for print_delivery_summary and send_summary_email."""

    def setup_method(self):
        MockSMTP.reset()

    def test_print_summary_after_send(self, tmp_path, monkeypatch, capsys):
        """print_delivery_summary outputs correct counts after a real send."""
        from forma.delivery_prepare import prepare_delivery
        from forma.delivery_send import print_delivery_summary, send_emails

        students_with_files = {
            "S001": ["report", "graph"],
            "S002": ["report", "graph"],
            "S003": ["report", "graph"],
        }
        roster = [_STUDENTS[0], _STUDENTS[1], _STUDENTS[2]]

        report_dir = _create_report_dir(tmp_path, students_with_files)
        manifest_path = _create_manifest(tmp_path, report_dir)
        roster_path = _create_roster(tmp_path, students=roster)
        staging_dir = str(tmp_path / "staging")
        template_path = _create_template(tmp_path)
        smtp_config_path = _create_smtp_config(tmp_path)

        prepare_delivery(manifest_path, roster_path, staging_dir)

        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_pw")
        monkeypatch.setattr("smtplib.SMTP", MockSMTP)

        log = send_emails(staging_dir, template_path, smtp_config_path)
        print_delivery_summary(log)

        captured = capsys.readouterr()
        assert "Total 3" in captured.out
        assert "3 success" in captured.out
        assert "0 failed" in captured.out

    def test_send_summary_email_after_send(self, tmp_path, monkeypatch):
        """send_summary_email sends to professor after real send."""
        from forma.delivery_prepare import prepare_delivery
        from forma.delivery_send import (
            SmtpConfig,
            send_emails,
            send_summary_email,
        )

        students_with_files = {"S001": ["report", "graph"]}
        roster = [_STUDENTS[0]]

        report_dir = _create_report_dir(tmp_path, students_with_files)
        manifest_path = _create_manifest(tmp_path, report_dir)
        roster_path = _create_roster(tmp_path, students=roster)
        staging_dir = str(tmp_path / "staging")
        template_path = _create_template(tmp_path)
        smtp_config_path = _create_smtp_config(tmp_path)

        prepare_delivery(manifest_path, roster_path, staging_dir)

        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_pw")
        monkeypatch.setattr("smtplib.SMTP", MockSMTP)

        log = send_emails(staging_dir, template_path, smtp_config_path)

        # Reset to capture summary email separately
        MockSMTP.reset()

        cfg = SmtpConfig(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="prof@univ.kr",
            sender_name="담당교수",
        )
        send_summary_email(log, cfg, password="test_pw")

        assert len(MockSMTP.sent_messages) == 1
        summary_msg = MockSMTP.sent_messages[0]
        assert summary_msg["To"] == "prof@univ.kr"
        assert "1" in summary_msg["Subject"]
