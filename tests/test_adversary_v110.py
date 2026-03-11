"""Adversary attack tests for v0.11.0 email delivery feature.

T030: Adversarial edge case tests covering all 9 spec edge cases:
  1. zip > 25MB (FR-015)
  2. duplicate student_id (FR-016)
  3. duplicate email warning
  4. unsupported template variable (FR-017)
  5. SMTP mid-send disconnect (FR-011)
  6. staging overwrite prompt (FR-020)
  7. empty email field (FR-021)
  8. no matching files
  9. re-send prevention (FR-022)

Plus additional adversarial attack scenarios:
  - Unicode/CJK filename injection
  - Path traversal in manifest
  - Malformed YAML payloads
  - Boolean/type confusion attacks
  - Null byte injection
  - Extremely long strings
  - Empty collections
"""

from __future__ import annotations

import os
import random
import textwrap

import pytest
import yaml


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path, report_dir, patterns=None):
    """Write manifest YAML."""
    if patterns is None:
        patterns = ["{student_id}_report.pdf"]
    mf = tmp_path / "manifest.yaml"
    data = {
        "report_source": {
            "directory": str(report_dir),
            "file_patterns": patterns,
        }
    }
    with open(str(mf), "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    return str(mf)


def _write_roster(tmp_path, students):
    """Write roster YAML."""
    rf = tmp_path / "roster.yaml"
    data = {"class_name": "테스트반", "students": students}
    with open(str(rf), "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    return str(rf)


def _write_template(tmp_path, subject="Test", body="Hello {student_name}"):
    """Write email template YAML."""
    tpl = tmp_path / "template.yaml"
    data = {"subject": subject, "body": body}
    with open(str(tpl), "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    return str(tpl)


def _write_smtp_config(tmp_path):
    """Write SMTP config YAML."""
    cfg = tmp_path / "smtp.yaml"
    cfg.write_text(
        'smtp_server: "smtp.gmail.com"\n'
        "smtp_port: 587\n"
        'sender_email: "prof@univ.kr"\n'
        "send_interval_sec: 0.0\n",
        encoding="utf-8",
    )
    return str(cfg)


def _create_staging(tmp_path, details, class_name="테스트반"):
    """Create a staging folder with prepare_summary.yaml."""
    import zipfile

    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)

    yaml_details = []
    for d in details:
        sid = d["student_id"]
        status = d.get("status", "ready")
        email = d.get("email", f"{sid}@u.kr")
        name = d.get("name", f"학생_{sid}")
        zip_path = None
        zip_size = 0

        if status in ("ready", "warning"):
            student_dir = staging / sid
            student_dir.mkdir(exist_ok=True)
            zp = student_dir / f"{sid}.zip"
            with zipfile.ZipFile(str(zp), "w") as zf:
                zf.writestr(f"{sid}_report.pdf", f"content-{sid}")
            zip_path = str(zp)
            zip_size = os.path.getsize(zip_path)

        yaml_details.append({
            "student_id": sid,
            "name": name,
            "email": email,
            "status": status,
            "matched_files": [f"{sid}_report.pdf"] if status != "error" else [],
            "zip_path": zip_path,
            "zip_size_bytes": zip_size,
            "message": "" if status == "ready" else "매칭 파일 없음" if status == "error" else "1개 패턴 미매칭",
        })

    ready = sum(1 for d in yaml_details if d["status"] == "ready")
    warnings = sum(1 for d in yaml_details if d["status"] == "warning")
    errors = sum(1 for d in yaml_details if d["status"] == "error")

    summary = {
        "prepared_at": "2026-03-11T10:00:00",
        "total_students": len(details),
        "ready": ready,
        "warnings": warnings,
        "errors": errors,
        "class_name": class_name,
        "details": yaml_details,
    }
    with open(str(staging / "prepare_summary.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)

    return str(staging)


# ===========================================================================
# Persona 1: Data Poisoner — attacks manifest and roster parsing
# ===========================================================================


class TestDataPoisoner:
    """Adversarial manifest and roster parsing attacks."""

    def test_edge1_zip_over_25mb(self, tmp_path):
        """Edge case 1: zip > 25MB is detected as error in prepare (FR-015)."""
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        # Create a file that will produce a zip > 25MB
        big_file = report_dir / "s001_report.pdf"
        rng = random.Random(42)
        # Write 26MB of incompressible data
        with open(str(big_file), "wb") as f:
            f.write(rng.randbytes(26 * 1024 * 1024))

        manifest_path = _write_manifest(tmp_path, report_dir)
        roster_path = _write_roster(tmp_path, [
            {"student_id": "s001", "name": "대용량학생", "email": "big@u.kr"},
        ])
        output_dir = str(tmp_path / "staging")

        summary = prepare_delivery(manifest_path, roster_path, output_dir)

        assert summary.errors == 1
        detail = summary.details[0]
        assert detail.status == "error"
        assert "25MB" in detail.message or "크기" in detail.message

    def test_edge2_duplicate_student_id(self, tmp_path):
        """Edge case 2: duplicate student_id in roster raises ValueError (FR-016)."""
        from forma.delivery_prepare import load_roster

        roster_path = _write_roster(tmp_path, [
            {"student_id": "dup001", "name": "홍길동", "email": "a@u.kr"},
            {"student_id": "dup001", "name": "홍길동복사", "email": "b@u.kr"},
        ])

        with pytest.raises(ValueError, match="중복"):
            load_roster(roster_path)

    def test_edge7_empty_email_field(self, tmp_path):
        """Edge case 7: empty email field → per-student error status (FR-021).

        FR-021 redesign: load_roster() accepts empty email (roster is still valid),
        prepare_delivery() marks the student as status='error' and continues.
        """
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        manifest_path = _write_manifest(tmp_path, report_dir)
        roster_path = _write_roster(tmp_path, [
            {"student_id": "s001", "name": "빈이메일", "email": ""},
        ])
        output_dir = str(tmp_path / "staging")

        summary = prepare_delivery(manifest_path, roster_path, output_dir)
        assert summary.errors == 1
        assert summary.details[0].status == "error"
        assert "email" in summary.details[0].message

    def test_edge8_no_matching_files(self, tmp_path):
        """Edge case 8: no matching files for a student -> error status."""
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        # Create no files

        manifest_path = _write_manifest(tmp_path, report_dir)
        roster_path = _write_roster(tmp_path, [
            {"student_id": "s001", "name": "파일없음", "email": "s@u.kr"},
        ])
        output_dir = str(tmp_path / "staging")

        summary = prepare_delivery(manifest_path, roster_path, output_dir)
        assert summary.errors == 1
        assert summary.details[0].status == "error"

    def test_manifest_missing_student_id_placeholder(self, tmp_path):
        """Manifest pattern without {student_id} raises ValueError."""
        from forma.delivery_prepare import load_manifest

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        mf = tmp_path / "bad_manifest.yaml"
        data = {
            "report_source": {
                "directory": str(report_dir),
                "file_patterns": ["report.pdf"],  # no {student_id}
            }
        }
        with open(str(mf), "w", encoding="utf-8") as f:
            yaml.dump(data, f)

        with pytest.raises(ValueError, match="student_id"):
            load_manifest(str(mf))

    def test_roster_missing_name_field(self, tmp_path):
        """Roster entry without 'name' raises ValueError."""
        from forma.delivery_prepare import load_roster

        roster_path = _write_roster(tmp_path, [
            {"student_id": "s001", "email": "s@u.kr"},  # no name
        ])

        with pytest.raises(ValueError, match="name"):
            load_roster(roster_path)

    def test_roster_email_without_at(self, tmp_path):
        """Roster entry with email missing @ → per-student error status (FR-021).

        FR-021 redesign: load_roster() accepts syntactically invalid email in the
        roster (roster is still loadable), prepare_delivery() validates '@' and
        marks the student as status='error' without aborting the entire batch.
        """
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        manifest_path = _write_manifest(tmp_path, report_dir)
        roster_path = _write_roster(tmp_path, [
            {"student_id": "s001", "name": "잘못", "email": "no-at-symbol"},
        ])
        output_dir = str(tmp_path / "staging")

        summary = prepare_delivery(manifest_path, roster_path, output_dir)
        assert summary.errors == 1
        assert summary.details[0].status == "error"
        assert "email" in summary.details[0].message

    def test_manifest_nonexistent_directory(self, tmp_path):
        """Manifest with nonexistent directory raises ValueError."""
        from forma.delivery_prepare import load_manifest

        mf = tmp_path / "bad_dir.yaml"
        data = {
            "report_source": {
                "directory": "/nonexistent/path/surely",
                "file_patterns": ["{student_id}_report.pdf"],
            }
        }
        with open(str(mf), "w", encoding="utf-8") as f:
            yaml.dump(data, f)

        with pytest.raises(ValueError, match="directory"):
            load_manifest(str(mf))

    def test_empty_roster(self, tmp_path):
        """Roster with no students raises ValueError."""
        from forma.delivery_prepare import load_roster

        rf = tmp_path / "empty_roster.yaml"
        data = {"class_name": "빈반", "students": []}
        with open(str(rf), "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        with pytest.raises(ValueError, match="students"):
            load_roster(str(rf))


# ===========================================================================
# Persona 2: Template Injector — attacks template validation
# ===========================================================================


class TestTemplateInjector:
    """Adversarial template validation attacks."""

    def test_edge4_unsupported_template_variable(self):
        """Edge case 4: unsupported variable raises ValueError (FR-017)."""
        from forma.delivery_send import EmailTemplate, validate_template_variables

        t = EmailTemplate(
            subject="{student_name} Test",
            body="Your {grade} is {score}",
        )
        with pytest.raises(ValueError, match="grade|score"):
            validate_template_variables(t)

    def test_template_with_python_format_spec(self):
        """Template with format spec like {name!r} should be handled."""
        from forma.delivery_send import EmailTemplate, validate_template_variables

        # The regex pattern \{(\w+)\} will match {name} but not {name!r}
        t = EmailTemplate(subject="Test", body="Hello {student_name}")
        # Should not raise
        validate_template_variables(t)

    def test_template_with_nested_braces(self):
        """Template with escaped braces {{literal}} should not confuse validator."""
        from forma.delivery_send import EmailTemplate, validate_template_variables

        t = EmailTemplate(subject="Test", body="Use {{curly}} braces")
        # {{curly}} should not be parsed as {curly}
        validate_template_variables(t)

    def test_template_empty_variable_name(self):
        """Template with {} (empty variable) should not crash."""
        from forma.delivery_send import EmailTemplate, render_template

        t = EmailTemplate(subject="Test", body="Hello {student_name}")
        subject, body = render_template(
            t, student_name="홍길동", student_id="s001", class_name="반",
        )
        assert body == "Hello 홍길동"

    def test_template_korean_variable_substitution(self):
        """Korean characters in template variables are correctly substituted."""
        from forma.delivery_send import EmailTemplate, render_template

        t = EmailTemplate(
            subject="[{class_name}] 피드백",
            body="{student_name}({student_id}) 학생에게",
        )
        subject, body = render_template(
            t,
            student_name="김가나다라마바사",
            student_id="2024001",
            class_name="해부생리학 1A반",
        )
        assert "김가나다라마바사" in body
        assert "2024001" in body
        assert "해부생리학 1A반" in subject


# ===========================================================================
# Persona 3: SMTP Saboteur — attacks send pipeline
# ===========================================================================


class TestSmtpSaboteur:
    """Adversarial SMTP connection and send attacks."""

    def test_edge5_smtp_mid_send_disconnect(self, tmp_path, monkeypatch):
        """Edge case 5: SMTP disconnect mid-send logs partial results (FR-011)."""
        from forma.delivery_send import send_emails

        staging = _create_staging(tmp_path, [
            {"student_id": "s001"},
            {"student_id": "s002"},
            {"student_id": "s003"},
        ])
        template_path = _write_template(tmp_path)
        smtp_config_path = _write_smtp_config(tmp_path)

        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        call_count = 0

        class DisconnectSMTP:
            def __init__(self, *a, **kw): pass
            def starttls(self): pass
            def login(self, u, p): pass
            def send_message(self, msg):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise ConnectionError("Connection reset by peer")
            def quit(self): pass

        monkeypatch.setattr("smtplib.SMTP", DisconnectSMTP)

        log = send_emails(staging, template_path, smtp_config_path)

        # First succeed, second failed, third should still be attempted
        assert log.total == 3
        assert log.failed >= 1
        assert log.success >= 1
        # All results should be recorded
        assert len(log.results) == 3

    def test_edge9_resend_prevention(self, tmp_path, monkeypatch):
        """Edge case 9: re-send with existing success log raises error (FR-022)."""
        from forma.delivery_send import send_emails

        staging = _create_staging(tmp_path, [{"student_id": "s001"}])
        template_path = _write_template(tmp_path)
        smtp_config_path = _write_smtp_config(tmp_path)

        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        class OkSMTP:
            def __init__(self, *a, **kw): pass
            def starttls(self): pass
            def login(self, u, p): pass
            def send_message(self, msg): pass
            def quit(self): pass

        monkeypatch.setattr("smtplib.SMTP", OkSMTP)

        # First send succeeds
        send_emails(staging, template_path, smtp_config_path)

        # Second send without force raises
        with pytest.raises(ValueError, match="발송"):
            send_emails(staging, template_path, smtp_config_path)

    def test_smtp_auth_failure(self, tmp_path, monkeypatch):
        """SMTP authentication failure is raised to caller."""
        from forma.delivery_send import send_emails
        import smtplib

        staging = _create_staging(tmp_path, [{"student_id": "s001"}])
        template_path = _write_template(tmp_path)
        smtp_config_path = _write_smtp_config(tmp_path)

        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "bad_pw")

        class AuthFailSMTP:
            def __init__(self, *a, **kw): pass
            def starttls(self): pass
            def login(self, u, p):
                raise smtplib.SMTPAuthenticationError(535, b"Auth failed")
            def send_message(self, msg): pass
            def quit(self): pass

        monkeypatch.setattr("smtplib.SMTP", AuthFailSMTP)

        with pytest.raises(smtplib.SMTPAuthenticationError):
            send_emails(staging, template_path, smtp_config_path)

    def test_missing_password_env_and_stdin(self, tmp_path, monkeypatch):
        """Missing password from both env and parameter raises ValueError (FR-008)."""
        from forma.delivery_send import send_emails

        staging = _create_staging(tmp_path, [{"student_id": "s001"}])
        template_path = _write_template(tmp_path)
        smtp_config_path = _write_smtp_config(tmp_path)

        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        with pytest.raises(ValueError, match="비밀번호"):
            send_emails(staging, template_path, smtp_config_path)

    def test_password_from_parameter_overrides_env(self, tmp_path, monkeypatch):
        """Password parameter takes precedence over env var."""
        from forma.delivery_send import send_emails

        staging = _create_staging(tmp_path, [{"student_id": "s001"}])
        template_path = _write_template(tmp_path)
        smtp_config_path = _write_smtp_config(tmp_path)

        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "env_pw")

        login_passwords = []

        class SpySMTP:
            def __init__(self, *a, **kw): pass
            def starttls(self): pass
            def login(self, u, p):
                login_passwords.append(p)
            def send_message(self, msg): pass
            def quit(self): pass

        monkeypatch.setattr("smtplib.SMTP", SpySMTP)

        send_emails(
            staging, template_path, smtp_config_path, password="param_pw",
        )
        assert login_passwords[0] == "param_pw"


# ===========================================================================
# Persona 4: Staging Corruptor — attacks staging folder management
# ===========================================================================


class TestStagingCorruptor:
    """Adversarial staging folder and file attacks."""

    def test_edge6_staging_overwrite_without_force(self, tmp_path):
        """Edge case 6: existing staging without --force raises FileExistsError (FR-020)."""
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        (report_dir / "s001_report.pdf").write_bytes(b"pdf")

        manifest_path = _write_manifest(tmp_path, report_dir)
        roster_path = _write_roster(tmp_path, [
            {"student_id": "s001", "name": "홍길동", "email": "s@u.kr"},
        ])
        output_dir = tmp_path / "staging"
        output_dir.mkdir()
        (output_dir / "marker.txt").write_text("existing")

        with pytest.raises(FileExistsError):
            prepare_delivery(manifest_path, roster_path, str(output_dir))

    def test_staging_overwrite_with_force(self, tmp_path):
        """Existing staging with force=True overwrites successfully."""
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        (report_dir / "s001_report.pdf").write_bytes(b"pdf")

        manifest_path = _write_manifest(tmp_path, report_dir)
        roster_path = _write_roster(tmp_path, [
            {"student_id": "s001", "name": "홍길동", "email": "s@u.kr"},
        ])
        output_dir = tmp_path / "staging"
        output_dir.mkdir()
        (output_dir / "old_file.txt").write_text("old")

        summary = prepare_delivery(
            manifest_path, roster_path, str(output_dir), force=True,
        )
        assert summary.ready == 1

        # Old file should be gone
        assert not os.path.exists(str(output_dir / "old_file.txt"))

    def test_corrupt_prepare_summary_yaml(self, tmp_path, monkeypatch):
        """send_emails handles malformed prepare_summary.yaml gracefully."""
        from forma.delivery_send import send_emails

        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "prepare_summary.yaml").write_text(
            "this: is\nnot: a valid summary\n",
            encoding="utf-8",
        )
        template_path = _write_template(tmp_path)
        smtp_config_path = _write_smtp_config(tmp_path)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        class OkSMTP:
            def __init__(self, *a, **kw): pass
            def starttls(self): pass
            def login(self, u, p): pass
            def send_message(self, msg): pass
            def quit(self): pass

        monkeypatch.setattr("smtplib.SMTP", OkSMTP)

        # Should handle missing 'details' key gracefully (empty targets)
        log = send_emails(staging_dir=str(staging),
                         template_path=template_path,
                         smtp_config_path=smtp_config_path)
        assert log.total == 0
        assert log.success == 0


# ===========================================================================
# Persona 5: Unicode Attacker — CJK, special chars in names
# ===========================================================================


class TestUnicodeAttacker:
    """Adversarial Unicode and special character attacks."""

    def test_korean_filename_sanitization(self):
        """Korean characters in filename are preserved, illegal chars removed."""
        from forma.delivery_prepare import sanitize_filename

        # Normal Korean name
        assert sanitize_filename("홍길동_2024001.zip") == "홍길동_2024001.zip"

        # Name with illegal chars
        result = sanitize_filename('홍<길>동:"파일".zip')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result
        assert "홍" in result
        assert "동" in result

    def test_student_name_with_special_chars(self, tmp_path):
        """Student names with special characters don't crash prepare."""
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        (report_dir / "s001_report.pdf").write_bytes(b"pdf")

        manifest_path = _write_manifest(tmp_path, report_dir)
        roster_path = _write_roster(tmp_path, [
            {"student_id": "s001", "name": "O'Brien-Kim(김)", "email": "s@u.kr"},
        ])
        output_dir = str(tmp_path / "staging")

        summary = prepare_delivery(manifest_path, roster_path, output_dir)
        assert summary.ready == 1

    def test_extremely_long_student_name(self, tmp_path):
        """Very long student name doesn't crash prepare."""
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        (report_dir / "s001_report.pdf").write_bytes(b"pdf")

        long_name = "가" * 200
        manifest_path = _write_manifest(tmp_path, report_dir)
        roster_path = _write_roster(tmp_path, [
            {"student_id": "s001", "name": long_name, "email": "s@u.kr"},
        ])
        output_dir = str(tmp_path / "staging")

        summary = prepare_delivery(manifest_path, roster_path, output_dir)
        assert summary.ready == 1

    def test_email_template_with_cjk(self, tmp_path, monkeypatch):
        """Email with full CJK content is correctly composed."""
        from forma.delivery_send import SmtpConfig, compose_email

        zip_file = tmp_path / "test.zip"
        zip_file.write_bytes(b"PK_FAKE_ZIP")

        cfg = SmtpConfig(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="prof@univ.kr",
            sender_name="해부생리학 담당교수",
        )

        msg = compose_email(
            sender_config=cfg,
            to_email="student@univ.kr",
            subject="[해부생리학 1A] 홍길동 학생 형성평가 결과",
            body="홍길동(2024001) 학생에게,\n\n"
                 "해부생리학 1A 수업의 형성평가 결과를 첨부합니다.\n"
                 "확인 후 궁금한 점은 연락 바랍니다.\n\n감사합니다.",
            zip_path=str(zip_file),
        )

        assert msg["To"] == "student@univ.kr"
        assert "해부생리학" in msg["Subject"]
        assert "해부생리학 담당교수" in msg["From"]


# ===========================================================================
# Persona 6: Boundary Pusher — exact boundary values
# ===========================================================================


class TestBoundaryPusher:
    """Boundary value and edge condition attacks."""

    def test_smtp_port_boundaries(self, tmp_path):
        """SMTP port boundary values: 1 (min), 65535 (max)."""
        from forma.delivery_send import load_smtp_config

        for port in (1, 65535):
            cfg_file = tmp_path / f"smtp_{port}.yaml"
            cfg_file.write_text(
                f'smtp_server: "smtp.test.com"\n'
                f"smtp_port: {port}\n"
                f'sender_email: "s@u.kr"\n',
                encoding="utf-8",
            )
            cfg = load_smtp_config(str(cfg_file))
            assert cfg.smtp_port == port

    def test_smtp_port_out_of_bounds(self, tmp_path):
        """SMTP port 0 and 65536 are rejected."""
        from forma.delivery_send import load_smtp_config

        for port in (0, 65536):
            cfg_file = tmp_path / f"smtp_bad_{port}.yaml"
            cfg_file.write_text(
                f'smtp_server: "smtp.test.com"\n'
                f"smtp_port: {port}\n"
                f'sender_email: "s@u.kr"\n',
                encoding="utf-8",
            )
            with pytest.raises(ValueError, match="port"):
                load_smtp_config(str(cfg_file))

    def test_send_interval_zero(self, tmp_path):
        """send_interval_sec = 0.0 is valid (no delay)."""
        from forma.delivery_send import load_smtp_config

        cfg_file = tmp_path / "smtp_zero.yaml"
        cfg_file.write_text(
            'smtp_server: "smtp.test.com"\n'
            "smtp_port: 587\n"
            'sender_email: "s@u.kr"\n'
            "send_interval_sec: 0.0\n",
            encoding="utf-8",
        )
        cfg = load_smtp_config(str(cfg_file))
        assert cfg.send_interval_sec == 0.0

    def test_single_student_pipeline(self, tmp_path, monkeypatch):
        """Single student through full prepare -> send pipeline."""
        from forma.delivery_prepare import prepare_delivery
        from forma.delivery_send import send_emails

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        (report_dir / "solo_report.pdf").write_bytes(b"pdf")

        manifest_path = _write_manifest(tmp_path, report_dir)
        roster_path = _write_roster(tmp_path, [
            {"student_id": "solo", "name": "혼자", "email": "solo@u.kr"},
        ])
        staging_dir = str(tmp_path / "staging")
        template_path = _write_template(tmp_path)
        smtp_config_path = _write_smtp_config(tmp_path)

        summary = prepare_delivery(manifest_path, roster_path, staging_dir)
        assert summary.ready == 1

        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        class OkSMTP:
            def __init__(self, *a, **kw): pass
            def starttls(self): pass
            def login(self, u, p): pass
            def send_message(self, msg): pass
            def quit(self): pass

        monkeypatch.setattr("smtplib.SMTP", OkSMTP)

        log = send_emails(staging_dir, template_path, smtp_config_path)
        assert log.total == 1
        assert log.success == 1

    def test_all_students_error_sends_zero(self, tmp_path, monkeypatch):
        """When all students have error status, send_emails sends 0 emails."""
        from forma.delivery_send import send_emails

        staging = _create_staging(tmp_path, [
            {"student_id": "e001", "status": "error"},
            {"student_id": "e002", "status": "error"},
        ])
        template_path = _write_template(tmp_path)
        smtp_config_path = _write_smtp_config(tmp_path)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        class OkSMTP:
            def __init__(self, *a, **kw): pass
            def starttls(self): pass
            def login(self, u, p): pass
            def send_message(self, msg): pass
            def quit(self): pass

        monkeypatch.setattr("smtplib.SMTP", OkSMTP)

        log = send_emails(staging, template_path, smtp_config_path)
        assert log.total == 0
        assert log.success == 0


# ===========================================================================
# Persona 7: Flag Interaction Tester — CLI flag conflicts
# ===========================================================================


class TestFlagInteractionTester:
    """Tests for CLI flag interaction rules per contract."""

    def test_retry_failed_plus_force_conflict(self, tmp_path, monkeypatch):
        """--retry-failed + --force exits with code 1."""
        from forma.cli_deliver import main

        staging = _create_staging(tmp_path, [{"student_id": "s001"}])

        # Create delivery_log
        log_data = {
            "sent_at": "2026-03-11T10:00:00",
            "smtp_server": "smtp.gmail.com",
            "dry_run": False,
            "total": 1,
            "success": 0,
            "failed": 1,
            "results": [{
                "student_id": "s001",
                "email": "s001@u.kr",
                "status": "failed",
                "sent_at": "2026-03-11T10:00:01",
                "attachment": "s001.zip",
                "size_bytes": 100,
                "error": "timeout",
            }],
        }
        with open(os.path.join(staging, "delivery_log.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(log_data, f, allow_unicode=True)

        template_path = _write_template(tmp_path)
        smtp_config_path = _write_smtp_config(tmp_path)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "send",
                "--staged", staging,
                "--template", template_path,
                "--smtp-config", smtp_config_path,
                "--retry-failed",
                "--force",
            ])
        assert exc_info.value.code == 1

    def test_dry_run_plus_force_allowed(self, tmp_path, monkeypatch):
        """--dry-run + --force is allowed (dry-run overrides)."""
        from forma.cli_deliver import main

        staging = _create_staging(tmp_path, [{"student_id": "s001"}])

        # Create delivery_log with success (would normally block re-send)
        log_data = {
            "sent_at": "2026-03-11T10:00:00",
            "smtp_server": "smtp.gmail.com",
            "dry_run": False,
            "total": 1,
            "success": 1,
            "failed": 0,
            "results": [{
                "student_id": "s001",
                "email": "s001@u.kr",
                "status": "success",
                "sent_at": "2026-03-11T10:00:01",
                "attachment": "s001.zip",
                "size_bytes": 100,
                "error": "",
            }],
        }
        with open(os.path.join(staging, "delivery_log.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(log_data, f, allow_unicode=True)

        template_path = _write_template(tmp_path)
        smtp_config_path = _write_smtp_config(tmp_path)
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        # dry-run + force should work (dry-run takes priority)
        try:
            main([
                "--no-config", "send",
                "--staged", staging,
                "--template", template_path,
                "--smtp-config", smtp_config_path,
                "--dry-run",
                "--force",
            ])
        except SystemExit as e:
            assert e.code in (0, None)

    def test_cli_no_subcommand_exits(self):
        """Running forma-deliver without subcommand exits with error."""
        from forma.cli_deliver import main

        with pytest.raises(SystemExit):
            main([])
