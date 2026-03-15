"""Integration tests for v0.11.1 credential consolidation.

T021: End-to-end integration tests covering:
  - _build_smtp_config() with field_map round-trip
  - get_smtp_config() from forma.json dict
  - send_emails() with pre-built SmtpConfig (bypass file load)
  - CLI forma.json fallback flow (prepare -> send without --smtp-config)
  - CLI --smtp-config deprecation warning path
  - CLI error path when no SMTP source available
  - Mixed scenario: both forma.json and --smtp-config -> explicit wins
  - forma.json with invalid smtp -> validation error
  - CI workflow file validation (US3)
  - pyproject.toml validation (US4)
  - config.py JSON_FIELD_MAP and resolution order validation
"""

from __future__ import annotations

import json
import os
import textwrap
import warnings
import zipfile

import pytest
import yaml


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _create_staged_dir(tmp_path, n_students=2):
    """Create a staging folder with prepare_summary and student zips."""
    staged = tmp_path / "staged"
    staged.mkdir()

    details = []
    for i in range(n_students):
        sid = f"s{i:03d}"
        name = f"Student{i}"
        email = f"s{i:03d}@u.kr"

        student_dir = staged / f"{sid}_{name}"
        student_dir.mkdir()
        zip_path = student_dir / f"{name}_{sid}.zip"
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr(f"{sid}_report.pdf", f"PDF content for {sid}")

        details.append({
            "student_id": sid,
            "name": name,
            "email": email,
            "status": "ready",
            "matched_files": [f"{sid}_report.pdf"],
            "zip_path": str(zip_path),
            "zip_size_bytes": os.path.getsize(str(zip_path)),
            "message": "",
        })

    summary = {
        "prepared_at": "2026-03-11T12:00:00+09:00",
        "total_students": n_students,
        "ready": n_students,
        "warnings": 0,
        "errors": 0,
        "class_name": "IntegrationTest",
        "details": details,
    }
    with open(str(staged / "prepare_summary.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)

    return str(staged)


def _write_template(tmp_path):
    """Write a simple email template."""
    tpl = tmp_path / "template.yaml"
    tpl.write_text(textwrap.dedent("""\
        subject: "[Integration] Report for {student_name}"
        body: |
          Dear {student_name},
          Your {class_name} report is attached.
    """), encoding="utf-8")
    return str(tpl)


def _write_smtp_yaml(tmp_path, server="yaml.smtp.com"):
    """Write a YAML-format SMTP config."""
    cfg = tmp_path / "smtp.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        smtp_server: "{server}"
        smtp_port: 587
        sender_email: "yaml@test.com"
        sender_name: "YAML Sender"
        use_tls: true
        send_interval_sec: 0
    """), encoding="utf-8")
    return str(cfg)


class MockSMTP:
    """Mock SMTP server for integration tests."""

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


# ===========================================================================
# IT-01: _build_smtp_config round-trip with field_map
# ===========================================================================


class TestBuildSmtpConfigIntegration:
    """Integration: _build_smtp_config with identity and custom field maps."""

    def test_identity_map_then_field_map_produce_same_config(self):
        """Same logical data via YAML (identity) and JSON (field_map) are equal."""
        from forma.delivery_send import SmtpConfig, _build_smtp_config

        yaml_data = {
            "smtp_server": "smtp.example.com",
            "smtp_port": 465,
            "sender_email": "test@example.com",
            "sender_name": "Prof",
            "use_tls": False,
            "send_interval_sec": 0.5,
        }
        json_data = {
            "server": "smtp.example.com",
            "port": 465,
            "sender_email": "test@example.com",
            "sender_name": "Prof",
            "use_tls": False,
            "send_interval_sec": 0.5,
        }
        from forma.config import JSON_FIELD_MAP

        cfg_yaml = _build_smtp_config(yaml_data)
        cfg_json = _build_smtp_config(json_data, field_map=JSON_FIELD_MAP)

        assert cfg_yaml == cfg_json
        assert isinstance(cfg_yaml, SmtpConfig)


# ===========================================================================
# IT-02: get_smtp_config from realistic forma.json
# ===========================================================================


class TestGetSmtpConfigIntegration:
    """Integration: get_smtp_config from full forma.json-like config."""

    def test_realistic_forma_json_config(self):
        """Realistic forma.json with llm + smtp sections parses correctly."""
        from forma.config import get_smtp_config

        config = {
            "llm": {"provider": "gemini", "api_key": "sk-xxx"},
            "naver_ocr": {"secret_key": "abc", "api_url": "https://ocr.example.com"},
            "smtp": {
                "server": "smtp.gmail.com",
                "port": 587,
                "sender_email": "professor@university.edu",
                "sender_name": "Prof. Kim",
                "use_tls": True,
                "send_interval_sec": 1.0,
            },
        }
        cfg = get_smtp_config(config)
        assert cfg.smtp_server == "smtp.gmail.com"
        assert cfg.smtp_port == 587
        assert cfg.sender_email == "professor@university.edu"
        assert cfg.sender_name == "Prof. Kim"
        assert cfg.use_tls is True
        assert cfg.send_interval_sec == 1.0


# ===========================================================================
# IT-03: send_emails with pre-built SmtpConfig
# ===========================================================================


class TestSendEmailsWithSmtpConfig:
    """Integration: send_emails() using smtp_config kwarg."""

    def test_dry_run_with_prebuilt_config(self, tmp_path):
        """send_emails with smtp_config kwarg in dry-run produces correct log."""
        from forma.delivery_send import SmtpConfig, send_emails

        staged = _create_staged_dir(tmp_path, n_students=3)
        tpl = _write_template(tmp_path)

        cfg = SmtpConfig(
            smtp_server="prebuilt.smtp.com",
            smtp_port=465,
            sender_email="pre@built.com",
            sender_name="Prebuilt",
            use_tls=True,
            send_interval_sec=0,
        )

        log = send_emails(
            staging_dir=staged,
            template_path=tpl,
            smtp_config_path="",
            dry_run=True,
            smtp_config=cfg,
        )

        assert log.smtp_server == "prebuilt.smtp.com"
        assert log.total == 3
        assert log.success == 3
        assert log.failed == 0
        assert log.dry_run is True

    def test_real_send_with_prebuilt_config(self, tmp_path, monkeypatch):
        """send_emails with smtp_config kwarg in real mode uses mock SMTP."""
        import unittest.mock
        from forma.delivery_send import SmtpConfig, send_emails

        staged = _create_staged_dir(tmp_path, n_students=2)
        tpl = _write_template(tmp_path)

        cfg = SmtpConfig(
            smtp_server="mock.smtp.com",
            smtp_port=587,
            sender_email="mock@test.com",
            send_interval_sec=0,
        )

        mock_smtp = unittest.mock.MagicMock()
        monkeypatch.setattr("smtplib.SMTP", lambda *a, **kw: mock_smtp)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_pw")

        log = send_emails(
            staging_dir=staged,
            template_path=tpl,
            smtp_config_path="",
            smtp_config=cfg,
        )

        assert log.smtp_server == "mock.smtp.com"
        assert log.success == 2
        assert mock_smtp.send_message.call_count == 2


# ===========================================================================
# IT-04: CLI end-to-end with forma.json fallback (US1)
# ===========================================================================


class TestCliFormaJsonFallback:
    """Integration: CLI send without --smtp-config, using forma.json fallback."""

    def setup_method(self):
        MockSMTP.reset()

    def test_prepare_then_send_via_forma_json(self, tmp_path, monkeypatch):
        """Full prepare -> send pipeline using forma.json for SMTP config."""
        from forma.cli_deliver import main

        # Create reports
        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        for sid in ("s001", "s002"):
            (report_dir / f"{sid}_report.pdf").write_bytes(b"%PDF-1.4 " + sid.encode())

        # Manifest
        mf = tmp_path / "manifest.yaml"
        mf.write_text(textwrap.dedent(f"""\
            report_source:
              directory: "{report_dir}"
              file_patterns:
                - "{{student_id}}_report.pdf"
        """), encoding="utf-8")

        # Roster
        roster = tmp_path / "roster.yaml"
        roster.write_text(yaml.dump({
            "class_name": "IT-04",
            "students": [
                {"student_id": "s001", "name": "A", "email": "a@u.kr"},
                {"student_id": "s002", "name": "B", "email": "b@u.kr"},
            ],
        }, allow_unicode=True), encoding="utf-8")

        output_dir = str(tmp_path / "staging")
        template_path = _write_template(tmp_path)

        # Step 1: prepare
        try:
            main([
                "--no-config", "prepare",
                "--manifest", str(mf),
                "--roster", str(roster),
                "--output-dir", output_dir,
            ])
        except SystemExit:
            pass

        # Step 2: send with forma.json fallback
        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: {
                "smtp": {
                    "server": "forma-json.smtp.com",
                    "sender_email": "fj@test.com",
                    "send_interval_sec": 0,
                }
            },
        )
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        try:
            main([
                "--no-config", "send",
                "--staged", output_dir,
                "--template", template_path,
                "--dry-run",
            ])
        except SystemExit as e:
            assert e.code == 0

        log_path = os.path.join(output_dir, "delivery_log.yaml")
        with open(log_path, encoding="utf-8") as f:
            log_data = yaml.safe_load(f)

        assert log_data["smtp_server"] == "forma-json.smtp.com"
        assert log_data["dry_run"] is True
        assert log_data["total"] == 2
        assert log_data["success"] == 2

    def test_dry_run_with_forma_json_no_deprecation(self, tmp_path, monkeypatch):
        """Using forma.json (no --smtp-config) emits NO deprecation warning."""
        from forma.cli_deliver import main

        staged = _create_staged_dir(tmp_path, n_students=1)
        tpl = _write_template(tmp_path)

        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: {
                "smtp": {
                    "server": "json.smtp.com",
                    "sender_email": "a@b.com",
                    "send_interval_sec": 0,
                }
            },
        )
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config", "send",
                    "--staged", staged,
                    "--template", tpl,
                    "--dry-run",
                ])
            except SystemExit:
                pass

            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 0

    def test_real_send_with_forma_json(self, tmp_path, monkeypatch):
        """Real send (mock SMTP) via forma.json smtp section."""
        from forma.cli_deliver import main

        staged = _create_staged_dir(tmp_path, n_students=1)
        tpl = _write_template(tmp_path)

        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: {
                "smtp": {
                    "server": "json-real.smtp.com",
                    "sender_email": "jr@test.com",
                    "send_interval_sec": 0,
                }
            },
        )
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_pw")
        monkeypatch.setattr("smtplib.SMTP", MockSMTP)

        try:
            main([
                "--no-config", "send",
                "--staged", staged,
                "--template", tpl,
            ])
        except SystemExit as e:
            assert e.code == 0

        assert len(MockSMTP.sent_messages) == 1
        log_path = os.path.join(staged, "delivery_log.yaml")
        with open(log_path, encoding="utf-8") as f:
            log_data = yaml.safe_load(f)
        assert log_data["smtp_server"] == "json-real.smtp.com"


# ===========================================================================
# IT-05: CLI --smtp-config deprecation path (US2)
# ===========================================================================


class TestCliDeprecationPath:
    """Integration: CLI --smtp-config triggers deprecation but still works."""

    def test_deprecated_flag_produces_correct_output(self, tmp_path, monkeypatch):
        """--smtp-config works correctly AND produces DeprecationWarning."""
        from forma.cli_deliver import main

        staged = _create_staged_dir(tmp_path, n_students=1)
        tpl = _write_template(tmp_path)
        smtp_path = _write_smtp_yaml(tmp_path)
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config", "send",
                    "--staged", staged,
                    "--template", tpl,
                    "--smtp-config", smtp_path,
                    "--dry-run",
                ])
            except SystemExit as e:
                assert e.code == 0

            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1

        log_path = os.path.join(staged, "delivery_log.yaml")
        with open(log_path, encoding="utf-8") as f:
            log_data = yaml.safe_load(f)
        assert log_data["smtp_server"] == "yaml.smtp.com"
        assert log_data["success"] == 1

    def test_deprecation_warning_korean_text(self, tmp_path, monkeypatch):
        """Deprecation warning message is in Korean and matches spec exactly."""
        from forma.cli_deliver import main

        staged = _create_staged_dir(tmp_path, n_students=1)
        tpl = _write_template(tmp_path)
        smtp_path = _write_smtp_yaml(tmp_path)
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config", "send",
                    "--staged", staged,
                    "--template", tpl,
                    "--smtp-config", smtp_path,
                    "--dry-run",
                ])
            except SystemExit:
                pass

            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            expected = (
                "--smtp-config는 향후 버전에서 제거됩니다. "
                "config.json의 smtp 섹션으로 마이그레이션하세요."
            )
            assert str(dep_warnings[0].message) == expected

    def test_backward_compat_identical_to_v110(self, tmp_path, monkeypatch):
        """--smtp-config YAML produces identical delivery behavior to v0.11.0."""
        from forma.cli_deliver import main

        staged = _create_staged_dir(tmp_path, n_students=2)
        tpl = _write_template(tmp_path)
        smtp_path = _write_smtp_yaml(tmp_path, server="v110-compat.smtp.com")
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config", "send",
                    "--staged", staged,
                    "--template", tpl,
                    "--smtp-config", smtp_path,
                    "--dry-run",
                ])
            except SystemExit as e:
                assert e.code == 0

        log_path = os.path.join(staged, "delivery_log.yaml")
        with open(log_path, encoding="utf-8") as f:
            log_data = yaml.safe_load(f)
        # v0.11.0 behavior: YAML server used, dry_run works, all succeed
        assert log_data["smtp_server"] == "v110-compat.smtp.com"
        assert log_data["total"] == 2
        assert log_data["success"] == 2
        assert log_data["dry_run"] is True


# ===========================================================================
# IT-06: Mixed scenario — both exist, explicit wins
# ===========================================================================


class TestMixedSmtpSources:
    """When both forma.json smtp and --smtp-config exist, explicit wins."""

    def setup_method(self):
        MockSMTP.reset()

    def test_explicit_smtp_config_overrides_forma_json(self, tmp_path, monkeypatch):
        """--smtp-config overrides forma.json smtp section."""
        from forma.cli_deliver import main

        staged = _create_staged_dir(tmp_path, n_students=1)
        tpl = _write_template(tmp_path)
        smtp_path = _write_smtp_yaml(tmp_path, server="explicit-yaml.smtp.com")

        # forma.json has a different server — should NOT be used
        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: {
                "smtp": {
                    "server": "forma-json-ignored.smtp.com",
                    "sender_email": "ignored@test.com",
                    "send_interval_sec": 0,
                }
            },
        )
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config", "send",
                    "--staged", staged,
                    "--template", tpl,
                    "--smtp-config", smtp_path,
                    "--dry-run",
                ])
            except SystemExit as e:
                assert e.code == 0

            # Explicit YAML used, not forma.json
            log_path = os.path.join(staged, "delivery_log.yaml")
            with open(log_path, encoding="utf-8") as f:
                log_data = yaml.safe_load(f)
            assert log_data["smtp_server"] == "explicit-yaml.smtp.com"

            # Deprecation warning emitted (because --smtp-config was used)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) >= 1

    def test_mixed_real_send(self, tmp_path, monkeypatch):
        """Mixed: real send uses --smtp-config server, not forma.json."""
        from forma.cli_deliver import main

        staged = _create_staged_dir(tmp_path, n_students=1)
        tpl = _write_template(tmp_path)
        smtp_path = _write_smtp_yaml(tmp_path, server="yaml-priority.smtp.com")

        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: {
                "smtp": {
                    "server": "json-not-used.smtp.com",
                    "sender_email": "nope@test.com",
                    "send_interval_sec": 0,
                }
            },
        )
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_pw")
        monkeypatch.setattr("smtplib.SMTP", MockSMTP)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config", "send",
                    "--staged", staged,
                    "--template", tpl,
                    "--smtp-config", smtp_path,
                ])
            except SystemExit as e:
                assert e.code == 0

        log_path = os.path.join(staged, "delivery_log.yaml")
        with open(log_path, encoding="utf-8") as f:
            log_data = yaml.safe_load(f)
        assert log_data["smtp_server"] == "yaml-priority.smtp.com"


# ===========================================================================
# IT-07: Missing both sources -> clean error (FR-006)
# ===========================================================================


class TestMissingSmtpBothSources:
    """No --smtp-config and no forma.json -> clean exit code 2."""

    def test_no_config_no_flag_exit_2(self, tmp_path, monkeypatch, capsys):
        """FR-006: No forma.json, no --smtp-config -> exit 2 with Korean error."""
        from forma.cli_deliver import main

        staged = _create_staged_dir(tmp_path, n_students=1)
        tpl = _write_template(tmp_path)

        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: (_ for _ in ()).throw(
                FileNotFoundError("No config file found")
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "send",
                "--staged", staged,
                "--template", tpl,
            ])
        assert exc_info.value.code == 2

        captured = capsys.readouterr()
        assert "SMTP 설정을 찾을 수 없습니다" in captured.err
        assert "--smtp-config" in captured.err
        assert "config.json" in captured.err

    def test_forma_json_no_smtp_section_exit_2(self, tmp_path, monkeypatch):
        """forma.json without 'smtp' section -> exit 2."""
        from forma.cli_deliver import main

        staged = _create_staged_dir(tmp_path, n_students=1)
        tpl = _write_template(tmp_path)

        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: {"llm": {"provider": "gemini"}},
        )

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "send",
                "--staged", staged,
                "--template", tpl,
            ])
        assert exc_info.value.code == 2

    def test_forma_json_smtp_not_dict_exit_2(self, tmp_path, monkeypatch):
        """forma.json with smtp as string (not dict) -> exit 2."""
        from forma.cli_deliver import main

        staged = _create_staged_dir(tmp_path, n_students=1)
        tpl = _write_template(tmp_path)

        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: {"smtp": "not-a-dict"},
        )

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "send",
                "--staged", staged,
                "--template", tpl,
            ])
        assert exc_info.value.code == 2


# ===========================================================================
# IT-08: forma.json with invalid smtp -> validation error
# ===========================================================================


class TestFormaJsonInvalidSmtp:
    """forma.json with structurally invalid smtp data."""

    def test_port_zero_raises(self):
        """Port=0 in forma.json smtp -> ValueError."""
        from forma.config import get_smtp_config

        config = {"smtp": {
            "server": "smtp.test.com",
            "port": 0,
            "sender_email": "a@b.com",
        }}
        with pytest.raises(ValueError, match="smtp_port"):
            get_smtp_config(config)

    def test_missing_server_raises(self):
        """Missing server in forma.json smtp -> ValueError."""
        from forma.config import get_smtp_config

        config = {"smtp": {"port": 587, "sender_email": "a@b.com"}}
        with pytest.raises(ValueError, match="smtp_server"):
            get_smtp_config(config)

    def test_missing_sender_email_raises(self):
        """Missing sender_email -> ValueError."""
        from forma.config import get_smtp_config

        config = {"smtp": {"server": "smtp.test.com", "port": 587}}
        with pytest.raises(ValueError, match="sender_email"):
            get_smtp_config(config)

    def test_invalid_sender_email_raises(self):
        """sender_email without '@' -> ValueError."""
        from forma.config import get_smtp_config

        config = {"smtp": {
            "server": "smtp.test.com",
            "port": 587,
            "sender_email": "no-at-sign",
        }}
        with pytest.raises(ValueError, match="sender_email"):
            get_smtp_config(config)

    def test_port_out_of_range_raises(self):
        """Port > 65535 -> ValueError."""
        from forma.config import get_smtp_config

        config = {"smtp": {
            "server": "smtp.test.com",
            "port": 99999,
            "sender_email": "a@b.com",
        }}
        with pytest.raises(ValueError, match="smtp_port"):
            get_smtp_config(config)

    def test_invalid_smtp_via_cli_exit_2(self, tmp_path, monkeypatch):
        """Invalid forma.json smtp via CLI -> exit 2."""
        from forma.cli_deliver import main

        staged = _create_staged_dir(tmp_path, n_students=1)
        tpl = _write_template(tmp_path)

        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: {"smtp": {
                "server": "smtp.test.com",
                "port": 0,
                "sender_email": "a@b.com",
            }},
        )

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "send",
                "--staged", staged,
                "--template", tpl,
            ])
        assert exc_info.value.code == 2


# ===========================================================================
# IT-09: CI workflow file validation (US3)
# ===========================================================================


class TestCIWorkflowValidation:
    """Validate .github/workflows/ci.yml structure."""

    @pytest.fixture(autouse=True)
    def _load_ci_yaml(self):
        ci_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            ".github", "workflows", "ci.yml",
        )
        assert os.path.exists(ci_path), f"CI workflow file not found: {ci_path}"
        with open(ci_path, encoding="utf-8") as f:
            self.ci = yaml.safe_load(f)

    def test_ci_file_is_valid_yaml(self):
        """ci.yml is valid YAML with basic structure."""
        assert isinstance(self.ci, dict)
        assert "jobs" in self.ci

    def test_has_test_job(self):
        """ci.yml has a 'test' job."""
        assert "test" in self.ci["jobs"]

    def test_has_lint_job(self):
        """ci.yml has a 'lint' job."""
        assert "lint" in self.ci["jobs"]

    def test_python_matrix_includes_311_and_312(self):
        """Test job python matrix includes 3.11 and 3.12."""
        test_job = self.ci["jobs"]["test"]
        matrix = test_job.get("strategy", {}).get("matrix", {})
        python_versions = matrix.get("python-version", [])
        assert "3.11" in python_versions
        assert "3.12" in python_versions

    def test_jdk_setup_step_exists(self):
        """Test job has a JDK setup step (for KoNLPy)."""
        test_job = self.ci["jobs"]["test"]
        steps = test_job.get("steps", [])
        jdk_steps = [
            s for s in steps
            if "setup-java" in str(s.get("uses", ""))
        ]
        assert len(jdk_steps) >= 1, "JDK setup step not found"

    def test_codecov_upload_step_exists(self):
        """Test job has a Codecov upload step."""
        test_job = self.ci["jobs"]["test"]
        steps = test_job.get("steps", [])
        codecov_steps = [
            s for s in steps
            if "codecov" in str(s.get("uses", "")).lower()
        ]
        assert len(codecov_steps) >= 1, "Codecov upload step not found"

    def test_lint_job_runs_ruff(self):
        """Lint job runs ruff check."""
        lint_job = self.ci["jobs"]["lint"]
        steps = lint_job.get("steps", [])
        ruff_steps = [
            s for s in steps
            if "ruff" in str(s.get("run", "")).lower()
               or "ruff" in str(s.get("name", "")).lower()
        ]
        assert len(ruff_steps) >= 1, "Ruff check step not found in lint job"


# ===========================================================================
# IT-10: pyproject.toml validation (US4)
# ===========================================================================


class TestPyprojectTomlValidation:
    """Validate pyproject.toml for v0.11.1 requirements."""

    @pytest.fixture(autouse=True)
    def _load_pyproject(self):
        pyproject_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "pyproject.toml",
        )
        assert os.path.exists(pyproject_path), "pyproject.toml not found"
        import tomllib
        with open(pyproject_path, "rb") as f:
            self.config = tomllib.load(f)

    def test_version_is_set(self):
        """Version field exists and follows semver format."""
        import re
        version = self.config["project"]["version"]
        assert re.match(r"^\d+\.\d+\.\d+", version), f"Invalid version format: {version}"

    def test_ruff_in_dev_dependencies(self):
        """ruff is in dev optional dependencies."""
        dev_deps = self.config["project"]["optional-dependencies"]["dev"]
        ruff_deps = [d for d in dev_deps if d.startswith("ruff")]
        assert len(ruff_deps) >= 1, "ruff not found in dev dependencies"

    def test_forma_deliver_entry_point(self):
        """forma-deliver entry point exists and points to cli_deliver:main."""
        scripts = self.config["project"]["scripts"]
        assert "forma-deliver" in scripts
        assert scripts["forma-deliver"] == "forma.cli_deliver:main"

    def test_requires_python_311(self):
        """Python >=3.11 is required."""
        requires = self.config["project"]["requires-python"]
        assert "3.11" in requires

    def test_pyyaml_in_dependencies(self):
        """PyYAML is in core dependencies."""
        deps = self.config["project"]["dependencies"]
        yaml_deps = [d for d in deps if "pyyaml" in d.lower()]
        assert len(yaml_deps) >= 1


# ===========================================================================
# IT-11: config.py JSON_FIELD_MAP validation (T024)
# ===========================================================================


class TestJsonFieldMapValidation:
    """Validate JSON_FIELD_MAP completeness in config.py."""

    def test_field_map_has_all_6_fields(self):
        """JSON_FIELD_MAP maps all 6 required fields correctly."""
        from forma.config import JSON_FIELD_MAP

        expected = {
            "server": "smtp_server",
            "port": "smtp_port",
            "sender_email": "sender_email",
            "sender_name": "sender_name",
            "use_tls": "use_tls",
            "send_interval_sec": "send_interval_sec",
        }
        assert JSON_FIELD_MAP == expected

    def test_field_map_covers_all_smtp_config_fields(self):
        """JSON_FIELD_MAP destination values cover all SmtpConfig fields."""
        from forma.config import JSON_FIELD_MAP
        from forma.delivery_send import SmtpConfig

        import dataclasses
        smtp_fields = {f.name for f in dataclasses.fields(SmtpConfig)}
        mapped_fields = set(JSON_FIELD_MAP.values())
        assert mapped_fields == smtp_fields

    def test_password_never_in_field_map(self):
        """FR-007: password must NEVER appear in JSON_FIELD_MAP."""
        from forma.config import JSON_FIELD_MAP

        assert "password" not in JSON_FIELD_MAP
        assert "password" not in JSON_FIELD_MAP.values()
        assert "smtp_password" not in JSON_FIELD_MAP.values()


# ===========================================================================
# IT-12: config.py resolution order
# ===========================================================================


class TestConfigResolutionOrder:
    """Validate load_config() resolution order."""

    def test_explicit_path_first(self, tmp_path):
        """Explicit config_path is used before any default."""
        from forma.config import load_config

        cfg_path = tmp_path / "explicit.json"
        cfg_path.write_text(json.dumps({"test": "explicit"}), encoding="utf-8")

        result = load_config(str(cfg_path))
        assert result == {"test": "explicit"}

    def test_file_not_found_when_nothing_exists(self, tmp_path, monkeypatch):
        """FileNotFoundError when no config file exists."""
        from forma.config import load_config

        monkeypatch.setattr("forma.config.AGENIX_CONFIG_PATH", str(tmp_path / "nope1"))
        monkeypatch.setattr("forma.config.DEFAULT_CONFIG_PATH", str(tmp_path / "nope2"))
        monkeypatch.setattr("forma.config.DEPRECATED_CONFIG_PATH", str(tmp_path / "nope3"))

        with pytest.raises(FileNotFoundError):
            load_config()

    def test_default_path_used_when_no_agenix(self, tmp_path, monkeypatch):
        """DEFAULT_CONFIG_PATH used when agenix path doesn't exist."""
        from forma.config import load_config

        default_path = tmp_path / "forma.json"
        default_path.write_text(json.dumps({"from": "default"}), encoding="utf-8")

        monkeypatch.setattr("forma.config.AGENIX_CONFIG_PATH", str(tmp_path / "no_agenix"))
        monkeypatch.setattr("forma.config.DEFAULT_CONFIG_PATH", str(default_path))

        result = load_config()
        assert result == {"from": "default"}

    def test_agenix_takes_priority_over_default(self, tmp_path, monkeypatch):
        """Agenix path (resolution order #2) takes priority over default (#3)."""
        from forma.config import load_config

        agenix_path = tmp_path / "agenix.json"
        agenix_path.write_text(json.dumps({"from": "agenix"}), encoding="utf-8")

        default_path = tmp_path / "forma.json"
        default_path.write_text(json.dumps({"from": "default"}), encoding="utf-8")

        monkeypatch.setattr("forma.config.AGENIX_CONFIG_PATH", str(agenix_path))
        monkeypatch.setattr("forma.config.DEFAULT_CONFIG_PATH", str(default_path))

        result = load_config()
        assert result == {"from": "agenix"}
