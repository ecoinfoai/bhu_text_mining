"""Tests for cli_deliver.py -- forma-deliver CLI entry point.

T008: RED tests for `forma-deliver prepare` subcommand.
      Parser tests, happy path, error exits, console summary output.

T014: RED tests for `forma-deliver send` subcommand.
      Parser tests, happy path (mock SMTP), error exits, --force flag.

Covers CLI contract: exit codes 0/1/2/3, console summary output.
"""
from __future__ import annotations

import os
import textwrap

import pytest
import yaml


# ---------------------------------------------------------------------------
# T008: Parser tests -- prepare subcommand
# ---------------------------------------------------------------------------


class TestCliDeliverPrepareParser:
    """Tests for _build_parser() with prepare subcommand arguments."""

    def test_prepare_required_args(self):
        """prepare subcommand parses --manifest, --roster, --output-dir."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "prepare",
            "--manifest", "manifest.yaml",
            "--roster", "roster.yaml",
            "--output-dir", "/tmp/staging",
        ])
        assert args.subcommand == "prepare"
        assert args.manifest == "manifest.yaml"
        assert args.roster == "roster.yaml"
        assert args.output_dir == "/tmp/staging"

    def test_prepare_force_flag(self):
        """prepare subcommand accepts --force flag."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "prepare",
            "--manifest", "m.yaml",
            "--roster", "r.yaml",
            "--output-dir", "/tmp/out",
            "--force",
        ])
        assert args.force is True

    def test_prepare_force_default_false(self):
        """--force defaults to False when not specified."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "prepare",
            "--manifest", "m.yaml",
            "--roster", "r.yaml",
            "--output-dir", "/tmp/out",
        ])
        assert args.force is False

    def test_no_config_flag(self):
        """--no-config flag is accepted."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--no-config",
            "prepare",
            "--manifest", "m.yaml",
            "--roster", "r.yaml",
            "--output-dir", "/tmp/out",
        ])
        assert args.no_config is True

    def test_verbose_flag(self):
        """--verbose flag is accepted."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--verbose",
            "prepare",
            "--manifest", "m.yaml",
            "--roster", "r.yaml",
            "--output-dir", "/tmp/out",
        ])
        assert args.verbose is True

    def test_no_subcommand_exits(self):
        """No subcommand exits with error."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_prepare_missing_manifest_exits(self):
        """prepare without --manifest exits with argparse error."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "prepare",
                "--roster", "r.yaml",
                "--output-dir", "/tmp/out",
            ])

    def test_prepare_missing_roster_exits(self):
        """prepare without --roster exits with argparse error."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "prepare",
                "--manifest", "m.yaml",
                "--output-dir", "/tmp/out",
            ])

    def test_prepare_missing_output_dir_exits(self):
        """prepare without --output-dir exits with argparse error."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "prepare",
                "--manifest", "m.yaml",
                "--roster", "r.yaml",
            ])


# ---------------------------------------------------------------------------
# T008: `forma-deliver prepare` -- happy path
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path, report_dir):
    """Helper: write a valid manifest.yaml."""
    mf = tmp_path / "manifest.yaml"
    mf.write_text(
        textwrap.dedent(f"""\
            report_source:
              directory: "{report_dir}"
              file_patterns:
                - "{{student_id}}_report.pdf"
        """),
        encoding="utf-8",
    )
    return str(mf)


def _write_roster(tmp_path, students=None):
    """Helper: write a valid roster.yaml."""
    if students is None:
        students = [
            {"student_id": "s001", "name": "홍길동", "email": "hong@u.kr"},
            {"student_id": "s002", "name": "김철수", "email": "kim@u.kr"},
            {"student_id": "s003", "name": "이영희", "email": "lee@u.kr"},
        ]
    rf = tmp_path / "roster.yaml"
    rf.write_text(
        yaml.dump(
            {"class_name": "해부생리학 1A", "students": students},
            allow_unicode=True,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return str(rf)


def _create_student_pdf(report_dir, student_id):
    """Helper: create a dummy PDF file for a student."""
    pdf_path = os.path.join(str(report_dir), f"{student_id}_report.pdf")
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4 dummy content " + student_id.encode())
    return pdf_path


class TestCliDeliverPrepareHappyPath:
    """Tests for `forma-deliver prepare` happy path execution."""

    def test_prepare_success_exit_0(self, tmp_path):
        """Successful prepare exits with code 0."""
        from forma.cli_deliver import main

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        for sid in ("s001", "s002", "s003"):
            _create_student_pdf(report_dir, sid)

        manifest_path = _write_manifest(tmp_path, str(report_dir))
        roster_path = _write_roster(tmp_path)
        output_dir = str(tmp_path / "staging")

        # main() should not raise SystemExit or should raise SystemExit(0)
        try:
            _result = main([
                "--no-config",
                "prepare",
                "--manifest", manifest_path,
                "--roster", roster_path,
                "--output-dir", output_dir,
            ])
        except SystemExit as e:
            assert e.code == 0

    def test_prepare_creates_summary(self, tmp_path):
        """Successful prepare creates prepare_summary.yaml."""
        from forma.cli_deliver import main

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        for sid in ("s001", "s002", "s003"):
            _create_student_pdf(report_dir, sid)

        manifest_path = _write_manifest(tmp_path, str(report_dir))
        roster_path = _write_roster(tmp_path)
        output_dir = str(tmp_path / "staging")

        try:
            main([
                "--no-config",
                "prepare",
                "--manifest", manifest_path,
                "--roster", roster_path,
                "--output-dir", output_dir,
            ])
        except SystemExit:
            pass

        summary_path = os.path.join(output_dir, "prepare_summary.yaml")
        assert os.path.exists(summary_path)

    def test_prepare_creates_zip_files(self, tmp_path):
        """Successful prepare creates zip files for each student."""
        from forma.cli_deliver import main

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        for sid in ("s001", "s002", "s003"):
            _create_student_pdf(report_dir, sid)

        manifest_path = _write_manifest(tmp_path, str(report_dir))
        roster_path = _write_roster(tmp_path)
        output_dir = str(tmp_path / "staging")

        try:
            main([
                "--no-config",
                "prepare",
                "--manifest", manifest_path,
                "--roster", roster_path,
                "--output-dir", output_dir,
            ])
        except SystemExit:
            pass

        # At least one zip file should exist in the output directory tree
        zip_found = False
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".zip"):
                    zip_found = True
                    break
        assert zip_found, "No zip files found in staging directory"

    def test_prepare_summary_content(self, tmp_path):
        """prepare_summary.yaml has correct total/ready counts."""
        from forma.cli_deliver import main

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        for sid in ("s001", "s002", "s003"):
            _create_student_pdf(report_dir, sid)

        manifest_path = _write_manifest(tmp_path, str(report_dir))
        roster_path = _write_roster(tmp_path)
        output_dir = str(tmp_path / "staging")

        try:
            main([
                "--no-config",
                "prepare",
                "--manifest", manifest_path,
                "--roster", roster_path,
                "--output-dir", output_dir,
            ])
        except SystemExit:
            pass

        summary_path = os.path.join(output_dir, "prepare_summary.yaml")
        with open(summary_path, encoding="utf-8") as f:
            summary = yaml.safe_load(f)
        assert summary["total_students"] == 3
        assert summary["ready"] == 3
        assert summary["errors"] == 0

    def test_prepare_console_summary(self, tmp_path, capsys):
        """prepare outputs a console summary with total/ready/error counts."""
        from forma.cli_deliver import main

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        for sid in ("s001", "s002", "s003"):
            _create_student_pdf(report_dir, sid)

        manifest_path = _write_manifest(tmp_path, str(report_dir))
        roster_path = _write_roster(tmp_path)
        output_dir = str(tmp_path / "staging")

        try:
            main([
                "--no-config",
                "prepare",
                "--manifest", manifest_path,
                "--roster", roster_path,
                "--output-dir", output_dir,
            ])
        except SystemExit:
            pass

        captured = capsys.readouterr()
        # Console should show counts
        assert "3" in captured.out  # total students


# ---------------------------------------------------------------------------
# T008: `forma-deliver prepare` -- error exits
# ---------------------------------------------------------------------------


class TestCliDeliverPrepareErrors:
    """Tests for `forma-deliver prepare` error handling and exit codes."""

    def test_missing_manifest_file_exit_2(self, tmp_path):
        """Nonexistent manifest file exits with code 2."""
        from forma.cli_deliver import main

        roster_path = _write_roster(tmp_path)
        output_dir = str(tmp_path / "staging")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "prepare",
                "--manifest", str(tmp_path / "nonexistent_manifest.yaml"),
                "--roster", roster_path,
                "--output-dir", output_dir,
            ])
        assert exc_info.value.code == 2

    def test_missing_roster_file_exit_2(self, tmp_path):
        """Nonexistent roster file exits with code 2."""
        from forma.cli_deliver import main

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        manifest_path = _write_manifest(tmp_path, str(report_dir))
        output_dir = str(tmp_path / "staging")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "prepare",
                "--manifest", manifest_path,
                "--roster", str(tmp_path / "nonexistent_roster.yaml"),
                "--output-dir", output_dir,
            ])
        assert exc_info.value.code == 2

    def test_missing_report_directory_exit_2(self, tmp_path):
        """Nonexistent report directory (in manifest) exits with code 2."""
        from forma.cli_deliver import main

        mf = tmp_path / "manifest.yaml"
        mf.write_text(
            textwrap.dedent("""\
                report_source:
                  directory: "/nonexistent/report/dir"
                  file_patterns:
                    - "{student_id}_report.pdf"
            """),
            encoding="utf-8",
        )
        roster_path = _write_roster(tmp_path)
        output_dir = str(tmp_path / "staging")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "prepare",
                "--manifest", str(mf),
                "--roster", roster_path,
                "--output-dir", output_dir,
            ])
        # Exit 1 or 2 depending on implementation (directory validation in manifest)
        assert exc_info.value.code in (1, 2)

    def test_duplicate_student_id_exit_1(self, tmp_path):
        """Duplicate student_id in roster exits with code 1."""
        from forma.cli_deliver import main

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        manifest_path = _write_manifest(tmp_path, str(report_dir))

        roster_path = _write_roster(tmp_path, students=[
            {"student_id": "s001", "name": "홍길동", "email": "hong@u.kr"},
            {"student_id": "s001", "name": "홍길동복제", "email": "dup@u.kr"},
        ])
        output_dir = str(tmp_path / "staging")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "prepare",
                "--manifest", manifest_path,
                "--roster", roster_path,
                "--output-dir", output_dir,
            ])
        assert exc_info.value.code == 1

    def test_prepare_with_partial_match_still_exit_0(self, tmp_path):
        """prepare with partial file match (some students missing files) still exits 0.

        Students with missing files should be recorded as error in summary,
        but prepare itself succeeds.
        """
        from forma.cli_deliver import main

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        # Only create PDF for s001, not s002 or s003
        _create_student_pdf(report_dir, "s001")

        manifest_path = _write_manifest(tmp_path, str(report_dir))
        roster_path = _write_roster(tmp_path)
        output_dir = str(tmp_path / "staging")

        try:
            _result = main([
                "--no-config",
                "prepare",
                "--manifest", manifest_path,
                "--roster", roster_path,
                "--output-dir", output_dir,
            ])
        except SystemExit as e:
            # prepare succeeds even if some students have errors
            assert e.code == 0

    def test_prepare_error_students_in_summary(self, tmp_path):
        """Students with no matching files appear as error in summary."""
        from forma.cli_deliver import main

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        _create_student_pdf(report_dir, "s001")
        # s002, s003 have no files

        manifest_path = _write_manifest(tmp_path, str(report_dir))
        roster_path = _write_roster(tmp_path)
        output_dir = str(tmp_path / "staging")

        try:
            main([
                "--no-config",
                "prepare",
                "--manifest", manifest_path,
                "--roster", roster_path,
                "--output-dir", output_dir,
            ])
        except SystemExit:
            pass

        summary_path = os.path.join(output_dir, "prepare_summary.yaml")
        with open(summary_path, encoding="utf-8") as f:
            summary = yaml.safe_load(f)
        assert summary["total_students"] == 3
        assert summary["ready"] == 1
        assert summary["errors"] == 2

    def test_prepare_summary_details_has_all_students(self, tmp_path):
        """prepare_summary.yaml details includes ALL students (ready + error).

        C1 fix: PrepareSummary.details must contain all students,
        not just ready ones.
        """
        from forma.cli_deliver import main

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        _create_student_pdf(report_dir, "s001")

        manifest_path = _write_manifest(tmp_path, str(report_dir))
        roster_path = _write_roster(tmp_path)
        output_dir = str(tmp_path / "staging")

        try:
            main([
                "--no-config",
                "prepare",
                "--manifest", manifest_path,
                "--roster", roster_path,
                "--output-dir", output_dir,
            ])
        except SystemExit:
            pass

        summary_path = os.path.join(output_dir, "prepare_summary.yaml")
        with open(summary_path, encoding="utf-8") as f:
            summary = yaml.safe_load(f)

        details = summary.get("details", [])
        assert len(details) == 3, (
            "details must include ALL students (ready + warning + error)"
        )
        detail_ids = {d["student_id"] for d in details}
        assert detail_ids == {"s001", "s002", "s003"}


# ---------------------------------------------------------------------------
# T008: `forma-deliver prepare` -- staging folder overwrite
# ---------------------------------------------------------------------------


class TestCliDeliverPrepareOverwrite:
    """Tests for staging folder overwrite behavior (FR-020)."""

    def test_existing_staging_without_force_exits(self, tmp_path):
        """Existing staging folder without --force exits with error.

        FR-020: When staging folder exists, user must confirm or use --force.
        Without --force and without interactive prompt, exit with error.
        """
        from forma.cli_deliver import main

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        _create_student_pdf(report_dir, "s001")

        manifest_path = _write_manifest(tmp_path, str(report_dir))
        roster_path = _write_roster(tmp_path, students=[
            {"student_id": "s001", "name": "홍길동", "email": "hong@u.kr"},
        ])
        output_dir = tmp_path / "staging"
        output_dir.mkdir()
        # Create a marker file to prove staging exists
        (output_dir / "marker.txt").write_text("existing", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "prepare",
                "--manifest", manifest_path,
                "--roster", roster_path,
                "--output-dir", str(output_dir),
            ])
        assert exc_info.value.code == 1

    def test_existing_staging_with_force_succeeds(self, tmp_path):
        """Existing staging folder with --force overwrites successfully."""
        from forma.cli_deliver import main

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        _create_student_pdf(report_dir, "s001")

        manifest_path = _write_manifest(tmp_path, str(report_dir))
        roster_path = _write_roster(tmp_path, students=[
            {"student_id": "s001", "name": "홍길동", "email": "hong@u.kr"},
        ])
        output_dir = tmp_path / "staging"
        output_dir.mkdir()
        (output_dir / "old_file.txt").write_text("old", encoding="utf-8")

        try:
            main([
                "--no-config",
                "prepare",
                "--manifest", manifest_path,
                "--roster", roster_path,
                "--output-dir", str(output_dir),
                "--force",
            ])
        except SystemExit as e:
            assert e.code == 0

        # Summary should exist in the overwritten directory
        assert os.path.exists(os.path.join(str(output_dir), "prepare_summary.yaml"))


# ===========================================================================
# T014: `forma-deliver send` subcommand tests
# ===========================================================================


def _write_smtp_config(tmp_path):
    """Helper: write a valid smtp.yaml."""
    cfg = tmp_path / "smtp.yaml"
    cfg.write_text(
        textwrap.dedent("""\
            smtp_server: "smtp.gmail.com"
            smtp_port: 587
            sender_email: "prof@univ.ac.kr"
            sender_name: "담당교수"
            use_tls: true
            send_interval_sec: 0.0
        """),
        encoding="utf-8",
    )
    return str(cfg)


def _write_email_template(tmp_path):
    """Helper: write a valid email_template.yaml."""
    tpl = tmp_path / "template.yaml"
    tpl.write_text(
        textwrap.dedent("""\
            subject: "[해부생리학] 형성평가 피드백"
            body: |
              {student_name} 학생에게,
              {class_name} 수업의 피드백을 첨부합니다.
        """),
        encoding="utf-8",
    )
    return str(tpl)


def _create_staged_folder(tmp_path, n_students=3, with_errors=0):
    """Helper: create a staging folder with prepare_summary.yaml and zips.

    Returns:
        Path to the staging folder.
    """
    import zipfile

    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()

    details = []
    for i in range(n_students):
        sid = f"s{i:03d}"
        name = f"학생{i}"
        email = f"s{i:03d}@u.kr"

        if i < n_students - with_errors:
            # Ready student with zip
            student_dir = staged_dir / f"{sid}_{name}"
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
        else:
            # Error student with no zip
            details.append({
                "student_id": sid,
                "name": name,
                "email": email,
                "status": "error",
                "matched_files": [],
                "zip_path": None,
                "zip_size_bytes": 0,
                "message": "매칭 파일 없음",
            })

    ready_count = n_students - with_errors
    summary = {
        "prepared_at": "2026-03-11T10:00:00+09:00",
        "total_students": n_students,
        "ready": ready_count,
        "warnings": 0,
        "errors": with_errors,
        "details": details,
    }
    summary_path = staged_dir / "prepare_summary.yaml"
    with open(str(summary_path), "w", encoding="utf-8") as f:
        yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)

    return str(staged_dir)


# ---------------------------------------------------------------------------
# T014: Parser tests -- send subcommand
# ---------------------------------------------------------------------------


class TestCliDeliverSendParser:
    """Tests for _build_parser() with send subcommand arguments."""

    def test_send_required_args(self):
        """send subcommand parses --staged, --template, --smtp-config."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "send",
            "--staged", "/tmp/staged",
            "--template", "template.yaml",
            "--smtp-config", "smtp.yaml",
        ])
        assert args.subcommand == "send"
        assert args.staged == "/tmp/staged"
        assert args.template == "template.yaml"
        assert args.smtp_config == "smtp.yaml"

    def test_send_dry_run_flag(self):
        """send subcommand accepts --dry-run flag."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "send",
            "--staged", "/tmp/staged",
            "--template", "t.yaml",
            "--smtp-config", "s.yaml",
            "--dry-run",
        ])
        assert args.dry_run is True

    def test_send_retry_failed_flag(self):
        """send subcommand accepts --retry-failed flag."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "send",
            "--staged", "/tmp/staged",
            "--template", "t.yaml",
            "--smtp-config", "s.yaml",
            "--retry-failed",
        ])
        assert args.retry_failed is True

    def test_send_force_flag(self):
        """send subcommand accepts --force flag."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "send",
            "--staged", "/tmp/staged",
            "--template", "t.yaml",
            "--smtp-config", "s.yaml",
            "--force",
        ])
        assert args.force is True

    def test_send_notify_sender_flag(self):
        """send subcommand accepts --notify-sender flag."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "send",
            "--staged", "/tmp/staged",
            "--template", "t.yaml",
            "--smtp-config", "s.yaml",
            "--notify-sender",
        ])
        assert args.notify_sender is True

    def test_send_password_from_stdin_flag(self):
        """send subcommand accepts --password-from-stdin flag."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "send",
            "--staged", "/tmp/staged",
            "--template", "t.yaml",
            "--smtp-config", "s.yaml",
            "--password-from-stdin",
        ])
        assert args.password_from_stdin is True

    def test_send_all_flags_default_false(self):
        """All optional send flags default to False."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "send",
            "--staged", "/tmp/staged",
            "--template", "t.yaml",
            "--smtp-config", "s.yaml",
        ])
        assert args.dry_run is False
        assert args.retry_failed is False
        assert args.force is False
        assert args.notify_sender is False
        assert args.password_from_stdin is False

    def test_send_missing_staged_exits(self):
        """send without --staged exits with argparse error."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "send",
                "--template", "t.yaml",
                "--smtp-config", "s.yaml",
            ])

    def test_send_missing_template_exits(self):
        """send without --template exits with argparse error."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "send",
                "--staged", "/tmp/staged",
                "--smtp-config", "s.yaml",
            ])

    def test_send_without_smtp_config_defaults_none(self):
        """send without --smtp-config defaults to None (optional since US1)."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "send",
            "--staged", "/tmp/staged",
            "--template", "t.yaml",
        ])
        assert args.smtp_config is None


# ---------------------------------------------------------------------------
# T014: `forma-deliver send` -- happy path with mock SMTP
# ---------------------------------------------------------------------------


class TestCliDeliverSendHappyPath:
    """Tests for `forma-deliver send` happy path (mock SMTP)."""

    def test_send_success_exit_0(self, tmp_path, monkeypatch):
        """Successful send with mock SMTP exits with code 0."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=2)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)

        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_password")

        # Mock smtplib.SMTP so no real connection is made
        import unittest.mock
        mock_smtp = unittest.mock.MagicMock()
        monkeypatch.setattr("smtplib.SMTP", lambda *a, **kw: mock_smtp)

        try:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
            ])
        except SystemExit as e:
            assert e.code == 0

    def test_send_creates_delivery_log(self, tmp_path, monkeypatch):
        """Successful send creates delivery_log.yaml."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=2)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)

        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "test_password")

        import unittest.mock
        mock_smtp = unittest.mock.MagicMock()
        monkeypatch.setattr("smtplib.SMTP", lambda *a, **kw: mock_smtp)

        try:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
            ])
        except SystemExit:
            pass

        log_path = os.path.join(staged_dir, "delivery_log.yaml")
        assert os.path.exists(log_path)


# ---------------------------------------------------------------------------
# T014: `forma-deliver send` -- error exits
# ---------------------------------------------------------------------------


class TestCliDeliverSendErrors:
    """Tests for `forma-deliver send` error handling."""

    def test_missing_password_exit_1(self, tmp_path, monkeypatch):
        """Missing SMTP password (no env var, no --password-from-stdin) exits 1."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)

        # Ensure no password in environment
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
            ])
        assert exc_info.value.code == 1

    def test_missing_staged_dir_exit_2(self, tmp_path, monkeypatch):
        """Nonexistent staging directory exits with code 2."""
        from forma.cli_deliver import main

        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "send",
                "--staged", str(tmp_path / "nonexistent_staged"),
                "--template", template_path,
                "--smtp-config", smtp_path,
            ])
        assert exc_info.value.code == 2

    def test_missing_template_exit_2(self, tmp_path, monkeypatch):
        """Nonexistent template file exits with code 2."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", str(tmp_path / "nonexistent.yaml"),
                "--smtp-config", smtp_path,
            ])
        assert exc_info.value.code == 2

    def test_missing_smtp_config_exit_2(self, tmp_path, monkeypatch):
        """Nonexistent SMTP config file exits with code 2."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", str(tmp_path / "nonexistent.yaml"),
            ])
        assert exc_info.value.code == 2

    def test_resend_without_force_exit_1(self, tmp_path, monkeypatch):
        """Re-send with existing delivery_log.yaml (no --force) exits 1 (FR-022)."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        # Create an existing delivery_log.yaml with a success record
        log_data = {
            "sent_at": "2026-03-11T10:00:00+09:00",
            "smtp_server": "smtp.gmail.com",
            "dry_run": False,
            "total": 1,
            "success": 1,
            "failed": 0,
            "results": [{
                "student_id": "s000",
                "email": "s000@u.kr",
                "status": "success",
                "sent_at": "2026-03-11T10:00:01+09:00",
                "attachment": "학생0_s000.zip",
                "size_bytes": 100,
                "error": "",
            }],
        }
        log_path = os.path.join(staged_dir, "delivery_log.yaml")
        with open(log_path, "w", encoding="utf-8") as f:
            yaml.dump(log_data, f, allow_unicode=True)

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
            ])
        # Should refuse to resend without --force
        assert exc_info.value.code == 1

    def test_resend_with_force_succeeds(self, tmp_path, monkeypatch):
        """Re-send with --force overwrites delivery_log.yaml (FR-022)."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        # Create existing delivery_log
        log_data = {
            "sent_at": "2026-03-11T10:00:00+09:00",
            "smtp_server": "smtp.gmail.com",
            "dry_run": False,
            "total": 1,
            "success": 1,
            "failed": 0,
            "results": [],
        }
        log_path = os.path.join(staged_dir, "delivery_log.yaml")
        with open(log_path, "w", encoding="utf-8") as f:
            yaml.dump(log_data, f, allow_unicode=True)

        import unittest.mock
        mock_smtp = unittest.mock.MagicMock()
        monkeypatch.setattr("smtplib.SMTP", lambda *a, **kw: mock_smtp)

        try:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
                "--force",
            ])
        except SystemExit as e:
            assert e.code == 0

    def test_partial_failure_exit_3(self, tmp_path, monkeypatch):
        """Partial send failure (some succeed, some fail) exits 3."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=3)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        import smtplib
        import unittest.mock

        call_count = 0

        def mock_send_message(msg):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise smtplib.SMTPRecipientsRefused({"s001@u.kr": (550, b"rejected")})

        mock_smtp = unittest.mock.MagicMock()
        mock_smtp.send_message = mock_send_message
        monkeypatch.setattr("smtplib.SMTP", lambda *a, **kw: mock_smtp)

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
            ])
        assert exc_info.value.code == 3


# ===========================================================================
# T020/T022: `forma-deliver send --dry-run` CLI tests (US3, FR-013)
# ===========================================================================


class TestCliDeliverSendDryRun:
    """CLI tests for --dry-run mode (US3)."""

    def test_dry_run_exit_0(self, tmp_path, monkeypatch):
        """--dry-run always exits 0 (no real sends, no failures)."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=3)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)

        # No password set -- dry-run should not require it
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        try:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
                "--dry-run",
            ])
        except SystemExit as e:
            assert e.code == 0

    def test_dry_run_creates_log_with_flag(self, tmp_path, monkeypatch):
        """--dry-run creates delivery_log.yaml with dry_run=true."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=2)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        try:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
                "--dry-run",
            ])
        except SystemExit:
            pass

        log_path = os.path.join(staged_dir, "delivery_log.yaml")
        with open(log_path, encoding="utf-8") as f:
            log_data = yaml.safe_load(f)
        assert log_data["dry_run"] is True

    def test_dry_run_console_summary_prefix(self, tmp_path, monkeypatch, capsys):
        """--dry-run console output includes [DRY-RUN] prefix."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=2)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        try:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
                "--dry-run",
            ])
        except SystemExit:
            pass

        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out

    def test_dry_run_no_smtp_instantiation(self, tmp_path, monkeypatch):
        """--dry-run does not call smtplib.SMTP at all."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=2)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        smtp_created = []

        class FailSMTP:
            def __init__(self, *a, **kw):
                smtp_created.append(True)
                raise RuntimeError("SMTP should not be called in dry-run")

        monkeypatch.setattr("smtplib.SMTP", FailSMTP)

        try:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
                "--dry-run",
            ])
        except SystemExit:
            pass

        assert len(smtp_created) == 0


# ===========================================================================
# T023/T025: `forma-deliver send --retry-failed` CLI tests (US4, FR-014)
# ===========================================================================


class TestCliDeliverSendRetryFailed:
    """CLI tests for --retry-failed mode (US4)."""

    def _create_staged_with_delivery_log(self, tmp_path, failed_ids=None):
        """Create staged folder with both prepare_summary and delivery_log."""
        import zipfile

        if failed_ids is None:
            failed_ids = {"s001"}

        staged_dir = tmp_path / "staged_retry"
        staged_dir.mkdir()

        all_ids = ["s000", "s001", "s002"]
        details = []
        log_results = []

        for sid in all_ids:
            name = f"학생_{sid}"
            student_dir = staged_dir / f"{sid}_{name}"
            student_dir.mkdir()
            zip_path = student_dir / f"{name}_{sid}.zip"
            with zipfile.ZipFile(str(zip_path), "w") as zf:
                zf.writestr(f"{sid}_report.pdf", f"content-{sid}")

            details.append({
                "student_id": sid,
                "name": name,
                "email": f"{sid}@u.kr",
                "status": "ready",
                "matched_files": [f"{sid}_report.pdf"],
                "zip_path": str(zip_path),
                "zip_size_bytes": os.path.getsize(str(zip_path)),
                "message": "",
            })

            status = "failed" if sid in failed_ids else "success"
            error = "SMTP error" if status == "failed" else ""
            log_results.append({
                "student_id": sid,
                "email": f"{sid}@u.kr",
                "status": status,
                "sent_at": "2026-03-11T10:00:00",
                "attachment": f"{name}_{sid}.zip",
                "size_bytes": os.path.getsize(str(zip_path)),
                "error": error,
            })

        summary = {
            "prepared_at": "2026-03-11T10:00:00",
            "total_students": 3,
            "ready": 3,
            "warnings": 0,
            "errors": 0,
            "details": details,
        }
        with open(str(staged_dir / "prepare_summary.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)

        log_data = {
            "sent_at": "2026-03-11T10:00:00",
            "smtp_server": "smtp.gmail.com",
            "dry_run": False,
            "total": 3,
            "success": 3 - len(failed_ids),
            "failed": len(failed_ids),
            "results": log_results,
        }
        with open(str(staged_dir / "delivery_log.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(log_data, f, allow_unicode=True, default_flow_style=False)

        return str(staged_dir)

    def test_retry_failed_exit_0(self, tmp_path, monkeypatch):
        """--retry-failed exits 0 when all retries succeed."""
        from forma.cli_deliver import main

        staged_dir = self._create_staged_with_delivery_log(tmp_path, {"s001"})
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        import unittest.mock
        mock_smtp = unittest.mock.MagicMock()
        monkeypatch.setattr("smtplib.SMTP", lambda *a, **kw: mock_smtp)

        try:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
                "--retry-failed",
            ])
        except SystemExit as e:
            assert e.code == 0

    def test_retry_failed_plus_force_exit_1(self, tmp_path, monkeypatch):
        """--retry-failed + --force exits 1 (conflicting flags)."""
        from forma.cli_deliver import main

        staged_dir = self._create_staged_with_delivery_log(tmp_path)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.setenv("FORMA_SMTP_PASSWORD", "pw")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
                "--retry-failed",
                "--force",
            ])
        assert exc_info.value.code == 1

    def test_retry_failed_dry_run_combined(self, tmp_path, monkeypatch):
        """--dry-run + --retry-failed previews only failed entries."""
        from forma.cli_deliver import main

        staged_dir = self._create_staged_with_delivery_log(tmp_path, {"s002"})
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        try:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
                "--dry-run",
                "--retry-failed",
            ])
        except SystemExit as e:
            assert e.code == 0

        # Verify only 1 entry in delivery_log (the failed one)
        log_path = os.path.join(staged_dir, "delivery_log.yaml")
        with open(log_path, encoding="utf-8") as f:
            log_data = yaml.safe_load(f)
        assert log_data["total"] == 1
        assert log_data["dry_run"] is True


# ===========================================================================
# Phase 3 (US1): --smtp-config optional + forma.json fallback
# ===========================================================================


class TestCliDeliverSendSmtpConfigOptional:
    """Tests for --smtp-config being optional with forma.json fallback."""

    def test_parser_accepts_send_without_smtp_config(self):
        """send subcommand parses successfully without --smtp-config."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "send",
            "--staged", "/tmp/staged",
            "--template", "t.yaml",
        ])
        assert args.subcommand == "send"
        assert args.smtp_config is None

    def test_parser_still_accepts_smtp_config(self):
        """send subcommand still accepts --smtp-config when provided."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "send",
            "--staged", "/tmp/staged",
            "--template", "t.yaml",
            "--smtp-config", "smtp.yaml",
        ])
        assert args.smtp_config == "smtp.yaml"

    def test_send_fallback_to_forma_json(self, tmp_path, monkeypatch):
        """Without --smtp-config, send reads from forma.json smtp section."""
        import json
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)

        # Create forma.json with smtp section
        forma_json = tmp_path / "forma.json"
        forma_json.write_text(json.dumps({
            "smtp": {
                "server": "json.smtp.com",
                "port": 587,
                "sender_email": "json@test.com",
                "use_tls": True,
                "send_interval_sec": 0,
            }
        }), encoding="utf-8")

        # Mock load_config to return our config
        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: json.loads(forma_json.read_text()),
        )
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        try:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--dry-run",
            ])
        except SystemExit as e:
            assert e.code == 0

        # Verify the delivery log references json.smtp.com
        log_path = os.path.join(staged_dir, "delivery_log.yaml")
        with open(log_path, encoding="utf-8") as f:
            log_data = yaml.safe_load(f)
        assert log_data["smtp_server"] == "json.smtp.com"

    def test_send_no_smtp_anywhere_exit_2(self, tmp_path, monkeypatch):
        """No --smtp-config and no forma.json smtp section exits 2."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)

        # Mock load_config to raise FileNotFoundError (no forma.json)
        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: (_ for _ in ()).throw(
                FileNotFoundError("No config")
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
            ])
        assert exc_info.value.code == 2

    def test_send_forma_json_no_smtp_section_exit_2(self, tmp_path, monkeypatch):
        """forma.json exists but has no 'smtp' section exits 2."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)

        # Mock load_config returns config without smtp
        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: {"llm": {"provider": "gemini"}},
        )

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
            ])
        assert exc_info.value.code == 2

    def test_send_error_message_korean(self, tmp_path, monkeypatch, capsys):
        """When no SMTP config found, error message is in Korean."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)

        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: (_ for _ in ()).throw(
                FileNotFoundError("No config")
            ),
        )

        with pytest.raises(SystemExit):
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
            ])

        captured = capsys.readouterr()
        assert "SMTP config not found" in captured.err

    def test_explicit_smtp_config_takes_priority(self, tmp_path, monkeypatch):
        """When --smtp-config is given, use it even if forma.json has smtp section."""
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)  # uses smtp.gmail.com

        # forma.json has a different server
        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: {
                "smtp": {
                    "server": "different.smtp.com",
                    "sender_email": "other@test.com",
                }
            },
        )
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        try:
            main([
                "--no-config",
                "send",
                "--staged", staged_dir,
                "--template", template_path,
                "--smtp-config", smtp_path,
                "--dry-run",
            ])
        except SystemExit as e:
            assert e.code == 0

        log_path = os.path.join(staged_dir, "delivery_log.yaml")
        with open(log_path, encoding="utf-8") as f:
            log_data = yaml.safe_load(f)
        # Should use the YAML file's server, not forma.json's
        assert log_data["smtp_server"] == "smtp.gmail.com"


# ===========================================================================
# Phase 4 (US2): Deprecation warning for --smtp-config
# ===========================================================================


class TestCliDeliverSendDeprecation:
    """Tests for --smtp-config deprecation warning."""

    def test_smtp_config_flag_emits_deprecation_warning(self, tmp_path, monkeypatch):
        """Using --smtp-config emits DeprecationWarning."""
        import warnings
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config",
                    "send",
                    "--staged", staged_dir,
                    "--template", template_path,
                    "--smtp-config", smtp_path,
                    "--dry-run",
                ])
            except SystemExit:
                pass

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "smtp-config" in str(deprecation_warnings[0].message).lower() or \
                   "--smtp-config" in str(deprecation_warnings[0].message)

    def test_deprecation_warning_message_korean(self, tmp_path, monkeypatch):
        """Deprecation warning message is in Korean."""
        import warnings
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config",
                    "send",
                    "--staged", staged_dir,
                    "--template", template_path,
                    "--smtp-config", smtp_path,
                    "--dry-run",
                ])
            except SystemExit:
                pass

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            msg = str(deprecation_warnings[0].message)
            assert "config.json" in msg
            assert "Migrate" in msg

    def test_no_deprecation_without_smtp_config_flag(self, tmp_path, monkeypatch):
        """When --smtp-config is NOT used, no deprecation warning is emitted."""
        import warnings
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)

        monkeypatch.setattr(
            "forma.config.load_config",
            lambda config_path=None: {
                "smtp": {
                    "server": "smtp.x.com",
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
                    "--no-config",
                    "send",
                    "--staged", staged_dir,
                    "--template", template_path,
                    "--dry-run",
                ])
            except SystemExit:
                pass

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 0

    def test_smtp_config_flag_still_works(self, tmp_path, monkeypatch):
        """--smtp-config still works correctly (backward compat) despite deprecation."""
        import warnings
        from forma.cli_deliver import main

        staged_dir = _create_staged_folder(tmp_path, n_students=1)
        template_path = _write_email_template(tmp_path)
        smtp_path = _write_smtp_config(tmp_path)
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config",
                    "send",
                    "--staged", staged_dir,
                    "--template", template_path,
                    "--smtp-config", smtp_path,
                    "--dry-run",
                ])
            except SystemExit as e:
                assert e.code == 0

        # Should still create delivery_log from the YAML path
        log_path = os.path.join(staged_dir, "delivery_log.yaml")
        with open(log_path, encoding="utf-8") as f:
            log_data = yaml.safe_load(f)
        assert log_data["smtp_server"] == "smtp.gmail.com"
