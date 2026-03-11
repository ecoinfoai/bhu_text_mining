"""Tests for delivery_prepare.py -- delivery preparation module.

T003: RED tests for DeliveryManifest, StudentEntry, StudentRoster,
      load_manifest(), load_roster() with validation edge cases.
T007: RED tests for match_files_for_student(), create_student_zip(),
      prepare_delivery().

Covers FR-001 ~ FR-005, FR-015, FR-016, FR-020, FR-021.
"""

from __future__ import annotations

import os
import textwrap

import pytest
import yaml


# ---------------------------------------------------------------------------
# T003: DeliveryManifest dataclass tests
# ---------------------------------------------------------------------------


class TestDeliveryManifest:
    """Tests for DeliveryManifest dataclass."""

    def test_create_manifest(self):
        """DeliveryManifest stores directory and file_patterns."""
        from forma.delivery_prepare import DeliveryManifest

        m = DeliveryManifest(
            directory="/reports/week4",
            file_patterns=["{student_id}_report.pdf"],
        )
        assert m.directory == "/reports/week4"
        assert m.file_patterns == ["{student_id}_report.pdf"]

    def test_manifest_is_frozen(self):
        """DeliveryManifest should be immutable (frozen dataclass)."""
        from forma.delivery_prepare import DeliveryManifest

        m = DeliveryManifest(
            directory="/reports",
            file_patterns=["{student_id}.pdf"],
        )
        with pytest.raises(AttributeError):
            m.directory = "/other"

    def test_manifest_multiple_patterns(self):
        """DeliveryManifest can have multiple file_patterns."""
        from forma.delivery_prepare import DeliveryManifest

        m = DeliveryManifest(
            directory="/out",
            file_patterns=[
                "{student_id}_report.pdf",
                "{student_id}_feedback.pdf",
            ],
        )
        assert len(m.file_patterns) == 2


# ---------------------------------------------------------------------------
# T003: StudentEntry dataclass tests
# ---------------------------------------------------------------------------


class TestStudentEntry:
    """Tests for StudentEntry dataclass."""

    def test_create_entry(self):
        """StudentEntry stores student_id, name, email."""
        from forma.delivery_prepare import StudentEntry

        e = StudentEntry(student_id="2024001", name="홍길동", email="hong@univ.ac.kr")
        assert e.student_id == "2024001"
        assert e.name == "홍길동"
        assert e.email == "hong@univ.ac.kr"

    def test_entry_is_frozen(self):
        """StudentEntry should be immutable (frozen dataclass)."""
        from forma.delivery_prepare import StudentEntry

        e = StudentEntry(student_id="s1", name="Test", email="t@e.com")
        with pytest.raises(AttributeError):
            e.name = "Other"


# ---------------------------------------------------------------------------
# T003: StudentRoster dataclass tests
# ---------------------------------------------------------------------------


class TestStudentRoster:
    """Tests for StudentRoster dataclass."""

    def test_create_roster(self):
        """StudentRoster stores class_name and students list."""
        from forma.delivery_prepare import StudentEntry, StudentRoster

        roster = StudentRoster(
            class_name="해부생리학 1A",
            students=[
                StudentEntry("s1", "홍길동", "hong@u.kr"),
                StudentEntry("s2", "김철수", "kim@u.kr"),
            ],
        )
        assert roster.class_name == "해부생리학 1A"
        assert len(roster.students) == 2

    def test_roster_is_frozen(self):
        """StudentRoster should be immutable."""
        from forma.delivery_prepare import StudentRoster

        roster = StudentRoster(class_name="A", students=[])
        with pytest.raises(AttributeError):
            roster.class_name = "B"


# ---------------------------------------------------------------------------
# T003: load_manifest() tests
# ---------------------------------------------------------------------------


class TestLoadManifest:
    """Tests for load_manifest() YAML loader with validation."""

    def test_load_valid_manifest(self, tmp_path):
        """load_manifest() returns DeliveryManifest from valid YAML."""
        from forma.delivery_prepare import load_manifest

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text(
            textwrap.dedent(f"""\
                report_source:
                  directory: "{report_dir}"
                  file_patterns:
                    - "{{student_id}}_report.pdf"
            """),
            encoding="utf-8",
        )

        m = load_manifest(str(manifest_file))
        assert m.directory == str(report_dir)
        assert m.file_patterns == ["{student_id}_report.pdf"]

    def test_load_manifest_missing_file(self, tmp_path):
        """load_manifest() raises FileNotFoundError for missing file."""
        from forma.delivery_prepare import load_manifest

        with pytest.raises(FileNotFoundError):
            load_manifest(str(tmp_path / "nonexistent.yaml"))

    def test_load_manifest_missing_directory_field(self, tmp_path):
        """load_manifest() raises ValueError when directory field is missing."""
        from forma.delivery_prepare import load_manifest

        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text(
            textwrap.dedent("""\
                report_source:
                  file_patterns:
                    - "{student_id}_report.pdf"
            """),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="directory"):
            load_manifest(str(manifest_file))

    def test_load_manifest_missing_file_patterns(self, tmp_path):
        """load_manifest() raises ValueError when file_patterns is missing."""
        from forma.delivery_prepare import load_manifest

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text(
            textwrap.dedent(f"""\
                report_source:
                  directory: "{report_dir}"
            """),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="file_patterns"):
            load_manifest(str(manifest_file))

    def test_load_manifest_empty_file_patterns(self, tmp_path):
        """load_manifest() raises ValueError when file_patterns is empty list."""
        from forma.delivery_prepare import load_manifest

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text(
            textwrap.dedent(f"""\
                report_source:
                  directory: "{report_dir}"
                  file_patterns: []
            """),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="file_patterns"):
            load_manifest(str(manifest_file))

    def test_load_manifest_nonexistent_directory(self, tmp_path):
        """load_manifest() raises ValueError when directory does not exist."""
        from forma.delivery_prepare import load_manifest

        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text(
            textwrap.dedent("""\
                report_source:
                  directory: "/nonexistent/path"
                  file_patterns:
                    - "{student_id}_report.pdf"
            """),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="directory"):
            load_manifest(str(manifest_file))

    def test_load_manifest_pattern_without_student_id(self, tmp_path):
        """load_manifest() raises ValueError when pattern lacks {student_id}."""
        from forma.delivery_prepare import load_manifest

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text(
            textwrap.dedent(f"""\
                report_source:
                  directory: "{report_dir}"
                  file_patterns:
                    - "report.pdf"
            """),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="student_id"):
            load_manifest(str(manifest_file))


# ---------------------------------------------------------------------------
# T003: load_roster() tests
# ---------------------------------------------------------------------------


class TestLoadRoster:
    """Tests for load_roster() YAML loader with validation."""

    def test_load_valid_roster(self, tmp_path):
        """load_roster() returns StudentRoster from valid YAML."""
        from forma.delivery_prepare import load_roster

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text(
            textwrap.dedent("""\
                class_name: "해부생리학 1A"
                students:
                  - student_id: "2024001"
                    name: "홍길동"
                    email: "hong@univ.ac.kr"
                  - student_id: "2024002"
                    name: "김철수"
                    email: "kim@univ.ac.kr"
            """),
            encoding="utf-8",
        )

        roster = load_roster(str(roster_file))
        assert roster.class_name == "해부생리학 1A"
        assert len(roster.students) == 2
        assert roster.students[0].student_id == "2024001"
        assert roster.students[0].name == "홍길동"

    def test_load_roster_missing_file(self, tmp_path):
        """load_roster() raises FileNotFoundError for missing file."""
        from forma.delivery_prepare import load_roster

        with pytest.raises(FileNotFoundError):
            load_roster(str(tmp_path / "nonexistent.yaml"))

    def test_load_roster_duplicate_student_id(self, tmp_path):
        """load_roster() raises ValueError for duplicate student_id (FR-016)."""
        from forma.delivery_prepare import load_roster

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text(
            textwrap.dedent("""\
                class_name: "A"
                students:
                  - student_id: "s1"
                    name: "학생1"
                    email: "s1@u.kr"
                  - student_id: "s1"
                    name: "학생1복제"
                    email: "s1dup@u.kr"
            """),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="중복.*s1|duplicate.*s1"):
            load_roster(str(roster_file))

    def test_load_roster_empty_email(self, tmp_path):
        """load_roster() accepts empty email (FR-021: checked in prepare_delivery)."""
        from forma.delivery_prepare import load_roster

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text(
            textwrap.dedent("""\
                class_name: "A"
                students:
                  - student_id: "s1"
                    name: "학생1"
                    email: ""
            """),
            encoding="utf-8",
        )

        roster = load_roster(str(roster_file))
        assert len(roster.students) == 1
        assert roster.students[0].email == ""

    def test_load_roster_missing_email_field(self, tmp_path):
        """load_roster() accepts missing email (FR-021: checked in prepare_delivery)."""
        from forma.delivery_prepare import load_roster

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text(
            textwrap.dedent("""\
                class_name: "A"
                students:
                  - student_id: "s1"
                    name: "학생1"
            """),
            encoding="utf-8",
        )

        roster = load_roster(str(roster_file))
        assert len(roster.students) == 1
        assert roster.students[0].email == ""

    def test_load_roster_missing_student_id(self, tmp_path):
        """load_roster() raises ValueError for missing student_id."""
        from forma.delivery_prepare import load_roster

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text(
            textwrap.dedent("""\
                class_name: "A"
                students:
                  - name: "학생1"
                    email: "s1@u.kr"
            """),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="student_id"):
            load_roster(str(roster_file))

    def test_load_roster_empty_students(self, tmp_path):
        """load_roster() raises ValueError when students list is empty."""
        from forma.delivery_prepare import load_roster

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text(
            textwrap.dedent("""\
                class_name: "A"
                students: []
            """),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="students"):
            load_roster(str(roster_file))

    def test_load_roster_missing_class_name(self, tmp_path):
        """load_roster() raises ValueError when class_name is missing."""
        from forma.delivery_prepare import load_roster

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text(
            textwrap.dedent("""\
                students:
                  - student_id: "s1"
                    name: "학생1"
                    email: "s1@u.kr"
            """),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="class_name"):
            load_roster(str(roster_file))

    def test_load_roster_email_without_at(self, tmp_path):
        """load_roster() accepts invalid email (FR-021: checked in prepare_delivery)."""
        from forma.delivery_prepare import load_roster

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text(
            textwrap.dedent("""\
                class_name: "A"
                students:
                  - student_id: "s1"
                    name: "학생1"
                    email: "invalid-email"
            """),
            encoding="utf-8",
        )

        roster = load_roster(str(roster_file))
        assert len(roster.students) == 1
        assert roster.students[0].email == "invalid-email"


# ---------------------------------------------------------------------------
# T003: sanitize_filename() tests
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    """Tests for sanitize_filename() utility."""

    def test_sanitize_normal_filename(self):
        """Normal filenames pass through unchanged."""
        from forma.delivery_prepare import sanitize_filename

        assert sanitize_filename("report_2024001.pdf") == "report_2024001.pdf"

    def test_sanitize_korean_filename(self):
        """Korean filenames are preserved."""
        from forma.delivery_prepare import sanitize_filename

        assert sanitize_filename("홍길동_보고서.pdf") == "홍길동_보고서.pdf"

    def test_sanitize_illegal_characters(self):
        """OS-illegal characters are removed."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename('file<>:"/\\|?*.pdf')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result
        assert "/" not in result
        assert "\\" not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result
        # Extension preserved
        assert result.endswith(".pdf")


# ---------------------------------------------------------------------------
# T007: match_files_for_student() tests
# ---------------------------------------------------------------------------


class TestMatchFilesForStudent:
    """Tests for match_files_for_student()."""

    def test_full_match_all_patterns(self, tmp_path):
        """All file_patterns match for a student → list of full paths."""
        from forma.delivery_prepare import DeliveryManifest, match_files_for_student

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        (report_dir / "s001_report.pdf").write_bytes(b"PDF1")
        (report_dir / "s001_feedback.pdf").write_bytes(b"PDF2")

        manifest = DeliveryManifest(
            directory=str(report_dir),
            file_patterns=["{student_id}_report.pdf", "{student_id}_feedback.pdf"],
        )
        matched = match_files_for_student(manifest, "s001")
        assert len(matched) == 2
        assert any("s001_report.pdf" in p for p in matched)
        assert any("s001_feedback.pdf" in p for p in matched)

    def test_partial_match(self, tmp_path):
        """Only some patterns match → partial list."""
        from forma.delivery_prepare import DeliveryManifest, match_files_for_student

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        (report_dir / "s001_report.pdf").write_bytes(b"PDF1")
        # No s001_feedback.pdf

        manifest = DeliveryManifest(
            directory=str(report_dir),
            file_patterns=["{student_id}_report.pdf", "{student_id}_feedback.pdf"],
        )
        matched = match_files_for_student(manifest, "s001")
        assert len(matched) == 1
        assert any("s001_report.pdf" in p for p in matched)

    def test_no_match(self, tmp_path):
        """No patterns match → empty list."""
        from forma.delivery_prepare import DeliveryManifest, match_files_for_student

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        manifest = DeliveryManifest(
            directory=str(report_dir),
            file_patterns=["{student_id}_report.pdf"],
        )
        matched = match_files_for_student(manifest, "s999")
        assert matched == []


# ---------------------------------------------------------------------------
# T007: create_student_zip() tests
# ---------------------------------------------------------------------------


class TestCreateStudentZip:
    """Tests for create_student_zip()."""

    def test_create_zip_success(self, tmp_path):
        """Creates a zip containing the matched files."""
        import zipfile

        from forma.delivery_prepare import create_student_zip

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        f1 = report_dir / "s001_report.pdf"
        f1.write_bytes(b"PDF_CONTENT_1")

        staging_dir = tmp_path / "staging" / "s001_홍길동"
        staging_dir.mkdir(parents=True)

        zip_path = create_student_zip(
            matched_files=[str(f1)],
            staging_dir=str(staging_dir),
            student_id="s001",
            student_name="홍길동",
        )

        assert os.path.exists(zip_path)
        assert zip_path.endswith(".zip")
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert len(names) == 1
            assert "s001_report.pdf" in names[0]

    def test_zip_with_multiple_files(self, tmp_path):
        """Zip contains all matched files."""
        import zipfile

        from forma.delivery_prepare import create_student_zip

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        f1 = report_dir / "s001_report.pdf"
        f1.write_bytes(b"PDF1")
        f2 = report_dir / "s001_feedback.pdf"
        f2.write_bytes(b"PDF2")

        staging_dir = tmp_path / "staging" / "s001"
        staging_dir.mkdir(parents=True)

        zip_path = create_student_zip(
            matched_files=[str(f1), str(f2)],
            staging_dir=str(staging_dir),
            student_id="s001",
            student_name="학생",
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert len(zf.namelist()) == 2


# ---------------------------------------------------------------------------
# T007: save_prepare_summary() tests
# ---------------------------------------------------------------------------


class TestSavePrepareSupmmary:
    """Tests for save_prepare_summary()."""

    def test_save_and_load_summary(self, tmp_path):
        """save_prepare_summary() writes YAML that can be parsed back."""
        from forma.delivery_prepare import (
            PrepareSummary,
            StudentPrepareResult,
            save_prepare_summary,
        )

        summary = PrepareSummary(
            prepared_at="2026-03-11T10:00:00",
            class_name="테스트반",
            total_students=3,
            ready=2,
            warnings=1,
            errors=0,
            details=[
                StudentPrepareResult(
                    student_id="s1", name="학생1", email="s1@u.kr",
                    status="ready", matched_files=["r.pdf"],
                    zip_path="/out/s1.zip", zip_size_bytes=1024,
                ),
                StudentPrepareResult(
                    student_id="s2", name="학생2", email="s2@u.kr",
                    status="ready", matched_files=["r.pdf"],
                    zip_path="/out/s2.zip", zip_size_bytes=2048,
                ),
                StudentPrepareResult(
                    student_id="s3", name="학생3", email="s3@u.kr",
                    status="warning", matched_files=["r.pdf"],
                    zip_path="/out/s3.zip", zip_size_bytes=512,
                    message="feedback.pdf 누락",
                ),
            ],
        )

        out_path = tmp_path / "prepare_summary.yaml"
        save_prepare_summary(summary, str(out_path))

        assert os.path.exists(out_path)
        with open(out_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["total_students"] == 3
        assert data["ready"] == 2
        assert data["warnings"] == 1
        assert len(data["details"]) == 3


# ---------------------------------------------------------------------------
# T007: prepare_delivery() orchestrator tests
# ---------------------------------------------------------------------------


class TestPrepareDelivery:
    """Tests for prepare_delivery() orchestrator."""

    def _make_test_fixtures(self, tmp_path, n_students=3, missing_files=None):
        """Helper: create manifest YAML, roster YAML, and report files.

        Args:
            tmp_path: Pytest tmp_path fixture.
            n_students: Number of students to create.
            missing_files: Set of student indices (0-based) for which to skip
                creating the report file.

        Returns:
            (manifest_path, roster_path, output_dir)
        """
        if missing_files is None:
            missing_files = set()

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        # Create report files
        for i in range(n_students):
            if i not in missing_files:
                (report_dir / f"s{i:03d}_report.pdf").write_bytes(
                    b"PDF_CONTENT_" + str(i).encode()
                )

        # Manifest
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            f"report_source:\n"
            f'  directory: "{report_dir}"\n'
            f"  file_patterns:\n"
            f'    - "{{student_id}}_report.pdf"\n',
            encoding="utf-8",
        )

        # Roster
        roster_path = tmp_path / "roster.yaml"
        students_yaml = ""
        for i in range(n_students):
            students_yaml += (
                f'  - student_id: "s{i:03d}"\n'
                f'    name: "학생{i}"\n'
                f'    email: "s{i:03d}@u.kr"\n'
            )
        roster_path.write_text(
            f'class_name: "테스트반"\n'
            f"students:\n{students_yaml}",
            encoding="utf-8",
        )

        output_dir = tmp_path / "staging"

        return str(manifest_path), str(roster_path), str(output_dir)

    def test_prepare_all_ready(self, tmp_path):
        """All students matched → all ready, summary correct."""
        from forma.delivery_prepare import prepare_delivery

        manifest_path, roster_path, output_dir = self._make_test_fixtures(
            tmp_path, n_students=3,
        )

        summary = prepare_delivery(manifest_path, roster_path, output_dir)

        assert summary.total_students == 3
        assert summary.ready == 3
        assert summary.warnings == 0
        assert summary.errors == 0
        assert len(summary.details) == 3
        # All details should be ready
        for d in summary.details:
            assert d.status == "ready"
            assert d.zip_path is not None
            assert os.path.exists(d.zip_path)

    def test_prepare_with_missing_files(self, tmp_path):
        """Students with no matching files → error status."""
        from forma.delivery_prepare import prepare_delivery

        manifest_path, roster_path, output_dir = self._make_test_fixtures(
            tmp_path, n_students=3, missing_files={2},
        )

        summary = prepare_delivery(manifest_path, roster_path, output_dir)

        assert summary.total_students == 3
        assert summary.ready == 2
        assert summary.errors == 1
        # details includes ALL students (C1 fix)
        assert len(summary.details) == 3
        error_results = [d for d in summary.details if d.status == "error"]
        assert len(error_results) == 1
        assert error_results[0].student_id == "s002"

    def test_prepare_partial_match_warning(self, tmp_path):
        """Student has some but not all patterns matched → warning."""
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        (report_dir / "s001_report.pdf").write_bytes(b"PDF1")
        (report_dir / "s001_feedback.pdf").write_bytes(b"PDF2")
        # s002 only has report, not feedback
        (report_dir / "s002_report.pdf").write_bytes(b"PDF3")

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            f"report_source:\n"
            f'  directory: "{report_dir}"\n'
            f"  file_patterns:\n"
            f'    - "{{student_id}}_report.pdf"\n'
            f'    - "{{student_id}}_feedback.pdf"\n',
            encoding="utf-8",
        )

        roster_path = tmp_path / "roster.yaml"
        roster_path.write_text(
            'class_name: "A"\n'
            "students:\n"
            '  - student_id: "s001"\n'
            '    name: "학생1"\n'
            '    email: "s1@u.kr"\n'
            '  - student_id: "s002"\n'
            '    name: "학생2"\n'
            '    email: "s2@u.kr"\n',
            encoding="utf-8",
        )

        output_dir = str(tmp_path / "staging")
        summary = prepare_delivery(str(manifest_path), str(roster_path), output_dir)

        assert summary.ready == 1
        assert summary.warnings == 1
        warning_results = [d for d in summary.details if d.status == "warning"]
        assert len(warning_results) == 1
        assert warning_results[0].student_id == "s002"
        # Warning student still gets a zip
        assert warning_results[0].zip_path is not None

    def test_prepare_staging_already_exists(self, tmp_path):
        """Staging dir already exists without --force → raises."""
        from forma.delivery_prepare import prepare_delivery

        manifest_path, roster_path, output_dir = self._make_test_fixtures(
            tmp_path, n_students=1,
        )
        os.makedirs(output_dir)

        with pytest.raises((FileExistsError, ValueError)):
            prepare_delivery(manifest_path, roster_path, output_dir, force=False)

    def test_prepare_staging_exists_with_force(self, tmp_path):
        """Staging dir exists with --force → overwrites."""
        from forma.delivery_prepare import prepare_delivery

        manifest_path, roster_path, output_dir = self._make_test_fixtures(
            tmp_path, n_students=1,
        )
        os.makedirs(output_dir)
        # Place a leftover file
        (tmp_path / "staging" / "old.txt").write_text("old")

        summary = prepare_delivery(manifest_path, roster_path, output_dir, force=True)
        assert summary.ready == 1

    def test_prepare_summary_yaml_created(self, tmp_path):
        """prepare_delivery() writes prepare_summary.yaml in output_dir."""
        from forma.delivery_prepare import prepare_delivery

        manifest_path, roster_path, output_dir = self._make_test_fixtures(
            tmp_path, n_students=2,
        )

        prepare_delivery(manifest_path, roster_path, output_dir)

        summary_path = os.path.join(output_dir, "prepare_summary.yaml")
        assert os.path.exists(summary_path)
        with open(summary_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["total_students"] == 2

    def test_prepare_zip_size_limit(self, tmp_path):
        """Zip exceeding 25MB → error status (FR-015)."""
        import random

        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        # Create incompressible random data slightly over 25MB
        rng = random.Random(42)
        large_content = rng.randbytes(26 * 1024 * 1024)
        (report_dir / "s001_report.pdf").write_bytes(large_content)

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            f"report_source:\n"
            f'  directory: "{report_dir}"\n'
            f"  file_patterns:\n"
            f'    - "{{student_id}}_report.pdf"\n',
            encoding="utf-8",
        )

        roster_path = tmp_path / "roster.yaml"
        roster_path.write_text(
            'class_name: "A"\n'
            "students:\n"
            '  - student_id: "s001"\n'
            '    name: "학생1"\n'
            '    email: "s1@u.kr"\n',
            encoding="utf-8",
        )

        output_dir = str(tmp_path / "staging")
        summary = prepare_delivery(str(manifest_path), str(roster_path), output_dir)

        assert summary.errors == 1
        error_results = [d for d in summary.details if d.status == "error"]
        assert len(error_results) == 1
        assert "25MB" in error_results[0].message or "25" in error_results[0].message


# ===========================================================================
# T030: ADVERSARIAL EDGE CASE TESTS (delivery_prepare)
# ===========================================================================


# ---------------------------------------------------------------------------
# Persona 2: Boundary Pusher — extreme input values
# ---------------------------------------------------------------------------


class TestAdversaryBoundaryPusherPrepare:
    """Persona 2: Extreme input values that push system limits."""

    def test_1000_students_roster(self, tmp_path):
        """1,000-student roster: system must handle without OOM or crash."""
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        # Create 1000 student PDFs
        for i in range(1000):
            (report_dir / f"s{i:04d}_report.pdf").write_bytes(b"PDF" + str(i).encode())

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            f"report_source:\n"
            f'  directory: "{report_dir}"\n'
            f"  file_patterns:\n"
            f'    - "{{student_id}}_report.pdf"\n',
            encoding="utf-8",
        )

        students_yaml = ""
        for i in range(1000):
            students_yaml += (
                f'  - student_id: "s{i:04d}"\n'
                f'    name: "학생{i}"\n'
                f'    email: "s{i:04d}@u.kr"\n'
            )
        roster_path = tmp_path / "roster.yaml"
        roster_path.write_text(
            f'class_name: "대규모반"\nstudents:\n{students_yaml}',
            encoding="utf-8",
        )

        output_dir = str(tmp_path / "staging")
        summary = prepare_delivery(str(manifest_path), str(roster_path), output_dir)

        assert summary.total_students == 1000
        assert summary.ready == 1000
        assert summary.errors == 0

    def test_empty_student_id_in_roster_rejected(self, tmp_path):
        """Empty string student_id should be rejected by load_roster()."""
        from forma.delivery_prepare import load_roster

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text(
            'class_name: "A"\n'
            "students:\n"
            '  - student_id: ""\n'
            '    name: "학생"\n'
            '    email: "s@u.kr"\n',
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="student_id"):
            load_roster(str(roster_file))

    def test_student_name_with_os_illegal_chars(self, tmp_path):
        """Student name with OS-illegal chars: sanitize_filename removes them."""
        from forma.delivery_prepare import sanitize_filename

        # All OS-illegal chars in one name
        dirty_name = '<홍길동>:kim"john/park\\jane|test?file*name'
        clean = sanitize_filename(dirty_name)

        for ch in '<>:"/\\|?*':
            assert ch not in clean
        # Korean characters survive
        assert "홍길동" in clean

    def test_file_pattern_without_student_id_rejected(self, tmp_path):
        """file_patterns entry without {student_id} is rejected (FR-001)."""
        from forma.delivery_prepare import load_manifest

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text(
            f"report_source:\n"
            f'  directory: "{report_dir}"\n'
            f"  file_patterns:\n"
            f'    - "report.pdf"\n',
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="student_id"):
            load_manifest(str(manifest_file))

    def test_zip_exactly_25mb_boundary(self, tmp_path):
        """Zip exactly 25MB: should be accepted (limit is > 25MB)."""
        from forma.delivery_prepare import _MAX_ZIP_BYTES

        # Verify the boundary constant
        assert _MAX_ZIP_BYTES == 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# Persona 3: Malicious Input — path traversal, injection
# ---------------------------------------------------------------------------


class TestAdversaryMaliciousInputPrepare:
    """Persona 3: Security attacks — path traversal, sanitize bypass."""

    def test_path_traversal_in_student_id(self, tmp_path):
        """student_id with '../../../etc' must NOT escape staging directory.

        EXPLOIT CHECK: sanitize_filename() only removes OS-illegal chars
        but does NOT strip '..' components. However, the directory structure
        uses student_id in os.path.join, and match_files_for_student uses
        it in pattern replacement. The zip itself is created inside a
        staging subdirectory based on sanitized student_id + name.
        """
        from forma.delivery_prepare import sanitize_filename

        # Path traversal payload
        malicious_id = "../../../etc/passwd"
        sanitized = sanitize_filename(malicious_id)

        # sanitize_filename removes '/' but '..' dots remain
        assert "/" not in sanitized
        # '../../../etc/passwd' → '......etcpasswd' (6 dots: 3 pairs of '..')
        assert sanitized == "......etcpasswd"

    def test_path_traversal_in_student_name_staging_dir(self, tmp_path):
        """Student name with path traversal in prepare_delivery staging dir.

        The staging subdirectory name is sanitize_filename(student_id + '_' + name).
        Verify it cannot escape the output directory.
        """
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        (report_dir / "s001_report.pdf").write_bytes(b"PDF1")

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            f"report_source:\n"
            f'  directory: "{report_dir}"\n'
            f"  file_patterns:\n"
            f'    - "{{student_id}}_report.pdf"\n',
            encoding="utf-8",
        )

        # Student name with path traversal attempt
        roster_path = tmp_path / "roster.yaml"
        roster_path.write_text(
            'class_name: "A"\n'
            "students:\n"
            '  - student_id: "s001"\n'
            '    name: "../../etc/shadow"\n'
            '    email: "s@u.kr"\n',
            encoding="utf-8",
        )

        output_dir = str(tmp_path / "staging")
        summary = prepare_delivery(str(manifest_path), str(roster_path), output_dir)

        # Check that all created files are inside the staging directory
        for detail in summary.details:
            if detail.zip_path:
                # Resolve to absolute path and verify containment
                abs_zip = os.path.realpath(detail.zip_path)
                abs_staging = os.path.realpath(output_dir)
                assert abs_zip.startswith(abs_staging), (
                    f"EXPLOIT: zip escaped staging dir: {abs_zip}"
                )

    def test_student_id_with_null_bytes(self, tmp_path):
        """Student ID with null bytes in filename: should be handled safely."""
        from forma.delivery_prepare import sanitize_filename

        # Null byte attack
        result = sanitize_filename("s001\x00evil")
        # Null bytes should at least not cause crashes
        assert isinstance(result, str)

    def test_manifest_yaml_bomb(self, tmp_path):
        """YAML bomb (billion laughs) in manifest: yaml.safe_load blocks it."""
        from forma.delivery_prepare import load_manifest

        # YAML alias bomb
        yaml_bomb = tmp_path / "bomb.yaml"
        yaml_bomb.write_text(
            'a: &a ["lol","lol"]\n'
            'b: &b [*a,*a]\n'
            'c: &c [*b,*b]\n'
            'report_source:\n'
            '  directory: "/tmp"\n'
            '  file_patterns:\n'
            '    - "{student_id}.pdf"\n',
            encoding="utf-8",
        )

        # yaml.safe_load should handle this safely (no exponential expansion
        # in safe_load for reasonable nesting levels)
        try:
            load_manifest(str(yaml_bomb))
        except (ValueError, FileNotFoundError):
            pass  # Expected: directory validation will fail, but no crash

    def test_duplicate_email_warning_not_error(self, tmp_path):
        """Same email on different student_ids: allowed per spec (edge case 3)."""
        from forma.delivery_prepare import load_roster

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text(
            'class_name: "A"\n'
            "students:\n"
            '  - student_id: "s001"\n'
            '    name: "홍길동"\n'
            '    email: "same@u.kr"\n'
            '  - student_id: "s002"\n'
            '    name: "김철수"\n'
            '    email: "same@u.kr"\n'
            '  - student_id: "s003"\n'
            '    name: "이영희"\n'
            '    email: "same@u.kr"\n',
            encoding="utf-8",
        )

        # Should not raise (duplicate email is allowed, only duplicate ID is error)
        roster = load_roster(str(roster_file))
        assert len(roster.students) == 3


# ---------------------------------------------------------------------------
# Persona 5: Data Mangler — corrupted data
# ---------------------------------------------------------------------------


class TestAdversaryDataManglerPrepare:
    """Persona 5: Corrupted YAML, incomplete data in prepare stage."""

    def test_manifest_yaml_indentation_error(self, tmp_path):
        """YAML with indentation errors should raise parse error."""
        from forma.delivery_prepare import load_manifest

        manifest_file = tmp_path / "bad_manifest.yaml"
        manifest_file.write_text(
            "report_source:\n"
            "directory: /reports\n"  # Missing indentation
            "  file_patterns:\n"
            '    - "{student_id}.pdf"\n',
            encoding="utf-8",
        )

        # Should raise ValueError (not a valid structure) or YAMLError
        with pytest.raises((ValueError, yaml.YAMLError)):
            load_manifest(str(manifest_file))

    def test_roster_non_dict_yaml(self, tmp_path):
        """Roster YAML that parses to a list instead of dict."""
        from forma.delivery_prepare import load_roster

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="형식"):
            load_roster(str(roster_file))

    def test_roster_whitespace_only_email(self, tmp_path):
        """Email field containing only whitespace → student marked as error in prepare_delivery.

        FR-021: Invalid email is handled per-student in prepare_delivery(), not in
        load_roster(). The roster loads successfully; prepare_delivery records the
        student as status='error' and continues with the rest.
        """
        from forma.delivery_prepare import load_roster, prepare_delivery

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text(
            'class_name: "A"\n'
            "students:\n"
            '  - student_id: "s001"\n'
            '    name: "학생1"\n'
            '    email: "   "\n',
            encoding="utf-8",
        )

        # load_roster succeeds — email validation deferred to prepare_delivery
        roster = load_roster(str(roster_file))
        assert len(roster.students) == 1
        assert roster.students[0].email.strip() == ""

        # prepare_delivery marks the student as error, does not raise
        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text(
            "report_source:\n"
            f"  directory: {str(tmp_path)}\n"
            "  file_patterns:\n"
            '    - "{student_id}_report.pdf"\n',
            encoding="utf-8",
        )
        output_dir = tmp_path / "staging"
        summary = prepare_delivery(
            str(manifest_file), str(roster_file), str(output_dir)
        )
        assert summary.errors == 1
        assert summary.details[0].status == "error"
        assert summary.details[0].student_id == "s001"

    def test_manifest_empty_yaml(self, tmp_path):
        """Empty YAML file for manifest should raise ValueError."""
        from forma.delivery_prepare import load_manifest

        manifest_file = tmp_path / "empty.yaml"
        manifest_file.write_text("", encoding="utf-8")

        with pytest.raises((ValueError, AttributeError)):
            load_manifest(str(manifest_file))

    def test_roster_student_id_as_integer(self, tmp_path):
        """student_id as integer in YAML (not quoted) should be coerced to str."""
        from forma.delivery_prepare import load_roster

        roster_file = tmp_path / "roster.yaml"
        roster_file.write_text(
            'class_name: "A"\n'
            "students:\n"
            "  - student_id: 2024001\n"  # Not quoted → YAML int
            '    name: "학생"\n'
            '    email: "s@u.kr"\n',
            encoding="utf-8",
        )

        roster = load_roster(str(roster_file))
        # Must be string after loading
        assert isinstance(roster.students[0].student_id, str)
        assert roster.students[0].student_id == "2024001"
