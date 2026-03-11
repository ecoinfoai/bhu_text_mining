"""Delivery prepare module -- collect student report files and create zip archives.

Handles manifest/roster parsing, file matching, zip creation, and staging
folder management for the email delivery pipeline.

FR Coverage: FR-001 ~ FR-005, FR-015, FR-016, FR-020, FR-021.
"""

from __future__ import annotations

import fcntl
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "DeliveryManifest",
    "StudentEntry",
    "StudentRoster",
    "StudentPrepareResult",
    "PrepareSummary",
    "load_manifest",
    "load_roster",
    "sanitize_filename",
    "match_files_for_student",
    "create_student_zip",
    "prepare_delivery",
    "save_prepare_summary",
]

# Characters illegal in filenames across major OSes
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*]')


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeliveryManifest:
    """Report source manifest defining file locations and patterns.

    Args:
        directory: Path to the report file directory.
        file_patterns: List of filename patterns containing ``{student_id}``.
    """

    directory: str
    file_patterns: list[str]


@dataclass(frozen=True)
class StudentEntry:
    """A single student entry in the roster.

    Args:
        student_id: Unique student identifier.
        name: Student name.
        email: Student email address.
    """

    student_id: str
    name: str
    email: str


@dataclass(frozen=True)
class StudentRoster:
    """Student roster for a class section.

    Args:
        class_name: Name of the class section.
        students: List of student entries.
    """

    class_name: str
    students: list[StudentEntry]


@dataclass
class StudentPrepareResult:
    """Per-student result from the prepare stage.

    Args:
        student_id: Student identifier.
        name: Student name.
        email: Student email address.
        status: One of ``"ready"``, ``"warning"``, ``"error"``.
        matched_files: List of matched report file paths.
        zip_path: Path to the generated zip file, or None on error.
        zip_size_bytes: Size of the zip file in bytes.
        message: Warning/error message (empty string if ready).
    """

    student_id: str
    name: str
    email: str
    status: str
    matched_files: list[str] = field(default_factory=list)
    zip_path: str | None = None
    zip_size_bytes: int = 0
    message: str = ""


@dataclass
class PrepareSummary:
    """Summary of the prepare stage.

    Args:
        prepared_at: ISO 8601 timestamp of preparation.
        class_name: Name of the class section (from roster).
        total_students: Total number of students in roster.
        ready: Count of students with status ``"ready"``.
        warnings: Count of students with status ``"warning"``.
        errors: Count of students with status ``"error"``.
        details: Per-student results (ready + warning + error).
    """

    prepared_at: str
    class_name: str
    total_students: int
    ready: int
    warnings: int
    errors: int
    details: list[StudentPrepareResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# YAML Loaders
# ---------------------------------------------------------------------------


def load_manifest(path: str) -> DeliveryManifest:
    """Load a delivery manifest from a YAML file.

    Args:
        path: Path to the manifest YAML file.

    Returns:
        Parsed ``DeliveryManifest``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required fields are missing or invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"매니페스트 파일을 찾을 수 없습니다: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    source = data.get("report_source") if isinstance(data, dict) else None
    if not isinstance(source, dict):
        raise ValueError("매니페스트에 'report_source' 섹션이 필요합니다.")

    directory = source.get("directory")
    if not directory:
        raise ValueError("매니페스트에 'directory' 필드가 필요합니다.")

    if not os.path.isdir(directory):
        raise ValueError(f"directory가 존재하지 않습니다: {directory}")

    file_patterns = source.get("file_patterns")
    if not file_patterns:
        raise ValueError("매니페스트에 'file_patterns' 필드가 필요합니다 (1개 이상).")

    for pat in file_patterns:
        if "{student_id}" not in pat:
            raise ValueError(
                f"file_patterns의 각 패턴에 '{{student_id}}'가 포함되어야 합니다: {pat}"
            )

    return DeliveryManifest(directory=str(directory), file_patterns=list(file_patterns))


def load_roster(path: str) -> StudentRoster:
    """Load a student roster from a YAML file.

    Args:
        path: Path to the roster YAML file.

    Returns:
        Parsed ``StudentRoster``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required fields are missing, duplicated, or invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"명부 파일을 찾을 수 없습니다: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("명부 파일 형식이 올바르지 않습니다.")

    class_name = data.get("class_name")
    if not class_name:
        raise ValueError("명부에 'class_name' 필드가 필요합니다.")

    students_raw = data.get("students")
    if not students_raw:
        raise ValueError("명부에 'students' 목록이 필요합니다 (1명 이상).")

    seen_ids: set[str] = set()
    students: list[StudentEntry] = []

    for i, entry in enumerate(students_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"학생 항목 {i + 1}이 올바른 형식이 아닙니다.")

        sid = entry.get("student_id")
        if not sid:
            raise ValueError(f"학생 항목 {i + 1}에 'student_id' 필드가 필요합니다.")

        sid = str(sid)
        if sid in seen_ids:
            raise ValueError(f"student_id 중복: {sid}")
        seen_ids.add(sid)

        name = entry.get("name")
        if not name:
            raise ValueError(f"학생 '{sid}'에 'name' 필드가 필요합니다.")

        # email may be empty/missing -- FR-021: handled in prepare_delivery()
        email = str(entry.get("email") or "")

        students.append(StudentEntry(student_id=sid, name=str(name), email=email))

    return StudentRoster(class_name=str(class_name), students=students)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


_MAX_FILENAME_BYTES = 200  # conservative limit (ext4 allows 255)


def sanitize_filename(name: str) -> str:
    """Remove OS-illegal characters and truncate to safe byte length.

    Args:
        name: Original filename.

    Returns:
        Sanitized filename with illegal characters removed and
        byte length limited to ``_MAX_FILENAME_BYTES``.
    """
    result = _ILLEGAL_CHARS.sub("", name)
    # Truncate to safe byte length (char-by-char to avoid mid-char cut)
    encoded = result.encode("utf-8")
    if len(encoded) > _MAX_FILENAME_BYTES:
        while len(result.encode("utf-8")) > _MAX_FILENAME_BYTES:
            result = result[:-1]
    return result


# Maximum zip size in bytes (25 MB, FR-015)
_MAX_ZIP_BYTES = 25 * 1024 * 1024


def match_files_for_student(
    manifest: DeliveryManifest, student_id: str,
) -> list[str]:
    """Match report files for a student based on manifest patterns.

    Args:
        manifest: Delivery manifest with directory and file patterns.
        student_id: Student identifier to substitute into patterns.

    Returns:
        List of full file paths that exist on disk.
    """
    matched: list[str] = []
    for pattern in manifest.file_patterns:
        filename = pattern.replace("{student_id}", student_id)
        full_path = os.path.join(manifest.directory, filename)
        if os.path.isfile(full_path):
            matched.append(full_path)
    return matched


def create_student_zip(
    matched_files: list[str],
    staging_dir: str,
    student_id: str,
    student_name: str,
) -> str:
    """Create a zip archive of matched files for a student.

    Args:
        matched_files: List of file paths to include.
        staging_dir: Directory where the zip will be created.
        student_id: Student identifier.
        student_name: Student name (used in zip filename).

    Returns:
        Path to the created zip file.
    """
    import zipfile

    zip_name = sanitize_filename(f"{student_name}_{student_id}.zip")
    zip_path = os.path.join(staging_dir, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in matched_files:
            arcname = os.path.basename(fpath)
            zf.write(fpath, arcname)

    return zip_path


def save_prepare_summary(summary: PrepareSummary, path: str) -> None:
    """Save a prepare summary to a YAML file.

    Args:
        summary: The prepare summary to save.
        path: Output file path.
    """
    details_list = []
    for d in summary.details:
        detail = {
            "student_id": d.student_id,
            "name": d.name,
            "email": d.email,
            "status": d.status,
            "matched_files": d.matched_files,
            "zip_path": d.zip_path,
            "zip_size_bytes": d.zip_size_bytes,
            "message": d.message,
        }
        details_list.append(detail)

    data = {
        "prepared_at": summary.prepared_at,
        "class_name": summary.class_name,
        "total_students": summary.total_students,
        "ready": summary.ready,
        "warnings": summary.warnings,
        "errors": summary.errors,
        "details": details_list,
    }

    path_obj = os.path.abspath(path)
    dir_name = os.path.dirname(path_obj)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        os.replace(tmp_path, path_obj)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def prepare_delivery(
    manifest_path: str,
    roster_path: str,
    output_dir: str,
    force: bool = False,
) -> PrepareSummary:
    """Orchestrate the full prepare stage.

    Loads manifest and roster, matches files per student, creates zip
    archives, and writes prepare_summary.yaml.

    Args:
        manifest_path: Path to the delivery manifest YAML.
        roster_path: Path to the student roster YAML.
        output_dir: Path for the staging folder output.
        force: If ``True``, overwrite existing staging folder.

    Returns:
        The ``PrepareSummary`` with all student results.

    Raises:
        FileNotFoundError: If manifest or roster files do not exist.
        ValueError: If manifest/roster validation fails.
        FileExistsError: If output_dir exists and force is ``False``.
    """
    from datetime import datetime, timezone

    manifest = load_manifest(manifest_path)
    roster = load_roster(roster_path)

    # Handle existing staging folder (FR-020)
    if os.path.exists(output_dir):
        if not force:
            raise FileExistsError(
                f"출력 폴더가 이미 존재합니다: {output_dir}\n"
                "--force 플래그로 덮어쓸 수 있습니다."
            )
        import shutil
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    details: list[StudentPrepareResult] = []

    for student in roster.students:
        # FR-021: skip students with empty or invalid email
        if not student.email or "@" not in student.email:
            details.append(StudentPrepareResult(
                student_id=student.student_id,
                name=student.name,
                email=student.email,
                status="error",
                message="email 누락 또는 형식 오류",
            ))
            continue

        # Match files
        matched = match_files_for_student(manifest, student.student_id)
        n_patterns = len(manifest.file_patterns)

        if not matched:
            # No files found at all → error
            details.append(StudentPrepareResult(
                student_id=student.student_id,
                name=student.name,
                email=student.email,
                status="error",
                message="매칭 파일 없음",
            ))
            continue

        # Create student staging directory
        dir_name = sanitize_filename(f"{student.student_id}_{student.name}")
        student_dir = os.path.join(output_dir, dir_name)
        os.makedirs(student_dir, exist_ok=True)

        # Create zip
        zip_path = create_student_zip(
            matched_files=matched,
            staging_dir=student_dir,
            student_id=student.student_id,
            student_name=student.name,
        )
        zip_size = os.path.getsize(zip_path)

        # Check zip size limit (FR-015)
        if zip_size > _MAX_ZIP_BYTES:
            os.remove(zip_path)
            details.append(StudentPrepareResult(
                student_id=student.student_id,
                name=student.name,
                email=student.email,
                status="error",
                matched_files=matched,
                message=f"zip 크기 초과 (25MB 제한): {zip_size} bytes",
            ))
            continue

        # Determine status based on match completeness
        if len(matched) < n_patterns:
            status = "warning"
            missing_count = n_patterns - len(matched)
            message = f"{missing_count}개 패턴 미매칭"
        else:
            status = "ready"
            message = ""

        details.append(StudentPrepareResult(
            student_id=student.student_id,
            name=student.name,
            email=student.email,
            status=status,
            matched_files=matched,
            zip_path=zip_path,
            zip_size_bytes=zip_size,
            message=message,
        ))

    ready_count = sum(1 for d in details if d.status == "ready")
    warning_count = sum(1 for d in details if d.status == "warning")
    error_count = sum(1 for d in details if d.status == "error")

    summary = PrepareSummary(
        prepared_at=datetime.now(timezone.utc).isoformat(),
        class_name=roster.class_name,
        total_students=len(roster.students),
        ready=ready_count,
        warnings=warning_count,
        errors=error_count,
        details=details,
    )

    # Save summary YAML
    summary_path = os.path.join(output_dir, "prepare_summary.yaml")
    save_prepare_summary(summary, summary_path)

    logger.info(
        "준비 완료: 전체 %d명 (ready=%d, warning=%d, error=%d)",
        summary.total_students, ready_count, warning_count, error_count,
    )

    return summary
