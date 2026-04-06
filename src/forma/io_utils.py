"""Atomic file write utilities for YAML, JSON, and text files.

Provides crash-safe atomic writes using the tempfile + os.replace pattern.
Optional advisory file locking via fcntl.flock for concurrent access.

Public API:
    atomic_write_yaml(data, path, *, lock=False)
    atomic_write_json(data, path, *, lock=False, indent=2)
    atomic_write_text(content, path, *, lock=False)
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def safe_filename(name: str) -> str:
    """Sanitize a string for safe use as a filename component.

    Args:
        name: Raw string (e.g. student ID from YAML).

    Returns:
        Sanitized string with path separators, parent-directory
        references, and null bytes removed.
    """
    safe = name.replace("/", "_").replace("\\", "_").replace("..", "_")
    safe = safe.replace("\x00", "")
    return safe or "unknown"


def _atomic_write(
    write_fn,
    path: str | Path,
    *,
    lock: bool = False,
) -> None:
    """Internal helper: atomic write with optional locking.

    Args:
        write_fn: Callable(file_object) that writes content.
        path: Target file path.
        lock: If True, acquire exclusive flock during replace.
    """
    path = str(path)
    parent = os.path.dirname(path) or "."
    tmp_path = None
    lock_file = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())

        if lock:
            lock_path = path + ".lock"
            lock_file = open(lock_path, "a")
            fcntl.flock(lock_file, fcntl.LOCK_EX)

        os.replace(tmp_path, path)
        tmp_path = None  # success — prevent cleanup

        if lock and lock_file is not None:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    finally:
        if lock_file is not None:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except OSError:
                pass
            lock_file.close()
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def atomic_write_yaml(
    data: Any,
    path: str | Path,
    *,
    lock: bool = False,
) -> None:
    """Write data to a YAML file atomically.

    Uses tempfile.mkstemp + os.replace to ensure either the old file
    or the new file exists -- never a partial write.

    Args:
        data: Python object serializable by PyYAML.
        path: Target file path.
        lock: If True, acquire an exclusive fcntl.flock on a
              companion .lock file during the replace operation.

    Raises:
        OSError: If the write or lock operation fails.
        TypeError: If data is not YAML-serializable.
    """

    def _write(f):
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    _atomic_write(_write, path, lock=lock)


def atomic_write_json(
    data: Any,
    path: str | Path,
    *,
    lock: bool = False,
    indent: int = 2,
) -> None:
    """Write data to a JSON file atomically.

    Args:
        data: Python object serializable by json.
        path: Target file path.
        lock: If True, acquire exclusive flock during replace.
        indent: JSON indentation level.

    Raises:
        OSError: If the write or lock operation fails.
        TypeError: If data is not JSON-serializable.
    """

    def _write(f):
        json.dump(data, f, ensure_ascii=False, indent=indent)

    _atomic_write(_write, path, lock=lock)


def atomic_write_text(
    content: str,
    path: str | Path,
    *,
    lock: bool = False,
) -> None:
    """Write text content to a file atomically.

    Args:
        content: String to write.
        path: Target file path.
        lock: If True, acquire exclusive flock during replace.

    Raises:
        OSError: If the write or lock operation fails.
    """

    def _write(f):
        f.write(content)

    _atomic_write(_write, path, lock=lock)
