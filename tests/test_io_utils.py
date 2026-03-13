"""Tests for io_utils module -- atomic file write utilities.

Covers:
- atomic_write_yaml(): normal write, error recovery, Unicode, empty data, non-serializable
- atomic_write_json(): normal write, indent parameter, error recovery
- atomic_write_text(): normal write, Unicode, error recovery
- Lock mode (lock=True): verify locking behavior
- Cleanup: verify no temp files remain after error
"""

from __future__ import annotations

import json
import os
import threading

import pytest
import yaml

from forma.io_utils import atomic_write_json, atomic_write_text, atomic_write_yaml


# ---------------------------------------------------------------------------
# atomic_write_yaml
# ---------------------------------------------------------------------------


class TestAtomicWriteYaml:
    """Tests for atomic_write_yaml()."""

    def test_normal_write(self, tmp_path):
        """Write a dict and verify YAML content on disk."""
        path = tmp_path / "out.yaml"
        data = {"key": "value", "number": 42}
        atomic_write_yaml(data, path)

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == data

    def test_unicode_content(self, tmp_path):
        """Korean and emoji characters are preserved via allow_unicode."""
        path = tmp_path / "unicode.yaml"
        data = {"name": "홍길동", "desc": "형성평가 분석"}
        atomic_write_yaml(data, path)

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == data

    def test_empty_dict(self, tmp_path):
        """Empty dict produces a valid YAML file."""
        path = tmp_path / "empty.yaml"
        atomic_write_yaml({}, path)

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == {} or loaded is None  # yaml.safe_load({}) may return None

    def test_empty_list(self, tmp_path):
        """Empty list produces a valid YAML file."""
        path = tmp_path / "empty_list.yaml"
        atomic_write_yaml([], path)

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == [] or loaded is None

    def test_no_temp_file_after_os_error(self, tmp_path):
        """No .tmp files remain in tmp_path after a write to missing parent dir."""
        path = tmp_path / "nonexistent_sub" / "bad.yaml"
        before_count = len([f for f in os.listdir(tmp_path) if f.endswith(".tmp")])
        with pytest.raises(OSError):
            atomic_write_yaml({"a": 1}, path)
        after_count = len([f for f in os.listdir(tmp_path) if f.endswith(".tmp")])
        assert after_count == before_count

    def test_error_recovery_missing_parent_dir(self, tmp_path):
        """Writing to a non-existent parent directory raises OSError."""
        path = tmp_path / "nonexistent" / "out.yaml"
        with pytest.raises(OSError):
            atomic_write_yaml({"a": 1}, path)

    def test_overwrites_existing_file(self, tmp_path):
        """Overwrites an existing file atomically."""
        path = tmp_path / "out.yaml"
        atomic_write_yaml({"v": 1}, path)
        atomic_write_yaml({"v": 2}, path)

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == {"v": 2}

    def test_list_data(self, tmp_path):
        """Top-level list data is properly serialized."""
        path = tmp_path / "list.yaml"
        data = [1, "two", {"three": 3}]
        atomic_write_yaml(data, path)

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == data

    def test_nested_structure(self, tmp_path):
        """Nested dicts and lists are preserved."""
        path = tmp_path / "nested.yaml"
        data = {"a": {"b": [1, 2, {"c": "deep"}]}}
        atomic_write_yaml(data, path)

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == data


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    """Tests for atomic_write_json()."""

    def test_normal_write(self, tmp_path):
        """Write a dict and verify JSON content on disk."""
        path = tmp_path / "out.json"
        data = {"key": "value", "number": 42}
        atomic_write_json(data, path)

        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_indent_parameter(self, tmp_path):
        """Custom indent value is respected in output."""
        path = tmp_path / "indented.json"
        data = {"a": 1}
        atomic_write_json(data, path, indent=4)

        raw = path.read_text(encoding="utf-8")
        # 4-space indent produces "    " prefix for nested keys
        assert '    "a"' in raw

    def test_default_indent_is_2(self, tmp_path):
        """Default indent is 2 spaces."""
        path = tmp_path / "default.json"
        data = {"a": 1}
        atomic_write_json(data, path)

        raw = path.read_text(encoding="utf-8")
        assert '  "a"' in raw

    def test_non_serializable_raises_type_error(self, tmp_path):
        """Non-JSON-serializable data raises TypeError."""
        path = tmp_path / "bad.json"
        with pytest.raises(TypeError):
            atomic_write_json({"s": {1, 2, 3}}, path)

    def test_no_temp_file_after_json_error(self, tmp_path):
        """No .tmp files remain in directory after a JSON serialization error."""
        path = tmp_path / "bad.json"
        before = set(os.listdir(tmp_path))
        with pytest.raises(Exception):
            atomic_write_json({"s": {1, 2, 3}}, path)
        after = set(os.listdir(tmp_path))
        leftover = {f for f in (after - before) if f.endswith(".tmp")}
        assert leftover == set()

    def test_error_recovery_missing_parent_dir(self, tmp_path):
        """Writing to a non-existent parent directory raises OSError."""
        path = tmp_path / "nonexistent" / "out.json"
        with pytest.raises(OSError):
            atomic_write_json({"a": 1}, path)

    def test_unicode_json(self, tmp_path):
        """Korean characters are preserved (ensure_ascii=False)."""
        path = tmp_path / "korean.json"
        data = {"name": "홍길동"}
        atomic_write_json(data, path)

        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data
        # Verify raw file has actual Korean chars, not \\uXXXX
        raw = path.read_text(encoding="utf-8")
        assert "홍길동" in raw


# ---------------------------------------------------------------------------
# atomic_write_text
# ---------------------------------------------------------------------------


class TestAtomicWriteText:
    """Tests for atomic_write_text()."""

    def test_normal_write(self, tmp_path):
        """Write a string and verify file content."""
        path = tmp_path / "out.txt"
        content = "Hello, world!"
        atomic_write_text(content, path)
        assert path.read_text(encoding="utf-8") == content

    def test_unicode_text(self, tmp_path):
        """Korean text is preserved."""
        path = tmp_path / "korean.txt"
        content = "형성평가 분석 도구"
        atomic_write_text(content, path)
        assert path.read_text(encoding="utf-8") == content

    def test_empty_string(self, tmp_path):
        """Empty string creates a file with empty content."""
        path = tmp_path / "empty.txt"
        atomic_write_text("", path)
        assert path.read_text(encoding="utf-8") == ""

    def test_error_recovery_missing_parent_dir(self, tmp_path):
        """Writing to a non-existent parent raises OSError."""
        path = tmp_path / "nonexistent" / "out.txt"
        with pytest.raises(OSError):
            atomic_write_text("hello", path)

    def test_no_temp_file_after_error(self, tmp_path):
        """No .tmp files remain after a write error."""
        path = tmp_path / "nonexistent" / "out.txt"
        # Parent doesn't exist, so mkstemp will fail or replace will fail
        before_count = len([f for f in os.listdir(tmp_path) if f.endswith(".tmp")])
        with pytest.raises(Exception):
            atomic_write_text("hello", path)
        after_count = len([f for f in os.listdir(tmp_path) if f.endswith(".tmp")])
        assert after_count == before_count

    def test_overwrites_existing_file(self, tmp_path):
        """Overwrites an existing file."""
        path = tmp_path / "out.txt"
        atomic_write_text("first", path)
        atomic_write_text("second", path)
        assert path.read_text(encoding="utf-8") == "second"


# ---------------------------------------------------------------------------
# Locking (lock=True)
# ---------------------------------------------------------------------------


class TestAtomicWriteLocking:
    """Tests for lock=True behavior."""

    def test_yaml_with_lock(self, tmp_path):
        """atomic_write_yaml(lock=True) creates file successfully."""
        path = tmp_path / "locked.yaml"
        data = {"locked": True}
        atomic_write_yaml(data, path, lock=True)

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == data

    def test_json_with_lock(self, tmp_path):
        """atomic_write_json(lock=True) creates file successfully."""
        path = tmp_path / "locked.json"
        data = {"locked": True}
        atomic_write_json(data, path, lock=True)

        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_text_with_lock(self, tmp_path):
        """atomic_write_text(lock=True) creates file successfully."""
        path = tmp_path / "locked.txt"
        atomic_write_text("locked content", path, lock=True)
        assert path.read_text(encoding="utf-8") == "locked content"

    def test_lock_file_created(self, tmp_path):
        """Lock file (.lock) is created when lock=True."""
        path = tmp_path / "out.yaml"
        atomic_write_yaml({"a": 1}, path, lock=True)
        lock_path = str(path) + ".lock"
        assert os.path.exists(lock_path)

    def test_concurrent_writes_with_lock(self, tmp_path):
        """Concurrent writes with lock=True produce valid final file."""
        path = tmp_path / "concurrent.yaml"
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    atomic_write_yaml({"writer": n, "iter": i}, path, lock=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Final file must be valid YAML
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert "writer" in loaded
        assert "iter" in loaded


# ---------------------------------------------------------------------------
# Path types (str and Path)
# ---------------------------------------------------------------------------


class TestPathTypes:
    """Verify both str and Path types work for the path parameter."""

    def test_yaml_with_string_path(self, tmp_path):
        """String path works for atomic_write_yaml."""
        path = str(tmp_path / "str.yaml")
        atomic_write_yaml({"a": 1}, path)
        with open(path, encoding="utf-8") as f:
            assert yaml.safe_load(f) == {"a": 1}

    def test_yaml_with_path_object(self, tmp_path):
        """pathlib.Path works for atomic_write_yaml."""
        path = tmp_path / "pathobj.yaml"
        atomic_write_yaml({"a": 1}, path)
        with open(path, encoding="utf-8") as f:
            assert yaml.safe_load(f) == {"a": 1}

    def test_json_with_string_path(self, tmp_path):
        """String path works for atomic_write_json."""
        path = str(tmp_path / "str.json")
        atomic_write_json({"a": 1}, path)
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"a": 1}

    def test_text_with_string_path(self, tmp_path):
        """String path works for atomic_write_text."""
        path = str(tmp_path / "str.txt")
        atomic_write_text("hello", path)
        with open(path, encoding="utf-8") as f:
            assert f.read() == "hello"


# ---------------------------------------------------------------------------
# Error recovery with lock=True
# ---------------------------------------------------------------------------


class TestLockErrorRecovery:
    """Verify lock files and temp files are cleaned up on write errors with lock=True."""

    def test_no_temp_file_after_write_error_with_lock(self, tmp_path):
        """No .tmp files remain after a write error even with lock=True."""
        from forma.io_utils import _atomic_write

        path = tmp_path / "fail.yaml"
        before = set(os.listdir(tmp_path))

        def _bad_write(f):
            f.write("partial data")
            raise RuntimeError("simulated write failure")

        with pytest.raises(RuntimeError, match="simulated write failure"):
            _atomic_write(_bad_write, path, lock=True)

        after = set(os.listdir(tmp_path))
        leftover = {f for f in (after - before) if f.endswith(".tmp")}
        assert leftover == set()

    def test_existing_file_preserved_on_write_error(self, tmp_path):
        """Existing file content is not corrupted when a write attempt fails."""
        from forma.io_utils import _atomic_write

        path = tmp_path / "preserve.yaml"
        atomic_write_yaml({"original": True}, path)

        def _bad_write(f):
            raise RuntimeError("simulated write failure")

        with pytest.raises(RuntimeError, match="simulated write failure"):
            _atomic_write(_bad_write, path)

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == {"original": True}

    def test_large_data_roundtrip(self, tmp_path):
        """Large data (1000+ keys) survives atomic write/read cycle."""
        path = tmp_path / "large.yaml"
        data = {f"key_{i}": f"value_{i}" for i in range(1000)}
        atomic_write_yaml(data, path, lock=True)

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == data
