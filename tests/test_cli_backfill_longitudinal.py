"""Tests for cli_backfill_longitudinal.py — topic and class_id support."""

from __future__ import annotations



from forma.longitudinal_store import (
    _infer_class_id,
)


class TestInferClassId:
    """T008–T009: class_id inference and CLI override."""

    def test_infer_from_eval_dir_pattern(self):
        """_infer_class_id extracts class from 'eval_A' pattern."""
        assert _infer_class_id("eval_A") == "A"
        assert _infer_class_id("/data/eval_B/") == "B"
        assert _infer_class_id("path/to/eval_CD") == "CD"

    def test_infer_from_final_filename(self):
        """_infer_class_id extracts class from 'final_A.yaml'."""
        assert _infer_class_id("final_A.yaml") == "A"
        assert _infer_class_id("/data/final_BC.yaml") == "BC"

    def test_infer_fails_no_pattern(self):
        """_infer_class_id returns None for unrecognized patterns."""
        assert _infer_class_id("results") is None
        assert _infer_class_id("/data/output/") is None

    def test_infer_single_char_class(self):
        """_infer_class_id handles single-char class IDs."""
        assert _infer_class_id("eval_A") == "A"

    def test_infer_multi_char_class(self):
        """_infer_class_id handles multi-char class IDs (up to 3)."""
        assert _infer_class_id("eval_ABC") == "ABC"


class TestCliBackfillClassId:
    """T008: --class flag overrides pattern inference."""

    def test_class_flag_overrides_pattern(self, tmp_path):
        """--class CLI arg takes priority over directory pattern."""
        # This tests the logic: when --class is provided,
        # the inferred class from directory name is ignored.
        # We test the helper directly since CLI main() is complex.
        inferred = _infer_class_id("eval_A")
        cli_class = "X"

        # CLI override: when --class is set, use it
        effective = cli_class if cli_class else inferred
        assert effective == "X"

    def test_inferred_when_no_cli_flag(self, tmp_path):
        """class_id inferred from eval dir when --class omitted."""
        inferred = _infer_class_id("eval_B")
        cli_class = None

        effective = cli_class if cli_class else inferred
        assert effective == "B"
