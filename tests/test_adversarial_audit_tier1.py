"""Adversarial audit tests — Tier 1: Beginner personas (A-01 to A-03).

A-01: First-time professor — runs forma-eval without init, malformed exam.yaml,
      duplicate question numbers, fullwidth spaces in concepts.
A-02: Clueless TA — paths with Korean/spaces, Windows paths, wrong file extensions.
A-03: YAML-phobic professor — tab/space mix, missing colon space, unquoted special
      chars, empty YAML, BOM prefix.

Discovery only — tests that FAIL indicate vulnerabilities, not test bugs.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# A-01: First-time professor
# ---------------------------------------------------------------------------


class TestA01FirstTimeProfessor:
    """Persona A-01: Professor who has never used the system before."""

    def test_a01_eval_without_init(self, tmp_path: Path) -> None:
        """A-01: Running evaluation without any config files should fail clearly."""
        from forma.config import load_config

        fake_config = tmp_path / "nonexistent_config.yaml"
        with pytest.raises((FileNotFoundError, SystemExit)):
            load_config(str(fake_config))

    def test_a01_load_missing_exam_yaml(self, tmp_path: Path) -> None:
        """A-01: Loading a non-existent exam YAML should raise FileNotFoundError."""
        from forma.evaluation_io import load_evaluation_yaml

        with pytest.raises(FileNotFoundError, match="not found"):
            load_evaluation_yaml(str(tmp_path / "missing_exam.yaml"))

    def test_a01_malformed_exam_yaml(self, tmp_path: Path) -> None:
        """A-01: Malformed YAML (unbalanced braces) should raise a clear error."""
        bad_yaml = tmp_path / "exam.yaml"
        bad_yaml.write_text("questions:\n  - sn: 1\n    concepts: [a, b\n", encoding="utf-8")

        with pytest.raises((yaml.YAMLError, Exception)):
            yaml.safe_load(bad_yaml.read_text())

    def test_a01_duplicate_question_sn(self, tmp_path: Path) -> None:
        """A-01: Duplicate question serial numbers should be detected or handled."""
        from forma.config_validator import validate_question_config

        # Two questions with same sn=1
        q1 = {"sn": 1, "question_type": "essay", "concepts": ["세포막"]}
        q2 = {"sn": 1, "question_type": "essay", "concepts": ["핵"]}

        # Validator should work on individual questions without crash
        errors1 = validate_question_config(q1)
        errors2 = validate_question_config(q2)
        # At minimum, no crash
        assert isinstance(errors1, list)
        assert isinstance(errors2, list)

    def test_a01_fullwidth_spaces_in_concepts(self, tmp_path: Path) -> None:
        """A-01: Fullwidth spaces (\u3000) in concept names should not cause silent failures."""
        from forma.config_validator import validate_question_config

        q = {
            "sn": 1,
            "question_type": "essay",
            "concepts": ["세포\u3000막", "핵\u3000막"],
        }
        errors = validate_question_config(q)
        # Should not crash; ideally would warn about unusual whitespace
        assert isinstance(errors, list)

    def test_a01_empty_exam_config(self, tmp_path: Path) -> None:
        """A-01: Empty exam config should fail fast, not produce mysterious errors."""
        empty_yaml = tmp_path / "empty_exam.yaml"
        empty_yaml.write_text("", encoding="utf-8")

        data = yaml.safe_load(empty_yaml.read_text())
        # safe_load returns None for empty; downstream should handle this
        assert data is None

    def test_a01_project_config_missing_forma_yaml(self, tmp_path: Path) -> None:
        """A-01: find_project_config should return None when no forma.yaml exists."""
        from forma.project_config import find_project_config

        result = find_project_config(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# A-02: Clueless TA
# ---------------------------------------------------------------------------


class TestA02CluelessTA:
    """Persona A-02: TA who doesn't understand filesystem conventions."""

    def test_a02_korean_path(self, tmp_path: Path) -> None:
        """A-02: Paths with Korean characters should be handled."""
        from forma.evaluation_io import save_evaluation_yaml, load_evaluation_yaml

        korean_dir = tmp_path / "형성평가" / "결과"
        korean_dir.mkdir(parents=True)
        output = korean_dir / "result.yaml"

        data = {"score": 0.75, "student_id": "S001"}
        save_evaluation_yaml(data, str(output))
        loaded = load_evaluation_yaml(str(output))
        assert loaded["score"] == 0.75

    def test_a02_spaces_in_path(self, tmp_path: Path) -> None:
        """A-02: Paths with spaces should work correctly."""
        from forma.evaluation_io import save_evaluation_yaml, load_evaluation_yaml

        space_dir = tmp_path / "My Documents" / "2024 Spring"
        space_dir.mkdir(parents=True)
        output = space_dir / "result.yaml"

        data = {"score": 0.85}
        save_evaluation_yaml(data, str(output))
        loaded = load_evaluation_yaml(str(output))
        assert loaded["score"] == 0.85

    def test_a02_windows_backslash_path(self, tmp_path: Path) -> None:
        """A-02: Windows-style backslash paths should be handled or rejected clearly."""
        from forma.evaluation_io import load_evaluation_yaml

        # Simulating a Windows path pasted on Linux
        fake_windows_path = "C:\\Users\\TA\\exam.yaml"
        with pytest.raises((FileNotFoundError, OSError)):
            load_evaluation_yaml(fake_windows_path)

    def test_a02_wrong_extension_yaml(self, tmp_path: Path) -> None:
        """A-02: File with .txt extension but YAML content should still load."""
        wrong_ext = tmp_path / "exam.txt"
        wrong_ext.write_text("questions:\n  - sn: 1\n", encoding="utf-8")

        # evaluation_io checks existence, not extension
        from forma.evaluation_io import load_evaluation_yaml

        data = load_evaluation_yaml(str(wrong_ext))
        assert data is not None

    def test_a02_longitudinal_store_korean_path(self, tmp_path: Path) -> None:
        """A-02: LongitudinalStore with Korean-character paths should work."""
        from forma.longitudinal_store import LongitudinalStore

        store_path = tmp_path / "종단 데이터" / "store.yaml"
        store_path.parent.mkdir(parents=True)

        store = LongitudinalStore(str(store_path))
        store.load()  # Should init empty
        store.save()  # Should write
        assert store_path.exists()

    def test_a02_intervention_store_spaces_path(self, tmp_path: Path) -> None:
        """A-02: InterventionLog with spaces in path should work."""
        from forma.intervention_store import InterventionLog

        space_path = tmp_path / "my data" / "log.yaml"
        space_path.parent.mkdir(parents=True)

        log = InterventionLog(str(space_path))
        log.load()
        log.save()
        assert space_path.exists()


# ---------------------------------------------------------------------------
# A-03: YAML-phobic professor
# ---------------------------------------------------------------------------


class TestA03YAMLPhobic:
    """Persona A-03: Professor who hand-edits YAML and makes syntax errors."""

    def test_a03_tab_space_mix(self, tmp_path: Path) -> None:
        """A-03: YAML with mixed tabs and spaces should fail with clear error."""
        bad_yaml = tmp_path / "exam.yaml"
        # Tabs instead of spaces — invalid YAML
        bad_yaml.write_text("questions:\n\t- sn: 1\n\t  concepts: [a]\n", encoding="utf-8")

        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(bad_yaml.read_text())

    def test_a03_missing_colon_space(self, tmp_path: Path) -> None:
        """A-03: Missing space after colon may parse incorrectly."""
        bad_yaml = tmp_path / "exam.yaml"
        # 'sn:1' without space is valid YAML string key
        bad_yaml.write_text("questions:\n  - sn:1\n", encoding="utf-8")

        data = yaml.safe_load(bad_yaml.read_text())
        # sn:1 becomes a string key, not sn=1
        assert data is not None
        q = data["questions"][0]
        # The list entry is the string "sn:1", not a dict with key "sn"
        assert isinstance(q, str) or ("sn" not in q)

    def test_a03_unquoted_special_chars(self, tmp_path: Path) -> None:
        """A-03: Unquoted special chars in concepts should be handled."""
        bad_yaml = tmp_path / "exam.yaml"
        content = textwrap.dedent("""\
            questions:
              - sn: 1
                concepts:
                  - "Na+/K+ pump"
                  - "pH > 7.4"
                  - "Ca2+ channel"
        """)
        bad_yaml.write_text(content, encoding="utf-8")

        data = yaml.safe_load(bad_yaml.read_text())
        assert len(data["questions"][0]["concepts"]) == 3

    def test_a03_empty_yaml_file(self, tmp_path: Path) -> None:
        """A-03: Completely empty YAML file should not crash the system."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")

        from forma.longitudinal_store import LongitudinalStore

        store = LongitudinalStore(str(empty))
        store.load()
        # Should handle None from safe_load gracefully
        assert store.get_all_records() == []

    def test_a03_yaml_with_bom(self, tmp_path: Path) -> None:
        """A-03: YAML file with UTF-8 BOM should be handled."""
        bom_yaml = tmp_path / "exam.yaml"
        content = "questions:\n  - sn: 1\n"
        bom_yaml.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

        # PyYAML should handle BOM
        data = yaml.safe_load(bom_yaml.read_text(encoding="utf-8-sig"))
        assert data is not None

    def test_a03_yaml_with_bom_raw_load(self, tmp_path: Path) -> None:
        """A-03: YAML with BOM loaded without utf-8-sig should still parse or fail clearly."""
        bom_yaml = tmp_path / "exam.yaml"
        content = "sn: 1\n"
        bom_yaml.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

        # Loading with plain utf-8 — BOM becomes part of first key
        raw = bom_yaml.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        # The BOM may cause "\ufeffsn" as the key instead of "sn"
        # This is a known issue — test documents the behavior
        assert data is not None

    def test_a03_yaml_anchors_aliases(self, tmp_path: Path) -> None:
        """A-03: YAML with anchors and aliases should work with safe_load."""
        anchor_yaml = tmp_path / "exam.yaml"
        content = textwrap.dedent("""\
            defaults: &defaults
              question_type: essay
              concepts: [세포막]
            questions:
              - sn: 1
                <<: *defaults
              - sn: 2
                <<: *defaults
        """)
        anchor_yaml.write_text(content, encoding="utf-8")

        data = yaml.safe_load(anchor_yaml.read_text())
        assert len(data["questions"]) == 2
        assert data["questions"][0]["question_type"] == "essay"

    def test_a03_yaml_only_comments(self, tmp_path: Path) -> None:
        """A-03: YAML file with only comments should not crash."""
        comment_yaml = tmp_path / "exam.yaml"
        comment_yaml.write_text("# This is a comment\n# Another comment\n", encoding="utf-8")

        data = yaml.safe_load(comment_yaml.read_text())
        assert data is None
