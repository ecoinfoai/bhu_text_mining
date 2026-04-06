"""Tests for week_config module — week.yaml discovery, loading, validation, merge, patterns."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# T004: Discovery tests
# ---------------------------------------------------------------------------


class TestFindWeekConfig:
    """Test find_week_config() upward directory search."""

    def test_find_in_cwd(self, tmp_path: Path) -> None:
        """Find week.yaml in the current working directory."""
        from forma.week_config import find_week_config

        (tmp_path / "week.yaml").write_text("week: 1\n", encoding="utf-8")
        result = find_week_config(tmp_path)
        assert result is not None
        assert result.name == "week.yaml"
        assert result.parent == tmp_path

    def test_find_in_parent_dir(self, tmp_path: Path) -> None:
        """Find week.yaml in a parent directory."""
        from forma.week_config import find_week_config

        (tmp_path / "week.yaml").write_text("week: 1\n", encoding="utf-8")
        child = tmp_path / "subdir"
        child.mkdir()
        result = find_week_config(child)
        assert result is not None
        assert result.parent == tmp_path

    def test_stop_at_first_match(self, tmp_path: Path) -> None:
        """Stop at the first week.yaml found (closest to start_dir)."""
        from forma.week_config import find_week_config

        # Parent has week.yaml with week: 1
        (tmp_path / "week.yaml").write_text("week: 1\n", encoding="utf-8")
        # Child also has week.yaml with week: 2
        child = tmp_path / "child"
        child.mkdir()
        (child / "week.yaml").write_text("week: 2\n", encoding="utf-8")

        result = find_week_config(child)
        assert result is not None
        assert result.parent == child  # Should find the closest one

    def test_return_none_when_not_found(self, tmp_path: Path) -> None:
        """Return None when no week.yaml exists."""
        from forma.week_config import find_week_config

        result = find_week_config(tmp_path)
        assert result is None

    def test_stop_before_forma_yaml_location(self, tmp_path: Path) -> None:
        """Stop searching at forma.yaml's directory (project root)."""
        from forma.week_config import find_week_config

        # Place forma.yaml at root and week.yaml above it (should not be found)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "forma.yaml").write_text("project:\n  course_name: test\n", encoding="utf-8")
        week_dir = project_root / "week01"
        week_dir.mkdir()

        result = find_week_config(week_dir)
        assert result is None  # No week.yaml in week_dir or project_root


# ---------------------------------------------------------------------------
# T005: Loading and validation tests
# ---------------------------------------------------------------------------


class TestLoadWeekConfig:
    """Test load_week_config() and validate_week_config()."""

    def _write_yaml(self, path: Path, data: dict) -> Path:
        """Helper to write a week.yaml file."""
        yaml_path = path / "week.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
        return yaml_path

    def test_valid_yaml_loads_to_dataclass(self, tmp_path: Path) -> None:
        """A valid week.yaml loads into a WeekConfiguration instance."""
        from forma.week_config import load_week_config, WeekConfiguration

        data = {
            "week": 3,
            "select": {
                "source": "test.yaml",
                "questions": [1, 3],
                "num_papers": 220,
            },
        }
        yaml_path = self._write_yaml(tmp_path, data)
        config = load_week_config(yaml_path)
        assert isinstance(config, WeekConfiguration)
        assert config.week == 3

    def test_malformed_yaml_raises_error(self, tmp_path: Path) -> None:
        """Malformed YAML raises an error."""
        from forma.week_config import load_week_config

        yaml_path = tmp_path / "week.yaml"
        yaml_path.write_text("week: {\ninvalid", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            load_week_config(yaml_path)

    def test_missing_week_field_raises_error(self, tmp_path: Path) -> None:
        """Missing 'week' field raises ValueError."""
        from forma.week_config import load_week_config

        yaml_path = self._write_yaml(tmp_path, {"select": {"source": "test.yaml"}})
        with pytest.raises(ValueError, match="week"):
            load_week_config(yaml_path)

    def test_week_less_than_1_raises_error(self, tmp_path: Path) -> None:
        """week < 1 raises ValueError."""
        from forma.week_config import load_week_config

        yaml_path = self._write_yaml(tmp_path, {"week": 0})
        with pytest.raises(ValueError, match="week"):
            load_week_config(yaml_path)

    def test_select_section_validation(self, tmp_path: Path) -> None:
        """Select section requires source, questions, num_papers."""
        from forma.week_config import validate_week_config

        # Missing source
        with pytest.raises(ValueError, match="source"):
            validate_week_config(
                {"week": 1, "select": {"questions": [1], "num_papers": 10}},
                required_section="select",
            )

        # Missing questions
        with pytest.raises(ValueError, match="questions"):
            validate_week_config(
                {"week": 1, "select": {"source": "test.yaml", "num_papers": 10}},
                required_section="select",
            )

        # num_papers < 1
        with pytest.raises(ValueError, match="num_papers"):
            validate_week_config(
                {"week": 1, "select": {"source": "test.yaml", "questions": [1], "num_papers": 0}},
                required_section="select",
            )

    def test_ocr_section_validation(self, tmp_path: Path) -> None:
        """OCR section requires num_questions and image_dir_pattern."""
        from forma.week_config import validate_week_config

        with pytest.raises(ValueError, match="num_questions"):
            validate_week_config(
                {"week": 1, "ocr": {"image_dir_pattern": "img_{class}"}},
                required_section="ocr",
            )

        with pytest.raises(ValueError, match="image_dir_pattern"):
            validate_week_config(
                {"week": 1, "ocr": {"num_questions": 5}},
                required_section="ocr",
            )

    def test_eval_section_validation(self, tmp_path: Path) -> None:
        """Eval section requires config, questions_used, responses_pattern."""
        from forma.week_config import validate_week_config

        with pytest.raises(ValueError, match="config"):
            validate_week_config(
                {"week": 1, "eval": {"questions_used": [1], "responses_pattern": "resp_{class}.yaml"}},
                required_section="eval",
            )

        with pytest.raises(ValueError, match="questions_used"):
            validate_week_config(
                {"week": 1, "eval": {"config": "test.yaml", "responses_pattern": "resp_{class}.yaml"}},
                required_section="eval",
            )

        with pytest.raises(ValueError, match="responses_pattern"):
            validate_week_config(
                {"week": 1, "eval": {"config": "test.yaml", "questions_used": [1]}},
                required_section="eval",
            )

    def test_crop_coords_format_validation(self, tmp_path: Path) -> None:
        """crop_coords must be list of 4-int lists with x1<x2 and y1<y2."""
        from forma.week_config import validate_week_config

        # Valid
        validate_week_config(
            {
                "week": 1,
                "ocr": {
                    "num_questions": 2,
                    "image_dir_pattern": "img",
                    "crop_coords": [[10, 20, 100, 200], [50, 60, 150, 260]],
                },
            },
            required_section="ocr",
        )

        # Wrong number of elements
        with pytest.raises(ValueError, match="crop_coords"):
            validate_week_config(
                {
                    "week": 1,
                    "ocr": {
                        "num_questions": 2,
                        "image_dir_pattern": "img",
                        "crop_coords": [[10, 20, 100]],
                    },
                },
                required_section="ocr",
            )

        # x1 >= x2
        with pytest.raises(ValueError, match="crop_coords"):
            validate_week_config(
                {
                    "week": 1,
                    "ocr": {
                        "num_questions": 2,
                        "image_dir_pattern": "img",
                        "crop_coords": [[100, 20, 50, 200]],
                    },
                },
                required_section="ocr",
            )


# ---------------------------------------------------------------------------
# T006: Merge priority tests
# ---------------------------------------------------------------------------


class TestMergeConfigs:
    """Test 5-level config merge priority."""

    def test_cli_overrides_week(self) -> None:
        """CLI flags override week.yaml values."""
        from forma.week_config import merge_week_configs

        result = merge_week_configs(
            cli={"provider": "anthropic"},
            week_config={"provider": "gemini"},
            eval_config={},
            project_config={},
            defaults={"provider": "gemini"},
            explicit_cli_keys={"provider"},
        )
        assert result["provider"] == "anthropic"

    def test_week_overrides_eval_config(self) -> None:
        """week.yaml overrides legacy eval-config."""
        from forma.week_config import merge_week_configs

        result = merge_week_configs(
            cli={},
            week_config={"skip_feedback": True},
            eval_config={"skip_feedback": False},
            project_config={},
            defaults={"skip_feedback": False},
            explicit_cli_keys=set(),
        )
        assert result["skip_feedback"] is True

    def test_eval_config_overrides_project(self) -> None:
        """Legacy eval-config overrides forma.yaml."""
        from forma.week_config import merge_week_configs

        result = merge_week_configs(
            cli={},
            week_config={},
            eval_config={"n_calls": 5},
            project_config={"n_calls": 3},
            defaults={"n_calls": 3},
            explicit_cli_keys=set(),
        )
        assert result["n_calls"] == 5

    def test_project_overrides_defaults(self) -> None:
        """forma.yaml overrides hardcoded defaults."""
        from forma.week_config import merge_week_configs

        result = merge_week_configs(
            cli={},
            week_config={},
            eval_config={},
            project_config={"dpi": 300},
            defaults={"dpi": 150},
            explicit_cli_keys=set(),
        )
        assert result["dpi"] == 300

    def test_missing_week_falls_back_gracefully(self) -> None:
        """When week.yaml not present, skip that layer."""
        from forma.week_config import merge_week_configs

        result = merge_week_configs(
            cli={},
            week_config={},
            eval_config={},
            project_config={"provider": "gemini"},
            defaults={"provider": "gemini", "n_calls": 3},
            explicit_cli_keys=set(),
        )
        assert result["provider"] == "gemini"
        assert result["n_calls"] == 3

    def test_full_5_level_merge(self) -> None:
        """Full merge with all layers populated."""
        from forma.week_config import merge_week_configs

        result = merge_week_configs(
            cli={"provider": "anthropic", "n_calls": 1},
            week_config={"skip_feedback": True, "n_calls": 5},
            eval_config={"skip_graph": True, "n_calls": 7},
            project_config={"dpi": 300, "skip_graph": False},
            defaults={"provider": "gemini", "n_calls": 3, "skip_feedback": False, "skip_graph": False, "dpi": 150},
            explicit_cli_keys={"provider", "n_calls"},
        )
        assert result["provider"] == "anthropic"  # CLI
        assert result["n_calls"] == 1  # CLI explicit
        assert result["skip_feedback"] is True  # week
        assert result["skip_graph"] is True  # eval_config (week didn't set it)
        assert result["dpi"] == 300  # project


# ---------------------------------------------------------------------------
# T007: {class} pattern resolution tests
# ---------------------------------------------------------------------------


class TestResolveClassPatterns:
    """Test resolve_class_patterns() for {class} placeholder replacement."""

    def test_replaces_class_in_path_fields(self) -> None:
        """Replace {class} in known path fields."""
        from forma.week_config import WeekConfiguration, resolve_class_patterns

        config = WeekConfiguration(
            week=1,
            ocr_image_dir_pattern="img_{class}_w1",
            ocr_ocr_output_pattern="ocr_{class}.yaml",
            ocr_join_output_pattern="join_{class}.yaml",
            eval_responses_pattern="resp_{class}.yaml",
            eval_output_pattern="eval_{class}",
        )
        resolved = resolve_class_patterns(config, "A")
        assert resolved.ocr_image_dir_pattern == "img_A_w1"
        assert resolved.ocr_ocr_output_pattern == "ocr_A.yaml"
        assert resolved.ocr_join_output_pattern == "join_A.yaml"
        assert resolved.eval_responses_pattern == "resp_A.yaml"
        assert resolved.eval_output_pattern == "eval_A"

    def test_error_when_class_pattern_but_no_class_id(self) -> None:
        """Raise error when {class} present but class_id is None."""
        from forma.week_config import WeekConfiguration, resolve_class_patterns

        config = WeekConfiguration(
            week=1,
            ocr_image_dir_pattern="img_{class}_w1",
        )
        with pytest.raises(ValueError, match=r"\{class\}"):
            resolve_class_patterns(config, None)

    def test_no_op_when_no_class_pattern(self) -> None:
        """No-op when value contains no {class}."""
        from forma.week_config import WeekConfiguration, resolve_class_patterns

        config = WeekConfiguration(
            week=1,
            ocr_image_dir_pattern="img_all_w1",
        )
        resolved = resolve_class_patterns(config, "A")
        assert resolved.ocr_image_dir_pattern == "img_all_w1"

    def test_resolve_relative_paths_relative_to_week_yaml(self, tmp_path: Path) -> None:
        """Relative paths resolved relative to week.yaml directory (FR-016)."""
        from forma.week_config import resolve_paths_relative_to

        week_yaml_dir = tmp_path / "week01"
        week_yaml_dir.mkdir()

        resolved = resolve_paths_relative_to(
            "subdir/results.yaml",
            week_yaml_dir,
        )
        assert resolved == str(week_yaml_dir / "subdir" / "results.yaml")

    def test_absolute_paths_unchanged(self, tmp_path: Path) -> None:
        """Absolute paths are not modified."""
        from forma.week_config import resolve_paths_relative_to

        resolved = resolve_paths_relative_to(
            "/absolute/path/results.yaml",
            tmp_path,
        )
        assert resolved == "/absolute/path/results.yaml"


# ---------------------------------------------------------------------------
# T008: Crop coords write-back tests
# ---------------------------------------------------------------------------


class TestSaveCropCoords:
    """Test save_crop_coords() write-back to week.yaml."""

    def test_saves_coordinates(self, tmp_path: Path) -> None:
        """Save crop coordinates to existing week.yaml."""
        from forma.week_config import save_crop_coords

        yaml_path = tmp_path / "week.yaml"
        yaml_path.write_text(
            textwrap.dedent("""\
            week: 1
            ocr:
              num_questions: 2
              image_dir_pattern: img_{class}
            """),
            encoding="utf-8",
        )
        coords = [[10, 20, 100, 200], [50, 60, 150, 260]]
        save_crop_coords(yaml_path, coords)

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["ocr"]["crop_coords"] == coords

    def test_preserves_other_fields(self, tmp_path: Path) -> None:
        """Other fields in week.yaml are preserved after write-back."""
        from forma.week_config import save_crop_coords

        yaml_path = tmp_path / "week.yaml"
        yaml_path.write_text(
            textwrap.dedent("""\
            week: 3
            select:
              source: test.yaml
              questions: [1, 3]
              num_papers: 220
            ocr:
              num_questions: 2
              image_dir_pattern: img_{class}
            """),
            encoding="utf-8",
        )
        save_crop_coords(yaml_path, [[10, 20, 100, 200]])

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["week"] == 3
        assert data["select"]["source"] == "test.yaml"
        assert data["ocr"]["num_questions"] == 2

    def test_preserves_original_on_write_failure(self, tmp_path: Path) -> None:
        """Original file preserved when write fails."""
        from forma.week_config import save_crop_coords

        # Make directory read-only to trigger write failure
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        ro_yaml = read_only_dir / "week.yaml"
        original_content = "week: 1\nocr:\n  num_questions: 2\n"
        ro_yaml.write_text(original_content, encoding="utf-8")

        # Record original mtime
        original_stat = ro_yaml.stat()
        read_only_dir.chmod(0o555)

        try:
            with pytest.raises(OSError):
                save_crop_coords(ro_yaml, [[10, 20, 100, 200]])

            # File should not have been modified (mtime unchanged)
            assert ro_yaml.stat().st_mtime == original_stat.st_mtime
        finally:
            read_only_dir.chmod(0o755)


# ---------------------------------------------------------------------------
# T008a: --class value validation test
# ---------------------------------------------------------------------------


class TestClassValueValidation:
    """Test warning when --class value not in classes.identifiers."""

    def test_warning_when_class_not_in_identifiers(self, caplog) -> None:
        """Log warning when class_id is not in identifiers list."""
        from forma.week_config import warn_if_class_unknown

        with caplog.at_level(logging.WARNING):
            warn_if_class_unknown("Z", ["A", "B", "C", "D"])
        assert "Z" in caplog.text
        assert any(
            "identifiers" in msg.lower() or "class" in msg.lower() for msg in [r.message for r in caplog.records]
        )

    def test_no_warning_when_class_in_identifiers(self, caplog) -> None:
        """No warning when class_id is valid."""
        from forma.week_config import warn_if_class_unknown

        with caplog.at_level(logging.WARNING):
            warn_if_class_unknown("A", ["A", "B", "C", "D"])
        assert not caplog.records


# -------------------------------------------------------------------
# T005: Lecture section tests
# -------------------------------------------------------------------


class TestLectureSection:
    """Test lecture section in week.yaml loading and validation."""

    def _write_yaml(self, path: Path, data: dict) -> Path:
        """Helper to write a week.yaml file."""
        yaml_path = path / "week.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
            )
        return yaml_path

    def test_load_week_config_lecture_section(
        self,
        tmp_path: Path,
    ) -> None:
        """week.yaml with lecture section populates lecture_* fields."""
        from forma.week_config import load_week_config

        data = {
            "week": 1,
            "lecture": {
                "transcript_pattern": "trans_{class}.txt",
                "concept_source": "concepts.yaml",
                "output_dir": "output/lecture",
                "extra_stopwords": ["커스텀"],
                "extra_abbreviations": ["PCR"],
            },
        }
        yaml_path = self._write_yaml(tmp_path, data)
        config = load_week_config(yaml_path)
        assert config.lecture_transcript_pattern == ("trans_{class}.txt")
        assert config.lecture_concept_source == "concepts.yaml"
        assert config.lecture_output_dir == "output/lecture"
        assert config.lecture_extra_stopwords == ["커스텀"]
        assert config.lecture_extra_abbreviations == ["PCR"]

    def test_load_week_config_no_lecture_section(
        self,
        tmp_path: Path,
    ) -> None:
        """Missing lecture section uses defaults (empty strings)."""
        from forma.week_config import load_week_config

        data = {"week": 1}
        yaml_path = self._write_yaml(tmp_path, data)
        config = load_week_config(yaml_path)
        assert config.lecture_transcript_pattern == ""
        assert config.lecture_concept_source == ""
        assert config.lecture_output_dir == ""
        assert config.lecture_extra_stopwords == []
        assert config.lecture_extra_abbreviations == []

    def test_lecture_transcript_pattern_class_resolution(
        self,
        tmp_path: Path,
    ) -> None:
        """{class} in lecture_transcript_pattern gets resolved."""
        from forma.week_config import (
            WeekConfiguration,
            resolve_class_patterns,
        )

        config = WeekConfiguration(
            week=1,
            lecture_transcript_pattern="trans_{class}.txt",
        )
        resolved = resolve_class_patterns(config, "B")
        assert resolved.lecture_transcript_pattern == ("trans_B.txt")

    def test_validate_week_config_lecture_section(self) -> None:
        """Validation checks for lecture section."""
        from forma.week_config import validate_week_config

        # Missing transcript_pattern should fail
        with pytest.raises(ValueError, match="transcript_pattern"):
            validate_week_config(
                {
                    "week": 1,
                    "lecture": {"concept_source": "c.yaml"},
                },
                required_section="lecture",
            )

        # Non-mapping lecture section should fail
        with pytest.raises(ValueError, match="lecture"):
            validate_week_config(
                {"week": 1, "lecture": "not_a_dict"},
                required_section="lecture",
            )
