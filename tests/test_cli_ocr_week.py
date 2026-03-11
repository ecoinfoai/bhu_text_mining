"""Tests for class-aware OCR scan/join — week.yaml integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _write_week_yaml(path: Path, **overrides) -> Path:
    """Write a sample week.yaml with ocr section."""
    data = {
        "week": 1,
        "ocr": {
            "num_questions": 2,
            "image_dir_pattern": "인구기_형성평가_1{class}_1주차",
            "ocr_output_pattern": "ocr_results_{class}.yaml",
            "join_output_pattern": "final_{class}.yaml",
        },
    }
    for k, v in overrides.items():
        if "." in k:
            section, key = k.split(".", 1)
            if section not in data:
                data[section] = {}
            data[section][key] = v
        else:
            data[k] = v
    yaml_path = path / "week.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def _write_forma_yaml(path: Path, **overrides) -> Path:
    """Write a sample forma.yaml."""
    data = {
        "project": {"course_name": "테스트"},
        "classes": {"identifiers": ["A", "B", "C", "D"]},
        "ocr": {
            "naver_config": "/run/agenix/forma-config",
            "credentials": "",
            "spreadsheet_url": "https://docs.google.com/spreadsheets/d/XXX",
        },
    }
    for k, v in overrides.items():
        if "." in k:
            section, key = k.split(".", 1)
            if section not in data:
                data[section] = {}
            data[section][key] = v
        else:
            data[k] = v
    yaml_path = path / "forma.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


# ---------------------------------------------------------------------------
# T034: scan --class A resolves image_dir_pattern correctly
# ---------------------------------------------------------------------------

class TestScanClassResolution:
    """Test that scan --class resolves patterns."""

    def test_resolves_image_dir_pattern(self, tmp_path: Path) -> None:
        """scan --class A resolves {class} in image_dir_pattern."""
        from forma.week_config import (
            WeekConfiguration,
            resolve_class_patterns,
        )

        config = WeekConfiguration(
            week=1,
            ocr_image_dir_pattern="인구기_형성평가_1{class}_1주차",
            ocr_ocr_output_pattern="ocr_{class}.yaml",
        )
        resolved = resolve_class_patterns(config, "A")
        assert resolved.ocr_image_dir_pattern == "인구기_형성평가_1A_1주차"
        assert resolved.ocr_ocr_output_pattern == "ocr_A.yaml"


# ---------------------------------------------------------------------------
# T035: crop coords saved after interactive selection
# ---------------------------------------------------------------------------

class TestCropCoordsSave:
    """Test crop coords are saved to week.yaml."""

    def test_coords_saved_after_selection(self, tmp_path: Path) -> None:
        """Crop coords written to week.yaml after show_image()."""
        from forma.week_config import save_crop_coords

        yaml_path = _write_week_yaml(tmp_path)
        coords = [[10, 20, 100, 200], [50, 60, 150, 260]]
        save_crop_coords(yaml_path, coords)

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["ocr"]["crop_coords"] == coords


# ---------------------------------------------------------------------------
# T036: existing crop coords reused without prompting
# ---------------------------------------------------------------------------

class TestCropCoordsReuse:
    """Test saved crop coords are reused."""

    def test_existing_coords_reused(self, tmp_path: Path) -> None:
        """When crop_coords present, no interactive picker needed."""
        yaml_path = _write_week_yaml(
            tmp_path,
            **{"ocr.crop_coords": [[10, 20, 100, 200], [50, 60, 150, 260]]},
        )
        from forma.week_config import load_week_config

        config = load_week_config(yaml_path)
        assert len(config.ocr_crop_coords) == 2
        assert config.ocr_crop_coords[0] == [10, 20, 100, 200]


# ---------------------------------------------------------------------------
# T037: --recrop forces interactive picker
# ---------------------------------------------------------------------------

class TestRecropFlag:
    """Test --recrop flag forces re-selection."""

    def test_recrop_flag_accepted(self) -> None:
        """--recrop flag is parsed without error."""
        from forma.cli_ocr import _parse_args

        args = _parse_args(["scan", "--config", "test.yaml", "--recrop"])
        assert args.recrop is True

    def test_no_recrop_default(self) -> None:
        """Default recrop is False."""
        from forma.cli_ocr import _parse_args

        args = _parse_args(["scan", "--config", "test.yaml"])
        assert args.recrop is False


# ---------------------------------------------------------------------------
# T038: error when {class} in pattern but --class not provided
# ---------------------------------------------------------------------------

class TestClassPatternError:
    """Test error when {class} present but --class not given."""

    def test_error_without_class_flag(self) -> None:
        """ValueError when {class} in pattern but class_id is None."""
        from forma.week_config import WeekConfiguration, resolve_class_patterns

        config = WeekConfiguration(
            week=1,
            ocr_image_dir_pattern="img_{class}_w1",
        )
        with pytest.raises(ValueError, match=r"\{class\}"):
            resolve_class_patterns(config, None)


# ---------------------------------------------------------------------------
# T038a: exit code 2 when resolved image dir does not exist
# ---------------------------------------------------------------------------

class TestImageDirNotFound:
    """Test error when resolved image directory doesn't exist."""

    def test_resolved_dir_not_found(self, tmp_path: Path) -> None:
        """Resolved image_dir_pattern pointing to non-existent dir."""
        from forma.week_config import resolve_class_patterns, WeekConfiguration

        config = WeekConfiguration(
            week=1,
            ocr_image_dir_pattern=str(tmp_path / "nonexistent_{class}"),
        )
        resolved = resolve_class_patterns(config, "A")
        assert not Path(resolved.ocr_image_dir_pattern).exists()


# ---------------------------------------------------------------------------
# T045: join --class A resolves OCR and join output paths
# ---------------------------------------------------------------------------

class TestJoinClassResolution:
    """Test join --class resolves paths."""

    def test_resolves_join_paths(self) -> None:
        """join --class A resolves ocr_output_pattern and join_output_pattern."""
        from forma.week_config import WeekConfiguration, resolve_class_patterns

        config = WeekConfiguration(
            week=1,
            ocr_ocr_output_pattern="ocr_{class}.yaml",
            ocr_join_output_pattern="final_{class}.yaml",
        )
        resolved = resolve_class_patterns(config, "A")
        assert resolved.ocr_ocr_output_pattern == "ocr_A.yaml"
        assert resolved.ocr_join_output_pattern == "final_A.yaml"


# ---------------------------------------------------------------------------
# T046: spreadsheet_url from forma.yaml; CSV fallback from week.yaml
# ---------------------------------------------------------------------------

class TestDataSourcePriority:
    """Test join data source priority."""

    def test_spreadsheet_url_from_forma(self, tmp_path: Path) -> None:
        """forma.yaml spreadsheet_url is available."""
        from forma.project_config import load_project_config

        _write_forma_yaml(tmp_path)
        config = load_project_config(tmp_path / "forma.yaml")
        assert config["ocr"]["spreadsheet_url"] == "https://docs.google.com/spreadsheets/d/XXX"

    def test_csv_fallback_from_week(self, tmp_path: Path) -> None:
        """week.yaml join_forms_csv is available as fallback."""
        from forma.week_config import load_week_config

        _write_week_yaml(tmp_path, **{"ocr.join_forms_csv": "forms_data.csv"})
        config = load_week_config(tmp_path / "week.yaml")
        assert config.ocr_join_forms_csv == "forms_data.csv"


# ---------------------------------------------------------------------------
# T047: error when neither Sheets URL nor CSV configured
# ---------------------------------------------------------------------------

class TestNoDataSource:
    """Test error when no data source for join."""

    def test_no_data_source_detected(self, tmp_path: Path) -> None:
        """Detect when neither spreadsheet_url nor forms_csv is set."""
        _write_forma_yaml(tmp_path, **{"ocr.spreadsheet_url": ""})
        _write_week_yaml(tmp_path)

        from forma.week_config import load_week_config
        from forma.project_config import load_project_config

        week_config = load_week_config(tmp_path / "week.yaml")
        proj_config = load_project_config(tmp_path / "forma.yaml")

        spreadsheet = proj_config.get("ocr", {}).get("spreadsheet_url", "")
        csv_path = week_config.ocr_join_forms_csv

        assert not spreadsheet
        assert not csv_path


# ---------------------------------------------------------------------------
# T047a: exit code 2 when resolved OCR results file not found
# ---------------------------------------------------------------------------

class TestOcrResultsNotFound:
    """Test error when resolved OCR results file doesn't exist."""

    def test_ocr_results_not_found(self, tmp_path: Path) -> None:
        """Resolved ocr_output_pattern pointing to non-existent file."""
        from forma.week_config import resolve_class_patterns, WeekConfiguration

        config = WeekConfiguration(
            week=1,
            ocr_ocr_output_pattern=str(tmp_path / "missing_{class}.yaml"),
        )
        resolved = resolve_class_patterns(config, "A")
        assert not Path(resolved.ocr_ocr_output_pattern).exists()
