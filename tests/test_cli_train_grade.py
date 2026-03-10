"""Tests for cli_train_grade.py — forma-train-grade CLI entry point.

T053 [US5]: CLI argument parsing, training workflow, error handling.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _write_store(path: Path, records: list[dict]) -> None:
    """Write a minimal longitudinal store YAML file."""
    store_data = {
        "records": {
            f"{r['student_id']}_{r['week']}_{r.get('question_sn', 1)}": r
            for r in records
        }
    }
    path.write_text(
        yaml.dump(store_data, allow_unicode=True),
        encoding="utf-8",
    )


def _make_store_records(n_students: int = 15, n_weeks: int = 4) -> list[dict]:
    """Generate store records for testing."""
    records = []
    for i in range(n_students):
        sid = f"S{i+1:03d}"
        base = 0.3 + (i / n_students) * 0.5
        for w in range(1, n_weeks + 1):
            score = max(0.0, min(1.0, base + (w - 2) * 0.05))
            records.append({
                "student_id": sid,
                "week": w,
                "question_sn": 1,
                "scores": {"ensemble_score": score, "concept_coverage": score * 0.9},
                "tier_level": 2 if score >= 0.45 else 0,
                "tier_label": "Proficient" if score >= 0.45 else "Beginning",
            })
    return records


def _write_grade_mapping(path: Path, n_students: int = 15) -> None:
    """Write a minimal grade mapping YAML file."""
    grades = {}
    for i in range(n_students):
        sid = f"S{i+1:03d}"
        if i < 3:
            grades[sid] = "A"
        elif i < 7:
            grades[sid] = "B"
        elif i < 11:
            grades[sid] = "C"
        elif i < 13:
            grades[sid] = "D"
        else:
            grades[sid] = "F"
    data = {"2024-1학기": grades}
    path.write_text(
        yaml.dump(data, allow_unicode=True),
        encoding="utf-8",
    )


class TestCliTrainGradeParser:
    """Tests for argument parsing."""

    def test_required_args(self):
        """Parser requires --store, --grades, --output."""
        from forma.cli_train_grade import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "store.yaml",
            "--grades", "grades.yaml",
            "--output", "model.pkl",
        ])
        assert args.store == "store.yaml"
        assert args.grades == "grades.yaml"
        assert args.output == "model.pkl"

    def test_optional_flags(self):
        """Optional flags are accepted."""
        from forma.cli_train_grade import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "s.yaml",
            "--grades", "g.yaml",
            "--output", "m.pkl",
            "--semester", "2024-1학기",
            "--min-students", "5",
            "--verbose",
        ])
        assert args.semester == "2024-1학기"
        assert args.min_students == 5
        assert args.verbose is True

    def test_no_config_flag(self):
        """--no-config flag is accepted."""
        from forma.cli_train_grade import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "s.yaml",
            "--grades", "g.yaml",
            "--output", "m.pkl",
            "--no-config",
        ])
        assert args.no_config is True


class TestCliTrainGradeWorkflow:
    """Tests for training workflow."""

    def test_successful_training(self, tmp_path: Path):
        """forma-train-grade with valid data produces a model file."""
        store_path = tmp_path / "store.yaml"
        grades_path = tmp_path / "grades.yaml"
        output_path = tmp_path / "grade_model.pkl"

        _write_store(store_path, _make_store_records(n_students=15))
        _write_grade_mapping(grades_path, n_students=15)

        from forma.cli_train_grade import main

        main([
            "--store", str(store_path),
            "--grades", str(grades_path),
            "--output", str(output_path),
            "--no-config",
        ])
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_store_not_found_error(self, tmp_path: Path):
        """Nonexistent store file exits with error."""
        from forma.cli_train_grade import main

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--store", str(tmp_path / "missing.yaml"),
                "--grades", str(tmp_path / "g.yaml"),
                "--output", str(tmp_path / "m.pkl"),
                "--no-config",
            ])
        assert exc_info.value.code == 1

    def test_grades_not_found_error(self, tmp_path: Path):
        """Nonexistent grades file exits with error."""
        store_path = tmp_path / "store.yaml"
        _write_store(store_path, _make_store_records())

        from forma.cli_train_grade import main

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--store", str(store_path),
                "--grades", str(tmp_path / "missing.yaml"),
                "--output", str(tmp_path / "m.pkl"),
                "--no-config",
            ])
        assert exc_info.value.code == 1

    def test_insufficient_students_error(self, tmp_path: Path):
        """Too few students exits with error."""
        store_path = tmp_path / "store.yaml"
        grades_path = tmp_path / "grades.yaml"
        _write_store(store_path, _make_store_records(n_students=3))
        _write_grade_mapping(grades_path, n_students=3)

        from forma.cli_train_grade import main

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--store", str(store_path),
                "--grades", str(grades_path),
                "--output", str(tmp_path / "m.pkl"),
                "--min-students", "10",
                "--no-config",
            ])
        assert exc_info.value.code == 1

    def test_verbose_flag(self, tmp_path: Path, capsys):
        """--verbose flag produces output."""
        store_path = tmp_path / "store.yaml"
        grades_path = tmp_path / "grades.yaml"
        output_path = tmp_path / "model.pkl"

        _write_store(store_path, _make_store_records(n_students=15))
        _write_grade_mapping(grades_path, n_students=15)

        from forma.cli_train_grade import main

        main([
            "--store", str(store_path),
            "--grades", str(grades_path),
            "--output", str(output_path),
            "--verbose",
            "--no-config",
        ])
        assert output_path.exists()
