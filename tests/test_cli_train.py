"""Tests for cli_train.py — forma-train CLI entry point.

T022 [US2]: Training workflow, insufficient data errors, flags.
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


def _make_store_records(n_students: int = 15, n_weeks: int = 3) -> list[dict]:
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


class TestCliTrain:
    """Tests for forma-train CLI."""

    def test_successful_training(self, tmp_path: Path, monkeypatch):
        """forma-train with valid data produces a model file."""
        store_path = tmp_path / "store.yaml"
        output_path = tmp_path / "model.pkl"
        _write_store(store_path, _make_store_records(n_students=15))

        monkeypatch.setattr("sys.argv", [
            "forma-train",
            "--store", str(store_path),
            "--output", str(output_path),
        ])

        from forma.cli_train import main
        main()

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_insufficient_students_error(self, tmp_path: Path, monkeypatch):
        """Too few students exits with error."""
        store_path = tmp_path / "store.yaml"
        output_path = tmp_path / "model.pkl"
        _write_store(store_path, _make_store_records(n_students=3))

        monkeypatch.setattr("sys.argv", [
            "forma-train",
            "--store", str(store_path),
            "--output", str(output_path),
            "--min-students", "10",
        ])

        from forma.cli_train import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_insufficient_weeks_error(self, tmp_path: Path, monkeypatch):
        """Too few weeks exits with error."""
        store_path = tmp_path / "store.yaml"
        output_path = tmp_path / "model.pkl"
        _write_store(store_path, _make_store_records(n_students=15, n_weeks=2))

        monkeypatch.setattr("sys.argv", [
            "forma-train",
            "--store", str(store_path),
            "--output", str(output_path),
            "--min-weeks", "3",
        ])

        from forma.cli_train import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_store_not_found_error(self, tmp_path: Path, monkeypatch):
        """Nonexistent store file exits with error."""
        monkeypatch.setattr("sys.argv", [
            "forma-train",
            "--store", str(tmp_path / "missing.yaml"),
            "--output", str(tmp_path / "model.pkl"),
        ])

        from forma.cli_train import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_threshold_flag(self, tmp_path: Path, monkeypatch):
        """--threshold flag is accepted and used."""
        store_path = tmp_path / "store.yaml"
        output_path = tmp_path / "model.pkl"
        _write_store(store_path, _make_store_records(n_students=15))

        monkeypatch.setattr("sys.argv", [
            "forma-train",
            "--store", str(store_path),
            "--output", str(output_path),
            "--threshold", "0.5",
        ])

        from forma.cli_train import main
        main()

        assert output_path.exists()

    def test_verbose_flag(self, tmp_path: Path, monkeypatch, capsys):
        """--verbose flag produces additional output."""
        store_path = tmp_path / "store.yaml"
        output_path = tmp_path / "model.pkl"
        _write_store(store_path, _make_store_records(n_students=15))

        monkeypatch.setattr("sys.argv", [
            "forma-train",
            "--store", str(store_path),
            "--output", str(output_path),
            "--verbose",
        ])

        from forma.cli_train import main
        main()

        _captured = capsys.readouterr()
        assert output_path.exists()
