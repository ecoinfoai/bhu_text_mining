"""Tests for cli_report_student.py — CLI argument parsing and execution.

TDD RED phase: tests written before implementation.
T017: CLI argument parsing and error handling.
"""

from __future__ import annotations

import os

import pytest

from forma.evaluation_types import LongitudinalRecord
from forma.longitudinal_store import LongitudinalStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    student_id: str = "s001",
    week: int = 1,
    question_sn: int = 1,
    concept_coverage: float = 0.7,
    llm_rubric: float = 0.6,
    ensemble_score: float = 0.65,
    rasch_ability: float = 0.5,
) -> LongitudinalRecord:
    return LongitudinalRecord(
        student_id=student_id,
        week=week,
        question_sn=question_sn,
        scores={
            "concept_coverage": concept_coverage,
            "llm_rubric": llm_rubric,
            "ensemble_score": ensemble_score,
            "rasch_ability": rasch_ability,
        },
        tier_level=2,
        tier_label="기전+용어",
    )


def _create_store(tmp_path) -> str:
    """Create a store with 2 students, 2 weeks, 2 questions."""
    store_path = str(tmp_path / "store.yaml")
    store = LongitudinalStore(store_path)
    for sid in ["s001", "s002"]:
        for week in [1, 2]:
            for qsn in [1, 2]:
                store.add_record(_make_record(
                    student_id=sid, week=week, question_sn=qsn,
                    ensemble_score=0.5 + 0.05 * week,
                ))
    store.save()
    return store_path


def _create_id_csv(tmp_path) -> str:
    """Create a minimal ID CSV."""
    csv_path = str(tmp_path / "id.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("타임스탬프,익명ID,분반을 선택하세요.,학번을 입력하세요.,이름을 입력하세요.\n")
        f.write("2024-01-01,anon1,A반,s001,홍길동\n")
        f.write("2024-01-01,anon2,B반,s002,김철수\n")
    return csv_path


# ---------------------------------------------------------------------------
# T017: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCliReportStudentParsing:
    """CLI argument parsing for forma-report-student."""

    def test_required_args_accepted(self, tmp_path):
        from forma.cli_report_student import _build_parser

        store_path = str(tmp_path / "store.yaml")
        csv_path = str(tmp_path / "id.csv")
        output = str(tmp_path / "out.pdf")

        parser = _build_parser()
        args = parser.parse_args([
            "--store", store_path,
            "--student", "s001",
            "--id-csv", csv_path,
            "--output", output,
        ])
        assert args.store == store_path
        assert args.student == "s001"
        assert args.id_csv == csv_path
        assert args.output == output

    def test_no_llm_flag(self, tmp_path):
        from forma.cli_report_student import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "s.yaml",
            "--student", "s001",
            "--id-csv", "id.csv",
            "--output", "out.pdf",
            "--no-llm",
        ])
        assert args.no_llm is True

    def test_no_llm_default_false(self, tmp_path):
        from forma.cli_report_student import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "s.yaml",
            "--student", "s001",
            "--id-csv", "id.csv",
            "--output", "out.pdf",
        ])
        assert args.no_llm is False

    def test_optional_args(self, tmp_path):
        from forma.cli_report_student import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--store", "s.yaml",
            "--student", "s001",
            "--id-csv", "id.csv",
            "--output", "out.pdf",
            "--weeks", "1", "2", "3",
            "--dpi", "200",
            "--no-config",
        ])
        assert args.weeks == [1, 2, 3]
        assert args.dpi == 200
        assert args.no_config is True

    def test_missing_required_arg_exits(self):
        from forma.cli_report_student import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--store", "s.yaml"])


class TestCliReportStudentExecution:
    """CLI main() execution tests."""

    def test_student_not_found_exits_1(self, tmp_path):
        from forma.cli_report_student import main

        store_path = _create_store(tmp_path)
        csv_path = _create_id_csv(tmp_path)
        output = str(tmp_path / "out.pdf")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--store", store_path,
                "--student", "nonexistent",
                "--id-csv", csv_path,
                "--output", output,
                "--no-config",
                "--no-llm",
            ])
        assert exc_info.value.code == 1

    def test_store_not_found_exits_1(self, tmp_path):
        from forma.cli_report_student import main

        csv_path = _create_id_csv(tmp_path)
        output = str(tmp_path / "out.pdf")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--store", str(tmp_path / "nonexistent.yaml"),
                "--student", "s001",
                "--id-csv", csv_path,
                "--output", output,
                "--no-config",
                "--no-llm",
            ])
        assert exc_info.value.code == 1

    def test_successful_generation(self, tmp_path):
        from forma.cli_report_student import main

        store_path = _create_store(tmp_path)
        csv_path = _create_id_csv(tmp_path)
        output = str(tmp_path / "out.pdf")

        result = main([
            "--store", store_path,
            "--student", "s001",
            "--id-csv", csv_path,
            "--output", output,
            "--no-config",
            "--no-llm",
        ])
        assert result is None  # success
        assert os.path.isfile(output)
        assert os.path.getsize(output) > 0


# ---------------------------------------------------------------------------
# T031: Batch mode tests
# ---------------------------------------------------------------------------


class TestCliBatchParsing:
    """CLI argument parsing for forma-report-student-batch."""

    def test_batch_parser_required_args(self, tmp_path):
        from forma.cli_report_student import _build_batch_parser

        parser = _build_batch_parser()
        args = parser.parse_args([
            "--store", "store.yaml",
            "--id-csv", "id.csv",
            "--output-dir", str(tmp_path / "out"),
        ])
        assert args.store == "store.yaml"
        assert args.id_csv == "id.csv"
        assert args.output_dir == str(tmp_path / "out")

    def test_batch_parser_optional_args(self, tmp_path):
        from forma.cli_report_student import _build_batch_parser

        parser = _build_batch_parser()
        args = parser.parse_args([
            "--store", "store.yaml",
            "--id-csv", "id.csv",
            "--output-dir", str(tmp_path / "out"),
            "--weeks", "1", "2",
            "--dpi", "200",
            "--no-llm",
            "--no-config",
        ])
        assert args.weeks == [1, 2]
        assert args.dpi == 200
        assert args.no_llm is True
        assert args.no_config is True

    def test_batch_parser_missing_required_exits(self):
        from forma.cli_report_student import _build_batch_parser

        parser = _build_batch_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--store", "store.yaml"])


class TestCliBatchExecution:
    """CLI batch_main() execution tests."""

    def test_batch_creates_output_dir(self, tmp_path):
        from forma.cli_report_student import batch_main

        store_path = _create_store(tmp_path)
        csv_path = _create_id_csv(tmp_path)
        output_dir = str(tmp_path / "new_output_dir")

        assert not os.path.exists(output_dir)
        batch_main([
            "--store", store_path,
            "--id-csv", csv_path,
            "--output-dir", output_dir,
            "--no-config",
            "--no-llm",
        ])
        assert os.path.isdir(output_dir)

    def test_batch_generates_pdf_per_student(self, tmp_path):
        from forma.cli_report_student import batch_main

        store_path = _create_store(tmp_path)
        csv_path = _create_id_csv(tmp_path)
        output_dir = str(tmp_path / "batch_out")

        batch_main([
            "--store", store_path,
            "--id-csv", csv_path,
            "--output-dir", output_dir,
            "--no-config",
            "--no-llm",
        ])

        # Store has s001 and s002
        assert os.path.isfile(os.path.join(output_dir, "s001.pdf"))
        assert os.path.isfile(os.path.join(output_dir, "s002.pdf"))
        assert os.path.getsize(os.path.join(output_dir, "s001.pdf")) > 0
        assert os.path.getsize(os.path.join(output_dir, "s002.pdf")) > 0

    def test_batch_continues_on_individual_error(self, tmp_path, monkeypatch):
        """Batch should continue when one student fails."""
        from forma.cli_report_student import batch_main

        store_path = _create_store(tmp_path)
        csv_path = _create_id_csv(tmp_path)
        output_dir = str(tmp_path / "batch_err")

        # Patch generate_pdf to fail for s001 only
        orig_generate = None

        from forma.student_longitudinal_report import StudentLongitudinalPDFReportGenerator

        orig_generate = StudentLongitudinalPDFReportGenerator.generate_pdf

        def patched_generate(self, student_data, *a, **kw):
            if student_data.student_id == "s001":
                raise RuntimeError("Simulated failure for s001")
            return orig_generate(self, student_data, *a, **kw)

        monkeypatch.setattr(
            StudentLongitudinalPDFReportGenerator, "generate_pdf", patched_generate,
        )

        batch_main([
            "--store", store_path,
            "--id-csv", csv_path,
            "--output-dir", output_dir,
            "--no-config",
            "--no-llm",
        ])

        # s001 should fail, s002 should succeed
        assert not os.path.isfile(os.path.join(output_dir, "s001.pdf"))
        assert os.path.isfile(os.path.join(output_dir, "s002.pdf"))

    def test_batch_outputs_progress(self, tmp_path, capsys):
        from forma.cli_report_student import batch_main

        store_path = _create_store(tmp_path)
        csv_path = _create_id_csv(tmp_path)
        output_dir = str(tmp_path / "batch_progress")

        batch_main([
            "--store", store_path,
            "--id-csv", csv_path,
            "--output-dir", output_dir,
            "--no-config",
            "--no-llm",
        ])

        captured = capsys.readouterr()
        assert "Complete:" in captured.out
        assert "2/2" in captured.out

    def test_batch_store_not_found_exits_1(self, tmp_path):
        from forma.cli_report_student import batch_main

        csv_path = _create_id_csv(tmp_path)
        output_dir = str(tmp_path / "batch_err2")

        with pytest.raises(SystemExit) as exc_info:
            batch_main([
                "--store", str(tmp_path / "nonexistent.yaml"),
                "--id-csv", csv_path,
                "--output-dir", output_dir,
                "--no-config",
                "--no-llm",
            ])
        assert exc_info.value.code == 1
