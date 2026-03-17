"""Tests for forma domain CLI subcommands (T010, T021, T032).

T010: Tests for extract subcommand argument parsing
T021: Tests for coverage subcommand argument parsing
T032: Tests for report subcommand argument parsing
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from forma.cli_domain import coverage_main, extract_main, report_main


class TestExtractCLIParsing:
    """Tests for 'forma domain extract' CLI argument parsing."""

    def test_missing_textbook_arg_exits(self) -> None:
        """Missing --textbook exits with error."""
        with pytest.raises(SystemExit) as exc_info:
            extract_main(["--output", "out.yaml"])
        assert exc_info.value.code != 0

    def test_missing_output_arg_exits(self) -> None:
        """Missing --output exits with error."""
        with pytest.raises(SystemExit) as exc_info:
            extract_main(["--textbook", "file.txt"])
        assert exc_info.value.code != 0

    def test_nonexistent_textbook_file_exits(self) -> None:
        """Non-existent textbook file exits with error code 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.yaml"
            with pytest.raises(SystemExit) as exc_info:
                extract_main([
                    "--textbook", "/nonexistent/file.txt",
                    "--output", str(output),
                ])
            assert exc_info.value.code == 1

    def test_valid_args_produce_output(self) -> None:
        """Valid arguments produce a YAML output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample textbook file
            textbook = Path(tmpdir) / "3장 피부.txt"
            textbook.write_text(
                "피부는 표피(epidermis)와 진피(dermis)로 구성된다. "
                "표피는 피부의 가장 바깥층이다. "
                "진피는 표피 아래에 위치한다. "
                "표피의 세포는 각질세포이다.",
                encoding="utf-8",
            )
            output = Path(tmpdir) / "concepts.yaml"

            extract_main([
                "--textbook", str(textbook),
                "--output", str(output),
            ])

            assert output.exists()
            assert output.stat().st_size > 0

    def test_multiple_textbook_files(self) -> None:
        """--textbook can be repeated for multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "3장 피부.txt"
            file2 = Path(tmpdir) / "4장 근육.txt"
            file1.write_text(
                "피부는 표피(epidermis)와 진피로 구성된다. 표피는 피부의 바깥층이다.",
                encoding="utf-8",
            )
            file2.write_text(
                "골격근(skeletal muscle)은 수의근이다. 골격근은 횡문근이다.",
                encoding="utf-8",
            )
            output = Path(tmpdir) / "concepts.yaml"

            extract_main([
                "--textbook", str(file1),
                "--textbook", str(file2),
                "--output", str(output),
            ])

            assert output.exists()

    def test_min_freq_flag(self) -> None:
        """--min-freq flag is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            textbook = Path(tmpdir) / "test.txt"
            textbook.write_text(
                "피부는 표피(epidermis)와 진피로 구성된다. 표피는 피부의 바깥층이다.",
                encoding="utf-8",
            )
            output = Path(tmpdir) / "concepts.yaml"

            extract_main([
                "--textbook", str(textbook),
                "--output", str(output),
                "--min-freq", "3",
            ])

            assert output.exists()

    def test_no_cache_flag(self) -> None:
        """--no-cache flag is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            textbook = Path(tmpdir) / "test.txt"
            textbook.write_text(
                "피부는 표피(epidermis)와 진피로 구성된다. 표피는 피부의 바깥층이다.",
                encoding="utf-8",
            )
            output = Path(tmpdir) / "concepts.yaml"

            extract_main([
                "--textbook", str(textbook),
                "--output", str(output),
                "--no-cache",
            ])

            assert output.exists()


# ----------------------------------------------------------------
# T021: Coverage CLI parsing
# ----------------------------------------------------------------


class TestCoverageCLIParsing:
    """Tests for 'forma domain coverage' CLI argument parsing."""

    def test_missing_concepts_arg_exits(self) -> None:
        """Missing --concepts exits with error."""
        with pytest.raises(SystemExit) as exc_info:
            coverage_main([
                "--transcripts", "file.txt",
                "--output", "out.yaml",
            ])
        assert exc_info.value.code != 0

    def test_missing_transcripts_arg_exits(self) -> None:
        """Missing --transcripts exits with error."""
        with pytest.raises(SystemExit) as exc_info:
            coverage_main([
                "--concepts", "concepts.yaml",
                "--output", "out.yaml",
            ])
        assert exc_info.value.code != 0

    def test_missing_output_arg_exits(self) -> None:
        """Missing --output exits with error."""
        with pytest.raises(SystemExit) as exc_info:
            coverage_main([
                "--concepts", "concepts.yaml",
                "--transcripts", "file.txt",
            ])
        assert exc_info.value.code != 0

    def test_nonexistent_concepts_file_exits(self) -> None:
        """Non-existent concepts file exits with error code 1."""
        with pytest.raises(SystemExit) as exc_info:
            coverage_main([
                "--concepts", "/nonexistent/concepts.yaml",
                "--transcripts", "/nonexistent/t.txt",
                "--output", "/tmp/out.yaml",
            ])
        assert exc_info.value.code == 1

    def test_accepts_optional_flags(self) -> None:
        """Optional flags --scope, --threshold, --week-config, --eval-store accepted."""
        from forma.cli_domain import _build_coverage_parser

        parser = _build_coverage_parser()
        args = parser.parse_args([
            "--concepts", "c.yaml",
            "--transcripts", "t.txt",
            "--output", "out.yaml",
            "--scope", "2장:확산",
            "--threshold", "0.7",
            "--week-config", "week.yaml",
            "--eval-store", "store.yaml",
        ])
        assert args.scope == "2장:확산"
        assert args.threshold == 0.7
        assert args.week_config == "week.yaml"
        assert args.eval_store == "store.yaml"

    def test_transcripts_repeatable(self) -> None:
        """--transcripts can be repeated."""
        from forma.cli_domain import _build_coverage_parser

        parser = _build_coverage_parser()
        args = parser.parse_args([
            "--concepts", "c.yaml",
            "--transcripts", "t1.txt",
            "--transcripts", "t2.txt",
            "--output", "out.yaml",
        ])
        assert args.transcripts == ["t1.txt", "t2.txt"]


# ----------------------------------------------------------------
# T032: Report CLI parsing
# ----------------------------------------------------------------


class TestReportCLIParsing:
    """Tests for 'forma domain report' CLI argument parsing."""

    def test_missing_coverage_arg_exits(self) -> None:
        """Missing --coverage exits with error."""
        with pytest.raises(SystemExit) as exc_info:
            report_main(["--output", "out.pdf"])
        assert exc_info.value.code != 0

    def test_missing_output_arg_exits(self) -> None:
        """Missing --output exits with error."""
        with pytest.raises(SystemExit) as exc_info:
            report_main(["--coverage", "coverage.yaml"])
        assert exc_info.value.code != 0

    def test_nonexistent_coverage_file_exits(self) -> None:
        """Non-existent coverage file exits with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            report_main([
                "--coverage", "/nonexistent/coverage.yaml",
                "--output", "/tmp/out.pdf",
            ])
        assert exc_info.value.code == 1

    def test_accepts_optional_flags(self) -> None:
        """Optional flags --course-name, --font-path, --dpi accepted."""
        from forma.cli_domain import _build_report_parser

        parser = _build_report_parser()
        args = parser.parse_args([
            "--coverage", "cov.yaml",
            "--output", "out.pdf",
            "--course-name", "인체구조와기능",
            "--font-path", "/path/to/font.ttf",
            "--dpi", "200",
        ])
        assert args.course_name == "인체구조와기능"
        assert args.font_path == "/path/to/font.ttf"
        assert args.dpi == 200
