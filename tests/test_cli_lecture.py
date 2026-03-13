"""Tests for forma.cli_lecture."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from forma.lecture_preprocessor import CleanedTranscript


def _make_cleaned(text: str = "세포막 단백질 인지질") -> CleanedTranscript:
    return CleanedTranscript(
        class_id="A",
        week=1,
        source_path="/tmp/test.txt",
        raw_text=text,
        cleaned_text=text,
        encoding_used="utf-8",
        char_count_raw=len(text),
        char_count_cleaned=len(text),
    )


def _make_analysis_result():
    from forma.lecture_analyzer import AnalysisResult
    return AnalysisResult(
        class_id="A",
        week=1,
        keyword_frequencies={"세포막": 2},
        top_keywords=["세포막"],
        network_image_path=None,
        topics=None,
        topic_skipped_reason="문장 수 부족",
        concept_coverage=None,
        emphasis_scores=None,
        triplets=None,
        triplet_skipped_reason=None,
        sentence_count=3,
        analysis_timestamp="2026-01-01T00:00:00",
    )


class TestMainAnalyze:
    """Test main_analyze() CLI entry point."""

    @patch("forma.cli_lecture.LectureReportGenerator")
    @patch("forma.cli_lecture.analyze_transcript")
    @patch("forma.cli_lecture.save_analysis_result")
    @patch("forma.cli_lecture.preprocess_transcript")
    def test_analyze_basic(
        self, mock_preprocess, mock_save, mock_analyze, mock_report_cls,
        tmp_path: Path,
    ) -> None:
        """Creates output files with valid args."""
        input_file = tmp_path / "lecture.txt"
        input_file.write_text("세포막 단백질 인지질", encoding="utf-8")
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        mock_preprocess.return_value = _make_cleaned()
        mock_analyze.return_value = _make_analysis_result()
        mock_save.return_value = output_dir / "analysis.yaml"
        mock_report_cls.return_value = MagicMock()

        from forma.cli_lecture import main_analyze
        main_analyze([
            "--input", str(input_file),
            "--output", str(output_dir),
            "--class", "A",
            "--week", "1",
            "--no-triplets",
        ])

        mock_preprocess.assert_called_once()
        mock_analyze.assert_called_once()

    @patch("forma.cli_lecture.LectureReportGenerator")
    @patch("forma.cli_lecture.analyze_transcript")
    @patch("forma.cli_lecture.save_analysis_result")
    @patch("forma.cli_lecture.preprocess_transcript")
    def test_analyze_with_concepts(
        self, mock_preprocess, mock_save, mock_analyze, mock_report_cls,
        tmp_path: Path,
    ) -> None:
        """--concepts flag triggers concept coverage."""
        input_file = tmp_path / "lecture.txt"
        input_file.write_text("세포막 단백질", encoding="utf-8")
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        concepts_file = tmp_path / "concepts.yaml"
        import yaml
        concepts_file.write_text(
            yaml.dump({"concepts": ["세포막", "단백질"]}, allow_unicode=True),
            encoding="utf-8",
        )

        mock_preprocess.return_value = _make_cleaned()
        mock_analyze.return_value = _make_analysis_result()
        mock_save.return_value = output_dir / "analysis.yaml"
        mock_report_cls.return_value = MagicMock()

        from forma.cli_lecture import main_analyze
        main_analyze([
            "--input", str(input_file),
            "--output", str(output_dir),
            "--class", "A",
            "--week", "1",
            "--concepts", str(concepts_file),
            "--no-triplets",
        ])

        # analyze_transcript should have been called with concepts
        call_kwargs = mock_analyze.call_args
        assert call_kwargs is not None

    @patch("forma.cli_lecture.LectureReportGenerator")
    @patch("forma.cli_lecture.analyze_transcript")
    @patch("forma.cli_lecture.save_analysis_result")
    @patch("forma.cli_lecture.load_analysis_result")
    @patch("forma.cli_lecture.preprocess_transcript")
    def test_analyze_no_cache(
        self, mock_preprocess, mock_load, mock_save, mock_analyze,
        mock_report_cls, tmp_path: Path,
    ) -> None:
        """--no-cache forces recompute even if cache exists."""
        input_file = tmp_path / "lecture.txt"
        input_file.write_text("세포막 단백질", encoding="utf-8")
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        mock_preprocess.return_value = _make_cleaned()
        mock_analyze.return_value = _make_analysis_result()
        mock_save.return_value = output_dir / "analysis.yaml"
        mock_report_cls.return_value = MagicMock()

        from forma.cli_lecture import main_analyze
        main_analyze([
            "--input", str(input_file),
            "--output", str(output_dir),
            "--class", "A",
            "--week", "1",
            "--no-cache",
            "--no-triplets",
        ])

        # With --no-cache, load_analysis_result should NOT be called
        mock_load.assert_not_called()
        mock_analyze.assert_called_once()

    def test_analyze_missing_input(self, tmp_path: Path) -> None:
        """Missing --input without week.yaml -> SystemExit."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        from forma.cli_lecture import main_analyze
        with pytest.raises(SystemExit):
            main_analyze([
                "--output", str(output_dir),
                "--class", "A",
                "--week", "1",
                "--no-triplets",
            ])

    def test_analyze_path_traversal(self, tmp_path: Path) -> None:
        """../ in path -> SystemExit with Korean error."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        from forma.cli_lecture import main_analyze
        with pytest.raises(SystemExit):
            main_analyze([
                "--input", "../../../etc/passwd",
                "--output", str(output_dir),
                "--class", "A",
                "--week", "1",
                "--no-triplets",
            ])

    def test_analyze_empty_transcript(self, tmp_path: Path) -> None:
        """Empty file -> SystemExit with Korean error."""
        input_file = tmp_path / "empty.txt"
        input_file.write_text("", encoding="utf-8")
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        from forma.cli_lecture import main_analyze
        with pytest.raises(SystemExit):
            main_analyze([
                "--input", str(input_file),
                "--output", str(output_dir),
                "--class", "A",
                "--week", "1",
                "--no-triplets",
            ])


def _make_analysis_result_for(class_id: str, week: int = 1):
    """Create an AnalysisResult with distinct keywords per class."""
    from forma.lecture_analyzer import AnalysisResult

    unique_kw = {"A": "ATP", "B": "DNA", "C": "RNA", "D": "GFR"}
    kw = unique_kw.get(class_id, "기타")
    return AnalysisResult(
        class_id=class_id,
        week=week,
        keyword_frequencies={"세포": 10, kw: 5},
        top_keywords=["세포", kw],
        network_image_path=None,
        topics=None,
        topic_skipped_reason="test",
        concept_coverage=None,
        emphasis_scores=None,
        triplets=None,
        triplet_skipped_reason=None,
        sentence_count=10,
        analysis_timestamp="2026-01-01T00:00:00",
    )


class TestMainCompare:
    """Test main_compare() CLI entry point."""

    @patch("forma.cli_lecture.LectureReportGenerator")
    @patch("forma.cli_lecture.load_analysis_result")
    def test_compare_basic(
        self, mock_load, mock_report_cls, tmp_path: Path,
    ) -> None:
        """--input-dir, --week, --classes, --output produces comparison files."""
        input_dir = tmp_path / "analyses"
        input_dir.mkdir()
        output_dir = tmp_path / "out"

        # Create analysis YAML stubs so path existence checks pass
        for cid in ("A", "B"):
            (input_dir / f"analysis_{cid}_w1.yaml").write_text("stub", encoding="utf-8")

        mock_load.side_effect = lambda p: _make_analysis_result_for(
            p.stem.split("_")[1], 1,
        )
        mock_report_cls.return_value = MagicMock()

        from forma.cli_lecture import main_compare
        main_compare([
            "--input-dir", str(input_dir),
            "--week", "1",
            "--classes", "A", "B",
            "--output", str(output_dir),
        ])

        assert mock_load.call_count == 2

    def test_compare_fewer_than_two_classes(self, tmp_path: Path) -> None:
        """Fewer than 2 classes exits with Korean error."""
        input_dir = tmp_path / "analyses"
        input_dir.mkdir()
        output_dir = tmp_path / "out"

        from forma.cli_lecture import main_compare
        with pytest.raises(SystemExit):
            main_compare([
                "--input-dir", str(input_dir),
                "--week", "1",
                "--classes", "A",
                "--output", str(output_dir),
            ])

    def test_compare_missing_analysis(self, tmp_path: Path) -> None:
        """Missing analysis YAML exits with Korean error."""
        input_dir = tmp_path / "analyses"
        input_dir.mkdir()
        output_dir = tmp_path / "out"

        from forma.cli_lecture import main_compare
        with pytest.raises(SystemExit):
            main_compare([
                "--input-dir", str(input_dir),
                "--week", "1",
                "--classes", "A", "B",
                "--output", str(output_dir),
            ])

    @patch("forma.cli_lecture.LectureReportGenerator")
    @patch("forma.cli_lecture.load_analysis_result")
    def test_compare_with_concepts(
        self, mock_load, mock_report_cls, tmp_path: Path,
    ) -> None:
        """--concepts flag triggers concept gap analysis."""
        input_dir = tmp_path / "analyses"
        input_dir.mkdir()
        output_dir = tmp_path / "out"

        for cid in ("A", "B"):
            (input_dir / f"analysis_{cid}_w1.yaml").write_text("stub", encoding="utf-8")

        mock_load.side_effect = lambda p: _make_analysis_result_for(
            p.stem.split("_")[1], 1,
        )
        mock_report_cls.return_value = MagicMock()

        concepts_file = tmp_path / "concepts.yaml"
        concepts_file.write_text(
            yaml.dump({"concepts": ["세포", "ATP", "DNA"]}, allow_unicode=True),
            encoding="utf-8",
        )

        from forma.cli_lecture import main_compare
        main_compare([
            "--input-dir", str(input_dir),
            "--week", "1",
            "--classes", "A", "B",
            "--output", str(output_dir),
            "--concepts", str(concepts_file),
        ])

        assert mock_load.call_count == 2


class TestMainClassCompare:
    """Test main_class_compare() CLI entry point."""

    @patch("forma.cli_lecture.LectureReportGenerator")
    @patch("forma.cli_lecture.save_merged_analysis")
    @patch("forma.cli_lecture.merge_analyses")
    @patch("forma.cli_lecture.load_analysis_result")
    def test_class_compare_basic(
        self, mock_load, mock_merge, mock_save_merged,
        mock_report_cls, tmp_path: Path,
    ) -> None:
        """--input-dir, --weeks, --classes, --output produces merged + comparison files."""
        input_dir = tmp_path / "analyses"
        input_dir.mkdir()
        output_dir = tmp_path / "out"

        # Create analysis YAML stubs for two classes, two weeks
        for cid in ("A", "B"):
            for w in (1, 2):
                (input_dir / f"analysis_{cid}_w{w}.yaml").write_text(
                    "stub", encoding="utf-8",
                )

        mock_load.side_effect = lambda p: _make_analysis_result_for(
            p.stem.split("_")[1],
            int(p.stem.split("w")[1]),
        )

        from forma.lecture_merge import MergedAnalysis

        mock_merge.side_effect = lambda analyses, class_id: MergedAnalysis(
            class_id=class_id,
            weeks=[1, 2],
            combined_keyword_frequencies={"세포": 20},
            per_session_keyword_frequencies={1: {"세포": 10}, 2: {"세포": 10}},
            session_boundary_markers=["--- Session 1 (Week 1) ---", "--- Session 2 (Week 2) ---"],
            merged_text="merged",
        )
        mock_save_merged.return_value = output_dir / "merged.yaml"
        mock_report_cls.return_value = MagicMock()

        from forma.cli_lecture import main_class_compare

        main_class_compare([
            "--input-dir", str(input_dir),
            "--weeks", "1", "2",
            "--classes", "A", "B",
            "--output", str(output_dir),
        ])

        # Should load 4 analysis files (2 classes x 2 weeks)
        assert mock_load.call_count == 4
        # Should merge 2 classes
        assert mock_merge.call_count == 2

    def test_class_compare_fewer_than_two_classes(self, tmp_path: Path) -> None:
        """Fewer than 2 classes exits with Korean error."""
        input_dir = tmp_path / "analyses"
        input_dir.mkdir()
        output_dir = tmp_path / "out"

        from forma.cli_lecture import main_class_compare

        with pytest.raises(SystemExit):
            main_class_compare([
                "--input-dir", str(input_dir),
                "--weeks", "1", "2",
                "--classes", "A",
                "--output", str(output_dir),
            ])

    def test_class_compare_missing_analysis(self, tmp_path: Path) -> None:
        """Missing per-session analysis YAML exits with Korean error."""
        input_dir = tmp_path / "analyses"
        input_dir.mkdir()
        output_dir = tmp_path / "out"

        # Only create files for class A, not B
        for w in (1, 2):
            (input_dir / f"analysis_A_w{w}.yaml").write_text(
                "stub", encoding="utf-8",
            )

        from forma.cli_lecture import main_class_compare

        with pytest.raises(SystemExit):
            main_class_compare([
                "--input-dir", str(input_dir),
                "--weeks", "1", "2",
                "--classes", "A", "B",
                "--output", str(output_dir),
            ])

    @patch("forma.cli_lecture.LectureReportGenerator")
    @patch("forma.cli_lecture.save_merged_analysis")
    @patch("forma.cli_lecture.merge_analyses")
    @patch("forma.cli_lecture.load_analysis_result")
    def test_class_compare_with_concepts(
        self, mock_load, mock_merge, mock_save_merged,
        mock_report_cls, tmp_path: Path,
    ) -> None:
        """--concepts flag triggers concept gap analysis in merged comparison."""
        input_dir = tmp_path / "analyses"
        input_dir.mkdir()
        output_dir = tmp_path / "out"

        for cid in ("A", "B"):
            for w in (1, 2):
                (input_dir / f"analysis_{cid}_w{w}.yaml").write_text(
                    "stub", encoding="utf-8",
                )

        mock_load.side_effect = lambda p: _make_analysis_result_for(
            p.stem.split("_")[1],
            int(p.stem.split("w")[1]),
        )

        from forma.lecture_merge import MergedAnalysis

        mock_merge.side_effect = lambda analyses, class_id: MergedAnalysis(
            class_id=class_id,
            weeks=[1, 2],
            combined_keyword_frequencies={"세포": 20},
            per_session_keyword_frequencies={1: {"세포": 10}, 2: {"세포": 10}},
            session_boundary_markers=["--- Session 1 (Week 1) ---", "--- Session 2 (Week 2) ---"],
            merged_text="merged",
        )
        mock_save_merged.return_value = output_dir / "merged.yaml"
        mock_report_cls.return_value = MagicMock()

        concepts_file = tmp_path / "concepts.yaml"
        concepts_file.write_text(
            yaml.dump({"concepts": ["세포", "ATP", "DNA"]}, allow_unicode=True),
            encoding="utf-8",
        )

        from forma.cli_lecture import main_class_compare

        main_class_compare([
            "--input-dir", str(input_dir),
            "--weeks", "1", "2",
            "--classes", "A", "B",
            "--output", str(output_dir),
            "--concepts", str(concepts_file),
        ])
