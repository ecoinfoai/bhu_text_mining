"""Tests for student_longitudinal_summary.py — cohort summary table PDF report.

TDD RED phase: tests written before implementation.
"""

from __future__ import annotations

import os

import yaml

from forma.student_longitudinal_data import AlertLevel


# ---------------------------------------------------------------------------
# Helpers — minimal longitudinal store YAML for testing
# ---------------------------------------------------------------------------


def _make_store_yaml(tmp_path: str, students: dict) -> str:
    """Create a minimal longitudinal store YAML file.

    Args:
        tmp_path: Directory for the temp file.
        students: {student_id: {week: {qsn: {metric: value}}}}.

    Returns:
        Path to the YAML file.
    """
    records = {}
    for sid, weeks in students.items():
        for week, questions in weeks.items():
            for qsn, scores in questions.items():
                key = f"{sid}_{week}_{qsn}"
                records[key] = {
                    "student_id": sid,
                    "week": week,
                    "question_sn": qsn,
                    "scores": scores,
                    "tier_level": 1,
                    "tier_label": "A",
                    "manual_override": False,
                }
    store_path = os.path.join(tmp_path, "store.yaml")
    with open(store_path, "w") as f:
        yaml.dump({"records": records}, f)
    return store_path


def _make_id_csv(tmp_path: str, students: list[tuple[str, str, str]]) -> str:
    """Create a minimal ID CSV file.

    Args:
        tmp_path: Directory for the temp file.
        students: List of (student_id, name, class_name).

    Returns:
        Path to the CSV file.
    """
    csv_path = os.path.join(tmp_path, "ids.csv")
    with open(csv_path, "w") as f:
        f.write("타임스탬프,익명ID,분반을 선택하세요.,학번을 입력하세요.,이름을 입력하세요.\n")
        for sid, name, cls in students:
            f.write(f"2024-01-01,anon,{cls},{sid},{name}\n")
    return csv_path


# ---------------------------------------------------------------------------
# Test: build_summary_rows
# ---------------------------------------------------------------------------


class TestBuildSummaryRows:
    """Tests for build_summary_rows function."""

    def test_build_summary_rows(self, tmp_path):
        """build_summary_rows returns correct rows with all expected fields."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.student_longitudinal_data import (
            build_cohort_distribution,
            parse_id_csv,
        )
        from forma.student_longitudinal_summary import build_summary_rows

        store_path = _make_store_yaml(
            str(tmp_path),
            {
                "s001": {
                    1: {1: {"ensemble_score": 0.6, "concept_coverage": 0.5}},
                    2: {1: {"ensemble_score": 0.7, "concept_coverage": 0.6}},
                },
                "s002": {
                    1: {1: {"ensemble_score": 0.3, "concept_coverage": 0.2}},
                    2: {1: {"ensemble_score": 0.25, "concept_coverage": 0.15}},
                },
            },
        )
        csv_path = _make_id_csv(
            str(tmp_path),
            [
                ("s001", "김철수", "A"),
                ("s002", "이영희", "B"),
            ],
        )

        store = LongitudinalStore(store_path)
        store.load()
        weeks = [1, 2]
        cohort = build_cohort_distribution(store, weeks)
        id_map = parse_id_csv(csv_path)

        rows = build_summary_rows(store, weeks, cohort, id_map)

        assert len(rows) == 2
        # Check fields exist
        for row in rows:
            assert hasattr(row, "student_id")
            assert hasattr(row, "student_name")
            assert hasattr(row, "class_name")
            assert hasattr(row, "weekly_ensemble")
            assert hasattr(row, "trend_direction")
            assert hasattr(row, "latest_percentile")
            assert hasattr(row, "alert_level")
            assert hasattr(row, "triggered_signals")

        # s001 should have weekly ensemble data for weeks 1 and 2
        s001 = next(r for r in rows if r.student_id == "s001")
        assert 1 in s001.weekly_ensemble
        assert 2 in s001.weekly_ensemble
        assert s001.student_name == "김철수"
        assert s001.class_name == "A"

    def test_summary_sorted_by_warning_then_name(self, tmp_path):
        """Rows are sorted: WARNING first, then CAUTION, then NORMAL, then by student_id."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.student_longitudinal_data import (
            build_cohort_distribution,
            parse_id_csv,
        )
        from forma.student_longitudinal_summary import build_summary_rows

        # s001: high scores -> NORMAL
        # s002: very low scores -> WARNING (critical signal)
        # s003: medium-low scores -> could be CAUTION
        store_path = _make_store_yaml(
            str(tmp_path),
            {
                "s001": {
                    1: {1: {"ensemble_score": 0.8, "concept_coverage": 0.9}},
                    2: {1: {"ensemble_score": 0.85, "concept_coverage": 0.9}},
                },
                "s002": {
                    1: {1: {"ensemble_score": 0.1, "concept_coverage": 0.1}},
                    2: {1: {"ensemble_score": 0.05, "concept_coverage": 0.05}},
                },
                "s003": {
                    1: {1: {"ensemble_score": 0.5, "concept_coverage": 0.5}},
                    2: {1: {"ensemble_score": 0.48, "concept_coverage": 0.28}},
                },
            },
        )
        csv_path = _make_id_csv(
            str(tmp_path),
            [
                ("s001", "김정상", "A"),
                ("s002", "이경고", "A"),
                ("s003", "박주의", "B"),
            ],
        )

        store = LongitudinalStore(store_path)
        store.load()
        weeks = [1, 2]
        cohort = build_cohort_distribution(store, weeks)
        id_map = parse_id_csv(csv_path)

        rows = build_summary_rows(store, weeks, cohort, id_map)

        assert len(rows) == 3
        # WARNING students should come first
        alert_order = [r.alert_level for r in rows]
        # Verify that WARNING comes before CAUTION comes before NORMAL
        warning_idx = [i for i, a in enumerate(alert_order) if a == AlertLevel.WARNING]
        caution_idx = [i for i, a in enumerate(alert_order) if a == AlertLevel.CAUTION]
        normal_idx = [i for i, a in enumerate(alert_order) if a == AlertLevel.NORMAL]

        if warning_idx and caution_idx:
            assert max(warning_idx) < min(caution_idx)
        if caution_idx and normal_idx:
            assert max(caution_idx) < min(normal_idx)
        if warning_idx and normal_idx:
            assert max(warning_idx) < min(normal_idx)


# ---------------------------------------------------------------------------
# Test: CohortSummaryPDFReportGenerator
# ---------------------------------------------------------------------------


class TestCohortSummaryReport:
    """Tests for CohortSummaryPDFReportGenerator."""

    def test_generate_summary_pdf_creates_file(self, tmp_path):
        """generate_pdf creates a non-zero PDF file."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.student_longitudinal_data import (
            build_cohort_distribution,
            parse_id_csv,
        )
        from forma.student_longitudinal_summary import (
            CohortSummaryPDFReportGenerator,
            build_summary_rows,
        )

        store_path = _make_store_yaml(
            str(tmp_path),
            {
                "s001": {
                    1: {1: {"ensemble_score": 0.6, "concept_coverage": 0.5}},
                    2: {1: {"ensemble_score": 0.7, "concept_coverage": 0.6}},
                },
                "s002": {
                    1: {1: {"ensemble_score": 0.3, "concept_coverage": 0.2}},
                    2: {1: {"ensemble_score": 0.25, "concept_coverage": 0.15}},
                },
            },
        )
        csv_path = _make_id_csv(
            str(tmp_path),
            [
                ("s001", "김철수", "A"),
                ("s002", "이영희", "B"),
            ],
        )

        store = LongitudinalStore(store_path)
        store.load()
        weeks = [1, 2]
        cohort = build_cohort_distribution(store, weeks)
        id_map = parse_id_csv(csv_path)
        rows = build_summary_rows(store, weeks, cohort, id_map)

        output_pdf = os.path.join(str(tmp_path), "summary.pdf")
        gen = CohortSummaryPDFReportGenerator()
        result = gen.generate_pdf(rows, weeks, output_pdf, course_name="테스트 과목")

        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_generate_summary_pdf_with_empty_store(self, tmp_path):
        """Empty store produces a PDF with '데이터 없음' message."""
        from forma.student_longitudinal_summary import (
            CohortSummaryPDFReportGenerator,
        )

        output_pdf = os.path.join(str(tmp_path), "empty_summary.pdf")
        gen = CohortSummaryPDFReportGenerator()
        result = gen.generate_pdf([], [1, 2], output_pdf)

        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0


# ---------------------------------------------------------------------------
# Test: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLISummary:
    """Tests for forma-report-student-summary CLI."""

    def test_cli_summary_args(self):
        """CLI parser accepts required and optional arguments."""
        from forma.cli_report_student import _build_summary_parser

        parser = _build_summary_parser()
        args = parser.parse_args(
            [
                "--store",
                "store.yaml",
                "--id-csv",
                "ids.csv",
                "--output",
                "out.pdf",
                "--weeks",
                "1",
                "2",
                "3",
                "--course-name",
                "생물학개론",
            ]
        )

        assert args.store == "store.yaml"
        assert args.id_csv == "ids.csv"
        assert args.output == "out.pdf"
        assert args.weeks == ["1", "2", "3"]
        assert args.course_name == "생물학개론"
