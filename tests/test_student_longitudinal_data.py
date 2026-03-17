"""Tests for student_longitudinal_data.py — cohort distributions, student data,
warning signals, ID CSV parsing, and anonymization.

TDD RED phase: tests written before implementation.
"""

from __future__ import annotations

import pytest

from forma.evaluation_types import LongitudinalRecord
from forma.longitudinal_store import LongitudinalStore
from forma.student_longitudinal_data import (
    AlertLevel,
    AnonymizedStudentSummary,
    CohortWeekStats,
    StudentLongitudinalData,
    anonymize,
    build_cohort_distribution,
    build_student_data,
    evaluate_warnings,
    parse_id_csv,
)


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


def _build_store(tmp_path, records: list[LongitudinalRecord]) -> LongitudinalStore:
    """Create and populate an in-memory store (no disk I/O needed for tests)."""
    path = str(tmp_path / "store.yaml")
    store = LongitudinalStore(path)
    for rec in records:
        store.add_record(rec)
    return store


# ---------------------------------------------------------------------------
# T003: CohortDistribution tests
# ---------------------------------------------------------------------------


class TestCohortDistribution:
    """build_cohort_distribution computes correct quartiles and stats."""

    def test_basic_stats_single_week(self, tmp_path):
        """Single week with 4 students should produce correct quartile stats."""
        records = [
            _make_record(student_id=f"s{i:03d}", week=1, question_sn=1, ensemble_score=score)
            for i, score in enumerate([0.4, 0.6, 0.8, 1.0], start=1)
        ] + [
            _make_record(student_id=f"s{i:03d}", week=1, question_sn=2, ensemble_score=score)
            for i, score in enumerate([0.3, 0.5, 0.7, 0.9], start=1)
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1])

        assert 1 in cohort.weekly_stats
        stats = cohort.weekly_stats[1]
        assert isinstance(stats, CohortWeekStats)
        assert stats.week == 1
        assert stats.n == 4
        # Each student's avg ensemble_score across Q1, Q2:
        # s001: (0.4+0.3)/2=0.35, s002: (0.6+0.5)/2=0.55
        # s003: (0.8+0.7)/2=0.75, s004: (1.0+0.9)/2=0.95
        assert stats.min == pytest.approx(0.35, abs=0.01)
        assert stats.max == pytest.approx(0.95, abs=0.01)
        assert stats.mean == pytest.approx(0.65, abs=0.01)

    def test_weekly_scores_contains_all_student_averages(self, tmp_path):
        """weekly_scores should contain per-student average ensemble_scores."""
        records = [
            _make_record(student_id="s001", week=1, question_sn=1, ensemble_score=0.6),
            _make_record(student_id="s001", week=1, question_sn=2, ensemble_score=0.8),
            _make_record(student_id="s002", week=1, question_sn=1, ensemble_score=0.5),
            _make_record(student_id="s002", week=1, question_sn=2, ensemble_score=0.7),
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1])

        scores = cohort.weekly_scores[1]
        assert len(scores) == 2
        assert sorted(scores) == pytest.approx([0.6, 0.7], abs=0.01)

    def test_multiple_weeks(self, tmp_path):
        """Stats should be computed per-week independently."""
        records = [
            _make_record(student_id="s001", week=1, question_sn=1, ensemble_score=0.5),
            _make_record(student_id="s002", week=1, question_sn=1, ensemble_score=0.9),
            _make_record(student_id="s001", week=2, question_sn=1, ensemble_score=0.3),
            _make_record(student_id="s002", week=2, question_sn=1, ensemble_score=0.7),
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1, 2])

        assert 1 in cohort.weekly_stats
        assert 2 in cohort.weekly_stats
        assert cohort.weekly_stats[1].mean == pytest.approx(0.7, abs=0.01)
        assert cohort.weekly_stats[2].mean == pytest.approx(0.5, abs=0.01)

    def test_per_question_concept_coverage(self, tmp_path):
        """weekly_q_scores should contain per-question concept_coverage lists."""
        records = [
            _make_record(student_id="s001", week=1, question_sn=1, concept_coverage=0.8),
            _make_record(student_id="s002", week=1, question_sn=1, concept_coverage=0.6),
            _make_record(student_id="s001", week=1, question_sn=2, concept_coverage=0.4),
            _make_record(student_id="s002", week=1, question_sn=2, concept_coverage=0.9),
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1])

        q1_scores = cohort.weekly_q_scores[1][1]
        assert sorted(q1_scores) == pytest.approx([0.6, 0.8], abs=0.01)
        q2_scores = cohort.weekly_q_scores[1][2]
        assert sorted(q2_scores) == pytest.approx([0.4, 0.9], abs=0.01)


# ---------------------------------------------------------------------------
# T004: StudentLongitudinalData tests
# ---------------------------------------------------------------------------


class TestStudentLongitudinalData:
    """build_student_data extracts correct per-week scores and trend."""

    def test_scores_by_week_extraction(self, tmp_path):
        """Should extract per-week, per-question score dicts."""
        records = [
            _make_record(student_id="s001", week=1, question_sn=1, ensemble_score=0.5, concept_coverage=0.4),
            _make_record(student_id="s001", week=1, question_sn=2, ensemble_score=0.6, concept_coverage=0.5),
            _make_record(student_id="s001", week=2, question_sn=1, ensemble_score=0.7, concept_coverage=0.6),
            # Other student data for cohort
            _make_record(student_id="s002", week=1, question_sn=1, ensemble_score=0.8),
            _make_record(student_id="s002", week=2, question_sn=1, ensemble_score=0.9),
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1, 2])
        data = build_student_data(store, "s001", weeks=[1, 2], cohort=cohort)

        assert isinstance(data, StudentLongitudinalData)
        assert data.student_id == "s001"
        assert data.weeks == [1, 2]
        assert 1 in data.scores_by_week
        assert 1 in data.scores_by_week[1]  # question_sn=1
        assert data.scores_by_week[1][1]["ensemble_score"] == pytest.approx(0.5)
        assert data.scores_by_week[1][2]["concept_coverage"] == pytest.approx(0.5)

    def test_trend_slope_positive(self, tmp_path):
        """Increasing scores should produce positive slope and '상승' direction."""
        records = [
            _make_record(student_id="s001", week=w, question_sn=1, ensemble_score=0.3 + w * 0.1)
            for w in range(1, 5)
        ] + [
            _make_record(student_id="s002", week=w, question_sn=1, ensemble_score=0.5)
            for w in range(1, 5)
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1, 2, 3, 4])
        data = build_student_data(store, "s001", weeks=[1, 2, 3, 4], cohort=cohort)

        assert data.trend_slope is not None
        assert data.trend_slope > 0.05
        assert data.trend_direction == "상승"

    def test_trend_slope_negative(self, tmp_path):
        """Decreasing scores should produce negative slope and '하강' direction."""
        records = [
            _make_record(student_id="s001", week=w, question_sn=1, ensemble_score=0.9 - w * 0.1)
            for w in range(1, 5)
        ] + [
            _make_record(student_id="s002", week=w, question_sn=1, ensemble_score=0.5)
            for w in range(1, 5)
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1, 2, 3, 4])
        data = build_student_data(store, "s001", weeks=[1, 2, 3, 4], cohort=cohort)

        assert data.trend_slope is not None
        assert data.trend_slope < -0.05
        assert data.trend_direction == "하강"

    def test_trend_slope_stable(self, tmp_path):
        """Flat scores should produce near-zero slope and '정체' direction."""
        records = [
            _make_record(student_id="s001", week=w, question_sn=1, ensemble_score=0.5)
            for w in range(1, 5)
        ] + [
            _make_record(student_id="s002", week=w, question_sn=1, ensemble_score=0.5)
            for w in range(1, 5)
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1, 2, 3, 4])
        data = build_student_data(store, "s001", weeks=[1, 2, 3, 4], cohort=cohort)

        assert data.trend_slope is not None
        assert abs(data.trend_slope) <= 0.05
        assert data.trend_direction == "정체"

    def test_single_week_insufficient_data(self, tmp_path):
        """Single week of data: trend_slope=None, trend_direction='데이터 부족'."""
        records = [
            _make_record(student_id="s001", week=1, question_sn=1, ensemble_score=0.5),
            _make_record(student_id="s002", week=1, question_sn=1, ensemble_score=0.8),
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1])
        data = build_student_data(store, "s001", weeks=[1], cohort=cohort)

        assert data.trend_slope is None
        assert data.trend_direction == "데이터 부족"

    def test_percentile_computation(self, tmp_path):
        """Student at top of cohort should have high percentile."""
        records = []
        for i in range(1, 11):
            records.append(
                _make_record(student_id=f"s{i:03d}", week=1, question_sn=1, ensemble_score=i * 0.1)
            )
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1])
        # s010 has ensemble=1.0, highest
        data = build_student_data(store, "s010", weeks=[1], cohort=cohort)
        assert data.percentiles_by_week[1] >= 90.0

        # s001 has ensemble=0.1, lowest
        data_low = build_student_data(store, "s001", weeks=[1], cohort=cohort)
        assert data_low.percentiles_by_week[1] <= 20.0


# ---------------------------------------------------------------------------
# T005: WarningSignal + AlertLevel tests
# ---------------------------------------------------------------------------


class TestWarningSignals:
    """evaluate_warnings returns correct signals and alert levels."""

    def test_warning_level_critical_low_score(self, tmp_path):
        """ensemble_score < 0.45 → risk_zone triggered (critical) → 경고."""
        records = [
            _make_record(student_id="s001", week=1, question_sn=1, ensemble_score=0.3),
            _make_record(student_id="s002", week=1, question_sn=1, ensemble_score=0.9),
            _make_record(student_id="s003", week=1, question_sn=1, ensemble_score=0.8),
            _make_record(student_id="s004", week=1, question_sn=1, ensemble_score=0.7),
            _make_record(student_id="s005", week=1, question_sn=1, ensemble_score=0.6),
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1])
        data = build_student_data(store, "s001", weeks=[1], cohort=cohort)
        signals, level = evaluate_warnings(data, cohort)

        assert level == AlertLevel.WARNING
        assert level.value == "경고"
        risk_zone = [s for s in signals if s.name == "위험 구간 진입" and s.triggered]
        assert len(risk_zone) == 1
        assert risk_zone[0].severity == "critical"

    def test_warning_level_critical_low_percentile(self, tmp_path):
        """Percentile < 20 → low_percentile triggered (critical) → 경고."""
        records = []
        for i in range(1, 11):
            records.append(
                _make_record(student_id=f"s{i:03d}", week=1, question_sn=1, ensemble_score=0.5 + i * 0.05)
            )
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1])
        # s001 has the lowest score
        data = build_student_data(store, "s001", weeks=[1], cohort=cohort)
        signals, level = evaluate_warnings(data, cohort)

        assert level == AlertLevel.WARNING
        low_pct = [s for s in signals if s.name == "하위 백분위" and s.triggered]
        assert len(low_pct) == 1
        assert low_pct[0].severity == "critical"

    def test_caution_level_non_critical_only(self, tmp_path):
        """Only non-critical signals (low coverage) → 주의."""
        # Student with low concept_coverage but decent ensemble_score
        records = [
            _make_record(
                student_id="s001", week=w, question_sn=1,
                concept_coverage=0.2, ensemble_score=0.6,
            )
            for w in range(1, 4)
        ]
        # Other students with higher scores to keep s001 above 20th percentile
        for i in range(2, 6):
            for w in range(1, 4):
                records.append(
                    _make_record(
                        student_id=f"s{i:03d}", week=w, question_sn=1,
                        concept_coverage=0.8, ensemble_score=0.5 + i * 0.02,
                    )
                )
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1, 2, 3])
        data = build_student_data(store, "s001", weeks=[1, 2, 3], cohort=cohort)
        signals, level = evaluate_warnings(data, cohort)

        assert level == AlertLevel.CAUTION
        assert level.value == "주의"
        low_cov = [s for s in signals if s.name == "저조한 개념 커버리지" and s.triggered]
        assert len(low_cov) == 1
        assert low_cov[0].severity == "non-critical"

    def test_normal_level_no_signals(self, tmp_path):
        """All metrics healthy → no signals triggered → 정상."""
        records = [
            _make_record(
                student_id="s001", week=w, question_sn=1,
                concept_coverage=0.8, ensemble_score=0.7,
            )
            for w in range(1, 4)
        ]
        for i in range(2, 6):
            for w in range(1, 4):
                records.append(
                    _make_record(
                        student_id=f"s{i:03d}", week=w, question_sn=1,
                        concept_coverage=0.8, ensemble_score=0.5,
                    )
                )
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1, 2, 3])
        data = build_student_data(store, "s001", weeks=[1, 2, 3], cohort=cohort)
        signals, level = evaluate_warnings(data, cohort)

        assert level == AlertLevel.NORMAL
        assert level.value == "정상"
        triggered = [s for s in signals if s.triggered]
        assert len(triggered) == 0

    def test_insufficient_data_marks_trend_signals(self, tmp_path):
        """< 3 weeks: trend-based signals require_min_weeks=3 → not triggered, detail='데이터 부족'."""
        records = [
            _make_record(student_id="s001", week=1, question_sn=1, ensemble_score=0.5),
            _make_record(student_id="s001", week=2, question_sn=1, ensemble_score=0.3),
            _make_record(student_id="s002", week=1, question_sn=1, ensemble_score=0.5),
            _make_record(student_id="s002", week=2, question_sn=1, ensemble_score=0.5),
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1, 2])
        data = build_student_data(store, "s001", weeks=[1, 2], cohort=cohort)
        signals, _ = evaluate_warnings(data, cohort)

        trend_signals = [s for s in signals if s.requires_min_weeks >= 3]
        for sig in trend_signals:
            assert not sig.triggered
            assert "데이터 부족" in sig.detail

    def test_consecutive_decline_signal(self, tmp_path):
        """3+ weeks of declining ensemble_score → consecutive decline triggered."""
        records = [
            _make_record(student_id="s001", week=1, question_sn=1, ensemble_score=0.8),
            _make_record(student_id="s001", week=2, question_sn=1, ensemble_score=0.7),
            _make_record(student_id="s001", week=3, question_sn=1, ensemble_score=0.6),
        ]
        # Other students keep s001's percentile above 20 and ensemble above 0.45
        for i in range(2, 7):
            for w in range(1, 4):
                records.append(
                    _make_record(
                        student_id=f"s{i:03d}", week=w, question_sn=1,
                        concept_coverage=0.8, ensemble_score=0.4,
                    )
                )
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1, 2, 3])
        data = build_student_data(store, "s001", weeks=[1, 2, 3], cohort=cohort)
        signals, level = evaluate_warnings(data, cohort)

        decline = [s for s in signals if s.name == "연속 하강" and s.triggered]
        assert len(decline) == 1
        assert decline[0].severity == "non-critical"


# ---------------------------------------------------------------------------
# T006: parse_id_csv tests
# ---------------------------------------------------------------------------


class TestParseIdCsv:
    """parse_id_csv correctly parses Google Forms CSV."""

    def test_basic_parsing(self, tmp_path):
        """Parses CSV with standard Google Forms columns."""
        csv_path = tmp_path / "id.csv"
        csv_path.write_text(
            "타임스탬프,익명ID,분반을 선택하세요.,학번을 입력하세요.,이름을 입력하세요.\n"
            "2026/03/10,abc123,A반,20210001,홍길동\n"
            "2026/03/10,def456,B반,20210002,김영희\n",
            encoding="utf-8",
        )
        result = parse_id_csv(str(csv_path))

        assert isinstance(result, dict)
        assert "20210001" in result
        assert result["20210001"] == ("홍길동", "A반")
        assert result["20210002"] == ("김영희", "B반")

    def test_missing_file_returns_empty(self, tmp_path):
        """Missing CSV file should return empty dict (or raise)."""
        result = parse_id_csv(str(tmp_path / "nonexistent.csv"))
        assert result == {}

    def test_malformed_csv_missing_columns(self, tmp_path):
        """CSV without required columns should return empty dict."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(
            "col1,col2,col3\na,b,c\n",
            encoding="utf-8",
        )
        result = parse_id_csv(str(csv_path))
        assert result == {}

    def test_duplicate_student_id_last_wins(self, tmp_path):
        """Duplicate 학번 entries: last row wins."""
        csv_path = tmp_path / "dup.csv"
        csv_path.write_text(
            "타임스탬프,익명ID,분반을 선택하세요.,학번을 입력하세요.,이름을 입력하세요.\n"
            "2026/03/10,abc123,A반,20210001,홍길동\n"
            "2026/03/11,abc123,B반,20210001,홍길순\n",
            encoding="utf-8",
        )
        result = parse_id_csv(str(csv_path))
        assert result["20210001"] == ("홍길순", "B반")


# ---------------------------------------------------------------------------
# T011: AnonymizedStudentSummary tests
# ---------------------------------------------------------------------------


class TestAnonymize:
    """anonymize() produces AnonymizedStudentSummary with zero PII."""

    def test_no_pii_in_summary(self, tmp_path):
        """Summary should not contain student_id, student_name, or class_name."""
        records = [
            _make_record(student_id="s001", week=1, question_sn=1, ensemble_score=0.6, concept_coverage=0.5),
            _make_record(student_id="s001", week=1, question_sn=2, ensemble_score=0.7, concept_coverage=0.6),
            _make_record(student_id="s002", week=1, question_sn=1, ensemble_score=0.5),
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1])
        data = build_student_data(store, "s001", weeks=[1], cohort=cohort)
        data.student_name = "홍길동"
        data.class_name = "A반"

        signals, level = evaluate_warnings(data, cohort)
        summary = anonymize(data, signals)

        assert isinstance(summary, AnonymizedStudentSummary)
        # Check no PII leakage
        summary_str = str(summary)
        assert "s001" not in summary_str
        assert "홍길동" not in summary_str
        assert "A반" not in summary_str

    def test_summary_contains_required_fields(self, tmp_path):
        """Summary should contain numerical fields for LLM interpretation."""
        records = [
            _make_record(student_id="s001", week=1, question_sn=1, ensemble_score=0.6, concept_coverage=0.5),
            _make_record(student_id="s001", week=1, question_sn=2, ensemble_score=0.7, concept_coverage=0.6),
            _make_record(student_id="s002", week=1, question_sn=1, ensemble_score=0.5),
        ]
        store = _build_store(tmp_path, records)
        cohort = build_cohort_distribution(store, weeks=[1])
        data = build_student_data(store, "s001", weeks=[1], cohort=cohort)

        signals, level = evaluate_warnings(data, cohort)
        summary = anonymize(data, signals)

        assert hasattr(summary, "weekly_coverage_q1")
        assert hasattr(summary, "weekly_coverage_q2")
        assert hasattr(summary, "weekly_ensemble")
        assert hasattr(summary, "percentiles")
        assert hasattr(summary, "trend_slope")
        assert hasattr(summary, "trend_direction")
        assert hasattr(summary, "alert_level")
        assert hasattr(summary, "triggered_signals")
        assert hasattr(summary, "component_breakdown")
