"""Adversarial tests for 023-longitudinal-report-overhaul.

12 personas systematically attack the new longitudinal report
features to find crashes, wrong output, and misleading results.
"""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import pytest
import yaml

from forma.evaluation_types import LongitudinalRecord
from forma.longitudinal_store import LongitudinalStore


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_record(
    student_id: str = "S001",
    week: int = 1,
    question_sn: int = 1,
    ensemble_score: float = 0.5,
    tier_level: int = 1,
    tier_label: str = "Developing",
    concept_scores: dict[str, float] | None = None,
    topic: str | None = None,
    class_id: str | None = None,
) -> LongitudinalRecord:
    """Build a LongitudinalRecord with new fields."""
    return LongitudinalRecord(
        student_id=student_id,
        week=week,
        question_sn=question_sn,
        scores={"ensemble_score": ensemble_score},
        tier_level=tier_level,
        tier_label=tier_label,
        concept_scores=concept_scores,
        topic=topic,
        class_id=class_id,
    )


def _build_store_with_records(
    tmp_path: Path,
    records: list[LongitudinalRecord],
    filename: str = "store.yaml",
) -> LongitudinalStore:
    """Create a store, add records, save, and return it."""
    path = str(tmp_path / filename)
    store = LongitudinalStore(path)
    for rec in records:
        store.add_record(rec)
    store.save()
    return store


def _run_cli(
    args: list[str],
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    """Run a forma CLI command and capture output."""
    cmd = ["uv", "run", "forma"] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(Path(__file__).resolve().parent.parent),
    )


# ===================================================================
# Persona 1: 컴퓨터 초보 교수
# Typos, wrong arg names, missing required flags
# ===================================================================


class TestPersona01ComputerBeginner:
    """Professor who types wrong argument names and forgets flags."""

    def test_missing_required_store(self):
        """Missing --store should give clear error, not traceback."""
        result = _run_cli(
            [
                "report",
                "longitudinal",
                "--class-name",
                "A반",
                "--output",
                "/tmp/out.pdf",
            ]
        )
        assert result.returncode != 0
        # Should mention --store in the error
        combined = result.stdout + result.stderr
        assert "store" in combined.lower() or "required" in combined.lower()

    def test_missing_required_class_name(self):
        """Missing --class-name should give clear error."""
        result = _run_cli(
            [
                "report",
                "longitudinal",
                "--store",
                "/tmp/fake.yaml",
                "--output",
                "/tmp/out.pdf",
            ]
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "class" in combined.lower() or "required" in combined.lower()

    def test_wrong_arg_name_typo(self):
        """--stores (typo) should fail gracefully."""
        result = _run_cli(
            [
                "report",
                "longitudinal",
                "--stores",
                "/tmp/fake.yaml",
                "--class-name",
                "A",
                "--output",
                "/tmp/out.pdf",
            ]
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        # Should NOT produce a Python traceback
        assert "Traceback" not in combined

    def test_nonexistent_store_file(self, tmp_path):
        """Store file that doesn't exist should fail with clear message."""
        result = _run_cli(
            [
                "report",
                "longitudinal",
                "--store",
                str(tmp_path / "nonexistent.yaml"),
                "--class-name",
                "A",
                "--output",
                str(tmp_path / "out.pdf"),
            ]
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "not found" in combined.lower() or "no such" in combined.lower()


# ===================================================================
# Persona 2: 500명 대규모 강좌 교수
# 10 classes x 50 students, performance stress test
# ===================================================================


class TestPersona02LargeScale:
    """Professor with 500 students across 10 classes."""

    def test_500_students_store_roundtrip(self, tmp_path):
        """500 students x 4 weeks should save/load without error."""
        records = []
        classes = list("ABCDEFGHIJ")  # 10 classes
        for cls_idx, cls in enumerate(classes):
            for stu in range(50):
                sid = f"S{cls_idx * 50 + stu + 1:04d}"
                for week in range(1, 5):
                    records.append(
                        _make_record(
                            student_id=sid,
                            week=week,
                            question_sn=1,
                            ensemble_score=random.uniform(0.2, 0.9),
                            class_id=cls,
                            topic="개념이해",
                        )
                    )
        _build_store_with_records(tmp_path, records)
        # Reload and verify count
        store2 = LongitudinalStore(str(tmp_path / "store.yaml"))
        store2.load()
        all_recs = store2.get_all_records()
        assert len(all_recs) == 2000  # 500 * 4 weeks

    def test_500_students_summary_build(self, tmp_path):
        """build_longitudinal_summary with 500 students should not crash."""
        records = []
        for stu in range(500):
            sid = f"S{stu + 1:04d}"
            for week in range(1, 5):
                records.append(
                    _make_record(
                        student_id=sid,
                        week=week,
                        question_sn=1,
                        ensemble_score=random.uniform(0.2, 0.9),
                        class_id="A",
                        topic="개념이해",
                    )
                )
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(
            store,
            [1, 2, 3, 4],
            "대규모 강좌",
        )
        assert summary.total_students == 500


# ===================================================================
# Persona 3: 악의적 입력 학생
# SQL injection, XSS, unicode bombs, huge text in student_id
# ===================================================================


class TestPersona03MaliciousInput:
    """Student with malicious content in responses."""

    def test_sql_injection_student_id(self, tmp_path):
        """Student ID with SQL injection should be stored literally."""
        evil_id = "'; DROP TABLE students; --"
        rec = _make_record(student_id=evil_id, week=1)
        store = _build_store_with_records(tmp_path, [rec])
        store.load()
        recs = store.get_all_records()
        assert len(recs) == 1
        assert recs[0].student_id == evil_id

    def test_xss_in_student_id(self, tmp_path):
        """XSS payload in student ID should not cause issues."""
        evil_id = '<script>alert("XSS")</script>'
        rec = _make_record(student_id=evil_id, week=1)
        store = _build_store_with_records(tmp_path, [rec])
        store.load()
        recs = store.get_all_records()
        assert recs[0].student_id == evil_id

    def test_unicode_bomb_student_id(self, tmp_path):
        """Zalgo text / combining characters in student ID."""
        zalgo = "S\u0300\u0301\u0302\u0303\u0304\u0305001"
        rec = _make_record(student_id=zalgo, week=1)
        store = _build_store_with_records(tmp_path, [rec])
        store.load()
        recs = store.get_all_records()
        assert len(recs) == 1

    def test_very_long_student_id(self, tmp_path):
        """10,000 char student ID should not crash store."""
        long_id = "S" * 10000
        rec = _make_record(student_id=long_id, week=1)
        store = _build_store_with_records(tmp_path, [rec])
        store.load()
        recs = store.get_all_records()
        assert len(recs) == 1

    def test_null_bytes_in_student_id(self, tmp_path):
        """Null bytes in student ID should be handled."""
        evil_id = "S001\x00DROP"
        rec = _make_record(student_id=evil_id, week=1)
        store = _build_store_with_records(tmp_path, [rec])
        store.load()
        recs = store.get_all_records()
        assert len(recs) == 1


# ===================================================================
# Persona 4: Legacy store 사용자
# Old longitudinal.yaml without topic/class_id fields
# ===================================================================


class TestPersona04LegacyStore:
    """User with old longitudinal store missing new fields."""

    def test_load_legacy_store_no_topic_no_class_id(self, tmp_path):
        """Legacy store without topic/class_id should load without error."""
        legacy_data = {
            "records": {
                "S001_1_1": {
                    "student_id": "S001",
                    "week": 1,
                    "question_sn": 1,
                    "scores": {"ensemble_score": 0.6},
                    "tier_level": 2,
                    "tier_label": "Developing",
                    "manual_override": False,
                },
                "S001_2_1": {
                    "student_id": "S001",
                    "week": 2,
                    "question_sn": 1,
                    "scores": {"ensemble_score": 0.7},
                    "tier_level": 2,
                    "tier_label": "Developing",
                    "manual_override": False,
                },
            }
        }
        path = str(tmp_path / "legacy.yaml")
        with open(path, "w") as f:
            yaml.dump(legacy_data, f)

        store = LongitudinalStore(path)
        store.load()
        recs = store.get_all_records()
        assert len(recs) == 2
        # topic and class_id should be None (or not present)
        for r in recs:
            if hasattr(r, "topic"):
                assert r.topic is None
            if hasattr(r, "class_id"):
                assert r.class_id is None

    def test_mixed_legacy_and_new_store(self, tmp_path):
        """Store with some records having topic and others not."""
        mixed_data = {
            "records": {
                "S001_1_1": {
                    "student_id": "S001",
                    "week": 1,
                    "question_sn": 1,
                    "scores": {"ensemble_score": 0.6},
                    "tier_level": 2,
                    "tier_label": "Developing",
                    "manual_override": False,
                    # No topic, no class_id
                },
                "S002_1_1": {
                    "student_id": "S002",
                    "week": 1,
                    "question_sn": 1,
                    "scores": {"ensemble_score": 0.7},
                    "tier_level": 2,
                    "tier_label": "Developing",
                    "manual_override": False,
                    "topic": "개념이해",
                    "class_id": "A",
                },
            }
        }
        path = str(tmp_path / "mixed.yaml")
        with open(path, "w") as f:
            yaml.dump(mixed_data, f)

        store = LongitudinalStore(path)
        store.load()
        recs = store.get_all_records()
        assert len(recs) == 2


# ===================================================================
# Persona 5: Windows 사용자
# Backslash paths, CRLF, CP949 encoding
# ===================================================================


class TestPersona05WindowsUser:
    """User on Windows with different path/encoding conventions."""

    def test_crlf_yaml(self, tmp_path):
        """YAML with CRLF line endings should parse correctly."""
        content = (
            "records:\r\n"
            "  S001_1_1:\r\n"
            "    student_id: S001\r\n"
            "    week: 1\r\n"
            "    question_sn: 1\r\n"
            "    scores:\r\n"
            "      ensemble_score: 0.6\r\n"
            "    tier_level: 2\r\n"
            "    tier_label: Developing\r\n"
            "    manual_override: false\r\n"
        )
        path = str(tmp_path / "crlf.yaml")
        with open(path, "wb") as f:
            f.write(content.encode("utf-8"))

        store = LongitudinalStore(path)
        store.load()
        recs = store.get_all_records()
        assert len(recs) == 1

    def test_unicode_path_with_korean(self, tmp_path):
        """Store at a path containing Korean characters."""
        korean_dir = tmp_path / "데이터" / "1주차"
        korean_dir.mkdir(parents=True)
        path = str(korean_dir / "store.yaml")

        store = LongitudinalStore(path)
        store.add_record(_make_record())
        store.save()

        store2 = LongitudinalStore(path)
        store2.load()
        assert len(store2.get_all_records()) == 1


# ===================================================================
# Persona 6: 1주차만 있는 교수
# Only 1 week of data, requests trends
# ===================================================================


class TestPersona06SingleWeek:
    """Professor with only 1 week of data."""

    def test_single_week_summary(self, tmp_path):
        """build_longitudinal_summary with 1 week should not crash."""
        records = [
            _make_record(
                student_id=f"S{i:03d}",
                week=1,
                ensemble_score=random.uniform(0.3, 0.8),
                topic="개념이해",
                class_id="A",
            )
            for i in range(10)
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()

        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(store, [1], "A반")
        assert summary.total_students == 10
        # With 1 week, trend should be 0 or NaN-safe
        for traj in summary.student_trajectories:
            # Trend with single point should not be NaN
            assert traj.overall_trend == 0.0 or not (
                traj.overall_trend != traj.overall_trend  # NaN check
            )

    def test_single_week_no_persistent_risk(self, tmp_path):
        """With 1 week, persistent risk = at_risk in that single week."""
        records = [
            _make_record(
                student_id="S001",
                week=1,
                ensemble_score=0.2,  # below 0.45
                topic="개념이해",
                class_id="A",
            )
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(store, [1], "A반")
        # This student should be marked as persistent risk
        assert "S001" in summary.persistent_risk_students


# ===================================================================
# Persona 7: 15주차 누적 교수
# 15 weeks x 4 classes, 100+ concepts
# ===================================================================


class TestPersona07FifteenWeeks:
    """Professor with 15 weeks and many concepts."""

    def test_15_weeks_summary_build(self, tmp_path):
        """15 weeks x 20 students should build summary without crash."""
        records = []
        for week in range(1, 16):
            for stu in range(20):
                sid = f"S{stu + 1:03d}"
                # Generate many concepts
                concepts = {f"concept_{c}": random.uniform(0, 1) for c in range(10)}
                records.append(
                    _make_record(
                        student_id=sid,
                        week=week,
                        question_sn=1,
                        ensemble_score=random.uniform(0.3, 0.9),
                        concept_scores=concepts,
                        topic="개념이해" if week % 2 == 0 else "적용",
                        class_id="A",
                    )
                )
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        weeks = list(range(1, 16))
        summary = build_longitudinal_summary(store, weeks, "장기추적")
        assert summary.total_students == 20
        assert len(summary.period_weeks) == 15

    def test_100_concepts_mastery(self, tmp_path):
        """100+ concepts should not crash mastery calculation."""
        records = []
        for week in [1, 2, 3]:
            for stu in range(5):
                concepts = {f"concept_{c:03d}": random.uniform(0, 1) for c in range(120)}
                records.append(
                    _make_record(
                        student_id=f"S{stu + 1:03d}",
                        week=week,
                        question_sn=1,
                        ensemble_score=0.6,
                        concept_scores=concepts,
                        topic="개념이해",
                        class_id="A",
                    )
                )
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(
            store,
            [1, 2, 3],
            "많은개념",
        )
        # Should have mastery changes for all 120 concepts
        assert len(summary.concept_mastery_changes) > 0


# ===================================================================
# Persona 8: topic 미정의 exam
# Exam YAML with no topic field on any question
# ===================================================================


class TestPersona08NoTopicExam:
    """Exam YAML where no questions have topic field."""

    def test_records_without_topic(self, tmp_path):
        """Records with topic=None should still produce valid summary."""
        records = [
            _make_record(
                student_id=f"S{i:03d}",
                week=w,
                ensemble_score=random.uniform(0.3, 0.8),
                topic=None,
                class_id="A",
            )
            for i in range(5)
            for w in range(1, 4)
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(store, [1, 2, 3], "무토픽")
        assert summary.total_students == 5
        # Should fall back to per-question tracking without error

    def test_no_topic_no_crash_on_student_trajectory(self, tmp_path):
        """Student trajectory grouping should work even without topic."""
        records = [
            _make_record(
                student_id="S001",
                week=w,
                question_sn=q,
                ensemble_score=0.5 + w * 0.05,
                topic=None,
            )
            for w in range(1, 4)
            for q in [1, 2]
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        traj = store.get_student_trajectory(
            "S001",
            "ensemble_score",
        )
        assert len(traj) == 3  # 3 weeks, averaged per week


# ===================================================================
# Persona 9: 동시 실행 교수
# 4 terminals writing to same longitudinal.yaml
# ===================================================================


class TestPersona09ConcurrentAccess:
    """Professor running 4 terminals writing to same file."""

    def test_concurrent_saves_no_corruption(self, tmp_path):
        """Multiple sequential saves should not corrupt the store."""
        path = str(tmp_path / "concurrent.yaml")
        # Simulate rapid sequential saves (not truly concurrent
        # but tests the file locking mechanism)
        for i in range(10):
            store = LongitudinalStore(path)
            if os.path.exists(path):
                store.load()
            store.add_record(
                _make_record(
                    student_id=f"S{i:03d}",
                    week=1,
                    ensemble_score=0.5,
                )
            )
            store.save()

        # Final load should have all 10 records
        final = LongitudinalStore(path)
        final.load()
        assert len(final.get_all_records()) == 10

    def test_backup_file_created(self, tmp_path):
        """Save should create a .bak backup file."""
        path = str(tmp_path / "backup_test.yaml")
        store = LongitudinalStore(path)
        store.add_record(_make_record())
        store.save()
        assert os.path.exists(path + ".bak")


# ===================================================================
# Persona 10: 한글 분반명 교수
# class_id = "가", "나", "다" instead of A, B, C
# ===================================================================


class TestPersona10KoreanClassNames:
    """Professor using Korean class names."""

    def test_korean_class_id_store(self, tmp_path):
        """Korean class_id should be stored and loaded correctly."""
        records = [
            _make_record(
                student_id=f"S{cls_idx * 3 + i:03d}",
                week=1,
                class_id=cls,
                topic="개념이해",
            )
            for cls_idx, cls in enumerate(["가", "나", "다"])
            for i in range(3)
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        recs = store.get_all_records()
        class_ids = {getattr(r, "class_id", None) for r in recs}
        # If class_id is supported, verify Korean values
        if None not in class_ids:
            assert "가" in class_ids
            assert "나" in class_ids
            assert "다" in class_ids

    def test_korean_class_id_in_summary(self, tmp_path):
        """Summary with Korean class_id should work."""
        records = [
            _make_record(
                student_id=f"S{i:03d}",
                week=w,
                ensemble_score=random.uniform(0.3, 0.8),
                class_id="가",
                topic="개념이해",
            )
            for i in range(5)
            for w in range(1, 4)
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(store, [1, 2, 3], "가반")
        assert summary.total_students == 5


# ===================================================================
# Persona 11: OCR 실패 다수
# 80% of students have empty responses (text: "")
# ===================================================================


class TestPersona11HighOCRFailure:
    """Class where 80% of students have empty/zero scores."""

    def test_mostly_zero_scores(self, tmp_path):
        """80% zero scores should not crash summary."""
        records = []
        for i in range(50):
            score = 0.0 if i < 40 else random.uniform(0.5, 0.9)
            for w in range(1, 4):
                records.append(
                    _make_record(
                        student_id=f"S{i + 1:03d}",
                        week=w,
                        ensemble_score=score,
                        topic="개념이해",
                        class_id="A",
                    )
                )
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(store, [1, 2, 3], "OCR실패")
        assert summary.total_students == 50
        # Most students should be persistent risk
        assert len(summary.persistent_risk_students) >= 30

    def test_all_zero_scores(self, tmp_path):
        """All students scoring 0.0 should not crash."""
        records = [
            _make_record(
                student_id=f"S{i + 1:03d}",
                week=w,
                ensemble_score=0.0,
                topic="개념이해",
                class_id="A",
            )
            for i in range(10)
            for w in range(1, 4)
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(store, [1, 2, 3], "전원0점")
        assert summary.total_students == 10
        # All should be persistent risk
        assert len(summary.persistent_risk_students) == 10
        # Class average should be 0.0
        for week, avg in summary.class_weekly_averages.items():
            assert avg == 0.0


# ===================================================================
# Persona 12: Single-topic exam
# All questions have topic: "개념이해", no "적용"
# ===================================================================


class TestPersona12SingleTopic:
    """All questions share the same topic."""

    def test_single_topic_summary(self, tmp_path):
        """Single topic should not crash topic-based grouping."""
        records = [
            _make_record(
                student_id=f"S{i + 1:03d}",
                week=w,
                question_sn=q,
                ensemble_score=random.uniform(0.3, 0.8),
                topic="개념이해",  # same topic for all
                class_id="A",
            )
            for i in range(5)
            for w in range(1, 4)
            for q in [1, 2, 3]
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(store, [1, 2, 3], "단일토픽")
        assert summary.total_students == 5

    def test_single_topic_trend_still_computed(self, tmp_path):
        """Trend should still work with a single topic grouping."""
        # Monotonically increasing scores
        records = [
            _make_record(
                student_id="S001",
                week=w,
                ensemble_score=0.3 + w * 0.1,
                topic="개념이해",
                class_id="A",
            )
            for w in range(1, 5)
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(store, [1, 2, 3, 4], "추세")
        # Student should have positive trend
        assert len(summary.student_trajectories) == 1
        assert summary.student_trajectories[0].overall_trend > 0


# ===================================================================
# Additional edge case attacks
# ===================================================================


class TestEdgeCaseAttacks:
    """Extra edge cases that cross persona boundaries."""

    def test_empty_store(self, tmp_path):
        """Empty store should not crash summary builder."""
        path = str(tmp_path / "empty.yaml")
        store = LongitudinalStore(path)
        store.save()
        store.load()
        assert len(store.get_all_records()) == 0

    def test_store_with_only_whitespace_yaml(self, tmp_path):
        """YAML file with only whitespace should load as empty."""
        path = str(tmp_path / "whitespace.yaml")
        with open(path, "w") as f:
            f.write("   \n\n  \n")
        store = LongitudinalStore(path)
        store.load()
        assert len(store.get_all_records()) == 0

    def test_negative_week_number(self, tmp_path):
        """Negative week number should not crash."""
        rec = _make_record(week=-1)
        store = _build_store_with_records(tmp_path, [rec])
        store.load()
        recs = store.get_all_records()
        assert len(recs) == 1

    def test_zero_week_number(self, tmp_path):
        """Week 0 should not crash."""
        rec = _make_record(week=0)
        store = _build_store_with_records(tmp_path, [rec])
        store.load()
        recs = store.get_all_records()
        assert len(recs) == 1

    def test_float_ensemble_score_boundaries(self, tmp_path):
        """Scores at exact boundaries (0.0, 0.45, 0.70, 1.0)."""
        records = [
            _make_record(
                student_id=f"S{i}",
                week=1,
                ensemble_score=score,
            )
            for i, score in enumerate([0.0, 0.45, 0.70, 1.0])
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        assert len(store.get_all_records()) == 4

    def test_duplicate_student_id_different_class(self, tmp_path):
        """Same student_id in different classes (edge case)."""
        records = [
            _make_record(
                student_id="S001",
                week=1,
                question_sn=1,
                class_id="A",
                topic="개념이해",
            ),
            _make_record(
                student_id="S001",
                week=1,
                question_sn=2,
                class_id="B",
                topic="적용",
            ),
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        recs = store.get_all_records()
        assert len(recs) == 2

    def test_very_large_ensemble_score(self, tmp_path):
        """Score > 1.0 (should not happen but test it)."""
        rec = _make_record(ensemble_score=999.99)
        store = _build_store_with_records(tmp_path, [rec])
        store.load()
        recs = store.get_all_records()
        assert recs[0].scores["ensemble_score"] == 999.99

    def test_nan_ensemble_score(self, tmp_path):
        """NaN score is rejected by store validation."""
        rec = _make_record(ensemble_score=float("nan"))
        with pytest.raises(ValueError, match="NaN"):
            _build_store_with_records(tmp_path, [rec])

    def test_missing_weeks_in_trajectory(self, tmp_path):
        """Student absent in week 2 but present in 1 and 3."""
        records = [
            _make_record(
                student_id="S001",
                week=1,
                ensemble_score=0.4,
                topic="개념이해",
            ),
            # week 2 missing for S001
            _make_record(
                student_id="S001",
                week=3,
                ensemble_score=0.6,
                topic="개념이해",
            ),
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        traj = store.get_student_trajectory(
            "S001",
            "ensemble_score",
        )
        assert len(traj) == 2
        assert traj[0] == (1, 0.4)
        assert traj[1] == (3, 0.6)


# ===================================================================
# parse_weeks_arg attack scenarios
# ===================================================================


class TestParseWeeksArgAttacks:
    """Attack the --weeks argument parser."""

    def test_single_int_zero(self):
        """--weeks 0 should produce empty list or raise."""
        from forma.cli_report_student import parse_weeks_arg

        result = parse_weeks_arg(["0"])
        # range(1, 1) = empty
        assert result == []

    def test_single_int_negative(self):
        """--weeks -1 should produce empty or raise."""
        from forma.cli_report_student import parse_weeks_arg

        result = parse_weeks_arg(["-1"])
        # range(1, 0) = empty
        assert result == []

    def test_range_reversed(self):
        """--weeks 5:2 (reversed range) should raise ValueError."""
        from forma.cli_report_student import parse_weeks_arg

        with pytest.raises(ValueError, match="Start must be"):
            parse_weeks_arg(["5:2"])

    def test_range_same_start_end(self):
        """--weeks 3:3 should produce [3]."""
        from forma.cli_report_student import parse_weeks_arg

        result = parse_weeks_arg(["3:3"])
        assert result == [3]

    def test_colon_only(self):
        """--weeks ':' should raise ValueError."""
        from forma.cli_report_student import parse_weeks_arg

        with pytest.raises(ValueError):
            parse_weeks_arg([":"])

    def test_double_colon(self):
        """--weeks '1:2:3' should raise ValueError."""
        from forma.cli_report_student import parse_weeks_arg

        with pytest.raises(ValueError, match="Invalid range"):
            parse_weeks_arg(["1:2:3"])

    def test_float_week(self):
        """--weeks 2.5 should raise ValueError."""
        from forma.cli_report_student import parse_weeks_arg

        with pytest.raises(ValueError):
            parse_weeks_arg(["2.5"])

    def test_non_numeric(self):
        """--weeks abc should raise ValueError."""
        from forma.cli_report_student import parse_weeks_arg

        with pytest.raises(ValueError):
            parse_weeks_arg(["abc"])

    def test_very_large_range(self):
        """--weeks 1:1000 should produce 1000 weeks."""
        from forma.cli_report_student import parse_weeks_arg

        result = parse_weeks_arg(["1:1000"])
        assert len(result) == 1000
        assert result[0] == 1
        assert result[-1] == 1000

    def test_list_with_duplicates(self):
        """--weeks 1 1 2 2 should deduplicate or keep all."""
        from forma.cli_report_student import parse_weeks_arg

        result = parse_weeks_arg(["1", "1", "2", "2"])
        # sorted list, may have dupes
        assert 1 in result
        assert 2 in result

    def test_empty_string(self):
        """--weeks '' should raise ValueError."""
        from forma.cli_report_student import parse_weeks_arg

        with pytest.raises(ValueError):
            parse_weeks_arg([""])


# ===================================================================
# parse_heatmap_layout attack scenarios
# ===================================================================


class TestParseHeatmapLayoutAttacks:
    """Attack the --heatmap-layout parser."""

    def test_valid_layout(self):
        """1:4 should parse correctly."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )

        assert parse_heatmap_layout("1:4") == (1, 4)

    def test_layout_with_x(self):
        """1x4 should raise (must use colon)."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )

        with pytest.raises(ValueError, match="rows:cols"):
            parse_heatmap_layout("1x4")

    def test_layout_with_asterisk(self):
        """2*3 should raise."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )

        with pytest.raises(ValueError, match="rows:cols"):
            parse_heatmap_layout("2*3")

    def test_layout_zero_rows(self):
        """0:4 should raise (non-positive)."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )

        with pytest.raises(ValueError, match="positive"):
            parse_heatmap_layout("0:4")

    def test_layout_zero_cols(self):
        """1:0 should raise (non-positive)."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )

        with pytest.raises(ValueError, match="positive"):
            parse_heatmap_layout("1:0")

    def test_layout_negative(self):
        """-1:4 should raise."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )

        with pytest.raises(ValueError, match="positive"):
            parse_heatmap_layout("-1:4")

    def test_layout_float(self):
        """1.5:2 should raise."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )

        with pytest.raises(ValueError, match="non-integer"):
            parse_heatmap_layout("1.5:2")

    def test_layout_text(self):
        """abc:def should raise."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )

        with pytest.raises(ValueError, match="non-integer"):
            parse_heatmap_layout("abc:def")

    def test_layout_multiple_colons(self):
        """1:2:3 should raise."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )

        with pytest.raises(ValueError, match="one ':'"):
            parse_heatmap_layout("1:2:3")

    def test_layout_empty(self):
        """'' should raise."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )

        with pytest.raises(ValueError):
            parse_heatmap_layout("")

    def test_layout_very_large(self):
        """100:100 should parse (no artificial limit)."""
        from forma.cli_report_longitudinal import (
            parse_heatmap_layout,
        )

        assert parse_heatmap_layout("100:100") == (100, 100)


# ===================================================================
# Topic statistics attack scenarios
# ===================================================================


class TestTopicStatisticsAttacks:
    """Attack topic statistics computation."""

    def test_topic_stats_no_topic_data(self, tmp_path):
        """No topic data should return empty list."""
        records = [
            _make_record(
                student_id="S001",
                week=w,
                topic=None,
            )
            for w in range(1, 4)
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            compute_topic_class_statistics,
        )

        stats = compute_topic_class_statistics(store, [1, 2, 3])
        assert stats == []

    def test_topic_stats_single_student(self, tmp_path):
        """Single student should have std=0."""
        records = [
            _make_record(
                student_id="S001",
                week=w,
                ensemble_score=0.5 + w * 0.1,
                topic="개념이해",
                class_id="A",
            )
            for w in range(1, 4)
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            compute_topic_class_statistics,
        )

        stats = compute_topic_class_statistics(store, [1, 2, 3])
        assert len(stats) > 0
        for s in stats:
            # Single student: std should be 0
            assert s.std == 0.0

    def test_topic_trends_fewer_than_3_weeks(self, tmp_path):
        """Trend with < 3 weeks should return empty."""
        records = [
            _make_record(
                student_id="S001",
                week=w,
                topic="개념이해",
                class_id="A",
            )
            for w in [1, 2]
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            compute_topic_trends,
        )

        trends = compute_topic_trends(store, [1, 2])
        assert trends == []

    def test_topic_trends_exactly_3_weeks(self, tmp_path):
        """Trend with exactly 3 weeks should compute."""
        records = [
            _make_record(
                student_id=f"S{i:03d}",
                week=w,
                ensemble_score=0.3 + w * 0.1,
                topic="개념이해",
                class_id="A",
            )
            for i in range(3)
            for w in [1, 2, 3]
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            compute_topic_trends,
        )

        trends = compute_topic_trends(store, [1, 2, 3])
        assert len(trends) == 1
        assert trends[0].topic == "개념이해"
        assert trends[0].n_weeks == 3
        # Increasing scores -> positive trend
        assert trends[0].kendall_tau > 0
        assert trends[0].spearman_rho > 0

    def test_topic_trends_all_same_score(self, tmp_path):
        """All identical scores should yield tau=0."""
        records = [
            _make_record(
                student_id="S001",
                week=w,
                ensemble_score=0.5,
                topic="개념이해",
                class_id="A",
            )
            for w in [1, 2, 3, 4]
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            compute_topic_trends,
        )

        trends = compute_topic_trends(store, [1, 2, 3, 4])
        if trends:
            # All same scores = no correlation; scipy returns NaN → converted to None
            assert trends[0].kendall_tau is None or trends[0].kendall_tau == 0.0


# ===================================================================
# Class filtering attack scenarios
# ===================================================================


class TestClassFilteringAttacks:
    """Attack class_id filtering in build_longitudinal_summary."""

    def test_filter_nonexistent_class(self, tmp_path):
        """Filter by class_id='Z' when no such class exists."""
        records = [
            _make_record(
                student_id="S001",
                week=1,
                class_id="A",
                topic="개념이해",
            ),
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(
            store,
            [1],
            "Z반",
            class_ids=["Z"],
        )
        assert summary.total_students == 0

    def test_filter_multiple_classes(self, tmp_path):
        """Filter by class_ids=['A', 'B'] should include both."""
        records = [
            _make_record(
                student_id=f"S{cls}{i}",
                week=1,
                class_id=cls,
                topic="개념이해",
            )
            for cls in ["A", "B", "C"]
            for i in range(3)
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(
            store,
            [1],
            "AB반",
            class_ids=["A", "B"],
        )
        assert summary.total_students == 6

    def test_filter_class_none_records(self, tmp_path):
        """Records with class_id=None should be excluded by filter."""
        records = [
            _make_record(
                student_id="S001",
                week=1,
                class_id=None,
                topic="개념이해",
            ),
            _make_record(
                student_id="S002",
                week=1,
                class_id="A",
                topic="개념이해",
            ),
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(
            store,
            [1],
            "A반",
            class_ids=["A"],
        )
        assert summary.total_students == 1

    def test_no_filter_includes_all(self, tmp_path):
        """No class_ids filter should include all students."""
        records = [
            _make_record(
                student_id=f"S{i:03d}",
                week=1,
                class_id=cls,
            )
            for i, cls in enumerate(["A", "B", "C", None])
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(
            store,
            [1],
            "전체",
        )
        assert summary.total_students == 4


# ===================================================================
# Topic weekly matrix attack
# ===================================================================


class TestTopicWeeklyMatrixAttacks:
    """Attack the get_topic_weekly_matrix method."""

    def test_no_topic_records_empty_matrix(self, tmp_path):
        """All records with topic=None → empty matrix."""
        records = [
            _make_record(
                student_id="S001",
                week=1,
                topic=None,
            ),
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        matrix = store.get_topic_weekly_matrix("ensemble_score")
        assert matrix == {}

    def test_mixed_topic_none_and_value(self, tmp_path):
        """Only records with topic != None appear in matrix."""
        records = [
            _make_record(
                student_id="S001",
                week=1,
                question_sn=1,
                topic=None,
            ),
            _make_record(
                student_id="S001",
                week=1,
                question_sn=2,
                topic="개념이해",
                ensemble_score=0.7,
            ),
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        matrix = store.get_topic_weekly_matrix("ensemble_score")
        assert "S001" in matrix
        assert "개념이해" in matrix["S001"]
        assert matrix["S001"]["개념이해"][1] == 0.7

    def test_same_topic_multiple_questions_averaged(self, tmp_path):
        """Two questions same topic same week → averaged."""
        records = [
            _make_record(
                student_id="S001",
                week=1,
                question_sn=1,
                ensemble_score=0.4,
                topic="적용",
            ),
            _make_record(
                student_id="S001",
                week=1,
                question_sn=2,
                ensemble_score=0.8,
                topic="적용",
            ),
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        matrix = store.get_topic_weekly_matrix("ensemble_score")
        avg = matrix["S001"]["적용"][1]
        assert abs(avg - 0.6) < 1e-9  # (0.4 + 0.8) / 2


# ===================================================================
# Risk threshold attack scenarios
# ===================================================================


class TestRiskThresholdAttacks:
    """Attack the risk threshold boundary."""

    def test_score_exactly_at_threshold(self, tmp_path):
        """Score = 0.45 (exactly threshold) is NOT at_risk."""
        records = [
            _make_record(
                student_id="S001",
                week=w,
                ensemble_score=0.45,
                topic="개념이해",
            )
            for w in [1, 2, 3]
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(store, [1, 2, 3], "경계")
        # 0.45 is NOT < 0.45, so not at risk
        assert "S001" not in summary.persistent_risk_students

    def test_score_just_below_threshold(self, tmp_path):
        """Score = 0.44 is at_risk (FormaDumper rounds to 2dp)."""
        records = [
            _make_record(
                student_id="S001",
                week=w,
                ensemble_score=0.44,
                topic="개념이해",
            )
            for w in [1, 2, 3]
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(store, [1, 2, 3], "경계")
        assert "S001" in summary.persistent_risk_students

    def test_yaml_rounding_affects_risk(self, tmp_path):
        """FormaDumper rounds 0.4499 to 0.45 -- NOT at_risk after save.

        This is a known precision limitation: scores are rounded
        to 2 decimal places during YAML serialization. A score of
        0.4499 (at_risk in memory) becomes 0.45 (NOT at_risk)
        after save/load. Document this as DEGRADED, not a bug.
        """
        records = [
            _make_record(
                student_id="S001",
                week=w,
                ensemble_score=0.4499,
                topic="개념이해",
            )
            for w in [1, 2, 3]
        ]
        store = _build_store_with_records(tmp_path, records)
        store.load()
        recs = store.get_all_records()
        # After roundtrip, 0.4499 becomes 0.45
        assert recs[0].scores["ensemble_score"] == 0.45
        from forma.longitudinal_report_data import (
            build_longitudinal_summary,
        )

        summary = build_longitudinal_summary(
            store,
            [1, 2, 3],
            "경계",
        )
        # NOT at_risk because 0.45 is not < 0.45
        assert "S001" not in summary.persistent_risk_students


# ===================================================================
# CLI integration attacks (new args)
# ===================================================================


class TestCLINewArgsAttacks:
    """Attack new CLI arguments via subprocess."""

    def test_heatmap_layout_invalid_format(self):
        """--heatmap-layout with invalid format should fail."""
        result = _run_cli(
            [
                "report",
                "longitudinal",
                "--store",
                "/tmp/fake.yaml",
                "--class-name",
                "A",
                "--output",
                "/tmp/out.pdf",
                "--heatmap-layout",
                "abc",
            ]
        )
        assert result.returncode != 0

    def test_mastery_top_n_zero(self):
        """--mastery-top-n 0 should be accepted or error gracefully."""
        result = _run_cli(
            [
                "report",
                "longitudinal",
                "--store",
                "/tmp/fake.yaml",
                "--class-name",
                "A",
                "--output",
                "/tmp/out.pdf",
                "--mastery-top-n",
                "0",
            ]
        )
        # Should fail because store doesn't exist, not crash
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined

    def test_mastery_top_n_negative(self):
        """--mastery-top-n -5 should not crash."""
        result = _run_cli(
            [
                "report",
                "longitudinal",
                "--store",
                "/tmp/fake.yaml",
                "--class-name",
                "A",
                "--output",
                "/tmp/out.pdf",
                "--mastery-top-n",
                "-5",
            ]
        )
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined

    def test_risk_threshold_zero(self):
        """--risk-threshold 0.0 should not crash."""
        result = _run_cli(
            [
                "report",
                "longitudinal",
                "--store",
                "/tmp/fake.yaml",
                "--class-name",
                "A",
                "--output",
                "/tmp/out.pdf",
                "--risk-threshold",
                "0.0",
            ]
        )
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined

    def test_risk_threshold_one(self):
        """--risk-threshold 1.0 means everyone is at risk."""
        result = _run_cli(
            [
                "report",
                "longitudinal",
                "--store",
                "/tmp/fake.yaml",
                "--class-name",
                "A",
                "--output",
                "/tmp/out.pdf",
                "--risk-threshold",
                "1.0",
            ]
        )
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined

    def test_classes_with_heatmap_no_layout(self):
        """--classes A B without --heatmap-layout should use default."""
        result = _run_cli(
            [
                "report",
                "longitudinal",
                "--store",
                "/tmp/fake.yaml",
                "--class-name",
                "A",
                "--output",
                "/tmp/out.pdf",
                "--classes",
                "A",
                "B",
            ]
        )
        combined = result.stdout + result.stderr
        # Should fail because store doesn't exist, not crash
        assert "Traceback" not in combined
