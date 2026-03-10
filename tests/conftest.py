"""pytest 전역 설정 — JAVA_HOME 자동 탐지 (NixOS 지원) + v0.10 shared fixtures."""
import os
import subprocess

import pytest


def _find_jvm_home() -> str | None:
    """JAVA_HOME을 찾는다. 이미 설정돼 있으면 그대로 반환."""
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]

    # nix develop 셸에서 JAVA_HOME 읽기 (NixOS 환경)
    try:
        result = subprocess.run(
            ["nix", "develop", "--command", "bash", "-c", "echo $JAVA_HOME"],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        java_home = result.stdout.strip()
        if java_home and os.path.isdir(java_home):
            return java_home
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


_jvm_home = _find_jvm_home()
if _jvm_home:
    os.environ["JAVA_HOME"] = _jvm_home


# ---------------------------------------------------------------------------
# v0.10 shared fixtures (008-intervention-path-prediction)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_intervention_records() -> list[dict]:
    """Sample InterventionRecord data for testing."""
    return [
        {
            "student_id": "s001",
            "week": 2,
            "intervention_type": "면담",
            "description": "학습 동기 상담",
        },
        {
            "student_id": "s001",
            "week": 3,
            "intervention_type": "보충학습",
            "description": "개념 보충 자료 제공",
        },
        {
            "student_id": "s002",
            "week": 2,
            "intervention_type": "과제부여",
            "description": "추가 과제 부여",
        },
        {
            "student_id": "s003",
            "week": 4,
            "intervention_type": "멘토링",
            "description": "선배 멘토 배정",
        },
    ]


@pytest.fixture()
def mock_longitudinal_store():
    """Mock LongitudinalStore with ensemble_scores for 4+ weeks."""
    from unittest.mock import MagicMock

    from forma.evaluation_types import LongitudinalRecord

    store = MagicMock()

    # 3 students, 4 weeks of data
    records = []
    students = ["s001", "s002", "s003"]
    weekly_scores = {
        "s001": [0.75, 0.70, 0.65, 0.60],  # declining
        "s002": [0.40, 0.45, 0.50, 0.55],  # improving
        "s003": [0.30, 0.25, 0.20, 0.15],  # at risk
    }
    for sid in students:
        for week_idx, score in enumerate(weekly_scores[sid], start=1):
            records.append(
                LongitudinalRecord(
                    student_id=sid,
                    week=week_idx,
                    question_sn=1,
                    scores={"ensemble_score": score, "concept_coverage": score * 0.9},
                    tier_level=3 if score >= 0.65 else (2 if score >= 0.45 else 1),
                    tier_label="Advanced" if score >= 0.65 else (
                        "Proficient" if score >= 0.45 else "Developing"
                    ),
                )
            )

    store.get_all_records.return_value = records
    store.get_student_history.side_effect = lambda sid: [
        r for r in records if r.student_id == sid
    ]
    return store


@pytest.fixture()
def sample_concept_dependencies() -> list[dict]:
    """List of concept dependency dicts for DAG testing.

    Graph structure:
        세포막 구조 → 물질 이동 → 삼투압 → 체액 균형
                                 ↗
                       확산
    """
    return [
        {"prerequisite": "세포막 구조", "dependent": "물질 이동"},
        {"prerequisite": "물질 이동", "dependent": "삼투압"},
        {"prerequisite": "삼투압", "dependent": "체액 균형"},
        {"prerequisite": "확산", "dependent": "삼투압"},
    ]


@pytest.fixture()
def sample_exam_yaml_with_deps(sample_concept_dependencies) -> dict:
    """Exam YAML dict containing concept_dependencies + knowledge_graph."""
    return {
        "exam_name": "Ch01_서론_FormativeTest",
        "questions": [
            {
                "question_sn": 1,
                "concepts": ["세포막 구조", "물질 이동", "삼투압"],
                "knowledge_graph": {
                    "edges": [
                        {"subject": "세포막 구조", "relation": "구성요소", "object": "물질 이동"},
                    ],
                },
            },
        ],
        "concept_dependencies": sample_concept_dependencies,
    }


@pytest.fixture()
def cyclic_concept_dependencies() -> list[dict]:
    """Dependencies with a cycle: A -> B -> C -> A."""
    return [
        {"prerequisite": "A", "dependent": "B"},
        {"prerequisite": "B", "dependent": "C"},
        {"prerequisite": "C", "dependent": "A"},
    ]


@pytest.fixture()
def sample_knowledge_graph_concepts() -> set[str]:
    """Set of concept names from the sample exam knowledge_graph."""
    return {"세포막 구조", "물질 이동", "삼투압", "체액 균형", "확산"}


@pytest.fixture()
def sample_grade_mapping() -> dict:
    """Sample grade mapping data for grade_predictor testing."""
    return {
        "2024-1학기": {
            "s001": "A",
            "s002": "B",
            "s003": "D",
            "s004": "F",
        },
        "2024-2학기": {
            "s001": "B",
            "s002": "C",
            "s003": "A",
        },
    }
