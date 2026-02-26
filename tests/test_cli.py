"""Tests for src.cli — bhu-exam CLI entrypoint."""
import json
import os
import tempfile

import pytest
import yaml

from src.cli import _load_config, _load_questions, _parse_args, main


class TestParseArgs:
    """Argument parsing tests."""

    def test_parse_args_minimal(self):
        """필수 인자만으로 파싱."""
        args = _parse_args([
            "--questions", "q.yaml",
            "--num-papers", "10",
            "--output", "exam.pdf",
        ])
        assert args.questions == "q.yaml"
        assert args.num_papers == 10
        assert args.output == "exam.pdf"
        # 기본값 확인
        assert args.year == 2025
        assert args.grade == 1
        assert args.semester == 2
        assert args.course == "감염미생물학"
        assert args.week == 3
        assert args.form_url is None
        assert args.student_ids is None
        assert args.font_path is None

    def test_parse_args_full(self):
        """전체 인자 파싱."""
        args = _parse_args([
            "--questions-json", '[{"topic":"T","text":"Q","limit":"50"}]',
            "--num-papers", "200",
            "--output", "/tmp/exam.pdf",
            "--year", "2026",
            "--grade", "2",
            "--semester", "1",
            "--course", "해부학",
            "--week", "5",
            "--form-url", "https://example.com/{student_id}",
            "--student-ids", "A001", "A002",
            "--font-path", "/usr/share/fonts/test.ttf",
        ])
        assert args.questions_json == '[{"topic":"T","text":"Q","limit":"50"}]'
        assert args.num_papers == 200
        assert args.output == "/tmp/exam.pdf"
        assert args.year == 2026
        assert args.grade == 2
        assert args.semester == 1
        assert args.course == "해부학"
        assert args.week == 5
        assert args.form_url == "https://example.com/{student_id}"
        assert args.student_ids == ["A001", "A002"]
        assert args.font_path == "/usr/share/fonts/test.ttf"


class TestLoadQuestions:
    """Question loading tests."""

    def test_load_questions_from_yaml(self, tmp_path):
        """YAML 파일에서 문제 로드."""
        questions = [
            {"topic": "개념이해", "text": "미생물의 정의", "limit": "100자 내외"},
            {"topic": "적용", "text": "감염 경로 3가지", "limit": "150자 내외"},
        ]
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(
            yaml.dump(questions, allow_unicode=True),
            encoding="utf-8",
        )
        result = _load_questions(str(yaml_file))
        assert result == questions
        assert len(result) == 2
        assert result[0]["topic"] == "개념이해"

    def test_load_questions_from_json(self):
        """JSON 문자열에서 문제 파싱."""
        json_str = json.dumps([
            {"topic": "T1", "text": "Q1", "limit": "50자"},
        ], ensure_ascii=False)
        result = _load_questions(json_str)
        assert len(result) == 1
        assert result[0]["topic"] == "T1"


class TestLoadConfig:
    """_load_config tests for unified and legacy YAML formats."""

    def test_load_config_unified_yaml(self, tmp_path):
        """통합 YAML에서 메타데이터 + 문제 추출."""
        unified = {
            "year": 2026,
            "grade": 1,
            "semester": 1,
            "course": "감염미생물학",
            "week": 5,
            "num-papers": 200,
            "form-url": "https://example.com/{student_id}",
            "questions": [
                {"topic": "개념이해", "text": "Q1", "limit": "200자 내외"},
                {"topic": "적용", "text": "Q2", "limit": "200자 내외"},
            ],
        }
        yaml_file = tmp_path / "exam.yaml"
        yaml_file.write_text(
            yaml.dump(unified, allow_unicode=True), encoding="utf-8",
        )
        result = _load_config(str(yaml_file))
        assert isinstance(result, dict)
        assert result["year"] == 2026
        assert result["course"] == "감염미생물학"
        assert result["num-papers"] == 200
        assert result["form-url"] == "https://example.com/{student_id}"
        assert len(result["questions"]) == 2
        assert result["questions"][0]["topic"] == "개념이해"

    def test_load_config_legacy_yaml(self, tmp_path):
        """기존 리스트 형식 하위 호환."""
        legacy = [
            {"topic": "개념이해", "text": "Q1", "limit": "100자 내외"},
        ]
        yaml_file = tmp_path / "q.yaml"
        yaml_file.write_text(
            yaml.dump(legacy, allow_unicode=True), encoding="utf-8",
        )
        result = _load_config(str(yaml_file))
        assert isinstance(result, dict)
        assert "questions" in result
        assert len(result["questions"]) == 1
        assert result["questions"][0]["text"] == "Q1"
        # 메타데이터 키는 없어야 함
        assert "year" not in result
        assert "num-papers" not in result


class TestMainIntegration:
    """Integration test: CLI → PDF generation."""

    def test_main_generates_pdf(self, tmp_path):
        """CLI main → PDF 파일 생성 확인."""
        questions = [
            {"topic": "개념이해", "text": "테스트 문제입니다.", "limit": "100자 내외"},
        ]
        yaml_file = tmp_path / "q.yaml"
        yaml_file.write_text(
            yaml.dump(questions, allow_unicode=True),
            encoding="utf-8",
        )
        output_pdf = tmp_path / "exam.pdf"

        main([
            "--questions", str(yaml_file),
            "--num-papers", "2",
            "--output", str(output_pdf),
        ])

        assert output_pdf.exists()
        assert output_pdf.stat().st_size > 0

    def test_main_config_generates_pdf(self, tmp_path):
        """--config로 통합 YAML → PDF 생성."""
        unified = {
            "year": 2026,
            "grade": 1,
            "semester": 1,
            "course": "감염미생물학",
            "week": 5,
            "num-papers": 2,
            "questions": [
                {"topic": "개념이해", "text": "테스트 문제입니다.", "limit": "100자 내외"},
            ],
        }
        yaml_file = tmp_path / "exam.yaml"
        yaml_file.write_text(
            yaml.dump(unified, allow_unicode=True), encoding="utf-8",
        )
        output_pdf = tmp_path / "exam.pdf"

        main([
            "--config", str(yaml_file),
            "--output", str(output_pdf),
        ])

        assert output_pdf.exists()
        assert output_pdf.stat().st_size > 0
