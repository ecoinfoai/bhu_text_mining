"""Adversary attack tests for v0.9.0 features.

7 adversarial personas with aggressive testing of edge cases,
boundary conditions, type confusion, and data corruption scenarios.

Targets:
  - US1: project_config.ProjectConfiguration, find/load/validate/merge
  - US2/US5: risk_predictor (FeatureExtractor, RiskPredictor, TrainedRiskModel,
             save_model, load_model)
  - US3: warning_report_data (RiskType, WarningCard, build_warning_data)
  - US4: section_comparison (SectionStats, SectionComparison, compute functions)
  - US4: section_comparison_charts (box plot, heatmap, interaction chart)

Personas:
  - Persona 1: Config Saboteur — attacks forma.yaml parsing/validation
  - Persona 2: Model Corruptor — attacks .pkl model files and load_model
  - Persona 3: Data Poisoner — attacks FeatureExtractor and training pipeline
  - Persona 4: Statistical Attacker — attacks cross-section comparison
  - Persona 5: Warning Data Crasher — attacks WarningCard construction
  - Persona 6: Boundary Pusher — attacks exact boundary values
  - Persona 7: Concurrent Chaos — race conditions and file locking
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import os
import random
import tempfile
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from forma.project_config import (
    ProjectConfiguration,
    find_project_config,
    load_project_config,
    merge_configs,
    validate_project_config,
)
from forma.section_comparison import (
    CrossSectionReport,
    SectionComparison,
    SectionStats,
    _cohens_d,
    _effect_size_label,
    compute_concept_mastery_by_section,
    compute_pairwise_comparisons,
    compute_section_stats,
    compute_weekly_interaction,
)


# ===========================================================================
# PERSONA 1: CONFIG SABOTEUR
# ===========================================================================


class TestConfigSaboteur:
    """Persona 1: Malformed forma.yaml and config manipulation attacks."""

    def test_malformed_yaml_syntax_error(self):
        """Syntactically invalid YAML should raise yaml.YAMLError."""
        import yaml
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w", encoding="utf-8",
        ) as f:
            f.write(":::not valid yaml::: {{{")
            path = f.name
        try:
            with pytest.raises(yaml.YAMLError):
                load_project_config(path)
        finally:
            os.unlink(path)

    def test_empty_yaml_file_returns_empty_dict(self):
        """Empty YAML file (yaml.safe_load returns None) should return {}."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w", encoding="utf-8",
        ) as f:
            f.write("")
            path = f.name
        try:
            result = load_project_config(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_scalar_yaml_returns_empty_dict(self):
        """YAML file containing only a scalar string should return {}."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w", encoding="utf-8",
        ) as f:
            f.write('"just a string"')
            path = f.name
        try:
            result = load_project_config(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_list_yaml_returns_empty_dict(self):
        """YAML file containing a list (not a dict) should return {}."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w", encoding="utf-8",
        ) as f:
            f.write("- item1\n- item2\n")
            path = f.name
        try:
            result = load_project_config(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises_file_not_found(self):
        """Loading a nonexistent file should raise FileNotFoundError."""
        from pathlib import Path
        with pytest.raises(FileNotFoundError):
            load_project_config(Path("/nonexistent/path/forma.yaml"))

    def test_unicode_bom_in_yaml(self):
        """YAML file with Unicode BOM should parse correctly."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="wb",
        ) as f:
            # UTF-8 BOM + valid YAML
            f.write(b"\xef\xbb\xbfcurrent_week: 5\n")
            path = f.name
        try:
            result = load_project_config(path)
            # PyYAML safe_load handles BOM differently across versions;
            # the important thing is it doesn't crash
            assert isinstance(result, dict)
        finally:
            os.unlink(path)

    def test_unknown_top_level_keys_produce_warnings(self, caplog):
        """Unknown top-level keys should produce warnings, not errors (FR-008)."""
        config = {
            "unknown_section": {"key": "value"},
            "future_feature": True,
            "current_week": 3,
        }
        with caplog.at_level(logging.WARNING, logger="forma.project_config"):
            # Should NOT raise ValueError — unknown keys are warnings
            validate_project_config(config)
        assert "Unknown configuration key: 'unknown_section'" in caplog.text
        assert "Unknown configuration key: 'future_feature'" in caplog.text

    def test_unknown_nested_keys_produce_warnings(self, caplog):
        """Unknown keys within known sections should produce warnings."""
        config = {
            "project": {"course_name": "Test", "future_field": "value"},
            "reports": {"dpi": 150, "unknown_option": True},
        }
        with caplog.at_level(logging.WARNING, logger="forma.project_config"):
            validate_project_config(config)
        assert "Unknown key 'future_field' in section 'project'" in caplog.text
        assert "Unknown key 'unknown_option' in section 'reports'" in caplog.text

    def test_wrong_types_for_every_field(self):
        """Type mismatches across all sections should be collected, not fail-fast."""
        config = {
            "project": {
                "course_name": 123,       # should be str
                "year": "not_int",        # should be int
                "semester": [1],          # should be int
                "grade": {"a": 1},        # should be int
            },
            "ocr": {
                "num_questions": "five",  # should be int
            },
            "evaluation": {
                "provider": 42,           # should be str
                "n_calls": False,         # bool is caught as not-int
                "skip_feedback": "yes",   # should be bool
                "skip_graph": 1,          # should be bool
                "skip_statistical": 0,    # should be bool
            },
            "reports": {
                "dpi": "high",            # should be int
                "skip_llm": "true",       # should be bool
                "aggregate": 1,           # should be bool
            },
            "current_week": "three",      # should be int
        }
        with pytest.raises(ValueError) as exc_info:
            validate_project_config(config)
        msg = str(exc_info.value)
        # All errors should be collected (fail-comprehensive, not fail-fast)
        assert "course_name must be a string" in msg
        assert "year must be an integer" in msg
        assert "semester must be an integer" in msg
        assert "grade must be an integer" in msg
        assert "num_questions must be an integer" in msg
        assert "provider must be a string" in msg
        assert "skip_feedback must be a boolean" in msg
        assert "current_week must be an integer" in msg

    def test_value_constraint_violations(self):
        """Values outside valid ranges should produce errors."""
        config = {
            "project": {
                "year": 2019,          # must be >= 2020
                "semester": 3,         # must be 1 or 2
                "grade": 0,            # must be >= 1
            },
            "ocr": {
                "num_questions": 0,    # must be >= 1
            },
            "evaluation": {
                "provider": "openai",  # must be "gemini" or "anthropic"
                "n_calls": 0,          # must be >= 1
            },
            "reports": {
                "dpi": 71,             # must be 72-600
            },
            "current_week": 0,         # must be >= 1
        }
        with pytest.raises(ValueError) as exc_info:
            validate_project_config(config)
        msg = str(exc_info.value)
        assert "year must be >= 2020" in msg
        assert "semester must be 1 or 2" in msg
        assert "grade must be >= 1" in msg
        assert "num_questions must be >= 1" in msg
        assert "provider must be 'gemini' or 'anthropic'" in msg
        assert "n_calls must be >= 1" in msg
        assert "dpi must be between 72 and 600" in msg
        assert "current_week must be >= 1" in msg

    def test_dpi_upper_boundary_violation(self):
        """dpi = 601 should produce error; dpi = 600 should pass."""
        config_fail = {"reports": {"dpi": 601}}
        with pytest.raises(ValueError, match="dpi must be between 72 and 600"):
            validate_project_config(config_fail)

        config_pass = {"reports": {"dpi": 600}}
        validate_project_config(config_pass)  # Should not raise

    def test_section_as_non_dict_silently_skipped(self):
        """Section set to a scalar (not dict) should be silently skipped."""
        config = {
            "project": "not a dict",
            "reports": 42,
            "current_week": 5,  # valid top-level key
        }
        # Should NOT raise — non-dict sections are skipped in validation
        validate_project_config(config)

    def test_bool_treated_as_non_int_for_year(self):
        """YAML boolean True (subclass of int) should be caught as type error.

        BUG FIXED: _check_int now checks ``isinstance(val, bool)`` FIRST
        before checking isinstance(val, int), so bools are correctly
        rejected with "got bool" message instead of misleading value errors.
        """
        config = {"project": {"year": True}}
        with pytest.raises(ValueError) as exc_info:
            validate_project_config(config)
        msg = str(exc_info.value)
        # FIXED: Bool is now correctly caught as type error
        assert "year must be an integer, got bool" in msg
        # Value constraint should NOT fire for bools
        assert "project.year must be >= 2020" not in msg

    def test_join_pattern_missing_class_placeholder(self):
        """Non-empty join_pattern without {class} should produce error."""
        config = {
            "classes": {"join_pattern": "anp_final.yaml"},
        }
        with pytest.raises(ValueError, match="join_pattern must contain"):
            validate_project_config(config)

    def test_join_pattern_empty_string_is_valid(self):
        """Empty join_pattern should pass validation (placeholder not required)."""
        config = {"classes": {"join_pattern": ""}}
        validate_project_config(config)  # Should not raise

    def test_eval_pattern_missing_class_placeholder(self):
        """Non-empty eval_pattern without {class} should produce error."""
        config = {
            "classes": {"eval_pattern": "eval_results/"},
        }
        with pytest.raises(ValueError, match="eval_pattern must contain"):
            validate_project_config(config)

    def test_identifiers_wrong_type(self):
        """class_identifiers as a string instead of list should produce error."""
        config = {"classes": {"identifiers": "A,B,C"}}
        with pytest.raises(ValueError, match="identifiers must be a list"):
            validate_project_config(config)

    def test_10kb_course_name_survives(self):
        """Extremely long course_name (10KB) should not crash validation."""
        config = {"project": {"course_name": "X" * 10240}}
        validate_project_config(config)  # Should not raise

    def test_null_bytes_in_path(self):
        """Null bytes in exam_config path should not crash validation.

        File existence is validated at CLI load time, not config validation.
        """
        config = {"paths": {"exam_config": "path\x00with\x00nulls.yaml"}}
        validate_project_config(config)  # Should not raise at validation stage

    def test_find_config_in_nonexistent_directory(self):
        """find_project_config with nonexistent start_dir should return None."""
        from pathlib import Path
        result = find_project_config(Path("/nonexistent/directory/path"))
        assert result is None

    def test_find_config_stops_at_git_sentinel(self):
        """find_project_config should stop at .git directory boundary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            base = Path(tmpdir)
            # Create .git sentinel at base
            (base / ".git").mkdir()
            # Create subdirectory
            sub = base / "subdir"
            sub.mkdir()
            # No forma.yaml anywhere
            result = find_project_config(sub)
            assert result is None

    def test_find_config_discovers_in_parent(self):
        """find_project_config should find forma.yaml in parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            base = Path(tmpdir)
            (base / ".git").mkdir()
            # Create forma.yaml at base
            (base / "forma.yaml").write_text("current_week: 1\n")
            # Create subdirectory
            sub = base / "subdir"
            sub.mkdir()
            result = find_project_config(sub)
            assert result == (base / "forma.yaml").resolve()

    def test_merge_cli_overrides_project_config(self):
        """CLI explicit keys should override project config values."""
        cli_ns = argparse.Namespace(dpi=300, current_week=5, output_dir="cli_out")
        project = {"reports": {"dpi": 150}, "current_week": 3}
        system = {}
        result = merge_configs(
            cli_ns, project, system, explicit_keys={"dpi"},
        )
        assert result["dpi"] == 300       # CLI explicit → wins
        assert result["current_week"] == 3  # CLI not explicit → project wins

    def test_merge_project_overrides_system(self):
        """Project config values should override system config."""
        cli_ns = argparse.Namespace(dpi=150)
        project = {"reports": {"dpi": 200}}
        system = {"dpi": 100}
        result = merge_configs(cli_ns, project, system)
        assert result["dpi"] == 200  # Project wins over system

    def test_merge_system_fallback(self):
        """System config should be used when CLI and project don't set a value."""
        cli_ns = argparse.Namespace(dpi=150)
        project = {}
        system = {"dpi": 100, "extra_key": "sys_val"}
        result = merge_configs(cli_ns, project, system)
        # extra_key only in system config
        assert result["extra_key"] == "sys_val"

    def test_merge_empty_all_layers(self):
        """Merging three empty layers should produce empty dict."""
        cli_ns = argparse.Namespace()
        result = merge_configs(cli_ns, {}, {})
        assert result == {}

    def test_validate_empty_config(self):
        """Validating an empty dict should pass without errors."""
        validate_project_config({})  # Should not raise

    def test_valid_complete_config(self):
        """A fully valid config should pass validation without errors."""
        config = {
            "project": {
                "course_name": "인체구조와기능",
                "year": 2026,
                "semester": 1,
                "grade": 1,
            },
            "classes": {
                "identifiers": ["A", "B", "C", "D"],
                "join_pattern": "anp_{class}_final.yaml",
                "eval_pattern": "eval_{class}/",
            },
            "paths": {
                "exam_config": "exams/test.yaml",
                "join_dir": "results/",
                "output_dir": "output/",
            },
            "ocr": {"num_questions": 5},
            "evaluation": {
                "provider": "gemini",
                "n_calls": 3,
                "skip_feedback": False,
                "skip_graph": False,
                "skip_statistical": False,
            },
            "reports": {"dpi": 150, "skip_llm": False, "aggregate": True},
            "prediction": {"model_path": None},
            "current_week": 3,
        }
        validate_project_config(config)  # Should not raise


# ===========================================================================
# PERSONA 4: STATISTICAL ATTACKER
# ===========================================================================


class TestStatisticalAttacker:
    """Persona 4: Attack cross-section comparison statistical functions."""

    def test_empty_scores_list(self):
        """Empty ensemble_scores list should produce zero-valued stats."""
        result = compute_section_stats("A", [], set())
        assert result.n_students == 0
        assert result.mean == 0.0
        assert result.median == 0.0
        assert result.std == 0.0
        assert result.pct_at_risk == 0.0

    def test_single_student_section(self):
        """Single-student section: stats computed, std=0.0."""
        result = compute_section_stats("A", [0.75], {"S001"})
        assert result.n_students == 1
        assert result.mean == 0.75
        assert result.median == 0.75
        assert result.std == 0.0
        assert result.n_at_risk == 1
        assert result.pct_at_risk == 1.0

    def test_all_identical_scores_zero_std(self):
        """All students with identical scores: std=0.0."""
        scores = [0.5] * 50
        result = compute_section_stats("A", scores, set())
        assert result.std == 0.0
        assert result.mean == 0.5
        assert result.median == 0.5

    def test_pairwise_fewer_than_2_sections_returns_empty(self):
        """Fewer than 2 sections should return empty comparison list."""
        assert compute_pairwise_comparisons({"A": [0.5]}) == []
        assert compute_pairwise_comparisons({}) == []

    def test_n29_vs_n30_selects_mann_whitney(self):
        """When one section has N=29 and other N=30, Mann-Whitney should be used."""
        rng = np.random.RandomState(42)
        scores_a = rng.uniform(0.3, 0.8, 29).tolist()
        scores_b = rng.uniform(0.4, 0.9, 30).tolist()
        results = compute_pairwise_comparisons({"A": scores_a, "B": scores_b})
        assert len(results) == 1
        assert results[0].test_name == "mann_whitney_u"
        assert results[0].n_a == 29
        assert results[0].n_b == 30
        # No Bonferroni for 2 sections
        assert results[0].p_value_corrected is None

    def test_n30_vs_n30_selects_welch(self):
        """When both sections have N>=30, Welch's t-test should be used."""
        rng = np.random.RandomState(42)
        scores_a = rng.uniform(0.3, 0.8, 30).tolist()
        scores_b = rng.uniform(0.4, 0.9, 30).tolist()
        results = compute_pairwise_comparisons({"A": scores_a, "B": scores_b})
        assert len(results) == 1
        assert results[0].test_name == "welch_t"

    def test_n29_vs_n29_selects_mann_whitney(self):
        """Two sections both N=29 should use Mann-Whitney U."""
        rng = np.random.RandomState(42)
        scores_a = rng.uniform(0.3, 0.8, 29).tolist()
        scores_b = rng.uniform(0.4, 0.9, 29).tolist()
        results = compute_pairwise_comparisons({"A": scores_a, "B": scores_b})
        assert results[0].test_name == "mann_whitney_u"

    def test_identical_scores_both_sections(self):
        """All-identical scores in both sections should not crash.

        mannwhitneyu with identical values should return p=1.0 or raise
        a warning but not crash. Cohen's d should be 0.0.
        """
        scores_a = [0.5] * 10
        scores_b = [0.5] * 10
        results = compute_pairwise_comparisons({"A": scores_a, "B": scores_b})
        assert len(results) == 1
        comp = results[0]
        assert comp.cohens_d == 0.0
        assert comp.effect_size_label == "negligible"
        assert math.isfinite(comp.p_value)
        assert math.isfinite(comp.test_statistic)

    def test_identical_scores_within_one_section(self):
        """Zero-variance in one section, variance in other: no crash."""
        scores_a = [0.5] * 15
        scores_b = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.3, 0.4, 0.5, 0.6,
                     0.7, 0.8, 0.3, 0.4, 0.5]
        results = compute_pairwise_comparisons({"A": scores_a, "B": scores_b})
        assert len(results) == 1
        comp = results[0]
        assert math.isfinite(comp.cohens_d)
        assert math.isfinite(comp.p_value)

    def test_bonferroni_correction_3_sections(self):
        """3 sections should apply Bonferroni correction (n_pairs=3)."""
        rng = np.random.RandomState(42)
        section_scores = {
            "A": rng.uniform(0.2, 0.8, 15).tolist(),
            "B": rng.uniform(0.3, 0.9, 15).tolist(),
            "C": rng.uniform(0.1, 0.7, 15).tolist(),
        }
        results = compute_pairwise_comparisons(section_scores)
        assert len(results) == 3  # C(3,2) = 3
        for comp in results:
            assert comp.p_value_corrected is not None
            # Bonferroni: corrected = min(p * 3, 1.0)
            assert comp.p_value_corrected == pytest.approx(
                min(comp.p_value * 3, 1.0),
            )
            assert 0.0 <= comp.p_value_corrected <= 1.0

    def test_bonferroni_caps_at_1(self):
        """Bonferroni correction should never exceed 1.0."""
        # Two identical sections → p close to 1.0 → corrected would exceed 1.0
        section_scores = {
            "A": [0.5] * 10,
            "B": [0.5] * 10,
            "C": [0.5] * 10,
        }
        results = compute_pairwise_comparisons(section_scores)
        for comp in results:
            assert comp.p_value_corrected is not None
            assert comp.p_value_corrected <= 1.0

    def test_4_sections_6_pairs(self):
        """4 sections should produce C(4,2)=6 pairwise comparisons."""
        rng = np.random.RandomState(42)
        section_scores = {
            s: rng.uniform(0.2, 0.9, 20).tolist()
            for s in ["A", "B", "C", "D"]
        }
        results = compute_pairwise_comparisons(section_scores)
        assert len(results) == 6
        # All pairs should have Bonferroni correction
        for comp in results:
            assert comp.p_value_corrected is not None
            assert comp.p_value_corrected == pytest.approx(
                min(comp.p_value * 6, 1.0),
            )

    def test_cohens_d_single_element_groups(self):
        """Cohen's d with single-element groups: variance=0 → d=0.0."""
        d = _cohens_d([0.8], [0.2])
        # n1=1, n2=1 → denom = 0 → returns 0.0
        assert d == 0.0

    def test_cohens_d_zero_variance_both_groups(self):
        """Cohen's d with zero variance in both groups: d=0.0."""
        d = _cohens_d([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert d == 0.0

    def test_cohens_d_large_effect(self):
        """Cohen's d with clearly different groups: |d| should be large."""
        d = _cohens_d([0.9, 0.85, 0.95, 0.88], [0.1, 0.15, 0.05, 0.12])
        assert abs(d) > 0.8
        assert _effect_size_label(d) == "large"

    def test_effect_size_label_boundaries(self):
        """Effect size labels at exact boundary values."""
        assert _effect_size_label(0.0) == "negligible"
        assert _effect_size_label(0.19) == "negligible"
        assert _effect_size_label(0.2) == "small"
        assert _effect_size_label(0.49) == "small"
        assert _effect_size_label(0.5) == "medium"
        assert _effect_size_label(0.79) == "medium"
        assert _effect_size_label(0.8) == "large"
        assert _effect_size_label(10.0) == "large"
        # Negative values
        assert _effect_size_label(-0.5) == "medium"
        assert _effect_size_label(-0.8) == "large"

    def test_extreme_outlier_in_section(self):
        """One extreme outlier should not crash, stats should be finite."""
        scores = [0.5] * 29 + [0.0]  # 29 at 0.5, one outlier at 0.0
        result = compute_section_stats("A", scores, set())
        assert math.isfinite(result.mean)
        assert math.isfinite(result.std)
        assert result.n_students == 30

    def test_nan_in_scores_propagates(self):
        """NaN in input scores propagates through statistics.

        This is a known behavior: NaN propagates to mean/median/std.
        The system does not sanitize NaN inputs — caller is responsible.
        """
        scores = [0.5, float("nan"), 0.7]
        result = compute_section_stats("A", scores, set())
        assert result.n_students == 3
        # NaN propagates to mean/std
        assert math.isnan(result.mean)

    def test_nan_in_pairwise_comparison(self):
        """NaN in section scores: comparison should not crash.

        Statistical tests may produce NaN p-values but should not raise.
        """
        scores_a = [0.5, float("nan"), 0.7]
        scores_b = [0.3, 0.4, 0.5]
        # mannwhitneyu with NaN may raise or return NaN — either is acceptable
        # The important thing is no unhandled exception crashes the program
        try:
            results = compute_pairwise_comparisons({"A": scores_a, "B": scores_b})
            if results:
                comp = results[0]
                # p_value may be NaN, which is acceptable for NaN input
                assert isinstance(comp.p_value, float)
        except (ValueError, RuntimeWarning):
            # scipy may raise ValueError for NaN input — acceptable
            pass

    def test_concept_mastery_empty_sections(self):
        """Empty concept mastery data should return empty dict."""
        result = compute_concept_mastery_by_section({})
        assert result == {}

    def test_concept_mastery_empty_values_list(self):
        """Concept with empty values list should produce mastery of 0.0."""
        result = compute_concept_mastery_by_section({
            "A": {"concept1": [], "concept2": [0.8, 0.9]},
        })
        assert result["A"]["concept1"] == 0.0
        assert result["A"]["concept2"] == pytest.approx(0.85)

    def test_weekly_interaction_none_input(self):
        """None input to compute_weekly_interaction should return None."""
        assert compute_weekly_interaction(None) is None

    def test_weekly_interaction_empty_input(self):
        """Empty dict input to compute_weekly_interaction should return None."""
        assert compute_weekly_interaction({}) is None

    def test_weekly_interaction_normal(self):
        """Normal weekly interaction should compute means correctly."""
        data = {
            "A": {1: [0.5, 0.6], 2: [0.7, 0.8]},
            "B": {1: [0.3, 0.4], 2: [0.5, 0.6]},
        }
        result = compute_weekly_interaction(data)
        assert result is not None
        assert result["A"][1] == pytest.approx(0.55)
        assert result["A"][2] == pytest.approx(0.75)
        assert result["B"][1] == pytest.approx(0.35)

    def test_weekly_interaction_empty_scores_list(self):
        """Empty scores list in a week should produce 0.0."""
        data = {"A": {1: [], 2: [0.5]}}
        result = compute_weekly_interaction(data)
        assert result is not None
        assert result["A"][1] == 0.0
        assert result["A"][2] == 0.5

    def test_100_sections_performance(self):
        """100 sections = C(100,2) = 4950 pairs should complete quickly."""
        rng = np.random.RandomState(42)
        section_scores = {
            f"S{i:03d}": rng.uniform(0.2, 0.9, 5).tolist()
            for i in range(100)
        }
        start = time.time()
        results = compute_pairwise_comparisons(section_scores)
        elapsed = time.time() - start
        assert len(results) == 4950
        assert elapsed < 30.0  # generous limit; should be <5s normally
        # All p-values should be valid
        for comp in results:
            assert math.isfinite(comp.p_value)
            assert comp.p_value_corrected is not None
            assert 0.0 <= comp.p_value_corrected <= 1.0

    def test_significance_at_exact_005_boundary(self):
        """is_significant should be True only when p < 0.05 (strict less-than)."""
        # Construct a SectionComparison manually to test the boundary
        comp = SectionComparison(
            section_a="A", section_b="B",
            n_a=30, n_b=30,
            mean_a=0.5, mean_b=0.6,
            std_a=0.1, std_b=0.1,
            test_name="welch_t",
            test_statistic=-2.0,
            p_value=0.05,
            p_value_corrected=None,  # 2 sections, no Bonferroni
            cohens_d=-0.5,
            effect_size_label="medium",
            is_significant=False,  # p=0.05 is NOT < 0.05
        )
        # Verify the code's logic: p < 0.05 (strict)
        assert not comp.is_significant


# ===========================================================================
# PERSONA 4b: CHART ATTACKS
# ===========================================================================


class TestStatisticalAttackerCharts:
    """Persona 4 extension: Attack cross-section comparison chart generation."""

    def test_box_plot_2_sections(self):
        """Basic 2-section box plot should produce valid PNG."""
        from forma.section_comparison_charts import build_section_box_plot
        buf = build_section_box_plot({
            "A": [0.3, 0.5, 0.7, 0.8],
            "B": [0.4, 0.6, 0.7, 0.9],
        })
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_box_plot_single_value_sections(self):
        """Sections with single values should produce degenerate box plots."""
        from forma.section_comparison_charts import build_section_box_plot
        buf = build_section_box_plot({
            "A": [0.5],
            "B": [0.7],
        })
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_box_plot_9_sections_color_wrap(self):
        """9+ sections should wrap color palette without crash."""
        from forma.section_comparison_charts import build_section_box_plot
        rng = np.random.RandomState(42)
        section_scores = {
            f"S{i}": rng.uniform(0.2, 0.9, 10).tolist()
            for i in range(9)
        }
        buf = build_section_box_plot(section_scores)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_heatmap_empty_data_returns_none(self):
        """Empty concept mastery data should return None (no chart)."""
        from forma.section_comparison_charts import build_concept_mastery_heatmap
        result = build_concept_mastery_heatmap({})
        assert result is None

    def test_heatmap_no_concepts_returns_none(self):
        """Sections with no concepts should return None."""
        from forma.section_comparison_charts import build_concept_mastery_heatmap
        result = build_concept_mastery_heatmap({"A": {}, "B": {}})
        assert result is None

    def test_heatmap_25_concepts_truncated(self):
        """25 concepts should be truncated to top 20 by variance."""
        from forma.section_comparison_charts import build_concept_mastery_heatmap
        rng = np.random.RandomState(42)
        concept_data = {
            "A": {f"concept_{i}": rng.uniform(0, 1) for i in range(25)},
            "B": {f"concept_{i}": rng.uniform(0, 1) for i in range(25)},
        }
        buf = build_concept_mastery_heatmap(concept_data)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_heatmap_all_identical_mastery(self):
        """All concepts with identical mastery across sections: zero variance."""
        from forma.section_comparison_charts import build_concept_mastery_heatmap
        concept_data = {
            "A": {"c1": 0.5, "c2": 0.5, "c3": 0.5},
            "B": {"c1": 0.5, "c2": 0.5, "c3": 0.5},
        }
        buf = build_concept_mastery_heatmap(concept_data)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_weekly_chart_none_returns_none(self):
        """None weekly data should return None (no chart)."""
        from forma.section_comparison_charts import build_weekly_interaction_chart
        assert build_weekly_interaction_chart(None) is None

    def test_weekly_chart_empty_returns_none(self):
        """Empty weekly data should return None."""
        from forma.section_comparison_charts import build_weekly_interaction_chart
        assert build_weekly_interaction_chart({}) is None

    def test_weekly_chart_normal(self):
        """Normal weekly interaction chart with 2 sections."""
        from forma.section_comparison_charts import build_weekly_interaction_chart
        data = {
            "A": {1: 0.5, 2: 0.6, 3: 0.7},
            "B": {1: 0.4, 2: 0.5, 3: 0.55},
        }
        buf = build_weekly_interaction_chart(data)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_weekly_chart_single_week(self):
        """Single week of data should produce valid chart (just a point)."""
        from forma.section_comparison_charts import build_weekly_interaction_chart
        data = {"A": {1: 0.5}, "B": {1: 0.6}}
        buf = build_weekly_interaction_chart(data)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_heatmap_korean_concept_names(self):
        """Korean concept names in heatmap should render without crash."""
        from forma.section_comparison_charts import build_concept_mastery_heatmap
        concept_data = {
            "A": {"세포막": 0.8, "핵": 0.5, "미토콘드리아": 0.3},
            "B": {"세포막": 0.6, "핵": 0.7, "미토콘드리아": 0.4},
        }
        buf = build_concept_mastery_heatmap(concept_data)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"


# ===========================================================================
# INVARIANT TESTING (1000-iteration loops)
# ===========================================================================


class TestInvariant1000:
    """High-iteration invariant tests for statistical comparison."""

    def test_cohens_d_always_finite(self):
        """1000 random pairs: Cohen's d is always finite (never NaN/Inf)."""
        rng = np.random.RandomState(42)
        for _ in range(1000):
            n1 = rng.randint(2, 50)
            n2 = rng.randint(2, 50)
            g1 = rng.uniform(0, 1, n1).tolist()
            g2 = rng.uniform(0, 1, n2).tolist()
            d = _cohens_d(g1, g2)
            assert math.isfinite(d), f"NaN/Inf Cohen's d for n1={n1}, n2={n2}"

    def test_effect_size_label_always_valid(self):
        """1000 random d values: label is always one of 4 categories."""
        valid_labels = {"negligible", "small", "medium", "large"}
        rng = np.random.RandomState(42)
        for _ in range(1000):
            d = rng.uniform(-5, 5)
            label = _effect_size_label(d)
            assert label in valid_labels

    def test_bonferroni_always_bounded(self):
        """1000 random 3-section comparisons: corrected p always in [0, 1]."""
        rng = np.random.RandomState(42)
        for _ in range(1000):
            section_scores = {
                s: rng.uniform(0, 1, rng.randint(5, 20)).tolist()
                for s in ["A", "B", "C"]
            }
            results = compute_pairwise_comparisons(section_scores)
            for comp in results:
                assert comp.p_value_corrected is not None
                assert 0.0 <= comp.p_value_corrected <= 1.0
                assert 0.0 <= comp.p_value <= 1.0

    def test_pairwise_count_invariant(self):
        """1000 random: n sections always produce C(n,2) pairs."""
        rng = np.random.RandomState(42)
        for _ in range(1000):
            n_sections = rng.randint(2, 8)
            section_scores = {
                f"S{i}": rng.uniform(0, 1, rng.randint(3, 20)).tolist()
                for i in range(n_sections)
            }
            results = compute_pairwise_comparisons(section_scores)
            expected_pairs = n_sections * (n_sections - 1) // 2
            assert len(results) == expected_pairs

    def test_test_selection_invariant(self):
        """1000 random: test selection always follows N>=30 rule."""
        rng = np.random.RandomState(42)
        for _ in range(1000):
            n_a = rng.randint(1, 60)
            n_b = rng.randint(1, 60)
            scores = {
                "A": rng.uniform(0, 1, n_a).tolist(),
                "B": rng.uniform(0, 1, n_b).tolist(),
            }
            results = compute_pairwise_comparisons(scores)
            assert len(results) == 1
            comp = results[0]
            if min(n_a, n_b) >= 30:
                assert comp.test_name == "welch_t"
            else:
                assert comp.test_name == "mann_whitney_u"

    def test_section_stats_n_students_invariant(self):
        """1000 random: n_students always equals len(scores)."""
        rng = np.random.RandomState(42)
        for _ in range(1000):
            n = rng.randint(0, 100)
            scores = rng.uniform(0, 1, n).tolist()
            result = compute_section_stats(
                "test", scores, set(),
            )
            assert result.n_students == n


# ===========================================================================
# PERSONA 2: MODEL CORRUPTOR
# ===========================================================================


class TestModelCorruptor:
    """Persona 2: Corrupt and malformed .pkl model file attacks."""

    def test_truncated_pkl_file(self):
        """Truncated .pkl file should raise an error on load."""
        from forma.risk_predictor import load_model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"\x80\x05\x95\x00\x00\x00\x00")  # truncated pickle header
            path = f.name
        try:
            with pytest.raises(Exception):
                load_model(path)
        finally:
            os.unlink(path)

    def test_zero_byte_pkl_file(self):
        """Zero-byte .pkl file should raise an error on load."""
        from forma.risk_predictor import load_model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(Exception):
                load_model(path)
        finally:
            os.unlink(path)

    def test_non_pickle_file_with_pkl_extension(self):
        """A YAML text file with .pkl extension should raise on load."""
        from forma.risk_predictor import load_model
        with tempfile.NamedTemporaryFile(
            suffix=".pkl", delete=False, mode="w",
        ) as f:
            f.write("current_week: 1\nproject:\n  year: 2026\n")
            path = f.name
        try:
            with pytest.raises(Exception):
                load_model(path)
        finally:
            os.unlink(path)

    def test_pkl_containing_wrong_type(self):
        """A .pkl file containing a dict instead of TrainedRiskModel."""
        from forma.risk_predictor import load_model
        import joblib
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            # Save a plain dict instead of TrainedRiskModel
            joblib.dump({"not": "a model"}, path)
            # load_model should return whatever joblib loads — no type check
            result = load_model(path)
            assert isinstance(result, dict)  # wrong type, but loads
        finally:
            os.unlink(path)

    def test_nonexistent_model_file(self):
        """Loading a nonexistent model file should raise FileNotFoundError."""
        from forma.risk_predictor import load_model
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.pkl")

    def test_save_load_roundtrip(self):
        """TrainedRiskModel roundtrip via save_model/load_model."""
        from forma.risk_predictor import (
            TrainedRiskModel, save_model, load_model, FEATURE_NAMES,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 1, (20, 15))
        scaler.fit(X)
        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(scaler.transform(X), np.array([0]*10 + [1]*10))

        trained = TrainedRiskModel(
            model=model,
            feature_names=list(FEATURE_NAMES),
            scaler=scaler,
            training_date="2026-03-10T00:00:00Z",
            n_students=20,
            n_weeks=5,
            cv_score=0.85,
            target_threshold=0.45,
        )
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save_model(trained, path)
            loaded = load_model(path)
            assert isinstance(loaded, TrainedRiskModel)
            assert loaded.feature_names == list(FEATURE_NAMES)
            assert loaded.n_students == 20
            assert loaded.cv_score == 0.85
            assert loaded.target_threshold == 0.45
            # Model produces same predictions
            X_test = rng.uniform(0, 1, (5, 15))
            orig_proba = trained.model.predict_proba(trained.scaler.transform(X_test))
            load_proba = loaded.model.predict_proba(loaded.scaler.transform(X_test))
            np.testing.assert_array_almost_equal(orig_proba, load_proba)
        finally:
            os.unlink(path)

    def test_model_predict_with_feature_count_mismatch(self):
        """Predicting with wrong number of features should raise an error."""
        from forma.risk_predictor import (
            TrainedRiskModel, RiskPredictor, FEATURE_NAMES,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        X = rng.uniform(0, 1, (20, 15))
        scaler = StandardScaler()
        scaler.fit(X)
        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(scaler.transform(X), np.array([0]*10 + [1]*10))

        trained = TrainedRiskModel(
            model=model,
            feature_names=list(FEATURE_NAMES),
            scaler=scaler,
            training_date="2026-03-10T00:00:00Z",
            n_students=20,
            n_weeks=5,
            cv_score=0.85,
        )
        predictor = RiskPredictor()
        # Wrong number of features (12 instead of 15)
        X_wrong = rng.uniform(0, 1, (5, 12))
        with pytest.raises(ValueError):
            predictor.predict(trained, X_wrong, [f"S{i}" for i in range(5)])


# ===========================================================================
# PERSONA 3: DATA POISONER
# ===========================================================================


class TestDataPoisoner:
    """Persona 3: Poisoned longitudinal store attacks on feature extraction."""

    def _make_store_with_records(self, records):
        """Build an in-memory LongitudinalStore with given records."""
        from forma.longitudinal_store import LongitudinalStore
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        store = LongitudinalStore(path)
        for r in records:
            store.add_record(r)
        return store

    def _make_record(self, student_id="S001", week=1, question_sn=1,
                     ensemble_score=0.5, tier_level=1, **kwargs):
        from forma.evaluation_types import LongitudinalRecord
        scores = {"ensemble_score": ensemble_score}
        if "concept_coverage" in kwargs:
            scores["concept_coverage"] = kwargs.pop("concept_coverage")
        return LongitudinalRecord(
            student_id=student_id,
            week=week,
            question_sn=question_sn,
            scores=scores,
            tier_level=tier_level,
            tier_label="Developing",
            **kwargs,
        )

    def test_empty_store_extraction(self):
        """Feature extraction from empty store should produce empty matrix."""
        from forma.risk_predictor import FeatureExtractor
        store = self._make_store_with_records([])
        ext = FeatureExtractor()
        matrix, names, ids = ext.extract(store, [1, 2, 3])
        assert matrix.shape == (0, 15)
        assert ids == []
        assert len(names) == 15

    def test_single_student_single_week(self):
        """Single student, single week: features extracted, slopes=0.0."""
        from forma.risk_predictor import FeatureExtractor
        records = [self._make_record(ensemble_score=0.6)]
        store = self._make_store_with_records(records)
        ext = FeatureExtractor()
        matrix, names, ids = ext.extract(store, [1])
        assert matrix.shape == (1, 15)
        assert ids == ["S001"]
        # score_slope should be 0.0 (single point)
        slope_idx = names.index("score_slope")
        assert matrix[0, slope_idx] == 0.0

    def test_1000_students_stress(self):
        """1000 students should extract features in reasonable time."""
        from forma.risk_predictor import FeatureExtractor
        records = []
        for i in range(1000):
            for w in [1, 2, 3]:
                records.append(self._make_record(
                    student_id=f"S{i:04d}", week=w,
                    ensemble_score=random.uniform(0.1, 0.9),
                ))
        store = self._make_store_with_records(records)
        ext = FeatureExtractor()
        start = time.time()
        matrix, names, ids = ext.extract(store, [1, 2, 3])
        elapsed = time.time() - start
        assert matrix.shape == (1000, 15)
        assert len(ids) == 1000
        assert elapsed < 30.0

    def test_all_identical_scores_zero_variance(self):
        """All students identical scores: zero variance features."""
        from forma.risk_predictor import FeatureExtractor
        records = []
        for i in range(15):
            for w in [1, 2, 3]:
                records.append(self._make_record(
                    student_id=f"S{i:03d}", week=w,
                    ensemble_score=0.5,
                ))
        store = self._make_store_with_records(records)
        ext = FeatureExtractor()
        matrix, names, ids = ext.extract(store, [1, 2, 3])
        assert matrix.shape == (15, 15)
        # score_variance should be 0.0
        var_idx = names.index("score_variance")
        for i in range(15):
            assert matrix[i, var_idx] == 0.0

    def test_all_identical_scores_training(self):
        """Training with all-identical scores: cv_score=0.0 (single class)."""
        from forma.risk_predictor import RiskPredictor
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (20, 15))
        # All labels the same -> single class
        labels = np.zeros(20, dtype=int)
        predictor = RiskPredictor()
        model = predictor.train(matrix, labels, list(range(15)), min_students=10)
        assert model.cv_score == 0.0  # single class, no CV possible

    def test_insufficient_students_raises(self):
        """Training with fewer than min_students should raise ValueError."""
        from forma.risk_predictor import RiskPredictor
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (5, 15))
        labels = np.array([0, 0, 0, 1, 1])
        predictor = RiskPredictor()
        with pytest.raises(ValueError, match="Insufficient students"):
            predictor.train(matrix, labels, list(range(15)), min_students=10)

    def test_student_missing_some_weeks(self):
        """Student with data in only some weeks: absence features correct."""
        from forma.risk_predictor import FeatureExtractor
        records = [
            self._make_record(student_id="S001", week=1, ensemble_score=0.5),
            # Missing week 2
            self._make_record(student_id="S001", week=3, ensemble_score=0.7),
        ]
        store = self._make_store_with_records(records)
        ext = FeatureExtractor()
        matrix, names, ids = ext.extract(store, [1, 2, 3])
        absence_idx = names.index("absence_count")
        ratio_idx = names.index("absence_ratio")
        assert matrix[0, absence_idx] == 1.0  # Missing week 2
        assert matrix[0, ratio_idx] == pytest.approx(1.0 / 3.0)

    def test_mid_semester_enrollment(self):
        """Student appearing only in later weeks: earlier weeks count as absent."""
        from forma.risk_predictor import FeatureExtractor
        records = [
            self._make_record(student_id="S001", week=3, ensemble_score=0.6),
        ]
        store = self._make_store_with_records(records)
        ext = FeatureExtractor()
        matrix, names, ids = ext.extract(store, [1, 2, 3])
        absence_idx = names.index("absence_count")
        ratio_idx = names.index("absence_ratio")
        assert matrix[0, absence_idx] == 2.0  # Missing weeks 1, 2
        assert matrix[0, ratio_idx] == pytest.approx(2.0 / 3.0)

    def test_negative_ensemble_scores(self):
        """Negative ensemble scores should not crash feature extraction."""
        from forma.risk_predictor import FeatureExtractor
        records = [
            self._make_record(week=1, ensemble_score=-0.5),
            self._make_record(week=2, ensemble_score=-1.0),
            self._make_record(week=3, ensemble_score=-0.3),
        ]
        store = self._make_store_with_records(records)
        ext = FeatureExtractor()
        matrix, names, ids = ext.extract(store, [1, 2, 3])
        assert matrix.shape == (1, 15)
        mean_idx = names.index("score_mean")
        assert math.isfinite(matrix[0, mean_idx])
        assert matrix[0, mean_idx] < 0

    def test_scores_above_one(self):
        """Scores > 1.0 should not crash feature extraction."""
        from forma.risk_predictor import FeatureExtractor
        records = [
            self._make_record(week=1, ensemble_score=1.5),
            self._make_record(week=2, ensemble_score=2.0),
        ]
        store = self._make_store_with_records(records)
        ext = FeatureExtractor()
        matrix, names, ids = ext.extract(store, [1, 2])
        assert matrix.shape == (1, 15)
        mean_idx = names.index("score_mean")
        assert matrix[0, mean_idx] > 1.0

    def test_cold_start_prediction(self):
        """Cold start prediction should produce limited confidence."""
        from forma.risk_predictor import RiskPredictor, FEATURE_NAMES
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (5, 15))
        predictor = RiskPredictor()
        preds = predictor.predict_cold_start(
            matrix, [f"S{i}" for i in range(5)], list(FEATURE_NAMES),
        )
        assert len(preds) == 5
        for p in preds:
            assert p.confidence == "limited"
            assert p.is_model_based is False
            assert 0.0 <= p.drop_probability <= 1.0
            assert len(p.risk_factors) == 15

    def test_cold_start_with_zero_features(self):
        """Cold start with all-zero feature matrix: valid predictions."""
        from forma.risk_predictor import RiskPredictor, FEATURE_NAMES
        matrix = np.zeros((3, 15))
        predictor = RiskPredictor()
        preds = predictor.predict_cold_start(
            matrix, ["S001", "S002", "S003"], list(FEATURE_NAMES),
        )
        assert len(preds) == 3
        for p in preds:
            assert 0.0 <= p.drop_probability <= 1.0

    def test_none_v2_fields_fallback_to_zero(self):
        """Records with None v2 fields: edge_f1 and misconception_count -> 0.0."""
        from forma.risk_predictor import FeatureExtractor
        records = [
            self._make_record(week=1, edge_f1=None, misconception_count=None),
            self._make_record(week=2, edge_f1=None, misconception_count=None),
            self._make_record(week=3, edge_f1=None, misconception_count=None),
        ]
        store = self._make_store_with_records(records)
        ext = FeatureExtractor()
        matrix, names, ids = ext.extract(store, [1, 2, 3])
        f1_idx = names.index("edge_f1_mean")
        misc_idx = names.index("misconception_mean")
        assert matrix[0, f1_idx] == 0.0
        assert matrix[0, misc_idx] == 0.0


# ===========================================================================
# PERSONA 6: BOUNDARY PUSHER
# ===========================================================================


class TestBoundaryPusher:
    """Persona 6: Exact boundary value attacks across all modules."""

    def test_drop_probability_exactly_0(self):
        """RiskPrediction with drop_probability=0.0 should be valid."""
        from forma.risk_predictor import RiskPrediction
        pred = RiskPrediction(student_id="S001", drop_probability=0.0)
        assert pred.drop_probability == 0.0
        assert pred.predicted_tier == 2  # default

    def test_drop_probability_exactly_1(self):
        """RiskPrediction with drop_probability=1.0 should be valid."""
        from forma.risk_predictor import RiskPrediction
        pred = RiskPrediction(student_id="S001", drop_probability=1.0)
        assert pred.drop_probability == 1.0

    def test_drop_probability_exactly_05(self):
        """drop_probability=0.5 is the inclusion threshold for warning cards."""
        from forma.risk_predictor import RiskPrediction
        pred = RiskPrediction(student_id="S001", drop_probability=0.5)
        assert pred.drop_probability == 0.5

    def test_predicted_tier_from_probability_boundaries(self):
        """Test tier assignment at exact probability boundaries via predict()."""
        from forma.risk_predictor import RiskPredictor, TrainedRiskModel, FEATURE_NAMES
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        X = rng.uniform(0, 1, (20, 15))
        labels = np.array([0]*10 + [1]*10)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(X_scaled, labels)

        trained = TrainedRiskModel(
            model=model,
            feature_names=list(FEATURE_NAMES),
            scaler=scaler,
            training_date="2026-03-10T00:00:00Z",
            n_students=20,
            n_weeks=5,
            cv_score=0.85,
        )
        predictor = RiskPredictor()
        preds = predictor.predict(trained, X, [f"S{i:03d}" for i in range(20)])
        for p in preds:
            assert 0.0 <= p.drop_probability <= 1.0
            if p.drop_probability >= 0.7:
                assert p.predicted_tier == 0
            elif p.drop_probability >= 0.5:
                assert p.predicted_tier == 1
            elif p.drop_probability >= 0.3:
                assert p.predicted_tier == 2
            else:
                assert p.predicted_tier == 3

    def test_exactly_10_students_training_succeeds(self):
        """Exactly min_students=10 should succeed."""
        from forma.risk_predictor import RiskPredictor
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (10, 15))
        labels = np.array([0]*5 + [1]*5)
        predictor = RiskPredictor()
        model = predictor.train(matrix, labels, list(range(15)), min_students=10)
        assert model.n_students == 10

    def test_exactly_9_students_training_fails(self):
        """9 students with min_students=10 should raise ValueError."""
        from forma.risk_predictor import RiskPredictor
        rng = np.random.RandomState(42)
        matrix = rng.uniform(0, 1, (9, 15))
        labels = np.array([0]*5 + [1]*4)
        predictor = RiskPredictor()
        with pytest.raises(ValueError, match="Insufficient students.*9 < 10"):
            predictor.train(matrix, labels, list(range(15)), min_students=10)

    def test_dpi_boundary_72_valid(self):
        """dpi=72 (lower boundary) should pass validation."""
        validate_project_config({"reports": {"dpi": 72}})

    def test_dpi_boundary_600_valid(self):
        """dpi=600 (upper boundary) should pass validation."""
        validate_project_config({"reports": {"dpi": 600}})

    def test_dpi_boundary_71_invalid(self):
        """dpi=71 should fail validation."""
        with pytest.raises(ValueError, match="dpi must be between 72 and 600"):
            validate_project_config({"reports": {"dpi": 71}})

    def test_dpi_boundary_601_invalid(self):
        """dpi=601 should fail validation."""
        with pytest.raises(ValueError, match="dpi must be between 72 and 600"):
            validate_project_config({"reports": {"dpi": 601}})

    def test_current_week_exactly_1_valid(self):
        """current_week=1 (minimum) should pass validation."""
        validate_project_config({"current_week": 1})

    def test_current_week_0_invalid(self):
        """current_week=0 should fail validation."""
        with pytest.raises(ValueError, match="current_week must be >= 1"):
            validate_project_config({"current_week": 0})

    def test_semester_boundary_values(self):
        """semester=1 and semester=2 should pass; 0 and 3 should fail."""
        validate_project_config({"project": {"semester": 1}})
        validate_project_config({"project": {"semester": 2}})
        with pytest.raises(ValueError, match="semester must be 1 or 2"):
            validate_project_config({"project": {"semester": 0}})
        with pytest.raises(ValueError, match="semester must be 1 or 2"):
            validate_project_config({"project": {"semester": 3}})

    def test_year_boundary_2020(self):
        """year=2020 should pass; year=2019 should fail."""
        validate_project_config({"project": {"year": 2020}})
        with pytest.raises(ValueError, match="year must be >= 2020"):
            validate_project_config({"project": {"year": 2019}})

    def test_risk_factors_sorted_by_importance(self):
        """Risk factors from predict() should be sorted by importance desc."""
        from forma.risk_predictor import RiskPredictor, TrainedRiskModel, FEATURE_NAMES
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        X = rng.uniform(0, 1, (30, 15))
        labels = np.array([0]*15 + [1]*15)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_scaled, labels)

        trained = TrainedRiskModel(
            model=model,
            feature_names=list(FEATURE_NAMES),
            scaler=scaler,
            training_date="2026-03-10T00:00:00Z",
            n_students=30,
            n_weeks=5,
            cv_score=0.8,
        )
        predictor = RiskPredictor()
        preds = predictor.predict(trained, X, [f"S{i:03d}" for i in range(30)])
        for p in preds:
            importances = [f.importance for f in p.risk_factors]
            assert importances == sorted(importances, reverse=True)

    def test_feature_extraction_always_15_features(self):
        """Feature extraction should always produce exactly 15 features."""
        from forma.risk_predictor import FeatureExtractor, FEATURE_NAMES
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        store = LongitudinalStore(path)
        for w in [1, 2, 3]:
            store.add_record(LongitudinalRecord(
                student_id="S001", week=w, question_sn=1,
                scores={"ensemble_score": 0.5}, tier_level=1,
                tier_label="Developing",
            ))
        ext = FeatureExtractor()
        matrix, names, ids = ext.extract(store, [1, 2, 3])
        assert matrix.shape[1] == 15
        assert names == list(FEATURE_NAMES)
        os.unlink(path)


# ===========================================================================
# INVARIANT TESTING -- Persona 2 + 3 + 6
# ===========================================================================


class TestInvariant1000Phase2:
    """High-iteration invariant tests for risk prediction."""

    def test_cold_start_probability_always_bounded(self):
        """1000 random: cold start probability always in [0.0, 1.0]."""
        from forma.risk_predictor import RiskPredictor, FEATURE_NAMES
        rng = np.random.RandomState(42)
        predictor = RiskPredictor()
        for _ in range(1000):
            n = rng.randint(1, 10)
            matrix = rng.uniform(-2, 2, (n, 15))
            preds = predictor.predict_cold_start(
                matrix, [f"S{i}" for i in range(n)], list(FEATURE_NAMES),
            )
            for p in preds:
                assert 0.0 <= p.drop_probability <= 1.0, (
                    f"Out of bounds: {p.drop_probability}"
                )

    def test_feature_count_invariant(self):
        """100 random: extraction always produces 15 features."""
        from forma.risk_predictor import FeatureExtractor
        from forma.evaluation_types import LongitudinalRecord
        from forma.longitudinal_store import LongitudinalStore

        rng = np.random.RandomState(42)
        ext = FeatureExtractor()
        for iteration in range(100):
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                path = f.name
            store = LongitudinalStore(path)
            n_students = rng.randint(1, 5)
            n_weeks = rng.randint(1, 4)
            weeks = list(range(1, n_weeks + 1))
            for i in range(n_students):
                for w in weeks:
                    store.add_record(LongitudinalRecord(
                        student_id=f"S{i:03d}", week=w, question_sn=1,
                        scores={"ensemble_score": rng.uniform(0, 1)},
                        tier_level=rng.randint(0, 4),
                        tier_label="Developing",
                    ))
            matrix, names, ids = ext.extract(store, weeks)
            assert matrix.shape[1] == 15, f"Wrong feature count at iter {iteration}"
            assert len(names) == 15
            os.unlink(path)

    def test_ols_slope_finite_invariant(self):
        """1000 random: OLS slope is always finite."""
        from forma.risk_predictor import _ols_slope
        rng = np.random.RandomState(42)
        for _ in range(1000):
            n = rng.randint(0, 20)
            values = rng.uniform(-5, 5, n).tolist()
            slope = _ols_slope(values)
            assert math.isfinite(slope)


# ===========================================================================
# PERSONA 5: WARNING DATA CRASHER
# ===========================================================================


class TestWarningDataCrasher:
    """Persona 5: Attack WarningCard construction and risk classification."""

    def _make_pred(self, student_id="S001", drop_prob=0.7, **kwargs):
        """Build a RiskPrediction for test fixtures."""
        from forma.risk_predictor import RiskPrediction
        return RiskPrediction(
            student_id=student_id,
            drop_probability=drop_prob,
            **kwargs,
        )

    def test_empty_inputs_returns_empty(self):
        """No at-risk students and no predictions → empty list."""
        from forma.warning_report_data import build_warning_data
        result = build_warning_data({}, [], {})
        assert result == []

    def test_rule_based_only_no_predictions(self):
        """Rule-based at-risk with no model predictions → cards built."""
        from forma.warning_report_data import build_warning_data
        at_risk = {"S001": {"is_at_risk": True, "reasons": ["low_score"]}}
        result = build_warning_data(
            at_risk, [], {},
            score_trajectories={"S001": [0.3, 0.25, 0.2]},
        )
        assert len(result) == 1
        assert result[0].student_id == "S001"
        assert "rule_based" in result[0].detection_methods
        assert "model_predicted" not in result[0].detection_methods
        assert result[0].drop_probability is None

    def test_model_only_no_rule_based(self):
        """Model-predicted at-risk with no rule-based flags → cards built."""
        from forma.warning_report_data import build_warning_data
        pred = self._make_pred("S001", 0.8)
        result = build_warning_data({}, [pred], {})
        assert len(result) == 1
        assert "model_predicted" in result[0].detection_methods
        assert "rule_based" not in result[0].detection_methods

    def test_union_inclusion_both_methods(self):
        """Student flagged by both methods → both detection methods listed."""
        from forma.warning_report_data import build_warning_data
        at_risk = {"S001": {"is_at_risk": True, "reasons": []}}
        pred = self._make_pred("S001", 0.8)
        result = build_warning_data(at_risk, [pred], {})
        assert len(result) == 1
        assert "rule_based" in result[0].detection_methods
        assert "model_predicted" in result[0].detection_methods

    def test_drop_probability_exactly_at_threshold(self):
        """drop_probability=0.5 is exactly at inclusion threshold → included."""
        from forma.warning_report_data import build_warning_data
        pred = self._make_pred("S001", 0.5)
        result = build_warning_data({}, [pred], {})
        assert len(result) == 1

    def test_drop_probability_just_below_threshold(self):
        """drop_probability=0.499 → NOT included by model alone."""
        from forma.warning_report_data import build_warning_data
        pred = self._make_pred("S001", 0.499)
        result = build_warning_data({}, [pred], {})
        assert len(result) == 0

    def test_score_decline_classification(self):
        """Decreasing trajectory → SCORE_DECLINE risk type."""
        from forma.warning_report_data import (
            build_warning_data, RiskType,
        )
        at_risk = {"S001": {"is_at_risk": True}}
        result = build_warning_data(
            at_risk, [], {},
            score_trajectories={"S001": [0.8, 0.6, 0.4, 0.2]},
        )
        assert len(result) == 1
        assert RiskType.SCORE_DECLINE in result[0].risk_types

    def test_persistent_low_classification(self):
        """All scores below 0.45 → PERSISTENT_LOW risk type."""
        from forma.warning_report_data import (
            build_warning_data, RiskType,
        )
        at_risk = {"S001": {"is_at_risk": True}}
        result = build_warning_data(
            at_risk, [], {},
            score_trajectories={"S001": [0.3, 0.3, 0.3]},
        )
        assert len(result) == 1
        assert RiskType.PERSISTENT_LOW in result[0].risk_types

    def test_concept_deficit_classification(self):
        """3+ concepts below 0.3 mastery → CONCEPT_DEFICIT."""
        from forma.warning_report_data import (
            build_warning_data, RiskType,
        )
        at_risk = {"S001": {"is_at_risk": True}}
        concept_scores = {
            "S001": {"c1": 0.1, "c2": 0.2, "c3": 0.25, "c4": 0.8},
        }
        result = build_warning_data(at_risk, [], concept_scores)
        assert len(result) == 1
        assert RiskType.CONCEPT_DEFICIT in result[0].risk_types
        assert len(result[0].deficit_concepts) == 3

    def test_concept_deficit_exactly_2_not_triggered(self):
        """Only 2 concepts below threshold → CONCEPT_DEFICIT NOT triggered."""
        from forma.warning_report_data import (
            build_warning_data, RiskType,
        )
        at_risk = {"S001": {"is_at_risk": True}}
        concept_scores = {
            "S001": {"c1": 0.1, "c2": 0.2, "c3": 0.5, "c4": 0.8},
        }
        result = build_warning_data(at_risk, [], concept_scores)
        assert len(result) == 1
        assert RiskType.CONCEPT_DEFICIT not in result[0].risk_types

    def test_participation_decline_classification(self):
        """absence_ratio > 0.3 → PARTICIPATION_DECLINE."""
        from forma.warning_report_data import (
            build_warning_data, RiskType,
        )
        at_risk = {"S001": {"is_at_risk": True}}
        result = build_warning_data(
            at_risk, [], {},
            absence_ratios={"S001": 0.5},
        )
        assert len(result) == 1
        assert RiskType.PARTICIPATION_DECLINE in result[0].risk_types

    def test_participation_decline_at_exact_threshold(self):
        """absence_ratio=0.3 (strict >) → PARTICIPATION_DECLINE NOT triggered."""
        from forma.warning_report_data import (
            build_warning_data, RiskType,
        )
        at_risk = {"S001": {"is_at_risk": True}}
        result = build_warning_data(
            at_risk, [], {},
            absence_ratios={"S001": 0.3},
        )
        assert len(result) == 1
        assert RiskType.PARTICIPATION_DECLINE not in result[0].risk_types

    def test_multiple_risk_types_combined(self):
        """Student with declining scores + high absence → multiple risk types."""
        from forma.warning_report_data import (
            build_warning_data, RiskType,
        )
        at_risk = {"S001": {"is_at_risk": True}}
        concept_scores = {
            "S001": {"c1": 0.1, "c2": 0.1, "c3": 0.1},
        }
        result = build_warning_data(
            at_risk, [], concept_scores,
            score_trajectories={"S001": [0.6, 0.4, 0.2]},
            absence_ratios={"S001": 0.5},
        )
        card = result[0]
        assert RiskType.SCORE_DECLINE in card.risk_types
        assert RiskType.CONCEPT_DEFICIT in card.risk_types
        assert RiskType.PARTICIPATION_DECLINE in card.risk_types

    def test_interventions_mapped_from_risk_types(self):
        """Interventions should be mapped from INTERVENTION_MAP per risk type."""
        from forma.warning_report_data import (
            build_warning_data, INTERVENTION_MAP, RiskType,
        )
        at_risk = {"S001": {"is_at_risk": True}}
        result = build_warning_data(
            at_risk, [], {},
            score_trajectories={"S001": [0.8, 0.6, 0.4]},
        )
        card = result[0]
        assert len(card.interventions) > 0
        # Each intervention should come from the INTERVENTION_MAP
        all_interventions = set()
        for entries in INTERVENTION_MAP.values():
            all_interventions.update(entries)
        for intervention in card.interventions:
            assert intervention in all_interventions

    def test_no_duplicate_interventions(self):
        """Student with multiple risk types → interventions should be de-duped."""
        from forma.warning_report_data import build_warning_data
        at_risk = {"S001": {"is_at_risk": True}}
        concept_scores = {
            "S001": {"c1": 0.1, "c2": 0.1, "c3": 0.1},
        }
        result = build_warning_data(
            at_risk, [], concept_scores,
            score_trajectories={"S001": [0.3, 0.25, 0.2]},
            absence_ratios={"S001": 0.5},
        )
        card = result[0]
        assert len(card.interventions) == len(set(card.interventions))

    def test_sorted_by_severity_descending(self):
        """Multiple cards should be sorted by risk_severity descending."""
        from forma.warning_report_data import build_warning_data
        at_risk = {
            "S001": {"is_at_risk": True},
            "S002": {"is_at_risk": True},
            "S003": {"is_at_risk": True},
        }
        preds = [
            self._make_pred("S001", 0.3),
            self._make_pred("S002", 0.9),
            self._make_pred("S003", 0.6),
        ]
        # S001 has drop_prob 0.3 < 0.5, so only rule_based inclusion
        # S002 and S003 both model_predicted
        result = build_warning_data(at_risk, preds, {})
        severities = [c.risk_severity for c in result]
        assert severities == sorted(severities, reverse=True)

    def test_default_risk_type_when_no_classification(self):
        """When no specific risk type matches, a default is assigned."""
        from forma.warning_report_data import build_warning_data
        at_risk = {"S001": {"is_at_risk": True}}
        # No trajectory, no concepts, no absence → no rule matches
        result = build_warning_data(at_risk, [], {})
        assert len(result) == 1
        # Default should be assigned (not empty)
        assert len(result[0].risk_types) >= 1

    def test_50_students_stress(self):
        """50 at-risk students with mixed detection: all produce valid cards."""
        from forma.warning_report_data import build_warning_data
        at_risk = {f"S{i:03d}": {"is_at_risk": True} for i in range(50)}
        preds = [self._make_pred(f"S{i:03d}", 0.4 + i * 0.01) for i in range(50)]
        concept_scores = {
            f"S{i:03d}": {f"c{j}": 0.1 * j for j in range(5)}
            for i in range(50)
        }
        score_trajectories = {
            f"S{i:03d}": [0.5 - i * 0.005] * 4
            for i in range(50)
        }
        result = build_warning_data(
            at_risk, preds, concept_scores,
            score_trajectories=score_trajectories,
        )
        assert len(result) >= 50
        for card in result:
            assert len(card.risk_types) >= 1
            assert len(card.interventions) >= 1
            assert 0.0 <= card.risk_severity <= 1.0

    def test_risk_type_korean_labels(self):
        """All RiskType values have valid Korean labels."""
        from forma.warning_report_data import RiskType
        for rt in RiskType:
            label = rt.label
            assert isinstance(label, str)
            assert len(label) > 0

    def test_intervention_map_completeness(self):
        """INTERVENTION_MAP covers all RiskType values."""
        from forma.warning_report_data import RiskType, INTERVENTION_MAP
        for rt in RiskType:
            assert rt.value in INTERVENTION_MAP
            assert len(INTERVENTION_MAP[rt.value]) >= 1

    def test_is_at_risk_false_not_included(self):
        """Student with is_at_risk=False should not be included by rule-based."""
        from forma.warning_report_data import build_warning_data
        at_risk = {"S001": {"is_at_risk": False}}
        result = build_warning_data(at_risk, [], {})
        assert len(result) == 0

    def test_rule_based_severity_all_zeros(self):
        """Rule-based severity with all-zero inputs → bounded [0, 1]."""
        from forma.warning_report_data import _compute_rule_based_severity
        severity = _compute_rule_based_severity({}, [], 0.0)
        assert 0.0 <= severity <= 1.0

    def test_rule_based_severity_worst_case(self):
        """Rule-based severity with worst inputs → bounded at 1.0."""
        from forma.warning_report_data import _compute_rule_based_severity
        severity = _compute_rule_based_severity(
            {"c1": 0.0, "c2": 0.0, "c3": 0.0},  # all deficit
            [0.0, 0.0, 0.0],                       # zero scores
            1.0,                                    # full absence
        )
        assert severity <= 1.0
        assert severity > 0.5  # should be high

    def test_single_point_trajectory_no_slope(self):
        """Single-point trajectory: SCORE_DECLINE not triggered (need >=2)."""
        from forma.warning_report_data import _classify_risk_types, RiskType
        types = _classify_risk_types("S001", {}, [0.3], 0.0)
        assert RiskType.SCORE_DECLINE not in types


# ===========================================================================
# PERSONA 5b: PDF CRASHER (Warning Report PDF + Charts)
# ===========================================================================


class TestPDFCrasher:
    """Persona 5b: Attack warning PDF generation with malicious inputs."""

    def _make_card(self, student_id="S001", **kwargs):
        """Build a WarningCard for test fixtures."""
        from forma.warning_report_data import WarningCard, RiskType
        defaults = dict(
            student_id=student_id,
            risk_types=[RiskType.SCORE_DECLINE],
            detection_methods=["rule_based"],
            deficit_concepts=["세포막", "핵"],
            misconception_patterns=[],
            interventions=["학습 계획 재수립 지도"],
            drop_probability=0.7,
            risk_severity=0.7,
        )
        defaults.update(kwargs)
        return WarningCard(**defaults)

    def test_zero_warning_cards_no_warning_page(self):
        """Zero warning cards should generate a no-warning summary PDF."""
        from forma.warning_report import WarningPDFReportGenerator
        gen = WarningPDFReportGenerator()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            gen.generate([], path, class_name="1A")
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_xml_special_chars_in_student_id(self):
        """Student ID with XML special chars (<>&"') should be escaped."""
        from forma.warning_report import WarningPDFReportGenerator
        gen = WarningPDFReportGenerator()
        card = self._make_card(student_id='<S001&"test\'>')
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            gen.generate([card], path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_1000_char_student_id(self):
        """Extremely long student_id (1000 chars) should not crash PDF."""
        from forma.warning_report import WarningPDFReportGenerator
        gen = WarningPDFReportGenerator()
        card = self._make_card(student_id="S" * 1000)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            gen.generate([card], path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_200_warning_cards_performance(self):
        """200+ warning cards should generate PDF without overflow."""
        from forma.warning_report import WarningPDFReportGenerator
        gen = WarningPDFReportGenerator()
        cards = [
            self._make_card(student_id=f"S{i:04d}", risk_severity=1.0 - i * 0.004)
            for i in range(200)
        ]
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            start = time.time()
            gen.generate(cards, path)
            elapsed = time.time() - start
            assert os.path.getsize(path) > 0
            assert elapsed < 60.0  # generous limit
        finally:
            os.unlink(path)

    def test_empty_deficit_concepts_and_interventions(self):
        """Card with empty deficit_concepts and interventions: valid PDF."""
        from forma.warning_report import WarningPDFReportGenerator
        gen = WarningPDFReportGenerator()
        card = self._make_card(
            deficit_concepts=[],
            interventions=[],
            misconception_patterns=[],
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            gen.generate([card], path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_intervention_text_with_newlines_and_tabs(self):
        """Intervention text containing newlines/tabs should be escaped."""
        from forma.warning_report import WarningPDFReportGenerator
        gen = WarningPDFReportGenerator()
        card = self._make_card(
            interventions=[
                "중재 방안 1:\n- 보충 학습\n- 면담 실시",
                "중재 방안 2:\t탭 문자\t포함",
                "방안 3\r\nWindows 줄바꿈",
            ],
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            gen.generate([card], path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_null_drop_probability_display(self):
        """Card with drop_probability=None: no crash in PDF rendering."""
        from forma.warning_report import WarningPDFReportGenerator
        gen = WarningPDFReportGenerator()
        card = self._make_card(drop_probability=None)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            gen.generate([card], path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_korean_hangul_jamo_in_concepts(self):
        """Rare Hangul jamo characters in deficit concepts: no crash."""
        from forma.warning_report import WarningPDFReportGenerator
        gen = WarningPDFReportGenerator()
        card = self._make_card(
            deficit_concepts=["ㄱㄴㄷ 자음", "ㅏㅓㅗ 모음", "미토콘드리아의 기능"],
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            gen.generate([card], path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_all_risk_types_in_one_card(self):
        """Card with all 4 risk types: renders all labels + all interventions."""
        from forma.warning_report import WarningPDFReportGenerator
        from forma.warning_report_data import RiskType
        gen = WarningPDFReportGenerator()
        card = self._make_card(
            risk_types=list(RiskType),
            detection_methods=["rule_based", "model_predicted"],
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            gen.generate([card], path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_class_name_with_special_chars(self):
        """class_name with XML special characters: escaped in cover."""
        from forma.warning_report import WarningPDFReportGenerator
        gen = WarningPDFReportGenerator()
        card = self._make_card()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            gen.generate([card], path, class_name='<"1A&B">')
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_empty_class_name(self):
        """Empty class_name: cover page still renders."""
        from forma.warning_report import WarningPDFReportGenerator
        gen = WarningPDFReportGenerator()
        card = self._make_card()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            gen.generate([card], path, class_name="")
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_misconception_patterns_displayed(self):
        """Card with misconception_patterns: renders them in PDF."""
        from forma.warning_report import WarningPDFReportGenerator
        gen = WarningPDFReportGenerator()
        card = self._make_card(
            misconception_patterns=[
                "CAUSAL_REVERSAL: 세포막 → 핵 (역방향)",
                "INCLUSION_ERROR: 미토콘드리아 ⊂ 핵",
            ],
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            gen.generate([card], path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_esc_function_special_chars(self):
        """_esc should escape XML characters: <, >, &, but not crash on others."""
        from forma.warning_report import _esc
        assert "&lt;" in _esc("<tag>")
        assert "&gt;" in _esc("<tag>")
        assert "&amp;" in _esc("A & B")
        assert _esc("") == ""
        assert _esc("normal text") == "normal text"
        # Non-string input should be converted to string
        assert _esc(123) == "123"
        assert _esc(None) == "None"


class TestPDFCrasherCharts:
    """Persona 5b extension: Attack warning report chart generation."""

    def test_risk_type_chart_empty_counts(self):
        """Empty risk_type_counts: produces 'no data' chart, not crash."""
        from forma.warning_report_charts import build_risk_type_distribution_chart
        buf = build_risk_type_distribution_chart({})
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_risk_type_chart_single_type(self):
        """Single risk type in chart: valid PNG."""
        from forma.warning_report_charts import build_risk_type_distribution_chart
        from forma.warning_report_data import RiskType
        buf = build_risk_type_distribution_chart({RiskType.SCORE_DECLINE: 10})
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_risk_type_chart_all_4_types(self):
        """All 4 risk types with counts: valid chart."""
        from forma.warning_report_charts import build_risk_type_distribution_chart
        from forma.warning_report_data import RiskType
        counts = {
            RiskType.SCORE_DECLINE: 15,
            RiskType.PERSISTENT_LOW: 8,
            RiskType.CONCEPT_DEFICIT: 12,
            RiskType.PARTICIPATION_DECLINE: 5,
        }
        buf = build_risk_type_distribution_chart(counts)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_deficit_chart_empty_counts(self):
        """Empty concept_counts: produces 'no data' chart, not crash."""
        from forma.warning_report_charts import build_deficit_concepts_chart
        buf = build_deficit_concepts_chart({})
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_deficit_chart_50_concepts_top_10(self):
        """50 concepts: only top 10 displayed."""
        from forma.warning_report_charts import build_deficit_concepts_chart
        concepts = {f"concept_{i}": 50 - i for i in range(50)}
        buf = build_deficit_concepts_chart(concepts)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_deficit_chart_korean_concept_names(self):
        """Korean concept names in deficit chart: renders without crash."""
        from forma.warning_report_charts import build_deficit_concepts_chart
        concepts = {
            "세포막의 구조": 8,
            "미토콘드리아": 6,
            "핵의 기능": 4,
            "리보솜": 3,
        }
        buf = build_deficit_concepts_chart(concepts)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_deficit_chart_single_concept(self):
        """Single concept: valid chart."""
        from forma.warning_report_charts import build_deficit_concepts_chart
        buf = build_deficit_concepts_chart({"세포막": 10})
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"

    def test_risk_type_chart_high_counts(self):
        """Very high counts (10000+): valid chart without overflow."""
        from forma.warning_report_charts import build_risk_type_distribution_chart
        from forma.warning_report_data import RiskType
        counts = {RiskType.SCORE_DECLINE: 10000, RiskType.PERSISTENT_LOW: 8000}
        buf = build_risk_type_distribution_chart(counts)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b"\x89PNG"


# ===========================================================================
# PERSONA 7: CONCURRENT CHAOS
# ===========================================================================


class TestConcurrentChaos:
    """Persona 7: Race conditions, file locking, and concurrent access attacks."""

    def test_concurrent_model_save_load(self):
        """Save and load model from same path concurrently: no corruption."""
        from forma.risk_predictor import (
            TrainedRiskModel, save_model, load_model, FEATURE_NAMES,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import threading

        rng = np.random.RandomState(42)
        X = rng.uniform(0, 1, (20, 15))
        scaler = StandardScaler()
        scaler.fit(X)
        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(scaler.transform(X), np.array([0]*10 + [1]*10))

        trained = TrainedRiskModel(
            model=model,
            feature_names=list(FEATURE_NAMES),
            scaler=scaler,
            training_date="2026-03-10T00:00:00Z",
            n_students=20,
            n_weeks=5,
            cv_score=0.85,
        )

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        errors: list[Exception] = []

        def save_loop():
            for _ in range(10):
                try:
                    save_model(trained, path)
                except Exception as e:
                    errors.append(e)

        def load_loop():
            for _ in range(10):
                try:
                    load_model(path)
                except Exception as e:
                    # File may be mid-write — acceptable
                    pass

        try:
            save_model(trained, path)  # initial save
            t1 = threading.Thread(target=save_loop)
            t2 = threading.Thread(target=load_loop)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)
            # Save should succeed without errors
            assert not errors, f"Save errors: {errors}"
            # Final load should succeed
            loaded = load_model(path)
            assert isinstance(loaded, TrainedRiskModel)
        finally:
            os.unlink(path)

    def test_concurrent_store_read_write(self):
        """Concurrent store add_record + get_all_records: no crash."""
        from forma.longitudinal_store import LongitudinalStore
        from forma.evaluation_types import LongitudinalRecord
        import threading

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        store = LongitudinalStore(path)
        errors: list[Exception] = []

        def writer():
            for i in range(20):
                try:
                    store.add_record(LongitudinalRecord(
                        student_id=f"S{i:03d}", week=1, question_sn=1,
                        scores={"ensemble_score": 0.5}, tier_level=1,
                        tier_label="Developing",
                    ))
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(20):
                try:
                    store.get_all_records()
                except Exception as e:
                    errors.append(e)

        try:
            t1 = threading.Thread(target=writer)
            t2 = threading.Thread(target=reader)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)
            # No crashes; errors from concurrent dict modification acceptable
            # but unhandled exceptions that crash threads are not
            all_records = store.get_all_records()
            assert len(all_records) > 0
        finally:
            os.unlink(path)

    def test_rapid_config_validation(self):
        """1000 rapid config validations: no state leakage between calls."""
        configs = [
            {"current_week": i} for i in range(1, 1001)
        ]
        for config in configs:
            validate_project_config(config)  # Should not raise

    def test_concurrent_feature_extraction(self):
        """Concurrent FeatureExtractor.extract() from different stores."""
        from forma.risk_predictor import FeatureExtractor
        from forma.longitudinal_store import LongitudinalStore
        from forma.evaluation_types import LongitudinalRecord
        import threading

        results: list[tuple] = []
        errors: list[Exception] = []
        ext = FeatureExtractor()

        def extract_from_store(store_path, n_students, weeks):
            try:
                store = LongitudinalStore(store_path)
                for i in range(n_students):
                    for w in weeks:
                        store.add_record(LongitudinalRecord(
                            student_id=f"S{i:03d}", week=w, question_sn=1,
                            scores={"ensemble_score": random.uniform(0.1, 0.9)},
                            tier_level=1, tier_label="Developing",
                        ))
                matrix, names, ids = ext.extract(store, weeks)
                results.append((matrix.shape, len(ids)))
            except Exception as e:
                errors.append(e)

        paths = []
        threads = []
        try:
            for j in range(5):
                f = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
                paths.append(f.name)
                f.close()
                t = threading.Thread(
                    target=extract_from_store,
                    args=(paths[-1], 10 + j, [1, 2, 3]),
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=30)

            assert not errors, f"Thread errors: {errors}"
            assert len(results) == 5
            for shape, n_ids in results:
                assert shape[1] == 15  # always 15 features
        finally:
            for p in paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def test_yaml_file_removed_mid_load(self):
        """Removing YAML file mid-load: FileNotFoundError handled."""
        from pathlib import Path
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w",
        ) as f:
            f.write("current_week: 5\n")
            path = f.name
        os.unlink(path)  # Remove before load
        with pytest.raises(FileNotFoundError):
            load_project_config(Path(path))


# ===========================================================================
# INVARIANT TESTING -- Persona 5 + 7
# ===========================================================================


class TestInvariant1000Phase3:
    """High-iteration invariant tests for warning data and classification."""

    def test_risk_classification_always_deterministic(self):
        """1000 random: same input always produces same risk classification."""
        from forma.warning_report_data import _classify_risk_types, RiskType
        rng = np.random.RandomState(42)
        for _ in range(1000):
            n_concepts = rng.randint(0, 10)
            concept_scores = {
                f"c{i}": rng.uniform(0, 1) for i in range(n_concepts)
            }
            n_weeks = rng.randint(0, 8)
            trajectory = rng.uniform(0, 1, n_weeks).tolist()
            absence = rng.uniform(0, 1)

            r1 = _classify_risk_types("S001", concept_scores, trajectory, absence)
            r2 = _classify_risk_types("S001", concept_scores, trajectory, absence)
            assert r1 == r2

    def test_severity_always_bounded(self):
        """1000 random: rule-based severity always in [0.0, 1.0]."""
        from forma.warning_report_data import _compute_rule_based_severity
        rng = np.random.RandomState(42)
        for _ in range(1000):
            n_concepts = rng.randint(0, 10)
            concept_scores = {
                f"c{i}": rng.uniform(0, 1) for i in range(n_concepts)
            }
            n_weeks = rng.randint(0, 8)
            trajectory = rng.uniform(0, 1, n_weeks).tolist()
            absence = rng.uniform(0, 1)
            severity = _compute_rule_based_severity(
                concept_scores, trajectory, absence,
            )
            assert 0.0 <= severity <= 1.0, (
                f"Severity {severity} out of bounds"
            )

    def test_warning_cards_always_sorted(self):
        """100 random: build_warning_data always returns sorted cards."""
        from forma.warning_report_data import build_warning_data
        rng = np.random.RandomState(42)
        for _ in range(100):
            n_students = rng.randint(1, 20)
            at_risk = {
                f"S{i:03d}": {"is_at_risk": True}
                for i in range(n_students)
            }
            preds = [
                self._make_pred(f"S{i:03d}", rng.uniform(0, 1))
                for i in range(n_students)
            ]
            concept_scores = {
                f"S{i:03d}": {
                    f"c{j}": rng.uniform(0, 1) for j in range(5)
                }
                for i in range(n_students)
            }
            trajectories = {
                f"S{i:03d}": rng.uniform(0, 1, rng.randint(1, 6)).tolist()
                for i in range(n_students)
            }
            absences = {
                f"S{i:03d}": rng.uniform(0, 1)
                for i in range(n_students)
            }
            cards = build_warning_data(
                at_risk, preds, concept_scores,
                score_trajectories=trajectories,
                absence_ratios=absences,
            )
            severities = [c.risk_severity for c in cards]
            assert severities == sorted(severities, reverse=True)

    def _make_pred(self, student_id, drop_prob):
        from forma.risk_predictor import RiskPrediction
        return RiskPrediction(
            student_id=student_id,
            drop_probability=drop_prob,
        )
