# Layer 1: Static Analysis Results

**Date:** 2026-04-06
**Scope:** `src/forma/` (94 modules) + `tests/` (formative-analysis v0.12.6)

## Summary

| Check | Status | Findings |
|-------|--------|----------|
| Ruff Lint | PASS | 0 violations |
| Ruff Format | FAIL | 226 files need reformatting |
| Type Checking (mypy) | SKIPPED | mypy not installed in project |
| Dead Code | WARN | 1 confirmed dead function, 1 questionable |
| Circular Imports | PASS | 0 cycles detected |
| Security Patterns | PASS | 0 issues found |

## 1. Ruff Lint

```
$ uv run ruff check src/ tests/
All checks passed!
```

No lint violations. All unused-import (F401) and unused-variable (F841) checks also pass.

## 2. Ruff Format

226 of 247 files would be reformatted. This is a project-wide formatting drift — every module in `src/forma/` and almost every test file is affected.

**Affected src/ files (92 files):**
- `src/forma/class_knowledge_aggregate.py`
- `src/forma/cli.py`
- `src/forma/cli_backfill_longitudinal.py`
- `src/forma/cli_deliver.py`
- `src/forma/cli_domain.py`
- `src/forma/cli_intervention.py`
- `src/forma/cli_lecture.py`
- `src/forma/cli_main.py`
- `src/forma/cli_ocr.py`
- `src/forma/cli_report.py`
- `src/forma/cli_report_batch.py`
- `src/forma/cli_report_longitudinal.py`
- `src/forma/cli_report_professor.py`
- `src/forma/cli_report_student.py`
- `src/forma/cli_report_warning.py`
- `src/forma/cli_train.py`
- `src/forma/cli_train_grade.py`
- `src/forma/concept_checker.py`
- `src/forma/concept_dependency.py`
- `src/forma/concept_network.py`
- `src/forma/config.py`
- `src/forma/config_validator.py`
- `src/forma/delivery_prepare.py`
- `src/forma/delivery_send.py`
- `src/forma/domain_concept_extractor.py`
- `src/forma/domain_coverage_analyzer.py`
- `src/forma/domain_coverage_charts.py`
- `src/forma/domain_coverage_report.py`
- `src/forma/domain_pedagogy_analyzer.py`
- `src/forma/embedding_cache.py`
- `src/forma/emphasis_map.py`
- `src/forma/ensemble_scorer.py`
- `src/forma/evaluation_io.py`
- `src/forma/evaluation_types.py`
- `src/forma/exam_generator.py`
- `src/forma/feedback_generator.py`
- `src/forma/font_utils.py`
- `src/forma/google_sheets.py`
- `src/forma/grade_predictor.py`
- `src/forma/graph_comparator.py`
- `src/forma/graph_visualizer.py`
- `src/forma/hub_gap.py`
- `src/forma/intervention_effect.py`
- `src/forma/intervention_store.py`
- `src/forma/io_utils.py`
- `src/forma/knowledge_graph_analysis.py`
- `src/forma/learning_path_charts.py`
- `src/forma/lecture_analyzer.py`
- `src/forma/lecture_comparison.py`
- `src/forma/lecture_gap_analysis.py`
- `src/forma/lecture_merge.py`
- `src/forma/lecture_preprocessor.py`
- `src/forma/lecture_processor.py`
- `src/forma/lecture_report.py`
- `src/forma/llm_evaluator.py`
- `src/forma/llm_ocr.py`
- `src/forma/llm_provider.py`
- `src/forma/longitudinal_report.py`
- `src/forma/longitudinal_report_charts.py`
- `src/forma/longitudinal_report_data.py`
- `src/forma/longitudinal_store.py`
- `src/forma/misconception_classifier.py`
- `src/forma/misconception_clustering.py`
- `src/forma/naver_ocr.py`
- `src/forma/network_analysis.py`
- `src/forma/ocr_compare.py`
- `src/forma/ocr_pipeline.py`
- `src/forma/pipeline_batch_evaluation.py`
- `src/forma/pipeline_evaluation.py`
- `src/forma/preprocess_imgs.py`
- `src/forma/professor_report.py`
- `src/forma/professor_report_charts.py`
- `src/forma/professor_report_data.py`
- `src/forma/professor_report_llm.py`
- `src/forma/project_config.py`
- `src/forma/prompt_templates.py`
- `src/forma/qr_decode.py`
- `src/forma/report_charts.py`
- `src/forma/report_data_loader.py`
- `src/forma/report_generator.py`
- `src/forma/report_utils.py`
- `src/forma/response_converter.py`
- `src/forma/risk_predictor.py`
- `src/forma/section_comparison.py`
- `src/forma/section_comparison_charts.py`
- `src/forma/statistical_analysis.py`
- `src/forma/student_longitudinal_charts.py`
- `src/forma/student_longitudinal_data.py`
- `src/forma/student_longitudinal_report.py`
- `src/forma/student_longitudinal_summary.py`
- `src/forma/student_report.py`
- `src/forma/topic_analysis.py`
- `src/forma/triplet_extractor.py`
- `src/forma/warning_report.py`
- `src/forma/warning_report_charts.py`
- `src/forma/warning_report_data.py`
- `src/forma/week_config.py`

**Affected tests/ files (134 files):** All test files except 21 are affected.

| Severity | File | Description | C3R Category |
|----------|------|-------------|--------------|
| Low | 226 files | Formatting drift from ruff standard style | Consistency |

## 3. Type Checking (mypy)

**Status: SKIPPED** — mypy is not installed in the project's dev dependencies.

```
$ uv run mypy src/forma/ --ignore-missing-imports
error: Failed to spawn: `mypy`
```

| Severity | Description | C3R Category |
|----------|-------------|--------------|
| Medium | No static type checking configured for the project | Correctness |

**Recommendation:** Add `mypy` to dev dependencies and configure `pyproject.toml` with basic mypy settings.

## 4. Dead Code

### 4.1 Confirmed Dead Function

| Severity | File | Line | Description | C3R Category |
|----------|------|------|-------------|--------------|
| Low | `src/forma/ocr_compare.py` | 172 | `run_comparison()` — defined but never called; only referenced in module docstring. Callers use `compare_single_image()` and `run_batch_comparison()` instead | Cleanliness |

### 4.2 Questionable (Low Confidence)

| Severity | File | Line | Description | C3R Category |
|----------|------|------|-------------|--------------|
| Low | `src/forma/mecab_shim.py` | 23 | `install()` — public function; module is imported at package init as side-effect (`__init__.py` line 1). May be called at import time or intended as manual utility | Cleanliness |

### 4.3 No Unused Imports or Variables

Ruff F401 (unused imports) and F841 (unused variables) checks both pass with zero violations.

## 5. Circular Imports

AST-based import graph analysis across all 94 modules in `src/forma/` found **no circular import cycles**.

## 6. Security Patterns

All manual security checks passed with no findings:

| Pattern | Result |
|---------|--------|
| `yaml.load()` without SafeLoader | **None found** — all YAML loading uses safe methods |
| `eval()` usage | **None found** — only occurrence is a comment in `cli_main.py:213` |
| `exec()` usage | **None found** |
| `subprocess` with `shell=True` | **None found** |
| Hardcoded secrets/passwords/API keys | **None found** |
| Bandit scan | **SKIPPED** — bandit not installed |

| Severity | Description | C3R Category |
|----------|-------------|--------------|
| Low | Neither bandit nor mypy are in dev dependencies, reducing automated safety coverage | Robustness |

## Severity Summary

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 0 |
| Medium | 1 (no type checking configured) |
| Low | 4 (226 formatting drift files, 1 dead function, 1 questionable dead function, missing dev tools) |

## Overall Assessment

The codebase is in **good static health**. Ruff lint passes cleanly with zero violations, there are no security anti-patterns, no circular imports, and no unused imports or variables. The two actionable items are:

1. **Formatting drift** (226 files) — a single `uv run ruff format src/ tests/` would resolve this
2. **Missing dev tooling** — adding `mypy` and `bandit` to dev dependencies would strengthen the static analysis safety net
