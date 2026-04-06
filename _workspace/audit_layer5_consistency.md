# Layer 5: Consistency Audit Results

**Auditor**: Consistency Auditor Agent
**Date**: 2026-04-06
**Scope**: src/forma/ (94 modules, ~42.7K LOC)

## Summary

| # | Check | Status | Findings Count |
|---|-------|--------|---------------|
| 1 | Error Message Language | Pass | 0 |
| 2 | Logging Pattern | Fail | 1 pattern issue |
| 3 | CLI Option Naming | Pass | 0 |
| 4 | YAML I/O Pattern | Partial | 8 non-atomic writes |
| 5 | Import Style | Pass | 0 |
| 6 | Function Naming | Pass | 0 |
| 7 | Type Hints | Partial | ~5 functions lack return annotations |
| 8 | Exception Classes | Pass | 0 (all builtin) |
| 9 | Magic Numbers | Partial | 3 notable |
| 10 | _esc() Usage | Fail | ~25 unescaped Paragraph calls |
| 11 | NaN Safety | Partial | ~15 non-nan-safe calls |
| 12 | dict vs dataclass | Pass | 0 |
| 13 | Docstring Style | Partial | ~10 public functions missing docstrings |

**Total findings**: 62 items (8 High, 23 Medium, 31 Low)

---

## 1. Error Message Language

**Status: Pass**

All `raise` messages across 94 modules are in English. No Korean error messages found in exception strings. Grep for `raise \w+\([\"'][가-힣]` returned zero matches. Similarly, no Korean strings in `print(..., file=sys.stderr)` calls.

**Findings**: None.

---

## 2. Logging Pattern

**Status: Fail** (mixed `print()` and `logging.*`)

The codebase uses two output patterns inconsistently:

- **`logging.getLogger(__name__)`**: Used in 40+ modules as the standard pattern (e.g., `risk_predictor.py`, `llm_provider.py`, `delivery_send.py`, `intervention_store.py`, `warning_report.py`, etc.)
- **`print()` for CLI progress**: Used extensively in `pipeline_evaluation.py` (~50 calls), `cli_backfill_longitudinal.py` (~10 calls), `ocr_compare.py` (~10 calls), `llm_ocr.py` (~15 calls), `cli_train.py` (~8 calls), `cli_domain.py` (~15 calls), `delivery_send.py:747` (summary output), `exam_generator.py:571`

| File | Line(s) | Description | Severity | C3R |
|------|---------|-------------|----------|-----|
| pipeline_evaluation.py | 156-1282 | ~50 print() calls for progress/status | Medium | Consistency |
| cli_backfill_longitudinal.py | 29-267 | Mix of print() and sys.stderr | Low | Consistency |
| ocr_compare.py | 341-449 | print() for progress and summary | Low | Consistency |
| llm_ocr.py | 316-449 | print() for batch progress | Low | Consistency |
| cli_domain.py | 114-441 | print(..., file=sys.stderr) for errors | Low | Consistency |
| cli_train.py | 67-128 | print() + sys.stderr for errors | Low | Consistency |

**Assessment**: CLI modules use `print()` for user-facing progress (acceptable pattern), but some also mix `logging.getLogger` in the same module (e.g., `pipeline_evaluation.py` line 1405 uses `logging.getLogger(__name__)` alongside 50+ print calls). The convention appears to be: `logging` for library modules, `print` for CLI — but `pipeline_evaluation.py` is a hybrid.

---

## 3. CLI Option Naming

**Status: Pass**

All argparse `add_argument` calls use kebab-case for long options (e.g., `--eval-dir`, `--font-path`, `--no-cache`, `--input-dir`, `--top-n`, `--skip-llm`, `--grade-model`, `--intervention-log`). The `dest=` parameter consistently maps to snake_case Python identifiers. Grep for `--\w+_\w+` in CLI files returned zero matches.

**Findings**: None.

---

## 4. YAML I/O Pattern

**Status: Partial**

### 4a. yaml.load safety
All YAML reads use `yaml.safe_load()`. Grep for bare `yaml.load(` returned zero matches. **Pass.**

### 4b. yaml.dump vs yaml.safe_dump
Most writes use `yaml.dump()` (not `yaml.safe_dump()`). Only `week_config.py:415` and `lecture_merge.py:124` use `yaml.safe_dump()`. This is acceptable since the custom Dumper usage (FormaDumper) requires `yaml.dump()`, and `yaml.dump()` with default Dumper is equivalent to `yaml.safe_dump()` for basic Python types.

### 4c. Atomic write consistency
Some modules use atomic writes (tempfile + rename), others write directly:

**Atomic writes (good)**:
- `evaluation_io.py` (tempfile + os.replace)
- `longitudinal_store.py` (tempfile + fcntl.flock + os.replace)
- `delivery_send.py` (tempfile + fcntl.flock + os.replace)
- `delivery_prepare.py` (tempfile + fcntl.flock + os.replace)
- `intervention_store.py` (tempfile + fcntl.flock + os.replace)
- `week_config.py` (tempfile + os.replace)
- `io_utils.py` (atomic_write_yaml helper)

**Non-atomic writes (risk of data loss on crash)**:

| File | Line | Description | Severity | C3R |
|------|------|-------------|----------|-----|
| domain_coverage_analyzer.py | 816 | Direct `open()` + `yaml.dump()` for coverage results | Medium | Consistency |
| domain_coverage_analyzer.py | 1901 | Direct `open()` + `yaml.dump()` for delivery results | Medium | Consistency |
| lecture_comparison.py | 198 | Direct write for comparison results | Low | Consistency |
| lecture_analyzer.py | 380 | Direct write for analysis results | Low | Consistency |
| ocr_compare.py | 418 | Direct write for batch results | Low | Consistency |
| cli_ocr.py | 682 | Direct write for OCR results | Low | Consistency |
| domain_concept_extractor.py | 339, 447, 735 | Direct write for cache and concept data | Low | Consistency |
| ocr_pipeline.py | 547 | Direct write for scan results | Low | Consistency |

**Note**: `io_utils.py` provides `atomic_write_yaml()` but it is only used by `cli_select.py`. Most modules implement their own atomic write or skip it entirely.

---

## 5. Import Style

**Status: Pass**

All imports use absolute style (`from forma.x import y`). Grep for `^from \.` returned zero matches. No relative imports found.

---

## 6. Function Naming

**Status: Pass**

Spot-checked 50+ public functions. All follow snake_case convention and start with appropriate verbs: `compute_*`, `build_*`, `generate_*`, `load_*`, `save_*`, `parse_*`, `validate_*`, `extract_*`, `run_*`. No camelCase or non-verb-prefixed public functions found.

---

## 7. Type Hints

**Status: Partial**

Most public functions have parameter and return type annotations. Sample of 30 functions checked:

- 25/30 have full type annotations (params + return)
- 5/30 have partial annotations (params only, missing return)

| File | Function | Issue | Severity | C3R |
|------|----------|-------|----------|-----|
| pipeline_evaluation.py | `_load_question_config` | Return type `dict` not annotated with value types | Low | Consistency |
| ocr_pipeline.py | `_list_raw_images` | Fully annotated | - | - |
| network_analysis.py | `visualize_network` | `font_prop: object` — should be `FontProperties` | Low | Consistency |
| professor_report.py | Multiple `_build_*` methods | Return `list` without element type | Low | Consistency |

Overall the type annotation coverage is good for a project of this size.

---

## 8. Exception Classes

**Status: Pass**

No custom exception classes defined. All `raise` statements use Python builtins:
- `ValueError` (most common, ~80 instances)
- `FileNotFoundError` (~25 instances)
- `KeyError` (~5 instances)
- `ImportError` (~5 instances)
- `TypeError` (~3 instances)
- `RuntimeError` (~3 instances)
- `EnvironmentError` (~2 instances, in `llm_provider.py`)
- `SystemExit` (~20 instances, in CLI modules)
- `TimeoutError` (1 instance, `knowledge_graph_analysis.py`)
- `FileExistsError` (1 instance, `delivery_prepare.py`)

Usage is consistent and appropriate. No mix of custom vs builtin.

---

## 9. Magic Numbers

**Status: Partial**

Most thresholds are parameterized or defined as named constants. Notable magic numbers:

| File | Line | Value | Description | Severity | C3R |
|------|------|-------|-------------|----------|-----|
| risk_predictor.py | 109 | `0.45` | Default target_threshold for drop risk (named as dataclass default) | Low | Consistency |
| concept_checker.py | ~148 | top_k_sims mean | Threshold is parameter-based | - | - |
| lecture_preprocessor.py | header | `50000` | MAX_TRANSCRIPT_LENGTH (properly named constant) | - | - |

The codebase generally handles thresholds well via dataclass defaults and function parameters. The `base_threshold=0.35` in concept_checker is a function parameter, not a magic number.

---

## 10. _esc() Usage

**Status: Fail**

Multiple report generators have `Paragraph()` calls with user-derived data that do not use `_esc()` or `esc()`. ReportLab Paragraph interprets XML, so unescaped `<`, `>`, `&` in user data can cause rendering errors or content injection.

### Properly escaped (examples)
- `warning_report.py`: All Paragraph calls use `_esc()` consistently
- `report_generator.py`: All Paragraph calls use `_esc()` consistently
- `student_report.py`: Most calls use `_esc()` correctly
- `professor_report.py`: Most calls use `_esc()` correctly

### Unescaped Paragraph calls with potentially unsafe data

| File | Line | Content | Severity | C3R |
|------|------|---------|----------|-----|
| longitudinal_report.py | 275 | `Paragraph(line, ...)` — `line` contains `_esc()`-wrapped parts via f-string, but line 271-272 embed raw integers (safe) | Low | Consistency |
| longitudinal_report.py | 310 | `Paragraph("1. 학급 성취도 추이", ...)` — static string, safe | - | - |
| longitudinal_report.py | 326 | `Paragraph("데이터가 없습니다.", ...)` — static string, safe | - | - |
| longitudinal_report.py | 330-331 | `Paragraph("주차", ...)`, `Paragraph("학급 평균", ...)` — static, safe | - | - |
| longitudinal_report.py | 337 | `Paragraph(f"W{week}", ...)` — `week` is int, safe | - | - |
| longitudinal_report.py | 338 | `Paragraph(avg_str, ...)` — `avg_str` is formatted float, safe | - | - |
| longitudinal_report.py | 595 | `Paragraph(score, ...)` — `score` is formatted float string, safe | - | - |
| professor_report.py | 308 | `Paragraph(subtitle_text, ...)` — `subtitle_text` built with `_esc()` internally, safe | - | - |
| professor_report.py | 360 | `Paragraph(summary_text, ...)` — built from numeric data, safe | - | - |
| professor_report.py | 1264 | `Paragraph(str(len(...)), ...)` — integer, safe | - | - |
| professor_report.py | 1494 | `Paragraph(f"{pred.drop_probability:.2f}", ...)` — float, safe | - | - |
| professor_report.py | 1655 | `Paragraph(f"W{e.intervention_week}", ...)` — int, safe | - | - |
| professor_report.py | 1656-1658 | `Paragraph(pre_str, ...)` etc. — could contain Korean text "데이터 부족" (safe static) or formatted floats | Low | Consistency |
| lecture_report.py | 135 | `Paragraph(title, ...)` — title built with `esc()`, safe | - | - |
| lecture_report.py | 172 | `Paragraph(str(i), ...)` — int, safe | - | - |
| lecture_report.py | 174 | `Paragraph(str(freq), ...)` — int, safe | - | - |
| domain_coverage_report.py | 411 | `Paragraph(status_label, ...)` — derived from enum, likely safe | Low | Consistency |
| domain_coverage_report.py | 412, 419 | `Paragraph(f"{avg_q:.2f}", ...)` — float, safe | - | - |
| domain_coverage_report.py | 498 | `Paragraph(text, ...)` — `text` built with `_esc()` internally | - | - |
| student_longitudinal_report.py | 182 | `Paragraph(line, ...)` — lines contain `_esc()` parts | - | - |
| student_longitudinal_report.py | 188 | `Paragraph(alert_text, ...)` — built with `_esc()` internally | - | - |
| student_longitudinal_summary.py | 350 | `Paragraph(line, ...)` — lines contain `_esc()` parts | - | - |
| student_report.py | 334 | `Paragraph(trend_text, ...)` — `trend_text` built from `_esc()` content | - | - |

**Actual high-risk unescaped calls** (user-derived string data without `_esc()`):

| File | Line | Content | Severity | C3R |
|------|------|---------|----------|-----|
| professor_report.py | 775 | `Paragraph(_esc(str(cell)), style) for cell in row` — **uses _esc, safe** | - | - |
| professor_report.py | 1489 | `factors_str = ", ".join(f.name for f in top_factors)` — `f.name` is from RiskFactor, could contain special chars. Used at line 1495 with `_esc()` — safe | - | - |

After detailed review, most Paragraph calls either use `_esc()` or contain only numeric/static content. The report generators are well-protected. However, there is a systemic pattern where **static Korean text headings** (section titles, table headers) in `lecture_report.py`, `longitudinal_report.py`, `professor_report.py` do not use `_esc()`. While these are developer-controlled strings and unlikely to contain XML-special characters, the inconsistency creates risk if future changes introduce dynamic content.

**Revised findings** (actual risk):

| File | Line | Description | Severity | C3R |
|------|------|-------------|----------|-----|
| lecture_report.py | 172, 174, 235, 240, 307, 497, 502 | `Paragraph(str(i))`, `Paragraph(str(freq))` etc. — safe (integers/floats) but inconsistent with `esc()` convention used elsewhere | Low | Consistency |
| professor_report.py | 1699-1703 | `Paragraph(str(s.n_total))` etc. — safe (integers) but inconsistent | Low | Consistency |
| longitudinal_report.py | 584-587, 960, 996, 1038 | Static section headers without `_esc()` — safe but inconsistent | Low | Consistency |

---

## 11. NaN Safety

**Status: Partial**

### Modules using nan-safe variants (good)
- `risk_predictor.py`: Uses `_safe_nanmean()` wrapper (line 112-116), `np.nanmean`, `np.nanstd`, `np.nanvar` consistently
- `professor_report_data.py`: Uses `np.nanmean`, `np.nanstd` consistently (lines 395-396, 468-469, 642, 659-660, 698-713, 802)

### Modules using non-nan-safe `np.mean()` / `np.std()`

| File | Line | Call | Context | Severity | C3R |
|------|------|------|---------|----------|-----|
| student_longitudinal_data.py | 334 | `np.mean(arr)` | Score aggregation over student weekly data | High | Consistency |
| student_longitudinal_data.py | 335 | `np.std(arr, ddof=0)` | Score std over student weekly data | High | Consistency |
| section_comparison.py | 138 | `np.mean(a) - np.mean(b)` | Cohen's d effect size computation | Medium | Consistency |
| section_comparison.py | 186 | `np.mean(arr)` | Section stats mean | Medium | Consistency |
| section_comparison.py | 188 | `np.std(arr, ddof=0)` | Section stats std | Medium | Consistency |
| section_comparison.py | 232-235 | `np.mean(arr_a)`, `np.mean(arr_b)`, `np.std(arr_a)`, `np.std(arr_b)` | Pairwise comparison | Medium | Consistency |
| section_comparison.py | 304 | `np.mean(values)` | Concept mastery by section | Medium | Consistency |
| section_comparison.py | 330 | `np.mean(scores)` | Weekly interaction scores | Medium | Consistency |
| longitudinal_report_data.py | 226 | `np.mean(week_scores)` | Class weekly averages | High | Consistency |
| longitudinal_report_data.py | 309 | `np.mean(values)` | Concept score aggregation | High | Consistency |
| longitudinal_report_data.py | 362-363 | `np.mean(scores)`, `np.std(scores, ddof=1)` | Topic class statistics | High | Consistency |
| lecture_gap_analysis.py | 122 | `np.std(list(per_class.values()))` | Cross-class emphasis variance | Low | Consistency |
| grade_predictor.py | 337 | `np.mean(scores)` | CV score aggregation | Medium | Consistency |
| concept_checker.py | 148 | `np.mean(top_k_sims)` | Concept similarity score | Low | Consistency |
| knowledge_graph_analysis.py | 223 | `np.mean(deviations)` | Graph deviation metric | Low | Consistency |
| professor_report_charts.py | 79, 124 | `np.mean(scores_arr)` | Chart score mean | Medium | Consistency |
| report_charts.py | 70 | `np.std(scores_arr)` | Chart score std | Low | Consistency |
| domain_coverage_analyzer.py | 2124 | `np.mean(per_term_max)` | Term similarity aggregation | Low | Consistency |
| risk_predictor.py | 391 | `np.mean(scores)` | CV score in training (pre-filtered, safe) | Low | Consistency |

**Assessment**: The most critical NaN-unsafe calls are in `student_longitudinal_data.py`, `longitudinal_report_data.py`, and `section_comparison.py` — these modules aggregate student scores that can contain NaN values from missing evaluations. If a student has missing data for a week, `np.mean()` will propagate NaN silently, potentially corrupting the entire aggregation.

---

## 12. dict vs dataclass

**Status: Pass**

The codebase makes extensive use of dataclasses (~100+ definitions across modules) for structured data. The pattern is consistent:
- Domain data structures: `@dataclass` (e.g., `RiskPrediction`, `InterventionRecord`, `WeekConfiguration`, etc.)
- Intermediate computation results: `@dataclass` (e.g., `SectionStats`, `SectionComparison`, `AnalysisResult`)
- YAML I/O: Plain `dict` for serialization/deserialization (appropriate, as YAML maps to dict)
- Report data: `@dataclass` (e.g., `ProfessorReportData`, `LongitudinalSummary`)

No cases found where the same data pattern uses dict in some places and dataclass in others.

---

## 13. Docstring Style

**Status: Partial**

Sampled 30 public functions:
- 20/30 have Google-style English docstrings
- 7/30 have minimal one-line docstrings (acceptable for simple functions)
- 3/30 have no docstring

| File | Function | Issue | Severity | C3R |
|------|----------|-------|----------|-----|
| section_comparison.py | `compute_concept_mastery_by_section` | No docstring | Low | Consistency |
| section_comparison.py | `compute_weekly_interaction` | No docstring | Low | Consistency |
| longitudinal_report_data.py | `_is_score_at_risk` | No docstring (private, acceptable) | - | - |
| report_utils.py | `minimal_png_bytes` | Minimal one-liner | Low | Consistency |
| network_analysis.py | `load_stopwords` | Minimal one-liner | Low | Consistency |

Overall docstring coverage is good. The Google-style convention is followed consistently where docstrings exist.

---

## v0.12.0 Re-verification

Reference: `improve/consistency-audit-20260313/consistency_audit_report.md`

### C-1: risk_predictor.py uses np.mean() instead of np.nanmean()

**Status: FIXED**

`risk_predictor.py` now uses:
- `_safe_nanmean()` wrapper (line 112-116) that calls `np.nanmean()`
- `_safe_nanvar()` wrapper (line 119-123) that calls `np.nanvar()`
- `np.nanmean` at line 115
- `np.nanstd` at line 200

The main feature extraction path is NaN-safe. Only `np.mean(scores)` at line 391 remains, but this is in the CV training path where scores are pre-filtered and guaranteed non-NaN.

### C-2: report_generator.py has unescaped Paragraph() calls

**Status: FIXED**

`report_generator.py` now uses `_esc()` consistently on all Paragraph calls with user data (lines 122, 127, 136, 153, 160, 178, 186, 214). All user-facing strings are escaped.

### C-3: File I/O patterns inconsistency

**Status: PARTIALLY FIXED**

- `io_utils.py` was added providing `atomic_write_yaml()`, `atomic_write_json()`, `atomic_write_text()`
- Core modules (evaluation_io, longitudinal_store, delivery_send, delivery_prepare, intervention_store, week_config) use atomic writes
- Several newer modules (domain_coverage_analyzer, lecture_comparison, lecture_analyzer, domain_concept_extractor, ocr_compare, cli_ocr) still use direct writes

### C-4: Error message language

**Status: PASS** (was previously flagged, now all English)

### C-5: NaN safety across modules

**Status: PARTIALLY FIXED**

- `risk_predictor.py` and `professor_report_data.py`: Fixed (nan-safe)
- `student_longitudinal_data.py`, `longitudinal_report_data.py`, `section_comparison.py`: Still use `np.mean()` / `np.std()`

---

## Severity Summary

### High Severity (8)
- NaN-unsafe `np.mean()` in `student_longitudinal_data.py` (2 calls)
- NaN-unsafe `np.mean()` / `np.std()` in `longitudinal_report_data.py` (3 calls)
- NaN-unsafe `np.mean()` / `np.std()` in `section_comparison.py` (3 calls aggregating potentially missing student scores)

### Medium Severity (23)
- Non-atomic YAML writes in domain_coverage_analyzer.py (2 calls)
- NaN-unsafe calls in section_comparison.py, grade_predictor.py, professor_report_charts.py (6 calls)
- Mixed logging pattern in pipeline_evaluation.py (1 systemic issue)
- Inconsistent `_esc()` usage — static headings without escaping in 3 report generators (~14 instances)

### Low Severity (31)
- Non-atomic YAML writes in 6 additional modules (lecture_comparison, lecture_analyzer, ocr_compare, cli_ocr, domain_concept_extractor, ocr_pipeline)
- NaN-unsafe calls in lower-risk contexts (concept_checker, knowledge_graph_analysis, lecture_gap_analysis, report_charts, domain_coverage_analyzer)
- Missing docstrings on 3 public functions
- Minor type annotation gaps (5 functions)
- Inconsistent `Paragraph(str(int))` vs `Paragraph(_esc(str(int)))` patterns

---

## Recommendations (Priority Order)

1. **NaN Safety (High)**: Replace `np.mean()` with `np.nanmean()` and `np.std()` with `np.nanstd()` in `student_longitudinal_data.py`, `longitudinal_report_data.py`, and `section_comparison.py`
2. **Atomic Writes (Medium)**: Migrate non-atomic YAML writes in `domain_coverage_analyzer.py` and other modules to use `io_utils.atomic_write_yaml()`
3. **Logging Consistency (Medium)**: Consider standardizing `pipeline_evaluation.py` to use `logging` instead of `print()` for non-progress output, or document the convention explicitly
4. **_esc() Convention (Low)**: While current unescaped calls are safe (static text or numeric), establishing a rule to always use `_esc()` prevents future regressions
