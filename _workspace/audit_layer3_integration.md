# Layer 3: Cross-Module Integration Results

## Test Execution Summary

| Path/Boundary | Tests | Pass | Fail | Notes |
|---|---|---|---|---|
| Path B: Evaluation Pipeline | 2 | 2 | 0 | concept_checker -> ensemble_scorer works |
| Path C: Report Pipeline | 2 | 2 | 0 | eval YAML round-trip + Rasch integration OK |
| Path D: Longitudinal Pipeline | 3 | 2 | 1 | NaN propagation bug confirmed |
| Path E: Risk/Warning Pipeline | 3 | 3 | 0 | Feature extraction + warning cards OK |
| Path G: Lecture Analysis | 2 | 2 | 0 | Preprocess -> analysis chain OK |
| Path H: Delivery Pipeline | 2 | 2 | 0 | Prepare -> send template rendering OK |
| Boundary #1: Empty exam | 1 | 1 | 0 | 0-question config handled gracefully |
| Boundary #2: Zero students | 2 | 2 | 0 | Empty input produces empty valid output |
| Boundary #4: LLM total failure | 1 | 1 | 0 | Layer 1 results usable without LLM |
| Boundary #5: NaN propagation | 2 | 0 | 2 | **NaN bugs in section_comparison + student_longitudinal** |
| Boundary #7: Large class (200) | 2 | 2 | 0 | No memory/performance issues |
| Boundary #8: Half-written YAML | 3 | 3 | 0 | Corrupt/empty YAML handled |
| Boundary #10: Missing forma.json | 1 | 1 | 0 | FileNotFoundError raised |
| Boundary #11: Model version mismatch | 1 | 1 | 0 | Stale pkl loads without crash |
| Boundary #12: Concurrent writes | 1 | 1 | 0 | Last-writer-wins (data loss possible) |
| Cross-module: TopicTrendResult | 1 | 1 | 0 | Duplicate dataclass confirmed |
| Cross-module: Student Longitudinal | 1 | 1 | 0 | Store -> data -> warnings chain OK |
| Cross-module: Delivery Chain | 1 | 1 | 0 | Roster -> template rendering OK |
| Cross-module: Section Comparison | 1 | 1 | 0 | Store data -> comparison stats OK |
| **TOTAL** | **32** | **29** | **3** | |

## Data Flow Path Results

### Path A: OCR Pipeline
Not directly tested (requires Naver OCR / LLM OCR external services). Would need full mock infrastructure.

### Path B: Evaluation Pipeline
- **concept_checker -> ensemble_scorer**: PASS. Concept check results feed correctly into `EnsembleScorer.compute_score()` with all optional layers set to None.
- **Empty student text**: PASS. `check_all_concepts()` raises `ValueError` on empty text (fail-fast). Pipeline callers must catch this before ensemble scoring.

### Path C: Report Pipeline
- **Evaluation YAML round-trip**: PASS. `save_evaluation_yaml()` -> `load_evaluation_yaml()` preserves all data.
- **Concept results -> Rasch IRT**: PASS. Binary matrix from concept results feeds into `RaschAnalyzer.fit()` and `ability_estimates()`.

### Path D: Longitudinal Pipeline
- **Store -> summary data**: PASS. 3 students x 3 weeks loads correctly into `build_longitudinal_summary()`.
- **NaN propagation**: **FAIL**. NaN ensemble_score in store propagates to `class_weekly_averages` via `np.mean()` (does not filter NaN).
- **Cross-week concept mismatch**: PASS. Different concepts per week handled correctly with union semantics.

### Path E: Risk/Warning Pipeline
- **Store -> feature extraction**: PASS. 3 students x 4 weeks produces correct (3, 15) feature matrix.
- **Warning card construction**: PASS. Union of rule-based + model-predicted at-risk students.
- **NaN through feature extractor**: PASS. `FeatureExtractor` uses `_safe_nanmean()` which handles NaN correctly.

### Path F: Domain Analysis
Not directly tested (requires textbook + concept list fixtures). Covered partially by existing unit tests.

### Path G: Lecture Analysis
- **Preprocess -> analysis**: PASS. Korean transcript preprocessed correctly via 8-step pipeline.
- **Mixed encoding**: PASS. Korean+English+special characters preserved through preprocessing.

### Path H: Delivery Pipeline
- **File matching**: PASS. `match_files_for_student()` finds PDF files by student ID pattern.
- **Template rendering**: PASS. `render_template()` correctly substitutes student name, ID, class into template.

## Boundary Condition Results

### Boundary #1: Empty exam (0 questions)
PASS. `_run_layer1()` returns empty dict for 0-question config.

### Boundary #2: Zero students
PASS. Both concept checker and longitudinal summary handle 0 students gracefully.

### Boundary #3: OCR total failure
Not tested (requires external OCR mock infrastructure).

### Boundary #4: LLM total failure
PASS. Ensemble scorer produces valid results from Layer 1 alone (llm_result=None).

### Boundary #5: NaN score propagation
**FAIL (3 locations)**:
1. `longitudinal_report_data.build_longitudinal_summary()` line ~226: `np.mean(week_scores)` does not filter NaN, propagates NaN to `class_weekly_averages`.
2. `section_comparison.compute_section_stats()` line ~187: `np.mean(arr)` propagates NaN to `SectionStats.mean`.
3. `student_longitudinal_data.build_student_data()`: NaN ensemble_score propagates to `trend_slope` via OLS without NaN filtering.

### Boundary #6: Cross-week concept mismatch
PASS. `_compute_concept_mastery_changes()` uses union of all concepts across weeks, defaults missing to 0.0.

### Boundary #7: Large class (200 students)
PASS. Feature extraction (200, 15) matrix in ~0.1s. Longitudinal summary 200 students in ~0.5s.

### Boundary #8: Half-written YAML
PASS. Truncated YAML raises `KeyError` on missing fields. Empty YAML returns empty records.

### Boundary #9: Mixed encoding input
PASS. Korean+English+special characters survive preprocessing.

### Boundary #10: Missing forma.json
PASS. `load_config()` with nonexistent path raises `FileNotFoundError`.

### Boundary #11: Model version mismatch
PASS. Old pkl dict loads via joblib without crash (attributes may be wrong but no exception).

### Boundary #12: Concurrent YAML writes
PASS (with caveat). File locking prevents corruption, but last-writer-wins semantics means data from the first writer is lost. No merge strategy exists.

## Discovered Integration Issues

| # | Issue | Modules Involved | Severity | Description |
|---|---|---|---|---|
| 1 | NaN propagation in class averages | `longitudinal_report_data.py` line ~226 | HIGH | `np.mean(week_scores)` does not use `np.nanmean()`. When any student has NaN ensemble_score, the entire week's class average becomes NaN. |
| 2 | NaN propagation in section stats | `section_comparison.py` line ~187 | HIGH | `np.mean(arr)` / `np.std(arr)` do not filter NaN. One NaN score corrupts section-level mean/std/median. |
| 3 | NaN propagation in student trend | `student_longitudinal_data.py` `build_student_data()` | HIGH | NaN ensemble_score is included in OLS slope computation, producing NaN `trend_slope`. No NaN filtering before `np.polyfit`. |
| 4 | Duplicate TopicTrendResult dataclass | `longitudinal_report_data.py` vs `student_longitudinal_data.py` | CRITICAL | Two `TopicTrendResult` classes with incompatible field types: `longitudinal_report_data` uses `float` for `kendall_tau`/`spearman_rho`, `student_longitudinal_data` uses `float | None`. Passing objects between modules will cause type confusion. |
| 5 | Concurrent write data loss | `longitudinal_store.py` | MEDIUM | File locking prevents corruption but last-writer-wins semantics causes silent data loss. No read-before-write merge strategy. |
| 6 | concept_checker rejects empty text | `concept_checker.py` -> `pipeline_evaluation.py` | LOW | `check_all_concepts()` raises `ValueError` on empty student_text. Pipeline must catch this to produce partial results for students with empty responses. |

## Severity Summary

| Severity | Count |
|---|---|
| CRITICAL | 1 (duplicate TopicTrendResult) |
| HIGH | 3 (NaN propagation in 3 modules) |
| MEDIUM | 1 (concurrent write data loss) |
| LOW | 1 (empty text error propagation) |

## Key Observations

1. **NaN handling is the dominant cross-module issue**. Three independent modules use `np.mean()`/`np.std()` instead of `np.nanmean()`/`np.nanstd()`. This creates a chain reaction: one NaN score from OCR or evaluation corrupts class-level statistics, trend analysis, and section comparisons.

2. **The FeatureExtractor in risk_predictor.py correctly uses `_safe_nanmean()`** — it is the only module that handles NaN properly. Other modules should follow this pattern.

3. **The TopicTrendResult duplication** is not just a naming conflict — the two classes have semantically different contracts (nullable vs. required fields). Code that passes `student_longitudinal_data.TopicTrendResult` to a function expecting `longitudinal_report_data.TopicTrendResult` will silently work until a `None` value hits a `float` operation.

4. **200-student scale** shows no performance or memory issues. The architecture handles large classes well.

5. **YAML corruption handling** is adequate — partial files raise clear errors, empty files produce empty stores.
