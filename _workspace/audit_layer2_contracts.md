# Layer 2: Module Contract Audit Results

## Summary

| Module Group | Modules Checked | Findings | Severity |
|---|---|---|---|
| Pipeline (P1) | 4 | 6 | 2 High, 3 Medium, 1 Low |
| Store (P1) | 3 | 4 | 1 High, 2 Medium, 1 Low |
| Data Loaders (P1) | 3 | 5 | 1 Critical, 2 High, 2 Medium |
| Reports (P2) | 6 | 4 | 1 High, 2 Medium, 1 Low |
| CLI (P2) | 4 | 2 | 2 Medium |
| Domain (P3) | 2 | 3 | 1 High, 1 Medium, 1 Low |
| Lecture (P3) | 3 | 1 | 1 Low |
| Predictors (P3) | 2 | 2 | 1 Medium, 1 Low |
| OCR (P4) | 3 | 1 | 1 Low |
| Delivery (P4) | 2 | 1 | 1 Low |
| Utils (P4) | 4 | 3 | 1 Medium, 2 Low |
| **Total** | **36** | **32** | **1 Critical, 7 High, 13 Medium, 11 Low** |

---

## Priority 1: Data Flow Core

### pipeline_evaluation.py

- **Public API**: `run_evaluation_pipeline()`, plus internal helpers `_run_layer1`, `_run_layer2_v1`, `_run_triplet_extraction`, `_run_graph_comparison`, `_run_feedback`, `_build_counseling_summary`, `_build_technical_report`, `_serialize_*`, `_load_question_config`, `_is_v2_question`, `_get_master_edges`, `_get_master_nodes`, `_get_rubric_tiers`, CLI `main()`
- **Unused APIs**: None significant. All internal helpers are used within the module.
- **Exception contract issues**: None. `_load_question_config` raises `KeyError` — only called internally with validated data.
- **Return type issues**:
  - **[P1-PE-1]** `run_evaluation_pipeline()` returns `None` — all pipeline output is via side-effect (YAML writes). This means callers cannot programmatically access evaluation results without re-reading YAML files. **Severity: Medium, C3R: Robustness**. `pipeline_evaluation.py:675`.
- **dict vs typed data**:
  - **[P1-PE-2]** `_build_counseling_summary()` and `_build_technical_report()` return raw `dict` instead of typed structures. These dicts are consumed by `report_generator.py`, `cli_report.py`, and YAML serialization. No schema validation on the dict structure. **Severity: High, C3R: Robustness**. `pipeline_evaluation.py:389,438`.
  - **[P1-PE-3]** `layer1_results` type is `dict[str, dict[int, list]]` — the inner `list` is untyped. Should be `list[ConceptMatchResult]`. **Severity: Low, C3R: Correctness**. `pipeline_evaluation.py:137`.

### pipeline_batch_evaluation.py

- **Public API**: `run_batch_evaluation()`, `main()`
- **Unused APIs**: None.
- **Exception contract issues**:
  - **[P1-PBE-1]** `_generate_class_reports()` catches all exceptions with bare `except Exception` and swallows them with a print. PDF generation failures are silently ignored. **Severity: Medium, C3R: Robustness**. `pipeline_batch_evaluation.py:159`.
- **Return type issues**: `run_batch_evaluation()` returns `None` — consistent with `run_evaluation_pipeline()`.

### ensemble_scorer.py

- **Public API**: `EnsembleScorer`, `classify_understanding_level()`, `normalize_score()`, constants `DEFAULT_WEIGHTS`, `WEIGHTS_V1`, `WEIGHTS_V2`, `UNDERSTANDING_THRESHOLDS`
- **Unused APIs**:
  - **[P1-ES-1]** `WEIGHTS_V2` is defined but never imported by any production module. Only `WEIGHTS_V1` alias is used (which equals `DEFAULT_WEIGHTS`). The v2 weight selection is hardcoded inside `compute_score()` using fallback defaults, not using `WEIGHTS_V2` dict. **Severity: Medium, C3R: Correctness**. `ensemble_scorer.py:45`.
- **Exception contract issues**: `EnsembleScorer.__init__` raises `ValueError` if weights don't sum to 1.0 — callers use defaults so this is unlikely to fire in practice.
- **Return type issues**: Consistent — always returns `EnsembleResult` dataclass. Good.

### evaluation_types.py

- **Public API**: 13 dataclasses: `ConceptMatchResult`, `LLMJudgeResult`, `AggregatedLLMResult`, `StatisticalResult`, `GraphMetricResult`, `EnsembleResult`, `TripletEdge`, `TripletExtractionResult`, `GraphComparisonResult`, `HubGapEntry`, `FeedbackResult`, `LongitudinalRecord`, `RubricTier`, `FailedCall`
- **Unused APIs**: All dataclasses are imported and used by at least one production module.
- **Return type issues**: None. All fields have explicit types.
- **Optional field propagation**:
  - **[P1-ET-1]** `StatisticalResult.rasch_theta` is `Optional[float]` — ensemble_scorer correctly handles this with `if ra is not None` check. Good.
  - `LongitudinalRecord` has 8 Optional fields — `longitudinal_store.py` correctly checks `is not None` before storing. Good.

### longitudinal_store.py

- **Public API**: `LongitudinalStore` class (load/save/add_record/get_student_history/get_all_records/get_class_snapshot/get_student_trajectory/get_class_weekly_matrix/get_topic_weekly_matrix), `snapshot_from_evaluation()`, `_compute_concept_scores()`, `_infer_class_id()`, `_record_key()`
- **Internal `_` APIs called from outside**:
  - **[P1-LS-1]** `_compute_concept_scores()` is imported and called from `test_adversary_v080.py`. `_record_key()` is imported in `test_adversary_v080.py`. These are test-only usages — acceptable but indicates the functions might benefit from being public. **Severity: Low, C3R: Correctness**.
- **dict vs typed data**:
  - **[P1-LS-2]** Internal storage uses raw `dict` (`self._records: dict[str, dict]`), but the public API consistently converts to `LongitudinalRecord` dataclass via `_to_record()`. **Severity: Low (acceptable design — internal optimization)**.
- **Exception contract issues**:
  - **[P1-LS-3]** `snapshot_from_evaluation()` has parameter `ensemble_results: dict` — the expected shape `{student_id: {qsn: EnsembleResult}}` is only documented in the docstring, not enforced by type. If a caller passes malformed data, `AttributeError` will propagate with no useful message. **Severity: Medium, C3R: Robustness**. `longitudinal_store.py:309`.

### intervention_store.py

- **Public API**: `InterventionRecord`, `InterventionLog` (load/save/add_record/get_records/update_outcome), `INTERVENTION_TYPES`
- **Unused APIs**: None — all used by `cli_intervention.py` and `intervention_effect.py`.
- **Exception contract issues**: `add_record()` validates `intervention_type`, `student_id`, `week`, `description`. Good fail-fast behavior.
- **Return type issues**: `update_outcome()` returns `bool` — callers in `cli_intervention.py` check the return value. Good.
- **dict vs typed data**:
  - **[P1-IS-1]** Same pattern as LongitudinalStore — internal `list[dict]` storage but public API returns `InterventionRecord` dataclass. Acceptable. **Severity: Low**.

### evaluation_io.py

- **Public API**: `FormaDumper`, `load_evaluation_yaml()`, `save_evaluation_yaml()`, `extract_student_responses()`
- **Unused APIs**: None.
- **Exception contract issues**:
  - `load_evaluation_yaml()` raises `FileNotFoundError` — properly documented and caught by callers.
  - `extract_student_responses()` raises `ValueError`/`KeyError` — properly documented.
- **Return type issues**: `load_evaluation_yaml()` returns `dict[str, Any]` — raw dict return. Callers must know expected shape.
  - **[P1-EIO-1]** `save_evaluation_yaml()` backup logic: if `os.replace()` for backup fails, the backup is silently lost (bare `except OSError: pass`). Original data is not at risk but backup guarantee is weakened. **Severity: Medium, C3R: Robustness**. `evaluation_io.py:106`.

### report_data_loader.py

- **Public API**: `ConceptDetail`, `QuestionReportData`, `StudentReportData`, `ClassDistributions`, `WeeklyDelta`, `load_all_student_data()`, `compute_class_distributions()`, `compute_weekly_delta()`
- **Unused APIs**: None — heavily used across report modules and CLIs.
- **Return type issues**: `load_all_student_data()` returns `tuple[list[StudentReportData], ClassDistributions]`. Good typed return.
- **Exception contract issues**:
  - **[P1-RDL-1]** `_load_yaml()` returns `None` on file not found but also on YAML parse errors (the underlying `yaml.safe_load` could raise). No explicit error handling for malformed YAML. **Severity: Medium, C3R: Robustness**. `report_data_loader.py:196`.
- **dict vs typed data**:
  - **[P1-RDL-2]** `load_all_student_data()` consumes raw YAML dicts and builds typed `StudentReportData`. However, graph comparison data is loaded from a separate path and patched onto existing objects via attribute assignment — if the YAML schema changes, no validation catches it. **Severity: Medium, C3R: Robustness**. `report_data_loader.py:436-486`.

### longitudinal_report_data.py

- **Public API**: `StudentTrajectory`, `ConceptMasteryChange`, `TopicWeekStats`, `TopicTrendResult`, `LongitudinalSummaryData`, `build_longitudinal_summary()`, `compute_topic_class_statistics()`, `compute_topic_trends()`
- **Unused APIs**: None.
- **Return type issues**:
  - **[P1-LRD-1] CRITICAL: Duplicate `TopicTrendResult` class.** Both `longitudinal_report_data.py:77` and `student_longitudinal_data.py:179` define `TopicTrendResult` with **different field signatures**. The `longitudinal_report_data` version has non-optional fields `kendall_tau: float` while the `student_longitudinal_data` version has `kendall_tau: float | None` plus an extra `interpretation: str` field. This is a naming collision that will cause confusion if a module accidentally imports from the wrong location. **Severity: Critical, C3R: Correctness**. `longitudinal_report_data.py:77`, `student_longitudinal_data.py:179`.
- **Optional field propagation**:
  - **[P1-LRD-2]** `compute_topic_trends()` calls `kendalltau()` / `spearmanr()` — if all values are constant, scipy returns `nan`. The `nan` is wrapped in `float()` and stored as a float, not `None`. Downstream consumers (report chart generators) do not check for NaN. **Severity: High, C3R: Correctness**. `longitudinal_report_data.py:420-421`.

### student_longitudinal_data.py

- **Public API** (via `__all__`): `AlertLevel`, `AnonymizedStudentSummary`, `CohortDistribution`, `CohortWeekStats`, `StudentLongitudinalData`, `TopicTrendResult`, `WarningSignal`, `anonymize`, `build_cohort_distribution`, `build_student_data`, `compute_topic_trends`, `evaluate_warnings`, `parse_id_csv`
- **Internal `_` APIs called from outside**:
  - **[P1-SLD-1]** `_compute_percentile` is imported from `test_adversary_student_longitudinal.py`. Test-only, acceptable. **Severity: Low, C3R: Correctness**.
- **Optional field propagation**:
  - **[P1-SLD-2]** `build_student_data()` returns `StudentLongitudinalData` with `trend_slope: float | None = None`. Callers in `evaluate_warnings()` correctly handle the `None` case. Good.
  - **[P1-SLD-3]** `compute_topic_trends()` — same NaN issue as longitudinal_report_data.py. When scipy returns NaN for constant inputs, it propagates as `float('nan')`. **Severity: High, C3R: Correctness**. `student_longitudinal_data.py:232-237`.

---

## Priority 2: User-Facing Outputs

### professor_report.py

- **Public API**: `ProfessorPDFReportGenerator` class with `generate_pdf()`
- **Exception contract issues**: Font not found raises `FileNotFoundError`. Callers must handle.
- **dict vs typed data**:
  - **[P2-PR-1]** `generate_pdf()` accepts typed `ProfessorReportData` dataclass (from `professor_report_data.py`). Good boundary.

### student_report.py

- **Public API**: `StudentPDFReportGenerator` class with `generate_pdf()`
- **Return type issues**: Returns `str` (output path). Consistent.
- **dict vs typed data**: Consumes `StudentReportData` and `ClassDistributions` typed structures. Good.

### longitudinal_report.py

- **Public API**: `LongitudinalPDFReportGenerator` class with `generate()`
- **Return type issues**: Returns `str` (output path). Consistent.
- **dict vs typed data**: Consumes `LongitudinalSummaryData`. Good.

### warning_report.py

- **Public API**: `WarningPDFReportGenerator` class with `generate()`
- **Return type issues**: Returns `str`. Consistent.

### report_generator.py (deprecated)

- **Public API**: `StudentReportGenerator` class with `generate_pdf()`, `generate_all_reports()`
- **Unused APIs**:
  - **[P2-RG-1]** This module is marked deprecated in docstring but still actively imported by `pipeline_batch_evaluation.py:146` and `pipeline_evaluation.py:1272`. No deprecation warning is emitted at import time. **Severity: High, C3R: Correctness**. `report_generator.py:1`.
- **dict vs typed data**:
  - **[P2-RG-2]** Both `counseling_data` and `config_data` parameters are raw `dict`. No validation. **Severity: Medium, C3R: Robustness**.

### student_longitudinal_report.py

- **Public API**: `StudentLongitudinalPDFGenerator` class with `generate()`
- **Return type issues**: Returns `str`. Consistent.
- **dict vs typed data**: Consumes typed `StudentLongitudinalData`, `CohortDistribution`, etc. Good boundary.

### CLI Modules (cli_report*.py)

- **Public API**: Each has a `main()` function registered as an entry point.
- **Exception contract issues**:
  - **[P2-CLI-1]** `cli_report_longitudinal.py` and `cli_report_professor.py` have late imports guarded by `try/except ImportError` for optional features (risk model, grade model). The ImportError is caught but replacement behavior differs — some silently skip, others print warnings. Inconsistent error reporting. **Severity: Medium, C3R: Robustness**.
  - **[P2-CLI-2]** `cli_report_batch.py` constructs `CrossSectionReport` data including `compute_concept_mastery_by_section()` but `compute_weekly_interaction()` from `section_comparison.py` is never called by any production CLI module — only in tests. This makes `CrossSectionReport.weekly_interaction` always `None` in practice. **Severity: Medium, C3R: Correctness**. `section_comparison.py:310`.

---

## Priority 3: Analysis

### domain_coverage_analyzer.py

- **Public API** (via `__all__`): 25+ functions and classes covering coverage analysis, delivery analysis, network comparison, quality ensemble, and cross-section comparison.
- **Internal `_` APIs in `__all__`**:
  - **[P3-DCA-1]** `_chunk_transcript_with_overlap` and `_DELIVERY_RUBRIC` are private-named (`_` prefix) but explicitly listed in `__all__`. This violates Python convention — `_` prefix implies internal. Either the names should drop the underscore or they should be removed from `__all__`. **Severity: Medium, C3R: Correctness**. `domain_coverage_analyzer.py:65-66`.
- **Return type issues**: Functions consistently return typed dataclasses. Good.
- **Exception contract issues**:
  - **[P3-DCA-2]** `save_coverage_yaml()` uses non-atomic file write (direct `open()` + `yaml.dump()`). Identified in Phase 2 as non-atomic YAML write. **Severity: High, C3R: Robustness** (already noted in Phase 2/5).

### domain_concept_extractor.py

- **Public API**: `TextbookConcept`, `extract_concepts()`, `extract_concepts_llm()`
- **Return type issues**: Consistent typed returns. Good.
- **Exception contract issues**:
  - **[P3-DCE-1]** `extract_concepts_llm()` calls LLM provider — failure modes include API errors, JSON parse errors. Callers must handle. **Severity: Low, C3R: Robustness**.

### lecture_analyzer.py, lecture_preprocessor.py, lecture_comparison.py

- **Public API**: Per CLAUDE.md spec — `analyze_transcript()`, `preprocess_transcript()`, `compare_sections()`, etc.
- **Return type issues**: Consistent typed returns (`AnalysisResult`, `CleanedTranscript`, `ComparisonResult`).
- **Exception contract issues**:
  - **[P3-LA-1]** `analyze_transcript()` uses independent stage failure pattern (FR-027) — each analysis stage can fail independently and the result includes `None` for failed stages. Callers (report generator) handle this with labeled placeholders. Good design. **Severity: Low, C3R: Robustness**.

### risk_predictor.py

- **Public API**: `FEATURE_NAMES`, `RiskFactor`, `RiskPrediction`, `TrainedRiskModel`, `FeatureExtractor`, `RiskPredictor`, `save_model()`, `load_model()`
- **Return type issues**: Consistent typed returns.
- **Exception contract issues**:
  - **[P3-RP-1]** `load_model()` uses `joblib.load()` which can raise `FileNotFoundError`, `ModuleNotFoundError` (if sklearn version differs), or pickle-related exceptions. Only `FileNotFoundError` is commonly caught by callers. **Severity: Medium, C3R: Robustness**. Risk predictor itself is well-structured.

### grade_predictor.py

- **Public API**: `VALID_GRADES`, `GRADE_ORDINAL_MAP`, `ORDINAL_GRADE_MAP`, `GRADE_FEATURE_NAMES`, `GradePrediction`, `TrainedGradeModel`, `GradeFeatureExtractor`, `GradePredictor`, `load_grade_mapping()`, `save_grade_model()`, `load_grade_model()`
- **Return type issues**: Consistent typed returns.
- **Unused APIs**:
  - **[P3-GP-1]** `ORDINAL_GRADE_MAP` reverse mapping is defined but only used in tests, not in production code. Minor. **Severity: Low, C3R: Correctness**. `grade_predictor.py:50`.

---

## Priority 4: Supporting

### OCR: ocr_pipeline.py, llm_ocr.py, naver_ocr.py

- **Public API**: `run_scan_pipeline()`, `run_join_pipeline()` (ocr_pipeline), `extract_text_via_llm()` (llm_ocr), `NCloudOCR` class (naver_ocr)
- **Exception contract issues**:
  - **[P4-OCR-1]** `naver_ocr.py` uses lazy config import: `from forma.config import get_naver_ocr_config, load_config` inside the constructor. If forma.json is missing, `FileNotFoundError` is raised at runtime. Documented behavior. **Severity: Low, C3R: Robustness**.

### Delivery: delivery_prepare.py, delivery_send.py

- **Public API**: `prepare_delivery()`, `send_emails()`, `load_smtp_config()`, `send_summary_email()`
- **Exception contract issues**:
  - **[P4-DL-1]** `delivery_send.py` imports `_build_smtp_config` from `config.py` — this is a private function called cross-module. **Severity: Low, C3R: Correctness**.

### Utils: font_utils.py, io_utils.py, config.py, project_config.py

- **font_utils.py**:
  - **Public API**: `find_korean_font()`, `register_korean_fonts()`, `esc()`, `strip_invisible()`
  - Heavily used across 17+ modules. Stable interface.

- **io_utils.py**:
  - **Public API**: `atomic_write_yaml()`, `atomic_write_json()`, `atomic_write_text()`
  - **Unused APIs**:
    - **[P4-IO-1]** `atomic_write_json()` and `atomic_write_text()` are only called from tests and `cli_select.py` (for YAML). The `io_utils.py` module provides atomic write but `evaluation_io.py` and `longitudinal_store.py` each implement their own atomic write logic independently. This duplication means `io_utils.py` is underutilized. **Severity: Medium, C3R: Robustness**.

- **config.py**:
  - **Public API**: `load_config()`, `get_llm_config()`, `get_smtp_config()`, `get_smtp_password()`, `get_naver_ocr_config()`, `get_quality_weights()`
  - **Internal `_` APIs called from outside**:
    - **[P4-CFG-1]** `_build_smtp_config` is called from `delivery_send.py` (a separate module). **Severity: Low, C3R: Correctness**. `config.py` → `delivery_send.py`.

- **project_config.py**:
  - **Public API**: `ProjectConfiguration`, `find_project_config()`, `load_project_config()`, `validate_project_config()`, `merge_configs()`, `apply_project_config()`
  - All functions used by CLI modules. Good.
  - **[P4-PC-1]** `validate_project_config()` logs warnings for unknown keys/sections but does not raise — silent degradation. This is acceptable for forward-compatibility but could mask typos. **Severity: Low, C3R: Robustness**.

---

## Cross-Cutting Issues

### CC-1: Duplicate `TopicTrendResult` dataclass (Critical)
- `longitudinal_report_data.py:77` defines `TopicTrendResult` with `kendall_tau: float` (non-optional)
- `student_longitudinal_data.py:179` defines `TopicTrendResult` with `kendall_tau: float | None` and extra `interpretation: str` field
- Different signatures, same name. Risk of accidental wrong import.
- **Severity: Critical, C3R: Correctness**

### CC-2: NaN propagation from scipy (High)
- Both `longitudinal_report_data.py:420` and `student_longitudinal_data.py:232` call `kendalltau()` / `spearmanr()` which return NaN for constant input
- The NaN is stored as a float, not converted to None
- Downstream chart generators and report formatters do not check for NaN
- **Severity: High, C3R: Correctness** (already flagged in Phase 2)

### CC-3: Triplicate atomic write implementations (Medium)
- `io_utils.py` provides `atomic_write_yaml()` with fsync + optional locking
- `evaluation_io.py` has its own `save_evaluation_yaml()` with atomic write + backup
- `longitudinal_store.py` has its own atomic save with fcntl locking + backup
- Three different implementations with subtly different semantics (fsync presence, backup behavior, lock scope)
- **Severity: Medium, C3R: Robustness**

### CC-4: Raw dict passing at pipeline → report boundary (High)
- `_build_counseling_summary()` returns raw `dict` — consumed by `report_generator.py` which navigates the dict by string keys
- No intermediate typed structure validates the dict shape
- Schema changes in the pipeline would silently produce broken reports
- **Severity: High, C3R: Robustness**

### CC-5: `report_generator.py` deprecated but still actively used (High)
- Docstring says deprecated, recommends `student_report.py`
- But `pipeline_evaluation.py:1272` and `pipeline_batch_evaluation.py:146` still import and use it
- No `warnings.warn()` deprecation notice at import
- **Severity: High, C3R: Correctness**

### CC-6: Private `_` functions called cross-module (Low)
- `config._build_smtp_config` → `delivery_send.py`
- `section_comparison._cohens_d`, `._effect_size_label` → tests only
- `student_longitudinal_data._compute_percentile` → tests only
- `longitudinal_store._compute_concept_scores`, `._record_key` → tests only
- Production cross-module usage limited to `config._build_smtp_config`
- **Severity: Low, C3R: Correctness**

### CC-7: `compute_weekly_interaction()` unused in production (Medium)
- Defined in `section_comparison.py:310`, tested in adversary/unit tests
- Never called from any CLI or production module
- `CrossSectionReport.weekly_interaction` is always `None` in practice
- **Severity: Medium, C3R: Correctness**

### CC-8: `WEIGHTS_V2` constant unused in production (Medium)
- Defined in `ensemble_scorer.py:45` but never imported
- v2 weight selection happens via inline fallback defaults in `compute_score()`
- The constant and `compute_score()` logic could diverge silently
- **Severity: Medium, C3R: Correctness**

---

## Severity Summary

| Severity | Count | Key Issues |
|---|---|---|
| **Critical** | 1 | CC-1: Duplicate `TopicTrendResult` with different signatures |
| **High** | 7 | CC-2: NaN propagation (x2), CC-4: raw dict at pipeline→report, CC-5: deprecated module still used, P1-PE-2: untyped counseling dict, P3-DCA-2: non-atomic write |
| **Medium** | 13 | CC-3: triplicate atomic write, CC-7/CC-8: unused APIs, P1-PE-1/P1-LS-3/P1-EIO-1/P1-RDL-1/P1-RDL-2: various robustness, P2-CLI-1/P2-CLI-2: inconsistency, P3-DCA-1/P3-RP-1/P4-IO-1: convention/underuse |
| **Low** | 11 | Minor convention violations, test-only cross-module `_` usage, unused reverse maps |
