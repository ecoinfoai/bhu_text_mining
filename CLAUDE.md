# formative-analysis Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-11

## Active Technologies
- Python >=3.11, <4 + scikit-learn>=1.4.0 (KMeans), networkx>=3.4.2 (DiGraph), numpy<2.1.0 (embedding matrix), matplotlib>=3.10.0 (Agg backend), ReportLab>=4.4.4 (Platypus API) — 전부 기존 deps (004-class-graph-clustering)
- YAML (PyYAML>=6.0) — 기존 파이프라인 출력 파일 읽기 전용; 신규 저장소 없음 (004-class-graph-clustering)
- Python >=3.11, <4 + ReportLab >=4.4.4 (Platypus API), matplotlib >=3.10.0 (Agg backend), PyYAML >=6.0, numpy (OLS 회귀), scikit-learn >=1.4.0 (기존 — v0.8.0 신규 사용 없음) (005-longitudinal-analysis)
- YAML key-value store (`LongitudinalStore` — 기존 구현 확장) (005-longitudinal-analysis)
- Python >=3.11, <4 + ReportLab >=4.4.4 (PDF), matplotlib >=3.10.0, PyYAML >=6.0 — all existing, no new deps (006-feedback-quality)
- N/A (feedback stored as string in evaluation YAML, rendered to PDF) (006-feedback-quality)
- Python >=3.11, <4 + PyYAML >=6.0, scikit-learn >=1.4.0 (LogisticRegression, StandardScaler), scipy >=1.12.0 (NEW — ttest_ind, mannwhitneyu), numpy <2.1.0, ReportLab >=4.4.4, matplotlib >=3.10.0, joblib (via scikit-learn) (007-config-risk-warning-comparison)
- YAML files (config, longitudinal store, evaluation results), joblib `.pkl` (model persistence) (007-config-risk-warning-comparison)
- Python >=3.11, <4 + PyYAML >=6.0, scikit-learn >=1.4.0 (LogisticRegression, OrdinalEncoder), networkx >=3.4.2 (DAG, topological_sort, cycle detection), numpy <2.1.0, matplotlib >=3.10.0, ReportLab >=4.4.4, joblib (via scikit-learn) (008-intervention-path-prediction)
- YAML files (intervention_log.yaml, grade_mapping.yaml), joblib .pkl (grade model) (008-intervention-path-prediction)
- Python >=3.11, <4 + PyYAML >=6.0 (기존), Python stdlib (smtplib, email.mime, zipfile, shutil, os) (009-email-delivery)
- YAML 파일 (manifest, roster, smtp config, prepare_summary, delivery_log) (009-email-delivery)
- Python >=3.11, <4 + PyYAML >=6.0 (existing), json (stdlib), warnings (stdlib), smtplib (stdlib) (010-credentials-ci)
- JSON files (`forma.json`), YAML files (`smtp.yaml` — deprecated path) (010-credentials-ci)
- N/A (Markdown documentation only) + N/A (no runtime dependencies) (011-readme-docs)
- Python >=3.11, <4 + ReportLab >=4.4.4, matplotlib >=3.10.0, PyYAML >=6.0, scikit-learn >=1.4.0, networkx >=3.4.2, smtplib (stdlib), ssl (stdlib), fcntl (stdlib), argparse (stdlib). No new runtime deps. (013-audit-hardening)
- YAML files (PyYAML safe_load), joblib .pkl (model persistence), filesystem (reports/zips) (013-audit-hardening)
- Python >=3.11, <4 + ReportLab >=4.4.4, matplotlib >=3.10.0, PyYAML >=6.0, numpy <2.1.0, scikit-learn >=1.4.0, joblib (via scikit-learn) (014-consistency-hardening)
- YAML files (pipeline I/O, longitudinal store, intervention log, delivery log), JSON files (forma.json config), joblib `.pkl` (trained models) (014-consistency-hardening)
- Filesystem — 6 existing `.md` files + 1 new `.md` file (001-update-docs)
- Python >=3.11, <4 + KoNLPy (Okt), kss, BERTopic, UMAP, HDBSCAN, sentence-transformers, networkx >=3.4.2, matplotlib >=3.10.0 (Agg backend), ReportLab >=4.4.4 (Platypus API), PyYAML >=6.0, numpy <2.1.0 — all existing deps, no new runtime deps (001-stt-lecture-analysis)
- YAML files (analysis results, comparison results), PNG (network charts), PDF (reports) (001-stt-lecture-analysis)
- Python >=3.11, <4 + PyYAML >=6.0, argparse (stdlib) (015-fix-pipeline-bugs)
- YAML files (pipeline I/O), no database (015-fix-pipeline-bugs)
- Python >=3.11, <4 + PyYAML >=6.0, ReportLab >=4.4.4 (Platypus API), matplotlib >=3.10.0 (Agg backend) — 전부 기존 deps, 신규 런타임 의존성 없음 (016-ocr-confidence)
- YAML files (OCR results, longitudinal store, week config) (016-ocr-confidence)
- Python >=3.11, <4 + google-genai (Gemini API), anthropic (fallback), PyYAML >=6.0 — 모두 기존 deps, 신규 추가 없음 (018-llm-vision-ocr)
- YAML 파일 (scan 결과, review_needed), JSON (forma.json 설정) (018-llm-vision-ocr)

- Python >=3.11, <4 + ReportLab >=4.4.4 (Platypus API), matplotlib >=3.10.0 (Agg backend), PyYAML >=6.0 (001-student-pdf-report)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

### CLI Commands

- `forma-init` — initialize `forma.yaml` project configuration file (v0.9.0)

- `forma-train` — train drop risk prediction model from longitudinal data (v0.9.0)

  Usage:
  ```
  forma-train --store <longitudinal.yaml> --output <model.pkl> [--weeks 1 2 3 4] [--threshold 0.45]
  ```

- `forma-report-warning` — generate early warning PDF report (v0.9.0)

  Usage:
  ```
  forma-report-warning --final <final.yaml> --config <exam.yaml> --eval-dir <dir> --output <out.pdf> \
    [--longitudinal-store <store.yaml>] [--week 4] [--model <risk.pkl>]
  ```

- `forma-intervention` — manage intervention activity records (v0.10.0)

  Usage:
  ```
  forma-intervention add --store <log.yaml> --student <ID> --week <N> --type <면담|보충학습|과제부여|멘토링|기타> [--description TEXT]
  forma-intervention list --store <log.yaml> [--student <ID>] [--week <N>]
  forma-intervention update --store <log.yaml> --id <N> --outcome <개선|유지|악화>
  ```

- `forma-train-grade` — train semester grade prediction model (v0.10.0)

  Usage:
  ```
  forma-train-grade --store <longitudinal.yaml> --grades <grade_mapping.yaml> --output <model.pkl> [--semester LABEL] [--min-students 10]
  ```

- `forma lecture analyze` — analyze a single STT lecture transcript (v0.12.4)

  Usage:
  ```
  forma lecture analyze --input <transcript.txt> --output <dir> --class <A> --week <1> \
    [--concepts <exam.yaml>] [--no-cache] [--top-n 50] [--no-triplets]
  ```

- `forma lecture compare` — compare class sections for the same session (v0.12.4)

  Usage:
  ```
  forma lecture compare --input-dir <dir> --week <1> --classes A B C D --output <dir> \
    [--concepts <exam.yaml>] [--top-n 50]
  ```

- `forma lecture class-compare` — compare class sections across all sessions combined (v0.12.4)

  Usage:
  ```
  forma lecture class-compare --input-dir <dir> --weeks 1 2 --classes A B C D --output <dir> \
    [--concepts <exam.yaml>] [--top-n 50]
  ```

- `forma-report-batch` — multi-class batch report generator

  Usage:
  ```
  forma-report-batch --config <exam.yaml> --join-dir <dir> --join-pattern "<pattern with {class}>" \
    --eval-pattern "<pattern with {class}>" --output-dir <out> --classes A B [--aggregate] [--no-individual]
  ```

## Code Style

Python >=3.11, <4: Follow standard conventions

## v0.9.0 New Modules (007-config-risk-warning-comparison)
- `project_config.py` — `ProjectConfiguration` dataclass, `find_project_config()`, `load_project_config()`, `validate_project_config()`, `merge_configs()`, `apply_project_config()`
- `risk_predictor.py` — `TrainedModel`, `RiskPrediction`, `FeatureExtractor` (15 features), `RiskPredictor` (LogisticRegression + StandardScaler), `save_model()`/`load_model()` (joblib)
- `warning_report_data.py` — `RiskType` enum (4 values), `INTERVENTION_MAP`, `WarningCard`, `build_warning_data()` (union of rule-based + model-predicted)
- `warning_report_charts.py` — `build_risk_type_distribution_chart()`, `build_deficit_concepts_chart()`
- `warning_report.py` — `WarningPDFReportGenerator` (cover + dashboard + per-student cards)
- `section_comparison.py` — `SectionStats`, `SectionComparison`, `compute_section_stats()`, `compute_pairwise_comparisons()` (Welch's t / Mann-Whitney U, Bonferroni correction)
- `cli_init.py` — `forma-init` interactive config initializer
- `cli_train.py` — `forma-train` model training CLI
- `cli_report_warning.py` — `forma-report-warning` early warning report CLI

## v0.10.0 New Modules (008-intervention-path-prediction)
- `intervention_store.py` — `InterventionRecord`, `InterventionLog` (YAML persistence), `INTERVENTION_TYPES` (면담/보충학습/과제부여/멘토링/기타)
- `intervention_effect.py` — `InterventionEffect`, `InterventionTypeSummary`, `compute_intervention_effects(log, store, window=2)`, `compute_type_summary(effects)`
- `concept_dependency.py` — `ConceptDependency`, `ConceptDependencyDAG`, `parse_concept_dependencies(exam_yaml)`, `build_and_validate_dag(deps)` (cycle detection via NetworkX)
- `learning_path.py` — `LearningPath`, `generate_learning_path(student_id, scores, dag)`, `ClassDeficitMap`, `build_class_deficit_map()`
- `learning_path_charts.py` — `build_learning_path_chart()`, `build_class_deficit_chart()`
- `grade_predictor.py` — `GradeFeatureExtractor` (21 features), `GradePredictor` (LogisticRegression + cold start), `GradePrediction`, `TrainedGradeModel`, `save_grade_model()`/`load_grade_model()` (joblib), `load_grade_mapping()`
- `cli_intervention.py` — `forma-intervention` CLI (add/list/update subcommands)
- `cli_train_grade.py` — `forma-train-grade` CLI (grade model training)

## v0.10.0 Existing Module Changes
- `professor_report.py` — `+intervention_effects`, `+intervention_type_summaries`, `+grade_predictions` kwargs in `generate_pdf()`; `_build_intervention_section()`, `_build_grade_prediction_section()`
- `student_report.py` — `+learning_path`, `+learning_path_chart`, `+grade_trend` kwargs; `_build_learning_path_section()`
- `longitudinal_report.py` — `+intervention_effects` kwarg in `generate()`; `_build_intervention_chart_section()`
- `longitudinal_report_charts.py` — `+build_intervention_effect_chart()`
- `cli_report.py` — `+--concept-deps`, `+--grade-model` flags (no `--intervention-log` per FR-013)
- `cli_report_professor.py` — `+--intervention-log`, `+--grade-model` flags
- `cli_report_longitudinal.py` — `+--intervention-log` flag

## v0.12.4 New Modules (001-stt-lecture-analysis)
- `lecture_preprocessor.py` — `BIOLOGY_ABBREVIATIONS` frozenset, `MAX_TRANSCRIPT_LENGTH=50000`, `preprocess_transcript(text, extra_stopwords, extra_abbreviations)` → `CleanedTranscript`, 8-step pipeline: UTF-8/EUC-KR decode, filler removal, char normalization, mixed-token split, 3-layer stopwords, abbreviation preservation, custom stopwords, empty validation
- `lecture_analyzer.py` — `analyze_transcript(cleaned, concepts, top_n, no_triplets, provider)` → `AnalysisResult`, orchestrates keyword extraction, network generation, topic modeling, emphasis scoring, triplet extraction; independent stage failure (FR-027)
- `lecture_merge.py` — `merge_transcripts(analyses, class_id)` → `MergedAnalysis`, preserves per-session keyword data alongside combined (FR-022), session boundary markers
- `lecture_comparison.py` — `compare_sections(analyses, concepts, top_n)` → `ComparisonResult`, exclusive top-N keywords (FR-017), concept gaps (FR-018), emphasis variance ranking (FR-019)
- `lecture_report.py` — `LectureReportGenerator` (story-based ReportLab Platypus), `generate_analysis_report(result)`, `generate_comparison_report(comparison)` → PDF; labeled placeholder for skipped/failed stages (FR-027)
- `cli_lecture.py` — `main_analyze()`, `main_compare()`, `main_class_compare()`, registered under `forma lecture` namespace (FR-026)

## v0.12.4 Existing Module Changes
- `cli_main.py` — `+("lecture", "analyze/compare/class-compare")` in `_COMMANDS`, `+"lecture"` in `_NESTED_GROUPS`, lecture subparser in `_build_parser()`
- `week_config.py` — `+lecture_*` fields in `WeekConfiguration`, `+"lecture_transcript_pattern"` in `_CLASS_PATTERN_FIELDS`, lecture section parsing in `load_week_config()`

## Recent Changes
- 018-llm-vision-ocr: Added Python >=3.11, <4 + google-genai (Gemini API), anthropic (fallback), PyYAML >=6.0 — 모두 기존 deps, 신규 추가 없음
- 016-ocr-confidence: Added Python >=3.11, <4 + PyYAML >=6.0, ReportLab >=4.4.4 (Platypus API), matplotlib >=3.10.0 (Agg backend) — 전부 기존 deps, 신규 런타임 의존성 없음
- 015-fix-pipeline-bugs: Added Python >=3.11, <4 + PyYAML >=6.0, argparse (stdlib)


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
