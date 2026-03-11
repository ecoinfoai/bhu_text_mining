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

## Recent Changes
- 011-readme-docs: Added N/A (Markdown documentation only) + N/A (no runtime dependencies)
- 010-credentials-ci: Added Python >=3.11, <4 + PyYAML >=6.0 (existing), json (stdlib), warnings (stdlib), smtplib (stdlib)
- 009-email-delivery: Added Python >=3.11, <4 + PyYAML >=6.0 (기존), Python stdlib (smtplib, email.mime, zipfile, shutil, os)


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
