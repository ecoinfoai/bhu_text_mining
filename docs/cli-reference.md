# CLI Reference

Complete reference for all 22 FormA CLI commands, accessible through the unified `forma` entry point.

> **Note:** The deprecated `forma-*` command names (e.g., `forma-exam`, `forma-train`) remain functional but emit a `DeprecationWarning`. Migrate to the `forma <subcommand>` format shown in this reference.

## Table of Contents

- [forma exam](#forma-exam)
- [forma ocr](#forma-ocr)
- [forma eval](#forma-eval)
- [forma eval batch](#forma-eval-batch)
- [forma report student](#forma-report-student)
- [forma report professor](#forma-report-professor)
- [forma report batch](#forma-report-batch)
- [forma report student-longitudinal](#forma-report-student-longitudinal)
- [forma report student-longitudinal-batch](#forma-report-student-longitudinal-batch)
- [forma report student-summary](#forma-report-student-summary)
- [forma report longitudinal](#forma-report-longitudinal)
- [forma report warning](#forma-report-warning)
- [forma train risk](#forma-train-risk)
- [forma train grade](#forma-train-grade)
- [forma init](#forma-init)
- [forma deliver](#forma-deliver)
- [forma intervention](#forma-intervention)
- [forma select](#forma-select)
- [forma lecture analyze](#forma-lecture-analyze)
- [forma lecture compare](#forma-lecture-compare)
- [forma lecture class-compare](#forma-lecture-class-compare)
- [forma backfill longitudinal](#forma-backfill-longitudinal)
- [forma domain extract](#forma-domain-extract)
- [forma domain coverage](#forma-domain-coverage)
- [forma domain report](#forma-domain-report)

---

## Global Flags

These flags are accepted by the top-level `forma` command and apply to all subcommands:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--version` | flag | -- | Print version and exit |
| `--verbose` | flag | false | Enable verbose output |
| `--no-config` | flag | false | Skip `forma.yaml` loading |
| `--font-path` | path | None | Custom font file path |
| `--dpi` | integer | 150 | Chart resolution DPI |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Input error (invalid arguments, missing required flags, data error) |
| 2 | File error (file not found, permission denied, write error) |
| 3 | Rendering error (font missing) or partial failure (some emails failed) |

> Accurate as of v0.13.0

---

## forma exam

Generate formative exam paper PDFs with randomized student IDs and QR codes.

**Synopsis:**

```bash
forma exam (--config <path> | --questions <path> | --questions-json <json>) --output <path> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--config` | path | yes (mutually exclusive) | -- | Unified YAML file path (metadata + questions) |
| `--questions` | path | yes (mutually exclusive) | -- | YAML file path containing question list |
| `--questions-json` | string | yes (mutually exclusive) | -- | Inline JSON string containing question list |
| `--output` | path | yes | -- | Output PDF file path |
| `--num-papers` | integer | no | None | Number of exam papers to generate (required via CLI or config YAML) |
| `--year` | integer | no | None | Academic year |
| `--grade` | integer | no | None | Grade level |
| `--semester` | integer | no | None | Semester number |
| `--course` | string | no | None | Course name |
| `--week` | integer | no | None | Week number |
| `--form-url` | string | no | None | Google Forms URL template |
| `--student-ids` | list | no | None | Space-separated student ID list |
| `--font-path` | path | no | None | Path to Korean font file |

Exactly one of `--config`, `--questions`, or `--questions-json` must be provided. When using `--config`, metadata fields (year, grade, semester, etc.) are read from the YAML file and can be overridden by CLI flags.

**Examples:**

```bash
# Generate from unified config YAML
forma exam --config exam.yaml --output exam.pdf

# Generate with explicit question file and paper count
forma exam --questions questions.yaml --num-papers 50 --output exam.pdf

# Generate with inline JSON questions
forma exam --questions-json '[{"topic":"T","text":"Q","limit":"50"}]' \
           --num-papers 30 --output exam.pdf
```

---

## forma ocr

OCR pipeline for scanned exam answer sheets. Has three subcommands: `scan`, `join`, and `compare`.

**Synopsis:**

```bash
forma ocr [--no-config] scan (--config <path> | --class <id>) [options]
forma ocr [--no-config] join [--class <id>] [--ocr-results <path>] [--output <path>] [options]
forma ocr [--no-config] compare (--image <path> | --image-dir <path>) --output <path> [options]
```

### forma ocr scan

Scan images, decode QR codes, run OCR, and produce a YAML result file.

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--config` | path | mutually exclusive | -- | OCR configuration YAML file path (legacy) |
| `--class` | string | mutually exclusive | -- | Class identifier; substitutes `{class}` in `week.yaml` path patterns |
| `--provider` | string | no | `gemini` | LLM provider (`gemini` or `anthropic`) |
| `--model` | string | no | None | LLM model ID override |
| `--subject` | string | no | None | Subject name (LLM prompt context) |
| `--question` | string | no | None | Question text (LLM prompt context) |
| `--answer-keywords` | string | no | None | Key terms, comma-separated (LLM prompt context) |
| `--num-questions` | integer | no | None | Number of questions (can be specified in config YAML) |
| `--recrop` | flag | no | false | Ignore saved crop coordinates; re-select |
| `--week-config` | path | no | auto-discover | `week.yaml` path |
| `--ocr-review-threshold` | float | no | 0.75 | OCR confidence review threshold |

**Examples:**

```bash
forma ocr scan --config ocr_config.yaml
forma ocr scan --class A --week-config week.yaml
forma ocr scan --class A --provider anthropic --num-questions 3
```

### forma ocr join

Join OCR results with Google Forms/Sheets responses to produce a unified output YAML.

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--class` | string | no | None | Class identifier; substitutes `{class}` in `week.yaml` path patterns |
| `--ocr-results` | path | no | None | OCR results YAML file path |
| `--output` | path | no | None | Output YAML file path |
| `--spreadsheet-url` | string | no | None | Google Sheets URL (preferred data source) |
| `--forms-csv` | path | no | None | Google Forms CSV file path (fallback) |
| `--credentials` | path | no | `credentials.json` | OAuth2 credentials JSON file path |
| `--manual-mapping` | path | no | None | Manual mapping YAML for unmatched students |
| `--student-id-column` | string | no | `student_id` | Student ID column name in the spreadsheet |
| `--week-config` | path | no | auto-discover | `week.yaml` path |
| `--ocr-review-threshold` | float | no | 0.75 | OCR confidence review threshold |

**Examples:**

```bash
forma ocr join --class A --week-config week.yaml

forma ocr join --ocr-results results.yaml --output final.yaml \
               --spreadsheet-url "https://docs.google.com/spreadsheets/d/XXX"

forma ocr join --ocr-results results.yaml --output final.yaml \
               --forms-csv fallback.csv --manual-mapping mapping.yaml
```

### forma ocr compare

Compare Naver OCR results against LLM Vision OCR results (research/diagnostic tool).

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--image` | path | mutually exclusive | -- | Single image file path to compare |
| `--image-dir` | path | mutually exclusive | -- | Directory of images for batch comparison |
| `--output` | path | yes | -- | Output YAML/directory path |
| `--provider` | string | no | `gemini` | LLM provider (`gemini` or `anthropic`) |
| `--model` | string | no | None | LLM model ID override |
| `--subject` | string | no | None | Subject name (LLM prompt context) |
| `--question` | string | no | None | Question text (LLM prompt context) |
| `--answer-keywords` | string | no | None | Key terms, comma-separated (LLM prompt context) |
| `--num-questions` | integer | no | 1 | Number of questions |
| `--class` | string | no | None | Class identifier |

**Examples:**

```bash
forma ocr compare --image scan001.jpg --output compare_result.yaml
forma ocr compare --image-dir scans/ --output compare_results/ --provider anthropic
```

### Common Options

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--no-config` | flag | no | false | Skip loading forma.yaml project configuration |

---

## forma eval

Multi-layer concept evaluation pipeline. Supports both direct CLI flags and a single eval-config YAML file for all options.

**Synopsis:**

```bash
forma eval [--eval-config <path>] [--config <path>] [--responses <path>] [--output <path>] [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--eval-config` | path | no | None | Evaluation config YAML (all options in one file) |
| `--config` | path | conditional | None | Exam YAML config path (required if no eval-config) |
| `--responses` | path | conditional | None | Student responses YAML path (required if no eval-config) |
| `--output` | path | conditional | None | Output directory (required if no eval-config) |
| `--api-key` | string | no | None | LLM API key (overrides env var) |
| `--provider` | string | no | `gemini` | LLM provider: `gemini` or `anthropic` |
| `--model` | string | no | None | LLM model ID override |
| `--skip-feedback` | flag | no | false | Skip feedback generation |
| `--skip-llm` | flag | no | false | Deprecated: use `--skip-feedback` instead |
| `--skip-graph` | flag | no | false | Skip triplet extraction and graph comparison |
| `--skip-stats` | flag | no | false | Skip Layer 3 statistical analysis |
| `--lecture-transcript` | path | no | None | Path to lecture transcript file |
| `--longitudinal-store` | path | no | None | Path to longitudinal data store YAML |
| `--generate-reports` | flag | no | false | Generate student PDF reports after evaluation |
| `--questions-used` | list (int) | no | None | Exam question serial numbers in q order (e.g., `1 3`) |
| `--n-calls` | integer | no | 3 | Number of independent LLM calls per evaluation |
| `--class` | string | no | None | Class section identifier (replaces `{class}` pattern in `week.yaml`) |
| `--week-config` | path | no | auto-discover | `week.yaml` path (auto-searched in current directory if omitted) |

Either `--eval-config` or all three of `--config`, `--responses`, `--output` must be provided. When `--class` and `--week-config` are used, paths are resolved from `week.yaml` patterns. CLI flags override values from the eval-config YAML.

**Examples:**

```bash
# Using eval-config YAML
forma eval --eval-config eval_w1_A.yaml

# Using explicit flags
forma eval --config exam.yaml --responses responses.yaml --output results/ \
           --provider gemini --skip-stats

# With question mapping from OCR join output
forma eval --config exam.yaml --responses final.yaml --output results/ \
           --questions-used 1 3
```

---

## forma eval batch

Batch evaluation pipeline for multiple class sections. Runs the full evaluation pipeline for each specified class.

**Synopsis:**

```bash
forma eval batch --config <path> --join-dir <path> --join-pattern <pattern> \
                 --output <path> --classes <ids...> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--config` | path | yes | -- | Exam YAML config path |
| `--join-dir` | path | yes | -- | Directory containing join output files |
| `--join-pattern` | string | yes | -- | Filename pattern with `{class}` placeholder |
| `--output` | path | yes | -- | Root output directory |
| `--classes` | list | yes | -- | Space-separated class identifiers |
| `--provider` | string | no | `gemini` | LLM provider: `gemini` or `anthropic` |
| `--api-key` | string | no | None | LLM API key (overrides env var) |
| `--model` | string | no | None | LLM model ID override |
| `--skip-feedback` | flag | no | false | Skip feedback generation |
| `--skip-llm` | flag | no | false | Deprecated: use `--skip-feedback` instead |
| `--skip-graph` | flag | no | false | Skip triplet extraction and graph comparison |
| `--skip-stats` | flag | no | false | Skip Layer 3 statistical analysis |
| `--lecture-transcript` | path | no | None | Path to lecture transcript file |
| `--longitudinal-store` | path | no | None | Path to longitudinal data store YAML |
| `--generate-reports` | flag | no | false | Generate student PDF reports |
| `--questions-used` | list (int) | no | None | Exam question serial numbers in q order |

**Examples:**

```bash
forma eval batch --config exam.yaml \
                 --join-dir results/anp_w1/ \
                 --join-pattern "anp_1{class}_final.yaml" \
                 --output results/anp_w1_eval/ \
                 --classes A B C D \
                 --provider gemini --generate-reports
```

---

## forma report student

Generate individual student PDF reports with evaluation results, feedback, and optional longitudinal comparisons.

**Synopsis:**

```bash
forma report student --final <path> --config <path> --eval-dir <path> --output-dir <path> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--final` | path | yes | -- | Student responses YAML file path |
| `--config` | path | yes | -- | Exam config YAML file path |
| `--eval-dir` | path | yes | -- | Evaluation results directory path |
| `--output-dir` | path | yes | -- | PDF output directory path |
| `--student` | string | no | None | Generate report for a specific student ID only |
| `--font-path` | path | no | None | Korean font file path (auto-detected if omitted) |
| `--dpi` | integer | no | 150 | Chart image resolution (range: 72-600) |
| `--verbose` | flag | no | false | Enable detailed logging |
| `--no-config` | flag | no | false | Skip loading forma.yaml project configuration |
| `--longitudinal-store` | path | no | None | Longitudinal store YAML path (enables change tracking) |
| `--week` | integer | no | None | Current week number (for comparison baseline; requires `--longitudinal-store`) |
| `--grade-model` | path | no | None | Grade prediction model file path (.pkl, for trend display) |
| `--concept-deps` | flag | no | false | Enable concept-dependency learning path (requires definitions in exam YAML) |
| `--intervention-log` | path | no | None | Intervention log YAML path (ignored in student reports per FR-013) |

**Examples:**

```bash
# Generate reports for all students
forma report student --final anp_1A_final.yaml --config exam.yaml \
             --eval-dir eval_1A/ --output-dir reports/

# Generate for a single student with longitudinal data
forma report student --final anp_1A_final.yaml --config exam.yaml \
             --eval-dir eval_1A/ --output-dir reports/ \
             --student S015 --longitudinal-store store.yaml --week 4

# With grade prediction and learning path
forma report student --final anp_1A_final.yaml --config exam.yaml \
             --eval-dir eval_1A/ --output-dir reports/ \
             --grade-model grade.pkl --concept-deps \
             --longitudinal-store store.yaml --week 4
```

---

## forma report professor

Generate a professor-facing class summary PDF report with analytics, LLM-powered analysis, and optional risk prediction.

**Synopsis:**

```bash
forma report professor --final <path> --config <path> --eval-dir <path> --output-dir <path> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--final` | path | yes | -- | Final results YAML file path |
| `--config` | path | yes | -- | Exam config YAML file path |
| `--eval-dir` | path | yes | -- | Evaluation results directory path |
| `--output-dir` | path | yes | -- | PDF output directory path |
| `--forma-config` | path | no | None | Forma configuration file path |
| `--class-name` | string | no | None | Class name (auto-extracted from filename if omitted) |
| `--skip-llm` | flag | no | false | Skip AI analysis generation |
| `--font-path` | path | no | None | Korean font file path |
| `--dpi` | integer | no | 150 | Chart DPI |
| `--verbose` | flag | no | false | Enable detailed logging |
| `--no-config` | flag | no | false | Skip loading forma.yaml project configuration |
| `--model` | path | no | None | Drop risk prediction model file path (.pkl) |
| `--transcript-dir` | path | no | None | Lecture transcript text files directory |
| `--longitudinal-store` | path | no | None | Longitudinal store YAML path (enables risk movement tracking) |
| `--week` | integer | no | None | Current week number (requires `--longitudinal-store`) |
| `--grade-model` | path | no | None | Grade prediction model file path (.pkl, from `forma train grade`) |
| `--intervention-log` | path | no | None | Intervention log YAML path (enables intervention effect analysis) |

**Examples:**

```bash
# Basic professor report
forma report professor --final anp_1A_final.yaml --config exam.yaml \
                       --eval-dir eval_1A/ --output-dir reports/

# Full-featured report with all optional data sources
forma report professor --final anp_1A_final.yaml --config exam.yaml \
                       --eval-dir eval_1A/ --output-dir reports/ \
                       --class-name "1A" --model risk.pkl --grade-model grade.pkl \
                       --longitudinal-store store.yaml --week 4 \
                       --intervention-log interventions.yaml \
                       --transcript-dir transcripts/
```

---

## forma report batch

Generate PDF reports for multiple class sections at once, with optional aggregate cross-section analysis.

**Synopsis:**

```bash
forma report batch --config <path> --join-dir <path> --join-pattern <pattern> \
                   --eval-pattern <pattern> --output-dir <path> --classes <ids...> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--config` | path | yes | -- | Exam YAML config path |
| `--join-dir` | path | yes | -- | Directory containing final YAML files |
| `--join-pattern` | string | yes | -- | Final YAML filename pattern with `{class}` placeholder |
| `--eval-pattern` | string | yes | -- | Evaluation directory pattern with `{class}` placeholder |
| `--output-dir` | path | yes | -- | Root output directory |
| `--classes` | list | yes | -- | Space-separated class identifiers |
| `--aggregate` | flag | no | false | Generate merged multi-class professor report |
| `--no-individual` | flag | no | false | Skip individual student PDF generation |
| `--skip-llm` | flag | no | false | Skip LLM analysis |
| `--font-path` | path | no | None | Path to Korean font file |
| `--dpi` | integer | no | 150 | Image DPI |
| `--verbose` | flag | no | false | Enable detailed logging |
| `--no-config` | flag | no | false | Skip loading forma.yaml project configuration |
| `--transcript-pattern` | string | no | None | Transcript directory pattern with `{class}` placeholder |

**Examples:**

```bash
# Generate per-class student + professor reports
forma report batch --config exam.yaml --join-dir results/anp_w1/ \
                   --join-pattern "anp_1{class}_final.yaml" \
                   --eval-pattern "eval_{class}/" \
                   --output-dir reports/ --classes A B C D

# Generate aggregate cross-section report only (no student PDFs)
forma report batch --config exam.yaml --join-dir results/anp_w1/ \
                   --join-pattern "anp_1{class}_final.yaml" \
                   --eval-pattern "eval_{class}/" \
                   --output-dir reports/ --classes A B C D \
                   --aggregate --no-individual
```

---

## forma report student-longitudinal

Generate an individual student longitudinal analysis PDF report with trajectory charts, percentile rankings, and optional LLM interpretation.

**Synopsis:**

```bash
forma-report-student --store <path> --student <id> --id-csv <path> --output <path> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--store` | path | yes | -- | Longitudinal store YAML file path |
| `--student` | string | yes | -- | Student ID |
| `--id-csv` | path | yes | -- | Student ID-name-class mapping CSV file path |
| `--output` | path | yes | -- | Output PDF file path |
| `--week` | string | no | None | (DEPRECATED) Use `--weeks` instead |
| `--weeks` | list | no | None | Week selection: single N (1..N), range `start:end`, or list `1 3 5` |
| `--font-path` | path | no | None | Korean font file path (auto-detected if omitted) |
| `--dpi` | integer | no | 150 | Chart DPI |
| `--no-llm` | flag | no | false | Generate report with charts only, no LLM calls |
| `--no-config` | flag | no | false | Skip loading forma.yaml project configuration |
| `--verbose` | flag | no | false | Enable detailed logging |

**Examples:**

```bash
# Generate a single student longitudinal report
forma-report-student --store longitudinal.yaml --student 2024001 \
                     --id-csv students.csv --output report_2024001.pdf

# With specific weeks and no LLM
forma-report-student --store longitudinal.yaml --student 2024001 \
                     --id-csv students.csv --output report_2024001.pdf \
                     --weeks 1 2 3 --no-llm
```

---

## forma report student-longitudinal-batch

Generate student longitudinal reports in batch for all students in the longitudinal store.

**Synopsis:**

```bash
forma-report-student-batch --store <path> --id-csv <path> --output-dir <path> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--store` | path | yes | -- | Longitudinal store YAML file path |
| `--id-csv` | path | yes | -- | Student ID-name-class mapping CSV file path |
| `--output-dir` | path | yes | -- | Output PDF directory path |
| `--week` | string | no | None | (DEPRECATED) Use `--weeks` instead |
| `--weeks` | list | no | None | Week selection: single N (1..N), range `start:end`, or list `1 3 5` |
| `--font-path` | path | no | None | Korean font file path (auto-detected if omitted) |
| `--dpi` | integer | no | 150 | Chart DPI |
| `--no-llm` | flag | no | false | Generate reports with charts only, no LLM calls |
| `--no-config` | flag | no | false | Skip loading forma.yaml project configuration |
| `--verbose` | flag | no | false | Enable detailed logging |

**Examples:**

```bash
# Generate reports for all students
forma-report-student-batch --store longitudinal.yaml --id-csv students.csv \
                           --output-dir student_reports/ --no-llm

# With specific week range
forma-report-student-batch --store longitudinal.yaml --id-csv students.csv \
                           --output-dir student_reports/ --weeks 1:4
```

---

## forma report student-summary

Generate a single PDF with a tabular overview of all students' longitudinal scores, trends, percentiles, and warning levels.

**Synopsis:**

```bash
forma-report-student-summary --store <path> --id-csv <path> --output <path> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--store` | path | yes | -- | Longitudinal store YAML file path |
| `--id-csv` | path | yes | -- | Student ID-name-class mapping CSV file path |
| `--output` | path | yes | -- | Output PDF file path |
| `--week` | string | no | None | (DEPRECATED) Use `--weeks` instead |
| `--weeks` | list | no | None | Week selection: single N (1..N), range `start:end`, or list `1 3 5` |
| `--course-name` | string | no | `""` | Course name (displayed on cover) |
| `--font-path` | path | no | None | Korean font file path (auto-detected if omitted) |
| `--dpi` | integer | no | 150 | Chart DPI |
| `--no-config` | flag | no | false | Skip loading forma.yaml project configuration |
| `--verbose` | flag | no | false | Enable detailed logging |

**Examples:**

```bash
# Generate cohort summary report
forma-report-student-summary --store longitudinal.yaml --id-csv students.csv \
                             --output summary.pdf --course-name "해부생리학"
```

---

## forma report longitudinal

Generate a longitudinal summary PDF report showing student trajectories, heatmaps, and concept mastery changes over multiple weeks.

**Synopsis:**

```bash
forma report longitudinal --store <path> --class-name <name> --output <path> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--store` | path | yes | -- | Longitudinal store YAML file path |
| `--class-name` | string | yes | -- | Class name (displayed on report cover) |
| `--output` | path | yes | -- | Output PDF file path |
| `--weeks` | list (int) | no | None | Week numbers to include (e.g., `1 2 3 4`); all weeks if omitted |
| `--exam-file` | path | no | None | Exam file path (for concept mastery analysis reference) |
| `--font-path` | path | no | None | Korean font file path (auto-detected if omitted) |
| `--no-config` | flag | no | false | Skip loading forma.yaml project configuration |
| `--model` | path | no | None | Drop risk prediction model file path (.pkl) |
| `--intervention-log` | path | no | None | Intervention log YAML path (enables pre/post intervention chart) |
| `--classes` | list | no | None | Filter by class IDs (e.g., `A B C D`); enables per-class heatmaps |
| `--heatmap-layout` | string | no | `1:N` | Heatmap subplot layout as `rows:cols` (e.g., `1:4`, `2:2`) |
| `--risk-threshold` | float | no | 0.45 | Persistent risk cutoff threshold |
| `--mastery-top-n` | integer | no | None | Show only top N concepts in mastery chart (ranked by absolute change) |

**Examples:**

```bash
# Basic longitudinal report with all weeks
forma report longitudinal --store longitudinal.yaml --class-name "1A" \
                          --output longitudinal_report.pdf

# With specific weeks and risk model
forma report longitudinal --store longitudinal.yaml --class-name "1A" \
                          --output longitudinal_report.pdf \
                          --weeks 1 2 3 4 --model risk.pkl

# With intervention effect chart
forma report longitudinal --store longitudinal.yaml --class-name "1A" \
                          --output longitudinal_report.pdf \
                          --intervention-log interventions.yaml
```

---

## forma report warning

Generate an early warning PDF report identifying at-risk students with risk types, deficit concepts, and recommended interventions.

**Synopsis:**

```bash
forma report warning --final <path> --config <path> --eval-dir <path> --output <path> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--final` | path | yes | -- | Final results YAML file path |
| `--config` | path | yes | -- | Exam config YAML file path |
| `--eval-dir` | path | yes | -- | Evaluation results directory path |
| `--output` | path | yes | -- | Output PDF file path |
| `--longitudinal-store` | path | no | None | Longitudinal store YAML path (for model predictions) |
| `--week` | integer | no | None | Current week number |
| `--model` | path | no | None | Pre-trained risk prediction model file path (.pkl) |
| `--font-path` | path | no | None | Korean font file path (auto-detected if omitted) |
| `--dpi` | integer | no | 150 | Chart DPI |
| `--verbose` | flag | no | false | Enable detailed logging |
| `--no-config` | flag | no | false | Skip loading forma.yaml project configuration |

**Examples:**

```bash
# Basic warning report using rule-based detection only
forma report warning --final anp_1A_final.yaml --config exam.yaml \
                     --eval-dir eval_1A/ --output warning.pdf

# With model-based prediction
forma report warning --final anp_1A_final.yaml --config exam.yaml \
                     --eval-dir eval_1A/ --output warning.pdf \
                     --longitudinal-store store.yaml --week 4 --model risk.pkl
```

---

## forma train risk

Train a drop risk prediction model from longitudinal data using logistic regression.

**Synopsis:**

```bash
forma train risk --store <path> --output <path> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--store` | path | yes | -- | Longitudinal store YAML file path |
| `--output` | path | yes | -- | Output model file path (.pkl) |
| `--threshold` | float | no | 0.45 | Drop definition score threshold |
| `--min-weeks` | integer | no | 3 | Minimum number of weeks required |
| `--min-students` | integer | no | 10 | Minimum number of students required |
| `--verbose` | flag | no | false | Enable detailed logging |

**Examples:**

```bash
# Train with default settings
forma train risk --store longitudinal.yaml --output risk_model.pkl

# Train with custom threshold and minimums
forma train risk --store longitudinal.yaml --output risk_model.pkl \
            --threshold 0.40 --min-weeks 4 --min-students 20
```

---

## forma train grade

Train a semester grade prediction model from longitudinal data and historical grade mappings.

**Synopsis:**

```bash
forma train grade --store <path> --grades <path> --output <path> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--store` | path | yes | -- | Longitudinal store YAML file path |
| `--grades` | path | yes | -- | Grade mapping YAML file path |
| `--output` | path | yes | -- | Output model file path (.pkl) |
| `--semester` | string | no | None | Semester label to use for training (uses last semester if omitted) |
| `--min-students` | integer | no | 10 | Minimum number of students with grades |
| `--verbose` | flag | no | false | Enable detailed logging |
| `--no-config` | flag | no | false | Skip loading forma.yaml project configuration |

**Examples:**

```bash
# Train using the last semester's grades
forma train grade --store longitudinal.yaml --grades grade_mapping.yaml \
                  --output grade_model.pkl

# Train for a specific semester
forma train grade --store longitudinal.yaml --grades grade_mapping.yaml \
                  --output grade_model.pkl --semester "2025-2"
```

---

## forma init

Interactively generate a `forma.yaml` project configuration template file.

**Synopsis:**

```bash
forma init [--output <path>] [--force]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--output` | path | no | `forma.yaml` | Output file path |
| `--force` | flag | no | false | Overwrite existing file |

The command prompts interactively for course name, academic year, semester, and class identifiers.

**Examples:**

```bash
# Generate default forma.yaml in current directory
forma init

# Generate at a custom path, overwriting if exists
forma init --output config/forma.yaml --force
```

---

## forma deliver

Report email delivery automation. Has two subcommands: `prepare` and `send`.

**Synopsis:**

```bash
forma deliver [--no-config] [--verbose] prepare --manifest <path> --roster <path> --output-dir <path> [options]
forma deliver [--no-config] [--verbose] send --staged <path> --template <path> [options]
```

### Common Options

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--no-config` | flag | no | false | Skip loading forma.yaml project configuration |
| `--verbose` | flag | no | false | Enable detailed logging |

### forma deliver prepare

Collect student report files and create zip archives for each student.

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--manifest` | path | yes | -- | Delivery manifest YAML file path |
| `--roster` | path | yes | -- | Student roster YAML file path |
| `--output-dir` | path | yes | -- | Staging folder output path |
| `--force` | flag | no | false | Overwrite existing staging folder |

**Examples:**

```bash
forma deliver prepare --manifest manifest.yaml --roster roster.yaml \
                      --output-dir staging/
```

### forma deliver send

Send emails with zip attachments via SMTP.

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--staged` | path | yes | -- | Staging folder path (created by `prepare`) |
| `--template` | path | yes | -- | Email template YAML file path |
| `--smtp-config` | path | no | None | SMTP config YAML file path (deprecated; use config.json smtp section instead) |
| `--dry-run` | flag | no | false | Preview only, no actual sending |
| `--retry-failed` | flag | no | false | Resend only previously failed emails |
| `--force` | flag | no | false | Ignore previous delivery records and resend all |
| `--notify-sender` | flag | no | false | Send summary email to the professor |
| `--password-from-stdin` | flag | no | false | Read SMTP password from stdin |

`--retry-failed` and `--force` cannot be used together. SMTP configuration resolution order: explicit `--smtp-config` path > `config.json` smtp section > error (exit 2). SMTP password is never stored in config files; use `--password-from-stdin` or the `FORMA_SMTP_PASSWORD` environment variable.

**Examples:**

```bash
# Dry-run preview
forma deliver send --staged staging/ --template template.yaml --dry-run

# Send emails using config.json SMTP configuration
echo "$SMTP_PASSWORD" | forma deliver send --staged staging/ --template template.yaml \
                                           --password-from-stdin

# Retry failed emails with explicit SMTP config (deprecated path)
forma deliver send --staged staging/ --template template.yaml \
                   --smtp-config smtp.yaml --retry-failed

# Send with professor summary notification
forma deliver send --staged staging/ --template template.yaml \
                   --notify-sender --password-from-stdin
```

---

## forma intervention

Manage intervention activity records (counseling, supplementary learning, assignments, mentoring). Has three subcommands: `add`, `list`, and `update`.

**Synopsis:**

```bash
forma intervention [--no-config] [--verbose] add --store <path> --student <id> --week <n> --type <type> [options]
forma intervention [--no-config] [--verbose] list --store <path> [options]
forma intervention [--no-config] [--verbose] update --store <path> --id <n> --outcome <outcome>
```

### Common Options

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--no-config` | flag | no | false | Skip loading forma.yaml project configuration |
| `--verbose` | flag | no | false | Enable detailed logging |

### forma intervention add

Add a new intervention record.

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--store` | path | yes | -- | Intervention log YAML file path |
| `--student` | string | yes | -- | Student ID |
| `--week` | integer | yes | -- | Week number (must be >= 1) |
| `--type` | string | yes | -- | Intervention type: one of `면담` (counseling), `보충학습` (supplementary learning), `과제부여` (assignment), `멘토링` (mentoring), `기타` (other) |
| `--description` | string | no | `""` | Description of the intervention |
| `--recorded-by` | string | no | None | Name of the person recording |
| `--follow-up-week` | integer | no | None | Follow-up week number |

**Examples:**

```bash
forma intervention add --store interventions.yaml --student S015 \
                       --week 3 --type "면담" \
                       --description "Discussed study habits" --recorded-by "Prof. Kim"
```

### forma intervention list

List intervention records with optional filters.

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--store` | path | yes | -- | Intervention log YAML file path |
| `--student` | string | no | None | Filter by student ID |
| `--week` | integer | no | None | Filter by week number |

**Examples:**

```bash
# List all records
forma intervention list --store interventions.yaml

# Filter by student
forma intervention list --store interventions.yaml --student S015

# Filter by week
forma intervention list --store interventions.yaml --week 3
```

### forma intervention update

Update the outcome of an existing intervention record.

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--store` | path | yes | -- | Intervention log YAML file path |
| `--id` | integer | yes | -- | Intervention record ID |
| `--outcome` | string | yes | -- | Outcome value: one of `개선` (improved), `유지` (maintained), `악화` (worsened) |

**Examples:**

```bash
forma intervention update --store interventions.yaml --id 1 --outcome "개선"
```

---

## forma select

Select questions from a source test bank and generate an exam PDF, driven by `week.yaml`.

**Synopsis:**

```bash
forma select [--week-config <path>] [--no-config]
```

**Behavior:**

1. Locates `week.yaml` (via `--week-config` or auto-discovery by walking upward from CWD, stopping at `.git`).
2. Reads `select.source` and `select.questions` to extract questions by `sn` number.
3. Writes `questions.yaml` with provenance metadata (source path, selected `sn` list, week number).
4. If `select.exam_output` is set in `week.yaml`, generates an exam PDF automatically.

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--week-config` | path | no | auto-discover | Path to `week.yaml`; discovered by walking upward from CWD if omitted |
| `--no-config` | flag | no | false | Disable `week.yaml` auto-discovery |

**Examples:**

```bash
# Auto-discover week.yaml and generate questions.yaml + exam PDF
forma select

# Specify week.yaml path explicitly
forma select --week-config path/to/week/week.yaml
```

**Prerequisite**: A valid `week.yaml` with `select.source` and `select.questions` set. See [Configuration](configuration.md#weekyaml) for the full `week.yaml` field reference.

---

## forma lecture analyze

Analyze a single STT lecture transcript: keyword extraction, network generation, topic modeling, emphasis scoring, and optional triplet extraction.

**Synopsis:**

```bash
forma lecture analyze --input <path> --output <dir> --class <id> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--input` | path | yes | -- | STT transcript file path |
| `--output` | path | yes | -- | Output directory |
| `--class` | string | yes | -- | Class identifier |
| `--week` | integer | no | 0 | Week number |
| `--concepts` | path | no | None | Exam concepts YAML path (for concept-aware analysis) |
| `--no-cache` | flag | no | false | Skip loading cached analysis results |
| `--top-n` | integer | no | 50 | Top keyword count |
| `--no-triplets` | flag | no | false | Skip triplet extraction |
| `--extra-stopwords` | list | no | [] | Additional stopwords |

**Examples:**

```bash
# Analyze a single transcript
forma lecture analyze --input transcript_A_w1.txt --output results/lecture/ \
                      --class A --week 1

# With concept awareness and no triplets
forma lecture analyze --input transcript_A_w1.txt --output results/lecture/ \
                      --class A --week 1 --concepts exam.yaml --no-triplets
```

---

## forma lecture compare

Compare lecture transcripts across class sections for a single session (week). Produces exclusive keyword lists, concept gap analysis, and emphasis variance rankings.

**Synopsis:**

```bash
forma lecture compare --input-dir <dir> --week <n> --classes <ids...> --output <dir> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--input-dir` | path | yes | -- | Directory containing analysis result YAML files |
| `--week` | integer | yes | -- | Week number |
| `--classes` | list | yes | -- | Class identifiers to compare (at least 2) |
| `--output` | path | yes | -- | Output directory |
| `--concepts` | path | no | None | Exam concepts YAML path |
| `--top-n` | integer | no | 50 | Top keyword count |

**Examples:**

```bash
forma lecture compare --input-dir results/lecture/ --week 1 \
                      --classes A B C D --output results/comparison/
```

---

## forma lecture class-compare

Compare lecture transcripts across class sections for all sessions combined. Merges per-session analysis results into class-level summaries before comparison.

**Synopsis:**

```bash
forma lecture class-compare --input-dir <dir> --weeks <ns...> --classes <ids...> --output <dir> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--input-dir` | path | yes | -- | Directory containing analysis result YAML files |
| `--weeks` | list (int) | yes | -- | Week numbers to include |
| `--classes` | list | yes | -- | Class identifiers to compare (at least 2) |
| `--output` | path | yes | -- | Output directory |
| `--concepts` | path | no | None | Exam concepts YAML path |
| `--top-n` | integer | no | 50 | Top keyword count |
| `--no-cache` | flag | no | false | Skip loading cached merged analysis |

**Examples:**

```bash
forma lecture class-compare --input-dir results/lecture/ \
                            --weeks 1 2 3 --classes A B C D \
                            --output results/class_comparison/
```

---

## forma backfill longitudinal

Backfill the longitudinal store from existing evaluation result directories. Useful for retroactively populating longitudinal data from past assessments.

**Synopsis:**

```bash
forma backfill longitudinal --eval-dir <dir> [--eval-dir <dir> ...] --store <path> --week <n> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--eval-dir` | path | yes | -- | Evaluation output directory (repeatable) |
| `--store` | path | yes | -- | Path to longitudinal store YAML (created if not exists) |
| `--week` | integer | yes | -- | Week number |
| `--exam-file` | string | no | `""` | Exam config filename for metadata |
| `--responses` | path | no | None | Path to final YAML for OCR confidence (repeatable) |

**Examples:**

```bash
# Backfill week 1 from multiple class evaluation directories
forma backfill longitudinal \
    --eval-dir eval_A --eval-dir eval_B --eval-dir eval_C \
    --store longitudinal.yaml --week 1 \
    --exam-file Ch01_FormativeTest.yaml

# With OCR confidence data from join output
forma backfill longitudinal \
    --eval-dir eval_A --eval-dir eval_B \
    --store longitudinal.yaml --week 2 \
    --responses final_A.yaml --responses final_B.yaml
```

---

## forma domain extract

Extract domain concepts from textbook text files. Uses LLM-based extraction (v2) when `--model` or `--summary` is provided; falls back to KoNLPy word-level extraction (v1) otherwise.

**Synopsis:**

```bash
forma domain extract --output <path> (--textbook <path> | --summary <path>) [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--textbook` | path | conditional | None | Textbook chapter text file path (repeatable; required unless `--summary` used alone) |
| `--summary` | path | no | None | Chapter summary Markdown file path (repeatable; triggers LLM extraction) |
| `--output` | path | yes | -- | Output concepts YAML file path |
| `--min-freq` | integer | no | 2 | Minimum frequency (v1 only; bilingual terms always included) |
| `--no-cache` | flag | no | false | Disable concept cache |
| `--model` | string | no | None | LLM model ID override (triggers LLM extraction) |
| `--chunk` | flag | no | auto | Force chunk splitting (even for small files) |
| `--no-chunk` | flag | no | auto | Disable chunk splitting (single call even for large files) |

At least one of `--textbook` or `--summary` must be provided. `--chunk` and `--no-chunk` are mutually exclusive.

**Examples:**

```bash
# v1 extraction (KoNLPy)
forma domain extract --textbook ch01.txt --textbook ch02.txt --output concepts.yaml

# v2 extraction (LLM) with chapter summaries
forma domain extract --textbook ch01.txt --summary ch01_summary.md \
                     --output concepts.yaml --model gemini-pro
```

---

## forma domain coverage

Analyze lecture coverage against textbook concepts using embedding similarity, term matching, and optional LLM analysis.

**Synopsis:**

```bash
forma domain coverage --concepts <path> --transcripts <path> [--transcripts <path> ...] --output <path> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--concepts` | path | yes | -- | Concepts YAML file (from `forma domain extract`) |
| `--transcripts` | path | yes | -- | Lecture transcript file path (repeatable) |
| `--output` | path | yes | -- | Output coverage YAML file path |
| `--week-config` | path | no | None | Week config YAML (with teaching scope) |
| `--scope` | string | no | None | CLI scope override (e.g. `"2장:확산,능동수송;3장:"`) |
| `--threshold` | float | no | 0.65 | Similarity threshold |
| `--eval-store` | path | no | None | Longitudinal data YAML (for formative assessment linking) |
| `--model` | string | no | None | LLM model ID override |
| `--no-pedagogy` | flag | no | false | Skip pedagogy analysis |
| `--no-network` | flag | no | false | Skip network graph generation |
| `--no-llm` | flag | no | false | Skip LLM calls (use embedding/term/density signals only) |

**Examples:**

```bash
# Basic coverage analysis
forma domain coverage --concepts concepts.yaml \
                      --transcripts lecture_w1.txt --transcripts lecture_w2.txt \
                      --output coverage.yaml

# With teaching scope and LLM analysis
forma domain coverage --concepts concepts.yaml \
                      --transcripts lecture_w1.txt \
                      --output coverage.yaml \
                      --week-config week.yaml --model gemini-flash
```

---

## forma domain report

Generate a domain delivery analysis PDF report from coverage/delivery analysis results.

**Synopsis:**

```bash
forma domain report --coverage <path> --output <path> [options]
```

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--coverage` | path | yes | -- | Delivery analysis result YAML |
| `--output` | path | yes | -- | Output PDF file path |
| `--course-name` | string | no | `""` | Course name (displayed in report header) |
| `--font-path` | path | no | None | Korean font path |
| `--dpi` | integer | no | 150 | Chart resolution |
| `--concepts` | path | no | None | Concepts YAML file (for network graph) |
| `--summary` | path | no | None | Chapter summary Markdown file path (for hierarchy analysis) |

**Examples:**

```bash
# Basic report
forma domain report --coverage coverage.yaml --output domain_report.pdf

# With course name and concept network graph
forma domain report --coverage coverage.yaml --output domain_report.pdf \
                    --course-name "Biology 101" --concepts concepts.yaml
```
