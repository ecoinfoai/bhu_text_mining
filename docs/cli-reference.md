# CLI Reference

Complete reference for all 15 FormA CLI commands, accessible through the unified `forma` entry point.

> **Note:** The deprecated `forma-*` command names (e.g., `forma-exam`, `forma-train`) remain functional but emit a `DeprecationWarning`. Migrate to the `forma <subcommand>` format shown in this reference.

## Table of Contents

- [forma exam](#forma-exam)
- [forma ocr](#forma-ocr)
- [forma eval](#forma-eval)
- [forma eval batch](#forma-eval-batch)
- [forma report student](#forma-report-student)
- [forma report professor](#forma-report-professor)
- [forma report batch](#forma-report-batch)
- [forma report longitudinal](#forma-report-longitudinal)
- [forma report warning](#forma-report-warning)
- [forma train risk](#forma-train-risk)
- [forma train grade](#forma-train-grade)
- [forma init](#forma-init)
- [forma deliver](#forma-deliver)
- [forma intervention](#forma-intervention)
- [forma select](#forma-select)

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Input error (invalid arguments, missing required flags, data error) |
| 2 | File error (file not found, permission denied, write error) |
| 3 | Rendering error (font missing) or partial failure (some emails failed) |

> Accurate as of v0.12.2

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

OCR pipeline for scanned exam answer sheets. Has two subcommands: `scan` and `join`.

**Synopsis:**

```bash
forma ocr [--no-config] scan --config <path> [options]
forma ocr [--no-config] join --ocr-results <path> --output <path> [options]
```

### forma ocr scan

Scan images, decode QR codes, run OCR, and produce a YAML result file.

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--config` | path | yes | -- | OCR configuration YAML file path |
| `--num-questions` | integer | no | None | Number of questions (can be specified in config YAML; defaults to 2 if unset) |
| `--class` | string | no | None | Class identifier; substitutes `{class}` in `week.yaml` path patterns |

**Examples:**

```bash
forma ocr scan --config ocr_config.yaml
forma ocr scan --config ocr_config.yaml --num-questions 3
```

### forma ocr join

Join OCR results with Google Forms/Sheets responses to produce a unified output YAML.

**Arguments:**

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--ocr-results` | path | yes | -- | OCR results YAML file path |
| `--output` | path | yes | -- | Output YAML file path |
| `--spreadsheet-url` | string | no | None | Google Sheets URL (preferred data source) |
| `--forms-csv` | path | no | None | Google Forms CSV file path (fallback) |
| `--credentials` | path | no | `credentials.json` | OAuth2 credentials JSON file path |
| `--manual-mapping` | path | no | None | Manual mapping YAML for unmatched students |
| `--student-id-column` | string | no | `student_id` | Student ID column name in the spreadsheet |
| `--class` | string | no | None | Class identifier; substitutes `{class}` in `week.yaml` path patterns |

At least one of `--spreadsheet-url` or `--forms-csv` must be provided.

**Examples:**

```bash
forma ocr join --ocr-results results.yaml --output final.yaml \
               --spreadsheet-url "https://docs.google.com/spreadsheets/d/XXX"

forma ocr join --ocr-results results.yaml --output final.yaml \
               --forms-csv fallback.csv --manual-mapping mapping.yaml
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

Either `--eval-config` or all three of `--config`, `--responses`, `--output` must be provided. CLI flags override values from the eval-config YAML.

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
