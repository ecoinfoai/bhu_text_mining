# Layer 8: Comfortableness Audit Results

## Summary

| Check | Status | Findings |
|-------|--------|----------|
| CLI --help quality | WARN | 18/20 commands have good help; 2 subcommand groups lack descriptions |
| Error message actionability | WARN | ~15 error messages lack actionable next-step suggestions |
| Config file examples | FAIL | No `forma.json` example exists; `week.yaml.example` missing `lecture_*` fields |
| Documentation <> Code sync | FAIL | 8 mismatches between cli-reference.md and actual CLI flags |
| Output directory handling | PASS | All CLIs auto-create output dirs via `os.makedirs(exist_ok=True)` |
| Progress indicators | WARN | 3 batch commands lack progress feedback |
| Default value reasonableness | PASS | All defaults are reasonable; no surprise overwrites |

---

## 1. CLI --help Quality

| Command | Has Descriptions | Required/Optional Clear | Examples | Language | Notes |
|---------|:---:|:---:|:---:|--------|-------|
| `forma --help` | YES | YES | N/A | EN | Good overview of subcommands |
| `forma-exam --help` | YES | YES | NO | EN | OK — mutually exclusive group is clear |
| `forma-ocr --help` | YES | YES | NO | EN | Subcommand descriptions present |
| `forma-eval --help` | YES | YES | NO | EN | Deprecated `--skip-llm` noted |
| `forma-eval-batch --help` | YES | YES | NO | EN | Good |
| `forma-report --help` | YES | YES | NO | EN | `--intervention-log` noted as ignored (FR-013) |
| `forma-report-professor --help` | YES | YES | NO | EN | Good |
| `forma-report-batch --help` | YES | YES | NO | EN | Good |
| `forma-report-longitudinal --help` | YES | YES | NO | EN | Good; new v0.13 flags documented |
| `forma-init --help` | YES | YES | NO | EN | Minimal but sufficient |
| `forma-train --help` | YES | YES | NO | EN | Good |
| `forma-report-warning --help` | YES | YES | NO | EN | Good |
| `forma-intervention --help` | PARTIAL | YES | NO | EN | Top-level help lacks subcommand arg descriptions |
| `forma-train-grade --help` | YES | YES | NO | EN | Good |
| `forma-deliver --help` | PARTIAL | YES | NO | EN | Top-level help lacks subcommand arg descriptions |
| `forma-select --help` | YES | YES | NO | EN | Minimal — no description of what it does |
| `forma-backfill-longitudinal --help` | YES | YES | NO | EN | Good |
| `forma-report-student --help` | YES | YES | NO | EN | DEPRECATED `--week` noted |
| `forma-report-student-batch --help` | YES | YES | NO | EN | Good |
| `forma-report-student-summary --help` | YES | YES | NO | EN | Good |

### Observations

- **No usage examples in any `--help` output**: All 20 commands rely purely on flag descriptions. The docs (cli-reference.md) has examples but they're not embedded in argparse `epilog`. Severity: LOW.
- **`forma-select --help`**: Description says "Formative assessment question selection and exam PDF generation" but doesn't explain the week.yaml dependency. Severity: LOW.
- **`forma-intervention --help` / `forma-deliver --help`**: Top-level help for subcommand groups only shows subcommand names + one-line descriptions. Users must run `forma-intervention add --help` to see actual flags. This is standard argparse behavior but worth noting. Severity: LOW.
- **Language consistency**: All help text is in English. Consistent. PASS.

---

## 2. Error Message Actionability

### Good patterns (actionable)

| File | Line | Message Pattern | Verdict |
|------|------|-----------------|---------|
| `cli_report.py` | 177-180 | `"Error: --week requires --longitudinal-store"` | GOOD: tells user what to add |
| `cli_deliver.py` | 233 | `"Error: --retry-failed and --force cannot be used together."` | GOOD: explains conflict |
| `cli_report_batch.py` | 113 | `parser.error("--join-pattern must contain {class}")` | GOOD: explains what's wrong |

### Non-actionable error messages (missing next steps)

| File | Line | Message | Issue | Severity |
|------|------|---------|-------|----------|
| `cli_ocr.py` | 366 | `"Error: week.yaml not found."` | Doesn't suggest `--week-config` flag or where to place the file | MEDIUM |
| `cli_ocr.py` | 477 | `"Error: week.yaml not found."` | Same as above (duplicated in join subcommand) | MEDIUM |
| `cli_report.py` | 135 | `"Error: --dpi must be between 72 and 600"` | Doesn't show the invalid value that was passed | LOW |
| `cli_train.py` | 78 | `"Error: Store contains no records"` | Doesn't suggest how to populate the store | MEDIUM |
| `cli_train_grade.py` | 96 | `"Error: Store contains no records"` | Same — no suggestion to run `forma-backfill-longitudinal` first | MEDIUM |
| `cli_train_grade.py` | 117 | `"Error: Grade mapping is empty"` | Doesn't explain expected format | MEDIUM |
| `cli_ocr.py` | 557 | `"Error: --output is required in batch mode."` | Good message but goes to stdout not stderr | LOW |
| `cli_ocr.py` | 607-639 | Multiple error prints | Go to stdout, not stderr (inconsistent with other CLIs) | LOW |
| `cli_lecture.py` | 62 | `"Error: --input argument is required."` | Should be handled by argparse `required=True` instead | LOW |
| `cli_domain.py` | 329 | `"Error: Concept list is empty."` | Doesn't suggest where to define concepts | MEDIUM |
| `cli_report_warning.py` | 104-119 | `logger.error("...file not found: %s", ...)` | Uses logger.error (inconsistent with print-based errors in other CLIs) | LOW |

### Inconsistent error output channels

- **stderr (correct)**: `cli_report.py`, `cli_train.py`, `cli_train_grade.py`, `cli_deliver.py`, `cli_intervention.py`, `cli_lecture.py`, `cli_domain.py`, `cli_init.py`
- **stdout (incorrect)**: `cli_ocr.py` lines 366, 477, 557, 607-639
- **logger.error**: `cli_report_warning.py` (outputs to stderr via logging, but inconsistent style)

Severity: LOW (functional but inconsistent UX).

---

## 3. Config File Examples

### `week.yaml.example` vs `week_config.py`

| Field | In Example | In Code | Status |
|-------|:---:|:---:|--------|
| `week` | YES | YES | MATCH |
| `select.*` | YES | YES | MATCH |
| `ocr.*` | YES | YES | MATCH |
| `eval.*` | YES | YES | MATCH |
| `lecture_transcript_pattern` | NO | YES | **MISSING** |
| `lecture_concept_source` | NO | YES | **MISSING** |
| `lecture_output_dir` | NO | YES | **MISSING** |
| `lecture_extra_stopwords` | NO | YES | **MISSING** |
| `lecture_extra_abbreviations` | NO | YES | **MISSING** |

**Finding**: `week.yaml.example` is missing the entire `lecture:` section added in v0.12.4.
- File: `week.yaml.example`
- Severity: **MEDIUM** — users setting up lecture analysis won't have a template to follow.

### `forma.json` example

- **No example file exists** (`forma.json.example` not found in repo).
- `project_config.py` defines `ProjectConfiguration` dataclass with fields like `course_name`, `academic_year`, `semester`, `class_ids`, `base_dir`, etc.
- `cli_init.py` generates `forma.yaml` interactively, which is a different format from `forma.json`.
- **Impact**: Users must read source code or run `forma init` to understand the config structure.
- Severity: **LOW** (since `forma init` generates the file interactively).

---

## 4. Documentation <> Code Sync

### Flags in code but NOT in cli-reference.md

| Command | Flag | File:Line | Severity |
|---------|------|-----------|----------|
| `forma eval` | `--class` | `pipeline_evaluation.py:1337` | **HIGH** — critical workflow flag undocumented |
| `forma eval` | `--week-config` | `pipeline_evaluation.py:1342` | **HIGH** — critical workflow flag undocumented |
| `forma report longitudinal` | `--classes` | `cli_report_longitudinal.py:56` | **MEDIUM** — filtering flag undocumented |
| `forma report longitudinal` | `--heatmap-layout` | `cli_report_longitudinal.py:60` | **MEDIUM** — layout customization undocumented |
| `forma report longitudinal` | `--risk-threshold` | `cli_report_longitudinal.py:64` | **MEDIUM** — threshold override undocumented |
| `forma report longitudinal` | `--mastery-top-n` | `cli_report_longitudinal.py:69` | **LOW** — display tuning undocumented |

### Commands in code but NOT in cli-reference.md

| Command | Entry Point | Severity |
|---------|-------------|----------|
| `forma-report-student` | Standalone longitudinal student report | **MEDIUM** — 3 standalone commands undocumented |
| `forma-report-student-batch` | Batch student longitudinal reports | **MEDIUM** |
| `forma-report-student-summary` | All-student summary table PDF | **MEDIUM** |

Note: These may be accessible via `forma report student-individual` / `forma report student-batch` / `forma report student-summary` subcommands, but the **standalone entry points** are not documented.

### Docs claim "22 commands" (line 3) but actual count

- Standalone entry points: 20 (`forma-*`)
- Unified subcommands: ~25+ (including nested like `forma ocr scan/join/compare`, `forma lecture analyze/compare/class-compare`, `forma domain extract/coverage/report`)
- The "22" count in cli-reference.md appears approximately correct for the unified subcommand count, but the exact number is now slightly higher.

### quickstart.md sync issues

| Step | Docs Command | Current Code | Status |
|------|-------------|--------------|--------|
| Step 1 | `forma exam --config ...` | Matches | OK |
| Step 2 | `forma ocr scan --config ocr_config.yaml` | Code uses `--class` with week.yaml more commonly | WARN: example shows legacy `--config` path |
| Step 3 | `forma eval --config ... --responses ... --output ...` | Matches | OK |
| Step 4 | `forma report student --final ...` | Matches | OK |
| Step 5 | `forma report professor --final ...` | Matches | OK |
| Step 5 tip | `forma train risk` | Actual command: `forma train risk` or `forma-train` | OK |
| Step 6 | `forma deliver prepare/send` | Matches | OK |
| Prerequisites | `config.json` | Code uses `forma.json` (v0.10.0+) | **WARN**: docs reference old `config.json` path |

---

## 5. Output Directory Handling

All CLI commands that produce output files properly auto-create directories:

| CLI Module | Line | Pattern | Status |
|------------|------|---------|--------|
| `cli_report_professor.py` | 87 | `os.makedirs(args.output_dir, exist_ok=True)` | PASS |
| `cli_report_student.py` | 435 | `os.makedirs(args.output_dir, exist_ok=True)` | PASS |
| `pipeline_evaluation.py` | 1255 | `os.makedirs(reports_dir, exist_ok=True)` | PASS |
| `pipeline_batch_evaluation.py` | 96 | `os.makedirs(class_dir, exist_ok=True)` | PASS |
| `student_report.py` | 147 | `os.makedirs(output_dir, exist_ok=True)` | PASS |
| `report_generator.py` | 106, 239 | `os.makedirs(..., exist_ok=True)` | PASS |
| `warning_report.py` | 131 | `os.makedirs(output_dir, exist_ok=True)` | PASS |
| `delivery_prepare.py` | 382, 435, 469 | `os.makedirs(..., exist_ok=True)` | PASS |
| `delivery_send.py` | 440 | `os.makedirs(dir_name, exist_ok=True)` | PASS |
| `ocr_pipeline.py` | 525 | `os.makedirs(out_dir, exist_ok=True)` | PASS |

**Verdict**: All output paths auto-create directories. No user-facing errors from missing directories. PASS.

---

## 6. Progress Indicators

### Commands WITH progress indicators

| Command | Module | Line | Pattern | Status |
|---------|--------|------|---------|--------|
| `forma eval` | `pipeline_evaluation.py` | 157, 287, 361, 859, 957, 1260 | `\r[pipeline] ...` inline progress | GOOD |
| `forma ocr scan` | `llm_ocr.py` | 317, 370, 379, 426, 440 | `\r [N/total]` inline progress | GOOD |
| `forma ocr scan` | `ocr_pipeline.py` | 151 | `\r [pct%] processed/total` | GOOD |
| `forma-report-student-batch` | `cli_report_student.py` | 554, 557 | `\r[report] N/total (id)` | GOOD |

### Commands WITHOUT progress indicators

| Command | Module | Impact | Severity |
|---------|--------|--------|----------|
| `forma-report-batch` | `cli_report_batch.py` | Processes N classes x M students — can take minutes with no output | **MEDIUM** |
| `forma deliver prepare` | `delivery_prepare.py` | Processes all students, creates zip files — silent | **LOW** |
| `forma deliver send` | `delivery_send.py` | Sends N emails — no per-email progress | **MEDIUM** |
| `forma-eval-batch` | `pipeline_batch_evaluation.py` | Each class shows pipeline progress, but no inter-class progress indicator | **LOW** |
| `forma lecture compare` | `cli_lecture.py` | Processes multiple transcripts — silent | **LOW** |

---

## 7. Default Value Reasonableness

| CLI | Flag | Default | Assessment |
|-----|------|---------|------------|
| Multiple | `--dpi` | 150 | GOOD: reasonable for screen/print balance |
| `forma-train` | `--threshold` | 0.45 | GOOD: matches documented concept threshold |
| `forma-train` | `--min-weeks` | 3 | GOOD: minimum for trend detection |
| `forma-train` / `forma-train-grade` | `--min-students` | 10 | GOOD: statistical minimum |
| `forma-report-longitudinal` | `--risk-threshold` | 0.45 | GOOD: consistent with train default |
| `forma-init` | `--output` | `forma.yaml` | GOOD: conventional name |
| `forma-init` | `--force` | false | GOOD: prevents accidental overwrite |
| `forma-deliver prepare` | `--force` | false | GOOD: prevents accidental overwrite |
| `forma-deliver send` | `--dry-run` | false | OK: live send is the expected default |
| `forma-deliver send` | `--force` | false | GOOD: prevents re-sending |
| `forma eval` | `--n-calls` | 3 | GOOD: 3-call reliability protocol |
| `forma ocr join` | `--credentials` | `credentials.json` | WARN: assumes file in CWD, but documented |
| `forma lecture analyze` | `--top-n` | 50 | GOOD: reasonable keyword count |

**No surprising defaults found.** All boolean flags default to `false`. All overwrite operations require `--force`. PASS.

---

## Severity Summary

| Severity | Count | Key Issues |
|----------|-------|------------|
| HIGH | 2 | `--class` and `--week-config` flags for `forma eval` undocumented in cli-reference.md |
| MEDIUM | 10 | Missing `lecture:` section in week.yaml.example; 5 undocumented longitudinal flags; 3 non-actionable error messages; 2 batch commands lack progress; 3 standalone commands undocumented |
| LOW | 12 | No argparse examples; inconsistent error channels (stdout vs stderr); minor doc count mismatch; logger vs print inconsistency |

### C3R Category: Comfortableness

All findings relate to user experience quality:
- **Discoverability**: Undocumented flags force users to read source code
- **Learnability**: Missing config examples slow onboarding
- **Feedback**: Silent batch operations leave users uncertain about progress
- **Consistency**: Mixed error output channels (stdout/stderr/logger) create unpredictable UX
- **Actionability**: Error messages that say "not found" without suggesting next steps require users to check docs
