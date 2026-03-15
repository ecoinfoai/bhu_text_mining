# Configuration Reference

Formative-analysis uses two configuration files: `config.json` for credentials and system-level settings, and `forma.yaml` for project-level settings. This document describes every field in both files.

## Table of Contents

- [config.json](#configjson)
- [forma.yaml](#formayaml)
- [week.yaml](#weekyaml)
- [Credential Security](#credential-security)
- [Generating forma.yaml with forma init](#generating-formayaml-with-forma-init)

---

## config.json

**Location:** `~/.config/formative-analysis/config.json`

Stores credentials and system-level settings that apply across all projects. This file is searched in the following order:

1. Explicit path argument (if provided)
2. `/run/agenix/forma-config` (NixOS agenix)
3. `~/.config/formative-analysis/config.json`
4. `~/.config/formative-analysis/forma.json` (deprecated — will be removed)

### smtp section

SMTP server settings for email delivery (`forma deliver send`).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `server` | string | yes | -- | SMTP server hostname (e.g., `smtp.gmail.com`) |
| `port` | integer | yes | -- | SMTP port number (587 for STARTTLS, 465 for SSL) |
| `sender_email` | string | yes | -- | Sender email address |
| `sender_name` | string | no | `""` | Display name shown to recipients |
| `use_tls` | boolean | no | `true` | Use STARTTLS encryption |
| `send_interval_sec` | number | no | `1.0` | Seconds to wait between emails |

> **Note:** The SMTP password is never stored in config.json. See [Credential Security](#credential-security) for how to supply it at runtime.

### llm section

LLM provider settings for feedback generation and analysis.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `provider` | string | no | `"gemini"` | LLM provider (`"gemini"` or `"anthropic"`) |
| `model` | string | no | -- | Model name (e.g., `gpt-4o`, `claude-sonnet-4-6`) |
| `api_key` | string | no | -- | API key (recommend using environment variables instead) |

### naver_ocr section

Naver CLOVA OCR credentials for answer sheet scanning.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_url` | string | no | -- | Naver CLOVA OCR API endpoint URL |
| `secret_key` | string | no | -- | Naver OCR secret key |

### Complete config.json example

```json
{
  "smtp": {
    "server": "smtp.gmail.com",
    "port": 587,
    "sender_email": "professor@university.edu",
    "sender_name": "Prof. Kim",
    "use_tls": true,
    "send_interval_sec": 1.0
  },
  "llm": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "api_key": "sk-ant-..."
  },
  "naver_ocr": {
    "api_url": "https://...",
    "secret_key": "..."
  }
}
```

---

## forma.yaml

**Location:** Auto-discovered by searching from the current working directory upward (stops at `.git` sentinel or filesystem root).

Stores project-level settings. All fields are optional and have sensible defaults.

**Resolution order:** CLI flags (highest priority) > forma.yaml > system defaults (argparse defaults)

### project section

Course metadata used in report headers and file naming.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `course_name` | string | no | `""` | Course name (e.g., "Human Anatomy") |
| `year` | integer | no | `0` | Academic year (must be >= 2020 when set) |
| `semester` | integer | no | `0` | Semester number (must be 1 or 2) |
| `grade` | integer | no | `0` | Student grade year (must be >= 1 when set) |

### classes section

Section/class configuration for multi-class batch processing.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `identifiers` | list | no | `[]` | List of class section identifiers (e.g., `[A, B, C, D]`) |
| `join_pattern` | string | no | `""` | File pattern for joined data; must contain `{class}` placeholder |
| `eval_pattern` | string | no | `""` | Directory pattern for evaluation results; must contain `{class}` placeholder |

### paths section

File and directory paths for input data and output.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `exam_config` | string | no | `""` | Path to exam configuration YAML file |
| `join_dir` | string | no | `""` | Path to joined data directory |
| `output_dir` | string | no | `""` | Path to PDF output directory |
| `longitudinal_store` | string | no | `""` | Path to longitudinal store YAML file |
| `font_path` | string/null | no | `null` | Path to Korean font file (`null` for auto-detection) |

### ocr section

OCR scanning settings for answer sheet digitization.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `ocr_model` | string/null | no | `null` | OCR model ID (e.g., `gemini-2.0-flash`; `null` uses provider default) |
| `naver_config` | string | no | `""` | Path to Naver OCR configuration (deprecated) |
| `credentials` | string | no | `""` | Credentials reference (resolved from environment variable) |
| `spreadsheet_url` | string | no | `""` | Google Sheets URL for OCR data |
| `num_questions` | integer | no | `5` | Number of questions per exam (must be >= 1) |

### evaluation section

LLM evaluation pipeline settings.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `provider` | string | no | `"gemini"` | LLM provider (`"gemini"` or `"anthropic"`) |
| `model` | string/null | no | `null` | LLM model name (`null` uses provider default) |
| `skip_feedback` | boolean | no | `false` | Skip feedback generation |
| `skip_graph` | boolean | no | `false` | Skip graph comparison |
| `skip_statistical` | boolean | no | `false` | Skip statistical analysis |
| `n_calls` | integer | no | `3` | Number of LLM calls per item (must be >= 1) |

### reports section

PDF report generation settings.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `dpi` | integer | no | `150` | Chart image resolution in DPI (72--600) |
| `skip_llm` | boolean | no | `false` | Skip all LLM analysis in reports |
| `aggregate` | boolean | no | `true` | Generate aggregate (cross-class) report |

### prediction section

Machine learning model settings.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_path` | string/null | no | `null` | Path to pre-trained risk prediction model (`.pkl` file) |

### Top-level fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `current_week` | integer | no | `1` | Current week number (must be >= 1) |

### Complete forma.yaml example

```yaml
project:
  course_name: "Human Anatomy"
  year: 2026
  semester: 1
  grade: 1

classes:
  identifiers: [A, B, C, D]
  join_pattern: "anp_{class}_final.yaml"
  eval_pattern: "eval_{class}/"

paths:
  exam_config: "exams/Ch01_FormativeTest.yaml"
  join_dir: "results/anp_w1"
  output_dir: "reports/"
  longitudinal_store: "store/longitudinal.yaml"
  font_path: null

ocr:
  ocr_model: null
  naver_config: ""
  credentials: ""
  spreadsheet_url: ""
  num_questions: 5

evaluation:
  provider: "gemini"
  model: null
  skip_feedback: false
  skip_graph: false
  skip_statistical: false
  n_calls: 3

reports:
  dpi: 150
  skip_llm: false
  aggregate: true

prediction:
  model_path: null

current_week: 1
```

---

## week.yaml

**Location:** One `week.yaml` per assessment week directory (e.g., `week_01/week.yaml`).

**Discovery:** `forma ocr`, `forma eval`, and `forma select` walk upward from the current working directory to find `week.yaml`, stopping at `.git` or the filesystem root. Override with `--week-config <path>` or disable with `--no-config`.

**Priority:** `week.yaml` takes precedence over `forma.yaml` for week-specific settings.

**Config merge precedence** (highest → lowest):

| Priority | Source | Scope |
|----------|--------|-------|
| 1 | CLI flags | Overrides everything |
| 2 | `week.yaml` | Per-week settings |
| 3 | `forma.yaml` | Per-semester settings |
| 4 | Argparse defaults | Fallback |

### Complete week.yaml example

```yaml
week: 1                                  # required — week number (>= 1)

select:
  source: "../exams/Ch01_FormativeTest.yaml"  # path to FormativeTest YAML source file
  questions: [1, 3]                      # sn numbers to extract
  num_papers: 50                         # number of exam paper copies to print
  form_url: ""                           # Google Forms pre-filled URL template
  exam_output: "week_01_exam.pdf"        # PDF output filename (triggers PDF generation)

ocr:
  num_questions: 2                       # answer areas per sheet
  image_dir_pattern: "scans_1{class}_w1" # image directory with {class} placeholder
  ocr_output_pattern: "scans_1{class}_w1/ocr_results.yaml"
  join_output_pattern: "scans_1{class}_w1/final.yaml"
  join_forms_csv: "forms_responses_w1.csv"  # CSV fallback for Google Forms data
  student_id_column: "student_id"        # column name for student ID
  crop_coords: []                        # auto-populated on first interactive scan

eval:
  config: "../exams/Ch01_FormativeTest.yaml"
  questions_used: [1, 3]                 # sn numbers used in evaluation
  responses_pattern: "scans_1{class}_w1/final.yaml"
  output_pattern: "scans_1{class}_w1/eval/"
  skip_feedback: false
  skip_graph: false
  generate_reports: false                # auto-generate student PDF reports
```

### week.yaml field reference

For the complete schema with all 21 fields, types, and descriptions, see [Data Formats — Week Configuration YAML](data-formats.md#week-configuration-yaml-weekyaml).

---

## Credential Security

The SMTP password is **never** stored in `config.json` or any configuration file. It must be supplied at runtime through one of two methods:

### 1. Standard input (`--password-from-stdin`)

Pipe the password into the `forma deliver send` command:

```bash
echo "mypassword" | forma deliver send --password-from-stdin --staged ./staging --template template.yaml
```

### 2. Environment variable (`FORMA_SMTP_PASSWORD`)

Export the password as an environment variable before running the command:

```bash
export FORMA_SMTP_PASSWORD="mypassword"
forma deliver send --staged ./staging --template template.yaml
```

If neither method is used, `forma deliver send` will exit with an error (unless `--dry-run` is specified).

### API keys

For LLM and OCR API keys, it is recommended to use environment variables rather than storing them directly in `config.json`. If stored in `config.json`, ensure the file has restrictive permissions (`chmod 600`).

---

## Generating forma.yaml with forma init

The `forma init` command creates a `forma.yaml` template in the current directory through an interactive wizard.

### Usage

```bash
forma init [--output PATH] [--force]
```

| Flag | Description |
|------|-------------|
| `--output PATH` | Output file path (default: `./forma.yaml`) |
| `--force` | Overwrite existing file |

### Interactive prompts

When you run `forma init`, you will be asked for:

1. **Course name** -- e.g., "Human Anatomy"
2. **Academic year** -- e.g., 2026
3. **Semester** -- 1 or 2
4. **Class identifiers** -- comma-separated, e.g., `A,B,C,D`

The generated file includes all sections with default values and inline comments explaining each field. You can then edit the file to fill in paths and adjust settings as needed.

### Example

```bash
$ forma init
Course name: Human Anatomy
Academic year: 2026
Semester: 1
Class identifiers: A,B,C,D
Configuration template generated: forma.yaml
```

For detailed YAML schema specifications, see [Data Formats](data-formats.md).
