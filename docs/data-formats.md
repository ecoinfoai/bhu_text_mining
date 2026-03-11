# Data Formats Reference

This document describes all YAML (and JSON) file schemas used by formative-analysis,
covering professor-authored input files, pipeline-generated output files, and
internal data stores.

## Summary

| File Type | Typical Location | Format | Primary Consumer | Auto-Generated |
|-----------|-----------------|--------|-----------------|----------------|
| Exam Configuration | `exams/*.yaml` | YAML | `forma-eval` | No |
| Grade Mapping | `grades/grade_mapping.yaml` | YAML | `forma-train-grade` | No |
| Student Roster | `delivery/roster.yaml` | YAML | `forma-deliver prepare` | No |
| Delivery Manifest | `delivery/manifest.yaml` | YAML | `forma-deliver prepare` | No |
| Email Template | `delivery/template.yaml` | YAML | `forma-deliver send` | No |
| SMTP Configuration (deprecated) | `smtp.yaml` | YAML | `forma-deliver send` | No |
| Credentials (forma.json) | `~/.config/formative-analysis/forma.json` | JSON | All CLI commands | No |
| Evaluation Results | `results/*/eval_*/res_lvl4/*.yaml` | YAML | `forma-report`, `forma-report-professor` | Yes |
| Longitudinal Store | `longitudinal.yaml` | YAML | `forma-report-longitudinal`, `forma-train` | Yes |
| Intervention Log | `intervention_log.yaml` | YAML | `forma-intervention`, `forma-report-professor` | Yes |
| Prepare Summary | `staging/prepare_summary.yaml` | YAML | `forma-deliver send` | Yes |
| Delivery Log | `staging/delivery_log.yaml` | YAML | Reference (audit trail) | Yes |
| Project Configuration | `forma.yaml` | YAML | All CLI commands | No (template via `forma-init`) |

## Table of Contents

- [Exam Configuration YAML](#exam-configuration-yaml)
- [Grade Mapping YAML](#grade-mapping-yaml)
- [Student Roster YAML](#student-roster-yaml)
- [Delivery Manifest YAML](#delivery-manifest-yaml)
- [Email Template YAML](#email-template-yaml)
- [SMTP Configuration YAML](#smtp-configuration-yaml-deprecated)
- [Credentials JSON (forma.json)](#credentials-json-formajson)
- [Evaluation Results YAML](#evaluation-results-yaml)
- [Longitudinal Store YAML](#longitudinal-store-yaml)
- [Intervention Log YAML](#intervention-log-yaml)
- [Prepare Summary YAML](#prepare-summary-yaml)
- [Delivery Log YAML](#delivery-log-yaml)
- [Project Configuration YAML](#project-configuration-yaml-formayaml)

---

### Exam Configuration YAML

**Purpose**: Defines the formative test structure including questions, model answers, rubrics, and support guidance for each question.

**Created by**: Manual (professor)

**Consumed by**: `forma-eval`, `forma-report`, `forma-report-professor`

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `metadata` | object | Yes | - | Exam metadata section |
| `metadata.chapter` | int | Yes | - | Chapter number |
| `metadata.chapter_name` | string | Yes | - | Chapter title |
| `metadata.course_name` | string | Yes | - | Course name |
| `metadata.year` | int | Yes | - | Academic year |
| `metadata.grade` | int | Yes | - | Student grade year |
| `metadata.semester` | int | Yes | - | Semester number (1 or 2) |
| `metadata.week_num` | int | Yes | - | Week number |
| `metadata.answer_limit` | string | Yes | - | Answer length guidance |
| `metadata.total_questions` | int | Yes | - | Number of questions |
| `metadata.generated_date` | string | Yes | - | Date the exam was created |
| `questions` | list | Yes | - | List of question objects |
| `questions[].sn` | int | Yes | - | Question serial number |
| `questions[].topic` | string | Yes | - | Question topic category |
| `questions[].question` | string | Yes | - | Question text |
| `questions[].limit` | string | Yes | - | Per-question answer length guidance |
| `questions[].model_answer` | string | Yes | - | Model (reference) answer |
| `questions[].purpose` | string | Yes | - | Educational purpose of the question |
| `questions[].keywords` | list[string] | Yes | - | Key concepts expected in the answer |
| `questions[].rubric` | object | Yes | - | Scoring rubric with `high`, `mid`, `low` |
| `questions[].rubric.high` | string | Yes | - | Criteria for high-level understanding |
| `questions[].rubric.mid` | string | Yes | - | Criteria for mid-level understanding |
| `questions[].rubric.low` | string | Yes | - | Criteria for low-level understanding |
| `questions[].support` | object | Yes | - | Support guidance per rubric tier |
| `questions[].support.high` | string | Yes | - | Enrichment guidance for high-level students |
| `questions[].support.mid` | string | Yes | - | Remediation guidance for mid-level students |
| `questions[].support.low` | string | Yes | - | Intervention guidance for low-level students |
| `concept_dependencies` | list | No | None | Optional prerequisite relationships |
| `concept_dependencies[].prerequisite` | string | Yes | - | Prerequisite concept name |
| `concept_dependencies[].dependent` | string | Yes | - | Dependent concept name |
| `pdf_questions` | list | No | None | Simplified question list for PDF rendering |

**Example:**

```yaml
metadata:
  chapter: 1
  chapter_name: Introduction
  course_name: Human Anatomy
  year: 2026
  grade: 1
  semester: 1
  week_num: 1
  answer_limit: "200 characters"
  total_questions: 2
  generated_date: '2026-02-17'

questions:
- sn: 1
  topic: Concept Understanding
  question: "Explain homeostasis and negative feedback."
  limit: "200 characters"
  model_answer: "Homeostasis is the maintenance of a stable internal environment..."
  purpose: "Assess understanding of core homeostasis concepts."
  keywords:
  - homeostasis
  - negative feedback
  - receptor
  rubric:
    high: "Accurately describes homeostasis and feedback loop components."
    mid: "Basic understanding but missing key components."
    low: "Does not understand homeostasis concept."
  support:
    high: "Research additional feedback examples."
    mid: "Review textbook diagrams of the feedback loop."
    low: "Use thermostat analogy for 1:1 tutoring."

concept_dependencies:
- prerequisite: "receptor"
  dependent: "integration center"
- prerequisite: "integration center"
  dependent: "effector"
```

---

### Grade Mapping YAML

**Purpose**: Maps student IDs to letter grades for each semester, used to train the grade prediction model.

**Created by**: Manual (professor)

**Consumed by**: `forma-train-grade`

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `<semester_label>` | object | Yes | - | Top-level key is the semester label (e.g., `"2025-1"`) |
| `<semester_label>.<student_id>` | string | Yes | - | Letter grade; must be one of `A`, `B`, `C`, `D`, `F` |

The file is a flat mapping of semester labels to student-grade pairs. Multiple semesters can be included for grade trend analysis.

**Example:**

```yaml
2025-1:
  S001: A
  S002: B
  S003: C
  S004: F

2025-2:
  S001: A
  S002: A
  S003: B
  S004: D
```

---

### Student Roster YAML

**Purpose**: Lists students in a class section with their contact information, used for email delivery.

**Created by**: Manual (professor)

**Consumed by**: `forma-deliver prepare`

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `class_name` | string | Yes | - | Class section name (e.g., `"1A"`) |
| `students` | list | Yes | - | List of student entry objects (at least 1) |
| `students[].student_id` | string | Yes | - | Unique student identifier |
| `students[].name` | string | Yes | - | Student display name |
| `students[].email` | string | No | `""` | Student email address; invalid or missing emails cause `"error"` status during prepare |

Student IDs must be unique within the roster. Duplicate IDs raise a `ValueError`.

**Example:**

```yaml
class_name: "1A"
students:
- student_id: "S001"
  name: "Kim Minjun"
  email: "minjun@example.com"
- student_id: "S002"
  name: "Lee Soyeon"
  email: "soyeon@example.com"
- student_id: "S003"
  name: "Park Jihun"
  email: ""
```

---

### Delivery Manifest YAML

**Purpose**: Defines where student report files are located and how to match them to individual students.

**Created by**: Manual (professor)

**Consumed by**: `forma-deliver prepare`

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `report_source` | object | Yes | - | Report source configuration section |
| `report_source.directory` | string | Yes | - | Path to the directory containing report files; must exist on disk |
| `report_source.file_patterns` | list[string] | Yes | - | Filename templates with `{student_id}` placeholder (at least 1) |

Each pattern in `file_patterns` must contain the literal string `{student_id}`, which is substituted with the actual student ID during file matching.

**Example:**

```yaml
report_source:
  directory: "output/reports/week3"
  file_patterns:
  - "{student_id}_report.pdf"
  - "{student_id}_feedback.pdf"
```

---

### Email Template YAML

**Purpose**: Defines the email subject and body templates for delivering reports to students.

**Created by**: Manual (professor)

**Consumed by**: `forma-deliver send`

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `subject` | string | Yes | - | Email subject line with optional template variables |
| `body` | string | Yes | - | Email body (plain text) with optional template variables |

Supported template variables (using `{variable_name}` syntax):

| Variable | Description |
|----------|-------------|
| `{student_name}` | Student display name from roster |
| `{student_id}` | Student identifier from roster |
| `{class_name}` | Class section name from roster |

Template rendering uses safe `str.replace()` (not `str.format()`) to prevent format string injection.

**Example:**

```yaml
subject: "[{class_name}] Formative Assessment Results"
body: |
  Dear {student_name},

  Your formative assessment results for {class_name} are attached.
  Please review the feedback carefully.

  Best regards,
  Professor
```

---

### SMTP Configuration YAML (Deprecated)

**Purpose**: Defines SMTP server connection settings for email delivery. As of v0.11.1, this file is deprecated in favor of the `smtp` section in `forma.json`.

**Created by**: Manual (professor)

**Consumed by**: `forma-deliver send` (via `--smtp-config` flag)

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `smtp_server` | string | Yes | - | SMTP server hostname |
| `smtp_port` | int | No | `587` | SMTP server port (1-65535) |
| `sender_email` | string | Yes | - | Sender email address (must contain `@`) |
| `sender_name` | string | No | `""` | Display name for sender |
| `use_tls` | bool | No | `true` | Whether to use STARTTLS |
| `send_interval_sec` | float | No | `1.0` | Minimum seconds between sends (rate limiting) |

The SMTP password is never stored in this file. It must be provided via the `FORMA_SMTP_PASSWORD` environment variable or `--password-from-stdin`.

**Example:**

```yaml
smtp_server: "smtp.example.com"
smtp_port: 587
sender_email: "professor@example.com"
sender_name: "Prof. Kim"
use_tls: true
send_interval_sec: 1.0
```

---

### Credentials JSON (forma.json)

**Purpose**: Centralized credentials and service configuration file. Stores API keys, SMTP settings, and OCR configuration.

**Created by**: Manual (system administrator or professor)

**Consumed by**: All CLI commands (via `config.load_config()`)

**Location resolution order:**

1. Explicit path via CLI argument
2. `/run/agenix/forma-config` (NixOS agenix)
3. `~/.config/formative-analysis/forma.json`
4. `~/.config/forma/config.json` (legacy)
5. `~/.config/bhu_text_mining/config.json` (legacy)
6. `~/.config/naver_ocr/naver_ocr_config.json` (legacy)

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `smtp` | object | No | - | SMTP server configuration section |
| `smtp.server` | string | Yes* | - | SMTP server hostname |
| `smtp.port` | int | No | `587` | SMTP server port (1-65535) |
| `smtp.sender_email` | string | Yes* | - | Sender email address |
| `smtp.sender_name` | string | No | `""` | Display name for sender |
| `smtp.use_tls` | bool | No | `true` | Whether to use STARTTLS |
| `smtp.send_interval_sec` | float | No | `1.0` | Minimum seconds between sends |
| `naver_ocr` | object | No | - | Naver OCR API configuration |
| `naver_ocr.secret_key` | string | Yes* | - | Naver OCR API secret key |
| `naver_ocr.api_url` | string | Yes* | - | Naver OCR API endpoint URL |
| `llm` | object | No | - | LLM provider configuration |
| `llm.provider` | string | No | `"gemini"` | LLM provider (`"gemini"` or `"anthropic"`) |
| `llm.api_key` | string | No | - | LLM API key |
| `llm.model` | string | No | - | LLM model name override |

\* Required only when the corresponding feature is used.

The SMTP password is never stored in `forma.json`. Use `FORMA_SMTP_PASSWORD` environment variable or `--password-from-stdin`.

Note: The `smtp` section uses different field names than the YAML format. The JSON field `server` maps to `smtp_server`, and `port` maps to `smtp_port`.

**Example:**

```json
{
  "smtp": {
    "server": "smtp.example.com",
    "port": 587,
    "sender_email": "professor@example.com",
    "sender_name": "Prof. Kim",
    "use_tls": true,
    "send_interval_sec": 1.0
  },
  "naver_ocr": {
    "secret_key": "your-secret-key",
    "api_url": "https://your-ocr-endpoint.apigw.ntruss.com/..."
  },
  "llm": {
    "provider": "gemini",
    "api_key": "your-api-key",
    "model": "gemini-2.0-flash"
  }
}
```

---

### Evaluation Results YAML

**Purpose**: Stores per-student evaluation scores, concept analysis, LLM feedback, and statistical results generated by the evaluation pipeline.

**Created by**: `forma-eval`

**Consumed by**: `forma-report`, `forma-report-professor`, `forma-report-longitudinal`

The pipeline produces three result files under `res_lvl4/`:

#### ensemble_results.yaml

Per-student ensemble scores and component breakdowns.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `students` | list | Yes | - | List of student result objects |
| `students[].student_id` | string | Yes | - | Student identifier |
| `students[].questions` | list | Yes | - | List of per-question results |
| `students[].questions[].question_sn` | int | Yes | - | Question serial number |
| `students[].questions[].ensemble_score` | float | Yes | - | Weighted ensemble score (0.0-1.0) |
| `students[].questions[].understanding_level` | string | Yes | - | Level: `"Advanced"`, `"Proficient"`, `"Developing"`, or `"Beginning"` |
| `students[].questions[].component_scores` | object | Yes | - | Per-metric scores before weighting |
| `students[].questions[].component_scores.concept_coverage` | float | Yes | - | Concept presence coverage ratio |
| `students[].questions[].component_scores.llm_rubric` | float | Yes | - | LLM rubric normalized score |
| `students[].questions[].component_scores.rasch_ability` | float | Yes | - | Rasch IRT ability estimate |

#### technical_report.yaml

Detailed technical analysis with concept-level details, LLM evaluation, and statistical analysis.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `students` | list | Yes | - | List of student result objects |
| `students[].questions[].ensemble_score` | float | Yes | - | Weighted ensemble score |
| `students[].questions[].understanding_level` | string | Yes | - | Understanding level classification |
| `students[].questions[].component_scores` | object | Yes | - | Component score breakdown |
| `students[].questions[].weights_used` | object | Yes | - | Weights applied to each component |
| `students[].questions[].concept_details` | list | Yes | - | Per-concept match results |
| `students[].questions[].concept_details[].concept` | string | Yes | - | Concept term |
| `students[].questions[].concept_details[].is_present` | bool | Yes | - | Whether concept was detected |
| `students[].questions[].concept_details[].similarity` | float | Yes | - | Cosine similarity score |
| `students[].questions[].concept_details[].threshold` | float | Yes | - | Adaptive threshold used |
| `students[].questions[].llm_evaluation` | object | Yes | - | Aggregated LLM evaluation |
| `students[].questions[].llm_evaluation.median_score` | float | Yes | - | Median rubric score across calls |
| `students[].questions[].llm_evaluation.label` | string | Yes | - | Rubric label (`"high"`, `"mid"`, `"low"`) |
| `students[].questions[].llm_evaluation.reasoning` | string | Yes | - | LLM reasoning text |
| `students[].questions[].llm_evaluation.misconceptions` | list[string] | Yes | - | Detected misconceptions |
| `students[].questions[].llm_evaluation.uncertain` | bool | Yes | - | Whether LLM flagged low confidence |
| `students[].questions[].llm_evaluation.icc_value` | float | No | - | ICC(2,1) inter-rater reliability |
| `students[].questions[].statistical_analysis` | object | No | - | Rasch IRT and LCA results |
| `students[].questions[].statistical_analysis.rasch_theta` | float | No | - | Estimated person ability (WLE) |
| `students[].questions[].statistical_analysis.rasch_theta_se` | float | No | - | Standard error of theta |
| `students[].questions[].statistical_analysis.lca_class` | int | No | - | Assigned latent class (0-based) |
| `students[].questions[].statistical_analysis.lca_class_probability` | float | No | - | Posterior probability |
| `students[].questions[].statistical_analysis.lca_exploratory_warning` | string | No | - | Mandatory warning for N < 60 |

#### counseling_summary.yaml

Student-facing feedback and counseling information.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `students` | list | Yes | - | List of student result objects |
| `students[].questions[].question_sn` | int | Yes | - | Question serial number |
| `students[].questions[].understanding_level` | string | Yes | - | Understanding level |
| `students[].questions[].concept_coverage` | float | Yes | - | Concept coverage ratio |
| `students[].questions[].support_guidance` | string | Yes | - | Support guidance text |
| `students[].questions[].misconceptions` | list[string] | Yes | - | Detected misconceptions |
| `students[].questions[].feedback` | string | Yes | - | LLM-generated coaching feedback text |
| `students[].questions[].tier_level` | int | Yes | - | Rubric tier level (0-3) |
| `students[].questions[].tier_label` | string | Yes | - | Rubric tier label |

**Example (ensemble_results.yaml):**

```yaml
students:
- student_id: "S001"
  questions:
  - question_sn: 1
    ensemble_score: 0.73
    understanding_level: "Proficient"
    component_scores:
      concept_coverage: 0.83
      llm_rubric: 0.67
      rasch_ability: 0.45
```

---

### Longitudinal Store YAML

**Purpose**: Persistent store tracking student evaluation records across weeks, enabling trend analysis and trajectory visualization.

**Created by**: `forma-eval` (via `snapshot_from_evaluation()`)

**Consumed by**: `forma-report-longitudinal`, `forma-train`, `forma-report-warning`, `forma-report-professor`

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `records` | object | Yes | - | Keyed by `"{student_id}_{week}_{question_sn}"` |
| `records.<key>.student_id` | string | Yes | - | Student identifier |
| `records.<key>.week` | int | Yes | - | Week number |
| `records.<key>.question_sn` | int | Yes | - | Question serial number |
| `records.<key>.scores` | object | Yes | - | Metric scores (e.g., `concept_coverage`, `llm_rubric`, `rasch_ability`) |
| `records.<key>.tier_level` | int | Yes | - | Rubric tier level (0-3) |
| `records.<key>.tier_label` | string | Yes | - | Rubric tier label |
| `records.<key>.manual_override` | bool | Yes | `false` | If `true`, record is preserved on re-evaluation |
| `records.<key>.node_recall` | float | No | - | Graph node recall (0.0-1.0); v2 field |
| `records.<key>.edge_f1` | float | No | - | Graph edge F1 score; v2 field |
| `records.<key>.misconception_count` | int | No | - | Number of wrong-direction edges; v2 field |
| `records.<key>.concept_scores` | object | No | - | Per-concept correctness ratio `{concept: float}`; v2 field |
| `records.<key>.exam_file` | string | No | - | Exam file basename; v2 field |
| `records.<key>.recorded_at` | string | No | - | ISO 8601 UTC timestamp; v2 field |

Records are keyed by a composite string `"{student_id}_{week}_{question_sn}"` for fast upsert. The store uses atomic writes with file locking (`fcntl.flock`) and `.bak` backups.

**Example:**

```yaml
records:
  S001_1_1:
    student_id: "S001"
    week: 1
    question_sn: 1
    scores:
      concept_coverage: 0.83
      llm_rubric: 0.67
      rasch_ability: 0.45
    tier_level: 2
    tier_label: "Proficient"
    manual_override: false
    node_recall: 0.75
    edge_f1: 0.60
    misconception_count: 1
    concept_scores:
      homeostasis: 1.0
      receptor: 0.5
    exam_file: "Ch01_FormativeTest.yaml"
    recorded_at: "2026-03-01T12:00:00+00:00"
```

---

### Intervention Log YAML

**Purpose**: Persistent log of intervention activities (counseling, supplementary learning, etc.) for tracking what actions were taken for at-risk students.

**Created by**: `forma-intervention add`

**Consumed by**: `forma-intervention list/update`, `forma-report-professor`, `forma-report-longitudinal`

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `_meta` | object | Yes | - | Metadata section |
| `_meta.next_id` | int | Yes | `1` | Next auto-increment ID |
| `records` | list | Yes | - | List of intervention record objects |
| `records[].id` | int | Yes | - | Auto-assigned unique identifier |
| `records[].student_id` | string | Yes | - | Student identifier |
| `records[].week` | int | Yes | - | Week number when intervention occurred |
| `records[].intervention_type` | string | Yes | - | One of the valid types (see below) |
| `records[].description` | string | No | `""` | Free-text description |
| `records[].recorded_by` | string | No | `null` | Name of the person who recorded |
| `records[].recorded_at` | string | Yes | auto | ISO 8601 UTC timestamp (auto-set on creation) |
| `records[].follow_up_week` | int | No | `null` | Week number for follow-up |
| `records[].outcome` | string | No | `null` | Outcome set later via `forma-intervention update` |

Valid `intervention_type` values:

| Value | Meaning |
|-------|---------|
| `면담` | Counseling session |
| `보충학습` | Supplementary learning |
| `과제부여` | Assignment |
| `멘토링` | Mentoring |
| `기타` | Other |

The log uses atomic writes with file locking and `.bak` backups.

**Example:**

```yaml
_meta:
  next_id: 3
records:
- id: 1
  student_id: "S015"
  week: 2
  intervention_type: "면담"
  description: "Discussed homeostasis misconceptions"
  recorded_by: "Prof. Kim"
  recorded_at: "2026-03-05T09:00:00+00:00"
  follow_up_week: 3
  outcome: null
- id: 2
  student_id: "S039"
  week: 2
  intervention_type: "보충학습"
  description: "Assigned extra practice on feedback loops"
  recorded_by: null
  recorded_at: "2026-03-05T10:30:00+00:00"
  follow_up_week: null
  outcome: null
```

---

### Prepare Summary YAML

**Purpose**: Records the results of the delivery preparation stage, listing per-student file matching status and zip archive paths.

**Created by**: `forma-deliver prepare`

**Consumed by**: `forma-deliver send`

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prepared_at` | string | Yes | - | ISO 8601 UTC timestamp |
| `class_name` | string | Yes | - | Class section name (from roster) |
| `total_students` | int | Yes | - | Total number of students in roster |
| `ready` | int | Yes | - | Count of students with all files matched |
| `warnings` | int | Yes | - | Count of students with partial file matches |
| `errors` | int | Yes | - | Count of students with errors (no files, invalid email, etc.) |
| `details` | list | Yes | - | Per-student results (all students) |
| `details[].student_id` | string | Yes | - | Student identifier |
| `details[].name` | string | Yes | - | Student name |
| `details[].email` | string | Yes | - | Student email address |
| `details[].status` | string | Yes | - | `"ready"`, `"warning"`, or `"error"` |
| `details[].matched_files` | list[string] | Yes | `[]` | List of matched report file paths |
| `details[].zip_path` | string | No | `null` | Path to generated zip file |
| `details[].zip_size_bytes` | int | Yes | `0` | Size of zip file in bytes |
| `details[].message` | string | Yes | `""` | Warning or error message |

Status values:

| Status | Meaning |
|--------|---------|
| `ready` | All file patterns matched successfully |
| `warning` | Some file patterns did not match |
| `error` | No files matched, invalid email, or zip size exceeds 25 MB |

**Example:**

```yaml
prepared_at: "2026-03-10T14:00:00+00:00"
class_name: "1A"
total_students: 3
ready: 2
warnings: 0
errors: 1
details:
- student_id: "S001"
  name: "Kim Minjun"
  email: "minjun@example.com"
  status: "ready"
  matched_files:
  - "output/reports/S001_report.pdf"
  zip_path: "staging/S001_Kim Minjun/Kim Minjun_S001.zip"
  zip_size_bytes: 524288
  message: ""
- student_id: "S002"
  name: "Lee Soyeon"
  email: "soyeon@example.com"
  status: "ready"
  matched_files:
  - "output/reports/S002_report.pdf"
  zip_path: "staging/S002_Lee Soyeon/Lee Soyeon_S002.zip"
  zip_size_bytes: 498000
  message: ""
- student_id: "S003"
  name: "Park Jihun"
  email: ""
  status: "error"
  matched_files: []
  zip_path: null
  zip_size_bytes: 0
  message: "email missing or invalid format"
```

---

### Delivery Log YAML

**Purpose**: Audit trail recording the outcome of each email delivery attempt, including success/failure status per student.

**Created by**: `forma-deliver send`

**Consumed by**: Reference only (audit trail); `forma-deliver send --retry-failed` reads previous log

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `sent_at` | string | Yes | - | ISO 8601 UTC timestamp of send session start |
| `smtp_server` | string | Yes | - | SMTP server hostname used |
| `dry_run` | bool | Yes | - | Whether this was a dry-run (no actual sending) |
| `total` | int | Yes | - | Total number of send targets |
| `success` | int | Yes | - | Number of successful sends |
| `failed` | int | Yes | - | Number of failed sends |
| `results` | list | Yes | - | Per-student delivery results |
| `results[].student_id` | string | Yes | - | Student identifier |
| `results[].email` | string | Yes | - | Recipient email address |
| `results[].status` | string | Yes | - | `"success"` or `"failed"` |
| `results[].sent_at` | string | Yes | - | ISO 8601 UTC timestamp of this send |
| `results[].attachment` | string | Yes | - | Zip file name |
| `results[].size_bytes` | int | Yes | - | Attachment size in bytes |
| `results[].error` | string | No | `""` | Error message (empty on success) |

**Example:**

```yaml
sent_at: "2026-03-10T15:00:00+00:00"
smtp_server: "smtp.example.com"
dry_run: false
total: 2
success: 2
failed: 0
results:
- student_id: "S001"
  email: "minjun@example.com"
  status: "success"
  sent_at: "2026-03-10T15:00:01+00:00"
  attachment: "Kim Minjun_S001.zip"
  size_bytes: 524288
  error: ""
- student_id: "S002"
  email: "soyeon@example.com"
  status: "success"
  sent_at: "2026-03-10T15:00:03+00:00"
  attachment: "Lee Soyeon_S002.zip"
  size_bytes: 498000
  error: ""
```

---

### Project Configuration YAML (forma.yaml)

**Purpose**: Project-level configuration file that provides default values for CLI flags, reducing repetitive command-line arguments across the project.

**Created by**: Manual or via `forma-init` template generator

**Consumed by**: All CLI commands (via `apply_project_config()`)

**Location**: Discovered by walking from the current directory upward until `forma.yaml` is found or a `.git` directory sentinel is reached.

**Merge precedence** (highest to lowest):

1. CLI flags (explicitly provided)
2. `forma.yaml` project configuration
3. System configuration (`config.py`)
4. argparse defaults

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `project` | object | No | - | Project metadata section |
| `project.course_name` | string | No | `""` | Course name |
| `project.year` | int | No | `0` | Academic year (>= 2020) |
| `project.semester` | int | No | `0` | Semester number (1 or 2) |
| `project.grade` | int | No | `0` | Student grade year (>= 1) |
| `classes` | object | No | - | Class section configuration |
| `classes.identifiers` | list[string] | No | `[]` | Class section identifiers (e.g., `["A", "B"]`) |
| `classes.join_pattern` | string | No | `""` | File pattern with `{class}` placeholder |
| `classes.eval_pattern` | string | No | `""` | Directory pattern with `{class}` placeholder |
| `paths` | object | No | - | File path configuration |
| `paths.exam_config` | string | No | `""` | Path to exam configuration YAML |
| `paths.join_dir` | string | No | `""` | Path to joined data directory |
| `paths.output_dir` | string | No | `""` | Path to output directory |
| `paths.longitudinal_store` | string | No | `""` | Path to longitudinal store YAML |
| `paths.font_path` | string | No | `null` | Path to Korean font file (auto-detect if null) |
| `ocr` | object | No | - | OCR configuration |
| `ocr.naver_config` | string | No | `""` | Path to Naver OCR configuration |
| `ocr.credentials` | string | No | `""` | Credentials reference |
| `ocr.spreadsheet_url` | string | No | `""` | Google Sheets URL |
| `ocr.num_questions` | int | No | `5` | Number of questions per exam (>= 1) |
| `evaluation` | object | No | - | Evaluation pipeline settings |
| `evaluation.provider` | string | No | `"gemini"` | LLM provider (`"gemini"` or `"anthropic"`) |
| `evaluation.model` | string | No | `null` | LLM model name override |
| `evaluation.skip_feedback` | bool | No | `false` | Skip feedback generation |
| `evaluation.skip_graph` | bool | No | `false` | Skip graph comparison |
| `evaluation.skip_statistical` | bool | No | `false` | Skip statistical analysis |
| `evaluation.n_calls` | int | No | `3` | Number of LLM calls per item (>= 1) |
| `reports` | object | No | - | Report generation settings |
| `reports.dpi` | int | No | `150` | Chart image resolution (72-600) |
| `reports.skip_llm` | bool | No | `false` | Skip all LLM analysis |
| `reports.aggregate` | bool | No | `true` | Generate aggregate report |
| `prediction` | object | No | - | Prediction model settings |
| `prediction.model_path` | string | No | `null` | Path to pre-trained risk prediction model |
| `current_week` | int | No | `1` | Current week number (>= 1); top-level key |

Validation rules:

- Unknown top-level keys produce a warning (not an error)
- `bool` values are rejected where `int` is expected (Python `bool` is a subclass of `int`)
- `classes.join_pattern` and `classes.eval_pattern` must contain `{class}` if non-empty

**Example:**

```yaml
project:
  course_name: "Human Anatomy"
  year: 2026
  semester: 1
  grade: 1

classes:
  identifiers: ["A", "B", "C", "D"]
  join_pattern: "results/anp_w{week}/anp_{class}_final.yaml"
  eval_pattern: "results/anp_w{week}/eval_{class}"

paths:
  exam_config: "exams/Ch01_FormativeTest.yaml"
  join_dir: "results/anp_w1"
  output_dir: "output"
  longitudinal_store: "longitudinal.yaml"

evaluation:
  provider: "gemini"
  n_calls: 3

reports:
  dpi: 150
  aggregate: true

current_week: 3
```
