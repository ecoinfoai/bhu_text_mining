# Quickstart Guide

Get from installation to your first evaluation report in 15 minutes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Create the Exam](#step-1-create-the-exam)
- [Step 2: Scan Student Answers](#step-2-scan-student-answers)
- [Step 3: Run Evaluation](#step-3-run-evaluation)
- [Step 4: Generate Student Reports](#step-4-generate-student-reports)
- [Step 5: Generate Professor Report](#step-5-generate-professor-report)
- [Step 6: Deliver Reports](#step-6-deliver-reports)
- [See Also](#see-also)

---

## Prerequisites

Before starting, ensure you have:

- **Python >= 3.11** and **JDK 17+** installed (see [README](../README.md#requirements) for platform-specific instructions)
- **mecab** Korean morphological analyzer installed
- **formative-analysis** installed in your environment:
  ```bash
  # NixOS
  nix develop && uv sync --extra dev

  # General
  uv sync --extra dev
  ```
- **config.json** configured with your API credentials at `~/.config/formative-analysis/config.json` (see [Configuration](configuration.md))
- **forma.yaml** initialized in your project directory (optional but recommended):
  ```bash
  forma init
  ```

---

## Step 1: Create the Exam

Generate a formative exam PDF with QR codes for each student.

**Command:**

```bash
forma exam --config exams/Ch01_FormativeTest.yaml --output exam_output.pdf
```

**Input:** An exam configuration YAML file defining questions, model answers, rubrics, and support guidance. See [Data Formats](data-formats.md#exam-configuration-yaml) for the full schema.

**Output:** A PDF file containing the exam paper with per-student QR codes for automated identification during scanning.

> **Tip:** Use `--num-papers 50` to specify the number of copies if not defined in the config YAML.

---

## Step 2: Scan Student Answers

After students complete the exam on paper, scan their answer sheets and extract text using OCR.

**Commands:**

```bash
# Scan answer sheet images and run OCR
forma ocr scan --config ocr_config.yaml

# Join OCR results with Google Sheets/Forms responses
forma ocr join --ocr-results results.yaml --output anp_1A_final.yaml \
               --spreadsheet-url "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID"
```

**Input:**
- Scanned answer sheet images in the directory specified by your OCR config
- A Google Sheets URL or CSV file with student form responses

**Output:** A joined YAML file (`anp_1A_final.yaml`) containing all student answers matched to their IDs via QR codes.

> **Note:** If you already have student responses in YAML format (e.g., from the sample data at `results/anp_w1/anp_1A_final.yaml`), you can skip this step and proceed directly to Step 3.

---

## Step 3: Run Evaluation

Run the 4-layer evaluation pipeline on student responses:

1. **Layer 1** -- Concept coverage + knowledge graph comparison
2. **Layer 2** -- LLM-based coaching feedback generation
3. **Layer 3** -- Rasch IRT statistical analysis
4. **Layer 4** -- Ensemble scoring + PDF report data

**Command:**

```bash
forma eval \
  --config exams/Ch01_FormativeTest.yaml \
  --responses results/anp_w1/anp_1A_final.yaml \
  --output results/anp_w1/eval_1A/ \
  --provider gemini
```

**Input:**
- Exam config YAML (`--config`)
- Student responses YAML (`--responses`) from Step 2 or existing data

**Output:** An evaluation directory (`eval_1A/`) containing:
- `res_lvl4/ensemble_results.yaml` -- per-student scores
- `res_lvl4/technical_report.yaml` -- detailed analysis
- `res_lvl4/counseling_summary.yaml` -- student-facing feedback

> **Tip:** Add `--skip-feedback` to skip LLM feedback generation (saves API cost during testing). Add `--longitudinal-store longitudinal.yaml` to accumulate data for multi-week tracking.

---

## Step 4: Generate Student Reports

Generate individual PDF reports for each student with their evaluation results, feedback, and concept analysis.

**Command:**

```bash
forma report student \
  --final results/anp_w1/anp_1A_final.yaml \
  --config exams/Ch01_FormativeTest.yaml \
  --eval-dir results/anp_w1/eval_1A/ \
  --output-dir reports/students/
```

**Input:**
- Student responses YAML (`--final`)
- Exam config YAML (`--config`)
- Evaluation results directory (`--eval-dir`) from Step 3

**Output:** One PDF report per student in the `reports/students/` directory, each containing:
- Understanding level and ensemble score breakdown
- Concept coverage analysis
- Personalized coaching feedback
- Knowledge graph visualization

> **Tip:** Use `--student S015` to generate a report for a single student. Add `--longitudinal-store store.yaml --week 4` for weekly change tracking.

---

## Step 5: Generate Professor Report

Generate a class-level summary report with statistics, concept analysis, and optional AI-powered insights.

**Command:**

```bash
forma report professor \
  --final results/anp_w1/anp_1A_final.yaml \
  --config exams/Ch01_FormativeTest.yaml \
  --eval-dir results/anp_w1/eval_1A/ \
  --output-dir reports/professor/
```

**Input:** Same as Step 4 (student responses, exam config, evaluation directory).

**Output:** A professor-facing PDF report containing:
- Class score distribution and statistics
- Per-concept mastery analysis
- At-risk student identification
- Knowledge graph aggregate view
- Misconception clustering analysis

> **Tip:** Add `--skip-llm` to skip AI analysis generation. Use `--model risk.pkl` for ML-based risk predictions (requires a trained model from `forma train risk`).

---

## Step 6: Deliver Reports

Package student reports and deliver them via email.

### 6a. Prepare delivery packages

```bash
forma deliver prepare \
  --manifest delivery/manifest.yaml \
  --roster delivery/roster.yaml \
  --output-dir delivery/staging/
```

**Input:**
- A delivery manifest YAML specifying where report files are and how to match them to students (see [Data Formats](data-formats.md#delivery-manifest-yaml))
- A student roster YAML with names and email addresses (see [Data Formats](data-formats.md#student-roster-yaml))

**Output:** A staging directory with per-student zip archives and a `prepare_summary.yaml` summarizing the preparation results.

### 6b. Send emails

```bash
echo "$FORMA_SMTP_PASSWORD" | forma deliver send \
  --staged delivery/staging/ \
  --template delivery/template.yaml \
  --password-from-stdin
```

**Input:**
- The staging directory from Step 6a (`--staged`)
- An email template YAML with subject and body (see [Data Formats](data-formats.md#email-template-yaml))
- SMTP configuration from `config.json` (see [Configuration](configuration.md#smtp-section))

**Output:** Emails sent to each student with their report zip attached. A `delivery_log.yaml` is saved as an audit trail.

> **Tip:** Use `--dry-run` to preview without sending. Use `--retry-failed` to resend only previously failed emails. Use `--notify-sender` to receive a summary email.

---

## See Also

- [docs/for_new_teachers.md](for_new_teachers.md) — Scenario-based guide: choose the right workflow for your situation
- [docs/weekly-workflow.md](weekly-workflow.md) — Step-by-step reference for the complete weekly assessment routine
- [docs/cli-reference.md](cli-reference.md) — Complete flag reference for all 15 commands
- [docs/configuration.md](configuration.md) — Full configuration reference (`config.json`, `forma.yaml`, `week.yaml`)
