# Weekly Workflow Guide

The professor's complete weekly routine from exam creation through intervention tracking and report delivery.

## Table of Contents

- [Pre-Semester Setup](#pre-semester-setup)
- [Step 1: Create the Week's Exam](#step-1-create-the-weeks-exam)
- [Step 2: Collect and Scan Student Answers](#step-2-collect-and-scan-student-answers)
- [Step 3: Run Evaluation](#step-3-run-evaluation)
- [Step 4: Generate Reports](#step-4-generate-reports)
- [Step 5: Deliver Reports](#step-5-deliver-reports)
- [Step 6: Track Interventions](#step-6-track-interventions)
- [Step 7: Longitudinal Analysis](#step-7-longitudinal-analysis)
- [Step 8: Generate Warning Reports](#step-8-generate-warning-reports)
- [Step 9: Grade Prediction](#step-9-grade-prediction)
- [Dependencies Between Steps](#dependencies-between-steps)
- [Troubleshooting](#troubleshooting)
- [See Also](#see-also)

---

## Pre-Semester Setup

These steps are performed once before the semester begins.

**1. Initialize the project configuration:**

```bash
forma-init --output forma.yaml
```

This launches an interactive wizard that creates a `forma.yaml` template with your course name, year, semester, and class identifiers. See [Configuration](configuration.md) for details.

**2. Configure SMTP credentials in `forma.json`:**

```json
{
  "smtp": {
    "server": "smtp.example.com",
    "port": 587,
    "sender_email": "professor@example.com",
    "sender_name": "Prof. Name",
    "use_tls": true
  }
}
```

The SMTP password is never stored in config files. Use the `FORMA_SMTP_PASSWORD` environment variable or `--password-from-stdin` at send time.

**3. Prepare the student roster:**

Create a `roster.yaml` mapping student IDs to names and email addresses. See [Data Formats](data-formats.md) for the schema.

**4. Prepare exam configuration:**

Write an exam YAML file with questions, rubrics, keywords, and (optionally) a knowledge graph.

---

## Week N Workflow

The following steps repeat each week throughout the semester.

### Step 1: Create the Week's Exam

```bash
forma-exam --config exams/week_N.yaml --output exams/week_N.pdf --num-papers 50
```

Each paper gets a unique QR code linking student IDs to their answer sheets. If `num-papers` and metadata are in the YAML, they do not need to be repeated on the command line.

### Step 2: Collect and Scan Student Answers

**Scan answer sheets:**

```bash
forma-ocr scan --config ocr_config.yaml
```

Reads scanned images, decodes QR codes, runs OCR, and produces a results YAML file.

**Join OCR results with online responses:**

```bash
forma-ocr join --ocr-results results.yaml --output final.yaml \
               --spreadsheet-url "https://docs.google.com/spreadsheets/d/XXX"
```

At least one of `--spreadsheet-url` or `--forms-csv` is required.

### Step 3: Run Evaluation

**Single class:**

```bash
forma-eval --config exams/week_N.yaml --responses results/final_A.yaml \
           --output results/eval_A/ --longitudinal-store longitudinal.yaml
```

**Multiple classes at once:**

```bash
forma-eval-batch --config exams/week_N.yaml \
                 --join-dir results/ --join-pattern "final_{class}.yaml" \
                 --output results/eval/ --classes A B C D \
                 --longitudinal-store longitudinal.yaml
```

The `--longitudinal-store` flag accumulates results across weeks automatically. This data is required for longitudinal analysis, risk prediction, and grade prediction in later steps.

### Step 4: Generate Reports

**Student individual reports (one PDF per student):**

```bash
forma-report --final results/final_A.yaml --config exams/week_N.yaml \
             --eval-dir results/eval_A/ --output-dir reports/students/ \
             --longitudinal-store longitudinal.yaml --week N
```

**Professor class summary report:**

```bash
forma-report-professor --final results/final_A.yaml --config exams/week_N.yaml \
                       --eval-dir results/eval_A/ --output-dir reports/professor/ \
                       --class-name "A" --longitudinal-store longitudinal.yaml --week N
```

**Multi-class batch reports (student + professor + optional aggregate):**

```bash
forma-report-batch --config exams/week_N.yaml \
                   --join-dir results/ --join-pattern "final_{class}.yaml" \
                   --eval-pattern "eval_{class}/" \
                   --output-dir reports/ --classes A B C D --aggregate
```

### Step 5: Deliver Reports

**Prepare per-student ZIP archives:**

```bash
forma-deliver prepare --manifest manifest.yaml --roster roster.yaml \
                      --output-dir staging/
```

**Send emails (use `--dry-run` first to preview):**

```bash
# Preview
forma-deliver send --staged staging/ --template email_template.yaml --dry-run

# Send for real
echo "$FORMA_SMTP_PASSWORD" | forma-deliver send \
    --staged staging/ --template email_template.yaml \
    --password-from-stdin --notify-sender
```

If some emails fail, use `--retry-failed` to resend only the failures:

```bash
echo "$FORMA_SMTP_PASSWORD" | forma-deliver send \
    --staged staging/ --template email_template.yaml \
    --password-from-stdin --retry-failed
```

### Step 6: Track Interventions

Record and track intervention activities for at-risk students. This step can happen any time after Step 3.

**Add a record:**

```bash
forma-intervention add --store interventions.yaml \
    --student S015 --week 3 --type "면담" \
    --description "Discussed study strategies"
```

Valid intervention types: `면담` (counseling), `보충학습` (supplementary learning), `과제부여` (assignment), `멘토링` (mentoring), `기타` (other).

**List records:**

```bash
forma-intervention list --store interventions.yaml
forma-intervention list --store interventions.yaml --student S015
```

**Update outcome:**

```bash
forma-intervention update --store interventions.yaml --id 1 --outcome "개선"
```

Valid outcomes: `개선` (improved), `유지` (maintained), `악화` (worsened).

Intervention data is included in professor and longitudinal reports via the `--intervention-log` flag. It is excluded from student reports per privacy policy (FR-013).

### Step 7: Longitudinal Analysis

Generate a multi-week trajectory report after accumulating 3+ weeks of data. Run periodically (every 3-4 weeks).

```bash
forma-report-longitudinal --store longitudinal.yaml \
                          --class-name "A" \
                          --output reports/longitudinal_A.pdf \
                          --intervention-log interventions.yaml
```

The report includes student trajectory charts, class heatmaps, risk analysis, and concept mastery changes over time.

### Step 8: Generate Warning Reports

Once sufficient longitudinal data exists, train a risk model and generate early warning reports.

**Train the risk model (after 3+ weeks of data):**

```bash
forma-train --store longitudinal.yaml --output risk_model.pkl
```

**Generate the warning report:**

```bash
forma-report-warning --final results/final_A.yaml --config exams/week_N.yaml \
                     --eval-dir results/eval_A/ --output warning_A.pdf \
                     --longitudinal-store longitudinal.yaml --week N \
                     --model risk_model.pkl
```

The warning report combines rule-based at-risk identification with model predictions and includes per-student warning cards with deficit concepts and recommended interventions.

### Step 9: Grade Prediction

At the end of the semester, train a grade prediction model from historical data.

**Train the grade model:**

```bash
forma-train-grade --store longitudinal.yaml \
                  --grades grade_mapping.yaml \
                  --output grade_model.pkl
```

The `grade_mapping.yaml` file maps student IDs to letter grades (A/B/C/D/F) grouped by semester. See [Data Formats](data-formats.md) for the schema.

**Use grade predictions in reports:**

Predictions appear automatically when `--grade-model` is provided:

```bash
forma-report-professor --final results/final_A.yaml --config exams/week_N.yaml \
                       --eval-dir results/eval_A/ --output-dir reports/ \
                       --grade-model grade_model.pkl \
                       --longitudinal-store longitudinal.yaml --week N
```

Student reports show a softened tier (upper/middle/lower) instead of raw letter grades.

---

## Dependencies Between Steps

```
Pre-Semester Setup
       |
       v
  Step 1 (Exam) --> Step 2 (Scan/Join) --> Step 3 (Evaluate)
                                                |
                              +-----------------+-----------------+
                              |                 |                 |
                              v                 v                 v
                     Step 4 (Reports)   Step 5 (Deliver)   Step 6 (Interventions)
                              |
                              v
              Step 7 (Longitudinal)  <-- requires 3+ weeks
                              |
                              v
              Step 8 (Warning)       <-- requires trained model
                              |
                              v
              Step 9 (Grade)         <-- requires grade history
```

- Step 2 must complete before Step 3
- Step 3 must complete before Steps 4, 5, and 6
- Step 5 depends on Step 4 (reports must exist to deliver)
- Step 6 can happen any time after Step 3
- Steps 7-9 require multiple weeks of data in the longitudinal store
- Steps 7 and 8 can run independently of each other
- Step 9 additionally requires historical grade mapping data

---

## Troubleshooting

**JDK not found**

Some dependencies (e.g., `konlpy`) require JDK 17+. Set `JAVA_HOME`:

```bash
export JAVA_HOME=/path/to/jdk
java -version
```

**mecab not found**

Install the Korean morphological analyzer:

```bash
# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ko-dic

# macOS
brew install mecab mecab-ko-dic

# NixOS (included in devShell)
nix develop
```

**SMTP authentication failure**

Verify `forma.json` smtp fields and confirm the password is available:

```bash
export FORMA_SMTP_PASSWORD="your-password"
echo "$FORMA_SMTP_PASSWORD" | forma-deliver send --staged staging/ \
    --template template.yaml --password-from-stdin --dry-run
```

**forma-eval fails on OCR results**

Verify `forma-ocr join` output format. Use `--questions-used` if question numbering differs from the exam config:

```bash
forma-eval --config exam.yaml --responses final.yaml --output results/ \
           --questions-used 1 3
```

**forma-report-warning crashes with model errors**

Ensure the risk model was trained with the same week range as the current longitudinal data. Retrain if needed:

```bash
forma-train --store longitudinal.yaml --output risk_model.pkl
```

**Korean font not found**

ReportLab requires NanumGothic for PDF rendering. Specify the path explicitly if auto-detection fails:

```bash
forma-report --final final.yaml --config exam.yaml --eval-dir eval/ \
             --output-dir reports/ --font-path /path/to/NanumGothic.ttf
```

---

## See Also

- [CLI Reference](cli-reference.md) -- complete flag lists for all 14 commands
- [Configuration](configuration.md) -- forma.json and forma.yaml setup guide
- [Data Formats](data-formats.md) -- YAML file schemas for all input/output files
- [Quickstart](quickstart.md) -- first-time setup and minimal working example
