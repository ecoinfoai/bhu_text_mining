# Configuration Setup Guide

This guide explains the three configuration files used by FormA, what each one does, and exactly what to write in every field. It is written for first-time users who have never configured the tool before.

## Overview: Three Files, Three Purposes

FormA uses three configuration files. Each lives in a different place and serves a different scope:

| File | Scope | Location | Purpose |
|------|-------|----------|---------|
| `config.json` | **Machine-wide** | `~/.config/formative-analysis/config.json` | API keys, SMTP credentials — secrets that never go into version control |
| `forma.yaml` | **Per-semester** | Project root (e.g., `anp2026/forma.yaml`) | Course info, class sections, LLM settings, output paths — shared across all weeks |
| `week.yaml` | **Per-week** | Each week's directory (e.g., `week_01/week.yaml`) | Image directories, file patterns, exam questions — changes every week |

**How they relate:**

```
~/.config/formative-analysis/config.json    (secrets — API keys, SMTP)
    |
    v
anp2026_formative_analysis/
    forma.yaml                               (semester settings)
    week_01/
        week.yaml                            (week 1 settings)
        scans_1A_w1/                         (scan images for class A)
        final_A.yaml                         (joined results)
        eval_A/                              (evaluation output)
    week_02/
        week.yaml                            (week 2 settings)
        ...
```

**Priority order** (highest to lowest): CLI flags > `week.yaml` > `forma.yaml` > `config.json` > built-in defaults. This means you can always override any setting by passing a CLI flag, without editing any file.

---

## File 1: config.json (Secrets)

### What it is

A JSON file that stores API keys and email server credentials. It lives in your home directory, **not** inside your project — this keeps secrets out of version control.

### Where to put it

```
~/.config/formative-analysis/config.json
```

Create the directory if it does not exist:

```bash
mkdir -p ~/.config/formative-analysis
```

Then create `config.json` with a text editor. Set restrictive permissions so only you can read it:

```bash
chmod 600 ~/.config/formative-analysis/config.json
```

### What to write in it

The file has up to three sections. Include only the sections you need.

```json
{
  "llm": {
    "provider": "gemini",
    "api_key": "AIzaSy..."
  },
  "smtp": {
    "server": "smtp.gmail.com",
    "port": 587,
    "sender_email": "professor@university.edu",
    "sender_name": "Prof. Kim",
    "use_tls": true,
    "send_interval_sec": 1.0
  },
  "naver_ocr": {
    "secret_key": "your-naver-secret-key",
    "api_url": "https://your-endpoint.apigw.ntruss.com/..."
  }
}
```

### Field-by-field explanation

#### `llm` section — LLM API access

You need this section if you use `forma eval` (AI-powered evaluation) or `forma ocr scan` (LLM Vision OCR).

| Field | What to write | Example |
|-------|---------------|---------|
| `provider` | Which LLM service to use. Write `"gemini"` for Google Gemini or `"anthropic"` for Anthropic Claude. | `"gemini"` |
| `api_key` | Your API key from the provider's console. For Gemini, get it from [Google AI Studio](https://aistudio.google.com/). For Anthropic, get it from [Anthropic Console](https://console.anthropic.com/). You can also set this as an environment variable (`GEMINI_API_KEY` or `ANTHROPIC_API_KEY`) instead of putting it here. | `"AIzaSy..."` |

> **Tip:** If you prefer not to store the API key in a file, set the environment variable instead and omit the `api_key` field entirely. FormA checks the environment variable automatically.

#### `smtp` section — Email delivery

You need this section only if you plan to email PDF reports to students using `forma deliver send`. Skip it entirely if you distribute reports by hand or through an LMS.

| Field | What to write | Example |
|-------|---------------|---------|
| `server` | Your university's SMTP server hostname. Ask your IT department if unsure. | `"smtp.gmail.com"` |
| `port` | SMTP port number. `587` is standard for STARTTLS; `465` for SSL. | `587` |
| `sender_email` | The "from" address students will see. | `"professor@university.edu"` |
| `sender_name` | Display name shown in the email. | `"Prof. Kim"` |
| `use_tls` | Whether to encrypt the connection. Almost always `true`. | `true` |
| `send_interval_sec` | Seconds to wait between emails (rate limiting). `1.0` is safe for most servers. | `1.0` |

> **Important:** The SMTP password is **never** stored in this file. When you run `forma deliver send`, provide it via the `FORMA_SMTP_PASSWORD` environment variable or `--password-from-stdin`.

#### `naver_ocr` section — Naver CLOVA OCR (optional)

You need this only if you use Naver CLOVA OCR for answer sheet scanning. **Most users should skip this** and use the default LLM Vision OCR instead (`forma ocr scan --provider gemini`).

| Field | What to write | Example |
|-------|---------------|---------|
| `secret_key` | Naver CLOVA OCR secret key from Naver Cloud Console. | `"abc123..."` |
| `api_url` | Naver CLOVA OCR API endpoint URL. Must start with `https://`. | `"https://..."` |

### Minimal config.json (most users)

If you only use Gemini for evaluation and OCR, and you set the API key via environment variable, your config.json can be as simple as:

```json
{
  "llm": {
    "provider": "gemini"
  }
}
```

Or even an empty object `{}` if you rely entirely on environment variables and do not send emails.

---

## File 2: forma.yaml (Semester Settings)

### What it is

A YAML file that stores settings for your entire semester: course name, class sections, LLM preferences, and file path conventions. Created once at the start of the semester, rarely changed after that.

### Where to put it

Place it at the root of your project directory. FormA searches upward from your current working directory to find it, so it covers all subdirectories (week folders) automatically.

```
anp2026_formative_analysis/
    forma.yaml              <-- here
    week_01/
    week_02/
```

### How to create it

Either run the interactive wizard:

```bash
forma init
```

Or create it manually with a text editor. Below is a complete, annotated example.

### Complete example with explanations

```yaml
# forma.yaml -- semester-level settings
# Only fill in the fields you need. Everything has a sensible default.

project:
  course_name: "Human Anatomy and Physiology"  # appears in report headers
  year: 2026                                    # academic year (>= 2020)
  semester: 1                                   # 1 = spring, 2 = fall
  grade: 1                                      # student year (1 = freshman)

classes:
  identifiers: [A, B, C, D]       # your class section labels
  join_pattern: "final_{class}.yaml"   # {class} is replaced with A, B, C, D
  eval_pattern: "eval_{class}"         # directory pattern for evaluation output

paths:
  join_dir: ""                     # base directory for joined data files
  output_dir: ""                   # where PDF reports are saved
  longitudinal_store: ""           # path to longitudinal tracking file
  font_path: null                  # Korean font path (null = auto-detect)

ocr:
  ocr_model: null                  # LLM model for OCR (null = provider default)
  spreadsheet_url: ""              # Google Sheets URL for student responses
  num_questions: 2                 # number of answer areas per exam sheet

evaluation:
  provider: "gemini"               # "gemini" or "anthropic"
  model: null                      # scoring model (null = provider default)
  n_calls: 3                       # LLM calls per question (1-5, higher = more reliable)

reports:
  dpi: 150                         # chart resolution (72-600)

prediction:
  model_path: null                 # risk prediction model (.pkl file)

current_week: 2                    # update this each week
```

### Field-by-field explanation

#### `project` section — Course metadata

This information appears in PDF report headers and file naming. Fill it in once at the start of the semester.

| Field | What to write | When to change |
|-------|---------------|----------------|
| `course_name` | Full course name as you want it to appear on reports. | Never (once set) |
| `year` | Academic year. Must be 2020 or later. | Each academic year |
| `semester` | `1` for spring/first semester, `2` for fall/second semester. | Each semester |
| `grade` | Student year level. `1` = freshman, `2` = sophomore, etc. | Each year |

#### `classes` section — Section configuration

Defines your class sections and the file naming patterns FormA uses to find data files for each section.

| Field | What to write | Example |
|-------|---------------|---------|
| `identifiers` | List of class section labels. These must match the labels you use in `--class` flags and in `{class}` patterns. | `[A, B, C, D]` |
| `join_pattern` | Filename pattern for joined data files. Must contain `{class}`. FormA replaces `{class}` with each identifier. | `"final_{class}.yaml"` produces `final_A.yaml`, `final_B.yaml`, etc. |
| `eval_pattern` | Directory pattern for evaluation results. Must contain `{class}`. | `"eval_{class}"` produces `eval_A/`, `eval_B/`, etc. |

> **When to set these:** Fill in `identifiers` at the start of the semester. Fill in `join_pattern` and `eval_pattern` if you use batch commands (`forma report batch`). If you always specify paths explicitly on the command line, you can leave the patterns empty.

#### `paths` section — Directory paths

These paths are used by report and batch commands so you do not have to type them every time.

| Field | What to write | Example |
|-------|---------------|---------|
| `join_dir` | Directory containing joined data files (the `final_*.yaml` files). Leave empty if you always specify it on the command line. | `"results/week_01"` |
| `output_dir` | Directory where PDF reports are saved. | `"reports/"` |
| `longitudinal_store` | Path to the YAML file that accumulates results across weeks. Create this file after your first evaluation; it grows over the semester. | `"store/longitudinal.yaml"` |
| `font_path` | Path to a `.ttf` Korean font file. Set to `null` and FormA will try to find one automatically. Only set this if auto-detection fails. | `null` |

> **Tip:** These paths are relative to the directory where you run the command, not relative to `forma.yaml`. If you always `cd` into your project root before running commands, relative paths work fine.

#### `ocr` section — OCR scanning settings

Controls how FormA processes scanned answer sheets.

| Field | What to write | Example |
|-------|---------------|---------|
| `ocr_model` | LLM model ID for OCR text extraction. Set to `null` to use the provider's default model (`gemini-2.5-flash`). Set a specific model if you want to pin the version. | `null` or `"gemini-2.5-flash"` |
| `spreadsheet_url` | Google Sheets URL if students submit answers online. FormA reads responses from this sheet during `forma ocr join`. Leave empty if you only use paper scans. | `"https://docs.google.com/spreadsheets/d/abc..."` |
| `num_questions` | Number of answer areas per exam sheet. This tells the OCR pipeline how many text regions to extract from each scanned image. | `2` |

#### `evaluation` section — LLM scoring settings

Controls the AI evaluation pipeline that scores student answers.

| Field | What to write | Example |
|-------|---------------|---------|
| `provider` | LLM provider for scoring. `"gemini"` (Google, free tier available) or `"anthropic"` (Claude, paid). | `"gemini"` |
| `model` | Specific model name. Set to `null` to use the provider's default (recommended). | `null` or `"claude-sonnet-4-6"` |
| `n_calls` | How many times to call the LLM per question. Higher values improve reliability through median aggregation. `3` is the recommended default. Use `1` for quick testing. | `3` |

> **Note:** `ocr.ocr_model` and `evaluation.model` are **separate settings**. The OCR model extracts text from images (needs a vision-capable model). The evaluation model scores written answers (needs a reasoning-capable model). You can use different models for each.

#### `reports` section — PDF output settings

| Field | What to write | Example |
|-------|---------------|---------|
| `dpi` | Chart image resolution in dots per inch. Higher values produce sharper charts but larger files. `150` is a good balance. | `150` |

#### `prediction` section — Risk prediction

| Field | What to write | Example |
|-------|---------------|---------|
| `model_path` | Path to a trained risk prediction model (`.pkl` file). Set to `null` until you have trained a model with `forma train risk`. | `null` or `"models/risk.pkl"` |

#### `current_week` — Top-level field

| Field | What to write | Example |
|-------|---------------|---------|
| `current_week` | The current week number. Update this each week. Used by commands that need to know which week it is. | `2` |

---

## File 3: week.yaml (Per-Week Settings)

### What it is

A YAML file that stores settings specific to one week's formative assessment: which exam questions were used, where the scanned images are, what the output files should be named. You create one `week.yaml` per week directory.

### Where to put it

Inside each week's directory:

```
anp2026_formative_analysis/
    forma.yaml
    week_01/
        week.yaml           <-- week 1 settings
        scans_1A_w1/         <-- scanned images for class A
        scans_1B_w1/         <-- scanned images for class B
    week_02/
        week.yaml           <-- week 2 settings
        ...
```

FormA discovers `week.yaml` by searching upward from your current directory. If you `cd` into `week_01/` and run a command, FormA finds `week_01/week.yaml` automatically.

### Complete example with explanations

```yaml
# week.yaml -- settings for one week's formative assessment

week: 1                    # week number (required, must be >= 1)

select:
  source: "../exams/Ch01_FormativeTest.yaml"   # path to question bank
  questions: [1, 3]                             # which questions to use
  num_papers: 220                               # exam copies to print
  form_url: "https://docs.google.com/forms/d/e/.../viewform?usp=pp_url&entry.123={student_id}"

ocr:
  num_questions: 2                              # answer areas per sheet
  image_dir_pattern: "scans_1{class}_w1"        # image directory pattern
  ocr_output_pattern: "ocr_results_{class}.yaml"
  join_output_pattern: "final_{class}.yaml"
  join_forms_csv: "week1_ids.csv"               # CSV with student IDs
  student_id_column: "student_id"               # column name in the CSV

eval:
  config: "../exams/Ch01_FormativeTest.yaml"    # exam config (answer key)
  questions_used: [1, 3]                        # must match select.questions
  responses_pattern: "final_{class}.yaml"       # input: joined results
  output_pattern: "eval_{class}"                # output: evaluation directory
  skip_feedback: false                          # generate written feedback?
  skip_graph: true                              # skip knowledge graph comparison?
  generate_reports: true                        # auto-generate student PDFs?
```

### Field-by-field explanation

#### `week` — Top-level required field

| Field | What to write | Example |
|-------|---------------|---------|
| `week` | The week number for this assessment. Must be >= 1. | `1` |

#### `select` section — Exam generation

These fields are used by `forma select` to generate printable exam PDFs.

| Field | What to write | Example |
|-------|---------------|---------|
| `source` | Path to the FormativeTest YAML file containing your question bank. Relative to this `week.yaml` file's directory. | `"../exams/Ch01_FormativeTest.yaml"` |
| `questions` | List of question serial numbers to include in this week's exam. These are the `sn` fields from your question bank. | `[1, 3]` |
| `num_papers` | How many copies of the exam to generate (one per student plus extras). | `220` |
| `form_url` | Google Forms URL template. The `{student_id}` placeholder is replaced with each student's ID to create pre-filled links. Leave empty if you do not use Google Forms. | `"https://docs.google.com/forms/d/e/.../viewform?usp=pp_url&entry.123={student_id}"` |

#### `ocr` section — Scan processing

These fields tell `forma ocr scan` and `forma ocr join` where to find images and where to write results.

| Field | What to write | Example |
|-------|---------------|---------|
| `num_questions` | Number of answer areas per scanned sheet. Must match the number of questions on the exam. | `2` |
| `image_dir_pattern` | Directory containing scanned images. `{class}` is replaced with the class label (A, B, ...). | `"scans_1{class}_w1"` |
| `ocr_output_pattern` | Where to save OCR results. `{class}` is replaced. | `"ocr_results_{class}.yaml"` |
| `join_output_pattern` | Where to save joined (OCR + Forms) results. `{class}` is replaced. | `"final_{class}.yaml"` |
| `join_forms_csv` | CSV file containing student IDs from Google Forms. Used during `forma ocr join` to match paper scans with online submissions. Leave empty if not applicable. | `"week1_ids.csv"` |
| `student_id_column` | Column name in the CSV that contains student IDs. This must match the exact header text in your CSV file. | `"student_id"` |
| `crop_coords` | Coordinates for cropping answer areas from scanned images. **Do not fill this in manually.** FormA populates it automatically the first time you run `forma ocr scan` and interactively select the crop region. | (auto-populated) |
| `review_threshold` | Confidence threshold for flagging low-quality OCR results (0.0-1.0). Results below this threshold are marked for manual review. Default is `0.75`. | `0.75` |

> **About `{class}` patterns:** Every field that contains `{class}` is expanded once per class section. When you run `forma ocr scan --class A`, FormA reads `image_dir_pattern: "scans_1{class}_w1"` and opens `scans_1A_w1/`. When you run `--class B`, it opens `scans_1B_w1/`. This lets you use one `week.yaml` for all sections.

#### `eval` section — Evaluation pipeline

These fields tell `forma eval` how to score student responses.

| Field | What to write | Example |
|-------|---------------|---------|
| `config` | Path to the exam configuration YAML (contains correct answers, concept tags, rubric). Usually the same as `select.source`. | `"../exams/Ch01_FormativeTest.yaml"` |
| `questions_used` | Which questions were actually used in this week's exam. Must match `select.questions`. | `[1, 3]` |
| `responses_pattern` | Input file pattern. `{class}` is replaced. This should point to the output of `forma ocr join`. | `"final_{class}.yaml"` |
| `output_pattern` | Output directory pattern. `{class}` is replaced. Evaluation results are written here. | `"eval_{class}"` |
| `skip_feedback` | Set to `true` to skip generating written feedback text. Useful for quick test runs. | `false` |
| `skip_graph` | Set to `true` to skip knowledge graph comparison (the triplet-based analysis). Saves time if you do not need it. | `true` |
| `generate_reports` | Set to `true` to automatically generate individual student PDF reports after evaluation. | `true` |

---

## Putting It All Together: A Typical Semester Setup

Here is what a real project directory looks like after two weeks of assessments:

```
anp2026_formative_analysis/
    forma.yaml                          # semester settings (created once)
    exams/
        Ch01_FormativeTest.yaml         # question bank - week 1
        Ch03_FormativeTest.yaml         # question bank - week 2
    week_01/
        week.yaml                       # week 1 settings
        scans_1A_w1/                    # scanned images - class A
        scans_1B_w1/                    # scanned images - class B
        ocr_results_A.yaml             # OCR output
        final_A.yaml                    # joined results
        eval_A/                         # evaluation output
            res_lvl1/concept_results.yaml
            res_lvl2/llm_results.yaml
            ...
    week_02/
        week.yaml                       # week 2 settings
        ...
```

### Step-by-step first-time setup

1. **Create `config.json`** with your API key:
   ```bash
   mkdir -p ~/.config/formative-analysis
   cat > ~/.config/formative-analysis/config.json << 'EOF'
   {
     "llm": {
       "provider": "gemini",
       "api_key": "AIzaSy..."
     }
   }
   EOF
   chmod 600 ~/.config/formative-analysis/config.json
   ```

2. **Create `forma.yaml`** in your project root:
   ```bash
   cd anp2026_formative_analysis/
   forma init
   ```
   Answer the interactive prompts, then edit the generated file to fill in patterns.

3. **Create `week.yaml`** for your first week:
   ```bash
   mkdir week_01 && cd week_01
   ```
   Create `week.yaml` with a text editor, filling in the exam source, questions, and file patterns.

4. **Run the pipeline:**
   ```bash
   forma ocr scan --class A          # scan answer sheets
   forma ocr join --class A          # merge with Google Forms data
   forma eval --class A              # AI-powered evaluation
   ```

5. **Next week:** Create `week_02/week.yaml` with updated paths and question numbers. Increment `current_week` in `forma.yaml`.

---

## Quick Reference: Which File Controls What

| Setting | Where to configure | Why |
|---------|-------------------|-----|
| API keys | `config.json` | Secrets stay out of version control |
| SMTP server | `config.json` | Secrets stay out of version control |
| Course name, year | `forma.yaml` | Fixed for the semester |
| Class sections (A, B, C, D) | `forma.yaml` | Fixed for the semester |
| LLM provider and model | `forma.yaml` | Usually the same all semester |
| OCR model | `forma.yaml` | Usually the same all semester |
| Which questions this week | `week.yaml` | Changes every week |
| Scan image directories | `week.yaml` | Changes every week |
| Output file patterns | `week.yaml` | Changes every week |
| Google Forms CSV | `week.yaml` | Changes every week |

---

## Troubleshooting

### "No config file found"

FormA cannot find `config.json`. Check that the file exists at `~/.config/formative-analysis/config.json` and is valid JSON. Run:

```bash
cat ~/.config/formative-analysis/config.json | python3 -m json.tool
```

If this prints an error, your JSON syntax is broken (usually a missing comma or quote).

### "week.yaml not found"

FormA searches upward from your current directory. Make sure you are inside (or below) the directory that contains `week.yaml`. Alternatively, specify the path explicitly:

```bash
forma ocr scan --class A --week-config path/to/week.yaml
```

### "Unknown key in config.json: 'xxx'"

You have a section name in `config.json` that FormA does not recognize. Valid top-level keys are: `llm`, `smtp`, `naver_ocr`. Check for typos.

### OCR results look wrong

This is almost always a scan quality issue, not a configuration issue. See [Tips and Gotchas in the New Teachers Guide](for_new_teachers.md#tips-and-gotchas) for scan quality requirements.

---

## Further Reading

- [Configuration Reference](configuration.md) — Complete field tables with types and constraints
- [CLI Reference](cli-reference.md) — Every command and flag
- [Getting Started for New Teachers](for_new_teachers.md) — Scenario-based walkthrough
- [Weekly Workflow](weekly-workflow.md) — Step-by-step weekly routine
