# formative-analysis (FormA)

An AI-powered formative assessment CLI toolkit for university professors. FormA automates the full assessment cycle — from exam generation and OCR scanning through knowledge-graph evaluation, personalized feedback, longitudinal tracking, and report delivery — enabling data-driven instructional decisions without manual grading overhead.

## Features

FormA provides 14 CLI commands spanning the entire assessment workflow. See [docs/cli-reference.md](docs/cli-reference.md) for full usage details.

| Command | Description |
|---------|-------------|
| `forma-exam` | Exam PDF generation with per-student QR codes |
| `forma-ocr` | OCR pipeline with `scan` and `join` subcommands for handwritten answer sheets |
| `forma-eval` | 4-layer knowledge graph evaluation pipeline (concept coverage, LLM feedback, IRT statistics, ensemble scoring) |
| `forma-eval-batch` | Multi-class batch evaluation across sections |
| `forma-report` | Individual student PDF report generation |
| `forma-report-professor` | Professor-facing class summary report with analytics |
| `forma-report-batch` | Multi-class batch report generation |
| `forma-report-longitudinal` | Multi-week longitudinal trend analysis report |
| `forma-report-warning` | Early warning report identifying at-risk students |
| `forma-train` | Train drop-risk prediction model from longitudinal data |
| `forma-train-grade` | Train semester grade prediction model |
| `forma-init` | Interactive project configuration initialization |
| `forma-deliver` | Report delivery via email with `prepare` and `send` subcommands |
| `forma-intervention` | Intervention activity tracking with `add`, `list`, and `update` subcommands |

## Requirements

- **Python** >= 3.11, < 4
- **JDK 17+** — required for [KoNLPy](https://konlpy.org/) / JPype (Korean morphological analysis)
- **mecab** — Korean morphological analyzer backend
- **uv** (recommended) — fast Python package manager

### Platform-specific installation

| Dependency | NixOS | Ubuntu / Debian | macOS |
|------------|-------|-----------------|-------|
| JDK 17 | Included in `flake.nix` | `sudo apt install openjdk-17-jdk` | `brew install openjdk@17` |
| mecab | Included in `flake.nix` | `sudo apt install mecab libmecab-dev mecab-ipadic-utf8` | `brew install mecab` |
| uv | Included in `flake.nix` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `brew install uv` |

## Installation

### NixOS (recommended)

The Nix flake provides Python 3.11, JDK 17, mecab, and uv automatically:

```bash
nix develop
uv sync --extra dev
```

### General (Ubuntu / macOS / Windows WSL)

After installing the system dependencies listed above:

```bash
# Using uv (recommended)
uv sync --extra dev

# Or using pip
pip install -e ".[dev]"
```

### Global CLI installation

```bash
uv tool install .
```

## Quick Start

Below is a minimal example using sample data included in the repository. See [docs/quickstart.md](docs/quickstart.md) for a complete walkthrough and [docs/weekly-workflow.md](docs/weekly-workflow.md) for the recommended weekly assessment routine.

```bash
# 1. Run evaluation on a single class
forma-eval \
  --config exams/Ch01_FormativeTest.yaml \
  --responses results/anp_w1/anp_1A_final.yaml \
  --output results/anp_w1/eval_1A/ \
  --provider gemini

# 2. Generate individual student reports
forma-report \
  --final results/anp_w1/anp_1A_final.yaml \
  --config exams/Ch01_FormativeTest.yaml \
  --eval-dir results/anp_w1/eval_1A/ \
  --output-dir reports/

# 3. Generate professor summary report
forma-report-professor \
  --final results/anp_w1/anp_1A_final.yaml \
  --config exams/Ch01_FormativeTest.yaml \
  --eval-dir results/anp_w1/eval_1A/ \
  --output-dir reports/

# 4. Prepare and deliver reports via email
forma-deliver prepare \
  --manifest delivery/manifest.yaml \
  --roster delivery/roster.yaml \
  --output-dir delivery/packages/

forma-deliver send \
  --staged delivery/packages/ \
  --template delivery/template.yaml
```

## Configuration

FormA uses two configuration files. See [docs/configuration.md](docs/configuration.md) for the full reference and [docs/data-formats.md](docs/data-formats.md) for all YAML schema definitions.

### `forma.json` — Credentials

Located at `~/.config/formative-analysis/forma.json`. Stores API keys and service credentials:

```json
{
  "naver_ocr": {
    "secret_key": "...",
    "api_url": "..."
  },
  "llm": {
    "provider": "gemini",
    "api_key": "..."
  },
  "smtp": {
    "server": "smtp.example.com",
    "port": 587,
    "sender_email": "professor@example.com",
    "sender_name": "Prof. Kim",
    "use_tls": true
  }
}
```

> **Note:** SMTP passwords are never stored in `forma.json`. Use `--password-from-stdin` or the `FORMA_SMTP_PASSWORD` environment variable.

### `forma.yaml` — Project settings

Auto-discovered from the current working directory. Initialize with:

```bash
forma-init
```

This creates a `forma.yaml` with project-level defaults (class names, output paths, thresholds, etc.).

## Development

```bash
# Run all tests
uv run pytest tests/ -q

# Run tests with coverage
uv run pytest tests/ --cov=forma --cov-report=term-missing

# Lint
uv run ruff check src/ tests/

# Format check
uv run black --check src/ tests/
```

CI runs automatically on push via GitHub Actions (`.github/workflows/ci.yml`): tests on Python 3.11, 3.12, and 3.13, ruff lint, and coverage reporting.

## Project Structure

```
formative-analysis/
├── src/forma/                  # Main package (79 modules)
│   ├── cli.py                  # forma-exam entry point
│   ├── cli_ocr.py              # forma-ocr entry point
│   ├── cli_report.py           # forma-report entry point
│   ├── cli_report_professor.py # forma-report-professor entry point
│   ├── cli_report_batch.py     # forma-report-batch entry point
│   ├── cli_report_longitudinal.py
│   ├── cli_report_warning.py
│   ├── cli_deliver.py          # forma-deliver entry point
│   ├── cli_intervention.py     # forma-intervention entry point
│   ├── cli_init.py             # forma-init entry point
│   ├── cli_train.py            # forma-train entry point
│   ├── cli_train_grade.py      # forma-train-grade entry point
│   ├── pipeline_evaluation.py  # 4-layer evaluation pipeline
│   ├── pipeline_batch_evaluation.py
│   ├── config.py               # Configuration management
│   ├── project_config.py       # Project-level YAML config
│   ├── triplet_extractor.py    # LLM knowledge graph extraction
│   ├── graph_comparator.py     # Fuzzy graph matching
│   ├── feedback_generator.py   # Coaching feedback generation
│   ├── ensemble_scorer.py      # Ensemble scoring
│   ├── risk_predictor.py       # Drop-risk ML model
│   ├── grade_predictor.py      # Grade prediction ML model
│   ├── intervention_store.py   # Intervention YAML store
│   ├── longitudinal_store.py   # Longitudinal data store
│   ├── delivery_prepare.py     # Email delivery preparation
│   ├── delivery_send.py        # SMTP email sending
│   └── ...                     # Additional analysis and report modules
├── tests/                      # Test suite (91 files, 2500+ tests)
├── docs/                       # User guides and references
├── exams/                      # Exam YAML configurations
├── results/                    # Evaluation output data
├── .github/workflows/          # CI/CD pipeline
├── pyproject.toml              # Project metadata and dependencies
└── flake.nix                   # NixOS development environment
```
