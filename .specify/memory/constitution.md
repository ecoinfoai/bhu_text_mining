<!--
  Sync Impact Report
  ==================
  Version change: N/A (initial) → 1.0.0
  Added principles:
    - I. Module-First Architecture
    - II. CLI Interface Mandate
    - III. Test-First Development (NON-NEGOTIABLE)
    - IV. Data Pipeline Integrity
    - V. Korean Text Safety
    - VI. LLM Boundary Separation
    - VII. Pattern Consistency
  Added sections:
    - Technology Constraints
    - Development Workflow
    - Governance
  Templates status:
    - .specify/templates/plan-template.md ✅ compatible (no update needed)
    - .specify/templates/spec-template.md ✅ compatible (no update needed)
    - .specify/templates/tasks-template.md ✅ compatible (no update needed)
    - .specify/templates/checklist-template.md ✅ compatible (no update needed)
  Follow-up TODOs: None
-->

# FormativeAnalysis Constitution

## Core Principles

### I. Module-First Architecture

Every feature MUST be implemented as a standalone module within `src/forma/`.
Each module MUST have a single clear purpose, be independently importable,
and carry its own unit tests in `tests/test_<module>.py`.

- Modules MUST NOT create circular imports.
- Shared utilities (e.g., `font_utils.py`) are extracted once two or
  more modules need the same logic.
- No "organizational-only" modules that merely re-export symbols.

### II. CLI Interface Mandate

Every user-facing pipeline MUST expose a CLI entry point registered in
`pyproject.toml` under `[project.scripts]` with the `forma-` prefix
(e.g., `forma-exam`, `forma-report`).

- CLI MUST use `argparse` with `--help` descriptions in Korean where
  the target audience is Korean-speaking users.
- Required arguments MUST fail fast with clear error messages
  when missing.
- All file path arguments MUST be validated for existence before
  proceeding.

### III. Test-First Development (NON-NEGOTIABLE)

TDD is mandatory for all new modules. The Red-Green-Refactor cycle
MUST be strictly enforced.

- Tests MUST be written and confirmed to fail before implementation.
- Tests MUST use `pytest` with fixtures; mocking for external
  resources (fonts, PDF rendering, LLM calls, file I/O)
  follows the pattern in `tests/test_report_generator.py`.
- Every public function MUST have at least one unit test.
- Edge cases (empty data, missing fields, zero-variance statistics)
  MUST be covered.

### IV. Data Pipeline Integrity

All YAML data MUST be loaded with `yaml.safe_load()` and files
opened with `encoding="utf-8"`.

- Dictionary access for user-supplied or parsed data MUST use
  `dict.get(key, default)` with safe fallback values.
- Student identity mapping (S001 → real name) MUST be done locally;
  real names MUST NOT appear in any data sent to external APIs.
- Data joining across YAML files MUST use `student_id` and
  `question_sn` as composite keys with explicit handling for
  missing entries (log warning, generate partial output).

### V. Korean Text Safety

All PDF and chart output MUST correctly render Korean text using
NanumGothic (or compatible) fonts discovered via
`forma.font_utils.find_korean_font()`.

- Text embedded in ReportLab `Paragraph` MUST be XML-escaped
  via `xml.sax.saxutils.escape()` to prevent OCR artifact crashes.
- matplotlib charts MUST use `FontProperties(fname=...)` for
  Korean axis labels and titles.
- File names containing Korean characters MUST be sanitized
  to remove OS-illegal characters (`< > : " / \ | ? *`).

### VI. LLM Boundary Separation

Post-processing modules (PDF generation, chart rendering,
data aggregation) MUST NOT invoke any LLM API calls. These
modules read only pre-computed YAML result files.

- The `report_data_loader`, `report_charts`, and
  `student_report` modules MUST have zero imports from
  `anthropic`, `google.genai`, or any LLM client library.
- LLM-dependent analysis (L2 feedback, rubric scoring)
  is exclusively handled by the evaluation pipeline
  (`pipeline_evaluation.py`).

### VII. Pattern Consistency

New modules MUST follow the coding patterns established
in the existing codebase:

- Google-style docstrings with `Args:`, `Returns:`, `Raises:`
  sections on all public functions and classes.
- Type hints on all function signatures and return types.
- Logging via `logging.getLogger(__name__)`.
- `matplotlib.use("Agg")` MUST be called before any
  `matplotlib.pyplot` import.
- `plt.close(fig)` MUST be called after every `fig.savefig()`
  to prevent memory leaks.
- Font registration MUST follow the pattern in
  `report_generator.py` lines 46-57.
- Understanding level color mapping MUST reuse `_LEVEL_COLORS`:
  Advanced=#2E7D32, Proficient=#1565C0, Developing=#F57F17,
  Beginning=#C62828.

## Technology Constraints

- **Language**: Python >=3.11, <4
- **PDF Generation**: ReportLab >=4.4.4 (Platypus API:
  `SimpleDocTemplate`, `Paragraph`, `Table`, `Image`)
- **Charts**: matplotlib >=3.10.0 with Agg backend
- **Data Format**: YAML (PyYAML >=6.0) for all pipeline
  input/output
- **Testing**: pytest >=8.3.4, pytest-cov >=6.0.0
- **Linting**: pylint >=3.3.3, black >=24.10.0
- **Package Build**: hatchling (via `pyproject.toml`)
- **Font**: NanumGothic (system-installed, auto-detected)
- **Target Platform**: Linux (NixOS primary), macOS, Windows
- **No Runtime Network**: PDF/chart generation MUST work
  fully offline

## Development Workflow

1. **Branch per feature**: Create a feature branch from `master`
   before starting work.
2. **Write tests first**: Create `tests/test_<module>.py` with
   failing tests before any implementation.
3. **Implement incrementally**: One module at a time, following
   the phase order in `tasks.md`.
4. **Run full test suite**: `pytest tests/ -v` must pass with
   zero failures before any commit.
5. **Code review via multi-agent**: Use Tester, Audit, and
   Adversary agents per the `/coding-standards`,
   `/code-review-assistant`, and `/failure-case-validator`
   skills.
6. **Commit conventions**: Imperative mood, prefixed with
   `feat:`, `fix:`, `docs:`, `test:`, `refactor:`.

## Governance

This Constitution supersedes all ad-hoc practices for the
formative-analysis project. All code contributions and AI-agent
generated code MUST comply with these principles.

- **Amendment process**: Any principle change MUST be documented
  in this file with version increment, ratification date update,
  and propagation to dependent templates.
- **Versioning**: MAJOR for principle removal/redefinition,
  MINOR for new principle addition, PATCH for clarifications.
- **Compliance review**: Every PR and `/speckit.implement` run
  MUST verify compliance against all seven principles.
- **Runtime guidance**: Use `CLAUDE.md` (when created) for
  session-specific development guidance that supplements
  but does not override this Constitution.

**Version**: 1.0.0 | **Ratified**: 2026-03-07 | **Last Amended**: 2026-03-07
