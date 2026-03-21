# Changelog

All notable changes to FormA are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.13.0] — 2026-03-22

### Added
- **Domain delivery analysis v3**: LLM-based delivery analysis, concept network graphs, cross-section statistics, hierarchical visualization, 9-section PDF report
- **Domain coverage analysis**: Textbook concept extraction (v2 LLM-based), 4-state coverage classification (covered/gap/skipped/extra), Spearman correlation, PDF report
- **Student longitudinal report**: Per-student multi-week PDF with charts, anonymized LLM interpretation, early warning indicators, cohort summary
- **STT lecture analysis** (`forma lecture`): Transcript preprocessing, keyword/topic/network analysis, section comparison, class-level comparison
- **Backfill longitudinal** (`forma backfill longitudinal`): Populate longitudinal store from past evaluation results
- **GitHub Pages documentation site**: MkDocs Material theme with auto-deploy

### Changed
- Unified CLI now has 22 commands (was 15)
- All CLI messages, logging output, and help text are in English
- Renamed `_korean_error()` → `_error_message()` in CLI main
- Version resolution uses `importlib.metadata` with fallback
- Updated all documentation to match current CLI commands and flags

### Removed
- Dead modules: `cohesion_analysis.py`, `visualize_cohesion.py`, `tesseract_processor.py`
- Dead functions: `_fuzzy_section_match()`, 5 unused functions in `knowledge_graph_analysis.py`
- Unused function parameters in internal helpers

### Fixed
- 11 incorrect CLI examples in `docs/for_new_teachers.md`
- Version synchronized across `pyproject.toml`, `cli_main.py`, `README.md`
- `--font-path` help text corrected

## [0.12.2] — 2026-03-08

### Added
- Unified `forma` CLI entry point replacing all `forma-*` legacy commands
- Legacy `forma-*` commands emit `DeprecationWarning`
- Adversarial test suite (159 tests, 12 personas)
- Updated documentation for unified CLI

## [0.11.1] — 2026-03-07

### Added
- Credential consolidation: `forma.json` replaces scattered config files
- CI integration tests for credential handling

## [0.11.0] — 2026-03-06

### Added
- Email delivery system (`forma deliver prepare/send`)
- SMTP configuration, roster-based delivery, dry-run mode
- Delivery logging and summary email notification
- 235 delivery tests (unit, integration, adversary)

## [0.10.0] — 2026-03-05

### Added
- Intervention tracking (`forma intervention add/list/update`)
- Intervention effect analysis with pre/post score comparison
- Concept dependency DAG with cycle detection
- Learning path generation based on deficit concepts
- Grade prediction model (`forma train grade`)
- Grade and intervention sections in professor/student/longitudinal reports

## [0.9.0] — 2026-03-04

### Added
- Project configuration system (`forma init`, `forma.yaml`)
- Drop-risk prediction model (`forma train risk`)
- Early warning report (`forma report warning`)
- Section comparison with Welch's t-test / Mann-Whitney U
- Risk type classification (score decline, persistent low, concept deficit, participation decline)

## [0.8.1] — 2026-03-03

### Fixed
- Feedback quality: encouraging tone, structured sections, truncation prevention

## [0.8.0] — 2026-03-02

### Added
- Longitudinal analysis: weekly tracking, trend reports, risk movement detection
- Longitudinal report with multi-week charts

## [0.7.3] — 2026-02-28

### Added
- Class-level knowledge graph clustering
- Misconception clustering analysis
- Graph visualization improvements

## [0.7.2] — 2026-02-27

### Added
- Misconception analysis module

## [0.7.1] — 2026-02-26

### Added
- Multi-class batch evaluation and reporting
- Graph visualization with centrality gap analysis
