# formative-analysis Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-07

## Active Technologies
- Python >=3.11, <4 + scikit-learn>=1.4.0 (KMeans), networkx>=3.4.2 (DiGraph), numpy<2.1.0 (embedding matrix), matplotlib>=3.10.0 (Agg backend), ReportLab>=4.4.4 (Platypus API) — 전부 기존 deps (004-class-graph-clustering)
- YAML (PyYAML>=6.0) — 기존 파이프라인 출력 파일 읽기 전용; 신규 저장소 없음 (004-class-graph-clustering)

- Python >=3.11, <4 + ReportLab >=4.4.4 (Platypus API), matplotlib >=3.10.0 (Agg backend), PyYAML >=6.0 (001-student-pdf-report)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

### CLI Commands

- `forma-report-batch` — multi-class batch report generator

  Usage:
  ```
  forma-report-batch --config <exam.yaml> --join-dir <dir> --join-pattern "<pattern with {class}>" \
    --eval-pattern "<pattern with {class}>" --output-dir <out> --classes A B [--aggregate] [--no-individual]
  ```

## Code Style

Python >=3.11, <4: Follow standard conventions

## Recent Changes
- 004-class-graph-clustering: Added Python >=3.11, <4 + scikit-learn>=1.4.0 (KMeans), networkx>=3.4.2 (DiGraph), numpy<2.1.0 (embedding matrix), matplotlib>=3.10.0 (Agg backend), ReportLab>=4.4.4 (Platypus API) — 전부 기존 deps
- 002-multi-class-batch-viz: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]

- 001-student-pdf-report: Added Python >=3.11, <4 + ReportLab >=4.4.4 (Platypus API), matplotlib >=3.10.0 (Agg backend), PyYAML >=6.0

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
