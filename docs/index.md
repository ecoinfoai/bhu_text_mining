# FormA — Formative Assessment Toolkit

[![CI](https://github.com/ecoinfoai/formative-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/ecoinfoai/formative-analysis/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ecoinfoai/formative-analysis/graph/badge.svg?token=AhHOn36NHd)](https://codecov.io/gh/ecoinfoai/formative-analysis)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.13.0-green.svg)](https://github.com/ecoinfoai/formative-analysis/releases)

An AI-powered formative assessment CLI toolkit for university professors. FormA automates the full assessment cycle — from exam generation and OCR scanning through knowledge-graph evaluation, personalized feedback, longitudinal tracking, and report delivery — enabling data-driven instructional decisions without manual grading overhead.

## Command Overview

All commands are accessible through the unified `forma` entry point.

```
forma
├── exam              Exam PDF generation with per-student QR codes
├── ocr               OCR pipeline (scan / join / compare)
├── eval              4-layer knowledge graph evaluation pipeline
│   └── batch         Multi-class batch evaluation
├── report
│   ├── student       Individual student PDF report
│   ├── professor     Professor-facing class summary report
│   ├── longitudinal  Multi-week longitudinal trend analysis
│   ├── warning       Early warning report for at-risk students
│   └── batch         Multi-class batch report generation
├── train
│   ├── risk          Train drop-risk prediction model
│   └── grade         Train semester grade prediction model
├── lecture
│   ├── analyze       Analyze a single STT lecture transcript
│   ├── compare       Compare class sections for the same session
│   └── class-compare Compare class sections across all sessions
├── domain
│   ├── extract       Extract domain concepts from exam config
│   ├── coverage      Analyze domain coverage in lecture transcripts
│   └── report        Generate domain coverage PDF report
├── backfill
│   └── longitudinal  Backfill longitudinal store from past evaluations
├── intervention      Intervention activity tracking (add / list / update)
├── deliver           Report email delivery (prepare / send)
├── init              Interactive project configuration initialization
└── select            Question selection and exam PDF generation
```

## Quick Links

- [Quick Start](quickstart.md) — Get up and running in minutes
- [CLI Reference](cli-reference.md) — Complete command and flag reference
- [Weekly Workflow](weekly-workflow.md) — Recommended weekly assessment workflow
- [Configuration](configuration.md) — Configuration file format and options
- [Guide for New Teachers](for_new_teachers.md) — Step-by-step guide for first-time users
