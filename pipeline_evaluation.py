"""Backward-compatible wrapper — real module moved to src/pipeline_evaluation.py."""
from src.pipeline_evaluation import main, run_evaluation_pipeline  # noqa: F401

if __name__ == "__main__":
    main()
