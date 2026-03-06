"""Backward-compatible wrapper — real module moved to src/pipeline_batch_evaluation.py."""
from src.pipeline_batch_evaluation import main, run_batch_evaluation  # noqa: F401

if __name__ == "__main__":
    main()
