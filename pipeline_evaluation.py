"""Backward-compatible wrapper — real module moved to src/forma/pipeline_evaluation.py."""
from forma.pipeline_evaluation import main, run_evaluation_pipeline  # noqa: F401

if __name__ == "__main__":
    main()
