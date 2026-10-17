#!/usr/bin/env python3
"""CLI Entrypoint to execute the DeepEval 2-Pillar RAG Evaluation Framework.

Usage examples:
  # Run offline fast benchmark using deterministic mock models (no API tokens required)
  uv run python scripts/run_rag_evaluation.py --mock

  # Run live evaluation against Gemini and Groq
  uv run python scripts/run_rag_evaluation.py --live

  # Run only generation metrics (Faithfulness, Hallucination, Relevancy)
  uv run python scripts/run_rag_evaluation.py --mock --metrics generation

  # Save evaluation report to JSON
  uv run python scripts/run_rag_evaluation.py --mock --output eval_report.json
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.rag.evaluation import main  # noqa: E402

if __name__ == "__main__":
    main()
