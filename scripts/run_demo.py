#!/usr/bin/env python3
"""Run 3 demonstration queries against the ML Experiment Analyst Agent.

Prerequisites:
    - MLflow running: ``make mlflow-up``
    - Demo data seeded: ``make seed-mlflow``
    - ``.env`` configured with API keys (see .env.example)

Usage::

    python scripts/run_demo.py
    # or: make run-demo
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEMO_QUERIES: list[dict[str, str]] = [
    {
        "label": "Query 1: Full analysis of binary-classification",
        "content": (
            "Analyze the binary-classification experiment and give me "
            "a complete report with overfitting diagnostics and suggestions."
        ),
    },
    {
        "label": "Query 2: Compare top runs of regression-v2",
        "content": (
            "Compare the top 3 runs of the regression-v2 experiment "
            "by val_rmse and generate a report."
        ),
    },
    {
        "label": "Query 3: Diagnose overfit-test problems",
        "content": (
            "Analyze the overfit-test experiment and identify any "
            "overfitting problems. Generate a report with your findings."
        ),
    },
]

SEPARATOR = "=" * 60


def _extract_response(result: object) -> str:
    """Extract the final message content from the agent result."""
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        if messages:
            last = messages[-1]
            return last.content if hasattr(last, "content") else str(last)
    return str(result)


def main() -> int:
    """Run all demo queries and print a summary."""
    from src.agent.builder import create_analyst_agent

    logger.info("Creating ML Experiment Analyst Agent...")
    try:
        agent = create_analyst_agent()
    except Exception as exc:
        logger.error("Failed to create agent: %s", exc)
        logger.error(
            "Check that .env is configured and the LLM provider is available."
        )
        return 1

    results: list[dict[str, object]] = []

    for i, query in enumerate(DEMO_QUERIES, 1):
        logger.info(SEPARATOR)
        logger.info("[%d/%d] %s", i, len(DEMO_QUERIES), query["label"])
        logger.info(SEPARATOR)

        start = time.time()
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": query["content"]}]}
            )
            elapsed = time.time() - start
            logger.info("Completed in %.1fs", elapsed)

            response = _extract_response(result)
            logger.info("Agent response:\n%s", response[:500])

            results.append(
                {"query": query["label"], "status": "OK", "time": elapsed}
            )
        except Exception as exc:
            elapsed = time.time() - start
            logger.error("Query failed after %.1fs: %s", elapsed, exc)
            results.append(
                {
                    "query": query["label"],
                    "status": f"FAILED: {exc}",
                    "time": elapsed,
                }
            )

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info(SEPARATOR)
    logger.info("DEMO SUMMARY")
    logger.info(SEPARATOR)
    for r in results:
        icon = "OK" if r["status"] == "OK" else "FAIL"
        logger.info("[%4s] %s (%.1fs)", icon, r["query"], r["time"])

    # ── Report check ─────────────────────────────────────────────────────────
    reports_dir = Path("data/agent-workspace/reports")
    if reports_dir.exists():
        reports = sorted(reports_dir.glob("*.md"))
        logger.info("Reports in %s: %d file(s)", reports_dir, len(reports))
        for rp in reports[-6:]:
            logger.info("  - %s (%d bytes)", rp.name, rp.stat().st_size)
    else:
        logger.warning("Reports directory not found: %s", reports_dir)

    failed = sum(1 for r in results if r["status"] != "OK")
    if failed:
        logger.error("%d/%d queries failed.", failed, len(results))
    else:
        logger.info("All %d queries completed successfully.", len(results))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
