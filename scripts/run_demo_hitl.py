#!/usr/bin/env python3
"""Demonstrate Human-in-the-Loop (HITL) with the ML Experiment Analyst Agent.

The agent is configured to pause before calling ``generate_report``, allowing
the user to approve, edit, or reject the report generation.

Prerequisites:
    - MLflow running and seeded (``python scripts/seed_mlflow.py``)
    - ``.env`` configured (see .env.example)

Usage::

    python scripts/run_demo_hitl.py
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

QUERY = (
    "Analyze the binary-classification experiment and generate a full "
    "report with overfitting diagnostics and suggestions."
)


def main() -> int:
    """Run the HITL demo."""
    from src.agent.builder import create_analyst_agent

    logger.info("Creating agent with HITL on generate_report...")
    try:
        agent = create_analyst_agent(hitl_tools=["generate_report"])
    except Exception as exc:
        logger.error("Failed to create agent: %s", exc)
        return 1

    logger.info("Sending query: %s", QUERY[:80])
    logger.info("The agent will pause before generating the report.")
    logger.info("-" * 60)

    try:
        # First invocation — agent will work until it tries to call
        # generate_report, then pause (interrupt).
        result = agent.invoke(
            {"messages": [{"role": "user", "content": QUERY}]}
        )

        # Check if the agent was interrupted (pending tool call)
        messages = result.get("messages", [])
        last_msg = messages[-1] if messages else None

        # When interrupted, the last message has pending tool_calls
        pending_calls = getattr(last_msg, "tool_calls", [])
        report_call = None
        for call in pending_calls:
            if call.get("name") == "generate_report":
                report_call = call
                break

        if not report_call:
            logger.info("Agent completed without triggering HITL.")
            if hasattr(last_msg, "content"):
                logger.info("Response: %s", last_msg.content[:300])
            return 0

        # Show the pending report parameters
        logger.info("Agent wants to generate a report with:")
        args = report_call.get("args", {})
        for key, value in args.items():
            logger.info("  %s: %s", key, value)

        # Ask user for approval
        logger.info("-" * 60)
        choice = input("Approve report generation? [y/n/edit]: ").strip().lower()

        if choice == "y":
            logger.info("Approved. Resuming agent...")
            # Resume by re-invoking — the agent continues from the
            # interrupted state.
            final_result = agent.invoke(None)
            final_messages = final_result.get("messages", [])
            if final_messages:
                last = final_messages[-1]
                if hasattr(last, "content"):
                    logger.info("Final response: %s", last.content[:500])
            logger.info("Report generated successfully.")

        elif choice == "edit":
            new_title = input("Enter new report title: ").strip()
            if new_title:
                report_call["args"]["report_title"] = new_title
            logger.info("Resuming with edited parameters...")
            final_result = agent.invoke(None)
            final_messages = final_result.get("messages", [])
            if final_messages:
                last = final_messages[-1]
                if hasattr(last, "content"):
                    logger.info("Final response: %s", last.content[:500])

        else:
            logger.info("Rejected. Aborting report generation.")
            return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 1
    except Exception as exc:
        logger.error("Error during HITL demo: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
