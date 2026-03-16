"""LangSmith observability helpers.

Provides lightweight wrappers around LangSmith's run-context APIs to add
custom metadata and tags to active traces without importing langsmith
directly in every module.

Usage:
    from src.observability.langsmith import add_run_metadata, tag_trace

    add_run_metadata("experiment_name", "binary-classification")
    tag_trace(["phase-1", "mlflow-analysis"])
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def add_run_metadata(key: str, value: Any) -> None:
    """Add a key-value pair to the currently active LangSmith trace.

    No-ops silently if LangSmith is not configured or no trace is active,
    so this is safe to call unconditionally.

    Args:
        key: Metadata key (e.g. "experiment_name").
        value: Metadata value (must be JSON-serialisable).
    """
    try:
        from langsmith import get_current_run_tree

        run = get_current_run_tree()
        if run is not None:
            run.add_metadata({key: value})
    except ImportError:
        logger.debug("langsmith not installed — skipping add_run_metadata")
    except Exception as exc:
        logger.debug("add_run_metadata failed (non-critical): %s", exc)


def tag_trace(tags: list[str]) -> None:
    """Add tags to the currently active LangSmith trace.

    No-ops silently if LangSmith is not configured.

    Args:
        tags: List of string tags to attach to the current run.
    """
    try:
        from langsmith import get_current_run_tree

        run = get_current_run_tree()
        if run is not None:
            existing = list(run.tags or [])
            run.tags = list(set(existing + tags))
    except ImportError:
        logger.debug("langsmith not installed — skipping tag_trace")
    except Exception as exc:
        logger.debug("tag_trace failed (non-critical): %s", exc)
