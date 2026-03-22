"""JSONL log reader for the governance dashboard.

Parses structured trace files produced by
:class:`~src.observability.governance.GovernanceCallbackHandler`
into pandas DataFrames for visualization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns returned by each function (used for empty-DataFrame fallbacks)
_RUN_COLUMNS = [
    "run_id",
    "date",
    "start_time",
    "end_time",
    "n_tool_calls",
    "n_errors",
    "total_duration_ms",
    "total_tokens",
]

_EVENT_COLUMNS = [
    "timestamp",
    "run_id",
    "event_type",
    "tool_name",
    "input_summary",
    "output_summary",
    "duration_ms",
    "tokens",
    "error",
]

_ANALYTICS_COLUMNS = [
    "tool_name",
    "call_count",
    "avg_duration_ms",
    "p95_duration_ms",
    "error_count",
    "error_rate",
]


def list_runs(log_dir: Path) -> pd.DataFrame:
    """Scan JSONL trace files and return a summary DataFrame of all runs.

    Args:
        log_dir: Root directory containing ``{YYYY-MM-DD}/{run_id}.jsonl`` files.

    Returns:
        DataFrame with columns: run_id, date, start_time, end_time,
        n_tool_calls, n_errors, total_duration_ms, total_tokens.
    """
    if not log_dir.exists():
        return pd.DataFrame(columns=_RUN_COLUMNS)

    runs: list[dict] = []
    for jsonl_path in sorted(log_dir.glob("*/*.jsonl")):
        try:
            events = _read_jsonl(jsonl_path)
        except Exception:
            logger.warning("Skipping unreadable file: %s", jsonl_path)
            continue

        if not events:
            continue

        date_str = jsonl_path.parent.name
        run_id = jsonl_path.stem
        tool_ends = [e for e in events if e.get("event_type") == "tool_end"]
        tool_errors = [e for e in events if e.get("event_type") == "tool_error"]
        # Use max chain_end duration as total run time (outermost chain)
        chain_ends = [e for e in events if e.get("event_type") == "chain_end"]
        chain_durations = [e.get("duration_ms", 0) or 0 for e in chain_ends]
        tool_duration = sum(e.get("duration_ms", 0) or 0 for e in tool_ends)
        total_duration = max(chain_durations) if chain_durations else tool_duration
        total_tokens = 0
        for e in events:
            tok = e.get("tokens")
            if isinstance(tok, dict):
                total_tokens += tok.get("total", 0)

        runs.append(
            {
                "run_id": run_id,
                "date": date_str,
                "start_time": events[0].get("timestamp"),
                "end_time": events[-1].get("timestamp"),
                "n_tool_calls": len(tool_ends),
                "n_errors": len(tool_errors),
                "total_duration_ms": round(total_duration, 1),
                "total_tokens": total_tokens,
            }
        )

    if not runs:
        return pd.DataFrame(columns=_RUN_COLUMNS)
    return pd.DataFrame(runs)


def load_run_events(log_path: Path) -> pd.DataFrame:
    """Read all events from a single JSONL trace file.

    Args:
        log_path: Path to a ``{run_id}.jsonl`` file.

    Returns:
        DataFrame with one row per event.
    """
    events = _read_jsonl(log_path)
    if not events:
        return pd.DataFrame(columns=_EVENT_COLUMNS)
    return pd.DataFrame(events)


def compute_tool_analytics(log_dir: Path) -> pd.DataFrame:
    """Aggregate tool usage statistics across all runs.

    Args:
        log_dir: Root directory containing trace files.

    Returns:
        DataFrame with per-tool stats: call_count, avg_duration_ms,
        p95_duration_ms, error_count, error_rate.
    """
    if not log_dir.exists():
        return pd.DataFrame(columns=_ANALYTICS_COLUMNS)

    all_events: list[dict] = []
    for jsonl_path in log_dir.glob("*/*.jsonl"):
        try:
            all_events.extend(_read_jsonl(jsonl_path))
        except Exception:
            logger.warning("Skipping unreadable file: %s", jsonl_path)

    if not all_events:
        return pd.DataFrame(columns=_ANALYTICS_COLUMNS)

    tool_ends = [e for e in all_events if e.get("event_type") == "tool_end"]
    tool_errors = [e for e in all_events if e.get("event_type") == "tool_error"]

    if not tool_ends and not tool_errors:
        return pd.DataFrame(columns=_ANALYTICS_COLUMNS)

    df_ends = pd.DataFrame(tool_ends)
    df_errors = pd.DataFrame(tool_errors)

    # Build per-tool stats from tool_end events
    stats: list[dict] = []
    all_tool_names: set[str] = set()
    if not df_ends.empty and "tool_name" in df_ends.columns:
        all_tool_names.update(df_ends["tool_name"].dropna().unique())
    if not df_errors.empty and "tool_name" in df_errors.columns:
        all_tool_names.update(df_errors["tool_name"].dropna().unique())

    for name in sorted(all_tool_names):
        ends = (
            df_ends[df_ends["tool_name"] == name]
            if not df_ends.empty and "tool_name" in df_ends.columns
            else pd.DataFrame()
        )
        errors = (
            df_errors[df_errors["tool_name"] == name]
            if not df_errors.empty and "tool_name" in df_errors.columns
            else pd.DataFrame()
        )

        call_count = len(ends)
        error_count = len(errors)
        total = call_count + error_count

        durations = (
            ends["duration_ms"].dropna()
            if not ends.empty and "duration_ms" in ends.columns
            else pd.Series(dtype=float)
        )

        stats.append(
            {
                "tool_name": name,
                "call_count": call_count,
                "avg_duration_ms": (round(durations.mean(), 1) if len(durations) > 0 else 0),
                "p95_duration_ms": (
                    round(durations.quantile(0.95), 1) if len(durations) > 0 else 0
                ),
                "error_count": error_count,
                "error_rate": (round(error_count / total, 3) if total > 0 else 0),
            }
        )

    return pd.DataFrame(stats)


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    events: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
