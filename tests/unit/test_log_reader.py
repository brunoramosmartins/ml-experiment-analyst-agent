"""Unit tests for the dashboard log reader."""

from __future__ import annotations

import json
from pathlib import Path

from src.dashboard.log_reader import (
    compute_tool_analytics,
    list_runs,
    load_run_events,
)


def _write_events(path: Path, events: list[dict]) -> None:
    """Write a list of events as JSONL to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


SAMPLE_EVENTS = [
    {
        "timestamp": "2026-03-22T10:00:00+00:00",
        "run_id": "run-001",
        "event_type": "chain_start",
        "tool_name": "AgentExecutor",
        "input_summary": "Analyze...",
        "output_summary": None,
        "duration_ms": None,
        "tokens": None,
        "error": None,
    },
    {
        "timestamp": "2026-03-22T10:00:01+00:00",
        "run_id": "run-001",
        "event_type": "tool_start",
        "tool_name": "load_experiment",
        "input_summary": '{"experiment_name": "test"}',
        "output_summary": None,
        "duration_ms": None,
        "tokens": None,
        "error": None,
    },
    {
        "timestamp": "2026-03-22T10:00:02+00:00",
        "run_id": "run-001",
        "event_type": "tool_end",
        "tool_name": "load_experiment",
        "input_summary": None,
        "output_summary": "Loaded 5 runs",
        "duration_ms": 1200.5,
        "tokens": None,
        "error": None,
    },
    {
        "timestamp": "2026-03-22T10:00:03+00:00",
        "run_id": "run-001",
        "event_type": "llm_end",
        "tool_name": None,
        "input_summary": None,
        "output_summary": None,
        "duration_ms": None,
        "tokens": {"prompt": 500, "completion": 100, "total": 600},
        "error": None,
    },
    {
        "timestamp": "2026-03-22T10:00:04+00:00",
        "run_id": "run-001",
        "event_type": "chain_end",
        "tool_name": None,
        "input_summary": None,
        "output_summary": "Analysis complete",
        "duration_ms": None,
        "tokens": None,
        "error": None,
    },
]


# ─── list_runs ───────────────────────────────────────────────────────────────


def test_list_runs_empty_dir(tmp_path: Path) -> None:
    df = list_runs(tmp_path)
    assert len(df) == 0
    assert "run_id" in df.columns
    assert "total_tokens" in df.columns


def test_list_runs_parses_files(tmp_path: Path) -> None:
    _write_events(tmp_path / "2026-03-22" / "run-001.jsonl", SAMPLE_EVENTS)

    df = list_runs(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "run-001"
    assert df.iloc[0]["n_tool_calls"] == 1  # 1 tool_end event
    assert df.iloc[0]["n_errors"] == 0
    assert df.iloc[0]["total_tokens"] == 600


# ─── load_run_events ─────────────────────────────────────────────────────────


def test_load_run_events(tmp_path: Path) -> None:
    path = tmp_path / "2026-03-22" / "run-001.jsonl"
    _write_events(path, SAMPLE_EVENTS)

    df = load_run_events(path)
    assert len(df) == 5
    assert set(df["event_type"]) == {
        "chain_start",
        "tool_start",
        "tool_end",
        "llm_end",
        "chain_end",
    }


# ─── compute_tool_analytics ─────────────────────────────────────────────────


def test_compute_tool_analytics(tmp_path: Path) -> None:
    # Two runs with different tools
    events_1 = [
        {
            "timestamp": "2026-03-22T10:00:00+00:00",
            "run_id": "r1",
            "event_type": "tool_end",
            "tool_name": "load_experiment",
            "input_summary": None,
            "output_summary": "ok",
            "duration_ms": 100.0,
            "tokens": None,
            "error": None,
        },
        {
            "timestamp": "2026-03-22T10:00:01+00:00",
            "run_id": "r1",
            "event_type": "tool_end",
            "tool_name": "compare_runs",
            "input_summary": None,
            "output_summary": "ok",
            "duration_ms": 200.0,
            "tokens": None,
            "error": None,
        },
    ]
    events_2 = [
        {
            "timestamp": "2026-03-22T11:00:00+00:00",
            "run_id": "r2",
            "event_type": "tool_end",
            "tool_name": "load_experiment",
            "input_summary": None,
            "output_summary": "ok",
            "duration_ms": 150.0,
            "tokens": None,
            "error": None,
        },
        {
            "timestamp": "2026-03-22T11:00:01+00:00",
            "run_id": "r2",
            "event_type": "tool_error",
            "tool_name": "load_experiment",
            "input_summary": None,
            "output_summary": None,
            "duration_ms": None,
            "tokens": None,
            "error": "connection failed",
        },
    ]

    _write_events(tmp_path / "2026-03-22" / "r1.jsonl", events_1)
    _write_events(tmp_path / "2026-03-22" / "r2.jsonl", events_2)

    df = compute_tool_analytics(tmp_path)
    assert len(df) == 2  # load_experiment and compare_runs

    load_row = df[df["tool_name"] == "load_experiment"].iloc[0]
    assert load_row["call_count"] == 2  # 2 tool_end events
    assert load_row["error_count"] == 1
    assert load_row["avg_duration_ms"] == 125.0  # (100 + 150) / 2
