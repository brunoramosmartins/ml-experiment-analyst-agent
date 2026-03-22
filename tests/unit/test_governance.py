"""Unit tests for GovernanceCallbackHandler."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.observability.governance import (
    GovernanceCallbackHandler,
    GovernanceLimitError,
)


@pytest.fixture()
def handler(tmp_path: Path) -> GovernanceCallbackHandler:
    return GovernanceCallbackHandler(
        run_id="test-run-123",
        log_dir=tmp_path,
        max_tokens=1000,
        max_consecutive_failures=2,
    )


def _find_log_file(tmp_path: Path) -> Path:
    """Find the single JSONL file created in tmp_path."""
    files = list(tmp_path.glob("*/*.jsonl"))
    assert len(files) == 1, f"Expected 1 log file, found {len(files)}"
    return files[0]


def _read_events(log_path: Path) -> list[dict]:
    """Read all events from a JSONL file."""
    events = []
    with log_path.open() as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


# ─── on_tool_start ───────────────────────────────────────────────────────────


def test_on_tool_start_creates_log_file(handler: GovernanceCallbackHandler, tmp_path: Path) -> None:
    handler.on_tool_start(
        {"name": "load_experiment"},
        '{"experiment_name": "test"}',
        run_id=uuid.uuid4(),
    )
    log_file = _find_log_file(tmp_path)
    events = _read_events(log_file)
    assert len(events) == 1
    assert events[0]["event_type"] == "tool_start"
    assert events[0]["tool_name"] == "load_experiment"


# ─── on_tool_end ─────────────────────────────────────────────────────────────


def test_on_tool_end_logs_duration(handler: GovernanceCallbackHandler, tmp_path: Path) -> None:
    rid = uuid.uuid4()
    handler.on_tool_start({"name": "compare_runs"}, "{}", run_id=rid)
    handler.on_tool_end("Results: ...", run_id=rid)

    log_file = _find_log_file(tmp_path)
    events = _read_events(log_file)
    end_event = [e for e in events if e["event_type"] == "tool_end"][0]
    assert end_event["duration_ms"] is not None
    assert end_event["duration_ms"] >= 0


# ─── on_tool_error ───────────────────────────────────────────────────────────


def test_on_tool_error_increments_failures(tmp_path: Path) -> None:
    handler = GovernanceCallbackHandler(
        run_id="test",
        log_dir=tmp_path,
        max_tokens=1000,
        max_consecutive_failures=5,  # High limit so it doesn't raise
    )
    handler.on_tool_error(RuntimeError("connection failed"), run_id=uuid.uuid4())
    assert handler.consecutive_failures == 1

    handler.on_tool_error(RuntimeError("timeout"), run_id=uuid.uuid4())
    assert handler.consecutive_failures == 2


def test_max_consecutive_failures_raises(
    handler: GovernanceCallbackHandler,
) -> None:
    handler.on_tool_error(RuntimeError("fail 1"), run_id=uuid.uuid4())
    with pytest.raises(GovernanceLimitError, match="Consecutive tool failures"):
        handler.on_tool_error(RuntimeError("fail 2"), run_id=uuid.uuid4())


# ─── on_llm_end ──────────────────────────────────────────────────────────────


def test_on_llm_end_tracks_tokens(
    handler: GovernanceCallbackHandler,
) -> None:
    mock_response = MagicMock()
    mock_response.llm_output = {
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    }
    handler.on_llm_end(mock_response, run_id=uuid.uuid4())
    assert handler.total_tokens == 150


def test_token_limit_exceeded_raises(
    handler: GovernanceCallbackHandler,
) -> None:
    mock_response = MagicMock()
    mock_response.llm_output = {
        "token_usage": {
            "prompt_tokens": 800,
            "completion_tokens": 300,
            "total_tokens": 1100,
        }
    }
    with pytest.raises(GovernanceLimitError, match="Token budget exceeded"):
        handler.on_llm_end(mock_response, run_id=uuid.uuid4())


# ─── Reset on success ────────────────────────────────────────────────────────


def test_successful_tool_resets_failure_count(
    handler: GovernanceCallbackHandler,
) -> None:
    handler.on_tool_error(RuntimeError("fail"), run_id=uuid.uuid4())
    assert handler.consecutive_failures == 1

    rid = uuid.uuid4()
    handler.on_tool_start({"name": "test"}, "{}", run_id=rid)
    handler.on_tool_end("ok", run_id=rid)
    assert handler.consecutive_failures == 0


# ─── Log schema ──────────────────────────────────────────────────────────────


def test_log_entry_schema_has_all_fields(
    handler: GovernanceCallbackHandler, tmp_path: Path
) -> None:
    handler.on_tool_start({"name": "diagnose_run"}, '{"run_id": "abc"}', run_id=uuid.uuid4())

    log_file = _find_log_file(tmp_path)
    events = _read_events(log_file)
    event = events[0]

    expected_keys = {
        "timestamp",
        "run_id",
        "event_type",
        "tool_name",
        "input_summary",
        "output_summary",
        "duration_ms",
        "tokens",
        "error",
    }
    assert set(event.keys()) == expected_keys


# ─── IO error resilience ─────────────────────────────────────────────────────


def test_io_error_does_not_crash(tmp_path: Path) -> None:
    # Point to a file path that will cause an IO error
    handler = GovernanceCallbackHandler(
        run_id="test",
        log_dir=tmp_path / "nonexistent" / "deep" / "path",
        max_tokens=50000,
        max_consecutive_failures=3,
    )
    # Should not raise — IO errors are caught and logged
    handler.on_tool_start({"name": "test"}, "{}", run_id=uuid.uuid4())
