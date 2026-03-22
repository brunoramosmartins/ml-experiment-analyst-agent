"""Governance callback handler for structured agent tracing.

Captures tool calls, LLM usage, and chain events as structured JSONL logs.
Enforces execution limits (token budget, consecutive failures).

Usage:
    handler = GovernanceCallbackHandler(
        run_id="abc123",
        log_dir=Path("data/logs/agent_traces"),
        max_tokens=50000,
        max_consecutive_failures=3,
    )
    agent.invoke({"messages": [...]}, config={"callbacks": [handler]})
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class GovernanceLimitError(Exception):
    """Raised when a governance limit (tokens, consecutive failures) is exceeded."""


class GovernanceCallbackHandler(BaseCallbackHandler):
    """Structured JSONL tracer with execution limit enforcement.

    Logs every tool call, LLM completion, and chain event to a JSONL file
    under ``{log_dir}/{YYYY-MM-DD}/{run_id}.jsonl``.

    Raises :class:`GovernanceLimitError` when:
    - Total token usage exceeds ``max_tokens``.
    - Consecutive tool errors reach ``max_consecutive_failures``.
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        log_dir: Path = Path("data/logs/agent_traces"),
        max_tokens: int = 50_000,
        max_consecutive_failures: int = 3,
    ) -> None:
        super().__init__()
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.log_dir = log_dir
        self.max_tokens = max_tokens
        self.max_consecutive_failures = max_consecutive_failures

        self._tool_start_times: dict[str, float] = {}
        self._total_tokens: int = 0
        self._consecutive_failures: int = 0

        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        self._log_file = self.log_dir / date_str / f"{self.run_id}.jsonl"

    # ─── Tool hooks ─────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        call_id = str(run_id) if run_id else uuid.uuid4().hex
        self._tool_start_times[call_id] = time.monotonic()
        self._write_event(
            event_type="tool_start",
            tool_name=serialized.get("name", "unknown"),
            input_summary=_truncate(input_str, 200),
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        call_id = str(run_id) if run_id else ""
        start = self._tool_start_times.pop(call_id, None)
        duration_ms = (
            round((time.monotonic() - start) * 1000, 1) if start else None
        )
        self._consecutive_failures = 0
        self._write_event(
            event_type="tool_end",
            output_summary=_truncate(str(output), 500),
            duration_ms=duration_ms,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._consecutive_failures += 1
        self._write_event(
            event_type="tool_error",
            error=str(error),
        )
        if self._consecutive_failures >= self.max_consecutive_failures:
            raise GovernanceLimitError(
                f"Consecutive tool failures reached limit "
                f"({self.max_consecutive_failures})"
            )

    # ─── LLM hooks ──────────────────────────────────────────────────────────

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tokens = _extract_token_usage(response)
        if tokens:
            self._total_tokens += tokens.get("total", 0)
        self._write_event(
            event_type="llm_end",
            tokens=tokens,
        )
        if self._total_tokens > self.max_tokens:
            raise GovernanceLimitError(
                f"Token budget exceeded: {self._total_tokens} > {self.max_tokens}"
            )

    # ─── Chain hooks ─────────────────────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        chain_name = serialized.get("id", ["unknown"])[-1]
        self._write_event(
            event_type="chain_start",
            tool_name=chain_name,
            input_summary=_truncate(str(inputs), 200),
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write_event(
            event_type="chain_end",
            output_summary=_truncate(str(outputs), 500),
        )

    # ─── Internal ────────────────────────────────────────────────────────────

    def _write_event(
        self,
        *,
        event_type: str,
        tool_name: str | None = None,
        input_summary: str | None = None,
        output_summary: str | None = None,
        duration_ms: float | None = None,
        tokens: dict[str, int] | None = None,
        error: str | None = None,
    ) -> None:
        """Append a single JSON event to the JSONL log file."""
        event = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event_type": event_type,
            "tool_name": tool_name,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "duration_ms": duration_ms,
            "tokens": tokens,
            "error": error,
        }
        try:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError as exc:
            logger.warning("Failed to write governance log: %s", exc)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed so far in this run."""
        return self._total_tokens

    @property
    def consecutive_failures(self) -> int:
        """Current consecutive tool failure count."""
        return self._consecutive_failures


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, appending '...' if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _extract_token_usage(response: LLMResult) -> dict[str, int] | None:
    """Extract token usage from an LLMResult, if available."""
    llm_output = response.llm_output or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage")
    if not usage:
        return None
    return {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }
