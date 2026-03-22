"""Unit tests for governance-related AgentConfig fields."""

from __future__ import annotations

from pathlib import Path

from src.agent.config import AgentConfig


def test_governance_config_defaults() -> None:
    config = AgentConfig()
    assert config.max_tokens_per_execution == 50_000
    assert config.execution_timeout_seconds == 300
    assert config.max_consecutive_failures == 3
    assert config.trace_log_dir == Path("data/logs/agent_traces")


def test_governance_config_from_env(monkeypatch: object) -> None:
    import pytest

    mp = pytest.MonkeyPatch() if not hasattr(monkeypatch, "setenv") else monkeypatch  # type: ignore[attr-defined]
    mp.setenv("AGENT_MAX_TOKENS", "100000")  # type: ignore[union-attr]
    mp.setenv("AGENT_TIMEOUT_SECONDS", "600")  # type: ignore[union-attr]
    mp.setenv("AGENT_MAX_FAILURES", "5")  # type: ignore[union-attr]
    mp.setenv("AGENT_TRACE_LOG_DIR", "/tmp/traces")  # type: ignore[union-attr]

    config = AgentConfig()
    assert config.max_tokens_per_execution == 100_000
    assert config.execution_timeout_seconds == 600
    assert config.max_consecutive_failures == 5
    assert config.trace_log_dir == Path("/tmp/traces")
