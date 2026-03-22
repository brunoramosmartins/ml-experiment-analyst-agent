from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for the ML Experiment Analyst Agent.

    All values default to environment variables so the agent can be configured
    without changing code. See .env.example for the full list of variables.
    """

    # ─── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.1:8b"))
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    temperature: float = 0.0
    max_tokens: int = 4096

    # ─── Filesystem backend ───────────────────────────────────────────────────
    workspace_path: Path = field(
        default_factory=lambda: Path(os.getenv("AGENT_WORKSPACE_PATH", "data/agent-workspace"))
    )

    # ─── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    # ─── Governance ─────────────────────────────────────────────────────────
    max_tokens_per_execution: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_TOKENS", "50000"))
    )
    execution_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("AGENT_TIMEOUT_SECONDS", "300"))
    )
    max_consecutive_failures: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_FAILURES", "3"))
    )
    trace_log_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AGENT_TRACE_LOG_DIR", "data/logs/agent_traces"))
    )
