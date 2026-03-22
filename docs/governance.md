# Agent Governance

The governance layer provides structured tracing, execution limits, and
auditability for every agent invocation. It complements LangSmith (which
captures LLM-level traces) with business-level logging: which tools were
called, with what inputs, for how long, and whether they succeeded.

---

## GovernanceCallbackHandler

The core component is `GovernanceCallbackHandler`
(`src/observability/governance.py`), a LangChain callback handler that
captures events as JSONL log lines.

### Events captured

| Event | Hook | Fields populated |
|-------|------|-----------------|
| `tool_start` | `on_tool_start` | tool_name, input_summary |
| `tool_end` | `on_tool_end` | output_summary, duration_ms |
| `tool_error` | `on_tool_error` | error |
| `llm_end` | `on_llm_end` | tokens |
| `chain_start` | `on_chain_start` | tool_name (chain name), input_summary |
| `chain_end` | `on_chain_end` | output_summary |

### JSONL schema

Each line in the log file is a JSON object:

```json
{
  "timestamp": "2026-03-22T14:30:00.123456+00:00",
  "run_id": "a1b2c3d4e5f6",
  "event_type": "tool_end",
  "tool_name": "load_experiment",
  "input_summary": null,
  "output_summary": "Experiment binary-classification: 12 runs...",
  "duration_ms": 1234.5,
  "tokens": null,
  "error": null
}
```

### Log file location

Logs are stored at:

```
data/logs/agent_traces/{YYYY-MM-DD}/{run_id}.jsonl
```

---

## Execution Limits

The governance layer enforces three configurable limits:

| Limit | Default | Env Var | Behavior |
|-------|---------|---------|----------|
| Token budget | 50,000 | `AGENT_MAX_TOKENS` | Raises `GovernanceLimitError` when total tokens exceed budget |
| Execution timeout | 300s | `AGENT_TIMEOUT_SECONDS` | Raises `TimeoutError` after the configured duration |
| Consecutive failures | 3 | `AGENT_MAX_FAILURES` | Raises `GovernanceLimitError` after N consecutive tool errors |

### Configuration

Set limits via environment variables in `.env`:

```bash
AGENT_MAX_TOKENS=50000
AGENT_TIMEOUT_SECONDS=300
AGENT_MAX_FAILURES=3
AGENT_TRACE_LOG_DIR=data/logs/agent_traces
```

Or pass them programmatically via `AgentConfig`:

```python
from src.agent.config import AgentConfig

config = AgentConfig(
    max_tokens_per_execution=100000,
    execution_timeout_seconds=600,
    max_consecutive_failures=5,
)
```

---

## Usage

Use `invoke_with_governance()` to automatically attach the callback handler:

```python
from src.agent.builder import create_analyst_agent, invoke_with_governance

agent = create_analyst_agent()
result = invoke_with_governance(agent, "Analyze the binary-classification experiment")
```

This will:
1. Create a fresh `GovernanceCallbackHandler` with a unique run ID
2. Pass it as a callback to `agent.invoke()`
3. Enforce token budget and timeout
4. Write JSONL logs to `data/logs/agent_traces/`

---

## Governance Dashboard

A Streamlit dashboard visualizes the trace logs:

```bash
streamlit run src/dashboard/app.py
```

See the [dashboard documentation](observability.md) for details.
