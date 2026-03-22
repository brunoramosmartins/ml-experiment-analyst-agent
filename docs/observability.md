# Observability

The agent uses two complementary observability layers:

| Layer | What it captures | Where to see it |
|---|---|---|
| **LangSmith** | Every LLM call, tool invocation, token count, latency | [smith.langchain.com](https://smith.langchain.com/) |
| **GovernanceCallbackHandler** | Structured JSONL per tool call | `data/logs/agent_traces/` |

---

## LangSmith Setup

LangSmith tracing is enabled by setting two environment variables:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=ml-experiment-analyst
```

No code changes required — LangGraph/deepagents automatically sends traces
when `LANGCHAIN_TRACING_V2=true`.

Disable in CI to avoid sending traces from automated runs:

```yaml
# .github/workflows/ci.yml
LANGCHAIN_TRACING_V2: "false"
```

---

## What is captured automatically

For each agent invocation, LangSmith records:

- **Input message** — the user's natural language query
- **LLM calls** — prompt, completion, model, token usage, latency
- **Tool calls** — tool name, input arguments, output string
- **Tool call sequence** — the order in which the agent called tools
- **Total tokens** — aggregated across the full run
- **Errors** — any exception raised during tool execution

---

## Custom metadata (Phase 1+)

Use the helpers in `src/observability/langsmith.py` to attach domain-specific
metadata to the active trace:

```python
from src.observability.langsmith import add_run_metadata, tag_trace

# Tag the trace with the current analysis context
add_run_metadata("experiment_name", "binary-classification")
add_run_metadata("n_runs_analyzed", 17)
tag_trace(["mlflow-analysis", "phase-1"])
```

These values appear in the LangSmith run detail view under **Metadata** and **Tags**.

---

## Accessing traces

1. Open [smith.langchain.com](https://smith.langchain.com/)
2. Navigate to the `ml-experiment-analyst` project
3. Click any run to see the full trace tree
4. Expand tool calls to inspect inputs and outputs

---

## Local JSONL Tracing (GovernanceCallbackHandler)

The `GovernanceCallbackHandler` (`src/observability/governance.py`) captures
tool-level events as structured JSONL logs under `data/logs/agent_traces/`.

Use `invoke_with_governance()` to automatically attach the handler:

```python
from src.agent.builder import create_analyst_agent, invoke_with_governance

agent = create_analyst_agent()
result = invoke_with_governance(agent, "Analyze the binary-classification experiment")
```

See [governance.md](governance.md) for the full JSONL schema and configuration.

---

## Governance Dashboard

A Streamlit dashboard provides a visual interface for the JSONL trace logs:

```bash
streamlit run src/dashboard/app.py
```

**Run Explorer** — browse executions, inspect tool call timelines, view
inputs/outputs for each step.

**Tool Analytics** — call frequency, average latency, and error rates per tool.
