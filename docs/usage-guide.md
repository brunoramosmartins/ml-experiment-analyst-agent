# Usage Guide

This guide covers how to run the ML Experiment Analyst Agent, customize queries, use Human-in-the-Loop mode, and view the governance dashboard.

**Prerequisites:** Complete the [Setup Guide](SETUP.md) first.

---

## Table of Contents

- [Running the Demo](#running-the-demo)
- [Writing Custom Queries](#writing-custom-queries)
- [Human-in-the-Loop Mode](#human-in-the-loop-mode)
- [Governance Dashboard](#governance-dashboard)
- [Using Your Own MLflow Experiments](#using-your-own-mlflow-experiments)
- [Switching LLM Providers](#switching-llm-providers)
- [Enabling Web Search](#enabling-web-search)

---

## Running the Demo

The demo script runs 3 queries against the agent and generates analysis reports:

```bash
python scripts/run_demo.py
```

**What it does:**

1. Analyzes the `binary-classification` experiment (full report with overfitting diagnostics)
2. Compares the top 3 runs of `regression-v2` by `val_rmse`
3. Analyzes the `overfit-test` experiment (identifies overfitting problems)

Each query is wrapped with governance tracking. Reports are saved to `data/agent-workspace/reports/`.

**Expected output:**

```
[1/3] Analyzing binary-classification experiment...
  -> Completed in 45.2s
[2/3] Comparing regression-v2 runs...
  -> Completed in 32.1s
[3/3] Analyzing overfit-test experiment...
  -> Completed in 38.7s

Summary: 3/3 queries succeeded
Reports saved: data/agent-workspace/reports/
```

---

## Writing Custom Queries

You can write your own script to query the agent:

```python
from src.agent.builder import create_analyst_agent, invoke_with_governance
from src.agent.config import AgentConfig

# Create the agent (uses settings from .env)
config = AgentConfig()
agent = create_analyst_agent(config)

# Run a query with governance tracking
result = invoke_with_governance(
    agent,
    "Analyze the binary-classification experiment. Focus on which hyperparameters "
    "drive val_accuracy and suggest 3 next experiments.",
    config,
)

# Extract the final response
messages = result.get("messages", [])
if messages:
    print(messages[-1].content)
```

**Tips for effective queries:**

- Be specific about the experiment name (must match MLflow exactly)
- Mention the target metric if you want pattern analysis (e.g., `val_accuracy`, `val_loss`)
- Ask for reports when you want a saved Markdown file
- The agent decides which tools to call based on your query

---

## Human-in-the-Loop Mode

HITL mode pauses the agent before executing specific tools, allowing you to approve, edit, or reject the action.

```bash
python scripts/run_demo_hitl.py
```

**How it works:**

1. The agent processes your query normally
2. When it reaches a protected tool (e.g., `generate_report`), it pauses
3. You see the tool name and parameters the agent wants to use
4. You choose:
   - `[y]es` — approve and continue
   - `[e]dit` — modify the parameters before continuing
   - `[n]o` — reject and let the agent try a different approach

**Example interaction:**

```
Agent wants to call: generate_report
Parameters:
  experiment_name: binary-classification
  report_title: Binary Classification Analysis

Approve? [y]es / [e]dit / [n]o: e
Enter new report_title: Q1 2026 Binary Classification Review
Resuming with updated parameters...
```

**Programmatic HITL setup:**

```python
from src.agent.builder import create_analyst_agent

# Protect specific tools
agent = create_analyst_agent(hitl_tools=["generate_report", "suggest_next_experiments"])
```

See [Human-in-the-Loop documentation](hitl.md) for details.

---

## Governance Dashboard

The Streamlit dashboard visualizes agent execution traces stored as JSONL files.

```bash
streamlit run src/dashboard/app.py
```

Open `http://localhost:8501` in your browser.

**Page 1 — Run Explorer:**

- List of all agent runs with date, tool calls, errors, duration, tokens
- Click a run to see the full event timeline
- Color-coded rows: green for successes, red for errors
- Expandable event details with input/output summaries

**Page 2 — Tool Analytics:**

- Call frequency per tool (bar chart)
- Average latency per tool (bar chart)
- Detailed table with call count, avg/p95 latency, error count, error rate

**Note:** The dashboard reads from `data/logs/agent_traces/`. Run the demo first to generate trace data.

---

## Using Your Own MLflow Experiments

To analyze experiments from your own MLflow server:

1. Update `MLFLOW_TRACKING_URI` in `.env` to point to your server:

   ```
   MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
   ```

2. If your server uses S3 artifacts, update the S3 credentials:

   ```
   MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
   AWS_ACCESS_KEY_ID=your-key
   AWS_SECRET_ACCESS_KEY=your-secret
   ```

3. Run the agent with your experiment name:

   ```python
   result = invoke_with_governance(
       agent,
       "Analyze the my-custom-experiment experiment and generate a full report.",
       config,
   )
   ```

The agent works with any MLflow experiment that has logged metrics and parameters.

---

## Switching LLM Providers

### Ollama (default, local, free)

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

Make sure Ollama is running and the model is pulled:

```bash
ollama pull llama3.1:8b
ollama serve  # if not already running
```

**Compatible models** (must support tool calling):
- `llama3.1:8b` (recommended)
- `llama3.2:3b` (lighter, still supports tools)
- `llama3:latest` does **NOT** support tool calling

### Anthropic (cloud, paid)

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-5-20250929
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Anthropic models are faster and more capable but require an API key and incur costs.

---

## Enabling Web Search

The `web_search` tool uses the Tavily API to search the web for ML techniques and papers. It is optional and disabled by default.

To enable:

1. Get an API key at [app.tavily.com](https://app.tavily.com/)
2. Add it to your `.env`:

   ```
   TAVILY_API_KEY=tvly-your-key-here
   ```

3. Restart the agent. The tool is automatically registered when the key is present.

When enabled, the agent can search for external context about model architectures, hyperparameter tuning strategies, or ML best practices to enrich its analysis.
