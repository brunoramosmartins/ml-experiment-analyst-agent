# ML Experiment Analyst Agent

[![CI](https://github.com/brunoramosmartins/ml-experiment-analyst-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/brunoramosmartins/ml-experiment-analyst-agent/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/brunoramosmartins/ml-experiment-analyst-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/brunoramosmartins/ml-experiment-analyst-agent)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](CHANGELOG.md)

> An autonomous agent that reads, compares, and interprets machine learning experiments — detecting patterns, identifying problems, and proposing next steps. Like a senior ML engineer who never sleeps.

---

## Demo

<!-- TODO: Record demo GIF with asciinema or screen capture tool.
     Run: python scripts/run_demo.py
     Show the agent analyzing an experiment and generating a report. -->

*Demo GIF coming soon — see [Quick Start](#quick-start) to try it yourself.*

---

## Highlights

- **Autonomous ML analysis** — Point the agent at any MLflow experiment and get a structured analysis report with zero manual work
- **Overfitting detection** — Automatically identifies train/val gaps with severity levels (LOW, MEDIUM, HIGH) across all runs
- **Hyperparameter pattern discovery** — Correlates numeric parameters with target metrics to find what drives performance
- **Next-experiment suggestions** — Generates concrete configurations with justifications and testable hypotheses
- **Built-in governance** — Token budgets, execution timeouts, failure thresholds, and full JSONL tracing of every agent action
- **Human-in-the-Loop** — Optionally pause the agent before critical tools for human approval, editing, or rejection

---

## Architecture

```
User Query (natural language)
    |
    v
+---------------------------------------------------+
|  ML Experiment Analyst Agent                      |
|  (deepagents / LangGraph)                         |
|                                                   |
|  Tools:                                           |
|  +-- load_experiment    --> MLflow Tracking Server |
|  +-- compare_runs       --> MLflow Tracking Server |
|  +-- diagnose_run       --> analysis/overfitting   |
|  +-- analyze_patterns   --> analysis/patterns      |
|  +-- suggest_next       --> analysis/suggestions   |
|  +-- generate_report    --> data/agent-workspace/  |
|  +-- web_search         --> Tavily API (optional)  |
+---------------------------------------------------+
    |                          |
    v                          v
+------------------+   +----------------------------+
| Governance Layer |   | Observability              |
| JSONL traces     |   | LangSmith (cloud tracing)  |
| Token budgets    |   +----------------------------+
| Failure limits   |
| Exec timeouts    |
+------------------+
    |
    v
+----------------------------+
| Streamlit Dashboard        |
| Run Explorer + Tool Stats  |
+----------------------------+
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/brunoramosmartins/ml-experiment-analyst-agent
cd ml-experiment-analyst-agent
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
pip install -e ".[dev]"

# 2. Install and start Ollama (https://ollama.com/)
ollama pull llama3.1:8b

# 3. Configure environment
cp .env.example .env
# Edit .env — LLM_PROVIDER=ollama works out of the box (no API key needed)
# Optionally fill in LANGCHAIN_API_KEY and TAVILY_API_KEY

# 4. Start MLflow (Docker required)
docker compose up -d

# 5. Seed demo experiments (3 synthetic experiments with 29 runs)
python scripts/seed_mlflow.py

# 6. Run the demo
python scripts/run_demo.py
```

Reports are saved to `data/agent-workspace/reports/`.

---

## Agent Tools

| Tool | What it does | When the agent uses it |
|---|---|---|
| `load_experiment` | Loads experiment metadata and all runs from MLflow | Start of every analysis |
| `compare_runs` | Side-by-side Markdown table of runs x metrics | Identify the best configuration |
| `diagnose_run` | Overfitting detection, metric gap analysis | Understand *why* a run underperformed |
| `analyze_patterns` | Correlation between hyperparameters and metrics | Find which parameters matter most |
| `suggest_next_experiments` | Concrete next-run configurations with justification | Plan the next iteration |
| `generate_report` | Full Markdown report saved to disk | Final deliverable |
| `web_search` | Search the web for ML techniques and papers (optional) | When external context is needed |

See [Tools Reference](docs/tools-reference.md) for full signatures, parameters, and error handling.

---

## Configuration

All settings are environment-driven via `.env` (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | LLM backend: `ollama` (local, free) or `anthropic` |
| `LLM_MODEL` | `llama3.1:8b` | Model name (must support tool calling) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `ANTHROPIC_API_KEY` | — | Required only when `LLM_PROVIDER=anthropic` |
| `LANGCHAIN_API_KEY` | — | LangSmith API key (optional, enables cloud tracing) |
| `LANGCHAIN_TRACING_V2` | `true` | Enable/disable LangSmith tracing |
| `TAVILY_API_KEY` | — | Tavily API key (optional, enables web search tool) |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server URL |
| `AGENT_WORKSPACE_PATH` | `data/agent-workspace` | Output directory for reports |
| `AGENT_MAX_TOKENS` | `50000` | Max tokens per agent invocation |
| `AGENT_TIMEOUT_SECONDS` | `300` | Execution timeout (seconds) |
| `AGENT_MAX_FAILURES` | `3` | Max consecutive tool failures before abort |
| `AGENT_TRACE_LOG_DIR` | `data/logs/agent_traces` | Directory for JSONL governance traces |

---

## Project Structure

```
src/
+-- agent/            # Agent builder, config, system prompt
+-- tools/            # 7 custom LangChain tools (6 core + web search)
+-- mlflow_client/    # MLflow access layer (client + models)
+-- analysis/         # Pure analysis functions (metrics, overfitting, patterns)
+-- report/           # Markdown report generator + templates
+-- observability/    # GovernanceCallbackHandler + LangSmith helpers
+-- dashboard/        # Streamlit governance dashboard

tests/
+-- unit/             # Unit tests (mocked MLflow)
+-- integration/      # E2E tests against real MLflow
+-- edge_cases/       # Edge case scenarios

scripts/
+-- seed_mlflow.py        # Populate MLflow with 3 synthetic experiments
+-- run_demo.py           # Run 3 demonstration queries with governance
+-- run_demo_hitl.py      # Human-in-the-loop demo

docs/                 # Architecture decisions, setup, tools reference, guides
```

---

## Documentation

| Document | Description |
|---|---|
| [Setup Guide](docs/SETUP.md) | Step-by-step installation and infrastructure setup |
| [Usage Guide](docs/usage-guide.md) | How to run the agent, customize queries, use HITL mode |
| [Tools Reference](docs/tools-reference.md) | Full API reference for all 7 tools |
| [Governance & Observability](docs/governance.md) | JSONL tracing, token budgets, execution limits |
| [Human-in-the-Loop](docs/hitl.md) | Pause agent before critical tools for human review |
| [Troubleshooting](docs/troubleshooting.md) | Common errors and solutions |
| [Architecture Decisions](docs/architecture-decisions.md) | ADRs: framework, LLM, backend, observability choices |
| [Demo Experiments](docs/demo-experiments.md) | Description of the 3 synthetic MLflow experiments |
| [Prompt Engineering Log](docs/prompt-engineering-log.md) | System prompt iteration history |
| [Contributing](CONTRIBUTING.md) | Branch naming, commit conventions, PR workflow |
| [Changelog](CHANGELOG.md) | Version history following Keep a Changelog |

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | [deepagents](https://pypi.org/project/deepagents/) (LangGraph) |
| LLM (local) | [Ollama](https://ollama.com/) + Llama 3.1 |
| LLM (cloud) | [Anthropic](https://www.anthropic.com/) Claude (swappable) |
| Experiment tracking | [MLflow](https://mlflow.org/) |
| Cloud observability | [LangSmith](https://smith.langchain.com/) |
| Web search | [Tavily](https://tavily.com/) |
| Dashboard | [Streamlit](https://streamlit.io/) |
| CI/CD | GitHub Actions (lint, typecheck, test with 75% coverage) |

---

## License

[MIT](LICENSE) - Bruno Ramos Martins
