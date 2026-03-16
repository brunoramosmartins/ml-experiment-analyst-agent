# ML Experiment Analyst Agent

[![CI](https://github.com/brunoramosmartins/ml-experiment-analyst-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/brunoramosmartins/ml-experiment-analyst-agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An autonomous agent that reads, compares, and interprets machine learning experiments — detecting patterns, identifying problems, and proposing next steps. Like a senior ML engineer who never sleeps.

---

## What is this?

Any mature ML pipeline accumulates dozens or hundreds of runs in MLflow. Manually navigating those runs to extract insights is slow, prone to confirmation bias, and does not scale.

This agent takes a natural language query, connects to a MLflow Tracking Server, and autonomously produces a structured analysis report — covering metric comparisons, overfitting diagnostics, hyperparameter pattern analysis, and next-experiment suggestions.

**Stack:** Python · [deepagents](https://pypi.org/project/deepagents/) (LangGraph) · MLflow · LangSmith · Anthropic Claude

---

## Demo

> *Demo GIF coming in Phase 6 (Portfolio Polish)*

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/brunoramosmartins/ml-experiment-analyst-agent
cd ml-experiment-analyst-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, LANGCHAIN_API_KEY, TAVILY_API_KEY

# 3. Start MLflow (Docker required)
make mlflow-up

# 4. Seed demo experiments
make seed-mlflow

# 5. Run the demo
make run-demo
```

---

## Architecture

```
User Query
    │
    ▼
ML Experiment Analyst Agent (deepagents / LangGraph)
    │
    ├── load_experiment   ──► MLflow Tracking Server
    ├── compare_runs      ──► MLflow Tracking Server
    ├── diagnose_run      ──► src/analysis/overfitting.py
    ├── analyze_patterns  ──► src/analysis/patterns.py
    ├── suggest_next      ──► src/analysis/suggestions.py
    └── generate_report   ──► data/agent-workspace/reports/
    │
    ├── GovernanceMiddleware ──► data/logs/agent_traces/ (JSONL)
    └── LangSmith            ──► Full execution traces
```

---

## Agent Tools

| Tool | What it does | When to use |
|---|---|---|
| `load_experiment` | Loads experiment metadata and all runs from MLflow | Start of every analysis |
| `compare_runs` | Side-by-side Markdown table of runs × metrics | Identify the best configuration |
| `diagnose_run` | Overfitting detection, metric gap analysis | Understand *why* a run performed as it did |
| `analyze_patterns` | Correlation between hyperparameters and metrics | Find which parameters matter most |
| `suggest_next_experiments` | Concrete next-run configurations with justification | Plan the next iteration |
| `generate_report` | Full Markdown report saved to disk | Final deliverable |

---

## Project Structure

```
src/
├── agent/          # Agent builder, config, system prompt
├── tools/          # 6 custom LangChain tools
├── mlflow_client/  # MLflow access layer
├── analysis/       # Pure analysis functions (metrics, overfitting, patterns)
├── report/         # Markdown report generator
├── observability/  # GovernanceMiddleware + LangSmith helpers
└── dashboard/      # Streamlit governance dashboard

tests/
├── unit/           # Unit tests (mocked MLflow)
├── integration/    # E2E tests against real MLflow
└── edge_cases/     # Edge case scenarios

scripts/
├── seed_mlflow.py  # Populate MLflow with 3 synthetic experiments
└── run_demo.py     # Run 3 demonstration queries
```

---

## References

- [Architecture Decision Records](docs/architecture-decisions.md)
- [Setup Guide](docs/SETUP.md)
- [Contributing](CONTRIBUTING.md)
- [deepagents](https://pypi.org/project/deepagents/) · [LangSmith](https://smith.langchain.com/) · [MLflow](https://mlflow.org/)
