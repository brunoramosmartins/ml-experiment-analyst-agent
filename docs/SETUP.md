# Setup Guide

## System Requirements

- Python 3.11+
- Docker + Docker Compose (for MLflow stack)
- Git

## Step-by-step

### 1. Clone the repository

```bash
git clone https://github.com/brunoramosmartins/ml-experiment-analyst-agent
cd ml-experiment-analyst-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

Validate the install:

```bash
python -c "from deepagents import create_deep_agent; print('deepagents OK')"
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in the required values:

| Variable | Where to get it | Required |
|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) | ✅ Yes |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com/) | ✅ Yes |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com/) | ✅ Yes |
| `MLFLOW_TRACKING_URI` | Leave as `http://localhost:5000` | ✅ Yes |
| `AWS_ACCESS_KEY_ID` | Leave as `minioadmin` (Docker default) | ✅ Yes |
| `AWS_SECRET_ACCESS_KEY` | Leave as `minioadmin` (Docker default) | ✅ Yes |

### 5. Start the MLflow infrastructure

```bash
make mlflow-up
```

This starts three Docker containers:
- **mlflow** — tracking server at `http://localhost:5000`
- **postgres** — metadata backend
- **minio** — artifact store, console at `http://localhost:9001`

Wait ~15 seconds, then verify:

```bash
curl http://localhost:5000/health
# Expected: {"status": "OK"}
```

### 6. Seed demo experiments

```bash
make seed-mlflow
```

This creates 3 experiments in MLflow:

| Experiment | Description |
|---|---|
| `binary-classification` | 17 runs, 4 model types, hyperparameter sweep |
| `regression-v2` | 7 runs, progressive feature engineering |
| `overfit-test` | 5 runs, intentional overfitting gradient |

Open `http://localhost:5000` to confirm.

### 7. Run the demo

```bash
make run-demo
```

> `scripts/run_demo.py` will be available after Phase 3.

---

## Common commands

```bash
make install       # Install all dependencies
make test          # Run unit tests with coverage
make lint          # Check code style (ruff)
make typecheck     # Type check (mypy)
make mlflow-up     # Start Docker stack
make mlflow-down   # Stop Docker stack
make seed-mlflow   # Populate MLflow with demo data
make run-demo      # Run agent demo (Phase 3+)
make run-dashboard # Start Streamlit dashboard (Phase 4+)
```

---

## Troubleshooting

**`Could not connect to MLflow`**
→ Make sure Docker is running and `make mlflow-up` completed without errors.

**`from deepagents import create_deep_agent` fails**
→ Run `pip install -e ".[dev]"` inside the virtual environment.

**`ANTHROPIC_API_KEY not set`**
→ Check that your `.env` file exists and the key is filled in correctly.

**MinIO bucket not found**
→ The `minio-init` container may have failed. Run `docker compose logs minio-init` to inspect.
