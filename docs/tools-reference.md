# Tools Reference

This document describes all 6 custom LangChain tools available to the ML Experiment Analyst Agent.
Each tool is a `@tool`-decorated function in `src/tools/` and is registered in `src/tools/__init__.py`.

---

## Table of Contents

| Tool | Purpose | Typical position in workflow |
|---|---|---|
| [`load_experiment`](#load_experiment) | Load experiment metadata + runs from MLflow | Step 1 |
| [`compare_runs`](#compare_runs) | Side-by-side metric comparison across runs | Step 2 |
| [`diagnose_run`](#diagnose_run) | Detect overfitting and other issues in one run | Step 3 |
| [`analyze_patterns`](#analyze_patterns) | Correlate hyperparameters with a target metric | Step 4 |
| [`suggest_next_experiments`](#suggest_next_experiments) | Generate next-experiment configurations | Step 5 |
| [`generate_report`](#generate_report) | Produce and save a full Markdown analysis report | Step 6 (final) |

---

## `load_experiment`

**File:** `src/tools/load_experiment.py`

**Purpose:** Entry point for any analysis session. Loads an MLflow experiment by name or ID and returns a structured text summary of its runs — including available metrics, date range, run statuses, and a table of run IDs.

### Signature

```python
load_experiment(
    experiment_name: str,
    max_runs: int = 50,
    filter_string: str = "",
    order_by: str = "",
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment_name` | `str` | required | Name or numeric ID of the experiment in MLflow |
| `max_runs` | `int` | `50` | Maximum number of runs to return |
| `filter_string` | `str` | `""` | MLflow filter expression, e.g. `"metrics.accuracy > 0.8"` |
| `order_by` | `str` | `""` | Sort expression, e.g. `"metrics.val_loss ASC"` |

### Returns

A formatted string containing:
- Experiment name, ID, lifecycle stage
- Total runs, date range, run statuses
- Available metrics (sampled from first 5 runs)
- Table of runs (run_id, run_name, status, start_time)
- Full run IDs for the first 10 runs

### Error cases

| Condition | Return message |
|---|---|
| Experiment not found | `ERROR loading experiment: Experiment 'X' not found...` |
| MLflow offline | `ERROR loading experiment: Could not connect to MLflow...` |
| No runs in experiment | Informational message with hint to log runs |

### Example agent prompt

```
"Analise o experimento binary-classification"
→ Agent calls: load_experiment(experiment_name="binary-classification")
```

---

## `compare_runs`

**File:** `src/tools/compare_runs.py`

**Purpose:** Compares multiple MLflow runs side by side. Generates a Markdown table with runs as rows and metrics/parameters as columns. Automatically highlights the best value per metric (bold).

### Signature

```python
compare_runs(
    run_ids: list[str],
    metrics_to_compare: list[str] | None = None,
    include_params: bool = True,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `run_ids` | `list[str]` | required | List of full MLflow run IDs to compare (max 20) |
| `metrics_to_compare` | `list[str] \| None` | `None` | Metrics to include; `None` = all available |
| `include_params` | `bool` | `True` | Whether to include hyperparameters in the table |

### Returns

A formatted string containing:
- Markdown table: run_id × (metrics + params)
- Best value per metric **bolded**
- Summary: which run won the most metrics

### Error cases

| Condition | Return message |
|---|---|
| Empty run_ids | `ERROR: No run IDs provided.` |
| > 20 run IDs | `ERROR: Too many run IDs.` |
| Run not found | Error listed in warnings section |

### Example agent prompt

```
"Compare os 3 melhores runs do experimento"
→ Agent calls: compare_runs(run_ids=["abc123", "def456", "ghi789"])
```

---

## `diagnose_run`

**File:** `src/tools/diagnose_run.py`

**Purpose:** Diagnoses a single MLflow run for common problems. Checks run status, missing validation metrics, overfitting (train/val gap), and suspicious metric values. Returns findings with severity icons.

### Signature

```python
diagnose_run(
    run_id: str,
    overfitting_threshold: float = 0.05,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `run_id` | `str` | required | Full MLflow run ID to diagnose |
| `overfitting_threshold` | `float` | `0.05` | Minimum train/val gap to flag as overfitting |

### Returns

A formatted string containing:
- Run metadata (status, all metrics, parameters)
- Numbered diagnostic findings, each with:
  - Severity icon: `✅ INFO`, `⚠️ WARNING`, `🔴 CRITICAL`
  - Metric gap details (train vs val, per metric)
  - Suggested action

### Severity levels

| Severity | Condition |
|---|---|
| `NONE` | Gap < threshold (default 0.05) |
| `LOW` | 0.05 ≤ gap < 0.10 |
| `MEDIUM` | 0.10 ≤ gap < 0.20 |
| `HIGH` | gap ≥ 0.20 |

### Error cases

| Condition | Return message |
|---|---|
| Run not found | `ERROR fetching run: ...` |

### Example agent prompt

```
"Por que o run abc123 teve performance baixa?"
→ Agent calls: diagnose_run(run_id="abc123...")
```

---

## `analyze_patterns`

**File:** `src/tools/analyze_patterns.py`

**Purpose:** Computes Pearson correlation between each numeric hyperparameter and a target metric across all runs of an experiment. Returns a ranked table of parameter impact with direction (positive/negative) and interpretation.

### Signature

```python
analyze_patterns(
    experiment_name: str,
    target_metric: str,
    min_runs: int = 5,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment_name` | `str` | required | Name or ID of the MLflow experiment |
| `target_metric` | `str` | required | Metric to correlate against (e.g. `"val_accuracy"`) |
| `min_runs` | `int` | `5` | Minimum runs required for meaningful analysis |

### Returns

A formatted string containing:
- Ranked table: parameter × (correlation, direction, strength, interpretation)
- Key insights: positive drivers, negative drivers, neutral parameters

### Limitations

- Only numeric parameters are analyzed (string params like `model_type` are skipped)
- Pearson measures linear relationships only
- Requires at least `min_runs` runs with the target metric logged

### Error cases

| Condition | Return message |
|---|---|
| Experiment not found | `ERROR: Could not find experiment...` |
| < min_runs | Informational message without crashing |
| No numeric params | `No numeric parameters found...` |

### Example agent prompt

```
"Quais hiperparâmetros mais impactam val_accuracy?"
→ Agent calls: analyze_patterns(experiment_name="binary-classification", target_metric="val_accuracy")
```

---

## `suggest_next_experiments`

**File:** `src/tools/suggest_next_experiments.py`

**Purpose:** Generates concrete hyperparameter configurations for the next experiments, based on the best-performing runs and parameter correlations. Each suggestion includes a proposed configuration, justification, and testable hypothesis.

### Signature

```python
suggest_next_experiments(
    experiment_name: str,
    optimization_goal: str = "maximize val_accuracy",
    num_suggestions: int = 3,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment_name` | `str` | required | Reference MLflow experiment |
| `optimization_goal` | `str` | `"maximize val_accuracy"` | Optimization objective in plain language |
| `num_suggestions` | `int` | `3` | Number of suggestions (capped at 5) |

### Returns

Numbered suggestions (up to `num_suggestions`), each with:
- Title describing the strategy
- Proposed hyperparameter values
- Justification based on historical data
- Testable hypothesis

### Suggestion strategies

1. **Refine best configuration** — based on the run with the best target metric
2. **Increase a positively-correlated parameter** — based on correlation analysis
3. **Decrease a negatively-correlated parameter** — based on correlation analysis

### Error cases

| Condition | Return message |
|---|---|
| Experiment not found | `ERROR: Could not find experiment...` |
| No runs | Informational message |

### Example agent prompt

```
"Quais experimentos devo rodar a seguir para melhorar val_loss?"
→ Agent calls: suggest_next_experiments(experiment_name="regression-v2", optimization_goal="minimize val_loss")
```

---

## `generate_report`

**File:** `src/tools/generate_report.py`

**Purpose:** Final tool in the analysis workflow. Generates a complete Markdown analysis report from scratch by running all analyses (metrics comparison, overfitting detection, pattern correlation, suggestions). Saves the report to disk and returns the path + preview.

### Signature

```python
generate_report(
    experiment_name: str,
    report_title: str,
    include_suggestions: bool = True,
    output_path: str | None = None,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment_name` | `str` | required | MLflow experiment to report on |
| `report_title` | `str` | required | Human-readable title for the report |
| `include_suggestions` | `bool` | `True` | Whether to include next-experiment suggestions |
| `output_path` | `str \| None` | `None` | Output path; defaults to `data/agent-workspace/reports/{timestamp}_{experiment}.md` |

### Returns

A confirmation string containing:
- ✅ Report saved path
- Summary stats (runs included, overfitting findings, suggestions)
- File size
- Preview of first 12 lines of the report

### Report sections

The generated Markdown file includes:

| Section | Content |
|---|---|
| Header | Experiment name, ID, runs analyzed, target metric, generation timestamp |
| Metrics Comparison | Table of all runs ranked by target metric |
| Diagnostics | Overfitting findings per run with severity |
| Hyperparameter Patterns | Correlation table (top 10 parameters) |
| Recommendations | Next-experiment suggestions with justifications |
| Limitations & Disclaimer | AI-generated report caveat |

### Error cases

| Condition | Return message |
|---|---|
| Experiment not found | `ERROR: Could not find experiment...` |
| No runs | `...no runs. Cannot generate a report...` |
| Cannot write file | `ERROR: Could not save report to '...'` |

### Example agent prompt

```
"Gera um relatório completo do experimento binary-classification"
→ Agent calls: generate_report(experiment_name="binary-classification", report_title="Binary Classification Analysis - March 2026")
```

---

## Using tools from Python

All tools are importable from `src.tools`:

```python
from src.tools import (
    load_experiment,
    compare_runs,
    diagnose_run,
    analyze_patterns,
    suggest_next_experiments,
    generate_report,
    ALL_TOOLS,  # list of all 6 tools, for passing to create_deep_agent
)
```

Tools can be invoked directly (useful for testing):

```python
result = load_experiment.invoke({"experiment_name": "binary-classification"})
print(result)
```

Or they can be registered with the agent (done automatically in `src/agent/builder.py`):

```python
from src.agent.builder import create_analyst_agent
agent = create_analyst_agent()
# Agent now has all 6 tools available
```
