"""Tool: suggest_next_experiments — generate concrete next-experiment configurations."""

from __future__ import annotations

from langchain_core.tools import tool

from src.analysis.patterns import correlate_params_metrics
from src.analysis.suggestions import AnalysisResult
from src.analysis.suggestions import suggest_next_experiments as _suggest
from src.mlflow_client.client import MLflowAnalystClient, MLflowClientError


@tool
def suggest_next_experiments(
    experiment_name: str,
    optimization_goal: str = "maximize val_accuracy",
    num_suggestions: int = 3,
) -> str:
    """Suggest concrete hyperparameter configurations for the next experiments.

    Based on the history of runs, this tool identifies the best-performing
    configurations and unexplored regions of the parameter space.
    Use after analyze_patterns to get actionable next steps.

    Args:
        experiment_name: Name or ID of the reference MLflow experiment.
        optimization_goal: Optimization objective in plain language,
            e.g. "maximize val_accuracy" or "minimize val_loss".
        num_suggestions: Number of suggestions to generate (default: 3, max: 5).
    """
    if num_suggestions > 5:
        num_suggestions = 5

    client = MLflowAnalystClient()

    try:
        exp = client.get_experiment(experiment_name)
    except MLflowClientError as exc:
        return f"ERROR: Could not find experiment '{experiment_name}': {exc}"

    try:
        run_infos = client.list_runs(exp.experiment_id, max_results=200)
    except MLflowClientError as exc:
        return f"ERROR: Could not list runs for '{experiment_name}': {exc}"

    if not run_infos:
        return (
            f"Experiment '{experiment_name}' has no runs. "
            "Run at least one experiment before requesting suggestions."
        )

    # Fetch full run details
    runs = []
    for info in run_infos:
        try:
            details = client.get_run_details(info.run_id)
            runs.append(details)
        except MLflowClientError:
            pass

    if not runs:
        return f"ERROR: Could not fetch run details for experiment '{experiment_name}'."

    # Parse target metric from optimization_goal
    goal_lower = optimization_goal.lower()
    any(t in goal_lower for t in {"minimize", "loss", "rmse", "mae", "mse", "error"})

    # Try to extract the metric name from the goal string
    target_metric = _extract_metric_from_goal(optimization_goal, runs)

    # Build correlation report
    correlation_report = None
    if len(runs) >= 3:
        correlation_report = correlate_params_metrics(runs, target_metric, min_runs=3)

    analysis = AnalysisResult(
        experiment_name=exp.name,
        experiment_id=exp.experiment_id,
        n_runs=len(runs),
        target_metric=target_metric,
        runs=runs,
        correlation_report=correlation_report,
    )

    suggestions = _suggest(analysis, num_suggestions=num_suggestions)

    if not suggestions:
        return (
            f"Could not generate suggestions for experiment '{experiment_name}'.\n"
            "This may happen if runs have no logged parameters or insufficient metric data.\n"
            f"Runs analyzed: {len(runs)}, target metric: `{target_metric}`"
        )

    lines = [
        f"# Next Experiment Suggestions: {experiment_name}",
        f"- **Optimization goal:** {optimization_goal}",
        f"- **Target metric:** `{target_metric}`",
        f"- **Based on:** {len(runs)} historical runs",
        f"- **Suggestions generated:** {len(suggestions)}",
        "",
    ]

    for i, s in enumerate(suggestions, 1):
        params_str = (
            ", ".join(f"`{k}={v}`" for k, v in s.params.items())
            if s.params
            else "_(see justification for guidance)_"
        )
        lines += [
            f"## Suggestion {i}: {s.title}",
            f"**Priority:** {s.priority}",
            "",
            f"**Proposed configuration:** {params_str}",
            "",
            f"**Justification:** {s.justification}",
            "",
            f"**Hypothesis:** {s.hypothesis}",
            "",
        ]

    lines += [
        "---",
        "_These suggestions are based on linear correlations and the best observed runs._",
        "_Validate hypotheses experimentally — ML optimization spaces are often non-linear._",
    ]

    return "\n".join(lines)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _extract_metric_from_goal(optimization_goal: str, runs: list) -> str:
    """Infer the target metric name from the optimization goal string.

    Falls back to the first available val_* metric, then the first metric overall.
    """
    # Collect all metrics across runs
    all_metrics: set[str] = set()
    for r in runs:
        all_metrics.update(r.metrics.keys())

    # Try to find a metric name in the goal string
    goal_lower = optimization_goal.lower()
    for metric in sorted(all_metrics, key=len, reverse=True):
        if metric.lower() in goal_lower:
            return metric

    # Fall back: prefer val_ metrics
    val_metrics = sorted(m for m in all_metrics if m.startswith("val_"))
    if val_metrics:
        return val_metrics[0]

    if all_metrics:
        return sorted(all_metrics)[0]

    return "val_accuracy"
