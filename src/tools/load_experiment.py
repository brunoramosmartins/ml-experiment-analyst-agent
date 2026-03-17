"""Tool: load_experiment — entry point for any MLflow analysis session."""

from __future__ import annotations

from langchain_core.tools import tool

from src.mlflow_client.client import MLflowAnalystClient, MLflowClientError


@tool
def load_experiment(
    experiment_name: str,
    max_runs: int = 50,
    filter_string: str = "",
    order_by: str = "",
) -> str:
    """Load an MLflow experiment and return its runs with metrics and parameters.

    Use this tool when the user mentions a specific experiment by name or ID.
    Always call this first before comparing runs or diagnosing problems.
    Returns a structured text summary you can use to decide the next analysis step.

    Args:
        experiment_name: Name or numeric ID of the experiment in MLflow.
        max_runs: Maximum number of runs to return (default: 50).
        filter_string: Optional MLflow filter expression, e.g. "metrics.accuracy > 0.8".
        order_by: Optional ordering expression, e.g. "metrics.val_loss ASC".
    """
    client = MLflowAnalystClient()

    try:
        exp = client.get_experiment(experiment_name)
    except MLflowClientError as exc:
        return f"ERROR loading experiment: {exc}"

    try:
        runs = client.list_runs(
            experiment_id=exp.experiment_id,
            filter_string=filter_string,
            order_by=order_by or None,
            max_results=max_runs,
        )
    except MLflowClientError as exc:
        return (
            f"Experiment '{exp.name}' found (ID: {exp.experiment_id}), "
            f"but could not list runs: {exc}"
        )

    if not runs:
        return (
            f"Experiment '{exp.name}' (ID: {exp.experiment_id}) exists but has no runs.\n"
            "Hint: run your training script and log runs with mlflow.start_run()."
        )

    # Collect available metrics by sampling first run details
    available_metrics: set[str] = set()
    for run_info in runs[:5]:
        try:
            details = client.get_run_details(run_info.run_id)
            available_metrics.update(details.metrics.keys())
        except MLflowClientError:
            pass

    # Build date range
    start_times = [r.start_time for r in runs if r.start_time]
    date_range = (
        f"{min(start_times).strftime('%Y-%m-%d')} → {max(start_times).strftime('%Y-%m-%d')}"
        if start_times
        else "unknown"
    )

    status_counts: dict[str, int] = {}
    for r in runs:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    lines = [
        f"# Experiment: {exp.name}",
        f"- **ID:** {exp.experiment_id}",
        f"- **Lifecycle stage:** {exp.lifecycle_stage}",
        f"- **Total runs returned:** {len(runs)} (max_runs={max_runs})",
        f"- **Date range:** {date_range}",
        f"- **Run statuses:** {', '.join(f'{k}: {v}' for k, v in status_counts.items())}",
        f"- **Available metrics:** {', '.join(sorted(available_metrics)) or '(none sampled)'}",
        "",
        "## Runs",
        "| run_id (short) | run_name | status | start_time |",
        "|---|---|---|---|",
    ]

    for r in runs:
        short_id = r.run_id[:8]
        start = r.start_time.strftime("%Y-%m-%d %H:%M") if r.start_time else "—"
        lines.append(f"| `{short_id}...` | {r.run_name or '—'} | {r.status} | {start} |")

    lines += [
        "",
        "_Use `compare_runs` with specific run IDs to see metric values side by side._",
        "_Use `diagnose_run` with a single run ID to detect overfitting._",
        "_Full run IDs for reference:_",
    ]
    for r in runs[:10]:
        lines.append(f"- `{r.run_id}` — {r.run_name or '(unnamed)'}")
    if len(runs) > 10:
        lines.append(f"- _(and {len(runs) - 10} more)_")

    return "\n".join(lines)
