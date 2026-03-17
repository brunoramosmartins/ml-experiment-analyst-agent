"""Tool: compare_runs — side-by-side comparison of multiple MLflow runs."""

from __future__ import annotations

from langchain_core.tools import tool

from src.mlflow_client.client import MLflowAnalystClient, MLflowClientError


@tool
def compare_runs(
    run_ids: list[str],
    metrics_to_compare: list[str] | None = None,
    include_params: bool = True,
) -> str:
    """Compare multiple MLflow runs side by side.

    Use this tool after load_experiment to identify which configuration produced
    the best results and understand the impact of hyperparameter changes.
    Returns a Markdown table with runs as rows and metrics/params as columns.

    Args:
        run_ids: List of MLflow run IDs to compare (obtain from load_experiment).
        metrics_to_compare: Which metrics to include. None means all available metrics.
        include_params: Whether to include hyperparameters in the comparison table.
    """
    if not run_ids:
        return "ERROR: No run IDs provided. Obtain run IDs from load_experiment first."

    if len(run_ids) > 20:
        return (
            f"ERROR: Too many run IDs ({len(run_ids)}). "
            "Limit to 20 runs per comparison for readability."
        )

    client = MLflowAnalystClient()

    # Fetch details for each run
    runs_data = []
    errors = []
    for run_id in run_ids:
        try:
            details = client.get_run_details(run_id)
            runs_data.append(details)
        except MLflowClientError as exc:
            errors.append(f"- `{run_id[:8]}...`: {exc}")

    if not runs_data:
        error_block = "\n".join(errors)
        return f"ERROR: Could not fetch any of the requested runs:\n{error_block}"

    # Collect all metrics and params across runs
    all_metrics: set[str] = set()
    all_params: set[str] = set()
    for r in runs_data:
        all_metrics.update(r.metrics.keys())
        if include_params:
            all_params.update(r.params.keys())

    # Filter metrics if requested
    if metrics_to_compare:
        display_metrics = [m for m in metrics_to_compare if m in all_metrics]
        missing = [m for m in metrics_to_compare if m not in all_metrics]
        if missing:
            errors.append(
                f"Warning: metrics not found in any run: {', '.join(missing)}. "
                f"Available: {', '.join(sorted(all_metrics))}"
            )
    else:
        display_metrics = sorted(all_metrics)

    display_params = sorted(all_params)
    columns = display_metrics + (display_params if include_params else [])

    if not columns:
        return "No metrics or parameters found in the requested runs."

    # Build table header
    header = "| run_id | run_name | " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * (len(columns) + 2)) + "|"
    rows = [header, separator]

    # Find best value per metric (for highlighting)
    best_per_metric: dict[str, float] = {}
    for m in display_metrics:
        values = [r.metrics[m] for r in runs_data if m in r.metrics]
        if values:
            lower_better = any(t in m for t in {"loss", "rmse", "mae", "mse", "error"})
            best_per_metric[m] = min(values) if lower_better else max(values)

    for r in runs_data:
        short_id = r.run_id[:8]
        cells = [f"`{short_id}...`", r.run_name or "—"]
        for col in columns:
            if col in r.metrics:
                val = r.metrics[col]
                formatted = f"{val:.4f}"
                # Bold the best value per metric
                if col in best_per_metric and val == best_per_metric[col]:
                    formatted = f"**{formatted}**"
                cells.append(formatted)
            elif col in r.params:
                cells.append(r.params[col])
            else:
                cells.append("—")
        rows.append("| " + " | ".join(cells) + " |")

    # Summary: which run won how many metrics
    winner_counts: dict[str, int] = {}
    for m in display_metrics:
        values = {r.run_id: r.metrics[m] for r in runs_data if m in r.metrics}
        if not values:
            continue
        lower_better = any(t in m for t in {"loss", "rmse", "mae", "mse", "error"})
        best_run_id = (
            min(values, key=values.__getitem__)
            if lower_better
            else max(values, key=values.__getitem__)
        )
        winner_counts[best_run_id] = winner_counts.get(best_run_id, 0) + 1

    lines = ["\n".join(rows), ""]
    lines.append("### Summary")
    if winner_counts:
        lines.append("Best run by metric count:")
        for run_id, count in sorted(winner_counts.items(), key=lambda x: -x[1]):
            run = next((r for r in runs_data if r.run_id == run_id), None)
            name = run.run_name if run and run.run_name else run_id[:8] + "..."
            lines.append(f"- `{name}` — best in {count}/{len(display_metrics)} metric(s)")

    if errors:
        lines.append("\n### Warnings")
        lines.extend(errors)

    return "\n".join(lines)
