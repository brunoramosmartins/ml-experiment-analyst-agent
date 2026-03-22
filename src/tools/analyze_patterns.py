"""Tool: analyze_patterns — hyperparameter × metric correlation analysis."""

from __future__ import annotations

from langchain_core.tools import tool

from src.analysis.patterns import correlate_params_metrics
from src.mlflow_client.client import MLflowAnalystClient, MLflowClientError


@tool
def analyze_patterns(
    experiment_name: str,
    target_metric: str,
    min_runs: int = 5,
) -> str:
    """Analyze correlations between hyperparameters and a target metric.

    Use this tool to identify WHICH parameters have the strongest impact on
    performance and in which direction. Requires at least min_runs runs with
    the target metric logged.

    Args:
        experiment_name: Name or ID of the MLflow experiment to analyze.
        target_metric: Metric to correlate against (e.g. "val_accuracy", "val_loss").
        min_runs: Minimum number of runs required for a meaningful analysis (default: 5).
    """
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
            "Cannot perform pattern analysis on an empty experiment."
        )

    # Fetch full details for all runs
    runs = []
    fetch_errors = 0
    for info in run_infos:
        try:
            details = client.get_run_details(info.run_id)
            runs.append(details)
        except MLflowClientError:
            fetch_errors += 1

    if not runs:
        return (
            f"ERROR: Could not fetch run details for any of the {len(run_infos)} runs "
            f"in '{experiment_name}'."
        )

    report = correlate_params_metrics(runs, target_metric, min_runs=min_runs)

    if not report.correlations and report.message:
        return (
            f"Pattern analysis for experiment '{experiment_name}' "
            f"(target: `{target_metric}`):\n\n"
            f"ℹ️ {report.message}\n\n"
            f"Runs fetched: {len(runs)}" + (f" ({fetch_errors} failed)" if fetch_errors else "")
        )

    lines = [
        f"# Hyperparameter Pattern Analysis: {experiment_name}",
        f"- **Target metric:** `{target_metric}`",
        f"- **Runs analyzed:** {report.n_runs}",
        f"- **Parameters evaluated:** {len(report.correlations)}",
        f"- **Note:** {report.message}",
        "",
        "## Parameter Impact Ranking",
        "_(sorted by absolute Pearson correlation with the target metric)_",
        "",
        "| Rank | Parameter | Correlation (r) | Direction | Interpretation |",
        "|---|---|---|---|---|",
    ]

    if not report.top_params:
        lines.append("| — | No numeric parameters found | — | — | — |")
    else:
        for i, p in enumerate(report.top_params, 1):
            if p.direction == "positive":
                interpretation = f"Higher `{p.param}` → better `{target_metric}`"
            elif p.direction == "negative":
                interpretation = f"Lower `{p.param}` → better `{target_metric}`"
            else:
                interpretation = f"Weak relationship with `{target_metric}`"

            strength = (
                "strong"
                if abs(p.correlation) >= 0.7
                else "moderate"
                if abs(p.correlation) >= 0.4
                else "weak"
            )
            lines.append(
                f"| {i} | `{p.param}` | {p.correlation:+.3f} | "
                f"{p.direction} ({strength}) | {interpretation} |"
            )

    # Key insights
    positives = [
        p for p in report.top_params if p.direction == "positive" and abs(p.correlation) >= 0.4
    ]
    negatives = [
        p for p in report.top_params if p.direction == "negative" and abs(p.correlation) >= 0.4
    ]
    neutrals = [p for p in report.top_params if p.direction == "neutral"]

    lines += ["", "## Key Insights"]

    if positives:
        lines.append(
            f"**Positive drivers** (increase to improve `{target_metric}`): "
            + ", ".join(f"`{p.param}` (r={p.correlation:+.2f})" for p in positives[:3])
        )
    if negatives:
        lines.append(
            f"**Negative drivers** (decrease to improve `{target_metric}`): "
            + ", ".join(f"`{p.param}` (r={p.correlation:+.2f})" for p in negatives[:3])
        )
    if neutrals:
        lines.append("**No clear effect:** " + ", ".join(f"`{p.param}`" for p in neutrals[:5]))
    if not positives and not negatives:
        lines.append(
            "No strong correlations found. Consider: more runs with varied hyperparameters, "
            "non-linear relationships (try RandomForest feature importances), "
            "or the experiment may need more hyperparameter diversity."
        )

    if fetch_errors:
        lines.append(f"\n_Note: {fetch_errors} run(s) could not be fetched and were skipped._")

    return "\n".join(lines)
