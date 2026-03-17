"""Tool: generate_report — produce and save a full Markdown analysis report."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from src.analysis.overfitting import detect_overfitting_trend
from src.analysis.patterns import correlate_params_metrics
from src.analysis.suggestions import AnalysisResult, suggest_next_experiments
from src.mlflow_client.client import MLflowAnalystClient, MLflowClientError
from src.report.generator import generate_markdown_report

_DEFAULT_REPORTS_DIR = Path(
    os.getenv("AGENT_WORKSPACE_PATH", "data/agent-workspace")
) / "reports"


@tool
def generate_report(
    experiment_name: str,
    report_title: str,
    include_suggestions: bool = True,
    output_path: str | None = None,
) -> str:
    """Generate and save a complete Markdown analysis report for an experiment.

    This is the FINAL tool in the analysis workflow. Call it after you have
    collected metrics, diagnosed runs, and identified patterns. The report is
    saved as a .md file and includes: metrics comparison, overfitting diagnostics,
    hyperparameter patterns, and (optionally) next-experiment recommendations.

    Args:
        experiment_name: Name or ID of the MLflow experiment to report on.
        report_title: Human-readable title for the report.
        include_suggestions: Whether to include next-experiment suggestions (default: True).
        output_path: File path to save the report. Defaults to
            data/agent-workspace/reports/{timestamp}_{experiment}.md
    """
    client = MLflowAnalystClient()

    # ── Fetch experiment and runs ─────────────────────────────────────────────
    try:
        exp = client.get_experiment(experiment_name)
    except MLflowClientError as exc:
        return f"ERROR: Could not find experiment '{experiment_name}': {exc}"

    try:
        run_infos = client.list_runs(exp.experiment_id, max_results=100)
    except MLflowClientError as exc:
        return f"ERROR: Could not list runs for '{experiment_name}': {exc}"

    if not run_infos:
        return (
            f"Experiment '{experiment_name}' has no runs. "
            "Cannot generate a report without data."
        )

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
            f"ERROR: Could not fetch run details for experiment '{experiment_name}'. "
            "Check MLflow connectivity."
        )

    # ── Determine target metric ───────────────────────────────────────────────
    all_metrics: set[str] = set()
    for r in runs:
        all_metrics.update(r.metrics.keys())

    val_metrics = sorted(m for m in all_metrics if m.startswith("val_"))
    if val_metrics:
        target_metric = val_metrics[0]
    elif all_metrics:
        target_metric = sorted(all_metrics)[0]
    else:
        target_metric = "val_accuracy"

    # ── Overfitting analysis ──────────────────────────────────────────────────
    overfitting_reports = detect_overfitting_trend(runs)

    # ── Pattern analysis ──────────────────────────────────────────────────────
    correlation_report = None
    if len(runs) >= 3:
        correlation_report = correlate_params_metrics(runs, target_metric, min_runs=3)

    # ── Suggestions ───────────────────────────────────────────────────────────
    suggestions = []
    if include_suggestions:
        analysis_for_suggestions = AnalysisResult(
            experiment_name=exp.name,
            experiment_id=exp.experiment_id,
            n_runs=len(runs),
            target_metric=target_metric,
            runs=runs,
            correlation_report=correlation_report,
        )
        suggestions = suggest_next_experiments(analysis_for_suggestions)

    # ── Assemble AnalysisResult ───────────────────────────────────────────────
    analysis = AnalysisResult(
        experiment_name=exp.name,
        experiment_id=exp.experiment_id,
        n_runs=len(runs),
        target_metric=target_metric,
        runs=runs,
        correlation_report=correlation_report,
        suggestions=suggestions,
    )

    # ── Generate Markdown ─────────────────────────────────────────────────────
    report_content = generate_markdown_report(analysis, overfitting_reports)

    # Prepend the custom title if different from experiment name
    if report_title and report_title.strip() != exp.name:
        title_block = f"# {report_title}\n\n_{exp.name}_\n\n---\n\n"
        report_content = title_block + report_content

    # ── Determine output path ─────────────────────────────────────────────────
    if output_path:
        save_path = Path(output_path)
    else:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in exp.name)
        save_path = _DEFAULT_REPORTS_DIR / f"{timestamp}_{safe_name}.md"

    # ── Save report ───────────────────────────────────────────────────────────
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(report_content, encoding="utf-8")
    except OSError as exc:
        return (
            f"ERROR: Could not save report to '{save_path}': {exc}\n"
            "Check that the output directory exists and is writable."
        )

    # ── Return summary ────────────────────────────────────────────────────────
    preview_lines = report_content.splitlines()[:12]
    preview = "\n".join(preview_lines)

    warnings = []
    if fetch_errors:
        warnings.append(f"{fetch_errors} run(s) could not be fetched and were skipped.")

    output_lines = [
        f"✅ Report saved: `{save_path}`",
        f"- **Experiment:** {exp.name}",
        f"- **Runs included:** {len(runs)}",
        "- **Overfitting findings:** "
        + str(sum(1 for r in overfitting_reports if r.is_overfit))
        + " run(s) with overfitting detected",
        f"- **Suggestions included:** {'Yes' if suggestions else 'No'}",
        f"- **File size:** {len(report_content)} characters",
        "",
        "## Report Preview (first 12 lines)",
        "```markdown",
        preview,
        "```",
    ]

    if warnings:
        output_lines += ["", "## Warnings"] + [f"- {w}" for w in warnings]

    return "\n".join(output_lines)
