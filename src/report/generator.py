"""Markdown report generator.

Transforms an AnalysisResult into a structured Markdown document
that the agent saves via FilesystemBackend.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.analysis.overfitting import OverfittingReport, OverfitSeverity
from src.analysis.suggestions import AnalysisResult


_SEVERITY_ICON = {
    OverfitSeverity.NONE: "✅",
    OverfitSeverity.LOW: "⚠️",
    OverfitSeverity.MEDIUM: "⚠️",
    OverfitSeverity.HIGH: "🔴",
}


def generate_markdown_report(
    analysis: AnalysisResult,
    overfitting_reports: list[OverfittingReport] | None = None,
) -> str:
    """Generate a structured Markdown report from an AnalysisResult.

    Args:
        analysis: Aggregated analysis result.
        overfitting_reports: Optional list of per-run overfitting reports.

    Returns:
        Full Markdown string ready to be written to disk.
    """
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    sections.append(f"# Analysis Report: {analysis.experiment_name}")
    sections.append(
        f"**Generated:** {now}  \n"
        f"**Experiment:** `{analysis.experiment_name}` (ID: `{analysis.experiment_id}`)  \n"
        f"**Runs analyzed:** {analysis.n_runs}  \n"
        f"**Target metric:** `{analysis.target_metric}`"
    )

    # ── Metrics comparison ────────────────────────────────────────────────────
    sections.append("## Metrics Comparison")
    if analysis.runs:
        from src.analysis.metrics import compare_metrics, rank_runs

        df = compare_metrics(analysis.runs)
        if analysis.target_metric in df.columns:
            is_lower = any(
                t in analysis.target_metric
                for t in {"loss", "rmse", "mae", "mse", "error"}
            )
            df = rank_runs(df, analysis.target_metric, ascending=is_lower)

        metric_cols = [c for c in df.columns if c != "run_name"]
        display_cols = ["run_name"] + metric_cols if "run_name" in df.columns else metric_cols
        sections.append(_df_to_markdown(df[display_cols]))
    else:
        sections.append("_No runs available for comparison._")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    sections.append("## Diagnostics")
    if overfitting_reports:
        overfit_found = [r for r in overfitting_reports if r.is_overfit]
        if overfit_found:
            rows = ["| Run | Severity | Worst gap | Message |", "|---|---|---|---|"]
            for r in overfit_found:
                icon = _SEVERITY_ICON.get(r.severity, "")
                worst = max(r.gaps, key=lambda g: g.gap, default=None)
                gap_str = f"{worst.gap:.3f} ({worst.metric_base})" if worst else "—"
                rows.append(f"| `{r.run_name or r.run_id}` | {icon} {r.severity.value} | {gap_str} | {r.message} |")
            sections.append("\n".join(rows))
        else:
            sections.append("✅ No overfitting detected in any analyzed run.")
    else:
        sections.append("_Overfitting analysis not performed._")

    # ── Patterns ──────────────────────────────────────────────────────────────
    sections.append("## Hyperparameter Patterns")
    report = analysis.correlation_report
    if report and report.correlations:
        rows = [
            "| Parameter | Correlation | Direction | Runs |",
            "|---|---|---|---|",
        ]
        for p in report.top_params[:10]:
            rows.append(
                f"| `{p.param}` | {p.correlation:.3f} | {p.direction} | {p.n_runs} |"
            )
        sections.append("\n".join(rows))
        sections.append(f"_{report.message}_")
    elif report:
        sections.append(f"_{report.message}_")
    else:
        sections.append("_Pattern analysis not performed._")

    # ── Recommendations ───────────────────────────────────────────────────────
    sections.append("## Recommendations")
    if analysis.suggestions:
        for i, s in enumerate(analysis.suggestions, 1):
            params_str = ", ".join(f"`{k}={v}`" for k, v in s.params.items())
            sections.append(
                f"### {i}. {s.title}\n\n"
                f"**Suggested params:** {params_str or '—'}  \n"
                f"**Justification:** {s.justification}  \n"
                f"**Hypothesis:** {s.hypothesis}"
            )
    else:
        sections.append("_No suggestions generated._")

    # ── Limitations & Disclaimer ──────────────────────────────────────────────
    sections.append("## Limitations & Disclaimer")
    sections.append(
        "- This report was generated autonomously by an AI agent. "
        "All conclusions should be validated by a domain expert.\n"
        "- Correlations are Pearson (linear) — non-linear relationships may be missed.\n"
        "- Overfitting detection is heuristic (train/val gap threshold). "
        "It does not account for dataset shift or class imbalance.\n"
        "- Suggestions are based on historical data only and do not guarantee improvement."
    )

    return "\n\n---\n\n".join(sections)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _df_to_markdown(df: "pd.DataFrame") -> str:  # type: ignore[name-defined]
    """Convert a DataFrame to a Markdown table string."""
    import pandas as pd

    if df.empty:
        return "_Empty._"

    # Round float columns for readability
    df = df.copy()
    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].round(4)

    lines = ["| " + " | ".join([df.index.name or "run_id"] + list(df.columns)) + " |"]
    lines.append("|" + "|".join(["---"] * (len(df.columns) + 1)) + "|")
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)
