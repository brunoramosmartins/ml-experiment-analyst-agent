"""Next-experiment suggestion engine.

Generates concrete hyperparameter configurations to try, based on:
- The best-performing runs in the comparison
- Parameter correlations with the target metric
- Unexplored regions of the parameter space
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.analysis.metrics import compare_metrics, rank_runs
from src.analysis.patterns import CorrelationReport
from src.mlflow_client.models import RunDetails


@dataclass
class ExperimentSuggestion:
    """A concrete suggestion for the next experiment run."""

    title: str
    params: dict[str, str]     # Suggested parameter values
    justification: str          # Why this configuration is suggested
    hypothesis: str             # What we expect to learn from this run
    priority: int = 1           # 1 = highest priority


@dataclass
class AnalysisResult:
    """Aggregated analysis result, used as input to the report generator."""

    experiment_name: str
    experiment_id: str
    n_runs: int
    target_metric: str
    runs: list[RunDetails] = field(default_factory=list)
    correlation_report: CorrelationReport | None = None
    suggestions: list[ExperimentSuggestion] = field(default_factory=list)


def suggest_next_experiments(
    analysis: AnalysisResult,
    num_suggestions: int = 3,
) -> list[ExperimentSuggestion]:
    """Generate concrete next-experiment suggestions from an AnalysisResult.

    Strategy:
    1. Find the best run by target metric.
    2. Use correlation report to identify high-impact parameters.
    3. Suggest: push best params further, penalise positive overfit indicators.

    Args:
        analysis: AnalysisResult with runs and optional correlation report.
        num_suggestions: Maximum number of suggestions to return.

    Returns:
        List of ExperimentSuggestion ordered by priority.
    """
    suggestions: list[ExperimentSuggestion] = []

    if not analysis.runs:
        return suggestions

    # ── Suggestion 1: based on best run ───────────────────────────────────────
    df = compare_metrics(analysis.runs)
    if analysis.target_metric in df.columns:
        is_lower_better = any(
            t in analysis.target_metric for t in {"loss", "rmse", "mae", "mse", "error"}
        )
        ranked = rank_runs(df, analysis.target_metric, ascending=is_lower_better)
        best_run_id = str(ranked.index[0])
        best_run = next((r for r in analysis.runs if r.run_id == best_run_id), None)

        if best_run and best_run.params:
            best_metric_val = ranked.iloc[0][analysis.target_metric]
            suggestions.append(
                ExperimentSuggestion(
                    title="Refine best configuration",
                    params=dict(best_run.params),
                    justification=(
                        f"Run '{best_run.run_name or best_run_id}' achieved the best "
                        f"{analysis.target_metric} = {best_metric_val:.4f}. "
                        "Fine-tune its parameters for potential improvement."
                    ),
                    hypothesis=(
                        f"Small perturbations around the best configuration "
                        f"may yield further gains in {analysis.target_metric}."
                    ),
                    priority=1,
                )
            )

    # ── Suggestion 2: based on top positive correlation ───────────────────────
    report = analysis.correlation_report
    if report and report.top_params:
        top = next(
            (p for p in report.top_params if p.direction == "positive"), None
        )
        if top:
            suggestions.append(
                ExperimentSuggestion(
                    title=f"Increase '{top.param}' (strong positive correlation)",
                    params={top.param: "increase from current best"},
                    justification=(
                        f"'{top.param}' has a Pearson correlation of {top.correlation:.2f} "
                        f"with {analysis.target_metric}. "
                        "Increasing it may improve performance."
                    ),
                    hypothesis=(
                        f"Higher values of '{top.param}' correlate with "
                        f"better {analysis.target_metric}."
                    ),
                    priority=2,
                )
            )

    # ── Suggestion 3: based on top negative correlation ───────────────────────
    if report and report.top_params:
        top_neg = next(
            (p for p in report.top_params if p.direction == "negative"), None
        )
        if top_neg:
            suggestions.append(
                ExperimentSuggestion(
                    title=f"Decrease '{top_neg.param}' (negative correlation)",
                    params={top_neg.param: "decrease from current best"},
                    justification=(
                        f"'{top_neg.param}' has a Pearson correlation of "
                        f"{top_neg.correlation:.2f} with {analysis.target_metric}. "
                        "Reducing it may improve performance."
                    ),
                    hypothesis=(
                        f"Lower values of '{top_neg.param}' are associated "
                        f"with better {analysis.target_metric}."
                    ),
                    priority=3,
                )
            )

    return suggestions[:num_suggestions]
