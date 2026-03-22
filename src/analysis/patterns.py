"""Hyperparameter × metric correlation analysis.

Identifies which parameters have the strongest linear relationship with a
target metric across a set of runs. Only numeric parameters are included.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.mlflow_client.models import RunDetails


@dataclass
class ParamCorrelation:
    """Correlation of one parameter with the target metric."""

    param: str
    correlation: float  # Pearson r ∈ [-1, 1]
    direction: str  # "positive" | "negative" | "neutral"
    n_runs: int  # Number of runs used (non-null pairs)


@dataclass
class CorrelationReport:
    """Result of correlating hyperparameters against a target metric."""

    target_metric: str
    experiment_id: str
    n_runs: int
    correlations: list[ParamCorrelation] = field(default_factory=list)
    message: str = ""

    @property
    def top_params(self) -> list[ParamCorrelation]:
        """Parameters sorted by |correlation| descending."""
        return sorted(self.correlations, key=lambda c: abs(c.correlation), reverse=True)


def correlate_params_metrics(
    runs: list[RunDetails],
    target_metric: str,
    min_runs: int = 3,
) -> CorrelationReport:
    """Compute Pearson correlation between each numeric param and a target metric.

    Args:
        runs: List of RunDetails. All must belong to the same experiment.
        target_metric: Metric to correlate against (e.g. "val_accuracy").
        min_runs: Minimum runs needed for meaningful correlation.

    Returns:
        CorrelationReport sorted by |correlation| descending.
    """
    experiment_id = runs[0].experiment_id if runs else ""

    if len(runs) < min_runs:
        return CorrelationReport(
            target_metric=target_metric,
            experiment_id=experiment_id,
            n_runs=len(runs),
            message=(
                f"Not enough runs for correlation analysis (need {min_runs}, got {len(runs)})."
            ),
        )

    # Build a DataFrame of numeric params + target metric
    rows = []
    for run in runs:
        if target_metric not in run.metrics:
            continue
        row: dict[str, object] = {target_metric: run.metrics[target_metric]}
        for k, v in run.params.items():
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                pass  # Skip non-numeric params (e.g. model_type)
        rows.append(row)

    if not rows:
        return CorrelationReport(
            target_metric=target_metric,
            experiment_id=experiment_id,
            n_runs=0,
            message=f"No runs contain the metric '{target_metric}'.",
        )

    df = pd.DataFrame(rows)
    numeric_params = [c for c in df.columns if c != target_metric]

    if not numeric_params:
        return CorrelationReport(
            target_metric=target_metric,
            experiment_id=experiment_id,
            n_runs=len(df),
            message="No numeric parameters found for correlation analysis.",
        )

    correlations: list[ParamCorrelation] = []
    for param in numeric_params:
        pair = df[[param, target_metric]].dropna()
        if len(pair) < 2 or pair[param].nunique() < 2:
            continue
        r = float(pair[param].corr(pair[target_metric]))
        if pd.isna(r):
            continue
        correlations.append(
            ParamCorrelation(
                param=param,
                correlation=round(r, 4),
                direction="positive" if r > 0.1 else ("negative" if r < -0.1 else "neutral"),
                n_runs=len(pair),
            )
        )

    return CorrelationReport(
        target_metric=target_metric,
        experiment_id=experiment_id,
        n_runs=len(df),
        correlations=sorted(correlations, key=lambda c: abs(c.correlation), reverse=True),
        message=f"Analyzed {len(df)} runs and {len(correlations)} numeric parameters.",
    )
