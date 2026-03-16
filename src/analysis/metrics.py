"""Metric comparison functions for ML experiment analysis.

All functions are pure (no side effects, no external calls) to facilitate
unit testing and reuse across tools.
"""

from __future__ import annotations

import pandas as pd

from src.mlflow_client.models import RunDetails


def compare_metrics(runs: list[RunDetails]) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame for a list of runs.

    Args:
        runs: List of RunDetails to compare.

    Returns:
        DataFrame with index=run_id, columns=all metrics found across runs.
        Missing metrics are NaN.
    """
    if not runs:
        return pd.DataFrame()

    rows = []
    for run in runs:
        row: dict[str, object] = {
            "run_id": run.run_id,
            "run_name": run.run_name,
        }
        row.update(run.metrics)
        rows.append(row)

    return pd.DataFrame(rows).set_index("run_id")


def rank_runs(
    df: pd.DataFrame,
    metric: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """Sort the comparison DataFrame by a specific metric.

    Args:
        df: DataFrame produced by compare_metrics().
        metric: Column name to sort by (e.g. "val_accuracy").
        ascending: True for loss-style metrics (lower is better).

    Returns:
        Sorted DataFrame. Rows with NaN in the metric column are placed last.

    Raises:
        ValueError: If the metric column does not exist in the DataFrame.
    """
    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found. Available: {list(df.columns)}"
        )
    return df.sort_values(metric, ascending=ascending, na_position="last")


def metric_delta(run_a: RunDetails, run_b: RunDetails) -> dict[str, float]:
    """Compute the signed delta (run_a − run_b) for all shared metrics.

    Args:
        run_a: Reference run.
        run_b: Comparison run.

    Returns:
        Dict mapping metric name to delta value.
        Only metrics present in both runs are included.
    """
    shared = set(run_a.metrics) & set(run_b.metrics)
    return {m: run_a.metrics[m] - run_b.metrics[m] for m in sorted(shared)}
