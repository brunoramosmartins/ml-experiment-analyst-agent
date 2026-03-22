"""Edge case tests for analysis modules.

Tests degenerate data: NaN values, ties, constant parameters, extreme gaps, etc.
No external dependencies — pure unit tests on data models and analysis functions.
"""

from __future__ import annotations

import math

from src.analysis.metrics import compare_metrics, metric_delta, rank_runs
from src.analysis.overfitting import OverfitSeverity, detect_overfitting
from src.analysis.patterns import correlate_params_metrics
from src.mlflow_client.models import RunDetails


def _make_run(
    run_id: str,
    metrics: dict[str, float],
    params: dict[str, str] | None = None,
) -> RunDetails:
    return RunDetails(
        run_id=run_id,
        experiment_id="exp-1",
        run_name=f"run-{run_id}",
        status="FINISHED",
        metrics=metrics,
        params=params or {},
    )


# ─── Metrics edge cases ─────────────────────────────────────────────────────


def test_compare_metrics_nan_values() -> None:
    runs = [
        _make_run("r1", {"val_accuracy": 0.9, "val_loss": float("nan")}),
        _make_run("r2", {"val_accuracy": 0.8, "val_loss": 0.2}),
    ]
    df = compare_metrics(runs)
    assert len(df) == 2
    assert "val_accuracy" in df.columns
    # NaN should be preserved, not crash
    assert math.isnan(df.loc[df.index[0], "val_loss"])


def test_rank_runs_all_same_value() -> None:
    runs = [
        _make_run("r1", {"val_accuracy": 0.85}),
        _make_run("r2", {"val_accuracy": 0.85}),
        _make_run("r3", {"val_accuracy": 0.85}),
    ]
    df = compare_metrics(runs)
    ranked = rank_runs(df, "val_accuracy")
    assert len(ranked) == 3  # Should not crash on ties


def test_metric_delta_identical_runs() -> None:
    run_a = _make_run("r1", {"val_accuracy": 0.9, "val_loss": 0.1})
    run_b = _make_run("r2", {"val_accuracy": 0.9, "val_loss": 0.1})
    deltas = metric_delta(run_a, run_b)
    for key, value in deltas.items():
        assert value == 0.0, f"Expected delta 0 for {key}, got {value}"


# ─── Overfitting edge cases ─────────────────────────────────────────────────


def test_overfitting_extreme_gap() -> None:
    run = _make_run("r1", {"train_accuracy": 1.0, "val_accuracy": 0.0})
    report = detect_overfitting(run)
    assert report.is_overfit is True
    assert report.severity == OverfitSeverity.HIGH


def test_overfitting_negative_gap() -> None:
    # Val better than train — not overfitting
    run = _make_run("r1", {"train_accuracy": 0.80, "val_accuracy": 0.85})
    report = detect_overfitting(run)
    assert report.is_overfit is False


# ─── Patterns edge cases ────────────────────────────────────────────────────


def test_correlate_constant_param() -> None:
    """All runs have the same parameter value — correlation is undefined."""
    runs = [
        _make_run("r1", {"val_accuracy": 0.9}, params={"lr": "0.01"}),
        _make_run("r2", {"val_accuracy": 0.8}, params={"lr": "0.01"}),
        _make_run("r3", {"val_accuracy": 0.7}, params={"lr": "0.01"}),
    ]
    report = correlate_params_metrics(runs, "val_accuracy")
    # Should not crash; either no correlations or NaN handled
    assert report.n_runs == 3
