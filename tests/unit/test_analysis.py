"""Unit tests for analysis modules: metrics, overfitting, patterns, suggestions."""

from __future__ import annotations

import pytest

from src.analysis.metrics import compare_metrics, metric_delta, rank_runs
from src.analysis.overfitting import (
    OverfitSeverity,
    detect_overfitting,
    detect_overfitting_trend,
)
from src.analysis.patterns import correlate_params_metrics
from src.analysis.suggestions import AnalysisResult, suggest_next_experiments
from src.mlflow_client.models import RunDetails

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _run(
    run_id: str,
    metrics: dict,
    params: dict | None = None,
    run_name: str = "",
) -> RunDetails:
    return RunDetails(
        run_id=run_id,
        experiment_id="exp-1",
        run_name=run_name or run_id,
        status="FINISHED",
        params={k: str(v) for k, v in (params or {}).items()},
        metrics=metrics,
    )


# ─── metrics.py ───────────────────────────────────────────────────────────────


class TestCompareMetrics:
    def test_returns_dataframe_with_all_metrics(self) -> None:
        runs = [
            _run("a", {"val_accuracy": 0.9, "val_f1": 0.88}),
            _run("b", {"val_accuracy": 0.85, "val_f1": 0.82}),
        ]
        df = compare_metrics(runs)
        assert set(df.columns) >= {"val_accuracy", "val_f1"}
        assert len(df) == 2

    def test_empty_list_returns_empty_dataframe(self) -> None:
        df = compare_metrics([])
        assert df.empty

    def test_missing_metrics_filled_with_nan(self) -> None:
        import pandas as pd

        runs = [
            _run("a", {"val_accuracy": 0.9}),
            _run("b", {"val_accuracy": 0.85, "val_f1": 0.82}),
        ]
        df = compare_metrics(runs)
        assert pd.isna(df.loc["a", "val_f1"])


class TestRankRuns:
    def test_ranks_descending_by_default(self) -> None:
        runs = [_run("a", {"val_acc": 0.7}), _run("b", {"val_acc": 0.9})]
        df = compare_metrics(runs)
        ranked = rank_runs(df, "val_acc")
        assert ranked.index[0] == "b"

    def test_ranks_ascending_for_loss(self) -> None:
        runs = [_run("a", {"val_loss": 0.3}), _run("b", {"val_loss": 0.1})]
        df = compare_metrics(runs)
        ranked = rank_runs(df, "val_loss", ascending=True)
        assert ranked.index[0] == "b"

    def test_raises_on_unknown_metric(self) -> None:
        runs = [_run("a", {"val_acc": 0.9})]
        df = compare_metrics(runs)
        with pytest.raises(ValueError, match="not found"):
            rank_runs(df, "nonexistent_metric")


class TestMetricDelta:
    def test_computes_delta_for_shared_metrics(self) -> None:
        a = _run("a", {"val_acc": 0.9, "val_f1": 0.88})
        b = _run("b", {"val_acc": 0.85, "val_f1": 0.82})
        delta = metric_delta(a, b)
        assert delta["val_acc"] == pytest.approx(0.05)
        assert delta["val_f1"] == pytest.approx(0.06)

    def test_only_shared_metrics_included(self) -> None:
        a = _run("a", {"val_acc": 0.9, "train_acc": 0.95})
        b = _run("b", {"val_acc": 0.85})
        delta = metric_delta(a, b)
        assert "val_acc" in delta
        assert "train_acc" not in delta


# ─── overfitting.py ───────────────────────────────────────────────────────────


class TestDetectOverfitting:
    def test_detects_high_overfit(self) -> None:
        run = _run("a", {"train_accuracy": 0.99, "val_accuracy": 0.55})
        report = detect_overfitting(run)
        assert report.is_overfit
        assert report.severity == OverfitSeverity.HIGH

    def test_no_overfit_when_gap_is_small(self) -> None:
        run = _run("a", {"train_accuracy": 0.90, "val_accuracy": 0.88})
        report = detect_overfitting(run)
        assert not report.is_overfit
        assert report.severity == OverfitSeverity.NONE

    def test_missing_val_metrics_returns_none_severity(self) -> None:
        run = _run("a", {"train_accuracy": 0.90})
        report = detect_overfitting(run)
        assert not report.is_overfit
        assert "missing" in report.message.lower()

    def test_missing_train_metrics_returns_none_severity(self) -> None:
        run = _run("a", {"val_accuracy": 0.85})
        report = detect_overfitting(run)
        assert not report.is_overfit

    def test_lower_is_better_metric(self) -> None:
        # For loss: val > train means overfitting
        run = _run("a", {"train_loss": 0.1, "val_loss": 0.5})
        report = detect_overfitting(run)
        assert report.is_overfit

    def test_affected_metrics_populated(self) -> None:
        run = _run(
            "a", {"train_accuracy": 0.99, "val_accuracy": 0.60, "train_f1": 0.98, "val_f1": 0.59}
        )
        report = detect_overfitting(run)
        assert len(report.affected_metrics) > 0


class TestDetectOverfittingTrend:
    def test_returns_one_report_per_run(self) -> None:
        runs = [
            _run("a", {"train_accuracy": 0.99, "val_accuracy": 0.55}),
            _run("b", {"train_accuracy": 0.90, "val_accuracy": 0.88}),
        ]
        reports = detect_overfitting_trend(runs)
        assert len(reports) == 2
        assert reports[0].is_overfit
        assert not reports[1].is_overfit

    def test_empty_list_returns_empty(self) -> None:
        assert detect_overfitting_trend([]) == []


# ─── patterns.py ──────────────────────────────────────────────────────────────


class TestCorrelateParamsMetrics:
    def test_finds_positive_correlation(self) -> None:
        runs = [
            _run("a", {"val_accuracy": 0.9}, params={"n_estimators": 200}),
            _run("b", {"val_accuracy": 0.85}, params={"n_estimators": 100}),
            _run("c", {"val_accuracy": 0.80}, params={"n_estimators": 50}),
            _run("d", {"val_accuracy": 0.75}, params={"n_estimators": 10}),
        ]
        report = correlate_params_metrics(runs, "val_accuracy")
        assert len(report.correlations) > 0
        top = report.top_params[0]
        assert top.param == "n_estimators"
        assert top.correlation > 0

    def test_not_enough_runs_returns_message(self) -> None:
        runs = [_run("a", {"val_accuracy": 0.9}, params={"n": 10})]
        report = correlate_params_metrics(runs, "val_accuracy", min_runs=3)
        assert "Not enough" in report.message
        assert report.correlations == []

    def test_missing_target_metric_in_all_runs(self) -> None:
        runs = [
            _run("a", {"train_accuracy": 0.9}, params={"n": 10}),
            _run("b", {"train_accuracy": 0.85}, params={"n": 5}),
            _run("c", {"train_accuracy": 0.80}, params={"n": 1}),
        ]
        report = correlate_params_metrics(runs, "val_accuracy")
        assert "val_accuracy" in report.message

    def test_non_numeric_params_ignored(self) -> None:
        runs = [
            _run("a", {"val_acc": 0.9}, params={"model": "rf", "n": 100}),
            _run("b", {"val_acc": 0.8}, params={"model": "lr", "n": 50}),
            _run("c", {"val_acc": 0.7}, params={"model": "dt", "n": 10}),
            _run("d", {"val_acc": 0.6}, params={"model": "gb", "n": 5}),
        ]
        report = correlate_params_metrics(runs, "val_acc")
        param_names = [c.param for c in report.correlations]
        assert "model" not in param_names


# ─── suggestions.py ───────────────────────────────────────────────────────────


class TestSuggestNextExperiments:
    def test_returns_suggestions(self) -> None:
        runs = [
            _run("a", {"val_accuracy": 0.9}, params={"n_estimators": 200}),
            _run("b", {"val_accuracy": 0.8}, params={"n_estimators": 100}),
        ]
        analysis = AnalysisResult(
            experiment_name="test-exp",
            experiment_id="1",
            n_runs=2,
            target_metric="val_accuracy",
            runs=runs,
        )
        suggestions = suggest_next_experiments(analysis)
        assert len(suggestions) >= 1
        assert suggestions[0].title
        assert suggestions[0].justification

    def test_empty_runs_returns_empty_list(self) -> None:
        analysis = AnalysisResult(
            experiment_name="empty",
            experiment_id="1",
            n_runs=0,
            target_metric="val_accuracy",
            runs=[],
        )
        assert suggest_next_experiments(analysis) == []

    def test_respects_num_suggestions_limit(self) -> None:
        runs = [
            _run("a", {"val_accuracy": 0.9}, params={"n": 100}),
            _run("b", {"val_accuracy": 0.8}, params={"n": 50}),
        ]
        analysis = AnalysisResult(
            experiment_name="test",
            experiment_id="1",
            n_runs=2,
            target_metric="val_accuracy",
            runs=runs,
        )
        suggestions = suggest_next_experiments(analysis, num_suggestions=1)
        assert len(suggestions) <= 1
