"""Unit tests for the Markdown report generator."""

from __future__ import annotations

import pandas as pd

from src.analysis.overfitting import MetricGap, OverfitSeverity, OverfittingReport
from src.analysis.patterns import CorrelationReport, ParamCorrelation
from src.analysis.suggestions import AnalysisResult, ExperimentSuggestion
from src.mlflow_client.models import RunDetails
from src.report.generator import _df_to_markdown, generate_markdown_report

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_run(run_id: str, name: str, metrics: dict[str, float]) -> RunDetails:
    return RunDetails(
        run_id=run_id,
        experiment_id="exp-1",
        run_name=name,
        status="FINISHED",
        metrics=metrics,
        params={"lr": "0.01", "epochs": "10"},
    )


def _make_analysis(
    runs: list[RunDetails] | None = None,
    target_metric: str = "val_accuracy",
    correlation_report: CorrelationReport | None = None,
    suggestions: list[ExperimentSuggestion] | None = None,
) -> AnalysisResult:
    runs = runs or []
    return AnalysisResult(
        experiment_name="test-experiment",
        experiment_id="exp-1",
        n_runs=len(runs),
        target_metric=target_metric,
        runs=runs,
        correlation_report=correlation_report,
        suggestions=suggestions or [],
    )


# ─── Header ──────────────────────────────────────────────────────────────────


def test_report_contains_header() -> None:
    analysis = _make_analysis(
        runs=[_make_run("r1", "run-1", {"val_accuracy": 0.9})]
    )
    report = generate_markdown_report(analysis)
    assert "# Analysis Report: test-experiment" in report
    assert "**Runs analyzed:** 1" in report
    assert "`val_accuracy`" in report


# ─── Metrics Comparison ──────────────────────────────────────────────────────


def test_metrics_comparison_with_runs() -> None:
    runs = [
        _make_run("r1", "run-1", {"val_accuracy": 0.9}),
        _make_run("r2", "run-2", {"val_accuracy": 0.85}),
    ]
    report = generate_markdown_report(_make_analysis(runs=runs))
    assert "## Metrics Comparison" in report
    assert "val_accuracy" in report


def test_metrics_comparison_no_runs() -> None:
    report = generate_markdown_report(_make_analysis(runs=[]))
    assert "_No runs available for comparison._" in report


# ─── Diagnostics ─────────────────────────────────────────────────────────────


def test_diagnostics_overfitting_detected() -> None:
    overfit_reports = [
        OverfittingReport(
            run_id="r1",
            run_name="run-1",
            is_overfit=True,
            severity=OverfitSeverity.HIGH,
            gaps=[
                MetricGap(
                    metric_base="accuracy",
                    train_value=0.98,
                    val_value=0.75,
                    gap=0.23,
                    severity=OverfitSeverity.HIGH,
                )
            ],
            message="High overfitting detected",
        )
    ]
    report = generate_markdown_report(_make_analysis(), overfit_reports)
    assert "HIGH" in report or "high" in report
    assert "0.23" in report.replace("0.230", "0.23")


def test_diagnostics_no_overfitting() -> None:
    overfit_reports = [
        OverfittingReport(
            run_id="r1",
            run_name="run-1",
            is_overfit=False,
            severity=OverfitSeverity.NONE,
            message="No overfitting",
        )
    ]
    report = generate_markdown_report(_make_analysis(), overfit_reports)
    assert "No overfitting detected" in report


def test_diagnostics_not_performed() -> None:
    report = generate_markdown_report(_make_analysis(), overfitting_reports=None)
    assert "_Overfitting analysis not performed._" in report


# ─── Patterns ────────────────────────────────────────────────────────────────


def test_patterns_with_correlations() -> None:
    corr_report = CorrelationReport(
        target_metric="val_accuracy",
        experiment_id="exp-1",
        n_runs=10,
        correlations=[
            ParamCorrelation(
                param="lr", correlation=0.85, direction="positive", n_runs=10
            ),
        ],
        message="Strong positive correlation found",
    )
    report = generate_markdown_report(
        _make_analysis(correlation_report=corr_report)
    )
    assert "`lr`" in report
    assert "0.85" in report.replace("0.850", "0.85")


def test_patterns_no_correlations() -> None:
    corr_report = CorrelationReport(
        target_metric="val_accuracy",
        experiment_id="exp-1",
        n_runs=2,
        correlations=[],
        message="Not enough runs for correlation analysis",
    )
    report = generate_markdown_report(
        _make_analysis(correlation_report=corr_report)
    )
    assert "Not enough runs" in report


def test_patterns_not_performed() -> None:
    report = generate_markdown_report(
        _make_analysis(correlation_report=None)
    )
    assert "_Pattern analysis not performed._" in report


# ─── Recommendations ─────────────────────────────────────────────────────────


def test_recommendations_with_suggestions() -> None:
    suggestions = [
        ExperimentSuggestion(
            title="Increase learning rate",
            params={"lr": "0.1"},
            justification="Higher lr showed improvement",
            hypothesis="Faster convergence",
        ),
        ExperimentSuggestion(
            title="Add dropout",
            params={"dropout": "0.3"},
            justification="Overfitting observed",
            hypothesis="Better generalization",
        ),
    ]
    report = generate_markdown_report(
        _make_analysis(suggestions=suggestions)
    )
    assert "### 1." in report
    assert "### 2." in report
    assert "`lr=0.1`" in report


def test_recommendations_empty() -> None:
    report = generate_markdown_report(_make_analysis(suggestions=[]))
    assert "_No suggestions generated._" in report


# ─── Limitations ─────────────────────────────────────────────────────────────


def test_limitations_always_present() -> None:
    report = generate_markdown_report(_make_analysis())
    assert "## Limitations & Disclaimer" in report


# ─── _df_to_markdown ─────────────────────────────────────────────────────────


def test_df_to_markdown_normal() -> None:
    df = pd.DataFrame(
        {"metric_a": [0.9, 0.8], "metric_b": [0.1, 0.2]},
        index=["run-1", "run-2"],
    )
    result = _df_to_markdown(df)
    assert "|" in result
    assert "metric_a" in result
    assert "run-1" in result


def test_df_to_markdown_empty() -> None:
    df = pd.DataFrame()
    result = _df_to_markdown(df)
    assert result == "_Empty._"


def test_df_to_markdown_rounds_floats() -> None:
    df = pd.DataFrame(
        {"val": [0.123456789]}, index=["run-1"]
    )
    result = _df_to_markdown(df)
    assert "0.1235" in result
    assert "0.123456789" not in result
