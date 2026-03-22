"""Edge case tests for custom LangChain tools.

Tests degenerate inputs: zero runs, missing metrics, MLflow offline, etc.
All tests mock MLflowAnalystClient — no real MLflow server required.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

from src.mlflow_client.client import MLflowClientError
from src.mlflow_client.models import ExperimentInfo, RunDetails, RunInfo

_load_experiment_mod = importlib.import_module("src.tools.load_experiment")
_compare_runs_mod = importlib.import_module("src.tools.compare_runs")
_diagnose_run_mod = importlib.import_module("src.tools.diagnose_run")
_analyze_patterns_mod = importlib.import_module("src.tools.analyze_patterns")
_suggest_mod = importlib.import_module("src.tools.suggest_next_experiments")
_generate_report_mod = importlib.import_module("src.tools.generate_report")


def _make_experiment(name: str = "edge-exp", exp_id: str = "1") -> ExperimentInfo:
    return ExperimentInfo(
        experiment_id=exp_id,
        name=name,
        artifact_location="s3://bucket",
        lifecycle_stage="active",
    )


def _make_run(
    run_id: str = "run-001",
    run_name: str = "edge-run",
    params: dict | None = None,
    metrics: dict | None = None,
) -> RunDetails:
    return RunDetails(
        run_id=run_id,
        experiment_id="1",
        run_name=run_name,
        status="FINISHED",
        params=params or {"lr": "0.01"},
        metrics=metrics or {"train_accuracy": 0.9, "val_accuracy": 0.8},
    )


# ─── Zero runs ───────────────────────────────────────────────────────────────


def test_load_experiment_zero_runs() -> None:
    with (
        patch.object(
            _load_experiment_mod, "MLflowAnalystClient"
        ) as mock_cls,
    ):
        client = mock_cls.return_value
        client.get_experiment.return_value = _make_experiment()
        client.list_runs.return_value = []

        result = _load_experiment_mod.load_experiment.invoke(
            {"experiment_name": "edge-exp"}
        )
        assert "0 run" in result.lower() or "no runs" in result.lower() or "0" in result


# ─── Single run ──────────────────────────────────────────────────────────────


def test_compare_runs_single_run() -> None:
    run = _make_run()
    with patch.object(_compare_runs_mod, "MLflowAnalystClient") as mock_cls:
        client = mock_cls.return_value
        client.get_run_details.return_value = run

        result = _compare_runs_mod.compare_runs.invoke(
            {"run_ids": ["run-001"], "experiment_name": "edge-exp"}
        )
        assert "run-001" in result or "edge-run" in result


def test_suggest_single_run() -> None:
    with patch.object(_suggest_mod, "MLflowAnalystClient") as mock_cls:
        client = mock_cls.return_value
        client.get_experiment.return_value = _make_experiment()
        client.list_runs.return_value = [
            RunInfo(
                run_id="run-001", experiment_id="1", run_name="r1",
                status="FINISHED",
            )
        ]
        client.get_run_details.return_value = _make_run()

        result = _suggest_mod.suggest_next_experiments.invoke(
            {"experiment_name": "edge-exp"}
        )
        assert isinstance(result, str)
        assert "ERROR" not in result


# ─── Missing metrics ─────────────────────────────────────────────────────────


def test_diagnose_run_no_val_metrics() -> None:
    run = _make_run(metrics={"train_accuracy": 0.95, "train_loss": 0.1})
    with patch.object(_diagnose_run_mod, "MLflowAnalystClient") as mock_cls:
        client = mock_cls.return_value
        client.get_run_details.return_value = run

        result = _diagnose_run_mod.diagnose_run.invoke({"run_id": "run-001"})
        # Should handle gracefully — no val metrics for overfitting detection
        assert "ERROR" not in result


def test_analyze_patterns_non_numeric_params() -> None:
    runs = [
        _make_run("r1", "run-1", params={"model": "rf", "scaler": "standard"}),
        _make_run("r2", "run-2", params={"model": "xgb", "scaler": "minmax"}),
        _make_run("r3", "run-3", params={"model": "lgbm", "scaler": "robust"}),
    ]
    with patch.object(_analyze_patterns_mod, "MLflowAnalystClient") as mock_cls:
        client = mock_cls.return_value
        client.get_experiment.return_value = _make_experiment()
        client.list_runs.return_value = [
            RunInfo(
                run_id=r.run_id, experiment_id="1",
                run_name=r.run_name, status="FINISHED",
            )
            for r in runs
        ]
        client.get_run_details.side_effect = runs

        result = _analyze_patterns_mod.analyze_patterns.invoke(
            {"experiment_name": "edge-exp", "target_metric": "val_accuracy"}
        )
        assert isinstance(result, str)


def test_generate_report_no_target_metric() -> None:
    runs = [
        _make_run("r1", "run-1", metrics={"other_metric": 0.5}),
    ]
    with patch.object(_generate_report_mod, "MLflowAnalystClient") as mock_cls:
        client = mock_cls.return_value
        client.get_experiment.return_value = _make_experiment()
        client.list_runs.return_value = [
            RunInfo(
                run_id="r1", experiment_id="1",
                run_name="run-1", status="FINISHED",
            )
        ]
        client.get_run_details.return_value = runs[0]

        result = _generate_report_mod.generate_report.invoke(
            {"experiment_name": "edge-exp", "report_title": "Edge Test Report"}
        )
        # Should generate a report even if target metric is missing
        assert isinstance(result, str)
        assert "ERROR" not in result


# ─── MLflow errors ───────────────────────────────────────────────────────────


def test_load_experiment_connection_error() -> None:
    with patch.object(_load_experiment_mod, "MLflowAnalystClient") as mock_cls:
        mock_cls.return_value.get_experiment.side_effect = MLflowClientError(
            "Connection refused"
        )

        result = _load_experiment_mod.load_experiment.invoke(
            {"experiment_name": "edge-exp"}
        )
        assert "ERROR" in result


def test_compare_runs_run_not_found() -> None:
    with patch.object(_compare_runs_mod, "MLflowAnalystClient") as mock_cls:
        mock_cls.return_value.get_run_details.side_effect = MLflowClientError(
            "Run not found"
        )

        result = _compare_runs_mod.compare_runs.invoke(
            {"run_ids": ["bad-id"], "experiment_name": "edge-exp"}
        )
        assert "ERROR" in result
