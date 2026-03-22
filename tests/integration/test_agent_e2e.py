"""End-to-end integration tests against a real MLflow server.

These tests are skipped automatically if MLflow is not available.
Run with:
    pytest tests/integration/ -v -m integration
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.integration


def test_mlflow_client_get_experiment(
    mlflow_client,
    seeded_experiment,  # type: ignore[no-untyped-def]
) -> None:
    exp_name, exp_id, _ = seeded_experiment
    exp = mlflow_client.get_experiment(exp_name)
    assert exp.name == exp_name
    assert exp.experiment_id == exp_id


def test_mlflow_client_list_runs(
    mlflow_client,
    seeded_experiment,  # type: ignore[no-untyped-def]
) -> None:
    _, exp_id, run_ids = seeded_experiment
    runs = mlflow_client.list_runs(exp_id)
    assert len(runs) == 3
    fetched_ids = {r.run_id for r in runs}
    assert fetched_ids == set(run_ids)


def test_load_experiment_tool_e2e(
    mlflow_available,
    seeded_experiment,  # type: ignore[no-untyped-def]
) -> None:
    exp_name, _, _ = seeded_experiment
    mod = importlib.import_module("src.tools.load_experiment")
    result = mod.load_experiment.invoke({"experiment_name": exp_name})
    assert exp_name in result
    assert "ERROR" not in result


def test_compare_runs_tool_e2e(
    mlflow_available,
    seeded_experiment,  # type: ignore[no-untyped-def]
) -> None:
    exp_name, _, run_ids = seeded_experiment
    mod = importlib.import_module("src.tools.compare_runs")
    result = mod.compare_runs.invoke({"run_ids": run_ids[:2], "experiment_name": exp_name})
    assert "val_accuracy" in result
    assert "ERROR" not in result


def test_generate_report_tool_e2e(
    mlflow_available,
    seeded_experiment,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    exp_name, _, _ = seeded_experiment
    mod = importlib.import_module("src.tools.generate_report")
    result = mod.generate_report.invoke({"experiment_name": exp_name})
    assert "Analysis Report" in result or "ERROR" not in result
