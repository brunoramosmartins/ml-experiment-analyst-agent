"""Unit tests for MLflowAnalystClient.

All tests use unittest.mock — no real MLflow server required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.mlflow_client.client import MLflowAnalystClient, MLflowClientError
from src.mlflow_client.models import RunDetails

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def client() -> MLflowAnalystClient:
    with patch("src.mlflow_client.client.mlflow"), \
         patch("src.mlflow_client.client.mlflow.MlflowClient"):
        c = MLflowAnalystClient(tracking_uri="http://fake:5000")
    return c


def _mock_run(
    run_id: str = "run-001",
    run_name: str = "test-run",
    params: dict | None = None,
    metrics: dict | None = None,
) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.info.experiment_id = "exp-1"
    run.info.run_name = run_name
    run.info.status = "FINISHED"
    run.info.start_time = 1_700_000_000_000
    run.info.end_time = 1_700_000_060_000
    run.info.artifact_uri = "s3://bucket/run-001"
    default_params = {"n_estimators": "100", "max_depth": "5"}
    default_metrics = {"train_accuracy": 0.95, "val_accuracy": 0.88}
    run.data.params = params if params is not None else default_params
    run.data.metrics = metrics if metrics is not None else default_metrics
    run.data.tags = {"model_type": "random_forest"}
    return run


# ─── get_experiment ────────────────────────────────────────────────────────────

def test_get_experiment_by_name(client: MLflowAnalystClient) -> None:
    mock_exp = MagicMock()
    mock_exp.experiment_id = "1"
    mock_exp.name = "binary-classification"
    mock_exp.artifact_location = "s3://bucket/1"
    mock_exp.lifecycle_stage = "active"
    mock_exp.creation_time = 1_700_000_000_000
    mock_exp.last_update_time = 1_700_000_000_000
    mock_exp.tags = {}

    client._client.get_experiment_by_name = MagicMock(return_value=mock_exp)

    exp = client.get_experiment("binary-classification")

    assert exp.name == "binary-classification"
    assert exp.experiment_id == "1"


def test_get_experiment_not_found_raises(client: MLflowAnalystClient) -> None:
    client._client.get_experiment_by_name = MagicMock(return_value=None)
    client._client.get_experiment = MagicMock(side_effect=Exception("not found"))

    with pytest.raises(MLflowClientError, match="not found"):
        client.get_experiment("nonexistent-experiment")


# ─── list_runs ────────────────────────────────────────────────────────────────

def test_list_runs_returns_run_info(client: MLflowAnalystClient) -> None:
    mock_run = _mock_run()

    with patch("src.mlflow_client.client.mlflow.search_runs", return_value=[mock_run]):
        runs = client.list_runs("1")

    assert len(runs) == 1
    assert runs[0].run_id == "run-001"
    assert runs[0].status == "FINISHED"


def test_list_runs_empty(client: MLflowAnalystClient) -> None:
    with patch("src.mlflow_client.client.mlflow.search_runs", return_value=[]):
        runs = client.list_runs("1")

    assert runs == []


# ─── get_run_details ──────────────────────────────────────────────────────────

def test_get_run_details_returns_full_data(client: MLflowAnalystClient) -> None:
    mock_run = _mock_run()
    client._client.get_run = MagicMock(return_value=mock_run)

    details = client.get_run_details("run-001")

    assert details.run_id == "run-001"
    assert details.params["n_estimators"] == "100"
    assert details.metrics["train_accuracy"] == pytest.approx(0.95)
    assert details.metrics["val_accuracy"] == pytest.approx(0.88)


def test_get_run_details_no_metrics_raises(client: MLflowAnalystClient) -> None:
    mock_run = _mock_run(metrics={})
    client._client.get_run = MagicMock(return_value=mock_run)

    with pytest.raises(MLflowClientError, match="no logged metrics"):
        client.get_run_details("run-001")


def test_get_run_details_not_found_raises(client: MLflowAnalystClient) -> None:
    from mlflow.exceptions import MlflowException

    client._client.get_run = MagicMock(side_effect=MlflowException("not found"))

    with pytest.raises(MLflowClientError, match="not found"):
        client.get_run_details("bad-run-id")


# ─── compare_runs ─────────────────────────────────────────────────────────────

def test_compare_runs_returns_dataframe(client: MLflowAnalystClient) -> None:
    import pandas as pd

    run_a = RunDetails(
        run_id="a", experiment_id="1", run_name="run-a", status="FINISHED",
        params={"lr": "0.01"}, metrics={"val_accuracy": 0.90},
    )
    run_b = RunDetails(
        run_id="b", experiment_id="1", run_name="run-b", status="FINISHED",
        params={"lr": "0.001"}, metrics={"val_accuracy": 0.85},
    )

    client.get_run_details = MagicMock(side_effect=[run_a, run_b])

    df = client.compare_runs(["a", "b"])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "val_accuracy" in df.columns


def test_compare_runs_empty_returns_empty_df(client: MLflowAnalystClient) -> None:
    import pandas as pd

    df = client.compare_runs([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ─── Additional edge cases ───────────────────────────────────────────────────


def test_list_runs_mlflow_exception(client: MLflowAnalystClient) -> None:
    with patch(
        "src.mlflow_client.client.mlflow.search_runs",
        side_effect=Exception("connection refused"),
    ):
        with pytest.raises(MLflowClientError, match="connection refused"):
            client.list_runs("1")


def test_compare_runs_partial_metrics(client: MLflowAnalystClient) -> None:
    import pandas as pd

    run_a = RunDetails(
        run_id="a", experiment_id="1", run_name="run-a", status="FINISHED",
        params={}, metrics={"val_accuracy": 0.9, "val_loss": 0.1},
    )
    run_b = RunDetails(
        run_id="b", experiment_id="1", run_name="run-b", status="FINISHED",
        params={}, metrics={"val_accuracy": 0.8},  # no val_loss
    )

    client.get_run_details = MagicMock(side_effect=[run_a, run_b])
    df = client.compare_runs(["a", "b"])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    # run_b should have NaN for val_loss
    assert pd.isna(df.loc[df["run_name"] == "run-b", "val_loss"].iloc[0])
