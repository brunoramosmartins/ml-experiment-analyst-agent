"""Fixtures for integration tests.

Integration tests require a running MLflow server. If the server is not
available, tests are automatically skipped.
"""

from __future__ import annotations

import os
import uuid

import pytest

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: requires external services (MLflow, Docker)",
    )


@pytest.fixture(scope="session")
def mlflow_available() -> bool:
    """Check if MLflow is reachable; skip the test if not."""
    try:
        import requests

        resp = requests.get(f"{MLFLOW_URI}/api/2.0/mlflow/experiments/search", timeout=2)
        return resp.status_code < 500
    except Exception:
        pytest.skip(f"MLflow not available at {MLFLOW_URI}")
        return False  # unreachable, keeps type checker happy


@pytest.fixture(scope="session")
def mlflow_client(mlflow_available: bool):  # type: ignore[no-untyped-def]
    """Return a real MLflowAnalystClient connected to the test server."""
    from src.mlflow_client.client import MLflowAnalystClient

    return MLflowAnalystClient(tracking_uri=MLFLOW_URI)


@pytest.fixture()
def seeded_experiment(mlflow_available: bool):  # type: ignore[no-untyped-def]
    """Create a temporary experiment with 3 runs, and clean up after.

    Yields:
        (experiment_name, experiment_id, run_ids)
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)

    exp_name = f"integration-test-{uuid.uuid4().hex[:8]}"
    exp_id = mlflow.create_experiment(exp_name)

    run_ids: list[str] = []
    for i in range(3):
        with mlflow.start_run(experiment_id=exp_id, run_name=f"run-{i}") as run:
            mlflow.log_params(
                {"learning_rate": str(0.01 * (i + 1)), "epochs": "10"}
            )
            mlflow.log_metrics(
                {
                    "train_accuracy": 0.90 + i * 0.02,
                    "val_accuracy": 0.85 + i * 0.01,
                    "train_loss": 0.10 - i * 0.01,
                    "val_loss": 0.15 - i * 0.005,
                }
            )
            run_ids.append(run.info.run_id)

    yield exp_name, exp_id, run_ids

    # Cleanup
    try:
        mlflow.delete_experiment(exp_id)
    except Exception:
        pass
