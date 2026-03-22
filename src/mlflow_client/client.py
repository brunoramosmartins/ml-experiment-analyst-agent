"""MLflow access layer for the ML Experiment Analyst Agent.

This client abstracts the MLflow SDK so tools and analysis modules never
import mlflow directly. All error handling is centralised here and surfaces
clear, actionable messages instead of raw stack traces.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import mlflow
import pandas as pd
from mlflow.entities import Run
from mlflow.exceptions import MlflowException

from src.mlflow_client.models import ExperimentInfo, RunDetails, RunInfo


class MLflowClientError(Exception):
    """Raised when the MLflow client encounters a recoverable error."""


class MLflowAnalystClient:
    """High-level MLflow client for experiment analysis.

    Args:
        tracking_uri: MLflow tracking server URI.
            Defaults to the MLFLOW_TRACKING_URI environment variable.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._tracking_uri: str = (
            tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000"
        )
        mlflow.set_tracking_uri(self._tracking_uri)
        self._client = mlflow.MlflowClient(tracking_uri=self._tracking_uri)

    # ─── Experiments ──────────────────────────────────────────────────────────

    def get_experiment(self, name_or_id: str) -> ExperimentInfo:
        """Fetch experiment metadata by name or ID.

        Args:
            name_or_id: Experiment name (e.g. "binary-classification") or numeric ID.

        Returns:
            ExperimentInfo with metadata.

        Raises:
            MLflowClientError: If the experiment is not found or the server is offline.
        """
        try:
            exp = self._client.get_experiment_by_name(name_or_id)
            if exp is None:
                # Try by ID
                try:
                    exp = self._client.get_experiment(name_or_id)
                except MlflowException:
                    exp = None

            if exp is None:
                raise MLflowClientError(
                    f"Experiment '{name_or_id}' not found. "
                    "Check the name with mlflow.search_experiments()."
                )

            return ExperimentInfo(
                experiment_id=exp.experiment_id,
                name=exp.name,
                artifact_location=exp.artifact_location,
                lifecycle_stage=exp.lifecycle_stage,
                creation_time=_ms_to_dt(exp.creation_time),
                last_update_time=_ms_to_dt(exp.last_update_time),
                tags=dict(exp.tags) if exp.tags else {},
            )
        except MLflowClientError:
            raise
        except Exception as exc:
            raise MLflowClientError(
                f"Could not connect to MLflow at {self._tracking_uri}. "
                f"Make sure the server is running. Details: {exc}"
            ) from exc

    # ─── Runs ─────────────────────────────────────────────────────────────────

    def list_runs(
        self,
        experiment_id: str,
        filter_string: str = "",
        order_by: str | None = None,
        max_results: int = 50,
    ) -> list[RunInfo]:
        """List runs for an experiment.

        Args:
            experiment_id: Experiment ID (numeric string).
            filter_string: MLflow filter expression, e.g. "metrics.val_accuracy > 0.8".
            order_by: Ordering, e.g. ["metrics.val_loss ASC"].
            max_results: Maximum number of runs to return.

        Returns:
            List of RunInfo ordered by start_time descending by default.

        Raises:
            MLflowClientError: If the experiment is not found or server is offline.
        """
        try:
            runs: list[Run] = mlflow.search_runs(  # type: ignore[assignment]
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                order_by=[order_by] if order_by else None,
                max_results=max_results,
                output_format="list",
            )
            return [
                RunInfo(
                    run_id=r.info.run_id,
                    experiment_id=r.info.experiment_id,
                    run_name=r.info.run_name or "",
                    status=r.info.status,
                    start_time=_ms_to_dt(r.info.start_time),
                    end_time=_ms_to_dt(r.info.end_time),
                    tags=dict(r.data.tags) if r.data.tags else {},
                )
                for r in runs
            ]
        except MlflowException as exc:
            raise MLflowClientError(
                f"Could not list runs for experiment '{experiment_id}'. MLflow error: {exc}"
            ) from exc
        except Exception as exc:
            raise MLflowClientError(f"Unexpected error listing runs: {exc}") from exc

    def get_run_details(self, run_id: str) -> RunDetails:
        """Fetch full details of a run including params, metrics, and tags.

        Args:
            run_id: MLflow run ID.

        Returns:
            RunDetails with params, metrics, and tags.

        Raises:
            MLflowClientError: If the run is not found or has no data.
        """
        try:
            run = self._client.get_run(run_id)
        except MlflowException as exc:
            raise MLflowClientError(
                f"Run '{run_id}' not found. Verify the run ID is correct. Details: {exc}"
            ) from exc
        except Exception as exc:
            raise MLflowClientError(f"Could not fetch run '{run_id}': {exc}") from exc

        metrics = {k: float(v) for k, v in run.data.metrics.items()}
        if not metrics:
            raise MLflowClientError(
                f"Run '{run_id}' has no logged metrics. "
                "The run may have failed before logging any data."
            )

        return RunDetails(
            run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            run_name=run.info.run_name or "",
            status=run.info.status,
            params=dict(run.data.params) if run.data.params else {},
            metrics=metrics,
            tags=dict(run.data.tags) if run.data.tags else {},
            artifact_uri=run.info.artifact_uri or "",
            start_time=_ms_to_dt(run.info.start_time),
            end_time=_ms_to_dt(run.info.end_time),
        )

    def compare_runs(self, run_ids: list[str]) -> pd.DataFrame:
        """Fetch multiple runs and return a comparison DataFrame.

        Rows are run IDs; columns are all params and metrics found across runs.
        Missing values are filled with NaN.

        Args:
            run_ids: List of MLflow run IDs.

        Returns:
            DataFrame with index=run_id, columns=params+metrics.

        Raises:
            MLflowClientError: If any run cannot be fetched.
        """
        if not run_ids:
            return pd.DataFrame()

        rows = []
        for run_id in run_ids:
            details = self.get_run_details(run_id)
            row: dict[str, object] = {"run_id": run_id, "run_name": details.run_name}
            row.update({f"param_{k}": v for k, v in details.params.items()})
            row.update(details.metrics)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("run_id")
        return df


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _ms_to_dt(ms: int | None) -> datetime | None:
    """Convert millisecond timestamp to UTC datetime."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
