"""Dataclasses for MLflow API responses.

These models provide a stable interface between the MLflow SDK and the rest of
the application. Tools and analysis modules depend on these types, not on
mlflow's own response objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExperimentInfo:
    """Metadata for an MLflow experiment."""

    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str
    creation_time: datetime | None = None
    last_update_time: datetime | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class RunInfo:
    """Summary of a single MLflow run (no detailed metrics/params)."""

    run_id: str
    experiment_id: str
    run_name: str
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class RunDetails:
    """Full details of a single MLflow run including params, metrics, and tags."""

    run_id: str
    experiment_id: str
    run_name: str
    status: str
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    artifact_uri: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def train_metrics(self) -> dict[str, float]:
        """Metrics prefixed with 'train_'."""
        return {k: v for k, v in self.metrics.items() if k.startswith("train_")}

    @property
    def val_metrics(self) -> dict[str, float]:
        """Metrics prefixed with 'val_'."""
        return {k: v for k, v in self.metrics.items() if k.startswith("val_")}
