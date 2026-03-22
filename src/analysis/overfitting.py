"""Overfitting detection for ML experiment runs.

Heuristic: for every train/val metric pair, compute gap = train − val.
A positive gap above the threshold is considered overfitting for
higher-is-better metrics (accuracy, f1, auc, r2).
For lower-is-better metrics (loss, rmse, mae, mse), the gap is val − train.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

if sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]  # noqa: UP042
        """Backport for Python < 3.11."""

from src.mlflow_client.models import RunDetails

# Metrics where *lower* values are better (gap = val − train)
_LOWER_IS_BETTER = {"loss", "rmse", "mae", "mse", "error"}


class OverfitSeverity(StrEnum):
    NONE = "none"
    LOW = "low"        # 0.05 ≤ gap < 0.10
    MEDIUM = "medium"  # 0.10 ≤ gap < 0.20
    HIGH = "high"      # gap ≥ 0.20


@dataclass
class MetricGap:
    """Gap between train and val for one metric."""

    metric_base: str  # e.g. "accuracy" (without train_/val_ prefix)
    train_value: float
    val_value: float
    gap: float
    severity: OverfitSeverity


@dataclass
class OverfittingReport:
    """Overfitting analysis result for a single run."""

    run_id: str
    run_name: str
    is_overfit: bool
    severity: OverfitSeverity
    gaps: list[MetricGap] = field(default_factory=list)
    message: str = ""

    @property
    def affected_metrics(self) -> list[str]:
        return [g.metric_base for g in self.gaps if g.severity != OverfitSeverity.NONE]


def detect_overfitting(
    run: RunDetails,
    threshold: float = 0.05,
) -> OverfittingReport:
    """Detect overfitting in a single run by comparing train vs val metrics.

    Args:
        run: RunDetails with metrics dict.
        threshold: Minimum gap to flag as LOW severity overfitting (default 0.05).

    Returns:
        OverfittingReport with gap details per metric.
    """
    train_metrics = run.train_metrics
    val_metrics = run.val_metrics

    if not train_metrics or not val_metrics:
        return OverfittingReport(
            run_id=run.run_id,
            run_name=run.run_name,
            is_overfit=False,
            severity=OverfitSeverity.NONE,
            message="Insufficient data: train or val metrics are missing.",
        )

    # Find metric pairs: train_X / val_X
    train_bases = {k.removeprefix("train_") for k in train_metrics}
    val_bases = {k.removeprefix("val_") for k in val_metrics}
    shared_bases = train_bases & val_bases

    if not shared_bases:
        return OverfittingReport(
            run_id=run.run_id,
            run_name=run.run_name,
            is_overfit=False,
            severity=OverfitSeverity.NONE,
            message="No matching train/val metric pairs found.",
        )

    gaps: list[MetricGap] = []
    for base in sorted(shared_bases):
        train_val = train_metrics[f"train_{base}"]
        val_val = val_metrics[f"val_{base}"]

        lower_is_better = any(token in base for token in _LOWER_IS_BETTER)
        gap = (val_val - train_val) if lower_is_better else (train_val - val_val)

        severity = _gap_to_severity(gap, threshold)
        gaps.append(
            MetricGap(
                metric_base=base,
                train_value=train_val,
                val_value=val_val,
                gap=round(gap, 4),
                severity=severity,
            )
        )

    worst = max(gaps, key=lambda g: g.gap)
    overall_severity = _gap_to_severity(worst.gap, threshold)
    is_overfit = overall_severity != OverfitSeverity.NONE

    message = (
        f"Overfitting detected ({overall_severity.value}): "
        f"worst gap {worst.gap:.3f} on '{worst.metric_base}' "
        f"(train={worst.train_value:.4f}, val={worst.val_value:.4f})."
        if is_overfit
        else "No significant overfitting detected."
    )

    return OverfittingReport(
        run_id=run.run_id,
        run_name=run.run_name,
        is_overfit=is_overfit,
        severity=overall_severity,
        gaps=gaps,
        message=message,
    )


def detect_overfitting_trend(
    runs: list[RunDetails],
    threshold: float = 0.05,
) -> list[OverfittingReport]:
    """Run detect_overfitting on a list of runs.

    Args:
        runs: List of RunDetails.
        threshold: Minimum gap threshold passed to detect_overfitting.

    Returns:
        List of OverfittingReport, one per run, in the same order as input.
    """
    return [detect_overfitting(run, threshold) for run in runs]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gap_to_severity(gap: float, threshold: float) -> OverfitSeverity:
    if gap < threshold:
        return OverfitSeverity.NONE
    if gap < 0.10:
        return OverfitSeverity.LOW
    if gap < 0.20:
        return OverfitSeverity.MEDIUM
    return OverfitSeverity.HIGH
