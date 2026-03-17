"""Tool: diagnose_run — detect overfitting and other issues in a single run."""

from __future__ import annotations

from langchain_core.tools import tool

from src.analysis.overfitting import OverfitSeverity, detect_overfitting
from src.mlflow_client.client import MLflowAnalystClient, MLflowClientError

_SEVERITY_ICON = {
    OverfitSeverity.NONE: "✅ INFO",
    OverfitSeverity.LOW: "⚠️ WARNING",
    OverfitSeverity.MEDIUM: "⚠️ WARNING",
    OverfitSeverity.HIGH: "🔴 CRITICAL",
}

_SEVERITY_ACTIONS = {
    OverfitSeverity.NONE: "No action needed.",
    OverfitSeverity.LOW: "Consider light regularization (e.g. increase weight_decay or dropout).",
    OverfitSeverity.MEDIUM: (
        "Add regularization, reduce model complexity, or increase training data. "
        "Investigate data leakage."
    ),
    OverfitSeverity.HIGH: (
        "Strong overfitting. Reduce model capacity, apply dropout/L2 regularization, "
        "add early stopping, or collect more data."
    ),
}


@tool
def diagnose_run(
    run_id: str,
    overfitting_threshold: float = 0.05,
) -> str:
    """Diagnose a single MLflow run for common problems.

    Checks for overfitting (train/val metric gap), missing validation metrics,
    and failed run status. Use this after compare_runs to understand WHY a
    specific run performed as it did.

    Args:
        run_id: Full MLflow run ID to diagnose.
        overfitting_threshold: Relative gap above which overfitting is flagged (default: 0.05).
    """
    client = MLflowAnalystClient()

    try:
        run = client.get_run_details(run_id)
    except MLflowClientError as exc:
        return f"ERROR fetching run `{run_id[:8]}...`: {exc}"

    diagnostics: list[str] = []

    # ── Check 1: Run status ───────────────────────────────────────────────────
    if run.status != "FINISHED":
        diagnostics.append(
            f"🔴 CRITICAL — Run status: `{run.status}`\n"
            f"  The run did not finish successfully. Metrics may be incomplete.\n"
            f"  **Action:** Check your training script logs for errors."
        )

    # ── Check 2: Missing validation metrics ──────────────────────────────────
    has_train = bool(run.train_metrics)
    has_val = bool(run.val_metrics)

    if not has_train and not has_val:
        diagnostics.append(
            "⚠️ WARNING — No train or val metrics logged.\n"
            "  Overfitting analysis is not possible without split metrics.\n"
            "  **Action:** Log `train_*` and `val_*` prefixed metrics in your training loop."
        )
    elif has_train and not has_val:
        diagnostics.append(
            "⚠️ WARNING — Only train metrics logged (no `val_*` metrics).\n"
            "  Cannot assess overfitting without a validation split.\n"
            "  **Action:** Add a validation set and log `val_*` metrics."
        )
    elif not has_train and has_val:
        diagnostics.append(
            "ℹ️ INFO — Only val metrics logged (no `train_*` metrics).\n"
            "  Overfitting analysis requires both train and val metrics.\n"
            "  **Action:** Also log `train_*` metrics during training."
        )

    # ── Check 3: Overfitting detection ────────────────────────────────────────
    overfit_report = detect_overfitting(run, threshold=overfitting_threshold)

    if overfit_report.gaps:
        icon = _SEVERITY_ICON.get(overfit_report.severity, "ℹ️ INFO")
        action = _SEVERITY_ACTIONS.get(overfit_report.severity, "")
        lines = [
            f"{icon} — Overfitting analysis: {overfit_report.message}",
            f"  **Action:** {action}",
            "  **Metric gaps (train − val):**",
        ]
        for gap in sorted(overfit_report.gaps, key=lambda g: -g.gap):
            gap_icon = _SEVERITY_ICON.get(gap.severity, "✅ INFO")
            lines.append(
                f"  - `{gap.metric_base}`: train={gap.train_value:.4f}, "
                f"val={gap.val_value:.4f}, gap={gap.gap:.4f} [{gap_icon}]"
            )
        diagnostics.append("\n".join(lines))
    elif overfit_report.message:
        diagnostics.append(f"ℹ️ INFO — {overfit_report.message}")

    # ── Check 4: Metric value sanity ──────────────────────────────────────────
    suspicious: list[str] = []
    for name, val in run.metrics.items():
        if "accuracy" in name and not (0.0 <= val <= 1.0):
            suspicious.append(f"`{name}` = {val:.4f} (expected in [0, 1])")
        if "loss" in name and val < 0:
            suspicious.append(f"`{name}` = {val:.4f} (negative loss is unusual)")

    if suspicious:
        diagnostics.append(
            "⚠️ WARNING — Suspicious metric values:\n"
            + "\n".join(f"  - {s}" for s in suspicious)
            + "\n  **Action:** Verify metric calculation in your training script."
        )

    # ── Summary output ────────────────────────────────────────────────────────
    name_label = run.run_name or run_id
    header = [
        f"# Diagnosis: `{name_label}`",
        f"- **Run ID:** `{run_id}`",
        f"- **Status:** {run.status}",
        f"- **All metrics:** {', '.join(f'`{k}={v:.4f}`' for k, v in sorted(run.metrics.items()))}",
        "- **Parameters:** "
        + (", ".join(f"`{k}={v}`" for k, v in sorted(run.params.items())) or "(none)"),
        "",
        f"## Diagnostic Results ({len(diagnostics)} finding(s))",
        "",
    ]

    if not diagnostics:
        header.append("✅ No issues found. This run looks healthy.")
    else:
        for i, d in enumerate(diagnostics, 1):
            header.append(f"### Finding {i}\n{d}")

    return "\n".join(header)
