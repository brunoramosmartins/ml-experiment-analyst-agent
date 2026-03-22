#!/usr/bin/env python3
"""
Seed MLflow with synthetic demo experiments for development and portfolio demonstration.

Experiments created:
  1. binary-classification  — 17 runs, multiple sklearn models, hyperparameter sweep
  2. regression-v2          — 7 runs, progressive feature engineering
  3. overfit-test           — 5 runs, intentional overfitting (for agent to detect)

Usage:
    make seed-mlflow
    # or
    python scripts/seed_mlflow.py
"""

import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

try:
    import mlflow
    import mlflow.sklearn
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -e '.[dev]'")
    sys.exit(1)

mlflow.set_tracking_uri(TRACKING_URI)
np.random.seed(42)


# ─── Experiment 1: Binary Classification ──────────────────────────────────────

def seed_binary_classification() -> None:
    """17 runs across 4 model types with hyperparameter sweep."""
    print("\n[1/3] Seeding binary-classification experiment...")
    mlflow.set_experiment("binary-classification")

    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    runs = [
        ("logistic_regression", {"C": 0.01, "max_iter": 100, "solver": "lbfgs"}),
        ("logistic_regression", {"C": 0.1, "max_iter": 100, "solver": "lbfgs"}),
        ("logistic_regression", {"C": 1.0, "max_iter": 200, "solver": "lbfgs"}),
        ("logistic_regression", {"C": 10.0, "max_iter": 200, "solver": "lbfgs"}),
        ("random_forest", {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 5}),
        ("random_forest", {"n_estimators": 50, "max_depth": 5, "min_samples_leaf": 2}),
        ("random_forest", {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 1}),
        ("random_forest", {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1}),
        ("gradient_boosting", {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3}),
        ("gradient_boosting", {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}),
        ("gradient_boosting", {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4}),
        ("gradient_boosting", {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4}),
        ("gradient_boosting", {"n_estimators": 200, "learning_rate": 0.01, "max_depth": 5}),
        ("decision_tree", {"max_depth": 3, "min_samples_leaf": 10}),
        ("decision_tree", {"max_depth": 5, "min_samples_leaf": 5}),
        ("decision_tree", {"max_depth": 10, "min_samples_leaf": 2}),
        ("decision_tree", {"max_depth": None, "min_samples_leaf": 1}),
    ]

    for model_type, params in runs:
        with mlflow.start_run():
            mlflow.set_tags({
                "model_type": model_type,
                "dataset": "synthetic-binary",
                "phase": "hyperparameter-sweep",
            })
            mlflow.log_params({"model_type": model_type, **params})

            if model_type == "logistic_regression":
                model = LogisticRegression(**params)
            elif model_type == "random_forest":
                model = RandomForestClassifier(random_state=42, **params)
            elif model_type == "gradient_boosting":
                model = GradientBoostingClassifier(random_state=42, **params)
            else:
                model = DecisionTreeClassifier(random_state=42, **params)

            model.fit(X_train, y_train)

            for split, X_s, y_s in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                preds = model.predict(X_s)
                has_proba = hasattr(model, "predict_proba")
                proba = model.predict_proba(X_s)[:, 1] if has_proba else None
                metrics = {
                    f"{split}_accuracy": float(accuracy_score(y_s, preds)),
                    f"{split}_f1": float(f1_score(y_s, preds)),
                }
                if proba is not None:
                    metrics[f"{split}_auc"] = float(roc_auc_score(y_s, proba))
                mlflow.log_metrics(metrics)

    print("  ✓ binary-classification: 17 runs")


# ─── Experiment 2: Regression with Feature Engineering ────────────────────────

def seed_regression() -> None:
    """7 runs with progressive feature engineering strategy."""
    print("\n[2/3] Seeding regression-v2 experiment...")
    mlflow.set_experiment("regression-v2")

    X_raw, y = make_regression(
        n_samples=1000, n_features=15, n_informative=8, noise=20, random_state=42
    )
    df = pd.DataFrame(X_raw, columns=[f"feature_{i}" for i in range(15)])

    run_configs = [
        {"name": "baseline-5-features", "features": list(range(5)),
         "alpha": 1.0, "engineering": "none"},
        {"name": "extended-10-features", "features": list(range(10)),
         "alpha": 1.0, "engineering": "none"},
        {"name": "all-15-features", "features": list(range(15)),
         "alpha": 1.0, "engineering": "none"},
        {"name": "polynomial-8-features", "features": list(range(8)),
         "alpha": 0.1, "engineering": "polynomial"},
        {"name": "high-regularization", "features": list(range(15)),
         "alpha": 100.0, "engineering": "none"},
        {"name": "optimal-regularization", "features": list(range(15)),
         "alpha": 10.0, "engineering": "none"},
        {"name": "poly-with-regularization", "features": list(range(8)),
         "alpha": 10.0, "engineering": "polynomial"},
    ]

    for cfg in run_configs:
        with mlflow.start_run(run_name=cfg["name"]):
            X = df[[f"feature_{i}" for i in cfg["features"]]].values
            if cfg["engineering"] == "polynomial":
                X = np.column_stack([X, X ** 2])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42,
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            mlflow.set_tags({
                "feature_engineering": cfg["engineering"],
                "phase": "feature-engineering-study",
                "dataset": "synthetic-regression",
            })
            mlflow.log_params({
                "model_type": "ridge",
                "alpha": cfg["alpha"],
                "n_features_input": len(cfg["features"]),
                "feature_engineering": cfg["engineering"],
            })

            model = Ridge(alpha=cfg["alpha"])
            model.fit(X_train, y_train)

            for split, X_s, y_s in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                preds = model.predict(X_s)
                mlflow.log_metrics({
                    f"{split}_rmse": float(np.sqrt(mean_squared_error(y_s, preds))),
                    f"{split}_mae": float(mean_absolute_error(y_s, preds)),
                    f"{split}_r2": float(r2_score(y_s, preds)),
                })


    print("  ✓ regression-v2: 7 runs")


# ─── Experiment 3: Intentional Overfitting ────────────────────────────────────

def seed_overfit() -> None:
    """5 runs with increasing complexity — intentional overfitting for agent to detect."""
    print("\n[3/3] Seeding overfit-test experiment...")
    mlflow.set_experiment("overfit-test")

    X, y = make_classification(
        n_samples=300,
        n_features=30,
        n_informative=4,
        n_redundant=12,
        n_classes=2,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.35, random_state=42)

    run_configs = [
        {"max_depth": 2,    "n_estimators": 10,  "gap_level": "none",      "lr": 0.1},
        {"max_depth": 4,    "n_estimators": 50,  "gap_level": "low",       "lr": 0.2},
        {"max_depth": 8,    "n_estimators": 100, "gap_level": "medium",    "lr": 0.3},
        {"max_depth": 15,   "n_estimators": 200, "gap_level": "high",      "lr": 0.3},
        {"max_depth": 30,   "n_estimators": 500, "gap_level": "critical",  "lr": 0.5},
    ]

    for cfg in run_configs:
        with mlflow.start_run():
            model = GradientBoostingClassifier(
                max_depth=cfg["max_depth"],
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["lr"],
                min_samples_leaf=1,
                subsample=1.0,
                random_state=42,
            )
            model.fit(X_train, y_train)

            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            train_proba = model.predict_proba(X_train)[:, 1]
            val_proba = model.predict_proba(X_val)[:, 1]

            train_acc = float(accuracy_score(y_train, train_preds))
            val_acc = float(accuracy_score(y_val, val_preds))
            train_f1 = float(f1_score(y_train, train_preds))
            val_f1 = float(f1_score(y_val, val_preds))

            mlflow.set_tags({
                "expected_overfit_level": cfg["gap_level"],
                "dataset": "synthetic-overfit",
                "phase": "overfitting-study",
                "warning": (
                    "intentional_overfit"
                    if cfg["gap_level"] in ("high", "critical")
                    else "none"
                ),
            })
            mlflow.log_params({
                "model_type": "gradient_boosting",
                "max_depth": cfg["max_depth"],
                "n_estimators": cfg["n_estimators"],
                "learning_rate": cfg["lr"],
                "min_samples_leaf": 1,
            })
            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "train_auc": float(roc_auc_score(y_train, train_proba)),
                "val_auc": float(roc_auc_score(y_val, val_proba)),
                "overfit_gap_accuracy": round(train_acc - val_acc, 4),
                "overfit_gap_f1": round(train_f1 - val_f1, 4),
            })

    print("  ✓ overfit-test: 5 runs (intentional overfitting gradient)")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Connecting to MLflow at: {TRACKING_URI}")
    try:
        mlflow.search_experiments()
    except Exception as e:
        print(f"\n✗ Could not connect to MLflow: {e}")
        print("  Make sure the stack is running: make mlflow-up")
        sys.exit(1)

    seed_binary_classification()
    seed_regression()
    seed_overfit()

    print(f"\n✅ All experiments seeded. Open MLflow UI at: {TRACKING_URI}")
