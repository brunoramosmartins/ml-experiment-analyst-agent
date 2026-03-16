# Demo Experiments

This document describes the three synthetic MLflow experiments created by `scripts/seed_mlflow.py` for development and portfolio demonstration.

---

## Experiment 1: `binary-classification`

**Purpose:** Provide a realistic hyperparameter sweep scenario for the agent to compare runs and identify the best model configuration.

### Setup

| Property | Value |
|---|---|
| Dataset | Synthetic binary classification (sklearn `make_classification`) |
| Samples | 2,000 (train 64% / val 16% / test 20%) |
| Features | 20 (10 informative, 5 redundant, 5 noise) |
| Total runs | 17 |
| Split strategy | Train / Val / Test (val used for model selection) |

### Models and hyperparameters

| Model type | Runs | Key parameter range |
|---|---|---|
| Logistic Regression | 4 | C ∈ {0.01, 0.1, 1.0, 10.0} |
| Random Forest | 4 | n_estimators ∈ {10, 50, 100, 200}, max_depth ∈ {3, 5, 10, None} |
| Gradient Boosting | 5 | n_estimators ∈ {50–200}, learning_rate ∈ {0.01, 0.05, 0.1} |
| Decision Tree | 4 | max_depth ∈ {3, 5, 10, None} |

### Metrics logged

- `train_accuracy`, `val_accuracy`, `test_accuracy`
- `train_f1`, `val_f1`, `test_f1`
- `train_auc`, `val_auc` (where applicable)

### Expected agent behavior

- `load_experiment` → loads 17 runs with metrics summary
- `compare_runs` → identifies top-3 by `val_auc` or `val_f1`
- `analyze_patterns` → finds that `n_estimators` and `max_depth` correlate most with performance
- `diagnose_run` → no overfitting in best runs (val ≈ train)

---

## Experiment 2: `regression-v2`

**Purpose:** Demonstrate a feature engineering progression study, where each run adds or modifies features to understand their impact on generalization.

### Setup

| Property | Value |
|---|---|
| Dataset | Synthetic regression (sklearn `make_regression`) |
| Samples | 1,000 (train 64% / val 16% / test 20%) |
| Features | 15 raw (8 informative) |
| Total runs | 7 |
| Model | Ridge regression (alpha varies) |

### Run progression

| Run name | Features used | Engineering | Alpha | Expected behavior |
|---|---|---|---|---|
| baseline-5-features | 5 | none | 1.0 | High RMSE — underfitting |
| extended-10-features | 10 | none | 1.0 | Improved R² |
| all-15-features | 15 | none | 1.0 | Best raw-feature baseline |
| polynomial-8-features | 8 (+squares) | polynomial | 0.1 | Improved, but low regularization risk |
| high-regularization | 15 | none | 100.0 | Underfitting due to excessive regularization |
| optimal-regularization | 15 | none | 10.0 | Good generalization |
| poly-with-regularization | 8 (+squares) | polynomial | 10.0 | Best overall (hypothesis) |

### Metrics logged

- `train_rmse`, `val_rmse`, `test_rmse`
- `train_mae`, `val_mae`, `test_mae`
- `train_r2`, `val_r2`, `test_r2`

### Expected agent behavior

- `analyze_patterns` → identifies that `alpha` and `n_features_input` are the most impactful parameters
- `compare_runs` → highlights `poly-with-regularization` or `optimal-regularization` as best
- `suggest_next_experiments` → suggests exploring alpha ∈ {5, 15} and polynomial degree 3

---

## Experiment 3: `overfit-test`

**Purpose:** Intentional overfitting progression — designed so the agent can detect and diagnose increasing train/val gaps with precision.

### Setup

| Property | Value |
|---|---|
| Dataset | Synthetic binary classification (high noise, few informative features) |
| Samples | 300 (train 65% / val 35%) — small dataset to amplify overfitting |
| Features | 30 (4 informative, 12 redundant, 14 noise) |
| Total runs | 5 |
| Model | Gradient Boosting (complexity increases across runs) |

### Overfitting gradient

| Run | max_depth | n_estimators | learning_rate | Expected gap level |
|---|---|---|---|---|
| 1 | 2 | 10 | 0.1 | none (val ≈ train) |
| 2 | 4 | 50 | 0.2 | low (~5% gap) |
| 3 | 8 | 100 | 0.3 | medium (~15% gap) |
| 4 | 15 | 200 | 0.3 | high (~25% gap) |
| 5 | 30 | 500 | 0.5 | critical (train ~0.99 / val ~0.55) |

### Metrics logged

- `train_accuracy`, `val_accuracy`
- `train_f1`, `val_f1`
- `train_auc`, `val_auc`
- `overfit_gap_accuracy` = train_accuracy − val_accuracy
- `overfit_gap_f1` = train_f1 − val_f1

### Tags on runs 4 and 5

```
warning: intentional_overfit
expected_overfit_level: high / critical
```

### Expected agent behavior

- `diagnose_run` on run 5 → `🔴 CRITICAL: train_accuracy=0.99 / val_accuracy=0.55, gap=0.44`
- `diagnose_run` on run 1 → `ℹ️ INFO: no significant overfitting detected`
- `analyze_patterns` → `max_depth` and `n_estimators` are the highest-impact parameters (positively correlated with train metrics, negatively with val metrics at high values)
- `suggest_next_experiments` → recommends regularization via `min_samples_leaf`, `subsample < 1.0`, and `max_depth ∈ {3, 5}`

---

## Running the seed script

```bash
# Start the MLflow stack first
make mlflow-up

# Seed all three experiments
make seed-mlflow
```

After seeding, open the MLflow UI at `http://localhost:5000` to verify all experiments and runs are visible.
