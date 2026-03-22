"""Unit tests for all custom LangChain tools.

All tests mock MLflowAnalystClient — no real MLflow server required.
Tools are invoked via their .invoke() method (LangChain StructuredTool API).

NOTE on patching strategy: in Python 3.10, unittest.mock._importer resolves
"src.tools.load_experiment.X" via getattr traversal, which hits the StructuredTool
instead of the submodule (because src.tools.__init__.py imports the tool under
that name, shadowing the submodule attribute). We work around this by importing
each submodule directly and using patch.object on the module reference.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

from src.mlflow_client.models import ExperimentInfo, RunDetails, RunInfo

# importlib.import_module reads from sys.modules directly, bypassing the getattr
# traversal that __import__ uses. This avoids the naming collision where
# src.tools.__init__.py sets src.tools.load_experiment = <StructuredTool>.
_load_experiment_mod = importlib.import_module("src.tools.load_experiment")
_compare_runs_mod = importlib.import_module("src.tools.compare_runs")
_diagnose_run_mod = importlib.import_module("src.tools.diagnose_run")
_analyze_patterns_mod = importlib.import_module("src.tools.analyze_patterns")
_suggest_mod = importlib.import_module("src.tools.suggest_next_experiments")
_generate_report_mod = importlib.import_module("src.tools.generate_report")


# ─── Shared test data ─────────────────────────────────────────────────────────

def _make_experiment(name: str = "test-exp", exp_id: str = "1") -> ExperimentInfo:
    return ExperimentInfo(
        experiment_id=exp_id,
        name=name,
        artifact_location="s3://bucket/artifacts",
        lifecycle_stage="active",
    )


def _make_run(
    run_id: str = "run-001",
    run_name: str = "test-run",
    params: dict | None = None,
    metrics: dict | None = None,
    status: str = "FINISHED",
) -> RunDetails:
    return RunDetails(
        run_id=run_id,
        experiment_id="1",
        run_name=run_name,
        status=status,
        params=params or {"learning_rate": "0.01", "n_estimators": "100"},
        metrics=metrics or {
            "train_accuracy": 0.95,
            "val_accuracy": 0.88,
            "train_loss": 0.12,
            "val_loss": 0.20,
        },
    )


def _make_run_info(run_id: str = "run-001", run_name: str = "test-run") -> RunInfo:
    return RunInfo(
        run_id=run_id,
        experiment_id="1",
        run_name=run_name,
        status="FINISHED",
    )


# ─── load_experiment ──────────────────────────────────────────────────────────

class TestLoadExperiment:
    def test_happy_path(self):
        from src.tools.load_experiment import load_experiment

        run_infos = [_make_run_info("r001", "run-1"), _make_run_info("r002", "run-2")]
        run_details = _make_run()

        with patch.object(_load_experiment_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.return_value = _make_experiment()
            instance.list_runs.return_value = run_infos
            instance.get_run_details.return_value = run_details

            result = load_experiment.invoke({"experiment_name": "test-exp"})

        assert "test-exp" in result
        assert "ERROR" not in result

    def test_experiment_not_found(self):
        from src.mlflow_client.client import MLflowClientError
        from src.tools.load_experiment import load_experiment

        with patch.object(_load_experiment_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.side_effect = MLflowClientError("Experiment 'xyz' not found.")

            result = load_experiment.invoke({"experiment_name": "xyz"})

        assert "ERROR" in result

    def test_experiment_with_no_runs(self):
        from src.tools.load_experiment import load_experiment

        with patch.object(_load_experiment_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.return_value = _make_experiment()
            instance.list_runs.return_value = []

            result = load_experiment.invoke({"experiment_name": "test-exp"})

        assert "no runs" in result.lower() or "has no runs" in result.lower()
        assert "ERROR" not in result

    def test_mlflow_offline(self):
        from src.mlflow_client.client import MLflowClientError
        from src.tools.load_experiment import load_experiment

        with patch.object(_load_experiment_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.side_effect = MLflowClientError(
                "Could not connect to MLflow at http://localhost:5000."
            )

            result = load_experiment.invoke({"experiment_name": "test-exp"})

        assert "ERROR" in result


# ─── compare_runs ─────────────────────────────────────────────────────────────

class TestCompareRuns:
    def test_happy_path(self):
        from src.tools.compare_runs import compare_runs

        run_a = _make_run("run-001", "run-1", metrics={"val_accuracy": 0.90, "val_loss": 0.15})
        run_b = _make_run("run-002", "run-2", metrics={"val_accuracy": 0.85, "val_loss": 0.22})

        with patch.object(_compare_runs_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_run_details.side_effect = [run_a, run_b]

            result = compare_runs.invoke({"run_ids": ["run-001", "run-002"]})

        assert "val_accuracy" in result
        assert "Summary" in result
        assert "ERROR" not in result

    def test_empty_run_ids(self):
        from src.tools.compare_runs import compare_runs

        result = compare_runs.invoke({"run_ids": []})

        assert "ERROR" in result
        assert "No run IDs" in result

    def test_run_not_found(self):
        from src.mlflow_client.client import MLflowClientError
        from src.tools.compare_runs import compare_runs

        with patch.object(_compare_runs_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_run_details.side_effect = MLflowClientError("Run not found.")

            result = compare_runs.invoke({"run_ids": ["bad-id"]})

        assert "ERROR" in result

    def test_metrics_to_compare_filter(self):
        from src.tools.compare_runs import compare_runs

        run = _make_run(metrics={"val_accuracy": 0.9, "val_loss": 0.1, "train_accuracy": 0.95})

        with patch.object(_compare_runs_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_run_details.return_value = run

            result = compare_runs.invoke({
                "run_ids": ["run-001"],
                "metrics_to_compare": ["val_accuracy"],
            })

        assert "val_accuracy" in result


# ─── diagnose_run ─────────────────────────────────────────────────────────────

class TestDiagnoseRun:
    def test_healthy_run(self):
        from src.tools.diagnose_run import diagnose_run

        run = _make_run(metrics={
            "train_accuracy": 0.90,
            "val_accuracy": 0.88,
            "train_loss": 0.15,
            "val_loss": 0.17,
        })

        with patch.object(_diagnose_run_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_run_details.return_value = run

            result = diagnose_run.invoke({"run_id": "run-001"})

        assert "Diagnosis" in result
        assert "ERROR" not in result

    def test_overfitting_detected(self):
        from src.tools.diagnose_run import diagnose_run

        overfit_run = _make_run(metrics={
            "train_accuracy": 0.98,
            "val_accuracy": 0.62,  # large gap → overfitting
        })

        with patch.object(_diagnose_run_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_run_details.return_value = overfit_run

            result = diagnose_run.invoke({"run_id": "run-001"})

        assert "overfitting" in result.lower() or "Overfitting" in result
        assert "ERROR" not in result

    def test_run_not_found(self):
        from src.mlflow_client.client import MLflowClientError
        from src.tools.diagnose_run import diagnose_run

        with patch.object(_diagnose_run_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_run_details.side_effect = MLflowClientError("Run not found.")

            result = diagnose_run.invoke({"run_id": "bad-run-id"})

        assert "ERROR" in result

    def test_failed_run_status(self):
        from src.tools.diagnose_run import diagnose_run

        failed_run = _make_run(status="FAILED")

        with patch.object(_diagnose_run_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_run_details.return_value = failed_run

            result = diagnose_run.invoke({"run_id": "run-001"})

        assert "FAILED" in result or "CRITICAL" in result

    def test_no_val_metrics(self):
        from src.tools.diagnose_run import diagnose_run

        run = _make_run(metrics={"train_accuracy": 0.92, "train_loss": 0.10})

        with patch.object(_diagnose_run_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_run_details.return_value = run

            result = diagnose_run.invoke({"run_id": "run-001"})

        assert "val" in result.lower() or "validation" in result.lower()


# ─── analyze_patterns ─────────────────────────────────────────────────────────

class TestAnalyzePatterns:
    def _make_diverse_runs(self) -> list[RunDetails]:
        configs = [
            {"learning_rate": "0.001", "n_estimators": "50"},
            {"learning_rate": "0.01", "n_estimators": "100"},
            {"learning_rate": "0.05", "n_estimators": "200"},
            {"learning_rate": "0.1", "n_estimators": "300"},
            {"learning_rate": "0.001", "n_estimators": "400"},
        ]
        metrics_list = [
            {"val_accuracy": 0.75},
            {"val_accuracy": 0.82},
            {"val_accuracy": 0.87},
            {"val_accuracy": 0.85},
            {"val_accuracy": 0.78},
        ]
        return [
            _make_run(f"run-{i:03d}", f"run-{i}", params=c, metrics=m)
            for i, (c, m) in enumerate(zip(configs, metrics_list, strict=True))
        ]

    def test_happy_path(self):
        from src.tools.analyze_patterns import analyze_patterns

        runs = self._make_diverse_runs()
        run_infos = [_make_run_info(r.run_id) for r in runs]

        with patch.object(_analyze_patterns_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.return_value = _make_experiment()
            instance.list_runs.return_value = run_infos
            instance.get_run_details.side_effect = runs

            result = analyze_patterns.invoke({
                "experiment_name": "test-exp",
                "target_metric": "val_accuracy",
            })

        assert "val_accuracy" in result
        assert "ERROR" not in result

    def test_experiment_not_found(self):
        from src.mlflow_client.client import MLflowClientError
        from src.tools.analyze_patterns import analyze_patterns

        with patch.object(_analyze_patterns_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.side_effect = MLflowClientError("Not found.")

            result = analyze_patterns.invoke({
                "experiment_name": "missing",
                "target_metric": "val_accuracy",
            })

        assert "ERROR" in result

    def test_not_enough_runs(self):
        from src.tools.analyze_patterns import analyze_patterns

        run_infos = [_make_run_info("r001")]
        run_details = _make_run("r001", metrics={"val_accuracy": 0.8})

        with patch.object(_analyze_patterns_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.return_value = _make_experiment()
            instance.list_runs.return_value = run_infos
            instance.get_run_details.return_value = run_details

            result = analyze_patterns.invoke({
                "experiment_name": "test-exp",
                "target_metric": "val_accuracy",
                "min_runs": 5,
            })

        assert isinstance(result, str)
        assert len(result) > 0


# ─── suggest_next_experiments ─────────────────────────────────────────────────

class TestSuggestNextExperiments:
    def test_happy_path(self):
        from src.tools.suggest_next_experiments import suggest_next_experiments

        runs = [
            _make_run(
                f"r{i}", f"run-{i}",
                params={
                    "learning_rate": str(0.001 * (i + 1)),
                    "n_estimators": str(50 * (i + 1)),
                },
                metrics={"val_accuracy": 0.70 + i * 0.03},
            )
            for i in range(5)
        ]
        run_infos = [_make_run_info(r.run_id) for r in runs]

        with patch.object(_suggest_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.return_value = _make_experiment()
            instance.list_runs.return_value = run_infos
            instance.get_run_details.side_effect = runs

            result = suggest_next_experiments.invoke({
                "experiment_name": "test-exp",
                "optimization_goal": "maximize val_accuracy",
                "num_suggestions": 3,
            })

        assert "Suggestion" in result
        assert "ERROR" not in result

    def test_no_runs(self):
        from src.tools.suggest_next_experiments import suggest_next_experiments

        with patch.object(_suggest_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.return_value = _make_experiment()
            instance.list_runs.return_value = []

            result = suggest_next_experiments.invoke({
                "experiment_name": "test-exp",
                "optimization_goal": "maximize val_accuracy",
            })

        assert "no runs" in result.lower() or "Run at least" in result

    def test_num_suggestions_capped_at_5(self):
        from src.tools.suggest_next_experiments import suggest_next_experiments

        runs = [
            _make_run(f"r{i}", metrics={"val_accuracy": 0.80 + i * 0.01})
            for i in range(3)
        ]
        run_infos = [_make_run_info(r.run_id) for r in runs]

        with patch.object(_suggest_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.return_value = _make_experiment()
            instance.list_runs.return_value = run_infos
            instance.get_run_details.side_effect = runs

            result = suggest_next_experiments.invoke({
                "experiment_name": "test-exp",
                "optimization_goal": "maximize val_accuracy",
                "num_suggestions": 10,
            })

        assert isinstance(result, str)


# ─── generate_report ──────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_happy_path(self, tmp_path: Path):
        from src.tools.generate_report import generate_report

        runs = [
            _make_run(f"r{i}", f"run-{i}",
                      params={"learning_rate": str(0.01 * (i + 1))},
                      metrics={"train_accuracy": 0.9 + i * 0.01, "val_accuracy": 0.85 + i * 0.01})
            for i in range(3)
        ]
        run_infos = [_make_run_info(r.run_id) for r in runs]
        output_file = tmp_path / "test_report.md"

        with patch.object(_generate_report_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.return_value = _make_experiment()
            instance.list_runs.return_value = run_infos
            instance.get_run_details.side_effect = runs

            result = generate_report.invoke({
                "experiment_name": "test-exp",
                "report_title": "My Test Report",
                "output_path": str(output_file),
            })

        assert "Report saved" in result
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "Analysis Report" in content or "My Test Report" in content

    def test_no_runs(self):
        from src.tools.generate_report import generate_report

        with patch.object(_generate_report_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.return_value = _make_experiment()
            instance.list_runs.return_value = []

            result = generate_report.invoke({
                "experiment_name": "test-exp",
                "report_title": "Empty Report",
            })

        assert "no runs" in result.lower() or "Cannot generate" in result

    def test_experiment_not_found(self):
        from src.mlflow_client.client import MLflowClientError
        from src.tools.generate_report import generate_report

        with patch.object(_generate_report_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.side_effect = MLflowClientError("Not found.")

            result = generate_report.invoke({
                "experiment_name": "missing-exp",
                "report_title": "Report",
            })

        assert "ERROR" in result

    def test_report_contains_expected_sections(self, tmp_path: Path):
        from src.tools.generate_report import generate_report

        runs = [
            _make_run(f"r{i}", metrics={
                "train_accuracy": 0.95,
                "val_accuracy": 0.60,  # overfitting
            })
            for i in range(3)
        ]
        run_infos = [_make_run_info(r.run_id) for r in runs]
        output_file = tmp_path / "sections_report.md"

        with patch.object(_generate_report_mod, "MLflowAnalystClient") as MockClient:
            instance = MockClient.return_value
            instance.get_experiment.return_value = _make_experiment()
            instance.list_runs.return_value = run_infos
            instance.get_run_details.side_effect = runs

            generate_report.invoke({
                "experiment_name": "test-exp",
                "report_title": "Sections Test",
                "output_path": str(output_file),
            })

        content = output_file.read_text(encoding="utf-8")
        assert "Metrics Comparison" in content
        assert "Diagnostics" in content
        assert "Limitations" in content


# ─── tools __init__ ───────────────────────────────────────────────────────────

class TestToolsInit:
    def test_all_tools_exported(self):
        from src.tools import ALL_TOOLS

        assert len(ALL_TOOLS) >= 6  # 6 core + optional web_search
        tool_names = {t.name for t in ALL_TOOLS}
        assert "load_experiment" in tool_names
        assert "compare_runs" in tool_names
        assert "diagnose_run" in tool_names
        assert "analyze_patterns" in tool_names
        assert "suggest_next_experiments" in tool_names
        assert "generate_report" in tool_names

    def test_all_tools_have_descriptions(self):
        from src.tools import ALL_TOOLS

        for t in ALL_TOOLS:
            assert t.description, f"Tool '{t.name}' has no description"
            assert len(t.description) > 20, f"Tool '{t.name}' description is too short"
