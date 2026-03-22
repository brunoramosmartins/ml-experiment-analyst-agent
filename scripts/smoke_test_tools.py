#!/usr/bin/env python3
"""
Smoke test para as 6 ferramentas da Phase 2 contra o MLflow real.

Pré-requisitos:
  - docker compose up -d  (MLflow rodando em http://localhost:5000)
  - python scripts/seed_mlflow.py  (dados demo carregados)

Uso:
  $env:PYTHONIOENCODING="utf-8"
  python scripts/smoke_test_tools.py
"""

import os
import sys

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

# ── Cor no terminal ──────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0


def section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


def ok(label: str, detail: str = "") -> None:
    global passed
    passed += 1
    suffix = f"  {YELLOW}({detail}){RESET}" if detail else ""
    print(f"  {GREEN}✓{RESET} {label}{suffix}")


def fail(label: str, error: str) -> None:
    global failed
    failed += 1
    print(f"  {RED}✗ {label}{RESET}")
    print(f"    {RED}{error}{RESET}")


def preview(text: str, lines: int = 6) -> None:
    for line in text.strip().splitlines()[:lines]:
        print(f"    {line}")
    total = len(text.strip().splitlines())
    if total > lines:
        print(f"    ... (+{total - lines} lines)")


# ── Imports das ferramentas ──────────────────────────────────────────────────
try:
    import mlflow

    from src.tools import ALL_TOOLS
    from src.tools.analyze_patterns import analyze_patterns
    from src.tools.compare_runs import compare_runs
    from src.tools.diagnose_run import diagnose_run
    from src.tools.generate_report import generate_report
    from src.tools.load_experiment import load_experiment
    from src.tools.suggest_next_experiments import suggest_next_experiments

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
except Exception as e:
    print(f"{RED}ERRO ao importar ferramentas: {e}{RESET}")
    sys.exit(1)

# ── Conectividade ────────────────────────────────────────────────────────────
section("0. Conectividade com MLflow")
try:
    experiments = mlflow.search_experiments()
    exp_names = [e.name for e in experiments if e.name != "Default"]
    ok("MLflow acessível", f"{len(exp_names)} experimentos encontrados")
    for name in exp_names:
        print(f"       - {name}")
except Exception as e:
    fail("Conexão com MLflow", str(e))
    print(f"\n{RED}Sem conexão com MLflow. Rode: docker compose up -d{RESET}")
    sys.exit(1)

# ── Coleta IDs reais para os testes ─────────────────────────────────────────
try:
    runs_binary = mlflow.search_runs(experiment_names=["binary-classification"], max_results=3)
    run_ids = runs_binary["run_id"].tolist()
    best_run_id = run_ids[0]
except Exception as e:
    print(f"{RED}Não foi possível carregar runs: {e}{RESET}")
    sys.exit(1)

# ── Teste 1: load_experiment ─────────────────────────────────────────────────
section("1. load_experiment")
try:
    result = load_experiment.invoke({"experiment_name": "binary-classification"})
    assert "binary-classification" in result
    assert "runs" in result.lower() or "run" in result.lower()
    ok("Carregou binary-classification")
    preview(result)
except Exception as e:
    fail("load_experiment", str(e))

try:
    result = load_experiment.invoke({"experiment_name": "experimento-inexistente-xyz"})
    lower = result.lower()
    assert "not found" in lower or "não encontrado" in lower or "no experiment" in lower
    ok("Experimento inexistente retorna mensagem de erro amigável")
except Exception as e:
    fail("load_experiment (not found)", str(e))

# ── Teste 2: compare_runs ────────────────────────────────────────────────────
section("2. compare_runs")
try:
    result = compare_runs.invoke({"run_ids": run_ids[:3]})
    assert "run" in result.lower()
    ok("Comparou 3 runs", f"IDs: {run_ids[:3]}")
    preview(result)
except Exception as e:
    fail("compare_runs (3 runs)", str(e))

try:
    result = compare_runs.invoke({"run_ids": []})
    assert len(result) > 0
    ok("Lista vazia retorna mensagem")
except Exception as e:
    fail("compare_runs (empty)", str(e))

# ── Teste 3: diagnose_run ────────────────────────────────────────────────────
section("3. diagnose_run")
try:
    result = diagnose_run.invoke({"run_id": best_run_id})
    assert len(result) > 0
    ok(f"Diagnosticou run {best_run_id[:8]}...")
    preview(result)
except Exception as e:
    fail("diagnose_run", str(e))

# Testar com run do experimento de overfitting (deve detectar problemas)
try:
    overfit_runs = mlflow.search_runs(experiment_names=["overfit-test"], max_results=1)
    if not overfit_runs.empty:
        overfit_run_id = overfit_runs["run_id"].iloc[0]
        result = diagnose_run.invoke({"run_id": overfit_run_id})
        ok("Diagnosticou run do overfit-test")
        preview(result)
except Exception as e:
    fail("diagnose_run (overfit)", str(e))

try:
    result = diagnose_run.invoke({"run_id": "run-id-invalido-000"})
    assert len(result) > 0
    ok("Run inválido retorna mensagem")
except Exception as e:
    fail("diagnose_run (invalid id)", str(e))

# ── Teste 4: analyze_patterns ────────────────────────────────────────────────
section("4. analyze_patterns")
try:
    result = analyze_patterns.invoke({
        "experiment_name": "binary-classification",
        "target_metric": "val_accuracy",
    })
    assert len(result) > 0
    ok("Analisou padrões em binary-classification (val_accuracy)")
    preview(result)
except Exception as e:
    fail("analyze_patterns", str(e))

try:
    result = analyze_patterns.invoke({
        "experiment_name": "regression-v2",
        "target_metric": "val_r2",
    })
    assert len(result) > 0
    ok("Analisou padrões em regression-v2 (val_r2)")
    preview(result)
except Exception as e:
    fail("analyze_patterns (regression)", str(e))

# ── Teste 5: suggest_next_experiments ───────────────────────────────────────
section("5. suggest_next_experiments")
try:
    result = suggest_next_experiments.invoke({
        "experiment_name": "binary-classification",
        "optimization_goal": "maximize val_accuracy",
        "num_suggestions": 3,
    })
    assert len(result) > 0
    ok("Gerou 3 sugestões para binary-classification")
    preview(result)
except Exception as e:
    fail("suggest_next_experiments", str(e))

# ── Teste 6: generate_report ─────────────────────────────────────────────────
section("6. generate_report")
try:
    result = generate_report.invoke({
        "experiment_name": "binary-classification",
        "report_title": "Smoke Test Report — Binary Classification",
        "include_suggestions": True,
    })
    assert len(result) > 0
    ok("Gerou relatório de binary-classification")
    preview(result)

    # Verificar se o arquivo foi salvo
    import re
    path_match = re.search(r"[\w\\/.-]+\.md", result)
    if path_match:
        report_path = path_match.group(0).strip("`")
        if os.path.exists(report_path):
            size = os.path.getsize(report_path)
            ok("Arquivo salvo em disco", f"{report_path} ({size} bytes)")
        else:
            fail("Arquivo salvo em disco", f"Path '{report_path}' não existe")
except Exception as e:
    fail("generate_report", str(e))

# ── Teste 7: ALL_TOOLS exportado corretamente ────────────────────────────────
section("7. Registro das ferramentas no agente")
try:
    assert len(ALL_TOOLS) == 6, f"Esperado 6 ferramentas, encontrado {len(ALL_TOOLS)}"
    ok(f"ALL_TOOLS exporta {len(ALL_TOOLS)} ferramentas")
    for t in ALL_TOOLS:
        ok(f"  {t.name}", t.description[:60] + "...")
except Exception as e:
    fail("ALL_TOOLS", str(e))

# ── Resultado final ───────────────────────────────────────────────────────────
section("RESULTADO")
total = passed + failed
color = GREEN if failed == 0 else RED
print(f"\n  {color}{BOLD}{passed}/{total} checks passaram{RESET}")
if failed > 0:
    print(f"  {RED}{failed} falha(s) — veja os detalhes acima{RESET}")
else:
    print(f"\n  {GREEN}{BOLD}Phase 2 validada com sucesso contra MLflow real!{RESET}")
    print("  Pronto para abrir o PR e criar a tag v0.2-tools.")

print()
sys.exit(0 if failed == 0 else 1)
