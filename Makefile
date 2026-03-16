.PHONY: install test lint typecheck format check seed-mlflow run-demo run-dashboard mlflow-up mlflow-down

# ─── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ─── Quality ──────────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

check: lint typecheck

# ─── Tests ────────────────────────────────────────────────────────────────────
test:
	pytest tests/unit/ --cov=src --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v

test-all:
	pytest tests/ --cov=src --cov-report=term-missing

# ─── Infrastructure ───────────────────────────────────────────────────────────
mlflow-up:
	docker compose up -d
	@echo "Waiting for MLflow to be ready..."
	@sleep 10
	@echo "MLflow UI: http://localhost:5000"
	@echo "MinIO console: http://localhost:9001"

mlflow-down:
	docker compose down

# ─── Data ─────────────────────────────────────────────────────────────────────
seed-mlflow:
	python scripts/seed_mlflow.py

# ─── Agent ────────────────────────────────────────────────────────────────────
run-demo:
	python scripts/run_demo.py

run-dashboard:
	streamlit run src/dashboard/app.py
