# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-22

### Added

- Executive README with architecture diagram, configuration table, and tech stack
- CHANGELOG.md following Keep a Changelog format
- Usage guide (`docs/usage-guide.md`) covering demo, custom queries, HITL, and dashboard
- Troubleshooting guide (`docs/troubleshooting.md`) with common errors and solutions
- `web_search` tool section in tools reference documentation
- Version badge in README

### Changed

- README rewritten as portfolio landing page with highlights, quick start, and documentation index
- Tools reference updated from 6 to 7 tools (added `web_search` section)
- Example prompts in tools reference translated from Portuguese to English
- Version bumped to 1.0.0

## [0.5.0] - 2026-03-22

### Added

- Unit tests for report generator (`test_report_generator.py`, 15 tests)
- Unit tests for LangSmith helpers (`test_langsmith.py`, 6 tests)
- Edge case tests for tools (`test_edge_tools.py`, 8 tests)
- Edge case tests for analysis modules (`test_edge_analysis.py`, 6 tests)
- Integration tests against real MLflow (`test_agent_e2e.py`, 5 tests)
- Integration test fixtures (`conftest.py` with `seeded_experiment`)
- 75% coverage threshold enforced in CI

### Changed

- Total test count increased from 98 to 124
- Code coverage increased to 83.68%
- CI workflow updated to run edge case tests and enforce coverage threshold

## [0.4.0] - 2026-03-22

### Added

- `GovernanceCallbackHandler` for structured JSONL tracing of every agent action
- Token budget enforcement (`AGENT_MAX_TOKENS`)
- Execution timeout via `ThreadPoolExecutor` (Windows-compatible)
- Consecutive failure threshold (`AGENT_MAX_FAILURES`)
- `invoke_with_governance()` wrapper in agent builder
- Human-in-the-Loop (HITL) support via `interrupt_on` parameter
- HITL demo script (`scripts/run_demo_hitl.py`)
- Streamlit governance dashboard with Run Explorer and Tool Analytics pages
- JSONL log reader utility (`src/dashboard/log_reader.py`)
- Governance documentation (`docs/governance.md`)
- HITL documentation (`docs/hitl.md`)

### Changed

- Agent builder updated to support HITL tools and governance callbacks
- Agent config extended with governance fields
- Demo script updated to use `invoke_with_governance()`
- Observability docs updated with JSONL tracing and dashboard sections

## [0.3.0] - 2026-03-17

### Added

- Agent builder (`src/agent/builder.py`) with `create_analyst_agent()` factory
- Agent configuration via environment variables (`src/agent/config.py`)
- Engineered system prompt (`src/agent/prompts/system_prompt.md`)
- Demo script (`scripts/run_demo.py`) with 3 demonstration queries
- LangSmith observability helpers (`src/observability/langsmith.py`)
- Architecture decision records (`docs/architecture-decisions.md`)
- Setup guide (`docs/SETUP.md`)

### Changed

- Project structure reorganized with `agent/` and `observability/` modules

## [0.2.0] - 2026-03-16

### Added

- `load_experiment` tool — load experiment metadata and runs from MLflow
- `compare_runs` tool — side-by-side metric comparison table
- `diagnose_run` tool — overfitting detection with severity levels
- `analyze_patterns` tool — hyperparameter-metric correlation analysis
- `suggest_next_experiments` tool — generate next-run configurations
- `generate_report` tool — full Markdown report saved to disk
- `web_search` tool — optional Tavily-powered web search
- Tools reference documentation (`docs/tools-reference.md`)
- Markdown report generator (`src/report/generator.py`)
- Analysis modules: metrics, overfitting, patterns, suggestions

## [0.1.0] - 2026-03-15

### Added

- Project scaffolding with `pyproject.toml` and development dependencies
- MLflow client wrapper (`src/mlflow_client/client.py`) with typed models
- Docker Compose stack for MLflow + PostgreSQL + MinIO
- Demo experiment seeder (`scripts/seed_mlflow.py`)
- CI pipeline with ruff, mypy, and pytest
- Contributing guidelines and branch naming conventions
- MIT license
