# Contributing

## Branch naming

| Type | Pattern | Example |
|---|---|---|
| Phase work | `phase/{number}-{slug}` | `phase/2-custom-tools` |
| Bug fix | `fix/{slug}` | `fix/mlflow-connection-timeout` |
| Hotfix on main | `hotfix/{slug}` | `hotfix/missing-env-var` |

Always branch off `main`.

## Commit conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

| Type | When to use |
|---|---|
| `feat` | New feature or tool |
| `fix` | Bug fix |
| `chore` | Maintenance (deps, config, scripts) |
| `docs` | Documentation only |
| `test` | Tests only |
| `refactor` | Code restructure without behavior change |
| `ci` | CI/CD pipeline changes |

**Scope** maps to the project module: `agent`, `tools`, `mlflow`, `analysis`, `report`, `observability`, `dashboard`, `repo`.

**Examples:**

```
feat(tools): implement load_experiment with MLflow client
fix(agent): handle FilesystemBackend path when workspace missing
docs(readme): add architecture diagram and demo GIF
test(analysis): add edge cases for empty run list in overfitting detection
chore(deps): upgrade langchain to 0.3.x
ci: add coverage threshold to pytest job
```

## Pull Request workflow

1. Open a draft PR early — not only when finished.
2. Link the related issue(s) in the PR description with `Closes #N`.
3. Every PR must have the milestone set.
4. PR titles follow the same Conventional Commits format.
5. All CI checks must pass before requesting review.
6. Use **Squash and merge** — one clean commit per PR on `main`.

**PR title format:**

```
[Phase N] <short description of what was done>
```

Example: `[Phase 2] Custom tools: 5 MLflow analysis tools implemented`

## Definition of done

A PR is ready to merge when:

- [ ] All CI jobs pass (lint, typecheck, test)
- [ ] All issue tasks checked off
- [ ] New code has unit tests
- [ ] No hardcoded credentials or absolute paths
- [ ] `docs/` updated if the change affects user-facing behavior

## Development setup

```bash
# 1. Clone and create virtual environment
git clone https://github.com/<user>/ml-experiment-analyst-agent
cd ml-experiment-analyst-agent
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Fill in your API keys in .env

# 4. Start MLflow (Docker required)
docker compose up -d

# 5. Seed demo experiments
python scripts/seed_mlflow.py

# 6. Run tests
pytest tests/unit/ -v

# 7. Run linter
ruff check src/ tests/
```

## Code style

- **Formatter:** `ruff format` (configured in `pyproject.toml`)
- **Linter:** `ruff check`
- **Type checker:** `mypy`
- **Python:** 3.11+
- All public functions must have type hints and a one-line docstring minimum.
- No bare `except:` — always catch specific exceptions.
