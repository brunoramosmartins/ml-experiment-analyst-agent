# Architecture Decision Records

> **Format:** Each ADR follows the structure: Context → Alternatives Considered → Decision Criteria → Decision → Consequences.

---

## ADR-001: Agent Framework — deepagents

**Status:** Accepted

### Context

The agent needs a framework that provides: tool calling, planning (todo list), filesystem access for report persistence, streaming output, and Human-in-the-Loop (HITL) support. The choice of framework directly impacts how much custom infrastructure needs to be built vs. leveraged.

### Alternatives Considered

| Framework | Notes |
|---|---|
| **CrewAI** | Good multi-agent support, but tightly opinionated on agent roles. Less composable for a single specialist agent. |
| **AutoGen** | Microsoft ecosystem, good for multi-agent conversations. Heavier setup for a single-agent tool-calling pattern. |
| **LangGraph (pure)** | Maximum flexibility, but requires building the planning loop, todo list, filesystem backend, and HITL from scratch. |
| **Agno** | Newer framework, less battle-tested. Ecosystem smaller. |
| **deepagents** | Built on top of LangGraph. Provides FilesystemBackend, write_todos/read_todos, HITL via interrupt_on, and streaming out of the box. |

### Decision Criteria

1. Native support for `FilesystemBackend` (agent writes reports to disk)
2. Built-in planning tools (`write_todos`, `read_todos`) without custom implementation
3. Native HITL via `interrupt_on` parameter
4. Full LangGraph compatibility (streaming, memory, checkpointing)
5. Anthropic Claude as first-class LLM option

### Decision

**deepagents** — it is the only framework that satisfies all five criteria without requiring custom infrastructure. Being built on LangGraph means the agent graph is portable to pure LangGraph if needed.

### Consequences

- **Positive:** Faster development of the planning and filesystem layers.
- **Positive:** LangSmith tracing works automatically via LangGraph.
- **Risk:** deepagents is newer than LangGraph core — breaking API changes are possible. Mitigated by pinning the version in `pyproject.toml`.

---

## ADR-002: Default LLM — Llama 3 via Ollama (local)

**Status:** Accepted — supersedes initial proposal of claude-sonnet-4-5

### Context

The agent performs analytical reasoning over structured data (MLflow metrics, parameters, run comparisons). The model must: follow multi-step instructions reliably, produce structured text output (Markdown reports, ranked lists), and handle tool calling accurately.

The initial proposal was to use `claude-sonnet-4-5` (deepagents default). However, **development and iteration velocity matter more than output quality at this stage** — running hundreds of test queries during Phases 1–3 against a paid API would accumulate significant cost. A local model enables unlimited free iteration before committing to a production LLM.

The architecture is designed to be **LLM-agnostic**: because deepagents is built on LangChain/LangGraph, swapping the model requires changing a single parameter in `src/agent/config.py` with no impact on tools, analysis logic, or report generation.

### Alternatives Considered

| Model | Provider | Cost | Tool Calling | Notes |
|---|---|---|---|---|
| **Llama 3.1 8B / 70B** | Local (Ollama) | Free | ✅ via Ollama | Good tool calling support from 3.1+. Runs on consumer hardware (8B) or a capable workstation (70B). |
| **Llama 3.2 3B** | Local (Ollama) | Free | ✅ | Lighter, faster, weaker reasoning on complex analysis. |
| **Mistral 7B / Mixtral** | Local (Ollama) | Free | ⚠️ Partial | Tool calling less reliable than Llama 3.1. |
| **claude-sonnet-4-5** | Anthropic API | Paid | ✅ Native | Best quality, but cost accumulates during iterative development. Reserved for Phase 8 (LLM evaluation). |
| **gpt-4o** | OpenAI API | Paid | ✅ Native | Strong, but adds a second paid dependency with no clear advantage over Claude for this use case. |
| **gemini-flash** | Google API | Free tier | ✅ | Free tier available, but LangChain integration is less mature than Anthropic's. |

### Decision Criteria

1. Zero cost during iterative development (Phases 1–5)
2. Sufficient tool calling reliability for multi-step agent workflows
3. Easy swap path to a production LLM without touching tools or analysis logic
4. Runs locally without internet dependency

### Decision

**Llama 3.1 8B** via **Ollama** (`langchain-ollama`) for development and local testing.

The model is configured in `src/agent/config.py` via the `LLM_PROVIDER` and `LLM_MODEL` environment variables, making the swap to any LangChain-compatible model a one-line config change.

**Planned Phase 8 — LLM Evaluation:** Run the same benchmark queries against Llama 3.1 8B, Llama 3.1 70B, and claude-sonnet-4-5. Compare: report quality, tool call accuracy, overfitting detection rate, and latency. Choose the production LLM based on data.

### Ollama setup

```bash
# Install Ollama: https://ollama.com/
ollama pull llama3.1:8b

# Verify
ollama run llama3.1:8b "What is overfitting in machine learning?"
```

Add to `.env`:
```
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

### Consequences

- **Positive:** Zero API cost during development — unlimited test runs.
- **Positive:** No internet dependency for the agent during local development.
- **Positive:** LLM swap is a single env var change — architecture is not coupled to any specific provider.
- **Trade-off:** Llama 3.1 8B has lower reasoning quality than claude-sonnet-4-5 on complex analysis. The 70B variant narrows this gap significantly but requires more hardware.
- **Trade-off:** Ollama must be running locally — adds one more service to the development stack.
- **Risk:** Tool calling reliability may be lower than with native Anthropic tool use. Mitigated by iterating on the system prompt (Phase 3) and by the planned LLM evaluation phase.

---

## ADR-003: Agent Backend — FilesystemBackend

**Status:** Accepted

### Context

The agent needs to persist generated reports between tool calls and make them available to the user after execution. The backend determines how the agent reads and writes files during its execution lifecycle.

### Alternatives Considered

| Backend | Notes |
|---|---|
| **StateBackend (ephemeral)** | Files exist only in memory during the run. No persistence after execution. Not suitable for report generation. |
| **FilesystemBackend** | Files are written to a local directory. Reports persist after execution. Simple and reproducible. |
| **StoreBackend** | Database-backed. More complex setup, overkill for a portfolio project where reproducibility is the goal. |

### Decision Criteria

1. Reports must persist after agent execution ends
2. Users must be able to browse generated reports via filesystem
3. Setup must be reproducible with no external dependencies
4. Backend root must be configurable via environment variable

### Decision

**FilesystemBackend** with root at `data/agent-workspace/`. Reports are written to `data/agent-workspace/reports/{timestamp}_{experiment}.md`.

### Consequences

- **Positive:** Zero infrastructure dependencies beyond local disk.
- **Positive:** Generated reports are browsable as plain Markdown files.
- **Consideration:** `data/agent-workspace/` is gitignored — reports are not committed to the repository.

---

## ADR-004: Observability Strategy — LangSmith

**Status:** Accepted

### Context

Agent observability requires capturing: which tools were called, in what order, with which inputs/outputs, total token usage, and latency per step. This is essential for debugging prompt failures and demonstrating system transparency in the portfolio.

### Alternatives Considered

| Approach | Notes |
|---|---|
| **Custom logging only** | Flexible, but requires building the tracing infrastructure from scratch. High effort, low standardization. |
| **Langfuse** | Open-source LangSmith alternative. Good UI, but requires self-hosting or a separate account. Adds complexity for a solo project. |
| **No observability** | Acceptable for simple scripts, not for an agent with complex tool-calling patterns. Debugging becomes very difficult. |
| **LangSmith** | Native integration with LangGraph/deepagents. Zero-config tracing — just set `LANGCHAIN_TRACING_V2=true`. Managed hosted service. |

### Decision Criteria

1. Zero-config integration with LangGraph
2. Visibility into every tool call (input, output, duration, tokens)
3. No self-hosting required
4. Ability to add custom metadata to traces

### Decision

**LangSmith** as the primary observability layer, complemented by a **GovernanceMiddleware** (Phase 4) that produces local JSONL logs for the Streamlit dashboard.

### Consequences

- **Positive:** Complete traces available from day one with minimal setup.
- **Positive:** LangSmith screenshots serve as portfolio evidence of observability maturity.
- **Consideration:** Requires `LANGCHAIN_API_KEY`. Free tier has usage limits — sufficient for development.

---

## ADR-005: Report Output Format — Markdown

**Status:** Accepted

### Context

The agent's primary deliverable is an analysis report. The format must be: readable by humans without tooling, portable (no proprietary formats), and easy to generate programmatically from a template.

### Alternatives Considered

| Format | Notes |
|---|---|
| **JSON structured** | Machine-readable, but not human-friendly without a viewer. Bad for portfolio demos. |
| **Markdown** | Human-readable in any text editor, renders natively on GitHub and VS Code. Easy to generate via string templates. |
| **HTML** | Richer formatting, but requires a browser and is harder to generate without a templating engine. |
| **PDF** | Professional look, but requires heavy dependencies (weasyprint, reportlab) and is not diff-friendly. |

### Decision Criteria

1. Readable without special tooling
2. Renders natively in GitHub (for portfolio visibility)
3. Diff-friendly (text-based)
4. Easy to generate from LLM output
5. Portable — shareable as a single file

### Decision

**Markdown** (`.md`) saved via `FilesystemBackend`. Reports follow a fixed structure: header, metrics comparison, diagnostics, patterns, recommendations, disclaimer.

### Consequences

- **Positive:** Reports are readable in any environment — terminal, VS Code, GitHub.
- **Positive:** The agent can generate Markdown naturally without post-processing.
- **Consideration:** No charts or visualizations in reports (v1.0 scope). Rich visuals are delegated to the Streamlit dashboard (Phase 4).
