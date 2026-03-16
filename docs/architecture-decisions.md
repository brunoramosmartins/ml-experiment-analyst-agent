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

## ADR-002: Default LLM — claude-sonnet-4-5

**Status:** Accepted

### Context

The agent performs analytical reasoning over structured data (MLflow metrics, parameters, run comparisons). The model must: follow multi-step instructions reliably, produce structured text output (Markdown reports, ranked lists), and handle tool calling accurately.

### Alternatives Considered

| Model | Notes |
|---|---|
| **claude-sonnet-4-5** | Anthropic's production Sonnet. Strong analytical reasoning, native tool use, good instruction following. Default in deepagents. |
| **gpt-4o** | Strong performance, but adds OpenAI dependency. The stack is already Anthropic-centric (deepagents default). |
| **gemini-pro** | Google ecosystem. Less tooling integration with LangChain/deepagents out of the box. |
| **claude-haiku-4-5** | Faster and cheaper, but lower reasoning quality for complex analysis tasks. |

### Decision Criteria

1. Quality of multi-step analytical reasoning
2. Accuracy on structured output generation (Markdown tables, JSON-like lists)
3. Reliability of tool call sequencing
4. Cost per run (budget matters for a portfolio project)
5. Native compatibility with deepagents

### Decision

**claude-sonnet-4-5** (`claude-sonnet-4-5-20250929`) — best balance of reasoning quality and cost for this use case. It is also the deepagents default, which minimizes configuration friction.

### Consequences

- **Positive:** No additional LLM provider setup beyond `ANTHROPIC_API_KEY`.
- **Positive:** Consistent with deepagents documentation and examples.
- **Consideration:** Token costs accumulate during development — use LangSmith to monitor usage per run.

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
