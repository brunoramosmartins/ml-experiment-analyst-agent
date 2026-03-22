# Prompt Engineering Log

This document tracks iterations on the ML Experiment Analyst Agent's system prompt (`src/agent/prompts/system_prompt.md`). Each version records what changed, why, and observations from testing.

---

## v1.0 — Initial Engineered Prompt

**Date:** 2026-03-22
**File:** `src/agent/prompts/system_prompt.md`

### Summary

Replaced the Phase 1 placeholder with a full system prompt covering:
- Role definition and domain context
- Tool table (6 core tools + optional web_search)
- 6-step standard analysis workflow with per-step instructions
- Analysis principles (evidence-based, uncertainty-aware)
- Stopping criteria (success, experiment not found, tool failure with 3 retries)
- Communication format (brief updates, report via tool, structured error messages)

### Design Rationale

1. **Additive to deepagents.** The deepagents framework injects its own system prompt covering tool calling mechanics, filesystem backend, and todo list usage. Our prompt focuses on domain knowledge and workflow — no duplication.

2. **Explicit workflow ordering.** The 6-step sequence (load -> compare -> diagnose -> patterns -> suggest -> report) prevents the agent from skipping steps or calling tools out of order.

3. **Conditional step 5.** `suggest_next_experiments` is only called when the user explicitly requests suggestions. This avoids unnecessary hallucination on recommendation tasks.

4. **Stopping criteria.** Three clear exit conditions prevent infinite loops: report generated, experiment not found, or 3 failed retries per tool.

5. **Communication format.** Brief progress updates keep the user informed without flooding the chat. Final report is always delivered via `generate_report` to ensure persistence.

### Observations

- *To be filled after testing with demo queries.*

---

## v2.0 — (Planned)

**Date:** TBD

### Changes

- *To be filled based on observations from v1.0 testing.*

### Observations

- *To be filled after testing.*

---

## v3.0 — (Planned)

**Date:** TBD

### Changes

- *To be filled based on observations from v2.0 testing.*

### Observations

- *To be filled after testing.*
