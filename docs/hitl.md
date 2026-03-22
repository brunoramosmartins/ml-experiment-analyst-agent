# Human-in-the-Loop (HITL)

Human-in-the-loop allows the agent to pause before executing specific tools,
giving the user a chance to approve, edit, or reject the action. This is
essential for production safety: the agent proposes, the human disposes.

---

## How it works

The HITL mechanism uses deepagents' `interrupt_on` parameter. When the agent
attempts to call a tool listed in `interrupt_on`, execution pauses and returns
the pending tool call to the caller.

```
User query
    |
    v
Agent reasoning → tool_1() → tool_2() → [generate_report] → PAUSE
                                                               |
                                                     User approves/rejects
                                                               |
                                                          Resume or abort
```

---

## Configuration

Pass `hitl_tools` when creating the agent:

```python
from src.agent.builder import create_analyst_agent

# Pause before generating reports
agent = create_analyst_agent(hitl_tools=["generate_report"])
```

You can protect any tool:

```python
# Pause before report generation AND web search
agent = create_analyst_agent(hitl_tools=["generate_report", "web_search"])
```

---

## Demo script

Run the interactive HITL demo:

```bash
python scripts/run_demo_hitl.py
```

The script:
1. Creates an agent with HITL on `generate_report`
2. Sends a query that triggers full analysis + report
3. When the agent tries to generate the report, it pauses
4. Displays the pending report parameters (title, experiment, etc.)
5. Asks for user input: **approve**, **edit** (change title), or **reject**
6. Resumes or aborts based on the user's decision

---

## How to handle the interrupt in custom code

```python
# First invocation — agent works until it hits an interrupt
result = agent.invoke({"messages": [{"role": "user", "content": query}]})

# Check for pending tool calls
messages = result.get("messages", [])
last_msg = messages[-1]
pending = getattr(last_msg, "tool_calls", [])

if pending:
    # Show pending action to user, get approval
    print(f"Agent wants to call: {pending[0]['name']}")
    print(f"With args: {pending[0]['args']}")

    if user_approves():
        # Resume — agent continues from where it paused
        final = agent.invoke(None)
    else:
        print("Rejected by user.")
```

---

## Design rationale

- **Why `generate_report`?** It's the most impactful tool — it writes files
  to disk. All other tools are read-only queries against MLflow.
- **Why not all tools?** Pausing on every tool call would make the agent
  unusable for analysis workflows that involve 5-10 sequential tool calls.
- **Extensibility:** Any tool can be added to `hitl_tools` as needed. For
  example, `web_search` could require approval in sensitive environments.
