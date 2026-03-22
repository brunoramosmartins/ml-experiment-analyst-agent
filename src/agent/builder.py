"""Agent factory for the ML Experiment Analyst Agent.

Deepagents API assumed:
    create_deep_agent(model, tools, system_prompt, backend) -> CompiledStateGraph
    FilesystemBackend(root: Path)

The agent is LLM-agnostic: set LLM_PROVIDER=ollama (default) or LLM_PROVIDER=anthropic
in your .env to switch models without touching this file.
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agent.config import AgentConfig

logger = logging.getLogger(__name__)


def _build_llm(config: AgentConfig) -> BaseChatModel:
    """Instantiate the LLM based on the configured provider."""
    if config.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.llm_model,
            base_url=config.ollama_base_url,
            temperature=config.temperature,
        )
    if config.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(  # type: ignore[call-arg]
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    if config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    raise ValueError(
        f"Unsupported LLM provider: {config.llm_provider!r}. "
        "Supported values: 'ollama', 'anthropic', 'openai'."
    )


def _load_system_prompt() -> str:
    """Load the system prompt from the markdown file."""
    prompt_path = Path(__file__).parent / "prompts" / "system_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return "You are an ML Experiment Analyst Agent."


def create_analyst_agent(
    config: AgentConfig | None = None,
    *,
    hitl_tools: list[str] | None = None,
) -> Any:
    """Create and return the ML Experiment Analyst Agent.

    The agent is built on deepagents (LangGraph). It uses a FilesystemBackend
    to persist generated reports to data/agent-workspace/.

    Args:
        config: Agent configuration. Uses environment variable defaults if None.
        hitl_tools: Tool names that require human approval before execution.
            Example: ``["generate_report"]`` pauses before generating reports.

    Returns:
        CompiledStateGraph ready for invocation via agent.invoke({...}).

    Example:
        agent = create_analyst_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": "hello"}]})
    """
    from deepagents import create_deep_agent  # type: ignore[import]
    from deepagents.backends import FilesystemBackend  # type: ignore[import]
    from dotenv import load_dotenv

    from src.tools import ALL_TOOLS

    load_dotenv()

    if config is None:
        config = AgentConfig()

    config.workspace_path.mkdir(parents=True, exist_ok=True)

    llm = _build_llm(config)
    system_prompt = _load_system_prompt()
    backend = FilesystemBackend(root_dir=config.workspace_path)

    interrupt_on = {name: True for name in hitl_tools} if hitl_tools else None

    agent = create_deep_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=system_prompt,
        backend=backend,
        interrupt_on=interrupt_on,  # type: ignore[arg-type]
    )

    logger.info(
        "Agent created — provider=%s, model=%s, tools=%d, workspace=%s, hitl=%s",
        config.llm_provider,
        config.llm_model,
        len(ALL_TOOLS),
        config.workspace_path,
        hitl_tools or "none",
    )

    return agent


def invoke_with_governance(
    agent: Any,
    message: str,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """Invoke the agent with governance callbacks and execution timeout.

    Attaches a :class:`GovernanceCallbackHandler` that logs structured JSONL
    events and enforces token/failure limits. The invocation is wrapped in a
    thread-based timeout (Windows-compatible).

    Args:
        agent: A compiled agent from :func:`create_analyst_agent`.
        message: The user message to send to the agent.
        config: Agent configuration (for governance limits). Defaults apply if None.

    Returns:
        The raw result dict from ``agent.invoke()``.

    Raises:
        GovernanceLimitError: If token budget or failure limit is exceeded.
        TimeoutError: If execution exceeds ``config.execution_timeout_seconds``.
    """
    from src.observability.governance import GovernanceCallbackHandler

    if config is None:
        config = AgentConfig()

    run_id = uuid.uuid4().hex[:12]
    handler = GovernanceCallbackHandler(
        run_id=run_id,
        log_dir=config.trace_log_dir,
        max_tokens=config.max_tokens_per_execution,
        max_consecutive_failures=config.max_consecutive_failures,
    )

    invoke_input = {"messages": [{"role": "user", "content": message}]}
    invoke_config: dict[str, Any] = {
        "callbacks": [handler],
        "run_name": f"analyst-agent-{run_id}",
        "metadata": {"run_id": run_id, "query_preview": message[:100]},
    }

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent.invoke, invoke_input, invoke_config)
        try:
            result = future.result(timeout=config.execution_timeout_seconds)
        except FuturesTimeout:
            logger.error(
                "Execution timed out after %ds (run_id=%s)",
                config.execution_timeout_seconds,
                run_id,
            )
            raise TimeoutError(
                f"Agent execution timed out after {config.execution_timeout_seconds}s"
            ) from None

    logger.info(
        "Run completed — run_id=%s, tokens=%d, log=%s",
        run_id,
        handler.total_tokens,
        handler._log_file,  # noqa: SLF001
    )

    return result
