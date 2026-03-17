"""Agent factory for the ML Experiment Analyst Agent.

Deepagents API assumed:
    create_deep_agent(model, tools, system_prompt, backend) -> CompiledStateGraph
    FilesystemBackend(root: Path)

The agent is LLM-agnostic: set LLM_PROVIDER=ollama (default) or LLM_PROVIDER=anthropic
in your .env to switch models without touching this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agent.config import AgentConfig


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
    raise ValueError(
        f"Unsupported LLM provider: {config.llm_provider!r}. "
        "Supported values: 'ollama', 'anthropic'."
    )


def _load_system_prompt() -> str:
    """Load the system prompt from the markdown file."""
    prompt_path = Path(__file__).parent / "prompts" / "system_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return "You are an ML Experiment Analyst Agent."


def create_analyst_agent(config: AgentConfig | None = None) -> Any:
    """Create and return the ML Experiment Analyst Agent.

    The agent is built on deepagents (LangGraph). It uses a FilesystemBackend
    to persist generated reports to data/agent-workspace/.

    Args:
        config: Agent configuration. Uses environment variable defaults if None.

    Returns:
        CompiledStateGraph ready for invocation via agent.invoke({...}).

    Example:
        agent = create_analyst_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": "hello"}]})
    """
    from deepagents import create_deep_agent  # type: ignore[import]
    from deepagents.backends import FilesystemBackend  # type: ignore[import]

    from src.tools import ALL_TOOLS

    if config is None:
        config = AgentConfig()

    config.workspace_path.mkdir(parents=True, exist_ok=True)

    llm = _build_llm(config)
    system_prompt = _load_system_prompt()
    backend = FilesystemBackend(root_dir=config.workspace_path)

    agent = create_deep_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=system_prompt,
        backend=backend,
    )

    return agent
