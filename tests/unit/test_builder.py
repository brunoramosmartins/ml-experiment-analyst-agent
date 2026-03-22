"""Unit tests for the agent builder module.

All tests mock deepagents — no real LLM or MLflow server required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.config import AgentConfig

# ─── _build_llm ──────────────────────────────────────────────────────────────


def test_build_llm_ollama() -> None:
    mock_ollama = MagicMock()
    with patch.dict("sys.modules", {"langchain_ollama": mock_ollama}):
        from importlib import reload

        import src.agent.builder as builder_mod

        reload(builder_mod)

        config = AgentConfig(llm_provider="ollama", llm_model="llama3.1:8b")
        builder_mod._build_llm(config)

        mock_ollama.ChatOllama.assert_called_once()


def test_build_llm_anthropic() -> None:
    mock_anthropic = MagicMock()
    with patch.dict("sys.modules", {"langchain_anthropic": mock_anthropic}):
        from importlib import reload

        import src.agent.builder as builder_mod

        reload(builder_mod)

        config = AgentConfig(llm_provider="anthropic", llm_model="claude-3-haiku")
        builder_mod._build_llm(config)

        mock_anthropic.ChatAnthropic.assert_called_once()


def test_build_llm_unsupported_raises() -> None:
    from src.agent.builder import _build_llm

    config = AgentConfig(llm_provider="invalid-provider")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        _build_llm(config)


# ─── _load_system_prompt ─────────────────────────────────────────────────────


def test_load_system_prompt_returns_content() -> None:
    from src.agent.builder import _load_system_prompt

    prompt = _load_system_prompt()
    assert len(prompt) > 100, "System prompt should be non-trivial"
    assert "ML Experiment Analyst Agent" in prompt
    assert "load_experiment" in prompt
    assert "generate_report" in prompt
    assert "Stopping Criteria" in prompt


def test_load_system_prompt_fallback() -> None:
    from src.agent.builder import _load_system_prompt

    fake_prompt_path = MagicMock()
    fake_prompt_path.exists.return_value = False

    with patch("src.agent.builder.Path") as mock_path_cls:
        mock_file = MagicMock()
        mock_file.parent.__truediv__ = MagicMock(
            return_value=MagicMock(
                __truediv__=MagicMock(return_value=fake_prompt_path)
            )
        )
        mock_path_cls.return_value = mock_file

        prompt = _load_system_prompt()
        assert prompt == "You are an ML Experiment Analyst Agent."


# ─── create_analyst_agent ────────────────────────────────────────────────────


def test_create_analyst_agent_passes_tools_and_prompt(tmp_path: Path) -> None:
    mock_create = MagicMock(return_value=MagicMock())
    mock_backend_cls = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "deepagents": MagicMock(create_deep_agent=mock_create),
            "deepagents.backends": MagicMock(FilesystemBackend=mock_backend_cls),
            "dotenv": MagicMock(),
            "langchain_ollama": MagicMock(),
        },
    ):
        from importlib import reload

        import src.agent.builder as builder_mod

        reload(builder_mod)

        config = AgentConfig(
            llm_provider="ollama",
            llm_model="llama3.1:8b",
            workspace_path=tmp_path / "workspace",
        )
        builder_mod.create_analyst_agent(config)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args

        # Verify tools: at least 6 core tools
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        if tools is None:
            tools = call_kwargs[0][1] if len(call_kwargs[0]) > 1 else []
        assert len(tools) >= 6, f"Expected at least 6 tools, got {len(tools)}"

        # Verify system_prompt is non-empty
        prompt = call_kwargs.kwargs.get("system_prompt") or (
            call_kwargs[0][2] if len(call_kwargs[0]) > 2 else ""
        )
        assert "ML Experiment Analyst" in str(prompt)


def test_create_analyst_agent_creates_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "new-workspace"
    assert not workspace.exists()

    with patch.dict(
        "sys.modules",
        {
            "deepagents": MagicMock(),
            "deepagents.backends": MagicMock(),
            "dotenv": MagicMock(),
            "langchain_ollama": MagicMock(),
        },
    ):
        from importlib import reload

        import src.agent.builder as builder_mod

        reload(builder_mod)

        config = AgentConfig(
            llm_provider="ollama",
            workspace_path=workspace,
        )
        builder_mod.create_analyst_agent(config)

    assert workspace.exists(), "Workspace directory should be created"
