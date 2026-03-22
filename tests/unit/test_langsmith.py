"""Unit tests for LangSmith observability helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.observability.langsmith import add_run_metadata, tag_trace

# ─── add_run_metadata ────────────────────────────────────────────────────────


def test_add_metadata_when_run_active() -> None:
    mock_run = MagicMock()
    mock_langsmith = MagicMock()
    mock_langsmith.get_current_run_tree.return_value = mock_run

    with patch.dict("sys.modules", {"langsmith": mock_langsmith}):
        add_run_metadata("experiment_name", "test-exp")

    mock_run.add_metadata.assert_called_once_with({"experiment_name": "test-exp"})


def test_add_metadata_no_active_run() -> None:
    mock_langsmith = MagicMock()
    mock_langsmith.get_current_run_tree.return_value = None

    with patch.dict("sys.modules", {"langsmith": mock_langsmith}):
        add_run_metadata("key", "value")  # Should not raise


def test_add_metadata_langsmith_not_installed() -> None:
    with patch.dict("sys.modules", {"langsmith": None}):
        add_run_metadata("key", "value")  # Should not raise


# ─── tag_trace ───────────────────────────────────────────────────────────────


def test_tag_trace_adds_tags() -> None:
    mock_run = MagicMock()
    mock_run.tags = ["existing-tag"]
    mock_langsmith = MagicMock()
    mock_langsmith.get_current_run_tree.return_value = mock_run

    with patch.dict("sys.modules", {"langsmith": mock_langsmith}):
        tag_trace(["new-tag"])

    assert "new-tag" in mock_run.tags
    assert "existing-tag" in mock_run.tags


def test_tag_trace_no_run() -> None:
    mock_langsmith = MagicMock()
    mock_langsmith.get_current_run_tree.return_value = None

    with patch.dict("sys.modules", {"langsmith": mock_langsmith}):
        tag_trace(["tag-a"])  # Should not raise


def test_tag_trace_langsmith_not_installed() -> None:
    with patch.dict("sys.modules", {"langsmith": None}):
        tag_trace(["tag-a"])  # Should not raise
