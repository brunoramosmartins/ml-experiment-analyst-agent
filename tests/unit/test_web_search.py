"""Unit tests for the web_search tool."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

_web_search_mod = importlib.import_module("src.tools.web_search")


# ─── Missing API key ────────────────────────────────────────────────────────


def test_web_search_no_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        result = _web_search_mod.web_search.invoke({"query": "test query"})
    assert "ERROR" in result
    assert "TAVILY_API_KEY" in result


# ─── Successful search ──────────────────────────────────────────────────────


def test_web_search_returns_formatted_results() -> None:
    mock_response = {
        "results": [
            {
                "title": "Gradient Boosting Guide",
                "url": "https://example.com/gb",
                "content": "A comprehensive guide to gradient boosting machines.",
            },
            {
                "title": "Overfitting Solutions",
                "url": "https://example.com/overfit",
                "content": "Common techniques to prevent overfitting in ML models.",
            },
        ]
    }

    mock_client = MagicMock()
    mock_client.search.return_value = mock_response
    mock_tavily_cls = MagicMock(return_value=mock_client)

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        patch.dict("sys.modules", {"tavily": MagicMock(TavilyClient=mock_tavily_cls)}),
    ):
        result = _web_search_mod.web_search.invoke({"query": "gradient boosting overfitting"})

    assert "Gradient Boosting Guide" in result
    assert "Overfitting Solutions" in result
    assert "https://example.com/gb" in result


# ─── Empty results ───────────────────────────────────────────────────────────


def test_web_search_no_results() -> None:
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}
    mock_tavily_cls = MagicMock(return_value=mock_client)

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        patch.dict("sys.modules", {"tavily": MagicMock(TavilyClient=mock_tavily_cls)}),
    ):
        result = _web_search_mod.web_search.invoke({"query": "obscure query xyz"})

    assert "No results found" in result


# ─── max_results clamping ────────────────────────────────────────────────────


def test_web_search_clamps_max_results() -> None:
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}
    mock_tavily_cls = MagicMock(return_value=mock_client)

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        patch.dict("sys.modules", {"tavily": MagicMock(TavilyClient=mock_tavily_cls)}),
    ):
        _web_search_mod.web_search.invoke({"query": "test", "max_results": 10})

    # Should have been clamped to 5
    mock_client.search.assert_called_once_with("test", max_results=5)


# ─── API error handling ──────────────────────────────────────────────────────


def test_web_search_handles_api_error() -> None:
    mock_tavily_cls = MagicMock(side_effect=RuntimeError("API down"))

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        patch.dict("sys.modules", {"tavily": MagicMock(TavilyClient=mock_tavily_cls)}),
    ):
        result = _web_search_mod.web_search.invoke({"query": "test"})

    assert "ERROR" in result
