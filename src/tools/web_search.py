"""Tool: web_search — search the web for ML techniques, papers, and best practices."""

from __future__ import annotations

import os

from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web for ML techniques, papers, or best practices.

    Use this tool when you need external context about a model architecture,
    hyperparameter tuning strategy, or machine learning best practice that is
    not available in the experiment data.

    Args:
        query: The search query (e.g., "gradient boosting overfitting remedies").
        max_results: Number of results to return (default: 3, max: 5).
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            "ERROR: TAVILY_API_KEY is not configured. "
            "Web search is unavailable. Set it in your .env file to enable this tool."
        )

    max_results = min(max(1, max_results), 5)

    try:
        from tavily import TavilyClient  # type: ignore[import]

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=max_results)
    except ImportError:
        return "ERROR: tavily-python is not installed. Run: pip install tavily-python"
    except Exception as exc:
        return f"ERROR: Web search failed — {exc}"

    results = response.get("results", [])
    if not results:
        return f"No results found for: {query}"

    lines = [f"# Web Search Results for: {query}\n"]
    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "No snippet available.")
        # Truncate long snippets
        if len(content) > 300:
            content = content[:297] + "..."
        lines.append(f"## {i}. {title}")
        lines.append(f"**URL:** {url}")
        lines.append(f"{content}\n")

    return "\n".join(lines)
