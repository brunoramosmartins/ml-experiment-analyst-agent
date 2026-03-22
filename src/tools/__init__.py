"""Custom LangChain tools for the ML Experiment Analyst Agent.

All tools are exported here so they can be imported in one line:

    from src.tools import ALL_TOOLS

The 6 core MLflow tools are always available. The optional ``web_search``
tool is included only when ``TAVILY_API_KEY`` is set in the environment.
"""

import os

from src.tools.analyze_patterns import analyze_patterns
from src.tools.compare_runs import compare_runs
from src.tools.diagnose_run import diagnose_run
from src.tools.generate_report import generate_report
from src.tools.load_experiment import load_experiment
from src.tools.suggest_next_experiments import suggest_next_experiments
from src.tools.web_search import web_search

ALL_TOOLS = [
    load_experiment,
    compare_runs,
    diagnose_run,
    analyze_patterns,
    suggest_next_experiments,
    generate_report,
]


def _check_tavily() -> None:
    """Append web_search tool if TAVILY_API_KEY is available."""
    from dotenv import load_dotenv

    load_dotenv()
    if os.getenv("TAVILY_API_KEY"):
        ALL_TOOLS.append(web_search)


_check_tavily()

__all__ = [
    "load_experiment",
    "compare_runs",
    "diagnose_run",
    "analyze_patterns",
    "suggest_next_experiments",
    "generate_report",
    "web_search",
    "ALL_TOOLS",
]
