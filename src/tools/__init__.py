"""Custom LangChain tools for the ML Experiment Analyst Agent.

All tools are exported here so they can be imported in one line:

    from src.tools import ALL_TOOLS
"""

from src.tools.analyze_patterns import analyze_patterns
from src.tools.compare_runs import compare_runs
from src.tools.diagnose_run import diagnose_run
from src.tools.generate_report import generate_report
from src.tools.load_experiment import load_experiment
from src.tools.suggest_next_experiments import suggest_next_experiments

ALL_TOOLS = [
    load_experiment,
    compare_runs,
    diagnose_run,
    analyze_patterns,
    suggest_next_experiments,
    generate_report,
]

__all__ = [
    "load_experiment",
    "compare_runs",
    "diagnose_run",
    "analyze_patterns",
    "suggest_next_experiments",
    "generate_report",
    "ALL_TOOLS",
]
