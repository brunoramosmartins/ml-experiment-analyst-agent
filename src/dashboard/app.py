"""Governance Dashboard — Streamlit app for agent execution tracing.

Launch with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.dashboard.log_reader import (
    compute_tool_analytics,
    list_runs,
    load_run_events,
)

LOG_DIR = Path(os.getenv("AGENT_TRACE_LOG_DIR", "data/logs/agent_traces"))

st.set_page_config(
    page_title="Agent Governance Dashboard",
    page_icon="🔍",
    layout="wide",
)

# ─── Sidebar navigation ─────────────────────────────────────────────────────

page = st.sidebar.radio("Navigation", ["Run Explorer", "Tool Analytics"])

# ─── Page 1: Run Explorer ────────────────────────────────────────────────────

if page == "Run Explorer":
    st.title("Run Explorer")

    runs_df = list_runs(LOG_DIR)

    if runs_df.empty:
        st.info("No trace logs found. Run the agent with governance enabled to generate logs.")
        st.code(
            'python -c "\n'
            "from src.agent.builder import create_analyst_agent, "
            "invoke_with_governance\\n"
            "agent = create_analyst_agent()\\n"
            "invoke_with_governance(agent, 'Analyze the binary-classification "
            "experiment')\n\"",
            language="bash",
        )
        st.stop()

    # Sort by start_time descending (most recent first)
    if "start_time" in runs_df.columns:
        runs_df = runs_df.sort_values("start_time", ascending=False).reset_index(drop=True)

    st.metric("Total Runs", len(runs_df))

    # Run list
    st.subheader("Executions")
    display_df = runs_df[
        [
            "run_id",
            "date",
            "start_time",
            "n_tool_calls",
            "n_errors",
            "total_duration_ms",
            "total_tokens",
        ]
    ].copy()
    display_df.columns = [
        "Run ID",
        "Date",
        "Started At",
        "Tool Calls",
        "Errors",
        "Duration (ms)",
        "Tokens",
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Run detail
    st.subheader("Run Detail")
    selected_run = st.selectbox(
        "Select a run to inspect",
        runs_df["run_id"].tolist(),
        format_func=lambda rid: f"{rid} ({runs_df.loc[runs_df['run_id'] == rid, 'start_time'].iloc[0]})",
    )

    if selected_run:
        # Find the JSONL file for this run
        matching = list(LOG_DIR.glob(f"*/{selected_run}.jsonl"))
        if matching:
            events_df = load_run_events(matching[0])

            if not events_df.empty:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                tool_events = events_df[events_df["event_type"].isin(["tool_start", "tool_end"])]
                error_events = events_df[events_df["event_type"] == "tool_error"]
                col1.metric("Tool Calls", len(tool_events) // 2)
                col2.metric("Errors", len(error_events))

                total_tok = 0
                for _, row in events_df.iterrows():
                    tok = row.get("tokens")
                    if isinstance(tok, dict):
                        total_tok += tok.get("total", 0)
                col3.metric("Total Tokens", total_tok)

                # Event timeline
                st.subheader("Event Timeline")
                timeline_df = events_df[
                    [
                        "timestamp",
                        "event_type",
                        "tool_name",
                        "duration_ms",
                        "error",
                    ]
                ].copy()
                timeline_df.columns = [
                    "Timestamp",
                    "Event",
                    "Tool",
                    "Duration (ms)",
                    "Error",
                ]

                def _color_event(row):  # type: ignore[no-untyped-def]
                    if row["Event"] == "tool_error":
                        return ["background-color: #ffcccc"] * len(row)
                    if row["Event"] == "tool_end":
                        return ["background-color: #ccffcc"] * len(row)
                    return [""] * len(row)

                styled = timeline_df.style.apply(_color_event, axis=1)
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # Expandable event details
                st.subheader("Event Details")
                for idx, row in events_df.iterrows():
                    event_type = row.get("event_type", "unknown")
                    tool = row.get("tool_name", "")
                    label = f"{event_type} — {tool}" if tool else event_type
                    with st.expander(label):
                        if row.get("input_summary"):
                            st.text_area(
                                "Input",
                                row["input_summary"],
                                height=80,
                                disabled=True,
                                key=f"in_{idx}",
                            )
                        if row.get("output_summary"):
                            st.text_area(
                                "Output",
                                row["output_summary"],
                                height=120,
                                disabled=True,
                                key=f"out_{idx}",
                            )
                        if row.get("error"):
                            st.error(row["error"])

# ─── Page 2: Tool Analytics ──────────────────────────────────────────────────

elif page == "Tool Analytics":
    st.title("Tool Analytics")

    analytics_df = compute_tool_analytics(LOG_DIR)

    if analytics_df.empty:
        st.info("No tool call data available yet.")
        st.stop()

    runs_df = list_runs(LOG_DIR)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", len(runs_df))
    col2.metric("Total Tool Calls", int(analytics_df["call_count"].sum()))
    total_errors = int(analytics_df["error_count"].sum())
    total_calls = int(analytics_df["call_count"].sum() + analytics_df["error_count"].sum())
    col3.metric(
        "Overall Error Rate",
        f"{total_errors / total_calls:.1%}" if total_calls > 0 else "0%",
    )

    # Call frequency
    st.subheader("Call Frequency by Tool")
    chart_data = analytics_df.set_index("tool_name")["call_count"]
    st.bar_chart(chart_data)

    # Average latency
    st.subheader("Average Latency by Tool (ms)")
    latency_data = analytics_df.set_index("tool_name")["avg_duration_ms"]
    st.bar_chart(latency_data)

    # Detailed stats table
    st.subheader("Detailed Statistics")
    display_analytics = analytics_df.copy()
    display_analytics.columns = [
        "Tool",
        "Calls",
        "Avg Latency (ms)",
        "P95 Latency (ms)",
        "Errors",
        "Error Rate",
    ]
    st.dataframe(display_analytics, use_container_width=True, hide_index=True)
