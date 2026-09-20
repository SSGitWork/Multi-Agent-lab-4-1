"""
Lab 4.1 — Distributed Tracing Across All Three Agents
======================================================
Tracing helpers for the instrumented Lab 3.4 pipeline.
"""

import os
import uuid
from typing import Any


def configure_tracing() -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", "lab-4-1")
    os.environ.setdefault("LANGSMITH_PROJECT", os.environ["LANGCHAIN_PROJECT"])


def make_span_metadata(agent_name: str, token_count: int, tool_calls: int) -> dict[str, Any]:
    return {
        "agent_name": agent_name,
        "token_count": token_count,
        "tool_calls": tool_calls,
    }


def new_run_id() -> str:
    return str(uuid.uuid4())


TRACE_FINDINGS: dict[str, str] = {
    "slowest_span": "Populate after a real traced pipeline run.",
    "highest_token_span": "Populate after a real traced pipeline run.",
    "notes": (
        "PM and Coder are independently instrumented. "
        "QA executes inside the Coder A2A workflow."
    ),
}
