"""
Lab 4.1 — Instrumented Pipeline
================================
Wrap the three agent nodes from Lab 3.4 with tracing decorators.
This file is the only place you add @traceable (or equivalent) calls.

Do NOT modify the original pipeline.py from Lab 3.4.

Structure
---------
Each instrumented node must:
  1. Call the original node function (imported from pipeline.py).
  2. Capture token_count and tool_calls from the node's return value
     or from LangSmith run context.
  3. Attach span metadata via make_span_metadata().
  4. Propagate the run_id from state into the child span so LangSmith
     links it to the root trace.

PipelineState extension
-----------------------
You need to add two fields to PipelineState (define TracedPipelineState
below — do not edit pipeline.py):

    run_id    : str          # root trace ID, set once in run_pipeline
    token_log : list[dict]   # one entry per node, appended by each node
"""

import asyncio
from typing import Any

# -- Import from Lab 3.4 pre-built files (do not modify those files) --------
from pipeline import (          # noqa: F401  (pipeline.py from Lab 3.4)
    PipelineState,
    pm_node as _original_pm_node,
    coder_node as _original_coder_node,
    build_pipeline,
)
from tracing import configure_tracing, make_span_metadata, new_run_id

# Uncomment once you implement tracing.py:
# from langsmith import traceable


# ---------------------------------------------------------------------------
# Extended state
# ---------------------------------------------------------------------------

# TODO: define TracedPipelineState that extends PipelineState with
#       run_id (str) and token_log (list[dict]).


# ---------------------------------------------------------------------------
# Instrumented node wrappers
# ---------------------------------------------------------------------------

# TODO: implement traced_pm_node(state) -> dict
#   • Calls _original_pm_node(state)
#   • Decorates / wraps with @traceable(name="pm_node")
#   • Attaches make_span_metadata("pm", token_count, tool_calls)
#   • Appends an entry to state["token_log"]


# TODO: implement traced_coder_node(state) -> dict
#   • Calls _original_coder_node(state)
#   • Decorates / wraps with @traceable(name="coder_node")
#   • Attaches make_span_metadata("coder", token_count, tool_calls)
#   • Appends an entry to state["token_log"]


# ---------------------------------------------------------------------------
# Instrumented pipeline builder
# ---------------------------------------------------------------------------

def build_instrumented_pipeline():
    """
    Build a LangGraph StateGraph identical to Lab 3.4's build_pipeline()
    but with traced_pm_node and traced_coder_node substituted in.

    Hint: copy build_pipeline() from pipeline.py and swap the node
    functions — do not import build_pipeline() and try to patch it.
    """
    raise NotImplementedError("Implement build_instrumented_pipeline()")


async def run_instrumented_pipeline(requirement: str) -> dict[str, Any]:
    """
    Entry point called by main.py.

    Steps
    -----
    1. Call configure_tracing() to initialise the backend.
    2. Generate a root run_id with new_run_id().
    3. Invoke the instrumented pipeline with the initial state.
    4. Return the final state dict.
    """
    raise NotImplementedError("Implement run_instrumented_pipeline()")
