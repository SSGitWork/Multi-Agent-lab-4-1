"""
Lab 4.1 — Instrumented Pipeline
================================
Tracing wrappers around the Lab 3.4 PM and Coder pipeline nodes.
"""

import inspect
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from pipeline import (
    PipelineState,
    pm_node as _original_pm_node,
    coder_node as _original_coder_node,
)
from tracing import configure_tracing, make_span_metadata, new_run_id


class TracedPipelineState(PipelineState):
    requirement: str
    spec: Any | None
    final_code: str
    run_id: str
    token_log: list[dict[str, Any]]


def _token_count_from_result(result: dict[str, Any]) -> int:
    usage = result.get("usage", {})
    if not isinstance(usage, dict):
        return 0
    try:
        return int(usage.get("total_tokens", 0))
    except (TypeError, ValueError):
        return 0


def _tool_calls_from_result(result: dict[str, Any]) -> int:
    calls = result.get("tool_calls", 0)
    if isinstance(calls, list):
        return len(calls)
    try:
        return int(calls)
    except (TypeError, ValueError):
        return 0


def _attach_current_span_metadata(metadata: dict[str, Any]) -> None:
    try:
        run_tree = get_current_run_tree()
        if run_tree is None:
            return
        run_tree.extra.setdefault("metadata", {}).update(metadata)
    except Exception:
        pass


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@traceable(name="pm_node")
async def traced_pm_node(state: TracedPipelineState) -> dict[str, Any]:
    result = await _maybe_await(_original_pm_node(state))
    token_count = _token_count_from_result(result)
    tool_calls = _tool_calls_from_result(result)
    metadata = make_span_metadata("pm", token_count, tool_calls)
    metadata["run_id"] = state["run_id"]
    _attach_current_span_metadata(metadata)
    token_log = list(state.get("token_log", []))
    token_log.append({**make_span_metadata("pm", token_count, tool_calls), "node": "pm_node"})
    return {**result, "token_log": token_log}


@traceable(name="coder_node")
async def traced_coder_node(state: TracedPipelineState) -> dict[str, Any]:
    result = await _maybe_await(_original_coder_node(state))
    token_count = _token_count_from_result(result)
    tool_calls = _tool_calls_from_result(result)
    metadata = make_span_metadata("coder", token_count, tool_calls)
    metadata["run_id"] = state["run_id"]
    _attach_current_span_metadata(metadata)
    token_log = list(state.get("token_log", []))
    token_log.append({**make_span_metadata("coder", token_count, tool_calls), "node": "coder_node"})
    return {**result, "token_log": token_log}


def build_instrumented_pipeline():
    graph = StateGraph(TracedPipelineState)
    graph.add_node("pm", traced_pm_node)
    graph.add_node("coder", traced_coder_node)
    graph.set_entry_point("pm")
    graph.add_edge("pm", "coder")
    graph.add_edge("coder", END)
    return graph.compile()


@traceable(name="instrumented_pipeline")
async def _invoke_as_root_trace(graph: Any, initial_state: TracedPipelineState) -> dict[str, Any]:
    return await graph.ainvoke(initial_state)


async def run_instrumented_pipeline(requirement: str) -> dict[str, Any]:
    """
    Configure tracing, create a root run ID, and run the traced graph.
    """
    configure_tracing()

    run_id = new_run_id()

    initial_state: TracedPipelineState = {
        "requirement": requirement,
        "spec": None,
        "final_code": "",
        "run_id": run_id,
        "token_log": [],
    }

    graph = build_instrumented_pipeline()

    # @traceable-decorated wrappers create LangSmith spans automatically
    # when LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 are enabled.
    return await graph.ainvoke(initial_state)