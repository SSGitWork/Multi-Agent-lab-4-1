"""
Lab 3.4 — solution/pipeline.py
================================
Reference implementation of the LangGraph pipeline.
"""


import asyncio
from typing import TypedDict

from langgraph.graph import END, StateGraph

from a2a import Broker
from coder_agent import run_coder_agent
from pm_agent import Spec, run_pm_agent
from qa_agent import run_qa_agent_a2a

MCP_SERVER = "./mcp_server.py"


class PipelineState(TypedDict):
    requirement: str
    spec: Spec | None
    final_code: str


async def pm_node(state: PipelineState) -> dict:
    """Create a PM specification from the incoming requirement."""
    spec = await run_pm_agent(state["requirement"])
    return {"spec": spec}


async def coder_node(state: PipelineState) -> dict:
    """Run Coder and QA concurrently using one shared A2A broker."""
    del state

    broker = Broker()

    final_code, _ = await asyncio.gather(
        run_coder_agent(broker, MCP_SERVER),
        run_qa_agent_a2a(broker, MCP_SERVER),
    )

    return {"final_code": final_code}


def build_pipeline():
    """Build and compile the PM -> Coder/QA -> END LangGraph."""
    graph = StateGraph(PipelineState)

    graph.add_node("pm", pm_node)
    graph.add_node("coder", coder_node)

    graph.set_entry_point("pm")
    graph.add_edge("pm", "coder")
    graph.add_edge("coder", END)

    return graph.compile()


async def run_pipeline(requirement: str) -> PipelineState:
    """Execute the full pipeline and return its final state."""
    pipeline = build_pipeline()

    return await pipeline.ainvoke(
        {
            "requirement": requirement,
            "spec": None,
            "final_code": "",
        }
    )
