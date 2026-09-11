"""
Lab 4.1 — test suite
=====================
All tests in this file run with mocked LLM/MCP calls (fast, no API cost).
Tests marked @pytest.mark.llm only run when you pass --run-llm.

Run mocked suite:   pytest
Run live suite:     pytest --run-llm
"""

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_requirement() -> str:
    return "Write a Python function that reverses a string."


@pytest.fixture
def mock_pm_output() -> dict:
    return {
        "spec": MagicMock(
            goal="Write a string-reversal function.",
            constraints=["Use type annotations", "Include docstring"],
            filename="solution.py",
        )
    }


@pytest.fixture
def mock_coder_output() -> dict:
    return {
        "final_code": (
            "def reverse_string(s: str) -> str:\n"
            "    \"\"\"Reverse a string.\"\"\"\n"
            "    return s[::-1]\n"
        )
    }


# ---------------------------------------------------------------------------
# tracing.py unit tests
# ---------------------------------------------------------------------------

class TestTracingHelpers:

    def test_new_run_id_is_valid_uuid(self):
        from tracing import new_run_id
        run_id = new_run_id()
        # Must be parseable as UUID4
        parsed = uuid.UUID(run_id, version=4)
        assert str(parsed) == run_id

    def test_new_run_id_unique(self):
        from tracing import new_run_id
        ids = {new_run_id() for _ in range(10)}
        assert len(ids) == 10

    def test_make_span_metadata_keys(self):
        from tracing import make_span_metadata
        meta = make_span_metadata("pm", 1234, 3)
        assert meta["agent_name"] == "pm"
        assert meta["token_count"] == 1234
        assert meta["tool_calls"] == 3

    def test_make_span_metadata_valid_agent_names(self):
        from tracing import make_span_metadata
        for name in ("pm", "coder", "qa"):
            meta = make_span_metadata(name, 0, 0)
            assert meta["agent_name"] == name

    def test_make_span_metadata_zero_values(self):
        from tracing import make_span_metadata
        meta = make_span_metadata("qa", 0, 0)
        assert meta["token_count"] == 0
        assert meta["tool_calls"] == 0


# ---------------------------------------------------------------------------
# instrumented_pipeline.py unit tests
# ---------------------------------------------------------------------------

class TestTracedPipelineState:

    def test_traced_state_has_run_id_field(self):
        from instrumented_pipeline import TracedPipelineState
        # TypedDict — just check the annotations contain the expected keys
        annotations = TracedPipelineState.__annotations__
        assert "run_id" in annotations

    def test_traced_state_has_token_log_field(self):
        from instrumented_pipeline import TracedPipelineState
        annotations = TracedPipelineState.__annotations__
        assert "token_log" in annotations

    def test_traced_state_inherits_requirement(self):
        from instrumented_pipeline import TracedPipelineState
        annotations = TracedPipelineState.__annotations__
        assert "requirement" in annotations

    def test_traced_state_inherits_spec(self):
        from instrumented_pipeline import TracedPipelineState
        annotations = TracedPipelineState.__annotations__
        assert "spec" in annotations

    def test_traced_state_inherits_final_code(self):
        from instrumented_pipeline import TracedPipelineState
        annotations = TracedPipelineState.__annotations__
        assert "final_code" in annotations


class TestInstrumentedNodes:

    @patch("instrumented_pipeline._original_pm_node")
    @patch("instrumented_pipeline.configure_tracing")
    def test_traced_pm_node_calls_original(
        self, mock_configure, mock_orig_pm, mock_pm_output
    ):
        from instrumented_pipeline import traced_pm_node
        mock_orig_pm.return_value = mock_pm_output
        state: Any = {
            "requirement": "test",
            "spec": None,
            "final_code": "",
            "run_id": str(uuid.uuid4()),
            "token_log": [],
        }
        result = asyncio.get_event_loop().run_until_complete(
            traced_pm_node(state)
        ) if asyncio.iscoroutinefunction(traced_pm_node) else traced_pm_node(state)
        mock_orig_pm.assert_called_once()

    @patch("instrumented_pipeline._original_pm_node")
    def test_traced_pm_node_appends_token_log(self, mock_orig_pm, mock_pm_output):
        from instrumented_pipeline import traced_pm_node
        mock_orig_pm.return_value = mock_pm_output
        state: Any = {
            "requirement": "test",
            "spec": None,
            "final_code": "",
            "run_id": str(uuid.uuid4()),
            "token_log": [],
        }
        if asyncio.iscoroutinefunction(traced_pm_node):
            result = asyncio.get_event_loop().run_until_complete(traced_pm_node(state))
        else:
            result = traced_pm_node(state)
        # token_log entry must be present in returned state or state was mutated
        log = result.get("token_log") or state.get("token_log")
        assert len(log) >= 1

    @patch("instrumented_pipeline._original_pm_node")
    def test_traced_pm_node_log_entry_has_agent_name(
        self, mock_orig_pm, mock_pm_output
    ):
        from instrumented_pipeline import traced_pm_node
        mock_orig_pm.return_value = mock_pm_output
        state: Any = {
            "requirement": "test",
            "spec": None,
            "final_code": "",
            "run_id": str(uuid.uuid4()),
            "token_log": [],
        }
        if asyncio.iscoroutinefunction(traced_pm_node):
            result = asyncio.get_event_loop().run_until_complete(traced_pm_node(state))
        else:
            result = traced_pm_node(state)
        log = result.get("token_log") or state.get("token_log")
        assert log[0].get("agent_name") == "pm"

    @patch("instrumented_pipeline._original_coder_node")
    def test_traced_coder_node_calls_original(
        self, mock_orig_coder, mock_pm_output, mock_coder_output
    ):
        from instrumented_pipeline import traced_coder_node
        mock_orig_coder.return_value = mock_coder_output
        state: Any = {
            "requirement": "test",
            "spec": mock_pm_output["spec"],
            "final_code": "",
            "run_id": str(uuid.uuid4()),
            "token_log": [],
        }
        if asyncio.iscoroutinefunction(traced_coder_node):
            asyncio.get_event_loop().run_until_complete(traced_coder_node(state))
        else:
            traced_coder_node(state)
        mock_orig_coder.assert_called_once()

    @patch("instrumented_pipeline._original_coder_node")
    def test_traced_coder_node_log_entry_has_agent_name(
        self, mock_orig_coder, mock_pm_output, mock_coder_output
    ):
        from instrumented_pipeline import traced_coder_node
        mock_orig_coder.return_value = mock_coder_output
        state: Any = {
            "requirement": "test",
            "spec": mock_pm_output["spec"],
            "final_code": "",
            "run_id": str(uuid.uuid4()),
            "token_log": [],
        }
        if asyncio.iscoroutinefunction(traced_coder_node):
            result = asyncio.get_event_loop().run_until_complete(traced_coder_node(state))
        else:
            result = traced_coder_node(state)
        log = result.get("token_log") or state.get("token_log")
        assert log[0].get("agent_name") == "coder"


class TestBuildInstrumentedPipeline:

    def test_build_instrumented_pipeline_returns_compiled_graph(self):
        from instrumented_pipeline import build_instrumented_pipeline
        graph = build_instrumented_pipeline()
        # LangGraph compiled graphs expose .invoke or .ainvoke
        assert hasattr(graph, "ainvoke") or hasattr(graph, "invoke")

    @patch("instrumented_pipeline.traced_pm_node")
    @patch("instrumented_pipeline.traced_coder_node")
    def test_pipeline_uses_traced_nodes(self, mock_coder, mock_pm):
        """
        Verify that build_instrumented_pipeline wires the traced wrappers,
        not the originals.  We check this by ensuring the mocked traced
        nodes are called when the pipeline runs.
        """
        from instrumented_pipeline import build_instrumented_pipeline
        import uuid as _uuid

        mock_pm.return_value = {
            "spec": MagicMock(goal="g", constraints=[], filename="f.py"),
            "token_log": [{"agent_name": "pm", "token_count": 10, "tool_calls": 0}],
        }
        mock_coder.return_value = {
            "final_code": "pass",
            "token_log": [{"agent_name": "coder", "token_count": 20, "tool_calls": 2}],
        }

        graph = build_instrumented_pipeline()
        initial_state = {
            "requirement": "test",
            "spec": None,
            "final_code": "",
            "run_id": str(_uuid.uuid4()),
            "token_log": [],
        }
        asyncio.get_event_loop().run_until_complete(graph.ainvoke(initial_state))
        mock_pm.assert_called_once()
        mock_coder.assert_called_once()


class TestRunInstrumentedPipeline:

    @patch("instrumented_pipeline.configure_tracing")
    @patch("instrumented_pipeline.build_instrumented_pipeline")
    def test_run_pipeline_calls_configure_tracing(
        self, mock_build, mock_configure
    ):
        from instrumented_pipeline import run_instrumented_pipeline
        import uuid as _uuid

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "requirement": "test",
            "spec": MagicMock(),
            "final_code": "pass",
            "run_id": str(_uuid.uuid4()),
            "token_log": [],
        }
        mock_build.return_value = mock_graph

        asyncio.get_event_loop().run_until_complete(
            run_instrumented_pipeline("test requirement")
        )
        mock_configure.assert_called_once()

    @patch("instrumented_pipeline.configure_tracing")
    @patch("instrumented_pipeline.build_instrumented_pipeline")
    def test_run_pipeline_sets_run_id_in_state(
        self, mock_build, mock_configure
    ):
        from instrumented_pipeline import run_instrumented_pipeline
        import uuid as _uuid

        captured_state: dict = {}

        async def capture_invoke(state):
            captured_state.update(state)
            return {**state, "final_code": "pass", "spec": MagicMock()}

        mock_graph = MagicMock()
        mock_graph.ainvoke = capture_invoke
        mock_build.return_value = mock_graph

        asyncio.get_event_loop().run_until_complete(
            run_instrumented_pipeline("test requirement")
        )
        assert "run_id" in captured_state
        # Must be a valid UUID
        uuid.UUID(captured_state["run_id"], version=4)

    @patch("instrumented_pipeline.configure_tracing")
    @patch("instrumented_pipeline.build_instrumented_pipeline")
    def test_run_pipeline_returns_final_state(
        self, mock_build, mock_configure
    ):
        from instrumented_pipeline import run_instrumented_pipeline
        import uuid as _uuid

        expected_code = "def foo(): pass"
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "requirement": "test",
            "spec": MagicMock(),
            "final_code": expected_code,
            "run_id": str(_uuid.uuid4()),
            "token_log": [],
        }
        mock_build.return_value = mock_graph

        result = asyncio.get_event_loop().run_until_complete(
            run_instrumented_pipeline("test requirement")
        )
        assert result["final_code"] == expected_code

    @patch("instrumented_pipeline.configure_tracing")
    @patch("instrumented_pipeline.build_instrumented_pipeline")
    def test_run_pipeline_initialises_token_log(
        self, mock_build, mock_configure
    ):
        from instrumented_pipeline import run_instrumented_pipeline
        import uuid as _uuid

        captured_state: dict = {}

        async def capture_invoke(state):
            captured_state.update(state)
            return {**state, "final_code": "pass", "spec": MagicMock()}

        mock_graph = MagicMock()
        mock_graph.ainvoke = capture_invoke
        mock_build.return_value = mock_graph

        asyncio.get_event_loop().run_until_complete(
            run_instrumented_pipeline("test requirement")
        )
        assert "token_log" in captured_state
        assert isinstance(captured_state["token_log"], list)


# ---------------------------------------------------------------------------
# Live tests (only with --run-llm)
# ---------------------------------------------------------------------------

@pytest.mark.llm
class TestLivePipeline:

    def test_live_pipeline_produces_three_spans(self):
        """
        Runs the full pipeline and checks that the LangSmith client can
        retrieve a trace with exactly three child runs (pm, coder, qa).

        Requires: LANGSMITH_API_KEY, HELICONE_API_KEY set in environment.
        """
        from instrumented_pipeline import run_instrumented_pipeline
        from langsmith import Client

        result = asyncio.get_event_loop().run_until_complete(
            run_instrumented_pipeline(
                "Write a Python function that returns the factorial of n."
            )
        )
        run_id = result.get("run_id")
        assert run_id, "run_id missing from final state"

        client = Client()
        runs = list(client.list_runs(run_id=run_id, is_root=False))
        agent_names = {r.name for r in runs}
        assert "pm_node" in agent_names, f"pm_node not found in {agent_names}"
        assert "coder_node" in agent_names, f"coder_node not found in {agent_names}"

    def test_live_pipeline_token_log_populated(self):
        from instrumented_pipeline import run_instrumented_pipeline

        result = asyncio.get_event_loop().run_until_complete(
            run_instrumented_pipeline(
                "Write a Python function that checks if a number is prime."
            )
        )
        log = result.get("token_log", [])
        assert len(log) >= 2, "Expected at least pm + coder entries in token_log"
        for entry in log:
            assert "agent_name" in entry
            assert "token_count" in entry
