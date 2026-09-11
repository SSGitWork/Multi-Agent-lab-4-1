# Lab 4.1 — Distributed Tracing Across All Three Agents

**Module 4 · Section 1 · Lab 1**
Build Autonomous Multi-Agent Systems · Saras AI Institute

---

## Objective

Instrument the three-agent pipeline from Lab 3.4 so that a single
pipeline run produces **one end-to-end trace** in LangSmith (or
Phoenix/Arize) spanning the PM, Coder, and QA agents.

You do **not** modify the Lab 3.4 agents. All instrumentation lives in
two new files: `tracing.py` and `instrumented_pipeline.py`.

---

## What You Will Build

```
tracing.py                ← Tracing helpers (you implement)
instrumented_pipeline.py  ← Traced node wrappers + pipeline (you implement)
```

### Pre-built (do not modify)

```
pm_agent.py      ← Lab 3.4 pm_agent
pipeline.py      ← Lab 3.4 pipeline
coder_agent.py   ← Lab 3.3 Coder agent
qa_agent.py      ← Lab 3.2 QA agent
mcp_server.py    ← Lab 3.1 MCP server
a2a.py           ← Lab 3.3 A2A broker
llm_client.py    ← Shared LLM client (Helicone proxy)
```

---

## Architecture

```
run_instrumented_pipeline()
    │
    │  configure_tracing()          ← sets LANGCHAIN_TRACING_V2=true
    │  run_id = new_run_id()        ← UUID4 root trace ID
    │
    └─ LangGraph StateGraph (TracedPipelineState)
           │
           ├── traced_pm_node       @traceable(name="pm_node")
           │       └── _original_pm_node()
           │
           └── traced_coder_node    @traceable(name="coder_node")
                   └── _original_coder_node()
                           ├── run_coder_agent()   ─┐ asyncio.gather
                           └── run_qa_agent_a2a()  ─┘
```

A `token_log` list accumulates one entry per node and is returned in
the final state so you can read per-agent cost data without querying
the LangSmith API.

---

## TracedPipelineState

Extends `PipelineState` from Lab 3.4 with two extra fields:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | UUID4 set once in `run_instrumented_pipeline` |
| `token_log` | `list[dict]` | One entry appended by each traced node |

Each `token_log` entry must contain:

```python
{
    "agent_name": "pm" | "coder" | "qa",
    "token_count": int,
    "tool_calls":  int,
    "node":        "pm_node" | "coder_node",
}
```

---

## Tasks

### `tracing.py`

| Function | What to implement |
|----------|-------------------|
| `configure_tracing()` | Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_PROJECT` via `os.environ` |
| `make_span_metadata(agent_name, token_count, tool_calls)` | Return a `dict` with those three keys |
| `new_run_id()` | Return `str(uuid.uuid4())` |

Fill in `TRACE_FINDINGS` at the bottom of the file after your first
successful traced run.

### `instrumented_pipeline.py`

| Symbol | What to implement |
|--------|-------------------|
| `TracedPipelineState` | TypedDict extending PipelineState with `run_id` and `token_log` |
| `traced_pm_node(state)` | `@traceable(name="pm_node")` wrapper; calls `_original_pm_node`; appends to `token_log` |
| `traced_coder_node(state)` | `@traceable(name="coder_node")` wrapper; calls `_original_coder_node`; appends to `token_log` |
| `build_instrumented_pipeline()` | `StateGraph(TracedPipelineState)` wired with the traced nodes |
| `run_instrumented_pipeline(requirement)` | Calls `configure_tracing()`, sets `run_id`, invokes graph inside `tracing_v2_enabled()` |

---

## Setup

```bash
pip install -r requirements.txt
```
## 🔑 Getting Your LangSmith API Keys (Quick Setup)

Before running the traced pipeline, you need to create a LangSmith account, generate an API key, and plug it into your environment variables.

### Steps

1. Go to LangSmith
   → [https://smith.langchain.com](https://smith.langchain.com)

2. Sign up (or log in) using your email or GitHub.

3. Create a new project

   * Click **“New Project”**
   * Name it something like: `lab-4-1`

4. Generate an API key

   * Go to **Settings → API Keys**
   * Click **Create API Key**
   * Copy the key

5. Paste your credentials into your environment file (.env)

```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=your_endpoint_here
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=project_name_here
```

That’s it—once this is set, your pipeline runs will automatically appear in your LangSmith dashboard.

---

## Running

```bash
python main.py
```

Expected output:

```
=== Lab 4.1 — Distributed Tracing ===

[PM]    Analysing requirement...
[Coder] Iteration 1 — generating fix...
[QA]    Reviewing word_counter.py...
...

--- Pipeline complete ---
run_id     : 3f8a1c2d-...
token_log  : [{'agent_name': 'pm', ...}, {'agent_name': 'coder', ...}]
final_code : def word_count(path: str) ...
```

Then open your LangSmith dashboard and verify a single trace with
`pm_node` and `coder_node` as child spans.

---

## Tests

```bash
# Mocked suite (fast, no API calls)
pytest

# Full live pipeline + LangSmith verification
pytest --run-llm
```

---

## Success Criteria

- [ ] `pytest` exits with 0 failures (mocked suite).
- [ ] `new_run_id()` returns a valid UUID4 string; each call returns a different value.
- [ ] `make_span_metadata()` returns a dict with `agent_name`, `token_count`, `tool_calls`.
- [ ] `TracedPipelineState` contains `run_id` and `token_log` in addition to all Lab 3.4 fields.
- [ ] `traced_pm_node` and `traced_coder_node` each call their original counterpart exactly once.
- [ ] Each traced node appends an entry with the correct `agent_name` to `token_log`.
- [ ] `build_instrumented_pipeline()` returns a compiled graph that calls the traced wrappers.
- [ ] `run_instrumented_pipeline()` calls `configure_tracing()` exactly once per invocation.
- [ ] `run_instrumented_pipeline()` sets a valid UUID4 as `run_id` in the initial state.
- [ ] `python main.py` completes end-to-end without errors.
- [ ] `TRACE_FINDINGS` is filled in with observations from a real traced run.
