"""
Lab 3.2 — solution/qa_agent.py
================================
Reference implementation of the QA agent.
"""


import asyncio
import os
import pathlib
import sys
from typing import Literal
from urllib.parse import urlparse, urlunparse

from a2a import Broker, Message
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field, model_validator

ENV_FILE = pathlib.Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------
_client: AsyncAzureOpenAI | None = None
# Use getenv so mocked tests can import qa_agent.py without Azure secrets.
MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")


def _normalise_azure_endpoint(endpoint: str) -> str:
    """Convert Azure endpoint variants to the resource-root endpoint."""
    parsed = urlparse(endpoint.rstrip("/"))
    path = parsed.path.rstrip("/")

    if path.endswith("/openai/v1"):
        path = path[: -len("/openai/v1")]
    elif path.endswith("/openai"):
        path = path[: -len("/openai")]

    return urlunparse((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "", ""))


def _get_client() -> AsyncAzureOpenAI:
    """Create an Azure OpenAI client only when a live call is needed."""
    global _client
    if _client is None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": endpoint,
                "AZURE_OPENAI_API_KEY": api_key,
                "AZURE_OPENAI_API_VERSION": api_version,
                "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            }.items()
            if not value
        ]

        if missing:
            raise RuntimeError(
                "Missing Azure OpenAI environment variable(s): "
                + ", ".join(missing)
            )

        _client = AsyncAzureOpenAI(
            azure_endpoint=_normalise_azure_endpoint(endpoint),
            api_key=api_key,
            api_version=api_version,
        )
    return _client


# ---------------------------------------------------------------------------
# MCP server path -- points to Lab 3.1
# ---------------------------------------------------------------------------
LAB31_SERVER = "./mcp_server.py"

# ---------------------------------------------------------------------------
# TODO 1 -- Define the Issue Pydantic model
# ---------------------------------------------------------------------------
class Issue(BaseModel):
    """One issue found during a code review."""

    severity: Literal["High", "Medium", "Low"]
    location: str
    description: str


# ---------------------------------------------------------------------------
# TODO 2 -- Define the IssueList Pydantic model
# ---------------------------------------------------------------------------
class IssueList(BaseModel):
    """Structured output returned by the LLM code review."""

    issues: list[Issue] = Field(default_factory=list)
    approved: bool

    @model_validator(mode="after")
    def enforce_approved(self) -> "IssueList":
        """Disallow approval when any High-severity issue exists."""
        if any(issue.severity == "High" for issue in self.issues):
            self.approved = False
        return self


# ---------------------------------------------------------------------------
# TODO 3 -- Implement _read_file_via_mcp
# ---------------------------------------------------------------------------
async def _read_file_via_mcp(server_script_path: str, filename: str) -> str:
    """Read a file from the Lab 3.1 MCP server and return its contents.

    Parameters
    ----------
    server_script_path:
        Absolute path to mcp_server.py from Lab 3.1.
    filename:
        Relative filename inside project_files/, e.g. "word_counter.py".

    Returns
    -------
    str
        File contents as returned by the MCP server.

    Implementation notes
    --------------------
    Use the same pattern from Lab 3.1 mcp_client.py:
        params = StdioServerParameters(command=sys.executable, args=[server_script_path])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("read_file", {"path": filename})
                return result.content[0].text
    """
    server_path = pathlib.Path(server_script_path).resolve()

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        cwd=str(server_path.parent),
        env=os.environ.copy(),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("read_file", {"path": filename})
            return result.content[0].text


# ---------------------------------------------------------------------------
# TODO 4 -- Implement _review_code
# ---------------------------------------------------------------------------
REVIEW_PROMPT = """
You are a meticulous Python QA engineer performing a code review.

Review the Python source code below. Identify every correctness bug, edge
case, documentation mismatch, algorithm error, and meaningful style problem.

Severity rules:
- High: logic errors, incorrect output, crashes, or behavior that contradicts
    the function's documented purpose.
- Medium: unhandled edge cases or incorrect algorithm behavior in less common
    scenarios.
- Low: maintainability or style issues, including missing type annotations or
    missing documentation.

For every issue:
- Use exactly one severity: High, Medium, or Low.
- Give a specific location, such as a function name and relevant operation.
- Give a specific, actionable description.

Set approved to true only when there are zero High-severity issues.
Pay close attention to whether function implementations honor their docstrings.

Python source code to review:

{code}
"""


async def _review_code(code: str) -> "IssueList":
    """Send code to the LLM for structured review and return an IssueList.

    Parameters
    ----------
    code:
        Python source code to review.

    Returns
    -------
    IssueList
        Structured list of issues with severities and approved flag.

    Implementation notes
    --------------------
    Write a REVIEW_PROMPT that instructs the LLM to:
        - Find every bug, style issue, and missing edge case
        - Assign severity: High / Medium / Low
        - Give the location (function name and brief description)
        - Set approved=True only if there are zero High-severity issues

    Use structured output:
        response = await _get_client().beta.chat.completions.parse(
            model=MODEL,
            messages=[{"role": "user", "content": REVIEW_PROMPT.format(code=code)}],
            response_format=IssueList,
        )
        return response.choices[0].message.parsed
    """
    response = await _get_client().beta.chat.completions.parse(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", MODEL),
        messages=[
            {
                "role": "user",
                "content": REVIEW_PROMPT.format(code=code),
            }
        ],
        response_format=IssueList,
    )

    return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# TODO 5 -- Implement run_qa_agent
# ---------------------------------------------------------------------------
async def run_qa_agent(
    server_script_path: str = LAB31_SERVER,
    filename: str = "word_counter.py",
) -> "IssueList":
    """Orchestrate the QA agent: read file via MCP, review with LLM.

    Parameters
    ----------
    server_script_path:
        Path to mcp_server.py from Lab 3.1.
    filename:
        File inside project_files/ to review.

    Returns
    -------
    IssueList
        Structured review result.

    Implementation notes
    --------------------
    1. Call _read_file_via_mcp(server_script_path, filename).
    2. Call _review_code(code).
    3. Return the IssueList.
    """
    code = await _read_file_via_mcp(server_script_path, filename)
    return await _review_code(code)


async def run_qa_agent_a2a(
    broker: Broker,
    server_script_path: str,
) -> None:
    """Receive Coder review requests and reply through the A2A broker."""
    while True:
        request = await broker.receive("qa")

        if request.intent != "review_request":
            raise ValueError(
                f"Unexpected message for QA agent: {request.intent}"
            )

        filename = request.payload.get("filename", "word_counter.py")
        code = await _read_file_via_mcp(server_script_path, filename)
        review = await _review_code(code)

        if review.approved:
            response = Message(
                sender="qa",
                receiver="coder",
                intent="approved",
                payload={"filename": filename},
                correlation_id=request.correlation_id,
            )
            await broker.send(response)
            return

        response = Message(
            sender="qa",
            receiver="coder",
            intent="fix_instruction",
            payload={
                "filename": filename,
                "issues": [issue.model_dump() for issue in review.issues],
            },
            correlation_id=request.correlation_id,
        )
        await broker.send(response)


# ---------------------------------------------------------------------------
# Entry point -- do not modify
# ---------------------------------------------------------------------------
async def _main() -> None:
    print("QA Agent reviewing: word_counter.py\n")
    result = await run_qa_agent()
    print(f"Approved: {result.approved}")
    print(f"Issues found: {len(result.issues)}\n")
    for issue in result.issues:
        print(f"  [{issue.severity}] {issue.location}")
        print(f"    {issue.description}\n")


if __name__ == "__main__":
    asyncio.run(_main())
