"""
Lab 3.4 — solution/pm_agent.py
================================
Reference implementation of the PM agent.
"""


import os
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel

load_dotenv(override=True)

TARGET_FILE = "./project_files/word_counter.py"
MODEL = "gpt-4.1-mini"

_client: AsyncAzureOpenAI | AsyncOpenAI | None = None


def _normalise_azure_endpoint(endpoint: str) -> str:
    """Convert an Azure /openai or /openai/v1 URL to a resource-root URL."""
    parsed = urlparse(endpoint.rstrip("/"))
    path = parsed.path.rstrip("/")

    if path.endswith("/openai/v1"):
        path = path[: -len("/openai/v1")]
    elif path.endswith("/openai"):
        path = path[: -len("/openai")]

    return urlunparse((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "", ""))


def _get_client() -> AsyncAzureOpenAI | AsyncOpenAI:
    """Create an Azure client when Azure variables are configured.

    Falls back to the existing OpenRouter/Helicone configuration when Azure
    configuration is absent.
    """
    global _client

    if _client is not None:
        return _client

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if azure_endpoint and azure_api_key and azure_api_version:
        _client = AsyncAzureOpenAI(
            azure_endpoint=_normalise_azure_endpoint(azure_endpoint),
            api_key=azure_api_key,
            api_version=azure_api_version,
        )
        return _client

    _client = AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("HELICONE_BASE_URL"),
    )
    return _client


def _active_model() -> str:
    """Azure requires the deployment name as the model parameter."""
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", MODEL)


class Spec(BaseModel):
    """Structured implementation specification created by the PM agent."""

    goal: str
    constraints: list[str]
    filename: str


PM_PROMPT = """
You are a software project manager preparing an implementation specification.

Analyse the following user requirement and return a concise structured plan.

Requirements:
- goal must be a one-sentence restatement of the requested work.
- constraints must contain specific, testable implementation requirements.
- filename must be exactly "word_counter.py".
- Do not include assumptions unrelated to the stated requirement.

User requirement:
{requirement}
"""


async def run_pm_agent(requirement: str) -> Spec:
    """Analyse a requirement and return a structured Spec."""
    response = await _get_client().beta.chat.completions.parse(
        model=_active_model(),
        messages=[
            {
                "role": "user",
                "content": PM_PROMPT.format(requirement=requirement),
            }
        ],
        response_format=Spec,
    )

    spec = response.choices[0].message.parsed
    if spec is None:
        raise RuntimeError("The PM LLM returned no structured specification.")

    spec.filename = "word_counter.py"
    return spec
