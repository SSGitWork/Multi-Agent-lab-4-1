"""
Shared LLM client.

Uses Azure OpenAI when Azure environment variables are configured.
Otherwise, falls back to the Helicone/OpenRouter proxy configuration.
"""

import os
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI

load_dotenv(override=True)

_client: AsyncAzureOpenAI | AsyncOpenAI | None = None

_HELICONE_BASE = os.getenv("HELICONE_BASE_URL")
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")

_AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
_AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
_AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
_AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

DEFAULT_MODEL = "gpt-4.1-mini"


def _normalise_azure_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint.rstrip("/"))
    path = parsed.path.rstrip("/")

    if path.endswith("/openai/v1"):
        path = path[: -len("/openai/v1")]
    elif path.endswith("/openai"):
        path = path[: -len("/openai")]

    return urlunparse((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "", ""))


def using_azure() -> bool:
    return all(
        [
            _AZURE_OPENAI_ENDPOINT,
            _AZURE_OPENAI_API_KEY,
            _AZURE_OPENAI_API_VERSION,
            _AZURE_OPENAI_DEPLOYMENT,
        ]
    )


def get_model() -> str:
    return _AZURE_OPENAI_DEPLOYMENT or DEFAULT_MODEL


def get_client() -> AsyncAzureOpenAI | AsyncOpenAI:
    global _client

    if _client is not None:
        return _client

    if using_azure():
        _client = AsyncAzureOpenAI(
            azure_endpoint=_normalise_azure_endpoint(_AZURE_OPENAI_ENDPOINT or ""),
            api_key=_AZURE_OPENAI_API_KEY,
            api_version=_AZURE_OPENAI_API_VERSION,
        )
        return _client

    headers: dict[str, str] = {}
    if _HELICONE_API_KEY:
        headers["Helicone-Auth"] = f"Bearer {_HELICONE_API_KEY}"

    _client = AsyncOpenAI(
        api_key=_OPENROUTER_API_KEY,
        base_url=_HELICONE_BASE,
        default_headers=headers or None,
    )
    return _client
