import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-llm",
        action="store_true",
        default=False,
        help="Run tests that make live LLM / LangSmith API calls.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-llm"):
        skip_llm = pytest.mark.skip(reason="Pass --run-llm to run live LLM tests.")
        for item in items:
            if item.get_closest_marker("llm"):
                item.add_marker(skip_llm)
