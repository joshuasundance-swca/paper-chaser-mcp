import runpy
import sys
import types
from typing import Any

import paper_chaser_mcp.deployment_runner as deployment_runner
from paper_chaser_mcp.parsing import _arxiv_id_from_url, _text
from paper_chaser_mcp.runtime import run_server
from paper_chaser_mcp.settings import AppSettings
from paper_chaser_mcp.tools import get_tool_definitions


class _RecordingApp:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _RecordingLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []

    def info(self, message: str, *args: Any) -> None:
        self.info_messages.append(message % args if args else message)

    def warning(self, message: str, *args: Any) -> None:
        self.warning_messages.append(message % args if args else message)


def test_runtime_logs_remote_http_configuration_and_runs_app() -> None:
    app = _RecordingApp()
    logger = _RecordingLogger()
    settings = AppSettings.from_env(
        {
            "SEMANTIC_SCHOLAR_API_KEY": "semantic-key",
            "CORE_API_KEY": "core-key",
            "PAPER_CHASER_ENABLE_SERPAPI": "true",
            "SERPAPI_API_KEY": "serp-key",
            "PAPER_CHASER_TRANSPORT": "streamable-http",
            "PAPER_CHASER_HTTP_HOST": "remote.example",
            "PAPER_CHASER_HTTP_PORT": "8081",
            "PAPER_CHASER_HTTP_PATH": "/custom-mcp",
            "PAPER_CHASER_HTTP_AUTH_TOKEN": "super-secret",
            "PAPER_CHASER_HTTP_AUTH_HEADER": "x-backend-auth",
            "PAPER_CHASER_ALLOWED_ORIGINS": ("https://allowed-one.example,https://allowed-two.example"),
        }
    )

    run_server(app=app, logger=logger, settings=settings)

    assert any("Semantic Scholar API key detected" in item for item in logger.info_messages)
    assert any("CORE API key set" in item for item in logger.info_messages)
    assert any("SerpApi Google Scholar enabled with API key" in item for item in logger.info_messages)
    assert any("HTTP auth token configured" in item for item in logger.info_messages)
    assert any("HTTP Origin allowlist configured for 2 entries" in item for item in logger.info_messages)
    assert any("Binding HTTP transport to remote.example" in item for item in logger.warning_messages)
    assert app.calls == [
        {
            "transport": "streamable-http",
            "host": "remote.example",
            "port": 8081,
            "path": "/custom-mcp",
        }
    ]


def test_deployment_runner_helpers_and_main_entrypoint(monkeypatch) -> None:
    recorded: list[tuple[str, str, int]] = []

    monkeypatch.setattr(
        deployment_runner,
        "run",
        lambda app, host, port: recorded.append((app, host, port)),
    )

    assert deployment_runner.resolve_bind_host({"PAPER_CHASER_HTTP_HOST": "  "}) == (deployment_runner.DEFAULT_HOST)
    assert deployment_runner.resolve_bind_port({"PORT": "  "}) == (deployment_runner.DEFAULT_PORT)

    deployment_runner.main()

    def _record_run(app: str, host: str, port: int) -> None:
        recorded.append((app, host, port))

    monkeypatch.setattr(
        "uvicorn.run",
        _record_run,
    )
    sys.modules.pop("paper_chaser_mcp.deployment_runner", None)
    runpy.run_module("paper_chaser_mcp.deployment_runner", run_name="__main__")

    assert recorded[0] == (
        "paper_chaser_mcp.deployment:app",
        deployment_runner.DEFAULT_HOST,
        deployment_runner.DEFAULT_PORT,
    )
    assert recorded[1] == recorded[0]


def test_parsing_helpers_and_tool_definitions_are_usable() -> None:
    assert _arxiv_id_from_url("") == ""
    assert _arxiv_id_from_url("https://arxiv.org/abs/2401.12345v2") == "2401.12345"
    assert _arxiv_id_from_url("not-an-arxiv-url") == "not-an-arxiv-url"
    assert _text(None) == ""
    assert _text(types.SimpleNamespace(text="  trimmed  ")) == "trimmed"

    tool_definitions = get_tool_definitions()
    expert_tool_definitions = get_tool_definitions(tool_profile="expert")

    assert tool_definitions
    assert {tool.name for tool in tool_definitions} == {
        "research",
        "follow_up_research",
        "resolve_reference",
        "inspect_source",
        "get_runtime_status",
    }
    assert any(tool.name == "search_papers" for tool in expert_tool_definitions)
    assert all(tool.description for tool in tool_definitions)
