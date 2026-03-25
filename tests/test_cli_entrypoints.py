import runpy
import sys
import types
from importlib.metadata import PackageNotFoundError

import scholar_search_mcp.cli as cli


def test_cli_version_fallback_and_parser_help(monkeypatch) -> None:
    monkeypatch.setattr(cli, "version", lambda _: (_ for _ in ()).throw(PackageNotFoundError()))

    parser = cli.build_parser()
    help_text = parser.format_help()

    assert cli._package_version() == "unknown"
    assert "deployment-http" in help_text
    assert "Run the package-local MCP server" in help_text


def test_cli_run_helpers_import_expected_entrypoints(monkeypatch) -> None:
    calls: list[str] = []

    server_module = types.ModuleType("scholar_search_mcp.server")
    server_module.main = lambda: calls.append("server")  # type: ignore[attr-defined]

    deployment_module = types.ModuleType("scholar_search_mcp.deployment_runner")
    deployment_module.main = lambda: calls.append("deployment")  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "scholar_search_mcp.server", server_module)
    monkeypatch.setitem(
        sys.modules,
        "scholar_search_mcp.deployment_runner",
        deployment_module,
    )

    cli._run_server()
    cli._run_deployment_http()

    assert calls == ["server", "deployment"]


def test_python_m_entrypoint_invokes_cli_main(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, "main", lambda *args: calls.append(args))
    sys.modules.pop("scholar_search_mcp.__main__", None)

    runpy.run_module("scholar_search_mcp.__main__", run_name="__main__")

    assert calls == [()]
