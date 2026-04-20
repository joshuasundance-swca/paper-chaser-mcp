"""Phase 1 guards: pin module-level singleton + import-time contract of server.py.

These tests protect the invariants consumers already rely on (cli.py, deployment.py,
scripts/run_expert_eval_batch.py, plus the package-level ``main`` re-export) so the
Phase 10 server split cannot silently break them.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import sys
from typing import Iterable, Iterator

import pytest

_PROVIDER_ENV_VARS = (
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "NVIDIA_API_KEY",
    "HUGGINGFACE_API_KEY",
    "OPENROUTER_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "CORE_API_KEY",
    "OPENALEX_API_KEY",
    "SERPAPI_API_KEY",
    "SCHOLARAPI_API_KEY",
    "GOVINFO_API_KEY",
    "CROSSREF_MAILTO",
    "UNPAYWALL_EMAIL",
    "PAPER_CHASER_TOOL_PROFILE",
    "PAPER_CHASER_ENABLE_EVAL_TRACE_CAPTURE",
    "PAPER_CHASER_EVAL_TRACE_PATH",
)


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch, names: Iterable[str] = _PROVIDER_ENV_VARS) -> None:
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_server_imports_without_any_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing ``paper_chaser_mcp.server`` must succeed with no provider env vars set."""

    _clear_provider_env(monkeypatch)
    server_module = importlib.import_module("paper_chaser_mcp.server")
    reloaded = importlib.reload(server_module)

    assert reloaded.app is not None
    assert reloaded.http_app is not None
    assert callable(reloaded.main)
    assert callable(reloaded.build_http_app)


def test_server_exposes_expected_module_attributes() -> None:
    from paper_chaser_mcp import server

    for attr in ("app", "http_app", "main", "build_http_app", "_execute_tool"):
        assert hasattr(server, attr), f"paper_chaser_mcp.server.{attr} missing"

    assert server.app is not None
    assert server.http_app is not None
    assert callable(server.main)
    assert callable(server.build_http_app)


def test_package_level_main_reexport_matches_server_main() -> None:
    import paper_chaser_mcp
    from paper_chaser_mcp import server

    assert hasattr(paper_chaser_mcp, "main")
    assert callable(paper_chaser_mcp.main)
    # Identity may break across importlib.reload in other tests in this session, so pin by
    # module + qualname instead. This still catches a rename or accidental removal.
    assert paper_chaser_mcp.main.__module__ == "paper_chaser_mcp.server"
    assert paper_chaser_mcp.main.__qualname__ == server.main.__qualname__ == "main"


def test_cli_references_server_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """cli._run_server() must delegate to paper_chaser_mcp.server.main without modification."""

    from paper_chaser_mcp import cli, server

    # Never run the real main; replace it on the server module before cli imports it.
    calls: list[str] = []
    monkeypatch.setattr(server, "main", lambda: calls.append("server-main"))

    cli._run_server()  # type: ignore[attr-defined]

    assert calls == ["server-main"]
    # And the original symbol is still callable on the server module for direct consumers.
    assert callable(getattr(importlib.import_module("paper_chaser_mcp.server"), "main"))


def test_deployment_references_server_build_http_app() -> None:
    from paper_chaser_mcp import deployment, server

    # deployment.py does: from .server import build_http_app
    assert "build_http_app" in vars(deployment) or getattr(deployment, "build_http_app", None) is server.build_http_app
    # The module-level `app` is a Starlette app built via build_http_app under the hood.
    assert deployment.app is not None


def test_execute_tool_private_helper_is_importable_for_scripts() -> None:
    """``scripts/run_expert_eval_batch.py`` imports ``server._execute_tool`` directly.

    Phase 10 should decide whether to promote this to a public name, but until then
    the private helper must stay importable at this exact path.
    """

    from paper_chaser_mcp.server import _execute_tool

    assert callable(_execute_tool)


async def test_guided_profile_advertises_only_guided_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under guided profile, list_tools exposes exactly the 5 guided entry points."""

    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("PAPER_CHASER_TOOL_PROFILE", "guided")

    server_module = importlib.import_module("paper_chaser_mcp.server")
    reloaded = importlib.reload(server_module)

    tools = await reloaded.list_tools()
    names = {tool.name for tool in tools}

    guided_expected = {
        "research",
        "follow_up_research",
        "resolve_reference",
        "inspect_source",
        "get_runtime_status",
    }
    assert names == guided_expected, f"guided profile tools drifted: {names}"


async def test_expert_profile_is_superset_of_guided(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under expert profile, list_tools exposes a strict superset of the guided 5."""

    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("PAPER_CHASER_TOOL_PROFILE", "expert")

    server_module = importlib.import_module("paper_chaser_mcp.server")
    reloaded = importlib.reload(server_module)

    tools = await reloaded.list_tools()
    names = {tool.name for tool in tools}

    guided_expected = {
        "research",
        "follow_up_research",
        "resolve_reference",
        "inspect_source",
        "get_runtime_status",
    }
    assert guided_expected.issubset(names), f"expert profile missing guided tools: {guided_expected - names}"
    assert len(names) > len(guided_expected), "expert profile should advertise more tools than guided"


_PACKAGE_MODULE_PREFIXES = ("paper_chaser_mcp",)


@contextlib.contextmanager
def _isolated_paper_chaser_import() -> Iterator[None]:
    """Evict ``paper_chaser_mcp`` (and submodules) from ``sys.modules`` and restore after."""

    snapshot = {
        name: module
        for name, module in sys.modules.items()
        if name == "paper_chaser_mcp" or name.startswith("paper_chaser_mcp.")
    }
    for name in list(snapshot):
        del sys.modules[name]
    try:
        yield
    finally:
        for name in [
            name for name in list(sys.modules) if name == "paper_chaser_mcp" or name.startswith("paper_chaser_mcp.")
        ]:
            del sys.modules[name]
        sys.modules.update(snapshot)


def test_main_delegates_under_fresh_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """``paper_chaser_mcp.main`` must resolve to the freshly-imported ``server.main``.

    Checking ``__module__``/``__qualname__`` on an already-imported module
    would happily pass even if Phase 10 left a stale alias behind. Re-importing
    from a clean ``sys.modules`` catches that regression.
    """

    _clear_provider_env(monkeypatch)

    with _isolated_paper_chaser_import():
        pkg = importlib.import_module("paper_chaser_mcp")
        server_module = importlib.import_module("paper_chaser_mcp.server")

        assert callable(pkg.main), "paper_chaser_mcp.main must be callable after a fresh import"
        assert pkg.main.__module__ == "paper_chaser_mcp.server", (
            f"paper_chaser_mcp.main is aliased from {pkg.main.__module__!r}, expected paper_chaser_mcp.server"
        )
        assert pkg.main.__qualname__ == server_module.main.__qualname__ == "main"
        # And the function object itself must match the server's entrypoint —
        # not just share a name.
        assert pkg.main is server_module.main, (
            "paper_chaser_mcp.main and paper_chaser_mcp.server.main diverged after a fresh import"
        )


def test_execute_tool_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """``server._execute_tool`` must return the documented ``get_runtime_status`` shape.

    Runs with every provider credential cleared so this test never leaks
    secrets from the ambient environment. Asserts only the top-level keys
    that ``get_runtime_status`` promises; the detailed shape is the dispatch
    layer's concern.
    """

    _clear_provider_env(monkeypatch)

    from paper_chaser_mcp.server import _execute_tool

    result = asyncio.run(_execute_tool("get_runtime_status", {}, ctx=None))

    assert isinstance(result, dict), f"_execute_tool should return a dict, got {type(result).__name__}"
    assert set(result.keys()) >= {
        "status",
        "runtimeSummary",
        "providerOrder",
        "providers",
        "warnings",
    }, f"unexpected top-level keys: {sorted(result.keys())}"
    assert result["status"] == "ok"
    assert isinstance(result["runtimeSummary"], dict)
    assert isinstance(result["providerOrder"], list)
    assert isinstance(result["providers"], list)
    assert isinstance(result["warnings"], list)


@pytest.mark.parametrize(
    ("profile", "expected_count"),
    [
        ("guided", 5),
        ("expert", 63),
    ],
)
def test_build_http_app_parity_guided_and_expert(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    expected_count: int,
) -> None:
    """Build a fresh HTTP app per profile and assert the advertised tool count.

    This guards the Phase 10 split: whatever module ends up owning
    ``build_http_app``/``list_tools`` must keep the guided/expert contract
    identical to today's behavior. The test reloads the server module instead
    of leaning on the ambient ``server.app`` singleton so a stale re-export
    cannot hide drift.
    """

    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("PAPER_CHASER_TOOL_PROFILE", profile)
    # Force every provider-gated tool to be advertised so the expert count
    # matches the snapshot in tests/test_tool_profile_advertisement.py.
    monkeypatch.setenv("PAPER_CHASER_HIDE_DISABLED_TOOLS", "false")

    with _isolated_paper_chaser_import():
        server_module = importlib.import_module("paper_chaser_mcp.server")

        # Exercise the build_http_app entrypoint so Phase 10 cannot quietly
        # drop it or return None.
        http_app = server_module.build_http_app()
        assert http_app is not None, f"build_http_app() returned None under profile={profile!r}"

        tools = asyncio.run(server_module.list_tools())
        names = {tool.name for tool in tools}

        assert len(names) == expected_count, (
            f"profile={profile!r} advertised {len(names)} tools, expected {expected_count}. names={sorted(names)}"
        )
