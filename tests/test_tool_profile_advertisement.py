"""Phase 1 pinning test: tool-list / profile advertisement contract.

Freezes the public tool surface exposed under each ``PAPER_CHASER_TOOL_PROFILE``
value and asserts:

* Under ``guided`` only the 5 guided tools are advertised.
* Under ``expert`` the 5 guided tools plus the expert surface (smart,
  result-set QA, landscape, graph, brokered/bulk search, provider-specific
  families, and regulatory primary-source tools) are advertised.
* ``PAPER_CHASER_HIDE_DISABLED_TOOLS`` masks provider-gated expert tools when
  the underlying enable-flags are off.
* Every advertised tool has a JSON-schema ``inputSchema`` with
  ``type: "object"`` and a ``properties`` dict (shape invariant).

Later phases that move code must not break this contract without updating
this pin.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from paper_chaser_mcp.settings import AppSettings
from paper_chaser_mcp.tools import get_tool_definitions

GUIDED_TOOLS: frozenset[str] = frozenset(
    {
        "research",
        "follow_up_research",
        "resolve_reference",
        "inspect_source",
        "get_runtime_status",
    }
)

EXPERT_SMART_TOOLS: frozenset[str] = frozenset(
    {
        "search_papers_smart",
        "ask_result_set",
        "map_research_landscape",
        "expand_research_graph",
    }
)

EXPERT_BROKERED_SEARCH_TOOLS: frozenset[str] = frozenset(
    {
        "search_papers",
        "search_papers_bulk",
    }
)

EXPERT_PROVIDER_TOOLS: frozenset[str] = frozenset(
    {
        "search_papers_core",
        "search_papers_semantic_scholar",
        "search_papers_openalex",
        "search_papers_arxiv",
        "search_papers_serpapi",
        "search_papers_scholarapi",
    }
)

EXPERT_REGULATORY_TOOLS: frozenset[str] = frozenset(
    {
        "search_federal_register",
        "get_federal_register_document",
        "get_cfr_text",
    }
)

ALL_PROVIDER_ENABLE_FLAGS: dict[str, bool] = {
    "enable_core": True,
    "enable_semantic_scholar": True,
    "enable_openalex": True,
    "enable_arxiv": True,
    "enable_serpapi": True,
    "enable_scholarapi": True,
    "enable_crossref": True,
    "enable_unpaywall": True,
    "enable_ecos": True,
    "enable_federal_register": True,
    "enable_govinfo_cfr": True,
    "enable_agentic": True,
    "govinfo_available": True,
}

NO_PROVIDER_ENABLE_FLAGS: dict[str, bool] = {key: False for key in ALL_PROVIDER_ENABLE_FLAGS}


def _settings_from_env_vars(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
) -> AppSettings:
    """Apply env vars via monkeypatch and rebuild settings via ``from_env``."""

    for key in (
        "PAPER_CHASER_TOOL_PROFILE",
        "PAPER_CHASER_HIDE_DISABLED_TOOLS",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return AppSettings.from_env()


@pytest.mark.parametrize(
    ("env_vars", "expected_profile", "expected_hide"),
    [
        ({"PAPER_CHASER_TOOL_PROFILE": "guided"}, "guided", True),
        (
            {
                "PAPER_CHASER_TOOL_PROFILE": "guided",
                "PAPER_CHASER_HIDE_DISABLED_TOOLS": "false",
            },
            "guided",
            False,
        ),
        ({"PAPER_CHASER_TOOL_PROFILE": "expert"}, "expert", False),
        (
            {
                "PAPER_CHASER_TOOL_PROFILE": "expert",
                "PAPER_CHASER_HIDE_DISABLED_TOOLS": "true",
            },
            "expert",
            True,
        ),
    ],
)
def test_settings_from_env_honors_profile_and_hide_flag(
    monkeypatch: pytest.MonkeyPatch,
    env_vars: dict[str, str],
    expected_profile: str,
    expected_hide: bool,
) -> None:
    settings = _settings_from_env_vars(monkeypatch, env_vars)
    assert settings.tool_profile == expected_profile
    assert settings.hide_disabled_tools is expected_hide


@pytest.mark.parametrize("hide_disabled", [False, True])
def test_guided_profile_advertises_exactly_five_tools(hide_disabled: bool) -> None:
    tools = get_tool_definitions(
        tool_profile="guided",
        hide_disabled_tools=hide_disabled,
        enabled_flags=NO_PROVIDER_ENABLE_FLAGS if hide_disabled else None,
    )
    names = {tool.name for tool in tools}
    assert names == GUIDED_TOOLS, (
        "Guided profile must advertise exactly the 5 guided tools regardless "
        f"of hide_disabled_tools={hide_disabled}; got extras="
        f"{sorted(names - GUIDED_TOOLS)} missing={sorted(GUIDED_TOOLS - names)}"
    )


def test_expert_profile_superset_includes_guided_and_expert_surface() -> None:
    tools = get_tool_definitions(
        tool_profile="expert",
        hide_disabled_tools=False,
    )
    names = {tool.name for tool in tools}
    assert GUIDED_TOOLS <= names
    assert EXPERT_SMART_TOOLS <= names
    assert EXPERT_BROKERED_SEARCH_TOOLS <= names
    assert EXPERT_PROVIDER_TOOLS <= names
    assert EXPERT_REGULATORY_TOOLS <= names
    # Expert profile is strictly larger than guided.
    assert names > GUIDED_TOOLS


def test_hide_disabled_tools_masks_provider_gated_expert_tools() -> None:
    visible_all = {
        tool.name
        for tool in get_tool_definitions(
            tool_profile="expert",
            hide_disabled_tools=True,
            enabled_flags=ALL_PROVIDER_ENABLE_FLAGS,
        )
    }
    visible_none = {
        tool.name
        for tool in get_tool_definitions(
            tool_profile="expert",
            hide_disabled_tools=True,
            enabled_flags=NO_PROVIDER_ENABLE_FLAGS,
        )
    }

    # Guided tools survive either way under expert profile.
    assert GUIDED_TOOLS <= visible_all
    assert GUIDED_TOOLS <= visible_none
    # Turning every provider flag off must strictly shrink the advertised set.
    assert visible_none < visible_all
    # Specifically, provider-gated expert tools should disappear.
    assert EXPERT_PROVIDER_TOOLS.isdisjoint(visible_none)
    assert EXPERT_SMART_TOOLS.isdisjoint(visible_none)
    assert EXPERT_REGULATORY_TOOLS.isdisjoint(visible_none)


@pytest.mark.parametrize("profile", ["guided", "expert"])
def test_every_advertised_tool_has_well_formed_input_schema(profile: str) -> None:
    tools = get_tool_definitions(
        tool_profile=profile,  # type: ignore[arg-type]
        hide_disabled_tools=False,
    )
    assert tools, f"profile={profile} advertised no tools"
    for tool in tools:
        assert tool.name, "tool missing name"
        assert tool.description, f"tool {tool.name!r} missing description"
        schema: dict[str, Any] = dict(tool.inputSchema)
        assert schema.get("type") == "object", f"tool {tool.name!r} inputSchema.type must be 'object'"
        properties = schema.get("properties")
        assert isinstance(properties, dict), (
            f"tool {tool.name!r} inputSchema.properties must be a dict, got {type(properties).__name__}"
        )


def test_fastmcp_server_list_tools_matches_guided_contract() -> None:
    """The live FastMCP server must advertise the same guided 5 tools.

    The module-level ``server.app`` is configured from the current process env;
    the CI baseline runs with the guided default, so we assert against that.
    """

    from fastmcp import Client

    from paper_chaser_mcp import server

    async def _collect() -> set[str]:
        async with Client(server.app) as client:
            tools = await client.list_tools()
        return {tool.name for tool in tools}

    if server.settings.tool_profile != "guided":
        pytest.skip("Live server not in guided profile; test validates the default CI baseline contract.")

    names = asyncio.run(_collect())
    assert GUIDED_TOOLS <= names
    # Under the guided default the live surface must not leak expert tools.
    assert not (names & (EXPERT_SMART_TOOLS | EXPERT_BROKERED_SEARCH_TOOLS)), (
        f"Guided server leaked expert tools: {sorted(names & (EXPERT_SMART_TOOLS | EXPERT_BROKERED_SEARCH_TOOLS))}"
    )
