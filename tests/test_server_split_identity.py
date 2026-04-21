"""Identity checks pinning the Phase 10 server subpackage layout.

These tests guard the Phase 10 extraction: the monolithic
``paper_chaser_mcp/server.py`` was split into a ``paper_chaser_mcp/server/``
subpackage. State, the FastMCP ``app``, ``_execute_tool``, runtime
initialization, ``main``, and the ``_require_*`` helpers live in the package
root (``__init__.py``); instructions, pure registration helpers, resource
factories, and prompt factories live in sibling submodules. The package
root remains the single public import path and preserves the monkeypatch
and ``importlib.reload`` contracts that the test suite depends on.
"""

from paper_chaser_mcp import server as _server
from paper_chaser_mcp.server import instructions as _instructions
from paper_chaser_mcp.server import registration as _registration
from paper_chaser_mcp.server import resources as _resources

_INSTRUCTION_NAMES = (
    "SERVER_INSTRUCTIONS",
    "GUIDED_SERVER_INSTRUCTIONS",
    "AGENT_WORKFLOW_GUIDE",
)

_REGISTRATION_NAMES = (
    "_format_tool_display_name",
    "_tool_tags",
    "_parameter_name",
    "_parameter_default",
    "_build_signature",
    "_sanitize_registered_tool_schema",
)

_RESOURCE_PAYLOAD_NAMES = (
    "_resource_text",
    "_paper_resource_payload",
    "_author_resource_payload",
)

_RESOURCE_DECORATED_NAMES = (
    "agent_workflows",
    "paper_resource",
    "author_resource",
    "search_session_resource",
    "paper_trail_resource",
)

_PROMPT_NAMES = (
    "plan_paper_chaser_search",
    "plan_smart_paper_chaser_search",
    "triage_literature",
    "plan_citation_chase",
    "refine_query",
)

_PACKAGE_ROOT_FUNCTIONS = (
    "_execute_tool",
    "_initialize_runtime",
    "_server_lifespan",
    "_configure_registered_tools",
    "_enabled_tool_flags",
    "_require_workspace_registry",
    "_require_semantic_client",
    "_require_openalex_client",
    "build_http_app",
    "list_tools",
    "call_tool",
    "main",
)


def test_instructions_module_owns_instruction_strings() -> None:
    for name in _INSTRUCTION_NAMES:
        value = getattr(_instructions, name)
        assert getattr(_server, name) is value, name


def test_registration_module_owns_pure_helpers() -> None:
    for name in _REGISTRATION_NAMES:
        value = getattr(_registration, name)
        assert getattr(_server, name) is value, name
        assert value.__module__ == "paper_chaser_mcp.server.registration", name


def test_resources_module_owns_resource_payload_helpers() -> None:
    for name in _RESOURCE_PAYLOAD_NAMES:
        value = getattr(_resources, name)
        assert getattr(_server, name) is value, name
        assert value.__module__ == "paper_chaser_mcp.server.resources", name


def test_resources_module_owns_decorated_resources() -> None:
    for name in _RESOURCE_DECORATED_NAMES:
        value = getattr(_server, name)
        assert callable(value), name
        assert value.__module__ == "paper_chaser_mcp.server.resources", name


def test_prompts_module_owns_prompt_factories() -> None:
    for name in _PROMPT_NAMES:
        value = getattr(_server, name)
        assert callable(value), name
        assert value.__module__ == "paper_chaser_mcp.server.prompts", name


def test_package_root_owns_runtime_and_lifecycle_symbols() -> None:
    """State, lifecycle, and ``main`` must live in ``server/__init__.py``.

    ``main.__module__`` is pinned by
    ``tests/test_server_singleton_contract.py``; this test additionally
    pins the surrounding runtime surface so that accidentally moving
    ``_execute_tool`` or ``_initialize_runtime`` into a submodule (which
    would break ``monkeypatch.setattr`` or ``importlib.reload``) fails
    loudly here.
    """

    for name in _PACKAGE_ROOT_FUNCTIONS:
        value = getattr(_server, name)
        assert callable(value), name
        assert value.__module__ == "paper_chaser_mcp.server", (
            f"{name!r} is defined in {value.__module__!r}; it must live in "
            "paper_chaser_mcp.server (the package __init__) to preserve the "
            "monkeypatch and importlib.reload contracts."
        )


def test_package_root_exposes_mutable_state_names() -> None:
    """Module-level state names monkeypatched by the test suite must exist
    on the package root so that ``monkeypatch.setattr(server, name, X)``
    is observed by ``_execute_tool`` and the registered tool wrappers.
    """

    state_names = (
        "client",
        "core_client",
        "openalex_client",
        "scholarapi_client",
        "serpapi_client",
        "crossref_client",
        "arxiv_client",
        "workspace_registry",
        "dispatch_tool",
        "http_app",
        "http_app_transport",
    )
    for name in state_names:
        assert hasattr(_server, name), name
