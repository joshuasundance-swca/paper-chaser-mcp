"""Phase 1 guards: explicit regressions for stored-memory invariants.

These tests pin the behaviors that stored-memory items flagged as easy-to-regress
during the Phase 10 refactor:

- Runtime status response shape and provider-set internal consistency.
  (Guided-only tool advertisement filtering is covered in
  ``tests/test_server_singleton_contract.py`` -- do not duplicate here.)
- Eval-trace capture enablement + documented warning path in
  ``AppSettings.runtime_warnings()``.
- Retrieval-mode constants stay import-stable from ``paper_chaser_mcp.agentic.models``.
- ScholarAPI smart-layer admission via ``AgenticRuntime(enable_scholarapi=...)``.
"""

from __future__ import annotations

from typing import Any

import pytest

from paper_chaser_mcp import server
from paper_chaser_mcp.settings import AppSettings

# ---------------------------------------------------------------------------
# get_runtime_status: response shape + provider-set internal consistency
# ---------------------------------------------------------------------------


async def test_runtime_status_exposes_known_top_level_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_runtime_status` must surface the documented top-level shape."""

    # Deterministic: no LLM provider env vars, so smart-layer stays in deterministic fallback.
    for var in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    result = await server._execute_tool("get_runtime_status", {})

    assert isinstance(result, dict)
    for key in ("status", "runtimeSummary", "providerOrder", "providers", "warnings"):
        assert key in result, f"get_runtime_status missing top-level key: {key}"

    assert result["status"] == "ok"
    assert isinstance(result["runtimeSummary"], dict)
    assert isinstance(result["providers"], list)
    assert isinstance(result["providerOrder"], list)
    assert isinstance(result["warnings"], list)


async def test_runtime_summary_contains_smart_provider_and_provider_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`runtimeSummary` must expose the active/disabled provider sets + smart-provider fields."""

    for var in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    result = await server._execute_tool("get_runtime_status", {})
    summary = result["runtimeSummary"]

    for key in (
        "activeProviderSet",
        "disabledProviderSet",
        "configuredSmartProvider",
        "activeSmartProvider",
        "providerOrderEffective",
        "healthStatus",
    ):
        assert key in summary, f"runtimeSummary missing {key}"

    assert isinstance(summary["activeProviderSet"], list)
    assert isinstance(summary["disabledProviderSet"], list)


async def test_runtime_provider_sets_are_internally_consistent_with_detailed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """activeProviderSet / disabledProviderSet must agree with per-provider detail rows."""

    for var in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    result = await server._execute_tool("get_runtime_status", {})
    summary = result["runtimeSummary"]
    rows_by_name = {row["provider"]: row for row in result["providers"] if isinstance(row, dict)}

    active_set = set(summary["activeProviderSet"])
    disabled_set = set(summary["disabledProviderSet"])

    # Active and disabled sets must be disjoint.
    assert not (active_set & disabled_set), f"providers cannot be both active and disabled: {active_set & disabled_set}"

    # Every provider named in either set must have a corresponding detail row.
    for name in active_set | disabled_set:
        assert name in rows_by_name, f"provider {name!r} appears in active/disabled set but has no detail row"

    # Disabled rows must not claim enabled=True.
    for name in disabled_set:
        row = rows_by_name[name]
        assert row.get("enabled") is not True, (
            f"provider {name!r} is in disabledProviderSet but row reports enabled=True"
        )


# ---------------------------------------------------------------------------
# Eval-trace capture enablement + documented warning path
# ---------------------------------------------------------------------------


_EVAL_TRACE_WARNING_FRAGMENT = "Eval trace capture is enabled without PAPER_CHASER_EVAL_TRACE_PATH"


@pytest.mark.parametrize(
    ("enable_capture", "trace_path", "expect_warning", "expect_capture_enabled"),
    [
        (None, None, False, False),
        ("1", None, True, True),
        ("1", "/tmp/trace.jsonl", False, True),
        (None, "/tmp/trace.jsonl", False, False),
    ],
)
def test_eval_trace_capture_warning_path(
    enable_capture: str | None,
    trace_path: str | None,
    expect_warning: bool,
    expect_capture_enabled: bool,
) -> None:
    env: dict[str, str] = {}
    if enable_capture is not None:
        env["PAPER_CHASER_ENABLE_EVAL_TRACE_CAPTURE"] = enable_capture
    if trace_path is not None:
        env["PAPER_CHASER_EVAL_TRACE_PATH"] = trace_path

    settings = AppSettings.from_env(env)
    warnings = settings.runtime_warnings()

    assert settings.enable_eval_trace_capture is expect_capture_enabled
    warning_present = any(_EVAL_TRACE_WARNING_FRAGMENT in w for w in warnings)
    assert warning_present is expect_warning, f"eval-trace warning presence mismatch: env={env}, warnings={warnings}"


# ---------------------------------------------------------------------------
# Retrieval-mode constants stay import-stable from agentic/models
# ---------------------------------------------------------------------------


def test_retrieval_mode_constants_importable_and_stable() -> None:
    """Bandit false-positive neutral retrieval-mode constants must keep their string values."""

    from paper_chaser_mcp.agentic.models import (
        RETRIEVAL_MODE_BROAD,
        RETRIEVAL_MODE_MIXED,
        RETRIEVAL_MODE_TARGETED,
    )

    assert RETRIEVAL_MODE_TARGETED == "targeted"
    assert RETRIEVAL_MODE_BROAD == "broad"
    assert RETRIEVAL_MODE_MIXED == "mixed"

    # first_pass_mode fields in PlannerFirstPassMode models use these exact string values.
    from paper_chaser_mcp.agentic import models as agentic_models

    assert hasattr(agentic_models, "RETRIEVAL_MODE_TARGETED")
    assert hasattr(agentic_models, "RETRIEVAL_MODE_BROAD")
    assert hasattr(agentic_models, "RETRIEVAL_MODE_MIXED")


# ---------------------------------------------------------------------------
# ScholarAPI smart-layer admission
# ---------------------------------------------------------------------------


def _minimal_agentic_runtime_kwargs(enable_scholarapi: bool) -> dict[str, Any]:
    from paper_chaser_mcp.agentic import (
        AgenticConfig,
        WorkspaceRegistry,
        resolve_provider_bundle,
    )

    settings = AppSettings.from_env({})
    config = AgenticConfig.from_settings(settings)
    bundle = resolve_provider_bundle(
        config=config,
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_http_referer=None,
        openrouter_title=None,
        nvidia_api_key=None,
        nvidia_nim_base_url=None,
        azure_openai_api_key=None,
        azure_openai_endpoint=None,
        azure_openai_api_version=None,
        azure_openai_planner_deployment=None,
        azure_openai_synthesis_deployment=None,
        anthropic_api_key=None,
        google_api_key=None,
        mistral_api_key=None,
        huggingface_api_key=None,
    )
    workspace = WorkspaceRegistry(ttl_seconds=60, enable_trace_log=False)

    return dict(
        config=config,
        provider_bundle=bundle,
        workspace_registry=workspace,
        client=None,
        core_client=None,
        openalex_client=None,
        scholarapi_client=None,
        arxiv_client=None,
        serpapi_client=None,
        enable_core=False,
        enable_semantic_scholar=False,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        enable_scholarapi=enable_scholarapi,
    )


@pytest.mark.parametrize("enable_scholarapi", [True, False])
def test_agentic_runtime_persists_enable_scholarapi_kwarg(enable_scholarapi: bool) -> None:
    """When ``enable_scholarapi`` is passed, AgenticRuntime must persist it verbatim.

    This is the minimal smart-layer admission guard from stored memory: the planner's
    providerPlan admits ScholarAPI only when this flag is True.
    """

    from paper_chaser_mcp.agentic import AgenticRuntime

    runtime = AgenticRuntime(**_minimal_agentic_runtime_kwargs(enable_scholarapi))

    assert runtime._enable_scholarapi is enable_scholarapi
