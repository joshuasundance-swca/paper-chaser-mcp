"""Phase 5 extraction pin: ``dispatch.runtime`` submodule.

Pins that runtime / provider-diagnostics helpers live at
``paper_chaser_mcp.dispatch.runtime`` after the Phase 5 modularization. Before
the extraction these imports fail (RED bar); after ``_core.py`` delegates to
the new submodule they go green. The existing facade contract
(``paper_chaser_mcp.dispatch._build_provider_diagnostics_snapshot``) must also
keep working via re-export from ``_core``.
"""

from __future__ import annotations

from typing import Any

import pytest

from paper_chaser_mcp.dispatch.runtime import (
    _annotate_runtime_provider_row,
    _build_provider_diagnostics_snapshot,
    _metadata_value_is_depleted,
    _provider_row_quota_limited,
    _runtime_provider_order,
    _smart_runtime_provider_state,
)


def test_runtime_provider_order_concatenates_sources() -> None:
    order = _runtime_provider_order(
        provider_order=["openalex", "arxiv"],  # type: ignore[list-item]
        smart_provider_order=["openai", "anthropic"],
    )
    # Raw broker order comes first, followed by the fixed default slate and
    # smart providers (inserted before the regulatory tail), then regulatory.
    assert order[:2] == ["openalex", "arxiv"]
    assert "openai" in order and "anthropic" in order
    assert order[-3:] == ["ecos", "federal_register", "govinfo"]


def test_runtime_provider_order_handles_none() -> None:
    order = _runtime_provider_order(provider_order=None, smart_provider_order=[])
    assert order[0] == "openalex"
    assert "scholarapi" in order
    assert order[-1] == "govinfo"


def test_smart_runtime_provider_state_defaults_when_runtime_missing() -> None:
    enabled, order, configured, active, settled = _smart_runtime_provider_state(None)
    assert enabled == {
        "openai": False,
        "azure-openai": False,
        "anthropic": False,
        "nvidia": False,
        "google": False,
        "mistral": False,
        "huggingface": False,
        "openrouter": False,
    }
    assert order[0] == "openai"
    assert configured is None
    assert active is None
    assert settled is True


class _FakeRuntime:
    def __init__(self, *, enabled: dict[str, bool], order: list[str], selection: dict[str, object]) -> None:
        self._enabled = enabled
        self._order = order
        self._provider_bundle = _FakeBundle(selection)

    def smart_provider_diagnostics(self) -> tuple[dict[str, bool], list[str]]:
        return self._enabled, self._order


class _FakeBundle:
    def __init__(self, selection: dict[str, object]) -> None:
        self._selection = selection

    def selection_metadata(self) -> dict[str, object]:
        return self._selection

    def provider_selection_settled(self) -> bool:
        return True


def test_smart_runtime_provider_state_reads_bundle_selection() -> None:
    runtime = _FakeRuntime(
        enabled={"openai": True, "anthropic": False},
        order=["openai", "anthropic"],
        selection={"configuredSmartProvider": "openai", "activeSmartProvider": "openai"},
    )
    enabled, order, configured, active, settled = _smart_runtime_provider_state(runtime)
    assert configured == "openai"
    assert active == "openai"
    assert settled is True
    assert enabled == {"openai": True, "anthropic": False}
    assert order == ["openai", "anthropic"]


@pytest.mark.parametrize(
    "value, depleted",
    [
        (0, True),
        (-3, True),
        (0.0, True),
        (5, False),
        ("0", True),
        ("-1", True),
        ("5", False),
        ("", False),
        ("not-a-number", False),
        (None, False),
        (True, False),
        (False, False),
    ],
)
def test_metadata_value_is_depleted(value: object, depleted: bool) -> None:
    assert _metadata_value_is_depleted(value) is depleted


def test_provider_row_quota_limited_via_outcome() -> None:
    assert _provider_row_quota_limited({"lastOutcome": "quota_exhausted"}) is True
    assert _provider_row_quota_limited({"lastOutcome": "success"}) is False


def test_provider_row_quota_limited_via_metadata() -> None:
    row: dict[str, Any] = {"lastQuotaMetadata": {"searches_left": 0}}
    assert _provider_row_quota_limited(row) is True
    row = {"lastQuotaMetadata": {"remainingRequests": 42}}
    assert _provider_row_quota_limited(row) is False
    row = {"lastQuotaMetadata": "not-a-dict"}
    assert _provider_row_quota_limited(row) is False


def test_annotate_runtime_provider_row_disabled() -> None:
    row: dict[str, object] = {"enabled": False}
    availability, health = _annotate_runtime_provider_row(row)
    assert availability == "disabled"
    assert health == "ok"
    assert row["runtimeAvailability"] == "disabled"
    assert row["runtimeHealth"] == "ok"
    assert "not currently enabled" in str(row["runtimeStateReason"])


def test_annotate_runtime_provider_row_active_quota_limited() -> None:
    row: dict[str, object] = {
        "enabled": True,
        "suppressed": False,
        "lastOutcome": "success",
        "lastQuotaMetadata": {"searches_left": 0},
    }
    availability, health = _annotate_runtime_provider_row(row)
    assert availability == "active"
    assert health == "quota_limited"


def test_annotate_runtime_provider_row_active_degraded_by_failures() -> None:
    row: dict[str, object] = {
        "enabled": True,
        "suppressed": False,
        "lastOutcome": "rate_limited",
        "consecutiveFailures": 3,
    }
    availability, health = _annotate_runtime_provider_row(row)
    assert availability == "active"
    assert health == "degraded"
    assert "rate_limited" in str(row["runtimeStateReason"])


def test_annotate_runtime_provider_row_suppressed() -> None:
    row: dict[str, object] = {"enabled": True, "suppressed": True}
    availability, health = _annotate_runtime_provider_row(row)
    assert availability == "suppressed"
    assert health == "ok"


def _diagnostics_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        include_recent_outcomes=False,
        provider_order=None,
        provider_registry=None,
        agentic_runtime=None,
        transport_mode="http",
        tool_profile="guided",
        hide_disabled_tools=False,
        session_ttl_seconds=None,
        embeddings_enabled=None,
        guided_research_latency_profile="standard",
        guided_follow_up_latency_profile="standard",
        guided_allow_paid_providers=False,
        guided_escalation_enabled=False,
        guided_escalation_max_passes=1,
        guided_escalation_allow_paid_providers=False,
        enable_core=True,
        enable_semantic_scholar=True,
        enable_openalex=True,
        enable_arxiv=True,
        enable_serpapi=False,
        enable_scholarapi=False,
        enable_crossref=True,
        enable_unpaywall=True,
        enable_ecos=True,
        enable_federal_register=True,
        enable_govinfo_cfr=True,
        ecos_client=None,
        serpapi_client=None,
        scholarapi_client=None,
    )
    base.update(overrides)
    return base


def test_build_provider_diagnostics_snapshot_no_registry_returns_summary() -> None:
    snapshot = _build_provider_diagnostics_snapshot(**_diagnostics_kwargs())  # type: ignore[arg-type]
    assert snapshot["generatedAt"] is None
    assert snapshot["providers"] == []
    assert isinstance(snapshot["providerOrder"], list)
    summary = snapshot["runtimeSummary"]
    assert summary["effectiveProfile"] == "guided"
    assert summary["transportMode"] == "http"
    assert summary["smartLayerEnabled"] is False
    # With no registry, active set mirrors configured set.
    assert set(summary["configuredProviderSet"]) == set(summary["activeProviderSet"])


def test_runtime_helpers_remain_reexported_from_core() -> None:
    """Moved helpers must still be reachable via ``_core`` + facade."""
    import paper_chaser_mcp.dispatch as facade
    from paper_chaser_mcp.dispatch import _core

    assert _core._runtime_provider_order is _runtime_provider_order
    assert _core._smart_runtime_provider_state is _smart_runtime_provider_state
    assert _core._build_provider_diagnostics_snapshot is _build_provider_diagnostics_snapshot
    assert _core._annotate_runtime_provider_row is _annotate_runtime_provider_row
    assert _core._provider_row_quota_limited is _provider_row_quota_limited
    assert _core._metadata_value_is_depleted is _metadata_value_is_depleted
    assert facade._build_provider_diagnostics_snapshot is _build_provider_diagnostics_snapshot
