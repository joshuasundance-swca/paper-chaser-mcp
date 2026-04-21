"""Smart-workflow leaf helpers hoisted out of ``graphs/_core``.

Phase 7c-3 moves six ``_core``-resident helpers into this module so Phase
7c-4 can Pattern-B-extract ``search_papers_smart`` into
:mod:`paper_chaser_mcp.agentic.graphs.smart_graph` without a cyclic import
between ``smart_graph`` and ``_core``. The canonical home for each helper
is now this module; ``_core`` re-imports the names at module top so legacy
call sites (including ``AgenticRuntime`` methods still living in ``_core``)
continue to resolve them via the existing module globals.

The hoist is a pure move — signatures, bodies, and runtime behaviour are
unchanged. None of the helpers import from ``_core``; each pulls its
dependencies directly from the model packages and sibling graphs modules so
importing ``smart_helpers`` cannot cycle back through ``_core``.
"""

from __future__ import annotations

from typing import Any

from ...models import CoverageSummary
from ...provider_runtime import provider_is_paywalled
from ..models import StructuredSourceRecord
from .source_records import _coverage_summary_line


__all__: list[str] = [
    "_best_next_internal_action",
    "_has_inspectable_sources",
    "_has_on_topic_sources",
    "_paid_providers_used",
    "_smart_coverage_summary",
    "_smart_provider_fallback_warnings",
]


def _paid_providers_used(providers: list[str]) -> list[str]:
    return sorted({provider for provider in providers if provider_is_paywalled(provider)})


def _smart_coverage_summary(
    *,
    providers_used: list[str],
    provider_outcomes: list[dict[str, Any]],
    search_mode: str,
    drift_warnings: list[str],
) -> CoverageSummary:
    attempted = [
        provider
        for provider in dict.fromkeys(
            [str(outcome.get("provider") or "").strip() for outcome in provider_outcomes if outcome.get("provider")]
            + list(providers_used)
        )
        if provider
    ]
    failed = [
        provider
        for provider in dict.fromkeys(
            str(outcome.get("provider") or "").strip()
            for outcome in provider_outcomes
            if str(outcome.get("statusBucket") or "") not in {"success", "empty", "skipped", ""}
        )
        if provider
    ]
    zero_results = [
        provider
        for provider in dict.fromkeys(
            str(outcome.get("provider") or "").strip()
            for outcome in provider_outcomes
            if str(outcome.get("statusBucket") or "") == "empty"
        )
        if provider
    ]
    # Invariant: providers_succeeded and providers_zero_results must be disjoint
    zero_results_set = set(zero_results)
    succeeded = [p for p in providers_used if p not in zero_results_set]
    return CoverageSummary(
        providersAttempted=attempted,
        providersSucceeded=succeeded,
        providersFailed=failed,
        providersZeroResults=zero_results,
        likelyCompleteness=("partial" if providers_used else ("incomplete" if failed else "unknown")),
        searchMode=search_mode,
        retrievalNotes=list(drift_warnings),
        summaryLine=_coverage_summary_line(
            attempted=attempted,
            failed=failed,
            zero_results=zero_results,
            likely_completeness=("partial" if providers_used else ("incomplete" if failed else "unknown")),
        ),
    )


def _has_inspectable_sources(records: list[StructuredSourceRecord]) -> bool:
    return any(
        record.topical_relevance != "off_topic"
        and bool(record.canonical_url or record.retrieved_url or record.full_text_url_found or record.abstract_observed)
        for record in records
    )


def _has_on_topic_sources(records: list[StructuredSourceRecord]) -> bool:
    return any(record.topical_relevance != "off_topic" for record in records)


def _best_next_internal_action(*, intent: str, has_sources: bool, result_status: str) -> str:
    if intent == "known_item":
        return "get_paper_details"
    if intent == "regulatory":
        return "inspect_source" if has_sources else "search_papers_smart"
    if has_sources:
        return "ask_result_set"
    if result_status == "partial":
        return "search_papers_smart"
    return "resolve_reference"


def _smart_provider_fallback_warnings(
    *,
    provider_selection: dict[str, Any],
    provider_outcomes: list[dict[str, Any]],
) -> list[str]:
    configured = str(provider_selection.get("configuredSmartProvider") or "").strip()
    active = str(provider_selection.get("activeSmartProvider") or "").strip()
    if not configured or not active or configured == active:
        return []

    endpoints = sorted(
        {
            str(outcome.get("endpoint") or "").strip()
            for outcome in provider_outcomes
            if str(outcome.get("provider") or "").strip() == configured
            and str(outcome.get("statusBucket") or "").strip() not in {"success", "empty", "skipped"}
            and str(outcome.get("endpoint") or "").strip()
        }
    )
    if endpoints:
        return [
            f"Smart provider '{configured}' fell back to deterministic mode after issues in {', '.join(endpoints)}; "
            "inspect providerOutcomes before trusting planning or expansion quality."
        ]
    return [
        f"Smart provider '{configured}' fell back to deterministic mode; inspect providerOutcomes before trusting "
        "planning or expansion quality."
    ]
