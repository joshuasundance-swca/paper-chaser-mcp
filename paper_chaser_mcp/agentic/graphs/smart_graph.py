"""Smart-search orchestration extracted in Phase 7c-2 from ``graphs/_core``.

Phase 7c-2 relocates the ``search_papers_smart`` orchestration body, its
smart-specific pure helpers, and the LangGraph ``StateGraph`` compilation
helper out of :mod:`paper_chaser_mcp.agentic.graphs._core` so the multiplexing
entry point and its LLM recovery routing live in a dedicated module.
``AgenticRuntime.search_papers_smart`` remains on the class as a thin delegate
(Pattern B — see the Phase 7c-2 prompt); the body is the module-level
``run_search_papers_smart`` coroutine. ``_maybe_compile_graphs`` is extracted
as a module-level function that takes ``runtime`` as a parameter (Pattern A).

LangGraph optional-dependency stubs (``START``, ``END``, ``StateGraph``,
``InMemorySaver``) are imported from :mod:`shared_state` rather than
``_core``; Phase 7a moved the single-source try/except into
``shared_state.py``, so importing from there preserves identity without
creating a back-edge to ``_core`` that would cycle during module loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...models import FailureSummary
from ..models import IntentLabel
from ..planner import normalize_query
from .shared_state import END, START, InMemorySaver, StateGraph

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._core import AgenticRuntime


__all__: list[str] = [
    "_dedupe_variants",
    "_initial_retrieval_query_text",
    "_result_coverage_label",
    "_smart_failure_summary",
    "maybe_compile_graphs",
]


def _initial_retrieval_query_text(*, normalized_query: str, focus: str | None, intent: IntentLabel) -> str:
    if intent in {"known_item", "author", "citation", "regulatory"}:
        return normalized_query
    normalized_focus = normalize_query(str(focus or ""))
    if not normalized_focus:
        return normalized_query
    combined = normalize_query(f"{normalized_query} {normalized_focus}")
    return combined if combined.lower() != normalized_query.lower() else normalized_query


def _result_coverage_label(candidates: list[dict[str, Any]]) -> str:
    if len(candidates) >= 20:
        return "broad"
    if len(candidates) >= 8:
        return "moderate"
    return "narrow"


def _dedupe_variants(variants: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        lowered = variant.strip().lower()
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(variant)
    return deduped


def _smart_failure_summary(
    *,
    provider_outcomes: list[dict[str, Any]],
    fallback_attempted: bool,
) -> FailureSummary | None:
    failures = [
        outcome
        for outcome in provider_outcomes
        if str(outcome.get("statusBucket") or "") not in {"success", "empty", "skipped", ""}
    ]
    if not failures:
        return None
    failed_providers = sorted({str(outcome.get("provider") or "unknown") for outcome in failures})
    return FailureSummary(
        outcome="fallback_success",
        whatFailed="One or more smart-search providers or provider-side stages failed.",
        whatStillWorked="The smart workflow returned the strongest available partial result set.",
        fallbackAttempted=fallback_attempted,
        fallbackMode="smart_provider_fallback",
        primaryPathFailureReason=", ".join(failed_providers),
        completenessImpact=(
            "Coverage may be partial because these providers or stages failed: " + ", ".join(failed_providers) + "."
        ),
        recommendedNextAction="review_partial_results",
    )


def maybe_compile_graphs(runtime: AgenticRuntime) -> dict[str, Any]:  # noqa: ARG001
    """Return compiled LangGraph placeholders when the optional dep is present.

    Pattern A extraction: takes ``runtime`` for forward-compat but does not yet
    depend on any per-instance state. Mirrors the legacy method body verbatim so
    behavior is unchanged.
    """

    if StateGraph is None or InMemorySaver is None:
        return {}
    compiled: dict[str, Any] = {}
    for graph_name in (
        "smart_search",
        "grounded_answer",
        "landscape_map",
        "graph_expand",
    ):
        graph = StateGraph(dict)
        graph.add_node("complete", lambda state: state)
        graph.add_edge(START, "complete")
        graph.add_edge("complete", END)
        compiled[graph_name] = graph.compile(checkpointer=InMemorySaver())
    return compiled
