"""Smart-search orchestration extracted in Phase 7c-2/7c-4 from ``graphs/_core``.

Phase 7c-2 relocated the ``search_papers_smart`` orchestration's smart-specific
pure helpers (``_dedupe_variants``, ``_initial_retrieval_query_text``,
``_result_coverage_label``, ``_smart_failure_summary``) and the LangGraph
``StateGraph`` compilation helper out of
:mod:`paper_chaser_mcp.agentic.graphs._core`. Phase 7c-4 completes the
Pattern B extraction by moving the full ``search_papers_smart`` method body
into the module-level ``run_search_papers_smart`` coroutine. The method on
``AgenticRuntime`` remains the public call-site and is a thin one-line
delegate: ``return await run_search_papers_smart(self, ...)``. This preserves
the historical public signature byte-for-byte while making the orchestration
independently testable without routing through a method.

``_maybe_compile_graphs`` stays as ``maybe_compile_graphs(runtime)`` — a
Pattern A extraction that takes the runtime explicitly but does not yet
depend on per-instance state.

LangGraph optional-dependency stubs (``START``, ``END``, ``StateGraph``,
``InMemorySaver``) are imported from :mod:`shared_state` rather than
``_core``; Phase 7a moved the single-source try/except into
``shared_state.py``, so importing from there preserves identity without
creating a back-edge to ``_core`` that would cycle during module loading.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from fastmcp import Context

from ...compat import build_agent_hints, build_resource_uris
from ...models import FailureSummary, Paper, dump_jsonable
from ...provider_runtime import ProviderBudgetState
from ..config import LatencyProfile
from ..models import (
    IntentLabel,
    ScoreBreakdown,
    SearchStrategyMetadata,
    SmartPaperHit,
    SmartSearchResponse,
)
from ..planner import (
    combine_variants,
    dedupe_variants,
    grounded_expansion_candidates,
    initial_retrieval_hypotheses,
    normalize_query,
    speculative_expansion_candidates,
)
from ..ranking import (
    evaluate_speculative_variants,
    merge_candidates,
    rerank_candidates,
    summarize_ranking_diagnostics,
)
from ..retrieval import RetrievalBatch, RetrievedCandidate, retrieve_variant
from ..workspace import ExpiredSearchSessionError, SearchSessionNotFoundError
from .hooks import _truncate_text
from .regulatory_routing import _derive_regulatory_query_flags
from .resolve_graph import _normalization_metadata
from .shared_state import END, START, InMemorySaver, StateGraph
from .smart_helpers import (
    _best_next_internal_action,
    _has_inspectable_sources,
    _has_on_topic_sources,
    _paid_providers_used,
    _smart_coverage_summary,
    _smart_provider_fallback_warnings,
)
from .source_records import (
    _answerability_from_source_records,
    _candidate_leads_from_source_records,
    _classify_topical_relevance_for_paper,
    _classify_topical_relevance_with_provenance,
    _evidence_from_source_records,
    _likely_unverified_from_source_records,
    _routing_summary_from_strategy,
    _source_record_from_paper,
    _verified_findings_from_source_records,
    _why_matched,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._core import AgenticRuntime


logger = logging.getLogger("paper-chaser-mcp")


__all__: list[str] = [
    "_dedupe_variants",
    "_initial_retrieval_query_text",
    "_result_coverage_label",
    "_smart_failure_summary",
    "maybe_compile_graphs",
    "run_search_papers_smart",
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


async def run_search_papers_smart(
    runtime: AgenticRuntime,
    *,
    query: str,
    limit: int,
    search_session_id: str | None = None,
    mode: str = "auto",
    year: str | None = None,
    venue: str | None = None,
    focus: str | None = None,
    latency_profile: LatencyProfile = "deep",
    provider_budget: dict[str, Any] | None = None,
    include_enrichment: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Smart concept-level discovery with grounded expansion and fusion."""
    from . import _core  # noqa: PLC0415 - defer for test-monkeypatch visibility

    if not runtime._config.enabled:
        return runtime._feature_not_configured(
            "Smart workflows are disabled. Set PAPER_CHASER_ENABLE_AGENTIC=true to use search_papers_smart.",
            fallback_tools=[
                "search_papers",
                "search_papers_bulk",
                "search_papers_match",
            ],
        )

    profile_settings = runtime._config.latency_profile_settings(latency_profile)
    provider_bundle = runtime._provider_bundle_for_profile(latency_profile)
    budget_state = ProviderBudgetState.from_mapping(provider_budget)
    request_id = f"smart-{uuid4().hex[:10]}"
    provider_outcomes: list[dict[str, Any]] = []
    stage_timings_ms: dict[str, int] = {}

    def _finish_stage(stage_name: str, started_at: float) -> None:
        stage_timings_ms[stage_name] = int((time.perf_counter() - started_at) * 1000)

    planning_started = time.perf_counter()
    normalized_query, planner = await _core.classify_query(  # type: ignore[attr-defined]
        query=query,
        mode=mode,
        year=year,
        venue=venue,
        focus=focus,
        provider_bundle=provider_bundle,
        request_outcomes=provider_outcomes,
        request_id=request_id,
    )
    normalization_warnings, repaired_inputs = _normalization_metadata(query, normalized_query)
    _finish_stage("planning", planning_started)
    await runtime._emit_smart_search_status(
        ctx=ctx,
        request_id=request_id,
        progress=5,
        message="Planning smart search",
        detail=(
            f"Intent '{planner.intent}' selected for "
            f"'{_truncate_text(normalized_query, limit=96)}' with "
            f"latency profile '{latency_profile}'."
        ),
    )

    if planner.intent == "known_item":
        return await runtime._search_known_item(
            query=normalized_query,
            limit=limit,
            planner=planner,
            provider_plan=planner.provider_plan or None,
            provider_budget=budget_state,
            search_session_id=search_session_id,
            latency_profile=latency_profile,
            request_id=request_id,
            include_enrichment=include_enrichment,
            provider_outcomes=provider_outcomes,
            stage_timings_ms=stage_timings_ms,
            normalization_warnings=normalization_warnings,
            repaired_inputs=repaired_inputs,
            ctx=ctx,
        )
    if planner.intent == "regulatory":
        regulatory_result = await runtime._search_regulatory(
            query=normalized_query,
            limit=limit,
            planner_intent=planner.intent,
            planner=planner,
            search_session_id=search_session_id,
            latency_profile=latency_profile,
            request_id=request_id,
            provider_outcomes=provider_outcomes,
            stage_timings_ms=stage_timings_ms,
            normalization_warnings=normalization_warnings,
            repaired_inputs=repaired_inputs,
            ctx=ctx,
        )
        if regulatory_result.get("structuredSources"):
            return regulatory_result
        _, _, _agency_guidance_route = _derive_regulatory_query_flags(query=normalized_query, planner=planner)
        if _agency_guidance_route:
            return regulatory_result

        # LLM-driven strategy revision — single recovery attempt instead of hard-coded branching.
        revision = await provider_bundle.arevise_search_strategy(
            original_query=normalized_query,
            original_intent="regulatory",
            tried_providers=sorted({str(o.get("provider") or "") for o in provider_outcomes if isinstance(o, dict)}),
            result_summary=(
                f"Regulatory routing returned no on-topic sources for '{_truncate_text(normalized_query, limit=96)}'."
            ),
            request_id=request_id,
        )
        revised_query = str(revision.get("revisedQuery") or normalized_query)
        revised_intent = str(revision.get("revisedIntent") or "review")
        revision_rationale = str(revision.get("rationale") or "")
        await runtime._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=22,
            message="Recovering from empty regulatory route",
            detail=(
                f"LLM revised strategy: intent='{revised_intent}', "
                f"query='{_truncate_text(revised_query, limit=80)}'. "
                f"Reason: {_truncate_text(revision_rationale, limit=120)}"
            ),
        )
        recovered = await runtime.search_papers_smart(
            query=revised_query,
            limit=limit,
            search_session_id=search_session_id,
            mode=revised_intent if revised_intent != "regulatory" else "review",
            year=year,
            venue=venue,
            focus=focus,
            latency_profile=latency_profile,
            provider_budget=provider_budget,
            include_enrichment=include_enrichment,
            ctx=ctx,
        )
        strategy_metadata = recovered.get("strategyMetadata") if isinstance(recovered, dict) else None
        if isinstance(strategy_metadata, dict):
            warnings = list(strategy_metadata.get("driftWarnings") or [])
            warnings.append(
                f"Regulatory routing returned no on-topic sources. LLM revised strategy to "
                f"intent='{revised_intent}': {_truncate_text(revision_rationale, limit=160)}"
            )
            strategy_metadata["driftWarnings"] = list(dict.fromkeys(warnings))
            strategy_metadata["intentSource"] = "fallback_recovery"
            strategy_metadata["intentConfidence"] = "medium"
            strategy_metadata["recoveryAttempted"] = True
            strategy_metadata["recoveryPath"] = ["regulatory", revised_intent]
            strategy_metadata["recoveryReason"] = (
                revision_rationale or "Regulatory retrieval returned no on-topic sources."
            )
        return recovered

    initial_candidates = initial_retrieval_hypotheses(
        normalized_query=normalized_query,
        focus=focus,
        planner=planner,
        config=profile_settings.search_config,
    )
    retrieval_hypotheses = list(planner.retrieval_hypotheses or planner.search_angles)

    await runtime._emit_smart_search_status(
        ctx=ctx,
        request_id=request_id,
        progress=15,
        message="Running initial retrieval",
        detail=(
            f"Searching {len(initial_candidates)} bounded initial retrieval path(s) for "
            f"'{_truncate_text(normalized_query, limit=96)}'."
        ),
    )
    first_retrieval_started = time.perf_counter()
    initial_tasks = [
        asyncio.create_task(
            retrieve_variant(
                variant=candidate.variant,
                variant_source=candidate.source,
                intent=planner.intent,
                year=year,
                venue=venue,
                enable_core=runtime._enable_core,
                enable_semantic_scholar=runtime._enable_semantic_scholar,
                enable_openalex=runtime._enable_openalex,
                enable_scholarapi=runtime._enable_scholarapi,
                enable_arxiv=runtime._enable_arxiv,
                enable_serpapi=runtime._enable_serpapi,
                core_client=runtime._core_client,
                semantic_client=runtime._client,
                openalex_client=runtime._openalex_client,
                scholarapi_client=runtime._scholarapi_client,
                arxiv_client=runtime._arxiv_client,
                serpapi_client=runtime._serpapi_client,
                provider_plan=(candidate.provider_plan or planner.provider_plan or None),
                widened=planner.intent == "review",
                is_expansion=False,
                allow_serpapi=(runtime._enable_serpapi and profile_settings.allow_serpapi_on_input),
                latency_profile=latency_profile,
                provider_registry=runtime._provider_registry,
                provider_budget=budget_state,
                request_outcomes=provider_outcomes,
                request_id=request_id,
            )
        )
        for candidate in initial_candidates
    ]
    initial_batches = await asyncio.gather(*initial_tasks)
    _finish_stage("firstRetrieval", first_retrieval_started)
    first_batch = initial_batches[0]
    await runtime._emit_smart_search_status(
        ctx=ctx,
        request_id=request_id,
        progress=30,
        message="Initial retrieval complete",
        detail=(
            f"Completed {len(initial_batches)} initial retrieval path(s). "
            f"Primary path: {runtime._describe_retrieval_batch(first_batch)}"
        ),
    )
    first_pass_papers = [candidate.paper for batch in initial_batches for candidate in batch.candidates]
    if search_session_id:
        try:
            prior_record = runtime._workspace_registry.get(search_session_id)
            first_pass_papers = list(prior_record.papers) + first_pass_papers
        except (SearchSessionNotFoundError, ExpiredSearchSessionError):
            pass

    grounded = await grounded_expansion_candidates(
        original_query=normalized_query,
        papers=first_pass_papers,
        config=profile_settings.search_config,
        provider_bundle=provider_bundle,
        focus=focus,
        venue=venue,
        year=year,
    )
    recommendation_task: asyncio.Task[list[RetrievedCandidate]] | None = None
    recommendation_started: float | None = None
    if profile_settings.enable_deep_recommendations:
        recommendation_started = time.perf_counter()
        recommendation_task = asyncio.create_task(
            runtime._semantic_recommendation_candidates(
                seed_candidates=[candidate for batch in initial_batches for candidate in batch.candidates],
                normalized_query=normalized_query,
                enabled=True,
                request_id=request_id,
                provider_outcomes=provider_outcomes,
                provider_budget=budget_state,
            )
        )

    grounded_variants = [
        candidate for candidate in grounded if candidate.variant.strip().lower() != normalized_query.lower()
    ]
    grounded_tasks: list[asyncio.Task[RetrievalBatch]] = []
    grounded_retrieval_started: float | None = None

    speculative_generation_started = time.perf_counter()
    speculative = (
        await speculative_expansion_candidates(
            original_query=normalized_query,
            papers=first_pass_papers,
            config=profile_settings.search_config,
            provider_bundle=provider_bundle,
            request_outcomes=provider_outcomes,
            request_id=request_id,
        )
        if profile_settings.enable_speculative_expansions
        else []
    )
    _finish_stage(
        "speculativeExpansionGeneration",
        speculative_generation_started,
    )

    grounded_variants_by_query = {candidate.variant.strip().lower() for candidate in grounded_variants}
    speculative_variants = [
        candidate
        for candidate in speculative
        if candidate.variant.strip().lower() != normalized_query.lower()
        and candidate.variant.strip().lower() not in grounded_variants_by_query
    ]
    variants = combine_variants(
        original_query=normalized_query,
        grounded=grounded,
        speculative=speculative_variants,
        config=profile_settings.search_config,
    )

    initial_variant_keys = {candidate.variant.strip().lower() for candidate in initial_candidates}

    expansion_variants = [
        candidate for candidate in variants[1:] if candidate.variant.strip().lower() not in initial_variant_keys
    ]
    grounded_variants = [
        candidate
        for candidate in dedupe_variants(
            grounded_variants,
            config=profile_settings.search_config,
        )
        if candidate.variant.strip().lower() in {variant.variant.strip().lower() for variant in expansion_variants}
    ]
    grounded_count = len([candidate for candidate in expansion_variants if candidate.source != "speculative"])
    speculative_count = len(expansion_variants) - grounded_count
    expansion_preview = ", ".join(
        f"'{_truncate_text(candidate.variant, limit=48)}'" for candidate in expansion_variants[:3]
    )
    if expansion_variants:
        await runtime._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=40,
            message="Expansion plan ready",
            detail=(
                f"Prepared {len(expansion_variants)} expansion variant(s): "
                f"{grounded_count} grounded and {speculative_count} speculative. "
                f"Preview: {expansion_preview}."
            ),
        )
        if grounded_variants:
            grounded_retrieval_started = time.perf_counter()
            grounded_tasks = [
                asyncio.create_task(
                    retrieve_variant(
                        variant=candidate.variant,
                        variant_source=candidate.source,
                        intent=planner.intent,
                        year=year,
                        venue=venue,
                        enable_core=runtime._enable_core,
                        enable_semantic_scholar=runtime._enable_semantic_scholar,
                        enable_openalex=runtime._enable_openalex,
                        enable_scholarapi=runtime._enable_scholarapi,
                        enable_arxiv=runtime._enable_arxiv,
                        enable_serpapi=runtime._enable_serpapi,
                        core_client=runtime._core_client,
                        semantic_client=runtime._client,
                        openalex_client=runtime._openalex_client,
                        scholarapi_client=runtime._scholarapi_client,
                        arxiv_client=runtime._arxiv_client,
                        serpapi_client=runtime._serpapi_client,
                        provider_plan=(candidate.provider_plan or planner.provider_plan or None),
                        widened=planner.intent == "review",
                        is_expansion=True,
                        allow_serpapi=(runtime._enable_serpapi and profile_settings.allow_serpapi_on_expansions),
                        latency_profile=latency_profile,
                        provider_registry=runtime._provider_registry,
                        provider_budget=budget_state,
                        request_outcomes=provider_outcomes,
                        request_id=request_id,
                    )
                )
                for candidate in grounded_variants
            ]
            await runtime._emit_smart_search_status(
                ctx=ctx,
                request_id=request_id,
                progress=45,
                message="Running grounded expansions",
                detail=(
                    f"Running {len(grounded_variants)} "
                    "grounded expansion variant(s) "
                    "while speculative planning continues in parallel."
                ),
            )
    else:
        await runtime._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=70,
            message="No grounded expansions to run",
            detail=(
                "Initial retrieval stayed close to the query, so no additional "
                "grounded or speculative variants were scheduled."
            ),
        )

    speculative_tasks: list[asyncio.Task[RetrievalBatch]] = []
    speculative_retrieval_started: float | None = None
    if speculative_variants:
        speculative_retrieval_started = time.perf_counter()
        speculative_tasks = [
            asyncio.create_task(
                retrieve_variant(
                    variant=candidate.variant,
                    variant_source=candidate.source,
                    intent=planner.intent,
                    year=year,
                    venue=venue,
                    enable_core=runtime._enable_core,
                    enable_semantic_scholar=runtime._enable_semantic_scholar,
                    enable_openalex=runtime._enable_openalex,
                    enable_scholarapi=runtime._enable_scholarapi,
                    enable_arxiv=runtime._enable_arxiv,
                    enable_serpapi=runtime._enable_serpapi,
                    core_client=runtime._core_client,
                    semantic_client=runtime._client,
                    openalex_client=runtime._openalex_client,
                    scholarapi_client=runtime._scholarapi_client,
                    arxiv_client=runtime._arxiv_client,
                    serpapi_client=runtime._serpapi_client,
                    provider_plan=(candidate.provider_plan or planner.provider_plan or None),
                    widened=planner.intent == "review",
                    is_expansion=True,
                    allow_serpapi=(runtime._enable_serpapi and profile_settings.allow_serpapi_on_expansions),
                    latency_profile=latency_profile,
                    provider_registry=runtime._provider_registry,
                    provider_budget=budget_state,
                    request_outcomes=provider_outcomes,
                    request_id=request_id,
                )
            )
            for candidate in speculative_variants
        ]

    remaining_batches: list[RetrievalBatch] = []
    expansion_tasks = [*grounded_tasks, *speculative_tasks]
    if expansion_tasks:
        try:
            for completed_index, task in enumerate(
                asyncio.as_completed(expansion_tasks),
                start=1,
            ):
                batch = await task
                remaining_batches.append(batch)
                expansion_progress = 45 + ((completed_index / len(expansion_tasks)) * 25)
                await runtime._emit_smart_search_status(
                    ctx=ctx,
                    request_id=request_id,
                    progress=expansion_progress,
                    message=(f"Expansion {completed_index}/{len(expansion_tasks)} complete"),
                    detail=runtime._describe_retrieval_batch(batch),
                )
        except Exception:
            for task in expansion_tasks:
                task.cancel()
            await asyncio.gather(*expansion_tasks, return_exceptions=True)
            if recommendation_task is not None:
                recommendation_task.cancel()
                await asyncio.gather(
                    recommendation_task,
                    return_exceptions=True,
                )
            raise
        if grounded_tasks and grounded_retrieval_started is not None:
            _finish_stage("groundedRetrieval", grounded_retrieval_started)
        if speculative_tasks and speculative_retrieval_started is not None:
            _finish_stage("speculativeRetrieval", speculative_retrieval_started)

    recommendation_candidates: list[RetrievedCandidate] = []
    if recommendation_task is not None:
        recommendation_candidates = await recommendation_task
        if recommendation_started is not None:
            _finish_stage("recommendationRetrieval", recommendation_started)
        await runtime._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=78,
            message="Recommendation fetch complete",
            detail=(
                f"Semantic Scholar recommendation expansion returned {len(recommendation_candidates)} candidate(s)."
            ),
        )

    all_candidates = [candidate for batch in initial_batches for candidate in batch.candidates]
    for batch in remaining_batches:
        all_candidates.extend(batch.candidates)
    all_candidates.extend(recommendation_candidates)

    merged = merge_candidates(all_candidates)
    rerank_started = time.perf_counter()
    rerank_bundle = (
        provider_bundle
        if profile_settings.use_embedding_rerank and provider_bundle.supports_embeddings()
        else runtime._deterministic_bundle
    )
    candidate_pool_size = profile_settings.search_config.candidate_pool_size
    if planner.query_specificity == "low" or planner.ambiguity_level != "low":
        candidate_pool_size = min(candidate_pool_size + 20, 160)
    reranked = await rerank_candidates(
        query=normalized_query,
        merged_candidates=merged,
        provider_bundle=rerank_bundle,
        candidate_concepts=planner.candidate_concepts,
        routing_confidence=planner.routing_confidence,
        query_specificity=planner.query_specificity,
        ambiguity_level=planner.ambiguity_level,
        candidate_pool_size=candidate_pool_size,
        request_outcomes=provider_outcomes,
        request_id=request_id,
        planner_anchor_type=getattr(planner, "anchor_type", None),
        planner_anchor_value=getattr(planner, "anchor_value", None),
    )
    _finish_stage("rerank", rerank_started)
    (
        accepted_speculative,
        rejected_speculative,
        drift_warnings,
    ) = evaluate_speculative_variants(
        ranked_candidates=reranked,
        config=profile_settings.search_config,
    )
    filtered_ranked = [
        candidate
        for candidate in reranked
        if not (
            "speculative" in candidate["variantSources"]
            and set(candidate["variants"]).issubset(set(rejected_speculative))
        )
    ]

    await runtime._emit_smart_search_status(
        ctx=ctx,
        request_id=request_id,
        progress=80,
        message="Reranking and deduplicating papers",
        detail=(f"Fusing {len(all_candidates)} candidate(s) into {len(filtered_ranked)} unique ranked paper(s)."),
    )

    if len(filtered_ranked) < 2 and planner.intent_source != "fallback_recovery":
        revision = await provider_bundle.arevise_search_strategy(
            original_query=normalized_query,
            original_intent=planner.intent,
            tried_providers=sorted(
                {
                    str(candidate.get("provider") or "").strip()
                    for candidate in reranked
                    if str(candidate.get("provider") or "").strip()
                }
            ),
            result_summary=(
                f"Smart search routed as {planner.intent} returned only {len(filtered_ranked)} "
                f"ranked candidate(s) for '{_truncate_text(normalized_query, limit=96)}'."
            ),
            request_id=request_id,
        )
        revised_query = normalize_query(str(revision.get("revisedQuery") or normalized_query))
        revised_intent = str(revision.get("revisedIntent") or planner.intent)
        revision_rationale = str(revision.get("rationale") or "").strip()
        if revised_query.lower() != normalized_query.lower() or revised_intent != planner.intent:
            await runtime._emit_smart_search_status(
                ctx=ctx,
                request_id=request_id,
                progress=82,
                message="Recovering from low-result route",
                detail=(
                    f"LLM revised strategy: intent='{revised_intent}', "
                    f"query='{_truncate_text(revised_query, limit=80)}'. "
                    f"Reason: {_truncate_text(revision_rationale, limit=120)}"
                ),
            )
            recovered = await runtime.search_papers_smart(
                query=revised_query,
                limit=limit,
                search_session_id=search_session_id,
                mode=(
                    revised_intent
                    if revised_intent
                    in {
                        "auto",
                        "discovery",
                        "review",
                        "known_item",
                        "author",
                        "citation",
                        "regulatory",
                    }
                    else "auto"
                ),
                year=year,
                venue=venue,
                focus=focus,
                latency_profile=latency_profile,
                provider_budget=provider_budget,
                include_enrichment=include_enrichment,
                ctx=ctx,
            )
            strategy_metadata = recovered.get("strategyMetadata") if isinstance(recovered, dict) else None
            if isinstance(strategy_metadata, dict):
                warnings = list(strategy_metadata.get("driftWarnings") or [])
                warnings.append(
                    f"Initial {planner.intent} route returned few candidates. LLM revised strategy to "
                    f"intent='{revised_intent}': {_truncate_text(revision_rationale, limit=160)}"
                )
                strategy_metadata["driftWarnings"] = list(dict.fromkeys(warnings))
                strategy_metadata["intentSource"] = "fallback_recovery"
                strategy_metadata["intentConfidence"] = "medium"
                strategy_metadata["recoveryAttempted"] = True
                strategy_metadata["recoveryPath"] = [planner.intent, revised_intent]
                strategy_metadata["recoveryReason"] = revision_rationale or "Initial route returned too few results."
            return recovered

    providers_used = sorted(
        {provider for batch in [*initial_batches, *remaining_batches] for provider in batch.providers_used}
        | {candidate.provider for candidate in recommendation_candidates}
    )
    provider_selection = provider_bundle.selection_metadata()
    provider_fallback_warnings = _smart_provider_fallback_warnings(
        provider_selection=provider_selection,
        provider_outcomes=provider_outcomes,
    )
    ranking_diagnostics = summarize_ranking_diagnostics(reranked, top_n=10)
    strategy_metadata = SearchStrategyMetadata(
        intent=planner.intent,
        intentSource=planner.intent_source,
        intentConfidence=planner.intent_confidence,
        intentCandidates=planner.intent_candidates,
        secondaryIntents=planner.secondary_intents,
        routingConfidence=planner.routing_confidence,
        querySpecificity=planner.query_specificity,
        ambiguityLevel=planner.ambiguity_level,
        queryType=planner.query_type,
        regulatorySubintent=planner.regulatory_subintent,
        regulatoryIntent=planner.regulatory_intent,
        entityCard=planner.entity_card,
        subjectCard=planner.subject_card,
        subjectChainGaps=list(planner.subject_chain_gaps),
        intentFamily=planner.intent_family,
        breadthEstimate=planner.breadth_estimate,
        firstPassMode=planner.first_pass_mode,
        intentRationale=planner.intent_rationale,
        latencyProfile=latency_profile,
        normalizedQuery=normalized_query,
        queryVariantsTried=_dedupe_variants(
            [candidate.variant for candidate in initial_candidates + expansion_variants]
        ),
        retrievalHypotheses=_dedupe_variants(retrieval_hypotheses),
        searchAngles=_dedupe_variants(planner.search_angles),
        uncertaintyFlags=list(dict.fromkeys(planner.uncertainty_flags)),
        acceptedExpansions=_dedupe_variants(
            [candidate.variant for candidate in grounded if candidate.variant != normalized_query]
            + [variant for variant in accepted_speculative if variant != normalized_query]
        ),
        rejectedExpansions=rejected_speculative,
        speculativeExpansions=[candidate.variant for candidate in variants if candidate.source == "speculative"],
        providersUsed=providers_used,
        paidProvidersUsed=_paid_providers_used(providers_used),
        resultCoverage=_result_coverage_label(filtered_ranked),
        driftWarnings=[*drift_warnings, *provider_fallback_warnings],
        providerBudgetApplied=budget_state.to_dict() if budget_state else {},
        providerOutcomes=provider_outcomes,
        stageTimingsMs=stage_timings_ms,
        recoveryAttempted=False,
        recoveryPath=[planner.intent],
        anchorType="query_concepts",
        anchorStrength=planner.routing_confidence,
        anchoredSubject=(planner.candidate_concepts[0] if planner.candidate_concepts else normalized_query),
        normalizationWarnings=normalization_warnings,
        repairedInputs=repaired_inputs,
        bestNextInternalAction=("ask_result_set" if filtered_ranked else "search_papers_smart"),
        rankingDiagnostics=ranking_diagnostics,
        **provider_selection,
    )

    top_candidates = filtered_ranked[:limit]
    candidate_relevance_inputs: list[
        tuple[dict[str, Any], ScoreBreakdown, str, Literal["on_topic", "weak_match", "off_topic"]]
    ] = []
    borderline_relevance_papers: list[dict[str, Any]] = []
    borderline_relevance: dict[str, dict[str, Any]] = {}
    for candidate in top_candidates:
        score_breakdown = ScoreBreakdown.model_validate(candidate["scoreBreakdown"])
        deterministic_topical_relevance = _classify_topical_relevance_for_paper(
            query=normalized_query,
            paper=candidate["paper"],
            query_similarity=score_breakdown.query_similarity,
            score_breakdown=score_breakdown,
        )
        paper_id = str(
            candidate["paper"].get("paperId")
            or candidate["paper"].get("canonicalId")
            or candidate["paper"].get("sourceId")
            or ""
        ).strip()
        candidate_relevance_inputs.append((candidate, score_breakdown, paper_id, deterministic_topical_relevance))
        if deterministic_topical_relevance == "weak_match" and hasattr(provider_bundle, "aclassify_relevance_batch"):
            borderline_relevance_papers.append(candidate["paper"])
    if borderline_relevance_papers:
        try:
            borderline_relevance = await provider_bundle.aclassify_relevance_batch(
                query=normalized_query,
                papers=borderline_relevance_papers,
                request_id=request_id,
            )
        except Exception:
            borderline_relevance = {}
    smart_hits: list[SmartPaperHit] = []
    llm_classification_overrides = 0
    for index, (candidate, score_breakdown, paper_id, deterministic_topical_relevance) in enumerate(
        candidate_relevance_inputs,
        start=1,
    ):
        relevance_entry = borderline_relevance.get(paper_id) or {}
        llm_classification_raw = relevance_entry.get("classification")
        llm_classification = cast(Any, llm_classification_raw) if llm_classification_raw else None
        classification_result = _classify_topical_relevance_with_provenance(
            query=normalized_query,
            paper=candidate["paper"],
            query_similarity=score_breakdown.query_similarity,
            score_breakdown=score_breakdown,
            llm_classification=llm_classification,
        )
        topical_relevance = classification_result.effective
        if classification_result.llm_override_ignored:
            llm_classification_overrides += 1
            logger.info(
                "llm_classification_override_ignored",
                extra={
                    "query": normalized_query,
                    "paperId": paper_id,
                    "deterministic": classification_result.deterministic,
                    "llmClassification": classification_result.llm,
                },
            )
        smart_hits.append(
            SmartPaperHit(
                paper=Paper.model_validate(candidate["paper"]),
                rank=index,
                whyMatched=_why_matched(
                    query=normalized_query,
                    paper=candidate["paper"],
                    matched_concepts=candidate.get("matchedConcepts") or [],
                ),
                matchedConcepts=candidate.get("matchedConcepts") or [],
                retrievedBy=candidate["providers"],
                topicalRelevance=topical_relevance,
                llmClassification=classification_result.llm,
                classificationSource=classification_result.source,
                relevanceSource=cast(Any, relevance_entry.get("relevanceSource"))
                if relevance_entry.get("relevanceSource")
                else None,
                relevanceConfidence=relevance_entry.get("relevanceConfidence"),
                relevanceReason=relevance_entry.get("relevanceReason"),
                scoreBreakdown=score_breakdown,
            )
        )
    if llm_classification_overrides:
        strategy_metadata.llm_classification_overrides = llm_classification_overrides
    if include_enrichment and runtime._enrichment_service is not None and smart_hits:
        await runtime._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=85,
            message="Applying paper enrichment",
            detail=("Enriching the final smart-ranked hits with Crossref, Unpaywall, and OpenAlex metadata."),
        )
        enrichment_started = time.perf_counter()
        smart_hits = await runtime._enrich_smart_hits(
            smart_hits=smart_hits,
            query=normalized_query,
            request_id=request_id,
            provider_outcomes=provider_outcomes,
        )
        _finish_stage("enrichment", enrichment_started)
        strategy_metadata.provider_outcomes = provider_outcomes
        strategy_metadata.stage_timings_ms = stage_timings_ms

    source_records = [
        _source_record_from_paper(
            hit.paper,
            note=hit.why_matched,
            topical_relevance=hit.topical_relevance,
            llm_classification=hit.llm_classification,
            classification_source=hit.classification_source,
        )
        for hit in smart_hits
    ]
    evidence_records = _evidence_from_source_records(source_records)
    lead_records = _candidate_leads_from_source_records(source_records)
    coverage_summary = _smart_coverage_summary(
        providers_used=providers_used,
        provider_outcomes=provider_outcomes,
        search_mode="smart_literature_review",
        drift_warnings=[*drift_warnings, *provider_fallback_warnings],
    )
    result_status = "succeeded" if smart_hits else "partial"
    source_records = [
        _source_record_from_paper(
            hit.paper,
            note=hit.why_matched,
            topical_relevance=hit.topical_relevance,
            llm_classification=hit.llm_classification,
            classification_source=hit.classification_source,
        )
        for hit in smart_hits
    ]
    lead_records = _candidate_leads_from_source_records(source_records)
    result_status = "partial" if smart_hits else "abstained"
    response = SmartSearchResponse(
        results=smart_hits,
        searchSessionId=search_session_id or "pending",
        strategyMetadata=strategy_metadata,
        nextStepHint=(
            "Ask a grounded follow-up question with ask_result_set, "
            "cluster the area with map_research_landscape, or expand "
            "anchors with expand_research_graph."
        ),
        agentHints=build_agent_hints(
            "search_papers_smart",
            {"brokerMetadata": {"resultQuality": "strong"}},
        ),
        resourceUris=[],
        verifiedFindings=_verified_findings_from_source_records(source_records),
        likelyUnverified=_likely_unverified_from_source_records(source_records),
        answerability=_answerability_from_source_records(
            result_status=result_status,
            evidence=evidence_records,
            leads=lead_records,
            evidence_gaps=list(strategy_metadata.drift_warnings),
        ),
        routingSummary=_routing_summary_from_strategy(
            strategy_metadata=strategy_metadata,
            coverage_summary=coverage_summary,
            result_status=result_status,
            evidence_gaps=list(strategy_metadata.drift_warnings),
        ),
        evidence=evidence_records,
        leads=lead_records,
        candidateLeads=lead_records,
        evidenceGaps=list(strategy_metadata.drift_warnings),
        structuredSources=source_records,
        coverageSummary=coverage_summary,
        failureSummary=_smart_failure_summary(
            provider_outcomes=provider_outcomes,
            fallback_attempted=bool(provider_fallback_warnings or rejected_speculative),
        ),
        resultStatus=result_status,
        hasInspectableSources=_has_inspectable_sources(source_records),
        bestNextInternalAction=_best_next_internal_action(
            intent=planner.intent,
            has_sources=_has_on_topic_sources(source_records),
            result_status=result_status,
        ),
    )
    if provider_fallback_warnings:
        response.agent_hints.warnings.extend(provider_fallback_warnings)
    await runtime._emit_smart_search_status(
        ctx=ctx,
        request_id=request_id,
        progress=90,
        message="Saving reusable result set",
        detail=(f"Saving {len(smart_hits)} ranked result(s) to the workspace registry."),
    )
    response_dict = dump_jsonable(response)
    save_started = time.perf_counter()
    record = await runtime._workspace_registry.asave_result_set(
        source_tool="search_papers_smart",
        payload=response_dict,
        query=normalized_query,
        metadata={"strategyMetadata": response_dict["strategyMetadata"]},
        search_session_id=search_session_id,
    )
    _finish_stage("saveResultSet", save_started)
    response.search_session_id = record.search_session_id
    response.strategy_metadata.stage_timings_ms = stage_timings_ms
    response.resource_uris = build_resource_uris(
        "search_papers_smart",
        dump_jsonable(response),
        record.search_session_id,
    )
    final_response_dict = runtime._workspace_registry.attach_source_aliases(dump_jsonable(response))
    record.payload = final_response_dict
    record.metadata["strategyMetadata"] = final_response_dict["strategyMetadata"]
    if runtime._config.enable_trace_log:
        runtime._workspace_registry.record_trace(
            record.search_session_id,
            step="smart_search",
            payload={
                "normalizedQuery": normalized_query,
                "queryVariantsTried": strategy_metadata.query_variants_tried,
                "acceptedExpansions": strategy_metadata.accepted_expansions,
                "rejectedExpansions": strategy_metadata.rejected_expansions,
                "latencyProfile": latency_profile,
                "providerBudgetApplied": (budget_state.to_dict() if budget_state else {}),
            },
        )
    await runtime._emit_smart_search_status(
        ctx=ctx,
        request_id=request_id,
        progress=100,
        message="Smart search complete",
        detail=(
            f"Smart search complete with {len(smart_hits)} ranked result(s). "
            f"searchSessionId={record.search_session_id}."
        ),
    )
    return final_response_dict
