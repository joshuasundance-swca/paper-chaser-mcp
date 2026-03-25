"""Additive smart-workflow orchestration for Paper Chaser MCP."""

from __future__ import annotations

import asyncio
import logging
import re
import statistics
import time
from difflib import SequenceMatcher
from typing import Any
from uuid import uuid4

from fastmcp import Context

from ..citation_repair import resolve_citation
from ..compat import build_agent_hints, build_resource_uris
from ..enrichment import (
    PaperEnrichmentService,
    attach_enrichments_to_paper_payload,
    hydrate_paper_for_enrichment,
)
from ..models import Paper, dump_jsonable
from ..provider_runtime import (
    ProviderBudgetState,
    ProviderDiagnosticsRegistry,
    execute_provider_call,
)
from ..search import _enrich_ss_paper
from .config import AgenticConfig, LatencyProfile
from .models import (
    AgentHints,
    AskResultSetResponse,
    EvidenceItem,
    GraphEdge,
    GraphNode,
    LandscapeResponse,
    LandscapeTheme,
    ResearchGraphResponse,
    ScoreBreakdown,
    SearchStrategyMetadata,
    SmartPaperHit,
    SmartSearchResponse,
    StructuredToolError,
)
from .planner import (
    classify_query,
    combine_variants,
    dedupe_variants,
    grounded_expansion_candidates,
    query_facets,
    query_terms,
    speculative_expansion_candidates,
)
from .providers import (
    COMMON_QUERY_WORDS,
    DeterministicProviderBundle,
    ModelProviderBundle,
)
from .ranking import evaluate_speculative_variants, merge_candidates, rerank_candidates
from .retrieval import (
    SMART_RETRIEVAL_FIELDS,
    RetrievalBatch,
    RetrievedCandidate,
    retrieve_variant,
)
from .workspace import (
    ExpiredSearchSessionError,
    SearchSessionNotFoundError,
    WorkspaceRegistry,
)

logger = logging.getLogger("paper-chaser-mcp")
SMART_SEARCH_PROGRESS_TOTAL = 100.0
_GRAPH_GENERIC_TERMS = COMMON_QUERY_WORDS | {
    "effect",
    "effects",
    "environmental",
    "impact",
    "impacts",
    "response",
    "responses",
    "review",
    "wildlife",
}
_COMPARISON_MARKERS = {
    "compare",
    "compared",
    "comparing",
    "comparison",
    "differences",
    "different",
    "tradeoff",
    "tradeoffs",
    "versus",
    "vs",
}
_THEME_LABEL_STOPWORDS = _GRAPH_GENERIC_TERMS | {
    "about",
    "across",
    "among",
    "analysis",
    "and",
    "approach",
    "approaches",
    "based",
    "between",
    "cluster",
    "clusters",
    "for",
    "from",
    "into",
    "method",
    "methods",
    "model",
    "models",
    "or",
    "that",
    "the",
    "these",
    "theme",
    "themes",
    "theory",
    "those",
    "using",
    "with",
}
_COMPARISON_FOCUS_STOPWORDS = _THEME_LABEL_STOPWORDS | {
    "noise",
    "paper",
    "papers",
    "results",
    "study",
    "studies",
}

InMemorySaver: Any = None
StateGraph: Any = None
START: Any = "__start__"
END: Any = "__end__"

try:  # pragma: no cover - optional dependency
    from langgraph.checkpoint.memory import InMemorySaver as _InMemorySaver
    from langgraph.graph import END as _END
    from langgraph.graph import START as _START
    from langgraph.graph import StateGraph as _StateGraph

    InMemorySaver = _InMemorySaver
    StateGraph = _StateGraph
    START = _START
    END = _END
except ImportError:  # pragma: no cover - optional dependency
    pass


class AgenticRuntime:
    """Runtime entry point for the additive smart-tool surface."""

    def __init__(
        self,
        *,
        config: AgenticConfig,
        provider_bundle: ModelProviderBundle,
        workspace_registry: WorkspaceRegistry,
        client: Any,
        core_client: Any,
        openalex_client: Any,
        arxiv_client: Any,
        serpapi_client: Any,
        enable_core: bool,
        enable_semantic_scholar: bool,
        enable_openalex: bool,
        enable_arxiv: bool,
        enable_serpapi: bool,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
        enrichment_service: PaperEnrichmentService | None = None,
    ) -> None:
        self._config = config
        self._provider_bundle = provider_bundle
        self._deterministic_bundle = DeterministicProviderBundle(config)
        self._workspace_registry = workspace_registry
        self._client = client
        self._core_client = core_client
        self._openalex_client = openalex_client
        self._arxiv_client = arxiv_client
        self._serpapi_client = serpapi_client
        self._enable_core = enable_core
        self._enable_semantic_scholar = enable_semantic_scholar
        self._enable_openalex = enable_openalex
        self._enable_arxiv = enable_arxiv
        self._enable_serpapi = enable_serpapi
        self._provider_registry = provider_registry
        self._enrichment_service = enrichment_service
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._compiled_graphs = self._maybe_compile_graphs()

    def _provider_bundle_for_profile(
        self,
        latency_profile: LatencyProfile,
    ) -> ModelProviderBundle:
        settings = self._config.latency_profile_settings(latency_profile)
        if settings.use_deterministic_bundle:
            return self._deterministic_bundle
        return self._provider_bundle

    async def aclose(self) -> None:
        """Cancel best-effort background work and close owned async resources."""
        background_tasks = [task for task in self._background_tasks if task is not None and not task.done()]
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        await self._workspace_registry.aclose()
        await self._provider_bundle.aclose()
        await self._deterministic_bundle.aclose()

    async def _emit_smart_search_status(
        self,
        *,
        ctx: Context | None,
        request_id: str,
        progress: float | None = None,
        message: str | None = None,
        detail: str | None = None,
    ) -> None:
        log_message = detail or message
        if log_message:
            logger.info("smart-search[%s] %s", request_id, log_message)
        if ctx is None or self._skip_context_notifications(ctx):
            return
        if progress is not None and message is not None:
            self._schedule_context_call(
                ctx.report_progress(
                    progress=progress,
                    total=SMART_SEARCH_PROGRESS_TOTAL,
                    message=message,
                )
            )
        if detail:
            self._schedule_context_call(ctx.info(detail, logger_name="paper-chaser"))
        await asyncio.sleep(0)

    async def _emit_tool_progress(
        self,
        *,
        ctx: Context | None,
        progress: float,
        total: float,
        message: str,
    ) -> None:
        if ctx is None or self._skip_context_notifications(ctx):
            return
        self._schedule_context_call(
            ctx.report_progress(
                progress=progress,
                total=total,
                message=message,
            )
        )
        await asyncio.sleep(0)

    def _schedule_context_call(self, operation: Any) -> None:
        task = asyncio.create_task(operation)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(self._consume_background_task)

    @staticmethod
    def _skip_context_notifications(ctx: Context) -> bool:
        transport = getattr(ctx, "transport", None)
        if not isinstance(transport, str):
            return False
        return transport.lower() == "stdio"

    @staticmethod
    def _consume_background_task(task: asyncio.Task[Any]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.debug(
                "Best-effort context notification failed.",
                exc_info=True,
            )

    @staticmethod
    def _describe_retrieval_batch(batch: RetrievalBatch) -> str:
        providers_text = ", ".join(batch.providers_used) if batch.providers_used else "none"
        message = (
            f"Variant '{_truncate_text(batch.variant)}' finished with "
            f"{len(batch.candidates)} candidate(s) from {providers_text}."
        )
        if batch.provider_errors:
            errors_text = "; ".join(
                f"{provider}: {_truncate_text(error, limit=90)}"
                for provider, error in sorted(batch.provider_errors.items())
            )
            message = f"{message} Errors: {errors_text}."
        return message

    async def search_papers_smart(
        self,
        *,
        query: str,
        limit: int,
        search_session_id: str | None = None,
        mode: str = "auto",
        year: str | None = None,
        venue: str | None = None,
        focus: str | None = None,
        latency_profile: LatencyProfile = "balanced",
        provider_budget: dict[str, Any] | None = None,
        include_enrichment: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Smart concept-level discovery with grounded expansion and fusion."""
        if not self._config.enabled:
            return self._feature_not_configured(
                "Smart workflows are disabled. Set PAPER_CHASER_ENABLE_AGENTIC=true to use search_papers_smart.",
                fallback_tools=[
                    "search_papers",
                    "search_papers_bulk",
                    "search_papers_match",
                ],
            )

        profile_settings = self._config.latency_profile_settings(latency_profile)
        provider_bundle = self._provider_bundle_for_profile(latency_profile)
        budget_state = ProviderBudgetState.from_mapping(provider_budget)
        request_id = f"smart-{uuid4().hex[:10]}"
        provider_outcomes: list[dict[str, Any]] = []
        stage_timings_ms: dict[str, int] = {}

        def _finish_stage(stage_name: str, started_at: float) -> None:
            stage_timings_ms[stage_name] = int((time.perf_counter() - started_at) * 1000)

        planning_started = time.perf_counter()
        normalized_query, planner = await classify_query(
            query=query,
            mode=mode,
            year=year,
            venue=venue,
            focus=focus,
            provider_bundle=provider_bundle,
            request_outcomes=provider_outcomes,
            request_id=request_id,
        )
        _finish_stage("planning", planning_started)
        await self._emit_smart_search_status(
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
            return await self._search_known_item(
                query=normalized_query,
                limit=limit,
                planner_intent=planner.intent,
                search_session_id=search_session_id,
                latency_profile=latency_profile,
                request_id=request_id,
                include_enrichment=include_enrichment,
                provider_outcomes=provider_outcomes,
                stage_timings_ms=stage_timings_ms,
                ctx=ctx,
            )

        await self._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=15,
            message="Running initial retrieval",
            detail=(
                f"Searching the literal query across enabled providers for "
                f"'{_truncate_text(normalized_query, limit=96)}'."
            ),
        )
        first_retrieval_started = time.perf_counter()
        first_batch = await retrieve_variant(
            variant=normalized_query,
            variant_source="from_input",
            intent=planner.intent,
            year=year,
            venue=venue,
            enable_core=self._enable_core,
            enable_semantic_scholar=self._enable_semantic_scholar,
            enable_openalex=self._enable_openalex,
            enable_arxiv=self._enable_arxiv,
            enable_serpapi=self._enable_serpapi,
            core_client=self._core_client,
            semantic_client=self._client,
            openalex_client=self._openalex_client,
            arxiv_client=self._arxiv_client,
            serpapi_client=self._serpapi_client,
            widened=planner.intent == "review",
            is_expansion=False,
            allow_serpapi=(self._enable_serpapi and profile_settings.allow_serpapi_on_input),
            latency_profile=latency_profile,
            provider_registry=self._provider_registry,
            provider_budget=budget_state,
            request_outcomes=provider_outcomes,
            request_id=request_id,
        )
        _finish_stage("firstRetrieval", first_retrieval_started)
        await self._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=30,
            message="Initial retrieval complete",
            detail=self._describe_retrieval_batch(first_batch),
        )
        first_pass_papers = [candidate.paper for candidate in first_batch.candidates]
        if search_session_id:
            try:
                prior_record = self._workspace_registry.get(search_session_id)
                first_pass_papers = list(prior_record.papers) + first_pass_papers
            except (SearchSessionNotFoundError, ExpiredSearchSessionError):
                pass

        grounded = grounded_expansion_candidates(
            original_query=normalized_query,
            papers=first_pass_papers,
            config=profile_settings.search_config,
            focus=focus,
            venue=venue,
            year=year,
        )
        recommendation_task: asyncio.Task[list[RetrievedCandidate]] | None = None
        recommendation_started: float | None = None
        if profile_settings.enable_deep_recommendations:
            recommendation_started = time.perf_counter()
            recommendation_task = asyncio.create_task(
                self._semantic_recommendation_candidates(
                    seed_candidates=first_batch.candidates,
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

        expansion_variants = variants[1:]
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
            await self._emit_smart_search_status(
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
                            enable_core=self._enable_core,
                            enable_semantic_scholar=self._enable_semantic_scholar,
                            enable_openalex=self._enable_openalex,
                            enable_arxiv=self._enable_arxiv,
                            enable_serpapi=self._enable_serpapi,
                            core_client=self._core_client,
                            semantic_client=self._client,
                            openalex_client=self._openalex_client,
                            arxiv_client=self._arxiv_client,
                            serpapi_client=self._serpapi_client,
                            widened=planner.intent == "review",
                            is_expansion=True,
                            allow_serpapi=(self._enable_serpapi and profile_settings.allow_serpapi_on_expansions),
                            latency_profile=latency_profile,
                            provider_registry=self._provider_registry,
                            provider_budget=budget_state,
                            request_outcomes=provider_outcomes,
                            request_id=request_id,
                        )
                    )
                    for candidate in grounded_variants
                ]
                await self._emit_smart_search_status(
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
            await self._emit_smart_search_status(
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
                        enable_core=self._enable_core,
                        enable_semantic_scholar=self._enable_semantic_scholar,
                        enable_openalex=self._enable_openalex,
                        enable_arxiv=self._enable_arxiv,
                        enable_serpapi=self._enable_serpapi,
                        core_client=self._core_client,
                        semantic_client=self._client,
                        openalex_client=self._openalex_client,
                        arxiv_client=self._arxiv_client,
                        serpapi_client=self._serpapi_client,
                        widened=planner.intent == "review",
                        is_expansion=True,
                        allow_serpapi=(self._enable_serpapi and profile_settings.allow_serpapi_on_expansions),
                        latency_profile=latency_profile,
                        provider_registry=self._provider_registry,
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
                    await self._emit_smart_search_status(
                        ctx=ctx,
                        request_id=request_id,
                        progress=expansion_progress,
                        message=(f"Expansion {completed_index}/{len(expansion_tasks)} complete"),
                        detail=self._describe_retrieval_batch(batch),
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
            await self._emit_smart_search_status(
                ctx=ctx,
                request_id=request_id,
                progress=78,
                message="Recommendation fetch complete",
                detail=(
                    f"Semantic Scholar recommendation expansion returned {len(recommendation_candidates)} candidate(s)."
                ),
            )

        all_candidates = list(first_batch.candidates)
        for batch in remaining_batches:
            all_candidates.extend(batch.candidates)
        all_candidates.extend(recommendation_candidates)

        merged = merge_candidates(all_candidates)
        rerank_started = time.perf_counter()
        rerank_bundle = provider_bundle if profile_settings.use_embedding_rerank else self._deterministic_bundle
        reranked = await rerank_candidates(
            query=normalized_query,
            merged_candidates=merged,
            provider_bundle=rerank_bundle,
            candidate_concepts=planner.candidate_concepts,
            candidate_pool_size=profile_settings.search_config.candidate_pool_size,
            request_outcomes=provider_outcomes,
            request_id=request_id,
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

        await self._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=80,
            message="Reranking and deduplicating papers",
            detail=(f"Fusing {len(all_candidates)} candidate(s) into {len(filtered_ranked)} unique ranked paper(s)."),
        )

        strategy_metadata = SearchStrategyMetadata(
            intent=planner.intent,
            latencyProfile=latency_profile,
            normalizedQuery=normalized_query,
            queryVariantsTried=[candidate.variant for candidate in variants],
            acceptedExpansions=_dedupe_variants(
                [candidate.variant for candidate in grounded if candidate.variant != normalized_query]
                + [variant for variant in accepted_speculative if variant != normalized_query]
            ),
            rejectedExpansions=rejected_speculative,
            speculativeExpansions=[candidate.variant for candidate in variants if candidate.source == "speculative"],
            providersUsed=sorted(
                {provider for batch in [first_batch, *remaining_batches] for provider in batch.providers_used}
                | {candidate.provider for candidate in recommendation_candidates}
            ),
            resultCoverage=_result_coverage_label(filtered_ranked),
            driftWarnings=drift_warnings,
            providerBudgetApplied=budget_state.to_dict() if budget_state else {},
            providerOutcomes=provider_outcomes,
            stageTimingsMs=stage_timings_ms,
        )

        top_candidates = filtered_ranked[:limit]
        smart_hits = [
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
                scoreBreakdown=ScoreBreakdown.model_validate(candidate["scoreBreakdown"]),
            )
            for index, candidate in enumerate(top_candidates, start=1)
        ]
        if include_enrichment and self._enrichment_service is not None and smart_hits:
            await self._emit_smart_search_status(
                ctx=ctx,
                request_id=request_id,
                progress=85,
                message="Applying paper enrichment",
                detail=("Enriching the final smart-ranked hits with Crossref and Unpaywall metadata."),
            )
            enrichment_started = time.perf_counter()
            smart_hits = await self._enrich_smart_hits(
                smart_hits=smart_hits,
                query=normalized_query,
                request_id=request_id,
                provider_outcomes=provider_outcomes,
            )
            _finish_stage("enrichment", enrichment_started)
            strategy_metadata.provider_outcomes = provider_outcomes
            strategy_metadata.stage_timings_ms = stage_timings_ms

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
        )
        await self._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=90,
            message="Saving reusable result set",
            detail=(f"Saving {len(smart_hits)} ranked result(s) to the workspace registry."),
        )
        response_dict = dump_jsonable(response)
        save_started = time.perf_counter()
        record = await self._workspace_registry.asave_result_set(
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
        final_response_dict = dump_jsonable(response)
        record.payload = final_response_dict
        record.metadata["strategyMetadata"] = final_response_dict["strategyMetadata"]
        if self._config.enable_trace_log:
            self._workspace_registry.record_trace(
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
        await self._emit_smart_search_status(
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

    async def ask_result_set(
        self,
        *,
        search_session_id: str,
        question: str,
        top_k: int,
        answer_mode: str,
        latency_profile: LatencyProfile = "balanced",
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Answer grounded follow-up questions against a saved result set."""
        if not self._config.enabled:
            return self._feature_not_configured(
                "Smart workflows are disabled. Set PAPER_CHASER_ENABLE_AGENTIC=true to use ask_result_set.",
                fallback_tools=[
                    "search_papers",
                    "get_paper_details",
                    "get_paper_citations",
                ],
            )

        try:
            record = self._workspace_registry.get(search_session_id)
        except ExpiredSearchSessionError:
            return self._expired_result_set_error(search_session_id)
        except SearchSessionNotFoundError:
            return self._missing_result_set_error(search_session_id)

        profile_settings = self._config.latency_profile_settings(latency_profile)
        provider_bundle = self._provider_bundle_for_profile(latency_profile)
        similarity_bundle = provider_bundle if profile_settings.use_embedding_rerank else self._deterministic_bundle

        await self._emit_tool_progress(
            ctx=ctx,
            progress=1,
            total=3,
            message="Retrieving evidence from the saved result set",
        )
        evidence_papers = (
            await self._workspace_registry.asearch_papers(
                search_session_id,
                question,
                top_k=top_k,
            )
            or record.papers[:top_k]
        )
        evidence_texts = [_paper_text(paper) for paper in evidence_papers]
        synthesis, evidence_scores = await asyncio.gather(
            provider_bundle.aanswer_question(
                question=question,
                evidence_papers=evidence_papers,
                answer_mode=answer_mode,
            ),
            similarity_bundle.abatched_similarity(
                question,
                evidence_texts,
            ),
        )
        evidence = [
            EvidenceItem(
                paper=Paper.model_validate(paper),
                excerpt=str(paper.get("abstract") or paper.get("title") or "")[:240],
                whyRelevant=_why_matched(
                    query=question,
                    paper=paper,
                    matched_concepts=[],
                ),
                relevanceScore=round(score, 6),
            )
            for paper, score in zip(evidence_papers, evidence_scores, strict=False)
        ]
        answer_text = str(synthesis.get("answer") or "")
        if _should_use_structured_comparison_answer(
            question=question,
            answer_mode=answer_mode,
            answer_text=answer_text,
            evidence_papers=evidence_papers,
        ):
            answer_text = _build_grounded_comparison_answer(
                question=question,
                evidence_papers=evidence_papers,
            )
        await self._emit_tool_progress(
            ctx=ctx,
            progress=2,
            total=3,
            message="Drafting grounded answer",
        )
        response = AskResultSetResponse(
            answer=answer_text,
            evidence=evidence,
            unsupportedAsks=list(synthesis.get("unsupportedAsks") or []),
            followUpQuestions=list(synthesis.get("followUpQuestions") or []),
            confidence=provider_bundle.normalize_confidence(synthesis.get("confidence")),
            searchSessionId=search_session_id,
            agentHints=build_agent_hints("ask_result_set", {}),
            resourceUris=build_resource_uris(
                "ask_result_set",
                {"results": [{"paper": item.paper.model_dump(by_alias=True)} for item in evidence]},
                search_session_id,
            ),
        )
        if self._config.enable_trace_log:
            self._workspace_registry.record_trace(
                search_session_id,
                step="ask_result_set",
                payload={
                    "question": question,
                    "answerMode": answer_mode,
                    "evidenceCount": len(evidence),
                },
            )
        await self._emit_tool_progress(
            ctx=ctx,
            progress=3,
            total=3,
            message="Grounded answer complete",
        )
        return dump_jsonable(response)

    async def map_research_landscape(
        self,
        *,
        search_session_id: str,
        max_themes: int,
        latency_profile: LatencyProfile = "balanced",
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Cluster a saved result set into themes, gaps, and disagreements."""
        if not self._config.enabled:
            return self._feature_not_configured(
                "Smart workflows are disabled. Set PAPER_CHASER_ENABLE_AGENTIC=true to use map_research_landscape.",
                fallback_tools=[
                    "search_papers",
                    "search_papers_bulk",
                    "get_paper_citations",
                ],
            )
        try:
            record = self._workspace_registry.get(search_session_id)
        except ExpiredSearchSessionError:
            return self._expired_result_set_error(search_session_id)
        except SearchSessionNotFoundError:
            return self._missing_result_set_error(search_session_id)

        profile_settings = self._config.latency_profile_settings(latency_profile)
        provider_bundle = self._provider_bundle_for_profile(latency_profile)
        clustering_bundle = provider_bundle if profile_settings.use_embedding_rerank else self._deterministic_bundle

        await self._emit_tool_progress(
            ctx=ctx,
            progress=1,
            total=3,
            message="Clustering saved result set",
        )
        clusters = await _cluster_papers(
            papers=record.papers,
            provider_bundle=clustering_bundle,
            max_themes=max_themes,
        )
        themes: list[LandscapeTheme] = []
        representative_papers: list[Paper] = []
        theme_semaphore = asyncio.Semaphore(3)

        async def _build_theme(
            cluster: list[dict[str, Any]],
        ) -> tuple[LandscapeTheme, list[Paper]]:
            seed_terms = _top_terms_for_cluster(cluster)
            async with theme_semaphore:
                fallback = await provider_bundle.alabel_theme(
                    seed_terms=seed_terms,
                    papers=cluster,
                )
                sampled_title = await self._maybe_sample_theme_label(
                    seed_terms=seed_terms,
                    papers=cluster,
                    fallback=fallback,
                    ctx=ctx,
                )
                title = _finalize_theme_label(
                    raw_label=sampled_title,
                    seed_terms=seed_terms,
                    papers=cluster,
                )
                summary = await provider_bundle.asummarize_theme(
                    title=title,
                    papers=cluster,
                )
            reps = [Paper.model_validate(paper) for paper in cluster[:3]]
            theme = LandscapeTheme(
                title=title,
                summary=summary,
                representativePapers=reps,
                matchedConcepts=seed_terms,
            )
            return theme, reps[:1]

        built_themes = await asyncio.gather(*[_build_theme(cluster) for cluster in clusters])
        for theme, reps in built_themes:
            themes.append(theme)
            representative_papers.extend(reps)
        gaps = _compute_gaps(record.papers)
        disagreements = _compute_disagreements(record.papers)
        suggested_next_searches = _suggest_next_searches(record.papers, themes)
        await self._emit_tool_progress(
            ctx=ctx,
            progress=2,
            total=3,
            message="Labelling themes and summarizing gaps",
        )
        response = LandscapeResponse(
            themes=themes,
            representativePapers=representative_papers[:max_themes],
            gaps=gaps,
            disagreements=disagreements,
            suggestedNextSearches=suggested_next_searches,
            searchSessionId=search_session_id,
            agentHints=build_agent_hints("map_research_landscape", {}),
            resourceUris=build_resource_uris(
                "map_research_landscape",
                {
                    "representativePapers": [
                        paper.model_dump(by_alias=True) for paper in representative_papers[:max_themes]
                    ]
                },
                search_session_id,
            ),
        )
        if self._config.enable_trace_log:
            self._workspace_registry.record_trace(
                search_session_id,
                step="map_research_landscape",
                payload={
                    "themeCount": len(themes),
                    "gaps": gaps,
                    "disagreements": disagreements,
                },
            )
        await self._emit_tool_progress(
            ctx=ctx,
            progress=3,
            total=3,
            message="Landscape mapping complete",
        )
        return dump_jsonable(response)

    async def expand_research_graph(
        self,
        *,
        seed_paper_ids: list[str] | None,
        seed_search_session_id: str | None,
        direction: str,
        hops: int,
        per_seed_limit: int,
        latency_profile: LatencyProfile = "balanced",
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Expand citation/reference/author relationships into a compact graph."""
        if not self._config.enabled:
            return self._feature_not_configured(
                "Smart workflows are disabled. Set PAPER_CHASER_ENABLE_AGENTIC=true to use expand_research_graph.",
                fallback_tools=[
                    "get_paper_citations",
                    "get_paper_references",
                    "get_paper_authors",
                ],
            )
        resolved_seeds, seed_session_id = self._resolve_graph_seeds(
            seed_paper_ids=seed_paper_ids,
            seed_search_session_id=seed_search_session_id,
        )
        if not resolved_seeds:
            return self._feature_not_configured(
                "expand_research_graph needs either seedPaperIds or seedSearchSessionId.",
                fallback_tools=[
                    "get_paper_details",
                    "get_paper_citations",
                    "get_paper_references",
                ],
            )
        await self._emit_tool_progress(
            ctx=ctx,
            progress=1,
            total=3,
            message="Expanding research graph",
        )
        profile_settings = self._config.latency_profile_settings(latency_profile)
        frontier_scoring_bundle = (
            self._provider_bundle_for_profile(latency_profile)
            if profile_settings.use_embedding_rerank
            else self._deterministic_bundle
        )

        frontier_papers: list[dict[str, Any]] = []
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        graph_warnings: list[str] = []
        queue = list(resolved_seeds)
        expansion_semaphore = asyncio.Semaphore(4)
        seed_record = None
        if seed_session_id:
            try:
                seed_record = self._workspace_registry.get(seed_session_id)
            except (ExpiredSearchSessionError, SearchSessionNotFoundError):
                seed_record = None
        graph_intent_text = _graph_intent_text(seed_record, resolved_seeds)

        async def _expand_seed(
            *,
            seed: dict[str, Any],
            seed_id: str,
            label: str,
        ) -> dict[str, Any]:
            async with expansion_semaphore:
                if direction == "authors":
                    try:
                        payload = await self._client.get_paper_authors(
                            paper_id=seed_id,
                            limit=min(per_seed_limit, 25),
                            fields=None,
                            offset=None,
                        )
                    except Exception as error:
                        return {"error": f"Could not expand authors for {label!r}: {error}"}
                    return {
                        "seed_id": seed_id,
                        "label": label,
                        "authors": [author for author in payload.get("data") or [] if isinstance(author, dict)],
                    }

                try:
                    payload = await (
                        self._client.get_paper_citations(
                            paper_id=seed_id,
                            limit=per_seed_limit,
                            fields=None,
                            offset=None,
                        )
                        if direction == "citations"
                        else self._client.get_paper_references(
                            paper_id=seed_id,
                            limit=per_seed_limit,
                            fields=None,
                            offset=None,
                        )
                    )
                except Exception as error:
                    return {"error": f"Could not expand {direction} for {label!r}: {error}"}
                related_papers = [candidate for candidate in payload.get("data") or [] if isinstance(candidate, dict)]
                scores = await _graph_frontier_scores(
                    seed=seed,
                    related_papers=related_papers,
                    provider_bundle=frontier_scoring_bundle,
                    intent_text=graph_intent_text,
                )
                return {
                    "seed_id": seed_id,
                    "label": label,
                    "related_papers": related_papers,
                    "scores": scores,
                }

        for _ in range(max(hops, 1)):
            next_queue: list[dict[str, Any]] = []
            expansion_tasks: list[asyncio.Task[dict[str, Any]]] = []
            for seed in queue:
                try:
                    seed_id = self._portable_seed_id(seed)
                except ValueError as error:
                    graph_warnings.append(str(error))
                    continue
                label = str(seed.get("title") or seed_id)
                nodes.setdefault(
                    seed_id,
                    GraphNode(id=seed_id, kind="paper", label=label, score=0.0),
                )
                expansion_tasks.append(
                    asyncio.create_task(
                        _expand_seed(
                            seed=seed,
                            seed_id=seed_id,
                            label=label,
                        )
                    )
                )
            for result in await asyncio.gather(*expansion_tasks):
                if error_message := result.get("error"):
                    graph_warnings.append(str(error_message))
                    continue
                seed_id = str(result["seed_id"])
                if direction == "authors":
                    for author in result.get("authors") or []:
                        if not author.get("authorId"):
                            continue
                        author_id = str(author["authorId"])
                        nodes.setdefault(
                            author_id,
                            GraphNode(
                                id=author_id,
                                kind="author",
                                label=str(author.get("name") or author_id),
                                score=float(author.get("citationCount") or 0),
                                attributes={
                                    "paperCount": author.get("paperCount"),
                                    "citationCount": author.get("citationCount"),
                                },
                            ),
                        )
                        edges.append(
                            GraphEdge(
                                source=seed_id,
                                target=author_id,
                                relation="authored_by",
                            )
                        )
                    continue

                related_papers = list(result.get("related_papers") or [])
                scores = list(result.get("scores") or [])
                ranked_related = [(paper, score) for paper, score in zip(related_papers, scores, strict=False)]
                if len(ranked_related) < len(related_papers):
                    ranked_related.extend((paper, 0.0) for paper in related_papers[len(ranked_related) :])
                ranked_related.sort(key=lambda item: item[1], reverse=True)
                retained_related = _filter_graph_frontier(ranked_related)
                dropped_related = len(ranked_related) - len(retained_related)
                if dropped_related > 0:
                    graph_warnings.append(
                        "Filtered "
                        f"{dropped_related} off-topic {direction} candidate(s) "
                        "while preserving the strongest topical frontier."
                    )
                for paper, score in retained_related:
                    try:
                        related_id = self._portable_seed_id(paper)
                    except ValueError as error:
                        graph_warnings.append(str(error))
                        continue
                    nodes.setdefault(
                        related_id,
                        GraphNode(
                            id=related_id,
                            kind="paper",
                            label=str(paper.get("title") or related_id),
                            score=score,
                            attributes={
                                "year": paper.get("year"),
                                "citationCount": paper.get("citationCount"),
                                "source": paper.get("source"),
                            },
                        ),
                    )
                    edges.append(
                        GraphEdge(
                            source=seed_id if direction == "citations" else related_id,
                            target=related_id if direction == "citations" else seed_id,
                            relation=("cites" if direction == "citations" else "references"),
                        )
                    )
                frontier_papers.extend([paper for paper, _ in retained_related])
                next_queue.extend([paper for paper, _ in retained_related[: min(5, len(retained_related))]])
            queue = next_queue

        if not nodes:
            return self._non_portable_seed_error(graph_warnings)

        ranked_frontier = sorted(
            [node for node in nodes.values() if node.kind == ("author" if direction == "authors" else "paper")],
            key=lambda node: node.score,
            reverse=True,
        )
        graph_session_id: str | None = None
        if frontier_papers:
            graph_record = await self._workspace_registry.asave_result_set(
                source_tool="expand_research_graph",
                payload={"data": frontier_papers},
                query=graph_intent_text or resolved_seeds[0].get("title") or resolved_seeds[0].get("paperId"),
                metadata={
                    "trailParentPaperId": resolved_seeds[0].get("paperId"),
                    "trailDirection": direction,
                    "originalQuery": graph_intent_text,
                },
            )
            graph_session_id = graph_record.search_session_id
        await self._emit_tool_progress(
            ctx=ctx,
            progress=2,
            total=3,
            message="Ranking graph frontier",
        )
        agent_hints = build_agent_hints("expand_research_graph", {})
        if graph_warnings:
            agent_hints.warnings.extend(graph_warnings[:3])
        response = ResearchGraphResponse(
            nodes=list(nodes.values()),
            edges=edges,
            frontier=ranked_frontier[:per_seed_limit],
            nextStepHint=(
                "Use the frontier papers or authors as new anchors, or ask "
                "grounded questions against the saved searchSessionId if one "
                "was returned."
            ),
            searchSessionId=graph_session_id or seed_session_id,
            agentHints=agent_hints,
            resourceUris=build_resource_uris(
                "expand_research_graph",
                {"data": frontier_papers},
                graph_session_id or seed_session_id,
            ),
        )
        if self._config.enable_trace_log and response.search_session_id:
            self._workspace_registry.record_trace(
                response.search_session_id,
                step="expand_research_graph",
                payload={
                    "direction": direction,
                    "hops": hops,
                    "frontierCount": len(response.frontier),
                },
            )
        await self._emit_tool_progress(
            ctx=ctx,
            progress=3,
            total=3,
            message="Graph expansion complete",
        )
        return dump_jsonable(response)

    async def _semantic_recommendation_candidates(
        self,
        *,
        seed_candidates: list[RetrievedCandidate],
        normalized_query: str,
        enabled: bool,
        request_id: str,
        provider_outcomes: list[dict[str, Any]],
        provider_budget: ProviderBudgetState | None,
    ) -> list[RetrievedCandidate]:
        if not enabled or not self._enable_semantic_scholar:
            return []
        positive_paper_ids: list[str] = []
        for candidate in seed_candidates:
            if candidate.provider != "semantic_scholar":
                continue
            paper_id = candidate.paper.get("paperId")
            if not isinstance(paper_id, str) or not paper_id.strip():
                continue
            normalized_id = paper_id.strip()
            if normalized_id in positive_paper_ids:
                continue
            positive_paper_ids.append(normalized_id)
            if len(positive_paper_ids) >= 5:
                break
        if not positive_paper_ids:
            return []

        recommendation_call = await execute_provider_call(
            provider="semantic_scholar",
            endpoint="get_recommendations_post",
            operation=lambda: self._client.get_recommendations_post(
                positive_paper_ids=positive_paper_ids,
                limit=8,
                fields=SMART_RETRIEVAL_FIELDS,
            ),
            registry=self._provider_registry,
            budget=provider_budget,
            request_outcomes=provider_outcomes,
            request_id=request_id,
            is_empty=lambda payload: not (payload or {}).get("recommendedPapers"),
        )
        if recommendation_call.payload is None:
            return []

        candidates: list[RetrievedCandidate] = []
        for rank, paper in enumerate(
            recommendation_call.payload.get("recommendedPapers") or [],
            start=1,
        ):
            if not isinstance(paper, dict):
                continue
            enriched = _enrich_ss_paper(Paper.model_validate(paper)).model_dump(
                by_alias=True,
                exclude_none=True,
                exclude_defaults=True,
            )
            candidates.append(
                RetrievedCandidate(
                    paper=enriched,
                    provider="semantic_scholar",
                    variant=normalized_query,
                    variant_source="deep_recommendation",
                    provider_rank=rank,
                )
            )
        return candidates

    async def _enrich_smart_hits(
        self,
        *,
        smart_hits: list[SmartPaperHit],
        query: str,
        request_id: str,
        provider_outcomes: list[dict[str, Any]],
    ) -> list[SmartPaperHit]:
        enrichment_service = self._enrichment_service
        if enrichment_service is None:
            return smart_hits
        enrichment_semaphore = asyncio.Semaphore(4)

        async def _enrich_hit(hit: SmartPaperHit) -> SmartPaperHit:
            async with enrichment_semaphore:
                enrichment_source = await hydrate_paper_for_enrichment(
                    hit.paper,
                    detail_client=self._client,
                )
                enriched_payload = await enrichment_service.enrich_paper_payload(
                    enrichment_source,
                    query=hit.paper.title or query,
                    request_id=request_id,
                    request_outcomes=provider_outcomes,
                )
            enriched_paper = attach_enrichments_to_paper_payload(
                hit.paper,
                enriched_paper=enriched_payload,
            )
            return hit.model_copy(update={"paper": Paper.model_validate(enriched_paper)})

        return list(await asyncio.gather(*[_enrich_hit(hit) for hit in smart_hits]))

    async def _search_known_item(
        self,
        *,
        query: str,
        limit: int,
        planner_intent: str,
        search_session_id: str | None,
        latency_profile: LatencyProfile,
        request_id: str,
        include_enrichment: bool,
        provider_outcomes: list[dict[str, Any]],
        stage_timings_ms: dict[str, int],
        ctx: Context | None,
    ) -> dict[str, Any]:
        await self._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=20,
            message="Resolving known item",
            detail=(f"Attempting direct known-item resolution for '{_truncate_text(query, limit=96)}'."),
        )
        known_item_started = time.perf_counter()
        known_item, resolution_strategy = await self._resolve_known_item(query)
        stage_timings_ms["knownItemResolution"] = int((time.perf_counter() - known_item_started) * 1000)
        if known_item is None:
            await self._emit_smart_search_status(
                ctx=ctx,
                request_id=request_id,
                progress=45,
                message="Known-item resolution fell back to broader retrieval",
                detail=(
                    "No exact paper anchor was confirmed, so the smart workflow is returning a broader candidate "
                    "set instead of a dead-end error."
                ),
            )
            return await self._fallback_known_item_search(
                query=query,
                limit=limit,
                planner_intent=planner_intent,
                search_session_id=search_session_id,
                latency_profile=latency_profile,
                request_id=request_id,
                include_enrichment=include_enrichment,
                provider_outcomes=provider_outcomes,
                stage_timings_ms=stage_timings_ms,
                ctx=ctx,
            )
        if include_enrichment and self._enrichment_service is not None:
            await self._emit_smart_search_status(
                ctx=ctx,
                request_id=request_id,
                progress=70,
                message="Applying paper enrichment",
                detail=("Enriching the resolved known item with Crossref and Unpaywall metadata."),
            )
            enrichment_started = time.perf_counter()
            enrichment_source = await hydrate_paper_for_enrichment(
                known_item,
                detail_client=self._client,
            )
            enriched_payload = await self._enrichment_service.enrich_paper_payload(
                enrichment_source,
                query=known_item.get("title") or query,
                request_id=request_id,
                request_outcomes=provider_outcomes,
            )
            known_item = attach_enrichments_to_paper_payload(
                known_item,
                enriched_paper=enriched_payload,
            )
            stage_timings_ms["enrichment"] = int((time.perf_counter() - enrichment_started) * 1000)
        hit = SmartPaperHit(
            paper=Paper.model_validate(known_item),
            rank=1,
            whyMatched=(
                "Direct identifier, citation, or title resolution."
                if resolution_strategy == "citation_resolution"
                else (
                    "Known-item recovery resolved a likely paper anchor; verify year, venue, or title details "
                    "if needed."
                )
            ),
            matchedConcepts=[],
            retrievedBy=[resolution_strategy],
            scoreBreakdown=ScoreBreakdown(finalScore=1.0),
        )
        strategy_metadata = SearchStrategyMetadata(
            intent=planner_intent,
            latencyProfile=latency_profile,
            normalizedQuery=query,
            queryVariantsTried=[query],
            acceptedExpansions=[],
            rejectedExpansions=[],
            speculativeExpansions=[],
            providersUsed=[str(known_item.get("source") or "semantic_scholar")],
            resultCoverage="known_item",
            driftWarnings=(
                []
                if resolution_strategy == "citation_resolution"
                else [
                    "Known-item fallback used title-style recovery; verify the anchor before treating it as canonical."
                ]
            ),
            providerOutcomes=provider_outcomes,
            stageTimingsMs=stage_timings_ms,
        )
        response = SmartSearchResponse(
            results=[hit][:limit],
            searchSessionId=search_session_id or "pending",
            strategyMetadata=strategy_metadata,
            nextStepHint=(
                "Inspect the resolved paper, then expand citations, references, or authors from this anchor."
            ),
            agentHints=build_agent_hints(
                "get_paper_details",
                {"paperId": known_item.get("paperId")},
            ),
            resourceUris=[],
        )
        await self._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=90,
            message="Saving reusable result set",
            detail="Saving the resolved known item as a reusable search session.",
        )
        save_started = time.perf_counter()
        record = await self._workspace_registry.asave_result_set(
            source_tool="search_papers_smart",
            payload=dump_jsonable(response),
            query=query,
            metadata={"strategyMetadata": dump_jsonable(strategy_metadata)},
            search_session_id=search_session_id,
        )
        stage_timings_ms["saveResultSet"] = int((time.perf_counter() - save_started) * 1000)
        response.search_session_id = record.search_session_id
        response.strategy_metadata.stage_timings_ms = stage_timings_ms
        response.resource_uris = build_resource_uris(
            "search_papers_smart",
            {"results": [{"paper": known_item}]},
            record.search_session_id,
        )
        final_response_dict = dump_jsonable(response)
        record.payload = final_response_dict
        record.metadata["strategyMetadata"] = final_response_dict["strategyMetadata"]
        await self._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=100,
            message="Known-item resolution complete",
            detail=(f"Known-item resolution complete. searchSessionId={record.search_session_id}."),
        )
        return final_response_dict

    async def _resolve_known_item(self, query: str) -> tuple[dict[str, Any] | None, str]:
        result = await resolve_citation(
            citation=query,
            max_candidates=5,
            client=self._client,
            enable_core=self._enable_core,
            enable_semantic_scholar=self._enable_semantic_scholar,
            enable_openalex=self._enable_openalex,
            enable_arxiv=self._enable_arxiv,
            enable_serpapi=self._enable_serpapi,
            core_client=self._core_client,
            openalex_client=self._openalex_client,
            arxiv_client=self._arxiv_client,
            serpapi_client=self._serpapi_client,
        )
        best_match = result.get("bestMatch")
        if isinstance(best_match, dict) and isinstance(best_match.get("paper"), dict):
            return best_match["paper"], "citation_resolution"

        try:
            semantic_match = dump_jsonable(await self._client.search_papers_match(query=query, fields=None))
        except Exception:
            semantic_match = None
        if isinstance(semantic_match, dict) and semantic_match.get("paperId"):
            return semantic_match, str(semantic_match.get("matchStrategy") or "semantic_title_match")

        if self._enable_openalex and self._openalex_client is not None:
            try:
                autocomplete = await self._openalex_client.paper_autocomplete(query=query, limit=5)
            except Exception:
                autocomplete = None
            if isinstance(autocomplete, dict):
                for match in autocomplete.get("matches") or []:
                    if not isinstance(match, dict):
                        continue
                    match_id = str(match.get("id") or "").strip()
                    match_title = str(match.get("displayName") or "").strip()
                    if not match_id or _known_item_title_similarity(query, match_title) < 0.72:
                        continue
                    paper = None
                    try:
                        paper = await self._openalex_client.get_paper_details(paper_id=match_id)
                    except Exception as exc:
                        logger.debug("OpenAlex known-item detail lookup failed for %s: %s", match_id, exc)
                    if paper is None:
                        continue
                    return dump_jsonable(paper), "openalex_autocomplete"

            try:
                openalex_search = await self._openalex_client.search(query=query, limit=3)
            except Exception:
                openalex_search = None
            if isinstance(openalex_search, dict):
                for paper in openalex_search.get("data") or []:
                    if not isinstance(paper, dict) or not paper.get("paperId"):
                        continue
                    if _known_item_title_similarity(query, str(paper.get("title") or "")) < 0.58:
                        continue
                    return paper, "openalex_search"
        return None, "none"

    async def _fallback_known_item_search(
        self,
        *,
        query: str,
        limit: int,
        planner_intent: str,
        search_session_id: str | None,
        latency_profile: LatencyProfile,
        request_id: str,
        include_enrichment: bool,
        provider_outcomes: list[dict[str, Any]],
        stage_timings_ms: dict[str, int],
        ctx: Context | None,
    ) -> dict[str, Any]:
        profile_settings = self._config.latency_profile_settings(latency_profile)
        provider_bundle = self._provider_bundle_for_profile(latency_profile)
        retrieval_started = time.perf_counter()
        batch = await retrieve_variant(
            variant=query,
            variant_source="from_input",
            intent=planner_intent,
            year=None,
            venue=None,
            enable_core=self._enable_core,
            enable_semantic_scholar=self._enable_semantic_scholar,
            enable_openalex=self._enable_openalex,
            enable_arxiv=self._enable_arxiv,
            enable_serpapi=self._enable_serpapi,
            core_client=self._core_client,
            semantic_client=self._client,
            openalex_client=self._openalex_client,
            arxiv_client=self._arxiv_client,
            serpapi_client=self._serpapi_client,
            widened=True,
            is_expansion=False,
            allow_serpapi=(self._enable_serpapi and profile_settings.allow_serpapi_on_input),
            latency_profile=latency_profile,
            provider_registry=self._provider_registry,
            provider_budget=None,
            request_outcomes=provider_outcomes,
            request_id=request_id,
        )
        stage_timings_ms["knownItemFallbackRetrieval"] = int((time.perf_counter() - retrieval_started) * 1000)

        merged_candidates = merge_candidates(batch.candidates)
        ranked_candidates = await rerank_candidates(
            query=query,
            merged_candidates=merged_candidates,
            provider_bundle=provider_bundle,
            candidate_concepts=[],
            candidate_pool_size=max(limit * 4, 8),
            request_outcomes=provider_outcomes,
            request_id=request_id,
        )
        top_candidates = ranked_candidates[:limit]
        smart_hits = [
            SmartPaperHit(
                paper=Paper.model_validate(candidate["paper"]),
                rank=index,
                whyMatched=(
                    "Exact known-item resolution was not confident, so this broader candidate was returned for manual "
                    "verification."
                ),
                matchedConcepts=candidate.get("matchedConcepts") or [],
                retrievedBy=candidate["providers"],
                scoreBreakdown=ScoreBreakdown.model_validate(candidate["scoreBreakdown"]),
            )
            for index, candidate in enumerate(top_candidates, start=1)
        ]
        if include_enrichment and self._enrichment_service is not None and smart_hits:
            smart_hits = await self._enrich_smart_hits(
                smart_hits=smart_hits,
                query=query,
                request_id=request_id,
                provider_outcomes=provider_outcomes,
            )

        strategy_metadata = SearchStrategyMetadata(
            intent=planner_intent,
            latencyProfile=latency_profile,
            normalizedQuery=query,
            queryVariantsTried=[query],
            acceptedExpansions=[],
            rejectedExpansions=[],
            speculativeExpansions=[],
            providersUsed=sorted(batch.providers_used),
            resultCoverage=("narrow" if smart_hits else "none"),
            driftWarnings=[
                "Exact known-item resolution was not confident, so the smart workflow fell back to a broader candidate "
                "set. Verify title, year, and venue before treating a result as canonical."
            ],
            providerOutcomes=provider_outcomes,
            stageTimingsMs=stage_timings_ms,
        )
        response = SmartSearchResponse(
            results=smart_hits,
            searchSessionId=search_session_id or "pending",
            strategyMetadata=strategy_metadata,
            nextStepHint=(
                "Pick the closest candidate, then inspect details or use exact-title and identifier tools to "
                "confirm the "
                "anchor before expanding citations."
            ),
            agentHints=build_agent_hints(
                "search_papers_smart",
                {"brokerMetadata": {"resultQuality": "low_relevance" if not smart_hits else "unknown"}},
            ),
            resourceUris=[],
        )
        record = await self._workspace_registry.asave_result_set(
            source_tool="search_papers_smart",
            payload=dump_jsonable(response),
            query=query,
            metadata={"strategyMetadata": dump_jsonable(strategy_metadata)},
            search_session_id=search_session_id,
        )
        response.search_session_id = record.search_session_id
        response.resource_uris = build_resource_uris(
            "search_papers_smart",
            {"results": [{"paper": hit.paper.model_dump(by_alias=True)} for hit in smart_hits]},
            record.search_session_id,
        )
        final_response_dict = dump_jsonable(response)
        record.payload = final_response_dict
        record.metadata["strategyMetadata"] = final_response_dict["strategyMetadata"]
        await self._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=100,
            message="Known-item fallback complete",
            detail=(
                f"Known-item fallback complete with {len(smart_hits)} candidate(s). "
                f"searchSessionId={record.search_session_id}."
            ),
        )
        return final_response_dict

    def _resolve_graph_seeds(
        self,
        *,
        seed_paper_ids: list[str] | None,
        seed_search_session_id: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if seed_paper_ids:
            return [{"paperId": paper_id} for paper_id in seed_paper_ids], (seed_search_session_id)
        if not seed_search_session_id:
            return [], seed_search_session_id
        record = self._workspace_registry.get(seed_search_session_id)
        return record.papers[:5], seed_search_session_id

    def _portable_seed_id(self, paper: dict[str, Any]) -> str:
        status = str(paper.get("expansionIdStatus") or "").strip().lower()
        recommended = paper.get("recommendedExpansionId")
        if status == "portable" and isinstance(recommended, str) and recommended.strip():
            return recommended.strip()
        if status == "not_portable":
            raise ValueError(
                "This paper does not expose a portable expansion identifier. "
                "Resolve it through DOI or a Semantic Scholar-native lookup first."
            )
        for candidate in (
            recommended,
            paper.get("canonicalId"),
            paper.get("paperId"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        raise ValueError(
            "This paper does not expose a portable expansion identifier. "
            "Resolve it through DOI or a Semantic Scholar-native lookup first."
        )

    async def _maybe_sample_theme_label(
        self,
        *,
        seed_terms: list[str],
        papers: list[dict[str, Any]],
        fallback: str,
        ctx: Context | None,
    ) -> str:
        if ctx is None or not ctx.client_supports_extension("sampling"):
            return fallback
        try:
            sample = await ctx.sample(
                [
                    f"Seed terms: {', '.join(seed_terms)}",
                    "Titles:",
                    *[str(paper.get("title") or "") for paper in papers[:5]],
                ],
                system_prompt="Write one short literature-theme label.",
                max_tokens=20,
            )
            text = str(sample.text or "").strip()
            return text or fallback
        except Exception:
            return fallback

    def _feature_not_configured(
        self,
        message: str,
        *,
        fallback_tools: list[str],
    ) -> dict[str, Any]:
        return dump_jsonable(
            StructuredToolError(
                error="FEATURE_NOT_CONFIGURED",
                message=message,
                fallbackTools=fallback_tools,
                agentHints=AgentHints(
                    nextToolCandidates=fallback_tools,
                    whyThisNextStep=("Use the closest raw retrieval path until the smart layer is enabled."),
                    safeRetry=("Enable the agentic config and retry the same smart tool."),
                    warnings=[],
                ),
            )
        )

    def _missing_result_set_error(self, search_session_id: str) -> dict[str, Any]:
        return dump_jsonable(
            StructuredToolError(
                error="UNKNOWN_SEARCH_SESSION",
                message=(
                    f"searchSessionId {search_session_id!r} was not found. "
                    "Start from search_papers_smart or reuse a fresh discovery result."
                ),
                fallbackTools=[
                    "search_papers_smart",
                    "search_papers",
                    "search_papers_bulk",
                ],
                agentHints=AgentHints(
                    nextToolCandidates=["search_papers_smart", "search_papers"],
                    whyThisNextStep=("Create a fresh reusable result set, then retry the grounded follow-up."),
                    safeRetry=("Do not invent searchSessionId values; reuse the one returned by the tool."),
                    warnings=[],
                ),
            )
        )

    def _non_portable_seed_error(self, warnings: list[str]) -> dict[str, Any]:
        return dump_jsonable(
            StructuredToolError(
                error="NON_PORTABLE_SEED",
                message=(
                    "The supplied graph seed does not expose a portable "
                    "Semantic Scholar expansion identifier. Resolve the paper "
                    "through DOI, use paper.recommendedExpansionId when it is "
                    "present, or start from a Semantic Scholar-native lookup."
                ),
                fallbackTools=[
                    "search_papers_match",
                    "get_paper_details",
                    "get_paper_citations",
                    "get_paper_references",
                ],
                agentHints=AgentHints(
                    nextToolCandidates=[
                        "search_papers_match",
                        "get_paper_details",
                        "search_papers",
                    ],
                    whyThisNextStep=("Resolve the paper to a portable identifier first, then retry graph expansion."),
                    safeRetry=("Avoid brokered paperId/sourceId values when expansionIdStatus is not_portable."),
                    warnings=warnings[:3],
                ),
            )
        )

    def _expired_result_set_error(self, search_session_id: str) -> dict[str, Any]:
        return dump_jsonable(
            StructuredToolError(
                error="EXPIRED_SEARCH_SESSION",
                message=(
                    f"searchSessionId {search_session_id!r} has expired. "
                    "Re-run the discovery or expansion step to rebuild the workspace."
                ),
                fallbackTools=[
                    "search_papers_smart",
                    "search_papers",
                    "get_paper_citations",
                ],
                agentHints=AgentHints(
                    nextToolCandidates=["search_papers_smart", "search_papers"],
                    whyThisNextStep=("Refresh the result set first, then continue with grounded QA or clustering."),
                    safeRetry=("Session TTL is finite by design; rebuild from the latest search results."),
                    warnings=[],
                ),
            )
        )

    def _maybe_compile_graphs(self) -> dict[str, Any]:
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


def _why_matched(
    *,
    query: str,
    paper: dict[str, Any],
    matched_concepts: list[str],
) -> str:
    title = str(paper.get("title") or paper.get("paperId") or "paper")
    if matched_concepts:
        return f"{title} matched concepts {', '.join(matched_concepts[:3])}."
    if paper.get("venue"):
        return f"{title} matched the query and carries useful venue context from {paper['venue']}."
    return f"{title} was retained because it stayed close to the original query after fusion and deduplication."


def _paper_text(paper: dict[str, Any]) -> str:
    authors = ", ".join(author.get("name", "") for author in (paper.get("authors") or []) if isinstance(author, dict))
    return " ".join(
        part
        for part in [
            str(paper.get("title") or ""),
            str(paper.get("abstract") or ""),
            str(paper.get("venue") or ""),
            str(paper.get("year") or ""),
            authors,
        ]
        if part
    )


def _truncate_text(value: str, *, limit: int = 72) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(limit - 3, 1)].rstrip()}..."


def _comparison_requested(question: str, answer_mode: str) -> bool:
    if answer_mode == "comparison":
        return True
    question_tokens = set(re.findall(r"[a-z0-9]{2,}", question.lower()))
    return bool(question_tokens & _COMPARISON_MARKERS)


def _looks_like_title_venue_list(answer_text: str, evidence_papers: list[dict[str, Any]]) -> bool:
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    if not lines:
        return True
    bullet_lines = [line for line in lines if line.startswith("- ")]
    if len(bullet_lines) < min(2, len(evidence_papers)):
        return False
    matched_lines = 0
    normalized_titles = [str(paper.get("title") or paper.get("paperId") or "").strip() for paper in evidence_papers[:4]]
    for line in bullet_lines[:4]:
        lower_line = line.lower()
        has_title = any(title and title in line for title in normalized_titles)
        has_weak_metadata_pattern = any(marker in lower_line for marker in ("venue", "year", "unknown")) or bool(
            re.search(r":\s*[^\n,]+,\s*(19|20)\d{2}\b", line)
        )
        if has_title and has_weak_metadata_pattern:
            matched_lines += 1
    return matched_lines >= min(2, len(bullet_lines))


def _should_use_structured_comparison_answer(
    *,
    question: str,
    answer_mode: str,
    answer_text: str,
    evidence_papers: list[dict[str, Any]],
) -> bool:
    del answer_text, evidence_papers
    return _comparison_requested(question, answer_mode)


def _build_grounded_comparison_answer(
    *,
    question: str,
    evidence_papers: list[dict[str, Any]],
) -> str:
    papers = evidence_papers[: min(3, len(evidence_papers))]
    if not papers:
        return "The saved result set does not contain enough evidence to make a grounded comparison."
    shared_terms = _shared_focus_terms(papers, question=question)
    shared_ground = (
        ", ".join(term.title() for term in shared_terms[:3])
        if shared_terms
        else "closely related problem settings from the saved result set"
    )
    detail_lines = []
    for paper in papers:
        title = str(paper.get("title") or paper.get("paperId") or "Untitled")
        year = paper.get("year")
        venue = str(paper.get("venue") or "venue not stated")
        descriptor = _paper_focus_phrase(paper, question=question)
        timing = str(year) if isinstance(year, int) else "year unknown"
        detail_lines.append(f"- {title} ({timing}; {venue}) emphasizes {descriptor}.")
    takeaway = _comparison_takeaway(papers, shared_terms)
    return "\n".join(
        [
            "Grounded comparison from the saved result set.",
            f"Shared ground: these papers converge on {shared_ground}.",
            "Key differences:",
            *detail_lines,
            f"Takeaway: {takeaway}",
        ]
    )


def _shared_focus_terms(papers: list[dict[str, Any]], *, question: str) -> list[str]:
    counts: dict[str, int] = {}
    question_tokens = set(_graph_topic_tokens(question))
    for paper in papers:
        for token in _graph_topic_tokens(_paper_text(paper)):
            if token in _COMPARISON_FOCUS_STOPWORDS or token in question_tokens or token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 1
    minimum_count = 2 if len(papers) >= 2 else 1
    return [
        token for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if count >= minimum_count
    ]


def _paper_focus_phrase(paper: dict[str, Any], *, question: str) -> str:
    question_tokens = set(_graph_topic_tokens(question))
    focus_tokens: list[str] = []
    for source_text in (str(paper.get("title") or ""), str(paper.get("abstract") or "")):
        for token in re.findall(r"[a-z0-9]{3,}", source_text.lower()):
            if token in _COMPARISON_FOCUS_STOPWORDS or token in question_tokens:
                continue
            if token in focus_tokens:
                continue
            focus_tokens.append(token)
            if len(focus_tokens) >= 3:
                break
        if len(focus_tokens) >= 3:
            break
    if focus_tokens:
        return ", ".join(focus_tokens)
    abstract = str(paper.get("abstract") or "").strip()
    if abstract:
        return _truncate_text(abstract.lower(), limit=96)
    return "the same core topic from a different angle"


def _comparison_takeaway(papers: list[dict[str, Any]], shared_terms: list[str]) -> str:
    years: list[int] = []
    for paper in papers:
        year = paper.get("year")
        if isinstance(year, int):
            years.append(year)
    venues = [str(paper.get("venue") or "").strip() for paper in papers if str(paper.get("venue") or "").strip()]
    if years and max(years) != min(years):
        return (
            f"the papers stay grounded in {', '.join(term.title() for term in shared_terms[:2]) or 'the same topic'}, "
            "but they span different publication periods, so they likely reflect different stages of the literature."
        )
    if len(set(venues)) > 1:
        venue_list = ", ".join(sorted(set(venues))[:2])
        return (
            "the main contrast is not the core topic but the research setting, "
            f"with evidence spread across {venue_list}."
        )
    return (
        "the papers are topically close, but they contribute different emphases, methods, or evaluation perspectives."
    )


def _label_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", text.lower())


def _theme_terms_from_papers(seed_terms: list[str], papers: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for paper in papers[:8]:
        for token in _label_tokens(str(paper.get("title") or "")):
            if token in _THEME_LABEL_STOPWORDS or token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 3
        for token in _label_tokens(str(paper.get("abstract") or "")):
            if token in _THEME_LABEL_STOPWORDS or token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 1
    for term in seed_terms:
        for token in _label_tokens(term):
            if token in _THEME_LABEL_STOPWORDS or token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 2
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _normalized_theme_label(raw_label: str) -> str:
    parts = [segment.strip() for segment in re.split(r"[/|,:;\-]+", raw_label) if segment.strip()]
    if not parts:
        return ""
    return " / ".join(part.title() for part in parts[:2])


def _finalize_theme_label(
    *,
    raw_label: str,
    seed_terms: list[str],
    papers: list[dict[str, Any]],
) -> str:
    normalized = " ".join(raw_label.split())
    parts = [segment.strip() for segment in re.split(r"[/|,:;\-]+", normalized) if segment.strip()]
    if normalized and parts:
        part_tokens = [_label_tokens(part) for part in parts]
        part_meaningful = [[token for token in tokens if token not in _THEME_LABEL_STOPWORDS] for tokens in part_tokens]
        if all(tokens for tokens in part_meaningful):
            return _normalized_theme_label(normalized)
    derived_terms = _theme_terms_from_papers(seed_terms, papers)
    if len(derived_terms) >= 2:
        return " / ".join(term.title() for term in derived_terms[:2])
    if derived_terms:
        return derived_terms[0].title()
    if normalized:
        tokens = [token for token in _label_tokens(normalized) if token not in _THEME_LABEL_STOPWORDS]
        if tokens:
            return " / ".join(token.title() for token in tokens[:2])
    return "General theme"


def _known_item_title_similarity(query: str, title: str) -> float:
    normalized_query = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
    normalized_title = " ".join(re.findall(r"[a-z0-9]+", title.lower()))
    if not normalized_query or not normalized_title:
        return 0.0
    query_tokens = {token for token in normalized_query.split() if len(token) >= 3 and not token.isdigit()}
    title_tokens = {token for token in normalized_title.split() if len(token) >= 3 and not token.isdigit()}
    overlap = len(query_tokens & title_tokens) / len(query_tokens) if query_tokens else 0.0
    return max(SequenceMatcher(None, normalized_query, normalized_title).ratio(), overlap)


def _top_terms_for_cluster(papers: list[dict[str, Any]]) -> list[str]:
    tokens: dict[str, int] = {}
    for paper in papers[:8]:
        for token in _paper_text(paper).lower().split():
            cleaned = "".join(character for character in token if character.isalnum())
            if len(cleaned) < 4:
                continue
            tokens[cleaned] = tokens.get(cleaned, 0) + 1
    return [
        term
        for term, _ in sorted(
            tokens.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:4]
    ]


async def _cluster_papers(
    *,
    papers: list[dict[str, Any]],
    provider_bundle: ModelProviderBundle,
    max_themes: int,
) -> list[list[dict[str, Any]]]:
    if not papers:
        return []
    remaining = list(papers)
    clusters: list[list[dict[str, Any]]] = []
    threshold = 0.22
    while remaining and len(clusters) < max(max_themes, 1):
        seed = remaining.pop(0)
        cluster = [seed]
        seed_text = _paper_text(seed)
        candidate_texts = [_paper_text(candidate) for candidate in remaining]
        similarities = await provider_bundle.abatched_similarity(
            seed_text,
            candidate_texts,
        )
        rest: list[dict[str, Any]] = []
        for candidate, similarity in zip(remaining, similarities, strict=False):
            if similarity >= threshold:
                cluster.append(candidate)
            else:
                rest.append(candidate)
        clusters.append(cluster)
        remaining = rest
    if remaining:
        clusters[-1].extend(remaining)
    return clusters[:max_themes]


def _compute_gaps(papers: list[dict[str, Any]]) -> list[str]:
    if not papers:
        return ["No papers were available to analyze for gaps."]
    years: list[int] = [paper["year"] for paper in papers if isinstance(paper.get("year"), int)]
    venues: set[str] = {
        str(paper["venue"]) for paper in papers if isinstance(paper.get("venue"), str) and paper.get("venue")
    }
    gaps: list[str] = []
    if years and max(years) - min(years) <= 1:
        gaps.append(
            "The current result set is concentrated in a narrow time window; earlier foundational work may be missing."
        )
    if len(venues) <= 1:
        gaps.append("Most papers cluster around one venue or source, so cross-community coverage may still be thin.")
    if not gaps:
        gaps.append(
            "Methodological diversity looks reasonable, but targeted "
            "negative-result or benchmark papers may still be underrepresented."
        )
    return gaps


def _compute_disagreements(papers: list[dict[str, Any]]) -> list[str]:
    if len(papers) < 3:
        return ["The result set is still small, so disagreements are not yet obvious."]
    years: list[int] = [paper["year"] for paper in papers if isinstance(paper.get("year"), int)]
    if years and statistics.pstdev(years) >= 2.5:
        return [
            "The papers span different periods, so assumptions and evaluation "
            "norms may disagree across older and newer work."
        ]
    return [
        "Evaluation setups and coverage differ across the returned papers, so "
        "direct comparisons should be made carefully."
    ]


def _suggest_next_searches(
    papers: list[dict[str, Any]],
    themes: list[LandscapeTheme],
) -> list[str]:
    suggestions: list[str] = []
    if themes:
        suggestions.append(f"{themes[0].title} benchmark papers")
        suggestions.append(f"{themes[0].title} survey")
    recent_years: list[int] = sorted(
        {paper["year"] for paper in papers if isinstance(paper.get("year"), int)},
        reverse=True,
    )
    if recent_years:
        suggestions.append(f"{recent_years[0]} follow-up work")
    deduped: list[str] = []
    seen: set[str] = set()
    for suggestion in suggestions:
        if suggestion not in seen:
            seen.add(suggestion)
            deduped.append(suggestion)
    return deduped[:3]


async def _graph_frontier_scores(
    *,
    seed: dict[str, Any],
    related_papers: list[dict[str, Any]],
    provider_bundle: ModelProviderBundle,
    intent_text: str | None = None,
) -> list[float]:
    if not related_papers:
        return []
    seed_title = str(seed.get("title") or "")
    seed_terms = [term for term in query_terms(seed_title or _paper_text(seed)) if term not in _GRAPH_GENERIC_TERMS]
    seed_facets = query_facets(seed_title or _paper_text(seed))
    seed_term_set = _graph_topic_tokens(seed_title or _paper_text(seed))
    normalized_intent_text = (intent_text or "").strip()
    intent_terms = [term for term in query_terms(normalized_intent_text) if term not in _GRAPH_GENERIC_TERMS]
    intent_facets = query_facets(normalized_intent_text)
    intent_term_set = _graph_topic_tokens(normalized_intent_text)
    query_similarities = await provider_bundle.abatched_similarity(
        _paper_text(seed),
        [_paper_text(related) for related in related_papers],
    )
    if normalized_intent_text:
        intent_similarities = await provider_bundle.abatched_similarity(
            normalized_intent_text,
            [_paper_text(related) for related in related_papers],
        )
    else:
        intent_similarities = query_similarities
    scores: list[float] = []
    for related, query_similarity, intent_similarity in zip(
        related_papers,
        query_similarities,
        intent_similarities,
        strict=False,
    ):
        related_title = str(related.get("title") or "")
        related_text = _paper_text(related).lower()
        related_tokens = _graph_topic_tokens(related_text)
        related_title_tokens = _graph_topic_tokens(related_title.lower())
        anchor_overlap = sum(term in related_tokens for term in seed_terms) / len(seed_terms) if seed_terms else 0.0
        intent_anchor_overlap = (
            sum(term in related_tokens for term in intent_terms) / len(intent_terms) if intent_terms else 0.0
        )
        facet_overlap = 0.0
        if seed_facets:
            matched_facets = 0
            for facet in seed_facets:
                facet_tokens = re.findall(r"[a-z0-9]{3,}", facet.lower())
                if not facet_tokens:
                    continue
                required = len(facet_tokens) if len(facet_tokens) <= 2 else 2
                if sum(token in related_tokens for token in facet_tokens) >= required:
                    matched_facets += 1
            facet_overlap = matched_facets / len(seed_facets)
        intent_facet_overlap = 0.0
        if intent_facets:
            matched_intent_facets = 0
            for facet in intent_facets:
                facet_tokens = re.findall(r"[a-z0-9]{3,}", facet.lower())
                if not facet_tokens:
                    continue
                required = len(facet_tokens) if len(facet_tokens) <= 2 else 2
                if sum(token in related_tokens for token in facet_tokens) >= required:
                    matched_intent_facets += 1
            intent_facet_overlap = matched_intent_facets / len(intent_facets)
        title_overlap = 0.0
        if seed_term_set and related_title_tokens:
            title_overlap = len(seed_term_set & related_title_tokens) / len(seed_term_set)
        intent_title_overlap = 0.0
        if intent_term_set and related_title_tokens:
            intent_title_overlap = len(intent_term_set & related_title_tokens) / len(intent_term_set)

        citation_count = related.get("citationCount")
        citation_bonus = 0.0
        if isinstance(citation_count, int) and citation_count > 0:
            citation_bonus = min(citation_count / 5000.0, 0.08)
        year = related.get("year")
        recency_bonus = 0.0
        if isinstance(year, int):
            current_year = time.gmtime().tm_year
            recency_bonus = max(0.0, 0.03 - max(0, current_year - year) * 0.005)
        topic_penalty = 0.0
        if seed_terms and anchor_overlap == 0.0:
            topic_penalty += 0.26
        elif seed_terms and anchor_overlap < 0.25:
            topic_penalty += 0.1
        if intent_terms and intent_anchor_overlap == 0.0:
            topic_penalty += 0.24
        elif intent_terms and intent_anchor_overlap < 0.25:
            topic_penalty += 0.1
        if seed_facets and facet_overlap == 0.0:
            topic_penalty += 0.2
        elif seed_facets and facet_overlap < 0.5:
            topic_penalty += 0.08
        if intent_facets and intent_facet_overlap == 0.0:
            topic_penalty += 0.2
        elif intent_facets and intent_facet_overlap < 0.5:
            topic_penalty += 0.08
        if title_overlap == 0.0:
            topic_penalty += 0.12
        elif title_overlap < 0.2:
            topic_penalty += 0.05
        if intent_term_set and intent_title_overlap == 0.0:
            topic_penalty += 0.14
        elif intent_term_set and intent_title_overlap < 0.2:
            topic_penalty += 0.06
        if seed_terms and intent_terms and anchor_overlap == 0.0 and intent_anchor_overlap == 0.0:
            topic_penalty += 0.12
        score = (
            (query_similarity * 0.28)
            + (intent_similarity * 0.24)
            + (anchor_overlap * 0.12)
            + (intent_anchor_overlap * 0.16)
            + (facet_overlap * 0.08)
            + (intent_facet_overlap * 0.12)
            + (title_overlap * 0.05)
            + (intent_title_overlap * 0.09)
            + citation_bonus
            + recency_bonus
            - topic_penalty
        )
        scores.append(round(max(score, 0.0), 6))
    return scores


def _graph_intent_text(
    record: Any | None,
    resolved_seeds: list[dict[str, Any]],
) -> str:
    if record is not None:
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        strategy_metadata = metadata.get("strategyMetadata")
        if isinstance(strategy_metadata, dict):
            normalized_query = strategy_metadata.get("normalizedQuery")
            if isinstance(normalized_query, str) and normalized_query.strip():
                return normalized_query.strip()
        original_query = metadata.get("originalQuery")
        if isinstance(original_query, str) and original_query.strip():
            return original_query.strip()
        if isinstance(record.query, str) and record.query.strip():
            return record.query.strip()
    for seed in resolved_seeds:
        for candidate in (seed.get("title"), seed.get("paperId")):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _filter_graph_frontier(
    ranked_related: list[tuple[dict[str, Any], float]],
) -> list[tuple[dict[str, Any], float]]:
    if not ranked_related:
        return []
    best_score = max(score for _, score in ranked_related)
    threshold = max(0.18, best_score * 0.45)
    retained = [(paper, score) for paper, score in ranked_related if score >= threshold]
    if retained:
        return retained
    return ranked_related[: min(3, len(ranked_related))]


def _graph_topic_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", text.lower()) if token not in _GRAPH_GENERIC_TERMS}


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
