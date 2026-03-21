"""Additive smart-workflow orchestration for Scholar Search MCP."""

from __future__ import annotations

import asyncio
import statistics
from typing import Any
from uuid import uuid4

from fastmcp import Context

from ..citation_repair import resolve_citation
from ..compat import build_agent_hints, build_resource_uris
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
    grounded_expansion_candidates,
    speculative_expansion_candidates,
)
from .providers import DeterministicProviderBundle, ModelProviderBundle
from .ranking import evaluate_speculative_variants, merge_candidates, rerank_candidates
from .retrieval import SMART_RETRIEVAL_FIELDS, RetrievedCandidate, retrieve_variant
from .workspace import (
    ExpiredSearchSessionError,
    SearchSessionNotFoundError,
    WorkspaceRegistry,
)

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
        self._compiled_graphs = self._maybe_compile_graphs()

    def _provider_bundle_for_profile(
        self,
        latency_profile: LatencyProfile,
    ) -> ModelProviderBundle:
        settings = self._config.latency_profile_settings(latency_profile)
        if settings.use_deterministic_bundle:
            return self._deterministic_bundle
        return self._provider_bundle

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
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Smart concept-level discovery with grounded expansion and fusion."""
        if not self._config.enabled:
            return self._feature_not_configured(
                "Smart workflows are disabled. Set "
                "SCHOLAR_SEARCH_ENABLE_AGENTIC=true to use "
                "search_papers_smart.",
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

        normalized_query, planner = classify_query(
            query=query,
            mode=mode,
            year=year,
            venue=venue,
            focus=focus,
            provider_bundle=provider_bundle,
        )
        if ctx is not None:
            await ctx.report_progress(
                progress=1,
                total=5,
                message="Planning smart search",
            )

        if planner.intent == "known_item":
            return await self._search_known_item(
                query=normalized_query,
                limit=limit,
                planner_intent=planner.intent,
                search_session_id=search_session_id,
                latency_profile=latency_profile,
                ctx=ctx,
            )

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
            allow_serpapi=(
                self._enable_serpapi and profile_settings.allow_serpapi_on_input
            ),
            latency_profile=latency_profile,
            provider_registry=self._provider_registry,
            provider_budget=budget_state,
            request_outcomes=provider_outcomes,
            request_id=request_id,
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
        speculative = (
            speculative_expansion_candidates(
                original_query=normalized_query,
                papers=first_pass_papers,
                config=profile_settings.search_config,
                provider_bundle=provider_bundle,
            )
            if profile_settings.enable_speculative_expansions
            else []
        )
        variants = combine_variants(
            original_query=normalized_query,
            grounded=grounded,
            speculative=speculative,
            config=profile_settings.search_config,
        )

        if ctx is not None:
            await ctx.report_progress(
                progress=2,
                total=5,
                message="Running grounded expansions",
            )

        remaining_batches = (
            await asyncio.gather(
                *[
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
                        allow_serpapi=(
                            self._enable_serpapi
                            and profile_settings.allow_serpapi_on_expansions
                        ),
                        latency_profile=latency_profile,
                        provider_registry=self._provider_registry,
                        provider_budget=budget_state,
                        request_outcomes=provider_outcomes,
                        request_id=request_id,
                    )
                    for candidate in variants[1:]
                ]
            )
            if len(variants) > 1
            else []
        )

        recommendation_candidates = await self._semantic_recommendation_candidates(
            seed_candidates=first_batch.candidates,
            normalized_query=normalized_query,
            enabled=profile_settings.enable_deep_recommendations,
            request_id=request_id,
            provider_outcomes=provider_outcomes,
            provider_budget=budget_state,
        )

        all_candidates = list(first_batch.candidates)
        for batch in remaining_batches:
            all_candidates.extend(batch.candidates)
        all_candidates.extend(recommendation_candidates)

        merged = merge_candidates(all_candidates)
        reranked = rerank_candidates(
            query=normalized_query,
            merged_candidates=merged,
            provider_bundle=provider_bundle,
            candidate_concepts=planner.candidate_concepts,
        )
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

        if ctx is not None:
            await ctx.report_progress(
                progress=3,
                total=5,
                message="Reranking and deduplicating papers",
            )

        strategy_metadata = SearchStrategyMetadata(
            intent=planner.intent,
            latencyProfile=latency_profile,
            normalizedQuery=normalized_query,
            queryVariantsTried=[candidate.variant for candidate in variants],
            acceptedExpansions=_dedupe_variants(
                [
                    candidate.variant
                    for candidate in grounded
                    if candidate.variant != normalized_query
                ]
                + [
                    variant
                    for variant in accepted_speculative
                    if variant != normalized_query
                ]
            ),
            rejectedExpansions=rejected_speculative,
            speculativeExpansions=[
                candidate.variant
                for candidate in variants
                if candidate.source == "speculative"
            ],
            providersUsed=sorted(
                {
                    provider
                    for batch in [first_batch, *remaining_batches]
                    for provider in batch.providers_used
                }
                | {candidate.provider for candidate in recommendation_candidates}
            ),
            resultCoverage=_result_coverage_label(filtered_ranked),
            driftWarnings=drift_warnings,
            providerBudgetApplied=budget_state.to_dict() if budget_state else {},
            providerOutcomes=provider_outcomes,
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
                scoreBreakdown=ScoreBreakdown.model_validate(
                    candidate["scoreBreakdown"]
                ),
            )
            for index, candidate in enumerate(top_candidates, start=1)
        ]

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
        response_dict = dump_jsonable(response)
        record = self._workspace_registry.save_result_set(
            source_tool="search_papers_smart",
            payload=response_dict,
            query=normalized_query,
            metadata={"strategyMetadata": response_dict["strategyMetadata"]},
            search_session_id=search_session_id,
        )
        response.search_session_id = record.search_session_id
        response.resource_uris = build_resource_uris(
            "search_papers_smart",
            dump_jsonable(response),
            record.search_session_id,
        )
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
                    "providerBudgetApplied": (
                        budget_state.to_dict() if budget_state else {}
                    ),
                },
            )
        if ctx is not None:
            await ctx.report_progress(
                progress=5,
                total=5,
                message="Smart search complete",
            )
        return dump_jsonable(response)

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
                "Smart workflows are disabled. Set "
                "SCHOLAR_SEARCH_ENABLE_AGENTIC=true to use ask_result_set.",
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

        provider_bundle = self._provider_bundle_for_profile(latency_profile)

        if ctx is not None:
            await ctx.report_progress(
                progress=1,
                total=3,
                message="Retrieving evidence from the saved result set",
            )
        evidence_papers = (
            self._workspace_registry.search_papers(
                search_session_id,
                question,
                top_k=top_k,
            )
            or record.papers[:top_k]
        )
        synthesis = provider_bundle.answer_question(
            question=question,
            evidence_papers=evidence_papers,
            answer_mode=answer_mode,
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
                relevanceScore=round(
                    provider_bundle.similarity(question, _paper_text(paper)),
                    6,
                ),
            )
            for paper in evidence_papers
        ]
        if ctx is not None:
            await ctx.report_progress(
                progress=2,
                total=3,
                message="Drafting grounded answer",
            )
        response = AskResultSetResponse(
            answer=str(synthesis.get("answer") or ""),
            evidence=evidence,
            unsupportedAsks=list(synthesis.get("unsupportedAsks") or []),
            followUpQuestions=list(synthesis.get("followUpQuestions") or []),
            confidence=provider_bundle.normalize_confidence(synthesis.get("confidence")),
            searchSessionId=search_session_id,
            agentHints=build_agent_hints("ask_result_set", {}),
            resourceUris=build_resource_uris(
                "ask_result_set",
                {
                    "results": [
                        {"paper": item.paper.model_dump(by_alias=True)}
                        for item in evidence
                    ]
                },
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
        if ctx is not None:
            await ctx.report_progress(
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
                "Smart workflows are disabled. Set "
                "SCHOLAR_SEARCH_ENABLE_AGENTIC=true to use "
                "map_research_landscape.",
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

        provider_bundle = self._provider_bundle_for_profile(latency_profile)

        if ctx is not None:
            await ctx.report_progress(
                progress=1,
                total=3,
                message="Clustering saved result set",
            )
        clusters = _cluster_papers(
            papers=record.papers,
            provider_bundle=provider_bundle,
            max_themes=max_themes,
        )
        themes: list[LandscapeTheme] = []
        representative_papers: list[Paper] = []
        for cluster in clusters:
            seed_terms = _top_terms_for_cluster(cluster)
            title = await self._maybe_sample_theme_label(
                seed_terms=seed_terms,
                papers=cluster,
                fallback=provider_bundle.label_theme(
                    seed_terms=seed_terms,
                    papers=cluster,
                ),
                ctx=ctx,
            )
            summary = provider_bundle.summarize_theme(
                title=title,
                papers=cluster,
            )
            reps = [Paper.model_validate(paper) for paper in cluster[:3]]
            representative_papers.extend(reps[:1])
            themes.append(
                LandscapeTheme(
                    title=title,
                    summary=summary,
                    representativePapers=reps,
                    matchedConcepts=seed_terms,
                )
            )
        gaps = _compute_gaps(record.papers)
        disagreements = _compute_disagreements(record.papers)
        suggested_next_searches = _suggest_next_searches(record.papers, themes)
        if ctx is not None:
            await ctx.report_progress(
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
                        paper.model_dump(by_alias=True)
                        for paper in representative_papers[:max_themes]
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
        if ctx is not None:
            await ctx.report_progress(
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
                "Smart workflows are disabled. Set "
                "SCHOLAR_SEARCH_ENABLE_AGENTIC=true to use "
                "expand_research_graph.",
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
                "expand_research_graph needs either seedPaperIds or "
                "seedSearchSessionId.",
                fallback_tools=[
                    "get_paper_details",
                    "get_paper_citations",
                    "get_paper_references",
                ],
            )
        if ctx is not None:
            await ctx.report_progress(
                progress=1,
                total=3,
                message="Expanding research graph",
            )
        provider_bundle = self._provider_bundle_for_profile(latency_profile)

        frontier_papers: list[dict[str, Any]] = []
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        graph_warnings: list[str] = []
        queue = list(resolved_seeds)
        for _ in range(max(hops, 1)):
            next_queue: list[dict[str, Any]] = []
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
                if direction == "authors":
                    try:
                        payload = await self._client.get_paper_authors(
                            paper_id=seed_id,
                            limit=min(per_seed_limit, 25),
                            fields=None,
                            offset=None,
                        )
                    except Exception as error:
                        graph_warnings.append(
                            f"Could not expand authors for {label!r}: {error}"
                        )
                        continue
                    for author in payload.get("data") or []:
                        if not isinstance(author, dict) or not author.get("authorId"):
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
                    graph_warnings.append(
                        f"Could not expand {direction} for {label!r}: {error}"
                    )
                    continue
                related_papers = [
                    candidate
                    for candidate in payload.get("data") or []
                    if isinstance(candidate, dict)
                ]
                for paper in related_papers:
                    related_id = self._portable_seed_id(paper)
                    nodes.setdefault(
                        related_id,
                        GraphNode(
                            id=related_id,
                            kind="paper",
                            label=str(paper.get("title") or related_id),
                            score=_graph_frontier_score(
                                seed=seed,
                                related=paper,
                                provider_bundle=provider_bundle,
                            ),
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
                            relation=(
                                "cites" if direction == "citations" else "references"
                            ),
                        )
                    )
                frontier_papers.extend(related_papers)
                next_queue.extend(related_papers[: min(5, len(related_papers))])
            queue = next_queue

        if not nodes:
            return self._non_portable_seed_error(graph_warnings)

        ranked_frontier = sorted(
            [
                node
                for node in nodes.values()
                if node.kind == ("author" if direction == "authors" else "paper")
            ],
            key=lambda node: node.score,
            reverse=True,
        )
        graph_session_id: str | None = None
        if frontier_papers:
            graph_record = self._workspace_registry.save_result_set(
                source_tool="expand_research_graph",
                payload={"data": frontier_papers},
                query=resolved_seeds[0].get("title")
                or resolved_seeds[0].get("paperId"),
                metadata={
                    "trailParentPaperId": resolved_seeds[0].get("paperId"),
                    "trailDirection": direction,
                },
            )
            graph_session_id = graph_record.search_session_id
        if ctx is not None:
            await ctx.report_progress(
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
        if ctx is not None:
            await ctx.report_progress(
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

    async def _search_known_item(
        self,
        *,
        query: str,
        limit: int,
        planner_intent: str,
        search_session_id: str | None,
        latency_profile: LatencyProfile,
        ctx: Context | None,
    ) -> dict[str, Any]:
        if ctx is not None:
            await ctx.report_progress(
                progress=2,
                total=5,
                message="Resolving known item",
            )
        known_item = await self._resolve_known_item(query)
        if known_item is None:
            return self._feature_not_configured(
                "The known-item route could not resolve that identifier; try "
                "search_papers_match or search_papers.",
                fallback_tools=[
                    "search_papers_match",
                    "get_paper_details",
                    "search_papers",
                ],
            )
        hit = SmartPaperHit(
            paper=Paper.model_validate(known_item),
            rank=1,
            whyMatched="Direct identifier, citation, or title resolution.",
            matchedConcepts=[],
            retrievedBy=["known_item_resolution"],
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
            driftWarnings=[],
        )
        response = SmartSearchResponse(
            results=[hit][:limit],
            searchSessionId=search_session_id or "pending",
            strategyMetadata=strategy_metadata,
            nextStepHint=(
                "Inspect the resolved paper, then expand citations, "
                "references, or authors from this anchor."
            ),
            agentHints=build_agent_hints(
                "get_paper_details",
                {"paperId": known_item.get("paperId")},
            ),
            resourceUris=[],
        )
        record = self._workspace_registry.save_result_set(
            source_tool="search_papers_smart",
            payload=dump_jsonable(response),
            query=query,
            metadata={"strategyMetadata": dump_jsonable(strategy_metadata)},
            search_session_id=search_session_id,
        )
        response.search_session_id = record.search_session_id
        response.resource_uris = build_resource_uris(
            "search_papers_smart",
            {"results": [{"paper": known_item}]},
            record.search_session_id,
        )
        if ctx is not None:
            await ctx.report_progress(
                progress=5,
                total=5,
                message="Known-item resolution complete",
            )
        return dump_jsonable(response)

    async def _resolve_known_item(self, query: str) -> dict[str, Any] | None:
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
            return best_match["paper"]
        return None

    def _resolve_graph_seeds(
        self,
        *,
        seed_paper_ids: list[str] | None,
        seed_search_session_id: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if seed_paper_ids:
            return [{"paperId": paper_id} for paper_id in seed_paper_ids], (
                seed_search_session_id
            )
        if not seed_search_session_id:
            return [], seed_search_session_id
        record = self._workspace_registry.get(seed_search_session_id)
        return record.papers[:5], seed_search_session_id

    def _portable_seed_id(self, paper: dict[str, Any]) -> str:
        status = str(paper.get("expansionIdStatus") or "").strip().lower()
        recommended = paper.get("recommendedExpansionId")
        if (
            status == "portable"
            and isinstance(recommended, str)
            and recommended.strip()
        ):
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
                    whyThisNextStep=(
                        "Use the closest raw retrieval path until the smart "
                        "layer is enabled."
                    ),
                    safeRetry=(
                        "Enable the agentic config and retry the same smart tool."
                    ),
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
                    whyThisNextStep=(
                        "Create a fresh reusable result set, then retry the "
                        "grounded follow-up."
                    ),
                    safeRetry=(
                        "Do not invent searchSessionId values; reuse the one "
                        "returned by the tool."
                    ),
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
                    whyThisNextStep=(
                        "Resolve the paper to a portable identifier first, "
                        "then retry graph expansion."
                    ),
                    safeRetry=(
                        "Avoid brokered paperId/sourceId values when "
                        "expansionIdStatus is not_portable."
                    ),
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
                    whyThisNextStep=(
                        "Refresh the result set first, then continue with "
                        "grounded QA or clustering."
                    ),
                    safeRetry=(
                        "Session TTL is finite by design; rebuild from the "
                        "latest search results."
                    ),
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
        return (
            f"{title} matched the query and carries useful venue context from "
            f"{paper['venue']}."
        )
    return (
        f"{title} was retained because it stayed close to the original query "
        "after fusion and deduplication."
    )


def _paper_text(paper: dict[str, Any]) -> str:
    authors = ", ".join(
        author.get("name", "")
        for author in (paper.get("authors") or [])
        if isinstance(author, dict)
    )
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


def _cluster_papers(
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
        rest: list[dict[str, Any]] = []
        for candidate in remaining:
            if (
                provider_bundle.similarity(seed_text, _paper_text(candidate))
                >= threshold
            ):
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
    years: list[int] = [
        paper["year"] for paper in papers if isinstance(paper.get("year"), int)
    ]
    venues: set[str] = {
        str(paper["venue"])
        for paper in papers
        if isinstance(paper.get("venue"), str) and paper.get("venue")
    }
    gaps: list[str] = []
    if years and max(years) - min(years) <= 1:
        gaps.append(
            "The current result set is concentrated in a narrow time window; "
            "earlier foundational work may be missing."
        )
    if len(venues) <= 1:
        gaps.append(
            "Most papers cluster around one venue or source, so cross-community "
            "coverage may still be thin."
        )
    if not gaps:
        gaps.append(
            "Methodological diversity looks reasonable, but targeted "
            "negative-result or benchmark papers may still be underrepresented."
        )
    return gaps


def _compute_disagreements(papers: list[dict[str, Any]]) -> list[str]:
    if len(papers) < 3:
        return ["The result set is still small, so disagreements are not yet obvious."]
    years: list[int] = [
        paper["year"] for paper in papers if isinstance(paper.get("year"), int)
    ]
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


def _graph_frontier_score(
    *,
    seed: dict[str, Any],
    related: dict[str, Any],
    provider_bundle: ModelProviderBundle,
) -> float:
    query_similarity = provider_bundle.similarity(
        _paper_text(seed),
        _paper_text(related),
    )
    citation_count = related.get("citationCount")
    citation_bonus = 0.0
    if isinstance(citation_count, int) and citation_count > 0:
        citation_bonus = min(citation_count / 2000.0, 0.25)
    year = related.get("year")
    recency_bonus = 0.0
    if isinstance(year, int):
        recency_bonus = max(0.0, 0.08 - max(0, 2026 - year) * 0.01)
    return round(query_similarity + citation_bonus + recency_bonus, 6)


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
