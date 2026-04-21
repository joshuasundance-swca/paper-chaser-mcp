"""Additive smart-workflow orchestration for Paper Chaser MCP."""

from __future__ import annotations

import asyncio
import logging
import re
import statistics
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal, cast
from uuid import uuid4

from fastmcp import Context

from ...citation_repair import build_match_metadata, parse_citation, resolve_citation
from ...compat import build_agent_hints, build_resource_uris
from ...enrichment import (
    PaperEnrichmentService,
    attach_enrichments_to_paper_payload,
    hydrate_paper_for_enrichment,
)
from ...identifiers import resolve_doi_from_paper_payload
from ...models import (
    CitationRecord,
    CoverageSummary,
    FailureSummary,
    Paper,
    PrimaryDocumentCoverage,
    RegulatoryTimeline,
    RegulatoryTimelineEvent,
    VerificationStatus,
    dump_jsonable,
)
from ...provider_runtime import (
    ProviderBudgetState,
    ProviderDiagnosticsRegistry,
    execute_provider_call,
    provider_is_paywalled,
)
from ...search import _enrich_ss_paper
from ..answer_modes import (
    SYNTHESIS_MODES,
    aclassify_question_mode,
    build_evidence_use_plan,
    classify_question_mode,
    evidence_pool_is_weak,
)
from ..config import AgenticConfig, LatencyProfile
from ..models import (
    AgentHints,
    AskResultSetResponse,
    EvidenceItem,
    GraphEdge,
    GraphNode,
    IntentLabel,
    LandscapeResponse,
    LandscapeTheme,
    PlannerDecision,
    ResearchGraphResponse,
    ScoreBreakdown,
    SearchStrategyMetadata,
    SmartPaperHit,
    SmartSearchResponse,
    StructuredSourceRecord,
    StructuredToolError,
)
from ..planner import (
    classify_query,
    combine_variants,
    dedupe_variants,
    grounded_expansion_candidates,
    initial_retrieval_hypotheses,
    looks_like_exact_title,
    looks_like_near_known_item_query,
    normalize_query,
    query_facets,
    query_terms,
    speculative_expansion_candidates,
)
from ..providers import (
    COMMON_QUERY_WORDS,
    DeterministicProviderBundle,
    ModelProviderBundle,
)
from ..ranking import (
    evaluate_speculative_variants,
    merge_candidates,
    rerank_candidates,
    summarize_ranking_diagnostics,
)
from ..retrieval import (
    SMART_RETRIEVAL_FIELDS,
    RetrievalBatch,
    RetrievedCandidate,
    retrieve_variant,
)
from ..selection_scoring import (
    infer_comparative_axis,
    score_papers_for_comparative_axis,
)
from ..workspace import (
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


def _paid_providers_used(providers: list[str]) -> list[str]:
    return sorted({provider for provider in providers if provider_is_paywalled(provider)})


_COMPARISON_FOCUS_STOPWORDS = _THEME_LABEL_STOPWORDS | {
    "noise",
    "paper",
    "papers",
    "results",
    "study",
    "studies",
}
_REGULATORY_SUBJECT_STOPWORDS = {
    "act",
    "administration",
    "agency",
    "analysis",
    "code",
    "codified",
    "current",
    "decision",
    "critical",
    "document",
    "documents",
    "designation",
    "drug",
    "endangered",
    "fda",
    "federal",
    "final",
    "food",
    "functions",
    "guidance",
    "habitat",
    "history",
    "industry",
    "listed",
    "listing",
    "notice",
    "part",
    "plants",
    "profile",
    "profiles",
    "recovery",
    "register",
    "regulatory",
    "review",
    "rule",
    "section",
    "software",
    "species",
    "status",
    "text",
    "threatened",
    "title",
    "under",
    "wildlife",
}
_AGENCY_GUIDANCE_TERMS = {
    "guidance",
    "guideline",
    "policy",
    "staff",
}
_AGENCY_AUTHORITY_TERMS = {
    "agency",
    "cdc",
    "cms",
    "epa",
    "fda",
    "food and drug administration",
    "hhs",
    "nih",
    "usda",
}
_AGENCY_GUIDANCE_QUERY_NOISE_TERMS = {
    "actual",
    "agency",
    "document",
    "documents",
    "drug",
    "food",
    "guidance",
    "industry",
    "management",
    "most",
    "policy",
    "recent",
    "relevant",
    "staff",
    "what",
}
_AGENCY_GUIDANCE_DOCUMENT_TERMS = {
    "advisories",
    "advisory",
    "guidance",
    "guideline",
    "guidelines",
    "notice",
    "notices",
    "policies",
    "policy",
    "recommendation",
    "recommendations",
    "roadmap",
}
_AGENCY_GUIDANCE_DISCUSSION_TERMS = {
    "concept",
    "concepts",
    "discussion",
    "framework",
    "proposal",
    "proposals",
    "proposed",
}
_CULTURAL_RESOURCE_DOCUMENT_TERMS = {
    "106",
    "achp",
    "archaeological",
    "archaeology",
    "consultation",
    "cultural",
    "heritage",
    "historic",
    "nagpra",
    "nhpa",
    "preservation",
    "sacred",
    "shpo",
    "thpo",
    "tribal",
}
_REGULATORY_QUERY_NOISE_TERMS = {
    "address",
    "actions",
    "current",
    "federal",
    "recent",
    "states",
    "united",
    "what",
}
_SPECIES_QUERY_NOISE_TERMS = {
    "about",
    "critical",
    "current",
    "cfr",
    "dossier",
    "ecos",
    "endangered",
    "federal",
    "final",
    "history",
    "listing",
    "plants",
    "profile",
    "register",
    "regulatory",
    "rule",
    "say",
    "species",
    "status",
    "text",
    "threatened",
    "under",
    "what",
    "wildlife",
}
_CFR_DOC_TYPE_GENERIC = {
    "and",
    "cfr",
    "chapter",
    "part",
    "section",
    "subchapter",
    "title",
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
        scholarapi_client: Any = None,
        arxiv_client: Any,
        serpapi_client: Any,
        ecos_client: Any = None,
        federal_register_client: Any = None,
        govinfo_client: Any = None,
        enable_core: bool,
        enable_semantic_scholar: bool,
        enable_openalex: bool,
        enable_scholarapi: bool = False,
        enable_arxiv: bool,
        enable_serpapi: bool,
        enable_ecos: bool = True,
        enable_federal_register: bool = True,
        enable_govinfo_cfr: bool = True,
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
        self._scholarapi_client = scholarapi_client
        self._arxiv_client = arxiv_client
        self._serpapi_client = serpapi_client
        self._ecos_client = ecos_client
        self._federal_register_client = federal_register_client
        self._govinfo_client = govinfo_client
        self._enable_core = enable_core
        self._enable_semantic_scholar = enable_semantic_scholar
        self._enable_openalex = enable_openalex
        self._enable_scholarapi = enable_scholarapi
        self._enable_arxiv = enable_arxiv
        self._enable_serpapi = enable_serpapi
        self._enable_ecos = enable_ecos
        self._enable_federal_register = enable_federal_register
        self._enable_govinfo_cfr = enable_govinfo_cfr
        self._provider_registry = provider_registry
        self._enrichment_service = enrichment_service
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._compiled_graphs = self._maybe_compile_graphs()

    def smart_provider_diagnostics(self) -> tuple[dict[str, bool], list[str]]:
        smart_providers = [
            "openai",
            "azure-openai",
            "anthropic",
            "nvidia",
            "google",
            "mistral",
            "huggingface",
            "openrouter",
        ]
        configured_provider = self._provider_bundle.configured_provider_name()
        enabled = {provider: False for provider in smart_providers}
        if self._config.enabled and configured_provider in enabled:
            enabled[configured_provider] = self._provider_bundle.is_available()
        order = ([configured_provider] if configured_provider in enabled else []) + [
            provider for provider in smart_providers if provider != configured_provider
        ]
        return enabled, order

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
        latency_profile: LatencyProfile = "deep",
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
        normalization_warnings, repaired_inputs = _normalization_metadata(query, normalized_query)
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
            regulatory_result = await self._search_regulatory(
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
                tried_providers=sorted(
                    {str(o.get("provider") or "") for o in provider_outcomes if isinstance(o, dict)}
                ),
                result_summary=(
                    f"Regulatory routing returned no on-topic sources for "
                    f"'{_truncate_text(normalized_query, limit=96)}'."
                ),
                request_id=request_id,
            )
            revised_query = str(revision.get("revisedQuery") or normalized_query)
            revised_intent = str(revision.get("revisedIntent") or "review")
            revision_rationale = str(revision.get("rationale") or "")
            await self._emit_smart_search_status(
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
            recovered = await self.search_papers_smart(
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

        await self._emit_smart_search_status(
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
                    enable_core=self._enable_core,
                    enable_semantic_scholar=self._enable_semantic_scholar,
                    enable_openalex=self._enable_openalex,
                    enable_scholarapi=self._enable_scholarapi,
                    enable_arxiv=self._enable_arxiv,
                    enable_serpapi=self._enable_serpapi,
                    core_client=self._core_client,
                    semantic_client=self._client,
                    openalex_client=self._openalex_client,
                    scholarapi_client=self._scholarapi_client,
                    arxiv_client=self._arxiv_client,
                    serpapi_client=self._serpapi_client,
                    provider_plan=(candidate.provider_plan or planner.provider_plan or None),
                    widened=planner.intent == "review",
                    is_expansion=False,
                    allow_serpapi=(self._enable_serpapi and profile_settings.allow_serpapi_on_input),
                    latency_profile=latency_profile,
                    provider_registry=self._provider_registry,
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
        await self._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=30,
            message="Initial retrieval complete",
            detail=(
                f"Completed {len(initial_batches)} initial retrieval path(s). "
                f"Primary path: {self._describe_retrieval_batch(first_batch)}"
            ),
        )
        first_pass_papers = [candidate.paper for batch in initial_batches for candidate in batch.candidates]
        if search_session_id:
            try:
                prior_record = self._workspace_registry.get(search_session_id)
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
                self._semantic_recommendation_candidates(
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
                            enable_scholarapi=self._enable_scholarapi,
                            enable_arxiv=self._enable_arxiv,
                            enable_serpapi=self._enable_serpapi,
                            core_client=self._core_client,
                            semantic_client=self._client,
                            openalex_client=self._openalex_client,
                            scholarapi_client=self._scholarapi_client,
                            arxiv_client=self._arxiv_client,
                            serpapi_client=self._serpapi_client,
                            provider_plan=(candidate.provider_plan or planner.provider_plan or None),
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
                        enable_scholarapi=self._enable_scholarapi,
                        enable_arxiv=self._enable_arxiv,
                        enable_serpapi=self._enable_serpapi,
                        core_client=self._core_client,
                        semantic_client=self._client,
                        openalex_client=self._openalex_client,
                        scholarapi_client=self._scholarapi_client,
                        arxiv_client=self._arxiv_client,
                        serpapi_client=self._serpapi_client,
                        provider_plan=(candidate.provider_plan or planner.provider_plan or None),
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

        all_candidates = [candidate for batch in initial_batches for candidate in batch.candidates]
        for batch in remaining_batches:
            all_candidates.extend(batch.candidates)
        all_candidates.extend(recommendation_candidates)

        merged = merge_candidates(all_candidates)
        rerank_started = time.perf_counter()
        rerank_bundle = (
            provider_bundle
            if profile_settings.use_embedding_rerank and provider_bundle.supports_embeddings()
            else self._deterministic_bundle
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

        await self._emit_smart_search_status(
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
                await self._emit_smart_search_status(
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
                recovered = await self.search_papers_smart(
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
                    strategy_metadata["recoveryReason"] = (
                        revision_rationale or "Initial route returned too few results."
                    )
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
            if deterministic_topical_relevance == "weak_match" and hasattr(
                provider_bundle, "aclassify_relevance_batch"
            ):
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
        if include_enrichment and self._enrichment_service is not None and smart_hits:
            await self._emit_smart_search_status(
                ctx=ctx,
                request_id=request_id,
                progress=85,
                message="Applying paper enrichment",
                detail=("Enriching the final smart-ranked hits with Crossref, Unpaywall, and OpenAlex metadata."),
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
        final_response_dict = self._workspace_registry.attach_source_aliases(dump_jsonable(response))
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
        latency_profile: LatencyProfile = "deep",
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
        similarity_bundle = (
            provider_bundle
            if profile_settings.use_embedding_rerank and provider_bundle.supports_embeddings()
            else self._deterministic_bundle
        )
        session_followup_metadata: dict[str, Any] | None = None
        session_strategy_metadata = record.payload.get("strategyMetadata") if isinstance(record.payload, dict) else None
        if isinstance(session_strategy_metadata, dict):
            session_followup_metadata = {
                "followUpMode": session_strategy_metadata.get("followUpMode"),
            }
        initial_question_mode = classify_question_mode(question, session_followup_metadata)
        contextual_question = _contextualize_follow_up_question(
            question=question,
            record=record,
            question_mode=initial_question_mode,
        )

        await self._emit_tool_progress(
            ctx=ctx,
            progress=1,
            total=3,
            message="Retrieving evidence from the saved result set",
        )
        selected_papers = (
            await self._workspace_registry.asearch_papers(
                search_session_id,
                contextual_question,
                top_k=top_k,
                # Use semantic retrieval whenever a smart model-backed provider is active.
                allow_model_similarity=(self._provider_bundle.configured_provider_name() != "deterministic"),
            )
            or record.papers[:top_k]
        )
        evidence_sources = [
            source
            for key in ("evidence", "structuredSources", "sources")
            for source in (record.payload.get(key) or [])
            if isinstance(source, dict)
        ]
        source_by_id: dict[str, dict[str, Any]] = {}
        source_by_title: dict[str, dict[str, Any]] = {}
        for source in evidence_sources:
            source_id = str(source.get("sourceId") or source.get("evidenceId") or "").strip()
            if source_id and source_id not in source_by_id:
                source_by_id[source_id] = source
            title_key = str(source.get("title") or "").strip().lower()
            if title_key and title_key not in source_by_title:
                source_by_title[title_key] = source
        evidence_papers: list[dict[str, Any]] = []
        for paper in selected_papers:
            enriched_paper = dict(paper)
            source_record = (
                source_by_id.get(str(paper.get("paperId") or "").strip())
                or source_by_id.get(str(paper.get("sourceId") or "").strip())
                or source_by_id.get(str(paper.get("canonicalId") or "").strip())
                or source_by_title.get(str(paper.get("title") or "").strip().lower())
            )
            if source_record is not None:
                field_map = {
                    "sourceId": "sourceId",
                    "sourceAlias": "sourceAlias",
                    "provider": "source",
                    "sourceType": "sourceType",
                    "verificationStatus": "verificationStatus",
                    "accessStatus": "accessStatus",
                    "canonicalUrl": "canonicalUrl",
                    "retrievedUrl": "retrievedUrl",
                }
                for source_field, paper_field in field_map.items():
                    if source_record.get(source_field) and not enriched_paper.get(paper_field):
                        enriched_paper[paper_field] = source_record.get(source_field)
            evidence_papers.append(enriched_paper)
        evidence_texts = [_paper_text(paper) for paper in evidence_papers]
        if hasattr(provider_bundle, "_mark_provider_used") and hasattr(provider_bundle, "configured_provider_name"):
            provider_bundle._mark_provider_used(provider_bundle.configured_provider_name())
        synthesis, evidence_scores = await asyncio.gather(
            provider_bundle.aanswer_question(
                question=question,
                evidence_papers=evidence_papers,
                answer_mode=answer_mode,
            ),
            similarity_bundle.abatched_similarity(
                contextual_question,
                evidence_texts,
            ),
        )
        evidence = [
            EvidenceItem(
                evidenceId=str(paper.get("sourceId") or paper.get("paperId") or paper.get("canonicalId") or "").strip()
                or None,
                paper=Paper.model_validate(paper),
                excerpt=str(paper.get("abstract") or paper.get("title") or "")[:600],
                whyRelevant=_why_matched(
                    query=contextual_question,
                    paper=paper,
                    matched_concepts=[],
                ),
                relevanceScore=round(score, 6),
            )
            for paper, score in zip(evidence_papers, evidence_scores, strict=False)
        ]
        answer_text = str(synthesis.get("answer") or "")
        unsupported_asks = list(synthesis.get("unsupportedAsks") or [])
        follow_up_questions = list(synthesis.get("followUpQuestions") or [])
        normalized_confidence = provider_bundle.normalize_confidence(synthesis.get("confidence"))
        valid_evidence_ids = {
            str(item.evidence_id or item.paper.paper_id or item.paper.canonical_id or "").strip()
            for item in evidence
            if str(item.evidence_id or item.paper.paper_id or item.paper.canonical_id or "").strip()
        }
        selected_evidence_ids = [
            str(identifier).strip()
            for identifier in synthesis.get("selectedEvidenceIds") or []
            if str(identifier).strip() in valid_evidence_ids
        ]
        selected_lead_ids = [str(identifier).strip() for identifier in synthesis.get("selectedLeadIds") or []]
        provider_used = str(
            (
                provider_bundle.selection_metadata().get("activeSmartProvider")
                if hasattr(provider_bundle, "selection_metadata")
                else None
            )
            or provider_bundle.active_provider_name()
        )
        degradation_reason: str | None = None

        source_records: list[StructuredSourceRecord] = []
        on_topic_evidence = 0

        # LLM batch relevance pre-pass for papers in the middle-zone similarity range.
        relevance_cache = cast(dict[str, Any], record.metadata.setdefault("relevanceCache", {}))
        relevance_cache_key = " ".join(str(contextual_question or "").lower().split())
        question_relevance_cache = cast(dict[str, Any], relevance_cache.setdefault(relevance_cache_key, {}))
        llm_relevance: dict[str, dict[str, Any]] = {}
        if hasattr(provider_bundle, "aclassify_relevance_batch"):
            middle_zone_papers: list[dict[str, Any]] = []
            for item in evidence:
                sim = float(item.relevance_score)
                if 0.12 <= sim <= 0.50:
                    paper_id = str(item.paper.paper_id or item.paper.canonical_id or item.evidence_id or "").strip()
                    cached_entry = question_relevance_cache.get(paper_id) if paper_id else None
                    if isinstance(cached_entry, dict) and str(cached_entry.get("classification") or "").strip():
                        llm_relevance[paper_id] = {
                            "classification": str(cached_entry.get("classification") or "weak_match"),
                            "rationale": str(cached_entry.get("rationale") or ""),
                            "fallback": bool(cached_entry.get("fallback")),
                            "provenance": str(cached_entry.get("provenance") or "model").strip() or "model",
                        }
                        continue
                    paper_dict = (
                        item.paper.model_dump(by_alias=True) if isinstance(item.paper, Paper) else dict(item.paper)
                    )
                    middle_zone_papers.append(paper_dict)
                    if len(middle_zone_papers) >= 20:
                        break
            if middle_zone_papers:
                try:
                    batch_relevance = await provider_bundle.aclassify_relevance_batch(
                        query=contextual_question,
                        papers=middle_zone_papers,
                    )
                    for paper_id, entry in batch_relevance.items():
                        normalized_paper_id = str(paper_id or "").strip()
                        if not normalized_paper_id:
                            continue
                        normalized_entry = {
                            "classification": str(entry.get("classification") or "weak_match"),
                            "rationale": str(entry.get("rationale") or "").strip(),
                            "fallback": bool(entry.get("fallback")),
                            "provenance": str(entry.get("provenance") or "model").strip() or "model",
                        }
                        for key in (
                            "relevanceSource",
                            "relevanceConfidence",
                            "relevanceReason",
                            "classificationRationale",
                            "degradedTrigger",
                        ):
                            if entry.get(key) is not None:
                                normalized_entry[key] = entry.get(key)
                        question_relevance_cache[normalized_paper_id] = normalized_entry
                        llm_relevance[normalized_paper_id] = normalized_entry
                except Exception:
                    llm_relevance = {}

        from ..relevance_fallback import classification_provenance_counts

        classification_provenance = classification_provenance_counts(llm_relevance)
        degraded_classification = bool(classification_provenance.get("degradedClassification"))

        for item in evidence:
            paper_id = str(item.paper.paper_id or item.paper.canonical_id or item.evidence_id or "").strip()
            relevance_entry = llm_relevance.get(paper_id) or {}
            relevance_rationale = str(relevance_entry.get("rationale") or "").strip()
            relevance_provenance = str(relevance_entry.get("provenance") or "model").strip() or "model"
            classification_result = _classify_topical_relevance_with_provenance(
                query=contextual_question,
                paper=item.paper,
                query_similarity=float(item.relevance_score),
                llm_classification=cast(Any, relevance_entry.get("classification")),
            )
            topical_relevance = classification_result.effective
            if classification_result.llm_override_ignored:
                logger.info(
                    "llm_classification_override_ignored",
                    extra={
                        "query": question,
                        "paperId": paper_id,
                        "deterministic": classification_result.deterministic,
                        "llmClassification": classification_result.llm,
                    },
                )
            if topical_relevance == "on_topic":
                on_topic_evidence += 1
            source_records.append(
                _source_record_from_paper(
                    item.paper,
                    note=(
                        (
                            relevance_rationale
                            if relevance_provenance == "model"
                            else f"{relevance_rationale} [relevance:{relevance_provenance}]".strip()
                        )
                        if relevance_rationale
                        else (
                            item.why_relevant
                            if relevance_provenance == "model"
                            else f"{item.why_relevant} [relevance:{relevance_provenance}]"
                        )
                    ),
                    topical_relevance=topical_relevance,
                    llm_classification=classification_result.llm,
                    classification_source=classification_result.source,
                    relevance_source=(
                        str(relevance_entry.get("relevanceSource")) if relevance_entry.get("relevanceSource") else None
                    ),
                    relevance_confidence=(
                        float(cast(Any, relevance_entry.get("relevanceConfidence")))
                        if relevance_entry.get("relevanceConfidence") is not None
                        else None
                    ),
                    relevance_reason=(
                        str(relevance_entry.get("relevanceReason")) if relevance_entry.get("relevanceReason") else None
                    ),
                    classification_rationale=(
                        str(relevance_entry.get("classificationRationale"))
                        if relevance_entry.get("classificationRationale")
                        else None
                    ),
                )
            )

        max_relevance = max((item.relevance_score for item in evidence), default=0.0)
        insufficient_evidence = (not evidence) or (max_relevance < 0.12 and on_topic_evidence == 0)
        # Build the set of "strong" evidence identifiers: on-topic sources
        # whose verification status is primary-source or metadata-verified.
        # This set feeds the strict ``grounded`` answerability gate and the
        # selected_evidence_ids filter further below.
        _strong_verification_states = {"verified_primary_source", "verified_metadata"}
        strong_evidence_ids: set[str] = set()
        # P0-2: track which on-topic strong sources actually have QA-readable
        # body text. The ``grounded`` answerability gate (below) requires at
        # least one selected evidence id in this set; otherwise we downgrade
        # to ``limited`` with degradation_reason="no_qa_readable_body_text".
        # Legacy/backfilled records may not populate qa_readable_text; we
        # treat ``access_status in {body_text_embedded, full_text_verified,
        # full_text_retrieved}`` as implying QA-readable for back-compat.
        _qa_readable_access_states = {"body_text_embedded", "full_text_verified", "full_text_retrieved"}
        qa_readable_evidence_ids: set[str] = set()
        for _record in source_records:
            if _record.topical_relevance != "on_topic":
                continue
            if str(_record.verification_status or "") not in _strong_verification_states:
                continue
            _ident = str(_record.source_id or "").strip()
            if _ident:
                strong_evidence_ids.add(_ident)
                _qa_flag = getattr(_record, "qa_readable_text", None)
                if _qa_flag is True:
                    qa_readable_evidence_ids.add(_ident)
                elif _qa_flag is None and str(_record.access_status or "") in _qa_readable_access_states:
                    qa_readable_evidence_ids.add(_ident)
        if strong_evidence_ids:
            _filtered_selected = [
                identifier for identifier in selected_evidence_ids if identifier in strong_evidence_ids
            ]
            if _filtered_selected:
                selected_evidence_ids = _filtered_selected
        should_abstain = bool(unsupported_asks) and (
            normalized_confidence == "low" or max_relevance < 0.25 or on_topic_evidence == 0
        )
        if should_abstain:
            answer_status: Literal["answered", "abstained", "insufficient_evidence"] = "abstained"
            answer_payload: str | None = None
        elif insufficient_evidence:
            answer_status = "insufficient_evidence"
            answer_payload = None
        else:
            answer_status = "answered"
            answer_payload = answer_text
        if answer_status == "answered" and not selected_evidence_ids and evidence and answer_text:
            selected_evidence_ids = [
                identifier
                for identifier in (
                    str(item.evidence_id or item.paper.paper_id or item.paper.canonical_id or "").strip()
                    for item in evidence[: min(3, len(evidence))]
                )
                if identifier
            ]

        # LLM-first follow-up answer-mode classification. The deterministic
        # keyword classifier misses paraphrased synthesis asks such as
        # "what do these papers say?" (no SYNTHESIS_SHAPE_MARKERS hit, so it
        # reports ``unknown``). Consult the provider bundle's classifier
        # first; fall back to the keyword heuristic when no LLM is available.
        async def _llm_mode_classifier(q: str, modes: tuple[str, ...]) -> str | None:
            try:
                return await provider_bundle.aclassify_answer_mode(question=q, modes=modes)
            except Exception:  # pragma: no cover - defensive
                return None

        resolved_question_mode = await aclassify_question_mode(
            question,
            session_followup_metadata,
            llm_classifier=_llm_mode_classifier,
        )

        evidence_use_plan = build_evidence_use_plan(
            question=question,
            answer_mode=answer_mode,
            evidence=evidence,
            source_records=source_records,
            unsupported_asks=unsupported_asks,
            llm_relevance=llm_relevance,
            question_mode=resolved_question_mode,
        )
        question_mode = str(evidence_use_plan.get("answerMode") or "unknown")
        if (
            question_mode == "selection"
            and not bool(evidence_use_plan.get("sufficient"))
            and not unsupported_asks
            and answer_text
            and len(strong_evidence_ids) >= 2
        ):
            evidence_use_plan = {
                **evidence_use_plan,
                "sufficient": True,
                "retrievalSufficiency": "thin",
                "confidence": "medium",
                "rationale": (
                    "Selection follow-up can be answered from two strong on-topic sources "
                    "using deterministic ranking signals."
                ),
                "unsupportedComponents": [],
            }
        plan_sufficient = bool(evidence_use_plan.get("sufficient"))
        anchored_selection_source_ids = {
            str(identifier).strip()
            for identifier in (evidence_use_plan.get("anchoredSelectionSourceIds") or [])
            if str(identifier).strip()
        }
        answerability = str(synthesis.get("answerability") or "limited")
        if answerability not in {"grounded", "limited", "insufficient"}:
            answerability = "limited"
        if answer_status == "answered" and selected_evidence_ids:
            _normalized_confidence_value = str(normalized_confidence or "").lower()
            _has_qa_readable_selected = any(
                identifier in qa_readable_evidence_ids for identifier in selected_evidence_ids
            )
            if (
                provider_used != "deterministic"
                and strong_evidence_ids
                and _has_qa_readable_selected
                and _normalized_confidence_value in {"high", "medium"}
            ):
                answerability = "grounded"
            else:
                answerability = "limited"
                if degradation_reason is None:
                    if not strong_evidence_ids:
                        degradation_reason = "weak_evidence_pool"
                    elif not _has_qa_readable_selected:
                        # P0-2: we have strong (metadata/primary) on-topic
                        # sources, but none of the selected evidence has QA-
                        # readable body text. Cannot ground a synthesis
                        # answer on URL-only or abstract-only metadata.
                        degradation_reason = "no_qa_readable_body_text"
                        answer_status = "insufficient_evidence"
                        answer_payload = None
                    elif provider_used == "deterministic":
                        degradation_reason = "deterministic_synthesis_fallback"
                    else:
                        degradation_reason = "low_confidence_synthesis"
        elif answer_status == "insufficient_evidence":
            answerability = "insufficient"
        elif answer_status == "abstained" or not selected_evidence_ids:
            answerability = "limited" if (evidence or unsupported_asks or follow_up_questions) else "insufficient"
        # Evidence-pool quality gate: for synthesis-heavy modes, a pool
        # dominated by weak_match / off_topic classifications cannot support a
        # polished answer. But only fire this gate when the plan itself is
        # *not* already sufficient — otherwise a legitimate 2-paper synthesis
        # co-located with off-topic noise would be killed.
        weak_pool = evidence_pool_is_weak(source_records)
        if question_mode in SYNTHESIS_MODES and weak_pool and not plan_sufficient:
            answer_status = "insufficient_evidence"
            answer_payload = None
            answerability = "insufficient"
        # Strict synthesis gate: when the plan says the evidence is
        # insufficient for the requested synthesis mode, never emit a polished
        # answer. Prefer ``abstained`` when we have unsupported_asks to
        # surface (we have data but not for this ask), else
        # ``insufficient_evidence``.
        if question_mode in SYNTHESIS_MODES and not plan_sufficient:
            if unsupported_asks:
                answer_status = "abstained"
                answerability = "limited"
            else:
                answer_status = "insufficient_evidence"
                answerability = "insufficient"
            answer_payload = None
        elif evidence_use_plan.get("retrievalSufficiency") == "insufficient":
            answer_status = "insufficient_evidence"
            answer_payload = None
            answerability = "insufficient"
        elif (
            question_mode == "comparison"
            and evidence_use_plan.get("retrievalSufficiency") == "thin"
            and answer_status == "insufficient_evidence"
            and not unsupported_asks
            and len(evidence) >= 2
            and answer_text
        ):
            answer_status = "answered"
            answer_payload = answer_text
            answerability = "limited"
            if not selected_evidence_ids:
                selected_evidence_ids = [
                    identifier
                    for identifier in (
                        str(item.evidence_id or item.paper.paper_id or item.paper.canonical_id or "").strip()
                        for item in evidence[:2]
                    )
                    if identifier
                ]
        fallback_selected = [
            entry
            for entry in llm_relevance.values()
            if bool(entry.get("fallback")) and str(entry.get("classification") or "") == "on_topic"
        ]
        # Generalize the fallback-only gate to all synthesis modes: if the
        # only on-topic signal came from deterministic fallbacks (LLM call
        # failed), do not allow a synthesis-heavy answer to stand.
        if (
            answer_status == "answered"
            and question_mode in SYNTHESIS_MODES
            and len(fallback_selected) >= max(on_topic_evidence, 1)
        ):
            answer_status = "insufficient_evidence"
            answer_payload = None
            answerability = "insufficient"
        if unsupported_asks and answer_status == "insufficient_evidence":
            answer_status = "abstained"
            answerability = "limited"
        if (
            degraded_classification
            and answer_status == "answered"
            and provider_used != "deterministic"
            and classification_provenance.get("total", 0) >= 2
            and on_topic_evidence <= 1
            and max_relevance < 0.35
        ):
            answer_status = "insufficient_evidence"
            answer_payload = None
            answerability = "insufficient"
            degraded_gap = (
                "degraded_classification: relevance classifier retried/fell back; require more on-topic evidence."
            )
            if degraded_gap not in unsupported_asks:
                unsupported_asks = unsupported_asks + [degraded_gap]
        await self._emit_tool_progress(
            ctx=ctx,
            progress=2,
            total=3,
            message="Drafting grounded answer",
        )
        evidence_gaps = unsupported_asks
        if answer_status == "abstained":
            evidence_gaps = evidence_gaps + [
                "Grounded follow-up abstained because evidence was weak or unsupported for the requested claim."
            ]
        elif answer_status == "insufficient_evidence":
            evidence_gaps = evidence_gaps + [
                "Grounded follow-up could not find enough on-topic evidence to answer safely."
            ]
        if evidence_use_plan is not None:
            insufficiency_reason = "; ".join(evidence_use_plan.get("unsupportedComponents") or [])
            if insufficiency_reason:
                evidence_gaps = evidence_gaps + [f"Evidence use plan: {insufficiency_reason}"]
        agent_hints = build_agent_hints("ask_result_set", {})
        if provider_used == "deterministic" and answer_status == "answered":
            degradation_reason = "deterministic_synthesis_fallback"
            answerability = "limited"
            fallback_warning = (
                "deterministic_synthesis_fallback: follow-up synthesis used a deterministic fallback; "
                "treat the answer as a lightweight summary."
            )
            if fallback_warning not in evidence_gaps:
                evidence_gaps = evidence_gaps + [fallback_warning]
            if fallback_warning not in agent_hints.warnings:
                agent_hints.warnings.append(fallback_warning)
        top_recommendation = _compute_top_recommendation(
            question=question,
            resolved_question_mode=resolved_question_mode,
            answer_status=answer_status,
            evidence=evidence,
            source_records=source_records,
            strong_evidence_ids=strong_evidence_ids,
            anchored_selection_source_ids=anchored_selection_source_ids,
        )
        if (
            question_mode == "selection"
            and top_recommendation is not None
            and not unsupported_asks
            and answer_status != "abstained"
        ):
            recommended_source_id = str(top_recommendation.get("sourceId") or "").strip()
            if answer_status != "answered":
                answer_status = "answered"
                answerability = "limited"
                degradation_reason = degradation_reason or "selection_recommendation_only"
            if not str(answer_payload or "").strip():
                answer_payload = _selection_answer_from_recommendation(top_recommendation, source_records)
            if recommended_source_id and not selected_evidence_ids:
                selected_evidence_ids = [recommended_source_id]
            evidence_gaps = [
                gap
                for gap in evidence_gaps
                if gap != "Grounded follow-up could not find enough on-topic evidence to answer safely."
            ]
        visible_lead_ids = {
            str(record.source_id or record.source_alias or "").strip()
            for record in _candidate_leads_from_source_records(source_records)
            if str(record.source_id or record.source_alias or "").strip()
        }
        if visible_lead_ids:
            selected_lead_ids = [identifier for identifier in selected_lead_ids if identifier in visible_lead_ids]
        else:
            selected_lead_ids = []
        response = AskResultSetResponse(
            answer=answer_payload,
            answerStatus=answer_status,
            evidence=evidence,
            unsupportedAsks=unsupported_asks,
            followUpQuestions=follow_up_questions,
            answerability=cast(Any, answerability),
            selectedEvidenceIds=selected_evidence_ids,
            selectedLeadIds=selected_lead_ids,
            confidence=normalized_confidence,
            searchSessionId=search_session_id,
            agentHints=agent_hints,
            resourceUris=build_resource_uris(
                "ask_result_set",
                {"results": [{"paper": item.paper.model_dump(by_alias=True)} for item in evidence]},
                search_session_id,
            ),
            verifiedFindings=_verified_findings_from_source_records(source_records),
            likelyUnverified=_likely_unverified_from_source_records(source_records),
            candidateLeads=_candidate_leads_from_source_records(source_records),
            evidenceGaps=evidence_gaps,
            structuredSources=source_records,
            coverageSummary=_smart_coverage_summary(
                providers_used=sorted({str(item.paper.source or "unknown") for item in evidence}),
                provider_outcomes=[],
                search_mode="grounded_follow_up",
                drift_warnings=[],
            ),
            providerUsed=provider_used,
            degradationReason=degradation_reason,
            evidenceUsePlan=evidence_use_plan,
            classificationProvenance=classification_provenance if classification_provenance.get("total") else None,
            degradedClassification=degraded_classification if classification_provenance.get("total") else None,
            topRecommendation=top_recommendation,
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
        latency_profile: LatencyProfile = "deep",
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
        clustering_bundle = (
            provider_bundle
            if profile_settings.use_embedding_rerank and provider_bundle.supports_embeddings()
            else self._deterministic_bundle
        )

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
        source_records = [_source_record_from_paper(paper) for paper in representative_papers[:max_themes]]
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
            verifiedFindings=_verified_findings_from_source_records(source_records),
            likelyUnverified=_likely_unverified_from_source_records(source_records),
            candidateLeads=_candidate_leads_from_source_records(source_records),
            evidenceGaps=gaps,
            structuredSources=source_records,
            coverageSummary=_smart_coverage_summary(
                providers_used=sorted({str(paper.source or "unknown") for paper in representative_papers[:max_themes]}),
                provider_outcomes=[],
                search_mode="landscape_map",
                drift_warnings=gaps,
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
        latency_profile: LatencyProfile = "deep",
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
        scoring_provider_bundle = self._provider_bundle_for_profile(latency_profile)
        frontier_scoring_bundle = (
            scoring_provider_bundle
            if profile_settings.use_embedding_rerank and scoring_provider_bundle.supports_embeddings()
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

    async def _search_regulatory(
        self,
        *,
        query: str,
        limit: int,
        planner_intent: IntentLabel,
        planner: PlannerDecision,
        search_session_id: str | None,
        latency_profile: LatencyProfile,
        request_id: str,
        provider_outcomes: list[dict[str, Any]],
        stage_timings_ms: dict[str, int],
        normalization_warnings: list[str],
        repaired_inputs: dict[str, Any],
        ctx: Context | None,
    ) -> dict[str, Any]:
        attempted: list[str] = []
        succeeded: list[str] = []
        failed: list[str] = []
        zero_results: list[str] = []
        structured_sources: list[StructuredSourceRecord] = []
        evidence_gaps: list[str] = []
        timeline_events: list[RegulatoryTimelineEvent] = []
        candidate_leads: list[StructuredSourceRecord] = []
        subject: str | None = None
        current_text_requested, history_requested, agency_guidance_mode = _derive_regulatory_query_flags(
            query=query, planner=planner
        )
        govinfo_matched = False
        federal_register_matched = False

        await self._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=20,
            message="Running regulatory primary-source retrieval",
            detail=(f"Routing '{_truncate_text(query, limit=96)}' through ECOS, Federal Register, and CFR providers."),
        )

        cfr_request = _parse_cfr_request(query)
        cfr_citation = _format_cfr_citation(cfr_request)
        anchored_subject_terms: set[str] = (
            _agency_guidance_subject_terms(query) if agency_guidance_mode else _regulatory_query_subject_terms(query)
        )
        agency_priority_terms: set[str] = (
            _agency_guidance_priority_terms(query) if agency_guidance_mode else _regulatory_query_priority_terms(query)
        )
        agency_facet_terms: list[set[str]] = _agency_guidance_facet_terms(query) if agency_guidance_mode else []
        prefer_recent_guidance = agency_guidance_mode and _guidance_query_prefers_recency(query)
        cultural_resource_boost = planner.intent_family == "heritage_cultural_resources"
        # LLM-first subject-card grounding for document-family ranking and
        # species-dossier weak-match demotion. Prefer the subject_card already
        # resolved by classify_query when available.
        from ..subject_grounding import (
            compute_subject_chain_gaps,
            resolve_subject_card,
            species_mentioned,
        )

        _subject_card = planner.subject_card or resolve_subject_card(query=query, planner=planner)
        _requested_family: str | None = _subject_card.requested_document_family if _subject_card else None
        _species_dossier_mode = planner.regulatory_intent == "species_dossier"
        filtered_document_count = 0
        species_anchor_type: str | None = None

        if agency_guidance_mode and self._enable_govinfo_cfr and self._govinfo_client is not None and not cfr_request:
            attempted.append("govinfo")
            started = time.perf_counter()
            try:
                govinfo_search = await self._govinfo_client.search_federal_register_documents(
                    query=query,
                    limit=min(max(limit, 5), 10),
                )
                stage_timings_ms["regulatoryGovInfoSearch"] = int((time.perf_counter() - started) * 1000)
                documents = list(govinfo_search.get("data") or [])
                if documents:
                    succeeded.append("govinfo")
                    ranked_documents = _rank_regulatory_documents(
                        documents,
                        subject_terms=anchored_subject_terms,
                        priority_terms=agency_priority_terms,
                        facet_terms=agency_facet_terms,
                        prefer_guidance=True,
                        prefer_recent=prefer_recent_guidance,
                        cultural_resource_boost=cultural_resource_boost,
                        requested_document_family=_requested_family,
                    )
                    for document in ranked_documents[:limit]:
                        if not isinstance(document, dict):
                            continue
                        if not _regulatory_document_matches_subject(
                            document,
                            subject_terms=anchored_subject_terms,
                            priority_terms=agency_priority_terms,
                            cfr_citation=None,
                        ):
                            filtered_document_count += 1
                            off_topic_lead = _source_record_from_regulatory_document(
                                document,
                                provider="govinfo",
                                topical_relevance="off_topic",
                            )
                            reason_note = (
                                "Filtered from grounded agency-guidance evidence because it did not match the "
                                "requested agency-guidance subject."
                            )
                            existing_note = off_topic_lead.note
                            candidate_leads.append(
                                off_topic_lead.model_copy(
                                    update={
                                        "note": (
                                            f"{reason_note} {existing_note}".strip() if existing_note else reason_note
                                        )
                                    }
                                )
                            )
                            continue
                        structured_sources.append(
                            _source_record_from_regulatory_document(
                                document,
                                provider="govinfo",
                                topical_relevance="on_topic",
                            )
                        )
                        govinfo_matched = True
                        subject = str(subject or document.get("title") or query)
                        timeline_events.append(
                            RegulatoryTimelineEvent(
                                eventType="agency_guidance",
                                eventDate=document.get("publicationDate"),
                                title=str(document.get("title") or document.get("citation") or "GovInfo document"),
                                citation=str(document.get("citation") or document.get("documentNumber") or "") or None,
                                canonicalUrl=document.get("sourceUrl"),
                                provider="govinfo",
                                verificationStatus=cast(
                                    VerificationStatus,
                                    document.get("verificationStatus") or "verified_metadata",
                                ),
                                note=str(document.get("note") or "GovInfo guidance/title search hit."),
                            )
                        )
                else:
                    zero_results.append("govinfo")
            except Exception as exc:
                stage_timings_ms["regulatoryGovInfoSearch"] = int((time.perf_counter() - started) * 1000)
                failed.append("govinfo")
                provider_outcomes.append(
                    {
                        "provider": "govinfo",
                        "endpoint": "search_federal_register_documents",
                        "statusBucket": "provider_error",
                        "error": str(exc),
                    }
                )
        if cfr_request and self._enable_govinfo_cfr and self._govinfo_client is not None:
            attempted.append("govinfo")
            started = time.perf_counter()
            try:
                cfr_payload = await self._govinfo_client.get_cfr_text(**cfr_request)
                stage_timings_ms["regulatoryGovInfo"] = int((time.perf_counter() - started) * 1000)
                succeeded.append("govinfo")
                govinfo_matched = True
                subject = str(cfr_payload.get("citation") or query)
                structured_sources.append(
                    _source_record_from_regulatory_document(
                        cfr_payload,
                        provider="govinfo",
                        topical_relevance="on_topic",
                    )
                )
                timeline_events.append(
                    RegulatoryTimelineEvent(
                        eventType="cfr_text",
                        eventDate=cfr_payload.get("effectiveDate"),
                        title=str(cfr_payload.get("citation") or "Current CFR text"),
                        citation=str(cfr_payload.get("citation") or "") or None,
                        canonicalUrl=cfr_payload.get("sourceUrl"),
                        provider="govinfo",
                        verificationStatus=cast(
                            VerificationStatus,
                            cfr_payload.get("verificationStatus") or "verified_primary_source",
                        ),
                        note="Authoritative CFR text retrieved from GovInfo.",
                    )
                )
            except Exception as exc:
                stage_timings_ms["regulatoryGovInfo"] = int((time.perf_counter() - started) * 1000)
                failed.append("govinfo")
                provider_outcomes.append(
                    {
                        "provider": "govinfo",
                        "endpoint": "get_cfr_text",
                        "statusBucket": "provider_error",
                        "error": str(exc),
                    }
                )

        species_hit: dict[str, Any] | None = None
        if self._enable_ecos and self._ecos_client is not None and not agency_guidance_mode:
            attempted.append("ecos")
            started = time.perf_counter()
            try:
                species_search: dict[str, Any] = {"data": []}
                species_ecos_origin: str | None = None
                # Finding 4 (4th rubber-duck pass): collect hits from each
                # variant and rank them by ``hit_count * provenance_factor``
                # (raw/corroborated = 1.0, planner-only = 0.9). Previously
                # the loop broke on the first variant with any hits, so a
                # planner-only variant could win over a later raw/corroborated
                # variant that would have offered stronger query-grounded
                # evidence. The provenance field was stamped on the winner
                # but nothing downstream actually consumed it.
                variant_hits: list[tuple[int, str, str, dict[str, Any]]] = []
                for variant_idx, (
                    species_query,
                    inferred_anchor_type,
                    variant_origin,
                ) in enumerate(_ecos_query_variants(query, planner=planner)):
                    variant_search = await self._ecos_client.search_species(
                        query=species_query,
                        limit=min(limit, 5),
                        match_mode="auto",
                    )
                    variant_data = list(variant_search.get("data") or [])
                    if not variant_data:
                        continue
                    variant_hits.append((variant_idx, inferred_anchor_type, variant_origin, variant_search))
                selected = _rank_ecos_variant_hits(variant_hits)
                if selected is not None:
                    species_search = selected[3]
                    species_anchor_type = selected[1]
                    species_ecos_origin = selected[2]
                stage_timings_ms["regulatoryEcosSearch"] = int((time.perf_counter() - started) * 1000)
                species_data = list(species_search.get("data") or [])
                if species_data:
                    succeeded.append("ecos")
                    species_hit = species_data[0] if isinstance(species_data[0], dict) else None
                    if species_hit is not None:
                        # Stamp provenance so downstream ranking can down-weight
                        # planner-only hits (potential LLM hallucination) vs
                        # query-corroborated hits.
                        species_hit["_ecosProvenance"] = (
                            "corroborated" if species_ecos_origin == "raw" else "planner_only"
                        )
                        anchored_subject_terms = _extract_subject_terms(
                            str(species_hit.get("commonName") or "") or None,
                            str(species_hit.get("scientificName") or "") or None,
                        )
                        subject = str(
                            species_hit.get("commonName") or species_hit.get("scientificName") or subject or query
                        )
                        structured_sources.append(
                            _source_record_from_regulatory_document(
                                {
                                    "title": (
                                        "ECOS species profile: "
                                        + str(
                                            species_hit.get("commonName")
                                            or species_hit.get("scientificName")
                                            or "Species"
                                        )
                                    ),
                                    "url": species_hit.get("profileUrl"),
                                    "speciesId": species_hit.get("speciesId"),
                                    "documentType": "Species Profile",
                                    "verificationStatus": "verified_metadata",
                                },
                                provider="ecos",
                                topical_relevance="on_topic",
                            )
                        )
                        profile = await self._ecos_client.get_species_profile(species_id=species_hit["speciesId"])
                        documents = await self._ecos_client.list_species_documents(species_id=species_hit["speciesId"])
                        for entity in profile.get("speciesEntities") or []:
                            if isinstance(entity, dict) and entity.get("listingDate"):
                                timeline_events.append(
                                    RegulatoryTimelineEvent(
                                        eventType="species_listing",
                                        eventDate=entity.get("listingDate"),
                                        title=str(entity.get("status") or "Species listing event"),
                                        provider="ecos",
                                        verificationStatus="verified_metadata",
                                        canonicalUrl=species_hit.get("profileUrl"),
                                        note=str(entity.get("statusCategory") or "ECOS species entity metadata."),
                                    )
                                )
                        doc_list = list(documents.get("data") or [])
                        if not doc_list:
                            evidence_gaps.append(
                                "ECOS returned a species profile but no dossier documents were available."
                            )
                        for document in doc_list[: max(limit, 10)]:
                            if not isinstance(document, dict):
                                continue
                            structured_sources.append(
                                _source_record_from_regulatory_document(
                                    document,
                                    provider="ecos",
                                    topical_relevance="on_topic",
                                )
                            )
                            timeline_events.append(
                                RegulatoryTimelineEvent(
                                    eventType=str(document.get("documentKind") or "regulatory_document"),
                                    eventDate=document.get("documentDate"),
                                    title=str(document.get("title") or document.get("documentKind") or "ECOS document"),
                                    citation=str(document.get("frCitation") or "") or None,
                                    canonicalUrl=document.get("url"),
                                    provider="ecos",
                                    verificationStatus="verified_metadata",
                                    note=str(document.get("documentType") or "ECOS dossier document."),
                                )
                            )
                else:
                    zero_results.append("ecos")
            except Exception as exc:
                stage_timings_ms["regulatoryEcosSearch"] = int((time.perf_counter() - started) * 1000)
                failed.append("ecos")
                provider_outcomes.append(
                    {
                        "provider": "ecos",
                        "endpoint": "search_species",
                        "statusBucket": "provider_error",
                        "error": str(exc),
                    }
                )

        if not anchored_subject_terms:
            anchored_subject_terms = _regulatory_query_subject_terms(query)
        fr_query = subject or query
        if self._enable_federal_register and self._federal_register_client is not None:
            attempted.append("federal_register")
            started = time.perf_counter()
            try:
                fr_search = await self._federal_register_client.search_documents(
                    query=fr_query,
                    limit=min(max(limit, 5), 10),
                )
                stage_timings_ms["regulatoryFederalRegister"] = int((time.perf_counter() - started) * 1000)
                documents = list(fr_search.get("data") or [])
                if documents:
                    succeeded.append("federal_register")
                    for document in documents[:limit]:
                        if not isinstance(document, dict):
                            continue
                        if not _regulatory_document_matches_subject(
                            document,
                            subject_terms=anchored_subject_terms,
                            priority_terms=agency_priority_terms,
                            cfr_citation=cfr_citation,
                        ):
                            filtered_document_count += 1
                            off_topic_lead = _source_record_from_regulatory_document(
                                document,
                                provider="federal_register",
                                topical_relevance="off_topic",
                            )
                            anchor_hint = subject or query
                            reason_note = (
                                "Filtered from verified regulatory timeline because it did not match the anchored "
                                f"subject '{anchor_hint}'."
                            )
                            existing_note = off_topic_lead.note
                            off_topic_lead = off_topic_lead.model_copy(
                                update={
                                    "note": (f"{reason_note} {existing_note}".strip() if existing_note else reason_note)
                                }
                            )
                            candidate_leads.append(off_topic_lead)
                            continue
                        if cfr_request and govinfo_matched and current_text_requested and not history_requested:
                            context_lead = _source_record_from_regulatory_document(
                                document,
                                provider="federal_register",
                                topical_relevance="weak_match",
                            )
                            context_note = (
                                "Demoted to background context because authoritative current CFR text was already "
                                "retrieved from GovInfo."
                            )
                            existing_note = context_lead.note
                            context_lead = context_lead.model_copy(
                                update={
                                    "note": (
                                        f"{context_note} {existing_note}".strip() if existing_note else context_note
                                    )
                                }
                            )
                            candidate_leads.append(context_lead)
                            continue
                        structured_sources.append(
                            _source_record_from_regulatory_document(
                                document,
                                provider="federal_register",
                                topical_relevance="on_topic",
                            )
                        )
                        federal_register_matched = True
                        timeline_events.append(
                            RegulatoryTimelineEvent(
                                eventType="rulemaking_history",
                                eventDate=document.get("publicationDate"),
                                title=str(
                                    document.get("title")
                                    or document.get("documentNumber")
                                    or "Federal Register document"
                                ),
                                citation=str(document.get("citation") or document.get("documentNumber") or "") or None,
                                canonicalUrl=(
                                    document.get("htmlUrl") or document.get("govInfoLink") or document.get("pdfUrl")
                                ),
                                provider="federal_register",
                                verificationStatus=cast(
                                    VerificationStatus,
                                    document.get("verificationStatus") or "verified_metadata",
                                ),
                                note=(
                                    (
                                        "Rulemaking history/context from the Federal Register. "
                                        + ", ".join(document.get("cfrReferences") or [])
                                    )
                                    if isinstance(document.get("cfrReferences"), list) and document.get("cfrReferences")
                                    else "Rulemaking history/context from the Federal Register."
                                ),
                            )
                        )
                else:
                    zero_results.append("federal_register")
            except Exception as exc:
                stage_timings_ms["regulatoryFederalRegister"] = int((time.perf_counter() - started) * 1000)
                failed.append("federal_register")
                provider_outcomes.append(
                    {
                        "provider": "federal_register",
                        "endpoint": "search_documents",
                        "statusBucket": "provider_error",
                        "error": str(exc),
                    }
                )

        if (
            self._enable_govinfo_cfr
            and self._govinfo_client is not None
            and not govinfo_matched
            and not federal_register_matched
        ):
            if "govinfo" not in attempted:
                attempted.append("govinfo")
            started = time.perf_counter()
            try:
                govinfo_search = await self._govinfo_client.search_federal_register_documents(
                    query=fr_query,
                    limit=min(max(limit, 5), 10),
                )
                stage_timings_ms["regulatoryGovInfoSearch"] = int((time.perf_counter() - started) * 1000)
                documents = list(govinfo_search.get("data") or [])
                if documents:
                    succeeded.append("govinfo")
                    for document in documents[:limit]:
                        if not isinstance(document, dict):
                            continue
                        if not _regulatory_document_matches_subject(
                            document,
                            subject_terms=anchored_subject_terms,
                            priority_terms=agency_priority_terms,
                            cfr_citation=cfr_citation,
                        ):
                            continue
                        structured_sources.append(
                            _source_record_from_regulatory_document(
                                document,
                                provider="govinfo",
                                topical_relevance="on_topic",
                            )
                        )
                        govinfo_matched = True
                        timeline_events.append(
                            RegulatoryTimelineEvent(
                                eventType="rulemaking_history",
                                eventDate=document.get("publicationDate"),
                                title=str(document.get("title") or document.get("citation") or "GovInfo document"),
                                citation=str(document.get("citation") or document.get("documentNumber") or "") or None,
                                canonicalUrl=document.get("sourceUrl"),
                                provider="govinfo",
                                verificationStatus=cast(
                                    VerificationStatus,
                                    document.get("verificationStatus") or "verified_metadata",
                                ),
                                note=str(
                                    document.get("note") or "GovInfo Federal Register primary-source recovery hit."
                                ),
                            )
                        )
                else:
                    zero_results.append("govinfo")
            except Exception as exc:
                stage_timings_ms["regulatoryGovInfoSearch"] = int((time.perf_counter() - started) * 1000)
                failed.append("govinfo")
                provider_outcomes.append(
                    {
                        "provider": "govinfo",
                        "endpoint": "search_federal_register_documents",
                        "statusBucket": "provider_error",
                        "error": str(exc),
                    }
                )

        structured_sources = _dedupe_structured_sources(structured_sources)
        # Species-dossier demotion: when the planner says we are answering a
        # species dossier question, regulatory hits that do NOT mention the
        # species must not be presented as on_topic evidence.
        if _species_dossier_mode and _subject_card is not None:
            demoted_sources: list[StructuredSourceRecord] = []
            species_demotion_count = 0
            for src_record in structured_sources:
                if src_record.topical_relevance != "on_topic":
                    demoted_sources.append(src_record)
                    continue
                doc_stub = {
                    "title": src_record.title or "",
                    "summary": src_record.note or "",
                    "citation": src_record.citation_text or "",
                }
                if species_mentioned(doc_stub, _subject_card):
                    demoted_sources.append(src_record)
                    continue
                species_demotion_count += 1
                demoted_sources.append(
                    src_record.model_copy(
                        update={
                            "topical_relevance": "weak_match",
                            "why_classified_as_weak_match": ("species_not_specifically_addressed"),
                        }
                    )
                )
            structured_sources = demoted_sources
            if species_demotion_count:
                evidence_gaps.append(
                    "Some regulatory documents were demoted to weak_match because "
                    "they do not specifically address the subject species."
                )
        timeline_events = sorted(timeline_events, key=lambda event: str(event.event_date or "9999-99-99"))
        if not timeline_events:
            evidence_gaps.append(
                "No primary-source regulatory timeline could be reconstructed from the "
                "currently enabled regulatory providers."
            )
        if agency_guidance_mode and not structured_sources:
            evidence_gaps.append(
                "No authoritative agency guidance match was found from "
                "GovInfo/Federal Register for the requested title."
            )
        if filtered_document_count > 0:
            evidence_gaps.append(
                f"Filtered {filtered_document_count} Federal Register hit(s) that did not match the anchored "
                "regulatory subject."
            )
        if species_hit is None and "ecos" in zero_results:
            evidence_gaps.append("No ECOS species dossier match was found for the query.")
        govinfo_text_retrieved = govinfo_matched and any(
            s.full_text_url_found for s in structured_sources if s.provider == "govinfo"
        )
        current_text_satisfied = (not current_text_requested) or govinfo_text_retrieved
        history_only = bool(current_text_requested and not current_text_satisfied and timeline_events)
        if current_text_requested and not govinfo_matched:
            evidence_gaps.append(
                "Current codified CFR text was not verified from GovInfo, so the result is limited to history/context."
            )
        primary_document_coverage = PrimaryDocumentCoverage(
            currentTextRequested=current_text_requested,
            govinfoAttempted="govinfo" in attempted,
            govinfoMatched=govinfo_matched,
            cfrAttempted=bool(cfr_request and "govinfo" in attempted),
            cfrMatched=bool(cfr_request and govinfo_matched),
            federalRegisterAttempted="federal_register" in attempted,
            federalRegisterMatched=federal_register_matched,
            historyOnly=history_only,
            currentTextSatisfied=current_text_satisfied,
        )

        provider_selection = self._provider_bundle_for_profile(latency_profile).selection_metadata()
        # P0-3 item 1: reconcile zero_results against succeeded/failed so a provider
        # recovered via a fallback path (e.g., govinfo) cannot remain classified as
        # returning zero results. See docs/ux-remediation-checklist.md.
        zero_results = [p for p in attempted if p not in succeeded and p not in failed]
        coverage_summary = CoverageSummary(
            providersAttempted=attempted,
            providersSucceeded=succeeded,
            providersFailed=failed,
            providersZeroResults=zero_results,
            likelyCompleteness=("partial" if structured_sources else ("incomplete" if failed else "unknown")),
            searchMode="regulatory_primary_source",
            retrievalNotes=[
                "Regulatory mode prioritizes ECOS, Federal Register, and CFR primary sources before broader synthesis."
            ],
            summaryLine=_coverage_summary_line(
                attempted=attempted,
                failed=failed,
                zero_results=zero_results,
                likely_completeness=("partial" if structured_sources else ("incomplete" if failed else "unknown")),
            ),
            primaryDocumentCoverage=primary_document_coverage,
        )
        failure_summary = None
        if failed:
            failure_summary = FailureSummary(
                outcome="fallback_success" if structured_sources else "total_failure",
                whatFailed="One or more regulatory providers failed during primary-source retrieval.",
                whatStillWorked=(
                    "Other regulatory providers still returned evidence."
                    if structured_sources
                    else "No regulatory provider returned usable source records."
                ),
                fallbackAttempted=True,
                fallbackMode="regulatory_primary_source",
                primaryPathFailureReason=", ".join(sorted(failed)),
                completenessImpact=(
                    "The regulatory history may be incomplete because at least one primary-source provider failed."
                ),
                recommendedNextAction="review_partial_results",
            )
        result_status = (
            "succeeded"
            if structured_sources and current_text_satisfied and failure_summary is None
            else ("partial" if structured_sources else "abstained")
        )
        anchor_type = (
            "agency_guidance_title"
            if agency_guidance_mode
            else (
                "cfr_citation"
                if cfr_request
                else (
                    (species_anchor_type or "species_common_name")
                    if species_hit is not None
                    else ("regulatory_subject_terms" if anchored_subject_terms else None)
                )
            )
        )
        anchor_strength: Literal["high", "medium", "low"] | None = (
            "high"
            if cfr_request or species_hit is not None or agency_guidance_mode
            else ("medium" if anchored_subject_terms else None)
        )
        retrieval_hypotheses = _regulatory_retrieval_hypotheses(
            query=query,
            planner=planner,
            subject=subject,
            anchor_type=anchor_type,
            cfr_citation=cfr_citation,
            current_text_requested=current_text_requested,
            history_requested=history_requested,
            agency_guidance_mode=agency_guidance_mode,
        )
        # Ensure hybrid policy+science questions surface a dedicated hypothesis
        # so downstream tools can see that both regulatory and literature
        # providers should be consulted.
        if planner.regulatory_intent == "hybrid_regulatory_plus_literature" and not any(
            "hybrid_policy_science" in str(entry) for entry in retrieval_hypotheses
        ):
            retrieval_hypotheses.append(
                "hybrid_policy_science: combine regulatory primary sources (Federal Register, CFR, "
                "agency guidance) with peer-reviewed literature to answer policy+evidence questions."
            )
        # Compute subject-chain evidence gaps from the regulatory hits we
        # retrieved. This surfaces e.g. "ECOS dossier found but no recovery
        # plan in primary sources" so the caller knows which rung is missing.
        subject_chain_gap_notes: list[str] = []
        if _subject_card is not None:
            document_stubs = [
                {
                    "title": record.title or "",
                    "summary": record.note or "",
                    "citation": record.citation_text or "",
                }
                for record in structured_sources
            ]
            subject_chain_gap_notes = compute_subject_chain_gaps(
                card=_subject_card,
                regulatory_intent=planner.regulatory_intent,
                documents=document_stubs,
            )
            if subject_chain_gap_notes:
                evidence_gaps.extend(subject_chain_gap_notes)
        strategy_metadata = SearchStrategyMetadata(
            intent=planner_intent,
            intentSource=planner.intent_source,
            intentConfidence=planner.intent_confidence,
            intentCandidates=planner.intent_candidates,
            secondaryIntents=planner.secondary_intents,
            routingConfidence=planner.routing_confidence,
            querySpecificity=planner.query_specificity,
            ambiguityLevel=planner.ambiguity_level,
            intentFamily=planner.intent_family,
            regulatoryIntent=planner.regulatory_intent,
            regulatorySubintent=planner.regulatory_subintent,
            subjectCard=_subject_card,
            subjectChainGaps=subject_chain_gap_notes,
            intentRationale=planner.intent_rationale,
            latencyProfile=latency_profile,
            normalizedQuery=query,
            queryVariantsTried=[query],
            retrievalHypotheses=retrieval_hypotheses,
            acceptedExpansions=[],
            rejectedExpansions=[],
            speculativeExpansions=[],
            providersUsed=succeeded,
            paidProvidersUsed=_paid_providers_used(succeeded),
            resultCoverage=("moderate" if structured_sources else "none"),
            driftWarnings=evidence_gaps,
            providerBudgetApplied={},
            providerOutcomes=provider_outcomes,
            stageTimingsMs=stage_timings_ms,
            recoveryAttempted=False,
            recoveryPath=[planner_intent],
            anchorType=anchor_type,
            anchorStrength=anchor_strength,
            anchoredSubject=subject or query,
            normalizationWarnings=normalization_warnings,
            repairedInputs=repaired_inputs,
            bestNextInternalAction=_best_next_internal_action(
                intent=planner_intent,
                has_sources=_has_on_topic_sources(structured_sources),
                result_status=result_status,
            ),
            **provider_selection,
        )
        evidence_records = _evidence_from_source_records(structured_sources)
        lead_records = _records_with_lead_reasons(
            _dedupe_structured_sources([*candidate_leads, *_candidate_leads_from_source_records(structured_sources)])[
                : max(limit, 6)
            ]
        )
        response = SmartSearchResponse(
            results=[],
            searchSessionId=search_session_id or "pending",
            strategyMetadata=strategy_metadata,
            nextStepHint=(
                "Inspect structuredSources and regulatoryTimeline first. Use get_document_text_ecos, "
                "get_federal_register_document, or get_cfr_text for deeper primary-source reading."
            ),
            agentHints=AgentHints(
                nextToolCandidates=[
                    "get_species_profile_ecos",
                    "list_species_documents_ecos",
                    "search_federal_register",
                    "get_cfr_text",
                ],
                whyThisNextStep=(
                    "Regulatory mode returns primary-source leads first, so the next move is direct document retrieval."
                ),
                safeRetry=(
                    "Retry with a species common name, scientific name, FR citation, or "
                    "explicit CFR citation for stronger routing."
                ),
                warnings=evidence_gaps,
            ),
            resourceUris=[],
            verifiedFindings=_verified_findings_from_source_records(structured_sources),
            likelyUnverified=_likely_unverified_from_source_records(structured_sources),
            answerability=_answerability_from_source_records(
                result_status=result_status,
                evidence=evidence_records,
                leads=lead_records,
                evidence_gaps=evidence_gaps,
            ),
            routingSummary=_routing_summary_from_strategy(
                strategy_metadata=strategy_metadata,
                coverage_summary=coverage_summary,
                result_status=result_status,
                evidence_gaps=evidence_gaps,
            ),
            evidence=evidence_records,
            leads=lead_records,
            candidateLeads=lead_records,
            evidenceGaps=evidence_gaps,
            structuredSources=structured_sources[: max(limit, len(structured_sources))],
            coverageSummary=coverage_summary,
            failureSummary=failure_summary,
            regulatoryTimeline=RegulatoryTimeline(
                query=query,
                subject=subject,
                events=timeline_events,
                evidenceGaps=evidence_gaps,
            ),
            resultStatus=result_status,
            hasInspectableSources=_has_inspectable_sources(structured_sources),
            bestNextInternalAction=_best_next_internal_action(
                intent=planner_intent,
                has_sources=_has_on_topic_sources(structured_sources),
                result_status=result_status,
            ),
        )
        record = await self._workspace_registry.asave_result_set(
            source_tool="search_papers_smart",
            payload=dump_jsonable(response),
            query=query,
            metadata={"strategyMetadata": dump_jsonable(strategy_metadata), "originalQuery": query},
            search_session_id=search_session_id,
        )
        response.search_session_id = record.search_session_id
        response.resource_uris = build_resource_uris(
            "search_papers_smart",
            dump_jsonable(response),
            record.search_session_id,
        )
        final_response_dict = self._workspace_registry.attach_source_aliases(dump_jsonable(response))
        record.payload = final_response_dict
        record.metadata["strategyMetadata"] = final_response_dict["strategyMetadata"]
        await self._emit_smart_search_status(
            ctx=ctx,
            request_id=request_id,
            progress=100,
            message="Regulatory smart search complete",
            detail=(f"Regulatory smart search complete. searchSessionId={record.search_session_id}."),
        )
        return final_response_dict

    async def _search_known_item(
        self,
        *,
        query: str,
        limit: int,
        planner: PlannerDecision,
        provider_plan: list[str] | None,
        provider_budget: ProviderBudgetState | None,
        search_session_id: str | None,
        latency_profile: LatencyProfile,
        request_id: str,
        include_enrichment: bool,
        provider_outcomes: list[dict[str, Any]],
        stage_timings_ms: dict[str, int],
        normalization_warnings: list[str],
        repaired_inputs: dict[str, Any],
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
                planner=planner,
                provider_plan=provider_plan,
                provider_budget=provider_budget,
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
        if include_enrichment and self._enrichment_service is not None:
            await self._emit_smart_search_status(
                ctx=ctx,
                request_id=request_id,
                progress=70,
                message="Applying paper enrichment",
                detail=("Enriching the resolved known item with Crossref, Unpaywall, and OpenAlex metadata."),
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
        provider_selection = self._provider_bundle_for_profile(latency_profile).selection_metadata()
        providers_used = [str(known_item.get("source") or "semantic_scholar")]
        provider_fallback_warnings = _smart_provider_fallback_warnings(
            provider_selection=provider_selection,
            provider_outcomes=provider_outcomes,
        )
        strategy_metadata = SearchStrategyMetadata(
            intent=planner.intent,
            intentSource=planner.intent_source,
            intentConfidence=planner.intent_confidence,
            intentCandidates=planner.intent_candidates,
            secondaryIntents=planner.secondary_intents,
            routingConfidence=planner.routing_confidence,
            querySpecificity=planner.query_specificity,
            ambiguityLevel=planner.ambiguity_level,
            intentFamily=planner.intent_family,
            regulatoryIntent=planner.regulatory_intent,
            intentRationale=planner.intent_rationale,
            latencyProfile=latency_profile,
            normalizedQuery=query,
            queryVariantsTried=[query],
            retrievalHypotheses=[],
            acceptedExpansions=[],
            rejectedExpansions=[],
            speculativeExpansions=[],
            providersUsed=providers_used,
            paidProvidersUsed=_paid_providers_used(providers_used),
            resultCoverage="known_item",
            driftWarnings=(
                []
                if resolution_strategy == "citation_resolution"
                else [_known_item_recovery_warning(resolution_strategy)]
            )
            + provider_fallback_warnings,
            providerBudgetApplied=(provider_budget.to_dict() if provider_budget else {}),
            providerOutcomes=provider_outcomes,
            stageTimingsMs=stage_timings_ms,
            recoveryAttempted=planner.intent_source == "fallback_recovery",
            recoveryPath=[planner.intent],
            anchorType=resolution_strategy,
            anchorStrength=_anchor_strength_for_resolution(resolution_strategy),
            anchoredSubject=str(known_item.get("title") or query),
            knownItemResolutionState=_known_item_resolution_state_for_strategy(
                resolution_strategy=resolution_strategy,
                known_item=known_item,
                query=query,
            ),
            normalizationWarnings=normalization_warnings,
            repairedInputs=repaired_inputs,
            bestNextInternalAction="get_paper_details",
            **provider_selection,
        )
        source_records = [
            _source_record_from_paper(
                hit.paper,
                note=hit.why_matched,
                topical_relevance=hit.topical_relevance,
            )
            for hit in [hit][:limit]
        ]
        evidence_records = _evidence_from_source_records(source_records)
        lead_records = _candidate_leads_from_source_records(source_records)
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
            verifiedFindings=_verified_findings_from_source_records(source_records),
            likelyUnverified=_likely_unverified_from_source_records(source_records),
            answerability=_answerability_from_source_records(
                result_status="succeeded",
                evidence=evidence_records,
                leads=lead_records,
                evidence_gaps=[],
            ),
            routingSummary=_routing_summary_from_strategy(
                strategy_metadata=strategy_metadata,
                coverage_summary=None,
                result_status="succeeded",
                evidence_gaps=[],
            ),
            evidence=evidence_records,
            leads=lead_records,
            candidateLeads=lead_records,
            structuredSources=source_records,
            resultStatus="succeeded",
            hasInspectableSources=True,
            bestNextInternalAction="get_paper_details",
        )
        if provider_fallback_warnings:
            response.agent_hints.warnings.extend(provider_fallback_warnings)
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
        final_response_dict = self._workspace_registry.attach_source_aliases(dump_jsonable(response))
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

        parsed = parse_citation(query)
        resolution_queries = _known_item_resolution_queries(query, parsed)

        for candidate_query in resolution_queries:
            try:
                semantic_match = dump_jsonable(
                    await self._client.search_papers_match(query=candidate_query, fields=None)
                )
            except Exception:
                semantic_match = None
            if isinstance(semantic_match, dict) and semantic_match.get("paperId"):
                return semantic_match, str(semantic_match.get("matchStrategy") or "semantic_title_match")

        if self._enable_openalex and self._openalex_client is not None:
            for candidate_query in resolution_queries:
                try:
                    autocomplete = await self._openalex_client.paper_autocomplete(query=candidate_query, limit=5)
                except Exception:
                    autocomplete = None
                if isinstance(autocomplete, dict):
                    for match in autocomplete.get("matches") or []:
                        if not isinstance(match, dict):
                            continue
                        match_id = str(match.get("id") or "").strip()
                        match_title = str(match.get("displayName") or "").strip()
                        if not match_id or _known_item_title_similarity(candidate_query, match_title) < 0.72:
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
                    openalex_search = await self._openalex_client.search(query=candidate_query, limit=3)
                except Exception:
                    openalex_search = None
                if isinstance(openalex_search, dict):
                    for paper in openalex_search.get("data") or []:
                        if not isinstance(paper, dict) or not paper.get("paperId"):
                            continue
                        if _known_item_title_similarity(candidate_query, str(paper.get("title") or "")) < 0.58:
                            continue
                        return paper, "openalex_search"
        return None, "none"

    async def _fallback_known_item_search(
        self,
        *,
        query: str,
        limit: int,
        planner: PlannerDecision,
        provider_plan: list[str] | None,
        provider_budget: ProviderBudgetState | None,
        search_session_id: str | None,
        latency_profile: LatencyProfile,
        request_id: str,
        include_enrichment: bool,
        provider_outcomes: list[dict[str, Any]],
        stage_timings_ms: dict[str, int],
        normalization_warnings: list[str],
        repaired_inputs: dict[str, Any],
        ctx: Context | None,
    ) -> dict[str, Any]:
        profile_settings = self._config.latency_profile_settings(latency_profile)
        provider_bundle = self._provider_bundle_for_profile(latency_profile)
        retrieval_started = time.perf_counter()
        batch = await retrieve_variant(
            variant=query,
            variant_source="from_input",
            intent=planner.intent,
            year=None,
            venue=None,
            enable_core=self._enable_core,
            enable_semantic_scholar=self._enable_semantic_scholar,
            enable_openalex=self._enable_openalex,
            enable_scholarapi=self._enable_scholarapi,
            enable_arxiv=self._enable_arxiv,
            enable_serpapi=self._enable_serpapi,
            core_client=self._core_client,
            semantic_client=self._client,
            openalex_client=self._openalex_client,
            scholarapi_client=self._scholarapi_client,
            arxiv_client=self._arxiv_client,
            serpapi_client=self._serpapi_client,
            provider_plan=provider_plan,
            widened=True,
            is_expansion=False,
            allow_serpapi=(self._enable_serpapi and profile_settings.allow_serpapi_on_input),
            latency_profile=latency_profile,
            provider_registry=self._provider_registry,
            provider_budget=provider_budget,
            request_outcomes=provider_outcomes,
            request_id=request_id,
        )
        stage_timings_ms["knownItemFallbackRetrieval"] = int((time.perf_counter() - retrieval_started) * 1000)

        merged_candidates = merge_candidates(batch.candidates)
        rerank_bundle = (
            provider_bundle
            if profile_settings.use_embedding_rerank and provider_bundle.supports_embeddings()
            else self._deterministic_bundle
        )
        ranked_candidates = await rerank_candidates(
            query=query,
            merged_candidates=merged_candidates,
            provider_bundle=rerank_bundle,
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

        provider_selection = self._provider_bundle_for_profile(latency_profile).selection_metadata()
        providers_used = sorted(batch.providers_used)
        provider_fallback_warnings = _smart_provider_fallback_warnings(
            provider_selection=provider_selection,
            provider_outcomes=provider_outcomes,
        )
        strategy_metadata = SearchStrategyMetadata(
            intent=planner.intent,
            intentSource=planner.intent_source,
            intentConfidence=planner.intent_confidence,
            intentCandidates=planner.intent_candidates,
            secondaryIntents=planner.secondary_intents,
            routingConfidence=planner.routing_confidence,
            querySpecificity=planner.query_specificity,
            ambiguityLevel=planner.ambiguity_level,
            intentFamily=planner.intent_family,
            regulatoryIntent=planner.regulatory_intent,
            intentRationale=planner.intent_rationale,
            latencyProfile=latency_profile,
            normalizedQuery=query,
            queryVariantsTried=[query],
            retrievalHypotheses=[],
            acceptedExpansions=[],
            rejectedExpansions=[],
            speculativeExpansions=[],
            providersUsed=providers_used,
            paidProvidersUsed=_paid_providers_used(providers_used),
            resultCoverage=("narrow" if smart_hits else "none"),
            driftWarnings=[
                "Exact known-item resolution was not confident, so the smart workflow fell back to a broader candidate "
                "set. Verify title, year, and venue before treating a result as canonical."
            ]
            + provider_fallback_warnings,
            providerBudgetApplied=(provider_budget.to_dict() if provider_budget else {}),
            providerOutcomes=provider_outcomes,
            stageTimingsMs=stage_timings_ms,
            recoveryAttempted=True,
            recoveryPath=["known_item", "discovery"],
            recoveryReason="No exact known-item anchor was confirmed.",
            anchorType="candidate_set",
            anchorStrength="low",
            anchoredSubject=query,
            knownItemResolutionState="needs_disambiguation",
            normalizationWarnings=normalization_warnings,
            repairedInputs=repaired_inputs,
            bestNextInternalAction=("ask_result_set" if smart_hits else "search_papers_smart"),
            **provider_selection,
        )
        source_records = [
            _source_record_from_paper(
                hit.paper,
                note=hit.why_matched,
                topical_relevance=hit.topical_relevance,
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
                "Pick the closest candidate, then inspect details or use exact-title and identifier tools to "
                "confirm the "
                "anchor before expanding citations."
            ),
            agentHints=build_agent_hints(
                "search_papers_smart",
                {"brokerMetadata": {"resultQuality": "low_relevance" if not smart_hits else "unknown"}},
            ),
            resourceUris=[],
            verifiedFindings=_verified_findings_from_source_records(source_records),
            likelyUnverified=_likely_unverified_from_source_records(source_records),
            answerability=_answerability_from_source_records(
                result_status=result_status,
                evidence=[],
                leads=lead_records,
                evidence_gaps=list(strategy_metadata.drift_warnings),
            ),
            routingSummary=_routing_summary_from_strategy(
                strategy_metadata=strategy_metadata,
                coverage_summary=None,
                result_status=result_status,
                evidence_gaps=list(strategy_metadata.drift_warnings),
            ),
            evidence=[],
            leads=lead_records,
            candidateLeads=lead_records,
            evidenceGaps=list(strategy_metadata.drift_warnings),
            structuredSources=source_records,
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
        final_response_dict = self._workspace_registry.attach_source_aliases(dump_jsonable(response))
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


def _source_record_from_paper(
    paper: Paper,
    *,
    note: str | None = None,
    topical_relevance: Literal["on_topic", "weak_match", "off_topic"] | None = None,
    llm_classification: Literal["on_topic", "weak_match", "off_topic"] | None = None,
    classification_source: Literal["deterministic", "llm", "llm_tiebreaker"] | None = None,
    relevance_source: str | None = None,
    relevance_confidence: float | None = None,
    relevance_reason: str | None = None,
    classification_rationale: str | None = None,
) -> StructuredSourceRecord:
    source_id = str(paper.canonical_id or paper.paper_id or "") or None
    return StructuredSourceRecord(
        sourceId=source_id,
        title=paper.title,
        provider=paper.source,
        sourceType=paper.source_type,
        verificationStatus=paper.verification_status,
        accessStatus=paper.access_status,
        topicalRelevance=topical_relevance,
        llmClassification=llm_classification,
        classificationSource=classification_source,
        confidence=paper.confidence,
        isPrimarySource=paper.is_primary_source,
        canonicalUrl=paper.canonical_url,
        retrievedUrl=paper.retrieved_url,
        fullTextUrlFound=paper.full_text_url_found,
        bodyTextEmbedded=paper.body_text_embedded,
        qaReadableText=paper.qa_readable_text,
        abstractObserved=paper.abstract_observed,
        openAccessRoute=paper.open_access_route,
        citationText=str(paper.canonical_id or paper.paper_id or "") or None,
        citation=_citation_record_from_paper(paper),
        date=str(paper.publication_date or paper.year or "") or None,
        note=note,
        relevanceSource=cast(Any, relevance_source) if relevance_source else None,
        relevanceConfidence=relevance_confidence,
        relevanceReason=relevance_reason,
        classificationRationale=classification_rationale,
    )


def _source_record_from_regulatory_document(
    document: dict[str, Any],
    *,
    provider: str,
    topical_relevance: Literal["on_topic", "weak_match", "off_topic"] | None = "on_topic",
    why_classified_as_weak_match: str | None = None,
) -> StructuredSourceRecord:
    title = str(
        document.get("title") or document.get("citation") or document.get("documentNumber") or "Regulatory source"
    )
    canonical_url = (
        document.get("url")
        or document.get("htmlUrl")
        or document.get("sourceUrl")
        or document.get("govInfoLink")
        or document.get("pdfUrl")
    )
    citation = (
        str(document.get("citation") or document.get("frCitation") or document.get("documentNumber") or "") or None
    )
    source_id = citation or str(document.get("documentNumber") or document.get("speciesId") or "") or None
    date = document.get("documentDate") or document.get("publicationDate") or document.get("effectiveDate")
    note = None
    if provider == "ecos":
        note = str(document.get("documentType") or document.get("documentKind") or "ECOS dossier document")
    elif provider == "federal_register":
        cfr_refs = document.get("cfrReferences") or []
        note = (
            ", ".join(cfr_refs)
            if isinstance(cfr_refs, list) and cfr_refs
            else "Federal Register primary-source discovery hit"
        )
    elif provider == "govinfo":
        note = str(
            document.get("note")
            or ("Authoritative CFR text" if document.get("markdown") else "GovInfo primary-source discovery hit")
        )
    has_inline_body = bool(document.get("markdown"))
    has_url = bool(canonical_url)
    family_match = document.get("_documentFamilyMatch")
    family_boost = document.get("_documentFamilyBoost")
    if has_inline_body:
        access_status: str = "body_text_embedded"
    elif has_url:
        access_status = "url_verified"
    else:
        access_status = "access_unverified"
    if has_inline_body:
        verification_status_default = "verified_primary_source" if provider == "govinfo" else "verified_metadata"
    else:
        verification_status_default = "verified_metadata"
    return StructuredSourceRecord(
        sourceId=source_id,
        title=title,
        provider=provider,
        sourceType="primary_regulatory",
        verificationStatus=str(document.get("verificationStatus") or verification_status_default),
        accessStatus=access_status,
        confidence="high" if (provider == "govinfo" and has_inline_body) else "medium",
        topicalRelevance=topical_relevance,
        isPrimarySource=True,
        canonicalUrl=canonical_url,
        retrievedUrl=canonical_url,
        fullTextUrlFound=has_url,
        fullTextRetrieved=has_inline_body,
        bodyTextEmbedded=has_inline_body,
        qaReadableText=has_inline_body,
        abstractObserved=False,
        openAccessRoute=("non_oa_or_unconfirmed" if (has_inline_body or has_url) else "unknown"),
        citationText=citation,
        citation=_citation_record_from_regulatory_document(
            document,
            provider=provider,
            citation_text=citation,
            canonical_url=str(canonical_url or "") or None,
        ),
        date=str(date or "") or None,
        note=note,
        whyClassifiedAsWeakMatch=why_classified_as_weak_match,
        documentFamilyMatch=str(family_match) if family_match else None,
        documentFamilyBoost=float(family_boost) if isinstance(family_boost, (int, float)) else None,
    )


def _compute_top_recommendation(
    *,
    question: str,
    resolved_question_mode: str,
    answer_status: str,
    evidence: list[EvidenceItem],
    source_records: list[StructuredSourceRecord],
    strong_evidence_ids: set[str],
    anchored_selection_source_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Build the P1-2 ``topRecommendation`` payload for selection follow-ups.

    Returns ``None`` unless:

    * ``resolved_question_mode`` is ``comparison`` or ``selection``;
    * ``answer_status`` is ``answered`` (mirrors the grounded gate; abstained
      or insufficient_evidence paths never carry a recommendation);
    * at least two strong, on-topic evidence items are available;
    * and at least one paper scores above zero on the inferred axis.
    """
    if resolved_question_mode not in {"comparison", "selection"}:
        return None
    anchored_selection_source_ids = {
        identifier for identifier in (anchored_selection_source_ids or set()) if identifier
    }
    if answer_status != "answered" and not (
        resolved_question_mode == "selection" and len(anchored_selection_source_ids) == 1
    ):
        return None
    strong_on_topic_evidence = [
        item
        for item in evidence
        if str(item.evidence_id or item.paper.paper_id or item.paper.canonical_id or "").strip() in strong_evidence_ids
    ]
    source_record_by_id = {
        str(record.source_id or "").strip(): record for record in source_records if str(record.source_id or "").strip()
    }
    anchored_items = [
        item
        for item in strong_on_topic_evidence
        if str(item.evidence_id or item.paper.paper_id or item.paper.canonical_id or "").strip()
        in anchored_selection_source_ids
    ]
    if resolved_question_mode == "selection" and len(anchored_items) == 1:
        primary_item = anchored_items[0]
        primary_pid = str(
            primary_item.evidence_id or primary_item.paper.paper_id or primary_item.paper.canonical_id or ""
        ).strip()
        rationale = _build_anchored_selection_rationale(primary_item, source_record_by_id.get(primary_pid))
        return {
            "sourceId": primary_pid,
            "comparativeAxis": "authority",
            "recommendationReason": rationale,
            "axisScore": 1.0,
            "alternativeRecommendations": [],
            "axis": "authority",
            "score": 1.0,
            "rationale": rationale,
            "alternatives": [],
        }
    if len(strong_on_topic_evidence) < 2:
        return None
    papers = [item.paper for item in strong_on_topic_evidence]
    relevance_scores: dict[str, float] = {}
    for item in strong_on_topic_evidence:
        pid = str(item.paper.paper_id or item.paper.canonical_id or "").strip()
        if pid:
            relevance_scores[pid] = float(item.relevance_score or 0.0)
    primary_axis = infer_comparative_axis(question)
    primary_scores = score_papers_for_comparative_axis(
        papers,
        question,
        primary_axis,
        relevance_scores=relevance_scores,
    )
    if not primary_scores:
        return None
    primary_pid, primary_score = max(primary_scores.items(), key=lambda kv: kv[1])
    if primary_score <= 0.0:
        return None

    alt_axes: list[str] = [ax for ax in ("recency", "authority", "coverage") if ax != primary_axis]
    alternatives: list[dict[str, Any]] = []
    for alt_axis in alt_axes[:2]:
        alt_scores = score_papers_for_comparative_axis(
            papers,
            question,
            alt_axis,  # type: ignore[arg-type]
            relevance_scores=relevance_scores,
        )
        if not alt_scores:
            continue
        alt_pid, alt_score = max(alt_scores.items(), key=lambda kv: kv[1])
        if alt_score <= 0.0:
            continue
        alternatives.append(
            {
                "sourceId": alt_pid,
                "axis": alt_axis,
                "score": round(alt_score, 4),
            }
        )

    # Find the corresponding evidence item for rationale enrichment.
    ranked_primary_item = cast(
        EvidenceItem | None,
        next(
            (
                item
                for item in strong_on_topic_evidence
                if str(item.paper.paper_id or item.paper.canonical_id or "").strip() == primary_pid
            ),
            None,
        ),
    )
    rationale = _build_top_recommendation_rationale(primary_axis, primary_score, ranked_primary_item)
    comparative_axis = _top_recommendation_axis_label(primary_axis)
    alternative_recommendations = [
        {
            "sourceId": str(entry.get("sourceId") or "").strip(),
            "comparativeAxis": _top_recommendation_axis_label(str(entry.get("axis") or "")),
            "axisScore": entry.get("score"),
        }
        for entry in alternatives
        if str(entry.get("sourceId") or "").strip()
    ]
    return {
        "sourceId": primary_pid,
        "comparativeAxis": comparative_axis,
        "recommendationReason": rationale,
        "axisScore": round(primary_score, 4),
        "alternativeRecommendations": alternative_recommendations,
        "axis": primary_axis,
        "score": round(primary_score, 4),
        "rationale": rationale,
        "alternatives": alternatives,
    }


def _selection_answer_from_recommendation(
    top_recommendation: dict[str, Any],
    source_records: list[StructuredSourceRecord],
) -> str:
    source_id = str(top_recommendation.get("sourceId") or "").strip()
    title = ""
    if source_id:
        matched = next((record for record in source_records if str(record.source_id or "").strip() == source_id), None)
        if matched is not None:
            title = str(matched.title or matched.citation_text or source_id).strip()
    reason = str(top_recommendation.get("recommendationReason") or "").strip()
    if title and reason:
        return f"Start with {title}. {reason}"
    if title:
        return f"Start with {title}."
    if reason:
        return reason
    return "Start with the recommended source from the saved evidence."


def _build_anchored_selection_rationale(
    item: EvidenceItem,
    source_record: StructuredSourceRecord | None,
) -> str:
    source_title = source_record.title if source_record is not None else None
    title = str(item.paper.title or source_title or "").strip()
    citation = str(source_record.citation_text or "").strip() if source_record is not None else ""
    if citation and citation not in title:
        return f"This is the exact verified primary source for the requested codified text ({citation})."
    if title:
        return f"This is the exact verified primary source for the requested codified text ({title})."
    return "This is the exact verified primary source for the requested codified text."


def _top_recommendation_axis_label(axis: str) -> str:
    return {
        "beginner": "beginner_friendly",
        "relevance_fallback": "relevance",
    }.get(axis, axis)


def _build_top_recommendation_rationale(
    axis: str,
    score: float,
    item: EvidenceItem | None,
) -> str:
    base = f"Highest score on axis '{axis}' ({score:.2f})."
    if item is None:
        return base
    paper = item.paper
    if axis == "recency" and paper.year is not None:
        return f"{base} Published in {paper.year}."
    if axis == "authority" and paper.citation_count is not None:
        return f"{base} Citation count: {paper.citation_count}."
    if axis == "beginner":
        return f"{base} Title signals beginner-friendly framing."
    if axis == "coverage":
        return f"{base} Broadest term overlap with the question."
    return base


def _verified_findings_from_source_records(records: list[StructuredSourceRecord]) -> list[str]:
    findings: list[str] = []
    for record in records:
        if record.verification_status not in {"verified_primary_source", "verified_metadata"}:
            continue
        if record.topical_relevance != "on_topic":
            continue
        title = record.title or record.citation_text or "Verified source"
        suffix = f" ({record.citation_text})" if record.citation_text and record.citation_text not in title else ""
        note = f": {record.note}" if record.note else ""
        findings.append(f"{title}{suffix}{note}")
    return findings[:6]


def _likely_unverified_from_source_records(records: list[StructuredSourceRecord]) -> list[str]:
    leads: list[str] = []
    for record in records:
        if (
            record.verification_status in {"verified_primary_source", "verified_metadata"}
            and record.topical_relevance == "on_topic"
        ):
            continue
        title = record.title or record.citation_text or "Unverified source"
        note = f": {record.note}" if record.note else ""
        leads.append(f"{title}{note}")
    return leads[:6]


def _lead_reason_for_source_record(record: StructuredSourceRecord) -> str:
    if record.topical_relevance == "off_topic":
        return "Excluded from grounded evidence because it drifted off the anchored subject."
    if record.verification_status not in {"verified_primary_source", "verified_metadata"}:
        return "Retained as a lead because the source is not yet verified strongly enough to support a grounded answer."
    return "Retained as a lead because it is related context, but not strong enough to ground the answer on its own."


def _records_with_lead_reasons(records: list[StructuredSourceRecord]) -> list[StructuredSourceRecord]:
    enriched: list[StructuredSourceRecord] = []
    for record in records:
        reason = _lead_reason_for_source_record(record)
        why_not_verified = record.why_not_verified
        if why_not_verified is None:
            if record.verification_status not in {"verified_primary_source", "verified_metadata"}:
                why_not_verified = f"Verification status was {record.verification_status or 'unverified'}."
            elif record.topical_relevance and record.topical_relevance != "on_topic":
                why_not_verified = f"Topical relevance was {record.topical_relevance}."
        enriched.append(
            record.model_copy(
                update={
                    "lead_reason": record.lead_reason or reason,
                    "why_not_verified": why_not_verified,
                }
            )
        )
    return enriched


def _candidate_leads_from_source_records(records: list[StructuredSourceRecord]) -> list[StructuredSourceRecord]:
    leads: list[StructuredSourceRecord] = []
    for record in records:
        if (
            record.verification_status in {"verified_primary_source", "verified_metadata"}
            and record.topical_relevance == "on_topic"
        ):
            continue
        leads.append(record)
    return _records_with_lead_reasons(_dedupe_structured_sources(leads)[:6])


def _evidence_from_source_records(records: list[StructuredSourceRecord]) -> list[StructuredSourceRecord]:
    evidence: list[StructuredSourceRecord] = []
    for record in records:
        if record.verification_status not in {"verified_primary_source", "verified_metadata"}:
            continue
        if record.topical_relevance != "on_topic":
            continue
        evidence.append(record)
    return _dedupe_structured_sources(evidence)[:6]


def _answerability_from_source_records(
    *,
    result_status: str,
    evidence: list[StructuredSourceRecord],
    leads: list[StructuredSourceRecord],
    evidence_gaps: list[str],
) -> Literal["grounded", "limited", "insufficient"]:
    if result_status == "succeeded" and evidence:
        return "grounded"
    if evidence or leads or evidence_gaps:
        return "limited"
    return "insufficient"


def _routing_summary_from_strategy(
    *,
    strategy_metadata: SearchStrategyMetadata,
    coverage_summary: CoverageSummary | None,
    result_status: str,
    evidence_gaps: list[str],
) -> dict[str, Any]:
    providers_attempted = list((coverage_summary.providers_attempted if coverage_summary else []) or [])
    providers_succeeded = list((coverage_summary.providers_succeeded if coverage_summary else []) or [])
    providers_failed = list((coverage_summary.providers_failed if coverage_summary else []) or [])
    provider_plan = list(
        cast(list[str] | None, getattr(strategy_metadata, "provider_plan", None))
        or strategy_metadata.providers_used
        or []
    )
    return {
        "intent": strategy_metadata.intent,
        "decisionConfidence": strategy_metadata.routing_confidence or strategy_metadata.intent_confidence,
        "anchorType": strategy_metadata.anchor_type,
        "anchorValue": strategy_metadata.anchored_subject,
        "providerPlan": provider_plan,
        "providersAttempted": providers_attempted,
        "providersMatched": providers_succeeded,
        "providersFailed": providers_failed,
        "providersNotAttempted": [provider for provider in provider_plan if provider not in providers_attempted],
        "whyPartial": (evidence_gaps[0] if (result_status != "succeeded" and evidence_gaps) else None),
    }


def _dedupe_structured_sources(records: list[StructuredSourceRecord]) -> list[StructuredSourceRecord]:
    deduped: list[StructuredSourceRecord] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for record in records:
        key = (record.title, record.canonical_url, record.citation_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


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


def _is_agency_guidance_query(query: str) -> bool:
    lowered = query.lower()
    if not any(term in lowered for term in _AGENCY_GUIDANCE_TERMS):
        return False
    return any(term in lowered for term in _AGENCY_AUTHORITY_TERMS)


def _extract_scientific_name_candidate(query: str) -> str | None:
    match = re.search(r"\b([A-Z][a-z]{2,})\s+([a-z][a-z-]{2,})\b", query)
    if not match:
        return None
    return f"{match.group(1)} {match.group(2)}"


def _extract_common_name_candidate(query: str) -> str | None:
    cleaned = re.sub(r"\b\d+\s*CFR\s*(?:Part\s*)?\d+(?:\.[\dA-Za-z-]+)?\b", " ", query, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d{4}-\d{4,6}\b", " ", cleaned)
    cleaned = re.sub(
        r"\b(?:"
        + "|".join(re.escape(term) for term in sorted(_SPECIES_QUERY_NOISE_TERMS, key=len, reverse=True))
        + r")\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(cleaned.split(" "))
    cleaned = " ".join(part for part in cleaned.split() if part)
    if not cleaned:
        return None
    token_count = len(re.findall(r"[A-Za-z][A-Za-z'-]{1,}", cleaned))
    if token_count < 2:
        return None
    return cleaned


def _ecos_query_variants(
    query: str,
    *,
    planner: PlannerDecision | None = None,
) -> list[tuple[str, str, str]]:
    """Return ECOS query candidates as ``(query, anchor_type, origin)`` tuples.

    ``origin`` is ``"raw"`` for regex/raw-query-derived candidates and
    ``"planner"`` for candidates supplied by the planner bundle's
    ``entityCard`` / ``subjectCard``. Raw candidates are emitted first so
    the ECOS loop tries query-grounded variants before falling back to
    planner-supplied names — this removes the hallucination-first risk
    where a plausible-but-wrong LLM species name returns real-but-wrong
    ECOS data and contaminates downstream ranking. The planner variants
    still run as a fallback so genuine LLM-emitted names (which can
    recover from genus-only / subspecies prose the regex misses) keep
    their recall.
    """

    variants: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def _add(candidate: str | None, anchor_type: str, origin: str) -> None:
        if not candidate:
            return
        normalized = " ".join(candidate.split())
        if not normalized:
            return
        marker = normalized.lower()
        if marker in seen:
            return
        seen.add(marker)
        variants.append((normalized, anchor_type, origin))

    opaque = _is_opaque_query(query)

    # Raw / regex-derived scientific/common-name variants stay first: even for
    # opaque queries (DOIs / URLs) the extractor almost never fires on them,
    # but when it does the name is query-grounded and beats any planner
    # fallback.
    _add(_extract_scientific_name_candidate(query), "species_scientific_name", "raw")
    _add(_extract_common_name_candidate(query), "species_common_name", "raw")

    # For opaque queries the raw full-query variant is an identifier — it can
    # incidentally match ECOS (e.g. a DOI fragment overlapping a species code)
    # and lock the loop onto real-but-wrong data before the planner names
    # ever run. Defer the raw-full-query probe until after the planner
    # fallbacks when the query is opaque. Non-opaque queries keep the
    # historical ordering so query-grounded subject terms still run first.
    if not opaque:
        _add(query, "regulatory_subject_terms", "raw")

    # Planner-supplied names run as a fallback for genus-only / subspecies
    # prose the regex cannot recover. Downstream callers stamp hits from
    # these variants with a lower-confidence ``ecosProvenance`` marker.
    if planner is not None:
        entity_card = planner.entity_card or {}
        if isinstance(entity_card, dict):
            _add(
                str(entity_card.get("scientificName") or "") or None,
                "species_scientific_name",
                "planner",
            )
            _add(
                str(entity_card.get("commonName") or "") or None,
                "species_common_name",
                "planner",
            )
        if planner.subject_card is not None:
            _add(planner.subject_card.scientific_name, "species_scientific_name", "planner")
            _add(planner.subject_card.common_name, "species_common_name", "planner")

    if opaque:
        _add(query, "regulatory_subject_terms", "raw")

    return variants


_ECOS_PROVENANCE_RANK: dict[str, int] = {"raw": 0, "planner": 1}


def _rank_ecos_variant_hits(
    variant_hits: list[tuple[int, str, str, dict[str, Any]]],
) -> tuple[int, str, str, dict[str, Any]] | None:
    """Finding 4 (5th rubber-duck pass): pick the best ECOS variant result.

    Each entry is ``(variant_idx, anchor_type, origin, search_payload)``. The
    earlier scoring (``hits * factor`` with a 0.9 planner factor) let a
    planner-only variant with two incidental hits outrank a corroborated raw
    variant with one solid hit, defeating the provenance-first intent.

    The ranking key is strictly tiered:

    1. Non-empty variants beat empty ones (a corroborated variant with zero
       hits cannot stand in for a variant with real hits).
    2. Provenance rank (raw/query-corroborated beats planner-supplied).
    3. Hit count (higher wins within the same tier).
    4. Original ``variant_idx`` to preserve the intentional ordering from
       ``_ecos_query_variants`` as the final tie-breaker.

    That guarantees any corroborated variant with ``>=1`` hit beats any
    planner-only variant regardless of hit count, while still using hit count
    to disambiguate variants sharing a provenance tier. Returns ``None`` when
    the input is empty.
    """

    if not variant_hits:
        return None
    scored: list[tuple[int, int, int, int, int, str, str, dict[str, Any]]] = []
    for variant_idx, anchor_type, origin, search_payload in variant_hits:
        hit_count = len(list(search_payload.get("data") or []))
        has_hits_rank = 0 if hit_count > 0 else 1
        provenance_rank = _ECOS_PROVENANCE_RANK.get(origin, 1)
        scored.append(
            (
                has_hits_rank,
                provenance_rank,
                -hit_count,
                variant_idx,
                hit_count,
                anchor_type,
                origin,
                search_payload,
            )
        )
    scored.sort()
    *_, variant_idx, _hit_count, anchor_type, origin, search_payload = scored[0]
    return (variant_idx, anchor_type, origin, search_payload)


_OPAQUE_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_OPAQUE_ARXIV_RE = re.compile(r"\barxiv[:\s]*\d{4}\.\d{4,5}\b", re.IGNORECASE)


def _is_opaque_query(query: str) -> bool:
    """Return True when ``query`` looks like an identifier rather than prose.

    Opaque queries (DOIs, arXiv ids, bare URLs, identifier-like tokens with
    almost no letters) are meaningless as ECOS ``regulatory_subject_terms``
    probes and tend to match ECOS entries incidentally. The caller uses this
    to defer the raw full-query variant until after planner-supplied names.
    """

    text = (query or "").strip()
    if not text:
        return False
    if _OPAQUE_DOI_RE.search(text) or _OPAQUE_ARXIV_RE.search(text):
        return True
    lowered = text.lower()
    if lowered.startswith(("http://", "https://", "www.")):
        return True
    # Dense identifier-like tokens: mostly non-alpha, or a single long token
    # with punctuation characteristic of identifiers.
    alpha = sum(1 for ch in text if ch.isalpha())
    total = len(text)
    if total >= 8 and alpha / total < 0.4:
        return True
    if " " not in text and any(ch in text for ch in "/:._") and alpha / max(total, 1) < 0.6:
        return True
    return False


def _query_requests_regulatory_history(query: str) -> bool:
    lowered = query.lower()
    return any(
        marker in lowered
        for marker in (
            "federal register",
            "history",
            "listing history",
            "rulemaking",
            "timeline",
            "proposed rule",
            "final rule",
            "chronology",
        )
    )


def _parse_cfr_request(query: str) -> dict[str, Any] | None:
    match = re.search(
        r"\b(?P<title>\d+)\s*CFR\s*(?:Part\s*)?(?P<part>\d+)(?:\.(?P<section>[\dA-Za-z-]+))?",
        query,
        re.IGNORECASE,
    )
    if not match:
        return None
    return {
        "title_number": int(match.group("title")),
        "part_number": int(match.group("part")),
        "section_number": match.group("section"),
    }


def _is_current_cfr_text_request(query: str) -> bool:
    lowered = query.lower()
    if _parse_cfr_request(query) is None:
        return False
    markers = (
        "current cfr",
        "current text",
        "codified text",
        "cfr section",
        "what does",
        "what does the",
        "text of",
        "say about",
        "under ",
    )
    return any(marker in lowered for marker in markers) or bool(re.search(r"\b\d+\s*cfr\s+\d+(?:\.\S+)?\b", lowered))


def _derive_regulatory_query_flags(
    *,
    query: str,
    planner: PlannerDecision | None,
) -> tuple[bool, bool, bool]:
    """Map ``planner.regulatory_intent`` into the three routing booleans.

    LLM-first: when the planner bundle actually ran a real LLM (signalled by
    ``planner.planner_source == "llm"``) and emitted a definitive
    ``regulatoryIntent`` label, trust it authoritatively so the LLM signal
    wins over query keywords (e.g. "listing history of the Pallid Sturgeon"
    tagged ``species_dossier`` must NOT also activate the rulemaking-history
    route). Provenance is keyed off ``planner_source`` rather than
    ``subject_card.source`` because an LLM planner can legitimately emit
    ``regulatoryIntent`` without also supplying subject-card grounding
    fields; in that case ``classify_query`` stamps the subject card as
    ``deterministic_fallback``, but the LLM's regulatory label is still
    authoritative.

    Falls back to the deterministic keyword/regex helpers when:

    * the bundle is deterministic (``planner_source`` is ``"deterministic"``
      or ``"deterministic_fallback"``) — in that case ``regulatoryIntent``
      itself came from deterministic heuristics and is no more reliable than
      the keyword helpers, so we prefer the keyword helpers to avoid losing
      secondary routes on queries like "Regulatory history ... under 50 CFR ...";
    * the LLM emitted ``unspecified`` / ``hybrid_regulatory_plus_literature``
      / ``None`` — mixed or uncommitted intent, so every keyword-matched
      sub-route may still be relevant.

    Returns ``(current_text_requested, history_requested, agency_guidance_mode)``.
    """

    intent = planner.regulatory_intent if planner is not None else None
    llm_authoritative = (
        planner is not None
        and planner.planner_source == "llm"
        and planner.regulatory_intent_source == "llm"
        and intent is not None
    )
    if llm_authoritative:
        if intent == "current_cfr_text":
            return (True, False, False)
        if intent == "rulemaking_history":
            return (False, True, False)
        if intent == "guidance_lookup":
            return (False, False, True)
        if intent == "species_dossier":
            return (False, False, False)
    return (
        _is_current_cfr_text_request(query),
        _query_requests_regulatory_history(query),
        _is_agency_guidance_query(query),
    )


def _year_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else text[:4]


def _citation_record_from_paper(paper: Paper) -> CitationRecord | None:
    doi, _ = resolve_doi_from_paper_payload(paper)
    authors = [str(author.name).strip() for author in paper.authors if getattr(author, "name", None)]
    journal_or_publisher = (
        paper.enrichments.crossref.publisher if paper.enrichments and paper.enrichments.crossref else None
    ) or paper.venue
    url = paper.canonical_url or paper.retrieved_url or paper.url or paper.pdf_url
    if not any([authors, paper.title, paper.year, journal_or_publisher, doi, url]):
        return None
    return CitationRecord(
        authors=authors,
        year=_year_text(paper.publication_date or paper.year),
        title=paper.title,
        journalOrPublisher=journal_or_publisher,
        doi=doi,
        url=url,
        sourceType=paper.source_type,
        confidence=cast(Any, paper.confidence),
    )


def _citation_record_from_regulatory_document(
    document: dict[str, Any],
    *,
    provider: str,
    citation_text: str | None,
    canonical_url: str | None,
) -> CitationRecord:
    return CitationRecord(
        authors=[],
        year=_year_text(
            document.get("documentDate") or document.get("publicationDate") or document.get("effectiveDate")
        ),
        title=str(document.get("title") or citation_text or "Regulatory source"),
        journalOrPublisher=(
            "GovInfo" if provider == "govinfo" else ("Federal Register" if provider == "federal_register" else "ECOS")
        ),
        doi=None,
        url=canonical_url,
        sourceType="primary_regulatory",
        confidence=("high" if provider == "govinfo" else "medium"),
    )


def _coverage_summary_line(
    *,
    attempted: list[str],
    failed: list[str],
    zero_results: list[str],
    likely_completeness: str,
) -> str:
    return (
        f"{len(attempted)} provider(s) searched, {len(failed)} failed, "
        f"{len(zero_results)} returned zero results, likely completeness: {likely_completeness}."
    )


def _classify_topical_relevance(
    *,
    query_similarity: float,
    title_facet_coverage: float,
    title_anchor_coverage: float,
    query_facet_coverage: float,
    query_anchor_coverage: float,
) -> Literal["on_topic", "weak_match", "off_topic"]:
    has_title_signal = (title_facet_coverage > 0.0) or (title_anchor_coverage > 0.0)
    has_title_or_body_signal = has_title_signal or (query_facet_coverage > 0.0) or (query_anchor_coverage > 0.0)
    has_facet_signal = (title_facet_coverage > 0.0) or (query_facet_coverage > 0.0)
    # Require a multi-token phrase match (facet) for the standard threshold, or a
    # strict majority of query terms when no phrase match exists.  A single-token
    # title hit with low similarity is a weak signal, not grounded evidence.
    if has_title_signal and ((has_facet_signal and query_similarity >= 0.25) or query_similarity > 0.5):
        return "on_topic"
    if query_similarity < 0.12 or not has_title_or_body_signal:
        return "off_topic"
    return "weak_match"


def _classify_topical_relevance_for_paper(
    *,
    query: str,
    paper: dict[str, Any] | Paper,
    query_similarity: float,
    score_breakdown: ScoreBreakdown | None = None,
    llm_classification: Literal["on_topic", "weak_match", "off_topic"] | None = None,
) -> Literal["on_topic", "weak_match", "off_topic"]:
    return _classify_topical_relevance_with_provenance(
        query=query,
        paper=paper,
        query_similarity=query_similarity,
        score_breakdown=score_breakdown,
        llm_classification=llm_classification,
    ).effective


@dataclass(frozen=True)
class TopicalRelevanceClassification:
    """Provenance-aware result of the topical-relevance gate.

    ``effective`` is the verdict callers should act on; it matches what the
    legacy ``_classify_topical_relevance_for_paper`` returned. ``deterministic``
    is the raw heuristic verdict, ``llm`` is the classifier verdict when
    available, and ``source`` records which signal produced ``effective``.

    ``llm_override_ignored`` is True exactly when the deterministic fast-path
    produced a clear on_topic/off_topic verdict and an LLM classification was
    available but disagreed. The effective verdict is still the deterministic
    one in that case — the flag is purely for observability so callers can log
    the override or surface a counter.
    """

    effective: Literal["on_topic", "weak_match", "off_topic"]
    deterministic: Literal["on_topic", "weak_match", "off_topic"]
    llm: Literal["on_topic", "weak_match", "off_topic"] | None
    source: Literal["deterministic", "llm", "llm_tiebreaker"]
    llm_override_ignored: bool


def _classify_topical_relevance_with_provenance(
    *,
    query: str,
    paper: dict[str, Any] | Paper,
    query_similarity: float,
    score_breakdown: ScoreBreakdown | None = None,
    llm_classification: Literal["on_topic", "weak_match", "off_topic"] | None = None,
) -> TopicalRelevanceClassification:
    title = str((paper.title if isinstance(paper, Paper) else paper.get("title")) or "")
    body_text = _paper_text(paper.model_dump(by_alias=True) if isinstance(paper, Paper) else paper)
    title_tokens = _graph_topic_tokens(title)
    body_tokens = _graph_topic_tokens(body_text)
    anchors = [term for term in query_terms(query) if term not in _GRAPH_GENERIC_TERMS]
    facets = query_facets(query)

    title_anchor_hits = sum(term in title_tokens for term in anchors)
    body_anchor_hits = sum(term in body_tokens for term in anchors)
    title_anchor_coverage = (title_anchor_hits / len(anchors)) if anchors else 0.0
    query_anchor_coverage = (body_anchor_hits / len(anchors)) if anchors else 0.0

    def _facet_coverage(tokens: set[str]) -> float:
        if not facets:
            return 0.0
        matched = 0
        for facet in facets:
            facet_tokens = [token for token in re.findall(r"[a-z0-9]{3,}", facet.lower()) if token]
            if not facet_tokens:
                continue
            required = len(facet_tokens) if len(facet_tokens) <= 2 else 2
            if sum(token in tokens for token in facet_tokens) >= required:
                matched += 1
        return matched / len(facets)

    title_facet_coverage = _facet_coverage(title_tokens)
    query_facet_coverage = _facet_coverage(body_tokens)
    if score_breakdown is not None:
        title_facet_coverage = max(title_facet_coverage, score_breakdown.title_facet_coverage)
        title_anchor_coverage = max(title_anchor_coverage, score_breakdown.title_anchor_coverage)
        query_facet_coverage = max(query_facet_coverage, score_breakdown.query_facet_coverage)
        query_anchor_coverage = max(query_anchor_coverage, score_breakdown.query_anchor_coverage)

    deterministic = _classify_topical_relevance(
        query_similarity=query_similarity,
        title_facet_coverage=title_facet_coverage,
        title_anchor_coverage=title_anchor_coverage,
        query_facet_coverage=query_facet_coverage,
        query_anchor_coverage=query_anchor_coverage,
    )
    has_title_signal = (title_facet_coverage > 0.0) or (title_anchor_coverage > 0.0)
    has_title_or_body_signal = has_title_signal or (query_facet_coverage > 0.0) or (query_anchor_coverage > 0.0)
    fast_path_on_topic = deterministic == "on_topic" and query_similarity > 0.5
    fast_path_off_topic = deterministic == "off_topic" and query_similarity < 0.12 and not has_title_or_body_signal
    strict_title_alignment_query = looks_like_exact_title(query) or looks_like_near_known_item_query(query)
    guard_llm_on_topic_promotion = (
        strict_title_alignment_query
        and deterministic == "weak_match"
        and llm_classification == "on_topic"
        and not has_title_signal
    )

    effective: Literal["on_topic", "weak_match", "off_topic"]
    source: Literal["deterministic", "llm", "llm_tiebreaker"]
    llm_override_ignored = False
    if fast_path_on_topic:
        effective = "on_topic"
        source = "deterministic"
        if llm_classification is not None and llm_classification != "on_topic":
            llm_override_ignored = True
    elif fast_path_off_topic:
        effective = "off_topic"
        source = "deterministic"
        if llm_classification is not None and llm_classification != "off_topic":
            llm_override_ignored = True
    elif guard_llm_on_topic_promotion:
        effective = deterministic
        source = "deterministic"
    elif llm_classification is not None:
        effective = llm_classification
        source = "llm_tiebreaker" if deterministic == "weak_match" else "llm"
    else:
        effective = deterministic
        source = "deterministic"

    return TopicalRelevanceClassification(
        effective=effective,
        deterministic=deterministic,
        llm=llm_classification,
        source=source,
        llm_override_ignored=llm_override_ignored,
    )


def _extract_subject_terms(*names: str | None) -> set[str]:
    tokens: set[str] = set()
    for name in names:
        if not name:
            continue
        for token in re.findall(r"[a-z0-9]{3,}", name.lower()):
            if token in _REGULATORY_SUBJECT_STOPWORDS:
                continue
            if len(token) <= 3:
                continue
            tokens.add(token)
    return tokens


def _format_cfr_citation(cfr_request: dict[str, Any] | None) -> str | None:
    if not cfr_request:
        return None
    title = cfr_request.get("title_number")
    part = cfr_request.get("part_number")
    section = cfr_request.get("section_number")
    if title is None or part is None:
        return None
    if section:
        return f"{title} CFR {part}.{section}"
    return f"{title} CFR {part}"


def _regulatory_retrieval_hypotheses(
    *,
    query: str,
    planner: PlannerDecision,
    subject: str | None,
    anchor_type: str | None,
    cfr_citation: str | None,
    current_text_requested: bool,
    history_requested: bool,
    agency_guidance_mode: bool,
) -> list[str]:
    hypotheses: list[str] = [str(item).strip() for item in planner.retrieval_hypotheses if str(item).strip()]
    anchor_subject = str(subject or planner.anchor_value or query).strip()
    if current_text_requested and cfr_citation:
        hypotheses.append(f"Current codified CFR text for {cfr_citation}.")
    if cfr_citation and not current_text_requested:
        hypotheses.append(f"40 CFR incorporation and referenced regulatory text for {cfr_citation}.")
    if agency_guidance_mode:
        hypotheses.append(f"Agency guidance documents directly addressing {anchor_subject}.")
        hypotheses.append(f"Federal Register notices or policy actions relevant to {anchor_subject}.")
    elif anchor_type in {"species_common_name", "species_scientific_name"}:
        hypotheses.append(f"ECOS species dossier and supporting recovery-plan materials for {anchor_subject}.")
        hypotheses.append(f"Federal Register listing or habitat actions for {anchor_subject}.")
    else:
        hypotheses.append(f"Federal Register final rule or notice history for {anchor_subject}.")
        hypotheses.append(f"Current CFR incorporation or agency primary-source text for {anchor_subject}.")
    if history_requested and not agency_guidance_mode:
        hypotheses.append(f"Rulemaking timeline milestones for {anchor_subject}.")
    for facet in query_facets(query):
        facet_text = str(facet).strip()
        if facet_text:
            hypotheses.append(facet_text)

    deduped: list[str] = []
    seen: set[str] = set()
    for hypothesis in hypotheses:
        normalized = str(hypothesis).strip()
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped[:5]


def _cfr_tokens(citation: str | None) -> set[str]:
    if not citation:
        return set()
    return {token for token in re.findall(r"[a-z0-9]{2,}", citation.lower()) if token not in _CFR_DOC_TYPE_GENERIC}


def _regulatory_document_matches_subject(
    document: dict[str, Any],
    *,
    subject_terms: set[str],
    priority_terms: set[str] | None = None,
    cfr_citation: str | None,
) -> bool:
    title = str(document.get("title") or "")
    summary = str(document.get("abstract") or document.get("excerpt") or document.get("summary") or "")
    cfr_refs_raw = document.get("cfrReferences")
    cfr_refs = cfr_refs_raw if isinstance(cfr_refs_raw, list) else []
    cfr_ref_text = " ".join(str(ref) for ref in cfr_refs)
    document_text = " ".join(
        part for part in [title, summary, str(document.get("citation") or ""), cfr_ref_text] if part
    )
    document_tokens = _graph_topic_tokens(document_text)
    title_tokens = _graph_topic_tokens(title)
    priority_overlap = len(document_tokens & set(priority_terms or set()))

    subject_match_required = bool(subject_terms)
    subject_title_overlap = len(subject_terms & title_tokens)
    subject_body_overlap = len(subject_terms & document_tokens)
    subject_match = subject_title_overlap > 0 or subject_body_overlap >= 2 or priority_overlap >= 2
    if priority_terms:
        subject_match = subject_match and priority_overlap > 0

    cfr_match = False
    cfr_expected_tokens = _cfr_tokens(cfr_citation)
    if cfr_expected_tokens:
        for ref in cfr_refs:
            ref_tokens = _cfr_tokens(str(ref))
            if cfr_expected_tokens.issubset(ref_tokens):
                cfr_match = True
                break
        if not cfr_match:
            cfr_match = cfr_expected_tokens.issubset(_cfr_tokens(str(document.get("citation") or "")))

    if subject_match_required and cfr_expected_tokens:
        return subject_match and cfr_match
    if subject_match_required:
        return subject_match
    if cfr_expected_tokens:
        return cfr_match
    return True


def _agency_guidance_subject_terms(query: str) -> set[str]:
    normalized = re.sub(r"[-_/]+", " ", query.lower())
    return {
        term
        for term in re.findall(r"[a-z0-9]{4,}", normalized)
        if term not in _REGULATORY_SUBJECT_STOPWORDS
        and term not in _AGENCY_GUIDANCE_QUERY_NOISE_TERMS
        and term not in _GRAPH_GENERIC_TERMS
        and len(term) > 3
    }


def _regulatory_query_subject_terms(query: str) -> set[str]:
    normalized = re.sub(r"[-_/]+", " ", query.lower())
    return {
        term
        for term in re.findall(r"[a-z0-9]{4,}", normalized)
        if term not in _REGULATORY_SUBJECT_STOPWORDS
        and term not in _REGULATORY_QUERY_NOISE_TERMS
        and term not in _GRAPH_GENERIC_TERMS
    }


def _regulatory_query_priority_terms(query: str) -> set[str]:
    authority_terms = {term.lower() for term in _AGENCY_AUTHORITY_TERMS if " " not in term}
    generic_regulatory_acronyms = {
        "cfr",
        "esa",
        "fr",
        "u.s",
        "us",
    }
    return {
        token.lower()
        for token in re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", query)
        if token.lower() not in authority_terms and token.lower() not in generic_regulatory_acronyms
    }


def _agency_guidance_priority_terms(query: str) -> set[str]:
    terms = _agency_guidance_subject_terms(query)
    for facet in query_facets(query):
        for token in re.findall(r"[a-z0-9]{4,}", facet.lower()):
            if token in _GRAPH_GENERIC_TERMS or token in _AGENCY_GUIDANCE_QUERY_NOISE_TERMS:
                continue
            if token in _REGULATORY_SUBJECT_STOPWORDS:
                continue
            terms.add(token)
    return terms


def _agency_guidance_facet_terms(query: str) -> list[set[str]]:
    facet_terms: list[set[str]] = []
    for facet in query_facets(query):
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]{4,}", facet.lower())
            if token not in _GRAPH_GENERIC_TERMS
            and token not in _AGENCY_GUIDANCE_QUERY_NOISE_TERMS
            and token not in _REGULATORY_SUBJECT_STOPWORDS
        }
        if len(tokens) >= 2:
            facet_terms.append(tokens)
    return facet_terms


def _guidance_query_prefers_recency(query: str) -> bool:
    lowered = query.lower()
    return any(marker in lowered for marker in {"current", "latest", "new", "newest", "recent"})


def _is_species_regulatory_query(query: str) -> bool:
    lowered = query.lower()
    regulatory_markers = {"esa", "final rule", "listing history", "listing status", "regulatory history"}
    species_markers = {
        "bat",
        "bird",
        "condor",
        "critical habitat",
        "endangered",
        "habitat",
        "listing",
        "recovery",
        "species",
        "threatened",
        "wildlife",
    }
    return any(marker in lowered for marker in regulatory_markers) and any(
        marker in lowered for marker in species_markers
    )


def _rank_regulatory_documents(
    documents: list[dict[str, Any]],
    *,
    subject_terms: set[str],
    priority_terms: set[str],
    facet_terms: list[set[str]],
    prefer_guidance: bool,
    prefer_recent: bool,
    cultural_resource_boost: bool = False,
    requested_document_family: str | None = None,
) -> list[dict[str, Any]]:
    from ..subject_grounding import detect_document_family_match

    def _score(document: dict[str, Any]) -> tuple[int, str]:
        title = str(document.get("title") or "")
        summary = str(document.get("abstract") or document.get("excerpt") or document.get("summary") or "")
        tokens = _graph_topic_tokens(" ".join(part for part in [title, summary] if part))
        title_tokens = _graph_topic_tokens(title)
        overlap = len(subject_terms & tokens)
        title_overlap = len(subject_terms & title_tokens)
        priority_overlap = len(tokens & priority_terms)
        facet_overlap = 0
        for facet in facet_terms:
            required = len(facet) if len(facet) <= 2 else 2
            if sum(token in tokens for token in facet) >= required:
                facet_overlap += 1
        document_type = str(document.get("documentType") or "").lower()
        publication_date = str(document.get("publicationDate") or "")
        publication_year_match = re.search(r"\b(19|20)\d{2}\b", publication_date)
        publication_year = int(publication_year_match.group(0)) if publication_year_match else 0
        guidance_form_bonus = 8 if (tokens & _AGENCY_GUIDANCE_DOCUMENT_TERMS) else 0
        discussion_form_penalty = 4 if (tokens & _AGENCY_GUIDANCE_DISCUSSION_TERMS) else 0
        guidance_bonus = (
            2 if prefer_guidance and any(token in tokens for token in {"guidance", "framework", "discussion"}) else 0
        )
        notice_bonus = 1 if document_type in {"notice", "rule"} else 0
        recency_bonus = max((publication_year - 2010) * 2, 0) if prefer_recent else 0
        cultural_overlap = len(tokens & _CULTURAL_RESOURCE_DOCUMENT_TERMS) if cultural_resource_boost else 0
        cultural_title_overlap = len(title_tokens & _CULTURAL_RESOURCE_DOCUMENT_TERMS) if cultural_resource_boost else 0
        cultural_bonus = (cultural_title_overlap * 6) + (cultural_overlap * 3)
        family_match, family_boost_fraction = detect_document_family_match(document, requested_document_family)
        family_bonus = int(round(family_boost_fraction * 24)) if family_match else 0
        if family_match:
            # Annotate the document so the source-record builder can surface
            # documentFamilyMatch / documentFamilyBoost to callers.
            document["_documentFamilyMatch"] = family_match
            document["_documentFamilyBoost"] = round(family_boost_fraction, 3)
        score = (
            (title_overlap * 5)
            + (overlap * 2)
            + (priority_overlap * 2)
            + (facet_overlap * 3)
            + guidance_form_bonus
            + guidance_bonus
            + notice_bonus
            + recency_bonus
            + cultural_bonus
            + family_bonus
            - discussion_form_penalty
        )
        return score, publication_date

    return sorted(documents, key=_score, reverse=True)


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


def _initial_retrieval_query_text(*, normalized_query: str, focus: str | None, intent: IntentLabel) -> str:
    if intent in {"known_item", "author", "citation", "regulatory"}:
        return normalized_query
    normalized_focus = normalize_query(str(focus or ""))
    if not normalized_focus:
        return normalized_query
    combined = normalize_query(f"{normalized_query} {normalized_focus}")
    return combined if combined.lower() != normalized_query.lower() else normalized_query


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


def _known_item_resolution_queries(query: str, parsed: Any) -> list[str]:
    queries: list[str] = []
    normalized_query = normalize_query(query)
    if normalized_query:
        queries.append(normalized_query)
    title_candidates = list(getattr(parsed, "title_candidates", []) or [])
    author_surnames = list(getattr(parsed, "author_surnames", []) or [])
    venue_hints = list(getattr(parsed, "venue_hints", []) or [])
    year = getattr(parsed, "year", None)

    if title_candidates:
        queries.extend(title_candidates[:3])
        compact_title_words = [
            token
            for token in re.findall(r"[A-Za-z0-9'-]+", title_candidates[0])
            if len(token) >= 3
            and token.lower() not in COMMON_QUERY_WORDS
            and token.lower() not in {"paper", "papers", "article", "articles", "study", "studies"}
        ]
        if len(compact_title_words) >= 2:
            queries.append(" ".join(compact_title_words[:8]))
        if author_surnames:
            title_words = re.findall(r"[A-Za-z0-9'-]+", title_candidates[0])[:8]
            if title_words:
                queries.append(" ".join([*author_surnames[:2], *title_words]))
        if venue_hints:
            queries.append(f"{title_candidates[0]} {venue_hints[0]}")
    if author_surnames and year is not None:
        queries.append(" ".join([*author_surnames[:2], str(year)]))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in queries:
        normalized_candidate = normalize_query(candidate)
        if not normalized_candidate:
            continue
        lowered = normalized_candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized_candidate)
    return deduped


def _normalization_metadata(raw_query: str, normalized_query: str) -> tuple[list[str], dict[str, Any]]:
    raw_text = str(raw_query or "")
    normalized_text = str(normalized_query or "")
    if raw_text == normalized_text:
        return [], {}
    return (
        ["The server normalized the incoming query before routing it."],
        {
            "query": {
                "from": raw_text,
                "to": normalized_text,
            }
        },
    )


def _anchor_strength_for_resolution(resolution_strategy: str) -> Literal["high", "medium", "low"]:
    if resolution_strategy == "citation_resolution":
        return "high"
    if resolution_strategy in {"semantic_title_match", "openalex_autocomplete"}:
        return "medium"
    return "low"


def _known_item_resolution_state_for_strategy(
    *,
    resolution_strategy: str,
    known_item: dict[str, Any],
    query: str,
) -> Literal["resolved_exact", "resolved_probable", "needs_disambiguation"]:
    """Derive a :data:`KnownItemResolutionState` for a resolved known-item payload.

    Uses the shared citation-repair metadata when available (``citation_resolution``
    round-trip) and falls back to deterministic title-similarity bands for the
    secondary resolution strategies.
    """
    title = str(known_item.get("title") or "")
    similarity = _known_item_title_similarity(query, title) if title else 0.0
    if resolution_strategy == "citation_resolution":
        metadata = build_match_metadata(
            query=query,
            paper=known_item,
            candidate_count=1,
            resolution_strategy=resolution_strategy,
        )
        state = metadata.get("knownItemResolutionState")
        if isinstance(state, str) and state in {
            "resolved_exact",
            "resolved_probable",
            "needs_disambiguation",
        }:
            return cast(Literal["resolved_exact", "resolved_probable", "needs_disambiguation"], state)
    if similarity >= 0.9:
        return "resolved_exact"
    if similarity >= 0.72:
        return "resolved_probable"
    return "needs_disambiguation"


def _known_item_recovery_warning(resolution_strategy: str) -> str:
    if resolution_strategy == "semantic_title_match":
        return "Known-item recovery used a semantic title match; verify the anchor before treating it as canonical."
    if resolution_strategy == "openalex_autocomplete":
        return "Known-item recovery used OpenAlex autocomplete; verify the anchor before treating it as canonical."
    if resolution_strategy == "openalex_search":
        return "Known-item recovery used OpenAlex search; verify the anchor before treating it as canonical."
    return "Known-item fallback used title-style recovery; verify the anchor before treating it as canonical."


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


def _contextualize_follow_up_question(
    *,
    question: str,
    record: Any | None,
    question_mode: str,
) -> str:
    normalized_question = str(question or "").strip()
    if question_mode not in {"comparison", "selection"}:
        return normalized_question
    session_intent = _graph_intent_text(record, [])
    if not session_intent:
        return normalized_question
    lowered_question = normalized_question.lower()
    lowered_intent = session_intent.lower()
    if lowered_intent and lowered_intent in lowered_question:
        return normalized_question
    if not normalized_question:
        return session_intent
    return f"{normalized_question} about {session_intent}"


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
