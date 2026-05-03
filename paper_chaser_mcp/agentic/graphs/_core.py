"""Additive smart-workflow orchestration for Paper Chaser MCP."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal, cast

from fastmcp import Context

from ...citation_repair import parse_citation, resolve_citation
from ...compat import build_agent_hints, build_resource_uris
from ...enrichment import (
    PaperEnrichmentService,
    attach_enrichments_to_paper_payload,
    hydrate_paper_for_enrichment,
)
from ...models import (
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
    classify_query as classify_query,  # noqa: PLC0414 - re-export for test monkeypatch visibility; see Phase 7c-4
)
from ..provider_base import (
    DeterministicProviderBundle,
    ModelProviderBundle,
)
from ..ranking import (
    merge_candidates,
    rerank_candidates,
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

from . import smart_graph  # noqa: E402 - used by _maybe_compile_graphs delegate
from .followup_graph import (  # noqa: E402,F401 - preserve legacy call-site names; see Phase 7b plan
    _build_grounded_comparison_answer,
    _comparison_requested,
    _comparison_takeaway,
    _contextualize_follow_up_question,
    _looks_like_title_venue_list,
    _paper_focus_phrase,
    _shared_focus_terms,
    _should_use_structured_comparison_answer,
)
from .hooks import (  # noqa: E402,F401 - preserve legacy call-site names; see Phase 7a plan
    _consume_background_task,
    _describe_retrieval_batch,
    _skip_context_notifications,
    _truncate_text,
)
from .inspect_graph import (  # noqa: E402,F401 - preserve legacy call-site names; see Phase 7b plan
    _cluster_papers,
    _compute_disagreements,
    _compute_gaps,
    _finalize_theme_label,
    _label_tokens,
    _normalized_theme_label,
    _suggest_next_searches,
    _theme_terms_from_papers,
    _top_terms_for_cluster,
)
from .regulatory_routing import (  # noqa: E402,F401 - preserve legacy call-site names; see Phase 7a plan
    _ECOS_PROVENANCE_RANK,
    _OPAQUE_ARXIV_RE,
    _OPAQUE_DOI_RE,
    _agency_guidance_facet_terms,
    _agency_guidance_priority_terms,
    _agency_guidance_subject_terms,
    _cfr_tokens,
    _derive_regulatory_query_flags,
    _ecos_query_variants,
    _extract_common_name_candidate,
    _extract_scientific_name_candidate,
    _extract_subject_terms,
    _format_cfr_citation,
    _guidance_query_prefers_recency,
    _is_agency_guidance_query,
    _is_current_cfr_text_request,
    _is_opaque_query,
    _is_species_regulatory_query,
    _parse_cfr_request,
    _query_requests_regulatory_history,
    _rank_ecos_variant_hits,
    _rank_regulatory_documents,
    _regulatory_document_matches_subject,
    _regulatory_query_priority_terms,
    _regulatory_query_subject_terms,
    _regulatory_retrieval_hypotheses,
)
from .research_graph import (  # noqa: E402,F401 - preserve legacy call-site names; see Phase 7b plan
    _filter_graph_frontier,
    _graph_frontier_scores,
    _graph_intent_text,
)
from .resolve_graph import (  # noqa: E402,F401 - preserve legacy call-site names; see Phase 7b plan
    _anchor_strength_for_resolution,
    _known_item_recovery_warning,
    _known_item_resolution_queries,
    _known_item_resolution_state_for_strategy,
    _known_item_title_similarity,
    _normalization_metadata,
)
from .shared_state import (  # noqa: E402,F401 - preserve legacy call-site names; see Phase 7a plan
    _AGENCY_AUTHORITY_TERMS,
    _AGENCY_GUIDANCE_DISCUSSION_TERMS,
    _AGENCY_GUIDANCE_DOCUMENT_TERMS,
    _AGENCY_GUIDANCE_QUERY_NOISE_TERMS,
    _AGENCY_GUIDANCE_TERMS,
    _CFR_DOC_TYPE_GENERIC,
    _COMPARISON_FOCUS_STOPWORDS,
    _COMPARISON_MARKERS,
    _CULTURAL_RESOURCE_DOCUMENT_TERMS,
    _GRAPH_GENERIC_TERMS,
    _REGULATORY_QUERY_NOISE_TERMS,
    _REGULATORY_SUBJECT_STOPWORDS,
    _SPECIES_QUERY_NOISE_TERMS,
    _THEME_LABEL_STOPWORDS,
    END,
    SMART_SEARCH_PROGRESS_TOTAL,
    START,
    InMemorySaver,
    StateGraph,
)
from .smart_graph import (  # noqa: E402,F401 - preserve legacy call-site names; see Phase 7c-2 plan
    _dedupe_variants,
    _initial_retrieval_query_text,
    _result_coverage_label,
    _smart_failure_summary,
)
from .smart_helpers import (  # noqa: E402,F401 - legacy re-export; canonical home is smart_helpers (Phase 7c-3)
    _best_next_internal_action,
    _has_inspectable_sources,
    _has_on_topic_sources,
    _paid_providers_used,
    _smart_coverage_summary,
    _smart_provider_fallback_warnings,
)
from .source_records import (  # noqa: E402,F401 - preserve legacy call-site names; see Phase 7a plan
    TopicalRelevanceClassification,
    _answerability_from_source_records,
    _candidate_leads_from_source_records,
    _citation_record_from_paper,
    _citation_record_from_regulatory_document,
    _classify_topical_relevance,
    _classify_topical_relevance_for_paper,
    _classify_topical_relevance_with_provenance,
    _coverage_summary_line,
    _dedupe_structured_sources,
    _evidence_from_source_records,
    _graph_topic_tokens,
    _lead_reason_for_source_record,
    _likely_unverified_from_source_records,
    _paper_text,
    _records_with_lead_reasons,
    _routing_summary_from_strategy,
    _source_record_from_paper,
    _source_record_from_regulatory_document,
    _structured_sources_with_enriched_leads,
    _verified_findings_from_source_records,
    _why_matched,
    _year_text,
)


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
        return _skip_context_notifications(ctx)

    @staticmethod
    def _consume_background_task(task: asyncio.Task[Any]) -> None:
        _consume_background_task(task)

    @staticmethod
    def _describe_retrieval_batch(batch: RetrievalBatch) -> str:
        return _describe_retrieval_batch(batch)

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
        _provider_plan_override: list[str] | None = None,
    ) -> dict[str, Any]:
        """Smart concept-level discovery with grounded expansion and fusion."""
        from .smart_graph import run_search_papers_smart

        return await run_search_papers_smart(
            self,
            query=query,
            limit=limit,
            search_session_id=search_session_id,
            mode=mode,
            year=year,
            venue=venue,
            focus=focus,
            latency_profile=latency_profile,
            provider_budget=provider_budget,
            include_enrichment=include_enrichment,
            ctx=ctx,
            _provider_plan_override=_provider_plan_override,
        )

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
        lead_records = _candidate_leads_from_source_records(source_records)
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
            candidateLeads=lead_records,
            evidenceGaps=gaps,
            structuredSources=_structured_sources_with_enriched_leads(source_records),
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
        known_item, resolution_strategy = await self._resolve_known_item(query, provider_plan=provider_plan)
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

    async def _resolve_known_item(
        self,
        query: str,
        *,
        provider_plan: list[str] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        allowed_providers = {str(provider).strip() for provider in (provider_plan or []) if str(provider).strip()}
        allow_semantic = self._enable_semantic_scholar and (
            not allowed_providers or "semantic_scholar" in allowed_providers
        )
        allow_openalex = self._enable_openalex and (not allowed_providers or "openalex" in allowed_providers)
        allow_core = self._enable_core and (not allowed_providers or "core" in allowed_providers)
        allow_arxiv = self._enable_arxiv and (not allowed_providers or "arxiv" in allowed_providers)
        allow_serpapi = self._enable_serpapi and (not allowed_providers or "serpapi" in allowed_providers)
        result = await resolve_citation(
            citation=query,
            max_candidates=5,
            client=self._client,
            enable_core=allow_core,
            enable_semantic_scholar=allow_semantic,
            enable_openalex=allow_openalex,
            enable_arxiv=allow_arxiv,
            enable_serpapi=allow_serpapi,
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

        if allow_semantic:
            for candidate_query in resolution_queries:
                try:
                    semantic_match = dump_jsonable(
                        await self._client.search_papers_match(query=candidate_query, fields=None)
                    )
                except Exception:
                    semantic_match = None
                if isinstance(semantic_match, dict) and semantic_match.get("paperId"):
                    return semantic_match, str(semantic_match.get("matchStrategy") or "semantic_title_match")

        if allow_openalex and self._openalex_client is not None:
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
        return smart_graph.maybe_compile_graphs(self)


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
