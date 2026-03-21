import logging

import pytest

from scholar_search_mcp import server
from scholar_search_mcp.agentic import (
    AgenticConfig,
    AgenticRuntime,
    WorkspaceRegistry,
    resolve_provider_bundle,
)
from scholar_search_mcp.agentic.models import PlannerDecision
from scholar_search_mcp.agentic.planner import (
    classify_query,
    grounded_expansion_candidates,
    looks_like_exact_title,
)
from scholar_search_mcp.agentic.ranking import merge_candidates, rerank_candidates
from scholar_search_mcp.agentic.retrieval import RetrievedCandidate, retrieve_variant
from scholar_search_mcp.enrichment import PaperEnrichmentService
from tests.helpers import (
    RecordingCrossrefClient,
    RecordingOpenAlexClient,
    RecordingSemanticClient,
    RecordingUnpaywallClient,
    _payload,
)


class RecordingContext:
    def __init__(self) -> None:
        self.progress_updates: list[dict[str, object]] = []
        self.info_messages: list[dict[str, object]] = []

    async def report_progress(
        self,
        *,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        self.progress_updates.append(
            {
                "progress": progress,
                "total": total,
                "message": message,
            }
        )

    async def info(
        self,
        message: str,
        logger_name: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        self.info_messages.append(
            {
                "message": message,
                "logger_name": logger_name,
                "extra": extra,
            }
        )


def _deterministic_runtime(
    *,
    semantic: RecordingSemanticClient,
    openalex: RecordingOpenAlexClient,
) -> tuple[WorkspaceRegistry, AgenticRuntime]:
    config = AgenticConfig(
        enabled=True,
        provider="deterministic",
        planner_model="deterministic",
        synthesis_model="deterministic",
        embedding_model="deterministic",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
    )
    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False)
    runtime = AgenticRuntime(
        config=config,
        provider_bundle=resolve_provider_bundle(config, openai_api_key=None),
        workspace_registry=registry,
        client=semantic,
        core_client=object(),
        openalex_client=openalex,
        arxiv_client=object(),
        serpapi_client=None,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=True,
        enable_arxiv=False,
        enable_serpapi=False,
    )
    return registry, runtime


def _deterministic_config() -> AgenticConfig:
    return AgenticConfig(
        enabled=True,
        provider="deterministic",
        planner_model="deterministic",
        synthesis_model="deterministic",
        embedding_model="deterministic",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
    )


@pytest.mark.asyncio
async def test_smart_search_and_follow_up_tools_work_with_deterministic_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    monkeypatch.setattr(server, "agentic_runtime", runtime)
    monkeypatch.setattr(server, "workspace_registry", registry)

    smart = _payload(
        await server.call_tool("search_papers_smart", {"query": "transformers"})
    )

    assert smart["searchSessionId"]
    assert smart["results"]
    assert smart["strategyMetadata"]["queryVariantsTried"]
    assert "semantic_scholar" in smart["strategyMetadata"]["providersUsed"]
    assert "resourceUris" in smart
    assert "agentHints" in smart

    ask = _payload(
        await server.call_tool(
            "ask_result_set",
            {
                "searchSessionId": smart["searchSessionId"],
                "question": "What does this result set say about transformers?",
            },
        )
    )

    assert ask["searchSessionId"] == smart["searchSessionId"]
    assert ask["answer"]
    assert ask["evidence"]
    assert ask["agentHints"]["nextToolCandidates"]

    landscape = _payload(
        await server.call_tool(
            "map_research_landscape",
            {
                "searchSessionId": smart["searchSessionId"],
                "maxThemes": 3,
            },
        )
    )

    assert landscape["themes"]
    assert landscape["searchSessionId"] == smart["searchSessionId"]
    assert landscape["suggestedNextSearches"]

    graph = _payload(
        await server.call_tool(
            "expand_research_graph",
            {
                "seedSearchSessionId": smart["searchSessionId"],
                "direction": "citations",
            },
        )
    )

    assert graph["nodes"]
    assert graph["edges"]
    assert graph["frontier"]


@pytest.mark.asyncio
async def test_search_papers_smart_include_enrichment_enriches_final_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)
    crossref = RecordingCrossrefClient()
    unpaywall = RecordingUnpaywallClient()
    runtime._enrichment_service = PaperEnrichmentService(
        crossref_client=crossref,
        unpaywall_client=unpaywall,
        enable_crossref=True,
        enable_unpaywall=True,
        provider_registry=server.provider_registry,
    )

    monkeypatch.setattr(server, "agentic_runtime", runtime)
    monkeypatch.setattr(server, "workspace_registry", registry)

    smart = _payload(
        await server.call_tool(
            "search_papers_smart",
            {"query": "transformers", "includeEnrichment": True},
        )
    )
    record = registry.get(smart["searchSessionId"])

    assert smart["results"]
    first_paper = smart["results"][0]["paper"]
    assert first_paper["enrichments"]["crossref"]["doi"] == "10.1234/crossref-query"
    assert first_paper["enrichments"]["unpaywall"]["isOa"] is True
    assert (
        record.papers[0]["enrichments"]["crossref"]["doi"]
        == "10.1234/crossref-query"
    )
    assert any(
        outcome["provider"] == "crossref"
        for outcome in smart["strategyMetadata"]["providerOutcomes"]
    )
    assert any(
        outcome["provider"] == "unpaywall"
        for outcome in smart["strategyMetadata"]["providerOutcomes"]
    )
    assert crossref.calls
    assert unpaywall.calls


@pytest.mark.asyncio
async def test_search_papers_smart_emits_progress_logs_and_provider_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    _, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)
    ctx = RecordingContext()

    caplog.set_level(logging.INFO, logger="scholar-search-mcp")

    payload = await runtime.search_papers_smart(
        query="transformers",
        focus="literature review",
        limit=5,
        latency_profile="fast",
        ctx=ctx,
    )

    assert payload["results"]
    progress_messages = [
        update["message"] for update in ctx.progress_updates if update["message"]
    ]
    assert "Planning smart search" in progress_messages
    assert "Running initial retrieval" in progress_messages
    assert "Expansion plan ready" in progress_messages
    assert any(
        str(message).startswith("Expansion 1/1 complete")
        for message in progress_messages
    )
    assert "Smart search complete" in progress_messages

    info_messages = [entry["message"] for entry in ctx.info_messages]
    assert any(
        "Prepared 1 expansion variant(s)" in str(message) for message in info_messages
    )
    assert any("searchSessionId=" in str(message) for message in info_messages)

    log_messages = [record.getMessage() for record in caplog.records]
    assert any("smart-search[" in message for message in log_messages)
    assert any("provider-call[" in message for message in log_messages)


@pytest.mark.asyncio
async def test_ask_result_set_normalizes_non_literal_confidence_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    monkeypatch.setattr(server, "agentic_runtime", runtime)
    monkeypatch.setattr(server, "workspace_registry", registry)

    smart = _payload(
        await server.call_tool("search_papers_smart", {"query": "transformers"})
    )

    original_answer_question = runtime._provider_bundle.answer_question

    def _patched_answer_question(
        *,
        question: str,
        evidence_papers: list[dict],
        answer_mode: str,
    ) -> dict:
        payload = original_answer_question(
            question=question,
            evidence_papers=evidence_papers,
            answer_mode=answer_mode,
        )
        payload["confidence"] = "0.79"
        return payload

    monkeypatch.setattr(
        runtime._provider_bundle,
        "answer_question",
        _patched_answer_question,
    )

    ask = _payload(
        await server.call_tool(
            "ask_result_set",
            {
                "searchSessionId": smart["searchSessionId"],
                "question": "What does this result set say about transformers?",
            },
        )
    )

    assert ask["confidence"] == "medium"


@pytest.mark.asyncio
async def test_expand_research_graph_returns_structured_error_for_non_portable_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    record = registry.save_result_set(
        source_tool="search_papers_smart",
        payload={
            "data": [
                {
                    "paperId": "W123",
                    "title": "OpenAlex-only paper",
                    "canonicalId": "W123",
                    "expansionIdStatus": "not_portable",
                }
            ]
        },
    )

    monkeypatch.setattr(server, "agentic_runtime", runtime)
    monkeypatch.setattr(server, "workspace_registry", registry)

    graph = _payload(
        await server.call_tool(
            "expand_research_graph",
            {
                "seedSearchSessionId": record.search_session_id,
                "direction": "citations",
            },
        )
    )

    assert graph["error"] == "NON_PORTABLE_SEED"
    assert "portable" in graph["message"]


def test_grounded_expansions_filter_stopwords_and_single_paper_noise() -> None:
    config = _deterministic_config()

    variants = grounded_expansion_candidates(
        original_query="tool-using agents for literature review",
        papers=[
            {
                "title": "Agentic systematic review workflow",
                "abstract": (
                    "This workflow supports literature review with agent orchestration."
                ),
            },
            {
                "title": "Retrieval-augmented literature review agents",
                "abstract": ("Retrieval improves grounded literature review agents."),
            },
            {
                "title": "Unrelated biomedical note",
                "abstract": "With inhibitors were measured in one assay only.",
            },
        ],
        config=config,
    )

    expanded = [candidate.variant.lower() for candidate in variants]

    assert all(not variant.endswith(" with") for variant in expanded)
    assert all(" were" not in variant for variant in expanded)
    assert all(" inhibitors" not in variant for variant in expanded)


def test_classify_query_respects_explicit_review_mode() -> None:
    class _Bundle:
        def plan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="discovery",
                candidateConcepts=["agents"],
                followUpMode="qa",
            )

    _, planner = classify_query(
        query="tool-using agents for literature review",
        mode="review",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    assert planner.intent == "review"
    assert planner.follow_up_mode == "claim_check"
    assert "literature review" in planner.candidate_concepts


def test_classify_query_treats_citation_like_input_as_known_item() -> None:
    class _Bundle:
        def plan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="discovery",
                candidateConcepts=["planetary boundaries"],
                followUpMode="qa",
            )

    _, planner = classify_query(
        query="Rockstrom et al planetary boundaries 2009 Nature 461 472",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    assert planner.intent == "known_item"


def test_looks_like_exact_title_identifies_title_cased_paper_queries() -> None:
    assert looks_like_exact_title("Attention Is All You Need")
    assert not looks_like_exact_title("tool-using agents for literature review")


def test_classify_query_routes_title_like_query_to_known_item() -> None:
    class _Bundle:
        def plan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="discovery",
                candidateConcepts=["transformers"],
                followUpMode="qa",
            )

    _, planner = classify_query(
        query="Attention Is All You Need",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    assert planner.intent == "known_item"


def test_rerank_candidates_prefers_multi_facet_consensus_match() -> None:
    provider_bundle = resolve_provider_bundle(
        _deterministic_config(),
        openai_api_key=None,
    )
    ranked = rerank_candidates(
        query="tool-using agents for literature review",
        merged_candidates=[
            {
                "paper": {
                    "paperId": "serp-1",
                    "title": "AI agents in clinical medicine: a systematic review",
                    "abstract": "A review of AI agents in clinical settings.",
                    "year": 2025,
                    "authors": [{"name": "Author One"}],
                },
                "providers": ["serpapi_google_scholar"],
                "variants": ["tool-using agents for literature review"],
                "variantSources": ["from_input"],
                "providerRanks": {"serpapi_google_scholar": 1},
                "retrievalCount": 1,
            },
            {
                "paper": {
                    "paperId": "sem-1",
                    "title": "Tool-using agents for systematic literature review",
                    "abstract": (
                        "Autonomous agents combine tool use and retrieval "
                        "for literature review workflows."
                    ),
                    "year": 2025,
                    "authors": [{"name": "Author Two"}],
                    "citationCount": 12,
                },
                "providers": ["semantic_scholar", "openalex"],
                "variants": ["tool-using agents for literature review"],
                "variantSources": ["from_input"],
                "providerRanks": {"semantic_scholar": 2, "openalex": 1},
                "retrievalCount": 2,
            },
        ],
        provider_bundle=provider_bundle,
        candidate_concepts=["tool agents", "literature review"],
    )

    assert ranked[0]["paper"]["paperId"] == "sem-1"
    assert ranked[0]["scoreBreakdown"]["providerConsensusBonus"] > 0
    assert ranked[0]["scoreBreakdown"]["queryFacetCoverage"] >= 0.5


def test_merge_candidates_links_title_fallback_to_doi_aliases() -> None:
    merged = merge_candidates(
        [
            RetrievedCandidate(
                paper={
                    "paperId": "W123",
                    "title": "Tool-using agents for systematic literature review",
                    "year": 2025,
                    "authors": [{"name": "Author Two"}],
                },
                provider="openalex",
                variant="tool-using agents for literature review",
                variant_source="from_input",
                provider_rank=1,
            ),
            RetrievedCandidate(
                paper={
                    "paperId": "10.1234/example-doi",
                    "canonicalId": "10.1234/example-doi",
                    "recommendedExpansionId": "10.1234/example-doi",
                    "title": "Tool-using agents for systematic literature review",
                    "year": 2025,
                    "authors": [{"name": "Author Two"}],
                },
                provider="semantic_scholar",
                variant="tool-using agents for literature review",
                variant_source="from_input",
                provider_rank=2,
            ),
        ]
    )

    assert len(merged) == 1
    assert merged[0]["providers"] == ["openalex", "semantic_scholar"]


def test_merge_candidates_merges_same_title_and_year_with_mismatched_authors() -> None:
    merged = merge_candidates(
        [
            RetrievedCandidate(
                paper={
                    "paperId": "oa-1",
                    "title": (
                        "LLM-Based Multi-Agent Systems for Software Engineering: "
                        "Literature Review, Vision, and the Road Ahead"
                    ),
                    "year": 2025,
                    "authors": [{"name": "First Provider Author"}],
                },
                provider="openalex",
                variant="llm multi-agent software engineering literature review",
                variant_source="from_input",
                provider_rank=1,
            ),
            RetrievedCandidate(
                paper={
                    "paperId": "sem-1",
                    "title": (
                        "LLM-Based Multi-Agent Systems for Software Engineering: "
                        "Literature Review, Vision, and the Road Ahead"
                    ),
                    "year": 2025,
                    "authors": [{"name": "Second Provider Author"}],
                },
                provider="semantic_scholar",
                variant="llm multi-agent software engineering literature review",
                variant_source="from_input",
                provider_rank=2,
            ),
        ]
    )

    assert len(merged) == 1
    assert merged[0]["providers"] == ["openalex", "semantic_scholar"]


def test_rerank_candidates_downweights_serpapi_snippet_echo_without_title_match() -> (
    None
):
    provider_bundle = resolve_provider_bundle(
        _deterministic_config(),
        openai_api_key=None,
    )
    ranked = rerank_candidates(
        query="tool-using agents for literature review",
        merged_candidates=[
            {
                "paper": {
                    "paperId": "serp-echo",
                    "title": "AI agents in clinical medicine: a systematic review",
                    "abstract": (
                        "Snippet text mentioning tool using agents for literature "
                        "review even though the title is medical."
                    ),
                    "year": 2025,
                    "authors": [{"name": "Author One"}],
                    "source": "serpapi_google_scholar",
                },
                "providers": ["serpapi_google_scholar"],
                "variants": ["tool-using agents for literature review"],
                "variantSources": ["from_input"],
                "providerRanks": {"serpapi_google_scholar": 1},
                "retrievalCount": 1,
            },
            {
                "paper": {
                    "paperId": "sem-good",
                    "title": "Tool-using agents for systematic literature review",
                    "abstract": (
                        "Autonomous agents combine tool use and retrieval "
                        "for literature review workflows."
                    ),
                    "year": 2025,
                    "authors": [{"name": "Author Two"}],
                    "source": "semantic_scholar",
                },
                "providers": ["semantic_scholar"],
                "variants": ["tool-using agents for literature review"],
                "variantSources": ["from_input"],
                "providerRanks": {"semantic_scholar": 2},
                "retrievalCount": 1,
            },
        ],
        provider_bundle=provider_bundle,
        candidate_concepts=["tool agents", "literature review"],
    )

    assert ranked[0]["paper"]["paperId"] == "sem-good"
    assert (
        ranked[0]["scoreBreakdown"]["titleFacetCoverage"]
        > (ranked[1]["scoreBreakdown"]["titleFacetCoverage"])
    )


def test_rerank_candidates_penalizes_review_papers_missing_title_anchor_terms() -> None:
    provider_bundle = resolve_provider_bundle(
        _deterministic_config(),
        openai_api_key=None,
    )
    ranked = rerank_candidates(
        query="tool-using agents for literature review",
        merged_candidates=[
            {
                "paper": {
                    "paperId": "sem-medical",
                    "title": (
                        "Cholestatic pruritus treatments: a systematic literature "
                        "review"
                    ),
                    "abstract": (
                        "Therapeutic agents were evaluated with multiple clinical "
                        "tools across treatment studies."
                    ),
                    "year": 2025,
                    "authors": [{"name": "Author One"}],
                    "source": "semantic_scholar",
                },
                "providers": ["semantic_scholar"],
                "variants": ["tool-using agents for literature review"],
                "variantSources": ["from_input"],
                "providerRanks": {"semantic_scholar": 1},
                "retrievalCount": 1,
            },
            {
                "paper": {
                    "paperId": "sem-tool",
                    "title": "Tool Use for Autonomous Agents",
                    "abstract": (
                        "A survey of tool-using autonomous agents and research "
                        "workflows."
                    ),
                    "year": 2024,
                    "authors": [{"name": "Author Two"}],
                    "source": "semantic_scholar",
                },
                "providers": ["semantic_scholar"],
                "variants": ["retrieval-augmented literature review agents"],
                "variantSources": ["speculative"],
                "providerRanks": {"semantic_scholar": 2},
                "retrievalCount": 1,
            },
        ],
        provider_bundle=provider_bundle,
        candidate_concepts=["tool-using agents", "literature review"],
    )

    assert ranked[0]["paper"]["paperId"] == "sem-tool"
    assert (
        ranked[0]["scoreBreakdown"]["titleAnchorCoverage"]
        > (ranked[1]["scoreBreakdown"]["titleAnchorCoverage"])
    )


@pytest.mark.asyncio
async def test_retrieve_variant_skips_serpapi_when_disabled_for_variant() -> None:
    class _EmptyClient:
        async def search(self, **kwargs: object) -> dict:
            return {"data": []}

    class _FailingSerpApiClient:
        async def search(self, **kwargs: object) -> dict:
            raise AssertionError("SerpApi should not be called for follow-on variants.")

    batch = await retrieve_variant(
        variant="retrieval augmented literature review agents",
        variant_source="speculative",
        intent="review",
        year=None,
        venue=None,
        enable_core=False,
        enable_semantic_scholar=False,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=True,
        core_client=_EmptyClient(),
        semantic_client=_EmptyClient(),
        openalex_client=_EmptyClient(),
        arxiv_client=_EmptyClient(),
        serpapi_client=_FailingSerpApiClient(),
        allow_serpapi=False,
    )

    assert batch.providers_used == []
    assert batch.candidates == []
