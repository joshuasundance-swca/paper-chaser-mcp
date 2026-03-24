import asyncio
import logging
from typing import Any, cast

import pytest

from scholar_search_mcp import server
from scholar_search_mcp.agentic import (
    AgenticConfig,
    AgenticRuntime,
    WorkspaceRegistry,
    resolve_provider_bundle,
)
from scholar_search_mcp.agentic.graphs import (
    _build_grounded_comparison_answer,
    _finalize_theme_label,
    _graph_frontier_scores,
)
from scholar_search_mcp.agentic.models import PlannerDecision
from scholar_search_mcp.agentic.planner import (
    classify_query,
    grounded_expansion_candidates,
    looks_like_exact_title,
)
from scholar_search_mcp.agentic.providers import OpenAIProviderBundle
from scholar_search_mcp.agentic.ranking import merge_candidates, rerank_candidates
from scholar_search_mcp.agentic.retrieval import (
    RetrievedCandidate,
    provider_limits,
    retrieve_variant,
)
from scholar_search_mcp.enrichment import PaperEnrichmentService
from scholar_search_mcp.provider_runtime import (
    ProviderDiagnosticsRegistry,
    ProviderPolicy,
    execute_provider_call,
)
from tests.helpers import (
    RecordingCrossrefClient,
    RecordingOpenAlexClient,
    RecordingSemanticClient,
    RecordingUnpaywallClient,
    _payload,
)


class RecordingContext:
    def __init__(self, *, transport: str | None = None) -> None:
        self.progress_updates: list[dict[str, object]] = []
        self.info_messages: list[dict[str, object]] = []
        self.transport = transport

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

    smart = _payload(await server.call_tool("search_papers_smart", {"query": "transformers"}))

    assert smart["searchSessionId"]
    assert smart["results"]
    assert smart["strategyMetadata"]["queryVariantsTried"]
    assert smart["strategyMetadata"]["stageTimingsMs"]
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

    async def semantic_search(**kwargs: object) -> dict:
        semantic.calls.append(("search_papers", dict(kwargs)))
        return {
            "total": 1,
            "offset": 0,
            "data": [
                {
                    "paperId": "semantic-1",
                    "title": "Transformers in Wildlife Acoustics",
                    "year": 2024,
                    "authors": [{"name": "Lead Author"}],
                    "venue": "Journal of Tests",
                }
            ],
        }

    async def empty_openalex_search(**kwargs: object) -> dict:
        openalex.calls.append(("search", dict(kwargs)))
        return {"total": 0, "offset": 0, "data": []}

    semantic.search_papers = semantic_search  # type: ignore[method-assign]
    openalex.search = empty_openalex_search  # type: ignore[method-assign]

    class MatchingCrossrefClient(RecordingCrossrefClient):
        async def search_work(self, query: str) -> dict:
            self.calls.append(("search_work", {"query": query}))
            return {
                "doi": "10.1234/crossref-query",
                "title": "Transformers in Wildlife Acoustics",
                "authors": [{"name": "Lead Author"}],
                "venue": "Journal of Tests",
                "publisher": "Crossref Publisher",
                "publicationType": "journal-article",
                "publicationDate": "2024-05-01",
                "year": 2024,
                "url": "https://doi.org/10.1234/crossref-query",
                "citationCount": 7,
            }

    crossref = MatchingCrossrefClient()
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
    assert record.papers[0]["enrichments"]["crossref"]["doi"] == "10.1234/crossref-query"
    assert any(outcome["provider"] == "crossref" for outcome in smart["strategyMetadata"]["providerOutcomes"])
    assert any(outcome["provider"] == "unpaywall" for outcome in smart["strategyMetadata"]["providerOutcomes"])
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
        ctx=cast(Any, ctx),
    )
    await asyncio.sleep(0)

    assert payload["results"]
    progress_messages = [update["message"] for update in ctx.progress_updates if update["message"]]
    assert "Planning smart search" in progress_messages
    assert "Running initial retrieval" in progress_messages
    assert "No grounded expansions to run" in progress_messages
    assert "Smart search complete" in progress_messages

    info_messages = [entry["message"] for entry in ctx.info_messages]
    assert any("Initial retrieval stayed close to the query" in str(message) for message in info_messages)
    assert any("searchSessionId=" in str(message) for message in info_messages)

    log_messages = [record.getMessage() for record in caplog.records]
    assert any("smart-search[" in message for message in log_messages)
    assert any("provider-call[" in message for message in log_messages)


@pytest.mark.asyncio
async def test_search_papers_smart_skips_context_notifications_on_stdio_transport() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    _, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)
    ctx = RecordingContext(transport="stdio")

    payload = await runtime.search_papers_smart(
        query="transformers",
        focus="literature review",
        limit=5,
        latency_profile="fast",
        ctx=cast(Any, ctx),
    )
    await asyncio.sleep(0)

    assert payload["results"]
    assert ctx.progress_updates == []
    assert ctx.info_messages == []


@pytest.mark.asyncio
async def test_smart_search_does_not_block_on_slow_context_notifications() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    _, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)
    blocker = asyncio.Event()

    class SlowContext:
        async def report_progress(self, **kwargs: object) -> None:
            del kwargs
            await blocker.wait()

        async def info(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            await blocker.wait()

    payload = await asyncio.wait_for(
        runtime.search_papers_smart(
            query="transformers",
            limit=3,
            latency_profile="fast",
            ctx=cast(Any, SlowContext()),
        ),
        timeout=2.0,
    )

    assert payload["results"]
    for task in list(runtime._background_tasks):
        task.cancel()
    await asyncio.gather(*runtime._background_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_ask_result_set_runs_synthesis_and_scoring_concurrently(
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
                    "paperId": "paper-1",
                    "title": "Retrieval-Augmented Agents",
                    "abstract": "Vector retrieval grounds agent answers.",
                }
            ]
        },
    )
    answer_started = asyncio.Event()
    scoring_started = asyncio.Event()
    overlap = asyncio.Event()
    release = asyncio.Event()

    async def _aanswer_question(**kwargs: object) -> dict[str, object]:
        del kwargs
        answer_started.set()
        if scoring_started.is_set():
            overlap.set()
        await release.wait()
        return {
            "answer": "Concurrent answer",
            "confidence": "high",
            "unsupportedAsks": [],
            "followUpQuestions": [],
        }

    async def _abatched_similarity(
        query: str,
        texts: list[str],
        **kwargs: object,
    ) -> list[float]:
        del query, texts, kwargs
        scoring_started.set()
        if answer_started.is_set():
            overlap.set()
        await release.wait()
        return [0.92]

    monkeypatch.setattr(runtime._provider_bundle, "aanswer_question", _aanswer_question)
    monkeypatch.setattr(
        runtime._provider_bundle,
        "abatched_similarity",
        _abatched_similarity,
    )

    task = asyncio.create_task(
        runtime.ask_result_set(
            search_session_id=record.search_session_id,
            question="What does this result set say about retrieval?",
            top_k=1,
            answer_mode="synthesis",
            latency_profile="deep",
        )
    )

    await asyncio.wait_for(overlap.wait(), timeout=1.0)
    release.set()
    ask = await task

    assert ask["answer"] == "Concurrent answer"
    assert ask["evidence"][0]["paper"]["paperId"] == "paper-1"


@pytest.mark.asyncio
async def test_ask_result_set_normalizes_non_literal_confidence_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    monkeypatch.setattr(server, "agentic_runtime", runtime)
    monkeypatch.setattr(server, "workspace_registry", registry)

    smart = _payload(await server.call_tool("search_papers_smart", {"query": "transformers"}))

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
async def test_ask_result_set_balanced_mode_skips_embedding_scoring() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    config = AgenticConfig(
        enabled=True,
        provider="openai",
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
    )
    provider_registry = ProviderDiagnosticsRegistry()
    bundle = OpenAIProviderBundle(
        config,
        api_key="sk-test",
        provider_registry=provider_registry,
    )

    async def _unexpected_aembed_texts(
        texts: list[str],
        **kwargs: object,
    ) -> list[tuple[float, ...] | None]:
        raise AssertionError(f"balanced ask_result_set should not call embeddings: {texts!r}, {kwargs!r}")

    bundle.aembed_texts = _unexpected_aembed_texts  # type: ignore[method-assign]

    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False)
    record = registry.save_result_set(
        source_tool="search_papers_smart",
        payload={
            "data": [
                {
                    "paperId": "paper-1",
                    "title": "Retrieval-Augmented Agents",
                    "abstract": "Vector retrieval grounds agent answers.",
                }
            ]
        },
    )
    runtime = AgenticRuntime(
        config=config,
        provider_bundle=bundle,
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
        provider_registry=provider_registry,
    )

    ask = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question="What does this result set say about retrieval?",
        top_k=1,
        answer_mode="synthesis",
        latency_profile="balanced",
    )

    assert ask["answer"]


@pytest.mark.asyncio
async def test_ask_result_set_comparison_uses_grounded_structure_when_model_answer_is_weak(
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
                    "paperId": "paper-1",
                    "title": "Retrieval-Augmented Agents",
                    "abstract": "Vector retrieval grounds agent answers with tool use.",
                    "venue": "Agent Systems",
                    "year": 2024,
                },
                {
                    "paperId": "paper-2",
                    "title": "Grounded Planning for Agents",
                    "abstract": "Planning improves grounded decision making and retrieval handoffs.",
                    "venue": "Planning Workshop",
                    "year": 2023,
                },
            ]
        },
    )

    async def _weak_list_answer(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "answer": (
                "Comparison grounded in the saved result set:\n"
                "- Retrieval-Augmented Agents: Agent Systems, 2024\n"
                "- Grounded Planning for Agents: Planning Workshop, 2023"
            ),
            "confidence": "medium",
            "unsupportedAsks": [],
            "followUpQuestions": [],
        }

    monkeypatch.setattr(runtime._provider_bundle, "aanswer_question", _weak_list_answer)
    monkeypatch.setattr(runtime._deterministic_bundle, "aanswer_question", _weak_list_answer)

    ask = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question="Compare these agent papers.",
        top_k=2,
        answer_mode="comparison",
        latency_profile="deep",
    )

    assert "Shared ground:" in ask["answer"]
    assert "Key differences:" in ask["answer"]
    assert "Takeaway:" in ask["answer"]
    assert "Comparison grounded in the saved result set:" not in ask["answer"]


@pytest.mark.asyncio
async def test_map_research_landscape_balanced_mode_skips_embedding_clustering() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    config = AgenticConfig(
        enabled=True,
        provider="openai",
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
    )
    provider_registry = ProviderDiagnosticsRegistry()
    bundle = OpenAIProviderBundle(
        config,
        api_key="sk-test",
        provider_registry=provider_registry,
    )

    async def _unexpected_aembed_texts(
        texts: list[str],
        **kwargs: object,
    ) -> list[tuple[float, ...] | None]:
        raise AssertionError(f"balanced landscape mapping should not call embeddings: {texts!r}, {kwargs!r}")

    bundle.aembed_texts = _unexpected_aembed_texts  # type: ignore[method-assign]

    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False)
    record = registry.save_result_set(
        source_tool="search_papers_smart",
        payload={
            "data": [
                {
                    "paperId": "paper-1",
                    "title": "Retrieval-Augmented Agents",
                    "abstract": "Vector retrieval grounds agent answers.",
                    "year": 2024,
                },
                {
                    "paperId": "paper-2",
                    "title": "Grounded Planning Agents",
                    "abstract": "Grounded planning improves answer quality.",
                    "year": 2023,
                },
            ]
        },
    )
    runtime = AgenticRuntime(
        config=config,
        provider_bundle=bundle,
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
        provider_registry=provider_registry,
    )

    landscape = await runtime.map_research_landscape(
        search_session_id=record.search_session_id,
        max_themes=2,
        latency_profile="balanced",
    )

    assert landscape["themes"]


@pytest.mark.asyncio
async def test_map_research_landscape_sanitizes_junk_theme_labels(
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
                    "paperId": "paper-1",
                    "title": "Noise exposure effects on birds",
                    "abstract": "Bird responses to anthropogenic noise exposure.",
                    "year": 2024,
                },
                {
                    "paperId": "paper-2",
                    "title": "Wildlife responses to anthropogenic noise",
                    "abstract": "Noise changes behavior across wildlife populations.",
                    "year": 2023,
                },
                {
                    "paperId": "paper-3",
                    "title": "Acoustic disturbance in birds and mammals",
                    "abstract": "Birds and mammals show disturbance under acoustic exposure.",
                    "year": 2022,
                },
            ]
        },
    )

    async def _junk_label(**kwargs: object) -> str:
        del kwargs
        return "Noise / And"

    monkeypatch.setattr(runtime._provider_bundle, "alabel_theme", _junk_label)

    landscape = await runtime.map_research_landscape(
        search_session_id=record.search_session_id,
        max_themes=1,
        latency_profile="deep",
    )

    assert landscape["themes"]
    theme_title = landscape["themes"][0]["title"]
    assert theme_title != "Noise / And"
    assert "And" not in theme_title
    assert any(token in theme_title.lower() for token in ("noise", "birds", "acoustic", "anthropogenic"))


@pytest.mark.asyncio
async def test_async_embeddings_degrade_without_sync_retry_when_openai_fails() -> None:
    config = AgenticConfig(
        enabled=True,
        provider="openai",
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
    )
    bundle = OpenAIProviderBundle(config, api_key="sk-test")

    class _EmbeddingsClient:
        async def create(self, **kwargs: object) -> object:
            del kwargs
            raise TimeoutError("simulated embedding timeout")

    class _AsyncClient:
        embeddings = _EmbeddingsClient()

    bundle._async_openai_client = _AsyncClient()
    bundle._openai_client = None
    bundle._embeddings = None

    def _unexpected_sync_retry(texts: list[str]) -> list[tuple[float, ...] | None]:
        raise AssertionError(f"sync retry should not run: {texts!r}")

    bundle.embed_texts = _unexpected_sync_retry  # type: ignore[method-assign]

    embeddings = await bundle.aembed_texts(
        ["alpha paper abstract", "beta paper abstract"],
        request_id="smart-timeout",
    )

    assert embeddings == [None, None]


@pytest.mark.asyncio
async def test_async_similarity_falls_back_to_lexical_scores_when_embeddings_fail() -> None:
    config = AgenticConfig(
        enabled=True,
        provider="openai",
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
    )
    bundle = OpenAIProviderBundle(config, api_key="sk-test")

    class _EmbeddingsClient:
        async def create(self, **kwargs: object) -> object:
            del kwargs
            raise TimeoutError("simulated embedding timeout")

    class _AsyncClient:
        embeddings = _EmbeddingsClient()

    bundle._async_openai_client = _AsyncClient()
    bundle._openai_client = None
    bundle._embeddings = None

    scores = await bundle.abatched_similarity(
        "transformer retrieval",
        ["transformer retrieval for grounded answers", "avian habitat management"],
        request_id="smart-timeout",
    )

    assert len(scores) == 2
    assert scores[0] > scores[1]


@pytest.mark.asyncio
async def test_search_papers_smart_records_embedding_timeout_provider_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    config = AgenticConfig(
        enabled=True,
        provider="openai",
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
        openai_timeout_seconds=3.0,
    )
    provider_registry = ProviderDiagnosticsRegistry()
    bundle = OpenAIProviderBundle(
        config,
        api_key="sk-test",
        provider_registry=provider_registry,
    )

    class _EmbeddingsClient:
        async def create(self, **kwargs: object) -> object:
            del kwargs
            raise TimeoutError("simulated stalled embeddings call")

    class _AsyncClient:
        embeddings = _EmbeddingsClient()

    bundle._async_openai_client = _AsyncClient()
    bundle._openai_client = None
    bundle._embeddings = None

    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False)
    runtime = AgenticRuntime(
        config=config,
        provider_bundle=bundle,
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
        provider_registry=provider_registry,
    )

    caplog.set_level(logging.INFO, logger="scholar-search-mcp")

    payload = await runtime.search_papers_smart(
        query="transformers",
        limit=5,
        latency_profile="deep",
    )

    assert payload["results"]
    openai_outcomes = [
        outcome
        for outcome in payload["strategyMetadata"]["providerOutcomes"]
        if outcome["provider"] == "openai" and outcome["endpoint"] == "embeddings.create"
    ]
    assert openai_outcomes
    assert openai_outcomes[-1]["statusBucket"] == "provider_error"
    assert "TimeoutError" in (openai_outcomes[-1]["error"] or "")

    log_messages = [record.getMessage() for record in caplog.records]
    assert any("embedding-batch[smart-" in message for message in log_messages)
    assert any("OpenAI embeddings.create exceeded total timeout" in message for message in log_messages)


@pytest.mark.asyncio
async def test_search_papers_smart_balanced_mode_skips_embedding_rerank() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    config = AgenticConfig(
        enabled=True,
        provider="openai",
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
    )
    provider_registry = ProviderDiagnosticsRegistry()
    bundle = OpenAIProviderBundle(
        config,
        api_key="sk-test",
        provider_registry=provider_registry,
    )

    class _EmbeddingsClient:
        async def create(self, **kwargs: object) -> object:
            raise AssertionError(f"balanced rerank should not call embeddings: {kwargs!r}")

    class _ResponsesClient:
        async def parse(self, **kwargs: object) -> object:
            del kwargs
            return {
                "intent": "discovery",
                "constraints": {},
                "seedIdentifiers": [],
                "candidateConcepts": ["transformers"],
                "providerPlan": ["semantic_scholar", "openalex"],
                "followUpMode": "qa",
            }

    class _AsyncClient:
        embeddings = _EmbeddingsClient()
        responses = _ResponsesClient()

    bundle._async_openai_client = _AsyncClient()
    bundle._openai_client = None
    bundle._embeddings = None

    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False)
    runtime = AgenticRuntime(
        config=config,
        provider_bundle=bundle,
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
        provider_registry=provider_registry,
    )

    payload = await runtime.search_papers_smart(
        query="transformers",
        limit=5,
        latency_profile="balanced",
    )

    assert payload["results"]
    assert not [
        outcome
        for outcome in payload["strategyMetadata"]["providerOutcomes"]
        if outcome["provider"] == "openai" and outcome["endpoint"] == "embeddings.create"
    ]


@pytest.mark.asyncio
async def test_search_papers_smart_deep_mode_skips_embeddings_when_disabled() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    config = AgenticConfig(
        enabled=True,
        provider="openai",
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        disable_embeddings=True,
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
    )
    provider_registry = ProviderDiagnosticsRegistry()
    bundle = OpenAIProviderBundle(
        config,
        api_key="sk-test",
        provider_registry=provider_registry,
    )

    class _EmbeddingsClient:
        async def create(self, **kwargs: object) -> object:
            raise AssertionError(f"global embedding disable should prevent embedding calls: {kwargs!r}")

    class _ResponsesClient:
        async def parse(self, **kwargs: object) -> object:
            del kwargs
            return {
                "intent": "discovery",
                "constraints": {},
                "seedIdentifiers": [],
                "candidateConcepts": ["transformers"],
                "providerPlan": ["semantic_scholar", "openalex"],
                "followUpMode": "qa",
            }

    class _AsyncClient:
        embeddings = _EmbeddingsClient()
        responses = _ResponsesClient()

    bundle._async_openai_client = _AsyncClient()
    bundle._openai_client = None
    bundle._embeddings = None

    registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        index_backend="faiss",
        similarity_fn=bundle.similarity,
        async_batched_similarity_fn=bundle.abatched_similarity,
        async_embed_query_fn=None,
        async_embed_texts_fn=None,
        embed_query_fn=None,
        embed_texts_fn=None,
    )
    runtime = AgenticRuntime(
        config=config,
        provider_bundle=bundle,
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
        provider_registry=provider_registry,
    )

    payload = await runtime.search_papers_smart(
        query="transformers",
        limit=5,
        latency_profile="deep",
    )

    assert payload["results"]
    assert not [
        outcome
        for outcome in payload["strategyMetadata"]["providerOutcomes"]
        if outcome["provider"] == "openai" and outcome["endpoint"] == "embeddings.create"
    ]


@pytest.mark.asyncio
async def test_async_planner_uses_total_timeout_and_falls_back() -> None:
    config = AgenticConfig(
        enabled=True,
        provider="openai",
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
        openai_timeout_seconds=0.01,
    )
    bundle = OpenAIProviderBundle(config, api_key="sk-test")

    class _ResponsesClient:
        async def parse(self, **kwargs: object) -> object:
            del kwargs
            await asyncio.sleep(60)
            return object()

    class _AsyncClient:
        responses = _ResponsesClient()

    bundle._async_openai_client = _AsyncClient()

    decision = await asyncio.wait_for(
        bundle.aplan_search(
            query="transformers",
            mode="auto",
        ),
        timeout=0.5,
    )

    assert isinstance(decision, PlannerDecision)
    assert decision.intent in {
        "discovery",
        "review",
        "known_item",
        "author",
        "citation",
    }


@pytest.mark.asyncio
async def test_async_embeddings_use_total_timeout_guard() -> None:
    config = AgenticConfig(
        enabled=True,
        provider="openai",
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
        openai_timeout_seconds=0.01,
    )
    bundle = OpenAIProviderBundle(config, api_key="sk-test")

    class _EmbeddingsClient:
        async def create(self, **kwargs: object) -> object:
            del kwargs
            await asyncio.sleep(60)
            return {"data": [{"embedding": [0.1, 0.2]}]}

    class _AsyncClient:
        embeddings = _EmbeddingsClient()

    bundle._async_openai_client = _AsyncClient()

    embeddings = await asyncio.wait_for(
        bundle.aembed_texts(["alpha paper abstract"], request_id="total-timeout"),
        timeout=0.5,
    )

    assert embeddings == [None]


@pytest.mark.asyncio
async def test_openai_embeddings_stall_does_not_block_planner_endpoint() -> None:
    registry = ProviderDiagnosticsRegistry()
    embeddings_started = asyncio.Event()
    planner_finished = asyncio.Event()
    release_embeddings = asyncio.Event()

    async def _slow_embeddings() -> dict[str, object]:
        embeddings_started.set()
        await release_embeddings.wait()
        return {"data": [{"embedding": [0.1, 0.2]}]}

    async def _fast_planner() -> dict[str, object]:
        planner_finished.set()
        return {"output": []}

    embeddings_task = asyncio.create_task(
        execute_provider_call(
            provider="openai",
            endpoint="embeddings.create",
            operation=_slow_embeddings,
            registry=registry,
            policy=ProviderPolicy(concurrency_limit=1, max_attempts=1),
            request_id="embeddings-request",
        )
    )

    await asyncio.wait_for(embeddings_started.wait(), timeout=1.0)

    planner_task = asyncio.create_task(
        execute_provider_call(
            provider="openai",
            endpoint="responses.parse:planner",
            operation=_fast_planner,
            registry=registry,
            policy=ProviderPolicy(concurrency_limit=1, max_attempts=1),
            request_id="planner-request",
        )
    )

    await asyncio.wait_for(planner_finished.wait(), timeout=1.0)
    planner_result = await planner_task

    release_embeddings.set()
    await embeddings_task

    assert planner_result.outcome.status_bucket == "success"


@pytest.mark.asyncio
async def test_enrich_paper_parallelizes_doi_known_crossref_and_unpaywall() -> None:
    crossref_started = asyncio.Event()
    unpaywall_started = asyncio.Event()
    overlap = asyncio.Event()
    release = asyncio.Event()

    class SlowCrossrefClient:
        async def get_work(self, doi: str) -> dict[str, object]:
            crossref_started.set()
            if unpaywall_started.is_set():
                overlap.set()
            await release.wait()
            return {
                "doi": doi,
                "title": "Crossref Paper",
                "publisher": "Crossref Publisher",
                "publicationType": "journal-article",
            }

    class SlowUnpaywallClient:
        async def get_open_access(self, doi: str) -> dict[str, object]:
            unpaywall_started.set()
            if crossref_started.is_set():
                overlap.set()
            await release.wait()
            return {
                "doi": doi,
                "isOa": True,
                "oaStatus": "gold",
                "bestOaUrl": "https://oa.example/landing",
                "pdfUrl": "https://oa.example/file.pdf",
                "license": "cc-by",
                "journalIsInDoaj": True,
            }

    service = PaperEnrichmentService(
        crossref_client=SlowCrossrefClient(),  # type: ignore[arg-type]
        unpaywall_client=SlowUnpaywallClient(),  # type: ignore[arg-type]
        enable_crossref=True,
        enable_unpaywall=True,
    )

    task = asyncio.create_task(service.enrich_paper(doi="10.1234/example"))

    await asyncio.wait_for(overlap.wait(), timeout=1.0)
    release.set()
    response = await task

    assert response.crossref is not None
    assert response.unpaywall is not None
    assert response.crossref.found is True
    assert response.unpaywall.found is True
    assert response.doi_resolution.resolved_doi == "10.1234/example"


@pytest.mark.asyncio
async def test_enrich_paper_reuses_existing_enrichments_without_refetching() -> None:
    class UnexpectedCrossrefClient:
        async def get_work(self, doi: str) -> dict[str, object]:
            raise AssertionError(f"Crossref should not be called for {doi}")

    class UnexpectedUnpaywallClient:
        async def get_open_access(self, doi: str) -> dict[str, object]:
            raise AssertionError(f"Unpaywall should not be called for {doi}")

    service = PaperEnrichmentService(
        crossref_client=UnexpectedCrossrefClient(),  # type: ignore[arg-type]
        unpaywall_client=UnexpectedUnpaywallClient(),  # type: ignore[arg-type]
        enable_crossref=True,
        enable_unpaywall=True,
    )

    response = await service.enrich_paper(
        paper={
            "paperId": "paper-1",
            "title": "Existing enrichments",
            "enrichments": {
                "crossref": {
                    "doi": "10.1234/existing",
                    "publisher": "Existing Publisher",
                },
                "unpaywall": {
                    "doi": "10.1234/existing",
                    "isOa": True,
                    "oaStatus": "gold",
                    "bestOaUrl": "https://oa.example/landing",
                    "pdfUrl": "https://oa.example/file.pdf",
                    "license": "cc-by",
                    "journalIsInDoaj": True,
                },
            },
        }
    )

    assert response.crossref is not None
    assert response.crossref.enrichment is not None
    assert response.unpaywall is not None
    assert response.unpaywall.enrichment is not None
    assert response.crossref.found is True
    assert response.crossref.enrichment.doi == "10.1234/existing"
    assert response.unpaywall.found is True
    assert response.unpaywall.enrichment.is_oa is True
    assert response.doi_resolution.resolution_source == "existing_enrichment"


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


@pytest.mark.asyncio
async def test_expand_research_graph_uses_scored_frontier_for_next_hop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GraphSemanticClient(RecordingSemanticClient):
        async def get_paper_citations(self, **kwargs) -> dict:
            self.calls.append(("get_paper_citations", kwargs))
            paper_id = kwargs["paper_id"]
            if paper_id == "seed-1":
                return {
                    "data": [{"paperId": f"low-{index}", "title": f"Low {index}"} for index in range(1, 6)]
                    + [{"paperId": "high-6", "title": "High 6"}]
                }
            if paper_id == "high-6":
                return {
                    "data": [
                        {
                            "paperId": "grandchild-1",
                            "title": "Grandchild node",
                        }
                    ]
                }
            return {"data": []}

    semantic = GraphSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)
    record = registry.save_result_set(
        source_tool="search_papers_smart",
        payload={
            "data": [
                {
                    "paperId": "seed-1",
                    "title": "Seed paper",
                    "canonicalId": "seed-1",
                    "recommendedExpansionId": "seed-1",
                    "expansionIdStatus": "portable",
                    "source": "semantic_scholar",
                }
            ]
        },
    )

    async def _fake_frontier_scores(
        *,
        seed: dict[str, Any],
        related_papers: list[dict[str, Any]],
        provider_bundle: Any,
        intent_text: str | None = None,
    ) -> list[float]:
        del seed, provider_bundle, intent_text
        if related_papers and related_papers[0]["paperId"] == "low-1":
            return [0.1, 0.2, 0.3, 0.4, 0.5, 0.99]
        return [0.8 for _ in related_papers]

    monkeypatch.setattr(
        "scholar_search_mcp.agentic.graphs._graph_frontier_scores",
        _fake_frontier_scores,
    )

    graph = await runtime.expand_research_graph(
        seed_paper_ids=None,
        seed_search_session_id=record.search_session_id,
        direction="citations",
        hops=2,
        per_seed_limit=6,
    )

    node_ids = {node["id"] for node in graph["nodes"]}

    assert "grandchild-1" in node_ids
    assert any(call[0] == "get_paper_citations" and call[1]["paper_id"] == "high-6" for call in semantic.calls)


@pytest.mark.asyncio
async def test_expand_research_graph_balanced_mode_skips_embedding_scoring() -> None:
    class GraphSemanticClient(RecordingSemanticClient):
        async def get_paper_citations(self, **kwargs) -> dict:
            self.calls.append(("get_paper_citations", kwargs))
            paper_id = kwargs["paper_id"]
            if paper_id == "seed-1":
                return {
                    "data": [
                        {
                            "paperId": "child-1",
                            "title": "Child paper",
                            "year": 2024,
                            "citationCount": 5,
                        }
                    ]
                }
            return {"data": []}

    semantic = GraphSemanticClient()
    openalex = RecordingOpenAlexClient()
    config = AgenticConfig(
        enabled=True,
        provider="openai",
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
    )
    provider_registry = ProviderDiagnosticsRegistry()
    bundle = OpenAIProviderBundle(
        config,
        api_key="sk-test",
        provider_registry=provider_registry,
    )

    class _EmbeddingsClient:
        async def create(self, **kwargs: object) -> object:
            raise AssertionError(f"balanced graph expansion should not call embeddings: {kwargs!r}")

    class _AsyncClient:
        embeddings = _EmbeddingsClient()

    bundle._async_openai_client = _AsyncClient()
    bundle._openai_client = None
    bundle._embeddings = None

    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False)
    runtime = AgenticRuntime(
        config=config,
        provider_bundle=bundle,
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
        provider_registry=provider_registry,
    )
    record = registry.save_result_set(
        source_tool="search_papers_smart",
        payload={
            "data": [
                {
                    "paperId": "seed-1",
                    "title": "Seed paper",
                    "canonicalId": "seed-1",
                    "recommendedExpansionId": "seed-1",
                    "expansionIdStatus": "portable",
                    "source": "semantic_scholar",
                }
            ]
        },
    )

    graph = await runtime.expand_research_graph(
        seed_paper_ids=None,
        seed_search_session_id=record.search_session_id,
        direction="citations",
        hops=1,
        per_seed_limit=5,
        latency_profile="balanced",
    )

    assert graph["frontier"]


@pytest.mark.asyncio
async def test_expand_research_graph_prefers_frontier_items_aligned_to_original_search_intent() -> None:
    class GraphSemanticClient(RecordingSemanticClient):
        async def get_paper_citations(self, **kwargs) -> dict:
            self.calls.append(("get_paper_citations", kwargs))
            paper_id = kwargs["paper_id"]
            if paper_id == "seed-1":
                return {
                    "data": [
                        {
                            "paperId": "off-topic",
                            "title": "Industrial noise control in factories",
                            "abstract": "Engineering controls for machinery noise in industrial settings.",
                            "year": 2025,
                            "citationCount": 2500,
                            "recommendedExpansionId": "off-topic",
                            "expansionIdStatus": "portable",
                        },
                        {
                            "paperId": "on-topic",
                            "title": "Noise exposure effects on birds",
                            "abstract": "Anthropogenic noise alters bird behaviour and wildlife responses.",
                            "year": 2024,
                            "citationCount": 40,
                            "recommendedExpansionId": "on-topic",
                            "expansionIdStatus": "portable",
                        },
                    ]
                }
            if paper_id == "on-topic":
                return {
                    "data": [
                        {
                            "paperId": "grandchild-1",
                            "title": "Bird communication under anthropogenic noise",
                            "recommendedExpansionId": "grandchild-1",
                            "expansionIdStatus": "portable",
                        }
                    ]
                }
            return {"data": []}

    semantic = GraphSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)
    record = registry.save_result_set(
        source_tool="search_papers_smart",
        payload={
            "data": [
                {
                    "paperId": "seed-1",
                    "title": "Environmental effects review",
                    "canonicalId": "seed-1",
                    "recommendedExpansionId": "seed-1",
                    "expansionIdStatus": "portable",
                    "source": "semantic_scholar",
                }
            ]
        },
        query="anthropogenic noise effects on wildlife",
        metadata={"strategyMetadata": {"normalizedQuery": "anthropogenic noise effects on wildlife"}},
    )

    graph = await runtime.expand_research_graph(
        seed_paper_ids=None,
        seed_search_session_id=record.search_session_id,
        direction="citations",
        hops=2,
        per_seed_limit=5,
    )

    frontier_ids = [node["id"] for node in graph["frontier"]]

    assert "on-topic" in frontier_ids
    assert "off-topic" not in frontier_ids
    assert "grandchild-1" in frontier_ids
    assert any(call[0] == "get_paper_citations" and call[1]["paper_id"] == "on-topic" for call in semantic.calls)
    assert all(call[1]["paper_id"] != "off-topic" for call in semantic.calls if call[0] == "get_paper_citations")


@pytest.mark.asyncio
async def test_search_papers_smart_known_item_falls_back_to_title_match_instead_of_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    async def fake_resolve_citation(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {"bestMatch": None, "alternatives": [], "resolutionConfidence": "low"}

    async def fake_search_papers_match(**kwargs: object) -> dict[str, object]:
        semantic.calls.append(("search_papers_match", dict(kwargs)))
        return {
            "paperId": "match-1",
            "title": "Recovered title match",
            "year": 2023,
            "venue": "Journal of Tests",
            "matchStrategy": "fuzzy_search",
        }

    monkeypatch.setattr("scholar_search_mcp.agentic.graphs.resolve_citation", fake_resolve_citation)
    semantic.search_papers_match = fake_search_papers_match  # type: ignore[method-assign]

    smart = await runtime.search_papers_smart(
        query="Dooling Popper traffic noise road construction birds 2016",
        limit=5,
        mode="known_item",
        latency_profile="fast",
    )

    assert smart["results"]
    assert smart["results"][0]["paper"]["paperId"] == "match-1"
    assert any("title-style recovery" in warning for warning in smart["strategyMetadata"]["driftWarnings"])


@pytest.mark.asyncio
async def test_search_papers_smart_known_item_uses_openalex_autocomplete_when_other_recovery_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    async def fake_resolve_citation(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {"bestMatch": None, "alternatives": [], "resolutionConfidence": "low"}

    async def fake_search_papers_match(**kwargs: object) -> dict[str, object]:
        semantic.calls.append(("search_papers_match", dict(kwargs)))
        return {"matchFound": False}

    async def fake_openalex_autocomplete(**kwargs: object) -> dict[str, object]:
        openalex.calls.append(("paper_autocomplete", dict(kwargs)))
        return {
            "matches": [
                {
                    "id": "W123",
                    "displayName": (
                        "Assessing the influence of rocket launch and landing noise on threatened and endangered "
                        "species at Vandenberg Space Force Base"
                    ),
                }
            ]
        }

    async def fake_openalex_details(**kwargs: object) -> dict[str, object]:
        openalex.calls.append(("get_paper_details", dict(kwargs)))
        return {
            "paperId": "W123",
            "title": (
                "Assessing the influence of rocket launch and landing noise on threatened and endangered species at "
                "Vandenberg Space Force Base"
            ),
            "source": "openalex",
            "recommendedExpansionId": "10.1121/10.0018203",
            "expansionIdStatus": "portable",
        }

    monkeypatch.setattr("scholar_search_mcp.agentic.graphs.resolve_citation", fake_resolve_citation)
    semantic.search_papers_match = fake_search_papers_match  # type: ignore[method-assign]
    openalex.paper_autocomplete = fake_openalex_autocomplete  # type: ignore[method-assign]
    openalex.get_paper_details = fake_openalex_details  # type: ignore[method-assign]

    smart = await runtime.search_papers_smart(
        query=(
            "Assessing the influence of rocket launch and landing noise on threatened and endangered species at "
            "Vandenberg Space Force Base"
        ),
        limit=5,
        mode="known_item",
        latency_profile="fast",
    )

    assert smart["results"]
    assert smart["results"][0]["paper"]["paperId"] == "W123"
    assert smart["results"][0]["retrievedBy"] == ["openalex_autocomplete"]


def test_grounded_comparison_answer_filters_question_echo_and_year_tokens() -> None:
    answer = _build_grounded_comparison_answer(
        question="Which results are directly about rocket launch noise or wildlife effects near launch sites?",
        evidence_papers=[
            {
                "title": "Acoustic monitoring near launch pads",
                "abstract": "Acoustic monitoring quantified impulsive exposure near launch facilities.",
                "year": 2024,
                "venue": "Journal A",
            },
            {
                "title": "Acoustic monitoring for endangered species response",
                "abstract": "Acoustic monitoring linked species observations to disturbance conditions.",
                "year": 2024,
                "venue": "Journal B",
            },
            {
                "title": "Acoustic monitoring and habitat response",
                "abstract": "Acoustic monitoring supported repeated habitat-response surveys.",
                "year": 2024,
                "venue": "Journal C",
            },
        ],
    )

    assert "Acoustic" in answer
    assert "Shared ground: these papers converge on Noise" not in answer
    assert "Shared ground: these papers converge on 2024" not in answer


def test_finalize_theme_label_replaces_article_noise_label_with_derived_terms() -> None:
    label = _finalize_theme_label(
        raw_label="The / Noise",
        seed_terms=["rocket launch", "species response"],
        papers=[
            {"title": "Launch acoustics and species monitoring", "abstract": "Acoustic monitoring near launch pads."},
            {"title": "Species response to launch acoustics", "abstract": "Monitoring species response to acoustics."},
        ],
    )

    assert label != "The / Noise"
    assert "Launch" in label or "Acoustics" in label or "Species" in label


@pytest.mark.asyncio
async def test_graph_frontier_scores_penalize_off_topic_high_citation_papers() -> None:
    bundle = resolve_provider_bundle(_deterministic_config(), openai_api_key=None)
    scores = await _graph_frontier_scores(
        seed={
            "title": "Anthropogenic noise effects on wildlife",
            "abstract": "Bird and mammal responses to anthropogenic noise.",
        },
        related_papers=[
            {
                "paperId": "off-topic",
                "title": "Brewery waste restoration methods",
                "abstract": "Industrial wastewater treatment and restoration planning.",
                "year": 2026,
                "citationCount": 1200,
            },
            {
                "paperId": "on-topic",
                "title": "Noise exposure effects on birds",
                "abstract": "Wildlife acoustic disturbance and bird responses.",
                "year": 2020,
                "citationCount": 10,
            },
        ],
        provider_bundle=bundle,
    )

    assert scores[1] > scores[0]


def test_grounded_expansions_filter_stopwords_and_single_paper_noise() -> None:
    config = _deterministic_config()

    variants = grounded_expansion_candidates(
        original_query="tool-using agents for literature review",
        papers=[
            {
                "title": "Agentic systematic review workflow",
                "abstract": ("This workflow supports literature review with agent orchestration."),
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


def test_grounded_expansions_dedupe_near_duplicate_variants() -> None:
    config = _deterministic_config().for_latency_profile("balanced")

    variants = grounded_expansion_candidates(
        original_query=(
            "Florida Scrub-Jay Aphelocoma coerulescens demography habitat survival reproduction conservation Brevard"
        ),
        papers=[
            {
                "title": "Florida Scrub-Jay demography Brevard County",
                "abstract": ("Demography and survival in Brevard County scrub-jay populations."),
            },
            {
                "title": "Florida Scrub-Jay demography habitat survival",
                "abstract": ("Habitat, survival, and reproduction of Florida Scrub-Jays."),
            },
            {
                "title": ("Florida Scrub-Jay metapopulation viability habitat connectivity"),
                "abstract": "Metapopulation viability and habitat connectivity.",
            },
        ],
        config=config,
    )

    expanded = [candidate.variant.lower() for candidate in variants]

    assert len(expanded) <= 2
    assert not (
        any("demography brevard county" in variant for variant in expanded)
        and any("demography habitat survival" in variant for variant in expanded)
    )


def test_balanced_profile_reduces_expansion_breadth() -> None:
    config = _deterministic_config().for_latency_profile("balanced")

    assert config.max_grounded_variants == 2
    assert config.max_total_variants == 4
    assert config.candidate_pool_size == 50


def test_provider_limits_reduce_expansion_fetch_sizes_for_balanced_review() -> None:
    initial = provider_limits(intent="review", widened=True, latency_profile="balanced")
    expansion = provider_limits(
        intent="review",
        widened=True,
        is_expansion=True,
        latency_profile="balanced",
    )

    assert initial["semantic_scholar"] > expansion["semantic_scholar"]
    assert initial["openalex"] > expansion["openalex"]
    assert expansion["semantic_scholar"] == 6
    assert expansion["openalex"] == 6
    assert expansion["serpapi_google_scholar"] == 2


@pytest.mark.asyncio
async def test_classify_query_respects_explicit_review_mode() -> None:
    class _Bundle:
        async def aplan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="discovery",
                candidateConcepts=["agents"],
                followUpMode="qa",
            )

    _, planner = await classify_query(
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


@pytest.mark.asyncio
async def test_classify_query_treats_citation_like_input_as_known_item() -> None:
    class _Bundle:
        async def aplan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="discovery",
                candidateConcepts=["planetary boundaries"],
                followUpMode="qa",
            )

    _, planner = await classify_query(
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


@pytest.mark.asyncio
async def test_classify_query_routes_title_like_query_to_known_item() -> None:
    class _Bundle:
        async def aplan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="discovery",
                candidateConcepts=["transformers"],
                followUpMode="qa",
            )

    _, planner = await classify_query(
        query="Attention Is All You Need",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    assert planner.intent == "known_item"


@pytest.mark.asyncio
async def test_rerank_candidates_prefers_multi_facet_consensus_match() -> None:
    provider_bundle = resolve_provider_bundle(
        _deterministic_config(),
        openai_api_key=None,
    )
    ranked = await rerank_candidates(
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
                    "abstract": ("Autonomous agents combine tool use and retrieval for literature review workflows."),
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


@pytest.mark.asyncio
async def test_rerank_candidates_downweights_serpapi_echo_without_title_match() -> None:
    provider_bundle = resolve_provider_bundle(
        _deterministic_config(),
        openai_api_key=None,
    )
    ranked = await rerank_candidates(
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
                    "abstract": ("Autonomous agents combine tool use and retrieval for literature review workflows."),
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
    assert ranked[0]["scoreBreakdown"]["titleFacetCoverage"] > (ranked[1]["scoreBreakdown"]["titleFacetCoverage"])


@pytest.mark.asyncio
async def test_rerank_candidates_penalizes_review_papers_missing_anchor_terms() -> None:
    provider_bundle = resolve_provider_bundle(
        _deterministic_config(),
        openai_api_key=None,
    )
    ranked = await rerank_candidates(
        query="tool-using agents for literature review",
        merged_candidates=[
            {
                "paper": {
                    "paperId": "sem-medical",
                    "title": ("Cholestatic pruritus treatments: a systematic literature review"),
                    "abstract": (
                        "Therapeutic agents were evaluated with multiple clinical tools across treatment studies."
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
                    "abstract": ("A survey of tool-using autonomous agents and research workflows."),
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
    assert ranked[0]["scoreBreakdown"]["titleAnchorCoverage"] > (ranked[1]["scoreBreakdown"]["titleAnchorCoverage"])


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
