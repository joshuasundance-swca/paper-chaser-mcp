import asyncio
import logging
import types
from typing import Any, cast

import pytest

from paper_chaser_mcp import server
from paper_chaser_mcp.agentic import (
    AgenticConfig,
    AgenticRuntime,
    WorkspaceRegistry,
    resolve_provider_bundle,
)
from paper_chaser_mcp.agentic.graphs import (
    _build_grounded_comparison_answer,
    _finalize_theme_label,
    _graph_frontier_scores,
)
from paper_chaser_mcp.agentic.models import PlannerDecision
from paper_chaser_mcp.agentic.planner import (
    classify_query,
    grounded_expansion_candidates,
    looks_like_exact_title,
)
from paper_chaser_mcp.agentic.providers import (
    AnthropicProviderBundle,
    AzureOpenAIProviderBundle,
    GoogleProviderBundle,
    NvidiaProviderBundle,
    OpenAIProviderBundle,
)
from paper_chaser_mcp.agentic.ranking import merge_candidates, rerank_candidates
from paper_chaser_mcp.agentic.retrieval import (
    RetrievedCandidate,
    provider_limits,
    retrieve_variant,
)
from paper_chaser_mcp.enrichment import PaperEnrichmentService
from paper_chaser_mcp.provider_runtime import (
    ProviderDiagnosticsRegistry,
    ProviderPolicy,
    execute_provider_call,
)
from tests.helpers import (
    RecordingCrossrefClient,
    RecordingOpenAlexClient,
    RecordingScholarApiClient,
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
    scholarapi: RecordingScholarApiClient | None = None,
    ecos: Any = None,
    federal_register: Any = None,
    govinfo: Any = None,
    disable_embeddings: bool = True,
    enable_semantic_scholar: bool = True,
    enable_openalex: bool = True,
    enable_scholarapi: bool = False,
    enable_ecos: bool = False,
    enable_federal_register: bool = False,
    enable_govinfo_cfr: bool = False,
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
        disable_embeddings=disable_embeddings,
    )
    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False)
    runtime = AgenticRuntime(
        config=config,
        provider_bundle=resolve_provider_bundle(config, openai_api_key=None),
        workspace_registry=registry,
        client=semantic,
        core_client=object(),
        openalex_client=openalex,
        scholarapi_client=scholarapi,
        arxiv_client=object(),
        serpapi_client=None,
        ecos_client=ecos,
        federal_register_client=federal_register,
        govinfo_client=govinfo,
        enable_core=False,
        enable_semantic_scholar=enable_semantic_scholar,
        enable_openalex=enable_openalex,
        enable_scholarapi=enable_scholarapi,
        enable_arxiv=False,
        enable_serpapi=False,
        enable_ecos=enable_ecos,
        enable_federal_register=enable_federal_register,
        enable_govinfo_cfr=enable_govinfo_cfr,
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


def _runtime_with_provider_bundle(
    *,
    config: AgenticConfig,
    provider_bundle: Any,
    semantic: RecordingSemanticClient,
    openalex: RecordingOpenAlexClient,
    provider_registry: ProviderDiagnosticsRegistry | None = None,
) -> tuple[WorkspaceRegistry, AgenticRuntime]:
    registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        similarity_fn=provider_bundle.similarity,
        async_batched_similarity_fn=provider_bundle.abatched_similarity,
        async_embed_query_fn=(provider_bundle.aembed_query if provider_bundle.supports_embeddings() else None),
        async_embed_texts_fn=(provider_bundle.aembed_texts if provider_bundle.supports_embeddings() else None),
        embed_query_fn=(provider_bundle.embed_query if provider_bundle.supports_embeddings() else None),
        embed_texts_fn=(provider_bundle.embed_texts if provider_bundle.supports_embeddings() else None),
    )
    runtime = AgenticRuntime(
        config=config,
        provider_bundle=provider_bundle,
        workspace_registry=registry,
        client=semantic,
        core_client=object(),
        openalex_client=openalex,
        arxiv_client=object(),
        serpapi_client=None,
        ecos_client=None,
        federal_register_client=None,
        govinfo_client=None,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=True,
        enable_arxiv=False,
        enable_serpapi=False,
        enable_ecos=False,
        enable_federal_register=False,
        enable_govinfo_cfr=False,
        provider_registry=provider_registry,
    )
    return registry, runtime


class _StructuredInvoker:
    def __init__(self, model: "_SequenceModel") -> None:
        self._model = model

    def invoke(self, messages: list[tuple[str, str]]) -> Any:
        assert messages
        return self._model.structured_responses.pop(0)

    async def ainvoke(self, messages: list[tuple[str, str]]) -> Any:
        assert messages
        return self._model.structured_responses.pop(0)


class _SequenceModel:
    def __init__(self, *, structured_responses: list[Any], text_responses: list[str]) -> None:
        self.structured_responses = list(structured_responses)
        self.text_responses = list(text_responses)

    def with_structured_output(self, schema: Any, method: str | None = None) -> _StructuredInvoker:
        assert schema is not None
        del method
        return _StructuredInvoker(self)

    def invoke(self, messages: list[tuple[str, str]]) -> Any:
        assert messages
        return types.SimpleNamespace(content=self.text_responses.pop(0))

    async def ainvoke(self, messages: list[tuple[str, str]]) -> Any:
        assert messages
        return types.SimpleNamespace(content=self.text_responses.pop(0))


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
    assert smart["verifiedFindings"] or smart["likelyUnverified"]
    assert smart["structuredSources"]
    assert smart["coverageSummary"]["searchMode"] == "smart_literature_review"

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
    assert ask["answerStatus"] in {"answered", "abstained", "insufficient_evidence"}
    if ask["answerStatus"] == "answered":
        assert ask["answer"]
    else:
        assert ask["answer"] is None
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
    assert landscape["structuredSources"]

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
async def test_search_papers_smart_routes_regulatory_queries_to_primary_sources() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    class FakeEcosClient:
        async def search_species(self, *, query: str, limit: int = 10, match_mode: str = "auto") -> dict[str, Any]:
            del limit, match_mode
            assert "regulatory history" in query.lower()
            return {
                "query": query,
                "matchMode": "auto",
                "total": 1,
                "data": [
                    {
                        "speciesId": "sp-1",
                        "commonName": "California condor",
                        "scientificName": "Gymnogyps californianus",
                        "profileUrl": "https://ecos.fws.gov/ecp/species/sp-1",
                    }
                ],
            }

        async def get_species_profile(self, *, species_id: str) -> dict[str, Any]:
            assert species_id == "sp-1"
            return {
                "species": {
                    "speciesId": "sp-1",
                    "commonName": "California condor",
                    "scientificName": "Gymnogyps californianus",
                    "profileUrl": "https://ecos.fws.gov/ecp/species/sp-1",
                },
                "speciesEntities": [
                    {
                        "entityId": 12,
                        "status": "Endangered",
                        "statusCategory": "Listed",
                        "listingDate": "1967-03-11",
                    }
                ],
            }

        async def list_species_documents(
            self, *, species_id: str, document_kinds: list[str] | None = None
        ) -> dict[str, Any]:
            del document_kinds
            assert species_id == "sp-1"
            return {
                "speciesId": species_id,
                "total": 2,
                "documentKindsApplied": [],
                "data": [
                    {
                        "documentKind": "recovery_plan",
                        "title": "California Condor Recovery Plan",
                        "url": "https://ecos.fws.gov/docs/recovery_plan.pdf",
                        "documentDate": "2023-01-15",
                        "documentType": "Recovery Plan",
                    },
                    {
                        "documentKind": "federal_register",
                        "title": "Endangered Status for California Condor",
                        "url": "https://www.govinfo.gov/link/fr/32/4001",
                        "documentDate": "1967-03-11",
                        "frCitation": "32 FR 4001",
                        "documentType": "Final Rule",
                    },
                ],
            }

    class FakeFederalRegisterClient:
        async def search_documents(self, *, query: str, limit: int = 10, **kwargs: Any) -> dict[str, Any]:
            del limit, kwargs
            assert "condor" in query.lower()
            return {
                "total": 1,
                "data": [
                    {
                        "documentNumber": "2024-12345",
                        "title": "Critical Habitat Revision for California Condor",
                        "documentType": "RULE",
                        "publicationDate": "2024-02-01",
                        "citation": "89 FR 83510",
                        "htmlUrl": "https://www.federalregister.gov/d/2024-12345",
                        "cfrReferences": ["50 CFR 17.95"],
                    }
                ],
            }

    class FakeGovInfoClient:
        async def get_cfr_text(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["title_number"] == 50
            assert kwargs["part_number"] == 17
            return {
                "titleNumber": 50,
                "partNumber": 17,
                "sectionNumber": "95",
                "citation": "50 CFR 17.95",
                "effectiveDate": "2024-02-01",
                "sourceUrl": "https://www.govinfo.gov/app/details/CFR-2024-title50-vol1/CFR-2024-title50-vol1-sec17-95",
                "markdown": "Authoritative CFR text",
                "verificationStatus": "verified_primary_source",
            }

    _, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        ecos=FakeEcosClient(),
        federal_register=FakeFederalRegisterClient(),
        govinfo=FakeGovInfoClient(),
        enable_ecos=True,
        enable_federal_register=True,
        enable_govinfo_cfr=True,
    )

    payload = await runtime.search_papers_smart(
        query="Regulatory history of California condor under 50 CFR 17.95",
        limit=5,
    )

    assert payload["results"] == []
    assert payload["strategyMetadata"]["intent"] == "regulatory"
    assert payload["structuredSources"]
    assert payload["verifiedFindings"]
    assert payload["coverageSummary"]["searchMode"] == "regulatory_primary_source"
    assert payload["coverageSummary"]["primaryDocumentCoverage"]["currentTextRequested"] is True
    assert payload["coverageSummary"]["primaryDocumentCoverage"]["currentTextSatisfied"] is True
    assert payload["regulatoryTimeline"]["events"]
    assert any(source["provider"] == "govinfo" for source in payload["structuredSources"])
    assert any(source["provider"] == "ecos" for source in payload["structuredSources"])
    assert any(source["provider"] == "federal_register" for source in payload["structuredSources"])


@pytest.mark.asyncio
async def test_search_papers_smart_regulatory_filters_unanchored_federal_register_hits() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    class FakeEcosClient:
        async def search_species(self, *, query: str, limit: int = 10, match_mode: str = "auto") -> dict[str, Any]:
            del limit, match_mode
            return {
                "query": query,
                "matchMode": "auto",
                "total": 1,
                "data": [
                    {
                        "speciesId": "sp-1",
                        "commonName": "California condor",
                        "scientificName": "Gymnogyps californianus",
                        "profileUrl": "https://ecos.fws.gov/ecp/species/sp-1",
                    }
                ],
            }

        async def get_species_profile(self, *, species_id: str) -> dict[str, Any]:
            assert species_id == "sp-1"
            return {
                "species": {
                    "speciesId": "sp-1",
                    "commonName": "California condor",
                    "scientificName": "Gymnogyps californianus",
                    "profileUrl": "https://ecos.fws.gov/ecp/species/sp-1",
                },
                "speciesEntities": [],
            }

        async def list_species_documents(
            self, *, species_id: str, document_kinds: list[str] | None = None
        ) -> dict[str, Any]:
            del document_kinds
            assert species_id == "sp-1"
            return {
                "speciesId": species_id,
                "total": 1,
                "documentKindsApplied": [],
                "data": [
                    {
                        "documentKind": "federal_register",
                        "title": "Endangered Status for California Condor",
                        "url": "https://www.govinfo.gov/link/fr/32/4001",
                        "documentDate": "1967-03-11",
                        "frCitation": "32 FR 4001",
                        "documentType": "Final Rule",
                    }
                ],
            }

    class FakeFederalRegisterClient:
        async def search_documents(self, *, query: str, limit: int = 10, **kwargs: Any) -> dict[str, Any]:
            del query, limit, kwargs
            return {
                "total": 3,
                "data": [
                    {
                        "documentNumber": "2024-12345",
                        "title": "Critical Habitat Revision for California Condor",
                        "documentType": "RULE",
                        "publicationDate": "2024-02-01",
                        "citation": "89 FR 83510",
                        "htmlUrl": "https://www.federalregister.gov/d/2024-12345",
                        "cfrReferences": ["50 CFR 17.95"],
                    },
                    {
                        "documentNumber": "2010-00001",
                        "title": "Designation of Critical Habitat for Polar Bear",
                        "documentType": "RULE",
                        "publicationDate": "2010-01-01",
                        "citation": "75 FR 76086",
                        "htmlUrl": "https://www.federalregister.gov/d/2010-00001",
                        "cfrReferences": ["50 CFR 17.95"],
                    },
                    {
                        "documentNumber": "2008-00002",
                        "title": "Wood Stork Recovery Notice",
                        "documentType": "NOTICE",
                        "publicationDate": "2008-05-01",
                        "citation": "73 FR 12345",
                        "htmlUrl": "https://www.federalregister.gov/d/2008-00002",
                        "cfrReferences": ["50 CFR 17.95"],
                    },
                ],
            }

    class FakeGovInfoClient:
        async def get_cfr_text(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["title_number"] == 50
            assert kwargs["part_number"] == 17
            return {
                "titleNumber": 50,
                "partNumber": 17,
                "sectionNumber": "95",
                "citation": "50 CFR 17.95",
                "effectiveDate": "2024-02-01",
                "sourceUrl": "https://www.govinfo.gov/app/details/CFR-2024-title50-vol1/CFR-2024-title50-vol1-sec17-95",
                "markdown": "Authoritative CFR text",
                "verificationStatus": "verified_primary_source",
            }

    _, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        ecos=FakeEcosClient(),
        federal_register=FakeFederalRegisterClient(),
        govinfo=FakeGovInfoClient(),
        enable_ecos=True,
        enable_federal_register=True,
        enable_govinfo_cfr=True,
    )

    payload = await runtime.search_papers_smart(
        query="Regulatory history of California condor under 50 CFR 17.95",
        limit=5,
    )

    titles = {source["title"] for source in payload["structuredSources"]}
    event_titles = {event["title"] for event in payload["regulatoryTimeline"]["events"]}

    assert "Critical Habitat Revision for California Condor" in titles
    assert "Critical Habitat Revision for California Condor" in event_titles
    assert "Designation of Critical Habitat for Polar Bear" not in titles
    assert "Designation of Critical Habitat for Polar Bear" not in event_titles
    assert "Wood Stork Recovery Notice" not in titles
    assert "Wood Stork Recovery Notice" not in event_titles
    assert not any("Polar Bear" in finding for finding in payload["verifiedFindings"])
    assert not any("Wood Stork" in finding for finding in payload["verifiedFindings"])
    candidate_titles = {lead["title"] for lead in payload["candidateLeads"]}
    assert "Designation of Critical Habitat for Polar Bear" in candidate_titles
    assert "Wood Stork Recovery Notice" in candidate_titles
    assert all(lead["topicalRelevance"] == "off_topic" for lead in payload["candidateLeads"])


@pytest.mark.asyncio
async def test_search_papers_smart_regulatory_marks_history_only_when_current_cfr_text_is_unresolved() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    class FakeEcosClient:
        async def search_species(self, *, query: str, limit: int = 10, match_mode: str = "auto") -> dict[str, Any]:
            del limit, match_mode
            return {
                "query": query,
                "matchMode": "auto",
                "total": 1,
                "data": [
                    {
                        "speciesId": "sp-1",
                        "commonName": "California condor",
                        "scientificName": "Gymnogyps californianus",
                        "profileUrl": "https://ecos.fws.gov/ecp/species/sp-1",
                    }
                ],
            }

        async def get_species_profile(self, *, species_id: str) -> dict[str, Any]:
            assert species_id == "sp-1"
            return {
                "species": {"speciesId": "sp-1", "commonName": "California condor"},
                "speciesEntities": [],
            }

        async def list_species_documents(
            self, *, species_id: str, document_kinds: list[str] | None = None
        ) -> dict[str, Any]:
            del species_id, document_kinds
            return {
                "data": [
                    {
                        "documentKind": "federal_register",
                        "title": "Endangered Status for California Condor",
                        "url": "https://www.govinfo.gov/link/fr/32/4001",
                        "documentDate": "1967-03-11",
                        "frCitation": "32 FR 4001",
                        "documentType": "Final Rule",
                    }
                ]
            }

    class FakeFederalRegisterClient:
        async def search_documents(self, *, query: str, limit: int = 10, **kwargs: Any) -> dict[str, Any]:
            del query, limit, kwargs
            return {
                "total": 1,
                "data": [
                    {
                        "documentNumber": "2024-12345",
                        "title": "Critical Habitat Revision for California Condor",
                        "documentType": "RULE",
                        "publicationDate": "2024-02-01",
                        "citation": "89 FR 83510",
                        "htmlUrl": "https://www.federalregister.gov/d/2024-12345",
                        "cfrReferences": ["50 CFR 17.95"],
                    }
                ],
            }

    class FailingGovInfoClient:
        async def get_cfr_text(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            raise ValueError("No GovInfo CFR result matched 50 CFR 17.95.")

    _, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        ecos=FakeEcosClient(),
        federal_register=FakeFederalRegisterClient(),
        govinfo=FailingGovInfoClient(),
        enable_ecos=True,
        enable_federal_register=True,
        enable_govinfo_cfr=True,
    )

    payload = await runtime.search_papers_smart(
        query="What does 50 CFR 17.95 say about California condor critical habitat?",
        limit=5,
    )

    primary_document = payload["coverageSummary"]["primaryDocumentCoverage"]
    assert primary_document["currentTextRequested"] is True
    assert primary_document["govinfoAttempted"] is True
    assert primary_document["currentTextSatisfied"] is False
    assert primary_document["historyOnly"] is True
    assert any("Current codified CFR text was not verified from GovInfo" in gap for gap in payload["evidenceGaps"])
    assert payload["failureSummary"]["outcome"] == "fallback_success"


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

    caplog.set_level(logging.INFO, logger="paper-chaser-mcp")

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
    config = AgenticConfig(
        enabled=True,
        provider="openai",
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
        disable_embeddings=False,
    )
    provider_registry = ProviderDiagnosticsRegistry()
    bundle = OpenAIProviderBundle(
        config,
        api_key="sk-test",
        provider_registry=provider_registry,
    )
    registry, runtime = _runtime_with_provider_bundle(
        config=config,
        provider_bundle=bundle,
        semantic=semantic,
        openalex=openalex,
        provider_registry=provider_registry,
    )
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

    def _unexpected_similarity(query: str, text: str) -> float:
        raise AssertionError(f"balanced ask_result_set should not use workspace model similarity: {query!r}, {text!r}")

    async def _unexpected_async_similarity(query: str, texts: list[str]) -> list[float]:
        raise AssertionError(f"balanced ask_result_set should not use workspace async similarity: {query!r}, {texts!r}")

    bundle.aembed_texts = _unexpected_aembed_texts  # type: ignore[method-assign]

    registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        similarity_fn=_unexpected_similarity,
        async_batched_similarity_fn=_unexpected_async_similarity,
    )
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
async def test_ask_result_set_abstains_when_question_is_unsupported() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    record = registry.save_result_set(
        source_tool="search_papers_smart",
        payload={
            "data": [
                {
                    "paperId": "paper-1",
                    "title": "Materials Design with OpenFOAM",
                    "abstract": "A computational fluid dynamics paper unrelated to coding-agent evaluation.",
                    "source": "semantic_scholar",
                    "verificationStatus": "verified_metadata",
                }
            ]
        },
    )

    async def _unsupported_answer_question(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "answer": "Shared ground: these papers converge on Access, Agents, Allowing.",
            "unsupportedAsks": ["What evaluation tradeoffs show up here?"],
            "followUpQuestions": ["Would you like a narrower query?"],
            "confidence": "low",
        }

    async def _low_similarity(query: str, texts: list[str], **kwargs: object) -> list[float]:
        del query, texts, kwargs
        return [0.05]

    runtime._provider_bundle.aanswer_question = _unsupported_answer_question  # type: ignore[method-assign]
    runtime._deterministic_bundle.abatched_similarity = _low_similarity  # type: ignore[method-assign]

    ask = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question="What evaluation tradeoffs show up here?",
        top_k=1,
        answer_mode="synthesis",
    )

    assert ask["answerStatus"] == "abstained"
    assert ask["answer"] is None
    assert ask["unsupportedAsks"] == ["What evaluation tradeoffs show up here?"]
    assert ask["followUpQuestions"] == ["Would you like a narrower query?"]
    assert ask["evidenceGaps"]
    assert ask["candidateLeads"]
    assert ask["candidateLeads"][0]["sourceId"] == "paper-1"
    assert ask["candidateLeads"][0]["topicalRelevance"] == "off_topic"
    assert not any("Materials Design with OpenFOAM" in finding for finding in ask["verifiedFindings"])


@pytest.mark.asyncio
async def test_search_papers_smart_known_item_fallback_honors_provider_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    scholarapi = RecordingScholarApiClient()
    registry, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        scholarapi=scholarapi,
        enable_openalex=False,
        enable_scholarapi=True,
    )

    async def fake_classify_query(**kwargs: object) -> tuple[str, PlannerDecision]:
        query = str(kwargs["query"])
        return (
            query,
            PlannerDecision(
                intent="known_item",
                constraints={},
                seedIdentifiers=[],
                candidateConcepts=[],
                providerPlan=["scholarapi"],
                followUpMode="qa",
            ),
        )

    async def fake_resolve_citation(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {"bestMatch": None, "alternatives": [], "resolutionConfidence": "low"}

    async def fake_search_papers_match(**kwargs: object) -> dict[str, object]:
        semantic.calls.append(("search_papers_match", dict(kwargs)))
        return {"matchFound": False}

    monkeypatch.setattr("paper_chaser_mcp.agentic.graphs.classify_query", fake_classify_query)
    monkeypatch.setattr("paper_chaser_mcp.agentic.graphs.resolve_citation", fake_resolve_citation)
    semantic.search_papers_match = fake_search_papers_match  # type: ignore[method-assign]

    smart = await runtime.search_papers_smart(
        query="Graphene Full-Text Retrieval Workflows",
        limit=5,
        latency_profile="fast",
    )

    assert smart["results"]
    assert smart["results"][0]["paper"]["paperId"] == "ScholarAPI:sa-1"
    assert not [call for call in semantic.calls if call[0] == "search_papers"]
    assert scholarapi.calls and scholarapi.calls[0][0] == "search"
    assert smart["strategyMetadata"]["providersUsed"] == ["scholarapi"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_budget", "expected_reason", "expected_budget_field", "expected_budget_value"),
    [
        (
            {"allow_paid_providers": False},
            "disallows paid providers",
            "allowPaidProviders",
            False,
        ),
        (
            {"max_scholarapi_calls": 0},
            "per-provider limit for scholarapi",
            "maxScholarApiCalls",
            0,
        ),
    ],
)
async def test_search_papers_smart_known_item_fallback_honors_scholarapi_budget_controls(
    monkeypatch: pytest.MonkeyPatch,
    provider_budget: dict[str, object],
    expected_reason: str,
    expected_budget_field: str,
    expected_budget_value: object,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    scholarapi = RecordingScholarApiClient()
    registry, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        scholarapi=scholarapi,
        enable_semantic_scholar=False,
        enable_openalex=False,
        enable_scholarapi=True,
    )

    async def fake_resolve_citation(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {"bestMatch": None, "alternatives": [], "resolutionConfidence": "low"}

    async def fake_search_papers_match(**kwargs: object) -> dict[str, object]:
        semantic.calls.append(("search_papers_match", dict(kwargs)))
        return {"matchFound": False}

    monkeypatch.setattr("paper_chaser_mcp.agentic.graphs.resolve_citation", fake_resolve_citation)
    semantic.search_papers_match = fake_search_papers_match  # type: ignore[method-assign]

    smart = await runtime.search_papers_smart(
        query="Graphene Full-Text Retrieval Workflows",
        limit=5,
        mode="known_item",
        latency_profile="fast",
        provider_budget=provider_budget,
    )

    assert smart["results"] == []
    assert scholarapi.calls == []
    assert smart["strategyMetadata"]["providerBudgetApplied"][expected_budget_field] == expected_budget_value
    assert any(
        outcome["provider"] == "scholarapi"
        and outcome["statusBucket"] == "skipped"
        and expected_reason in str(outcome.get("fallbackReason") or "")
        for outcome in smart["strategyMetadata"]["providerOutcomes"]
    )


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
        disable_embeddings=False,
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

    caplog.set_level(logging.INFO, logger="paper-chaser-mcp")

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
async def test_search_papers_smart_reports_deterministic_active_provider_when_azure_falls_back() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    config = AgenticConfig(
        enabled=True,
        provider="azure-openai",
        planner_model="planner-deployment",
        synthesis_model="synthesis-deployment",
        embedding_model="embedding-deployment",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
    )
    bundle = AzureOpenAIProviderBundle(
        config,
        api_key=None,
        azure_endpoint=None,
        api_version=None,
    )
    _, runtime = _runtime_with_provider_bundle(
        config=config,
        provider_bundle=bundle,
        semantic=semantic,
        openalex=openalex,
    )

    payload = await runtime.search_papers_smart(
        query="transformers",
        limit=5,
        latency_profile="balanced",
    )

    assert payload["results"]
    assert payload["strategyMetadata"]["configuredSmartProvider"] == "azure-openai"
    assert payload["strategyMetadata"]["activeSmartProvider"] == "deterministic"
    assert payload["strategyMetadata"]["plannerModelSource"] == "deterministic"
    assert payload["strategyMetadata"]["synthesisModelSource"] == "deterministic"
    assert payload["strategyMetadata"]["plannerModel"] == "azure-openai:deterministic-planner"
    assert payload["strategyMetadata"]["synthesisModel"] == "azure-openai:deterministic-synthesizer"
    assert any("fell back to deterministic mode" in warning for warning in payload["strategyMetadata"]["driftWarnings"])
    assert any("fell back to deterministic mode" in warning for warning in payload["agentHints"]["warnings"])


@pytest.mark.asyncio
async def test_search_papers_smart_smoke_uses_azure_openai_provider_and_embeddings() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    config = AgenticConfig(
        enabled=True,
        provider="azure-openai",
        planner_model="planner-deployment",
        synthesis_model="synthesis-deployment",
        embedding_model="embedding-deployment",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
        disable_embeddings=False,
        planner_model_source="azure_deployment",
        synthesis_model_source="azure_deployment",
    )
    provider_registry = ProviderDiagnosticsRegistry()
    bundle = AzureOpenAIProviderBundle(
        config,
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-10-21",
        provider_registry=provider_registry,
    )

    class _ResponsesClient:
        def __init__(self) -> None:
            self.calls = 0

        async def parse(self, **kwargs: object) -> object:
            del kwargs
            self.calls += 1
            if self.calls == 1:
                return types.SimpleNamespace(
                    output_parsed={
                        "intent": "discovery",
                        "constraints": {},
                        "seedIdentifiers": [],
                        "candidateConcepts": ["transformers"],
                        "providerPlan": ["semantic_scholar", "openalex"],
                        "followUpMode": "qa",
                    }
                )
            return types.SimpleNamespace(output_parsed={"expansions": []})

    class _EmbeddingsClient:
        async def create(self, **kwargs: object) -> object:
            payload = kwargs["input"]
            if isinstance(payload, list):
                return {
                    "data": [{"embedding": [1.0, 0.0] if "transformers" in text else [0.0, 1.0]} for text in payload]
                }
            return {"data": [{"embedding": [1.0, 0.0]}]}

    class _AsyncClient:
        responses = _ResponsesClient()
        embeddings = _EmbeddingsClient()

    bundle._async_openai_client = _AsyncClient()
    bundle._openai_client = None
    _, runtime = _runtime_with_provider_bundle(
        config=config,
        provider_bundle=bundle,
        semantic=semantic,
        openalex=openalex,
        provider_registry=provider_registry,
    )

    payload = await runtime.search_papers_smart(
        query="transformers",
        limit=5,
        latency_profile="deep",
    )

    assert payload["results"]
    assert payload["strategyMetadata"]["configuredSmartProvider"] == "azure-openai"
    assert payload["strategyMetadata"]["activeSmartProvider"] == "azure-openai"
    assert payload["strategyMetadata"]["plannerModelSource"] == "azure_deployment"
    assert payload["strategyMetadata"]["synthesisModelSource"] == "azure_deployment"
    assert any(
        outcome["provider"] == "azure-openai"
        and outcome["endpoint"] == "responses.parse:planner"
        and outcome["statusBucket"] == "success"
        for outcome in payload["strategyMetadata"]["providerOutcomes"]
    )
    assert any(
        outcome["provider"] == "azure-openai"
        and outcome["endpoint"] == "embeddings.create"
        and outcome["statusBucket"] == "success"
        for outcome in payload["strategyMetadata"]["providerOutcomes"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "bundle_factory", "planner_model_name", "synthesis_model_name"),
    [
        (
            "anthropic",
            lambda config, registry: AnthropicProviderBundle(
                config,
                api_key="sk-ant-test",
                provider_registry=registry,
            ),
            "claude-haiku-4-5",
            "claude-sonnet-4-6",
        ),
        (
            "nvidia",
            lambda config, registry: NvidiaProviderBundle(
                config,
                api_key="nvidia-key",
                provider_registry=registry,
            ),
            "nvidia/nemotron-3-nano-30b-a3b",
            "nvidia/nemotron-3-super-120b-a12b",
        ),
        (
            "google",
            lambda config, registry: GoogleProviderBundle(
                config,
                api_key="google-key",
                provider_registry=registry,
            ),
            "gemini-2.5-flash",
            "gemini-2.5-flash",
        ),
    ],
)
async def test_search_papers_smart_smoke_langchain_providers_skip_embedding_rerank(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    bundle_factory: Any,
    planner_model_name: str,
    synthesis_model_name: str,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    config = AgenticConfig(
        enabled=True,
        provider=provider_name,
        planner_model=planner_model_name,
        synthesis_model=synthesis_model_name,
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
        disable_embeddings=False,
        planner_model_source="provider_default",
        synthesis_model_source="provider_default",
    )
    provider_registry = ProviderDiagnosticsRegistry()
    bundle = bundle_factory(config, provider_registry)
    planner = _SequenceModel(
        structured_responses=[
            {
                "intent": "discovery",
                "constraints": {},
                "seedIdentifiers": [],
                "candidateConcepts": ["transformers"],
                "providerPlan": ["semantic_scholar", "openalex"],
                "followUpMode": "qa",
            },
            {"expansions": []},
        ],
        text_responses=[],
    )
    synthesizer = _SequenceModel(structured_responses=[], text_responses=[])

    monkeypatch.setattr(bundle, "_load_models", lambda: (planner, synthesizer))

    async def _explode_similarity(*args: object, **kwargs: object) -> list[float]:
        raise AssertionError("deep rerank should not use chat-only provider embeddings")

    monkeypatch.setattr(bundle, "abatched_similarity", _explode_similarity)
    _, runtime = _runtime_with_provider_bundle(
        config=config,
        provider_bundle=bundle,
        semantic=semantic,
        openalex=openalex,
        provider_registry=provider_registry,
    )

    payload = await runtime.search_papers_smart(
        query="transformers",
        limit=5,
        latency_profile="deep",
    )

    assert payload["results"]
    assert payload["strategyMetadata"]["configuredSmartProvider"] == provider_name
    assert payload["strategyMetadata"]["activeSmartProvider"] == provider_name
    assert payload["strategyMetadata"]["plannerModelSource"] == "provider_default"
    assert payload["strategyMetadata"]["synthesisModelSource"] == "provider_default"
    assert any(
        outcome["provider"] == provider_name
        and outcome["endpoint"] == "structured:planner"
        and outcome["statusBucket"] == "success"
        for outcome in payload["strategyMetadata"]["providerOutcomes"]
    )
    assert not [
        outcome
        for outcome in payload["strategyMetadata"]["providerOutcomes"]
        if outcome["provider"] == provider_name and outcome["endpoint"] == "embeddings.create"
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
async def test_enrich_paper_adds_openalex_metadata_when_enabled() -> None:
    class OpenAlexClient:
        async def get_paper_details(self, paper_id: str) -> dict[str, object]:
            return {
                "paperId": paper_id,
                "source": "openalex",
                "sourceId": "W42",
                "canonicalId": "10.1234/example",
                "venue": "OpenAlex Venue",
                "publicationTypes": "journal-article",
                "publicationDate": "2024-06-01",
                "year": 2024,
                "url": "https://doi.org/10.1234/example",
                "pdfUrl": "https://openalex.example/file.pdf",
                "citationCount": 12,
            }

    service = PaperEnrichmentService(
        crossref_client=None,
        unpaywall_client=None,
        openalex_client=OpenAlexClient(),  # type: ignore[arg-type]
        enable_crossref=False,
        enable_unpaywall=False,
        enable_openalex=True,
    )

    response = await service.enrich_paper(doi="10.1234/example")

    assert response.openalex is not None
    assert response.openalex.found is True
    assert response.openalex.lookup_id == "10.1234/example"
    assert response.enrichments is not None
    assert response.enrichments.openalex is not None
    assert response.enrichments.openalex.source_id == "W42"
    assert response.enrichments.openalex.citation_count == 12
    assert response.crossref is not None
    assert response.crossref.found is False
    assert response.unpaywall is not None
    assert response.unpaywall.found is False
    assert response.doi_resolution.resolved_doi == "10.1234/example"


@pytest.mark.asyncio
async def test_enrich_paper_rejects_mismatched_openalex_doi() -> None:
    class MismatchedOpenAlexClient:
        async def get_paper_details(self, paper_id: str) -> dict[str, object]:
            return {
                "paperId": paper_id,
                "source": "openalex",
                "sourceId": "W99",
                "canonicalId": "10.9999/other-work",
                "url": "https://doi.org/10.9999/other-work",
                "publicationDate": "2025-01-01",
                "year": 2025,
            }

    service = PaperEnrichmentService(
        crossref_client=None,
        unpaywall_client=None,
        openalex_client=MismatchedOpenAlexClient(),  # type: ignore[arg-type]
        enable_crossref=False,
        enable_unpaywall=False,
        enable_openalex=True,
    )

    response = await service.enrich_paper(doi="10.1234/example")

    assert response.openalex is not None
    assert response.openalex.found is False
    assert response.openalex.enrichment is None
    assert response.doi_resolution.resolved_doi == "10.1234/example"
    assert response.enrichments is None


@pytest.mark.asyncio
async def test_enrich_paper_prefers_trusted_doi_over_openalex_mismatch() -> None:
    class MismatchedOpenAlexClient:
        async def get_paper_details(self, paper_id: str) -> dict[str, object]:
            return {
                "paperId": paper_id,
                "source": "openalex",
                "sourceId": "W100",
                "canonicalId": "10.9999/openalex-mismatch",
                "url": "https://doi.org/10.9999/openalex-mismatch",
                "publicationDate": "2025-01-01",
                "year": 2025,
            }

    service = PaperEnrichmentService(
        crossref_client=RecordingCrossrefClient(),
        unpaywall_client=RecordingUnpaywallClient(),
        openalex_client=MismatchedOpenAlexClient(),  # type: ignore[arg-type]
        enable_crossref=True,
        enable_unpaywall=True,
        enable_openalex=True,
    )

    response = await service.enrich_paper(doi="10.1234/example")

    assert response.crossref is not None
    assert response.crossref.found is True
    assert response.unpaywall is not None
    assert response.unpaywall.found is True
    assert response.openalex is not None
    assert response.openalex.found is False
    assert response.doi_resolution.resolved_doi == "10.1234/example"
    assert response.doi_resolution.resolution_source == "doi"


@pytest.mark.asyncio
async def test_enrich_paper_query_only_abstains_without_anchor() -> None:
    crossref = RecordingCrossrefClient()
    unpaywall = RecordingUnpaywallClient()
    openalex = RecordingOpenAlexClient()
    service = PaperEnrichmentService(
        crossref_client=crossref,
        unpaywall_client=unpaywall,
        openalex_client=openalex,
        enable_crossref=True,
        enable_unpaywall=True,
        enable_openalex=True,
    )

    response = await service.enrich_paper(query="Attention Is All You Need")

    assert crossref.calls == []
    assert unpaywall.calls == []
    assert openalex.calls == []
    assert response.crossref is not None
    assert response.crossref.found is False
    assert response.unpaywall is not None
    assert response.unpaywall.found is False
    assert response.openalex is not None
    assert response.openalex.found is False
    assert response.enrichments is None
    assert response.doi_resolution.resolved_doi is None


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
        "paper_chaser_mcp.agentic.graphs._graph_frontier_scores",
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

    monkeypatch.setattr("paper_chaser_mcp.agentic.graphs.resolve_citation", fake_resolve_citation)
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

    monkeypatch.setattr("paper_chaser_mcp.agentic.graphs.resolve_citation", fake_resolve_citation)
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
    assert initial["scholarapi"] > expansion["scholarapi"]
    assert expansion["semantic_scholar"] == 6
    assert expansion["openalex"] == 6
    assert expansion["scholarapi"] == 4
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


@pytest.mark.asyncio
async def test_retrieve_variant_includes_scholarapi_when_enabled() -> None:
    class _EmptyClient:
        async def search(self, **kwargs: object) -> dict:
            return {"data": []}

    scholarapi = RecordingScholarApiClient()

    batch = await retrieve_variant(
        variant="graphene full text retrieval",
        variant_source="from_input",
        intent="discovery",
        year="2024",
        venue=None,
        enable_core=False,
        enable_semantic_scholar=False,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        enable_scholarapi=True,
        core_client=_EmptyClient(),
        semantic_client=_EmptyClient(),
        openalex_client=_EmptyClient(),
        scholarapi_client=scholarapi,
        arxiv_client=_EmptyClient(),
        serpapi_client=None,
        provider_plan=["scholarapi"],
    )

    assert batch.providers_used == ["scholarapi"]
    assert batch.candidates
    assert batch.candidates[0].provider == "scholarapi"
    assert scholarapi.calls[0][0] == "search"


@pytest.mark.asyncio
async def test_search_papers_smart_can_use_scholarapi_when_enabled() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    class _SmartScholarApiClient(RecordingScholarApiClient):
        async def search(self, **kwargs: object) -> dict:
            self.calls.append(("search", dict(kwargs)))
            return {
                "provider": "scholarapi",
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "ScholarAPI:smart-1",
                        "title": "Graphene Full-Text Retrieval Workflows",
                        "abstract": "A paper about graphene retrieval workflows with accessible full text.",
                        "year": 2024,
                        "authors": [{"name": "Lead Author"}],
                        "source": "scholarapi",
                    }
                ],
            }

    scholarapi = _SmartScholarApiClient()
    registry, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        scholarapi=scholarapi,
        enable_semantic_scholar=False,
        enable_openalex=False,
        enable_scholarapi=True,
    )

    payload = await runtime.search_papers_smart(
        query="graphene full text retrieval",
        limit=5,
        latency_profile="fast",
        provider_budget={"max_scholarapi_calls": 1},
    )

    assert payload["results"]
    assert payload["strategyMetadata"]["providersUsed"] == ["scholarapi"]
    assert payload["strategyMetadata"]["paidProvidersUsed"] == ["scholarapi"]
    assert payload["strategyMetadata"]["configuredSmartProvider"] == "deterministic"
    assert payload["strategyMetadata"]["activeSmartProvider"] == "deterministic"
    assert payload["strategyMetadata"]["plannerModelSource"] == "deterministic"
    assert payload["strategyMetadata"]["synthesisModelSource"] == "deterministic"
    assert payload["strategyMetadata"]["providerBudgetApplied"]["maxScholarApiCalls"] == 1
    assert scholarapi.calls
