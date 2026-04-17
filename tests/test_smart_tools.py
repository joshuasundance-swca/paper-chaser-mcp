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
from paper_chaser_mcp.agentic.models import IntentCandidate, PlannerDecision
from paper_chaser_mcp.agentic.planner import (
    _estimate_ambiguity_level,
    _estimate_query_specificity,
    _top_evidence_phrases,
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
from paper_chaser_mcp.agentic.ranking import (
    merge_candidates,
    rerank_candidates,
    summarize_ranking_diagnostics,
)
from paper_chaser_mcp.agentic.retrieval import (
    RetrievedCandidate,
    provider_limits,
    retrieve_variant,
)
from paper_chaser_mcp.enrichment import PaperEnrichmentService
from paper_chaser_mcp.models.tools import (
    AskResultSetArgs,
    ExpandResearchGraphArgs,
    MapResearchLandscapeArgs,
    SmartSearchPapersArgs,
)
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
    assert smart["answerability"] == "limited"
    assert smart["routingSummary"]["intent"] == "discovery"
    assert smart["leads"]
    assert smart["leads"] == smart["candidateLeads"]
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
    assert ask["providerUsed"] == "deterministic"
    if ask["answerStatus"] == "answered":
        assert ask["answer"]
        assert ask["answerability"] == "limited"
        assert ask["degradationReason"] == "deterministic_synthesis_fallback"
        assert any("deterministic_synthesis_fallback" in warning for warning in ask["agentHints"]["warnings"])
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
            # Finding 4 (4th rubber-duck pass): the dispatcher now probes every
            # ``_ecos_query_variants`` entry instead of breaking on the first
            # hit, so this fake has to accept regex-extracted subqueries too.
            # Return the California condor match for variants that plausibly
            # describe it (or carry the "regulatory history" subject terms);
            # return an empty payload otherwise.
            lowered = query.lower()
            relevant = (
                "regulatory history" in lowered
                or "california condor" in lowered
                or "condor" in lowered
            )
            if not relevant:
                return {"query": query, "matchMode": "auto", "total": 0, "data": []}
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
    assert payload["strategyMetadata"]["anchorType"] == "cfr_citation"
    assert payload["strategyMetadata"]["anchorStrength"] == "high"
    assert payload["strategyMetadata"]["anchoredSubject"]
    assert payload["strategyMetadata"]["bestNextInternalAction"] == "inspect_source"
    assert payload["structuredSources"]
    assert any(source.get("sourceAlias") for source in payload["structuredSources"])
    assert payload["verifiedFindings"]
    assert payload["resultStatus"] == "succeeded"
    assert payload["hasInspectableSources"] is True
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
    assert all("leadReason" in lead for lead in payload["candidateLeads"])
    assert all("whyNotVerified" in lead for lead in payload["candidateLeads"])


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
async def test_search_papers_smart_regulatory_uses_govinfo_federal_register_fallback() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    class EmptyEcosClient:
        async def search_species(self, *, query: str, limit: int = 10, match_mode: str = "auto") -> dict[str, Any]:
            del query, limit, match_mode
            return {"data": []}

    class EmptyFederalRegisterClient:
        async def search_documents(self, *, query: str, limit: int = 10, **kwargs: Any) -> dict[str, Any]:
            del query, limit, kwargs
            return {"total": 0, "data": []}

    class GovInfoSearchClient:
        async def search_federal_register_documents(self, *, query: str, limit: int = 10) -> dict[str, Any]:
            del limit
            assert "northern long-eared bat" in query.lower()
            return {
                "total": 1,
                "data": [
                    {
                        "title": "Endangered Species Status for the Northern Long-Eared Bat",
                        "documentNumber": "2022-25998",
                        "citation": "87 FR 73488",
                        "publicationDate": "2022-11-30",
                        "sourceUrl": "https://www.govinfo.gov/app/details/FR-2022-11-30/2022-25998",
                        "verificationStatus": "verified_metadata",
                        "note": "GovInfo Federal Register primary-source recovery hit.",
                    }
                ],
            }

    _, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        ecos=EmptyEcosClient(),
        federal_register=EmptyFederalRegisterClient(),
        govinfo=GovInfoSearchClient(),
        enable_ecos=True,
        enable_federal_register=True,
        enable_govinfo_cfr=True,
    )

    payload = await runtime.search_papers_smart(
        query="Northern long-eared bat ESA listing status and final rule history",
        limit=5,
    )

    assert payload["strategyMetadata"]["intent"] == "regulatory"
    assert payload["strategyMetadata"]["intentSource"] == "planner"
    assert payload["strategyMetadata"]["intentConfidence"] == "medium"
    assert payload["strategyMetadata"]["anchorType"] == "regulatory_subject_terms"
    assert payload["strategyMetadata"]["recoveryAttempted"] is False
    assert "govinfo" in payload["coverageSummary"]["providersAttempted"]
    assert "govinfo" in payload["coverageSummary"]["providersSucceeded"]
    assert any(source["provider"] == "govinfo" for source in payload["structuredSources"])
    assert any(source.get("sourceAlias") for source in payload["structuredSources"])
    assert payload["verifiedFindings"]


@pytest.mark.asyncio
async def test_search_papers_smart_recovers_known_item_when_forced_regulatory_mode_returns_nothing() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    exact_title = (
        "Ecosystem experiment reveals benefits of natural and simulated beaver dams to a threatened "
        "population of steelhead (Oncorhynchus mykiss)"
    )

    async def semantic_match(**kwargs: Any) -> dict[str, Any]:
        semantic.calls.append(("search_papers_match", dict(kwargs)))
        return {
            "paperId": "1539acae748ff423bf4ebd2da0d95933e94f59ee",
            "title": exact_title,
            "year": 2016,
            "authors": [{"name": "N. Bouwes"}],
            "venue": "Scientific Reports",
            "source": "semantic_scholar",
            "matchStrategy": "semantic_title_match",
        }

    semantic.search_papers_match = semantic_match  # type: ignore[method-assign]

    class EmptyEcosClient:
        async def search_species(self, *, query: str, limit: int = 10, match_mode: str = "auto") -> dict[str, Any]:
            del query, limit, match_mode
            return {"data": []}

    class EmptyFederalRegisterClient:
        async def search_documents(self, *, query: str, limit: int = 10, **kwargs: Any) -> dict[str, Any]:
            del query, limit, kwargs
            return {"total": 0, "data": []}

    class EmptyGovInfoClient:
        async def search_federal_register_documents(self, *, query: str, limit: int = 10) -> dict[str, Any]:
            del query, limit
            return {"total": 0, "data": []}

    _, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        ecos=EmptyEcosClient(),
        federal_register=EmptyFederalRegisterClient(),
        govinfo=EmptyGovInfoClient(),
        enable_ecos=True,
        enable_federal_register=True,
        enable_govinfo_cfr=True,
    )

    # Simulate an LLM that correctly identifies this as a paper title lookup
    # and suggests known_item recovery.  The DeterministicProviderBundle fallback
    # cannot reliably detect all long scientific titles, so the test verifies the
    # recovery *mechanism* works correctly when the strategy revision returns known_item.
    async def _known_item_revision(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "revisedQuery": exact_title,
            "revisedIntent": "known_item",
            "revisedProviders": ["semantic_scholar"],
            "rationale": "Query looks like a paper title; retried with semantic known-item recovery.",
        }

    runtime._provider_bundle.arevise_search_strategy = _known_item_revision  # type: ignore[method-assign]
    runtime._deterministic_bundle.arevise_search_strategy = _known_item_revision  # type: ignore[method-assign]

    payload = await runtime.search_papers_smart(
        query=exact_title,
        limit=5,
        mode="regulatory",
    )

    assert payload["strategyMetadata"]["intent"] == "known_item"
    assert payload["strategyMetadata"]["intentSource"] == "fallback_recovery"
    assert payload["strategyMetadata"]["recoveryAttempted"] is True
    assert payload["strategyMetadata"]["recoveryPath"] == ["regulatory", "known_item"]
    assert payload["strategyMetadata"]["bestNextInternalAction"] == "get_paper_details"
    assert payload["results"][0]["paper"]["title"] == exact_title
    assert payload["resultStatus"] == "succeeded"
    assert payload["hasInspectableSources"] is True
    assert any("known-item recovery" in warning.lower() for warning in payload["strategyMetadata"]["driftWarnings"])


@pytest.mark.asyncio
async def test_search_papers_smart_low_result_recovery_uses_revised_strategy() -> None:
    exact_title = "Attention Is All You Need"

    class SparseSemanticClient(RecordingSemanticClient):
        async def search_papers(self, **kwargs: Any) -> dict[str, Any]:
            query = str(kwargs.get("query") or "")
            self.calls.append(("search_papers", dict(kwargs)))
            if query == "agent routing benchmark":
                return {"total": 0, "offset": 0, "data": []}
            return {"total": 1, "offset": 0, "data": []}

    semantic = SparseSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    async def _revise(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "revisedQuery": exact_title,
            "revisedIntent": "known_item",
            "revisedProviders": ["semantic_scholar"],
            "rationale": "Retry as an exact-title lookup.",
        }

    async def _match(**kwargs: Any) -> dict[str, Any]:
        semantic.calls.append(("search_papers_match", dict(kwargs)))
        return {
            "paperId": "paper-attention",
            "title": exact_title,
            "year": 2017,
            "authors": [{"name": "Ashish Vaswani"}],
            "venue": "NeurIPS",
            "source": "semantic_scholar",
            "matchStrategy": "semantic_title_match",
        }

    runtime._provider_bundle.arevise_search_strategy = _revise  # type: ignore[method-assign]
    runtime._deterministic_bundle.arevise_search_strategy = _revise  # type: ignore[method-assign]
    semantic.search_papers_match = _match  # type: ignore[method-assign]

    payload = await runtime.search_papers_smart(
        query="agent routing benchmark",
        limit=5,
        mode="auto",
    )

    assert payload["resultStatus"] == "succeeded"
    assert payload["results"][0]["paper"]["title"] == exact_title
    assert payload["strategyMetadata"]["recoveryAttempted"] is True
    assert payload["strategyMetadata"]["recoveryPath"] == ["discovery", "known_item"]


@pytest.mark.asyncio
async def test_search_papers_smart_regulatory_ecos_common_name_query_uses_species_anchor_variant() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    class CommonNameOnlyEcosClient:
        async def search_species(self, *, query: str, limit: int = 10, match_mode: str = "auto") -> dict[str, Any]:
            del limit, match_mode
            if query.strip().lower() != "northern long-eared bat":
                return {"query": query, "total": 0, "data": []}
            return {
                "query": query,
                "total": 1,
                "data": [
                    {
                        "speciesId": "sp-nleb",
                        "commonName": "Northern long-eared bat",
                        "scientificName": "Myotis septentrionalis",
                        "profileUrl": "https://ecos.fws.gov/ecp/species/sp-nleb",
                    }
                ],
            }

        async def get_species_profile(self, *, species_id: str) -> dict[str, Any]:
            assert species_id == "sp-nleb"
            return {
                "species": {
                    "speciesId": "sp-nleb",
                    "commonName": "Northern long-eared bat",
                    "scientificName": "Myotis septentrionalis",
                    "profileUrl": "https://ecos.fws.gov/ecp/species/sp-nleb",
                },
                "speciesEntities": [],
            }

        async def list_species_documents(
            self, *, species_id: str, document_kinds: list[str] | None = None
        ) -> dict[str, Any]:
            del document_kinds
            assert species_id == "sp-nleb"
            return {
                "speciesId": species_id,
                "total": 1,
                "data": [
                    {
                        "documentKind": "federal_register",
                        "title": "Endangered Species Status for the Northern Long-Eared Bat",
                        "url": "https://www.federalregister.gov/d/2022-25998",
                        "documentDate": "2022-11-30",
                        "frCitation": "87 FR 73488",
                        "documentType": "Final Rule",
                    }
                ],
            }

    _, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        ecos=CommonNameOnlyEcosClient(),
        federal_register=None,
        govinfo=None,
        enable_ecos=True,
        enable_federal_register=False,
        enable_govinfo_cfr=False,
    )

    payload = await runtime.search_papers_smart(
        query="northern long-eared bat ECOS species profile",
        limit=5,
    )

    assert payload["strategyMetadata"]["intent"] == "regulatory"
    assert payload["strategyMetadata"]["anchorType"] == "species_common_name"
    assert payload["strategyMetadata"]["anchoredSubject"] == "Northern long-eared bat"
    assert "No ECOS species dossier match was found for the query." not in payload["evidenceGaps"]
    assert any(source["provider"] == "ecos" for source in payload["structuredSources"])


@pytest.mark.asyncio
async def test_search_papers_smart_regulatory_ecos_scientific_name_query_uses_species_anchor_variant() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    class ScientificNameOnlyEcosClient:
        async def search_species(self, *, query: str, limit: int = 10, match_mode: str = "auto") -> dict[str, Any]:
            del limit, match_mode
            if query.strip() != "Myotis septentrionalis":
                return {"query": query, "total": 0, "data": []}
            return {
                "query": query,
                "total": 1,
                "data": [
                    {
                        "speciesId": "sp-myotis",
                        "commonName": "Northern long-eared bat",
                        "scientificName": "Myotis septentrionalis",
                        "profileUrl": "https://ecos.fws.gov/ecp/species/sp-myotis",
                    }
                ],
            }

        async def get_species_profile(self, *, species_id: str) -> dict[str, Any]:
            assert species_id == "sp-myotis"
            return {
                "species": {
                    "speciesId": "sp-myotis",
                    "commonName": "Northern long-eared bat",
                    "scientificName": "Myotis septentrionalis",
                    "profileUrl": "https://ecos.fws.gov/ecp/species/sp-myotis",
                },
                "speciesEntities": [],
            }

        async def list_species_documents(
            self, *, species_id: str, document_kinds: list[str] | None = None
        ) -> dict[str, Any]:
            del document_kinds
            assert species_id == "sp-myotis"
            return {"speciesId": species_id, "total": 0, "data": []}

    _, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        ecos=ScientificNameOnlyEcosClient(),
        federal_register=None,
        govinfo=None,
        enable_ecos=True,
        enable_federal_register=False,
        enable_govinfo_cfr=False,
    )

    payload = await runtime.search_papers_smart(
        query="Myotis septentrionalis endangered species profile ECOS",
        limit=5,
    )

    assert payload["strategyMetadata"]["intent"] == "regulatory"
    assert payload["strategyMetadata"]["anchorType"] == "species_scientific_name"
    assert payload["strategyMetadata"]["anchoredSubject"] == "Northern long-eared bat"
    assert "No ECOS species dossier match was found for the query." not in payload["evidenceGaps"]
    assert any(source["provider"] == "ecos" for source in payload["structuredSources"])


@pytest.mark.asyncio
async def test_search_papers_smart_regulatory_cfr_current_text_demotes_history_to_leads() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    class FakeEcosClient:
        async def search_species(self, *, query: str, limit: int = 10, match_mode: str = "auto") -> dict[str, Any]:
            del query, limit, match_mode
            return {"total": 0, "data": []}

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
        query="What does 50 CFR 17.95 say about California condor critical habitat?",
        limit=5,
    )

    assert any(source["provider"] == "govinfo" for source in payload["structuredSources"])
    assert not any(source["provider"] == "federal_register" for source in payload["structuredSources"])
    assert any(lead["provider"] == "federal_register" for lead in payload["candidateLeads"])


@pytest.mark.asyncio
async def test_search_papers_smart_regulatory_agency_guidance_routes_authority_first_without_ecos() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    class FailingEcosClient:
        async def search_species(self, *, query: str, limit: int = 10, match_mode: str = "auto") -> dict[str, Any]:
            raise AssertionError(
                f"ECOS should not be queried for FDA guidance title lookups: {query}, {limit}, {match_mode}"
            )

    class FakeFederalRegisterClient:
        async def search_documents(self, *, query: str, limit: int = 10, **kwargs: Any) -> dict[str, Any]:
            del limit, kwargs
            assert "clinical decision support software guidance" in query.lower()
            return {
                "total": 1,
                "data": [
                    {
                        "documentNumber": "2022-11111",
                        "title": "Clinical Decision Support Software Guidance",
                        "documentType": "NOTICE",
                        "publicationDate": "2022-09-28",
                        "citation": "87 FR 59000",
                        "htmlUrl": "https://www.federalregister.gov/d/2022-11111",
                    }
                ],
            }

    class FakeGovInfoClient:
        async def search_federal_register_documents(self, *, query: str, limit: int = 10) -> dict[str, Any]:
            del limit
            assert "clinical decision support software guidance" in query.lower()
            return {
                "total": 1,
                "data": [
                    {
                        "title": "Clinical Decision Support Software Guidance for Industry and FDA Staff",
                        "documentNumber": "2022-11111",
                        "citation": "87 FR 59000",
                        "publicationDate": "2022-09-28",
                        "sourceUrl": "https://www.govinfo.gov/app/details/FR-2022-09-28/2022-11111",
                        "verificationStatus": "verified_metadata",
                        "note": "GovInfo regulatory guidance recovery hit.",
                    }
                ],
            }

    _, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        ecos=FailingEcosClient(),
        federal_register=FakeFederalRegisterClient(),
        govinfo=FakeGovInfoClient(),
        enable_ecos=True,
        enable_federal_register=True,
        enable_govinfo_cfr=True,
    )

    payload = await runtime.search_papers_smart(
        query="Clinical Decision Support Software Guidance for Industry and Food and Drug Administration Staff",
        limit=5,
    )

    assert payload["strategyMetadata"]["intent"] == "regulatory"
    assert payload["strategyMetadata"]["anchorType"] == "agency_guidance_title"
    assert "ecos" not in payload["coverageSummary"]["providersAttempted"]
    assert "govinfo" in payload["coverageSummary"]["providersAttempted"]
    assert any(source["provider"] == "govinfo" for source in payload["structuredSources"])


@pytest.mark.asyncio
async def test_search_papers_smart_broad_agency_guidance_ranks_lifecycle_documents_ahead_of_off_target_hits() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    class FailingEcosClient:
        async def search_species(self, *, query: str, limit: int = 10, match_mode: str = "auto") -> dict[str, Any]:
            raise AssertionError(
                f"ECOS should not be queried for broad FDA guidance discovery: {query}, {limit}, {match_mode}"
            )

    class FakeFederalRegisterClient:
        async def search_documents(self, *, query: str, limit: int = 10, **kwargs: Any) -> dict[str, Any]:
            del limit, kwargs
            assert "fda" in query.lower()
            return {
                "total": 3,
                "data": [
                    {
                        "documentNumber": "2024-11001",
                        "title": (
                            "Marketing Submission Recommendations for a Predetermined Change Control Plan for "
                            "Artificial Intelligence-Enabled Device Software Functions"
                        ),
                        "documentType": "NOTICE",
                        "publicationDate": "2024-12-04",
                        "citation": "89 FR 96645",
                        "htmlUrl": "https://www.federalregister.gov/d/2024-11001",
                    },
                    {
                        "documentNumber": "2019-00123",
                        "title": (
                            "Proposed Regulatory Framework for Modifications to Artificial Intelligence/"
                            "Machine Learning-Based Software as a Medical Device"
                        ),
                        "documentType": "NOTICE",
                        "publicationDate": "2019-04-02",
                        "citation": "84 FR 12789",
                        "htmlUrl": "https://www.federalregister.gov/d/2019-00123",
                    },
                    {
                        "documentNumber": "2022-11111",
                        "title": "Clinical Decision Support Software Guidance for Industry and FDA Staff",
                        "documentType": "NOTICE",
                        "publicationDate": "2022-09-28",
                        "citation": "87 FR 59000",
                        "htmlUrl": "https://www.federalregister.gov/d/2022-11111",
                    },
                ],
            }

    class FakeGovInfoClient:
        async def search_federal_register_documents(self, *, query: str, limit: int = 10) -> dict[str, Any]:
            del limit
            assert "fda" in query.lower()
            return {
                "total": 3,
                "data": [
                    {
                        "title": (
                            "Marketing Submission Recommendations for a Predetermined Change Control Plan for "
                            "Artificial Intelligence-Enabled Device Software Functions"
                        ),
                        "documentNumber": "2024-11001",
                        "citation": "89 FR 96645",
                        "publicationDate": "2024-12-04",
                        "sourceUrl": "https://www.govinfo.gov/app/details/FR-2024-12-04/2024-11001",
                        "verificationStatus": "verified_metadata",
                        "note": "GovInfo agency guidance hit.",
                    },
                    {
                        "title": (
                            "Proposed Regulatory Framework for Modifications to Artificial Intelligence/"
                            "Machine Learning-Based Software as a Medical Device"
                        ),
                        "documentNumber": "2019-00123",
                        "citation": "84 FR 12789",
                        "publicationDate": "2019-04-02",
                        "sourceUrl": "https://www.govinfo.gov/app/details/FR-2019-04-02/2019-00123",
                        "verificationStatus": "verified_metadata",
                        "note": "GovInfo agency guidance hit.",
                    },
                    {
                        "title": "Clinical Decision Support Software Guidance for Industry and FDA Staff",
                        "documentNumber": "2022-11111",
                        "citation": "87 FR 59000",
                        "publicationDate": "2022-09-28",
                        "sourceUrl": "https://www.govinfo.gov/app/details/FR-2022-09-28/2022-11111",
                        "verificationStatus": "verified_metadata",
                        "note": "GovInfo agency guidance hit.",
                    },
                ],
            }

    _, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        ecos=FailingEcosClient(),
        federal_register=FakeFederalRegisterClient(),
        govinfo=FakeGovInfoClient(),
        enable_ecos=True,
        enable_federal_register=True,
        enable_govinfo_cfr=True,
    )

    payload = await runtime.search_papers_smart(
        query=(
            "What recent FDA guidance or discussion documents are most relevant to lifecycle management of "
            "AI or machine-learning-enabled medical devices?"
        ),
        limit=5,
    )

    titles = [source["title"] for source in payload["structuredSources"]]
    lead_titles = [lead["title"] for lead in payload["candidateLeads"]]

    assert payload["strategyMetadata"]["intent"] == "regulatory"
    assert payload["strategyMetadata"]["anchorType"] == "agency_guidance_title"
    assert "ecos" not in payload["coverageSummary"]["providersAttempted"]
    assert titles[0].startswith("Marketing Submission Recommendations for a Predetermined Change Control Plan")
    assert any("Proposed Regulatory Framework" in title for title in titles)
    assert "Clinical Decision Support Software Guidance for Industry and FDA Staff" not in titles
    assert "Clinical Decision Support Software Guidance for Industry and FDA Staff" in lead_titles


@pytest.mark.asyncio
async def test_search_papers_smart_broad_epa_guidance_uses_generic_anchor_scoring() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    class FailingEcosClient:
        async def search_species(self, *, query: str, limit: int = 10, match_mode: str = "auto") -> dict[str, Any]:
            raise AssertionError(
                f"ECOS should not be queried for broad EPA guidance discovery: {query}, {limit}, {match_mode}"
            )

    class FakeFederalRegisterClient:
        async def search_documents(self, *, query: str, limit: int = 10, **kwargs: Any) -> dict[str, Any]:
            del limit, kwargs
            assert "epa" in query.lower()
            return {
                "total": 3,
                "data": [
                    {
                        "documentNumber": "2024-31001",
                        "title": "Drinking Water Health Advisories for PFAS and PFOA",
                        "documentType": "NOTICE",
                        "publicationDate": "2024-05-01",
                        "citation": "89 FR 31001",
                        "htmlUrl": "https://www.federalregister.gov/d/2024-31001",
                    },
                    {
                        "documentNumber": "2023-28002",
                        "title": "PFAS Strategic Roadmap Policy Update",
                        "documentType": "NOTICE",
                        "publicationDate": "2023-06-15",
                        "citation": "88 FR 28002",
                        "htmlUrl": "https://www.federalregister.gov/d/2023-28002",
                    },
                    {
                        "documentNumber": "2022-14003",
                        "title": "Air Emissions Guidance for Stationary Sources",
                        "documentType": "NOTICE",
                        "publicationDate": "2022-02-03",
                        "citation": "87 FR 14003",
                        "htmlUrl": "https://www.federalregister.gov/d/2022-14003",
                    },
                ],
            }

    class FakeGovInfoClient:
        async def search_federal_register_documents(self, *, query: str, limit: int = 10) -> dict[str, Any]:
            del limit
            assert "epa" in query.lower()
            return {
                "total": 3,
                "data": [
                    {
                        "title": "Drinking Water Health Advisories for PFAS and PFOA",
                        "documentNumber": "2024-31001",
                        "citation": "89 FR 31001",
                        "publicationDate": "2024-05-01",
                        "sourceUrl": "https://www.govinfo.gov/app/details/FR-2024-05-01/2024-31001",
                        "verificationStatus": "verified_metadata",
                        "note": "GovInfo agency guidance hit.",
                    },
                    {
                        "title": "PFAS Strategic Roadmap Policy Update",
                        "documentNumber": "2023-28002",
                        "citation": "88 FR 28002",
                        "publicationDate": "2023-06-15",
                        "sourceUrl": "https://www.govinfo.gov/app/details/FR-2023-06-15/2023-28002",
                        "verificationStatus": "verified_metadata",
                        "note": "GovInfo agency guidance hit.",
                    },
                    {
                        "title": "Air Emissions Guidance for Stationary Sources",
                        "documentNumber": "2022-14003",
                        "citation": "87 FR 14003",
                        "publicationDate": "2022-02-03",
                        "sourceUrl": "https://www.govinfo.gov/app/details/FR-2022-02-03/2022-14003",
                        "verificationStatus": "verified_metadata",
                        "note": "GovInfo agency guidance hit.",
                    },
                ],
            }

    _, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        ecos=FailingEcosClient(),
        federal_register=FakeFederalRegisterClient(),
        govinfo=FakeGovInfoClient(),
        enable_ecos=True,
        enable_federal_register=True,
        enable_govinfo_cfr=True,
    )

    payload = await runtime.search_papers_smart(
        query="What recent EPA guidance or policy documents are most relevant to PFAS in drinking water?",
        limit=5,
    )

    titles = [source["title"] for source in payload["structuredSources"]]
    lead_titles = [lead["title"] for lead in payload["candidateLeads"]]

    assert payload["strategyMetadata"]["intent"] == "regulatory"
    assert payload["strategyMetadata"]["anchorType"] == "agency_guidance_title"
    assert "ecos" not in payload["coverageSummary"]["providersAttempted"]
    assert titles[0] == "Drinking Water Health Advisories for PFAS and PFOA"
    assert "PFAS Strategic Roadmap Policy Update" in titles
    assert "Air Emissions Guidance for Stationary Sources" not in titles
    assert "Air Emissions Guidance for Stationary Sources" in lead_titles


@pytest.mark.asyncio
async def test_search_papers_smart_broad_pfas_discovery_does_not_route_to_known_item() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    async def semantic_search(**kwargs: object) -> dict[str, Any]:
        semantic.calls.append(("search_papers", dict(kwargs)))
        return {
            "total": 1,
            "offset": 0,
            "data": [
                {
                    "paperId": "pfas-remediation-review",
                    "title": "Field-deployable PFAS remediation methods for soils and groundwater",
                    "abstract": "Compares adsorption, soil washing, stabilization, and thermal treatment.",
                    "year": 2024,
                    "source": "semantic_scholar",
                }
            ],
        }

    async def empty_openalex_search(**kwargs: object) -> dict[str, Any]:
        openalex.calls.append(("search", dict(kwargs)))
        return {"total": 0, "offset": 0, "data": []}

    semantic.search_papers = semantic_search  # type: ignore[method-assign]
    openalex.search = empty_openalex_search  # type: ignore[method-assign]

    _, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    payload = await runtime.search_papers_smart(
        query=(
            "What is the current evidence on PFAS remediation in soils and groundwater, especially for "
            "field-deployable methods?"
        ),
        limit=5,
    )

    assert payload["strategyMetadata"]["intent"] in {"discovery", "review"}
    assert payload["strategyMetadata"]["intent"] != "known_item"
    assert payload["strategyMetadata"]["querySpecificity"] == "low"
    assert payload["strategyMetadata"]["ambiguityLevel"] in {"medium", "high"}
    assert payload["strategyMetadata"]["retrievalHypotheses"]
    assert len(payload["strategyMetadata"]["queryVariantsTried"]) >= 2
    assert payload["results"]


@pytest.mark.asyncio
async def test_search_papers_smart_regulatory_pfas_filters_collateral_rules_from_grounded_evidence() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    class FakeFederalRegisterClient:
        async def search_documents(self, *, query: str, limit: int = 10, **kwargs: Any) -> dict[str, Any]:
            del limit, kwargs
            assert "pfas" in query.lower()
            return {
                "total": 3,
                "data": [
                    {
                        "documentNumber": "2024-22013",
                        "title": "Designation of PFOA and PFOS as Hazardous Substances",
                        "documentType": "RULE",
                        "publicationDate": "2024-05-08",
                        "citation": "89 FR 39214",
                        "htmlUrl": "https://www.federalregister.gov/d/2024-22013",
                        "abstract": "PFAS hazardous-substances action under CERCLA.",
                    },
                    {
                        "documentNumber": "2024-12001",
                        "title": "National Primary Drinking Water Regulations for Lead and Copper",
                        "documentType": "RULE",
                        "publicationDate": "2024-02-01",
                        "citation": "89 FR 12001",
                        "htmlUrl": "https://www.federalregister.gov/d/2024-12001",
                        "abstract": "Updates lead and copper requirements for drinking water systems.",
                    },
                    {
                        "documentNumber": "2024-13002",
                        "title": "Vessel Incidental Discharge National Standards of Performance",
                        "documentType": "RULE",
                        "publicationDate": "2024-03-01",
                        "citation": "89 FR 13002",
                        "htmlUrl": "https://www.federalregister.gov/d/2024-13002",
                        "abstract": "Discharge standards for vessels.",
                    },
                ],
            }

    _, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        federal_register=FakeFederalRegisterClient(),
        enable_federal_register=True,
        enable_govinfo_cfr=False,
        enable_ecos=False,
    )

    payload = await runtime.search_papers_smart(
        query="What recent U.S. federal regulatory actions address PFAS in drinking water and hazardous substances?",
        limit=5,
        mode="regulatory",
    )

    titles = [source["title"] for source in payload["structuredSources"]]
    lead_titles = [lead["title"] for lead in payload["candidateLeads"]]

    assert "Designation of PFOA and PFOS as Hazardous Substances" in titles
    assert "National Primary Drinking Water Regulations for Lead and Copper" not in titles
    assert "Vessel Incidental Discharge National Standards of Performance" not in titles
    assert "National Primary Drinking Water Regulations for Lead and Copper" in lead_titles
    assert "Vessel Incidental Discharge National Standards of Performance" in lead_titles
    assert payload["strategyMetadata"]["retrievalHypotheses"]


@pytest.mark.asyncio
async def test_search_papers_smart_promotes_borderline_weak_match_with_relevance_batch() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    async def semantic_search(**kwargs: object) -> dict[str, Any]:
        semantic.calls.append(("search_papers", dict(kwargs)))
        return {
            "total": 1,
            "offset": 0,
            "data": [
                {
                    "paperId": "paper-microplastic-thresholds",
                    "title": "Microplastic Effect Thresholds for Freshwater Benthic Macroinvertebrates",
                    "abstract": (
                        "Synthesizes threshold evidence for microplastic exposure in freshwater benthic "
                        "macroinvertebrates across river and lake systems."
                    ),
                    "year": 2024,
                    "source": "semantic_scholar",
                }
            ],
        }

    async def empty_openalex_search(**kwargs: object) -> dict[str, Any]:
        openalex.calls.append(("search", dict(kwargs)))
        return {"total": 0, "offset": 0, "data": []}

    semantic.search_papers = semantic_search  # type: ignore[method-assign]
    openalex.search = empty_openalex_search  # type: ignore[method-assign]

    _, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    relevance_calls = 0

    async def _relevance_batch(**kwargs: object) -> dict[str, dict[str, str]]:
        nonlocal relevance_calls
        relevance_calls += 1
        del kwargs
        return {
            "paper-microplastic-thresholds": {
                "classification": "on_topic",
                "rationale": (
                    "The title and abstract directly match the user's requested microplastics, freshwater, and "
                    "benthic macroinvertebrate focus."
                ),
            }
        }

    runtime._provider_bundle.aclassify_relevance_batch = _relevance_batch  # type: ignore[method-assign]
    runtime._deterministic_bundle.aclassify_relevance_batch = _relevance_batch  # type: ignore[method-assign]

    payload = await runtime.search_papers_smart(
        query="microplastic effects in freshwater benthic macroinvertebrates",
        limit=5,
    )

    assert relevance_calls == 1
    assert payload["results"][0]["topicalRelevance"] == "on_topic"
    assert payload["structuredSources"][0]["topicalRelevance"] == "on_topic"


@pytest.mark.asyncio
async def test_search_papers_smart_initial_retrieval_blends_focus_for_broad_environmental_query() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    async def semantic_search(**kwargs: object) -> dict[str, Any]:
        semantic.calls.append(("search_papers", dict(kwargs)))
        return {
            "total": 1,
            "offset": 0,
            "data": [
                {
                    "paperId": "wetland-1",
                    "title": "Coastal marsh wetland restoration and climate resilience",
                    "year": 2024,
                    "source": "semantic_scholar",
                }
            ],
        }

    async def empty_openalex_search(**kwargs: object) -> dict[str, Any]:
        openalex.calls.append(("search", dict(kwargs)))
        return {"total": 0, "offset": 0, "data": []}

    semantic.search_papers = semantic_search  # type: ignore[method-assign]
    openalex.search = empty_openalex_search  # type: ignore[method-assign]

    _, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    await runtime.search_papers_smart(
        query="What are the most effective wetland restoration strategies for improving climate resilience?",
        focus="prioritize coastal marsh field studies and management guidance relevant to restoration practitioners",
        limit=5,
    )

    first_query = str(semantic.calls[0][1].get("query") or "")
    assert "coastal" in first_query.lower()
    assert "marsh" in first_query.lower()


@pytest.mark.asyncio
async def test_search_papers_smart_initial_retrieval_blends_focus_for_broad_fire_management_query() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    async def semantic_search(**kwargs: object) -> dict[str, Any]:
        semantic.calls.append(("search_papers", dict(kwargs)))
        return {
            "total": 1,
            "offset": 0,
            "data": [
                {
                    "paperId": "fire-1",
                    "title": "Prescribed fire and mechanical thinning for wildfire risk and biodiversity",
                    "year": 2023,
                    "source": "semantic_scholar",
                }
            ],
        }

    async def empty_openalex_search(**kwargs: object) -> dict[str, Any]:
        openalex.calls.append(("search", dict(kwargs)))
        return {"total": 0, "offset": 0, "data": []}

    semantic.search_papers = semantic_search  # type: ignore[method-assign]
    openalex.search = empty_openalex_search  # type: ignore[method-assign]

    _, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    await runtime.search_papers_smart(
        query=(
            "What does recent evidence say about prescribed fire versus mechanical thinning for reducing wildfire risk?"
        ),
        focus=(
            "prefer reviews meta-analyses and field studies relevant to retaining biodiversity in western U.S. forests"
        ),
        limit=5,
    )

    first_query = str(semantic.calls[0][1].get("query") or "")
    assert "biodiversity" in first_query.lower()
    assert "forests" in first_query.lower()


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

    async def _aanswer_question(**kwargs: object) -> dict[str, object]:
        del kwargs
        answer_started.set()
        return {
            "answer": (
                "Concurrent answer generation and relevance scoring both ran for this saved result set, "
                "which preserved grounded evidence selection for the follow-up response."
            ),
            "confidence": "high",
            "unsupportedAsks": [],
            "followUpQuestions": [],
            "answerability": "grounded",
            "selectedEvidenceIds": ["paper-1"],
            "selectedLeadIds": [],
            "citedPaperIds": ["paper-1"],
            "evidenceSummary": "The saved paper discusses retrieval grounding.",
            "missingEvidenceDescription": "",
        }

    async def _abatched_similarity(
        query: str,
        texts: list[str],
        **kwargs: object,
    ) -> list[float]:
        del query, texts, kwargs
        scoring_started.set()
        return [0.92]

    monkeypatch.setattr(runtime._provider_bundle, "aanswer_question", _aanswer_question)
    monkeypatch.setattr(runtime._deterministic_bundle, "aanswer_question", _aanswer_question)
    monkeypatch.setattr(
        runtime._provider_bundle,
        "abatched_similarity",
        _abatched_similarity,
    )
    monkeypatch.setattr(
        runtime._deterministic_bundle,
        "abatched_similarity",
        _abatched_similarity,
    )
    monkeypatch.setattr(
        runtime,
        "_provider_bundle_for_profile",
        lambda latency_profile: runtime._provider_bundle,
    )

    ask = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question="What does this result set say about retrieval?",
        top_k=1,
        answer_mode="synthesis",
        latency_profile="deep",
    )

    assert answer_started.is_set()
    assert scoring_started.is_set()
    assert "Concurrent answer generation" in ask["answer"]
    assert ask["evidence"][0]["paper"]["paperId"] == "paper-1"
    assert ask["evidence"][0]["evidenceId"] == "paper-1"
    assert ask["answerability"] == "grounded"
    assert ask["selectedEvidenceIds"] == ["paper-1"]


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
async def test_ask_result_set_balanced_mode_uses_semantic_similarity_when_model_provider_active() -> None:
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
        raise AssertionError(
            f"ask_result_set should use workspace similarity, not provider embeddings: {texts!r}, {kwargs!r}"
        )

    bundle.aembed_texts = _unexpected_aembed_texts  # type: ignore[method-assign]

    registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
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
    search_calls: list[bool] = []
    original_asearch_papers = registry.asearch_papers

    async def _recorded_asearch_papers(
        search_session_id: str,
        query: str,
        top_k: int = 8,
        *,
        allow_model_similarity: bool = True,
    ) -> list[dict[str, Any]]:
        search_calls.append(allow_model_similarity)
        return await original_asearch_papers(
            search_session_id,
            query,
            top_k=top_k,
            allow_model_similarity=allow_model_similarity,
        )

    registry.asearch_papers = _recorded_asearch_papers  # type: ignore[method-assign]
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
    assert search_calls == [True]


@pytest.mark.asyncio
async def test_ask_result_set_caches_middle_zone_relevance_and_surfaces_rationale() -> None:
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
                    "abstract": "Vector retrieval grounds agent answers in cited evidence.",
                    "source": "semantic_scholar",
                    "verificationStatus": "verified_metadata",
                }
            ]
        },
    )

    async def _answer_question(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "answer": (
                "Retrieval-augmented agents use retrieval to ground answers in supporting documents, "
                "which improves factual consistency and evidence tracing across the saved result set."
            ),
            "unsupportedAsks": [],
            "followUpQuestions": [],
            "confidence": "medium",
            "answerability": "grounded",
            "selectedEvidenceIds": ["paper-1"],
            "selectedLeadIds": [],
            "citedPaperIds": ["paper-1"],
            "evidenceSummary": "The saved paper supports retrieval-grounded agents.",
            "missingEvidenceDescription": "",
        }

    async def _middle_zone_similarity(query: str, texts: list[str], **kwargs: object) -> list[float]:
        del query, texts, kwargs
        return [0.30]

    relevance_calls = 0

    async def _relevance_batch(**kwargs: object) -> dict[str, dict[str, str]]:
        nonlocal relevance_calls
        relevance_calls += 1
        del kwargs
        return {
            "paper-1": {
                "classification": "on_topic",
                "rationale": "The abstract directly discusses retrieval grounding for agent answers.",
            }
        }

    runtime._provider_bundle.aanswer_question = _answer_question  # type: ignore[method-assign]
    runtime._deterministic_bundle.aanswer_question = _answer_question  # type: ignore[method-assign]
    runtime._provider_bundle.abatched_similarity = _middle_zone_similarity  # type: ignore[method-assign]
    runtime._deterministic_bundle.abatched_similarity = _middle_zone_similarity  # type: ignore[method-assign]
    runtime._provider_bundle.aclassify_relevance_batch = _relevance_batch  # type: ignore[method-assign]
    runtime._deterministic_bundle.aclassify_relevance_batch = _relevance_batch  # type: ignore[method-assign]

    first = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question="What does this result set say about retrieval grounding?",
        top_k=1,
        answer_mode="synthesis",
    )
    second = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question="What does this result set say about retrieval grounding?",
        top_k=1,
        answer_mode="synthesis",
    )

    assert relevance_calls == 1
    expected_rationale = "The abstract directly discusses retrieval grounding for agent answers."
    assert first["structuredSources"][0]["note"] == expected_rationale
    assert second["structuredSources"][0]["note"] == expected_rationale


@pytest.mark.asyncio
async def test_ask_result_set_returns_evidence_use_plan_for_comparison_when_support_is_thin() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    record = registry.save_result_set(
        source_tool="search_papers_smart",
        payload={
            "data": [
                {
                    "paperId": "paper-1",
                    "title": "PFAS adsorption review",
                    "abstract": "Adsorption methods are summarized.",
                    "source": "semantic_scholar",
                    "verificationStatus": "verified_metadata",
                }
            ]
        },
    )

    async def _answer_question(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "answer": (
                "Adsorption is discussed in the saved evidence, but direct comparison support remains incomplete."
            ),
            "unsupportedAsks": [],
            "followUpQuestions": [],
            "confidence": "medium",
            "answerability": "limited",
            "selectedEvidenceIds": ["paper-1"],
            "selectedLeadIds": [],
            "citedPaperIds": ["paper-1"],
            "evidenceSummary": "The saved paper only covers adsorption.",
            "missingEvidenceDescription": "Direct membrane comparison evidence is missing.",
        }

    async def _similarity(query: str, texts: list[str], **kwargs: object) -> list[float]:
        del query, texts, kwargs
        return [0.63]

    runtime._provider_bundle.aanswer_question = _answer_question  # type: ignore[method-assign]
    runtime._deterministic_bundle.aanswer_question = _answer_question  # type: ignore[method-assign]
    runtime._provider_bundle.abatched_similarity = _similarity  # type: ignore[method-assign]
    runtime._deterministic_bundle.abatched_similarity = _similarity  # type: ignore[method-assign]

    ask = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question="Compare adsorption and membranes for PFAS removal.",
        top_k=1,
        answer_mode="comparison",
    )

    assert ask["answerStatus"] == "insufficient_evidence"
    assert ask["evidenceUsePlan"]["answerSubtype"] == "comparison"
    assert ask["evidenceUsePlan"]["retrievalSufficiency"] == "insufficient"
    assert ask["evidenceUsePlan"]["directlyResponsiveIds"] == ["paper-1"]


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
    assert ask["answerability"] == "limited"
    assert ask["selectedEvidenceIds"] == []
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

    # Comparison answers now pass through the evidence-use-plan safety gate instead of
    # being accepted just because the model produced list-formatted comparison prose.
    assert ask["answerStatus"] in {"answered", "insufficient_evidence"}
    assert ask["evidenceUsePlan"]["answerSubtype"] == "comparison"
    if ask["answerStatus"] == "answered":
        assert ask["answer"] is not None
        assert "Comparison grounded in the saved result set" in ask["answer"]
    else:
        assert ask["answer"] is None
        assert ask["evidenceUsePlan"]["retrievalSufficiency"] in {"thin", "insufficient"}


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


@pytest.mark.asyncio
async def test_search_papers_smart_known_item_retries_parsed_title_candidates_when_raw_query_misses(
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
        if kwargs.get("query") == "planetary boundaries":
            return {
                "paperId": "planetary-1",
                "title": "Planetary Boundaries: Exploring the Safe Operating Space for Humanity",
                "year": 2009,
                "venue": "Ecology and Society",
                "matchStrategy": "semantic_title_match",
            }
        return {"matchFound": False}

    monkeypatch.setattr("paper_chaser_mcp.agentic.graphs.resolve_citation", fake_resolve_citation)
    semantic.search_papers_match = fake_search_papers_match  # type: ignore[method-assign]

    smart = await runtime.search_papers_smart(
        query="Rockstrom et al. 2009 planetary boundaries Nature paper",
        limit=5,
        mode="known_item",
        latency_profile="fast",
    )

    searched_queries = [call[1]["query"] for call in semantic.calls if call[0] == "search_papers_match"]

    assert "Rockstrom et al. 2009 planetary boundaries Nature paper" in searched_queries
    assert "planetary boundaries" in searched_queries
    assert smart["results"]
    assert smart["results"][0]["paper"]["paperId"] == "planetary-1"


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


@pytest.mark.asyncio
async def test_grounded_expansions_use_provider_generated_variants() -> None:
    config = _deterministic_config()

    class _Bundle:
        async def asuggest_grounded_expansions(self, **kwargs: object) -> list[Any]:
            del kwargs
            return [
                types.SimpleNamespace(
                    variant="agentic systematic review workflow",
                    source="from_retrieved_evidence",
                    rationale="Missing workflow-specific evidence angle.",
                ),
                types.SimpleNamespace(
                    variant="retrieval augmented literature review agents",
                    source="hypothesis",
                    rationale="Missing retrieval-specific angle.",
                ),
            ]

    variants = await grounded_expansion_candidates(
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
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    expanded = [candidate.variant.lower() for candidate in variants]

    assert "agentic systematic review workflow" in expanded
    assert "retrieval augmented literature review agents" in expanded


@pytest.mark.asyncio
async def test_grounded_expansions_dedupe_near_duplicate_variants() -> None:
    config = _deterministic_config().for_latency_profile("balanced")

    class _Bundle:
        async def asuggest_grounded_expansions(self, **kwargs: object) -> list[Any]:
            del kwargs
            return [
                types.SimpleNamespace(
                    variant="Florida Scrub-Jay demography Brevard County",
                    source="from_retrieved_evidence",
                    rationale="County-specific angle.",
                ),
                types.SimpleNamespace(
                    variant="Florida Scrub Jay demography Brevard County",
                    source="hypothesis",
                    rationale="Near-duplicate county angle.",
                ),
                types.SimpleNamespace(
                    variant="Florida Scrub-Jay metapopulation viability habitat connectivity",
                    source="hypothesis",
                    rationale="Connectivity angle.",
                ),
            ]

    variants = await grounded_expansion_candidates(
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
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    expanded = [candidate.variant.lower() for candidate in variants]

    assert len(expanded) <= 3
    assert sum("demography brevard county" in variant for variant in expanded) == 1
    assert any("metapopulation viability habitat connectivity" in variant for variant in expanded)


def test_balanced_profile_reduces_expansion_breadth() -> None:
    config = _deterministic_config().for_latency_profile("balanced")

    assert config.max_grounded_variants == 2
    assert config.max_total_variants == 4
    assert config.candidate_pool_size == 50


def test_expert_smart_args_default_to_deep_latency_profile() -> None:
    assert SmartSearchPapersArgs(query="quality defaults").latency_profile == "deep"
    assert AskResultSetArgs(searchSessionId="ssn-1", question="What matters most?").latency_profile == "deep"
    assert MapResearchLandscapeArgs(searchSessionId="ssn-1").latency_profile == "deep"
    assert ExpandResearchGraphArgs(seedPaperIds=["paper-1"]).latency_profile == "deep"


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
async def test_classify_query_preserves_planner_known_item_for_citation_repair_input() -> None:
    class _Bundle:
        async def aplan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="known_item",
                queryType="citation_repair",
                querySpecificity="high",
                ambiguityLevel="low",
                candidateConcepts=["planetary boundaries"],
                retrievalHypotheses=["planetary boundaries 2009 nature"],
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
    assert planner.intent_source == "planner"
    assert planner.query_type == "citation_repair"


def test_looks_like_exact_title_identifies_title_cased_paper_queries() -> None:
    assert looks_like_exact_title("Attention Is All You Need")
    assert looks_like_exact_title("attention is all you need")
    assert not looks_like_exact_title("tool-using agents for literature review")


@pytest.mark.asyncio
async def test_classify_query_preserves_planner_known_item_for_title_like_query() -> None:
    class _Bundle:
        async def aplan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="known_item",
                queryType="known_item",
                querySpecificity="high",
                ambiguityLevel="low",
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
    assert planner.intent_source == "planner"


@pytest.mark.asyncio
async def test_classify_query_marks_broad_multi_intent_query_as_low_specificity_and_ambiguous() -> None:
    class _Bundle:
        async def aplan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="discovery",
                querySpecificity="low",
                ambiguityLevel="high",
                queryType="broad_concept",
                breadthEstimate=4,
                firstPassMode="broad",
                searchAngles=[
                    "wetland restoration climate resilience",
                    "coastal marsh restoration strategies",
                ],
                retrievalHypotheses=[
                    "evidence on restoration strategies that improve resilience",
                    "comparative evidence for coastal marsh interventions",
                ],
                intentCandidates=[
                    IntentCandidate(intent="review", confidence="medium", rationale="Planner saw synthesis language."),
                ],
                candidateConcepts=["wetland restoration", "climate resilience", "coastal marshes"],
                followUpMode="qa",
            )

    _, planner = await classify_query(
        query=(
            "What are the most effective wetland restoration strategies for "
            "improving climate resilience in coastal marshes?"
        ),
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    assert planner.query_specificity == "low"
    assert planner.ambiguity_level == "high"
    assert planner.intent_candidates[0].intent in {"discovery", "review"}


@pytest.mark.asyncio
async def test_classify_query_keeps_broad_year_bearing_environmental_query_in_discovery() -> None:
    class _Bundle:
        async def aplan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="discovery",
                querySpecificity="low",
                ambiguityLevel="medium",
                queryType="broad_concept",
                breadthEstimate=3,
                firstPassMode="mixed",
                searchAngles=["soil carbon sequestration land use change"],
                candidateConcepts=["soil carbon sequestration", "land use change", "nature-based solutions"],
                followUpMode="qa",
            )

    _, planner = await classify_query(
        query="soil carbon sequestration land-use change climate mitigation nature-based solutions 2022",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    assert planner.intent == "discovery"
    assert planner.intent_source == "planner"
    assert planner.query_specificity == "low"


@pytest.mark.asyncio
async def test_classify_query_infers_regulatory_subintent_and_entity_card() -> None:
    class _Bundle:
        async def aplan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="regulatory",
                queryType="regulatory",
                querySpecificity="high",
                ambiguityLevel="low",
                followUpMode="qa",
            )

    _, planner = await classify_query(
        query="Find the species dossier and primary regulatory history for the desert tortoise under the ESA.",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    assert planner.intent == "regulatory"
    assert planner.regulatory_subintent in {"species_dossier", "rulemaking_history"}
    assert planner.entity_card is not None
    assert planner.entity_card.get("commonName") == "desert tortoise"


@pytest.mark.asyncio
async def test_classify_query_still_uses_strong_known_item_fast_path() -> None:
    class _Bundle:
        async def aplan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(intent="discovery", followUpMode="qa")

    _, planner = await classify_query(
        query="10.1038/nature12373",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    assert planner.intent == "known_item"
    assert planner.intent_source in {"heuristic_override", "hybrid_agreement"}


@pytest.mark.asyncio
async def test_classify_query_still_uses_strong_regulatory_fast_path() -> None:
    class _Bundle:
        async def aplan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(intent="discovery", followUpMode="qa")

    _, planner = await classify_query(
        query="50 CFR 17.11 northern spotted owl",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    assert planner.intent == "regulatory"
    assert planner.intent_source in {"heuristic_override", "hybrid_agreement"}


def test_estimate_query_specificity_covers_high_low_and_medium_paths() -> None:
    assert (
        _estimate_query_specificity(
            normalized_query="10.1038/nature12373",
            focus=None,
            year=None,
            venue=None,
        )
        == "high"
    )
    assert (
        _estimate_query_specificity(
            normalized_query="what are the most effective wetland restoration strategies for coastal resilience",
            focus=None,
            year=None,
            venue=None,
        )
        == "low"
    )
    assert (
        _estimate_query_specificity(
            normalized_query="retrieval agents benchmark",
            focus="biomedical literature",
            year="2024",
            venue=None,
        )
        == "high"
    )
    assert (
        _estimate_query_specificity(
            normalized_query="retrieval grounded agents",
            focus=None,
            year=None,
            venue=None,
        )
        == "medium"
    )


def test_estimate_ambiguity_level_covers_primary_branching() -> None:
    assert (
        _estimate_ambiguity_level(
            candidates=[IntentCandidate(intent="discovery", confidence="high", rationale="broad")],
            routing_confidence="low",
            query_specificity="medium",
        )
        == "high"
    )
    assert (
        _estimate_ambiguity_level(
            candidates=[IntentCandidate(intent="discovery", confidence="high", rationale="broad")],
            routing_confidence="high",
            query_specificity="low",
        )
        == "medium"
    )
    assert (
        _estimate_ambiguity_level(
            candidates=[
                IntentCandidate(intent="discovery", confidence="high", rationale="broad"),
                IntentCandidate(intent="review", confidence="high", rationale="synthesis"),
            ],
            routing_confidence="high",
            query_specificity="medium",
        )
        == "high"
    )
    assert (
        _estimate_ambiguity_level(
            candidates=[
                IntentCandidate(intent="discovery", confidence="high", rationale="broad"),
                IntentCandidate(intent="review", confidence="medium", rationale="synthesis"),
            ],
            routing_confidence="high",
            query_specificity="low",
        )
        == "high"
    )
    assert (
        _estimate_ambiguity_level(
            candidates=[
                IntentCandidate(intent="discovery", confidence="high", rationale="broad"),
                IntentCandidate(intent="review", confidence="low", rationale="synthesis"),
            ],
            routing_confidence="high",
            query_specificity="low",
        )
        == "medium"
    )


def test_top_evidence_phrases_returns_repeated_distinctive_bigrams() -> None:
    phrases = _top_evidence_phrases(
        [
            {"title": "Retrieval grounded agents and citation graphs"},
            {"title": "Retrieval grounded agents for evidence tracing"},
            {"title": "Grounded agents with retrieval and citation graphs"},
        ],
        limit=3,
    )

    assert phrases
    assert any("retrieval grounded" in phrase for phrase in phrases)


@pytest.mark.asyncio
async def test_retrieve_variant_preserves_explicit_provider_plan_order() -> None:
    class _SemanticClient:
        async def search_papers(self, **kwargs: object) -> dict[str, Any]:
            return {
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "sem-1",
                        "title": "Semantic result",
                        "source": "semantic_scholar",
                    }
                ],
            }

    class _OpenAlexClient:
        async def search(self, **kwargs: object) -> dict[str, Any]:
            return {
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "oa-1",
                        "title": "OpenAlex result",
                        "source": "openalex",
                    }
                ],
            }

    class _EmptyClient:
        async def search(self, **kwargs: object) -> dict[str, Any]:
            return {"data": []}

    batch = await retrieve_variant(
        variant="wetland restoration climate resilience",
        variant_source="hypothesis",
        intent="discovery",
        year=None,
        venue=None,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=True,
        enable_arxiv=False,
        enable_serpapi=False,
        enable_scholarapi=False,
        core_client=_EmptyClient(),
        semantic_client=_SemanticClient(),
        openalex_client=_OpenAlexClient(),
        scholarapi_client=None,
        arxiv_client=_EmptyClient(),
        serpapi_client=None,
        provider_plan=["openalex", "semantic_scholar"],
    )

    assert batch.providers_used == ["openalex", "semantic_scholar"]
    assert [outcome["provider"] for outcome in batch.provider_outcomes[:2]] == ["openalex", "semantic_scholar"]


@pytest.mark.asyncio
async def test_rerank_candidates_adds_bridge_bonus_for_broad_query_hypotheses() -> None:
    provider_bundle = resolve_provider_bundle(
        _deterministic_config(),
        openai_api_key=None,
    )

    ranked = await rerank_candidates(
        query="wetland restoration climate resilience coastal marshes",
        merged_candidates=[
            {
                "paper": {
                    "paperId": "bridge-1",
                    "title": "Wetland restoration for climate resilience in coastal marshes",
                    "abstract": "Integrates restoration, resilience, and marsh outcomes.",
                    "year": 2024,
                    "authors": [{"name": "Author One"}],
                    "source": "openalex",
                },
                "providers": ["openalex"],
                "variants": ["wetland restoration climate resilience", "coastal marshes climate resilience"],
                "variantSources": ["hypothesis", "from_input"],
                "providerRanks": {"openalex": 1},
                "retrievalCount": 2,
            },
            {
                "paper": {
                    "paperId": "narrow-1",
                    "title": "Wetland restoration interventions",
                    "abstract": "Focuses narrowly on restoration interventions.",
                    "year": 2024,
                    "authors": [{"name": "Author Two"}],
                    "source": "semantic_scholar",
                },
                "providers": ["semantic_scholar"],
                "variants": ["wetland restoration interventions"],
                "variantSources": ["from_input"],
                "providerRanks": {"semantic_scholar": 1},
                "retrievalCount": 1,
            },
        ],
        provider_bundle=provider_bundle,
        candidate_concepts=["wetland restoration", "climate resilience", "coastal marshes"],
        routing_confidence="medium",
        query_specificity="low",
        ambiguity_level="high",
    )

    assert ranked[0]["paper"]["paperId"] == "bridge-1"
    assert ranked[0]["scoreBreakdown"]["bridgeCoverageBonus"] > 0
    assert ranked[0]["scoreBreakdown"]["broadQueryMode"] is True


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
async def test_rerank_candidates_demotes_generic_bridge_hit_without_anchor_or_title_coverage() -> None:
    provider_bundle = resolve_provider_bundle(
        _deterministic_config(),
        openai_api_key=None,
    )
    ranked = await rerank_candidates(
        query="wetland restoration climate resilience coastal marshes",
        merged_candidates=[
            {
                "paper": {
                    "paperId": "anchor-good",
                    "title": "Wetland Restoration for Climate Resilience in Coastal Marshes",
                    "abstract": "Restoration interventions improve climate resilience across coastal marsh systems.",
                    "year": 2024,
                    "authors": [{"name": "Author One"}],
                    "source": "semantic_scholar",
                },
                "providers": ["semantic_scholar"],
                "variants": ["wetland restoration climate resilience coastal marshes"],
                "variantSources": ["from_input"],
                "providerRanks": {"semantic_scholar": 2},
                "retrievalCount": 1,
            },
            {
                "paper": {
                    "paperId": "bridge-generic",
                    "title": "Ecosystem Adaptation Pathways Under Sea Level Rise",
                    "abstract": (
                        "Synthesizes ecosystem adaptation pathways across coastal "
                        "systems without discussing wetland restoration or marsh "
                        "resilience directly."
                    ),
                    "year": 2025,
                    "authors": [{"name": "Author Two"}],
                    "source": "openalex",
                },
                "providers": ["openalex", "semantic_scholar"],
                "variants": ["coastal systems adaptation", "marsh adaptation pathways"],
                "variantSources": ["hypothesis", "from_input"],
                "providerRanks": {"openalex": 1, "semantic_scholar": 1},
                "retrievalCount": 2,
            },
        ],
        provider_bundle=provider_bundle,
        candidate_concepts=["wetland restoration", "climate resilience", "coastal marshes"],
        routing_confidence="medium",
        query_specificity="low",
        ambiguity_level="high",
    )

    assert ranked[0]["paper"]["paperId"] == "anchor-good"
    generic_breakdown = next(item for item in ranked if item["paper"]["paperId"] == "bridge-generic")["scoreBreakdown"]
    assert generic_breakdown["bridgeCoverageBonus"] > 0
    assert generic_breakdown["titleFacetCoverage"] == 0.0
    assert generic_breakdown["titleAnchorCoverage"] == 0.0
    assert generic_breakdown["relevanceClassificationBonus"] <= 0.0


@pytest.mark.asyncio
async def test_rerank_candidates_anchored_broad_nitrate_headwater_stream() -> None:
    provider_bundle = resolve_provider_bundle(
        _deterministic_config(),
        openai_api_key=None,
    )
    ranked = await rerank_candidates(
        query="nitrate loading in headwater streams agricultural watersheds",
        merged_candidates=[
            {
                "paper": {
                    "paperId": "on-topic-anchor",
                    "title": "Nitrate Loading Dynamics in Headwater Streams of Agricultural Watersheds",
                    "abstract": (
                        "We quantify nitrate loading across headwater streams draining "
                        "agricultural watersheds and identify riparian buffer controls."
                    ),
                    "year": 2024,
                    "authors": [{"name": "River Scientist"}],
                    "source": "semantic_scholar",
                },
                "providers": ["semantic_scholar", "openalex"],
                "variants": ["nitrate loading in headwater streams agricultural watersheds"],
                "variantSources": ["from_input"],
                "providerRanks": {"semantic_scholar": 1, "openalex": 2},
                "retrievalCount": 2,
            },
            {
                "paper": {
                    "paperId": "drift-generic-stream",
                    "title": "General Review of Stream Ecology and Biogeochemistry",
                    "abstract": (
                        "A broad overview of stream ecology and biogeochemistry across "
                        "biomes with no specific treatment of nitrate or headwater systems."
                    ),
                    "year": 2023,
                    "authors": [{"name": "Generalist Reviewer"}],
                    "source": "openalex",
                },
                "providers": ["openalex", "semantic_scholar", "crossref"],
                "variants": ["stream ecology"],
                "variantSources": ["hypothesis"],
                "providerRanks": {"openalex": 1, "semantic_scholar": 1, "crossref": 1},
                "retrievalCount": 3,
            },
        ],
        provider_bundle=provider_bundle,
        candidate_concepts=["nitrate loading", "headwater streams", "agricultural watersheds"],
        routing_confidence="medium",
        query_specificity="low",
        ambiguity_level="medium",
    )

    assert ranked[0]["paper"]["paperId"] == "on-topic-anchor"
    on_topic_breakdown = ranked[0]["scoreBreakdown"]
    drift_breakdown = next(item for item in ranked if item["paper"]["paperId"] == "drift-generic-stream")[
        "scoreBreakdown"
    ]
    assert on_topic_breakdown["broadQueryRegime"] == "anchored_broad"
    assert drift_breakdown["broadQueryRegime"] == "anchored_broad"
    assert drift_breakdown["anchoredIntentPenalty"] > 0
    assert on_topic_breakdown["finalScore"] > drift_breakdown["finalScore"]


@pytest.mark.asyncio
async def test_rerank_candidates_anchored_broad_pesticide_pollinator() -> None:
    provider_bundle = resolve_provider_bundle(
        _deterministic_config(),
        openai_api_key=None,
    )
    ranked = await rerank_candidates(
        query="pesticide mixture exposure effects on pollinator health",
        merged_candidates=[
            {
                "paper": {
                    "paperId": "on-topic-pollinator",
                    "title": "Pesticide Mixture Exposure Reduces Pollinator Colony Health",
                    "abstract": (
                        "Field trials show pesticide mixture exposure degrades pollinator "
                        "health outcomes across honeybee and bumblebee colonies."
                    ),
                    "year": 2024,
                    "authors": [{"name": "Bee Ecologist"}],
                    "source": "semantic_scholar",
                },
                "providers": ["semantic_scholar", "openalex"],
                "variants": ["pesticide mixture exposure effects on pollinator health"],
                "variantSources": ["from_input"],
                "providerRanks": {"semantic_scholar": 1, "openalex": 3},
                "retrievalCount": 2,
            },
            {
                "paper": {
                    "paperId": "drift-agri-economics",
                    "title": "Agricultural Economics of Crop Production Systems",
                    "abstract": (
                        "An economic analysis of crop production systems and market "
                        "dynamics across global agricultural supply chains."
                    ),
                    "year": 2022,
                    "authors": [{"name": "Economist"}],
                    "source": "openalex",
                },
                "providers": ["openalex", "semantic_scholar", "crossref"],
                "variants": ["agricultural economics"],
                "variantSources": ["hypothesis"],
                "providerRanks": {"openalex": 1, "semantic_scholar": 1, "crossref": 1},
                "retrievalCount": 3,
            },
        ],
        provider_bundle=provider_bundle,
        candidate_concepts=["pesticide mixture", "pollinator health"],
        routing_confidence="medium",
        query_specificity="low",
        ambiguity_level="medium",
    )

    assert ranked[0]["paper"]["paperId"] == "on-topic-pollinator"
    on_topic_breakdown = ranked[0]["scoreBreakdown"]
    drift_breakdown = next(item for item in ranked if item["paper"]["paperId"] == "drift-agri-economics")[
        "scoreBreakdown"
    ]
    assert on_topic_breakdown["broadQueryRegime"] == "anchored_broad"
    assert drift_breakdown["broadQueryRegime"] == "anchored_broad"
    assert drift_breakdown["semanticFitGate"] < 1.0
    assert drift_breakdown["anchoredIntentPenalty"] > 0
    assert on_topic_breakdown["finalScore"] > drift_breakdown["finalScore"]


@pytest.mark.asyncio
async def test_rerank_candidates_anchored_broad_wildfire_cultural_resource() -> None:
    provider_bundle = resolve_provider_bundle(
        _deterministic_config(),
        openai_api_key=None,
    )
    ranked = await rerank_candidates(
        query="wildfire impacts on cultural resource preservation",
        merged_candidates=[
            {
                "paper": {
                    "paperId": "on-topic-cultural",
                    "title": "Wildfire Impacts on Cultural Resource Preservation in Public Lands",
                    "abstract": (
                        "Post-fire assessments document wildfire damage to cultural "
                        "resource sites and propose preservation mitigation strategies."
                    ),
                    "year": 2024,
                    "authors": [{"name": "Heritage Scholar"}],
                    "source": "semantic_scholar",
                },
                "providers": ["semantic_scholar", "openalex"],
                "variants": ["wildfire impacts on cultural resource preservation"],
                "variantSources": ["from_input"],
                "providerRanks": {"semantic_scholar": 1, "openalex": 2},
                "retrievalCount": 2,
            },
            {
                "paper": {
                    "paperId": "drift-forestry-generic",
                    "title": "Forestry Practices and Timber Yield Optimization",
                    "abstract": (
                        "A study of silvicultural forestry practices aimed at optimizing "
                        "timber yield across temperate forest plantations."
                    ),
                    "year": 2023,
                    "authors": [{"name": "Forester"}],
                    "source": "openalex",
                },
                "providers": ["openalex", "semantic_scholar", "crossref"],
                "variants": ["forestry practices"],
                "variantSources": ["hypothesis"],
                "providerRanks": {"openalex": 1, "semantic_scholar": 1, "crossref": 1},
                "retrievalCount": 3,
            },
        ],
        provider_bundle=provider_bundle,
        candidate_concepts=["wildfire impacts", "cultural resource preservation"],
        routing_confidence="medium",
        query_specificity="low",
        ambiguity_level="medium",
        planner_anchor_type="query_concepts",
        planner_anchor_value="cultural resource preservation",
    )

    assert ranked[0]["paper"]["paperId"] == "on-topic-cultural"
    on_topic_breakdown = ranked[0]["scoreBreakdown"]
    drift_breakdown = next(item for item in ranked if item["paper"]["paperId"] == "drift-forestry-generic")[
        "scoreBreakdown"
    ]
    assert on_topic_breakdown["broadQueryRegime"] == "anchored_broad"
    assert drift_breakdown["broadQueryRegime"] == "anchored_broad"
    assert drift_breakdown["anchoredIntentPenalty"] > 0
    assert on_topic_breakdown["finalScore"] > drift_breakdown["finalScore"]
    diagnostics = summarize_ranking_diagnostics(ranked, top_n=5)
    assert diagnostics
    assert diagnostics[0]["paperId"] == "on-topic-cultural"
    assert "scoreBreakdown" in diagnostics[0]
    assert diagnostics[0]["scoreBreakdown"]["broadQueryRegime"] == "anchored_broad"


@pytest.mark.asyncio
async def test_classify_query_infers_hybrid_regulatory_plus_literature_for_cultural_resource_query() -> None:
    class _Bundle:
        async def aplan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="regulatory",
                queryType="regulatory",
                querySpecificity="medium",
                ambiguityLevel="medium",
                candidateConcepts=["blue creek historic district", "section 106", "nhpa"],
                secondaryIntents=["review"],
                followUpMode="qa",
            )

    _, planner = await classify_query(
        query=(
            "Summarize recent scholarship and Section 106/NHPA regulatory history for the Blue Creek Historic District."
        ),
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    assert planner.intent == "regulatory"
    assert planner.regulatory_subintent == "hybrid_regulatory_plus_literature"
    assert "review" in planner.secondary_intents
    assert planner.entity_card is not None
    assert planner.entity_card.get("subjectArea") == "cultural_resources"
    assert planner.entity_card.get("documentFamily") == "consultation_or_preservation"


@pytest.mark.asyncio
async def test_classify_query_title_like_without_identifier_can_stay_in_discovery_when_planner_is_broad() -> None:
    class _Bundle:
        async def aplan_search(self, **kwargs: object) -> PlannerDecision:
            return PlannerDecision(
                intent="discovery",
                queryType="broad_concept",
                querySpecificity="medium",
                ambiguityLevel="medium",
                breadthEstimate=3,
                firstPassMode="mixed",
                candidateConcepts=["urban heat inequity", "major us cities"],
                followUpMode="qa",
            )

    _, planner = await classify_query(
        query="Disproportionate exposure to urban heat island intensity across major US cities",
        mode="auto",
        year=None,
        venue=None,
        focus="environmental justice literature",
        provider_bundle=_Bundle(),  # type: ignore[arg-type]
    )

    assert planner.intent == "discovery"
    assert planner.intent_source == "planner"


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
