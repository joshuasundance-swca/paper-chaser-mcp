from __future__ import annotations

import pytest

from paper_chaser_mcp import server
from paper_chaser_mcp.agentic import WorkspaceRegistry
from paper_chaser_mcp.citation_repair import (
    ParsedCitation,
    RankedCitationCandidate,
    _classify_resolution_confidence,
    _filtered_alternative_candidates,
    _normalize_identifier_for_openalex,
    _normalize_identifier_for_semantic_scholar,
    _serialize_citation_response,
    _venue_hint_in_text,
    looks_like_citation_query,
    parse_citation,
    resolve_citation,
)
from paper_chaser_mcp.enrichment import PaperEnrichmentService
from tests.helpers import (
    RecordingCrossrefClient,
    RecordingOpenAlexClient,
    RecordingSemanticClient,
    RecordingUnpaywallClient,
    _payload,
)


def test_parse_citation_extracts_environmental_reference_fields() -> None:
    parsed = parse_citation("Rockstrom et al planetary boundaries 2009 Nature 461 472")

    assert parsed.year == 2009
    assert "rockstrom" in parsed.author_surnames
    assert "nature" in parsed.venue_hints
    assert any("planetary boundaries" in candidate.lower() for candidate in parsed.title_candidates)


def test_parse_citation_extracts_bibliography_title_after_year() -> None:
    parsed = parse_citation("Vaswani A, Shazeer N, Parmar N, et al. 2017. Attention Is All You Need. NeurIPS.")

    assert parsed.year == 2017
    assert {"vaswani", "shazeer", "parmar"} <= set(parsed.author_surnames)
    assert "neurips" in parsed.venue_hints
    assert "Attention Is All You Need" in parsed.title_candidates


def test_parse_citation_does_not_infer_first_title_token_as_author_for_noisy_title_like_query() -> None:
    parsed = parse_citation(
        "RAGTruth hallucination corpus trustworthy retrieval-augmented language models Wu Zhu ACL 2024"
    )

    assert "ragtruth" not in parsed.author_surnames
    assert any(
        candidate.lower().startswith("ragtruth")
        and "trustworthy" in candidate.lower()
        and "wu" not in candidate.lower()
        and "zhu" not in candidate.lower()
        for candidate in parsed.title_candidates
    )


def test_looks_like_citation_query_is_false_for_broad_environmental_discovery_question() -> None:
    assert (
        looks_like_citation_query(
            "What is the current evidence on PFAS remediation in soils and groundwater, "
            "especially for field-deployable methods?"
        )
        is False
    )


def test_looks_like_citation_query_is_false_for_broad_year_bearing_environmental_query() -> None:
    # Long query (>= 6 tokens including year): always False
    assert (
        looks_like_citation_query(
            "soil carbon sequestration land-use change climate mitigation nature-based solutions 2022"
        )
        is False
    )
    # Shorter queries (6-7 tokens) that also lack bibliographic structure
    assert looks_like_citation_query("nature-based solutions for urban stormwater management 2022") is False
    assert looks_like_citation_query("microplastic pollution monitoring in freshwater ecosystems 2022 2023") is False
    assert looks_like_citation_query("ecotoxicology of pharmaceuticals in aquatic environments 2023") is False


def test_venue_hint_in_text_requires_word_boundaries() -> None:
    assert _venue_hint_in_text("published in nature reviews microbiology", "nature") is True
    assert _venue_hint_in_text("internationalnaturepark field report", "nature") is False


def test_filtered_alternative_candidates_drop_weak_year_only_matches() -> None:
    best = RankedCitationCandidate(
        paper={"paperId": "best", "title": "Attention Is All You Need", "year": 2017},
        score=0.96,
        resolution_strategy="exact_title",
        matched_fields=["title", "author", "year"],
        conflicting_fields=[],
        title_similarity=0.99,
        year_delta=0,
        author_overlap=2,
        candidate_count=3,
        why_selected="Best candidate.",
    )
    weak = RankedCitationCandidate(
        paper={"paperId": "weak", "title": "Orthopedic Review Handbook", "year": 2017},
        score=0.46,
        resolution_strategy="sparse_metadata",
        matched_fields=["year"],
        conflicting_fields=["title", "author"],
        title_similarity=0.18,
        year_delta=0,
        author_overlap=0,
        candidate_count=3,
        why_selected="Weak candidate.",
    )

    assert (
        _filtered_alternative_candidates(
            candidates=[best, weak],
            confidence="high",
        )
        == []
    )


def test_serialize_citation_response_caps_high_confidence_when_resolution_state_needs_disambiguation() -> None:
    parsed = ParsedCitation(
        original_text="Smith 2020 climate change adaptation",
        normalized_text="Smith 2020 climate change adaptation",
        year=2020,
        title_candidates=["climate change adaptation"],
        author_surnames=["smith"],
    )
    best = RankedCitationCandidate(
        paper={
            "paperId": "paper-2022",
            "title": "Climate Change Adaptation",
            "year": 2022,
            "authors": [{"name": "Jane Doe"}],
            "venue": "Adaptation Science",
        },
        score=0.86,
        resolution_strategy="exact_title",
        matched_fields=["title"],
        conflicting_fields=["year"],
        title_similarity=0.96,
        year_delta=2,
        author_overlap=0,
        candidate_count=2,
        why_selected="Leading exact-title match.",
    )
    runner_up = RankedCitationCandidate(
        paper={
            "paperId": "paper-2020",
            "title": "Climate Change Adaptation in Practice",
            "year": 2020,
            "authors": [{"name": "Alice Smith"}],
            "venue": "Climate Policy",
        },
        score=0.83,
        resolution_strategy="fuzzy_search",
        matched_fields=["title", "author", "year"],
        conflicting_fields=[],
        title_similarity=0.82,
        year_delta=0,
        author_overlap=1,
        candidate_count=2,
        why_selected="Near-tie runner-up.",
    )

    payload = _serialize_citation_response(
        citation=parsed.original_text,
        parsed=parsed,
        candidates=[best, runner_up],
    )

    assert payload["knownItemResolutionState"] == "needs_disambiguation"
    assert payload["resolutionConfidence"] == "medium"
    assert payload["bestMatch"]["paper"]["paperId"] == "paper-2022"
    assert payload["alternatives"][0]["paper"]["paperId"] == "paper-2020"


@pytest.mark.asyncio
async def test_resolve_citation_short_circuits_after_high_confidence_title_match() -> None:
    class FastExactSemanticClient(RecordingSemanticClient):
        async def search_papers_match(self, **kwargs) -> dict:
            self.calls.append(("search_papers_match", kwargs))
            return {
                "paperId": "transformer-1",
                "title": "Attention Is All You Need",
                "year": 2017,
                "venue": "NeurIPS",
                "authors": [
                    {"name": "Ashish Vaswani"},
                    {"name": "Noam Shazeer"},
                    {"name": "Niki Parmar"},
                ],
                "matchFound": True,
                "matchStrategy": "exact_title",
                "matchConfidence": "high",
                "matchedFields": ["title", "author", "year"],
                "candidateCount": 1,
            }

        async def search_snippets(self, **kwargs) -> dict:
            raise AssertionError(f"snippet recovery should be skipped after exact title match: {kwargs!r}")

        async def search_papers(self, **kwargs) -> dict:
            raise AssertionError(f"sparse metadata search should be skipped after exact title match: {kwargs!r}")

    semantic = FastExactSemanticClient()

    payload = await resolve_citation(
        citation=("Vaswani A, Shazeer N, Parmar N, et al. 2017. Attention Is All You Need. NeurIPS."),
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    assert payload["resolutionConfidence"] == "high"
    assert payload["resolutionStrategy"] == "exact_title"
    assert semantic.calls == [
        (
            "search_papers_match",
            {
                "query": "Attention Is All You Need",
                "fields": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_resolve_citation_tool_returns_best_match_and_search_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False)

    async def fake_match(**kwargs: object) -> dict:
        semantic.calls.append(("search_papers_match", dict(kwargs)))
        return {
            "paperId": "ss-planetary",
            "title": "A safe operating space for humanity",
            "year": 2009,
            "venue": "Nature",
            "authors": [{"name": "Johan Rockstrom"}],
            "matchFound": True,
            "matchStrategy": "fuzzy_search",
            "matchConfidence": "high",
            "matchedFields": ["title", "author", "year", "venue"],
            "titleSimilarity": 0.93,
            "yearDelta": 0,
            "authorOverlap": 1,
            "candidateCount": 3,
        }

    async def empty_search(**kwargs: object) -> dict:
        semantic.calls.append(("search_papers", dict(kwargs)))
        return {"total": 0, "offset": 0, "data": []}

    async def empty_snippets(**kwargs: object) -> dict:
        semantic.calls.append(("search_snippets", dict(kwargs)))
        return {"data": []}

    async def empty_openalex_search(**kwargs: object) -> dict:
        openalex.calls.append(("search", dict(kwargs)))
        return {"total": 0, "offset": 0, "data": []}

    semantic.search_papers_match = fake_match  # type: ignore[method-assign]
    semantic.search_papers = empty_search  # type: ignore[method-assign]
    semantic.search_snippets = empty_snippets  # type: ignore[method-assign]
    openalex.search = empty_openalex_search  # type: ignore[method-assign]

    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "openalex_client", openalex)
    monkeypatch.setattr(server, "workspace_registry", registry)
    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_openalex", True)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "enable_serpapi", False)

    payload = _payload(
        await server.call_tool(
            "resolve_citation",
            {"citation": "Rockstrom et al planetary boundaries 2009 Nature 461 472"},
        )
    )

    assert payload["bestMatch"]["paper"]["paperId"] == "ss-planetary"
    assert payload["resolutionConfidence"] in {"medium", "high"}
    assert payload["searchSessionId"]
    assert payload["agentHints"]["nextToolCandidates"][0] == "get_paper_details"
    assert any(uri == "paper://ss-planetary" for uri in payload["resourceUris"])
    record = registry.get(payload["searchSessionId"])
    assert record.papers[0]["paperId"] == "ss-planetary"


@pytest.mark.asyncio
async def test_resolve_citation_include_enrichment_only_updates_best_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False)

    class MatchingCrossrefClient(RecordingCrossrefClient):
        async def search_work(self, query: str) -> dict:
            self.calls.append(("search_work", {"query": query}))
            return {
                "doi": "10.1234/crossref-query",
                "title": "Climate adaptation pathways",
                "authors": [{"name": "Lead Author"}],
                "venue": "Nature Climate Change",
                "publisher": "Crossref Publisher",
                "publicationType": "journal-article",
                "publicationDate": "2021-05-01",
                "year": 2021,
                "url": "https://doi.org/10.1234/crossref-query",
                "citationCount": 7,
            }

    crossref = MatchingCrossrefClient()
    unpaywall = RecordingUnpaywallClient()
    enrichment_service = PaperEnrichmentService(
        crossref_client=crossref,
        unpaywall_client=unpaywall,
        enable_crossref=True,
        enable_unpaywall=True,
        provider_registry=server.provider_registry,
    )

    async def fake_match(**kwargs: object) -> dict:
        semantic.calls.append(("search_papers_match", dict(kwargs)))
        return {
            "paperId": "ss-enriched",
            "title": "Climate adaptation pathways",
            "year": 2021,
            "venue": "Nature Climate Change",
            "authors": [{"name": "Lead Author"}],
            "matchFound": True,
            "matchStrategy": "fuzzy_search",
            "matchConfidence": "high",
            "matchedFields": ["title", "author", "year"],
            "candidateCount": 2,
        }

    async def empty_search(**kwargs: object) -> dict:
        semantic.calls.append(("search_papers", dict(kwargs)))
        return {"total": 0, "offset": 0, "data": []}

    async def empty_snippets(**kwargs: object) -> dict:
        semantic.calls.append(("search_snippets", dict(kwargs)))
        return {"data": []}

    async def empty_openalex_search(**kwargs: object) -> dict:
        openalex.calls.append(("search", dict(kwargs)))
        return {"total": 0, "offset": 0, "data": []}

    semantic.search_papers_match = fake_match  # type: ignore[method-assign]
    semantic.search_papers = empty_search  # type: ignore[method-assign]
    semantic.search_snippets = empty_snippets  # type: ignore[method-assign]
    openalex.search = empty_openalex_search  # type: ignore[method-assign]

    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "openalex_client", openalex)
    monkeypatch.setattr(server, "workspace_registry", registry)
    monkeypatch.setattr(server, "enrichment_service", enrichment_service)
    monkeypatch.setattr(server, "enable_crossref", True)
    monkeypatch.setattr(server, "enable_unpaywall", True)
    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_openalex", True)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "enable_serpapi", False)

    payload = _payload(
        await server.call_tool(
            "resolve_citation",
            {
                "citation": "Climate adaptation pathways 2021 lead author",
                "includeEnrichment": True,
            },
        )
    )

    best_paper = payload["bestMatch"]["paper"]
    assert best_paper["enrichments"]["crossref"]["doi"] == "10.1234/crossref-query"
    assert best_paper["enrichments"]["unpaywall"]["isOa"] is True
    assert payload["alternatives"] == []
    assert crossref.calls == [("search_work", {"query": "Climate adaptation pathways"})]
    assert unpaywall.calls == [("get_open_access", {"doi": "10.1234/crossref-query"})]


@pytest.mark.asyncio
async def test_search_papers_match_surfaces_match_metadata() -> None:
    client = server.SemanticScholarClient()

    async def fake_request(
        method: str,
        endpoint: str,
        *,
        params: dict[str, object] | None = None,
        **_: object,
    ) -> dict:
        assert method == "GET"
        assert endpoint == "paper/search/match"
        assert params is not None
        return {
            "paperId": "paper-1",
            "title": "Attention Is All You Need",
            "year": 2017,
            "venue": "NeurIPS",
            "authors": [
                {"name": "Ashish Vaswani"},
                {"name": "Noam Shazeer"},
            ],
        }

    client._request = fake_request  # type: ignore[assignment,method-assign]

    result = await client.search_papers_match("Attention Is All You Need")

    assert result["matchFound"] is True
    assert result["matchStrategy"] == "exact_title"
    assert result["matchConfidence"] == "high"
    assert "title" in result["matchedFields"]
    assert result["candidateCount"] == 1


@pytest.mark.asyncio
async def test_resolve_citation_report_style_input_abstains_instead_of_forcing_bad_paper_match() -> None:
    class ReportStyleSemanticClient(RecordingSemanticClient):
        async def search_papers_match(self, **kwargs) -> dict:
            self.calls.append(("search_papers_match", kwargs))
            return {
                "paperId": "bird-noise-article",
                "title": "Some lessons from the effects of highway noise on birds",
                "year": 2016,
                "venue": "Proceedings of Meetings on Acoustics",
                "authors": [
                    {"name": "Robert J. Dooling"},
                    {"name": "Arthur N. Popper"},
                ],
                "matchFound": True,
                "matchStrategy": "fuzzy_search",
                "matchConfidence": "medium",
                "matchedFields": ["author", "year"],
                "conflictingFields": ["title", "venue"],
                "titleSimilarity": 0.51,
                "yearDelta": 0,
                "authorOverlap": 2,
                "candidateCount": 4,
            }

        async def search_snippets(self, **kwargs) -> dict:
            self.calls.append(("search_snippets", kwargs))
            return {"data": []}

        async def search_papers(self, **kwargs) -> dict:
            self.calls.append(("search_papers", kwargs))
            return {"total": 0, "offset": 0, "data": []}

    semantic = ReportStyleSemanticClient()

    payload = await resolve_citation(
        citation=(
            "Dooling RJ, Popper AN. 2016. Technical Guidance for Assessment and "
            "Mitigation of the Effects of Highway and Road Construction Noise on Birds. Caltrans."
        ),
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    assert payload["resolutionConfidence"] == "low"
    assert payload["bestMatch"] is None
    assert payload["alternatives"]
    assert payload["alternatives"][0]["paper"]["paperId"] == "bird-noise-article"
    assert payload["message"].lower().startswith("this citation looks report-like")


@pytest.mark.asyncio
async def test_resolve_citation_regulatory_input_redirects_to_regulatory_tools_without_paper_search() -> None:
    semantic = RecordingSemanticClient()

    payload = await resolve_citation(
        citation="Federal Register (2012). Listing of loggerhead sea turtle DPS. 77 FR 4632.",
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    assert payload["bestMatch"] is None
    assert payload["resolutionConfidence"] == "low"
    assert payload["extractedFields"]["looksLikeRegulatory"] is True
    assert payload["inferredFields"]["likelyOutputType"] == "regulatory_primary_source"
    assert "federal register" in payload["message"].lower()
    assert semantic.calls == []


@pytest.mark.asyncio
async def test_resolve_citation_bare_doi_identifier_resolution_is_high_confidence() -> None:
    semantic = RecordingSemanticClient()

    payload = await resolve_citation(
        citation="10.1038/nrn3241",
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    assert payload["bestMatch"]["paper"]["paperId"] == "DOI:10.1038/nrn3241"
    assert payload["resolutionConfidence"] == "high"
    assert semantic.calls[0] == ("get_paper_details", {"paper_id": "DOI:10.1038/nrn3241", "fields": None})


@pytest.mark.asyncio
async def test_resolve_citation_semantic_scholar_url_resolution_is_high_confidence() -> None:
    semantic = RecordingSemanticClient()
    paper_url = (
        "https://www.semanticscholar.org/paper/Attention-Is-All-You-Need/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )

    payload = await resolve_citation(
        citation=paper_url,
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    assert payload["bestMatch"]["paper"]["paperId"] == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    assert payload["resolutionConfidence"] == "high"
    assert semantic.calls[0] == (
        "get_paper_details",
        {"paper_id": "204e3073870fae3d05bcbc2f6a8e263d9b72e776", "fields": None},
    )


@pytest.mark.asyncio
async def test_resolve_citation_arxiv_identifier_resolution_is_high_confidence() -> None:
    semantic = RecordingSemanticClient()

    payload = await resolve_citation(
        citation="arXiv:1706.03762",
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    assert payload["bestMatch"]["paper"]["paperId"] == "ARXIV:1706.03762"
    assert payload["resolutionConfidence"] == "high"
    assert semantic.calls[0] == ("get_paper_details", {"paper_id": "ARXIV:1706.03762", "fields": None})


def test_classify_resolution_confidence_identifier_match_remains_high() -> None:
    assert (
        _classify_resolution_confidence(
            best_score=0.61,
            runner_up_score=0.0,
            matched_fields=["identifier"],
            conflicting_fields=["year", "author"],
            resolution_strategy="identifier",
        )
        == "high"
    )


@pytest.mark.asyncio
async def test_resolve_citation_identifier_with_noisy_surrounding_text_stays_high_confidence() -> None:
    semantic = RecordingSemanticClient()

    payload = await resolve_citation(
        citation="10.1038/nrn3241 some mismatched title fragment from another paper",
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    assert payload["bestMatch"]["paper"]["paperId"] == "DOI:10.1038/nrn3241"
    assert payload["resolutionConfidence"] == "high"
    assert payload["knownItemResolutionState"] == "resolved_exact"


def test_serialize_citation_response_abstention_keeps_plausible_alternative_when_best_is_withheld() -> None:
    parsed = ParsedCitation(
        original_text="Sparse ambiguous climate citation",
        normalized_text="Sparse ambiguous climate citation",
        title_candidates=["Climate adaptation pathways"],
        author_surnames=["smith"],
    )
    weak = RankedCitationCandidate(
        paper={
            "paperId": "candidate-1",
            "title": "Climate adaptation pathways",
            "year": 2021,
            "authors": [{"name": "Alice Smith"}],
        },
        score=0.34,
        resolution_strategy="sparse_metadata",
        matched_fields=["title", "author"],
        conflicting_fields=[],
        title_similarity=0.68,
        year_delta=None,
        author_overlap=1,
        candidate_count=1,
        why_selected="Plausible but still weak.",
    )

    payload = _serialize_citation_response(
        citation=parsed.original_text,
        parsed=parsed,
        candidates=[weak],
    )

    assert payload["bestMatch"] is None
    assert payload["alternatives"]
    assert payload["alternatives"][0]["paper"]["paperId"] == "candidate-1"


def test_classify_resolution_confidence_exact_title_with_two_key_conflicts_is_medium() -> None:
    # exact_title with 2+ conflicting key fields (author + year) must not return "high"
    assert (
        _classify_resolution_confidence(
            best_score=0.88,
            runner_up_score=0.0,
            matched_fields=["title"],
            conflicting_fields=["author", "year", "venue"],
            resolution_strategy="exact_title",
        )
        == "medium"
    )


@pytest.mark.asyncio
async def test_resolve_citation_recovers_ragtruth_from_noisy_acl_prompt() -> None:
    class RagTruthSemanticClient(RecordingSemanticClient):
        async def search_papers_match(self, **kwargs) -> dict:
            self.calls.append(("search_papers_match", kwargs))
            query = str(kwargs["query"])
            normalized = query.lower()
            if (
                "ragtruth" in normalized
                and "trustworthy" in normalized
                and "wu" not in normalized
                and "zhu" not in normalized
            ):
                return {
                    "paperId": "ragtruth-final",
                    "title": (
                        "RAGTruth: A Hallucination Corpus for Developing Trustworthy "
                        "Retrieval-Augmented Language Models"
                    ),
                    "year": 2024,
                    "venue": "ACL 2024",
                    "authors": [
                        {"name": "Shuai Wu"},
                        {"name": "Yichao Zhu"},
                    ],
                    "externalIds": {"DOI": "10.18653/v1/2024.acl-long.123"},
                    "matchFound": True,
                    "matchStrategy": "fuzzy_search",
                    "matchConfidence": "high",
                    "matchedFields": ["title", "year", "venue"],
                    "candidateCount": 2,
                }
            return {}

        async def search_snippets(self, **kwargs) -> dict:
            self.calls.append(("search_snippets", kwargs))
            return {"data": []}

        async def search_papers(self, **kwargs) -> dict:
            self.calls.append(("search_papers", kwargs))
            return {"total": 0, "offset": 0, "data": []}

    semantic = RagTruthSemanticClient()

    payload = await resolve_citation(
        citation="RAGTruth hallucination corpus trustworthy retrieval-augmented language models Wu Zhu ACL 2024",
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    assert payload["bestMatch"]["paper"]["paperId"] == "ragtruth-final"
    assert payload["resolutionConfidence"] in {"medium", "high"}
    assert any(
        call[0] == "search_papers_match"
        and "wu" not in str(call[1]["query"]).lower()
        and "zhu" not in str(call[1]["query"]).lower()
        for call in semantic.calls
    )


@pytest.mark.asyncio
async def test_resolve_citation_prefers_final_publication_over_preprint_when_candidates_compete() -> None:
    class PublicationPreferenceSemanticClient(RecordingSemanticClient):
        async def search_papers_match(self, **kwargs) -> dict:
            self.calls.append(("search_papers_match", kwargs))
            return {}

        async def search_snippets(self, **kwargs) -> dict:
            self.calls.append(("search_snippets", kwargs))
            return {"data": []}

        async def search_papers(self, **kwargs) -> dict:
            self.calls.append(("search_papers", kwargs))
            return {
                "total": 2,
                "offset": 0,
                "data": [
                    {
                        "paperId": "ragtruth-preprint",
                        "title": (
                            "RAGTruth: A Hallucination Corpus for Developing Trustworthy "
                            "Retrieval-Augmented Language Models"
                        ),
                        "year": 2024,
                        "venue": "arXiv",
                        "source": "arxiv",
                        "publicationTypes": ["preprint"],
                        "authors": [
                            {"name": "Shuai Wu"},
                            {"name": "Yichao Zhu"},
                        ],
                    },
                    {
                        "paperId": "ragtruth-final",
                        "title": (
                            "RAGTruth: A Hallucination Corpus for Developing Trustworthy "
                            "Retrieval-Augmented Language Models"
                        ),
                        "year": 2024,
                        "venue": "ACL 2024",
                        "source": "semantic_scholar",
                        "publicationTypes": ["conference"],
                        "canonicalId": "10.18653/v1/2024.acl-long.123",
                        "externalIds": {"DOI": "10.18653/v1/2024.acl-long.123"},
                        "authors": [
                            {"name": "Shuai Wu"},
                            {"name": "Yichao Zhu"},
                        ],
                    },
                ],
            }

    semantic = PublicationPreferenceSemanticClient()

    payload = await resolve_citation(
        citation="RAGTruth A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models 2024",
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    assert payload["bestMatch"]["paper"]["paperId"] == "ragtruth-final"
    assert payload["alternatives"][0]["paper"]["paperId"] == "ragtruth-preprint"


def test_classify_resolution_confidence_exact_title_with_one_key_conflict_remains_high() -> None:
    # exact_title with only one key conflict is still allowed to be "high"
    assert (
        _classify_resolution_confidence(
            best_score=0.88,
            runner_up_score=0.0,
            matched_fields=["title", "author"],
            conflicting_fields=["year"],
            resolution_strategy="exact_title",
        )
        == "high"
    )


def test_classify_resolution_confidence_exact_title_no_conflicts_is_high() -> None:
    assert (
        _classify_resolution_confidence(
            best_score=0.88,
            runner_up_score=0.0,
            matched_fields=["title", "author", "year"],
            conflicting_fields=[],
            resolution_strategy="exact_title",
        )
        == "high"
    )


def test_classify_resolution_confidence_openalex_exact_title_with_two_key_conflicts_is_medium() -> None:
    # openalex_exact_title strategy shares the same suffix check
    assert (
        _classify_resolution_confidence(
            best_score=0.85,
            runner_up_score=0.0,
            matched_fields=["title"],
            conflicting_fields=["author", "year"],
            resolution_strategy="openalex_exact_title",
        )
        == "medium"
    )


def test_identifier_normalizers_cover_openalex_and_generic_url_branches() -> None:
    assert _normalize_identifier_for_openalex("doi:10.1038/nrn3241", "doi") == "10.1038/nrn3241"
    assert _normalize_identifier_for_openalex("https://openalex.org/W12345", "url") == "W12345"
    assert _normalize_identifier_for_openalex("https://example.org/paper", "url") is None
    assert (
        _normalize_identifier_for_semantic_scholar("https://example.org/paper", "url")
        == "URL:https://example.org/paper"
    )


@pytest.mark.asyncio
async def test_resolve_citation_exact_title_with_conflicting_author_year_venue_downgrades_confidence() -> None:
    """Regression: exact_title match should not return high confidence when author/year/venue all conflict."""

    class ConflictingExactTitleClient(RecordingSemanticClient):
        async def search_papers_match(self, **kwargs) -> dict:
            self.calls.append(("search_papers_match", kwargs))
            return {
                "paperId": "planetary-2021",
                "title": "Planetary Boundaries",
                "year": 2021,
                "venue": "Some Journal 2021",
                "authors": [{"name": "Vincent Bellinkx"}],
                "matchFound": True,
                "matchStrategy": "exact_title",
                "matchConfidence": "high",
                "matchedFields": ["title"],
                "conflictingFields": ["author", "year", "venue"],
                "candidateCount": 1,
            }

        async def search_snippets(self, **kwargs) -> dict:
            self.calls.append(("search_snippets", kwargs))
            return {"data": []}

        async def search_papers(self, **kwargs) -> dict:
            self.calls.append(("search_papers", kwargs))
            return {"total": 0, "offset": 0, "data": []}

    semantic = ConflictingExactTitleClient()

    payload = await resolve_citation(
        citation="Rockstrom et al planetary boundaries 2009 Nature 461 472",
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    # With author, year, and venue all conflicting, confidence must not be "high"
    assert payload["resolutionConfidence"] != "high", (
        "exact_title match with 3 conflicting key fields must not return high confidence"
    )


@pytest.mark.asyncio
async def test_resolve_citation_prefers_year_aligned_candidate_over_conflicting_exact_title_match() -> None:
    class MixedCandidateClient(RecordingSemanticClient):
        async def search_papers_match(self, **kwargs) -> dict:
            self.calls.append(("search_papers_match", kwargs))
            return {
                "paperId": "planetary-2021",
                "title": "Planetary Boundaries",
                "year": 2021,
                "venue": "Some Journal 2021",
                "authors": [{"name": "Vincent Bellinkx"}],
                "matchFound": True,
                "matchStrategy": "exact_title",
                "matchConfidence": "high",
                "matchedFields": ["title"],
                "conflictingFields": ["author", "year", "venue"],
                "candidateCount": 1,
            }

        async def search_snippets(self, **kwargs) -> dict:
            self.calls.append(("search_snippets", kwargs))
            return {"data": []}

        async def search_papers(self, **kwargs) -> dict:
            self.calls.append(("search_papers", kwargs))
            return {
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "planetary-2009",
                        "title": "Planetary Boundaries",
                        "year": 2009,
                        "venue": "Nature",
                        "authors": [
                            {"name": "Johan Rockstrom"},
                            {"name": "Will Steffen"},
                        ],
                    }
                ],
            }

    semantic = MixedCandidateClient()

    payload = await resolve_citation(
        citation="Rockstrom et al planetary boundaries 2009 Nature 461 472",
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    assert payload["bestMatch"]["paper"]["paperId"] == "planetary-2009"
    if payload["alternatives"]:
        assert payload["alternatives"][0]["paper"]["paperId"] == "planetary-2021"


@pytest.mark.asyncio
async def test_resolve_citation_case_insensitive_title_match_for_human_domination() -> None:
    class CapsVariantClient(RecordingSemanticClient):
        async def search_papers_match(self, **kwargs) -> dict:
            self.calls.append(("search_papers_match", kwargs))
            return {
                "paperId": "human-domination",
                "title": "Human Domination of Earth's Ecosystems",
                "year": 1997,
                "venue": "Science",
                "authors": [{"name": "Peter M. Vitousek"}],
                "matchFound": True,
                "matchStrategy": "exact_title",
                "matchConfidence": "high",
                "matchedFields": ["title"],
                "conflictingFields": [],
                "candidateCount": 1,
            }

    semantic = CapsVariantClient()

    payload = await resolve_citation(
        citation="HUMAN DOMINATION OF EARTH'S ECOSYSTEMS",
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    assert payload["bestMatch"]["paper"]["paperId"] == "human-domination"
    assert payload["resolutionConfidence"] == "high"


# ============================================================
# P2-2: Citation resolution for natural-language classic papers
# ============================================================


def test_parse_citation_extracts_author_initials_watson_crick() -> None:
    parsed = parse_citation(
        "Watson, J. D., & Crick, F. H. C. (1953). Molecular structure of nucleic acids. Nature, 171(4356), 737-738."
    )

    assert parsed.year == 1953
    assert "watson" in parsed.author_surnames
    assert "crick" in parsed.author_surnames
    assert "nature" in parsed.venue_hints


def test_sparse_search_queries_prioritizes_journal_hint() -> None:
    from paper_chaser_mcp.citation_repair import _sparse_search_queries

    parsed = parse_citation("Watson Crick 1953 molecular structure nucleic acids Nature")
    queries = _sparse_search_queries(parsed)

    assert queries, "expected at least one sparse query"
    first_lower = queries[0].lower()
    assert "nature" in first_lower, f"expected venue hint first, got queries={queries!r}"
    assert "watson" in first_lower or "1953" in first_lower


def test_rank_candidate_penalizes_year_mismatch_watson_crick_variant() -> None:
    from paper_chaser_mcp.citation_repair import _rank_candidate

    parsed = parse_citation("Watson Crick 1953 molecular structure nucleic acids Nature")

    old_paper = {
        "paperId": "old",
        "title": "molecular structure",
        "year": 1953,
        "authors": [{"name": "James D Watson"}, {"name": "Francis H Crick"}],
        "venue": "Nature",
    }
    modern_paper = {
        "paperId": "modern",
        "title": "molecular structure of nucleic acids a modern reanalysis",
        "year": 2020,
        "authors": [{"name": "Jane Doe"}],
        "venue": "Some Journal",
    }

    old_ranked = _rank_candidate(
        paper=old_paper,
        parsed=parsed,
        resolution_strategy="sparse_metadata",
        candidate_count=2,
        snippet_text=None,
    )
    modern_ranked = _rank_candidate(
        paper=modern_paper,
        parsed=parsed,
        resolution_strategy="sparse_metadata",
        candidate_count=2,
        snippet_text=None,
    )

    assert old_ranked.score > modern_ranked.score, (
        f"expected 1953 match to outrank 2020 variant; old={old_ranked.score:.3f} modern={modern_ranked.score:.3f}"
    )


def test_rank_candidate_boosts_multi_author_match() -> None:
    from paper_chaser_mcp.citation_repair import _rank_candidate

    parsed = parse_citation("Watson Crick 1953 molecular structure nucleic acids Nature")

    both = {
        "paperId": "both",
        "title": "molecular structure of nucleic acids",
        "year": 1953,
        "authors": [{"name": "James Watson"}, {"name": "Francis Crick"}],
        "venue": "Nature",
    }
    one_only = {
        "paperId": "one",
        "title": "molecular structure of nucleic acids",
        "year": 1953,
        "authors": [{"name": "James Watson"}, {"name": "Rosalind Franklin"}],
        "venue": "Nature",
    }

    both_ranked = _rank_candidate(
        paper=both,
        parsed=parsed,
        resolution_strategy="sparse_metadata",
        candidate_count=2,
        snippet_text=None,
    )
    one_ranked = _rank_candidate(
        paper=one_only,
        parsed=parsed,
        resolution_strategy="sparse_metadata",
        candidate_count=2,
        snippet_text=None,
    )

    assert both_ranked.author_overlap == 2
    assert one_ranked.author_overlap == 1
    assert both_ranked.score - one_ranked.score >= 0.15, (
        f"expected multi-author match to beat single-author by >=0.15; "
        f"both={both_ranked.score:.3f} one={one_ranked.score:.3f}"
    )


@pytest.mark.asyncio
async def test_resolve_citation_famous_paper_registry_watson_crick() -> None:
    """Natural-language classics should resolve via the famous-paper registry
    without hitting any upstream provider."""

    class NetworkForbiddenClient:
        def __getattr__(self, name: str):
            async def _raise(*args, **kwargs):
                raise AssertionError(f"no network calls allowed; method {name!r} was invoked")

            return _raise

    client = NetworkForbiddenClient()

    payload = await resolve_citation(
        citation="Watson and Crick 1953 DNA Nature",
        max_candidates=5,
        client=client,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=None,
        arxiv_client=None,
        serpapi_client=None,
    )

    best = payload["bestMatch"]
    assert best is not None, "expected famous-paper registry to produce a bestMatch"
    paper = best["paper"]
    external_ids = paper.get("externalIds") or {}
    doi = str(external_ids.get("DOI") or "").lower()
    assert doi == "10.1038/171737a0", f"expected Watson/Crick DOI, got {external_ids!r}"
    assert payload["resolutionConfidence"] in {"medium", "high"}
