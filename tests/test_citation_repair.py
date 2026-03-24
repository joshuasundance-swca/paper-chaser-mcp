from __future__ import annotations

import pytest

from scholar_search_mcp import server
from scholar_search_mcp.agentic import WorkspaceRegistry
from scholar_search_mcp.citation_repair import (
    RankedCitationCandidate,
    _filtered_alternative_candidates,
    parse_citation,
    resolve_citation,
)
from scholar_search_mcp.enrichment import PaperEnrichmentService
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
