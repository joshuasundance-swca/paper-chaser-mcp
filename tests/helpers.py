import json
from typing import Any


class DummyResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class DummyAsyncClient:
    def __init__(self, responses: list[DummyResponse]) -> None:
        self._responses = responses
        self.calls = 0

    async def __aenter__(self) -> "DummyAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def request(self, **kwargs) -> DummyResponse:
        response = self._responses[self.calls]
        self.calls += 1
        return response


class RecordingSemanticClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def search_papers(self, **kwargs) -> dict:
        self.calls.append(("search_papers", kwargs))
        return kwargs.pop(
            "_response",
            {"total": 1, "offset": 0, "data": [{"paperId": "semantic-1"}]},
        )

    async def search_papers_bulk(self, **kwargs) -> dict:
        self.calls.append(("search_papers_bulk", kwargs))
        return {"total": 1, "token": None, "data": [{"paperId": "bulk-1"}]}

    async def search_papers_match(self, **kwargs) -> dict:
        self.calls.append(("search_papers_match", kwargs))
        return {"paperId": "match-1", "title": "Best match"}

    async def paper_autocomplete(self, **kwargs) -> dict:
        self.calls.append(("paper_autocomplete", kwargs))
        return {"matches": [{"id": "ac-1", "title": "Autocomplete result"}]}

    async def get_paper_details(self, **kwargs) -> dict:
        self.calls.append(("get_paper_details", kwargs))
        return {"paperId": kwargs["paper_id"]}

    async def get_paper_citations(self, **kwargs) -> dict:
        self.calls.append(("get_paper_citations", kwargs))
        return {"data": [{"paperId": kwargs["paper_id"]}]}

    async def get_paper_references(self, **kwargs) -> dict:
        self.calls.append(("get_paper_references", kwargs))
        return {"data": [{"paperId": kwargs["paper_id"]}]}

    async def get_paper_authors(self, **kwargs) -> dict:
        self.calls.append(("get_paper_authors", kwargs))
        return {"total": 1, "offset": 0, "data": [{"authorId": "a-1"}]}

    async def get_author_info(self, **kwargs) -> dict:
        self.calls.append(("get_author_info", kwargs))
        return {"authorId": kwargs["author_id"]}

    async def get_author_papers(self, **kwargs) -> dict:
        self.calls.append(("get_author_papers", kwargs))
        return {"data": [{"authorId": kwargs["author_id"]}]}

    async def search_authors(self, **kwargs) -> dict:
        self.calls.append(("search_authors", kwargs))
        return {"total": 1, "offset": 0, "data": [{"authorId": "a-1"}]}

    async def batch_get_authors(self, **kwargs) -> list[dict[str, str]]:
        self.calls.append(("batch_get_authors", kwargs))
        return [{"authorId": aid} for aid in kwargs["author_ids"]]

    async def search_snippets(self, **kwargs) -> dict:
        self.calls.append(("search_snippets", kwargs))
        return {"data": [{"score": 0.9, "text": "snippet text"}]}

    async def get_recommendations(self, **kwargs) -> dict:
        self.calls.append(("get_recommendations", kwargs))
        return {"recommendedPapers": [{"paperId": kwargs["paper_id"]}]}

    async def get_recommendations_post(self, **kwargs) -> dict:
        self.calls.append(("get_recommendations_post", kwargs))
        return {"recommendedPapers": [{"paperId": "rec-post-1"}]}

    async def batch_get_papers(self, **kwargs) -> list[dict[str, str]]:
        self.calls.append(("batch_get_papers", kwargs))
        return [{"paperId": paper_id} for paper_id in kwargs["paper_ids"]]


class RecordingOpenAlexClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def paper_autocomplete(self, **kwargs) -> dict:
        self.calls.append(("paper_autocomplete", kwargs))
        return {
            "matches": [
                {
                    "id": "W-auto-1",
                    "displayName": "OpenAlex autocomplete",
                    "source": "openalex",
                }
            ]
        }

    async def search(self, **kwargs) -> dict:
        self.calls.append(("search", kwargs))
        return {
            "total": 1,
            "offset": 0,
            "data": [{"paperId": "W1", "source": "openalex"}],
        }

    async def search_bulk(self, **kwargs) -> dict:
        self.calls.append(("search_bulk", kwargs))
        return {
            "total": 1,
            "data": [{"paperId": "W1", "source": "openalex"}],
            "pagination": {"hasMore": True, "nextCursor": "oa-next"},
        }

    async def search_entities(self, **kwargs) -> dict:
        self.calls.append(("search_entities", kwargs))
        return {
            "entityType": kwargs["entity_type"],
            "total": 1,
            "offset": 0,
            "data": [
                {
                    "entityId": "https://openalex.org/S1",
                    "displayName": "OpenAlex Source",
                    "source": "openalex",
                }
            ],
            "pagination": {"hasMore": True, "nextCursor": "oa-entities"},
        }

    async def search_works_by_entity(self, **kwargs) -> dict:
        self.calls.append(("search_works_by_entity", kwargs))
        return {
            "entityType": kwargs["entity_type"],
            "entityId": kwargs["entity_id"],
            "total": 1,
            "offset": 0,
            "data": [{"paperId": "W-entity-1", "source": "openalex"}],
            "pagination": {"hasMore": True, "nextCursor": "oa-entity-papers"},
        }

    async def get_paper_details(self, **kwargs) -> dict:
        self.calls.append(("get_paper_details", kwargs))
        return {"paperId": kwargs["paper_id"], "source": "openalex"}

    async def get_paper_citations(self, **kwargs) -> dict:
        self.calls.append(("get_paper_citations", kwargs))
        return {
            "total": 1,
            "offset": 0,
            "data": [{"paperId": "W2", "source": "openalex"}],
            "pagination": {"hasMore": True, "nextCursor": "oa-cites"},
        }

    async def get_paper_references(self, **kwargs) -> dict:
        self.calls.append(("get_paper_references", kwargs))
        return {
            "total": 1,
            "offset": kwargs.get("offset", 0),
            "data": [{"paperId": "W3", "source": "openalex"}],
            "pagination": {"hasMore": True, "nextCursor": "25"},
        }

    async def search_authors(self, **kwargs) -> dict:
        self.calls.append(("search_authors", kwargs))
        return {
            "total": 1,
            "offset": 0,
            "data": [{"authorId": "A1", "name": "OpenAlex Author"}],
            "pagination": {"hasMore": True, "nextCursor": "oa-authors"},
        }

    async def get_author_info(self, **kwargs) -> dict:
        self.calls.append(("get_author_info", kwargs))
        return {"authorId": kwargs["author_id"], "name": "OpenAlex Author"}

    async def get_author_papers(self, **kwargs) -> dict:
        self.calls.append(("get_author_papers", kwargs))
        return {
            "total": 1,
            "offset": 0,
            "data": [{"paperId": "W4", "source": "openalex"}],
            "pagination": {"hasMore": True, "nextCursor": "oa-author-papers"},
        }


class RecordingCrossrefClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def get_work(self, doi: str) -> dict:
        self.calls.append(("get_work", {"doi": doi}))
        return {
            "doi": doi,
            "title": "Crossref Paper",
            "authors": [{"name": "Crossref Author"}],
            "venue": "Journal of Tests",
            "publisher": "Crossref Publisher",
            "publicationType": "journal-article",
            "publicationDate": "2024-05-01",
            "year": 2024,
            "url": f"https://doi.org/{doi}",
            "citationCount": 42,
        }

    async def search_work(self, query: str) -> dict:
        self.calls.append(("search_work", {"query": query}))
        return {
            "doi": "10.1234/crossref-query",
            "title": "Crossref Query Paper",
            "authors": [{"name": "Crossref Query Author"}],
            "venue": "Journal of Query Tests",
            "publisher": "Crossref Publisher",
            "publicationType": "journal-article",
            "publicationDate": "2023-02-15",
            "year": 2023,
            "url": "https://doi.org/10.1234/crossref-query",
            "citationCount": 7,
        }


class RecordingUnpaywallClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def get_open_access(self, doi: str) -> dict:
        self.calls.append(("get_open_access", {"doi": doi}))
        return {
            "doi": doi,
            "isOa": True,
            "oaStatus": "gold",
            "bestOaUrl": f"https://oa.example/{doi}",
            "pdfUrl": f"https://oa.example/{doi}.pdf",
            "license": "cc-by",
            "journalIsInDoaj": True,
        }


class RecordingEcosClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def search_species(self, **kwargs) -> dict:
        self.calls.append(("search_species", kwargs))
        return {
            "query": kwargs["query"],
            "matchMode": kwargs.get("match_mode", "auto"),
            "total": 1,
            "data": [
                {
                    "speciesId": "8104",
                    "commonName": "California least tern",
                    "scientificName": "Sternula antillarum browni",
                    "statusCategory": "Animal",
                    "listingStatus": "Endangered",
                    "group": "Birds",
                    "leadAgency": "FWS",
                    "profileUrl": "https://ecos.fws.gov/ecp/species/8104",
                }
            ],
        }

    async def get_species_profile(self, **kwargs) -> dict:
        self.calls.append(("get_species_profile", kwargs))
        return {
            "species": {
                "speciesId": "8104",
                "commonName": "California least tern",
                "scientificName": "Sternula antillarum browni",
                "group": "Birds",
                "profileUrl": "https://ecos.fws.gov/ecp/species/8104",
            },
            "speciesEntities": [
                {
                    "entityId": 96,
                    "agency": "FWS",
                    "status": "Endangered",
                    "statusCategory": "Animal",
                }
            ],
            "lifeHistory": "Ground-nesting seabird.",
            "range": {"historicalRangeStates": ["CA"]},
            "documents": {
                "recoveryPlans": [
                    {
                        "documentKind": "recovery_plan",
                        "title": "Revised California Least Tern Recovery Plan",
                        "url": (
                            "https://ecos.fws.gov/docs/recovery_plan/"
                            "850927_w signature.pdf"
                        ),
                        "documentDate": "09/27/1985",
                    }
                ],
                "fiveYearReviews": [
                    {
                        "documentKind": "five_year_review",
                        "title": "California Least Tern 5YR 2025",
                        "url": "https://ecosphere-documents-production-public.s3.amazonaws.com/sams/public_docs/species_nonpublish/30669.pdf",
                        "documentDate": "08/28/2025",
                    }
                ],
                "biologicalOpinions": [],
                "federalRegisterDocuments": [],
                "otherRecoveryDocs": [],
            },
            "conservationPlanLinks": [],
        }

    async def list_species_documents(self, **kwargs) -> dict:
        self.calls.append(("list_species_documents", kwargs))
        return {
            "speciesId": "8104",
            "total": 2,
            "documentKindsApplied": kwargs.get("document_kinds") or [],
            "data": [
                {
                    "documentKind": "five_year_review",
                    "title": "California Least Tern 5YR 2025",
                    "url": "https://ecosphere-documents-production-public.s3.amazonaws.com/sams/public_docs/species_nonpublish/30669.pdf",
                    "documentDate": "08/28/2025",
                },
                {
                    "documentKind": "recovery_plan",
                    "title": "Revised California Least Tern Recovery Plan",
                    "url": (
                        "https://ecos.fws.gov/docs/recovery_plan/850927_w signature.pdf"
                    ),
                    "documentDate": "09/27/1985",
                },
            ],
        }

    async def get_document_text(self, **kwargs) -> dict:
        self.calls.append(("get_document_text", kwargs))
        return {
            "document": {
                "title": "California Least Tern 5YR 2025",
                "url": kwargs["url"],
            },
            "markdown": (
                "# Five-Year Review\n\n## Recommendation\n\nRetain endangered status."
            ),
            "contentType": "application/pdf",
            "extractionStatus": "ok",
            "warnings": [],
        }


def _payload(response: list) -> Any:
    assert len(response) == 1
    return json.loads(response[0].text)


def _streamable_http_event_payload(body: str) -> dict[str, Any]:
    for line in body.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"No SSE data payload found in response: {body!r}")


class DummySerpApiAsyncClient:
    """Minimal async HTTP client stub for SerpApi tests."""

    def __init__(self, response: DummyResponse) -> None:
        self._response = response

    async def __aenter__(self) -> "DummySerpApiAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str, *, params: dict) -> DummyResponse:
        return self._response
