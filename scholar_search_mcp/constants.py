"""Shared constants for the Scholar Search MCP package."""

API_BASE_URL = "https://api.semanticscholar.org/graph/v1"
CORE_API_BASE = "https://api.core.ac.uk/v3/search/works"
ARXIV_API_BASE = "https://export.arxiv.org/api/query"

MAX_429_RETRIES = 6

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"

DEFAULT_PAPER_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "authors",
    "citationCount",
    "referenceCount",
    "influentialCitationCount",
    "venue",
    "publicationTypes",
    "publicationDate",
    "url",
]

DEFAULT_AUTHOR_FIELDS = [
    "authorId",
    "name",
    "affiliations",
    "homepage",
    "paperCount",
    "citationCount",
    "hIndex",
]
