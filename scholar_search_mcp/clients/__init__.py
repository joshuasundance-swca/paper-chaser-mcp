"""Provider clients for Scholar Search MCP."""

from .arxiv import ArxivClient
from .core import CoreApiClient
from .crossref import CrossrefClient
from .openalex import OpenAlexClient
from .semantic_scholar import SemanticScholarClient
from .serpapi import SerpApiScholarClient
from .unpaywall import UnpaywallClient

__all__ = [
    "ArxivClient",
    "CoreApiClient",
    "CrossrefClient",
    "OpenAlexClient",
    "SemanticScholarClient",
    "SerpApiScholarClient",
    "UnpaywallClient",
]
