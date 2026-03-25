"""Provider clients for Paper Chaser MCP."""

from .arxiv import ArxivClient
from .core import CoreApiClient
from .crossref import CrossrefClient
from .ecos import EcosClient
from .federal_register import FederalRegisterClient
from .govinfo import GovInfoClient
from .openalex import OpenAlexClient
from .semantic_scholar import SemanticScholarClient
from .serpapi import SerpApiScholarClient
from .unpaywall import UnpaywallClient

__all__ = [
    "ArxivClient",
    "CoreApiClient",
    "CrossrefClient",
    "EcosClient",
    "FederalRegisterClient",
    "GovInfoClient",
    "OpenAlexClient",
    "SemanticScholarClient",
    "SerpApiScholarClient",
    "UnpaywallClient",
]
