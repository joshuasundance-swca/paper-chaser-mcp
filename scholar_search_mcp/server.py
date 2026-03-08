"""Scholar Search MCP Server - Semantic Scholar API via Model Context Protocol."""

import asyncio
import json
import logging
import os
from typing import Any, Optional

import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scholar-search-mcp")

API_BASE_URL = "https://api.semanticscholar.org/graph/v1"

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


class SemanticScholarClient:
    """Semantic Scholar API client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        max_retries: int = 4,
        base_delay: float = 1.0,
    ) -> dict[str, Any]:
        """Send HTTP request with exponential backoff on 429."""
        url = f"{API_BASE_URL}/{endpoint}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(max_retries + 1):
                response = await client.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    params=params,
                    json=json_data,
                )

                if response.status_code == 429:
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            delay = max(delay, float(retry_after))
                        logger.warning(
                            "Rate limited (429), retrying in %.1fs (%s/%s)",
                            delay,
                            attempt + 1,
                            max_retries,
                        )
                        await asyncio.sleep(delay)
                        continue
                    response.raise_for_status()

                response.raise_for_status()
                return response.json()

    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[list[str]] = None,
        year: Optional[str] = None,
        venue: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Search papers."""
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        if year:
            params["year"] = year
        if venue:
            params["venue"] = ",".join(venue)
        return await self._request("GET", "paper/search", params=params)

    async def get_paper_details(
        self,
        paper_id: str,
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get paper details."""
        params = {"fields": ",".join(fields or DEFAULT_PAPER_FIELDS)}
        return await self._request("GET", f"paper/{paper_id}", params=params)

    async def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get papers that cite this paper."""
        params = {
            "limit": min(limit, 1000),
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        return await self._request(
            "GET", f"paper/{paper_id}/citations", params=params
        )

    async def get_paper_references(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get paper references."""
        params = {
            "limit": min(limit, 1000),
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        return await self._request(
            "GET", f"paper/{paper_id}/references", params=params
        )

    async def get_author_info(
        self,
        author_id: str,
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get author info."""
        params = {"fields": ",".join(fields or DEFAULT_AUTHOR_FIELDS)}
        return await self._request("GET", f"author/{author_id}", params=params)

    async def get_author_papers(
        self,
        author_id: str,
        limit: int = 100,
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get author papers."""
        params = {
            "limit": min(limit, 1000),
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        return await self._request(
            "GET", f"author/{author_id}/papers", params=params
        )

    async def get_recommendations(
        self,
        paper_id: str,
        limit: int = 10,
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get paper recommendations."""
        params = {
            "limit": min(limit, 100),
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        return await self._request(
            "GET",
            f"recommendations/v1/papers/forpaper/{paper_id}",
            params=params,
        )

    async def batch_get_papers(
        self,
        paper_ids: list[str],
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Batch get papers (up to 500)."""
        json_data = {"ids": paper_ids[:500]}
        params = {"fields": ",".join(fields or DEFAULT_PAPER_FIELDS)}
        return await self._request(
            "POST", "paper/batch", params=params, json_data=json_data
        )


app = Server("scholar-search")
api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
client = SemanticScholarClient(api_key=api_key)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        Tool(
            name="search_papers",
            description="Search academic papers by keyword. Optional filters: year, venue.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {
                        "type": "number",
                        "description": "Max results (default 10, max 100)",
                        "default": 10,
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to return",
                    },
                    "year": {
                        "type": "string",
                        "description": "Year filter, e.g. '2020-2023' or '2023'",
                    },
                    "venue": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Venue names to filter",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_paper_details",
            description="Get paper details. Supports DOI, ArXiv ID, Semantic Scholar ID, or URL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "Paper ID (DOI, ArXiv ID, S2 ID, etc.)",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to return",
                    },
                },
                "required": ["paper_id"],
            },
        ),
        Tool(
            name="get_paper_citations",
            description="Get list of papers that cite this paper.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper ID"},
                    "limit": {
                        "type": "number",
                        "description": "Max results (default 100, max 1000)",
                        "default": 100,
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to return",
                    },
                },
                "required": ["paper_id"],
            },
        ),
        Tool(
            name="get_paper_references",
            description="Get list of references of this paper.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper ID"},
                    "limit": {
                        "type": "number",
                        "description": "Max results (default 100, max 1000)",
                        "default": 100,
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to return",
                    },
                },
                "required": ["paper_id"],
            },
        ),
        Tool(
            name="get_author_info",
            description="Get author details.",
            inputSchema={
                "type": "object",
                "properties": {
                    "author_id": {"type": "string", "description": "Author ID"},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to return",
                    },
                },
                "required": ["author_id"],
            },
        ),
        Tool(
            name="get_author_papers",
            description="Get papers by author.",
            inputSchema={
                "type": "object",
                "properties": {
                    "author_id": {"type": "string", "description": "Author ID"},
                    "limit": {
                        "type": "number",
                        "description": "Max results (default 100, max 1000)",
                        "default": 100,
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to return",
                    },
                },
                "required": ["author_id"],
            },
        ),
        Tool(
            name="get_paper_recommendations",
            description="Get similar paper recommendations for a paper.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper ID"},
                    "limit": {
                        "type": "number",
                        "description": "Max results (default 10, max 100)",
                        "default": 10,
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to return",
                    },
                },
                "required": ["paper_id"],
            },
        ),
        Tool(
            name="batch_get_papers",
            description="Get details for multiple papers (up to 500).",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of paper IDs",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to return",
                    },
                },
                "required": ["paper_ids"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name == "search_papers":
        result = await client.search_papers(
            query=arguments["query"],
            limit=arguments.get("limit", 10),
            fields=arguments.get("fields"),
            year=arguments.get("year"),
            venue=arguments.get("venue"),
        )
    elif name == "get_paper_details":
        result = await client.get_paper_details(
            paper_id=arguments["paper_id"],
            fields=arguments.get("fields"),
        )
    elif name == "get_paper_citations":
        result = await client.get_paper_citations(
            paper_id=arguments["paper_id"],
            limit=arguments.get("limit", 100),
            fields=arguments.get("fields"),
        )
    elif name == "get_paper_references":
        result = await client.get_paper_references(
            paper_id=arguments["paper_id"],
            limit=arguments.get("limit", 100),
            fields=arguments.get("fields"),
        )
    elif name == "get_author_info":
        result = await client.get_author_info(
            author_id=arguments["author_id"],
            fields=arguments.get("fields"),
        )
    elif name == "get_author_papers":
        result = await client.get_author_papers(
            author_id=arguments["author_id"],
            limit=arguments.get("limit", 100),
            fields=arguments.get("fields"),
        )
    elif name == "get_paper_recommendations":
        result = await client.get_recommendations(
            paper_id=arguments["paper_id"],
            limit=arguments.get("limit", 10),
            fields=arguments.get("fields"),
        )
    elif name == "batch_get_papers":
        result = await client.batch_get_papers(
            paper_ids=arguments["paper_ids"],
            fields=arguments.get("fields"),
        )
    else:
        raise ValueError(f"Unknown tool: {name}")

    return [
        TextContent(
            type="text",
            text=json.dumps(result, ensure_ascii=False, indent=2),
        )
    ]


def main() -> None:
    """Run the MCP server."""
    import anyio
    from mcp.server.stdio import stdio_server

    logger.info("Starting Scholar Search MCP Server...")
    if api_key:
        logger.info("API key detected")
    else:
        logger.warning("No API key; using public rate limits")

    async def arun() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options(),
            )

    anyio.run(arun)


if __name__ == "__main__":
    main()
