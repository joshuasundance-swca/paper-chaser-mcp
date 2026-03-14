"""Search fallback and result-shaping helpers."""

import logging
from typing import Any, Optional

logger = logging.getLogger("scholar-search-mcp")


def _merge_search_results(
    s2_response: dict[str, Any],
    arxiv_response: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    """Merge Semantic Scholar and arXiv results into the shared response shape."""
    s2_data = list(s2_response.get("data") or [])
    arxiv_entries = list(arxiv_response.get("entries") or [])
    for paper in s2_data:
        paper.setdefault("source", "semantic_scholar")

    seen_arxiv_ids = set()
    for paper in s2_data:
        external_ids = paper.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv")
        if arxiv_id:
            seen_arxiv_ids.add(str(arxiv_id))

    merged = list(s2_data)
    for paper in arxiv_entries:
        arxiv_id = paper.get("paperId") or ""
        if arxiv_id and arxiv_id not in seen_arxiv_ids:
            seen_arxiv_ids.add(arxiv_id)
            merged.append(paper)

    merged = merged[:limit]
    return {
        "total": len(merged),
        "offset": s2_response.get("offset", 0),
        "data": merged,
    }


def _core_response_to_merged(
    core_response: dict[str, Any], limit: int
) -> dict[str, Any]:
    """Convert CORE search response to unified shape (total, offset, data)."""
    entries = list(core_response.get("entries") or [])
    return {
        "total": core_response.get("total", len(entries)),
        "offset": 0,
        "data": entries[:limit],
    }


async def search_papers_with_fallback(
    *,
    query: str,
    limit: int,
    year: Optional[str],
    fields: Optional[list[str]],
    venue: Optional[list[str]],
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_arxiv: bool,
    core_client: Any,
    semantic_client: Any,
    arxiv_client: Any,
) -> dict[str, Any]:
    """Execute the CORE -> Semantic Scholar -> arXiv search fallback chain."""
    result = None
    if enable_core:
        try:
            core_response = await core_client.search(
                query=query,
                limit=limit,
                start=0,
                year=year,
            )
            if core_response.get("entries"):
                result = _core_response_to_merged(core_response, limit)
                logger.info("search_papers: using CORE API results")
        except Exception as exc:
            logger.info(
                "search_papers: CORE failed (%s), falling back to next channel",
                exc,
            )

    if result is None and enable_semantic_scholar:
        try:
            s2_response = await semantic_client.search_papers(
                query=query,
                limit=limit,
                fields=fields,
                year=year,
                venue=venue,
            )
            if s2_response.get("data"):
                response_data = s2_response.get("data") or []
                result = {
                    "total": s2_response.get("total", len(response_data)),
                    "offset": s2_response.get("offset", 0),
                    "data": response_data[:limit],
                }
                for paper in result["data"]:
                    paper.setdefault("source", "semantic_scholar")
                logger.info("search_papers: using Semantic Scholar results")
        except Exception as exc:
            logger.info(
                "search_papers: Semantic Scholar failed (%s), "
                "falling back to next channel",
                exc,
            )

    if result is None and enable_arxiv:
        arxiv_response = await arxiv_client.search(
            query=query,
            limit=limit,
            year=year,
        )
        if arxiv_response.get("entries"):
            arxiv_merged_response = {
                "total": arxiv_response.get("totalResults", 0),
                "entries": arxiv_response["entries"],
            }
            result = _core_response_to_merged(arxiv_merged_response, limit)
            logger.info("search_papers: using arXiv results")

    if result is None:
        return {"total": 0, "offset": 0, "data": []}
    return result
