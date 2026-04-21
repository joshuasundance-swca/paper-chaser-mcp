"""Snippet-search fallback helpers for the Semantic Scholar degraded path.

When Semantic Scholar's ``search_papers_snippet`` endpoint degrades (returns
``degraded=True`` with empty ``data``), the dispatch layer invokes these
helpers to synthesize best-effort paper matches from the regular
``search_papers`` endpoint so agents still receive grounded evidence
instead of an empty-result dead end.

Extracted from ``paper_chaser_mcp.dispatch._core`` as part of the Phase 2
refactor. Behavior is preserved verbatim; only the module boundary moves.
"""

from __future__ import annotations

import re
from typing import Any

from ..models import dump_jsonable


def _snippet_fallback_query(query: str) -> str:
    normalized = " ".join(str(query or "").strip().strip("\"'").split())
    tokens = re.findall(r"[A-Za-z0-9]{3,}", normalized)
    return " ".join(tokens[:10]) if tokens else normalized


def _snippet_fallback_results(
    degraded_payload: dict[str, Any],
    papers_payload: dict[str, Any],
) -> dict[str, Any]:
    fallback_items: list[dict[str, Any]] = []
    for index, paper in enumerate((papers_payload.get("data") or []), start=1):
        if not isinstance(paper, dict):
            continue
        snippet_text = str(paper.get("abstract") or paper.get("title") or "").strip()
        if not snippet_text:
            continue
        fallback_items.append(
            {
                "score": round(max(0.0, 1.0 - ((index - 1) * 0.05)), 6),
                "snippet": {
                    "text": snippet_text[:400],
                    "snippetKind": "fallback_paper_match",
                    "section": "abstract" if paper.get("abstract") else "title",
                },
                "paper": {
                    "paperId": paper.get("paperId"),
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "url": paper.get("url"),
                },
            }
        )
    if not fallback_items:
        return degraded_payload

    payload = dict(degraded_payload)
    payload["data"] = fallback_items
    payload["fallbackUsed"] = "search_papers"
    payload["message"] = (
        "Semantic Scholar snippet search could not serve this query, so the "
        "server returned best-effort paper matches from search_papers instead."
    )
    return payload


async def _maybe_fallback_snippet_search(
    *,
    serialized: dict[str, Any],
    args_dict: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    if serialized.get("degraded") is not True or serialized.get("data"):
        return serialized
    fallback_query = _snippet_fallback_query(str(args_dict.get("query") or ""))
    if not fallback_query:
        return serialized
    try:
        fallback_payload = dump_jsonable(
            await client.search_papers(
                query=fallback_query,
                limit=args_dict.get("limit", 10),
                fields=["paperId", "title", "year", "url", "abstract"],
                year=args_dict.get("year"),
                publication_date_or_year=args_dict.get("publication_date_or_year"),
                fields_of_study=args_dict.get("fields_of_study"),
                min_citation_count=args_dict.get("min_citation_count"),
                venue=[args_dict["venue"]] if args_dict.get("venue") else None,
            )
        )
    except Exception:
        return serialized
    return _snippet_fallback_results(serialized, fallback_payload)


__all__ = (
    "_snippet_fallback_query",
    "_snippet_fallback_results",
    "_maybe_fallback_snippet_search",
)
