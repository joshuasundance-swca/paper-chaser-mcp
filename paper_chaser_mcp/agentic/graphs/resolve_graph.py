"""Known-item resolution helpers extracted in Phase 7b from ``graphs/_core``.

This module owns the pure helpers used by ``AgenticRuntime._search_known_item``
and friends. The orchestration methods themselves (``_search_known_item``,
``_resolve_known_item``, ``_fallback_known_item_search``) stay on the class
because they capture runtime state (provider clients, background tasks,
context progress events). Anything that reads only the query or the parsed
citation metadata lives here so the class body shrinks and the helpers become
directly testable without instantiating a full runtime.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Literal, cast

from ...citation_repair import build_match_metadata
from ..planner import normalize_query
from ..provider_helpers import COMMON_QUERY_WORDS


def _known_item_title_similarity(query: str, title: str) -> float:
    normalized_query = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
    normalized_title = " ".join(re.findall(r"[a-z0-9]+", title.lower()))
    if not normalized_query or not normalized_title:
        return 0.0
    query_tokens = {token for token in normalized_query.split() if len(token) >= 3 and not token.isdigit()}
    title_tokens = {token for token in normalized_title.split() if len(token) >= 3 and not token.isdigit()}
    overlap = len(query_tokens & title_tokens) / len(query_tokens) if query_tokens else 0.0
    return max(SequenceMatcher(None, normalized_query, normalized_title).ratio(), overlap)


def _known_item_resolution_queries(query: str, parsed: Any) -> list[str]:
    queries: list[str] = []
    normalized_query = normalize_query(query)
    if normalized_query:
        queries.append(normalized_query)
    title_candidates = list(getattr(parsed, "title_candidates", []) or [])
    author_surnames = list(getattr(parsed, "author_surnames", []) or [])
    venue_hints = list(getattr(parsed, "venue_hints", []) or [])
    year = getattr(parsed, "year", None)

    if title_candidates:
        queries.extend(title_candidates[:3])
        compact_title_words = [
            token
            for token in re.findall(r"[A-Za-z0-9'-]+", title_candidates[0])
            if len(token) >= 3
            and token.lower() not in COMMON_QUERY_WORDS
            and token.lower() not in {"paper", "papers", "article", "articles", "study", "studies"}
        ]
        if len(compact_title_words) >= 2:
            queries.append(" ".join(compact_title_words[:8]))
        if author_surnames:
            title_words = re.findall(r"[A-Za-z0-9'-]+", title_candidates[0])[:8]
            if title_words:
                queries.append(" ".join([*author_surnames[:2], *title_words]))
        if venue_hints:
            queries.append(f"{title_candidates[0]} {venue_hints[0]}")
    if author_surnames and year is not None:
        queries.append(" ".join([*author_surnames[:2], str(year)]))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in queries:
        normalized_candidate = normalize_query(candidate)
        if not normalized_candidate:
            continue
        lowered = normalized_candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized_candidate)
    return deduped


def _normalization_metadata(raw_query: str, normalized_query: str) -> tuple[list[str], dict[str, Any]]:
    raw_text = str(raw_query or "")
    normalized_text = str(normalized_query or "")
    if raw_text == normalized_text:
        return [], {}
    return (
        ["The server normalized the incoming query before routing it."],
        {
            "query": {
                "from": raw_text,
                "to": normalized_text,
            }
        },
    )


def _anchor_strength_for_resolution(resolution_strategy: str) -> Literal["high", "medium", "low"]:
    if resolution_strategy == "citation_resolution":
        return "high"
    if resolution_strategy in {"semantic_title_match", "openalex_autocomplete"}:
        return "medium"
    return "low"


def _known_item_resolution_state_for_strategy(
    *,
    resolution_strategy: str,
    known_item: dict[str, Any],
    query: str,
) -> Literal["resolved_exact", "resolved_probable", "needs_disambiguation"]:
    """Derive a :data:`KnownItemResolutionState` for a resolved known-item payload.

    Uses the shared citation-repair metadata when available (``citation_resolution``
    round-trip) and falls back to deterministic title-similarity bands for the
    secondary resolution strategies.
    """
    title = str(known_item.get("title") or "")
    similarity = _known_item_title_similarity(query, title) if title else 0.0
    if resolution_strategy == "citation_resolution":
        metadata = build_match_metadata(
            query=query,
            paper=known_item,
            candidate_count=1,
            resolution_strategy=resolution_strategy,
        )
        state = metadata.get("knownItemResolutionState")
        if isinstance(state, str) and state in {
            "resolved_exact",
            "resolved_probable",
            "needs_disambiguation",
        }:
            return cast(Literal["resolved_exact", "resolved_probable", "needs_disambiguation"], state)
    if similarity >= 0.9:
        return "resolved_exact"
    if similarity >= 0.72:
        return "resolved_probable"
    return "needs_disambiguation"


def _known_item_recovery_warning(resolution_strategy: str) -> str:
    if resolution_strategy == "semantic_title_match":
        return "Known-item recovery used a semantic title match; verify the anchor before treating it as canonical."
    if resolution_strategy == "openalex_autocomplete":
        return "Known-item recovery used OpenAlex autocomplete; verify the anchor before treating it as canonical."
    if resolution_strategy == "openalex_search":
        return "Known-item recovery used OpenAlex search; verify the anchor before treating it as canonical."
    return "Known-item fallback used title-style recovery; verify the anchor before treating it as canonical."
