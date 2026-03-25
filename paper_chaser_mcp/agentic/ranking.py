"""Deduplication and reranking for smart search candidates."""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from typing import Any

from .config import AgenticConfig
from .planner import query_facets, query_terms
from .providers import ModelProviderBundle
from .retrieval import RetrievedCandidate

PROVIDER_QUALITY_BONUS = {
    "semantic_scholar": 0.045,
    "openalex": 0.035,
    "core": 0.015,
    "arxiv": 0.015,
    "serpapi_google_scholar": 0.0,
}
GENERIC_RESEARCH_TERMS = {
    "agent",
    "agents",
    "analysis",
    "framework",
    "frameworks",
    "literature",
    "method",
    "methods",
    "paper",
    "papers",
    "research",
    "review",
    "reviews",
    "study",
    "studies",
    "survey",
    "surveys",
    "system",
    "systems",
}


def canonical_dedupe_key(paper: dict[str, Any]) -> str:
    """Deduplicate by DOI, arXiv, portable IDs, then normalized title fallback."""
    return candidate_identity_keys(paper)[0]


def candidate_identity_keys(paper: dict[str, Any]) -> list[str]:
    """Return all stable identity keys that may connect duplicates."""
    recommended = _normalized_string(paper.get("recommendedExpansionId"))
    canonical = _normalized_string(paper.get("canonicalId"))
    source_id = _normalized_string(paper.get("sourceId"))
    paper_id = _normalized_string(paper.get("paperId"))
    keys: list[str] = []

    doi_candidate = next(
        (
            candidate
            for candidate in [recommended, canonical, paper_id, source_id]
            if candidate and candidate.startswith("10.")
        ),
        None,
    )
    if doi_candidate:
        keys.append(f"doi:{doi_candidate}")

    arxiv_candidate = next(
        (
            candidate
            for candidate in [recommended, canonical, paper_id, source_id]
            if candidate and _looks_like_arxiv(candidate)
        ),
        None,
    )
    if arxiv_candidate:
        keys.append(f"arxiv:{arxiv_candidate}")

    for portable_id in [recommended, canonical, paper_id, source_id]:
        if portable_id:
            key = f"id:{portable_id}"
            if key not in keys:
                keys.append(key)

    title = _normalized_title(paper.get("title"))
    year = _normalized_string(paper.get("year")) or "unknown"
    first_author = _normalized_title(_first_author_name(paper))
    title_year_key = f"title:{title}|year:{year}"
    if title_year_key not in keys:
        keys.append(title_year_key)
    title_key = f"title:{title}|year:{year}|author:{first_author}"
    if title_key not in keys:
        keys.append(title_key)
    return keys


def merge_candidates(candidates: list[RetrievedCandidate]) -> list[dict[str, Any]]:
    """Merge per-provider candidates into canonical paper records."""
    merged: dict[str, dict[str, Any]] = {}
    alias_map: dict[str, str] = {}
    for candidate in candidates:
        identity_keys = candidate_identity_keys(candidate.paper)
        existing_key = next(
            (alias_map[identity_key] for identity_key in identity_keys if identity_key in alias_map),
            None,
        )
        key = existing_key or canonical_dedupe_key(candidate.paper)
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                "paper": dict(candidate.paper),
                "providers": {candidate.provider},
                "variants": {candidate.variant},
                "variantSources": {candidate.variant_source},
                "providerRanks": {candidate.provider: candidate.provider_rank},
                "retrievalCount": 1,
            }
            for identity_key in identity_keys:
                alias_map[identity_key] = key
            continue
        existing["providers"].add(candidate.provider)
        existing["variants"].add(candidate.variant)
        existing["variantSources"].add(candidate.variant_source)
        existing["providerRanks"][candidate.provider] = min(
            existing["providerRanks"].get(candidate.provider, candidate.provider_rank),
            candidate.provider_rank,
        )
        existing["retrievalCount"] += 1
        existing["paper"] = _merge_paper_dicts(existing["paper"], candidate.paper)
        for identity_key in identity_keys:
            alias_map[identity_key] = key

    merged_list = list(merged.values())
    for item in merged_list:
        item["providers"] = sorted(item["providers"])
        item["variants"] = sorted(item["variants"])
        item["variantSources"] = sorted(item["variantSources"])
    return merged_list


async def rerank_candidates(
    *,
    query: str,
    merged_candidates: list[dict[str, Any]],
    provider_bundle: ModelProviderBundle,
    candidate_concepts: list[str],
    candidate_pool_size: int | None = None,
    request_outcomes: list[dict[str, Any]] | None = None,
    request_id: str | None = None,
) -> list[dict[str, Any]]:
    """Score and rank the merged candidate pool."""
    if not merged_candidates:
        return []
    current_year = time.gmtime().tm_year
    paper_texts = [_paper_text(item["paper"]) for item in merged_candidates]
    if candidate_pool_size is not None and len(merged_candidates) > candidate_pool_size:
        pre_ranked: list[tuple[float, dict[str, Any], str]] = []
        for item, paper_text in zip(merged_candidates, paper_texts):
            paper = item["paper"]
            fused_rank_score = sum(1.0 / (60.0 + rank) for rank in item["providerRanks"].values())
            provider_bonus = max(
                (PROVIDER_QUALITY_BONUS.get(provider, 0.0) for provider in item["providers"]),
                default=0.0,
            )
            provider_bonus += min(max(len(item["providers"]) - 1, 0) * 0.02, 0.06)
            citation_count = paper.get("citationCount")
            citation_bonus = 0.0
            if isinstance(citation_count, int) and citation_count > 0:
                citation_bonus += min(math.log1p(citation_count) / 25.0, 0.12)
            year = paper.get("year")
            if isinstance(year, int) and year <= current_year:
                age = max(current_year - year, 0)
                citation_bonus += max(0.0, 0.05 - min(age * 0.004, 0.05))
            pre_ranked.append(
                (
                    _lexical_similarity(query, paper_text) + fused_rank_score + provider_bonus + citation_bonus,
                    item,
                    paper_text,
                )
            )
        pre_ranked.sort(key=lambda item: item[0], reverse=True)
        trimmed = pre_ranked[:candidate_pool_size]
        merged_candidates = [item for _, item, _ in trimmed]
        paper_texts = [paper_text for _, _, paper_text in trimmed]

    query_similarities = await provider_bundle.abatched_similarity(
        query,
        paper_texts,
        request_outcomes=request_outcomes,
        request_id=request_id,
    )
    facets = query_facets(query)
    terms = query_terms(query)
    anchor_terms = [term for term in terms if term not in GENERIC_RESEARCH_TERMS]

    for item, paper_text, query_similarity in zip(
        merged_candidates,
        paper_texts,
        query_similarities,
    ):
        paper = item["paper"]
        title_text = _paper_title_text(paper)
        lowered_paper_text = paper_text.lower()
        paper_tokens = _tokenize_text(lowered_paper_text)
        title_tokens = _tokenize_text(title_text.lower())
        fused_rank_score = sum(1.0 / (60.0 + rank) for rank in item["providerRanks"].values())
        matched_candidate_concepts = [
            concept for concept in candidate_concepts if concept and _concept_matches_tokens(concept, paper_tokens)
        ]
        matched_facets = [facet for facet in facets if _paper_matches_facet(paper_tokens, facet)]
        matched_terms = [term for term in terms if term in paper_tokens]
        matched_title_facets = [facet for facet in facets if _paper_matches_facet(title_tokens, facet)]
        matched_title_terms = [term for term in terms if term in title_tokens]
        matched_anchor_terms = [term for term in anchor_terms if term in paper_tokens]
        matched_title_anchor_terms = [term for term in anchor_terms if term in title_tokens]
        matched_concepts = _dedupe_strings(matched_facets + matched_candidate_concepts)
        facet_coverage = len(matched_facets) / len(facets) if facets else 0.0
        term_coverage = len(matched_terms) / len(terms) if terms else 0.0
        title_facet_coverage = len(matched_title_facets) / len(facets) if facets else 0.0
        title_term_coverage = len(matched_title_terms) / len(terms) if terms else 0.0
        anchor_coverage = len(matched_anchor_terms) / len(anchor_terms) if anchor_terms else 0.0
        title_anchor_coverage = len(matched_title_anchor_terms) / len(anchor_terms) if anchor_terms else 0.0
        concept_bonus = (
            min(len(matched_candidate_concepts) * 0.02, 0.08)
            + min(facet_coverage * 0.12, 0.12)
            + min(term_coverage * 0.06, 0.06)
            + min(anchor_coverage * 0.08, 0.08)
            + min(title_facet_coverage * 0.16, 0.16)
            + min(title_term_coverage * 0.08, 0.08)
            + min(title_anchor_coverage * 0.14, 0.14)
        )
        provider_bonus = max(
            (PROVIDER_QUALITY_BONUS.get(provider, 0.0) for provider in item["providers"]),
            default=0.0,
        )
        provider_bonus += min(max(len(item["providers"]) - 1, 0) * 0.02, 0.06)
        citation_count = paper.get("citationCount")
        citation_bonus = 0.0
        if isinstance(citation_count, int) and citation_count > 0:
            citation_bonus += min(math.log1p(citation_count) / 25.0, 0.12)
        year = paper.get("year")
        if isinstance(year, int) and year <= current_year:
            age = max(current_year - year, 0)
            citation_bonus += max(0.0, 0.05 - min(age * 0.004, 0.05))

        drift_penalty = 0.0
        if "speculative" in item["variantSources"] and query_similarity < 0.12:
            drift_penalty = 0.12
        facet_penalty = 0.0
        if len(facets) >= 2:
            if not matched_facets:
                facet_penalty = 0.12
            elif len(matched_facets) == 1:
                facet_penalty = 0.045
        elif facets and not matched_facets:
            facet_penalty = 0.05
        if terms and term_coverage < 0.35:
            facet_penalty += 0.03
        if len(facets) >= 2:
            if not matched_title_facets:
                facet_penalty += 0.08
            elif len(matched_title_facets) == 1:
                facet_penalty += 0.02
        elif facets and not matched_title_facets:
            facet_penalty += 0.04
        if terms and title_term_coverage < 0.25:
            facet_penalty += 0.05
        if anchor_terms and not matched_anchor_terms:
            facet_penalty += 0.05
        if anchor_terms and not matched_title_anchor_terms:
            facet_penalty += 0.08
        final_score = (
            fused_rank_score
            + query_similarity
            + concept_bonus
            + provider_bonus
            + citation_bonus
            - drift_penalty
            - facet_penalty
        )
        item["matchedConcepts"] = matched_concepts
        item["scoreBreakdown"] = {
            "fusedRankScore": round(fused_rank_score, 6),
            "querySimilarity": round(query_similarity, 6),
            "conceptCoverageBonus": round(concept_bonus, 6),
            "providerConsensusBonus": round(provider_bonus, 6),
            "queryFacetCoverage": round(facet_coverage, 6),
            "queryTermCoverage": round(term_coverage, 6),
            "queryAnchorCoverage": round(anchor_coverage, 6),
            "titleFacetCoverage": round(title_facet_coverage, 6),
            "titleTermCoverage": round(title_term_coverage, 6),
            "titleAnchorCoverage": round(title_anchor_coverage, 6),
            "citationRecencyPrior": round(citation_bonus, 6),
            "driftPenalty": round(drift_penalty, 6),
            "queryFacetPenalty": round(facet_penalty, 6),
            "finalScore": round(final_score, 6),
        }
        item["querySimilarity"] = query_similarity
        item["finalScore"] = final_score
    return sorted(merged_candidates, key=lambda item: item["finalScore"], reverse=True)


def evaluate_speculative_variants(
    *,
    ranked_candidates: list[dict[str, Any]],
    config: AgenticConfig,
) -> tuple[list[str], list[str], list[str]]:
    """Accept or reject speculative variants based on grounded contribution."""
    top_pool = ranked_candidates[: config.speculative_top_pool_cutoff]
    contributions: dict[str, int] = defaultdict(int)
    drift_rejections: set[str] = set()
    for item in top_pool:
        for variant in item["variants"]:
            if "speculative" in item["variantSources"]:
                if item.get("querySimilarity", 0.0) >= config.drift_similarity_threshold:
                    contributions[variant] += 1
                else:
                    drift_rejections.add(variant)

    accepted = sorted(
        variant
        for variant, count in contributions.items()
        if count >= config.speculative_accept_min_novel_papers and variant not in drift_rejections
    )
    rejected = sorted(variant for variant in drift_rejections if variant not in accepted)
    drift_warnings = [
        (
            f"Rejected speculative expansion '{variant}' because its "
            "retrieved papers drifted too far from the original concept."
        )
        for variant in rejected
    ]
    return accepted, rejected, drift_warnings


def _paper_text(paper: dict[str, Any]) -> str:
    authors = ", ".join(author.get("name", "") for author in (paper.get("authors") or []) if isinstance(author, dict))
    abstract = ""
    if paper.get("source") != "serpapi_google_scholar":
        abstract = str(paper.get("abstract") or "")
    return " ".join(
        part
        for part in [
            str(paper.get("title") or ""),
            abstract,
            str(paper.get("venue") or ""),
            str(paper.get("year") or ""),
            authors,
        ]
        if part
    )


def _merge_paper_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if value in (None, "", [], {}):
            continue
        current = merged.get(key)
        if current in (None, "", [], {}):
            merged[key] = value
            continue
        if isinstance(current, list) and isinstance(value, list):
            seen = {repr(item) for item in current}
            merged[key] = current + [item for item in value if repr(item) not in seen]
    return merged


def _first_author_name(paper: dict[str, Any]) -> str:
    authors = paper.get("authors") or []
    if authors and isinstance(authors[0], dict):
        return str(authors[0].get("name") or "")
    return ""


def _normalized_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _normalized_string(value: Any) -> str:
    return str(value or "").strip().lower()


def _looks_like_arxiv(value: str) -> bool:
    normalized = value.lower()
    return normalized.startswith("arxiv:") or bool(re.match(r"^\d{4}\.\d{4,5}(?:v\d+)?$", normalized))


def _paper_title_text(paper: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in [
            str(paper.get("title") or ""),
            str(paper.get("venue") or ""),
        ]
        if part
    )


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens = _tokenize_text(left)
    right_tokens = _tokenize_text(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    return intersection / math.sqrt(len(left_tokens) * len(right_tokens))


def _tokenize_text(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _paper_matches_facet(paper_tokens: set[str], facet: str) -> bool:
    facet_tokens = [token for token in re.findall(r"[a-z0-9]{3,}", facet.lower()) if token]
    if not facet_tokens:
        return False
    matched = sum(token in paper_tokens for token in facet_tokens)
    required = len(facet_tokens) if len(facet_tokens) <= 2 else 2
    return matched >= required


def _concept_matches_tokens(concept: str, paper_tokens: set[str]) -> bool:
    concept_tokens = [token for token in re.findall(r"[a-z0-9]{3,}", concept.lower()) if token]
    if not concept_tokens:
        return False
    if len(concept_tokens) == 1:
        return concept_tokens[0] in paper_tokens
    return _paper_matches_facet(paper_tokens, concept)


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        lowered = value.strip().lower()
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(value)
    return deduped
