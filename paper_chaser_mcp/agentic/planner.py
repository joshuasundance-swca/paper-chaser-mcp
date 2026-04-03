"""Query understanding, routing, and bounded expansion helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal, cast
from urllib.parse import urlparse

from ..citation_repair import looks_like_citation_query
from .config import AgenticConfig
from .models import ExpansionCandidate, PlannerDecision
from .providers import COMMON_QUERY_WORDS, ModelProviderBundle

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
ARXIV_RE = re.compile(r"(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE)
FACET_SPLIT_RE = re.compile(
    r"\b(?:for|in|on|about|into|within|across|via|through|regarding|around)\b",
    re.IGNORECASE,
)
GENERIC_EVIDENCE_WORDS = COMMON_QUERY_WORDS | {
    "with",
    "were",
    "from",
    "into",
    "using",
    "use",
    "used",
    "their",
    "this",
    "that",
    "these",
    "those",
    "they",
    "them",
    "within",
    "across",
    "based",
    "approach",
    "approaches",
    "method",
    "methods",
    "analysis",
    "results",
    "finding",
    "findings",
    "quality",
    "different",
}
QUERY_FACET_TOKEN_ALLOWLIST = {
    "agent",
    "agents",
    "review",
    "reviews",
    "survey",
    "surveys",
    "tool",
    "tools",
}
TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
REGULATORY_QUERY_TERMS = {
    "biological opinion",
    "cfr",
    "code of federal regulations",
    "critical habitat",
    "ecos",
    "endangered",
    "federal register",
    "five-year review",
    "five year review",
    "incidental take",
    "listing history",
    "proposed rule",
    "recovery plan",
    "regulation",
    "regulatory history",
    "rulemaking",
    "section 7",
    "species dossier",
    "threatened",
}

VARIANT_DEDUPE_STOPWORDS = (
    TITLE_STOPWORDS
    | GENERIC_EVIDENCE_WORDS
    | {
        "florida",
        "review",
        "scrub",
    }
)


def normalize_query(query: str) -> str:
    """Collapse whitespace and keep the original wording intact."""
    return " ".join(query.strip().split())


def query_facets(query: str) -> list[str]:
    """Extract compact multi-token facets that should stay visible in results."""
    normalized = re.sub(r"[-_/]+", " ", normalize_query(query).lower())
    facets: list[str] = []
    seen: set[str] = set()
    for segment in FACET_SPLIT_RE.split(normalized):
        tokens = [
            token
            for token in re.findall(r"[A-Za-z0-9]{3,}", segment)
            if (token not in GENERIC_EVIDENCE_WORDS or token in QUERY_FACET_TOKEN_ALLOWLIST)
        ]
        if len(tokens) >= 2:
            facet = " ".join(tokens[:3])
            if facet not in seen:
                seen.add(facet)
                facets.append(facet)
    if facets:
        return facets[:3]

    fallback_tokens = [
        token for token in re.findall(r"[A-Za-z0-9]{3,}", normalized) if token not in GENERIC_EVIDENCE_WORDS
    ]
    for token in fallback_tokens[:3]:
        if token not in seen:
            seen.add(token)
            facets.append(token)
    return facets


def query_terms(query: str) -> list[str]:
    """Return distinctive normalized query terms for lightweight coverage checks."""
    normalized = re.sub(r"[-_/]+", " ", normalize_query(query).lower())
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9]{3,}", normalized):
        if token in seen:
            continue
        if token in GENERIC_EVIDENCE_WORDS and token not in QUERY_FACET_TOKEN_ALLOWLIST:
            continue
        seen.add(token)
        terms.append(token)
    return terms[:8]


def looks_like_url(query: str) -> bool:
    """Return True when *query* is a plausible URL."""
    if not query:
        return False
    parsed = urlparse(query)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def looks_like_exact_title(query: str) -> bool:
    """Heuristically identify likely exact-title lookups."""
    normalized = normalize_query(query)
    if not normalized or normalized.endswith("?"):
        return False
    stripped = normalized.strip("\"'")
    words = re.findall(r"[A-Za-z][A-Za-z0-9'/-]*", stripped)
    if not 2 <= len(words) <= 14:
        return False
    significant_words = [word for word in words if len(word) > 2 and word.lower() not in TITLE_STOPWORDS]
    if len(significant_words) < 2:
        return False
    title_like_words = [word for word in significant_words if word[:1].isupper() or word.isupper() or "-" in word]
    return len(title_like_words) >= max(2, int(len(significant_words) * 0.6))


def detect_regulatory_intent(query: str, focus: str | None = None) -> bool:
    """Return True when the ask is more likely a regulatory primary-source workflow."""

    normalized = normalize_query(" ".join(part for part in [query, focus or ""] if part)).lower()
    if not normalized:
        return False
    if any(term in normalized for term in REGULATORY_QUERY_TERMS):
        return True
    if re.search(r"\b\d+\s*(?:cfr|f\.?\s*r\.?)\b", normalized):
        return True
    if "species" in normalized and any(
        marker in normalized for marker in {"history", "listing", "recovery", "dossier"}
    ):
        return True
    return False


async def classify_query(
    *,
    query: str,
    mode: str,
    year: str | None,
    venue: str | None,
    focus: str | None,
    provider_bundle: ModelProviderBundle,
    request_outcomes: list[dict[str, Any]] | None = None,
    request_id: str | None = None,
) -> tuple[str, PlannerDecision]:
    """Normalize and classify a smart-search request."""
    normalized = normalize_query(query)
    planner = await provider_bundle.aplan_search(
        query=normalized,
        mode=mode,
        year=year,
        venue=venue,
        focus=focus,
        request_outcomes=request_outcomes,
        request_id=request_id,
    )
    if mode != "auto":
        planner.intent = cast(
            Literal["discovery", "review", "known_item", "author", "citation", "regulatory"],
            mode,
        )
        if mode == "review":
            planner.follow_up_mode = "claim_check"
    else:
        if detect_regulatory_intent(normalized, focus):
            planner.intent = "regulatory"
        elif (
            DOI_RE.search(normalized)
            or ARXIV_RE.search(normalized)
            or looks_like_url(normalized)
            or looks_like_exact_title(query)
            or looks_like_citation_query(normalized)
        ):
            planner.intent = "known_item"
    merged_concepts = list(planner.candidate_concepts)
    merged_concepts.extend(query_facets(normalized))
    if focus:
        merged_concepts.extend(query_facets(focus))
    deduped_concepts: list[str] = []
    seen_concepts: set[str] = set()
    for concept in merged_concepts:
        lowered = concept.strip().lower()
        if not lowered or lowered in seen_concepts:
            continue
        seen_concepts.add(lowered)
        deduped_concepts.append(concept.strip())
    planner.candidate_concepts = deduped_concepts[:8]
    return normalized, planner


def grounded_expansion_candidates(
    *,
    original_query: str,
    papers: list[dict[str, Any]],
    config: AgenticConfig,
    focus: str | None = None,
    venue: str | None = None,
    year: str | None = None,
) -> list[ExpansionCandidate]:
    """Create grounded query variants from retrieved evidence only."""
    variants: list[ExpansionCandidate] = []
    base_query = normalize_query(original_query)
    suffixes = [item for item in [focus, venue, year] if item]
    if suffixes:
        variants.append(
            ExpansionCandidate(
                variant=" ".join([base_query, *suffixes]),
                source="from_input",
                rationale=("Adds explicit user-provided constraints to the literal query."),
            )
        )

    phrase_candidates = _top_evidence_phrases(papers, limit=config.max_grounded_variants * 3)
    query_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", base_query.lower()))
    for phrase in phrase_candidates:
        lowered_query = base_query.lower()
        if phrase.lower() in lowered_query:
            continue
        phrase_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", phrase.lower()))
        if len([token for token in phrase_tokens if token not in query_tokens]) < 2:
            continue
        variants.append(
            ExpansionCandidate(
                variant=f"{base_query} {phrase}",
                source="from_retrieved_evidence",
                rationale=(f"Evidence from the first-pass results repeatedly mentioned '{phrase}'."),
            )
        )
        if len(variants) >= config.max_grounded_variants:
            break

    deduped: list[ExpansionCandidate] = []
    seen: set[str] = set()
    seen_signatures: list[frozenset[str]] = []
    for candidate in variants:
        lowered = candidate.variant.lower()
        if lowered in seen:
            continue
        signature = _variant_signature(candidate.variant)
        if any(_signatures_are_near_duplicates(signature, prior) for prior in seen_signatures):
            continue
        seen.add(lowered)
        seen_signatures.append(signature)
        deduped.append(candidate)
    return deduped[: config.max_grounded_variants]


async def speculative_expansion_candidates(
    *,
    original_query: str,
    papers: list[dict[str, Any]],
    config: AgenticConfig,
    provider_bundle: ModelProviderBundle,
    request_outcomes: list[dict[str, Any]] | None = None,
    request_id: str | None = None,
) -> list[ExpansionCandidate]:
    """Generate bounded speculative variants through the provider bundle."""
    evidence_texts = [
        " ".join(
            part
            for part in [
                str(paper.get("title") or ""),
                str(paper.get("abstract") or ""),
            ]
            if part
        )
        for paper in papers[:8]
    ]
    return (
        await provider_bundle.asuggest_speculative_expansions(
            query=normalize_query(original_query),
            evidence_texts=[text for text in evidence_texts if text],
            max_variants=config.max_speculative_variants,
            request_outcomes=request_outcomes,
            request_id=request_id,
        )
    )[: config.max_speculative_variants]


def combine_variants(
    *,
    original_query: str,
    grounded: list[ExpansionCandidate],
    speculative: list[ExpansionCandidate],
    config: AgenticConfig,
) -> list[ExpansionCandidate]:
    """Return the capped variant list in grounded-first order."""
    variants = [
        ExpansionCandidate(
            variant=normalize_query(original_query),
            source="from_input",
            rationale="Literal user query.",
        )
    ]
    variants.extend(grounded[: config.max_grounded_variants])
    variants.extend(speculative[: config.max_speculative_variants])

    deduped: list[ExpansionCandidate] = []
    seen: set[str] = set()
    seen_signatures: list[frozenset[str]] = []
    for candidate in variants:
        lowered = candidate.variant.lower()
        if lowered in seen:
            continue
        signature = _variant_signature(candidate.variant)
        if any(_signatures_are_near_duplicates(signature, prior) for prior in seen_signatures):
            continue
        seen.add(lowered)
        seen_signatures.append(signature)
        deduped.append(candidate)
        if len(deduped) >= config.max_total_variants:
            break
    return deduped


def dedupe_variants(
    candidates: list[ExpansionCandidate],
    *,
    config: AgenticConfig,
) -> list[ExpansionCandidate]:
    """Apply the same near-duplicate suppression used by final variant planning."""

    deduped: list[ExpansionCandidate] = []
    seen: set[str] = set()
    seen_signatures: list[frozenset[str]] = []
    for candidate in candidates:
        lowered = candidate.variant.lower()
        if lowered in seen:
            continue
        signature = _variant_signature(candidate.variant)
        if any(_signatures_are_near_duplicates(signature, prior) for prior in seen_signatures):
            continue
        seen.add(lowered)
        seen_signatures.append(signature)
        deduped.append(candidate)
        if len(deduped) >= config.max_total_variants:
            break
    return deduped


def _variant_signature(text: str) -> frozenset[str]:
    tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9]{3,}", normalize_query(text).lower())
        if token not in VARIANT_DEDUPE_STOPWORDS
    }
    return frozenset(tokens)


def _signatures_are_near_duplicates(
    left: frozenset[str],
    right: frozenset[str],
) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    overlap = len(left & right)
    shorter = min(len(left), len(right))
    longer = max(len(left), len(right))
    if shorter == 0:
        return False
    coverage = overlap / shorter
    jaccard = overlap / len(left | right)
    return coverage >= 0.8 or (coverage >= 0.67 and jaccard >= 0.5 and longer <= 8)


def _top_evidence_phrases(
    papers: list[dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    phrases: Counter[str] = Counter()
    for paper in papers[:10]:
        title = str(paper.get("title") or "")
        per_paper_phrases: set[str] = set()
        title_words = re.findall(r"[A-Za-z0-9]{2,}", title.lower())
        for index in range(len(title_words) - 1):
            left = title_words[index]
            right = title_words[index + 1]
            if left in GENERIC_EVIDENCE_WORDS or right in GENERIC_EVIDENCE_WORDS or len(left) < 3 or len(right) < 3:
                continue
            bigram = f"{left} {right}"
            if len(bigram) < 9:
                continue
            per_paper_phrases.add(bigram)
        phrases.update(per_paper_phrases)
    scored = [(phrase, count) for phrase, count in phrases.items() if count >= 2]
    scored.sort(
        key=lambda item: (
            item[0].count(" "),
            item[1],
            len(item[0]),
        ),
        reverse=True,
    )
    return [phrase for phrase, _ in scored[:limit]]
