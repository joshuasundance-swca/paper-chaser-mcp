"""Query understanding, routing, and bounded expansion helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal, cast
from urllib.parse import urlparse

from ..citation_repair import looks_like_citation_query
from .config import AgenticConfig
from .models import ExpansionCandidate, IntentCandidate, IntentLabel, PlannerDecision
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
LITERATURE_QUERY_TERMS = {
    "article",
    "citation",
    "doi",
    "evidence",
    "journal",
    "literature",
    "meta-analysis",
    "paper",
    "peer-reviewed",
    "review",
    "scholarly",
    "scientific",
    "study",
    "systematic review",
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
QUERYISH_TITLE_BLOCKERS = {
    "compare",
    "effects",
    "evidence",
    "history",
    "include",
    "listing",
    "regulatory",
    "review",
    "status",
    "studies",
    "study",
    "survey",
    "systematic",
    "what",
}
STRONG_REGULATORY_TITLE_BLOCKERS = {
    "critical habitat",
    "ecos",
    "esa",
    "federal register",
    "final rule",
    "listing status",
    "regulatory history",
    "rulemaking",
}
REGULATORY_QUERY_TERMS = {
    "biological opinion",
    "cfr",
    "code of federal regulations",
    "critical habitat",
    "ecos",
    "esa",
    "final rule",
    "federal register",
    "five-year review",
    "five year review",
    "incidental take",
    "listing status",
    "listing history",
    "proposed rule",
    "recovery plan",
    "regulation",
    "regulatory history",
    "rulemaking",
    "section 7",
    "species dossier",
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
    lowered = normalized.lower()
    if any(marker in lowered for marker in STRONG_REGULATORY_TITLE_BLOCKERS):
        return False
    if re.search(r"\b\d+\s*(?:cfr|f\.?\s*r\.?)\b", lowered):
        return False
    if any(phrase in lowered for phrase in ("what does", "what is", "what are", "include representative")):
        return False
    stripped = normalized.strip("\"'")
    words = re.findall(r"[A-Za-z][A-Za-z0-9'/-]*", stripped)
    if not 2 <= len(words) <= 24:
        return False
    significant_words = [word for word in words if len(word) > 2 and word.lower() not in TITLE_STOPWORDS]
    if len(significant_words) < 2:
        return False
    if sum(word.lower() in QUERYISH_TITLE_BLOCKERS for word in significant_words) >= 3:
        return False
    title_like_words = [word for word in significant_words if word[:1].isupper() or word.isupper() or "-" in word]
    if len(title_like_words) >= max(2, int(len(significant_words) * 0.45)):
        return True
    return bool(re.search(r"\([A-Z][A-Za-z]+(?:\s+[a-z][A-Za-z-]+)+\)", stripped)) and len(significant_words) >= 6


def detect_regulatory_intent(query: str, focus: str | None = None) -> bool:
    """Return True when the ask is more likely a regulatory primary-source workflow."""

    normalized = normalize_query(" ".join(part for part in [query, focus or ""] if part)).lower()
    if not normalized:
        return False
    if any(term in normalized for term in REGULATORY_QUERY_TERMS):
        return True
    if re.search(r"\b(?:endangered|threatened)\b", normalized) and any(
        marker in normalized
        for marker in {
            "cfr",
            "critical habitat",
            "esa",
            "federal register",
            "final rule",
            "listing",
            "recovery plan",
            "rulemaking",
            "species status",
        }
    ):
        return True
    if re.search(r"\b\d+\s*(?:cfr|f\.?\s*r\.?)\b", normalized):
        return True
    if "species" in normalized and any(
        marker in normalized for marker in {"history", "listing", "recovery", "dossier"}
    ):
        return True
    return False


def detect_literature_intent(query: str, focus: str | None = None) -> bool:
    """Return True when the ask explicitly signals literature or scholarly retrieval."""

    normalized = normalize_query(" ".join(part for part in [query, focus or ""] if part)).lower()
    if not normalized:
        return False
    if any(term in normalized for term in LITERATURE_QUERY_TERMS):
        return True
    return bool(re.search(r"\b(?:doi|peer-reviewed|systematic review|meta-analysis|scientific reports?)\b", normalized))


def _source_for_intent_candidate(
    intent_source: Literal["explicit", "planner", "heuristic_override", "hybrid_agreement", "fallback_recovery"],
) -> Literal["explicit", "planner", "heuristic", "hybrid", "fallback"]:
    mapping: dict[
        Literal["explicit", "planner", "heuristic_override", "hybrid_agreement", "fallback_recovery"],
        Literal["explicit", "planner", "heuristic", "hybrid", "fallback"],
    ] = {
        "explicit": "explicit",
        "planner": "planner",
        "heuristic_override": "heuristic",
        "hybrid_agreement": "hybrid",
        "fallback_recovery": "fallback",
    }
    return mapping[intent_source]


def _confidence_rank(confidence: Literal["high", "medium", "low"]) -> int:
    return {"high": 3, "medium": 2, "low": 1}[confidence]


def _upsert_intent_candidate(
    *,
    candidates: list[IntentCandidate],
    intent: IntentLabel,
    confidence: Literal["high", "medium", "low"],
    source: Literal["explicit", "planner", "heuristic", "hybrid", "fallback"],
    rationale: str,
) -> None:
    for index, existing in enumerate(candidates):
        if existing.intent != intent:
            continue
        merged_confidence = (
            confidence if _confidence_rank(confidence) >= _confidence_rank(existing.confidence) else existing.confidence
        )
        merged_source = existing.source
        if _confidence_rank(confidence) >= _confidence_rank(existing.confidence):
            merged_source = source
        merged_rationale = existing.rationale
        if rationale and rationale not in merged_rationale:
            merged_rationale = f"{merged_rationale} {rationale}".strip() if merged_rationale else rationale
        candidates[index] = existing.model_copy(
            update={
                "confidence": merged_confidence,
                "source": merged_source,
                "rationale": merged_rationale,
            }
        )
        return
    candidates.append(
        IntentCandidate(
            intent=intent,
            confidence=confidence,
            source=source,
            rationale=rationale,
        )
    )


def _sort_intent_candidates(
    candidates: list[IntentCandidate],
    *,
    preferred_intent: IntentLabel,
) -> list[IntentCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.intent != preferred_intent,
            -_confidence_rank(candidate.confidence),
            candidate.intent,
        ),
    )


def _strong_known_item_signal(normalized_query: str) -> bool:
    return bool(
        DOI_RE.search(normalized_query) or ARXIV_RE.search(normalized_query) or looks_like_url(normalized_query)
    )


def _strong_regulatory_signal(normalized_query: str, focus: str | None = None) -> bool:
    combined = normalize_query(" ".join(part for part in [normalized_query, focus or ""] if part)).lower()
    if re.search(r"\b\d+\s*(?:f\.?\s*r\.?)\s*\d+\b", combined):
        return True
    if re.search(r"\bfederal register\b", combined) and re.search(r"\b\d+\b", combined):
        return True
    if re.search(r"\b\d+\s*(?:cfr|f\.?\s*r\.?)\b", combined):
        return True
    return bool(re.search(r"\b\d{4}-\d{4,6}\b", combined))


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
    planner.intent_source = "planner"
    planner.intent_confidence = "medium"
    intent_candidates = list(planner.intent_candidates)
    _upsert_intent_candidate(
        candidates=intent_candidates,
        intent=planner.intent,
        confidence=planner.intent_confidence,
        source="planner",
        rationale="Model planner selected this intent as the best initial route.",
    )
    if mode != "auto":
        planner.intent = cast(
            Literal["discovery", "review", "known_item", "author", "citation", "regulatory"],
            mode,
        )
        planner.intent_source = "explicit"
        planner.intent_confidence = "high"
        planner.intent_rationale = "Intent was set explicitly by the caller."
        if mode == "review":
            planner.follow_up_mode = "claim_check"
        _upsert_intent_candidate(
            candidates=intent_candidates,
            intent=cast(IntentLabel, planner.intent),
            confidence="high",
            source="explicit",
            rationale="Explicit mode parameter from the caller.",
        )
    else:
        citation_like_signal = looks_like_citation_query(normalized)
        title_like_signal = looks_like_exact_title(query)
        literature_signal = detect_literature_intent(normalized, focus)
        strong_known_item_signal = _strong_known_item_signal(normalized)
        strong_regulatory_signal = _strong_regulatory_signal(normalized, focus)
        regulatory_signal = detect_regulatory_intent(normalized, focus)
        if strong_known_item_signal:
            _upsert_intent_candidate(
                candidates=intent_candidates,
                intent="known_item",
                confidence="high",
                source="heuristic",
                rationale="Strong known-item signal (DOI, arXiv, or URL) was detected.",
            )
        elif citation_like_signal:
            _upsert_intent_candidate(
                candidates=intent_candidates,
                intent="known_item",
                confidence="medium",
                source="heuristic",
                rationale="Citation-like wording suggests a known-item anchor.",
            )
        elif title_like_signal:
            _upsert_intent_candidate(
                candidates=intent_candidates,
                intent="known_item",
                confidence="low",
                source="heuristic",
                rationale="Title-like phrasing suggests a possible known-item lookup.",
            )

        if strong_regulatory_signal:
            _upsert_intent_candidate(
                candidates=intent_candidates,
                intent="regulatory",
                confidence="high",
                source="heuristic",
                rationale="Explicit regulatory citation or rulemaking marker was detected.",
            )
        elif regulatory_signal:
            _upsert_intent_candidate(
                candidates=intent_candidates,
                intent="regulatory",
                confidence="low",
                source="heuristic",
                rationale="Regulatory phrasing is present but not strongly anchored.",
            )
        if literature_signal:
            _upsert_intent_candidate(
                candidates=intent_candidates,
                intent="review",
                confidence="low",
                source="heuristic",
                rationale="Literature/review language suggests synthesis intent.",
            )

        heuristic_override: IntentLabel | None = None
        heuristic_override_confidence: Literal["high", "medium", "low"] = "medium"
        heuristic_rationale = ""
        known_item_override = strong_known_item_signal or (
            (citation_like_signal or title_like_signal) and not strong_regulatory_signal
        )
        if known_item_override:
            heuristic_override = "known_item"
            heuristic_override_confidence = "high" if strong_known_item_signal else "medium"
            heuristic_rationale = (
                "Strong known-item signal overrode planner routing."
                if strong_known_item_signal
                else "Known-item guardrails (citation/title pattern) overrode planner routing."
            )
        elif regulatory_signal:
            heuristic_override = "regulatory"
            heuristic_override_confidence = "high" if strong_regulatory_signal else "medium"
            heuristic_rationale = (
                "Strong regulatory signal overrode planner routing."
                if strong_regulatory_signal
                else "Regulatory guardrails overrode planner routing."
            )

        if heuristic_override is not None:
            if planner.intent == heuristic_override:
                planner.intent_source = "hybrid_agreement"
                planner.intent_confidence = "high"
                planner.intent_rationale = "Planner intent matched strong deterministic guardrail signals."
            else:
                planner.intent = heuristic_override
                planner.intent_source = "heuristic_override"
                planner.intent_confidence = heuristic_override_confidence
                planner.intent_rationale = heuristic_rationale
        elif planner.intent_rationale.strip() == "":
            planner.intent_rationale = "Planner intent selected without a strong deterministic override."

    _upsert_intent_candidate(
        candidates=intent_candidates,
        intent=cast(IntentLabel, planner.intent),
        confidence=planner.intent_confidence,
        source=_source_for_intent_candidate(planner.intent_source),
        rationale=planner.intent_rationale or "Final routed intent after planner and guardrail reconciliation.",
    )
    sorted_candidates = _sort_intent_candidates(intent_candidates, preferred_intent=cast(IntentLabel, planner.intent))
    planner.intent_candidates = sorted_candidates[:4]
    planner.secondary_intents = [
        candidate.intent for candidate in planner.intent_candidates if candidate.intent != planner.intent
    ][:3]
    primary_candidate = next(
        (candidate for candidate in planner.intent_candidates if candidate.intent == planner.intent),
        None,
    )
    planner.routing_confidence = (
        primary_candidate.confidence if primary_candidate is not None else planner.intent_confidence
    )
    if not planner.intent_rationale:
        planner.intent_rationale = "Intent routed from planner defaults."
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
