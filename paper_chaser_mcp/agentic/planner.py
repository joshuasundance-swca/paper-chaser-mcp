"""Query understanding, routing, and bounded expansion helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal, cast
from urllib.parse import urlparse

from ..citation_repair import looks_like_citation_query
from .config import AgenticConfig
from .models import (
    RETRIEVAL_MODE_MIXED,
    RETRIEVAL_MODE_TARGETED,
    ExpansionCandidate,
    IntentCandidate,
    IntentLabel,
    PlannerDecision,
    PlannerQueryType,
    RegulatoryIntentLabel,
)
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
HYPOTHESIS_QUERY_STOPWORDS = {
    "current",
    "different",
    "effective",
    "effectiveness",
    "especially",
    "evidence",
    "field",
    "latest",
    "methods",
    "most",
    "recent",
    "research",
    "review",
    "studies",
    "study",
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
    "aquatic",
    "bioaccumulation",
    "biodiversity",
    "climate",
    "compare",
    "contamination",
    "ecotoxicology",
    "ecosystem",
    "effects",
    "evidence",
    "exposure",
    "history",
    "include",
    "listing",
    "marine",
    "mitigation",
    "monitoring",
    "pollution",
    "regulatory",
    "review",
    "status",
    "studies",
    "study",
    "survey",
    "systematic",
    "terrestrial",
    "toxicity",
    "transport",
    "trophic",
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
AGENCY_REGULATORY_MARKERS = {
    "agency",
    "cdc",
    "cms",
    "epa",
    "fda",
    "food and drug administration",
    "hhs",
    "nih",
    "usda",
}
REGULATORY_QUERY_TERMS = {
    "106 consultation",
    "ac hp",
    "agency guidance",
    "archaeology",
    "archaeology guidance",
    "biological opinion",
    "cfr",
    "clinical decision support",
    "code of federal regulations",
    "contaminant limit",
    "critical habitat",
    "drinking water standard",
    "ecos",
    "esa",
    "fda",
    "final rule",
    "food and drug administration",
    "federal register",
    "five-year review",
    "five year review",
    "guidance for industry",
    "health advisory",
    "historic district",
    "historic preservation",
    "incidental take",
    "listing status",
    "listing history",
    "maximum contaminant level",
    "mcl",
    "nhpa",
    "proposed rule",
    "recovery plan",
    "regulation",
    "regulatory history",
    "rulemaking",
    "safe drinking water act",
    "sdwa",
    "section 106",
    "section 7",
    "species dossier",
    "tribal consultation",
    "thpo",
    "shpo",
    "sacred site",
    "cultural resources",
    "cultural landscape",
}

_CULTURAL_RESOURCE_MARKERS = {
    "archaeological",
    "archaeology",
    "cultural resources",
    "cultural landscape",
    "historic district",
    "historic preservation",
    "historic property",
    "nhpa",
    "sacred site",
    "section 106",
    "tribal consultation",
    "thpo",
    "shpo",
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
    if detect_regulatory_intent(normalized):
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
    queryish_count = sum(word.lower() in QUERYISH_TITLE_BLOCKERS for word in significant_words)
    if queryish_count >= 3:
        return False
    title_like_words = [word for word in significant_words if word[:1].isupper() or word.isupper() or "-" in word]
    if len(title_like_words) >= max(2, int(len(significant_words) * 0.45)):
        return True
    if (
        queryish_count == 0
        and not _query_starts_broad(normalized)
        and 4 <= len(significant_words) <= 12
        and any(word.lower() in TITLE_STOPWORDS for word in words)
    ):
        return True
    return bool(re.search(r"\([A-Z][A-Za-z]+(?:\s+[a-z][A-Za-z-]+)+\)", stripped)) and len(significant_words) >= 6


def looks_like_near_known_item_query(query: str) -> bool:
    """Heuristically identify short near-known-item prompts.

    This catches prompts like model/paper/system names (for example CLIP,
    Toolformer, DSPy, or Med-PaLM M) that are not exact-title lookups but still
    behave like anchored known-item requests for topical-relevance purposes.
    """

    normalized = normalize_query(query)
    if not normalized or normalized.endswith("?"):
        return False
    if detect_regulatory_intent(normalized):
        return False
    lowered = normalized.lower()
    if any(marker in lowered for marker in STRONG_REGULATORY_TITLE_BLOCKERS):
        return False
    if any(
        phrase in lowered
        for phrase in (
            "what is",
            "what are",
            "how does",
            "how do",
            "compare",
            "survey",
            "review",
            "literature",
            "state of the art",
            "overview",
            "introduction",
            "tutorial",
        )
    ):
        return False
    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+./:-]*", query.strip("\"' "))
    terms = query_terms(normalized)
    if not raw_tokens or not terms or len(terms) > 4 or len(raw_tokens) > 5:
        return False
    queryish_count = sum(term.lower() in QUERYISH_TITLE_BLOCKERS for term in terms)
    if queryish_count >= 2:
        return False

    def _identifierish(token: str) -> bool:
        stripped = token.strip()
        if len(stripped) <= 1:
            return stripped.isupper()
        upper_count = sum(char.isupper() for char in stripped)
        lower_count = sum(char.islower() for char in stripped)
        return bool(
            stripped.isupper()
            or "-" in stripped
            or any(char.isdigit() for char in stripped)
            or re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", stripped)
            or (upper_count >= 2 and lower_count >= 1)
        )

    if any(_identifierish(token) for token in raw_tokens):
        return True
    return len(raw_tokens) == 1 and raw_tokens[0][:1].isupper() and len(raw_tokens[0]) >= 5


def detect_regulatory_intent(query: str, focus: str | None = None) -> bool:
    """Return True when the ask is more likely a regulatory primary-source workflow."""

    normalized = normalize_query(" ".join(part for part in [query, focus or ""] if part)).lower()
    if not normalized:
        return False
    if any(term in normalized for term in REGULATORY_QUERY_TERMS):
        return True
    if any(term in normalized for term in _CULTURAL_RESOURCE_MARKERS):
        return True
    if any(term in normalized for term in {"guidance", "policy", "policies"}) and any(
        marker in normalized for marker in AGENCY_REGULATORY_MARKERS
    ):
        return True
    if any(marker in normalized for marker in {"esa", "listing status", "listing history", "final rule"}) and any(
        marker in normalized
        for marker in {
            "bat",
            "bird",
            "condor",
            "habitat",
            "listed",
            "listing",
            "recovery",
            "species",
            "status",
            "threatened",
            "wildlife",
        }
    ):
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
    if "scholarship" in normalized:
        return True
    return bool(re.search(r"\b(?:doi|peer-reviewed|systematic review|meta-analysis|scientific reports?)\b", normalized))


def _detect_cultural_resource_intent(query: str, focus: str | None = None) -> bool:
    normalized = normalize_query(" ".join(part for part in [query, focus or ""] if part)).lower()
    return bool(normalized) and any(term in normalized for term in _CULTURAL_RESOURCE_MARKERS)


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


def _query_starts_broad(query: str) -> bool:
    lowered = normalize_query(query).lower()
    return lowered.startswith(("what ", "which ", "how ", "compare ", "summarize ", "identify ", "find "))


_DEFINITIONAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhat\s+(is|are|does)\b"),
    re.compile(r"\bdefine\b"),
    re.compile(r"\bexplain\b"),
    re.compile(r"\boverview\s+of\b"),
    re.compile(r"\bintroduction\s+to\b"),
    re.compile(r"\bguide\s+to\b"),
    re.compile(r"\bprimer\s+on\b"),
)


def _is_definitional_query(query: str) -> bool:
    """Return True when the query asks for a definition, overview, or primer.

    Used to bias retrieval and ranking toward canonical/foundational papers and
    survey articles when the user is seeking conceptual grounding rather than a
    specific result.
    """

    normalized = normalize_query(query or "").lower()
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _DEFINITIONAL_PATTERNS)


def _infer_regulatory_subintent(query: str, focus: str | None = None) -> str | None:
    normalized = normalize_query(" ".join(part for part in [query, focus or ""] if part)).lower()
    if not normalized or not detect_regulatory_intent(query, focus):
        return None
    if "cfr" in normalized and any(
        marker in normalized for marker in {"current text", "codified text", "what does", "under"}
    ):
        return "current_cfr_text"
    if any(marker in normalized for marker in {"guidance", "guideline", "handbook", "manual"}):
        return "guidance_lookup"
    if any(
        marker in normalized for marker in {"species dossier", "recovery plan", "critical habitat", "species profile"}
    ):
        return "species_dossier"
    if detect_literature_intent(query, focus) and (
        _detect_cultural_resource_intent(query, focus)
        or any(
            marker in normalized
            for marker in {"history", "timeline", "rulemaking", "final rule", "proposed rule", "listing history"}
        )
    ):
        return "hybrid_regulatory_plus_literature"
    if any(
        marker in normalized
        for marker in {"history", "timeline", "rulemaking", "final rule", "proposed rule", "listing history"}
    ):
        return "rulemaking_history"
    if detect_literature_intent(query, focus):
        return "hybrid_regulatory_plus_literature"
    return None


_VALID_REGULATORY_INTENTS: frozenset[str] = frozenset(
    {
        "current_cfr_text",
        "rulemaking_history",
        "species_dossier",
        "guidance_lookup",
        "hybrid_regulatory_plus_literature",
    },
)


def _has_literature_corroboration(
    *,
    planner: PlannerDecision,
    query: str,
    focus: str | None,
) -> bool:
    """Return True when the query-side signals support a hybrid regulatory+literature label.

    The LLM planner occasionally emits ``hybrid_regulatory_plus_literature``
    for regulation-only asks (e.g. "what does EPA require for stormwater
    discharges?"). Trusting that label alone causes the guided workflow to
    tack on an unnecessary literature review pass. This helper checks the
    independent keyword signal on the query, planner-emitted secondary
    intents, and retrieval hypotheses so downstream consumers can demand
    corroboration before honoring the hybrid route.

    Kept in sync with ``dispatch._guided_should_add_review_pass`` so valid
    hybrid labels are not stripped at planner-time only to be accepted at
    dispatch-time (or vice versa).
    """
    if detect_literature_intent(query, focus):
        return True
    for secondary in planner.secondary_intents:
        label = str(secondary).strip().lower()
        if label in {"review", "literature"}:
            return True
    literature_hypothesis_markers = (
        "literature",
        "peer-review",
        "peer review",
        "peer-reviewed",
        "systematic review",
        "meta-analysis",
        "hybrid_policy_science",
    )
    for hypothesis in planner.retrieval_hypotheses:
        lowered = str(hypothesis).lower()
        if any(marker in lowered for marker in literature_hypothesis_markers):
            return True
    return False


def _derive_regulatory_intent(
    *,
    planner: PlannerDecision,
    query: str,
    focus: str | None,
) -> RegulatoryIntentLabel | None:
    """Map LLM planner signals + deterministic cues to a canonical regulatory intent.

    LLM-first: the planner's own ``regulatory_subintent`` (set from its
    ``queryType``/``retrievalHypotheses``/``intent`` fields when available) is
    the preferred source. Deterministic heuristics only fire when the planner
    did not emit a canonical label. When the query is clearly regulatory but
    none of the specific labels match, returns ``"unspecified"``. Returns
    ``None`` for non-regulatory queries.
    """

    existing = planner.regulatory_subintent
    if existing and existing in _VALID_REGULATORY_INTENTS:
        if existing == "hybrid_regulatory_plus_literature" and not _has_literature_corroboration(
            planner=planner, query=query, focus=focus
        ):
            # Fall through to deterministic inference when the LLM emits a
            # hybrid label without any query-side literature cue.
            pass
        else:
            return cast(RegulatoryIntentLabel, existing)

    is_regulatory = planner.intent == "regulatory" or detect_regulatory_intent(query, focus)
    if not is_regulatory:
        return None

    # Consider hybrid literature + regulatory cues from retrieval hypotheses
    # and search angles the LLM already produced.
    llm_signal_text = " ".join(
        str(entry).lower()
        for bucket in (planner.retrieval_hypotheses, planner.search_angles, planner.candidate_concepts)
        for entry in bucket
    )
    if any(
        marker in llm_signal_text
        for marker in (
            "policy and literature",
            "regulatory and literature",
            "scientific and regulatory",
            "hybrid",
        )
    ) and detect_literature_intent(query, focus):
        return "hybrid_regulatory_plus_literature"

    inferred = _infer_regulatory_subintent(query, focus)
    if inferred and inferred in _VALID_REGULATORY_INTENTS:
        return cast(RegulatoryIntentLabel, inferred)
    return "unspecified"


def _infer_entity_card(query: str, focus: str | None = None) -> dict[str, str] | None:
    normalized = normalize_query(" ".join(part for part in [query, focus or ""] if part))
    lowered = normalized.lower()
    if not normalized:
        return None

    authority_context = None
    if "esa" in lowered or "endangered species" in lowered:
        authority_context = "ESA"
    elif "epa" in lowered or "safe drinking water act" in lowered or "sdwa" in lowered:
        authority_context = "EPA/SDWA"
    elif "fda" in lowered:
        authority_context = "FDA"
    elif "nhpa" in lowered or "section 106" in lowered or _detect_cultural_resource_intent(query, focus):
        authority_context = "NHPA/Section 106"

    requested_document_family = None
    if "critical habitat" in lowered:
        requested_document_family = "critical_habitat"
    elif "recovery plan" in lowered:
        requested_document_family = "recovery_plan"
    elif any(marker in lowered for marker in {"listing", "final rule", "proposed rule", "rulemaking"}):
        requested_document_family = "listing_rule"
    elif "guidance" in lowered:
        requested_document_family = "guidance"
    elif _detect_cultural_resource_intent(query, focus):
        requested_document_family = "consultation_or_preservation"

    scientific_match = re.search(r"\b([A-Z][a-z]+\s+[a-z]{3,})\b", query)
    scientific_name = scientific_match.group(1) if scientific_match else None

    cultural_subject_match = re.search(
        (
            r"\b([A-Z][A-Za-z0-9'’.-]+(?:\s+[A-Z][A-Za-z0-9'’.-]+){0,5}"
            r"\s+(?:Historic District|National Monument|Cultural Landscape|Sacred Site))\b"
        ),
        query,
    )

    common_name = None
    if cultural_subject_match:
        common_name = cultural_subject_match.group(1).strip()
    subject_match = re.search(
        r"\b(?:for|about)\s+(?:the\s+)?([a-z][a-z-]+(?:\s+[a-z][a-z-]+){0,3})\s+(?:under|with|in|on)\b",
        lowered,
    )
    if common_name is None and subject_match:
        candidate_subject = subject_match.group(1).strip()
        if candidate_subject and candidate_subject not in {"the species", "species dossier", "regulatory history"}:
            common_name = candidate_subject
    lowered_tokens = [token for token in re.findall(r"[a-z]{3,}", lowered) if token not in REGULATORY_QUERY_TERMS]
    species_markers = {
        "bat",
        "bird",
        "condor",
        "dolphin",
        "fish",
        "frog",
        "habitat",
        "owl",
        "species",
        "tortoise",
        "wolf",
    }
    if common_name is None:
        for index, token in enumerate(lowered_tokens):
            if token in species_markers and index > 0:
                previous = lowered_tokens[index - 1]
                if previous not in {"the", "for", "about", "under"}:
                    common_name = f"{previous} {token}"
                    break
    if common_name is None and len(lowered_tokens) >= 2 and detect_regulatory_intent(query, focus):
        common_name = " ".join(lowered_tokens[:2])

    if not any([common_name, scientific_name, authority_context, requested_document_family]):
        return None
    card: dict[str, str] = {}
    if common_name:
        card["commonName"] = common_name
    if scientific_name:
        card["scientificName"] = scientific_name
    if authority_context:
        card["authorityContext"] = authority_context
    if requested_document_family:
        card["requestedDocumentFamily"] = requested_document_family
    if _detect_cultural_resource_intent(query, focus):
        card["subjectArea"] = "cultural_resources"
        card["documentFamily"] = "consultation_or_preservation"
    return card


def _looks_broad_concept_query(
    *,
    normalized_query: str,
    focus: str | None,
    year: str | None,
    venue: str | None,
    terms: list[str] | None = None,
) -> bool:
    terms = terms if terms is not None else query_terms(normalized_query)
    has_constraints = bool(focus or year or venue)
    queryish_term_count = sum(term in QUERYISH_TITLE_BLOCKERS for term in terms)
    if _query_starts_broad(normalized_query) and len(terms) >= 6 and not has_constraints:
        return True
    if queryish_term_count >= 3 and len(terms) >= 6:
        return True
    if queryish_term_count >= 2 and len(terms) >= 8 and not has_constraints:
        return True
    return False


def _estimate_query_specificity(
    *,
    normalized_query: str,
    focus: str | None,
    year: str | None,
    venue: str | None,
    planner_query_type: PlannerQueryType | None = None,
    planner_specificity: Literal["high", "medium", "low"] | None = None,
) -> Literal["high", "medium", "low"]:
    """Estimate how specific a query is.

    When the LLM-authored ``planner_query_type`` or ``planner_specificity`` are
    provided we prefer them over raw text heuristics. The title/citation
    regex-based "high" promotion is suppressed whenever the LLM signalled a
    broad-concept query or already chose ``low`` specificity — this avoids the
    "long conceptual question happens to look title-like" false-positive that
    used to force those queries into known-item recovery.
    """
    terms = query_terms(normalized_query)
    broad_concept_signal = _looks_broad_concept_query(
        normalized_query=normalized_query,
        focus=focus,
        year=year,
        venue=venue,
        terms=terms,
    )
    if _strong_known_item_signal(normalized_query) or _strong_regulatory_signal(normalized_query, focus):
        return "high"
    llm_disagrees_with_title_heuristic = planner_query_type == "broad_concept" or planner_specificity == "low"
    if (
        not broad_concept_signal
        and not llm_disagrees_with_title_heuristic
        and (looks_like_exact_title(normalized_query) or looks_like_citation_query(normalized_query))
    ):
        return "high"
    has_constraints = bool(focus or year or venue)
    if broad_concept_signal:
        return "low"
    if planner_specificity is not None:
        # Honor an explicit low/high label from the planner whenever we have
        # not already short-circuited above.
        if planner_specificity == "low":
            return "low"
        if planner_specificity == "high":
            return "high"
    if has_constraints and len(terms) <= 5:
        return "high"
    if _query_starts_broad(normalized_query) and len(terms) >= 6:
        return "low"
    return "medium"


def _estimate_ambiguity_level(
    *,
    candidates: list[IntentCandidate],
    routing_confidence: Literal["high", "medium", "low"],
    query_specificity: Literal["high", "medium", "low"],
) -> Literal["low", "medium", "high"]:
    if routing_confidence == "low":
        return "high"
    if len(candidates) < 2:
        return "medium" if query_specificity == "low" else "low"
    primary_rank = _confidence_rank(candidates[0].confidence)
    secondary_rank = _confidence_rank(candidates[1].confidence)
    if secondary_rank >= primary_rank:
        return "high"
    if primary_rank - secondary_rank == 1:
        return "high" if query_specificity == "low" else "medium"
    if query_specificity == "low" and secondary_rank >= 1:
        return "medium"
    return "low"


def _ordered_provider_plan(base_plan: list[str], preferred_order: list[str]) -> list[str]:
    ordered = [provider for provider in preferred_order if provider in base_plan]
    ordered.extend(provider for provider in base_plan if provider not in ordered)
    return ordered


def initial_retrieval_hypotheses(
    *,
    normalized_query: str,
    focus: str | None,
    planner: PlannerDecision,
    config: AgenticConfig,
) -> list[ExpansionCandidate]:
    base_query = normalize_query(" ".join(part for part in [normalized_query, focus or ""] if part))
    if not base_query:
        base_query = normalized_query
    base_plan = list(planner.provider_plan)
    candidates: list[ExpansionCandidate] = [
        ExpansionCandidate(
            variant=base_query,
            source="from_input",
            rationale="Literal user query.",
            providerPlan=base_plan,
        )
    ]
    if planner.intent in {"known_item", "author", "citation", "regulatory"}:
        return candidates
    if planner.first_pass_mode == RETRIEVAL_MODE_TARGETED:
        return candidates

    planner_angles = [str(angle).strip() for angle in planner.search_angles if str(angle).strip()]
    if _is_definitional_query(normalized_query):
        definitional_angles = [f"{base_query} survey", f"{base_query} foundational paper"]
        planner_angles = definitional_angles + [angle for angle in planner_angles if angle not in definitional_angles]
    if not planner_angles:
        return candidates

    max_extra = max(config.max_initial_hypotheses - 1, 0)
    if planner.first_pass_mode == RETRIEVAL_MODE_MIXED:
        planner_angles = planner_angles[: max_extra or 1]
    else:
        planner_angles = planner_angles[:max_extra]

    seen_variants: set[str] = {base_query.lower()}
    preferred_plans = [
        _ordered_provider_plan(base_plan, ["openalex", "semantic_scholar", "core", "arxiv", "scholarapi"]),
        _ordered_provider_plan(base_plan, ["semantic_scholar", "openalex", "scholarapi", "core", "arxiv"]),
        _ordered_provider_plan(base_plan, ["core", "openalex", "semantic_scholar", "arxiv", "scholarapi"]),
        _ordered_provider_plan(base_plan, ["arxiv", "core", "openalex", "semantic_scholar", "scholarapi"]),
    ]
    for index, angle in enumerate(planner_angles):
        lowered = normalize_query(angle).lower()
        if not lowered or lowered in seen_variants:
            continue
        seen_variants.add(lowered)
        candidates.append(
            ExpansionCandidate(
                variant=angle,
                source="hypothesis",
                rationale="Planner-generated retrieval angle.",
                providerPlan=preferred_plans[min(index, len(preferred_plans) - 1)],
            )
        )
    return candidates[: max(config.max_initial_hypotheses, 1)]


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
    if "guidance" in combined and any(
        marker in combined for marker in ("fda", "food and drug administration", "agency", "guidance for industry")
    ):
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
    # Snapshot LLM-originated grounding signals BEFORE any deterministic
    # fallback mutates ``planner.entity_card`` / ``planner.candidate_concepts``
    # / ``planner.subject_card``. ``resolve_subject_card`` needs this to tell
    # apart "LLM emitted phase-4 signals" from "deterministic extractor
    # populated these fields after the LLM returned nothing".
    llm_emitted_grounding_signals = bool(planner.entity_card or planner.candidate_concepts or planner.subject_card)
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
        strong_known_item_signal = _strong_known_item_signal(normalized)
        strong_regulatory_signal = _strong_regulatory_signal(normalized, focus)
        if strong_known_item_signal:
            _upsert_intent_candidate(
                candidates=intent_candidates,
                intent="known_item",
                confidence="high",
                source="heuristic",
                rationale="Strong known-item signal (DOI, arXiv, or URL) was detected.",
            )
        if strong_regulatory_signal:
            _upsert_intent_candidate(
                candidates=intent_candidates,
                intent="regulatory",
                confidence="high",
                source="heuristic",
                rationale="Explicit regulatory citation or rulemaking marker was detected.",
            )

        heuristic_override: IntentLabel | None = None
        heuristic_override_confidence: Literal["high", "medium", "low"] = "high"
        heuristic_rationale = ""
        if strong_known_item_signal:
            heuristic_override = "known_item"
            heuristic_rationale = "Strong known-item signal overrode planner routing."
        elif strong_regulatory_signal:
            heuristic_override = "regulatory"
            heuristic_rationale = "Strong regulatory citation signal (CFR/FR reference) overrode planner routing."

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
        elif (
            planner.intent == "known_item"
            and planner.intent_source == "planner"
            and not strong_known_item_signal
            and (
                # Either the deterministic text heuristic says broad-concept, or
                # the LLM itself labelled the query as broad / low-specificity /
                # highly ambiguous. Trust the planner's own signals first and
                # only fall back to heuristics for missing/low-confidence cases.
                _looks_broad_concept_query(
                    normalized_query=normalized,
                    focus=focus,
                    year=year,
                    venue=venue,
                )
                or (
                    planner.query_type == "broad_concept"
                    and (planner.query_specificity == "low" or planner.ambiguity_level == "high")
                )
                # Exact-title-looking queries with no identifier and high
                # ambiguity are common cultural-resource / regulatory traps
                # (e.g. "Section 106 consultation for offshore wind") — keep
                # known-item reasoning active as a secondary pass but stop
                # force-routing them into pure title-match recovery.
                or (looks_like_exact_title(normalized) and planner.ambiguity_level == "high")
            )
        ):
            previous_intent = planner.intent
            fallback_intent: IntentLabel = "discovery"
            for candidate in planner.intent_candidates:
                if candidate.intent != "known_item" and candidate.confidence in {"high", "medium"}:
                    fallback_intent = candidate.intent
                    break
            planner.intent = fallback_intent
            planner.intent_source = "heuristic_override"
            planner.intent_confidence = "medium"
            planner.intent_rationale = (
                f"Planner selected '{previous_intent}' but the query looks like a broad conceptual "
                "question (no DOI/arXiv/URL, planner specificity/ambiguity signals disagree); "
                f"demoted to '{fallback_intent}' to avoid force-routing discovery work into "
                "known-item recovery."
            )
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
    merged_secondary_intents = [
        candidate.intent
        for candidate in planner.intent_candidates
        if candidate.intent != planner.intent and candidate.confidence in {"high", "medium"}
    ]
    for intent_label in planner.secondary_intents:
        if intent_label != planner.intent and intent_label not in merged_secondary_intents:
            merged_secondary_intents.append(intent_label)
    planner.secondary_intents = cast(list[IntentLabel], merged_secondary_intents[:3])
    primary_candidate = next(
        (candidate for candidate in planner.intent_candidates if candidate.intent == planner.intent),
        None,
    )
    planner.routing_confidence = (
        primary_candidate.confidence if primary_candidate is not None else planner.intent_confidence
    )
    if not planner.intent_rationale:
        planner.intent_rationale = "Intent routed from planner defaults."
    if planner.intent == "regulatory":
        if not planner.regulatory_subintent:
            planner.regulatory_subintent = _infer_regulatory_subintent(query, focus)
        if planner.entity_card is None:
            planner.entity_card = _infer_entity_card(query, focus)
    if planner.regulatory_intent is None:
        planner.regulatory_intent = _derive_regulatory_intent(planner=planner, query=query, focus=focus)
        if planner.regulatory_intent is not None and planner.regulatory_intent_source != "llm":
            planner.regulatory_intent_source = "deterministic_fallback"
    elif planner.regulatory_intent == "hybrid_regulatory_plus_literature" and not _has_literature_corroboration(
        planner=planner, query=query, focus=focus
    ):
        # LLM emitted the hybrid label directly. Require the same query-side
        # corroboration as the deterministic derivation path to avoid
        # forcing a literature review pass onto regulation-only asks.
        planner.regulatory_intent = _derive_regulatory_intent(planner=planner, query=query, focus=focus)
        # The LLM's original hybrid emission failed corroboration, so the
        # final label is deterministically derived -- record the downgrade so
        # downstream gates don't treat it as LLM-authoritative.
        if planner.regulatory_intent is not None:
            planner.regulatory_intent_source = "deterministic_fallback"
        else:
            planner.regulatory_intent_source = "unspecified"
    if planner.subject_card is None and planner.regulatory_intent is not None:
        # LLM-first subject card; uses planner.entity_card / candidate_concepts
        # and falls back to deterministic extraction when needed. The
        # ``llm_bundle_available`` flag reflects the *actual* planner execution:
        # it is True only when the provider bundle's ``aplan_search`` successfully
        # ran an LLM call. LLM bundles that silently fall back to
        # ``super().plan_search()`` (see provider_openai/provider_langchain) stamp
        # ``planner.planner_source="deterministic_fallback"``, which must not be
        # mistaken for a genuine LLM emission here even when the deterministic
        # shim happens to populate ``candidateConcepts`` / ``entityCard``.
        # ``intent_source`` is unreliable for this purpose -- explicit mode and
        # heuristic overrides rewrite it independently of planner provenance.
        from .subject_grounding import resolve_subject_card

        planner.subject_card = resolve_subject_card(
            query=query,
            focus=focus,
            planner=planner,
            llm_bundle_available=(planner.planner_source == "llm"),
            llm_emitted_grounding_signals=llm_emitted_grounding_signals,
        )
    if planner.regulatory_intent == "hybrid_regulatory_plus_literature":
        hybrid_marker = (
            "hybrid_policy_science: fuse regulatory primary sources (Federal Register, CFR, agency guidance) "
            "with peer-reviewed literature to answer questions that mix policy and scientific evidence."
        )
        if not any("hybrid_policy_science" in str(entry) for entry in planner.retrieval_hypotheses):
            planner.retrieval_hypotheses.append(hybrid_marker)
    if not planner.intent_family and _detect_cultural_resource_intent(query, focus):
        planner.intent_family = "heritage_cultural_resources"
    if not planner.search_angles and planner.retrieval_hypotheses:
        planner.search_angles = list(planner.retrieval_hypotheses)
    if not planner.retrieval_hypotheses and planner.search_angles:
        planner.retrieval_hypotheses = list(planner.search_angles)
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


async def grounded_expansion_candidates(
    *,
    original_query: str,
    papers: list[dict[str, Any]],
    config: AgenticConfig,
    provider_bundle: ModelProviderBundle,
    focus: str | None = None,
    venue: str | None = None,
    year: str | None = None,
) -> list[ExpansionCandidate]:
    """Create grounded query variants from retrieved evidence via a second provider pass."""
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

    provider_variants = await provider_bundle.asuggest_grounded_expansions(
        query=base_query,
        papers=papers,
        max_variants=config.max_grounded_variants,
    )
    variants.extend(provider_variants)

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
