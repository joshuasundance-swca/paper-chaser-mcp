"""Regulatory and literature intent detectors.

These helpers classify a query as regulatory/literature/cultural-resource and
infer regulatory sub-intents plus the compact entity card used downstream to
route primary-source workflows. They depend only on the constants module and
the low-level normalization helpers, so they never import the async planner
orchestrator.
"""

from __future__ import annotations

import re

from .constants import (
    _CULTURAL_RESOURCE_MARKERS,
    AGENCY_REGULATORY_MARKERS,
    ARXIV_RE,
    DOI_RE,
    LITERATURE_QUERY_TERMS,
    REGULATORY_QUERY_TERMS,
)
from .normalization import looks_like_url, normalize_query


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


__all__ = [
    "_detect_cultural_resource_intent",
    "_infer_entity_card",
    "_infer_regulatory_subintent",
    "_strong_known_item_signal",
    "_strong_regulatory_signal",
    "detect_literature_intent",
    "detect_regulatory_intent",
]
