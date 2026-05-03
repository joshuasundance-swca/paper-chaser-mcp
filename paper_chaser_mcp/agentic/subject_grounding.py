"""Subject-card grounding and document-family ranking helpers.

This module is the LLM-first subject resolver called from the regulatory /
known-item retrieval paths in ``graphs.py``. It reshapes the planner's
LLM-derived outputs (``entity_card``, ``candidate_concepts``,
``regulatory_intent``) into a canonical :class:`SubjectCard` that downstream
ranking, on-topic classification, and subject-chain gap detection can rely on.

When the planner ran in deterministic-fallback mode (no LLM bundle available),
the card is still emitted but marked ``confidence="deterministic_fallback"``
and ``source="deterministic_fallback"`` so that downstream code can disclose
the grounding limit to the caller.
"""

from __future__ import annotations

import re
from typing import Any, cast

from .models import (
    DocumentFamilyLabel,
    PlannerDecision,
    RegulatoryIntentLabel,
    SubjectCard,
    SubjectCardConfidence,
)
from .planner import (
    _detect_cultural_resource_intent,
    _infer_entity_card,
    normalize_query,
)

_ENTITY_FAMILY_MAP: dict[str, DocumentFamilyLabel] = {
    "recovery_plan": "recovery_plan",
    "critical_habitat": "critical_habitat",
    "listing_rule": "listing_rule",
    "guidance": "agency_guidance",
    "consultation_or_preservation": "consultation_guidance",
    "cfr_current_text": "cfr_current_text",
}


_DOC_FAMILY_PHRASES: tuple[tuple[DocumentFamilyLabel, tuple[str, ...]], ...] = (
    ("recovery_plan", ("recovery plan",)),
    ("critical_habitat", ("critical habitat",)),
    ("listing_rule", ("listing rule", "final rule", "proposed rule", "listing status")),
    ("programmatic_agreement", ("programmatic agreement",)),
    ("tribal_policy", ("tribal policy", "tribal consultation", "nagpra", "thpo")),
    ("consultation_guidance", ("section 106", "nhpa consultation", "section 7 consultation")),
    ("cfr_current_text", ("current cfr", "cfr current text", "codified text")),
    ("rulemaking_notice", ("federal register notice", "frn ", "rulemaking notice")),
    ("agency_guidance", ("guidance document", "handbook", "manual")),
)


def _derive_document_family(
    *,
    entity_family: str | None,
    normalized_lower: str,
    regulatory_intent: RegulatoryIntentLabel | None,
) -> DocumentFamilyLabel | None:
    if entity_family and entity_family in _ENTITY_FAMILY_MAP:
        return _ENTITY_FAMILY_MAP[entity_family]
    for family, phrases in _DOC_FAMILY_PHRASES:
        if any(phrase in normalized_lower for phrase in phrases):
            return family
    if regulatory_intent == "species_dossier":
        return "recovery_plan"
    if regulatory_intent == "current_cfr_text":
        return "cfr_current_text"
    if regulatory_intent == "guidance_lookup":
        return "agency_guidance"
    if regulatory_intent == "rulemaking_history":
        return "rulemaking_notice"
    return None


def resolve_subject_card(
    *,
    query: str,
    focus: str | None = None,
    planner: PlannerDecision | None = None,
    llm_bundle_available: bool = False,
    llm_emitted_grounding_signals: bool | None = None,
) -> SubjectCard:
    """Resolve a query into a :class:`SubjectCard`.

    Prefers LLM-derived planner signals. Falls back to the deterministic
    ``_infer_entity_card`` extractor. ``llm_bundle_available`` indicates that
    an LLM-capable bundle was used. ``llm_emitted_grounding_signals`` is a
    snapshot captured by the caller **before** deterministic fallback mutated
    ``planner.entity_card`` / ``planner.candidate_concepts``; it tells this
    resolver whether the LLM itself actually produced any phase-4 grounding
    signal. Only when both flags are true is the card stamped
    ``source="planner_llm"``. When ``llm_emitted_grounding_signals`` is left
    at ``None`` we fall back to the legacy derived check for backwards
    compatibility.
    """

    entity_card = (planner.entity_card if planner else None) or {}
    if not entity_card:
        entity_card = _infer_entity_card(query, focus) or {}

    common_name = entity_card.get("commonName")
    scientific_name = entity_card.get("scientificName")
    authority = entity_card.get("authorityContext")
    entity_family = entity_card.get("requestedDocumentFamily") or entity_card.get("documentFamily")

    aliases: list[str] = []
    seen_aliases: set[str] = set()
    for raw_alias in entity_card.get("aliases") or entity_card.get("alias") or []:
        cleaned = str(raw_alias).strip()
        lowered = cleaned.lower()
        if cleaned and lowered not in seen_aliases:
            aliases.append(cleaned)
            seen_aliases.add(lowered)

    taxonomy_hints: dict[str, str] = {}
    raw_taxonomy_hints = entity_card.get("taxonomyHints") or entity_card.get("taxonomy")
    if isinstance(raw_taxonomy_hints, dict):
        for raw_key, raw_value in raw_taxonomy_hints.items():
            key = str(raw_key).strip()
            value = str(raw_value).strip()
            if key and value:
                taxonomy_hints[key] = value
    for taxonomy_key in ("kingdom", "phylum", "class", "order", "family", "genus"):
        value = str(entity_card.get(taxonomy_key) or "").strip()
        if value and taxonomy_key not in taxonomy_hints:
            taxonomy_hints[taxonomy_key] = value

    primary_source_anchors: list[str] = []
    seen_anchors: set[str] = set()
    for raw_anchor in (
        entity_card.get("primarySourceAnchors")
        or entity_card.get("primary_source_anchors")
        or entity_card.get("anchorUrls")
        or entity_card.get("primarySourceUrls")
        or []
    ):
        cleaned = str(raw_anchor).strip()
        if cleaned and cleaned not in seen_anchors:
            primary_source_anchors.append(cleaned)
            seen_anchors.add(cleaned)

    normalized_lower = normalize_query(f"{query} {focus or ''}").lower()
    regulatory_intent = planner.regulatory_intent if planner else None
    requested_family = _derive_document_family(
        entity_family=entity_family,
        normalized_lower=normalized_lower,
        regulatory_intent=regulatory_intent,
    )

    subject_terms: list[str] = []
    if planner and planner.candidate_concepts:
        subject_terms = [concept for concept in planner.candidate_concepts if concept][:6]
    if not subject_terms:
        seen: set[str] = set()
        for candidate in (common_name, scientific_name, entity_family):
            if not candidate:
                continue
            token = str(candidate).strip()
            if token and token.lower() not in seen:
                subject_terms.append(token)
                seen.add(token.lower())

    # Provenance rule: ``planner_llm`` may only be stamped when the caller
    # confirms an LLM-backed bundle produced the underlying signals.
    # ``entity_card`` / ``candidate_concepts`` populated at resolve-time alone
    # do not imply LLM origin — the deterministic shim (and deterministic
    # post-processing in ``classify_query``) populates them too. Callers pass
    # ``llm_emitted_grounding_signals`` as a snapshot taken before any
    # deterministic mutation so we can tell them apart.
    if llm_emitted_grounding_signals is None:
        # Legacy behaviour for callers that have not migrated yet. This is
        # deliberately conservative: it will stamp ``planner_llm`` based on
        # any post-fallback signals, which may over-attribute. New callers
        # should always pass the snapshot.
        has_signals = bool(planner and (planner.entity_card or planner.candidate_concepts))
    else:
        has_signals = bool(llm_emitted_grounding_signals)
    confidence: SubjectCardConfidence
    source: str
    if llm_bundle_available and has_signals:
        confidence = "medium"
        source = "planner_llm"
    else:
        confidence = "deterministic_fallback"
        source = "deterministic_fallback"
    if common_name or scientific_name:
        if source == "planner_llm":
            confidence = "high"

    # Heritage / cultural-resource queries without a species or rule often
    # only have a cultural subject — keep the card shape consistent.
    if not (common_name or scientific_name) and _detect_cultural_resource_intent(query, focus):
        if not subject_terms:
            cultural_raw: Any = entity_card.get("cultural_subjects", [])
            if isinstance(cultural_raw, list):
                subject_terms = [str(item) for item in cultural_raw if item]

    return SubjectCard(
        commonName=common_name,
        scientificName=scientific_name,
        agency=authority,
        requestedDocumentFamily=requested_family,
        subjectTerms=subject_terms,
        aliases=aliases,
        taxonomyHints=taxonomy_hints,
        primarySourceAnchors=primary_source_anchors,
        confidence=confidence,
        source=cast(Any, source),
    )


# ---------------------------------------------------------------------------
# Document-family ranking boost
# ---------------------------------------------------------------------------

_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "recovery_plan": ("recovery plan",),
    "critical_habitat": ("critical habitat",),
    "listing_rule": ("listing rule", "final rule", "proposed rule", "listing determination"),
    "consultation_guidance": (
        "section 106",
        "section 7 consultation",
        "nhpa consultation",
        "consultation guidance",
        "consultation handbook",
    ),
    "cfr_current_text": ("code of federal regulations", "cfr "),
    "rulemaking_notice": ("federal register", "frn", "rule notice", "proposed rule", "final rule"),
    "programmatic_agreement": ("programmatic agreement",),
    "tribal_policy": ("tribal policy", "tribal consultation", "nagpra", "thpo", "tribal historic preservation"),
    "agency_guidance": ("guidance", "handbook", "manual"),
}


def detect_document_family_match(
    document: dict[str, Any],
    requested_family: str | None,
) -> tuple[str | None, float]:
    """Return (matched_family, boost) for a regulatory document.

    ``boost`` is a small additive score that callers can layer on top of the
    base regulatory ranking. Returns ``(None, 0.0)`` when no family match is
    detected or no requested family was specified.
    """

    if not requested_family:
        return None, 0.0
    keywords = _FAMILY_KEYWORDS.get(requested_family)
    if not keywords:
        return None, 0.0

    haystack_parts: list[str] = []
    for key in ("title", "summary", "citation", "documentFamily", "documentType", "docType"):
        value = document.get(key)
        if value:
            haystack_parts.append(str(value))
    tags = document.get("tags") or document.get("categories") or []
    if isinstance(tags, list):
        haystack_parts.extend(str(tag) for tag in tags)
    haystack = " ".join(haystack_parts).lower()
    if not haystack:
        return None, 0.0

    for keyword in keywords:
        if keyword in haystack:
            return requested_family, 0.25
    return None, 0.0


# ---------------------------------------------------------------------------
# Species-dossier gating
# ---------------------------------------------------------------------------

_SCIENTIFIC_NAME_RE = re.compile(r"\b([A-Z][a-z]+)\s+([a-z]{3,})\b")


def species_mentioned(document: dict[str, Any], card: SubjectCard) -> bool:
    """Return True when the document text contains the card's species signal."""

    haystack_parts: list[str] = []
    for key in ("title", "summary", "citation", "abstract"):
        value = document.get(key)
        if value:
            haystack_parts.append(str(value))
    haystack = " ".join(haystack_parts).lower()
    if not haystack:
        return False
    if card.scientific_name and card.scientific_name.lower() in haystack:
        return True
    if card.common_name and card.common_name.lower() in haystack:
        return True
    if card.scientific_name:
        match = _SCIENTIFIC_NAME_RE.search(card.scientific_name)
        if match:
            genus = match.group(1).lower()
            if genus in haystack:
                return True
    return False


def compute_subject_chain_gaps(
    *,
    card: SubjectCard,
    regulatory_intent: RegulatoryIntentLabel | None,
    documents: list[dict[str, Any]],
) -> list[str]:
    """Identify subject-chain gaps (e.g. ECOS profile but no recovery plan)."""

    gaps: list[str] = []
    if regulatory_intent != "species_dossier":
        return gaps
    if not (card.common_name or card.scientific_name):
        return gaps
    has_recovery_plan = False
    has_critical_habitat = False
    has_species_mention = False
    for document in documents:
        family_match, _ = detect_document_family_match(document, "recovery_plan")
        if family_match:
            has_recovery_plan = True
        ch_match, _ = detect_document_family_match(document, "critical_habitat")
        if ch_match:
            has_critical_habitat = True
        if species_mentioned(document, card):
            has_species_mention = True
    requested_family = str(card.requested_document_family or "").strip()
    require_recovery_plan = requested_family in {"", "recovery_plan"}
    require_critical_habitat = requested_family in {"", "critical_habitat"}
    if has_species_mention and require_recovery_plan and not has_recovery_plan:
        gaps.append("species_evidence_without_recovery_plan")
    if has_species_mention and require_critical_habitat and not has_critical_habitat:
        gaps.append("species_evidence_without_critical_habitat")
    if not has_species_mention and documents:
        gaps.append("regulatory_documents_without_species_specific_evidence")
    return gaps
