"""Regulatory routing and ranking helpers.

Phase 7a extraction: pure module-level helpers that drive the
regulatory branch of the smart-search graph. These helpers translate
planner output + raw queries into ECOS variant lists, CFR citations,
regulatory retrieval hypotheses, and document rankings. They contain no
``AgenticRuntime`` state, so they live outside the class to shrink
``_core.py`` and make the regulatory contract independently testable.

The heavier ``AgenticRuntime._search_regulatory`` method intentionally
stays on the class because it owns the provider wiring, enable flags,
context-notification scheduling, and background-task bookkeeping — all
of which are stateful concerns. See ``docs/seam-maps/graphs.md`` for
the full extraction plan and the rationale for the cut line.
"""

from __future__ import annotations

import re
from typing import Any

from ..models import PlannerDecision
from ..planner import query_facets
from .shared_state import (
    _AGENCY_AUTHORITY_TERMS,
    _AGENCY_GUIDANCE_DISCUSSION_TERMS,
    _AGENCY_GUIDANCE_DOCUMENT_TERMS,
    _AGENCY_GUIDANCE_QUERY_NOISE_TERMS,
    _AGENCY_GUIDANCE_TERMS,
    _CFR_DOC_TYPE_GENERIC,
    _CULTURAL_RESOURCE_DOCUMENT_TERMS,
    _GRAPH_GENERIC_TERMS,
    _REGULATORY_QUERY_NOISE_TERMS,
    _REGULATORY_SUBJECT_STOPWORDS,
    _SPECIES_QUERY_NOISE_TERMS,
)
from .source_records import _graph_topic_tokens


def _is_agency_guidance_query(query: str) -> bool:
    lowered = query.lower()
    if not any(term in lowered for term in _AGENCY_GUIDANCE_TERMS):
        return False
    return any(term in lowered for term in _AGENCY_AUTHORITY_TERMS)


def _extract_scientific_name_candidate(query: str) -> str | None:
    match = re.search(r"\b([A-Z][a-z]{2,})\s+([a-z][a-z-]{2,})\b", query)
    if not match:
        return None
    return f"{match.group(1)} {match.group(2)}"


def _extract_common_name_candidate(query: str) -> str | None:
    cleaned = re.sub(r"\b\d+\s*CFR\s*(?:Part\s*)?\d+(?:\.[\dA-Za-z-]+)?\b", " ", query, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d{4}-\d{4,6}\b", " ", cleaned)
    cleaned = re.sub(
        r"\b(?:"
        + "|".join(re.escape(term) for term in sorted(_SPECIES_QUERY_NOISE_TERMS, key=len, reverse=True))
        + r")\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(cleaned.split(" "))
    cleaned = " ".join(part for part in cleaned.split() if part)
    if not cleaned:
        return None
    token_count = len(re.findall(r"[A-Za-z][A-Za-z'-]{1,}", cleaned))
    if token_count < 2:
        return None
    return cleaned


def _ecos_query_variants(
    query: str,
    *,
    planner: PlannerDecision | None = None,
) -> list[tuple[str, str, str]]:
    """Return ECOS query candidates as ``(query, anchor_type, origin)`` tuples.

    ``origin`` is ``"raw"`` for regex/raw-query-derived candidates and
    ``"planner"`` for candidates supplied by the planner bundle's
    ``entityCard`` / ``subjectCard``. Raw candidates are emitted first so
    the ECOS loop tries query-grounded variants before falling back to
    planner-supplied names — this removes the hallucination-first risk
    where a plausible-but-wrong LLM species name returns real-but-wrong
    ECOS data and contaminates downstream ranking. The planner variants
    still run as a fallback so genuine LLM-emitted names (which can
    recover from genus-only / subspecies prose the regex misses) keep
    their recall.
    """

    variants: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def _add(candidate: str | None, anchor_type: str, origin: str) -> None:
        if not candidate:
            return
        normalized = " ".join(candidate.split())
        if not normalized:
            return
        marker = normalized.lower()
        if marker in seen:
            return
        seen.add(marker)
        variants.append((normalized, anchor_type, origin))

    opaque = _is_opaque_query(query)

    _add(_extract_scientific_name_candidate(query), "species_scientific_name", "raw")
    _add(_extract_common_name_candidate(query), "species_common_name", "raw")

    if not opaque:
        _add(query, "regulatory_subject_terms", "raw")

    if planner is not None:
        entity_card = planner.entity_card or {}
        if isinstance(entity_card, dict):
            _add(
                str(entity_card.get("scientificName") or "") or None,
                "species_scientific_name",
                "planner",
            )
            _add(
                str(entity_card.get("commonName") or "") or None,
                "species_common_name",
                "planner",
            )
        if planner.subject_card is not None:
            _add(planner.subject_card.scientific_name, "species_scientific_name", "planner")
            _add(planner.subject_card.common_name, "species_common_name", "planner")

    if opaque:
        _add(query, "regulatory_subject_terms", "raw")

    return variants


_ECOS_PROVENANCE_RANK: dict[str, int] = {"raw": 0, "planner": 1}


def _rank_ecos_variant_hits(
    variant_hits: list[tuple[int, str, str, dict[str, Any]]],
) -> tuple[int, str, str, dict[str, Any]] | None:
    """Finding 4 (5th rubber-duck pass): pick the best ECOS variant result.

    Each entry is ``(variant_idx, anchor_type, origin, search_payload)``. The
    earlier scoring (``hits * factor`` with a 0.9 planner factor) let a
    planner-only variant with two incidental hits outrank a corroborated raw
    variant with one solid hit, defeating the provenance-first intent.

    The ranking key is strictly tiered:

    1. Non-empty variants beat empty ones (a corroborated variant with zero
       hits cannot stand in for a variant with real hits).
    2. Provenance rank (raw/query-corroborated beats planner-supplied).
    3. Hit count (higher wins within the same tier).
    4. Original ``variant_idx`` to preserve the intentional ordering from
       ``_ecos_query_variants`` as the final tie-breaker.

    That guarantees any corroborated variant with ``>=1`` hit beats any
    planner-only variant regardless of hit count, while still using hit count
    to disambiguate variants sharing a provenance tier. Returns ``None`` when
    the input is empty.
    """

    if not variant_hits:
        return None
    scored: list[tuple[int, int, int, int, int, str, str, dict[str, Any]]] = []
    for variant_idx, anchor_type, origin, search_payload in variant_hits:
        hit_count = len(list(search_payload.get("data") or []))
        has_hits_rank = 0 if hit_count > 0 else 1
        provenance_rank = _ECOS_PROVENANCE_RANK.get(origin, 1)
        scored.append(
            (
                has_hits_rank,
                provenance_rank,
                -hit_count,
                variant_idx,
                hit_count,
                anchor_type,
                origin,
                search_payload,
            )
        )
    scored.sort()
    *_, variant_idx, _hit_count, anchor_type, origin, search_payload = scored[0]
    return (variant_idx, anchor_type, origin, search_payload)


_OPAQUE_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_OPAQUE_ARXIV_RE = re.compile(r"\barxiv[:\s]*\d{4}\.\d{4,5}\b", re.IGNORECASE)


def _is_opaque_query(query: str) -> bool:
    """Return True when ``query`` looks like an identifier rather than prose.

    Opaque queries (DOIs, arXiv ids, bare URLs, identifier-like tokens with
    almost no letters) are meaningless as ECOS ``regulatory_subject_terms``
    probes and tend to match ECOS entries incidentally. The caller uses this
    to defer the raw full-query variant until after planner-supplied names.
    """

    text = (query or "").strip()
    if not text:
        return False
    if _OPAQUE_DOI_RE.search(text) or _OPAQUE_ARXIV_RE.search(text):
        return True
    lowered = text.lower()
    if lowered.startswith(("http://", "https://", "www.")):
        return True
    alpha = sum(1 for ch in text if ch.isalpha())
    total = len(text)
    if total >= 8 and alpha / total < 0.4:
        return True
    if " " not in text and any(ch in text for ch in "/:._") and alpha / max(total, 1) < 0.6:
        return True
    return False


def _query_requests_regulatory_history(query: str) -> bool:
    lowered = query.lower()
    return any(
        marker in lowered
        for marker in (
            "federal register",
            "history",
            "listing history",
            "rulemaking",
            "timeline",
            "proposed rule",
            "final rule",
            "chronology",
        )
    )


def _parse_cfr_request(query: str) -> dict[str, Any] | None:
    match = re.search(
        r"\b(?P<title>\d+)\s*CFR\s*(?:Part\s*)?(?P<part>\d+)(?:\.(?P<section>[\dA-Za-z-]+))?",
        query,
        re.IGNORECASE,
    )
    if not match:
        return None
    return {
        "title_number": int(match.group("title")),
        "part_number": int(match.group("part")),
        "section_number": match.group("section"),
    }


def _is_current_cfr_text_request(query: str) -> bool:
    lowered = query.lower()
    if _parse_cfr_request(query) is None:
        return False
    markers = (
        "current cfr",
        "current text",
        "codified text",
        "cfr section",
        "what does",
        "what does the",
        "text of",
        "say about",
        "under ",
    )
    return any(marker in lowered for marker in markers) or bool(re.search(r"\b\d+\s*cfr\s+\d+(?:\.\S+)?\b", lowered))


def _derive_regulatory_query_flags(
    *,
    query: str,
    planner: PlannerDecision | None,
) -> tuple[bool, bool, bool]:
    """Map ``planner.regulatory_intent`` into the three routing booleans.

    LLM-first: when the planner bundle actually ran a real LLM (signalled by
    ``planner.planner_source == "llm"``) and emitted a definitive
    ``regulatoryIntent`` label, trust it authoritatively so the LLM signal
    wins over query keywords (e.g. "listing history of the Pallid Sturgeon"
    tagged ``species_dossier`` must NOT also activate the rulemaking-history
    route). Provenance is keyed off ``planner_source`` rather than
    ``subject_card.source`` because an LLM planner can legitimately emit
    ``regulatoryIntent`` without also supplying subject-card grounding
    fields; in that case ``classify_query`` stamps the subject card as
    ``deterministic_fallback``, but the LLM's regulatory label is still
    authoritative.

    Falls back to the deterministic keyword/regex helpers when:

    * the bundle is deterministic (``planner_source`` is ``"deterministic"``
      or ``"deterministic_fallback"``) — in that case ``regulatoryIntent``
      itself came from deterministic heuristics and is no more reliable than
      the keyword helpers, so we prefer the keyword helpers to avoid losing
      secondary routes on queries like "Regulatory history ... under 50 CFR ...";
    * the LLM emitted ``unspecified`` / ``hybrid_regulatory_plus_literature``
      / ``None`` — mixed or uncommitted intent, so every keyword-matched
      sub-route may still be relevant.

    Returns ``(current_text_requested, history_requested, agency_guidance_mode)``.
    """

    intent = planner.regulatory_intent if planner is not None else None
    llm_authoritative = (
        planner is not None
        and planner.planner_source == "llm"
        and planner.regulatory_intent_source == "llm"
        and intent is not None
    )
    if llm_authoritative:
        if intent == "current_cfr_text":
            return (True, False, False)
        if intent == "rulemaking_history":
            return (False, True, False)
        if intent == "guidance_lookup":
            return (False, False, True)
        if intent == "species_dossier":
            return (False, False, False)
    return (
        _is_current_cfr_text_request(query),
        _query_requests_regulatory_history(query),
        _is_agency_guidance_query(query),
    )


def _extract_subject_terms(*names: str | None) -> set[str]:
    tokens: set[str] = set()
    for name in names:
        if not name:
            continue
        for token in re.findall(r"[a-z0-9]{3,}", name.lower()):
            if token in _REGULATORY_SUBJECT_STOPWORDS:
                continue
            if len(token) <= 3:
                continue
            tokens.add(token)
    return tokens


def _format_cfr_citation(cfr_request: dict[str, Any] | None) -> str | None:
    if not cfr_request:
        return None
    title = cfr_request.get("title_number")
    part = cfr_request.get("part_number")
    section = cfr_request.get("section_number")
    if title is None or part is None:
        return None
    if section:
        return f"{title} CFR {part}.{section}"
    return f"{title} CFR {part}"


def _regulatory_retrieval_hypotheses(
    *,
    query: str,
    planner: PlannerDecision,
    subject: str | None,
    anchor_type: str | None,
    cfr_citation: str | None,
    current_text_requested: bool,
    history_requested: bool,
    agency_guidance_mode: bool,
) -> list[str]:
    hypotheses: list[str] = [str(item).strip() for item in planner.retrieval_hypotheses if str(item).strip()]
    anchor_subject = str(subject or planner.anchor_value or query).strip()
    if current_text_requested and cfr_citation:
        hypotheses.append(f"Current codified CFR text for {cfr_citation}.")
    if cfr_citation and not current_text_requested:
        hypotheses.append(f"40 CFR incorporation and referenced regulatory text for {cfr_citation}.")
    if agency_guidance_mode:
        hypotheses.append(f"Agency guidance documents directly addressing {anchor_subject}.")
        hypotheses.append(f"Federal Register notices or policy actions relevant to {anchor_subject}.")
    elif anchor_type in {"species_common_name", "species_scientific_name"}:
        hypotheses.append(f"ECOS species dossier and supporting recovery-plan materials for {anchor_subject}.")
        hypotheses.append(f"Federal Register listing or habitat actions for {anchor_subject}.")
    else:
        hypotheses.append(f"Federal Register final rule or notice history for {anchor_subject}.")
        hypotheses.append(f"Current CFR incorporation or agency primary-source text for {anchor_subject}.")
    if history_requested and not agency_guidance_mode:
        hypotheses.append(f"Rulemaking timeline milestones for {anchor_subject}.")
    for facet in query_facets(query):
        facet_text = str(facet).strip()
        if facet_text:
            hypotheses.append(facet_text)

    deduped: list[str] = []
    seen: set[str] = set()
    for hypothesis in hypotheses:
        normalized = str(hypothesis).strip()
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped[:5]


def _cfr_tokens(citation: str | None) -> set[str]:
    if not citation:
        return set()
    return {token for token in re.findall(r"[a-z0-9]{2,}", citation.lower()) if token not in _CFR_DOC_TYPE_GENERIC}


def _regulatory_document_matches_subject(
    document: dict[str, Any],
    *,
    subject_terms: set[str],
    priority_terms: set[str] | None = None,
    cfr_citation: str | None,
) -> bool:
    title = str(document.get("title") or "")
    summary = str(document.get("abstract") or document.get("excerpt") or document.get("summary") or "")
    cfr_refs_raw = document.get("cfrReferences")
    cfr_refs = cfr_refs_raw if isinstance(cfr_refs_raw, list) else []
    cfr_ref_text = " ".join(str(ref) for ref in cfr_refs)
    document_text = " ".join(
        part for part in [title, summary, str(document.get("citation") or ""), cfr_ref_text] if part
    )
    document_tokens = _graph_topic_tokens(document_text)
    title_tokens = _graph_topic_tokens(title)
    priority_overlap = len(document_tokens & set(priority_terms or set()))

    subject_match_required = bool(subject_terms)
    subject_title_overlap = len(subject_terms & title_tokens)
    subject_body_overlap = len(subject_terms & document_tokens)
    subject_match = subject_title_overlap > 0 or subject_body_overlap >= 2 or priority_overlap >= 2
    if priority_terms:
        subject_match = subject_match and priority_overlap > 0

    cfr_match = False
    cfr_expected_tokens = _cfr_tokens(cfr_citation)
    if cfr_expected_tokens:
        for ref in cfr_refs:
            ref_tokens = _cfr_tokens(str(ref))
            if cfr_expected_tokens.issubset(ref_tokens):
                cfr_match = True
                break
        if not cfr_match:
            cfr_match = cfr_expected_tokens.issubset(_cfr_tokens(str(document.get("citation") or "")))

    if subject_match_required and cfr_expected_tokens:
        return subject_match and cfr_match
    if subject_match_required:
        return subject_match
    if cfr_expected_tokens:
        return cfr_match
    return True


def _agency_guidance_subject_terms(query: str) -> set[str]:
    normalized = re.sub(r"[-_/]+", " ", query.lower())
    return {
        term
        for term in re.findall(r"[a-z0-9]{4,}", normalized)
        if term not in _REGULATORY_SUBJECT_STOPWORDS
        and term not in _AGENCY_GUIDANCE_QUERY_NOISE_TERMS
        and term not in _GRAPH_GENERIC_TERMS
        and len(term) > 3
    }


def _regulatory_query_subject_terms(query: str) -> set[str]:
    normalized = re.sub(r"[-_/]+", " ", query.lower())
    return {
        term
        for term in re.findall(r"[a-z0-9]{4,}", normalized)
        if term not in _REGULATORY_SUBJECT_STOPWORDS
        and term not in _REGULATORY_QUERY_NOISE_TERMS
        and term not in _GRAPH_GENERIC_TERMS
    }


def _regulatory_query_priority_terms(query: str) -> set[str]:
    authority_terms = {term.lower() for term in _AGENCY_AUTHORITY_TERMS if " " not in term}
    generic_regulatory_acronyms = {
        "cfr",
        "esa",
        "fr",
        "u.s",
        "us",
    }
    return {
        token.lower()
        for token in re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", query)
        if token.lower() not in authority_terms and token.lower() not in generic_regulatory_acronyms
    }


def _agency_guidance_priority_terms(query: str) -> set[str]:
    terms = _agency_guidance_subject_terms(query)
    for facet in query_facets(query):
        for token in re.findall(r"[a-z0-9]{4,}", facet.lower()):
            if token in _GRAPH_GENERIC_TERMS or token in _AGENCY_GUIDANCE_QUERY_NOISE_TERMS:
                continue
            if token in _REGULATORY_SUBJECT_STOPWORDS:
                continue
            terms.add(token)
    return terms


def _agency_guidance_facet_terms(query: str) -> list[set[str]]:
    facet_terms: list[set[str]] = []
    for facet in query_facets(query):
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]{4,}", facet.lower())
            if token not in _GRAPH_GENERIC_TERMS
            and token not in _AGENCY_GUIDANCE_QUERY_NOISE_TERMS
            and token not in _REGULATORY_SUBJECT_STOPWORDS
        }
        if len(tokens) >= 2:
            facet_terms.append(tokens)
    return facet_terms


def _guidance_query_prefers_recency(query: str) -> bool:
    lowered = query.lower()
    return any(marker in lowered for marker in {"current", "latest", "new", "newest", "recent"})


def _is_species_regulatory_query(query: str) -> bool:
    lowered = query.lower()
    regulatory_markers = {"esa", "final rule", "listing history", "listing status", "regulatory history"}
    species_markers = {
        "bat",
        "bird",
        "condor",
        "critical habitat",
        "endangered",
        "habitat",
        "listing",
        "recovery",
        "species",
        "threatened",
        "wildlife",
    }
    return any(marker in lowered for marker in regulatory_markers) and any(
        marker in lowered for marker in species_markers
    )


def _rank_regulatory_documents(
    documents: list[dict[str, Any]],
    *,
    subject_terms: set[str],
    priority_terms: set[str],
    facet_terms: list[set[str]],
    prefer_guidance: bool,
    prefer_recent: bool,
    cultural_resource_boost: bool = False,
    requested_document_family: str | None = None,
) -> list[dict[str, Any]]:
    from ..subject_grounding import detect_document_family_match

    def _score(document: dict[str, Any]) -> tuple[int, str]:
        title = str(document.get("title") or "")
        summary = str(document.get("abstract") or document.get("excerpt") or document.get("summary") or "")
        tokens = _graph_topic_tokens(" ".join(part for part in [title, summary] if part))
        title_tokens = _graph_topic_tokens(title)
        overlap = len(subject_terms & tokens)
        title_overlap = len(subject_terms & title_tokens)
        priority_overlap = len(tokens & priority_terms)
        facet_overlap = 0
        for facet in facet_terms:
            required = len(facet) if len(facet) <= 2 else 2
            if sum(token in tokens for token in facet) >= required:
                facet_overlap += 1
        document_type = str(document.get("documentType") or "").lower()
        publication_date = str(document.get("publicationDate") or "")
        publication_year_match = re.search(r"\b(19|20)\d{2}\b", publication_date)
        publication_year = int(publication_year_match.group(0)) if publication_year_match else 0
        guidance_form_bonus = 8 if (tokens & _AGENCY_GUIDANCE_DOCUMENT_TERMS) else 0
        discussion_form_penalty = 4 if (tokens & _AGENCY_GUIDANCE_DISCUSSION_TERMS) else 0
        guidance_bonus = (
            2 if prefer_guidance and any(token in tokens for token in {"guidance", "framework", "discussion"}) else 0
        )
        notice_bonus = 1 if document_type in {"notice", "rule"} else 0
        recency_bonus = max((publication_year - 2010) * 2, 0) if prefer_recent else 0
        cultural_overlap = len(tokens & _CULTURAL_RESOURCE_DOCUMENT_TERMS) if cultural_resource_boost else 0
        cultural_title_overlap = len(title_tokens & _CULTURAL_RESOURCE_DOCUMENT_TERMS) if cultural_resource_boost else 0
        cultural_bonus = (cultural_title_overlap * 6) + (cultural_overlap * 3)
        family_match, family_boost_fraction = detect_document_family_match(document, requested_document_family)
        family_bonus = int(round(family_boost_fraction * 24)) if family_match else 0
        if family_match:
            document["_documentFamilyMatch"] = family_match
            document["_documentFamilyBoost"] = round(family_boost_fraction, 3)
        score = (
            (title_overlap * 5)
            + (overlap * 2)
            + (priority_overlap * 2)
            + (facet_overlap * 3)
            + guidance_form_bonus
            + guidance_bonus
            + notice_bonus
            + recency_bonus
            + cultural_bonus
            + family_bonus
            - discussion_form_penalty
        )
        return score, publication_date

    return sorted(documents, key=_score, reverse=True)
