"""Source-record and topical-relevance helpers for ``paper_chaser_mcp.agentic.graphs``.

This submodule owns pure helpers that translate upstream ``Paper`` / regulatory
document payloads into ``StructuredSourceRecord`` / ``CitationRecord`` rows,
split those rows into grounded evidence versus leads, derive the topical-
relevance classification, and produce the small routing/coverage/answerability
summaries that ride alongside them.

Extracted in Phase 7a commit 3 from the original flat ``graphs.py``. The
functions stay importable from ``paper_chaser_mcp.agentic.graphs`` via the
facade and from ``paper_chaser_mcp.agentic.graphs._core`` via a re-import, so
existing monkeypatch seams and call sites continue to work unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, cast

from ...identifiers import resolve_doi_from_paper_payload
from ...models import (
    CitationRecord,
    CoverageSummary,
    Paper,
)
from ..models import (
    ScoreBreakdown,
    SearchStrategyMetadata,
    StructuredSourceRecord,
)
from ..planner import (
    looks_like_exact_title,
    looks_like_near_known_item_query,
    query_facets,
    query_terms,
)
from .shared_state import _GRAPH_GENERIC_TERMS


def _why_matched(
    *,
    query: str,
    paper: dict[str, Any],
    matched_concepts: list[str],
) -> str:
    title = str(paper.get("title") or paper.get("paperId") or "paper")
    if matched_concepts:
        return f"{title} matched concepts {', '.join(matched_concepts[:3])}."
    if paper.get("venue"):
        return f"{title} matched the query and carries useful venue context from {paper['venue']}."
    return f"{title} was retained because it stayed close to the original query after fusion and deduplication."


def _year_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else text[:4]


def _paper_text(paper: dict[str, Any]) -> str:
    authors = ", ".join(author.get("name", "") for author in (paper.get("authors") or []) if isinstance(author, dict))
    return " ".join(
        part
        for part in [
            str(paper.get("title") or ""),
            str(paper.get("abstract") or ""),
            str(paper.get("venue") or ""),
            str(paper.get("year") or ""),
            authors,
        ]
        if part
    )


def _graph_topic_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", text.lower()) if token not in _GRAPH_GENERIC_TERMS}


def _citation_record_from_paper(paper: Paper) -> CitationRecord | None:
    doi, _ = resolve_doi_from_paper_payload(paper)
    authors = [str(author.name).strip() for author in paper.authors if getattr(author, "name", None)]
    journal_or_publisher = (
        paper.enrichments.crossref.publisher if paper.enrichments and paper.enrichments.crossref else None
    ) or paper.venue
    url = paper.canonical_url or paper.retrieved_url or paper.url or paper.pdf_url
    if not any([authors, paper.title, paper.year, journal_or_publisher, doi, url]):
        return None
    return CitationRecord(
        authors=authors,
        year=_year_text(paper.publication_date or paper.year),
        title=paper.title,
        journalOrPublisher=journal_or_publisher,
        doi=doi,
        url=url,
        sourceType=paper.source_type,
        confidence=cast(Any, paper.confidence),
    )


def _citation_record_from_regulatory_document(
    document: dict[str, Any],
    *,
    provider: str,
    citation_text: str | None,
    canonical_url: str | None,
) -> CitationRecord:
    return CitationRecord(
        authors=[],
        year=_year_text(
            document.get("documentDate") or document.get("publicationDate") or document.get("effectiveDate")
        ),
        title=str(document.get("title") or citation_text or "Regulatory source"),
        journalOrPublisher=(
            "GovInfo" if provider == "govinfo" else ("Federal Register" if provider == "federal_register" else "ECOS")
        ),
        doi=None,
        url=canonical_url,
        sourceType="primary_regulatory",
        confidence=("high" if provider == "govinfo" else "medium"),
    )


def _source_record_from_paper(
    paper: Paper,
    *,
    note: str | None = None,
    topical_relevance: Literal["on_topic", "weak_match", "off_topic"] | None = None,
    llm_classification: Literal["on_topic", "weak_match", "off_topic"] | None = None,
    classification_source: Literal["deterministic", "llm", "llm_tiebreaker"] | None = None,
    relevance_source: str | None = None,
    relevance_confidence: float | None = None,
    relevance_reason: str | None = None,
    classification_rationale: str | None = None,
) -> StructuredSourceRecord:
    source_id = str(paper.canonical_id or paper.paper_id or "") or None
    return StructuredSourceRecord(
        sourceId=source_id,
        title=paper.title,
        provider=paper.source,
        sourceType=paper.source_type,
        verificationStatus=paper.verification_status,
        accessStatus=paper.access_status,
        topicalRelevance=topical_relevance,
        llmClassification=llm_classification,
        classificationSource=classification_source,
        confidence=paper.confidence,
        isPrimarySource=paper.is_primary_source,
        canonicalUrl=paper.canonical_url,
        retrievedUrl=paper.retrieved_url,
        fullTextUrlFound=paper.full_text_url_found,
        bodyTextEmbedded=paper.body_text_embedded,
        qaReadableText=paper.qa_readable_text,
        abstractObserved=paper.abstract_observed,
        openAccessRoute=paper.open_access_route,
        citationText=str(paper.canonical_id or paper.paper_id or "") or None,
        citation=_citation_record_from_paper(paper),
        date=str(paper.publication_date or paper.year or "") or None,
        note=note,
        relevanceSource=cast(Any, relevance_source) if relevance_source else None,
        relevanceConfidence=relevance_confidence,
        relevanceReason=relevance_reason,
        classificationRationale=classification_rationale,
    )


def _source_record_from_regulatory_document(
    document: dict[str, Any],
    *,
    provider: str,
    topical_relevance: Literal["on_topic", "weak_match", "off_topic"] | None = "on_topic",
    why_classified_as_weak_match: str | None = None,
) -> StructuredSourceRecord:
    title = str(
        document.get("title") or document.get("citation") or document.get("documentNumber") or "Regulatory source"
    )
    canonical_url = (
        document.get("url")
        or document.get("htmlUrl")
        or document.get("sourceUrl")
        or document.get("govInfoLink")
        or document.get("pdfUrl")
    )
    citation = (
        str(document.get("citation") or document.get("frCitation") or document.get("documentNumber") or "") or None
    )
    source_id = citation or str(document.get("documentNumber") or document.get("speciesId") or "") or None
    date = document.get("documentDate") or document.get("publicationDate") or document.get("effectiveDate")
    note = None
    if provider == "ecos":
        note = str(document.get("documentType") or document.get("documentKind") or "ECOS dossier document")
    elif provider == "federal_register":
        cfr_refs = document.get("cfrReferences") or []
        note = (
            ", ".join(cfr_refs)
            if isinstance(cfr_refs, list) and cfr_refs
            else "Federal Register primary-source discovery hit"
        )
    elif provider == "govinfo":
        note = str(
            document.get("note")
            or ("Authoritative CFR text" if document.get("markdown") else "GovInfo primary-source discovery hit")
        )
    has_inline_body = bool(document.get("markdown"))
    has_url = bool(canonical_url)
    family_match = document.get("_documentFamilyMatch")
    family_boost = document.get("_documentFamilyBoost")
    if has_inline_body:
        access_status: str = "body_text_embedded"
    elif has_url:
        access_status = "url_verified"
    else:
        access_status = "access_unverified"
    if has_inline_body:
        verification_status_default = "verified_primary_source" if provider == "govinfo" else "verified_metadata"
    else:
        verification_status_default = "verified_metadata"
    return StructuredSourceRecord(
        sourceId=source_id,
        title=title,
        provider=provider,
        sourceType="primary_regulatory",
        verificationStatus=str(document.get("verificationStatus") or verification_status_default),
        accessStatus=access_status,
        confidence="high" if (provider == "govinfo" and has_inline_body) else "medium",
        topicalRelevance=topical_relevance,
        isPrimarySource=True,
        canonicalUrl=canonical_url,
        retrievedUrl=canonical_url,
        fullTextUrlFound=has_url,
        fullTextRetrieved=has_inline_body,
        bodyTextEmbedded=has_inline_body,
        qaReadableText=has_inline_body,
        abstractObserved=False,
        openAccessRoute=("non_oa_or_unconfirmed" if (has_inline_body or has_url) else "unknown"),
        citationText=citation,
        citation=_citation_record_from_regulatory_document(
            document,
            provider=provider,
            citation_text=citation,
            canonical_url=str(canonical_url or "") or None,
        ),
        date=str(date or "") or None,
        note=note,
        whyClassifiedAsWeakMatch=why_classified_as_weak_match,
        documentFamilyMatch=str(family_match) if family_match else None,
        documentFamilyBoost=float(family_boost) if isinstance(family_boost, (int, float)) else None,
    )


def _coverage_summary_line(
    *,
    attempted: list[str],
    failed: list[str],
    zero_results: list[str],
    likely_completeness: str,
) -> str:
    return (
        f"{len(attempted)} provider(s) searched, {len(failed)} failed, "
        f"{len(zero_results)} returned zero results, likely completeness: {likely_completeness}."
    )


def _classify_topical_relevance(
    *,
    query_similarity: float,
    title_facet_coverage: float,
    title_anchor_coverage: float,
    query_facet_coverage: float,
    query_anchor_coverage: float,
) -> Literal["on_topic", "weak_match", "off_topic"]:
    has_title_signal = (title_facet_coverage > 0.0) or (title_anchor_coverage > 0.0)
    has_title_or_body_signal = has_title_signal or (query_facet_coverage > 0.0) or (query_anchor_coverage > 0.0)
    has_facet_signal = (title_facet_coverage > 0.0) or (query_facet_coverage > 0.0)
    # Require a multi-token phrase match (facet) for the standard threshold, or a
    # strict majority of query terms when no phrase match exists.  A single-token
    # title hit with low similarity is a weak signal, not grounded evidence.
    if has_title_signal and ((has_facet_signal and query_similarity >= 0.25) or query_similarity > 0.5):
        return "on_topic"
    if query_similarity < 0.12 or not has_title_or_body_signal:
        return "off_topic"
    return "weak_match"


def _classify_topical_relevance_for_paper(
    *,
    query: str,
    paper: dict[str, Any] | Paper,
    query_similarity: float,
    score_breakdown: ScoreBreakdown | None = None,
    llm_classification: Literal["on_topic", "weak_match", "off_topic"] | None = None,
) -> Literal["on_topic", "weak_match", "off_topic"]:
    return _classify_topical_relevance_with_provenance(
        query=query,
        paper=paper,
        query_similarity=query_similarity,
        score_breakdown=score_breakdown,
        llm_classification=llm_classification,
    ).effective


@dataclass(frozen=True)
class TopicalRelevanceClassification:
    """Provenance-aware result of the topical-relevance gate.

    ``effective`` is the verdict callers should act on; it matches what the
    legacy ``_classify_topical_relevance_for_paper`` returned. ``deterministic``
    is the raw heuristic verdict, ``llm`` is the classifier verdict when
    available, and ``source`` records which signal produced ``effective``.

    ``llm_override_ignored`` is True exactly when the deterministic fast-path
    produced a clear on_topic/off_topic verdict and an LLM classification was
    available but disagreed. The effective verdict is still the deterministic
    one in that case — the flag is purely for observability so callers can log
    the override or surface a counter.
    """

    effective: Literal["on_topic", "weak_match", "off_topic"]
    deterministic: Literal["on_topic", "weak_match", "off_topic"]
    llm: Literal["on_topic", "weak_match", "off_topic"] | None
    source: Literal["deterministic", "llm", "llm_tiebreaker"]
    llm_override_ignored: bool


def _classify_topical_relevance_with_provenance(
    *,
    query: str,
    paper: dict[str, Any] | Paper,
    query_similarity: float,
    score_breakdown: ScoreBreakdown | None = None,
    llm_classification: Literal["on_topic", "weak_match", "off_topic"] | None = None,
) -> TopicalRelevanceClassification:
    title = str((paper.title if isinstance(paper, Paper) else paper.get("title")) or "")
    body_text = _paper_text(paper.model_dump(by_alias=True) if isinstance(paper, Paper) else paper)
    title_tokens = _graph_topic_tokens(title)
    body_tokens = _graph_topic_tokens(body_text)
    anchors = [term for term in query_terms(query) if term not in _GRAPH_GENERIC_TERMS]
    facets = query_facets(query)

    title_anchor_hits = sum(term in title_tokens for term in anchors)
    body_anchor_hits = sum(term in body_tokens for term in anchors)
    title_anchor_coverage = (title_anchor_hits / len(anchors)) if anchors else 0.0
    query_anchor_coverage = (body_anchor_hits / len(anchors)) if anchors else 0.0

    def _facet_coverage(tokens: set[str]) -> float:
        if not facets:
            return 0.0
        matched = 0
        for facet in facets:
            facet_tokens = [token for token in re.findall(r"[a-z0-9]{3,}", facet.lower()) if token]
            if not facet_tokens:
                continue
            required = len(facet_tokens) if len(facet_tokens) <= 2 else 2
            if sum(token in tokens for token in facet_tokens) >= required:
                matched += 1
        return matched / len(facets)

    title_facet_coverage = _facet_coverage(title_tokens)
    query_facet_coverage = _facet_coverage(body_tokens)
    if score_breakdown is not None:
        title_facet_coverage = max(title_facet_coverage, score_breakdown.title_facet_coverage)
        title_anchor_coverage = max(title_anchor_coverage, score_breakdown.title_anchor_coverage)
        query_facet_coverage = max(query_facet_coverage, score_breakdown.query_facet_coverage)
        query_anchor_coverage = max(query_anchor_coverage, score_breakdown.query_anchor_coverage)

    deterministic = _classify_topical_relevance(
        query_similarity=query_similarity,
        title_facet_coverage=title_facet_coverage,
        title_anchor_coverage=title_anchor_coverage,
        query_facet_coverage=query_facet_coverage,
        query_anchor_coverage=query_anchor_coverage,
    )
    has_title_signal = (title_facet_coverage > 0.0) or (title_anchor_coverage > 0.0)
    has_title_or_body_signal = has_title_signal or (query_facet_coverage > 0.0) or (query_anchor_coverage > 0.0)
    fast_path_on_topic = deterministic == "on_topic" and query_similarity > 0.5
    fast_path_off_topic = deterministic == "off_topic" and query_similarity < 0.12 and not has_title_or_body_signal
    strict_title_alignment_query = looks_like_exact_title(query) or looks_like_near_known_item_query(query)
    guard_llm_on_topic_promotion = (
        strict_title_alignment_query
        and deterministic == "weak_match"
        and llm_classification == "on_topic"
        and not has_title_signal
    )

    effective: Literal["on_topic", "weak_match", "off_topic"]
    source: Literal["deterministic", "llm", "llm_tiebreaker"]
    llm_override_ignored = False
    if fast_path_on_topic:
        effective = "on_topic"
        source = "deterministic"
        if llm_classification is not None and llm_classification != "on_topic":
            llm_override_ignored = True
    elif fast_path_off_topic:
        effective = "off_topic"
        source = "deterministic"
        if llm_classification is not None and llm_classification != "off_topic":
            llm_override_ignored = True
    elif guard_llm_on_topic_promotion:
        effective = deterministic
        source = "deterministic"
    elif llm_classification is not None:
        effective = llm_classification
        source = "llm_tiebreaker" if deterministic == "weak_match" else "llm"
    else:
        effective = deterministic
        source = "deterministic"

    return TopicalRelevanceClassification(
        effective=effective,
        deterministic=deterministic,
        llm=llm_classification,
        source=source,
        llm_override_ignored=llm_override_ignored,
    )


def _verified_findings_from_source_records(records: list[StructuredSourceRecord]) -> list[str]:
    findings: list[str] = []
    for record in records:
        if record.verification_status not in {"verified_primary_source", "verified_metadata"}:
            continue
        if record.topical_relevance != "on_topic":
            continue
        title = record.title or record.citation_text or "Verified source"
        suffix = f" ({record.citation_text})" if record.citation_text and record.citation_text not in title else ""
        note = f": {record.note}" if record.note else ""
        findings.append(f"{title}{suffix}{note}")
    return findings[:6]


def _likely_unverified_from_source_records(records: list[StructuredSourceRecord]) -> list[str]:
    leads: list[str] = []
    for record in records:
        if (
            record.verification_status in {"verified_primary_source", "verified_metadata"}
            and record.topical_relevance == "on_topic"
        ):
            continue
        title = record.title or record.citation_text or "Unverified source"
        note = f": {record.note}" if record.note else ""
        leads.append(f"{title}{note}")
    return leads[:6]


def _lead_reason_for_source_record(record: StructuredSourceRecord) -> str:
    if record.topical_relevance == "off_topic":
        return "Excluded from grounded evidence because it drifted off the anchored subject."
    if record.verification_status not in {"verified_primary_source", "verified_metadata"}:
        return "Retained as a lead because the source is not yet verified strongly enough to support a grounded answer."
    return "Retained as a lead because it is related context, but not strong enough to ground the answer on its own."


def _records_with_lead_reasons(records: list[StructuredSourceRecord]) -> list[StructuredSourceRecord]:
    enriched: list[StructuredSourceRecord] = []
    for record in records:
        reason = _lead_reason_for_source_record(record)
        why_not_verified = record.why_not_verified
        if why_not_verified is None:
            if record.verification_status not in {"verified_primary_source", "verified_metadata"}:
                why_not_verified = f"Verification status was {record.verification_status or 'unverified'}."
            elif record.topical_relevance and record.topical_relevance != "on_topic":
                why_not_verified = f"Topical relevance was {record.topical_relevance}."
        enriched.append(
            record.model_copy(
                update={
                    "lead_reason": record.lead_reason or reason,
                    "why_not_verified": why_not_verified,
                }
            )
        )
    return enriched


def _dedupe_structured_sources(records: list[StructuredSourceRecord]) -> list[StructuredSourceRecord]:
    deduped: list[StructuredSourceRecord] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for record in records:
        key = (record.title, record.canonical_url, record.citation_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _candidate_leads_from_source_records(records: list[StructuredSourceRecord]) -> list[StructuredSourceRecord]:
    leads: list[StructuredSourceRecord] = []
    for record in records:
        if (
            record.verification_status in {"verified_primary_source", "verified_metadata"}
            and record.topical_relevance == "on_topic"
        ):
            continue
        leads.append(record)
    return _records_with_lead_reasons(_dedupe_structured_sources(leads)[:6])


def _evidence_from_source_records(records: list[StructuredSourceRecord]) -> list[StructuredSourceRecord]:
    evidence: list[StructuredSourceRecord] = []
    for record in records:
        if record.verification_status not in {"verified_primary_source", "verified_metadata"}:
            continue
        if record.topical_relevance != "on_topic":
            continue
        evidence.append(record)
    return _dedupe_structured_sources(evidence)[:6]


def _answerability_from_source_records(
    *,
    result_status: str,
    evidence: list[StructuredSourceRecord],
    leads: list[StructuredSourceRecord],
    evidence_gaps: list[str],
) -> Literal["grounded", "limited", "insufficient"]:
    if result_status == "succeeded" and evidence:
        return "grounded"
    if evidence or leads or evidence_gaps:
        return "limited"
    return "insufficient"


def _routing_summary_from_strategy(
    *,
    strategy_metadata: SearchStrategyMetadata,
    coverage_summary: CoverageSummary | None,
    result_status: str,
    evidence_gaps: list[str],
) -> dict[str, Any]:
    providers_attempted = list((coverage_summary.providers_attempted if coverage_summary else []) or [])
    providers_succeeded = list((coverage_summary.providers_succeeded if coverage_summary else []) or [])
    providers_failed = list((coverage_summary.providers_failed if coverage_summary else []) or [])
    provider_plan = list(
        cast(list[str] | None, getattr(strategy_metadata, "provider_plan", None))
        or strategy_metadata.providers_used
        or []
    )
    return {
        "intent": strategy_metadata.intent,
        "decisionConfidence": strategy_metadata.routing_confidence or strategy_metadata.intent_confidence,
        "anchorType": strategy_metadata.anchor_type,
        "anchorValue": strategy_metadata.anchored_subject,
        "providerPlan": provider_plan,
        "providersAttempted": providers_attempted,
        "providersMatched": providers_succeeded,
        "providersFailed": providers_failed,
        "providersNotAttempted": [provider for provider in provider_plan if provider not in providers_attempted],
        "whyPartial": (evidence_gaps[0] if (result_status != "succeeded" and evidence_gaps) else None),
    }
