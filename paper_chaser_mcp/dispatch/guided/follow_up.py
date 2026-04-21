"""Guided follow-up research helpers (Phase 3 extraction)."""

from __future__ import annotations

import re
from typing import Any, Literal, cast

from ...guided_semantic import build_follow_up_decision, explicit_source_reference
from ..normalization import _guided_normalize_whitespace
from .inspect_source import (
    _guided_extract_source_reference_from_question,
    _guided_select_follow_up_source,
)
from .response import _guided_contract_fields
from .sources import _guided_dedupe_source_records, _guided_source_coverage_summary
from .strategy_metadata import _guided_execution_provenance_payload
from .trust import (
    _guided_confidence_signals,
    _guided_result_state,
    _guided_trust_summary,
)


def _guided_follow_up_introspection_facets(question: str) -> set[str]:
    text = question.lower()
    facets: set[str] = set()
    if any(
        marker in text
        for marker in (
            "provider",
            "providers",
            "coverage",
            "searched",
            "search mode",
            "completeness",
            "attempted",
            "succeeded",
            "failed",
            "zero results",
            "zero-result",
            "zero result",
        )
    ):
        facets.add("coverage")
    if any(
        marker in text
        for marker in (
            "evidence gap",
            "evidence gaps",
            "what prevented",
            "blocking gap",
            "missing evidence",
            "what was missing",
            "why abstained",
            "why partial",
            "why incomplete",
            "prevented a grounded",
            "prevented grounded",
        )
    ):
        facets.add("evidence_gaps")
    if any(
        marker in text
        for marker in (
            "fallback",
            "what failed",
            "failure",
            "still worked",
            "degraded",
        )
    ):
        facets.add("failure_summary")
    if any(
        marker in text
        for marker in (
            "verified finding",
            "verified findings",
            "strongest verified",
            "strongest finding",
            "main finding",
            "best finding",
            "top finding",
            "trusted finding",
        )
    ):
        facets.add("verified_findings")
    if any(
        marker in text
        for marker in (
            "trust summary",
            "on-topic",
            "off-topic",
            "weak match",
            "how many verified",
            "how many on-topic",
            "how many sources",
            "trust state",
        )
    ):
        facets.add("trust_summary")
    if any(
        marker in text
        for marker in (
            "which of these",
            "which are relevant",
            "which are actual",
            "which returned items",
            "which are off-target",
            "which are off-topic",
            "which are off target",
            "classify these",
            "relevant versus off-target",
            "relevant vs off-target",
            "actual guidance",
            "weak match",
            "weak matches",
        )
    ):
        facets.add("relevance_triage")
    if any(
        marker in text
        for marker in (
            "what does this result mean",
            "result meaning",
            "what status",
            "why was this result",
        )
    ):
        facets.add("result_meaning")
    if not facets and any(
        marker in text
        for marker in (
            "which sources",
            "what sources",
            "which source",
            "which documents",
            "what documents",
            "what records",
        )
    ):
        facets.add("source_overview")
    if explicit_source_reference(question):
        facets.add("specific_source")
    return facets


def _guided_is_usable_answer_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(re.search(r"[A-Za-z0-9]", text))


def _guided_source_metadata_answers(question: str, sources: list[dict[str, Any]]) -> list[str]:
    source = _guided_select_follow_up_source(question, sources)
    if source is None:
        return []

    normalized_question = _guided_normalize_whitespace(question).lower()
    is_source_overview_question = any(
        marker in normalized_question
        for marker in ("which sources", "what sources", "which records", "what records", "which documents")
    )
    raw_citation = source.get("citation")
    citation: dict[str, Any] = cast(dict[str, Any], raw_citation) if isinstance(raw_citation, dict) else {}
    answers: list[str] = []

    if any(marker in normalized_question for marker in ("author", "authors", "who wrote", "written by")):
        raw_authors = citation.get("authors")
        authors: list[Any] = raw_authors if isinstance(raw_authors, list) else []
        author_names = [str(author).strip() for author in authors if str(author).strip()]
        if author_names:
            answers.append("Authors listed for this source: " + ", ".join(author_names) + ".")

    if any(marker in normalized_question for marker in ("venue", "journal", "publisher", "published in")):
        venue = str(citation.get("journalOrPublisher") or "").strip()
        if venue:
            answers.append(f"Venue listed for this source: {venue}.")

    if any(marker in normalized_question for marker in ("doi", "identifier")):
        doi = str(citation.get("doi") or "").strip()
        if doi:
            answers.append(f"DOI listed for this source: {doi}.")

    if any(marker in normalized_question for marker in ("year", "publication year", "published")):
        year = str(citation.get("year") or source.get("date") or "").strip()
        if year:
            answers.append(f"Publication year listed for this source: {year}.")

    if not is_source_overview_question and any(
        marker in normalized_question for marker in ("title", "which paper", "which source", "what paper")
    ):
        title = str(source.get("title") or source.get("sourceId") or "").strip()
        if title:
            answers.append(f"Matched source title: {title}.")

    return answers


def _guided_relevance_triage_answers(
    *,
    session_state: dict[str, Any],
    follow_up_decision: Any,
) -> list[str]:
    sources = [source for source in session_state.get("sources") or [] if isinstance(source, dict)]
    leads = [lead for lead in session_state.get("unverifiedLeads") or [] if isinstance(lead, dict)]
    selected_evidence_ids = set(follow_up_decision.selected_evidence_ids)
    selected_lead_ids = set(follow_up_decision.selected_lead_ids)

    if selected_evidence_ids:
        sources = [
            source
            for source in sources
            if str(source.get("sourceId") or source.get("sourceAlias") or "").strip() in selected_evidence_ids
        ]
    if selected_lead_ids:
        leads = [
            lead
            for lead in leads
            if str(lead.get("sourceId") or lead.get("sourceAlias") or "").strip() in selected_lead_ids
        ]

    strong: list[str] = []
    weak: list[str] = []
    off_target: list[str] = []
    for candidate in sources + leads:
        title = str(candidate.get("title") or candidate.get("sourceId") or "").strip()
        if not title:
            continue
        provider = str(candidate.get("provider") or "unknown provider")
        detail = f"{title} ({provider})"
        topical_relevance = str(candidate.get("topicalRelevance") or "weak_match")
        verification_status = str(candidate.get("verificationStatus") or "unverified")
        is_primary = bool(candidate.get("isPrimarySource"))
        if (
            topical_relevance == "on_topic"
            and verification_status in {"verified_primary_source", "verified_metadata"}
            and is_primary
        ):
            strong.append(detail)
        elif topical_relevance == "off_topic":
            off_target.append(detail)
        else:
            weak.append(detail)

    answers: list[str] = []
    if strong:
        answers.append("Strong on-topic guidance records: " + "; ".join(strong[:3]) + ".")
    if weak:
        answers.append("Related but weaker or less certain records: " + "; ".join(weak[:3]) + ".")
    if off_target:
        answers.append("Off-target records kept only as leads: " + "; ".join(off_target[:3]) + ".")
    if not answers:
        answers.append("The saved session did not contain enough source detail to classify relevance confidently.")
    return answers


def _guided_requested_metadata_facets(question: str) -> set[str]:
    lowered = _guided_normalize_whitespace(question).lower()
    facets: set[str] = set()
    if any(marker in lowered for marker in ("author", "authors", "who wrote", "written by")):
        facets.add("authors")
    if any(marker in lowered for marker in ("venue", "journal", "publisher", "published in")):
        facets.add("venue")
    if any(marker in lowered for marker in ("doi", "identifier")):
        facets.add("identifier")
    if any(marker in lowered for marker in ("publication year", "what year", "year published", "published in")):
        facets.add("year")
    if any(
        marker in lowered
        for marker in (
            "what records",
            "what sources",
            "which documents",
            "what documents",
            "which paper",
            "which source",
        )
    ):
        facets.add("inventory")
    return facets


def _guided_metadata_answer_is_responsive(
    *,
    question: str,
    answer_text: Any,
    sources: list[dict[str, Any]],
    leads: list[dict[str, Any]],
    selected_evidence_ids: list[Any],
    selected_lead_ids: list[Any],
) -> bool:
    requested_facets = _guided_requested_metadata_facets(question)
    if not requested_facets:
        return True

    answer_lower = _guided_normalize_whitespace(answer_text).lower()
    if not answer_lower:
        return False

    selected_source_ids = {
        _guided_normalize_whitespace(identifier)
        for identifier in selected_evidence_ids
        if _guided_normalize_whitespace(identifier)
    }
    selected_lead_ids_set = {
        _guided_normalize_whitespace(identifier)
        for identifier in selected_lead_ids
        if _guided_normalize_whitespace(identifier)
    }
    candidate_records = [
        record
        for record in sources
        if _guided_normalize_whitespace(record.get("sourceId") or record.get("sourceAlias")) in selected_source_ids
    ] + [
        record
        for record in leads
        if _guided_normalize_whitespace(record.get("sourceId") or record.get("sourceAlias")) in selected_lead_ids_set
    ]
    if not candidate_records and len(sources) == 1 and requested_facets - {"inventory"}:
        candidate_records = sources[:1]
    if not candidate_records:
        return False

    citation_payloads = [
        citation for citation in (record.get("citation") for record in candidate_records) if isinstance(citation, dict)
    ]

    def _contains_any(values: list[str]) -> bool:
        return any(value.lower() in answer_lower for value in values if value)

    if "authors" in requested_facets:
        author_values = [
            _guided_normalize_whitespace(author.get("name") if isinstance(author, dict) else author)
            for citation in citation_payloads
            for author in (citation.get("authors") or [])
        ]
        if not author_values or not _contains_any(author_values):
            return False
    if "venue" in requested_facets:
        venue_values = [
            _guided_normalize_whitespace(citation.get("journalOrPublisher")) for citation in citation_payloads
        ]
        if not venue_values or not _contains_any(venue_values):
            return False
    if "identifier" in requested_facets:
        doi_values = [_guided_normalize_whitespace(citation.get("doi")) for citation in citation_payloads]
        if not doi_values or not _contains_any(doi_values):
            return False
    if "year" in requested_facets:
        year_values = []
        for record in candidate_records:
            citation = record.get("citation")
            citation_dict = citation if isinstance(citation, dict) else {}
            year_values.append(_guided_normalize_whitespace(citation_dict.get("year") or record.get("date")))
        if not year_values or not _contains_any(year_values):
            return False
    if "inventory" in requested_facets:
        inventory_values = [
            _guided_normalize_whitespace(record.get("title") or record.get("sourceId") or record.get("sourceAlias"))
            for record in candidate_records
        ]
        if not inventory_values or not _contains_any(inventory_values):
            return False
    return True


def _guided_follow_up_response_mode(question: str, session_strategy_metadata: dict[str, Any]) -> str:
    lowered = question.lower()
    facets = _guided_follow_up_introspection_facets(question)
    if "relevance_triage" in facets:
        return "relevance_triage"
    if any(marker in lowered for marker in ("compare", "versus", "vs", "tradeoff", "tradeoffs")):
        return "comparison"
    if facets or any(
        marker in lowered
        for marker in (
            "author",
            "authors",
            "who wrote",
            "written by",
            "venue",
            "journal",
            "publisher",
            "published in",
            "doi",
            "identifier",
            "publication year",
            "what year",
            "what venue",
            "what records",
            "what sources",
            "which documents",
        )
    ):
        return "metadata"
    if any(marker in lowered for marker in ("mechanism", "pathway", "causal", "how does")):
        return "mechanism_summary"
    if any(
        marker in lowered for marker in ("regulatory history", "timeline", "rulemaking", "listing", "critical habitat")
    ):
        return "regulatory_chain"
    if any(marker in lowered for marker in ("trade-off", "tradeoff", "tradeoffs", "practical implications")):
        return "intervention_tradeoff"
    if any(
        marker in lowered
        for marker in (
            "limitation",
            "limitations",
            "validation",
            "validated",
            "operationally useful",
            "most useful",
            "practical",
            "implementation",
        )
    ):
        return "evidence_planning"
    follow_up_mode = str(session_strategy_metadata.get("followUpMode") or "").strip().lower()
    if follow_up_mode == "comparison":
        return "comparison"
    if follow_up_mode == "claim_check":
        return "evidence_planning"
    return "metadata" if explicit_source_reference(question) else "evidence_planning"


def _guided_follow_up_answer_mode(question: str, session_strategy_metadata: dict[str, Any]) -> str:
    response_mode = _guided_follow_up_response_mode(question, session_strategy_metadata)
    if response_mode == "comparison":
        return "comparison"
    if response_mode in {"mechanism_summary", "regulatory_chain", "intervention_tradeoff"}:
        return "claim_check"
    return "qa"


async def _answer_follow_up_from_session_state(
    *,
    question: str,
    session_state: dict[str, Any] | None,
    response_mode: str,
) -> dict[str, Any] | None:
    if session_state is None:
        return None
    if response_mode not in {"metadata", "relevance_triage"}:
        return None
    answer_parts: list[str] = []
    coverage = cast(dict[str, Any] | None, session_state.get("coverage")) or {}
    failure_summary = cast(dict[str, Any] | None, session_state.get("failureSummary")) or {}
    evidence_gaps = list(session_state.get("evidenceGaps") or [])
    verified_findings = [
        finding for finding in session_state.get("verifiedFindings") or [] if isinstance(finding, dict)
    ]
    sources = [source for source in session_state.get("sources") or [] if isinstance(source, dict)]
    trust_summary = cast(dict[str, Any] | None, session_state.get("trustSummary")) or {}
    facets = _guided_follow_up_introspection_facets(question)
    follow_up_decision = build_follow_up_decision(
        question=question,
        session_state=session_state,
        facets=facets,
    )
    metadata_answers = _guided_source_metadata_answers(question, sources)
    if not facets and not metadata_answers and not follow_up_decision.answer_from_session:
        return None

    answer_parts.extend(metadata_answers)

    if "coverage" in facets:
        attempted = list(coverage.get("providersAttempted") or [])
        succeeded = list(coverage.get("providersSucceeded") or [])
        failed = list(coverage.get("providersFailed") or [])
        zero_results = list(coverage.get("providersZeroResults") or [])
        likely_completeness = str(coverage.get("likelyCompleteness") or "unknown")
        sentences: list[str] = []
        if attempted:
            sentences.append(f"Providers searched were {', '.join(attempted)}.")
        else:
            sentences.append("No provider-attempt summary was saved for this session.")
        if failed:
            sentences.append(f"Failed providers: {', '.join(failed)}.")
        else:
            sentences.append("No provider failures were recorded.")
        if zero_results:
            sentences.append(f"Zero-result providers: {', '.join(zero_results)}.")
        if succeeded:
            sentences.append(f"Successful providers: {', '.join(succeeded)}.")
        if coverage.get("searchMode"):
            sentences.append(f"Search mode was {coverage['searchMode']}.")
        sentences.append(f"Likely completeness was {likely_completeness}.")
        answer_parts.append(" ".join(sentences))

    if "evidence_gaps" in facets:
        if evidence_gaps:
            if "specific" in question.lower() or len(evidence_gaps) == 1:
                answer_parts.append(f"The main evidence gap was: {evidence_gaps[0]}")
            else:
                answer_parts.append("Key evidence gaps were: " + "; ".join(evidence_gaps[:3]) + ".")
        else:
            answer_parts.append("No explicit evidence gaps were recorded in the saved session.")

    if "failure_summary" in facets:
        outcome = str(failure_summary.get("outcome") or "no_failure")
        what_failed = str(failure_summary.get("whatFailed") or "").strip()
        what_still_worked = str(failure_summary.get("whatStillWorked") or "").strip()
        completeness_impact = str(failure_summary.get("completenessImpact") or "").strip()
        fallback_attempted = bool(failure_summary.get("fallbackAttempted"))
        summary_sentences = [f"Failure outcome was {outcome}."]
        if what_failed:
            summary_sentences.append(what_failed)
        if what_still_worked:
            summary_sentences.append(what_still_worked)
        if fallback_attempted:
            fallback_mode = str(failure_summary.get("fallbackMode") or "fallback")
            summary_sentences.append(f"Fallback was attempted via {fallback_mode}.")
        if completeness_impact:
            summary_sentences.append(completeness_impact)
        answer_parts.append(" ".join(summary_sentences))

    if "verified_findings" in facets:
        if verified_findings:
            strongest_finding = str(verified_findings[0].get("claim") or "").strip()
            if strongest_finding:
                answer_parts.append(f"The strongest verified finding in the saved session was: {strongest_finding}.")
            if len(verified_findings) > 1:
                remaining_findings = [
                    str(finding.get("claim") or "").strip()
                    for finding in verified_findings[1:3]
                    if str(finding.get("claim") or "").strip()
                ]
                if remaining_findings:
                    answer_parts.append("Other verified findings included: " + "; ".join(remaining_findings) + ".")
        else:
            answer_parts.append("No verified findings were recorded in the saved session.")

    if "trust_summary" in facets:
        answer_parts.append(
            "Trust summary: "
            f"{int(trust_summary.get('verifiedSourceCount') or 0)} verified source(s), "
            f"{int(trust_summary.get('onTopicSourceCount') or 0)} on-topic, "
            f"{int(trust_summary.get('weakMatchCount') or 0)} weak match, and "
            f"{int(trust_summary.get('offTopicCount') or 0)} off-topic."
        )

    if "result_meaning" in facets:
        result_meaning = str(session_state.get("resultMeaning") or "").strip()
        if result_meaning:
            answer_parts.append(result_meaning)

    if "source_overview" in facets:
        selected_evidence_ids = set(follow_up_decision.selected_evidence_ids)
        selected_sources = [
            source
            for source in sources
            if str(source.get("sourceId") or source.get("sourceAlias") or "").strip() in selected_evidence_ids
        ] or sources
        if selected_sources:
            source_titles = [
                str(source.get("title") or source.get("sourceId") or "").strip()
                for source in selected_sources[:3]
                if str(source.get("title") or source.get("sourceId") or "").strip()
            ]
            if source_titles:
                answer_parts.append("Saved sources included: " + "; ".join(source_titles) + ".")
        else:
            unverified_leads = [lead for lead in session_state.get("unverifiedLeads") or [] if isinstance(lead, dict)]
            selected_lead_ids = set(follow_up_decision.selected_lead_ids)
            selected_leads = [
                lead
                for lead in unverified_leads
                if str(lead.get("sourceId") or lead.get("sourceAlias") or "").strip() in selected_lead_ids
            ] or unverified_leads
            lead_titles = [
                str(lead.get("title") or lead.get("sourceId") or "").strip()
                for lead in selected_leads[:3]
                if str(lead.get("title") or lead.get("sourceId") or "").strip()
            ]
            if lead_titles:
                answer_parts.append("Saved source leads included: " + "; ".join(lead_titles) + ".")
            else:
                answer_parts.append("No saved sources were available for this session.")

    if "relevance_triage" in facets:
        answer_parts.extend(
            _guided_relevance_triage_answers(
                session_state=session_state,
                follow_up_decision=follow_up_decision,
            )
        )

    if "specific_source" in facets:
        source_reference = _guided_extract_source_reference_from_question(question)
        if source_reference:
            source = None
            match_type = "unresolved"
            lower_reference = source_reference.lower()
            session_leads = [lead for lead in session_state.get("unverifiedLeads") or [] if isinstance(lead, dict)]
            source_matches = [
                candidate
                for candidate in sources + session_leads
                if (
                    lower_reference == _guided_normalize_whitespace(candidate.get("sourceId")).lower()
                    or lower_reference == _guided_normalize_whitespace(candidate.get("sourceAlias")).lower()
                    or lower_reference == _guided_normalize_whitespace(candidate.get("citationText")).lower()
                    or lower_reference == _guided_normalize_whitespace(candidate.get("canonicalUrl")).lower()
                    or lower_reference == _guided_normalize_whitespace(candidate.get("retrievedUrl")).lower()
                )
            ]
            if len(source_matches) == 1:
                source = source_matches[0]
                match_type = "session_local_match"
            if source is not None:
                source_title = str(source.get("title") or source.get("sourceId") or "requested source")
                source_provider = str(source.get("provider") or "unknown provider")
                source_relevance = str(source.get("topicalRelevance") or "unknown relevance")
                answer_parts.append(
                    f"Source {source_title} ({source_provider}) was matched via {match_type} "
                    f"with relevance {source_relevance}."
                )
            else:
                answer_parts.append(
                    f"No saved source matched '{source_reference}' in this session. "
                    "Use inspect_source with an exact sourceId for direct inspection."
                )

    answer_parts = [part.strip() for part in answer_parts if _guided_is_usable_answer_text(part)]
    if not answer_parts:
        return None

    # Filter sources to only those referenced in the follow-up answer
    # to avoid re-serializing the entire session source set (payload efficiency).
    _referenced_ids = set(follow_up_decision.selected_evidence_ids + follow_up_decision.selected_lead_ids)
    if _referenced_ids:
        _filtered_sources = [
            s for s in sources if str(s.get("sourceId") or s.get("sourceAlias") or "").strip() in _referenced_ids
        ]
    else:
        _filtered_sources = sources[:3]  # Fallback: cap at 3 when no explicit selection
    _filtered_leads = (
        [
            lead
            for lead in (session_state.get("unverifiedLeads") or [])
            if isinstance(lead, dict)
            and str(lead.get("sourceId") or lead.get("sourceAlias") or "").strip() in _referenced_ids
        ]
        if _referenced_ids
        else []
    )
    _filtered_sources = _guided_dedupe_source_records(_filtered_sources)
    _filtered_leads = _guided_dedupe_source_records(_filtered_leads)
    _filtered_findings = [
        f
        for f in (session_state.get("verifiedFindings") or [])
        if isinstance(f, dict)
        and str(f.get("sourceId") or f.get("claim") or "").strip()
        in {s.get("title") or s.get("sourceId") for s in _filtered_sources}
    ]

    # Ensure trustSummary.authoritativeButWeak is always present (possibly
    # empty) for shape consistency across research / follow_up_research /
    # inspect_source synthesis paths.
    _session_trust_summary = _guided_trust_summary(
        [*_filtered_sources, *_filtered_leads],
        list(session_state.get("evidenceGaps") or []),
    )
    _session_trust_summary.setdefault("authoritativeButWeak", [])
    _session_coverage = _guided_source_coverage_summary(
        sources=_filtered_sources,
        leads=_filtered_leads,
        base_coverage=cast(dict[str, Any] | None, session_state.get("coverage")),
    )

    # P0-1 Fix #2: gate ``answered`` on the saved trust summary. When the
    # session has no on-topic or no verified sources the introspection fast
    # path has nothing grounded to lean on, so fall back to
    # ``insufficient_evidence`` instead of synthesising a confident answer.
    def _coerce_positive_int(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return parsed if parsed > 0 else 0

    _on_topic_count = _coerce_positive_int(_session_trust_summary.get("onTopicSourceCount"))
    _verified_count = _coerce_positive_int(_session_trust_summary.get("verifiedSourceCount"))
    # P0-1 Fix #2: block the introspection fast path from advertising
    # ``answered`` only when the answer rests entirely on topical claims
    # pulled from a weak pool. Metadata-style questions (coverage,
    # evidenceGaps, failureSummary, resultMeaning, trustSummary,
    # verifiedFindings) are always legitimately answerable from saved
    # session structure — they are never topical claims about the domain.
    # Likewise, answers grounded in explicitly selected evidence / lead ids
    # remain legitimate introspection. Only when the answer is built solely
    # from ``follow_up_decision.answer_from_session`` *and* the pool reports
    # zero on-topic / verified sources *and* the decision did not pin any
    # ids do we downgrade to ``insufficient_evidence``.
    _has_meta_backing = bool(facets) or bool(metadata_answers)
    _has_strong_session_evidence = (
        _has_meta_backing
        or _on_topic_count >= 1
        or _verified_count >= 1
        or bool(follow_up_decision.selected_evidence_ids)
        or bool(follow_up_decision.selected_lead_ids)
    )
    _answer_status_value: Literal["answered", "insufficient_evidence"] = (
        "answered" if _has_strong_session_evidence else "insufficient_evidence"
    )
    _answer_text = " ".join(answer_parts) if _has_strong_session_evidence else ""
    _selected_evidence_ids = follow_up_decision.selected_evidence_ids if _has_strong_session_evidence else []
    _selected_lead_ids = follow_up_decision.selected_lead_ids if _has_strong_session_evidence else []

    return {
        "searchSessionId": session_state["searchSessionId"],
        "answerStatus": _answer_status_value,
        "answer": _answer_text,
        "evidence": [],
        "selectedEvidenceIds": _selected_evidence_ids,
        "selectedLeadIds": _selected_lead_ids,
        "unsupportedAsks": [] if _has_strong_session_evidence else [question],
        "followUpQuestions": [],
        "verifiedFindings": _filtered_findings,
        "sources": _filtered_sources,
        "unverifiedLeads": _filtered_leads,
        "evidenceGaps": session_state["evidenceGaps"],
        "trustSummary": _session_trust_summary,
        "coverage": _session_coverage,
        "failureSummary": session_state["failureSummary"],
        "resultMeaning": session_state["resultMeaning"],
        "nextActions": session_state["nextActions"],
        "resultState": _guided_result_state(
            status=_answer_status_value,
            sources=sources,
            evidence_gaps=evidence_gaps,
            search_session_id=str(session_state.get("searchSessionId") or ""),
        ),
        **(
            await _guided_contract_fields(
                query=str(session_state.get("query") or ""),
                intent=str(session_state.get("intent") or "discovery"),
                status=_answer_status_value,
                sources=sources,
                unverified_leads=cast(list[dict[str, Any]], session_state.get("unverifiedLeads") or []),
                evidence_gaps=evidence_gaps,
                coverage_summary=coverage,
                strategy_metadata=cast(dict[str, Any] | None, session_state.get("strategyMetadata")),
                timeline=cast(dict[str, Any] | None, session_state.get("timeline")),
            )
        ),
        "executionProvenance": _guided_execution_provenance_payload(
            execution_mode="session_introspection",
            answer_source="saved_session_metadata",
            passes_run=0,
        ),
        "confidenceSignals": _guided_confidence_signals(
            status=_answer_status_value,
            sources=sources,
            evidence_gaps=evidence_gaps,
            synthesis_mode="session_introspection",
            subject_chain_gaps=cast(
                list[str] | None,
                (session_state.get("strategyMetadata") or {}).get("subjectChainGaps")
                if isinstance(session_state.get("strategyMetadata"), dict)
                else None,
            ),
        ),
    }
