"""Guided session management helpers (Phase 3 extraction).

Extracted from :mod:`paper_chaser_mcp.dispatch._core`. This submodule owns
the helpers that inspect the ``workspace_registry``, resolve saved search
sessions, and rebuild a ``session_state`` payload from a stored record for
``resolve_reference`` / ``inspect_source`` / ``follow_up_research`` flows.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

from ...models.common import SessionCandidate, SessionResolution
from ..normalization import _guided_normalize_whitespace
from .findings import _guided_findings_from_sources, _guided_unverified_leads_from_sources
from .sources import (
    _guided_dedupe_source_records,
    _guided_merge_source_record_sets,
    _guided_merge_source_records,
    _guided_source_coverage_summary,
    _guided_source_matches_reference,
    _guided_source_record_from_paper,
    _guided_source_record_from_structured_source,
    _guided_source_records_share_surface,
)

# Forward references to helpers that still live in ``_core.py``. The imports
# below succeed because this submodule is imported at the bottom of
# ``_core.py`` (after ``_core``'s top-level defs have executed), so the
# partially-initialised ``_core`` module already has the attributes.
from .._core import (  # noqa: E402 — see note above; lint-exempt forward refs
    _find_record_source_with_resolution,
    _guided_failure_summary,
    _guided_follow_up_status,
    _guided_next_actions,
    _guided_record_source_candidates,
    _guided_result_meaning,
    _guided_result_state,
    _guided_sources_all_off_topic,
    _guided_trust_summary,
)

logger = logging.getLogger(__name__)

_GUIDED_RECOVERABLE_SESSION_TOOLS = {"research", "search_papers_smart"}



def _guided_session_exists(
    *,
    workspace_registry: Any,
    search_session_id: str | None,
) -> bool:
    if workspace_registry is None:
        return False
    normalized_id = _guided_normalize_whitespace(search_session_id)
    if not normalized_id:
        return False
    try:
        workspace_registry.get(normalized_id)
    except Exception:
        return False
    return True




def _guided_active_session_ids(workspace_registry: Any) -> list[str]:
    if workspace_registry is None:
        return []
    records = getattr(workspace_registry, "_records", None)
    if not isinstance(records, dict):
        return []
    active: list[tuple[float, str]] = []
    now = time.time()
    for session_id, record in records.items():
        if not isinstance(session_id, str) or not session_id.strip():
            continue
        if record is None:
            continue
        is_expired = getattr(record, "is_expired", None)
        expired = False
        if callable(is_expired):
            try:
                expired = bool(is_expired(now))
            except Exception as exc:
                logger.debug("Failed to evaluate session expiration for %s: %s", session_id, exc)
                expired = True
        if expired:
            continue
        created_at = float(getattr(record, "created_at", 0.0) or 0.0)
        active.append((created_at, session_id))
    active.sort(key=lambda item: item[0], reverse=True)
    return [session_id for _, session_id in active]




def _guided_candidate_records(
    workspace_registry: Any,
    *,
    require_sources: bool = False,
) -> list[Any]:
    if workspace_registry is None:
        return []
    active_records = getattr(workspace_registry, "active_records", None)
    records: list[Any] = []
    if callable(active_records):
        try:
            active = active_records(source_tools=_GUIDED_RECOVERABLE_SESSION_TOOLS)
            records = list(cast(Any, active)) if active is not None else []
        except Exception as exc:
            logger.debug("Failed to read active workspace records: %s", exc)
            records = []
    if not records:
        for session_id in _guided_active_session_ids(workspace_registry):
            record = None
            try:
                record = workspace_registry.get(session_id)
            except Exception as exc:
                logger.debug("Failed to load workspace record %s: %s", session_id, exc)
            if str(getattr(record, "source_tool", "") or "") not in _GUIDED_RECOVERABLE_SESSION_TOOLS:
                continue
            records.append(record)
    if require_sources:
        records = [record for record in records if _guided_record_source_candidates(record)]
    return records




def _guided_latest_compatible_session_id(
    workspace_registry: Any,
    *,
    require_sources: bool = False,
) -> str | None:
    records = _guided_candidate_records(workspace_registry, require_sources=require_sources)
    if not records:
        return None
    return str(getattr(records[0], "search_session_id", "") or "") or None




def _guided_unique_compatible_session_id(
    workspace_registry: Any,
    *,
    require_sources: bool = False,
) -> str | None:
    records = _guided_candidate_records(workspace_registry, require_sources=require_sources)
    if len(records) != 1:
        return None
    return str(getattr(records[0], "search_session_id", "") or "") or None




def _guided_resolve_session_id_for_source(
    workspace_registry: Any,
    source_id: str | None,
) -> tuple[str | None, str | None]:
    normalized_source_id = _guided_normalize_whitespace(source_id)
    if not normalized_source_id:
        return _guided_unique_compatible_session_id(workspace_registry, require_sources=True), None

    matched_records: list[tuple[str | None, str | None]] = []
    for record in _guided_candidate_records(workspace_registry, require_sources=True):
        resolved, match_type = _find_record_source_with_resolution(
            workspace_registry=workspace_registry,
            search_session_id=str(getattr(record, "search_session_id", "") or ""),
            source_id=normalized_source_id,
        )
        if resolved is not None:
            matched_records.append((str(getattr(record, "search_session_id", "") or "") or None, match_type))
    if len(matched_records) == 1:
        return matched_records[0]
    return _guided_unique_compatible_session_id(workspace_registry, require_sources=True), None




def _guided_infer_single_session_id(workspace_registry: Any) -> str | None:
    return _guided_unique_compatible_session_id(workspace_registry)




def _guided_extract_search_session_id(arguments: dict[str, Any]) -> Any:
    return next(
        (
            arguments.get(key)
            for key in (
                "searchSessionId",
                "search_session_id",
                "sessionId",
                "session_id",
                "session",
            )
            if arguments.get(key) is not None
        ),
        None,
    )


def _guided_session_candidates(
    workspace_registry: Any,
    *,
    require_sources: bool = False,
    limit: int = 5,
) -> list[SessionCandidate]:
    now = time.time()
    candidates: list[SessionCandidate] = []
    for record in _guided_candidate_records(workspace_registry, require_sources=require_sources)[:limit]:
        payload = record.payload if isinstance(record.payload, dict) else {}
        sources = _guided_record_source_candidates(record)
        query = str(getattr(record, "query", None) or payload.get("query") or "").strip() or None
        summary = (
            str(payload.get("summary") or (sources[0].get("title") if sources else "") or query or "").strip() or None
        )
        age_seconds = max(0, int(now - float(getattr(record, "created_at", 0.0) or 0.0)))
        candidate = SessionCandidate(
            searchSessionId=str(getattr(record, "search_session_id", "") or ""),
            sourceTool=str(getattr(record, "source_tool", "") or "unknown"),
            query=query,
            summary=summary,
            ageSeconds=age_seconds,
            sourceCount=len(sources),
        )
        candidates.append(candidate)
    return candidates




def _guided_follow_up_session_resolution(
    *,
    arguments: dict[str, Any],
    normalized_arguments: dict[str, Any],
    normalization: dict[str, Any],
    workspace_registry: Any,
) -> dict[str, Any]:
    requested = _guided_normalize_whitespace(_guided_extract_search_session_id(arguments))
    resolved = _guided_normalize_whitespace(normalized_arguments.get("searchSessionId"))
    candidates = _guided_session_candidates(workspace_registry)
    if requested and resolved and requested == resolved:
        mode = "provided_explicitly"
        visible_candidates: list[SessionCandidate] = []
    elif requested and resolved:
        mode = "repaired_to_unique_active_session"
        visible_candidates = []
    elif not requested and resolved:
        mode = "inferred_single_active_session"
        visible_candidates = []
    elif len(candidates) > 1:
        mode = "ambiguous"
        visible_candidates = candidates
    elif requested:
        mode = "session_unavailable"
        visible_candidates = candidates
    else:
        mode = "missing"
        visible_candidates = candidates
    resolution = SessionResolution(
        requestedSearchSessionId=requested,
        resolvedSearchSessionId=resolved,
        resolutionMode=mode,
        warnings=list(normalization.get("warnings") or []),
        candidates=visible_candidates,
    )
    return resolution.model_dump(by_alias=True, exclude_none=True)




def _guided_inspect_session_resolution(
    *,
    arguments: dict[str, Any],
    normalized_arguments: dict[str, Any],
    normalization: dict[str, Any],
    workspace_registry: Any,
) -> dict[str, Any]:
    requested = _guided_normalize_whitespace(_guided_extract_search_session_id(arguments))
    resolved = _guided_normalize_whitespace(normalized_arguments.get("searchSessionId"))
    normalized_source_id = _guided_normalize_whitespace(normalized_arguments.get("sourceId"))
    source_inferred_session_id, _ = _guided_resolve_session_id_for_source(workspace_registry, normalized_source_id)
    candidates = _guided_session_candidates(workspace_registry, require_sources=True)
    if requested and resolved and requested == resolved:
        mode = "provided_explicitly"
        visible_candidates: list[SessionCandidate] = []
    elif requested and resolved and source_inferred_session_id and resolved == source_inferred_session_id:
        mode = "repaired_to_source_bearing_session"
        visible_candidates = []
    elif requested and resolved:
        mode = "repaired_to_unique_active_session"
        visible_candidates = []
    elif not requested and resolved and source_inferred_session_id and resolved == source_inferred_session_id:
        mode = "inferred_source_bearing_session"
        visible_candidates = []
    elif not requested and resolved:
        mode = "inferred_single_active_session"
        visible_candidates = []
    elif len(candidates) > 1:
        mode = "ambiguous"
        visible_candidates = candidates
    elif requested:
        mode = "session_unavailable"
        visible_candidates = candidates
    else:
        mode = "missing"
        visible_candidates = candidates
    resolution = SessionResolution(
        requestedSearchSessionId=requested,
        resolvedSearchSessionId=resolved,
        resolutionMode=mode,
        warnings=list(normalization.get("warnings") or []),
        candidates=visible_candidates,
    )
    return resolution.model_dump(by_alias=True, exclude_none=True)




def _guided_saved_session_topicality(
    candidates: list[dict[str, Any]] | None,
) -> tuple[bool, bool]:
    """Return ``(has_sources, all_off_topic)`` for saved-session candidates.

    Fifth rubber-duck pass: routing the empty-current-response path to
    ``inspect_source`` must distinguish a saved session that still holds
    on-topic/weak-match evidence from one where every stored candidate has
    already been classified ``off_topic``. The latter should fall through to
    ``research`` like the current-response all-off-topic path does.
    """

    items = [c for c in (candidates or []) if isinstance(c, dict)]
    has_sources = bool(items)
    all_off_topic = has_sources and all(
        str(c.get("topicalRelevance") or "").strip().lower() == "off_topic" for c in items
    )
    return has_sources, all_off_topic




def _guided_session_findings(payload: dict[str, Any], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for finding in payload.get("verifiedFindings") or []:
        if isinstance(finding, dict):
            claim = str(finding.get("claim") or "").strip()
            if claim:
                findings.append(
                    {
                        "claim": claim,
                        "supportingSourceIds": list(finding.get("supportingSourceIds") or []),
                        "trustLevel": str(finding.get("trustLevel") or "verified"),
                    }
                )
            continue
        if isinstance(finding, str) and finding.strip():
            supporting_source_ids = [
                str(source.get("sourceId") or "")
                for source in sources
                if finding.strip() in str(source.get("title") or source.get("note") or source.get("sourceId") or "")
            ]
            findings.append(
                {
                    "claim": finding.strip(),
                    "supportingSourceIds": [source_id for source_id in supporting_source_ids if source_id][:1],
                    "trustLevel": "verified",
                }
            )
    return findings or _guided_findings_from_sources(sources)




def _guided_session_state(
    *,
    workspace_registry: Any,
    search_session_id: str | None,
) -> dict[str, Any] | None:
    if workspace_registry is None:
        return None
    normalized_search_session_id = _guided_normalize_whitespace(search_session_id)
    if not normalized_search_session_id:
        return None
    try:
        record = workspace_registry.get(normalized_search_session_id)
    except Exception:
        return None
    payload = record.payload if isinstance(record.payload, dict) else {}
    has_explicit_source_payload = any(
        isinstance(payload.get(key), list) and bool(payload.get(key))
        for key in ("evidence", "sources", "structuredSources", "leads", "candidateLeads", "unverifiedLeads")
    )
    payload_sources = [source for source in payload.get("sources") or [] if isinstance(source, dict)]
    structured_sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for index, source in enumerate(payload.get("structuredSources") or [], start=1)
        if isinstance(source, dict)
    ]
    evidence_sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for index, source in enumerate(payload.get("evidence") or [], start=1)
        if isinstance(source, dict)
    ]
    query = str(record.query or payload.get("query") or "")
    paper_sources = (
        [
            _guided_source_record_from_paper(query, paper, index=index)
            for index, paper in enumerate(record.papers, start=1)
            if isinstance(paper, dict)
        ]
        if not has_explicit_source_payload
        else []
    )
    sources = _guided_merge_source_record_sets(payload_sources, structured_sources, evidence_sources, paper_sources)
    lead_sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for key in ("unverifiedLeads", "leads", "candidateLeads")
        for index, source in enumerate(payload.get(key) or [], start=1)
        if isinstance(source, dict)
    ]
    unverified_leads = _guided_merge_source_record_sets(lead_sources) or _guided_unverified_leads_from_sources(sources)
    verified_findings = _guided_session_findings(payload, sources)
    evidence_gaps = list(payload.get("evidenceGaps") or [])
    coverage = cast(dict[str, Any] | None, payload.get("coverage") or payload.get("coverageSummary"))
    status = _guided_follow_up_status(payload.get("status") or payload.get("answerStatus"))
    failure_summary = _guided_failure_summary(
        failure_summary=cast(dict[str, Any] | None, payload.get("failureSummary")),
        status=status,
        sources=sources,
        evidence_gaps=evidence_gaps,
        all_sources_off_topic=_guided_sources_all_off_topic(sources),
    )
    # Forward ``subjectChainGaps`` recorded on the original
    # ``strategyMetadata`` so that the rebuilt ``trustSummary`` surfaces the
    # same regulatory subject-chain gap explanation seen on the initial
    # research turn. Without this, saved-session follow-ups silently lose the
    # "subject chain incomplete" trust signal.
    strategy_metadata_payload = payload.get("strategyMetadata")
    subject_chain_gaps = (
        [str(item).strip() for item in strategy_metadata_payload.get("subjectChainGaps") or [] if str(item).strip()]
        if isinstance(strategy_metadata_payload, dict)
        else []
    )
    return {
        "searchSessionId": record.search_session_id,
        "query": str(record.query or payload.get("query") or ""),
        "intent": str(payload.get("intent") or payload.get("strategyMetadata", {}).get("intent") or "discovery"),
        "status": status,
        "sources": sources,
        "unverifiedLeads": unverified_leads,
        "verifiedFindings": verified_findings,
        "evidenceGaps": evidence_gaps,
        "trustSummary": _guided_trust_summary(
            [*sources, *unverified_leads],
            evidence_gaps,
            subject_chain_gaps=subject_chain_gaps or None,
        ),
        "coverage": _guided_source_coverage_summary(
            sources=sources,
            leads=unverified_leads,
            base_coverage=coverage,
        ),
        "failureSummary": failure_summary,
        "resultMeaning": payload.get("resultMeaning")
        or _guided_result_meaning(
            status=status,
            verified_findings=verified_findings,
            evidence_gaps=evidence_gaps,
            coverage=coverage,
            failure_summary=failure_summary,
            source_count=len(sources),
            all_sources_off_topic=_guided_sources_all_off_topic(sources),
        ),
        "nextActions": payload.get("nextActions")
        or _guided_next_actions(
            search_session_id=record.search_session_id,
            status=status,
            has_sources=bool(sources),
            all_sources_off_topic=_guided_sources_all_off_topic(sources),
        ),
        "strategyMetadata": cast(
            dict[str, Any] | None,
            payload.get("strategyMetadata") or record.metadata.get("strategyMetadata"),
        ),
        "routingSummary": cast(
            dict[str, Any] | None,
            payload.get("routingSummary") or record.metadata.get("routingSummary"),
        ),
        "timeline": cast(dict[str, Any] | None, payload.get("timeline") or payload.get("regulatoryTimeline")),
        "resultState": payload.get("resultState")
        or _guided_result_state(
            status=status,
            sources=sources,
            evidence_gaps=evidence_gaps,
            search_session_id=record.search_session_id,
        ),
    }




def _guided_enrich_records_from_saved_session(
    current_records: list[dict[str, Any]],
    saved_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in current_records:
        best_match: dict[str, Any] | None = None
        for saved in saved_records:
            if any(
                _guided_source_matches_reference(saved, reference)
                for reference in (
                    record.get("sourceId"),
                    record.get("sourceAlias"),
                    record.get("citationText"),
                    record.get("canonicalUrl"),
                    record.get("retrievedUrl"),
                )
                if _guided_normalize_whitespace(reference)
            ):
                best_match = saved
                break
            if _guided_source_records_share_surface(saved, record):
                best_match = saved
                break
        enriched.append(_guided_merge_source_records(best_match, record) if best_match is not None else record)
    return _guided_dedupe_source_records(enriched)


