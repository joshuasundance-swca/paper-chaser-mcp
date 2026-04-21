"""Guided research-argument normalization helpers (Phase 3 extraction).

Helpers that sanitize and repair incoming arguments to guided tools
(``research``, ``follow_up_research``, ``inspect_source``) and produce
``InputNormalization`` payloads. Extracted from
:mod:`paper_chaser_mcp.dispatch._core`.
"""

from __future__ import annotations

import logging
from typing import Any

from ...models.common import InputNormalization, NormalizationRepair
from ..normalization import (
    _guided_normalize_citation_surface,
    _guided_normalize_whitespace,
    _guided_normalize_year_hint,
    _guided_strip_research_prefix,
)
from .inspect_source import _guided_extract_question
from .resolve_reference import _guided_note_repair
from .sessions import (
    _guided_candidate_records,
    _guided_extract_search_session_id,
    _guided_infer_single_session_id,
    _guided_session_exists,
)
from .sources import _guided_extract_source_id
from .trust import _guided_record_source_candidates


def _guided_normalize_research_arguments(arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_args = dict(arguments)
    repairs: list[dict[str, str]] = []
    warnings: list[str] = []

    raw_query = _guided_normalize_whitespace(arguments.get("query"))
    normalized_query = _guided_normalize_citation_surface(_guided_strip_research_prefix(raw_query))
    if not normalized_query:
        raw_focus = _guided_normalize_whitespace(arguments.get("focus"))
        if raw_focus:
            normalized_query = _guided_normalize_citation_surface(raw_focus)
            warnings.append("query was empty; reused normalized focus as the research query.")
    normalized_args["query"] = normalized_query or raw_query
    _guided_note_repair(
        repairs,
        field="query",
        original=arguments.get("query"),
        normalized=normalized_args["query"],
        reason="query_normalization",
    )

    normalized_focus = _guided_normalize_citation_surface(_guided_normalize_whitespace(arguments.get("focus")))
    normalized_args["focus"] = normalized_focus or None
    _guided_note_repair(
        repairs,
        field="focus",
        original=arguments.get("focus"),
        normalized=normalized_args["focus"],
        reason="focus_normalization",
    )

    normalized_venue = _guided_normalize_whitespace(arguments.get("venue")) or None
    normalized_args["venue"] = normalized_venue
    _guided_note_repair(
        repairs,
        field="venue",
        original=arguments.get("venue"),
        normalized=normalized_venue,
        reason="venue_normalization",
    )

    normalized_year = _guided_normalize_year_hint(arguments.get("year"))
    normalized_args["year"] = normalized_year
    _guided_note_repair(
        repairs,
        field="year",
        original=arguments.get("year"),
        normalized=normalized_year,
        reason="year_normalization",
    )

    normalization = {
        "normalizedQuery": normalized_args.get("query"),
        "normalizedFocus": normalized_args.get("focus"),
        "normalizedVenue": normalized_args.get("venue"),
        "normalizedYear": normalized_args.get("year"),
        "repairs": repairs,
        "warnings": warnings,
    }
    return normalized_args, normalization


def _guided_normalize_follow_up_arguments(
    arguments: dict[str, Any],
    *,
    workspace_registry: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_args = dict(arguments)
    for alias in ("search_session_id", "sessionId", "session_id", "session", "prompt", "query"):
        normalized_args.pop(alias, None)
    repairs: list[dict[str, str]] = []
    warnings: list[str] = []

    raw_search_session_id = _guided_extract_search_session_id(arguments)
    normalized_search_session_id: str | None = _guided_normalize_whitespace(raw_search_session_id)
    if normalized_search_session_id and not _guided_session_exists(
        workspace_registry=workspace_registry,
        search_session_id=normalized_search_session_id,
    ):
        inferred_id = _guided_infer_single_session_id(workspace_registry)
        if inferred_id is not None:
            warnings.append(
                "searchSessionId "
                f"'{normalized_search_session_id}' was unavailable; using active session '{inferred_id}'."
            )
            normalized_search_session_id = inferred_id
        else:
            warnings.append(
                f"searchSessionId '{normalized_search_session_id}' was unavailable and could not be repaired safely."
            )
            normalized_search_session_id = None
    if not normalized_search_session_id:
        inferred_id = _guided_infer_single_session_id(workspace_registry)
        if inferred_id is not None:
            normalized_search_session_id = inferred_id
            warnings.append(f"searchSessionId was missing; inferred active session '{inferred_id}'.")
        elif len(_guided_candidate_records(workspace_registry)) > 1:
            warnings.append(
                "searchSessionId was missing and multiple active sessions exist; provide an explicit searchSessionId."
            )
    normalized_args["searchSessionId"] = normalized_search_session_id
    _guided_note_repair(
        repairs,
        field="searchSessionId",
        original=raw_search_session_id,
        normalized=normalized_search_session_id,
        reason="session_id_normalization",
    )

    raw_question = _guided_extract_question(arguments)
    normalized_question = _guided_normalize_whitespace(raw_question)
    normalized_args["question"] = normalized_question
    _guided_note_repair(
        repairs,
        field="question",
        original=raw_question,
        normalized=normalized_question,
        reason="question_normalization",
    )
    if not normalized_question:
        warnings.append("question was empty after normalization; follow-up quality may be limited.")

    normalization = {
        "normalizedSearchSessionId": normalized_args.get("searchSessionId"),
        "normalizedQuestion": normalized_args.get("question"),
        "repairs": repairs,
        "warnings": warnings,
    }
    return normalized_args, normalization


def _guided_normalize_inspect_arguments(
    arguments: dict[str, Any],
    *,
    workspace_registry: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_args = dict(arguments)
    for alias in (
        "search_session_id",
        "sessionId",
        "session_id",
        "session",
        "source_id",
        "source",
        "sourceRef",
        "evidenceId",
        "evidence_id",
        "leadId",
        "lead_id",
        "id",
    ):
        normalized_args.pop(alias, None)
    repairs: list[dict[str, str]] = []
    warnings: list[str] = []

    raw_source_id = _guided_extract_source_id(arguments)
    normalized_source_id = _guided_normalize_whitespace(raw_source_id)

    raw_search_session_id = _guided_extract_search_session_id(arguments)
    normalized_search_session_id: str | None = _guided_normalize_whitespace(raw_search_session_id)
    if normalized_search_session_id and not _guided_session_exists(
        workspace_registry=workspace_registry,
        search_session_id=normalized_search_session_id,
    ):
        inferred_id = _guided_infer_single_session_id(workspace_registry)
        if inferred_id is not None:
            warnings.append(
                "searchSessionId "
                f"'{normalized_search_session_id}' was unavailable; using active session '{inferred_id}'."
            )
            normalized_search_session_id = inferred_id
        else:
            warnings.append(
                f"searchSessionId '{normalized_search_session_id}' was unavailable and could not be repaired safely."
            )
            normalized_search_session_id = None
    if not normalized_search_session_id:
        inferred_id = _guided_infer_single_session_id(workspace_registry)
        if inferred_id is not None:
            normalized_search_session_id = inferred_id
            warnings.append(f"searchSessionId was missing; inferred active session '{inferred_id}'.")
        elif len(_guided_candidate_records(workspace_registry)) > 1:
            warnings.append(
                "searchSessionId was missing and multiple active sessions exist; provide an explicit searchSessionId."
            )
    normalized_args["searchSessionId"] = normalized_search_session_id
    _guided_note_repair(
        repairs,
        field="searchSessionId",
        original=raw_search_session_id,
        normalized=normalized_search_session_id,
        reason="session_id_normalization",
    )

    if not normalized_source_id and normalized_search_session_id and workspace_registry is not None:
        record = None
        try:
            record = workspace_registry.get(normalized_search_session_id)
        except Exception as exc:
            logger.debug(
                "Failed to load workspace record %s while inferring sourceId: %s",
                normalized_search_session_id,
                exc,
            )
        if record is not None:
            candidates = _guided_record_source_candidates(record)
            if len(candidates) == 1:
                normalized_source_id = _guided_normalize_whitespace(candidates[0].get("sourceId"))
                warnings.append(f"sourceId was missing; inferred the only inspectable source '{normalized_source_id}'.")
    normalized_args["sourceId"] = normalized_source_id
    _guided_note_repair(
        repairs,
        field="sourceId",
        original=raw_source_id,
        normalized=normalized_source_id,
        reason="source_id_normalization",
    )
    if not normalized_source_id:
        warnings.append("sourceId was empty after normalization; source inspection may fail.")

    normalization = {
        "normalizedSearchSessionId": normalized_args.get("searchSessionId"),
        "normalizedSourceId": normalized_args.get("sourceId"),
        "repairs": repairs,
        "warnings": warnings,
    }
    return normalized_args, normalization


def _guided_normalization_payload(normalization: dict[str, Any]) -> dict[str, Any] | None:
    repairs = [repair for repair in normalization.get("repairs") or [] if isinstance(repair, dict)]
    warnings = [warning for warning in normalization.get("warnings") or [] if isinstance(warning, str) and warning]
    if not repairs and not warnings:
        return None
    payload = InputNormalization.model_validate(
        {
            **normalization,
            "repairs": [
                NormalizationRepair.model_validate(repair).model_dump(by_alias=True, exclude_none=True)
                for repair in repairs
            ],
            "warnings": warnings,
        }
    )
    return payload.model_dump(by_alias=True, exclude_none=True)


logger = logging.getLogger(__name__)
