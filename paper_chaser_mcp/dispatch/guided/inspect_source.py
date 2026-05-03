"""Guided ``inspect_source`` / saved-session helpers (Phase 3 extraction).

Helpers that build the guided ``inspect_source`` payload and select/augment
saved-session records for inspection and follow-up. Extracted from
:mod:`paper_chaser_mcp.dispatch._core`.
"""

from __future__ import annotations

from typing import Any

from ...guided_semantic import explicit_source_reference
from ...models.common import SourceResolution
from .._core import (  # noqa: E402 — forward refs
    _candidate_is_inspectable,
)
from ..normalization import _guided_normalize_whitespace
from .sources import (
    _guided_dedupe_source_records,
    _guided_source_matches_reference,
)


def _guided_extract_question(arguments: dict[str, Any]) -> Any:
    return next(
        (arguments.get(key) for key in ("question", "prompt", "query") if arguments.get(key) is not None),
        None,
    )


def _guided_compact_source_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Project a saved-session source candidate to its disambiguation-critical fields.

    Includes inspectability-signaling fields (canonicalUrl / retrievedUrl /
    fullTextUrlFound / abstractObserved) so agents can verify a candidate
    would be inspectable without a round trip.
    """
    keep_keys = (
        "sourceId",
        "title",
        "topicalRelevance",
        "canonicalUrl",
        "retrievedUrl",
        "fullTextUrlFound",
        "abstractObserved",
        "confidence",
        "accessStatus",
        "verificationStatus",
        "publicationYear",
    )
    projected: dict[str, Any] = {}
    for key in keep_keys:
        value = candidate.get(key)
        if value not in (None, "", [], {}):
            projected[key] = value
    candidate_rationale = next(
        (
            str(candidate.get(key) or "").strip()
            for key in ("whyClassifiedAsWeakMatch", "whyNotVerified", "leadReason", "note")
            if str(candidate.get(key) or "").strip()
        ),
        "",
    )
    if candidate_rationale:
        projected["candidateRationale"] = candidate_rationale
    return projected


def _guided_source_resolution_payload(
    *,
    requested_source_id: str | None,
    resolved_source_id: str | None,
    match_type: str | None,
    available_source_ids: list[str] | None = None,
    available_candidates: list[dict[str, Any]] | None = None,
    candidates_have_inspectable: bool | None = None,
) -> dict[str, Any]:
    compact_candidates: list[dict[str, Any]] = []
    if available_candidates:
        for candidate in available_candidates:
            if not isinstance(candidate, dict):
                continue
            projection = _guided_compact_source_candidate(candidate)
            if projection.get("sourceId"):
                compact_candidates.append(projection)
    if candidates_have_inspectable is None:
        candidates_have_inspectable = any(_candidate_is_inspectable(candidate) for candidate in compact_candidates)
    resolution = SourceResolution(
        requestedSourceId=_guided_normalize_whitespace(requested_source_id),
        resolvedSourceId=_guided_normalize_whitespace(resolved_source_id),
        matchType=match_type,
        availableSourceIds=available_source_ids or [],
        availableSourceCandidates=compact_candidates,
        candidatesHaveInspectable=bool(candidates_have_inspectable) if compact_candidates else False,
    )
    return resolution.model_dump(by_alias=True, exclude_none=True)


def _guided_extract_source_reference_from_question(question: str) -> str | None:
    return explicit_source_reference(question)


def _guided_select_follow_up_source(question: str, sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not sources:
        return None
    if len(sources) == 1:
        return sources[0]

    source_reference = _guided_extract_source_reference_from_question(question)
    if source_reference:
        lowered_reference = source_reference.lower()
        for source in sources:
            if lowered_reference == _guided_normalize_whitespace(source.get("sourceId")).lower():
                return source
            if lowered_reference == _guided_normalize_whitespace(source.get("sourceAlias")).lower():
                return source
    return None


def _guided_append_selected_saved_records(
    current_records: list[dict[str, Any]],
    saved_records: list[dict[str, Any]],
    selected_ids: list[Any],
) -> list[dict[str, Any]]:
    augmented = list(current_records)
    for selected_id in selected_ids:
        normalized_selected_id = _guided_normalize_whitespace(selected_id)
        if not normalized_selected_id:
            continue
        if any(_guided_source_matches_reference(record, normalized_selected_id) for record in augmented):
            continue
        matched_saved = next(
            (saved for saved in saved_records if _guided_source_matches_reference(saved, normalized_selected_id)),
            None,
        )
        if matched_saved is not None:
            augmented.append(matched_saved)
    return _guided_dedupe_source_records(augmented)
