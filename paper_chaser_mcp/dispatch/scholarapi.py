"""ScholarAPI-specific dispatch helpers.

Extracted from :mod:`paper_chaser_mcp.dispatch._core` as part of the Phase 5
modularization. These helpers are pure functions that translate ScholarAPI
provider exceptions and payloads into :mod:`paper_chaser_mcp.provider_runtime`
status buckets, fallback reasons, and quota metadata. Behavior is preserved
verbatim; only the module boundary moves.

``_provider_error_text`` is intentionally co-located here because the Phase 5
audit showed it is used exclusively by the ScholarAPI status-bucket accounting
path (``_call_explicit_scholarapi_tool`` in ``_core``). If a second caller
appears outside the ScholarAPI subsystem, promote it to a shared
``dispatch/errors.py`` helper rather than growing its scope silently here.
"""

from __future__ import annotations

from typing import Any

from ..clients.scholarapi import (
    ScholarApiError,
    ScholarApiKeyMissingError,
    ScholarApiQuotaError,
    ScholarApiUpstreamError,
)
from ..provider_runtime import ProviderStatusBucket


def _provider_error_text(exc: Exception) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _scholarapi_status_bucket(exc: Exception) -> ProviderStatusBucket:
    if isinstance(exc, ScholarApiKeyMissingError):
        return "auth_error"
    if isinstance(exc, ScholarApiQuotaError):
        return "quota_exhausted"
    if isinstance(exc, ScholarApiUpstreamError):
        return "provider_error"
    if isinstance(exc, ScholarApiError):
        return "provider_error"
    return "provider_error"


def _scholarapi_fallback_reason(status_bucket: ProviderStatusBucket) -> str | None:
    reasons = {
        "auth_error": "Provider authentication failed.",
        "quota_exhausted": "Provider quota was exhausted.",
        "provider_error": "Provider returned an upstream error.",
    }
    return reasons.get(status_bucket)


def _scholarapi_quota_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metadata: dict[str, Any] = {}
    request_id = payload.get("requestId")
    if isinstance(request_id, str) and request_id.strip():
        metadata["requestId"] = request_id.strip()
    request_cost = payload.get("requestCost")
    if request_cost is not None:
        metadata["requestCost"] = request_cost
    pagination = payload.get("pagination")
    if isinstance(pagination, dict):
        if "hasMore" in pagination:
            metadata["hasMore"] = bool(pagination.get("hasMore"))
        next_cursor = pagination.get("nextCursor")
        if next_cursor is not None:
            metadata["nextCursor"] = next_cursor
    return metadata


def _scholarapi_payload_is_empty(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return payload is None
    if isinstance(payload.get("data"), list):
        return len(payload.get("data") or []) == 0
    if isinstance(payload.get("results"), list):
        return len(payload.get("results") or []) == 0
    return False
