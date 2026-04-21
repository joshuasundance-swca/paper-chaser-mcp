"""Phase 5 extraction pin: ``dispatch.scholarapi`` submodule.

Pins that ScholarAPI provider helpers live at ``paper_chaser_mcp.dispatch.scholarapi``
after the Phase 5 modularization. These direct imports would have failed before
the extraction (RED bar); once ``_core.py`` delegates to the new submodule they
go green.
"""

from __future__ import annotations

import pytest

from paper_chaser_mcp.clients.scholarapi import (
    ScholarApiError,
    ScholarApiKeyMissingError,
    ScholarApiQuotaError,
    ScholarApiUpstreamError,
)
from paper_chaser_mcp.dispatch.scholarapi import (
    _provider_error_text,
    _scholarapi_fallback_reason,
    _scholarapi_payload_is_empty,
    _scholarapi_quota_metadata,
    _scholarapi_status_bucket,
)


@pytest.mark.parametrize(
    "exc, expected",
    [
        (ScholarApiKeyMissingError("missing"), "auth_error"),
        (ScholarApiQuotaError("quota"), "quota_exhausted"),
        (ScholarApiUpstreamError("bad"), "provider_error"),
        (ScholarApiError("base"), "provider_error"),
        (RuntimeError("generic"), "provider_error"),
    ],
)
def test_status_bucket_maps_exception_classes(exc: Exception, expected: str) -> None:
    assert _scholarapi_status_bucket(exc) == expected


@pytest.mark.parametrize(
    "bucket, expected",
    [
        ("auth_error", "Provider authentication failed."),
        ("quota_exhausted", "Provider quota was exhausted."),
        ("provider_error", "Provider returned an upstream error."),
        ("success", None),
    ],
)
def test_fallback_reason_table(bucket: str, expected: str | None) -> None:
    assert _scholarapi_fallback_reason(bucket) == expected  # type: ignore[arg-type]


def test_quota_metadata_extracts_fields() -> None:
    payload = {
        "requestId": "  req-42  ",
        "requestCost": 0.25,
        "pagination": {"hasMore": 1, "nextCursor": "cursor-token"},
    }
    meta = _scholarapi_quota_metadata(payload)
    assert meta == {
        "requestId": "req-42",
        "requestCost": 0.25,
        "hasMore": True,
        "nextCursor": "cursor-token",
    }


def test_quota_metadata_handles_non_dict_and_missing_fields() -> None:
    assert _scholarapi_quota_metadata(None) == {}
    assert _scholarapi_quota_metadata("string") == {}
    assert _scholarapi_quota_metadata({"pagination": "not-a-dict"}) == {}


def test_payload_is_empty_rules() -> None:
    assert _scholarapi_payload_is_empty(None) is True
    assert _scholarapi_payload_is_empty({"data": []}) is True
    assert _scholarapi_payload_is_empty({"results": []}) is True
    assert _scholarapi_payload_is_empty({"data": [{"id": 1}]}) is False
    assert _scholarapi_payload_is_empty({"results": [{"id": 1}]}) is False
    assert _scholarapi_payload_is_empty({"other": "ok"}) is False
    # Non-dict, non-None returns False per current contract
    assert _scholarapi_payload_is_empty(42) is False


def test_provider_error_text_formats_exception() -> None:
    assert _provider_error_text(RuntimeError("boom")) == "RuntimeError: boom"
    # Empty message falls back to class name alone.
    assert _provider_error_text(RuntimeError("")) == "RuntimeError"


def test_scholarapi_helpers_remain_reexported_from_core() -> None:
    """Moved helpers must still be reachable via ``_core`` for facade stability."""
    from paper_chaser_mcp.dispatch import _core

    assert _core._scholarapi_status_bucket is _scholarapi_status_bucket
    assert _core._scholarapi_fallback_reason is _scholarapi_fallback_reason
    assert _core._scholarapi_quota_metadata is _scholarapi_quota_metadata
    assert _core._scholarapi_payload_is_empty is _scholarapi_payload_is_empty
    assert _core._provider_error_text is _provider_error_text
