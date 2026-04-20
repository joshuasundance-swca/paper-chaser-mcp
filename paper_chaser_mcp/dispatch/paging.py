"""Cursor encode/decode helpers for guided pagination.

Extracted from ``paper_chaser_mcp.dispatch._core`` as part of the Phase 2
refactor. These helpers are pure (no I/O) and cover the one-pair-at-a-time
structured cursor flow used by ``search_papers``, ``search_papers_bulk``,
and the other offset-backed tools.

The matching bulk-token helpers remain in ``_core`` for now; they are
closely interleaved with provider-specific retrieval code and will move in
a later phase.
"""

from __future__ import annotations

from typing import Any

from ..utils.cursor import (
    PROVIDER,
    SUPPORTED_VERSIONS,
    cursor_from_offset,
    decode_cursor,
    is_legacy_offset,
)

CURSOR_REUSE_HINT = (
    "Pass pagination.nextCursor back exactly as returned. Do not derive, edit, "
    "or fabricate cursors, and do not reuse them across a different tool or "
    "different query context."
)


def _cursor_to_offset(
    cursor: str | None,
    tool: str | None = None,
    context_hash: str | None = None,
    expected_provider: str = PROVIDER,
) -> int | None:
    """Decode an opaque pagination cursor to an integer offset.

    Accepts both structured server-issued cursors (URL-safe base64 JSON) and
    legacy plain integer strings for backward compatibility.

    Returns ``None`` when *cursor* is ``None`` (start from the beginning).

    When *tool* is provided, structured cursors are validated to ensure they
    were issued by the same tool.  When *context_hash* is also provided, the
    cursor's embedded hash is compared against the current request's context,
    so a cursor from a different query on the same tool is rejected.

    In production all dispatch handlers pass *tool* and *context_hash*
    explicitly.  Passing ``tool=None`` skips cross-tool validation; this is
    intentional only for the ``cursor=None`` early-return path.

    Raises ``ValueError`` for stale, mistyped, corrupted, cross-tool, or
    cross-query cursors, unsupported versions, unknown providers, and negative
    integer offsets.
    """
    if cursor is None:
        return None
    if is_legacy_offset(cursor):
        offset = int(cursor)
        if offset < 0:
            raise ValueError(
                f"Invalid pagination cursor {cursor!r}: offset must be non-negative. "
                "code=INVALID_CURSOR. "
                f"{CURSOR_REUSE_HINT} Restart the request without a cursor."
            )
        return offset
    # Structured cursor: decode and validate
    try:
        state = decode_cursor(cursor)
    except ValueError:
        raise ValueError(
            f"Invalid pagination cursor {cursor!r}: cannot be decoded. "
            "code=INVALID_CURSOR. "
            f"{CURSOR_REUSE_HINT} Restart the request without a cursor."
        )
    if state.provider != expected_provider:
        raise ValueError(
            f"Invalid pagination cursor: cursor provider {state.provider!r} does not "
            f"match expected provider {expected_provider!r}. "
            "code=INVALID_CURSOR. "
            f"{CURSOR_REUSE_HINT} Restart the request without a cursor."
        )
    if state.version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Invalid pagination cursor: cursor version {state.version} is not "
            f"supported (supported: {sorted(SUPPORTED_VERSIONS)}). "
            "code=INVALID_CURSOR. "
            f"{CURSOR_REUSE_HINT} Restart the request without a cursor."
        )
    if tool is not None and state.tool != tool:
        raise ValueError(
            f"Invalid pagination cursor: cursor was issued by tool {state.tool!r} "
            f"but is being used with tool {tool!r}. "
            "code=INVALID_CURSOR. "
            f"{CURSOR_REUSE_HINT} Restart the request without a cursor."
        )
    if context_hash is not None and state.context_hash is not None and state.context_hash != context_hash:
        raise ValueError(
            "Invalid pagination cursor: cursor was issued for a different query "
            "context and cannot be reused here. "
            "code=INVALID_CURSOR. "
            f"{CURSOR_REUSE_HINT} Restart the request without a cursor."
        )
    return state.offset


def _encode_next_cursor(
    result: dict[str, Any],
    tool: str,
    context_hash: str | None = None,
    provider: str = PROVIDER,
) -> dict[str, Any]:
    """Re-encode a plain integer ``nextCursor`` in *result* as a structured cursor.

    Operates on the serialized dict returned by ``dump_jsonable``.  The
    ``pagination`` key is present on all offset-backed tool responses.

    The *context_hash* is embedded into the new cursor so that future requests
    can validate that the cursor belongs to the same query stream.

    If ``pagination.nextCursor`` is already a structured (non-integer) cursor or
    is ``None``, the result is returned unchanged.
    """
    pagination = result.get("pagination")
    if not isinstance(pagination, dict):
        return result
    raw_cursor = pagination.get("nextCursor")
    if raw_cursor is None:
        return result
    if is_legacy_offset(raw_cursor):
        pagination["nextCursor"] = cursor_from_offset(
            tool,
            int(raw_cursor),
            context_hash=context_hash,
            provider=provider,
        )
    return result


__all__ = ("CURSOR_REUSE_HINT", "_cursor_to_offset", "_encode_next_cursor")
