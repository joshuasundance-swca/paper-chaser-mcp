"""Structured server-issued cursor utilities for offset-based pagination.

Offset-backed tools (get_paper_citations, get_paper_references, get_paper_authors,
get_author_papers, search_authors) encode continuation state as URL-safe base64
JSON cursors instead of raw integer offsets.

This prevents accidental cursor reuse across tools and aligns with MCP best
practices for opaque server-issued continuation tokens.

Bulk-search cursors (provider tokens) are NOT processed here; they pass through
the dispatch layer unchanged.
"""

import base64
import json
from dataclasses import dataclass

CURSOR_VERSION = 1
PROVIDER = "semantic_scholar"

# Offset-backed tools that use structured cursors
OFFSET_TOOLS: frozenset[str] = frozenset(
    {
        "get_paper_citations",
        "get_paper_references",
        "get_paper_authors",
        "get_author_papers",
        "search_authors",
    }
)


@dataclass
class CursorState:
    """Decoded state carried inside a structured pagination cursor.

    Attributes
    ----------
    tool:
        Name of the MCP tool that issued this cursor.
    provider:
        Data provider name (always ``"semantic_scholar"`` for offset tools).
    offset:
        Integer page offset to pass to the next API call.
    version:
        Schema version for forward-compatibility.
    context_hash:
        Optional hash of query context (reserved for future use).
    """

    tool: str
    provider: str
    offset: int
    version: int = CURSOR_VERSION
    context_hash: str | None = None


def encode_cursor(payload: dict) -> str:
    """Encode a cursor payload dict as a URL-safe base64 JSON string.

    Parameters
    ----------
    payload:
        Dictionary with at minimum the keys ``tool``, ``provider``,
        ``offset``, and ``version``.

    Returns
    -------
    str
        URL-safe base64-encoded JSON cursor string.
    """
    json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(json_bytes).decode("ascii")


def decode_cursor(cursor: str) -> CursorState:
    """Decode a structured cursor string into a :class:`CursorState`.

    Parameters
    ----------
    cursor:
        A URL-safe base64-encoded JSON cursor string as produced by
        :func:`encode_cursor`.

    Returns
    -------
    CursorState
        The decoded cursor state.

    Raises
    ------
    ValueError
        If the cursor cannot be decoded or is missing required fields.
    """
    try:
        json_bytes = base64.urlsafe_b64decode(cursor.encode("ascii"))
        payload = json.loads(json_bytes)
    except Exception as exc:
        raise ValueError(
            f"Corrupted pagination cursor {cursor!r}: cannot decode. "
            "Restart the request without a cursor."
        ) from exc

    try:
        return CursorState(
            tool=payload["tool"],
            provider=payload["provider"],
            offset=int(payload["offset"]),
            version=int(payload.get("version", CURSOR_VERSION)),
            context_hash=payload.get("context_hash"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Corrupted pagination cursor {cursor!r}: missing required fields. "
            "Restart the request without a cursor."
        ) from exc


def is_legacy_offset(cursor: str) -> bool:
    """Return ``True`` if *cursor* looks like a legacy integer offset string.

    Legacy offset cursors are plain stringified integers, e.g. ``"100"``.
    Structured cursors are URL-safe base64 JSON and will not parse as integers.

    Parameters
    ----------
    cursor:
        The cursor string to inspect.
    """
    try:
        int(cursor)
        return True
    except (ValueError, TypeError):
        return False


def cursor_from_offset(tool: str, offset: int) -> str:
    """Build a structured cursor encoding *offset* for *tool*.

    Parameters
    ----------
    tool:
        Name of the MCP tool issuing the cursor.
    offset:
        Next page offset.

    Returns
    -------
    str
        URL-safe base64 cursor string.
    """
    return encode_cursor(
        {
            "tool": tool,
            "provider": PROVIDER,
            "offset": offset,
            "version": CURSOR_VERSION,
        }
    )
