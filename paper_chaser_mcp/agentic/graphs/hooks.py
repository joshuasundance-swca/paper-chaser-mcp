"""Graph hooks and lifecycle helpers.

Phase 7a extraction: module-level helpers that previously lived as static
methods on :class:`AgenticRuntime`. These helpers are pure (or side-effect
only), stateless, and therefore safe to move off the class without touching
``self``. ``AgenticRuntime`` keeps thin compatibility wrappers so legacy
call sites (tests that patch ``runtime._skip_context_notifications`` etc.)
continue to work.

See ``docs/seam-maps/graphs.md`` for the extraction plan.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastmcp import Context

from ..retrieval import RetrievalBatch

logger = logging.getLogger("paper-chaser-mcp")


def _skip_context_notifications(ctx: Context) -> bool:
    """Return True when ``ctx`` is a stdio transport that should not receive
    non-essential context notifications."""

    transport = getattr(ctx, "transport", None)
    if not isinstance(transport, str):
        return False
    return transport.lower() == "stdio"


def _consume_background_task(task: asyncio.Task[Any]) -> None:
    """Consume the result of a best-effort background notification task.

    Swallows ``CancelledError`` silently and logs any other exception at
    ``DEBUG`` so background notification failures never bubble up to the
    user-facing search path.
    """

    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.debug(
            "Best-effort context notification failed.",
            exc_info=True,
        )


def _truncate_text(value: str, *, limit: int = 72) -> str:
    """Collapse internal whitespace in ``value`` and truncate it to ``limit``
    characters, appending an ellipsis if the original was longer."""

    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(limit - 3, 1)].rstrip()}..."


def _describe_retrieval_batch(batch: RetrievalBatch) -> str:
    """Summarise a :class:`RetrievalBatch` into a one-line human-readable
    description suitable for context notifications."""

    providers_text = ", ".join(batch.providers_used) if batch.providers_used else "none"
    message = (
        f"Variant '{_truncate_text(batch.variant)}' finished with "
        f"{len(batch.candidates)} candidate(s) from {providers_text}."
    )
    if batch.provider_errors:
        errors_text = "; ".join(
            f"{provider}: {_truncate_text(error, limit=90)}"
            for provider, error in sorted(batch.provider_errors.items())
        )
        message = f"{message} Errors: {errors_text}."
    return message
