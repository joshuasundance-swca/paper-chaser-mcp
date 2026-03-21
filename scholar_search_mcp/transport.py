"""Shared transport modules for compatibility and monkeypatching in tests."""

import asyncio
import inspect
import logging
from typing import Any

import httpx

logger = logging.getLogger("scholar-search-mcp")


async def maybe_close_async_resource(resource: Any) -> None:
    """Close an async-capable client or adapter when it exposes a close hook."""
    if resource is None:
        return
    for method_name in ("aclose", "close"):
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
            if inspect.isawaitable(result):
                await result
        except RuntimeError as error:
            if "Event loop is closed" not in str(error):
                raise
            logger.debug(
                "Ignoring async resource close after the original event loop "
                "has already been torn down.",
                exc_info=True,
            )
        return


__all__ = ["asyncio", "httpx", "maybe_close_async_resource"]
