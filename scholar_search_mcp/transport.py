"""Shared transport modules for compatibility and monkeypatching in tests."""

import asyncio
import inspect
import logging
import ssl
from typing import Any

import httpx

logger = logging.getLogger("scholar-search-mcp")


def build_httpx_verify_config(
    *,
    verify_tls: bool = True,
    ca_bundle: str | None = None,
    prefer_system_store: bool = False,
) -> bool | str | ssl.SSLContext:
    """Build an httpx verify configuration with optional system trust fallback."""

    if not verify_tls:
        return False
    normalized_bundle = str(ca_bundle or "").strip()
    if normalized_bundle:
        return normalized_bundle
    if prefer_system_store:
        return ssl.create_default_context()
    return True


def is_tls_verification_error(error: Exception) -> bool:
    """Return True when an exception looks like a certificate verification failure."""

    message = str(error)
    return "certificate verify failed" in message.lower() or "CERTIFICATE_VERIFY_FAILED" in message


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
                "Ignoring async resource close after the original event loop has already been torn down.",
                exc_info=True,
            )
        return


__all__ = [
    "asyncio",
    "build_httpx_verify_config",
    "httpx",
    "is_tls_verification_error",
    "maybe_close_async_resource",
]
