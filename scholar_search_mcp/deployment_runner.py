"""Launch the deployment wrapper with container-friendly bind defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping

from uvicorn import run

# Container platforms must bind on all interfaces so the platform ingress can
# reach the MCP server process.
DEFAULT_HOST = "0.0.0.0"  # nosec B104
DEFAULT_PORT = 8080


def resolve_bind_host(env: Mapping[str, str] | None = None) -> str:
    source = env or os.environ
    candidate = source.get("SCHOLAR_SEARCH_HTTP_HOST", DEFAULT_HOST).strip()
    return candidate or DEFAULT_HOST


def resolve_bind_port(env: Mapping[str, str] | None = None) -> int:
    source = env or os.environ
    raw_value = source.get("PORT") or source.get("SCHOLAR_SEARCH_HTTP_PORT") or str(DEFAULT_PORT)
    candidate = raw_value.strip()
    if not candidate:
        return DEFAULT_PORT
    return int(candidate)


def main() -> None:
    run(
        "scholar_search_mcp.deployment:app",
        host=resolve_bind_host(),
        port=resolve_bind_port(),
    )


if __name__ == "__main__":
    main()
