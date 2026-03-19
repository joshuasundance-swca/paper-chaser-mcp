"""Environment-backed settings helpers."""

import os
from collections.abc import Mapping
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from .models.tools import (
    DEFAULT_SEARCH_PROVIDER_ORDER,
    SearchProvider,
    _normalize_provider_name,
)


def _env_bool(key: str, default: bool = True) -> bool:
    """Parse env as bool: 1/true/yes => True; 0/false/no => False."""
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes")


def _parse_env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes")


def _parse_optional_string(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _parse_provider_order(
    env: Mapping[str, str],
    key: str,
) -> tuple[SearchProvider, ...]:
    value = env.get(key)
    if value is None or value == "":
        return DEFAULT_SEARCH_PROVIDER_ORDER

    providers = [segment.strip() for segment in value.split(",") if segment.strip()]
    if not providers:
        raise ValueError(f"{key} must list at least one provider when it is set.")

    normalized_providers = [
        _normalize_provider_name(provider) for provider in providers
    ]

    duplicates = [
        provider
        for index, provider in enumerate(normalized_providers)
        if provider in normalized_providers[:index]
    ]
    if duplicates:
        duplicate_text = ", ".join(duplicates)
        raise ValueError(f"{key} cannot repeat providers: {duplicate_text}")

    return tuple(cast(SearchProvider, provider) for provider in normalized_providers)


def _parse_csv_strings(env: Mapping[str, str], key: str) -> tuple[str, ...]:
    value = env.get(key)
    if value is None or value == "":
        return ()
    return tuple(segment.strip() for segment in value.split(",") if segment.strip())


class AppSettings(BaseModel):
    """Typed application settings loaded from environment variables."""

    model_config = ConfigDict(frozen=True)

    semantic_scholar_api_key: str | None = None
    core_api_key: str | None = None
    openalex_api_key: str | None = None
    openalex_mailto: str | None = None
    serpapi_api_key: str | None = None
    enable_core: bool = True
    enable_semantic_scholar: bool = True
    enable_openalex: bool = True
    enable_arxiv: bool = True
    enable_serpapi: bool = False
    provider_order: tuple[SearchProvider, ...] = DEFAULT_SEARCH_PROVIDER_ORDER
    transport: Literal["stdio", "http", "streamable-http", "sse"] = "stdio"
    http_host: str = "127.0.0.1"
    http_port: int = 8000
    http_path: str = "/mcp"
    http_auth_token: str | None = None
    http_auth_header: str = "authorization"
    allowed_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "AppSettings":
        env = environ if environ is not None else os.environ
        return cls(
            semantic_scholar_api_key=_parse_optional_string(
                env,
                "SEMANTIC_SCHOLAR_API_KEY",
            ),
            core_api_key=_parse_optional_string(env, "CORE_API_KEY"),
            openalex_api_key=_parse_optional_string(env, "OPENALEX_API_KEY"),
            openalex_mailto=_parse_optional_string(env, "OPENALEX_MAILTO"),
            serpapi_api_key=_parse_optional_string(env, "SERPAPI_API_KEY"),
            enable_core=_parse_env_bool(env, "SCHOLAR_SEARCH_ENABLE_CORE", True),
            enable_semantic_scholar=_parse_env_bool(
                env,
                "SCHOLAR_SEARCH_ENABLE_SEMANTIC_SCHOLAR",
                True,
            ),
            enable_openalex=_parse_env_bool(
                env,
                "SCHOLAR_SEARCH_ENABLE_OPENALEX",
                True,
            ),
            enable_arxiv=_parse_env_bool(env, "SCHOLAR_SEARCH_ENABLE_ARXIV", True),
            enable_serpapi=_parse_env_bool(
                env,
                "SCHOLAR_SEARCH_ENABLE_SERPAPI",
                False,
            ),
            provider_order=_parse_provider_order(env, "SCHOLAR_SEARCH_PROVIDER_ORDER"),
            transport=cast_transport(env.get("SCHOLAR_SEARCH_TRANSPORT")),
            http_host=env.get("SCHOLAR_SEARCH_HTTP_HOST", "127.0.0.1"),
            http_port=int(env.get("SCHOLAR_SEARCH_HTTP_PORT", "8000")),
            http_path=env.get("SCHOLAR_SEARCH_HTTP_PATH", "/mcp"),
            http_auth_token=_parse_optional_string(
                env, "SCHOLAR_SEARCH_HTTP_AUTH_TOKEN"
            ),
            http_auth_header=env.get(
                "SCHOLAR_SEARCH_HTTP_AUTH_HEADER",
                "authorization",
            )
            .strip()
            .lower(),
            allowed_origins=_parse_csv_strings(env, "SCHOLAR_SEARCH_ALLOWED_ORIGINS"),
        )


def cast_transport(
    value: str | None,
) -> Literal["stdio", "http", "streamable-http", "sse"]:
    """Normalize the configured FastMCP transport."""
    if value is None or value == "":
        return "stdio"
    normalized = value.strip().lower()
    if normalized in {"stdio", "http", "streamable-http", "sse"}:
        return cast(Literal["stdio", "http", "streamable-http", "sse"], normalized)
    raise ValueError(
        "SCHOLAR_SEARCH_TRANSPORT must be one of: stdio, http, streamable-http, sse"
    )
