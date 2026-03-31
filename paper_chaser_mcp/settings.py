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

AgenticProvider = Literal[
    "openai",
    "azure-openai",
    "anthropic",
    "nvidia",
    "google",
    "mistral",
    "huggingface",
    "deterministic",
]
AgenticIndexBackend = Literal["memory", "faiss"]


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

    normalized_providers = [_normalize_provider_name(provider) for provider in providers]

    duplicates = [
        provider for index, provider in enumerate(normalized_providers) if provider in normalized_providers[:index]
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


def _parse_positive_int(
    env: Mapping[str, str],
    key: str,
    default: int,
) -> int:
    value = env.get(key)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return parsed


def _parse_positive_float(
    env: Mapping[str, str],
    key: str,
    default: float,
) -> float:
    value = env.get(key)
    if value is None or value == "":
        return default
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{key} must be a positive number.")
    return parsed


class AppSettings(BaseModel):
    """Typed application settings loaded from environment variables."""

    model_config = ConfigDict(frozen=True)

    openai_api_key: str | None = None
    nvidia_api_key: str | None = None
    nvidia_nim_base_url: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_planner_deployment: str | None = None
    azure_openai_synthesis_deployment: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    mistral_api_key: str | None = None
    huggingface_api_key: str | None = None
    huggingface_base_url: str = "https://router.huggingface.co/v1"
    semantic_scholar_api_key: str | None = None
    core_api_key: str | None = None
    openalex_api_key: str | None = None
    openalex_mailto: str | None = None
    serpapi_api_key: str | None = None
    scholarapi_api_key: str | None = None
    govinfo_api_key: str | None = None
    crossref_mailto: str | None = None
    unpaywall_email: str | None = None
    ecos_base_url: str = "https://ecos.fws.gov"
    enable_core: bool = False
    enable_semantic_scholar: bool = True
    enable_openalex: bool = True
    enable_arxiv: bool = True
    enable_serpapi: bool = False
    enable_scholarapi: bool = False
    enable_crossref: bool = True
    enable_unpaywall: bool = True
    enable_ecos: bool = True
    enable_federal_register: bool = True
    enable_govinfo_cfr: bool = True
    hide_disabled_tools: bool = False
    provider_order: tuple[SearchProvider, ...] = DEFAULT_SEARCH_PROVIDER_ORDER
    transport: Literal["stdio", "http", "streamable-http", "sse"] = "stdio"
    http_host: str = "127.0.0.1"
    http_port: int = 8000
    http_path: str = "/mcp"
    http_auth_token: str | None = None
    http_auth_header: str = "authorization"
    allowed_origins: tuple[str, ...] = ()
    enable_agentic: bool = False
    agentic_provider: AgenticProvider = "openai"
    planner_model: str = "gpt-5.4-mini"
    synthesis_model: str = "gpt-5.4"
    embedding_model: str = "text-embedding-3-large"
    disable_embeddings: bool = True
    agentic_openai_timeout_seconds: float = 30.0
    agentic_index_backend: AgenticIndexBackend = "memory"
    session_ttl_seconds: int = 1800
    enable_agentic_trace_log: bool = False
    crossref_timeout_seconds: float = 30.0
    unpaywall_timeout_seconds: float = 30.0
    ecos_timeout_seconds: float = 30.0
    federal_register_timeout_seconds: float = 30.0
    govinfo_timeout_seconds: float = 30.0
    govinfo_document_timeout_seconds: float = 60.0
    govinfo_max_document_size_mb: int = 25
    ecos_document_timeout_seconds: float = 60.0
    ecos_document_conversion_timeout_seconds: float = 60.0
    ecos_max_document_size_mb: int = 25
    ecos_verify_tls: bool = True
    ecos_ca_bundle: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "AppSettings":
        env = environ if environ is not None else os.environ
        return cls(
            openai_api_key=_parse_optional_string(env, "OPENAI_API_KEY"),
            nvidia_api_key=_parse_optional_string(env, "NVIDIA_API_KEY"),
            nvidia_nim_base_url=_parse_optional_string(env, "NVIDIA_NIM_BASE_URL"),
            azure_openai_api_key=_parse_optional_string(env, "AZURE_OPENAI_API_KEY"),
            azure_openai_endpoint=_parse_optional_string(env, "AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_version=_parse_optional_string(env, "AZURE_OPENAI_API_VERSION"),
            azure_openai_planner_deployment=_parse_optional_string(env, "AZURE_OPENAI_PLANNER_DEPLOYMENT"),
            azure_openai_synthesis_deployment=_parse_optional_string(env, "AZURE_OPENAI_SYNTHESIS_DEPLOYMENT"),
            anthropic_api_key=_parse_optional_string(env, "ANTHROPIC_API_KEY"),
            google_api_key=_parse_optional_string(env, "GOOGLE_API_KEY"),
            mistral_api_key=_parse_optional_string(env, "MISTRAL_API_KEY"),
            huggingface_api_key=_parse_optional_string(env, "HUGGINGFACE_API_KEY"),
            huggingface_base_url=(
                env.get("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1").strip()
                or "https://router.huggingface.co/v1"
            ),
            semantic_scholar_api_key=_parse_optional_string(
                env,
                "SEMANTIC_SCHOLAR_API_KEY",
            ),
            core_api_key=_parse_optional_string(env, "CORE_API_KEY"),
            openalex_api_key=_parse_optional_string(env, "OPENALEX_API_KEY"),
            openalex_mailto=_parse_optional_string(env, "OPENALEX_MAILTO"),
            serpapi_api_key=_parse_optional_string(env, "SERPAPI_API_KEY"),
            scholarapi_api_key=_parse_optional_string(env, "SCHOLARAPI_API_KEY"),
            govinfo_api_key=_parse_optional_string(env, "GOVINFO_API_KEY"),
            crossref_mailto=_parse_optional_string(env, "CROSSREF_MAILTO"),
            unpaywall_email=_parse_optional_string(env, "UNPAYWALL_EMAIL"),
            ecos_base_url=env.get("ECOS_BASE_URL", "https://ecos.fws.gov").strip() or "https://ecos.fws.gov",
            enable_core=_parse_env_bool(env, "PAPER_CHASER_ENABLE_CORE", False),
            enable_semantic_scholar=_parse_env_bool(
                env,
                "PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR",
                True,
            ),
            enable_openalex=_parse_env_bool(
                env,
                "PAPER_CHASER_ENABLE_OPENALEX",
                True,
            ),
            enable_arxiv=_parse_env_bool(env, "PAPER_CHASER_ENABLE_ARXIV", True),
            enable_serpapi=_parse_env_bool(
                env,
                "PAPER_CHASER_ENABLE_SERPAPI",
                False,
            ),
            enable_scholarapi=_parse_env_bool(
                env,
                "PAPER_CHASER_ENABLE_SCHOLARAPI",
                False,
            ),
            enable_crossref=_parse_env_bool(
                env,
                "PAPER_CHASER_ENABLE_CROSSREF",
                True,
            ),
            enable_unpaywall=_parse_env_bool(
                env,
                "PAPER_CHASER_ENABLE_UNPAYWALL",
                True,
            ),
            enable_ecos=_parse_env_bool(
                env,
                "PAPER_CHASER_ENABLE_ECOS",
                True,
            ),
            enable_federal_register=_parse_env_bool(
                env,
                "PAPER_CHASER_ENABLE_FEDERAL_REGISTER",
                True,
            ),
            enable_govinfo_cfr=_parse_env_bool(
                env,
                "PAPER_CHASER_ENABLE_GOVINFO_CFR",
                True,
            ),
            hide_disabled_tools=_parse_env_bool(
                env,
                "PAPER_CHASER_HIDE_DISABLED_TOOLS",
                False,
            ),
            provider_order=_parse_provider_order(env, "PAPER_CHASER_PROVIDER_ORDER"),
            transport=cast_transport(env.get("PAPER_CHASER_TRANSPORT")),
            http_host=env.get("PAPER_CHASER_HTTP_HOST", "127.0.0.1"),
            http_port=int(env.get("PAPER_CHASER_HTTP_PORT", "8000")),
            http_path=env.get("PAPER_CHASER_HTTP_PATH", "/mcp"),
            http_auth_token=_parse_optional_string(env, "PAPER_CHASER_HTTP_AUTH_TOKEN"),
            http_auth_header=env.get(
                "PAPER_CHASER_HTTP_AUTH_HEADER",
                "authorization",
            )
            .strip()
            .lower(),
            allowed_origins=_parse_csv_strings(env, "PAPER_CHASER_ALLOWED_ORIGINS"),
            enable_agentic=_parse_env_bool(
                env,
                "PAPER_CHASER_ENABLE_AGENTIC",
                False,
            ),
            agentic_provider=cast_agentic_provider(env.get("PAPER_CHASER_AGENTIC_PROVIDER")),
            planner_model=env.get("PAPER_CHASER_PLANNER_MODEL", "gpt-5.4-mini"),
            synthesis_model=env.get("PAPER_CHASER_SYNTHESIS_MODEL", "gpt-5.4"),
            embedding_model=env.get(
                "PAPER_CHASER_EMBEDDING_MODEL",
                "text-embedding-3-large",
            ),
            disable_embeddings=_parse_env_bool(
                env,
                "PAPER_CHASER_DISABLE_EMBEDDINGS",
                True,
            ),
            agentic_openai_timeout_seconds=_parse_positive_float(
                env,
                "PAPER_CHASER_AGENTIC_OPENAI_TIMEOUT_SECONDS",
                30.0,
            ),
            agentic_index_backend=cast_agentic_index_backend(env.get("PAPER_CHASER_AGENTIC_INDEX_BACKEND")),
            session_ttl_seconds=_parse_positive_int(
                env,
                "PAPER_CHASER_SESSION_TTL_SECONDS",
                1800,
            ),
            enable_agentic_trace_log=_parse_env_bool(
                env,
                "PAPER_CHASER_ENABLE_AGENTIC_TRACE_LOG",
                False,
            ),
            crossref_timeout_seconds=_parse_positive_float(
                env,
                "CROSSREF_TIMEOUT_SECONDS",
                30.0,
            ),
            unpaywall_timeout_seconds=_parse_positive_float(
                env,
                "UNPAYWALL_TIMEOUT_SECONDS",
                30.0,
            ),
            federal_register_timeout_seconds=_parse_positive_float(
                env,
                "FEDERAL_REGISTER_TIMEOUT_SECONDS",
                30.0,
            ),
            govinfo_timeout_seconds=_parse_positive_float(
                env,
                "GOVINFO_TIMEOUT_SECONDS",
                30.0,
            ),
            govinfo_document_timeout_seconds=_parse_positive_float(
                env,
                "GOVINFO_DOCUMENT_TIMEOUT_SECONDS",
                60.0,
            ),
            govinfo_max_document_size_mb=_parse_positive_int(
                env,
                "GOVINFO_MAX_DOCUMENT_SIZE_MB",
                25,
            ),
            ecos_timeout_seconds=_parse_positive_float(
                env,
                "ECOS_TIMEOUT_SECONDS",
                30.0,
            ),
            ecos_document_timeout_seconds=_parse_positive_float(
                env,
                "ECOS_DOCUMENT_TIMEOUT_SECONDS",
                60.0,
            ),
            ecos_document_conversion_timeout_seconds=_parse_positive_float(
                env,
                "ECOS_DOCUMENT_CONVERSION_TIMEOUT_SECONDS",
                60.0,
            ),
            ecos_max_document_size_mb=_parse_positive_int(
                env,
                "ECOS_MAX_DOCUMENT_SIZE_MB",
                25,
            ),
            ecos_verify_tls=_parse_env_bool(
                env,
                "ECOS_VERIFY_TLS",
                True,
            ),
            ecos_ca_bundle=_parse_optional_string(env, "ECOS_CA_BUNDLE"),
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
    raise ValueError("PAPER_CHASER_TRANSPORT must be one of: stdio, http, streamable-http, sse")


def cast_agentic_provider(value: str | None) -> AgenticProvider:
    """Normalize the configured smart-workflow model provider."""
    if value is None or value == "":
        return "openai"
    normalized = value.strip().lower()
    if normalized in {
        "openai",
        "azure-openai",
        "anthropic",
        "nvidia",
        "google",
        "mistral",
        "huggingface",
        "deterministic",
    }:
        return cast(AgenticProvider, normalized)
    raise ValueError(
        "PAPER_CHASER_AGENTIC_PROVIDER must be one of: openai, azure-openai, "
        "anthropic, nvidia, google, mistral, huggingface, deterministic"
    )


def cast_agentic_index_backend(value: str | None) -> AgenticIndexBackend:
    """Normalize the configured smart-workflow index backend."""
    if value is None or value == "":
        return "memory"
    normalized = value.strip().lower()
    if normalized in {"memory", "faiss"}:
        return cast(AgenticIndexBackend, normalized)
    raise ValueError("PAPER_CHASER_AGENTIC_INDEX_BACKEND must be one of: memory, faiss")
