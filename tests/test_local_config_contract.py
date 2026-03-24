from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"
DOCKER_COMPOSE = REPO_ROOT / "docker-compose.yaml"
INSPECTOR_COMPOSE = REPO_ROOT / "compose.inspector.yaml"

EXPECTED_LOCAL_CONFIG_KEYS = {
    "OPENAI_API_KEY",
    "CORE_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "OPENALEX_API_KEY",
    "OPENALEX_MAILTO",
    "SERPAPI_API_KEY",
    "CROSSREF_MAILTO",
    "UNPAYWALL_EMAIL",
    "ECOS_BASE_URL",
    "SCHOLAR_SEARCH_ENABLE_CORE",
    "SCHOLAR_SEARCH_ENABLE_SEMANTIC_SCHOLAR",
    "SCHOLAR_SEARCH_ENABLE_OPENALEX",
    "SCHOLAR_SEARCH_ENABLE_ARXIV",
    "SCHOLAR_SEARCH_ENABLE_SERPAPI",
    "SCHOLAR_SEARCH_ENABLE_CROSSREF",
    "SCHOLAR_SEARCH_ENABLE_UNPAYWALL",
    "SCHOLAR_SEARCH_ENABLE_ECOS",
    "SCHOLAR_SEARCH_ENABLE_AGENTIC",
    "CROSSREF_TIMEOUT_SECONDS",
    "UNPAYWALL_TIMEOUT_SECONDS",
    "ECOS_TIMEOUT_SECONDS",
    "ECOS_DOCUMENT_TIMEOUT_SECONDS",
    "ECOS_DOCUMENT_CONVERSION_TIMEOUT_SECONDS",
    "ECOS_MAX_DOCUMENT_SIZE_MB",
    "ECOS_VERIFY_TLS",
    "ECOS_CA_BUNDLE",
    "SCHOLAR_SEARCH_PROVIDER_ORDER",
    "SCHOLAR_SEARCH_AGENTIC_PROVIDER",
    "SCHOLAR_SEARCH_PLANNER_MODEL",
    "SCHOLAR_SEARCH_SYNTHESIS_MODEL",
    "SCHOLAR_SEARCH_EMBEDDING_MODEL",
    "SCHOLAR_SEARCH_DISABLE_EMBEDDINGS",
    "SCHOLAR_SEARCH_AGENTIC_OPENAI_TIMEOUT_SECONDS",
    "SCHOLAR_SEARCH_AGENTIC_INDEX_BACKEND",
    "SCHOLAR_SEARCH_SESSION_TTL_SECONDS",
    "SCHOLAR_SEARCH_ENABLE_AGENTIC_TRACE_LOG",
    "SCHOLAR_SEARCH_TRANSPORT",
    "SCHOLAR_SEARCH_HTTP_PATH",
    "SCHOLAR_SEARCH_HTTP_AUTH_TOKEN",
    "SCHOLAR_SEARCH_HTTP_AUTH_HEADER",
    "SCHOLAR_SEARCH_ALLOWED_ORIGINS",
    "SCHOLAR_SEARCH_PUBLISHED_HOST",
    "SCHOLAR_SEARCH_PUBLISHED_PORT",
}

OPTIONAL_COMPOSE_ONLY_KEYS = {
    "IMAGE",
}

BLOCKED_PUBLIC_CONFIG_KEYS = {
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "ACR_NAME",
    "IMAGE_REPOSITORY",
    "AZURE_PRIVATE_RUNNER_LABELS_JSON",
}


def _parse_env_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, _ = line.partition("=")
        assert sep == "=", f"{path.name} contains a non-assignment line: {raw_line!r}"
        keys.add(key)
    return keys


def _parse_compose_substitution_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"\$\{([A-Z0-9_]+)(?::-[^}]*)?\}", text))


def _iter_compose_short_syntax_port_mappings(text: str) -> list[str]:
    mappings: list[str] = []
    in_ports_block = False
    ports_indent = 0

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip())
        if in_ports_block and indent <= ports_indent and not stripped.startswith("-"):
            in_ports_block = False

        if in_ports_block:
            if not stripped.startswith("-"):
                continue
            mapping = stripped[1:].split("#", 1)[0].strip().strip("\"'")
            if mapping:
                mappings.append(mapping)
            continue

        if stripped == "ports:":
            in_ports_block = True
            ports_indent = indent

    return mappings


def test_env_example_lists_supported_public_local_config_keys() -> None:
    keys = _parse_env_keys(ENV_EXAMPLE)

    assert EXPECTED_LOCAL_CONFIG_KEYS <= keys
    assert not (keys & BLOCKED_PUBLIC_CONFIG_KEYS)


def test_compose_uses_only_documented_local_config_keys() -> None:
    compose_keys = _parse_compose_substitution_keys(DOCKER_COMPOSE)

    assert EXPECTED_LOCAL_CONFIG_KEYS <= compose_keys
    undocumented = compose_keys - EXPECTED_LOCAL_CONFIG_KEYS - OPTIONAL_COMPOSE_ONLY_KEYS
    assert undocumented == set()


def test_compose_does_not_expose_container_bind_host_or_internal_port() -> None:
    compose_keys = _parse_compose_substitution_keys(DOCKER_COMPOSE)

    assert "SCHOLAR_SEARCH_HTTP_HOST" not in compose_keys
    assert "SCHOLAR_SEARCH_HTTP_PORT" not in compose_keys


def test_compose_explicitly_opts_into_http_deployment_wrapper() -> None:
    text = DOCKER_COMPOSE.read_text(encoding="utf-8")

    assert "deployment-http" in text


def test_inspector_compose_keeps_ports_localhost_bound_when_present() -> None:
    if not INSPECTOR_COMPOSE.exists():
        pytest.skip("compose.inspector.yaml is not part of this checkout.")

    text = INSPECTOR_COMPOSE.read_text(encoding="utf-8")
    for mapping in _iter_compose_short_syntax_port_mappings(text):
        assert mapping.startswith("127.0.0.1:")
