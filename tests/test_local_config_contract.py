from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"
DOCKER_COMPOSE = REPO_ROOT / "docker-compose.yaml"

EXPECTED_LOCAL_CONFIG_KEYS = {
    "CORE_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "OPENALEX_API_KEY",
    "OPENALEX_MAILTO",
    "SERPAPI_API_KEY",
    "SCHOLAR_SEARCH_ENABLE_CORE",
    "SCHOLAR_SEARCH_ENABLE_SEMANTIC_SCHOLAR",
    "SCHOLAR_SEARCH_ENABLE_OPENALEX",
    "SCHOLAR_SEARCH_ENABLE_ARXIV",
    "SCHOLAR_SEARCH_ENABLE_SERPAPI",
    "SCHOLAR_SEARCH_PROVIDER_ORDER",
    "SCHOLAR_SEARCH_TRANSPORT",
    "SCHOLAR_SEARCH_HTTP_PATH",
    "SCHOLAR_SEARCH_HTTP_AUTH_TOKEN",
    "SCHOLAR_SEARCH_HTTP_AUTH_HEADER",
    "SCHOLAR_SEARCH_ALLOWED_ORIGINS",
    "SCHOLAR_SEARCH_PUBLISHED_HOST",
    "SCHOLAR_SEARCH_PUBLISHED_PORT",
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


def test_env_example_lists_supported_public_local_config_keys() -> None:
    keys = _parse_env_keys(ENV_EXAMPLE)

    assert keys == EXPECTED_LOCAL_CONFIG_KEYS
    assert not (keys & BLOCKED_PUBLIC_CONFIG_KEYS)


def test_compose_uses_only_documented_local_config_keys() -> None:
    compose_keys = _parse_compose_substitution_keys(DOCKER_COMPOSE)

    assert compose_keys == EXPECTED_LOCAL_CONFIG_KEYS


def test_compose_does_not_expose_container_bind_host_or_internal_port() -> None:
    compose_keys = _parse_compose_substitution_keys(DOCKER_COMPOSE)

    assert "SCHOLAR_SEARCH_HTTP_HOST" not in compose_keys
    assert "SCHOLAR_SEARCH_HTTP_PORT" not in compose_keys
