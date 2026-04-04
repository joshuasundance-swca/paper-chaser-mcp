from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"
DOCKER_COMPOSE = REPO_ROOT / "docker-compose.yaml"
INSPECTOR_COMPOSE = REPO_ROOT / "compose.inspector.yaml"
README = REPO_ROOT / "README.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"

EXPECTED_LOCAL_CONFIG_KEYS = {
    "OPENAI_API_KEY",
    "HUGGINGFACE_API_KEY",
    "HUGGINGFACE_BASE_URL",
    "NVIDIA_API_KEY",
    "NVIDIA_NIM_BASE_URL",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_PLANNER_DEPLOYMENT",
    "AZURE_OPENAI_SYNTHESIS_DEPLOYMENT",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "CORE_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "OPENALEX_API_KEY",
    "OPENALEX_MAILTO",
    "SERPAPI_API_KEY",
    "SCHOLARAPI_API_KEY",
    "GOVINFO_API_KEY",
    "CROSSREF_MAILTO",
    "UNPAYWALL_EMAIL",
    "ECOS_BASE_URL",
    "PAPER_CHASER_ENABLE_CORE",
    "PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR",
    "PAPER_CHASER_ENABLE_OPENALEX",
    "PAPER_CHASER_ENABLE_ARXIV",
    "PAPER_CHASER_ENABLE_SERPAPI",
    "PAPER_CHASER_ENABLE_SCHOLARAPI",
    "PAPER_CHASER_ENABLE_CROSSREF",
    "PAPER_CHASER_ENABLE_UNPAYWALL",
    "PAPER_CHASER_ENABLE_ECOS",
    "PAPER_CHASER_ENABLE_FEDERAL_REGISTER",
    "PAPER_CHASER_ENABLE_GOVINFO_CFR",
    "PAPER_CHASER_ENABLE_AGENTIC",
    "PAPER_CHASER_HIDE_DISABLED_TOOLS",
    "CROSSREF_TIMEOUT_SECONDS",
    "UNPAYWALL_TIMEOUT_SECONDS",
    "ECOS_TIMEOUT_SECONDS",
    "FEDERAL_REGISTER_TIMEOUT_SECONDS",
    "GOVINFO_TIMEOUT_SECONDS",
    "GOVINFO_DOCUMENT_TIMEOUT_SECONDS",
    "GOVINFO_MAX_DOCUMENT_SIZE_MB",
    "ECOS_DOCUMENT_TIMEOUT_SECONDS",
    "ECOS_DOCUMENT_CONVERSION_TIMEOUT_SECONDS",
    "ECOS_MAX_DOCUMENT_SIZE_MB",
    "ECOS_VERIFY_TLS",
    "ECOS_CA_BUNDLE",
    "PAPER_CHASER_PROVIDER_ORDER",
    "PAPER_CHASER_AGENTIC_PROVIDER",
    "PAPER_CHASER_PLANNER_MODEL",
    "PAPER_CHASER_SYNTHESIS_MODEL",
    "PAPER_CHASER_EMBEDDING_MODEL",
    "PAPER_CHASER_DISABLE_EMBEDDINGS",
    "PAPER_CHASER_AGENTIC_OPENAI_TIMEOUT_SECONDS",
    "PAPER_CHASER_AGENTIC_INDEX_BACKEND",
    "PAPER_CHASER_SESSION_TTL_SECONDS",
    "PAPER_CHASER_ENABLE_AGENTIC_TRACE_LOG",
    "PAPER_CHASER_GUIDED_RESEARCH_LATENCY_PROFILE",
    "PAPER_CHASER_GUIDED_FOLLOW_UP_LATENCY_PROFILE",
    "PAPER_CHASER_GUIDED_ALLOW_PAID_PROVIDERS",
    "PAPER_CHASER_GUIDED_ESCALATION_ENABLED",
    "PAPER_CHASER_GUIDED_ESCALATION_MAX_PASSES",
    "PAPER_CHASER_GUIDED_ESCALATION_ALLOW_PAID_PROVIDERS",
    "PAPER_CHASER_TRANSPORT",
    "PAPER_CHASER_HTTP_PATH",
    "PAPER_CHASER_HTTP_AUTH_TOKEN",
    "PAPER_CHASER_HTTP_AUTH_HEADER",
    "PAPER_CHASER_ALLOWED_ORIGINS",
    "PAPER_CHASER_PUBLISHED_HOST",
    "PAPER_CHASER_PUBLISHED_PORT",
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


def test_inspector_compose_uses_only_documented_local_config_keys() -> None:
    if not INSPECTOR_COMPOSE.exists():
        pytest.skip("compose.inspector.yaml is not part of this checkout.")

    compose_keys = _parse_compose_substitution_keys(INSPECTOR_COMPOSE)

    undocumented = compose_keys - EXPECTED_LOCAL_CONFIG_KEYS - OPTIONAL_COMPOSE_ONLY_KEYS
    assert undocumented == set()


def test_compose_does_not_expose_container_bind_host_or_internal_port() -> None:
    compose_keys = _parse_compose_substitution_keys(DOCKER_COMPOSE)

    assert "PAPER_CHASER_HTTP_HOST" not in compose_keys
    assert "PAPER_CHASER_HTTP_PORT" not in compose_keys


def test_compose_explicitly_opts_into_http_deployment_wrapper() -> None:
    text = DOCKER_COMPOSE.read_text(encoding="utf-8")

    assert "deployment-http" in text


def test_inspector_compose_keeps_ports_localhost_bound_when_present() -> None:
    if not INSPECTOR_COMPOSE.exists():
        pytest.skip("compose.inspector.yaml is not part of this checkout.")

    text = INSPECTOR_COMPOSE.read_text(encoding="utf-8")
    for mapping in _iter_compose_short_syntax_port_mappings(text):
        assert mapping.startswith("127.0.0.1:")


def test_readme_agentic_install_guidance_matches_declared_optional_extras() -> None:
    readme = README.read_text(encoding="utf-8")
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    for expected in (
        "ai = [",
        "openai = [",
        "nvidia = [",
        "anthropic = [",
        "google = [",
        "mistral = [",
        "ai-faiss = [",
        "all = [",
    ):
        assert expected in pyproject

    for expected in (
        'pip install -e ".[ai]"',
        'pip install -e ".[ai,openai]"',
        'pip install -e ".[ai,nvidia]"',
        'pip install -e ".[ai,anthropic]"',
        'pip install -e ".[ai,google]"',
        'pip install -e ".[ai,mistral]"',
        'pip install -e ".[all]"',
        'pip install -e ".[all,ai-faiss]"',
    ):
        assert expected in readme

    assert "Azure OpenAI uses the same `openai` extra" in readme


def test_readme_eval_install_guidance_matches_declared_optional_extras() -> None:
    readme = README.read_text(encoding="utf-8")
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    for expected in (
        "eval-foundry = [",
        "eval-huggingface = [",
        "eval = [",
    ):
        assert expected in pyproject

    for expected in (
        'pip install -e ".[eval-foundry]"',
        'pip install -e ".[eval-huggingface]"',
        'pip install -e ".[eval]"',
    ):
        assert expected in readme

    assert "eval publishing helpers use separate extras" in readme
