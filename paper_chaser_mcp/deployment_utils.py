"""Helpers for deployment automation and smoke-test resolution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse


def _string_output(outputs: Mapping[str, Any], key: str) -> str | None:
    value = outputs.get(key)
    if not isinstance(value, Mapping):
        return None
    raw = value.get("value")
    if not isinstance(raw, str):
        return None
    normalized = raw.strip()
    return normalized or None


def _normalize_health_url(candidate: str) -> str:
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Smoke test health URL must use http or https.")
    if not parsed.netloc:
        raise ValueError("Smoke test health URL must include a host.")
    return candidate


def resolve_smoke_test_health_url(
    explicit_url: str | None,
    deployment_output: Mapping[str, Any],
) -> str:
    """Resolve the post-deploy health URL from an explicit value or Azure outputs."""

    if explicit_url:
        return _normalize_health_url(explicit_url.strip())

    properties = deployment_output.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}

    outputs = properties.get("outputs", {})
    if not isinstance(outputs, Mapping):
        outputs = {}

    health_url = _string_output(outputs, "containerAppHealthUrl")
    if health_url:
        return _normalize_health_url(health_url)

    container_app_fqdn = _string_output(outputs, "containerAppFqdn")
    if container_app_fqdn:
        return _normalize_health_url(f"https://{container_app_fqdn}/healthz")

    raise ValueError(
        "Unable to resolve smoke test health URL from deployment outputs. "
        "Provide SMOKE_TEST_HEALTH_URL or ensure the deployment outputs expose "
        "containerAppHealthUrl or containerAppFqdn."
    )
