from starlette.applications import Starlette
from starlette.requests import Request
from starlette.testclient import TestClient

from paper_chaser_mcp.deployment import (
    DeploymentSecurityMiddleware,
    create_deployment_app,
)
from paper_chaser_mcp.deployment_runner import resolve_bind_host, resolve_bind_port
from paper_chaser_mcp.deployment_utils import resolve_smoke_test_health_url
from paper_chaser_mcp.settings import AppSettings


def _request(path: str, headers: dict[str, str] | None = None) -> Request:
    raw_headers = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in (headers or {}).items()]
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": raw_headers,
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 443),
            "root_path": "",
        }
    )


def _middleware(env: dict[str, str] | None = None) -> DeploymentSecurityMiddleware:
    settings = AppSettings.from_env(env or {})
    return DeploymentSecurityMiddleware(Starlette(), settings=settings)


def test_root_mcp_path_protects_all_paths() -> None:
    middleware = _middleware({"PAPER_CHASER_HTTP_PATH": "/"})

    assert middleware._is_protected_path("/")
    assert middleware._is_protected_path("/healthz")


def test_missing_auth_token_short_circuits_authorization() -> None:
    middleware = _middleware({"PAPER_CHASER_HTTP_PATH": "/mcp"})

    assert middleware._authorized(_request("/mcp")) is True


def test_invalid_bearer_authorization_header_is_rejected() -> None:
    middleware = _middleware(
        {
            "PAPER_CHASER_HTTP_AUTH_TOKEN": "super-secret",
            "PAPER_CHASER_HTTP_AUTH_HEADER": "authorization",
            "PAPER_CHASER_HTTP_PATH": "/mcp",
        }
    )

    assert middleware._authorized(_request("/mcp", {"Authorization": "Basic super-secret"})) is False


def test_custom_auth_header_compares_raw_value() -> None:
    middleware = _middleware(
        {
            "PAPER_CHASER_HTTP_AUTH_TOKEN": "super-secret",
            "PAPER_CHASER_HTTP_AUTH_HEADER": "x-backend-auth",
            "PAPER_CHASER_HTTP_PATH": "/mcp",
        }
    )

    assert middleware._authorized(_request("/mcp", {"X-Backend-Auth": "super-secret"})) is True


def test_resolve_smoke_test_health_url_prefers_explicit_url() -> None:
    url = resolve_smoke_test_health_url(
        "https://private.example.internal/healthz",
        {
            "properties": {
                "outputs": {
                    "containerAppHealthUrl": {"value": "https://ignored/healthz"},
                }
            }
        },
    )

    assert url == "https://private.example.internal/healthz"


def test_resolve_smoke_test_health_url_uses_health_output() -> None:
    url = resolve_smoke_test_health_url(
        None,
        {"properties": {"outputs": {"containerAppHealthUrl": {"value": "https://aca-dev.internal/healthz"}}}},
    )

    assert url == "https://aca-dev.internal/healthz"


def test_resolve_smoke_test_health_url_falls_back_to_fqdn() -> None:
    url = resolve_smoke_test_health_url(
        None,
        {
            "properties": {
                "outputs": {
                    "containerAppFqdn": {"value": "aca-dev.internal"},
                }
            }
        },
    )

    assert url == "https://aca-dev.internal/healthz"


def test_resolve_smoke_test_health_url_requires_http_scheme() -> None:
    try:
        resolve_smoke_test_health_url("ftp://private.example.internal/healthz", {})
    except ValueError as error:
        assert "http or https" in str(error)
    else:
        raise AssertionError("Expected invalid scheme to raise ValueError.")


def test_resolve_smoke_test_health_url_requires_outputs_when_explicit_missing() -> None:
    try:
        resolve_smoke_test_health_url(None, {"properties": {"outputs": {}}})
    except ValueError as error:
        assert "Unable to resolve smoke test health URL" in str(error)
    else:
        raise AssertionError("Expected missing outputs to raise ValueError.")


def test_resolve_smoke_test_health_url_rejects_missing_host() -> None:
    try:
        resolve_smoke_test_health_url("https:///healthz", {})
    except ValueError as error:
        assert "include a host" in str(error)
    else:
        raise AssertionError("Expected missing host to raise ValueError.")


def test_resolve_smoke_test_health_url_ignores_non_mapping_outputs_and_blank_values() -> None:
    try:
        resolve_smoke_test_health_url(
            None,
            {
                "properties": {
                    "outputs": {
                        "containerAppHealthUrl": {"value": "   "},
                        "containerAppFqdn": "aca-dev.internal",
                    }
                }
            },
        )
    except ValueError as error:
        assert "Unable to resolve smoke test health URL" in str(error)
    else:
        raise AssertionError("Expected invalid outputs to raise ValueError.")


def test_resolve_smoke_test_health_url_handles_non_mapping_properties() -> None:
    try:
        resolve_smoke_test_health_url(None, {"properties": "not-a-mapping"})
    except ValueError as error:
        assert "Unable to resolve smoke test health URL" in str(error)
    else:
        raise AssertionError("Expected non-mapping properties to raise ValueError.")


def test_resolve_smoke_test_health_url_falls_back_from_non_string_health_output_to_fqdn() -> None:
    url = resolve_smoke_test_health_url(
        None,
        {
            "properties": {
                "outputs": {
                    "containerAppHealthUrl": {"value": 123},
                    "containerAppFqdn": {"value": "aca-dev.internal"},
                }
            }
        },
    )

    assert url == "https://aca-dev.internal/healthz"


def test_resolve_bind_host_uses_http_host_env() -> None:
    assert resolve_bind_host({"PAPER_CHASER_HTTP_HOST": "0.0.0.0"}) == "0.0.0.0"


def test_resolve_bind_port_prefers_platform_port() -> None:
    assert resolve_bind_port({"PORT": "9090", "PAPER_CHASER_HTTP_PORT": "8080"}) == 9090


def test_resolve_bind_port_falls_back_to_http_port() -> None:
    assert resolve_bind_port({"PAPER_CHASER_HTTP_PORT": "8181"}) == 8181


def test_deployment_app_supports_exact_mcp_path_without_redirect() -> None:
    headers = {"accept": "application/json, text/event-stream"}
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }

    with TestClient(create_deployment_app()) as client:
        init_response = client.post("/mcp", json=initialize, headers=headers, follow_redirects=False)
        session_id = init_response.headers["mcp-session-id"]
        initialized_response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={**headers, "mcp-session-id": session_id},
            follow_redirects=False,
        )
        tools_response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers={**headers, "mcp-session-id": session_id},
            follow_redirects=False,
        )

    assert init_response.status_code == 200
    assert init_response.headers.get("location") is None
    assert initialized_response.status_code == 202
    assert initialized_response.headers.get("location") is None
    assert tools_response.status_code == 200
    assert tools_response.headers.get("location") is None
