import ssl

import pytest

from paper_chaser_mcp.transport import (
    build_httpx_verify_config,
    is_tls_verification_error,
    maybe_close_async_resource,
)


def test_build_httpx_verify_config_handles_boolean_bundle_and_system_store() -> None:
    assert build_httpx_verify_config(verify_tls=False) is False
    assert build_httpx_verify_config(ca_bundle="  C:/certs/custom.pem  ") == "C:/certs/custom.pem"

    verify_config = build_httpx_verify_config(prefer_system_store=True)

    assert isinstance(verify_config, ssl.SSLContext)
    assert build_httpx_verify_config() is True


def test_is_tls_verification_error_matches_common_ssl_messages() -> None:
    assert is_tls_verification_error(RuntimeError("certificate verify failed for upstream")) is True
    assert is_tls_verification_error(RuntimeError("CERTIFICATE_VERIFY_FAILED")) is True
    assert is_tls_verification_error(RuntimeError("socket closed unexpectedly")) is False


@pytest.mark.asyncio
async def test_maybe_close_async_resource_prefers_aclose_and_awaits() -> None:
    events: list[str] = []

    class _Resource:
        async def aclose(self) -> None:
            events.append("aclose")

        def close(self) -> None:
            events.append("close")

    await maybe_close_async_resource(_Resource())

    assert events == ["aclose"]


@pytest.mark.asyncio
async def test_maybe_close_async_resource_ignores_closed_loop_runtime_error() -> None:
    class _Resource:
        def close(self) -> None:
            raise RuntimeError("Event loop is closed")

    await maybe_close_async_resource(_Resource())


@pytest.mark.asyncio
async def test_maybe_close_async_resource_re_raises_other_runtime_errors() -> None:
    class _Resource:
        def close(self) -> None:
            raise RuntimeError("other close failure")

    with pytest.raises(RuntimeError, match="other close failure"):
        await maybe_close_async_resource(_Resource())
