"""ASGI deployment wrapper with health checks and optional HTTP hardening."""

from __future__ import annotations

import secrets
from typing import Any

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from .server import build_http_app
from .settings import AppSettings


class DeploymentSecurityMiddleware(BaseHTTPMiddleware):
    """Enforce optional origin allowlists and shared-token auth for MCP traffic."""

    def __init__(self, app: Any, *, settings: AppSettings) -> None:
        super().__init__(app)
        self._mcp_path = settings.http_path.rstrip("/") or "/"
        self._allowed_origins = frozenset(settings.allowed_origins)
        self._auth_token = settings.http_auth_token
        self._auth_header = settings.http_auth_header

    def _is_protected_path(self, path: str) -> bool:
        if self._mcp_path == "/":
            return path == "/" or path.startswith("/")
        return path == self._mcp_path or path.startswith(f"{self._mcp_path}/")

    def _authorized(self, request: Request) -> bool:
        if not self._auth_token:
            return True

        raw_value = request.headers.get(self._auth_header)
        if raw_value is None:
            return False

        if self._auth_header == "authorization":
            scheme, _, token = raw_value.partition(" ")
            if scheme.lower() != "bearer" or not token:
                return False
            return secrets.compare_digest(token.strip(), self._auth_token)

        return secrets.compare_digest(raw_value, self._auth_token)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if self._is_protected_path(request.url.path):
            origin = request.headers.get("origin")
            if origin and self._allowed_origins and origin not in self._allowed_origins:
                return JSONResponse(
                    {
                        "error": "origin_not_allowed",
                        "message": "Origin is not allowed for this MCP deployment.",
                    },
                    status_code=403,
                )

            if not self._authorized(request):
                return JSONResponse(
                    {
                        "error": "unauthorized",
                        "message": (
                            "Valid deployment auth credentials are required for the "
                            "MCP endpoint."
                        ),
                    },
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )

        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response


async def healthz(_: Request) -> Response:
    return JSONResponse({"status": "ok"})


def create_deployment_app(settings: AppSettings | None = None) -> Starlette:
    deployment_settings = settings or AppSettings.from_env()
    mcp_app = build_http_app(
        path=deployment_settings.http_path,
        transport=(
            deployment_settings.transport
            if deployment_settings.transport != "stdio"
            else "streamable-http"
        ),
    )

    app = Starlette(
        routes=[
            Route("/healthz", endpoint=healthz),
            Mount("/", app=mcp_app),
        ],
        lifespan=getattr(mcp_app, "lifespan", None),
    )
    app.add_middleware(DeploymentSecurityMiddleware, settings=deployment_settings)
    return app


app = create_deployment_app()
