"""Command-line entrypoints for local and containerized MCP usage."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version


def _package_version() -> str:
    try:
        return version("paper-chaser-mcp")
    except PackageNotFoundError:
        return "unknown"


def _run_server() -> None:
    from .server import main as run_main

    run_main()


def _run_deployment_http() -> None:
    from .deployment_runner import main as run_main

    run_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-chaser-mcp",
        description=(
            "Run the Paper Chaser MCP server. The default command starts the "
            "package-local MCP server using PAPER_CHASER_* environment "
            "settings, which means stdio transport unless you explicitly opt "
            "into an HTTP transport."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help=("Run the package-local MCP server using PAPER_CHASER_* settings (defaults to stdio)."),
    )
    run_parser.set_defaults(handler=_run_server)

    deployment_parser = subparsers.add_parser(
        "deployment-http",
        help=("Run the HTTP deployment wrapper with /healthz and optional Origin/auth enforcement."),
    )
    deployment_parser.set_defaults(handler=_run_deployment_http)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = getattr(args, "handler", None)
    if handler is None:
        _run_server()
        return
    handler()
