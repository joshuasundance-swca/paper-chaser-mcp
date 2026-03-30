"""Run an opt-in live contract burn-in against ScholarAPI."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from paper_chaser_mcp.clients.scholarapi import ScholarApiClient
from paper_chaser_mcp.clients.scholarapi.burnin import (
    DEFAULT_BURNIN_OUTPUT,
    DEFAULT_MISSING_PAPER_ID,
    collect_burnin_report,
    write_burnin_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an opt-in live ScholarAPI contract burn-in.")
    parser.add_argument("--query", default="graph neural networks", help="Search query to use for the search probe.")
    parser.add_argument("--list-query", default=None, help="Optional query to use for the list probe.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum results to request from search/list.")
    parser.add_argument(
        "--missing-paper-id",
        default=DEFAULT_MISSING_PAPER_ID,
        help="Paper id to use for the missing-content 404 probe. Use an empty string to disable it.",
    )
    parser.add_argument(
        "--include-pdf-probe",
        action="store_true",
        help="Also fetch a PDF for the first returned paper if one is available.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_BURNIN_OUTPUT),
        help="Where to write the JSON burn-in report.",
    )
    return parser


async def _run(args: argparse.Namespace) -> Path:
    api_key = os.environ.get("SCHOLARAPI_API_KEY")
    if not api_key:
        raise SystemExit("SCHOLARAPI_API_KEY is required to run the ScholarAPI burn-in.")

    client = ScholarApiClient(api_key=api_key)
    try:
        report = await collect_burnin_report(
            client=client,
            search_query=args.query,
            list_query=args.list_query,
            limit=args.limit,
            missing_paper_id=(args.missing_paper_id or None),
            include_pdf_probe=bool(args.include_pdf_probe),
        )
    finally:
        await client.aclose()

    output_path = write_burnin_report(Path(args.output), report)
    print(f"ScholarAPI burn-in report written to {output_path}")
    return output_path


def main() -> None:
    args = _parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
