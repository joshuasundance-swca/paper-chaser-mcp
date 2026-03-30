from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_chaser_mcp.clients.scholarapi.burnin import (
    collect_burnin_report,
    write_burnin_report,
)
from paper_chaser_mcp.clients.scholarapi.errors import ScholarApiError


class _FakeScholarApiClient:
    async def search(self, **kwargs: object) -> dict:
        return {
            "provider": "scholarapi",
            "total": 1,
            "requestId": "req-search-1",
            "requestCost": "3",
            "data": [{"paperId": "ScholarAPI:sa-1"}],
            "pagination": {"hasMore": True, "nextCursor": "search-next"},
        }

    async def list_papers(self, **kwargs: object) -> dict:
        return {
            "provider": "scholarapi",
            "total": 1,
            "requestId": "req-list-1",
            "requestCost": "2",
            "data": [{"paperId": "ScholarAPI:sa-list-1"}],
            "pagination": {"hasMore": True, "nextCursor": "2024-03-01T12:30:45.123Z"},
        }

    async def get_text(self, paper_id: str) -> dict:
        if paper_id == "paper-chaser-burnin-missing-id":
            raise ScholarApiError("ScholarAPI resource not found or content unavailable. (request id req-missing-1)")
        return {
            "provider": "scholarapi",
            "paperId": f"ScholarAPI:{paper_id}",
            "text": "example full text",
        }

    async def get_pdf(self, paper_id: str) -> dict:
        return {
            "provider": "scholarapi",
            "paperId": f"ScholarAPI:{paper_id}",
            "mimeType": "application/pdf",
            "byteLength": 8,
        }


@pytest.mark.asyncio
async def test_collect_burnin_report_summarizes_search_list_and_missing_probe() -> None:
    report = await collect_burnin_report(
        client=_FakeScholarApiClient(),
        search_query="graph neural networks",
        include_pdf_probe=True,
    )

    assert report["search"]["status"] == "success"
    assert report["search"]["hasRequestId"] is True
    assert report["search"]["hasRequestCost"] is True
    assert report["search"]["hasNextCursor"] is True
    assert report["list"]["nextCursor"] == "2024-03-01T12:30:45.123Z"
    assert report["probes"]["getText"]["status"] == "success"
    assert report["probes"]["getPdf"]["mimeType"] == "application/pdf"
    assert report["probes"]["missingText"]["status"] == "error"
    assert report["probes"]["missingText"]["errorType"] == "ScholarApiError"


def test_write_burnin_report_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "burnin.json"
    report = {"search": {"status": "success"}}

    written = write_burnin_report(output, report)

    assert written == output
    assert json.loads(output.read_text(encoding="utf-8")) == report
