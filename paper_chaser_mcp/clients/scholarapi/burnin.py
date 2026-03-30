"""Opt-in live contract burn-in helpers for ScholarAPI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol


class ScholarApiBurninClient(Protocol):
    async def search(self, **kwargs: Any) -> dict[str, Any]: ...

    async def list_papers(self, **kwargs: Any) -> dict[str, Any]: ...

    async def get_text(self, paper_id: str) -> dict[str, Any]: ...

    async def get_pdf(self, paper_id: str) -> dict[str, Any]: ...


DEFAULT_BURNIN_OUTPUT = Path("build/scholarapi-burnin.json")
DEFAULT_MISSING_PAPER_ID = "paper-chaser-burnin-missing-id"


def _raw_scholarapi_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    prefix = "ScholarAPI:"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :]
    return normalized


def _summarize_page(payload: dict[str, Any]) -> dict[str, Any]:
    raw_pagination = payload.get("pagination")
    pagination: dict[str, Any] = raw_pagination if isinstance(raw_pagination, dict) else {}
    raw_data = payload.get("data")
    data: list[Any] = raw_data if isinstance(raw_data, list) else []
    first_paper_id = None
    if data and isinstance(data[0], dict):
        first_paper_id = data[0].get("paperId")
    return {
        "status": "success",
        "provider": payload.get("provider"),
        "total": payload.get("total"),
        "requestId": payload.get("requestId"),
        "requestCost": payload.get("requestCost"),
        "hasRequestId": bool(payload.get("requestId")),
        "hasRequestCost": payload.get("requestCost") is not None,
        "hasNextCursor": bool(pagination.get("nextCursor")),
        "nextCursor": pagination.get("nextCursor"),
        "hasMore": bool(pagination.get("hasMore")),
        "firstPaperId": first_paper_id,
        "resultCount": len(data),
    }


def _summarize_probe_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "success", "valueType": type(payload).__name__}
    summary: dict[str, Any] = {
        "status": "success",
        "provider": payload.get("provider"),
        "payloadKeys": sorted(payload.keys()),
    }
    if isinstance(payload.get("text"), str):
        summary["textLength"] = len(payload.get("text") or "")
    if isinstance(payload.get("results"), list):
        summary["resultCount"] = len(payload.get("results") or [])
    if payload.get("mimeType"):
        summary["mimeType"] = payload.get("mimeType")
    if payload.get("byteLength") is not None:
        summary["byteLength"] = payload.get("byteLength")
    return summary


async def _capture_operation(operation: Callable[[], Awaitable[Any]]) -> dict[str, Any]:
    try:
        payload = await operation()
    except Exception as exc:
        return {
            "status": "error",
            "errorType": type(exc).__name__,
            "error": str(exc),
        }
    if isinstance(payload, dict) and "pagination" in payload:
        return _summarize_page(payload)
    return _summarize_probe_payload(payload)


async def collect_burnin_report(
    *,
    client: ScholarApiBurninClient,
    search_query: str,
    list_query: str | None = None,
    limit: int = 5,
    missing_paper_id: str | None = DEFAULT_MISSING_PAPER_ID,
    include_pdf_probe: bool = False,
) -> dict[str, Any]:
    """Collect a small live contract report for ScholarAPI search/list/text paths."""

    report: dict[str, Any] = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "searchQuery": search_query,
        "listQuery": list_query or search_query,
        "search": await _capture_operation(
            lambda: client.search(
                query=search_query,
                limit=limit,
            )
        ),
        "list": await _capture_operation(
            lambda: client.list_papers(
                query=list_query or search_query,
                limit=limit,
            )
        ),
        "probes": {},
    }

    candidate_paper_id = _raw_scholarapi_id(report["search"].get("firstPaperId"))
    if candidate_paper_id is None:
        candidate_paper_id = _raw_scholarapi_id(report["list"].get("firstPaperId"))

    if candidate_paper_id is not None:
        report["probes"]["getText"] = await _capture_operation(lambda: client.get_text(candidate_paper_id))
        if include_pdf_probe:
            report["probes"]["getPdf"] = await _capture_operation(lambda: client.get_pdf(candidate_paper_id))

    if missing_paper_id:
        report["probes"]["missingText"] = await _capture_operation(lambda: client.get_text(missing_paper_id))

    return report


def write_burnin_report(path: Path, report: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path
