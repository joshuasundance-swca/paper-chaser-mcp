"""Batch summaries and ledger writers for promoted eval captures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .capture import _first_dict


def build_batch_summary(
    report: dict[str, Any],
    events: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
    *,
    batch_id: str | None = None,
    run_id: str | None = None,
    scenario_file: str | None = None,
) -> dict[str, Any]:
    runs = [item for item in report.get("runs") or [] if isinstance(item, dict)]
    tool_counts: dict[str, int] = {}
    for run in runs:
        tool_name = str(run.get("tool") or "unknown")
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

    task_family_counts: dict[str, int] = {}
    prompt_family_counts: dict[str, int] = {}
    for row in queue_rows:
        review = _first_dict(row.get("review"))
        family = str(review.get("task_family") or "unknown")
        task_family_counts[family] = task_family_counts.get(family, 0) + 1
        trace = _first_dict(row.get("trace"))
        prompt_family = str(
            trace.get("prompt_family")
            or _first_dict(_first_dict(trace.get("telemetry")).get("heuristic_summary")).get("promptFamily")
            or "unknown"
        )
        prompt_family_counts[prompt_family] = prompt_family_counts.get(prompt_family, 0) + 1

    durations = [
        int(event.get("durationMs") or 0) for event in events if isinstance(event, dict) and event.get("durationMs")
    ]
    provider_attempts = 0
    fallback_count = 0
    total_retries = 0
    warning_count = 0
    abstention_count = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        payload = _first_dict(event.get("payload"))
        output = _first_dict(payload.get("output"))
        provider_summary = _first_dict(output.get("providerPathwaySummary"))
        provider_attempts += int(provider_summary.get("attemptCount") or 0)
        total_retries += int(provider_summary.get("totalRetries") or 0)
        fallback_count += len(provider_summary.get("fallbackReasons") or [])
        warning_count += len(output.get("warnings") or [])
        if str(output.get("answerStatus") or "") in {"abstained", "insufficient_evidence"}:
            abstention_count += 1

    first_batch_id = next(
        (item.get("batchId") for item in events if isinstance(item, dict) and item.get("batchId")),
        None,
    )
    first_run_id = next(
        (item.get("runId") for item in events if isinstance(item, dict) and item.get("runId")),
        None,
    )

    return {
        "batchId": batch_id or report.get("batchId") or first_batch_id,
        "runId": run_id or report.get("runId") or first_run_id,
        "generatedAt": report.get("generatedAt"),
        "scenarioFile": scenario_file or report.get("scenarioFile"),
        "toolCounts": tool_counts,
        "taskFamilyCounts": task_family_counts,
        "promptFamilyCounts": prompt_family_counts,
        "runCount": len(runs),
        "capturedEventCount": len(events),
        "reviewQueueRowCount": len(queue_rows),
        "totalDurationMs": sum(durations),
        "maxDurationMs": max(durations, default=0),
        "providerAttemptCount": provider_attempts,
        "fallbackCount": fallback_count,
        "totalRetries": total_retries,
        "abstentionCount": abstention_count,
        "warningCount": warning_count,
        "schemaVersion": 1,
    }


def build_batch_ledger_rows(
    report: dict[str, Any],
    events: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs = [item for item in report.get("runs") or [] if isinstance(item, dict)]
    ledger_rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        event = events[index] if index < len(events) and isinstance(events[index], dict) else {}
        queue_row = queue_rows[index] if index < len(queue_rows) and isinstance(queue_rows[index], dict) else {}
        payload = _first_dict(event.get("payload"))
        output = _first_dict(payload.get("output"))
        review = _first_dict(queue_row.get("review"))
        provider_summary = _first_dict(output.get("providerPathwaySummary"))
        ledger_rows.append(
            {
                "batchId": event.get("batchId") or report.get("batchId"),
                "runId": event.get("runId") or report.get("runId"),
                "scenarioName": run.get("name"),
                "tool": run.get("tool"),
                "taskFamily": payload.get("taskFamily") or review.get("task_family"),
                "promptFamily": _first_dict(output.get("heuristicSummary")).get("promptFamily"),
                "searchSessionId": event.get("searchSessionId") or output.get("searchSessionId"),
                "capturedEventId": event.get("eventId"),
                "reviewQueueRowId": queue_row.get("trace_id"),
                "answerStatus": output.get("answerStatus"),
                "resultStatus": output.get("resultStatus") or output.get("status"),
                "durationMs": event.get("durationMs"),
                "sourceCount": output.get("sourceCount"),
                "providerCount": len(provider_summary.get("providersUsed") or []),
                "fallbackCount": len(provider_summary.get("fallbackReasons") or []),
                "totalRetries": provider_summary.get("totalRetries") or 0,
            }
        )
    return ledger_rows


def write_batch_ledger_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "batchId",
        "runId",
        "scenarioName",
        "tool",
        "taskFamily",
        "promptFamily",
        "searchSessionId",
        "capturedEventId",
        "reviewQueueRowId",
        "answerStatus",
        "resultStatus",
        "durationMs",
        "sourceCount",
        "providerCount",
        "fallbackCount",
        "totalRetries",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
