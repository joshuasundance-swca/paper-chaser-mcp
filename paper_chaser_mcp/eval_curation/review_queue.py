"""Review queue loading and rendering for captured live eval events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .capture import _first_dict


def load_captured_eval_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_review_queue_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        task_family = str(payload.get("taskFamily") or "").strip()
        if not task_family:
            continue
        event_id = str(event.get("eventId") or "").strip()
        tool_name = str(payload.get("tool") or "unknown")
        input_payload = _first_dict(payload.get("input"))
        output_payload = _first_dict(payload.get("output"))
        queue.append(
            {
                "trace_id": event_id,
                "run_id": event.get("runId"),
                "batch_id": event.get("batchId"),
                "reviewed_at": None,
                "trace": {
                    "source_tool": tool_name,
                    "prompt_family": _first_dict(output_payload.get("heuristicSummary")).get("promptFamily"),
                    "query": input_payload.get("query"),
                    "query_context": input_payload.get("query") or input_payload.get("searchSessionId"),
                    "follow_up_question": input_payload.get("question"),
                    "source_id": input_payload.get("sourceId"),
                    "search_session_id": output_payload.get("searchSessionId") or event.get("searchSessionId"),
                    "duration_ms": event.get("durationMs"),
                    "telemetry": {
                        "provider_pathway_summary": output_payload.get("providerPathwaySummary") or {},
                        "stage_timings_ms": output_payload.get("stageTimingsMs") or {},
                        "confidence_signals": output_payload.get("confidenceSignals") or {},
                        "heuristic_summary": output_payload.get("heuristicSummary") or {},
                        "failure_summary": output_payload.get("failureSummary"),
                    },
                    "captured_output": output_payload,
                },
                "review": {
                    "promote": False,
                    "task_family": task_family,
                    "id": event_id,
                    "tags": payload.get("tags") or ["captured-live", task_family],
                    "expected": {},
                    "why_it_matters": "Fill in after reviewing this captured live run.",
                    "evaluation_target": payload.get("evaluationTarget", "internal_llm_role"),
                    "label_schema_version": 1,
                    "labels": {
                        "verdict": None,
                        "qualityBucket": None,
                        "split": "unreviewed",
                        "trainingEligibility": "undecided",
                        "trainingObjective": None,
                        "preferredExportFormats": [],
                        "notes": None,
                    },
                },
            }
        )
    return queue


def write_review_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def render_review_queue(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)
