"""Helpers for capturing live eval candidates and shaping review queues."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _compact_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "sourceId": source.get("sourceId"),
        "sourceAlias": source.get("sourceAlias"),
        "title": source.get("title"),
        "provider": source.get("provider"),
        "sourceType": source.get("sourceType"),
        "verificationStatus": source.get("verificationStatus"),
        "accessStatus": source.get("accessStatus"),
        "confidence": source.get("confidence"),
    }


def _compact_recommendations(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    compact: list[str] = []
    for item in items[:6]:
        if isinstance(item, str) and item.strip():
            compact.append(item.strip())
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("label") or item.get("action") or "").strip()
            if title:
                compact.append(title)
    return compact


def _compact_theme(theme: dict[str, Any]) -> dict[str, Any]:
    representative = []
    for paper in theme.get("representativePapers") or []:
        if isinstance(paper, dict):
            representative.append(
                {
                    "paperId": paper.get("paperId"),
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                }
            )
    return {
        "title": theme.get("title"),
        "summary": theme.get("summary"),
        "matchedConcepts": theme.get("matchedConcepts") or [],
        "representativePapers": representative,
    }


def _compact_graph_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": node.get("id"),
        "kind": node.get("kind"),
        "label": node.get("label"),
        "score": node.get("score"),
    }


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _compact_provider_outcomes(items: Any) -> dict[str, Any]:
    if not isinstance(items, list):
        return {
            "attemptCount": 0,
            "successCount": 0,
            "providersUsed": [],
            "totalRetries": 0,
            "fallbackReasons": [],
            "maxLatencyMs": 0,
        }
    outcomes = [item for item in items if isinstance(item, dict)]
    providers_used = sorted({str(item.get("provider") or "").strip() for item in outcomes if item.get("provider")})
    fallback_reasons = [
        str(item.get("fallbackReason") or "").strip()
        for item in outcomes
        if str(item.get("fallbackReason") or "").strip()
    ]
    success_count = sum(1 for item in outcomes if str(item.get("statusBucket") or "") == "success")
    total_retries = sum(int(item.get("retries") or 0) for item in outcomes)
    max_latency_ms = max((int(item.get("latencyMs") or 0) for item in outcomes), default=0)
    return {
        "attemptCount": len(outcomes),
        "successCount": success_count,
        "providersUsed": providers_used,
        "totalRetries": total_retries,
        "fallbackReasons": fallback_reasons,
        "maxLatencyMs": max_latency_ms,
    }


def _extract_telemetry(result: dict[str, Any]) -> dict[str, Any]:
    execution_provenance = _first_dict(result.get("executionProvenance"))
    strategy_metadata = _first_dict(result.get("strategyMetadata"))
    provider_outcomes = result.get("providerOutcomes")
    if not isinstance(provider_outcomes, list):
        provider_outcomes = execution_provenance.get("providerOutcomes")
    if not isinstance(provider_outcomes, list):
        provider_outcomes = strategy_metadata.get("providerOutcomes")
    if not isinstance(provider_outcomes, list):
        provider_outcomes = []

    stage_timings_ms = execution_provenance.get("stageTimingsMs")
    if not isinstance(stage_timings_ms, dict):
        stage_timings_ms = strategy_metadata.get("stageTimingsMs")
    if not isinstance(stage_timings_ms, dict):
        stage_timings_ms = {}

    confidence_signals = {
        "intentConfidence": strategy_metadata.get("intentConfidence"),
        "routingConfidence": strategy_metadata.get("routingConfidence"),
        "answerability": result.get("answerability"),
    }

    return {
        "providerOutcomes": provider_outcomes,
        "providerPathwaySummary": _compact_provider_outcomes(provider_outcomes),
        "stageTimingsMs": stage_timings_ms,
        "strategyMetadata": strategy_metadata,
        "confidenceSignals": {key: value for key, value in confidence_signals.items() if value is not None},
        "failureSummary": result.get("failureSummary"),
        "abstentionDetails": result.get("abstentionDetails"),
    }


def build_eval_capture_payload(
    tool_name: str, arguments: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any] | None:
    search_session_id = result.get("searchSessionId")
    telemetry = _extract_telemetry(result)
    if tool_name == "research":
        sources = [item for item in result.get("sources") or [] if isinstance(item, dict)]
        return {
            "tool": tool_name,
            "taskFamily": "planner",
            "evaluationTarget": "internal_llm_role",
            "input": {
                "query": arguments.get("query"),
            },
            "output": {
                "searchSessionId": search_session_id,
                "intent": result.get("intent"),
                "status": result.get("status"),
                "summary": result.get("summary"),
                "routingSummary": result.get("routingSummary"),
                "coverageSummary": result.get("coverageSummary") or result.get("coverage"),
                "trustSummary": result.get("trustSummary"),
                "resultState": result.get("resultState"),
                "executionProvenance": result.get("executionProvenance"),
                "stageTimingsMs": telemetry["stageTimingsMs"],
                "providerOutcomes": telemetry["providerOutcomes"],
                "providerPathwaySummary": telemetry["providerPathwaySummary"],
                "confidenceSignals": telemetry["confidenceSignals"],
                "failureSummary": telemetry["failureSummary"],
                "evidenceGaps": result.get("evidenceGaps") or [],
                "sourceCount": len(sources),
                "sources": [_compact_source(source) for source in sources[:8]],
            },
            "tags": ["captured-live", "guided", "planner"],
        }
    if tool_name == "follow_up_research":
        sources = [item for item in result.get("sources") or [] if isinstance(item, dict)]
        return {
            "tool": tool_name,
            "taskFamily": "synthesis",
            "evaluationTarget": "internal_llm_role",
            "input": {
                "searchSessionId": arguments.get("searchSessionId") or search_session_id,
                "question": arguments.get("question"),
            },
            "output": {
                "searchSessionId": search_session_id,
                "answerStatus": result.get("answerStatus"),
                "answer": result.get("answer"),
                "selectedEvidenceIds": result.get("selectedEvidenceIds") or [],
                "selectedLeadIds": result.get("selectedLeadIds") or [],
                "abstentionDetails": result.get("abstentionDetails"),
                "resultState": result.get("resultState"),
                "executionProvenance": result.get("executionProvenance"),
                "stageTimingsMs": telemetry["stageTimingsMs"],
                "providerOutcomes": telemetry["providerOutcomes"],
                "providerPathwaySummary": telemetry["providerPathwaySummary"],
                "confidenceSignals": telemetry["confidenceSignals"],
                "failureSummary": telemetry["failureSummary"],
                "evidenceGaps": result.get("evidenceGaps") or [],
                "sourceCount": len(sources),
                "sources": [_compact_source(source) for source in sources[:8]],
            },
            "tags": ["captured-live", "guided", "synthesis"],
        }
    if tool_name == "inspect_source":
        source = _first_dict(result.get("source"))
        return {
            "tool": tool_name,
            "taskFamily": "provenance",
            "evaluationTarget": "internal_llm_role",
            "input": {
                "searchSessionId": arguments.get("searchSessionId") or search_session_id,
                "sourceId": arguments.get("sourceId"),
            },
            "output": {
                "searchSessionId": search_session_id,
                "source": _compact_source(source),
                "sourceResolution": result.get("sourceResolution"),
                "resultState": result.get("resultState"),
                "executionProvenance": result.get("executionProvenance"),
                "stageTimingsMs": telemetry["stageTimingsMs"],
                "providerOutcomes": telemetry["providerOutcomes"],
                "providerPathwaySummary": telemetry["providerPathwaySummary"],
                "directReadRecommendations": _compact_recommendations(result.get("directReadRecommendations")),
            },
            "tags": ["captured-live", "guided", "provenance"],
        }
    if tool_name == "get_runtime_status":
        return {
            "tool": tool_name,
            "taskFamily": "runtime",
            "evaluationTarget": "internal_llm_role",
            "input": {},
            "output": {
                "runtimeSummary": result.get("runtimeSummary"),
                "warnings": result.get("warnings") or [],
            },
            "tags": ["captured-live", "runtime"],
        }
    if tool_name == "search_papers_smart":
        structured_sources = [item for item in result.get("structuredSources") or [] if isinstance(item, dict)]
        strategy_metadata = result.get("strategyMetadata") if isinstance(result.get("strategyMetadata"), dict) else {}
        return {
            "tool": tool_name,
            "taskFamily": "planner",
            "toolRole": "expert_smart_search",
            "evaluationTarget": "internal_llm_role",
            "input": {
                "query": arguments.get("query"),
                "mode": arguments.get("mode"),
                "latencyProfile": arguments.get("latencyProfile"),
            },
            "output": {
                "searchSessionId": result.get("searchSessionId"),
                "resultStatus": result.get("resultStatus"),
                "answerability": result.get("answerability"),
                "routingSummary": result.get("routingSummary"),
                "strategyMetadata": strategy_metadata,
                "coverageSummary": result.get("coverageSummary"),
                "providerOutcomes": telemetry["providerOutcomes"],
                "providerPathwaySummary": telemetry["providerPathwaySummary"],
                "stageTimingsMs": telemetry["stageTimingsMs"],
                "confidenceSignals": telemetry["confidenceSignals"],
                "failureSummary": telemetry["failureSummary"],
                "evidenceGaps": result.get("evidenceGaps") or [],
                "sourceCount": len(structured_sources),
                "sources": [_compact_source(source) for source in structured_sources[:8]],
            },
            "tags": ["captured-live", "expert", "planner", "search_papers_smart"],
        }
    if tool_name == "ask_result_set":
        evidence = [item for item in result.get("evidence") or [] if isinstance(item, dict)]
        structured_sources = [item for item in result.get("structuredSources") or [] if isinstance(item, dict)]
        return {
            "tool": tool_name,
            "taskFamily": "synthesis",
            "toolRole": "expert_grounded_qa",
            "evaluationTarget": "internal_llm_role",
            "input": {
                "searchSessionId": arguments.get("searchSessionId"),
                "question": arguments.get("question"),
                "answerMode": arguments.get("answerMode"),
            },
            "output": {
                "searchSessionId": result.get("searchSessionId"),
                "answerStatus": result.get("answerStatus"),
                "answer": result.get("answer"),
                "coverageSummary": result.get("coverageSummary"),
                "providerOutcomes": telemetry["providerOutcomes"],
                "providerPathwaySummary": telemetry["providerPathwaySummary"],
                "stageTimingsMs": telemetry["stageTimingsMs"],
                "confidenceSignals": telemetry["confidenceSignals"],
                "failureSummary": telemetry["failureSummary"],
                "evidence": evidence[:8],
                "selectedEvidenceIds": [
                    str(item.get("evidenceId") or "").strip()
                    for item in evidence
                    if str(item.get("evidenceId") or "").strip()
                ],
                "sourceCount": len(structured_sources),
                "sources": [_compact_source(source) for source in structured_sources[:8]],
            },
            "tags": ["captured-live", "expert", "synthesis", "ask_result_set"],
        }
    if tool_name == "map_research_landscape":
        themes = [item for item in result.get("themes") or [] if isinstance(item, dict)]
        structured_sources = [item for item in result.get("structuredSources") or [] if isinstance(item, dict)]
        return {
            "tool": tool_name,
            "taskFamily": "misc",
            "toolRole": "landscape_mapping",
            "evaluationTarget": "internal_llm_role",
            "input": {
                "searchSessionId": arguments.get("searchSessionId"),
                "maxThemes": arguments.get("maxThemes"),
                "latencyProfile": arguments.get("latencyProfile"),
            },
            "output": {
                "searchSessionId": result.get("searchSessionId"),
                "themeCount": len(themes),
                "themes": [_compact_theme(theme) for theme in themes[:5]],
                "gaps": result.get("gaps") or [],
                "disagreements": result.get("disagreements") or [],
                "suggestedNextSearches": result.get("suggestedNextSearches") or [],
                "coverageSummary": result.get("coverageSummary"),
                "providerOutcomes": telemetry["providerOutcomes"],
                "providerPathwaySummary": telemetry["providerPathwaySummary"],
                "stageTimingsMs": telemetry["stageTimingsMs"],
                "confidenceSignals": telemetry["confidenceSignals"],
                "sourceCount": len(structured_sources),
                "sources": [_compact_source(source) for source in structured_sources[:8]],
            },
            "tags": ["captured-live", "expert", "misc", "map_research_landscape"],
        }
    if tool_name == "expand_research_graph":
        nodes = [item for item in result.get("nodes") or [] if isinstance(item, dict)]
        frontier = [item for item in result.get("frontier") or [] if isinstance(item, dict)]
        return {
            "tool": tool_name,
            "taskFamily": "misc",
            "toolRole": "graph_expansion",
            "evaluationTarget": "internal_llm_role",
            "input": {
                "seedSearchSessionId": arguments.get("seedSearchSessionId"),
                "seedPaperIds": arguments.get("seedPaperIds") or [],
                "direction": arguments.get("direction"),
                "hops": arguments.get("hops"),
            },
            "output": {
                "searchSessionId": result.get("searchSessionId"),
                "nodeCount": len(nodes),
                "edgeCount": len(result.get("edges") or []),
                "frontier": [_compact_graph_node(node) for node in frontier[:8]],
                "warnings": (result.get("agentHints") or {}).get("warnings") or [],
                "providerOutcomes": telemetry["providerOutcomes"],
                "providerPathwaySummary": telemetry["providerPathwaySummary"],
                "stageTimingsMs": telemetry["stageTimingsMs"],
                "confidenceSignals": telemetry["confidenceSignals"],
            },
            "tags": ["captured-live", "expert", "misc", "expand_research_graph"],
        }
    return None


def maybe_capture_eval_candidate(
    *,
    workspace_registry: Any,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    run_id: str | None = None,
    batch_id: str | None = None,
    duration_ms: int | None = None,
) -> None:
    capture_fn = getattr(workspace_registry, "capture_eval_event", None)
    if not callable(capture_fn) or not isinstance(result, dict):
        return
    payload = build_eval_capture_payload(tool_name, arguments, result)
    if payload is None:
        return
    capture_fn(
        event_type="guided_tool_result",
        payload=payload,
        search_session_id=result.get("searchSessionId") if isinstance(result.get("searchSessionId"), str) else None,
        run_id=run_id,
        batch_id=batch_id,
        duration_ms=duration_ms,
    )


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
    for row in queue_rows:
        review = _first_dict(row.get("review"))
        family = str(review.get("task_family") or "unknown")
        task_family_counts[family] = task_family_counts.get(family, 0) + 1

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


def write_review_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def render_review_queue(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)
