"""Orchestration for static and live eval canary runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import (
    SCHEMA_VERSION,
    _iter_raw_eval_rows,
    _jsonl_files,
    _result_item,
)
from .validators import (
    _validate_planner_live_response,
    _validate_runtime_live_response,
    _validate_synthesis_live_response,
    validate_eval_item,
)


def run_eval_canary(
    dataset_root: Path,
    *,
    family_filter: str | None = None,
    item_id_filter: str | None = None,
) -> dict[str, Any]:
    files = _jsonl_files(dataset_root, family_filter=family_filter)
    seen_ids: set[str] = set()
    item_results: list[dict[str, Any]] = []
    family_counts: dict[str, dict[str, int]] = {}

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    raw_item = json.loads(line)
                except json.JSONDecodeError as error:
                    item_results.append(
                        _result_item(
                            item_id=f"{file_path.name}:{line_number}",
                            family="unknown",
                            path=str(file_path),
                            line_number=line_number,
                            errors=[f"invalid JSON: {error.msg}"],
                            warnings=[],
                        )
                    )
                    continue

                if item_id_filter is not None:
                    item_id = str(raw_item.get("meta", {}).get("id") or "")
                    if item_id != item_id_filter:
                        continue

                result = validate_eval_item(raw_item, path=file_path, line_number=line_number)
                if result["id"] in seen_ids:
                    result["errors"].append(f"duplicate item id: {result['id']}")
                    result["status"] = "failed"
                else:
                    seen_ids.add(result["id"])
                item_results.append(result)

                family = result["family"]
                counts = family_counts.setdefault(
                    family,
                    {"itemCount": 0, "passed": 0, "warnings": 0, "failed": 0},
                )
                counts["itemCount"] += 1
                if result["status"] == "passed":
                    counts["passed"] += 1
                elif result["status"] == "warning":
                    counts["warnings"] += 1
                else:
                    counts["failed"] += 1

    summary = {
        "datasetRoot": str(dataset_root),
        "familiesSelected": sorted({result["family"] for result in item_results}),
        "fileCount": len(files),
        "itemCount": len(item_results),
        "passedItems": sum(1 for item in item_results if item["status"] == "passed"),
        "warningItems": sum(1 for item in item_results if item["status"] == "warning"),
        "failedItems": sum(1 for item in item_results if item["status"] == "failed"),
        "itemIdFilter": item_id_filter,
    }

    families = [
        {"family": family, **counts} for family, counts in sorted(family_counts.items(), key=lambda item: item[0])
    ]

    return {
        "schemaVersion": SCHEMA_VERSION,
        "summary": summary,
        "families": families,
        "items": item_results,
    }


def _default_live_tool_sequence(
    raw_item: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]] | None:
    meta = raw_item.get("meta") or {}
    family = meta.get("task_family")
    payload = raw_item.get("input") or {}
    live_eval = raw_item.get("live_eval") or {}
    if family == "runtime":
        return [("get_runtime_status", {})]
    if family == "planner":
        query = payload.get("query")
        if isinstance(query, str) and query.strip():
            return [("research", {"query": query})]
    if family == "synthesis":
        research_query = live_eval.get("research_query")
        question = payload.get("follow_up_question")
        if (
            isinstance(research_query, str)
            and research_query.strip()
            and isinstance(question, str)
            and question.strip()
        ):
            return [
                ("research", {"query": research_query}),
                ("follow_up_research", {"question": question}),
            ]
    return None


async def run_live_eval_canary(
    dataset_root: Path,
    *,
    family_filter: str | None = None,
    item_id_filter: str | None = None,
) -> dict[str, Any]:
    base_report = run_eval_canary(dataset_root, family_filter=family_filter, item_id_filter=item_id_filter)
    try:
        from paper_chaser_mcp.server import _execute_tool
    except Exception as exc:  # pragma: no cover - import failure is runtime/environment specific
        base_report["liveSummary"] = {
            "attempted": 0,
            "passed": 0,
            "failed": 0,
            "skipped": len(base_report["items"]),
            "error": f"Unable to import live runtime: {exc}",
        }
        return base_report

    row_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path, line_number, raw_eval_item in _iter_raw_eval_rows(
        dataset_root,
        family_filter=family_filter,
        item_id_filter=item_id_filter,
    ):
        key = (str(path), str(raw_eval_item.get("meta", {}).get("id") or ""), line_number)
        row_map[key] = raw_eval_item

    attempted = 0
    passed = 0
    failed = 0
    skipped = 0

    for item in base_report["items"]:
        key = (item["path"], item["id"], item["lineNumber"])
        matched_item = row_map.get(key)
        if item["status"] == "failed" or matched_item is None:
            item["liveEvaluation"] = {
                "status": "skipped",
                "reason": "schema_failed_or_missing_raw_item",
            }
            skipped += 1
            continue

        tool_sequence = _default_live_tool_sequence(matched_item)
        if tool_sequence is None:
            item["liveEvaluation"] = {
                "status": "skipped",
                "reason": "no_live_mapping_for_family",
            }
            skipped += 1
            continue

        tool_name = tool_sequence[-1][0]
        attempted += 1
        try:
            response: dict[str, Any] | None = None
            previous_response: dict[str, Any] | None = None
            executed_steps: list[dict[str, Any]] = []
            for index, (sequence_tool_name, base_arguments) in enumerate(tool_sequence):
                arguments = dict(base_arguments)
                if sequence_tool_name == "follow_up_research" and previous_response is not None:
                    search_session_id = previous_response.get("searchSessionId")
                    if isinstance(search_session_id, str) and search_session_id:
                        arguments["searchSessionId"] = search_session_id
                current_response = await _execute_tool(sequence_tool_name, arguments, ctx=None)
                executed_steps.append(
                    {
                        "tool": sequence_tool_name,
                        "arguments": arguments,
                        "searchSessionId": (
                            current_response.get("searchSessionId") if isinstance(current_response, dict) else None
                        ),
                    }
                )
                previous_response = current_response if isinstance(current_response, dict) else None
                if index == len(tool_sequence) - 1:
                    response = current_response if isinstance(current_response, dict) else {}
            if response is None:
                response = {}
        except Exception as exc:  # pragma: no cover - environment/provider specific
            item["liveEvaluation"] = {
                "status": "failed",
                "tool": tool_name,
                "arguments": tool_sequence,
                "errors": [f"tool execution failed: {type(exc).__name__}: {exc}"],
            }
            failed += 1
            continue

        family = matched_item["meta"]["task_family"]
        expected_value = matched_item.get("expected")
        expected: dict[str, Any] = expected_value if isinstance(expected_value, dict) else {}
        if family == "runtime":
            live_errors = _validate_runtime_live_response(expected, response)
        elif family == "planner":
            live_errors = _validate_planner_live_response(expected, response)
        elif family == "synthesis":
            live_errors = _validate_synthesis_live_response(expected, response)
        else:
            live_errors = []

        live_status = "passed" if not live_errors else "failed"
        if live_status == "passed":
            passed += 1
        else:
            failed += 1
        item["liveEvaluation"] = {
            "status": live_status,
            "tool": tool_name,
            "arguments": tool_sequence,
            "errors": live_errors,
            "observedIntent": (response.get("intent") if isinstance(response, dict) else None),
            "executedSteps": executed_steps,
        }

    base_report["liveSummary"] = {
        "attempted": attempted,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "mode": "configured_runtime",
        "note": (
            "Live mode evaluates the currently configured Paper Chaser runtime. "
            "It does not force all providers or tools on; compare configs "
            "explicitly when you want provider-specific model evaluation."
        ),
    }
    return base_report


def render_canary_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "Eval Canary Summary",
        f"dataset root: {summary['datasetRoot']}",
        f"files: {summary['fileCount']}",
        f"items: {summary['itemCount']}",
        f"passed: {summary['passedItems']}",
        f"warnings: {summary['warningItems']}",
        f"failed: {summary['failedItems']}",
        "",
        "Per family:",
    ]
    for family in report["families"]:
        lines.append(
            f"- {family['family']}: {family['itemCount']} items, {family['passed']} passed, "
            f"{family['warnings']} warnings, {family['failed']} failed"
        )
    return "\n".join(lines)
