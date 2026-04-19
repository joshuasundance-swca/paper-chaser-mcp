"""Portable export helpers for eval and training dataset generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _prompt_from_reviewed_trace(row: dict[str, Any]) -> str:
    trace = row.get("trace") or {}
    source_tool = str(trace.get("source_tool") or "guided_tool")
    if trace.get("follow_up_question"):
        return str(trace.get("follow_up_question"))
    if trace.get("query"):
        return str(trace.get("query"))
    if trace.get("source_id"):
        return f"Inspect source {trace.get('source_id')} from saved session."
    return f"Execute {source_tool}."


def _assistant_from_reviewed_trace(row: dict[str, Any]) -> str:
    trace = row.get("trace") or {}
    captured_output = trace.get("captured_output") or {}
    review = row.get("review") or {}
    expected = review.get("expected") or {}
    payload = {"captured_output": captured_output, "review_expected": expected}
    return json.dumps(payload, ensure_ascii=True)


def export_foundry_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in rows:
        exported.append(
            {
                "id": row.get("meta", {}).get("id") or row.get("trace_id"),
                "task_family": row.get("meta", {}).get("task_family") or row.get("review", {}).get("task_family"),
                "input": row.get("input") or row.get("trace") or {},
                "expected": row.get("expected") or row.get("review", {}).get("expected") or {},
                "metadata": {
                    "tags": row.get("meta", {}).get("tags") or row.get("review", {}).get("tags") or [],
                    "evaluation_target": row.get("evaluation_target") or row.get("review", {}).get("evaluation_target"),
                    "review_labels": row.get("review_labels") or row.get("review", {}).get("labels") or {},
                },
            }
        )
    return exported


def export_hf_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in rows:
        exported.append(
            {
                "id": row.get("meta", {}).get("id") or row.get("trace_id"),
                "task_family": row.get("meta", {}).get("task_family") or row.get("review", {}).get("task_family"),
                "input": row.get("input") or row.get("trace") or {},
                "expected": row.get("expected") or row.get("review", {}).get("expected") or {},
                "evaluation_target": row.get("evaluation_target") or row.get("review", {}).get("evaluation_target"),
                "tags": row.get("meta", {}).get("tags") or row.get("review", {}).get("tags") or [],
                "review_labels": row.get("review_labels") or row.get("review", {}).get("labels") or {},
                "lineage": row.get("lineage") or {"traceId": row.get("trace_id")},
            }
        )
    return exported


def export_training_chat_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in rows:
        review = row.get("review") or {}
        labels = review.get("labels") or {}
        training_eligibility = str(labels.get("trainingEligibility") or "undecided")
        if training_eligibility not in {"approved", "gold"}:
            continue
        exported.append(
            {
                "messages": [
                    {"role": "user", "content": _prompt_from_reviewed_trace(row)},
                    {"role": "assistant", "content": _assistant_from_reviewed_trace(row)},
                ],
                "metadata": {
                    "trace_id": row.get("trace_id"),
                    "task_family": review.get("task_family"),
                    "training_objective": labels.get("trainingObjective"),
                    "tags": review.get("tags") or [],
                },
            }
        )
    return exported


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def render_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)
