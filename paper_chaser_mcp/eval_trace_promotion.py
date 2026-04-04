"""Helpers for promoting reviewed live traces into evaluation dataset rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_reviewed_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_eval_input(task_family: str, trace: dict[str, Any]) -> dict[str, Any]:
    if task_family == "planner":
        return {"query": trace.get("query", "")}
    if task_family == "synthesis":
        return {
            "query_context": trace.get("query_context", trace.get("query", "")),
            "follow_up_question": trace.get("follow_up_question", ""),
            "evidence_quality": trace.get("evidence_quality", "mixed"),
        }
    if task_family == "abstention":
        return {"query": trace.get("query", "")}
    if task_family == "provenance":
        return {
            "query_context": trace.get("query_context", trace.get("query", "")),
            "source_id": trace.get("source_id", ""),
        }
    if task_family in {"runtime", "misc"}:
        return {"query_context": trace.get("query_context", trace.get("query", ""))}
    raise ValueError(f"Unsupported task family for promotion: {task_family}")


def promote_reviewed_traces(rows: list[dict[str, Any]], *, dataset_version: str) -> list[dict[str, Any]]:
    promoted: list[dict[str, Any]] = []
    for row in rows:
        review = row.get("review") or {}
        if not review.get("promote"):
            continue
        task_family = str(review.get("task_family") or "").strip()
        if not task_family:
            raise ValueError("review.task_family is required for promoted traces")
        trace = row.get("trace") or {}
        promoted.append(
            {
                "meta": {
                    "id": review.get("id") or row.get("trace_id"),
                    "task_family": task_family,
                    "dataset_version": dataset_version,
                    "schema_version": 1,
                    "origin": "trace_mined",
                    "review_status": review.get("review_status", "validated"),
                    "tags": review.get("tags", ["trace-promoted"]),
                },
                "input": _build_eval_input(task_family, trace),
                "expected": review.get("expected", {}),
                "why_it_matters": review.get("why_it_matters", "Promoted from a reviewed live trace."),
                "notes": review.get("notes"),
                "evaluation_target": review.get("evaluation_target", "internal_llm_role"),
                "review_labels": review.get("labels") or {},
                "lineage": {
                    "traceId": row.get("trace_id"),
                    "sourceTool": trace.get("source_tool"),
                    "reviewedAt": row.get("reviewed_at"),
                },
            }
        )
    return promoted


def write_promoted_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def render_promoted_rows(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)
