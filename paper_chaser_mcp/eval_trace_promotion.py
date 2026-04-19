"""Helpers for promoting reviewed live traces into evaluation dataset rows."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

TASK_FAMILY_SEED_FILES = {
    "planner": "planner.seed.jsonl",
    "synthesis": "synthesis.seed.jsonl",
    "abstention": "abstention.seed.jsonl",
    "provenance": "provenance.seed.jsonl",
    "runtime": "runtime.seed.jsonl",
    "misc": "misc.seed.jsonl",
}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _infer_training_objective(task_family: str) -> str:
    return {
        "planner": "planner_routing",
        "synthesis": "grounded_synthesis",
        "abstention": "abstention_safety",
        "provenance": "source_provenance",
        "runtime": "runtime_truth",
        "misc": "supporting_reasoning",
    }.get(task_family, "trace_eval")


def _infer_expected(task_family: str, trace: dict[str, Any]) -> dict[str, Any]:
    captured_output = _dict(trace.get("captured_output"))
    provider_summary = _dict(_dict(trace.get("telemetry")).get("provider_pathway_summary"))
    providers = _list_of_strings(provider_summary.get("providersUsed"))
    if not providers:
        providers = _list_of_strings(_dict(captured_output.get("coverageSummary")).get("providersAttempted"))
    if task_family == "planner":
        intent = str(
            captured_output.get("intent") or _dict(captured_output.get("routingSummary")).get("intent") or "discovery"
        )
        status = str(captured_output.get("status") or captured_output.get("resultStatus") or "succeeded")
        return {
            "acceptable_intents": [intent],
            "unacceptable_intents": [],
            "acceptable_provider_hints": providers or ["semantic_scholar"],
            "must_surface_clarification": False,
            "should_allow_partial": status == "partial",
        }
    if task_family == "synthesis":
        answer_status = str(captured_output.get("answerStatus") or "answered")
        selected_evidence_ids = _list_of_strings(captured_output.get("selectedEvidenceIds"))
        return {
            "expected_answer_status": answer_status,
            "should_abstain": answer_status in {"abstained", "insufficient_evidence"},
            "must_cite_evidence": bool(selected_evidence_ids),
            "should_preserve_uncertainty": True,
            "required_evidence_traits": (
                ["insufficient-evidence"]
                if answer_status in {"abstained", "insufficient_evidence"}
                else ["grounded-evidence"]
            ),
        }
    if task_family == "abstention":
        answer_status = str(captured_output.get("answerStatus") or "answered")
        behavior = "abstain" if answer_status in {"abstained", "insufficient_evidence"} else "answer"
        return {
            "correct_behavior": behavior,
            "required_markers": ["uncertainty"],
            "disallowed_patterns": ["invented certainty"],
            "should_preserve_uncertainty": True,
        }
    if task_family == "provenance":
        source = _dict(captured_output.get("source"))
        return {
            "expected_source_type": str(source.get("sourceType") or "unknown"),
            "expected_trust_state": str(source.get("verificationStatus") or "unverified"),
            "expected_access_state": str(source.get("accessStatus") or "access_unverified"),
            "should_recommend_direct_read": True,
        }
    if task_family == "runtime":
        runtime_summary = _dict(captured_output.get("runtimeSummary"))
        return {
            "expected_profile": str(runtime_summary.get("effectiveProfile") or "expert"),
            "must_report_configured_provider": True,
            "must_report_active_provider": True,
            "must_surface_warnings": False,
            "must_include_sets": ["activeProviders", "disabledProviders"],
        }
    return {
        "supporting_role": str(trace.get("source_tool") or "misc_trace"),
        "required_markers": ["task-output"],
        "disallowed_patterns": ["empty-output"],
        "must_preserve_cost_awareness": False,
    }


def apply_yolo_review_defaults(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated_rows = copy.deepcopy(rows)
    session_queries: dict[str, str] = {}
    for row in updated_rows:
        trace = _dict(row.get("trace"))
        review = _dict(row.get("review"))
        if str(review.get("task_family") or "") == "planner":
            search_session_id = str(trace.get("search_session_id") or "").strip()
            query = str(trace.get("query") or "").strip()
            if search_session_id and query:
                session_queries[search_session_id] = query

    for row in updated_rows:
        trace = _dict(row.get("trace"))
        review = row.setdefault("review", {})
        labels = review.setdefault("labels", {})
        task_family = str(review.get("task_family") or "misc").strip()
        review["promote"] = True
        review.setdefault("id", row.get("trace_id"))
        review.setdefault("review_status", "validated")
        review.setdefault("tags", ["trace-promoted", task_family, str(trace.get("source_tool") or "trace")])
        review.setdefault("evaluation_target", "internal_llm_role")
        review.setdefault("why_it_matters", f"YOLO-promoted trace for {task_family} evaluation.")
        review.setdefault("notes", "Auto-promoted in yolo mode. Review later if needed.")
        if not _dict(review.get("expected")):
            review["expected"] = _infer_expected(task_family, trace)

        labels.setdefault("verdict", "gold")
        labels.setdefault("qualityBucket", "high")
        labels.setdefault("split", "train_candidate")
        labels.setdefault("trainingEligibility", "approved")
        labels.setdefault("trainingObjective", _infer_training_objective(task_family))
        labels.setdefault("preferredExportFormats", ["foundry-eval", "hf-dataset", "training-chat"])
        labels.setdefault("notes", "Auto-approved in yolo mode.")

        if task_family == "synthesis":
            live_eval = _dict(row.get("live_eval"))
            if not live_eval.get("research_query"):
                search_session_id = str(trace.get("search_session_id") or "").strip()
                research_query = session_queries.get(search_session_id)
                if research_query:
                    row["live_eval"] = {"research_query": research_query}
    return updated_rows


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
                    "batchId": row.get("batch_id"),
                    "runId": row.get("run_id"),
                },
                **({"live_eval": row.get("live_eval")} if isinstance(row.get("live_eval"), dict) else {}),
            }
        )
    return promoted


def group_promoted_rows_by_family(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        meta = _dict(row.get("meta"))
        family = str(meta.get("task_family") or "").strip()
        if not family:
            continue
        grouped.setdefault(family, []).append(row)
    return grouped


def write_promoted_rows_by_family(dataset_root: Path, rows: list[dict[str, Any]]) -> list[Path]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for family, family_rows in sorted(group_promoted_rows_by_family(rows).items()):
        file_name = TASK_FAMILY_SEED_FILES.get(family, f"{family}.seed.jsonl")
        path = dataset_root / file_name
        write_promoted_rows(path, family_rows)
        written.append(path)
    return written


def build_live_matrix_payload(
    providers: list[str],
    *,
    latency_profile: str = "balanced",
    disable_embeddings: bool = True,
) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for provider in providers:
        provider_name = str(provider).strip()
        if not provider_name:
            continue
        scenarios.append(
            {
                "name": f"{provider_name}-live",
                "env": {
                    "PAPER_CHASER_ENABLE_AGENTIC": "true",
                    "PAPER_CHASER_AGENTIC_PROVIDER": provider_name,
                    "PAPER_CHASER_DISABLE_EMBEDDINGS": "true" if disable_embeddings else "false",
                    "PAPER_CHASER_GUIDED_RESEARCH_LATENCY_PROFILE": latency_profile,
                    "PAPER_CHASER_GUIDED_FOLLOW_UP_LATENCY_PROFILE": latency_profile,
                },
            }
        )
    return {"scenarios": scenarios}


def write_live_matrix_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_promoted_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def render_promoted_rows(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)
