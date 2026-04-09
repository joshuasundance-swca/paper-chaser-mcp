from pathlib import Path

from paper_chaser_mcp.agentic.workspace import WorkspaceRegistry
from paper_chaser_mcp.eval_curation import (
    build_batch_ledger_rows,
    build_batch_summary,
    build_review_queue_rows,
    load_captured_eval_events,
    maybe_capture_eval_candidate,
    write_batch_ledger_csv,
)


def test_workspace_registry_capture_eval_event_writes_jsonl(tmp_path: Path) -> None:
    trace_path = tmp_path / "captured.jsonl"
    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False, eval_trace_path=str(trace_path))
    registry.capture_eval_event(
        event_type="guided_tool_result",
        payload={"tool": "research"},
        search_session_id=None,
        run_id="run_test_001",
        batch_id="batch_test_001",
        duration_ms=123,
    )

    text = trace_path.read_text(encoding="utf-8")
    assert "guided_tool_result" in text
    assert '"tool": "research"' in text
    assert '"runId": "run_test_001"' in text
    assert '"batchId": "batch_test_001"' in text
    assert '"durationMs": 123' in text


def test_maybe_capture_eval_candidate_records_guided_research(tmp_path: Path) -> None:
    trace_path = tmp_path / "captured.jsonl"
    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False, eval_trace_path=str(trace_path))
    maybe_capture_eval_candidate(
        workspace_registry=registry,
        tool_name="research",
        arguments={"query": "PFAS remediation"},
        result={
            "searchSessionId": "ssn_test",
            "intent": "discovery",
            "status": "succeeded",
            "summary": "Summary",
            "routingSummary": {
                "intent": "discovery",
                "querySpecificity": "low",
                "ambiguityLevel": "high",
                "retrievalHypotheses": ["pfas groundwater remediation"],
                "passModes": ["auto"],
            },
            "sources": [{"sourceId": "source-1", "title": "Paper", "provider": "openalex"}],
            "executionProvenance": {
                "executionMode": "guided_research",
                "serverPolicyApplied": "quality_first",
                "passesRun": 1,
                "passModes": ["auto"],
            },
            "resultState": {
                "status": "succeeded",
                "groundedness": "grounded",
                "hasInspectableSources": True,
                "canAnswerFollowUp": True,
                "bestNextInternalAction": "follow_up_research",
                "missingEvidenceType": "none",
            },
        },
        run_id="run_test_002",
        batch_id="batch_test_002",
        duration_ms=245,
    )

    text = trace_path.read_text(encoding="utf-8")
    assert '"taskFamily": "planner"' in text
    assert '"searchSessionId": "ssn_test"' in text
    assert '"providerPathwaySummary"' in text
    assert '"heuristicSummary"' in text
    assert '"promptFamily": "broad_ambiguous_literature"' in text
    assert '"querySpecificity": "low"' in text
    assert '"runId": "run_test_002"' in text


def test_build_review_queue_rows_from_captured_events_fixture() -> None:
    sample_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "captured-eval-events.sample.jsonl"
    events = load_captured_eval_events(sample_path)
    queue = build_review_queue_rows(events)

    assert len(queue) == 2
    assert queue[0]["batch_id"] == "batch_eval_20260404"
    assert queue[0]["run_id"] == "run_eval_20260404"
    assert queue[0]["review"]["task_family"] == "planner"
    assert queue[1]["review"]["task_family"] == "synthesis"
    assert queue[0]["review"]["promote"] is False
    assert queue[0]["review"]["labels"]["trainingEligibility"] == "undecided"
    assert queue[0]["trace"]["duration_ms"] == 412
    assert queue[0]["trace"]["telemetry"]["provider_pathway_summary"]["attemptCount"] == 2


def test_maybe_capture_eval_candidate_records_expert_tool(tmp_path: Path) -> None:
    trace_path = tmp_path / "captured-expert.jsonl"
    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False, eval_trace_path=str(trace_path))
    maybe_capture_eval_candidate(
        workspace_registry=registry,
        tool_name="search_papers_smart",
        arguments={"query": "PFAS remediation", "mode": "auto"},
        result={
            "searchSessionId": "ssn_expert",
            "resultStatus": "returned_results",
            "answerability": "answerable",
            "routingSummary": {"intent": "discovery"},
            "strategyMetadata": {
                "intent": "discovery",
                "stageTimingsMs": {"planning": 18},
                "querySpecificity": "low",
                "ambiguityLevel": "high",
                "retrievalHypotheses": ["pfas groundwater remediation"],
            },
            "coverageSummary": {"providersAttempted": ["semantic_scholar"]},
            "providerOutcomes": [
                {
                    "provider": "semantic_scholar",
                    "statusBucket": "success",
                    "latencyMs": 95,
                    "retries": 1,
                }
            ],
            "structuredSources": [{"sourceId": "source-1", "title": "Paper", "provider": "openalex"}],
        },
        duration_ms=601,
    )

    text = trace_path.read_text(encoding="utf-8")
    assert '"toolRole": "expert_smart_search"' in text
    assert '"stageTimingsMs": {"planning": 18}' in text
    assert '"heuristicSummary"' in text
    assert '"promptFamily": "broad_ambiguous_literature"' in text
    assert '"durationMs": 601' in text


def test_build_batch_summary_counts_prompt_families() -> None:
    events = [
        {
            "eventId": "evt-1",
            "payload": {
                "tool": "research",
                "taskFamily": "planner",
                "output": {"heuristicSummary": {"promptFamily": "mixed_regulatory_literature"}},
            },
        },
        {
            "eventId": "evt-2",
            "payload": {
                "tool": "search_papers_smart",
                "taskFamily": "planner",
                "output": {"heuristicSummary": {"promptFamily": "broad_ambiguous_literature"}},
            },
        },
    ]
    queue = build_review_queue_rows(events)

    summary = build_batch_summary({"runs": [{"tool": "research"}, {"tool": "search_papers_smart"}]}, events, queue)

    assert summary["promptFamilyCounts"]["mixed_regulatory_literature"] == 1
    assert summary["promptFamilyCounts"]["broad_ambiguous_literature"] == 1


def test_build_batch_summary_and_ledger_rows(tmp_path: Path) -> None:
    sample_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "captured-eval-events.sample.jsonl"
    events = load_captured_eval_events(sample_path)
    queue = build_review_queue_rows(events)
    report = {
        "batchId": "batch_eval_20260404",
        "runId": "run_eval_20260404",
        "generatedAt": 1775289720000,
        "scenarioFile": "tests/fixtures/evals/expert-batch.sample.json",
        "runs": [
            {"name": "planner", "tool": "research", "arguments": {}, "result": {}},
            {"name": "synthesis", "tool": "follow_up_research", "arguments": {}, "result": {}},
        ],
    }

    summary = build_batch_summary(report, events, queue)
    assert summary["runCount"] == 2
    assert summary["capturedEventCount"] == 2
    assert summary["taskFamilyCounts"]["planner"] == 1
    assert summary["fallbackCount"] == 1
    assert summary["totalRetries"] == 1

    ledger_rows = build_batch_ledger_rows(report, events, queue)
    assert len(ledger_rows) == 2
    assert ledger_rows[0]["batchId"] == "batch_eval_20260404"
    assert ledger_rows[0]["capturedEventId"] == "evt_planner_001"
    assert ledger_rows[0]["providerCount"] == 2

    ledger_path = tmp_path / "batch-ledger.csv"
    write_batch_ledger_csv(ledger_path, ledger_rows)
    csv_text = ledger_path.read_text(encoding="utf-8")
    assert "batchId,runId,scenarioName,tool" in csv_text
    assert "evt_planner_001" in csv_text
