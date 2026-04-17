import json
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


def test_environmental_science_eval_seed_fixture_covers_planner_synthesis_and_provenance() -> None:
    seed_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "environmental_science.seed.jsonl"
    rows = [json.loads(line) for line in seed_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    task_families = {row["meta"]["task_family"] for row in rows}

    assert {"planner", "synthesis", "provenance"}.issubset(task_families)
    assert any("deterministic" in json.dumps(row).lower() for row in rows)
    assert any("desert tortoise" in json.dumps(row).lower() for row in rows)
    assert any("pfas" in json.dumps(row).lower() for row in rows)


def test_stress_test_eval_seed_fixture_covers_core_stress_scenarios() -> None:
    seed_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "stress_test_remediation.seed.jsonl"
    rows = [json.loads(line) for line in seed_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    row_text = "\n".join(json.dumps(row) for row in rows).lower()

    assert len(rows) >= 6
    assert "microplastics" in row_text
    assert "mariana trench amphipods" in row_text
    assert "6ppd" in row_text
    assert "healing crystals" in row_text
    assert "mixed intent" in row_text
    assert "answered-but-abstaining" in row_text


def test_eval_autopilot_profiles_include_environmental_science_slice() -> None:
    profile_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "eval-autopilot-profiles.sample.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = payload["profiles"]["environmental-science-slice"]

    assert profile["generation"]["seedPreset"] == "environmental-consulting"
    assert profile["generation"]["emitFollowUp"] is True
    assert profile["workflow"]["autopilotPolicy"] == "review"
    assert "cross-provider-best" in profile["workflow"]["matrixPreset"]


def test_maybe_capture_eval_candidate_attaches_ranking_diagnostics_for_abstained_smart_search(
    tmp_path: Path,
) -> None:
    """Workstream G: abstained/failed cases should preserve ranking-diagnostic telemetry."""
    trace_path = tmp_path / "captured.jsonl"
    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False, eval_trace_path=str(trace_path))
    maybe_capture_eval_candidate(
        workspace_registry=registry,
        tool_name="search_papers_smart",
        arguments={"query": "ambiguous heritage ceramics query", "mode": "discovery"},
        result={
            "searchSessionId": "ssn_abs_1",
            "resultStatus": "abstained",
            "answerability": "insufficient_evidence",
            "strategyMetadata": {
                "intent": "discovery",
                "scoreBreakdown": {"semantic": 0.12, "lexical": 0.08, "recency": 0.02},
            },
            "rankingDiagnostics": {
                "topCandidates": [{"sourceId": "abs-1", "score": 0.12, "reason": "off_topic"}],
                "filteredCount": 17,
            },
            "preFilterCandidates": [
                {"sourceId": "abs-1", "provider": "openalex", "topicalRelevance": "off_topic"},
                {"sourceId": "abs-2", "provider": "semantic_scholar", "topicalRelevance": "unknown"},
            ],
            "classificationProvenance": {"router": "heuristic", "confidence": "low"},
            "synthesisMode": "abstain_with_leads",
            "evidenceQualityProfile": {"tier": "weak", "sampleSize": 0},
            "structuredSources": [],
        },
        run_id="run_abs_001",
        batch_id="batch_abs_001",
        duration_ms=180,
    )

    text = trace_path.read_text(encoding="utf-8")
    assert '"rankingDiagnostics"' in text
    assert '"scoreBreakdown"' in text
    assert '"classificationProvenance"' in text
    assert '"synthesisMode": "abstain_with_leads"' in text
    assert '"evidenceQualityProfile"' in text
    assert '"preFilterCandidates"' in text


def test_maybe_capture_eval_candidate_omits_ranking_diagnostics_for_succeeded_case(
    tmp_path: Path,
) -> None:
    """Successful runs without telemetry should not gain a ``rankingDiagnostics`` key."""
    trace_path = tmp_path / "captured.jsonl"
    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False, eval_trace_path=str(trace_path))
    maybe_capture_eval_candidate(
        workspace_registry=registry,
        tool_name="research",
        arguments={"query": "clean PFAS question"},
        result={
            "searchSessionId": "ssn_ok",
            "status": "succeeded",
            "summary": "ok",
            "sources": [{"sourceId": "s-1", "title": "Paper", "provider": "openalex"}],
        },
        run_id="run_ok_001",
        batch_id="batch_ok_001",
        duration_ms=100,
    )

    text = trace_path.read_text(encoding="utf-8")
    assert '"rankingDiagnostics"' not in text
