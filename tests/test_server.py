import importlib
import json
from typing import Any

import pytest

from paper_chaser_mcp import server
from tests.helpers import RecordingSemanticClient, _streamable_http_event_payload


def test_build_http_app_uses_requested_path() -> None:
    app = server.build_http_app(path="/custom-mcp", transport="streamable-http")
    route_paths = [route.path for route in app.routes]

    assert "/custom-mcp" in route_paths


def test_run_server_uses_http_transport_settings() -> None:
    from paper_chaser_mcp.runtime import run_server
    from paper_chaser_mcp.settings import AppSettings

    calls: list[dict[str, Any]] = []

    class DummyApp:
        def run(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    class DummyLogger:
        def info(self, *args: Any, **kwargs: Any) -> None:
            return None

        def warning(self, *args: Any, **kwargs: Any) -> None:
            return None

    run_server(
        app=DummyApp(),
        logger=DummyLogger(),
        settings=AppSettings.from_env(
            {
                "PAPER_CHASER_TRANSPORT": "streamable-http",
                "PAPER_CHASER_HTTP_HOST": "0.0.0.0",
                "PAPER_CHASER_HTTP_PORT": "9000",
                "PAPER_CHASER_HTTP_PATH": "/custom-mcp",
            }
        ),
    )

    assert calls == [
        {
            "transport": "streamable-http",
            "host": "0.0.0.0",
            "port": 9000,
            "path": "/custom-mcp",
        }
    ]


@pytest.mark.asyncio
async def test_execute_tool_preserves_result_and_captures_eval_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    trace_path = tmp_path / "captured-events.jsonl"
    registry = server.WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        eval_trace_path=str(trace_path),
    )

    async def fake_dispatch_tool(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "searchSessionId": "ssn_execute_tool",
            "intent": "discovery",
            "status": "succeeded",
            "summary": "Captured through _execute_tool.",
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
        }

    monkeypatch.setattr(server, "dispatch_tool", fake_dispatch_tool)
    monkeypatch.setattr(server, "workspace_registry", registry)
    monkeypatch.setenv("PAPER_CHASER_EVAL_RUN_ID", "run_server_test")
    monkeypatch.setenv("PAPER_CHASER_EVAL_BATCH_ID", "batch_server_test")

    result = await server._execute_tool("research", {"query": "PFAS remediation"})

    assert result["searchSessionId"] == "ssn_execute_tool"
    payload = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert payload["runId"] == "run_server_test"
    assert payload["batchId"] == "batch_server_test"
    assert payload["durationMs"] >= 0
    assert payload["payload"]["tool"] == "research"
    assert payload["payload"]["output"]["searchSessionId"] == "ssn_execute_tool"


def test_streamable_http_app_handles_initialize_and_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.testclient import TestClient

    semantic = RecordingSemanticClient()
    monkeypatch.setattr(server, "client", semantic)

    headers = {"accept": "application/json, text/event-stream"}
    with TestClient(server.http_app) as client:
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        init_response = client.post("/mcp", json=initialize, headers=headers)
        session_id = init_response.headers["mcp-session-id"]
        init_payload = _streamable_http_event_payload(init_response.text)

        initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        initialized_response = client.post(
            "/mcp",
            json=initialized,
            headers={**headers, "mcp-session-id": session_id},
        )

        call = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "resolve_reference",
                "arguments": {"reference": "transformers"},
            },
        }
        call_response = client.post(
            "/mcp",
            json=call,
            headers={**headers, "mcp-session-id": session_id},
        )
        call_payload = _streamable_http_event_payload(call_response.text)

    assert init_response.status_code == 200
    assert initialized_response.status_code == 202
    instructions = init_payload["result"]["instructions"]
    assert "research" in instructions
    assert "follow_up_research" in instructions
    assert "resolve_reference" in instructions
    assert "inspect_source" in instructions
    assert call_response.status_code == 200
    structured = call_payload["result"]["structuredContent"]
    assert structured["status"] in {"resolved", "multiple_candidates", "no_match", "regulatory_primary_source"}
    assert structured["resolutionType"]
    assert "nextActions" in structured
    assert call_payload["result"]["isError"] is False


@pytest.mark.asyncio
async def test_fastmcp_client_returns_structured_tool_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastmcp import Client

    semantic = RecordingSemanticClient()
    monkeypatch.setattr(server, "client", semantic)

    async with Client(server.app) as client:
        tools = await client.list_tools()
        tool_map = {tool.name: tool for tool in tools}
        result = await client.call_tool(
            "resolve_reference",
            {"reference": "transformers"},
        )

    assert result.data["resolutionType"]
    assert result.data["status"] in {"resolved", "multiple_candidates", "no_match", "regulatory_primary_source"}
    assert "nextActions" in result.data
    assert tool_map["research"].annotations.readOnlyHint is True
    assert tool_map["research"].annotations.idempotentHint is True
    assert tool_map["research"].annotations.openWorldHint is True


@pytest.mark.asyncio
async def test_fastmcp_match_tool_unwraps_wrapped_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastmcp import Client

    semantic_client = server.SemanticScholarClient()

    async def fake_request(*args: object, **kwargs: object) -> dict:
        return {
            "paperId": None,
            "title": None,
            "data": [{"paperId": "wrapped-1", "title": "Wrapped best match"}],
        }

    monkeypatch.setattr(semantic_client, "_request", fake_request)
    monkeypatch.setattr(server, "client", semantic_client)

    async with Client(server.app) as client:
        result = await client.call_tool(
            "resolve_reference",
            {"reference": "wrapped match"},
        )

    best_match = result.data["bestMatch"]
    assert best_match["paper"]["paperId"] == "wrapped-1"
    assert best_match["paper"]["title"] == "Wrapped best match"


@pytest.mark.asyncio
async def test_fastmcp_resource_and_prompt_support_agent_onboarding() -> None:
    from fastmcp import Client

    async with Client(server.app) as client:
        resources = await client.list_resources()
        resource_templates = await client.list_resource_templates()
        prompts = await client.list_prompts()
        guide = await client.read_resource("guide://paper-chaser/agent-workflows")
        plan = await client.get_prompt(
            "plan_paper_chaser_search",
            {"topic": "transformers"},
        )
        feature_plan = await client.get_prompt(
            "plan_paper_chaser_search",
            {
                "topic": "transformers",
                "mode": "feature_probe",
                "focus_prompt": "Probe the OpenAlex author workflow UX",
            },
        )

    assert any(str(resource.uri) == "guide://paper-chaser/agent-workflows" for resource in resources)
    assert any(str(template.uriTemplate) == "paper://{paper_id}" for template in resource_templates)
    assert any(str(template.uriTemplate) == "author://{author_id}" for template in resource_templates)
    assert any(str(template.uriTemplate) == "search://{search_session_id}" for template in resource_templates)
    assert any(
        str(template.uriTemplate) == "trail://paper/{paper_id}?direction={direction}" for template in resource_templates
    )
    assert any(prompt.name == "plan_paper_chaser_search" for prompt in prompts)
    assert any(prompt.name == "plan_smart_paper_chaser_search" for prompt in prompts)
    assert any(prompt.name == "triage_literature" for prompt in prompts)
    assert any(prompt.name == "plan_citation_chase" for prompt in prompts)
    assert any(prompt.name == "refine_query" for prompt in prompts)
    assert "research" in guide[0].text
    assert "follow_up_research" in guide[0].text
    assert "resolve_reference" in guide[0].text
    assert "inspect_source" in guide[0].text
    assert "searchSessionId" in guide[0].text
    assert "trust" in guide[0].text.lower()
    assert "research" in plan.messages[0].content.text
    assert "resolve_reference" in plan.messages[0].content.text
    assert "follow_up_research" in plan.messages[0].content.text
    assert "Mode: feature_probe." in feature_plan.messages[0].content.text
    assert "Probe the OpenAlex author workflow UX" in (feature_plan.messages[0].content.text)
    assert "GitHub Copilot coding agent" in feature_plan.messages[0].content.text


def test_tool_descriptions_include_workflow_guidance() -> None:
    from paper_chaser_mcp.tools import TOOL_DESCRIPTIONS

    assert "default guided entry point" in TOOL_DESCRIPTIONS["research"].lower()
    assert "trust-graded" in TOOL_DESCRIPTIONS["research"].lower()
    assert "grounded follow-up" in TOOL_DESCRIPTIONS["follow_up_research"].lower()
    assert "best match" in TOOL_DESCRIPTIONS["resolve_reference"].lower()
    assert "source id" in TOOL_DESCRIPTIONS["inspect_source"].lower()
    assert "runtime" in TOOL_DESCRIPTIONS["get_runtime_status"].lower()
    assert "abstract_inverted_index" in TOOL_DESCRIPTIONS["get_paper_details_openalex"]
    assert "cited_by_api_url" in TOOL_DESCRIPTIONS["get_paper_citations_openalex"]
    assert "batched OpenAlex ID lookups" in TOOL_DESCRIPTIONS["get_paper_references_openalex"]
    assert "OpenAlex author profile" in TOOL_DESCRIPTIONS["get_author_info_openalex"]
    assert "search_authors_openalex" in TOOL_DESCRIPTIONS["get_author_papers_openalex"]


def test_deployment_app_exposes_health_endpoint() -> None:
    from starlette.testclient import TestClient

    from paper_chaser_mcp.deployment import create_deployment_app

    with TestClient(create_deployment_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_deployment_app_rejects_missing_auth_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.testclient import TestClient

    monkeypatch.setenv("PAPER_CHASER_HTTP_AUTH_TOKEN", "super-secret")
    monkeypatch.setenv("PAPER_CHASER_HTTP_PATH", "/mcp")

    import paper_chaser_mcp.deployment as deployment

    importlib.reload(deployment)

    with TestClient(deployment.create_deployment_app()) as client:
        response = client.post("/mcp", json={})

    assert response.status_code == 401


def test_deployment_app_allows_expected_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.testclient import TestClient

    monkeypatch.setenv("PAPER_CHASER_HTTP_AUTH_TOKEN", "super-secret")
    monkeypatch.setenv("PAPER_CHASER_HTTP_PATH", "/mcp")

    import paper_chaser_mcp.deployment as deployment

    importlib.reload(deployment)

    with TestClient(deployment.create_deployment_app()) as client:
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={
                "Authorization": "Bearer super-secret",
                "Accept": "application/json, text/event-stream",
            },
        )

    assert response.status_code != 401


def test_deployment_app_rejects_unlisted_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.testclient import TestClient

    monkeypatch.setenv("PAPER_CHASER_ALLOWED_ORIGINS", "https://allowed.example")

    import paper_chaser_mcp.deployment as deployment

    importlib.reload(deployment)

    with TestClient(deployment.create_deployment_app()) as client:
        response = client.post(
            "/mcp",
            json={},
            headers={"Origin": "https://blocked.example"},
        )

    assert response.status_code == 403


def test_deployment_app_allows_expected_backend_header_and_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.testclient import TestClient

    monkeypatch.setenv("PAPER_CHASER_HTTP_AUTH_TOKEN", "super-secret")
    monkeypatch.setenv("PAPER_CHASER_HTTP_AUTH_HEADER", "x-backend-auth")
    monkeypatch.setenv("PAPER_CHASER_ALLOWED_ORIGINS", "https://allowed.example")

    import paper_chaser_mcp.deployment as deployment

    importlib.reload(deployment)

    with TestClient(deployment.create_deployment_app()) as client:
        response = client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={
                "Accept": "application/json, text/event-stream",
                "Origin": "https://allowed.example",
                "X-Backend-Auth": "super-secret",
            },
        )

    assert response.status_code not in {401, 403}


def test_github_copilot_instructions_align_with_repo_workflow_docs() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    instructions = (repo_root / ".github" / "copilot-instructions.md").read_text()

    assert "GitHub cloud coding agent" in instructions
    assert "docs/golden-paths.md" in instructions
    assert "docs/agent-handoff.md" in instructions
    assert "research` for discovery, literature review, known-item recovery" in instructions
    assert "follow_up_research` for one grounded follow-up" in instructions
    assert "resolve_reference` for DOI/arXiv/URL/citation/reference cleanup" in instructions
    assert "search_papers_bulk` is not a generic" in instructions
    assert "PAPER_CHASER_TOOL_PROFILE=expert" in instructions
    assert "`leads` separate from `evidence`" in instructions
    assert "Legacy `unverifiedLeads` and" in instructions
    assert "verifiedFindings" in instructions
    assert "not a generic" in instructions
    assert "provider-specific tool contracts honest" in instructions
    assert "OpenAlex" in instructions
    assert "python -m pytest" in instructions
    assert "python -m ruff check ." in instructions


def test_server_instructions_surface_continuation_and_schema_cues() -> None:
    instructions = server.SERVER_INSTRUCTIONS
    normalized_instructions = " ".join(instructions.split())

    assert "DEFAULT GUIDED RESEARCH" in instructions
    assert "follow_up_research" in instructions
    assert "resolve_reference" in instructions
    assert "inspect_source" in instructions
    assert "guidedPolicy" in instructions
    assert "search_papers_smart" in instructions
    assert "resolve_citation" in instructions
    assert "ask_result_set" in instructions
    assert "searchSessionId" in instructions
    assert "search_papers_core, search_papers_serpapi, and" in instructions
    assert "only accept query/limit/year" in instructions
    assert "Semantic Scholar pivot rather than another page" in instructions
    assert "NOT relevance-ranked" in instructions
    assert "retrievalNote" in instructions
    assert "citationCount:desc" in instructions
    assert "prefer search_papers or search_papers_semantic_scholar" in instructions
    assert "paper.recommendedExpansionId" in instructions
    assert "paper.expansionIdStatus is not_portable" in instructions
    assert "outside the indexed paper surface" in normalized_instructions
    assert "affiliation, coauthor, venue, or topic clues" in instructions
    assert "*_openalex tools" in instructions
    assert "agentic UX review loops" in instructions
    assert "reproduction-ready issues" in instructions
    assert "evidence/leads/routingSummary/coverageSummary/evidenceGaps" in instructions


@pytest.mark.asyncio
async def test_agent_workflow_resource_mentions_pivots_and_provider_contracts() -> None:
    from fastmcp import Client

    async with Client(server.app) as client:
        guide = await client.read_resource("guide://paper-chaser/agent-workflows")

    guide_text = guide[0].text
    assert "research" in guide_text
    assert "follow_up_research" in guide_text
    assert "resolve_reference" in guide_text
    assert "inspect_source" in guide_text
    assert "get_runtime_status" in guide_text
    assert "searchSessionId" in guide_text
    assert "answerability" in guide_text
    assert "routingSummary" in guide_text
    assert "abstains" in guide_text or "abstain" in guide_text
    assert "Expert/operator-only fallback" in guide_text
    assert "search_papers_smart" in guide_text
    assert "get_provider_diagnostics" in guide_text
    assert "pagination.nextCursor" in guide_text
    assert "code, docs, or both" in guide_text


@pytest.mark.asyncio
async def test_plan_prompt_mentions_continuation_vs_pivot_and_schema_limits() -> None:
    from fastmcp import Client

    async with Client(server.app) as client:
        plan = await client.get_prompt(
            "plan_paper_chaser_search",
            {"topic": "transformers"},
        )

    prompt_text = plan.messages[0].content.text
    assert "research" in prompt_text
    assert "resolve_reference" in prompt_text
    assert "follow_up_research" in prompt_text
    assert "inspect_source" in prompt_text
    assert "Treat abstentions and clarification requests as real outputs" in prompt_text
    assert "Only fall back to the expert surface" in prompt_text
    assert "search_papers_smart" in prompt_text
    assert "searchSessionId" in prompt_text
    assert "search_papers_bulk" in prompt_text
    assert "For regulatory work, prefer the guided path first" in prompt_text
    assert "pagination.nextCursor as opaque" in prompt_text
    assert "Mode: smoke." in prompt_text
    assert "GitHub Copilot coding agent" in prompt_text
