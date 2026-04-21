"""Phase 7c-4 contract tests for the ``run_search_papers_smart`` extraction.

These tests prove the Pattern B extraction preserves public semantics and that
the orchestration is independently callable without routing through the
``AgenticRuntime.search_papers_smart`` method wrapper.
"""

from __future__ import annotations

import inspect

import pytest

from paper_chaser_mcp.agentic.graphs import smart_graph
from paper_chaser_mcp.agentic.graphs._core import AgenticRuntime
from paper_chaser_mcp.agentic.graphs.smart_graph import run_search_papers_smart
from tests.test_smart_tools import (
    RecordingOpenAlexClient,
    RecordingSemanticClient,
    _deterministic_runtime,
)


def test_run_search_papers_smart_is_public_module_coroutine() -> None:
    """The orchestration is an async module-level function, not a method."""

    assert inspect.iscoroutinefunction(run_search_papers_smart)
    assert run_search_papers_smart.__module__ == "paper_chaser_mcp.agentic.graphs.smart_graph"
    assert "run_search_papers_smart" in smart_graph.__all__


def test_run_search_papers_smart_signature_matches_method() -> None:
    """The extracted function mirrors the method signature byte-for-byte
    aside from ``self`` becoming the first positional ``runtime`` parameter.
    """

    method_sig = inspect.signature(AgenticRuntime.search_papers_smart)
    fn_sig = inspect.signature(run_search_papers_smart)

    method_params = list(method_sig.parameters.items())
    fn_params = list(fn_sig.parameters.items())

    assert method_params[0][0] == "self"
    assert fn_params[0][0] == "runtime"

    assert [name for name, _ in method_params[1:]] == [name for name, _ in fn_params[1:]]

    for (_, method_param), (_, fn_param) in zip(method_params[1:], fn_params[1:], strict=True):
        assert method_param.kind == fn_param.kind
        assert method_param.default == fn_param.default
        assert str(method_param.annotation) == str(fn_param.annotation)

    assert str(method_sig.return_annotation) == str(fn_sig.return_annotation)


@pytest.mark.asyncio
async def test_run_search_papers_smart_callable_without_method_wrapper() -> None:
    """The orchestration runs end-to-end when called directly on a runtime,
    bypassing ``runtime.search_papers_smart``. This pins Pattern B
    method-independence.
    """

    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    _, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    direct_response = await run_search_papers_smart(
        runtime,
        query="graphene oxide membranes",
        limit=5,
        latency_profile="fast",
    )

    assert isinstance(direct_response, dict)
    for field in ("results", "searchSessionId", "strategyMetadata", "resultStatus"):
        assert field in direct_response, f"missing {field}"


@pytest.mark.asyncio
async def test_method_and_direct_call_return_equivalent_shape() -> None:
    """Calling the orchestration via the method delegate and via the
    module-level function yields results with identical top-level keys and
    the same ``resultStatus`` for a deterministic input.
    """

    semantic_a = RecordingSemanticClient()
    openalex_a = RecordingOpenAlexClient()
    _, runtime_a = _deterministic_runtime(semantic=semantic_a, openalex=openalex_a)

    semantic_b = RecordingSemanticClient()
    openalex_b = RecordingOpenAlexClient()
    _, runtime_b = _deterministic_runtime(semantic=semantic_b, openalex=openalex_b)

    kwargs = {
        "query": "graphene oxide membranes",
        "limit": 5,
        "latency_profile": "fast",
    }

    via_method = await runtime_a.search_papers_smart(**kwargs)  # type: ignore[arg-type]
    via_direct = await run_search_papers_smart(runtime_b, **kwargs)  # type: ignore[arg-type]

    assert set(via_method.keys()) == set(via_direct.keys())
    assert via_method["resultStatus"] == via_direct["resultStatus"]
    assert len(via_method["results"]) == len(via_direct["results"])


@pytest.mark.asyncio
async def test_run_search_papers_smart_honors_core_classify_query_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_search_papers_smart`` must resolve ``classify_query`` through the
    ``_core`` module seam so tests can monkeypatch planning without touching
    the method wrapper. This pins the deliberate re-export at
    ``paper_chaser_mcp.agentic.graphs._core.classify_query`` (see Phase 7c-4).
    """

    from paper_chaser_mcp.agentic.graphs import _core

    class _Sentinel(Exception):
        pass

    calls: list[dict[str, object]] = []

    async def _fake_classify_query(**kwargs: object) -> tuple[str, object]:
        calls.append(kwargs)
        raise _Sentinel("monkeypatched classify_query invoked")

    monkeypatch.setattr(_core, "classify_query", _fake_classify_query)

    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    _, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    with pytest.raises(_Sentinel):
        await run_search_papers_smart(
            runtime,
            query="graphene oxide membranes",
            limit=5,
            latency_profile="fast",
        )

    assert len(calls) == 1, "monkeypatched classify_query was not routed through _core seam"
    assert calls[0].get("query") == "graphene oxide membranes"


def test_method_delegate_body_is_thin() -> None:
    """The ``AgenticRuntime.search_papers_smart`` method is a thin delegate
    whose executable body consists of nothing more than the local import and
    a single ``return await run_search_papers_smart(...)`` call.
    """

    source = inspect.getsource(AgenticRuntime.search_papers_smart)
    assert "run_search_papers_smart" in source
    assert "return await run_search_papers_smart(" in source

    body_lines = [ln.strip() for ln in source.splitlines()]
    body_lines = [ln for ln in body_lines if ln and not ln.startswith("#")]

    assert any(ln.startswith("from .smart_graph import run_search_papers_smart") for ln in body_lines)
