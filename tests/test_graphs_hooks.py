"""Phase 7a: hooks submodule identity and behavioural contract tests."""

from __future__ import annotations

import asyncio
import logging

import pytest

from paper_chaser_mcp.agentic.graphs import _core as core_module
from paper_chaser_mcp.agentic.graphs import hooks
from paper_chaser_mcp.agentic.retrieval import RetrievalBatch

_EXTRACTED = (
    "_consume_background_task",
    "_describe_retrieval_batch",
    "_skip_context_notifications",
    "_truncate_text",
)


def test_core_and_submodule_expose_the_same_callables() -> None:
    for name in _EXTRACTED:
        submodule_value = getattr(hooks, name)
        core_value = getattr(core_module, name)
        assert submodule_value is core_value, (
            f"{name}: _core and submodule must share the same object so "
            "legacy monkeypatch and call sites keep working after Phase 7a"
        )


def test_runtime_staticmethods_delegate_to_module_helpers() -> None:
    from paper_chaser_mcp.agentic.graphs._core import AgenticRuntime

    # The class-level static methods should be thin wrappers: calling
    # them directly must produce the same result as the module-level helper.
    class FakeCtx:
        transport = "stdio"

    assert AgenticRuntime._skip_context_notifications(FakeCtx()) is True  # type: ignore[arg-type]
    assert hooks._skip_context_notifications(FakeCtx()) is True  # type: ignore[arg-type]


def test_skip_context_notifications_for_non_stdio() -> None:
    class FakeCtx:
        transport = "streamable-http"

    class NoTransport:
        pass

    assert hooks._skip_context_notifications(FakeCtx()) is False  # type: ignore[arg-type]
    assert hooks._skip_context_notifications(NoTransport()) is False  # type: ignore[arg-type]


def test_truncate_text_collapses_whitespace_and_adds_ellipsis() -> None:
    assert hooks._truncate_text("abc") == "abc"
    assert hooks._truncate_text("  hello   world  ") == "hello world"
    truncated = hooks._truncate_text("x" * 200, limit=10)
    assert truncated.endswith("...")
    assert len(truncated) == 10


def test_describe_retrieval_batch_includes_variant_and_providers() -> None:
    batch = RetrievalBatch(
        variant="alpha",
        variant_source="test",
        candidates=[],
        providers_used=["openalex", "semanticscholar"],
        provider_timings_ms={},
        provider_errors={"core": "boom"},
        provider_outcomes=[],
    )
    message = hooks._describe_retrieval_batch(batch)
    assert "alpha" in message
    assert "openalex" in message
    assert "semanticscholar" in message
    assert "core" in message
    assert "boom" in message


def test_consume_background_task_swallows_cancelled_and_logs_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def cancelled() -> None:
        raise asyncio.CancelledError()

    async def boom() -> None:
        raise RuntimeError("bang")

    loop = asyncio.new_event_loop()
    try:
        cancelled_task = loop.create_task(cancelled())
        boom_task = loop.create_task(boom())
        with caplog.at_level(logging.DEBUG, logger="paper-chaser-mcp"):
            # Let the tasks complete so .result() can be called synchronously.
            loop.run_until_complete(asyncio.gather(cancelled_task, boom_task, return_exceptions=True))
            hooks._consume_background_task(cancelled_task)
            hooks._consume_background_task(boom_task)
    finally:
        loop.close()

    assert any("best-effort context notification" in rec.getMessage().lower() for rec in caplog.records)
