import asyncio

import pytest

from scholar_search_mcp.agentic.workspace import WorkspaceRegistry


def _embedding(text: str) -> tuple[float, ...]:
    lowered = text.lower()
    return (
        float("retrieval" in lowered or "vector" in lowered),
        float("citation" in lowered or "graph" in lowered),
        float("biomedical" in lowered or "medicine" in lowered),
    )


def test_workspace_registry_uses_faiss_when_enabled() -> None:
    pytest.importorskip("faiss")
    pytest.importorskip("langchain_community.vectorstores")

    registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        index_backend="faiss",
        embed_query_fn=_embedding,
        embed_texts_fn=lambda texts: [_embedding(text) for text in texts],
    )

    record = registry.save_result_set(
        source_tool="search_papers_smart",
        query="retrieval agents",
        payload={
            "results": [
                {
                    "paper": {
                        "paperId": "paper-1",
                        "title": "Retrieval-Augmented Agents for Scientific Search",
                        "abstract": "Vector retrieval improves agent workflows.",
                        "authors": [{"name": "Ada Lovelace"}],
                        "year": 2025,
                    }
                },
                {
                    "paper": {
                        "paperId": "paper-2",
                        "title": "Citation Graph Analysis for Literature Review",
                        "abstract": "Graph traversal supports citation chasing.",
                        "authors": [{"name": "Grace Hopper"}],
                        "year": 2024,
                    }
                },
            ]
        },
    )

    assert record.vector_store is not None

    papers = registry.search_papers(
        record.search_session_id,
        "vector retrieval",
        top_k=1,
    )

    assert papers
    assert papers[0]["paperId"] == "paper-1"


@pytest.mark.asyncio
async def test_workspace_registry_async_search_uses_batched_similarity() -> None:
    calls: list[tuple[str, list[str]]] = []

    async def _batched_similarity(query: str, texts: list[str]) -> list[float]:
        calls.append((query, texts))
        return [0.95, 0.05]

    registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        async_batched_similarity_fn=_batched_similarity,
    )

    record = registry.save_result_set(
        source_tool="search_papers_smart",
        query="retrieval agents",
        payload={
            "results": [
                {
                    "paper": {
                        "paperId": "paper-1",
                        "title": "Retrieval-Augmented Agents for Scientific Search",
                        "abstract": "Vector retrieval improves agent workflows.",
                    }
                },
                {
                    "paper": {
                        "paperId": "paper-2",
                        "title": "Citation Graph Analysis for Literature Review",
                        "abstract": "Graph traversal supports citation chasing.",
                    }
                },
            ]
        },
    )

    papers = await registry.asearch_papers(
        record.search_session_id,
        "vector retrieval",
        top_k=1,
    )

    assert calls
    assert papers
    assert papers[0]["paperId"] == "paper-1"


@pytest.mark.asyncio
async def test_workspace_registry_async_save_builds_vector_store() -> None:
    fallback_calls: list[tuple[str, list[str]]] = []
    vector_calls: list[tuple[str, int]] = []
    build_started = asyncio.Event()
    release_build = asyncio.Event()

    async def _batched_similarity(query: str, texts: list[str]) -> list[float]:
        fallback_calls.append((query, texts))
        return [0.05, 0.95]

    async def _embed_query(_: str) -> tuple[float, ...]:
        return (1.0, 0.0)

    async def _embed_texts(
        texts: list[str],
    ) -> list[tuple[float, ...] | None]:
        return [(1.0, 0.0) for _ in texts]

    class _Document:
        def __init__(self, paper: dict[str, object]) -> None:
            self.metadata = {"paper": paper}

    class _VectorStore:
        def __init__(self, paper: dict[str, object]) -> None:
            self._paper = paper

        async def asimilarity_search(self, query: str, k: int) -> list[_Document]:
            vector_calls.append((query, k))
            return [_Document(self._paper)]

    registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        index_backend="faiss",
        async_batched_similarity_fn=_batched_similarity,
        async_embed_query_fn=_embed_query,
        async_embed_texts_fn=_embed_texts,
    )

    async def _fake_build(indexed_papers):
        build_started.set()
        await release_build.wait()
        return _VectorStore(indexed_papers[0].paper)

    registry._abuild_vector_store = _fake_build  # type: ignore[method-assign]

    record = await registry.asave_result_set(
        source_tool="search_papers_smart",
        query="retrieval agents",
        payload={
            "results": [
                {
                    "paper": {
                        "paperId": "paper-1",
                        "title": "Retrieval-Augmented Agents for Scientific Search",
                        "abstract": "Vector retrieval improves agent workflows.",
                    }
                },
                {
                    "paper": {
                        "paperId": "paper-2",
                        "title": "Citation Graph Analysis for Literature Review",
                        "abstract": "Graph traversal supports citation chasing.",
                    }
                },
            ]
        },
    )

    await asyncio.wait_for(build_started.wait(), timeout=1.0)
    assert record.vector_store_status == "pending"
    assert record.vector_store is None

    before_ready = await registry.asearch_papers(
        record.search_session_id,
        "graph traversal",
        top_k=1,
    )

    assert fallback_calls
    assert before_ready[0]["paperId"] == "paper-2"

    release_build.set()
    assert record.vector_index_task is not None
    await asyncio.wait_for(record.vector_index_task, timeout=1.0)

    assert record.vector_store_status == "ready"
    assert record.vector_store is not None

    after_ready = await registry.asearch_papers(
        record.search_session_id,
        "graph traversal",
        top_k=1,
    )

    assert vector_calls == [("graph traversal", 1)]
    assert after_ready[0]["paperId"] == "paper-1"
    await registry.aclose()
