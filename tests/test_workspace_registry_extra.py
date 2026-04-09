import asyncio
import logging
import sys
import types
from typing import Any

import pytest

from paper_chaser_mcp.agentic import workspace as workspace_module
from paper_chaser_mcp.agentic.workspace import (
    ExpiredSearchSessionError,
    IndexedPaper,
    SearchSessionNotFoundError,
    SearchSessionRecord,
    WorkspaceRegistry,
    _cosine_similarity,
    _tokenize,
    _vectorize,
    author_identity_keys,
    paper_identity_keys,
    paper_search_text,
)


def _sample_payload() -> dict[str, Any]:
    return {
        "data": [
            {
                "paperId": "paper-1",
                "sourceId": "source-1",
                "canonicalId": "canonical-1",
                "recommendedExpansionId": "recommended-1",
                "title": "Retrieval Agents for Literature Search",
                "abstract": "Vector retrieval improves grounded answers.",
                "venue": "Journal of Agents",
                "year": 2024,
                "summary": "Compact summary.",
                "authors": [{"name": "Ada Lovelace"}],
            },
            {
                "authorId": "author-1",
                "sourceId": "author-source-1",
                "name": "Grace Hopper",
                "affiliations": ["Navy"],
            },
        ],
        "results": [
            {
                "paper": {
                    "paperId": "paper-2",
                    "title": "Citation Graph Navigation",
                    "abstract": "Citation trails support reviews.",
                    "authors": [{"name": "Grace Hopper"}],
                    "year": 2023,
                }
            }
        ],
        "representativePapers": [
            {
                "paperId": "paper-3",
                "title": "Theme Mapping in Scientific Corpora",
                "abstract": "Theme extraction helps synthesis.",
            }
        ],
        "bestMatch": {
            "paper": {
                "paperId": "paper-4",
                "title": "Recovered Citation Match",
                "abstract": "Recovered from sparse citation text.",
            }
        },
        "alternatives": [
            {
                "paper": {
                    "paperId": "paper-5",
                    "title": "Alternative Citation Candidate",
                    "abstract": "Backup citation candidate.",
                }
            }
        ],
    }


def _sync_embedding(text: str) -> tuple[float, ...] | None:
    normalized = text.lower()
    if not normalized.strip():
        return None
    return (
        float("retrieval" in normalized or "vector" in normalized),
        float("citation" in normalized or "graph" in normalized),
    )


async def _async_embedding(text: str) -> tuple[float, ...] | None:
    return _sync_embedding(text)


def _sync_embeddings(texts: list[str]) -> list[tuple[float, ...] | None]:
    return [_sync_embedding(text) for text in texts]


async def _async_embeddings(texts: list[str]) -> list[tuple[float, ...] | None]:
    return _sync_embeddings(texts)


class _FakeDocument:
    def __init__(self, page_content: str, metadata: dict[str, Any]) -> None:
        self.page_content = page_content
        self.metadata = metadata


class _FakeEmbeddings:
    pass


class _FakeVectorStore:
    def __init__(self, documents: list[_FakeDocument]) -> None:
        self._documents = documents

    def similarity_search(self, query: str, k: int) -> list[Any]:
        del query
        return self._documents[:k]

    async def asimilarity_search(self, query: str, k: int) -> list[Any]:
        del query
        return self._documents[:k]


def _install_fake_langchain_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_sync: bool = False,
    fail_async: bool = False,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _FakeFaiss:
        @staticmethod
        def from_documents(
            documents: list[_FakeDocument],
            embeddings: Any,
        ) -> _FakeVectorStore:
            texts = [document.page_content for document in documents]
            captured["sync_documents"] = texts
            captured["sync_vectors"] = embeddings.embed_documents(texts)
            captured["sync_query"] = embeddings.embed_query("retrieval probe")
            captured["sync_async_vectors"] = asyncio.run(embeddings.aembed_documents(texts))
            captured["sync_async_query"] = asyncio.run(embeddings.aembed_query("retrieval probe"))
            if fail_sync:
                raise RuntimeError("sync faiss failed")
            return _FakeVectorStore(documents)

        @staticmethod
        async def afrom_documents(
            documents: list[_FakeDocument],
            embeddings: Any,
        ) -> _FakeVectorStore:
            texts = [document.page_content for document in documents]
            captured["async_documents"] = texts
            captured["async_sync_vectors"] = embeddings.embed_documents(texts)
            captured["async_sync_query"] = embeddings.embed_query("graph probe")
            captured["async_vectors"] = await embeddings.aembed_documents(texts)
            captured["async_query"] = await embeddings.aembed_query("graph probe")
            if fail_async:
                raise RuntimeError("async faiss failed")
            return _FakeVectorStore(documents)

    langchain_community = types.ModuleType("langchain_community")
    vectorstores = types.ModuleType("langchain_community.vectorstores")
    setattr(vectorstores, "FAISS", _FakeFaiss)
    langchain_community.vectorstores = vectorstores  # type: ignore[attr-defined]

    langchain_core = types.ModuleType("langchain_core")
    documents = types.ModuleType("langchain_core.documents")
    setattr(documents, "Document", _FakeDocument)
    embeddings = types.ModuleType("langchain_core.embeddings")
    setattr(embeddings, "Embeddings", _FakeEmbeddings)
    langchain_core.documents = documents  # type: ignore[attr-defined]
    langchain_core.embeddings = embeddings  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "langchain_community", langchain_community)
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", vectorstores)
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", documents)
    monkeypatch.setitem(sys.modules, "langchain_core.embeddings", embeddings)
    return captured


def test_workspace_registry_renders_resources_and_tracks_entities(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="paper-chaser-mcp")

    registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=True)
    record = registry.save_result_set(
        source_tool="search_papers_smart",
        query="retrieval agents",
        payload=_sample_payload(),
        metadata={"topic": "agents"},
    )
    registry.record_trace(
        record.search_session_id,
        step="rerank",
        payload={"papers": len(record.papers)},
    )

    search_resource = registry.render_search_resource(record.search_session_id)
    paper_resource = registry.render_paper_resource("recommended-1")
    author_resource = registry.render_author_resource("author-source-1")

    older_trail = SearchSessionRecord(
        search_session_id="trail-old",
        source_tool="get_paper_citations",
        created_at=1.0,
        expires_at=9_999_999_999.0,
        payload={},
        metadata={
            "trailParentPaperId": "paper-1",
            "trailDirection": "citations",
        },
    )
    newer_trail = SearchSessionRecord(
        search_session_id="trail-new",
        source_tool="get_paper_citations",
        created_at=2.0,
        expires_at=9_999_999_999.0,
        payload={},
        metadata={
            "trailParentPaperId": "paper-1",
            "trailDirection": "citations",
        },
    )
    registry._store_record(older_trail)
    registry._store_record(newer_trail)

    assert len(record.papers) == 5
    assert len(record.authors) == 1
    assert search_resource["metadata"] == {"topic": "agents"}
    assert "# Search Session" in search_resource["markdown"]
    assert "Retrieval Agents for Literature Search" in search_resource["markdown"]
    assert registry.find_paper("canonical-1") == record.papers[0]
    assert registry.find_author("author-1") == record.authors[0]
    assert paper_resource is not None
    assert "## Abstract" in paper_resource["markdown"]
    assert "Ada Lovelace" in paper_resource["markdown"]
    assert author_resource is not None
    assert "Grace Hopper" in author_resource["markdown"]
    assert "Navy" in author_resource["markdown"]
    assert (
        registry.find_trail(
            paper_id="paper-1",
            direction="citations",
        )
        == newer_trail
    )
    assert any("agentic-trace" in record.getMessage() for record in caplog.records)


def test_workspace_helpers_cleanup_and_search_behaviors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _tokenize("AI-driven Retrieval, 2026!") == [
        "ai",
        "driven",
        "retrieval",
        "2026",
    ]
    assert _vectorize("") == {}
    assert _cosine_similarity({}, {"token": 1.0}) == 0.0
    assert _cosine_similarity(_vectorize("graph retrieval"), _vectorize("retrieval")) > 0
    assert "Ada Lovelace" in paper_search_text(_sample_payload()["data"][0])
    assert paper_identity_keys(_sample_payload()["data"][0]) == {
        "paper-1",
        "source-1",
        "canonical-1",
        "recommended-1",
    }
    assert author_identity_keys(_sample_payload()["data"][1]) == {
        "author-1",
        "author-source-1",
    }

    registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        similarity_fn=lambda query, text: (
            1.0 if "citation graph" in text.lower() and "citation" in query.lower() else 0.0
        ),
    )
    record = registry.save_result_set(
        source_tool="search_papers_smart",
        query="citation graph",
        payload=_sample_payload(),
    )

    assert record.is_expired(now=record.expires_at)
    assert registry.search_papers(record.search_session_id, "citation graph", top_k=1)[0]["paperId"] == "paper-2"

    with pytest.raises(SearchSessionNotFoundError):
        registry.get("missing-session")

    monkeypatch.setattr(registry, "_cleanup", lambda: None)
    record.expires_at = 0.0
    with pytest.raises(ExpiredSearchSessionError):
        registry.get(record.search_session_id)

    cancel_calls: list[str] = []
    monkeypatch.setattr(
        WorkspaceRegistry,
        "_cancel_record_index_task",
        staticmethod(lambda record: cancel_calls.append(record.search_session_id)),
    )

    expired_record = SearchSessionRecord(
        search_session_id="expired-session",
        source_tool="search_papers",
        created_at=1.0,
        expires_at=1.0,
        payload={},
    )
    registry_cleanup = WorkspaceRegistry(ttl_seconds=1, enable_trace_log=False)
    registry_cleanup._records[expired_record.search_session_id] = expired_record
    monkeypatch.setattr(workspace_module, "_now", lambda: 2.0)
    registry_cleanup._cleanup()

    original = SearchSessionRecord(
        search_session_id="replacement",
        source_tool="search_papers",
        created_at=1.0,
        expires_at=100.0,
        payload={},
    )
    replacement = SearchSessionRecord(
        search_session_id="replacement",
        source_tool="search_papers_smart",
        created_at=2.0,
        expires_at=100.0,
        payload={},
    )
    registry_cleanup._store_record(original)
    registry_cleanup._store_record(replacement)

    assert "expired-session" not in registry_cleanup._records
    assert cancel_calls == ["expired-session", "replacement"]


def test_workspace_builds_sync_vector_store_and_handles_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_langchain_modules(monkeypatch)
    registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        index_backend="faiss",
        embed_query_fn=_sync_embedding,
        embed_texts_fn=_sync_embeddings,
        async_embed_query_fn=_async_embedding,
        async_embed_texts_fn=_async_embeddings,
    )
    record = registry.save_result_set(
        source_tool="search_papers_smart",
        payload=_sample_payload(),
    )

    assert registry._can_build_vector_store_sync() is True
    assert registry._can_build_vector_store_async() is True
    assert record.vector_store_status == "ready"
    assert captured["sync_documents"]
    assert captured["sync_vectors"]
    assert captured["sync_query"] == [1.0, 0.0]
    assert registry.search_papers(record.search_session_id, "retrieval", top_k=1)[0]["paperId"] == "paper-1"

    failing_registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        index_backend="faiss",
        embed_query_fn=_sync_embedding,
        embed_texts_fn=lambda texts: [None for _ in texts],
    )
    failed_record = failing_registry.save_result_set(
        source_tool="search_papers_smart",
        payload=_sample_payload(),
    )
    assert failed_record.vector_store is None
    assert failed_record.vector_store_status == "failed"
    assert failed_record.vector_store_error is not None
    assert (
        failing_registry._search_vector_store(
            failed_record,
            query="retrieval",
            top_k=1,
        )
        == []
    )

    class _BrokenStore:
        def similarity_search(self, query: str, k: int) -> list[object]:
            del query, k
            raise RuntimeError("broken search")

    class _MixedStore:
        def similarity_search(self, query: str, k: int) -> list[object]:
            del query, k
            return [
                types.SimpleNamespace(metadata={"paper": {"paperId": "paper-9"}}),
                types.SimpleNamespace(metadata={"paper": "not-a-dict"}),
            ]

    record.vector_store = _MixedStore()
    assert registry._search_vector_store(record, query="retrieval", top_k=2) == [{"paperId": "paper-9"}]

    record.vector_store = _BrokenStore()
    assert registry._search_vector_store(record, query="retrieval", top_k=2) == []


def test_workspace_indexes_guided_evidence_and_relevant_leads_for_search() -> None:
    registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        similarity_fn=lambda query, text: (
            1.0 if "planetary boundaries" in text.lower() and "planetary" in query.lower() else 0.0
        ),
    )
    record = registry.save_result_set(
        source_tool="research",
        query="planetary boundaries",
        payload={
            "evidence": [
                {
                    "evidenceId": "10.1038/461472a",
                    "title": "A safe operating space for humanity",
                    "citation": {
                        "authors": ["Johan Rockstrom"],
                        "year": "2009",
                        "journalOrPublisher": "Nature",
                    },
                    "whyIncluded": "Foundational planetary boundaries framing.",
                    "topicalRelevance": "on_topic",
                }
            ],
            "candidateLeads": [
                {
                    "sourceId": "lead-1",
                    "title": "Planetary boundaries in Earth system governance",
                    "provider": "openalex",
                    "sourceType": "repository_record",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "weak_match",
                    "confidence": "medium",
                }
            ],
            "unverifiedLeads": [
                {
                    "sourceId": "lead-2",
                    "title": "Unrelated freshwater policy record",
                    "provider": "core",
                    "sourceType": "repository_record",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "off_topic",
                    "confidence": "medium",
                }
            ],
        },
    )

    indexed_ids = {paper.get("paperId") for paper in record.papers}

    assert "10.1038/461472a" in indexed_ids
    assert "lead-1" in indexed_ids
    assert "lead-2" not in indexed_ids
    assert registry.search_papers(record.search_session_id, "planetary boundaries", top_k=2)


@pytest.mark.asyncio
async def test_workspace_async_vector_store_paths_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_langchain_modules(monkeypatch)
    registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        index_backend="faiss",
        async_embed_query_fn=_async_embedding,
        async_embed_texts_fn=_async_embeddings,
        embed_query_fn=_sync_embedding,
        embed_texts_fn=_sync_embeddings,
    )
    record = await registry.asave_result_set(
        source_tool="search_papers_smart",
        payload=_sample_payload(),
    )
    assert record.vector_index_task is not None
    await asyncio.wait_for(record.vector_index_task, timeout=1.0)
    assert record.vector_store_status == "ready"
    assert captured["async_vectors"]
    assert captured["async_query"] == [0.0, 1.0]
    assert await registry.asearch_papers(record.search_session_id, "retrieval", top_k=1)
    await registry.aclose()

    no_hook_registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        index_backend="faiss",
    )
    no_hook_record = await no_hook_registry.asave_result_set(
        source_tool="search_papers_smart",
        payload=_sample_payload(),
    )
    assert no_hook_record.vector_store_status == "unavailable"

    record_for_population = no_hook_registry._build_record(
        source_tool="search_papers_smart",
        payload=_sample_payload(),
        query=None,
        metadata=None,
        search_session_id=None,
    )
    assert (
        await no_hook_registry._asearch_vector_store(
            record_for_population,
            query="graph",
            top_k=1,
        )
        == []
    )

    async def _raise_runtime_error(_: list[IndexedPaper]) -> Any:
        raise RuntimeError("async build failed")

    no_hook_registry._abuild_vector_store = _raise_runtime_error  # type: ignore[assignment,method-assign]
    await no_hook_registry._populate_vector_store(record_for_population)
    assert record_for_population.vector_store_status == "failed"
    assert record_for_population.vector_store_error == "async build failed"

    async def _return_none(_: list[IndexedPaper]) -> None:
        return None

    no_hook_registry._abuild_vector_store = _return_none  # type: ignore[assignment,method-assign]
    await no_hook_registry._populate_vector_store(record_for_population)
    assert record_for_population.vector_store_status == "failed"
    assert "FAISS index creation failed" in (record_for_population.vector_store_error or "")

    async def _raise_cancelled(_: list[IndexedPaper]) -> Any:
        raise asyncio.CancelledError

    pending_task = asyncio.create_task(asyncio.sleep(60))
    record_for_population.vector_index_task = pending_task
    no_hook_registry._abuild_vector_store = _raise_cancelled  # type: ignore[assignment,method-assign]
    with pytest.raises(asyncio.CancelledError):
        await no_hook_registry._populate_vector_store(record_for_population)
    assert record_for_population.vector_index_task is None
    pending_task.cancel()
    await asyncio.gather(pending_task, return_exceptions=True)

    class _BrokenAsyncStore:
        async def asimilarity_search(self, query: str, k: int) -> list[object]:
            del query, k
            raise RuntimeError("broken async search")

    class _MixedAsyncStore:
        async def asimilarity_search(self, query: str, k: int) -> list[object]:
            del query, k
            return [
                types.SimpleNamespace(metadata={"paper": {"paperId": "paper-8"}}),
                types.SimpleNamespace(metadata={"paper": None}),
            ]

    record_for_population.vector_store = _MixedAsyncStore()
    assert await no_hook_registry._asearch_vector_store(
        record_for_population,
        query="graph",
        top_k=2,
    ) == [{"paperId": "paper-8"}]

    record_for_population.vector_store = _BrokenAsyncStore()
    assert (
        await no_hook_registry._asearch_vector_store(
            record_for_population,
            query="graph",
            top_k=2,
        )
        == []
    )

    async_similarity_calls: list[tuple[str, list[str]]] = []

    async def _async_similarity(query: str, texts: list[str]) -> list[float]:
        async_similarity_calls.append((query, texts))
        return [0.0, 1.0, 0.0, 0.0, 0.0]

    async_registry = WorkspaceRegistry(
        ttl_seconds=1800,
        enable_trace_log=False,
        async_batched_similarity_fn=_async_similarity,
    )
    async_record = async_registry.save_result_set(
        source_tool="search_papers_smart",
        payload=_sample_payload(),
    )
    results = await async_registry.asearch_papers(
        async_record.search_session_id,
        "citation graph",
        top_k=1,
    )
    assert async_similarity_calls
    assert results[0]["paperId"] == "paper-2"

    close_registry = WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False)
    close_record = SearchSessionRecord(
        search_session_id="close-me",
        source_tool="search_papers",
        created_at=1.0,
        expires_at=100.0,
        payload={},
    )
    close_record.vector_index_task = asyncio.create_task(asyncio.sleep(60))
    close_registry._records[close_record.search_session_id] = close_record
    await close_registry.aclose()
    assert close_record.vector_index_task is not None
    assert close_record.vector_index_task.cancelled() is True


@pytest.mark.asyncio
async def test_workspace_background_task_consumer_handles_errors() -> None:
    async def _cancelled() -> None:
        raise asyncio.CancelledError

    async def _fails() -> None:
        raise RuntimeError("background failure")

    cancelled_task = asyncio.create_task(_cancelled())
    failed_task = asyncio.create_task(_fails())

    for task in (cancelled_task, failed_task):
        try:
            await task
        except BaseException:
            pass

    WorkspaceRegistry._consume_background_task(cancelled_task)
    WorkspaceRegistry._consume_background_task(failed_task)
