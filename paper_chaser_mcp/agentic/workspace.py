"""In-memory search-session registry and lightweight semantic indexing."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..renderers.resources import (
    render_author_resource_payload,
    render_paper_resource_payload,
    render_search_resource_payload,
)

logger = logging.getLogger("paper-chaser-mcp")

TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


class SearchSessionError(ValueError):
    """Base error for search session lookups."""


class SearchSessionNotFoundError(SearchSessionError):
    """Raised when a session handle is unknown."""


class ExpiredSearchSessionError(SearchSessionError):
    """Raised when a session exists but has passed its TTL."""


def _now() -> float:
    return time.time()


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _vectorize(text: str) -> dict[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    length = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return {token: value / length for token, value in counts.items()}


def _cosine_similarity(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(token, 0.0) for token, weight in left.items())


def paper_search_text(paper: dict[str, Any]) -> str:
    """Flatten a paper payload into one indexable string."""
    authors = paper.get("authors") or []
    author_names = ", ".join(
        author.get("name", "") for author in authors if isinstance(author, dict) and author.get("name")
    )
    return " ".join(
        part
        for part in [
            str(paper.get("title") or ""),
            str(paper.get("abstract") or ""),
            str(paper.get("venue") or ""),
            str(paper.get("year") or ""),
            author_names,
            str(paper.get("summary") or ""),
        ]
        if part
    ).strip()


def paper_identity_keys(paper: dict[str, Any]) -> set[str]:
    """Return the lookup keys that can identify a paper across resources."""
    keys: set[str] = set()
    for candidate in (
        paper.get("paperId"),
        paper.get("sourceId"),
        paper.get("canonicalId"),
        paper.get("recommendedExpansionId"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            keys.add(candidate.strip())
    return keys


def _fallback_paper_identity_key(paper: dict[str, Any]) -> str:
    """Return a stable fallback key when a paper lacks portable identifiers."""
    title = re.sub(r"[^a-z0-9]+", " ", str(paper.get("title") or "").lower()).strip()
    year = str(paper.get("year") or "").strip()
    if title:
        return f"title:{title}|year:{year}"
    return ""


def author_identity_keys(author: dict[str, Any]) -> set[str]:
    """Return the lookup keys that can identify an author across resources."""
    keys: set[str] = set()
    for candidate in (author.get("authorId"), author.get("sourceId")):
        if isinstance(candidate, str) and candidate.strip():
            keys.add(candidate.strip())
    return keys


@dataclass
class IndexedPaper:
    """One indexed paper inside a search session."""

    paper: dict[str, Any]
    text: str
    vector: dict[str, float]


@dataclass
class SearchSessionRecord:
    """A reusable result set keyed by searchSessionId."""

    search_session_id: str
    source_tool: str
    created_at: float
    expires_at: float
    payload: dict[str, Any]
    query: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    papers: list[dict[str, Any]] = field(default_factory=list)
    authors: list[dict[str, Any]] = field(default_factory=list)
    indexed_papers: list[IndexedPaper] = field(default_factory=list)
    vector_store: Any | None = None
    vector_store_status: str = "unavailable"
    vector_store_error: str | None = None
    vector_index_task: asyncio.Task[Any] | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    def is_expired(self, now: float | None = None) -> bool:
        return (now or _now()) >= self.expires_at


class WorkspaceRegistry:
    """In-memory registry for reusable search result sets."""

    def __init__(
        self,
        *,
        ttl_seconds: int = 1800,
        enable_trace_log: bool = False,
        eval_trace_path: str | None = None,
        index_backend: str = "memory",
        similarity_fn: Callable[[str, str], float] | None = None,
        async_batched_similarity_fn: Callable[[str, list[str]], Awaitable[list[float]]] | None = None,
        async_embed_query_fn: Callable[[str], Awaitable[tuple[float, ...] | None]] | None = None,
        async_embed_texts_fn: Callable[
            [list[str]],
            Awaitable[list[tuple[float, ...] | None]],
        ]
        | None = None,
        embed_query_fn: Callable[[str], tuple[float, ...] | None] | None = None,
        embed_texts_fn: Callable[[list[str]], list[tuple[float, ...] | None]] | None = None,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._enable_trace_log = enable_trace_log
        self._eval_trace_path = Path(eval_trace_path) if eval_trace_path else None
        self._index_backend = index_backend
        self._similarity_fn = similarity_fn
        self._async_batched_similarity_fn = async_batched_similarity_fn
        self._async_embed_query_fn = async_embed_query_fn
        self._async_embed_texts_fn = async_embed_texts_fn
        self._embed_query_fn = embed_query_fn
        self._embed_texts_fn = embed_texts_fn
        self._records: dict[str, SearchSessionRecord] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()

    def _cleanup(self) -> None:
        now = _now()
        expired = [search_session_id for search_session_id, record in self._records.items() if record.is_expired(now)]
        for search_session_id in expired:
            record = self._records.pop(search_session_id, None)
            if record is not None:
                self._cancel_record_index_task(record)

    def save_result_set(
        self,
        *,
        source_tool: str,
        payload: dict[str, Any],
        query: str | None = None,
        metadata: dict[str, Any] | None = None,
        search_session_id: str | None = None,
    ) -> SearchSessionRecord:
        """Persist a tool result and return the saved search-session record."""
        self._cleanup()
        record = self._build_record(
            source_tool=source_tool,
            payload=payload,
            query=query,
            metadata=metadata,
            search_session_id=search_session_id,
        )
        if record.indexed_papers and self._can_build_vector_store_sync():
            record.vector_store = self._build_vector_store(record.indexed_papers)
            if record.vector_store is not None:
                record.vector_store_status = "ready"
            else:
                record.vector_store_status = "failed"
                record.vector_store_error = "FAISS index creation failed; using in-memory similarity scoring."
        elif record.vector_store_status == "pending":
            record.vector_store_status = "unavailable"
        self._store_record(record)
        return record

    async def asave_result_set(
        self,
        *,
        source_tool: str,
        payload: dict[str, Any],
        query: str | None = None,
        metadata: dict[str, Any] | None = None,
        search_session_id: str | None = None,
    ) -> SearchSessionRecord:
        """Persist a tool result without blocking on background index creation."""
        self._cleanup()
        record = self._build_record(
            source_tool=source_tool,
            payload=payload,
            query=query,
            metadata=metadata,
            search_session_id=search_session_id,
        )
        self._store_record(record)
        if record.indexed_papers and self._can_build_vector_store_async():
            task = asyncio.create_task(self._populate_vector_store(record))
            record.vector_index_task = task
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(self._consume_background_task)
        elif record.vector_store_status == "pending":
            record.vector_store_status = "unavailable"
        return record

    def get(self, search_session_id: str) -> SearchSessionRecord:
        """Return the active record for *search_session_id* or raise."""
        self._cleanup()
        record = self._records.get(search_session_id)
        if record is None:
            raise SearchSessionNotFoundError(f"Unknown searchSessionId {search_session_id!r}.")
        if record.is_expired():
            self._records.pop(search_session_id, None)
            raise ExpiredSearchSessionError(f"searchSessionId {search_session_id!r} has expired.")
        return record

    def latest(self, *, source_tool: str | None = None) -> SearchSessionRecord | None:
        """Return the newest active record, optionally filtered by *source_tool*."""
        self._cleanup()
        candidates = [
            record for record in self._records.values() if source_tool is None or record.source_tool == source_tool
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda record: record.created_at)

    def attach_source_aliases(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a payload copy with stable per-session source aliases attached."""
        return self._attach_source_aliases(payload)

    def active_records(
        self,
        *,
        source_tools: set[str] | None = None,
    ) -> list[SearchSessionRecord]:
        """Return active records ordered from newest to oldest."""
        self._cleanup()
        records = list(self._records.values())
        if source_tools is not None:
            records = [record for record in records if record.source_tool in source_tools]
        return sorted(records, key=lambda record: record.created_at, reverse=True)

    def search_papers(
        self,
        search_session_id: str,
        query: str,
        top_k: int = 8,
        *,
        allow_model_similarity: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant papers within a saved result set."""
        record = self.get(search_session_id)
        if record.vector_store is not None:
            papers = self._search_vector_store(record, query=query, top_k=top_k)
            if papers:
                return papers
        if allow_model_similarity and self._similarity_fn is not None:
            ranked = sorted(
                ((self._similarity_fn(query, item.text), item.paper) for item in record.indexed_papers),
                key=lambda item: item[0],
                reverse=True,
            )
        else:
            query_vector = _vectorize(query)
            ranked = sorted(
                ((_cosine_similarity(query_vector, item.vector), item.paper) for item in record.indexed_papers),
                key=lambda item: item[0],
                reverse=True,
            )
        return [paper for score, paper in ranked[:top_k] if score > 0]

    async def asearch_papers(
        self,
        search_session_id: str,
        query: str,
        top_k: int = 8,
        *,
        allow_model_similarity: bool = True,
    ) -> list[dict[str, Any]]:
        """Async retrieval optimized for batched semantic ranking."""
        record = self.get(search_session_id)
        if record.vector_store is not None:
            papers = await self._asearch_vector_store(record, query=query, top_k=top_k)
            if papers:
                return papers
        if allow_model_similarity and self._async_batched_similarity_fn is not None:
            texts = [item.text for item in record.indexed_papers]
            scores = await self._async_batched_similarity_fn(query, texts)
            ranked = sorted(
                zip(scores, (item.paper for item in record.indexed_papers), strict=False),
                key=lambda item: item[0],
                reverse=True,
            )
            return [paper for score, paper in ranked[:top_k] if score > 0]
        return self.search_papers(
            search_session_id,
            query,
            top_k=top_k,
            allow_model_similarity=allow_model_similarity,
        )

    async def aclose(self) -> None:
        """Cancel any in-flight async index builds owned by the registry."""
        tasks = [
            record.vector_index_task
            for record in self._records.values()
            if (record.vector_index_task is not None and not record.vector_index_task.done())
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()

    def record_trace(
        self,
        search_session_id: str,
        *,
        step: str,
        payload: dict[str, Any],
    ) -> None:
        """Attach a JSON-serializable trace event to a saved result set."""
        record = self.get(search_session_id)
        event = {
            "timestamp": int(_now() * 1000),
            "step": step,
            "payload": payload,
        }
        record.trace.append(event)
        if self._enable_trace_log:
            logger.info("agentic-trace %s", json.dumps(event, sort_keys=True))

    def capture_eval_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        search_session_id: str | None = None,
        run_id: str | None = None,
        batch_id: str | None = None,
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        """Persist a compact eval-candidate event for later review or promotion."""
        event = {
            "eventId": f"evt_{uuid.uuid4().hex[:12]}",
            "timestamp": int(_now() * 1000),
            "eventType": event_type,
            "searchSessionId": search_session_id,
            "runId": run_id,
            "batchId": batch_id,
            "durationMs": duration_ms,
            "payload": payload,
        }
        if search_session_id:
            try:
                self.record_trace(
                    search_session_id,
                    step=f"eval:{event_type}",
                    payload=payload,
                )
            except SearchSessionError:
                pass
        if self._enable_trace_log:
            logger.info("eval-candidate %s", json.dumps(event, sort_keys=True))
        if self._eval_trace_path is not None:
            self._eval_trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._eval_trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        return event

    def find_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Find a paper payload by any known identifier across active sessions."""
        self._cleanup()
        for record in self._records.values():
            for paper in record.papers:
                if paper_id in paper_identity_keys(paper):
                    return paper
        return None

    def find_author(self, author_id: str) -> dict[str, Any] | None:
        """Find an author payload by ID across active sessions."""
        self._cleanup()
        for record in self._records.values():
            for author in record.authors:
                if author_id in author_identity_keys(author):
                    return author
        return None

    def find_trail(
        self,
        *,
        paper_id: str,
        direction: str,
    ) -> SearchSessionRecord | None:
        """Find the most recent saved trail session for a paper/direction pair."""
        self._cleanup()
        candidates = [
            record
            for record in self._records.values()
            if record.metadata.get("trailParentPaperId") == paper_id
            and record.metadata.get("trailDirection") == direction
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda record: record.created_at)

    def render_search_resource(self, search_session_id: str) -> dict[str, Any]:
        """Render one saved search session as a resource payload."""
        record = self.get(search_session_id)
        return render_search_resource_payload(record)

    def render_paper_resource(self, paper_id: str) -> dict[str, Any] | None:
        """Render one cached paper if it exists in the active sessions."""
        paper = self.find_paper(paper_id)
        if paper is None:
            return None
        return render_paper_resource_payload(paper, fallback_paper_id=paper_id)

    def render_author_resource(self, author_id: str) -> dict[str, Any] | None:
        """Render one cached author if it exists in the active sessions."""
        author = self.find_author(author_id)
        if author is None:
            return None
        return render_author_resource_payload(author, fallback_author_id=author_id)

    def _new_search_session_id(self) -> str:
        return f"ssn_{uuid.uuid4().hex[:12]}"

    def _build_record(
        self,
        *,
        source_tool: str,
        payload: dict[str, Any],
        query: str | None,
        metadata: dict[str, Any] | None,
        search_session_id: str | None,
    ) -> SearchSessionRecord:
        normalized_payload = self._attach_source_aliases(payload)
        papers = self._extract_papers(normalized_payload)
        authors = self._extract_authors(normalized_payload)
        created_at = _now()
        record_metadata = dict(metadata or {})
        if alias_map := normalized_payload.get("sessionSourceAliases"):
            record_metadata["sessionSourceAliases"] = alias_map
        for key in ("strategyMetadata", "routingSummary", "resultStatus", "answerability"):
            value = normalized_payload.get(key)
            if value not in (None, "", [], {}):
                record_metadata.setdefault(key, value)
        coverage_summary = normalized_payload.get("coverageSummary") or normalized_payload.get("coverage")
        if isinstance(coverage_summary, dict) and coverage_summary:
            record_metadata.setdefault("coverageSummary", coverage_summary)
        if isinstance(normalized_payload.get("timeline"), dict):
            record_metadata.setdefault("timeline", normalized_payload.get("timeline"))
        record = SearchSessionRecord(
            search_session_id=search_session_id or self._new_search_session_id(),
            source_tool=source_tool,
            created_at=created_at,
            expires_at=created_at + self._ttl_seconds,
            payload=normalized_payload,
            query=query,
            metadata=record_metadata,
            papers=papers,
            authors=authors,
            indexed_papers=[
                IndexedPaper(
                    paper=paper,
                    text=text,
                    vector=_vectorize(text),
                )
                for paper in papers
                if (text := paper_search_text(paper))
            ],
        )
        if self._index_backend == "faiss" and record.indexed_papers:
            record.vector_store_status = "pending"
        return record

    @staticmethod
    def _attach_source_aliases(payload: dict[str, Any]) -> dict[str, Any]:
        normalized_payload = dict(payload)
        alias_map: dict[str, str] = {}
        alias_index = 1
        for key in ("evidence", "leads", "sources", "structuredSources", "candidateLeads", "unverifiedLeads"):
            entries = normalized_payload.get(key)
            if not isinstance(entries, list):
                continue
            updated_entries: list[Any] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    updated_entries.append(entry)
                    continue
                source_id = str(entry.get("sourceId") or entry.get("evidenceId") or "").strip()
                if not source_id:
                    updated_entries.append(entry)
                    continue
                alias = alias_map.get(source_id)
                if alias is None:
                    alias = f"src_{alias_index}"
                    alias_map[source_id] = alias
                    alias_index += 1
                updated_entry = dict(entry)
                if not updated_entry.get("sourceAlias"):
                    updated_entry["sourceAlias"] = alias
                updated_entries.append(updated_entry)
            normalized_payload[key] = updated_entries
        if alias_map:
            normalized_payload["sessionSourceAliases"] = alias_map
        return normalized_payload

    def _store_record(self, record: SearchSessionRecord) -> None:
        existing = self._records.get(record.search_session_id)
        if existing is not None and existing is not record:
            self._cancel_record_index_task(existing)
        self._records[record.search_session_id] = record

    @staticmethod
    def _cancel_record_index_task(record: SearchSessionRecord) -> None:
        task = record.vector_index_task
        if task is not None and not task.done():
            task.cancel()

    @staticmethod
    def _consume_background_task(task: asyncio.Task[Any]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.debug("Background vector index build failed.", exc_info=True)

    def _can_build_vector_store_sync(self) -> bool:
        return self._index_backend == "faiss" and self._embed_query_fn is not None and self._embed_texts_fn is not None

    def _can_build_vector_store_async(self) -> bool:
        return self._index_backend == "faiss" and (
            (self._async_embed_query_fn is not None and self._async_embed_texts_fn is not None)
            or self._can_build_vector_store_sync()
        )

    async def _populate_vector_store(self, record: SearchSessionRecord) -> None:
        try:
            vector_store = await self._abuild_vector_store(record.indexed_papers)
        except asyncio.CancelledError:
            record.vector_index_task = None
            raise
        except Exception as error:
            record.vector_store = None
            record.vector_store_status = "failed"
            record.vector_store_error = str(error)
            record.vector_index_task = None
            logger.exception("Async FAISS index creation failed; falling back to in-memory similarity scoring.")
            return

        record.vector_store = vector_store
        record.vector_index_task = None
        if vector_store is not None:
            record.vector_store_status = "ready"
            record.vector_store_error = None
            return
        record.vector_store_status = "failed"
        record.vector_store_error = "FAISS index creation failed; using in-memory similarity scoring."

    def _build_vector_store(self, indexed_papers: list[IndexedPaper]) -> Any | None:
        if self._index_backend != "faiss" or self._embed_query_fn is None or self._embed_texts_fn is None:
            return None
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document
            from langchain_core.embeddings import Embeddings
        except ImportError:
            logger.info(
                "FAISS backend requested but optional ai-faiss extras are not "
                "installed; falling back to in-memory similarity scoring."
            )
            return None

        class _CallableEmbeddings(Embeddings):
            def __init__(
                self,
                *,
                embed_query_fn: Callable[[str], tuple[float, ...] | None],
                embed_texts_fn: Callable[[list[str]], list[tuple[float, ...] | None]],
                async_embed_query_fn: Callable[
                    [str],
                    Awaitable[tuple[float, ...] | None],
                ]
                | None = None,
                async_embed_texts_fn: Callable[
                    [list[str]],
                    Awaitable[list[tuple[float, ...] | None]],
                ]
                | None = None,
            ) -> None:
                self._embed_query_fn = embed_query_fn
                self._embed_texts_fn = embed_texts_fn
                self._async_embed_query_fn = async_embed_query_fn
                self._async_embed_texts_fn = async_embed_texts_fn

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                vectors = self._embed_texts_fn(texts)
                normalized_vectors: list[list[float]] = []
                for text, vector in zip(texts, vectors):
                    if vector is None:
                        raise ValueError(f"No embedding vector was generated for {text!r}.")
                    normalized_vectors.append(list(vector))
                return normalized_vectors

            def embed_query(self, text: str) -> list[float]:
                vector = self._embed_query_fn(text)
                if vector is None:
                    raise ValueError("No embedding vector was generated for the query text.")
                return list(vector)

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                if self._async_embed_texts_fn is not None:
                    vectors = await self._async_embed_texts_fn(texts)
                    normalized_vectors: list[list[float]] = []
                    for text, vector in zip(texts, vectors, strict=False):
                        if vector is None:
                            raise ValueError(f"No embedding vector was generated for {text!r}.")
                        normalized_vectors.append(list(vector))
                    return normalized_vectors
                return self.embed_documents(texts)

            async def aembed_query(self, text: str) -> list[float]:
                if self._async_embed_query_fn is not None:
                    vector = await self._async_embed_query_fn(text)
                    if vector is None:
                        raise ValueError("No embedding vector was generated for the query text.")
                    return list(vector)
                return self.embed_query(text)

        documents = [Document(page_content=item.text, metadata={"paper": item.paper}) for item in indexed_papers]
        try:
            return FAISS.from_documents(
                documents,
                _CallableEmbeddings(
                    embed_query_fn=self._embed_query_fn,
                    embed_texts_fn=self._embed_texts_fn,
                    async_embed_query_fn=self._async_embed_query_fn,
                    async_embed_texts_fn=self._async_embed_texts_fn,
                ),
            )
        except Exception:
            logger.exception("FAISS index creation failed; falling back to in-memory similarity scoring.")
            return None

    async def _abuild_vector_store(self, indexed_papers: list[IndexedPaper]) -> Any | None:
        if self._index_backend != "faiss":
            return None
        if (
            self._async_embed_query_fn is None or self._async_embed_texts_fn is None
        ) and not self._can_build_vector_store_sync():
            return None
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document
            from langchain_core.embeddings import Embeddings
        except ImportError:
            logger.info(
                "FAISS backend requested but optional ai-faiss extras are not "
                "installed; falling back to in-memory similarity scoring."
            )
            return None

        class _CallableEmbeddings(Embeddings):
            def __init__(
                self,
                *,
                embed_query_fn: Callable[[str], tuple[float, ...] | None] | None,
                embed_texts_fn: Callable[
                    [list[str]],
                    list[tuple[float, ...] | None],
                ]
                | None,
                async_embed_query_fn: Callable[
                    [str],
                    Awaitable[tuple[float, ...] | None],
                ]
                | None,
                async_embed_texts_fn: Callable[
                    [list[str]],
                    Awaitable[list[tuple[float, ...] | None]],
                ]
                | None,
            ) -> None:
                self._embed_query_fn = embed_query_fn
                self._embed_texts_fn = embed_texts_fn
                self._async_embed_query_fn = async_embed_query_fn
                self._async_embed_texts_fn = async_embed_texts_fn

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                if self._embed_texts_fn is None:
                    raise ValueError("No sync embedding hook is configured.")
                vectors = self._embed_texts_fn(texts)
                normalized_vectors: list[list[float]] = []
                for text, vector in zip(texts, vectors, strict=False):
                    if vector is None:
                        raise ValueError(f"No embedding vector was generated for {text!r}.")
                    normalized_vectors.append(list(vector))
                return normalized_vectors

            def embed_query(self, text: str) -> list[float]:
                if self._embed_query_fn is None:
                    raise ValueError("No sync query embedding hook is configured.")
                vector = self._embed_query_fn(text)
                if vector is None:
                    raise ValueError("No embedding vector was generated for the query text.")
                return list(vector)

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                if self._async_embed_texts_fn is not None:
                    vectors = await self._async_embed_texts_fn(texts)
                    normalized_vectors: list[list[float]] = []
                    for text, vector in zip(texts, vectors, strict=False):
                        if vector is None:
                            raise ValueError(f"No embedding vector was generated for {text!r}.")
                        normalized_vectors.append(list(vector))
                    return normalized_vectors
                return self.embed_documents(texts)

            async def aembed_query(self, text: str) -> list[float]:
                if self._async_embed_query_fn is not None:
                    vector = await self._async_embed_query_fn(text)
                    if vector is None:
                        raise ValueError("No embedding vector was generated for the query text.")
                    return list(vector)
                return self.embed_query(text)

        documents = [Document(page_content=item.text, metadata={"paper": item.paper}) for item in indexed_papers]
        try:
            return await FAISS.afrom_documents(
                documents,
                _CallableEmbeddings(
                    embed_query_fn=self._embed_query_fn,
                    embed_texts_fn=self._embed_texts_fn,
                    async_embed_query_fn=self._async_embed_query_fn,
                    async_embed_texts_fn=self._async_embed_texts_fn,
                ),
            )
        except Exception:
            logger.exception("Async FAISS index creation failed; falling back to in-memory similarity scoring.")
            return None

    def _search_vector_store(
        self,
        record: SearchSessionRecord,
        *,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        vector_store = record.vector_store
        if vector_store is None:
            return []
        try:
            documents = vector_store.similarity_search(query, k=top_k)
        except Exception:
            logger.exception("FAISS similarity search failed; falling back to in-memory similarity scoring.")
            return []

        papers: list[dict[str, Any]] = []
        for document in documents:
            metadata = getattr(document, "metadata", {})
            paper = metadata.get("paper")
            if isinstance(paper, dict):
                papers.append(paper)
        return papers

    async def _asearch_vector_store(
        self,
        record: SearchSessionRecord,
        *,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        vector_store = record.vector_store
        if vector_store is None:
            return []
        try:
            documents = await vector_store.asimilarity_search(query, k=top_k)
        except Exception:
            logger.exception("Async FAISS similarity search failed; falling back to in-memory similarity scoring.")
            return []

        papers: list[dict[str, Any]] = []
        for document in documents:
            metadata = getattr(document, "metadata", {})
            paper = metadata.get("paper")
            if isinstance(paper, dict):
                papers.append(paper)
        return papers

    def _extract_papers(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        papers: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        def add_paper(candidate: dict[str, Any]) -> None:
            identity_keys = paper_identity_keys(candidate)
            fallback_key = _fallback_paper_identity_key(candidate)
            dedupe_keys = identity_keys or ({fallback_key} if fallback_key else set())
            if not dedupe_keys or dedupe_keys & seen_keys:
                return
            seen_keys.update(dedupe_keys)
            papers.append(candidate)

        def year_from_source(candidate: dict[str, Any]) -> int | None:
            for value in (
                candidate.get("year"),
                (candidate.get("citation") or {}).get("year") if isinstance(candidate.get("citation"), dict) else None,
                candidate.get("date"),
            ):
                text = str(value or "").strip()
                if not text:
                    continue
                match = re.search(r"\b(19|20)\d{2}\b", text)
                if match:
                    try:
                        return int(match.group(0))
                    except ValueError:
                        continue
            return None

        def authors_from_source(candidate: dict[str, Any]) -> list[dict[str, Any]]:
            citation = candidate.get("citation")
            if not isinstance(citation, dict):
                return []
            authors = citation.get("authors")
            if not isinstance(authors, list):
                return []
            normalized: list[dict[str, Any]] = []
            for author in authors:
                if isinstance(author, dict):
                    name = str(author.get("name") or "").strip()
                    if name:
                        normalized.append({"name": name})
                elif isinstance(author, str):
                    name = author.strip()
                    if name:
                        normalized.append({"name": name})
            return normalized

        def paper_from_source(candidate: dict[str, Any]) -> dict[str, Any] | None:
            source_id = str(candidate.get("sourceId") or candidate.get("evidenceId") or "").strip()
            title = str(candidate.get("title") or candidate.get("citationText") or "").strip()
            if not source_id and not title:
                return None
            citation_raw = candidate.get("citation")
            citation: dict[str, Any] = citation_raw if isinstance(citation_raw, dict) else {}
            summary = str(
                candidate.get("summary")
                or candidate.get("note")
                or candidate.get("whyIncluded")
                or candidate.get("whyRelevant")
                or candidate.get("whyNotVerified")
                or ""
            ).strip()
            paper: dict[str, Any] = {
                "paperId": source_id or title,
                "sourceId": source_id or title,
                "canonicalId": source_id or title,
                "title": title or source_id,
                "abstract": str(candidate.get("excerpt") or candidate.get("abstract") or "").strip() or None,
                "summary": summary or None,
                "venue": str(
                    citation.get("journalOrPublisher")
                    or candidate.get("venue")
                    or candidate.get("provider")
                    or candidate.get("source")
                    or ""
                ).strip()
                or None,
                "year": year_from_source(candidate),
                "authors": authors_from_source(candidate),
                "source": str(candidate.get("provider") or candidate.get("source") or "").strip() or None,
                "canonicalUrl": str(candidate.get("canonicalUrl") or "").strip() or None,
                "retrievedUrl": str(candidate.get("retrievedUrl") or "").strip() or None,
                "sourceType": str(candidate.get("sourceType") or "").strip() or None,
                "verificationStatus": str(candidate.get("verificationStatus") or "").strip() or None,
                "accessStatus": str(candidate.get("accessStatus") or "").strip() or None,
                "topicalRelevance": str(candidate.get("topicalRelevance") or "").strip() or None,
                "confidence": str(candidate.get("confidence") or "").strip() or None,
            }
            return {key: value for key, value in paper.items() if value not in (None, "", [], {})}

        for candidate in payload.get("data") or []:
            if isinstance(candidate, dict) and ("paperId" in candidate or "title" in candidate):
                add_paper(candidate)
        for candidate in payload.get("results") or []:
            if isinstance(candidate, dict) and isinstance(candidate.get("paper"), dict):
                add_paper(candidate["paper"])
        for candidate in payload.get("representativePapers") or []:
            if isinstance(candidate, dict) and ("paperId" in candidate or "title" in candidate):
                add_paper(candidate)
        citation_candidates = [
            payload.get("bestMatch"),
            *(payload.get("alternatives") or []),
        ]
        for candidate in citation_candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("paper"), dict):
                add_paper(candidate["paper"])
        for key in ("evidence", "sources", "structuredSources", "leads", "candidateLeads", "unverifiedLeads"):
            for candidate in payload.get(key) or []:
                if not isinstance(candidate, dict):
                    continue
                if str(candidate.get("topicalRelevance") or "").strip().lower() == "off_topic":
                    continue
                paper = paper_from_source(candidate)
                if paper is not None:
                    add_paper(paper)
        return papers

    def _extract_authors(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        authors: list[dict[str, Any]] = []
        for candidate in payload.get("data") or []:
            if isinstance(candidate, dict) and ("authorId" in candidate or "affiliations" in candidate):
                authors.append(candidate)
        return authors
