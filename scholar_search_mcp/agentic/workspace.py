"""In-memory search-session registry and lightweight semantic indexing."""

from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("scholar-search-mcp")

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
        author.get("name", "")
        for author in authors
        if isinstance(author, dict) and author.get("name")
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
        index_backend: str = "memory",
        similarity_fn: Callable[[str, str], float] | None = None,
        embed_query_fn: Callable[[str], tuple[float, ...] | None] | None = None,
        embed_texts_fn: Callable[[list[str]], list[tuple[float, ...] | None]]
        | None = None,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._enable_trace_log = enable_trace_log
        self._index_backend = index_backend
        self._similarity_fn = similarity_fn
        self._embed_query_fn = embed_query_fn
        self._embed_texts_fn = embed_texts_fn
        self._records: dict[str, SearchSessionRecord] = {}

    def _cleanup(self) -> None:
        now = _now()
        expired = [
            search_session_id
            for search_session_id, record in self._records.items()
            if record.is_expired(now)
        ]
        for search_session_id in expired:
            self._records.pop(search_session_id, None)

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
        papers = self._extract_papers(payload)
        authors = self._extract_authors(payload)
        record = SearchSessionRecord(
            search_session_id=search_session_id or self._new_search_session_id(),
            source_tool=source_tool,
            created_at=_now(),
            expires_at=_now() + self._ttl_seconds,
            payload=payload,
            query=query,
            metadata=dict(metadata or {}),
            papers=papers,
            authors=authors,
            indexed_papers=[
                IndexedPaper(
                    paper=paper,
                    text=paper_search_text(paper),
                    vector=_vectorize(paper_search_text(paper)),
                )
                for paper in papers
                if paper_search_text(paper)
            ],
        )
        if record.indexed_papers:
            record.vector_store = self._build_vector_store(record.indexed_papers)
        self._records[record.search_session_id] = record
        return record

    def get(self, search_session_id: str) -> SearchSessionRecord:
        """Return the active record for *search_session_id* or raise."""
        self._cleanup()
        record = self._records.get(search_session_id)
        if record is None:
            raise SearchSessionNotFoundError(
                f"Unknown searchSessionId {search_session_id!r}."
            )
        if record.is_expired():
            self._records.pop(search_session_id, None)
            raise ExpiredSearchSessionError(
                f"searchSessionId {search_session_id!r} has expired."
            )
        return record

    def search_papers(
        self,
        search_session_id: str,
        query: str,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant papers within a saved result set."""
        record = self.get(search_session_id)
        if record.vector_store is not None:
            papers = self._search_vector_store(record, query=query, top_k=top_k)
            if papers:
                return papers
        if self._similarity_fn is not None:
            ranked = sorted(
                (
                    (self._similarity_fn(query, item.text), item.paper)
                    for item in record.indexed_papers
                ),
                key=lambda item: item[0],
                reverse=True,
            )
        else:
            query_vector = _vectorize(query)
            ranked = sorted(
                (
                    (_cosine_similarity(query_vector, item.vector), item.paper)
                    for item in record.indexed_papers
                ),
                key=lambda item: item[0],
                reverse=True,
            )
        return [paper for score, paper in ranked[:top_k] if score > 0]

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
        markdown_lines = [
            f"# Search Session {record.search_session_id}",
            "",
            f"- Source tool: `{record.source_tool}`",
            f"- Query: {record.query or 'n/a'}",
            f"- Papers: {len(record.papers)}",
            f"- Authors: {len(record.authors)}",
        ]
        if record.papers:
            markdown_lines.extend(["", "## Top papers"])
            for paper in record.papers[:5]:
                markdown_lines.append(
                    f"- {paper.get('title') or paper.get('paperId') or 'Untitled'}"
                )
        return {
            "markdown": "\n".join(markdown_lines),
            "data": record.payload,
            "metadata": record.metadata,
        }

    def render_paper_resource(self, paper_id: str) -> dict[str, Any] | None:
        """Render one cached paper if it exists in the active sessions."""
        paper = self.find_paper(paper_id)
        if paper is None:
            return None
        markdown_lines = [
            f"# {paper.get('title') or paper.get('paperId') or paper_id}",
            "",
            f"- Paper ID: `{paper.get('paperId') or paper_id}`",
        ]
        if paper.get("year"):
            markdown_lines.append(f"- Year: {paper['year']}")
        if paper.get("venue"):
            markdown_lines.append(f"- Venue: {paper['venue']}")
        if paper.get("authors"):
            author_names = ", ".join(
                author.get("name", "")
                for author in paper["authors"]
                if isinstance(author, dict) and author.get("name")
            )
            if author_names:
                markdown_lines.append(f"- Authors: {author_names}")
        if paper.get("abstract"):
            markdown_lines.extend(["", "## Abstract", "", str(paper["abstract"])])
        return {"markdown": "\n".join(markdown_lines), "data": paper}

    def render_author_resource(self, author_id: str) -> dict[str, Any] | None:
        """Render one cached author if it exists in the active sessions."""
        author = self.find_author(author_id)
        if author is None:
            return None
        markdown_lines = [
            f"# {author.get('name') or author_id}",
            "",
            f"- Author ID: `{author.get('authorId') or author_id}`",
        ]
        if author.get("affiliations"):
            affiliations = ", ".join(author.get("affiliations") or [])
            markdown_lines.append(f"- Affiliations: {affiliations}")
        return {"markdown": "\n".join(markdown_lines), "data": author}

    def _new_search_session_id(self) -> str:
        return f"ssn_{uuid.uuid4().hex[:12]}"

    def _build_vector_store(self, indexed_papers: list[IndexedPaper]) -> Any | None:
        if (
            self._index_backend != "faiss"
            or self._embed_query_fn is None
            or self._embed_texts_fn is None
        ):
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
            ) -> None:
                self._embed_query_fn = embed_query_fn
                self._embed_texts_fn = embed_texts_fn

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                vectors = self._embed_texts_fn(texts)
                normalized_vectors: list[list[float]] = []
                for text, vector in zip(texts, vectors):
                    if vector is None:
                        raise ValueError(
                            f"No embedding vector was generated for {text!r}."
                        )
                    normalized_vectors.append(list(vector))
                return normalized_vectors

            def embed_query(self, text: str) -> list[float]:
                vector = self._embed_query_fn(text)
                if vector is None:
                    raise ValueError(
                        "No embedding vector was generated for the query text."
                    )
                return list(vector)

        documents = [
            Document(page_content=item.text, metadata={"paper": item.paper})
            for item in indexed_papers
        ]
        try:
            return FAISS.from_documents(
                documents,
                _CallableEmbeddings(
                    embed_query_fn=self._embed_query_fn,
                    embed_texts_fn=self._embed_texts_fn,
                ),
            )
        except Exception:
            logger.exception(
                "FAISS index creation failed; falling back to in-memory similarity "
                "scoring."
            )
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
            logger.exception(
                "FAISS similarity search failed; falling back to in-memory "
                "similarity scoring."
            )
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
        for candidate in payload.get("data") or []:
            if isinstance(candidate, dict) and (
                "paperId" in candidate or "title" in candidate
            ):
                papers.append(candidate)
        for candidate in payload.get("results") or []:
            if isinstance(candidate, dict) and isinstance(candidate.get("paper"), dict):
                papers.append(candidate["paper"])
        for candidate in payload.get("representativePapers") or []:
            if isinstance(candidate, dict) and (
                "paperId" in candidate or "title" in candidate
            ):
                papers.append(candidate)
        citation_candidates = [
            payload.get("bestMatch"),
            *(payload.get("alternatives") or []),
        ]
        for candidate in citation_candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("paper"), dict):
                papers.append(candidate["paper"])
        return papers

    def _extract_authors(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        authors: list[dict[str, Any]] = []
        for candidate in payload.get("data") or []:
            if isinstance(candidate, dict) and (
                "authorId" in candidate or "affiliations" in candidate
            ):
                authors.append(candidate)
        return authors
