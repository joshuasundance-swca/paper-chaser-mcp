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
