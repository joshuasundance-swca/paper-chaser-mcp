"""Paper Chaser resource registrations (decorated at package import time)."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Literal

from ..models import dump_jsonable
from ..renderers.resources import (
    render_author_resource_payload,
    render_paper_resource_payload,
)


def _resource_text(payload: dict[str, Any]) -> str:
    return json.dumps(dump_jsonable(payload), ensure_ascii=False, indent=2)


def _paper_resource_payload(paper: dict[str, Any]) -> dict[str, Any]:
    return render_paper_resource_payload(paper)


def _author_resource_payload(author: dict[str, Any]) -> dict[str, Any]:
    return render_author_resource_payload(author)


def register_resources(
    app: Any,
    *,
    require_workspace_registry: Callable[[], Any],
    require_semantic_client: Callable[[], Any],
    require_openalex_client: Callable[[], Any],
    agent_workflow_guide: str,
    logger: logging.Logger,
) -> dict[str, Callable[..., Any]]:
    """Attach Paper Chaser resource handlers to ``app`` and return the decorated callables."""

    @app.resource(
        "guide://paper-chaser/agent-workflows",
        title="Paper Chaser agent workflows",
        description="How to choose the right Paper Chaser tools and pagination flow.",
    )
    def agent_workflows() -> str:
        """Return a compact workflow guide for agents."""
        return agent_workflow_guide

    @app.resource(
        "paper://{paper_id}",
        title="Paper resource",
        description="Compact cached or fetched paper payload plus markdown summary.",
        mime_type="application/json",
    )
    async def paper_resource(paper_id: str) -> str:
        registry = require_workspace_registry()
        semantic_client = require_semantic_client()
        oa_client = require_openalex_client()
        cached = registry.render_paper_resource(paper_id)
        if cached is not None:
            return _resource_text(cached)
        last_error: Exception | None = None
        for fetch in (
            lambda: semantic_client.get_paper_details(paper_id),
            lambda: oa_client.get_paper_details(paper_id),
        ):
            try:
                paper = await fetch()
                return _resource_text(_paper_resource_payload(paper))
            except Exception as exc:
                last_error = exc
                logger.debug("Paper resource fetch failed for %r: %s", paper_id, exc)
        raise ValueError(f"Could not resolve paper resource for {paper_id!r}.") from last_error

    @app.resource(
        "author://{author_id}",
        title="Author resource",
        description="Compact cached or fetched author payload plus markdown summary.",
        mime_type="application/json",
    )
    async def author_resource(author_id: str) -> str:
        registry = require_workspace_registry()
        semantic_client = require_semantic_client()
        oa_client = require_openalex_client()
        cached = registry.render_author_resource(author_id)
        if cached is not None:
            return _resource_text(cached)
        last_error: Exception | None = None
        for fetch in (
            lambda: semantic_client.get_author_info(author_id),
            lambda: oa_client.get_author_info(author_id),
        ):
            try:
                author = await fetch()
                return _resource_text(_author_resource_payload(author))
            except Exception as exc:
                last_error = exc
                logger.debug("Author resource fetch failed for %r: %s", author_id, exc)
        raise ValueError(f"Could not resolve author resource for {author_id!r}.") from last_error

    @app.resource(
        "search://{search_session_id}",
        title="Search session resource",
        description="Saved result-set handle surfaced from tool outputs.",
        mime_type="application/json",
    )
    def search_session_resource(search_session_id: str) -> str:
        registry = require_workspace_registry()
        return _resource_text(registry.render_search_resource(search_session_id))

    @app.resource(
        "trail://paper/{paper_id}?direction={direction}",
        title="Paper trail resource",
        description=("Citation or reference trail for a paper, preferably discovered through tool outputs."),
        mime_type="application/json",
    )
    async def paper_trail_resource(
        paper_id: str,
        direction: Literal["citations", "references"],
    ) -> str:
        registry = require_workspace_registry()
        semantic_client = require_semantic_client()
        cached_trail = registry.find_trail(paper_id=paper_id, direction=direction)
        if cached_trail is not None:
            return _resource_text(registry.render_search_resource(cached_trail.search_session_id))
        payload = await (
            semantic_client.get_paper_citations(paper_id=paper_id, limit=25, fields=None, offset=None)
            if direction == "citations"
            else semantic_client.get_paper_references(
                paper_id=paper_id,
                limit=25,
                fields=None,
                offset=None,
            )
        )
        title = "Citations" if direction == "citations" else "References"
        summary = {
            "markdown": (
                f"# {title} trail for `{paper_id}`\n\n"
                f"- Direction: {direction}\n"
                f"- Results: {len((payload or {}).get('data') or [])}"
            ),
            "data": payload,
        }
        return _resource_text(summary)

    return {
        "agent_workflows": agent_workflows,
        "paper_resource": paper_resource,
        "author_resource": author_resource,
        "search_session_resource": search_session_resource,
        "paper_trail_resource": paper_trail_resource,
    }


__all__ = [
    "_author_resource_payload",
    "_paper_resource_payload",
    "_resource_text",
    "register_resources",
]
