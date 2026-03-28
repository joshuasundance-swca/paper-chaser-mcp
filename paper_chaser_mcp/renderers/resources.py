"""Shared resource payload renderers for server and workspace resources."""

from __future__ import annotations

from typing import Any


def render_search_resource_payload(record: Any) -> dict[str, Any]:
    """Render one saved search session as a resource payload."""
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
            markdown_lines.append(f"- {paper.get('title') or paper.get('paperId') or 'Untitled'}")
    return {
        "markdown": "\n".join(markdown_lines),
        "data": record.payload,
        "metadata": record.metadata,
    }


def render_paper_resource_payload(
    paper: dict[str, Any],
    *,
    fallback_paper_id: str | None = None,
) -> dict[str, Any]:
    """Render one paper payload as markdown plus its source data."""
    authors = ", ".join(
        author.get("name", "")
        for author in (paper.get("authors") or [])
        if isinstance(author, dict) and author.get("name")
    )
    paper_identifier = paper.get("paperId") or paper.get("canonicalId") or fallback_paper_id or "unknown"
    markdown_lines = [
        f"# {paper.get('title') or paper.get('paperId') or fallback_paper_id or 'Paper'}",
        "",
        f"- Paper ID: `{paper_identifier}`",
    ]
    if paper.get("year"):
        markdown_lines.append(f"- Year: {paper['year']}")
    if paper.get("venue"):
        markdown_lines.append(f"- Venue: {paper['venue']}")
    if authors:
        markdown_lines.append(f"- Authors: {authors}")
    if paper.get("abstract"):
        markdown_lines.extend(["", "## Abstract", "", str(paper["abstract"])])
    return {"markdown": "\n".join(markdown_lines), "data": paper}


def render_author_resource_payload(
    author: dict[str, Any],
    *,
    fallback_author_id: str | None = None,
) -> dict[str, Any]:
    """Render one author payload as markdown plus its source data."""
    author_identifier = author.get("authorId") or fallback_author_id or "unknown"
    markdown_lines = [
        f"# {author.get('name') or author_identifier or 'Author'}",
        "",
        f"- Author ID: `{author_identifier}`",
    ]
    affiliations = author.get("affiliations") or []
    if affiliations:
        markdown_lines.append(f"- Affiliations: {', '.join(affiliations)}")
    return {"markdown": "\n".join(markdown_lines), "data": author}
