"""Identifier helpers for provider bundles."""

from __future__ import annotations

from typing import Any


def relevance_paper_identifier(paper: dict[str, Any], index: int | None = None) -> str:
    return str(
        paper.get("paperId")
        or paper.get("paper_id")
        or paper.get("canonicalId")
        or paper.get("sourceId")
        or (f"paper-{index}" if index is not None else "")
    ).strip()


def _default_selected_evidence_ids(evidence_papers: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for paper in evidence_papers[:limit]:
        if not isinstance(paper, dict):
            continue
        for key in ("paperId", "sourceId", "canonicalId"):
            value = str(paper.get(key) or "").strip()
            if value and value not in seen:
                seen.add(value)
                selected.append(value)
                break
    return selected
