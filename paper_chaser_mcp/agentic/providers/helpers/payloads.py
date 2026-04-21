"""Payload builders and sanitizers for smart-layer provider adapters."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ...models import ExpansionCandidate
from .nlp import COMMON_QUERY_WORDS, _tokenize

_LITERATURE_PROVIDER_SET = {"semantic_scholar", "openalex", "scholarapi", "core", "arxiv"}
_REGULATORY_PROVIDER_SET = {"ecos", "federal_register", "govinfo", "tavily", "perplexity"}
_PRIMARY_SOURCE_PROVIDER_SET = {"ecos", "federal_register", "govinfo", "agency_primary_source"}
_SUCCESS_CRITERIA = {"current_text_required", "timeline_required", "dossier_required", "guidance_doc_required"}


def _paper_evidence_payload(papers: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return [
        {
            "paperId": paper.get("paperId") or paper.get("sourceId") or paper.get("canonicalId"),
            "title": paper.get("title"),
            "abstract": str(paper.get("abstract") or "")[:1500] or None,
            "venue": paper.get("venue"),
            "year": paper.get("year"),
            "provider": paper.get("source") or paper.get("provider"),
            "sourceType": paper.get("sourceType"),
            "verificationStatus": paper.get("verificationStatus"),
            "accessStatus": paper.get("accessStatus"),
            "canonicalUrl": paper.get("canonicalUrl") or paper.get("url"),
        }
        for paper in papers[:limit]
    ]


def _build_theme_label_payload(
    seed_terms: list[str],
    papers: list[dict[str, Any]],
    *,
    limit: int = 6,
) -> dict[str, Any]:
    return {
        "seed_terms": seed_terms,
        "titles": [paper.get("title") for paper in papers[:limit]],
    }


def _build_theme_summary_payload(title: str, papers: list[dict[str, Any]], *, limit: int = 5) -> dict[str, Any]:
    return {
        "title": title,
        "papers": _paper_evidence_payload(papers, limit=limit),
    }


def _build_answer_payload(
    question: str,
    answer_mode: str,
    evidence_papers: list[dict[str, Any]],
    *,
    limit: int = 12,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer_mode": answer_mode,
        "evidence": _paper_evidence_payload(evidence_papers, limit=limit),
    }


def _filter_expansion_candidates(
    query: str,
    expansions: list[Any],
    *,
    max_variants: int,
) -> list[ExpansionCandidate]:
    variants: list[ExpansionCandidate] = []
    query_tokens = set(_tokenize(query))
    valid_sources = {"from_input", "from_retrieved_evidence", "speculative", "hypothesis"}
    for item in expansions[:max_variants]:
        if isinstance(item, BaseModel):
            payload = item.model_dump()
        elif isinstance(item, dict):
            payload = dict(item)
        else:
            payload = {
                "variant": getattr(item, "variant", ""),
                "source": getattr(item, "source", "speculative"),
                "rationale": getattr(item, "rationale", ""),
            }
        variant = str(payload.get("variant") or "").strip()
        if not variant:
            continue
        source = str(payload.get("source") or "speculative").strip()
        if source not in valid_sources:
            payload["source"] = "speculative"
        new_tokens = [token for token in _tokenize(variant) if token not in query_tokens]
        if not new_tokens or all(token in COMMON_QUERY_WORDS for token in new_tokens):
            continue
        variants.append(ExpansionCandidate.model_validate(payload))
    return variants


def _sanitize_provider_plan(*, intent: str, provider_plan: list[str]) -> list[str]:
    allowed = _REGULATORY_PROVIDER_SET if intent == "regulatory" else _LITERATURE_PROVIDER_SET
    deduped: list[str] = []
    seen: set[str] = set()
    for provider in provider_plan:
        normalized = str(provider or "").strip()
        if not normalized or normalized in seen:
            continue
        if normalized not in allowed:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    if deduped:
        return deduped
    if intent == "regulatory":
        return ["ecos", "federal_register", "govinfo"]
    return ["semantic_scholar", "openalex", "scholarapi", "core", "arxiv"]


def _sanitize_primary_sources(primary_sources: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for provider in primary_sources:
        normalized = str(provider or "").strip()
        if not normalized or normalized in seen:
            continue
        if normalized not in _PRIMARY_SOURCE_PROVIDER_SET:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _sanitize_success_criteria(criteria: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for criterion in criteria:
        normalized = str(criterion or "").strip()
        if not normalized or normalized in seen:
            continue
        if normalized not in _SUCCESS_CRITERIA:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
