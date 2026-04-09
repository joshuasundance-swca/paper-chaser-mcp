"""Citation parsing, scoring, and recovery helpers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal
from urllib.parse import urlparse

from .models import (
    CitationResolutionCandidate,
    CitationResolutionResponse,
    Paper,
    dump_jsonable,
)
from .models.tools import DEFAULT_SEARCH_PROVIDER_ORDER

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
ARXIV_RE = re.compile(
    r"(?:arxiv:)?(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z][\w.-]+/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
PAGES_RE = re.compile(r"\b\d{1,4}\s*[-:]\s*\d{1,4}\b")
QUOTED_RE = re.compile(r'["“”](.+?)["“”]')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'/-]*")
REGULATORY_CITATION_RE = re.compile(
    r"\b\d+\s*(?:F\.?\s*R\.?|FED(?:ERAL)?\.?\s+REG(?:ISTER)?\.?)\s*\d+\b|\b\d+\s+CFR\b",
    re.IGNORECASE,
)
VENUE_HINTS = (
    "annual review",
    "annual review of ecology",
    "ecological applications",
    "ecology letters",
    "nature sustainability",
    "nature",
    "science",
    "pnas",
    "proceedings of the national academy of sciences",
    "journal of applied ecology",
    "global change biology",
    "environmental science",
    "environmental research letters",
    "acl",
    "cvpr",
    "emnlp",
    "iclr",
    "icml",
    "naacl",
    "neurips",
    "nips",
)
NON_PAPER_TERMS = {
    "dataset",
    "datasheet",
    "dissertation",
    "guidance",
    "guidelines",
    "handbook",
    "manual",
    "package",
    "policy",
    "report",
    "software",
    "standard",
    "thesis",
    "whitepaper",
}
REGULATORY_TERMS = {
    "federal register",
    "fed. reg",
    "cfr",
    "code of federal regulations",
    "final rule",
    "proposed rule",
    "rulemaking",
    "notice of",
}
GENERIC_TITLE_WORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "using",
    "study",
    "studies",
    "paper",
    "papers",
    "framework",
}

logger = logging.getLogger("paper-chaser-mcp.citation-repair")


def _normalize_identifier_for_semantic_scholar(identifier: str, identifier_type: str | None) -> str:
    normalized = normalize_citation_text(identifier)
    if not normalized:
        return normalized
    lowered = normalized.lower()
    if identifier_type == "doi" or DOI_RE.fullmatch(normalized):
        if lowered.startswith("doi:"):
            return f"DOI:{normalized[4:].strip()}"
        doi_match = DOI_RE.search(normalized)
        if doi_match:
            return f"DOI:{doi_match.group(0)}"
    if identifier_type == "arxiv" or ARXIV_RE.fullmatch(normalized):
        if lowered.startswith("arxiv:"):
            return f"ARXIV:{normalized[6:].strip()}"
        return f"ARXIV:{normalized}"
    if identifier_type == "url" and looks_like_url(normalized):
        parsed = urlparse(normalized)
        if parsed.netloc.lower().endswith("semanticscholar.org"):
            path_parts = [part for part in parsed.path.split("/") if part]
            if path_parts:
                candidate = path_parts[-1]
                if re.fullmatch(r"[A-Fa-f0-9]{40}", candidate):
                    return candidate
        return f"URL:{normalized}"
    return normalized


def _normalize_identifier_for_openalex(identifier: str, identifier_type: str | None) -> str | None:
    normalized = normalize_citation_text(identifier)
    if not normalized:
        return None
    lowered = normalized.lower()
    if identifier_type == "doi" or DOI_RE.fullmatch(normalized):
        if lowered.startswith("doi:"):
            normalized = normalized[4:].strip()
        elif lowered.startswith("https://doi.org/"):
            normalized = normalized[16:]
        elif lowered.startswith("http://doi.org/"):
            normalized = normalized[15:]
        doi_match = DOI_RE.search(normalized)
        return doi_match.group(0) if doi_match else None
    if lowered.startswith("https://openalex.org/w") or lowered.startswith("http://openalex.org/w"):
        return normalized.rstrip("/").rsplit("/", 1)[-1]
    if re.fullmatch(r"W\d+", normalized, re.IGNORECASE):
        return normalized
    return None


@dataclass(slots=True)
class ParsedCitation:
    """Deterministic features extracted from a citation-like query."""

    original_text: str
    normalized_text: str
    identifier: str | None = None
    identifier_type: str | None = None
    year: int | None = None
    quoted_fragments: list[str] = field(default_factory=list)
    title_candidates: list[str] = field(default_factory=list)
    author_surnames: list[str] = field(default_factory=list)
    venue_hints: list[str] = field(default_factory=list)
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    looks_like_non_paper: bool = False
    looks_like_regulatory: bool = False


@dataclass(slots=True)
class RankedCitationCandidate:
    """One ranked citation-repair candidate before serialization."""

    paper: dict[str, Any]
    score: float
    resolution_strategy: str
    matched_fields: list[str]
    conflicting_fields: list[str]
    title_similarity: float
    year_delta: int | None
    author_overlap: int
    candidate_count: int | None
    why_selected: str


def normalize_citation_text(value: str) -> str:
    """Collapse whitespace while preserving the user's wording."""
    return " ".join(str(value or "").strip().split())


def looks_like_url(value: str) -> bool:
    """Return True when *value* looks like a URL."""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def looks_like_paper_identifier(value: str) -> bool:
    """Return True when *value* resembles a DOI, arXiv ID, or URL."""
    normalized = normalize_citation_text(value)
    lowered = normalized.lower()
    return bool(
        normalized
        and (
            DOI_RE.search(normalized)
            or ARXIV_RE.search(normalized)
            or looks_like_url(normalized)
            or lowered.startswith("doi:")
            or lowered.startswith("arxiv:")
        )
    )


def looks_like_citation_query(value: str) -> bool:
    """Heuristically identify incomplete or bibliography-style citations."""
    normalized = normalize_citation_text(value)
    if not normalized:
        return False
    lowered = normalized.lower()
    question_like = normalized.endswith("?") or lowered.startswith(
        (
            "what ",
            "which ",
            "how ",
            "why ",
            "compare ",
            "summarize ",
            "identify ",
            "find ",
            "show ",
        )
    )
    word_count = len(WORD_RE.findall(normalized))
    if looks_like_paper_identifier(normalized):
        return True
    if QUOTED_RE.search(normalized):
        return True
    if "et al" in lowered:
        return True
    if YEAR_RE.search(normalized) and not question_like and word_count <= 18:
        return True
    if PAGES_RE.search(normalized):
        return True
    if any(venue in lowered for venue in VENUE_HINTS):
        return True
    if "," in normalized and not question_like and YEAR_RE.search(normalized):
        return True
    return False


def parse_citation(
    citation: str,
    *,
    title_hint: str | None = None,
    author_hint: str | None = None,
    year_hint: str | None = None,
    venue_hint: str | None = None,
    doi_hint: str | None = None,
) -> ParsedCitation:
    """Parse a partial citation into deterministic structured cues."""
    normalized = normalize_citation_text(citation)
    identifier, identifier_type = _extract_identifier(normalized, doi_hint=doi_hint)
    year = _extract_year(normalized, year_hint)
    venue_hints = _extract_venue_hints(normalized, venue_hint=venue_hint)
    author_surnames = _extract_author_surnames(
        normalized,
        author_hint=author_hint,
        citation_like=looks_like_citation_query(normalized),
    )
    title_candidates = _extract_title_candidates(
        normalized,
        title_hint=title_hint,
        author_surnames=author_surnames,
        year=year,
        venue_hints=venue_hints,
    )
    pages = _extract_pages(normalized)
    volume, issue = _extract_volume_issue(normalized)
    lowered = normalized.lower()
    looks_like_regulatory = bool(REGULATORY_CITATION_RE.search(normalized)) or any(
        term in lowered for term in REGULATORY_TERMS
    )
    return ParsedCitation(
        original_text=citation,
        normalized_text=normalized,
        identifier=identifier,
        identifier_type=identifier_type,
        year=year,
        quoted_fragments=[match.strip() for match in QUOTED_RE.findall(normalized)],
        title_candidates=title_candidates,
        author_surnames=author_surnames,
        venue_hints=venue_hints,
        volume=volume,
        issue=issue,
        pages=pages,
        looks_like_non_paper=looks_like_regulatory or any(term in lowered for term in NON_PAPER_TERMS),
        looks_like_regulatory=looks_like_regulatory,
    )


def build_match_metadata(
    *,
    query: str,
    paper: dict[str, Any],
    candidate_count: int | None,
    resolution_strategy: str,
) -> dict[str, Any]:
    """Return additive match metadata for title and citation resolution."""
    parsed = parse_citation(query)
    ranked = _rank_candidate(
        paper=paper,
        parsed=parsed,
        resolution_strategy=resolution_strategy,
        candidate_count=candidate_count,
        snippet_text=None,
    )
    confidence = _classify_resolution_confidence(
        best_score=ranked.score,
        runner_up_score=None,
        matched_fields=ranked.matched_fields,
        conflicting_fields=ranked.conflicting_fields,
        resolution_strategy=ranked.resolution_strategy,
    )
    return {
        "matchConfidence": confidence,
        "matchedFields": ranked.matched_fields,
        "titleSimilarity": ranked.title_similarity,
        "yearDelta": ranked.year_delta,
        "authorOverlap": ranked.author_overlap,
        "candidateCount": candidate_count if candidate_count is not None else 1,
    }


async def resolve_citation(
    *,
    citation: str,
    max_candidates: int,
    client: Any,
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_openalex: bool,
    enable_arxiv: bool,
    enable_serpapi: bool,
    core_client: Any,
    openalex_client: Any,
    arxiv_client: Any,
    serpapi_client: Any,
    title_hint: str | None = None,
    author_hint: str | None = None,
    year_hint: str | None = None,
    venue_hint: str | None = None,
    doi_hint: str | None = None,
    include_enrichment: bool = False,
    enrichment_service: Any = None,
) -> dict[str, Any]:
    """Resolve an incomplete or almost-right citation to canonical papers."""
    parsed = parse_citation(
        citation,
        title_hint=title_hint,
        author_hint=author_hint,
        year_hint=year_hint,
        venue_hint=venue_hint,
        doi_hint=doi_hint,
    )
    candidate_map: dict[str, RankedCitationCandidate] = {}
    stage_order: list[str] = []

    if parsed.looks_like_regulatory and not parsed.identifier:
        return _serialize_citation_response(
            citation=citation,
            parsed=parsed,
            candidates=[],
        )

    if parsed.identifier:
        identifier_candidate = await _resolve_identifier_candidate(
            parsed=parsed,
            client=client,
            openalex_client=openalex_client,
            enable_openalex=enable_openalex,
        )
        if identifier_candidate is not None:
            candidate_map[_paper_identity_key(identifier_candidate.paper)] = identifier_candidate
            stage_order.append("identifier")
            if identifier_candidate.score >= 0.95:
                response = _serialize_citation_response(
                    citation=citation,
                    parsed=parsed,
                    candidates=[identifier_candidate],
                )
                if include_enrichment and enrichment_service is not None:
                    response = await _enrich_best_match(
                        response,
                        enrichment_service=enrichment_service,
                        detail_client=client,
                    )
                return response

    title_candidates = parsed.title_candidates[:3]
    if title_candidates:
        matched_candidates = await _resolve_title_candidates(
            parsed=parsed,
            title_candidates=title_candidates,
            client=client,
        )
        for candidate in matched_candidates:
            _merge_ranked_candidate(candidate_map, candidate)
        if matched_candidates:
            stage_order.append("title_match")
            ranked_candidates = _ranked_candidates(
                candidate_map,
                max_candidates=max_candidates,
            )
            if _best_candidate_confidence(ranked_candidates) == "high":
                response = _serialize_citation_response(
                    citation=citation,
                    parsed=parsed,
                    candidates=ranked_candidates,
                )
                if include_enrichment and enrichment_service is not None:
                    response = await _enrich_best_match(
                        response,
                        enrichment_service=enrichment_service,
                        detail_client=client,
                    )
                response["resolutionStrategy"] = ranked_candidates[0].resolution_strategy
                return response

    snippet_candidates = await _resolve_snippet_candidates(
        parsed=parsed,
        max_candidates=max_candidates,
        client=client,
    )
    for candidate in snippet_candidates:
        _merge_ranked_candidate(candidate_map, candidate)
    if snippet_candidates:
        stage_order.append("snippet_recovery")

    sparse_candidates = await _resolve_sparse_metadata_candidates(
        parsed=parsed,
        max_candidates=max_candidates,
        client=client,
        enable_core=enable_core,
        enable_semantic_scholar=enable_semantic_scholar,
        enable_arxiv=enable_arxiv,
        enable_serpapi=enable_serpapi,
        core_client=core_client,
        arxiv_client=arxiv_client,
        serpapi_client=serpapi_client,
        enable_openalex=enable_openalex,
        openalex_client=openalex_client,
    )
    for candidate in sparse_candidates:
        _merge_ranked_candidate(candidate_map, candidate)
    if sparse_candidates:
        stage_order.append("sparse_metadata")

    ranked_candidates = _ranked_candidates(
        candidate_map,
        max_candidates=max_candidates,
    )
    response = _serialize_citation_response(
        citation=citation,
        parsed=parsed,
        candidates=ranked_candidates,
    )
    if include_enrichment and enrichment_service is not None:
        response = await _enrich_best_match(
            response,
            enrichment_service=enrichment_service,
            detail_client=client,
        )
    if stage_order:
        response["resolutionStrategy"] = (
            ranked_candidates[0].resolution_strategy if ranked_candidates else stage_order[-1]
        )
    return response


async def _enrich_best_match(
    response: dict[str, Any],
    *,
    enrichment_service: Any,
    detail_client: Any | None,
) -> dict[str, Any]:
    from .enrichment import (
        attach_enrichments_to_paper_payload,
        hydrate_paper_for_enrichment,
    )

    best_match = response.get("bestMatch")
    if not isinstance(best_match, dict) or not isinstance(
        best_match.get("paper"),
        dict,
    ):
        return response
    hydrated_paper = await hydrate_paper_for_enrichment(
        best_match["paper"],
        detail_client=detail_client,
    )
    enriched_paper = await enrichment_service.enrich_paper_payload(
        hydrated_paper,
        query=(hydrated_paper.get("title") or best_match["paper"].get("title") or response.get("normalizedCitation")),
    )
    updated_response = dict(response)
    updated_best_match = dict(best_match)
    updated_best_match["paper"] = attach_enrichments_to_paper_payload(
        best_match["paper"],
        enriched_paper=enriched_paper,
    )
    updated_response["bestMatch"] = updated_best_match
    return updated_response


def _serialize_citation_response(
    *,
    citation: str,
    parsed: ParsedCitation,
    candidates: list[RankedCitationCandidate],
) -> dict[str, Any]:
    best = candidates[0] if candidates else None
    runner_up_score = candidates[1].score if len(candidates) > 1 else None
    confidence = _classify_resolution_confidence(
        best_score=best.score if best is not None else None,
        runner_up_score=runner_up_score,
        matched_fields=best.matched_fields if best is not None else [],
        conflicting_fields=best.conflicting_fields if best is not None else [],
        resolution_strategy=best.resolution_strategy if best is not None else "none",
    )
    abstain = bool(
        best is not None
        and (
            confidence == "low"
            or (
                parsed.looks_like_non_paper
                and best.resolution_strategy
                not in {
                    "identifier",
                    "identifier_openalex",
                    "exact_title",
                    "openalex_exact_title",
                    "crossref_exact_title",
                }
                and "identifier" not in best.matched_fields
            )
        )
    )
    alternatives = (
        _abstention_candidates(candidates)
        if abstain
        else _filtered_alternative_candidates(
            candidates=candidates,
            confidence=confidence,
        )
    )
    best_for_response = None if abstain else best
    serializable_candidates = ([best] if best is not None else []) if not abstain else alternatives
    if not abstain and best is not None:
        serializable_candidates = [best, *alternatives]
    response = CitationResolutionResponse(
        bestMatch=_to_candidate_model(best_for_response) if best_for_response is not None else None,
        alternatives=[_to_candidate_model(candidate) for candidate in alternatives],
        resolutionConfidence=confidence,
        resolutionStrategy=best.resolution_strategy if best is not None else "none",
        matchedFields=best.matched_fields if best is not None else [],
        conflictingFields=best.conflicting_fields if best is not None else [],
        normalizedCitation=parsed.normalized_text or normalize_citation_text(citation),
        extractedFields=_parsed_fields_payload(parsed),
        inferredFields=_inferred_fields_payload(parsed, serializable_candidates),
        candidateCount=len(serializable_candidates),
        message=_resolution_message(
            parsed=parsed,
            candidates=serializable_candidates,
            confidence=confidence,
        ),
    )
    return dump_jsonable(response)


def _abstention_candidates(
    candidates: list[RankedCitationCandidate],
) -> list[RankedCitationCandidate]:
    abstained: list[RankedCitationCandidate] = []
    for candidate in candidates:
        if candidate.score < 0.4:
            continue
        if (
            candidate.title_similarity < 0.5
            and candidate.author_overlap == 0
            and "identifier" not in candidate.matched_fields
        ):
            continue
        abstained.append(candidate)
    return abstained[:3]


def _filtered_alternative_candidates(
    *,
    candidates: list[RankedCitationCandidate],
    confidence: str,
) -> list[RankedCitationCandidate]:
    if len(candidates) <= 1:
        return []
    best = candidates[0]
    filtered: list[RankedCitationCandidate] = []
    for candidate in candidates[1:]:
        if candidate.score < 0.4:
            continue
        if confidence == "high":
            if candidate.score < max(0.5, best.score - 0.18):
                continue
            if (
                "identifier" not in candidate.matched_fields
                and "title" not in candidate.matched_fields
                and candidate.author_overlap == 0
            ):
                continue
            if candidate.title_similarity < 0.72 and candidate.author_overlap == 0:
                continue
            if (
                "title" in candidate.conflicting_fields
                and candidate.title_similarity < 0.78
                and "identifier" not in candidate.matched_fields
            ):
                continue
        else:
            if (
                candidate.title_similarity < 0.55
                and candidate.author_overlap == 0
                and "identifier" not in candidate.matched_fields
            ):
                continue
            if (
                candidate.author_overlap == 0
                and "author" in candidate.conflicting_fields
                and candidate.year_delta not in {0, 1}
                and "identifier" not in candidate.matched_fields
            ):
                continue
        filtered.append(candidate)
    return filtered


def _to_candidate_model(
    candidate: RankedCitationCandidate,
) -> CitationResolutionCandidate:
    return CitationResolutionCandidate(
        paper=Paper.model_validate(candidate.paper),
        score=round(candidate.score, 6),
        resolutionStrategy=candidate.resolution_strategy,
        matchedFields=candidate.matched_fields,
        conflictingFields=candidate.conflicting_fields,
        titleSimilarity=round(candidate.title_similarity, 6),
        yearDelta=candidate.year_delta,
        authorOverlap=candidate.author_overlap,
        candidateCount=candidate.candidate_count,
        whySelected=candidate.why_selected,
    )


def _parsed_fields_payload(parsed: ParsedCitation) -> dict[str, Any]:
    return {
        "identifier": parsed.identifier,
        "identifierType": parsed.identifier_type,
        "candidateYear": parsed.year,
        "quotedFragments": parsed.quoted_fragments,
        "titleCandidates": parsed.title_candidates,
        "authorSurnames": parsed.author_surnames,
        "venueHints": parsed.venue_hints,
        "volume": parsed.volume,
        "issue": parsed.issue,
        "pages": parsed.pages,
        "looksLikeNonPaper": parsed.looks_like_non_paper,
        "looksLikeRegulatory": parsed.looks_like_regulatory,
    }


def _inferred_fields_payload(
    parsed: ParsedCitation,
    candidates: list[RankedCitationCandidate],
) -> dict[str, Any]:
    if parsed.looks_like_regulatory:
        likely_output_type = "regulatory_primary_source"
    elif parsed.looks_like_non_paper:
        likely_output_type = "non_paper_candidate"
    else:
        likely_output_type = "paper"
    disambiguation_fields: list[str] = []
    if candidates:
        best = candidates[0]
        if "year" in best.conflicting_fields:
            disambiguation_fields.append("year")
        if "author" in best.conflicting_fields:
            disambiguation_fields.append("author")
        if "venue" in best.conflicting_fields:
            disambiguation_fields.append("venue")
    return {
        "likelyOutputType": likely_output_type,
        "needsDisambiguationBy": disambiguation_fields,
        "primaryTitleCandidate": (parsed.title_candidates[0] if parsed.title_candidates else None),
    }


def _resolution_message(
    *,
    parsed: ParsedCitation,
    candidates: list[RankedCitationCandidate],
    confidence: str,
) -> str:
    if not candidates:
        if parsed.looks_like_regulatory:
            return (
                "This input looks like a Federal Register or CFR citation rather than a paper. "
                "Use search_federal_register for discovery, get_federal_register_document for one notice or rule, "
                "or get_cfr_text for codified text."
            )
        if parsed.looks_like_non_paper:
            return (
                "No confident paper match was found. This input may refer to a "
                "dataset, report, dissertation, software package, or another "
                "non-paper output."
            )
        return (
            "No confident paper match was found. Try adding an author surname, "
            "year, DOI fragment, quote fragment, or venue hint."
        )
    best = candidates[0]
    if parsed.looks_like_regulatory and confidence != "high":
        return (
            "This input looks regulatory rather than paper-like. Prefer the Federal Register or CFR tools over "
            "forcing a scholarly citation match."
        )
    if parsed.looks_like_non_paper and confidence != "high":
        return (
            "This citation looks report-like or otherwise outside the indexed paper surface. "
            "The server is returning low-confidence alternatives instead of forcing a paper match."
        )
    if confidence == "high":
        return (
            "Resolved the citation with strong field agreement. Use the returned "
            "paper as an anchor for details, citations, or grounded follow-up."
        )
    if confidence == "medium":
        return (
            "One candidate leads, but some fields disagree or the citation is "
            "still somewhat incomplete. Review the alternatives before citing it."
        )
    if best.conflicting_fields:
        fields = ", ".join(best.conflicting_fields)
        return (
            "The leading candidate is plausible but not fully confirmed. The "
            f"fastest way to disambiguate is to confirm the {fields}."
        )
    return (
        "The server found plausible candidates but cannot confidently force a "
        "single canonical citation from the available clues."
    )


async def _resolve_identifier_candidate(
    *,
    parsed: ParsedCitation,
    client: Any,
    openalex_client: Any,
    enable_openalex: bool,
) -> RankedCitationCandidate | None:
    if not parsed.identifier:
        return None
    semantic_identifier = _normalize_identifier_for_semantic_scholar(parsed.identifier, parsed.identifier_type)
    openalex_identifier = _normalize_identifier_for_openalex(parsed.identifier, parsed.identifier_type)
    last_error: Exception | None = None
    for strategy, resolver in (
        (
            "identifier",
            lambda: client.get_paper_details(
                paper_id=semantic_identifier,
                fields=None,
            ),
        ),
        (
            "identifier_openalex",
            lambda: openalex_client.get_paper_details(paper_id=openalex_identifier),
        ),
    ):
        if strategy == "identifier_openalex" and not enable_openalex:
            continue
        if strategy == "identifier_openalex" and not openalex_identifier:
            continue
        try:
            paper = dump_jsonable(await resolver())
        except Exception as exc:
            last_error = exc
            logger.debug(
                "Citation identifier resolution via %s failed for %r: %s",
                strategy,
                parsed.identifier,
                exc,
            )
        else:
            return _rank_candidate(
                paper=paper,
                parsed=parsed,
                resolution_strategy=strategy,
                candidate_count=1,
                snippet_text=None,
            )
    if last_error is not None:
        logger.debug(
            "No citation identifier candidate resolved for %r after fallback attempts.",
            parsed.identifier,
        )
    return None


async def _resolve_title_candidates(
    *,
    parsed: ParsedCitation,
    title_candidates: list[str],
    client: Any,
) -> list[RankedCitationCandidate]:
    ranked: list[RankedCitationCandidate] = []
    for title_candidate in title_candidates:
        try:
            payload = dump_jsonable(await client.search_papers_match(query=title_candidate, fields=None))
        except Exception as exc:
            logger.debug(
                "Citation title recovery failed for %r: %s",
                title_candidate,
                exc,
            )
        else:
            if not payload.get("paperId"):
                continue
            strategy = str(payload.get("matchStrategy") or "title_match")
            candidate = _rank_candidate(
                paper=payload,
                parsed=parsed,
                resolution_strategy=strategy,
                candidate_count=int(payload.get("candidateCount") or 1),
                snippet_text=None,
            )
            ranked.append(candidate)
            if (
                _classify_resolution_confidence(
                    best_score=candidate.score,
                    runner_up_score=None,
                    matched_fields=candidate.matched_fields,
                    conflicting_fields=candidate.conflicting_fields,
                    resolution_strategy=candidate.resolution_strategy,
                )
                == "high"
            ):
                break
    return ranked


def _ranked_candidates(
    candidate_map: dict[str, RankedCitationCandidate],
    *,
    max_candidates: int,
) -> list[RankedCitationCandidate]:
    return sorted(
        candidate_map.values(),
        key=lambda item: (
            item.score,
            _publication_preference_score(item.paper),
            item.author_overlap,
            item.title_similarity,
        ),
        reverse=True,
    )[:max_candidates]


def _best_candidate_confidence(
    candidates: list[RankedCitationCandidate],
) -> Literal["high", "medium", "low"]:
    best = candidates[0] if candidates else None
    runner_up_score = candidates[1].score if len(candidates) > 1 else None
    return _classify_resolution_confidence(
        best_score=best.score if best is not None else None,
        runner_up_score=runner_up_score,
        matched_fields=best.matched_fields if best is not None else [],
        conflicting_fields=best.conflicting_fields if best is not None else [],
        resolution_strategy=best.resolution_strategy if best is not None else "none",
    )


async def _resolve_snippet_candidates(
    *,
    parsed: ParsedCitation,
    max_candidates: int,
    client: Any,
) -> list[RankedCitationCandidate]:
    if parsed.quoted_fragments:
        snippet_query = max(parsed.quoted_fragments, key=len)
    elif parsed.title_candidates:
        snippet_query = parsed.title_candidates[0]
    else:
        return []
    try:
        snippet_payload = dump_jsonable(await client.search_snippets(query=snippet_query, limit=max_candidates))
    except Exception:
        return []

    ranked: list[RankedCitationCandidate] = []
    for item in (snippet_payload.get("data") or [])[:max_candidates]:
        if not isinstance(item, dict):
            continue
        paper = item.get("paper")
        if not isinstance(paper, dict) or not paper.get("paperId"):
            continue
        enriched = paper
        try:
            enriched = dump_jsonable(
                await client.get_paper_details(
                    paper_id=str(paper["paperId"]),
                    fields=None,
                )
            )
        except Exception as exc:
            logger.debug(
                "Citation snippet enrichment failed for paper %r: %s",
                paper.get("paperId"),
                exc,
            )
        ranked.append(
            _rank_candidate(
                paper=enriched,
                parsed=parsed,
                resolution_strategy="snippet_recovery",
                candidate_count=len(snippet_payload.get("data") or []),
                snippet_text=_snippet_text(item),
            )
        )
    return ranked


async def _resolve_sparse_metadata_candidates(
    *,
    parsed: ParsedCitation,
    max_candidates: int,
    client: Any,
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_arxiv: bool,
    enable_serpapi: bool,
    core_client: Any,
    arxiv_client: Any,
    serpapi_client: Any,
    enable_openalex: bool,
    openalex_client: Any,
) -> list[RankedCitationCandidate]:
    from .search_executor import SearchClientBundle, SearchExecutor

    search_queries = _sparse_search_queries(parsed)
    ranked: list[RankedCitationCandidate] = []
    search_executor = SearchExecutor()
    for query in search_queries[:3]:
        try:
            search_trace = await search_executor.search_with_fallback(
                query=query,
                limit=max(max_candidates * 4, 12),
                year=str(parsed.year) if parsed.year is not None else None,
                fields=None,
                venue=parsed.venue_hints[:1] or None,
                preferred_provider=("semantic_scholar" if enable_semantic_scholar else None),
                provider_order=None,
                default_provider_order=DEFAULT_SEARCH_PROVIDER_ORDER,
                ss_only_filters=[],
                enabled={
                    "core": enable_core,
                    "semantic_scholar": enable_semantic_scholar,
                    "arxiv": enable_arxiv,
                    "serpapi_google_scholar": enable_serpapi,
                    "openalex": False,
                },
                clients=SearchClientBundle(
                    core_client=core_client,
                    semantic_client=client,
                    arxiv_client=arxiv_client,
                    serpapi_client=serpapi_client,
                ),
                provider_registry=None,
                allow_default_hedging=False,
                publication_date_or_year=None,
                fields_of_study=None,
                publication_types=None,
                open_access_pdf=None,
                min_citation_count=None,
            )
        except Exception as exc:
            logger.debug(
                "Citation sparse metadata recovery failed for %r: %s",
                query,
                exc,
            )
        else:
            data = search_trace.result.model_dump(by_alias=True)["data"] if search_trace.result is not None else []
            for paper in data[: max_candidates * 2]:
                if not isinstance(paper, dict):
                    continue
                ranked.append(
                    _rank_candidate(
                        paper=paper,
                        parsed=parsed,
                        resolution_strategy="sparse_metadata",
                        candidate_count=len(data),
                        snippet_text=None,
                    )
                )
            if ranked:
                break

    if enable_openalex and openalex_client is not None:
        openalex_query = search_queries[0] if search_queries else parsed.normalized_text
        try:
            openalex_payload = dump_jsonable(
                await openalex_client.search(
                    query=openalex_query,
                    limit=max_candidates,
                    year=str(parsed.year) if parsed.year is not None else None,
                )
            )
        except Exception:
            openalex_payload = {}
        for paper in (openalex_payload.get("data") or [])[:max_candidates]:
            if not isinstance(paper, dict):
                continue
            ranked.append(
                _rank_candidate(
                    paper=paper,
                    parsed=parsed,
                    resolution_strategy="openalex_metadata",
                    candidate_count=len(openalex_payload.get("data") or []),
                    snippet_text=None,
                )
            )
    return ranked


def _merge_ranked_candidate(
    candidate_map: dict[str, RankedCitationCandidate],
    candidate: RankedCitationCandidate,
) -> None:
    key = _paper_identity_key(candidate.paper)
    existing = candidate_map.get(key)
    if existing is None:
        candidate_map[key] = candidate
        return
    existing.score = max(existing.score, candidate.score)
    existing.title_similarity = max(
        existing.title_similarity,
        candidate.title_similarity,
    )
    existing.author_overlap = max(existing.author_overlap, candidate.author_overlap)
    if existing.year_delta is None:
        existing.year_delta = candidate.year_delta
    elif candidate.year_delta is not None:
        existing.year_delta = min(existing.year_delta, candidate.year_delta)
    existing.matched_fields = _dedupe_strings([*existing.matched_fields, *candidate.matched_fields])
    existing.conflicting_fields = _dedupe_strings([*existing.conflicting_fields, *candidate.conflicting_fields])
    if candidate.score >= existing.score:
        existing.resolution_strategy = candidate.resolution_strategy
        existing.why_selected = candidate.why_selected
        existing.candidate_count = candidate.candidate_count


def _rank_candidate(
    *,
    paper: dict[str, Any],
    parsed: ParsedCitation,
    resolution_strategy: str,
    candidate_count: int | None,
    snippet_text: str | None,
) -> RankedCitationCandidate:
    upstream_title_similarity = paper.get("titleSimilarity")
    title_similarity = max(
        _title_similarity(parsed, paper),
        float(upstream_title_similarity or 0.0),
    )
    author_overlap = max(
        _author_overlap(parsed, paper),
        int(paper.get("authorOverlap") or 0),
    )
    year_delta = _year_delta(parsed, paper)
    if year_delta is None and paper.get("yearDelta") is not None:
        try:
            year_delta = int(paper["yearDelta"])
        except (TypeError, ValueError):
            year_delta = None
    venue_overlap = _venue_overlap(parsed, paper)
    identifier_hit = resolution_strategy.startswith("identifier") or _identifier_hit(parsed, paper)
    snippet_alignment = _snippet_alignment(parsed, paper, snippet_text=snippet_text)
    source_confidence = _source_confidence(resolution_strategy)
    publication_preference = _publication_preference_score(paper)
    upstream_confidence = str(paper.get("matchConfidence") or "").lower()

    score = 0.0
    if identifier_hit:
        score += 0.55
    score += title_similarity * 0.35
    score += min(author_overlap, 2) * 0.05
    if year_delta == 0:
        score += 0.08
    elif year_delta == 1:
        score += 0.04
    elif year_delta is not None and year_delta > 1:
        score -= min(year_delta, 3) * 0.02
    if venue_overlap:
        score += 0.05
    if snippet_alignment > 0:
        score += min(snippet_alignment, 1.0) * 0.05
    score += source_confidence * 0.05
    score += publication_preference * 0.03
    if upstream_confidence == "high":
        score += 0.25
    elif upstream_confidence == "medium":
        score += 0.12
    score = max(0.0, min(score, 1.0))

    matched_fields: list[str] = []
    conflicting_fields: list[str] = []
    matched_fields.extend(str(field) for field in paper.get("matchedFields") or [])
    conflicting_fields.extend(str(field) for field in paper.get("conflictingFields") or [])
    if identifier_hit:
        matched_fields.append("identifier")
    if title_similarity >= 0.72:
        matched_fields.append("title")
    elif parsed.title_candidates:
        conflicting_fields.append("title")
    if author_overlap > 0:
        matched_fields.append("author")
    elif parsed.author_surnames:
        conflicting_fields.append("author")
    if year_delta == 0:
        matched_fields.append("year")
    elif parsed.year is not None and year_delta is not None and year_delta > 1:
        conflicting_fields.append("year")
    if venue_overlap:
        matched_fields.append("venue")
    elif parsed.venue_hints:
        conflicting_fields.append("venue")
    if snippet_alignment >= 0.35:
        matched_fields.append("snippet")

    why_selected = _why_selected(
        matched_fields=matched_fields,
        conflicting_fields=conflicting_fields,
        paper=paper,
        parsed=parsed,
        resolution_strategy=resolution_strategy,
    )
    return RankedCitationCandidate(
        paper=paper,
        score=score,
        resolution_strategy=resolution_strategy,
        matched_fields=_dedupe_strings(matched_fields),
        conflicting_fields=_dedupe_strings(conflicting_fields),
        title_similarity=title_similarity,
        year_delta=year_delta,
        author_overlap=author_overlap,
        candidate_count=candidate_count,
        why_selected=why_selected,
    )


def _why_selected(
    *,
    matched_fields: list[str],
    conflicting_fields: list[str],
    paper: dict[str, Any],
    parsed: ParsedCitation,
    resolution_strategy: str,
) -> str:
    title = str(paper.get("title") or paper.get("paperId") or "this paper")
    if matched_fields:
        matched_text = ", ".join(matched_fields)
        if conflicting_fields:
            conflicting_text = ", ".join(conflicting_fields)
            return (
                f"{title} matched on {matched_text} via {resolution_strategy}, "
                f"but still conflicts on {conflicting_text}."
            )
        return f"{title} matched on {matched_text} via {resolution_strategy}."
    if parsed.looks_like_non_paper:
        return f"{title} is the nearest paper-like candidate, but the input may describe a non-paper output."
    return f"{title} is a weak fallback candidate from {resolution_strategy}."


def _classify_resolution_confidence(
    *,
    best_score: float | None,
    runner_up_score: float | None,
    matched_fields: list[str],
    conflicting_fields: list[str],
    resolution_strategy: str,
) -> Literal["high", "medium", "low"]:
    if best_score is None:
        return "low"
    gap = best_score - (runner_up_score or 0.0)
    high_signal_fields = {"title", "author", "year"} & set(matched_fields)
    key_conflicting = {"author", "year", "venue"} & set(conflicting_fields)
    if resolution_strategy.startswith("identifier") and "identifier" in matched_fields:
        return "high"
    if resolution_strategy.endswith("exact_title") and "title" in matched_fields:
        if len(key_conflicting) >= 2:
            return "medium"
        return "high"
    if len(high_signal_fields) >= 3 and len(conflicting_fields) <= 1:
        return "high"
    if best_score >= 0.82 and gap >= 0.12 and len(conflicting_fields) <= 1:
        return "high"
    if "title" in matched_fields and len(key_conflicting) <= 1:
        supporting_fields = {"author", "year", "venue", "identifier", "snippet"} & set(matched_fields)
        if supporting_fields and best_score >= 0.5:
            return "medium"
    if (
        resolution_strategy in {"fuzzy_search", "citation_ranked", "snippet_recovery"}
        and "title" in matched_fields
        and (len(high_signal_fields) >= 2 or best_score >= 0.55)
    ):
        return "medium"
    if best_score >= 0.68 and gap >= 0.05 and len(matched_fields) >= 1:
        return "medium"
    return "low"


def _extract_identifier(
    text: str,
    *,
    doi_hint: str | None,
) -> tuple[str | None, str | None]:
    for candidate in (doi_hint, text):
        if not candidate:
            continue
        doi_match = DOI_RE.search(candidate)
        if doi_match:
            return doi_match.group(0), "doi"
        arxiv_match = ARXIV_RE.search(candidate)
        if arxiv_match:
            raw = arxiv_match.group(0)
            return raw if raw.lower().startswith("arxiv:") else f"arXiv:{raw}", "arxiv"
        url_match = URL_RE.search(candidate)
        if url_match:
            return url_match.group(0), "url"
    return None, None


def _extract_year(text: str, year_hint: str | None) -> int | None:
    for candidate in (year_hint, text):
        if not candidate:
            continue
        match = YEAR_RE.search(candidate)
        if match:
            return int(match.group(0))
    return None


def _extract_pages(text: str) -> str | None:
    match = PAGES_RE.search(text)
    return match.group(0) if match else None


def _extract_volume_issue(text: str) -> tuple[str | None, str | None]:
    numeric_tokens = re.findall(r"\b\d{1,4}\b", text)
    if len(numeric_tokens) >= 2:
        return numeric_tokens[0], numeric_tokens[1]
    if len(numeric_tokens) == 1:
        return numeric_tokens[0], None
    return None, None


def _extract_venue_hints(text: str, *, venue_hint: str | None) -> list[str]:
    hints: list[str] = []
    if venue_hint:
        hints.append(normalize_citation_text(venue_hint))
    lowered = text.lower()
    for venue in VENUE_HINTS:
        if venue in lowered:
            hints.append(venue)
    return _dedupe_strings(hints)


def _extract_author_surnames(
    text: str,
    *,
    author_hint: str | None,
    citation_like: bool,
) -> list[str]:
    surnames: list[str] = []
    if author_hint:
        surnames.extend(token for token in WORD_RE.findall(author_hint) if len(token) >= 3)
    for segment in re.split(r"[,;]", text):
        words = WORD_RE.findall(segment)
        if not words:
            continue
        first = words[0]
        tail = words[1:]
        if len(first) >= 3 and first[0].isupper() and tail and all(len(word) <= 2 for word in tail[:3]):
            surnames.append(first)
    lowered = text.lower()
    words = WORD_RE.findall(text)
    if "et al" in lowered and words:
        surnames.append(words[0])
    if "," in text and words:
        surnames.append(words[0])
    return _dedupe_strings([surname.lower() for surname in surnames])


def _extract_title_candidates(
    text: str,
    *,
    title_hint: str | None,
    author_surnames: list[str],
    year: int | None,
    venue_hints: list[str],
) -> list[str]:
    candidates: list[str] = []
    normalized = normalize_citation_text(text)
    if title_hint:
        candidates.append(normalize_citation_text(title_hint))
    candidates.extend(match.strip() for match in QUOTED_RE.findall(normalized))
    if year is not None:
        year_match = re.search(rf"\b{year}\b", normalized)
        if year_match:
            suffix = normalized[year_match.end() :]
            for fragment in re.split(r"[.;]", suffix):
                cleaned = normalize_citation_text(fragment).strip(" -,:")
                if not cleaned:
                    continue
                for venue in venue_hints:
                    cleaned = re.sub(
                        re.escape(venue),
                        " ",
                        cleaned,
                        flags=re.IGNORECASE,
                    )
                cleaned = re.sub(r"\bet\s+al\b", " ", cleaned, flags=re.IGNORECASE)
                cleaned = PAGES_RE.sub(" ", cleaned)
                cleaned = re.sub(r"[,;:()]+", " ", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
                words = WORD_RE.findall(cleaned)
                if len(words) < 2 or len(words) > 18:
                    continue
                if (
                    len(words) <= 4
                    and words[0].lower() in author_surnames
                    and all(len(word) <= 2 for word in words[1:])
                ):
                    continue
                candidates.append(" ".join(words))

    working = normalized
    if year is not None:
        working = re.sub(rf"\b{year}\b", " ", working)
    if DOI_RE.search(working):
        working = DOI_RE.sub(" ", working)
    if ARXIV_RE.search(working):
        working = ARXIV_RE.sub(" ", working)
    if URL_RE.search(working):
        working = URL_RE.sub(" ", working)
    if venue_hints:
        for venue in venue_hints:
            working = re.sub(re.escape(venue), " ", working, flags=re.IGNORECASE)
    working = PAGES_RE.sub(" ", working)
    working = re.sub(r"\bet\s+al\b", " ", working, flags=re.IGNORECASE)
    working = re.sub(r"[,;:()]+", " ", working)
    working = re.sub(r"\s+", " ", working).strip()
    words = WORD_RE.findall(working)
    if words:
        candidates.append(" ".join(words))
        if len(words) > 1 and words[0].lower() in author_surnames:
            candidates.append(" ".join(words[1:]))
        if len(words) >= 6:
            candidates.extend(" ".join(words[:-count]) for count in (1, 2) if len(words) - count >= 4)
        if len(words) > 3 and words[:2] == ["et", "al"]:
            candidates.append(" ".join(words[2:]))
        if len(words) > 3 and author_surnames and words[0].lower() in author_surnames:
            candidates.append(" ".join(words[1:]))
    compact_tokens = [
        token for token in (word.lower() for word in words) if len(token) >= 3 and token not in GENERIC_TITLE_WORDS
    ]
    if compact_tokens:
        candidates.append(" ".join(compact_tokens[:10]))
    return _dedupe_strings(candidate for candidate in candidates if 2 <= len(WORD_RE.findall(candidate)) <= 18)


def _sparse_search_queries(parsed: ParsedCitation) -> list[str]:
    queries: list[str] = []
    if parsed.title_candidates:
        queries.append(parsed.title_candidates[0])
    if parsed.author_surnames and parsed.title_candidates:
        queries.append(
            " ".join(
                [
                    *parsed.author_surnames[:2],
                    *WORD_RE.findall(parsed.title_candidates[0])[:8],
                ]
            )
        )
    if parsed.author_surnames and parsed.year is not None:
        queries.append(" ".join([*parsed.author_surnames[:2], str(parsed.year)]))
    if parsed.venue_hints and parsed.title_candidates:
        queries.append(f"{parsed.title_candidates[0]} {parsed.venue_hints[0]}")
    queries.append(parsed.normalized_text)
    return _dedupe_strings(query for query in queries if normalize_citation_text(query))


def _paper_identity_key(paper: dict[str, Any]) -> str:
    for candidate in (
        paper.get("paperId"),
        paper.get("canonicalId"),
        paper.get("recommendedExpansionId"),
        paper.get("sourceId"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    title = normalize_citation_text(str(paper.get("title") or "")).lower()
    year = str(paper.get("year") or "")
    return f"title:{title}|year:{year}"


def _title_similarity(parsed: ParsedCitation, paper: dict[str, Any]) -> float:
    title = normalize_citation_text(str(paper.get("title") or "")).lower()
    if not title:
        return 0.0
    candidates = parsed.title_candidates or [parsed.normalized_text]
    best = 0.0
    for candidate in candidates:
        normalized_candidate = normalize_citation_text(candidate).lower()
        if not normalized_candidate:
            continue
        best = max(
            best,
            SequenceMatcher(None, normalized_candidate, title).ratio(),
            _token_overlap_ratio(normalized_candidate, title),
            _weighted_token_overlap_ratio(normalized_candidate, title),
        )
    return best


def _author_overlap(parsed: ParsedCitation, paper: dict[str, Any]) -> int:
    if not parsed.author_surnames:
        return 0
    author_names = {
        _surname(str(author.get("name") or ""))
        for author in (paper.get("authors") or [])
        if isinstance(author, dict) and author.get("name")
    }
    return sum(1 for surname in parsed.author_surnames if surname in author_names)


def _year_delta(parsed: ParsedCitation, paper: dict[str, Any]) -> int | None:
    if parsed.year is None or paper.get("year") is None:
        return None
    try:
        return abs(int(paper["year"]) - int(parsed.year))
    except (TypeError, ValueError):
        return None


def _venue_overlap(parsed: ParsedCitation, paper: dict[str, Any]) -> bool:
    if not parsed.venue_hints:
        return False
    venue = normalize_citation_text(str(paper.get("venue") or "")).lower()
    if not venue:
        return False
    return any(hint.lower() in venue or venue in hint.lower() for hint in parsed.venue_hints)


def _identifier_hit(parsed: ParsedCitation, paper: dict[str, Any]) -> bool:
    if not parsed.identifier:
        return False
    lowered_identifier = parsed.identifier.lower()
    external_ids = paper.get("externalIds") or {}
    candidates = [
        str(paper.get("paperId") or ""),
        str(paper.get("canonicalId") or ""),
        str(paper.get("recommendedExpansionId") or ""),
        str(external_ids.get("DOI") or ""),
        str(external_ids.get("ArXiv") or ""),
    ]
    if parsed.identifier_type == "doi":
        normalized_identifier = lowered_identifier.removeprefix("doi:")
        return any(
            normalized_identifier == candidate.lower().removeprefix("doi:") for candidate in candidates if candidate
        )
    if parsed.identifier_type == "arxiv":
        normalized_identifier = lowered_identifier.removeprefix("arxiv:")
        return any(
            normalized_identifier == candidate.lower().removeprefix("arxiv:") for candidate in candidates if candidate
        )
    return any(lowered_identifier == candidate.lower() for candidate in candidates if candidate)


def _snippet_alignment(
    parsed: ParsedCitation,
    paper: dict[str, Any],
    *,
    snippet_text: str | None,
) -> float:
    if not snippet_text:
        return 0.0
    paper_text = " ".join(
        part for part in [str(paper.get("title") or ""), str(paper.get("abstract") or "")] if part
    ).lower()
    if not paper_text:
        return 0.0
    return _token_overlap_ratio(snippet_text.lower(), paper_text)


def _snippet_text(item: dict[str, Any]) -> str | None:
    snippet = item.get("snippet")
    if isinstance(snippet, dict) and snippet.get("text"):
        return str(snippet["text"])
    if item.get("text"):
        return str(item["text"])
    return None


def _source_confidence(strategy: str) -> float:
    mapping = {
        "identifier": 1.0,
        "identifier_openalex": 0.92,
        "exact_title": 0.9,
        "openalex_exact_title": 0.9,
        "crossref_exact_title": 0.84,
        "fuzzy_search": 0.82,
        "citation_ranked": 0.74,
        "snippet_recovery": 0.7,
        "sparse_metadata": 0.65,
        "openalex_metadata": 0.6,
    }
    return mapping.get(strategy, 0.55)


def _publication_preference_score(paper: dict[str, Any]) -> float:
    publication_types_raw = paper.get("publicationTypes")
    if isinstance(publication_types_raw, str):
        publication_types = {publication_types_raw.lower()}
    elif isinstance(publication_types_raw, list):
        publication_types = {str(value).lower() for value in publication_types_raw}
    else:
        publication_types = set()
    source = str(paper.get("source") or "").lower()
    venue = normalize_citation_text(str(paper.get("venue") or "")).lower()
    external_ids = paper.get("externalIds") or {}
    doi = str(external_ids.get("DOI") or paper.get("doi") or "").strip()

    score = 0.0
    if doi or str(paper.get("canonicalId") or "").lower().startswith("10."):
        score += 1.0
    if venue and "arxiv" not in venue:
        score += 0.6
    if publication_types & {"journal-article", "proceedings-article", "conference", "conference-paper"}:
        score += 0.8
    if source == "arxiv" or "preprint" in publication_types or venue == "arxiv":
        score -= 1.2
    return score


def _surname(name: str) -> str:
    words = [word.lower() for word in WORD_RE.findall(name)]
    return words[-1] if words else ""


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = {token for token in re.findall(r"[a-z0-9]{3,}", left.lower()) if token not in GENERIC_TITLE_WORDS}
    right_tokens = {token for token in re.findall(r"[a-z0-9]{3,}", right.lower()) if token not in GENERIC_TITLE_WORDS}
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    return len(intersection) / len(left_tokens)


def _weighted_token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = [token for token in re.findall(r"[a-z0-9]{3,}", left.lower()) if token not in GENERIC_TITLE_WORDS]
    right_token_set = {
        token for token in re.findall(r"[a-z0-9]{3,}", right.lower()) if token not in GENERIC_TITLE_WORDS
    }
    if not left_tokens or not right_token_set:
        return 0.0
    matched_weight = sum(max(len(token) - 2, 1) for token in left_tokens if token in right_token_set)
    total_weight = sum(max(len(token) - 2, 1) for token in left_tokens)
    return matched_weight / total_weight if total_weight else 0.0


def _dedupe_strings(values: Any) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_citation_text(str(value))
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped
