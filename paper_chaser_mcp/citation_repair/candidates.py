"""Citation parsing and candidate-classification helpers.

Phase 9a extracted this module from ``paper_chaser_mcp/citation_repair/_core.py``.
It owns the :class:`ParsedCitation` dataclass, the famous-paper registry, the
``_extract_*`` feature extractors that back :func:`parse_citation`, the
confidence classifiers, and the deterministic sparse-query synthesizer. The
module keeps its imports narrow so :mod:`ranking` can depend on it without
creating a top-level cycle (ranking-dependent helpers like
:func:`build_match_metadata` use function-local imports).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .normalization import (
    ARXIV_RE,
    DOI_RE,
    GENERIC_TITLE_WORDS,
    NON_PAPER_TERMS,
    PAGES_RE,
    QUOTED_RE,
    REGULATORY_CITATION_RE,
    REGULATORY_TERMS,
    URL_RE,
    VENUE_HINTS,
    WORD_RE,
    YEAR_RE,
    _dedupe_strings,
    _venue_hint_in_text,
    normalize_citation_text,
)

if TYPE_CHECKING:  # pragma: no cover - import-time forward references only.
    from .ranking import RankedCitationCandidate


# Registry of famous "classic" papers that are commonly referenced only by
# author surnames + year (e.g. "Watson and Crick 1953"). Natural-language
# references like this would otherwise fail to resolve because the sparse
# queries lack a specific title; the registry short-circuits such lookups to
# canonical metadata without hitting any upstream provider.
_FAMOUS_CITATION_ENTRIES: list[dict[str, Any]] = [
    {
        "surnames": frozenset({"watson", "crick"}),
        "year": 1953,
        "venue_hint": "nature",
        "paper": {
            "paperId": "famous:watson-crick-1953",
            "title": ("Molecular Structure of Nucleic Acids: A Structure for Deoxyribose Nucleic Acid"),
            "year": 1953,
            "venue": "Nature",
            "authors": [
                {"name": "James D. Watson"},
                {"name": "Francis H. C. Crick"},
            ],
            "externalIds": {"DOI": "10.1038/171737a0"},
        },
    },
    {
        "surnames": frozenset({"einstein"}),
        "year": 1905,
        "venue_hint": "annalen",
        "paper": {
            "paperId": "famous:einstein-1905-relativity",
            "title": "Zur Elektrodynamik bewegter Körper",
            "year": 1905,
            "venue": "Annalen der Physik",
            "authors": [{"name": "Albert Einstein"}],
            "externalIds": {"DOI": "10.1002/andp.19053221004"},
        },
    },
    {
        "surnames": frozenset({"shannon"}),
        "year": 1948,
        "venue_hint": "bell",
        "paper": {
            "paperId": "famous:shannon-1948-communication",
            "title": "A Mathematical Theory of Communication",
            "year": 1948,
            "venue": "Bell System Technical Journal",
            "authors": [{"name": "Claude E. Shannon"}],
            "externalIds": {"DOI": "10.1002/j.1538-7305.1948.tb01338.x"},
        },
    },
    {
        "surnames": frozenset({"turing"}),
        "year": 1937,
        "venue_hint": "london",
        "paper": {
            "paperId": "famous:turing-1937-computable",
            "title": "On Computable Numbers, with an Application to the Entscheidungsproblem",
            "year": 1937,
            "venue": "Proceedings of the London Mathematical Society",
            "authors": [{"name": "Alan M. Turing"}],
            "externalIds": {"DOI": "10.1112/plms/s2-42.1.230"},
        },
    },
    {
        "surnames": frozenset({"lewis"}),
        "year": 2020,
        "venue_hint": "arxiv",
        "paper": {
            "paperId": "famous:lewis-2020-rag",
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "year": 2020,
            "venue": "arXiv",
            "authors": [{"name": "Patrick Lewis"}],
            "externalIds": {"ArXiv": "2005.11401"},
        },
    },
]


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


def _lookup_famous_citation(parsed: "ParsedCitation") -> dict[str, Any] | None:
    """Match a parsed citation against the famous-paper registry.

    Returns a canonical paper dict when author surnames + year align with a
    known classic; otherwise returns ``None``.
    """
    if parsed.year is None or not parsed.author_surnames:
        return None
    parsed_surnames = {surname.lower() for surname in parsed.author_surnames}
    for entry in _FAMOUS_CITATION_ENTRIES:
        if entry["year"] != parsed.year:
            continue
        surnames: frozenset[str] = entry["surnames"]
        if not surnames.issubset(parsed_surnames):
            continue
        return entry["paper"]
    return None


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
    word_count = len(WORD_RE.findall(normalized)) + len(YEAR_RE.findall(normalized))
    if looks_like_paper_identifier(normalized):
        return True
    if QUOTED_RE.search(normalized):
        return True
    if "et al" in lowered:
        return True
    if YEAR_RE.search(normalized) and not question_like and word_count <= 18:
        hyphenated_count = len(re.findall(r"\b[A-Za-z0-9]+-[A-Za-z0-9]+\b", normalized))
        has_bibliographic_cue = (
            "," in normalized or "et al" in lowered or any(_venue_hint_in_text(lowered, venue) for venue in VENUE_HINTS)
        )
        if word_count >= 6 and not has_bibliographic_cue:
            return False
        if hyphenated_count >= 2 and not has_bibliographic_cue:
            return False
        return True
    if PAGES_RE.search(normalized):
        return True
    if any(_venue_hint_in_text(lowered, venue) for venue in VENUE_HINTS):
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
    # Local import avoids a top-level cycle with :mod:`ranking`, which
    # itself depends on :class:`ParsedCitation` and :func:`_why_selected`.
    from .ranking import _rank_candidate

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
        title_similarity=ranked.title_similarity,
    )
    resolution_state = classify_known_item_resolution_state(
        resolution_confidence=confidence,
        resolution_strategy=ranked.resolution_strategy,
        matched_fields=ranked.matched_fields,
        conflicting_fields=ranked.conflicting_fields,
        title_similarity=ranked.title_similarity,
        year_delta=ranked.year_delta,
        author_overlap=ranked.author_overlap,
        best_score=ranked.score,
        runner_up_score=None,
        candidate_count=candidate_count if candidate_count is not None else 1,
        has_best_match=True,
    )
    return {
        "matchConfidence": confidence,
        "matchedFields": ranked.matched_fields,
        "titleSimilarity": ranked.title_similarity,
        "yearDelta": ranked.year_delta,
        "authorOverlap": ranked.author_overlap,
        "candidateCount": candidate_count if candidate_count is not None else 1,
        "knownItemResolutionState": resolution_state,
    }


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


def classify_known_item_resolution_state(
    *,
    resolution_confidence: Literal["high", "medium", "low"],
    resolution_strategy: str,
    matched_fields: list[str] | None,
    conflicting_fields: list[str] | None,
    title_similarity: float | None,
    year_delta: int | None,
    author_overlap: int | None,
    best_score: float | None,
    runner_up_score: float | None,
    candidate_count: int | None,
    has_best_match: bool = True,
) -> Literal["resolved_exact", "resolved_probable", "needs_disambiguation"]:
    """Classify a known-item / resolve_reference outcome into one of three
    execution-provenance labels.

    Rules (ordered):

    * ``needs_disambiguation`` when there is no best match, when the best score
      is implausibly low, or when a runner-up sits within a small epsilon of
      the best candidate (the agent should clarify).
    * ``resolved_exact`` when an identifier round-trip succeeded without a
      title conflict, OR when a fuzzy/exact-title match has very high title
      similarity AND at least one corroborating field (author or year) AND no
      conflicts on key bibliographic fields.
    * ``resolved_probable`` otherwise when confidence is ``high`` but a key
      field conflicts, or when confidence is ``medium``, or when a fuzzy hit
      lands in the 0.72–0.9 title-similarity band.
    * Fall back to ``needs_disambiguation`` when confidence is ``low``.
    """
    matched = set(matched_fields or [])
    conflicting = set(conflicting_fields or [])
    key_conflicts = {"author", "year", "venue"} & conflicting
    title_conflict = "title" in conflicting

    if not has_best_match:
        return "needs_disambiguation"
    if best_score is not None and best_score < 0.5 and "identifier" not in matched:
        return "needs_disambiguation"
    if (
        best_score is not None
        and runner_up_score is not None
        and (best_score - runner_up_score) < 0.05
        and "identifier" not in matched
    ):
        return "needs_disambiguation"
    if (candidate_count or 0) >= 2 and resolution_confidence == "low":
        return "needs_disambiguation"

    if resolution_confidence == "high":
        if resolution_strategy.startswith("identifier") and "identifier" in matched and not title_conflict:
            return "resolved_exact"
        if (
            title_similarity is not None
            and title_similarity >= 0.9
            and (year_delta == 0 or (author_overlap or 0) >= 1)
            and not key_conflicts
            and not title_conflict
        ):
            return "resolved_exact"
        # Confidence is high but a key field disagrees — treat as probable,
        # not exact.
        return "resolved_probable"

    if resolution_confidence == "medium":
        return "resolved_probable"

    # resolution_confidence == "low"
    if title_similarity is not None and title_similarity >= 0.72 and (author_overlap or 0) >= 1 and not title_conflict:
        return "resolved_probable"
    return "needs_disambiguation"


def _classify_resolution_confidence(
    *,
    best_score: float | None,
    runner_up_score: float | None,
    matched_fields: list[str],
    conflicting_fields: list[str],
    resolution_strategy: str,
    title_similarity: float | None = None,
) -> Literal["high", "medium", "low"]:
    if best_score is None:
        return "low"
    gap = best_score - (runner_up_score or 0.0)
    high_signal_fields = {"title", "author", "year"} & set(matched_fields)
    key_conflicting = {"author", "year", "venue"} & set(conflicting_fields)
    title_is_conflicting = "title" in set(conflicting_fields)
    if resolution_strategy.startswith("identifier") and "identifier" in matched_fields:
        if title_is_conflicting:
            return "medium"
        return "high"
    if resolution_strategy.endswith("exact_title") and "title" in matched_fields:
        if len(key_conflicting) >= 2:
            return "medium"
        return "high"
    if (
        resolution_strategy in {"fuzzy_search", "citation_ranked", "snippet_recovery"}
        and "title" in matched_fields
        and not title_is_conflicting
        and title_similarity is not None
        and title_similarity >= 0.9
    ):
        if ("year" in matched_fields or "author" in matched_fields) and best_score >= 0.42:
            return "high"
        return "medium"
    if title_is_conflicting:
        # Title conflict caps confidence at medium regardless of other signals.
        # Only promote to medium when no other key fields conflict.
        if len(high_signal_fields) >= 2 and len(key_conflicting) == 0:
            return "medium"
        if best_score >= 0.68 and gap >= 0.05 and len(key_conflicting) == 0:
            return "medium"
        return "low"
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
        if _venue_hint_in_text(lowered, venue):
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
    # APA co-author form: "Surname, I. I." — captures surnames preceding
    # initial sequences regardless of leading "&"/"and" separators.
    for match in re.finditer(
        r"\b([A-Z][a-zA-Z'\-]{2,}),\s*(?:[A-Z]\.\s*){1,4}",
        text,
    ):
        surnames.append(match.group(1))
    # Natural-language two-author form: "Watson and Crick" / "Watson & Crick".
    for match in re.finditer(
        r"\b([A-Z][a-zA-Z'\-]{2,})\s+(?:and|&)\s+([A-Z][a-zA-Z'\-]{2,})\b",
        text,
    ):
        surnames.append(match.group(1))
        surnames.append(match.group(2))
    # Whitespace-only surname list before a 4-digit year, e.g. "Watson Crick 1953".
    for match in re.finditer(
        r"\b([A-Z][a-zA-Z'\-]{2,})\s+([A-Z][a-zA-Z'\-]{2,})\s+\d{4}\b",
        text,
    ):
        surnames.append(match.group(1))
        surnames.append(match.group(2))
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
    if parsed.venue_hints and parsed.author_surnames and parsed.year is not None:
        queries.append(
            " ".join(
                [
                    parsed.venue_hints[0],
                    parsed.author_surnames[0],
                    str(parsed.year),
                ]
            )
        )
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


# ``looks_like_paper_identifier`` is re-exported from :mod:`normalization`
# here because :func:`looks_like_citation_query` depends on it; callers that
# want the primitive should reach through :mod:`normalization` directly.
from .normalization import looks_like_paper_identifier  # noqa: E402,F401
