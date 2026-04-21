"""Evidence-gap analysis and deterministic-fallback helpers for provider adapters."""

from __future__ import annotations

import re
from typing import Any

from .nlp import COMMON_QUERY_WORDS, THEME_LABEL_STOPWORDS, _tokenize, _top_terms

COMPARISON_STOPWORDS = COMMON_QUERY_WORDS | {
    "about",
    "across",
    "agent",
    "agents",
    "compare",
    "comparison",
    "different",
    "directly",
    "effects",
    "generic",
    "more",
    "near",
    "paper",
    "papers",
    "query",
    "which",
}
GAP_QUESTION_MARKERS = {
    "gap",
    "gaps",
    "limitation",
    "limitations",
    "missing",
    "unknown",
    "uncertainty",
    "uncertainties",
    "understudied",
    "underrepresented",
}
_MONTH_NAME_TO_NUMBER = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}
_ADEQUACY_PREFIX = "adequacy assessment:"
_ECOS_GAP_TEXT = "No ECOS species dossier match was found for the query."
_SPECIES_QUERY_TERMS = {
    "critical habitat",
    "dossier",
    "ecos",
    "endangered",
    "esa",
    "habitat",
    "listed",
    "listing",
    "recovery",
    "species",
    "threatened",
    "wildlife",
}
BEHAVIOR_TERMS = {"behavior", "behaviour", "acoustic", "communication", "response"}
PHYSIOLOGY_TERMS = {
    "physiology",
    "physiological",
    "hormone",
    "cortisol",
    "stress",
    "endocrine",
}
DEMOGRAPHY_TERMS = {
    "demographic",
    "demographics",
    "population",
    "survival",
    "reproduction",
    "fitness",
    "fecundity",
}
COMMUNITY_TERMS = {"community", "ecosystem", "ecosystems", "assemblage", "foodweb"}
MULTISTRESSOR_TERMS = {
    "interaction",
    "interactions",
    "combined",
    "cumulative",
    "multiple",
    "multistressor",
    "climate",
    "habitat",
    "pollution",
}
LONGITUDINAL_TERMS = {"longterm", "longitudinal", "chronic", "temporal"}
GEO_TOKENS = {
    "africa",
    "antarctic",
    "arctic",
    "asia",
    "australia",
    "canada",
    "china",
    "europe",
    "global",
    "northamerica",
    "southamerica",
    "tropics",
    "usa",
}
TAXON_GROUPS: dict[str, set[str]] = {
    "birds": {"bird", "birds", "avian"},
    "mammals": {"mammal", "mammals", "cetacean", "cetaceans", "bat", "bats"},
    "fish": {"fish", "fishes"},
    "amphibians": {"amphibian", "amphibians", "frog", "frogs", "toad", "toads"},
    "reptiles": {"reptile", "reptiles", "lizard", "lizards", "snake", "snakes"},
    "invertebrates": {"invertebrate", "invertebrates", "insect", "insects"},
}


def _theme_label_terms(seed_terms: list[str], papers: list[dict[str, Any]]) -> list[str]:
    if papers:
        title_terms = _top_terms(
            [str(paper.get("title") or "") for paper in papers],
            limit=6,
        )
        prioritized = [term for term in title_terms if term not in THEME_LABEL_STOPWORDS]
        if prioritized:
            return prioritized[:3]
    normalized_seed_terms = [
        " ".join(token.capitalize() for token in _tokenize(term)) for term in seed_terms if _tokenize(term)
    ]
    return [term for term in normalized_seed_terms if term][:2]


def _compact_theme_label(seed_terms: list[str], papers: list[dict[str, Any]]) -> str:
    chosen_terms = _theme_label_terms(seed_terms, papers)
    if papers and len(chosen_terms) >= 2:
        return " / ".join(term.title() for term in chosen_terms[:2])
    if chosen_terms:
        return " / ".join(chosen_terms[:2])
    if papers and papers[0].get("venue"):
        return f"{papers[0]['venue']} cluster"
    return "General theme"


def _paper_terms(paper: dict[str, Any]) -> set[str]:
    tokens = _tokenize(
        " ".join(
            part
            for part in [
                str(paper.get("title") or ""),
                str(paper.get("abstract") or ""),
                str(paper.get("venue") or ""),
            ]
            if part
        )
    )
    normalized_tokens = set(tokens)
    if "north" in normalized_tokens and "america" in normalized_tokens:
        normalized_tokens.add("northamerica")
    if "south" in normalized_tokens and "america" in normalized_tokens:
        normalized_tokens.add("southamerica")
    if "long" in normalized_tokens and "term" in normalized_tokens:
        normalized_tokens.add("longterm")
    return normalized_tokens


def _normalize_gap_text(gap: str) -> str | None:
    text = str(gap or "").strip()
    if not text:
        return None
    if text.lower().startswith(_ADEQUACY_PREFIX):
        return None
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _ecos_gap_is_relevant(*, query: str, intent: str, anchor_type: str | None) -> bool:
    if intent != "regulatory":
        return False
    if anchor_type in {"species_common_name", "species_scientific_name"}:
        return True
    lowered = str(query or "").lower()
    return any(term in lowered for term in _SPECIES_QUERY_TERMS)


def _query_month_year_references(query: str) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    pattern = r"\b(" + "|".join(_MONTH_NAME_TO_NUMBER.keys()) + r")\s+((?:19|20)\d{2})\b"
    for match in re.finditer(pattern, str(query or ""), re.IGNORECASE):
        month_name = match.group(1).lower()
        year = match.group(2)
        references.append((f"{month_name} {year}", f"{year}-{_MONTH_NAME_TO_NUMBER[month_name]}"))
    return references


def _timeline_gap_statements(query: str, timeline: dict[str, Any] | None) -> list[str]:
    references = _query_month_year_references(query)
    if not references:
        return []
    descriptor = "final action" if "final action" in str(query or "").lower() else "event"
    events = list((timeline or {}).get("events") or [])
    if not events:
        return [
            f"The retrieved timeline does not cover the {reference} {descriptor} referenced in the query."
            for reference, _ in references
        ]
    event_text = " ".join(
        str(item.get(key) or "")
        for item in events
        if isinstance(item, dict)
        for key in ("eventDate", "date", "publicationDate", "title", "citation", "note")
    ).lower()
    gaps: list[str] = []
    for reference, numeric_reference in references:
        if reference not in event_text and numeric_reference not in event_text:
            gaps.append(f"The retrieved timeline does not cover the {reference} {descriptor} referenced in the query.")
    return gaps


def _hypothesis_gap_statements(retrieval_hypotheses: list[str]) -> list[str]:
    gaps: list[str] = []
    for hypothesis in retrieval_hypotheses:
        text = str(hypothesis or "").strip()
        normalized = _normalize_gap_text(text)
        if not normalized:
            continue
        lowered = normalized.lower()
        if any(
            marker in lowered
            for marker in (
                "no ",
                "missing",
                "required",
                "not included",
                "not recover",
                "not found",
                "absent",
                "incomplete",
                "would be required",
            )
        ):
            gaps.append(normalized)
            continue
        gaps.append(f"Missing evidence covering {text.rstrip('.')}.")
    return gaps


def generate_evidence_gaps_without_llm(
    *,
    query: str,
    intent: str,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    retrieval_hypotheses: list[str],
    coverage_summary: dict[str, Any] | None,
    timeline: dict[str, Any] | None,
    anchor_type: str | None,
) -> list[str]:
    del coverage_summary
    filtered: list[str] = []
    ecos_relevant = _ecos_gap_is_relevant(query=query, intent=intent, anchor_type=anchor_type)
    for gap in evidence_gaps:
        normalized = _normalize_gap_text(str(gap or ""))
        if not normalized:
            continue
        if normalized == _ECOS_GAP_TEXT and not ecos_relevant:
            continue
        filtered.append(normalized)

    filtered.extend(_timeline_gap_statements(query, timeline))

    verified_on_topic = any(
        str(source.get("topicalRelevance") or "") == "on_topic"
        and str(source.get("verificationStatus") or "") in {"verified_primary_source", "verified_metadata"}
        for source in sources
        if isinstance(source, dict)
    )
    if not verified_on_topic:
        filtered.extend(_hypothesis_gap_statements(retrieval_hypotheses))

    if not filtered and not verified_on_topic:
        query_text = " ".join(str(query or "").split())
        if query_text:
            filtered.append(f"No verified on-topic sources addressed the requested evidence for: {query_text}.")

    deduped: list[str] = []
    seen: set[str] = set()
    for gap in filtered:
        normalized = str(gap or "").strip()
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped[:6]


def _question_focus_terms(question: str) -> list[str]:
    return [token for token in _tokenize(question) if token not in COMPARISON_STOPWORDS]


def _paper_focus_cues(paper: dict[str, Any], *, question_terms: list[str]) -> list[str]:
    terms = _paper_terms(paper)
    cues = [term for term in question_terms if term in terms]
    if cues:
        return cues[:3]
    title_tokens = [token for token in _tokenize(str(paper.get("title") or "")) if token not in THEME_LABEL_STOPWORDS]
    return title_tokens[:3]


def _paper_alignment_bucket(paper: dict[str, Any], *, question_terms: list[str]) -> str:
    if not question_terms:
        return "related"
    terms = _paper_terms(paper)
    overlap = sum(term in terms for term in question_terms)
    overlap_ratio = overlap / len(question_terms)
    if overlap_ratio >= 0.5 or overlap >= 3:
        return "direct"
    if overlap > 0:
        return "analog"
    return "broad"


def _format_paper_anchor(paper: dict[str, Any]) -> str:
    title = str(paper.get("title") or paper.get("paperId") or "Untitled")
    venue = str(paper.get("venue") or "venue unknown")
    year = paper.get("year")
    year_text = str(year) if isinstance(year, int) else "year unknown"
    return f"{title} ({venue}; {year_text})"


def _deterministic_comparison_answer(question: str, evidence_papers: list[dict[str, Any]]) -> str:
    question_terms = _question_focus_terms(question)
    direct: list[str] = []
    analog: list[str] = []
    broad: list[str] = []
    for paper in evidence_papers[:5]:
        bucket = _paper_alignment_bucket(paper, question_terms=question_terms)
        cues = _paper_focus_cues(paper, question_terms=question_terms)
        cue_text = ", ".join(cues) if cues else "broader contextual overlap"
        line = f"- {_format_paper_anchor(paper)}; strongest cues: {cue_text}."
        if bucket == "direct":
            direct.append(line)
        elif bucket == "analog":
            analog.append(line)
        else:
            broad.append(line)

    sections = ["Comparison grounded in the saved result set."]
    if direct:
        sections.append("Most directly aligned papers:")
        sections.extend(direct)
    if analog:
        sections.append("Related analog papers:")
        sections.extend(analog)
    if broad:
        sections.append("Broader context papers:")
        sections.extend(broad)

    takeaway_parts: list[str] = []
    if direct:
        takeaway_parts.append(f"{len(direct)} paper(s) are directly aligned to the query focus")
    if analog:
        takeaway_parts.append(f"{len(analog)} provide analog evidence")
    if broad:
        takeaway_parts.append(f"{len(broad)} are only broader context")
    if takeaway_parts:
        sections.append("Takeaway: " + "; ".join(takeaway_parts) + ".")
    return "\n".join(sections)


def _deterministic_theme_summary(title: str, papers: list[dict[str, Any]]) -> str:
    if not papers:
        return f"{title}: no papers were available to summarize."

    venues = sorted(
        {str(paper["venue"]) for paper in papers if isinstance(paper.get("venue"), str) and paper.get("venue")}
    )
    years = sorted({paper["year"] for paper in papers if isinstance(paper.get("year"), int)})
    representative_titles = [
        str(paper.get("title") or "").strip() for paper in papers[:3] if str(paper.get("title") or "").strip()
    ]
    top_terms = _top_terms(
        [
            " ".join(part for part in [str(paper.get("title") or ""), str(paper.get("abstract") or "")] if part)
            for paper in papers
        ],
        limit=5,
    )
    top_terms = [term for term in top_terms if term not in THEME_LABEL_STOPWORDS]

    venue_text = f" across {', '.join(venues[:2])}" if venues else ""
    if years:
        year_text = f" spanning {years[0]}-{years[-1]}" if len(years) > 1 else f" in {years[0]}"
    else:
        year_text = ""
    title_text = (
        f" Representative papers include {', '.join(representative_titles[:2])}." if representative_titles else ""
    )
    term_text = f" The cluster centers on {', '.join(top_terms[:3])}." if top_terms else ""
    return f"{title} groups {len(papers)} papers{venue_text}{year_text}.{title_text}{term_text}"


def _deterministic_gap_insights(evidence_papers: list[dict[str, Any]]) -> list[str]:
    if not evidence_papers:
        return []
    paper_terms = [_paper_terms(paper) for paper in evidence_papers]
    behavior_hits = sum(bool(terms & BEHAVIOR_TERMS) for terms in paper_terms)
    physiology_hits = sum(bool(terms & PHYSIOLOGY_TERMS) for terms in paper_terms)
    demography_hits = sum(bool(terms & DEMOGRAPHY_TERMS) for terms in paper_terms)
    community_hits = sum(bool(terms & COMMUNITY_TERMS) for terms in paper_terms)
    multistressor_hits = sum(bool(terms & MULTISTRESSOR_TERMS) for terms in paper_terms)
    longitudinal_hits = sum(bool(terms & LONGITUDINAL_TERMS) for terms in paper_terms)

    represented_taxa = {group for terms in paper_terms for group, cues in TAXON_GROUPS.items() if terms & cues}
    represented_geographies = {token for terms in paper_terms for token in GEO_TOKENS if token in terms}

    insights: list[str] = []
    if behavior_hits >= max(physiology_hits + demography_hits, 1):
        insights.append("behavioral responses are better covered than physiological or demographic consequences")
    if longitudinal_hits <= max(1, len(evidence_papers) // 2):
        insights.append("long-term or chronic exposure evidence is still thin")
    if community_hits <= max(1, len(evidence_papers) // 2):
        insights.append("community- and ecosystem-level impacts remain underrepresented")
    if multistressor_hits < max(1, len(evidence_papers) // 2):
        insights.append("interactions with other stressors are rarely studied directly")
    if 0 < len(represented_taxa) <= 2 and len(evidence_papers) >= 3:
        insights.append("taxonomic coverage is still narrow relative to the breadth of affected systems")
    if 0 < len(represented_geographies) <= 2 and len(evidence_papers) >= 3:
        insights.append("geographic coverage looks concentrated in a small set of regions")
    deduped: list[str] = []
    seen: set[str] = set()
    for insight in insights:
        if insight not in seen:
            seen.add(insight)
            deduped.append(insight)
    return deduped[:5]
