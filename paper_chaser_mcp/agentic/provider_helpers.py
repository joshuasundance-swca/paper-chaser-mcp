"""Pure schemas and helper functions for smart-layer provider adapters."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

from .models import ExpansionCandidate, PlannerDecision

_ResponseModelT = TypeVar("_ResponseModelT", bound=BaseModel)

TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
COMMON_QUERY_WORDS = {
    "paper",
    "papers",
    "research",
    "study",
    "studies",
    "review",
    "recent",
    "latest",
    "work",
    "works",
}
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
MAX_EMBED_TEXT_LENGTH = 6_000
THEME_LABEL_STOPWORDS = COMMON_QUERY_WORDS | {
    "effect",
    "effects",
    "impact",
    "impacts",
    "response",
    "responses",
    "change",
    "changes",
    "documenting",
    "evidence",
    "findings",
    "highlighted",
    "main",
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
_LITERATURE_PROVIDER_SET = {"semantic_scholar", "openalex", "scholarapi", "core", "arxiv"}
_REGULATORY_PROVIDER_SET = {"ecos", "federal_register", "govinfo", "tavily", "perplexity"}
_PRIMARY_SOURCE_PROVIDER_SET = {"ecos", "federal_register", "govinfo", "agency_primary_source"}
_PLANNER_INTENTS = {"discovery", "review", "known_item", "author", "citation", "regulatory"}
_ANCHOR_TYPES = {
    "cfr_citation",
    "fr_citation",
    "document_number",
    "species_common_name",
    "species_scientific_name",
    "agency_guidance_title",
    "regulatory_subject",
}
_SUCCESS_CRITERIA = {"current_text_required", "timeline_required", "dossier_required", "guidance_doc_required"}


class _PlannerConstraintsSchema(BaseModel):
    """OpenAI Structured Outputs-compatible planner constraints."""

    year: str | None = None
    venue: str | None = None
    focus: str | None = None


class _PlannerResponseSchema(BaseModel):
    """Structured planner response that avoids free-form object maps."""

    intent: Literal["discovery", "review", "known_item", "author", "citation", "regulatory"] = "discovery"
    constraints: _PlannerConstraintsSchema = Field(default_factory=_PlannerConstraintsSchema)
    seedIdentifiers: list[str] = Field(default_factory=list)
    candidateConcepts: list[str] = Field(default_factory=list)
    providerPlan: list[str] = Field(default_factory=list)
    authorityFirst: bool = True
    anchorType: str | None = None
    anchorValue: str | None = None
    requiredPrimarySources: list[str] = Field(default_factory=list)
    successCriteria: list[str] = Field(default_factory=list)
    followUpMode: Literal["qa", "claim_check", "comparison"] = "qa"

    def to_planner_decision(self) -> PlannerDecision:
        intent = self.intent if self.intent in _PLANNER_INTENTS else "discovery"
        provider_plan = _sanitize_provider_plan(intent=intent, provider_plan=self.providerPlan)
        anchor_type = self.anchorType if self.anchorType in _ANCHOR_TYPES else None
        required_primary = _sanitize_primary_sources(self.requiredPrimarySources)
        success_criteria = _sanitize_success_criteria(self.successCriteria)
        return PlannerDecision(
            intent=intent,
            constraints={key: value for key, value in self.constraints.model_dump(exclude_none=True).items() if value},
            seedIdentifiers=self.seedIdentifiers,
            candidateConcepts=self.candidateConcepts,
            providerPlan=provider_plan,
            authorityFirst=bool(self.authorityFirst),
            anchorType=anchor_type,
            anchorValue=str(self.anchorValue or "").strip() or None,
            requiredPrimarySources=required_primary,
            successCriteria=success_criteria,
            followUpMode=self.followUpMode,
        )


class _ExpansionSchema(BaseModel):
    variant: str
    source: str = Field(default="speculative")
    rationale: str = Field(default="")


class _ExpansionListSchema(BaseModel):
    expansions: list[_ExpansionSchema] = Field(default_factory=list)


class _AnswerSchema(BaseModel):
    answer: str = Field(default="")
    unsupportedAsks: list[str] = Field(default_factory=list)
    followUpQuestions: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    answerability: Literal["grounded", "limited", "insufficient"] = "limited"
    selectedEvidenceIds: list[str] = Field(default_factory=list)
    selectedLeadIds: list[str] = Field(default_factory=list)


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _normalized_embedding_text(text: str) -> str:
    return " ".join(text.split())[:MAX_EMBED_TEXT_LENGTH]


def _top_terms(texts: list[str], *, limit: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(token for token in _tokenize(text) if token not in COMMON_QUERY_WORDS)
    return [term for term, _ in counts.most_common(limit)]


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


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens: Counter[str] = Counter(_tokenize(left))
    right_tokens: Counter[str] = Counter(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = set(left_tokens) & set(right_tokens)
    numerator = sum(left_tokens[token] * right_tokens[token] for token in intersection)
    left_norm = math.sqrt(sum(value * value for value in left_tokens.values()))
    right_norm = math.sqrt(sum(value * value for value in right_tokens.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    try:
        import numpy as np
    except ImportError:
        numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    left_array = np.array(left)
    right_array = np.array(right)
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left_array, right_array) / (left_norm * right_norm))


def _normalize_confidence_label(value: Any) -> Literal["high", "medium", "low"]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"high", "medium", "low"}:
            return normalized  # type: ignore[return-value]
        if normalized in {"strong", "very_high", "very high"}:
            return "high"
        if normalized in {"moderate", "mid", "mixed"}:
            return "medium"
        if normalized in {"weak", "uncertain", "insufficient"}:
            return "low"
        try:
            numeric = float(normalized)
        except ValueError:
            numeric = None
        if numeric is not None:
            if numeric >= 0.8:
                return "high"
            if numeric >= 0.5:
                return "medium"
            return "low"
    if isinstance(value, (int, float)):
        if value >= 0.8:
            return "high"
        if value >= 0.5:
            return "medium"
        return "low"
    return "medium"


def _paper_evidence_payload(papers: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return [
        {
            "paperId": paper.get("paperId") or paper.get("sourceId") or paper.get("canonicalId"),
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
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
    limit: int = 6,
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
    valid_sources = {"from_input", "from_retrieved_evidence", "speculative"}
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


def _langchain_message_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    if isinstance(response, BaseModel):
        return response.model_dump_json()
    return str(response).strip()


def _extract_json_object(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return None
    start = text.find("{")
    depth = 0
    for index, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1].strip()
    return None


def _normalize_label_text(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"^(theme|label)\s*:\s*", "", line, flags=re.IGNORECASE)
        line = line.strip().strip('"').strip("'")
        if line:
            return line
    return ""


def _normalize_theme_label_output(text: str) -> str:
    label = _normalize_label_text(text)
    if label:
        return label
    return text.strip().strip('"').strip("'")


def _coerce_langchain_structured_response(
    response: Any,
    response_model: type[_ResponseModelT],
) -> _ResponseModelT:
    if isinstance(response, response_model):
        return response
    if isinstance(response, BaseModel):
        return response_model.model_validate(response.model_dump())
    if isinstance(response, dict):
        return response_model.model_validate(response)
    text = _langchain_message_text(response)
    if text:
        try:
            return response_model.model_validate_json(text)
        except Exception:
            json_payload = _extract_json_object(text)
            if json_payload:
                return response_model.model_validate_json(json_payload)
    raise ValueError("LangChain provider did not return structured output.")


def _extract_seed_identifiers(query: str) -> list[str]:
    identifiers: list[str] = []
    for pattern in (
        r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
        r"(arxiv:\d{4}\.\d{4,5}(?:v\d+)?)",
        r"((?:https?://)[^\s]+)",
        r"(\d{4}\.\d{4,5}(?:v\d+)?)",
    ):
        for match in re.findall(pattern, query, flags=re.IGNORECASE):
            identifiers.append(str(match))
    seen: set[str] = set()
    deduped: list[str] = []
    for identifier in identifiers:
        if identifier in seen:
            continue
        seen.add(identifier)
        deduped.append(identifier)
    return deduped


def _normalize_answer_schema_output(
    *,
    parsed_answer: _AnswerSchema,
    evidence_papers: list[dict[str, Any]],
    confidence_normalizer: Any,
) -> dict[str, Any]:
    payload = parsed_answer.model_dump()
    payload["confidence"] = confidence_normalizer(payload.get("confidence"))
    valid_evidence_ids = _collect_evidence_ids(evidence_papers)
    selected_evidence_ids = [
        str(identifier).strip()
        for identifier in payload.get("selectedEvidenceIds") or []
        if str(identifier).strip() in valid_evidence_ids
    ]
    payload["selectedEvidenceIds"] = selected_evidence_ids
    payload["selectedLeadIds"] = [str(identifier).strip() for identifier in payload.get("selectedLeadIds") or []]
    if payload.get("answerability") == "grounded" and not selected_evidence_ids and evidence_papers:
        payload["answerability"] = "limited"
    return payload


def _collect_evidence_ids(evidence_papers: list[dict[str, Any]]) -> set[str]:
    valid_ids: set[str] = set()
    for paper in evidence_papers:
        if not isinstance(paper, dict):
            continue
        for key in ("paperId", "sourceId", "canonicalId"):
            value = str(paper.get(key) or "").strip()
            if value:
                valid_ids.add(value)
    return valid_ids


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
