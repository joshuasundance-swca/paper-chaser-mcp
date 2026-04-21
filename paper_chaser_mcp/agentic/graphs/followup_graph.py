"""Follow-up / ``ask_result_set`` helpers extracted in Phase 7b from ``graphs/_core``.

These pure helpers back the comparison-answer and follow-up question
contextualization paths used by ``AgenticRuntime.ask_result_set`` and the
planner's follow-up grounding. The orchestrating class methods stay on
``AgenticRuntime`` because they thread through the saved result-set store,
provider fan-out, and context events; the structured-text shaping moves
here so it can be tested without instantiating a full runtime.
"""

from __future__ import annotations

import re
from typing import Any

from .hooks import _truncate_text
from .research_graph import _graph_intent_text
from .shared_state import _COMPARISON_FOCUS_STOPWORDS, _COMPARISON_MARKERS
from .source_records import _graph_topic_tokens, _paper_text


def _comparison_requested(question: str, answer_mode: str) -> bool:
    if answer_mode == "comparison":
        return True
    question_tokens = set(re.findall(r"[a-z0-9]{2,}", question.lower()))
    return bool(question_tokens & _COMPARISON_MARKERS)


def _looks_like_title_venue_list(answer_text: str, evidence_papers: list[dict[str, Any]]) -> bool:
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    if not lines:
        return True
    bullet_lines = [line for line in lines if line.startswith("- ")]
    if len(bullet_lines) < min(2, len(evidence_papers)):
        return False
    matched_lines = 0
    normalized_titles = [str(paper.get("title") or paper.get("paperId") or "").strip() for paper in evidence_papers[:4]]
    for line in bullet_lines[:4]:
        lower_line = line.lower()
        has_title = any(title and title in line for title in normalized_titles)
        has_weak_metadata_pattern = any(marker in lower_line for marker in ("venue", "year", "unknown")) or bool(
            re.search(r":\s*[^\n,]+,\s*(19|20)\d{2}\b", line)
        )
        if has_title and has_weak_metadata_pattern:
            matched_lines += 1
    return matched_lines >= min(2, len(bullet_lines))


def _should_use_structured_comparison_answer(
    *,
    question: str,
    answer_mode: str,
    answer_text: str,
    evidence_papers: list[dict[str, Any]],
) -> bool:
    del answer_text, evidence_papers
    return _comparison_requested(question, answer_mode)


def _build_grounded_comparison_answer(
    *,
    question: str,
    evidence_papers: list[dict[str, Any]],
) -> str:
    papers = evidence_papers[: min(3, len(evidence_papers))]
    if not papers:
        return "The saved result set does not contain enough evidence to make a grounded comparison."
    shared_terms = _shared_focus_terms(papers, question=question)
    shared_ground = (
        ", ".join(term.title() for term in shared_terms[:3])
        if shared_terms
        else "closely related problem settings from the saved result set"
    )
    detail_lines = []
    for paper in papers:
        title = str(paper.get("title") or paper.get("paperId") or "Untitled")
        year = paper.get("year")
        venue = str(paper.get("venue") or "venue not stated")
        descriptor = _paper_focus_phrase(paper, question=question)
        timing = str(year) if isinstance(year, int) else "year unknown"
        detail_lines.append(f"- {title} ({timing}; {venue}) emphasizes {descriptor}.")
    takeaway = _comparison_takeaway(papers, shared_terms)
    return "\n".join(
        [
            "Grounded comparison from the saved result set.",
            f"Shared ground: these papers converge on {shared_ground}.",
            "Key differences:",
            *detail_lines,
            f"Takeaway: {takeaway}",
        ]
    )


def _shared_focus_terms(papers: list[dict[str, Any]], *, question: str) -> list[str]:
    counts: dict[str, int] = {}
    question_tokens = set(_graph_topic_tokens(question))
    for paper in papers:
        for token in _graph_topic_tokens(_paper_text(paper)):
            if token in _COMPARISON_FOCUS_STOPWORDS or token in question_tokens or token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 1
    minimum_count = 2 if len(papers) >= 2 else 1
    return [
        token for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if count >= minimum_count
    ]


def _paper_focus_phrase(paper: dict[str, Any], *, question: str) -> str:
    question_tokens = set(_graph_topic_tokens(question))
    focus_tokens: list[str] = []
    for source_text in (str(paper.get("title") or ""), str(paper.get("abstract") or "")):
        for token in re.findall(r"[a-z0-9]{3,}", source_text.lower()):
            if token in _COMPARISON_FOCUS_STOPWORDS or token in question_tokens:
                continue
            if token in focus_tokens:
                continue
            focus_tokens.append(token)
            if len(focus_tokens) >= 3:
                break
        if len(focus_tokens) >= 3:
            break
    if focus_tokens:
        return ", ".join(focus_tokens)
    abstract = str(paper.get("abstract") or "").strip()
    if abstract:
        return _truncate_text(abstract.lower(), limit=96)
    return "the same core topic from a different angle"


def _comparison_takeaway(papers: list[dict[str, Any]], shared_terms: list[str]) -> str:
    years: list[int] = []
    for paper in papers:
        year = paper.get("year")
        if isinstance(year, int):
            years.append(year)
    venues = [str(paper.get("venue") or "").strip() for paper in papers if str(paper.get("venue") or "").strip()]
    if years and max(years) != min(years):
        return (
            f"the papers stay grounded in {', '.join(term.title() for term in shared_terms[:2]) or 'the same topic'}, "
            "but they span different publication periods, so they likely reflect different stages of the literature."
        )
    if len(set(venues)) > 1:
        venue_list = ", ".join(sorted(set(venues))[:2])
        return (
            "the main contrast is not the core topic but the research setting, "
            f"with evidence spread across {venue_list}."
        )
    return (
        "the papers are topically close, but they contribute different emphases, methods, or evaluation perspectives."
    )


def _contextualize_follow_up_question(
    *,
    question: str,
    record: Any | None,
    question_mode: str,
) -> str:
    normalized_question = str(question or "").strip()
    if question_mode not in {"comparison", "selection"}:
        return normalized_question
    session_intent = _graph_intent_text(record, [])
    if not session_intent:
        return normalized_question
    lowered_question = normalized_question.lower()
    lowered_intent = session_intent.lower()
    if lowered_intent and lowered_intent in lowered_question:
        return normalized_question
    if not normalized_question:
        return session_intent
    return f"{normalized_question} about {session_intent}"
