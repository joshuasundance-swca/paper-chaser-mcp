"""Guided reference-resolution helpers (Phase 3 extraction).

Small helpers that produce guided ``resolve_reference`` / clarification
payloads. Extracted from :mod:`paper_chaser_mcp.dispatch._core`.
"""

from __future__ import annotations

import re
from typing import Any

from ...agentic.planner import (
    detect_regulatory_intent,
    looks_like_exact_title,
)
from ...citation_repair import (
    looks_like_citation_query,
    looks_like_paper_identifier,
    parse_citation,
)
from ..normalization import _guided_normalize_whitespace

from .._core import (  # noqa: E402 — forward refs
    _GUIDED_REFERENCE_UNCERTAINTY_MARKERS,
)
from .strategy_metadata import _guided_reference_signal_words



def _guided_note_repair(
    repairs: list[dict[str, str]],
    *,
    field: str,
    original: Any,
    normalized: Any,
    reason: str,
) -> None:
    if original == normalized:
        return
    repairs.append(
        {
            "field": field,
            "from": str(original if original is not None else ""),
            "to": str(normalized if normalized is not None else ""),
            "reason": reason,
        }
    )
def _guided_underspecified_reference_clarification(
    *,
    query: str,
    focus: str | None,
) -> dict[str, Any] | None:
    combined = _guided_normalize_whitespace(" ".join(part for part in [query, focus or ""] if part))
    if not combined or looks_like_paper_identifier(combined):
        return None
    parsed = parse_citation(combined)
    if parsed.identifier:
        return None

    citation_like = bool(
        parsed.year is not None or looks_like_citation_query(combined) or looks_like_exact_title(combined)
    )
    if not citation_like:
        return None

    if parsed.author_surnames or parsed.venue_hints:
        return None

    strongest_candidate_words = max(
        (len(_guided_reference_signal_words(candidate)) for candidate in parsed.title_candidates),
        default=0,
    )
    weak_anchor = strongest_candidate_words <= 4
    uncertainty_hits = sum(
        1
        for marker in _GUIDED_REFERENCE_UNCERTAINTY_MARKERS
        if re.search(rf"\b{re.escape(marker)}\b", combined, re.IGNORECASE)
    )
    if not weak_anchor or (uncertainty_hits == 0 and not parsed.looks_like_non_paper):
        return None

    if parsed.looks_like_non_paper or detect_regulatory_intent(query, focus):
        return {
            "reason": "underspecified_reference_fragment",
            "question": (
                "This looks like a vague reference fragment and may point to either a paper or a policy-style "
                "document. Add an exact title, one author surname, an agency or venue, or confirm which type "
                "of source you want before the server guesses."
            ),
            "options": [
                "add exact title",
                "add author surname",
                "add agency or venue",
                "paper vs policy source",
            ],
            "canProceedWithoutAnswer": True,
        }
    return {
        "reason": "underspecified_reference_fragment",
        "question": (
            "This looks like a vague paper/reference fragment. Add an exact title, one author surname, or a "
            "venue/year clue before guided research infers a likely paper from weak hints."
        ),
        "options": [
            "add exact title",
            "add author surname",
            "add venue or year",
            "use resolve_reference",
        ],
        "canProceedWithoutAnswer": True,
    }
