"""Phase 7c-1: variant combination and near-duplicate reconciliation helpers.

Extracted from ``planner/_core.py`` so the orchestrator can focus on the
async classification/expansion flow and the variant deduplication math can be
exercised in isolation. Public callers should continue to import these symbols
from ``paper_chaser_mcp.agentic.planner`` (facade) or, for tests, from either
this submodule or ``planner._core`` (identity is preserved).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..config import AgenticConfig
from ..models import ExpansionCandidate
from .constants import GENERIC_EVIDENCE_WORDS, VARIANT_DEDUPE_STOPWORDS
from .normalization import normalize_query


def combine_variants(
    *,
    original_query: str,
    grounded: list[ExpansionCandidate],
    speculative: list[ExpansionCandidate],
    config: AgenticConfig,
) -> list[ExpansionCandidate]:
    """Return the capped variant list in grounded-first order."""
    variants = [
        ExpansionCandidate(
            variant=normalize_query(original_query),
            source="from_input",
            rationale="Literal user query.",
        )
    ]
    variants.extend(grounded[: config.max_grounded_variants])
    variants.extend(speculative[: config.max_speculative_variants])

    deduped: list[ExpansionCandidate] = []
    seen: set[str] = set()
    seen_signatures: list[frozenset[str]] = []
    for candidate in variants:
        lowered = candidate.variant.lower()
        if lowered in seen:
            continue
        signature = _variant_signature(candidate.variant)
        if any(_signatures_are_near_duplicates(signature, prior) for prior in seen_signatures):
            continue
        seen.add(lowered)
        seen_signatures.append(signature)
        deduped.append(candidate)
        if len(deduped) >= config.max_total_variants:
            break
    return deduped


def dedupe_variants(
    candidates: list[ExpansionCandidate],
    *,
    config: AgenticConfig,
) -> list[ExpansionCandidate]:
    """Apply the same near-duplicate suppression used by final variant planning."""

    deduped: list[ExpansionCandidate] = []
    seen: set[str] = set()
    seen_signatures: list[frozenset[str]] = []
    for candidate in candidates:
        lowered = candidate.variant.lower()
        if lowered in seen:
            continue
        signature = _variant_signature(candidate.variant)
        if any(_signatures_are_near_duplicates(signature, prior) for prior in seen_signatures):
            continue
        seen.add(lowered)
        seen_signatures.append(signature)
        deduped.append(candidate)
        if len(deduped) >= config.max_total_variants:
            break
    return deduped


def _variant_signature(text: str) -> frozenset[str]:
    tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9]{3,}", normalize_query(text).lower())
        if token not in VARIANT_DEDUPE_STOPWORDS
    }
    return frozenset(tokens)


def _signatures_are_near_duplicates(
    left: frozenset[str],
    right: frozenset[str],
) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    overlap = len(left & right)
    shorter = min(len(left), len(right))
    longer = max(len(left), len(right))
    if shorter == 0:
        return False
    coverage = overlap / shorter
    jaccard = overlap / len(left | right)
    return coverage >= 0.8 or (coverage >= 0.67 and jaccard >= 0.5 and longer <= 8)


def _top_evidence_phrases(
    papers: list[dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    phrases: Counter[str] = Counter()
    for paper in papers[:10]:
        title = str(paper.get("title") or "")
        per_paper_phrases: set[str] = set()
        title_words = re.findall(r"[A-Za-z0-9]{2,}", title.lower())
        for index in range(len(title_words) - 1):
            left = title_words[index]
            right = title_words[index + 1]
            if left in GENERIC_EVIDENCE_WORDS or right in GENERIC_EVIDENCE_WORDS or len(left) < 3 or len(right) < 3:
                continue
            bigram = f"{left} {right}"
            if len(bigram) < 9:
                continue
            per_paper_phrases.add(bigram)
        phrases.update(per_paper_phrases)
    scored = [(phrase, count) for phrase, count in phrases.items() if count >= 2]
    scored.sort(
        key=lambda item: (
            item[0].count(" "),
            item[1],
            len(item[0]),
        ),
        reverse=True,
    )
    return [phrase for phrase, _ in scored[:limit]]
