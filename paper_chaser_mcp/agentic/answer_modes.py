"""Follow-up answer-mode classification and evidence-use planning.

This module centralizes the semantics that keep grounded follow-up answers
honest. It exposes:

* ``ANSWER_MODES`` -- the canonical set of follow-up answer modes.
* ``classify_question_mode`` -- deterministic keyword classifier.
* ``build_evidence_use_plan`` -- LLM-signal-aware synthesis-plan builder.
* ``evidence_pool_is_weak`` -- guard that flags pools dominated by weak_match /
  off_topic classifications.

The implementation reuses the LLM relevance classifications produced inside
``ask_result_set`` (via ``provider_bundle.aclassify_relevance_batch``) rather
than making a separate LLM call, so the evidence-sufficiency signals we rely on
carry real model provenance (with ``fallback=True`` when the LLM was
unavailable).
"""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from .models import EvidenceItem, StructuredSourceRecord

# Type aliases for optional LLM-backed classification.
#
# ``LLMModeClassifier`` is a synchronous callable that takes the raw question
# plus the allowed mode list and returns one of ``ANSWER_MODES`` (or ``None``
# if the LLM abstains / the call fails). ``AsyncLLMModeClassifier`` is the
# awaitable sibling used by ``aclassify_question_mode``.
LLMModeClassifier = Callable[[str, tuple[str, ...]], str | None]
AsyncLLMModeClassifier = Callable[[str, tuple[str, ...]], Awaitable[str | None]]

ANSWER_MODES: tuple[str, ...] = (
    "metadata",
    "relevance_triage",
    "comparison",
    "selection",
    "mechanism_summary",
    "regulatory_chain",
    "intervention_tradeoff",
    "unknown",
)

# Modes that require synthesis over multiple on-topic sources. A polished answer
# in any of these modes without corroborating non-fallback evidence is exactly
# the failure we are trying to prevent.
SYNTHESIS_MODES: frozenset[str] = frozenset(
    {"comparison", "selection", "mechanism_summary", "regulatory_chain", "intervention_tradeoff"}
)

# Modes where deterministic salvage over saved session state is acceptable when
# the LLM is unavailable.
SALVAGEABLE_MODES: frozenset[str] = frozenset({"metadata", "relevance_triage"})

_WEAK_POOL_THRESHOLD_ENV = "PAPER_CHASER_WEAK_EVIDENCE_POOL_THRESHOLD"
_DEFAULT_WEAK_POOL_THRESHOLD = 0.6

# Minimum number of non-fallback on_topic sources required to support a
# synthesis-mode answer.
_SYNTHESIS_MIN_RESPONSIVE = 2
_STRONG_SELECTION_VERIFICATION_STATES = frozenset({"verified_primary_source", "verified_metadata"})
_SELECTION_CURRENT_TEXT_MARKERS: tuple[str, ...] = (
    "current text",
    "current codified",
    "codified text",
    "current codified text",
    "current cfr",
    "cfr text",
    "regulatory text",
    "rule text",
    "official text",
)
_SELECTION_PRIMARY_SOURCE_MARKERS: tuple[str, ...] = (
    "cfr",
    "code of federal regulations",
    "codified",
    "regulatory text",
    "current text",
)
_CFR_REFERENCE_RE = re.compile(r"\b\d+\s*cfr\s*\d+(?:\.\d+)*\b", re.IGNORECASE)


def _weak_pool_threshold() -> float:
    raw = os.environ.get(_WEAK_POOL_THRESHOLD_ENV, "").strip()
    if not raw:
        return _DEFAULT_WEAK_POOL_THRESHOLD
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_WEAK_POOL_THRESHOLD
    if not 0.0 < value <= 1.0:
        return _DEFAULT_WEAK_POOL_THRESHOLD
    return value


def classify_question_mode(
    question: str,
    session_metadata: dict[str, Any] | None = None,
    *,
    llm_classifier: LLMModeClassifier | None = None,
    classifier_cache: dict[str, str] | None = None,
) -> str:
    """Classify a follow-up question into one of ``ANSWER_MODES``.

    Resolution order (fail-closed; the first signal we trust wins):

    1. A planner/session ``followUpMode`` hint when it maps to a concrete
       answer mode. This is the LLM-native signal already emitted by the
       planner, so we prefer it over every other heuristic.
    2. An optional LLM-backed classifier (``llm_classifier``) when supplied.
       This is the LLM-first primary path: it catches paraphrased synthesis
       questions that the deterministic keyword table would otherwise miss.
       Results are memoised in ``classifier_cache`` (keyed on the raw
       question) so retries and subsequent follow-ups do not pay for another
       LLM call.
    3. The deterministic keyword heuristic (preserved as the always-available
       fallback).
    4. ``"unknown"`` when nothing else matches.

    The LLM classifier is treated as advisory: a return value outside
    ``ANSWER_MODES`` is ignored and we continue to the keyword fallback.
    """
    normalised_question = (question or "").strip()
    session_hint = _resolve_session_followup_mode(session_metadata)
    if session_hint is not None:
        return session_hint

    if llm_classifier is not None and normalised_question:
        cached: str | None = None
        if classifier_cache is not None:
            cached_value = classifier_cache.get(normalised_question)
            if isinstance(cached_value, str) and cached_value in ANSWER_MODES:
                cached = cached_value
        if cached is not None:
            return cached
        try:
            llm_mode = llm_classifier(normalised_question, ANSWER_MODES)
        except Exception:
            llm_mode = None
        if isinstance(llm_mode, str):
            candidate = llm_mode.strip().lower()
            if candidate in ANSWER_MODES and candidate != "unknown":
                if classifier_cache is not None:
                    classifier_cache[normalised_question] = candidate
                return candidate

    return _classify_question_mode_keyword(normalised_question)


async def aclassify_question_mode(
    question: str,
    session_metadata: dict[str, Any] | None = None,
    *,
    llm_classifier: AsyncLLMModeClassifier | None = None,
    classifier_cache: dict[str, str] | None = None,
) -> str:
    """Async counterpart to :func:`classify_question_mode`.

    Mirrors the synchronous resolution order but awaits ``llm_classifier``
    when one is provided. Exceptions from the LLM coroutine are swallowed so
    that safety-critical gating never crashes on a provider hiccup -- we
    simply fall through to the keyword heuristic.
    """
    normalised_question = (question or "").strip()
    session_hint = _resolve_session_followup_mode(session_metadata)
    if session_hint is not None:
        return session_hint

    if llm_classifier is not None and normalised_question:
        if classifier_cache is not None:
            cached_value = classifier_cache.get(normalised_question)
            if isinstance(cached_value, str) and cached_value in ANSWER_MODES:
                return cached_value
        try:
            llm_mode = await llm_classifier(normalised_question, ANSWER_MODES)
        except Exception:
            llm_mode = None
        if isinstance(llm_mode, str):
            candidate = llm_mode.strip().lower()
            if candidate in ANSWER_MODES and candidate != "unknown":
                if classifier_cache is not None:
                    classifier_cache[normalised_question] = candidate
                return candidate

    return _classify_question_mode_keyword(normalised_question)


def _resolve_session_followup_mode(session_metadata: dict[str, Any] | None) -> str | None:
    """Extract a concrete answer mode from a planner/session hint.

    Returns ``None`` when no usable hint is available (including the generic
    ``"qa"`` planner default, which just means "no opinion").
    """
    if not session_metadata:
        return None
    raw = session_metadata.get("followUpMode")
    if not isinstance(raw, str):
        return None
    hint = raw.strip().lower()
    if not hint or hint == "qa":
        return None
    if hint in ANSWER_MODES and hint != "unknown":
        return hint
    if hint == "claim_check":
        # Historic alias -- treat claim-check prompts as mechanism summaries
        # so they route through synthesis-mode gating.
        return "mechanism_summary"
    return None


def _classify_question_mode_keyword(question: str) -> str:
    """Deterministic keyword heuristic (the legacy classifier).

    Preserved as the always-available fallback. The public
    :func:`classify_question_mode` entry point consults this only after the
    planner hint and optional LLM classifier have been considered.
    """
    lowered = (question or "").lower()
    if _looks_like_relevance_triage(lowered):
        return "relevance_triage"
    # Selection markers are checked before the generic compare/versus branch so
    # single-best questions ("which is best", "most recent", "beginner-friendly")
    # route to ``selection`` rather than being swallowed by the broader
    # comparison bucket. Explicit "compare X and Y" phrasing still wins because
    # neither "compare" nor "versus" appears in the selection marker list.
    if any(
        marker in lowered
        for marker in (
            "which is best",
            "which one is best",
            "best starting point",
            "most suitable",
            "beginner-friendly",
            "beginner friendly",
            "most recent",
            "most authoritative",
            "most cited",
            "most influential",
            "easiest to",
            "should i start with",
            "start with first",
            "which source should i start with",
            "which paper should i start with",
            "where should i start",
            "which source should i read first",
            "which paper should i read first",
            "read first",
            "most accessible",
            "most up to date",
            "most up-to-date",
            "latest paper",
            "which should i",
        )
    ):
        return "selection"
    metadata_facets = _metadata_facets(lowered)
    if metadata_facets:
        return "metadata"
    if any(marker in lowered for marker in ("compare", "versus", " vs ", " vs.", "tradeoff", "trade-off")):
        if "trade" in lowered and "intervent" in lowered:
            return "intervention_tradeoff"
        return "comparison"
    if any(marker in lowered for marker in ("mechanism", "pathway", "causal", "how does", "why does")):
        return "mechanism_summary"
    if any(
        marker in lowered
        for marker in (
            "regulatory history",
            "rulemaking",
            "listing history",
            "timeline",
            "critical habitat",
            "final rule",
            "proposed rule",
        )
    ):
        return "regulatory_chain"
    if any(
        marker in lowered
        for marker in (
            "practical implication",
            "practical implications",
            "intervention tradeoff",
            "policy tradeoff",
            "cost vs",
            "benefit vs",
        )
    ):
        return "intervention_tradeoff"
    return "unknown"


# Phrase-level cues for paraphrased synthesis questions the keyword classifier
# is not designed to catch (e.g. "summarize the findings", "walk me through
# what's known", "what does the literature conclude"). Used by the
# fail-closed branch in :func:`build_evidence_use_plan` so that unknown-mode
# questions over a weak evidence pool abstain instead of slipping through.
_SYNTHESIS_SHAPE_MARKERS: tuple[str, ...] = (
    "summarize",
    "summarise",
    "summary of",
    "synthes",
    "literature",
    "findings",
    "conclude",
    "conclusion",
    "overall",
    "walk me through",
    "walk through",
    "what's known",
    "what is known",
    "known about",
    "compare",
    "contrast",
    "versus",
    "explain",
    "overview",
    "consensus",
    "state of the art",
    "state-of-the-art",
)


def _looks_like_synthesis_question(question: str) -> bool:
    """Return True when ``question`` exhibits synthesis-shape cues.

    Requires an actual synthesis-style phrase -- a bare question mark is NOT
    sufficient, because routine Q&A questions use ``?`` too and we do not
    want the fail-closed branch to swallow every unclassified question over
    a weak-ish pool. Only consulted inside the unknown-mode fail-closed
    branch, so false positives there collapse to an ``insufficient`` verdict
    rather than a wrong mode.
    """
    lowered = (question or "").strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _SYNTHESIS_SHAPE_MARKERS)


def _looks_like_relevance_triage(lowered_question: str) -> bool:
    markers = (
        "which of these",
        "which papers",
        "which sources",
        "are any of these",
        "which entries",
        "which results",
        "relevant to",
        "off-topic",
        "off topic",
    )
    return any(marker in lowered_question for marker in markers)


def _metadata_facets(lowered_question: str) -> list[str]:
    facets: list[str] = []
    if any(marker in lowered_question for marker in ("author", "authors", "who wrote", "written by")):
        facets.append("authors")
    if any(marker in lowered_question for marker in ("venue", "journal", "publisher", "published in")):
        facets.append("venue")
    if any(marker in lowered_question for marker in ("doi", "identifier")):
        facets.append("identifier")
    if any(marker in lowered_question for marker in ("publication year", "what year", "year published")):
        facets.append("year")
    if any(
        marker in lowered_question for marker in ("what records", "what sources", "which documents", "what documents")
    ):
        facets.append("inventory")
    return facets


def selection_anchor_candidate_ids(
    question: str,
    evidence: list[EvidenceItem],
    source_records: list[StructuredSourceRecord],
) -> list[str]:
    """Return strong exact-anchor candidates for safe selection follow-ups."""
    lowered_question = (question or "").strip().lower()
    if not lowered_question:
        return []
    wants_current_text = any(marker in lowered_question for marker in _SELECTION_CURRENT_TEXT_MARKERS)
    cfr_references = {
        re.sub(r"\s+", " ", match.group(0).lower()).strip()
        for match in _CFR_REFERENCE_RE.finditer(question or "")
    }
    candidate_ids: list[str] = []
    for item, source in zip(evidence, source_records, strict=False):
        evidence_id = str(item.evidence_id or item.paper.paper_id or item.paper.canonical_id or "").strip()
        if not evidence_id or source.topical_relevance != "on_topic":
            continue
        if str(source.verification_status or "").strip() not in _STRONG_SELECTION_VERIFICATION_STATES:
            continue
        provider_text = " ".join(
            part
            for part in (
                source.title,
                source.citation_text,
                source.source_id,
                source.canonical_url,
                source.retrieved_url,
                item.paper.title,
            )
            if isinstance(part, str) and part.strip()
        ).lower()
        is_primary = (
            bool(source.is_primary_source)
            or str(source.verification_status or "").strip() == "verified_primary_source"
            or "primary" in str(source.source_type or "").lower()
        )
        cfr_match = any(reference in provider_text for reference in cfr_references)
        current_text_match = wants_current_text and is_primary and any(
            marker in provider_text for marker in _SELECTION_PRIMARY_SOURCE_MARKERS
        )
        if cfr_match or current_text_match:
            candidate_ids.append(evidence_id)
    return list(dict.fromkeys(candidate_ids))


def _is_on_topic(source: StructuredSourceRecord) -> bool:
    return getattr(source, "topical_relevance", None) == "on_topic"


def _is_weak_or_off_topic(source: StructuredSourceRecord) -> bool:
    label = getattr(source, "topical_relevance", None)
    return label in {"weak_match", "off_topic"}


def _classification_for(evidence_id: str, llm_relevance: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    if not llm_relevance or not evidence_id:
        return {}
    entry = llm_relevance.get(evidence_id)
    if isinstance(entry, dict):
        return entry
    return {}


def evidence_pool_is_weak(
    source_records: list[StructuredSourceRecord],
    *,
    threshold: float | None = None,
) -> bool:
    """Return True when the saved evidence pool is dominated by weak/off-topic hits.

    The threshold defaults to 60% and can be overridden via
    ``PAPER_CHASER_WEAK_EVIDENCE_POOL_THRESHOLD``.
    """
    if not source_records:
        return False
    effective_threshold = threshold if threshold is not None else _weak_pool_threshold()
    weak = sum(1 for record in source_records if _is_weak_or_off_topic(record))
    return (weak / len(source_records)) > effective_threshold


def build_evidence_use_plan(
    *,
    question: str,
    answer_mode: str,
    evidence: list[EvidenceItem],
    source_records: list[StructuredSourceRecord],
    unsupported_asks: list[str],
    llm_relevance: dict[str, dict[str, Any]] | None = None,
    question_mode: str | None = None,
) -> dict[str, Any]:
    """Build a synthesis plan with evidence-sufficiency signals.

    Produces a dict with the existing ``EvidenceUsePlan`` keys plus additive
    diagnostic fields used by ``ask_result_set``:

    * ``answerMode`` -- one of ``ANSWER_MODES``.
    * ``responsiveEvidenceIds`` -- ids classified as on_topic by the LLM
      relevance pass, excluding entries where the LLM call fell back to a
      deterministic heuristic.
    * ``unsupportedAspects`` -- plain-language gaps (echoed from
      ``unsupported_asks`` plus mode-specific diagnostics).
    * ``sufficient`` -- overall gate: ``True`` only when the mode's evidence
      threshold is satisfied and ``unsupported_asks`` is not large.
    * ``rationale`` -- short string describing why the plan is/isn't sufficient.
    * ``answerSubtype`` / ``directlyResponsiveIds`` /
      ``unsupportedComponents`` / ``retrievalSufficiency`` / ``confidence`` --
      back-compat keys for the existing ``EvidenceUsePlan`` model.
    * ``evidenceQualityProfile`` -- coarse label ``high``/``medium``/``low``.
    * ``synthesisMode`` -- coarse label suitable for confidence signals.
    """
    mode = question_mode or classify_question_mode(question)
    # Treat claim_check (historic answer_mode) as synthesis.
    treat_as_synthesis = mode in SYNTHESIS_MODES or answer_mode in {"claim_check", "comparison"}

    responsive_ids: list[str] = []
    fallback_only_on_topic = 0
    on_topic_count = 0
    anchored_selection_ids = (
        selection_anchor_candidate_ids(question, evidence, source_records) if mode == "selection" else []
    )
    for item, source in zip(evidence, source_records, strict=False):
        evidence_id = str(item.evidence_id or item.paper.paper_id or item.paper.canonical_id or "").strip()
        if not evidence_id:
            continue
        on_topic = _is_on_topic(source) and float(item.relevance_score) >= 0.25
        if not on_topic:
            continue
        on_topic_count += 1
        classification = _classification_for(evidence_id, llm_relevance)
        is_fallback = bool(classification.get("fallback"))
        if is_fallback:
            fallback_only_on_topic += 1
            continue
        responsive_ids.append(evidence_id)

    unsupported_components: list[str] = [component for component in (unsupported_asks or []) if str(component).strip()]

    sufficient = True
    retrieval_sufficiency: Literal["sufficient", "thin", "insufficient"] = "sufficient"
    confidence: Literal["high", "medium", "low"] = "medium"
    rationale = "Evidence pool supports the requested synthesis."

    if treat_as_synthesis:
        if mode == "selection" and len(anchored_selection_ids) == 1 and not unsupported_components:
            retrieval_sufficiency = "sufficient"
            confidence = "medium"
            rationale = "Selection follow-up is anchored to one exact strong source in the saved evidence."
        elif len(responsive_ids) >= _SYNTHESIS_MIN_RESPONSIVE and len(unsupported_components) <= 1:
            retrieval_sufficiency = "sufficient"
            confidence = "high"
            rationale = f"{len(responsive_ids)} non-fallback on-topic sources support the requested {mode}."
        elif len(responsive_ids) == 1 and mode != "comparison" and not unsupported_components:
            retrieval_sufficiency = "thin"
            confidence = "low"
            sufficient = False
            rationale = "Only one non-fallback on-topic source is available; synthesis would lean on a single paper."
            unsupported_components.append(
                "Only one directly responsive source is available for this synthesis request."
            )
        else:
            retrieval_sufficiency = "insufficient"
            confidence = "low"
            sufficient = False
            if mode == "comparison":
                rationale = "Comparison requested, but fewer than two non-fallback on-topic sources were retrieved."
                unsupported_components.append("Comparison requires at least two directly responsive sources.")
            else:
                rationale = f"{mode} requested, but the saved evidence does not directly support the synthesis."
                unsupported_components.append("The saved evidence does not directly support the requested synthesis.")
            if fallback_only_on_topic:
                unsupported_components.append(
                    f"{fallback_only_on_topic} on-topic classification(s) relied on a deterministic "
                    "relevance fallback; treat them as leads rather than grounded support."
                )
    else:
        if mode == "metadata":
            retrieval_sufficiency = "sufficient" if source_records else "insufficient"
            confidence = "high" if source_records else "low"
            sufficient = bool(source_records)
            rationale = (
                "Metadata question; answerable from the saved session inventory."
                if source_records
                else "Metadata question with no saved sources to inspect."
            )
        elif mode == "relevance_triage":
            retrieval_sufficiency = "sufficient" if source_records else "insufficient"
            confidence = "medium" if source_records else "low"
            sufficient = bool(source_records)
            rationale = "Relevance triage can run against the saved record even without fresh synthesis."
        elif mode == "unknown":
            # Generic Q&A / synthesis questions we cannot confidently route.
            #
            # Fail-closed safety gate (ws-followup-llm-gate): if the question
            # looks like a synthesis ask (paraphrased "summarize", "what does
            # the literature conclude", comparatives, any question mark) AND
            # the saved evidence pool is dominated by weak/off-topic hits,
            # refuse to declare the plan sufficient. Previously the unknown
            # branch unconditionally returned ``sufficient=True`` and relied on
            # downstream heuristics, which is brittle in exactly the case where
            # the keyword classifier misses a paraphrased synthesis question.
            synthesis_shaped = _looks_like_synthesis_question(question)
            weak_pool = evidence_pool_is_weak(source_records)
            if synthesis_shaped and weak_pool:
                retrieval_sufficiency = "insufficient"
                confidence = "low"
                sufficient = False
                rationale = (
                    "Unclassified follow-up with synthesis-shape cues over a weak evidence pool; "
                    "failing closed rather than producing a polished answer from weak matches."
                )
                unsupported_components.append(
                    "The saved evidence is dominated by weak/off-topic matches; "
                    "the requested synthesis is not grounded."
                )
            elif on_topic_count >= 1:
                retrieval_sufficiency = "sufficient"
                confidence = "medium"
                sufficient = True
                rationale = (
                    "Unclassified follow-up with on-topic evidence; deferring to downstream answer-status heuristics."
                )
            else:
                retrieval_sufficiency = "thin"
                confidence = "low"
                sufficient = True
                rationale = (
                    "Unclassified follow-up with no on-topic evidence; "
                    "deferring to downstream answer-status heuristics."
                )

    # De-duplicate while preserving order.
    unsupported_components = list(dict.fromkeys(component for component in unsupported_components if component))

    quality_profile: Literal["high", "medium", "low"]
    if retrieval_sufficiency == "sufficient" and confidence == "high":
        quality_profile = "high"
    elif retrieval_sufficiency == "insufficient":
        quality_profile = "low"
    else:
        quality_profile = "medium"

    synthesis_label = (
        "grounded"
        if sufficient and retrieval_sufficiency == "sufficient"
        else "limited"
        if sufficient
        else "insufficient"
    )

    return {
        "answerMode": mode,
        "answerSubtype": mode,
        "responsiveEvidenceIds": responsive_ids,
        "directlyResponsiveIds": responsive_ids,
        "unsupportedAspects": unsupported_components,
        "unsupportedComponents": unsupported_components,
        "retrievalSufficiency": retrieval_sufficiency,
        "confidence": confidence,
        "sufficient": sufficient,
        "rationale": rationale,
        "evidenceQualityProfile": quality_profile,
        "synthesisMode": synthesis_label,
        "onTopicCount": on_topic_count,
        "fallbackOnlyOnTopic": fallback_only_on_topic,
        "anchoredSelectionSourceIds": anchored_selection_ids,
    }


__all__ = [
    "ANSWER_MODES",
    "SYNTHESIS_MODES",
    "SALVAGEABLE_MODES",
    "AsyncLLMModeClassifier",
    "LLMModeClassifier",
    "aclassify_question_mode",
    "classify_question_mode",
    "build_evidence_use_plan",
    "evidence_pool_is_weak",
    "selection_anchor_candidate_ids",
]
