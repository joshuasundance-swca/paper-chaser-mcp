"""Query understanding, routing, and bounded expansion helpers.

Phase 6 moved the regex/term-set constants, low-level text normalization,
regulatory/literature intent helpers, and specificity/ambiguity estimators
into dedicated submodules (``constants``, ``normalization``, ``regulatory``,
``specificity``). This module still hosts the async ``classify_query``
orchestrator, hypothesis/grounded/speculative expansion, variant combining
and deduplication, evidence-phrase extraction, and the intent reconciliation
helpers; Phase 7 will split those out further.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from ..config import AgenticConfig
from ..models import (
    ExpansionCandidate,
    IntentLabel,
    PlannerDecision,
)
from ..provider_base import ModelProviderBundle
from .hypotheses import (
    _ordered_provider_plan as _ordered_provider_plan,
)
from .hypotheses import (
    _sort_intent_candidates as _sort_intent_candidates,
)
from .hypotheses import (
    _source_for_intent_candidate as _source_for_intent_candidate,
)
from .hypotheses import (
    _upsert_intent_candidate as _upsert_intent_candidate,
)
from .hypotheses import (
    initial_retrieval_hypotheses as initial_retrieval_hypotheses,
)
from .normalization import (
    looks_like_exact_title,
    normalize_query,
    query_facets,
)
from .reconciliation import (
    _VALID_REGULATORY_INTENTS as _VALID_REGULATORY_INTENTS,
)
from .reconciliation import (
    _derive_regulatory_intent as _derive_regulatory_intent,
)
from .reconciliation import (
    _has_literature_corroboration as _has_literature_corroboration,
)
from .regulatory import (
    _detect_cultural_resource_intent,
    _infer_entity_card,
    _infer_regulatory_subintent,
    _strong_known_item_signal,
    _strong_regulatory_signal,
)
from .specificity import (
    _looks_broad_concept_query,
)
from .variants import (
    _signatures_are_near_duplicates as _signatures_are_near_duplicates,
)
from .variants import (
    _top_evidence_phrases as _top_evidence_phrases,
)
from .variants import (
    _variant_signature as _variant_signature,
)
from .variants import (
    combine_variants as combine_variants,
)
from .variants import (
    dedupe_variants as dedupe_variants,
)


async def classify_query(
    *,
    query: str,
    mode: str,
    year: str | None,
    venue: str | None,
    focus: str | None,
    provider_bundle: ModelProviderBundle,
    request_outcomes: list[dict[str, Any]] | None = None,
    request_id: str | None = None,
) -> tuple[str, PlannerDecision]:
    """Normalize and classify a smart-search request."""
    normalized = normalize_query(query)
    planner = await provider_bundle.aplan_search(
        query=normalized,
        mode=mode,
        year=year,
        venue=venue,
        focus=focus,
        request_outcomes=request_outcomes,
        request_id=request_id,
    )
    # Snapshot LLM-originated grounding signals BEFORE any deterministic
    # fallback mutates ``planner.entity_card`` / ``planner.candidate_concepts``
    # / ``planner.subject_card``. ``resolve_subject_card`` needs this to tell
    # apart "LLM emitted phase-4 signals" from "deterministic extractor
    # populated these fields after the LLM returned nothing".
    llm_emitted_grounding_signals = bool(planner.entity_card or planner.candidate_concepts or planner.subject_card)
    planner.intent_source = "planner"
    planner.intent_confidence = "medium"
    intent_candidates = list(planner.intent_candidates)
    _upsert_intent_candidate(
        candidates=intent_candidates,
        intent=planner.intent,
        confidence=planner.intent_confidence,
        source="planner",
        rationale="Model planner selected this intent as the best initial route.",
    )
    if mode != "auto":
        planner.intent = cast(
            Literal["discovery", "review", "known_item", "author", "citation", "regulatory"],
            mode,
        )
        planner.intent_source = "explicit"
        planner.intent_confidence = "high"
        planner.intent_rationale = "Intent was set explicitly by the caller."
        if mode == "review":
            planner.follow_up_mode = "claim_check"
        _upsert_intent_candidate(
            candidates=intent_candidates,
            intent=cast(IntentLabel, planner.intent),
            confidence="high",
            source="explicit",
            rationale="Explicit mode parameter from the caller.",
        )
    else:
        strong_known_item_signal = _strong_known_item_signal(normalized)
        strong_regulatory_signal = _strong_regulatory_signal(normalized, focus)
        if strong_known_item_signal:
            _upsert_intent_candidate(
                candidates=intent_candidates,
                intent="known_item",
                confidence="high",
                source="heuristic",
                rationale="Strong known-item signal (DOI, arXiv, or URL) was detected.",
            )
        if strong_regulatory_signal:
            _upsert_intent_candidate(
                candidates=intent_candidates,
                intent="regulatory",
                confidence="high",
                source="heuristic",
                rationale="Explicit regulatory citation or rulemaking marker was detected.",
            )

        heuristic_override: IntentLabel | None = None
        heuristic_override_confidence: Literal["high", "medium", "low"] = "high"
        heuristic_rationale = ""
        if strong_known_item_signal:
            heuristic_override = "known_item"
            heuristic_rationale = "Strong known-item signal overrode planner routing."
        elif strong_regulatory_signal:
            heuristic_override = "regulatory"
            heuristic_rationale = "Strong regulatory citation signal (CFR/FR reference) overrode planner routing."

        if heuristic_override is not None:
            if planner.intent == heuristic_override:
                planner.intent_source = "hybrid_agreement"
                planner.intent_confidence = "high"
                planner.intent_rationale = "Planner intent matched strong deterministic guardrail signals."
            else:
                planner.intent = heuristic_override
                planner.intent_source = "heuristic_override"
                planner.intent_confidence = heuristic_override_confidence
                planner.intent_rationale = heuristic_rationale
        elif (
            planner.intent == "known_item"
            and planner.intent_source == "planner"
            and not strong_known_item_signal
            and (
                # Either the deterministic text heuristic says broad-concept, or
                # the LLM itself labelled the query as broad / low-specificity /
                # highly ambiguous. Trust the planner's own signals first and
                # only fall back to heuristics for missing/low-confidence cases.
                _looks_broad_concept_query(
                    normalized_query=normalized,
                    focus=focus,
                    year=year,
                    venue=venue,
                )
                or (
                    planner.query_type == "broad_concept"
                    and (planner.query_specificity == "low" or planner.ambiguity_level == "high")
                )
                # Exact-title-looking queries with no identifier and high
                # ambiguity are common cultural-resource / regulatory traps
                # (e.g. "Section 106 consultation for offshore wind") — keep
                # known-item reasoning active as a secondary pass but stop
                # force-routing them into pure title-match recovery.
                or (looks_like_exact_title(normalized) and planner.ambiguity_level == "high")
            )
        ):
            previous_intent = planner.intent
            fallback_intent: IntentLabel = "discovery"
            for candidate in planner.intent_candidates:
                if candidate.intent != "known_item" and candidate.confidence in {"high", "medium"}:
                    fallback_intent = candidate.intent
                    break
            planner.intent = fallback_intent
            planner.intent_source = "heuristic_override"
            planner.intent_confidence = "medium"
            planner.intent_rationale = (
                f"Planner selected '{previous_intent}' but the query looks like a broad conceptual "
                "question (no DOI/arXiv/URL, planner specificity/ambiguity signals disagree); "
                f"demoted to '{fallback_intent}' to avoid force-routing discovery work into "
                "known-item recovery."
            )
        elif planner.intent_rationale.strip() == "":
            planner.intent_rationale = "Planner intent selected without a strong deterministic override."

    _upsert_intent_candidate(
        candidates=intent_candidates,
        intent=cast(IntentLabel, planner.intent),
        confidence=planner.intent_confidence,
        source=_source_for_intent_candidate(planner.intent_source),
        rationale=planner.intent_rationale or "Final routed intent after planner and guardrail reconciliation.",
    )
    sorted_candidates = _sort_intent_candidates(intent_candidates, preferred_intent=cast(IntentLabel, planner.intent))
    planner.intent_candidates = sorted_candidates[:4]
    merged_secondary_intents = [
        candidate.intent
        for candidate in planner.intent_candidates
        if candidate.intent != planner.intent and candidate.confidence in {"high", "medium"}
    ]
    for intent_label in planner.secondary_intents:
        if intent_label != planner.intent and intent_label not in merged_secondary_intents:
            merged_secondary_intents.append(intent_label)
    planner.secondary_intents = cast(list[IntentLabel], merged_secondary_intents[:3])
    primary_candidate = next(
        (candidate for candidate in planner.intent_candidates if candidate.intent == planner.intent),
        None,
    )
    planner.routing_confidence = (
        primary_candidate.confidence if primary_candidate is not None else planner.intent_confidence
    )
    if not planner.intent_rationale:
        planner.intent_rationale = "Intent routed from planner defaults."
    if planner.intent == "regulatory":
        if not planner.regulatory_subintent:
            planner.regulatory_subintent = _infer_regulatory_subintent(query, focus)
        if planner.entity_card is None:
            planner.entity_card = _infer_entity_card(query, focus)
    if planner.regulatory_intent is None:
        planner.regulatory_intent = _derive_regulatory_intent(planner=planner, query=query, focus=focus)
        if planner.regulatory_intent is not None and planner.regulatory_intent_source != "llm":
            planner.regulatory_intent_source = "deterministic_fallback"
    elif planner.regulatory_intent == "hybrid_regulatory_plus_literature" and not _has_literature_corroboration(
        planner=planner, query=query, focus=focus
    ):
        # LLM emitted the hybrid label directly. Require the same query-side
        # corroboration as the deterministic derivation path to avoid
        # forcing a literature review pass onto regulation-only asks.
        planner.regulatory_intent = _derive_regulatory_intent(planner=planner, query=query, focus=focus)
        # The LLM's original hybrid emission failed corroboration, so the
        # final label is deterministically derived -- record the downgrade so
        # downstream gates don't treat it as LLM-authoritative.
        if planner.regulatory_intent is not None:
            planner.regulatory_intent_source = "deterministic_fallback"
        else:
            planner.regulatory_intent_source = "unspecified"
    if planner.subject_card is None and planner.regulatory_intent is not None:
        # LLM-first subject card; uses planner.entity_card / candidate_concepts
        # and falls back to deterministic extraction when needed. The
        # ``llm_bundle_available`` flag reflects the *actual* planner execution:
        # it is True only when the provider bundle's ``aplan_search`` successfully
        # ran an LLM call. LLM bundles that silently fall back to
        # ``super().plan_search()`` (see provider_openai/provider_langchain) stamp
        # ``planner.planner_source="deterministic_fallback"``, which must not be
        # mistaken for a genuine LLM emission here even when the deterministic
        # shim happens to populate ``candidateConcepts`` / ``entityCard``.
        # ``intent_source`` is unreliable for this purpose -- explicit mode and
        # heuristic overrides rewrite it independently of planner provenance.
        from ..subject_grounding import resolve_subject_card

        planner.subject_card = resolve_subject_card(
            query=query,
            focus=focus,
            planner=planner,
            llm_bundle_available=(planner.planner_source == "llm"),
            llm_emitted_grounding_signals=llm_emitted_grounding_signals,
        )
    if planner.regulatory_intent == "hybrid_regulatory_plus_literature":
        hybrid_marker = (
            "hybrid_policy_science: fuse regulatory primary sources (Federal Register, CFR, agency guidance) "
            "with peer-reviewed literature to answer questions that mix policy and scientific evidence."
        )
        if not any("hybrid_policy_science" in str(entry) for entry in planner.retrieval_hypotheses):
            planner.retrieval_hypotheses.append(hybrid_marker)
    if not planner.intent_family and _detect_cultural_resource_intent(query, focus):
        planner.intent_family = "heritage_cultural_resources"
    if not planner.search_angles and planner.retrieval_hypotheses:
        planner.search_angles = list(planner.retrieval_hypotheses)
    if not planner.retrieval_hypotheses and planner.search_angles:
        planner.retrieval_hypotheses = list(planner.search_angles)
    merged_concepts = list(planner.candidate_concepts)
    merged_concepts.extend(query_facets(normalized))
    if focus:
        merged_concepts.extend(query_facets(focus))
    deduped_concepts: list[str] = []
    seen_concepts: set[str] = set()
    for concept in merged_concepts:
        lowered = concept.strip().lower()
        if not lowered or lowered in seen_concepts:
            continue
        seen_concepts.add(lowered)
        deduped_concepts.append(concept.strip())
    planner.candidate_concepts = deduped_concepts[:8]
    return normalized, planner


async def grounded_expansion_candidates(
    *,
    original_query: str,
    papers: list[dict[str, Any]],
    config: AgenticConfig,
    provider_bundle: ModelProviderBundle,
    focus: str | None = None,
    venue: str | None = None,
    year: str | None = None,
) -> list[ExpansionCandidate]:
    """Create grounded query variants from retrieved evidence via a second provider pass."""
    variants: list[ExpansionCandidate] = []
    base_query = normalize_query(original_query)
    suffixes = [item for item in [focus, venue, year] if item]
    if suffixes:
        variants.append(
            ExpansionCandidate(
                variant=" ".join([base_query, *suffixes]),
                source="from_input",
                rationale=("Adds explicit user-provided constraints to the literal query."),
            )
        )

    provider_variants = await provider_bundle.asuggest_grounded_expansions(
        query=base_query,
        papers=papers,
        max_variants=config.max_grounded_variants,
    )
    variants.extend(provider_variants)

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
    return deduped[: config.max_grounded_variants]


async def speculative_expansion_candidates(
    *,
    original_query: str,
    papers: list[dict[str, Any]],
    config: AgenticConfig,
    provider_bundle: ModelProviderBundle,
    request_outcomes: list[dict[str, Any]] | None = None,
    request_id: str | None = None,
) -> list[ExpansionCandidate]:
    """Generate bounded speculative variants through the provider bundle."""
    evidence_texts = [
        " ".join(
            part
            for part in [
                str(paper.get("title") or ""),
                str(paper.get("abstract") or ""),
            ]
            if part
        )
        for paper in papers[:8]
    ]
    return (
        await provider_bundle.asuggest_speculative_expansions(
            query=normalize_query(original_query),
            evidence_texts=[text for text in evidence_texts if text],
            max_variants=config.max_speculative_variants,
            request_outcomes=request_outcomes,
            request_id=request_id,
        )
    )[: config.max_speculative_variants]


# Phase 7c-1: combine_variants, dedupe_variants, _variant_signature,
# _signatures_are_near_duplicates, and _top_evidence_phrases were extracted
# to ``paper_chaser_mcp.agentic.planner.variants``. See the top-of-module
# re-imports for identity-preserving access via ``planner._core``.
