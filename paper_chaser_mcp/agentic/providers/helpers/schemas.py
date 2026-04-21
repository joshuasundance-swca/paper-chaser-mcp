"""Pydantic schemas for smart-layer provider adapters."""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...models import (
    DocumentFamilyLabel,
    PlannerDecision,
    RegulatoryIntentLabel,
    SubjectCard,
    SubjectCardConfidence,
)
from .payloads import (
    _sanitize_primary_sources,
    _sanitize_provider_plan,
    _sanitize_success_criteria,
)

logger = logging.getLogger(__name__)

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

# Enum sets used by the optional planner-LLM schema extensions for
# ``regulatoryIntent`` and ``subjectCard``. Kept here (not imported from
# ``planner.py``) to avoid a circular import; kept in sync with
# ``RegulatoryIntentLabel`` / ``DocumentFamilyLabel`` in ``models.py``.
_VALID_REGULATORY_INTENT_VALUES: frozenset[str] = frozenset(
    {
        "current_cfr_text",
        "rulemaking_history",
        "species_dossier",
        "guidance_lookup",
        "hybrid_regulatory_plus_literature",
        "unspecified",
    },
)
_VALID_DOCUMENT_FAMILY_VALUES: frozenset[str] = frozenset(
    {
        "recovery_plan",
        "critical_habitat",
        "listing_rule",
        "consultation_guidance",
        "cfr_current_text",
        "rulemaking_notice",
        "programmatic_agreement",
        "tribal_policy",
        "agency_guidance",
        "unspecified",
    },
)
# The LLM may only self-report high/medium/low. The "deterministic_fallback"
# sentinel is reserved for the deterministic extractor and must not be set
# from an LLM response.
_VALID_LLM_SUBJECT_CARD_CONFIDENCE: frozenset[str] = frozenset({"high", "medium", "low"})


class _PlannerConstraintsSchema(BaseModel):
    """OpenAI Structured Outputs-compatible planner constraints."""

    year: str | None = None
    venue: str | None = None
    focus: str | None = None


class _PlannerSubjectCardSchema(BaseModel):
    """Optional LLM-native subject card emitted by the planner.

    Additive: legacy planner responses that omit this object continue to
    parse. When present, ``to_planner_decision`` materializes a
    :class:`SubjectCard` with ``source="planner_llm"`` so the deterministic
    fallback in ``classify_query`` is skipped.
    """

    commonName: str | None = None
    scientificName: str | None = None
    agency: str | None = None
    requestedDocumentFamily: str | None = None
    subjectTerms: list[str] = Field(default_factory=list)
    confidence: str | None = None

    def has_signal(self) -> bool:
        if any(
            isinstance(value, str) and value.strip()
            for value in (
                self.commonName,
                self.scientificName,
                self.agency,
                self.requestedDocumentFamily,
            )
        ):
            return True
        return bool([term for term in self.subjectTerms if isinstance(term, str) and term.strip()])


class _PlannerResponseSchema(BaseModel):
    """Structured planner response that avoids free-form object maps."""

    intent: Literal["discovery", "review", "known_item", "author", "citation", "regulatory"] = "discovery"
    querySpecificity: Literal["high", "medium", "low"] = "medium"
    ambiguityLevel: Literal["low", "medium", "high"] = "low"
    constraints: _PlannerConstraintsSchema = Field(default_factory=_PlannerConstraintsSchema)
    seedIdentifiers: list[str] = Field(default_factory=list)
    candidateConcepts: list[str] = Field(default_factory=list)
    providerPlan: list[str] = Field(default_factory=list)
    authorityFirst: bool = True
    anchorType: str | None = None
    anchorValue: str | None = None
    requiredPrimarySources: list[str] = Field(default_factory=list)
    successCriteria: list[str] = Field(default_factory=list)
    queryType: str = Field(default="broad_concept")
    breadthEstimate: int = Field(default=2, ge=1, le=4)
    searchAngles: list[str] = Field(default_factory=list)
    uncertaintyFlags: list[str] = Field(default_factory=list)
    firstPassMode: str = Field(default="targeted")
    retrievalHypotheses: list[str] = Field(default_factory=list)
    followUpMode: Literal["qa", "claim_check", "comparison"] = "qa"
    # Additive LLM-native fields for regulatory intent + subject grounding.
    # Optional so legacy planner responses without them still parse.
    regulatoryIntent: str | None = None
    subjectCard: _PlannerSubjectCardSchema | None = None

    def to_planner_decision(self) -> PlannerDecision:
        intent = self.intent if self.intent in _PLANNER_INTENTS else "discovery"
        provider_plan = _sanitize_provider_plan(intent=intent, provider_plan=self.providerPlan)
        anchor_type = self.anchorType if self.anchorType in _ANCHOR_TYPES else None
        required_primary = _sanitize_primary_sources(self.requiredPrimarySources)
        success_criteria = _sanitize_success_criteria(self.successCriteria)
        query_type = cast(
            Any,
            {
                "broad_concept": "broad_concept",
                "known_item": "known_item",
                "citation_repair": "citation_repair",
                "regulatory": "regulatory",
                "author": "author",
                "review": "review",
            }.get(self.queryType, "broad_concept"),
        )
        first_pass_mode = cast(
            Any,
            {"targeted": "targeted", "broad": "broad", "mixed": "mixed"}.get(
                self.firstPassMode,
                "targeted",
            ),
        )
        breadth_estimate = max(1, min(4, self.breadthEstimate))
        search_angles = [str(angle).strip() for angle in self.searchAngles if str(angle).strip()][:4]
        retrieval_hypotheses = [
            str(hypothesis).strip() for hypothesis in self.retrievalHypotheses if str(hypothesis).strip()
        ][:4]
        uncertainty_flags = [str(flag).strip() for flag in self.uncertaintyFlags if str(flag).strip()][:6]
        regulatory_intent = self._coerce_regulatory_intent()
        regulatory_intent_source: Literal["llm", "deterministic_fallback", "unspecified"] = (
            "llm" if regulatory_intent is not None else "unspecified"
        )
        subject_card = self._coerce_subject_card()
        return PlannerDecision(
            intent=intent,
            querySpecificity=self.querySpecificity,
            ambiguityLevel=self.ambiguityLevel,
            constraints={key: value for key, value in self.constraints.model_dump(exclude_none=True).items() if value},
            seedIdentifiers=self.seedIdentifiers,
            candidateConcepts=self.candidateConcepts,
            providerPlan=provider_plan,
            authorityFirst=bool(self.authorityFirst),
            anchorType=anchor_type,
            anchorValue=str(self.anchorValue or "").strip() or None,
            requiredPrimarySources=required_primary,
            successCriteria=success_criteria,
            queryType=query_type,
            breadthEstimate=breadth_estimate,
            searchAngles=search_angles,
            uncertaintyFlags=uncertainty_flags,
            firstPassMode=first_pass_mode,
            retrievalHypotheses=retrieval_hypotheses,
            followUpMode=self.followUpMode,
            regulatoryIntent=regulatory_intent,
            regulatoryIntentSource=regulatory_intent_source,
            subjectCard=subject_card,
        )

    def _coerce_regulatory_intent(self) -> RegulatoryIntentLabel | None:
        raw = self.regulatoryIntent
        if raw is None:
            return None
        value = str(raw).strip()
        if not value:
            return None
        if value in _VALID_REGULATORY_INTENT_VALUES:
            return cast(RegulatoryIntentLabel, value)
        logger.warning(
            "Planner LLM returned invalid regulatoryIntent=%r; falling back to deterministic derivation.",
            value,
        )
        return None

    def _coerce_subject_card(self) -> SubjectCard | None:
        card = self.subjectCard
        if card is None or not card.has_signal():
            return None

        def _clean(value: str | None) -> str | None:
            if value is None:
                return None
            cleaned = str(value).strip()
            return cleaned or None

        requested_family_raw = _clean(card.requestedDocumentFamily)
        requested_family: DocumentFamilyLabel | None
        if requested_family_raw is None:
            requested_family = None
        elif requested_family_raw in _VALID_DOCUMENT_FAMILY_VALUES:
            requested_family = cast(DocumentFamilyLabel, requested_family_raw)
        else:
            logger.warning(
                "Planner LLM returned invalid subjectCard.requestedDocumentFamily=%r; dropping field.",
                requested_family_raw,
            )
            requested_family = None

        confidence_raw = _clean(card.confidence)
        confidence: SubjectCardConfidence
        if confidence_raw is None:
            confidence = "medium"
        elif confidence_raw in _VALID_LLM_SUBJECT_CARD_CONFIDENCE:
            confidence = cast(SubjectCardConfidence, confidence_raw)
        else:
            logger.warning(
                "Planner LLM returned invalid subjectCard.confidence=%r; defaulting to 'medium'.",
                confidence_raw,
            )
            confidence = "medium"

        subject_terms: list[str] = []
        seen_terms: set[str] = set()
        for term in card.subjectTerms:
            if not isinstance(term, str):
                continue
            cleaned = term.strip()
            lowered = cleaned.lower()
            if not cleaned or lowered in seen_terms:
                continue
            seen_terms.add(lowered)
            subject_terms.append(cleaned)
        subject_terms = subject_terms[:6]

        return SubjectCard(
            commonName=_clean(card.commonName),
            scientificName=_clean(card.scientificName),
            agency=_clean(card.agency),
            requestedDocumentFamily=requested_family,
            subjectTerms=subject_terms,
            confidence=confidence,
            source="planner_llm",
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
    citedPaperIds: list[str] = Field(default_factory=list)
    evidenceSummary: str = Field(default="")
    missingEvidenceDescription: str = Field(default="")

    @model_validator(mode="after")
    def _validate_answer_contract(self) -> "_AnswerSchema":
        if self.answerability == "insufficient":
            return self
        answer_text = self.answer.strip()
        has_explicit_grounding = bool(self.citedPaperIds or self.evidenceSummary.strip())
        if has_explicit_grounding and answer_text and len(answer_text) < 100:
            raise ValueError("answer must be at least 100 characters when answerability is not insufficient")
        if has_explicit_grounding and answer_text and not self.evidenceSummary.strip():
            raise ValueError("evidenceSummary is required when answer text is present")
        return self


class _ReviseStrategySchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    revised_query: str = Field(default="", alias="revisedQuery")
    revised_intent: Literal["discovery", "review", "known_item", "author", "citation", "regulatory"] = Field(
        default="discovery", alias="revisedIntent"
    )
    revised_providers: list[str] = Field(default_factory=list, alias="revisedProviders")
    rationale: str = Field(default="")


class _RelevanceClassificationItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    paper_id: str = Field(alias="paperId", default="")
    classification: Literal["on_topic", "weak_match", "off_topic"] = "weak_match"
    rationale: str = Field(default="")


class _RelevanceBatchSchema(BaseModel):
    classifications: list[_RelevanceClassificationItem] = Field(default_factory=list)


class _AdequacyJudgmentSchema(BaseModel):
    adequacy: Literal["succeeded", "partial", "insufficient"] = "partial"
    reason: str = Field(default="")


class AnswerStatusValidation(BaseModel):
    """LLM-powered classification of whether a research answer is substantive."""

    classification: Literal["answered", "abstained", "insufficient_evidence"] = "abstained"
    reasoning: str = Field(default="")


class _EvidenceGapSchema(BaseModel):
    gaps: list[str] = Field(default_factory=list)


class _AnswerModeClassificationSchema(BaseModel):
    """LLM classification for follow-up answer modes (ask_result_set routing).

    ``answerMode`` must be one of :data:`agentic.answer_modes.ANSWER_MODES`.
    Callers validate the value against ``ANSWER_MODES`` before honoring it
    and treat anything else as equivalent to ``"unknown"``, so invalid model
    output never forces a specific route.
    """

    answerMode: str = Field(default="unknown")
