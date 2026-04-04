from paper_chaser_mcp.agentic.provider_helpers import (
    _AnswerSchema,
    _extract_json_object,
    _normalize_answer_schema_output,
    _PlannerConstraintsSchema,
    _PlannerResponseSchema,
)


def _normalize_confidence(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"high", "medium", "low"}:
        return normalized
    return "medium"


def test_planner_response_schema_supports_regulatory_routing_fields() -> None:
    schema = _PlannerResponseSchema(
        intent="regulatory",
        constraints=_PlannerConstraintsSchema(focus="current text"),
        seedIdentifiers=["50 CFR 17.11"],
        candidateConcepts=["northern long-eared bat"],
        providerPlan=["govinfo", "federal_register", "ecos", "semantic_scholar"],
        authorityFirst=True,
        anchorType="cfr_citation",
        anchorValue="50 CFR 17.11",
        requiredPrimarySources=["govinfo", "semantic_scholar"],
        successCriteria=["current_text_required", "timeline_required", "invalid_criterion"],
        followUpMode="qa",
    )

    decision = schema.to_planner_decision()

    assert decision.intent == "regulatory"
    assert decision.provider_plan == ["govinfo", "federal_register", "ecos"]
    assert decision.authority_first is True
    assert decision.anchor_type == "cfr_citation"
    assert decision.anchor_value == "50 CFR 17.11"
    assert decision.required_primary_sources == ["govinfo"]
    assert decision.success_criteria == ["current_text_required", "timeline_required"]


def test_planner_response_schema_falls_back_to_default_provider_plan_for_intent() -> None:
    schema = _PlannerResponseSchema(
        intent="review",
        constraints=_PlannerConstraintsSchema(),
        seedIdentifiers=[],
        candidateConcepts=[],
        providerPlan=["govinfo", "not_a_provider"],
        followUpMode="claim_check",
    )

    decision = schema.to_planner_decision()

    assert decision.intent == "review"
    assert decision.provider_plan == ["semantic_scholar", "openalex", "scholarapi", "core", "arxiv"]


def test_normalize_answer_schema_output_filters_unknown_selected_evidence_ids() -> None:
    parsed = _AnswerSchema(
        answer="GovInfo provides the current text.",
        unsupportedAsks=[],
        followUpQuestions=[],
        confidence="high",
        answerability="grounded",
        selectedEvidenceIds=["paper-1", "paper-x"],
        selectedLeadIds=["lead-1"],
    )
    normalized = _normalize_answer_schema_output(
        parsed_answer=parsed,
        evidence_papers=[
            {"paperId": "paper-1", "title": "Authoritative CFR source"},
            {"paperId": "paper-2", "title": "Supporting rulemaking history"},
        ],
        confidence_normalizer=_normalize_confidence,
    )

    assert normalized["confidence"] == "high"
    assert normalized["selectedEvidenceIds"] == ["paper-1"]
    assert normalized["selectedLeadIds"] == ["lead-1"]
    assert normalized["answerability"] == "grounded"


def test_normalize_answer_schema_output_downgrades_grounded_without_evidence_links() -> None:
    parsed = _AnswerSchema(
        answer="The evidence is suggestive.",
        unsupportedAsks=[],
        followUpQuestions=[],
        confidence="high",
        answerability="grounded",
        selectedEvidenceIds=[],
    )
    normalized = _normalize_answer_schema_output(
        parsed_answer=parsed,
        evidence_papers=[{"paperId": "paper-1", "title": "Source"}],
        confidence_normalizer=_normalize_confidence,
    )

    assert normalized["answerability"] == "limited"


def test_extract_json_object_ignores_prose_prefix_and_extracts_fenced_json() -> None:
    assert _extract_json_object('prefix {"answer":"not-accepted"} suffix') is None
    assert (
        _extract_json_object('```json\n{"answer":"accepted","confidence":"high"}\n```')
        == '{"answer":"accepted","confidence":"high"}'
    )
