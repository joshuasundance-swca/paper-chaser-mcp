from paper_chaser_mcp.guided_semantic import (
    EvidenceDecision,
    FollowUpDecision,
    GuidedAnswer,
    ProviderPlan,
    RegulatoryAnchor,
    RoutingDecision,
    explicit_source_reference,
)


def test_guided_semantic_models_accept_wire_aliases() -> None:
    decision = RoutingDecision.model_validate(
        {
            "intent": "regulatory",
            "confidence": "high",
            "rationale": "Explicit CFR citation anchored the route.",
            "anchor": {
                "anchorType": "cfr_citation",
                "anchorValue": "50 CFR 17.11",
                "requiredPrimarySources": ["govinfo"],
                "successCriteria": ["current_text_required"],
                "subjectTerms": ["northern long-eared bat"],
            },
            "providerPlan": {
                "providers": ["govinfo", "federal_register"],
                "authorityFirst": True,
                "rationale": "GovInfo should dominate current-text retrieval.",
            },
        }
    )

    assert isinstance(decision.anchor, RegulatoryAnchor)
    assert isinstance(decision.provider_plan, ProviderPlan)
    assert decision.anchor.anchor_type == "cfr_citation"
    assert decision.provider_plan.authority_first is True


def test_guided_semantic_models_validate_follow_up_and_answer_contracts() -> None:
    follow_up = FollowUpDecision.model_validate(
        {
            "answerFromSession": True,
            "selectedEvidenceIds": ["50 CFR 17.11"],
            "selectedLeadIds": ["2024-99999"],
            "unsupportedAsks": [],
            "rationale": "The answer is directly supported by the saved evidence.",
        }
    )
    evidence = EvidenceDecision.model_validate(
        {
            "evidenceId": "50 CFR 17.11",
            "includeAs": "evidence",
            "whyIncluded": "Authoritative current text for the cited CFR section.",
            "whyNotVerified": None,
        }
    )
    answer = GuidedAnswer.model_validate(
        {
            "resultStatus": "succeeded",
            "answerability": "grounded",
            "summary": "GovInfo current text answered the request.",
            "evidenceGaps": [],
            "nextActions": ["Inspect the CFR source before citing it externally."],
        }
    )

    assert follow_up.answer_from_session is True
    assert follow_up.selected_evidence_ids == ["50 CFR 17.11"]
    assert evidence.include_as == "evidence"
    assert answer.result_status == "succeeded"
    assert answer.answerability == "grounded"


def test_explicit_source_reference_accepts_evidence_id_markers() -> None:
    assert explicit_source_reference("inspect source id: src_7") == "src_7"
    assert explicit_source_reference("evidence id = 50-CFR-17.11") == "50-CFR-17.11"
    assert explicit_source_reference("lead id: 2022-25998") == "2022-25998"
