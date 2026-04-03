import pytest

openai_pydantic = pytest.importorskip("openai.lib._pydantic", reason="openai not installed")
to_strict_json_schema = openai_pydantic.to_strict_json_schema

from paper_chaser_mcp.agentic.providers import (  # noqa: E402
    _PlannerConstraintsSchema,
    _PlannerResponseSchema,
)


def test_planner_response_schema_is_openai_strict_compatible() -> None:
    schema = to_strict_json_schema(_PlannerResponseSchema)

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "intent",
        "constraints",
        "seedIdentifiers",
        "candidateConcepts",
        "providerPlan",
        "authorityFirst",
        "anchorType",
        "anchorValue",
        "requiredPrimarySources",
        "successCriteria",
        "followUpMode",
    ]

    constraints_ref = schema["properties"]["constraints"]["$ref"]
    constraints_key = constraints_ref.rsplit("/", maxsplit=1)[-1]
    constraints_schema = schema["$defs"][constraints_key]

    assert constraints_schema["type"] == "object"
    assert constraints_schema["additionalProperties"] is False
    assert constraints_schema["required"] == ["year", "venue", "focus"]
    assert constraints_schema["properties"]["year"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]


def test_planner_response_schema_round_trips_constraints() -> None:
    response = _PlannerResponseSchema(
        intent="review",
        constraints=_PlannerConstraintsSchema(
            year="2024",
            focus="benchmarking",
        ),
        seedIdentifiers=["10.1000/test-doi"],
        candidateConcepts=["tool agents"],
        providerPlan=["semantic_scholar", "openalex"],
        followUpMode="claim_check",
    )

    decision = response.to_planner_decision()

    assert decision.intent == "review"
    assert decision.constraints == {"year": "2024", "focus": "benchmarking"}
    assert decision.seed_identifiers == ["10.1000/test-doi"]
    assert decision.candidate_concepts == ["tool agents"]
    assert decision.provider_plan == ["semantic_scholar", "openalex"]
    assert decision.follow_up_mode == "claim_check"


def test_planner_response_schema_round_trips_regulatory_fields() -> None:
    response = _PlannerResponseSchema(
        intent="regulatory",
        constraints=_PlannerConstraintsSchema(focus="current text"),
        seedIdentifiers=["50 CFR 17.11"],
        candidateConcepts=["northern long-eared bat"],
        providerPlan=["govinfo", "federal_register", "ecos"],
        authorityFirst=True,
        anchorType="cfr_citation",
        anchorValue="50 CFR 17.11",
        requiredPrimarySources=["govinfo"],
        successCriteria=["current_text_required"],
        followUpMode="qa",
    )

    decision = response.to_planner_decision()

    assert decision.intent == "regulatory"
    assert decision.provider_plan == ["govinfo", "federal_register", "ecos"]
    assert decision.authority_first is True
    assert decision.anchor_type == "cfr_citation"
    assert decision.anchor_value == "50 CFR 17.11"
    assert decision.required_primary_sources == ["govinfo"]
    assert decision.success_criteria == ["current_text_required"]
