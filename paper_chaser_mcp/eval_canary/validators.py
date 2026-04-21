"""Validators for eval canary items and live tool responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models.common import (
    AbstentionDetails,
    GuidedExecutionProvenance,
    GuidedResultState,
    RuntimeSummary,
)
from .schema import (
    ABSTENTION_BEHAVIORS,
    ANSWER_STATUSES,
    ORIGINS,
    PLANNER_INTENTS,
    PROVENANCE_ACCESS_STATES,
    PROVENANCE_SOURCE_TYPES,
    PROVENANCE_TRUST_STATES,
    REVIEW_STATUSES,
    RUNTIME_PROFILES,
    SCHEMA_VERSION,
    TASK_FAMILIES,
    _result_item,
    _validate_structured_model,
)


def _validate_meta(meta: Any, errors: list[str], warnings: list[str]) -> tuple[str, str]:
    if not isinstance(meta, dict):
        errors.append("meta must be an object")
        return "", ""

    item_id = str(meta.get("id") or "").strip()
    family = str(meta.get("task_family") or "").strip()
    dataset_version = str(meta.get("dataset_version") or "").strip()
    schema_version = meta.get("schema_version")
    origin = str(meta.get("origin") or "").strip()
    review_status = str(meta.get("review_status") or "").strip()
    tags = meta.get("tags")

    if not item_id:
        errors.append("meta.id is required")
    if family not in TASK_FAMILIES:
        errors.append(f"meta.task_family must be one of {sorted(TASK_FAMILIES)}")
    if not dataset_version:
        errors.append("meta.dataset_version is required")
    if schema_version != SCHEMA_VERSION:
        errors.append(f"meta.schema_version must be {SCHEMA_VERSION}")
    if origin not in ORIGINS:
        errors.append(f"meta.origin must be one of {sorted(ORIGINS)}")
    if review_status not in REVIEW_STATUSES:
        errors.append(f"meta.review_status must be one of {sorted(REVIEW_STATUSES)}")
    if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and tag.strip() for tag in tags):
        errors.append("meta.tags must be a non-empty list of strings")

    if review_status == "draft":
        warnings.append("item is still marked draft and should not be used as a hard gate")

    return item_id, family


def _validate_input(family: str, payload: Any, errors: list[str]) -> None:
    if not isinstance(payload, dict):
        errors.append("input must be an object")
        return

    def _required(field: str) -> None:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"input.{field} is required for {family}")

    if family == "planner":
        _required("query")
    elif family == "synthesis":
        _required("query_context")
        _required("follow_up_question")
        _required("evidence_quality")
    elif family == "abstention":
        _required("query")
    elif family == "provenance":
        _required("query_context")
        _required("source_id")
    elif family in {"runtime", "misc"}:
        _required("query_context")


def _validate_expected(family: str, payload: Any, errors: list[str]) -> None:
    if not isinstance(payload, dict):
        errors.append("expected must be an object")
        return

    if family == "planner":
        acceptable_intents = payload.get("acceptable_intents")
        unacceptable_intents = payload.get("unacceptable_intents")
        provider_hints = payload.get("acceptable_provider_hints")
        if not isinstance(acceptable_intents, list) or not acceptable_intents:
            errors.append("expected.acceptable_intents must be a non-empty list")
        elif any(intent not in PLANNER_INTENTS for intent in acceptable_intents):
            errors.append(f"expected.acceptable_intents must only contain {sorted(PLANNER_INTENTS)}")
        if not isinstance(unacceptable_intents, list):
            errors.append("expected.unacceptable_intents must be a list")
        elif any(intent not in PLANNER_INTENTS for intent in unacceptable_intents):
            errors.append(f"expected.unacceptable_intents must only contain {sorted(PLANNER_INTENTS)}")
        if (
            not isinstance(provider_hints, list)
            or not provider_hints
            or not all(isinstance(item, str) and item.strip() for item in provider_hints)
        ):
            errors.append("expected.acceptable_provider_hints must be a non-empty list of strings")
        if not isinstance(payload.get("must_surface_clarification"), bool):
            errors.append("expected.must_surface_clarification must be a boolean")
        if not isinstance(payload.get("should_allow_partial"), bool):
            errors.append("expected.should_allow_partial must be a boolean")

    elif family == "synthesis":
        answer_status = payload.get("expected_answer_status")
        if answer_status not in ANSWER_STATUSES:
            errors.append(f"expected.expected_answer_status must be one of {sorted(ANSWER_STATUSES)}")
        if not isinstance(payload.get("should_abstain"), bool):
            errors.append("expected.should_abstain must be a boolean")
        if not isinstance(payload.get("must_cite_evidence"), bool):
            errors.append("expected.must_cite_evidence must be a boolean")
        if not isinstance(payload.get("should_preserve_uncertainty"), bool):
            errors.append("expected.should_preserve_uncertainty must be a boolean")
        required_traits = payload.get("required_evidence_traits")
        if (
            not isinstance(required_traits, list)
            or not required_traits
            or not all(isinstance(item, str) and item.strip() for item in required_traits)
        ):
            errors.append("expected.required_evidence_traits must be a non-empty list of strings")

    elif family == "abstention":
        if payload.get("correct_behavior") not in ABSTENTION_BEHAVIORS:
            errors.append(f"expected.correct_behavior must be one of {sorted(ABSTENTION_BEHAVIORS)}")
        required_markers = payload.get("required_markers")
        disallowed_patterns = payload.get("disallowed_patterns")
        if (
            not isinstance(required_markers, list)
            or not required_markers
            or not all(isinstance(item, str) and item.strip() for item in required_markers)
        ):
            errors.append("expected.required_markers must be a non-empty list of strings")
        if (
            not isinstance(disallowed_patterns, list)
            or not disallowed_patterns
            or not all(isinstance(item, str) and item.strip() for item in disallowed_patterns)
        ):
            errors.append("expected.disallowed_patterns must be a non-empty list of strings")
        if not isinstance(payload.get("should_preserve_uncertainty"), bool):
            errors.append("expected.should_preserve_uncertainty must be a boolean")

    elif family == "provenance":
        if payload.get("expected_source_type") not in PROVENANCE_SOURCE_TYPES:
            errors.append(f"expected.expected_source_type must be one of {sorted(PROVENANCE_SOURCE_TYPES)}")
        if payload.get("expected_trust_state") not in PROVENANCE_TRUST_STATES:
            errors.append(f"expected.expected_trust_state must be one of {sorted(PROVENANCE_TRUST_STATES)}")
        if payload.get("expected_access_state") not in PROVENANCE_ACCESS_STATES:
            errors.append(f"expected.expected_access_state must be one of {sorted(PROVENANCE_ACCESS_STATES)}")
        if not isinstance(payload.get("should_recommend_direct_read"), bool):
            errors.append("expected.should_recommend_direct_read must be a boolean")

    elif family == "runtime":
        if payload.get("expected_profile") not in RUNTIME_PROFILES:
            errors.append(f"expected.expected_profile must be one of {sorted(RUNTIME_PROFILES)}")
        if not isinstance(payload.get("must_report_configured_provider"), bool):
            errors.append("expected.must_report_configured_provider must be a boolean")
        if not isinstance(payload.get("must_report_active_provider"), bool):
            errors.append("expected.must_report_active_provider must be a boolean")
        if not isinstance(payload.get("must_surface_warnings"), bool):
            errors.append("expected.must_surface_warnings must be a boolean")
        include_sets = payload.get("must_include_sets")
        if (
            not isinstance(include_sets, list)
            or not include_sets
            or not all(isinstance(item, str) and item.strip() for item in include_sets)
        ):
            errors.append("expected.must_include_sets must be a non-empty list of strings")

    elif family == "misc":
        supporting_role = payload.get("supporting_role")
        if not isinstance(supporting_role, str) or not supporting_role.strip():
            errors.append("expected.supporting_role must be a non-empty string")
        required_markers = payload.get("required_markers")
        disallowed_patterns = payload.get("disallowed_patterns")
        if (
            not isinstance(required_markers, list)
            or not required_markers
            or not all(isinstance(item, str) and item.strip() for item in required_markers)
        ):
            errors.append("expected.required_markers must be a non-empty list of strings")
        if (
            not isinstance(disallowed_patterns, list)
            or not disallowed_patterns
            or not all(isinstance(item, str) and item.strip() for item in disallowed_patterns)
        ):
            errors.append("expected.disallowed_patterns must be a non-empty list of strings")
        if not isinstance(payload.get("must_preserve_cost_awareness"), bool):
            errors.append("expected.must_preserve_cost_awareness must be a boolean")


def validate_eval_item(raw_item: dict[str, Any], *, path: Path, line_number: int) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    item_id, family = _validate_meta(raw_item.get("meta"), errors, warnings)
    _validate_input(family, raw_item.get("input"), errors)
    _validate_expected(family, raw_item.get("expected"), errors)

    why_it_matters = raw_item.get("why_it_matters")
    if not isinstance(why_it_matters, str) or not why_it_matters.strip():
        errors.append("why_it_matters is required and must be a non-empty string")

    if family and not path.name.startswith(f"{family}."):
        warnings.append(f"file name {path.name!r} does not start with the task family {family!r}")

    return _result_item(
        item_id=item_id or f"{path.name}:{line_number}",
        family=family or "unknown",
        path=str(path),
        line_number=line_number,
        errors=errors,
        warnings=warnings,
    )


def _planner_provider_candidates(response: dict[str, Any]) -> set[str]:
    providers: set[str] = set()
    coverage = response.get("coverage")
    if isinstance(coverage, dict):
        for key in ("providersAttempted", "providersSucceeded", "providersFailed"):
            value = coverage.get(key)
            if isinstance(value, list):
                providers.update(str(item) for item in value if isinstance(item, str))
    coverage_summary = response.get("coverageSummary")
    if isinstance(coverage_summary, dict):
        for key in (
            "providersAttempted",
            "providersSucceeded",
            "providersFailed",
            "providersZeroResults",
        ):
            value = coverage_summary.get(key)
            if isinstance(value, list):
                providers.update(str(item) for item in value if isinstance(item, str))
    routing_summary = response.get("routingSummary")
    if isinstance(routing_summary, dict):
        for key in (
            "providersAttempted",
            "providersSucceeded",
            "providersFailed",
            "providerHints",
        ):
            value = routing_summary.get(key)
            if isinstance(value, list):
                providers.update(str(item) for item in value if isinstance(item, str))
        value = routing_summary.get("selectedProvider")
        if isinstance(value, str) and value:
            providers.add(value)
    providers_used = response.get("providersUsed")
    if isinstance(providers_used, list):
        providers.update(str(item) for item in providers_used if isinstance(item, str))
    provider_outcomes = response.get("providerOutcomes")
    if isinstance(provider_outcomes, list):
        for outcome in provider_outcomes:
            if isinstance(outcome, dict) and isinstance(outcome.get("provider"), str):
                providers.add(outcome["provider"])
    sources = response.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict) and isinstance(source.get("provider"), str):
                providers.add(source["provider"])
    return providers


def _validate_runtime_live_response(expected: dict[str, Any], response: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    runtime_summary = response.get("runtimeSummary")
    if not isinstance(runtime_summary, dict):
        return ["live response is missing runtimeSummary"]
    _validate_structured_model("runtimeSummary", RuntimeSummary, runtime_summary, errors)

    if runtime_summary.get("effectiveProfile") != expected.get("expected_profile"):
        observed_profile = runtime_summary.get("effectiveProfile")
        expected_profile = expected.get("expected_profile")
        errors.append(f"effectiveProfile={observed_profile!r} did not match expected {expected_profile!r}")
    if expected.get("must_report_configured_provider") and not runtime_summary.get("configuredSmartProvider"):
        errors.append("runtimeSummary.configuredSmartProvider was required but missing")
    if expected.get("must_report_active_provider") and not runtime_summary.get("activeSmartProvider"):
        errors.append("runtimeSummary.activeSmartProvider was required but missing")
    if expected.get("must_surface_warnings") and not response.get("warnings"):
        errors.append("warnings were expected but none were returned")
    include_sets = expected.get("must_include_sets") or []
    for field_name in include_sets:
        if field_name not in runtime_summary:
            errors.append(f"runtimeSummary.{field_name} was required but missing")
    return errors


def _validate_planner_live_response(expected: dict[str, Any], response: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    intent = response.get("intent")
    acceptable_intents = expected.get("acceptable_intents") or []
    unacceptable_intents = expected.get("unacceptable_intents") or []
    if acceptable_intents and intent not in acceptable_intents:
        errors.append(f"intent={intent!r} was not in acceptable_intents={acceptable_intents!r}")
    if unacceptable_intents and intent in unacceptable_intents:
        errors.append(f"intent={intent!r} was in unacceptable_intents={unacceptable_intents!r}")

    provenance = response.get("executionProvenance")
    if not isinstance(provenance, dict):
        errors.append("executionProvenance was expected but missing")
    else:
        _validate_structured_model("executionProvenance", GuidedExecutionProvenance, provenance, errors)

    result_state = response.get("resultState")
    if result_state is not None:
        _validate_structured_model("resultState", GuidedResultState, result_state, errors)

    if not isinstance(response.get("status"), str) or not response.get("status"):
        errors.append("status was expected but missing")
    if not isinstance(response.get("summary"), str) or not response.get("summary"):
        errors.append("summary was expected but missing")

    providers_seen = _planner_provider_candidates(response)
    provider_hints = expected.get("acceptable_provider_hints") or []
    if provider_hints and not any(provider in providers_seen for provider in provider_hints):
        providers_seen_sorted = sorted(providers_seen)
        errors.append(
            "none of the acceptable_provider_hints "
            f"{provider_hints!r} appeared in providers seen "
            f"{providers_seen_sorted!r}"
        )

    must_surface_clarification = expected.get("must_surface_clarification")
    clarification = response.get("clarification")
    result_status = response.get("status") or (
        response.get("resultState", {}).get("status") if isinstance(response.get("resultState"), dict) else None
    )
    if must_surface_clarification and not clarification and result_status != "needs_disambiguation":
        errors.append("clarification or needs_disambiguation was expected but not present")

    should_allow_partial = expected.get("should_allow_partial")
    if should_allow_partial is False and result_status == "partial":
        errors.append("partial result was returned where should_allow_partial=false")

    return errors


def _validate_synthesis_live_response(expected: dict[str, Any], response: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    answer_status = response.get("answerStatus")
    if answer_status not in ANSWER_STATUSES:
        errors.append(f"answerStatus={answer_status!r} was not one of {sorted(ANSWER_STATUSES)}")

    expected_answer_status = expected.get("expected_answer_status")
    if expected_answer_status and answer_status != expected_answer_status:
        errors.append(f"answerStatus={answer_status!r} did not match expected_answer_status={expected_answer_status!r}")

    should_abstain = expected.get("should_abstain")
    if should_abstain and answer_status not in {"abstained", "insufficient_evidence"}:
        errors.append("expected abstention or insufficient_evidence but got a non-abstaining answerStatus")
    if should_abstain is False and answer_status == "abstained":
        errors.append("unexpected abstained answerStatus for non-abstaining synthesis item")

    provenance = response.get("executionProvenance")
    if isinstance(provenance, dict):
        _validate_structured_model("executionProvenance", GuidedExecutionProvenance, provenance, errors)
    else:
        errors.append("executionProvenance was expected but missing")

    result_state = response.get("resultState")
    if isinstance(result_state, dict):
        _validate_structured_model("resultState", GuidedResultState, result_state, errors)

    abstention_details = response.get("abstentionDetails")
    if answer_status in {"abstained", "insufficient_evidence"}:
        if not isinstance(abstention_details, dict):
            errors.append("abstentionDetails was expected for abstained or insufficient_evidence responses")
        else:
            _validate_structured_model("abstentionDetails", AbstentionDetails, abstention_details, errors)

    if not isinstance(response.get("nextActions"), list):
        errors.append("nextActions was expected but missing")

    return errors
