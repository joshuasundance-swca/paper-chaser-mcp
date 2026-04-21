"""Phase 8b-2 identity tests for the providers/helpers subpackage extraction.

These tests assert that the public and private symbols reachable via the
``paper_chaser_mcp.agentic.provider_helpers`` shim are the *same object* as
the ones defined inside the new ``paper_chaser_mcp.agentic.providers.helpers``
subpackage. That protects downstream callers against any future drift where
someone accidentally rewrites the shim to return a duplicated copy of a class
or function.
"""

from __future__ import annotations

from paper_chaser_mcp.agentic import provider_helpers as _shim
from paper_chaser_mcp.agentic.providers import helpers as _helpers
from paper_chaser_mcp.agentic.providers.helpers import (
    evidence as _evidence,
)
from paper_chaser_mcp.agentic.providers.helpers import (
    nlp as _nlp,
)
from paper_chaser_mcp.agentic.providers.helpers import (
    payloads as _payloads,
)
from paper_chaser_mcp.agentic.providers.helpers import (
    schemas as _schemas,
)


def test_nlp_module_owns_tokenization_helpers() -> None:
    for name in (
        "_tokenize",
        "_top_terms",
        "_lexical_similarity",
        "_cosine_similarity",
        "_normalize_confidence_label",
        "_langchain_message_text",
        "_extract_json_object",
        "_normalize_label_text",
        "_normalize_theme_label_output",
        "_coerce_langchain_structured_response",
        "_extract_seed_identifiers",
        "_collect_evidence_ids",
        "_normalize_answer_schema_output",
        "_normalized_embedding_text",
    ):
        value = getattr(_nlp, name)
        assert getattr(_helpers, name) is value
        assert getattr(_shim, name) is value


def test_schemas_module_owns_pydantic_validators() -> None:
    for name in (
        "_PlannerConstraintsSchema",
        "_PlannerSubjectCardSchema",
        "_PlannerResponseSchema",
        "_ExpansionSchema",
        "_ExpansionListSchema",
        "_AnswerSchema",
        "_ReviseStrategySchema",
        "_RelevanceClassificationItem",
        "_RelevanceBatchSchema",
        "_AdequacyJudgmentSchema",
        "AnswerStatusValidation",
        "_EvidenceGapSchema",
        "_AnswerModeClassificationSchema",
    ):
        value = getattr(_schemas, name)
        assert getattr(_helpers, name) is value
        assert getattr(_shim, name) is value
        assert value.__module__ == "paper_chaser_mcp.agentic.providers.helpers.schemas"


def test_payloads_module_owns_payload_builders() -> None:
    for name in (
        "_paper_evidence_payload",
        "_build_theme_label_payload",
        "_build_theme_summary_payload",
        "_build_answer_payload",
        "_filter_expansion_candidates",
        "_sanitize_provider_plan",
        "_sanitize_primary_sources",
        "_sanitize_success_criteria",
    ):
        value = getattr(_payloads, name)
        assert getattr(_helpers, name) is value
        assert getattr(_shim, name) is value
        assert value.__module__ == "paper_chaser_mcp.agentic.providers.helpers.payloads"


def test_evidence_module_owns_gap_generation() -> None:
    for name in (
        "generate_evidence_gaps_without_llm",
        "_theme_label_terms",
        "_compact_theme_label",
        "_paper_terms",
        "_normalize_gap_text",
        "_ecos_gap_is_relevant",
        "_query_month_year_references",
        "_timeline_gap_statements",
        "_hypothesis_gap_statements",
        "_question_focus_terms",
        "_paper_focus_cues",
        "_paper_alignment_bucket",
        "_format_paper_anchor",
        "_deterministic_comparison_answer",
        "_deterministic_theme_summary",
        "_deterministic_gap_insights",
    ):
        value = getattr(_evidence, name)
        assert getattr(_helpers, name) is value
        assert getattr(_shim, name) is value
        assert value.__module__ == "paper_chaser_mcp.agentic.providers.helpers.evidence"


def test_shim_public_surface_matches_subpackage() -> None:
    assert _shim.AnswerStatusValidation is _schemas.AnswerStatusValidation
    assert _shim.COMMON_QUERY_WORDS is _nlp.COMMON_QUERY_WORDS
    assert _shim.generate_evidence_gaps_without_llm is _evidence.generate_evidence_gaps_without_llm


def test_shim_preserves_all_pinned_test_seams() -> None:
    pinned = (
        "_AdequacyJudgmentSchema",
        "_AnswerSchema",
        "_EvidenceGapSchema",
        "_ExpansionListSchema",
        "_ExpansionSchema",
        "_PlannerConstraintsSchema",
        "_PlannerResponseSchema",
        "_PlannerSubjectCardSchema",
        "_RelevanceBatchSchema",
        "_RelevanceClassificationItem",
        "_ReviseStrategySchema",
        "_extract_json_object",
        "_normalize_answer_schema_output",
    )
    for name in pinned:
        assert hasattr(_shim, name), name
        assert getattr(_shim, name) is getattr(_helpers, name)
