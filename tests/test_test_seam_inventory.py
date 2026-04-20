"""Phase 1 pinning test: enumerate and freeze private test-only seams.

This test records every private (underscore-prefixed) symbol that the test
suite currently imports from ``paper_chaser_mcp.*``. Later refactor phases
must either preserve these seams (e.g. via re-export) or explicitly update
``KNOWN_TEST_SEAMS`` when moving code. New private imports are rejected to
prevent silent growth of the test-only surface.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PACKAGE_PREFIX = "paper_chaser_mcp"


KNOWN_TEST_SEAMS: frozenset[tuple[str, str]] = frozenset(
    {
        ("paper_chaser_mcp.agentic.graphs", "_build_grounded_comparison_answer"),
        ("paper_chaser_mcp.agentic.graphs", "_classify_topical_relevance_with_provenance"),
        ("paper_chaser_mcp.agentic.graphs", "_derive_regulatory_query_flags"),
        ("paper_chaser_mcp.agentic.graphs", "_ecos_query_variants"),
        ("paper_chaser_mcp.agentic.graphs", "_finalize_theme_label"),
        ("paper_chaser_mcp.agentic.graphs", "_graph_frontier_scores"),
        ("paper_chaser_mcp.agentic.graphs", "_has_inspectable_sources"),
        ("paper_chaser_mcp.agentic.graphs", "_has_on_topic_sources"),
        ("paper_chaser_mcp.agentic.graphs", "_is_agency_guidance_query"),
        ("paper_chaser_mcp.agentic.graphs", "_is_current_cfr_text_request"),
        ("paper_chaser_mcp.agentic.graphs", "_is_opaque_query"),
        ("paper_chaser_mcp.agentic.graphs", "_query_requests_regulatory_history"),
        ("paper_chaser_mcp.agentic.graphs", "_rank_ecos_variant_hits"),
        ("paper_chaser_mcp.agentic.graphs", "_rank_regulatory_documents"),
        ("paper_chaser_mcp.agentic.graphs", "_source_record_from_regulatory_document"),
        ("paper_chaser_mcp.agentic.planner", "_estimate_ambiguity_level"),
        ("paper_chaser_mcp.agentic.planner", "_estimate_query_specificity"),
        ("paper_chaser_mcp.agentic.planner", "_has_literature_corroboration"),
        ("paper_chaser_mcp.agentic.planner", "_is_definitional_query"),
        ("paper_chaser_mcp.agentic.planner", "_top_evidence_phrases"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_AdequacyJudgmentSchema"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_AnswerSchema"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_EvidenceGapSchema"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_ExpansionListSchema"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_ExpansionSchema"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_PlannerConstraintsSchema"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_PlannerResponseSchema"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_PlannerSubjectCardSchema"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_RelevanceBatchSchema"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_RelevanceClassificationItem"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_ReviseStrategySchema"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_extract_json_object"),
        ("paper_chaser_mcp.agentic.provider_helpers", "_normalize_answer_schema_output"),
        ("paper_chaser_mcp.agentic.providers", "_PlannerConstraintsSchema"),
        ("paper_chaser_mcp.agentic.providers", "_PlannerResponseSchema"),
        ("paper_chaser_mcp.agentic.providers", "_coerce_langchain_structured_response"),
        ("paper_chaser_mcp.agentic.providers", "_cosine_similarity"),
        ("paper_chaser_mcp.agentic.providers", "_extract_seed_identifiers"),
        ("paper_chaser_mcp.agentic.providers", "_lexical_similarity"),
        ("paper_chaser_mcp.agentic.providers", "_normalize_confidence_label"),
        ("paper_chaser_mcp.agentic.providers", "_normalized_embedding_text"),
        ("paper_chaser_mcp.agentic.providers", "_tokenize"),
        ("paper_chaser_mcp.agentic.providers", "_top_terms"),
        ("paper_chaser_mcp.agentic.relevance_fallback", "_compose_classification_rationale"),
        ("paper_chaser_mcp.agentic.relevance_fallback", "_signal_profile"),
        ("paper_chaser_mcp.agentic.workspace", "_cosine_similarity"),
        ("paper_chaser_mcp.agentic.workspace", "_tokenize"),
        ("paper_chaser_mcp.agentic.workspace", "_vectorize"),
        ("paper_chaser_mcp.citation_repair", "_classify_resolution_confidence"),
        ("paper_chaser_mcp.citation_repair", "_filtered_alternative_candidates"),
        ("paper_chaser_mcp.citation_repair", "_normalize_identifier_for_openalex"),
        ("paper_chaser_mcp.citation_repair", "_normalize_identifier_for_semantic_scholar"),
        ("paper_chaser_mcp.citation_repair", "_rank_candidate"),
        ("paper_chaser_mcp.citation_repair", "_serialize_citation_response"),
        ("paper_chaser_mcp.citation_repair", "_sparse_search_queries"),
        ("paper_chaser_mcp.citation_repair", "_title_similarity"),
        ("paper_chaser_mcp.citation_repair", "_venue_hint_in_text"),
        ("paper_chaser_mcp.clients.serpapi.normalize", "_parse_year_range"),
        ("paper_chaser_mcp.dispatch", "_answer_follow_up_from_session_state"),
        ("paper_chaser_mcp.dispatch", "_apply_follow_up_response_mode"),
        ("paper_chaser_mcp.dispatch", "_assign_verification_status"),
        ("paper_chaser_mcp.dispatch", "_authoritative_but_weak_source_ids"),
        ("paper_chaser_mcp.dispatch", "_cursor_to_offset"),
        ("paper_chaser_mcp.dispatch", "_guided_citation_from_paper"),
        ("paper_chaser_mcp.dispatch", "_guided_confidence_signals"),
        ("paper_chaser_mcp.dispatch", "_guided_contract_fields"),
        ("paper_chaser_mcp.dispatch", "_guided_failure_summary"),
        ("paper_chaser_mcp.dispatch", "_guided_finalize_response"),
        ("paper_chaser_mcp.dispatch", "_guided_is_mixed_intent_query"),
        ("paper_chaser_mcp.dispatch", "_guided_mentions_literature"),
        ("paper_chaser_mcp.dispatch", "_guided_session_state"),
        ("paper_chaser_mcp.dispatch", "_guided_should_add_review_pass"),
        ("paper_chaser_mcp.dispatch", "_guided_source_record_from_paper"),
        ("paper_chaser_mcp.dispatch", "_guided_source_record_from_structured_source"),
        ("paper_chaser_mcp.dispatch", "_guided_trust_summary"),
        ("paper_chaser_mcp.eval_publish", "_create_ai_project_client"),
        ("paper_chaser_mcp.eval_publish", "_create_default_credential"),
        ("paper_chaser_mcp.eval_publish", "_create_hf_api"),
        ("paper_chaser_mcp.parsing", "_arxiv_id_from_url"),
        ("paper_chaser_mcp.parsing", "_text"),
        ("paper_chaser_mcp.provider_runtime", "_classify_exception"),
        ("paper_chaser_mcp.provider_runtime", "_default_fallback_reason"),
        ("paper_chaser_mcp.provider_runtime", "_empty_payload"),
        ("paper_chaser_mcp.provider_runtime", "_format_exception"),
        ("paper_chaser_mcp.provider_runtime", "_log_request_scoped_provider_event"),
        ("paper_chaser_mcp.provider_runtime", "_provider_semaphore_key"),
        ("paper_chaser_mcp.provider_runtime", "_quota_metadata_from_payload"),
        ("paper_chaser_mcp.provider_runtime", "_retry_delay_seconds"),
        ("paper_chaser_mcp.provider_runtime", "_shorten_runtime_log_text"),
        ("paper_chaser_mcp.search", "_dump_search_response"),
        ("paper_chaser_mcp.search", "_enrich_ss_paper"),
        ("paper_chaser_mcp.search", "_metadata"),
        ("paper_chaser_mcp.search", "_result_quality"),
        ("paper_chaser_mcp.server", "_execute_tool"),
    }
)


def _discover_private_seams() -> set[tuple[str, str]]:
    """Walk ``tests/`` and collect private ``from paper_chaser_mcp.*`` imports."""

    seams: set[tuple[str, str]] = set()
    for path in sorted(TESTS_DIR.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module
            if module is None or not module.startswith(PACKAGE_PREFIX):
                continue
            for alias in node.names:
                name = alias.name
                if not name.startswith("_"):
                    continue
                if name.startswith("__") and name.endswith("__"):
                    continue
                seams.add((module, name))
    return seams


def test_no_new_private_test_seams() -> None:
    """Fail if any NEW private ``paper_chaser_mcp`` seam appears in tests.

    If a later phase intentionally adds a new private test-only seam, update
    ``KNOWN_TEST_SEAMS`` in the same change so reviewers see the growth.
    """

    discovered = _discover_private_seams()
    new_seams = discovered - KNOWN_TEST_SEAMS
    assert not new_seams, (
        "New private paper_chaser_mcp test seams detected. Either eliminate the "
        "private import or add it to KNOWN_TEST_SEAMS explicitly: "
        f"{sorted(new_seams)}"
    )


def test_known_seams_still_importable() -> None:
    """Every pinned seam must remain importable.

    A Phase 2+ refactor that moves code must update ``KNOWN_TEST_SEAMS`` or
    preserve the name via re-export; otherwise this test fails.
    """

    missing: list[tuple[str, str]] = []
    for module_name, attr_name in sorted(KNOWN_TEST_SEAMS):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing.append((module_name, attr_name))
            continue
        if not hasattr(module, attr_name):
            missing.append((module_name, attr_name))
    assert not missing, (
        "Pinned private test seams are no longer importable. Either restore "
        "the re-export or update KNOWN_TEST_SEAMS: "
        f"{missing}"
    )
