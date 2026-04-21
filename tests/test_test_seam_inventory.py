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
import importlib.util
from pathlib import Path

TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
PACKAGE_PREFIX = "paper_chaser_mcp"


KNOWN_TEST_SEAMS: frozenset[tuple[str, str]] = frozenset(
    {
        ("paper_chaser_mcp.agentic.graphs", "_FACADE_EXPORTS"),
        ("paper_chaser_mcp.agentic.graphs", "_build_grounded_comparison_answer"),
        ("paper_chaser_mcp.agentic.graphs", "_classify_topical_relevance_with_provenance"),
        ("paper_chaser_mcp.agentic.graphs", "_core"),
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
        ("paper_chaser_mcp.agentic.graphs._core", "_anchor_strength_for_resolution"),
        ("paper_chaser_mcp.agentic.graphs._core", "_best_next_internal_action"),
        ("paper_chaser_mcp.agentic.graphs._core", "_build_grounded_comparison_answer"),
        ("paper_chaser_mcp.agentic.graphs._core", "_cluster_papers"),
        ("paper_chaser_mcp.agentic.graphs._core", "_comparison_requested"),
        ("paper_chaser_mcp.agentic.graphs._core", "_comparison_takeaway"),
        ("paper_chaser_mcp.agentic.graphs._core", "_compute_disagreements"),
        ("paper_chaser_mcp.agentic.graphs._core", "_compute_gaps"),
        ("paper_chaser_mcp.agentic.graphs._core", "_contextualize_follow_up_question"),
        ("paper_chaser_mcp.agentic.graphs._core", "_dedupe_variants"),
        ("paper_chaser_mcp.agentic.graphs._core", "_filter_graph_frontier"),
        ("paper_chaser_mcp.agentic.graphs._core", "_finalize_theme_label"),
        ("paper_chaser_mcp.agentic.graphs._core", "_graph_frontier_scores"),
        ("paper_chaser_mcp.agentic.graphs._core", "_graph_intent_text"),
        ("paper_chaser_mcp.agentic.graphs._core", "_has_inspectable_sources"),
        ("paper_chaser_mcp.agentic.graphs._core", "_has_on_topic_sources"),
        ("paper_chaser_mcp.agentic.graphs._core", "_initial_retrieval_query_text"),
        ("paper_chaser_mcp.agentic.graphs._core", "_known_item_recovery_warning"),
        ("paper_chaser_mcp.agentic.graphs._core", "_known_item_resolution_queries"),
        ("paper_chaser_mcp.agentic.graphs._core", "_known_item_resolution_state_for_strategy"),
        ("paper_chaser_mcp.agentic.graphs._core", "_known_item_title_similarity"),
        ("paper_chaser_mcp.agentic.graphs._core", "_label_tokens"),
        ("paper_chaser_mcp.agentic.graphs._core", "_looks_like_title_venue_list"),
        ("paper_chaser_mcp.agentic.graphs._core", "_normalization_metadata"),
        ("paper_chaser_mcp.agentic.graphs._core", "_normalized_theme_label"),
        ("paper_chaser_mcp.agentic.graphs._core", "_paid_providers_used"),
        ("paper_chaser_mcp.agentic.graphs._core", "_paper_focus_phrase"),
        ("paper_chaser_mcp.agentic.graphs._core", "_result_coverage_label"),
        ("paper_chaser_mcp.agentic.graphs._core", "_shared_focus_terms"),
        ("paper_chaser_mcp.agentic.graphs._core", "_should_use_structured_comparison_answer"),
        ("paper_chaser_mcp.agentic.graphs._core", "_smart_coverage_summary"),
        ("paper_chaser_mcp.agentic.graphs._core", "_smart_failure_summary"),
        ("paper_chaser_mcp.agentic.graphs._core", "_smart_provider_fallback_warnings"),
        ("paper_chaser_mcp.agentic.graphs._core", "_suggest_next_searches"),
        ("paper_chaser_mcp.agentic.graphs._core", "_theme_terms_from_papers"),
        ("paper_chaser_mcp.agentic.graphs._core", "_top_terms_for_cluster"),
        ("paper_chaser_mcp.agentic.graphs.followup_graph", "_build_grounded_comparison_answer"),
        ("paper_chaser_mcp.agentic.graphs.followup_graph", "_comparison_requested"),
        ("paper_chaser_mcp.agentic.graphs.followup_graph", "_comparison_takeaway"),
        ("paper_chaser_mcp.agentic.graphs.followup_graph", "_contextualize_follow_up_question"),
        ("paper_chaser_mcp.agentic.graphs.followup_graph", "_looks_like_title_venue_list"),
        ("paper_chaser_mcp.agentic.graphs.followup_graph", "_paper_focus_phrase"),
        ("paper_chaser_mcp.agentic.graphs.followup_graph", "_shared_focus_terms"),
        ("paper_chaser_mcp.agentic.graphs.followup_graph", "_should_use_structured_comparison_answer"),
        ("paper_chaser_mcp.agentic.graphs.hooks", "_consume_background_task"),
        ("paper_chaser_mcp.agentic.graphs.hooks", "_describe_retrieval_batch"),
        ("paper_chaser_mcp.agentic.graphs.hooks", "_skip_context_notifications"),
        ("paper_chaser_mcp.agentic.graphs.hooks", "_truncate_text"),
        ("paper_chaser_mcp.agentic.graphs.inspect_graph", "_cluster_papers"),
        ("paper_chaser_mcp.agentic.graphs.inspect_graph", "_compute_disagreements"),
        ("paper_chaser_mcp.agentic.graphs.inspect_graph", "_compute_gaps"),
        ("paper_chaser_mcp.agentic.graphs.inspect_graph", "_finalize_theme_label"),
        ("paper_chaser_mcp.agentic.graphs.inspect_graph", "_label_tokens"),
        ("paper_chaser_mcp.agentic.graphs.inspect_graph", "_normalized_theme_label"),
        ("paper_chaser_mcp.agentic.graphs.inspect_graph", "_suggest_next_searches"),
        ("paper_chaser_mcp.agentic.graphs.inspect_graph", "_theme_terms_from_papers"),
        ("paper_chaser_mcp.agentic.graphs.inspect_graph", "_top_terms_for_cluster"),
        ("paper_chaser_mcp.agentic.graphs.regulatory_routing", "_ecos_query_variants"),
        ("paper_chaser_mcp.agentic.graphs.regulatory_routing", "_format_cfr_citation"),
        ("paper_chaser_mcp.agentic.graphs.regulatory_routing", "_is_agency_guidance_query"),
        ("paper_chaser_mcp.agentic.graphs.regulatory_routing", "_is_current_cfr_text_request"),
        ("paper_chaser_mcp.agentic.graphs.regulatory_routing", "_is_opaque_query"),
        ("paper_chaser_mcp.agentic.graphs.regulatory_routing", "_parse_cfr_request"),
        ("paper_chaser_mcp.agentic.graphs.research_graph", "_filter_graph_frontier"),
        ("paper_chaser_mcp.agentic.graphs.research_graph", "_graph_frontier_scores"),
        ("paper_chaser_mcp.agentic.graphs.research_graph", "_graph_intent_text"),
        ("paper_chaser_mcp.agentic.graphs.resolve_graph", "_anchor_strength_for_resolution"),
        ("paper_chaser_mcp.agentic.graphs.resolve_graph", "_known_item_recovery_warning"),
        ("paper_chaser_mcp.agentic.graphs.resolve_graph", "_known_item_resolution_queries"),
        ("paper_chaser_mcp.agentic.graphs.resolve_graph", "_known_item_resolution_state_for_strategy"),
        ("paper_chaser_mcp.agentic.graphs.resolve_graph", "_known_item_title_similarity"),
        ("paper_chaser_mcp.agentic.graphs.resolve_graph", "_normalization_metadata"),
        ("paper_chaser_mcp.agentic.graphs.shared_state", "_COMPARISON_FOCUS_STOPWORDS"),
        ("paper_chaser_mcp.agentic.graphs.shared_state", "_GRAPH_GENERIC_TERMS"),
        ("paper_chaser_mcp.agentic.graphs.shared_state", "_THEME_LABEL_STOPWORDS"),
        ("paper_chaser_mcp.agentic.graphs.smart_graph", "_dedupe_variants"),
        ("paper_chaser_mcp.agentic.graphs.smart_graph", "_initial_retrieval_query_text"),
        ("paper_chaser_mcp.agentic.graphs.smart_graph", "_result_coverage_label"),
        ("paper_chaser_mcp.agentic.graphs.smart_graph", "_smart_failure_summary"),
        ("paper_chaser_mcp.agentic.graphs.smart_helpers", "_best_next_internal_action"),
        ("paper_chaser_mcp.agentic.graphs.smart_helpers", "_has_inspectable_sources"),
        ("paper_chaser_mcp.agentic.graphs.smart_helpers", "_has_on_topic_sources"),
        ("paper_chaser_mcp.agentic.graphs.smart_helpers", "_paid_providers_used"),
        ("paper_chaser_mcp.agentic.graphs.smart_helpers", "_smart_coverage_summary"),
        ("paper_chaser_mcp.agentic.graphs.smart_helpers", "_smart_provider_fallback_warnings"),
        ("paper_chaser_mcp.agentic.graphs.source_records", "_answerability_from_source_records"),
        ("paper_chaser_mcp.agentic.graphs.source_records", "_candidate_leads_from_source_records"),
        ("paper_chaser_mcp.agentic.graphs.source_records", "_classify_topical_relevance"),
        ("paper_chaser_mcp.agentic.graphs.source_records", "_evidence_from_source_records"),
        ("paper_chaser_mcp.agentic.graphs.source_records", "_graph_topic_tokens"),
        ("paper_chaser_mcp.agentic.graphs.source_records", "_paper_text"),
        ("paper_chaser_mcp.agentic.graphs.source_records", "_routing_summary_from_strategy"),
        ("paper_chaser_mcp.agentic.graphs.source_records", "_year_text"),
        ("paper_chaser_mcp.agentic.planner", "_CULTURAL_RESOURCE_MARKERS"),
        ("paper_chaser_mcp.agentic.planner", "_DEFINITIONAL_PATTERNS"),
        ("paper_chaser_mcp.agentic.planner", "_FACADE_EXPORTS"),
        ("paper_chaser_mcp.agentic.planner", "_core"),
        ("paper_chaser_mcp.agentic.planner", "_detect_cultural_resource_intent"),
        ("paper_chaser_mcp.agentic.planner", "_estimate_ambiguity_level"),
        ("paper_chaser_mcp.agentic.planner", "_estimate_query_specificity"),
        ("paper_chaser_mcp.agentic.planner", "_has_literature_corroboration"),
        ("paper_chaser_mcp.agentic.planner", "_infer_entity_card"),
        ("paper_chaser_mcp.agentic.planner", "_infer_regulatory_subintent"),
        ("paper_chaser_mcp.agentic.planner", "_is_definitional_query"),
        ("paper_chaser_mcp.agentic.planner", "_looks_broad_concept_query"),
        ("paper_chaser_mcp.agentic.planner", "_query_starts_broad"),
        ("paper_chaser_mcp.agentic.planner", "_strong_known_item_signal"),
        ("paper_chaser_mcp.agentic.planner", "_strong_regulatory_signal"),
        ("paper_chaser_mcp.agentic.planner", "_top_evidence_phrases"),
        ("paper_chaser_mcp.agentic.planner._core", "_VALID_REGULATORY_INTENTS"),
        ("paper_chaser_mcp.agentic.planner._core", "_derive_regulatory_intent"),
        ("paper_chaser_mcp.agentic.planner._core", "_has_literature_corroboration"),
        ("paper_chaser_mcp.agentic.planner._core", "_ordered_provider_plan"),
        ("paper_chaser_mcp.agentic.planner._core", "_signatures_are_near_duplicates"),
        ("paper_chaser_mcp.agentic.planner._core", "_sort_intent_candidates"),
        ("paper_chaser_mcp.agentic.planner._core", "_source_for_intent_candidate"),
        ("paper_chaser_mcp.agentic.planner._core", "_top_evidence_phrases"),
        ("paper_chaser_mcp.agentic.planner._core", "_upsert_intent_candidate"),
        ("paper_chaser_mcp.agentic.planner._core", "_variant_signature"),
        ("paper_chaser_mcp.agentic.planner.constants", "_CULTURAL_RESOURCE_MARKERS"),
        ("paper_chaser_mcp.agentic.planner.constants", "_DEFINITIONAL_PATTERNS"),
        ("paper_chaser_mcp.agentic.planner.hypotheses", "_ordered_provider_plan"),
        ("paper_chaser_mcp.agentic.planner.hypotheses", "_sort_intent_candidates"),
        ("paper_chaser_mcp.agentic.planner.hypotheses", "_source_for_intent_candidate"),
        ("paper_chaser_mcp.agentic.planner.hypotheses", "_upsert_intent_candidate"),
        ("paper_chaser_mcp.agentic.planner.reconciliation", "_VALID_REGULATORY_INTENTS"),
        ("paper_chaser_mcp.agentic.planner.reconciliation", "_derive_regulatory_intent"),
        ("paper_chaser_mcp.agentic.planner.reconciliation", "_has_literature_corroboration"),
        ("paper_chaser_mcp.agentic.planner.regulatory", "_detect_cultural_resource_intent"),
        ("paper_chaser_mcp.agentic.planner.regulatory", "_infer_entity_card"),
        ("paper_chaser_mcp.agentic.planner.regulatory", "_infer_regulatory_subintent"),
        ("paper_chaser_mcp.agentic.planner.regulatory", "_strong_known_item_signal"),
        ("paper_chaser_mcp.agentic.planner.regulatory", "_strong_regulatory_signal"),
        ("paper_chaser_mcp.agentic.planner.specificity", "_estimate_ambiguity_level"),
        ("paper_chaser_mcp.agentic.planner.specificity", "_estimate_query_specificity"),
        ("paper_chaser_mcp.agentic.planner.specificity", "_is_definitional_query"),
        ("paper_chaser_mcp.agentic.planner.specificity", "_looks_broad_concept_query"),
        ("paper_chaser_mcp.agentic.planner.specificity", "_query_starts_broad"),
        ("paper_chaser_mcp.agentic.planner.variants", "_signatures_are_near_duplicates"),
        ("paper_chaser_mcp.agentic.planner.variants", "_top_evidence_phrases"),
        ("paper_chaser_mcp.agentic.planner.variants", "_variant_signature"),
        ("paper_chaser_mcp.agentic.provider_base", "_default_selected_evidence_ids"),
        ("paper_chaser_mcp.agentic.provider_base", "_fallback_query_facets"),
        ("paper_chaser_mcp.agentic.provider_base", "_fallback_query_terms"),
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
        ("paper_chaser_mcp.agentic.providers.base.classification", "_fallback_query_facets"),
        ("paper_chaser_mcp.agentic.providers.base.classification", "_fallback_query_terms"),
        ("paper_chaser_mcp.agentic.providers.base.identifiers", "_default_selected_evidence_ids"),
        ("paper_chaser_mcp.agentic.relevance_fallback", "_compose_classification_rationale"),
        ("paper_chaser_mcp.agentic.relevance_fallback", "_signal_profile"),
        ("paper_chaser_mcp.agentic.workspace", "_cosine_similarity"),
        ("paper_chaser_mcp.agentic.workspace", "_tokenize"),
        ("paper_chaser_mcp.agentic.workspace", "_vectorize"),
        ("paper_chaser_mcp.citation_repair", "_classify_resolution_confidence"),
        ("paper_chaser_mcp.citation_repair", "_core"),
        ("paper_chaser_mcp.citation_repair", "_filtered_alternative_candidates"),
        ("paper_chaser_mcp.citation_repair", "_normalize_identifier_for_openalex"),
        ("paper_chaser_mcp.citation_repair", "_normalize_identifier_for_semantic_scholar"),
        ("paper_chaser_mcp.citation_repair", "_rank_candidate"),
        ("paper_chaser_mcp.citation_repair", "_serialize_citation_response"),
        ("paper_chaser_mcp.citation_repair", "_sparse_search_queries"),
        ("paper_chaser_mcp.citation_repair", "_title_similarity"),
        ("paper_chaser_mcp.citation_repair", "_venue_hint_in_text"),
        ("paper_chaser_mcp.citation_repair._core", "_author_overlap"),
        ("paper_chaser_mcp.citation_repair._core", "_identifier_hit"),
        ("paper_chaser_mcp.citation_repair._core", "_publication_preference_score"),
        ("paper_chaser_mcp.citation_repair._core", "_rank_candidate"),
        ("paper_chaser_mcp.citation_repair._core", "_snippet_alignment"),
        ("paper_chaser_mcp.citation_repair._core", "_token_overlap_ratio"),
        ("paper_chaser_mcp.citation_repair._core", "_venue_overlap"),
        ("paper_chaser_mcp.citation_repair._core", "_year_delta"),
        ("paper_chaser_mcp.citation_repair.ranking", "_author_overlap"),
        ("paper_chaser_mcp.citation_repair.ranking", "_identifier_hit"),
        ("paper_chaser_mcp.citation_repair.ranking", "_publication_preference_score"),
        ("paper_chaser_mcp.citation_repair.ranking", "_rank_candidate"),
        ("paper_chaser_mcp.citation_repair.ranking", "_snippet_alignment"),
        ("paper_chaser_mcp.citation_repair.ranking", "_token_overlap_ratio"),
        ("paper_chaser_mcp.citation_repair.ranking", "_venue_overlap"),
        ("paper_chaser_mcp.citation_repair.ranking", "_year_delta"),
        ("paper_chaser_mcp.cli", "_package_version"),
        ("paper_chaser_mcp.cli", "_run_deployment_http"),
        ("paper_chaser_mcp.cli", "_run_server"),
        ("paper_chaser_mcp.clients.serpapi.normalize", "_parse_year_range"),
        ("paper_chaser_mcp.dispatch", "_FACADE_EXPORTS"),
        ("paper_chaser_mcp.dispatch", "_answer_follow_up_from_session_state"),
        ("paper_chaser_mcp.dispatch", "_apply_follow_up_response_mode"),
        ("paper_chaser_mcp.dispatch", "_assign_verification_status"),
        ("paper_chaser_mcp.dispatch", "_authoritative_but_weak_source_ids"),
        ("paper_chaser_mcp.dispatch", "_build_provider_diagnostics_snapshot"),
        ("paper_chaser_mcp.dispatch", "_compose_why_classified_weak_match"),
        ("paper_chaser_mcp.dispatch", "_core"),
        ("paper_chaser_mcp.dispatch", "_cursor_to_offset"),
        ("paper_chaser_mcp.dispatch", "_direct_read_recommendation_details"),
        ("paper_chaser_mcp.dispatch", "_direct_read_recommendations"),
        ("paper_chaser_mcp.dispatch", "_evidence_quality_detail"),
        ("paper_chaser_mcp.dispatch", "_guided_abstention_details_payload"),
        ("paper_chaser_mcp.dispatch", "_guided_best_next_internal_action"),
        ("paper_chaser_mcp.dispatch", "_guided_citation_from_paper"),
        ("paper_chaser_mcp.dispatch", "_guided_confidence_signals"),
        ("paper_chaser_mcp.dispatch", "_guided_contract_fields"),
        ("paper_chaser_mcp.dispatch", "_guided_deterministic_evidence_gaps"),
        ("paper_chaser_mcp.dispatch", "_guided_failure_summary"),
        ("paper_chaser_mcp.dispatch", "_guided_finalize_response"),
        ("paper_chaser_mcp.dispatch", "_guided_is_mixed_intent_query"),
        ("paper_chaser_mcp.dispatch", "_guided_machine_failure_payload"),
        ("paper_chaser_mcp.dispatch", "_guided_mentions_literature"),
        ("paper_chaser_mcp.dispatch", "_guided_merge_coverage_summaries"),
        ("paper_chaser_mcp.dispatch", "_guided_next_actions"),
        ("paper_chaser_mcp.dispatch", "_guided_normalize_follow_up_arguments"),
        ("paper_chaser_mcp.dispatch", "_guided_normalize_inspect_arguments"),
        ("paper_chaser_mcp.dispatch", "_guided_result_meaning"),
        ("paper_chaser_mcp.dispatch", "_guided_result_state"),
        ("paper_chaser_mcp.dispatch", "_guided_saved_session_topicality"),
        ("paper_chaser_mcp.dispatch", "_guided_session_state"),
        ("paper_chaser_mcp.dispatch", "_guided_should_add_review_pass"),
        ("paper_chaser_mcp.dispatch", "_guided_source_metadata_answers"),
        ("paper_chaser_mcp.dispatch", "_guided_source_record_from_paper"),
        ("paper_chaser_mcp.dispatch", "_guided_source_record_from_structured_source"),
        ("paper_chaser_mcp.dispatch", "_guided_sources_from_fr_documents"),
        ("paper_chaser_mcp.dispatch", "_guided_summary"),
        ("paper_chaser_mcp.dispatch", "_guided_trust_summary"),
        ("paper_chaser_mcp.dispatch", "_paper_topical_relevance"),
        ("paper_chaser_mcp.dispatch", "_synthesis_path"),
        ("paper_chaser_mcp.dispatch", "_topical_relevance_from_signals"),
        ("paper_chaser_mcp.dispatch._core", "_annotate_runtime_provider_row"),
        ("paper_chaser_mcp.dispatch._core", "_build_provider_diagnostics_snapshot"),
        ("paper_chaser_mcp.dispatch._core", "_metadata_value_is_depleted"),
        ("paper_chaser_mcp.dispatch._core", "_provider_error_text"),
        ("paper_chaser_mcp.dispatch._core", "_provider_row_quota_limited"),
        ("paper_chaser_mcp.dispatch._core", "_runtime_provider_order"),
        ("paper_chaser_mcp.dispatch._core", "_scholarapi_fallback_reason"),
        ("paper_chaser_mcp.dispatch._core", "_scholarapi_payload_is_empty"),
        ("paper_chaser_mcp.dispatch._core", "_scholarapi_quota_metadata"),
        ("paper_chaser_mcp.dispatch._core", "_scholarapi_status_bucket"),
        ("paper_chaser_mcp.dispatch._core", "_smart_runtime_provider_state"),
        ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_get_paper_citations_openalex"),
        ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_search_papers_openalex"),
        ("paper_chaser_mcp.dispatch.expert.openalex", "_dispatch_search_papers_openalex_by_entity"),
        ("paper_chaser_mcp.dispatch.expert.openalex", "_require_openalex"),
        ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_get_cfr_text"),
        ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_get_document_text_ecos"),
        ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_get_federal_register_document"),
        ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_get_species_profile_ecos"),
        ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_list_species_documents_ecos"),
        ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_search_federal_register"),
        ("paper_chaser_mcp.dispatch.expert.regulatory", "_dispatch_search_species_ecos"),
        ("paper_chaser_mcp.dispatch.guided.citations", "_assign_verification_status"),
        ("paper_chaser_mcp.dispatch.guided.citations", "_guided_citation_from_paper"),
        ("paper_chaser_mcp.dispatch.guided.citations", "_guided_citation_from_structured_source"),
        ("paper_chaser_mcp.dispatch.guided.citations", "_guided_journal_or_publisher"),
        ("paper_chaser_mcp.dispatch.guided.citations", "_guided_normalize_access_axes"),
        ("paper_chaser_mcp.dispatch.guided.citations", "_guided_normalize_verification_status"),
        ("paper_chaser_mcp.dispatch.guided.citations", "_guided_open_access_route"),
        ("paper_chaser_mcp.dispatch.guided.citations", "_guided_year_text"),
        ("paper_chaser_mcp.dispatch.guided.findings", "_guided_findings_from_sources"),
        ("paper_chaser_mcp.dispatch.guided.findings", "_guided_unverified_leads_from_sources"),
        ("paper_chaser_mcp.dispatch.guided.follow_up", "_answer_follow_up_from_session_state"),
        ("paper_chaser_mcp.dispatch.guided.follow_up", "_guided_follow_up_answer_mode"),
        ("paper_chaser_mcp.dispatch.guided.follow_up", "_guided_follow_up_introspection_facets"),
        ("paper_chaser_mcp.dispatch.guided.follow_up", "_guided_follow_up_response_mode"),
        ("paper_chaser_mcp.dispatch.guided.follow_up", "_guided_is_usable_answer_text"),
        ("paper_chaser_mcp.dispatch.guided.follow_up", "_guided_metadata_answer_is_responsive"),
        ("paper_chaser_mcp.dispatch.guided.follow_up", "_guided_relevance_triage_answers"),
        ("paper_chaser_mcp.dispatch.guided.follow_up", "_guided_requested_metadata_facets"),
        ("paper_chaser_mcp.dispatch.guided.follow_up", "_guided_source_metadata_answers"),
        ("paper_chaser_mcp.dispatch.guided.inspect_source", "_guided_append_selected_saved_records"),
        ("paper_chaser_mcp.dispatch.guided.inspect_source", "_guided_compact_source_candidate"),
        ("paper_chaser_mcp.dispatch.guided.inspect_source", "_guided_extract_question"),
        ("paper_chaser_mcp.dispatch.guided.inspect_source", "_guided_extract_source_reference_from_question"),
        ("paper_chaser_mcp.dispatch.guided.inspect_source", "_guided_select_follow_up_source"),
        ("paper_chaser_mcp.dispatch.guided.inspect_source", "_guided_source_resolution_payload"),
        ("paper_chaser_mcp.dispatch.guided.research", "_guided_normalization_payload"),
        ("paper_chaser_mcp.dispatch.guided.research", "_guided_normalize_follow_up_arguments"),
        ("paper_chaser_mcp.dispatch.guided.research", "_guided_normalize_inspect_arguments"),
        ("paper_chaser_mcp.dispatch.guided.research", "_guided_normalize_research_arguments"),
        ("paper_chaser_mcp.dispatch.guided.resolve_reference", "_guided_note_repair"),
        ("paper_chaser_mcp.dispatch.guided.resolve_reference", "_guided_underspecified_reference_clarification"),
        ("paper_chaser_mcp.dispatch.guided.response", "_guided_compact_response_if_needed"),
        ("paper_chaser_mcp.dispatch.guided.response", "_guided_contract_fields"),
        ("paper_chaser_mcp.dispatch.guided.response", "_guided_finalize_response"),
        ("paper_chaser_mcp.dispatch.guided.sessions", "_guided_active_session_ids"),
        ("paper_chaser_mcp.dispatch.guided.sessions", "_guided_enrich_records_from_saved_session"),
        ("paper_chaser_mcp.dispatch.guided.sessions", "_guided_extract_search_session_id"),
        ("paper_chaser_mcp.dispatch.guided.sessions", "_guided_infer_single_session_id"),
        ("paper_chaser_mcp.dispatch.guided.sessions", "_guided_saved_session_topicality"),
        ("paper_chaser_mcp.dispatch.guided.sessions", "_guided_session_exists"),
        ("paper_chaser_mcp.dispatch.guided.sessions", "_guided_session_findings"),
        ("paper_chaser_mcp.dispatch.guided.sources", "_guided_dedupe_source_records"),
        ("paper_chaser_mcp.dispatch.guided.sources", "_guided_extract_source_id"),
        ("paper_chaser_mcp.dispatch.guided.sources", "_guided_merge_source_record_sets"),
        ("paper_chaser_mcp.dispatch.guided.sources", "_guided_merge_source_records"),
        ("paper_chaser_mcp.dispatch.guided.sources", "_guided_source_coverage_summary"),
        ("paper_chaser_mcp.dispatch.guided.sources", "_guided_source_id"),
        ("paper_chaser_mcp.dispatch.guided.sources", "_guided_source_identity"),
        ("paper_chaser_mcp.dispatch.guided.sources", "_guided_source_matches_reference"),
        ("paper_chaser_mcp.dispatch.guided.sources", "_guided_source_record_from_paper"),
        ("paper_chaser_mcp.dispatch.guided.sources", "_guided_source_record_from_structured_source"),
        ("paper_chaser_mcp.dispatch.guided.sources", "_guided_source_records_share_surface"),
        ("paper_chaser_mcp.dispatch.guided.sources", "_guided_sources_from_fr_documents"),
        ("paper_chaser_mcp.dispatch.guided.strategy_metadata", "_guided_abstention_details_payload"),
        ("paper_chaser_mcp.dispatch.guided.strategy_metadata", "_guided_execution_provenance_payload"),
        ("paper_chaser_mcp.dispatch.guided.strategy_metadata", "_guided_is_agency_guidance_query"),
        ("paper_chaser_mcp.dispatch.guided.strategy_metadata", "_guided_is_known_item_query"),
        ("paper_chaser_mcp.dispatch.guided.strategy_metadata", "_guided_is_mixed_intent_query"),
        ("paper_chaser_mcp.dispatch.guided.strategy_metadata", "_guided_live_strategy_metadata"),
        ("paper_chaser_mcp.dispatch.guided.strategy_metadata", "_guided_mentions_literature"),
        ("paper_chaser_mcp.dispatch.guided.strategy_metadata", "_guided_provider_budget_payload"),
        ("paper_chaser_mcp.dispatch.guided.strategy_metadata", "_guided_reference_signal_words"),
        ("paper_chaser_mcp.dispatch.guided.strategy_metadata", "_guided_should_escalate_research"),
        ("paper_chaser_mcp.dispatch.guided.strategy_metadata", "_guided_strategy_metadata_from_runs"),
        ("paper_chaser_mcp.dispatch.guided.trust", "_guided_deterministic_fallback_used"),
        ("paper_chaser_mcp.dispatch.guided.trust", "_guided_failure_summary"),
        ("paper_chaser_mcp.dispatch.guided.trust", "_guided_follow_up_status"),
        ("paper_chaser_mcp.dispatch.guided.trust", "_guided_missing_evidence_type"),
        ("paper_chaser_mcp.dispatch.guided.trust", "_guided_next_actions"),
        ("paper_chaser_mcp.dispatch.guided.trust", "_guided_partial_recovery_possible"),
        ("paper_chaser_mcp.dispatch.guided.trust", "_guided_result_meaning"),
        ("paper_chaser_mcp.dispatch.guided.trust", "_guided_sources_all_off_topic"),
        ("paper_chaser_mcp.dispatch.guided.trust", "_guided_summary"),
        ("paper_chaser_mcp.dispatch.guided.trust", "_guided_trust_summary"),
        ("paper_chaser_mcp.dispatch.normalization", "_guided_normalize_citation_surface"),
        ("paper_chaser_mcp.dispatch.normalization", "_guided_normalize_source_locator"),
        ("paper_chaser_mcp.dispatch.normalization", "_guided_normalize_whitespace"),
        ("paper_chaser_mcp.dispatch.normalization", "_guided_normalize_year_hint"),
        ("paper_chaser_mcp.dispatch.normalization", "_guided_strip_research_prefix"),
        ("paper_chaser_mcp.dispatch.paging", "_cursor_to_offset"),
        ("paper_chaser_mcp.dispatch.paging", "_encode_next_cursor"),
        ("paper_chaser_mcp.dispatch.relevance", "_facet_match"),
        ("paper_chaser_mcp.dispatch.relevance", "_paper_topical_relevance"),
        ("paper_chaser_mcp.dispatch.relevance", "_tokenize_relevance_text"),
        ("paper_chaser_mcp.dispatch.relevance", "_topical_relevance_from_signals"),
        ("paper_chaser_mcp.dispatch.runtime", "_annotate_runtime_provider_row"),
        ("paper_chaser_mcp.dispatch.runtime", "_build_provider_diagnostics_snapshot"),
        ("paper_chaser_mcp.dispatch.runtime", "_metadata_value_is_depleted"),
        ("paper_chaser_mcp.dispatch.runtime", "_provider_row_quota_limited"),
        ("paper_chaser_mcp.dispatch.runtime", "_runtime_provider_order"),
        ("paper_chaser_mcp.dispatch.runtime", "_smart_runtime_provider_state"),
        ("paper_chaser_mcp.dispatch.scholarapi", "_provider_error_text"),
        ("paper_chaser_mcp.dispatch.scholarapi", "_scholarapi_fallback_reason"),
        ("paper_chaser_mcp.dispatch.scholarapi", "_scholarapi_payload_is_empty"),
        ("paper_chaser_mcp.dispatch.scholarapi", "_scholarapi_quota_metadata"),
        ("paper_chaser_mcp.dispatch.scholarapi", "_scholarapi_status_bucket"),
        ("paper_chaser_mcp.dispatch.smart.ask", "_dispatch_ask_result_set"),
        ("paper_chaser_mcp.dispatch.smart.graph", "_dispatch_expand_research_graph"),
        ("paper_chaser_mcp.dispatch.smart.landscape", "_dispatch_map_research_landscape"),
        ("paper_chaser_mcp.dispatch.smart.search", "_dispatch_search_papers_smart"),
        ("paper_chaser_mcp.dispatch.snippet_fallback", "_maybe_fallback_snippet_search"),
        ("paper_chaser_mcp.dispatch.snippet_fallback", "_snippet_fallback_query"),
        ("paper_chaser_mcp.dispatch.snippet_fallback", "_snippet_fallback_results"),
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
        ("paper_chaser_mcp.server", "_arxiv_id_from_url"),
        ("paper_chaser_mcp.server", "_core_response_to_merged"),
        ("paper_chaser_mcp.server", "_env_bool"),
        ("paper_chaser_mcp.server", "_execute_tool"),
        ("paper_chaser_mcp.server", "_initialize_runtime"),
        ("paper_chaser_mcp.server", "_merge_search_results"),
        ("paper_chaser_mcp.server", "_text"),
    }
)


def _is_importable_submodule(dotted: str) -> bool:
    """Return ``True`` when ``dotted`` names a real, importable module.

    The Phase 7c-1 rubber-duck review flagged a recurring bypass in new
    planner tests: ``from ... import submodule as handle`` followed by
    ``handle._private`` dotted access. That pattern slipped past the original
    from-import-only walker. To close it we need to tell module-handle
    imports apart from symbol imports; a module handle is exactly a name
    that ``importlib.util.find_spec`` can resolve as a submodule/package.
    """

    if not dotted.startswith(PACKAGE_PREFIX):
        return False
    try:
        spec = importlib.util.find_spec(dotted)
    except (ImportError, ValueError, ModuleNotFoundError):  # pragma: no cover - defensive
        return False
    return spec is not None


def _collect_module_handles(tree: ast.AST) -> dict[str, str]:
    """Return ``{binding_name: dotted_module}`` for paper_chaser_mcp imports.

    Recognises three handle-introducing forms:

    * ``import paper_chaser_mcp.X.Y`` - binds ``paper_chaser_mcp`` to the
      top-level package (downstream dotted access on that name is typically
      ``paper_chaser_mcp.X.Y.something``, which is an ``ast.Attribute`` chain
      whose innermost ``ast.Name`` is ``paper_chaser_mcp``; we still track
      the top-level binding so direct ``paper_chaser_mcp._private`` access
      would be caught, even though it is not idiomatic today).
    * ``import paper_chaser_mcp.X.Y as handle`` - binds ``handle`` to the
      full dotted module path.
    * ``from paper_chaser_mcp.X import Y`` (with or without ``as handle``)
      where ``Y`` resolves to an importable submodule - binds ``Y`` (or its
      alias) to ``paper_chaser_mcp.X.Y``. Symbol imports (``Y`` is a value,
      not a submodule) are deliberately skipped here because they are
      already covered by the from-import walker below.
    """

    handles: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_dotted = alias.name
                if not module_dotted.startswith(PACKAGE_PREFIX):
                    continue
                if alias.asname is not None:
                    handles[alias.asname] = module_dotted
                else:
                    # ``import X.Y`` exposes ``X`` in the local namespace.
                    top = module_dotted.split(".", 1)[0]
                    handles[top] = top
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if module is None or not module.startswith(PACKAGE_PREFIX):
                continue
            for alias in node.names:
                candidate = f"{module}.{alias.name}"
                if not _is_importable_submodule(candidate):
                    continue
                binding = alias.asname or alias.name
                handles[binding] = candidate
    return handles


def _collect_virtual_private_seams(tree: ast.AST, handles: dict[str, str]) -> set[tuple[str, str]]:
    """Discover ``handle._private`` dotted accesses as synthesized seams.

    For every ``ast.Attribute`` whose ``value`` is an ``ast.Name`` bound to
    a module handle, if ``attr`` starts with ``_`` and is not a Python
    dunder, synthesize ``(handle_dotted_module, attr)`` as a virtual seam.
    These are folded into the main inventory so the same allowlist governs
    both from-imports and handle-based dotted access.

    Limitations: the walker tracks only *top-level* module bindings. It does
    not follow aliases-of-aliases (``x = module_handle``), does not resolve
    attribute chains deeper than one hop (``hyp.sub._helper`` - treated as
    access on ``hyp.sub`` which is not a handle), and does not catch private
    access via ``getattr(handle, "_helper")``. Those bypasses remain
    possible; the goal is preventing the common, accidental form reviewers
    saw in Phase 7c-1, not bulletproof interception.
    """

    seams: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        dotted = handles.get(node.value.id)
        if dotted is None:
            continue
        attr = node.attr
        if not attr.startswith("_"):
            continue
        if attr.startswith("__") and attr.endswith("__"):
            continue
        seams.add((dotted, attr))
    return seams


def _discover_private_seams() -> set[tuple[str, str]]:
    """Recursively walk ``tests/`` and collect every private seam reached.

    Two kinds of seams are folded into a single inventory:

    * ``from paper_chaser_mcp.X import _private`` - the classic form.
    * ``paper_chaser_mcp.X.handle._private`` attribute access where
      ``handle`` was introduced via a module-handle import (see
      :func:`_collect_module_handles`). This closes the Phase 7c-1 bypass
      where dotted private access on a ``SUBMODULE as alias`` binding
      escaped the original walker.
    """

    seams: set[tuple[str, str]] = set()
    self_name = Path(__file__).name
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if path.name == self_name:
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
        handles = _collect_module_handles(tree)
        seams |= _collect_virtual_private_seams(tree, handles)
    return seams


# Private ``paper_chaser_mcp.*`` symbols imported by files under ``scripts/``.
# This mirrors ``KNOWN_TEST_SEAMS`` but tracks the script-side surface
# separately so a refactor has to explicitly acknowledge expansion on either
# side. Today only ``scripts/run_expert_eval_batch.py`` reaches into a
# private seam; anything new must be added here or eliminated.
KNOWN_SCRIPT_SEAMS: frozenset[tuple[str, str]] = frozenset(
    {
        ("paper_chaser_mcp.server", "_execute_tool"),
    }
)


def _discover_private_script_seams() -> set[tuple[str, str]]:
    """Recursively walk ``scripts/`` and collect private ``from paper_chaser_mcp.*`` imports."""

    seams: set[tuple[str, str]] = set()
    if not SCRIPTS_DIR.exists():
        return seams
    for path in sorted(SCRIPTS_DIR.rglob("*.py")):
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


def test_no_new_private_script_seams() -> None:
    """Fail if any NEW private ``paper_chaser_mcp`` seam appears under ``scripts/``.

    The script-side private surface is tracked separately from the test
    surface. Adding a new one is a durable coupling that reviewers should
    see explicitly in the diff.
    """

    discovered = _discover_private_script_seams()
    new_seams = discovered - KNOWN_SCRIPT_SEAMS
    assert not new_seams, (
        "New private paper_chaser_mcp script seams detected. Either eliminate "
        "the private import or add it to KNOWN_SCRIPT_SEAMS explicitly: "
        f"{sorted(new_seams)}"
    )


def test_known_script_seams_still_importable() -> None:
    """Every pinned script seam must remain importable.

    Mirrors :func:`test_known_seams_still_importable` for the ``scripts/``
    tree so production entry points like ``run_expert_eval_batch.py`` do
    not silently lose their private handshake with the package.
    """

    missing: list[tuple[str, str]] = []
    for module_name, attr_name in sorted(KNOWN_SCRIPT_SEAMS):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing.append((module_name, attr_name))
            continue
        if not hasattr(module, attr_name):
            missing.append((module_name, attr_name))
    assert not missing, (
        "Pinned private script seams are no longer importable. Either restore "
        "the re-export or update KNOWN_SCRIPT_SEAMS: "
        f"{missing}"
    )
