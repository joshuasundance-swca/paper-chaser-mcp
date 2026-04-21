"""Graphs package facade.

Phase 7a converted ``paper_chaser_mcp/agentic/graphs.py`` into a package so its
helpers can be extracted into focused submodules without disturbing the
existing import surface. This facade re-exports every module-level symbol the
old flat module exposed so callers that do
``from paper_chaser_mcp.agentic.graphs import X`` keep working.

Phase 7a (this commit) is a pure subpackage split with no helper extraction
yet — subsequent commits on this branch move low-risk infrastructure into
``shared_state.py``, ``source_records.py``, ``hooks.py``, and
``regulatory_routing.py``. The orchestration surface (``AgenticRuntime`` and
its methods) stays in ``_core`` and is slated for Phase 7b/7c.
"""

from __future__ import annotations

from . import _core as _core
from ._core import (
    _AGENCY_AUTHORITY_TERMS as _AGENCY_AUTHORITY_TERMS,
)
from ._core import (
    _AGENCY_GUIDANCE_DISCUSSION_TERMS as _AGENCY_GUIDANCE_DISCUSSION_TERMS,
)
from ._core import (
    _AGENCY_GUIDANCE_DOCUMENT_TERMS as _AGENCY_GUIDANCE_DOCUMENT_TERMS,
)
from ._core import (
    _AGENCY_GUIDANCE_QUERY_NOISE_TERMS as _AGENCY_GUIDANCE_QUERY_NOISE_TERMS,
)
from ._core import (
    _AGENCY_GUIDANCE_TERMS as _AGENCY_GUIDANCE_TERMS,
)
from ._core import (
    _CFR_DOC_TYPE_GENERIC as _CFR_DOC_TYPE_GENERIC,
)
from ._core import (
    _COMPARISON_FOCUS_STOPWORDS as _COMPARISON_FOCUS_STOPWORDS,
)
from ._core import (
    _COMPARISON_MARKERS as _COMPARISON_MARKERS,
)
from ._core import (
    _CULTURAL_RESOURCE_DOCUMENT_TERMS as _CULTURAL_RESOURCE_DOCUMENT_TERMS,
)
from ._core import (
    _GRAPH_GENERIC_TERMS as _GRAPH_GENERIC_TERMS,
)
from ._core import (
    _OPAQUE_ARXIV_RE as _OPAQUE_ARXIV_RE,
)
from ._core import (
    _OPAQUE_DOI_RE as _OPAQUE_DOI_RE,
)
from ._core import (
    _REGULATORY_QUERY_NOISE_TERMS as _REGULATORY_QUERY_NOISE_TERMS,
)
from ._core import (
    _REGULATORY_SUBJECT_STOPWORDS as _REGULATORY_SUBJECT_STOPWORDS,
)
from ._core import (
    _SPECIES_QUERY_NOISE_TERMS as _SPECIES_QUERY_NOISE_TERMS,
)
from ._core import (
    _THEME_LABEL_STOPWORDS as _THEME_LABEL_STOPWORDS,
)
from ._core import (
    END as END,
)
from ._core import (
    SMART_SEARCH_PROGRESS_TOTAL as SMART_SEARCH_PROGRESS_TOTAL,
)
from ._core import (
    START as START,
)
from ._core import (
    AgenticRuntime as AgenticRuntime,
)
from ._core import (
    InMemorySaver as InMemorySaver,
)
from ._core import (
    StateGraph as StateGraph,
)
from ._core import (
    TopicalRelevanceClassification as TopicalRelevanceClassification,
)
from ._core import (
    _agency_guidance_facet_terms as _agency_guidance_facet_terms,
)
from ._core import (
    _agency_guidance_priority_terms as _agency_guidance_priority_terms,
)
from ._core import (
    _agency_guidance_subject_terms as _agency_guidance_subject_terms,
)
from ._core import (
    _anchor_strength_for_resolution as _anchor_strength_for_resolution,
)
from ._core import (
    _answerability_from_source_records as _answerability_from_source_records,
)
from ._core import (
    _best_next_internal_action as _best_next_internal_action,
)
from ._core import (
    _build_anchored_selection_rationale as _build_anchored_selection_rationale,
)
from ._core import (
    _build_grounded_comparison_answer as _build_grounded_comparison_answer,
)
from ._core import (
    _build_top_recommendation_rationale as _build_top_recommendation_rationale,
)
from ._core import (
    _candidate_leads_from_source_records as _candidate_leads_from_source_records,
)
from ._core import (
    _cfr_tokens as _cfr_tokens,
)
from ._core import (
    _citation_record_from_paper as _citation_record_from_paper,
)
from ._core import (
    _citation_record_from_regulatory_document as _citation_record_from_regulatory_document,
)
from ._core import (
    _classify_topical_relevance as _classify_topical_relevance,
)
from ._core import (
    _classify_topical_relevance_for_paper as _classify_topical_relevance_for_paper,
)
from ._core import (
    _classify_topical_relevance_with_provenance as _classify_topical_relevance_with_provenance,
)
from ._core import (
    _cluster_papers as _cluster_papers,
)
from ._core import (
    _comparison_requested as _comparison_requested,
)
from ._core import (
    _comparison_takeaway as _comparison_takeaway,
)
from ._core import (
    _compute_disagreements as _compute_disagreements,
)
from ._core import (
    _compute_gaps as _compute_gaps,
)
from ._core import (
    _compute_top_recommendation as _compute_top_recommendation,
)
from ._core import (
    _contextualize_follow_up_question as _contextualize_follow_up_question,
)
from ._core import (
    _coverage_summary_line as _coverage_summary_line,
)
from ._core import (
    _dedupe_structured_sources as _dedupe_structured_sources,
)
from ._core import (
    _dedupe_variants as _dedupe_variants,
)
from ._core import (
    _derive_regulatory_query_flags as _derive_regulatory_query_flags,
)
from ._core import (
    _ecos_query_variants as _ecos_query_variants,
)
from ._core import (
    _evidence_from_source_records as _evidence_from_source_records,
)
from ._core import (
    _extract_common_name_candidate as _extract_common_name_candidate,
)
from ._core import (
    _extract_scientific_name_candidate as _extract_scientific_name_candidate,
)
from ._core import (
    _extract_subject_terms as _extract_subject_terms,
)
from ._core import (
    _filter_graph_frontier as _filter_graph_frontier,
)
from ._core import (
    _finalize_theme_label as _finalize_theme_label,
)
from ._core import (
    _format_cfr_citation as _format_cfr_citation,
)
from ._core import (
    _graph_frontier_scores as _graph_frontier_scores,
)
from ._core import (
    _graph_intent_text as _graph_intent_text,
)
from ._core import (
    _graph_topic_tokens as _graph_topic_tokens,
)
from ._core import (
    _guidance_query_prefers_recency as _guidance_query_prefers_recency,
)
from ._core import (
    _has_inspectable_sources as _has_inspectable_sources,
)
from ._core import (
    _has_on_topic_sources as _has_on_topic_sources,
)
from ._core import (
    _initial_retrieval_query_text as _initial_retrieval_query_text,
)
from ._core import (
    _is_agency_guidance_query as _is_agency_guidance_query,
)
from ._core import (
    _is_current_cfr_text_request as _is_current_cfr_text_request,
)
from ._core import (
    _is_opaque_query as _is_opaque_query,
)
from ._core import (
    _is_species_regulatory_query as _is_species_regulatory_query,
)
from ._core import (
    _known_item_recovery_warning as _known_item_recovery_warning,
)
from ._core import (
    _known_item_resolution_queries as _known_item_resolution_queries,
)
from ._core import (
    _known_item_resolution_state_for_strategy as _known_item_resolution_state_for_strategy,
)
from ._core import (
    _known_item_title_similarity as _known_item_title_similarity,
)
from ._core import (
    _label_tokens as _label_tokens,
)
from ._core import (
    _lead_reason_for_source_record as _lead_reason_for_source_record,
)
from ._core import (
    _likely_unverified_from_source_records as _likely_unverified_from_source_records,
)
from ._core import (
    _looks_like_title_venue_list as _looks_like_title_venue_list,
)
from ._core import (
    _normalization_metadata as _normalization_metadata,
)
from ._core import (
    _normalized_theme_label as _normalized_theme_label,
)
from ._core import (
    _paid_providers_used as _paid_providers_used,
)
from ._core import (
    _paper_focus_phrase as _paper_focus_phrase,
)
from ._core import (
    _paper_text as _paper_text,
)
from ._core import (
    _parse_cfr_request as _parse_cfr_request,
)
from ._core import (
    _query_requests_regulatory_history as _query_requests_regulatory_history,
)
from ._core import (
    _rank_ecos_variant_hits as _rank_ecos_variant_hits,
)
from ._core import (
    _rank_regulatory_documents as _rank_regulatory_documents,
)
from ._core import (
    _records_with_lead_reasons as _records_with_lead_reasons,
)
from ._core import (
    _regulatory_document_matches_subject as _regulatory_document_matches_subject,
)
from ._core import (
    _regulatory_query_priority_terms as _regulatory_query_priority_terms,
)
from ._core import (
    _regulatory_query_subject_terms as _regulatory_query_subject_terms,
)
from ._core import (
    _regulatory_retrieval_hypotheses as _regulatory_retrieval_hypotheses,
)
from ._core import (
    _result_coverage_label as _result_coverage_label,
)
from ._core import (
    _routing_summary_from_strategy as _routing_summary_from_strategy,
)
from ._core import (
    _selection_answer_from_recommendation as _selection_answer_from_recommendation,
)
from ._core import (
    _shared_focus_terms as _shared_focus_terms,
)
from ._core import (
    _should_use_structured_comparison_answer as _should_use_structured_comparison_answer,
)
from ._core import (
    _smart_coverage_summary as _smart_coverage_summary,
)
from ._core import (
    _smart_failure_summary as _smart_failure_summary,
)
from ._core import (
    _smart_provider_fallback_warnings as _smart_provider_fallback_warnings,
)
from ._core import (
    _source_record_from_paper as _source_record_from_paper,
)
from ._core import (
    _source_record_from_regulatory_document as _source_record_from_regulatory_document,
)
from ._core import (
    _suggest_next_searches as _suggest_next_searches,
)
from ._core import (
    _theme_terms_from_papers as _theme_terms_from_papers,
)
from ._core import (
    _top_recommendation_axis_label as _top_recommendation_axis_label,
)
from ._core import (
    _top_terms_for_cluster as _top_terms_for_cluster,
)
from ._core import (
    _truncate_text as _truncate_text,
)
from ._core import (
    _verified_findings_from_source_records as _verified_findings_from_source_records,
)
from ._core import (
    _why_matched as _why_matched,
)
from ._core import (
    _year_text as _year_text,
)
from ._core import (
    logger as logger,
)

__all__ = [
    "AgenticRuntime",
    "TopicalRelevanceClassification",
]
