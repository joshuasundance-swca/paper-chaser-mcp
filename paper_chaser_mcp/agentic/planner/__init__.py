"""Planner package facade.

Phase 6 converted ``paper_chaser_mcp/agentic/planner.py`` into a package so its
helpers can be extracted into focused submodules without disturbing the
existing import surface. This facade re-exports every module-level symbol the
old flat module exposed so callers that do
``from paper_chaser_mcp.agentic.planner import X`` keep working.

Phase 6 extracts the low-risk foundations (constants, normalization,
regulatory/literature intent helpers, specificity estimators) into their own
submodules. The async ``classify_query`` orchestrator plus hypothesis,
grounded/speculative expansion, and variant reconciliation remain in
``_core`` and are slated for Phase 7.
"""

from __future__ import annotations

from ...citation_repair import (
    looks_like_citation_query as looks_like_citation_query,
)
from . import _core as _core
from .constants import (
    _CULTURAL_RESOURCE_MARKERS as _CULTURAL_RESOURCE_MARKERS,
)
from .constants import (
    _DEFINITIONAL_PATTERNS as _DEFINITIONAL_PATTERNS,
)
from .constants import (
    AGENCY_REGULATORY_MARKERS as AGENCY_REGULATORY_MARKERS,
)
from .constants import (
    ARXIV_RE as ARXIV_RE,
)
from .constants import (
    DOI_RE as DOI_RE,
)
from .constants import (
    FACET_SPLIT_RE as FACET_SPLIT_RE,
)
from .constants import (
    GENERIC_EVIDENCE_WORDS as GENERIC_EVIDENCE_WORDS,
)
from .constants import (
    HYPOTHESIS_QUERY_STOPWORDS as HYPOTHESIS_QUERY_STOPWORDS,
)
from .constants import (
    LITERATURE_QUERY_TERMS as LITERATURE_QUERY_TERMS,
)
from .constants import (
    QUERY_FACET_TOKEN_ALLOWLIST as QUERY_FACET_TOKEN_ALLOWLIST,
)
from .constants import (
    QUERYISH_TITLE_BLOCKERS as QUERYISH_TITLE_BLOCKERS,
)
from .constants import (
    REGULATORY_QUERY_TERMS as REGULATORY_QUERY_TERMS,
)
from .constants import (
    STRONG_REGULATORY_TITLE_BLOCKERS as STRONG_REGULATORY_TITLE_BLOCKERS,
)
from .constants import (
    TITLE_STOPWORDS as TITLE_STOPWORDS,
)
from .constants import (
    VARIANT_DEDUPE_STOPWORDS as VARIANT_DEDUPE_STOPWORDS,
)
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
    looks_like_exact_title as looks_like_exact_title,
)
from .normalization import (
    looks_like_near_known_item_query as looks_like_near_known_item_query,
)
from .normalization import (
    looks_like_url as looks_like_url,
)
from .normalization import (
    normalize_query as normalize_query,
)
from .normalization import (
    query_facets as query_facets,
)
from .normalization import (
    query_terms as query_terms,
)
from .orchestrator import (
    classify_query as classify_query,
)
from .orchestrator import (
    grounded_expansion_candidates as grounded_expansion_candidates,
)
from .orchestrator import (
    speculative_expansion_candidates as speculative_expansion_candidates,
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
    _detect_cultural_resource_intent as _detect_cultural_resource_intent,
)
from .regulatory import (
    _infer_entity_card as _infer_entity_card,
)
from .regulatory import (
    _infer_regulatory_subintent as _infer_regulatory_subintent,
)
from .regulatory import (
    _strong_known_item_signal as _strong_known_item_signal,
)
from .regulatory import (
    _strong_regulatory_signal as _strong_regulatory_signal,
)
from .regulatory import (
    detect_literature_intent as detect_literature_intent,
)
from .regulatory import (
    detect_regulatory_intent as detect_regulatory_intent,
)
from .specificity import (
    _confidence_rank as _confidence_rank,
)
from .specificity import (
    _estimate_ambiguity_level as _estimate_ambiguity_level,
)
from .specificity import (
    _estimate_query_specificity as _estimate_query_specificity,
)
from .specificity import (
    _is_definitional_query as _is_definitional_query,
)
from .specificity import (
    _looks_broad_concept_query as _looks_broad_concept_query,
)
from .specificity import (
    _query_starts_broad as _query_starts_broad,
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

__all__ = [
    "classify_query",
    "combine_variants",
    "dedupe_variants",
    "detect_literature_intent",
    "detect_regulatory_intent",
    "grounded_expansion_candidates",
    "initial_retrieval_hypotheses",
    "looks_like_exact_title",
    "looks_like_near_known_item_query",
    "looks_like_url",
    "normalize_query",
    "query_facets",
    "query_terms",
    "speculative_expansion_candidates",
]
