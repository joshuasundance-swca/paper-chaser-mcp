"""Compatibility facade for the planner submodules.

Phase 7c-1 completed the planner breakup. ``classify_query`` and the grounded/
speculative expansion helpers now live in :mod:`.orchestrator`; retrieval
hypothesis generation and intent-candidate bookkeeping in :mod:`.hypotheses`;
regulatory-intent reconciliation in :mod:`.reconciliation`; and variant
combining/deduplication plus evidence-phrase extraction in :mod:`.variants`.

This module now only re-exports those symbols under
``paper_chaser_mcp.agentic.planner._core`` so existing callers and test seams
that import from it continue to work. ``symbol is _core.symbol`` identity is
preserved for every re-exported helper.
"""

from __future__ import annotations

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
