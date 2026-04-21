"""Shared constants and optional-dependency stubs for the graphs subpackage.

Phase 7a extraction: this module owns the module-level constants that were
previously defined at the top of ``paper_chaser_mcp/agentic/graphs.py`` plus
the LangGraph optional-dependency stubs. Keeping them in a focused module makes
it obvious that they are pure state with no behavioural coupling to
``AgenticRuntime`` and lets other graphs submodules depend on them without
reaching into ``_core``.

These constants are consumed by the regulatory-routing helpers, source-record
classifiers, theme/label normalisation, and the orchestration code that still
lives on ``AgenticRuntime``. Tests may import them from either
``paper_chaser_mcp.agentic.graphs`` (the facade) or from
``paper_chaser_mcp.agentic.graphs.shared_state`` directly; the facade's
re-export keeps legacy callers working.
"""

from __future__ import annotations

from typing import Any

from ..provider_helpers import COMMON_QUERY_WORDS

SMART_SEARCH_PROGRESS_TOTAL = 100.0

_GRAPH_GENERIC_TERMS = COMMON_QUERY_WORDS | {
    "effect",
    "effects",
    "environmental",
    "impact",
    "impacts",
    "response",
    "responses",
    "review",
    "wildlife",
}

_COMPARISON_MARKERS = {
    "compare",
    "compared",
    "comparing",
    "comparison",
    "differences",
    "different",
    "tradeoff",
    "tradeoffs",
    "versus",
    "vs",
}

_THEME_LABEL_STOPWORDS = _GRAPH_GENERIC_TERMS | {
    "about",
    "across",
    "among",
    "analysis",
    "and",
    "approach",
    "approaches",
    "based",
    "between",
    "cluster",
    "clusters",
    "for",
    "from",
    "into",
    "method",
    "methods",
    "model",
    "models",
    "or",
    "that",
    "the",
    "these",
    "theme",
    "themes",
    "theory",
    "those",
    "using",
    "with",
}

_COMPARISON_FOCUS_STOPWORDS = _THEME_LABEL_STOPWORDS | {
    "noise",
    "paper",
    "papers",
    "results",
    "study",
    "studies",
}

_REGULATORY_SUBJECT_STOPWORDS = {
    "act",
    "administration",
    "agency",
    "analysis",
    "code",
    "codified",
    "current",
    "decision",
    "critical",
    "document",
    "documents",
    "designation",
    "drug",
    "endangered",
    "fda",
    "federal",
    "final",
    "food",
    "functions",
    "guidance",
    "habitat",
    "history",
    "industry",
    "listed",
    "listing",
    "notice",
    "part",
    "plants",
    "profile",
    "profiles",
    "recovery",
    "register",
    "regulatory",
    "review",
    "rule",
    "section",
    "software",
    "species",
    "status",
    "text",
    "threatened",
    "title",
    "under",
    "wildlife",
}

_AGENCY_GUIDANCE_TERMS = {
    "guidance",
    "guideline",
    "policy",
    "staff",
}

_AGENCY_AUTHORITY_TERMS = {
    "agency",
    "cdc",
    "cms",
    "epa",
    "fda",
    "food and drug administration",
    "hhs",
    "nih",
    "usda",
}

_AGENCY_GUIDANCE_QUERY_NOISE_TERMS = {
    "actual",
    "agency",
    "document",
    "documents",
    "drug",
    "food",
    "guidance",
    "industry",
    "management",
    "most",
    "policy",
    "recent",
    "relevant",
    "staff",
    "what",
}

_AGENCY_GUIDANCE_DOCUMENT_TERMS = {
    "advisories",
    "advisory",
    "guidance",
    "guideline",
    "guidelines",
    "notice",
    "notices",
    "policies",
    "policy",
    "recommendation",
    "recommendations",
    "roadmap",
}

_AGENCY_GUIDANCE_DISCUSSION_TERMS = {
    "concept",
    "concepts",
    "discussion",
    "framework",
    "proposal",
    "proposals",
    "proposed",
}

_CULTURAL_RESOURCE_DOCUMENT_TERMS = {
    "106",
    "achp",
    "archaeological",
    "archaeology",
    "consultation",
    "cultural",
    "heritage",
    "historic",
    "nagpra",
    "nhpa",
    "preservation",
    "sacred",
    "shpo",
    "thpo",
    "tribal",
}

_REGULATORY_QUERY_NOISE_TERMS = {
    "address",
    "actions",
    "current",
    "federal",
    "recent",
    "states",
    "united",
    "what",
}

_SPECIES_QUERY_NOISE_TERMS = {
    "about",
    "critical",
    "current",
    "cfr",
    "dossier",
    "ecos",
    "endangered",
    "federal",
    "final",
    "history",
    "listing",
    "plants",
    "profile",
    "register",
    "regulatory",
    "rule",
    "say",
    "species",
    "status",
    "text",
    "threatened",
    "under",
    "what",
    "wildlife",
}

_CFR_DOC_TYPE_GENERIC = {
    "and",
    "cfr",
    "chapter",
    "part",
    "section",
    "subchapter",
    "title",
}

# LangGraph optional-dependency stubs. When the dependency is available the
# real classes/constants replace these placeholders. Importers should always
# go through this module (directly or via the graphs package facade) so that
# a single import-time try/except governs the runtime binding.
InMemorySaver: Any = None
StateGraph: Any = None
START: Any = "__start__"
END: Any = "__end__"

try:  # pragma: no cover - optional dependency
    from langgraph.checkpoint.memory import InMemorySaver as _InMemorySaver
    from langgraph.graph import END as _END
    from langgraph.graph import START as _START
    from langgraph.graph import StateGraph as _StateGraph

    InMemorySaver = _InMemorySaver
    StateGraph = _StateGraph
    START = _START
    END = _END
except ImportError:  # pragma: no cover - optional dependency
    pass
