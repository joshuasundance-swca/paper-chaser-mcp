"""Planner-layer regex and term-set constants.

Phase 6 extracted these from the ``planner`` monolith so the regex patterns and
keyword/stopword sets live in a dependency-free module that every other
planner submodule can import without creating cycles. Only ``re`` and the
provider-layer ``COMMON_QUERY_WORDS`` allowlist are needed here.
"""

from __future__ import annotations

import re

from ..providers import COMMON_QUERY_WORDS

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
ARXIV_RE = re.compile(r"(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE)
FACET_SPLIT_RE = re.compile(
    r"\b(?:for|in|on|about|into|within|across|via|through|regarding|around)\b",
    re.IGNORECASE,
)
GENERIC_EVIDENCE_WORDS = COMMON_QUERY_WORDS | {
    "with",
    "were",
    "from",
    "into",
    "using",
    "use",
    "used",
    "their",
    "this",
    "that",
    "these",
    "those",
    "they",
    "them",
    "within",
    "across",
    "based",
    "approach",
    "approaches",
    "method",
    "methods",
    "analysis",
    "results",
    "finding",
    "findings",
    "quality",
    "different",
}
QUERY_FACET_TOKEN_ALLOWLIST = {
    "agent",
    "agents",
    "review",
    "reviews",
    "survey",
    "surveys",
    "tool",
    "tools",
}
HYPOTHESIS_QUERY_STOPWORDS = {
    "current",
    "different",
    "effective",
    "effectiveness",
    "especially",
    "evidence",
    "field",
    "latest",
    "methods",
    "most",
    "recent",
    "research",
    "review",
    "studies",
    "study",
}
LITERATURE_QUERY_TERMS = {
    "article",
    "citation",
    "doi",
    "evidence",
    "journal",
    "literature",
    "meta-analysis",
    "paper",
    "peer-reviewed",
    "review",
    "scholarly",
    "scientific",
    "study",
    "systematic review",
}
TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
QUERYISH_TITLE_BLOCKERS = {
    "aquatic",
    "bioaccumulation",
    "biodiversity",
    "climate",
    "compare",
    "contamination",
    "ecotoxicology",
    "ecosystem",
    "effects",
    "evidence",
    "exposure",
    "history",
    "include",
    "listing",
    "marine",
    "mitigation",
    "monitoring",
    "pollution",
    "regulatory",
    "review",
    "status",
    "studies",
    "study",
    "survey",
    "systematic",
    "terrestrial",
    "toxicity",
    "transport",
    "trophic",
    "what",
}
STRONG_REGULATORY_TITLE_BLOCKERS = {
    "critical habitat",
    "ecos",
    "esa",
    "federal register",
    "final rule",
    "listing status",
    "regulatory history",
    "rulemaking",
}
AGENCY_REGULATORY_MARKERS = {
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
REGULATORY_QUERY_TERMS = {
    "106 consultation",
    "ac hp",
    "agency guidance",
    "archaeology",
    "archaeology guidance",
    "biological opinion",
    "cfr",
    "clinical decision support",
    "code of federal regulations",
    "contaminant limit",
    "critical habitat",
    "drinking water standard",
    "ecos",
    "esa",
    "fda",
    "final rule",
    "food and drug administration",
    "federal register",
    "five-year review",
    "five year review",
    "guidance for industry",
    "health advisory",
    "historic district",
    "historic preservation",
    "incidental take",
    "listing status",
    "listing history",
    "maximum contaminant level",
    "mcl",
    "nhpa",
    "proposed rule",
    "recovery plan",
    "regulation",
    "regulatory history",
    "rulemaking",
    "safe drinking water act",
    "sdwa",
    "section 106",
    "section 7",
    "species dossier",
    "tribal consultation",
    "thpo",
    "shpo",
    "sacred site",
    "cultural resources",
    "cultural landscape",
}

_CULTURAL_RESOURCE_MARKERS = {
    "archaeological",
    "archaeology",
    "cultural resources",
    "cultural landscape",
    "historic district",
    "historic preservation",
    "historic property",
    "nhpa",
    "sacred site",
    "section 106",
    "tribal consultation",
    "thpo",
    "shpo",
}

VARIANT_DEDUPE_STOPWORDS = (
    TITLE_STOPWORDS
    | GENERIC_EVIDENCE_WORDS
    | {
        "florida",
        "review",
        "scrub",
    }
)

_DEFINITIONAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhat\s+(is|are|does)\b"),
    re.compile(r"\bdefine\b"),
    re.compile(r"\bexplain\b"),
    re.compile(r"\boverview\s+of\b"),
    re.compile(r"\bintroduction\s+to\b"),
    re.compile(r"\bguide\s+to\b"),
    re.compile(r"\bprimer\s+on\b"),
)

__all__ = [
    "AGENCY_REGULATORY_MARKERS",
    "ARXIV_RE",
    "DOI_RE",
    "FACET_SPLIT_RE",
    "GENERIC_EVIDENCE_WORDS",
    "HYPOTHESIS_QUERY_STOPWORDS",
    "LITERATURE_QUERY_TERMS",
    "QUERY_FACET_TOKEN_ALLOWLIST",
    "QUERYISH_TITLE_BLOCKERS",
    "REGULATORY_QUERY_TERMS",
    "STRONG_REGULATORY_TITLE_BLOCKERS",
    "TITLE_STOPWORDS",
    "VARIANT_DEDUPE_STOPWORDS",
    "_CULTURAL_RESOURCE_MARKERS",
    "_DEFINITIONAL_PATTERNS",
]
