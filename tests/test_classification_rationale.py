"""Tests for the user-facing ``classificationRationale`` surface.

Workstream C of the environmental-science remediation plan adds a short,
human-readable rationale for topical relevance classifications. This suite
covers the three production paths:

1. Deterministic three-way fallback composes a rationale parametrically from
   the signal profile (no domain hardcoding, no leakage of debug strings).
2. LLM-authored rationales are preserved verbatim (or truncated) on the
   ``classificationRationale`` field by :func:`annotate_llm_entry`.
3. ``_guided_trust_summary`` surfaces a top-level ``trustRationale`` string and
   a per-bucket ``classificationRationaleByBucket`` breakdown built from the
   structured source records.
"""

from __future__ import annotations

from paper_chaser_mcp.agentic.relevance_fallback import (
    _compose_classification_rationale,
    _signal_profile,
    annotate_llm_entry,
    classify_paper_deterministic,
)
from paper_chaser_mcp.dispatch import _guided_trust_summary


def test_classify_paper_deterministic_emits_short_rationale() -> None:
    query = "machine learning for protein folding"
    paper = {
        "title": "Completely unrelated topic about medieval history",
        "abstract": "This paper discusses 14th-century European trade routes.",
    }

    entry = classify_paper_deterministic(query=query, paper=paper)

    rationale = entry.get("classificationRationale")
    assert isinstance(rationale, str) and rationale
    # Must be a concise user-facing string, not the debug relevanceReason.
    assert len(rationale) <= 150
    assert "Deterministic three-way tier" not in rationale
    assert "Signals:" not in rationale
    # Off-topic label should be chosen for this clearly unrelated paper.
    assert entry["classification"] == "off_topic"


def test_compose_classification_rationale_is_parametric() -> None:
    on_topic_profile = {
        "similarity": 0.62,
        "title_anchor_coverage": 0.8,
        "body_anchor_coverage": 0.9,
        "title_facet_coverage": 1.0,
        "body_facet_coverage": 1.0,
    }
    off_topic_profile = {
        "similarity": 0.01,
        "title_anchor_coverage": 0.0,
        "body_anchor_coverage": 0.0,
        "title_facet_coverage": 0.0,
        "body_facet_coverage": 0.0,
    }
    weak_profile = {
        "similarity": 0.18,
        "title_anchor_coverage": 0.0,
        "body_anchor_coverage": 0.4,
        "title_facet_coverage": 0.0,
        "body_facet_coverage": 0.5,
    }

    on_text = _compose_classification_rationale("on_topic", on_topic_profile)
    off_text = _compose_classification_rationale("off_topic", off_topic_profile)
    weak_text = _compose_classification_rationale("weak_match", weak_profile)

    # All outputs are bounded, nonempty, and distinct per label family.
    for text in (on_text, off_text, weak_text):
        assert text and len(text) <= 150

    assert "Strong topical overlap" in on_text
    assert "No query-term overlap" in off_text
    # Weak-match should mention the signal that triggered it (body anchors)
    # without hardcoding any domain phrase.
    assert "abstract" in weak_text.lower() or "body" in weak_text.lower()


def test_annotate_llm_entry_preserves_and_truncates_rationale() -> None:
    short_entry: dict[str, object] = {
        "classification": "weak_match",
        "rationale": "Related to the query but focuses on a tangential sub-problem.",
    }
    annotate_llm_entry(short_entry)
    assert short_entry["classificationRationale"] == short_entry["rationale"]

    long_rationale = "x" * 500
    long_entry: dict[str, object] = {
        "classification": "off_topic",
        "rationale": long_rationale,
    }
    annotate_llm_entry(long_entry)
    truncated = long_entry["classificationRationale"]
    assert isinstance(truncated, str)
    assert len(truncated) <= 150

    # When the LLM omits a rationale, no classificationRationale is set.
    empty_entry: dict[str, object] = {"classification": "on_topic", "rationale": ""}
    annotate_llm_entry(empty_entry)
    assert "classificationRationale" not in empty_entry


def test_signal_profile_feeds_deterministic_rationale() -> None:
    # End-to-end check that _signal_profile outputs are usable inputs.
    profile = _signal_profile(
        "quantum entanglement experiments",
        {"title": "Bell inequality violations", "abstract": "Quantum entanglement test."},
    )
    rationale = _compose_classification_rationale("on_topic", profile)
    assert rationale and len(rationale) <= 150


def test_guided_trust_summary_surfaces_trust_rationale() -> None:
    sources = [
        {
            "topicalRelevance": "weak_match",
            "verificationStatus": "verified_metadata",
            "classificationRationale": "Partial facet coverage; some query aspects missing in title.",
            "note": "Loosely related.",
        },
        {
            "topicalRelevance": "off_topic",
            "verificationStatus": "unverified",
            "classificationRationale": "No query-term overlap in title or abstract.",
            "note": "Different topic.",
        },
    ]

    summary = _guided_trust_summary(sources, evidence_gaps=["missing on-topic evidence"])

    assert "trustRationale" in summary
    trust_rationale = summary["trustRationale"]
    assert isinstance(trust_rationale, str) and trust_rationale
    # Off-topic rationale takes precedence when available.
    assert "No query-term overlap" in trust_rationale
    assert len(trust_rationale) <= 280

    bucket = summary["classificationRationaleByBucket"]
    assert "No query-term overlap in title or abstract." in bucket["offTopic"]
    assert "Partial facet coverage; some query aspects missing in title." in bucket["weakMatch"]


def test_guided_trust_summary_backward_compatible_without_rationales() -> None:
    # Existing shape must not regress: no classificationRationale entries
    # should still yield a trustRationale (fallback to strengthExplanation).
    sources = [
        {
            "topicalRelevance": "on_topic",
            "verificationStatus": "verified_primary_source",
        },
    ]
    summary = _guided_trust_summary(sources, evidence_gaps=[])
    assert summary["trustRationale"] == summary["strengthExplanation"]
    assert summary["classificationRationaleByBucket"] == {"weakMatch": [], "offTopic": []}
