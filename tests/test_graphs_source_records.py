"""Phase 7a: source_records submodule identity and behavioural contract tests.

Ensures the facade and the ``source_records`` submodule expose the same
callables for every public-ish helper in the extraction set, and pins the
minimum behavioural contract so future refactors cannot silently break the
seams. See ``docs/seam-maps/graphs.md`` for the extraction plan.
"""

from __future__ import annotations

from paper_chaser_mcp.agentic.graphs import _core as core_module
from paper_chaser_mcp.agentic.graphs import source_records
from paper_chaser_mcp.agentic.models import SearchStrategyMetadata, StructuredSourceRecord

_EXTRACTED = (
    "TopicalRelevanceClassification",
    "_answerability_from_source_records",
    "_candidate_leads_from_source_records",
    "_citation_record_from_paper",
    "_citation_record_from_regulatory_document",
    "_classify_topical_relevance",
    "_classify_topical_relevance_for_paper",
    "_classify_topical_relevance_with_provenance",
    "_coverage_summary_line",
    "_dedupe_structured_sources",
    "_evidence_from_source_records",
    "_graph_topic_tokens",
    "_lead_reason_for_source_record",
    "_likely_unverified_from_source_records",
    "_paper_text",
    "_records_with_lead_reasons",
    "_routing_summary_from_strategy",
    "_source_record_from_paper",
    "_source_record_from_regulatory_document",
    "_verified_findings_from_source_records",
    "_why_matched",
    "_year_text",
)


def test_core_and_submodule_expose_the_same_callables() -> None:
    for name in _EXTRACTED:
        submodule_value = getattr(source_records, name)
        core_value = getattr(core_module, name)
        assert submodule_value is core_value, (
            f"{name}: _core and submodule must share the same object so "
            "legacy monkeypatch and call sites keep working after Phase 7a"
        )


def test_year_text_accepts_varied_shapes() -> None:
    assert source_records._year_text(None) is None
    assert source_records._year_text("") is None
    assert source_records._year_text(2023) == "2023"
    assert source_records._year_text("Published 2019-04-01") == "2019"
    assert source_records._year_text("JournalTitle") == "Jour"


def test_paper_text_concatenates_common_fields() -> None:
    paper = {
        "title": "Alpha",
        "abstract": "Beta",
        "venue": "Gamma",
        "year": 2020,
        "authors": [{"name": "A. Author"}, "ignored-non-dict"],
    }
    text = source_records._paper_text(paper)
    for needle in ("Alpha", "Beta", "Gamma", "2020", "A. Author"):
        assert needle in text


def test_graph_topic_tokens_filters_generic_and_short_tokens() -> None:
    from paper_chaser_mcp.agentic.graphs.shared_state import _GRAPH_GENERIC_TERMS

    tokens = source_records._graph_topic_tokens("the abalone sea otter monitoring program")
    assert "abalone" in tokens
    assert "monitoring" in tokens
    for bad in _GRAPH_GENERIC_TERMS:
        if len(bad) >= 3:
            assert bad not in tokens


def test_classify_topical_relevance_boundary_conditions() -> None:
    on_topic = source_records._classify_topical_relevance(
        query_similarity=0.3,
        title_facet_coverage=1.0,
        title_anchor_coverage=0.5,
        query_facet_coverage=0.5,
        query_anchor_coverage=0.2,
    )
    assert on_topic == "on_topic"

    off_topic = source_records._classify_topical_relevance(
        query_similarity=0.05,
        title_facet_coverage=0.0,
        title_anchor_coverage=0.0,
        query_facet_coverage=0.0,
        query_anchor_coverage=0.0,
    )
    assert off_topic == "off_topic"

    weak = source_records._classify_topical_relevance(
        query_similarity=0.2,
        title_facet_coverage=0.0,
        title_anchor_coverage=0.0,
        query_facet_coverage=0.5,
        query_anchor_coverage=0.0,
    )
    assert weak == "weak_match"


def test_evidence_and_leads_partition_records() -> None:
    evidence_record = StructuredSourceRecord(
        title="Verified on-topic",
        verificationStatus="verified_primary_source",
        topicalRelevance="on_topic",
    )
    lead_record = StructuredSourceRecord(
        title="Unverified",
        verificationStatus="unverified",
        topicalRelevance="on_topic",
    )
    off_topic_record = StructuredSourceRecord(
        title="Off-topic",
        verificationStatus="verified_metadata",
        topicalRelevance="off_topic",
    )
    records = [evidence_record, lead_record, off_topic_record]

    evidence = source_records._evidence_from_source_records(records)
    leads = source_records._candidate_leads_from_source_records(records)

    assert [r.title for r in evidence] == ["Verified on-topic"]
    lead_titles = {r.title for r in leads}
    assert lead_titles == {"Unverified", "Off-topic"}
    for lead in leads:
        assert lead.lead_reason


def test_answerability_reflects_evidence_and_leads() -> None:
    evidence = [
        StructuredSourceRecord(
            title="x",
            verificationStatus="verified_metadata",
            topicalRelevance="on_topic",
        )
    ]
    assert (
        source_records._answerability_from_source_records(
            result_status="succeeded",
            evidence=evidence,
            leads=[],
            evidence_gaps=[],
        )
        == "grounded"
    )
    assert (
        source_records._answerability_from_source_records(
            result_status="partial",
            evidence=[],
            leads=evidence,
            evidence_gaps=[],
        )
        == "limited"
    )
    assert (
        source_records._answerability_from_source_records(
            result_status="partial",
            evidence=[],
            leads=[],
            evidence_gaps=[],
        )
        == "insufficient"
    )


def test_routing_summary_includes_core_routing_fields() -> None:
    strategy = SearchStrategyMetadata(
        intent="review",
        intentConfidence="medium",
        routingConfidence="high",
        anchorType="topic",
        anchoredSubject="x",
        providersUsed=["openalex"],
    )
    summary = source_records._routing_summary_from_strategy(
        strategy_metadata=strategy,
        coverage_summary=None,
        result_status="succeeded",
        evidence_gaps=[],
    )
    assert summary["intent"] == "review"
    assert summary["decisionConfidence"] == "high"
    assert summary["anchorType"] == "topic"
    assert summary["providerPlan"] == ["openalex"]
    assert summary["whyPartial"] is None
