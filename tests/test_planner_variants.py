"""Phase 7c-1: tests for the extracted ``planner.variants`` submodule."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.agentic.config import AgenticConfig
from paper_chaser_mcp.agentic.models import ExpansionCandidate
from paper_chaser_mcp.agentic.planner import variants as variants_module
from paper_chaser_mcp.agentic.planner._core import (
    _signatures_are_near_duplicates,
    _top_evidence_phrases,
    _variant_signature,
    combine_variants,
    dedupe_variants,
)


def _make_config(
    *,
    max_grounded_variants: int = 3,
    max_speculative_variants: int = 3,
    max_total_variants: int = 6,
) -> AgenticConfig:
    return AgenticConfig(
        enabled=True,
        provider="deterministic",
        planner_model="deterministic",
        synthesis_model="deterministic",
        embedding_model="deterministic",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
        max_grounded_variants=max_grounded_variants,
        max_speculative_variants=max_speculative_variants,
        max_total_variants=max_total_variants,
    )


def _make_candidate(text: str) -> ExpansionCandidate:
    return ExpansionCandidate(
        variant=text,
        source="speculative",
        rationale="test",
    )


def test_variant_signature_drops_stopwords_and_short_tokens() -> None:
    signature = _variant_signature("The effects of climate change on reef fish")
    assert "climate" in signature
    assert "change" in signature
    assert "the" not in signature
    assert "on" not in signature


def test_signatures_are_near_duplicates_detects_high_coverage() -> None:
    left = _variant_signature("coral reef bleaching thermal stress")
    right = _variant_signature("coral reef bleaching under thermal stress")
    assert _signatures_are_near_duplicates(left, right) is True


def test_signatures_are_near_duplicates_rejects_disjoint_queries() -> None:
    left = _variant_signature("neural machine translation architecture")
    right = _variant_signature("protein folding structural biology")
    assert _signatures_are_near_duplicates(left, right) is False


def test_combine_variants_deduplicates_and_caps() -> None:
    config = _make_config(
        max_grounded_variants=3,
        max_speculative_variants=3,
        max_total_variants=4,
    )
    grounded = [
        _make_candidate("coral reef bleaching"),
        _make_candidate("coral reef bleaching"),
    ]
    speculative = [
        _make_candidate("ocean acidification impacts"),
        _make_candidate("coral thermal stress response"),
    ]
    combined = combine_variants(
        original_query="reef fish ecology",
        grounded=grounded,
        speculative=speculative,
        config=config,
    )
    assert len(combined) <= config.max_total_variants
    assert combined[0].source == "from_input"
    variants = [c.variant for c in combined]
    assert "reef fish ecology" in variants


def test_dedupe_variants_preserves_order_and_caps() -> None:
    config = _make_config(max_total_variants=3)
    candidates = [
        _make_candidate("alpha beta gamma"),
        _make_candidate("alpha beta gamma"),
        _make_candidate("delta epsilon zeta"),
        _make_candidate("eta theta iota"),
        _make_candidate("kappa lambda mu"),
    ]
    result = dedupe_variants(candidates, config=config)
    assert len(result) == 3
    assert [c.variant for c in result][:2] == ["alpha beta gamma", "delta epsilon zeta"]


def test_top_evidence_phrases_returns_recurring_bigrams() -> None:
    papers = [
        {"title": "Coral reef bleaching under thermal stress"},
        {"title": "Thermal stress effects on coral reef ecosystems"},
        {"title": "Bleaching events in tropical coral reef systems"},
        {"title": "Coral reef recovery after thermal bleaching"},
    ]
    phrases = _top_evidence_phrases(papers, limit=3)
    assert any("coral reef" in phrase for phrase in phrases)


def test_variants_submodule_exposes_expected_symbols() -> None:
    expected = {
        "combine_variants",
        "dedupe_variants",
        "_variant_signature",
        "_signatures_are_near_duplicates",
        "_top_evidence_phrases",
    }
    missing = expected - set(dir(variants_module))
    assert not missing, f"variants submodule missing: {missing}"


def test_core_symbols_identity_matches_variants_submodule() -> None:
    assert combine_variants is variants_module.combine_variants
    assert dedupe_variants is variants_module.dedupe_variants
    assert _variant_signature is variants_module._variant_signature
    assert _signatures_are_near_duplicates is variants_module._signatures_are_near_duplicates
    assert _top_evidence_phrases is variants_module._top_evidence_phrases


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
