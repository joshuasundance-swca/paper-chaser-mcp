"""Phase 6 red-bar: pin regulatory / literature intent helpers."""

from __future__ import annotations

from paper_chaser_mcp.agentic import planner as legacy_planner
from paper_chaser_mcp.agentic.planner import (
    _detect_cultural_resource_intent as _facade_detect_cultural_resource_intent,
)
from paper_chaser_mcp.agentic.planner import (
    _infer_entity_card as _facade_infer_entity_card,
)
from paper_chaser_mcp.agentic.planner import (
    _infer_regulatory_subintent as _facade_infer_regulatory_subintent,
)
from paper_chaser_mcp.agentic.planner import (
    _strong_known_item_signal as _facade_strong_known_item_signal,
)
from paper_chaser_mcp.agentic.planner import (
    _strong_regulatory_signal as _facade_strong_regulatory_signal,
)
from paper_chaser_mcp.agentic.planner.regulatory import (
    _detect_cultural_resource_intent,
    _infer_entity_card,
    _infer_regulatory_subintent,
    _strong_known_item_signal,
    _strong_regulatory_signal,
    detect_literature_intent,
    detect_regulatory_intent,
)


def test_detect_regulatory_intent_keyword_families() -> None:
    assert detect_regulatory_intent("What does the ESA say about listing status?")
    assert detect_regulatory_intent("50 CFR 17 critical habitat rulemaking history")
    assert detect_regulatory_intent("EPA guidance on safe drinking water act")
    assert detect_regulatory_intent("Section 106 tribal consultation NHPA")
    assert not detect_regulatory_intent("Attention Is All You Need")


def test_detect_literature_intent_peer_review_terms() -> None:
    assert detect_literature_intent("systematic review of climate adaptation")
    assert detect_literature_intent("peer-reviewed meta-analysis of statins")
    assert detect_literature_intent("journal article on transformers")
    assert not detect_literature_intent("50 CFR 17 critical habitat")


def test_cultural_resource_intent_positive_negative() -> None:
    assert _detect_cultural_resource_intent("Section 106 consultation for historic district")
    assert _detect_cultural_resource_intent("tribal consultation sacred site")
    assert not _detect_cultural_resource_intent("transformer self-attention mechanism")


def test_infer_regulatory_subintent_returns_expected_labels() -> None:
    assert _infer_regulatory_subintent("what does 40 cfr 141 current text say under sdwa") == "current_cfr_text"
    guidance = _infer_regulatory_subintent("FDA guidance for industry on clinical trials")
    assert guidance == "guidance_lookup"
    dossier = _infer_regulatory_subintent("recovery plan and critical habitat for red wolf")
    assert dossier == "species_dossier"
    assert _infer_regulatory_subintent("Attention Is All You Need") is None


def test_strong_signals() -> None:
    assert _strong_known_item_signal("10.1234/abcd.efgh")
    assert _strong_known_item_signal("https://example.com/paper")
    assert not _strong_known_item_signal("a broad literature review request")
    assert _strong_regulatory_signal("50 CFR 17")
    assert _strong_regulatory_signal("FDA guidance for industry")
    assert not _strong_regulatory_signal("review of transformer models")


def test_infer_entity_card_extracts_authority_family_subject() -> None:
    card = _infer_entity_card(
        "recovery plan for the red wolf under ESA",
    )
    assert card is not None
    assert card.get("authorityContext") == "ESA"
    assert card.get("requestedDocumentFamily") == "recovery_plan"
    # Subject extraction might populate commonName with species context.
    assert any(key in card for key in ("commonName", "scientificName"))


def test_legacy_planner_reexports_regulatory_helpers() -> None:
    # Public helpers stay module-handle readable via hasattr.
    for name in (
        "detect_regulatory_intent",
        "detect_literature_intent",
    ):
        assert hasattr(legacy_planner, name), f"legacy planner missing {name}"

    # Private helpers are pinned via direct-from-import identity checks so the
    # seam firewall (tests/test_test_seam_inventory.py) can see them.
    assert _facade_detect_cultural_resource_intent is _detect_cultural_resource_intent
    assert _facade_infer_regulatory_subintent is _infer_regulatory_subintent
    assert _facade_strong_regulatory_signal is _strong_regulatory_signal
    assert _facade_strong_known_item_signal is _strong_known_item_signal
    assert _facade_infer_entity_card is _infer_entity_card
