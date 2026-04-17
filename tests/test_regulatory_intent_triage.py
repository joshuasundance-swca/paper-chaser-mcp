"""LLM-first regulatory triage: ``_derive_regulatory_query_flags``.

The three regulatory sub-routes (current CFR text retrieval, rulemaking
history, agency guidance) were previously selected exclusively via brittle
keyword/regex helpers in ``graphs.py``. The planner already emits a rich
``regulatoryIntent`` enum (``current_cfr_text`` / ``rulemaking_history`` /
``guidance_lookup`` / ``species_dossier`` / ``hybrid_regulatory_plus_literature``
/ ``unspecified``), so we consume that LLM signal authoritatively and only fall
back to the deterministic helpers when the label is ``unspecified``/missing or
``hybrid_regulatory_plus_literature``.

Audit regressions these tests lock in:

* ``"What does CFR 40 Part 403.5 actually prohibit compared to the prior 2010
  version?"`` previously tripped BOTH ``current_text`` AND ``history``; when
  the LLM tags it ``current_cfr_text``, only that route fires.
* ``"Tell me about the listing history of the Pallid Sturgeon"`` previously
  fired the rulemaking-history helper even though it is species-regulatory;
  when the LLM tags it ``species_dossier``, none of the three sub-routes fire
  and the query flows through the species/ECOS path.
* When the planner bundle is a shim / returns ``unspecified``, the
  deterministic helpers still run unchanged.
"""

from __future__ import annotations

from paper_chaser_mcp.agentic.graphs import (
    _derive_regulatory_query_flags,
    _ecos_query_variants,
    _is_agency_guidance_query,
    _is_current_cfr_text_request,
    _query_requests_regulatory_history,
)
from paper_chaser_mcp.agentic.models import PlannerDecision
from paper_chaser_mcp.agentic.planner import _has_literature_corroboration


def _planner(
    regulatory_intent: str | None,
    *,
    llm: bool = True,
    **extra: object,
) -> PlannerDecision:
    """Build a ``PlannerDecision`` for triage tests.

    ``llm=True`` (default) stamps a ``subjectCard`` with
    ``source="planner_llm"`` so the LLM-authoritative branch of
    ``_derive_regulatory_query_flags`` fires. ``llm=False`` leaves the
    subject card absent (deterministic bundle) to exercise the keyword
    fallback.
    """

    payload: dict[str, object] = {
        "intent": "regulatory",
        "regulatoryIntent": regulatory_intent,
    }
    if llm:
        if "subjectCard" not in extra:
            payload["subjectCard"] = {
                "confidence": "high",
                "source": "planner_llm",
            }
        payload.setdefault("plannerSource", "llm")
    payload.update(extra)
    return PlannerDecision.model_validate(payload)


class TestDeriveRegulatoryQueryFlags:
    def test_planner_current_cfr_text_wins_over_history_keywords(self) -> None:
        """Comparison-to-prior-version phrasing that trips BOTH deterministic
        helpers must not double-route when the LLM has classified the query as
        ``current_cfr_text``."""

        query = (
            "What does 40 CFR 403.5 actually prohibit and what is the "
            "final rule history?"
        )
        # Baseline: deterministic helpers misfire on BOTH routes.
        assert _is_current_cfr_text_request(query) is True
        assert _query_requests_regulatory_history(query) is True

        flags = _derive_regulatory_query_flags(
            query=query, planner=_planner("current_cfr_text")
        )
        assert flags == (True, False, False)

    def test_planner_species_dossier_suppresses_history_route(self) -> None:
        """"Listing history of the Pallid Sturgeon" must not activate the
        federal-register rulemaking-history route just because the word
        "history" appears in the query."""

        query = "Tell me about the listing history of the Pallid Sturgeon"
        assert _query_requests_regulatory_history(query) is True  # baseline misfire

        flags = _derive_regulatory_query_flags(
            query=query, planner=_planner("species_dossier")
        )
        assert flags == (False, False, False)

    def test_planner_guidance_lookup_wins(self) -> None:
        flags = _derive_regulatory_query_flags(
            query="EPA guidance on PFAS drinking-water standards",
            planner=_planner("guidance_lookup"),
        )
        assert flags == (False, False, True)

    def test_planner_rulemaking_history_wins(self) -> None:
        flags = _derive_regulatory_query_flags(
            query="Pallid Sturgeon federal register rulemaking history",
            planner=_planner("rulemaking_history"),
        )
        assert flags == (False, True, False)

    def test_hybrid_label_defers_to_deterministic_helpers(self) -> None:
        """``hybrid_regulatory_plus_literature`` is a mixed-intent signal; the
        downstream routing still wants each keyword-matched sub-route to fire."""

        query = "Current CFR 40 Part 403.5 text and peer-reviewed studies"
        expected = (
            _is_current_cfr_text_request(query),
            _query_requests_regulatory_history(query),
            _is_agency_guidance_query(query),
        )
        flags = _derive_regulatory_query_flags(
            query=query, planner=_planner("hybrid_regulatory_plus_literature")
        )
        assert flags == expected

    def test_unspecified_falls_back_to_deterministic(self) -> None:
        """``unspecified`` / ``None`` preserves the historical keyword-heuristic
        contract even when the bundle is LLM-capable."""

        query = "Federal register rulemaking history for the Pallid Sturgeon"
        expected = (
            _is_current_cfr_text_request(query),
            _query_requests_regulatory_history(query),
            _is_agency_guidance_query(query),
        )
        assert (
            _derive_regulatory_query_flags(query=query, planner=_planner("unspecified"))
            == expected
        )
        assert (
            _derive_regulatory_query_flags(query=query, planner=_planner(None))
            == expected
        )
        assert _derive_regulatory_query_flags(query=query, planner=None) == expected

    def test_deterministic_bundle_prefers_keyword_helpers_over_label(self) -> None:
        """When the planner ran without an LLM (``subject_card`` absent or
        stamped ``deterministic_fallback``), ``regulatoryIntent`` itself comes
        from deterministic heuristics that can miss secondary sub-routes. The
        keyword helpers run unchanged so queries like "Regulatory history ...
        under 50 CFR 17.95" still activate BOTH current-text and history
        despite the deterministic label picking only ``current_cfr_text``."""

        query = "Regulatory history of California condor under 50 CFR 17.95"
        expected = (
            _is_current_cfr_text_request(query),
            _query_requests_regulatory_history(query),
            _is_agency_guidance_query(query),
        )
        assert expected == (True, True, False)

        # No subject_card → deterministic bundle.
        flags_no_card = _derive_regulatory_query_flags(
            query=query,
            planner=_planner("current_cfr_text", llm=False),
        )
        assert flags_no_card == expected

        # Explicit deterministic_fallback source.
        flags_deterministic = _derive_regulatory_query_flags(
            query=query,
            planner=_planner(
                "current_cfr_text",
                llm=False,
                subjectCard={
                    "confidence": "deterministic_fallback",
                    "source": "deterministic_fallback",
                },
            ),
        )
        assert flags_deterministic == expected


class TestEcosQueryVariantsPlannerFirst:
    def test_planner_entity_card_supplies_scientific_name_over_regex(self) -> None:
        """Genus-only / non-binomial mentions either escape the regex or cause
        it to latch onto adjacent prose (e.g. ``"Scaphirhynchus critical"``);
        the planner entity card surfaces the true scientific name as an
        additional variant ahead of the regex-derived ones."""

        query = "Scaphirhynchus critical habitat designation"
        planner = _planner(
            "species_dossier",
            entityCard={
                "scientificName": "Scaphirhynchus albus",
                "commonName": "Pallid Sturgeon",
            },
        )
        variants = _ecos_query_variants(query, planner=planner)
        scientific_values = [value for value, anchor in variants if anchor == "species_scientific_name"]
        common_values = [value for value, anchor in variants if anchor == "species_common_name"]

        # Planner-supplied names are present (preceding any regex candidates).
        assert "Scaphirhynchus albus" in scientific_values
        assert "Pallid Sturgeon" in common_values
        # Planner candidates are tried first so ECOS sees them before the
        # noisy regex-extracted variants.
        assert scientific_values.index("Scaphirhynchus albus") == 0
        assert common_values.index("Pallid Sturgeon") == 0

    def test_ecos_variants_without_planner_preserve_regex_behavior(self) -> None:
        """Deterministic callers (no planner) keep the pre-existing variant
        contract: regex-extracted scientific/common name plus the raw query."""

        variants_with = _ecos_query_variants("northern long-eared bat ECOS species profile")
        variants_without = _ecos_query_variants(
            "northern long-eared bat ECOS species profile", planner=None
        )
        assert variants_with == variants_without
        # The regex noise-removal still surfaces "northern long-eared bat".
        common_values = [value for value, anchor in variants_with if anchor == "species_common_name"]
        assert any("northern long-eared bat" in value.lower() for value in common_values)


class TestPlannerLlmRegulatoryIntentAuthority:
    """Finding 1 regression: LLM planner's `regulatoryIntent` must be
    authoritative whenever `planner_source == "llm"`, even if the LLM
    omitted subject-card grounding fields (which causes `classify_query`
    to stamp the subject card as `deterministic_fallback`)."""

    def test_llm_regulatory_intent_honored_without_subject_card_grounding(self) -> None:
        query = "Tell me about the listing history of the Pallid Sturgeon"
        # Baseline: deterministic helper misfires.
        assert _query_requests_regulatory_history(query) is True

        # LLM planner emitted regulatoryIntent but NO grounding signals, so
        # subject-card stamping falls back to `deterministic_fallback`.
        planner = PlannerDecision.model_validate(
            {
                "intent": "regulatory",
                "regulatoryIntent": "species_dossier",
                "plannerSource": "llm",
                "subjectCard": {
                    "confidence": "deterministic_fallback",
                    "source": "deterministic_fallback",
                },
            }
        )
        flags = _derive_regulatory_query_flags(query=query, planner=planner)
        # `species_dossier` must suppress the rulemaking-history route even
        # though subject-card provenance is deterministic_fallback.
        assert flags == (False, False, False)

    def test_llm_regulatory_intent_honored_with_no_subject_card_at_all(self) -> None:
        query = "Current text of 40 CFR 403.5"
        planner = PlannerDecision.model_validate(
            {
                "intent": "regulatory",
                "regulatoryIntent": "current_cfr_text",
                "plannerSource": "llm",
            }
        )
        flags = _derive_regulatory_query_flags(query=query, planner=planner)
        assert flags == (True, False, False)


class TestPlannerLiteratureCorroborationHybridHypothesis:
    """Finding 2 regression: planner-time `_has_literature_corroboration`
    must accept `hybrid_policy_science` (and literature-shaped markers) in
    `retrievalHypotheses` as corroboration, matching the dispatch-time
    gate in `dispatch._guided_should_add_review_pass`. Otherwise valid
    hybrid labels are stripped at planner-time before they reach dispatch."""

    def test_hybrid_policy_science_hypothesis_is_corroboration(self) -> None:
        query = "What does EPA require for PFAS discharges?"
        planner = PlannerDecision.model_validate(
            {
                "intent": "regulatory",
                "regulatoryIntent": "hybrid_regulatory_plus_literature",
                "plannerSource": "llm",
                "retrievalHypotheses": [
                    "hybrid_policy_science: fuse regulatory primary sources with peer-reviewed studies",
                ],
            }
        )
        assert _has_literature_corroboration(planner=planner, query=query, focus=None) is True

    def test_literature_marker_in_hypothesis_is_corroboration(self) -> None:
        query = "EPA stormwater discharge standards"
        planner = PlannerDecision.model_validate(
            {
                "intent": "regulatory",
                "plannerSource": "llm",
                "retrievalHypotheses": [
                    "Peer-reviewed literature on discharge efficacy",
                ],
            }
        )
        assert _has_literature_corroboration(planner=planner, query=query, focus=None) is True

    def test_no_corroboration_without_any_literature_signal(self) -> None:
        query = "What does EPA require for stormwater discharges?"
        planner = PlannerDecision.model_validate(
            {
                "intent": "regulatory",
                "plannerSource": "llm",
                "retrievalHypotheses": ["regulatory primary sources only"],
            }
        )
        assert _has_literature_corroboration(planner=planner, query=query, focus=None) is False
