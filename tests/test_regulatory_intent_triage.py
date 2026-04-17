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

from typing import Any

from paper_chaser_mcp.agentic.graphs import (
    _derive_regulatory_query_flags,
    _ecos_query_variants,
    _is_agency_guidance_query,
    _is_current_cfr_text_request,
    _is_opaque_query,
    _query_requests_regulatory_history,
    _rank_ecos_variant_hits,
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
        if regulatory_intent is not None:
            payload.setdefault("regulatoryIntentSource", "llm")
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
    def test_planner_entity_card_supplies_scientific_name_as_fallback(self) -> None:
        """Genus-only / non-binomial mentions either escape the regex or cause
        it to latch onto adjacent prose (e.g. ``"Scaphirhynchus critical"``);
        the planner entity card surfaces the true scientific name as a
        fallback variant AFTER the raw/regex-derived ones so a hallucinated
        LLM species name cannot lock ECOS onto real-but-wrong data."""

        query = "Scaphirhynchus critical habitat designation"
        planner = _planner(
            "species_dossier",
            entityCard={
                "scientificName": "Scaphirhynchus albus",
                "commonName": "Pallid Sturgeon",
            },
        )
        variants = _ecos_query_variants(query, planner=planner)
        scientific = [(value, origin) for value, anchor, origin in variants if anchor == "species_scientific_name"]
        common = [(value, origin) for value, anchor, origin in variants if anchor == "species_common_name"]

        # Planner-supplied names are still present, but tagged "planner".
        assert ("Scaphirhynchus albus", "planner") in scientific
        assert ("Pallid Sturgeon", "planner") in common

        # Raw variants must appear before planner variants in the overall list
        # so ECOS tries query-grounded candidates first.
        origins = [origin for _value, _anchor, origin in variants]
        if "planner" in origins and "raw" in origins:
            assert origins.index("raw") < origins.index("planner")

    def test_ecos_variants_without_planner_preserve_regex_behavior(self) -> None:
        """Deterministic callers (no planner) keep the pre-existing variant
        contract: regex-extracted scientific/common name plus the raw query,
        all tagged ``"raw"``."""

        variants_with = _ecos_query_variants("northern long-eared bat ECOS species profile")
        variants_without = _ecos_query_variants(
            "northern long-eared bat ECOS species profile", planner=None
        )
        assert variants_with == variants_without
        # All variants come from raw/regex when no planner is supplied.
        assert all(origin == "raw" for _value, _anchor, origin in variants_with)
        common_values = [value for value, anchor, _origin in variants_with if anchor == "species_common_name"]
        assert any("northern long-eared bat" in value.lower() for value in common_values)

    def test_raw_variants_precede_planner_fallbacks(self) -> None:
        """Finding 3 regression: when the planner emits a species name not
        supported by the raw query, the raw/regex variants must be tried
        FIRST so ECOS corroborates the planner name via the query itself
        where possible. This removes the hallucination-first risk where a
        plausible-but-wrong LLM species name returns real-but-wrong ECOS
        data and contaminates downstream ranking."""

        query = "Myotis lucifugus hibernation sites in Vermont"
        planner = _planner(
            "species_dossier",
            entityCard={
                "scientificName": "Myotis septentrionalis",
                "commonName": "Northern long-eared bat",
            },
        )
        variants = _ecos_query_variants(query, planner=planner)
        raw_sci_idx = next(
            (i for i, (_v, a, o) in enumerate(variants) if a == "species_scientific_name" and o == "raw"),
            None,
        )
        planner_sci_idx = next(
            (i for i, (_v, a, o) in enumerate(variants) if a == "species_scientific_name" and o == "planner"),
            None,
        )
        assert raw_sci_idx is not None, "regex should have extracted Myotis lucifugus"
        assert planner_sci_idx is not None
        assert raw_sci_idx < planner_sci_idx


class TestOpaqueQueryEcosVariantOrdering:
    """Finding 3 (4th rubber-duck pass): when the query is opaque (a DOI, a
    bare URL, an arXiv id, or an identifier-shaped token), the raw full-query
    variant tagged ``regulatory_subject_terms`` is meaningless to ECOS and can
    incidentally match an unrelated species. The break-on-first ECOS loop
    would then lock onto that noise before trying the planner-supplied names.
    For opaque queries the raw full-query variant must be deferred until
    AFTER the planner fallbacks."""

    def test_opaque_doi_defers_raw_subject_terms_after_planner(self) -> None:
        query = "10.1234/ecosystem.2020.123"
        planner = _planner(
            "species_dossier",
            entityCard={
                "scientificName": "Scaphirhynchus albus",
                "commonName": "Pallid Sturgeon",
            },
        )
        variants = _ecos_query_variants(query, planner=planner)

        subject_terms_idx = next(
            (i for i, (_v, a, _o) in enumerate(variants) if a == "regulatory_subject_terms"),
            None,
        )
        planner_sci_idx = next(
            (i for i, (_v, a, o) in enumerate(variants) if a == "species_scientific_name" and o == "planner"),
            None,
        )
        assert subject_terms_idx is not None
        assert planner_sci_idx is not None
        assert planner_sci_idx < subject_terms_idx, (
            "planner species variant must precede opaque raw subject-terms variant"
        )

    def test_non_opaque_query_preserves_raw_subject_terms_first(self) -> None:
        query = "Pallid Sturgeon critical habitat designation"
        planner = _planner(
            "species_dossier",
            entityCard={"scientificName": "Scaphirhynchus albus"},
        )
        variants = _ecos_query_variants(query, planner=planner)
        origins_by_anchor = [(a, o) for _v, a, o in variants]
        # Raw full-query (regulatory_subject_terms, raw) appears before any
        # planner variant.
        raw_subj_idx = next(
            (i for i, (a, o) in enumerate(origins_by_anchor) if a == "regulatory_subject_terms" and o == "raw"),
            None,
        )
        planner_idx = next(
            (i for i, (a, o) in enumerate(origins_by_anchor) if o == "planner"),
            None,
        )
        assert raw_subj_idx is not None
        assert planner_idx is not None
        assert raw_subj_idx < planner_idx


class TestIsOpaqueQuery:
    def test_doi_is_opaque(self) -> None:
        assert _is_opaque_query("10.1038/s41586-022-04567-8") is True

    def test_arxiv_id_is_opaque(self) -> None:
        assert _is_opaque_query("arXiv:2301.12345") is True

    def test_url_is_opaque(self) -> None:
        assert _is_opaque_query("https://example.com/paper.pdf") is True

    def test_prose_query_is_not_opaque(self) -> None:
        assert _is_opaque_query("Pallid Sturgeon critical habitat designation") is False
        assert _is_opaque_query("40 CFR 403.5 current text") is False


class TestEcosVariantRankingProvenance:
    """Finding 4 (4th rubber-duck pass): ``_ecosProvenance`` is stamped on
    the chosen ECOS species hit, but the old break-on-first loop never
    consumed it — whichever variant returned first won regardless of
    corroboration. Now ``_rank_ecos_variant_hits`` weights each variant's
    hit count by a provenance factor (raw/corroborated = 1.0,
    planner-only = 0.9) so a planner-only result cannot beat a
    roughly-equal raw match by accident."""

    def _variant(
        self, idx: int, anchor: str, origin: str, hits: int
    ) -> tuple[int, str, str, dict[str, Any]]:
        return (
            idx,
            anchor,
            origin,
            {"data": [{"speciesId": f"sp-{idx}-{i}"} for i in range(hits)]},
        )

    def test_corroborated_variant_wins_tie_against_planner(self) -> None:
        raw = self._variant(0, "species_scientific_name", "raw", 1)
        planner = self._variant(1, "species_scientific_name", "planner", 1)
        picked = _rank_ecos_variant_hits([raw, planner])
        assert picked is not None and picked[2] == "raw"

    def test_corroborated_variant_wins_when_planner_has_one_more_hit(self) -> None:
        # raw: 2 * 1.0 = 2.0 vs planner: 2 * 0.9 = 1.8 → raw wins.
        raw = self._variant(0, "species_scientific_name", "raw", 2)
        planner = self._variant(1, "species_scientific_name", "planner", 2)
        picked = _rank_ecos_variant_hits([planner, raw])
        assert picked is not None and picked[2] == "raw"

    def test_planner_wins_when_score_materially_exceeds_raw(self) -> None:
        # raw: 1 * 1.0 = 1.0 vs planner: 3 * 0.9 = 2.7 → planner wins.
        raw = self._variant(0, "species_scientific_name", "raw", 1)
        planner = self._variant(1, "species_scientific_name", "planner", 3)
        picked = _rank_ecos_variant_hits([raw, planner])
        assert picked is not None and picked[2] == "planner"

    def test_empty_input_returns_none(self) -> None:
        assert _rank_ecos_variant_hits([]) is None

    def test_earlier_variant_index_breaks_remaining_ties(self) -> None:
        raw_a = self._variant(0, "species_scientific_name", "raw", 1)
        raw_b = self._variant(2, "species_common_name", "raw", 1)
        picked = _rank_ecos_variant_hits([raw_b, raw_a])
        assert picked is not None and picked[0] == 0


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
                "regulatoryIntentSource": "llm",
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
                "regulatoryIntentSource": "llm",
                "plannerSource": "llm",
            }
        )
        flags = _derive_regulatory_query_flags(query=query, planner=planner)
        assert flags == (True, False, False)

    def test_deterministic_fallback_regulatory_intent_does_not_claim_llm_authority(self) -> None:
        """Regression (Finding 2, 4th pass): when the LLM omits or malforms
        ``regulatoryIntent`` it is coerced to ``None`` and then derived by the
        deterministic fallback. That derived value must NOT masquerade as
        LLM-authoritative and suppress the keyword helpers. The gate keys off
        ``regulatory_intent_source`` rather than the (``planner_source==llm`` +
        non-None regulatory_intent) combination."""

        query = "Regulatory history of California condor under 50 CFR 17.95"
        expected = (
            _is_current_cfr_text_request(query),
            _query_requests_regulatory_history(query),
            _is_agency_guidance_query(query),
        )
        # Deterministic-fallback label on an LLM planner: must defer to
        # keyword helpers even though ``planner_source`` is ``llm``.
        planner = PlannerDecision.model_validate(
            {
                "intent": "regulatory",
                "regulatoryIntent": "species_dossier",
                "regulatoryIntentSource": "deterministic_fallback",
                "plannerSource": "llm",
                "subjectCard": {
                    "confidence": "high",
                    "source": "planner_llm",
                },
            }
        )
        flags = _derive_regulatory_query_flags(query=query, planner=planner)
        assert flags == expected
        # Unspecified source also defers to keyword helpers.
        planner_unspecified = PlannerDecision.model_validate(
            {
                "intent": "regulatory",
                "regulatoryIntent": "species_dossier",
                "regulatoryIntentSource": "unspecified",
                "plannerSource": "llm",
                "subjectCard": {
                    "confidence": "high",
                    "source": "planner_llm",
                },
            }
        )
        assert (
            _derive_regulatory_query_flags(query=query, planner=planner_unspecified)
            == expected
        )


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
