"""Tests for LLM-first mixed regulatory+literature intent classification.

The planner already emits ``regulatoryIntent == "hybrid_regulatory_plus_literature"``
as an LLM-first signal (with a deterministic fallback inside the planner). The
guided dispatch layer consumes it in two places:

* ``_guided_is_mixed_intent_query`` accepts an optional
  ``planner_regulatory_intent`` kwarg so upfront-planner callers bypass the
  keyword heuristic entirely.
* ``_guided_should_add_review_pass`` inspects ``strategyMetadata.regulatoryIntent``
  after the first smart pass and forces a review pass when the planner tagged
  the query as hybrid, regardless of whether the keyword heuristic would have
  caught it.
"""

from __future__ import annotations

from paper_chaser_mcp.dispatch import (
    _guided_is_mixed_intent_query,
    _guided_mentions_literature,
    _guided_should_add_review_pass,
)


class TestGuidedIsMixedIntentQuery:
    """Verify planner-first consumption with deterministic keyword fallback."""

    def test_planner_hybrid_signal_honored_when_keywords_miss(self) -> None:
        """When the planner labels the query hybrid, trust it even if the keyword
        heuristic would have returned False (no literature keyword present)."""

        query = "federal rules on climate research"
        # Keyword fallback alone should miss this: "research" is not in the
        # guided literature term list, so _guided_mentions_literature is False.
        assert _guided_mentions_literature(query) is False
        # Deterministic call (no planner context) => False.
        assert _guided_is_mixed_intent_query(query) is False
        # Planner-first call => True.
        assert (
            _guided_is_mixed_intent_query(
                query,
                planner_regulatory_intent="hybrid_regulatory_plus_literature",
            )
            is True
        )

    def test_planner_non_hybrid_signal_does_not_force_mixed(self) -> None:
        """A non-hybrid planner label must not spuriously promote mixed intent.

        The keyword heuristic still runs for the final answer, preserving
        the existing deterministic contract.
        """

        query = "a general question with no regulatory or literature markers"
        assert (
            _guided_is_mixed_intent_query(
                query,
                planner_regulatory_intent="current_cfr_text",
            )
            is False
        )

    def test_deterministic_fallback_unchanged_when_planner_absent(self) -> None:
        """When no planner signal is provided, behavior matches the pre-existing
        regulatory + literature keyword heuristic exactly."""

        mixed_query = "CFR regulations and peer-reviewed studies on PFAS"
        pure_regulatory = "CFR section 40 text on hazardous waste"
        pure_literature = "peer-reviewed studies on climate adaptation"

        assert _guided_is_mixed_intent_query(mixed_query) is True
        assert _guided_is_mixed_intent_query(pure_regulatory) is False
        assert _guided_is_mixed_intent_query(pure_literature) is False


class TestGuidedShouldAddReviewPass:
    """Verify the review-pass trigger prefers the planner's LLM signal."""

    def test_planner_hybrid_intent_triggers_review_pass(self) -> None:
        """When the planner emits ``hybrid_regulatory_plus_literature`` AND the
        query carries an independent literature cue, the review pass fires.

        This replaces the prior "LLM signal alone is enough" contract -- the
        hybrid label must now be corroborated by the query itself (see
        ``_guided_mentions_literature``), the planner's ``secondaryIntents``,
        or explicit literature-shaped ``retrievalHypotheses`` -- so the
        planner cannot hallucinate its way into an unneeded review pass.
        """

        primary_smart = {
            "strategyMetadata": {
                "regulatoryIntent": "hybrid_regulatory_plus_literature",
                "intentSource": "planner",
                "secondaryIntents": [],
                "ambiguityLevel": "low",
                "querySpecificity": "high",
                "retrievalHypotheses": [],
            }
        }
        should_run, reason = _guided_should_add_review_pass(
            initial_intent="regulatory",
            # "literature" is an independent corroborating cue
            query="federal rules on climate change and peer-reviewed literature",
            focus=None,
            primary_smart=primary_smart,
            pass_modes=["regulatory"],
        )
        assert should_run is True
        assert reason == "planner_hybrid_regulatory_plus_literature"

    def test_planner_hybrid_intent_without_corroboration_does_not_trigger(self) -> None:
        """Regression guard for Finding 2: a regulation-only query where the
        planner hallucinates ``hybrid_regulatory_plus_literature`` must NOT
        trigger the review pass, because no independent literature cue
        corroborates the hybrid label.
        """

        primary_smart = {
            "strategyMetadata": {
                "regulatoryIntent": "hybrid_regulatory_plus_literature",
                "intentSource": "planner",
                "secondaryIntents": [],
                "ambiguityLevel": "low",
                "querySpecificity": "high",
                "retrievalHypotheses": [],
            }
        }
        should_run, reason = _guided_should_add_review_pass(
            initial_intent="regulatory",
            query="What does EPA require for stormwater discharges?",
            focus=None,
            primary_smart=primary_smart,
            pass_modes=["regulatory"],
        )
        assert should_run is False
        assert reason is None

    def test_planner_hybrid_intent_with_secondary_review_triggers_review_pass(self) -> None:
        """Corroboration via ``secondaryIntents=["review"]`` is enough to honor
        the planner's hybrid label even when the query itself has no keyword
        literature cue.
        """

        primary_smart = {
            "strategyMetadata": {
                "regulatoryIntent": "hybrid_regulatory_plus_literature",
                "intentSource": "planner",
                "secondaryIntents": ["review"],
                "ambiguityLevel": "low",
                "querySpecificity": "high",
                "retrievalHypotheses": [],
            }
        }
        should_run, reason = _guided_should_add_review_pass(
            initial_intent="regulatory",
            query="federal rules on climate mitigation",
            focus=None,
            primary_smart=primary_smart,
            pass_modes=["regulatory"],
        )
        assert should_run is True
        assert reason == "planner_hybrid_regulatory_plus_literature"

    def test_no_planner_hybrid_preserves_existing_heuristic(self) -> None:
        """When the planner does not emit the hybrid label, the existing
        deterministic heuristic path must be preserved unchanged."""

        # Pure regulatory metadata with no literature signals => no review pass.
        primary_smart = {
            "strategyMetadata": {
                "regulatoryIntent": "current_cfr_text",
                "intentSource": "planner",
                "secondaryIntents": [],
                "ambiguityLevel": "low",
                "querySpecificity": "high",
                "retrievalHypotheses": [],
            }
        }
        should_run, reason = _guided_should_add_review_pass(
            initial_intent="regulatory",
            query="Current 40 CFR 261 hazardous waste text",
            focus=None,
            primary_smart=primary_smart,
            pass_modes=["regulatory"],
        )
        assert should_run is False
        assert reason is None

    def test_review_already_run_short_circuits(self) -> None:
        """If the review pass already ran, no duplicate is scheduled even when
        the planner says hybrid (contract preserved)."""

        primary_smart = {
            "strategyMetadata": {
                "regulatoryIntent": "hybrid_regulatory_plus_literature",
            }
        }
        should_run, reason = _guided_should_add_review_pass(
            initial_intent="regulatory",
            query="whatever",
            focus=None,
            primary_smart=primary_smart,
            pass_modes=["regulatory", "review"],
        )
        assert should_run is False
        assert reason is None
