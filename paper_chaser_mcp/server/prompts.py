"""Paper Chaser prompt registrations (decorated at package import time)."""

from __future__ import annotations

from typing import Any, Callable, Literal


def register_prompts(app: Any) -> dict[str, Callable[..., Any]]:
    """Attach Paper Chaser prompt handlers to ``app`` and return the decorated callables."""

    @app.prompt(
        name="plan_paper_chaser_search",
        title="Plan Paper Chaser",
        description="Generate a tool-first plan for a literature search task.",
    )
    def plan_paper_chaser_search(
        topic: str,
        goal: str = "find relevant papers, follow citations, and summarize next steps",
        mode: Literal["smoke", "comprehensive", "feature_probe"] = "smoke",
        focus_prompt: str | None = None,
    ) -> str:
        """Create a reusable research workflow prompt for clients."""
        mode_guidance = {
            "smoke": (
                "Run a smoke-style review that stays focused on the primary golden "
                "paths: quick discovery, known-item lookup, pagination, author pivot, "
                "and optional citation export."
            ),
            "comprehensive": (
                "Run a comprehensive UX review: cover the smoke baseline first, then "
                "add deeper probes for references, paper-to-author pivots, snippet "
                "recovery, and explicit OpenAlex workflows."
            ),
            "feature_probe": (
                "Run a feature-probe review: keep a short smoke baseline, then spend "
                "most of the effort on the requested feature or UX hypothesis and the "
                "tool paths that exercise it."
            ),
        }
        focus_text = f" Focus prompt: {focus_prompt}." if focus_prompt else " No extra focus prompt was supplied."
        return (
            f"You are planning a Paper Chaser workflow about '{topic}'. Goal: {goal}. "
            f"Mode: {mode}. {mode_guidance[mode]}{focus_text} "
            "Default to the guided surface. Start with research for discovery, known-item "
            "recovery, citation repair, and regulatory routing. If the user already has a "
            "citation-like string, use resolve_reference first. If guided research returns "
            "needs_disambiguation with clarification for an underspecified fragment, tighten "
            "the anchor instead of forcing retrieval. Reuse searchSessionId with "
            "follow_up_research for one grounded question and inspect_source for provenance. "
            "Treat abstentions and clarification requests as real outputs, not failures to hide. "
            "Only fall back to the expert surface when the task explicitly requires provider-specific control, "
            "pagination, or provider-native payloads. On the expert surface, search_papers is "
            "the quick brokered path, search_papers_bulk is the exhaustive Semantic Scholar-style "
            "path, and search_papers_smart/map_research_landscape/expand_research_graph are the "
            "deeper agentic tools. If the task explicitly needs OpenAlex-native DOI/ID lookup, "
            "OpenAlex cursor pagination, or OpenAlex author/citation semantics, use the dedicated "
            "*_openalex tools instead of the default broker. For exact paper follow-through, use "
            "get_paper_details, get_paper_citations, get_paper_references, search_authors, "
            "get_author_info, and get_author_papers as needed. "
            "For regulatory work, prefer the guided path first; if you need exact primary-source "
            "control, pivot into search_federal_register, get_federal_register_document, get_cfr_text, "
            "or the ECOS tools. "
            "If your goal is repo-local eval bootstrap or workflow QA instead of answering an end-user "
            "research ask, prefer scripts/generate_eval_topics.py, scripts/run_eval_autopilot.py, and "
            "scripts/run_eval_workflow.py so the run stays reproducible and produces the expected bundle "
            "artifacts. "
            "If you uncover a defect or confusing UX, summarize the exact tool calls, "
            "expected vs actual behavior, and whether the best follow-up is a code "
            "change, a documentation update, or both so the result can turn into an "
            "actionable issue for a GitHub Copilot coding agent. "
            "Treat pagination.nextCursor as opaque: reuse it exactly as returned, do "
            "not edit or fabricate it, and keep it scoped to the tool/query flow that "
            "produced it."
        )

    @app.prompt(
        name="plan_smart_paper_chaser_search",
        title="Plan Smart Paper Chaser",
        description=("Generate a smart-tool-first research plan for concept-level discovery."),
    )
    def plan_smart_paper_chaser_search(
        topic: str,
        goal: str = ("map the literature, answer grounded follow-up questions, and identify the best next actions"),
        mode: Literal["discovery", "review", "known_item", "author", "citation"] = "discovery",
    ) -> str:
        return (
            f"You are planning a smart Paper Chaser workflow about '{topic}'. "
            f"Goal: {goal}. Mode: {mode}. Start with research unless you have a concrete "
            "reason to force the expert smart surface. If you do need expert smart behavior, "
            "use search_papers_smart for concept-level discovery and reuse searchSessionId across "
            "ask_result_set, map_research_landscape, and expand_research_graph. For broken citations "
            "or almost-right references, prefer resolve_reference on the guided path before broader "
            "discovery. If the smart workflow cannot stay grounded, drop back to research, "
            "inspect_source, or the raw expert tools such as search_papers, search_papers_bulk, "
            "get_paper_details, get_paper_citations, get_paper_references, search_authors, and "
            "get_author_papers. When you already have the right paper and want richer metadata or OA "
            "signals, use the enrichment tools after resolution."
        )

    @app.prompt(
        name="triage_literature",
        title="Triage Literature",
        description="Turn a research topic into a compact triage workflow.",
    )
    def triage_literature(
        topic: str,
        goal: str = "identify core themes, strongest anchors, and the next best tool call",
    ) -> str:
        return (
            f"Triage literature for '{topic}'. Goal: {goal}. Start with "
            "research. Inspect resultStatus, answerability, evidence, leads, routingSummary, "
            "coverageSummary, evidenceGaps, failureSummary, "
            "and clarification. Save the searchSessionId, then ask one grounded question with "
            "follow_up_research. If one hit becomes a strong anchor, use inspect_source for "
            "provenance before treating it as settled."
        )

    @app.prompt(
        name="plan_citation_chase",
        title="Plan Citation Chase",
        description="Generate a citation-expansion workflow from a paper anchor.",
    )
    def plan_citation_chase(
        paper_id: str,
        direction: Literal["citations", "references"] = "citations",
        goal: str = "find the most influential neighboring work and preserve provenance",
    ) -> str:
        return (
            f"Plan a citation chase from paper '{paper_id}' in the "
            f"'{direction}' direction. Goal: {goal}. Prefer expand_research_graph "
            "for a compact frontier. If you need "
            "provider-native control or pagination, use get_paper_citations or "
            "get_paper_references directly and treat pagination.nextCursor as opaque."
        )

    @app.prompt(
        name="refine_query",
        title="Refine Query",
        description="Generate a bounded query-refinement workflow for the current topic.",
    )
    def refine_query(
        query: str,
        weakness: str = "results are too broad, too narrow, or too noisy",
    ) -> str:
        return (
            f"Refine the query '{query}'. Problem signal: {weakness}. "
            "Try research first and inspect resultStatus, answerability, routingSummary, coverageSummary, "
            "evidenceGaps, failureSummary, and clarification. If the guided path abstains, "
            "add a concrete anchor such as a year, "
            "venue, DOI, species name, agency, or title fragment. Use get_runtime_status when behavior "
            "differs across environments and you need the active runtime truth."
        )

    return {
        "plan_paper_chaser_search": plan_paper_chaser_search,
        "plan_smart_paper_chaser_search": plan_smart_paper_chaser_search,
        "triage_literature": triage_literature,
        "plan_citation_chase": plan_citation_chase,
        "refine_query": refine_query,
    }


__all__ = ["register_prompts"]
