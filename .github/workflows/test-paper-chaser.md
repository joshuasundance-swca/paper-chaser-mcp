---
description: |
  Exercise the primary Paper Chaser MCP golden paths with an agent so the
  repo gets a high-level regression check of discovery, smart result-set
  research, pagination, author pivot, and optional citation-export workflows.
  Evaluates agent UX quality: intuitiveness, unnecessary round trips, missing
  features, and friction points that make common workflows harder than they
  should be.
  This workflow is manual-only by design because it can consume repository
  secrets in a public repo. Manual runs can switch between smoke,
  comprehensive, and feature-probe UX review modes with an optional focus
  prompt.
  Requires a repository or organization secret named COPILOT_GITHUB_TOKEN.

on:
  workflow_dispatch:
    inputs:
      mode:
        description: Choose the UX review depth for this run.
        type: choice
        default: smoke
        options: [smoke, comprehensive, feature_probe]
      focus_prompt:
        description: Optional feature, workflow, or UX hypothesis to probe.
        type: string
        required: false

permissions: read-all

network: defaults

engine: copilot

timeout-minutes: 20

safe-outputs:
  create-issue:
    title-prefix: "[agentic-test] "
    labels: [automation, testing, agentic, needs-copilot]
    max: 1

tools:
  github:
    toolsets: [issues]
  bash: true

mcp-servers:
  paper-chaser:
    type: stdio
    container: python:3.12
    entrypoint: sh
    entrypointArgs:
      - -lc
      - cd ${GITHUB_WORKSPACE} && pip install -e .[dev] >/tmp/paper-chaser-agentic-install.log && python -m paper_chaser_mcp
    mounts:
      - ${{ github.workspace }}:${{ github.workspace }}:rw
      - /tmp:/tmp:rw
    env:
      GITHUB_WORKSPACE: "${{ github.workspace }}"
      CORE_API_KEY: "${{ secrets.CORE_API_KEY }}"
      SEMANTIC_SCHOLAR_API_KEY: "${{ secrets.SEMANTIC_SCHOLAR_API_KEY }}"
      OPENAI_API_KEY: "${{ secrets.OPENAI_API_KEY }}"
      PAPER_CHASER_ENABLE_AGENTIC: "true"
    allowed:
      - search_papers
      - search_papers_smart
      - ask_result_set
      - map_research_landscape
      - expand_research_graph
      - search_papers_bulk
      - search_papers_core
      - search_papers_semantic_scholar
      - search_papers_serpapi
      - search_papers_arxiv
      - search_papers_openalex
      - search_papers_openalex_bulk
      - resolve_citation
      - search_papers_match
      - get_paper_details
      - get_paper_details_openalex
      - get_paper_citations
      - get_paper_references
      - get_paper_citations_openalex
      - get_paper_references_openalex
      - get_paper_authors
      - search_authors
      - get_author_info
      - get_author_papers
      - search_authors_openalex
      - get_author_info_openalex
      - get_author_papers_openalex
      - get_paper_citation_formats
      - search_snippets
---

# Test Paper Chaser MCP

You are validating the `paper-chaser-mcp` server as a critical end-to-end reviewer,
not just a functional tester. Your primary job is to evaluate **agent UX quality**:
how intuitive the tools feel, how many round trips tasks require, what features are
missing, and where the experience creates unnecessary friction.

Use the MCP tools first; use bash only when you need short local validation or to
inspect structured output more carefully.

## Agent UX evaluation mandate

For every tool interaction, actively note:

- **Intuitiveness**: Did the tool do what you expected from its name and description
  alone? Would a new agent use it correctly on the first try?
- **Round trips**: How many tool calls did this task require? Could it be done in
  fewer? Flag any workflow that needs 3+ calls where 1-2 should suffice.
- **Missing features**: What did you wish you could do but couldn't? What fields or
  capabilities would make a task obviously easier?
- **Confusing contracts**: Anything where the response shape, field names, or error
  messages were ambiguous, surprising, or inconsistent with what the description promised.
- **Dead ends**: Any path that left you without a clear next step or that returned
  data you couldn't act on without undocumented knowledge.

Keep a running tally of friction points throughout the run. These are as important
as functional pass/fail results.

## Run context

- Requested mode: `${{ inputs.mode }}`
- Requested focus prompt: `${{ inputs.focus_prompt }}`
- This workflow should be dispatched manually from the Actions tab so operators
  can choose the review depth intentionally.
- When a focus prompt is present, restate it in your notes and use it to choose
  the deeper probes after the baseline checks.

## Prerequisites

- Configure a repository or organization Actions secret named
  `COPILOT_GITHUB_TOKEN`. The GitHub Agentic Workflow Copilot engine fails in
  activation before the MCP server starts if this secret is missing.
- `CORE_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY` remain optional. They improve
  coverage and rate limits but are not required for the workflow to start.

## Goals

1. Confirm the quick-discovery path returns structured, useful metadata.
2. Confirm the smart-discovery path creates a reusable `searchSessionId` and
   supports grounded follow-up.
3. Confirm the exhaustive-retrieval path handles cursor pagination safely.
4. Confirm citation repair, known-item lookup, and citation chasing still work from the same MCP
   surface agents use in production.
5. Confirm the author-pivot path works from the same MCP surface agents use in
   production.
6. Confirm optional citation export still works when SerpApi is available.
7. **Critically evaluate agent UX**: identify every friction point, unnecessary
   round trip, missing feature, confusing field name, or dead-end response that
   degrades the experience. This goal is weighted equally with the functional checks.

## Mode guide

- `smoke`: run the baseline checks below and keep the review focused on the
  primary golden paths.
- `comprehensive`: run the smoke checks, then add deeper follow-up probes for
  references, paper-to-author pivots, snippet recovery, and OpenAlex-specific
  paths.
- `feature_probe`: run a short baseline sanity check first, then spend most of
  the remaining time on the supplied focus prompt and whichever tools best match
  that feature or UX hypothesis.

## Test protocol

1. **Choose the review plan**
   - Determine the effective mode from the run context above. If the mode is
     blank for any reason, use `smoke`.
   - If a focus prompt is present, note which tools or paths it most likely
     exercises before you start the deeper probes.
   - Always run at least a lightweight baseline so regressions in the core
     workflow are still visible.
   - **Before you begin**: open a running notes section called "UX friction log"
     where you will record every moment the experience felt awkward, slow, or
     required undocumented knowledge.

2. **Quick discovery** (all modes)
   - Call `search_papers(query="graph neural networks", limit=5)`.
   - Record `brokerMetadata.providerUsed`, `brokerMetadata.nextStepHint`, and the
     titles/providers of the first few results.
   - Verify every returned paper has a `title` and at least one author entry.
   - **UX check**: Was `nextStepHint` actionable on first read, or did it require
     interpretation? Did the response tell you clearly what to do next without
     reading the docs?

3. **Smart discovery and grounded follow-up** (all modes)
   - Call `search_papers_smart(query="graph neural networks", limit=5)`.
   - Record `searchSessionId`, `strategyMetadata.queryVariantsTried`,
     `acceptedExpansions`, `rejectedExpansions`, and the top `whyMatched`
     explanations.
   - Call `ask_result_set(searchSessionId=..., question="What does this result set say about graph neural network benchmarks?")`.
   - Confirm the smart response exposes `resourceUris`, `agentHints`, and a
     reusable `searchSessionId`.
   - **UX check**: Did the smart result set reduce round trips compared with the
     raw flow, or did it hide too much reasoning behind opaque metadata?

4. **Citation repair and known-item lookup** (all modes)
   - Call `resolve_citation(citation="Rockstrom et al planetary boundaries 2009 Nature 461 472")`.
   - Record `resolutionConfidence`, `resolutionStrategy`, `matchedFields`,
     `conflictingFields`, and whether the response surfaced plausible alternatives.
   - Then call `search_papers_match(query="Attention Is All You Need")`.
   - Use the returned identifier with `get_paper_details`.
   - Confirm the detailed record includes a stable identifier, title, and author
     metadata.
   - **UX check**: Did citation repair reduce the need to guess between title match,
     snippets, and broad search? If not, note the friction explicitly.

5. **Known-item lookup detail** (all modes)
   - Call `search_papers_match(query="Attention Is All You Need")`.
   - Use the returned identifier with `get_paper_details`.
   - Confirm the detailed record includes a stable identifier, title, and author
     metadata.
   - **UX check**: How many round trips did this require? Could the match result
     itself carry enough detail to skip `get_paper_details` for most tasks?

6. **Exhaustive retrieval / pagination** (all modes)
   - Call `search_papers_bulk(query="graph neural networks", limit=5)`.
   - If `pagination.nextCursor` is present, call `search_papers_bulk` again with
     the exact `cursor` value that was returned.
   - Confirm there are no duplicate `paperId` values across the first two pages.
   - If no second page is available, note that result instead of forcing a
     failure.
   - **UX check**: Is the pagination contract obvious from the response alone,
     or does it require reading docs to know that `cursor` must be passed back
     unchanged?

7. **Provider-specific spot checks** (smoke + comprehensive)
   - Run `search_papers_core(query="transformer architecture", limit=3)`.
   - Run `search_papers_semantic_scholar(query="transformer architecture", limit=3)`.
   - Run `search_papers_arxiv(query="transformer architecture", limit=3)`.
   - Verify each provider returns the normalized response shape the MCP server
     promises. For arXiv, look for arXiv-native identifiers/URLs when present;
     for CORE and Semantic Scholar, note whether DOI metadata is populated.
   - **UX check**: Are the per-provider tools obviously named and scoped? Does
     an agent know when to use a provider-specific tool versus `search_papers`?

8. **Author pivot** (all modes)
   - Run `search_authors(query="Yoshua Bengio", limit=3)`.
   - Use the top author identifier with `get_author_info`.
   - Call `get_author_papers(author_id=..., publicationDateOrYear="2022-", limit=5)`.
   - Confirm the author profile is stable enough to continue from and that the
     paper list response is structured cleanly even if recent results are sparse.
   - **UX check**: How many round trips does a "who is this author and what have
     they published recently?" task require? What fields are missing that would
     make author-to-paper pivots easier?

9. **Optional SerpApi citation export** (smoke + comprehensive)
   - If the SerpApi tools are configured, run
     `search_papers_serpapi(query="Attention Is All You Need", limit=3)`.
   - If a result exposes `scholarResultId`, call
     `get_paper_citation_formats(result_id=paper.scholarResultId)`.
   - Confirm the citation-export response returns at least one named format.
   - If SerpApi is unavailable, disabled, or returns no usable `scholarResultId`,
     record a clean skip instead of failing the workflow.
   - **UX check**: Is it obvious from the search result that `scholarResultId`
     is the right identifier to pass to `get_paper_citation_formats`? Or does
     it require reading docs?

10. **No-results behavior** (all modes)
   - Call `search_papers(query="asdkfjhasdkjfh research paper nonsense", limit=3)`.
   - Verify the server responds cleanly with an empty/low-result payload rather
     than malformed metadata or an unhelpful failure.
   - **UX check**: Does the empty-result payload suggest a clear recovery path
     (e.g., broaden query, try different provider), or does it just say "no results"?

11. **Comprehensive-only follow-up probes**
   - If the effective mode is `comprehensive`, also:
     - Call `map_research_landscape(searchSessionId=..., maxThemes=4)` using the
       smart search session and confirm the response surfaces themes, gaps, and
       suggested next searches clearly.
     - Call `expand_research_graph(seedSearchSessionId=..., direction="citations")`
       and confirm the graph frontier is actionable rather than decorative.
     - Call `get_paper_references` for the known paper and confirm backward
       references are structured clearly.
     - Call `get_paper_authors` for the known paper and confirm the paper-to-
       author pivot feels first-class.
     - Call `search_snippets(query="attention is all you need", limit=3)` and
       confirm the response degrades cleanly if the provider does not support the
       phrase query.
     - Run `search_papers_openalex(query="transformer architecture", limit=3)`
       and then either `get_paper_details_openalex` or an OpenAlex citation /
       author tool on one returned item so OpenAlex-specific UX gets coverage.

12. **Feature-probe follow-up**
    - If the effective mode is `feature_probe`, use the supplied focus prompt to
      choose the most relevant extra tool path. Examples:
      - feature about concept discovery or grounded follow-up → use
        `search_papers_smart`, `ask_result_set`, `map_research_landscape`, and
        `expand_research_graph`
      - feature about OpenAlex DOI or author flows → use the explicit
        `*_openalex` tools
      - feature about known-item recovery or citation repair → use
        `resolve_citation`, `search_papers_match`, `get_paper_details`, and
        `search_snippets`
      - feature about citations or references → use `get_paper_citations` and
        `get_paper_references`
    - Keep the baseline notes, but spend most of the detailed analysis on the
      requested feature or UX hypothesis.

13. **UX friction summary** (all modes)
    - Review your UX friction log and produce a structured summary:
      - **Unnecessary round trips**: list each multi-step workflow that required
        more tool calls than expected. State the minimum you'd expect and the
        actual count.
      - **Missing features**: list capabilities you wished existed. Be specific:
        e.g., "bulk author lookup", "cross-provider deduplication", "search with
        date range in brokered path".
      - **Confusing contracts**: list fields, tool names, or response shapes that
        were ambiguous or inconsistent. Explain what you expected versus what
        you got.
      - **Dead ends**: list any point where the response left no clear next step.
      - **Positive signals**: briefly note what felt especially smooth or
        well-designed so it is not accidentally degraded in future changes.

13. **Assessment and issue creation**
    - Summarize the effective mode, any supplied focus prompt, the exact tool
      calls you used, whether cursor handling felt safe and clear, whether author
      workflows felt first-class, and whether any provider-specific gaps stood
      out.
    - Prioritize the highest-impact item from your UX friction summary for the
      issue. Prefer issues that reduce round trips or add a clearly missing
      feature over issues that are purely cosmetic.
    - If you find a concrete defect or UX regression, create at most one GitHub
      issue with reproduction steps, expected vs actual behavior, and clear
      remediation guidance that points to likely code and/or documentation
      follow-up for a GitHub Copilot coding agent.
    - Before creating a new issue, search existing open issues for one whose
      body contains a matching `agent-loop-key` comment for this failure. If a
      matching issue exists, add a comment with the latest repro details and
      increment the attempt counter instead of opening a duplicate.
    - When creating a new issue, include the following markers at the top of the
      body so the agentic loop can track convergence (no code block — paste
      these HTML comments directly into the issue body):

      <!-- agent-loop-key: <short-slug-for-the-failure> -->
      <!-- agent-loop-origin: golden-path-smoke -->
      <!-- agent-loop-attempts: 1 -->
      <!-- agent-loop-last-run: <ISO 8601 timestamp of this run> -->

    - When updating an existing issue, increment `agent-loop-attempts` and
      update `agent-loop-last-run`. If `agent-loop-attempts` reaches 3 or more,
      add the label `needs-human` and remove `needs-copilot` in your comment so
      the assigner stops retrying automatically.
