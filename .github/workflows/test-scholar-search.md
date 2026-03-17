---
description: |
  Exercise the primary Scholar Search MCP golden paths with an agent so the
  repo gets a scheduled, high-level regression check of discovery,
  pagination, author pivot, and optional citation-export workflows. Manual
  runs can switch between smoke, comprehensive, and feature-probe UX review
  modes with an optional focus prompt.
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
  schedule: weekly on sunday

permissions: read-all

network: defaults

engine: copilot

timeout-minutes: 20

safe-outputs:
  create-issue:
    title-prefix: "[agentic-test] "
    labels: [automation, testing]
    max: 1

tools:
  github:
    toolsets: [issues]
  bash: true

mcp-servers:
  scholar-search:
    type: stdio
    container: python:3.12
    entrypoint: sh
    entrypointArgs:
      - -lc
      - cd ${GITHUB_WORKSPACE} && pip install -e .[dev] >/tmp/scholar-search-agentic-install.log && python -m scholar_search_mcp
    mounts:
      - ${{ github.workspace }}:${{ github.workspace }}:rw
      - /tmp:/tmp:rw
    env:
      GITHUB_WORKSPACE: "${{ github.workspace }}"
      CORE_API_KEY: "${{ secrets.CORE_API_KEY }}"
      SEMANTIC_SCHOLAR_API_KEY: "${{ secrets.SEMANTIC_SCHOLAR_API_KEY }}"
    allowed:
      - search_papers
      - search_papers_bulk
      - search_papers_core
      - search_papers_semantic_scholar
      - search_papers_serpapi
      - search_papers_arxiv
      - search_papers_openalex
      - search_papers_openalex_bulk
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

# Test Scholar Search MCP

You are validating the `scholar-search-mcp` server as an end-to-end user would.
Use the MCP tools first; use bash only when you need short local validation or to
inspect structured output more carefully.

This workflow is a smoke test for the primary user journeys, but manual runs can
expand into a broader UX review loop instead of staying fixed on one script.
Focus on the core paths that should stay reliable for agents: quick discovery,
known-item lookup, pagination, citation chasing, author pivot, and optional
citation export when SerpApi is available.

## Run context

- Requested mode: `${{ inputs.mode }}`
- Requested focus prompt: `${{ inputs.focus_prompt }}`
- Scheduled runs do not supply manual inputs; when the mode is blank, treat this
  run as `smoke`.
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
2. Confirm the exhaustive-retrieval path handles cursor pagination safely.
3. Confirm known-item lookup and citation chasing still work from the same MCP
   surface agents use in production.
4. Confirm the author-pivot path works from the same MCP surface agents use in
   production.
5. Confirm optional citation export still works when SerpApi is available.
6. Capture UX feedback about missing fields, confusing metadata, no-result
   behavior, and whether the findings suggest code changes, documentation
   updates, or a GitHub issue ready for a Copilot coding agent.

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
   - Determine the effective mode from the run context above. If no manual mode
     is present, use `smoke`.
   - If a focus prompt is present, note which tools or paths it most likely
     exercises before you start the deeper probes.
   - Always run at least a lightweight baseline so regressions in the core
     workflow are still visible.

2. **Quick discovery** (all modes)
   - Call `search_papers(query="graph neural networks", limit=5)`.
   - Record `brokerMetadata.providerUsed`, `brokerMetadata.nextStepHint`, and the
     titles/providers of the first few results.
   - Verify every returned paper has a `title` and at least one author entry.

3. **Known-item lookup** (all modes)
   - Call `search_papers_match(query="Attention Is All You Need")`.
   - Use the returned identifier with `get_paper_details`.
   - Confirm the detailed record includes a stable identifier, title, and author
     metadata.

4. **Exhaustive retrieval / pagination** (all modes)
   - Call `search_papers_bulk(query="graph neural networks", limit=5)`.
   - If `pagination.nextCursor` is present, call `search_papers_bulk` again with
     the exact `cursor` value that was returned.
   - Confirm there are no duplicate `paperId` values across the first two pages.
   - If no second page is available, note that result instead of forcing a
     failure.

5. **Provider-specific spot checks** (smoke + comprehensive)
   - Run `search_papers_core(query="transformer architecture", limit=3)`.
   - Run `search_papers_semantic_scholar(query="transformer architecture", limit=3)`.
   - Run `search_papers_arxiv(query="transformer architecture", limit=3)`.
   - Verify each provider returns the normalized response shape the MCP server
     promises. For arXiv, look for arXiv-native identifiers/URLs when present;
     for CORE and Semantic Scholar, note whether DOI metadata is populated.

6. **Author pivot** (all modes)
   - Run `search_authors(query="Yoshua Bengio", limit=3)`.
   - Use the top author identifier with `get_author_info`.
   - Call `get_author_papers(author_id=..., publicationDateOrYear="2022-", limit=5)`.
   - Confirm the author profile is stable enough to continue from and that the
     paper list response is structured cleanly even if recent results are sparse.

7. **Optional SerpApi citation export** (smoke + comprehensive)
   - If the SerpApi tools are configured, run
     `search_papers_serpapi(query="Attention Is All You Need", limit=3)`.
   - If a result exposes `scholarResultId`, call
     `get_paper_citation_formats(result_id=paper.scholarResultId)`.
   - Confirm the citation-export response returns at least one named format.
   - If SerpApi is unavailable, disabled, or returns no usable `scholarResultId`,
     record a clean skip instead of failing the workflow.

8. **No-results behavior** (all modes)
   - Call `search_papers(query="asdkfjhasdkjfh research paper nonsense", limit=3)`.
   - Verify the server responds cleanly with an empty/low-result payload rather
     than malformed metadata or an unhelpful failure.

9. **Comprehensive-only follow-up probes**
   - If the effective mode is `comprehensive`, also:
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

10. **Feature-probe follow-up**
    - If the effective mode is `feature_probe`, use the supplied focus prompt to
      choose the most relevant extra tool path. Examples:
      - feature about OpenAlex DOI or author flows → use the explicit
        `*_openalex` tools
      - feature about known-item recovery → use `search_papers_match`,
        `get_paper_details`, and `search_snippets`
      - feature about citations or references → use `get_paper_citations` and
        `get_paper_references`
    - Keep the baseline notes, but spend most of the detailed analysis on the
      requested feature or UX hypothesis.

11. **Assessment and issue creation**
    - Summarize the effective mode, any supplied focus prompt, the exact tool
      calls you used, whether cursor handling felt safe and clear, whether author
      workflows felt first-class, and whether any provider-specific gaps stood
      out.
    - If you find a concrete defect or UX regression, create at most one GitHub
      issue with reproduction steps, expected vs actual behavior, and clear
      remediation guidance that points to likely code and/or documentation
      follow-up for a GitHub Copilot coding agent.
