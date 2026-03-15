---
description: |
  Exercise the public Scholar Search MCP tool surface with an agent so the repo
  gets a scheduled, high-level regression check of the golden-path workflows.

on:
  workflow_dispatch:
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
    command: bash
    args:
      - -lc
      - pip install -e .[dev] >/tmp/scholar-search-agentic-install.log && python -m scholar_search_mcp
    env:
      CORE_API_KEY: "${{ secrets.CORE_API_KEY }}"
      SEMANTIC_SCHOLAR_API_KEY: "${{ secrets.SEMANTIC_SCHOLAR_API_KEY }}"
    allowed:
      - search_papers
      - search_papers_bulk
      - search_papers_core
      - search_papers_semantic_scholar
      - search_papers_arxiv
      - search_papers_match
      - get_paper_details
      - get_paper_citations
      - get_paper_references
---

# Test Scholar Search MCP

You are validating the `scholar-search-mcp` server as an end-to-end user would.
Use the MCP tools first; use bash only when you need short local validation or to
inspect structured output more carefully.

## Goals

1. Confirm the quick-discovery path returns structured, useful metadata.
2. Confirm the exhaustive-retrieval path handles cursor pagination safely.
3. Confirm known-item lookup and citation chasing still work from the same MCP
   surface agents use in production.
4. Capture UX feedback about missing fields, confusing metadata, or no-result
   behavior.

## Test protocol

1. **Quick discovery**
   - Call `search_papers(query="graph neural networks", limit=5)`.
   - Record `brokerMetadata.providerUsed`, `brokerMetadata.nextStepHint`, and the
     titles/providers of the first few results.
   - Verify every returned paper has a `title` and at least one author entry.

2. **Known-item lookup**
   - Call `search_papers_match(query="Attention Is All You Need")`.
   - Use the returned identifier with `get_paper_details`.
   - Confirm the detailed record includes a stable identifier, title, and author
     metadata.

3. **Exhaustive retrieval / pagination**
   - Call `search_papers_bulk(query="graph neural networks", limit=5)`.
   - If `pagination.nextCursor` is present, call `search_papers_bulk` again with
     the exact `cursor` value that was returned.
   - Confirm there are no duplicate `paperId` values across the first two pages.
   - If no second page is available, note that result instead of forcing a
     failure.

4. **Provider-specific spot checks**
   - Run `search_papers_core(query="transformer architecture", limit=3)`.
   - Run `search_papers_semantic_scholar(query="transformer architecture", limit=3)`.
   - Run `search_papers_arxiv(query="transformer architecture", limit=3)`.
   - Verify each provider returns the normalized response shape the MCP server
     promises. For arXiv, look for arXiv-native identifiers/URLs when present;
     for CORE and Semantic Scholar, note whether DOI metadata is populated.

5. **No-results behavior**
   - Call `search_papers(query="asdkfjhasdkjfh research paper nonsense", limit=3)`.
   - Verify the server responds cleanly with an empty/low-result payload rather
     than malformed metadata or an unhelpful failure.

6. **Assessment**
   - Summarize whether the server returned consistent metadata, whether cursor
     handling felt safe and clear, and whether any provider-specific gaps stood
     out.
   - If you find a concrete defect or UX regression, create at most one GitHub
     issue with reproduction steps and clear remediation guidance.
