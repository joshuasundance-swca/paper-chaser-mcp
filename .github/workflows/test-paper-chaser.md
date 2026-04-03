---
description: |
  Exercise the primary Paper Chaser MCP golden paths with an agent so the
  repo gets a high-level regression check of the default guided UX first,
  with optional expert-surface follow-up when requested. Evaluates agent UX
  quality: intuitiveness, unnecessary round trips, missing features, and
  friction points that make common workflows harder than they should be.
  This workflow is manual-only by design because it can consume repository
  secrets in a public repo. Manual runs can switch between smoke,
  comprehensive, and feature-probe UX review modes, choose the guided or
  expert tool profile, and optionally supply a focus prompt.
  Requires a repository or organization secret named COPILOT_GITHUB_TOKEN.

on:
  workflow_dispatch:
    inputs:
      mode:
        description: Choose the UX review depth for this run.
        type: choice
        default: smoke
        options: [smoke, comprehensive, feature_probe]
      tool_profile:
        description: Choose whether to validate the guided default surface or the expert surface.
        type: choice
        default: guided
        options: [guided, expert]
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
    container: python:3.14
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
      PAPER_CHASER_TOOL_PROFILE: "${{ inputs.tool_profile }}"
      PAPER_CHASER_HIDE_DISABLED_TOOLS: "true"
      PAPER_CHASER_DISABLE_EMBEDDINGS: "true"
      PAPER_CHASER_GUIDED_RESEARCH_LATENCY_PROFILE: "deep"
      PAPER_CHASER_GUIDED_FOLLOW_UP_LATENCY_PROFILE: "deep"
      PAPER_CHASER_GUIDED_ALLOW_PAID_PROVIDERS: "true"
      PAPER_CHASER_GUIDED_ESCALATION_ENABLED: "true"
      PAPER_CHASER_GUIDED_ESCALATION_MAX_PASSES: "2"
      PAPER_CHASER_GUIDED_ESCALATION_ALLOW_PAID_PROVIDERS: "true"
    allowed:
      - research
      - follow_up_research
      - resolve_reference
      - inspect_source
      - get_runtime_status
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
      - search_federal_register
      - get_federal_register_document
      - get_cfr_text
---

# Test Paper Chaser MCP

You are validating the `paper-chaser-mcp` server as a critical end-to-end
reviewer, not just a functional tester. Your primary job is to evaluate
**agent UX quality**: how intuitive the tools feel, how many round trips tasks
require, what features are missing, and where the experience creates
unnecessary friction.

Use the MCP tools first; use bash only when you need short local validation or
to inspect structured output more carefully.

## Agent UX evaluation mandate

For every tool interaction, actively note:

- **Intuitiveness**: Did the tool do what you expected from its name and
  description alone? Would a new agent use it correctly on the first try?
- **Round trips**: How many tool calls did this task require? Could it be done
  in fewer? Flag any workflow that needs 3+ calls where 1-2 should suffice.
- **Missing features**: What did you wish you could do but couldn't? What
  fields or capabilities would make a task obviously easier?
- **Confusing contracts**: Anything where the response shape, field names, or
  error messages were ambiguous, surprising, or inconsistent with what the
  description promised.
- **Dead ends**: Any path that left you without a clear next step or that
  returned data you couldn't act on without undocumented knowledge.

Keep a running tally of friction points throughout the run. These are as
important as functional pass/fail results.

## Run context

- Requested mode: `${{ inputs.mode }}`
- Requested tool profile: `${{ inputs.tool_profile }}`
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

1. Confirm the default guided path is usable without reading the full docs.
2. Confirm guided follow-up and source inspection feel first-class and safe.
3. Confirm reference resolution and runtime-status surfaces are actionable.
4. Confirm regulatory routing prefers trustworthy primary-source behavior or
   safe abstention over plausible garbage.
5. Confirm expert-only tools stay intentional and clearly secondary to the
   guided path.
6. Confirm the guided wrappers surface actionable decision metadata such as
  `executionProvenance`, `sessionResolution`, `sourceResolution`, and
  `abstentionDetails`.
7. Confirm disabled embeddings do not regress the guided contract. Embeddings
  are intentionally off by default in this repository and that is not a defect
  for this workflow.
8. **Critically evaluate agent UX**: identify every friction point,
   unnecessary round trip, missing feature, confusing field name, or dead-end
   response that degrades the experience.

## Mode guide

- `smoke`: run the guided baseline checks below and keep the review focused on
  the primary golden paths.
- `comprehensive`: run the smoke checks, then add deeper expert-surface probes
  when the workflow is running with `tool_profile=expert`.
- `feature_probe`: run a short baseline sanity check first, then spend most of
  the remaining time on the supplied focus prompt and whichever tools best match
  that feature or UX hypothesis.

## Tool profile guide

- `guided`: validate the default 5-tool public surface. Expert tools may be
  hidden, and that is expected behavior.
- `expert`: validate the same guided baseline first, then use the enabled
  visible expert tools where they add meaningful coverage. This workflow keeps
  disabled tools hidden, so expert runs do not necessarily advertise every
  optional tool family.

## Test protocol

1. **Choose the review plan**
   - Determine the effective mode and requested tool profile from the run
     context above. If the mode is blank for any reason, use `smoke`.
   - Call `get_runtime_status()` before deeper probes and record
     `runtimeSummary.effectiveProfile`, `configuredSmartProvider`,
     `activeSmartProvider`, `guidedPolicy`,
     `guidedResearchLatencyProfile`, `guidedFollowUpLatencyProfile`, and any warnings.
   - If the effective profile differs from the requested tool profile, flag it
     as a defect immediately.
   - If a focus prompt is present, note which tools or paths it most likely
     exercises before you start the deeper probes.
   - Always run at least a lightweight baseline so regressions in the core
     workflow are still visible.
   - **Before you begin**: open a running notes section called "UX friction log"
     where you will record every moment the experience felt awkward, slow, or
     required undocumented knowledge.

2. **Guided discovery** (all modes)
   - Call `research(query="graph neural networks", limit=5)`.
   - Record `status`, `searchSessionId`, `summary`, `trustSummary`, the first
  few `sources`, any `unverifiedLeads`, and `executionProvenance`.
   - Verify the response gives a clear next step through `nextActions` or
     `clarification`.
   - **UX check**: Could a low-context agent understand what to do next from the
     response alone, without reading docs?

3. **Guided grounded follow-up** (all modes)
   - Call `follow_up_research(searchSessionId=..., question="What evaluation tradeoffs or benchmark limitations show up here?")`.
   - Record `answerStatus`, the answer body if present, any unsupported asks,
  `sessionResolution`, and `nextActions`.
   - Confirm weak evidence produces an explicit abstention or
     insufficient-evidence response rather than answer-shaped filler.
   - **UX check**: Did this reduce round trips compared with dropping back into
     raw tools, or did it still require too much manual interpretation?

4. **Source-level audit** (all modes)
   - Call `inspect_source(searchSessionId=..., sourceId=...)` using one source
     from the guided result.
   - Record `verificationStatus`, `topicalRelevance`, `canonicalUrl`,
     `retrievedUrl`, `sourceResolution`, and any direct-read recommendations.
   - **UX check**: Is it obvious how to inspect one source before citing it, or
     does the user need hidden knowledge about source ids and provenance fields?

5. **Reference resolution** (all modes)
   - Call `resolve_reference(reference="Rockstrom et al planetary boundaries 2009 Nature 461 472")`.
   - Record `resolutionType`, `status`, whether `bestMatch` is usable, what
     `alternatives` were offered, and whether `nextActions` pointed to the
     correct next tool.
   - If time allows, also try `resolve_reference(reference="Attention Is All You Need")`.
   - **UX check**: Did this remove the need to guess between broad search,
     title match, and citation repair?

6. **Regulatory routing and trust separation** (all modes)
   - Call `research(query="regulatory history of California condor under 50 CFR 17.95", limit=5)`.
   - Confirm the response prefers trustworthy primary-source behavior or safe
     abstention.
   - Unrelated wildlife notices must not appear as verified findings or
     timeline events.
   - If `unverifiedLeads` are present, verify they are clearly separated from
     trusted findings.
   - **UX check**: Does the response make the regulatory confidence state
     obvious, or could a low-context user mistake unverified leads for verified
     evidence?

7. **No-results / abstention behavior** (all modes)
   - Call `research(query="asdkfjhasdkjfh research paper nonsense", limit=3)`.
   - Verify the server responds cleanly with an abstention, partial, or empty
     evidence payload rather than malformed metadata or an unhelpful failure.
   - Record `abstentionDetails` when present and verify the recovery hints are actionable.
   - **UX check**: Does the empty-result payload suggest a clear recovery path,
     or does it just stop?

8. **Comprehensive expert follow-up**
   - If the effective profile is `expert`, also:
     - Call `search_papers_smart(query="graph neural networks", limit=5)`.
     - Call `ask_result_set(searchSessionId=..., question="What benchmarks or evaluation tradeoffs show up here?")`.
     - Call `map_research_landscape(searchSessionId=..., maxThemes=4)`.
     - Call `search_papers_bulk(query="graph neural networks", limit=5)` and
       continue with `cursor=pagination.nextCursor` if a second page exists.
     - Call `search_authors(query="Yoshua Bengio", limit=3)`, then
       `get_author_info`, then `get_author_papers`.
     - Run `search_papers_openalex(query="transformer architecture", limit=3)`
       and then an OpenAlex details, citation, or author follow-up on one
       returned item.
     - For direct regulatory control, use `search_federal_register(...)` and/or
       `get_cfr_text(...)`.
   - If the effective profile is `guided`, note that expert tools are
     intentionally hidden and do not treat their absence as a failure.

9. **Feature-probe follow-up**
   - If the effective mode is `feature_probe`, use the supplied focus prompt to
     choose the most relevant extra tool path.
   - Examples:
     - default discovery or trust quality -> `research`,
       `follow_up_research`, `inspect_source`, `get_runtime_status`
     - expert clustering or graph expansion -> `search_papers_smart`,
       `ask_result_set`, `map_research_landscape`, `expand_research_graph`
       when the profile is `expert`
     - known-item recovery -> `resolve_reference`, then expert
       `resolve_citation` / `search_papers_match` only if the profile is
       `expert`
     - regulatory primary-source workflows -> `research`, then expert
       `search_federal_register` / `get_federal_register_document` /
       `get_cfr_text` when the profile is `expert`
   - Keep the baseline notes, but spend most of the detailed analysis on the
     requested feature or UX hypothesis.

10. **UX friction summary** (all modes)
    - Review your UX friction log and produce a structured summary:
      - **Unnecessary round trips**: list each multi-step workflow that
        required more tool calls than expected. State the minimum you'd expect
        and the actual count.
      - **Missing features**: list capabilities you wished existed. Be
        specific.
      - **Confusing contracts**: list fields, tool names, or response shapes
        that were ambiguous or inconsistent. Explain what you expected versus
        what you got.
      - **Dead ends**: list any point where the response left no clear next
        step.
      - **Positive signals**: briefly note what felt especially smooth or
        well-designed so it is not accidentally degraded in future changes.

11. **Assessment and issue creation**
    - Summarize the effective mode, requested tool profile, any supplied focus
      prompt, the exact tool calls you used, whether abstention felt safe and
      clear, and whether any expert-only gaps stood out.
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
     - Do not file an issue just because embeddings are disabled. That default is
       intentional for this repository unless a workflow run demonstrates a
       concrete guided-contract failure caused by it.
    - When creating a new issue, include the following markers at the top of the
      body so the agentic loop can track convergence (no code block - paste
      these HTML comments directly into the issue body):

      <!-- agent-loop-key: <short-slug-for-the-failure> -->
      <!-- agent-loop-origin: golden-path-smoke -->
      <!-- agent-loop-attempts: 1 -->
      <!-- agent-loop-last-run: <ISO 8601 timestamp of this run> -->

    - When updating an existing issue, increment `agent-loop-attempts` and
      update `agent-loop-last-run`. If `agent-loop-attempts` reaches 3 or more,
      add the label `needs-human` and remove `needs-copilot` in your comment so
      the assigner stops retrying automatically.
