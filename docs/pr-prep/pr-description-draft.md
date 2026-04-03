# Guided-First Contract Reset, Trust Gating, and Regulatory Routing

> Branch-scoped PR prep material. This file is intended to describe the delta
> from the chosen base branch for the current branch and is not meant to replace the durable
> product documentation in `README.md` and `docs/`.

## Summary

This branch is a large product-level reset of the Paper Chaser MCP surface rather than a narrow feature increment. Compared with the chosen base branch, it moves the project from a broad smart/raw-first default toward a guided-first public contract that is intentionally smaller, safer, and easier for low-context agents to use correctly.

At a high level, this branch:

- makes the default advertised MCP surface a five-tool guided workflow
- preserves the broader raw, smart, provider-specific, and direct regulatory tools behind an explicit expert profile
- upgrades guided responses from trust-labeled output to trust-gated output
- adds a dedicated regulatory investigation path that prefers primary sources over paper-shaped synthesis
- realigns docs, packaging manifests, prompts, workflow harnesses, and tests around that contract

Substantive diff against the chosen base branch at the time of this draft
(excluding `docs/pr-prep` support artifacts to avoid recursive noise):

- 47 files changed
- 6,026 insertions
- 2,279 deletions
- 4 branch-only commits

This PR should be reviewed as a contract migration with implementation, documentation, workflow, and regression-harness consequences, not as a routine refactor.

## Why This Change Exists

The throughline across the code, docs, tests, and workflow assets is that the old default surface was too easy for low-context agents to misuse. There were too many starting points, too much provider-level nuance exposed too early, and not enough hard boundaries between trustworthy evidence and plausible filler.

This branch responds to that by optimizing for four product goals:

1. Make the obvious first move safe.
2. Make weak evidence produce abstention or clarification instead of answer-shaped prose.
3. Separate trusted findings from weak or off-topic leads.
4. Give operators access to expert depth without making that complexity the default onboarding path.

The result is a more opinionated product surface: fewer default tools, stronger response semantics, more explicit runtime truth, and better alignment between what the docs promise and what the server actually advertises.

## What Changed

### 1. The public MCP contract is now guided-first

The most important breaking change is the shift in default tool advertisement. In guided mode, the server now exposes only:

- `research`
- `follow_up_research`
- `resolve_reference`
- `inspect_source`
- `get_runtime_status`

This is not just documentation. The current server startup path registers only the visible tools for the active profile, and the tool-definition helper now emits profile-filtered schemas rather than an always-full surface.

That means this branch changes both product messaging and actual `list_tools` behavior. Clients that previously assumed `search_papers`, `search_papers_smart`, `resolve_citation`, or diagnostics tools would always be advertised by default will need to migrate.

The expert profile remains available, but it is repositioned as an intentional operator fallback rather than the public contract of record.

### 2. Guided workflows are implemented as productized wrappers

This branch does not throw away the existing expert/runtime substrate. Instead, it adds guided wrappers that sit over the raw and smart primitives and present a smaller, more constrained contract:

- `research` becomes the default entry point for discovery, literature review, known-item recovery, citation repair, and regulatory routing.
- `follow_up_research` becomes the grounded question-answering path over an existing `searchSessionId`.
- `resolve_reference` generalizes known-item cleanup beyond citations alone.
- `inspect_source` exposes source-level provenance and trust details.
- `get_runtime_status` gives a curated runtime summary without forcing clients into low-level provider diagnostics.

Architecturally, this is a pragmatic choice. It avoids duplicating the entire retrieval/orchestration stack while still letting the public contract diverge meaningfully from the expert internals.

### 3. Trust gating is now enforced, not just described

The branch strengthens the evidence model so that guided outputs distinguish between:

- verified, on-topic findings that can be promoted into trusted summaries
- weak, filtered, or off-topic material that should remain visible only as leads
- evidence gaps and failure summaries that define the limits of the answer

This is the substantive product shift behind safe abstention. Guided outputs now carry explicit fields such as:

- `verifiedFindings`
- `unverifiedLeads`
- `trustSummary`
- `coverage`
- `failureSummary`
- `nextActions`
- `clarification`

Similarly, grounded follow-up now has explicit answer gating through `answerStatus=answered|abstained|insufficient_evidence`, and `answer` can be `null` when the evidence does not justify a response.

This is a deliberate move away from answer-shaped filler and toward outputs that are auditable even when they are incomplete.

### 4. Regulatory asks now route through a dedicated primary-source workflow

One of the most consequential behavior changes is in the smart/orchestration layer: regulatory and species-history queries are no longer treated as ordinary paper search.

Instead, the planner/runtime can detect regulatory intent and pivot into a path that prioritizes primary sources such as:

- ECOS
- Federal Register
- GovInfo / current CFR text

The intended output is not simply a ranked list of papers. It is a subject-anchored primary-source trail that can include structured source records, regulatory timelines, evidence gaps, and explicit fallback behavior when current codified text cannot be verified.

This is a significant product broadening. Paper Chaser becomes less of a pure scholarly search broker and more of a guided research router that can choose the right evidence class for the user’s question.

### 5. Retrieval results now carry stronger trust and coverage metadata

The retrieval and broker layers are upgraded so that guided and smart flows can reason over normalized provider metadata instead of inferring everything downstream from loosely shaped records.

Across provider outputs, the branch introduces or standardizes metadata such as:

- source type
- verification status
- access status
- canonical and retrieved URLs
- open-access route
- confidence and relevance-related indicators
- coverage summaries
- failure summaries

This matters because the guided layer now depends on this metadata to decide whether a record should be trusted, surfaced only as a lead, or excluded from verified findings altogether.

The broker also becomes more honest about partiality. Single-page retrieval can now report explicit coverage and failure states rather than silently implying completeness.

### 6. Docs, packaging, prompts, and workflows were all reset to match the contract

This branch does not leave the contract change implicit in code. It rewrites the surrounding surfaces so they tell the same story:

- `README.md` now teaches guided-first usage
- `docs/golden-paths.md` codifies the new default workflow hierarchy
- `docs/guided-reset-migration-note.md` makes the breaking change explicit
- `docs/agent-handoff.md` is rewritten around guided-first status and expectations
- workflow guidance and lockfiles are updated to validate the new profile split
- packaging/manifests now reflect the guided versus expert story more intentionally

This documentation pass also tightens a few contract details that were easy to
blur in prose:

- guided workflow instructions now refer to the public `unverifiedLeads` field
	instead of the internal `candidateLeads` terminology
- expert advertisement language now makes it clear that visible tools still
	depend on provider availability and `PAPER_CHASER_HIDE_DISABLED_TOOLS`
- the checked-in Microsoft plugin sample is guided-first by default, with the
	expert package called out as an intentional operator switch

This matters for maintainability. The branch treats docs and workflow harnesses as part of the product contract rather than as passive documentation.

### 7. Tests and fixtures now define product behavior more explicitly

The changed test suite is not just catching regressions in code paths. It is acting as a specification for the intended product behavior.

The updated tests and fixtures assert that:

- guided mode advertises only the five public tools
- trust gating keeps off-topic material out of verified findings
- grounded follow-up abstains when evidence is insufficient
- regulatory workflows build subject-anchored primary-source outputs
- unrelated regulatory materials do not quietly re-enter trusted summaries
- runtime truth remains internally consistent
- benchmark and UX corpora reflect the guided-first experience

That gives the branch a stronger acceptance harness than before, but it also means future surface changes will have to keep code, docs, and tests synchronized.

## Key Architectural Decisions

### Guided and expert are now explicit product tiers

This branch establishes a durable two-tier model:

- `guided` is the default public contract for low-context users and agents.
- `expert` remains available for power users who need raw retrieval, provider-specific control, or deeper smart tooling.

That is a deeper change than renaming tools. It creates an architectural obligation to decide, for every future feature, whether it belongs in guided mode, expert mode, or both.

### Public contract metadata moves into the tool-spec layer

Tool visibility, description, and profile membership are centralized through the tool-spec registry instead of being spread informally across docs and server code.

This is a good long-term move because it creates one place to reason about:

- which tools belong to which profile
- how tools should be described publicly
- how result behavior should be presented to agents

### Structured models are becoming the shared vocabulary of the product

The branch expands the shared model layer around:

- source provenance
- citation records
- coverage summaries
- failure summaries
- runtime summaries
- regulatory timeline support

This is important because it lays groundwork for more durable downstream uses, especially export, source inspection, and session-based follow-up.

### Regulatory routing is treated as a first-class mode, not a special case

Instead of forcing regulatory/history requests through paper search and cleaning up the result afterward, the orchestration layer now has a separate path for that class of work.

That is the right architectural direction if the goal is trustworthy outputs, but it increases the system’s scope and makes intent classification more important.

## Risks And Migration Concerns

### Breaking change: default tool discovery

This is the main PR risk. Any client that depends on default discovery of old tools such as `search_papers`, `search_papers_smart`, `resolve_citation`, or `get_provider_diagnostics` may regress under guided mode.

Even if the underlying expert behavior still exists, the discovery contract has changed. That needs to be called out clearly in the PR and in any release notes.

### Wrapper drift risk

The guided tools are wrappers over legacy expert/raw/smart internals. That keeps the implementation incremental, but it also creates the possibility that wrapper semantics drift away from underlying behavior over time.

Areas to watch:

- `research` intent detection and wrapper-side trust shaping
- `resolve_reference` status mapping
- `inspect_source` dependence on stored result-set schema
- runtime summaries diverging from deeper diagnostics

### Heuristic routing and threshold brittleness

The regulatory path and trust gating both rely on heuristics and thresholds.

Potential failure modes:

- legitimate mixed academic/regulatory queries getting over-routed into regulatory mode
- sparse but relevant evidence being treated as too weak and causing over-abstention
- subject-matching rules excluding valid regulatory materials that use alternate naming or indirect language

### Manifest/runtime mismatch risk

Because this PR changes code, manifests, workflow assets, and docs together, there is long-term risk that one of these surfaces drifts later.

This branch improves alignment, but it also raises the maintenance bar: future tool-surface changes need synchronized updates across the implementation, manifests, docs, and regression harnesses.

### Diagnostic visibility tradeoff

Hiding disabled tools and defaulting to guided mode is better for low-context UX, but it can make diagnosis harder for operators who do not realize a tool is missing because of profile or provider settings.

That tradeoff appears intentional, but it should be acknowledged in review.

## Long-Term Impacts

### The repo now has a clearer public contract of record

This branch makes it much easier to answer, "What is the default Paper Chaser experience?" That should help low-context agents, documentation quality, and future review discipline.

### The project is moving from tool collection to product surface

The branch treats tool selection, trust semantics, and runtime truth as user experience design problems, not just implementation details. That is a meaningful maturation of the project.

### Session-based workflows become more central

`searchSessionId` is increasingly the backbone of follow-up behavior, source inspection, and planned export functionality. That makes session schema stability more important over time.

### Future provider work will be judged by trust semantics, not just reach

New providers will need to normalize enough metadata to participate in guided trust gating. Returning records alone will no longer be enough.

### The dual-surface maintenance cost is now real

Guided and expert profiles are both meaningful. That is valuable, but it creates a durable maintenance burden around docs, packaging, tests, and runtime behavior.

## Reviewer Guide

Reviewers should evaluate this PR as a coordinated product migration across several layers.

Suggested review order:

1. Public contract and tool visibility
2. Guided wrapper semantics and runtime status behavior
3. Smart-layer trust gating and regulatory routing
4. Shared models and retrieval metadata
5. Docs, manifests, workflow assets, and migration notes
6. Tests and fixtures that lock in the new behavior

Questions reviewers should answer:

- Does guided mode advertise exactly the surface we want as the default public contract?
- Are the abstention and trust boundaries strict enough to prevent answer-shaped filler?
- Does regulatory routing improve correctness without over-capturing mixed queries?
- Are runtime and diagnostic summaries internally consistent?
- Are the docs and packaging assets aligned with the actual server behavior?
- Do the tests meaningfully enforce the intended migration rather than merely snapshotting current output?

## Validation And Testing

The code changes appear to be supported by a broad test realignment, especially in:

- guided wrapper behavior
- server tool advertisement
- regulatory routing and trust gating
- benchmark and UX fixtures
- workflow synchronization

This PR-prep pass did not run the full validation suite. Before opening the PR, we should still run at least the focused contract tests and ideally the full expected validation stack for the repository.

## Proposed PR Description

### What this PR does

This PR resets Paper Chaser’s default MCP experience around a guided-first public contract. Instead of advertising the broad raw/smart/provider-specific surface by default, guided mode now exposes five tools: `research`, `follow_up_research`, `resolve_reference`, `inspect_source`, and `get_runtime_status`.

Under the hood, the branch productizes those flows as wrappers over the existing retrieval and smart-runtime substrate, while tightening response semantics around trust gating, abstention, source inspection, runtime truth, and regulatory routing. Regulatory asks can now route into a dedicated primary-source path built around ECOS, Federal Register, and GovInfo instead of being treated as ordinary paper search.

The branch also aligns the surrounding repo surfaces with that contract: docs, prompts, manifests, GitHub workflow assets, and regression tests are all updated to reflect guided-first onboarding with expert-mode fallback.

### Why this change

The previous default surface was too easy for low-context agents to misuse. There were too many ways to start, too much provider nuance exposed too early, and not enough hard separation between verified findings and plausible but weak evidence.

This PR makes the safe path more obvious and more enforceable:

- guided discovery starts from `research`
- grounded follow-up is explicit and session-based
- weak evidence can abstain instead of producing answer-shaped filler
- off-topic or filtered material stays in lead buckets instead of trusted summaries
- runtime/profile truth is directly inspectable
- expert depth remains available, but only when intentionally selected

### Main changes

- switch default MCP advertisement to the five guided tools
- add profile-aware tool visibility and profile-filtered tool definitions
- implement guided wrappers for research, follow-up, reference resolution, source inspection, and runtime status
- strengthen trust-gated outputs with `verifiedFindings`, `unverifiedLeads`, `trustSummary`, coverage/failure metadata, and explicit next actions
- add answer gating for follow-up via `answerStatus=answered|abstained|insufficient_evidence`
- add a dedicated regulatory primary-source workflow and timeline-oriented output behavior
- normalize retrieval metadata so provider results can participate in trust gating consistently
- update docs, manifests, workflow assets, and tests to codify the guided-versus-expert split

### Risks

- breaking default `list_tools` discovery for clients that expect old smart/raw tools
- wrapper drift between guided contracts and expert internals over time
- heuristic routing or threshold tuning causing over-abstention or over-routing
- future manifest/docs/workflow drift if the contract changes again without synchronized updates

### Follow-up questions for review

- Is the guided five-tool surface exactly the right default contract?
- Are the regulatory routing heuristics appropriately strict?
- Is the current expert-mode story stable enough for operator use without being over-promoted?
- Do we want any additional migration guidance for clients that start from `search_papers_smart` or `search_papers` today?
