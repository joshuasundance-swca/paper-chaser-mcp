# Paper Chaser Golden Paths

This document captures the current default operating model for low-context
agents and users. The public contract is guided-first, trust-graded, and
abstention-safe.

The same paths are exercised by the agentic workflow in
`.github/workflows/test-paper-chaser.md`.

The workflow supports `smoke`, `comprehensive`, and `feature_probe` modes plus
a `tool_profile` input (`guided` by default, `expert` when you intentionally
want raw/provider-specific coverage). After editing the Markdown workflow,
recompile it with:

```bash
gh aw compile test-paper-chaser --dir .github/workflows
```

## Primary personas

- **Low-context researcher**: wants reliable results without learning the full
  provider/tool matrix.
- **Decision-support user**: wants strong evidence and explicit uncertainty
  boundaries.
- **Known-item resolver**: has a DOI/citation/title fragment and wants the
  safest canonical anchor.
- **Regulatory researcher**: needs auditable primary-source history, not
  paper-shaped filler.
- **Operator/power user**: wants expert control over providers, pagination, and
  deeper smart orchestration.

## Default operating rules

1. Start with guided tools.
2. Let guided `research` use its server-owned quality-first policy; low-context clients should not need to choose `fast`, `balanced`, or `deep` for normal use.
3. Reuse `searchSessionId`; do not restart discovery unless needed.
4. Treat `abstained`, `insufficient_evidence`, and `needs_disambiguation` as
   successful safety behavior.
5. Inspect `sources` and `trustSummary` before presenting claims as settled.
6. Escalate to expert tools only when there is a concrete control need.

## Guided golden paths

### 1. Guided discovery and review (`research`)

1. Start with `research` for discovery, literature review, known-item recovery,
   citation repair, and regulatory history asks.
2. Inspect `resultStatus`, `answerability`, `evidence`, `leads`,
  `routingSummary`, `coverageSummary`, `evidenceGaps`, `failureSummary`,
  `executionProvenance`, and `nextActions`.
3. When `routingSummary.querySpecificity=low` or
   `routingSummary.ambiguityLevel=medium|high`, treat
   `routingSummary.retrievalHypotheses` as the server's bounded interpretation
   of the broad ask rather than as final evidence.
3. Save `searchSessionId` for follow-up steps.

**Example**

```text
research(query="retrieval-augmented generation for coding agents", limit=5)
→ inspect resultStatus/answerability/evidence/routingSummary
→ save searchSessionId
```

**Success signals**

- `resultStatus=succeeded` or `partial` with clear evidence semantics.
- Evidence and routing fields are usable without reading raw provider payloads.
- `nextActions` points to the next safe step.

### 2. Grounded follow-up (`follow_up_research`)

1. Use `follow_up_research` with `searchSessionId` from `research`.
2. Gate on `answerStatus` before trusting the answer body.
3. If not answered, follow `nextActions` and inspect source-level evidence.
4. If `searchSessionId` is omitted, expect inference only when exactly one compatible saved session exists.

**Example**

```text
follow_up_research(searchSessionId="...", question="What evaluation tradeoffs show up here?")
→ if answerStatus=answered: use answer + evidence
→ if answerStatus=abstained|insufficient_evidence: inspect_source + refine research query
```

**Success signals**

- No answer-shaped filler when evidence is weak.
- Explicit unsupported asks and evidence gaps on abstention paths.
- Mixed saved sessions can still answer relevance-triage questions by classifying
  saved sources and leads into on-topic evidence, weaker context, and off-topic items.

### 3. Known-item and citation cleanup (`resolve_reference`)

1. Use `resolve_reference` when the user already has a citation, DOI, URL,
   arXiv id, title fragment, or regulatory reference.
2. Treat `status` as the decision gate:
   `resolved`, `multiple_candidates`, `no_match`, `regulatory_primary_source`.
3. Use `bestMatch`/`alternatives` and `nextActions` to decide whether to pivot
   back into `research` with a stronger anchor.
4. Exact DOI, arXiv, and supported paper URLs should resolve as direct anchors before fuzzy recovery.

### 4. Source-level audit (`inspect_source`)

1. Use `inspect_source` with `searchSessionId` and `sourceId` from guided
   outputs.
2. Review provenance and trust state before citing.
3. Follow direct-read recommendations for primary sources.
4. If `searchSessionId` is omitted and more than one compatible session exists, provide it explicitly instead of expecting newest-session rebinding.

**Success signals**

- A clear source-level reason to trust, hedge, or reject a claim.
- No hidden dependency on external docs to interpret fields.

### 5. Runtime sanity checks (`get_runtime_status`)

1. Use `get_runtime_status` when behavior differs across environments.
2. Check `runtimeSummary.effectiveProfile`, transport, smart provider status,
   provider visibility, and warnings.
3. Use this before troubleshooting with expert diagnostics.
4. Interpret `configuredSmartProvider` as the configured bundle and `activeSmartProvider` as the latest effective execution path, including deterministic fallback.

## Regulatory-specific guided behavior

For species/regulatory requests in `research`:

1. Expect primary-source routing before synthesis.
2. Validate `structuredSources` and `regulatoryTimeline` first.
3. If no trustworthy subject anchor exists, expect
   `status=needs_disambiguation` or `abstained`, not fabricated chronology.
4. Unrelated Federal Register hits should not appear as verified findings or
   timeline events.
5. For broad agency-guidance prompts, expect the top guided summary to prefer
  the most relevant query-anchored guidance or policy documents and to retain
  weaker authority records as leads rather than grounded evidence.
6. For hybrid regulatory-plus-literature asks, expect guided `research` to run
   a blended pass when routing stays low-specificity or high-ambiguity. The
   summary and `routingSummary.passModes` should make that explicit.

## Expert fallback paths (intentional)

Use these only under `PAPER_CHASER_TOOL_PROFILE=expert` or in operator flows.

### 1. Expert smart orchestration

Use when you need deeper clustering/graph expansion than guided tools expose.

```text
search_papers_smart(...)
→ ask_result_set(...)
→ map_research_landscape(...)
→ expand_research_graph(...)
```

### 2. Raw/provider-specific retrieval control

Use when you need explicit provider order, pagination semantics, or provider
native payloads.

```text
search_papers(...)
→ search_papers_bulk(...)  # exhaustive traversal
→ provider-specific tools (semantic_scholar/openalex/scholarapi/serpapi/core/arxiv)
```

### 3. Direct regulatory primary-source tooling

Use when legal/audit workflows require direct source control.

```text
search_species_ecos(...) / list_species_documents_ecos(...)
→ search_federal_register(...) / get_federal_register_document(...)
→ get_cfr_text(...)
```

### 4. Repo-local eval bootstrap and workflow handoff

Use this when the task is evaluating planner or workflow behavior in this repo
rather than answering an end-user research question.

1. Start with `scripts/generate_eval_topics.py` for planner-led topic
  generation, taxonomy assignment, ranking, pruning, balancing, and scenario
  emission.
2. Use `scripts/run_eval_autopilot.py` when you want profile-driven generation,
  immutable run bundles, holdout reporting, and guarded workflow handoff.
3. Use `scripts/run_eval_workflow.py` when you intentionally want expert batch
  capture, review or promotion, dataset splitting, and live provider-matrix
  evaluation.
4. For narrow one-seed experiments, prefer the checked-in exploratory profiles
  over ad hoc threshold edits. `single-seed-exploratory-review` keeps review
  gating, `single-seed-exploratory-safe` relaxes safe-policy thresholds, and
  `single-seed-diagnostic-force` is the explicit downstream-debugging path.
5. Prefer single-seed diversification when you need broader one-seed coverage;
  it asks the planner for review, regulatory, and methods-oriented variants
  instead of relying only on looser workflow gating.

## Abstention and clarification policy

- **Never hide abstention**: do not convert abstained outputs into prose
  conclusions.
- **Use concrete narrowing hints**: DOI, exact title, species name, agency,
  year, venue, CFR/FR citation.
- **Prefer `inspect_source` before re-querying** when a source is present but
  trust is uncertain.
- **Escalate to expert tools** only when guided `nextActions` clearly indicate
  a control gap.

## Migration from older smart/raw-first usage

1. Replace default `search_papers_smart` starts with `research`.
2. Replace default `ask_result_set` starts with `follow_up_research`.
3. Replace first-pass `resolve_citation`/`search_papers_match` with
   `resolve_reference`.
4. Keep old tools for expert-only workflows with explicit profile selection.
5. Treat guided `executionProvenance`, `sessionResolution`, and
  `sourceResolution` as part of the public default contract.
6. Do not send `latencyProfile` to guided `research`; the server owns that
  policy and currently defaults to a deep-backed quality-first path.

## Future work

- Expand benchmark corpus coverage for guided abstention and clarification
  quality.
- Continue reducing expert-only round trips for common guided workflows.
- Keep the eval-bootstrap docs, server guidance, and sample autopilot profiles
  synchronized whenever guarded workflow thresholds or single-seed
  diversification behavior changes.
