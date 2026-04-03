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
2. Reuse `searchSessionId`; do not restart discovery unless needed.
3. Treat `abstained`, `insufficient_evidence`, and `needs_disambiguation` as
   successful safety behavior.
4. Inspect `sources` and `trustSummary` before presenting claims as settled.
5. Escalate to expert tools only when there is a concrete control need.

## Guided golden paths

### 1. Guided discovery and review (`research`)

1. Start with `research` for discovery, literature review, known-item recovery,
   citation repair, and regulatory history asks.
2. Inspect `status`, `findings`, `sources`, `trustSummary`, `coverage`,
   `failure`, and `nextActions`.
3. Save `searchSessionId` for follow-up steps.

**Example**

```text
research(query="retrieval-augmented generation for coding agents", limit=5)
→ inspect status/findings/sources/trustSummary
→ save searchSessionId
```

**Success signals**

- `status=succeeded` or `partial` with clear evidence semantics.
- Trust fields are usable without reading raw provider payloads.
- `nextActions` points to the next safe step.

### 2. Grounded follow-up (`follow_up_research`)

1. Use `follow_up_research` with `searchSessionId` from `research`.
2. Gate on `answerStatus` before trusting the answer body.
3. If not answered, follow `nextActions` and inspect source-level evidence.

**Example**

```text
follow_up_research(searchSessionId="...", question="What evaluation tradeoffs show up here?")
→ if answerStatus=answered: use answer + evidence
→ if answerStatus=abstained|insufficient_evidence: inspect_source + refine research query
```

**Success signals**

- No answer-shaped filler when evidence is weak.
- Explicit unsupported asks and evidence gaps on abstention paths.

### 3. Known-item and citation cleanup (`resolve_reference`)

1. Use `resolve_reference` when the user already has a citation, DOI, URL,
   arXiv id, title fragment, or regulatory reference.
2. Treat `status` as the decision gate:
   `resolved`, `multiple_candidates`, `no_match`, `regulatory_primary_source`.
3. Use `bestMatch`/`alternatives` and `nextActions` to decide whether to pivot
   back into `research` or into direct expert primary-source tools.

### 4. Source-level audit (`inspect_source`)

1. Use `inspect_source` with `searchSessionId` and `sourceId` from guided
   outputs.
2. Review provenance and trust state before citing.
3. Follow direct-read recommendations for primary sources.

**Success signals**

- A clear source-level reason to trust, hedge, or reject a claim.
- No hidden dependency on external docs to interpret fields.

### 5. Runtime sanity checks (`get_runtime_status`)

1. Use `get_runtime_status` when behavior differs across environments.
2. Check `runtimeSummary.effectiveProfile`, transport, smart provider status,
   provider visibility, and warnings.
3. Use this before troubleshooting with expert diagnostics.

## Regulatory-specific guided behavior

For species/regulatory requests in `research`:

1. Expect primary-source routing before synthesis.
2. Validate `structuredSources` and `regulatoryTimeline` first.
3. If no trustworthy subject anchor exists, expect
   `status=needs_disambiguation` or `abstained`, not fabricated chronology.
4. Unrelated Federal Register hits should not appear as verified findings or
   timeline events.

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

## Future work

- Expand benchmark corpus coverage for guided abstention and clarification
  quality.
- Continue reducing expert-only round trips for common guided workflows.
