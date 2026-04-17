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
- `coverageSummary.providersSucceeded` reflects only providers that returned
  results (zero-result providers are excluded).
- `failureSummary.fallbackAttempted` is `true` whenever a fallback mode was
  applied, and `outcome` is `partial_success` (not `no_failure`) when the
  response abstained with zero sources.

#### How to read the new trust + grounding signals

Guided `research` and `follow_up_research` responses now expose additive trust
and grounding cues. Treat them as hints layered on top of `answerability` and
`resultStatus`, not as replacements.

- `confidenceSignals.evidenceQualityProfile` buckets the supporting pool as
  `strong` (multiple directly responsive sources), `mixed` (some responsive,
  some weak), `weak` (mostly filtered or adjacent), or `authoritative_but_weak`
  (primary-source authority without topical fit). Use `strong`/`mixed` to
  justify confident synthesis; treat `weak` and `authoritative_but_weak` as
  reasons to qualify claims or route to `inspect_source`.
- `confidenceSignals.synthesisMode` describes how the answer was assembled
  (for example `grounded_synthesis`, `evidence_triage`, `session_introspection`,
  `deterministic_salvage`). Only `grounded_synthesis` should be treated as a
  normal answer; the others require caution.
- `confidenceSignals.evidenceProfileDetail` gives a short rationale behind the
  profile bucket, and `confidenceSignals.synthesisPath` records which synthesis
  branch ran. `confidenceSignals.trustRevisionNarrative` explains any trust
  re-grade that happened after initial ranking (for example when a highly
  ranked record was demoted because the subject chain did not match).
- `trustSummary.authoritativeButWeak` collects primary-source records that the
  server judged authoritative (agency rule, CFR text, species listing, etc.)
  but that did not topically respond to the query. Cite them only after
  confirming responsiveness with `inspect_source`; do not promote them into
  grounded evidence.
- `searchStrategy.regulatoryIntent` (when present) is one of
  `current_cfr_text`, `rulemaking_history`, `species_dossier`,
  `guidance_lookup`, or `hybrid_regulatory_plus_literature`. Use it to decide
  whether to drive with CFR text, Federal Register history, species profiles,
  agency guidance, or a blended regulatory-plus-literature pass.
- `searchStrategy.subjectCard` is the subject-grounding card used for species
  and regulatory workflows. It is populated LLM-first from planner outputs
  (`entity_card`, `candidate_concepts`, `regulatory_intent`) with a
  deterministic fallback when no LLM bundle is available. Fields:
  `commonName`, `scientificName`, `agency`, `requestedDocumentFamily`,
  `subjectTerms` (up to six candidate concepts), `confidence`
  (`high` / `medium` / `low` / `deterministic_fallback`), and `source`
  (`planner_llm` / `deterministic_fallback` / `hybrid`). Use it to confirm the
  server grounded the right entity before trusting species dossiers or
  agency-specific summaries. Alias lists, taxonomy trees, and pre-resolved
  primary-source anchors are *not* on the card today — see the "Known gaps"
  subsection of `docs/guided-smart-robustness.md`.
- `searchStrategy.subjectChainGaps` lists missing links in the subject chain
  (for example `missing_species_specific_evidence`,
  `missing_rulemaking_history`, `missing_guidance_anchor`). Treat each gap as
  an explicit evidence boundary and reflect it in any response to the user.
- `searchStrategy.intentFamily` records the detected intent family (for
  example `literature_review`, `known_item`, `heritage_cultural_resources`,
  `regulatory_timeline`). Use it to validate that the server interpreted the
  ask the same way you did.

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
- `answerStatus` is now validated against both deterministic refusal patterns
  and optional LLM-based validation when a provider bundle is available.
- `canAnswerFollowUp` reflects capability (`has_sources AND has_session`), not
  just permission.

### 3. Known-item and citation cleanup (`resolve_reference`)

1. Use `resolve_reference` when the user already has a citation, DOI, URL,
   arXiv id, title fragment, or regulatory reference.
2. Treat `status` as the decision gate. The full set is:
   `resolved` (one confident match), `multiple_candidates` (several plausible
   matches — pick from `bestMatch`/`alternatives`), `needs_disambiguation`
   (best match's key metadata — author/year/venue — conflicts and is unsafe to
   cite directly), `no_match`, and `regulatory_primary_source`.
3. Use `bestMatch`/`alternatives` and `nextActions` to decide whether to pivot
   back into `research` with a stronger anchor.
4. Exact DOI, arXiv, and supported paper URLs should resolve as direct anchors before fuzzy recovery.

### 4. Source-level audit (`inspect_source`)

1. Use `inspect_source` with `searchSessionId` and `sourceId` from guided
   outputs.
2. Review provenance and trust state before citing.
3. Use `whyClassifiedAsWeakMatch` plus `confidenceSignals.sourceScopeLabel` /
  `confidenceSignals.sourceScopeReason` to distinguish authoritative but scope-limited records from directly responsive ones.
4. Follow direct-read recommendations for primary sources. When
  `directReadRecommendationDetails` is present, each entry carries
  `{trustLevel, whyRecommended, cautions}`; prefer higher `trustLevel` entries
  first and honor the `cautions` when summarizing the source.
5. If `searchSessionId` is omitted and more than one compatible session exists, provide it explicitly instead of expecting newest-session rebinding.

**Success signals**

- A clear source-level reason to trust, hedge, or reject a claim.
- No hidden dependency on external docs to interpret fields.

### 5. Runtime sanity checks (`get_runtime_status`)

1. Use `get_runtime_status` when behavior differs across environments.
2. Check `runtimeSummary.effectiveProfile`, transport, smart provider status,
   provider visibility, and warnings.
3. Use this before troubleshooting with expert diagnostics.
4. Interpret `configuredSmartProvider` as the configured bundle and `activeSmartProvider` as the latest effective execution path, including deterministic fallback.
5. Prefer the consolidated structured health fields over string-parsing:
   - `runtimeSummary.healthStatus` is one of
     `nominal | degraded | fallback_active | critical`.
     - `nominal`: configured smart provider healthy and no optional providers disabled.
     - `degraded`: smart provider healthy, but one or more optional providers are disabled or suppressed.
     - `fallback_active`: configured smart provider is unavailable and deterministic fallback is in use.
     - `critical`: the smart layer failed to initialize and no usable smart fallback is present.
   - `runtimeSummary.fallbackActive` is a boolean mirror of the fallback state; when true,
     `runtimeSummary.fallbackReason` carries a human-readable explanation (omitted otherwise).
   - `runtimeSummary.structuredWarnings` is a list of
     `{code, severity, message, subject}` entries. `severity` is one of
     `info | warning | critical`. Known `code` values include
     `smart_provider_fallback`, `provider_disabled`, `chat_only_smart_provider`,
     `tools_hidden`, `stdio_transport`, `ecos_tls_disabled`,
     `guided_hides_expert`, and `narrow_provider_order`.
   - The legacy `runtimeSummary.warnings: list[str]` (also mirrored as the
     top-level `warnings` field on the tool response) is preserved for
     backward compatibility and stays byte-identical to the strings emitted in
     `structuredWarnings[*].message`.

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
7. Papers without DOI or other identifiers now default to `unverified`
   verification status (previously `verified_metadata`). This tightens
   integrity for regulatory workflows where identifier coverage is sparse.
8. Regulatory source-type recognition is expanded (13 types, up from 4),
   covering rules, proposed rules, notices, executive orders, presidential
   documents, and guidance documents.

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
- Monitor citation-repair behavior after the tightened year-mismatch penalty
  and reduced upstream-confidence bonus to catch regressions in near-miss
  resolution scenarios.
- Continue with cross-domain remediation plan workstreams (reranking,
  relevance-classification resilience, cultural-resource routing) now that the
  schema and runtime foundations from the stress-test remediation are in place.
