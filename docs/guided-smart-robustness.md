# Guided And Smart Robustness Notes

This document captures the server-side behaviors added to make guided and smart workflows more tolerant of imperfect client inputs and more explicit about internal recovery.

## Goals

- Recover internally before abstaining.
- Avoid assuming perfect `searchSessionId` and `sourceId` handling by clients.
- Expose machine-readable routing, recovery, and result-state metadata.
- Keep deterministic heuristics as guardrails rather than dominant routing logic.

## Guided Dispatch

The guided wrappers in `paper_chaser_mcp/dispatch.py` now normalize and repair incoming arguments before validation.

- `research` normalizes whitespace, strips common wrapper phrases, and canonicalizes citation-like surfaces where it is safe.
- `follow_up_research` and `inspect_source` normalize alternate session/source field names and can recover only when one compatible saved session is uniquely identifiable.
- Guided responses now include:
  - `resultState`
  - `inputNormalization`
  - `machineFailure` when the smart runtime returns an invalid payload or raises unexpectedly

### Source Resolution

`inspect_source` accepts more than exact `sourceId` matches.

- exact `sourceId`
- case-folded exact matches
- canonical URL / citation text matches
- session-local index aliases such as `source-1`
- unique partial title matches

## Smart Metadata

`paper_chaser_mcp/agentic/models.py` and `paper_chaser_mcp/agentic/planner.py` now support richer planning metadata.

- `intentCandidates`
- `secondaryIntents`
- `routingConfidence`
- `intentRationale`
- `recoveryAttempted`
- `recoveryPath`
- `recoveryReason`
- `stoppedRecoveryBecause`
- `anchorType`
- `anchorStrength`
- `anchoredSubject`
- `normalizationWarnings`
- `repairedInputs`

This metadata is intended for downstream clients and debugging tools. It should be treated as additive contract, not a replacement for the main result payload.

## Recovery Semantics

Smart search recovery is explicit rather than implicit.

- empty regulatory routes can recover into semantic known-item resolution
- empty regulatory routes can recover into literature review when the query supports it
- known-item resolution can widen into broader candidate retrieval instead of hard failing

Each recovery path should annotate strategy metadata so clients can tell:

- what route won
- whether fallback was used
- why fallback happened
- which anchor the server trusted most

## Session Registry

`paper_chaser_mcp/agentic/workspace.py` remains the source of truth for saved result sets.

- sessions are still TTL-bound
- saved records still carry payload, metadata, indexed papers, authors, and trace events
- the registry now exposes active records ordered by recency so dispatch can make safe session-inference decisions

## Testing Expectations

Focused regressions should cover:

- optional `searchSessionId` for guided follow-up and source inspection
- source alias resolution
- input normalization metadata
- smart recovery provenance and anchor metadata
- mixed and degraded flows that still return actionable result state
- saved-session relevance triage over mixed evidence and leads
- recommendation-first summaries that still preserve the full audit payload

## Phase 4 trust and grounding additions

The `llm-guidance` phase-4 wave added several LLM-first robustness behaviors
layered on top of the guided dispatch path. Agents should treat them as
additive signals with deterministic fallbacks.

### Weak-match rationale path

`inspect_source` now emits a one-sentence `whyClassifiedAsWeakMatch` rationale
whenever a source landed in `trustSummary.authoritativeButWeak` or was filtered
out of grounded evidence. The rationale is composed deterministically at
inspect time from the structured classification signals that were already
attached to the source upstream — in priority order: `classificationRationale`,
`whyClassifiedAsWeakMatch`, `whyWeak`, `note`, `whyNotVerified`, and up to two
`subjectChainGaps` pulled from the session's `strategyMetadata`. Fragments are
de-duplicated, joined into a single <=200-char sentence, and truncated with an
ellipsis if necessary. This is an intentional design choice: the upstream
classifier (LLM-first during ranking) is where the semantic judgement happens,
so per-source LLM calls at inspect time would inflate token budgets without
adding signal the classifier has not already produced. Clients should cite the
composed sentence when explaining why an authoritative source was not promoted
into the answer.

### Species-dossier on_topic demotion

For regulatory intents resolved to `species_dossier`, the relevance fallback
demotes candidates from `on_topic` to `weak_match` when the pool contains no
species-specific evidence for the grounded subject. The demotion is reported
through `confidenceSignals.trustRevisionNarrative` and surfaces the gap as
`subjectChainGaps=["missing_species_specific_evidence"]`. This prevents
adjacent agency notices from becoming grounded dossier evidence for the wrong
species.

### Known-item weak-signal breadth preservation

The known-item gate now refuses to force a title-like query into known-item
resolution when the query carries no DOI, arXiv id, or URL anchor and the
planner reports high ambiguity. Such queries stay on the broader retrieval
path so weak but genuine signals are preserved instead of being collapsed into
a fabricated exact match. `knownItemResolutionState=needs_disambiguation` is
the expected surface for these cases.

### `hybrid_policy_science` retrieval hypothesis

Planner routing can now emit a `hybrid_policy_science` retrieval hypothesis
alongside the existing regulatory and literature hypotheses. It signals that
both a regulatory primary-source pass and a literature pass should run and
their evidence should be merged under a shared subject card. `searchStrategy`
exposes this hypothesis so clients can tell when a response blends both
streams; it typically co-occurs with
`searchStrategy.regulatoryIntent="hybrid_regulatory_plus_literature"`.

### Known gaps (honest tech debt)

The following behaviors are *not* implemented today. They are tracked here so
future iterations can address them rather than having docs quietly overclaim.

- **Subject-card session-level cache.** `resolve_subject_card` in
  `paper_chaser_mcp/agentic/subject_grounding.py` re-derives the card each time
  it is called. The resolver is pure and cheap (it reshapes planner outputs or
  falls back to the deterministic `_infer_entity_card` extractor), but there
  is no explicit per-session cache. The planner does populate
  `PlannerDecision.subject_card` once and downstream call sites read from that
  field, which functions as an implicit single-resolution-per-request memo —
  but it is not a cache across sessions, tools, or follow-up calls.
- **Per-source LLM-composed weak-match rationale.** `whyClassifiedAsWeakMatch`
  at `inspect_source` time is composed deterministically (see the weak-match
  rationale section above). A future iteration could add an optional
  LLM-rewrite pass guarded by a budget check, but that is not shipped.
