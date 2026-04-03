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
