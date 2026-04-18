# Guided Reset Migration Note

This note describes the breaking default-surface change introduced by the
guided reset.

## What Changed

- The default public tool profile is now `guided`.
- Default tool discovery (`list_tools`) now advertises only:
  - `research`
  - `follow_up_research`
  - `resolve_reference`
  - `inspect_source`
  - `get_runtime_status`
- Raw/provider-specific tools remain available through the expert profile.
- Guided `research` no longer honors client `latencyProfile`; the server owns
  that policy and currently applies a deep-backed quality-first path.
- Expert smart tools now default to `latencyProfile=deep`; `balanced` is the
  lower-latency fallback and `fast` is for smoke or debug use only.

## Why This Is Breaking

Clients that previously assumed smart/raw tools were always advertised (for
example `search_papers_smart`, `search_papers`, `resolve_citation`,
`get_provider_diagnostics`) may now fail capability checks in default
environments even though the server still supports those tools in expert mode.

## Migration Paths

### 1) Stay On The Default Guided Profile (recommended)

- Keep `PAPER_CHASER_TOOL_PROFILE=guided` (or omit it).
- Update clients to start from `research`.
- Reuse `searchSessionId` with `follow_up_research`.
- If `searchSessionId` is omitted, guided follow-up and source inspection infer it only when one compatible saved session exists.
- Use `inspect_source` for provenance and direct-read follow-through.
- Use `resolve_reference` for citation/identifier normalization.
- Expect exact DOI, arXiv, and supported paper URLs to resolve before fuzzy citation repair.
- Treat `abstained` and `needs_disambiguation` as intended safe outcomes.
- Expect guided responses to surface `executionProvenance`, and expect
  ambiguous session/source flows to return `sessionResolution` and
  `sourceResolution` instead of raw exceptions.

### 2) Opt Into Expert Surface

- Set `PAPER_CHASER_TOOL_PROFILE=expert`.
- Leave `PAPER_CHASER_HIDE_DISABLED_TOOLS=false` if you want expert
  `list_tools` output to advertise the broadest visible expert surface.
- Use expert packaging metadata when publishing a full tool set.
- Keep guided tools as the default decision path in agent prompts even in expert
  environments unless there is a concrete expert-only requirement.

## Contract Changes To Handle

- Guided `research` returns explicit trust and control fields:
  - `resultStatus`: `succeeded|partial|needs_disambiguation|abstained|failed`
  - `answerability`, `routingSummary`, `coverageSummary`, `evidence`, `leads`,
    `evidenceGaps`, `failureSummary`, `nextActions`, `executionProvenance`
  - `summary` now leads with a short recommendation-first statement when the
    server has a clear top result.
- Guided follow-up returns explicit answer gating:
  - `answerStatus`: `answered|abstained|insufficient_evidence`
  - `answer` is `null` when not safely answerable.
  - Grounded `answered` now requires a non-deterministic synthesis provider
    plus at least one on-topic, verified source with qa-readable text. Cases
    that used to return filler now return `insufficient_evidence`.
  - Follow-up responses are **compact by default**. `sources` are collapsed to
    `selectedEvidenceIds`/`selectedLeadIds`, and legacy `verifiedFindings` /
    `unverifiedLeads` are omitted. Pass `responseMode="standard"` or
    `responseMode="debug"` for richer payloads, or `includeLegacyFields=true`
    to restore the legacy compatibility views.
  - Comparative / selection asks expose a structured `topRecommendation` with
    `sourceId`, `recommendationReason`, and `comparativeAxis`.
  - ambiguity and reuse state are surfaced through `sessionResolution`
  - saved-session introspection can classify mixed source sets into on-topic,
    weaker, and off-target groups when the stored metadata is sufficient
- Guided source inspection now splits access state into `fullTextUrlFound`
  (URL discovered), `bodyTextEmbedded` (body text indexed into the saved
  session), and `qaReadableText` (body actually available to the current
  synthesis call). `AccessStatus` values now include `url_verified`,
  `body_text_embedded`, and `qa_readable_text` alongside prior states.
- Guided source inspection now returns structured `sourceResolution` details on
  ambiguity instead of failing with a raw `ValueError`.
- Runtime truth is now expected to include:
  - `effectiveProfile`
  - `configuredSmartProvider`
  - `activeSmartProvider`
  - `guidedPolicy`
  - `guidedResearchLatencyProfile`
  - `guidedFollowUpLatencyProfile`
  - where `activeSmartProvider` means the latest effective execution path, including deterministic fallback
  - internally consistent active/disabled provider sets.

## Release-Readiness Checks

- Validate both profile contracts in CI:
  - guided: only guided tools are advertised
  - expert: the intended expert-visible validation subset is advertised and
    behaves consistently with the expert docs for that workflow configuration
- Run fixture-based acceptance checks for:
  - low-context guided success
  - safe abstention over plausible garbage
  - regulatory subject correctness
  - runtime-summary truth consistency
