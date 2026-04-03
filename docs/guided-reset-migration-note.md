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
- Use `inspect_source` for provenance and direct-read follow-through.
- Use `resolve_reference` for citation/identifier normalization.
- Treat `abstained` and `needs_disambiguation` as intended safe outcomes.

### 2) Opt Into Expert Surface

- Set `PAPER_CHASER_TOOL_PROFILE=expert`.
- Leave `PAPER_CHASER_HIDE_DISABLED_TOOLS=false` if you want expert
  `list_tools` output to advertise the broadest visible expert surface.
- Use expert packaging metadata when publishing a full tool set.
- Keep guided tools as the default decision path in agent prompts even in expert
  environments unless there is a concrete expert-only requirement.

## Contract Changes To Handle

- Guided `research` returns explicit trust and control fields:
  - `status`: `succeeded|partial|needs_disambiguation|abstained|failed`
  - `verifiedFindings`, `sources`, `unverifiedLeads`, `evidenceGaps`,
    `trustSummary`, `failureSummary`, `resultMeaning`, `nextActions`
- Guided follow-up returns explicit answer gating:
  - `answerStatus`: `answered|abstained|insufficient_evidence`
  - `answer` is `null` when not safely answerable.
- Runtime truth is now expected to include:
  - `effectiveProfile`
  - `configuredSmartProvider`
  - `activeSmartProvider`
  - internally consistent active/disabled provider sets.

## Release-Readiness Checks

- Validate both profile contracts in CI:
  - guided: only guided tools are advertised
  - expert: the expert-visible surface is advertised; under default expert
    settings with disabled-tool hiding off, that is typically the broadest
    expert surface
- Run fixture-based acceptance checks for:
  - low-context guided success
  - safe abstention over plausible garbage
  - regulatory subject correctness
  - runtime-summary truth consistency
