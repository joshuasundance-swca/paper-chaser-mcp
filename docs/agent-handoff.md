# Agent Handoff

This document is the current working handoff for `paper-chaser-mcp`. It is
meant to give any follow-on agent enough context to validate the repo,
understand the current product contract, and continue from the highest-value
next steps without re-discovering project state.

## Current Status

- The shipped public/default surface is **guided-first**. The default
  `PAPER_CHASER_TOOL_PROFILE=guided` contract exposes exactly 5 tools:
  `research`, `follow_up_research`, `resolve_reference`, `inspect_source`, and
  `get_runtime_status`.
- The **expert** surface remains available behind
  `PAPER_CHASER_TOOL_PROFILE=expert`. That profile exposes the guided tools plus
  raw/provider-specific families, smart graph tools, and direct regulatory
  primary-source tools.
- Guided responses are now **trust-gated**, not just trust-labeled. Verified
  findings must be on-topic and backed by verified evidence. Weak, filtered, or
  off-topic items belong in `unverifiedLeads`, not `verifiedFindings`.
- Guided follow-up is now **abstention-safe**. `follow_up_research` returns
  `answerStatus=answered|abstained|insufficient_evidence` and should not emit
  answer-shaped filler when evidence is weak.
- Regulatory routing is now **subject-anchored**. `research` should either
  build a trustworthy primary-source trail or return
  `needs_disambiguation` / `abstained`; unrelated wildlife notices should not
  appear as verified findings or timeline events.
- Runtime reporting is now **internally truthful**. `get_runtime_status` and
  expert diagnostics should agree on `effectiveProfile`, smart-provider state,
  and active/disabled provider sets.
- The current checked-in package version is `0.2.1` in both `pyproject.toml`
  and `server.json`.
- The current coverage-gated validation baseline after the latest release-readiness pass is:
  `python -m pytest --cov=paper_chaser_mcp --cov-report=term-missing --cov-fail-under=85`
  => `630 passed`, total coverage `85.00%`.

## Start Here

Read these in order before making behavior or guidance changes:

1. `README.md`
2. `docs/golden-paths.md`
3. `docs/guided-reset-migration-note.md`
4. `.github/copilot-instructions.md`
5. This file: `docs/agent-handoff.md`

For release work, also read `docs/release-publishing-plan.md`.

If you touch the checked-in GitHub agentic workflow, read both:

- `.github/workflows/test-paper-chaser.md`
- `.github/workflows/test-paper-chaser.lock.yml`

The workflow is driven through `workflow_dispatch` and supports `smoke`,
`comprehensive`, and `feature_probe` modes plus a `tool_profile` input.
After editing the Markdown workflow, recompile it with:

```bash
gh aw compile test-paper-chaser --dir .github/workflows
```

## Public Contract Snapshot

### Guided profile

- `research`: default entry point for discovery, literature review, known-item
  recovery, citation repair, and regulatory routing.
- `follow_up_research`: one grounded follow-up over a saved `searchSessionId`.
- `resolve_reference`: DOI/arXiv/URL/citation/reference cleanup.
- `inspect_source`: per-source provenance and trust inspection.
- `get_runtime_status`: profile/provider/runtime sanity check.

### Expert profile

Use only when there is a concrete control need.

- Smart orchestration:
  `search_papers_smart`, `ask_result_set`, `map_research_landscape`,
  `expand_research_graph`
- Raw and provider-specific retrieval:
  `search_papers`, `search_papers_bulk`, provider-specific paper/author paths
- Direct regulatory primary-source tools:
  `search_federal_register`, `get_federal_register_document`, `get_cfr_text`

## Module Map

- `paper_chaser_mcp/settings.py`
  Environment contract, profile selection, transport/runtime knobs.
- `paper_chaser_mcp/tool_specs/`
  Published tool schemas, visibility rules, guided vs expert advertisement.
- `paper_chaser_mcp/tool_specs/descriptions.py`
  User-facing tool descriptions and contract prose.
- `paper_chaser_mcp/dispatch.py`
  Main tool routing layer, including the guided wrappers and runtime-status view.
- `paper_chaser_mcp/agentic/models.py`
  Smart/guided response models, including trust buckets and `unverifiedLeads`.
- `paper_chaser_mcp/agentic/graphs.py`
  Smart orchestration, trust gating, regulatory routing, and abstention logic.
- `paper_chaser_mcp/server.py`
  FastMCP server surface, instructions, resources, prompts, and transport setup.
- `tests/test_dispatch.py`
  Guided wrapper behavior, tool visibility, and routing.
- `tests/test_smart_tools.py`
  Smart trust/regulatory behavior and abstention semantics.
- `tests/test_prompt_corpus.py`
  Guided UX corpus expectations.
- `tests/test_provider_benchmark_corpus.py`
  Acceptance corpus for safe abstention and benchmark consistency.
- `tests/test_agentic_workflow.py`
  Contract tests for `.github/workflows/test-paper-chaser.md` and doc sync.

## Validation Commands

Baseline full-stack validation:

```bash
python -m pip check
pre-commit run --all-files
pytest -q
python -m mypy --config-file pyproject.toml
python -m ruff check .
python -m bandit -c pyproject.toml -r paper_chaser_mcp
python -m build
python -m pip_audit . --progress-spinner off
```

Good focused checks while iterating on guided contracts and docs:

```bash
pytest tests/test_dispatch.py tests/test_smart_tools.py tests/test_agentic_workflow.py -q
pytest tests/test_prompt_corpus.py tests/test_provider_benchmark_corpus.py -q
```

If `.github/workflows/test-paper-chaser.md` changes:

```bash
gh aw compile test-paper-chaser --dir .github/workflows
pytest tests/test_agentic_workflow.py -q
```

## What Was Added In This Pass

- Guided-default public surface and expert fallback via
  `PAPER_CHASER_TOOL_PROFILE`.
- Canonical guided contracts for `research`, `follow_up_research`,
  `resolve_reference`, `inspect_source`, and `get_runtime_status`.
- Trust gating that treats topical drift as a first-class failure mode rather
  than a cosmetic label.
- `unverifiedLeads` as the audit bucket for weak, filtered, or off-topic items.
- Safer regulatory routing that keeps unrelated Federal Register items out of
  verified findings and timeline events.
- Runtime-summary fixes so the top-level summary and per-provider rows describe
  the same environment truth.
- Guided-first README, golden paths, migration note, and Microsoft packaging
  assets.
- Updated UX/benchmark fixtures and tests focused on low-context success and
  safe abstention.

## Known Hotspots

1. Threshold tuning for `on_topic` vs `weak_match` still needs real-world
   iteration. The current behavior intentionally prefers safe abstention over
   plausible garbage.
2. `unverifiedLeads` is useful, but the product can still get better at
   summarizing why a lead was excluded without making users inspect raw source
   records.
3. The expert smart surface remains powerful but broad. Keep it out of default
   docs and workflows unless the task explicitly requires it.
4. The agentic workflow in `.github/workflows/test-paper-chaser.md` is a real
   product-facing regression harness. Keep its guided-default story, its
   `feature_probe` mode, and its compiled lock file synchronized.
5. Provider-specific long-form docs are intentionally expert/operator docs.
   Do not "simplify" them by removing provider nuance, but do keep them from
   leaking into guided onboarding.

## Suggested Next Steps

1. Keep collecting real guided queries and tune trust thresholds, clarification
   prompts, and abstention language from that evidence.
2. Improve `unverifiedLeads` ergonomics so users can see why something was
   excluded without mistaking it for verified support.
3. Continue tightening the expert smart surface until landscape/graph tools
   consistently meet the same trust bar as guided outputs.
4. If the GitHub agentic workflow needs broader expert coverage, keep the
   default `tool_profile=guided` and add explicit expert checks rather than
   flipping the default experience back to raw/smart-first.
5. For release work, follow `docs/release-publishing-plan.md`; do not rely on
   stale branch/tag instructions from older notes.

## Ready Handoff Prompt

Use this prompt for the next coding agent if you want a clean continuation:

```text
Read README.md, docs/golden-paths.md, docs/guided-reset-migration-note.md,
.github/copilot-instructions.md, and docs/agent-handoff.md. Treat the guided
profile as the public contract of record and the expert profile as an explicit
fallback. Preserve safe abstention, unverifiedLeads separation, and runtime
truth. If you edit .github/workflows/test-paper-chaser.md, also update
.github/workflows/test-paper-chaser.lock.yml with:
gh aw compile test-paper-chaser --dir .github/workflows
Then run the smallest focused tests first, followed by pytest -q.
```

## Commit Hygiene

- Keep the diff honest: guided-doc changes should land with any matching test
  updates.
- Commit `.github/workflows/test-paper-chaser.md` and
  `.github/workflows/test-paper-chaser.lock.yml` together.
- Do not revert unrelated user changes in a dirty worktree.
- Prefer a small, reviewable follow-up over a broad "cleanup" commit that mixes
  behavior, docs, and release metadata without a clear theme.
