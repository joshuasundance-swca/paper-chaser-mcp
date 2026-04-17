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
- Guided responses are now **trust-gated**, not just trust-labeled. Grounded
  support belongs in `evidence`; weak, filtered, or off-topic items belong in
  `leads`. Legacy `verifiedFindings` and `unverifiedLeads` are compatibility
  views, not the primary contract.
- Guided follow-up is now **abstention-safe**. `follow_up_research` returns
  `answerStatus=answered|abstained|insufficient_evidence` and should not emit
  answer-shaped filler when evidence is weak.
- Guided follow-up is also **stateful over mixed saved sessions**. When saved
  metadata is strong enough, it can classify sources and leads into on-topic,
  weaker, and off-target groups instead of immediately deferring to a fresh
  retrieval or a metadata-free abstention.
- Guided execution policy is now **server-owned and quality-first**. Guided
  `research` ignores client `latencyProfile`, uses the configured guided
  defaults from `settings.py`, and can run one bounded review escalation when
  the first pass is too weak.
- Guided ambiguity handling is now **structured**. Guided wrappers surface
  `executionProvenance`, `sessionResolution`, `sourceResolution`, and
  `abstentionDetails` so clients can recover without reading exception text.
- Guided routing summaries now surface **query specificity and bounded
  hypotheses**. `routingSummary` can include `querySpecificity`,
  `ambiguityLevel`, `secondaryIntents`, `retrievalHypotheses`, and blended
  `passModes` so clients can react without reading raw smart-layer metadata.
- Regulatory routing is now **subject-anchored**. `research` should either
  build a trustworthy primary-source trail or return
  `needs_disambiguation` / `abstained`; unrelated wildlife notices should not
  appear in grounded `evidence` or timeline events.
- Guided hybrid regulatory requests are now **ambiguity-aware**. A request that
  initially looks regulatory can still add a literature-review pass when the
  regulatory smart pass reports low specificity, high ambiguity, or a review
  secondary intent.
- Broad agency-guidance prompts now stay on the regulatory path, prefer the most
  relevant query-anchored guidance or policy documents in the top summary, and
  retain weaker authority hits as leads rather than silently drifting into known-item recovery.
- Runtime reporting is now **internally truthful**. `get_runtime_status` and
  expert diagnostics should agree on `effectiveProfile`, smart-provider state,
  and active/disabled provider sets. `configuredSmartProvider` is the configured
  smart bundle; `activeSmartProvider` is the latest effective execution path,
  including deterministic fallback when smart execution degrades.
- The repo now includes a **portable eval curation funnel**. Live trace capture,
  review-queue generation, trace promotion, portable exports, service-specific
  publish helpers, and expert batch artifact generation are all checked in.
- The repo now also includes a **profile-driven eval bootstrap path**.
  `scripts/generate_eval_topics.py`, `scripts/run_eval_autopilot.py`, and
  `scripts/run_eval_workflow.py` support ranked topic generation, immutable run
  bundles, guarded workflow handoff, and exploratory single-seed review loops.
- The repo now includes a **cross-domain remediation plan** in
  `docs/cross-domain-remediation-plan.md` covering reranking, routing,
  regulatory breadth, cultural-resource coverage, known-item recovery, and a
  durable cross-domain eval pack derived from recent manual tool exercises.
- Narrow-run eval profiles can now use **single-seed diversification** to ask
  the planner for review, regulatory, and methods-oriented variants instead of
  depending only on looser workflow thresholds.
- The eval capture layer now records **batch-safe offline telemetry** such as
  `runId`, `batchId`, `durationMs`, compact provider-pathway summaries, stage
  timings, confidence signals, and batch-level summary or ledger artifacts.
- Eval capture now also records **heuristic tuning context** such as prompt
  family, query specificity, ambiguity level, retrieval-hypothesis counts, and
  mixed-pass hints so review queues can tune routing thresholds with live traces
  instead of static prompt assumptions.
- The current checked-in package version is `0.2.1` in both `pyproject.toml`
  and `server.json`.
- The current coverage-gated validation baseline after the latest stress-test remediation pass is:
  `python -m pytest --cov=paper_chaser_mcp --cov-report=term-missing --cov-fail-under=85`
  => `962 passed`, total coverage ≥ `85%`.

### Fleet-mode `llm-guidance` branch (in progress, not merged)

The `llm-guidance` branch bundles a focused LLM-first quality pass. All
commits live locally and have not been pushed. HEAD is `5a3ccdd` with the
following additions on top of `main`:

#### Phase 1-3

- **Workstream G — eval fixture expansion** (`21e8006`): cross-domain fixture
  corpus + `tests/test_cross_domain_slices.py` behavioral regression harness.
- **Workstream B/A — relevance & reranking resilience** (`fad75ab`):
  `paper_chaser_mcp/agentic/relevance_fallback.py` introduces a three-way
  deterministic tier (on_topic/weak_match/off_topic) with provenance, a
  degraded-mode cap, and anchored reranking diagnostics.
- **Workstream A/F — follow-up weak-pool gate** (`f315797`): synthesis
  integrity guard so `follow_up_research` cannot promote answer-shaped
  filler when the pool lacks grounded support.
- **Workstream C/D — classification rationale UX + heritage intent family**
  (`ece5fa7`): adds `classificationRationale` on structured source records
  and smart hits, `trustRationale` + `classificationRationaleByBucket` on
  `inspect_source` trust summaries, and a `heritage_cultural_resources`
  intent family with a regulatory ranking boost for Section-106 / NHPA /
  tribal-consultation documents.
- **Workstream E — known-item resolution states** (`dce6e8b`): adds
  `knownItemResolutionState` with `resolved_exact` / `resolved_probable` /
  `needs_disambiguation`, tightens the known-item gate so broad conceptual
  queries labeled `broad_concept`/`low`/`high` by the LLM are not
  force-routed into known-item, and wires the state through
  `citation_repair.py` + the graphs known-item branch.

#### Phase 4 — Residual durable-plan workstreams

- **Env-sci eval autopilot** (`3f68553`): new `tests/fixtures/evals/env_sci_benchmark_pack.json`
  (16 rows across literature/regulatory/follow-up/inspection/species-dossier),
  judge rubric template `tests/fixtures/evals/env_sci_judge_rubric.json`,
  `tests/test_env_sci_eval_slices.py` (8 tests incl. 5 failure-family slices),
  `EnvSciJudgeRubric` dataclass + `build_env_sci_judge_prompt` in
  `eval_curation.py`, and `--slice env-sci` flag on `run_eval_autopilot.py`.
- **Trust UX deepen** (`c74b492`): `whyClassifiedAsWeakMatch` sentence on
  `inspect_source`, `confidenceSignals.evidenceProfileDetail`/`synthesisPath`/
  `trustRevisionNarrative`, `trustSummary.authoritativeButWeak` bucket, and
  parallel `directReadRecommendationDetails` with `{trustLevel, whyRecommended,
  cautions}`.
- **Regulatory intent split + subject-card grounding** (`594b5ce`): planner-level
  `regulatoryIntent` enum (`current_cfr_text` / `rulemaking_history` /
  `species_dossier` / `guidance_lookup` / `hybrid_regulatory_plus_literature`),
  `hybrid_policy_science` retrieval hypothesis, new `agentic/subject_grounding.py`
  LLM-first SubjectCard resolver (deterministic fallback cached per session),
  document-family boost on regulatory ranking, species-dossier `on_topic`
  demotion when no species-specific evidence is present, subject-chain
  evidence gaps surfaced on strategy metadata, and a stricter known-item
  demotion for title-like queries without DOI/URL under high ambiguity.
- **Ruff-format reformat + handoff** (`5a3ccdd`).

#### Validation baseline (HEAD `5a3ccdd`)

- `python -m pytest -q` => **1134 passed, 2 skipped**
- `python -m pytest --cov=paper_chaser_mcp --cov-fail-under=85` => **85.64%**
- `python -m ruff check .` clean
- `python -m mypy --config-file pyproject.toml` clean across **160** source files
- `python -m bandit -c pyproject.toml -r paper_chaser_mcp` clean
- `pre-commit run --all-files` clean (ruff/ruff-format/mypy/bandit/checkov/
  hadolint/PSRule/secret-scan/typos/etc.)

A live MCP probe script `scripts/live_probe_mcp.py` loads `.env` and
drives the guided surface over stdio via the `mcp` Python client for manual
smoke checks.

## Start Here

Read these in order before making behavior or guidance changes:

1. `README.md`
2. `docs/golden-paths.md`
3. `docs/guided-reset-migration-note.md`
4. `.github/copilot-instructions.md`
5. This file: `docs/agent-handoff.md`

If you are changing guided recovery, input normalization, or result-state metadata,
also read `docs/guided-smart-robustness.md`.

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
  recovery, citation repair, and regulatory routing. Includes
  `executionProvenance` for the applied guided policy.
- `follow_up_research`: one grounded follow-up over a saved `searchSessionId`.
  Also surfaces `sessionResolution` on ambiguity and `abstentionDetails` when
  evidence is insufficient.
- `resolve_reference`: DOI/arXiv/URL/citation/reference cleanup with exact identifier normalization before fuzzy recovery.
- `inspect_source`: per-source provenance and trust inspection; omitted `searchSessionId` only works when one compatible saved session exists, and ambiguity now returns structured `sessionResolution` / `sourceResolution` payloads.
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
- `paper_chaser_mcp/eval_curation.py`
  Eval-candidate capture payloads, review-queue generation, and batch summary
  or ledger helpers.
- `paper_chaser_mcp/eval_exports.py`
  Portable eval and training export helpers.
- `paper_chaser_mcp/eval_publish.py`
  Optional Foundry and Hugging Face publish helpers.
- `paper_chaser_mcp/eval_trace_promotion.py`
  Reviewed-trace promotion into durable eval rows.
- `tests/test_dispatch.py`
  Guided wrapper behavior, tool visibility, and routing.
- `tests/test_smart_tools.py`
  Smart trust/regulatory behavior and abstention semantics.
- `tests/test_eval_curation.py`
  Capture, review-queue, and batch-artifact coverage.
- `tests/test_eval_exports.py`
  Portable export coverage for Foundry, Hugging Face, and training-chat rows.
- `tests/test_eval_publish.py`
  Publish-helper validation coverage.
- `tests/test_eval_trace_promotion.py`
  Reviewed-trace promotion coverage.
- `tests/test_prompt_corpus.py`
  Guided UX corpus expectations.
- `tests/test_provider_benchmark_corpus.py`
  Acceptance corpus for safe abstention and benchmark consistency.
- `tests/test_agentic_workflow.py`
  Contract tests for `.github/workflows/test-paper-chaser.md` and doc sync.
- `tests/test_schema_invariants.py`
  Coverage-summary field invariants, author dedup, and active-provider-set correctness.
- `tests/test_answer_validation.py`
  LLM answer-status validation, refusal detection, and async contract-field behavior.
- `tests/test_provider_runtime_fixes.py`
  Capability-based follow-up eligibility and SerpApi venue parsing guards.
- `tests/test_payload_efficiency.py`
  Payload stripping and selected-evidence-only source inclusion.
- `tests/test_citation_repair_fixes.py`
  Year-penalty weighting, title-similarity length penalty, and confidence capping.
- `tests/test_regulatory_routing.py`
  Verification-status defaults, expanded regulatory source types, and routing contracts.

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
pytest tests/test_eval_curation.py tests/test_eval_exports.py tests/test_eval_publish.py tests/test_eval_trace_promotion.py tests/test_eval_canary.py -q
```

If `.github/workflows/test-paper-chaser.md` changes:

```bash
gh aw compile test-paper-chaser --dir .github/workflows
pytest tests/test_agentic_workflow.py -q
```

## Stress-Test Remediation (Phase 1–8)

A comprehensive 8-phase stress test exercised schema integrity, LLM answer
validation, provider runtime contracts, payload efficiency, citation repair,
and regulatory routing. The remediation added 117 new tests across 6 new test
modules and made targeted fixes throughout the codebase.

### Phase 1: Schema Integrity Fixes

- `_smart_coverage_summary()` in `graphs.py`: `providersSucceeded` now excludes
  zero-result providers (was raw `providers_used`).
- `_guided_failure_summary()` in `dispatch.py`: `fallbackAttempted` forced
  `True` when `fallbackMode` is set; `outcome` is `partial_success` (not
  `no_failure`) when abstained with 0 sources.
- Active provider set in `dispatch.py`: suppressed providers excluded from
  `activeProviderSet`.
- Author dedup: `_deduplicate_authors()` groups by `(surname, first_initial)`,
  keeps longest form.
- Tests: `tests/test_schema_invariants.py` (15 tests).

### Phase 2: LLM Answer Status Validation

- New `AnswerStatusValidation` Pydantic schema in `provider_helpers.py`.
- `avalidate_answer_status()` added to `ModelProviderBundle`,
  `OpenAIProviderBundle`, `LangChainChatProviderBundle`.
- `classify_answerability()` in `guided_semantic.py` now accepts `answer_text`
  and includes deterministic refusal detection via `_REFUSAL_PATTERNS`.
- `_guided_contract_fields()` in `dispatch.py` is now async and runs LLM
  validation when a provider bundle is available.
- Tests: `tests/test_answer_validation.py` (25 tests).

### Phase 3: Provider Runtime Fixes

- `canAnswerFollowUp` is now capability-based (`has_sources AND has_session`)
  not just permission-based.
- SerpApi venue parsing: `_validate_venue_candidate()` rejects author-pattern
  strings and over-long venues.
- Tests: `tests/test_provider_runtime_fixes.py` (16 tests).

### Phase 4: Payload Efficiency

- Follow-up responses only include sources referenced in
  `selectedEvidenceIds`.
- Evidence and source records stripped of null/empty fields via
  `strip_null_fields()`.
- Tests: `tests/test_payload_efficiency.py` (9 tests).

### Phase 5: Citation Repair

- Year penalty in `_rank_candidate()` increased:
  `-0.04×min(delta,5)` + hard penalty for `delta>5`.
- Upstream confidence bonus reduced (`0.25→0.15` for high).
- Title conflict caps confidence at `"medium"` in
  `_classify_resolution_confidence()`.
- Length-difference penalty added to `_title_similarity()`.
- Tests: `tests/test_citation_repair_fixes.py` (13 tests).

### Phase 6: Regulatory Routing

- `_assign_verification_status()` default changed from `verified_metadata` to
  `unverified` for papers without DOI or identifiers.
- `_REGULATORY_SOURCE_TYPES` expanded from 4 to 13 entries.
- Tests: `tests/test_regulatory_routing.py` (39 tests).

## What Was Added In Earlier Passes

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
- A new eval-funnel doc set covering dataset schema, platform strategy,
  integrations, trace promotion, and model-selection context.
- Portable eval modules and scripts for capture, queue building, promotion,
  export, publish validation, and expert batch execution.
- Offline batch artifacts for eval curation: `expert-batch-report.json`,
  `captured-events.jsonl`, `review-queue.jsonl`, `batch-summary.json`, and
  `batch-ledger.csv`.
- Profile-driven eval bootstrap artifacts and wrappers for ranked topic
  generation, autopilot review gating, immutable run bundles, and exploratory
  single-seed profiles.

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
6. The eval funnel is now broader than the first doc pass. Keep the README,
  trace-promotion guide, integration guide, platform strategy, and this
  handoff doc synchronized whenever batch artifacts, queue shape, or export
  semantics change.
7. The current narrow-run exploratory blocker is family cross-check
  disagreement. Single-seed diversification improved coverage in live runs,
  but valid review or policy pivots can still be blocked by cross-check
  thresholds before workflow handoff.
8. Citation repair now penalizes year mismatch more aggressively and caps
   confidence when title conflicts exist. Watch for regressions in near-miss
   citation resolution where a ±1 year difference is expected.
9. Papers without DOI or identifiers now default to `unverified` verification
   status. This is correct for integrity but may increase `unverified` counts
   in regulatory result sets where primary-source metadata is sparse.

## Suggested Next Steps

1. Keep collecting real guided queries and tune trust thresholds, clarification
   prompts, and abstention language from that evidence.
2. Use `docs/environmental-science-remediation-plan.md` as the implementation
  plan for the latest environmental-science review, especially for follow-up
  degradation, species-specific regulatory grounding, and provenance UX.
3. Improve `unverifiedLeads` ergonomics so users can see why something was
   excluded without mistaking it for verified support.
4. Continue tightening the expert smart surface until landscape/graph tools
   consistently meet the same trust bar as guided outputs.
5. If the GitHub agentic workflow needs broader expert coverage, keep the
   default `tool_profile=guided` and add explicit expert checks rather than
   flipping the default experience back to raw/smart-first.
6. For release work, follow `docs/release-publishing-plan.md`; do not rely on
   stale branch/tag instructions from older notes.
7. Calibrate family cross-check disagreement handling for exploratory one-seed
  runs so valid review or policy pivots are not over-penalized once topic
  count and family coverage improve.
8. Use the stress-test remediation as a baseline and continue with the
  cross-domain remediation plan workstreams (reranking quality,
  relevance-classification resilience, cultural-resource routing).
9. Monitor citation-repair regressions after the tightened year penalty and
  reduced upstream confidence bonus. Collect examples where the new weights
  over-penalize legitimate near-miss citations.
10. Follow-up synthesis integrity (Workstreams A + F) landed via
    `paper_chaser_mcp/agentic/answer_modes.py`, which centralizes
    question-mode classification, weak-pool detection, and the evidence-use
    plan. `ask_result_set` now gates synthesis-heavy answers through a
    weak-pool check (skipped when the plan is already sufficient), a strict
    synthesis-sufficiency check, and a generalized fallback-only gate
    (previously comparison-only). Deferred items worth a near-term follow-up:

    - Replace the keyword heuristic in `classify_question_mode` with a real
      LLM answer-mode classifier. It is deliberately distinct from the
      per-paper `llm_relevance` evidence-mode classifier; conflating them is
      documented tech debt.
    - Align `dispatch.py::_guided_follow_up_response_mode` (still has its own
      keyword logic, emits `"evidence_planning"` not in `ANSWER_MODES`) with
      `classify_question_mode`.
    - Empirically tune the weak-pool threshold
      (`PAPER_CHASER_WEAK_EVIDENCE_POOL_THRESHOLD`, default 0.6).
    - Consider extending the fallback-only gate to `"unknown"` mode. Today
      generic synthesis-shaped questions that don't keyword-match any
      synthesis family classify as `"unknown"` and get a permissive plan so
      downstream `should_abstain` / coverage / unsupported-asks machinery
      retains ownership.
    - Real provider-lane bounded recovery (distinct from the current
      deterministic fallback path) remains unimplemented and is currently
      dead code.

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
