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
  LLM-first SubjectCard resolver with deterministic fallback (no cross-request
  cache — see "Known gaps" in `docs/guided-smart-robustness.md`),
  document-family boost on regulatory ranking, species-dossier `on_topic`
  demotion when no species-specific evidence is present, subject-chain
  evidence gaps surfaced on strategy metadata, and a stricter known-item
  demotion for title-like queries without DOI/URL under high ambiguity.
- **Ruff-format reformat + handoff** (`5a3ccdd`).

#### Phase 5 — Docs alignment + native planner schema

- **User-facing docs + plan status markers** (`57e23ec`).
- **Native planner LLM emission** of `regulatoryIntent` + `subjectCard`
  via `_PlannerResponseSchema` (`9c2468d`); deterministic fallback preserved.
- **Durable handoff summary** (`319f8a0`).

#### Phase 6 — Integration red-team + critique remediation

- **End-to-end signal red-team** (`b3df3d4`): new
  `tests/test_phase4_signal_integration.py` covers the full plumbing from
  planner → smart layer → dispatch response. Surfaced a critical bug where
  `strategyMetadata.{intentFamily,regulatoryIntent,subjectCard,subjectChainGaps}`
  were emitted internally but dropped by the dispatch serializer.
- **Docs-drift cleanup** (`4637cde`): aligned overclaims in
  `guided-smart-robustness.md`, `golden-paths.md`, and this handoff with
  actual code behaviour; added a "Known gaps" subsection for honest tech
  debt (no cross-request SubjectCard cache, no richer SubjectCard fields,
  no per-source LLM-composed weak-match rationale).
- **Surface strategy signals + LLM-first follow-up gate + provenance fix**
  (`df918df`): extended `routingSummary` with Phase 4/5 fields (additive
  Option C); added LLM-first `classify_question_mode` with async variant,
  session-hint precedence, classifier cache, and fail-closed unknown-mode
  branch for paraphrased synthesis over weak pools; fixed
  `resolve_subject_card` provenance so `DeterministicProviderBundle`
  correctly stamps `source="deterministic_fallback"`.
- **Regulatory `unspecified` fallback** (`8ce97d7`): `_infer_regulatory_subintent`
  now returns `None` when no specific cue is present, so
  `_derive_regulatory_intent` reaches its documented `unspecified` fallback
  for broad regulatory queries instead of biasing onto rulemaking timelines.
- **Dispatch compat + trust threading** (`3abda1e`): restored legacy
  `fullTextObserved` key alongside `fullTextUrlFound` (dual-emit), restored
  `verified_metadata` default for DOI-less scholarly records with basic
  metadata, and threaded `subjectChainGaps` into `_guided_confidence_signals`
  and `_guided_trust_summary` so machine-readable trust signals mirror the
  human rationale.

- **Test-masking cleanup** (`a16a6e8`): loosened brittle prose assertions
  and pinned provenance stamps strictly in `test_trust_ux_deepen.py` and
  `test_planner_llm_schema.py` so future regressions in trust sentence
  composition or subject-card provenance surface directly.

#### Phase 7 — LLM-first remediation round

- **LLM-first literature/mixed-intent classification** (`13c4a87`): planner
  `regulatoryIntent=hybrid_regulatory_plus_literature` now drives
  `_guided_should_add_review_pass`; keyword heuristic becomes last-resort
  fallback for deterministic bundles.
- **LLM bundles correctly report `is_deterministic=False`** (`3a6b8c0`):
  OpenAI and LangChain bundles override the shim default, and
  `classify_query` snapshots grounding signals before deterministic
  fallback can pollute them.
- **Saved-session trust summary carries `subjectChainGaps`** (`7dc9710`):
  `_guided_session_state` now threads the signal into rebuilt trust
  summaries so session-introspection follow-ups inherit live-path trust
  signals.
- **Surface LLM classification alongside deterministic verdict**
  (`f85988b`): Candidate/PaperRecord carry `llmClassification` +
  `classificationSource`; smart-search strategyMetadata tallies
  `llmClassificationOverrides` when the deterministic gate disagrees
  with the LLM signal.
- **`ask_result_set` routes question-mode via LLM classifier**
  (`dfb12a4`): `aclassify_question_mode` is invoked with an explicit
  `followUpMode` hint and passed into `build_evidence_use_plan`,
  rescuing paraphrased synthesis queries over weak pools.
- **LLM-first regulatory triage** (`5ac8111`):
  `_derive_regulatory_query_flags` in `graphs.py` consults planner
  `regulatoryIntent` + `subject_card.source` before falling back to the
  CFR regex / agency-guidance keyword helpers; `_ecos_query_variants`
  prepends planner-emitted species names so scientific-name
  recognition does not depend solely on regex.
- **Planner source provenance + hybrid corroboration** (`96cdc02`):
  every provider bundle stamps `planner_source ∈ {llm, deterministic,
  deterministic_fallback}` on the planner object. Subject-card provenance
  keys off this flag instead of the unreliable `intent_source`, so
  explicit-mode LLM planner runs no longer get downgraded and LLM
  providers that internally fall back to deterministic are now stamped
  correctly. The hybrid regulatory+literature label also now requires
  query-side corroboration (literature keyword cue, `review`/`literature`
  secondary intent, or `hybrid_policy_science` retrieval hypothesis) in
  both `_derive_regulatory_intent` and `_guided_should_add_review_pass`.
- **Concrete `aclassify_answer_mode` on LLM bundles** (`0776faf`):
  `OpenAIProviderBundle` and `LangChainChatProviderBundle` now implement
  `aclassify_answer_mode` via a compact structured-output schema so
  `ask_result_set` actually reaches an LLM classifier in production
  instead of the no-op base stub.

#### Phase 7.5 — Two more critique + UX remediation rounds

- **Follow-up next-action alignment** (`1796a94`):
  `_guided_best_next_internal_action` now returns `"research"` when
  `sources=[]` on weak statuses, so `follow_up_research` without an
  inferable session no longer contradicts itself with an
  `inspect_source` hint.
- **Abstention details on partial results** (`5f19a89`): `research`
  responses with `status="partial"` now populate
  `abstentionDetails.refinementHints` with concrete retry strategies
  (`weak_topical_match` / `narrow_evidence_pool`), eliminating the
  silent "partial means you figure it out" UX.
- **LLM regulatoryIntent honored without subject-card grounding**
  (`42f519c`): `_derive_regulatory_query_flags` now keys authority off
  `planner_source == "llm"` (plus non-null `regulatory_intent`) rather
  than `subject_card.source`, so an LLM that emits regulatoryIntent
  without grounding fields no longer gets overridden by keyword
  heuristics.
- **Planner-time hybrid corroboration matches dispatch** (`5cf7cca`):
  `_has_literature_corroboration` now accepts
  `hybrid_policy_science` / `literature` / `peer-reviewed` markers in
  `retrievalHypotheses`, mirroring `_guided_should_add_review_pass`
  so valid hybrid labels are not stripped before dispatch sees them.
- **ECOS provenance + raw-first variant order** (`156e3c8`):
  `_ecos_query_variants` now emits raw/regex-derived candidates first
  with planner names as fallback; each variant is tagged with an
  `origin` field, and winning hits carry `_ecosProvenance`
  (`corroborated` vs `planner_only`) so downstream ranking can
  down-weight hallucinated planner-only matches.

#### Phase 7.6 — Fourth rubber-duck round

- **Prefer `inspect_source` when saved session is still inspectable**
  (`de553a4`): `_guided_best_next_internal_action` now accepts a
  `saved_session_has_sources` signal so follow-ups with empty response
  `sources` but a usable saved session no longer contradict
  `failureSummary.recommendedNextAction`.
- **Route all-off-topic results to `research`** (`9ee3168`): when every
  returned source is weak/off-topic, `bestNextInternalAction` defaults
  to `research` instead of wasting a call on `inspect_source`.
- **`regulatory_intent_source` stamped separately from `planner_source`**
  (`b231e47`): planner now tracks whether `regulatoryIntent` was
  actually emitted by the LLM (`regulatory_intent_source="llm"`) vs
  deterministically backfilled so the LLM-authoritative route in
  `_derive_regulatory_query_flags` cannot be triggered by a keyword
  fallback masquerading as LLM output.
- **Defer raw full-query ECOS variant for opaque queries** (`8498e26`):
  `_is_opaque_query` detects DOI/URL/arXiv-shaped inputs and orders
  planner species variants first so opaque prose doesn't starve
  legitimate LLM-supplied names.
- **Rank ECOS variants by `hits × provenance_factor`** (`f616b71`):
  raw variants keep factor 1.0, planner-only variants get 0.9, so the
  previously-unused `_ecosProvenance` metadata now genuinely
  influences selection.

#### Phase 7.7 — Fifth rubber-duck round (regression hardening)

- **Two-bool saved-session signal** (`251334a`): replaces the bare
  `saved_session_has_sources` bool with
  `(saved_session_has_sources, saved_session_all_off_topic)` so an
  all-off-topic saved session no longer bypasses the
  "all off-topic → research" routing fix.
- **Machine failure payload honors saved-session signal** (`9d2f2de`):
  `_guided_machine_failure_payload` now threads the same signal so
  follow-up failure paths behave consistently with the response-level
  fallbacks.
- **`hasInspectableSources` agrees with saved-session routing**
  (`236a22d`): regression test pins that when
  `bestNextInternalAction="inspect_source"` via the saved-session
  path, `hasInspectableSources` is also True.
- **Strict provenance-first ECOS ranking** (`7a5e460`): sort key is
  now `(has_hits, provenance, hit_count, variant_idx)` so any
  corroborated variant with ≥1 hit beats any planner-only variant,
  regardless of hit count.

#### Phase 7.8 — Cross-field consistency with the two-bool saved-session signal

- **`_guided_next_actions` / `_guided_machine_failure_payload` honor
  `saved_session_inspectable`** (`2436b1c`): emits an `inspect_source`
  nextAction entry that agrees with `failureSummary.recommendedNextAction`
  and `resultState.bestNextInternalAction`. Fixes R6 findings where saved
  session was inspectable but `nextActions` still pointed nowhere useful.
- **Route empty / all-off-topic guided paths consistently** (`6e8714d`):
  introduces shared `_guided_sources_all_off_topic` predicate; extends
  `_guided_failure_summary` and `_guided_next_actions` with
  `all_sources_off_topic`; normal `follow_up_research`/`ask_result_set`
  branch now threads saved-session inspectability into
  failureSummary/nextActions/resultState. Resolves R7 findings.

#### Phase 7.9 — Exhaustive off-topic cross-field sweep (R8–R11)

- **`hasInspectableSources` effective predicate** (`edbd41b`): aligns
  `_guided_result_state.hasInspectableSources` with the same
  `has_sources AND NOT all_sources_off_topic` predicate that drives
  routing. Previously True for all-off-topic pools.
- **canAnswerFollowUp / failureSummary / groundedness / follow_up
  saved-session** (`436fd7c`): `_guided_result_state.canAnswerFollowUp`
  now uses inspectable_sources; `_guided_failure_summary.whatStillWorked`
  no longer claims inspectable when pool is all-off-topic; answered +
  all-off-topic downgrades `groundedness` to
  `"insufficient_evidence"` and sets `missingEvidenceType="off_topic_only"`;
  `follow_up_research` threads saved-session signals when current pool
  is all-off-topic (previously only when empty).
- **abstentionDetails effective inspectable** (`4e27b06`): zero
  `inspectableSourceCount` and flip `canInspectSources=False` when every
  source is off-topic.
- **Abstention category + resultMeaning off-topic** (`11be79e`): force
  `category="off_topic_only"` when sources are non-empty but all
  off-topic (stops "Inspect the returned sources..." hints). Add
  `all_sources_off_topic` kwarg to `_guided_result_meaning`; six
  research/follow_up callers pass it so prose no longer claims "some
  relevant evidence" for off-topic-only pools.

#### Phase 7.10 — Off-topic convergence into smart-search + LLM-first ranking gates + UX audit

- **Off-topic-aware smart search** (`b763678`): `agentic/graphs.py`
  adds `_has_on_topic_sources` and excludes `topical_relevance ==
  "off_topic"` from `_has_inspectable_sources`. All four
  `SmartSearchResponse` builders (search, regulatory early/late, review)
  now drive `best_next_internal_action` from on-topic-aware
  `has_sources`, so raw provider hits that are entirely off-topic no
  longer advertise `hasInspectableSources` or `bestNextInternalAction:
  "inspect_source"`. Pinned by `tests/test_graphs_inspectable_helpers.py`.
- **LLM-first ranking gates** (`17c6eeb`): `agentic/ranking.py`
  drops the `len(facets) >= 2` heuristic from `broad_query_mode`
  (planner signals only) and adds `concept_bonus_gate_scale`: confident
  `off_topic` zeroes the concept bonus, confident `weak_match` halves
  it, fallback classifications pass through. Exposed as
  `conceptBonusGateScale` in `scoreBreakdown`. Pinned by two
  monkey-patched regression tests in `test_smart_tools.py`.
- **inspect_source candidate metadata + resolve_reference status split**
  (`ccff280`): `SourceResolution` now carries
  `availableSourceCandidates` (compact projection: sourceId, title,
  topicalRelevance, canonicalUrl, retrievedUrl, confidence, accessStatus,
  verificationStatus, publicationYear) plus `candidatesHaveInspectable`
  on the unresolved `inspect_source` branch. `resolve_reference` now
  emits distinct `nextActions` for `resolved` vs `multiple_candidates`
  (previously shared), and `golden-paths.md` documents the full status
  set including `needs_disambiguation` (previously undocumented).
- **Adaptive anchor thresholds + planner-gated decay priors**
  (Finding #3 + Finding #4 of the LLM-first ranking audit):
  `agentic/ranking.py` now derives `anchorThresholdScale`
  (`0.6` when planner `query_specificity=="low"` or
  `ambiguity_level=="high"`, else `1.0`) and applies it to the three
  anchor-match thresholds (`0.34`, `0.5`, `0.25`). It also derives
  `yearDecayScale` / `citationDecayScale` (`0.5` when
  `routing_confidence=="low"` or `query_specificity=="low"`, else
  `1.0`) and splits the combined citation/year prior so each decay can
  be dampened independently. All three scales are exposed in
  `scoreBreakdown` and fall back to strict defaults when planner
  signals are absent. Pinned by two new regression tests in
  `test_smart_tools.py` plus an updated assertion on the existing
  anchored-broad nitrate/headwater test (the drift candidate now
  clears the relaxed threshold — on-topic still wins on title-anchor
  coverage).

#### Validation baseline (HEAD pending Finding #3/#4 commit)

- `python -m pytest -q` => **1290 passed, 2 skipped** (live-only)
- `python -m ruff check .` clean
- `python -m mypy --config-file pyproject.toml` clean
- `pre-commit run --all-files` clean

#### Validation baseline (HEAD `ccff280`)

- `python -m pytest -q` => **1285 passed, 2 skipped** (live-only)
- `python -m ruff check .` clean
- `python -m mypy --config-file pyproject.toml` clean
- `pre-commit run --all-files` clean

#### Validation baseline (earlier HEAD `11be79e`)

- `python -m pytest -q` => **1275 passed, 2 skipped** (live-only)
- `python -m ruff check .` clean
- `python -m mypy --config-file pyproject.toml` clean across **171** source files
- `python -m bandit -c pyproject.toml -r paper_chaser_mcp` clean (0 Medium, 0 High)
- `python -m bandit -c pyproject.toml -r paper_chaser_mcp` clean
- `pre-commit run --all-files` clean (ruff/ruff-format/mypy/bandit/checkov/
  hadolint/PSRule/secret-scan/typos/etc.)

A live MCP probe script `scripts/live_probe_mcp.py` loads `.env` and
drives the guided surface over stdio via the `mcp` Python client for manual
smoke checks.

##### UX audit follow-up — runtime health consolidation

`get_runtime_status` now emits structured health signals alongside the existing
free-form warnings, so low-context agents can branch on state without
string-parsing.

- New `runtimeSummary.healthStatus` enum:
  `nominal | degraded | fallback_active | critical`.
  - `critical` is reserved for the case where the smart layer failed to
    initialize and there is no usable smart fallback.
  - `fallback_active` means the configured smart provider is unavailable and
    deterministic fallback is in use.
  - `degraded` means the smart provider is healthy but at least one optional
    non-smart provider is disabled or suppressed.
  - `nominal` means the configured smart provider is healthy and every
    optional provider in scope is enabled.
- New `runtimeSummary.fallbackActive: bool` mirror + `runtimeSummary.fallbackReason`
  (present only when `fallbackActive` is true).
- New `runtimeSummary.structuredWarnings: list[{code, severity, message, subject}]`.
  Severity is one of `info | warning | critical`. Codes are classified at
  emission time (no regex post-hoc classification of free-form strings);
  known codes include `smart_provider_fallback`, `provider_disabled`,
  `chat_only_smart_provider`, `tools_hidden`, `stdio_transport`,
  `ecos_tls_disabled`, `guided_hides_expert`, and `narrow_provider_order`.
- The legacy `runtimeSummary.warnings: list[str]` (also mirrored on the
  top-level `warnings` field of the tool response) is preserved byte-identical
  for backward compatibility.

All additions are additive. Regression coverage lives in
`tests/test_dispatch.py::test_get_runtime_status_nominal_when_all_providers_healthy`,
`test_get_runtime_status_fallback_active_reflects_deterministic_smart_provider`,
and `test_get_runtime_status_structured_warnings_encode_legacy_strings`.

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
