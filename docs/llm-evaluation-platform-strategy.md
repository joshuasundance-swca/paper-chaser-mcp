# LLM Evaluation Platform Strategy

This document explains how Paper Chaser MCP should use external evaluation and
iteration platforms alongside the repo-local evaluation assets.

The short version is:

- keep the repo-local datasets, canaries, and contract checks as the portable
  source of truth
- use platform tooling selectively where it adds leverage
- prefer trace-driven curation and active-learning loops over static benchmark
  worship

The primary target is the under-the-hood LLM capability in the Paper Chaser
runtime: structured outputs, routing judgment, grounded synthesis, evidence-id
selection, safe abstention, and related supporting behavior. External MCP-client
tool-use evaluation is a separate concern.

## Core strategy

Paper Chaser MCP should use a layered evaluation stack.

### Local repo-first layer

This repo should remain the system of record for:

- role definitions
- evaluation dataset schema
- seed and benchmark datasets
- deterministic contract checks
- role-aware canary runners
- release-gate logic for public contract safety

That is what keeps the evaluation program portable across model providers and
platforms.

### Platform-assisted layer

External platforms are useful when they add one of these:

- scalable batch evaluation
- judge orchestration and trace visualization
- dataset/version management beyond what is convenient in git alone
- experiment tracking across models or prompts
- custom evaluator hosting

They should not replace the repo-local truth. They should amplify it.

## Azure AI Foundry

Azure AI Foundry is best used here as a secondary evaluation and observability
layer rather than the primary source of truth.

### Best fit

Foundry is strongest for:

- batch evaluation of model or agent runs
- built-in evaluators for relevance, groundedness, and safety
- custom evaluators for role-specific judgments
- agent evaluation over tool-using workflows
- dataset versioning inside a project context
- tracing and observability for multi-step workflows

For Paper Chaser MCP, that maps well to:

- synthesis quality comparisons
- planner or routing quality sampled through evaluator rubrics
- nightly or pre-release batch checks against curated datasets
- shadow evaluation on sampled traces after larger behavior changes

### Best use in this repo

Use Foundry for:

- nightly or scheduled evaluation tiers over promoted datasets
- custom evaluators for role-specific questions such as citation trust,
  grounded follow-up quality, and tool-selection quality
- trace inspection for complex agentic workflows
- post-release monitoring of sampled runs where cloud-hosted analysis is worth
  the cost

Do not make Foundry the only evaluation system. Keep the contract and dataset
definitions in git and mirror selected datasets into Foundry when helpful.

### Limits and cautions

Use Foundry carefully because:

- evaluator and region support can change over time
- custom evaluators are still operational dependencies, not just code
- platform-specific metadata can encourage lock-in if the repo stops being the
  canonical definition of tasks and metrics
- cloud evaluation is slower and more expensive than local deterministic checks

## Hugging Face

Hugging Face is best used here as a dataset, benchmark, and experiment
management companion, not as the only evaluation runtime.

### Best fit

Hugging Face is strongest for:

- versioned datasets and public or private dataset hosting
- benchmark distribution and reuse
- evaluation libraries such as `evaluate`
- broader benchmark harnesses such as LightEval
- experiment reporting and comparison through ecosystem integrations
- model and judge iteration for teams that want to train or adapt evaluators

For Paper Chaser MCP, that maps well to:

- hosting promoted evaluation datasets outside the repo when scale or sharing
  becomes painful in git
- comparing ranking, retrieval, or judge-model behavior across versions
- publishing or consuming benchmark-style evaluation slices without rewriting
  local dataset definitions

### Best use in this repo

Use Hugging Face for:

- dataset mirroring when the evaluation bank grows beyond comfortable git use
- public or internal benchmark sharing
- LightEval or `evaluate` integration for selected retrieval or ranking tasks
- optional judge-model experimentation if the project later trains a small
  internal evaluator model

Do not assume Hugging Face replaces active review, trace governance, or
contract-safe local checks.

### Limits and cautions

The biggest gaps for this repo are:

- no built-in active-learning workflow specific to these guided contracts
- no native replacement for strong domain review on provenance and abstention
- easy temptation to optimize for public benchmark styles that do not match the
  actual MCP contract

## Active learning and live-trace curation

The long-term evaluation advantage for this repo will not come from finding one
perfect benchmark. It will come from systematic curation from real traces.

### Recommended loop

Use a repeatable loop:

1. Collect traces from guided and expert workflows.
2. Score them with cheap deterministic and heuristic checks.
3. Sample for review using uncertainty, disagreement, and coverage-gap rules.
4. Review them against role-specific rubrics.
5. Promote accepted examples into versioned datasets.
6. Keep a frozen regression set and a living recent-failure set.

### What to sample

Sample traces that show:

- planner ambiguity or weak routing confidence
- deterministic fallback activation in smart paths
- surprising abstentions or suspicious non-abstentions
- evidence versus leads boundary disagreements
- provenance confusion or source-resolution failures
- repeated user reformulations after an answer
- cross-provider disagreement when available

### Promotion rules

Promote traces into benchmark sets only when they are:

- clearly labeled by a rubric
- reviewed to the appropriate confidence tier
- useful as a stable regression or capability slice
- documented with origin, date, and version metadata

Do not let raw live traces become hidden benchmark debt. Every promoted example
needs provenance and review state.

## Suggested operating model

Use three tiers.

### Tier 1: Local deterministic canaries

Run on every meaningful evaluation or contract change.

Use:

- `scripts/run_eval_canaries.py`
- seed datasets in `tests/fixtures/evals/`
- focused pytest coverage

### Tier 2: Richer offline or batch evaluation

Run on schedule or before larger releases.

Use:

- expanded curated datasets
- judge-assisted comparisons
- local batch reports or cloud batch evaluation when useful

This is the right place for Azure AI Foundry batch evaluations and optional
Hugging Face dataset exports.

For repeatable model comparisons, use matrix execution across runtime
configurations rather than pretending one run can represent every provider.

### Tier 3: Shadow and trace-driven evaluation

Run after or around behavior changes that affect real-world quality.

Use:

- sampled real traces
- human review on targeted slices
- disagreement and uncertainty sampling
- promotion of reviewed failures into the next dataset version

An effective curation funnel is:

1. optional live eval-candidate capture in the runtime
2. review-queue generation for human triage
3. offline batch summary and ledger generation for drift inspection
4. reviewed-trace promotion into eval rows
5. selective export into training or fine-tuning corpora when policy permits

This keeps evaluation and training data related but not conflated. Review stays
in the middle.

Current local artifact posture:

- `captured-events.jsonl` is the compact event stream of record for curation
- `review-queue.jsonl` is the reviewer-facing triage surface
- `expert-batch-report.json` preserves raw batch outcomes
- `batch-summary.json` is the aggregate offline drift and throughput surface
- `batch-ledger.csv` is the flat dashboard-friendly batch artifact

Those artifacts are intentionally portable and credential free. They should be
usable before any Foundry or Hugging Face publish step is configured.

Portable integration stance:

- Azure AI Foundry: export promoted eval rows in a Foundry-friendly JSONL shape,
  then register or upload them with `scripts/upload_foundry_eval_dataset.py`.
- Hugging Face: export review-approved rows in a dataset-friendly JSONL shape,
  then publish or sync them with `scripts/upload_hf_eval_assets.py`.

This repo should prefer portable export surfaces first and optional service
connectors second. That keeps the core evaluation workflow usable even when cloud
credentials or external SDKs are unavailable.

Concrete service notes:

- Azure AI Foundry is the strongest fit for cloud batch evaluation, evaluator
  orchestration, and model-comparison workflows.
- Hugging Face is the strongest fit for dataset distribution, training-export
  interoperability, and optionally mounted storage via `hf-mount` when a shared
  filesystem-like capture sink is useful.

The first repo-local helpers for this are:

- `scripts/promote_eval_traces.py`
- `scripts/upload_foundry_eval_dataset.py`
- `scripts/upload_hf_eval_assets.py`
- `tests/fixtures/evals/trace-promotion.sample.jsonl`

## Recommended division of labor

Use this repo for:

- contract definitions
- dataset schema
- seed and promoted benchmark sources
- deterministic runner logic
- release-safe canaries

Use Azure AI Foundry for:

- batch and scheduled cloud evaluations
- custom evaluators over hosted runs
- tracing-heavy workflow inspection

For the current architecture, Foundry agent or tool-call evaluators are most
useful when they map onto internal role behavior such as routing judgment or
tool-selection quality inside the runtime. They should not distract from the
fact that much of Paper Chaser orchestration is still app-managed rather than
free-form model-driven tool use.

Use Hugging Face for:

- dataset mirroring and sharing
- benchmark packaging
- optional evaluator-model experimentation

Use active-learning trace workflows for:

- keeping the eval bank realistic
- closing the loop on recent failures
- making the evaluation program evolve with actual user behavior

## Maintenance rule

If platform-specific workflows are adopted later, document them as optional
augmentation paths. Do not move the authoritative task definitions out of this
repo.
