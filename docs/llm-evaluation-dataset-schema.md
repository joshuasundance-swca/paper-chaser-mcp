# LLM Evaluation Dataset Schema

This document defines the first stable schema for role-based LLM evaluation
items in Paper Chaser MCP.

It is designed for:

- small human-reviewed seed sets
- future synthetic expansion around those seeds
- deterministic offline canary validation
- eventual live or shadow evaluation runners

The schema is intentionally pragmatic. It is specific enough to support planner,
synthesis, abstention, and provenance tasks without trying to solve every
future evaluation need in the first version.

## Design goals

The schema should:

- align with the role boundaries in `docs/llm-evaluation-program-plan.md`
- be easy to store as JSONL
- support seed sets written by humans
- support lightweight deterministic validation in a first runner
- leave room for future judge prompts, live observations, and score records

The intended target for these datasets is usually the internal LLM role inside
Paper Chaser rather than an external MCP client model.

## Storage format

Each evaluation item is one JSON object per line in a `.jsonl` file.

Recommended layout:

```text
tests/
  fixtures/
    evals/
      planner.seed.jsonl
      synthesis.seed.jsonl
      abstention.seed.jsonl
      provenance.seed.jsonl
      runtime.seed.jsonl
      misc.seed.jsonl
      judge-rubrics.json
```

Each file should contain items from exactly one `task_family`.

Judge rubrics are stored separately in `judge-rubrics.json` because they are
shared evaluator assets rather than per-item benchmark rows.

## Top-level row shape

```json
{
  "meta": {
    "id": "planner_known_item_transformer_001",
    "task_family": "planner",
    "dataset_version": "0.1.0",
    "schema_version": 1,
    "origin": "human",
    "review_status": "validated",
    "tags": ["known-item", "anchor-strength", "scientist"]
  },
  "input": {
    "query": "Attention Is All You Need 2017 Vaswani"
  },
  "expected": {
    "acceptable_intents": ["known_item"],
    "unacceptable_intents": ["regulatory"],
    "acceptable_provider_hints": ["semantic_scholar", "arxiv"],
    "must_surface_clarification": false,
    "should_allow_partial": false
  },
  "why_it_matters": "Known-item recovery is a core researcher workflow and should not degrade into broad discovery.",
  "notes": "Good first-pass planner canary for anchor recognition."
}
```

## Required fields

Every item must contain:

- `meta`
- `input`
- `expected`
- `why_it_matters`

### `meta`

Required fields:

- `id`: stable unique identifier
- `task_family`: one of `planner`, `synthesis`, `abstention`, `provenance`, `runtime`, `misc`
- `dataset_version`: semantic version string for the dataset release
- `schema_version`: currently `1`
- `origin`: one of `human`, `synthetic_reviewed`, `trace_mined`, `adversarial`
- `review_status`: one of `draft`, `validated`, `needs_review`
- `tags`: non-empty list of short slice labels

Recommended optional field:

- `evaluation_target`: for example `internal_llm_role` or `workflow_contract`
- `live_eval`: optional override for future live execution settings; when omitted,
  some families can still derive a default live execution path from the item
  itself
- `review_labels`: optional reviewed-label payload carried forward from promoted
  traces when the dataset row came from the trace-curation funnel
- `lineage`: optional provenance payload linking a promoted row back to the
  reviewed trace that produced it

Example `live_eval` fields:

- `research_query` for synthesis items that need a preceding retrieval step

For the current role-based seed sets, prefer `internal_llm_role` unless the row
is explicitly about public contract behavior rather than model capability.

For live execution mode, remember that the runner uses the currently configured
Paper Chaser runtime. It does not force every provider or tool on automatically.
That is intentional for under-the-hood model evaluation: compare runtime
configurations explicitly when you want provider-by-provider evaluation.

### `input`

The `input` object stores the user-facing task setup.

Common fields:

- `query`
- `query_context`
- `follow_up_question`
- `source_id`
- `search_session_id`
- `evidence_quality`
- `notes`

Not every field is required for every family. The family-specific rules below
define the minimum expected inputs.

### `expected`

The `expected` object stores the role-specific acceptance conditions for the
item. It is intentionally compact and does not try to encode every possible
score.

### `why_it_matters`

Short explanation of why this slice matters to real users of the MCP server.
This should read like a product-facing rationale, not an implementation note.

## Family-specific rules

### Planner

Required input fields:

- `query`

Required expected fields:

- `acceptable_intents`: non-empty list
- `unacceptable_intents`: list, possibly empty
- `acceptable_provider_hints`: non-empty list
- `must_surface_clarification`: boolean
- `should_allow_partial`: boolean

Optional expected fields:

- `required_markers`
- `disallowed_markers`
- `expected_route_shape`

### Synthesis

Required input fields:

- `query_context`
- `follow_up_question`
- `evidence_quality`

Required expected fields:

- `expected_answer_status`: one of `answered`, `abstained`, `insufficient_evidence`
- `should_abstain`: boolean
- `must_cite_evidence`: boolean
- `should_preserve_uncertainty`: boolean
- `required_evidence_traits`: non-empty list

Optional expected fields:

- `subtask`: `follow_up`, `comparison`, `theme_label`, `theme_summary`
- `disallowed_patterns`

### Abstention

Required input fields:

- `query`

Required expected fields:

- `correct_behavior`: one of `answer`, `abstain`, `clarify`, `answer_with_caveats`
- `required_markers`: non-empty list
- `disallowed_patterns`: non-empty list
- `should_preserve_uncertainty`: boolean

Optional expected fields:

- `linked_task_family`
- `evidence_quality`

### Provenance

Required input fields:

- `query_context`
- `source_id`

Required expected fields:

- `expected_source_type`: one of `scholarly_article`, `primary_regulatory`, `secondary_regulatory`, `unknown`
- `expected_trust_state`: one of `verified_primary_source`, `verified_metadata`, `unverified`
- `expected_access_state`: one of `full_text_verified`, `abstract_only`, `access_unverified`, `restricted`
- `should_recommend_direct_read`: boolean

Optional expected fields:

- `expected_resolution`: `exact`, `ambiguous`, `missing_session`
- `required_markers`

### Runtime

Required input fields:

- `query_context`

Required expected fields:

- `expected_profile`: one of `guided`, `expert`
- `must_report_configured_provider`: boolean
- `must_report_active_provider`: boolean
- `must_surface_warnings`: boolean
- `must_include_sets`: non-empty list of runtime set field names

### Misc

`misc` is the catch-all family for supporting tasks that are important to the
evaluation program but do not cleanly belong to planner, synthesis,
abstention, provenance, or runtime.

Use it for:

- query expansion
- reranking
- embeddings and fallback honesty
- trace promotion and curation workflows
- judge calibration and evaluator governance

Required input fields:

- `query_context`

Required expected fields:

- `supporting_role`: non-empty string
- `required_markers`: non-empty list of strings
- `disallowed_patterns`: non-empty list of strings
- `must_preserve_cost_awareness`: boolean

## Governance fields and conventions

Use `origin` and `review_status` to make dataset maturity explicit.

- `human`: directly authored by a maintainer or reviewer
- `synthetic_reviewed`: generated synthetically and then reviewed
- `trace_mined`: derived from product traces or regression failures
- `adversarial`: deliberately designed to break brittle logic

Use `review_status` conservatively:

- `draft`: not yet trusted as a gate
- `validated`: reviewed and suitable for canaries
- `needs_review`: needs human attention before continued use

## Reviewed label schema

Promoted trace rows can carry reviewer-state forward in an optional
`review_labels` object.

This is the contract used by:

- `scripts/promote_eval_traces.py`
- `scripts/export_eval_assets.py`
- `scripts/upload_foundry_eval_dataset.py`
- `scripts/upload_hf_eval_assets.py`

Recommended shape:

```json
{
  "review_labels": {
    "verdict": "gold",
    "qualityBucket": "high",
    "split": "train_candidate",
    "trainingEligibility": "approved",
    "trainingObjective": "grounded_synthesis",
    "preferredExportFormats": ["foundry-eval", "hf-dataset", "training-chat"],
    "notes": "Strong reviewed grounded synthesis example."
  },
  "lineage": {
    "traceId": "trace_synthesis_pfas_001",
    "sourceTool": "follow_up_research",
    "reviewedAt": "2026-04-04T08:15:00Z"
  }
}
```

### `review_labels` fields

- `verdict`: nullable string. Recommended values: `gold`, `accepted`,
  `rejected`, `needs_follow_up`.
- `qualityBucket`: nullable string. Recommended values: `high`, `medium`,
  `low`.
- `split`: string describing intended dataset placement. Recommended values:
  `unreviewed`, `train_candidate`, `train`, `dev`, `test`, `holdout`,
  `excluded`.
- `trainingEligibility`: string describing whether a row is allowed into
  training exports. Recommended values: `undecided`, `approved`, `gold`,
  `rejected`.
- `trainingObjective`: nullable string describing the intended training target,
  for example `planner_routing` or `grounded_synthesis`.
- `preferredExportFormats`: list of allowed export surfaces. Current supported
  values are `foundry-eval`, `hf-dataset`, and `training-chat`.
- `notes`: nullable reviewer note.

### `lineage` fields

`lineage` keeps promoted rows auditable.

Recommended fields:

- `traceId`: reviewed trace identifier
- `sourceTool`: tool name that generated the original trace
- `reviewedAt`: timestamp of the human review event, if known
- `batchId`: optional batch identifier when the row came from an expert batch or
  another grouped capture run
- `runId`: optional run identifier for the parent capture execution

### Operational meaning

- `trainingEligibility` is the gate used by `scripts/export_eval_assets.py`
  when producing chat-style training rows.
- `preferredExportFormats` is advisory metadata for downstream tooling. It does
  not override hard validation rules for a given export format.
- `split` should be explicit once a promoted row is treated as part of a stable
  benchmark release.
- `lineage` should be preserved when mirroring rows into Azure AI Foundry or
  Hugging Face so reviewers can trace benchmark slices back to their source.

### Batch and trace context

The repo now emits compact batch and run identifiers in the capture layer for
offline analysis and artifact correlation.

These identifiers are not required for every dataset row, but when a row is
derived from the trace-promotion funnel they should be carried forward when
known:

- `batchId`: groups multiple captured runs from the same offline batch
- `runId`: identifies one execution of the capture or batch runner

They are especially useful when correlating:

- `captured-events.jsonl`
- `review-queue.jsonl`
- `expert-batch-report.json`
- `batch-summary.json`
- `batch-ledger.csv`

## Future extensions

The first schema is intentionally small. If needed later, add optional fields
for:

- `reference_observation`
- `judge_rubric`
- `gold_claims`
- `gold_source_ids`
- `acceptable_alternatives`
- `operational_notes`

Additive fields are preferred over breaking changes.

## Companion tooling

The initial deterministic checker for this schema is:

- `scripts/run_eval_canaries.py`

The reusable validation logic lives in:

- `paper_chaser_mcp/eval_canary.py`

The companion program-level plan is:

- `docs/llm-evaluation-program-plan.md`

The companion platform and trace-curation strategy document is:

- `docs/llm-evaluation-platform-strategy.md`

The companion trace-promotion workflow document is:

- `docs/llm-evaluation-trace-promotion.md`
