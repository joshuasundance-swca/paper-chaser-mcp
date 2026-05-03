# LLM Evaluation Trace Promotion

This document defines the first trace-promotion workflow for Paper Chaser MCP.

The goal is to convert reviewed real traces into durable evaluation rows without
letting benchmarks silently drift or accumulate unlabeled production artifacts.

## Purpose

Trace promotion is the bridge between:

- real runtime behavior
- human review of what went right or wrong
- durable evaluation assets used in future regressions and model comparisons

## Workflow

1. Capture a live trace or reviewed run.
2. Review it against the appropriate rubric.
3. Decide whether it should be promoted.
4. Record the expected behavior and why it matters.
5. Convert it into a role-based evaluation row.

## Optional live instrumentation

The repo now supports optional eval-candidate capture directly from live guided
tool runs.

Relevant env vars:

- `PAPER_CHASER_ENABLE_EVAL_TRACE_CAPTURE=true`
- `PAPER_CHASER_EVAL_TRACE_PATH=...`

When enabled, guided `research`, `follow_up_research`, `inspect_source`, and
`get_runtime_status` calls can emit compact JSONL events that are designed for a
curation funnel rather than low-level provider debugging.

The capture layer now also supports expert smart-tool surfaces:

- `search_papers_smart`
- `ask_result_set`
- `map_research_landscape`
- `expand_research_graph`

Those events are intentionally smaller and more review-friendly than raw
provider traces. They keep the parts that matter for eval and training curation:

- normalized user input
- compact guided output state
- structured execution provenance
- source identifiers and trust state
- runtime truth snapshots

The current event envelope is also intended to support offline telemetry and
batch artifact generation rather than only human review.

Current event-level metadata includes:

- `eventId`
- `timestamp`
- `eventType`
- `searchSessionId`
- `runId`
- `batchId`
- `durationMs`

Current compact output telemetry can include:

- `stageTimingsMs`
- `providerOutcomes`
- `providerPathwaySummary`
- `confidenceSignals`
- `failureSummary`
- `abstentionDetails`

For follow-up traces, the capture layer also preserves `evidenceUsePlan` when
it is present. For terminal non-success outcomes, captured events can include
ranking and synthesis diagnostics such as `rankingDiagnostics`,
`preFilterCandidates`, `scoreBreakdown`, `classificationProvenance`,
`synthesisMode`, and `evidenceQualityProfile`.

This is intentionally not a full low-level trace. It is the compact curation
surface that feeds triage, promotion, and offline reporting.

## Review queue builder

Use:

- `scripts/build_eval_review_queue.py`

to convert captured live eval events into a review queue JSONL file. That queue
is the human-review stage before promotion.

The review queue now includes a lightweight label block so reviewers can decide:

- verdict and quality bucket
- holdout or train-candidate split
- training eligibility
- preferred export formats

The queue also preserves enough execution context for triage without reopening
the raw event file:

- `run_id`
- `batch_id`
- trace-level `duration_ms`
- trace-level `telemetry.provider_pathway_summary`
- trace-level `telemetry.stage_timings_ms`
- trace-level `telemetry.confidence_signals`
- trace-level `telemetry.failure_summary`

When present in the captured trace, promotion can now also preserve
`evidenceUsePlan` and the ranking-diagnostics bundle into the promoted eval
row so later tuning and regression analysis do not lose the original decision
signals.

## Sample reviewed trace format

Store reviewed traces as JSONL with one record per line.

Each row should include:

- `trace_id`
- `reviewed_at`
- `trace`
- `review`

The `trace` object stores the useful runtime context.

The `review` object stores:

- `promote`
- `task_family`
- `id`
- `tags`
- `expected`
- `why_it_matters`
- optional `notes`

The sample captured-event and review-queue inputs live at:

- `tests/fixtures/evals/captured-eval-events.sample.jsonl`
- `tests/fixtures/evals/trace-promotion.sample.jsonl`

## Batch artifacts

The expert batch runner now supports a richer offline artifact set.

Use:

- `scripts/run_expert_eval_batch.py`

Current artifact set:

- `expert-batch-report.json`: raw scenario outcomes plus top-level `batchId`,
  `runId`, `generatedAt`, and `scenarioFile`
- `captured-events.jsonl`: compact event stream for curation and telemetry
- `review-queue.jsonl`: reviewer-facing queue derived from captured events
- `batch-summary.json`: aggregate counters and timings for the batch
- `batch-ledger.csv`: flat per-run ledger for spreadsheet or dashboard use

The current summary artifact is designed for cheap offline comparisons and
artifact inspection. It currently includes:

- `batchId`
- `runId`
- `generatedAt`
- `scenarioFile`
- `toolCounts`
- `taskFamilyCounts`
- `runCount`
- `capturedEventCount`
- `reviewQueueRowCount`
- `totalDurationMs`
- `maxDurationMs`
- `providerAttemptCount`
- `fallbackCount`
- `totalRetries`
- `abstentionCount`
- `warningCount`
- `schemaVersion`

The current ledger artifact is meant to be easy to join or visualize without
reparsing nested JSON:

- `batchId`
- `runId`
- `scenarioName`
- `tool`
- `taskFamily`
- `searchSessionId`
- `capturedEventId`
- `reviewQueueRowId`
- `answerStatus`
- `resultStatus`
- `durationMs`
- `sourceCount`
- `providerCount`
- `fallbackCount`
- `totalRetries`

This keeps raw artifacts and derived metrics separate:

- raw JSON and JSONL are for inspection and promotion
- summary JSON is for fast drift checks
- ledger CSV is for dashboard-friendly offline analysis

## Promotion helper

Use:

- `scripts/promote_eval_traces.py`
- `scripts/build_eval_review_queue.py`

Portable export helpers now exist for downstream systems:

- `scripts/export_eval_assets.py --format foundry-eval`
- `scripts/export_eval_assets.py --format hf-dataset`
- `scripts/export_eval_assets.py --format training-chat`

These exports are intentionally file-based and portable. They prepare records for
Azure AI Foundry and Hugging Face workflows without forcing the repo to depend on
either SDK in the core runtime.

If you want the capture sink itself to live on Hugging Face storage, `hf-mount`
is a practical option: mount an HF Bucket locally and point
`PAPER_CHASER_EVAL_TRACE_PATH` at a JSONL file inside that mounted path.
That preserves the repo's file-oriented capture flow while moving storage to a
shared remote bucket.

It reads reviewed-trace JSONL and writes promoted evaluation JSONL rows using
`origin=trace_mined`.

Promotion should continue to treat the trace-derived event and batch metadata as
lineage context, not as a substitute for human review state.

## Governance rules

- Promote only reviewed traces.
- Keep lineage back to the original trace id.
- Prefer small high-value promotions over bulk dumping raw traces into the eval bank.
- Use promoted traces to refresh recent-failure datasets while keeping a stable
  frozen regression set alongside them.
