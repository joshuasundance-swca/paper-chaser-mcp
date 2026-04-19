# LLM Evaluation Integrations

This document records the near-term integration posture for Azure AI Foundry and
Hugging Face around the Paper Chaser evaluation and curation funnel.

## Design stance

Prefer this order:

1. portable local capture and review artifacts
2. portable local telemetry and batch-summary artifacts
3. portable export surfaces
4. optional service-specific upload or sync steps

That keeps the core workflow usable even when cloud credentials or external
SDKs are unavailable.

Current local-first artifact posture:

- raw captured events in JSONL
- review queues in JSONL
- promoted rows in portable JSONL export formats
- batch summary and ledger artifacts for offline drift and throughput analysis

The batch-summary and ledger artifacts are intentionally local and credential
free. They exist to make the curation funnel inspectable even when no cloud
evaluation platform is involved.

## Azure AI Foundry

High-value immediate opportunities:

- Export promoted eval rows into a Foundry-friendly JSONL shape using
  `scripts/export_eval_assets.py --format foundry-eval`
- Upload exported JSONL directly into a Foundry project dataset using
  `scripts/upload_foundry_eval_dataset.py`
- Register role-based eval datasets in Foundry for batch evaluation and model
  comparison
- Create custom evaluators aligned to repo rubrics for planner, synthesis,
  abstention, provenance, and runtime truth
- Use Foundry project datasets as the cloud-scale mirror of local reviewed eval
  banks, not as the only source of truth

Good insertion points in this repo:

- `scripts/run_eval_canaries.py`
- `scripts/run_expert_eval_batch.py`
- `scripts/export_eval_assets.py`
- `scripts/upload_foundry_eval_dataset.py`
- `scripts/promote_eval_traces.py`
- `tests/fixtures/evals/judge-rubrics.json`

Current documented upload path:

- Construct `AIProjectClient(endpoint=..., credential=DefaultAzureCredential())`
- Use `project_client.datasets.upload_file(name=..., version=..., file_path=...)`
  for JSONL exports
- Use `project_client.datasets.upload_folder(...)` for folder-backed exports

Install the required SDKs when using the upload helper:

```bash
pip install -e ".[eval-foundry]"
```

Equivalent direct package install:

```bash
pip install azure-ai-projects azure-identity
```

Example:

```bash
python scripts/upload_foundry_eval_dataset.py \
  --input promoted-foundry.jsonl \
  --dataset-name paper-chaser-evals \
  --dataset-version 2026.04.04 \
  --project-endpoint "$AZURE_AI_PROJECT_ENDPOINT"
```

Recommended posture:

- Local capture and review remain the authoritative curation loop.
- Local batch summaries and ledgers remain the first drift-inspection layer.
- Foundry becomes the scalable batch-eval and comparison layer.

## Hugging Face

High-value immediate opportunities:

- Export review-approved rows into HF-dataset-friendly JSONL using
  `scripts/export_eval_assets.py --format hf-dataset`
- Export approved training rows into chat-style JSONL using
  `scripts/export_eval_assets.py --format training-chat`
- Use HF Datasets as the versioned public or private distribution layer for
  reviewed eval banks
- Push reviewed exports into a dataset repo or bucket using
  `scripts/upload_hf_eval_assets.py`

### hf-mount

`hf-mount` is relevant here.

It mounts Hugging Face Buckets and Hub repos as local filesystems. Based on the
current public docs and changelog, it is a strong fit for:

- mounting a writable HF Bucket as the sink for live eval-candidate capture
- mounting a large read-only dataset repo locally for review or training export
- agentic or filesystem-oriented workflows that benefit from `ls`, `cat`,
  `find`, and lazy reads instead of custom SDK logic

Why it matters for this repo:

- the capture funnel already writes JSONL artifacts to a filesystem path
- `hf-mount` can make that path an HF Bucket mount instead of a purely local
  directory
- that means capture artifacts can flow into shared remote storage without
  changing the repo's core file-oriented design

Where it fits best:

- live capture sink: good fit
- large read-heavy reviewed datasets: good fit
- high-consistency multi-writer transactional storage: poor fit

Important caveats from current hf-mount docs:

- eventual consistency rather than strict consistency
- read-heavy workloads are the best fit
- Buckets are read-write, but repos are read-only in this pattern
- first reads incur network latency
- not ideal for heavy concurrent writers or correctness-critical file-locking

Recommended posture:

- Use local paths by default.
- Optionally point `PAPER_CHASER_EVAL_TRACE_PATH` at an hf-mount bucket mount
  when shared remote capture is useful.
- Keep reviewed promotion and export steps explicit rather than automatically
  mutating versioned dataset repos.
- Treat `batch-summary.json` and `batch-ledger.csv` as local operator artifacts,
  not as replacements for the reviewed dataset exports.

Current documented upload paths:

- Dataset repos: `create_repo(..., repo_type="dataset")` followed by
  `upload_file(...)` or `upload_folder(...)`
- Buckets: `create_bucket(...)` followed by `batch_bucket_files(...)` for
  single-file uploads or `sync_bucket(...)` for directory syncs

Install the required SDK when using the upload helper:

```bash
pip install -e ".[eval-huggingface]"
```

Equivalent direct package install:

```bash
pip install huggingface_hub
```

Important distinction:

- `.[ai,huggingface]` is the smart-provider chat-router extra
- `.[eval-huggingface]` is the eval publish helper extra

They solve different problems and do not replace each other.

Example dataset repo upload:

```bash
python scripts/upload_hf_eval_assets.py dataset-repo \
  --input reviewed-hf.jsonl \
  --expected-format hf-dataset \
  --repo-id my-org/paper-chaser-evals
```

Example bucket upload:

```bash
python scripts/upload_hf_eval_assets.py bucket \
  --input reviewed-traces.jsonl \
  --bucket-id my-org/paper-chaser-eval \
  --remote-path review/reviewed-traces.jsonl
```

Example flow with `hf-mount`:

```bash
hf-mount start --hf-token "$HF_TOKEN" bucket my-org/paper-chaser-eval /mnt/paper-chaser-eval

export PAPER_CHASER_ENABLE_EVAL_TRACE_CAPTURE=true
export PAPER_CHASER_EVAL_TRACE_PATH=/mnt/paper-chaser-eval/live/captured-events.jsonl

python scripts/build_eval_review_queue.py \
  --input /mnt/paper-chaser-eval/live/captured-events.jsonl \
  --output /mnt/paper-chaser-eval/review/review-queue.jsonl

python scripts/export_eval_assets.py \
  --input /mnt/paper-chaser-eval/review/reviewed-traces.jsonl \
  --output - \
  --format hf-dataset
```

The final export can stay on stdout for piping into later upload or packaging
steps, which is useful when you do not want more local scratch files.

## Practical rollout order

1. Keep local JSONL capture and review queue as the default.
2. Generate local summary and ledger artifacts for offline drift inspection.
3. Use `hf-mount` bucket mounts for shared capture sinks where needed.
4. Export reviewed rows to Foundry eval JSONL and HF dataset JSONL.
5. Upload reviewed exports into Foundry datasets or Hugging Face repos/buckets
  with the service-specific scripts.
