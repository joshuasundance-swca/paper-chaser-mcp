# LLM Selection Guide

This guide explains how Paper Chaser MCP uses LLMs today, what each model role
is responsible for, and what to evaluate when choosing or changing models.

The goal is to give maintainers and deployers an intuitive mental model without
tying the documentation to fragile implementation details. For current runtime
truth, treat these code paths as the sources of record:

- `paper_chaser_mcp/settings.py` for public environment variables and defaults
- `paper_chaser_mcp/agentic/config.py` for effective planner/synthesis model
  resolution
- `paper_chaser_mcp/agentic/provider_base.py` for the shared provider-role
  contract
- `paper_chaser_mcp/agentic/provider_openai.py` for the OpenAI and Azure
  OpenAI path
- `paper_chaser_mcp/agentic/provider_langchain.py` for Anthropic, Google,
  Mistral, NVIDIA, and Hugging Face chat-provider paths
- `paper_chaser_mcp/agentic/planner.py` for query classification and variant
  generation
- `paper_chaser_mcp/agentic/graphs.py` for grounded answering, landscape
  mapping, clustering, and expert smart workflows
- `paper_chaser_mcp/dispatch.py` for the guided wrappers that expose planner
  and synthesis behavior through the public MCP surface

## Mental model

The smart layer has three model-facing responsibilities:

| Role | What it does | Main public entrypoints |
| --- | --- | --- |
| Planner | Turns a user request into a search strategy | `research`, `search_papers_smart` |
| Synthesizer | Turns saved evidence into grounded answers or cluster summaries | `follow_up_research`, `ask_result_set`, `map_research_landscape` |
| Supporting model work | Generates speculative query variants and optional embeddings for ranking or clustering | internal to `search_papers_smart`, `ask_result_set`, and landscape mapping |

For the supporting work, the current implementation does not introduce a third
separate chat model. It reuses existing configured roles:

- speculative query expansion uses the configured planner model
- embeddings use the configured embedding model
- similarity scoring uses embeddings only when the active provider supports
  them; otherwise it falls back to lexical scoring with no LLM call

The current design assumes that planning and synthesis are different jobs.
Planning benefits from a fast, reliable model that follows instructions well.
Synthesis benefits from a stronger model that can stay grounded across more
evidence, produce better structured output, and refuse unsupported claims.

## Where model choices come from

Public configuration starts in `paper_chaser_mcp/settings.py` and is mirrored in
`.env.example` and the README configuration section.

The main knobs are:

- `PAPER_CHASER_ENABLE_AGENTIC`
- `PAPER_CHASER_AGENTIC_PROVIDER`
- `PAPER_CHASER_PLANNER_MODEL`
- `PAPER_CHASER_SYNTHESIS_MODEL`
- `PAPER_CHASER_EMBEDDING_MODEL`
- `PAPER_CHASER_DISABLE_EMBEDDINGS`
- `AZURE_OPENAI_PLANNER_DEPLOYMENT`
- `AZURE_OPENAI_SYNTHESIS_DEPLOYMENT`

Effective runtime selection is resolved in `paper_chaser_mcp/agentic/config.py`:

- OpenAI uses the checked-in planner and synthesis defaults directly.
- Azure OpenAI uses the same defaults unless deployment-name overrides are set.
- Anthropic, Google, Mistral, NVIDIA, and Hugging Face auto-swap to provider
  defaults only when the planner and synthesis settings are still at the
  checked-in OpenAI defaults.
- Deterministic mode makes no external LLM calls and reports deterministic
  planner and synthesizer names through runtime metadata.

The effective configured provider and the effective active provider are surfaced
through `selection_metadata()` in `paper_chaser_mcp/agentic/provider_base.py`
and appear in runtime-facing outputs such as `get_runtime_status`.

## Bootstrapping eval traces and lower bounds

The fastest way to bootstrap a useful eval set is not to start with the weakest
model you hope might work. Start with the strongest configuration you trust,
capture good traces, promote those traces into seed eval rows, and then run
progressively smaller planner and synthesis models against the same seed set.

That gives you two related assets:

- a higher-trust trace bank for seed evals and training-style examples
- a lower-bound matrix showing where smaller models stop preserving the same
  observable behavior

The practical sequence is:

1. Run `scripts/run_eval_workflow.py` with a trusted provider and model setup in
   `yolo` or `ui` review mode to collect traces.
2. Promote the reviewed traces into seed datasets.
3. Re-run the same dataset through a matrix of model ladders such as best,
   medium, and small variants for the same provider.
4. Inspect the matrix viewer for divergence in item status, observed intent,
   executed tool sequence, and search-session lineage.

Important interpretation rule: no observed divergence does not prove two models
reasoned the same way. It only means the currently captured eval signals did not
surface a meaningful difference. Two models can still converge on the same
final answer through different retrieval paths or different hidden reasoning.

The workflow wrapper supports custom matrix scenarios in the form:

```text
--matrix-scenario name|provider|planner_model|synthesis_model
```

It also supports canned presets for common ladders, currently including:

- `openai-lower-bound`
- `anthropic-lower-bound`
- `google-lower-bound`
- `nvidia-lower-bound`
- `cross-provider-best`
- `cross-provider-lower-bound`

That makes it possible to run a strong-to-weak ladder such as:

- `openai-best|openai|gpt-5.4-mini|gpt-5.4`
- `openai-mid|openai|gpt-4.1-mini|gpt-4.1`
- `openai-small|openai|gpt-4o-mini|gpt-4o-mini`

Use the strongest rung to create trace-derived seeds. Use the smaller rungs to
measure where quality drops, abstention changes, or tool routing starts to
shift.

To grow the seed corpus itself, the repo now also includes
`scripts/generate_eval_topics.py`. That script uses the configured planner model
to expand one or more seed asks into realistic research-topic and query
candidates, classify likely intent, and optionally emit an expert batch scenario
file for trace capture.

It can also:

- attach topic families and tags from a product-area taxonomy JSON file
- merge and deduplicate prior generated-topic files so repeated bootstrap runs
  accumulate a cleaner candidate pool
- score and rank topics before trace capture so you can keep only the most
  promising candidates

Typical flow:

1. Start with a few realistic user asks or product themes.
2. Generate planner-derived topic candidates and optional batch scenarios.
3. Run `scripts/run_eval_workflow.py` against the generated scenario file using
  the strongest model rung you trust.
4. Promote the stable traces into seeds.
5. Sweep smaller model rungs against the same seeds to find lower bounds.

The taxonomy file accepted by `generate_eval_topics.py` is a JSON array, or an
object with a top-level `rules` array. Each rule may include:

- `family`
- `tags`
- `keywords`
- `intents`

Example:

```json
{
  "rules": [
    {
      "family": "environmental_remediation",
      "tags": ["water", "pfas"],
      "keywords": ["pfas", "groundwater", "remediation"],
      "intents": ["discovery", "review"]
    },
    {
      "family": "ml_benchmarks",
      "tags": ["benchmark", "evaluation"],
      "keywords": ["benchmark", "leaderboard", "evaluation"]
    }
  ]
}
```

The repo also now includes a checked-in starter taxonomy at:

- `tests/fixtures/evals/topic-taxonomy.sample.json`

It also includes two broader preset fixtures:

- `tests/fixtures/evals/topic-taxonomy.balanced-science.sample.json`
- `tests/fixtures/evals/topic-taxonomy.environmental-consulting.sample.json`

`generate_eval_topics.py` uses that file by default unless you override it with
`--taxonomy-file`.

The generator now also supports taxonomy presets directly:

```text
--taxonomy-preset balanced-science
--taxonomy-preset environmental-consulting
```

The default preset is `balanced-science` so topic generation does not drift into
an environment-only eval funnel when you actually want a more representative
cross-domain pool.

The repo also includes checked-in seed starter files for the same two common
bootstrap modes:

- `tests/fixtures/evals/topic-seeds.balanced-science.sample.txt`
- `tests/fixtures/evals/topic-seeds.environmental-consulting.sample.txt`

When you do not provide `--seed-query` or `--seed-file`,
`generate_eval_topics.py` automatically falls back to the balanced-science seed
starter so you can get a representative topic pool with a single command.

Repeated bootstrap runs can be merged with:

```text
--merge-inputs prior-topics-a.json prior-topics-b.json
```

That merge step deduplicates by normalized query and prefers the current run's
topic metadata when the same query appears in multiple files.

Generated topics are also scored and ranked before output. The score reflects a
mix of signals such as:

- novelty relative to the seed ask
- query specificity and usable length
- planner intent richness
- provider-plan depth
- candidate concepts and success criteria
- taxonomy-family and tag coverage

You can filter and trim the ranked pool with:

```text
--min-quality-score 30
--max-topics 25
```

That makes it easier to use the generator as a realistic eval-candidate funnel
instead of a raw prompt explosion tool.

For review workflows, `generate_eval_topics.py` can now also emit ranked topic
exports in JSONL, CSV, and markdown table form:

```text
--jsonl-output ranked-topics.jsonl
--csv-output ranked-topics.csv
--markdown-output ranked-topics.md
```

Those exports are meant for spreadsheet review, annotation queues, or quick
triage outside the primary JSON artifact.

The repo also now includes a lightweight local topic viewer:

```text
python scripts/view_generated_topics.py --input generated-topics.json
```

That viewer shows ranked topics, families, intents, tags, quality scores, and
quality signals so you can inspect why a topic ranked highly before sending it
into the trace-capture workflow.

For the easiest integrated path, `generate_eval_topics.py` can also launch the
viewer directly after generation:

```text
--launch-viewer
```

That requires `--output` to be a file path so the viewer has a persisted JSON
artifact to load.

There is now also an integrated shortcut mode:

```text
--easy-button
```

That mode fills in a sensible default bundle of outputs under
`build/eval-workflow`, enables follow-up scenario generation, raises the minimum
quality threshold, keeps a bounded top-ranked topic set, and launches the topic
viewer unless you instead hand off directly into the workflow.

It also enables two additional optimization passes by default:

- AI-assisted weak-topic rewrite or drop
- family-aware round-robin balancing of the final topic pool

One-command starter example:

```text
python scripts/generate_eval_topics.py \
  --easy-button
```

With no explicit seeds, that command uses the balanced-science starter seed set
and the balanced-science taxonomy preset automatically.

You can also hand off directly into the full eval workflow:

```text
python scripts/generate_eval_topics.py \
  --easy-button \
  --launch-workflow \
  --workflow-matrix-preset cross-provider-lower-bound \
  --workflow-launch-matrix-viewer
```

That path generates the ranked topic pool and batch scenario first, then calls
`run_eval_workflow.py` using the generated scenario file.

There is now also a higher-level orchestration wrapper:

```text
python scripts/run_eval_autopilot.py --profile balanced-science-safe
```

That wrapper gives you one stable entrypoint for:

- profile-driven topic generation
- immutable run-bundle artifact directories
- dry-run inspection of commands and outputs
- exact and near-overlap holdout reporting
- autopilot decision reporting before workflow handoff
- per-stage logs and partial run-state persistence

Each run writes a timestamped bundle containing:

- generated topics in JSON, JSONL, CSV, and markdown
- the generated expert batch scenario file
- an autopilot JSON report and markdown memo
- a run manifest with exact commands and output paths
- a run-state JSON file with stage status and command outcomes
- stage-specific logs for generation and workflow execution
- a holdout report when the selected profile defines a holdout seed file
- workflow artifacts when workflow handoff is allowed and executed

The sample checked-in profile file is:

- `tests/fixtures/evals/eval-autopilot-profiles.sample.json`

The sample holdout seed file is:

- `tests/fixtures/evals/topic-seeds.holdout.sample.txt`

You can inspect the wrapper plan without executing generation or workflow steps:

```text
python scripts/run_eval_autopilot.py \
  --profile balanced-science-safe \
  --dry-run
```

The workflow handoff is now guarded. If the generated pool recommends human
review, the workflow launch is suppressed unless you explicitly force it:

```text
--force-launch-workflow
```

This is intentional. The goal is to make it easy to automate the healthy path
without silently normalizing risky AI-in-the-loop behavior.

Profiles can now also tune the safe-policy thresholds through the profile file's
`workflow.thresholds` object. That makes it possible to keep a conservative
balanced-science bootstrap while defining a separate exploratory profile for
narrower one-seed runs without changing the repository defaults.

The same `workflow.thresholds` object can now also tune selected hard-blocker
thresholds. That means exploratory profiles can relax narrow-run blockers such
as minimum topic count or family dominance in a controlled, explicit way rather
than being silently forced through the global defaults.

The sample profile file now includes both:

- a `single-seed-exploratory-safe` profile for low-friction narrow-run testing
- a `single-seed-diagnostic-force` profile for intentionally forcing workflow
  handoff when you are debugging downstream workflow behavior rather than
  evaluating topic-pool quality

Those narrow-run profiles also enable a dedicated single-seed diversification
pass during topic generation. That pass asks the planner for additional review,
regulatory, and methods-oriented variants so one-seed runs have a better chance
of producing more than one intent or thematic angle.

The generator now also exposes explicit knobs for the two optimization passes:

```text
--ai-prune-mode off|rewrite|rewrite-or-drop
--ai-prune-below-score 35
--domain-balance-mode off|round-robin
--domain-balance-max-share 0.4
```

The pruning pass attempts to rewrite weak topics into stronger, more specific
asks using another planner-assisted pass. In `rewrite-or-drop` mode, topics that
still look weak after that pass are removed before workflow handoff.

Every rewritten or dropped weak topic is now also recorded in `pruneSummary.audit`
so the viewer and downstream review can show before or after state instead of
silently changing the pool.

The balancing pass prevents the final pool from becoming a pure score-ordered
list dominated by one family. `round-robin` tries to preserve stronger topics
while still keeping multiple families visible in the selected set.

The generator also runs a planner-informed family cross-check over the final
topics. This does not replace the assigned family, but it does surface cases
where the assigned family disagrees with the strongest taxonomy-backed
alternative. Those disagreements are included in each topic's
`familyCrossCheck` field and summarized in the pool-level metadata.

## AI-In-The-Loop Risks

This flow is now capable of going from generated topics to workflow execution
with very little human intervention. That is useful, but it creates clear
failure modes.

The main places it can go wrong are:

- domain collapse: one family dominates the generated pool and silently narrows
  eval coverage
- intent collapse: almost everything becomes discovery-style retrieval and
  under-tests known-item, review, or regulatory routing
- low-quality prompt inflation: the LLM generates many plausible-looking but
  weak eval asks that are still syntactically clean
- taxonomy drift: keyword tagging pulls topics into the wrong family because of
  shallow lexical overlap
- automation over-trust: launching the workflow on a weak or narrow topic pool
  can create polished but misleading artifacts

To make those risks visible, the generated topic payload now includes:

- `summary`
- `riskWarnings`
- `reviewRecommendation`
- `pruneSummary`
- `balanceSummary`
- `familyCrossCheckSummary`
- `generationWarnings`
- `hardBlockers`

The current recommendation is intentionally conservative:

- `ai-assisted-ok` only when the pool looks reasonably balanced and not too weak
- `human-review-required` when the pool is too small, too narrow, too low
  quality, or too dominated by one intent or family

That means AI-in-the-loop is supported best as an assistant and accelerator,
not yet as a fully trusted replacement for judgment. If the pool carries risk
warnings, the safer pattern is still:

1. inspect the ranked topic viewer
2. trim or regenerate the pool
3. then launch the trace workflow

The only time to bypass that recommendation is when you explicitly want to test
how the downstream workflow behaves under weak or skewed generated inputs. That
is what `--force-launch-workflow` is for.

For workflow handoff there are now three autopilot policies:

```text
--workflow-autopilot-policy safe|review|blocked
```

- `review`: suppress workflow launch only when the pool recommends human review
- `safe`: require a clean pool with stronger balance, quality, and cross-check
  conditions before launching automatically
- `blocked`: never auto-launch the workflow, even if `--launch-workflow` is set

`safe` is the right default for low-touch automation because it refuses to hand
off when any family cross-check disagreement or other risk warning remains.

The autopilot report now also includes a workflow decision trace with the
evaluated checks, actual values, expected thresholds, and the first failing
reason. That decision trace is the primary diagnostic surface when a run is
suppressed under `safe` or `review` policy.

There is also now a distinct hard-blocker tier in the generated payload. Hard
blockers represent conditions that should suppress workflow handoff regardless
of the softer review recommendation unless you explicitly force the run.

The current hard blockers include cases such as:

- too few surviving topics
- no high-quality topics remaining
- a single family dominating beyond the hard threshold
- too much family cross-check disagreement
- too high a rewritten-topic share

The pruning flow is also now stricter about semantic drift. Rewrites are only
accepted when the inferred intent is preserved; otherwise the candidate rewrite
is rejected and recorded in the prune audit trail.

## Planner role

### What the planner is responsible for

The planner is the model-facing part of search strategy selection. Its job is
not to answer the user directly. Its job is to decide how the system should
search.

In practice, the planner is responsible for:

- classifying query intent such as discovery, review, known-item, author,
  citation, or regulatory
- extracting search constraints such as year, venue, and focus hints
- identifying anchors such as DOI, arXiv id, URL, or regulatory citation
- selecting an initial provider plan
- choosing follow-up mode hints for downstream grounded QA
- proposing bounded speculative query expansions for deeper expert-path search

### Planner entrypoints and call path

Planner behavior is reached from two places:

- Guided `research` through `paper_chaser_mcp/dispatch.py`
- Expert `search_papers_smart` through `paper_chaser_mcp/dispatch.py`

The main internal path is:

1. `dispatch.py` normalizes request arguments.
2. `paper_chaser_mcp/agentic/graphs.py` runs the smart search flow.
3. `paper_chaser_mcp/agentic/planner.py` calls `classify_query()`.
4. The provider bundle calls `aplan_search()` or `plan_search()`.
5. The result becomes a `PlannerDecision` that drives routing and later
   metadata.

The same planner area also owns speculative expansion generation through
`suggest_speculative_expansions()` and `asuggest_speculative_expansions()`.

### Planner guardrails

The planner is intentionally not trusted alone. `paper_chaser_mcp/agentic/planner.py`
adds deterministic guardrails for strong known-item and regulatory signals.
That means:

- explicit mode always wins
- DOI, arXiv id, and URL signals can override weak planner choices
- regulatory markers can override weak planner choices
- the final routed intent carries provenance such as `planner`, `explicit`,
  `heuristic_override`, or `hybrid_agreement`

### What a good planner model must be good at

Choose a planner model that is strong on:

- short, precise instruction following
- stable structured output
- intent classification under ambiguity
- extracting constraints without over-interpreting the query
- selecting sane providers from limited context
- staying cheap and fast enough to run early in the workflow

Choose conservatism over creativity. A planner that is occasionally narrow is
usually safer than one that invents anchors, providers, or query refinements.

### Planner failure modes

Poor planner models tend to fail in a few predictable ways:

- misclassifying known-item requests as broad discovery
- missing regulatory anchors and routing to paper search
- over-expanding the query and causing topic drift
- emitting malformed structured output
- choosing overly broad provider plans that add latency and noise

The code is designed to degrade gracefully here. Both the OpenAI-compatible path
and the LangChain-backed path fall back to deterministic planning when model
calls fail or structured parsing breaks.

## Synthesizer role

### What the synthesizer is responsible for

The synthesizer turns already-retrieved evidence into outputs meant for the
user. It is used for three distinct jobs:

- grounded follow-up answers over a saved session
- expert `ask_result_set` answers
- cluster labeling and cluster summaries in `map_research_landscape`

The synthesizer is not supposed to rediscover the literature. By the time it
runs, the retrieval and session state already exist.

### Synthesis entrypoints and call path

The relevant entrypoints are:

- Guided `follow_up_research` in `paper_chaser_mcp/dispatch.py`
- Expert `ask_result_set` in `paper_chaser_mcp/dispatch.py`
- Expert `map_research_landscape` in `paper_chaser_mcp/dispatch.py`

Guided `follow_up_research` still depends on `ask_result_set` for grounded
synthesis, but the wrapper now classifies the question shape itself and
preserves explicit metadata / relevance-triage introspection semantics before
choosing that path.

The main internal methods are defined by the provider bundle contract in
`paper_chaser_mcp/agentic/provider_base.py`:

- `answer_question()` / `aanswer_question()`
- `label_theme()` / `alabel_theme()`
- `summarize_theme()` / `asummarize_theme()`

The orchestration lives in `paper_chaser_mcp/agentic/graphs.py`.

### What a good synthesis model must be good at

Choose a synthesis model that is strong on:

- grounded multi-document reasoning
- structured JSON output that survives schema validation
- selecting evidence ids correctly
- refusing unsupported claims instead of answering confidently anyway
- summarizing several papers without flattening real uncertainty
- following tight output constraints for answerability and confidence labels

This role benefits more from raw capability than the planner role does. A
synthesis model is doing the user-facing trust work.

### Synthesis failure modes

Poor synthesis models usually fail by:

- inventing support that is not in the evidence set
- choosing invalid evidence ids
- returning malformed schema output
- collapsing nuanced evidence into overconfident prose
- ignoring abstention cues when the evidence is weak

The runtime mitigates these risks by validating selected evidence ids,
normalizing confidence labels, downgrading answerability when support is weak,
and falling back to deterministic behavior when needed.

## Other model-mediated work

Not every LLM-related behavior fits neatly into planner versus synthesizer.
These supporting uses still matter for model choice.

### Speculative query expansion

`paper_chaser_mcp/agentic/planner.py` and the provider bundles use model calls
to suggest bounded speculative query variants. This is part of the smart search
fanout logic, not the grounded answer contract.

The model used here is the configured planner model, not the synthesis model.
In practice that means:

- OpenAI and Azure OpenAI call expansion generation with `planner_model_name`
- Anthropic, Google, Mistral, NVIDIA, and Hugging Face use their configured
  planner-side chat model through `provider_langchain.py`
- deterministic mode uses no LLM and falls back to heuristic expansion logic

This work needs models that are good at:

- proposing plausible adjacent wording without drifting too far
- staying concise and deduplicable
- respecting max-variant limits

### Optional embeddings

Embeddings are configured through `PAPER_CHASER_EMBEDDING_MODEL`, but disabled
by default with `PAPER_CHASER_DISABLE_EMBEDDINGS=true`.

The model used here is the configured embedding model, which is separate from
the planner and synthesis chat models.

Today:

- the OpenAI and Azure OpenAI paths can supply embeddings through
  `PAPER_CHASER_EMBEDDING_MODEL`
- the LangChain chat-provider path remains chat-only in this repo and does not
  use a separate embedding model
- if embeddings are unavailable or disabled, lexical similarity and
  deterministic clustering remain the fallback

Embeddings influence reranking, workspace retrieval, and some clustering
choices, but the codebase intentionally keeps the system usable without them.

That means the current non-planner, non-synthesis model usage breaks down like
this:

| Supporting task | Model used today |
| --- | --- |
| speculative expansion generation | planner model |
| query-to-paper similarity scoring with embeddings enabled | embedding model |
| query-to-paper similarity scoring with embeddings disabled or unsupported | no LLM; lexical fallback |
| workspace vector indexing when enabled | embedding model |
| clustering or reranking when embeddings are unsupported | no LLM; deterministic or lexical fallback |

### Runtime truth and fallback visibility

Runtime and diagnostics surfaces are not separate model roles, but they are how
users learn what happened in production. `selection_metadata()` and the guided
runtime surfaces communicate:

- configured provider versus active provider
- planner model and synthesis model
- where each model choice came from
- whether deterministic fallback took over

These diagnostics do not use a separate model. They report which of the
existing configured models ran, and whether the runtime had to fall back to
deterministic behavior instead.

That metadata is part of the user-facing trust contract and should stay aligned
with any future model changes.

## Current default model resolution

These are the checked-in effective defaults described in
`paper_chaser_mcp/agentic/config.py` and the README.

| Provider | Planner default | Synthesis default | Notes |
| --- | --- | --- | --- |
| `openai` | `gpt-5.4-mini` | `gpt-5.4` | Direct checked-in defaults |
| `azure-openai` | `gpt-5.4-mini` | `gpt-5.4` | Deployment-name overrides can replace both |
| `anthropic` | `claude-haiku-4-5` | `claude-sonnet-4-6` | Auto-swapped only when OpenAI defaults are still untouched |
| `google` | `gemini-2.5-flash` | `gemini-2.5-pro` | Auto-swapped only when OpenAI defaults are still untouched |
| `mistral` | `mistral-medium-latest` | `mistral-large-latest` | Auto-swapped only when OpenAI defaults are still untouched |
| `nvidia` | `nvidia/nemotron-3-nano-30b-a3b` | `nvidia/nemotron-3-super-120b-a12b` | Auto-swapped only when OpenAI defaults are still untouched |
| `huggingface` | `moonshotai/Kimi-K2.5` | `moonshotai/Kimi-K2.5` | Chat-only path in this repo; planner and synthesis currently use the same default |
| `deterministic` | n/a | n/a | No external LLM calls |

## How to think about choosing models

### Choosing a planner model

Favor a planner model when it is:

- lower latency than your synthesis model
- cheaper per request
- reliable at structured outputs
- strong at classification and constraint extraction
- conservative under ambiguity

Planner models do not need to be the strongest reasoning models in the stack.
They need to be fast, consistent, and boring in the best possible way.

### Choosing a synthesis model

Favor a synthesis model when it is:

- better at long-context grounded reasoning
- stronger at schema-following under pressure
- better at abstaining honestly
- better at multi-paper comparison and nuanced summaries
- still predictable enough that deterministic fallback is rare rather than
  normal

If you only spend extra capability budget in one place, synthesis is usually
the better place to spend it.

### Choosing whether to enable embeddings

Enable embeddings only if you specifically need the additional ranking or
clustering quality and have verified the provider path supports them.

Keep them disabled when:

- deterministic fallback quality is already acceptable
- your provider path is chat-only in this repo
- reliability matters more than small ranking gains

## External fit notes on the current defaults

External vendor documentation reviewed in April 2026 broadly supports the role
split used by this repo:

- fast or mini models are generally marketed for routing, volume, and low
  latency work
- stronger or pro or large models are generally marketed for deeper reasoning,
  synthesis, and more complex agent behavior

That said, a few defaults deserve extra scrutiny before future releases:

- the Anthropic default names should be re-verified against current public model
  ids before relying on them as durable defaults
- the NVIDIA Nemotron ids should be re-verified against the current hosted or
  NIM catalog actually used in deployment
- the Hugging Face default currently uses the same Kimi model for planner and
  synthesis, which weakens the intended role split even if the path remains
  functional

This is a good place to be opinionated: if a provider cannot support a clean
planner versus synthesis split in practice, document that clearly instead of
pretending all providers are equally mature in this repo.

## Maintenance rules for this document

When model behavior changes, update this guide together with:

- `README.md`
- `.env.example`
- `paper_chaser_mcp/settings.py`
- `paper_chaser_mcp/agentic/config.py`
- any user-facing runtime metadata or diagnostics docs affected by the change

Prefer updating code paths, role descriptions, defaults, and decision rules.
Avoid line-number references or copying implementation details that are likely
to drift.
