# LLM Evaluation Program Plan

This document lays out a rigorous, maintainable plan for evaluating LLM
performance in Paper Chaser MCP across different roles, generating evaluation
datasets, and operating an evaluation program over time.

The primary target of this evaluation program is the LLM serving under the hood
inside Paper Chaser, not an external client model deciding which public MCP
tool to call.

Companion documents and first artifacts now live at:

- `docs/llm-evaluation-dataset-schema.md`
- `docs/llm-evaluation-platform-strategy.md`
- `tests/fixtures/evals/`
- `scripts/run_eval_canaries.py`

It is intentionally written as a planning and design document, not as an
implementation spec tied to fragile line-level details. For current runtime
truth and existing evaluation surfaces, use these code and doc paths as the
sources of record:

- `paper_chaser_mcp/dispatch.py`
- `paper_chaser_mcp/agentic/planner.py`
- `paper_chaser_mcp/agentic/graphs.py`
- `paper_chaser_mcp/agentic/ranking.py`
- `paper_chaser_mcp/agentic/provider_base.py`
- `paper_chaser_mcp/agentic/provider_openai.py`
- `paper_chaser_mcp/agentic/provider_langchain.py`
- `tests/test_dispatch.py`
- `tests/test_smart_tools.py`
- `tests/test_agentic_providers_extra.py`
- `tests/test_agentic_workflow.py`
- `tests/fixtures/ux_prompt_corpus.json`
- `tests/fixtures/provider_benchmark_corpus.json`
- `docs/golden-paths.md`
- `docs/provider-upgrade-program.md`
- `docs/agent-handoff.md`

## Goals

The evaluation program should answer four questions reliably:

1. Is each LLM-backed role doing the right job?
2. Is the full guided or expert workflow behaving safely and usefully?
3. Are model, prompt, or provider changes actually improvements?
4. Is production behavior staying aligned with the intended contract over time?

For this repo, the evaluation program should prioritize trust and safe failure
over answer-shaped optimism. The product contract already depends on grounded
evidence, safe abstention, explicit ambiguity handling, and runtime truth.
Evaluation should reinforce those properties.

## Evaluation target

The main system under test is the internal model behavior inside the Paper
Chaser runtime.

That means the evaluation program should primarily measure whether the chosen
planner, synthesis, and supporting models can:

- emit valid structured outputs when the runtime expects them
- make good routing and judgment calls
- preserve groundedness and uncertainty correctly
- select or justify evidence correctly
- provide enough domain knowledge to support search planning and synthesis
  without inventing unsupported claims

This is different from evaluating whether an external agent can discover and use
the public guided tools well.

That external-agent question still matters for product UX, and the repo already
has some coverage for it through guided-contract and workflow tests. But the
primary LLM-evaluation question for model choice and training is whether the
internal provider-backed LLM can serve its runtime role well.

In the current architecture, much of provider invocation and tool orchestration
is application-managed rather than free-form model tool calling. So the main
native eval targets today are structured outputs, routing quality, grounded
judgment, evidence handling, and safe failure behavior. If the internal runtime
later introduces more model-driven tool-calling decisions, that should become a
separate explicitly scored capability.

When running live native evaluations, the right default is to test the model in
the currently configured runtime rather than pretending every provider or tool
is always available. If you want to compare provider-backed model behavior,
evaluate across multiple runtime configurations deliberately instead of forcing a
single run to impersonate all configurations at once.

## Scope By Role

Paper Chaser MCP has several distinct evaluation targets.

| Role | Main responsibility | Main surfaces |
| --- | --- | --- |
| Planner | classify intent, pick providers, extract anchors and constraints, shape search strategy | `research`, `search_papers_smart` |
| Synthesizer | answer grounded follow-ups, label themes, summarize clusters | `follow_up_research`, `ask_result_set`, `map_research_landscape` |
| Trust gating | decide when evidence is too weak, off-topic, or ambiguous | guided wrappers and smart-tool result shaping |
| Supporting retrieval logic | query expansion, reranking, embedding-backed similarity, candidate merging | smart search internals and ranking utilities |
| Source/provenance reasoning | inspect sources, preserve evidence and lead separation, communicate direct-read next actions | `inspect_source`, guided contract fields |
| Runtime truth and fallback reporting | report configured vs active provider, profile, fallback path, and warnings accurately | `get_runtime_status`, provider diagnostics |

These roles should not be collapsed into one generic quality score. Each role
needs its own evaluation logic, and end-to-end workflow evaluation should sit on
top of that role-level foundation.

## Existing Evaluation Foundation

The repo already has useful building blocks, but they are not yet a full LLM
evaluation program.

Current strengths:

- fixture-based prompt and benchmark corpora already exist in
  `tests/fixtures/ux_prompt_corpus.json` and
  `tests/fixtures/provider_benchmark_corpus.json`
- guided wrapper behavior, abstention semantics, regulatory routing, and
  runtime truth are already tested in `tests/test_dispatch.py`,
  `tests/test_smart_tools.py`, and `tests/test_agentic_workflow.py`
- provider fallback behavior and provider selection metadata are already tested
  in `tests/test_agentic_providers_extra.py`
- the checked-in workflow harness already acts as a product-facing regression
  surface for guided and expert flows
- the repo now has a portable trace-curation funnel with:
  - live eval-candidate capture in `paper_chaser_mcp/eval_curation.py`
  - human-review queue generation in `scripts/build_eval_review_queue.py`
  - trace promotion in `scripts/promote_eval_traces.py`
  - portable export helpers in `scripts/export_eval_assets.py`
  - service-specific publish helpers in `scripts/upload_foundry_eval_dataset.py`
    and `scripts/upload_hf_eval_assets.py`
  - expert batch artifact generation in `scripts/run_expert_eval_batch.py`

Current gaps:

- little live or role-specific quality evaluation of planner decisions
- little live or role-specific quality evaluation of synthesis correctness and
  groundedness
- limited evaluation of embeddings, reranking, and speculative expansion as
  distinct quality contributors
- no formal judge-calibration process
- no dedicated dataset governance, versioning, or drift management plan
- no formal online or shadow-evaluation plan

Near-term note:

The repo now has first-pass offline telemetry and batch artifacts, but it still
does not have the full governance layer described later in this plan. In
particular, queue prioritization, reviewer-agreement modeling, dedupe,
contamination checks, and stricter split enforcement remain follow-on work.

## Evaluation Principles

The evaluation program should follow these principles.

### Evaluate roles separately and together

Every important change should be observable at two levels:

- component level: planner, synthesis, trust gating, reranking, provenance
- workflow level: complete guided and expert user paths

Good end-to-end performance can hide component regressions. Good component
scores can also fail to translate into better workflows. Both views are needed.

### Prefer role-correctness over fluent output

For this repo, fluent but unsupported output is a failure. That is especially
important for:

- grounded follow-up
- regulatory timelines
- citation and source inspection
- runtime truth reporting

### Use multiple evaluator types

No single evaluation method is sufficient. Use a layered stack:

- deterministic checks where possible
- retrieval and ranking metrics where applicable
- rubric-based LLM judges for nuanced generation and planning quality
- pairwise judging for selection between variants
- human audit for calibration, policy-sensitive cases, and edge cases

### Separate offline, shadow, and online evaluation

Offline evaluation is for repeatability and iteration.
Shadow evaluation is for validating candidates on real traffic without user
impact.
Online monitoring is for drift, regression, and operational safety after
deployment.

The current implementation is strongest in the offline layer:

- deterministic canaries
- portable reviewed-trace promotion
- batch-summary and ledger artifacts for offline drift inspection

Shadow and online evaluation are still planned surfaces rather than complete
runtime features.

## Evaluation Methods By Role

### Planner

Treat the planner as a decision system, not a prose generator.

Primary questions:

- can it emit valid structured planner outputs consistently?
- did it choose the right intent?
- did it extract the right anchors and constraints?
- did it choose a sane provider plan?
- did it overuse expensive or unnecessary routes?
- did it fail safely when the query was ambiguous?

Recommended offline metrics:

- structured-output validity rate
- intent accuracy or macro-F1
- top-k acceptable-route recall when more than one route is valid
- anchor extraction accuracy for DOI, arXiv, URL, and regulatory citations
- abstain or clarify precision and recall on ambiguous cases
- provider-plan quality score based on acceptable route sets, not single exact
  route strings
- plan efficiency penalties for unnecessary escalation or provider fanout

Recommended dataset slices:

- clean known-item requests
- ambiguous literature versus regulatory requests
- mixed-intent requests
- queries missing required clarification
- adversarial queries designed to trigger over-routing or wrong routing

### Synthesizer

Treat synthesis as grounded claim construction over saved evidence.

Primary questions:

- can it emit valid structured answer payloads when required?
- does the answer stay inside the supplied evidence?
- are selected evidence ids valid and relevant?
- does the answer abstain when the evidence is weak?
- does the answer preserve uncertainty honestly?
- do theme labels and summaries accurately reflect cluster content?

Recommended offline metrics:

- structured-output validity rate
- claim-level groundedness or faithfulness
- unsupported-claim rate
- answer relevance to the actual follow-up question
- evidence-id selection validity
- abstention precision and recall
- citation or evidence support rate
- pairwise preference against baseline for theme labels and summaries when exact
  gold labels are unrealistic

Recommended dataset slices:

- strong evidence, single-theme follow-ups
- multi-paper comparison tasks
- weak evidence tasks that should abstain
- contradictory evidence tasks
- noisy evidence bundles with distractors
- cluster summarization sets with human-reviewed summaries

### Trust Gating

Trust gating deserves explicit evaluation instead of being treated as a side
effect of synthesis.

Primary questions:

- did the system put the right items in evidence versus leads?
- did it abstain when evidence was weak or off-topic?
- did it ask for clarification instead of bluffing?
- did it preserve the intended public contract fields?

Recommended metrics:

- abstention precision and recall
- evidence bucket precision
- lead bucket recall for weak but potentially useful items
- ambiguity-handling correctness
- rate of answer-shaped filler on insufficient-evidence cases

### Supporting Retrieval Logic

Evaluate supporting roles by their effect on downstream quality, not by whether
their intermediate text looks plausible.

For supporting roles, the main under-the-hood question is not whether the model
can call public MCP tools. It is whether the internal model behavior improves
retrieval, ranking, clustering, or summaries in the ways Paper Chaser actually
uses it.

#### Query expansion

Metrics:

- delta Recall@k versus baseline retrieval
- delta MRR or nDCG versus baseline retrieval
- semantic drift rate
- redundancy rate
- added cost and latency per useful gain

#### Reranking and similarity scoring

Metrics:

- nDCG@k
- MRR@k
- Recall@k
- lift over first-stage ranking
- stability across repeated runs when stochastic providers are involved

#### Embeddings

Metrics:

- retrieval lift versus lexical fallback
- ranking quality by query slice
- operational reliability by provider and profile

### Source And Provenance Reasoning

For `inspect_source` and related provenance behavior, evaluate:

- source-resolution correctness
- trust-state correctness
- direct-read recommendation usefulness
- claim-to-source support quality
- handling of ambiguous or conflicting source identifiers

### Runtime Truth And Fallback Reporting

This role is not about content quality. It is about system truthfulness.

Evaluate:

- configured provider versus active provider consistency
- deterministic fallback visibility
- effective profile visibility
- warning accuracy
- internal consistency between top-level summaries and provider rows

## Dataset Program

The dataset program should combine curated, synthetic, trace-mined, and
adversarial data rather than depending on only one source.

### Dataset families

At minimum, maintain separate dataset families for:

1. planner and routing
2. grounded follow-up and synthesis
3. abstention and ambiguity handling
4. source inspection and provenance
5. retrieval support tasks such as expansion and reranking
6. runtime truth and fallback reporting

In the current repo artifacts, the first-pass seed layout uses:

- `planner`
- `synthesis`
- `abstention`
- `provenance`
- `runtime`
- `misc`

`misc` is the explicit bucket for supporting tasks and evaluation-operations
concerns that do not fit neatly into the other role families yet.

### Data sources

Use four data sources in parallel:

- human-authored seed sets for high-quality core examples
- production trace mining for realism and failure discovery
- synthetic expansion from strong models with review for coverage
- adversarial sets for robustness and safety

### Dataset generation strategy

Use this workflow:

1. Start from real product tasks, prompts, or incident patterns.
2. Write a small human-reviewed seed set for each task family.
3. Use synthetic generation to expand coverage around the seeds.
4. Review a calibrated sample of synthetic items with humans.
5. Promote accepted items into versioned datasets.
6. Continuously add new failures harvested from tests and production traces.

### Labeling guidance

Store labels at the smallest meaningful unit.

Examples:

- planner examples should store acceptable routes and unacceptable routes, not
  just one exact label
- synthesis examples should store claim-level support and required evidence ids
- abstention examples should store whether the correct behavior is answer,
  abstain, or clarify
- source inspection examples should store expected trust state and next action

### Dataset governance

Every dataset release should have:

- version number
- task family
- origin such as `human`, `synthetic_reviewed`, `trace_mined`, or `adversarial`
- review status
- schema version
- intended use and non-use
- notes on known limitations

Maintain two broad dataset classes:

- frozen regression sets that change rarely
- living capability sets that absorb new failures and new slices over time

## Evaluator Stack

The evaluator stack should be layered and role-aware.

### Deterministic checks

Use deterministic checks first wherever possible.

Examples:

- schema validity
- required fields present
- valid evidence ids only
- route or tool constraint checks
- runtime field consistency
- exact identifier normalization

### Retrieval metrics

Use standard IR metrics for retrieval-support tasks.

Examples:

- Recall@k
- MRR@k
- nDCG@k
- coverage of gold evidence

### LLM-as-judge

Use LLM judges for nuanced evaluation, but do not treat them as ground truth.

Recommended use:

- rubric-based scoring for grounded synthesis
- rubric-based scoring for planner quality when exact route matching is too
  brittle
- pairwise judging for model or prompt comparisons

Required safeguards:

- calibrate against human labels on a recurring audit set
- keep judge prompts criterion-specific
- use pairwise judging when choosing between candidate systems
- maintain anchor sets to detect judge drift

### Human review

Use human review for:

- judge calibration
- high-risk or policy-sensitive examples
- ambiguous routing cases
- provenance and trust-sensitive slices
- sampled failures before major releases

## Experimental Design

When comparing model, prompt, or provider changes:

- change one variable at a time when possible
- evaluate at both role level and workflow level
- compare on frozen regression sets and recent-failure sets
- measure both quality and efficiency
- treat cost and latency as first-class metrics, not footnotes

For expert tools and smart profiles, compare at least:

- `fast`
- `balanced`
- `deep`

For provider changes, compare at least:

- planner quality
- synthesis quality
- abstention quality
- runtime fallback behavior
- latency and cost per successful task

## Operational Evaluation After Deployment

The evaluation program should not stop at offline tests.

### Shadow evaluation

Before rollout, run candidate systems against sampled real traces without user
impact. Compare:

- role-level outputs
- workflow completion behavior
- abstention behavior
- source and evidence selections
- cost and latency

### Canary suites

Maintain a compact, high-signal canary set that covers:

- guided contract safety
- recent incidents
- adversarial routing
- abstention quality
- regulatory subject anchoring
- runtime truth

### Online monitoring

Monitor production by slice, including:

- task family
- provider
- profile
- model version
- cost
- latency
- abstention rate
- fallback rate
- user correction or refinement rate when observable

### Drift management

Detect drift in:

- input distribution
- failure categories
- judge score distributions
- provider outcome envelopes
- user follow-up patterns that suggest confusion or dissatisfaction

When drift is detected, convert representative failures into new eval examples.

## Proposed Repo Artifact Layout

This is a recommended future layout, not a current requirement.

```text
docs/
  llm-evaluation-program-plan.md
  llm-evaluation-dataset-schema.md
tests/
  fixtures/
    evals/
      planner/
      synthesis/
      abstention/
      provenance/
      retrieval/
      runtime/
scripts/
  run_eval_canaries.py
  run_eval_offline.py
  run_eval_shadow.py
```

Prefer task-family grouping over provider-only grouping. Provider comparisons
matter, but the product contract is role-based first.

## Phased Implementation Plan

### Phase 1: Define the contract

Deliverables:

- this planning document
- dataset schema for role-based evaluation items
- documented metric taxonomy
- documented evaluator stack and governance rules

### Phase 2: Build seed datasets

Deliverables:

- planner seed set
- synthesis seed set
- abstention seed set
- provenance seed set
- runtime truth seed set

Target:

- 25 to 50 high-quality seed items per task family

Current initial state:

- first seed datasets now exist for planner, synthesis, abstention, and
  provenance under `tests/fixtures/evals/`
- runtime and misc/supporting seed datasets also now exist under
  `tests/fixtures/evals/`
- these are intentionally small curated canaries, not full benchmark sets

### Phase 3: Expand and calibrate

Deliverables:

- synthetic expansion around seed sets
- human calibration pass on sampled items
- initial judge prompts and anchor sets
- recent-failure ingestion workflow

### Phase 4: Add repeatable tooling

Deliverables:

- offline evaluation runner
- canary runner
- structured result output for trend tracking
- provider and profile comparison report format

Current initial state:

- `scripts/run_eval_canaries.py` provides the first deterministic canary pass
  over the seed datasets and emits structured JSON output

### Phase 4.5: Add evaluator governance assets

Deliverables:

- judge-rubric definitions
- calibration anchor sets
- platform strategy for cloud or hosted evaluation layers

Current initial state:

- `tests/fixtures/evals/judge-rubrics.json` defines the first shared rubric set
- `docs/llm-evaluation-platform-strategy.md` defines how local evals,
  Azure AI Foundry, Hugging Face, and trace curation should interact

### Phase 5: Connect to workflow and release checks

Deliverables:

- canary gate for important model or provider changes
- shadow-eval plan for major behavior changes
- release checklist integration

## Initial High-Priority Work For This Repo

The highest-value near-term work is:

1. define a dataset schema for role-based evaluation items
2. create small curated seed sets for planner, synthesis, abstention, and
   provenance
3. create a human-audited judge calibration set
4. add role-aware canary suites for guided `research`, `follow_up_research`,
   `inspect_source`, and runtime truth
5. add a comparison harness for provider, prompt, and profile changes

The next concrete follow-on after the current artifacts should be a real
judge-calibration anchor set and a trace-promotion workflow, not just more seed
files.

If the team can only do one thing first, do not start with generic judge
automation. Start by building high-quality seed sets around the public guided
contract and the known failure modes already visible in current tests and docs.

## Research Basis

This plan is informed by current research and mature tooling patterns across:

- agent evaluation methods for planners, routers, and tool-using systems
- grounded RAG evaluation and claim-evidence verification
- dataset governance practices such as datasheets and versioned evaluation sets
- LLM-as-judge calibration, pairwise evaluation, and judge-drift mitigation
- adversarial and abstention-focused benchmark design
- shadow evaluation, canary evaluation, and production monitoring

Representative external references worth keeping in mind:

- Anthropic guidance on defining success and evaluating agents
- Google guidance on methodical agent evaluation and Vertex evaluation patterns
- OpenAI evaluation guidance and cookbook examples
- Azure AI evaluation concepts and built-in evaluator patterns
- Ragas, LangSmith, DeepEval, and MLflow evaluation patterns
- benchmark and methodology work such as BEIR, FEVER, HotpotQA, CheckList,
  AgentBench, and newer LLM-as-judge research

## Maintenance Rules For This Document

Update this plan when any of these change materially:

- the public guided contract
- the role boundaries between planner, synthesis, and supporting logic
- the dataset families we intend to maintain
- the evaluator stack or release-gate strategy
- the recommended artifact layout

Prefer updating principles, roles, metrics, dataset families, and phases.
Avoid line-number references or transient implementation details.
