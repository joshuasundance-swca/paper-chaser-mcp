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

Guided `follow_up_research` delegates internally to `ask_result_set`, so the
default public surface already depends on synthesis quality.

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
