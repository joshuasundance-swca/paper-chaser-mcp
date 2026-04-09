# OpenRouter Provider Guide

This guide describes how to add and operate OpenRouter as an agentic provider
for `paper-chaser-mcp`.

Unlike the provider-specific search docs in this repo, OpenRouter is not a
scholarly metadata source. It is a smart-layer model router that presents an
OpenAI-compatible API over many underlying model vendors and provider endpoints.
That makes it a fit for the planner and synthesizer roles in the smart layer,
not for the direct paper-search broker.

This document is intentionally implementation-focused. It is written to support
an actual repo change with red/green TDD, not just to catalog OpenRouter
features.

## Current Repo Status

Today the repo ships these agentic providers:

- `openai`
- `azure-openai`
- `anthropic`
- `nvidia`
- `google`
- `mistral`
- `huggingface`
- `deterministic`

OpenRouter is **not** implemented yet.

That means it is currently absent from:

- `paper_chaser_mcp/settings.py` provider parsing and `AppSettings`
- `paper_chaser_mcp/agentic/config.py` provider-default model selection
- `paper_chaser_mcp/agentic/providers.py` bundle resolution
- `paper_chaser_mcp/provider_runtime.py` paywalled-provider policy and
  diagnostics
- `paper_chaser_mcp/dispatch.py` runtime provider summary ordering
- `paper_chaser_mcp/server.py` smart-bundle construction
- `scripts/generate_eval_topics.py` provider handoff wiring
- local config, deployment, and IaC contracts

So the job is not just "add one bundle class". Provider parity in this repo
also requires config, runtime reporting, deployment scaffolding, and test
coverage parity.

## Why OpenRouter Fits This Repo

If you remember only a few things, make them these:

1. OpenRouter belongs in the smart layer, not the scholarly retrieval layer.
2. The repo should treat OpenRouter as an OpenAI-compatible router, not as a
   completely new model API family.
3. Structured output is the main correctness risk, because this repo relies on
   schema-shaped planner and synthesizer outputs.
4. Embeddings should stay disabled for the first pass unless explicitly proven
   against the real OpenRouter surface used here.
5. Provider parity includes runtime diagnostics, infra, and deployment docs,
   not only Python code.

OpenRouter is attractive here because it provides:

- one OpenAI-compatible base URL: `https://openrouter.ai/api/v1`
- model routing across many labs and hosting providers
- request-level provider preferences such as ordering, fallback control,
  latency or throughput preference, and data-policy filters
- structured outputs and tool-calling support on the chat-completions surface
- usage and cost metadata in the response body

Primary references:

- Overview: https://openrouter.ai/
- Quickstart: https://openrouter.ai/docs/quickstart
- API reference overview: https://openrouter.ai/docs/api-reference/overview
- Provider routing: https://openrouter.ai/docs/guides/routing/provider-selection

## Recommended Runtime Role

OpenRouter should be added as an **agentic provider** for:

- planner calls
- synthesizer calls
- optional future tool-calling or structured-output-heavy smart workflows

It should **not** be added to:

- `DEFAULT_SEARCH_PROVIDER_ORDER`
- the direct paper-search broker
- guided or expert scholarly provider lists such as Semantic Scholar,
  OpenAlex, CORE, or arXiv

That keeps the contract honest. OpenRouter decides how LLM requests are routed.
The scholarly providers still decide how paper metadata is retrieved.

## Integration Shape

### Recommended approach

Add an `OpenRouterProviderBundle` as an **OpenAI-compatible smart-provider
bundle**.

The implementation should live alongside the existing OpenAI-compatible smart
providers, and it should preserve the current fallback semantics:

- configured provider name should be `openrouter`
- active provider should remain `openrouter` when requests succeed
- active provider should fall back to `deterministic` on missing keys,
  unavailable dependencies, or structured-output failure paths that the bundle
  cannot safely recover from

### Why not make it a pure `LangChainChatProviderBundle`?

There is an existing precedent for OpenAI-compatible routers in the repo:
Hugging Face uses `ChatOpenAI(..., base_url=...)` and stays chat-only.

That path is useful, but OpenRouter should not be modeled as *only* a
LangChain chat wrapper if the goal is parity with the repo's stronger OpenAI
path.

The direct OpenAI-compatible bundle already owns:

- sync and async SDK clients
- deterministic fallback behavior
- schema-shaped planner and synthesis flows
- the place where future embeddings would live if they are ever validated

So the recommended design is:

1. Subclass `OpenAIProviderBundle` for OpenRouter.
2. Override the OpenAI client loaders so they use OpenRouter's base URL and
   optional attribution headers.
3. Override the LangChain model loaders so they use an OpenAI-compatible client
   with `base_url` set to OpenRouter.
4. Keep embeddings disabled in phase one, even if the class inherits the
   embedding hooks.

That gives the repo the right long-term shape without pretending OpenRouter is
identical to direct OpenAI.

### LangChain recommendation

Use LangChain deliberately, not as the entire implementation strategy.

Recommended use:

- use `langchain_openai.ChatOpenAI` or `init_chat_model(...,
  model_provider="openai", base_url=...)` for planner and synthesizer chat
  models
- keep the direct OpenAI-compatible client path for places where the repo wants
  tighter response normalization or future SDK-only features

Do **not** let a LangChain-only implementation silently reduce parity by:

- dropping runtime metadata behavior already present in the OpenAI bundle
- implying embedding support that has not been validated
- weakening structured-output enforcement for planner and synthesizer schemas

## OpenRouter Surface That Matters Here

The repo only needs a subset of the OpenRouter API.

| Repo need | OpenRouter surface | Why it matters |
| --- | --- | --- |
| planner and synthesizer chat | `POST /api/v1/chat/completions` | main smart-layer request path |
| structured planner output | `response_format` with `json_object` or `json_schema` | this repo depends on schema-shaped outputs |
| tool calling | `tools`, `tool_choice` | relevant for future smart workflows and for parity reasoning |
| provider control | `provider` object | OpenRouter-only routing knobs |
| model catalog validation | `GET /api/v1/models` | useful for smoke tests and future validation tooling |
| usage and cost inspection | `usage` in response and `/api/v1/generation` | useful for operator visibility, not required for first pass |

Important API behavior from the current docs:

- OpenRouter is OpenAI-chat compatible at the request and response shape level.
- Optional attribution headers are `HTTP-Referer` and `X-OpenRouter-Title`
  (`X-Title` is also accepted by OpenRouter, but the OpenRouter docs name the
  former header explicitly).
- When a parameter is unsupported by the selected upstream provider, OpenRouter
  may ignore it unless `provider.require_parameters=true` is used.
- Structured output supports both `json_object` and `json_schema`.
- Tool calls are normalized into the OpenAI chat-completions shape.
- Usage can include cost and BYOK metadata.

## Repo-Specific Operating Principles

1. Treat OpenRouter as paywalled in runtime diagnostics and provider policy.
2. Keep the first implementation chat-only unless embeddings are validated with
   tests and real API evidence.
3. Default structured-output calls to OpenRouter providers that support the
   requested parameters.
4. Keep public config small and obvious; do not expose the entire OpenRouter
   routing matrix on day one.
5. Preserve deterministic fallback behavior exactly.

## Proposed Public Config Surface

The minimum viable config for parity should be:

| Variable | Kind | Recommended default | Purpose |
| --- | --- | --- | --- |
| `PAPER_CHASER_AGENTIC_PROVIDER` | non-secret | `openai` | set to `openrouter` to select the provider |
| `OPENROUTER_API_KEY` | secret | none | OpenRouter authentication |
| `OPENROUTER_BASE_URL` | non-secret | `https://openrouter.ai/api/v1` | override for compatibility or testing |
| `OPENROUTER_HTTP_REFERER` | non-secret | empty | optional app attribution header |
| `OPENROUTER_TITLE` | non-secret | empty | optional app attribution header |

For the current planned bring-up in this repo, the important part is that the
explicit OpenRouter model settings should be preserved rather than overwritten.
The immediate implementation target is:

| Variable | Current bring-up value | Intended role |
| --- | --- | --- |
| `PAPER_CHASER_PLANNER_MODEL` | `arcee-ai/trinity-mini` | fast planner / query strategy / classification |
| `PAPER_CHASER_SYNTHESIS_MODEL` | `arcee-ai/trinity-large-thinking` | grounded synthesis / follow-up answer shaping |

That changes the first implementation priority slightly:

- first make sure explicit configured model names survive end-to-end through
  `AppSettings`, `AgenticConfig`, bundle resolution, and runtime metadata
- only after that path is green decide whether to add checked-in OpenRouter
  provider defaults for the case where users leave the repo's OpenAI defaults
  unchanged

For a second wave, these are the most defensible optional additions:

| Variable | Kind | Suggested scope |
| --- | --- | --- |
| `OPENROUTER_REQUIRE_PARAMETERS` | non-secret | default `true` for structured-output-sensitive requests |
| `OPENROUTER_ALLOW_FALLBACKS` | non-secret | default `true` |
| `OPENROUTER_PROVIDER_SORT` | non-secret | `price`, `throughput`, or `latency` |
| `OPENROUTER_ZDR` | non-secret | require zero-data-retention endpoints |
| `OPENROUTER_DATA_COLLECTION` | non-secret | `allow` or `deny` |

Do not expose every OpenRouter knob immediately. The repo's current config style
favors a small public surface with behaviorally significant defaults.

## Required Code Touchpoints

An implementation that actually matches repo requirements will need updates in
at least these places:

### Python runtime and config

- `paper_chaser_mcp/settings.py`
  add `openrouter` to `AgenticProvider`, add OpenRouter env fields, and parse
  the new vars in `AppSettings.from_env()`
- `paper_chaser_mcp/agentic/config.py`
  add provider-default model handling for `openrouter` if the repo wants
  sensible checked-in defaults when the OpenAI defaults are still configured
- `paper_chaser_mcp/agentic/provider_openai.py`
  add `OpenRouterProviderBundle` or equivalent OpenAI-compatible subclass
- `paper_chaser_mcp/agentic/providers.py`
  resolve `config.provider == "openrouter"`
- `paper_chaser_mcp/server.py`
  thread OpenRouter settings into `resolve_provider_bundle()`
- `scripts/generate_eval_topics.py`
  thread OpenRouter settings through the eval bootstrap path

### Runtime reporting and policy

- `paper_chaser_mcp/provider_runtime.py`
  add `openrouter` to `ProviderName` and `DEFAULT_PROVIDER_POLICIES`
- `paper_chaser_mcp/dispatch.py`
  add OpenRouter to runtime summary ordering and provider diagnostics output
- `paper_chaser_mcp/agentic/graphs.py`
  include `openrouter` in `smart_provider_diagnostics()` ordering and enabled
  state reporting

### Local config and deployment parity

- `.env.example`
- `docker-compose.yaml`
- `compose.inspector.yaml`
- `infra/main.bicep`
- `infra/modules/containerApp.bicep`
- generated `infra/main.json`
- deployment and security docs that enumerate smart-provider config surfaces

### Test contracts

- `tests/test_settings.py`
- `tests/test_agentic_providers_extra.py`
- `tests/test_dispatch.py`
- `tests/test_provider_runtime.py`
- `tests/test_local_config_contract.py`
- `tests/test_validate_deployment.py`

## Default Model Policy

OpenRouter model IDs are provider-prefixed, for example:

- `openai/gpt-5.4-mini`
- `openai/gpt-5.4`
- `anthropic/claude-sonnet-4.5`
- `google/gemini-3-flash-preview`

The repo should follow the same policy used for other additive providers:

- if planner and synthesis are still at the checked-in OpenAI defaults, swap to
  checked-in OpenRouter defaults when `agentic_provider=openrouter`
- if the user explicitly configured model names, honor them

For the current rollout, the explicit-model rule is the one that matters most.
The first green implementation should preserve:

- planner: `arcee-ai/trinity-mini`
- synthesis: `arcee-ai/trinity-large-thinking`

before it tries to define any checked-in OpenRouter provider defaults.

Conservative first-pass defaults would be:

- planner: `openai/gpt-5.4-mini`
- synthesis: `openai/gpt-5.4`

That keeps model behavior close to the current OpenAI baseline while still
exercising the OpenRouter transport.

### Current bring-up recommendation

Given the current `.env` setup, the rollout should treat the Trinity pair as the
real acceptance target for phase one:

- `arcee-ai/trinity-mini` for planning because it should be cheaper and faster
  for classification and query-shaping work
- `arcee-ai/trinity-large-thinking` for synthesis because it is the more likely
  candidate to preserve answer quality once grounded evidence is available

Expected operational tradeoffs:

- planner latency should stay relatively modest if the model follows schema
  instructions well
- synthesis latency will likely be higher, and this repo should expect that in
  follow-up or smart-answer paths
- the main technical risk is not topical quality but structured-output
  compliance, especially for the thinking-oriented synthesis model

So initial smoke validation should answer these questions in order:

1. Does `arcee-ai/trinity-mini` reliably return the planner schema under the
   repo's current structured-output path?
2. Does `arcee-ai/trinity-large-thinking` reliably return the answer and theme
   schemas without drifting into prose-first output?
3. When either fails, does the bundle degrade to deterministic mode cleanly and
   report that via `activeSmartProvider`?

If the answer to question 2 is unstable, the implementation should still ship
OpenRouter support, but the doc and tests should record that the configured
synthesis model may need to change before OpenRouter becomes the recommended
operator default.

## Structured Output Requirements

This repo is not tolerant of hand-wavy schema support. Planner and synthesizer
paths depend on structured outputs landing reliably.

OpenRouter specifically adds a routing hazard here: if the request includes
`response_format` or `tools`, a routed upstream provider may ignore unsupported
parameters unless the request demands parameter support.

So the implementation should do one of these:

1. set `provider.require_parameters=true` on structured-output-sensitive calls,
   or
2. constrain OpenRouter routing to a narrower set of providers that are known to
   honor the requested schema or tool parameters

The first option is the better repo default because it is safer and aligns with
the current planner and synthesizer contract.

For the current Trinity bring-up, `provider.require_parameters=true` should be
treated as part of the initial correctness plan rather than an optional hardening
step. It is the most direct way to avoid silent parameter downgrades while
testing whether those two models can satisfy this repo's schema requirements.

## Embeddings Policy

The repo already defaults embeddings off, and OpenRouter documentation gathered
for this design is centered on chat completions rather than a validated
embedding surface that this codebase already uses.

Recommendation:

- phase one: `supports_embeddings()` should return `False` for OpenRouter
- do not wire `OPENROUTER_EMBEDDING_MODEL`
- do not try to reuse the OpenAI embedding path until it has explicit tests and
  verified API behavior

That keeps parity honest. The provider is still fully usable for planner and
synthesizer roles.

## Headers and Attribution

OpenRouter allows optional attribution headers:

- `HTTP-Referer`
- `X-OpenRouter-Title`

They are not required for correctness, but if the repo exposes them they should
be treated as **non-secret deployment config**, not secret material.

Recommended behavior:

- only send them when configured
- do not fail provider initialization when they are missing
- keep them out of secret stores in Azure docs and IaC

## Initial Bring-Up Plan

Because `OPENROUTER_API_KEY` is already configured locally and the planner and
synthesis models are already chosen, the next implementation plan should be more
concrete than a generic provider-addition outline.

### Phase 0: explicit-model config path

Goal: prove the repo can select OpenRouter and preserve the configured Trinity
models exactly.

Required outcomes:

- `AppSettings.from_env()` reads `OPENROUTER_API_KEY`
- `PAPER_CHASER_AGENTIC_PROVIDER=openrouter` parses correctly
- `AgenticConfig.from_settings()` preserves
  `arcee-ai/trinity-mini` and `arcee-ai/trinity-large-thinking`
- runtime metadata reports those model names without rewriting them to OpenAI
  defaults

This phase should land before any provider-default-model work.

### Phase 1: bundle construction and deterministic fallback

Goal: create an OpenRouter bundle that behaves like the existing additive smart
providers from the outside.

Required outcomes:

- `resolve_provider_bundle()` returns `OpenRouterProviderBundle`
- missing key or missing dependency paths fall back to deterministic mode
- `configuredSmartProvider` reports `openrouter`
- `activeSmartProvider` reports either `openrouter` or `deterministic`

### Phase 2: schema smoke tests against the chosen models

Goal: validate the exact planner and synthesizer models selected for bring-up.

Required outcomes:

- planner schema succeeds with `arcee-ai/trinity-mini`
- synthesis or answer schema succeeds with
  `arcee-ai/trinity-large-thinking`
- failures are explicit and testable, not hidden behind ad hoc text parsing

This is the point where the implementation should decide whether the synthesis
path needs a model swap, a stricter request shape, or a narrower OpenRouter
routing policy.

### Phase 3: runtime and deployment parity

Goal: finish the provider as a first-class repo feature rather than a local
experiment.

Required outcomes:

- diagnostics list OpenRouter in provider order and policy snapshots
- compose, `.env.example`, Azure docs, and Bicep know about OpenRouter
- local and Azure config treat OpenRouter attribution headers as non-secret

### Phase 4: optional provider-default support

Goal: only after the explicit-model path is stable, decide whether to add a
checked-in OpenRouter default model pair for users who switch providers but do
not override planner and synthesis.

This phase is optional for the first PR.

## Provider Routing Policy For This Repo

OpenRouter exposes a rich `provider` object with routing controls. The repo
should use that carefully.

Recommended first-pass behavior:

- keep fallbacks enabled by default
- require parameter support for structured-output-sensitive requests
- do not expose provider ordering, latency sorting, or data-policy toggles as
  public config until there is a concrete operator need

Good second-wave operator features:

- `provider.sort`
- `provider.only`
- `provider.ignore`
- `provider.zdr`
- `provider.data_collection`

Those are valuable, but they are not necessary to ship a correct first pass.

## Red/Green TDD Plan

The safest path is to add OpenRouter in narrow, testable slices.

### Phase 1: settings and config contract

Red:

- extend `tests/test_settings.py` so `PAPER_CHASER_AGENTIC_PROVIDER=openrouter`
  parses successfully
- add tests for `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`,
  `OPENROUTER_HTTP_REFERER`, and `OPENROUTER_TITLE`
- add an `AgenticConfig.from_settings()` test proving explicit configured model
  names survive unchanged when the provider is `openrouter`
- add a second `AgenticConfig.from_settings()` test for provider-default-model
  behavior only if the implementation chooses to support that in the first pass

Green:

- update `settings.py`
- update `agentic/config.py`

### Phase 2: provider bundle resolution

Red:

- extend `tests/test_agentic_providers_extra.py` so
  `resolve_provider_bundle(_config(provider="openrouter"), ...)` returns the
  new bundle class
- assert selection metadata reports
  `configuredSmartProvider == "openrouter"`
- assert deterministic fallback is preserved when the key is missing or the
  runtime cannot initialize the provider

Green:

- add `OpenRouterProviderBundle`
- wire it in `agentic/providers.py`
- wire settings through `server.py` and `scripts/generate_eval_topics.py`

Add these model-specific red tests here if possible:

- selection metadata shows planner model `arcee-ai/trinity-mini`
- selection metadata shows synthesis model
  `arcee-ai/trinity-large-thinking`

### Phase 3: runtime diagnostics and policy

Red:

- extend `tests/test_dispatch.py` so runtime diagnostics include OpenRouter in
  provider ordering and paywalled reporting
- extend `tests/test_provider_runtime.py` so `policy_for_provider("openrouter")`
  exists and `provider_is_paywalled("openrouter")` is true

Green:

- update `provider_runtime.py`
- update `dispatch.py`
- update `agentic/graphs.py`

### Phase 4: schema bring-up for the Trinity pair

Red:

- add focused provider tests that exercise planner-schema coercion for
  `arcee-ai/trinity-mini`
- add focused provider tests that exercise synthesis-schema coercion for
  `arcee-ai/trinity-large-thinking`
- assert that unsupported structured-output paths fall back cleanly instead of
  being accepted as success

Green:

- finalize the OpenRouter request shape
- set `provider.require_parameters=true` for schema-sensitive requests
- add any minimal response normalization needed for these models

### Phase 5: deployment and local contract parity

Red:

- extend `tests/test_local_config_contract.py` for the new env vars
- extend `tests/test_validate_deployment.py` for the new Bicep params and
  environment variables

Green:

- update `.env.example`
- update local compose files
- update Bicep and generated ARM JSON
- update deployment docs and security-model docs

### Phase 6: behavior regression checks

Red:

- add focused tests that OpenRouter stays chat-only in the first pass
- add tests that structured-output-sensitive calls enable the safer OpenRouter
  request behavior rather than silently relying on provider luck

Green:

- finish the request-shaping behavior in the provider bundle

## Acceptance Checklist

Do not call the implementation complete until all of these are true:

- `PAPER_CHASER_AGENTIC_PROVIDER=openrouter` works end to end
- runtime metadata reports configured and active OpenRouter provider names
- explicit configured model names survive end to end, including the current
  Trinity planner and synthesis pair
- deterministic fallback still activates cleanly
- OpenRouter is marked paywalled in diagnostics
- `.env.example`, compose files, deployment docs, and Bicep templates all know
  about the new provider
- the provider does not claim embeddings unless embeddings are actually tested
- structured-output-sensitive requests are protected against upstream providers
  silently ignoring unsupported parameters
- the planner path is proven against `arcee-ai/trinity-mini`
- the synthesis path is proven against `arcee-ai/trinity-large-thinking`, or
  the docs explicitly record why a different synthesis model was required

## Focused Validation Commands

During implementation, the smallest useful loops are:

```bash
python -m pytest tests/test_settings.py tests/test_agentic_providers_extra.py -q
python -m pytest tests/test_dispatch.py tests/test_provider_runtime.py -q
python -m pytest tests/test_local_config_contract.py tests/test_validate_deployment.py -q
```

Once the explicit-model config path is green, add a focused loop for the Trinity
pair provider tests before broadening the rollout:

```bash
python -m pytest tests/test_agentic_providers_extra.py -k "openrouter or trinity" -q
```

If there are no OpenRouter-specific tests yet, the first PR should add them and
use a tighter selector matching the actual new test names.

Before finishing the work, rerun the normal repo baseline expected for touched
areas.

## Practical Recommendation

Ship OpenRouter in two steps.

Step 1:

- add OpenRouter as a chat-only smart provider with deterministic fallback,
  runtime diagnostics, and deployment parity
- use OpenAI-compatible clients plus LangChain chat models with a custom base
  URL
- default to safer structured-output routing behavior

Step 2:

- only after Step 1 is stable, evaluate whether OpenRouter should expose more
  routing knobs or any embedding support in this repo

That gets the repo to parity without overclaiming capabilities or widening the
public contract too early.
