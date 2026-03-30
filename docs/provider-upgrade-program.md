# Provider Upgrade Program

This document is the repo-level operational summary for the provider upgrade
work. It sits above the provider-specific guides and records the runtime policy,
public MCP surface, and acceptance gates that now define the production
contract.

Use this together with:

- `docs/semantic-scholar-api-guide.md`
- `docs/openalex-api-guide.md`
- `docs/serpapi-google-scholar-api-guide.md`
- `docs/scholarapi-api-guide.md`

## What Shipped

The provider upgrade program now ships these cross-cutting changes:

- A shared provider execution policy with normalized status buckets, bounded
  retries with jitter, per-provider concurrency caps, suppression state, and
  outcome telemetry.
- A diagnostics surface via `get_provider_diagnostics` so operators can inspect
  provider health, throttling, retries, and fallback reasons without tailing
  logs.
- Smart-tool latency controls via `latencyProfile=fast|balanced|deep`, plus
  optional `providerBudget` on `search_papers_smart`.
- A direct OpenAI Responses API path with structured outputs and a fallback path
  to the existing LangChain-backed bundle.
- Expanded OpenAlex and SerpApi explicit tool surfaces without breaking the
  existing MCP tool names.
- CORE disabled by default pending stability validation.

## Provider Roles

| Provider | Runtime role | Default broker status | Best-fit work | Current disposition |
| --- | --- | --- | --- | --- |
| Semantic Scholar | Primary scholarly graph | Enabled, first | General scholarly discovery, exact-ish known-item recovery, citations/references, author graph, smart retrieval | Ship |
| arXiv | Preprint and recency specialist | Enabled, second | Preprints, recent papers, identifier-based arXiv recovery, free fallback coverage | Ship |
| CORE | Experimental lexical fallback | Disabled by default | Sparse lexical recovery when explicitly enabled and monitored | Defer by default |
| OpenAlex | Secondary graph and disambiguation surface | Explicit tools only | DOI/work lookup, autocomplete, source/institution/topic pivots, author workflows, citation/reference traversal | Ship as explicit provider |
| SerpApi Google Scholar | Guarded recall-recovery layer | Disabled by default; explicit tools only unless smart routing opts in | Cited-by, versions, author profile/article flows, citation export, quota-aware recovery | Ship with budgets |
| OpenAI | Smart orchestration and synthesis provider | Used by smart tools when enabled | Planning, synthesis, reranking support, theme labeling, grounded answer formatting | Ship with fallback |

## Shared Execution Policy

Every provider call now returns the same normalized outcome envelope:

- `provider`
- `endpoint`
- `statusBucket`
- `retries`
- `latencyMs`
- `cache`
- `quota`
- `fallbackReason`

The runtime policy is intentionally conservative:

- retry transient provider failures only a bounded number of times
- add jitter to avoid synchronized retries
- cap concurrent calls per provider
- suppress providers temporarily after repeated failures
- surface suppression and fallback reasons in diagnostics rather than hiding
  them inside logs

## Smart-Tool Profiles

`search_papers_smart`, `ask_result_set`, `map_research_landscape`, and
`expand_research_graph` accept `latencyProfile`.

| Profile | Intent | Behavior |
| --- | --- | --- |
| `fast` | interactive debugging and quick answers | prefers deterministic or narrow-path behavior, skips broadening and expensive fanout |
| `balanced` | default production behavior | keeps the existing smart workflow feel while respecting the new runtime controls |
| `deep` | explicit research mode | allows controlled multi-provider fanout, including recommendation enrichment and guarded SerpApi usage |

`search_papers_smart` also accepts `providerBudget`, which lets advanced clients
cap provider fanout explicitly instead of relying on default routing only.

## Public MCP Additions

New explicit tools added by the upgrade:

- `paper_autocomplete_openalex`
- `search_entities_openalex`
- `search_papers_openalex_by_entity`
- `search_papers_serpapi_cited_by`
- `search_papers_serpapi_versions`
- `get_author_profile_serpapi`
- `get_author_articles_serpapi`
- `get_serpapi_account_status`
- `get_provider_diagnostics`

Backward compatibility rules:

- existing tool names stay intact
- `balanced` remains the default smart latency profile
- new parameters are optional
- provider diagnostics are additive and do not change existing response shapes

## Provider Dossier Snapshot

### Semantic Scholar

- Current role: primary graph and first broker hop
- Strengths: broad normalized paper graph, citations/references, author graph,
  recommendations, bulk search, structured filters
- Risks: strict throttling and 429 sensitivity, especially for broad smart-path
  fanout
- Operational posture: keep fields narrow, prefer batch and recommendation
  enrichment, and reserve dataset-scale work for offline flows
- Decision: ship as primary provider

### OpenAlex

- Current role: explicit secondary graph and disambiguation provider
- Strengths: DOI/work lookup, autocomplete, source/institution/topic pivots,
  strong entity filtering, cursor pagination
- Risks: semantic mismatch with the brokered Semantic-Scholar-shaped flow,
  credit cost on list traffic, URI-shaped identifiers
- Operational posture: keep explicit, prefer API key plus contact email in
  production, use entity pivots for venue/topic disambiguation
- Decision: ship as explicit provider, not as an automatic broker hop

### SerpApi Google Scholar

- Current role: guarded recall-recovery and citation-discovery layer
- Strengths: cited-by, versions, author profiles, citation export, quota-aware
  account surface
- Risks: paid traffic, quota exhaustion, and a higher operational/compliance
  burden than the open scholarly APIs
- Operational posture: explicit tools first, budget-aware smart routing second,
  never silent broad fanout
- Decision: ship behind explicit enablement and budgets

### CORE

- Current role: optional lexical fallback only
- Strengths: free coverage when healthy
- Risks: redirect and 5xx instability, plus limited filter compatibility
- Operational posture: disabled by default until it proves stable under the new
  diagnostics and retry policy
- Decision: defer as a default provider

### arXiv

- Current role: free preprint and recency fallback
- Strengths: arXiv-native identifiers and recent/preprint coverage
- Risks: narrower metadata shape and weaker general-purpose lexical discovery
  than the primary scholarly graphs
- Operational posture: keep narrow and honest; use for arXiv-native recovery and
  recency-oriented fallback
- Decision: ship as a focused fallback

### OpenAI

- Current role: smart-path planning, synthesis, labeling, and embeddings
- Strengths: structured output support, flexible synthesis, and direct Responses
  API integration
- Risks: extra latency and cost when overused, plus dependency on upstream model
  availability
- Operational posture: use `fast` for smoke tests, `balanced` by default, and
  `deep` only when broader fanout is intentional
- Decision: ship with direct Responses support and fallback behavior

## Benchmark Corpus

The rollout now includes a benchmark corpus fixture at:

- `tests/fixtures/provider_benchmark_corpus.json`

The corpus covers:

- DOI lookup
- title and citation repair
- author disambiguation
- broad topical discovery
- citation expansion
- smart landscape mapping
- outage drills for Semantic Scholar, OpenAlex, CORE, SerpApi, arXiv, and
  OpenAI

Treat this as the shared acceptance harness for future provider changes.

## Acceptance Gates

The upgrade is considered healthy when these conditions hold:

- no duplicate launches in steady state
- direct-tool p95 stays at or below 10 seconds on the benchmark corpus
- `balanced` smart-tool p95 stays at or below 20 seconds on the benchmark corpus
- provider-originated error rate drops materially from the captured baseline
- SerpApi spend is attributable to successful recovery or discovery value
- CORE remains out of the default broker until it clears the same gates as the
  other providers

## Maintenance Rule

If runtime policy and provider docs drift, update this file first, then update
the provider-specific guide that owns the deeper provider semantics.
