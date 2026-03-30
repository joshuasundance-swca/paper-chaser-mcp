# ScholarAPI Integration Guide

This document captures the implementation plan for integrating ScholarAPI into
`paper-chaser-mcp` without breaking the repo's current provider posture:

- keep the default tool surface small and obvious
- keep provider-specific contracts honest
- avoid silent paid fanout
- preserve the existing citation/author graph strengths of Semantic Scholar and
  OpenAlex instead of pretending ScholarAPI can replace them

ScholarAPI is attractive here because it adds a combination the repo does not
currently have from one provider: relevance search, ingestion-oriented listing,
full plain text retrieval, batch text retrieval, and PDF download under one
authenticated API.

## Shipped Status

ScholarAPI is now shipped in this repo as an explicit provider family.

The current implementation includes:

- `SCHOLARAPI_API_KEY` and `PAPER_CHASER_ENABLE_SCHOLARAPI` parsing in
  `paper_chaser_mcp/settings.py`
- a dedicated `paper_chaser_mcp/clients/scholarapi/` client package
- explicit MCP tools for ranked search, indexed-at listing, full-text
  retrieval, batch text retrieval, and PDF retrieval
- opt-in raw broker routing through `preferredProvider` or `providerOrder`
- smart-layer retrieval support when ScholarAPI is enabled
- deployment, docs, and test coverage for the shipped surface

The rest of this guide keeps the original design rationale, contract notes, and
integration constraints that informed the implementation.

## External Contract Summary

### Base transport

- Base URL: `https://scholarapi.net/api/v1`
- Authentication: `X-API-Key` header on every request
- OpenAPI version advertised: `1.0.5` / OAS `3.1.0`
- Response metadata headers:
  - `X-Request-Id`
  - `X-Request-Cost`
  - `Server-Timing`
- Rate-limit response: `429` with `Retry-After`
- Credit exhaustion response: `402 payment_required`

### Supported endpoints relevant to this repo

| Endpoint | Best repo use | Notes |
| --- | --- | --- |
| `GET /search` | explicit topical discovery | relevance-ranked, requires at least one `q`, cursor-paginated with opaque `next_cursor` |
| `GET /list` | ingestion, monitoring, continuous corpus scans | sorted by `indexed_at` ascending, `q` optional, continue with `indexed_after=next_indexed_after` |
| `GET /text/{id}` | one-document full text retrieval | returns `text/plain`, content may come from PDF or source web page |
| `GET /texts/{ids}` | batch full text retrieval | up to 100 IDs, preserves request order, unavailable items become `null` |
| `GET /pdf/{id}` | binary PDF retrieval | returns `application/pdf`; representation over MCP needs an explicit design |

### Query semantics that matter

- `q` can be repeated, and repeated values are OR-ed
- quoted phrases are supported
- boolean operators are supported inside a query string: `AND`, `OR`, `NOT`
- parentheses are supported
- search covers title, abstract, author names, and full body when available
- punctuation, case, and many stopwords are weak signals rather than strict
  match constraints

### Filter semantics that matter

- `indexed_after` and `indexed_before` require RFC 3339 UTC timestamps ending in
  `Z`
- `published_after` and `published_before` use `YYYY-MM-DD`
- if a record has no `published_date`, date filtering falls back to `indexed_at`
- `has_text=true|false|null`
- `has_pdf=true|false|null`
- `limit` is documented up to `1000`

### Publication payload shape

The main result object includes the fields most relevant to the repo's unified
`Paper` model:

- `id`
- `title`
- `authors` as strings
- `abstract`
- `doi`
- `journal`
- `journal_publisher`
- `journal_issn`
- `journal_issue`
- `journal_pages`
- `url`
- `published_date`
- `published_date_raw`
- `indexed_at`
- `has_text`
- `has_pdf`

What it does **not** provide is just as important:

- no citation graph endpoints
- no author profile endpoints
- no recommendation endpoints
- no explicit best-match endpoint
- no DOI/native-paper lookup endpoint beyond searching and filtering the result
  payload

## Product Positioning Constraints

ScholarAPI markets itself primarily as an API for open-access full texts and
metadata at scale. The public site emphasizes:

- open-access corpus access rather than paywalled content recovery
- millions of full texts and abstracts
- monitoring and ingestion workflows
- RAG and AI-training workflows
- "best alternative to scraping Google Scholar"

That positioning matters for integration decisions:

- ScholarAPI is strongest as a discovery-plus-content provider
- ScholarAPI is not a replacement for Semantic Scholar's citation and author
  graph
- ScholarAPI is not a replacement for OpenAlex entity pivots
- ScholarAPI is likely most valuable where this repo currently relies on
  external OA enrichment or document fetch recovery after metadata search

## Recommended Runtime Role

### Phase-1 recommendation

Ship ScholarAPI first as an **explicit provider family**, not as a default
broker hop.

Rationale:

1. The API is authenticated and credit-metered, so silent fanout is a cost bug.
2. Its identifiers are provider-local and not expansion-portable.
3. It lacks citations, references, author graph, and known-item lookup
   primitives that would let it behave like the current primary graph.
4. Its unique value is full text and ingestion, which deserves explicit tools.

### Phase-2 option

After the explicit tools are stable, consider an **opt-in broker role** for
`search_papers`, but only behind all of these conditions:

- explicit enable flag remains required
- broker order must never include ScholarAPI unless the operator opted in
- provider diagnostics must surface request cost and credit failures clearly
- the user experience must explain that `search_papers_bulk` is not a
  continuation of ScholarAPI search results

Default-broker inclusion should be deferred until those conditions are met.

## Recommended MCP Surface

These tool names keep the provider contract honest and make the differences from
Semantic Scholar and OpenAlex obvious.

### Explicit search and ingestion tools

- `search_papers_scholarapi`
  - wraps `GET /search`
  - use for relevance-ranked topical discovery
  - cursor-based continuation via `next_cursor`

- `list_papers_scholarapi`
  - wraps `GET /list`
  - use for ingestion, continuous monitoring, and exhaustive date-window scans
  - continuation via `next_indexed_after`
  - do **not** describe this as page 2 of `search_papers_scholarapi`

### Explicit content tools

- `get_paper_text_scholarapi`
  - wraps `GET /text/{id}`

- `get_paper_texts_scholarapi`
  - wraps `GET /texts/{ids}`
  - preserve request order and `null` placeholders

- `get_paper_pdf_scholarapi`
  - wraps `GET /pdf/{id}`
  - response design is an explicit decision item; see below

### Optional future tool

- `search_papers_smart` integration hook
  - only after the explicit tools work
  - use ScholarAPI as an explicit full-text expansion source, not as an
    invisible planner-side citation graph

## Response-Normalization Plan

### Publication to `Paper`

Normalize ScholarAPI publications into the shared `Paper` shape with additive
provider-specific extras.

Recommended mapping:

| ScholarAPI field | Normalized target | Notes |
| --- | --- | --- |
| `id` | `paperId` and `sourceId` | prefer a namespaced `paperId` such as `ScholarAPI:<id>` to avoid collisions; keep raw ID in `sourceId` |
| `title` | `title` | direct |
| `abstract` | `abstract` | direct |
| `authors[]` | `authors[].name` | string-to-Author conversion |
| `published_date` | `publicationDate` | direct |
| `published_date` | `year` | derive from ISO date |
| `journal` | `venue` | best current fit in shared model |
| `url` | `url` | direct |
| `doi` | `canonicalId` and `recommendedExpansionId` when present | DOI is the only portable bridge into existing graph tools |
| `has_pdf` | additive field `hasPdf` | preserve as extra provider field |
| `has_text` | additive field `hasText` | preserve as extra provider field |
| `indexed_at` | additive field `indexedAt` | needed for monitoring/list continuation |
| `published_date_raw` | additive field `publishedDateRaw` | preserve source fidelity |
| `journal_publisher` | additive field `journalPublisher` | no shared first-class slot today |
| `journal_issn` | additive field `journalIssn` | preserve list |
| `journal_issue` | additive field `journalIssue` | preserve |
| `journal_pages` | additive field `journalPages` | preserve |

Recommended portability behavior:

- when `doi` exists:
  - `canonicalId = doi`
  - `recommendedExpansionId = doi`
  - `expansionIdStatus = portable`
- when `doi` is absent:
  - `canonicalId = sourceId`
  - `recommendedExpansionId = null`
  - `expansionIdStatus = not_portable`

This keeps the existing expansion contract honest: ScholarAPI IDs should not be
passed into Semantic Scholar graph tools unless a DOI bridge exists.

### Search/list pagination envelopes

For explicit MCP tools, keep the repo's opaque-cursor contract even though
ScholarAPI itself exposes two different continuation styles.

- `search_papers_scholarapi`
  - encode ScholarAPI `next_cursor` into the repo's opaque cursor wrapper
- `list_papers_scholarapi`
  - encode `next_indexed_after` into the same opaque cursor contract even
    though upstream uses a timestamp parameter

That preserves the current repo rule: callers pass back `pagination.nextCursor`
unchanged and do not learn provider-specific cursor syntax.

## Binary PDF Design Decision

`GET /pdf/{id}` returns `application/pdf`, which does not map cleanly onto the
repo's current JSON-first MCP responses.

Recommended order of preference:

1. Return a structured object with base64 content only when the caller
   explicitly asks for the binary payload.
2. Prefer a resource-oriented follow-up path if the MCP client can open binary
   resources safely.
3. Do not overload `paper.pdfUrl` with the authenticated ScholarAPI endpoint,
   because that URL is not directly useful without headers and would imply a
   public-download contract that does not exist.

This should remain an explicit design choice instead of being hidden inside a
normal paper payload.

## Repo Integration Map

### 1. Settings and local config

Update:

- `paper_chaser_mcp/settings.py`
  - add `scholarapi_api_key: str | None`
  - add `enable_scholarapi: bool = False`
  - parse `SCHOLARAPI_API_KEY`
  - parse `PAPER_CHASER_ENABLE_SCHOLARAPI`

- `paper_chaser_mcp/runtime.py`
  - add startup logging for ScholarAPI enablement and credential presence

- `tests/test_local_config_contract.py`
  - add `SCHOLARAPI_API_KEY`
  - add `PAPER_CHASER_ENABLE_SCHOLARAPI`

- `docker-compose.yaml`
- `compose.inspector.yaml`
  - propagate the new variables if the container/runtime contract expects all
    public local config knobs to be exposed through Compose

### 2. Provider client

Add a new package:

- `paper_chaser_mcp/clients/scholarapi/`

Recommended contents:

- `client.py`
  - shared `httpx.AsyncClient`
  - auth header injection
  - request-cost/header capture
  - `search_papers`
  - `list_papers`
  - `get_text`
  - `get_texts`
  - `get_pdf`
- `normalize.py`
  - publication-to-`Paper` conversion
- `models.py`
  - typed raw-response helpers only if needed; otherwise normalize directly

Operational requirements:

- honor `Retry-After` on `429`
- surface `402` distinctly as credit exhaustion
- preserve `X-Request-Id` in error paths for debugging
- avoid retrying malformed requests or auth failures

### 3. Tool args, schemas, and descriptions

Update:

- `paper_chaser_mcp/models/tools.py`
  - add typed args for ScholarAPI search/list/text/pdf tools
- `paper_chaser_mcp/tool_specs/descriptions.py`
  - describe the provider honestly as credit-metered and full-text oriented
- `paper_chaser_mcp/tool_specs/__init__.py`
  - register the new tools and tags
- `paper_chaser_mcp/dispatch.py`
  - validate args, call the new client, wrap cursors, and finalize results

### 4. Server wiring

Update:

- `paper_chaser_mcp/server.py`
  - instantiate `ScholarApiClient`
  - plumb `enable_scholarapi`
  - close the async client during lifespan shutdown

### 5. Optional broker integration

Only if Phase 2 is approved, update:

- `paper_chaser_mcp/models/tools.py`
  - add `scholarapi` provider alias to broker-facing enums
- `paper_chaser_mcp/search.py`
  - add ScholarAPI as a broker provider with accurate next-step guidance
- `paper_chaser_mcp/search_executor.py`
  - add provider request routing and outcome normalization
- `paper_chaser_mcp/provider_runtime.py`
  - include ScholarAPI diagnostics and status accounting

If brokered, the metadata must say explicitly that:

- ScholarAPI is a paid/authenticated provider
- `search_papers_bulk` remains a Semantic Scholar pivot, not a ScholarAPI
  continuation
- `list_papers_scholarapi` is the correct continuation for ScholarAPI ingestion

### 6. Smart-layer integration

Defer until after explicit tools land.

If integrated later, use ScholarAPI for:

- explicit full-text retrieval after result-set narrowing
- monitoring or corpus-growth workflows
- RAG/document grounding when the user asked for accessible full text

Do not use ScholarAPI as a silent substitute for:

- author graph expansion
- citation chasing
- recommendation generation

## Azure and Hosted Deployment Follow-up

If ScholarAPI becomes a deployed provider, update these surfaces too:

- `docs/azure-deployment.md`
  - add `scholarapi-api-key` to the Key Vault secret list if deployment should
    support it
  - add `enableScholarApi` and its env mapping to the provider matrix
- `infra/main.bicep`
- `infra/modules/containerApp.bicep`
- generated `infra/main.json`

Recommended policy matches SerpApi:

- no Azure secret requirement until the provider is deliberately enabled
- keep it opt-in and credit-aware

## Testing Plan

### Unit tests

- client request construction for all five endpoints
- auth-header injection
- `429` with `Retry-After`
- `402 payment_required`
- `404` for text/PDF retrieval
- normalization of DOI/no-DOI records
- `texts/{ids}` null preservation and request-order preservation
- cursor wrapping for both `search` and `list`

### Integration-style tests with mocks

- explicit tool dispatch
- tool descriptions and schema publication
- provider-disabled error messages
- settings parsing and local config contract coverage

### Broker tests, only if Phase 2 lands

- provider-order parsing
- broker metadata correctness
- default broker order unchanged unless explicitly configured
- next-step guidance correctly points users to `list_papers_scholarapi` rather
  than `search_papers_bulk`

## Acceptance Gates

Treat ScholarAPI as ready to ship only when these conditions hold:

1. Explicit search/list/text tools work with no live-network dependency in CI.
2. Credit failures, auth failures, and rate limits are distinguishable in both
   tool errors and provider diagnostics.
3. Search and list continuation semantics are explicit and cannot be confused.
4. ScholarAPI IDs never masquerade as Semantic Scholar expansion-safe IDs.
5. The README, provider docs, and Azure docs stay aligned on enablement and
   credential requirements.
6. The default broker remains unchanged unless an explicit follow-up change
   deliberately adds ScholarAPI to it.

## Recommended Delivery Sequence

1. Add settings parsing, client package, and explicit ScholarAPI tools.
2. Add docs, tests, and local config propagation.
3. Add provider diagnostics coverage for ScholarAPI request cost and failure
   modes.
4. Evaluate whether broker participation is worth the credit cost and semantic
   mismatch.
5. Only after that, consider smart-layer or hosted-deployment expansion.

That sequence preserves the repo's current strengths while adding ScholarAPI in
the area where it is genuinely differentiated: authenticated discovery plus
accessible full-text retrieval and monitoring.
