# Scholar Search MCP

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/joshuasundance-swca/scholar-search-mcp)

A FastMCP-based MCP server that integrates the [CORE API v3](https://api.core.ac.uk/docs/v3), [Semantic Scholar API](https://www.semanticscholar.org/product/api), [OpenAlex API](https://developers.openalex.org/), [arXiv API](https://info.arxiv.org/help/api/user-manual.html), and optionally [SerpApi Google Scholar](https://serpapi.com/google-scholar-api) so AI assistants (e.g. Claude, Cursor) can search and fetch academic paper metadata.

The package now uses FastMCP for tool/resource/prompt registration, Pydantic for strict tool inputs and normalized provider payloads, and provider clients organized as expandable subpackages under `scholar_search_mcp/clients/`.

## Features

- **Search papers** – Keyword search with a configurable **fallback chain**: defaults to **Semantic Scholar** first, then **arXiv**, then optionally **CORE** (disabled by default until explicitly enabled), then optionally **SerpApi Google Scholar** (opt-in paid). You can keep the default broker behavior, set a `preferredProvider`, override `providerOrder`, or call provider-specific `search_papers_*` tools directly. Returns a single best-effort page — not paginated.
- **Bulk paper search** – Boolean-syntax search via `/paper/search/bulk` with cursor-based pagination (up to 1,000 returned papers/call). **The default ordering is NOT relevance-ranked** — bulk retrieval uses exhaustive corpus traversal with an internal ordering; every response includes a `retrievalNote` field describing the active ordering contract. For citation-ranked traversal pass `sort='citationCount:desc'`. The upstream bulk endpoint may ignore small `limit` values internally, so this server truncates returned data to the requested limit; use `search_papers` or `search_papers_semantic_scholar` for small targeted pages. Treat `pagination.nextCursor` as opaque and pass it back unchanged as `cursor`
- **Best-match / autocomplete** – Single best title match and typeahead completions
- **Paper details** – Full metadata (title, authors, abstract, citations, etc.)
- **Citations & references** – Papers that cite or are cited by a given paper; treat `pagination.nextCursor` as opaque and pass it back unchanged as `cursor`
- **Paper authors** – Author listing for a specific paper; treat `pagination.nextCursor` as opaque and pass it back unchanged as `cursor`
- **Author search & batch** – Search authors by name with cursor pagination, or fetch up to 1,000 author profiles in one call
- **Snippet search** – Quote-like text snippet search returning snippet text, paper metadata, and score
- **Batch lookup** – Fetch up to 500 papers in one call
- **Recommendations** – Similar papers via single-seed GET or multi-seed POST
- **Citation formats** – Get MLA, APA, BibTeX, and other citation export formats for a Google Scholar paper (requires SerpApi)
- **OpenAlex-native workflows** – Explicit OpenAlex search, cursor-paginated retrieval, work lookup by DOI/OpenAlex ID, autocomplete, source/institution/topic pivots, cited-by/reference traversal, and author pivots without forcing OpenAlex into the brokered Semantic-Scholar-shaped flow
- **Provider execution policy** – Shared retries with jitter, bounded concurrency, suppression/circuit-breaker state, normalized provider outcomes, and a diagnostics tool for live provider health
- **Shared rate limiter** – One 1 req/s pacing lock shared across all Semantic Scholar endpoints
- **Structured FastMCP outputs** – Tools return structured content instead of JSON blobs embedded in text
- **Agent onboarding aids** – Ships a workflow guide resource and a planning prompt alongside the tools
- **Additive smart research layer** – `search_papers_smart`, `ask_result_set`, `map_research_landscape`, and `expand_research_graph` add concept-level discovery, grounded follow-up QA, theme mapping, and compact graph expansion without changing the existing raw-tool contract
- **Latency profiles and provider budgets** – Smart tools accept `latencyProfile` (`fast`, `balanced`, `deep`) and `search_papers_smart` also accepts an optional `providerBudget` for guarded multi-provider fanout
- **Compatibility-first agent UX** – Primary read tools now surface `agentHints`, `clarification`, `resourceUris`, and reusable `searchSessionId` handles so agents can keep moving without reading the full docs
- **Microsoft-facing packaging assets** – Ships `mcp-tools.core.json`, `mcp-tools.full.json`, and `microsoft-plugin.sample.json` for Streamable HTTP deployments and declarative-agent packaging

## Installation

```bash
pip install scholar-search-mcp
```

Optional extras for the additive AI layer:

```bash
pip install scholar-search-mcp[ai]
```

Optional FAISS backend extras:

```bash
pip install scholar-search-mcp[ai,ai-faiss]
```

### Local Quick Start

The Azure deployment scaffold is optional. Local `python -m scholar_search_mcp`
still defaults to `stdio` transport and does not require any Azure deployment
variables.

If you want a local env template for shell runs or Docker, copy
`.env.example` to `.env`, fill in only the providers you use, and keep the
resulting file uncommitted. The repo ignores `.env`, `.env.local`, other
`.env.*` variants except `.env.example`, and `.local-*` planning/artifact
files. Docker now keeps those local-only files out of the image build context.

## Configuration

### Claude Desktop

Edit the config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add:

Set notes outside the JSON block so the example remains valid JSON. In the sample below, CORE is disabled and Semantic Scholar plus arXiv are enabled.

```json
{
  "mcpServers": {
    "scholar-search": {
      "command": "python",
      "args": ["-m", "scholar_search_mcp"],
      "env": {
        "SCHOLAR_SEARCH_ENABLE_CORE": "false",
        "SCHOLAR_SEARCH_ENABLE_SEMANTIC_SCHOLAR": "true",
        "SCHOLAR_SEARCH_ENABLE_ARXIV": "true"
      }
    }
  }
}
```

If you have API keys (optional but recommended for search):

The next example is also valid JSON. It enables all three search providers and supplies API keys for the services that support them.

```json
{
  "mcpServers": {
    "scholar-search": {
      "command": "python",
      "args": ["-m", "scholar_search_mcp"],
      "env": {
        "CORE_API_KEY": "your-core-api-key-here",
        "SEMANTIC_SCHOLAR_API_KEY": "your-semantic-scholar-api-key-here",
        "SCHOLAR_SEARCH_ENABLE_CORE": "true",
        "SCHOLAR_SEARCH_ENABLE_SEMANTIC_SCHOLAR": "true",
        "SCHOLAR_SEARCH_ENABLE_ARXIV": "true"
      }
    }
  }
}
```

### Cursor

Add an MCP server in Cursor settings with the same `command`, `args`, and `env` as above.

### Pagination cursor contract

For every paginated tool:

- `pagination.nextCursor` is an **opaque** server-issued token
- pass it back as `cursor` **exactly as returned**
- do **not** derive, edit, increment, or fabricate it
- do **not** reuse it across a different tool or different query flow

The server validates those boundaries and returns an actionable `INVALID_CURSOR`
error if a cursor is malformed, cross-tool, or stale for the current query
context.

### API keys (optional)

**Default search fallback order:** When you call `search_papers`, the server tries sources in order and uses the first that succeeds:

1. **Semantic Scholar** – Primary scholarly graph. Works without a key with lower limits; set `SEMANTIC_SCHOLAR_API_KEY` for higher limits.
2. **arXiv** – Free preprint and recency fallback; no key required.
3. **CORE API** – Disabled by default until explicitly enabled. Set `SCHOLAR_SEARCH_ENABLE_CORE=true` to include it in the broker. `CORE_API_KEY` is optional and raises limits when present.
4. **SerpApi Google Scholar** – Optional paid recall-recovery provider, disabled by default. Enable with `SCHOLAR_SEARCH_ENABLE_SERPAPI=true` and set `SERPAPI_API_KEY`. See [SerpApi pricing](https://serpapi.com/pricing).

OpenAlex remains an explicit provider surface rather than a default broker hop. Use the `*_openalex` tools for autocomplete, institution/source/topic pivots, or OpenAlex-native citation and author workflows.

`search_papers` is a **brokered single-page search**: it returns results from the first provider in the effective chain that succeeds and does **not** support cursor-based pagination. By default the effective chain is the order above, but you can:

- set `preferredProvider` to try one provider first and then fall back through the rest of the configured chain
- set `providerOrder` to override the provider chain for a single call (omitted providers are skipped for that request)
- set `SCHOLAR_SEARCH_PROVIDER_ORDER` to override the default broker order for a deployment
- use `search_papers_core`, `search_papers_semantic_scholar`, `search_papers_serpapi`, or `search_papers_arxiv` for single-provider searches

Provider names accepted in `preferredProvider`, `providerOrder`, and `SCHOLAR_SEARCH_PROVIDER_ORDER` are `core`, `semantic_scholar`, `arxiv`, and either `serpapi` or `serpapi_google_scholar`. Broker metadata continues to report the SerpApi provider as `serpapi_google_scholar`.

Every response includes a `brokerMetadata` field that makes this contract explicit:

```json
{
  "data": [...],
  "brokerMetadata": {
    "mode": "brokered_single_page",
    "providerUsed": "semantic_scholar",
    "continuationSupported": false
  }
}
```

| Field                  | Description                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------- |
| `mode`                 | Always `"brokered_single_page"` for `search_papers`.                                                 |
| `providerUsed`         | Which provider supplied the results: `core`, `semantic_scholar`, `serpapi_google_scholar`, `arxiv`, or `none` if no provider returned results. |
| `continuationSupported`| Always `false` — use `search_papers_bulk` for paginated retrieval.                                   |

When SerpApi supplies the results, the response looks like:

```json
{
  "data": [
    {
      "title": "Attention Is All You Need",
      "source": "serpapi_google_scholar",
      "sourceId": "result_id_from_scholar",
      "canonicalId": "10.xxxx/cluster-or-doi",
      "year": 2017,
      "citationCount": 80000
    }
  ],
  "brokerMetadata": {
    "mode": "brokered_single_page",
    "providerUsed": "serpapi_google_scholar",
    "continuationSupported": false
  }
}
```

`brokerMetadata` now also exposes:

- `attemptedProviders` - ordered provider decisions (`returned_results`, `returned_no_results`, `failed`, `skipped`)
- `semanticScholarOnlyFilters` - which requested filters forced the broker to skip non-compatible providers
- `recommendedPaginationTool` - currently always `search_papers_bulk` for exhaustive retrieval

#### Cross-provider paper ID portability

Brokered `search_papers` results normalize metadata across providers, but raw
provider IDs are **not automatically portable** into Semantic Scholar expansion
tools such as `get_paper_citations`, `get_paper_references`, `get_paper_authors`,
or the author-pivot flow that follows them. When a brokered result did **not**
come from Semantic Scholar:

- prefer `paper.recommendedExpansionId` when it is present
- check `paper.expansionIdStatus` before reusing any returned identifier
- if `paper.expansionIdStatus` is `not_portable`, do **not** retry with
  brokered `paperId`, `sourceId`, or `canonicalId`; resolve the paper through a
  DOI or a Semantic Scholar-native lookup first

This matters for brokered CORE results in particular: when no DOI is available,
`paper.canonicalId` may still be only a CORE-native identifier rather than a
Semantic Scholar-compatible expansion ID.

#### Provider order and filter-based skipping

Provider order controls *which providers are eligible and in what order they are attempted*, but compatibility rules still apply. If you request Semantic Scholar-only filters such as `publicationDateOrYear`, `fieldsOfStudy`, `publicationTypes`, `openAccessPdf`, or `minCitationCount`, the broker will skip `core` and `serpapi(_google_scholar)` even if they appear earlier in `providerOrder`. Example: `providerOrder=["core","semantic_scholar","arxiv"]` with `publicationDateOrYear="2020:2024"` will skip CORE and continue to Semantic Scholar because CORE cannot honor that filter.

### Enable/disable search channels

Control which sources are used in the `search_papers` fallback chain via environment variables:


| Variable                                 | Default | Description                                                            |
| ---------------------------------------- | ------- | ---------------------------------------------------------------------- |
| `SCHOLAR_SEARCH_ENABLE_CORE`             | `false` | Use CORE API. Disabled by default until explicitly enabled.             |
| `SCHOLAR_SEARCH_ENABLE_SEMANTIC_SCHOLAR` | `true`  | Use Semantic Scholar.                                                  |
| `SCHOLAR_SEARCH_ENABLE_SERPAPI`          | `false` | Use SerpApi Google Scholar (opt-in, **paid**). Set `SERPAPI_API_KEY`.  |
| `SCHOLAR_SEARCH_ENABLE_ARXIV`            | `true`  | Use arXiv as the default free fallback after Semantic Scholar.         |
| `SCHOLAR_SEARCH_PROVIDER_ORDER`          | `semantic_scholar,arxiv,core,serpapi_google_scholar` | Comma-separated default broker order for `search_papers`. Omit a provider to remove it from the default broker chain. Accepts `serpapi` as a shorthand for `serpapi_google_scholar`. |

SerpApi is disabled by default to prevent unexpected costs. When enabled without an API key, affected tool calls return a clear error rather than silently failing.

SerpApi-specific filters (`publicationDateOrYear`, `fieldsOfStudy`, `publicationTypes`, `openAccessPdf`, `minCitationCount`) bypass SerpApi in the fallback chain (same behaviour as CORE).

Example: enable SerpApi as an additional coverage fallback:

```json
"env": {
  "SCHOLAR_SEARCH_ENABLE_SERPAPI": "true",
  "SERPAPI_API_KEY": "your-serpapi-key-here"
}
```


Example: CORE and arXiv only (skip Semantic Scholar):

```json
"env": {
  "SCHOLAR_SEARCH_ENABLE_SEMANTIC_SCHOLAR": "false"
}
```

Example: prefer Semantic Scholar first, then arXiv for this deployment:

```json
"env": {
  "SCHOLAR_SEARCH_PROVIDER_ORDER": "semantic_scholar,arxiv"
}
```

### OpenAlex configuration

OpenAlex is exposed through the explicit `*_openalex` tools rather than the
default `search_papers` broker. The OpenAlex tools are enabled by default and do
not require an API key, but you can configure them with:

| Variable | Default | Description |
| --- | --- | --- |
| `SCHOLAR_SEARCH_ENABLE_OPENALEX` | `true` | Enable or disable the explicit OpenAlex tool family |
| `OPENALEX_API_KEY` | unset | Optional OpenAlex premium API key |
| `OPENALEX_MAILTO` | unset | Optional contact email for the OpenAlex polite pool; recommended for production use |

OpenAlex tool year inputs currently accept `YYYY`, `YYYY:YYYY`, `YYYY-YYYY`,
`YYYY-`, and `-YYYY`. The client currently uses conservative built-in pacing
and retry defaults (`min_interval=0.05s`, `max_retries=2`) rather than extra
environment variables.

### Paper enrichment configuration

Crossref and Unpaywall are exposed as explicit paper-enrichment tools rather
than broker providers. They do not participate in `search_papers` ranking or
fallback order. Use them after you already have a paper, DOI, or DOI-bearing
identifier.

| Variable | Default | Description |
| --- | --- | --- |
| `SCHOLAR_SEARCH_ENABLE_CROSSREF` | `true` | Enable explicit Crossref paper enrichment |
| `CROSSREF_MAILTO` | unset | Optional contact email included in Crossref requests and user agent metadata |
| `CROSSREF_TIMEOUT_SECONDS` | `30` | Timeout for Crossref enrichment requests |
| `SCHOLAR_SEARCH_ENABLE_UNPAYWALL` | `true` | Enable explicit Unpaywall OA enrichment |
| `UNPAYWALL_EMAIL` | unset | Required contact email for Unpaywall lookups |
| `UNPAYWALL_TIMEOUT_SECONDS` | `30` | Timeout for Unpaywall enrichment requests |

Known-item tools (`search_papers_match`, `get_paper_details`,
`resolve_citation`) and `search_papers_smart` also expose
`includeEnrichment=true` for opt-in Crossref + Unpaywall augmentation on the
final resolved paper or final smart hits. This enrichment is post-resolution and
does not change provider ordering, retrieval, or ranking.

### AI augmentation configuration

The raw retrieval tools remain the contract of record. The smart layer is
additive and opt-in.

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | unset | Optional API key for the OpenAI-backed smart layer. If the smart layer is enabled without a key, the runtime falls back to deterministic planning/synthesis helpers. |
| `SCHOLAR_SEARCH_ENABLE_AGENTIC` | `false` | Enable additive smart tools such as `search_papers_smart`. |
| `SCHOLAR_SEARCH_AGENTIC_PROVIDER` | `openai` | Provider bundle for the smart layer. Current supported values are `openai` and `deterministic`. |
| `SCHOLAR_SEARCH_PLANNER_MODEL` | `gpt-5.4-mini` | Planning / routing model for smart discovery. Lower-latency GPT-5.4 tier for query analysis and routing. |
| `SCHOLAR_SEARCH_SYNTHESIS_MODEL` | `gpt-5.4` | Synthesis model for grounded answers and theme labeling. |
| `SCHOLAR_SEARCH_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model name for smart ranking/indexing metadata. |
| `SCHOLAR_SEARCH_DISABLE_EMBEDDINGS` | `false` | Disable all embedding generation and embedding-based similarity paths. Smart workflows fall back to lexical scoring, and FAISS-backed workspace indexing is effectively disabled even if selected. |
| `SCHOLAR_SEARCH_AGENTIC_OPENAI_TIMEOUT_SECONDS` | `30` | Client-side timeout for OpenAI-backed smart-layer requests. Keeps planner, synthesis, and embedding calls from hanging on the critical path for minutes before degrading to deterministic or lexical fallbacks. |
| `SCHOLAR_SEARCH_AGENTIC_INDEX_BACKEND` | `memory` | Workspace index backend. `memory` is the recommended default for the current small saved result sets; `faiss` is an optional upgrade when the `ai-faiss` extra is installed and the deployment image includes it. |
| `SCHOLAR_SEARCH_SESSION_TTL_SECONDS` | `1800` | TTL for cached reusable `searchSessionId` result sets. |
| `SCHOLAR_SEARCH_ENABLE_AGENTIC_TRACE_LOG` | `false` | Emit local JSONL-style trace events for smart workflows. |

When the smart layer is enabled, the new tools are:

- `search_papers_smart` for concept-level discovery, query expansion, and multi-provider fusion
- `ask_result_set` for grounded QA, claim checks, and comparisons over a saved `searchSessionId`
- `map_research_landscape` for theme clustering, gaps, and disagreements
- `expand_research_graph` for compact citation/reference/author graph expansion

### Transport configuration

The server defaults to local **stdio** transport, which is the recommended mode for desktop MCP clients. FastMCP also supports HTTP-compatible transports for local development, integration testing, and controlled deployments:

| Variable | Default | Description |
| --- | --- | --- |
| `SCHOLAR_SEARCH_TRANSPORT` | `stdio` | One of `stdio`, `http`, `streamable-http`, or `sse` |
| `SCHOLAR_SEARCH_HTTP_HOST` | `127.0.0.1` | Host to bind when using an HTTP transport |
| `SCHOLAR_SEARCH_HTTP_PORT` | `8000` | Port to bind when using an HTTP transport |
| `SCHOLAR_SEARCH_HTTP_PATH` | `/mcp` | MCP endpoint path when using an HTTP transport |
| `SCHOLAR_SEARCH_HTTP_AUTH_TOKEN` | unset | Optional shared token required by the deployment wrapper for requests to the MCP endpoint |
| `SCHOLAR_SEARCH_HTTP_AUTH_HEADER` | `authorization` | Header name checked by the deployment wrapper; `authorization` expects `Bearer <token>` |
| `SCHOLAR_SEARCH_ALLOWED_ORIGINS` | unset | Optional comma-separated Origin allowlist enforced by the deployment wrapper |

> [!IMPORTANT]
> HTTP transport compatibility is available, but this repository does **not**
> yet ship a hardened public deployment profile. Before exposing the server
> beyond localhost, add Origin validation/allowlisting, authentication, and
> TLS in front of the ASGI app. This follows the MCP HTTP transport guidance
> confirmed via Context7: servers must validate `Origin` and should implement
> authentication for HTTP transports.

Use `scholar_search_mcp.server.build_http_app(...)` if you need to inject
deployment-specific Starlette middleware around the FastMCP ASGI app.

For private Azure hosting, this repository also ships
`scholar_search_mcp.deployment:app`, an ASGI wrapper that adds `/healthz`,
optional shared-token authentication, and optional Origin allowlisting in front
of the MCP app. The Azure deployment scaffold in [docs/azure-deployment.md](docs/azure-deployment.md)
uses that wrapper behind private Azure API Management and private endpoints so
trusted clients never see the backend token or upstream provider API keys.
The tracked Azure workflow is manual-only and supports a `bootstrap` mode for
first-time environment bring-up before the `full` private-runner deployment.

Example local/integration HTTP run:

```bash
SCHOLAR_SEARCH_TRANSPORT=streamable-http \
SCHOLAR_SEARCH_HTTP_HOST=0.0.0.0 \
SCHOLAR_SEARCH_HTTP_PORT=8000 \
python -m scholar_search_mcp
```

That path serves the FastMCP app directly over streamable HTTP at
`http://127.0.0.1:8000/mcp` by default. It is the simplest local option when
you want HTTP transport without the deployment wrapper.

If you want local parity with the HTTP wrapper used by Compose and Azure,
run:

```bash
PORT=8000 SCHOLAR_SEARCH_HTTP_HOST=127.0.0.1 scholar-search-mcp deployment-http
```

`scholar-search-mcp deployment-http` launches the same
`scholar_search_mcp.deployment:app` wrapper used by hosted HTTP deployments.
It prefers `PORT` when it is set, and otherwise falls back to
`SCHOLAR_SEARCH_HTTP_PORT`. That wrapper keeps the same streamable HTTP MCP
transport, adds `/healthz`, and optionally enforces
`SCHOLAR_SEARCH_HTTP_AUTH_TOKEN` and `SCHOLAR_SEARCH_ALLOWED_ORIGINS` for
`/mcp`.

In all of these local HTTP modes, clients talk only to your local MCP endpoint.
They do **not** need upstream provider API keys. Those keys, when present, stay
server-side and only let your local server use higher provider limits or
optional paid providers such as SerpApi.

### Docker MCP package (stdio)

For local MCP clients that launch servers as subprocesses, use the image in
stdio mode. For unpublished local iteration, build and run `scholar-search-mcp:local`.
For the reusable public package, use the published GHCR tag:

```bash
docker run --rm -i ghcr.io/joshuasundance-swca/scholar-search-mcp:latest
```

For a locally built image:

```bash
docker run --rm -i scholar-search-mcp:local
```

A Docker-backed MCP client entry typically looks like:

```json
{
  "mcpServers": {
    "scholar-search": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "ghcr.io/joshuasundance-swca/scholar-search-mcp:latest"]
    }
  }
}
```

This mode is ideal for local desktop MCP usage because the host launches and
owns the server process lifecycle.

The repo also ships `server.json` so the public OCI image and MCP package
metadata stay aligned for registry/discovery tooling. The public-package
workflow is tag-driven: a `v*` tag publishes the GHCR image first, then
publishes the same committed `server.json` metadata to the MCP Registry via
GitHub OIDC.

### Docker Compose (HTTP wrapper mode)

For local HTTP testing, MCP Inspector, or bridge-style integrations, this repo
ships `docker-compose.yaml` with localhost-only defaults. Compose now
explicitly starts the `deployment-http` subcommand, so HTTP wrapper behavior
does not depend on the image's default transport.

Compose intentionally keeps the container bind host and internal port fixed at
`0.0.0.0:8080`, because those are container-runtime details rather than
consumer-facing behavior. The compose file still lets downstream users override
the user-facing knobs that matter locally: transport, MCP path, provider keys,
provider toggles, auth, and the published host port mapping.

As a rule of thumb for this repo: make behavior configurable, keep secrets out
of git, and avoid exposing infrastructure internals as knobs unless a
downstream consumer actually benefits from changing them.

1. Copy `.env.example` to `.env`.
2. Fill in any optional provider keys you want to use.
3. Start the service:

```bash
docker compose -f docker-compose.yaml up --build
```

The service listens on `http://127.0.0.1:8000` by default, serves
`/healthz` for probes, and exposes MCP over `http://127.0.0.1:8000/mcp`.

```bash
curl http://127.0.0.1:8000/healthz
```

If you set `SCHOLAR_SEARCH_HTTP_AUTH_TOKEN` and leave
`SCHOLAR_SEARCH_HTTP_AUTH_HEADER=authorization`, the deployment wrapper expects
`Authorization: Bearer <token>` on `/mcp`. The checked-in Azure scaffold
overrides the header name to `x-backend-auth` and has API Management inject
that header for backend-only traffic. The published host defaults to
`127.0.0.1`; only change `SCHOLAR_SEARCH_PUBLISHED_HOST` when you intentionally
want the container reachable beyond the local machine.

If you leave the provider key fields blank, local clients still work. The
server falls back to the free/default provider paths where supported, and
SerpApi stays disabled by default.

### Docker Compose Inspector sidecar

For browser-based debugging without installing Node locally, use the dedicated
Inspector stack:

```bash
docker compose -f compose.inspector.yaml up --build
```

This stack keeps Inspector separate from the MCP server image and binds the UI
and proxy to localhost only:

- Inspector UI: `http://127.0.0.1:6274`
- Inspector proxy: `http://127.0.0.1:6277`

Inspector proxy authentication remains enabled by default. Use
`docker compose -f compose.inspector.yaml logs mcp-inspector` to read the
session token that Inspector prints on startup.

Inside Inspector, connect using Streamable HTTP and set:

- URL: `http://scholar-search-mcp:8080/mcp`
- Transport: `streamable-http`

`compose.inspector.yaml` accepts `IMAGE` overrides, so you can test a specific
tag without editing files:

```bash
IMAGE=ghcr.io/joshuasundance-swca/scholar-search-mcp:latest docker compose -f compose.inspector.yaml up
```

## Tools

Primary defaults for agents:

- Start with `search_papers_smart` when the task is concept-level discovery, a literature review, or a grounded follow-up workflow that should reuse a `searchSessionId`.
- Start with `search_papers` for quick literature discovery.
- Switch to `search_papers_bulk` for exhaustive retrieval or multi-page collection.
- For small targeted pages, prefer `search_papers` or `search_papers_semantic_scholar` instead of `search_papers_bulk`.
- Use `resolve_citation` for incomplete references, broken bibliography lines, and almost-right citations.
- Use `search_papers_match` or `get_paper_details` for known-item lookup.
- Use `get_paper_metadata_crossref`, `get_paper_open_access_unpaywall`, or
  `enrich_paper` once you already have a stable paper or DOI and want additive
  metadata or OA/PDF discovery.
- Expand with `get_paper_citations`, `get_paper_references`, and `search_authors`
  once you have a paper or author anchor.
- Use the `*_openalex` tools when you explicitly want OpenAlex-native DOI/ID
  lookup, OpenAlex cursor pagination, or OpenAlex author/citation semantics.
- Reach for `search_snippets` only when quote or phrase recovery is needed.
- On primary read tools, inspect `agentHints`, `clarification`, `resourceUris`,
  and `searchSessionId` before making the next tool call.


| Tool                             | Description                                                                                              |
| -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `search_papers_smart`            | Additive smart discovery entry point for concept-level research, literature reviews, known-item routing, and saved-result-set workflows. Supports `latencyProfile` (`fast`, `balanced`, `deep`) plus optional `providerBudget`, and returns smart-ranked hits plus `strategyMetadata`, `resourceUris`, `agentHints`, and a reusable `searchSessionId`. |
| `ask_result_set`                 | Grounded follow-up over a saved `searchSessionId`. Supports `qa`, `claim_check`, and `comparison` modes, accepts optional `latencyProfile`, and returns paper-level evidence plus suggested next questions. |
| `map_research_landscape`         | Cluster a saved `searchSessionId` into 3-5 themes, representative papers, gaps, disagreements, and suggested next searches. Accepts optional `latencyProfile`. |
| `expand_research_graph`          | Expand one or more paper anchors or a saved `searchSessionId` into a compact citation/reference/author graph with frontier ranking and reusable resource handles. Accepts optional `latencyProfile`. |
| `search_papers`                  | Primary entry point for quick literature discovery. Single-page best-effort brokered search. Default order: Semantic Scholar → arXiv → CORE → SerpApi Google Scholar, with CORE and SerpApi still subject to their enable flags. Use `brokerMetadata` to see where results came from and decide whether to broaden, narrow, paginate, or pivot. Optional filters: `limit`, `fields`, `year`, `venue`, `preferredProvider`, `providerOrder`, `publicationDateOrYear`, `fieldsOfStudy`, `publicationTypes`, `openAccessPdf`, `minCitationCount`. No pagination — for paginated retrieval use `search_papers_bulk`. |
| `search_papers_core`             | Single-page CORE-only search with the same normalized response shape as `search_papers`. Exposes only the inputs CORE actually honors: `query`, `limit`, and `year`. |
| `search_papers_semantic_scholar` | Single-page Semantic Scholar-only search with the same normalized response shape as `search_papers`. Exposes the Semantic Scholar-compatible inputs: `query`, `limit`, `fields`, `year`, `venue`, `publicationDateOrYear`, `fieldsOfStudy`, `publicationTypes`, `openAccessPdf`, and `minCitationCount`. |
| `search_papers_serpapi`          | Single-page SerpApi Google Scholar-only search. **Requires SerpApi** (`SCHOLAR_SEARCH_ENABLE_SERPAPI=true` + `SERPAPI_API_KEY`). Exposes only the inputs SerpApi actually honors: `query`, `limit`, and `year`. |
| `search_papers_arxiv`            | Single-page arXiv-only search with the same normalized response shape as `search_papers`. Exposes only the inputs arXiv actually honors: `query`, `limit`, and `year`. |
| `search_papers_openalex`         | Single-page OpenAlex-only search with the same normalized top-level response shape as `search_papers`. Exposes only the inputs OpenAlex explicitly honors here: `query`, `limit`, and `year`. |
| `search_papers_openalex_bulk`    | Cursor-paginated OpenAlex search for explicit OpenAlex retrieval flows. Treat `pagination.nextCursor` as opaque, pass it back unchanged as `cursor`, and keep it scoped to the same tool/query flow; supports `query`, `limit`, and `year`. |
| `paper_autocomplete_openalex`    | Lightweight OpenAlex work autocomplete for typeahead and known-item disambiguation before committing to a full work search. |
| `search_entities_openalex`       | Search OpenAlex `source`, `institution`, or `topic` entities for two-step pivot workflows. |
| `search_papers_openalex_by_entity` | Retrieve OpenAlex works constrained to one OpenAlex `source`, `institution`, or `topic` entity ID, with cursor pagination for explicit disambiguation workflows. |
| `search_papers_bulk`             | Primary exhaustive retrieval tool. Paginated bulk paper search (Semantic Scholar) with advanced boolean query syntax (up to 1,000 returned papers/call). The upstream bulk endpoint may ignore small `limit` values internally, so this server truncates returned data to the requested limit; prefer `search_papers` or `search_papers_semantic_scholar` for small targeted pages. Treat `pagination.nextCursor` as opaque, pass it back unchanged as `cursor`, and do not derive/edit/fabricate it; `pagination.hasMore` signals more results. |
| `resolve_citation`               | First-class citation-repair workflow for incomplete, malformed, or almost-right references. Stages identifier extraction, title recovery, quote/snippet recovery, and sparse metadata search; returns the best canonical paper candidate, alternatives, confidence, conflicts, and reusable `searchSessionId` metadata. Optional `includeEnrichment=true` enriches only `bestMatch.paper` after resolution. |
| `search_papers_match`            | Known-item lookup for messy or partial titles; finds the single paper whose title best matches the query string, falls back to fuzzy Semantic Scholar title search on exact-match 400/404 misses, and returns a structured no-match payload when the item still cannot be recovered. Optional `includeEnrichment=true` adds post-match Crossref + Unpaywall metadata to the final paper only. |
| `paper_autocomplete`             | Return paper title completions for a partial query (typeahead)                                           |
| `get_paper_details`              | Known-item lookup by identifier: get one paper by DOI, ArXiv ID, S2 ID, or URL. Optional `includeEnrichment=true` adds post-lookup Crossref + Unpaywall metadata without changing the base lookup path. |
| `get_paper_metadata_crossref`    | Explicit Crossref enrichment for one known paper or DOI. DOI-first, with optional title/bibliographic query fallback when no DOI can be resolved from the inputs. Returns a normalized Crossref work summary plus additive enrichment fields. |
| `get_paper_open_access_unpaywall` | Explicit Unpaywall enrichment for one known paper or DOI. DOI-only lookup for OA status, best OA URL, PDF URL, license, and DOAJ signal. Requires `UNPAYWALL_EMAIL`. |
| `enrich_paper`                   | Combined Crossref + Unpaywall enrichment orchestrator for one paper, DOI, or DOI-bearing identifier. Resolves DOI first, runs Crossref, then Unpaywall, and returns one merged `enrichments` payload plus per-provider results. |
| `get_paper_details_openalex`     | Known-item lookup using OpenAlex semantics: get one work by OpenAlex W-id, OpenAlex work URL, or DOI, with abstract reconstruction from OpenAlex's `abstract_inverted_index` when possible |
| `get_paper_citations`            | Citation chasing outward: papers that cite the given paper (`cited by`); for Semantic Scholar expansion prefer `paper.recommendedExpansionId`, and if `paper.expansionIdStatus` is `not_portable` resolve the paper through DOI or a Semantic Scholar-native lookup before expanding; treat `pagination.nextCursor` as opaque, pass it back unchanged as `cursor`, and keep it scoped to the same tool/query flow |
| `get_paper_citations_openalex`   | OpenAlex cited-by expansion using the work's `cited_by_api_url`; treat `pagination.nextCursor` as opaque, pass it back unchanged as `cursor`, and keep it scoped to the same tool/query flow |
| `get_paper_references`           | Citation chasing backward: references behind the given paper; for Semantic Scholar expansion prefer `paper.recommendedExpansionId`, and if `paper.expansionIdStatus` is `not_portable` resolve the paper through DOI or a Semantic Scholar-native lookup before expanding; treat `pagination.nextCursor` as opaque, pass it back unchanged as `cursor`, and keep it scoped to the same tool/query flow |
| `get_paper_references_openalex`  | OpenAlex backward-reference expansion that hydrates `referenced_works` through batched OpenAlex ID lookups and returns an opaque server-issued cursor for the next slice |
| `get_paper_authors`              | Authors of the given paper; for this Semantic Scholar expansion path prefer `paper.recommendedExpansionId`, and if `paper.expansionIdStatus` is `not_portable` resolve the paper through DOI or a Semantic Scholar-native lookup before expanding; treat `pagination.nextCursor` as opaque, pass it back unchanged as `cursor`, and keep it scoped to the same tool/query flow |
| `get_author_info`                | Author profile by Semantic Scholar author ID (typically from `search_authors` or `get_paper_authors`)   |
| `get_author_info_openalex`       | OpenAlex author profile by OpenAlex A-id or OpenAlex author URL                                          |
| `get_author_papers`              | Papers by author; requires a Semantic Scholar author ID; treat `pagination.nextCursor` as opaque, pass it back unchanged as `cursor`, and keep it scoped to the same tool/query flow; supports `publicationDateOrYear` |
| `search_authors`                 | Search for authors by name using a plain-text query; the server normalizes exact-name punctuation before calling Semantic Scholar, and common-name workflows should add affiliation, coauthor, venue, or topic clues before confirming the best candidate; treat `pagination.nextCursor` as opaque, pass it back unchanged as `cursor`, and keep it scoped to the same tool/query flow |
| `search_authors_openalex`        | Search OpenAlex authors by name for an explicit OpenAlex author workflow; confirm common-name matches with affiliation or profile metadata before expanding papers; treat `pagination.nextCursor` as opaque, pass it back unchanged as `cursor`, and keep it scoped to the same tool/query flow |
| `get_author_papers_openalex`     | Papers by OpenAlex author with optional `year` filtering and OpenAlex cursor pagination; this keeps the two-step OpenAlex author flow explicit: `search_authors_openalex`/`get_author_info_openalex` first, then expand papers |
| `batch_get_authors`              | Details for up to 1,000 author IDs in one call                                                           |
| `search_snippets`                | Special-purpose recovery tool for quote or phrase validation when title/keyword search is weak; returns snippet text, paper metadata, and score, and degrades provider 4xx/5xx failures to an empty result with retry guidance |
| `get_paper_recommendations`      | Similar papers for a given paper (GET single-seed)                                                       |
| `get_paper_recommendations_post` | Similar papers from positive and negative seed sets (POST multi-seed)                                    |
| `batch_get_papers`               | Details for up to 500 paper IDs                                                                          |
| `search_papers_serpapi_cited_by` | Explicit Google Scholar cited-by expansion through SerpApi. Use when recall recovery or citation discovery matters more than latency. |
| `search_papers_serpapi_versions` | Explicit Google Scholar all-versions expansion through SerpApi cluster IDs. |
| `get_author_profile_serpapi`     | Google Scholar author profile retrieval through SerpApi for explicit author-centric workflows. |
| `get_author_articles_serpapi`    | Paginated Google Scholar author article retrieval through SerpApi. |
| `get_paper_citation_formats`     | Citation export step for MLA, APA, BibTeX, etc. from a Google Scholar paper. **Requires SerpApi** (`SCHOLAR_SEARCH_ENABLE_SERPAPI=true` + `SERPAPI_API_KEY`). Pass `result_id=paper.scholarResultId` (not `paper.sourceId`) from a `serpapi_google_scholar` result. Single non-paginated response. |
| `get_serpapi_account_status`     | Read-only SerpApi quota and throughput snapshot for budget-aware routing and troubleshooting. |
| `get_provider_diagnostics`       | Live provider diagnostics showing recent status buckets, throttling/suppression state, retries, and fallback reasons across providers. |


## Resources and prompts

- Resource: `guide://scholar-search/agent-workflows` - compact onboarding guide for choosing tools and following pagination safely
- Resource: `paper://{paper_id}` - compact markdown + structured payload for a resolved paper
- Resource: `author://{author_id}` - compact markdown + structured payload for a resolved author
- Resource: `search://{searchSessionId}` - saved result set surfaced from tool outputs
- Resource: `trail://paper/{paper_id}?direction=citations|references` - compact citation/reference trail resource
- Prompt: `plan_scholar_search` - reusable planning prompt for raw-vs-smart literature-search workflows
- Prompt: `plan_smart_scholar_search` - smart-tool-first planning prompt for concept discovery
- Prompt: `triage_literature` - compact triage workflow for theme mapping and next-step selection
- Prompt: `plan_citation_chase` - citation-expansion planning prompt
- Prompt: `refine_query` - bounded query-refinement prompt for broad or noisy searches

Primary read-tool responses now also surface:

- `agentHints` - recommended next tools, retry guidance, and warnings
- `clarification` - bounded clarification fallback when the server cannot safely disambiguate on its own
- `resourceUris` - follow-on resources that compatible clients can open directly
- `searchSessionId` - reusable result-set handle for smart follow-up workflows and cached expansion/search trails

## Microsoft Packaging Assets

This repository keeps one universal MCP server surface and ships additive
packaging assets for Microsoft-oriented clients:

- `mcp-tools.core.json` - conservative raw-tool subset
- `mcp-tools.full.json` - full Microsoft-facing package including smart tools
- `microsoft-plugin.sample.json` - sample declarative-agent / plugin-oriented metadata

These assets target Streamable HTTP and compact tool outputs. They are
packaging guidance, not a separate runtime build.


## Testing with MCP Inspector

The recommended local path is the Docker sidecar workflow:

```bash
docker compose -f compose.inspector.yaml up --build
```

This keeps Inspector out of the production MCP image and binds Inspector ports
to localhost only.

If you prefer a host-installed Inspector, you can still run:

```bash
npm install -g @modelcontextprotocol/inspector
mcp-inspector python -m scholar_search_mcp
```

## Development

Install the package with development extras:

```bash
pip install -e .[dev]
```

Project dependencies are declared in `pyproject.toml`; there is no separate runtime `requirements.txt` to keep in sync.

Run the local test suite:

```bash
pytest
```

### Full local validation

The repo's CI-equivalent local gate is broader than `pytest` alone. For a
thorough local pass, run:

```bash
python -m pip check
pre-commit run --all-files
python -m pytest --cov=scholar_search_mcp --cov-report=term-missing --cov-fail-under=85
python -m mypy --config-file pyproject.toml
python -m ruff check .
python -m bandit -c pyproject.toml -r scholar_search_mcp
python -m build
python -m pip_audit . --progress-spinner off
```

If you prefer to invoke the heavier hook-managed checks through pre-commit,
`pre-commit run --hook-stage manual --all-files` runs the manual-stage
`pip check`, coverage, build, and `pip-audit` hooks defined in
`.pre-commit-config.yaml`.

When you touch Azure IaC, deployment docs, the Dockerfile, the APIM policy, or
the Azure deployment workflow, also run:

```bash
python scripts/validate_psrule_azure.py
python scripts/validate_deployment.py --skip-docker
```

For parity with the `Deploy Azure` workflow's full deployment validation path,
run:

```bash
python scripts/validate_deployment.py --require-az --require-docker --image-tag scholar-search-mcp:ci-validate
```

### GitHub Agentic Workflow smoke test

The repository now includes an agentic regression workflow source at
`.github/workflows/test-scholar-search.md` and its compiled lock file at
`.github/workflows/test-scholar-search.lock.yml`. After editing the Markdown
workflow, recompile it and then run the normal validation stack so pre-commit
can normalize the generated lock file:

```bash
gh aw compile test-scholar-search --dir .github/workflows
```

What this workflow does:

- Runs the agent against the local `scholar-search` MCP server inside GitHub
  Actions. Set the `GH_AW_MODEL_AGENT_COPILOT` Actions variable to `gpt-5.4`
  (or another model) to control which model is used.
- Exercises the primary golden paths instead of every tool: quick discovery,
  known-item lookup, bulk pagination, citation chasing, author pivot, and
  optional SerpApi citation export.
- **Evaluates agent UX quality** in every step: intuitiveness, unnecessary
  round trips, missing features, confusing field contracts, and dead-end
  responses. Produces a structured "UX friction summary" before creating any
  issue.
- Supports manual `workflow_dispatch` inputs so maintainers can run a default
  `smoke` pass, a broader `comprehensive` UX review, or a `feature_probe` with
  an optional focus prompt for a new feature or suspected rough edge.
- Produces a high-level smoke test that catches agent-facing workflow regressions
  that unit tests can miss and can turn concrete findings into one actionable
  GitHub issue for follow-on coding-agent work.
- Runs manually only. This is intentional because the workflow can consume
  repository secrets in a public repo.

How it runs in GitHub:

- The editable source of truth is `.github/workflows/test-scholar-search.md`.
- `gh aw compile ...` generates `.github/workflows/test-scholar-search.lock.yml`,
  which is the Actions workflow file GitHub actually runs.
- Once both files are committed to the default branch and the required secrets
  are configured, maintainers run the workflow manually with
  `workflow_dispatch` from the Actions tab.
- Manual dispatches can select `smoke`, `comprehensive`, or `feature_probe`
  mode and optionally pass a free-form focus prompt.

Required secrets and variables for this workflow:

- `COPILOT_GITHUB_TOKEN` is required. The GitHub Copilot CLI engine fails in
  the activation job before the repo checkout or MCP startup steps if this
  secret is not present.
- `GH_AW_MODEL_AGENT_COPILOT` (Actions variable, optional): controls the
  agent model. Set to `gpt-5.4` to use GPT-5.4. If unset, the engine uses
  its default model.
- `CORE_API_KEY` is optional.
- `SEMANTIC_SCHOLAR_API_KEY` is optional.

How to update and use it:

1. Edit `.github/workflows/test-scholar-search.md`.
2. Recompile it with `gh aw compile test-scholar-search --dir .github/workflows`.
3. Run the normal validation stack so pre-commit can normalize the generated
   lock file.
4. Commit both the `.md` source and `.lock.yml` output together.
5. Push the branch, then run `Test Scholar Search MCP` from the GitHub Actions
   UI. For on-demand UX reviews, use `workflow_dispatch` inputs to choose the
   run mode and provide an optional focus prompt such as a new feature, a
   provider-specific flow, or a confusing agent interaction to probe.

The repository also includes `.github/workflows/agentic-assign.yml`, a
lightweight workflow that automatically assigns GitHub Copilot to any issue
labeled both `agentic` and `needs-copilot`, unless the issue also carries
`needs-human`, `blocked`, or `no-agent`. It listens to direct `issues` events
and to completed `Test Scholar Search MCP` runs so verifier-created issues still
get assigned even when the original issue event does not fan out into a second
workflow run. For the actual Copilot assignment API call it prefers
`GH_AW_GITHUB_TOKEN`, then falls back to `COPILOT_GITHUB_TOKEN`, then to the
default Actions token. This avoids 403 failures on repositories where the
default `GITHUB_TOKEN` cannot perform Copilot issue assignment even though the
workflow has `issues: write` permissions.

The normal `Validate` workflow now also recompiles `test-scholar-search.md` on
CI and fails if `.github/workflows/test-scholar-search.lock.yml` is stale, so
pull requests cannot silently drift out of sync.

See [SECURITY.md](SECURITY.md) for the public-repo security posture and the
recommended private reporting path for vulnerabilities.

For this repo, there is no separate deployment step beyond checking in the
workflow source and compiled lock file. The workflow is "deployed" when GitHub
Actions sees the committed `.lock.yml` on the branch where it should run.

Install and run the configured pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

`pre-commit install` now installs both the fast `pre-commit` hooks and the
heavier `pre-push` gates configured in `.pre-commit-config.yaml`. Manual-stage
hooks are not invoked automatically; run `pre-commit run --hook-stage manual
--all-files` (or the direct commands above) when you want the full local gate.

The development extras now include `pytest`, `pytest-asyncio`, `pytest-cov`,
`ruff`, `mypy`, `bandit`, `build`, `pip-audit`, `types-defusedxml`, and
`pre-commit`.

GitHub dependency automation is configured for both Python packages and GitHub Actions via Dependabot, with pull requests checked by the dependency review workflow.

For maintainer orientation after the module split, start with `docs/agent-handoff.md`. The public MCP surface stays in `scholar_search_mcp/server.py`, while implementation now lives in `scholar_search_mcp/dispatch.py`, `scholar_search_mcp/search.py`, `scholar_search_mcp/tools.py`, `scholar_search_mcp/runtime.py`, `scholar_search_mcp/models/`, and provider subpackages under `scholar_search_mcp/clients/`.

## Guides

- [GitHub Copilot Instructions](.github/copilot-instructions.md) - repo-specific guidance for GitHub Copilot and the GitHub cloud coding agent, including workflow defaults and durable planning expectations.
- [Agent Handoff](docs/agent-handoff.md) - current repo status, validation commands, and next recommended work for follow-on agents.
- [Scholar Search Golden Paths](docs/golden-paths.md) - primary personas, workflow defaults, success signals, and future workflow-oriented follow-up work.
- [Azure Deployment](docs/azure-deployment.md) - deployment modes, required secrets and variables, and validation paths for the private Azure rollout.
- [Azure Architecture](docs/azure-architecture.md) - trust boundaries, runtime topology, and credential separation for the Azure scaffold.
- [Azure Security Model](docs/azure-security-model.md) - credential classes, Key Vault usage, and backend-auth separation in the Azure rollout.
- [Provider Upgrade Program](docs/provider-upgrade-program.md) - provider roles, latency profiles, diagnostics, benchmark corpus, and acceptance gates for the reliability-first provider upgrade.
- [OpenAlex API Guide](docs/openalex-api-guide.md) - implementation-focused guidance for the repo's explicit OpenAlex MCP surface, including authentication, credit-based limits, paging, `/works` semantics, and normalization caveats.
- [Semantic Scholar API Guide](docs/semantic-scholar-api-guide.md) - practical guidance for respectful and effective Semantic Scholar API usage with async rate limiting, retries, and `.env`-based local development.
- [SerpApi Google Scholar Guide](docs/serpapi-google-scholar-api-guide.md) - deep research notes on SerpApi capabilities, tradeoffs, and cost/compliance considerations; the repo now ships the explicit cited-by, versions, author, account, and citation-format flows documented there.
- [FastMCP Migration Plan](docs/fastmcp-migration-plan.md) - historical architecture rationale for the FastMCP migration and compatibility surface.

## License

MIT

## Links

- [CORE API v3 Documentation](https://api.core.ac.uk/docs/v3)
- [Semantic Scholar API](https://api.semanticscholar.org/api-docs)
- [arXiv API User's Manual](https://info.arxiv.org/help/api/user-manual.html)
