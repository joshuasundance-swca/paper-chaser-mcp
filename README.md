# Paper Chaser MCP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Fork status:** This repository is a divergent fork mid-rename. The code,
> package, CLI, public MCP branding, and package/image identifiers now use
> `paper-chaser-mcp`. The only remaining old-project reference is the current
> GitHub repository slug, which still uses `scholar-search-mcp` until the repo
> itself is renamed.

An MCP server for academic research — search papers, chase citations, look up authors, repair broken references, explore species dossiers, and retrieve regulatory text, all from one FastMCP server that AI assistants can call directly.

**Providers:** [Semantic Scholar](https://www.semanticscholar.org/product/api) · [arXiv](https://info.arxiv.org/help/api/user-manual.html) · [OpenAlex](https://developers.openalex.org/) · [CORE](https://api.core.ac.uk/docs/v3) · [SerpApi Google Scholar](https://serpapi.com/google-scholar-api) (opt-in, paid) · [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) · [Unpaywall](https://unpaywall.org/products/api) · [ECOS](https://ecos.fws.gov/) · [FederalRegister.gov](https://www.federalregister.gov/developers/documentation/api/v1) · [GovInfo](https://api.govinfo.gov/docs/)

---

## Contents

- [What can it do?](#what-can-it-do)
- [Quick tool decision guide](#quick-tool-decision-guide)
- [Installation](#installation)
- [Configuration](#configuration)
- [Tools](#tools)
- [Resources and prompts](#resources-and-prompts)
- [Development](#development)
- [Guides](#guides)
- [License](#license)

---

## What can it do?

### Academic paper search
- **Brokered single-page search** (`search_papers`) — tries Semantic Scholar → arXiv → CORE → SerpApi in order; returns `brokerMetadata` with `nextStepHint` so agents know what to do next
- **Exhaustive paginated retrieval** (`search_papers_bulk`) — cursor-based bulk traversal up to 1,000 papers/call; default ordering is **not relevance-ranked**, read `retrievalNote` each page
- **Provider-specific searches** — `search_papers_semantic_scholar`, `search_papers_arxiv`, `search_papers_core`, `search_papers_serpapi`, `search_papers_openalex`, `search_papers_openalex_bulk`

### Smart research layer *(requires `[ai]` extra + `PAPER_CHASER_ENABLE_AGENTIC=true`)*
- **`search_papers_smart`** — concept-level discovery with query expansion, multi-provider fusion, reranking, and a reusable `searchSessionId`; supports `latencyProfile` (`fast` / `balanced` / `deep`) and an optional `providerBudget`
- **`ask_result_set`** — grounded QA, claim checks, and comparisons over a saved result set
- **`map_research_landscape`** — theme clustering, gaps, and suggested next searches
- **`expand_research_graph`** — compact citation/reference/author graph expansion from anchors or a saved session

### Known-item lookup and citation repair
- **`resolve_citation`** — stages identifier extraction, title recovery, snippet search, and sparse metadata; abstains on regulatory or non-paper references and steers agents to the right tool
- **`search_papers_match`** — messy-title lookup with fuzzy Semantic Scholar fallback and cross-provider exact-title confirmation
- **`get_paper_details`** — lookup by DOI, arXiv ID, Semantic Scholar ID, or URL
- **`paper_autocomplete`** — typeahead completions for partial titles

### Citations, references, and authors
- **`get_paper_citations`** / **`get_paper_references`** — cursor-paginated forward and backward citation chasing
- **`search_authors`** → **`get_author_info`** → **`get_author_papers`** — full author pivot workflow
- **`batch_get_papers`** / **`batch_get_authors`** — fetch up to 500 papers or 1,000 authors in one call
- **`get_paper_recommendations`** / **`get_paper_recommendations_post`** — similar papers by single or multi-seed

### Paper enrichment and OA discovery
- **`enrich_paper`** — combined Crossref + Unpaywall orchestrator; opt in with `includeEnrichment=true` on lookup and smart tools
- **`get_paper_metadata_crossref`** / **`get_paper_open_access_unpaywall`** — explicit per-provider enrichment

### OpenAlex-native workflows
- **`get_paper_details_openalex`** — work lookup by OpenAlex W-id or DOI with abstract reconstruction
- **`get_paper_citations_openalex`** / **`get_paper_references_openalex`** — OpenAlex citation and reference traversal
- **`search_authors_openalex`** → **`get_author_info_openalex`** → **`get_author_papers_openalex`** — explicit OpenAlex author pivot
- **`search_entities_openalex`** / **`search_papers_openalex_by_entity`** — source, institution, and topic pivots
- **`paper_autocomplete_openalex`** — OpenAlex typeahead

### ECOS species dossiers *(U.S. Fish and Wildlife Service)*
- **`search_species_ecos`** → **`get_species_profile_ecos`** → **`list_species_documents_ecos`** → **`get_document_text_ecos`**
- Returns species listings, recovery plans, five-year reviews, biological opinions, and conservation-plan links; converts PDF/HTML documents to Markdown with bounded timeouts

### Federal Register and CFR regulatory text
- **`search_federal_register`** — keyless discovery on FederalRegister.gov
- **`get_federal_register_document`** — authoritative GovInfo retrieval for one notice or rule, with FederalRegister.gov HTML fallback
- **`get_cfr_text`** — authoritative CFR part/section XML from GovInfo; requires `GOVINFO_API_KEY`

### SerpApi extras *(opt-in, paid)*
- `search_papers_serpapi_cited_by`, `search_papers_serpapi_versions`, `get_author_profile_serpapi`, `get_author_articles_serpapi`, `get_paper_citation_formats`, `get_serpapi_account_status`

### Infrastructure
- **Provider execution policy** — shared retries with jitter, bounded concurrency, suppression/circuit-breaker state
- **Agent UX metadata** — every primary read tool returns `agentHints`, `clarification`, `resourceUris`, and (where applicable) a reusable `searchSessionId`
- **Structured outputs** — tools return structured content, not JSON-in-text blobs
- **`get_provider_diagnostics`** — live provider health, throttle state, and retry counts

---

## Quick tool decision guide

| Goal | Start here |
|---|---|
| Concept discovery or literature review | `search_papers_smart` (smart layer) |
| Quick paper search | `search_papers` → read `brokerMetadata.nextStepHint` |
| All papers on a topic / multi-page | `search_papers_bulk` (cursor loop) |
| Broken or incomplete citation | `resolve_citation` |
| Known paper by messy title | `search_papers_match` |
| Known paper by DOI / arXiv ID / URL | `get_paper_details` |
| Papers that cite X / refs behind X | `get_paper_citations` / `get_paper_references` |
| Author's work | `search_authors` → `get_author_info` → `get_author_papers` |
| OA status or full-text PDF | `get_paper_open_access_unpaywall` or `enrich_paper` |
| OpenAlex DOI/ID, cursor pagination, entity pivots | `*_openalex` tools |
| Species dossier / recovery documents | `search_species_ecos` → `get_species_profile_ecos` → `list_species_documents_ecos` |
| Federal Register notice or rule | `search_federal_register` → `get_federal_register_document` |
| CFR regulatory text | `get_cfr_text` |
| Quote or phrase recovery (last resort) | `search_snippets` |
| Provider health | `get_provider_diagnostics` |

After `search_papers`: read `brokerMetadata.nextStepHint`.
After `search_papers_smart`: reuse `searchSessionId` with `ask_result_set`, `map_research_landscape`, or `expand_research_graph`.
For Semantic Scholar expansion tools: prefer `paper.recommendedExpansionId`; if `paper.expansionIdStatus` is `not_portable`, resolve through DOI first.

---

## Installation

Because this fork has diverged from the upstream PyPI package, install from
source until the rename and republish are complete:

```bash
pip install -e .
```

Optional extras for the additive AI layer:

```bash
pip install -e .[ai]
```

Optional FAISS backend extras:

```bash
pip install -e .[ai,ai-faiss]
```

### Local Quick Start

The Azure deployment scaffold is optional. Local `python -m paper_chaser_mcp`
still defaults to `stdio` transport and does not require any Azure deployment
variables.

If you want a local env template for shell runs or Docker, copy
`.env.example` to `.env`, fill in only the providers you use, and keep the
resulting file uncommitted. The repo ignores `.env`, `.env.local`, other
`.env.*` variants except `.env.example`, and `.local-*` planning/artifact
files. Docker now keeps those local-only files out of the image build context.

## Configuration

### Configuration overview

The repo supports four common setup patterns:

| Setup | Default transport | Start here | Main configuration knobs |
| --- | --- | --- | --- |
| Desktop MCP client (`python -m` / `paper-chaser-mcp`) | `stdio` | Claude Desktop or Cursor examples below | Provider toggles, provider API keys, optional smart-layer settings |
| Direct local HTTP run | `streamable-http` or `http` when you opt in | Transport section below | `PAPER_CHASER_TRANSPORT`, `PAPER_CHASER_HTTP_HOST`, `PAPER_CHASER_HTTP_PORT`, `PAPER_CHASER_HTTP_PATH` |
| Local Docker Compose | HTTP wrapper over `streamable-http` | `docker-compose.yaml` section below | `.env.example` values plus `PAPER_CHASER_PUBLISHED_HOST` / `PAPER_CHASER_PUBLISHED_PORT` |
| Private Azure deployment | HTTP wrapper behind APIM | [docs/azure-deployment.md](docs/azure-deployment.md) | App env vars plus Azure secrets, Bicep params, and deployment workflow settings |

For local development, copy `.env.example` to `.env` or `.env.local` and fill in only the providers you actually use. That file is the public local-config contract for shell runs and Docker Compose. Azure deployment identifiers and secrets are intentionally documented separately in [docs/azure-deployment.md](docs/azure-deployment.md), not in `.env.example`.

### Environment variable map

The application-level variables are grouped like this:

| Group | Variables | Notes |
| --- | --- | --- |
| Search broker | `PAPER_CHASER_ENABLE_CORE`, `PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR`, `PAPER_CHASER_ENABLE_ARXIV`, `PAPER_CHASER_ENABLE_SERPAPI`, `PAPER_CHASER_PROVIDER_ORDER`, provider API keys | Controls the default `search_papers` fallback chain |
| Explicit provider/tool families | `PAPER_CHASER_ENABLE_OPENALEX`, `PAPER_CHASER_ENABLE_CROSSREF`, `PAPER_CHASER_ENABLE_UNPAYWALL`, `PAPER_CHASER_ENABLE_ECOS`, `PAPER_CHASER_ENABLE_FEDERAL_REGISTER`, `PAPER_CHASER_ENABLE_GOVINFO_CFR` | These families are explicit tool surfaces rather than broker hops |
| Provider runtime tuning | `CROSSREF_TIMEOUT_SECONDS`, `UNPAYWALL_TIMEOUT_SECONDS`, `ECOS_TIMEOUT_SECONDS`, `ECOS_DOCUMENT_TIMEOUT_SECONDS`, `ECOS_DOCUMENT_CONVERSION_TIMEOUT_SECONDS`, `ECOS_MAX_DOCUMENT_SIZE_MB`, `ECOS_VERIFY_TLS`, `ECOS_CA_BUNDLE`, `FEDERAL_REGISTER_TIMEOUT_SECONDS`, `GOVINFO_TIMEOUT_SECONDS`, `GOVINFO_DOCUMENT_TIMEOUT_SECONDS`, `GOVINFO_MAX_DOCUMENT_SIZE_MB` | Timeouts, size bounds, and TLS behavior |
| Smart layer | `OPENAI_API_KEY`, `PAPER_CHASER_ENABLE_AGENTIC`, `PAPER_CHASER_AGENTIC_PROVIDER`, `PAPER_CHASER_PLANNER_MODEL`, `PAPER_CHASER_SYNTHESIS_MODEL`, `PAPER_CHASER_EMBEDDING_MODEL`, `PAPER_CHASER_DISABLE_EMBEDDINGS`, `PAPER_CHASER_AGENTIC_OPENAI_TIMEOUT_SECONDS`, `PAPER_CHASER_AGENTIC_INDEX_BACKEND`, `PAPER_CHASER_SESSION_TTL_SECONDS`, `PAPER_CHASER_ENABLE_AGENTIC_TRACE_LOG` | Additive smart workflows only |
| Direct HTTP runtime | `PAPER_CHASER_TRANSPORT`, `PAPER_CHASER_HTTP_HOST`, `PAPER_CHASER_HTTP_PORT`, `PAPER_CHASER_HTTP_PATH` | Used for direct shell runs and hosted HTTP bindings |
| HTTP wrapper security | `PAPER_CHASER_HTTP_AUTH_TOKEN`, `PAPER_CHASER_HTTP_AUTH_HEADER`, `PAPER_CHASER_ALLOWED_ORIGINS` | Enforced by the deployment wrapper rather than the raw stdio runtime |
| Docker Compose publish settings | `PAPER_CHASER_PUBLISHED_HOST`, `PAPER_CHASER_PUBLISHED_PORT` | Controls the host-side port mapping only |

The distinction that tends to look inconsistent at first is intentional:

- `.env.example` includes the shared local-shell and compose-facing knobs.
- `PAPER_CHASER_HTTP_HOST` and `PAPER_CHASER_HTTP_PORT` are supported runtime variables for direct shell runs and hosted deployments, but Compose intentionally does not expose them because the container bind host and internal port stay fixed at `0.0.0.0:8080`.
- Compose exposes `PAPER_CHASER_PUBLISHED_HOST` and `PAPER_CHASER_PUBLISHED_PORT` instead, because those are the user-facing knobs for local HTTP access.

### Claude Desktop

Edit the config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add the following. In this example CORE is disabled and Semantic Scholar plus arXiv are enabled:

```json
{
  "mcpServers": {
    "paper-chaser": {
      "command": "python",
      "args": ["-m", "paper_chaser_mcp"],
      "env": {
        "PAPER_CHASER_ENABLE_CORE": "false",
        "PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR": "true",
        "PAPER_CHASER_ENABLE_ARXIV": "true"
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
    "paper-chaser": {
      "command": "python",
      "args": ["-m", "paper_chaser_mcp"],
      "env": {
        "CORE_API_KEY": "your-core-api-key-here",
        "SEMANTIC_SCHOLAR_API_KEY": "your-semantic-scholar-api-key-here",
        "PAPER_CHASER_ENABLE_CORE": "true",
        "PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR": "true",
        "PAPER_CHASER_ENABLE_ARXIV": "true"
      }
    }
  }
}
```

For regulation-oriented workflows, `search_federal_register` is keyless, while
`get_federal_register_document` and `get_cfr_text` use `GOVINFO_API_KEY` for
authoritative GovInfo retrieval. Without that key, Federal Register lookups can
still fall back to FederalRegister.gov HTML when a document number is known, but
CFR retrieval remains GovInfo-only.

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

### Search broker configuration

These settings control the default `search_papers` broker path. Provider-specific
families such as OpenAlex, Crossref, Unpaywall, ECOS, Federal Register, and
GovInfo are documented separately below because they are explicit tool surfaces,
not broker fallbacks.

**Default search fallback order:** When you call `search_papers`, the server tries sources in order and uses the first that succeeds:

1. **Semantic Scholar** – Primary scholarly graph. Works without a key with lower limits; set `SEMANTIC_SCHOLAR_API_KEY` for higher limits.
2. **arXiv** – Free preprint and recency fallback; no key required.
3. **CORE API** – Disabled by default until explicitly enabled. Set `PAPER_CHASER_ENABLE_CORE=true` to include it in the broker. `CORE_API_KEY` is optional and raises limits when present.
4. **SerpApi Google Scholar** – Optional paid recall-recovery provider, disabled by default. Enable with `PAPER_CHASER_ENABLE_SERPAPI=true` and set `SERPAPI_API_KEY`. See [SerpApi pricing](https://serpapi.com/pricing).

OpenAlex remains an explicit provider surface rather than a default broker hop. Use the `*_openalex` tools for autocomplete, institution/source/topic pivots, or OpenAlex-native citation and author workflows.

`search_papers` is a **brokered single-page search**: it returns results from the first provider in the effective chain that succeeds and does **not** support cursor-based pagination. By default the effective chain is the order above, but you can:

- set `preferredProvider` to try one provider first and then fall back through the rest of the configured chain
- set `providerOrder` to override the provider chain for a single call (omitted providers are skipped for that request)
- set `PAPER_CHASER_PROVIDER_ORDER` to override the default broker order for a deployment
- use `search_papers_core`, `search_papers_semantic_scholar`, `search_papers_serpapi`, or `search_papers_arxiv` for single-provider searches

Provider names accepted in `preferredProvider`, `providerOrder`, and `PAPER_CHASER_PROVIDER_ORDER` are `core`, `semantic_scholar`, `arxiv`, and either `serpapi` or `serpapi_google_scholar`. Broker metadata continues to report the SerpApi provider as `serpapi_google_scholar`.

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
| `PAPER_CHASER_ENABLE_CORE`             | `false` | Use CORE API. Disabled by default until explicitly enabled.             |
| `PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR` | `true`  | Use Semantic Scholar.                                                  |
| `PAPER_CHASER_ENABLE_SERPAPI`          | `false` | Use SerpApi Google Scholar (opt-in, **paid**). Set `SERPAPI_API_KEY`.  |
| `PAPER_CHASER_ENABLE_ARXIV`            | `true`  | Use arXiv as the default free fallback after Semantic Scholar.         |
| `PAPER_CHASER_PROVIDER_ORDER`          | `semantic_scholar,arxiv,core,serpapi_google_scholar` | Comma-separated default broker order for `search_papers`. Omit a provider to remove it from the default broker chain. Accepts `serpapi` as a shorthand for `serpapi_google_scholar`. |

SerpApi is disabled by default to prevent unexpected costs. When enabled without an API key, affected tool calls return a clear error rather than silently failing.

SerpApi-specific filters (`publicationDateOrYear`, `fieldsOfStudy`, `publicationTypes`, `openAccessPdf`, `minCitationCount`) bypass SerpApi in the fallback chain (same behaviour as CORE).

Example: enable SerpApi as an additional coverage fallback:

```json
"env": {
  "PAPER_CHASER_ENABLE_SERPAPI": "true",
  "SERPAPI_API_KEY": "your-serpapi-key-here"
}
```


Example: CORE and arXiv only (skip Semantic Scholar):

```json
"env": {
  "PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR": "false"
}
```

Example: prefer Semantic Scholar first, then arXiv for this deployment:

```json
"env": {
  "PAPER_CHASER_PROVIDER_ORDER": "semantic_scholar,arxiv"
}
```

<details>
<summary><strong>OpenAlex configuration</strong></summary>

OpenAlex is exposed through the explicit `*_openalex` tools rather than the
default `search_papers` broker. The OpenAlex tools are enabled by default and do
not require an API key, but you can configure them with:

| Variable | Default | Description |
| --- | --- | --- |
| `PAPER_CHASER_ENABLE_OPENALEX` | `true` | Enable or disable the explicit OpenAlex tool family |
| `OPENALEX_API_KEY` | unset | Optional OpenAlex premium API key |
| `OPENALEX_MAILTO` | unset | Optional contact email for the OpenAlex polite pool; recommended for production use |

OpenAlex tool year inputs currently accept `YYYY`, `YYYY:YYYY`, `YYYY-YYYY`,
`YYYY-`, and `-YYYY`. The client currently uses conservative built-in pacing
and retry defaults (`min_interval=0.05s`, `max_retries=2`) rather than extra
environment variables.

</details>

<details>
<summary><strong>Paper enrichment configuration</strong></summary>

Crossref and Unpaywall are exposed as explicit paper-enrichment tools rather
than broker providers. They do not participate in `search_papers` ranking or
fallback order. Use them after you already have a paper, DOI, or DOI-bearing
identifier.

| Variable | Default | Description |
| --- | --- | --- |
| `PAPER_CHASER_ENABLE_CROSSREF` | `true` | Enable explicit Crossref paper enrichment |
| `CROSSREF_MAILTO` | unset | Optional contact email included in Crossref requests and user agent metadata |
| `CROSSREF_TIMEOUT_SECONDS` | `30` | Timeout for Crossref enrichment requests |
| `PAPER_CHASER_ENABLE_UNPAYWALL` | `true` | Enable explicit Unpaywall OA enrichment |
| `UNPAYWALL_EMAIL` | unset | Required contact email for Unpaywall lookups |
| `UNPAYWALL_TIMEOUT_SECONDS` | `30` | Timeout for Unpaywall enrichment requests |

Known-item tools (`search_papers_match`, `get_paper_details`,
`resolve_citation`) and `search_papers_smart` also expose
`includeEnrichment=true` for opt-in Crossref + Unpaywall augmentation on the
final resolved paper or final smart hits. This enrichment is post-resolution and
does not change provider ordering, retrieval, or ranking.

</details>

<details>
<summary><strong>ECOS configuration</strong></summary>

ECOS is exposed as a separate species/document tool family rather than a
`search_papers` provider. It is enabled by default and uses structured Pull
Reports plus the species-profile JSON that powers the public ECOS species page.

| Variable | Default | Description |
| --- | --- | --- |
| `PAPER_CHASER_ENABLE_ECOS` | `true` | Enable or disable the ECOS species/document tool family |
| `ECOS_BASE_URL` | `https://ecos.fws.gov` | Base URL for ECOS species, Pull Reports, and document links |
| `ECOS_TIMEOUT_SECONDS` | `30` | Timeout for ECOS species/Pull Reports requests |
| `ECOS_DOCUMENT_TIMEOUT_SECONDS` | `60` | Timeout for ECOS document fetches before Markdown conversion |
| `ECOS_DOCUMENT_CONVERSION_TIMEOUT_SECONDS` | `60` | Timeout for ECOS Markdown conversion after the document bytes have been fetched |
| `ECOS_MAX_DOCUMENT_SIZE_MB` | `25` | Maximum fetched document size before returning `extractionStatus=too_large` |
| `ECOS_VERIFY_TLS` | `true` | Verify TLS certificates for ECOS species, Pull Reports, and document requests |
| `ECOS_CA_BUNDLE` | unset | Optional CA bundle path to use when ECOS TLS verification needs a custom trust store |

The core package now includes `markitdown[pdf]` so `get_document_text_ecos` can
convert PDF, HTML, and text-like ECOS documents to Markdown with plugins kept
off by default.

</details>

<details>
<summary><strong>Federal Register and GovInfo configuration</strong></summary>

Federal Register discovery (`search_federal_register`) is keyless. GovInfo-backed
retrieval (`get_federal_register_document`, `get_cfr_text`) uses `GOVINFO_API_KEY`
for authoritative content. Without that key, Federal Register lookups can still
fall back to FederalRegister.gov HTML when a document number is known, but CFR
retrieval remains GovInfo-only.

| Variable | Default | Description |
| --- | --- | --- |
| `PAPER_CHASER_ENABLE_FEDERAL_REGISTER` | `true` | Enable or disable the Federal Register discovery tool family |
| `PAPER_CHASER_ENABLE_GOVINFO_CFR` | `true` | Enable or disable GovInfo-backed Federal Register document and CFR retrieval |
| `GOVINFO_API_KEY` | unset | Required for authoritative `get_federal_register_document` and `get_cfr_text` calls; without a key, Federal Register lookup falls back to FederalRegister.gov HTML for known document numbers |
| `FEDERAL_REGISTER_TIMEOUT_SECONDS` | `30` | Timeout for FederalRegister.gov discovery requests |
| `GOVINFO_TIMEOUT_SECONDS` | `30` | Timeout for GovInfo metadata and granule requests |
| `GOVINFO_DOCUMENT_TIMEOUT_SECONDS` | `60` | Timeout for GovInfo document fetches before Markdown conversion |
| `GOVINFO_MAX_DOCUMENT_SIZE_MB` | `25` | Maximum fetched GovInfo document size before returning a structured `too_large` warning |

</details>

<details>
<summary><strong>AI augmentation configuration</strong></summary>

The raw retrieval tools remain the contract of record. The smart layer is
additive and opt-in.

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | unset | Optional API key for the OpenAI-backed smart layer. If the smart layer is enabled without a key, the runtime falls back to deterministic planning/synthesis helpers. |
| `PAPER_CHASER_ENABLE_AGENTIC` | `false` | Enable additive smart tools such as `search_papers_smart`. |
| `PAPER_CHASER_AGENTIC_PROVIDER` | `openai` | Provider bundle for the smart layer. Current supported values are `openai` and `deterministic`. |
| `PAPER_CHASER_PLANNER_MODEL` | `gpt-5.4-mini` | Planning / routing model for smart discovery. Lower-latency GPT-5.4 tier for query analysis and routing. |
| `PAPER_CHASER_SYNTHESIS_MODEL` | `gpt-5.4` | Synthesis model for grounded answers and theme labeling. |
| `PAPER_CHASER_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model name for smart ranking/indexing metadata. |
| `PAPER_CHASER_DISABLE_EMBEDDINGS` | `true` | Disable all embedding generation and embedding-based similarity paths. Smart workflows fall back to lexical scoring, and FAISS-backed workspace indexing is effectively disabled even if selected. |
| `PAPER_CHASER_AGENTIC_OPENAI_TIMEOUT_SECONDS` | `30` | Client-side timeout for OpenAI-backed smart-layer requests. Keeps planner, synthesis, and embedding calls from hanging on the critical path for minutes before degrading to deterministic or lexical fallbacks. |
| `PAPER_CHASER_AGENTIC_INDEX_BACKEND` | `memory` | Workspace index backend. `memory` is the recommended default for the current small saved result sets; `faiss` is an optional upgrade when the `ai-faiss` extra is installed and the deployment image includes it. |
| `PAPER_CHASER_SESSION_TTL_SECONDS` | `1800` | TTL for cached reusable `searchSessionId` result sets. |
| `PAPER_CHASER_ENABLE_AGENTIC_TRACE_LOG` | `false` | Emit local JSONL-style trace events for smart workflows. |

When the smart layer is enabled, the new tools are:

- `search_papers_smart` for concept-level discovery, query expansion, and multi-provider fusion
- `ask_result_set` for grounded QA, claim checks, and comparisons over a saved `searchSessionId`
- `map_research_landscape` for theme clustering, gaps, and disagreements
- `expand_research_graph` for compact citation/reference/author graph expansion

</details>

<details>
<summary><strong>Transport configuration</strong></summary>

The server defaults to local **stdio** transport, which is the recommended mode for desktop MCP clients. FastMCP also supports HTTP-compatible transports for local development, integration testing, and controlled deployments:

| Variable | Default | Description |
| --- | --- | --- |
| `PAPER_CHASER_TRANSPORT` | `stdio` | One of `stdio`, `http`, `streamable-http`, or `sse` |
| `PAPER_CHASER_HTTP_HOST` | `127.0.0.1` | Host to bind when using an HTTP transport |
| `PAPER_CHASER_HTTP_PORT` | `8000` | Port to bind when using an HTTP transport |
| `PAPER_CHASER_HTTP_PATH` | `/mcp` | MCP endpoint path when using an HTTP transport |
| `PAPER_CHASER_HTTP_AUTH_TOKEN` | unset | Optional shared token required by the deployment wrapper for requests to the MCP endpoint |
| `PAPER_CHASER_HTTP_AUTH_HEADER` | `authorization` | Header name checked by the deployment wrapper; `authorization` expects `Bearer <token>` |
| `PAPER_CHASER_ALLOWED_ORIGINS` | unset | Optional comma-separated Origin allowlist enforced by the deployment wrapper |

`.env.example` intentionally documents `PAPER_CHASER_TRANSPORT` and `PAPER_CHASER_HTTP_PATH`, but not `PAPER_CHASER_HTTP_HOST` or `PAPER_CHASER_HTTP_PORT`. For direct shell runs you can still set those two variables explicitly; Compose keeps the container binding fixed and instead exposes `PAPER_CHASER_PUBLISHED_HOST` / `PAPER_CHASER_PUBLISHED_PORT` for the local host-side mapping.

> [!IMPORTANT]
> HTTP transport compatibility is available, but this repository does **not**
> yet ship a hardened public deployment profile. Before exposing the server
> beyond localhost, add Origin validation/allowlisting, authentication, and
> TLS in front of the ASGI app. This follows the MCP HTTP transport guidance
> confirmed via Context7: servers must validate `Origin` and should implement
> authentication for HTTP transports.

Use `paper_chaser_mcp.server.build_http_app(...)` if you need to inject
deployment-specific Starlette middleware around the FastMCP ASGI app.

For private Azure hosting, this repository also ships
`paper_chaser_mcp.deployment:app`, an ASGI wrapper that adds `/healthz`,
optional shared-token authentication, and optional Origin allowlisting in front
of the MCP app. The Azure deployment scaffold in [docs/azure-deployment.md](docs/azure-deployment.md)
uses that wrapper behind private Azure API Management and private endpoints so
trusted clients never see the backend token or upstream provider API keys.
The tracked Azure workflow is manual-only and supports a `bootstrap` mode for
first-time environment bring-up before the `full` private-runner deployment.

Example local/integration HTTP run:

```bash
PAPER_CHASER_TRANSPORT=streamable-http \
PAPER_CHASER_HTTP_HOST=0.0.0.0 \
PAPER_CHASER_HTTP_PORT=8000 \
python -m paper_chaser_mcp
```

That path serves the FastMCP app directly over streamable HTTP at
`http://127.0.0.1:8000/mcp` by default. It is the simplest local option when
you want HTTP transport without the deployment wrapper.

If you want local parity with the HTTP wrapper used by Compose and Azure,
run:

```bash
PORT=8000 PAPER_CHASER_HTTP_HOST=127.0.0.1 paper-chaser-mcp deployment-http
```

`paper-chaser-mcp deployment-http` launches the same
`paper_chaser_mcp.deployment:app` wrapper used by hosted HTTP deployments.
It prefers `PORT` when it is set, and otherwise falls back to
`PAPER_CHASER_HTTP_PORT`. That wrapper keeps the same streamable HTTP MCP
transport, adds `/healthz`, and optionally enforces
`PAPER_CHASER_HTTP_AUTH_TOKEN` and `PAPER_CHASER_ALLOWED_ORIGINS` for
`/mcp`.

In all of these local HTTP modes, clients talk only to your local MCP endpoint.
They do **not** need upstream provider API keys. Those keys, when present, stay
server-side and only let your local server use higher provider limits or
optional paid providers such as SerpApi.

</details>

### Docker MCP package (stdio)

For local MCP clients that launch servers as subprocesses, use the image in
stdio mode. For unpublished local iteration, build and run `paper-chaser-mcp:local`.
For the reusable public package, use the published GHCR tag:

```bash
docker run --rm -i ghcr.io/joshuasundance-swca/paper-chaser-mcp:latest
```

For a locally built image:

```bash
docker run --rm -i paper-chaser-mcp:local
```

A Docker-backed MCP client entry typically looks like:

```json
{
  "mcpServers": {
    "paper-chaser": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "ghcr.io/joshuasundance-swca/paper-chaser-mcp:latest"]
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

Compose also intentionally overrides the app default transport to `streamable-http` inside the container. The underlying application default remains `stdio`; the compose file opts into HTTP wrapper mode so browser tools and bridge-style clients can connect over `http://127.0.0.1:8000/mcp` without extra shell flags.

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

If you set `PAPER_CHASER_HTTP_AUTH_TOKEN` and leave
`PAPER_CHASER_HTTP_AUTH_HEADER=authorization`, the deployment wrapper expects
`Authorization: Bearer <token>` on `/mcp`. The checked-in Azure scaffold
overrides the header name to `x-backend-auth` and has API Management inject
that header for backend-only traffic. The published host defaults to
`127.0.0.1`; only change `PAPER_CHASER_PUBLISHED_HOST` when you intentionally
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

- URL: `http://paper-chaser-mcp:8080/mcp`
- Transport: `streamable-http`

`compose.inspector.yaml` accepts `IMAGE` overrides, so you can test a specific
tag without editing files:

```bash
IMAGE=ghcr.io/joshuasundance-swca/paper-chaser-mcp:latest docker compose -f compose.inspector.yaml up
```

## Tools

Full tool reference. See the [Quick tool decision guide](#quick-tool-decision-guide) above for where to start.

### Smart research layer

| Tool | Description |
| --- | --- |
| `search_papers_smart` | Concept-level discovery with query expansion, multi-provider fusion, reranking, and reusable `searchSessionId`. Supports `latencyProfile` (`fast`/`balanced`/`deep`) and optional `providerBudget`. |
| `ask_result_set` | Grounded QA, claim checks, and comparisons over a saved `searchSessionId`. |
| `map_research_landscape` | Cluster a saved result set into themes, gaps, disagreements, and next-search suggestions. |
| `expand_research_graph` | Expand paper anchors or a saved session into a citation/reference/author graph with frontier ranking. |

### Paper search

| Tool | Description |
| --- | --- |
| `search_papers` | Brokered single-page search (Semantic Scholar → arXiv → CORE → SerpApi). Read `brokerMetadata.nextStepHint`. |
| `search_papers_bulk` | Paginated bulk search (Semantic Scholar) up to 1,000 papers/call with boolean query syntax. |
| `search_papers_semantic_scholar` | Single-page Semantic Scholar-only search with full filter support. |
| `search_papers_arxiv` | Single-page arXiv-only search. |
| `search_papers_core` | Single-page CORE-only search. |
| `search_papers_serpapi` | Single-page SerpApi Google Scholar search. **Requires SerpApi.** |
| `search_papers_openalex` | Single-page OpenAlex-only search. |
| `search_papers_openalex_bulk` | Cursor-paginated OpenAlex search. |
| `search_papers_openalex_by_entity` | OpenAlex works constrained to one source, institution, or topic entity ID. |

### Known-item lookup and citation repair

| Tool | Description |
| --- | --- |
| `resolve_citation` | Citation-repair workflow for incomplete or malformed references. Abstains on regulatory references. |
| `search_papers_match` | Known-item lookup for messy or partial titles with cross-provider confirmation. |
| `get_paper_details` | Lookup by DOI, arXiv ID, Semantic Scholar ID, or URL. Optional `includeEnrichment`. |
| `get_paper_details_openalex` | OpenAlex work lookup by W-id, URL, or DOI with abstract reconstruction. |
| `paper_autocomplete` | Paper title typeahead completions. |
| `paper_autocomplete_openalex` | OpenAlex work typeahead for known-item disambiguation. |

### Citations, references, and authors

| Tool | Description |
| --- | --- |
| `get_paper_citations` | Papers that cite the given paper (Semantic Scholar). Cursor-paginated. |
| `get_paper_citations_openalex` | OpenAlex cited-by expansion. Cursor-paginated. |
| `get_paper_references` | References behind the given paper (Semantic Scholar). Cursor-paginated. |
| `get_paper_references_openalex` | OpenAlex backward-reference expansion. Cursor-paginated. |
| `get_paper_authors` | Authors of the given paper (Semantic Scholar). |
| `search_authors` | Search authors by name (Semantic Scholar). |
| `search_authors_openalex` | Search OpenAlex authors by name. |
| `get_author_info` | Author profile by Semantic Scholar author ID. |
| `get_author_info_openalex` | OpenAlex author profile by A-id or URL. |
| `get_author_papers` | Papers by Semantic Scholar author. Cursor-paginated. |
| `get_author_papers_openalex` | Papers by OpenAlex author with `year` filter and cursor pagination. |
| `batch_get_papers` | Details for up to 500 paper IDs in one call. |
| `batch_get_authors` | Details for up to 1,000 author IDs in one call. |
| `get_paper_recommendations` | Similar papers by single seed (GET). |
| `get_paper_recommendations_post` | Similar papers from positive/negative seed sets (POST). |

### Paper enrichment and OA discovery

| Tool | Description |
| --- | --- |
| `enrich_paper` | Combined Crossref + Unpaywall enrichment for one paper or DOI. |
| `get_paper_metadata_crossref` | Explicit Crossref enrichment for a known paper or DOI. |
| `get_paper_open_access_unpaywall` | Unpaywall OA status, PDF URL, and license lookup by DOI. Requires `UNPAYWALL_EMAIL`. |

### OpenAlex entities

| Tool | Description |
| --- | --- |
| `search_entities_openalex` | Search OpenAlex source, institution, or topic entities for pivot workflows. |

### ECOS species dossiers

| Tool | Description |
| --- | --- |
| `search_species_ecos` | ECOS species discovery by common or scientific name. |
| `get_species_profile_ecos` | Full ECOS species dossier: listings, documents, and conservation plans. |
| `list_species_documents_ecos` | Flatten one dossier into a sorted document inventory. |
| `get_document_text_ecos` | Fetch and convert an ECOS document (PDF/HTML/text) to Markdown. |

### Federal Register and CFR

| Tool | Description |
| --- | --- |
| `search_federal_register` | Keyless Federal Register discovery for notices, rules, and proposed rules. |
| `get_federal_register_document` | Retrieve one Federal Register document by number, FR citation, or GovInfo link. |
| `get_cfr_text` | CFR part or section text from GovInfo. Requires `GOVINFO_API_KEY`. |

### SerpApi extras *(opt-in, paid)*

| Tool | Description |
| --- | --- |
| `search_papers_serpapi_cited_by` | Google Scholar cited-by expansion via SerpApi. |
| `search_papers_serpapi_versions` | Google Scholar all-versions expansion via SerpApi cluster IDs. |
| `get_author_profile_serpapi` | Google Scholar author profile via SerpApi. |
| `get_author_articles_serpapi` | Paginated Google Scholar author articles via SerpApi. |
| `get_paper_citation_formats` | Citation export (MLA, APA, BibTeX, etc.) from Google Scholar. **Requires SerpApi.** |
| `get_serpapi_account_status` | SerpApi quota and throughput snapshot. |

### Recovery and diagnostics

| Tool | Description |
| --- | --- |
| `search_snippets` | Quote or phrase recovery when title/keyword search is weak. Last-resort tool. |
| `get_provider_diagnostics` | Live provider health, throttling state, retries, and fallback reasons. |


### ECOS walkthrough

California least tern is a representative end-to-end ECOS flow:

1. Call `search_species_ecos` with `query="California least tern"` to get the ECOS species id `8104`.
2. Call `get_species_profile_ecos` with `species_id="8104"` to inspect the species dossier, grouped recovery documents, biological opinions, and conservation-plan links.
3. Call `list_species_documents_ecos` with `species_id="8104"` and, for example, `documentKinds=["recovery_plan","five_year_review","biological_opinion"]` to flatten the document inventory.
4. Call `get_document_text_ecos` on the 2025 five-year-review PDF or the revised recovery plan PDF to turn the source document into Markdown for downstream analysis.

## Resources and prompts

- Resource: `guide://paper-chaser/agent-workflows` - compact onboarding guide for choosing tools and following pagination safely
- Resource: `paper://{paper_id}` - compact markdown + structured payload for a resolved paper
- Resource: `author://{author_id}` - compact markdown + structured payload for a resolved author
- Resource: `search://{searchSessionId}` - saved result set surfaced from tool outputs
- Resource: `trail://paper/{paper_id}?direction=citations|references` - compact citation/reference trail resource
- Prompt: `plan_paper_chaser_search` - reusable planning prompt for raw-vs-smart literature-search workflows
- Prompt: `plan_smart_paper_chaser_search` - smart-tool-first planning prompt for concept discovery
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
mcp-inspector python -m paper_chaser_mcp
```

## Development

Install the package with development extras:

```bash
pip install -e ".[all]"
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
python -m pytest --cov=paper_chaser_mcp --cov-report=term-missing --cov-fail-under=85
python -m mypy --config-file pyproject.toml
python -m ruff check .
python -m bandit -c pyproject.toml -r paper_chaser_mcp
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
python scripts/validate_deployment.py --require-az --require-docker --image-tag paper-chaser-mcp:ci-validate
```

### GitHub Agentic Workflow smoke test

The repository now includes an agentic regression workflow source at
`.github/workflows/test-paper-chaser.md` and its compiled lock file at
`.github/workflows/test-paper-chaser.lock.yml`. After editing the Markdown
workflow, recompile it and then run the normal validation stack so pre-commit
can normalize the generated lock file:

```bash
gh aw compile test-paper-chaser --dir .github/workflows
```

What this workflow does:

- Runs the agent against the local `paper-chaser` MCP server inside GitHub
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

- The editable source of truth is `.github/workflows/test-paper-chaser.md`.
- `gh aw compile ...` generates `.github/workflows/test-paper-chaser.lock.yml`,
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

1. Edit `.github/workflows/test-paper-chaser.md`.
2. Recompile it with `gh aw compile test-paper-chaser --dir .github/workflows`.
3. Run the normal validation stack so pre-commit can normalize the generated
   lock file.
4. Commit both the `.md` source and `.lock.yml` output together.
5. Push the branch, then run `Test Paper Chaser MCP` from the GitHub Actions
   UI. For on-demand UX reviews, use `workflow_dispatch` inputs to choose the
   run mode and provide an optional focus prompt such as a new feature, a
   provider-specific flow, or a confusing agent interaction to probe.

The repository also includes `.github/workflows/agentic-assign.yml`, a
lightweight workflow that automatically assigns GitHub Copilot to any issue
labeled both `agentic` and `needs-copilot`, unless the issue also carries
`needs-human`, `blocked`, or `no-agent`. It listens to direct `issues` events
and to completed `Test Paper Chaser MCP` runs so verifier-created issues still
get assigned even when the original issue event does not fan out into a second
workflow run. For the actual Copilot assignment API call it prefers
`GH_AW_GITHUB_TOKEN`, then falls back to `COPILOT_GITHUB_TOKEN`, then to the
default Actions token. This avoids 403 failures on repositories where the
default `GITHUB_TOKEN` cannot perform Copilot issue assignment even though the
workflow has `issues: write` permissions.

The normal `Validate` workflow now also recompiles `test-paper-chaser.md` on
CI and fails if `.github/workflows/test-paper-chaser.lock.yml` is stale, so
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

For maintainer orientation after the module split, start with `docs/agent-handoff.md`. The public MCP surface stays in `paper_chaser_mcp/server.py`, while implementation now lives in `paper_chaser_mcp/dispatch.py`, `paper_chaser_mcp/search.py`, `paper_chaser_mcp/tools.py`, `paper_chaser_mcp/runtime.py`, `paper_chaser_mcp/models/`, and provider subpackages under `paper_chaser_mcp/clients/`.

## Guides

- [GitHub Copilot Instructions](.github/copilot-instructions.md) - repo-specific guidance for GitHub Copilot and the GitHub cloud coding agent, including workflow defaults and durable planning expectations.
- [Agent Handoff](docs/agent-handoff.md) - current repo status, validation commands, and next recommended work for follow-on agents.
- [Paper Chaser Golden Paths](docs/golden-paths.md) - primary personas, workflow defaults, success signals, and future workflow-oriented follow-up work.
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

### Protocol and runtime

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastMCP](https://gofastmcp.com/)
- [SerpApi pricing](https://serpapi.com/pricing)

### Scholarly providers

- [Semantic Scholar API](https://api.semanticscholar.org/api-docs)
- [arXiv API User's Manual](https://info.arxiv.org/help/api/user-manual.html)
- [CORE API v3 Documentation](https://api.core.ac.uk/docs/v3)
- [OpenAlex API docs](https://docs.openalex.org/)
- [SerpApi Google Scholar API](https://serpapi.com/google-scholar-api)
- [Crossref REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)
- [Unpaywall API](https://unpaywall.org/products/api)

### Regulatory and species sources

- [ECOS](https://ecos.fws.gov/)
- [FederalRegister.gov API](https://www.federalregister.gov/developers/documentation/api/v1)
- [GovInfo API](https://api.govinfo.gov/docs/)
