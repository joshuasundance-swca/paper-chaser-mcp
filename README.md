# Paper Chaser MCP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/joshuasundance-swca/paper-chaser-mcp)

> **Release status:** The repository, CLI, Docker image metadata, and public
> MCP identity are now aligned on `paper-chaser-mcp`. GHCR images and GitHub
> Release assets are the primary public distribution channels; PyPI remains
> intentionally gated until account recovery and trusted-publisher setup are
> complete.

An MCP server for academic research — search papers, chase citations, look up authors, repair broken references, explore species dossiers, and retrieve regulatory text, all from one FastMCP server that AI assistants can call directly.

**Providers:** [Semantic Scholar](https://www.semanticscholar.org/product/api) · [arXiv](https://info.arxiv.org/help/api/user-manual.html) · [OpenAlex](https://developers.openalex.org/) · [CORE](https://api.core.ac.uk/docs/v3) · [SerpApi Google Scholar](https://serpapi.com/google-scholar-api) (opt-in, paid) · [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) · [Unpaywall](https://unpaywall.org/products/api) · [ECOS](https://ecos.fws.gov/) · [FederalRegister.gov](https://www.federalregister.gov/developers/documentation/api/v1) · [GovInfo](https://api.govinfo.gov/docs/)

---

## Contents

- [What can it do?](#what-can-it-do)
- [Quick start](#quick-start)
- [Quick tool decision guide](#quick-tool-decision-guide)
- [Core workflows](#core-workflows)
- [Agent response contract](#agent-response-contract)
- [Installation](#installation)
- [Configuration](#configuration)
- [Tools](#tools)
- [Resources and prompts](#resources-and-prompts)
- [Microsoft packaging assets](#microsoft-packaging-assets)
- [Testing with MCP Inspector](#testing-with-mcp-inspector)
- [Development](#development)
- [Guides](#guides)
- [License](#license)
- [Links](#links)

---

## What can it do?

Paper Chaser MCP combines one stable research-oriented MCP surface with an additive smart layer and a few specialized non-paper workflows:

- **Discovery**: `search_papers` for a fast brokered first page, `search_papers_bulk` for exhaustive retrieval, and provider-specific search tools when you need explicit source control.
- **Concept-level research**: `search_papers_smart` plus `ask_result_set`, `map_research_landscape`, and `expand_research_graph` for grounded follow-up over a reusable `searchSessionId`, with trust-graded `verifiedFindings`, `likelyUnverified`, `evidenceGaps`, `structuredSources`, and coverage/failure summaries.
- **Known-item recovery**: `resolve_citation`, `search_papers_match`, `get_paper_details`, and autocomplete tools for messy titles, incomplete references, DOIs, arXiv IDs, and URLs.
- **Citation and author pivots**: citation/reference traversal, recommendations, batch lookups, and author workflows across Semantic Scholar and explicit OpenAlex tool families.
- **Enrichment and access**: Crossref, Unpaywall, and OpenAlex enrichment for metadata, open-access status, citation context, and PDF discovery after you already have a paper or DOI-bearing identifier.
- **Regulatory and species workflows**: ECOS dossiers plus Federal Register and CFR retrieval for primary-source research outside the normal paper graph. `search_papers_smart` can now route clearly regulatory asks into a primary-source timeline instead of forcing a paper-only workflow.
- **Agent-friendly outputs**: structured responses, `brokerMetadata.nextStepHint`, `agentHints`, `clarification`, `resourceUris`, `searchSessionId`, and provider diagnostics.

## Quick start

If you want the fastest local path, install from source and add the server to your MCP client in stdio mode:

```bash
pip install -e .
```

```json
{
  "mcpServers": {
    "paper-chaser": {
      "command": "python",
      "args": ["-m", "paper_chaser_mcp"],
      "env": {
        "PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR": "true",
        "PAPER_CHASER_ENABLE_ARXIV": "true",
        "PAPER_CHASER_ENABLE_CORE": "false"
      }
    }
  }
}
```

Then start with one of these prompts in your MCP client:

- `Find recent papers on language-model alignment.`
- `Resolve this citation: Vaswani et al. 2017 Attention Is All You Need.`
- `Map the research themes in retrieval-augmented generation for coding agents.`

If you want a local env template for shell runs or Docker Compose, copy `.env.example` to `.env` and fill in only the providers you use.

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
| ScholarAPI monitoring / indexed scans | `list_papers_scholarapi` |
| Papers that cite X / refs behind X | `get_paper_citations` / `get_paper_references` |
| Author's work | `search_authors` → `get_author_info` → `get_author_papers` |
| OA status or full-text PDF for a known paper | `get_paper_open_access_unpaywall` or `enrich_paper` |
| OpenAlex DOI/ID, cursor pagination, entity pivots | `*_openalex` tools |
| Species dossier / recovery documents | `search_species_ecos` → `get_species_profile_ecos` → `list_species_documents_ecos` |
| Federal Register notice or rule | `search_federal_register` → `get_federal_register_document` |
| CFR regulatory text | `get_cfr_text` |
| Quote or phrase recovery (last resort) | `search_snippets` |
| Provider health | `get_provider_diagnostics` |

After `search_papers`: read `brokerMetadata.nextStepHint`.
After `search_papers_smart`: reuse `searchSessionId` with `ask_result_set`, `map_research_landscape`, or `expand_research_graph`.
After trust-graded responses: read `verifiedFindings`, `likelyUnverified`, `evidenceGaps`, `structuredSources`, and `coverageSummary` before treating a result set as complete.
For Semantic Scholar expansion tools: prefer `paper.recommendedExpansionId`; if `paper.expansionIdStatus` is `not_portable`, resolve through DOI first.
For `enrich_paper`: treat it as additive metadata lookup, not known-item resolution. Query-only calls without a paper or DOI anchor now abstain instead of guessing a canonical DOI.

## Core workflows

### Quick discovery

Use `search_papers` when you want a strong first page quickly. It is a brokered single-page search, not a pagination entry point.

```text
search_papers(query="large language model alignment", limit=10)
→ inspect brokerMetadata.providerUsed and brokerMetadata.nextStepHint
→ if you need exhaustive retrieval, switch to search_papers_bulk
→ if you find a promising anchor paper, pivot to citations, references, or authors
```

### Concept discovery and grounded follow-up

Use `search_papers_smart` for literature reviews, concept discovery, and question-driven exploration.

```text
search_papers_smart(query="retrieval-augmented generation for coding agents", limit=10)
→ save searchSessionId
→ ask_result_set(searchSessionId="...", question="What evaluation tradeoffs show up here?")
→ map_research_landscape(searchSessionId="...")
→ expand_research_graph(seedSearchSessionId="...", direction="citations")
```

For clearly regulatory or environmental-history asks, `search_papers_smart` can also be the first step:

```text
search_papers_smart(query="regulatory history of California condor under 50 CFR 17.95", limit=5)
→ inspect verifiedFindings, structuredSources, regulatoryTimeline, coverageSummary, failureSummary
→ pivot into get_species_profile_ecos, list_species_documents_ecos, get_federal_register_document, or get_cfr_text
```

### Known-item lookup and citation chasing

Use `resolve_citation` for broken references, `search_papers_match` for messy titles, and `get_paper_details` for DOI/arXiv/URL lookup.

```text
resolve_citation(citation="Rockstrom et al planetary boundaries 2009 Nature 461 472")
→ confirm the canonical paper or alternatives
→ get_paper_citations(paper_id=paper.recommendedExpansionId)
→ cursor-loop with pagination.nextCursor if you need more results
```

### Regulatory and species follow-through

Use the non-paper tools when the source is clearly regulatory or species-oriented rather than scholarly.

```text
search_species_ecos(query="California least tern")
→ get_species_profile_ecos(species_id="8104")
→ list_species_documents_ecos(species_id="8104")
→ get_document_text_ecos(...)

search_federal_register(query="California least tern")
→ get_federal_register_document(...)
→ get_cfr_text(...) when you need the affected CFR text
```

## Agent response contract

These fields are the high-value response contracts to pay attention to:

| Field or pattern | Where it appears | What to do with it |
| --- | --- | --- |
| `brokerMetadata.nextStepHint` | `search_papers` | Treat it as the server's recommended next move after the first page |
| `searchSessionId` | smart workflows and saved result sets | Reuse it for grounded follow-up instead of rerunning discovery |
| `verifiedFindings` / `likelyUnverified` / `evidenceGaps` | smart workflows | Treat these as the trust boundary: verified evidence first, weaker leads second, explicit gaps third |
| `structuredSources` | smart workflows | Use this when you need provenance, access state, primary-source status, or audit-ready citations |
| `coverageSummary` / `failureSummary` | brokered and smart workflows | Use these to understand partial success, provider coverage, and whether retries or alternate tools are justified |
| `runtimeSummary` | `get_provider_diagnostics` | Use this to confirm the effective transport, enabled providers, warnings, embeddings state, and active broker order |
| `agentHints` | primary read tools | Use for retry guidance, follow-on tool suggestions, and warning handling |
| `clarification` | ambiguous cases | Ask the user only when the server surfaces a bounded clarification request |
| `resourceUris` | primary read tools | Open follow-on resources directly when your MCP client supports them |
| `pagination.nextCursor` | paginated tools | Pass it back exactly as returned; it is opaque and tool-specific |
| `paper.recommendedExpansionId` | brokered and known-item flows | Prefer this for Semantic Scholar expansion tools |
| `paper.expansionIdStatus=not_portable` | non-Semantic-Scholar results | Resolve through DOI or a Semantic Scholar-native lookup before citation/reference expansion |

---

## Installation

Current distribution options:

- Source checkout: the most direct local path today, especially for development and MCP desktop clients.
- GHCR image: the primary container distribution channel for Docker-backed MCP clients.
- GitHub Release assets: `v*` tags build wheel and sdist artifacts and attach them to a draft GitHub Release for review.
- PyPI: intentionally gated for now; use source installs or GitHub Release artifacts until that path is re-enabled.

For local source installs:

```bash
pip install -e .
```

Optional extras for the additive AI layer:

- Shared smart-layer runtime only, including deterministic mode: `pip install -e ".[ai]"`
- OpenAI or Azure OpenAI provider support: `pip install -e ".[ai,openai]"`
- Hugging Face chat-router support: `pip install -e ".[ai,huggingface]"`
- NVIDIA provider support: `pip install -e ".[ai,nvidia]"`
- Anthropic provider support: `pip install -e ".[ai,anthropic]"`
- Google provider support: `pip install -e ".[ai,google]"`
- Mistral provider support: `pip install -e ".[ai,mistral]"`
- Add `,ai-faiss` to any of the commands above if you want the optional FAISS backend.

Azure OpenAI uses the same `openai` extra.
Hugging Face uses a dedicated `huggingface` extra that installs the OpenAI-compatible SDK plus the LangChain OpenAI adapter; this repo documents it as a chat-only smart-provider path with embeddings disabled.

## Configuration

The full local environment-variable contract lives in `.env.example`. That file mirrors the public local knobs supported by `docker-compose.yaml`. Azure-specific identifiers, secrets, and Bicep parameters are intentionally documented separately in [docs/azure-deployment.md](docs/azure-deployment.md).

### Desktop MCP clients

Use stdio transport for desktop MCP clients unless you specifically need HTTP.
See the [Quick start](#quick-start) JSON example above for the server definition.

- **Claude Desktop** config path:
  - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
  - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- **Cursor**: add the same MCP server definition in Cursor settings.

### Search broker and feature flags

`search_papers` is a brokered single-page search. It returns the first successful provider in the active chain and does **not** paginate. Use `search_papers_bulk` for exhaustive retrieval.

| Area | Default | Main variables | Notes |
| --- | --- | --- | --- |
| Search broker | `semantic_scholar,arxiv,core,serpapi_google_scholar` | `PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR`, `PAPER_CHASER_ENABLE_ARXIV`, `PAPER_CHASER_ENABLE_CORE`, `PAPER_CHASER_ENABLE_SERPAPI`, `PAPER_CHASER_PROVIDER_ORDER` | SerpApi is opt-in and paid; CORE is off by default |
| OpenAlex tool family | enabled | `PAPER_CHASER_ENABLE_OPENALEX`, `OPENALEX_API_KEY`, `OPENALEX_MAILTO` | Explicit tool family, not a default broker hop |
| ScholarAPI tool family | disabled | `PAPER_CHASER_ENABLE_SCHOLARAPI`, `SCHOLARAPI_API_KEY` | Explicit discovery, monitoring, full-text, and PDF family; also available as an opt-in broker target via `preferredProvider` or `providerOrder`. ScholarAPI-sourced paper results now include a separate `contentAccess` block for access/full-text metadata. |
| Enrichment | enabled | `PAPER_CHASER_ENABLE_CROSSREF`, `CROSSREF_MAILTO`, `CROSSREF_TIMEOUT_SECONDS`, `PAPER_CHASER_ENABLE_UNPAYWALL`, `UNPAYWALL_EMAIL`, `UNPAYWALL_TIMEOUT_SECONDS`, `PAPER_CHASER_ENABLE_OPENALEX` | Used after you already have a paper or DOI |
| ECOS | enabled | `PAPER_CHASER_ENABLE_ECOS`, `ECOS_BASE_URL`, `ECOS_TIMEOUT_SECONDS`, document timeout and size vars, TLS vars | Species and document workflows |
| Federal Register / GovInfo | enabled | `PAPER_CHASER_ENABLE_FEDERAL_REGISTER`, `PAPER_CHASER_ENABLE_GOVINFO_CFR`, `GOVINFO_API_KEY`, GovInfo timeout and size vars | Federal Register search is keyless; authoritative CFR retrieval uses GovInfo |
| Smart layer | disabled | `OPENAI_API_KEY`, `HUGGINGFACE_API_KEY`, `HUGGINGFACE_BASE_URL`, `NVIDIA_API_KEY`, `NVIDIA_NIM_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `PAPER_CHASER_ENABLE_AGENTIC`, model and index vars | Additive only; supports `openai`, `azure-openai`, `anthropic`, `nvidia`, `google`, `mistral`, `huggingface`, and `deterministic`. OpenAI ships with checked-in model defaults, Anthropic, NVIDIA, Google, and Mistral auto-swap to provider defaults when those OpenAI defaults are left untouched, and Azure OpenAI can override both roles with deployment names. Hugging Face is documented as an OpenAI-compatible chat router configured with `HUGGINGFACE_BASE_URL`; it remains chat-only in this repo and does not enable embeddings. `NVIDIA_NIM_BASE_URL` is optional for self-hosted NIMs; leave it empty for hosted NVIDIA API Catalog access. Embeddings remain disabled by default. When ScholarAPI is enabled, smart discovery can also route through it and cap it via `providerBudget.maxScholarApiCalls`. |
| Hide disabled tools | disabled | `PAPER_CHASER_HIDE_DISABLED_TOOLS` | Opt-in. When true, `list_tools` hides disabled explicit provider families, hides generic Semantic Scholar-backed tools when Semantic Scholar is disabled, and hides brokered search or citation-repair entry points when no usable backend remains. Leave false for full-contract compatibility. |

### Smart-layer model defaults

These are the effective planner/synthesis defaults when you enable `PAPER_CHASER_ENABLE_AGENTIC=true` and do not intentionally override the model vars.

| `PAPER_CHASER_AGENTIC_PROVIDER` | Default planner | Default synthesis | Resolution rule |
| --- | --- | --- | --- |
| `openai` | `gpt-5.4-mini` | `gpt-5.4` | Uses the checked-in `PAPER_CHASER_PLANNER_MODEL` and `PAPER_CHASER_SYNTHESIS_MODEL` defaults directly |
| `azure-openai` | `gpt-5.4-mini` | `gpt-5.4` | Uses the same model vars unless `AZURE_OPENAI_PLANNER_DEPLOYMENT` or `AZURE_OPENAI_SYNTHESIS_DEPLOYMENT` is set; when present, those deployment names win |
| `anthropic` | `claude-haiku-4-5` | `claude-sonnet-4-6` | Runtime swaps to these provider defaults only when planner/synthesis are still set to the checked-in OpenAI defaults |
| `nvidia` | `nvidia/nemotron-3-nano-30b-a3b` | `nvidia/nemotron-3-super-120b-a12b` | Runtime swaps to these provider defaults only when planner/synthesis are still set to the checked-in OpenAI defaults |
| `google` | `gemini-2.5-flash` | `gemini-2.5-pro` | Runtime swaps to these provider defaults only when planner/synthesis are still set to the checked-in OpenAI defaults |
| `mistral` | `mistral-medium-latest` | `mistral-large-latest` | Runtime swaps to these provider defaults only when planner/synthesis are still set to the checked-in OpenAI defaults |
| `huggingface` | `moonshotai/Kimi-K2.5` | `moonshotai/Kimi-K2.5` | Runtime swaps to these provider defaults only when planner/synthesis are still set to the checked-in OpenAI defaults; requests are sent to `HUGGINGFACE_BASE_URL` and the path remains chat-only |
| `deterministic` | n/a | n/a | No external LLM calls; model selection metadata is reported as deterministic instead |

`PAPER_CHASER_EMBEDDING_MODEL` defaults to `text-embedding-3-large`, but embeddings stay off until you set `PAPER_CHASER_DISABLE_EMBEDDINGS=false`. In the current provider surface, embeddings are only used by providers that explicitly support them, which means the documented Hugging Face path remains chat-only even though it uses an OpenAI-compatible router.

Recommended baseline: enable Semantic Scholar, OpenAlex, Crossref, and Unpaywall for general scholarly workflows; enable ScholarAPI when you want explicit full-text or PDF retrieval; keep SerpApi opt-in because it is a paid recall-recovery path.

Broker rules that matter most:

- Default search fallback order is Semantic Scholar, then arXiv, then CORE, then SerpApi when enabled.
- `preferredProvider`, `providerOrder`, and `PAPER_CHASER_PROVIDER_ORDER` accept `core`, `semantic_scholar`, `arxiv`, `scholarapi`, and `serpapi` or `serpapi_google_scholar`.
- Semantic Scholar-only filters such as `publicationDateOrYear`, `fieldsOfStudy`, `publicationTypes`, `openAccessPdf`, and `minCitationCount` can force the broker to skip incompatible providers.
- Broker responses surface `brokerMetadata.providerUsed`, `brokerMetadata.attemptedProviders`, and `brokerMetadata.recommendedPaginationTool` so agents can follow the right next step.

### Transport and deployment modes

| Mode | Default | Main variables | Use when |
| --- | --- | --- | --- |
| Desktop stdio | `stdio` | none required | Claude Desktop, Cursor, local MCP subprocess launches |
| Direct HTTP run | opt in | `PAPER_CHASER_TRANSPORT`, `PAPER_CHASER_HTTP_HOST`, `PAPER_CHASER_HTTP_PORT`, `PAPER_CHASER_HTTP_PATH` | Local integration testing without the deployment wrapper |
| HTTP wrapper | opt in | `PAPER_CHASER_HTTP_AUTH_TOKEN`, `PAPER_CHASER_HTTP_AUTH_HEADER`, `PAPER_CHASER_ALLOWED_ORIGINS` | Local parity with hosted HTTP deployments |
| Docker Compose publish settings | localhost defaults | `PAPER_CHASER_PUBLISHED_HOST`, `PAPER_CHASER_PUBLISHED_PORT` | Control the host-side HTTP port mapping only |

Key distinctions:

- `PAPER_CHASER_HTTP_HOST` and `PAPER_CHASER_HTTP_PORT` control the direct shell and hosted deployments. Docker Compose keeps the container bind at `0.0.0.0:8080` and uses `PAPER_CHASER_PUBLISHED_HOST` / `PAPER_CHASER_PUBLISHED_PORT` for the host-side mapping.
- `paper-chaser-mcp deployment-http` runs the deployment wrapper used by Compose and Azure. It adds `/healthz` plus optional auth and Origin enforcement in front of the MCP endpoint.

Example direct local HTTP run:

```bash
PAPER_CHASER_TRANSPORT=streamable-http \
PAPER_CHASER_HTTP_HOST=127.0.0.1 \
PAPER_CHASER_HTTP_PORT=8000 \
python -m paper_chaser_mcp
```

If you need the full Azure deployment story, including the `bootstrap` and `full` workflow modes, read [docs/azure-deployment.md](docs/azure-deployment.md), [docs/azure-architecture.md](docs/azure-architecture.md), and [docs/azure-security-model.md](docs/azure-security-model.md).

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
workflow is tag-driven for GHCR: a `v*` tag publishes the reusable container
image to `ghcr.io/joshuasundance-swca/paper-chaser-mcp`. MCP Registry
publication is intentionally decoupled into a separate manual workflow so GHCR
shipping does not depend on registry availability.

Python package publishing is prepared separately in
`.github/workflows/publish-pypi.yml`: pull requests build and `twine check` the
distribution, and the actual publish jobs stay dormant until the repository
variable `ENABLE_PYPI_PUBLISHING` is set to `true`. After PyPI/TestPyPI access
is restored and the trusted publishers are registered, manual dispatch can
publish to TestPyPI and a `v*` tag can publish to PyPI.

GitHub Release assets are handled separately in
`.github/workflows/publish-github-release.yml`: a `v*` tag or manual dispatch
builds wheel and sdist artifacts, verifies them with `twine check`, generates
`SHA256SUMS`, and uploads them to a draft GitHub Release page so Python
artifacts can be reviewed before broader public promotion.

### Docker Compose (HTTP wrapper mode)

For local HTTP testing, MCP Inspector, or bridge-style integrations, this repo
ships `docker-compose.yaml` with localhost-only defaults. Compose
explicitly starts the `deployment-http` subcommand, so HTTP wrapper behavior
does not depend on the image's default transport.

Compose keeps the container bind host and internal port fixed at
`0.0.0.0:8080` and overrides the app default transport to `streamable-http`,
so browser tools and bridge-style clients can connect over
`http://127.0.0.1:8000/mcp` without extra shell flags. The compose file
exposes the user-facing knobs: transport, MCP path, provider keys, provider
toggles, auth, and the published host port mapping.

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
| `search_papers_smart` | Concept-level discovery with query expansion, multi-provider fusion, reranking, reusable `searchSessionId`, and trust-graded sections (`verifiedFindings`, `likelyUnverified`, `evidenceGaps`, `structuredSources`, `coverageSummary`, `failureSummary`). In `auto` mode it can also route clearly regulatory asks into a primary-source timeline. Supports `mode`, `latencyProfile` (`fast`/`balanced`/`deep`), and optional `providerBudget`. |
| `ask_result_set` | Grounded QA, claim checks, and comparisons over a saved `searchSessionId`. |
| `map_research_landscape` | Cluster a saved result set into themes, gaps, disagreements, and next-search suggestions. |
| `expand_research_graph` | Expand paper anchors or a saved session into a citation/reference/author graph with frontier ranking. |

### Paper search

| Tool | Description |
| --- | --- |
| `search_papers` | Brokered single-page search (default: Semantic Scholar → arXiv → CORE → SerpApi). Read `brokerMetadata.nextStepHint`; ScholarAPI is also available as an explicit opt-in broker target. |
| `search_papers_bulk` | Paginated bulk search (Semantic Scholar) up to 1,000 papers/call with boolean query syntax. |
| `search_papers_semantic_scholar` | Single-page Semantic Scholar-only search with full filter support. |
| `search_papers_arxiv` | Single-page arXiv-only search. |
| `search_papers_core` | Single-page CORE-only search. |
| `search_papers_serpapi` | Single-page SerpApi Google Scholar search. **Requires SerpApi.** |
| `search_papers_scholarapi` | Single-page ScholarAPI relevance-ranked search. **Requires ScholarAPI.** |
| `search_papers_openalex` | Single-page OpenAlex-only search. |
| `search_papers_openalex_bulk` | Cursor-paginated OpenAlex search. |
| `list_papers_scholarapi` | Cursor-paginated ScholarAPI monitoring/list flow sorted by `indexed_at`. |
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
| `enrich_paper` | Combined Crossref + Unpaywall + OpenAlex enrichment for one known paper or DOI. Query-only calls without an anchor abstain instead of resolving a paper. |
| `get_paper_metadata_crossref` | Explicit Crossref enrichment for a known paper or DOI. |
| `get_paper_open_access_unpaywall` | Unpaywall OA status, PDF URL, and license lookup by DOI. Requires `UNPAYWALL_EMAIL`. |

### ScholarAPI text and PDF retrieval

| Tool | Description |
| --- | --- |
| `get_paper_text_scholarapi` | Fetch one ScholarAPI plain-text full document by ScholarAPI paper id. |
| `get_paper_texts_scholarapi` | Batch full-text retrieval for up to 100 ScholarAPI paper ids. Preserves order and null placeholders. |
| `get_paper_pdf_scholarapi` | Fetch one ScholarAPI PDF as structured metadata plus base64-encoded content. |

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

Primary read-tool responses also surface:

- `agentHints` - recommended next tools, retry guidance, and warnings
- `clarification` - bounded clarification fallback when the server cannot safely disambiguate on its own
- `resourceUris` - follow-on resources that compatible clients can open directly
- `searchSessionId` - reusable result-set handle for smart follow-up workflows and cached expansion/search trails

## Microsoft packaging assets

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
pip install -e ".[dev]"
```

If you also want the additive AI layer plus every hosted-provider integration in the same environment:

```bash
pip install -e ".[all]"
```

`all` expands to `ai,openai,huggingface,nvidia,anthropic,google,mistral,dev`, so Azure OpenAI still uses the same `openai` extra while Hugging Face remains a separate chat-only OpenAI-compatible install surface.

If you need the optional FAISS backend locally as well:

```bash
pip install -e ".[all,ai-faiss]"
```

Project dependencies are declared in `pyproject.toml`; there is no separate runtime `requirements.txt` to keep in sync.

Run the local test suite:

```bash
pytest
```

Install and run the configured pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

`pre-commit install` installs both the fast `pre-commit` hooks and the
heavier `pre-push` gates configured in `.pre-commit-config.yaml`. Manual-stage
hooks are not invoked automatically; run `pre-commit run --hook-stage manual
--all-files` (or the direct commands above) when you want the full local gate.

The development extras include `pytest`, `pytest-asyncio`, `pytest-cov`,
`ruff`, `mypy`, `bandit`, `build`, `bumpver`, `pip-audit`,
`shellcheck-py`, `types-defusedxml`, and `pre-commit`. GitHub dependency automation is configured for both Python
packages and GitHub Actions via Dependabot, with pull requests checked by the
dependency review workflow.

For local parity with CI on GitHub workflow files, keep `shellcheck` available
on `PATH` before running `pre-commit`. Installing `shellcheck-py` in the active
repo venv satisfies this for many setups; verify with `shellcheck --version`
instead of assuming inline workflow bash is being linted locally.

### Version bumps

Version metadata is managed with `bumpver` from `pyproject.toml`. The checked-in
package version stays in plain PEP 440 form such as `0.2.0`, while the release
tag shape remains `v0.2.0` to match the existing publish workflow trigger.

For PR-branch-safe review, dry-run a patch bump without touching git state:

```bash
bumpver update --patch --dry --no-fetch --no-commit --no-tag-commit --no-push
```

For an actual release-prep branch, update the checked-in version contract but
still leave commit, tag, and push under explicit maintainer control:

```bash
bumpver update --patch --no-commit --no-tag-commit --no-push
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

The repository includes an agentic regression workflow at
`.github/workflows/test-paper-chaser.md` (source) and
`.github/workflows/test-paper-chaser.lock.yml` (compiled lock file). It runs
the agent against the local MCP server inside GitHub Actions, exercises the
primary golden paths, evaluates agent UX quality, and can file actionable
issues for follow-on work.

After editing the Markdown workflow, recompile and validate:

```bash
gh aw compile test-paper-chaser --dir .github/workflows
pre-commit run --all-files
```

Commit both the `.md` source and `.lock.yml` output together, then run
`Test Paper Chaser MCP` from the GitHub Actions UI.

**Workflow modes:** `smoke` (default), `comprehensive` (broader UX review), or
`feature_probe` (with an optional focus prompt). Select via `workflow_dispatch`
inputs.

**Required secrets:** `COPILOT_GITHUB_TOKEN` is required.
`GH_AW_MODEL_AGENT_COPILOT` (Actions variable, optional) controls the agent
model. `CORE_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY` are optional.

The repository also includes `.github/workflows/agentic-assign.yml`, which
automatically assigns GitHub Copilot to issues labeled `agentic` and
`needs-copilot` (unless also labeled `needs-human`, `blocked`, or `no-agent`).
The `Validate` workflow recompiles `test-paper-chaser.md` on CI and fails if
the lock file is stale, so pull requests cannot silently drift out of sync.
The workflow is "deployed" when GitHub Actions sees the committed `.lock.yml`
on the branch where it should run.

See [SECURITY.md](SECURITY.md) for the public-repo security posture and the
recommended private reporting path for vulnerabilities.

For maintainer orientation after the module split, start with `docs/agent-handoff.md`. The public MCP surface stays in `paper_chaser_mcp/server.py`, while implementation lives in `paper_chaser_mcp/dispatch.py`, `paper_chaser_mcp/search.py`, `paper_chaser_mcp/tools.py`, `paper_chaser_mcp/runtime.py`, `paper_chaser_mcp/models/`, and provider subpackages under `paper_chaser_mcp/clients/`.

## Guides

- [GitHub Copilot Instructions](.github/copilot-instructions.md) - repo-specific guidance for GitHub Copilot and the GitHub cloud coding agent, including workflow defaults and durable planning expectations.
- [Agent Handoff](docs/agent-handoff.md) - current repo status, validation commands, and next recommended work for follow-on agents.
- [Release And Publishing Plan](docs/release-publishing-plan.md) - the current release playbook for GHCR, GitHub Release assets, manual MCP Registry publication, and dormant PyPI.
- [Paper Chaser Golden Paths](docs/golden-paths.md) - primary personas, workflow defaults, success signals, and future workflow-oriented follow-up work.
- [Azure Deployment](docs/azure-deployment.md) - deployment modes, required secrets and variables, and validation paths for the private Azure rollout.
- [Azure Architecture](docs/azure-architecture.md) - trust boundaries, runtime topology, and credential separation for the Azure scaffold.
- [Azure Security Model](docs/azure-security-model.md) - credential classes, Key Vault usage, and backend-auth separation in the Azure rollout.
- [Provider Upgrade Program](docs/provider-upgrade-program.md) - provider roles, latency profiles, diagnostics, benchmark corpus, and acceptance gates for the reliability-first provider upgrade.
- [ScholarAPI Integration Guide](docs/scholarapi-api-guide.md) - planning guide for adding ScholarAPI as an explicit discovery, monitoring, full-text, and PDF provider without weakening the current graph-oriented provider contracts.
- [OpenAlex API Guide](docs/openalex-api-guide.md) - implementation-focused guidance for the repo's explicit OpenAlex MCP surface, including authentication, credit-based limits, paging, `/works` semantics, and normalization caveats.
- [Semantic Scholar API Guide](docs/semantic-scholar-api-guide.md) - practical guidance for respectful and effective Semantic Scholar API usage with async rate limiting, retries, and `.env`-based local development.
- [SerpApi Google Scholar Guide](docs/serpapi-google-scholar-api-guide.md) - deep research notes on SerpApi capabilities, tradeoffs, and cost/compliance considerations; the repo ships the explicit cited-by, versions, author, account, and citation-format flows documented there.
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
- [ScholarAPI docs](https://scholarapi.net/docs/api)
- [SerpApi Google Scholar API](https://serpapi.com/google-scholar-api)
- [Crossref REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)
- [Unpaywall API](https://unpaywall.org/products/api)

### Regulatory and species sources

- [ECOS](https://ecos.fws.gov/)
- [FederalRegister.gov API](https://www.federalregister.gov/developers/documentation/api/v1)
- [GovInfo API](https://api.govinfo.gov/docs/)
