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
- [Guided vs expert profiles](#guided-vs-expert-profiles)
- [Quick start](#quick-start)
- [Quick tool decision guide](#quick-tool-decision-guide)
- [Core workflows](#core-workflows)
- [Agent response contract](#agent-response-contract)
- [Deferred export design](#deferred-export-design)
- [Migration note](#migration-note)
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

Paper Chaser MCP is now guided-first: the default public surface is designed to
be hard to misuse and explicit about trust.

- **Default research entrypoint**: `research` handles discovery, known-item
  recovery, citation repair, and regulatory routing in one trust-graded path,
  with a server-owned quality-first policy for guided use.
- **Grounded follow-up**: `follow_up_research` answers against one saved
  `searchSessionId`; if you omit it, the server only infers a session when the
  choice is unique. Saved-session follow-up can classify mixed source sets into
  on-topic evidence, weaker context, and off-target leads when the stored
  metadata is already sufficient.
- **Decision metadata**: guided responses surface `executionProvenance`, and
  ambiguous follow-up or source-inspection flows return structured
  `sessionResolution` / `sourceResolution` payloads instead of opaque errors.
- **Reference-first recovery**: `resolve_reference` handles DOI/arXiv/URL,
  citation fragments, and regulatory-style references, and exact DOI/arXiv/
  paper-URL inputs resolve as exact anchors rather than falling through to fuzzy repair.
  Ambiguous title-only or conflicting metadata matches can now return
  `multiple_candidates` or `needs_disambiguation`; treat those as candidate
  anchors, not citation-ready resolutions.
- **Compact top-level answer**: guided `research` leads with a short
  recommendation-first `summary`, while keeping the structured evidence,
  leads, and provenance fields available below it.
- **Source auditability**: `inspect_source` exposes one `sourceId` with
  provenance, trust state, weak-match rationale, and quality-aware direct-read next steps; omitted `searchSessionId`
  is only accepted when one compatible saved session exists.
- **Runtime truth**: `get_runtime_status` surfaces active profile/transport and
  provider-state warnings without requiring low-level diagnostics. `configuredSmartProvider`
  is the configured smart bundle; `activeSmartProvider` is the latest effective execution path, cold-start snapshots emit an explicit provisional warning instead of claiming deterministic fallback before the first smart call settles, and the top-level provider sets now split `disabledProviderSet`, `suppressedProviderSet`, `degradedProviderSet`, and `quotaLimitedProviderSet` instead of collapsing them.
- **Expert depth remains available**: raw/smart/provider-specific tools still
  exist for operator workflows under the expert profile.

## Guided vs expert profiles

Use `PAPER_CHASER_TOOL_PROFILE` to choose the advertised surface:

| Profile | Default | Exposed surface | Intended user |
| --- | --- | --- | --- |
| `guided` | yes | `research`, `follow_up_research`, `resolve_reference`, `inspect_source`, `get_runtime_status` | Low-context users and agents |
| `expert` | no | Guided tools plus raw/provider-specific families (`search_papers*`, smart graph tools, regulatory direct tools, full diagnostics), subject to enabled features and disabled-tool visibility | Power users and operator workflows |

Practical default: `PAPER_CHASER_TOOL_PROFILE=guided` with
`PAPER_CHASER_HIDE_DISABLED_TOOLS=true`.

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
        "PAPER_CHASER_TOOL_PROFILE": "guided",
        "PAPER_CHASER_HIDE_DISABLED_TOOLS": "true",
        "PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR": "true",
        "PAPER_CHASER_ENABLE_ARXIV": "true",
        "PAPER_CHASER_ENABLE_CORE": "false"
      }
    }
  }
}
```

Then start with one of these prompts in your MCP client:

- `Research retrieval-augmented generation for coding agents and return only trustworthy findings.`
- `Use my last searchSessionId to answer one grounded follow-up question.`
- `Resolve this citation fragment: Vaswani et al. 2017 Attention Is All You Need.`
- `Research the regulatory history of California condor under 50 CFR 17.95.`

If you want a local env template for shell runs or Docker Compose, copy `.env.example` to `.env` and fill in only the providers you use.

---

## Quick tool decision guide

| Goal | Start here |
|---|---|
| Discovery, literature review, or regulatory history | `research` |
| Grounded follow-up over saved results | `follow_up_research` |
| Citation/DOI/arXiv/URL/reference cleanup | `resolve_reference` |
| Audit one returned source before relying on it | `inspect_source` |
| Explain environment/runtime differences | `get_runtime_status` |
| Need direct provider control or specialized pagination | switch to expert profile and use raw/provider-specific tools |

## Core workflows

### 1. Guided research first

```text
research(query="retrieval-augmented generation for coding agents", limit=5)
→ inspect resultStatus, answerability, summary, evidence, leads, routingSummary
→ if resultStatus=needs_disambiguation with clarification.reason=underspecified_reference_fragment:
  tighten the anchor or pivot to resolve_reference instead of forcing retrieval
→ if resultStatus=abstained and sources are suppressed: inspect suppressedSourceSummaries before rerunning or escalating
→ save searchSessionId for follow-up or source inspection
```

### 2. Ask one grounded follow-up

```text
follow_up_research(searchSessionId="...", question="What evaluation tradeoffs show up here?")
→ inspect answerStatus
→ if answered: use answer + evidence (compact default: sources are identified by selectedEvidenceIds)
→ if abstained/insufficient_evidence: use nextActions, suppressedSourceSummaries, and inspect_source
→ mixed saved sessions can still answer relevance-triage questions such as which items are on-topic vs off-target
→ uniquely anchored recommendation asks can also return a safe start-here answer plus topRecommendation
→ if you omit searchSessionId and multiple saved sessions exist: provide it explicitly
→ for full source records pass responseMode="standard"; for diagnostics responseMode="debug"
→ for selection asks ("where should I start?", "most recent?"), read topRecommendation
```

### 3. Resolve references before broad search when possible

```text
resolve_reference(reference="10.1038/nrn3241")
→ exact DOI/arXiv/paper URL should resolve directly when supported
resolve_reference(reference="Rockstrom et al planetary boundaries 2009 Nature 461 472")
→ inspect status and bestMatch/alternatives
→ only treat bestMatch as citation-ready when status=resolved
→ if status=multiple_candidates or needs_disambiguation: pick a candidate or add a stronger author/year/venue clue before citing it
→ if resolved: run research with the resolved anchor
```

### 4. Inspect one source before citing it

```text
inspect_source(searchSessionId="...", evidenceId="...")
→ inspect verificationStatus, topicalRelevance, whyClassifiedAsWeakMatch, confidenceSignals, canonicalUrl, directReadRecommendations
→ if searchSessionId is omitted and inference is ambiguous, rerun with an explicit saved session id
```

### 5. Handle abstention and clarification explicitly

- If `research.resultStatus` is `abstained` or `needs_disambiguation`, do not invent
  synthesis. Narrow with a concrete anchor: DOI, exact title, species name,
  agency, year, or venue.
- When `research` returns `needs_disambiguation` with
  `clarification.reason=underspecified_reference_fragment`, the server is
  intentionally stopping before speculative retrieval on a vague
  citation/reference fragment. Tighten the clue set or switch to
  `resolve_reference`.
- If `follow_up_research.answerStatus` is `abstained` or
  `insufficient_evidence`, treat it as a safety signal. Use `inspect_source`
  and rerun `research` with tighter scope.

### 6. Expert fallback when you need fine control

```text
PAPER_CHASER_TOOL_PROFILE=expert
→ search_papers_smart / ask_result_set / map_research_landscape / expand_research_graph
→ search_papers / search_papers_bulk and provider-specific families
→ search_federal_register / get_federal_register_document / get_cfr_text for direct regulatory primary-source control
```

For expert smart tools, `deep` is the default quality-first mode. Use
`balanced` only when lower latency matters enough to justify a narrower pass,
and reserve `fast` for smoke tests or debugging.

Guided `research` no longer accepts a public `latencyProfile` knob. The server
owns that policy and currently applies a deep-backed quality-first path with
one bounded review escalation when the first pass is too weak.

## Agent response contract

Treat these as the main guided contracts:

| Field or pattern | Where it appears | What to do with it |
| --- | --- | --- |
| `resultStatus` | `research` | `succeeded`, `partial`, `needs_disambiguation`, `abstained`, `failed` |
| `answerability` | `research`, `follow_up_research` | `grounded`, `limited`, `insufficient` |
| `evidence` | `research`, `follow_up_research` | Canonical grounded source records for inspection and citation |
| `leads` | `research`, `follow_up_research`, expert smart tools | Review weak, filtered, or off-topic leads without promoting them into grounded evidence |
| `evidenceGaps` | `research`, `follow_up_research` | Treat as explicit limits on the current answer, not hidden caveats |
| `routingSummary` | `research`, `follow_up_research` | Check intent, anchor, provider plan, regulatory subtype or entity card when present, and why the result is partial |
| `coverageSummary` | `research`, `follow_up_research` | Check provider coverage and completeness before relying on synthesis |
| `executionProvenance` | guided tools | Inspect which server policy, latency defaults, and fallback path produced the result |
| `confidenceSignals` | `research`, `follow_up_research`, `inspect_source` | Inspect additive trust cues such as evidence quality, synthesis mode, and source-scope labels without replacing `answerability` |
| `evidenceUsePlan` | `follow_up_research` | For synthesis-style follow-ups, inspect answer subtype, directly responsive evidence ids, unsupported parts, and retrieval sufficiency before trusting the answer |
| `sessionResolution` | `follow_up_research`, `inspect_source` | Use when a session was inferred, repaired, missing, or ambiguous |
| `sourceResolution` | `inspect_source` | Use when the requested source id was matched, unresolved, or needs a retry with available ids |
| `abstentionDetails` | guided tools on weak evidence | Treat as the actionable reason and recovery hint for abstention or insufficient evidence |
| `nextActions` | guided tools | Treat as server-preferred recovery path on weak evidence |
| `clarification` | `research` | Ask the user only when a bounded clarification request is provided |
| `answerStatus` | `follow_up_research` | `answered`, `abstained`, `insufficient_evidence`. Grounded `answered` requires on-topic verified source + qa-readable text + non-deterministic provider + medium+ confidence; otherwise expect `insufficient_evidence`. |
| `topRecommendation` | `follow_up_research` (comparative/selection asks) | Structured pick with `sourceId`, `recommendationReason`, `comparativeAxis` (e.g. `beginner_friendly`, `recency`, `authority`). Unique anchored "where should I start?" asks can safely answer through this path even when broader synthesis would stay limited. |
| `responseMode` | `follow_up_research` input | `compact` (default, hides full sources and legacy fields), `standard`, `debug`. |
| `includeLegacyFields` | `follow_up_research` input | Set `true` to restore legacy `verifiedFindings`/`unverifiedLeads` in compact mode. |
| `fullTextUrlFound` / `bodyTextEmbedded` / `qaReadableText` | `inspect_source` | Distinguish URL discovery, embedded body text, and text actually available to QA synthesis. `fullTextObserved` may still appear as a compatibility alias, but the split fields are the durable contract. |
| `evidenceId` | `evidence[*]` | Pass to `inspect_source` for per-source provenance checks |
| `runtimeSummary` | `get_runtime_status` and expert diagnostics | Confirm effective profile, smart provider state, and warnings |

For broad agency-guidance discovery, guided routing stays on the
regulatory primary-source path. Off-topic authority documents may still appear
as `leads`, but they should not displace more relevant query-anchored guidance
or policy documents from the top-level recommendation.

For source-level audits, treat `whyClassifiedAsWeakMatch` and
`confidenceSignals.sourceScopeLabel` / `confidenceSignals.sourceScopeReason` as
the primary explanation of why an authoritative record was retained as a weak
match or off-topic lead.

Additional trust and grounding signals landed in the `llm-guidance` phase-4
wave. Guided responses can expose `confidenceSignals.evidenceQualityProfile`,
`confidenceSignals.synthesisMode`, `confidenceSignals.evidenceProfileDetail`,
`confidenceSignals.synthesisPath`, `confidenceSignals.trustRevisionNarrative`,
and a `trustSummary.authoritativeButWeak` bucket for primary-source records
that are authoritative but not topically responsive. `searchStrategy` may
surface `regulatoryIntent`, `intentFamily`, a `subjectCard` for species and
regulatory grounding, and `subjectChainGaps` describing missing subject-chain
evidence. `inspect_source` pairs each direct-read suggestion with a
`directReadRecommendationDetails` entry shaped as
`{trustLevel, whyRecommended, cautions}` so agents can prioritize direct reads
by quality. See [Paper Chaser Golden Paths](docs/golden-paths.md) and
[Guided And Smart Robustness Notes](docs/guided-smart-robustness.md) for how
to read and act on these signals.

## Deferred export design

Session export is intentionally deferred in this wave. The planned future shape is
`export_search_session(searchSessionId, format)` with `format` in `ris`,
`bibtex`, or `csv`, using the guided-v2 source/citation schema so export can land
without another public-contract rewrite.

## Migration note

If you previously used the smart/raw-first surface directly:

1. Start with `research` instead of `search_papers_smart` or `search_papers`.
2. Use `follow_up_research` instead of `ask_result_set` for default grounded QA.
3. Use `resolve_reference` instead of `resolve_citation`/`search_papers_match`
   as your first known-item recovery step.
4. Keep expert tools for explicit operator workflows by setting
   `PAPER_CHASER_TOOL_PROFILE=expert`.
5. Do not send `latencyProfile` to guided `research`; the server now owns that
  policy internally.
6. For expert smart tools, `deep` is now the default. Choose `balanced`
  explicitly when you want the lower-latency fallback.
7. Expect guided wrappers to surface `executionProvenance`, `sessionResolution`,
  `sourceResolution`, and `abstentionDetails`.

For the detailed breaking-change note, see [Guided Reset Migration Note](docs/guided-reset-migration-note.md).

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
- Azure AI Foundry eval publishing helpers: `pip install -e ".[eval-foundry]"`
- Hugging Face eval publishing helpers: `pip install -e ".[eval-huggingface]"`
- Both eval publishing helper surfaces: `pip install -e ".[eval]"`
- Add `,ai-faiss` to any of the commands above if you want the optional FAISS backend.

Azure OpenAI uses the same `openai` extra.
Hugging Face uses a dedicated `huggingface` extra that installs the OpenAI-compatible SDK plus the LangChain OpenAI adapter; this repo documents it as a chat-only smart-provider path with embeddings disabled.
The eval publishing helpers use separate extras on purpose: `eval-foundry` is for Azure AI Foundry dataset upload support, and `eval-huggingface` is for Hugging Face dataset-repo or bucket publishing support. Those extras are independent from the smart-provider chat runtime.

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

Guided mode starts from `research`. The brokered and provider-specific controls
below are expert-path controls.

| Area | Default | Main variables | Notes |
| --- | --- | --- | --- |
| Tool profile | `guided` | `PAPER_CHASER_TOOL_PROFILE` | `guided` exposes the 5 low-context tools; `expert` exposes the broader raw/provider-specific surface, subject to enabled features and `PAPER_CHASER_HIDE_DISABLED_TOOLS` |
| Guided policy | quality-first | `PAPER_CHASER_GUIDED_RESEARCH_LATENCY_PROFILE`, `PAPER_CHASER_GUIDED_FOLLOW_UP_LATENCY_PROFILE`, `PAPER_CHASER_GUIDED_ALLOW_PAID_PROVIDERS`, `PAPER_CHASER_GUIDED_ESCALATION_ENABLED`, `PAPER_CHASER_GUIDED_ESCALATION_MAX_PASSES`, `PAPER_CHASER_GUIDED_ESCALATION_ALLOW_PAID_PROVIDERS` | Guided `research` / `follow_up_research` use these server-owned defaults instead of honoring client `latencyProfile` knobs |
| Search broker | `semantic_scholar,arxiv,core,serpapi_google_scholar` | `PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR`, `PAPER_CHASER_ENABLE_ARXIV`, `PAPER_CHASER_ENABLE_CORE`, `PAPER_CHASER_ENABLE_SERPAPI`, `PAPER_CHASER_PROVIDER_ORDER` | SerpApi is opt-in and paid; CORE is off by default |
| OpenAlex tool family | enabled | `PAPER_CHASER_ENABLE_OPENALEX`, `OPENALEX_API_KEY`, `OPENALEX_MAILTO` | Explicit tool family, not a default broker hop |
| ScholarAPI tool family | disabled | `PAPER_CHASER_ENABLE_SCHOLARAPI`, `SCHOLARAPI_API_KEY` | Explicit discovery, monitoring, full-text, and PDF family; also available as an opt-in broker target via `preferredProvider` or `providerOrder`. ScholarAPI-sourced paper results now include a separate `contentAccess` block for access/full-text metadata. |
| Enrichment | enabled | `PAPER_CHASER_ENABLE_CROSSREF`, `CROSSREF_MAILTO`, `CROSSREF_TIMEOUT_SECONDS`, `PAPER_CHASER_ENABLE_UNPAYWALL`, `UNPAYWALL_EMAIL`, `UNPAYWALL_TIMEOUT_SECONDS`, `PAPER_CHASER_ENABLE_OPENALEX` | Used after you already have a paper or DOI |
| ECOS | enabled | `PAPER_CHASER_ENABLE_ECOS`, `ECOS_BASE_URL`, `ECOS_TIMEOUT_SECONDS`, document timeout and size vars, TLS vars | Species and document workflows |
| Federal Register / GovInfo | enabled | `PAPER_CHASER_ENABLE_FEDERAL_REGISTER`, `PAPER_CHASER_ENABLE_GOVINFO_CFR`, `GOVINFO_API_KEY`, GovInfo timeout and size vars | Federal Register search is keyless; authoritative CFR retrieval uses GovInfo |
| Smart layer | disabled | `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_HTTP_REFERER`, `OPENROUTER_TITLE`, `HUGGINGFACE_API_KEY`, `HUGGINGFACE_BASE_URL`, `NVIDIA_API_KEY`, `NVIDIA_NIM_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `PAPER_CHASER_ENABLE_AGENTIC`, model and index vars | Additive only; supports `openai`, `azure-openai`, `anthropic`, `nvidia`, `google`, `mistral`, `huggingface`, `openrouter`, and `deterministic`. OpenAI ships with checked-in model defaults, Anthropic, NVIDIA, Google, and Mistral auto-swap to provider defaults when those OpenAI defaults are left untouched, and Azure OpenAI can override both roles with deployment names. Hugging Face and OpenRouter are documented as OpenAI-compatible chat routers configured with `HUGGINGFACE_BASE_URL` and `OPENROUTER_BASE_URL`; both remain chat-only in this repo and do not enable embeddings. OpenRouter preserves explicit planner and synthesis model names such as provider-prefixed model IDs. `NVIDIA_NIM_BASE_URL` is optional for self-hosted NIMs; leave it empty for hosted NVIDIA API Catalog access. Embeddings remain disabled by default because they have been unreliable in this codebase, and improving them is out of scope for the current release. When ScholarAPI is enabled, smart discovery can also route through it and cap it via `providerBudget.maxScholarApiCalls`. |
| Hide disabled tools | guided default `true`, expert default `false` | `PAPER_CHASER_HIDE_DISABLED_TOOLS` | Guided mode keeps this on to reduce dead-end tool picks; expert mode usually leaves it off for operator visibility |

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
| `openrouter` | none | none | Runtime preserves explicit planner/synthesis model values and sends requests to `OPENROUTER_BASE_URL`; the first-pass path remains chat-only |
| `deterministic` | n/a | n/a | No external LLM calls; model selection metadata is reported as deterministic instead |

`PAPER_CHASER_EMBEDDING_MODEL` defaults to `text-embedding-3-large`, but embeddings stay off until you set `PAPER_CHASER_DISABLE_EMBEDDINGS=false`. They remain off by default because embeddings have been unreliable in this codebase and improving them is outside the scope of the current guided-policy release. In the current provider surface, embeddings are only used by providers that explicitly support them, which means the documented Hugging Face path remains chat-only even though it uses an OpenAI-compatible router.

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

### Guided default tools

| Tool | Description |
| --- | --- |
| `research` | Default trust-graded entrypoint for discovery, known-item recovery, citation repair, and regulatory routing. |
| `follow_up_research` | Grounded follow-up over a saved `searchSessionId`; returns explicit abstention/insufficient-evidence states when needed. |
| `resolve_reference` | Resolve citation-like input (citation, DOI, arXiv, URL, title fragment, regulatory reference) into the safest next anchor. |
| `inspect_source` | Inspect one `sourceId` from a guided result set for provenance, trust state, and direct-read follow-through. |
| `get_runtime_status` | Guided runtime summary for active profile, transport, smart-provider state, and warnings. |

### Expert smart research layer

These tools are expert profile paths for deeper orchestration and provider
control.

| Tool | Description |
| --- | --- |
| `search_papers_smart` | Concept-level discovery with query expansion, multi-provider fusion, reranking, reusable `searchSessionId`, and an evidence-first expert contract (`resultStatus`, `answerability`, `routingSummary`, `evidence`, `leads`, `evidenceGaps`, `structuredSources`, `coverageSummary`, `failureSummary`). Legacy trust fields remain available as compatibility views. In `auto` mode it can also route clearly regulatory asks into a primary-source timeline. `latencyProfile` defaults to `deep` for highest-quality expert work; use `balanced` for lower latency and reserve `fast` for smoke tests. Optional `providerBudget` remains available for advanced clients. |
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
- Prompt: `plan_paper_chaser_search` - reusable planning prompt with guided-first defaults and explicit expert fallback
- Prompt: `plan_smart_paper_chaser_search` - planning prompt for intentional expert smart-mode workflows
- Prompt: `triage_literature` - guided triage workflow for trust-aware theme mapping and next-step selection
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

- `mcp-tools.core.json` - guided low-context default surface (`research`, `follow_up_research`, `resolve_reference`, `inspect_source`, `get_runtime_status`)
- `mcp-tools.full.json` - guided + expert package for environments intentionally running with `PAPER_CHASER_TOOL_PROFILE=expert`
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
python -m pytest --cov=paper_chaser_mcp --cov-report=term-missing --cov-fail-under=87
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

**Workflow inputs:** `mode` (`smoke`, `comprehensive`, or `feature_probe`),
`tool_profile` (`guided` by default, `expert` when you intentionally want
raw/provider-specific coverage), and an optional `focus_prompt`. Select them
via `workflow_dispatch` inputs.

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
- [LLM Selection Guide](docs/llm-selection-guide.md) - planner versus synthesis responsibilities, current smart-layer model defaults, the eval-bootstrap funnel around `generate_eval_topics.py` and `run_eval_autopilot.py`, and criteria for choosing LLMs in this repo.
- [LLM Evaluation Program Plan](docs/llm-evaluation-program-plan.md) - role-based evaluation strategy, dataset-generation plan, evaluator stack, and phased rollout for rigorous LLM performance measurement in this repo.
- [LLM Evaluation Dataset Schema](docs/llm-evaluation-dataset-schema.md) - JSONL schema, field rules, governance conventions, and storage layout for role-based evaluation seed sets and future benchmark expansion.
- [LLM Evaluation Platform Strategy](docs/llm-evaluation-platform-strategy.md) - how to combine repo-local evals with Azure AI Foundry, Hugging Face, and live-trace active-learning loops without losing portability.
- [LLM Evaluation Trace Promotion](docs/llm-evaluation-trace-promotion.md) - workflow and helper format for promoting reviewed live traces into durable evaluation rows.

Optional live eval-candidate capture can be enabled with `PAPER_CHASER_ENABLE_EVAL_TRACE_CAPTURE=true` and `PAPER_CHASER_EVAL_TRACE_PATH=...`, then converted into a review queue with `scripts/build_eval_review_queue.py` before promotion.

Portable exports for downstream evaluation and training systems are available via `scripts/export_eval_assets.py`, including Foundry-friendly eval JSONL, Hugging Face dataset JSONL, and chat-style training JSONL from review-approved traces.

Service-specific publish helpers are available via `scripts/upload_foundry_eval_dataset.py` and `scripts/upload_hf_eval_assets.py` for pushing reviewed exports into a Foundry project dataset, a Hugging Face dataset repo, or a Hugging Face bucket.

Expert batch curation runs can now emit `batch-summary.json` and `batch-ledger.csv` alongside the raw report, captured events, and review queue so offline drift and throughput checks do not depend on replaying the full JSONL artifacts.

For repo-local eval bootstrap, the current top-level workflow is:

- `scripts/generate_eval_topics.py` for planner-led topic generation, taxonomy assignment, ranking, pruning, balancing, and scenario emission
- `scripts/run_eval_autopilot.py` for profile-driven generation, immutable run bundles, holdout checks, and guarded workflow handoff
- `scripts/run_eval_workflow.py` for expert batch capture, review or promotion, dataset splitting, and live provider-matrix evaluation

The checked-in autopilot sample profiles now include balanced-science defaults plus narrow-run profiles such as `single-seed-exploratory-review`, `single-seed-exploratory-safe`, and `single-seed-diagnostic-force`. Those narrow-run profiles can enable single-seed diversification so one-seed runs ask the planner for additional review, regulatory, and methods-oriented variants instead of depending only on looser workflow thresholds.

See `docs/llm-evaluation-integrations.md` for the current Foundry and Hugging Face integration posture, including when `hf-mount` is a good fit for a shared capture sink.
- [Release And Publishing Plan](docs/release-publishing-plan.md) - the current release playbook for GHCR, GitHub Release assets, manual MCP Registry publication, and dormant PyPI.
- [Guided Reset Migration Note](docs/guided-reset-migration-note.md) - breaking default-surface change, guided-vs-expert split, and client migration checklist.
- [Paper Chaser Golden Paths](docs/golden-paths.md) - primary personas, workflow defaults, success signals, and future workflow-oriented follow-up work.
- [Azure Deployment](docs/azure-deployment.md) - deployment modes, required secrets and variables, and validation paths for the private Azure rollout.
- [Azure Architecture](docs/azure-architecture.md) - trust boundaries, runtime topology, and credential separation for the Azure scaffold.
- [Azure Security Model](docs/azure-security-model.md) - credential classes, Key Vault usage, and backend-auth separation in the Azure rollout.
- [Provider Upgrade Program](docs/provider-upgrade-program.md) - provider roles, latency profiles, diagnostics, benchmark corpus, and acceptance gates for the reliability-first provider upgrade.
- [OpenRouter Provider Guide](docs/openrouter-api-guide.md) - implementation-focused guidance for adding and operating OpenRouter as a chat-only smart-layer provider, including the current Trinity bring-up plan.
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
