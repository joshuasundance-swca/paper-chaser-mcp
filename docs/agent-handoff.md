# Agent Handoff

This document is the current working handoff for the fork. It is intended to give any follow-on agent enough context to validate the repo, understand the recent hardening work, and continue from the highest-value next steps without re-discovering project state.

## Current Status

- Local development baseline is configured through `pyproject.toml` and `.pre-commit-config.yaml`.
- Dev extras now include `shellcheck-py`, and local workflow-file parity expects
  `shellcheck` on `PATH` so `actionlint` can lint inline bash before CI.
- The default free broker path is `Semantic Scholar -> arXiv -> CORE` for
  `search_papers`; SerpApi remains an optional paid recovery hop at the end of
  the broker order when enabled, but it is disabled by default.
- XML parsing uses `defusedxml`.
- README configuration examples are valid JSON.
- GitHub Actions now validates pushes and pull requests.
- The main validation workflow now treats Python 3.14 as the shipped default
  runtime lane: it runs the lockfile freshness check, coverage-gated tests,
  build, dependency audit, deployment validation, and the default Docker smoke
  path. Python 3.13 remains in the matrix as a compatibility lane.
- The repo now includes a private-network Azure deployment scaffold: Docker
  packaging, a deployment wrapper ASGI app, Bicep infrastructure under
  `infra/`, a manual OIDC deployment workflow with `bootstrap` and `full`
  modes, and operator-facing Azure docs.
- The Docker image is now a reusable public MCP package surface as well: the
  default image contract is stdio-first with a stable `paper-chaser-mcp`
  entrypoint, `server.json` tracks the public MCP/OCI metadata, and
  `.github/workflows/publish-public-mcp-package.yml` now handles tag-driven
  GHCR publication only, while `.github/workflows/publish-mcp-registry.yml`
  separately handles manual MCP Registry publication after the GHCR image
  exists. The GHCR workflow keeps the semver checks plus SBOM/provenance
  output, and the registry workflow keeps the OIDC publish step independent.
- Version bumps are now managed through `bumpver` in `pyproject.toml`. The repo
  keeps checked-in package versions in PEP 440 form while preserving `v*` git
  tags for the publish workflow, and the default bumpver config keeps commit,
  tag, and push disabled so PR branches can stage release-prep diffs safely.
- A dedicated `.github/workflows/publish-pypi.yml` workflow now covers the
  Python package path separately from GHCR/MCP Registry publication: PRs build
  and validate distributions, while the actual publish jobs are gated behind
  `vars.ENABLE_PYPI_PUBLISHING == 'true'`. That keeps the workflow safe to merge
  before PyPI/TestPyPI account recovery; once access is restored and the
  trusted publishers are registered, manual dispatch can target TestPyPI and
  `v*` tags can publish to PyPI.
- A dedicated `.github/workflows/publish-github-release.yml` workflow now covers
  GitHub Release page artifacts independently of GHCR and PyPI: PRs validate
  that wheel/sdist builds still work, while `v*` tags and manual dispatch build
  the Python artifacts, generate `SHA256SUMS`, and attach them to a draft
  GitHub Release for review.
- `docs/release-publishing-plan.md` is now the durable playbook for release
  prep, tagging, GHCR publication, GitHub Release asset review, manual MCP
  Registry publication, and the no-tag-overwrite policy.
- The release-asset and PyPI workflows now build and validate distributions on
  Python 3.14 by default, while still keeping PR-only 3.14 proof jobs in place
  as an explicit packaging cross-check.
- Deployment asset validation is now a first-class workflow: pre-commit and the
  main CI workflow run `scripts/validate_deployment.py`, and the validator can
  lint/build the Bicep, validate the APIM policy XML, build the Docker image,
  and smoke-test the secured `/healthz` and `/mcp/` paths locally, including
  the APIM-style Origin allowlist and the Azure-scaffold
  `PAPER_CHASER_HTTP_AUTH_HEADER=x-backend-auth` contract.
- `paper_chaser_mcp/deployment_runner.py` powers the explicit
  `paper-chaser-mcp deployment-http` command used by Compose and the Azure
  deployment path. It wraps `paper_chaser_mcp.deployment:app` with Uvicorn
  and prefers `PORT` over `PAPER_CHASER_HTTP_PORT`.
- `paper_chaser_mcp/deployment_utils.py` resolves the post-deploy `/healthz`
  smoke-test target from `SMOKE_TEST_HEALTH_URL`, `containerAppHealthUrl`, or
  `containerAppFqdn`.
- `paper_chaser_mcp/server.py` is now a compatibility facade over smaller modules.
- The repo now exposes a compatibility-first smart layer on top of the stable
  raw MCP surface: `search_papers_smart`, `ask_result_set`,
  `map_research_landscape`, and `expand_research_graph`.
- Primary read tools now surface additive agent UX metadata:
  `agentHints`, `clarification`, `resourceUris`, and reusable
  `searchSessionId` handles where appropriate.
- Published MCP tool schemas are now sanitized for flatter, more
  Microsoft-friendly compatibility while internal Pydantic validation remains
  strict.
- The server now exposes additive read-only resources for papers, authors,
  saved searches, and citation/reference trails, plus helper prompts for
  smart-search planning and query refinement.
- Agent-facing workflow guidance now prioritizes quick discovery, exhaustive retrieval,
  citation chasing, known-item lookup, and author pivots.
- OpenAlex now has an explicit provider-specific MCP surface for OpenAlex-native
  search, cursor pagination, DOI/OpenAlex-ID lookup, citation/reference
  traversal, and author pivots without changing the default broker path.
- The additive enrichment layer now combines Crossref, Unpaywall, and OpenAlex
  metadata for known-item and smart `includeEnrichment` workflows without
  changing ranking or base-paper resolution semantics.
- Combined enrichment is now stricter too: query-only `enrich_paper` calls
  abstain unless a DOI-bearing identifier or anchored paper payload is
  available, and OpenAlex enrichment is rejected when the returned DOI does
  not match the trusted input DOI.
- ScholarAPI-sourced paper payloads now also expose a separate `contentAccess`
  block so access/full-text metadata stays distinct from bibliographic
  enrichment.
- Tool advertisement can now stay compatibility-first or hide disabled tool
  families at startup through `PAPER_CHASER_HIDE_DISABLED_TOOLS=false|true`,
  including generic Semantic Scholar-backed tools when that backend is off and
  brokered/citation-repair entry points when no usable backend remains.
- The repo now also exposes a regulation-oriented raw-tool slice: keyless
  `search_federal_register` discovery, GovInfo-backed
  `get_federal_register_document`, GovInfo-only `get_cfr_text`, and ECOS
  `frCitation` enrichment for GovInfo-linked Federal Register items so species
  dossiers can hand off directly into regulatory primary sources.
- `docs/golden-paths.md` records the primary personas, golden paths, concrete example
  flows, and success signals for future agent work.
- `.github/copilot-instructions.md` now gives GitHub-native guidance for Copilot
  and the GitHub cloud coding agent so repo planning expectations are durable
  outside the runtime MCP surface.
- `tests/test_local_config_contract.py` and
  `tests/test_repo_security_hygiene.py` now protect the public/local config
  split: `.env.example`, `docker-compose.yaml`, `.gitignore`,
  `.dockerignore`, and the Azure workflow's secret-vs-variable boundary are
  contract-tested.
- Durable docs now distinguish the wrapper's generic
  `authorization`-header default from the Azure scaffold's
  `x-backend-auth` override, and the published validation commands now match
  the repo's real pre-commit/CI/deployment gate layout.
- The highest-priority workflow-level rough edges from the last agent UX review
  have now been fixed in code, tests, and durable docs.
- Author lookup and author-pivot guidance now call out Semantic Scholar field
  support, plain-text author search normalization, and cross-provider paper ID
  portability for downstream expansion tools.
- Brokered paper results now distinguish expansion-safe identifiers with
  `recommendedExpansionId` and `expansionIdStatus` so CORE-native fallback IDs
  are not mistaken for Semantic Scholar-compatible expansion inputs.
- Exact-title lookup now degrades more cleanly: punctuation-heavy 400/404 misses
  on `search_papers_match` fall back to fuzzy title search (unquoted, then
  quoted-phrase variants) and then to a structured no-match payload.  The
  fallback search window was increased to 100 results (the API maximum).
- Known-item recovery is now stricter and safer: `resolve_citation` abstains on
  weak report-style or non-paper-like references instead of forcing a bad paper
  match, and `search_papers_match` can now recover exact-title misses through
  strict OpenAlex or Crossref confirmation before falling back to no-match.
- Regulatory known-item recovery is now explicit too: `resolve_citation`
  recognizes Federal Register and CFR-style references as non-paper primary
  sources, abstains early, and steers agents toward `search_federal_register`,
  `get_federal_register_document`, or `get_cfr_text` instead of paper search.
- Federal Register retrieval is more resilient for agent workflows: historical
  FR citations now parse out of longer citation strings, GovInfo granule
  failures fall back to FederalRegister.gov HTML when metadata is available,
  and `search_federal_register` now tolerates document-number and CFR-filter
  combinations that previously surfaced brittle 400 errors.
- Smart known-item routing no longer ends in a dead-end structured error when
  citation repair abstains. The smart workflow now falls back to title/OpenAlex
  recovery and then to a broader candidate set with explicit warnings so agents
  can keep moving while still treating the result as unconfirmed.
- DOI-backed enrichment is now safer against upstream provider drift: OpenAlex
  mismatches no longer overwrite the trusted resolved DOI, and unanchored
  query-only enrichment no longer invents a canonical DOI from the top
  Crossref hit.
- Semantic Scholar citation/reference list normalization now treats top-level
  `data: null` payloads as empty lists, preventing valid graph-expansion seeds
  from failing when upstream reference data is missing instead of merely empty.
- Brokered `search_papers` now suppresses obviously low-relevance Semantic
  Scholar hits for title-like known-item queries instead of returning garbage
  false positives.
- Smart follow-up quality is tighter: `ask_result_set` comparison mode uses a
  grounded structured fallback, `map_research_landscape` sanitizes weak theme
  labels, and `expand_research_graph` keeps saved-session intent in frontier
  scoring while filtering off-topic next-hop candidates.
- `get_serpapi_account_status` now returns only a sanitized public quota summary
  rather than the raw SerpApi account payload.
- `search_snippets` now degrades provider 4xx/5xx failures to an empty payload
  with retry guidance instead of surfacing the raw provider error.
- `.github/workflows/test-paper-chaser.md` now defines a GitHub Agentic
  Workflow smoke test for the MCP server, with the compiled workflow checked in
  as `.github/workflows/test-paper-chaser.lock.yml`.
- That workflow now targets the primary golden paths explicitly: quick
  discovery, known-item lookup, pagination, citation chasing, author pivot,
  and optional SerpApi citation export when credentials are available.
- The workflow now also accepts manual `workflow_dispatch` inputs for `smoke`,
  `comprehensive`, and `feature_probe` runs, plus an optional focus prompt so
  maintainers can direct UX review loops and turn the findings into actionable
  issues for GitHub Copilot coding agents.
- The GitHub Agentic Workflow MCP config for `paper-chaser` must stay
  containerized. Current `gh-aw` MCP Gateway releases reject legacy stdio
  `command`/`args` server definitions and require `container`-based config.
- The verifier workflow is now manual-only on purpose because it can consume
  repository secrets in a public repo. Maintainers should trigger it from the
  Actions tab when they want a UX regression pass.
- The verifier creates issues with `agentic` and `needs-copilot` labels in
  addition to `automation` and `testing`, so the auto-assignment workflow can
  pick them up. Issues include a stable `<!-- agent-loop-key: ... -->` body
  marker plus attempt metadata to prevent infinite retry loops. After 3
  failed attempts, `needs-human` replaces `needs-copilot`.
- A new `agentic-assign.yml` workflow assigns GitHub Copilot only when an issue
  carries both `agentic` and `needs-copilot` labels but not `needs-human`,
  `blocked`, or `no-agent`. It listens to direct `issues` events and completed
  `Test Paper Chaser MCP` runs because issues created from the verifier
  workflow do not reliably fan out into a second `issues`-triggered workflow.
- The Copilot auto-assignment path must use GitHub's Copilot-specific issue
  assignment contract (`copilot-swe-agent[bot]` plus `agent_assignment`). A
  plain issue assignee write with `Copilot` can log success in Actions while
  leaving the issue unassigned.
- The assignment workflow should prefer an explicit secret-backed token for the
  API call (`GH_AW_GITHUB_TOKEN`, then `COPILOT_GITHUB_TOKEN`, then
  `GITHUB_TOKEN`). In practice some repositories return 403 on the Copilot
  assignment endpoint when using the default Actions token even with
  `issues: write`, so the fallback order is part of the durable workflow
  contract.
- The workflow emphasizes agent UX evaluation: each test step includes an
  explicit UX check for intuitiveness, unnecessary round trips, missing
  features, confusing contracts, and dead-end responses. A new step 11
  produces a structured "UX friction summary" before issue creation. The
  model can be configured via the `GH_AW_MODEL_AGENT_COPILOT` Actions
  variable (e.g., set to `gpt-5.4` to use GPT-5.4).
- Microsoft-oriented packaging assets are now checked in as
  `mcp-tools.core.json`, `mcp-tools.full.json`, and
  `microsoft-plugin.sample.json` so one universal server can still ship a
  conservative or full Streamable HTTP packaging profile.
- The README tool table now includes all three regulatory tools
  (`search_federal_register`, `get_federal_register_document`, `get_cfr_text`)
  and a dedicated Federal Register / GovInfo configuration section documents
  `PAPER_CHASER_ENABLE_FEDERAL_REGISTER`, `PAPER_CHASER_ENABLE_GOVINFO_CFR`,
  `GOVINFO_API_KEY`, `FEDERAL_REGISTER_TIMEOUT_SECONDS`,
  `GOVINFO_TIMEOUT_SECONDS`, `GOVINFO_DOCUMENT_TIMEOUT_SECONDS`, and
  `GOVINFO_MAX_DOCUMENT_SIZE_MB`.
- OpenAlex institution/source/topic pivots are now first-class tools:
  `search_entities_openalex` and `search_papers_openalex_by_entity` expose
  explicit entity-scoped work retrieval with OpenAlex cursor pagination.
- ScholarAPI is now integrated as an explicit provider family for ranked
  discovery, indexed-at listing, full-text retrieval, and PDF retrieval, and it
  can also be steered into the raw `search_papers` broker through
  `preferredProvider` or `providerOrder` without changing the default free
  broker order.
- The smart/agentic retrieval layer can now also include ScholarAPI when it is
  enabled: planner `providerPlan` guidance now admits ScholarAPI, smart
  retrieval respects that plan, and smart-provider budgets can cap it through
  `maxScholarApiCalls` without changing the default raw broker order.
- `pyproject.toml` project description and keywords are updated to reflect the
  full current provider surface (Semantic Scholar, CORE, arXiv, OpenAlex,
  SerpApi, Crossref, Unpaywall, ECOS, Federal Register, GovInfo).

## Recent Repo Rename Follow-up

The GitHub repository slug now matches the Paper Chaser package and public MCP
identity. The remaining rebrand-sensitive guardrails are:

1. Keep repo URL metadata and links aligned across `README.md`,
  `pyproject.toml`, `server.json`, and Docker OCI labels in `Dockerfile`.
2. Re-scan for exact old-slug references with a search like
  `scholar-search-mcp|scholar_search_mcp|SCHOLAR_SEARCH_` and confirm that only
  intentionally historical or provider-specific references remain.
3. Re-run the rebrand-sensitive validation subset when touching repo metadata:
  `python -m pytest tests/test_mcp_package_contract.py tests/test_local_config_contract.py tests/test_microsoft_assets.py`
  and `python scripts/validate_deployment.py --skip-az --skip-docker`.
4. Review external integrations that are not stored in this repo, such as MCP
  Registry listing metadata, GHCR package description settings, and any
  third-party docs that pinned the old repo slug.

## Module Map

- `paper_chaser_mcp/__main__.py` is the `python -m` entrypoint.
- `paper_chaser_mcp/server.py` is the public MCP facade and compatibility layer used by tests and package entrypoints.
- `paper_chaser_mcp/compat.py` owns schema sanitization plus additive
  `agentHints` / `clarification` / `resourceUris` / `searchSessionId`
  enrichment for raw-tool responses.
- `paper_chaser_mcp/dispatch.py` routes MCP tool calls through a dispatch map.
- `paper_chaser_mcp/search.py` owns the `search_papers` fallback chain and merged response helpers.
- `paper_chaser_mcp/tools.py` defines MCP tool schemas.
- `paper_chaser_mcp/runtime.py` owns stdio startup.
- `paper_chaser_mcp/settings.py` contains environment parsing helpers.
- `paper_chaser_mcp/agentic/` contains the additive smart-tool runtime:
  config, models, provider adapters, planner, retrieval, ranking, workspace,
  and LangGraph scaffolding.
- `paper_chaser_mcp/clients/federal_register/` contains the keyless
  FederalRegister.gov discovery client.
- `paper_chaser_mcp/clients/govinfo/` contains the GovInfo retrieval client
  for authoritative Federal Register and CFR text.
- `paper_chaser_mcp/deployment.py` wraps the HTTP app with `/healthz`,
  optional backend-token auth, and optional Origin allowlisting for hosted
  deployments.
- `paper_chaser_mcp/deployment_runner.py` powers the explicit
  `paper-chaser-mcp deployment-http` runtime used for HTTP-wrapper hosting in
  Compose and Azure-hosted deployments.
- `paper_chaser_mcp/deployment_utils.py` resolves smoke-test endpoints from
  Azure deployment outputs or an explicit environment override.
- `paper_chaser_mcp/clients/` contains provider clients for Semantic Scholar,
  arXiv, CORE, OpenAlex, SerpApi, Crossref, Unpaywall, ECOS, Federal Register,
  and GovInfo.
- `paper_chaser_mcp/models/common.py` contains shared Pydantic models including `Paper` (with `scholarResultId`).
- `paper_chaser_mcp/parsing.py`, `paper_chaser_mcp/constants.py`, and `paper_chaser_mcp/transport.py` hold shared helper code and compatibility imports.
- `scripts/validate_deployment.py` validates the Azure/Docker deployment path.
- `infra/` contains the Azure Container Apps + API Management Bicep scaffold
  and parameter files for `dev`, `staging`, and `prod`.

## Validation Commands

From the project root, install the package with development extras and then run the validation commands inside the repository virtual environment:

```bash
pip install -e ".[all]"
```

Then run:

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

If you prefer to trigger the heavy hook-managed checks through pre-commit,
`pre-commit run --hook-stage manual --all-files` runs the manual-stage
`pip check`, coverage, build, and `pip-audit` hooks.

If you touch the Azure deployment wrapper, Dockerfile, Bicep, APIM policy,
deployment docs, or deployment workflow, also run:

```bash
python scripts/validate_psrule_azure.py
python scripts/validate_deployment.py --skip-docker
```

For full parity with the `Deploy Azure` workflow's full-deployment path, run:

```bash
python scripts/validate_deployment.py --require-az --require-docker --image-tag paper-chaser-mcp:ci-validate
```

If you edit `.github/workflows/test-paper-chaser.md`, recompile it before
finishing and then rerun the standard validation stack so pre-commit can
normalize the generated lock file:

```bash
gh aw compile test-paper-chaser --dir .github/workflows
```

## What Was Added In This Pass

- `Paper.scholarResultId` is now a first-class model field (not just an extra). This
  makes it visible in the JSON schema so agents can discover it without reading long
  tool descriptions. The field is always `None` for non-SerpApi results.
- The additive smart-tool layer now exists under `paper_chaser_mcp/agentic/`.
  It keeps the raw MCP tools stable while adding concept-level discovery,
  grounded follow-up QA, landscape mapping, and compact graph expansion
  through reusable `searchSessionId` workspaces.
- FastMCP `Context` is now injected into tool handlers without appearing in the
  public MCP schema. That enables progress updates, optional sampling, and
  optional bounded elicitation without changing tool arguments.
- Raw-tool responses now carry additive `agentHints`, `clarification`,
  `resourceUris`, and `searchSessionId` metadata so low-context agents can
  continue the workflow more easily.
- A first-class `resolve_citation` workflow now sits alongside
  `search_papers_match`, `search_snippets`, and `get_paper_details` so agents
  can repair incomplete references without guessing which raw tool to try next.
- The server now exposes `paper://{paper_id}`, `author://{author_id}`,
  `search://{searchSessionId}`, and
  `trail://paper/{paper_id}?direction=citations|references` resources.
- Checked-in Microsoft packaging assets now document a conservative raw-tool
  subset and a fuller smart-tool package over Streamable HTTP without forking
  the runtime.
- `BrokerMetadata.nextStepHint` was added as a new field. The `_metadata()` helper
  in `search.py` now populates it with provider-specific guidance:
  - For `serpapi_google_scholar` results: hints that `paper.scholarResultId` can be
    passed to `get_paper_citation_formats`.
  - For "none" results: hints to broaden the query or try `search_papers_bulk`.
  - For all other results: hints to use `search_papers_bulk` or citation expansion.
- `SERVER_INSTRUCTIONS` was restructured as a numbered decision tree for faster
  agent scanning: QUICK DISCOVERY → EXHAUSTIVE → CITATION REPAIR → KNOWN ITEM
  → CITATION → AUTHOR → SNIPPET.
- `AGENT_WORKFLOW_GUIDE` was rewritten with a quick-decision-table format under
  `guide://paper-chaser/agent-workflows`.
- `plan_paper_chaser_search` prompt was updated to reference `brokerMetadata.nextStepHint`.
- `docs/golden-paths.md` now includes concrete example requests and tool sequences
  for each golden path.
- `search_papers_match` now normalizes wrapped Semantic Scholar match responses
  to one clean paper-shaped payload.
- `search_papers_match` can now recover exact-title misses via strict OpenAlex
  and Crossref confirmation before falling back to structured no-match.
- `resolve_citation` now prefers abstention plus alternatives for weak
  report-style or non-paper-looking references instead of forcing a canonical
  paper match.
- `search_papers` now suppresses low-relevance title-like false positives and
  routes agents toward the dedicated known-item tools instead.
- `get_serpapi_account_status` now returns a strict sanitized quota summary with
  no raw upstream credential or account-identifier passthrough.
- `ask_result_set(answerMode="comparison")` now uses a grounded structured
  fallback, `map_research_landscape` sanitizes weak labels before emission, and
  `expand_research_graph` keeps saved-session intent in graph-frontier scoring.
- CORE search now follows redirects so brokered fallbacks into CORE do not
  record avoidable failed provider attempts for predictable 301s.
- `brokerMetadata.nextStepHint` now distinguishes between the closest
  continuation path and a Semantic Scholar/provider pivot.
- Provider-specific search tool schemas now expose only the parameters their
  providers actually honor.
- Success-metric tests now lock workflow cues into tool descriptions, runtime
  onboarding resources, prompt text, and `.github/copilot-instructions.md`.
- CORE search now retries short-lived 5xx responses, which was required after
  live broker smoke testing exposed transient backend shard failures on the
  CORE fallback hop.
- A hosted deployment path now exists for Azure: `paper_chaser_mcp.deployment`
  adds `/healthz`, optional backend-token auth, and optional Origin allowlists
  in front of the FastMCP HTTP app.
- The repo now ships a Dockerfile, Bicep infrastructure scaffold, APIM policy
  template, and `.github/workflows/deploy-azure.yml` for private-network Azure
  deployment without committing live secrets.
- Deployment validation is now codified in `scripts/validate_deployment.py` and
  wired into pre-commit and CI so Bicep, Docker, APIM policy XML, and the
  secured HTTP wrapper are exercised before release work ships.

## Progress Snapshot

- Baseline validation (`python -m pytest`, `python -m mypy --config-file pyproject.toml`,
  `python -m ruff check .`, and `python -m bandit -c pyproject.toml -r paper_chaser_mcp`)
  passed in this environment after installing `.[dev]`.
- The current pass keeps runtime behavior stable and focuses on schema discoverability,
  structured hints, and tighter agent guidance.
- The current pass also adds the compatibility-first smart workflow surface,
  workspace registry, schema sanitization, and resource-oriented follow-up UX.
- The highest-impact UX issues from the last live agent pass were addressed with
  targeted runtime fixes and regression coverage.
- The latest pass tightens expansion-ID guidance so brokered CORE `canonicalId`
  fallback values are no longer presented as universally portable into Semantic
  Scholar citation/author expansion flows.
- Current follow-up work is now mostly product-shaping work around provider
  preferences and whether retry-recovered provider behavior should be surfaced
  to agents.
- Optional elicitation now supports one bounded clarification round on
  `search_papers`, `search_papers_match`, and `search_authors` when the client
  advertises MCP elicitation support. Declined or cancelled elicitation falls
  back to the normal `clarification` response field.
- The latest pass also hardens author lookup UX: unsupported author fields now
  fail locally with a clearer validation error, exact-name punctuation is
  normalized before `/author/search`, and paper-to-author expansion errors now
  explain when a brokered provider ID is not portable to Semantic Scholar.
- The current pass also improves messy-title and quote-recovery reliability:
  `search_papers_match` uses a fuzzy Semantic Scholar fallback before returning
  a structured no-match payload, and `search_snippets` now returns empty
  degraded results on provider 400/404/5xx responses.
- A live broker smoke test was completed against the configured providers in
  this workspace. It confirmed the new hint wording live and exposed a
  transient CORE 500, which is now mitigated by short retries in the client.
- Provider enablement is now fully parameterized in the Azure Bicep scaffold.
  All ten providers (`enableSemanticScholar`, `enableArxiv`, `enableCore`,
  `enableOpenAlex`, `enableSerpApi`, `enableCrossref`, `enableUnpaywall`,
  `enableEcos`, `enableFederalRegister`, `enableGovinfoCfr`) are controlled by named Bicep parameters in `infra/main.bicep`
  and threaded through to the Container App. The previously-hardcoded
  `PAPER_CHASER_ENABLE_CORE=true` mismatch is fixed: CORE now defaults to
  `false` in Bicep, `.bicepparam` files, and both Compose files, matching the
  application default in `settings.py`. The `core-api-key` Key Vault secret is
  also now conditional — it is only mounted when `enableCore=true`, so
  environments without a CORE API key no longer require an orphaned Key Vault
  entry. Email/URL overrides for Crossref, Unpaywall, and ECOS (`crossrefMailto`,
  `unpayWallEmail`, `ecosBaseUrl`, `ecosVerifyTls`) are exposed as plain Bicep
  params and injected as env vars only when non-empty (Crossref/Unpaywall/ECOS
  do not require Key Vault secrets). A new provider enablement matrix table and a
  complete Bicep parameters reference table are now in
  `docs/azure-deployment.md`.

- The repo now carries a checked-in agentic workflow source/lock pair for
  high-level MCP smoke testing, covering quick discovery, known-item lookup,
  pagination, provider spot checks, author pivot, optional citation export,
  and no-result behavior.
- Manual workflow_dispatch runs can now switch between smoke, comprehensive,
  and feature-probe modes, with an optional focus prompt to steer deeper UX
  probes toward a specific feature or rough edge.
- The latest pass adds explicit OpenAlex tools instead of brokering OpenAlex
  through `search_papers`, because the OpenAlex guide's citation, author, and
  cursor semantics differ enough from Semantic Scholar to warrant a separate
  provider-specific surface.
- The agentic workflow now runs `paper-chaser` through a `python:3.14`
  container mounted to `${GITHUB_WORKSPACE}` so the generated MCP Gateway
  config matches the current schema.
- `search_papers_bulk` now truncates returned data to the requested `limit`
  because the upstream Semantic Scholar bulk endpoint can ignore small limits;
  agents should still prefer `search_papers` or
  `search_papers_semantic_scholar` for small targeted pages.
- The Azure deployment scaffold has a full local validation path:
  `python scripts/validate_deployment.py --require-az --require-docker`
  exercises Bicep lint/build, Docker build, blocked-origin `403`, missing
  backend-header `401`, and allowed-origin `X-Backend-Auth` smoke checks for
  `/mcp/`. The lighter routine path remains
  `python scripts/validate_deployment.py --skip-docker`.
- Targeted pre-commit checks over changed deployment files should include
  `actionlint`, `gitleaks`, `ruff`, `mypy`, `bandit`, and the local deployment
  validator hook.

## Follow-up Completed

1. `search_papers_match` response-shape ambiguity is fixed.
  Wrapped Semantic Scholar match responses are now normalized to one clean
  paper payload, and regression tests cover both direct client normalization and
  FastMCP structured output.

2. The default `search_papers` broker path now follows predictable CORE
  redirects.
  CORE requests explicitly follow redirects so avoidable 301-driven broker
  failures do not pollute `brokerMetadata.attemptedProviders`.

3. `brokerMetadata.nextStepHint` now distinguishes continuation from pivots.
  Semantic Scholar responses describe bulk retrieval as the closest
  continuation path when appropriate, while CORE/arXiv/SerpApi responses label
  it as a Semantic Scholar pivot. Venue-filtered searches explicitly warn that
  bulk retrieval broadens semantics.

4. Provider-specific search schemas no longer over-advertise unsupported
  filters.
  CORE, SerpApi, and arXiv tool schemas now expose only `query`, `limit`, and
  `year`, while the Semantic Scholar-only tool retains the broader compatible
  filter set.

5. Live broker smoke coverage is complete for the current environment.
  A fresh process using the workspace MCP config exercised the default broker
  path, a SerpApi-preferred path, and a venue-filtered Semantic Scholar path.
  The hint wording behaved as intended live, and the CORE path now survives
  transient backend 500s by retrying before falling through.

6. `search_papers_bulk` now enforces small requested limits client-side.
   The Semantic Scholar bulk API may still return its large provider batch even
   when asked for fewer records, so the client truncates `data` after
   normalization and the durable docs now steer small targeted queries toward
   `search_papers` or `search_papers_semantic_scholar`.

7. Author lookup/pivot errors now surface actionable Semantic Scholar guidance.
    `search_authors` normalizes exact-name punctuation, author-field inputs are
    validated against the supported author schema before the upstream call, and
    `get_paper_authors`/author-profile failures now explain when to retry with
    `paper.recommendedExpansionId` and when brokered identifiers still require a
    DOI or Semantic Scholar-native lookup first.

8. Known-item and snippet recovery now degrade gracefully.
    `search_papers_match` no longer leaks upstream 404s for no-match title
   lookups; it retries through fuzzy Semantic Scholar search and returns a
   structured no-match payload when the item still cannot be recovered. Common
   reasons include punctuation drift and outputs outside the indexed paper
   surface such as dissertations or software releases. `search_snippets`
   similarly returns empty degraded results with retry guidance on provider
   4xx/5xx failures.

9. Brokered expansion IDs now separate portability from canonicalization.
   `Paper` now exposes `recommendedExpansionId` and `expansionIdStatus` so
   agents can tell when a brokered result already has a Semantic
   Scholar-compatible identifier and when provider-native IDs still require a
   DOI or Semantic Scholar lookup first. This is especially important for
   CORE results whose `canonicalId` may still fall back to a raw CORE id when
   no DOI is present.

10. `search_papers_match` fallback now uses quoted-phrase search as a final
    recovery layer before returning a structured no-match payload.
    Semantic Scholar's `/paper/search` endpoint treats unquoted title words as
    separate keywords, so common-word titles like "Attention Is All You Need"
    can rank below 100 generic attention-mechanism papers.  The fallback now
    appends quoted variants (e.g. `'"Attention Is All You Need"'`) after the
    plain keyword searches so the exact phrase gets a second chance via
    Semantic Scholar's phrase-match semantics.  The fallback limit was also
    increased from 30 to 100 (the API maximum) to widen the candidate window
    for relevance-ranked results.

11. `search_papers_bulk` default ordering is now explicitly disclosed.
    The bulk endpoint uses exhaustive corpus traversal with an internal ordering
    that is NOT relevance-ranked.  This is a semantic pivot from `search_papers`
    results, not "page 2". Every `search_papers_bulk` response now includes a
    `retrievalNote` field describing the active ordering contract.
    `brokerMetadata.nextStepHint` from `search_papers` (Semantic Scholar path)
    also explicitly warns about the ordering change and suggests
    `sort='citationCount:desc'` as a citation-ranked alternative.
    The `search_papers_bulk` tool description also calls out this ordering
    difference prominently so agents don't need extra round trips to infer it.

## Known Hotspots

- `CoreApiClient._result_to_paper()` remains the densest parsing logic and should keep getting defensive tests before behavior changes.
- The compatibility contract in `paper_chaser_mcp/server.py` is now important. Future cleanup should avoid removing re-exported symbols that tests and downstream imports still rely on.
- `pyproject.toml` is the single source of truth for Python dependencies; no parallel runtime dependency file should be reintroduced casually.
- Dependency version ranges remain intentionally loose.
- `search.py::_metadata()` now carries product-level UX weight because agents use `nextStepHint` as operational guidance rather than decorative metadata.
- `clients/openalex/client.py` now carries OpenAlex-specific normalization logic,
  especially DOI/OpenAlex-ID parsing, abstract reconstruction, and batched
  `referenced_works` hydration.

## Suggested Next Steps

1. Merge the `0.2.0` release-prep branch described in `docs/release-publishing-plan.md`, then tag `v0.2.0` from updated `master` after the normal validation pass.
2. Consider whether budget-aware provider preferences should become a more explicit first-class planning concept in the docs and prompt surfaces.
3. Decide whether retry-recovered provider behavior should remain invisible or become broker metadata.
4. Add more negative tests for CORE schema drift, especially malformed author shapes, journal fields, and URL containers.
5. Consider moving from per-request `httpx.AsyncClient` creation to shared clients if connection reuse becomes important.
6. Decide whether the compatibility facade in `paper_chaser_mcp/server.py` should remain broad or be narrowed with an explicit supported surface.
7. Revisit `.github/copilot-instructions.md` and `docs/golden-paths.md` whenever future agent-facing search behavior materially changes.
8. ~~OpenAlex institution/source pivots~~ — **completed**: `search_entities_openalex`
   and `search_papers_openalex_by_entity` are now first-class tools.
9. Tune the smart discovery thresholds and trace logs once real-world
   `search_papers_smart` runs accumulate enough UX feedback.
10. Expand the agentic workflow once stable secrets are available for deeper provider-specific assertions, for example optional SerpApi or OpenAI-backed smart-tool coverage.
11. Act on the first UX friction report produced by the
    `GH_AW_MODEL_AGENT_COPILOT`-configured verifier. Focus on the
    highest-impact item from the structured "UX friction summary": likely a
    round-trip reduction or a clearly missing feature rather than cosmetic work.
12. Consider adding a dedicated ECOS-to-regulatory golden path in
    `docs/golden-paths.md` covering the full chain from species dossier through
    `frCitation` to GovInfo/FR retrieval, now that all three regulatory tools
    are stable and documented.
13. Evaluate whether a Federal Register walkthrough (analogous to the ECOS
    California-least-tern walkthrough in the README) should be added to help
    agents discover the discovery→direct-retrieval→CFR-text chain without
    reading the full tool descriptions.
14. Keep the repo metadata sweep complete whenever GitHub-side settings change,
  especially if MCP Registry listing metadata, GHCR package descriptions, or
  external docs drift from the renamed Paper Chaser identity.

## Ready Handoff Prompt

Use this prompt for the next agent if the goal is to act on the UX review or expand coverage:

```text
You are picking up paper-chaser-mcp after a documentation refresh pass. Read README.md, docs/golden-paths.md, and docs/agent-handoff.md first.

Focus on the highest-value remaining work from the Suggested Next Steps list:
1. Decide whether retry-recovered provider behavior should remain invisible or become broker metadata.
2. Tune smart discovery thresholds and trace logging as real-world search_papers_smart runs accumulate UX feedback.
3. Act on the UX friction summary produced by the GH_AW_MODEL_AGENT_COPILOT-configured verifier — start with the highest-impact item (likely a round-trip reduction or a missing feature signal) rather than cosmetic changes.

Your task:
- Inspect the broker, onboarding, and durable-doc surfaces that describe the
  intended workflow contract.
- Implement the smallest complete change you can defend for any provider-steering,
  metadata, or UX contract decision you touch.
- Add or update targeted tests and durable docs if agent-facing behavior changes.

Validation target:
- python -m pip check
- pre-commit run --all-files
- python -m pytest --cov=paper_chaser_mcp --cov-report=term-missing --cov-fail-under=85
- python -m mypy --config-file pyproject.toml
- python -m ruff check .
- python -m bandit -c pyproject.toml -r paper_chaser_mcp
- python -m build
- python -m pip_audit . --progress-spinner off

Do not broaden scope unless a blocking dependency makes it necessary.
```

## Commit Hygiene

- Keep validation and documentation updates in the same change as the code they describe.
- Prefer commit messages that make the validation or handoff intent obvious to the next reviewer or agent.
