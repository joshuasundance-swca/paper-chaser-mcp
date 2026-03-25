# Azure Deployment

This guide describes how to deploy the Paper Chaser MCP server to Azure with
private ingress, private secret access, and a private delivery path. The
tracked workflow supports a two-phase rollout: a GitHub-hosted bootstrap for
control-plane infrastructure, then a private-runner full deployment for image
build/push, workload rollout, and smoke testing.

## What gets committed

- `Dockerfile`
- `infra/` Bicep templates
- `.github/workflows/deploy-azure.yml`
- deployment docs and policy templates

## What stays out of the repo

- Azure Key Vault secret values
- production URLs and private hostnames
- nonpublic smoke-test URLs used by automation
- subscription-specific secure parameters beyond environment or repository variables

## Runtime entrypoint

The Azure Container App should explicitly run:

```text
paper-chaser-mcp deployment-http
```

`paper-chaser-mcp deployment-http` launches the
`paper_chaser_mcp.deployment:app` wrapper, binds to `PORT` when the platform
provides it, and otherwise falls back to `PAPER_CHASER_HTTP_PORT` (default
`8080` in the deployment path). The deployment wrapper adds:

- `/healthz`
- optional shared-token enforcement for `/mcp`
- optional Origin allowlisting for `/mcp`

This explicit subcommand keeps the hosted HTTP wrapper contract stable even if
the public OCI package defaults to stdio transport for local MCP subprocess
usage.

## Required Azure setup

1. Create a dedicated resource group per environment.
2. Create a GitHub OIDC federated identity in Azure.
3. Provide a private self-hosted GitHub Actions runner with Docker, Azure CLI,
   private DNS resolution, and line-of-sight to the deployment VNet for full
   deployments. Bootstrap mode can run on a GitHub-hosted runner because it
   uses Azure control-plane APIs only.
4. Create GitHub environment secrets for deployment identifiers, plus the
   optional runner-label variable if you need to override the default labels.
5. Populate Key Vault with these secret names before production deployment:
   - `openai-api-key` if you enable the smart layer with `agenticProvider=openai`
   - `core-api-key` **only** if you set `enableCore=true`; CORE is disabled by
     default in this scaffold (matching the application default). Leave this
     secret out until you deliberately opt in.
   - `semantic-scholar-api-key`
   - `openalex-api-key`
   - `openalex-mailto`
   - `serpapi-api-key` if `enableSerpApi=true`
   - `mcp-backend-auth-token`

   Crossref, Unpaywall, ECOS, and Federal Register are enabled by default but
   **do not require Key Vault secrets**. Their polite-pool contact values
   (`crossrefMailto`, `unpayWallEmail`) are plain Bicep string params and can
   be left empty to use the application defaults.

   GovInfo CFR is enabled by default and works without a key (falls back to
   degraded HTML recovery), but provisioning a `govinfo-api-key` in Key Vault
   grants authoritative XML access.

## Provider enablement matrix

The table below shows the default state of every search provider in this Azure
scaffold, the Bicep parameter that controls it, and whether it needs a Key Vault
secret.

| Provider | Default | Bicep param | Requires Key Vault secret? |
| --- | --- | --- | --- |
| Semantic Scholar | **on** | `enableSemanticScholar` | Yes (`semantic-scholar-api-key`) |
| arXiv | **on** | `enableArxiv` | No |
| CORE | **off** | `enableCore` | Yes (`core-api-key`) — only needed when enabled |
| OpenAlex | **on** | `enableOpenAlex` | Yes (`openalex-api-key`, `openalex-mailto`) |
| SerpApi | off | `enableSerpApi` | Yes (`serpapi-api-key`) — only needed when enabled |
| Crossref | **on** | `enableCrossref` | No (optional `crossrefMailto` plain param) |
| Unpaywall | **on** | `enableUnpaywall` | No (optional `unpayWallEmail` plain param) |
| ECOS | **on** | `enableEcos` | No |
| Federal Register | **on** | `enableFederalRegister` | No |
| GovInfo CFR | **on** | `enableGovinfoCfr` | Optional (`govinfo-api-key`) — authoritative XML; degrades to HTML without it |

The smart layer (`enableAgentic=true` with `agenticProvider=openai`) adds a
separate `openai-api-key` requirement.

## Bicep parameters reference

All Bicep parameters below can be set in the environment `.bicepparam` file or
passed as overrides at deployment time. Sensitive runtime values always belong
in Key Vault, never in `.bicepparam`.

### Infrastructure shape

| Parameter | Default | Purpose |
| --- | --- | --- |
| `location` | resource group location | Azure region for all resources |
| `environmentName` | *(required)* | Environment label used in resource names (e.g. `dev`, `staging`, `prod`) |
| `appName` | `paper-chaser` | Short application name prefix for resource names |
| `imageRepository` | `paper-chaser-mcp` | Container image name inside ACR |
| `imageTag` | `latest` | Container image tag to deploy |
| `deployMode` | `full` | `bootstrap` (first run) or `full` (image + workload) |
| `containerCpu` | `1` | CPU cores for the Container App container |
| `containerMemory` | `2Gi` | Memory for the Container App container |
| `minReplicas` | `1` | Minimum scale-to-zero boundary (0 for dev, ≥1 for prod) |
| `maxReplicas` | `3` | Maximum horizontal replica count |
| `tags` | application/environment/managedBy | Safe resource tags; do not add secrets or private identifiers |

### API Management

| Parameter | Default | Purpose |
| --- | --- | --- |
| `apiManagementSku` | `StandardV2` | APIM SKU (`StandardV2` or `PremiumV2`) |
| `apiManagementApiPath` | `paper-chaser` | Relative API path in APIM |
| `apiManagementPublisherName` | `Paper Chaser MCP` | Publisher display name in APIM |
| `apiManagementPublisherEmail` | `owner@example.invalid` | Publisher email in APIM; use a non-secret operational inbox |

### Provider enable flags

| Parameter | Default | Controls env var |
| --- | --- | --- |
| `enableSemanticScholar` | `true` | `PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR` |
| `enableArxiv` | `true` | `PAPER_CHASER_ENABLE_ARXIV` |
| `enableCore` | **`false`** | `PAPER_CHASER_ENABLE_CORE` |
| `enableOpenAlex` | `true` | `PAPER_CHASER_ENABLE_OPENALEX` |
| `enableSerpApi` | `false` | `PAPER_CHASER_ENABLE_SERPAPI` |
| `enableCrossref` | `true` | `PAPER_CHASER_ENABLE_CROSSREF` |
| `enableUnpaywall` | `true` | `PAPER_CHASER_ENABLE_UNPAYWALL` |
| `enableEcos` | `true` | `PAPER_CHASER_ENABLE_ECOS` |
| `enableFederalRegister` | `true` | `PAPER_CHASER_ENABLE_FEDERAL_REGISTER` |
| `enableGovinfoCfr` | `true` | `PAPER_CHASER_ENABLE_GOVINFO_CFR` |

Note: setting `enableCore=true` also causes the scaffold to mount the
`core-api-key` Key Vault secret into the container. That secret must exist in
Key Vault before you run a `full` deployment with CORE enabled.

### Provider contact / URL overrides

| Parameter | Default | Controls env var |
| --- | --- | --- |
| `crossrefMailto` | `''` (use app default) | `CROSSREF_MAILTO` — only injected when non-empty |
| `unpayWallEmail` | `''` (use app default) | `UNPAYWALL_EMAIL` — only injected when non-empty |
| `ecosBaseUrl` | `''` (use app default) | `ECOS_BASE_URL` — only injected when non-empty |
| `ecosVerifyTls` | `true` | `ECOS_VERIFY_TLS` |

### Smart / agentic layer

| Parameter | Default | Controls env var |
| --- | --- | --- |
| `enableAgentic` | `false` | `PAPER_CHASER_ENABLE_AGENTIC` |
| `agenticProvider` | `openai` | `PAPER_CHASER_AGENTIC_PROVIDER` |
| `plannerModel` | `gpt-5.4-mini` | `PAPER_CHASER_PLANNER_MODEL` |
| `synthesisModel` | `gpt-5.4` | `PAPER_CHASER_SYNTHESIS_MODEL` |
| `embeddingModel` | `text-embedding-3-large` | `PAPER_CHASER_EMBEDDING_MODEL` |
| `disableEmbeddings` | `true` | `PAPER_CHASER_DISABLE_EMBEDDINGS` |
| `agenticOpenAiTimeoutSeconds` | `30` | `PAPER_CHASER_AGENTIC_OPENAI_TIMEOUT_SECONDS` |
| `agenticIndexBackend` | `memory` | `PAPER_CHASER_AGENTIC_INDEX_BACKEND` |
| `sessionTtlSeconds` | `1800` | `PAPER_CHASER_SESSION_TTL_SECONDS` |
| `enableAgenticTraceLog` | `false` | `PAPER_CHASER_ENABLE_AGENTIC_TRACE_LOG` |

## Deployment modes

### `bootstrap`

Use bootstrap mode the first time you stand up an environment.

- Runs on a GitHub-hosted runner.
- Uses GitHub OIDC plus Azure control-plane access only.
- Creates the network, ACR, Key Vault, managed identities, monitoring, and
  other shared infrastructure.
- Does not build or push an image.
- Does not deploy the Container App or API Management.
- Does not run a private smoke test.

### `full`

Use full mode after bootstrap and after Key Vault contains the expected secret
names.

- Runs on the private self-hosted runner.
- Builds, scans, and pushes the container image.
- Deploys the Container App and API Management.
- Passes the checked-in smart-layer defaults through the Container App:
  `PAPER_CHASER_ENABLE_AGENTIC`, `PAPER_CHASER_AGENTIC_PROVIDER`,
  `PAPER_CHASER_PLANNER_MODEL`, `PAPER_CHASER_SYNTHESIS_MODEL`,
  `PAPER_CHASER_EMBEDDING_MODEL`,
  `PAPER_CHASER_AGENTIC_INDEX_BACKEND`,
  `PAPER_CHASER_SESSION_TTL_SECONDS`, and
  `PAPER_CHASER_ENABLE_AGENTIC_TRACE_LOG`.
- Resolves the health-check URL from `SMOKE_TEST_HEALTH_URL` when provided, or
  falls back to Azure deployment outputs (`containerAppHealthUrl` first, then
  `containerAppFqdn`) for the first rollout.
- Smoke-tests `/healthz` after the deployment succeeds.

## Requirements table

Use this table as the deployment readiness checklist. The last two columns are the important ones when you are trying to work out what you can realistically supply yourself.

| Requirement | Why it exists | Where it is used | Typical control needed | Can the repo provide it? | Likely blocker if your permissions are limited |
| --- | --- | --- | --- | --- | --- |
| Dedicated Azure resource group per environment | Keeps `dev`, `staging`, and `prod` isolated and gives the Bicep deployment a target scope | `infra/main.bicep`, `az deployment group what-if`, `az deployment group create` | Azure subscription or resource-group deployment rights | No. The repo only provides the templates. | Yes if you cannot create or deploy resources in the target subscription or resource group. |
| Private network foundation for APIM, Container Apps, Key Vault, and ACR | The scaffold is private-by-default and expects private endpoints, private DNS, and VNet integration | `infra/modules/network.bicep` and the service modules referenced by `infra/main.bicep` | Azure networking and resource deployment rights in the target environment | Partially. The repo defines the IaC, but not the actual Azure environment. | Yes if you cannot deploy networking resources or do not control private DNS and routing. |
| GitHub OIDC federated identity in Azure | Lets `azure/login` authenticate without a stored client secret | `.github/workflows/deploy-azure.yml` | Azure and Entra permissions to create or manage the workload identity binding | No. This must exist in your tenant. | Yes if you cannot create or manage the federated credential or service principal setup. |
| Private self-hosted GitHub Actions runner | The build, Trivy scan, ACR push, deployment, and smoke test all run from a network that can reach private Azure endpoints | `.github/workflows/deploy-azure.yml` | GitHub runner administration plus network/DNS access to the private environment | No. The repo can target a runner, but it cannot provision or register one. | Yes if you cannot register runners or cannot place one on the trusted network. |
| GitHub Actions variables: optional `AZURE_PRIVATE_RUNNER_LABELS_JSON` | Supplies the private-runner label override without committing values to git | `.github/workflows/deploy-azure.yml` | GitHub repository or environment variable management rights | No. The repo can document the name only. | Usually not, but choose generic labels if you do not want the label set itself to reveal network or platform details. |
| GitHub Actions secrets: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `ACR_NAME`, optional `IMAGE_REPOSITORY`, `SMOKE_TEST_HEALTH_URL` | Supplies deployment identifiers and the private health endpoint without treating them as public workflow metadata | `.github/workflows/deploy-azure.yml` and post-deploy smoke checks | GitHub repository or environment secret management rights | No. The repo cannot create or populate real secrets. | Yes if you cannot edit repository or environment secrets. |
| Key Vault secret values: optional `openai-api-key`, `core-api-key`, `semantic-scholar-api-key`, `openalex-api-key`, `openalex-mailto`, optional `serpapi-api-key`, `mcp-backend-auth-token` | Supplies upstream provider credentials and the APIM-to-backend shared token | `infra/main.bicep`, runtime secret references, APIM named value resolution | Azure Key Vault secret-set permissions plus access to the real credential values | No. The repo can reference the secret names, not provide their contents. | Yes if you do not own the provider accounts, do not have the actual secrets, or cannot write them into Key Vault. |
| ACR push path and image scanning prerequisites | The workflow expects Docker, Trivy, and ACR login to succeed before deployment | `.github/workflows/deploy-azure.yml`, `scripts/validate_deployment.py`, `Dockerfile` | Runner access, ACR access, and the ability to fix image vulnerabilities when found | Partially. The repo provides the image and validation logic. | Sometimes. You may be blocked if you cannot push to ACR or cannot remediate policy-failing image findings. |
| APIM client-access configuration | Private access is in place, but consumer authentication still needs operational setup such as subscription keys and optional JWT policy | `infra/modules/apim.bicep`, `docs/azure-security-model.md` | APIM administration rights | Partially. The repo defines the gateway and backend token path. | Yes if you need to manage client credentials but do not have APIM admin rights. |
| A formal ACR quarantine and promotion process | This is the remaining hardening step beyond the current Trivy gate | Documented as a future step in this guide | Registry policy control plus a release workflow owner | Not yet. The repo does not currently implement end-to-end quarantine promotion. | Yes if your security bar requires quarantine-based promotion before release. |

## What this repo can and cannot provide

The repo can provide the infrastructure templates, the deployment workflow, the Docker image build, the local validation scripts, and the documentation for how the pieces fit together.

The repo cannot provide:

- Azure subscription, Entra, GitHub admin, or network permissions
- a private self-hosted runner
- private DNS reachability or VNet routing
- real Key Vault secret values or third-party API keys
- a production health endpoint URL
- a finished ACR quarantine and release-promotion process

## Permission quick read

If your access is mostly limited to editing code in this repo, the items above that you usually cannot supply yourself are the Azure federated identity, the private runner, the GitHub environment secrets and any runner-label variable overrides, the real Key Vault secret values, and any private DNS or networking prerequisites.

If you do have Azure resource-group deployment rights but not broader platform permissions, the most common remaining blockers are the federated identity setup, runner registration, secret ownership, and private network or DNS administration.

## GitHub environment variables

Configure these as environment or repository variables, not committed values:

- `AZURE_PRIVATE_RUNNER_LABELS_JSON` (optional JSON array overriding the default `["self-hosted","linux","azure-private"]` runner labels)

Treat variables as low-sensitivity workflow metadata only. If a value reveals
tenant, subscription, registry, resource-group, or private endpoint details,
prefer a secret instead. When you do use runner labels here, keep them generic.

## GitHub environment secrets

Configure these as environment secrets rather than variables:

- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`
- `AZURE_RESOURCE_GROUP`
- `ACR_NAME`
- `IMAGE_REPOSITORY` (optional; defaults to `paper-chaser-mcp`)
- `SMOKE_TEST_HEALTH_URL` (optional on the first full deployment, recommended afterward)

These values are not all credentials, but they are still deployment-specific
identifiers. Using secrets for them keeps the workflow from treating your
subscription, tenant, registry, and resource-group names as ordinary public
metadata. `SMOKE_TEST_HEALTH_URL` remains optional on the first rollout because
the workflow can derive a fallback from Azure deployment outputs for that run.

## Configuration placement guide

Use these buckets when you add new knobs so the scaffold gets more flexible
without turning git or Actions metadata into a secret store.

| Put it in | Good candidates | Keep out |
| --- | --- | --- |
| Git-tracked Bicep params and `.bicepparam` files | safe defaults, SKUs, replica counts, API paths, publisher metadata, tags | API keys, backend tokens, tenant-specific private hostnames |
| GitHub variables | low-sensitivity workflow metadata such as generic runner label overrides or model-selection flags | deployment identifiers you would not want treated as ordinary public metadata |
| GitHub secrets | deployment identifiers you want masked in workflow logs, plus small workflow-time sensitive values such as a private smoke-test URL | long-lived runtime provider credentials when Key Vault is available |
| Azure Key Vault | runtime API keys, backend shared tokens, provider contact values that should stay server-side | non-secret build metadata and default shape/config values |
| Local `.env` or `docker compose --env-file` | developer-only overrides for local shells, MCP Inspector, and local Compose runs | committed credentials |

If a setting is secret and the running application needs it, prefer Key Vault.
If it is just metadata or a safe default, prefer a Bicep parameter or GitHub
variable. The safest default is to keep git-tracked configuration about shape
and behavior, and keep secrets in systems built for secrets.

## Smart workflow deployment notes

The Azure scaffold keeps the smart layer off by default and embeddings disabled
by default. Turn on the smart layer only when you intend to pay for server-side
planning and grounded synthesis. Opt in to embeddings only when you need
embedding-based similarity ranking and have budgeted for the additional API
calls.

- Set `enableAgentic=true` to expose the additive smart tools.
- Leave `agenticProvider=deterministic` if you want the smart-tool surface
  without paid model calls.
- Use `agenticProvider=openai` and seed the `openai-api-key` Key Vault secret
  when you want the LangChain-backed OpenAI path.
- Set `disableEmbeddings=false` to opt in to embedding-based ranking, frontier
  scoring, and workspace indexing. Without this, smart workflows use lexical
  similarity as the default scoring path.
- Use `agenticOpenAiTimeoutSeconds` to cap OpenAI-backed planner, synthesis,
  and embedding requests so smart calls degrade instead of hanging on the
  request critical path.
- Keep `agenticIndexBackend=memory` unless your runtime image includes the
  optional FAISS extra and you have validated that path in your environment.
  The current smart workflows cap saved result sets at relatively small sizes,
  so FAISS is a scale-up option rather than the recommended default.

## Deployment flow

1. Run the validation stack.
2. For first-time environments, run `bootstrap` mode from a GitHub-hosted
   runner to create the private Azure foundation.
3. Seed Key Vault with the expected secret names and values.
4. Run `full` mode from the private self-hosted runner.
5. Build the container image on that runner.
6. Block promotion if Trivy finds high or critical vulnerabilities.
7. Push the image to private ACR.
8. Run a private-safe `az deployment group what-if` gate against `infra/main.bicep`.
9. Apply the deployment.
10. Smoke test `/healthz` through the private health URL or the deployment-output fallback.

## Validation matrix

Use these validation layers before you deploy anything real.

### Default local CI-equivalent gate

- `python -m pip check`
- `pre-commit run --all-files`
- `python -m pytest --cov=paper_chaser_mcp --cov-report=term-missing --cov-fail-under=85`
- `python -m mypy --config-file pyproject.toml`
- `python -m ruff check .`
- `python -m bandit -c pyproject.toml -r paper_chaser_mcp`
- `python -m build`
- `python -m pip_audit . --progress-spinner off`

If you prefer to trigger the heavier hook-managed checks through pre-commit,
`pre-commit run --hook-stage manual --all-files` runs the manual-stage
`pip check`, coverage, build, and `pip-audit` hooks.

### Infrastructure-specific validation

Run these when you touch Azure IaC, deployment docs, the Dockerfile, the APIM
policy, or the deployment workflow:

```text
python scripts/validate_psrule_azure.py
python scripts/validate_deployment.py --require-az --skip-docker
```

Use this command when you want parity with the `Deploy Azure` workflow's full
deployment validation path:

```text
python scripts/validate_deployment.py --require-az --require-docker --image-tag paper-chaser-mcp:ci-validate
```

`scripts/validate_deployment.py` currently validates:

- `server.json` public MCP metadata exists, uses the GitHub namespace form,
  declares a stdio OCI package, and keeps the GHCR identifier aligned with the
  repo URL and version
- APIM policy XML is well-formed
- `az bicep lint` on `infra/main.bicep`
- `az bicep build` on `infra/main.bicep`
- `az bicep build-params` for each committed `.bicepparam`
- `docker build` of the runtime image
- Docker OCI labels stay aligned with `server.json` for server name, version,
  and source URL
- container smoke test for `/healthz`
- blocked-origin `/mcp/` request returns `403`
- missing configured backend auth header on `/mcp/` returns `401`
- allowed-origin `/mcp/` request with the configured backend auth header is allowed past the deployment wrapper

The validator currently models the checked-in Azure scaffold, so its Docker
smoke test sets `PAPER_CHASER_HTTP_AUTH_HEADER=x-backend-auth` and expects
`X-Backend-Auth` specifically.

### Pre-deploy Azure checks

- run `az deployment group what-if` locally or in a private environment when you need the full diff
- use `deployMode=bootstrap` for the first infrastructure bring-up, then `deployMode=full` after Key Vault is seeded
- review the diff before apply
- confirm Key Vault contains the expected secret names
- confirm the private runner resolves the ACR, Key Vault, Container Apps, and APIM private DNS zones correctly
- remember that the bootstrap workflow path runs
  `scripts/validate_deployment.py --require-az --skip-docker`, while the full
  workflow reruns the validator with Docker enabled and the fixed validation
  image tag `paper-chaser-mcp:ci-validate`
- keep the GitHub Actions workflow on suppressed CLI output so successful
  runs do not publish Azure resource identifiers

### Post-deploy smoke checks

- store the health endpoint in the `SMOKE_TEST_HEALTH_URL` GitHub environment secret once the endpoint is known
- call the health endpoint
- verify the private APIM route requires client credentials
- verify the backend Container App URL cannot be used without the backend-only token
- run a minimal MCP initialize request through APIM from inside the private network

## Client protection strategy

The scaffold separates credentials by trust boundary.

### Between client and APIM

- Use APIM subscription keys for per-client access control.
- Add JWT validation if you want tenant-aware identity and revocation.

### Between APIM and the Container App

- APIM injects the `X-Backend-Auth` header in the checked-in scaffold.
- The Container App validates it before serving `/mcp`.
- The shared token stays in Key Vault and is referenced by both the Container App and APIM through managed identities.

The wrapper itself supports a configurable backend header via
`PAPER_CHASER_HTTP_AUTH_HEADER` and defaults to `authorization` when you do
not override it. The Azure scaffold deliberately sets that header name to
`x-backend-auth` so APIM can keep backend credentials separate from
client-facing authorization.

### Between the Container App and upstream providers

- Upstream provider keys stay in Key Vault and are only accessible to the Container App identity.
- Clients never receive those keys and cannot reuse them.

## Recommended next hardening steps

1. Enable JWT validation in APIM for named client applications.
2. Add APIM diagnostics and Application Insights alerts.
3. Rotate Key Vault secrets on a schedule and roll Container App revisions after rotation.
4. Add a formal image-promotion workflow if you want Azure Container Registry quarantine to gate release promotion as well as registry policy.
