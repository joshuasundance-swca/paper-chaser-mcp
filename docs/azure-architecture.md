# Azure Architecture

This repository includes a private-network Azure hosting scaffold for the Paper Chaser MCP server.

## Recommended topology

```text
Private MCP client
  - corporate network, VPN, or peered VNet
   |
   v
Azure API Management
  - client auth and subscription keys
  - throttling and gateway logs
  - private endpoint ingress only
  - outbound VNet integration to private backends
  - injects backend-only auth header
   |
   v
Azure Container Apps
  - paper-chaser-mcp deployment-http -> paper_chaser_mcp.deployment:app
  - /healthz for probes
  - /mcp for streamable HTTP MCP traffic
  - optional server-side smart layer for concept-level discovery, grounded QA,
    and searchSessionId workspaces
  - managed environment public network disabled; private endpoint on the environment
  - backend token still required at /mcp
   |
   +--> Managed identity --> Azure Key Vault private endpoint
   |
   +--> Managed identity --> Azure Container Registry private endpoint
   |
   +--> Application Insights / Log Analytics
   |
   +--> External scholarly APIs

GitHub Actions bootstrap runner
  - GitHub-hosted
  - Azure control-plane only
  - creates private networking and shared infrastructure

GitHub Actions private runner
  - Docker build
  - private DNS resolution
  - push to private ACR
  - private smoke tests
```

## Protection model

### Upstream provider keys

The server can use keys for CORE, Semantic Scholar, OpenAlex, SerpApi, and the
optional OpenAI, Azure OpenAI, Anthropic, NVIDIA, Google, or Mistral-backed smart layer. Those
keys are **server-side only**.

- They are stored in Azure Key Vault.
- The Container App reads them through Key Vault references.
- The Container App authenticates to Key Vault with a managed identity.
- Clients calling the MCP server never receive these upstream provider keys.

CORE is **disabled by default** (`enableCore=false` in the Bicep scaffold,
matching the application default). When CORE is disabled, the `core-api-key`
Key Vault secret is not mounted and does not need to exist. Set `enableCore=true`
in the relevant `.bicepparam` file and seed `core-api-key` only when you want
the CORE fallback hop.

When the smart layer is enabled, the Container App reads the provider-specific
credential from Key Vault and keeps all LangChain/LangGraph planning and
synthesis calls server-side. NVIDIA is currently integrated as a chat-only
provider in this repo; embeddings remain a later cross-provider overhaul. For `agenticProvider=azure-openai`,
the Container App also receives `AZURE_OPENAI_ENDPOINT` and, when configured,
`AZURE_OPENAI_API_VERSION` from non-secret Bicep params.

### Client access credentials

Clients should authenticate to Azure API Management, not to the Container App directly.

- APIM is reachable through a private endpoint in this scaffold.
- APIM subscription keys are the default client credential model.
- For stronger identity requirements, add Microsoft Entra JWT validation in APIM policy.
- APIM can rate-limit by subscription or caller identity.

### Backend protection

The Container App also enforces a separate backend-only shared token.

- The deployment wrapper reads `PAPER_CHASER_HTTP_AUTH_TOKEN` from a Key Vault-backed Container App secret.
- The checked-in Azure scaffold sets `PAPER_CHASER_HTTP_AUTH_HEADER=x-backend-auth`.
- APIM injects that value into the `X-Backend-Auth` header.
- Direct requests to the Container App fail without that token, even if the app URL is discovered.

Outside the Azure scaffold, the wrapper still defaults to
`Authorization: Bearer <token>` when `PAPER_CHASER_HTTP_AUTH_HEADER` is left
at `authorization`.

This means client credentials and backend credentials are intentionally different:

1. Client credentials prove who is allowed to use the private MCP API.
2. Backend credentials prove the request passed through the gateway you control.

## Public repo hygiene

This repository does not commit:

- real provider API keys
- real backend auth tokens
- real tenant-specific secrets
- private endpoint hostnames
- production endpoints or other internal-only hostnames

Only templates, placeholders, and workflow structure belong in source control.

The first rollout should use bootstrap mode before the full deployment mode so
the private network and secret stores exist before the workload is introduced.
