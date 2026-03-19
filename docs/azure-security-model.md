# Azure Security Model

This document focuses on how the Azure scaffold protects the MCP service and the credentials around it.

## Threat boundaries

There are three different credential classes in this system.

### 1. Client credentials

These are how internal users or applications access the private MCP endpoint.

- Preferred control plane: Azure API Management private endpoint
- Recommended mechanisms:
  - APIM subscription keys
  - Microsoft Entra JWT validation for higher-assurance callers

These credentials are issued to clients. They are revocable, rotatable, and should be unique per client or integration.

### 2. Backend gateway credential

This is the credential APIM uses when it forwards traffic to the actual Container App.

- Header name in the Azure scaffold: `X-Backend-Auth`
- Secret source: Azure Key Vault
- Runtime use: the Container App deployment wrapper validates the secret
  before forwarding traffic to FastMCP

The deployment wrapper itself supports a configurable header name via
`SCHOLAR_SEARCH_HTTP_AUTH_HEADER` and defaults to `authorization` when you do
not override it. The Azure scaffold intentionally sets that value to
`x-backend-auth` so backend gateway credentials stay separate from
client-facing authorization headers.

This credential is **not** for clients. It exists so a discovered Container App URL is not enough to use the backend directly, even inside the private network.

### 3. Upstream provider API keys

These are the credentials the MCP server uses to talk to scholarly providers.

- `CORE_API_KEY`
- `SEMANTIC_SCHOLAR_API_KEY`
- `OPENALEX_API_KEY`
- `SERPAPI_API_KEY`
- optional `OPENALEX_MAILTO` contact value

These are server-side secrets only.

## Storage and access rules

### Key Vault

All server-side secrets belong in Azure Key Vault.

- Do not commit them.
- Do not store them in plain YAML.
- Do not bake them into Docker images.

### Managed identity

The Container App uses a managed identity so it can read the Key Vault secret references at runtime.

API Management uses its system-assigned managed identity so it can resolve the backend auth token from Key Vault without copying that secret into source control or workflow YAML. This keeps the Key Vault firewall enabled while still allowing the APIM named value to refresh from Key Vault.

Required access in this scaffold:

- `AcrPull` on the container registry
- `Key Vault Secrets User` on the Key Vault

## Why client API keys do not leak upstream API keys

Client credentials only grant access to your APIM gateway.

They do **not** grant direct access to:

- Key Vault
- the backend-only auth token
- provider API keys
- Azure resource management APIs

This separation matters because clients may be untrusted or semi-trusted consumers of the MCP server. They should be able to invoke the server without ever learning how the server authenticates to third-party providers.

## Public repository controls

Because this is a public repo, the scaffold keeps the repository safe by design.

- Bicep parameters contain only placeholders and environment-neutral values.
- Workflow files read deployment identifiers from GitHub environment secrets.
- Nonpublic smoke-test URLs are expected to come from GitHub environment secrets.
- The deployment workflow is manual-only and splits bootstrap from full deploys.
- Bootstrap can run on a GitHub-hosted runner because it only uses Azure
  control-plane APIs.
- The deployment workflow suppresses successful Azure CLI deployment output
  so resource identifiers are not echoed into public Actions logs.
- Secret values are injected at runtime from Azure, not stored in git.
- Documentation uses placeholder emails, names, and URLs.
- Only the full deployment job is expected to run from a private self-hosted
  runner, because it needs private DNS reachability for ACR push and smoke
  testing.

## Suggested operational practices

1. Issue a distinct APIM subscription key per customer or integration.
2. Rate-limit per subscription.
3. Rotate `mcp-backend-auth-token` and upstream provider secrets periodically.
4. Add APIM JWT validation for enterprise tenants that need stronger caller identity.
5. Keep the APIM, Container Apps, Key Vault, and ACR private DNS zones resolvable from every trusted network that needs deployment or runtime access.
