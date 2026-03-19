# Security Policy

## Supported scope

Please report vulnerabilities in:

- the published Python package
- the MCP server runtime and HTTP transport wrappers
- the checked-in GitHub Actions workflows
- the Azure deployment scaffold under `infra/`

## Public repo posture

This repository is intentionally public, but the intended deployment model keeps
secrets and private infrastructure details out of source control:

- runtime credentials belong in Azure Key Vault, not in git
- the Azure deployment workflow is manual-only
- first-time rollouts should use `bootstrap` mode before `full`
- only the full deployment path should use a private self-hosted runner
- HTTP transport is for local, controlled, or private-gateway deployments, not
  direct public internet exposure

## Recommended repo settings

For a public repository, keep these controls enabled outside the codebase as
well:

- protect the default branch and require the main validation workflow before
  merge
- keep default GitHub Actions token permissions at read-only and grant elevated
  permissions per workflow or per job only where needed
- enable GitHub secret scanning and push protection
- keep deployment identifiers in environment secrets, not ordinary workflow
  variables
- keep runtime credentials in Azure Key Vault, not GitHub
- prune unused repository-level secrets and prefer environment scoping when a
  workflow does need them
- keep self-hosted runner labels generic so they do not reveal private network
  or platform details

## Additional hardening opportunities

These are worthwhile next steps, but they require out-of-band platform changes
and are not fully enforceable from the repo alone:

- if you later reintroduce package publishing, prefer short-lived or
  registry-native publishing over long-lived static tokens
- if you later publish to PyPI, prefer trusted publishing over an API token
- use a restricted runner group instead of a repo-scoped self-hosted runner
- add environment reviewers for higher-risk deployment environments

## Reporting a vulnerability

Please do not open a public issue for suspected vulnerabilities.

Preferred path:

1. Use GitHub private vulnerability reporting or a GitHub security advisory for
   this repository if it is enabled.
2. If that is not available, contact the maintainer through a private channel
   before public disclosure.

When you report an issue, include:

- affected component or file
- impact and attack preconditions
- clear reproduction steps or proof of concept
- any suggested remediation or compensating controls

## What to avoid in reports

Do not include live credentials, production-only hostnames, private IPs, or
other sensitive environment details in public discussion or logs.
