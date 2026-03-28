# Release And Publishing Plan

This document is the durable release playbook for the current `paper-chaser-mcp`
publishing model. It reflects the repo state after GHCR, MCP Registry, GitHub
Release assets, and dormant PyPI publication were split into separate workflows.

## Goals

The release process should:

1. Keep versioned releases immutable once tagged.
2. Make GHCR the primary public distribution channel for the containerized MCP package.
3. Publish wheel and sdist artifacts to the GitHub Release page for Python users who cannot or should not rely on PyPI yet.
4. Keep MCP Registry publication explicit and reviewable instead of coupling it to GHCR success.
5. Leave PyPI dormant until account recovery and trusted-publisher setup are complete.

## Current Publishing Surfaces

| Surface | Workflow | Trigger | Current posture |
| --- | --- | --- | --- |
| GHCR image | `.github/workflows/publish-public-mcp-package.yml` | `v*` tag or manual dispatch | Primary public container distribution |
| GitHub Release assets | `.github/workflows/publish-github-release.yml` | `v*` tag or manual dispatch | Builds wheel/sdist, verifies with `twine check`, uploads to a draft Release |
| MCP Registry listing | `.github/workflows/publish-mcp-registry.yml` | Manual dispatch only | Run after the GHCR image is confirmed |
| PyPI / TestPyPI | `.github/workflows/publish-pypi.yml` | PR build, manual dispatch, `v*` tag | Build path is active; publish path remains gated behind `ENABLE_PYPI_PUBLISHING == 'true'` |

## Versioning Policy

1. Use stable semver-like PEP 440 versions in `pyproject.toml` and `server.json`, for example `0.2.0`.
2. Use `v*` git tags to trigger publish workflows, for example `v0.2.0`.
3. Do not move or overwrite release tags after publication.
4. If a tagged release needs correction, cut a new version such as `0.2.1` rather than rewriting `0.2.0`.
5. Do not introduce prerelease tags (`-rc.1`, `-beta.1`, etc.) until the version contract, tests, and workflows are explicitly updated to support them end-to-end.

## Release Phases

### Phase 1: Release-Prep PR

Create a dedicated release-prep branch from `master` and keep the diff small.

Expected contents:

1. Bump `pyproject.toml` `project.version`.
2. Bump `pyproject.toml` `tool.bumpver.current_version`.
3. Bump `server.json` `version`.
4. Bump `server.json` GHCR package identifier to the same version.
5. Optionally decide whether the maturity classifier should remain Alpha or move to Beta.

Suggested command:

```bash
bumpver update --minor --no-commit --no-tag-commit --no-push
```

Run focused validation immediately, then the normal release-prep validation stack.

### Phase 2: Merge The Release-Prep PR

Do not tag from an unmerged branch. Merge the version bump first so all release
workflows operate on the committed `master` contract.

### Phase 3: Tag The Release

From up-to-date `master`:

```bash
git switch master
git pull --ff-only origin master
git tag v0.2.0
git push origin v0.2.0
```

That tag should trigger:

1. GHCR publication through `.github/workflows/publish-public-mcp-package.yml`.
2. GitHub Release asset publication through `.github/workflows/publish-github-release.yml`.

It should not automatically trigger MCP Registry publication.

### Phase 4: Inspect Before Broad Promotion

After the tag workflows finish, inspect the outputs before treating the release
as broadly public.

Review checklist:

1. GHCR image exists at `ghcr.io/joshuasundance-swca/paper-chaser-mcp:0.2.0`.
2. GHCR tags look correct: exact version, major/minor tags, and `latest` when appropriate.
3. OCI labels and MCP metadata match `server.json`.
4. The draft GitHub Release contains the wheel, sdist, and `SHA256SUMS`.
5. The wheel and sdist names match the version contract.
6. Release notes and presentation are acceptable for external users.
7. The package visibility state on GHCR is what you intend.

### Phase 5: Promote Discoverability

If the release looks good:

1. Make the GHCR package public if it is still private.
2. Publish the draft GitHub Release.
3. Run `.github/workflows/publish-mcp-registry.yml` manually for the same tag.

MCP Registry publication should happen only after the GHCR image and GitHub
Release assets are confirmed.

### Phase 6: PyPI Later

PyPI remains a separate concern.

Do not enable it until all of the following are true:

1. TestPyPI and PyPI account access is recovered.
2. Trusted publishers are registered for the final repo/workflow/environment identity.
3. The repository variable `ENABLE_PYPI_PUBLISHING` is intentionally set to `true`.

Until then, source installs and GitHub Release assets are the Python-user path.

## Release-Day Validation Checklist

Before tagging, run at minimum:

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

If the release touches Docker or deployment-facing files, also run:

```bash
python scripts/validate_psrule_azure.py
python scripts/validate_deployment.py --skip-docker
```

## Local Workflow-Lint Parity

Workflow-file release changes are only trustworthy when local validation has CI
parity.

Rules:

1. Keep `shellcheck` available on `PATH` before running `pre-commit`.
2. Use redirected logs plus explicit completion checks for long pre-commit runs on Windows.
3. Do not treat partial integrated-terminal output as proof that validation is complete.

Recommended Windows pattern:

```bash
pre-commit run --files <changed-files> > precommit-release.log 2>&1
```

Then wait for completion, read the log, and only then commit or push.

## Recommended Sequence For 0.2.0

Assuming the workflow and documentation groundwork is already merged:

1. Create and merge a small `0.2.0` release-prep PR.
2. Tag `v0.2.0` from `master`.
3. Inspect GHCR and the draft GitHub Release.
4. Make GHCR public and publish the GitHub Release if the artifacts look correct.
5. Run manual MCP Registry publication.
6. Leave PyPI gated.

## If Something Goes Wrong

If a release artifact or image is wrong after the tag is pushed:

1. Do not move the tag.
2. Fix the problem on `master`.
3. Bump to the next version, for example `0.2.1`.
4. Retag with the new version.

This keeps the published record immutable and avoids ambiguity across GHCR,
GitHub Release assets, and MCP Registry metadata.
